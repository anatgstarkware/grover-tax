//! Reversible-gate-circuit simulator AIR for the grover-tax iadd256 benchmark.
//!
//! Proves gate-by-gate execution of a {NOP,NOT,CNOT,TOFFOLI} circuit over a
//! 512-qubit state encoded as 32 limbs of 16 bits. One trace row per gate; the
//! pc sweeps the gate list K times (K = fixture repetitions) and the state is
//! threaded continuously via a telescoping state lookup. Boundary states (x at
//! pc 0, y at pc K*n_gates) are public.
//!
//! This binary is self-contained so the existing `native-iadd-air` binary
//! (src/main.rs) keeps building unchanged.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use itertools::Itertools;
use num_traits::{One, Zero};
use serde::Deserialize;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    assert_constraints_on_trace, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator,
    Relation, RelationEntry, TraceLocationAllocator,
};

// ----------------------------------------------------------------------------
// Encoding constants
// ----------------------------------------------------------------------------

const N_QUBITS: usize = 512;
const LIMB_BITS: usize = 16;
const N_LIMBS: usize = N_QUBITS / LIMB_BITS; // 32
const STATE_BYTES: usize = N_QUBITS / 8; // 64

// State-relation width: shot_id, pc, then 32 limbs.
const STATE_WIDTH: usize = 2 + N_LIMBS;

// Dynamic range-check tables span pos in 0..LIMB_BITS, value in 0..2^16.
// Pad both T_lo and T_hi to 2^17 rows so (pos, value) fits one preprocessed
// pair of columns (pos in 0..16 -> 4 bits, value 16 bits => up to 2^16 entries
// per table; we lay them out as a flat table of valid (pos,value) pairs).
const RC_LOG_SIZE: u32 = 16; // 2^16 padded rows for each dynamic-RC table.

const NO_CTRL: u16 = 0xFFFF;

const M31_MODULUS_U32: u32 = (1 << 31) - 1;
const LANE_COUNT: usize = 1 << LOG_N_LANES;

// ----------------------------------------------------------------------------
// Relations
// ----------------------------------------------------------------------------

stwo_constraint_framework::relation!(StateElements, STATE_WIDTH);
// q-decode membership table: (q, limb_idx, bit_pos, mask).
stwo_constraint_framework::relation!(QDecodeElements, 4);
// Dynamic range-check tables: (pos, value).
stwo_constraint_framework::relation!(RangeLoElements, 2);
stwo_constraint_framework::relation!(RangeHiElements, 2);
// Program-consistency table: (slot/pc_in_prog, opcode_scalar, target, ctrl_a, ctrl_b).
stwo_constraint_framework::relation!(ProgramElements, 5);

#[derive(Clone)]
struct LookupElements {
    state: StateElements,
    qdecode: QDecodeElements,
    rc_lo: RangeLoElements,
    rc_hi: RangeHiElements,
    program: ProgramElements,
}

impl LookupElements {
    fn draw(channel: &mut impl Channel) -> Self {
        Self {
            state: StateElements::draw(channel),
            qdecode: QDecodeElements::draw(channel),
            rc_lo: RangeLoElements::draw(channel),
            rc_hi: RangeHiElements::draw(channel),
            program: ProgramElements::draw(channel),
        }
    }
}

// ----------------------------------------------------------------------------
// CLI / fixture
// ----------------------------------------------------------------------------

#[derive(Parser, Debug)]
struct Args {
    /// Grover-tax v0.3-iadd fixture JSON.
    #[arg(long, default_value = "../../../../fixtures/v0.3-iadd256-k4-n16.json")]
    fixture: PathBuf,

    /// Number of fixture samples (shots) to include. Defaults to 1.
    #[arg(long, default_value_t = 1)]
    samples: usize,

    /// Override repetitions K. Defaults to fixture repetitions.
    #[arg(long)]
    repetitions: Option<usize>,

    /// Generate + self-check the witness but skip STWO proof generation.
    #[arg(long)]
    no_prove: bool,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    version: String,
    repetitions: usize,
    n_samples: usize,
    num_qubits: usize,
    circuit_byte_serialisation_hex: String,
    test_cases: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    x_hex: String,
    y_hex: String,
}

// ----------------------------------------------------------------------------
// Circuit parser (GTV1)
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Gate {
    opcode: u8,
    target: u16,
    ctrl_a: u16,
    ctrl_b: u16,
}

const OP_NOP: u8 = 0;
const OP_NOT: u8 = 1;
const OP_CNOT: u8 = 2;
const OP_TOFFOLI: u8 = 3;

fn parse_gtv1(hex_str: &str) -> Result<Vec<Gate>> {
    let bytes = hex::decode(hex_str).context("decoding circuit hex")?;
    if bytes.len() < 8 || &bytes[0..4] != b"GTV1" {
        bail!("bad GTV1 magic");
    }
    let n_gates = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + 8 * n_gates;
    if bytes.len() != expected {
        bail!(
            "GTV1 length mismatch: got {} expected {}",
            bytes.len(),
            expected
        );
    }
    let mut gates = Vec::with_capacity(n_gates);
    let mut off = 8;
    for _ in 0..n_gates {
        let opcode = bytes[off];
        // bytes[off+1] is pad.
        let target = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]);
        let ctrl_a = u16::from_le_bytes([bytes[off + 4], bytes[off + 5]]);
        let ctrl_b = u16::from_le_bytes([bytes[off + 6], bytes[off + 7]]);
        gates.push(Gate {
            opcode,
            target,
            ctrl_a,
            ctrl_b,
        });
        off += 8;
    }
    Ok(gates)
}

// ----------------------------------------------------------------------------
// State helpers (16-bit limbs)
// ----------------------------------------------------------------------------

fn state_to_limbs(bytes: &[u8]) -> [u32; N_LIMBS] {
    debug_assert_eq!(bytes.len(), STATE_BYTES);
    let mut limbs = [0u32; N_LIMBS];
    for (j, limb) in limbs.iter_mut().enumerate() {
        let lo = bytes[2 * j] as u32;
        let hi = bytes[2 * j + 1] as u32;
        *limb = lo | (hi << 8);
    }
    limbs
}

fn limbs_to_state(limbs: &[u32; N_LIMBS]) -> [u8; STATE_BYTES] {
    let mut out = [0u8; STATE_BYTES];
    for (j, &limb) in limbs.iter().enumerate() {
        out[2 * j] = (limb & 0xFF) as u8;
        out[2 * j + 1] = ((limb >> 8) & 0xFF) as u8;
    }
    out
}

#[inline]
fn qubit_decode(q: u16) -> (u32, u32, u32) {
    let limb_idx = (q as u32) / LIMB_BITS as u32;
    let bit_pos = (q as u32) % LIMB_BITS as u32;
    let mask = 1u32 << bit_pos;
    (limb_idx, bit_pos, mask)
}

// ----------------------------------------------------------------------------
// Witness row
// ----------------------------------------------------------------------------

/// Decoded read of a single qubit (target / ctrl_a / ctrl_b).
#[derive(Clone, Copy)]
struct ReadCols {
    active: u32,       // enabler*active flag (0 if this control isn't live)
    q: u32,            // qubit index (0 when inactive)
    limb_idx: u32,     // decoded limb index
    bit_pos: u32,      // decoded bit position
    mask: u32,         // 2^bit_pos
    lsel: [u32; N_LIMBS], // limb-select one-hot (all zero if inactive)
    lo: u32,
    hi: u32,
    bit: u32,
}

impl ReadCols {
    fn inactive() -> Self {
        Self {
            active: 0,
            q: 0,
            limb_idx: 0,
            bit_pos: 0,
            mask: 1, // 2^0; harmless since lsel is all zero so L=0.
            lsel: [0; N_LIMBS],
            lo: 0,
            hi: 0,
            bit: 0,
        }
    }

    fn live(limbs: &[u32; N_LIMBS], q: u16) -> Self {
        let (limb_idx, bit_pos, mask) = qubit_decode(q);
        let l = limbs[limb_idx as usize];
        let lo = l & ((1u32 << bit_pos) - 1);
        let bit = (l >> bit_pos) & 1;
        let hi = l >> (bit_pos + 1);
        let mut lsel = [0u32; N_LIMBS];
        lsel[limb_idx as usize] = 1;
        Self {
            active: 1,
            q: q as u32,
            limb_idx,
            bit_pos,
            mask,
            lsel,
            lo,
            hi,
            bit,
        }
    }
}

#[derive(Clone)]
struct Row {
    enabler: u32,
    is_nop: u32,
    is_not: u32,
    is_cnot: u32,
    is_toffoli: u32,
    shot_id: u32,
    pc: u32,
    in_limb: [u32; N_LIMBS],
    out_limb: [u32; N_LIMBS],
    target: ReadCols,
    ctrl_a: ReadCols,
    ctrl_b: ReadCols,
    ab: u32,
    fire: u32,
    delta: u32, // new_t - t_bit, signed in {-1,0,1}; stored as M31.
}

impl Row {
    fn padding() -> Self {
        Self {
            enabler: 0,
            is_nop: 0,
            is_not: 0,
            is_cnot: 0,
            is_toffoli: 0,
            shot_id: 0,
            pc: 0,
            in_limb: [0; N_LIMBS],
            out_limb: [0; N_LIMBS],
            target: ReadCols::inactive(),
            ctrl_a: ReadCols::inactive(),
            ctrl_b: ReadCols::inactive(),
            ab: 0,
            fire: 0,
            delta: 0,
        }
    }
}

// ----------------------------------------------------------------------------
// Column layout
// ----------------------------------------------------------------------------
//
// Per row:
//   enabler                         (1)
//   is_nop,is_not,is_cnot,is_toffoli(4)
//   shot_id, pc                     (2)
//   in_limb[0..32]                  (32)
//   out_limb[0..32]                 (32)
//   target read block               (READ_COLS)
//   ctrl_a read block               (READ_COLS)
//   ctrl_b read block               (READ_COLS)
//   ab, fire, delta                 (3)
//
// READ_COLS = q(1)+limb_idx(1)+bit_pos(1)+mask(1)+lsel(32)+lo(1)+hi(1)+bit(1)
const READ_COLS: usize = 4 + N_LIMBS + 3; // 39
const TRACE_COLUMNS: usize = 1 + 4 + 2 + N_LIMBS + N_LIMBS + 3 * READ_COLS + 3;

fn delta_to_m31(delta: i64) -> u32 {
    // new_t - t_bit in {-1,0,1}; represent in M31.
    if delta >= 0 {
        delta as u32
    } else {
        (M31_MODULUS_U32 as i64 + delta) as u32
    }
}

// ----------------------------------------------------------------------------
// Witness generation + self-check
// ----------------------------------------------------------------------------

struct LookupCounts {
    qdecode: Vec<u32>,
    rc_lo: Vec<u32>,
    rc_hi: Vec<u32>,
}

impl LookupCounts {
    fn new() -> Self {
        Self {
            qdecode: vec![0; N_QUBITS],
            rc_lo: vec![0; 1 << RC_LOG_SIZE],
            rc_hi: vec![0; 1 << RC_LOG_SIZE],
        }
    }

    /// Reduce: add another shot's local counts into this one (component-wise).
    fn add_assign(&mut self, other: &LookupCounts) {
        for (a, b) in self.qdecode.iter_mut().zip(&other.qdecode) {
            *a += *b;
        }
        for (a, b) in self.rc_lo.iter_mut().zip(&other.rc_lo) {
            *a += *b;
        }
        for (a, b) in self.rc_hi.iter_mut().zip(&other.rc_hi) {
            *a += *b;
        }
    }
}

/// Build all witness rows for the selected shots and assert each shot's final
/// state matches y_hex. Returns rows plus lookup multiplicity counts.
///
/// PARALLELISM (two-phase, trace bit-identical to the serial version):
///   Phase 1 here parallelizes over SHOTS. Shots are fully independent: shot s
///   owns the contiguous scalar row block `[s*K*n_gates, (s+1)*K*n_gates)`, has
///   its own initial state x_s and its own sequential chain to y_s (gates and K
///   reps are threaded strictly in order *within* a shot). Each shot writes a
///   disjoint `&mut [Row]` slice (`par_chunks_mut`) and accumulates its own LOCAL
///   `LookupCounts`; we then REDUCE (sum) the per-shot counts at the end.
///   Phase 2 — packing the scalar `Vec<Row>` into `PackedM31` words — happens
///   later in `generate_main_trace` / `gen_main_interaction` over PACKED rows,
///   so a shot block being non-16-aligned (2547*K rows) can never cause a
///   packed-word data race: no two threads ever touch the same packed word here
///   (they touch disjoint scalar `Row` cells), and packing reads the finished
///   `Vec<Row>` single-threaded-per-word afterwards.
fn build_rows(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
    rc_lo_index: &RcIndex,
    rc_hi_index: &RcIndex,
) -> Result<(Vec<Row>, LookupCounts)> {
    use rayon::prelude::*;

    let n_gates = gates.len();
    let shot_rows = k * n_gates;
    let total_rows = cases.len() * shot_rows;

    // Pre-allocate the full scalar row buffer; each shot fills a disjoint block.
    let mut rows = vec![Row::padding(); total_rows];

    // Phase 1: simulate every shot in parallel into its own disjoint row block,
    // each producing its own local LookupCounts. Returns Err on the first shot
    // whose simulation fails or whose final state mismatches y_hex.
    let per_shot: Vec<Result<LookupCounts>> = rows
        .par_chunks_mut(shot_rows)
        .zip(cases.par_iter())
        .enumerate()
        .map(|(shot_id, (block, case))| {
            simulate_shot(gates, k, shot_id, case, block, rc_lo_index, rc_hi_index)
        })
        .collect();

    // Phase 1 reduce: sum the per-shot local counts into one global LookupCounts.
    // Arrays are small (512, 2^16, 2^16); a serial reduce is cheap.
    let mut counts = LookupCounts::new();
    for result in per_shot {
        let local = result?;
        counts.add_assign(&local);
    }

    Ok((rows, counts))
}

/// Simulate a single shot sequentially, filling its row block and returning its
/// LOCAL lookup multiplicity counts. The chain (K reps * n_gates gates) is run
/// strictly in order, threading the 512-bit state from x_s to y_s, and the final
/// state is checked against y_hex.
fn simulate_shot(
    gates: &[Gate],
    k: usize,
    shot_id: usize,
    case: &TestCase,
    block: &mut [Row],
    rc_lo_index: &RcIndex,
    rc_hi_index: &RcIndex,
) -> Result<LookupCounts> {
    let n_gates = gates.len();
    let x = hex::decode(&case.x_hex).context("decoding x_hex")?;
    let y = hex::decode(&case.y_hex).context("decoding y_hex")?;
    if x.len() != STATE_BYTES || y.len() != STATE_BYTES {
        bail!("state must be {STATE_BYTES} bytes");
    }
    let mut counts = LookupCounts::new();
    let mut limbs = state_to_limbs(&x);
    let mut pc: u32 = 0;
    let mut row_idx = 0usize;

    for _rep in 0..k {
        for gate in gates {
            let in_limb = limbs;
            let (is_nop, is_not, is_cnot, is_toffoli) = match gate.opcode {
                OP_NOP => (1, 0, 0, 0),
                OP_NOT => (0, 1, 0, 0),
                OP_CNOT => (0, 0, 1, 0),
                OP_TOFFOLI => (0, 0, 0, 1),
                other => bail!("unknown opcode {other}"),
            };
            let a_active = is_cnot + is_toffoli;
            let b_active = is_toffoli;

            // Reads.
            let target = ReadCols::live(&in_limb, gate.target);
            let ctrl_a = if a_active == 1 {
                debug_assert_ne!(gate.ctrl_a, NO_CTRL);
                ReadCols::live(&in_limb, gate.ctrl_a)
            } else {
                ReadCols::inactive()
            };
            let ctrl_b = if b_active == 1 {
                debug_assert_ne!(gate.ctrl_b, NO_CTRL);
                ReadCols::live(&in_limb, gate.ctrl_b)
            } else {
                ReadCols::inactive()
            };

            let a_bit = ctrl_a.bit;
            let b_bit = ctrl_b.bit;
            let t_bit = target.bit;
            let ab = a_bit * b_bit;
            let fire = is_not + is_cnot * a_bit + is_toffoli * ab;
            debug_assert!(fire <= 1);
            let new_t = t_bit ^ fire;
            let delta_signed = new_t as i64 - t_bit as i64;

            // Apply write.
            let mut out_limb = in_limb;
            let tl = target.limb_idx as usize;
            if delta_signed > 0 {
                out_limb[tl] += target.mask;
            } else if delta_signed < 0 {
                out_limb[tl] -= target.mask;
            }
            limbs = out_limb;

            // Lookup multiplicities (local to this shot).
            count_read(&mut counts, &target, rc_lo_index, rc_hi_index);
            count_read(&mut counts, &ctrl_a, rc_lo_index, rc_hi_index);
            count_read(&mut counts, &ctrl_b, rc_lo_index, rc_hi_index);

            block[row_idx] = Row {
                enabler: 1,
                is_nop,
                is_not,
                is_cnot,
                is_toffoli,
                shot_id: shot_id as u32,
                pc,
                in_limb,
                out_limb,
                target,
                ctrl_a,
                ctrl_b,
                ab,
                fire,
                delta: delta_to_m31(delta_signed),
            };

            pc += 1;
            row_idx += 1;
        }
    }
    debug_assert_eq!(row_idx, k * n_gates);

    // Self-check: final state equals y.
    let final_bytes = limbs_to_state(&limbs);
    if final_bytes.as_slice() != y.as_slice() {
        bail!(
            "shot {shot_id}: simulated final state does not match y_hex\n  got {}\n  exp {}",
            hex::encode(final_bytes),
            case.y_hex
        );
    }

    Ok(counts)
}

fn count_read(counts: &mut LookupCounts, read: &ReadCols, lo_idx: &RcIndex, hi_idx: &RcIndex) {
    if read.active == 0 {
        return;
    }
    counts.qdecode[read.q as usize] += 1;
    counts.rc_lo[lo_idx.row(read.bit_pos, read.lo)] += 1;
    counts.rc_hi[hi_idx.row(read.bit_pos, read.hi)] += 1;
}

// ----------------------------------------------------------------------------
// Dynamic range-check tables (preprocessed)
// ----------------------------------------------------------------------------
//
// T_lo = {(pos, v): 0 <= v < 2^pos,        pos in 0..16}
// T_hi = {(pos, v): 0 <= v < 2^(15-pos),   pos in 0..16}
// Each has sum_pos 2^pos / 2^(15-pos) = 2^16 - 1 entries; padded to 2^16 rows.
// Padding rows reuse the (pos=0, v=0) tuple (a genuine table member) so the
// LogUp membership math stays sound: extra supply of an existing tuple is fine
// as long as the multiplicity-trace counts only real reads (which it does).

/// Maps (pos, value) -> a row index in the flattened preprocessed table, and
/// holds the (pos, value) contents of each row for trace generation.
struct RcIndex {
    pos_col: Vec<u32>,
    val_col: Vec<u32>,
    // offset[pos] = first row index for this pos block.
    offset: [usize; LIMB_BITS + 1],
}

impl RcIndex {
    /// `bound(pos)` returns the exclusive value bound for this pos.
    fn build(bound: impl Fn(usize) -> u32) -> Self {
        let size = 1usize << RC_LOG_SIZE;
        let mut pos_col = vec![0u32; size];
        let mut val_col = vec![0u32; size];
        let mut offset = [0usize; LIMB_BITS + 1];
        let mut row = 0usize;
        for pos in 0..LIMB_BITS {
            offset[pos] = row;
            let b = bound(pos);
            for v in 0..b {
                pos_col[row] = pos as u32;
                val_col[row] = v;
                row += 1;
            }
        }
        offset[LIMB_BITS] = row;
        debug_assert!(row <= size);
        // Remaining rows stay (pos=0, v=0): a valid member, inert padding.
        Self {
            pos_col,
            val_col,
            offset,
        }
    }

    #[inline]
    fn row(&self, pos: u32, value: u32) -> usize {
        self.offset[pos as usize] + value as usize
    }
}

fn build_rc_lo() -> RcIndex {
    RcIndex::build(|pos| 1u32 << pos)
}

fn build_rc_hi() -> RcIndex {
    RcIndex::build(|pos| 1u32 << (LIMB_BITS - 1 - pos))
}

// ----------------------------------------------------------------------------
// Program-consistency table (the single hidden program)
// ----------------------------------------------------------------------------
//
// One row per program slot (gate) i in 0..n_gates, padded to a power of two.
// Row i stores the canonical op tuple (opcode_scalar, target, ctrl_a, ctrl_b)
// as WITNESS (the hidden program) plus a multiplicity = number of executions of
// slot i = samples*K (every shot runs the program K times). Inactive controls
// canonicalise to 0, matching ReadCols::inactive().q == 0 on the use side. Slot
// index is a PREPROCESSED column (public row layout). Padding rows carry
// multiplicity 0, so their op contents are inert (never addressed: pc_in_prog
// stays in 0..n_gates).
struct ProgramTable {
    log_size: u32,
    slot: Vec<u32>,         // preprocessed slot index 0..size
    opcode_scalar: Vec<u32>, // witness
    target: Vec<u32>,        // witness
    ctrl_a: Vec<u32>,        // witness
    ctrl_b: Vec<u32>,        // witness
    multiplicity: Vec<u32>,  // witness: samples*K on real slots, 0 on padding
}

fn build_program_table(gates: &[Gate], samples: usize, k: usize) -> ProgramTable {
    let n_gates = gates.len();
    let padded = n_gates.next_power_of_two().max(LANE_COUNT);
    let log_size = padded.ilog2();
    let mut slot = vec![0u32; padded];
    let mut opcode_scalar = vec![0u32; padded];
    let mut target = vec![0u32; padded];
    let mut ctrl_a = vec![0u32; padded];
    let mut ctrl_b = vec![0u32; padded];
    let mut multiplicity = vec![0u32; padded];
    let mult = (samples * k) as u32;
    for (i, g) in gates.iter().enumerate() {
        slot[i] = i as u32;
        let (sc, a_active, b_active) = match g.opcode {
            OP_NOP => (0u32, false, false),
            OP_NOT => (1u32, false, false),
            OP_CNOT => (2u32, true, false),
            OP_TOFFOLI => (3u32, true, true),
            _ => (0u32, false, false),
        };
        opcode_scalar[i] = sc;
        target[i] = g.target as u32;
        ctrl_a[i] = if a_active { g.ctrl_a as u32 } else { 0 };
        ctrl_b[i] = if b_active { g.ctrl_b as u32 } else { 0 };
        multiplicity[i] = mult;
    }
    // Padding slots keep an in-range index sequence (inert; multiplicity 0).
    for (i, s) in slot.iter_mut().enumerate().skip(n_gates) {
        *s = i as u32;
    }
    ProgramTable {
        log_size,
        slot,
        opcode_scalar,
        target,
        ctrl_a,
        ctrl_b,
        multiplicity,
    }
}

// ----------------------------------------------------------------------------
// FrameworkEval
// ----------------------------------------------------------------------------

#[derive(Clone)]
struct GateEval {
    log_n_rows: u32,
    elements: LookupElements,
}

impl FrameworkEval for GateEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::one();

        // Preprocessed: pc_in_prog = (per-shot pc) mod n_gates. Public row layout
        // (the program slot each execution row addresses). Read first so the main
        // component's preprocessed tree carries exactly this one column.
        let pc_in_prog = eval.get_preprocessed_column(pp_id("gate_pc_in_prog"));

        let enabler = eval.next_trace_mask();
        let is_nop = eval.next_trace_mask();
        let is_not = eval.next_trace_mask();
        let is_cnot = eval.next_trace_mask();
        let is_toffoli = eval.next_trace_mask();
        let shot_id = eval.next_trace_mask();
        let pc = eval.next_trace_mask();

        let in_limb = (0..N_LIMBS).map(|_| eval.next_trace_mask()).collect_vec();
        let out_limb = (0..N_LIMBS).map(|_| eval.next_trace_mask()).collect_vec();

        let target = read_masks(&mut eval);
        let ctrl_a = read_masks(&mut eval);
        let ctrl_b = read_masks(&mut eval);

        let ab = eval.next_trace_mask();
        let fire = eval.next_trace_mask();
        let delta = eval.next_trace_mask();

        // --- Opcode booleanity + one-hot sum = enabler. ---
        for op in [&is_nop, &is_not, &is_cnot, &is_toffoli] {
            eval.add_constraint(op.clone() * (op.clone() - one.clone()));
        }
        eval.add_constraint(
            enabler.clone()
                - is_nop.clone()
                - is_not.clone()
                - is_cnot.clone()
                - is_toffoli.clone(),
        );

        let a_active = is_cnot.clone() + is_toffoli.clone();
        let b_active = is_toffoli.clone();

        // --- Per-read structural constraints. ---
        // target is always active iff enabler.
        read_constraints(&mut eval, &target, enabler.clone(), &in_limb);
        read_constraints(&mut eval, &ctrl_a, a_active.clone(), &in_limb);
        read_constraints(&mut eval, &ctrl_b, b_active.clone(), &in_limb);

        // --- Fire / delta logic. ---
        let a_bit = ctrl_a.bit.clone();
        let b_bit = ctrl_b.bit.clone();
        let t_bit = target.bit.clone();
        // ab = a_bit*b_bit.
        eval.add_constraint(ab.clone() - a_bit.clone() * b_bit.clone());
        // fire = is_not + is_cnot*a_bit + is_toffoli*ab.
        eval.add_constraint(
            fire.clone()
                - is_not.clone()
                - is_cnot.clone() * a_bit.clone()
                - is_toffoli.clone() * ab.clone(),
        );
        // new_t = t_bit + fire - 2*t_bit*fire (XOR); delta = new_t - t_bit.
        // delta = fire - 2*t_bit*fire = fire*(1 - 2*t_bit).
        eval.add_constraint(
            delta.clone() - fire.clone() + t_bit.clone() * fire.clone() * BaseField::from_u32_unchecked(2),
        );

        // --- Write: out_limb[j] = in_limb[j] + lsel_t[j]*delta*mask_t. ---
        for j in 0..N_LIMBS {
            eval.add_constraint(
                out_limb[j].clone()
                    - in_limb[j].clone()
                    - target.lsel[j].clone() * delta.clone() * target.mask.clone(),
            );
        }

        // --- State threading (telescoping). ---
        let mut input_state = Vec::with_capacity(STATE_WIDTH);
        input_state.push(shot_id.clone());
        input_state.push(pc.clone());
        input_state.extend(in_limb.iter().cloned());
        let mut output_state = Vec::with_capacity(STATE_WIDTH);
        output_state.push(shot_id.clone());
        output_state.push(pc.clone() + one.clone());
        output_state.extend(out_limb.iter().cloned());

        let mult = E::EF::from(enabler.clone());
        eval.add_to_relation(RelationEntry::new(
            &self.elements.state,
            mult.clone(),
            &input_state,
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.elements.state,
            -mult,
            &output_state,
        ));

        // --- q-decode membership for each active read. ---
        add_qdecode_lookup(&mut eval, &self.elements.qdecode, &target, enabler.clone());
        add_qdecode_lookup(&mut eval, &self.elements.qdecode, &ctrl_a, a_active.clone());
        add_qdecode_lookup(&mut eval, &self.elements.qdecode, &ctrl_b, b_active.clone());

        // --- Dynamic range checks (lo/hi) for each active read. ---
        add_rc_lookup(&mut eval, &self.elements.rc_lo, &self.elements.rc_hi, &target, enabler.clone());
        add_rc_lookup(&mut eval, &self.elements.rc_lo, &self.elements.rc_hi, &ctrl_a, a_active);
        add_rc_lookup(&mut eval, &self.elements.rc_lo, &self.elements.rc_hi, &ctrl_b, b_active);

        // --- Program-consistency (use side, +enabler). ---
        // opcode_scalar = is_not*1 + is_cnot*2 + is_toffoli*3 (NOP -> 0).
        // The op fields are the execution row's existing q columns (ctrl_a.q /
        // ctrl_b.q are 0 when the control is inactive, matching the program
        // table's canonical zero for absent controls). pc_in_prog (preprocessed)
        // = pc mod n_gates indexes the single hidden program. Balances against
        // the program table's -multiplicity supply iff every execution row's
        // (opcode, target, ctrl_a, ctrl_b) equals program[pc_in_prog], i.e. all
        // K*N executions run the SAME program in the right cyclic order.
        let opcode_scalar = is_not.clone()
            + is_cnot.clone() * BaseField::from_u32_unchecked(2)
            + is_toffoli.clone() * BaseField::from_u32_unchecked(3);
        let prog_entry = [
            pc_in_prog.clone(),
            opcode_scalar,
            target.q.clone(),
            ctrl_a.q.clone(),
            ctrl_b.q.clone(),
        ];
        eval.add_to_relation(RelationEntry::new(
            &self.elements.program,
            E::EF::from(enabler.clone()),
            &prog_entry,
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

struct ReadMasks<F> {
    q: F,
    limb_idx: F,
    bit_pos: F,
    mask: F,
    lsel: Vec<F>,
    lo: F,
    hi: F,
    bit: F,
}

/// Structural constraints for a single qubit read.
fn read_constraints<E: EvalAtRow>(
    eval: &mut E,
    r: &ReadMasks<E::F>,
    active: E::F,
    in_limb: &[E::F],
) {
    let one = E::F::one();
    // lsel booleanity.
    for s in &r.lsel {
        eval.add_constraint(s.clone() * (s.clone() - one.clone()));
    }
    // sum lsel = active; sum j*lsel_j = limb_idx.
    let mut sum = E::F::zero();
    let mut weighted = E::F::zero();
    for (j, s) in r.lsel.iter().enumerate() {
        sum += s.clone();
        weighted += s.clone() * BaseField::from_u32_unchecked(j as u32);
    }
    eval.add_constraint(sum - active.clone());
    eval.add_constraint(weighted - r.limb_idx.clone());
    // selected limb L = sum lsel_j * in_limb_j.
    let mut l = E::F::zero();
    for (j, s) in r.lsel.iter().enumerate() {
        l += s.clone() * in_limb[j].clone();
    }
    // variable split: L = hi*2^(pos+1) + bit*2^pos + lo = hi*2*mask + bit*mask + lo.
    let two = BaseField::from_u32_unchecked(2);
    eval.add_constraint(
        l - r.hi.clone() * r.mask.clone() * two
            - r.bit.clone() * r.mask.clone()
            - r.lo.clone(),
    );
    // bit booleanity.
    eval.add_constraint(r.bit.clone() * (r.bit.clone() - one.clone()));
    // inactive => bit forced 0.
    eval.add_constraint((one.clone() - active.clone()) * r.bit.clone());
}

fn read_masks<E: EvalAtRow>(eval: &mut E) -> ReadMasks<E::F> {
    let q = eval.next_trace_mask();
    let limb_idx = eval.next_trace_mask();
    let bit_pos = eval.next_trace_mask();
    let mask = eval.next_trace_mask();
    let lsel = (0..N_LIMBS).map(|_| eval.next_trace_mask()).collect_vec();
    let lo = eval.next_trace_mask();
    let hi = eval.next_trace_mask();
    let bit = eval.next_trace_mask();
    ReadMasks {
        q,
        limb_idx,
        bit_pos,
        mask,
        lsel,
        lo,
        hi,
        bit,
    }
}

fn add_qdecode_lookup<E: EvalAtRow>(
    eval: &mut E,
    elements: &QDecodeElements,
    r: &ReadMasks<E::F>,
    active: E::F,
) {
    // Membership of (q, limb_idx, bit_pos, mask) in the 512-row table, with
    // multiplicity = active (0 for inactive reads => inert).
    let entry = [r.q.clone(), r.limb_idx.clone(), r.bit_pos.clone(), r.mask.clone()];
    eval.add_to_relation(RelationEntry::new(
        elements,
        E::EF::from(active),
        &entry,
    ));
}

fn add_rc_lookup<E: EvalAtRow>(
    eval: &mut E,
    lo_elements: &RangeLoElements,
    hi_elements: &RangeHiElements,
    r: &ReadMasks<E::F>,
    active: E::F,
) {
    let lo_entry = [r.bit_pos.clone(), r.lo.clone()];
    let hi_entry = [r.bit_pos.clone(), r.hi.clone()];
    eval.add_to_relation(RelationEntry::new(
        lo_elements,
        E::EF::from(active.clone()),
        &lo_entry,
    ));
    eval.add_to_relation(RelationEntry::new(
        hi_elements,
        E::EF::from(active),
        &hi_entry,
    ));
}

// ----------------------------------------------------------------------------
// Table FrameworkEvals (supply side of each lookup table)
// ----------------------------------------------------------------------------

#[derive(Clone)]
struct QDecodeTableEval {
    elements: QDecodeElements,
}

impl FrameworkEval for QDecodeTableEval {
    fn log_size(&self) -> u32 {
        N_QUBITS.ilog2()
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let q = eval.get_preprocessed_column(pp_id("gate_qdecode_q"));
        let limb_idx = eval.get_preprocessed_column(pp_id("gate_qdecode_limb"));
        let bit_pos = eval.get_preprocessed_column(pp_id("gate_qdecode_pos"));
        let mask = eval.get_preprocessed_column(pp_id("gate_qdecode_mask"));
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[q, limb_idx, bit_pos, mask],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct RcLoTableEval {
    elements: RangeLoElements,
}

impl FrameworkEval for RcLoTableEval {
    fn log_size(&self) -> u32 {
        RC_LOG_SIZE
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let pos = eval.get_preprocessed_column(pp_id("gate_rc_lo_pos"));
        let val = eval.get_preprocessed_column(pp_id("gate_rc_lo_val"));
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[pos, val],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct RcHiTableEval {
    elements: RangeHiElements,
}

impl FrameworkEval for RcHiTableEval {
    fn log_size(&self) -> u32 {
        RC_LOG_SIZE
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let pos = eval.get_preprocessed_column(pp_id("gate_rc_hi_pos"));
        let val = eval.get_preprocessed_column(pp_id("gate_rc_hi_val"));
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[pos, val],
        ));
        eval.finalize_logup();
        eval
    }
}

/// Program-consistency table (supply side). Slot index is preprocessed; the op
/// fields (opcode_scalar, target, ctrl_a, ctrl_b) are WITNESS (the hidden
/// program) and the multiplicity column counts executions of that slot (K*N on
/// real slots, 0 on padding). Emits -multiplicity / combine(slot, op...).
#[derive(Clone)]
struct ProgramTableEval {
    log_size: u32,
    elements: ProgramElements,
}

impl FrameworkEval for ProgramTableEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let slot = eval.get_preprocessed_column(pp_id("gate_prog_slot"));
        let opcode_scalar = eval.next_trace_mask();
        let target = eval.next_trace_mask();
        let ctrl_a = eval.next_trace_mask();
        let ctrl_b = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[slot, opcode_scalar, target, ctrl_a, ctrl_b],
        ));
        eval.finalize_logup();
        eval
    }
}

fn pp_id(id: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: id.to_owned() }
}

fn preprocessed_column_ids() -> Vec<PreProcessedColumnId> {
    vec![
        pp_id("gate_pc_in_prog"),
        pp_id("gate_qdecode_q"),
        pp_id("gate_qdecode_limb"),
        pp_id("gate_qdecode_pos"),
        pp_id("gate_qdecode_mask"),
        pp_id("gate_rc_lo_pos"),
        pp_id("gate_rc_lo_val"),
        pp_id("gate_rc_hi_pos"),
        pp_id("gate_rc_hi_val"),
        pp_id("gate_prog_slot"),
    ]
}

// ----------------------------------------------------------------------------
// Components bundle
// ----------------------------------------------------------------------------

type GateComponent = FrameworkComponent<GateEval>;
type QDecodeComponent = FrameworkComponent<QDecodeTableEval>;
type RcLoComponent = FrameworkComponent<RcLoTableEval>;
type RcHiComponent = FrameworkComponent<RcHiTableEval>;
type ProgramComponent = FrameworkComponent<ProgramTableEval>;

struct Components {
    main: GateComponent,
    qdecode: QDecodeComponent,
    rc_lo: RcLoComponent,
    rc_hi: RcHiComponent,
    program: ProgramComponent,
}

impl Components {
    fn component_refs(&self) -> Vec<&dyn Component> {
        vec![
            &self.main as &dyn Component,
            &self.qdecode as &dyn Component,
            &self.rc_lo as &dyn Component,
            &self.rc_hi as &dyn Component,
            &self.program as &dyn Component,
        ]
    }

    fn prover_refs(&self) -> Vec<&dyn stwo::prover::ComponentProver<SimdBackend>> {
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.qdecode as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.rc_lo as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.rc_hi as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<SimdBackend>,
        ]
    }

    fn trace_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::concat_cols(
            self.component_refs()
                .into_iter()
                .map(|c| c.trace_log_degree_bounds()),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn build_components(
    log_n_rows: u32,
    program_log_size: u32,
    elements: &LookupElements,
    main_sum: SecureField,
    qdecode_sum: SecureField,
    rc_lo_sum: SecureField,
    rc_hi_sum: SecureField,
    program_sum: SecureField,
) -> Components {
    let mut allocator =
        TraceLocationAllocator::new_with_preprocessed_columns(&preprocessed_column_ids());
    let main = GateComponent::new(
        &mut allocator,
        GateEval {
            log_n_rows,
            elements: elements.clone(),
        },
        main_sum,
    );
    let qdecode = QDecodeComponent::new(
        &mut allocator,
        QDecodeTableEval {
            elements: elements.qdecode.clone(),
        },
        qdecode_sum,
    );
    let rc_lo = RcLoComponent::new(
        &mut allocator,
        RcLoTableEval {
            elements: elements.rc_lo.clone(),
        },
        rc_lo_sum,
    );
    let rc_hi = RcHiComponent::new(
        &mut allocator,
        RcHiTableEval {
            elements: elements.rc_hi.clone(),
        },
        rc_hi_sum,
    );
    let program = ProgramComponent::new(
        &mut allocator,
        ProgramTableEval {
            log_size: program_log_size,
            elements: elements.program.clone(),
        },
        program_sum,
    );
    Components {
        main,
        qdecode,
        rc_lo,
        rc_hi,
        program,
    }
}

// ----------------------------------------------------------------------------
// Trace generation (column-major)
// ----------------------------------------------------------------------------

/// Single scalar cell of `row` at canonical column index `col`. The column order
/// is exactly the serial `put(...)` sequence the old fill used (and the order
/// `evaluate` reads masks): enabler, is_{nop,not,cnot,toffoli}, shot_id, pc,
/// in_limb[32], out_limb[32], then target/ctrl_a/ctrl_b read blocks
/// (q,limb_idx,bit_pos,mask,lsel[32],lo,hi,bit) each, then ab,fire,delta.
#[inline]
fn cell_at(row: &Row, col: usize) -> u32 {
    debug_assert!(col < TRACE_COLUMNS);
    let mut c = col;
    // Header (7 cols).
    const HEADER: [fn(&Row) -> u32; 7] = [
        |r| r.enabler,
        |r| r.is_nop,
        |r| r.is_not,
        |r| r.is_cnot,
        |r| r.is_toffoli,
        |r| r.shot_id,
        |r| r.pc,
    ];
    if c < HEADER.len() {
        return HEADER[c](row);
    }
    c -= HEADER.len();
    if c < N_LIMBS {
        return row.in_limb[c];
    }
    c -= N_LIMBS;
    if c < N_LIMBS {
        return row.out_limb[c];
    }
    c -= N_LIMBS;
    // Three read blocks of READ_COLS each.
    if c < 3 * READ_COLS {
        let which = c / READ_COLS;
        let mut rc = c % READ_COLS;
        let r = match which {
            0 => &row.target,
            1 => &row.ctrl_a,
            _ => &row.ctrl_b,
        };
        // Block order: q, limb_idx, bit_pos, mask, lsel[32], lo, hi, bit.
        return match rc {
            0 => r.q,
            1 => r.limb_idx,
            2 => r.bit_pos,
            3 => r.mask,
            _ => {
                rc -= 4;
                if rc < N_LIMBS {
                    r.lsel[rc]
                } else {
                    match rc - N_LIMBS {
                        0 => r.lo,
                        1 => r.hi,
                        _ => r.bit,
                    }
                }
            }
        };
    }
    c -= 3 * READ_COLS;
    // Tail: ab, fire, delta.
    match c {
        0 => row.ab,
        1 => row.fire,
        _ => row.delta,
    }
}

/// Phase-2 packing of the main trace. The scalar `Vec<Row>` (filled in parallel
/// over shots in `build_rows`) is packed into `PackedM31` columns. Columns are
/// fully independent, so we pack IN PARALLEL OVER COLUMNS: each task owns one
/// column's whole `Vec<PackedM31>` and fills every packed word for it. No two
/// threads ever touch the same column or the same packed word, so a shot block
/// (k*n_gates rows) being non-16-aligned can never cause a packed-word race. The
/// cell→column mapping is identical to the old serial `Col::set(row_idx, ..)`
/// fill, so the trace is bit-identical.
fn generate_main_trace(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    use rayon::prelude::*;
    use stwo::prover::backend::simd::column::BaseColumn;

    let n_vec = padded_rows / LANE_COUNT;
    let pad = Row::padding();
    let domain = CanonicCoset::new(log_n_rows).circle_domain();

    (0..TRACE_COLUMNS)
        .into_par_iter()
        .map(|col| {
            let col_data: Vec<PackedM31> = (0..n_vec)
                .map(|vec_row| {
                    PackedM31::from_array(std::array::from_fn(|lane| {
                        let idx = vec_row * LANE_COUNT + lane;
                        let row = rows.get(idx).unwrap_or(&pad);
                        BaseField::from_u32_unchecked(cell_at(row, col))
                    }))
                })
                .collect();
            CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
                domain,
                BaseColumn::from_simd(col_data),
            )
        })
        .collect()
}

fn col_from_values(values: &[u32]) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let log_size = values.len().ilog2();
    let mut col = Col::<SimdBackend, BaseField>::zeros(values.len());
    for (i, &v) in values.iter().enumerate() {
        col.set(i, BaseField::from_u32_unchecked(v));
    }
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

fn generate_qdecode_preprocessed(
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut q = vec![0u32; N_QUBITS];
    let mut limb = vec![0u32; N_QUBITS];
    let mut pos = vec![0u32; N_QUBITS];
    let mut mask = vec![0u32; N_QUBITS];
    for qi in 0..N_QUBITS {
        let (l, p, m) = qubit_decode(qi as u16);
        q[qi] = qi as u32;
        limb[qi] = l;
        pos[qi] = p;
        mask[qi] = m;
    }
    vec![
        col_from_values(&q),
        col_from_values(&limb),
        col_from_values(&pos),
        col_from_values(&mask),
    ]
}

fn generate_rc_preprocessed(
    idx: &RcIndex,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&idx.pos_col), col_from_values(&idx.val_col)]
}

/// Preprocessed pc_in_prog column for the main trace: pc mod n_gates on real
/// rows, 0 on padding (inert: padding has enabler 0).
fn generate_pc_in_prog_preprocessed(
    rows: &[Row],
    padded_rows: usize,
    n_gates: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let ng = n_gates as u32;
    let mut vals = vec![0u32; padded_rows];
    for (i, r) in rows.iter().enumerate() {
        vals[i] = r.pc % ng;
    }
    col_from_values(&vals)
}

/// Preprocessed slot-index column for the program table.
fn generate_prog_slot_preprocessed(
    prog: &ProgramTable,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    col_from_values(&prog.slot)
}

/// Program-table witness (multiplicity tree): op columns then multiplicity, in
/// the order ProgramTableEval reads them.
fn generate_program_witness(
    prog: &ProgramTable,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    vec![
        col_from_values(&prog.opcode_scalar),
        col_from_values(&prog.target),
        col_from_values(&prog.ctrl_a),
        col_from_values(&prog.ctrl_b),
        col_from_values(&prog.multiplicity),
    ]
}

fn generate_multiplicity_trace(
    counts: &[u32],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(counts)]
}

// ----------------------------------------------------------------------------
// Interaction traces
// ----------------------------------------------------------------------------

fn packed_rows(rows: &[Row], padded_rows: usize, mut f: impl FnMut(usize, &[&Row; LANE_COUNT])) {
    let pad = Row::padding();
    let n_vec = padded_rows / LANE_COUNT;
    for vec_row in 0..n_vec {
        let lane: [&Row; LANE_COUNT] = std::array::from_fn(|lane| {
            let idx = vec_row * LANE_COUNT + lane;
            rows.get(idx).unwrap_or(&pad)
        });
        f(vec_row, &lane);
    }
}

#[inline]
fn pack(lane: &[&Row; LANE_COUNT], get: impl Fn(&Row) -> u32) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|l| {
        BaseField::from_u32_unchecked(get(lane[l]))
    }))
}

/// Main component interaction trace. Logup batches mirror the relation entries
/// emitted in `evaluate`, finalized in pairs. Order must match (12 entries -> 6
/// full pairs):
///   pair0: state_in (+enabler), state_out (-enabler)
///   pair1: qdecode target (+enabler), qdecode ctrl_a (+a_active)
///   pair2: qdecode ctrl_b (+b_active), rc_lo target (+enabler)
///   pair3: rc_hi target (+enabler), rc_lo ctrl_a (+a_active)
///   pair4: rc_hi ctrl_a (+a_active), rc_lo ctrl_b (+b_active)
///   pair5: rc_hi ctrl_b (+b_active), program (+enabler)
fn gen_main_interaction(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    el: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(log_n_rows);

    // Entry combiners.
    let state_in = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        let mut v = Vec::with_capacity(STATE_WIDTH);
        v.push(pack(lane, |r| r.shot_id));
        v.push(pack(lane, |r| r.pc));
        for j in 0..N_LIMBS {
            v.push(pack(lane, |r| r.in_limb[j]));
        }
        el.state.combine(&v)
    };
    let state_out = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        let mut v = Vec::with_capacity(STATE_WIDTH);
        v.push(pack(lane, |r| r.shot_id));
        v.push(pack(lane, |r| r.pc + 1));
        for j in 0..N_LIMBS {
            v.push(pack(lane, |r| r.out_limb[j]));
        }
        el.state.combine(&v)
    };
    let enabler = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.enabler);
    let a_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_cnot + r.is_toffoli);
    let b_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_toffoli);

    let qdecode = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &ReadCols| -> PackedSecureField {
        el.qdecode.combine(&[
            pack(lane, |r| sel(r).q),
            pack(lane, |r| sel(r).limb_idx),
            pack(lane, |r| sel(r).bit_pos),
            pack(lane, |r| sel(r).mask),
        ])
    };
    let rc_lo = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &ReadCols| -> PackedSecureField {
        el.rc_lo
            .combine(&[pack(lane, |r| sel(r).bit_pos), pack(lane, |r| sel(r).lo)])
    };
    let rc_hi = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &ReadCols| -> PackedSecureField {
        el.rc_hi
            .combine(&[pack(lane, |r| sel(r).bit_pos), pack(lane, |r| sel(r).hi)])
    };
    // Program use-side denominator. pc_in_prog = pc mod n_gates (preprocessed in
    // the AIR; recomputed here for the prover). opcode_scalar from the one-hot.
    let ng = n_gates as u32;
    let program = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.program.combine(&[
            pack(lane, |r| r.pc % ng),
            pack(lane, |r| r.is_not + 2 * r.is_cnot + 3 * r.is_toffoli),
            pack(lane, |r| r.target.q),
            pack(lane, |r| r.ctrl_a.q),
            pack(lane, |r| r.ctrl_b.q),
        ])
    };

    fn sel_t(r: &Row) -> &ReadCols {
        &r.target
    }
    fn sel_a(r: &Row) -> &ReadCols {
        &r.ctrl_a
    }
    fn sel_b(r: &Row) -> &ReadCols {
        &r.ctrl_b
    }

    // Helper to write one logup column for a (+num/denom) pair of relation
    // entries. num_i = mult_i; combined fraction = m0/d0 + m1/d1.
    let write_pair = |gen: &mut LogupTraceGenerator,
                          num0: &dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31,
                          den0: &dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField,
                          sign0: i32,
                          num1: Option<&dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31>,
                          den1: Option<&dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField>,
                          sign1: i32| {
        let mut col = gen.new_col();
        packed_rows(rows, padded_rows, |vec_row, lane| {
            let m0 = PackedSecureField::from(num0(lane));
            let d0 = den0(lane);
            let m0 = if sign0 < 0 { -m0 } else { m0 };
            let (num, den) = match (num1, den1) {
                (Some(n1), Some(d1)) => {
                    let m1 = PackedSecureField::from(n1(lane));
                    let m1 = if sign1 < 0 { -m1 } else { m1 };
                    let dd1 = d1(lane);
                    (m0 * dd1 + m1 * d0, d0 * dd1)
                }
                _ => (m0, d0),
            };
            col.write_frac(vec_row, num, den);
        });
        col.finalize_col();
    };

    // pair0: state_in (+enabler), state_out (-enabler).
    write_pair(
        &mut gen,
        &enabler,
        &state_in,
        1,
        Some(&enabler),
        Some(&state_out),
        -1,
    );
    // pair1: qdecode target (+enabler), qdecode ctrl_a (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| qdecode(l, sel_t),
        1,
        Some(&a_active),
        Some(&|l| qdecode(l, sel_a)),
        1,
    );
    // pair2: qdecode ctrl_b (+b_active), rc_lo target (+enabler).
    write_pair(
        &mut gen,
        &b_active,
        &|l| qdecode(l, sel_b),
        1,
        Some(&enabler),
        Some(&|l| rc_lo(l, sel_t)),
        1,
    );
    // pair3: rc_hi target (+enabler), rc_lo ctrl_a (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| rc_hi(l, sel_t),
        1,
        Some(&a_active),
        Some(&|l| rc_lo(l, sel_a)),
        1,
    );
    // pair4: rc_hi ctrl_a (+a_active), rc_lo ctrl_b (+b_active).
    write_pair(
        &mut gen,
        &a_active,
        &|l| rc_hi(l, sel_a),
        1,
        Some(&b_active),
        Some(&|l| rc_lo(l, sel_b)),
        1,
    );
    // pair5: rc_hi ctrl_b (+b_active), program (+enabler).
    write_pair(
        &mut gen,
        &b_active,
        &|l| rc_hi(l, sel_b),
        1,
        Some(&enabler),
        Some(&program),
        1,
    );

    gen.finalize_last()
}

/// Debug-only: assert the main component's AIR constraints (algebraic + logup)
/// directly on the committed trace columns, pinpointing the first violated
/// constraint index. Builds the trees `[preprocessed(empty), main, interaction]`
/// the main `GateEval` expects and runs `assert_constraints_on_trace`.
fn assert_main_constraints(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    elements: &LookupElements,
    main_interaction: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    main_sum: SecureField,
) {
    // Main trace columns as plain M31 vectors (circle-domain / bit-reversed
    // order, matching what `assert_constraints_on_trace` expects).
    let main_cols = generate_main_trace(rows, padded_rows, log_n_rows);
    let main_vals: Vec<Vec<BaseField>> = main_cols
        .iter()
        .map(|c| c.values.to_cpu())
        .collect();
    let interaction_vals: Vec<Vec<BaseField>> = main_interaction
        .iter()
        .map(|c| c.values.to_cpu())
        .collect();
    // The main component reads one preprocessed column: pc_in_prog.
    let pp_vals: Vec<Vec<BaseField>> =
        vec![generate_pc_in_prog_preprocessed(rows, padded_rows, n_gates).values.to_cpu()];

    // Tree layout: [preprocessed, original, interaction].
    let preprocessed: Vec<&Vec<BaseField>> = pp_vals.iter().collect();
    let original: Vec<&Vec<BaseField>> = main_vals.iter().collect();
    let interaction: Vec<&Vec<BaseField>> = interaction_vals.iter().collect();
    let trace = TreeVec::new(vec![preprocessed, original, interaction]);

    let eval = GateEval {
        log_n_rows,
        elements: elements.clone(),
    };
    assert_constraints_on_trace(
        &trace,
        log_n_rows,
        |assert_eval| {
            eval.evaluate(assert_eval);
        },
        main_sum,
    );
}

/// Debug-only: assert a table (supply-side) component's constraints directly on
/// its committed columns. Trees: `[preprocessed, multiplicity, interaction]`.
fn assert_table_constraints<Ev: FrameworkEval + Sync>(
    log_size: u32,
    preprocessed: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    multiplicity: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    interaction: &[CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    claimed_sum: SecureField,
    eval: Ev,
) {
    let pp_vals: Vec<Vec<BaseField>> = preprocessed.iter().map(|c| c.values.to_cpu()).collect();
    let mult_vals: Vec<Vec<BaseField>> = multiplicity.iter().map(|c| c.values.to_cpu()).collect();
    let int_vals: Vec<Vec<BaseField>> = interaction.iter().map(|c| c.values.to_cpu()).collect();
    let trace = TreeVec::new(vec![
        pp_vals.iter().collect::<Vec<_>>(),
        mult_vals.iter().collect::<Vec<_>>(),
        int_vals.iter().collect::<Vec<_>>(),
    ]);
    assert_constraints_on_trace(
        &trace,
        log_size,
        |assert_eval| {
            eval.evaluate(assert_eval);
        },
        claimed_sum,
    );
}

/// Supply-side interaction trace for a multi-column table looked up with a
/// single relation. `combine_row(i)` returns the combined denominator for row i.
fn gen_table_interaction<R>(
    counts: &[u32],
    log_size: u32,
    combine_row: impl Fn(usize) -> PackedSecureField,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
)
where
    R: Sized,
{
    let mut gen = LogupTraceGenerator::new(log_size);
    let mut col = gen.new_col();
    for vec_row in 0..(1usize << (log_size - LOG_N_LANES)) {
        let denom = combine_row(vec_row);
        let packed_counts = PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(counts[(vec_row << LOG_N_LANES) + lane])
        }));
        col.write_frac(vec_row, -PackedSecureField::from(packed_counts), denom);
    }
    col.finalize_col();
    gen.finalize_last()
}

// ----------------------------------------------------------------------------
// Public boundary sums
// ----------------------------------------------------------------------------

fn public_boundary_sum(
    cases: &[TestCase],
    n_gates: usize,
    k: usize,
    state: &StateElements,
) -> Result<SecureField> {
    let mut sum = SecureField::zero();
    let total_pc = (n_gates * k) as u32;
    for (shot_id, case) in cases.iter().enumerate() {
        let x = hex::decode(&case.x_hex)?;
        let y = hex::decode(&case.y_hex)?;
        let x_limbs = state_to_limbs(&x);
        let y_limbs = state_to_limbs(&y);
        let initial = state_tuple(shot_id as u32, 0, &x_limbs);
        let final_ = state_tuple(shot_id as u32, total_pc, &y_limbs);
        let ci: SecureField = state.combine(&initial);
        let cf: SecureField = state.combine(&final_);
        sum += ci.inverse() - cf.inverse();
    }
    Ok(sum)
}

fn state_tuple(shot_id: u32, pc: u32, limbs: &[u32; N_LIMBS]) -> [BaseField; STATE_WIDTH] {
    std::array::from_fn(|i| {
        let v = if i == 0 {
            shot_id
        } else if i == 1 {
            pc
        } else {
            limbs[i - 2]
        };
        BaseField::from_u32_unchecked(v)
    })
}

/// Supply (table-side) public sum for a single-relation table: sum_i count_i / denom_i.
/// Returns the negated value (matching the supply emission of -multiplicity).
fn table_public_sum<R: Relation<BaseField, SecureField>>(
    counts: &[u32],
    elements: &R,
    row_tuple: impl Fn(usize) -> Vec<BaseField>,
) -> SecureField {
    let mut sum = SecureField::zero();
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let denom: SecureField = elements.combine(&row_tuple(i));
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    -sum
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let fixture_path = normalize(args.fixture);
    let fixture: Fixture = {
        let f = File::open(&fixture_path)
            .with_context(|| format!("opening fixture {}", fixture_path.display()))?;
        serde_json::from_reader(f).context("parsing fixture")?
    };

    if fixture.version != "v0.3-iadd" {
        bail!("expected v0.3-iadd fixture, got {}", fixture.version);
    }
    if fixture.num_qubits != N_QUBITS {
        bail!("expected {N_QUBITS} qubits, got {}", fixture.num_qubits);
    }

    let gates = parse_gtv1(&fixture.circuit_byte_serialisation_hex)?;
    let n_gates = gates.len();
    let k = args.repetitions.unwrap_or(fixture.repetitions);
    let samples = args
        .samples
        .min(fixture.n_samples)
        .min(fixture.test_cases.len());
    if samples == 0 || k == 0 {
        bail!("samples and repetitions must be non-zero");
    }

    let rc_lo_index = build_rc_lo();
    let rc_hi_index = build_rc_hi();
    let program = build_program_table(&gates, samples, k);

    let cases = &fixture.test_cases[..samples];

    // `trace_gen_start` marks the beginning of the full witness build (shot
    // simulation + every column fill + interaction traces). We stop the clock
    // immediately before the FRI `prove` call; `prove_s`/`verify_s` stay as-is.
    let trace_gen_start = Instant::now();

    let build_start = Instant::now();
    let (rows, counts) = build_rows(&gates, cases, k, &rc_lo_index, &rc_hi_index)?;
    let build_elapsed = build_start.elapsed();

    let real_rows = rows.len();
    let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let log_n_rows = padded_rows.ilog2();

    if (samples * k * n_gates) >= M31_MODULUS_U32 as usize {
        bail!("pc timeline does not fit in M31");
    }

    eprintln!(
        "gate-air: samples={} K={} n_gates={} real_rows={} padded_rows={} log_rows={} columns={}",
        samples, k, n_gates, real_rows, padded_rows, log_n_rows, TRACE_COLUMNS
    );
    eprintln!(
        "gate-air: shots simulated and self-checked (final state == y) in {:.3}s",
        build_elapsed.as_secs_f64()
    );

    if args.no_prove {
        println!(
            "{{\"schema\":\"gate-air-report/v1\",\"samples\":{samples},\"repetitions\":{k},\"n_gates\":{n_gates},\"real_rows\":{real_rows},\"padded_rows\":{padded_rows},\"log_rows\":{log_n_rows},\"trace_columns\":{TRACE_COLUMNS},\"proved\":false,\"self_check\":\"final_state_matches_y\"}}"
        );
        return Ok(());
    }

    // ---- Proving ----
    let config = PcsConfig::default();
    let max_log_size = log_n_rows.max(RC_LOG_SIZE);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let prover_channel = &mut Blake2sM31Channel::default();
    config.mix_into(prover_channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

    // Tree 0: preprocessed. Order MUST match preprocessed_column_ids():
    //   pc_in_prog, qdecode(4), rc_lo(2), rc_hi(2), prog_slot.
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut pp = vec![generate_pc_in_prog_preprocessed(&rows, padded_rows, n_gates)];
    pp.extend(generate_qdecode_preprocessed());
    pp.extend(generate_rc_preprocessed(&rc_lo_index));
    pp.extend(generate_rc_preprocessed(&rc_hi_index));
    pp.push(generate_prog_slot_preprocessed(&program));
    tree_builder.extend_evals(pp);
    tree_builder.commit(prover_channel);

    // Tree 1: main trace + table multiplicities + program witness (op cols+mult).
    let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
    main_trace.extend(generate_multiplicity_trace(&counts.qdecode));
    main_trace.extend(generate_multiplicity_trace(&counts.rc_lo));
    main_trace.extend(generate_multiplicity_trace(&counts.rc_hi));
    main_trace.extend(generate_program_witness(&program));
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(main_trace);
    tree_builder.commit(prover_channel);

    // Draw relation elements.
    let elements = LookupElements::draw(prover_channel);

    // Interaction traces.
    let (main_interaction, main_sum) =
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
    let (qdecode_interaction, qdecode_sum) = {
        let el = elements.qdecode.clone();
        let q: Vec<u32> = (0..N_QUBITS as u32).collect();
        gen_table_interaction::<QDecodeElements>(&counts.qdecode, N_QUBITS.ilog2(), |vec_row| {
            el.combine(&[
                pack_seq(&q, vec_row),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).0),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).1),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).2),
            ])
        })
    };
    let (rc_lo_interaction, rc_lo_sum) = {
        let el = elements.rc_lo.clone();
        gen_table_interaction::<RangeLoElements>(&counts.rc_lo, RC_LOG_SIZE, |vec_row| {
            el.combine(&[
                pack_seq(&rc_lo_index.pos_col, vec_row),
                pack_seq(&rc_lo_index.val_col, vec_row),
            ])
        })
    };
    let (rc_hi_interaction, rc_hi_sum) = {
        let el = elements.rc_hi.clone();
        gen_table_interaction::<RangeHiElements>(&counts.rc_hi, RC_LOG_SIZE, |vec_row| {
            el.combine(&[
                pack_seq(&rc_hi_index.pos_col, vec_row),
                pack_seq(&rc_hi_index.val_col, vec_row),
            ])
        })
    };
    let (program_interaction, program_sum) = {
        let el = elements.program.clone();
        gen_table_interaction::<ProgramElements>(&program.multiplicity, program.log_size, |vec_row| {
            el.combine(&[
                pack_seq(&program.slot, vec_row),
                pack_seq(&program.opcode_scalar, vec_row),
                pack_seq(&program.target, vec_row),
                pack_seq(&program.ctrl_a, vec_row),
                pack_seq(&program.ctrl_b, vec_row),
            ])
        })
    };

    // Debug: directly assert the main AIR constraints over the committed trace
    // (no FRI / proof). Pinpoints the first violated constraint index. Cheap;
    // intended for the small k4-n16 N=1 case. Gate with GATE_AIR_ASSERT=1.
    if std::env::var("GATE_AIR_ASSERT").is_ok() {
        assert_main_constraints(
            &rows,
            padded_rows,
            log_n_rows,
            n_gates,
            &elements,
            &main_interaction,
            main_sum,
        );
        eprintln!("gate-air: GATE_AIR_ASSERT main OK");

        // qdecode table.
        let qd_pp = generate_qdecode_preprocessed();
        let qd_mult = generate_multiplicity_trace(&counts.qdecode);
        assert_table_constraints(
            N_QUBITS.ilog2(),
            &qd_pp,
            &qd_mult,
            &qdecode_interaction,
            qdecode_sum,
            QDecodeTableEval {
                elements: elements.qdecode.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT qdecode OK");

        // rc_lo table.
        let lo_pp = generate_rc_preprocessed(&rc_lo_index);
        let lo_mult = generate_multiplicity_trace(&counts.rc_lo);
        assert_table_constraints(
            RC_LOG_SIZE,
            &lo_pp,
            &lo_mult,
            &rc_lo_interaction,
            rc_lo_sum,
            RcLoTableEval {
                elements: elements.rc_lo.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT rc_lo OK");

        // rc_hi table.
        let hi_pp = generate_rc_preprocessed(&rc_hi_index);
        let hi_mult = generate_multiplicity_trace(&counts.rc_hi);
        assert_table_constraints(
            RC_LOG_SIZE,
            &hi_pp,
            &hi_mult,
            &rc_hi_interaction,
            rc_hi_sum,
            RcHiTableEval {
                elements: elements.rc_hi.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT rc_hi OK");

        // program-consistency table.
        let prog_pp = vec![generate_prog_slot_preprocessed(&program)];
        let prog_wit = generate_program_witness(&program);
        assert_table_constraints(
            program.log_size,
            &prog_pp,
            &prog_wit,
            &program_interaction,
            program_sum,
            ProgramTableEval {
                log_size: program.log_size,
                elements: elements.program.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT program OK (all components satisfied on trace)");

        // Validate the prover cross-check (boundary identity incl. program) here
        // too, so the assert path exercises soundness wiring without FRI. Allow
        // skipping the heavy FRI prove (laptop rule) via GATE_AIR_ASSERT_ONLY.
        if std::env::var("GATE_AIR_ASSERT_ONLY").is_ok() {
            let boundary = public_boundary_sum(cases, n_gates, k, &elements.state)?;
            if main_sum + qdecode_sum + rc_lo_sum + rc_hi_sum + program_sum != boundary {
                bail!("ASSERT_ONLY: main+tables+program != boundary");
            }
            eprintln!("gate-air: GATE_AIR_ASSERT_ONLY cross-check OK (skipping FRI prove)");
            return Ok(());
        }
    }

    // Cross-check claimed sums against the public boundary + table sums.
    //
    // LogUp sign accounting (matches the iadd reference in ../src/main.rs):
    //   * The main component emits, per row:
    //       state:   +enabler / state_in   -enabler / state_out   (telescopes
    //                to the public boundary = sum_shot (1/ci - 1/cf)).
    //       lookups: +active / row         (the "use" side, positive mult).
    //   * Each table component emits the "supply" side: -mult / row.
    //   `gen_table_interaction` therefore returns the *supply* sum
    //   (-sum count/denom), and `table_public_sum` recomputes that same
    //   negated supply sum -- so the per-table checks below compare like with
    //   like and are already correct.
    //   `gen_main_interaction` returns the boundary PLUS the positive use
    //   sides, i.e. main_sum = boundary - (qdecode_sum + rc_lo_sum + rc_hi_sum
    //   + program_sum), because each use side equals -(its supply sum).
    //   Rearranged, the global identity is:
    //     main_sum + qdecode_sum + rc_lo_sum + rc_hi_sum + program_sum == boundary.
    //   The program use side balances the program table's -multiplicity supply
    //   iff every execution row's (opcode,target,ctrl_a,ctrl_b) equals
    //   program[pc_in_prog]: the single hidden program, run in cyclic order.
    //
    // SOUNDNESS INVARIANT (per-input guarantee): the "P run K times on EACH of
    // N inputs" statement comes from public_boundary_sum being PER-SHOT -- it
    // sums N distinct source/sink pairs (shot_i, 0, x_i) and
    // (shot_i, K*n_gates, y_i). If this boundary is ever collapsed to a single
    // (x, y) pair, the statement silently degrades to "P run K*N times" with no
    // per-input binding -- keep it per-shot. Program-consistency (forcing one
    // shared program) does NOT replace this: it constrains WHICH program runs,
    // not that each of the N inputs is independently mapped x_i -> y_i.
    let boundary = public_boundary_sum(cases, n_gates, k, &elements.state)?;
    if main_sum + qdecode_sum + rc_lo_sum + rc_hi_sum + program_sum != boundary {
        bail!("main claimed sum != public boundary sum");
    }
    let qdecode_expected = table_public_sum(&counts.qdecode, &elements.qdecode, |qi| {
        let (l, p, m) = qubit_decode(qi as u16);
        vec![
            BaseField::from_u32_unchecked(qi as u32),
            BaseField::from_u32_unchecked(l),
            BaseField::from_u32_unchecked(p),
            BaseField::from_u32_unchecked(m),
        ]
    });
    if qdecode_sum != qdecode_expected {
        bail!("qdecode claimed sum mismatch");
    }
    let rc_lo_expected = table_public_sum(&counts.rc_lo, &elements.rc_lo, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_lo_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_lo_index.val_col[i]),
        ]
    });
    if rc_lo_sum != rc_lo_expected {
        bail!("rc_lo claimed sum mismatch");
    }
    let rc_hi_expected = table_public_sum(&counts.rc_hi, &elements.rc_hi, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_hi_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_hi_index.val_col[i]),
        ]
    });
    if rc_hi_sum != rc_hi_expected {
        bail!("rc_hi claimed sum mismatch");
    }
    let program_expected = table_public_sum(&program.multiplicity, &elements.program, |i| {
        vec![
            BaseField::from_u32_unchecked(program.slot[i]),
            BaseField::from_u32_unchecked(program.opcode_scalar[i]),
            BaseField::from_u32_unchecked(program.target[i]),
            BaseField::from_u32_unchecked(program.ctrl_a[i]),
            BaseField::from_u32_unchecked(program.ctrl_b[i]),
        ]
    });
    if program_sum != program_expected {
        bail!("program claimed sum mismatch");
    }

    // Order MUST match the verifier's reconstruction below and build_components.
    let claimed_sums = vec![main_sum, qdecode_sum, rc_lo_sum, rc_hi_sum, program_sum];
    prover_channel.mix_felts(&claimed_sums);

    // Tree 2: interaction (same component order as the claimed sums).
    let mut interaction = main_interaction;
    interaction.extend(qdecode_interaction);
    interaction.extend(rc_lo_interaction);
    interaction.extend(rc_hi_interaction);
    interaction.extend(program_interaction);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction);
    tree_builder.commit(prover_channel);

    let components = build_components(
        log_n_rows,
        program.log_size,
        &elements,
        main_sum,
        qdecode_sum,
        rc_lo_sum,
        rc_hi_sum,
        program_sum,
    );
    let sizes = components.trace_log_sizes();
    let prover_refs = components.prover_refs();

    // Full witness-build wall: everything from the start of shot simulation
    // through all column + interaction-trace generation, up to (not including)
    // the FRI prove. This is the metric we sweep for trace-gen scaling.
    let trace_gen_elapsed = trace_gen_start.elapsed();
    eprintln!(
        "gate-air: trace generation (full witness build) in {:.3}s",
        trace_gen_elapsed.as_secs_f64()
    );

    let prove_start = Instant::now();
    let proof =
        prove::<SimdBackend, Blake2sM31MerkleChannel>(&prover_refs, prover_channel, commitment_scheme)?;
    let prove_elapsed = prove_start.elapsed();

    // ---- Verify ----
    let verify_start = Instant::now();
    let verifier_channel = &mut Blake2sM31Channel::default();
    config.mix_into(verifier_channel);
    let commitment_scheme_v = &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
    commitment_scheme_v.commit(proof.commitments[0], &sizes[0], verifier_channel);
    commitment_scheme_v.commit(proof.commitments[1], &sizes[1], verifier_channel);
    let v_elements = LookupElements::draw(verifier_channel);
    let v_boundary = public_boundary_sum(cases, n_gates, k, &v_elements.state)?;
    let v_qdecode = table_public_sum(&counts.qdecode, &v_elements.qdecode, |qi| {
        let (l, p, m) = qubit_decode(qi as u16);
        vec![
            BaseField::from_u32_unchecked(qi as u32),
            BaseField::from_u32_unchecked(l),
            BaseField::from_u32_unchecked(p),
            BaseField::from_u32_unchecked(m),
        ]
    });
    let v_rc_lo = table_public_sum(&counts.rc_lo, &v_elements.rc_lo, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_lo_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_lo_index.val_col[i]),
        ]
    });
    let v_rc_hi = table_public_sum(&counts.rc_hi, &v_elements.rc_hi, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_hi_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_hi_index.val_col[i]),
        ]
    });
    let v_program = table_public_sum(&program.multiplicity, &v_elements.program, |i| {
        vec![
            BaseField::from_u32_unchecked(program.slot[i]),
            BaseField::from_u32_unchecked(program.opcode_scalar[i]),
            BaseField::from_u32_unchecked(program.target[i]),
            BaseField::from_u32_unchecked(program.ctrl_a[i]),
            BaseField::from_u32_unchecked(program.ctrl_b[i]),
        ]
    });
    // The main component's claimed sum is NOT the public boundary: the main
    // component emits the boundary telescoping PLUS the positive "use" sides of
    // every lookup, so main_sum = boundary - (qdecode + rc_lo + rc_hi +
    // program). This must equal the prover's `main_sum`, both for the channel
    // mix (Fiat-Shamir) and for the component's claimed sum used in the
    // OODS/DEEP-ALI check. Subtraction order is irrelevant (commutative) but the
    // mix_felts ORDER below must match the prover's exactly.
    let v_main = v_boundary - v_qdecode - v_rc_lo - v_rc_hi - v_program;
    let v_claimed = vec![v_main, v_qdecode, v_rc_lo, v_rc_hi, v_program];
    verifier_channel.mix_felts(&v_claimed);
    let v_components = build_components(
        log_n_rows,
        program.log_size,
        &v_elements,
        v_main,
        v_qdecode,
        v_rc_lo,
        v_rc_hi,
        v_program,
    );
    commitment_scheme_v.commit(proof.commitments[2], &sizes[2], verifier_channel);
    verify(
        &v_components.component_refs(),
        verifier_channel,
        commitment_scheme_v,
        proof,
    )?;
    let verify_elapsed = verify_start.elapsed();

    println!(
        "{{\"schema\":\"gate-air-report/v1\",\"samples\":{samples},\"repetitions\":{k},\"n_gates\":{n_gates},\"real_rows\":{real_rows},\"padded_rows\":{padded_rows},\"log_rows\":{log_n_rows},\"trace_columns\":{TRACE_COLUMNS},\"proved\":true,\"trace_gen_s\":{:.3},\"prove_s\":{:.3},\"verify_s\":{:.3}}}",
        trace_gen_elapsed.as_secs_f64(),
        prove_elapsed.as_secs_f64(),
        verify_elapsed.as_secs_f64()
    );

    Ok(())
}

// Pack a flat sequence column for a vec_row.
fn pack_seq(values: &[u32], vec_row: usize) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(values[(vec_row << LOG_N_LANES) + lane])
    }))
}

// Pack a decoded value computed from the qubit index for the qdecode table.
fn pack_decode(vec_row: usize, f: impl Fn(usize) -> u32) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(f((vec_row << LOG_N_LANES) + lane))
    }))
}

fn normalize(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}
