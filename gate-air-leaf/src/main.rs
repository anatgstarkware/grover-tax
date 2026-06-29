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
use stwo::core::pcs::{CommitmentSchemeVerifier, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
// Trace-gen backend: ALWAYS SimdBackend. All witness/interaction/preprocessed columns are built
// with cheap per-element CPU column ops (`Col::set`, `BaseColumn::from_simd`, LogupTraceGenerator),
// which require a SimdBackend-layout column. The prover backend may differ (see `ProverBackend`);
// `to_prover` bridges trace-gen columns to the prover backend at the `extend_evals` boundary.
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
// Prover backend (commit + prove_ex): SimdBackend by default; obelyzk GpuBackend under `gpu`
// (model A: CPU trace-gen, GPU commit+prove_ex); device-resident CudaBackend under `cuda`.
// SimdBackend/GpuBackend share trace-gen's column layout (rewrap is identity/transmute-compatible);
// CudaBackend stores device columns, so `to_prover` performs a real host->device upload.
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
use stwo::prover::backend::gpu::GpuBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::core::proof_of_work::GrindOps;
use stwo::prover::{prove_ex, CommitmentSchemeProver};
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    assert_constraints_on_trace, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator,
    Relation, RelationEntry, TraceLocationAllocator,
};

// In-circuit verifier of the gate_air STARK proof (Design A, Milestone 2).
mod circuit_statement;
// Accumulator-diff scaffold for the GPU constraint kernel (CPU-vs-GPU composition diff).
mod accumulator_diff;
#[cfg(feature = "gpu")]
mod gpu_tracegen;
mod leaf;

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

// Interaction-trace proof-of-work bits (canonical transcript; matches the in-circuit verifier's
// ProofConfig). Tiny grind (~2^8), present so the in-circuit verifier can replay the transcript.
const INTERACTION_POW_BITS: u32 = 8;

// Blowup factor for the BASE gate_air proof (the shard / "leaves"). The (n_queries, pow_bits) and
// lifting are derived from this via `leaf::leaf_pcs_config` to a ~96-bit-secure config (passes the
// privacy-verifier security test: pow + n_queries*blowup >= 96). Sweep knob: 1/2/3.
const BASE_LOG_BLOWUP_FACTOR: u32 = 1;

const NO_CTRL: u16 = 0xFFFF;

const M31_MODULUS_U32: u32 = (1 << 31) - 1;
const LANE_COUNT: usize = 1 << LOG_N_LANES;

// ----------------------------------------------------------------------------
// Relations
// ----------------------------------------------------------------------------

// ONE shared LogUp relation (single drawn (z,α)). The 5 logical relations are
// distinguished by a distinct id TAG prepended as the first tuple element — this
// matches the in-circuit verifier's single-relation model (circuits_stark_verifier:
// one acc.interaction_elements, relation id as a constant in the tuple). Width =
// widest payload (state = STATE_WIDTH = 34) + 1 tag.
const GATE_REL_WIDTH: usize = 1 + STATE_WIDTH;
stwo_constraint_framework::relation!(GateRel, 35);
const _: () = assert!(GATE_REL_WIDTH <= 35);

// Relation id tags (distinct constants; the prover and the in-circuit verifier must agree).
const TAG_STATE: u32 = 1;
const TAG_QDECODE: u32 = 2;
const TAG_RC_LO: u32 = 3;
const TAG_RC_HI: u32 = 4;
const TAG_PROGRAM: u32 = 5;

/// All five logical relations share the SAME drawn `(z,α)`; the fields are clones of
/// the single `GateRel`, kept under named handles for readable call sites. The tag
/// prepended at each combine is what keeps the relations separate.
#[derive(Clone)]
struct LookupElements {
    state: GateRel,
    qdecode: GateRel,
    rc_lo: GateRel,
    rc_hi: GateRel,
    program: GateRel,
}

impl LookupElements {
    fn draw(channel: &mut impl Channel) -> Self {
        let rel = GateRel::draw(channel);
        Self {
            state: rel.clone(),
            qdecode: rel.clone(),
            rc_lo: rel.clone(),
            rc_hi: rel.clone(),
            program: rel,
        }
    }

    /// Fixed challenges for byte-identity tests (mirrors `draw`: one `GateRel` shared
    /// across all five relations). Used by the K4 GPU interaction validation.
    #[cfg(feature = "gpu-cuda")]
    fn dummy() -> Self {
        let rel = GateRel::dummy();
        Self {
            state: rel.clone(),
            qdecode: rel.clone(),
            rc_lo: rel.clone(),
            rc_hi: rel.clone(),
            program: rel,
        }
    }
}

/// Packed tag constant for prover-side `combine` tuples.
fn ptag(tag: u32) -> PackedM31 {
    PackedM31::broadcast(BaseField::from_u32_unchecked(tag))
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

#[derive(Debug, Clone, Deserialize)]
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
        let state_tag = one.clone() * BaseField::from_u32_unchecked(TAG_STATE);
        let mut input_state = Vec::with_capacity(GATE_REL_WIDTH);
        input_state.push(state_tag.clone());
        input_state.push(shot_id.clone());
        input_state.push(pc.clone());
        input_state.extend(in_limb.iter().cloned());
        let mut output_state = Vec::with_capacity(GATE_REL_WIDTH);
        output_state.push(state_tag);
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
            one.clone() * BaseField::from_u32_unchecked(TAG_PROGRAM),
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
    elements: &GateRel,
    r: &ReadMasks<E::F>,
    active: E::F,
) {
    // Membership of (q, limb_idx, bit_pos, mask) in the 512-row table, with
    // multiplicity = active (0 for inactive reads => inert). Tagged TAG_QDECODE.
    let entry = [
        E::F::one() * BaseField::from_u32_unchecked(TAG_QDECODE),
        r.q.clone(),
        r.limb_idx.clone(),
        r.bit_pos.clone(),
        r.mask.clone(),
    ];
    eval.add_to_relation(RelationEntry::new(
        elements,
        E::EF::from(active),
        &entry,
    ));
}

fn add_rc_lookup<E: EvalAtRow>(
    eval: &mut E,
    lo_elements: &GateRel,
    hi_elements: &GateRel,
    r: &ReadMasks<E::F>,
    active: E::F,
) {
    let lo_entry = [
        E::F::one() * BaseField::from_u32_unchecked(TAG_RC_LO),
        r.bit_pos.clone(),
        r.lo.clone(),
    ];
    let hi_entry = [
        E::F::one() * BaseField::from_u32_unchecked(TAG_RC_HI),
        r.bit_pos.clone(),
        r.hi.clone(),
    ];
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
    elements: GateRel,
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
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_QDECODE);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, q, limb_idx, bit_pos, mask],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct RcLoTableEval {
    elements: GateRel,
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
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC_LO);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, pos, val],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct RcHiTableEval {
    elements: GateRel,
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
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC_HI);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, pos, val],
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
    elements: GateRel,
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
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_PROGRAM);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, slot, opcode_scalar, target, ctrl_a, ctrl_b],
        ));
        eval.finalize_logup();
        eval
    }
}

fn pp_id(id: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: id.to_owned() }
}

/// Number of preprocessed columns (count-only uses; the order is `preprocessed_column_ids`).
const N_PREPROCESSED_COLS: usize = 10;

/// Each preprocessed column paired with its log_size, in a fixed canonical listing order, then
/// STABLE-sorted ascending by size. The committed preprocessed tree MUST be size-sorted (stwo's
/// lifted Merkle sorts each tree's columns by length, and the in-circuit verifier does NOT re-sort
/// the preprocessed tree). The sizes are DYNAMIC: `gate_pc_in_prog` is sized with the main trace
/// (`main_log_size = log_n_rows`) and `gate_prog_slot` with the program table — so when the main
/// trace grows past the range-check tables (RC_LOG_SIZE = 16) the sort order changes (pc_in_prog
/// moves after rc). A static order is only correct while main_log_size <= 16.
fn preprocessed_columns_sorted(
    main_log_size: u32,
    program_log_size: u32,
) -> Vec<(PreProcessedColumnId, u32)> {
    let q = N_QUBITS.ilog2();
    let mut cols = vec![
        (pp_id("gate_qdecode_q"), q),
        (pp_id("gate_qdecode_limb"), q),
        (pp_id("gate_qdecode_pos"), q),
        (pp_id("gate_qdecode_mask"), q),
        (pp_id("gate_prog_slot"), program_log_size),
        (pp_id("gate_pc_in_prog"), main_log_size),
        (pp_id("gate_rc_lo_pos"), RC_LOG_SIZE),
        (pp_id("gate_rc_lo_val"), RC_LOG_SIZE),
        (pp_id("gate_rc_hi_pos"), RC_LOG_SIZE),
        (pp_id("gate_rc_hi_val"), RC_LOG_SIZE),
    ];
    cols.sort_by_key(|&(_, s)| s); // stable: ties keep the listing order above
    cols
}

fn preprocessed_column_ids(main_log_size: u32, program_log_size: u32) -> Vec<PreProcessedColumnId> {
    preprocessed_columns_sorted(main_log_size, program_log_size)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
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

    fn prover_refs(&self) -> Vec<&dyn stwo::prover::ComponentProver<ProverBackend>> {
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.qdecode as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.rc_lo as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.rc_hi as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<ProverBackend>,
        ]
    }

    fn trace_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        // Use stwo's CANONICAL column sizes (not a plain component-concat). stwo's verifier
        // reindexes the PREPROCESSED tree GLOBALLY by each component's preprocessed_column_indices
        // (i.e. into preprocessed_column_ids() order), so the preprocessed sizes land in the
        // committed order — which the lifted Merkle commits sorted by size and the in-circuit
        // verifier (circuits_stark_verifier) does NOT re-sort. A naive concat would order the
        // preprocessed sizes by component instead, mismatching the committed tree ("Root mismatch").
        stwo::core::air::Components {
            components: self.component_refs(),
            n_preprocessed_columns: N_PREPROCESSED_COLS,
        }
        .column_log_sizes()
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
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(
        &preprocessed_column_ids(log_n_rows, program_log_size),
    );
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
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
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
            CircleEvaluation::<TraceBackend, _, BitReversedOrder>::new(
                domain,
                BaseColumn::from_simd(col_data),
            )
        })
        .collect()
}

fn col_from_values(values: &[u32]) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let log_size = values.len().ilog2();
    let mut col = Col::<TraceBackend, BaseField>::zeros(values.len());
    for (i, &v) in values.iter().enumerate() {
        col.set(i, BaseField::from_u32_unchecked(v));
    }
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Converts trace-gen (SimdBackend) columns to the prover backend at the `extend_evals` boundary.
///
/// * default / `gpu`: `ProverBackend == SimdBackend`, or obelyzk `GpuBackend` whose columns are
///   layout-identical to SimdBackend — a cheap rewrap (`CircleEvaluation::new(domain, values)`).
/// * `cuda`: `ProverBackend == CudaBackend` with device-resident `BaseFieldVec` columns; copy each
///   column's host values into a device column via `FromIterator<BaseField> for BaseFieldVec`
///   (`to_cpu()` is a no-op on SimdBackend host data, then `.collect()` uploads to the device).
#[cfg(not(feature = "cuda"))]
fn to_prover(
    cols: Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
) -> Vec<CircleEvaluation<ProverBackend, BaseField, BitReversedOrder>> {
    cols.into_iter()
        .map(|e| CircleEvaluation::new(e.domain, e.values))
        .collect()
}

#[cfg(feature = "cuda")]
fn to_prover(
    cols: Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
) -> Vec<CircleEvaluation<ProverBackend, BaseField, BitReversedOrder>> {
    cols.into_iter()
        .map(|e| {
            let domain = e.domain;
            let values: Col<ProverBackend, BaseField> = e.values.to_cpu().into_iter().collect();
            CircleEvaluation::new(domain, values)
        })
        .collect()
}

fn generate_qdecode_preprocessed(
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
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
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&idx.pos_col), col_from_values(&idx.val_col)]
}

/// Preprocessed pc_in_prog column for the main trace: pc mod n_gates on real
/// rows, 0 on padding (inert: padding has enabler 0).
fn generate_pc_in_prog_preprocessed(
    rows: &[Row],
    padded_rows: usize,
    n_gates: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
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
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    col_from_values(&prog.slot)
}

/// Program-table witness (multiplicity tree): op columns then multiplicity, in
/// the order ProgramTableEval reads them.
fn generate_program_witness(
    prog: &ProgramTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
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
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(counts)]
}

// ----------------------------------------------------------------------------
// Interaction traces
// ----------------------------------------------------------------------------

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
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(log_n_rows);

    // Entry combiners.
    let state_in = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        let mut v = Vec::with_capacity(GATE_REL_WIDTH);
        v.push(ptag(TAG_STATE));
        v.push(pack(lane, |r| r.shot_id));
        v.push(pack(lane, |r| r.pc));
        for j in 0..N_LIMBS {
            v.push(pack(lane, |r| r.in_limb[j]));
        }
        el.state.combine(&v)
    };
    let state_out = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        let mut v = Vec::with_capacity(GATE_REL_WIDTH);
        v.push(ptag(TAG_STATE));
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
            ptag(TAG_QDECODE),
            pack(lane, |r| sel(r).q),
            pack(lane, |r| sel(r).limb_idx),
            pack(lane, |r| sel(r).bit_pos),
            pack(lane, |r| sel(r).mask),
        ])
    };
    let rc_lo = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &ReadCols| -> PackedSecureField {
        el.rc_lo.combine(&[
            ptag(TAG_RC_LO),
            pack(lane, |r| sel(r).bit_pos),
            pack(lane, |r| sel(r).lo),
        ])
    };
    let rc_hi = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &ReadCols| -> PackedSecureField {
        el.rc_hi.combine(&[
            ptag(TAG_RC_HI),
            pack(lane, |r| sel(r).bit_pos),
            pack(lane, |r| sel(r).hi),
        ])
    };
    // Program use-side denominator. pc_in_prog = pc mod n_gates (preprocessed in
    // the AIR; recomputed here for the prover). opcode_scalar from the one-hot.
    let ng = n_gates as u32;
    let program = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.program.combine(&[
            ptag(TAG_PROGRAM),
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

    // Write one logup column for a pair of relation entries: fraction = m0/d0 + m1/d1.
    // PARALLEL over vec_rows: the per-row combine (35-element dot products) is the cost; computing
    // the (num, den) fractions in a rayon par_iter and feeding `col_from_par_iter` fans it over all
    // cores (the old sequential packed_rows loop ran this on one core — the trace-gen bottleneck).
    // Generic over the closure types (not &dyn) so the Sync bound holds without lifetime grief.
    #[allow(clippy::too_many_arguments)]
    fn write_pair_par<N0, D0, N1, D1>(
        gen: &mut LogupTraceGenerator,
        rows: &[Row],
        n_vec: usize,
        num0: N0,
        den0: D0,
        sign0: i32,
        num1: N1,
        den1: D1,
        sign1: i32,
    ) where
        N0: Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync,
        D0: Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync,
        N1: Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync,
        D1: Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync,
    {
        use rayon::prelude::*;
        let pad = Row::padding();
        let col_iter = (0..n_vec).into_par_iter().map(|vec_row| {
            let lane: [&Row; LANE_COUNT] =
                std::array::from_fn(|l| rows.get(vec_row * LANE_COUNT + l).unwrap_or(&pad));
            let m0 = PackedSecureField::from(num0(&lane));
            let m0 = if sign0 < 0 { -m0 } else { m0 };
            let d0 = den0(&lane);
            let m1 = PackedSecureField::from(num1(&lane));
            let m1 = if sign1 < 0 { -m1 } else { m1 };
            let d1 = den1(&lane);
            (m0 * d1 + m1 * d0, d0 * d1)
        });
        gen.col_from_par_iter(col_iter);
    }
    let n_vec = padded_rows / LANE_COUNT;
    let write_pair = |gen: &mut LogupTraceGenerator,
                      num0: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync),
                      den0: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync),
                      sign0: i32,
                      num1: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync),
                      den1: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync),
                      sign1: i32| {
        write_pair_par(gen, rows, n_vec, num0, den0, sign0, num1, den1, sign1);
    };

    // pair0: state_in (+enabler), state_out (-enabler).
    write_pair(
        &mut gen,
        &enabler,
        &state_in,
        1,
        &enabler,
        &state_out,
        -1,
    );
    // pair1: qdecode target (+enabler), qdecode ctrl_a (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| qdecode(l, sel_t),
        1,
        &a_active,
        &|l| qdecode(l, sel_a),
        1,
    );
    // pair2: qdecode ctrl_b (+b_active), rc_lo target (+enabler).
    write_pair(
        &mut gen,
        &b_active,
        &|l| qdecode(l, sel_b),
        1,
        &enabler,
        &|l| rc_lo(l, sel_t),
        1,
    );
    // pair3: rc_hi target (+enabler), rc_lo ctrl_a (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| rc_hi(l, sel_t),
        1,
        &a_active,
        &|l| rc_lo(l, sel_a),
        1,
    );
    // pair4: rc_hi ctrl_a (+a_active), rc_lo ctrl_b (+b_active).
    write_pair(
        &mut gen,
        &a_active,
        &|l| rc_hi(l, sel_a),
        1,
        &b_active,
        &|l| rc_lo(l, sel_b),
        1,
    );
    // pair5: rc_hi ctrl_b (+b_active), program (+enabler).
    write_pair(
        &mut gen,
        &b_active,
        &|l| rc_hi(l, sel_b),
        1,
        &enabler,
        &program,
        1,
    );

    // LogupTraceGenerator already emits SimdBackend (== TraceBackend) columns; conversion to the
    // prover backend happens later via `to_prover` at the `extend_evals` boundary.
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
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
    main_interaction: &[CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>],
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
    preprocessed: &[CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>],
    multiplicity: &[CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>],
    interaction: &[CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>],
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
fn gen_table_interaction(
    counts: &[u32],
    log_size: u32,
    combine_row: impl Fn(usize) -> PackedSecureField,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
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
    // LogupTraceGenerator already emits SimdBackend (== TraceBackend) columns; conversion to the
    // prover backend happens later via `to_prover` at the `extend_evals` boundary.
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

// ----------------------------------------------------------------------------
// Public boundary sums
// ----------------------------------------------------------------------------

fn public_boundary_sum(
    cases: &[TestCase],
    n_gates: usize,
    k: usize,
    state: &GateRel,
) -> Result<SecureField> {
    let mut sum = SecureField::zero();
    let total_pc = (n_gates * k) as u32;
    // Tagged state tuple: [TAG_STATE, shot_id, pc, limbs...].
    let tagged = |t: &[BaseField; STATE_WIDTH]| -> Vec<BaseField> {
        let mut v = Vec::with_capacity(GATE_REL_WIDTH);
        v.push(BaseField::from_u32_unchecked(TAG_STATE));
        v.extend_from_slice(t);
        v
    };
    for (shot_id, case) in cases.iter().enumerate() {
        let x = hex::decode(&case.x_hex)?;
        let y = hex::decode(&case.y_hex)?;
        let x_limbs = state_to_limbs(&x);
        let y_limbs = state_to_limbs(&y);
        let initial = state_tuple(shot_id as u32, 0, &x_limbs);
        let final_ = state_tuple(shot_id as u32, total_pc, &y_limbs);
        let ci: SecureField = state.combine(&tagged(&initial));
        let cf: SecureField = state.combine(&tagged(&final_));
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
    tag: u32,
    row_tuple: impl Fn(usize) -> Vec<BaseField>,
) -> SecureField {
    let mut sum = SecureField::zero();
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let mut tuple = vec![BaseField::from_u32_unchecked(tag)];
        tuple.extend(row_tuple(i));
        let denom: SecureField = elements.combine(&tuple);
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    -sum
}

// ----------------------------------------------------------------------------
// GPU trace-gen inputs (device path)
// ----------------------------------------------------------------------------

/// Flatten the host-side inputs the K1/K4 device kernels consume: the gate list
/// (opcode, target, ctrl_a, ctrl_b per gate), each shot's initial 32-limb state,
/// and the RcIndex lo/hi offsets. Mirrors the prep in `gpu_tracegen::k1_byte_identity`.
#[cfg(feature = "cuda")]
fn gpu_flat_inputs(
    gates: &[Gate],
    cases: &[TestCase],
    rc_lo_index: &RcIndex,
    rc_hi_index: &RcIndex,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let mut gates_flat = Vec::with_capacity(gates.len() * 4);
    for g in gates {
        gates_flat.push(g.opcode as u32);
        gates_flat.push(g.target as u32);
        gates_flat.push(g.ctrl_a as u32);
        gates_flat.push(g.ctrl_b as u32);
    }
    let mut x_states = Vec::with_capacity(cases.len() * N_LIMBS);
    for c in cases {
        let bytes = hex::decode(&c.x_hex).context("decoding x_hex for GPU trace-gen")?;
        x_states.extend_from_slice(&state_to_limbs(&bytes));
    }
    let off_lo: Vec<u32> = (0..LIMB_BITS).map(|p| rc_lo_index.offset[p] as u32).collect();
    let off_hi: Vec<u32> = (0..LIMB_BITS).map(|p| rc_hi_index.offset[p] as u32).collect();
    Ok((gates_flat, x_states, off_lo, off_hi))
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

    // GPU soundness gates (then exit): GATE_AIR_GPU_TEST selects which kernel to validate
    // byte-identically against the CPU reference. "k4" → K4 LogUp interaction; anything else
    // (e.g. "1"/"k1") → K1 main trace-gen + histograms.
    #[cfg(feature = "gpu-cuda")]
    if let Ok(which) = std::env::var("GATE_AIR_GPU_TEST") {
        if which == "k4" {
            gpu_tracegen::k4_byte_identity(&gates, cases, k, &rc_lo_index, &rc_hi_index)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else {
            gpu_tracegen::k1_byte_identity(&gates, cases, k, &rc_lo_index, &rc_hi_index)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        return Ok(());
    }

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
    let max_log_size = log_n_rows.max(RC_LOG_SIZE);
    // SECURE base config (~96-bit) instead of PcsConfig::default() (which is a 13-bit TOY: blowup 1,
    // n_queries 3). leaf_pcs_config sets n_queries/pow_bits/fold_step=4 + lifting = trace+blowup so
    // the base proof passes the privacy-verifier security test. The in-circuit verifier replays this
    // exact config, so its verification circuit now reflects the real (secure) decommitment cost.
    // Base blowup is a sweep knob (env BASE_BLOWUP overrides the default const).
    let base_blowup: u32 = std::env::var("BASE_BLOWUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(BASE_LOG_BLOWUP_FACTOR);
    let config = leaf::leaf_pcs_config(max_log_size, base_blowup);
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let prover_channel = &mut Blake2sM31Channel::default();
    // Canonical transcript (matches circuits_stark_verifier::verify replay): salt, then config.
    let channel_salt = 0u32;
    prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
    config.mix_into(prover_channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
    // Memory-footprint fix (candidate 1): DROP stored polynomial coefficients. With store=false the
    // prover takes the barycentric OODS path (build_weights_hash_map + CudaBackend::barycentric_
    // eval_at_point, byte-identical to the coeffs path) instead of keeping every committed column's
    // coefficients device-resident (~14GB at 2^24). The ExtendedStarkProof aux is built from Merkle/
    // FRI data and the OODS sampled_values (not coeffs), so the proof — and the in-circuit verifier's
    // input — is unchanged. fp byte-identity gate confirms this.
    // commitment_scheme.set_store_polynomials_coefficients();  // disabled: barycentric OODS path

    // Tree 0: preprocessed. The committed order MUST equal preprocessed_column_ids(...) AND be
    // ascending by size (the lifted Merkle commits columns sorted by length; the in-circuit verifier
    // does NOT re-sort this tree). Build the columns in the SAME canonical listing order as
    // preprocessed_columns_sorted, tag each with its size, then STABLE-sort by size — so for ANY
    // main_log_size the committed order matches the ids (e.g. pc_in_prog moves after rc when main>16).
    let t_phase = Instant::now();
    let mut tree_builder = commitment_scheme.tree_builder();
    let qsz = N_QUBITS.ilog2();
    let mut tagged: Vec<(u32, CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>)> =
        generate_qdecode_preprocessed().into_iter().map(|c| (qsz, c)).collect();
    tagged.push((program.log_size, generate_prog_slot_preprocessed(&program)));
    tagged.push((log_n_rows, generate_pc_in_prog_preprocessed(&rows, padded_rows, n_gates)));
    tagged.extend(generate_rc_preprocessed(&rc_lo_index).into_iter().map(|c| (RC_LOG_SIZE, c)));
    tagged.extend(generate_rc_preprocessed(&rc_hi_index).into_iter().map(|c| (RC_LOG_SIZE, c)));
    tagged.sort_by_key(|(s, _)| *s); // stable: identical key+listing order as preprocessed_columns_sorted
    let pp: Vec<_> = tagged.into_iter().map(|(_, c)| c).collect();
    tree_builder.extend_evals(to_prover(pp));
    tree_builder.commit(prover_channel);
    eprintln!("gate-air: [phase] preprocessed gen+commit {:.3}s", t_phase.elapsed().as_secs_f64());

    // Public claim (empty for gate_air; the boundary is reconstructed by the verifier).
    let public_claim = pack_public_claim(&[]);
    prover_channel.mix_felts(&public_claim);

    // Under `cuda`, the dominant trees (main + interaction) are generated ON the GPU and handed to
    // the CudaBackend commit device-to-device (no host upload), UNLESS GATE_AIR_CPU_TRACEGEN=1 forces
    // the legacy CPU-build + upload path. The small columns (multiplicity / program witness / table
    // interactions / preprocessed) always stay on the CPU-generate + upload path.
    #[cfg(feature = "cuda")]
    let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

    // Tree 1: main trace + table multiplicities + program witness (op cols+mult).
    let t_phase = Instant::now();
    let small_main = {
        let mut v = generate_multiplicity_trace(&counts.qdecode);
        v.extend(generate_multiplicity_trace(&counts.rc_lo));
        v.extend(generate_multiplicity_trace(&counts.rc_hi));
        v.extend(generate_program_witness(&program));
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    #[cfg(feature = "cuda")]
    if gpu_tracegen {
        // Device K1: 191 main columns generated on the GPU, fed in as device-resident BaseFieldVecs.
        let (gates_flat, x_states, off_lo, off_hi) =
            gpu_flat_inputs(&gates, cases, &rc_lo_index, &rc_hi_index)?;
        let (mut main_dev, _qd, _lo, _hi) = gpu_tracegen::gpu_gen_main_trace_device(
            &gates_flat, &x_states, &off_lo, &off_hi,
            k as u32, n_gates as u32, samples as u32, padded_rows, log_n_rows,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        main_dev.extend(to_prover(small_main));
        eprintln!("gate-air: [phase] main_trace witness gen (GPU K1) {:.3}s", t_phase.elapsed().as_secs_f64());
        tree_builder.extend_evals(main_dev);
    } else {
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        eprintln!("gate-air: [phase] main_trace witness gen (CPU) {:.3}s", t_phase.elapsed().as_secs_f64());
        tree_builder.extend_evals(to_prover(main_trace));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        eprintln!("gate-air: [phase] main_trace witness gen {:.3}s", t_phase.elapsed().as_secs_f64());
        tree_builder.extend_evals(to_prover(main_trace));
    }
    let t_phase = Instant::now();
    tree_builder.commit(prover_channel);
    eprintln!("gate-air: [phase] tree1 commit (NTT+Merkle) {:.3}s", t_phase.elapsed().as_secs_f64());

    // Interaction-trace PoW grind, then mix the nonce (canonical transcript).
    let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
    prover_channel.mix_u64(interaction_pow_nonce);

    // Draw relation elements.
    let elements = LookupElements::draw(prover_channel);

    // Install gate_air's drawn LogUp challenges for the GPU constraint kernel (gate_air-specific
    // hook: the (z, alpha) live inside the opaque GateEval, which the generic ComponentProver can't
    // reach). No-op unless the kernel gate (CUDA_GPU_CONSTRAINTS=1 + gate_air main) fires.
    #[cfg(feature = "cuda")]
    {
        // Install the downstream gate_air GPU constraint kernel into the generic CudaBackend prover,
        // then thread the drawn (z, alpha) challenges to it.
        gate_air_cuda_kernel::register();
        let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.state);
        gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
    }

    // Interaction traces.
    let t_phase = Instant::now();
    // Device K4 (under `cuda`, unless GATE_AIR_CPU_TRACEGEN): the 24 main-interaction columns are
    // generated on the GPU using the REAL drawn `elements` and handed to the commit device-resident
    // (no upload); `claimed_sum` becomes `main_sum`. `main_interaction` (CPU SimdBackend cols) is
    // only materialized on the CPU path or when GATE_AIR_ASSERT needs it for `assert_main_constraints`.
    #[cfg(feature = "cuda")]
    let main_interaction_device = if gpu_tracegen {
        let (gates_flat, x_states, off_lo, off_hi) =
            gpu_flat_inputs(&gates, cases, &rc_lo_index, &rc_hi_index)?;
        let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
            &gates_flat, &x_states, &off_lo, &off_hi,
            k as u32, n_gates as u32, samples as u32, padded_rows, log_n_rows, &elements,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Some((cols, claimed))
    } else {
        None
    };
    #[cfg(feature = "cuda")]
    let (main_interaction, main_sum) = if let Some((_, claimed)) = &main_interaction_device {
        // GPU path: skip CPU interaction gen (the dominant cost); claimed_sum == CPU main_sum.
        // Only build CPU cols if GATE_AIR_ASSERT needs them for the on-trace constraint check.
        let cpu_cols = if std::env::var("GATE_AIR_ASSERT").is_ok() {
            gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements).0
        } else {
            Vec::new()
        };
        (cpu_cols, *claimed)
    } else {
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements)
    };
    #[cfg(not(feature = "cuda"))]
    let (main_interaction, main_sum) =
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
    let (qdecode_interaction, qdecode_sum) = {
        let el = elements.qdecode.clone();
        let q: Vec<u32> = (0..N_QUBITS as u32).collect();
        gen_table_interaction(&counts.qdecode, N_QUBITS.ilog2(), |vec_row| {
            el.combine(&[
                ptag(TAG_QDECODE),
                pack_seq(&q, vec_row),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).0),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).1),
                pack_decode(vec_row, |qi| qubit_decode(qi as u16).2),
            ])
        })
    };
    let (rc_lo_interaction, rc_lo_sum) = {
        let el = elements.rc_lo.clone();
        gen_table_interaction(&counts.rc_lo, RC_LOG_SIZE, |vec_row| {
            el.combine(&[
                ptag(TAG_RC_LO),
                pack_seq(&rc_lo_index.pos_col, vec_row),
                pack_seq(&rc_lo_index.val_col, vec_row),
            ])
        })
    };
    let (rc_hi_interaction, rc_hi_sum) = {
        let el = elements.rc_hi.clone();
        gen_table_interaction(&counts.rc_hi, RC_LOG_SIZE, |vec_row| {
            el.combine(&[
                ptag(TAG_RC_HI),
                pack_seq(&rc_hi_index.pos_col, vec_row),
                pack_seq(&rc_hi_index.val_col, vec_row),
            ])
        })
    };
    let (program_interaction, program_sum) = {
        let el = elements.program.clone();
        gen_table_interaction(&program.multiplicity, program.log_size, |vec_row| {
            el.combine(&[
                ptag(TAG_PROGRAM),
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
    let qdecode_expected = table_public_sum(&counts.qdecode, &elements.qdecode, TAG_QDECODE, |qi| {
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
    let rc_lo_expected = table_public_sum(&counts.rc_lo, &elements.rc_lo, TAG_RC_LO, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_lo_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_lo_index.val_col[i]),
        ]
    });
    if rc_lo_sum != rc_lo_expected {
        bail!("rc_lo claimed sum mismatch");
    }
    let rc_hi_expected = table_public_sum(&counts.rc_hi, &elements.rc_hi, TAG_RC_HI, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_hi_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_hi_index.val_col[i]),
        ]
    });
    if rc_hi_sum != rc_hi_expected {
        bail!("rc_hi claimed sum mismatch");
    }
    let program_expected = table_public_sum(&program.multiplicity, &elements.program, TAG_PROGRAM, |i| {
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

    eprintln!("gate-air: [phase] interaction witness gen+sumcheck {:.3}s", t_phase.elapsed().as_secs_f64());
    // Tree 2: interaction (same component order as the claimed sums). The 24 main-interaction columns
    // come first, then the four small table interactions (always CPU-built + uploaded). Under the GPU
    // path the main columns are already device-resident; the CPU columns are converted via `to_prover`.
    let t_phase = Instant::now();
    let small_interaction = {
        let mut v = qdecode_interaction;
        v.extend(rc_lo_interaction);
        v.extend(rc_hi_interaction);
        v.extend(program_interaction);
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    #[cfg(feature = "cuda")]
    if let Some((main_dev, _)) = main_interaction_device {
        let mut interaction = main_dev;
        interaction.extend(to_prover(small_interaction));
        tree_builder.extend_evals(interaction);
    } else {
        let mut interaction = main_interaction;
        interaction.extend(small_interaction);
        tree_builder.extend_evals(to_prover(interaction));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut interaction = main_interaction;
        interaction.extend(small_interaction);
        tree_builder.extend_evals(to_prover(interaction));
    }
    tree_builder.commit(prover_channel);
    eprintln!("gate-air: [phase] tree2 commit (NTT+Merkle) {:.3}s", t_phase.elapsed().as_secs_f64());

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
    // prove_ex (vs prove) yields the ExtendedStarkProof (proof + aux) the in-circuit verifier needs;
    // M2a validates it via the native verify below (`extended.proof`). M2d will keep `extended`
    // whole + align the Fiat-Shamir transcript (salt / interaction-PoW / public-claim) to the
    // circuits_stark_verifier replay.
    let extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(
        &prover_refs,
        prover_channel,
        commitment_scheme,
        false,
    )?;
    let prove_elapsed = prove_start.elapsed();

    // ---- Full-proof byte-identity fingerprint (read-only), gated by GATE_AIR_PROOF_HASH ----
    // Deterministic SHA-256 over the serde-serialized ExtendedStarkProof (commitments,
    // sampled_values, decommitments, FRI, proof_of_work, claimed sums via sampled_values + aux).
    // The CPU/SimdBackend run is the golden oracle; a `--features cuda` run on the same
    // fixture+samples must print the SAME hex. See P5_GPU_CONSTRAINT_SCOPE.md Deliverable 2 §2.1.
    if std::env::var("GATE_AIR_PROOF_HASH").is_ok() {
        emit_proof_fingerprint(&extended);
    }

    // ---- In-circuit verification (Design A, Milestone 2), gated by GATE_AIR_INCIRCUIT ----
    if std::env::var("GATE_AIR_INCIRCUIT").is_ok() {
        use circuit_statement::{GateAirStatement, gate_air_components};
        use circuits::blake::ReducedHashValue;
        use circuits::context::{Context, TraceContext};
        use circuits::ivalue::NoValue;
        use circuits::ops::Guess;
        use circuits_stark_verifier::proof::{ProofConfig, empty_proof};
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use circuits_stark_verifier::verify::verify as circuit_verify;

        let n_pp = N_PREPROCESSED_COLS;
        let cfg =
            ProofConfig::new(&gate_air_components::<NoValue>(), n_pp, &config, INTERACTION_POW_BITS);
        let pp_root: ReducedHashValue<SecureField> = extended.proof.commitments[0].into();
        let mut boundary = Vec::with_capacity(cases.len());
        for case in cases {
            let x = state_to_limbs(&hex::decode(&case.x_hex)?);
            let y = state_to_limbs(&hex::decode(&case.y_hex)?);
            boundary.push((x, y));
        }
        let total_pc = (n_gates * k) as u32;
        let claim: Vec<SecureField> = vec![main_sum, qdecode_sum, rc_lo_sum, rc_hi_sum, program_sum];

        // NoValue circuit shape (the reference the real assignment is checked against).
        let novalue_circuit = {
            let empty = empty_proof(&cfg);
            let mut nv = Context::<NoValue>::default();
            let pv = empty.guess(&mut nv);
            let stmt = GateAirStatement::<NoValue>::new(
                &mut nv,
                log_n_rows,
                program.log_size,
                pp_root.clone(),
                boundary.clone(),
                total_pc,
            );
            circuit_verify(&mut nv, &pv, &cfg, &stmt);
            nv.finalize(false).context.circuit
        };
        // Build the real assignment and check it against the NoValue shape.
        let mut ctx = TraceContext::default();
        let circuit_proof =
            proof_from_stark_proof(&extended, &cfg, claim, interaction_pow_nonce, channel_salt);
        let pv = circuit_proof.guess(&mut ctx);
        let stmt = GateAirStatement::new(
            &mut ctx,
            log_n_rows,
            program.log_size,
            pp_root,
            boundary,
            total_pc,
        );
        circuit_verify(&mut ctx, &pv, &cfg, &stmt);
        let ctx = ctx.finalize(true);
        novalue_circuit.check(ctx.values()).expect("gate-air: in-circuit verify FAILED");
        eprintln!("gate-air: in-circuit verify OK");
    }

    // ---- Multiverifier-tree integration (Milestone 3), gated by GATE_AIR_FOLD ----
    // Prove the gate_air verification circuit as a foldable leaf, ONE PER SHARD, and fold the N
    // distinct leaves into a root.
    //
    // Sharding (Phase 0): the `samples` shots are partitioned into equal-sized shards of
    // `GATE_AIR_SHARD_SHOTS` shots (default 2). Each shard gets its OWN distinct base proof (its
    // own shots' trace) and its OWN distinct leaf (its shots' boundary outputs). N_shards is then
    // derived from the shot count (ceil(samples / shots_per_shard)). For backward compatibility
    // GATE_AIR_FOLD, if set to a value LARGER than the derived shard count, raises the leaf count
    // to that many shards by REUSING the equal-sized partition cyclically only when samples are
    // exhausted is NOT done — instead the derived N_shards is authoritative and GATE_AIR_FOLD's
    // presence merely enables the path. Its numeric value is ignored for partitioning; the shard
    // count is `ceil(samples / shots_per_shard)`. (Documented knob-semantics change.)
    //
    // EQUAL-SHAPE REQUIREMENT: every leaf shares one `AggregateConfig` (one trusted
    // `leaf_preprocessed_root` + target padding sizes). The leaf circuit shape depends on
    // `boundary.len()` (per-shot loop in `public_logup_sum` + the output-hash preimage),
    // `main_log_size`, `program_log_size` and `total_pc`. `main_log_size`/`program_log_size`/
    // `total_pc` are shot-count-independent (same program, same k); only `boundary.len()` varies.
    // So ALL shards must hold exactly `shots_per_shard` shots. A ragged final shard (fewer real
    // shots) is PADDED by repeating its last real shot up to `shots_per_shard`, so it shares the
    // shape. The padding shots are genuine, independently-verified (x->y) executions (a duplicate
    // of a real shot), so the proof stays sound; they only inflate that shard's output binding by
    // re-committing a shot that is already committed.
    //
    // INTER-SHARD BINDING / H_i ENCODING (project item M3c): the iadd256 shots are INDEPENDENT —
    // each shot is its own (x_i -> y_i) execution of the SAME hidden program; there is no
    // shot->shot (or shard->shard) data dependency (the per-shot boundary in `public_boundary_sum`
    // / `GateAirStatement::public_logup_sum` sums N independent source/sink pairs, and
    // program-consistency forces ONE shared program across all of them). Therefore the root proves
    // the UNION of independent shard executions and NO inter-shard binding is needed beyond the
    // existing per-leaf `output_values = blake(preprocessed_root || {x,y per shard shot})`. The
    // root aggregates these distinct per-shard output hashes (verified below via rv.leaf_outputs).
    // The remaining M3c refinement (fold the program commitment H_P into each H_i) is a binding
    // STRENGTHENING, not a correctness fix for sharding, and is left for the code owner.
    if std::env::var("GATE_AIR_FOLD").is_ok() {
        use circuit_statement::gate_air_components;
        use circuits::blake::ReducedHashValue;
        use circuits::ivalue::NoValue;
        use circuits_stark_verifier::proof::ProofConfig;
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use leaf::{GateAirLeafParams, derive_aggregate_config, leaf_pcs_config, prove_gate_air_leaf};
        use recursive_aggregate::{
            AggregateOutput, PoolSet, TreeProof, ZkBlind, prove_root_verification,
            recursive_aggregate_prove, recursive_aggregate_prove_streaming,
        };
        use stwo::core::proof::ExtendedStarkProof;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleHasher;

        // The base-proof tuple `prove_base_shard` returns. Named so the pipeline producer can send
        // it over a channel; `prove_ex` yields `ExtendedStarkProof<MC::H>` with
        // `MC::H = Blake2sM31MerkleHasher`, so this is backend-independent (cuda vs simd).
        type BaseShardOutput = (
            ExtendedStarkProof<Blake2sM31MerkleHasher>,
            Vec<SecureField>,
            u64,
            u32,
            u32,
            u32,
            Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
            u32,
        );

        const LOG_BLOWUP_FACTOR: u32 = 3;

        // Shard partition: equal-sized shards of `shots_per_shard` shots; ragged final shard is
        // padded (below) so all shards share the leaf circuit shape.
        let shots_per_shard: usize = std::env::var("GATE_AIR_SHARD_SHOTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2)
            .min(samples);
        let n_shards = samples.div_ceil(shots_per_shard);
        eprintln!(
            "gate-air: sharding {samples} shots into {n_shards} shard(s) of {shots_per_shard} shot(s) each \
             (final shard padded by shot-repeat if ragged)"
        );

        // Build each shard's equal-sized `cases` slice. The final shard repeats its last real shot
        // up to `shots_per_shard` so it shares the shape; padding shots are extra independent (x->y)
        // executions (a duplicate), still sound.
        let shard_case_sets: Vec<Vec<TestCase>> = (0..n_shards)
            .map(|s| {
                let start = s * shots_per_shard;
                let end = (start + shots_per_shard).min(samples);
                let mut v: Vec<TestCase> = cases[start..end].to_vec();
                while v.len() < shots_per_shard {
                    v.push(cases[end - 1].clone()); // repeat last real shot to equalize shape
                }
                v
            })
            .collect();

        // Per-shard base proof: same trace-gen + commit + prove_ex pipeline as the single proof
        // above, but over this shard's shots. Returns the distinct ExtendedStarkProof plus the
        // claim / nonce / log_n_rows the leaf needs. The closure mirrors the inline body verbatim;
        // type inference keeps the ExtendedStarkProof generic (cuda vs simd) implicit.
        let prove_base_shard = |shard_cases: &[TestCase]| -> Result<_> {
            let shard_samples = shard_cases.len();
            let program = build_program_table(&gates, shard_samples, k);
            let (rows, counts) = build_rows(&gates, shard_cases, k, &rc_lo_index, &rc_hi_index)?;
            let real_rows = rows.len();
            let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
            let log_n_rows = padded_rows.ilog2();
            let max_log_size = log_n_rows.max(RC_LOG_SIZE);
            let base_blowup: u32 = std::env::var("BASE_BLOWUP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(BASE_LOG_BLOWUP_FACTOR);
            let config = leaf::leaf_pcs_config(max_log_size, base_blowup);
            let twiddles = ProverBackend::precompute_twiddles(
                CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );
            let prover_channel = &mut Blake2sM31Channel::default();
            let channel_salt = 0u32;
            prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
            config.mix_into(prover_channel);
            let mut commitment_scheme =
                CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
                    config, &twiddles,
                );
            // commitment_scheme.set_store_polynomials_coefficients();  // disabled: barycentric OODS path

            // Tree 0: preprocessed (canonical listing order, then stable-sort by size).
            let mut tree_builder = commitment_scheme.tree_builder();
            let qsz = N_QUBITS.ilog2();
            let mut tagged: Vec<(u32, CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>)> =
                generate_qdecode_preprocessed().into_iter().map(|c| (qsz, c)).collect();
            tagged.push((program.log_size, generate_prog_slot_preprocessed(&program)));
            tagged.push((log_n_rows, generate_pc_in_prog_preprocessed(&rows, padded_rows, n_gates)));
            tagged.extend(generate_rc_preprocessed(&rc_lo_index).into_iter().map(|c| (RC_LOG_SIZE, c)));
            tagged.extend(generate_rc_preprocessed(&rc_hi_index).into_iter().map(|c| (RC_LOG_SIZE, c)));
            tagged.sort_by_key(|(s, _)| *s);
            let pp: Vec<_> = tagged.into_iter().map(|(_, c)| c).collect();
            tree_builder.extend_evals(to_prover(pp));
            tree_builder.commit(prover_channel);

            let public_claim = pack_public_claim(&[]);
            prover_channel.mix_felts(&public_claim);

            #[cfg(feature = "cuda")]
            let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

            // Tree 1: main trace + table multiplicities + program witness.
            let small_main = {
                let mut v = generate_multiplicity_trace(&counts.qdecode);
                v.extend(generate_multiplicity_trace(&counts.rc_lo));
                v.extend(generate_multiplicity_trace(&counts.rc_hi));
                v.extend(generate_program_witness(&program));
                v
            };
            let mut tree_builder = commitment_scheme.tree_builder();
            #[cfg(feature = "cuda")]
            if gpu_tracegen {
                let (gates_flat, x_states, off_lo, off_hi) =
                    gpu_flat_inputs(&gates, shard_cases, &rc_lo_index, &rc_hi_index)?;
                let (mut main_dev, _qd, _lo, _hi) = gpu_tracegen::gpu_gen_main_trace_device(
                    &gates_flat, &x_states, &off_lo, &off_hi,
                    k as u32, n_gates as u32, shard_samples as u32, padded_rows, log_n_rows,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                main_dev.extend(to_prover(small_main));
                tree_builder.extend_evals(main_dev);
            } else {
                let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
                main_trace.extend(small_main);
                tree_builder.extend_evals(to_prover(main_trace));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
                main_trace.extend(small_main);
                tree_builder.extend_evals(to_prover(main_trace));
            }
            tree_builder.commit(prover_channel);

            let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
            prover_channel.mix_u64(interaction_pow_nonce);
            let elements = LookupElements::draw(prover_channel);

            #[cfg(feature = "cuda")]
            {
                gate_air_cuda_kernel::register();
                let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.state);
                gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
            }

            // Interaction traces.
            #[cfg(feature = "cuda")]
            let main_interaction_device = if gpu_tracegen {
                let (gates_flat, x_states, off_lo, off_hi) =
                    gpu_flat_inputs(&gates, shard_cases, &rc_lo_index, &rc_hi_index)?;
                let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
                    &gates_flat, &x_states, &off_lo, &off_hi,
                    k as u32, n_gates as u32, shard_samples as u32, padded_rows, log_n_rows, &elements,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                Some((cols, claimed))
            } else {
                None
            };
            #[cfg(feature = "cuda")]
            let (main_interaction, main_sum) = if let Some((_, claimed)) = &main_interaction_device {
                (Vec::new(), *claimed)
            } else {
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements)
            };
            #[cfg(not(feature = "cuda"))]
            let (main_interaction, main_sum) =
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
            let (qdecode_interaction, qdecode_sum) = {
                let el = elements.qdecode.clone();
                let q: Vec<u32> = (0..N_QUBITS as u32).collect();
                gen_table_interaction(&counts.qdecode, N_QUBITS.ilog2(), |vec_row| {
                    el.combine(&[
                        ptag(TAG_QDECODE),
                        pack_seq(&q, vec_row),
                        pack_decode(vec_row, |qi| qubit_decode(qi as u16).0),
                        pack_decode(vec_row, |qi| qubit_decode(qi as u16).1),
                        pack_decode(vec_row, |qi| qubit_decode(qi as u16).2),
                    ])
                })
            };
            let (rc_lo_interaction, rc_lo_sum) = {
                let el = elements.rc_lo.clone();
                gen_table_interaction(&counts.rc_lo, RC_LOG_SIZE, |vec_row| {
                    el.combine(&[
                        ptag(TAG_RC_LO),
                        pack_seq(&rc_lo_index.pos_col, vec_row),
                        pack_seq(&rc_lo_index.val_col, vec_row),
                    ])
                })
            };
            let (rc_hi_interaction, rc_hi_sum) = {
                let el = elements.rc_hi.clone();
                gen_table_interaction(&counts.rc_hi, RC_LOG_SIZE, |vec_row| {
                    el.combine(&[
                        ptag(TAG_RC_HI),
                        pack_seq(&rc_hi_index.pos_col, vec_row),
                        pack_seq(&rc_hi_index.val_col, vec_row),
                    ])
                })
            };
            let (program_interaction, program_sum) = {
                let el = elements.program.clone();
                gen_table_interaction(&program.multiplicity, program.log_size, |vec_row| {
                    el.combine(&[
                        ptag(TAG_PROGRAM),
                        pack_seq(&program.slot, vec_row),
                        pack_seq(&program.opcode_scalar, vec_row),
                        pack_seq(&program.target, vec_row),
                        pack_seq(&program.ctrl_a, vec_row),
                        pack_seq(&program.ctrl_b, vec_row),
                    ])
                })
            };

            // Cross-check claimed sums against this shard's public boundary + table sums.
            let boundary_sum = public_boundary_sum(shard_cases, n_gates, k, &elements.state)?;
            if main_sum + qdecode_sum + rc_lo_sum + rc_hi_sum + program_sum != boundary_sum {
                bail!("shard main claimed sum != public boundary sum");
            }

            let claimed_sums = vec![main_sum, qdecode_sum, rc_lo_sum, rc_hi_sum, program_sum];
            prover_channel.mix_felts(&claimed_sums);

            // Tree 2: interaction (same component order as the claimed sums).
            let small_interaction = {
                let mut v = qdecode_interaction;
                v.extend(rc_lo_interaction);
                v.extend(rc_hi_interaction);
                v.extend(program_interaction);
                v
            };
            let mut tree_builder = commitment_scheme.tree_builder();
            #[cfg(feature = "cuda")]
            if let Some((main_dev, _)) = main_interaction_device {
                let mut interaction = main_dev;
                interaction.extend(to_prover(small_interaction));
                tree_builder.extend_evals(interaction);
            } else {
                let mut interaction = main_interaction;
                interaction.extend(small_interaction);
                tree_builder.extend_evals(to_prover(interaction));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mut interaction = main_interaction;
                interaction.extend(small_interaction);
                tree_builder.extend_evals(to_prover(interaction));
            }
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
            let prover_refs = components.prover_refs();
            let extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(
                &prover_refs,
                prover_channel,
                commitment_scheme,
                false,
            )?;

            // Per-shard boundary (x->y per shard shot) for the leaf's GateAirStatement + output hash.
            let mut shard_boundary = Vec::with_capacity(shard_cases.len());
            for case in shard_cases {
                let x = state_to_limbs(&hex::decode(&case.x_hex)?);
                let y = state_to_limbs(&hex::decode(&case.y_hex)?);
                shard_boundary.push((x, y));
            }
            let claim: Vec<SecureField> =
                vec![main_sum, qdecode_sum, rc_lo_sum, rc_hi_sum, program_sum];
            Ok((
                extended,
                claim,
                interaction_pow_nonce,
                channel_salt,
                log_n_rows,
                program.log_size,
                shard_boundary,
                (n_gates * k) as u32,
            ))
        };

        let n_pp = N_PREPROCESSED_COLS;

        // PIPELINE opt-in: with GATE_AIR_PIPELINE set AND >1 shard, overlap GPU base-proving
        // (producer) with CPU leaf-wrap + streaming fold (consumer). The producer proves shards
        // 1..n_shards on a dedicated thread while the consumer wraps + folds in shard order; only
        // shard 0's base is proved eagerly here (it's needed to derive `agg`). With the flag unset
        // (default) the existing sequential path below runs UNCHANGED.
        //
        // SOUNDNESS GATE (pending, on-box, NOT run here — laptop only): the streaming path must
        // yield a recursion_fingerprint BYTE-IDENTICAL to the sequential path for the same fixture
        // (e.g. k1-n4 samples=4 GATE_AIR_SHARD_SHOTS=2, GATE_AIR_PIPELINE set vs unset). That
        // one-flag diff is the trust gate before this path is used in anger.
        let pipeline = std::env::var("GATE_AIR_PIPELINE").is_ok() && n_shards > 1;

        // Prove the per-shard base proof(s) (each is itself heavy / GPU-bound). In the sequential
        // path, prove all up front. In the pipeline path, prove ONLY shard 0 here (the rest are
        // produced concurrently by the producer thread, below).
        let t = Instant::now();
        let mut shard_bases = Vec::with_capacity(n_shards);
        if pipeline {
            eprintln!("gate-air: proving shard 0 base proof eagerly (pipeline) ...");
            shard_bases.push(prove_base_shard(&shard_case_sets[0])?);
        } else {
            eprintln!("gate-air: proving {n_shards} distinct per-shard base proof(s) ...");
            for (s, shard_cases) in shard_case_sets.iter().enumerate() {
                eprintln!("gate-air: base proof for shard {s} ({} shots) ...", shard_cases.len());
                shard_bases.push(prove_base_shard(shard_cases)?);
            }
        }
        eprintln!(
            "gate-air: base proof(s) (so far) in {:.1}s",
            t.elapsed().as_secs_f64()
        );

        // All shards share the SAME circuit shape (equal shot count, same program -> same
        // preprocessed root / target sizes). Derive ONE AggregateConfig from shard 0's params and
        // reuse it for every leaf (the "one trusted leaf_preprocessed_root for all leaves"
        // invariant). cfg is shape-only (gate_air_components::<NoValue>) so any shard's `config`
        // works; use shard 0's.
        let (
            ref base0_extended,
            ref _base0_claim,
            _base0_nonce,
            _base0_salt,
            base0_log_n_rows,
            base0_prog_log_size,
            ref base0_boundary,
            base0_total_pc,
        ) = shard_bases[0];
        let base0_config = leaf::leaf_pcs_config(
            base0_log_n_rows.max(RC_LOG_SIZE),
            std::env::var("BASE_BLOWUP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(BASE_LOG_BLOWUP_FACTOR),
        );
        let cfg = ProofConfig::new(
            &gate_air_components::<NoValue>(),
            n_pp,
            &base0_config,
            INTERACTION_POW_BITS,
        );
        let pp_root0: ReducedHashValue<SecureField> = base0_extended.proof.commitments[0].into();
        let shape_params = GateAirLeafParams {
            main_log_size: base0_log_n_rows,
            program_log_size: base0_prog_log_size,
            preprocessed_root: pp_root0,
            boundary: base0_boundary.clone(),
            total_pc: base0_total_pc,
        };

        eprintln!("gate-air: deriving aggregate config ...");
        let t = Instant::now();
        let agg = derive_aggregate_config(&cfg, &shape_params, LOG_BLOWUP_FACTOR);
        eprintln!(
            "gate-air: config derived in {:.1}s (target qm31_ops={})",
            t.elapsed().as_secs_f64(),
            agg.target_padding_sizes.qm31_ops
        );

        // Partition the machine so independent leaf proves run concurrently (POOL_THREADS sweet spot).
        let pool_threads: usize = std::env::var("POOL_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48);
        let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(pool_threads);
        let pools = PoolSet::new((cores / pool_threads).max(1), pool_threads);

        // Per-shard distinct leaf: build the GateAirLeafParams for THIS shard (its own boundary +
        // preprocessed root), convert THIS shard's base proof to circuit values, and prove the leaf.
        // The leaves are now DISTINCT (each commits to its shard's own shots' (x,y) outputs).
        let n_leaves = n_shards;
        let (cfg_ref, agg_ref) = (&cfg, &agg);

        // `wrap_leaf` turns one shard's base-proof tuple into its TreeProof leaf. Same params
        // construction + proof_from_stark_proof + prove_gate_air_leaf as the sequential path; only
        // WHEN it runs differs between the two paths.
        let wrap_leaf = |base: &BaseShardOutput| -> TreeProof {
            let (extended_i, claim_i, nonce_i, salt_i, log_n_rows_i, prog_log_i, boundary_i, total_pc_i) =
                base;
            let pp_root_i: ReducedHashValue<SecureField> = extended_i.proof.commitments[0].into();
            let params_i = GateAirLeafParams {
                main_log_size: *log_n_rows_i,
                program_log_size: *prog_log_i,
                preprocessed_root: pp_root_i,
                boundary: boundary_i.clone(),
                total_pc: *total_pc_i,
            };
            let p = proof_from_stark_proof(extended_i, cfg_ref, claim_i.clone(), *nonce_i, *salt_i);
            prove_gate_air_leaf(p, cfg_ref, &params_i, agg_ref)
        };

        let (leaves, out) = if pipeline {
            // PIPELINE: producer thread proves shards 1.. (GPU) and sends each base over a bounded
            // sync_channel (depth 1, so the producer doesn't race far ahead of the consumer / blow
            // memory). The consumer (this thread) wraps each base into a leaf in shard order
            // (starting with shard 0, already proved) and streams the leaves into
            // `recursive_aggregate_prove_streaming` via a second channel. So GPU base-proving of
            // shard i+1 overlaps CPU leaf-wrap + fold of shard i. The leaves Vec is collected in
            // shard order (0..N) for the unchanged prove_root_verification + fingerprint below.
            eprintln!("gate-air: PIPELINED base||recursion (streaming frontier fold)");
            let t = Instant::now();
            let mut leaves_vec: Vec<TreeProof> = Vec::with_capacity(n_leaves);
            let (base_tx, base_rx) = std::sync::mpsc::sync_channel::<Result<BaseShardOutput>>(1);
            let (leaf_tx, leaf_rx) = std::sync::mpsc::channel::<TreeProof>();
            let shard0_base = shard_bases.into_iter().next().unwrap();

            let out = std::thread::scope(|scope| -> Result<AggregateOutput> {
                // PRODUCER: prove shards 1..n_shards on the GPU (single producer — one GPU).
                let producer = scope.spawn(|| {
                    for shard_cases in shard_case_sets[1..].iter() {
                        let r = prove_base_shard(shard_cases);
                        let is_err = r.is_err();
                        // Stop on send failure (consumer gone) or after forwarding an error.
                        if base_tx.send(r).is_err() || is_err {
                            break;
                        }
                    }
                });

                // FOLD: run the streaming fold on a worker thread driven by `leaf_rx`, so the
                // consumer (this thread) can keep wrapping the next leaf while a fold proceeds.
                let folder = scope.spawn(|| {
                    recursive_aggregate_prove_streaming(leaf_rx, n_leaves, agg_ref, &pools)
                });

                // CONSUMER (this thread): wrap shard 0 first, then each received base in order.
                let leaf0 = wrap_leaf(&shard0_base);
                leaves_vec.push(leaf0.clone());
                leaf_tx.send(leaf0).expect("fold thread dropped early");
                for _ in 1..n_shards {
                    let base = base_rx.recv().expect("producer hung up early")?;
                    let leaf = wrap_leaf(&base);
                    leaves_vec.push(leaf.clone());
                    leaf_tx.send(leaf).expect("fold thread dropped early");
                }
                drop(leaf_tx);
                producer.join().expect("base producer thread panicked");
                Ok(folder.join().expect("fold thread panicked"))
            })?;
            eprintln!(
                "gate-air: pipelined {n_leaves} leaves + fold in {:.1}s ({} levels)",
                t.elapsed().as_secs_f64(),
                out.n_levels
            );
            eprintln!("gate-air: multiverifier fold OK");
            (leaves_vec, out)
        } else {
            eprintln!("gate-air: proving {n_leaves} distinct leaves (one per shard) ...");
            let t = Instant::now();
            let leaf_jobs: Vec<_> = shard_bases
                .iter()
                .map(|base| move || wrap_leaf(base))
                .collect();
            let leaves = pools.map(leaf_jobs);
            eprintln!("gate-air: {n_leaves} distinct leaves proved in {:.1}s", t.elapsed().as_secs_f64());

            let t = Instant::now();
            let out = recursive_aggregate_prove(leaves.clone(), &agg, &pools);
            eprintln!(
                "gate-air: folded to root in {:.1}s ({} levels)",
                t.elapsed().as_secs_f64(),
                out.n_levels
            );
            eprintln!("gate-air: multiverifier fold OK");
            (leaves, out)
        };

        // Root verification: verify the root proof + unpack the leaf outputs, with the single
        // zk-blinding (the only published proof). Completes the recursion pipeline.
        let zk = ZkBlind {
            seed: [7u8; 32],
            n_padding: leaf_pcs_config(1, LOG_BLOWUP_FACTOR).fri_config.n_queries,
        };
        let t = Instant::now();
        let rv = prove_root_verification(&out.root, &leaves, &agg, LOG_BLOWUP_FACTOR, Some(zk));
        eprintln!(
            "gate-air: root verification OK in {:.1}s (trace 2^{}, {} leaf outputs unpacked + zk-blinded)",
            t.elapsed().as_secs_f64(),
            rv.trace_log_size,
            rv.leaf_outputs.len()
        );

        // ---- Recursion byte-identity fingerprint (precompute ON vs OFF validation) ----
        // The precompute optimization only changes HOW each node/leaf's tree0 is built, never WHAT.
        // So the leaf proofs, every internal node proof (folded into `out.root`), and the root
        // proof must be byte-identical between precompute ON (default) and OFF
        // (GATE_AIR_NO_PRECOMPUTE=1). `Proof<QM31>` is purely Vec/array/struct of QM31 (no maps),
        // so its `{:?}` Debug form is a deterministic, cross-process canonical encoding. We fold
        // every leaf proof + its output values, the root proof + its output values, and the
        // unpacked leaf outputs into one SHA-256 and print it for the two runs to compare.
        {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(b"gate-air/recursion-proofs/debug/v1");
            hasher.update(format!("n_leaves={n_leaves} n_levels={}", out.n_levels).as_bytes());
            for (i, l) in leaves.iter().enumerate() {
                hasher.update(format!("leaf[{i}].proof={:?}", l.proof).as_bytes());
                hasher.update(format!("leaf[{i}].pp_root={:?}", l.preprocessed_root).as_bytes());
                hasher.update(format!("leaf[{i}].outs={:?}", l.output_values).as_bytes());
            }
            hasher.update(format!("root.proof={:?}", out.root.proof).as_bytes());
            hasher.update(format!("root.pp_root={:?}", out.root.preprocessed_root).as_bytes());
            hasher.update(format!("root.outs={:?}", out.root.output_values).as_bytes());
            hasher.update(format!("rv.proof={:?}", rv.proof).as_bytes());
            hasher.update(format!("rv.leaf_outputs={:?}", rv.leaf_outputs).as_bytes());
            let digest = hasher.finalize();
            let mode = if std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok() {
                "PRECOMPUTE_OFF"
            } else {
                "PRECOMPUTE_ON"
            };
            // The fold completing = every leaf proof verified in-circuit by its parent node; the
            // root verification completing = the root proof verified in-circuit. Both self-verify.
            println!("gate-air: recursion_fingerprint[{mode}]={}", hex::encode(digest));
            println!("gate-air: recursion self-verify (fold+root) OK [{mode}]");
        }
    }

    let proof = extended.proof;

    // ---- Verify ----
    let verify_start = Instant::now();
    let verifier_channel = &mut Blake2sM31Channel::default();
    // Mirror the prover's canonical transcript exactly.
    verifier_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
    config.mix_into(verifier_channel);
    let commitment_scheme_v = &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
    commitment_scheme_v.commit(proof.commitments[0], &sizes[0], verifier_channel);
    verifier_channel.mix_felts(&public_claim);
    commitment_scheme_v.commit(proof.commitments[1], &sizes[1], verifier_channel);
    verifier_channel.mix_u64(interaction_pow_nonce);
    let v_elements = LookupElements::draw(verifier_channel);
    let v_boundary = public_boundary_sum(cases, n_gates, k, &v_elements.state)?;
    let v_qdecode = table_public_sum(&counts.qdecode, &v_elements.qdecode, TAG_QDECODE, |qi| {
        let (l, p, m) = qubit_decode(qi as u16);
        vec![
            BaseField::from_u32_unchecked(qi as u32),
            BaseField::from_u32_unchecked(l),
            BaseField::from_u32_unchecked(p),
            BaseField::from_u32_unchecked(m),
        ]
    });
    let v_rc_lo = table_public_sum(&counts.rc_lo, &v_elements.rc_lo, TAG_RC_LO, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_lo_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_lo_index.val_col[i]),
        ]
    });
    let v_rc_hi = table_public_sum(&counts.rc_hi, &v_elements.rc_hi, TAG_RC_HI, |i| {
        vec![
            BaseField::from_u32_unchecked(rc_hi_index.pos_col[i]),
            BaseField::from_u32_unchecked(rc_hi_index.val_col[i]),
        ]
    });
    let v_program = table_public_sum(&program.multiplicity, &v_elements.program, TAG_PROGRAM, |i| {
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

// ----------------------------------------------------------------------------
// Full-proof byte-identity fingerprint (read-only)
// ----------------------------------------------------------------------------
//
// Computes a STABLE, deterministic SHA-256 over the serde-serialized `ExtendedStarkProof`
// and prints `gate-air: proof_fingerprint=<hex>`. This is a READ-ONLY tap: it only serializes
// the already-produced proof, touching no CPU constraint-eval / prover / verifier math.
//
// Determinism: the proof is a pure function of (fixture, samples, canonical Fiat-Shamir
// transcript). The transcript is fixed — `channel_salt` is a fixed scalar mixed first, then
// `config.mix_into`, then the fixed INTERACTION_POW_BITS grind (deterministic nonce for a fixed
// transcript). No RNG/salt is drawn outside the channel. serde_json serializes struct fields in
// declaration order and field elements / hashes as plain numbers / byte arrays, so the byte stream
// is identical across runs and across backends (SimdBackend vs CudaBackend). The backend under test
// is therefore the ONLY possible source of divergence — which is the point of the comparison.
//
// We hash the serde form (not `format!("{:?}", ..)`) because it is a canonical, version-stable
// encoding; the Debug form would also be deterministic (it is what `cuda_byte_identity.rs` uses)
// and is kept as the fallback below if serialization were ever to fail.
fn emit_proof_fingerprint<H>(extended: &stwo::core::proof::ExtendedStarkProof<H>)
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    stwo::core::proof::StarkProof<H>: serde::Serialize,
{
    use sha2::{Digest, Sha256};
    // Fingerprint ONLY the verifier-consumed `proof` (StarkProof): it is entirely Vec/struct-based
    // and therefore serializes deterministically. The `aux` (in-circuit-verifier helper data)
    // contains HashMaps (all_node_values / all_values) whose serde iteration order is randomized
    // per process — including it makes the fingerprint differ run-to-run even on the SAME backend,
    // which is NOT a proof divergence. Byte-identity of the actual proof = byte-identity of `.proof`.
    let mut hasher = Sha256::new();
    match serde_json::to_vec(&extended.proof) {
        Ok(bytes) => {
            hasher.update(b"gate-air/stark-proof/serde/v1");
            hasher.update(&bytes);
        }
        Err(e) => {
            // Fallback: stable Debug fingerprint (matches stwo's cuda_byte_identity.rs pattern).
            eprintln!("gate-air: proof serde failed ({e}); falling back to Debug fingerprint");
            hasher.update(b"gate-air/stark-proof/debug/v1");
            hasher.update(format!("{:?}", extended.proof).as_bytes());
        }
    }
    let digest = hasher.finalize();
    println!("gate-air: proof_fingerprint={}", hex::encode(digest));
}

fn normalize(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}
