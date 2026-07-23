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

mod base; // Base (per-shard) gate_air prover, extracted from this file (mirrors `leaf.rs`).
mod circuit_statement; // In-circuit verifier of the gate_air STARK proof.
mod fingerprint; // Proof-fingerprint helpers behind the env-gated hooks (distinct from the `diag` feature).
#[cfg(feature = "gpu-cuda")]
mod gpu_tracegen;
mod leaf;
mod recursion_consts; // PINNED recursion constants, keyed per operating point.
mod topology; // Free topology params + env-var knob layer.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
#[allow(unused_imports)]
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
// Trace-gen backend is ALWAYS SimdBackend (columns are built with CPU column ops); `to_prover`
// bridges to `ProverBackend` at the `extend_evals` boundary (a real host->device upload under cuda).
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo::core::proof_of_work::GrindOps;
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove_ex, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation, RelationEntry,
    TraceLocationAllocator,
};
// Only used by the `#[cfg(test)]` on-trace constraint helpers (T5).
#[cfg(test)]
use stwo_constraint_framework::assert_constraints_on_trace;

use base::*;
use topology::TopologyConfig;

// Encoding constants
const N_QUBITS: usize = 512;
const LIMB_BITS: usize = 16;
const N_LIMBS: usize = N_QUBITS / LIMB_BITS; // 32
const STATE_BYTES: usize = N_QUBITS / 8; // 64

/// Production log-size `R` of the ts-ordering range-check (rc) supply table: a single block
/// enumerating `[0, 2^RC_LOG)` with `val[i] = i`, so one lookup per access checks
/// `d = pc - prev_ts ∈ [0, 2^RC_LOG)`. FIXED at 2^25 (sized for the k≈8000 target). The PUBLIC,
/// trusted `R` (never read from a proof), threaded into the base prover + `GateAirStatement`; tests
/// pass their own small `R`. Sound while every honest `d` fits: `d_max = k*n_gates - 1 < 2^RC_LOG`
/// (k ≲ 8000; base prove-entry `debug_assert` guards it). `LOG_N_LANES <= rc_log <= log_n_rows` and
/// `2^RC_LOG < p`, so no SIMD underflow, field wrap, or raised FRI floor.
pub(crate) const RC_LOG: u32 = 25;

/// Log-size of the tree-0 twiddle / eval (committed) domain: the MAX over every committed column's
/// log-size (`main` = `log_n_rows`, the `rc` membership table = `rc_log`, the `program` table, the
/// `boundary` table). The tree-0 columns are interpolated on twiddles of this size, so the twiddle
/// tree MUST cover the LARGEST committed column — not just `log_n_rows.max(rc_log)`. For a REAL large
/// shard (`k*n_gates >= 512`, so `main >> boundary = shots*512` and `program = n_gates` are tiny) this
/// reduces to `log_n_rows` (dynamic-rc_log property preserved: NO domain inflation). It only differs
/// for TINY fixtures where `k*n_gates < 512`, so `boundary` / `program` outsize the main trace and the
/// old fixed `RC_LOG_SIZE=16` floor used to (incidentally) cover them.
pub(crate) fn tree0_max_log_size(
    log_n_rows: u32,
    rc_log: u32,
    program_log_size: u32,
    boundary_log_size: u32,
) -> u32 {
    log_n_rows
        .max(rc_log)
        .max(program_log_size)
        .max(boundary_log_size)
}

// Interaction-trace proof-of-work bits (canonical transcript; matches the in-circuit verifier's
// ProofConfig). Tiny grind (~2^8), present so the in-circuit verifier can replay the transcript.
const INTERACTION_POW_BITS: u32 = 8;

const NO_CTRL: u16 = 0xFFFF;

// Timestamp/range-check soundness (authoritative spot). `ts = pc + 1`, `pc` the PREPROCESSED,
// verifier-pinned per-shot program counter — so ts is a fixed affine function of pc the prover cannot
// reorder, and ts is inlined (not a witness column). The `+1` keeps the smallest real ts = 1 > 0 =
// the init boundary node's ts (plain `pc` would collide pc=0's accesses with init). Within a gate step
// the (<=3) accesses hit distinct addresses, so sharing ts = pc+1 never collides two on one chain;
// same-address accesses are in different steps (distinct pc), so their ts strictly increases.
//
// The diff `d = pc - prev_ts` is a SINGLE 25-bit column, proven in [0, 2^RC_LOG) by one LogUp lookup
// into the EXACT-range rc table (no slack). Honest `d <= k*n_gates - 1 < 2^RC_LOG` (completeness); the
// bound < p (RC_LOG <= TS_RC_BITS = 25) means the field subtraction cannot wrap, so a cyclic stale-read
// chain is impossible. pc-pinned ts (program order) + `prev_ts < ts` on every access ⇒ forward DAG.
const TS_RC_BITS: usize = 25;

const M31_MODULUS_U32: u32 = (1 << 31) - 1;
const LANE_COUNT: usize = 1 << LOG_N_LANES;

// ONE shared LogUp relation (single drawn (z,α)); logical relations are distinguished by a distinct id
// TAG prepended as the first tuple element, matching the in-circuit verifier's single-relation model
// one acc.interaction_elements, relation id as a constant in the tuple). Width =
// widest payload (program = slot,opcode,target,ctrl_a,ctrl_b = 5) + 1 tag = 6.
#[allow(dead_code)]
const GATE_REL_WIDTH: usize = 6;
stwo_constraint_framework::relation!(GateRel, 6);

// Relation id tags (distinct constants; prover and in-circuit verifier must agree). TAG_QUBITMEM =
// per-qubit chain-lookup relation; TAG_RC = ts-ordering range-check (main looks up `d` as (TAG_RC, d),
// the rc table supplies (TAG_RC, value) for value in [0, 2^RC_LOG)).
const TAG_QUBITMEM: u32 = 1;
const TAG_RC: u32 = 2;
const TAG_PROGRAM: u32 = 5;
// H_P program binding. The program table emits `-mult` on TAG_PROGRAM (internal, cancels main's
// demand) and `+mult` on TAG_PROGRAM_PUB (public dangling term P_pub in `program_sum`). The leaf
// supplies −P_pub over its guessed program Vars — which it also hashes into H_P — binding H_P to the
// executed program. A distinct tag is required (reusing TAG_PROGRAM would make the +mult cancel the
// −mult, vacuous). CPU-side (`gen_program_interaction`), not in the GPU K4 (MAIN-only) kernel.
const TAG_PROGRAM_PUB: u32 = 6;

// x/y binding: the boundary's final `y` is re-keyed to a FIXED public ts `TS_FINAL` so it surfaces as
// an unconsumed public LogUp term (the leaf supplies the matching term over its guessed x/y, forcing
// guessed == committed). TS_FINAL must exceed every real ts (0..=k*n_gates) and be a valid M31, so the
// public tuples never alias an interior chain node: 2^30 < p and >> any real ts.
const TS_FINAL: u32 = 1 << 30;

/// The logical relations share the SAME drawn `(z,α)` (clones of one `GateRel`); the tag prepended at
/// each combine is what keeps them separate.
#[derive(Clone)]
struct LookupElements {
    qubitmem: GateRel,
    rc: GateRel,
    program: GateRel,
}

impl LookupElements {
    fn draw(channel: &mut impl Channel) -> Self {
        let rel = GateRel::draw(channel);
        Self {
            qubitmem: rel.clone(),
            rc: rel.clone(),
            program: rel,
        }
    }

    /// Fixed challenges for byte-identity tests. Used by the K4 GPU interaction validation.
    #[cfg(feature = "gpu-cuda")]
    fn dummy() -> Self {
        let rel = GateRel::dummy();
        Self {
            qubitmem: rel.clone(),
            rc: rel.clone(),
            program: rel,
        }
    }
}

/// Packed tag constant for prover-side `combine` tuples.
fn ptag(tag: u32) -> PackedM31 {
    PackedM31::broadcast(BaseField::from_u32_unchecked(tag))
}

// CLI / fixture

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

// Circuit parser (GTV1)

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

// State helpers (16-bit limbs)

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

#[allow(dead_code)] // used by the cuda/gpu-cuda trace-gen + byte-identity paths
fn limbs_to_state(limbs: &[u32; N_LIMBS]) -> [u8; STATE_BYTES] {
    let mut out = [0u8; STATE_BYTES];
    for (j, &limb) in limbs.iter().enumerate() {
        out[2 * j] = (limb & 0xFF) as u8;
        out[2 * j + 1] = ((limb >> 8) & 0xFF) as u8;
    }
    out
}

#[inline]
#[allow(dead_code)] // used by the cuda/gpu-cuda trace-gen + byte-identity paths
fn qubit_decode(q: u16) -> (u32, u32, u32) {
    let limb_idx = (q as u32) / LIMB_BITS as u32;
    let bit_pos = (q as u32) % LIMB_BITS as u32;
    let mask = 1u32 << bit_pos;
    (limb_idx, bit_pos, mask)
}

// Witness row

/// One qubit-memory access (chain lookup) for target / ctrl_a / ctrl_b.
/// The access timestamp is NOT a witness column: it is the affine function `ts = pc + 1` of the
/// verifier-pinned preprocessed `pc`, inlined at every use site. `prev_ts` is the ts of the previous
/// access to this addr (0 = the init boundary node). `d` is the ordering diff
/// `d = ts - prev_ts - 1 = (pc+1) - prev_ts - 1 = pc - prev_ts`, range-checked by a SINGLE LogUp
/// lookup into the dynamic rc supply table (`d ∈ [0, 2^RC_LOG_SIZE)`), proving `prev_ts < ts`
/// (forward-DAG / no-stale-read). `active` gates the terms.
#[derive(Clone, Copy)]
struct AccessCols {
    addr: u32,    // qubit index 0..511 (0 when inactive, matches program canon)
    prev_ts: u32, // predecessor's ts at this addr (0 if this is the first access)
    v: u32,       // v_before (the value read); for a control this is also v_after
    d: u32, // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LOG_SIZE))
}

impl AccessCols {
    fn inactive() -> Self {
        Self {
            addr: 0,
            prev_ts: 0,
            v: 0,
            d: 0,
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
    target: AccessCols,
    // NOTE: the target's post-gate value `v_after` is NOT a witness column — it equals
    // `v_before + delta` (a pinned equality), inlined at every use site.
    ctrl_a: AccessCols,
    ctrl_b: AccessCols,
    ab: u32,
    fire: u32,
    delta: u32, // v_after - v_before, signed in {-1,0,1}; stored as M31.
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
            target: AccessCols::inactive(),
            ctrl_a: AccessCols::inactive(),
            ctrl_b: AccessCols::inactive(),
            ab: 0,
            fire: 0,
            delta: 0,
        }
    }
}

// Column layout. Per row, one access block = ACCESS_COLS core + 1 rc diff col:
//   is_nop,is_not,is_cnot,is_toffoli                                 (4)
//   target access: addr,prev_ts,v_before, d                         (ACCESS_BLOCK)
//   ctrl_a access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ctrl_b access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ab, fire, delta                                                  (3)
// `enabler`/`shot_id`/`pc` are shard-invariant positional values in the preprocessed tree0. `ts =
// pc + 1` and the target `v_after = v_before + delta` are NOT columns — inlined (see TS_RC_BITS above).
const ACCESS_COLS: usize = 3; // addr, prev_ts, v (core access cols; ts inlined = pc+1)
const ACCESS_BLOCK: usize = ACCESS_COLS + 1; // core cols + the single rc diff col `d`
const TRACE_COLUMNS: usize = 4 + ACCESS_BLOCK + ACCESS_BLOCK + ACCESS_BLOCK + 3; // 4 + 3*4 + 3 = 19

fn delta_to_m31(delta: i64) -> u32 {
    // delta in {-1,0,1}, represented in M31.
    if delta >= 0 {
        delta as u32
    } else {
        (M31_MODULUS_U32 as i64 + delta) as u32
    }
}

// Witness generation + self-check

/// Build all witness rows for the selected shots, asserting each shot's final state matches y_hex.
/// Parallelizes over SHOTS (each owns a disjoint contiguous row block, chained in order within the
/// shot), trace-bit-identical to serial; packing into `PackedM31` happens later single-threaded/word.
fn build_rows(gates: &[Gate], cases: &[TestCase], k: usize) -> Result<(Vec<Row>, BoundaryTable)> {
    use rayon::prelude::*;

    let n_gates = gates.len();
    let shot_rows = k * n_gates;
    let total_rows = cases.len() * shot_rows;

    // LOUD completeness guard (not debug_assert): the max honest ts diff d_max = k*n_gates - 1 must
    // stay below 2^TS_RC_BITS, else the rc-table range-check would silently REJECT honest proofs.
    let d_max: u64 = (k as u64) * (n_gates as u64) - 1;
    if d_max >= (1u64 << TS_RC_BITS) {
        bail!(
            "ts-ordering range-check budget exceeded: k*n_gates = {}*{} makes the max ts diff d_max = {} >= 2^{} \
             (the rc-table bound); honest proofs would silently fail the range-check. Widen TS_RC_BITS / the rc-limb split.",
            k, n_gates, d_max, TS_RC_BITS
        );
    }

    // Pre-allocate the full scalar row buffer; each shot fills a disjoint block.
    let mut rows = vec![Row::padding(); total_rows];
    // Per-shot boundary rows: 512 entries per shot, holding init/final (x, y, ts_last).
    let mut boundary = BoundaryTable::new(cases.len());

    // Phase 1: simulate every shot in parallel into its own disjoint row block +
    // boundary block. Returns Err on the first shot whose simulation fails or whose
    // final state mismatches y_hex.
    let per_shot: Vec<Result<()>> = rows
        .par_chunks_mut(shot_rows)
        .zip(boundary.per_shot_mut().par_iter_mut())
        .zip(cases.par_iter())
        .enumerate()
        .map(|(shot_id, ((block, bnd), case))| simulate_shot(gates, k, shot_id, case, block, bnd))
        .collect();

    for result in per_shot {
        result?;
    }

    Ok((rows, boundary))
}

/// One memory access: reads (prev_ts, v_before) from `last_*[addr]`; the access ts is the inlined
/// program-order `pc + 1`. Returns the `AccessCols` with `v = v_before` and the single rc diff
/// `d = ts - prev_ts - 1 = pc - prev_ts` (>= 0) that the rc-table lookup range-checks. The caller
/// updates `last_*[addr]` to the post-access ts/value.
fn do_access(addr: u32, pc: u32, last_ts: &[u32], last_val: &[u32]) -> AccessCols {
    let a = addr as usize;
    let prev_ts = last_ts[a];
    let v_before = last_val[a];
    let ts = pc + 1;
    debug_assert!(
        ts > prev_ts,
        "ts {ts} must exceed prev_ts {prev_ts} (program order)"
    );
    let d = ts - prev_ts - 1; // = pc - prev_ts; completeness (d < 2^TS_RC_BITS) checked in build_rows.
    debug_assert!(
        (d as u64) < (1u64 << TS_RC_BITS),
        "diff {d} exceeds range-check bound"
    );
    AccessCols {
        addr,
        prev_ts,
        v: v_before,
        d,
    }
}

/// Simulate a single shot sequentially, filling its row block: the chain (K reps * n_gates gates) is
/// run strictly in order, threading the 512-bit state from x_s to y_s, checked against y_hex.
fn simulate_shot(
    gates: &[Gate],
    k: usize,
    shot_id: usize,
    case: &TestCase,
    block: &mut [Row],
    bnd: &mut [BoundaryRow],
) -> Result<()> {
    let n_gates = gates.len();
    let x = hex::decode(&case.x_hex).context("decoding x_hex")?;
    let y = hex::decode(&case.y_hex).context("decoding y_hex")?;
    if x.len() != STATE_BYTES || y.len() != STATE_BYTES {
        bail!("state must be {STATE_BYTES} bytes");
    }
    debug_assert_eq!(bnd.len(), N_QUBITS);

    // Per-shot qubit-memory state: last[addr] = (ts, value). Reset each shot.
    // `ts` is the PROGRAM-ORDER timestamp `pc + 1`: an affine function of the preprocessed pc, so an
    // address's accesses are timestamped in program order and cannot be reordered by the prover.
    let x_bit = |addr: usize| -> u32 { qubit_bit(&x, addr) };
    let mut last_ts = vec![0u32; N_QUBITS];
    let mut last_val: Vec<u32> = (0..N_QUBITS).map(x_bit).collect();
    let mut pc: u32 = 0;
    let mut row_idx = 0usize;

    for _rep in 0..k {
        for gate in gates {
            let (is_nop, is_not, is_cnot, is_toffoli) = match gate.opcode {
                OP_NOP => (1, 0, 0, 0),
                OP_NOT => (0, 1, 0, 0),
                OP_CNOT => (0, 0, 1, 0),
                OP_TOFFOLI => (0, 0, 0, 1),
                other => bail!("unknown opcode {other}"),
            };
            let a_active = is_cnot + is_toffoli;
            let b_active = is_toffoli;

            // Control reads first (they feed the gate-apply), then target read+write. ts is the
            // program-order timestamp `pc + 1`, shared by all accesses of this gate step (no slot).
            // The three accesses of a gate touch DISTINCT addrs (a reversible gate can't use its
            // target as a control), so sharing ts within the step never collides two accesses on the
            // same per-address chain; two accesses to the same addr are in different steps (distinct
            // pc), so they still get strictly increasing ts. In iadd the three accesses touch distinct
            // addrs.
            let ts = pc + 1;
            let ctrl_a = if a_active == 1 {
                debug_assert_ne!(gate.ctrl_a, NO_CTRL);
                let ac = do_access(gate.ctrl_a as u32, pc, &last_ts, &last_val);
                last_ts[gate.ctrl_a as usize] = ts;
                // read: value propagates unchanged.
                ac
            } else {
                AccessCols::inactive()
            };
            let ctrl_b = if b_active == 1 {
                debug_assert_ne!(gate.ctrl_b, NO_CTRL);
                let ac = do_access(gate.ctrl_b as u32, pc, &last_ts, &last_val);
                last_ts[gate.ctrl_b as usize] = ts;
                ac
            } else {
                AccessCols::inactive()
            };
            let target = do_access(gate.target as u32, pc, &last_ts, &last_val);

            let a_bit = ctrl_a.v;
            let b_bit = ctrl_b.v;
            let t_bit = target.v;
            let ab = a_bit * b_bit;
            let fire = is_not + is_cnot * a_bit + is_toffoli * ab;
            debug_assert!(fire <= 1);
            let v_after = t_bit ^ fire;
            let delta_signed = v_after as i64 - t_bit as i64;

            // Commit the target write to memory (ts = pc+1, inlined).
            last_ts[gate.target as usize] = ts;
            last_val[gate.target as usize] = v_after;

            block[row_idx] = Row {
                enabler: 1,
                is_nop,
                is_not,
                is_cnot,
                is_toffoli,
                shot_id: shot_id as u32,
                pc,
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

    // Boundary rows: init x, final y (== last_val), ts_last (0 if untouched).
    for addr in 0..N_QUBITS {
        let y_bit = qubit_bit(&y, addr);
        // Self-check: simulated final value equals y's bit at this addr.
        if last_val[addr] != y_bit {
            bail!(
                "shot {shot_id}: simulated final qubit {addr} = {} != y bit {}",
                last_val[addr],
                y_bit
            );
        }
        bnd[addr] = BoundaryRow {
            shot_id: shot_id as u32,
            addr: addr as u32,
            x: x_bit(addr),
            y: y_bit,
            ts_last: last_ts[addr],
        };
    }

    Ok(())
}

/// Bit `addr` of a little-endian byte state (bit `addr` = byte `addr/8`, bit `addr%8`).
#[inline]
fn qubit_bit(bytes: &[u8], addr: usize) -> u32 {
    ((bytes[addr / 8] >> (addr % 8)) & 1) as u32
}

// rc supply table: a single 2^R-row block enumerating exactly [0, 2^R) with `val[i] = i` (R = rc_log;
// no pos selector, no split). Main looks up (TAG_RC, d) per active access; the table supplies
// -multiplicity / (TAG_RC, value), pinning d < 2^R with no slack (see RC_LOG / TS_RC_BITS above).

// Boundary table: per (shot, addr) emits on TAG_QUBITMEM the INTERNAL final Use[+1](shot,addr,ts_last,y)
// (cancels main's last chain Yield) and the PUBLIC final Yield[-1](shot,addr,TS_FINAL,y) (re-keys y to
// the fixed public ts). `shot`/`addr` preprocessed; `x`/`y`/`ts_last` witness (`x` booleanity-only —
// main carries x publicly via its ts=0 init Use). Nets to the public term B = Σ(+[0,x] − [TS_FINAL,y]),
// which the leaf's public_logup_sum matches over guessed x/y.

/// Convert a committed `ProgramTable` into the leaf's `ProgramRows`. Same per-slot
/// tuple the base commits + the leaf binds via TAG_PROGRAM_PUB and hashes into H_P.
fn program_rows_from_table(prog: &ProgramTable) -> leaf::ProgramRows {
    leaf::ProgramRows {
        slot: prog.slot.clone(),
        opcode_scalar: prog.opcode_scalar.clone(),
        target: prog.target.clone(),
        ctrl_a: prog.ctrl_a.clone(),
        ctrl_b: prog.ctrl_b.clone(),
        multiplicity: prog.multiplicity.clone(),
    }
}

/// The ONE shared hiding nonce for H_P = blake(program ‖ nonce), identical across all leaves of a run.
/// A fixed default (the final proof is zk-blinded at the wrapper, so H_P hiding rests on that blinding;
/// a random nonce per RUN — not per leaf — can be wired later without changing the binding).
///
/// The deterministic override the byte-identity / cross-backend oracle runs need lives in the test
/// setup (T4 `incircuit_self_verify` / T7 `full_proof_gpu_vs_simd_identity`), NOT in this production
/// path — the prove path carries only the fixed default.
fn hiding_nonce() -> [u32; 2] {
    // Fixed default (deterministic). Distinct-per-run randomness is a future refinement; the nonce is
    // binding-inert, so a fixed value does not affect soundness (only the strength of program hiding).
    [0x1234_5678, 0x9abc_def0]
}

// FrameworkEval

// Table FrameworkEvals (supply side of each lookup table)

/// Qubit-memory boundary table (supply side). Per (shot, addr) — `shot`/`addr` preprocessed,
/// `x`/`y`/`ts_last` witness — emits on TAG_QUBITMEM the chain head + tail:
///   init  Yield[-1](shot, addr, 0, x)
///   final Use [+1](shot, addr, ts_last, y)
/// Two terms/row => 1 batch. Booleanity on x, y (1-bit values).
#[derive(Clone)]
struct BoundaryTableEval {
    log_size: u32,
    elements: GateRel,
}

impl FrameworkEval for BoundaryTableEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::one();
        let shot = eval.get_preprocessed_column(pp_id("gate_bnd_shot"));
        let addr = eval.get_preprocessed_column(pp_id("gate_bnd_addr"));
        // Real-row enabler (1 on real (shot,addr), 0 on padding). Gates the boundary emission so
        // padding rows (non-power-of-two n_shots*N_QUBITS) inject no unmatched LogUp terms.
        let bnd_enabler = eval.get_preprocessed_column(pp_id("gate_bnd_enabler"));
        let x = eval.next_trace_mask();
        let y = eval.next_trace_mask();
        let ts_last = eval.next_trace_mask();
        // Booleanity of the boundary values.
        eval.add_constraint(x.clone() * (x.clone() - one.clone()));
        eval.add_constraint(y.clone() * (y.clone() - one.clone()));

        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_QUBITMEM);
        let ts_final = E::F::one() * BaseField::from_u32_unchecked(TS_FINAL);
        // Phase-3 x/y binding (re-keyed boundary). The init anchor is NO LONGER consumed here:
        // main's first Use +1/(shot,addr,0,x) is left DANGLING as a PUBLIC term. The boundary now
        //   (B) INTERNAL final Use  +1 / (shot, addr, ts_last, y)  -> cancels main's last Yield.
        //   (D) PUBLIC  final Yield -1 / (shot, addr, TS_FINAL, y)  -> re-keys y to a fixed public ts.
        // Net base claimed_sum (per shot,addr) = +[0,x] − [TS_FINAL,y]; the leaf's public_logup_sum
        // supplies −that over its guessed x/y, so the verifier balance forces guessed == committed.
        // (`x` is retained as a witness column with its booleanity constraint above; it is no longer
        // emitted by the boundary — it lives only on main's dangling init term.)
        // (B) internal final Use[+bnd_enabler]: +bnd_enabler / (shot, addr, ts_last, y).
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            E::EF::from(bnd_enabler.clone()),
            &[tag.clone(), shot.clone(), addr.clone(), ts_last, y.clone()],
        ));
        // (D) public final Yield[-bnd_enabler]: -bnd_enabler / (shot, addr, TS_FINAL, y).
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(bnd_enabler),
            &[tag, shot, addr, ts_final, y],
        ));
        eval.finalize_logup_in_pairs();
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
        // H_P binding (Fork A): two supply terms, paired into ONE batch (so the program interaction
        // stays 4 columns — no tree2 layout shift, no GPU K4 change):
        //   (internal, -mult) / combine(TAG_PROGRAM,     slot, op, t, a, b)  -> cancels main's demand
        //   (public,   +mult) / combine(TAG_PROGRAM_PUB, slot, op, t, a, b)  -> dangling P_pub
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_PROGRAM);
        let tag_pub = E::F::one() * BaseField::from_u32_unchecked(TAG_PROGRAM_PUB);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity.clone()),
            &[
                tag,
                slot.clone(),
                opcode_scalar.clone(),
                target.clone(),
                ctrl_a.clone(),
                ctrl_b.clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            E::EF::from(multiplicity),
            &[tag_pub, slot, opcode_scalar, target, ctrl_a, ctrl_b],
        ));
        eval.finalize_logup_in_pairs();
        eval
    }
}

/// ts-ordering range-check table (supply side). `val` is preprocessed (the table membership,
/// val[i]=i over [0,2^log_size)); `multiplicity` is witness (count of real `d` lookups landing on
/// this row). Emits -multiplicity / combine(TAG_RC, val) — one term/row => 1 batch => 4 interaction
/// columns. `log_size` is the trusted construction input `R = rc_log` (production: RC_LOG = 25).
#[derive(Clone)]
struct RcTableEval {
    log_size: u32,
    elements: GateRel,
}

impl FrameworkEval for RcTableEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let val = eval.get_preprocessed_column(pp_id("gate_rc_val"));
        let multiplicity = eval.next_trace_mask();
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, val],
        ));
        eval.finalize_logup();
        eval
    }
}

fn pp_id(id: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: id.to_owned() }
}

/// Number of preprocessed columns (count-only uses; the order is `preprocessed_column_ids`).
/// prog_slot + (enabler/shot_id/pc/pc_in_prog) + (bnd_shot/bnd_addr/bnd_enabler) + rc_val = 9.
const N_PREPROCESSED_COLS: usize = 9;

/// Each preprocessed column paired with its log_size, in canonical order then STABLE-sorted ascending
/// by size — the committed tree MUST be size-sorted (stwo's lifted Merkle sorts by length; the
/// in-circuit verifier does not re-sort). `gate_rc_val` is sized at the trusted `rc_log` (see RC_LOG),
/// never read from the proof — it sizes the [0,2^rc_log) table pinned by the preprocessed root.
fn preprocessed_columns_sorted(
    main_log_size: u32,
    program_log_size: u32,
    boundary_log_size: u32,
    rc_log: u32,
) -> Vec<(PreProcessedColumnId, u32)> {
    let mut cols = vec![
        (pp_id("gate_prog_slot"), program_log_size),
        // Shard-invariant positional main-trace columns (tree0). Sized with the main trace.
        (pp_id("gate_enabler"), main_log_size),
        (pp_id("gate_shot_id"), main_log_size),
        (pp_id("gate_pc"), main_log_size),
        (pp_id("gate_pc_in_prog"), main_log_size),
        // Qubit-memory boundary positional columns. Sized with the boundary table.
        (pp_id("gate_bnd_shot"), boundary_log_size),
        (pp_id("gate_bnd_addr"), boundary_log_size),
        (pp_id("gate_bnd_enabler"), boundary_log_size),
        // ts-ordering range-check table membership (val[i]=i). Sized at the DYNAMIC rc_log.
        (pp_id("gate_rc_val"), rc_log),
    ];
    cols.sort_by_key(|&(_, s)| s); // stable: ties keep the listing order above
    cols
}

fn preprocessed_column_ids(
    main_log_size: u32,
    program_log_size: u32,
    boundary_log_size: u32,
    rc_log: u32,
) -> Vec<PreProcessedColumnId> {
    preprocessed_columns_sorted(main_log_size, program_log_size, boundary_log_size, rc_log)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

// Components bundle

type GateComponent = FrameworkComponent<GateEval>;
type ProgramComponent = FrameworkComponent<ProgramTableEval>;
type BoundaryComponent = FrameworkComponent<BoundaryTableEval>;
type RcComponent = FrameworkComponent<RcTableEval>;

struct Components {
    main: GateComponent,
    program: ProgramComponent,
    boundary: BoundaryComponent,
    rc: RcComponent,
}

impl Components {
    fn component_refs(&self) -> Vec<&dyn Component> {
        vec![
            &self.main as &dyn Component,
            &self.program as &dyn Component,
            &self.boundary as &dyn Component,
            &self.rc as &dyn Component,
        ]
    }

    fn prover_refs(&self) -> Vec<&dyn stwo::prover::ComponentProver<ProverBackend>> {
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.boundary as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.rc as &dyn stwo::prover::ComponentProver<ProverBackend>,
        ]
    }

    /// Component provers bound to `SimdBackend` — the host/SIMD oracle path the cross-backend
    /// byte-identity tests (T6/T7) prove against under a `cuda` build (where `ProverBackend` is the
    /// CudaBackend). `FrameworkComponent<E>` implements `ComponentProver` for both backends. Only
    /// referenced from test code (`prove_tiny_base`), hence `cfg(test)` + `cuda`.
    #[cfg(all(test, feature = "cuda"))]
    fn prover_refs_simd(
        &self,
    ) -> Vec<&dyn stwo::prover::ComponentProver<stwo::prover::backend::simd::SimdBackend>> {
        use stwo::prover::backend::simd::SimdBackend;
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.boundary as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.rc as &dyn stwo::prover::ComponentProver<SimdBackend>,
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
    boundary_log_size: u32,
    rc_log: u32,
    elements: &LookupElements,
    main_sum: SecureField,
    program_sum: SecureField,
    boundary_sum: SecureField,
    rc_sum: SecureField,
) -> Components {
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(
        &preprocessed_column_ids(log_n_rows, program_log_size, boundary_log_size, rc_log),
    );
    let main = GateComponent::new(
        &mut allocator,
        GateEval {
            log_n_rows,
            elements: elements.clone(),
        },
        main_sum,
    );
    let program = ProgramComponent::new(
        &mut allocator,
        ProgramTableEval {
            log_size: program_log_size,
            elements: elements.program.clone(),
        },
        program_sum,
    );
    let boundary = BoundaryComponent::new(
        &mut allocator,
        BoundaryTableEval {
            log_size: boundary_log_size,
            elements: elements.qubitmem.clone(),
        },
        boundary_sum,
    );
    let rc = RcComponent::new(
        &mut allocator,
        RcTableEval {
            log_size: rc_log,
            elements: elements.rc.clone(),
        },
        rc_sum,
    );
    Components {
        main,
        program,
        boundary,
        rc,
    }
}

// Trace generation (column-major)

/// Single scalar cell of `row` at canonical column index `col`. The column order
/// matches the order `evaluate` reads masks (each access block = ACCESS_COLS core + 1 rc diff col):
///   is_{nop,not,cnot,toffoli}(4),
///   target(addr,prev_ts,v, d),
///   ctrl_a(addr,prev_ts,v, d), ctrl_b(addr,prev_ts,v, d),
///   ab, fire, delta (3).
/// NOTE: enabler, shot_id, pc are NOT here — they are preprocessed (tree0). ts (= pc+1) and the
/// target's v_after (= v_before+delta) are NOT columns either — they are inlined in `evaluate`.
#[inline]
fn cell_at(row: &Row, col: usize) -> u32 {
    debug_assert!(col < TRACE_COLUMNS);
    #[inline]
    fn access_cell(a: &AccessCols, i: usize) -> u32 {
        // 0..ACCESS_COLS: addr, prev_ts, v ; then d.
        match i {
            0 => a.addr,
            1 => a.prev_ts,
            2 => a.v,
            _ => a.d,
        }
    }
    let mut c = col;
    // Header (4 cols).
    const HEADER: [fn(&Row) -> u32; 4] =
        [|r| r.is_nop, |r| r.is_not, |r| r.is_cnot, |r| r.is_toffoli];
    if c < HEADER.len() {
        return HEADER[c](row);
    }
    c -= HEADER.len();
    // target block: ACCESS_BLOCK access cols (no v_after column — inlined as v_before+delta).
    if c < ACCESS_BLOCK {
        return access_cell(&row.target, c);
    }
    c -= ACCESS_BLOCK;
    // ctrl_a block.
    if c < ACCESS_BLOCK {
        return access_cell(&row.ctrl_a, c);
    }
    c -= ACCESS_BLOCK;
    // ctrl_b block.
    if c < ACCESS_BLOCK {
        return access_cell(&row.ctrl_b, c);
    }
    c -= ACCESS_BLOCK;
    // Tail: ab, fire, delta.
    match c {
        0 => row.ab,
        1 => row.fire,
        _ => row.delta,
    }
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

/// Preprocessed positional columns for the boundary table: (shot, addr) per row.
fn generate_boundary_preprocessed(
    bnd: &BoundaryTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let shot: Vec<u32> = bnd.rows.iter().map(|r| r.shot_id).collect();
    let addr: Vec<u32> = bnd.rows.iter().map(|r| r.addr).collect();
    // Real-row enabler: 1 for the first `n_shots*N_QUBITS` rows, 0 on padding. POSITIONAL /
    // shard-invariant (depends only on the shot count). Gates the boundary emission so a non-power-of-
    // two `n_shots*N_QUBITS` (e.g. 9024 shots) does not inject unmatched LogUp terms on padding rows.
    let real = bnd.n_shots * N_QUBITS;
    let enabler: Vec<u32> = (0..bnd.rows.len()).map(|i| (i < real) as u32).collect();
    vec![
        col_from_values(&shot),
        col_from_values(&addr),
        col_from_values(&enabler),
    ]
}

/// Boundary-table witness (x, y, ts_last), in the order BoundaryTableEval reads them.
fn generate_boundary_witness(
    bnd: &BoundaryTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let x: Vec<u32> = bnd.rows.iter().map(|r| r.x).collect();
    let y: Vec<u32> = bnd.rows.iter().map(|r| r.y).collect();
    let ts_last: Vec<u32> = bnd.rows.iter().map(|r| r.ts_last).collect();
    vec![
        col_from_values(&x),
        col_from_values(&y),
        col_from_values(&ts_last),
    ]
}

/// Preprocessed `enabler` column: 1 on real rows, 0 on padding. SHARD-INVARIANT and POSITIONAL —
/// depends only on how many real rows the (k, n_gates, n_shots) shape produces, never on secret
/// content. Used by the AIR both in the opcode one-hot constraint and as the LogUp numerator.
fn generate_enabler_preprocessed(
    rows: &[Row],
    padded_rows: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let mut vals = vec![0u32; padded_rows];
    for (i, r) in rows.iter().enumerate() {
        vals[i] = r.enabler;
    }
    col_from_values(&vals)
}

/// Preprocessed `shot_id` column: shot index of each row (= row / (k*n_gates)), 0 on padding.
/// SHARD-INVARIANT and POSITIONAL (positional payload in the state-relation tuple).
fn generate_shot_id_preprocessed(
    rows: &[Row],
    padded_rows: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let mut vals = vec![0u32; padded_rows];
    for (i, r) in rows.iter().enumerate() {
        vals[i] = r.shot_id;
    }
    col_from_values(&vals)
}

/// Preprocessed `pc` column: monotonic per-shot program counter (= row % (k*n_gates)), 0 on
/// padding. SHARD-INVARIANT and POSITIONAL (positional payload in the state-relation tuple).
fn generate_pc_preprocessed(
    rows: &[Row],
    padded_rows: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let mut vals = vec![0u32; padded_rows];
    for (i, r) in rows.iter().enumerate() {
        vals[i] = r.pc;
    }
    col_from_values(&vals)
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

/// Preprocessed rc-table membership column (val), the single column RcTableEval reads.
fn generate_rc_preprocessed(
    rc: &RcTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&rc.val)]
}

/// rc-table witness (multiplicity tree): a single multiplicity column.
fn generate_rc_witness(
    rc: &RcTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&rc.multiplicity)]
}

// Interaction traces

#[inline]
fn pack(lane: &[&Row; LANE_COUNT], get: impl Fn(&Row) -> u32) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|l| {
        BaseField::from_u32_unchecked(get(lane[l]))
    }))
}

/// Main component interaction trace. Logup batches mirror the relation entries
/// emitted in `evaluate`, finalized in pairs. Order must match (5 batches -> 4 columns each):
///   pair0: qubitmem target Use (+enabler), target Yield (-enabler)
///   pair1: qubitmem ctrl_a Use (+a_active), ctrl_a Yield (-a_active)
///   pair2: qubitmem ctrl_b Use (+b_active), ctrl_b Yield (-b_active)
///   pair3: rc target d (+enabler), rc ctrl_a d (+a_active)
///   pair4: rc ctrl_b d (+b_active), program (+enabler)
/// 10 relation entries -> 5 pairs => 5 batches => 20 interaction columns. The rc terms are the
/// ts-ordering range-check single-`d` lookups (TAG_RC, d) mirroring `add_rc_lookup`.
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

    fn sel_t(r: &Row) -> &AccessCols {
        &r.target
    }
    fn sel_a(r: &Row) -> &AccessCols {
        &r.ctrl_a
    }
    fn sel_b(r: &Row) -> &AccessCols {
        &r.ctrl_b
    }

    // Chain USE (predecessor): (shot, addr, prev_ts, v_before).
    let qm_use = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.qubitmem.combine(&[
            ptag(TAG_QUBITMEM),
            pack(lane, |r| r.shot_id),
            pack(lane, |r| sel(r).addr),
            pack(lane, |r| sel(r).prev_ts),
            pack(lane, |r| sel(r).v),
        ])
    };
    // Chain YIELD (successor): (shot, addr, ts=pc+1, v_after). ts is the inlined `pc + 1` (not a
    // column). For the target, v_after = v_before + delta; for controls the value propagates (== v).
    let qm_yield_t = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.qubitmem.combine(&[
            ptag(TAG_QUBITMEM),
            pack(lane, |r| r.shot_id),
            pack(lane, |r| r.target.addr),
            pack(lane, |r| r.pc + 1),
            // v_after = v_before + delta, computed as the exact bit v_before ^ fire (canonical M31,
            // avoids the non-canonical p that a raw `v_before + delta_to_m31(-1)` would produce).
            pack(lane, |r| r.target.v ^ r.fire),
        ])
    };
    let qm_yield_ctrl =
        |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
            el.qubitmem.combine(&[
                ptag(TAG_QUBITMEM),
                pack(lane, |r| r.shot_id),
                pack(lane, |r| sel(r).addr),
                pack(lane, |r| r.pc + 1),
                pack(lane, |r| sel(r).v),
            ])
        };
    let enabler = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.enabler);
    let a_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_cnot + r.is_toffoli);
    let b_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_toffoli);

    // ts-ordering range-check use-side denominator: (TAG_RC, d) — single `d` per access.
    let rc_d = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.rc.combine(&[ptag(TAG_RC), pack(lane, |r| sel(r).d)])
    };

    // Program use-side denominator. pc_in_prog = pc mod n_gates (preprocessed in
    // the AIR; recomputed here for the prover). opcode_scalar from the one-hot.
    let ng = n_gates as u32;
    let program = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.program.combine(&[
            ptag(TAG_PROGRAM),
            pack(lane, |r| r.pc % ng),
            pack(lane, |r| r.is_not + 2 * r.is_cnot + 3 * r.is_toffoli),
            pack(lane, |r| r.target.addr),
            pack(lane, |r| r.ctrl_a.addr),
            pack(lane, |r| r.ctrl_b.addr),
        ])
    };

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

    // pair0: qubitmem target Use (+enabler), target Yield (-enabler).
    write_pair(
        &mut gen,
        &enabler,
        &|l| qm_use(l, sel_t),
        1,
        &enabler,
        &qm_yield_t,
        -1,
    );
    // pair1: qubitmem ctrl_a Use (+a_active), ctrl_a Yield (-a_active).
    write_pair(
        &mut gen,
        &a_active,
        &|l| qm_use(l, sel_a),
        1,
        &a_active,
        &|l| qm_yield_ctrl(l, sel_a),
        -1,
    );
    // pair2: qubitmem ctrl_b Use (+b_active), ctrl_b Yield (-b_active).
    write_pair(
        &mut gen,
        &b_active,
        &|l| qm_use(l, sel_b),
        1,
        &b_active,
        &|l| qm_yield_ctrl(l, sel_b),
        -1,
    );
    // pair3: rc target d (+enabler), rc ctrl_a d (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| rc_d(l, sel_t),
        1,
        &a_active,
        &|l| rc_d(l, sel_a),
        1,
    );
    // pair4: rc ctrl_b d (+b_active), program (+enabler). The trailing 3 rc + 1 program fold into
    // these two pairs (matches `finalize_logup_in_pairs` on the 10-entry stream).
    write_pair(
        &mut gen,
        &b_active,
        &|l| rc_d(l, sel_b),
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

/// Assert the main component's AIR constraints (algebraic + logup) directly on the committed trace
/// columns, pinpointing the first violated constraint index. Builds the trees
/// `[preprocessed(empty), main, interaction]` the main `GateEval` expects and runs
/// `assert_constraints_on_trace`. Test-support (drives the `on_trace_constraints_all` test, T5),
/// out of the prove path — hence `#[cfg(test)]`.
#[cfg(test)]
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
    let main_vals: Vec<Vec<BaseField>> = main_cols.iter().map(|c| c.values.to_cpu()).collect();
    let interaction_vals: Vec<Vec<BaseField>> =
        main_interaction.iter().map(|c| c.values.to_cpu()).collect();
    // The main component reads four preprocessed columns IN THIS ORDER (matching
    // `GateEval::evaluate`'s `get_preprocessed_column` calls): enabler, shot_id, pc, pc_in_prog.
    // `assert_constraints_on_trace` feeds them positionally to the eval's preprocessed reads.
    let pp_vals: Vec<Vec<BaseField>> = vec![
        generate_enabler_preprocessed(rows, padded_rows)
            .values
            .to_cpu(),
        generate_shot_id_preprocessed(rows, padded_rows)
            .values
            .to_cpu(),
        generate_pc_preprocessed(rows, padded_rows).values.to_cpu(),
        generate_pc_in_prog_preprocessed(rows, padded_rows, n_gates)
            .values
            .to_cpu(),
    ];

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

/// Assert a table (supply-side) component's constraints directly on its committed columns. Trees:
/// `[preprocessed, multiplicity, interaction]`. Test-support (T5), out of the prove path.
#[cfg(test)]
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

/// Packs one boundary field over a vec_row's lanes.
fn pack_boundary(
    bnd: &BoundaryTable,
    vec_row: usize,
    f: &dyn Fn(&BoundaryRow) -> u32,
) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(f(&bnd.rows[(vec_row << LOG_N_LANES) + lane]))
    }))
}

/// Boundary component interaction trace: per (shot, addr) row emit the internal final
/// Use[+1](shot, addr, ts_last, y) and the PUBLIC final Yield[-1](shot, addr, TS_FINAL, y) on
/// TAG_QUBITMEM. Two terms per row -> one batch (paired), matching `BoundaryTableEval`. CPU-only in
/// both the CPU and cuda paths (the CUDA kernel covers only the gate_air MAIN component).
fn gen_boundary_interaction(
    bnd: &BoundaryTable,
    el: &GateRel,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(bnd.log_size);
    let mut col = gen.new_col();
    let n_vec = 1usize << (bnd.log_size - LOG_N_LANES);
    let pack_f = |vec_row: usize, f: &dyn Fn(&BoundaryRow) -> u32| pack_boundary(bnd, vec_row, f);
    let ts_final = PackedM31::broadcast(BaseField::from_u32_unchecked(TS_FINAL));
    let real = bnd.n_shots * N_QUBITS;
    for vec_row in 0..n_vec {
        let tag = ptag(TAG_QUBITMEM);
        let shot = pack_f(vec_row, &|r| r.shot_id);
        let addr = pack_f(vec_row, &|r| r.addr);
        let y = pack_f(vec_row, &|r| r.y);
        let ts_last = pack_f(vec_row, &|r| r.ts_last);
        // Real-row enabler per lane (mirrors the gate_bnd_enabler preprocessed column).
        let enabler = PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(((vec_row << LOG_N_LANES) + lane < real) as u32)
        }));
        let enabler = PackedSecureField::from(enabler);
        // Phase-3 re-keyed boundary (mirrors BoundaryTableEval), gated by the real-row enabler:
        //   (B) internal final Use[+enabler] / (shot, addr, ts_last, y).
        let d_use: PackedSecureField = el.combine(&[tag, shot, addr, ts_last, y]);
        //   (D) public   final Yield[-enabler] / (shot, addr, TS_FINAL, y).
        let d_pub: PackedSecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        // fraction = (+enabler)/d_use + (-enabler)/d_pub.
        col.write_frac(vec_row, enabler * d_pub + (-enabler) * d_use, d_use * d_pub);
    }
    col.finalize_col();
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

/// Program-table interaction (H_P binding, Fork A). Mirrors `gen_boundary_interaction`'s two-term/
/// one-batch shape so the program component stays 4 interaction columns. Per real slot row emits:
///   (internal, -mult) / combine(TAG_PROGRAM,     slot, op, t, a, b)  — cancels main's demand,
///   (public,   +mult) / combine(TAG_PROGRAM_PUB, slot, op, t, a, b)  — the dangling P_pub.
/// Padding rows carry multiplicity 0, so both fractions vanish (numerator 0). The returned claimed
/// sum is `program_sum` = Σ_slot [ -mult/d_int + mult/d_pub ] = P_pub (the internal part is cancelled
/// by main's demand only in the GLOBAL sum, not within this component; `program_sum` itself carries
/// BOTH terms, and the global identity becomes main + program + boundary + rc == B + P_pub, where the
/// `-mult/d_int` inside program_sum cancels main's `+enabler/d_int`, leaving net P_pub).
fn gen_program_interaction(
    prog: &ProgramTable,
    el: &GateRel,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(prog.log_size);
    let mut col = gen.new_col();
    let n_vec = 1usize << (prog.log_size - LOG_N_LANES);
    let tag = ptag(TAG_PROGRAM);
    let tag_pub = ptag(TAG_PROGRAM_PUB);
    for vec_row in 0..n_vec {
        let slot = pack_seq(&prog.slot, vec_row);
        let op = pack_seq(&prog.opcode_scalar, vec_row);
        let t = pack_seq(&prog.target, vec_row);
        let a = pack_seq(&prog.ctrl_a, vec_row);
        let b = pack_seq(&prog.ctrl_b, vec_row);
        let mult = PackedSecureField::from(pack_seq(&prog.multiplicity, vec_row));
        let d_int: PackedSecureField = el.combine(&[tag, slot, op, t, a, b]);
        let d_pub: PackedSecureField = el.combine(&[tag_pub, slot, op, t, a, b]);
        // fraction = (-mult)/d_int + (+mult)/d_pub.
        col.write_frac(vec_row, (-mult) * d_pub + mult * d_int, d_int * d_pub);
    }
    col.finalize_col();
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

/// Program-table PUBLIC term P_pub = Σ_slot mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b) — the
/// dangling public part the leaf's `public_logup_sum` supplies the negation of (binding the guessed
/// program to the committed one). Recomputed from the committed program witness.
fn program_public_term(prog: &ProgramTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag_pub = BaseField::from_u32_unchecked(TAG_PROGRAM_PUB);
    for i in 0..prog.multiplicity.len() {
        let count = prog.multiplicity[i];
        if count == 0 {
            continue;
        }
        let denom: SecureField = el.combine(&[
            tag_pub,
            BaseField::from_u32_unchecked(prog.slot[i]),
            BaseField::from_u32_unchecked(prog.opcode_scalar[i]),
            BaseField::from_u32_unchecked(prog.target[i]),
            BaseField::from_u32_unchecked(prog.ctrl_a[i]),
            BaseField::from_u32_unchecked(prog.ctrl_b[i]),
        ]);
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    sum
}

/// Program component claimed sum (H_P binding): Σ_slot [ -mult/combine(TAG_PROGRAM,...) +
/// mult/combine(TAG_PROGRAM_PUB,...) ]. Must equal `program_sum`; recomputed from the committed
/// program witness so a mistranscribed term is caught before FRI.
fn program_claimed_sum(prog: &ProgramTable, el: &GateRel) -> SecureField {
    let internal = table_public_sum(&prog.multiplicity, el, TAG_PROGRAM, |i| {
        vec![
            BaseField::from_u32_unchecked(prog.slot[i]),
            BaseField::from_u32_unchecked(prog.opcode_scalar[i]),
            BaseField::from_u32_unchecked(prog.target[i]),
            BaseField::from_u32_unchecked(prog.ctrl_a[i]),
            BaseField::from_u32_unchecked(prog.ctrl_b[i]),
        ]
    });
    // `table_public_sum` returns the NEGATED supply (-Σ count/denom), i.e. exactly the -mult internal
    // term. The public term adds +Σ mult/denom_pub = program_public_term.
    internal + program_public_term(prog, el)
}

/// Boundary component claimed sum (supply side): Σ_rows [ +1/combine(ts_last,y) − 1/combine(TS_FINAL,y) ]
/// — the re-keyed final terms (B)+(D). Must equal `boundary_sum`; recomputed from the committed
/// boundary table (y/ts_last witness) so a mistranscribed term is caught before FRI.
fn boundary_public_sum(bnd: &BoundaryTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag = BaseField::from_u32_unchecked(TAG_QUBITMEM);
    let ts_final = BaseField::from_u32_unchecked(TS_FINAL);
    // Only REAL rows emit (gated by gate_bnd_enabler); padding rows contribute nothing.
    let real = bnd.n_shots * N_QUBITS;
    for r in &bnd.rows[..real] {
        let shot = BaseField::from_u32_unchecked(r.shot_id);
        let addr = BaseField::from_u32_unchecked(r.addr);
        let y = BaseField::from_u32_unchecked(r.y);
        let ts_last = BaseField::from_u32_unchecked(r.ts_last);
        // (B) internal final Use[+1] and (D) public final Yield[-1]: +1/d_use − 1/d_pub.
        let d_use: SecureField = el.combine(&[tag, shot, addr, ts_last, y]);
        let d_pub: SecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        sum += d_use.inverse() - d_pub.inverse();
    }
    sum
}

/// Phase-3 PUBLIC boundary term B = Σ_rows [ +1/combine(shot,addr,0,x) − 1/combine(shot,addr,TS_FINAL,y) ].
/// This is exactly the base's total dangling (unconsumed) LogUp sum: main leaves +[0,x] & −[ts_last,y]
/// per touched (shot,addr), the boundary consumes [ts_last,y] and re-emits −[TS_FINAL,y], so the net is
/// +[0,x] − [TS_FINAL,y]. The base's committed claimed sums must satisfy
/// `main_sum + program_sum + boundary_sum == B` (was `== 0`), and the LEAF's `public_logup_sum` equals
/// `−B` over its GUESSED x/y — so the verifier balance `public_logup_sum + Σ claimed_sums == 0` forces
/// guessed == committed. For UNTOUCHED addrs (ts_last=0) the prover's `x == y`, so the ts=0 term here
/// (using `x`) faithfully matches the actual dangling +[0,y]; the equality thus also checks x==y there.
fn boundary_public_term(bnd: &BoundaryTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag = BaseField::from_u32_unchecked(TAG_QUBITMEM);
    let zero = BaseField::zero();
    let ts_final = BaseField::from_u32_unchecked(TS_FINAL);
    // Only REAL rows are dangling; padding rows emit no main NOR boundary term (gated by enablers).
    let real = bnd.n_shots * N_QUBITS;
    for r in &bnd.rows[..real] {
        let shot = BaseField::from_u32_unchecked(r.shot_id);
        let addr = BaseField::from_u32_unchecked(r.addr);
        let x = BaseField::from_u32_unchecked(r.x);
        let y = BaseField::from_u32_unchecked(r.y);
        let d_init: SecureField = el.combine(&[tag, shot, addr, zero, x]);
        let d_pub: SecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        sum += d_init.inverse() - d_pub.inverse();
    }
    sum
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

// GPU trace-gen inputs (device path)

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
    let off_lo: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_lo_index.offset[p] as u32)
        .collect();
    let off_hi: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_hi_index.offset[p] as u32)
        .collect();
    Ok((gates_flat, x_states, off_lo, off_hi))
}

/// Multi-shard resident OOM fix (opt-in `GATE_AIR_POOL_TRIM`, default OFF). At a shard boundary,
/// trims the calling thread's device mem pool (`cudaMemPoolTrimTo` after a stream-0 sync) so the next
/// shard starts clean and can run fully resident. Byte-identical: trims only already-free segments,
/// never a live allocation or committed value; default OFF ⇒ never called.
#[cfg(feature = "cuda")]
fn pool_trim_at_shard_boundary_if_enabled() {
    if std::env::var("GATE_AIR_POOL_TRIM").is_ok() {
        // SAFETY: FFI. Sync (stream 0) + trim the current device's pool. No args, no pointers.
        unsafe { stwo::stwo_cuda::bindings::cuda_pool_trim() };
    }
}

/// Build the (size-sorted) preprocessed tree-0 columns for one shard shape. Shared by the
/// per-shard rebuild path AND the precompute build, so the committed column order/sizes are
/// IDENTICAL by construction. tree-0 is SHARD-INVARIANT: every column is POSITIONAL
/// (enabler/shot_id/pc/pc_in_prog from row index, prog_slot/program witness from the shared program
/// with constant multiplicity = shots_per_shard*k, bnd_shot/bnd_addr from the boundary layout), so
/// for a fixed (k, n_gates, shots_per_shard) shape these columns are the same for every shard. The
/// rc-table membership columns (pos, val) are also shard-invariant (fixed [0,range) table).
fn build_tree0_columns(
    program: &ProgramTable,
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    rc_log: u32,
    boundary: &BoundaryTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let mut tagged: Vec<(
        u32,
        CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>,
    )> = vec![
        (program.log_size, generate_prog_slot_preprocessed(program)),
        (log_n_rows, generate_enabler_preprocessed(rows, padded_rows)),
        (log_n_rows, generate_shot_id_preprocessed(rows, padded_rows)),
        (log_n_rows, generate_pc_preprocessed(rows, padded_rows)),
        (
            log_n_rows,
            generate_pc_in_prog_preprocessed(rows, padded_rows, n_gates),
        ),
    ];
    tagged.extend(
        generate_boundary_preprocessed(boundary)
            .into_iter()
            .map(|c| (boundary.log_size, c)),
    );
    // rc membership table sized at the DYNAMIC rc_log (single [0,2^rc_log) val column).
    let rc = RcTable::new(rc_log);
    tagged.extend(
        generate_rc_preprocessed(&rc)
            .into_iter()
            .map(|c| (rc_log, c)),
    );
    tagged.sort_by_key(|(s, _)| *s); // stable: identical key+listing order as preprocessed_columns_sorted
    tagged.into_iter().map(|(_, c)| c).collect()
}

/// The PINNED per-operating-point unpacker verify [`CircuitConfig`] — `op`'s `PinnedConfigs` rebuilt
/// (`to_derived`) at the default node/leaf blowup the pinned points were captured with. Asserts
/// `n == op.n()` (the unpacker is per-operating-point; only this point's leaf count is pinned).
fn pinned_unpacker_config(
    op: recursion_consts::OperatingPoint,
    n: usize,
) -> circuit_verifier::verify::CircuitConfig {
    assert_eq!(
        n,
        op.n(),
        "unpacker config requested for n={n} but this operating point has N={}",
        op.n()
    );
    op.pinned()
        .to_derived(
            topology::RECURSION_LOG_BLOWUP,
            topology::RECURSION_LOG_BLOWUP,
        )
        .unpacker
}

fn verify_gate_air_root_leaves(
    rv: &recursive_aggregate::root_prover::RootVerificationOutput,
    op: recursion_consts::OperatingPoint,
    n: usize,
) -> anyhow::Result<()> {
    use circuit_verifier::verify::{verify_circuit, CircuitPublicData};

    // (1) The trusted unpacker verify config is the PINNED per-N const (no recompute/commit at verify).
    //     Its `preprocessed_root` is the canonical unpacker root; a proof whose unpacker baked a forged
    //     child root has a different preprocessed root and is REJECTED here.
    let verify_config = pinned_unpacker_config(op, n);

    // (2) Verify the published proof with the CALLER-COMMITTED per-leaf outputs.
    let output_values: Vec<SecureField> = rv.leaf_outputs.iter().flatten().copied().collect();
    verify_circuit(
        verify_config,
        rv.proof.clone(),
        CircuitPublicData { output_values },
    )
    .map(|_| ())
    .map_err(|e| anyhow::anyhow!("trusted gate_air root verification failed (leaf-recursion): {e}"))
}

// main

/// Whether the env var `name` is ENABLED under default-ON / opt-out semantics:
/// enabled unless explicitly set to "0"/"false"/"FALSE" (same idiom as
/// CUDA_GPU_CONSTRAINTS in stwo-cuda-backend). An unset var is ON.
fn env_flag_default_on(name: &str) -> bool {
    !matches!(
        std::env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

/// Sharded multiverifier-tree fold path (GATE_AIR_FOLD). DEFAULT-ON; opt out with
/// GATE_AIR_FOLD=0 (or "false"/"FALSE").
fn fold_enabled() -> bool {
    env_flag_default_on("GATE_AIR_FOLD")
}

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

    // `rc_lo_index` now feeds ONLY the CUDA trace-gen glue (device offset buffers); unused on the
    // CPU-only default build after the per-address +1 counter fix removed the range-check lookup.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let rc_lo_index = build_rc_lo();
    let program = build_program_table(&gates, samples, k);

    let cases = &fixture.test_cases[..samples];

    // The GPU K1/K4 trace-gen byte-identity self-checks (GPU trace == CPU recompute) are NOT a
    // prove-path hook: they run as the `diag`-gated `#[test]`s `k1_trace_identity` (T1a) /
    // `k4_interaction_identity` (T1b), which call the `gpu_tracegen::k1_byte_identity` /
    // `k4_byte_identity` harnesses directly on the box.

    // `trace_gen_start` marks the beginning of the full witness build (shot
    // simulation + every column fill + interaction traces). We stop the clock
    // immediately before the FRI `prove` call; `prove_s`/`verify_s` stay as-is.
    let trace_gen_start = Instant::now();

    // Shape scalars. `build_rows` returns exactly `cases.len() * k * n_gates` rows
    // (`cases.len() == samples`), so `real_rows` and the derived padding are computed here CHEAPLY —
    // without materializing the O(samples) `Vec<Row>`. The FOLD path (which returns before the
    // single-proof path below) only ever needs these scalars; each shard rebuilds its OWN rows in
    // `prove_base_shard` and derives shape from `shard_bases[0]`. Only the non-fold single-proof
    // path (and the no-prove report) consumes the full witness, so `rows`/`counts` are built there.
    let real_rows = samples * k * n_gates;
    let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let log_n_rows = padded_rows.ilog2();

    // NOTE: there is intentionally no GLOBAL `real_rows < M31_MODULUS` precheck here. `pc`/`ts` are
    // PER-SHARD, not global (each shard's `pc = row % (k*n_gates)`, `ts` in 0..=k*n_gates; shot_id is
    // shard-local, `last_ts` resets per shot, and the LogUp chain balances within a shot). The only
    // M31/range quantity that matters is the per-shard `d_max = k*n_gates - 1`, guarded loudly inside
    // `build_rows` against 2^TS_RC_BITS (which is < TS_FINAL < p, so it also covers those bounds).
    // A global `samples*k*n_gates` bound was a leftover from the pre-sharding design and wrongly
    // rejected honest large-`samples` runs (e.g. --samples 9024 at k=100).

    // In FOLD mode this top-level buffer is DEAD (each shard rebuilds its own), so skip it — at
    // large N it is the single O(N)-scaling host allocation (`Row` is 88 bytes) and OOMs the box.
    let fold_active = fold_enabled();
    if fold_active {
        prove_folded(&gates, n_gates, k, samples, cases, &rc_lo_index)
    } else {
        prove_monolithic(
            &gates,
            n_gates,
            k,
            samples,
            cases,
            &rc_lo_index,
            program,
            real_rows,
            padded_rows,
            log_n_rows,
            trace_gen_start,
            args.no_prove,
        )
    }
}

/// Sharded multiverifier-tree fold path (extracted from `main`'s `if fold_active` branch).
/// Pure code move: the body below is the former fold branch verbatim (param-threaded).
#[allow(clippy::too_many_arguments)]
fn prove_folded(
    gates: &[Gate],
    n_gates: usize,
    k: usize,
    samples: usize,
    cases: &[TestCase],
    rc_lo_index: &RcIndex,
) -> Result<()> {
    use circuit_statement::gate_air_components;
    use circuits::blake::HashValue;
    use circuits::context::FinalizedContext;
    use circuits::ivalue::NoValue;
    use circuits::wrappers::U32Wrapper;
    use circuits_stark_verifier::proof::{Proof, ProofConfig};
    use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
    use leaf::{
        build_gate_air_leaf_circuit, build_recursion_precompute, pinned_aggregate_config,
        GateAirLeafParams,
    };
    use recursive_aggregate::pools::PoolSet;
    use recursive_aggregate::precomputes::RecursionPrecompute;
    use recursive_aggregate::prove::recursive_aggregate_prove_leaves;
    use recursive_aggregate::prove_streaming::recursive_aggregate_prove_leaves_streaming;
    use recursive_aggregate::root_prover::{prove_root_verification_leaves, LeafBottom, ZkBlind};
    use recursive_aggregate::AggregateOutput;
    use recursive_aggregate::{AggregateConfig, TreeProof};
    use stwo::core::fields::qm31::QM31;

    // The base-proof tuple `prove_base_shard` returns (defined in `base.rs`, imported here so the
    // fold block's unqualified references resolve). `prove_ex` yields `ExtendedStarkProof<MC::H>`
    // with `MC::H = Blake2sMerkleHasher`, so this is backend-independent (cuda vs simd).
    use base::BaseShardOutput;

    // All FREE topology params in one place, honoring the existing env sweep knobs (BASE_BLOWUP,
    // BASE_FAN_ARITY, RECURSION_SHARD_SHOTS). Defaults reproduce the current production values, so
    // this is a byte-identical no-op. Threaded through the derive/prove calls below; the base
    // blowup, fold arity, base-fan arity, and shots-per-shard are all read off it.
    let topo = TopologyConfig::from_env();

    // Shard partition: equal-sized shards of `shots_per_shard` shots; ragged final shard is
    // padded (below) so all shards share the leaf circuit shape.
    let shots_per_shard: usize = topo.shots_per_shard.min(samples);
    let n_shards = samples.div_ceil(shots_per_shard);
    eprintln!(
        "gate-air: sharding {samples} shots into {n_shards} shard(s) of {shots_per_shard} shot(s) each \
         (final shard padded by shot-repeat if ragged)"
    );

    // The pinned operating point (k, shots_per_shard → N). Gates to the 3 Tanuj curve points; any other
    // (k, shots) panics (unsupported). The recursion consts (roots + unpacker config) key off this.
    let op = recursion_consts::OperatingPoint::from_params(k, topo.shots_per_shard);

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
    // `prove_base_shard` takes the shared base precompute by reference (`Some` = reuse, `None` =
    // the legacy rebuild-tree0-per-shard fallback, selected by `GATE_AIR_NO_BASE_PRECOMPUTE`).
    // tree0/twiddles/program (and the cuda N3 device buffers) are shard-invariant, so on the
    // reuse path they come from `pc` and only the per-shard witness is built here.

    let n_pp = N_PREPROCESSED_COLS;

    // ---- Base-proof precompute (shard-invariant work built ONCE) ----
    // tree0 + twiddles + the N1 program table + (cuda) the N3 device buffers are shard-invariant, so
    // build them once and share the `Arc` into each `prove_base_shard` call. The build asserts tree0's
    // root equals an independent shard-0 rebuild (the load-bearing soundness check); the
    // precompute-ON == rebuild-per-shard byte-identity is covered by the `base_precompute_identity` test.

    // Two independent heavy precomputes run CONCURRENTLY and join before proving (byte-identical to
    // serial): (1) GPU `BaseProverPrecompute::new` (tree0 + twiddles + N1 + cuda N3); (2) CPU recursion
    // config + `RecursionPrecompute`. They share NO data — (2) is a pure function of PUBLIC params, (1)
    // never reads the recursion config — so `thread::scope` borrows read-only.
    //
    // PROVE-WINDOW timer: starts after startup (fixture load / shot-sim / CUDA init, all excluded),
    // spans the precomputes + base proving + fold + root verification, and STOPS before the trusted
    // verify (a self-check, not prover output). This is the SP1-comparable prover time.
    let t_prove_window = Instant::now();
    #[allow(clippy::type_complexity)]
    let (
        base_precompute,
        cfg,
        leaf_cfg,
        recursion_pre,
        boundary_log_size,
        leaf_program,
        leaf_nonce,
    ): (
        Option<std::sync::Arc<BaseProverPrecompute>>,
        ProofConfig,
        AggregateConfig,
        RecursionPrecompute,
        u32,
        leaf::ProgramRows,
        [u32; 2],
    ) = std::thread::scope(|scope| -> Result<_> {
        // --- SPAWNED (CPU): recursion config + precompute, from PUBLIC params only. ---
        let cpu_build = scope.spawn(|| -> Result<_> {
    // Shard-0 shape (identical to what `prove_base_shard` computes for shard 0, and to the shape
    // block inside `BaseProverPrecompute::new`): program table, rows, boundary, row/rc log sizes.
    // `total_pc = k*n_gates` and `preprocessed_root` are PUBLIC (never a proof field).
    let shape_program0 = build_program_table(gates, shots_per_shard, k);
    let (shape_rows0, _shape_boundary0) = build_rows(gates, &shard_case_sets[0], k)?;
    let shape_real_rows0 = shape_rows0.len();
    let shape_padded_rows0 =
        shape_real_rows0.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let shape_log_n_rows0 = shape_padded_rows0.ilog2();
    let shape_rc_log0 = RC_LOG;
    // `cfg` (base circuit ProofConfig): the PCS sized from row/rc log (matches the old `base0_config`
    // read off the proved base, which used `base0_log_n_rows.max(base0_rc_log)`).
    let base0_config = leaf::leaf_pcs_config(
        shape_log_n_rows0.max(shape_rc_log0),
        topo.base_log_blowup,
    );
    let cfg = ProofConfig::new(
        &gate_air_components::<NoValue>(),
        n_pp,
        &base0_config,
        INTERACTION_POW_BITS,
    );
    // The per-shard boundary SHAPE (n_shots * 512 rows, padded) is shard-invariant.
    let boundary_log_size = BoundaryTable::new(shots_per_shard).log_size;
    // Shard-invariant program table + one shared hiding nonce, hashed into H_P by every leaf.
    let leaf_program = program_rows_from_table(&shape_program0);
    let leaf_nonce = hiding_nonce();
    // Shard-0 boundary (x->y limb pairs per shot), identical to what `prove_base_shard` builds for
    // shard 0. Only its SHAPE feeds the NoValue config derivation, but we build the real pairs so
    // `shape_params` is byte-identical to the old (proved-base-sourced) value.
    let shape_boundary_pairs: Vec<([u32; N_LIMBS], [u32; N_LIMBS])> = {
        let mut v = Vec::with_capacity(shard_case_sets[0].len());
        for case in &shard_case_sets[0] {
            let x = state_to_limbs(&hex::decode(&case.x_hex)?);
            let y = state_to_limbs(&hex::decode(&case.y_hex)?);
            v.push((x, y));
        }
        v
    };
    // The base preprocessed root is a WITNESS in the leaf statement (`GateAirStatement::new`
    // GUESSES it — circuit_statement.rs), so its VALUE never enters any preprocessed trace nor any
    // leaf/level1/fold node tree (all built from `NoValue` shapes, where the guessed root's value is
    // ignored). We therefore give `shape_params` a byte-irrelevant ZERO placeholder here.
    let placeholder_base_pp_root: HashValue<SecureField> =
        HashValue(std::array::from_fn(|_| U32Wrapper::new_unsafe(SecureField::zero())));
    let shape_params = GateAirLeafParams {
        main_log_size: shape_log_n_rows0,
        program_log_size: shape_program0.log_size,
        boundary_log_size,
        rc_log: RC_LOG,
        preprocessed_root: placeholder_base_pp_root,
        boundary: shape_boundary_pairs,
        total_pc: (k * n_gates) as u32,
        program: leaf_program.clone(),
        nonce: leaf_nonce,
    };
    // Assemble the recursion config from the PINNED verifier consts (no fixed-point loop, no shape
    // derivation) + build its up-front `RecursionPrecompute`. The heavy per-arity `PreprocessedTree`
    // commits happen HERE (in parallel with the GPU precompute); each asserts its committed root
    // equals the pinned const (the soundness tripwire).
    let t_cfg = Instant::now();
    let (leaf_cfg, recursion_pre) = {
        let agg = pinned_aggregate_config(op, &topo, &cfg, &shape_params);
        // Production: build node shapes only for the arities this point's fold uses.
        let pre = build_recursion_precompute(&agg, op, &cfg, &shape_params, false);
        eprintln!(
            "gate-air: leaf-recursion config + precompute built up front in {:.1}s (node target qm31_ops={})",
            t_cfg.elapsed().as_secs_f64(),
            agg.node_target_padding_sizes.qm31_ops,
        );
        (agg, pre)
    };
        Ok((cfg, leaf_cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce))
    });

        // --- MAIN THREAD (GPU): base precompute build (unconditional). ---
        let base_precompute: Option<std::sync::Arc<BaseProverPrecompute>> = {
            let t_pc = Instant::now();
            // Shard 0's shape (every shard shares it: equal shot count, same program + k).
            let program0 = build_program_table(gates, shots_per_shard, k);
            let (rows0, boundary0) = build_rows(gates, &shard_case_sets[0], k)?;
            let real_rows0 = rows0.len();
            let padded_rows0 = real_rows0.next_power_of_two().max(1 << (LOG_N_LANES + 2));
            let log_n_rows0 = padded_rows0.ilog2();
            let rc_log0 = RC_LOG;
            let max_log_size0 =
                tree0_max_log_size(log_n_rows0, rc_log0, program0.log_size, boundary0.log_size);
            let base_blowup: u32 = topo.base_log_blowup;
            let config0 = leaf::leaf_pcs_config(max_log_size0, base_blowup);
            #[cfg(feature = "cuda")]
            let (gates_flat0, _x0, off_lo0, off_hi0) =
                gpu_flat_inputs(&gates, &shard_case_sets[0], &rc_lo_index, &rc_lo_index)?;
            let pc = BaseProverPrecompute::new(
                config0,
                max_log_size0,
                program0,
                &rows0,
                boundary0,
                padded_rows0,
                log_n_rows0,
                n_gates,
                rc_log0,
                #[cfg(feature = "cuda")]
                &gates_flat0,
                #[cfg(feature = "cuda")]
                &off_lo0,
                #[cfg(feature = "cuda")]
                &off_hi0,
            )?;
            // Load-bearing soundness check: cached tree0 root == independent shard-0 rebuild. DEBUG-ONLY
            // (the release proof is byte-identical — the check feeds nothing into the proof), so the hot
            // path pays nothing. `tests::tree0_precompute_matches_rebuild` gives CI coverage; this call
            // additionally guards the real per-run data (and the cuda tree0 path the test cannot reach).
            #[cfg(debug_assertions)]
            assert_tree0_matches_rebuild(&pc, &rows0, n_gates);
            eprintln!(
                "gate-air: base precompute built (tree0+twiddles+N1{}) in {:.3}s",
                if cfg!(feature = "cuda") { "+N3" } else { "" },
                t_pc.elapsed().as_secs_f64()
            );
            Some(std::sync::Arc::new(pc))
        };

        // --- JOIN: both precomputes complete here, before any proving. ---
        let (cfg, leaf_cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce) = cpu_build
            .join()
            .expect("recursion config/precompute thread panicked")?;
        Ok((
            base_precompute,
            cfg,
            leaf_cfg,
            recursion_pre,
            boundary_log_size,
            leaf_program,
            leaf_nonce,
        ))
    })?;

    let base_precompute_ref = base_precompute.as_deref();
    let recursion_pre_ref = &recursion_pre;

    // PIPELINE opt-in (GATE_AIR_PIPELINE + >1 shard): overlap GPU base-proving (producer) with CPU
    // leaf-wrap + streaming fold (consumer). The producer proves all shards while the consumer wraps +
    // folds in shard order; unset (default) runs the sequential path below unchanged. Byte-identity of
    // the streaming vs sequential recursion_fingerprint is validated on-box, not here.
    let pipeline = env_flag_default_on("GATE_AIR_PIPELINE") && n_shards > 1;

    // MULTI-GPU base proving: how many GPUs prove base shards concurrently in one process. Default 1
    // (single-producer, device 0, byte-identical). With GATE_AIR_BASE_GPUS=G>1 (+ pipeline) the producer
    // spawns G threads, thread n binds GPU n once and proves its shards, all feeding the SAME ordered
    // consumer channel. Clamped to the visible device count (fail-loud) and to producer-shard count;
    // only meaningful with the CUDA backend.
    let base_gpus: usize = {
        let requested = std::env::var("GATE_AIR_BASE_GPUS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&g| g >= 1)
            .unwrap_or(1);
        #[cfg(feature = "cuda")]
        {
            if requested > 1 {
                let visible = gpu_tracegen::backend_device_count();
                assert!(
                    visible >= requested,
                    "GATE_AIR_BASE_GPUS={requested} but only {visible} CUDA device(s) visible \
                     (set CUDA_VISIBLE_DEVICES or lower the knob) — no silent fallback"
                );
            }
            requested
        }
        #[cfg(not(feature = "cuda"))]
        {
            if requested > 1 {
                eprintln!("gate-air: GATE_AIR_BASE_GPUS>1 ignored on non-cuda build (using 1)");
            }
            1
        }
    };
    if base_gpus > 1 && !pipeline {
        // Multi-GPU base proving distributes producer shards across GPUs inside the PIPELINE
        // consumer; without GATE_AIR_PIPELINE there are no producer threads to distribute. Fail
        // loud rather than silently prove everything on device 0.
        bail!(
            "GATE_AIR_BASE_GPUS={base_gpus} requires GATE_AIR_PIPELINE (multi-GPU base proving \
             runs on the pipeline producer threads); set GATE_AIR_PIPELINE or GATE_AIR_BASE_GPUS=1"
        );
    }
    // (Per-device base precompute is now implemented — see `BaseProverPrecompute::device_parts`:
    // each producer thread lazily rebuilds tree0/twiddles/N3 on ITS device, so the shared
    // precompute is multi-GPU-safe and NO_BASE_PRECOMPUTE is NOT required. Device 0 keeps the
    // eager fields => single-GPU byte-identical.)

    // Prove the per-shard base proof(s) (each is itself heavy / GPU-bound). In the sequential
    // path, prove all up front here. In the pipeline path, prove NOTHING here — ALL shards
    // (0..n_shards) are produced concurrently by the producer thread(s) below (shard 0 included),
    // so `shard_bases` stays empty in that branch.
    let t = Instant::now();
    let mut shard_bases = Vec::with_capacity(n_shards);
    if pipeline {
        // No eager base proof: the producers below cover every shard (0..n_shards). Shard 0 maps
        // to gpu `0 % g == 0`, i.e. device 0 — the same device it was proved on when it was eager
        // — so its base proof is byte-identical.
        eprintln!("gate-air: pipeline: all {n_shards} base proofs produced concurrently (no eager shard) ...");
    } else {
        eprintln!("gate-air: proving {n_shards} distinct per-shard base proof(s) ...");
        for (s, shard_cases) in shard_case_sets.iter().enumerate() {
            eprintln!(
                "gate-air: base proof for shard {s} ({} shots) ...",
                shard_cases.len()
            );
            let tb = Instant::now();
            shard_bases.push(base::prove_base_shard(
                base_precompute_ref,
                shard_cases,
                gates,
                k,
                n_gates,
                &topo,
                rc_lo_index,
            )?);
            eprintln!(
                "gate-air: MEASURE t_base[shard {s}]={:.3}s",
                tb.elapsed().as_secs_f64()
            );
            // RESIDENT multi-shard OOM fix (GATE_AIR_POOL_TRIM): shard `s`'s device buffers are
            // dropped now; trim the pool so shard s+1 starts clean and fits fully resident.
            #[cfg(feature = "cuda")]
            pool_trim_at_shard_boundary_if_enabled();
        }
    }
    eprintln!(
        "gate-air: base proof(s) (so far) in {:.1}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Base-proof byte-identity fingerprint (validation-only), gated by
    // GATE_AIR_BASE_PROOF_HASH. The base-precompute optimization only changes HOW each shard's
    // tree0/twiddles/program/N3 are built, never WHAT — so every shard's base `ExtendedStarkProof`
    // GATE_AIR_BASE_PROOF_HASH. Light, read-only SHA over the proved base shards — an env-gated
    // observation hook (off by default; ~free when off), NOT a path toggle. The precompute-ON ==
    // rebuild-per-shard A/B comparison this print used to anchor is now the `base_precompute_identity`
    // test (T2), which drives `prove_base_shard` `Some(pc)` vs `None` and compares
    // `fingerprint::base_proof_fingerprint` directly. Prints on the sequential path (all bases up
    // front) and continues into the fold (no early exit).
    if !pipeline && std::env::var("GATE_AIR_BASE_PROOF_HASH").is_ok() {
        println!(
            "gate-air: base_proof_fingerprint={}",
            fingerprint::base_proof_fingerprint(&shard_bases)
        );
    }

    // Partition the machine so independent leaf proves run concurrently. Each pool holds one in-flight
    // multi-GB `TreeProof`, so #pools == #proofs-in-flight == the peak-RAM multiplier: a big box wants
    // cores/24 pools; a memory-limited box collapses to 1 (no RAM multiplier, avoids the N>=4 OOM).
    // Default 24 is box-measured. When a single pool is used we clamp its workers to the core count
    // (else `PoolSet::new(1, 24)` oversubscribes with big thread stacks). Thread-count only —
    // byte-identical output.
    let pool_threads: usize = std::env::var("POOL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(pool_threads);
    let n_pools = (cores / pool_threads).max(1);
    // With a single pool there is no sibling proof to run alongside it, so let that one pool use
    // all cores rather than reserving `pool_threads` (48) large thread stacks it can't schedule.
    let threads_per_pool = if n_pools == 1 {
        pool_threads.min(cores)
    } else {
        pool_threads
    };
    let pools = PoolSet::new(n_pools, threads_per_pool);

    // Per-shard distinct leaf: build the GateAirLeafParams for THIS shard (its own boundary +
    // preprocessed root), convert THIS shard's base proof to circuit values, and prove the leaf.
    // The leaves are now DISTINCT (each commits to its shard's own shots' (x,y) outputs).
    let n_shards_bases = n_shards;
    let cfg_ref = &cfg;

    // `make_base` turns one shard's base-proof tuple into a `(Proof<QM31>, GateAirLeafParams)`
    // base. Same params construction + proof_from_stark_proof as before; the per-leaf circuit
    // wrap is done later by `prove_gate_air_leaf`. Base `i` == shard `i`.
    let make_base = |base: &BaseShardOutput| -> (Proof<QM31>, GateAirLeafParams) {
        let (
            extended_i,
            claim_i,
            nonce_i,
            salt_i,
            log_n_rows_i,
            prog_log_i,
            boundary_i,
            total_pc_i,
        ) = base;
        let pp_root_i: HashValue<SecureField> = extended_i.proof.commitments[0].into();
        let params_i = GateAirLeafParams {
            main_log_size: *log_n_rows_i,
            program_log_size: *prog_log_i,
            boundary_log_size,
            rc_log: RC_LOG,
            preprocessed_root: pp_root_i,
            boundary: boundary_i.clone(),
            total_pc: *total_pc_i,
            // Shard-invariant program + shared nonce (same for every base).
            program: leaf_program.clone(),
            nonce: leaf_nonce,
        };
        let p = proof_from_stark_proof(extended_i, cfg_ref, claim_i.clone(), *nonce_i, *salt_i);
        (p, params_i)
    };

    // Base↔leaf overlap (pipeline only): as bases arrive from the GPU producer channel, feed them into
    // `recursive_aggregate_prove_leaves_streaming`, which wraps each into a leaf AND folds the tree
    // progressively on the CPU `pools`, so GPU base-proving overlaps both the leaf-wrap and the fold.
    // Pipeline yields the already-folded `(leaves, AggregateOutput)`; the non-pipeline path yields `bases`.
    let overlap_leaves = pipeline;
    type BaseWithParams = (Proof<QM31>, GateAirLeafParams);
    type OverlappedFold = Option<(Vec<TreeProof>, AggregateOutput)>;
    let (bases, overlapped_fold): (Vec<BaseWithParams>, OverlappedFold) = if pipeline {
        // PIPELINE: producer thread(s) prove shards 1.. (GPU) and send each base — TAGGED WITH ITS
        // SHARD INDEX — over a channel. With GATE_AIR_BASE_GPUS>1, up to `base_gpus` producer
        // threads run concurrently (one per GPU, round-robin over shards 1..n_shards): thread `g`
        // binds device `g` (cuda_set_device(g) via set_base_gpu) and proves the shards assigned to
        // it. The consumer (this thread) REORDERS the tagged bases back into strict shard order.
        // Base-proving overlaps across GPUs (base_wall ≈ ⌈(N-1)/G⌉·t_base). Once every base is
        // materialized in shard order we group them into base-nodes and fold (see TODO above:
        // the base-node/fold step no longer overlaps base-proving under base-fanning).
        eprintln!(
            "gate-air: PIPELINED base-proving (G-wide), then base-node group + fold, base_gpus={base_gpus}"
        );
        let t = Instant::now();
        // Shard-indexed base slots (filled in order by the consumer). `Option` lets us place each
        // base at its shard position regardless of producer arrival order.
        let mut bases_vec: Vec<Option<(Proof<QM31>, GateAirLeafParams)>> =
            (0..n_shards_bases).map(|_| None).collect();
        // OVERLAP (Model 1): the streaming coordinator wraps + folds progressively and returns the
        // ordered leaves + the folded root; captured here (escapes the producer `thread::scope`).
        // Left `None` on the non-overlap path (`bases_vec` is used instead).
        let mut overlap_result: Option<(Vec<TreeProof>, AggregateOutput)> = None;
        // Tagged bases: (shard_index, result). Unbounded so no producer blocks a peer.
        let (base_tx, base_rx) = std::sync::mpsc::channel::<(usize, Result<BaseShardOutput>)>();
        // `shard_bases` is empty in the pipeline path — every shard (0..n_shards) is proved by the
        // producers below.
        debug_assert!(
            shard_bases.is_empty(),
            "pipeline path proves all shards in producers"
        );
        let n_producer_shards = n_shards; // shards 0..n_shards (shard 0 included)
        let g = base_gpus.min(n_producer_shards.max(1));

        std::thread::scope(|scope| -> Result<()> {
            // PRODUCERS: `g` threads, one per GPU. Producer-shard `s` is proved on gpu `s % g`
            // (round-robin keyed on shard index; g == 1 ⇒ all on gpu 0). Each producer binds its device
            // once via `set_base_gpu(gpu)`, so its `device_parts()`/pool/trim act on ITS device. Shared
            // borrows (read-only `gates`/`topo`/`rc_lo_index`/`shard_case_sets`, Copy `base_precompute`)
            // outlive this `thread::scope`.
            let gates_ref = &gates;
            let topo_ref = &topo;
            let rc_lo_index_ref = &rc_lo_index;
            let shard_case_sets_ref = &shard_case_sets;
            let producers: Vec<_> = (0..g)
                .map(|gpu| {
                    let base_tx = base_tx.clone();
                    scope.spawn(move || {
                        #[cfg(feature = "cuda")]
                        gpu_tracegen::set_base_gpu(gpu);
                        // Class-1 SIGSEGV fix: a PRIVATE rayon pool whose workers are all bound to
                        // THIS producer's device (`gpu`), so the OODS-phase fan-outs inside
                        // `prove_base_shard` (`pcs/mod.rs` `build_weights_hash_map` par_iter + OODS
                        // `par_map_cols`) run on device `gpu` instead of the global pool's
                        // device-0 workers (which deref device-`gpu` pointers => SIGSEGV). Built
                        // once per producer; each `prove_base_shard` runs inside `pool.install`.
                        #[cfg(feature = "gpu-cuda")]
                        let oods_pool = gpu_tracegen::build_device_bound_pool(gpu);
                        // Shards `s` in 0..n_shards with `s % g == gpu` are proved on this gpu.
                        for shard_idx in (0..n_shards).filter(|s| s % g == gpu) {
                            let tb = Instant::now();
                            #[cfg(feature = "gpu-cuda")]
                            let r = oods_pool.install(|| {
                                base::prove_base_shard(
                                    base_precompute_ref,
                                    &shard_case_sets_ref[shard_idx],
                                    gates_ref,
                                    k,
                                    n_gates,
                                    topo_ref,
                                    rc_lo_index_ref,
                                )
                            });
                            #[cfg(not(feature = "gpu-cuda"))]
                            let r = base::prove_base_shard(
                                base_precompute_ref,
                                &shard_case_sets_ref[shard_idx],
                                gates_ref,
                                k,
                                n_gates,
                                topo_ref,
                                rc_lo_index_ref,
                            );
                            eprintln!(
                                "gate-air: MEASURE t_base[shard {shard_idx}] (gpu {gpu})={:.3}s",
                                tb.elapsed().as_secs_f64()
                            );
                            let is_err = r.is_err();
                            if base_tx.send((shard_idx, r)).is_err() || is_err {
                                break;
                            }
                            // RESIDENT multi-shard OOM fix (GATE_AIR_POOL_TRIM): this shard's
                            // device buffers dropped when `prove_base_shard_ref` returned (the sent
                            // base is host-side). Trim THIS producer's own device pool (bound via
                            // set_base_gpu(gpu)) so its NEXT round-robin shard starts clean.
                            #[cfg(feature = "cuda")]
                            pool_trim_at_shard_boundary_if_enabled();
                        }
                    })
                })
                .collect();
            // Drop the parent's tx clone so `base_rx` disconnects once all producers finish.
            drop(base_tx);

            // CONSUMER (this thread). Every shard (0..n_shards) arrives tagged from the producers.
            if overlap_leaves {
                // OVERLAP (Model 1): feed each base into `recursive_aggregate_prove_leaves_streaming`
                // AS IT ARRIVES, concurrent with the GPU producers still proving later shards. The
                // coordinator owns the single build+prove+level1+fold-node pool and folds progressively;
                // the injected `build` closure (make_base + build_gate_air_leaf_circuit) runs INSIDE its
                // pool workers, with proving-utils running `prove_leaf` on the SAME worker right after,
                // so GPU base-proving overlaps BOTH the leaf build+prove and the fold. Leaf i = shard i
                // (index-tagged), byte-identical to the sequential build+prove+fold.
                let agg = &leaf_cfg;
                let pre = recursion_pre_ref;
                let make_base_ref = &make_base;
                let pools_ref = &pools;
                // The build closure the coordinator runs per leaf (heavy — runs inside a pool
                // worker via the crate's `pool.install`); proving-utils proves it in the same worker.
                let build = move |base: BaseShardOutput| -> FinalizedContext<QM31> {
                    let (proof, params) = make_base_ref(&base);
                    build_gate_air_leaf_circuit::<QM31>(proof, cfg_ref, &params)
                };
                // The coordinator reads `(shard_idx, base)`; a small forward loop on THIS thread
                // pulls tagged producer results and forwards the Ok bases, so a base `Err` still
                // short-circuits via `?` (as the non-overlap drain does). The coordinator runs on
                // its own scope thread so it folds while this thread keeps draining producers.
                let (leaf_tx, leaf_rx) = std::sync::mpsc::channel::<(usize, BaseShardOutput)>();
                let fold_handle = scope.spawn(move || {
                    recursive_aggregate_prove_leaves_streaming(
                        leaf_rx, n_shards, build, agg, pre, pools_ref,
                    )
                });
                let mut base_err: Option<anyhow::Error> = None;
                for _ in 0..n_shards {
                    let (shard_idx, base) = base_rx.recv().expect("producer hung up early");
                    match base {
                        Ok(base) => {
                            // Coordinator gone (already errored/panicked) ⇒ stop forwarding.
                            if leaf_tx.send((shard_idx, base)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            base_err = Some(e);
                            break;
                        }
                    }
                }
                drop(leaf_tx); // close the stream → coordinator finishes (or errors, on a short-circuit)
                for (gpu, p) in producers.into_iter().enumerate() {
                    p.join()
                        .unwrap_or_else(|_| panic!("base producer thread (gpu {gpu}) panicked"));
                }
                // A base error wins: return it via `?` and DROP the coordinator's join (whose recv
                // then failed) — do not unwrap its panic. Otherwise all leaves were delivered, so
                // the coordinator completed; unwrap its `(leaves, out)` (re-panicking a genuine
                // wrap/fold worker panic on this thread).
                let fold_join = fold_handle.join();
                if let Some(e) = base_err {
                    return Err(e);
                }
                let (leaves, out) = fold_join.expect("streaming leaf fold coordinator panicked");
                overlap_result = Some((leaves, out));
            } else {
                // Drain every tagged base (0..n_shards) into its shard slot.
                for _ in 0..n_shards {
                    let (shard_idx, base) = base_rx.recv().expect("producer hung up early");
                    let base = base?;
                    bases_vec[shard_idx] = Some(make_base(&base));
                }
                for (gpu, p) in producers.into_iter().enumerate() {
                    p.join()
                        .unwrap_or_else(|_| panic!("base producer thread (gpu {gpu}) panicked"));
                }
            }
            Ok(())
        })?;
        if overlap_leaves {
            // The coordinator already wrapped every leaf AND folded the tree during base-proving
            // (Model 1). `overlap_result` carries the ordered leaves + the folded root.
            let (leaves, out) =
                overlap_result.expect("overlap coordinator must have produced (leaves, out)");
            eprintln!(
                "gate-air: pipelined {n_shards_bases} bases proved + leaves wrapped + folded (overlap) in {:.1}s ({} levels)",
                t.elapsed().as_secs_f64(),
                out.n_levels
            );
            (Vec::new(), Some((leaves, out)))
        } else {
            // Dense shard-ordered bases (every slot filled by the drain above).
            let bases: Vec<(Proof<QM31>, GateAirLeafParams)> = bases_vec
                .into_iter()
                .enumerate()
                .map(|(i, b)| {
                    b.unwrap_or_else(|| panic!("base {i} missing after pipelined proving"))
                })
                .collect();
            eprintln!(
                "gate-air: pipelined {n_shards_bases} bases proved in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            (bases, None)
        }
    } else {
        eprintln!("gate-air: building {n_shards_bases} bases (one per shard) ...");
        let t = Instant::now();
        let bases: Vec<(Proof<QM31>, GateAirLeafParams)> =
            shard_bases.iter().map(make_base).collect();
        eprintln!(
            "gate-air: {n_shards_bases} bases built in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        (bases, None)
    };

    // ---- Bottom layer + fold + root verification ----
    // The bases (`(Proof<QM31>, GateAirLeafParams)` in shard order) are materialized above.
    // Prove one standalone leaf per base (`prove_gate_air_leaf`), fold the leaves via
    // `recursive_aggregate_prove_leaves` (level-0 level1-node layer + shared fold-node fold), and unpack via
    // `LeafBottom` / `prove_root_verification_leaves`. Binds `base_nodes` (the fold's height-1
    // inputs), `out`, and `rv` for the shared fingerprint block below.
    let (base_nodes, out, rv) = {
        // Config + precompute already built up front from PUBLIC params (reused whether or not
        // the pipeline overlap wrapped leaves early).
        let agg = leaf_cfg;

        // Leaves + folded root. Under the pipeline overlap (Model 1, "hide the fold behind
        // base-proving"), the coordinator ALREADY wrapped every leaf AND folded the whole tree
        // during base-proving, so we reuse its `(leaves, out)` and SKIP the separate fold. The
        // non-overlap arm wraps the materialized bases (pool-parallel) then runs the classic
        // collect-then-fold `recursive_aggregate_prove_leaves`.
        //
        // POOL-PARALLEL (non-overlap wrap): leaves are independent + deterministic — each
        // proves its own base proof against the immutable shared `cfg`/`agg`, no shared mutable
        // state — so we dispatch one job per leaf across the recursion `pools` (`pools.map`
        // preserves input order, so leaf `i` stays shard `i`); this changes only wall time.
        let (leaves, out): (Vec<TreeProof>, AggregateOutput) =
            if let Some((leaves, out)) = overlapped_fold {
                eprintln!(
                "gate-air: reusing {} leaves + folded root from base-proving overlap ({} levels)",
                leaves.len(),
                out.n_levels
            );
                (leaves, out)
            } else {
                let cfg_ref = &cfg;
                // Build+prove each leaf + fold: proving-utils builds each leaf circuit (via the
                // injected `build` closure), proves it (build+prove+drop per leaf, never all resident),
                // then runs the level-0 level1-node layer + shared fold-node up-tree fold. Leaf `i`
                // stays shard `i` (input order preserved).
                let build = move |(proof, params): (Proof<QM31>, GateAirLeafParams)| {
                    build_gate_air_leaf_circuit::<QM31>(proof, cfg_ref, &params)
                };
                let tf = Instant::now();
                let (leaves, out) =
                    recursive_aggregate_prove_leaves(bases, build, &agg, recursion_pre_ref, &pools);
                eprintln!(
                    "gate-air: {} leaf/leaves proved + folded to root in {:.1}s ({} levels)",
                    leaves.len(),
                    tf.elapsed().as_secs_f64(),
                    out.n_levels
                );
                (leaves, out)
            };
        eprintln!("gate-air: multiverifier fold OK");

        // Root verification: unpack from the raw leaves + self-verify.
        let zk = ZkBlind {
            seed: [7u8; 32],
            n_padding: agg.node_pcs_config.fri_config.n_queries,
        };
        let bottom = LeafBottom {
            leaves: leaves.clone(),
        };
        // Pinned per-N unpacker config (the trusted verify const); the prover's one-shot unpacker
        // tree asserts its committed root == this const's `preprocessed_root`.
        let unpacker_config = pinned_unpacker_config(op, leaves.len());
        let t = Instant::now();
        let rv =
            prove_root_verification_leaves(&out.root, &bottom, &agg, &unpacker_config, Some(zk));
        eprintln!(
                "gate-air: root verification OK in {:.1}s (trace 2^{}, {} leaf outputs unpacked + zk-blinded)",
                t.elapsed().as_secs_f64(),
                rv.trace_log_size,
                rv.leaf_outputs.len()
            );

        // TRUSTED FINAL VERIFY (step 3): check the published proof against the PINNED per-N unpacker
        // config const — the real soundness anchor for the leaf-recursion arm. Its `preprocessed_root`
        // (the canonical unpacker root) PINS every baked child root (leaf tree0 + level1/fold roots),
        // and the per-leaf outputs are taken from `rv.leaf_outputs` (caller-committed), not the proof.
        let n_leaves = rv.leaf_outputs.len();
        eprintln!(
                "gate-air: MEASURE prove_window (precompute->root-verify, excl startup+trusted-verify)={:.1}s",
                t_prove_window.elapsed().as_secs_f64()
            );
        let tv = Instant::now();
        verify_gate_air_root_leaves(&rv, op, n_leaves)
            .expect("trusted gate_air root verification failed (leaf-recursion)");
        eprintln!(
                "gate-air: TRUSTED root verify OK in {:.1}s (canonical unpacker root, {} caller-committed outputs)",
                tv.elapsed().as_secs_f64(),
                n_leaves,
            );
        // The fold's height-1 inputs are the leaves themselves under leaf-recursion (b=1); expose
        // them as `base_nodes` for the shared fingerprint block.
        (leaves, out, rv)
    };
    // Root verification already ran inside the mode branch above (`rv`, `out`, `base_nodes` bound).

    // ---- Recursion byte-identity fingerprint (env-gated observation hook) ----
    // The per-run byte-identity anchor: a light SHA over every leaf/node proof folded into
    // `out.root`, the root proof, and the unpacked leaf outputs. Env-gated by GATE_AIR_RECURSION_FP
    // (off by default; ~free when off) — flip it at RUN time on the production-fast binary so the
    // captured fingerprint reflects production. This print remains the single-run byte-identity anchor
    // (the k=500 gate `32d827a2`).
    if std::env::var("GATE_AIR_RECURSION_FP").is_ok() {
        println!(
            "gate-air: recursion_fingerprint={}",
            fingerprint::recursion_fingerprint(&base_nodes, &out, &rv)
        );
    }
    // The fold completing = every leaf proof verified in-circuit by its parent node; the root
    // verification completing = the root proof verified in-circuit. Both self-verify (always-on).
    println!("gate-air: recursion self-verify (fold+root) OK");

    // Recursion path is self-contained (per-shard base proofs are built above); the
    // monolithic full-`samples` base proof + native verify below are not needed here
    // (and the monolithic trace would OOM a small-VRAM GPU), so return now.
    Ok(())
}

/// Monolithic single-proof path (extracted from `main`'s non-fold `else` branch).
/// Pure code move: builds `rows`/`boundary` (the former `else`-arm), honors `--no-prove`,
/// then runs the single prove+verify path verbatim (param-threaded).
#[allow(clippy::too_many_arguments)]
fn prove_monolithic(
    gates: &[Gate],
    n_gates: usize,
    k: usize,
    samples: usize,
    cases: &[TestCase],
    // Feeds ONLY the CUDA trace-gen glue (device offset buffers); unused on the CPU-only default
    // build (mirrors the former `let rc_lo_index` binding's cfg_attr allow in `main`).
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] rc_lo_index: &RcIndex,
    program: ProgramTable,
    real_rows: usize,
    padded_rows: usize,
    log_n_rows: u32,
    trace_gen_start: Instant,
    no_prove: bool,
) -> Result<()> {
    // HARDENING (Bug-1): the non-fold path materializes ONE global `Vec<Row>` over ALL `cases`
    // (real_rows * size_of::<Row>()). Fine for a single-proof run (one shard, ~24 GB even at
    // 2^28 rows), but a FULL-workload run that reaches here by mistake — e.g. GATE_AIR_FOLD not
    // actually propagating to the process — requests ~2 TB (samples*k*n_gates*88 B) and aborts
    // opaquely (OOM). Fail LOUD instead, pointing at FOLD. The cap is far above any legit single
    // proof and far below the accidental all-shots build, so it never rejects an honest run.
    let build_bytes = (real_rows as u128) * (std::mem::size_of::<Row>() as u128);
    const NONFOLD_BUILD_CAP_BYTES: u128 = 64 << 30; // 64 GiB
    if build_bytes > NONFOLD_BUILD_CAP_BYTES {
        bail!(
            "non-fold build_rows would allocate {} GiB ({} rows * {} B/Row) — too large for a \
             single (non-fold) proof; set GATE_AIR_FOLD=1 for sharded proving, or reduce --samples",
            build_bytes >> 30,
            real_rows,
            std::mem::size_of::<Row>(),
        );
    }
    let build_start = Instant::now();
    let (rows, boundary) = build_rows(gates, cases, k)?;
    let build_elapsed = build_start.elapsed();

    eprintln!(
        "gate-air: samples={} K={} n_gates={} real_rows={} padded_rows={} log_rows={} columns={}",
        samples, k, n_gates, real_rows, padded_rows, log_n_rows, TRACE_COLUMNS
    );
    eprintln!(
        "gate-air: shots simulated and self-checked (final state == y) in {:.3}s",
        build_elapsed.as_secs_f64()
    );

    if no_prove {
        println!(
            "{{\"schema\":\"gate-air-report/v1\",\"samples\":{samples},\"repetitions\":{k},\"n_gates\":{n_gates},\"real_rows\":{real_rows},\"padded_rows\":{padded_rows},\"log_rows\":{log_n_rows},\"trace_columns\":{TRACE_COLUMNS},\"proved\":false,\"self_check\":\"final_state_matches_y\"}}"
        );
        return Ok(());
    }
    // ---- Proving ----
    // Fixed rc-table log-size = RC_LOG; rc_log <= log_n_rows so the .max reduces to log_n_rows (the
    // rc table never raises the FRI/twiddle domain floor).
    let rc_log = RC_LOG;
    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    // SECURE base config (~96-bit) instead of PcsConfig::default() (which is a 13-bit TOY: blowup 1,
    // n_queries 3). leaf_pcs_config sets n_queries/pow_bits/fold_step=4 + lifting = trace+blowup so
    // the base proof passes the privacy-verifier security test. The in-circuit verifier replays this
    // exact config, so its verification circuit now reflects the real (secure) decommitment cost.
    // Base blowup is a sweep knob (env BASE_BLOWUP overrides the default), read off the unified
    // TopologyConfig so this monolithic (non-fold) path resolves the same value as the recursion path.
    let base_blowup: u32 = TopologyConfig::from_env().base_log_blowup;
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
    // Memory-footprint fix: DROP stored polynomial coefficients (store=false) so the prover takes the
    // barycentric OODS path instead of keeping every column's coeffs device-resident (~14GB at 2^24).
    // The proof is built from Merkle/FRI + OODS sampled_values (not coeffs), so it — and the in-circuit
    // verifier's input — is byte-identical (confirmed by the proof fingerprint).

    // Tree 0: preprocessed. The committed order MUST equal preprocessed_column_ids(...) AND be ascending
    // by size (the lifted Merkle sorts columns by length; the in-circuit verifier does not re-sort).
    // Built in canonical order then STABLE-sorted by size, so it matches the ids for any main_log_size.
    // The single-proof path proves exactly once, so there is no shard-invariant reuse to exploit
    // here; it shares the SAME (size-sorted) tree0 column builder as the sharded path so the
    // committed column order/sizes are identical by construction. (N4's PTX module cache + the N3
    // single upload still apply on the cuda path.)
    let t_phase = Instant::now();
    let mut tree_builder = commitment_scheme.tree_builder();
    let pp = build_tree0_columns(
        &program,
        &rows,
        padded_rows,
        log_n_rows,
        n_gates,
        rc_log,
        &boundary,
    );
    tree_builder.extend_evals(to_prover(pp));
    tree_builder.commit(prover_channel);
    eprintln!(
        "gate-air: [phase] preprocessed gen+commit {:.3}s",
        t_phase.elapsed().as_secs_f64()
    );

    // Public claim (empty for gate_air; the boundary is reconstructed by the verifier).
    let public_claim = pack_public_claim(&[]);
    prover_channel.mix_felts(&public_claim);

    // Under `cuda`, the dominant trees (main + interaction) are generated ON the GPU and handed to
    // the CudaBackend commit device-to-device (no host upload) — the unconditional production path.
    // The CPU-tracegen == GPU-tracegen byte-identity the old GATE_AIR_CPU_TRACEGEN A/B arm covered
    // is now T1a/T1b (identical trace ⇒ identical proof). The small columns (multiplicity / program
    // witness / table interactions / preprocessed) always stay on the CPU-generate + upload path.
    // The CPU trace-gen arm survives ONLY on the non-cuda (SimdBackend) build below.

    // ts-ordering range-check supply table (multiplicity counted from the active accesses' `d`
    // lookups). Shard-invariant membership (val) lives in tree0; only the multiplicity is witness.
    let rc_table = build_rc_table(&rows, rc_log);

    // Tree 1: main trace + program witness (op cols+mult) + boundary witness + rc multiplicity.
    let t_phase = Instant::now();
    let small_main = {
        let mut v = generate_program_witness(&program);
        v.extend(generate_boundary_witness(&boundary));
        v.extend(generate_rc_witness(&rc_table));
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    // Holds K1's column-major main-trace device buffer so K4 (interaction) can reuse
    // it instead of re-running K0/K1.
    #[cfg(feature = "cuda")]
    let d_main_cols: cudarc::driver::CudaSlice<u32> = {
        // Device K1: 191 main columns generated on the GPU, fed in as device-resident BaseFieldVecs.
        let (gates_flat, x_states, off_lo, off_hi) =
            gpu_flat_inputs(&gates, cases, &rc_lo_index, &rc_lo_index)?;
        let (main_dev, _lo, d_cols) = gpu_tracegen::gpu_gen_main_trace_device(
            &gates_flat,
            &x_states,
            &off_lo,
            &off_hi,
            k as u32,
            n_gates as u32,
            samples as u32,
            padded_rows,
            log_n_rows,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        eprintln!(
            "gate-air: [phase] main_trace witness gen (GPU K1) {:.3}s",
            t_phase.elapsed().as_secs_f64()
        );
        let mut main_dev = main_dev;
        main_dev.extend(to_prover(small_main));
        tree_builder.extend_evals(main_dev);
        d_cols
    };
    #[cfg(not(feature = "cuda"))]
    {
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        eprintln!(
            "gate-air: [phase] main_trace witness gen {:.3}s",
            t_phase.elapsed().as_secs_f64()
        );
        tree_builder.extend_evals(to_prover(main_trace));
    }
    let t_phase = Instant::now();
    tree_builder.commit(prover_channel);
    eprintln!(
        "gate-air: [phase] tree1 commit (NTT+Merkle) {:.3}s",
        t_phase.elapsed().as_secs_f64()
    );
    // Hold the ~24 GB main-trace device buffer resident from the tree1 commit through K4.
    // See `MainTrace::from_k1` / `gpu_gen_interaction_device`.
    #[cfg(feature = "cuda")]
    let main_k1: gpu_tracegen::MainTrace =
        gpu_tracegen::MainTrace::from_k1(d_main_cols).map_err(|e| anyhow::anyhow!(e))?;

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
        let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.qubitmem);
        gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
    }

    // Interaction traces.
    let t_phase = Instant::now();
    // Device K4 (under `cuda`): the 24 main-interaction columns are generated on the GPU using the
    // REAL drawn `elements` and handed to the commit device-resident (no upload); `claimed_sum`
    // becomes `main_sum`. The CPU (SimdBackend) interaction gen survives only on the non-cuda build.
    #[cfg(feature = "cuda")]
    let main_interaction_device = {
        let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
            &main_k1,
            n_gates as u32,
            padded_rows,
            log_n_rows,
            real_rows as u64,
            (k * n_gates) as u64,
            &elements,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        (cols, claimed)
    };
    // K4 has consumed the main trace; FREE the ~24 GB resident `d_cols` DEVICE buffer NOW (before
    // tree2), not at end-of-prove, and synchronize so the freed memory is reservable by tree2's pool
    // (the device would OOM at 2^25 with it pinned). `free_after_k4` consumes the buffer explicitly
    // (drop alone returns it to the driver but not to tree2's pool without the sync).
    #[cfg(feature = "cuda")]
    main_k1.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
    // GPU path: skip CPU interaction gen (the dominant cost); claimed_sum == CPU main_sum. The CPU
    // interaction cols the on-trace constraint check (T5) needs are rebuilt in the test itself.
    #[cfg(feature = "cuda")]
    let main_sum: SecureField = main_interaction_device.1;
    #[cfg(not(feature = "cuda"))]
    let (main_interaction, main_sum) =
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
    // H_P binding (Fork A): program supply carries internal (-mult, TAG_PROGRAM) + public (+mult,
    // TAG_PROGRAM_PUB) terms, paired => 4 interaction cols (see gen_program_interaction).
    let (program_interaction, program_sum) = gen_program_interaction(&program, &elements.program);
    let (boundary_interaction, boundary_sum) =
        gen_boundary_interaction(&boundary, &elements.qubitmem);
    // rc supply: -multiplicity / combine(TAG_RC, val).
    let (rc_interaction, rc_sum) = {
        let el = elements.rc.clone();
        gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
            el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
        })
    };

    // The direct on-trace AIR-constraint self-check (all 4 components' constraints zero on the
    // committed trace) that GATE_AIR_ASSERT ran is now the `on_trace_constraints_all` test (T5),
    // which drives `assert_main_constraints` / `assert_table_constraints` over a tiny CPU fixture.
    // GATE_AIR_ASSERT_ONLY's "claimed sums net to B + P_pub" check is the existing
    // `shard_claimed_sums_net_to_public` test. Neither is a prove-path hook any more; the always-on
    // cross-check below (which feeds the transcript) stays.

    // Cross-check the committed claimed sums. The base is NOT internally balanced — two public
    // dangling terms surface:
    //   B     = Σ_{shot,addr} ( +1/combine(shot,addr,0,x) − 1/combine(shot,addr,TS_FINAL,y) )  [x/y],
    //   P_pub = Σ_slot mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b)                         [program],
    // so the identity is `main + program + boundary + rc == B + P_pub`. The leaf's `public_logup_sum`
    // equals −(B + P_pub) over its guessed x/y AND program Vars, so the in-circuit balance forces
    // guessed == committed (the x/y recursion binding + the H_P program binding — the same program Vars
    // feed H_P, so it commits to the LogUp-bound program). The preprocessed `shot_id` forbids cross-shot
    // chain mixing; rc demand (main) and supply (rc_sum) cancel. We ALSO recompute the supply sums
    // independently below, so a mistranscribed term is caught before FRI.
    let b_public = boundary_public_term(&boundary, &elements.qubitmem);
    let p_pub = program_public_term(&program, &elements.program);
    if main_sum + program_sum + boundary_sum + rc_sum != b_public + p_pub {
        bail!("claimed sums do not net to the public terms B + P_pub");
    }
    let program_expected = program_claimed_sum(&program, &elements.program);
    if program_sum != program_expected {
        bail!("program claimed sum mismatch");
    }
    let boundary_expected = boundary_public_sum(&boundary, &elements.qubitmem);
    if boundary_sum != boundary_expected {
        bail!("boundary claimed sum mismatch");
    }
    let rc_expected = table_public_sum(&rc_table.multiplicity, &elements.rc, TAG_RC, |i| {
        vec![BaseField::from_u32_unchecked(rc_table.val[i])]
    });
    if rc_sum != rc_expected {
        bail!("rc claimed sum mismatch");
    }

    // Order MUST match the verifier's reconstruction below and build_components.
    let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
    prover_channel.mix_felts(&claimed_sums);

    eprintln!(
        "gate-air: [phase] interaction witness gen+sumcheck {:.3}s",
        t_phase.elapsed().as_secs_f64()
    );
    // Tree 2: interaction (same component order as the claimed sums): main, program, boundary, rc.
    let t_phase = Instant::now();
    let small_interaction = {
        let mut v = program_interaction;
        v.extend(boundary_interaction);
        v.extend(rc_interaction);
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    #[cfg(feature = "cuda")]
    {
        let mut interaction = main_interaction_device.0;
        interaction.extend(to_prover(small_interaction));
        tree_builder.extend_evals(interaction);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut interaction = main_interaction;
        interaction.extend(small_interaction);
        tree_builder.extend_evals(to_prover(interaction));
    }
    tree_builder.commit(prover_channel);
    eprintln!(
        "gate-air: [phase] tree2 commit (NTT+Merkle) {:.3}s",
        t_phase.elapsed().as_secs_f64()
    );

    let components = build_components(
        log_n_rows,
        program.log_size,
        boundary.log_size,
        rc_log,
        &elements,
        main_sum,
        program_sum,
        boundary_sum,
        rc_sum,
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
        fingerprint::emit_proof_fingerprint(&extended);
    }

    // The in-circuit self-verification (standalone monolithic base proof `circuit_verify(...).check()`
    // passes) that GATE_AIR_INCIRCUIT ran is now the `incircuit_self_verify` test (T4). It is a heavy
    // self-check, not prover output, so it left the prove path.

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
    // Supply sums the verifier recomputes (same-process native verify: it has the committed
    // multiplicities + boundary witness + rc table). PHASE-3: the base is no longer internally balanced
    // — the global identity is main + program + boundary + rc == B (the public dangling boundary term),
    // so the main component's claimed sum is v_main = B − v_program − v_boundary − v_rc (the rc demand
    // in main cancels the rc supply).
    // H_P (Fork A): the program claimed sum now carries the internal -mult/TAG_PROGRAM AND the public
    // +mult/TAG_PROGRAM_PUB term (`program_claimed_sum`).
    let v_program = program_claimed_sum(&program, &v_elements.program);
    let v_boundary = boundary_public_sum(&boundary, &v_elements.qubitmem);
    let v_rc = table_public_sum(&rc_table.multiplicity, &v_elements.rc, TAG_RC, |i| {
        vec![BaseField::from_u32_unchecked(rc_table.val[i])]
    });
    let v_b_public = boundary_public_term(&boundary, &v_elements.qubitmem);
    let v_p_pub = program_public_term(&program, &v_elements.program);
    // main_sum = (B + P_pub) − program − boundary − rc. mix_felts ORDER must match the prover's exactly.
    let v_main = (v_b_public + v_p_pub) - v_program - v_boundary - v_rc;
    let v_claimed = vec![v_main, v_program, v_boundary, v_rc];
    verifier_channel.mix_felts(&v_claimed);
    let v_components = build_components(
        log_n_rows,
        program.log_size,
        boundary.log_size,
        rc_log,
        &v_elements,
        v_main,
        v_program,
        v_boundary,
        v_rc,
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
#[allow(dead_code)] // used by the cuda/gpu-cuda interaction paths
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

#[cfg(test)]
mod tests;
