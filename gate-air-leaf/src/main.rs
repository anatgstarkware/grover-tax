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
// Trace-gen backend: ALWAYS SimdBackend. All witness/interaction/preprocessed columns are built
// with cheap per-element CPU column ops (`Col::set`, `BaseColumn::from_simd`, LogupTraceGenerator),
// which require a SimdBackend-layout column. The prover backend may differ (see `ProverBackend`);
// `to_prover` bridges trace-gen columns to the prover backend at the `extend_evals` boundary.
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
// Prover backend (commit + prove_ex): SimdBackend by default; obelyzk GpuBackend under `gpu`
// (model A: CPU trace-gen, GPU commit+prove_ex); device-resident CudaBackend under `cuda`.
// SimdBackend/GpuBackend share trace-gen's column layout (rewrap is identity/transmute-compatible);
// CudaBackend stores device columns, so `to_prover` performs a real host->device upload.
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo::core::proof_of_work::GrindOps;
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
    assert_constraints_on_trace, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator,
    Relation, RelationEntry, TraceLocationAllocator,
};

// In-circuit verifier of the gate_air STARK proof (Design A, Milestone 2).
mod circuit_statement;
// Accumulator-diff scaffold for the GPU constraint kernel (CPU-vs-GPU composition diff).
mod accumulator_diff;
#[cfg(feature = "gpu-cuda")]
mod gpu_tracegen;
mod leaf;

// ----------------------------------------------------------------------------
// Encoding constants
// ----------------------------------------------------------------------------

const N_QUBITS: usize = 512;
const LIMB_BITS: usize = 16;
const N_LIMBS: usize = N_QUBITS / LIMB_BITS; // 32
const STATE_BYTES: usize = N_QUBITS / 8; // 64

// Log-size of the ts-ordering range-check (rc) supply table. The table is a SINGLE block enumerating
// EXACTLY [0, 2^RC_LOG_SIZE) with `val[i] = i` — one lookup per access checks `d ∈ [0, 2^RC_LOG_SIZE)`.
// RC_LOG_SIZE is DYNAMIC (not a fixed const): `RC_LOG_SIZE = ceil(log2(k*n_gates)) = ceil(log2(total_pc))`
// (`rc_log_size(total_pc)` below). This is the smallest power-of-two bound that still contains every
// honest `d`: `d_max = k*n_gates - 1 < 2^RC_LOG_SIZE` (completeness, NO slack) and `2^RC_LOG_SIZE < p`
// (RC_LOG_SIZE <= 25 for k <= 2000, n_gates ~2547, so the field subtraction cannot wrap). Because
// rc_log <= log_n_rows (`k*n_gates <= samples*k*n_gates`), the rc column is never the largest committed
// column, so it does NOT raise the twiddle/FRI domain floor (the `.max(rc_log)` sites reduce to
// log_n_rows). It also sizes the GPU rc-multiplicity histogram.

/// Dynamic rc-table log-size: `ceil(log2(total_pc))` where `total_pc = k*n_gates`. The rc table
/// enumerates exactly `[0, 2^rc_log_size)`, which contains every honest `d = pc - prev_ts` since
/// `d_max = total_pc - 1 < 2^ceil(log2(total_pc))`.
///
/// FLOORED AT `LOG_N_LANES` (the SIMD lane log-count): the rc table is a committed SIMD column and is
/// range-summed by a `LogupTraceGenerator`, both of which require at least one full SIMD lane
/// (`>= 2^LOG_N_LANES` rows). A raw `ceil(log2(total_pc))` below `LOG_N_LANES` (only possible for TINY
/// fixtures with `total_pc < 2^LOG_N_LANES = 16`) underflows those SIMD ops. For ANY real run
/// (`total_pc = k*n_gates >= 2547 >> 16`) `ceil(log2(total_pc)) >= LOG_N_LANES` already, so the floor
/// is a NO-OP — the dynamic-rc_log property (rc_log = ceil(log2(k*n_gates)) <= log_n_rows, no eval-
/// domain inflation) is fully preserved. Widening the range to `[0, 2^LOG_N_LANES)` for a tiny fixture
/// stays SOUND: honest `d <= total_pc - 1` is still contained, and `2^LOG_N_LANES = 16 << p` so the
/// field subtraction still cannot wrap (the forward-DAG / no-stale-read argument holds). The floor
/// lives in this ONE function so the prover and the in-circuit verifier (both call `rc_log_size`)
/// derive the identical R with no separate constant.
pub(crate) fn rc_log_size(total_pc: usize) -> u32 {
    (total_pc as u32)
        .next_power_of_two()
        .ilog2()
        .max(LOG_N_LANES as u32)
        // FIXED rc=25 POLICY (intentional — NOT a stale experiment): pin the rc table to 2^25 for
        // ALL k so every curve point runs at the k=8000 target's rc size (dynamic rc = 25 at
        // k≈8000). Makes the whole curve directly represent the k=8000-relevant per-shard cost.
        // SOUND (a wider rc range still contains every honest d = pc - prev_ts); BYTE-CHANGING vs the
        // dynamic-rc proofs (so fingerprints differ from the dynamic set — expected). No-op at
        // k ≳ 6600 where dynamic rc ≥ 25 already.
        .max(25)
}

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

// Blowup factor for the BASE gate_air proof (the shard / "leaves") now lives in the unified
// `recursive_aggregate::TopologyConfig` (`base_log_blowup`, default `BASE_LOG_BLOWUP`, env
// `BASE_BLOWUP`). The (n_queries, pow_bits) and lifting are derived from it via `leaf::leaf_pcs_config`
// to a ~96-bit-secure config (pow + n_queries*blowup >= 96). Sweep knob: 1/2/3.

const NO_CTRL: u16 = 0xFFFF;

// Program-order timestamp encoding. `ts = pc + 1`, where `pc` is the PREPROCESSED per-shot program
// counter (`gate_pc`, strictly increasing in program order, verifier-pinned). Because `pc` is
// preprocessed the prover CANNOT reorder an address's accesses relative to program order: ts is a
// fixed affine function of the verifier-pinned pc. `ts` is therefore NOT a witness column — it is
// inlined as `pc + 1` everywhere (Yield tuple, range-check reconstruction), and the old PIN
// constraint (`ts == pc*STRIDE + slot`) is removed as vacuous. No per-gate slot is needed: within a
// gate step the (up to 3) accesses hit DISTINCT qubit addresses (a reversible gate cannot use its
// target as a control), so sharing ts = pc+1 across the step never collides two accesses on the same
// per-address chain; two accesses to the SAME address are necessarily in different gate steps
// (distinct pc), so they still get strictly increasing ts. The `+1` keeps the smallest real ts = 1
// (at pc=0) > 0 = the init boundary node's ts, so init's ts=0 tuple stays distinct from every real
// access (do NOT use plain `pc`: that collides pc=0's accesses with the init node). Max real ts per
// shot = (k*n_gates-1) + 1 = k*n_gates; at k=2000, n_gates=2547 this is ~5.1e6 < 2^23, far below
// TS_FINAL = 2^30 and p = 2^31-1 (no aliasing, no wraparound).

// The ts-ordering diff `d = ts - prev_ts - 1 = (pc+1) - prev_ts - 1 = pc - prev_ts` is a SINGLE 25-bit
// column (no limb split). We prove `d ∈ [0, 2^RC_LOG_SIZE)` by a SINGLE LogUp lookup into the dynamic
// rc supply table (which enumerates exactly that range). Because the table is the EXACT range (not a
// padded power-of-two over-bound), the lookup pins `d < 2^RC_LOG_SIZE` with NO slack. Honest
// `d = pc - prev_ts <= pc <= k*n_gates - 1 < 2^RC_LOG_SIZE` (completeness). The absolute bound
// 2^RC_LOG_SIZE - 1 < p = 2^31-1 (RC_LOG_SIZE <= 25) guarantees the field subtraction cannot wrap, so
// a cyclic (stale-read) chain — which would need Σ(ts_i - prev_ts_i) ≡ 0 mod p with each term >= 1 — is
// impossible. See the soundness argument: pc-pinned ts gives program order, the range-check
// `prev_ts < ts` on EVERY access forces the chain to be a forward DAG, and both together defeat the
// reorder. TS_RC_BITS is the hard upper bound on RC_LOG_SIZE (RC_LOG_SIZE = ceil(log2(k*n_gates)) <= 25).
const TS_RC_BITS: usize = 25;

const M31_MODULUS_U32: u32 = (1 << 31) - 1;
const LANE_COUNT: usize = 1 << LOG_N_LANES;

// ----------------------------------------------------------------------------
// Relations
// ----------------------------------------------------------------------------

// ONE shared LogUp relation (single drawn (z,α)). The logical relations are
// distinguished by a distinct id TAG prepended as the first tuple element — this
// matches the in-circuit verifier's single-relation model (circuits_stark_verifier:
// one acc.interaction_elements, relation id as a constant in the tuple). Width =
// widest payload (program = slot,opcode,target,ctrl_a,ctrl_b = 5) + 1 tag = 6.
#[allow(dead_code)]
const GATE_REL_WIDTH: usize = 6;
stwo_constraint_framework::relation!(GateRel, 6);

// Relation id tags (distinct constants; the prover and the in-circuit verifier must agree).
// Qubit-memory encoding: TAG_QUBITMEM is the per-qubit chain-lookup relation (replaces the
// old whole-state TAG_STATE). TAG_RC is the ts-ordering range-check relation: the main component
// looks up `d = ts - prev_ts - 1` as (TAG_RC, d) and the rc supply table supplies (TAG_RC, value)
// for every value in [0, 2^RC_LOG_SIZE).
const TAG_QUBITMEM: u32 = 1;
const TAG_RC: u32 = 2;
const TAG_PROGRAM: u32 = 5;
// H_P program-commitment binding (OPEN #3, Fork A). The program table emits its supply on TWO tags:
//   - TAG_PROGRAM (internal): -mult / combine(TAG_PROGRAM, slot, op, t, a, b) — UNCHANGED, still
//     cancels main's program DEMAND, so base program-consistency is fully preserved.
//   - TAG_PROGRAM_PUB (public): +mult / combine(TAG_PROGRAM_PUB, slot, op, t, a, b) — a DANGLING
//     public term P_pub that surfaces in the committed `program_sum`. The leaf's `public_logup_sum`
//     supplies −P_pub over its GUESSED program Vars (slot preprocessed-pinned, mult pinned to the
//     public shape value samples*k, op/addresses guessed), so the verifier balance forces the leaf's
//     guessed program == the committed (LogUp-bound) program. The leaf then hashes those SAME guessed
//     Vars into H_P = blake(program ‖ nonce). This mirrors the boundary re-key's two-term principle
//     (an internal cancelling term + a public dangling term) using a distinguishing tag instead of a
//     distinguishing ts. A distinct tag is required: reusing TAG_PROGRAM for the +mult term would make
//     it cancel the −mult term (net 0, vacuous). SOUNDNESS CRUX: H_P is bound to the executed program,
//     not free. NOTE: the program supply interaction is CPU-side (`gen_program_interaction`), NOT in
//     the GPU K4 kernel (which only builds the MAIN component); so this change does NOT touch K4 /
//     evaluate_gate_air.cu / the decline-guard (the MAIN component fingerprint is unchanged).
const TAG_PROGRAM_PUB: u32 = 6;

// Phase-3 x/y binding: the boundary's final value `y` is re-keyed to a FIXED public timestamp
// `TS_FINAL` so it surfaces as an UNCONSUMED public LogUp term (see `BoundaryTableEval`). The leaf's
// `public_logup_sum` supplies the matching term over its GUESSED x/y, forcing guessed == committed
// (the recursion's public-output binding). `TS_FINAL` must exceed every real per-address ts
// (real ts in 0..=k*n_gates, tiny) and stay a valid M31, so the public tuples never alias an interior
// chain node or the ts=0 init node. 2^30 < p = 2^31-1 and >> any real ts.
const TS_FINAL: u32 = 1 << 30;

/// The logical relations share the SAME drawn `(z,α)`; the fields are clones of the
/// single `GateRel`, kept under named handles for readable call sites. The tag
/// prepended at each combine is what keeps the relations separate.
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

// ----------------------------------------------------------------------------
// Witness row
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Column layout
// ----------------------------------------------------------------------------
//
// Per row (qubit-memory encoding), one access block = ACCESS_COLS core + 1 rc diff col:
//   is_nop,is_not,is_cnot,is_toffoli                                 (4)
//   target access: addr,prev_ts,v_before, d                         (ACCESS_BLOCK)
//   ctrl_a access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ctrl_b access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ab, fire, delta                                                  (3)
//
// `enabler`, `shot_id`, `pc` are SHARD-INVARIANT POSITIONAL values (enabler = real/padding
// indicator, shot_id = row / (k*n_gates), pc = row % (k*n_gates)). They live in the PREPROCESSED
// tree (tree0) — see `preprocessed_columns_sorted` (gate_enabler / gate_shot_id / gate_pc). The
// prover cannot lie about them (fixed, public, verifier-pinned), and tree0 stays shard-invariant.
//
// The access timestamp `ts` is NOT a witness column: it is the affine `ts = pc + 1` of the
// preprocessed `pc`, inlined at every use site (Yield tuple + range-check reconstruction). The old
// PIN constraint is gone (vacuous). Timestamp ordering is now ONE algebraic constraint + a LogUp
// lookup per active access (soundness-critical):
//   RANGE: active*((pc+1) - prev_ts - 1 - d) = 0 pins the witness `d` column to ts-prev_ts-1 =
//          pc-prev_ts, and `d` is range-checked by a SINGLE LogUp lookup into the dynamic rc supply
//          table (d ∈ [0, 2^RC_LOG_SIZE)). The exact-range table pins d with NO slack, so prev_ts < ts,
//          forcing the chain to be a forward DAG (no stale-read cycle).
// The target's `v_after` is likewise NOT a witness column: it equals `v_before + delta`, inlined at
// its Yield tuple and booleanity constraint. The two controls' written value equals `v` (reads
// propagate the value).
const ACCESS_COLS: usize = 3; // addr, prev_ts, v (the core access cols read by AccessMasks; ts inlined = pc+1)
const ACCESS_BLOCK: usize = ACCESS_COLS + 1; // core cols + the single rc diff col `d`
const TRACE_COLUMNS: usize = 4 + ACCESS_BLOCK + ACCESS_BLOCK + ACCESS_BLOCK + 3; // 4 + 3*4 + 3 = 19

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

/// Build all witness rows for the selected shots and assert each shot's final
/// state matches y_hex.
///
/// PARALLELISM (two-phase, trace bit-identical to the serial version):
///   Phase 1 here parallelizes over SHOTS. Shots are fully independent: shot s
///   owns the contiguous scalar row block `[s*K*n_gates, (s+1)*K*n_gates)`, has
///   its own initial state x_s and its own sequential chain to y_s (gates and K
///   reps are threaded strictly in order *within* a shot). Each shot writes a
///   disjoint `&mut [Row]` slice (`par_chunks_mut`). Returns Err on the first shot
///   whose simulation fails or whose final state mismatches y_hex.
///   Phase 2 — packing the scalar `Vec<Row>` into `PackedM31` words — happens
///   later in `generate_main_trace` / `gen_main_interaction` over PACKED rows,
///   so a shot block being non-16-aligned (2547*K rows) can never cause a
///   packed-word data race: no two threads ever touch the same packed word here
///   (they touch disjoint scalar `Row` cells), and packing reads the finished
///   `Vec<Row>` single-threaded-per-word afterwards.
fn build_rows(gates: &[Gate], cases: &[TestCase], k: usize) -> Result<(Vec<Row>, BoundaryTable)> {
    use rayon::prelude::*;

    let n_gates = gates.len();
    let shot_rows = k * n_gates;
    let total_rows = cases.len() * shot_rows;

    // LOUD completeness guard (release-mode, not debug_assert): the ts-ordering diff
    // `d = ts - prev_ts - 1 = (pc+1) - prev_ts - 1 = pc - prev_ts` is bounded by the max per-shot pc.
    // The largest pc is k*n_gates - 1, and the largest honest d occurs when prev_ts = 0, i.e.
    // d_max = k*n_gates - 1. If that could reach 2^TS_RC_BITS the rc-table range-check would REJECT an
    // honest proof (silent completeness failure at large k), so fail here loudly instead.
    // (No-silent-fallback rule.)
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

/// Simulate a single shot sequentially, filling its row block. The chain
/// (K reps * n_gates gates) is run strictly in order, threading the 512-bit state
/// from x_s to y_s, and the final state is checked against y_hex.
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

    // One access: read predecessor (prev_ts, v_before) from last[addr]; the access ts is `pc + 1`
    // (program-order timestamp, inlined — not stored). Computes d = ts - prev_ts - 1 = pc - prev_ts
    // directly (>= 0 since prev_ts is an earlier program-order ts or the init 0) — the SINGLE rc diff
    // column the rc-table lookup range-checks (proving prev_ts < ts). Returns the filled AccessCols
    // with v = v_before. The caller sets last[addr] to the post-access ts (= pc+1) / value (v_before
    // for reads, v_after for the target write).
    let do_access = |addr: u32, pc: u32, last_ts: &[u32], last_val: &[u32]| -> AccessCols {
        let a = addr as usize;
        let prev_ts = last_ts[a];
        let v_before = last_val[a];
        let ts = pc + 1;
        debug_assert!(
            ts > prev_ts,
            "ts {ts} must exceed prev_ts {prev_ts} (program order)"
        );
        let d = ts - prev_ts - 1; // = pc - prev_ts
                                  // Completeness guard (checked once at build_rows before any access; see build_rows). Here d is
                                  // guaranteed < 2^RC_LOG_SIZE (<= 2^TS_RC_BITS).
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
    };

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

// ----------------------------------------------------------------------------
// ts-ordering range-check (rc) supply table
// ----------------------------------------------------------------------------
//
// The rc table is a SINGLE dynamic 2^RC_LOG_SIZE-row supply table enumerating EXACTLY the range
// [0, 2^RC_LOG_SIZE) with `val[i] = i` (RC_LOG_SIZE = rc_log_size(total_pc) = ceil(log2(k*n_gates))).
// There is NO `pos` selector and NO two-block split — one value column, one lookup per access. The
// membership count is exactly 2^RC_LOG_SIZE (a full power-of-two block), so there are NO padding rows
// beyond the genuine members; if the caller ever pads it stays a genuine member (val=0). The main
// component looks up (TAG_RC, d) for the SINGLE diff `d` of every active access; the table supplies
// -multiplicity / (TAG_RC, value). Because the table enumerates the EXACT range [0, 2^RC_LOG_SIZE)
// (not a padded over-bound), the lookup pins d < 2^RC_LOG_SIZE with NO slack. Completeness:
// d_max = k*n_gates - 1 < 2^RC_LOG_SIZE.

/// Supply table for the ts-ordering range-check. `val` is PREPROCESSED (the table membership,
/// shard-invariant, `val[i] = i`); `multiplicity` is WITNESS (count of real `d` lookups landing on
/// that row). The table has `2^log_size` rows enumerating exactly `[0, 2^log_size)`.
struct RcTable {
    log_size: u32,
    val: Vec<u32>,          // preprocessed: val[i] = i for i in [0, 2^log_size)
    multiplicity: Vec<u32>, // witness
}

impl RcTable {
    fn new(rc_log: u32) -> Self {
        let size = 1usize << rc_log;
        // Single block enumerating exactly [0, 2^rc_log): val[i] = i (every row a genuine member).
        let val = (0..size as u32).collect();
        Self {
            log_size: rc_log,
            val,
            multiplicity: vec![0u32; size],
        }
    }

    /// Count the single `d` lookup of one active access into the multiplicity column. `d` indexes the
    /// table directly since `val[i] = i` (row_of(d) == d).
    #[inline]
    fn count_access(&mut self, a: &AccessCols) {
        self.multiplicity[a.d as usize] += 1;
    }
}

/// Build the rc supply table and its multiplicity column by counting every ACTIVE access's single
/// `d` lookup. An access is active iff its owning gate fires it: the target on every real row, a
/// control iff the opcode uses it. Padding rows (enabler = 0) emit no lookup, so they are skipped.
fn build_rc_table(rows: &[Row], rc_log: u32) -> RcTable {
    let mut table = RcTable::new(rc_log);
    for r in rows {
        if r.enabler == 0 {
            continue;
        }
        // target is active on every real row.
        table.count_access(&r.target);
        // controls active per opcode (a_active = is_cnot + is_toffoli, b_active = is_toffoli).
        if r.is_cnot + r.is_toffoli == 1 {
            table.count_access(&r.ctrl_a);
        }
        if r.is_toffoli == 1 {
            table.count_access(&r.ctrl_b);
        }
    }
    table
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

/// Maps (pos, value) -> a row index in the flattened range-check table. The per-address +1 counter
/// fix (and later the single-`d` rc lookup) removed the range-check LOOKUP; this now survives ONLY so
/// the CUDA trace-gen glue (`gpu_flat_inputs` -> `off_lo`/`off_hi`) can keep its device-buffer layout
/// unchanged (the K1/K4 kernels IGNORE those offsets). The pos/val columns and `row()` accessor are
/// unused by the CPU path (hence `dead_code`). Size is `2^LIMB_BITS`, independent of the now-dynamic
/// rc supply table (this index is per-`pos`-block, sum_pos 2^pos = 2^16 - 1 <= 2^LIMB_BITS rows).
#[allow(dead_code)]
struct RcIndex {
    pos_col: Vec<u32>,
    val_col: Vec<u32>,
    // offset[pos] = first row index for this pos block.
    offset: [usize; LIMB_BITS + 1],
}

impl RcIndex {
    /// `bound(pos)` returns the exclusive value bound for this pos.
    fn build(bound: impl Fn(usize) -> u32) -> Self {
        let size = 1usize << LIMB_BITS;
        let mut pos_col = vec![0u32; size];
        let mut val_col = vec![0u32; size];
        let mut offset = [0usize; LIMB_BITS + 1];
        let mut row = 0usize;
        // `pos` indexes `offset` AND drives `bound(pos)` / the written `pos` value, so the range loop
        // is intentional (no slice to iterate).
        #[allow(clippy::needless_range_loop)]
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
}

fn build_rc_lo() -> RcIndex {
    RcIndex::build(|pos| 1u32 << pos)
}

// ----------------------------------------------------------------------------
// Qubit-memory boundary table (init/final anchoring, per shot)
// ----------------------------------------------------------------------------
//
// PHASE-3 re-keyed boundary. Per (shot, addr) emits on TAG_QUBITMEM two terms:
//   (B) INTERNAL final Use [+1](shot, addr, ts_last, y)  -> cancels main's last chain Yield.
//   (D) PUBLIC   final Yield[-1](shot, addr, TS_FINAL, y) -> re-keys y to a fixed public ts.
// `shot`/`addr` are PREPROCESSED (positional); `x`/`y`/`ts_last` are WITNESS (`x` is now booleanity-
// checked only — main's dangling init Use +1(shot,addr,0,x) at ts=0 carries x publicly). The base
// therefore nets to the PUBLIC term B = Σ(+[0,x] − [TS_FINAL,y]); the leaf's public_logup_sum equals
// −B over guessed x/y, forcing guessed == committed. Untouched addr => ts_last = 0 and (prover data)
// x == y, so B's ts=0 term matches the actual dangling +[0,y]. `shot`/`addr` in the tuple isolate shots.

#[derive(Clone, Copy, Default)]
struct BoundaryRow {
    shot_id: u32, // preprocessed
    addr: u32,    // preprocessed (Seq 0..511, repeating per shot)
    x: u32,       // witness (init value, 1 bit)
    y: u32,       // witness (final value, 1 bit)
    ts_last: u32, // witness (last ts at this addr this shot; 0 if untouched)
}

/// Flat list of `n_shots * N_QUBITS` boundary rows, plus the padded power-of-two size.
struct BoundaryTable {
    rows: Vec<BoundaryRow>,
    log_size: u32,
    n_shots: usize,
}

impl BoundaryTable {
    fn new(n_shots: usize) -> Self {
        let real = n_shots * N_QUBITS;
        let padded = real.next_power_of_two().max(LANE_COUNT);
        let mut rows = vec![BoundaryRow::default(); padded];
        // Pre-fill positional (shot, addr) for real rows so untouched-simulation shots still
        // carry a valid tuple; simulate_shot overwrites the witness fields (x, y, ts_last).
        for (i, r) in rows.iter_mut().enumerate().take(real) {
            r.shot_id = (i / N_QUBITS) as u32;
            r.addr = (i % N_QUBITS) as u32;
        }
        Self {
            rows,
            log_size: padded.ilog2(),
            n_shots,
        }
    }

    /// Mutable per-shot chunks of exactly `N_QUBITS` rows (for parallel simulation fill).
    fn per_shot_mut(&mut self) -> Vec<&mut [BoundaryRow]> {
        let real = self.n_shots * N_QUBITS;
        self.rows[..real].chunks_mut(N_QUBITS).collect()
    }
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
    slot: Vec<u32>,          // preprocessed slot index 0..size
    opcode_scalar: Vec<u32>, // witness
    target: Vec<u32>,        // witness
    ctrl_a: Vec<u32>,        // witness
    ctrl_b: Vec<u32>,        // witness
    multiplicity: Vec<u32>,  // witness: samples*K on real slots, 0 on padding
}

/// Log-size of the program table (one row per gate, padded to a power of two, floored at LANE_COUNT).
/// Pure function of `n_gates` (shard-invariant), so it can be recovered without the table itself.
fn program_log_size(n_gates: usize) -> u32 {
    n_gates.next_power_of_two().max(LANE_COUNT).ilog2()
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

/// Convert a committed `ProgramTable` into the leaf's `ProgramRows` (H_P, Fork A). Same per-slot
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
/// Overridable via GATE_AIR_HP_NONCE="w0,w1" for deterministic byte-identity / oracle runs; otherwise
/// a fixed default (the final proof is zk-blinded at the wrapper, so H_P hiding rests on that blinding;
/// a random nonce per RUN — not per leaf — can be wired later without changing the binding).
fn hiding_nonce() -> [u32; 2] {
    if let Ok(s) = std::env::var("GATE_AIR_HP_NONCE") {
        let parts: Vec<u32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if parts.len() == 2 {
            return [parts[0], parts[1]];
        }
    }
    // Fixed default (deterministic). Distinct-per-run randomness is a future refinement; the nonce is
    // binding-inert, so a fixed value does not affect soundness (only the strength of program hiding).
    [0x1234_5678, 0x9abc_def0]
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
        let two = BaseField::from_u32_unchecked(2);

        // Preprocessed (tree0) columns: SHARD-INVARIANT POSITIONAL values (verifier-pinned).
        //   enabler   = real-row indicator (1 on real rows, 0 on padding)
        //   shot_id   = row / (k*n_gates)   (partitions the per-qubit chains per shot)
        //   pc        = row % (k*n_gates)   (per-shot PROGRAM COUNTER, program order) — the access
        //               timestamp is the inlined affine `ts = pc + 1` (verifier-pinned program order
        //               is what forbids reordering an address's accesses).
        //   pc_in_prog= pc mod n_gates      (the program slot each execution row addresses)
        let enabler = eval.get_preprocessed_column(pp_id("gate_enabler"));
        let shot_id = eval.get_preprocessed_column(pp_id("gate_shot_id"));
        let pc = eval.get_preprocessed_column(pp_id("gate_pc"));
        let pc_in_prog = eval.get_preprocessed_column(pp_id("gate_pc_in_prog"));

        let is_nop = eval.next_trace_mask();
        let is_not = eval.next_trace_mask();
        let is_cnot = eval.next_trace_mask();
        let is_toffoli = eval.next_trace_mask();

        // target access: addr, prev_ts, v_before (ts inlined = pc+1; v_after inlined = v_before+delta).
        let target = access_masks(&mut eval);
        let ctrl_a = access_masks(&mut eval);
        let ctrl_b = access_masks(&mut eval);

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

        // The target's post-gate value `v_after` is NOT a witness column: it equals
        // `v_before + delta` (the pinned gate-apply equality, now inlined). `delta = fire*(1-2*v_before)`
        // is enforced below, so `v_after = v_before + delta` remains a bit and carries the write forward.
        let t_bit = target.v.clone(); // v_before
        let v_after = t_bit.clone() + delta.clone();

        // --- Value booleanity (memory values are 1 bit). The target's written value is the derived
        // `v_after = v_before + delta`; booleanity on it keeps the memory value a bit. ---
        for v in [&target.v, &v_after, &ctrl_a.v, &ctrl_b.v] {
            eval.add_constraint(v.clone() * (v.clone() - one.clone()));
        }

        // --- Gate-apply on the memory values. ---
        let a_bit = ctrl_a.v.clone();
        let b_bit = ctrl_b.v.clone();
        // ab = v_a * v_b.
        eval.add_constraint(ab.clone() - a_bit.clone() * b_bit.clone());
        // fire = is_not + is_cnot*v_a + is_toffoli*ab.
        eval.add_constraint(
            fire.clone()
                - is_not.clone()
                - is_cnot.clone() * a_bit.clone()
                - is_toffoli.clone() * ab.clone(),
        );
        // v_after = v_before XOR fire; delta = v_after - v_before = fire*(1 - 2*v_before). (The
        // `v_after - v_before - delta = 0` equality is now vacuous — v_after is defined as v_before+delta.)
        eval.add_constraint(delta.clone() - fire.clone() + t_bit.clone() * fire.clone() * two);

        // The access timestamp is the affine `ts = pc + 1` of the preprocessed pc, shared by all three
        // accesses of the step (no witness column, no per-access slot). Computed once, inlined below.
        let ts = pc.clone() + one.clone();

        // --- Qubit-memory chain: per active access Use(predecessor) + Yield(successor). ---
        // Target (always active iff enabler): Use(prev_ts, v_before), Yield(ts=pc+1, v_after=v_before+delta).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &target,
            &ts,
            &v_after,
            enabler.clone(),
        );
        // Control C1 (active iff is_cnot+is_toffoli): read propagates value (v_after = v).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &ctrl_a,
            &ts,
            &ctrl_a.v,
            a_active.clone(),
        );
        // Control C2 (active iff is_toffoli).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &ctrl_b,
            &ts,
            &ctrl_b.v,
            b_active.clone(),
        );

        // --- ts-ordering RANGE-CHECK LOOKUPs (emitted here so their relation-batch order is
        // qubitmem-pairs (6 terms), then the 3 single-`d` rc lookups (target, ctrl_a, ctrl_b), then
        // program; finalize-in-pairs folds the trailing 3 rc + 1 program into 2 batches:
        // (rc_t, rc_a) and (rc_b, program). Mirrored exactly by `gen_main_interaction` and MainGate. ---
        add_rc_lookup(&mut eval, &self.elements.rc, &target, enabler.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_a, a_active.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_b, b_active.clone());

        // --- ts-ordering: RANGE-CHECK prev_ts < ts (soundness-critical). ---
        // The old PIN constraint `active*(ts - (pc*TS_STRIDE + slot)) = 0` is GONE: ts is now
        // structurally `pc + 1` (inlined), so the pin is vacuous. Only the RANGE reconstruction
        // remains: active*((pc+1) - prev_ts - 1 - d) = 0 pins the witness `d` to
        // pc - prev_ts, range-checked by the single rc-table lookup above =>
        // d ∈ [0, 2^TS_RC_BITS), i.e. prev_ts < ts. Together with the structurally program-ordered ts
        // and the LogUp chain balance this forces a FORWARD DAG (no stale-read cycle): each read
        // observes the program-order-last write. Inactive accesses (active=0) unconstrained.
        add_ts_range(&mut eval, &ts, &target, enabler.clone());
        add_ts_range(&mut eval, &ts, &ctrl_a, a_active);
        add_ts_range(&mut eval, &ts, &ctrl_b, b_active);

        // --- Program-consistency (use side, +enabler). ---
        // opcode_scalar = is_not + 2*is_cnot + 3*is_toffoli (NOP -> 0). Addresses are the access
        // addr columns (0 for inactive controls, matching the program table's canonical zero).
        let opcode_scalar = is_not.clone()
            + is_cnot.clone() * two
            + is_toffoli.clone() * BaseField::from_u32_unchecked(3);
        let prog_entry = [
            one.clone() * BaseField::from_u32_unchecked(TAG_PROGRAM),
            pc_in_prog.clone(),
            opcode_scalar,
            target.addr.clone(),
            ctrl_a.addr.clone(),
            ctrl_b.addr.clone(),
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

struct AccessMasks<F> {
    addr: F,
    prev_ts: F,
    v: F,
    d: F, // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LOG_SIZE))
}

fn access_masks<E: EvalAtRow>(eval: &mut E) -> AccessMasks<E::F> {
    // ts is NOT a column — it is the inlined `pc + 1`. Per-access columns: addr, prev_ts, v, d.
    let addr = eval.next_trace_mask();
    let prev_ts = eval.next_trace_mask();
    let v = eval.next_trace_mask();
    // d follows v (matches `cell_at`'s per-access column order).
    let d = eval.next_trace_mask();
    AccessMasks {
        addr,
        prev_ts,
        v,
        d,
    }
}

/// Emit the chain Use(predecessor) + Yield(successor) pair for one access, gated by `active`.
/// `ts` is the access's inlined timestamp expression (`pc + 1`, shared across the step); `v_out` is
/// the value written forward (v_after = v_before+delta for the target, v for a control read).
fn add_qubitmem_pair<E: EvalAtRow>(
    eval: &mut E,
    elements: &GateRel,
    shot_id: &E::F,
    a: &AccessMasks<E::F>,
    ts: &E::F,
    v_out: &E::F,
    active: E::F,
) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_QUBITMEM);
    // Use predecessor: +active / (shot, addr, prev_ts, v_before).
    let use_entry = [
        tag.clone(),
        shot_id.clone(),
        a.addr.clone(),
        a.prev_ts.clone(),
        a.v.clone(),
    ];
    eval.add_to_relation(RelationEntry::new(
        elements,
        E::EF::from(active.clone()),
        &use_entry,
    ));
    // Yield successor: -active / (shot, addr, ts=pc+1, v_after).
    let yield_entry = [
        tag,
        shot_id.clone(),
        a.addr.clone(),
        ts.clone(),
        v_out.clone(),
    ];
    eval.add_to_relation(RelationEntry::new(
        elements,
        -E::EF::from(active),
        &yield_entry,
    ));
}

/// Emit the range-check reconstruction for one access, gated by `active`:
///   RANGE: active*(ts - prev_ts - 1 - d) = 0 — pins the witness `d` column to the diff
///          d = ts - prev_ts - 1 = pc - prev_ts. `d` itself is range-checked by the rc-table LOOKUP
///          (`add_rc_lookup`), not here, so d ∈ [0, 2^RC_LOG_SIZE) (prev_ts < ts). `ts` is the inlined
///          `pc + 1` expression.
/// The old PIN constraint is removed (ts is structurally pc+1, so the pin is vacuous). The
/// reconstruction is gated by `active`; the `d` LOOKUP is also gated by `active` (an inactive access
/// emits no rc term). Inactive accesses (active = 0) leave prev_ts/d free.
fn add_ts_range<E: EvalAtRow>(eval: &mut E, ts: &E::F, a: &AccessMasks<E::F>, active: E::F) {
    let one = E::F::one();
    let d = ts.clone() - a.prev_ts.clone() - one;
    eval.add_constraint(active * (d - a.d.clone()));
}

/// Emit the single rc-table range-check LOOKUP for one access, gated by `active`: `d` is looked up as
/// (TAG_RC, d); the rc supply table supplies each in-range value. One term/access (mirrored by
/// `gen_main_interaction` and the in-circuit MainGate).
fn add_rc_lookup<E: EvalAtRow>(eval: &mut E, rc: &GateRel, a: &AccessMasks<E::F>, active: E::F) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
    eval.add_to_relation(RelationEntry::new(
        rc,
        E::EF::from(active),
        &[tag, a.d.clone()],
    ));
}

// ----------------------------------------------------------------------------
// Table FrameworkEvals (supply side of each lookup table)
// ----------------------------------------------------------------------------

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
/// columns. `log_size` is DYNAMIC (= rc_log_size(total_pc) = ceil(log2(k*n_gates))).
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

/// Each preprocessed column paired with its log_size, in a fixed canonical listing order, then
/// STABLE-sorted ascending by size. The committed preprocessed tree MUST be size-sorted (stwo's
/// lifted Merkle sorts each tree's columns by length, and the in-circuit verifier does NOT re-sort
/// the preprocessed tree). The sizes are DYNAMIC: `gate_pc_in_prog` is sized with the main trace,
/// `gate_prog_slot` with the program table, `gate_bnd_*` with the boundary table, and the rc table
/// column `gate_rc_val` at the DYNAMIC `rc_log` (= ceil(log2(k*n_gates)); <= main_log_size).
/// SOUNDNESS: `rc_log` MUST be derived from the public (k, n_gates) both sides trust (never read from
/// the proof) — it sizes the [0,2^rc_log) membership table pinned by the preprocessed root.
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

// ----------------------------------------------------------------------------
// Components bundle
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Trace generation (column-major)
// ----------------------------------------------------------------------------

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
// Boundary supply interaction + public sums
// ----------------------------------------------------------------------------

/// Boundary component interaction trace (PHASE-3 re-keyed): per (shot, addr) row emit the internal
/// final Use[+1](shot, addr, ts_last, y) and the PUBLIC final Yield[-1](shot, addr, TS_FINAL, y) on
/// TAG_QUBITMEM. Two terms per row -> one batch (paired), matching `BoundaryTableEval`.
///
/// GPU TODO (validated on the box, NOT here): the boundary component is CPU-only — the opt-in CUDA
/// kernel (`gate-air-cuda-kernel` / `evaluate_gate_air.cu`) applies ONLY to the gate_air MAIN
/// component (fingerprint `is_gate_air_main`), and the K4 device interaction path computes ONLY the
/// main interaction; `boundary_interaction` here is always built on the host in both the CPU and cuda
/// paths. MAIN is UNCHANGED by this fix, so no GPU kernel edit is required for correctness. BUT this
/// change alters the whole base-proof fingerprint (Phase-2 byte-identity must be re-established on the
/// box), and if a future device-side boundary path is added it must mirror the (B)+(D) re-key +
/// TS_FINAL exactly.
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
    let pack_f = |vec_row: usize, f: &dyn Fn(&BoundaryRow) -> u32| -> PackedM31 {
        PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(f(&bnd.rows[(vec_row << LOG_N_LANES) + lane]))
        }))
    };
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
    let off_lo: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_lo_index.offset[p] as u32)
        .collect();
    let off_hi: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_hi_index.offset[p] as u32)
        .collect();
    Ok((gates_flat, x_states, off_lo, off_hi))
}

// ----------------------------------------------------------------------------
// Base-proof precompute (shard-invariant work built once, reused across shards)
// ----------------------------------------------------------------------------

/// MULTI-SHARD RESIDENT OOM FIX (opt-in `GATE_AIR_POOL_TRIM`, default OFF). At a SHARD BOUNDARY —
/// after shard N's base proof completes and its device buffers are dropped, before shard N+1
/// allocates — trim the CALLING thread's device mem pool so shard N+1 starts from a clean pool and
/// can run FULLY RESIDENT (the fastest path, the alternative to STREAM_MAIN/LOWMEM). `cuda_pool_trim`
/// does `cudaStreamSynchronize(0)` (so shard N's stream-0-ordered `cudaFreeAsync`s have landed) then
/// `cudaMemPoolTrimTo(pool, 0)` for the current device (the per-device `g_mem_pool` macro indexes
/// `cudaGetDevice()`), releasing already-FREE cached segments back to the driver. It is init-guarded
/// (no-op if this device's pool isn't up).
///
/// BYTE-IDENTITY: `cudaMemPoolTrimTo` only returns segments that are already FREE+drained to the OS;
/// it never touches a LIVE allocation (the per-device precompute tree0/twiddles/N3 stay live and
/// untouched), and it changes only WHERE/WHEN device memory is reused, never any committed value.
/// Default OFF => not called => byte-identical to today. Composes with RESIDENT mode (no STREAM_MAIN
/// required — that is the point).
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

/// Shard-invariant base-proof precompute. Built ONCE before the shard loop and shared (by `Arc`)
/// across every shard's base proof, so the work that does not depend on the shard's secret shots is
/// done exactly once instead of N times:
///
/// 1. tree-0 (the 13-column preprocessed commitment) — interpolated + LDE + Merkle-committed ONCE
///    and reused via `CommitmentSchemeProver::commit_tree(MaybeOwned::Borrowed(..))` (re-mixes the
///    SAME root into each shard's fresh channel — transcript unchanged).
/// 2. twiddles — `precompute_twiddles` once, shared by reference.
/// 3. the N1 program table — identical across shards (multiplicity = shots_per_shard*k constant).
/// 4. (cuda) the N3 device-resident gate-list / RcIndex-offset buffers — uploaded ONCE.
///
/// N4 (the GATE_SIM + INTERACTION PTX modules) is a process-level OnceLock cache in gpu_tracegen,
/// not part of this struct.
// `config`/`boundary`/`padded_rows`/`log_n_rows` are read by the cuda `build_device_parts` and by the
// debug/test-only `assert_tree0_matches_rebuild`; in a NON-cuda RELEASE build (assert compiled out)
// they are populated-but-unread, so allow dead_code in exactly that config (warning stays live
// everywhere else to catch genuine dead fields).
#[cfg_attr(not(any(debug_assertions, test, feature = "cuda")), allow(dead_code))]
struct BaseProverPrecompute {
    config: stwo::core::pcs::PcsConfig,
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    /// Shared N1 program table (constant multiplicity across shards).
    program: ProgramTable,
    /// Shard-invariant boundary table SHAPE (positional (shot, addr); witness x/y/ts_last are
    /// per-shard, but the preprocessed columns depend only on the shape, which is shard-invariant).
    boundary: BoundaryTable,
    /// Fixed shard shape (every shard holds `shots_per_shard` shots → same row count).
    padded_rows: usize,
    log_n_rows: u32,
    /// Dynamic rc-table log-size (= ceil(log2(k*n_gates))); shard-invariant. Used to REBUILD tree0
    /// (device n != 0 replica / the debug rebuild check) with the same rc membership sizing.
    rc_log: u32,
    /// (cuda) shape needed to REBUILD the device-resident parts on a producer's device (device != 0).
    #[cfg(feature = "cuda")]
    max_log_size: u32,
    #[cfg(feature = "cuda")]
    n_gates: usize,
    /// (cuda) device-resident N3 inputs uploaded once ON DEVICE 0: gate list + RcIndex lo/hi offsets.
    /// Only the per-shard `x_states` upload remains in `prove_base_shard`. For devices != 0 the
    /// equivalent buffers live in `device_parts` (built lazily per device from the owned inputs).
    #[cfg(feature = "cuda")]
    d_gates: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_hi: cudarc::driver::CudaSlice<u32>,
    // MULTI-GPU ("option A"): the device-resident precompute (tree0 + twiddles + N3 buffers) is bound
    // to the device it was built on. Device 0's copy is the eager fields above (built + soundness-
    // asserted in `new`). For a producer thread on device n != 0, `device_parts()` lazily REBUILDS
    // the same device-resident parts on device n (from the owned host inputs below) and caches them
    // in slot n. tree0 is shard-invariant, so a device-n rebuild is byte-identical to device 0's —
    // this is just per-device REPLICATION of the same precompute, not different data. Slot 0 stays
    // empty (device 0 uses the eager fields); slots 1..MAX filled on demand.
    #[cfg(feature = "cuda")]
    rows0: Vec<Row>,
    #[cfg(feature = "cuda")]
    gates_flat: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_lo: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_hi: Vec<u32>,
    #[cfg(feature = "cuda")]
    device_parts: [std::sync::OnceLock<DevicePrecompute>; MAX_BASE_GPUS],
}

/// Number of base GPUs supported by the per-device precompute cache (matches gpu_tracegen's cap).
#[cfg(feature = "cuda")]
const MAX_BASE_GPUS: usize = 16;

/// (cuda) The device-resident half of the base precompute, bound to ONE device. Built once per
/// device (device 0 eagerly in `BaseProverPrecompute::new`, devices != 0 lazily in `device_parts`).
/// tree0/twiddles are re-derived identically on each device (shard-invariant inputs), so replicating
/// them per device does not change any committed value.
#[cfg(feature = "cuda")]
struct DevicePrecompute {
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: cudarc::driver::CudaSlice<u32>,
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    d_off_hi: cudarc::driver::CudaSlice<u32>,
}

impl BaseProverPrecompute {
    /// Build the precompute from shard 0's shape (`program0`, `rows0`). `rows0`/`program0` are
    /// shard-invariant (see [`build_tree0_columns`]), so the cached tree-0 is valid for every shard.
    ///
    /// SOUNDNESS GUARD: the committed tree-0 root is exactly what each shard's transcript mixes (via
    /// `commit_tree`), so the caller asserts it equals an independently rebuilt shard-0 root + matches
    /// column count/sizes (the load-bearing check in `assert_tree0_matches_rebuild`).
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: stwo::core::pcs::PcsConfig,
        max_log_size: u32,
        program0: ProgramTable,
        rows0: &[Row],
        boundary0: BoundaryTable,
        padded_rows: usize,
        log_n_rows: u32,
        n_gates: usize,
        rc_log: u32,
        #[cfg(feature = "cuda")] gates_flat: &[u32],
        #[cfg(feature = "cuda")] off_lo: &[u32],
        #[cfg(feature = "cuda")] off_hi: &[u32],
    ) -> Result<Self> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        // Scratch pool used only for tree-0's build (the committed tree owns its own polynomials, so
        // the pool can be dropped afterwards; per-shard witness trees use their scheme's own pool).
        let pool = BaseColumnPool::<ProverBackend>::new();

        // Build + commit tree-0 ONCE. `lifting_log_size = None` lets CommitmentTreeProver derive it
        // from the columns' max domain size, exactly as the per-shard `tree_builder().commit()` path
        // does (it uses the same scheme config, which has lifting_log_size from leaf_pcs_config). The
        // store flag is FALSE to match the base proof's barycentric-OODS path (no stored coeffs).
        let cols = build_tree0_columns(
            &program0,
            rows0,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            &boundary0,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            config.fri_config.log_blowup_factor,
            &twiddles,
            false, // store_polynomials_coefficients: barycentric OODS path (matches base proof)
            config.lifting_log_size,
            &pool,
        );

        #[cfg(feature = "cuda")]
        let dev = gpu_tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        #[cfg(feature = "cuda")]
        let d_gates = dev
            .htod_copy(gates_flat.to_vec())
            .map_err(|e| anyhow::anyhow!("htod gates (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_lo = dev
            .htod_copy(off_lo.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_lo (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_hi = dev
            .htod_copy(off_hi.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_hi (precompute): {e}"))?;

        Ok(Self {
            config,
            twiddles,
            tree0,
            program: program0,
            boundary: boundary0,
            padded_rows,
            log_n_rows,
            rc_log,
            #[cfg(feature = "cuda")]
            max_log_size,
            #[cfg(feature = "cuda")]
            n_gates,
            #[cfg(feature = "cuda")]
            d_gates,
            #[cfg(feature = "cuda")]
            d_off_lo,
            #[cfg(feature = "cuda")]
            d_off_hi,
            // Owned rebuild inputs so `device_parts` can replicate the device-resident parts on a
            // producer thread's device (n != 0). Cheap: shard-0 rows + the small N3 flat arrays.
            #[cfg(feature = "cuda")]
            rows0: rows0.to_vec(),
            #[cfg(feature = "cuda")]
            gates_flat: gates_flat.to_vec(),
            #[cfg(feature = "cuda")]
            off_lo: off_lo.to_vec(),
            #[cfg(feature = "cuda")]
            off_hi: off_hi.to_vec(),
            // Slot 0 stays empty (device 0 uses the eager `twiddles`/`tree0`/`d_*` fields above);
            // slots 1..MAX are filled lazily by `device_parts` on first use from each device's thread.
            #[cfg(feature = "cuda")]
            device_parts: [const { std::sync::OnceLock::new() }; MAX_BASE_GPUS],
        })
    }

    /// (cuda) Build the DEVICE-RESIDENT precompute parts (twiddles + tree0 + N3 buffers) on the
    /// CALLING thread's current device, from the shard-invariant host inputs. Same construction as
    /// `new` (byte-identical tree0), factored so device-0 (`new`) and device-n (`device_parts`) share
    /// it. The caller must have already bound its device (via gpu_tracegen::set_base_gpu / cuda_device).
    #[cfg(feature = "cuda")]
    fn build_device_parts(&self) -> Result<DevicePrecompute> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(self.max_log_size + 1 + self.config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let pool = BaseColumnPool::<ProverBackend>::new();
        let cols = build_tree0_columns(
            &self.program,
            &self.rows0,
            self.padded_rows,
            self.log_n_rows,
            self.n_gates,
            self.rc_log,
            &self.boundary,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            self.config.fri_config.log_blowup_factor,
            &twiddles,
            false,
            self.config.lifting_log_size,
            &pool,
        );
        let dev = gpu_tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        let d_gates = dev
            .htod_copy(self.gates_flat.clone())
            .map_err(|e| anyhow::anyhow!("htod gates (device precompute): {e}"))?;
        let d_off_lo = dev
            .htod_copy(self.off_lo.clone())
            .map_err(|e| anyhow::anyhow!("htod off_lo (device precompute): {e}"))?;
        let d_off_hi = dev
            .htod_copy(self.off_hi.clone())
            .map_err(|e| anyhow::anyhow!("htod off_hi (device precompute): {e}"))?;
        Ok(DevicePrecompute {
            twiddles,
            tree0,
            d_gates,
            d_off_lo,
            d_off_hi,
        })
    }

    /// (cuda) The device-resident precompute for the CALLING thread's base GPU ordinal. Device 0
    /// returns the eager fields built in `new` (byte-identical to the single-GPU path). Devices != 0
    /// lazily build + cache their own replica on FIRST use from that device's producer thread. tree0
    /// is shard-invariant, so every device's replica commits the identical root.
    #[cfg(feature = "cuda")]
    fn device_parts(&self) -> DevicePartsRef<'_> {
        let ord = gpu_tracegen::base_gpu_ordinal();
        if ord == 0 {
            return DevicePartsRef {
                twiddles: &self.twiddles,
                tree0: &self.tree0,
                d_gates: &self.d_gates,
                d_off_lo: &self.d_off_lo,
                d_off_hi: &self.d_off_hi,
            };
        }
        let slot = self
            .device_parts
            .get(ord)
            .unwrap_or_else(|| panic!("base gpu ordinal {ord} >= {MAX_BASE_GPUS}"));
        // Build once per device; the build runs on THIS producer thread (already bound to device
        // `ord`). Fatal on failure (matches `new`'s `?` — a broken precompute cannot proceed).
        let parts = slot.get_or_init(|| {
            self.build_device_parts()
                .unwrap_or_else(|e| panic!("device {ord} precompute build failed: {e}"))
        });
        DevicePartsRef {
            twiddles: &parts.twiddles,
            tree0: &parts.tree0,
            d_gates: &parts.d_gates,
            d_off_lo: &parts.d_off_lo,
            d_off_hi: &parts.d_off_hi,
        }
    }
}

/// (cuda) Borrowed view of the device-resident precompute parts (device 0's eager fields or a
/// device-n cached replica), so `prove_base_shard` reads them uniformly regardless of ordinal.
#[cfg(feature = "cuda")]
struct DevicePartsRef<'a> {
    twiddles: &'a stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: &'a stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: &'a cudarc::driver::CudaSlice<u32>,
    d_off_lo: &'a cudarc::driver::CudaSlice<u32>,
    d_off_hi: &'a cudarc::driver::CudaSlice<u32>,
}

/// LOAD-BEARING SOUNDNESS CHECK for the base precompute. Independently rebuilds shard 0's tree-0 the
/// OLD way (a fresh throwaway `CommitmentSchemeProver` + `tree_builder().commit()`) and asserts:
///
/// - the cached tree-0 root == the freshly-rebuilt root (the value mixed into each shard channel),
/// - the cached tree-0 column count == the rebuilt column count,
/// - each cached column's committed domain log_size == the rebuilt column's.
///
/// A mismatch (wrong column order / blowup / lifting / sort) aborts before any reused proof is built.
/// Run on shard 0 only (all shards share the shape). `GATE_AIR_NO_BASE_PRECOMPUTE` skips reuse, so
/// this check is a no-op there (the rebuild path is exercised directly per shard).
///
/// Compiled ONLY in debug or test builds: the runtime caller is `#[cfg(debug_assertions)]` and the CI
/// coverage is `tests::tree0_precompute_matches_rebuild`. Absent from the release binary (its cost is
/// a full duplicate tree0 build), so `--release` pays nothing and stays byte-identical.
#[cfg(any(debug_assertions, test))]
fn assert_tree0_matches_rebuild(pc: &BaseProverPrecompute, rows0: &[Row], n_gates: usize) {
    // Rebuild via the exact old path (fresh scheme/channel; columns from the same builder).
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(
            tree0_max_log_size(
                pc.log_n_rows,
                pc.rc_log,
                pc.program.log_size,
                pc.boundary.log_size,
            ) + 1
                + pc.config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );
    let mut scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(pc.config, &twiddles);
    let cols = build_tree0_columns(
        &pc.program,
        rows0,
        pc.padded_rows,
        pc.log_n_rows,
        n_gates,
        pc.rc_log,
        &pc.boundary,
    );
    let n_cols = cols.len();
    let mut tb = scheme.tree_builder();
    tb.extend_evals(to_prover(cols));
    let mut throwaway_channel = Blake2sM31Channel::default();
    tb.commit(&mut throwaway_channel);

    let rebuilt = &scheme.trees[0];
    // 1. Root equality (the value mixed into every shard's transcript).
    assert_eq!(
        pc.tree0.commitment.root(),
        rebuilt.commitment.root(),
        "base-precompute tree0 root != rebuilt shard-0 root (column order/blowup/lifting mismatch)"
    );
    // 2. Column count.
    assert_eq!(
        pc.tree0.polynomials.len(),
        n_cols,
        "base-precompute tree0 column count != rebuilt"
    );
    assert_eq!(
        rebuilt.polynomials.len(),
        n_cols,
        "rebuilt tree0 column count != expected"
    );
    assert_eq!(
        n_cols, N_PREPROCESSED_COLS,
        "tree0 column count != N_PREPROCESSED_COLS"
    );
    // 3. Per-column committed domain sizes (the lifted-Merkle sort order).
    for (i, (a, b)) in pc
        .tree0
        .polynomials
        .iter()
        .zip(rebuilt.polynomials.iter())
        .enumerate()
    {
        assert_eq!(
            a.evals.domain.log_size(),
            b.evals.domain.log_size(),
            "base-precompute tree0 column {i} size != rebuilt"
        );
    }
    eprintln!(
        "gate-air: base-precompute tree0 root-equality OK ({} cols, root matches rebuilt shard-0)",
        n_cols
    );
}

/// SOUNDNESS (base pp-root pin, step 1): recompute the CANONICAL base gate_air preprocessed (tree0)
/// root at BUILD TIME purely from the trusted PUBLIC config — the program table, `k`, `n_gates`,
/// `shots_per_shard`, the shard-invariant row shape, the boundary layout, `rc_log = rc_log_size(k *
/// n_gates)`, and the base blowup. The canonical base preprocessed (tree0) root a base proof commits,
/// recomputed from the trusted PUBLIC shape so it can be compared against a forgeable proof value.
///
/// It must NOT read `base_extended.proof` (the prover's `commitments[0]` — a forgeable value). tree0
/// is SHARD-INVARIANT (every preprocessed column is positional / shape-derived, see
/// [`build_tree0_columns`]), so ANY shard's rows recompute the same root; the caller passes shard 0's
/// rows (already materialized for the base proof). The build mirrors [`BaseProverPrecompute::new`]'s
/// tree0 path exactly (same columns, blowup, lifting, `store=false`), so the recomputed root equals
/// the honest prover's committed base preprocessed root by construction.
///
/// REBUILD-ASSERT GUARD (debug/test only, mirrors [`assert_tree0_matches_rebuild`]): the tree0 root is
/// re-derived via a fresh `CommitmentSchemeProver` + `tree_builder().commit()` and asserted equal, so
/// a column-order / blowup / lifting / sort divergence aborts before the canonical constant is baked.
/// Compiled out of `--release` (byte-identical, pays nothing).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn canonical_base_preprocessed_root(
    program: &ProgramTable,
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    rc_log: u32,
    boundary: &BoundaryTable,
    pcs_config: stwo::core::pcs::PcsConfig,
) -> circuits::blake::HashValue<SecureField> {
    use circuits::blake::HashValue;
    use stwo::prover::mempool::BaseColumnPool;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::CommitmentTreeProver;

    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let pool = BaseColumnPool::<ProverBackend>::new();
    // Same tree0 build as `BaseProverPrecompute::new` (barycentric-OODS path: store=false).
    let cols = build_tree0_columns(
        program,
        rows,
        padded_rows,
        log_n_rows,
        n_gates,
        rc_log,
        boundary,
    );
    let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
    let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
        polys,
        pcs_config.fri_config.log_blowup_factor,
        &twiddles,
        false,
        pcs_config.lifting_log_size,
        &pool,
    );
    let canonical: HashValue<SecureField> = tree0.commitment.root().into();

    // Rebuild-assert (debug/test only): independent fresh-scheme rebuild must give the same root.
    #[cfg(any(debug_assertions, test))]
    {
        let twiddles_r = ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let mut scheme = CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            pcs_config,
            &twiddles_r,
        );
        let cols_r = build_tree0_columns(
            program,
            rows,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            boundary,
        );
        let mut tb = scheme.tree_builder();
        tb.extend_evals(to_prover(cols_r));
        let mut throwaway_channel = Blake2sM31Channel::default();
        tb.commit(&mut throwaway_channel);
        assert_eq!(
            tree0.commitment.root(),
            scheme.trees[0].commitment.root(),
            "canonical base tree0 root != independent rebuild (column order/blowup/lifting mismatch)"
        );
    }

    canonical
}

/// TRUSTED FINAL VERIFIER (step 3) for the LEAF/R1/R2 recursion. Independently checks the single
/// published root-verification proof `rv` against a CANONICAL unpacker circuit recomputed here from
/// the TRUSTED PUBLIC `(n, config)` — never from any prover-supplied value — closing the base pp-root
/// soundness hole for the leaf/R1/R2 arm:
///
///   1. Recompute the canonical unpacker `CircuitConfig` (`leaf_r1r2_unpacker_verify_config`) — its
///      `preprocessed_root` is the canonical LeafR1R2 unpacker root, built through the SAME shared
///      builder the prover used but with a `NoValue` witness, so it is byte-identical to the honest
///      proof's preprocessed root. The child roots (leaf tree0, R1/R2, short leaf-node / short root)
///      are BAKED as constants in that circuit, so this canonical root PINS them: a proof whose
///      unpacker baked a forged child root has a different preprocessed root and is REJECTED here. All
///      those child roots are canonical config-derived values already on `config` (never prover
///      reported), so — unlike the base-fanning path — no externally-supplied base-node roots are
///      needed.
///   2. `verify_circuit` the proof against that canonical config, with the CALLER-COMMITTED outputs
///      (`rv.leaf_outputs`, the per-leaf output digests) as public data — NOT values lifted from the
///      proof.
///
/// `zk_n_padding` must equal the prover's blinding `n_padding` (the root PCS `n_queries`) so the
/// recomputed circuit's component sizes match; `None` for an unblinded (test) proof. Modeled on
/// `privacy_circuit_verify::verify_recursive_circuit`.
fn verify_gate_air_root_leaves(
    rv: &recursive_aggregate::RootVerificationOutput,
    config: &recursive_aggregate::AggregateConfig,
    n: usize,
    log_blowup_factor: u32,
    zk_n_padding: Option<usize>,
) -> anyhow::Result<()> {
    use circuit_verifier::verify::{verify_circuit, CircuitPublicData};
    use recursive_aggregate::leaf_r1r2_unpacker_verify_config;

    // (1) Canonical unpacker verify config recomputed from trusted public params (NoValue), sharing
    //     the prover's builder ⇒ byte-identical preprocessed root/shape. Every baked child root (leaf
    //     tree0, R1/R2, short variants) is a canonical value already on `config`.
    let verify_config =
        leaf_r1r2_unpacker_verify_config(n, config, log_blowup_factor, zk_n_padding);

    // (2) Verify the published proof with the CALLER-COMMITTED per-leaf outputs.
    let output_values: Vec<SecureField> = rv.leaf_outputs.iter().flatten().copied().collect();
    verify_circuit(
        verify_config,
        rv.proof.clone(),
        CircuitPublicData { output_values },
    )
    .map(|_| ())
    .map_err(|e| anyhow::anyhow!("trusted gate_air root verification failed (leaf/R1/R2): {e}"))
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

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

    // GPU soundness gates (then exit): GATE_AIR_GPU_TEST selects which kernel to validate
    // byte-identically against the CPU reference. "k4" → K4 LogUp interaction; anything else
    // (e.g. "1"/"k1") → K1 main trace-gen + histograms.
    #[cfg(all(feature = "gpu-cuda", feature = "diag"))]
    if let Ok(which) = std::env::var("GATE_AIR_GPU_TEST") {
        if which == "k4" {
            gpu_tracegen::k4_byte_identity(&gates, cases, k, &rc_lo_index, &rc_lo_index)
                .map_err(|e| anyhow::anyhow!(e))?;
        } else {
            gpu_tracegen::k1_byte_identity(&gates, cases, k, &rc_lo_index, &rc_lo_index)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        return Ok(());
    }

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
    let (rows, boundary) = if fold_active {
        (Vec::<Row>::new(), BoundaryTable::new(0))
    } else {
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
        let (rows, boundary) = build_rows(&gates, cases, k)?;
        let build_elapsed = build_start.elapsed();

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
        (rows, boundary)
    };

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
    if fold_active {
        use circuit_statement::gate_air_components;
        use circuits::blake::HashValue;
        use circuits::ivalue::NoValue;
        use circuits::wrappers::U32Wrapper;
        use circuits_stark_verifier::proof::{Proof, ProofConfig};
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use leaf::{
            build_recursion_precompute, derive_aggregate_config, prove_gate_air_leaf,
            GateAirLeafParams,
        };
        use recursive_aggregate::AggregateOutput;
        use recursive_aggregate::{
            prove_root_verification_leaves, recursive_aggregate_prove_leaves,
            recursive_aggregate_prove_leaves_streaming, AggregateConfig, LeafBottom, PoolSet,
            RecursionPrecompute, TopologyConfig, TreeProof, ZkBlind,
        };
        use stwo::core::fields::qm31::QM31;
        use stwo::core::proof::ExtendedStarkProof;
        use stwo::core::utils::MaybeOwned;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;

        // The base-proof tuple `prove_base_shard` returns. Named so the pipeline producer can send
        // it over a channel; `prove_ex` yields `ExtendedStarkProof<MC::H>` with
        // `MC::H = Blake2sMerkleHasher`, so this is backend-independent (cuda vs simd).
        type BaseShardOutput = (
            ExtendedStarkProof<Blake2sMerkleHasher>,
            Vec<SecureField>,
            u64,
            u32,
            u32,
            u32,
            Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
            u32,
        );

        // All FREE topology params in one place, honoring the existing env sweep knobs (BASE_BLOWUP,
        // BASE_FAN_ARITY, GATE_AIR_SHARD_SHOTS). Defaults reproduce the current production values, so
        // this is a byte-identical no-op. Threaded through the derive/prove calls below; the base
        // blowup, fold arity, base-fan arity, and shots-per-shard are all read off it.
        let topo = TopologyConfig::from_env();
        let recursion_log_blowup = topo.recursion_log_blowup;
        let leaf_log_blowup = topo.leaf_log_blowup;

        // Shard partition: equal-sized shards of `shots_per_shard` shots; ragged final shard is
        // padded (below) so all shards share the leaf circuit shape.
        let shots_per_shard: usize = topo.shots_per_shard.min(samples);
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
        // `prove_base_shard` takes the shared base precompute by reference (`Some` = reuse, `None` =
        // the legacy rebuild-tree0-per-shard fallback, selected by `GATE_AIR_NO_BASE_PRECOMPUTE`).
        // tree0/twiddles/program (and the cuda N3 device buffers) are shard-invariant, so on the
        // reuse path they come from `pc` and only the per-shard witness is built here.
        let prove_base_shard = |precompute: Option<&BaseProverPrecompute>,
                                shard_cases: &[TestCase]|
         -> Result<_> {
            let shard_samples = shard_cases.len();
            let (rows, boundary) = build_rows(&gates, shard_cases, k)?;
            let real_rows = rows.len();
            let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
            let log_n_rows = padded_rows.ilog2();
            // Dynamic rc-table log-size = ceil(log2(k*n_gates)); <= log_n_rows (never raises the floor).
            let rc_log = rc_log_size(k * gates.len());
            let max_log_size = tree0_max_log_size(
                log_n_rows,
                rc_log,
                program_log_size(gates.len()),
                boundary.log_size,
            );
            let base_blowup: u32 = topo.base_log_blowup;
            let config = leaf::leaf_pcs_config(max_log_size, base_blowup);

            // Twiddles: shared by reference from the precompute, else built fresh per shard.
            let owned_twiddles = if precompute.is_none() {
                Some(ProverBackend::precompute_twiddles(
                    CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                        .circle_domain()
                        .half_coset,
                ))
            } else {
                None
            };
            // MULTI-GPU: the device-resident precompute parts (twiddles/tree0/N3) for THIS thread's
            // device. On device 0 these are the eager fields (byte-identical to before); on device
            // n != 0 they are the lazily-built per-device replica. `None` (no-precompute fallback)
            // leaves `dp` None and uses `owned_twiddles` / the per-shard rebuild, unchanged.
            #[cfg(feature = "cuda")]
            let dp = precompute.map(|pc| pc.device_parts());
            #[cfg(feature = "cuda")]
            let twiddles = match &dp {
                Some(dp) => dp.twiddles,
                None => owned_twiddles.as_ref().unwrap(),
            };
            #[cfg(not(feature = "cuda"))]
            let twiddles = match precompute {
                Some(pc) => &pc.twiddles,
                None => owned_twiddles.as_ref().unwrap(),
            };
            // N1 program table: shared from the precompute (constant multiplicity across shards), else
            // rebuilt. `shard_samples == shots_per_shard` for every shard, so the multiplicity matches.
            let owned_program = if precompute.is_none() {
                Some(build_program_table(&gates, shard_samples, k))
            } else {
                None
            };
            let program: &ProgramTable = match precompute {
                Some(pc) => &pc.program,
                None => owned_program.as_ref().unwrap(),
            };

            let prover_channel = &mut Blake2sM31Channel::default();
            let channel_salt = 0u32;
            prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
            config.mix_into(prover_channel);
            let mut commitment_scheme = CommitmentSchemeProver::<
                ProverBackend,
                Blake2sM31MerkleChannel,
            >::new(config, twiddles);
            // commitment_scheme.set_store_polynomials_coefficients();  // disabled: barycentric OODS path

            // Tree 0: reuse the precomputed commitment (re-mix the SAME root into THIS shard's
            // channel via `commit_tree` — no NTT/Merkle rebuild), else rebuild it the old way. Under
            // multi-GPU the reused tree0 is THIS device's replica (`dp.tree0`); its root is identical
            // to device 0's (shard-invariant), so the transcript mix is unchanged.
            #[cfg(feature = "cuda")]
            match &dp {
                Some(dp) => {
                    commitment_scheme.commit_tree(MaybeOwned::Borrowed(dp.tree0), prover_channel);
                }
                None => {
                    let pp = build_tree0_columns(
                        program,
                        &rows,
                        padded_rows,
                        log_n_rows,
                        n_gates,
                        rc_log,
                        &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }
            #[cfg(not(feature = "cuda"))]
            match precompute {
                Some(pc) => {
                    commitment_scheme.commit_tree(MaybeOwned::Borrowed(&pc.tree0), prover_channel);
                }
                None => {
                    // Old path: build the (size-sorted) preprocessed columns, then interpolate + LDE +
                    // Merkle-commit them inline (the shard-invariant work this precompute eliminates).
                    let pp = build_tree0_columns(
                        program,
                        &rows,
                        padded_rows,
                        log_n_rows,
                        n_gates,
                        rc_log,
                        &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }

            let public_claim = pack_public_claim(&[]);
            prover_channel.mix_felts(&public_claim);

            #[cfg(feature = "cuda")]
            let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

            // ts-ordering range-check supply table (multiplicity counted from active-access lookups).
            let rc_table = build_rc_table(&rows, rc_log);

            // Tree 1: main trace + program witness + boundary witness + rc multiplicity.
            let small_main = {
                let mut v = generate_program_witness(program);
                v.extend(generate_boundary_witness(&boundary));
                v.extend(generate_rc_witness(&rc_table));
                v
            };
            let mut tree_builder = commitment_scheme.tree_builder();
            // Holds K1's column-major main-trace device buffer so K4 (interaction) can
            // reuse it instead of re-running K0/K1. `None` on the CPU path.
            #[cfg(feature = "cuda")]
            let mut d_main_cols: Option<cudarc::driver::CudaSlice<u32>> = None;
            #[cfg(feature = "cuda")]
            if gpu_tracegen {
                // N3: gate list + RcIndex offsets are shard-invariant. On the reuse path they are
                // already device-resident in the precompute (uploaded once); only this shard's
                // `x_states` is uploaded here. On the fallback path they're uploaded per shard.
                let mut x_states = Vec::with_capacity(shard_cases.len() * N_LIMBS);
                for c in shard_cases {
                    let bytes =
                        hex::decode(&c.x_hex).context("decoding x_hex for GPU trace-gen")?;
                    x_states.extend_from_slice(&state_to_limbs(&bytes));
                }
                let (main_dev, _qd, _lo, _hi, d_cols) = match &dp {
                    // Multi-GPU: use THIS device's N3 buffers (device 0's eager d_*, or the per-device
                    // replica) — feeding device-0 buffers to a device-n kernel would be an illegal
                    // cross-device access.
                    Some(dp) => gpu_tracegen::gpu_gen_main_trace_device_d(
                        dp.d_gates,
                        &x_states,
                        dp.d_off_lo,
                        dp.d_off_hi,
                        k as u32,
                        n_gates as u32,
                        shard_samples as u32,
                        padded_rows,
                        log_n_rows,
                    ),
                    None => {
                        let (gates_flat, _x, off_lo, off_hi) =
                            gpu_flat_inputs(&gates, shard_cases, &rc_lo_index, &rc_lo_index)?;
                        gpu_tracegen::gpu_gen_main_trace_device(
                            &gates_flat,
                            &x_states,
                            &off_lo,
                            &off_hi,
                            k as u32,
                            n_gates as u32,
                            shard_samples as u32,
                            padded_rows,
                            log_n_rows,
                        )
                    }
                }
                .map_err(|e| anyhow::anyhow!(e))?;
                let mut main_dev = main_dev;
                main_dev.extend(to_prover(small_main));
                tree_builder.extend_evals(main_dev);
                d_main_cols = Some(d_cols);
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

            // Hold the ~24 GB main-trace device buffer resident from the tree1 commit through K4.
            // See `MainTrace::from_k1`.
            #[cfg(feature = "cuda")]
            let mut main_k1: Option<gpu_tracegen::MainTrace> = match d_main_cols.take() {
                Some(d_cols) => {
                    Some(gpu_tracegen::MainTrace::from_k1(d_cols).map_err(|e| anyhow::anyhow!(e))?)
                }
                None => None,
            };

            let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
            prover_channel.mix_u64(interaction_pow_nonce);
            let elements = LookupElements::draw(prover_channel);

            #[cfg(feature = "cuda")]
            {
                gate_air_cuda_kernel::register();
                let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.qubitmem);
                gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
            }

            // Interaction traces.
            #[cfg(feature = "cuda")]
            let main_interaction_device = if gpu_tracegen {
                let main = main_k1
                    .as_ref()
                    .expect("K1 main-trace buffer must exist on the GPU path");
                let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
                    main,
                    n_gates as u32,
                    padded_rows,
                    log_n_rows,
                    real_rows as u64,
                    (k * n_gates) as u64,
                    &elements,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                Some((cols, claimed))
            } else {
                None
            };
            // K4 done: FREE the ~24 GB resident `d_cols` DEVICE buffer NOW (before tree2), not at
            // end-of-shard, and synchronize so tree2's pool can reserve it. `free_after_k4` consumes
            // the buffer explicitly.
            #[cfg(feature = "cuda")]
            if let Some(m) = main_k1.take() {
                m.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
            }
            #[cfg(feature = "cuda")]
            let (main_interaction, main_sum) = if let Some((_, claimed)) = &main_interaction_device
            {
                (Vec::new(), *claimed)
            } else {
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements)
            };
            #[cfg(not(feature = "cuda"))]
            let (main_interaction, main_sum) =
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
            // H_P binding (Fork A): program supply now carries an internal (-mult, TAG_PROGRAM) AND a
            // public (+mult, TAG_PROGRAM_PUB) term, paired into one batch => still 4 interaction cols.
            let (program_interaction, program_sum) =
                gen_program_interaction(program, &elements.program);
            let (boundary_interaction, boundary_sum) =
                gen_boundary_interaction(&boundary, &elements.qubitmem);
            // rc supply: -multiplicity / combine(TAG_RC, val).
            let (rc_interaction, rc_sum) = {
                let el = elements.rc.clone();
                gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                    el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
                })
            };

            // Phase-3 x/y binding + H_P program binding (Fork A): the base is NOT internally balanced.
            //   - boundary re-keys y to TS_FINAL, leaving B = Σ(+[0,x] − [TS_FINAL,y]);
            //   - program supply adds a public P_pub = Σ mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b)
            //     (its internal -mult/TAG_PROGRAM term cancels main's program demand).
            // So the base's claimed sums net to B + P_pub (not 0). The leaf's public_logup_sum supplies
            // −B (over guessed x/y) AND −P_pub (over guessed program Vars), so the verifier balance
            // forces guessed x/y == committed AND guessed program == committed. rc demand (main) and rc
            // supply (rc_sum) cancel, contributing 0. (stwo's native verify does NOT require
            // Σ claimed_sums == 0; this is a prover self-check.)
            // Prover self-check (DEBUG-ONLY, compiled out in --release): the base's claimed LogUp sums
            // must net to the public terms B + P_pub. Pure tripwire — `b_public`/`p_pub` feed nothing
            // downstream (only `claimed_sums` below is mixed), so gating changes no committed value
            // (release byte-identical). Per-shard cost (two small-table scans) is thus paid only in
            // debug. CI coverage: `tests::shard_claimed_sums_net_to_public` (CPU/Simd fixture); this
            // runtime check additionally guards each run's real secret shot data in debug builds.
            #[cfg(debug_assertions)]
            {
                let b_public = boundary_public_term(&boundary, &elements.qubitmem);
                let p_pub = program_public_term(program, &elements.program);
                if main_sum + program_sum + boundary_sum + rc_sum != b_public + p_pub {
                    bail!("shard claimed sums do not net to the public terms B + P_pub");
                }
            }

            let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
            prover_channel.mix_felts(&claimed_sums);

            // Tree 2: interaction (same component order as the claimed sums): main, program, boundary, rc.
            let small_interaction = {
                let mut v = program_interaction;
                v.extend(boundary_interaction);
                v.extend(rc_interaction);
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
                boundary.log_size,
                rc_log,
                &elements,
                main_sum,
                program_sum,
                boundary_sum,
                rc_sum,
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
            let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];
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

        // ---- Base-proof precompute (shard-invariant work built ONCE) ----
        // tree0 (preprocessed commitment) + twiddles + the N1 program table + (cuda) the N3 device
        // buffers are identical across every shard's base proof, so build them once here and share
        // the `Arc` by reference into each `prove_base_shard` call (the Arc matters for lifetime
        // across the pipeline producer thread). `GATE_AIR_NO_BASE_PRECOMPUTE` falls back to the old
        // rebuild-per-shard path (the A/B control arm); the cache build also asserts tree0's root
        // equals an independent shard-0 rebuild (the load-bearing soundness gate).
        let no_base_precompute = std::env::var("GATE_AIR_NO_BASE_PRECOMPUTE").is_ok();

        // ---- PARALLEL PRECOMPUTES (pure scheduling; byte-identical) ----
        // Two independent, heavy precomputes run CONCURRENTLY on separate threads and join before any
        // proving:
        //   (1) GPU  — `BaseProverPrecompute::new` (tree0 + twiddles + N1 + cuda N3), ~7.5s.
        //   (2) CPU  — the recursion config + `RecursionPrecompute` (heavy `CircuitPrecompute`s on the
        //              SimdBackend), ~9.2s.
        // They share NO data: (2) is a pure function of PUBLIC params (shard-0 shape + the
        // independently-recomputed canonical base preprocessed root) and never reads `pc`; (1) never
        // reads the recursion config. Running them in parallel collapses the segment from ~16.7s
        // serial to ~max(7.5, 9.2) ≈ 9.2s. Everything the CPU closure captures (`gates`,
        // `shard_case_sets`, `topo`, scalars) is `Send`/`Sync`, borrowed read-only via `thread::scope`.
        // (Formerly the config was derived from shard 0's PROVED base, so shard 0 had to be proved
        // eagerly on this thread; that dependency is gone, so both precomputes now just parallelize and
        // ALL shards are proved uniformly in the producer/base loop below.)
        // PROVE-WINDOW timer: starts HERE (after startup — fixture load, shot-sim, CUDA init — which is
        // excluded), spans the two precomputes ‖ + base proving + fold + root verification, and STOPS
        // before the trusted verify (a soundness self-check, not prover output). This is the
        // SP1-comparable prover time; the process WALL additionally includes startup + trusted verify.
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
        let shape_program0 = build_program_table(&gates, shots_per_shard, k);
        let (shape_rows0, _shape_boundary0) = build_rows(&gates, &shard_case_sets[0], k)?;
        let shape_real_rows0 = shape_rows0.len();
        let shape_padded_rows0 =
            shape_real_rows0.next_power_of_two().max(1 << (LOG_N_LANES + 2));
        let shape_log_n_rows0 = shape_padded_rows0.ilog2();
        let shape_rc_log0 = rc_log_size(k * n_gates);
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
        // leaf/R1/R2 `CircuitPrecompute` (all built from `NoValue` shapes, where the guessed root's
        // value is ignored). We therefore give `shape_params` a byte-irrelevant ZERO placeholder here.
        let placeholder_base_pp_root: HashValue<SecureField> =
            HashValue(std::array::from_fn(|_| U32Wrapper::new_unsafe(SecureField::zero())));
        let shape_params = GateAirLeafParams {
            main_log_size: shape_log_n_rows0,
            program_log_size: shape_program0.log_size,
            boundary_log_size,
            preprocessed_root: placeholder_base_pp_root,
            boundary: shape_boundary_pairs,
            total_pc: (k * n_gates) as u32,
            program: leaf_program.clone(),
            nonce: leaf_nonce,
        };
        // Derive the recursion config + its up-front `RecursionPrecompute`. The heavy
        // `CircuitPrecompute` builds happen HERE (in parallel with the GPU precompute).
        let t_cfg = Instant::now();
        let (leaf_cfg, recursion_pre) = {
            let (agg, shapes) = derive_aggregate_config(
                &cfg,
                &shape_params,
                topo.fold_arity,
                recursion_log_blowup,
                leaf_log_blowup,
            );
            let pre = build_recursion_precompute(shapes);
            eprintln!(
                "gate-air: leaf/R1/R2 config + precompute built up front in {:.1}s (node target qm31_ops={})",
                t_cfg.elapsed().as_secs_f64(),
                agg.node_target_padding_sizes.qm31_ops,
            );
            (agg, pre)
        };
            Ok((cfg, leaf_cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce))
        });

            // --- MAIN THREAD (GPU): base precompute build. ---
            let base_precompute: Option<std::sync::Arc<BaseProverPrecompute>> =
                if no_base_precompute {
                    eprintln!("gate-air: base precompute DISABLED (GATE_AIR_NO_BASE_PRECOMPUTE) — rebuilding tree0/twiddles/program/N3 per shard");
                    None
                } else {
                    let t_pc = Instant::now();
                    // Shard 0's shape (every shard shares it: equal shot count, same program + k).
                    let program0 = build_program_table(&gates, shots_per_shard, k);
                    let (rows0, boundary0) = build_rows(&gates, &shard_case_sets[0], k)?;
                    let real_rows0 = rows0.len();
                    let padded_rows0 = real_rows0.next_power_of_two().max(1 << (LOG_N_LANES + 2));
                    let log_n_rows0 = padded_rows0.ilog2();
                    let rc_log0 = rc_log_size(k * n_gates);
                    let max_log_size0 = tree0_max_log_size(
                        log_n_rows0,
                        rc_log0,
                        program0.log_size,
                        boundary0.log_size,
                    );
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
                    // Load-bearing soundness gate: cached tree0 root == independent shard-0 rebuild.
                    // DEBUG-ONLY (compiled out in --release): this rebuilds tree0 (interpolate + Merkle), a
                    // costly once-per-run duplicate. Gated so the release hot path pays nothing — the release
                    // proof is byte-identical (the check computes nothing that feeds the proof). The invariant
                    // has CI coverage via `tests::tree0_precompute_matches_rebuild` (CPU/Simd fixture); this
                    // runtime call additionally guards the REAL per-run data (and, in a cuda debug build, the
                    // cuda tree0 path the test cannot exercise).
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
            let (
                cfg,
                leaf_cfg,
                recursion_pre,
                boundary_log_size,
                leaf_program,
                leaf_nonce,
            ) = cpu_build
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

        // PIPELINE opt-in: with GATE_AIR_PIPELINE set AND >1 shard, overlap GPU base-proving
        // (producer) with CPU leaf-wrap + streaming fold (consumer). The producer proves ALL shards
        // 0..n_shards on dedicated thread(s) while the consumer wraps + folds in shard order; NO shard
        // is proved eagerly on this thread (the recursion config no longer depends on a proved base).
        // With the flag unset (default) the existing sequential path below runs UNCHANGED.
        //
        // SOUNDNESS GATE (pending, on-box, NOT run here — laptop only): the streaming path must
        // yield a recursion_fingerprint BYTE-IDENTICAL to the sequential path for the same fixture
        // (e.g. k1-n4 samples=4 GATE_AIR_SHARD_SHOTS=2, GATE_AIR_PIPELINE set vs unset). That
        // one-flag diff is the trust gate before this path is used in anger.
        let pipeline = env_flag_default_on("GATE_AIR_PIPELINE") && n_shards > 1;

        // MULTI-GPU base proving ("option A"): the number of GPUs to prove base shards on
        // concurrently, in ONE process. Default 1 = today's single-producer, single-GPU path (device
        // 0 throughout) — byte-identical. With GATE_AIR_BASE_GPUS=G>1 (and the pipeline active) the
        // producer side spawns up to G threads, thread n does cuda_set_device(n) once and proves its
        // assigned shards on GPU n, all feeding the SAME ordered consumer channel. Clamped to the
        // visible device count (fail-loud if the box has fewer than requested) and to the number of
        // producer shards. Only meaningful with the CUDA backend; on non-cuda builds it stays 1.
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
                shard_bases.push(prove_base_shard(base_precompute_ref, shard_cases)?);
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
        // must be BYTE-IDENTICAL with the precompute ON (default) vs OFF (GATE_AIR_NO_BASE_PRECOMPUTE).
        // This fingerprints the base proofs DIRECTLY (the layer this change touches), so the A/B check
        // runs on the laptop without the memory-heavy recursion fold. Requires the sequential path
        // (all shard bases proved up front); prints + exits before the fold. (The downstream
        // leaves/fold/root are a deterministic function of these base proofs, so base byte-identity
        // implies the full recursion_fingerprint is identical too — validated on-box at #10.)
        if !pipeline && std::env::var("GATE_AIR_BASE_PROOF_HASH").is_ok() {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(b"gate-air/base-shard-proofs/serde/v1");
            hasher.update(format!("n_shards={n_shards}").as_bytes());
            for (s, base) in shard_bases.iter().enumerate() {
                let bytes =
                    serde_json::to_vec(&base.0.proof).expect("serialize base shard StarkProof");
                hasher.update(format!("shard[{s}].proof=").as_bytes());
                hasher.update(&bytes);
                hasher.update(format!("shard[{s}].claim={:?}", base.1).as_bytes());
                hasher.update(format!("shard[{s}].nonce={}", base.2).as_bytes());
            }
            let digest = hasher.finalize();
            let mode = if no_base_precompute {
                "BASE_PRECOMPUTE_OFF"
            } else {
                "BASE_PRECOMPUTE_ON"
            };
            println!(
                "gate-air: base_proof_fingerprint[{mode}]={}",
                hex::encode(digest)
            );
            return Ok(());
        }

        // The recursion config + `shape_params` + `cfg` + `b` are already built UP FRONT (above, from
        // public params) and reused for every leaf (the "one trusted leaf_preprocessed_root for all
        // leaves" invariant). Nothing shape-related is derived from the proved bases here.

        // Partition the machine so independent leaf proves run concurrently (POOL_THREADS sweet spot).
        // MEMORY/THROUGHPUT TRADEOFF: each concurrent pool holds one large in-flight `TreeProof`
        // (FRI layers + Merkle decommits, multi-GB) while it proves a leaf/fold node, so the number
        // of pools == the number of proofs in flight == the multiplier on peak host RAM. On a big box
        // (192 vCPU / g4) we want K = cores/24 pools for real leaf/fold concurrency; on a memory-limited
        // box (12 vCPU, ~40-85GB) K collapses to 1 (12/24 -> 0 -> max(1)), which is what we want: a
        // single in-flight fold proof, no RAM multiplier. That already prevents the N>=4 concurrency OOM.
        // Default 24 is box-measured (BOX_VALIDATION_LOG #22 + overlap: pt=24 beat pt=48 on wall).
        //
        // The remaining waste: `PoolSet::new(1, 24)` would still spawn 24 rayon OS threads (each with
        // a large default stack, and 24 > 12 cores oversubscribes) for a pool that only ever runs one
        // proof at a time. When a single pool is used we therefore clamp its worker count to the
        // actual core count, so we don't reserve big thread stacks it can't schedule. This is a
        // pure thread-count change (rayon fan-out over NTT/Merkle/FRI is order-independent) and does
        // not touch any proof value -> byte-identical output.
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

        // Materialize every base in shard order. Leaf proving + the fold + root-verify happen AFTER
        // this.
        //
        // BASE↔LEAF OVERLAP (pipeline only), "hide the fold behind base-proving" (Model 1):
        // each leaf verifies exactly ONE base, so as bases arrive from the GPU producer channel we feed
        // them into `recursive_aggregate_prove_leaves_streaming`, which WRAPS each base into a leaf AND
        // folds the whole tree (level-0 leaf→R1 layer + shared R2 up-tree fold) PROGRESSIVELY on the CPU
        // `pools` — so GPU base-proving overlaps with BOTH the CPU leaf-wrap AND the fold (no separate
        // fold tail). The leaf/R1/R2 config + precompute are already built up front (`leaf_cfg` /
        // `recursion_pre`). The non-pipeline path keeps the "materialize all bases, then wrap" flow.
        // Pipeline yields the already-folded `(leaves, AggregateOutput)`; the non-pipeline path yields
        // `bases`.
        let overlap_leaves = pipeline;
        let (bases, overlapped_fold): (
            Vec<(Proof<QM31>, GateAirLeafParams)>,
            Option<(Vec<TreeProof>, AggregateOutput)>,
        ) = if pipeline {
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
                // PRODUCERS: `g` threads, one per GPU. Producer-shard `s` (s in 0..n_shards) is proved
                // on gpu `s % g` — so shard 0 → gpu 0, shard 1 → gpu 1, …, wrapping mod g. Shard 0
                // lands on gpu 0 (0 % g == 0), the same device it used to be proved on eagerly, so its
                // base proof is byte-identical. This keys the round-robin on the SHARD INDEX (matching
                // the shard→gpu spread the box expects). For g == 1 every shard maps to gpu 0 (single
                // producer), unchanged. Each producer binds its device ONCE via `set_base_gpu(gpu)`
                // (thread-local ordinal + cudaSetDevice), so its `device_parts()` returns ITS device's
                // replica and its pool/trim act on ITS device. Shared borrows are moved (by-ref);
                // `prove_base_shard`/`shard_case_sets` are read-only, `base_precompute_ref` is Copy.
                let prove_base_shard_ref = &prove_base_shard;
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
                                    prove_base_shard_ref(
                                        base_precompute_ref,
                                        &shard_case_sets_ref[shard_idx],
                                    )
                                });
                                #[cfg(not(feature = "gpu-cuda"))]
                                let r = prove_base_shard_ref(
                                    base_precompute_ref,
                                    &shard_case_sets_ref[shard_idx],
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
                    // coordinator owns the single wrap+R1+R2 pool and folds progressively; the injected
                    // `wrap` closure (make_base + prove_gate_air_leaf) runs INSIDE its pool workers, so
                    // GPU base-proving overlaps BOTH the leaf-wrap and the fold. Leaf i = shard i
                    // (index-tagged), byte-identical to the sequential wrap+fold.
                    let agg = &leaf_cfg;
                    let pre = recursion_pre_ref;
                    let make_base_ref = &make_base;
                    let pools_ref = &pools;
                    // The wrap closure the coordinator runs per leaf (heavy — runs inside a pool
                    // worker via the crate's `pool.install`). Keeps the crate leaf-agnostic.
                    let wrap = move |base: BaseShardOutput| -> TreeProof {
                        let (proof, params) = make_base_ref(&base);
                        prove_gate_air_leaf(proof, cfg_ref, &params, agg, pre)
                    };
                    // The coordinator reads `(shard_idx, base)`; a small forward loop on THIS thread
                    // pulls tagged producer results and forwards the Ok bases, so a base `Err` still
                    // short-circuits via `?` (as the non-overlap drain does). The coordinator runs on
                    // its own scope thread so it folds while this thread keeps draining producers.
                    let (leaf_tx, leaf_rx) = std::sync::mpsc::channel::<(usize, BaseShardOutput)>();
                    let fold_handle = scope.spawn(move || {
                        recursive_aggregate_prove_leaves_streaming(
                            leaf_rx, n_shards, wrap, agg, pre, pools_ref,
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
                        p.join().unwrap_or_else(|_| {
                            panic!("base producer thread (gpu {gpu}) panicked")
                        });
                    }
                    // A base error wins: return it via `?` and DROP the coordinator's join (whose recv
                    // then failed) — do not unwrap its panic. Otherwise all leaves were delivered, so
                    // the coordinator completed; unwrap its `(leaves, out)` (re-panicking a genuine
                    // wrap/fold worker panic on this thread).
                    let fold_join = fold_handle.join();
                    if let Some(e) = base_err {
                        return Err(e);
                    }
                    let (leaves, out) =
                        fold_join.expect("streaming leaf fold coordinator panicked");
                    overlap_result = Some((leaves, out));
                } else {
                    // Drain every tagged base (0..n_shards) into its shard slot.
                    for _ in 0..n_shards {
                        let (shard_idx, base) = base_rx.recv().expect("producer hung up early");
                        let base = base?;
                        bases_vec[shard_idx] = Some(make_base(&base));
                    }
                    for (gpu, p) in producers.into_iter().enumerate() {
                        p.join().unwrap_or_else(|_| {
                            panic!("base producer thread (gpu {gpu}) panicked")
                        });
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
        // `recursive_aggregate_prove_leaves` (level-0 R1 layer + shared R2 fold), and unpack via
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
                let (leaves, out): (Vec<TreeProof>, AggregateOutput) = if let Some((leaves, out)) =
                    overlapped_fold
                {
                    eprintln!(
                            "gate-air: reusing {} leaves + folded root from base-proving overlap ({} levels)",
                            leaves.len(),
                            out.n_levels
                        );
                    (leaves, out)
                } else {
                    let cfg_ref = &cfg;
                    let agg_ref = &agg;
                    let pre_ref = recursion_pre_ref;
                    let tg = Instant::now();
                    let jobs: Vec<_> = bases
                        .into_iter()
                        .enumerate()
                        .map(|(i, (proof, params))| {
                            move || {
                                let tl = Instant::now();
                                let leaf =
                                    prove_gate_air_leaf(proof, cfg_ref, &params, agg_ref, pre_ref);
                                eprintln!(
                                    "gate-air: MEASURE t_leaf[{i}]={:.3}s",
                                    tl.elapsed().as_secs_f64()
                                );
                                leaf
                            }
                        })
                        .collect();
                    let leaves: Vec<TreeProof> = pools.map(jobs);
                    eprintln!(
                        "gate-air: {} leaf/leaves proved in {:.1}s",
                        leaves.len(),
                        tg.elapsed().as_secs_f64()
                    );
                    // Fold: level-0 R1 layer over the leaves + shared R2 up-tree fold.
                    let tf = Instant::now();
                    let out = recursive_aggregate_prove_leaves(
                        leaves.clone(),
                        &agg,
                        recursion_pre_ref,
                        &pools,
                    );
                    eprintln!(
                        "gate-air: folded to root in {:.1}s ({} levels)",
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
                let t = Instant::now();
                let rv = prove_root_verification_leaves(
                    &out.root,
                    &bottom,
                    &agg,
                    recursion_log_blowup,
                    Some(zk),
                );
                eprintln!(
                    "gate-air: root verification OK in {:.1}s (trace 2^{}, {} leaf outputs unpacked + zk-blinded)",
                    t.elapsed().as_secs_f64(),
                    rv.trace_log_size,
                    rv.leaf_outputs.len()
                );

                // TRUSTED FINAL VERIFY (step 3): check the published proof against a canonical unpacker
                // circuit recomputed here from the trusted public `(n, config)` — the real soundness
                // anchor for the LeafR1R2 arm. The canonical unpacker root PINS every baked child root
                // (canonical leaf tree0 root + R1/R2/short roots), and the per-leaf outputs are taken
                // from `rv.leaf_outputs` (caller-committed), not the proof.
                let n_leaves = rv.leaf_outputs.len();
                eprintln!(
                    "gate-air: MEASURE prove_window (precompute->root-verify, excl startup+trusted-verify)={:.1}s",
                    t_prove_window.elapsed().as_secs_f64()
                );
                let tv = Instant::now();
                verify_gate_air_root_leaves(
                    &rv,
                    &agg,
                    n_leaves,
                    recursion_log_blowup,
                    Some(agg.node_pcs_config.fri_config.n_queries),
                )
                .expect("trusted gate_air root verification failed (leaf/R1/R2)");
                eprintln!(
                    "gate-air: TRUSTED root verify OK in {:.1}s (canonical unpacker root, {} caller-committed outputs)",
                    tv.elapsed().as_secs_f64(),
                    n_leaves,
                );
                // The fold's height-1 inputs are the leaves themselves under LeafR1R2 (b=1); expose
                // them as `base_nodes` for the shared fingerprint block.
                (leaves, out, rv)
        };
        // Root verification already ran inside the mode branch above (`rv`, `out`, `base_nodes` bound).

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
            hasher.update(
                format!(
                    "n_base_nodes={} n_levels={}",
                    base_nodes.len(),
                    out.n_levels
                )
                .as_bytes(),
            );
            for (i, bn) in base_nodes.iter().enumerate() {
                hasher.update(format!("base_node[{i}].proof={:?}", bn.proof).as_bytes());
                hasher.update(
                    format!("base_node[{i}].pp_root={:?}", bn.preprocessed_root).as_bytes(),
                );
                hasher.update(format!("base_node[{i}].outs={:?}", bn.output_values).as_bytes());
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
            println!(
                "gate-air: recursion_fingerprint[{mode}]={}",
                hex::encode(digest)
            );
            println!("gate-air: recursion self-verify (fold+root) OK [{mode}]");
        }

        // Recursion path is self-contained (per-shard base proofs are built above); the
        // monolithic full-`samples` base proof + native verify below are not needed here
        // (and the monolithic trace would OOM a small-VRAM GPU), so return now.
        return Ok(());
    }

    // ---- Proving ----
    // Dynamic rc-table log-size = ceil(log2(k*n_gates)); rc_log <= log_n_rows so the .max reduces to
    // log_n_rows (the rc table never raises the FRI/twiddle domain floor).
    let rc_log = rc_log_size(k * n_gates);
    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    // SECURE base config (~96-bit) instead of PcsConfig::default() (which is a 13-bit TOY: blowup 1,
    // n_queries 3). leaf_pcs_config sets n_queries/pow_bits/fold_step=4 + lifting = trace+blowup so
    // the base proof passes the privacy-verifier security test. The in-circuit verifier replays this
    // exact config, so its verification circuit now reflects the real (secure) decommitment cost.
    // Base blowup is a sweep knob (env BASE_BLOWUP overrides the default), read off the unified
    // TopologyConfig so this monolithic (non-fold) path resolves the same value as the recursion path.
    let base_blowup: u32 = recursive_aggregate::TopologyConfig::from_env().base_log_blowup;
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
    // the CudaBackend commit device-to-device (no host upload), UNLESS GATE_AIR_CPU_TRACEGEN=1 forces
    // the legacy CPU-build + upload path. The small columns (multiplicity / program witness / table
    // interactions / preprocessed) always stay on the CPU-generate + upload path.
    #[cfg(feature = "cuda")]
    let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

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
    // it instead of re-running K0/K1. `None` on the CPU path.
    #[cfg(feature = "cuda")]
    let mut d_main_cols: Option<cudarc::driver::CudaSlice<u32>> = None;
    #[cfg(feature = "cuda")]
    if gpu_tracegen {
        // Device K1: 191 main columns generated on the GPU, fed in as device-resident BaseFieldVecs.
        let (gates_flat, x_states, off_lo, off_hi) =
            gpu_flat_inputs(&gates, cases, &rc_lo_index, &rc_lo_index)?;
        let (main_dev, _qd, _lo, _hi, d_cols) = gpu_tracegen::gpu_gen_main_trace_device(
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
        // MEM PROBE 1: right after K1 completes, before tree1 commit. d_main is resident here.
        stwo::stwo_cuda::cuda_mem_probe("PROBE1_after_K1");
        // HYPOTHESIS TEST (GATE_AIR_TRIM_AFTER_K1=1, default OFF): does returning pool-cached-freed
        // K1 scratch to the driver free enough contiguous space for tree1 commit to proceed? One-shot
        // sync + cudaMemPoolTrimTo(0). Only touches ALREADY-FREED pool segments; live d_main untouched.
        if std::env::var("GATE_AIR_TRIM_AFTER_K1").is_ok() {
            unsafe {
                stwo::stwo_cuda::bindings::cuda_pool_trim();
            }
            stwo::stwo_cuda::cuda_mem_probe("PROBE1b_after_K1_trim");
        }
        let mut main_dev = main_dev;
        main_dev.extend(to_prover(small_main));
        tree_builder.extend_evals(main_dev);
        d_main_cols = Some(d_cols);
    } else {
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        eprintln!(
            "gate-air: [phase] main_trace witness gen (CPU) {:.3}s",
            t_phase.elapsed().as_secs_f64()
        );
        tree_builder.extend_evals(to_prover(main_trace));
    }
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
    // MEM PROBE 2: immediately before the tree1 commit loop (first tree1-column NTT/alloc). This is
    // the driver+pool state ENTERING the LDE loop that OOMs at 2^25.
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE2_before_tree1");
    let t_phase = Instant::now();
    tree_builder.commit(prover_channel);
    eprintln!(
        "gate-air: [phase] tree1 commit (NTT+Merkle) {:.3}s",
        t_phase.elapsed().as_secs_f64()
    );
    // Hold the ~24 GB main-trace device buffer resident from the tree1 commit through K4.
    // See `MainTrace::from_k1` / `gpu_gen_interaction_device`.
    #[cfg(feature = "cuda")]
    let mut main_k1: Option<gpu_tracegen::MainTrace> = match d_main_cols.take() {
        Some(d_cols) => {
            Some(gpu_tracegen::MainTrace::from_k1(d_cols).map_err(|e| anyhow::anyhow!(e))?)
        }
        None => None,
    };

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
    // Device K4 (under `cuda`, unless GATE_AIR_CPU_TRACEGEN): the 24 main-interaction columns are
    // generated on the GPU using the REAL drawn `elements` and handed to the commit device-resident
    // (no upload); `claimed_sum` becomes `main_sum`. `main_interaction` (CPU SimdBackend cols) is
    // only materialized on the CPU path or when GATE_AIR_ASSERT needs it for `assert_main_constraints`.
    #[cfg(feature = "cuda")]
    let main_interaction_device = if gpu_tracegen {
        let main = main_k1
            .as_ref()
            .expect("K1 main-trace buffer must exist on the GPU path");
        let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
            main,
            n_gates as u32,
            padded_rows,
            log_n_rows,
            real_rows as u64,
            (k * n_gates) as u64,
            &elements,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Some((cols, claimed))
    } else {
        None
    };
    // K4 has consumed the main trace; FREE the ~24 GB resident `d_cols` DEVICE buffer NOW (before
    // tree2), not at end-of-prove, and synchronize so the freed memory is reservable by tree2's pool
    // (the device would OOM at 2^25 with it pinned). `free_after_k4` consumes the buffer explicitly
    // (drop alone returns it to the driver but not to tree2's pool without the sync).
    #[cfg(feature = "cuda")]
    if let Some(m) = main_k1.take() {
        m.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
    }
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
        eprintln!("gate-air: GATE_AIR_ASSERT program OK");

        // qubit-memory boundary table.
        let bnd_pp = generate_boundary_preprocessed(&boundary);
        let bnd_wit = generate_boundary_witness(&boundary);
        assert_table_constraints(
            boundary.log_size,
            &bnd_pp,
            &bnd_wit,
            &boundary_interaction,
            boundary_sum,
            BoundaryTableEval {
                log_size: boundary.log_size,
                elements: elements.qubitmem.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT boundary OK");

        // ts-ordering range-check supply table.
        let rc_pp = generate_rc_preprocessed(&rc_table);
        let rc_wit = generate_rc_witness(&rc_table);
        assert_table_constraints(
            rc_table.log_size,
            &rc_pp,
            &rc_wit,
            &rc_interaction,
            rc_sum,
            RcTableEval {
                log_size: rc_table.log_size,
                elements: elements.rc.clone(),
            },
        );
        eprintln!("gate-air: GATE_AIR_ASSERT rc OK (all components satisfied on trace)");

        // Validate the prover cross-check here too, so the assert path exercises the full LogUp
        // balance (incl. the rc demand/supply cancellation) without FRI. Skip the heavy FRI prove via
        // GATE_AIR_ASSERT_ONLY.
        if std::env::var("GATE_AIR_ASSERT_ONLY").is_ok() {
            let b_public = boundary_public_term(&boundary, &elements.qubitmem);
            let p_pub = program_public_term(&program, &elements.program);
            if main_sum + program_sum + boundary_sum + rc_sum != b_public + p_pub {
                bail!("ASSERT_ONLY: claimed sums do not net to the public terms B + P_pub");
            }
            eprintln!("gate-air: GATE_AIR_ASSERT_ONLY cross-check OK (skipping FRI prove)");
            return Ok(());
        }
    }

    // Cross-check the committed claimed sums. PHASE-3 + H_P (Fork A): the base is NOT internally
    // balanced — two public dangling terms surface in the committed claimed sums:
    //   B     = Σ_{shot,addr} ( +1/combine(shot,addr,0,x) − 1/combine(shot,addr,TS_FINAL,y) )  [x/y],
    //   P_pub = Σ_slot mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b)                         [program],
    // so the global identity is now
    //     main_sum + program_sum + boundary_sum + rc_sum == B + P_pub.
    // The leaf's `public_logup_sum` equals −(B + P_pub) over the guessed x/y AND guessed program Vars,
    // so the in-circuit verifier balance `public_logup_sum + Σ claimed_sums == 0` forces guessed ==
    // committed (the recursion x/y binding AND the H_P program binding). We ALSO cross-check the supply
    // sums independently (program, boundary) against a direct recomputation, so a mistranscribed supply
    // term is caught before FRI.
    //
    // SOUNDNESS: per-input binding (x_i -> y_i) — x via main's dangling init term at ts=0, y via the
    // boundary's public TS_FINAL yield; the leaf pins BOTH through B. Program binding — the program
    // table's public +mult/TAG_PROGRAM_PUB term surfaces as P_pub; the leaf reconstructs it over its
    // guessed (slot, op, t, a, b, mult) and hashes those SAME Vars into H_P, so H_P commits to the
    // LogUp-bound program (its internal -mult/TAG_PROGRAM term still cancels main's demand, so program
    // consistency is unchanged). The preprocessed `shot_id` in every QubitMem tuple forbids cross-shot
    // chain mixing. Chain acyclicity is the degree-1 equality on the ts range-check. The rc demand
    // (main) and supply (rc_sum) cancel exactly, contributing 0 to the net.
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
    // MEM PROBE 4: after free_after_k4 (main trace freed), immediately BEFORE the tree2 commit whose
    // NTT (ifft.cu) is the 2^26 OOM site. This is the true free entering the failing tree2-eval alloc.
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE4_before_tree2_commit");
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

    // MEM PROBE 5: composition entry — resident set going INTO prove_ex (composition/OODS/quotient/
    // FRI). At 2^25 this was the whole-prove high-water (LAST_STATE composition-entry ~26.6 GiB); the
    // in-composition spike itself is caught by the smi sampler. This is the binding-phase reading.
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE5_composition_entry");
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
    // MEM PROBE 6: after prove_ex returns — post-composition/FRI settled state (pool_reserved reflects
    // the pool's high-water reservation across composition/quotient, the phases with no earlier probe).
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE6_after_prove_ex");

    // ---- Full-proof byte-identity fingerprint (read-only), gated by GATE_AIR_PROOF_HASH ----
    // Deterministic SHA-256 over the serde-serialized ExtendedStarkProof (commitments,
    // sampled_values, decommitments, FRI, proof_of_work, claimed sums via sampled_values + aux).
    // The CPU/SimdBackend run is the golden oracle; a `--features cuda` run on the same
    // fixture+samples must print the SAME hex. See P5_GPU_CONSTRAINT_SCOPE.md Deliverable 2 §2.1.
    #[cfg(feature = "diag")]
    if std::env::var("GATE_AIR_PROOF_HASH").is_ok() {
        emit_proof_fingerprint(&extended);
    }

    // ---- In-circuit verification (Design A, Milestone 2), gated by GATE_AIR_INCIRCUIT ----
    if std::env::var("GATE_AIR_INCIRCUIT").is_ok() {
        use circuit_statement::{gate_air_components, GateAirStatement};
        use circuits::blake::HashValue;
        use circuits::context::{Context, TraceContext};
        use circuits::ivalue::NoValue;
        use circuits::ops::Guess;
        use circuits_stark_verifier::proof::{empty_proof, ProofConfig};
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use circuits_stark_verifier::verify::verify as circuit_verify;

        let n_pp = N_PREPROCESSED_COLS;
        let cfg = ProofConfig::new(
            &gate_air_components::<NoValue>(),
            n_pp,
            &config,
            INTERACTION_POW_BITS,
        );
        let pp_root: HashValue<SecureField> = extended.proof.commitments[0].into();
        let mut boundary_xy = Vec::with_capacity(cases.len());
        for case in cases {
            let x = state_to_limbs(&hex::decode(&case.x_hex)?);
            let y = state_to_limbs(&hex::decode(&case.y_hex)?);
            boundary_xy.push((x, y));
        }
        let total_pc = (n_gates * k) as u32;
        let boundary_log_size = boundary.log_size;
        let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];

        // NoValue circuit shape (the reference the real assignment is checked against).
        let novalue_circuit = {
            let empty = empty_proof(&cfg);
            let mut nv = Context::<NoValue>::default();
            let pv = empty.guess(&mut nv);
            let stmt = GateAirStatement::<NoValue>::new(
                &mut nv,
                log_n_rows,
                program.log_size,
                boundary_log_size,
                pp_root.clone(),
                boundary_xy.clone(),
                total_pc,
                program_rows_from_table(&program),
                hiding_nonce(),
            );
            circuit_verify(&mut nv, &pv, &cfg, &stmt);
            // Mirror the real leaf: build H_P (consumes the nonce + program Vars) so the self-check
            // exercises the H_P sub-circuit. Its terminal hash Vars are not re-hashed here (the real
            // leaf feeds them into the output hash), so mark them used.
            let h_p = stmt.compute_h_p(&mut nv);
            for w in h_p.iter() {
                nv.mark_as_unused(*w.get());
            }
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
            boundary_log_size,
            pp_root,
            boundary_xy,
            total_pc,
            program_rows_from_table(&program),
            hiding_nonce(),
        );
        circuit_verify(&mut ctx, &pv, &cfg, &stmt);
        let h_p = stmt.compute_h_p(&mut ctx);
        for w in h_p.iter() {
            ctx.mark_as_unused(*w.get());
        }
        let ctx = ctx.finalize(true);
        novalue_circuit
            .check(ctx.values())
            .expect("gate-air: in-circuit verify FAILED");
        eprintln!("gate-air: in-circuit verify OK");
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
#[cfg(feature = "diag")]
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

// ============================================================================
// CI coverage for the two prover self-checks that were moved off the release hot path
// (`assert_tree0_matches_rebuild` and the per-shard claimed-sums balance). These run on the CPU
// (`ProverBackend == SimdBackend`) over a tiny self-consistent fixture, so they exercise the same
// invariants `cargo test` while the release binary compiles the runtime checks out. NOTE: the CPU
// path is exercised here — the debug-gated runtime calls additionally guard the cuda path and each
// run's real secret data.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::qm31::QM31;

    /// A tiny self-consistent fixture: an all-NOP circuit (every gate leaves the state unchanged, so
    /// `y == x`), `k` reps, `n_shots` shots. NOP gates still ACCESS their target qubit, so the
    /// memory-chain / rc-table / boundary / program machinery is fully exercised. Distinct targets
    /// per gate keep the per-address ts chains simple. Returns (gates, cases, k).
    fn nop_fixture(n_gates: usize, n_shots: usize, k: usize) -> (Vec<Gate>, Vec<TestCase>, usize) {
        assert!(n_gates <= N_QUBITS, "one distinct target qubit per gate");
        let gates: Vec<Gate> = (0..n_gates)
            .map(|i| Gate {
                opcode: OP_NOP,
                target: i as u16,
                ctrl_a: NO_CTRL,
                ctrl_b: NO_CTRL,
            })
            .collect();
        // Deterministic but non-trivial 64-byte states; y == x since NOP is the identity.
        let cases: Vec<TestCase> = (0..n_shots)
            .map(|s| {
                let mut st = [0u8; STATE_BYTES];
                for (b, byte) in st.iter_mut().enumerate() {
                    *byte = ((s * 31 + b * 7 + 1) & 0xff) as u8;
                }
                let hex = hex::encode(st);
                TestCase {
                    x_hex: hex.clone(),
                    y_hex: hex,
                }
            })
            .collect();
        (gates, cases, k)
    }

    /// Shape params shared by both tests: build shard-0 rows/boundary/program + pcs config for the
    /// fixture, matching `main`'s precompute setup.
    fn shard0_shape(
        gates: &[Gate],
        cases: &[TestCase],
        k: usize,
    ) -> (
        Vec<Row>,
        BoundaryTable,
        ProgramTable,
        usize,
        u32,
        u32,
        stwo::core::pcs::PcsConfig,
    ) {
        let (rows, boundary) = build_rows(gates, cases, k).expect("build_rows");
        let real_rows = rows.len();
        let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
        let log_n_rows = padded_rows.ilog2();
        let program = build_program_table(gates, cases.len(), k);
        let max_log_size = tree0_max_log_size(
            log_n_rows,
            rc_log_size(k * gates.len()),
            program.log_size,
            boundary.log_size,
        );
        let config = leaf::leaf_pcs_config(
            max_log_size,
            recursive_aggregate::TopologyConfig::default().base_log_blowup,
        );
        (
            rows,
            boundary,
            program,
            padded_rows,
            log_n_rows,
            max_log_size,
            config,
        )
    }

    /// Proves ONE tiny gate_air base STARK on the CPU (SimdBackend) for `(gates, cases, k)`, returning
    /// the circuit-form base `Proof<QM31>` + its `GateAirLeafParams` (what `prove_gate_air_leaf` consumes).
    /// Replicates the `--recurse` single-base CPU prove sequence (tree0 commit → witness/interaction
    /// gen → prove_ex → proof_from_stark_proof) for a self-contained test.
    fn prove_tiny_base(
        gates: &[Gate],
        cases: &[TestCase],
        k: usize,
    ) -> (
        circuits_stark_verifier::proof::Proof<QM31>,
        leaf::GateAirLeafParams,
        circuits_stark_verifier::proof::ProofConfig,
        circuits::blake::HashValue<QM31>,
    ) {
        use circuit_statement::gate_air_components;
        use circuits::blake::HashValue;
        use circuits::ivalue::NoValue;
        use circuits_stark_verifier::proof::ProofConfig;
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::{prove_ex, CommitmentSchemeProver};
        // `pack_public_claim` is imported at module scope (main.rs line 52).

        let n_gates = gates.len();
        let (rows, boundary) = build_rows(gates, cases, k).expect("build_rows");
        let real_rows = rows.len();
        let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
        let log_n_rows = padded_rows.ilog2();
        let rc_log = rc_log_size(k * n_gates);
        let program = build_program_table(gates, cases.len(), k);
        let max_log_size =
            tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
        // TOY (INSECURE) base PCS: blowup 1, ONE FRI query, no grind. This is a LAPTOP-SAFETY lever:
        // the in-circuit STARK verifier (`emit_one_base`) builds a decommit circuit whose size scales
        // with `n_queries`, so a 1-query base makes the base-NODE trace ~2^15 instead of the
        // production ~2^22 (23 queries) — the whole prove→fold→verify roundtrip then runs in seconds
        // / <1GB on a laptop. NOT secure (bypasses `leaf_pcs_config`'s 70/23-query floor + the >=96-bit
        // assert); the production `--recurse` path uses `leaf_pcs_config`. The recursion (base-node /
        // node / root) proofs derive their OWN PCS from their tiny traces, so they stay cheap even at
        // the default query counts — only the base config drives the trace size.
        let config = PcsConfig {
            pow_bits: 0,
            fri_config: FriConfig {
                log_blowup_factor: 1,
                log_last_layer_degree_bound: 0,
                n_queries: 1,
                fold_step: 4,
            },
            lifting_log_size: Some(max_log_size + 1),
        };
        let rc_table = build_rc_table(&rows, rc_log);

        let twiddles = TraceBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let prover_channel = &mut Blake2sM31Channel::default();
        let channel_salt = 0u32;
        prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
        config.mix_into(prover_channel);
        let mut commitment_scheme =
            CommitmentSchemeProver::<TraceBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

        // Tree 0: preprocessed.
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

        let public_claim = pack_public_claim(&[]);
        prover_channel.mix_felts(&public_claim);

        // Tree 1: main + program/boundary/rc witness.
        let small_main = {
            let mut v = generate_program_witness(&program);
            v.extend(generate_boundary_witness(&boundary));
            v.extend(generate_rc_witness(&rc_table));
            v
        };
        let mut tree_builder = commitment_scheme.tree_builder();
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        tree_builder.extend_evals(to_prover(main_trace));
        tree_builder.commit(prover_channel);

        let interaction_pow_nonce = TraceBackend::grind(prover_channel, INTERACTION_POW_BITS);
        prover_channel.mix_u64(interaction_pow_nonce);
        let elements = LookupElements::draw(prover_channel);

        let (main_interaction, main_sum) =
            gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
        let (program_interaction, program_sum) =
            gen_program_interaction(&program, &elements.program);
        let (boundary_interaction, boundary_sum) =
            gen_boundary_interaction(&boundary, &elements.qubitmem);
        let (rc_interaction, rc_sum) = {
            let el = elements.rc.clone();
            gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
            })
        };
        let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
        prover_channel.mix_felts(&claimed_sums);

        // Tree 2: interaction (main, program, boundary, rc).
        let small_interaction = {
            let mut v = program_interaction;
            v.extend(boundary_interaction);
            v.extend(rc_interaction);
            v
        };
        let mut tree_builder = commitment_scheme.tree_builder();
        let mut interaction = main_interaction;
        interaction.extend(small_interaction);
        tree_builder.extend_evals(to_prover(interaction));
        tree_builder.commit(prover_channel);

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
        let prover_refs = components.prover_refs();
        let extended = prove_ex::<TraceBackend, Blake2sM31MerkleChannel>(
            &prover_refs,
            prover_channel,
            commitment_scheme,
            false,
        )
        .expect("base prove_ex");

        // Circuit-form config + params.
        let cfg = ProofConfig::new(
            &gate_air_components::<NoValue>(),
            N_PREPROCESSED_COLS,
            &config,
            INTERACTION_POW_BITS,
        );
        let pp_root: HashValue<SecureField> = extended.proof.commitments[0].into();
        let mut boundary_xy = Vec::with_capacity(cases.len());
        for case in cases {
            let x = state_to_limbs(&hex::decode(&case.x_hex).unwrap());
            let y = state_to_limbs(&hex::decode(&case.y_hex).unwrap());
            boundary_xy.push((x, y));
        }
        let total_pc = (n_gates * k) as u32;
        let params = leaf::GateAirLeafParams {
            main_log_size: log_n_rows,
            program_log_size: program.log_size,
            boundary_log_size: boundary.log_size,
            preprocessed_root: pp_root,
            boundary: boundary_xy,
            total_pc,
            program: program_rows_from_table(&program),
            nonce: hiding_nonce(),
        };
        let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];
        let circuit_proof =
            proof_from_stark_proof(&extended, &cfg, claim, interaction_pow_nonce, channel_salt);
        // Canonical base preprocessed root recomputed from the trusted shape + toy config (step 1).
        // Equals `pp_root` (the committed root) in this honest test — the recompute path the
        // production base-fanning config uses to pin the base root against a forgeable proof value.
        let canonical_base_pp_root = canonical_base_preprocessed_root(
            &program,
            &rows,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            &boundary,
            config,
        );
        (circuit_proof, params, cfg, canonical_base_pp_root)
    }

    /// The LEAF/R1/R2 prove→fold→verify→self-verify roundtrip over `n_leaves`
    /// standalone gate_air leaves, using the toy per-base PCS from `prove_tiny_base`: one leaf per base
    /// (`prove_gate_air_leaf`),
    /// a level-0 leaf-verifying (R1) layer + shared R2 up-tree fold (`recursive_aggregate_prove_leaves`),
    /// and the leaf unpacker (`prove_root_verification_leaves` / `LeafBottom`). `n_leaves == 1` is a
    /// lone-leaf root (no R1, no R2 — laptop-safe); `n_leaves >= 2` builds the level-0 R1 layer (and, at
    /// `n > k`, an R2 up-tree node) which floors ~2^22 → heavy.
    fn leaf_r1r2_roundtrip(n_leaves: usize, log_blowup_factor: u32, fold_arity: usize) {
        use circuits_stark_verifier::proof::Proof;
        use leaf::{
            build_recursion_precompute, derive_aggregate_config, prove_gate_air_leaf,
            GateAirLeafParams,
        };
        use recursive_aggregate::{
            prove_root_verification_leaves, recursive_aggregate_prove_leaves, LeafBottom, PoolSet,
            TreeProof,
        };

        let (gates, cases, k) = nop_fixture(4, 2, 1);
        // LeafR1R2 uses the leaf preprocessed root (not the base-fanning canonical base root), so the
        // recomputed canonical base pp root is unused here.
        let (proof0, params0, cfg, _canonical_base_pp_root) = prove_tiny_base(&gates, &cases, k);
        let (config, shapes) = derive_aggregate_config(
            &cfg,
            &params0,
            fold_arity,
            log_blowup_factor,
            log_blowup_factor,
        );
        let pre = build_recursion_precompute(shapes);

        let make_base =
            || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };
        let leaves: Vec<TreeProof> = (0..n_leaves)
            .map(|_| {
                let (p, params) = make_base();
                prove_gate_air_leaf(p, &cfg, &params, &config, &pre)
            })
            .collect();
        assert_eq!(leaves.len(), n_leaves);

        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(2);
        let pools = PoolSet::new(1, cores.max(1));
        let out = recursive_aggregate_prove_leaves(leaves.clone(), &config, &pre, &pools);

        let bottom = LeafBottom { leaves };
        let rv =
            prove_root_verification_leaves(&out.root, &bottom, &config, log_blowup_factor, None);
        assert_eq!(
            rv.leaf_outputs.len(),
            n_leaves,
            "root exposes one H_i per leaf"
        );

        // TRUSTED FINAL VERIFY (step 3): check `rv` against a canonical unpacker circuit recomputed
        // from trusted public `(n, config)` — canonical unpacker root (pins every baked child root
        // incl. the canonical leaf tree0 root + R1/R2/short roots) and the caller-committed
        // `rv.leaf_outputs`. `None` blinding matches the unblinded test proof above.
        verify_gate_air_root_leaves(&rv, &config, n_leaves, log_blowup_factor, None)
            .expect("trusted gate_air root verification failed (leaf/R1/R2 roundtrip)");

        eprintln!(
            "gate-air: leaf_r1r2 roundtrip OK (N={n_leaves}, n_levels={}, root trace 2^{}) [trusted verify OK]",
            out.n_levels, rv.trace_log_size
        );
    }

    /// End-to-end LeafR1R2 correctness gate: ONE standalone leaf that IS the root (no R1 level-0 node,
    /// no R2 up-tree fold). Validates the leaf-topology WIRING — `derive_aggregate_config`,
    /// `prove_gate_air_leaf`, and the leaf unpacker reconstructing + binding a single-leaf tree
    /// (`prove_root_verification_leaves`'s final `verify_circuit` sanity check). The multi-leaf R1/R2
    /// path is exercised by the env-gated heavy variant + proving-utils' restored `smoke_cairo_tree`.
    ///
    /// RUN-GUARD (laptop-safety): env-gated to GATE_AIR_HEAVY_RECURSION so plain `cargo test` never
    /// executes a real recursion prove/verify on a laptop. Run it on the CPU VM with the guard set.
    #[test]
    fn leaf_r1r2_end_to_end() {
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "leaf_r1r2_end_to_end: SKIPPED (recursion prove/verify). Set \
                 GATE_AIR_HEAVY_RECURSION=1 to run."
            );
            return;
        }
        // N=1: lone leaf is the root; no fold. Blowup 1 keeps the leaf + root-verify proofs minimal.
        leaf_r1r2_roundtrip(
            1,
            1,
            recursive_aggregate::TopologyConfig::default().fold_arity,
        );
    }

    /// HEAVY (box-only): the LeafR1R2 roundtrip WITH the level-0 R1 layer + R2 up-tree fold. R2 floors
    /// ~2^22 (several GB); OOMs a laptop, so env-gated to GATE_AIR_HEAVY_RECURSION=1. Plain `cargo test`
    /// compiles + SKIPS it.
    #[test]
    fn leaf_r1r2_end_to_end_with_r2() {
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "leaf_r1r2_end_to_end_with_r2: SKIPPED (heavy: R1/R2 nodes ~2^22). Set \
                 GATE_AIR_HEAVY_RECURSION=1 on the CPU VM to run the with-R1/R2 roundtrip."
            );
            return;
        }
        // N=2: two leaves → one level-0 R1 leaf-node that IS the root (N <= k, no R2). Bump to N > k to
        // also exercise the R2 up-tree fold once a box run confirms the R1 layer.
        leaf_r1r2_roundtrip(
            2,
            3,
            recursive_aggregate::TopologyConfig::default().fold_arity,
        );
    }

    /// SCHEDULING-INDEPENDENCE (byte-identity) roundtrip for the overlapped
    /// [`recursive_aggregate_prove_leaves_streaming`]: prove `n_leaves` tiny gate_air leaves ONCE,
    /// then fold them (a) in order via the collect-then-fold [`recursive_aggregate_prove_leaves`] and
    /// (b) in a SCRAMBLED arrival order via the streaming coordinator (identity `wrap`, so only the
    /// SCHEDULE differs), and assert the root proof (bytes + pp_root + outs), `n_levels`, and the
    /// returned ordered leaves are BIT-EQUAL. Because the only difference is arrival/completion order,
    /// equality proves the streaming path is byte-identical to the sequential one — the acceptance
    /// invariant for "hide the fold behind base-proving". `k` is the default fold arity.
    ///
    /// HEAVY (box-only): builds real R1 (and, at `n > k`, R2) multiverifier nodes (~2^22, GBs) so it
    /// OOMs a laptop; env-gated to GATE_AIR_HEAVY_RECURSION. Plain `cargo test` compiles + SKIPS it.
    fn leaf_r1r2_streaming_equiv(n_leaves: usize, log_blowup_factor: u32, fold_arity: usize) {
        use circuits_stark_verifier::proof::Proof;
        use leaf::{
            build_recursion_precompute, derive_aggregate_config, prove_gate_air_leaf,
            GateAirLeafParams,
        };
        use recursive_aggregate::{
            recursive_aggregate_prove_leaves, recursive_aggregate_prove_leaves_streaming,
            AggregateOutput, PoolSet, TreeProof,
        };

        let (gates, cases, k) = nop_fixture(4, 2, 1);
        let (proof0, params0, cfg, _canonical_base_pp_root) = prove_tiny_base(&gates, &cases, k);
        let (config, shapes) = derive_aggregate_config(
            &cfg,
            &params0,
            fold_arity,
            log_blowup_factor,
            log_blowup_factor,
        );
        let pre = build_recursion_precompute(shapes);

        let make_base =
            || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };
        let leaves: Vec<TreeProof> = (0..n_leaves)
            .map(|_| {
                let (p, params) = make_base();
                prove_gate_air_leaf(p, &cfg, &params, &config, &pre)
            })
            .collect();

        // Bit-identity signature of a folded root (proof + pp_root + outs) and its leaves. `TreeProof`
        // is only `Clone`, so compare via the same deterministic `{:?}` canonicalisation the
        // recursion fingerprint uses.
        let sig = |leaves: &[TreeProof], out: &AggregateOutput| -> String {
            let mut s = format!("n_levels={}", out.n_levels);
            s += &format!("|root.proof={:?}", out.root.proof);
            s += &format!("|root.pp={:?}", out.root.preprocessed_root);
            s += &format!("|root.outs={:?}", out.root.output_values);
            for (i, l) in leaves.iter().enumerate() {
                s += &format!("|leaf[{i}].proof={:?}", l.proof);
                s += &format!("|leaf[{i}].pp={:?}", l.preprocessed_root);
                s += &format!("|leaf[{i}].outs={:?}", l.output_values);
            }
            s
        };

        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(2);
        // (a) Sequential collect-then-fold (the reference).
        let pools_seq = PoolSet::new(1, cores.max(1));
        let out_seq = recursive_aggregate_prove_leaves(leaves.clone(), &config, &pre, &pools_seq);
        let seq_sig = sig(&leaves, &out_seq);

        // (b) Streaming, SCRAMBLED arrival order (reverse), identity `wrap` (the leaves already
        // exist — only the schedule differs). Try n_pools 1 and 2 to cover both worker counts.
        for n_pools in [1usize, 2] {
            let pools = PoolSet::new(n_pools, (cores / n_pools).max(1));
            let (tx, rx) = std::sync::mpsc::channel::<(usize, TreeProof)>();
            // Scramble: send indices in reverse (a base-producer never guarantees arrival order).
            for i in (0..n_leaves).rev() {
                tx.send((i, leaves[i].clone())).unwrap();
            }
            drop(tx);
            let (leaves_out, out_stream) = recursive_aggregate_prove_leaves_streaming(
                rx,
                n_leaves,
                |t: TreeProof| t, // identity wrap
                &config,
                &pre,
                &pools,
            );
            assert_eq!(
                sig(&leaves_out, &out_stream),
                seq_sig,
                "n_leaves={n_leaves} n_pools={n_pools}: streaming fold not bit-identical to sequential"
            );
        }
        eprintln!(
            "gate-air: leaf_r1r2 streaming-equiv OK (N={n_leaves}, bit-identical to sequential, scrambled arrival, n_pools 1+2)"
        );
    }

    /// (box-only) Scheduling-independence roundtrip over the required n ∈ {1, 2, k, k+1, ragged
    /// r==1, ~2k+3}. Env-gated (heavy R1/R2 proving). Proves streaming == sequential byte-for-byte.
    #[test]
    fn leaf_r1r2_streaming_equiv_sweep() {
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "leaf_r1r2_streaming_equiv_sweep: SKIPPED (heavy: real R1/R2 proving ~2^22). Set \
                 GATE_AIR_HEAVY_RECURSION=1 on the CPU VM to run the out-of-order equivalence sweep."
            );
            return;
        }
        let k = recursive_aggregate::TopologyConfig::default().fold_arity;
        // n = k+1 is the ragged r==1 case (splits into k-1 and 2); 2k+3 exercises multi-group + carry.
        for n in [1usize, 2, k, k + 1, 2 * k + 3] {
            leaf_r1r2_streaming_equiv(n, 1, k);
        }
    }

    /// TERMINATION + PANIC PROPAGATION for the streaming coordinator: a `wrap` closure that panics
    /// must make the coordinator re-panic on the parent (via `thread::scope` join) — no hang, no
    /// silent drop — for BOTH n_pools == 1 and > 1. The panic fires INSIDE `wrap`, before any R1/fold
    /// node proves, so the machinery under test is pure scheduling/termination.
    ///
    /// HEAVY (box-only): `derive_aggregate_config` builds the ~2^22 R1/R2 node preprocessed shapes
    /// (heavy REGARDLESS of leaf size — a node verifies `fold_arity` in-circuit STARK proofs), which
    /// OOMs a laptop; env-gated to GATE_AIR_HEAVY_RECURSION. Plain `cargo test` compiles + SKIPS it.
    #[test]
    fn leaf_r1r2_streaming_wrap_panic_propagates() {
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "leaf_r1r2_streaming_wrap_panic_propagates: SKIPPED (heavy: config build ~2^22). \
                 Set GATE_AIR_HEAVY_RECURSION=1 on the CPU VM to run the panic-propagation test."
            );
            return;
        }
        use recursive_aggregate::{recursive_aggregate_prove_leaves_streaming, PoolSet, TreeProof};

        // Build the smallest valid LeafR1R2 config from a tiny base. No recursion PROVE runs (wrap
        // panics first), but the config build itself is the heavy part gated above.
        let (gates, cases, k) = nop_fixture(4, 2, 1);
        let (_p0, params0, cfg, _r) = prove_tiny_base(&gates, &cases, k);
        let fold_arity = recursive_aggregate::TopologyConfig::default().fold_arity;
        let (config, shapes) = leaf::derive_aggregate_config(&cfg, &params0, fold_arity, 1, 1);
        let pre = leaf::build_recursion_precompute(shapes);

        for n_pools in [1usize, 2] {
            let config = &config;
            let pre = &pre;
            let n_leaves = 2usize; // one R1 group (n <= k); wrap panics before any node proves.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let pools = PoolSet::new(n_pools, 2);
                let (tx, rx) = std::sync::mpsc::channel::<(usize, usize)>();
                for i in 0..n_leaves {
                    tx.send((i, i)).unwrap();
                }
                drop(tx);
                // Every `wrap` panics — a worker panic must re-panic on the coordinator's
                // `thread::scope` join (not hang, not be silently dropped). The panic fires inside
                // `wrap`, before any R1/fold node runs, so no real proving happens (laptop-safe).
                recursive_aggregate_prove_leaves_streaming(
                    rx,
                    n_leaves,
                    |i: usize| -> TreeProof { panic!("intentional wrap panic at leaf {i}") },
                    config,
                    pre,
                    &pools,
                );
            }));
            assert!(
                result.is_err(),
                "n_pools={n_pools}: a panicking wrap must re-panic on the parent (no hang, no silent drop)"
            );
        }
        eprintln!(
            "gate-air: streaming wrap-panic propagation OK (re-panics, no hang; n_pools 1+2)"
        );
    }

    /// (#1a) The precompute's cached tree-0 must match an independent rebuild (root + column count +
    /// per-column committed sizes). This is the CI net for `assert_tree0_matches_rebuild`, which is
    /// now debug-gated off the release hot path.
    #[test]
    fn tree0_precompute_matches_rebuild() {
        let (gates, cases, k) = nop_fixture(4, 2, 1);
        let n_gates = gates.len();
        let (rows0, boundary0, program0, padded_rows, log_n_rows, max_log_size, config) =
            shard0_shape(&gates, &cases, k);
        let rc_log = rc_log_size(k * n_gates);
        let pc = BaseProverPrecompute::new(
            config,
            max_log_size,
            program0,
            &rows0,
            boundary0,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
        )
        .expect("precompute new");
        // Panics on any mismatch (root / column count / sizes) — the invariant under test.
        assert_tree0_matches_rebuild(&pc, &rows0, n_gates);
    }

    /// (#2a) The base shard's claimed LogUp sums must net to the public terms B + P_pub. This is the
    /// CI net for the per-shard self-check, which is now debug-gated off the release hot path. Mirrors
    /// the exact sum computation in `prove_base_shard`.
    #[test]
    fn shard_claimed_sums_net_to_public() {
        let (gates, cases, k) = nop_fixture(4, 2, 1);
        let n_gates = gates.len();
        let (rows, boundary, program, padded_rows, log_n_rows, _max_log_size, _config) =
            shard0_shape(&gates, &cases, k);
        let rc_table = build_rc_table(&rows, rc_log_size(k * n_gates));

        // Draw the LogUp relation exactly as the prover does (salt=0, then config is mixed in the
        // real path; for a self-contained balance check the challenge just needs to be consistent
        // across all four sums + the public terms, which one draw guarantees).
        let mut channel = Blake2sM31Channel::default();
        channel.mix_felts(&[BaseField::from_u32_unchecked(0).into()]);
        let elements = LookupElements::draw(&mut channel);

        let (_mi, main_sum) =
            gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
        let (_pi, program_sum) = gen_program_interaction(&program, &elements.program);
        let (_bi, boundary_sum) = gen_boundary_interaction(&boundary, &elements.qubitmem);
        let (_ri, rc_sum) = {
            let el = elements.rc.clone();
            gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
            })
        };

        let b_public = boundary_public_term(&boundary, &elements.qubitmem);
        let p_pub = program_public_term(&program, &elements.program);
        assert_eq!(
            main_sum + program_sum + boundary_sum + rc_sum,
            b_public + p_pub,
            "base claimed sums must net to B + P_pub"
        );
    }
}
