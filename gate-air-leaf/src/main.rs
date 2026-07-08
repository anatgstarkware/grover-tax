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
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
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

// Log-size of the ts-ordering range-check (rc) supply table (2^16 rows). The table holds the two
// value ranges [0,2^RC_LO_BITS) and [0,2^RC_HI_BITS) keyed by a `pos` selector (pos=0 => lo range,
// pos=1 => hi range); its membership count is 2^RC_LO_BITS + 2^RC_HI_BITS = 33792 <= 2^16, so it
// fits one 2^16 table. Also doubles as the twiddle/FRI domain floor (the committed domain must
// cover the largest committed column) and as the GPU histogram size.
const RC_LOG_SIZE: u32 = 16;

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

// Range-check width for the ts-ordering diff `d = ts - prev_ts - 1`. We prove `d ∈ [0, 2^TS_RC_BITS)`
// by a LogUp rc-table lookup (below): `d` is split into two limbs `d = rc_lo + 2^RC_LO_BITS * rc_hi`
// (RC_LO_BITS + RC_HI_BITS == TS_RC_BITS) and each limb is looked up into the rc supply table's
// matching exact range block — `rc_lo` into [0,2^RC_LO_BITS), `rc_hi` into [0,2^RC_HI_BITS). Because
// the table blocks are the EXACT ranges (not a padded power-of-two bound), the two lookups pin
// rc_lo < 2^RC_LO_BITS and rc_hi < 2^RC_HI_BITS with NO slack, so the reconstructed d ranges over
// exactly [0, 2^TS_RC_BITS) = [0, 2^25) and nothing larger. Honest `d = pc - prev_ts <= pc <= ts_max
// - 1 < 2^23 (k<=2000), so 25 bits is complete with margin; the absolute bound 2^25 - 1 < p = 2^31-1
// guarantees the field subtraction cannot wrap, so a cyclic (stale-read) chain — which would need
// Σ(ts_i - prev_ts_i) ≡ 0 mod p with each term >= 1 — is impossible. See the soundness argument:
// pc-pinned ts gives program order, the range-check `prev_ts < ts` on EVERY access forces the chain
// to be a forward DAG, and both together defeat the reorder.
const TS_RC_BITS: usize = 25;
// Limb split of `d` for the rc-table lookup. RC_LO_BITS + RC_HI_BITS == TS_RC_BITS. The split is
// chosen so the two exact-range blocks fit ONE 2^RC_LOG_SIZE table: 2^15 + 2^10 = 33792 <= 2^16.
const RC_LO_BITS: usize = 15;
const RC_HI_BITS: usize = 10;
const _: () = assert!(RC_LO_BITS + RC_HI_BITS == TS_RC_BITS);
const _: () = assert!((1usize << RC_LO_BITS) + (1usize << RC_HI_BITS) <= (1usize << RC_LOG_SIZE));

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
// looks up each limb of `d = ts - prev_ts - 1` as (TAG_RC, pos, limb) and the rc supply table
// supplies (TAG_RC, pos, value) for every value in the pos-block's exact range.
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
/// access to this addr (0 = the init boundary node). `rc_lo`/`rc_hi` are the two limbs of the
/// ordering diff `d = ts - prev_ts - 1 = (pc+1) - prev_ts - 1 = pc - prev_ts = rc_lo + 2^RC_LO_BITS *
/// rc_hi`, each range-checked by a LogUp lookup into the rc supply table (rc_lo into [0,2^RC_LO_BITS),
/// rc_hi into [0,2^RC_HI_BITS)), proving `prev_ts < ts` (forward-DAG / no-stale-read). `active` gates
/// the terms.
#[derive(Clone, Copy)]
struct AccessCols {
    addr: u32,    // qubit index 0..511 (0 when inactive, matches program canon)
    prev_ts: u32, // predecessor's ts at this addr (0 if this is the first access)
    v: u32,       // v_before (the value read); for a control this is also v_after
    rc_lo: u32,   // low limb of d = ts - prev_ts - 1 = pc - prev_ts  (d & (2^RC_LO_BITS - 1))
    rc_hi: u32,   // high limb of d                                   (d >> RC_LO_BITS)
}

impl AccessCols {
    fn inactive() -> Self {
        Self {
            addr: 0,
            prev_ts: 0,
            v: 0,
            rc_lo: 0,
            rc_hi: 0,
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
// Per row (qubit-memory encoding), one access block = ACCESS_COLS core + RC_N_LIMBS rc limbs:
//   is_nop,is_not,is_cnot,is_toffoli                                 (4)
//   target access: addr,prev_ts,v_before, rc_lo,rc_hi                (ACCESS_BLOCK)
//   ctrl_a access: addr,prev_ts,v, rc_lo,rc_hi                       (ACCESS_BLOCK)
//   ctrl_b access: addr,prev_ts,v, rc_lo,rc_hi                       (ACCESS_BLOCK)
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
//   RANGE: active*((pc+1) - prev_ts - 1 - rc_lo - 2^RC_LO_BITS*rc_hi) = 0 reconstructs
//          d = ts-prev_ts-1 = pc-prev_ts from its two limbs, and each limb is range-checked by a
//          LogUp lookup into the rc supply table (rc_lo ∈ [0,2^RC_LO_BITS), rc_hi ∈ [0,2^RC_HI_BITS)).
//          The exact-range table blocks pin d ∈ [0, 2^TS_RC_BITS) with NO slack, so prev_ts < ts,
//          forcing the chain to be a forward DAG (no stale-read cycle).
// The target's `v_after` is likewise NOT a witness column: it equals `v_before + delta`, inlined at
// its Yield tuple and booleanity constraint. The two controls' written value equals `v` (reads
// propagate the value).
const ACCESS_COLS: usize = 3; // addr, prev_ts, v (the core access cols read by AccessMasks; ts inlined = pc+1)
const RC_N_LIMBS: usize = 2; // rc_lo, rc_hi
const ACCESS_BLOCK: usize = ACCESS_COLS + RC_N_LIMBS; // core cols + range-check limbs
const TRACE_COLUMNS: usize = 4 + ACCESS_BLOCK + ACCESS_BLOCK + ACCESS_BLOCK + 3;

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
fn build_rows(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
) -> Result<(Vec<Row>, BoundaryTable)> {
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
        .map(|(shot_id, ((block, bnd), case))| {
            simulate_shot(gates, k, shot_id, case, block, bnd)
        })
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
    // (program-order timestamp, inlined — not stored). Splits d = ts - prev_ts - 1 = pc - prev_ts
    // (>= 0 since prev_ts is an earlier program-order ts or the init 0) into the two limbs (rc_lo,
    // rc_hi) the rc-table lookup range-checks (proving prev_ts < ts). Returns the filled AccessCols
    // with v = v_before. The caller sets last[addr] to the post-access ts (= pc+1) / value (v_before
    // for reads, v_after for the target write).
    let do_access = |addr: u32, pc: u32, last_ts: &[u32], last_val: &[u32]| -> AccessCols {
        let a = addr as usize;
        let prev_ts = last_ts[a];
        let v_before = last_val[a];
        let ts = pc + 1;
        debug_assert!(ts > prev_ts, "ts {ts} must exceed prev_ts {prev_ts} (program order)");
        let d = ts - prev_ts - 1; // = pc - prev_ts
        // Completeness guard (checked once at build_rows before any access; see build_rows). Here d is
        // guaranteed < 2^TS_RC_BITS, so the limb split is exact.
        debug_assert!((d as u64) < (1u64 << TS_RC_BITS), "diff {d} exceeds range-check bound");
        let rc_lo = d & ((1u32 << RC_LO_BITS) - 1);
        let rc_hi = d >> RC_LO_BITS;
        AccessCols {
            addr,
            prev_ts,
            v: v_before,
            rc_lo,
            rc_hi,
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
// The rc table is a single 2^RC_LOG_SIZE-row `[0, range)` supply table keyed by a `pos` selector:
//   pos = RC_POS_LO (0): value block  {0, 1, ..., 2^RC_LO_BITS - 1}   (the low-limb range)
//   pos = RC_POS_HI (1): value block  {0, 1, ..., 2^RC_HI_BITS - 1}   (the high-limb range)
// Membership count = 2^RC_LO_BITS + 2^RC_HI_BITS = 33792 <= 2^RC_LOG_SIZE = 65536; the remaining
// rows are padding reusing the (pos=RC_POS_LO, value=0) tuple (a genuine member: multiplicity-counting
// only counts real limb lookups, so extra supply of an existing tuple is inert). The main component
// looks up (TAG_RC, pos, limb) for each of the two limbs of every active access; the table supplies
// -multiplicity / (TAG_RC, pos, value). Because each pos-block enumerates its EXACT range (not a
// padded power-of-two bound), the lookups pin rc_lo < 2^RC_LO_BITS and rc_hi < 2^RC_HI_BITS with NO
// slack — reconstruction d = rc_lo + 2^RC_LO_BITS*rc_hi then ranges over exactly [0, 2^TS_RC_BITS).
const RC_POS_LO: u32 = 0;
const RC_POS_HI: u32 = 1;

/// Supply table for the ts-ordering range-check. `pos`/`val` are PREPROCESSED (the table membership,
/// shard-invariant); `multiplicity` is WITNESS (count of real limb lookups landing on that row).
struct RcTable {
    log_size: u32,
    pos: Vec<u32>,         // preprocessed
    val: Vec<u32>,         // preprocessed
    multiplicity: Vec<u32>, // witness
    lo_len: usize,         // # rows in the RC_POS_LO block (= 2^RC_LO_BITS)
}

impl RcTable {
    fn new() -> Self {
        let size = 1usize << RC_LOG_SIZE;
        let lo_len = 1usize << RC_LO_BITS;
        let hi_len = 1usize << RC_HI_BITS;
        let mut pos = vec![RC_POS_LO; size];
        let mut val = vec![0u32; size];
        let mut row = 0usize;
        for v in 0..lo_len {
            pos[row] = RC_POS_LO;
            val[row] = v as u32;
            row += 1;
        }
        for v in 0..hi_len {
            pos[row] = RC_POS_HI;
            val[row] = v as u32;
            row += 1;
        }
        debug_assert!(row <= size);
        // Remaining rows stay (pos=RC_POS_LO, val=0): a valid member, inert padding.
        Self {
            log_size: RC_LOG_SIZE,
            pos,
            val,
            multiplicity: vec![0u32; size],
            lo_len,
        }
    }

    /// Row index of the (pos, value) tuple in the flattened table. lo block first, then hi block.
    #[inline]
    fn row_of(&self, pos: u32, value: u32) -> usize {
        match pos {
            RC_POS_HI => self.lo_len + value as usize,
            _ => value as usize,
        }
    }

    /// Count the two limb lookups of one active access into the multiplicity column.
    #[inline]
    fn count_access(&mut self, a: &AccessCols) {
        let lo = self.row_of(RC_POS_LO, a.rc_lo);
        let hi = self.row_of(RC_POS_HI, a.rc_hi);
        self.multiplicity[lo] += 1;
        self.multiplicity[hi] += 1;
    }
}

/// Build the rc supply table and its multiplicity column by counting every ACTIVE access's two
/// limb lookups. An access is active iff its owning gate fires it: the target on every real row, a
/// control iff the opcode uses it. Padding rows (enabler = 0) emit no lookup, so they are skipped.
fn build_rc_table(rows: &[Row]) -> RcTable {
    let mut table = RcTable::new();
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
/// fix removed the range-check LOOKUP; this now survives ONLY so the CUDA trace-gen glue
/// (`gpu_flat_inputs` -> `off_lo`/`off_hi`) can keep its device-buffer layout unchanged. The
/// pos/val columns and `row()` accessor are unused by the CPU path (hence `dead_code`).
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
        let size = 1usize << RC_LOG_SIZE;
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
        // qubitmem-pairs, then rc-pairs (target lo/hi, ctrl_a lo/hi, ctrl_b lo/hi), then program;
        // mirrored exactly by `gen_main_interaction` and the in-circuit MainGate). ---
        add_rc_lookup(&mut eval, &self.elements.rc, &target, enabler.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_a, a_active.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_b, b_active.clone());

        // --- ts-ordering: RANGE-CHECK prev_ts < ts (soundness-critical). ---
        // The old PIN constraint `active*(ts - (pc*TS_STRIDE + slot)) = 0` is GONE: ts is now
        // structurally `pc + 1` (inlined), so the pin is vacuous. Only the RANGE reconstruction
        // remains: active*((pc+1) - prev_ts - 1 - rc_lo - 2^RC_LO_BITS*rc_hi) = 0 reconstructs
        // d = pc - prev_ts from its two limbs, each range-checked by the rc-table lookup above =>
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
    rc_lo: F, // low limb of d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LO_BITS))
    rc_hi: F, // high limb of d                                   (range-checked into [0,2^RC_HI_BITS))
}

fn access_masks<E: EvalAtRow>(eval: &mut E) -> AccessMasks<E::F> {
    // ts is NOT a column — it is the inlined `pc + 1`. Per-access columns: addr, prev_ts, v, rc_lo, rc_hi.
    let addr = eval.next_trace_mask();
    let prev_ts = eval.next_trace_mask();
    let v = eval.next_trace_mask();
    // rc_lo, rc_hi follow v (matches `cell_at`'s per-access column order).
    let rc_lo = eval.next_trace_mask();
    let rc_hi = eval.next_trace_mask();
    AccessMasks {
        addr,
        prev_ts,
        v,
        rc_lo,
        rc_hi,
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
    let yield_entry = [tag, shot_id.clone(), a.addr.clone(), ts.clone(), v_out.clone()];
    eval.add_to_relation(RelationEntry::new(
        elements,
        -E::EF::from(active),
        &yield_entry,
    ));
}

/// Emit the range-check reconstruction for one access, gated by `active`:
///   RANGE: active*(ts - prev_ts - 1 - rc_lo - 2^RC_LO_BITS*rc_hi) = 0 — reconstructs the diff
///          d = ts - prev_ts - 1 = pc - prev_ts from its two limbs. The limbs themselves are
///          range-checked by the rc-table LOOKUP (`add_rc_lookup`), not here, so d ∈ [0, 2^TS_RC_BITS)
///          (prev_ts < ts). `ts` is the inlined `pc + 1` expression.
/// The old PIN constraint is removed (ts is structurally pc+1, so the pin is vacuous). The
/// reconstruction is gated by `active`; the two limb LOOKUPs are also gated by `active` (an inactive
/// access emits no rc term). Inactive accesses (active = 0) leave prev_ts/rc_lo/rc_hi free.
fn add_ts_range<E: EvalAtRow>(
    eval: &mut E,
    ts: &E::F,
    a: &AccessMasks<E::F>,
    active: E::F,
) {
    let one = E::F::one();
    // RANGE reconstruction: d = rc_lo + 2^RC_LO_BITS * rc_hi.
    let recon = a.rc_lo.clone() + a.rc_hi.clone() * BaseField::from_u32_unchecked(1u32 << RC_LO_BITS);
    let d = ts.clone() - a.prev_ts.clone() - one;
    eval.add_constraint(active * (d - recon));
}

/// Emit the two rc-table range-check LOOKUPs for one access, gated by `active`. Each limb is looked
/// up as (TAG_RC, pos, limb); the rc supply table supplies each in-range (pos, value). Emitting the
/// two terms consecutively (lo then hi) makes them a single finalize-in-pairs batch (mirrored by
/// `gen_main_interaction` and the in-circuit MainGate).
fn add_rc_lookup<E: EvalAtRow>(eval: &mut E, rc: &GateRel, a: &AccessMasks<E::F>, active: E::F) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
    let pos_lo = E::F::one() * BaseField::from_u32_unchecked(RC_POS_LO);
    let pos_hi = E::F::one() * BaseField::from_u32_unchecked(RC_POS_HI);
    eval.add_to_relation(RelationEntry::new(
        rc,
        E::EF::from(active.clone()),
        &[tag.clone(), pos_lo, a.rc_lo.clone()],
    ));
    eval.add_to_relation(RelationEntry::new(
        rc,
        E::EF::from(active),
        &[tag, pos_hi, a.rc_hi.clone()],
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
            &[tag, slot.clone(), opcode_scalar.clone(), target.clone(), ctrl_a.clone(), ctrl_b.clone()],
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

/// ts-ordering range-check table (supply side). `pos`/`val` are preprocessed (the table membership);
/// `multiplicity` is witness (count of real limb lookups landing on this row). Emits
/// -multiplicity / combine(TAG_RC, pos, val) — one term/row => 1 batch => 4 interaction columns.
#[derive(Clone)]
struct RcTableEval {
    elements: GateRel,
}

impl FrameworkEval for RcTableEval {
    fn log_size(&self) -> u32 {
        RC_LOG_SIZE
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let pos = eval.get_preprocessed_column(pp_id("gate_rc_pos"));
        let val = eval.get_preprocessed_column(pp_id("gate_rc_val"));
        let multiplicity = eval.next_trace_mask();
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, pos, val],
        ));
        eval.finalize_logup();
        eval
    }
}

fn pp_id(id: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: id.to_owned() }
}

/// Number of preprocessed columns (count-only uses; the order is `preprocessed_column_ids`).
/// prog_slot + (enabler/shot_id/pc/pc_in_prog) + (bnd_shot/bnd_addr/bnd_enabler) + (rc_pos/rc_val) = 10.
const N_PREPROCESSED_COLS: usize = 10;

/// Each preprocessed column paired with its log_size, in a fixed canonical listing order, then
/// STABLE-sorted ascending by size. The committed preprocessed tree MUST be size-sorted (stwo's
/// lifted Merkle sorts each tree's columns by length, and the in-circuit verifier does NOT re-sort
/// the preprocessed tree). The sizes are DYNAMIC: `gate_pc_in_prog` is sized with the main trace,
/// `gate_prog_slot` with the program table, `gate_bnd_*` with the boundary table, and the rc table
/// columns are fixed at RC_LOG_SIZE.
fn preprocessed_columns_sorted(
    main_log_size: u32,
    program_log_size: u32,
    boundary_log_size: u32,
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
        // ts-ordering range-check table membership (pos, val). Fixed at RC_LOG_SIZE.
        (pp_id("gate_rc_pos"), RC_LOG_SIZE),
        (pp_id("gate_rc_val"), RC_LOG_SIZE),
    ];
    cols.sort_by_key(|&(_, s)| s); // stable: ties keep the listing order above
    cols
}

fn preprocessed_column_ids(
    main_log_size: u32,
    program_log_size: u32,
    boundary_log_size: u32,
) -> Vec<PreProcessedColumnId> {
    preprocessed_columns_sorted(main_log_size, program_log_size, boundary_log_size)
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
    elements: &LookupElements,
    main_sum: SecureField,
    program_sum: SecureField,
    boundary_sum: SecureField,
    rc_sum: SecureField,
) -> Components {
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(
        &preprocessed_column_ids(log_n_rows, program_log_size, boundary_log_size),
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
/// matches the order `evaluate` reads masks (each access block = ACCESS_COLS core + RC_N_LIMBS limbs):
///   is_{nop,not,cnot,toffoli}(4),
///   target(addr,prev_ts,v, rc_lo,rc_hi),
///   ctrl_a(addr,prev_ts,v, rc_lo,rc_hi), ctrl_b(addr,prev_ts,v, rc_lo,rc_hi),
///   ab, fire, delta (3).
/// NOTE: enabler, shot_id, pc are NOT here — they are preprocessed (tree0). ts (= pc+1) and the
/// target's v_after (= v_before+delta) are NOT columns either — they are inlined in `evaluate`.
#[inline]
fn cell_at(row: &Row, col: usize) -> u32 {
    debug_assert!(col < TRACE_COLUMNS);
    #[inline]
    fn access_cell(a: &AccessCols, i: usize) -> u32 {
        // 0..ACCESS_COLS: addr, prev_ts, v ; then rc_lo, rc_hi.
        match i {
            0 => a.addr,
            1 => a.prev_ts,
            2 => a.v,
            3 => a.rc_lo,
            _ => a.rc_hi,
        }
    }
    let mut c = col;
    // Header (4 cols).
    const HEADER: [fn(&Row) -> u32; 4] = [
        |r| r.is_nop,
        |r| r.is_not,
        |r| r.is_cnot,
        |r| r.is_toffoli,
    ];
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

/// Preprocessed rc-table membership columns (pos, val), in the order RcTableEval reads them.
fn generate_rc_preprocessed(
    rc: &RcTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&rc.pos), col_from_values(&rc.val)]
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
/// emitted in `evaluate`, finalized in pairs. Order must match (7 entries -> 4 columns):
///   pair0: qubitmem target Use (+enabler), target Yield (-enabler)
///   pair1: qubitmem ctrl_a Use (+a_active), ctrl_a Yield (-a_active)
///   pair2: qubitmem ctrl_b Use (+b_active), ctrl_b Yield (-b_active)
///   pair3: rc target lo (+enabler), target hi (+enabler)
///   pair4: rc ctrl_a lo (+a_active), ctrl_a hi (+a_active)
///   pair5: rc ctrl_b lo (+b_active), ctrl_b hi (+b_active)
///   col6:  program (+enabler)  [odd tail => a singleton batch, not a pair]
/// 13 relation entries -> 6 pairs + 1 singleton => 7 batches => 28 interaction columns. The rc terms
/// are the ts-ordering range-check limb lookups (TAG_RC, pos, limb) mirroring `add_rc_lookup`.
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

    // ts-ordering range-check use-side denominators: (TAG_RC, pos, limb) per limb of each access.
    let rc_lo = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.rc.combine(&[
            ptag(TAG_RC),
            ptag(RC_POS_LO),
            pack(lane, |r| sel(r).rc_lo),
        ])
    };
    let rc_hi = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.rc.combine(&[
            ptag(TAG_RC),
            ptag(RC_POS_HI),
            pack(lane, |r| sel(r).rc_hi),
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
    // Write one logup column for a SINGLE relation entry: fraction = m0/d0. Used for the odd tail
    // (program) so the prover-side batching matches `finalize_logup_in_pairs`'s last (singleton)
    // chunk when the relation-entry count is odd.
    fn write_single_par<N0, D0>(
        gen: &mut LogupTraceGenerator,
        rows: &[Row],
        n_vec: usize,
        num0: N0,
        den0: D0,
        sign0: i32,
    ) where
        N0: Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync,
        D0: Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync,
    {
        use rayon::prelude::*;
        let pad = Row::padding();
        let col_iter = (0..n_vec).into_par_iter().map(|vec_row| {
            let lane: [&Row; LANE_COUNT] =
                std::array::from_fn(|l| rows.get(vec_row * LANE_COUNT + l).unwrap_or(&pad));
            let m0 = PackedSecureField::from(num0(&lane));
            let m0 = if sign0 < 0 { -m0 } else { m0 };
            let d0 = den0(&lane);
            (m0, d0)
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
    // pair3: rc target lo (+enabler), rc target hi (+enabler).
    write_pair(
        &mut gen,
        &enabler,
        &|l| rc_lo(l, sel_t),
        1,
        &enabler,
        &|l| rc_hi(l, sel_t),
        1,
    );
    // pair4: rc ctrl_a lo (+a_active), rc ctrl_a hi (+a_active).
    write_pair(
        &mut gen,
        &a_active,
        &|l| rc_lo(l, sel_a),
        1,
        &a_active,
        &|l| rc_hi(l, sel_a),
        1,
    );
    // pair5: rc ctrl_b lo (+b_active), rc ctrl_b hi (+b_active).
    write_pair(
        &mut gen,
        &b_active,
        &|l| rc_lo(l, sel_b),
        1,
        &b_active,
        &|l| rc_hi(l, sel_b),
        1,
    );
    // col6: program (+enabler). Odd tail => a singleton batch (matches finalize_logup_in_pairs).
    write_single_par(&mut gen, rows, n_vec, enabler, program, 1);

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
    // The main component reads four preprocessed columns IN THIS ORDER (matching
    // `GateEval::evaluate`'s `get_preprocessed_column` calls): enabler, shot_id, pc, pc_in_prog.
    // `assert_constraints_on_trace` feeds them positionally to the eval's preprocessed reads.
    let pp_vals: Vec<Vec<BaseField>> = vec![
        generate_enabler_preprocessed(rows, padded_rows).values.to_cpu(),
        generate_shot_id_preprocessed(rows, padded_rows).values.to_cpu(),
        generate_pc_preprocessed(rows, padded_rows).values.to_cpu(),
        generate_pc_in_prog_preprocessed(rows, padded_rows, n_gates).values.to_cpu(),
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
    let off_lo: Vec<u32> = (0..LIMB_BITS).map(|p| rc_lo_index.offset[p] as u32).collect();
    let off_hi: Vec<u32> = (0..LIMB_BITS).map(|p| rc_hi_index.offset[p] as u32).collect();
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
    boundary: &BoundaryTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let mut tagged: Vec<(u32, CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>)> = vec![
        (program.log_size, generate_prog_slot_preprocessed(program)),
        (log_n_rows, generate_enabler_preprocessed(rows, padded_rows)),
        (log_n_rows, generate_shot_id_preprocessed(rows, padded_rows)),
        (log_n_rows, generate_pc_preprocessed(rows, padded_rows)),
        (log_n_rows, generate_pc_in_prog_preprocessed(rows, padded_rows, n_gates)),
    ];
    tagged.extend(
        generate_boundary_preprocessed(boundary)
            .into_iter()
            .map(|c| (boundary.log_size, c)),
    );
    let rc = RcTable::new();
    tagged.extend(
        generate_rc_preprocessed(&rc)
            .into_iter()
            .map(|c| (RC_LOG_SIZE, c)),
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
            &program0, rows0, padded_rows, log_n_rows, n_gates, &boundary0,
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
            &self.program, &self.rows0, self.padded_rows, self.log_n_rows, self.n_gates,
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
        Ok(DevicePrecompute { twiddles, tree0, d_gates, d_off_lo, d_off_hi })
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
fn assert_tree0_matches_rebuild(
    pc: &BaseProverPrecompute,
    rows0: &[Row],
    n_gates: usize,
) {
    // Rebuild via the exact old path (fresh scheme/channel; columns from the same builder).
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(
            pc.log_n_rows.max(RC_LOG_SIZE) + 1 + pc.config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );
    let mut scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(pc.config, &twiddles);
    let cols = build_tree0_columns(
        &pc.program, rows0, pc.padded_rows, pc.log_n_rows, n_gates, &pc.boundary,
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
    // large N it is the single O(N)-scaling host allocation (`Row` is 776 bytes) and OOMs the box.
    let fold_active = std::env::var("GATE_AIR_FOLD").is_ok();
    let (rows, boundary) = if fold_active {
        (Vec::<Row>::new(), BoundaryTable::new(0))
    } else {
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
        use circuits_stark_verifier::proof::{Proof, ProofConfig};
        use stwo::core::fields::qm31::QM31;
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use leaf::{
            GateAirLeafParams, derive_aggregate_config, derive_base_fanning_config,
            prove_base_node, prove_gate_air_leaf,
        };
        use recursive_aggregate::{
            BaseFanBottom, BaseOutput, FoldMode, LeafBottom, PoolSet, TopologyConfig, TreeProof,
            ZkBlind, base_fan_group_sizes, prove_root_verification,
            prove_root_verification_leaves, recursive_aggregate_prove,
            recursive_aggregate_prove_leaves,
        };
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
            let max_log_size = log_n_rows.max(RC_LOG_SIZE);
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
            let mut commitment_scheme =
                CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
                    config, twiddles,
                );
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
                        program, &rows, padded_rows, log_n_rows, n_gates, &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }
            #[cfg(not(feature = "cuda"))]
            match precompute {
                Some(pc) => {
                    commitment_scheme
                        .commit_tree(MaybeOwned::Borrowed(&pc.tree0), prover_channel);
                }
                None => {
                    // Old path: build the (size-sorted) preprocessed columns, then interpolate + LDE +
                    // Merkle-commit them inline (the shard-invariant work this precompute eliminates).
                    let pp = build_tree0_columns(
                        program, &rows, padded_rows, log_n_rows, n_gates, &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }

            // tree0 EVICTION on the SHARDED path: NOT IMPLEMENTED — genuine cross-shard fork.
            // On the production (Some(dp)) path tree0 is committed as MaybeOwned::Borrowed(dp.tree0),
            // a reference into the SHARED, cross-shard-reused Arc<BaseProverPrecompute> (each shard
            // re-commits the same tree0). Two hard blockers:
            //   1. MaybeOwned has Deref but NO DerefMut, and dp.tree0 is behind an Arc — the columns
            //      prove_ex reads (commitment_scheme.trees[0] == the borrow) CANNOT be mutated here.
            //   2. Even if they could, dehydrating them would free the shared precompute's device
            //      buffers + re-key to sentinels, so the NEXT shard's commit_tree(Borrowed(dp.tree0))
            //      would reuse freed/staged columns → corruption across shards.
            // Resolving this needs an explicit decision (per-shard clone of tree0, or
            // rehydrate-tree0-at-shard-end, or scope eviction to single-shard). Until then, fail LOUD
            // rather than silently no-op (no silent fallback): the eviction win is delivered on the
            // single-shot path (the 2^26 fit test), which uses the inline OWNED tree0 above.
            #[cfg(feature = "cuda")]
            assert!(
                !stwo::prover::backend::cuda::fused_commit::evict_tree0_enabled(),
                "GATE_AIR_EVICT_TREE0 is not supported on the sharded base-proof path: tree0 is a \
                 borrowed reference into the shared cross-shard BaseProverPrecompute and cannot be \
                 evicted without breaking multi-shard reuse. Use the single-shot path, or resolve \
                 the per-shard-copy / rehydrate-at-shard-end fork first."
            );

            let public_claim = pack_public_claim(&[]);
            prover_channel.mix_felts(&public_claim);

            #[cfg(feature = "cuda")]
            let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

            // ts-ordering range-check supply table (multiplicity counted from active-access lookups).
            let rc_table = build_rc_table(&rows);

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
                    let bytes = hex::decode(&c.x_hex)
                        .context("decoding x_hex for GPU trace-gen")?;
                    x_states.extend_from_slice(&state_to_limbs(&bytes));
                }
                let (main_dev, _qd, _lo, _hi, d_cols) = match &dp {
                    // Multi-GPU: use THIS device's N3 buffers (device 0's eager d_*, or the per-device
                    // replica) — feeding device-0 buffers to a device-n kernel would be an illegal
                    // cross-device access.
                    Some(dp) => gpu_tracegen::gpu_gen_main_trace_device_d(
                        dp.d_gates, &x_states, dp.d_off_lo, dp.d_off_hi,
                        k as u32, n_gates as u32, shard_samples as u32, padded_rows, log_n_rows,
                    ),
                    None => {
                        let (gates_flat, _x, off_lo, off_hi) =
                            gpu_flat_inputs(&gates, shard_cases, &rc_lo_index, &rc_lo_index)?;
                        gpu_tracegen::gpu_gen_main_trace_device(
                            &gates_flat, &x_states, &off_lo, &off_hi,
                            k as u32, n_gates as u32, shard_samples as u32, padded_rows, log_n_rows,
                        )
                    }
                }
                .map_err(|e| anyhow::anyhow!(e))?;
                // Fix (b) (`GATE_AIR_FUSED_INTERP`): same as the single-proof path — feed the 188
                // borrowed `d_cols` eval views as `CircleCoefficients` via `extend_polys` (skip the
                // batched interpolate) so the per-column-interpolate commit path handles them without
                // the second main-trace resident copy; `small_main` (different size) stays on
                // `extend_evals`. Order is 188 main THEN small_main, matching the flag-off path.
                if std::env::var("GATE_AIR_FUSED_INTERP").is_ok() {
                    use stwo::prover::poly::circle::CircleCoefficients;
                    let main_polys: Vec<CircleCoefficients<ProverBackend>> = main_dev
                        .into_iter()
                        .map(|e| CircleCoefficients::new(e.values))
                        .collect();
                    tree_builder.extend_polys(main_polys);
                    tree_builder.extend_evals(to_prover(small_main));
                } else {
                    let mut main_dev = main_dev;
                    main_dev.extend(to_prover(small_main));
                    tree_builder.extend_evals(main_dev);
                }
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

            // GATE_AIR_STREAM_MAIN (default OFF): dehydrate the ~24 GB main-trace device buffer to
            // the host + free it now that the tree1 commit is done, so it is not resident for K4's
            // interaction allocs or the tree2 commit. K4 rehydrates it only for its kernel loop.
            // Flag OFF: kept resident (byte-for-byte the previous path). See `MainTrace::from_k1`.
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
                // Composes with GATE_AIR_BOUNDARY_TRIM (option-0) if the coordinator enables it.
                stwo::prover::backend::cuda::fused_commit::boundary_trim_if_enabled();
                let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
                    main, n_gates as u32, padded_rows, log_n_rows,
                    real_rows as u64, (k * n_gates) as u64, &elements,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                Some((cols, claimed))
            } else {
                None
            };
            // K4 done: FREE the main trace NOW (before tree2), not at end-of-shard. Under
            // GATE_AIR_STREAM_MAIN_LOWMEM this `cuMemFree`s the ~24 GB resident `d_cols` DEVICE buffer
            // AND synchronizes so tree2's pool can reserve it (else tree2 OOMs the card); under plain
            // GATE_AIR_STREAM_MAIN it frees the dehydrated HOST copy early. `free_after_k4` consumes
            // the buffer explicitly. Flag OFF: `main_k1` is None → no-op.
            #[cfg(feature = "cuda")]
            if let Some(m) = main_k1.take() {
                m.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
            }
            #[cfg(feature = "cuda")]
            let (main_interaction, main_sum) = if let Some((_, claimed)) = &main_interaction_device {
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
            // rc supply: -multiplicity / combine(TAG_RC, pos, val).
            let (rc_interaction, rc_sum) = {
                let el = elements.rc.clone();
                gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                    el.combine(&[
                        ptag(TAG_RC),
                        pack_seq(&rc_table.pos, vec_row),
                        pack_seq(&rc_table.val, vec_row),
                    ])
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
            // Option (a) — arm the interaction (tree2) staging scope for this commit ONLY. Under
            // GATE_AIR_STREAM_INTERACTION this makes `evaluate_polynomials` host-stage the OWNED
            // interaction eval columns (dehydrate + free their device buffers) so the composition
            // kernel row-tiles tree2 instead of holding it whole-resident. The arm flag scopes the
            // staging to THIS commit (never tree0/tree1); flag OFF => no-op (staging only fires when
            // GATE_AIR_STREAM_INTERACTION is set), so the arm/disarm is byte-identical off the flag.
            #[cfg(feature = "cuda")]
            stwo::prover::backend::cuda::fused_commit::arm_interaction_commit(true);
            tree_builder.commit(prover_channel);
            #[cfg(feature = "cuda")]
            stwo::prover::backend::cuda::fused_commit::arm_interaction_commit(false);

            let components = build_components(
                log_n_rows,
                program.log_size,
                boundary.log_size,
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
        let base_precompute: Option<std::sync::Arc<BaseProverPrecompute>> = if no_base_precompute {
            eprintln!("gate-air: base precompute DISABLED (GATE_AIR_NO_BASE_PRECOMPUTE) — rebuilding tree0/twiddles/program/N3 per shard");
            None
        } else {
            let t_pc = Instant::now();
            // Shard 0's shape (every shard shares it: equal shot count, same program + k).
            let program0 = build_program_table(&gates, shots_per_shard, k);
            let (rows0, boundary0) =
                build_rows(&gates, &shard_case_sets[0], k)?;
            let real_rows0 = rows0.len();
            let padded_rows0 = real_rows0.next_power_of_two().max(1 << (LOG_N_LANES + 2));
            let log_n_rows0 = padded_rows0.ilog2();
            let max_log_size0 = log_n_rows0.max(RC_LOG_SIZE);
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
        let base_precompute_ref = base_precompute.as_deref();

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
        // path, prove all up front. In the pipeline path, prove ONLY shard 0 here (the rest are
        // produced concurrently by the producer thread, below).
        let t = Instant::now();
        let mut shard_bases = Vec::with_capacity(n_shards);
        if pipeline {
            eprintln!("gate-air: proving shard 0 base proof eagerly (pipeline) ...");
            // The eager shard-0 base is proved on THIS (main) thread. Bind it to device 0 so its
            // device-resident work (and the device-0 producer that follows) share device 0's pool /
            // module caches consistently (matches every producer's `set_base_gpu(gpu)`), rather than
            // relying on the implicit default device.
            #[cfg(feature = "cuda")]
            gpu_tracegen::set_base_gpu(0);
            let tb0 = Instant::now();
            // Class-1 SIGSEGV fix (uniformity): run the eager shard-0 OODS fan-out inside a private
            // pool whose workers are bound to device 0, so it does NOT depend on the global rayon
            // pool's (accidental) device-0 binding. Today it happens to work on device 0 via the
            // global pool; binding it makes every base proof — eager and producer — uniformly use a
            // device-bound private pool.
            #[cfg(feature = "gpu-cuda")]
            let shard0_out = {
                let oods_pool0 = gpu_tracegen::build_device_bound_pool(0);
                oods_pool0.install(|| prove_base_shard(base_precompute_ref, &shard_case_sets[0]))?
            };
            #[cfg(not(feature = "gpu-cuda"))]
            let shard0_out = prove_base_shard(base_precompute_ref, &shard_case_sets[0])?;
            shard_bases.push(shard0_out);
            eprintln!("gate-air: MEASURE t_base[shard 0]={:.3}s", tb0.elapsed().as_secs_f64());
            // Trim device 0's pool so shard 0's now-freed 2^25 buffers are returned before the device-0
            // producer proves shard 1 (mirrors the sequential loop + the producer per-shard trim; the
            // eager shard-0 previously had NO trim, leaving its reserved segments resident on device 0
            // → the multi-shard 2^25 OOM that GATE_AIR_POOL_TRIM is meant to prevent). The producer
            // threads spawn AFTER this returns, so no concurrent device-0 alloc races this trim.
            #[cfg(feature = "cuda")]
            pool_trim_at_shard_boundary_if_enabled();
        } else {
            eprintln!("gate-air: proving {n_shards} distinct per-shard base proof(s) ...");
            for (s, shard_cases) in shard_case_sets.iter().enumerate() {
                eprintln!("gate-air: base proof for shard {s} ({} shots) ...", shard_cases.len());
                let tb = Instant::now();
                shard_bases.push(prove_base_shard(base_precompute_ref, shard_cases)?);
                eprintln!("gate-air: MEASURE t_base[shard {s}]={:.3}s", tb.elapsed().as_secs_f64());
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
                let bytes = serde_json::to_vec(&base.0.proof)
                    .expect("serialize base shard StarkProof");
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
        let base0_config =
            leaf::leaf_pcs_config(base0_log_n_rows.max(RC_LOG_SIZE), topo.base_log_blowup);
        let cfg = ProofConfig::new(
            &gate_air_components::<NoValue>(),
            n_pp,
            &base0_config,
            INTERACTION_POW_BITS,
        );
        let pp_root0: HashValue<SecureField> = base0_extended.proof.commitments[0].into();
        // Boundary table log-size is shard-invariant (n_shots_per_shard * 512 rows, padded).
        let boundary_log_size = BoundaryTable::new(shots_per_shard).log_size;

        // H_P (OPEN #3, Fork A): the program table + hiding nonce the leaf hashes into H_P and binds
        // to the base. The program is SHARD-INVARIANT (same gates, same mult = shots_per_shard*k), so
        // build it once and clone into every leaf. ONE shared nonce across all leaves.
        let leaf_program = program_rows_from_table(&build_program_table(&gates, shots_per_shard, k));
        let leaf_nonce = hiding_nonce();
        let shape_params = GateAirLeafParams {
            main_log_size: base0_log_n_rows,
            program_log_size: base0_prog_log_size,
            boundary_log_size,
            preprocessed_root: pp_root0,
            boundary: base0_boundary.clone(),
            total_pc: base0_total_pc,
            program: leaf_program.clone(),
            nonce: leaf_nonce,
        };

        // Base-fanning arity `b`: how many bases (shards) group under one base-node (base-fanning
        // mode). Under LeafR1R2 the bottom is one leaf per base (b is unused there).
        let b = topo.base_fan_arity;
        // NOTE: config derivation moved into the FoldMode dispatch below, so each mode derives ONLY
        // its own config (base-fanning `BaseFanConfig` vs leaf/R1/R2 `AggregateConfig`) after the
        // shared base materialization.

        // Partition the machine so independent leaf proves run concurrently (POOL_THREADS sweet spot).
        // MEMORY/THROUGHPUT TRADEOFF: each concurrent pool holds one large in-flight `TreeProof`
        // (FRI layers + Merkle decommits, multi-GB) while it proves a leaf/fold node, so the number
        // of pools == the number of proofs in flight == the multiplier on peak host RAM. On a big box
        // (96 vCPU) we want K = cores/48 pools for real leaf/fold concurrency; on a memory-limited box
        // (12 vCPU, ~40-85GB) K collapses to 1 (12/48 -> 0 -> max(1)), which is what we want: a single
        // in-flight fold proof, no RAM multiplier. That already prevents the N>=4 concurrency OOM.
        //
        // The remaining waste: `PoolSet::new(1, 48)` would still spawn 48 rayon OS threads (each with
        // a large default stack, and 48 > 12 cores oversubscribes) for a pool that only ever runs one
        // proof at a time. When a single pool is used we therefore clamp its worker count to the
        // actual core count, so we don't reserve ~48 big thread stacks on a 12-core box. This is a
        // pure thread-count change (rayon fan-out over NTT/Merkle/FRI is order-independent) and does
        // not touch any proof value -> byte-identical output.
        let pool_threads: usize = std::env::var("POOL_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48);
        let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(pool_threads);
        let n_pools = (cores / pool_threads).max(1);
        // With a single pool there is no sibling proof to run alongside it, so let that one pool use
        // all cores rather than reserving `pool_threads` (48) large thread stacks it can't schedule.
        let threads_per_pool = if n_pools == 1 { pool_threads.min(cores) } else { pool_threads };
        let pools = PoolSet::new(n_pools, threads_per_pool);

        // Per-shard distinct leaf: build the GateAirLeafParams for THIS shard (its own boundary +
        // preprocessed root), convert THIS shard's base proof to circuit values, and prove the leaf.
        // The leaves are now DISTINCT (each commits to its shard's own shots' (x,y) outputs).
        let n_shards_bases = n_shards;
        let cfg_ref = &cfg;

        // `make_base` turns one shard's base-proof tuple into a `(Proof<QM31>, GateAirLeafParams)`
        // base. Same params construction + proof_from_stark_proof as before; the per-leaf circuit
        // wrap is now deferred to `prove_base_node` (which proves one base-node over a GROUP of
        // these bases). Base `i` == shard `i`.
        let make_base = |base: &BaseShardOutput| -> (Proof<QM31>, GateAirLeafParams) {
            let (extended_i, claim_i, nonce_i, salt_i, log_n_rows_i, prog_log_i, boundary_i, total_pc_i) =
                base;
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

        // Materialize every base in shard order (shared by both FoldModes). Base-node grouping / leaf
        // proving + the fold + root-verify happen in the FoldMode dispatch AFTER this.
        //
        // TODO(box): re-add base||base-node streaming overlap. The pre-base-fanning pipeline path
        // streamed each wrapped LEAF into `recursive_aggregate_prove_streaming` as its base arrived,
        // overlapping GPU base-proving with the CPU leaf-wrap + fold. For now we collapse to: prove all
        // bases (GPU producers still run concurrently below), then group + fold sequentially.
        let bases: Vec<(Proof<QM31>, GateAirLeafParams)> = if pipeline {
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
            // Tagged bases: (shard_index, result). Unbounded so no producer blocks a peer.
            let (base_tx, base_rx) =
                std::sync::mpsc::channel::<(usize, Result<BaseShardOutput>)>();
            let shard0_base = shard_bases.into_iter().next().unwrap();
            let n_producer_shards = n_shards - 1; // shards 1..n_shards
            let g = base_gpus.min(n_producer_shards.max(1));

            std::thread::scope(|scope| -> Result<()> {
                // PRODUCERS: `g` threads, one per GPU. Producer-shard `s` (s in 1..n_shards) is proved
                // on gpu `s % g` — so shard 1 → gpu 1, shard 2 → gpu 2, …, and gpu 0 handles only
                // shards where `s % g == 0` (for N<=g, NONE — gpu 0 then just does the eager shard 0).
                // This keys the round-robin on the SHARD INDEX (matching the shard→gpu spread the box
                // expects: "shard 1 on gpu 1"), and avoids overloading device 0, which already proved
                // the eager shard 0, with a second base. For g == 1 every shard maps to gpu 0 (single
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
                            // Shards `s` in 1..n_shards with `s % g == gpu` are proved on this gpu.
                            for shard_idx in (1..n_shards).filter(|s| s % g == gpu) {
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

                // CONSUMER (this thread): materialize shard 0 (eager, in-order), then drain tagged
                // bases into their shard slots.
                bases_vec[0] = Some(make_base(&shard0_base));
                for _ in 1..n_shards {
                    let (shard_idx, base) = base_rx.recv().expect("producer hung up early");
                    let base = base?;
                    bases_vec[shard_idx] = Some(make_base(&base));
                }
                for (gpu, p) in producers.into_iter().enumerate() {
                    p.join()
                        .unwrap_or_else(|_| panic!("base producer thread (gpu {gpu}) panicked"));
                }
                Ok(())
            })?;
            // Dense shard-ordered bases (every slot filled by the drain above).
            let bases: Vec<(Proof<QM31>, GateAirLeafParams)> = bases_vec
                .into_iter()
                .enumerate()
                .map(|(i, b)| b.unwrap_or_else(|| panic!("base {i} missing after pipelined proving")))
                .collect();
            eprintln!(
                "gate-air: pipelined {n_shards_bases} bases proved in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            bases
        } else {
            eprintln!("gate-air: building {n_shards_bases} bases (one per shard) ...");
            let t = Instant::now();
            let bases: Vec<(Proof<QM31>, GateAirLeafParams)> = shard_bases
                .iter()
                .map(make_base)
                .collect();
            eprintln!(
                "gate-air: {n_shards_bases} bases built in {:.1}s",
                t.elapsed().as_secs_f64()
            );
            bases
        };

        // ---- Bottom layer + fold + root verification, dispatched on FoldMode ----
        // The bases (`(Proof<QM31>, GateAirLeafParams)` in shard order) are materialized identically
        // above for both modes. Only the BOTTOM layer + the config it is built against + the unpacker
        // differ; the shared up-tree R2 fold (`recursive_aggregate_prove`) is identical.
        //   - BaseFanning (default): group `b` bases per base-node (`prove_base_node`), fold the
        //     base-nodes, unpack via `BaseFanBottom` / `prove_root_verification` — the current path,
        //     behavior-unchanged.
        //   - LeafR1R2: prove one standalone leaf per base (`prove_gate_air_leaf`, b=1), fold the
        //     leaves via `recursive_aggregate_prove_leaves` (level-0 R1 layer + shared R2 fold), unpack
        //     via `LeafBottom` / `prove_root_verification_leaves`.
        // Both branches bind `base_nodes` (the fold's height-1 inputs), `out`, and `rv` for the shared
        // fingerprint block below.
        let (base_nodes, out, rv) = match topo.fold_mode {
            FoldMode::BaseFanning => {
                eprintln!("gate-air: deriving base-fanning config (b={b}) ...");
                let t = Instant::now();
                let config = derive_base_fanning_config(
                    &cfg,
                    &shape_params,
                    b,
                    topo.fold_arity,
                    recursion_log_blowup,
                );
                eprintln!(
                    "gate-air: config derived in {:.1}s (node target qm31_ops={})",
                    t.elapsed().as_secs_f64(),
                    config.agg.node_target_padding_sizes.qm31_ops,
                );
                let config_ref = &config;

                // Group `n_shards_bases` bases (in shard order) into base-nodes per
                // `base_fan_group_sizes`, prove one base-node per group, and collect the fold input
                // (base-nodes) + the unpacker hints (all base outputs in shard order + the per-group
                // reported base-node roots).
                //
                // POOL-PARALLEL: base-nodes are independent + deterministic, so we dispatch one job
                // per group across the recursion `pools` (mirroring the R1 leaf-node layer in
                // `recursive_aggregate_prove_leaves`: pre-slice groups in order, build one job per
                // group, `pools.map` which preserves input order). Each job owns its `group` and only
                // shares an immutable `config_ref` (no shared mutable state), so this changes only
                // wall time, not the proofs — the outputs are collected in the same shard order the
                // serial loop produced.
                let group_and_prove_base_nodes =
                    |bases: Vec<(Proof<QM31>, GateAirLeafParams)>|
                     -> (Vec<TreeProof>, Vec<BaseOutput>, Vec<HashValue<QM31>>) {
                        let sizes = base_fan_group_sizes(bases.len(), b);
                        // Pre-slice the bases into contiguous, shard-ordered groups (one per base-node),
                        // exactly as the serial loop's iterator consumed them.
                        let mut it = bases.into_iter();
                        let groups: Vec<Vec<(Proof<QM31>, GateAirLeafParams)>> = sizes
                            .iter()
                            .map(|&m| (0..m).map(|_| it.next().unwrap()).collect())
                            .collect();
                        // One job per group, dispatched pool-parallel. `pools.map` assigns jobs
                        // round-robin and returns results in input (shard) order.
                        let jobs: Vec<_> = groups
                            .into_iter()
                            .enumerate()
                            .map(|(g, group)| {
                                move || {
                                    let m = group.len();
                                    let tn = Instant::now();
                                    let (bn, outs) = prove_base_node(group, config_ref);
                                    eprintln!(
                                        "gate-air: MEASURE t_base_node[{g}] (arity {m})={:.3}s",
                                        tn.elapsed().as_secs_f64()
                                    );
                                    (bn, outs)
                                }
                            })
                            .collect();
                        let results: Vec<(TreeProof, Vec<BaseOutput>)> = pools.map(jobs);
                        // Flatten in shard order (identical to the serial loop's push order).
                        let mut base_nodes: Vec<TreeProof> = Vec::with_capacity(results.len());
                        let mut all_base_outputs: Vec<BaseOutput> = Vec::new();
                        let mut base_node_roots: Vec<HashValue<QM31>> = Vec::new();
                        for (bn, outs) in results {
                            base_node_roots.push(bn.preprocessed_root.clone());
                            base_nodes.push(bn);
                            all_base_outputs.extend(outs);
                        }
                        (base_nodes, all_base_outputs, base_node_roots)
                    };

                let tg = Instant::now();
                let (base_nodes, all_base_outputs, base_node_roots) =
                    group_and_prove_base_nodes(bases);
                eprintln!(
                    "gate-air: {} base-node(s) proved in {:.1}s",
                    base_nodes.len(),
                    tg.elapsed().as_secs_f64()
                );
                let tf = Instant::now();
                let out = recursive_aggregate_prove(base_nodes.clone(), &config.agg, &pools);
                eprintln!(
                    "gate-air: folded to root in {:.1}s ({} levels)",
                    tf.elapsed().as_secs_f64(),
                    out.n_levels
                );
                eprintln!("gate-air: multiverifier fold OK");

                // Root verification with the single zk-blinding (the only published proof). The
                // unpacker's bottom is the base-fanning bottom: all bases in shard order + `b` + the
                // per-group reported base-node roots.
                let zk = ZkBlind {
                    seed: [7u8; 32],
                    n_padding: config.agg.node_pcs_config.fri_config.n_queries,
                };
                let bottom = BaseFanBottom { bases: all_base_outputs, b, base_node_roots };
                let t = Instant::now();
                let rv = prove_root_verification(
                    &out.root,
                    &bottom,
                    &config.agg,
                    recursion_log_blowup,
                    Some(zk),
                );
                eprintln!(
                    "gate-air: root verification OK in {:.1}s (trace 2^{}, {} leaf outputs unpacked + zk-blinded)",
                    t.elapsed().as_secs_f64(),
                    rv.trace_log_size,
                    rv.leaf_outputs.len()
                );
                (base_nodes, out, rv)
            }
            FoldMode::LeafR1R2 => {
                eprintln!("gate-air: LeafR1R2 mode — deriving leaf/R1/R2 config ...");
                let t = Instant::now();
                let agg = derive_aggregate_config(
                    &cfg,
                    &shape_params,
                    topo.fold_arity,
                    recursion_log_blowup,
                );
                eprintln!(
                    "gate-air: leaf config derived in {:.1}s (node target qm31_ops={})",
                    t.elapsed().as_secs_f64(),
                    agg.node_target_padding_sizes.qm31_ops,
                );

                // One standalone LEAF per base (b=1), in shard order.
                let tg = Instant::now();
                let leaves: Vec<TreeProof> = bases
                    .into_iter()
                    .enumerate()
                    .map(|(i, (proof, params))| {
                        let tl = Instant::now();
                        let leaf = prove_gate_air_leaf(proof, &cfg, &params, &agg);
                        eprintln!(
                            "gate-air: MEASURE t_leaf[{i}]={:.3}s",
                            tl.elapsed().as_secs_f64()
                        );
                        leaf
                    })
                    .collect();
                eprintln!(
                    "gate-air: {} leaf/leaves proved in {:.1}s",
                    leaves.len(),
                    tg.elapsed().as_secs_f64()
                );

                // Fold: level-0 R1 layer over the leaves + shared R2 up-tree fold.
                let tf = Instant::now();
                let out =
                    recursive_aggregate_prove_leaves(leaves.clone(), &agg, &pools);
                eprintln!(
                    "gate-air: folded to root in {:.1}s ({} levels)",
                    tf.elapsed().as_secs_f64(),
                    out.n_levels
                );
                eprintln!("gate-air: multiverifier fold OK");

                // Root verification: unpack from the raw leaves + self-verify.
                let zk = ZkBlind {
                    seed: [7u8; 32],
                    n_padding: agg.node_pcs_config.fri_config.n_queries,
                };
                let bottom = LeafBottom { leaves: leaves.clone() };
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
                // The fold's height-1 inputs are the leaves themselves under LeafR1R2 (b=1); expose
                // them as `base_nodes` for the shared fingerprint block.
                (leaves, out, rv)
            }
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
                format!("n_base_nodes={} n_levels={}", base_nodes.len(), out.n_levels).as_bytes(),
            );
            for (i, bn) in base_nodes.iter().enumerate() {
                hasher.update(format!("base_node[{i}].proof={:?}", bn.proof).as_bytes());
                hasher.update(format!("base_node[{i}].pp_root={:?}", bn.preprocessed_root).as_bytes());
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
            println!("gate-air: recursion_fingerprint[{mode}]={}", hex::encode(digest));
            println!("gate-air: recursion self-verify (fold+root) OK [{mode}]");
        }

        // Recursion path is self-contained (per-shard base proofs are built above); the
        // monolithic full-`samples` base proof + native verify below are not needed here
        // (and the monolithic trace would OOM a small-VRAM GPU), so return now.
        return Ok(());
    }

    // ---- Proving ----
    let max_log_size = log_n_rows.max(RC_LOG_SIZE);
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
        &program, &rows, padded_rows, log_n_rows, n_gates, &boundary,
    );
    tree_builder.extend_evals(to_prover(pp));
    tree_builder.commit(prover_channel);
    eprintln!("gate-air: [phase] preprocessed gen+commit {:.3}s", t_phase.elapsed().as_secs_f64());

    // tree0 EVICTION (GATE_AIR_EVICT_TREE0, default OFF): the tree0 commit is done and tree0's four
    // FULL-DOMAIN preprocessed columns (enabler/shot_id/pc/pc_in_prog, ~4×512 MiB at 2^26) are now
    // resident-but-UNREAD until composition/quotient+OODS in prove_ex. Host-stage them (D2H + free)
    // to drop ~2 GiB off the tree2-commit peak. The composition/quotient (quotient.rs is_staged →
    // rehydrate_block), OODS (poly.rs is_staged → rehydrate_owned), decommit (column.rs
    // host_batch_get) and the composition kernel (cuda_component_prover.rs stash-key passthrough)
    // are all per-column is_staged-guarded and tree-agnostic, so a staged tree0 column is rehydrated
    // transparently — no new wiring. This is the OWNED single-shot tree0 (built inline just above),
    // so eviction is proof-scoped and safe (no shared cross-shard precompute here).
    //
    // SAFETY GUARD: eviction is INCOMPATIBLE with GATE_AIR_STREAM_COMMIT (stream_tree1). The tree1
    // commit that runs next calls fused_commit::clear_stash() at its head (poly.rs), which would WIPE
    // tree0's just-staged stash entries before prove_ex rehydrates them → the readers would then see
    // is_staged()==false and dereference the freed sentinel device_ptr (corruption). Fail LOUD rather
    // than silently corrupt: require the two flags not be combined until the clear_stash-vs-tree0
    // ordering is resolved.
    #[cfg(feature = "cuda")]
    if stwo::prover::backend::cuda::fused_commit::evict_tree0_enabled() {
        assert!(
            std::env::var("GATE_AIR_STREAM_COMMIT").is_err(),
            "GATE_AIR_EVICT_TREE0 is currently incompatible with GATE_AIR_STREAM_COMMIT: the tree1 \
             commit's clear_stash() would wipe tree0's staged entries before prove_ex reads them. \
             Resolve the clear_stash-vs-tree0 ordering before combining these flags."
        );
        // tree0 is trees[0], the OWNED inline tree just committed above. Stage its full-domain
        // columns (eval length == max over tree0's columns) via the exported helper.
        if let Some(stwo::core::utils::MaybeOwned::Owned(tree0)) = commitment_scheme.trees.first_mut()
        {
            let max_size = tree0
                .polynomials
                .iter()
                .map(|p| p.evals.values.size)
                .max()
                .unwrap_or(0);
            stwo::prover::backend::cuda::fused_commit::evict_tree0_columns_at_size(
                tree0.polynomials.iter_mut().map(|p| &mut p.evals.values),
                max_size,
            );
        }
    }

    // Public claim (empty for gate_air; the boundary is reconstructed by the verifier).
    let public_claim = pack_public_claim(&[]);
    prover_channel.mix_felts(&public_claim);

    // Under `cuda`, the dominant trees (main + interaction) are generated ON the GPU and handed to
    // the CudaBackend commit device-to-device (no host upload), UNLESS GATE_AIR_CPU_TRACEGEN=1 forces
    // the legacy CPU-build + upload path. The small columns (multiplicity / program witness / table
    // interactions / preprocessed) always stay on the CPU-generate + upload path.
    #[cfg(feature = "cuda")]
    let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

    // ts-ordering range-check supply table (multiplicity counted from the active accesses' limb
    // lookups). Shard-invariant membership (pos/val) lives in tree0; only the multiplicity is witness.
    let rc_table = build_rc_table(&rows);

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
            &gates_flat, &x_states, &off_lo, &off_hi,
            k as u32, n_gates as u32, samples as u32, padded_rows, log_n_rows,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        eprintln!("gate-air: [phase] main_trace witness gen (GPU K1) {:.3}s", t_phase.elapsed().as_secs_f64());
        // MEM PROBE 1: right after K1 completes, before tree1 commit. d_main is resident here.
        stwo::stwo_cuda::cuda_mem_probe("PROBE1_after_K1");
        // HYPOTHESIS TEST (GATE_AIR_TRIM_AFTER_K1=1, default OFF): does returning pool-cached-freed
        // K1 scratch to the driver free enough contiguous space for tree1 commit to proceed? One-shot
        // sync + cudaMemPoolTrimTo(0). Only touches ALREADY-FREED pool segments; live d_main untouched.
        if std::env::var("GATE_AIR_TRIM_AFTER_K1").is_ok() {
            unsafe { stwo::stwo_cuda::bindings::cuda_pool_trim(); }
            stwo::stwo_cuda::cuda_mem_probe("PROBE1b_after_K1_trim");
        }
        // Fix (b) (`GATE_AIR_FUSED_INTERP`): the 188 `main_dev` columns are BORROWED views into
        // `d_cols` holding UN-interpolated base-domain evals (see `gpu_gen_main_trace_device`). Feed
        // them as `CircleCoefficients` via `extend_polys` so tree1's batched in-place interpolate is
        // SKIPPED (that interpolate would clobber the borrowed views AND double main-trace residency
        // — the OOM at 2^25). The CudaBackend commit's fused per-column-interpolate path handles them
        // (copy-to-temp + per-column b2n), leaving `d_cols` intact for K4. The tiny `small_main`
        // columns stay on the interpolating `extend_evals` path (they are a DIFFERENT size — 2^9 /
        // 2^16 / program-table — so they never share the borrowed main group). Column ORDER is 188
        // main THEN small_main, byte-identical to the flag-off `main_dev.extend(small_main)` order.
        // Off-flag: legacy `extend_evals` over 188+small_main, byte-for-byte unchanged.
        if std::env::var("GATE_AIR_FUSED_INTERP").is_ok() {
            use stwo::prover::poly::circle::CircleCoefficients;
            let main_polys: Vec<CircleCoefficients<ProverBackend>> = main_dev
                .into_iter()
                .map(|e| CircleCoefficients::new(e.values))
                .collect();
            tree_builder.extend_polys(main_polys);
            tree_builder.extend_evals(to_prover(small_main));
        } else {
            let mut main_dev = main_dev;
            main_dev.extend(to_prover(small_main));
            tree_builder.extend_evals(main_dev);
        }
        d_main_cols = Some(d_cols);
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
    // MEM PROBE 2: immediately before the tree1 commit loop (first tree1-column NTT/alloc). This is
    // the driver+pool state ENTERING the LDE loop that OOMs at 2^25.
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE2_before_tree1");
    let t_phase = Instant::now();
    tree_builder.commit(prover_channel);
    eprintln!("gate-air: [phase] tree1 commit (NTT+Merkle) {:.3}s", t_phase.elapsed().as_secs_f64());
    // T1 sub-timer report (grep `[T1]`): decomposes the tree1 commit into NTT / dehydrate D2H /
    // reclaim barrier / build_leaves H2D / absorb. Prints only under GATE_AIR_T1_TIMERS/PROVE_EX_TIMERS.
    #[cfg(feature = "cuda")]
    if stwo::prover::backend::cuda::fused_commit::t1_timers_on() {
        stwo::prover::backend::cuda::fused_commit::t1_report("tree1 commit");
    }

    // GATE_AIR_STREAM_MAIN (device-capacity fix, default OFF): the tree1 commit (which borrowed into
    // the resident K1 buffer) is done, so DEHYDRATE the ~24 GB main-trace device buffer to the host
    // and FREE it now. It is no longer resident for the K4 interaction allocs (`d_inter` etc.) or the
    // tree2 commit — the two phases that OOM at 2^25 with it pinned. K4 rehydrates it (H2D) only for
    // its kernel loop and frees it again before tree2. Flag OFF: kept resident (byte-for-byte the
    // previous path). See `MainTrace::from_k1` / `gpu_gen_interaction_device`.
    #[cfg(feature = "cuda")]
    let mut main_k1: Option<gpu_tracegen::MainTrace> = match d_main_cols.take() {
        Some(d_cols) => Some(gpu_tracegen::MainTrace::from_k1(d_cols).map_err(|e| anyhow::anyhow!(e))?),
        None => None,
    };
    #[cfg(feature = "cuda")]
    stwo::stwo_cuda::cuda_mem_probe("PROBE3_after_stream_main_dehydrate");

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
        // OPTION-0 (GATE_AIR_BOUNDARY_TRIM, default OFF): one-shot pool defrag at the tree1->K4
        // boundary. The streamed tree1 commit just finished; the CUDA pool caches ~23.5 GiB of freed
        // tree1 eval segments (Part-A notrim), which block the fresh contiguous 3 GiB d_inter alloc
        // below at 2^25. This releases the cached (already-freed) segments to the OS ONCE so d_inter
        // fits. Composes with (does NOT require) GATE_AIR_STREAM_MAIN, which additionally removes the
        // ~24 GB main pin. Live buffers (tree0, twiddles) untouched → byte-identical.
        stwo::prover::backend::cuda::fused_commit::boundary_trim_if_enabled();
        // MEM PROBE 3b: AFTER the boundary trim, BEFORE the interaction (K4) allocs. PROBE3 above is
        // measured BEFORE the trim, so it does not show whether the trim reclaimed the pool's
        // cached-freed tree1 segments. This reading is the true free entering the K4/tree2 phase.
        stwo::stwo_cuda::cuda_mem_probe("PROBE3b_after_boundary_trim");
        let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
            main, n_gates as u32, padded_rows, log_n_rows,
            real_rows as u64, (k * n_gates) as u64, &elements,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Some((cols, claimed))
    } else {
        None
    };
    // K4 has consumed the main trace; FREE it NOW (before tree2), not at end-of-prove. Under
    // GATE_AIR_STREAM_MAIN_LOWMEM this `cuMemFree`s the ~24 GB `d_cols` DEVICE buffer that was kept
    // resident across K4 AND synchronizes so the freed memory is reservable by tree2's pool (the
    // device would OOM at 2^25 with it pinned); under plain GATE_AIR_STREAM_MAIN it frees the ~24 GB
    // dehydrated HOST copy early. `free_after_k4` consumes the buffer explicitly (drop alone returns
    // it to the driver but not to tree2's pool without the sync). Flag OFF: `main_k1` is None → no-op.
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
    // rc supply: -multiplicity / combine(TAG_RC, pos, val).
    let (rc_interaction, rc_sum) = {
        let el = elements.rc.clone();
        gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
            el.combine(&[
                ptag(TAG_RC),
                pack_seq(&rc_table.pos, vec_row),
                pack_seq(&rc_table.val, vec_row),
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
        vec![
            BaseField::from_u32_unchecked(rc_table.pos[i]),
            BaseField::from_u32_unchecked(rc_table.val[i]),
        ]
    });
    if rc_sum != rc_expected {
        bail!("rc claimed sum mismatch");
    }

    // Order MUST match the verifier's reconstruction below and build_components.
    let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
    prover_channel.mix_felts(&claimed_sums);

    eprintln!("gate-air: [phase] interaction witness gen+sumcheck {:.3}s", t_phase.elapsed().as_secs_f64());
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
    // Option (a) — arm the interaction (tree2) staging scope for this commit ONLY. Under
    // GATE_AIR_STREAM_INTERACTION this makes `evaluate_polynomials` host-stage the OWNED
    // interaction eval columns (dehydrate + free their device buffers) so the tree2 commit does not
    // hold all 28 eval columns resident (and the composition kernel row-tiles tree2). The arm flag
    // scopes the staging to THIS commit (never tree0/tree1); flag OFF => no-op, so byte-identical.
    #[cfg(feature = "cuda")]
    stwo::prover::backend::cuda::fused_commit::arm_interaction_commit(true);
    tree_builder.commit(prover_channel);
    #[cfg(feature = "cuda")]
    stwo::prover::backend::cuda::fused_commit::arm_interaction_commit(false);
    eprintln!("gate-air: [phase] tree2 commit (NTT+Merkle) {:.3}s", t_phase.elapsed().as_secs_f64());

    let components = build_components(
        log_n_rows,
        program.log_size,
        boundary.log_size,
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
    // T2 sub-timer report (grep `[T2]`): the prove_ex staged-consumer rehydrate H2D vs kernel-compute
    // split for OODS + quotient. Prints only under GATE_AIR_T1_TIMERS/PROVE_EX_TIMERS.
    #[cfg(feature = "cuda")]
    if stwo::prover::backend::cuda::fused_commit::t1_timers_on() {
        stwo::prover::backend::cuda::fused_commit::t2_report("prove_ex");
    }

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
        use circuit_statement::{GateAirStatement, gate_air_components};
        use circuits::blake::HashValue;
        use circuits::context::{Context, TraceContext};
        use circuits::ivalue::NoValue;
        use circuits::ops::Guess;
        use circuits_stark_verifier::proof::{ProofConfig, empty_proof};
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use circuits_stark_verifier::verify::verify as circuit_verify;

        let n_pp = N_PREPROCESSED_COLS;
        let cfg =
            ProofConfig::new(&gate_air_components::<NoValue>(), n_pp, &config, INTERACTION_POW_BITS);
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
        novalue_circuit.check(ctx.values()).expect("gate-air: in-circuit verify FAILED");
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
        vec![
            BaseField::from_u32_unchecked(rc_table.pos[i]),
            BaseField::from_u32_unchecked(rc_table.val[i]),
        ]
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
            .map(|i| Gate { opcode: OP_NOP, target: i as u16, ctrl_a: NO_CTRL, ctrl_b: NO_CTRL })
            .collect();
        // Deterministic but non-trivial 64-byte states; y == x since NOP is the identity.
        let cases: Vec<TestCase> = (0..n_shots)
            .map(|s| {
                let mut st = [0u8; STATE_BYTES];
                for (b, byte) in st.iter_mut().enumerate() {
                    *byte = ((s * 31 + b * 7 + 1) & 0xff) as u8;
                }
                let hex = hex::encode(st);
                TestCase { x_hex: hex.clone(), y_hex: hex }
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
    ) -> (Vec<Row>, BoundaryTable, ProgramTable, usize, u32, u32, stwo::core::pcs::PcsConfig) {
        let (rows, boundary) = build_rows(gates, cases, k).expect("build_rows");
        let real_rows = rows.len();
        let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
        let log_n_rows = padded_rows.ilog2();
        let max_log_size = log_n_rows.max(RC_LOG_SIZE);
        let program = build_program_table(gates, cases.len(), k);
        let config = leaf::leaf_pcs_config(
            max_log_size,
            recursive_aggregate::TopologyConfig::default().base_log_blowup,
        );
        (rows, boundary, program, padded_rows, log_n_rows, max_log_size, config)
    }


    /// Proves ONE tiny gate_air base STARK on the CPU (SimdBackend) for `(gates, cases, k)`, returning
    /// the circuit-form base `Proof<QM31>` + its `GateAirLeafParams` (what `prove_base_node` consumes).
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
    ) {
        use circuit_statement::gate_air_components;
        use circuits::blake::HashValue;
        use circuits::ivalue::NoValue;
        use circuits_stark_verifier::proof::ProofConfig;
        use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
        use stwo::core::fri::FriConfig;
        use stwo::core::pcs::PcsConfig;
        use stwo::prover::{prove_ex, CommitmentSchemeProver};
        use stwo::prover::poly::circle::PolyOps;
        // `pack_public_claim` is imported at module scope (main.rs line 52).

        let n_gates = gates.len();
        let (rows, boundary) = build_rows(gates, cases, k).expect("build_rows");
        let real_rows = rows.len();
        let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
        let log_n_rows = padded_rows.ilog2();
        let max_log_size = log_n_rows.max(RC_LOG_SIZE);
        let program = build_program_table(gates, cases.len(), k);
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
        let rc_table = build_rc_table(&rows);

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
        let pp = build_tree0_columns(&program, &rows, padded_rows, log_n_rows, n_gates, &boundary);
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
        let (program_interaction, program_sum) = gen_program_interaction(&program, &elements.program);
        let (boundary_interaction, boundary_sum) =
            gen_boundary_interaction(&boundary, &elements.qubitmem);
        let (rc_interaction, rc_sum) = {
            let el = elements.rc.clone();
            gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                el.combine(&[
                    ptag(TAG_RC),
                    pack_seq(&rc_table.pos, vec_row),
                    pack_seq(&rc_table.val, vec_row),
                ])
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
            log_n_rows, program.log_size, boundary.log_size,
            &elements, main_sum, program_sum, boundary_sum, rc_sum,
        );
        let prover_refs = components.prover_refs();
        let extended = prove_ex::<TraceBackend, Blake2sM31MerkleChannel>(
            &prover_refs, prover_channel, commitment_scheme, false,
        )
        .expect("base prove_ex");

        // Circuit-form config + params.
        let cfg = ProofConfig::new(&gate_air_components::<NoValue>(), N_PREPROCESSED_COLS, &config, INTERACTION_POW_BITS);
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
        let circuit_proof = proof_from_stark_proof(&extended, &cfg, claim, interaction_pow_nonce, channel_salt);
        (circuit_proof, params, cfg)
    }

    /// End-to-end base-fanning correctness gate (test i): prove `N` tiny gate_air bases on SimdBackend,
    /// group them into base-nodes (arity `b`), fold with R2, prove the root verification (unpack +
    /// self-verify). A green run means: every base verified in-circuit inside its base-node, every
    /// base-node + R2 node verified in-circuit by its parent (the fold), and the root proof verified
    /// in-circuit + reconstructed the tree from the base hints (the final `verify_circuit` sanity
    /// check in `prove_root_verification`). This replaces the removed cairo smoke tests as the
    /// recursion guard.
    #[test]
    fn base_fanning_end_to_end() {
        // RUN-GUARD (laptop-safety): env-gated to GATE_AIR_HEAVY_RECURSION so plain `cargo test` never
        // executes a real recursion prove/verify on a laptop (it can spike RAM / run for minutes). This
        // guard does NOT change the base-fanning PROVING path — with the guard set the body below runs
        // verbatim; it only prevents an accidental heavy run. Run it on the CPU VM with the guard set.
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "base_fanning_end_to_end: SKIPPED (recursion prove/verify). Set \
                 GATE_AIR_HEAVY_RECURSION=1 to run."
            );
            return;
        }
        // Option A: single-base-node root, no R2. `b >= n_bases` so all `n_bases` bases
        // fold into ONE base-node that IS the root — `recursive_aggregate_prove` builds NO R2 node.
        // With `single_base_node=true` the config pads the base-node to its OWN natural target (~2^17
        // with the toy 1-query base PCS, vs the R2-driven ~2^22 of the production fixed point), so the
        // base-node proof + the root-verify are both small → the full prove → fold(trivial) → verify →
        // self-verify roundtrip. It validates the base-fanning-specific WIRING: `b` bases folded into one
        // base-node, the base-node fold-hash (`build_gate_air_base_node_circuit`), and the unpacker
        // reconstructing that base-node from `b` bases + binding it to the self-verified root
        // (`prove_root_verification`'s final `verify_circuit` sanity check). The R2 up-tree fold is the
        // shared multiverifier path, covered byte-identically by recursive_aggregate's topology unit
        // tests (6/6) + the CircuitStatement multiverifier regression (7/7); its full at-scale exercise
        // is the env-gated `base_fanning_end_to_end_with_r2` heavy variant (box only).
        //
        // b=2, N=2 → base_fan_group_sizes(2,2) = [2]: ONE base-node folding 2 bases = the root.
        // Recursion blowup 1 (smallest lift) keeps the base-node + root-verify proofs minimal.
        // fold_arity = the TopologyConfig default (unused here: single-base-node builds no R2 node).
        base_fanning_roundtrip(
            2,
            2,
            true,
            1,
            recursive_aggregate::TopologyConfig::default().fold_arity,
        );
    }

    /// HEAVY (box-only): the full base-fanning roundtrip WITH a real R2 up-tree fold — exercises the
    /// base-node ↔ R2 interface at scale. R2 verifies FOLD_ARITY node-proofs so the node trace floors
    /// at ~2^22 (~several GB–28 GB depending on blowup); OOMs a laptop, so it is env-gated to
    /// GATE_AIR_HEAVY_RECURSION=1 (run on the CPU VM). Plain `cargo test` compiles + SKIPS it.
    #[test]
    fn base_fanning_end_to_end_with_r2() {
        if std::env::var("GATE_AIR_HEAVY_RECURSION").is_err() {
            eprintln!(
                "base_fanning_end_to_end_with_r2: SKIPPED (heavy: R2 node ~2^22). Set \
                 GATE_AIR_HEAVY_RECURSION=1 on the CPU VM to run the with-R2 roundtrip."
            );
            return;
        }
        // b=2, N=3 → [2,1]: a full-b + a short base-node folded by an R2 node into the root.
        // fold_arity = the TopologyConfig default (the production R2 arity the heavy path exercises).
        base_fanning_roundtrip(
            2,
            3,
            false,
            3,
            recursive_aggregate::TopologyConfig::default().fold_arity,
        );
    }

    /// The base-fanning prove→fold→verify→self-verify roundtrip over `n_bases` bases at base-fan arity
    /// `b`, using the TOY per-base PCS from `prove_tiny_base`. `single_base_node` selects the config
    /// mode: `true` requires `b >= n_bases` (one base-node root, no R2 — laptop-safe); `false` allows
    /// an R2 up-tree fold (heavy). Shared by the laptop test + the env-gated heavy variant.
    fn base_fanning_roundtrip(
        b: usize,
        n_bases: usize,
        single_base_node: bool,
        log_blowup_factor: u32,
        fold_arity: usize,
    ) {
        use recursive_aggregate::{
            base_fan_group_sizes, BaseFanBottom, BaseOutput, PoolSet, TreeProof,
            prove_root_verification, recursive_aggregate_prove,
        };
        use leaf::{derive_base_fanning_config_ex, prove_base_node, GateAirLeafParams};
        use circuits_stark_verifier::proof::Proof;

        if single_base_node {
            assert!(b >= n_bases, "single-base-node mode needs b >= n_bases (one base-node root)");
        }

        let (gates, cases, k) = nop_fixture(4, 2, 1);

        // Prove one shared tiny base shape; reuse it for all N bases (identical bases are fine — the
        // fold/unpack don't require distinctness). The base config/params shape is shard-invariant.
        let (proof0, params0, cfg) = prove_tiny_base(&gates, &cases, k);
        let config = derive_base_fanning_config_ex(
            &cfg,
            &params0,
            b,
            fold_arity,
            log_blowup_factor,
            single_base_node,
        );

        // Build N (proof, params) bases (clone the shared one).
        let make_base = || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };

        // Group into base-nodes per base_fan_group_sizes and prove each.
        let sizes = base_fan_group_sizes(n_bases, b);
        let mut base_nodes: Vec<TreeProof> = Vec::new();
        let mut all_base_outputs: Vec<BaseOutput> = Vec::new();
        let mut base_node_roots = Vec::new();
        let mut made = 0usize;
        for &m in &sizes {
            let group: Vec<(Proof<QM31>, GateAirLeafParams)> = (0..m).map(|_| make_base()).collect();
            made += m;
            let (bn, outs) = prove_base_node(group, &config);
            base_node_roots.push(bn.preprocessed_root.clone());
            base_nodes.push(bn);
            all_base_outputs.extend(outs);
        }
        assert_eq!(made, n_bases);
        assert_eq!(all_base_outputs.len(), n_bases);
        assert_eq!(base_node_roots.len(), sizes.len());

        // Fold the base-nodes. In single-base-node mode there is exactly one, so this is a no-op fold
        // (that base-node IS the root); otherwise it builds the R2 up-tree fold. ONE pool caps RAM.
        let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(2);
        let pools = PoolSet::new(1, cores.max(1));
        let out = recursive_aggregate_prove(base_nodes, &config.agg, &pools);

        // Root verification: unpack from the bases + self-verify.
        let bottom = BaseFanBottom { bases: all_base_outputs, b, base_node_roots };
        let rv = prove_root_verification(&out.root, &bottom, &config.agg, log_blowup_factor, None);
        assert_eq!(rv.leaf_outputs.len(), n_bases, "root exposes one H_i per base");
        eprintln!(
            "gate-air: base_fanning roundtrip OK (b={b}, N={n_bases}, single_base_node={single_base_node}, groups={sizes:?}, n_levels={}, root trace 2^{})",
            out.n_levels, rv.trace_log_size
        );
    }

    /// The LEAF/R1/R2 (`FoldMode::LeafR1R2`) prove→fold→verify→self-verify roundtrip over `n_leaves`
    /// standalone gate_air leaves, using the toy per-base PCS from `prove_tiny_base`. Mirrors
    /// `base_fanning_roundtrip` but for the leaf topology: one leaf per base (`prove_gate_air_leaf`),
    /// a level-0 leaf-verifying (R1) layer + shared R2 up-tree fold (`recursive_aggregate_prove_leaves`),
    /// and the leaf unpacker (`prove_root_verification_leaves` / `LeafBottom`). `n_leaves == 1` is a
    /// lone-leaf root (no R1, no R2 — laptop-safe); `n_leaves >= 2` builds the level-0 R1 layer (and, at
    /// `n > k`, an R2 up-tree node) which floors ~2^22 → heavy.
    fn leaf_r1r2_roundtrip(n_leaves: usize, log_blowup_factor: u32, fold_arity: usize) {
        use leaf::{GateAirLeafParams, derive_aggregate_config, prove_gate_air_leaf};
        use recursive_aggregate::{
            LeafBottom, PoolSet, TreeProof, prove_root_verification_leaves,
            recursive_aggregate_prove_leaves,
        };
        use circuits_stark_verifier::proof::Proof;

        let (gates, cases, k) = nop_fixture(4, 2, 1);
        let (proof0, params0, cfg) = prove_tiny_base(&gates, &cases, k);
        let config = derive_aggregate_config(&cfg, &params0, fold_arity, log_blowup_factor);

        let make_base = || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };
        let leaves: Vec<TreeProof> = (0..n_leaves)
            .map(|_| {
                let (p, params) = make_base();
                prove_gate_air_leaf(p, &cfg, &params, &config)
            })
            .collect();
        assert_eq!(leaves.len(), n_leaves);

        let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(2);
        let pools = PoolSet::new(1, cores.max(1));
        let out = recursive_aggregate_prove_leaves(leaves.clone(), &config, &pools);

        let bottom = LeafBottom { leaves };
        let rv = prove_root_verification_leaves(&out.root, &bottom, &config, log_blowup_factor, None);
        assert_eq!(rv.leaf_outputs.len(), n_leaves, "root exposes one H_i per leaf");
        eprintln!(
            "gate-air: leaf_r1r2 roundtrip OK (N={n_leaves}, n_levels={}, root trace 2^{})",
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
        leaf_r1r2_roundtrip(1, 1, recursive_aggregate::TopologyConfig::default().fold_arity);
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
        leaf_r1r2_roundtrip(2, 3, recursive_aggregate::TopologyConfig::default().fold_arity);
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
        let pc = BaseProverPrecompute::new(
            config,
            max_log_size,
            program0,
            &rows0,
            boundary0,
            padded_rows,
            log_n_rows,
            n_gates,
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
        let rc_table = build_rc_table(&rows);

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
                el.combine(&[
                    ptag(TAG_RC),
                    pack_seq(&rc_table.pos, vec_row),
                    pack_seq(&rc_table.val, vec_row),
                ])
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
