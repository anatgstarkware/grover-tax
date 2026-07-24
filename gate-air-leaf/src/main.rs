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

mod air; // AIR assembly (Components + shared LookupElements/GateRel + preprocessed layout + AIR consts).
mod circuit_statement; // In-circuit verifier of the gate_air STARK proof.
mod components; // Per-component FrameworkEvals + their relation ids (gate/program/qubitmem/range_check).
mod fingerprint; // Proof-fingerprint helpers behind the env-gated hooks (distinct from the `diag` feature).
mod leaf;
mod preprocessed; // PUBLIC preprocessed columns (verifier-known, positional): identity/order + shape-derived generators.
mod prover; // Base (per-shard) gate_air prover, extracted from this file (mirrors `leaf.rs`).
mod recursion_consts;
mod tracegen; // Trace / witness generation (CPU column builders + LogUp gens + absorbed GPU trace-gen). // PINNED recursion constants, keyed per operating point.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
#[allow(unused_imports)]
use itertools::Itertools;
use num_traits::Zero;
use serde::Deserialize;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::CommitmentSchemeVerifier;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
// Trace-gen backend is ALWAYS SimdBackend (columns are built with CPU column ops); `to_prover`
// bridges to `ProverBackend` at the `extend_evals` boundary (a real host->device upload under cuda).
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo::core::proof_of_work::GrindOps;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::{prove_ex, CommitmentSchemeProver};
use stwo_constraint_framework::Relation;

use air::*;
use components::range_check::TAG_RC;
use preprocessed::N_PREPROCESSED_COLS;
use prover::*;

// Trace/witness-generation items relocated to `tracegen` (Step-1 module reorg). Imported explicitly
// (not a glob) so the gate-local `tracegen::{TRACE_COLUMNS,GATE_REL_WIDTH}` gpu consts don't collide
// with the crate-root shared consts of the same name.
use tracegen::{
    boundary_public_sum, boundary_public_term, build_rc_lo, build_rc_table, build_rows,
    build_tree0_columns, gen_boundary_interaction, gen_program_interaction, gen_table_interaction,
    generate_boundary_witness, generate_program_witness, generate_rc_witness, pack_seq,
    program_claimed_sum, program_public_term, state_to_limbs, table_public_sum, to_prover,
    BoundaryTable, RcIndex, Row,
};
// The CPU main-interaction generator is used only on the non-cuda prove path (the cuda path uses the
// GPU K4 kernel), so gate its import to match.
#[cfg(not(feature = "cuda"))]
use tracegen::gen_main_interaction;
#[cfg(feature = "cuda")]
use tracegen::gpu_flat_inputs;

/// Default shots per base shard (partition knob). `n_shards = ceil(samples / shots_per_shard)`.
/// Env override `RECURSION_SHARD_SHOTS` (`> 0`) is parsed in `main`.
const SHOTS_PER_SHARD: usize = 2;

/// Parse env var `k` into `T`, `None` if unset or unparseable. The binary's sole env-parsing helper.
fn parse_env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|s| s.parse().ok())
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
    op.pinned().unpacker_config()
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
    // `k4_interaction_identity` (T1b), which call the `tracegen::k1_byte_identity` /
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
    use circuits_stark_verifier::proof::{empty_proof, Proof, ProofConfig};
    use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
    use leaf::{build_gate_air_leaf_circuit, GateAirLeafParams};
    use recursive_aggregate::pools::PoolSet;
    use recursive_aggregate::precomputes::RecursionPrecompute;
    use recursive_aggregate::prove::recursive_aggregate_prove_leaves;
    use recursive_aggregate::prove_streaming::recursive_aggregate_prove_leaves_streaming;
    use recursive_aggregate::root_prover::{prove_root_verification_leaves, LeafBottom, ZkBlind};
    use recursive_aggregate::AggregateOutput;
    use recursive_aggregate::TreeProof;
    use stwo::core::fields::qm31::QM31;

    // The base-proof tuple `prove_base_shard` returns (defined in `base.rs`, imported here so the
    // fold block's unqualified references resolve). `prove_ex` yields `ExtendedStarkProof<MC::H>`
    // with `MC::H = Blake2sMerkleHasher`, so this is backend-independent (cuda vs simd).
    use prover::BaseShardOutput;

    // Free base/partition params, honoring the existing env sweep knobs (BASE_BLOWUP,
    // RECURSION_SHARD_SHOTS). Defaults reproduce the current production values, so this is a
    // byte-identical no-op. Threaded through the derive/prove calls below.
    let base_log_blowup: u32 = parse_env("BASE_BLOWUP").unwrap_or(prover::BASE_LOG_BLOWUP);
    let shots_per_shard_raw: usize = parse_env::<usize>("RECURSION_SHARD_SHOTS")
        .filter(|&n| n > 0)
        .unwrap_or(SHOTS_PER_SHARD);

    // Shard partition: equal-sized shards of `shots_per_shard` shots; ragged final shard is
    // padded (below) so all shards share the leaf circuit shape.
    let shots_per_shard: usize = shots_per_shard_raw.min(samples);
    let n_shards = samples.div_ceil(shots_per_shard);
    eprintln!(
        "gate-air: sharding {samples} shots into {n_shards} shard(s) of {shots_per_shard} shot(s) each \
         (final shard padded by shot-repeat if ragged)"
    );

    // The pinned operating point (k, shots_per_shard → N). Gates to the 3 Tanuj curve points; any other
    // (k, shots) panics (unsupported). The recursion consts (roots + unpacker config) key off this.
    let op = recursion_consts::OperatingPoint::from_params(k, shots_per_shard_raw);

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
    let (base_precompute, cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce): (
        Option<std::sync::Arc<BaseProverPrecompute>>,
        ProofConfig,
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
        base_log_blowup,
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
    let recursion_pre = {
        // Production: build node shapes only for the arities this point's fold uses. The runtime
        // config is assembled + cached inside the returned precompute (prove entry points read it
        // from there); gate-air holds only the pinned const `op.pinned()`.
        let pre = recursive_aggregate::precomputes::build_recursion_precompute(
            build_gate_air_leaf_circuit::<NoValue>(empty_proof(&cfg), &cfg, &shape_params),
            op.pinned(),
            false,
        );
        eprintln!(
            "gate-air: leaf-recursion config + precompute built up front in {:.1}s (node target qm31_ops={})",
            t_cfg.elapsed().as_secs_f64(),
            pre.node_target_padding_sizes().qm31_ops,
        );
        pre
    };
        Ok((cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce))
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
            let base_blowup: u32 = base_log_blowup;
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
            crate::diag::assert_tree0_matches_rebuild(&pc, &rows0, n_gates);
            eprintln!(
                "gate-air: base precompute built (tree0+twiddles+N1{}) in {:.3}s",
                if cfg!(feature = "cuda") { "+N3" } else { "" },
                t_pc.elapsed().as_secs_f64()
            );
            Some(std::sync::Arc::new(pc))
        };

        // --- JOIN: both precomputes complete here, before any proving. ---
        let (cfg, recursion_pre, boundary_log_size, leaf_program, leaf_nonce) = cpu_build
            .join()
            .expect("recursion config/precompute thread panicked")?;
        Ok((
            base_precompute,
            cfg,
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
                let visible = tracegen::backend_device_count();
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
            shard_bases.push(prover::prove_base_shard(
                base_precompute_ref,
                shard_cases,
                gates,
                k,
                n_gates,
                base_log_blowup,
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
    // Byte-neutral tuning knob: optionally deprioritize pool workers (`None` = off).
    let pool_nice = std::env::var("RECURSION_POOL_NICE").ok();
    let pools = PoolSet::new(n_pools, threads_per_pool, pool_nice);

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
            // borrows (read-only `gates`/`rc_lo_index`/`shard_case_sets`, Copy `base_precompute` and
            // `base_log_blowup`) outlive this `thread::scope`.
            let gates_ref = &gates;
            let rc_lo_index_ref = &rc_lo_index;
            let shard_case_sets_ref = &shard_case_sets;
            // OODS pool thread count — byte-neutral tuning knob (default 4). Read once here (this
            // region owns the producer loop) so the value is passed into each device-bound pool.
            #[cfg(feature = "gpu-cuda")]
            let oods_pool_threads: usize = parse_env::<usize>("GATE_AIR_OODS_POOL_THREADS")
                .filter(|&n| n > 0)
                .unwrap_or(4);
            let producers: Vec<_> = (0..g)
                .map(|gpu| {
                    let base_tx = base_tx.clone();
                    scope.spawn(move || {
                        #[cfg(feature = "cuda")]
                        tracegen::set_base_gpu(gpu);
                        // Class-1 SIGSEGV fix: a PRIVATE rayon pool whose workers are all bound to
                        // THIS producer's device (`gpu`), so the OODS-phase fan-outs inside
                        // `prove_base_shard` (`pcs/mod.rs` `build_weights_hash_map` par_iter + OODS
                        // `par_map_cols`) run on device `gpu` instead of the global pool's
                        // device-0 workers (which deref device-`gpu` pointers => SIGSEGV). Built
                        // once per producer; each `prove_base_shard` runs inside `pool.install`.
                        #[cfg(feature = "gpu-cuda")]
                        let oods_pool = tracegen::build_device_bound_pool(gpu, oods_pool_threads);
                        // Shards `s` in 0..n_shards with `s % g == gpu` are proved on this gpu.
                        for shard_idx in (0..n_shards).filter(|s| s % g == gpu) {
                            let tb = Instant::now();
                            #[cfg(feature = "gpu-cuda")]
                            let r = oods_pool.install(|| {
                                prover::prove_base_shard(
                                    base_precompute_ref,
                                    &shard_case_sets_ref[shard_idx],
                                    gates_ref,
                                    k,
                                    n_gates,
                                    base_log_blowup,
                                    rc_lo_index_ref,
                                )
                            });
                            #[cfg(not(feature = "gpu-cuda"))]
                            let r = prover::prove_base_shard(
                                base_precompute_ref,
                                &shard_case_sets_ref[shard_idx],
                                gates_ref,
                                k,
                                n_gates,
                                base_log_blowup,
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
                        leaf_rx, n_shards, build, pre, pools_ref,
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
        // the pipeline overlap wrapped leaves early); the runtime config is cached in the precompute.

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
                    recursive_aggregate_prove_leaves(bases, build, recursion_pre_ref, &pools);
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
            n_padding: recursion_pre_ref.node_pcs_config().fri_config.n_queries,
        };
        let bottom = LeafBottom {
            leaves: leaves.clone(),
        };
        // Pinned per-N unpacker config (the trusted verify const); the prover's one-shot unpacker
        // tree asserts its committed root == this const's `preprocessed_root`.
        let unpacker_config = pinned_unpacker_config(op, leaves.len());
        let t = Instant::now();
        let rv = prove_root_verification_leaves(
            &out.root,
            &bottom,
            recursion_pre_ref,
            &unpacker_config,
            Some(zk),
        );
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
    // Base blowup is a sweep knob (env BASE_BLOWUP overrides the default); resolved identically to
    // the recursion path so this monolithic (non-fold) path uses the same value.
    let base_blowup: u32 = parse_env("BASE_BLOWUP").unwrap_or(prover::BASE_LOG_BLOWUP);
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
        let (main_dev, _lo, d_cols) = tracegen::gpu_gen_main_trace_device(
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
    let main_k1: tracegen::MainTrace =
        tracegen::MainTrace::from_k1(d_main_cols).map_err(|e| anyhow::anyhow!(e))?;

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
        let (z, alpha_powers) = tracegen::gate_air_relation_m31x4(&elements.qubitmem);
        gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
    }

    // Interaction traces.
    let t_phase = Instant::now();
    // Device K4 (under `cuda`): the 24 main-interaction columns are generated on the GPU using the
    // REAL drawn `elements` and handed to the commit device-resident (no upload); `claimed_sum`
    // becomes `main_sum`. The CPU (SimdBackend) interaction gen survives only on the non-cuda build.
    #[cfg(feature = "cuda")]
    let main_interaction_device = {
        let (cols, claimed) = tracegen::gpu_gen_interaction_device(
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

fn normalize(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

#[cfg(any(debug_assertions, test))]
mod diag; // Debug/diag-only runtime self-checks moved off the release prover hot path.
#[cfg(test)]
mod test_utils; // Test-only support helpers (fixtures, base-prove oracles, on-trace asserts).
#[cfg(test)]
mod tests;
