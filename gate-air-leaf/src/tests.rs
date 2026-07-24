//! Unit + integration tests for the gate_air base/leaf/recursion pipeline. CI coverage for the
//! prover self-checks moved off the release hot path (`assert_tree0_matches_rebuild`, per-shard
//! claimed-sums balance), the CPU/GPU byte-identity tests, and the per-layer recursion-config drift
//! tests (`recursion_consts_tests`, gate_air's AIR-specific config). The generic recursion
//! prove-machinery tests (streaming byte-identity, panic propagation) live in proving-utils. The
//! `cuda,diag` tests are `#[serial]` (shared-device CUDA module load) and gated off the CPU build.

use super::*;
// Most trace/witness-gen items are re-exported through `use super::*` (main.rs's own `use
// tracegen::{...}`). The one exception is `gen_main_interaction`, which main.rs imports only on the
// non-cuda path (`#[cfg(not(feature = "cuda"))]`), so under a `cuda` build `on_trace_constraints_all`
// needs it imported explicitly here.
#[cfg(feature = "cuda")]
use crate::tracegen::gen_main_interaction;
// The debug/diag rebuild self-check (relocated to `diag`) — the CI net `tree0_precompute_matches_rebuild`
// calls it directly.
use crate::diag::assert_tree0_matches_rebuild;
// Test-only support helpers (relocated to `test_utils`): fixtures + shape + on-trace constraint asserts,
// and (cuda,diag) the base-prove byte-identity oracles.
use crate::test_utils::{
    assert_main_constraints, assert_table_constraints, nop_fixture, shard0_shape,
};
#[cfg(all(feature = "cuda", feature = "diag"))]
use crate::test_utils::{prove_tiny_base, prove_tiny_base_on_gpu};
// Preprocessed column generators (relocated to `preprocessed`) still named by the remaining tests.
use crate::preprocessed::{
    generate_boundary_preprocessed, generate_prog_slot_preprocessed, generate_rc_preprocessed,
};
// Per-component evals + relation ids (module reorg); imported from their owning component files.
use crate::components::program::ProgramEval;
use crate::components::qubitmem::QubitMemEval;
use crate::components::range_check::{RangeCheckEval, TAG_RC};
// Serializes the concurrent GPU byte-identity tests; only referenced by the `cuda,diag` tests.
#[cfg(all(feature = "cuda", feature = "diag"))]
use serial_test::serial;

/// The rc log-size (`R`) unit tests construct their bases + leaves with. Small (the SIMD minimum
/// `LOG_N_LANES`, not the production `RC_LOG = 25`) so test traces stay tiny and fast; every honest
/// `d = pc - prev_ts` in these tiny fixtures fits `[0, 2^TEST_RC_LOG)`. The base prover and the
/// verifying statement in a given test MUST both use this value (threaded explicitly, never derived).
const TEST_RC_LOG: u32 = LOG_N_LANES;

/// (#1a) The precompute's cached tree-0 must match an independent rebuild (root + column count +
/// per-column committed sizes). This is the CI net for `assert_tree0_matches_rebuild`, which is
/// now debug-gated off the release hot path.
#[test]
fn tree0_precompute_matches_rebuild() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let n_gates = gates.len();
    let (rows0, boundary0, program0, padded_rows, log_n_rows, max_log_size, config) =
        shard0_shape(&gates, &cases, k, TEST_RC_LOG);
    // Must match the R `shard0_shape` sized `config`/`max_log_size` with (same base construction).
    let rc_log = TEST_RC_LOG;
    // Under `cuda`, `BaseProverPrecompute::new` also uploads the shard-invariant N3 device
    // buffers (gate list + RcIndex offsets), so the flat inputs are built here.
    #[cfg(feature = "cuda")]
    let (gates_flat0, _x0, off_lo0, off_hi0) = {
        let rc_lo = build_rc_lo();
        gpu_flat_inputs(&gates, &cases, &rc_lo, &rc_lo).expect("gpu_flat_inputs")
    };
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
        #[cfg(feature = "cuda")]
        &gates_flat0,
        #[cfg(feature = "cuda")]
        &off_lo0,
        #[cfg(feature = "cuda")]
        &off_hi0,
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
        shard0_shape(&gates, &cases, k, TEST_RC_LOG);
    let rc_table = build_rc_table(&rows, TEST_RC_LOG);

    // Draw the LogUp relation exactly as the prover does (salt=0, then config is mixed in the
    // real path; for a self-contained balance check the challenge just needs to be consistent
    // across all four sums + the public terms, which one draw guarantees).
    let mut channel = Blake2sM31Channel::default();
    channel.mix_felts(&[BaseField::from_u32_unchecked(0).into()]);
    let elements = LookupElements::draw(&mut channel);

    let (_mi, main_sum) = gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
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

/// (T5) `on_trace_constraints_all` — ALL FOUR components' AIR constraints (main, program,
/// boundary, rc) evaluate to zero on the committed trace (no FRI / proof). Extends the main-only
/// `main_explicit_constraints_zero_on_valid_rows` to the missing 3 table components, via the
/// `assert_main_constraints` / `assert_table_constraints` helpers the removed `GATE_AIR_ASSERT`
/// prove-path hook used. Laptop `cargo test` (CPU/Simd fixture). A violated constraint panics
/// with its first-violated index.
#[test]
fn on_trace_constraints_all() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let n_gates = gates.len();
    let (rows, boundary, program, padded_rows, log_n_rows, _max_log_size, _config) =
        shard0_shape(&gates, &cases, k, TEST_RC_LOG);
    let rc_table = build_rc_table(&rows, TEST_RC_LOG);

    // Draw the LogUp relation exactly as `shard_claimed_sums_net_to_public` does.
    let mut channel = Blake2sM31Channel::default();
    channel.mix_felts(&[BaseField::from_u32_unchecked(0).into()]);
    let elements = LookupElements::draw(&mut channel);

    let (main_interaction, main_sum) =
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
    let (program_interaction, program_sum) = gen_program_interaction(&program, &elements.program);
    let (boundary_interaction, boundary_sum) =
        gen_boundary_interaction(&boundary, &elements.qubitmem);
    let (rc_interaction, rc_sum) = {
        let el = elements.rc.clone();
        gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
            el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
        })
    };

    // (1) main component.
    assert_main_constraints(
        &rows,
        padded_rows,
        log_n_rows,
        n_gates,
        k,
        &elements,
        &main_interaction,
        main_sum,
    );
    // (2) program-consistency table.
    let prog_pp = vec![generate_prog_slot_preprocessed(program.log_size)];
    let prog_wit = generate_program_witness(&program);
    assert_table_constraints(
        program.log_size,
        &prog_pp,
        &prog_wit,
        &program_interaction,
        program_sum,
        ProgramEval {
            log_size: program.log_size,
            elements: elements.program.clone(),
        },
    );
    // (3) qubit-memory boundary table.
    let bnd_pp = generate_boundary_preprocessed(boundary.n_shots, boundary.log_size);
    let bnd_wit = generate_boundary_witness(&boundary);
    assert_table_constraints(
        boundary.log_size,
        &bnd_pp,
        &bnd_wit,
        &boundary_interaction,
        boundary_sum,
        QubitMemEval {
            log_size: boundary.log_size,
            elements: elements.qubitmem.clone(),
        },
    );
    // (4) ts-ordering range-check supply table.
    let rc_pp = generate_rc_preprocessed(rc_table.log_size);
    let rc_wit = generate_rc_witness(&rc_table);
    assert_table_constraints(
        rc_table.log_size,
        &rc_pp,
        &rc_wit,
        &rc_interaction,
        rc_sum,
        RangeCheckEval {
            log_size: rc_table.log_size,
            elements: elements.rc.clone(),
        },
    );
}

// ========================================================================
// Box GPU tests (T1a/T1b/T2/T3/T4/T6/T7). Gated on `cuda,diag` — they need the device / real
// params and are VALIDATED ON THE BOX (`cargo test --features cuda,diag`). On a laptop they are
// compiled but not built into the default (SimdBackend) test binary. Each replaces a removed
// prove-path flag with a real `#[test]` (see DIAG_FLAG_SEPARATION_SCOPE.md coverage matrix).
// ========================================================================

/// (T1a) `k1_trace_identity` — GPU K1 main trace == CPU recompute, cell-by-cell + rc histogram.
/// Promotes the `tracegen::k1_byte_identity` harness (formerly reachable only via the removed
/// `GATE_AIR_GPU_TEST=k1` prove-path flag) to a real test.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[serial]
fn k1_trace_identity() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let rc_lo = build_rc_lo();
    tracegen::k1_byte_identity(&gates, &cases, k, &rc_lo, &rc_lo)
        .expect("GPU K1 main trace != CPU recompute");
}

/// (T1b) `k4_interaction_identity` — GPU K4 LogUp interaction == CPU recompute. Promotes the
/// `tracegen::k4_byte_identity` harness (formerly `GATE_AIR_GPU_TEST=k4`) to a real test.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[serial]
fn k4_interaction_identity() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let rc_lo = build_rc_lo();
    tracegen::k4_byte_identity(&gates, &cases, k, &rc_lo, &rc_lo)
        .expect("GPU K4 interaction != CPU recompute");
}

/// (T2) `base_precompute_identity` — a base shard proved with the shard-invariant precompute
/// (`Some(pc)`) is BYTE-IDENTICAL to one proved rebuild-per-shard (`None`). Replaces the removed
/// `GATE_AIR_NO_BASE_PRECOMPUTE` A/B arm + `GATE_AIR_BASE_PROOF_HASH` compare; extends
/// `tree0_precompute_matches_rebuild` (tree0 only) to the full base proof via
/// `fingerprint::base_proof_fingerprint`.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[ignore = "box-only: GPU base prove"]
#[serial]
fn base_precompute_identity() {
    use prover::{prove_base_shard, BaseProverPrecompute};
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let n_gates = gates.len();
    let rc_lo = build_rc_lo();
    let (rows0, boundary0, program0, padded_rows, log_n_rows, max_log_size, config) =
        shard0_shape(&gates, &cases, k, RC_LOG);
    let (gates_flat0, _x0, off_lo0, off_hi0) =
        gpu_flat_inputs(&gates, &cases, &rc_lo, &rc_lo).expect("gpu_flat_inputs");
    let pc = BaseProverPrecompute::new(
        config,
        max_log_size,
        program0,
        &rows0,
        boundary0,
        padded_rows,
        log_n_rows,
        n_gates,
        RC_LOG,
        &gates_flat0,
        &off_lo0,
        &off_hi0,
    )
    .expect("precompute new");

    // Prove shard 0 both ways and fingerprint the two base proofs; they must be byte-identical.
    let with_pc = prove_base_shard(
        Some(&pc),
        &cases,
        &gates,
        k,
        n_gates,
        prover::BASE_LOG_BLOWUP,
        &rc_lo,
    )
    .expect("prove_base_shard(Some)");
    let rebuild = prove_base_shard(
        None,
        &cases,
        &gates,
        k,
        n_gates,
        prover::BASE_LOG_BLOWUP,
        &rc_lo,
    )
    .expect("prove_base_shard(None)");
    assert_eq!(
        fingerprint::base_proof_fingerprint(std::slice::from_ref(&with_pc)),
        fingerprint::base_proof_fingerprint(std::slice::from_ref(&rebuild)),
        "base precompute-ON base proof != rebuild-per-shard base proof"
    );
}

/// (T4) `incircuit_self_verify` — a standalone monolithic gate_air base proof verifies
/// IN-CIRCUIT: the NoValue-shaped `circuit_verify` circuit `.check()`s the real assignment.
/// Replaces the removed `GATE_AIR_INCIRCUIT` prove-path hook. Uses the deterministic
/// `hiding_nonce()` (its fixed default is deterministic, so no override is needed here).
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[ignore = "box-only: in-circuit verify build"]
#[serial]
fn incircuit_self_verify() {
    use circuit_statement::{gate_air_components, GateAirStatement};
    use circuits::blake::HashValue;
    use circuits::context::{Context, TraceContext};
    use circuits::ivalue::NoValue;
    use circuits::ops::Guess;
    use circuits_stark_verifier::proof::empty_proof;
    use circuits_stark_verifier::verify::verify as circuit_verify;
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let n_gates = gates.len();
    // `prove_tiny_base` builds the base + returns the circuit-form config; rebuild the shape
    // scalars it used so the statement matches.
    let (_rows, boundary, program, _padded, log_n_rows, _mls, _cfg_pcs) =
        shard0_shape(&gates, &cases, k, TEST_RC_LOG);
    let (circuit_proof, params, cfg, _canon) = prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let pp_root: HashValue<SecureField> = params.preprocessed_root.clone();
    let boundary_xy = params.boundary.clone();
    let total_pc = params.total_pc;

    // NoValue circuit shape.
    let novalue_circuit = {
        let empty = empty_proof(&cfg);
        let mut nv = Context::<NoValue>::default();
        let pv = empty.guess(&mut nv);
        let stmt = GateAirStatement::<NoValue>::new(
            &mut nv,
            log_n_rows,
            program.log_size,
            boundary.log_size,
            params.rc_log,
            pp_root.clone(),
            boundary_xy.clone(),
            total_pc,
            program_rows_from_table(&program),
            hiding_nonce(),
        );
        circuit_verify(&mut nv, &pv, &cfg, &stmt);
        let h_p = stmt.compute_h_p(&mut nv);
        for w in h_p.iter() {
            nv.mark_as_unused(*w.get());
        }
        nv.finalize(false).context.circuit
    };
    // Real assignment checked against the NoValue shape.
    let mut ctx = TraceContext::default();
    let pv = circuit_proof.guess(&mut ctx);
    let _ = gate_air_components::<NoValue>();
    let stmt = GateAirStatement::new(
        &mut ctx,
        log_n_rows,
        program.log_size,
        boundary.log_size,
        params.rc_log,
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
    let _ = n_gates;
    novalue_circuit
        .check(ctx.values())
        .expect("gate-air: in-circuit verify FAILED");
}

/// (T6) `gpu_vs_host_constraints_identity` — the GPU-kernel constraint composition (CudaBackend,
/// registered kernel = unconditional primary) yields a base proof BYTE-IDENTICAL to the audited
/// host-delegate (SimdBackend, which never calls the MAIN-host-delegate panic). Replaces the
/// removed `CUDA_GPU_CONSTRAINTS` / `CUDA_CONSTRAINT_CPU_FALLBACK` A/B toggles. Exercises the two
/// prover paths ONLY — no core verification is patched.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[ignore = "box-only: GPU base prove"]
#[serial]
fn gpu_vs_host_constraints_identity() {
    // SimdBackend (host constraint delegate) oracle: `prove_tiny_base` always proves on
    // `TraceBackend` == SimdBackend, so its serialized StarkProof is the host-path proof.
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let (host_proof, _p, _cfg, _r) = prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    // GPU path: the same tiny base proved on the CudaBackend with the registered gate_air kernel
    // as the unconditional constraint primary. `prove_tiny_base_on_gpu` mirrors `prove_tiny_base`
    // on `ProverBackend` (== CudaBackend under cuda), so byte-identity of the serialized proofs
    // is the GPU-kernel == host-delegate check.
    let (gpu_proof, _p2) = prove_tiny_base_on_gpu(&gates, &cases, k, TEST_RC_LOG);
    assert_eq!(
        format!("{:?}", host_proof),
        format!("{:?}", gpu_proof),
        "GPU-kernel constraint composition != host-delegate (SimdBackend) base proof"
    );
}

/// (T7) `full_proof_gpu_vs_simd_identity` — the FULL serialized base proof from the CudaBackend
/// (GPU) path equals the SimdBackend (SIMD) path. The permanent form of the #88 manual
/// `GATE_AIR_PROOF_HASH` cross-backend oracle. Uses the deterministic `hiding_nonce()` default.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[ignore = "box-only: GPU base prove"]
#[serial]
fn full_proof_gpu_vs_simd_identity() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let (simd_proof, _p, _cfg, _r) = prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let (gpu_proof, _p2) = prove_tiny_base_on_gpu(&gates, &cases, k, TEST_RC_LOG);
    assert_eq!(
        format!("{:?}", simd_proof),
        format!("{:?}", gpu_proof),
        "full proof GPU (CudaBackend) != SimdBackend"
    );
}
