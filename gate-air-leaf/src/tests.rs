//! Unit + integration tests for the gate_air base/leaf/recursion pipeline. CI coverage for the
//! prover self-checks moved off the release hot path (`assert_tree0_matches_rebuild`, per-shard
//! claimed-sums balance), the CPU/GPU byte-identity tests, and the per-layer recursion-config drift
//! tests (`recursion_consts_tests`, gate_air's AIR-specific config). The generic recursion
//! prove-machinery tests (streaming byte-identity, panic propagation) live in proving-utils. The
//! `cuda,diag` tests are `#[serial]` (shared-device CUDA module load) and gated off the CPU build.

use super::*;
// Trace/witness-gen items relocated to `tracegen` (Step-1 reorg). Glob is safe here: tests never
// name the gate-local `TRACE_COLUMNS`/`GATE_REL_WIDTH` gpu consts, so no ambiguity with the crate-root
// consts brought in via `use super::*`.
use crate::tracegen::*;
// Per-component evals + relation ids (module reorg); imported from their owning component files.
use crate::components::gate::GateEval;
use crate::components::program::ProgramEval;
use crate::components::qubitmem::QubitMemEval;
use crate::components::range_check::{RangeCheckEval, TAG_RC};
// stwo trace-column types/traits the tests build columns with. Previously reached transitively through
// `main`'s imports; imported here directly now that the trace-gen items (Step-1) and the AIR items
// (Step-2, `air`) live in their own modules.
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
// `FrameworkEval` for the eval trait bound in the tests' constraint checks; `assert_constraints_on_trace`
// for the on-trace constraint helpers (T5).
use stwo_constraint_framework::{assert_constraints_on_trace, FrameworkEval};
// Only the `cuda,diag` byte-identity tests name `Proof<QM31>` now (`prove_tiny_base[_on_gpu]`).
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::fields::qm31::QM31;
// Serializes the concurrent GPU byte-identity tests; only referenced by the `cuda,diag` tests.
#[cfg(all(feature = "cuda", feature = "diag"))]
use serial_test::serial;

/// The rc log-size (`R`) unit tests construct their bases + leaves with. Small (the SIMD minimum
/// `LOG_N_LANES`, not the production `RC_LOG = 25`) so test traces stay tiny and fast; every honest
/// `d = pc - prev_ts` in these tiny fixtures fits `[0, 2^TEST_RC_LOG)`. The base prover and the
/// verifying statement in a given test MUST both use this value (threaded explicitly, never derived).
const TEST_RC_LOG: u32 = LOG_N_LANES;

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
    rc_log: u32,
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
    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    let config = leaf::leaf_pcs_config(max_log_size, prover::BASE_LOG_BLOWUP);
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
/// gen → prove_ex → proof_from_stark_proof) for a self-contained test. Only the `cuda,diag`
/// byte-identity tests consume it now, so it is gated with them off the CPU build.
#[cfg(all(feature = "cuda", feature = "diag"))]
fn prove_tiny_base(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
    rc_log: u32,
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
    // `rc_log` (R) is a test-chosen construction input: the base proof and the leaf statement fed
    // this base MUST use the SAME R (see `GateAirLeafParams::rc_log`).
    let program = build_program_table(gates, cases.len(), k);
    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
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
    // Oracle prove is on `TraceBackend` (== SimdBackend) — the column builders already return
    // `TraceBackend` evals, so they feed the scheme directly (NOT via `to_prover`, which under
    // `cuda` would upload to the CudaBackend and mismatch this SimdBackend scheme).
    tree_builder.extend_evals(pp);
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
    tree_builder.extend_evals(main_trace);
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
    tree_builder.extend_evals(interaction);
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
    // `prove_tiny_base` is the SIMD (`TraceBackend`) oracle even under `cuda`, so it needs
    // `SimdBackend` component provers (not the `ProverBackend`/CudaBackend `prover_refs()`).
    #[cfg(feature = "cuda")]
    let prover_refs = components.prover_refs_simd();
    #[cfg(not(feature = "cuda"))]
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
        rc_log,
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

/// GPU (CudaBackend) twin of [`prove_tiny_base`] for the cross-backend byte-identity tests
/// (T6/T7). Same fixture, same toy PCS, same transcript — the ONLY difference is the prover
/// backend: the commitment scheme + `prove_ex` run on `ProverBackend` (== CudaBackend under
/// `cuda`) with the gate_air GPU constraint kernel registered as the unconditional constraint
/// primary. The witness columns are built on the host (same builders as `prove_tiny_base`) and
/// uploaded via `to_prover`, so the divergence under test is the CONSTRAINT-composition path
/// (GPU kernel vs host delegate), which is what T6/T7 assert is byte-identical.
#[cfg(all(feature = "cuda", feature = "diag"))]
fn prove_tiny_base_on_gpu(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
    rc_log: u32,
) -> (
    circuits_stark_verifier::proof::Proof<QM31>,
    leaf::GateAirLeafParams,
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

    let n_gates = gates.len();
    let (rows, boundary) = build_rows(gates, cases, k).expect("build_rows");
    let real_rows = rows.len();
    let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let log_n_rows = padded_rows.ilog2();
    let program = build_program_table(gates, cases.len(), k);
    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
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

    // Register the gate_air GPU constraint kernel — the unconditional constraint primary on the
    // CudaBackend prove below (no env opt-in any more).
    gate_air_cuda_kernel::register();

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
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);

    // Tree 0.
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

    // Tree 1.
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

    let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
    prover_channel.mix_u64(interaction_pow_nonce);
    let elements = LookupElements::draw(prover_channel);

    // Thread the drawn (z, alpha) to the GPU kernel.
    let (z, alpha_powers) = tracegen::gate_air_relation_m31x4(&elements.qubitmem);
    gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);

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
    let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
    prover_channel.mix_felts(&claimed_sums);

    // Tree 2.
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
    let extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(
        &prover_refs,
        prover_channel,
        commitment_scheme,
        false,
    )
    .expect("gpu base prove_ex");

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
        rc_log,
        preprocessed_root: pp_root,
        boundary: boundary_xy,
        total_pc,
        program: program_rows_from_table(&program),
        nonce: hiding_nonce(),
    };
    let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];
    let circuit_proof =
        proof_from_stark_proof(&extended, &cfg, claim, interaction_pow_nonce, channel_salt);
    (circuit_proof, params)
}

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
        &elements,
        &main_interaction,
        main_sum,
    );
    // (2) program-consistency table.
    let prog_pp = vec![generate_prog_slot_preprocessed(&program)];
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
    let bnd_pp = generate_boundary_preprocessed(&boundary);
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
    let rc_pp = generate_rc_preprocessed(&rc_table);
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

// PINNED recursion-const capture + per-layer DRIFT tests (box-only, `#[ignore]`d), in their own
// submodule; they rebuild the REAL per-layer verifier config and assert it equals the pinned consts.
mod recursion_consts_tests;

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

/// Component provers bound to `SimdBackend` — the host/SIMD oracle path the cross-backend
/// byte-identity tests (T6/T7) prove against under a `cuda` build (where `ProverBackend` is the
/// CudaBackend). `FrameworkComponent<E>` implements `ComponentProver` for both backends. Only
/// referenced from test code (`prove_tiny_base`), hence `cuda`.
#[cfg(feature = "cuda")]
impl super::Components {
    fn prover_refs_simd(
        &self,
    ) -> Vec<&dyn stwo::prover::ComponentProver<stwo::prover::backend::simd::SimdBackend>> {
        use stwo::prover::backend::simd::SimdBackend;
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.qubitmem as &dyn stwo::prover::ComponentProver<SimdBackend>,
            &self.range_check as &dyn stwo::prover::ComponentProver<SimdBackend>,
        ]
    }
}

/// Assert the main component's AIR constraints (algebraic + logup) directly on the committed trace
/// columns, pinpointing the first violated constraint index. Builds the trees
/// `[preprocessed(empty), main, interaction]` the main `GateEval` expects and runs
/// `assert_constraints_on_trace`. Test-support (drives the `on_trace_constraints_all` test, T5),
/// out of the prove path.
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
