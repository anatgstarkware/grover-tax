//! Test-only support helpers relocated off the production files (main.rs / prover.rs / tests.rs).
//! The whole module is `#[cfg(test)]` (declared in main.rs); individual items keep their own extra
//! cfg gates (`feature = "cuda"` / `feature = "diag"`). Pure relocation — no logic change; consumed
//! by `tests` and `recursion_consts_tests`.

// AIR consts/relation + the crate-root prover items (`ProgramTable`/`build_program_table`/
// `generate_main_trace`/`BASE_LOG_BLOWUP`) are glob-imported. Trace-gen items use a glob (as the
// former `tests.rs` did) so this module's varied cuda/non-cuda helpers never trip an unused-import
// warning on the subset a given build compiles; the gate-local `TRACE_COLUMNS`/`GATE_REL_WIDTH` gpu
// consts are never named here, so no ambiguity with the crate-root consts pulled in via `air::*`.
use crate::air::components::gate::GateEval;
use crate::air::*;
use crate::preprocessed::{
    generate_enabler_preprocessed, generate_pc_in_prog_preprocessed, generate_pc_preprocessed,
    generate_shot_id_preprocessed,
};
use crate::prover::*;
use crate::tracegen::*;
use crate::{tree0_max_log_size, Gate, TestCase};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{assert_constraints_on_trace, FrameworkEval};

// Imports named ONLY by the `cuda,diag` base-prove byte-identity oracles below (not compiled on the
// default SimdBackend test build), so gated to match and avoid unused-import warnings there.
#[cfg(all(feature = "cuda", feature = "diag"))]
use crate::air::components::range_check::TAG_RC;
#[cfg(all(feature = "cuda", feature = "diag"))]
use crate::preprocessed::N_PREPROCESSED_COLS;
#[cfg(all(feature = "cuda", feature = "diag"))]
use crate::{hiding_nonce, program_rows_from_table};
#[cfg(all(feature = "cuda", feature = "diag"))]
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::channel::{Blake2sM31Channel, Channel};
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::fields::qm31::QM31;
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::poly::circle::CanonicCoset;
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::proof_of_work::GrindOps;
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
#[cfg(all(feature = "cuda", feature = "diag"))]
use stwo_constraint_framework::Relation;

// Component provers bound to `SimdBackend`

/// Component provers bound to `SimdBackend` — the host/SIMD oracle path the cross-backend
/// byte-identity tests (T6/T7) prove against under a `cuda` build (where `ProverBackend` is the
/// CudaBackend). `FrameworkComponent<E>` implements `ComponentProver` for both backends. Only
/// referenced from test code (`prove_tiny_base`), hence `cuda`.
#[cfg(feature = "cuda")]
impl crate::Components {
    pub(crate) fn prover_refs_simd(
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

// Fixture + shape helpers

/// A tiny self-consistent fixture: an all-NOP circuit (every gate leaves the state unchanged, so
/// `y == x`), `k` reps, `n_shots` shots. NOP gates still ACCESS their target qubit, so the
/// memory-chain / rc-table / boundary / program machinery is fully exercised. Distinct targets
/// per gate keep the per-address ts chains simple. Returns (gates, cases, k).
pub(crate) fn nop_fixture(
    n_gates: usize,
    n_shots: usize,
    k: usize,
) -> (Vec<Gate>, Vec<TestCase>, usize) {
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

/// Shape params shared by the base-pipeline tests and the recursion-const drift tests: build shard-0
/// rows/boundary/program + pcs config for the fixture, matching the precompute setup above.
#[allow(clippy::type_complexity)]
pub(crate) fn shard0_shape(
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
    let config = crate::leaf::leaf_pcs_config(max_log_size, BASE_LOG_BLOWUP);
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

// Base-prove helpers (cuda,diag byte-identity oracles)

/// Proves ONE tiny gate_air base STARK on the CPU (SimdBackend) for `(gates, cases, k)`, returning
/// the circuit-form base `Proof<QM31>` + its `GateAirLeafParams` (what `prove_gate_air_leaf` consumes).
/// Replicates the `--recurse` single-base CPU prove sequence (tree0 commit → witness/interaction
/// gen → prove_ex → proof_from_stark_proof) for a self-contained test. Only the `cuda,diag`
/// byte-identity tests consume it now, so it is gated with them off the CPU build.
#[cfg(all(feature = "cuda", feature = "diag"))]
pub(crate) fn prove_tiny_base(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
    rc_log: u32,
) -> (
    circuits_stark_verifier::proof::Proof<QM31>,
    crate::leaf::GateAirLeafParams,
    circuits_stark_verifier::proof::ProofConfig,
    circuits::blake::HashValue<QM31>,
) {
    use crate::circuit_statement::gate_air_components;
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
    let prover_refs = components.prover_refs_simd();
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
    let params = crate::leaf::GateAirLeafParams {
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
pub(crate) fn prove_tiny_base_on_gpu(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
    rc_log: u32,
) -> (
    circuits_stark_verifier::proof::Proof<QM31>,
    crate::leaf::GateAirLeafParams,
) {
    use crate::circuit_statement::gate_air_components;
    use circuits::blake::HashValue;
    use circuits::ivalue::NoValue;
    use circuits_stark_verifier::proof::ProofConfig;
    use circuits_stark_verifier::proof_from_stark_proof::proof_from_stark_proof;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::PcsConfig;
    use stwo::prover::backend::CudaBackend as ProverBackend;
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
    let (z, alpha_powers) = crate::gpu_tracegen::gate_air_relation_m31x4(&elements.qubitmem);
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
    let params = crate::leaf::GateAirLeafParams {
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

/// Soundness (base pp-root pin): recompute the canonical base tree0 root at build time purely from the
/// trusted public config (program table, k, n_gates, shard-invariant row shape, boundary layout,
/// rc_log = RC_LOG, base blowup), so it can be compared against a forgeable proof value. Must NOT read
/// the prover's `commitments[0]`. tree0 is shard-invariant (see [`build_tree0_columns`]), so any
/// shard's rows recompute the same root; the build mirrors [`BaseProverPrecompute::new`] exactly, so the
/// result equals the honest committed root by construction. A debug/test rebuild-assert (mirrors
/// `diag::assert_tree0_matches_rebuild`) aborts on any column-order/blowup/lifting/sort divergence.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn canonical_base_preprocessed_root(
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
    use stwo::prover::backend::CudaBackend as ProverBackend;
    use stwo::prover::mempool::BaseColumnPool;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::{CommitmentSchemeProver, CommitmentTreeProver};

    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let pool = BaseColumnPool::<ProverBackend>::new();
    // Same tree0 build as `BaseProverPrecompute::new`.
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

    // Rebuild-assert (debug/test only): an independent fresh-scheme rebuild must give the same root.
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

// On-trace constraint assertions (drive the `on_trace_constraints_all` test, T5)

/// Assert the main component's AIR constraints (algebraic + logup) directly on the committed trace
/// columns, pinpointing the first violated constraint index. Builds the trees
/// `[preprocessed(empty), main, interaction]` the main `GateEval` expects and runs
/// `assert_constraints_on_trace`. Test-support (drives the `on_trace_constraints_all` test, T5),
/// out of the prove path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_main_constraints(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    k: usize,
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
    let n_real = rows.len();
    let pp_vals: Vec<Vec<BaseField>> = vec![
        generate_enabler_preprocessed(n_real, padded_rows)
            .values
            .to_cpu(),
        generate_shot_id_preprocessed(n_real, padded_rows, k, n_gates)
            .values
            .to_cpu(),
        generate_pc_preprocessed(n_real, padded_rows, k, n_gates)
            .values
            .to_cpu(),
        generate_pc_in_prog_preprocessed(n_real, padded_rows, k, n_gates)
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
pub(crate) fn assert_table_constraints<Ev: FrameworkEval + Sync>(
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
