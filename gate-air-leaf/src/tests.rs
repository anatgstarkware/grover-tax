//! Unit + integration tests for the gate_air base/leaf/recursion pipeline. CI coverage for the
//! prover self-checks moved off the release hot path (`assert_tree0_matches_rebuild`, per-shard
//! claimed-sums balance) plus the CPU/GPU byte-identity and recursion-roundtrip tests. The
//! `cuda,diag` tests are `#[serial]` (shared-device CUDA module load) and gated off the CPU build.

use super::*;
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
    let config = leaf::leaf_pcs_config(max_log_size, TopologyConfig::default().base_log_blowup);
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
    let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.qubitmem);
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

/// The leaf-recursion prove→fold→verify→self-verify roundtrip over `n_leaves`
/// standalone gate_air leaves, using the toy per-base PCS from `prove_tiny_base`: one leaf per base
/// (`prove_gate_air_leaf`),
/// a level-0 leaf-verifying (level1-node) layer + shared fold-node up-tree fold (`recursive_aggregate_prove_leaves`),
/// and the leaf unpacker (`prove_root_verification_leaves` / `LeafBottom`). `n_leaves == 1` is a
/// lone-leaf root (no level1-node, no fold-node — laptop-safe); `n_leaves >= 2` builds the level-0 level1-node layer (and, at
/// `n > k`, an fold-node up-tree node) which floors ~2^22 → heavy.
fn leaf_recursion_roundtrip(n_leaves: usize, log_blowup_factor: u32, fold_arity: usize) {
    use circuit_verifier::verify::{verify_circuit, CircuitPublicData};
    use circuits_stark_verifier::proof::Proof;
    use leaf::{build_recursion_precompute, prove_gate_air_leaf, GateAirLeafParams};
    use recursion_consts_tests::derive_aggregate_config;
    use recursive_aggregate::pools::PoolSet;
    use recursive_aggregate::prove::recursive_aggregate_prove_leaves;
    use recursive_aggregate::root_prover::{prove_root_verification_leaves, LeafBottom};
    use recursive_aggregate::test_utils::unpacker_verify_config;
    use recursive_aggregate::TreeProof;

    let (gates, cases, k) = nop_fixture(4, 2, 1);
    // leaf-recursion uses the leaf preprocessed root (not the base-fanning canonical base root), so the
    // recomputed canonical base pp root is unused here. Tiny-base tests are NOT a pinned curve point;
    // they build the (real) config + unpacker config via the recompute helpers, not the pinned consts.
    let (proof0, params0, cfg, _canonical_base_pp_root) =
        prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let op = recursion_consts::OperatingPoint::K500N174; // placeholder key (tests recompute their own roots)
    let config = derive_aggregate_config(
        &cfg,
        &params0,
        fold_arity,
        log_blowup_factor,
        log_blowup_factor,
    );
    // test: build all arities (placeholder op.n() differs from the small fold N)
    let pre = build_recursion_precompute(&config, op, &cfg, &params0, true);

    let make_base = || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };
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
    // Recompute the unpacker config for this tiny config (production supplies the pinned const).
    let unpacker_config = unpacker_verify_config(n_leaves, &config, log_blowup_factor, None);
    let rv = prove_root_verification_leaves(&out.root, &bottom, &config, &unpacker_config, None);
    assert_eq!(
        rv.leaf_outputs.len(),
        n_leaves,
        "root exposes one H_i per leaf"
    );

    // TRUSTED FINAL VERIFY (step 3): check `rv` against the (recomputed) canonical unpacker config —
    // its canonical root pins every baked child root; `rv.leaf_outputs` are caller-committed. `None`
    // blinding matches the unblinded test proof above.
    let output_values: Vec<SecureField> = rv.leaf_outputs.iter().flatten().copied().collect();
    verify_circuit(
        unpacker_config,
        rv.proof.clone(),
        CircuitPublicData { output_values },
    )
    .expect("trusted gate_air root verification failed (leaf-recursion roundtrip)");

    eprintln!(
            "gate-air: leaf_recursion roundtrip OK (N={n_leaves}, n_levels={}, root trace 2^{}) [trusted verify OK]",
            out.n_levels, rv.trace_log_size
        );
}

/// End-to-end leaf-recursion correctness gate: ONE standalone leaf that IS the root (no level-0 level1-node,
/// no fold-node up-tree fold). Validates the leaf-topology WIRING — `derive_aggregate_config`,
/// `prove_gate_air_leaf`, and the leaf unpacker reconstructing + binding a single-leaf tree
/// (`prove_root_verification_leaves`'s final `verify_circuit` sanity check). The multi-leaf level1/fold-node
/// path is exercised by the env-gated heavy variant + proving-utils' restored `smoke_cairo_tree`.
///
/// RUN-GUARD (laptop-safety): env-gated to HEAVY_RECURSION so plain `cargo test` never
/// executes a real recursion prove/verify on a laptop. Run it on the CPU VM with the guard set.
#[test]
fn leaf_recursion_end_to_end() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!(
            "leaf_recursion_end_to_end: SKIPPED (recursion prove/verify). Set \
                 HEAVY_RECURSION=1 to run."
        );
        return;
    }
    // N=1: lone leaf is the root; no fold. Blowup 1 keeps the leaf + root-verify proofs minimal.
    leaf_recursion_roundtrip(1, 1, TopologyConfig::default().fold_arity);
}

/// HEAVY (box-only): the leaf-recursion roundtrip WITH the level-0 level1-node layer + fold-node up-tree fold. The fold-node floors
/// ~2^22 (several GB); OOMs a laptop, so env-gated to HEAVY_RECURSION=1. Plain `cargo test`
/// compiles + SKIPS it.
#[test]
fn leaf_recursion_end_to_end_with_fold() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!(
                "leaf_recursion_end_to_end_with_fold: SKIPPED (heavy: level1/fold-node nodes ~2^22). Set \
                 HEAVY_RECURSION=1 on the CPU VM to run the with-level1/fold-node roundtrip."
            );
        return;
    }
    // N=2: two leaves → one level-0 level1-node that IS the root (N <= k, no fold-node). Bump to N > k to
    // also exercise the fold-node up-tree fold once a box run confirms the level1-node layer.
    leaf_recursion_roundtrip(2, 3, TopologyConfig::default().fold_arity);
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
/// HEAVY (box-only): builds real level1 (and, at `n > k`, fold-node) multiverifier nodes (~2^22, GBs) so it
/// OOMs a laptop; env-gated to HEAVY_RECURSION. Plain `cargo test` compiles + SKIPS it.
fn leaf_recursion_streaming_equiv(n_leaves: usize, log_blowup_factor: u32, fold_arity: usize) {
    use circuits_stark_verifier::proof::Proof;
    use leaf::{build_recursion_precompute, prove_gate_air_leaf, GateAirLeafParams};
    use recursion_consts_tests::derive_aggregate_config;
    use recursive_aggregate::pools::PoolSet;
    use recursive_aggregate::prove::recursive_aggregate_prove_leaves;
    use recursive_aggregate::prove_streaming::recursive_aggregate_prove_leaves_streaming;
    use recursive_aggregate::{AggregateOutput, TreeProof};

    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let (proof0, params0, cfg, _canonical_base_pp_root) =
        prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let op = recursion_consts::OperatingPoint::K500N174; // placeholder key (tests recompute their own roots)
    let config = derive_aggregate_config(
        &cfg,
        &params0,
        fold_arity,
        log_blowup_factor,
        log_blowup_factor,
    );
    // test: build all arities (placeholder op.n() differs from the small fold N)
    let pre = build_recursion_precompute(&config, op, &cfg, &params0, true);

    let make_base = || -> (Proof<QM31>, GateAirLeafParams) { (proof0.clone(), params0.clone()) };
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
            "gate-air: leaf_recursion streaming-equiv OK (N={n_leaves}, bit-identical to sequential, scrambled arrival, n_pools 1+2)"
        );
}

/// (box-only) Scheduling-independence roundtrip over the required n ∈ {1, 2, k, k+1, ragged
/// r==1, ~2k+3}. Env-gated (heavy level1/fold-node proving). Proves streaming == sequential byte-for-byte.
#[test]
fn leaf_recursion_streaming_equiv_sweep() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!(
                "leaf_recursion_streaming_equiv_sweep: SKIPPED (heavy: real level1/fold-node proving ~2^22). Set \
                 HEAVY_RECURSION=1 on the CPU VM to run the out-of-order equivalence sweep."
            );
        return;
    }
    let k = TopologyConfig::default().fold_arity;
    // n = k+1 is the ragged r==1 case (splits into k-1 and 2); 2k+3 exercises multi-group + carry.
    for n in [1usize, 2, k, k + 1, 2 * k + 3] {
        leaf_recursion_streaming_equiv(n, 1, k);
    }
}

/// TERMINATION + PANIC PROPAGATION for the streaming coordinator: a `wrap` closure that panics
/// must make the coordinator re-panic on the parent (via `thread::scope` join) — no hang, no
/// silent drop — for BOTH n_pools == 1 and > 1. The panic fires INSIDE `wrap`, before any level1/fold
/// node proves, so the machinery under test is pure scheduling/termination.
///
/// HEAVY (box-only): `derive_aggregate_config` builds the ~2^22 level1/fold-node preprocessed shapes
/// (heavy REGARDLESS of leaf size — a node verifies `fold_arity` in-circuit STARK proofs), which
/// OOMs a laptop; env-gated to HEAVY_RECURSION. Plain `cargo test` compiles + SKIPS it.
#[test]
fn leaf_recursion_streaming_wrap_panic_propagates() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!(
            "leaf_recursion_streaming_wrap_panic_propagates: SKIPPED (heavy: config build ~2^22). \
                 Set HEAVY_RECURSION=1 on the CPU VM to run the panic-propagation test."
        );
        return;
    }
    use recursive_aggregate::pools::PoolSet;
    use recursive_aggregate::prove_streaming::recursive_aggregate_prove_leaves_streaming;
    use recursive_aggregate::TreeProof;

    // Build the smallest valid leaf-recursion config from a tiny base. No recursion PROVE runs (wrap
    // panics first), but the config build itself is the heavy part gated above.
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let (_p0, params0, cfg, _r) = prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let fold_arity = TopologyConfig::default().fold_arity;
    let op = recursion_consts::OperatingPoint::K500N174; // placeholder key (config recomputed for tiny base)
    let config = recursion_consts_tests::derive_aggregate_config(&cfg, &params0, fold_arity, 1, 1);
    let pre = leaf::build_recursion_precompute(&config, op, &cfg, &params0, true);

    for n_pools in [1usize, 2] {
        let config = &config;
        let pre = &pre;
        let n_leaves = 2usize; // one level1 group (n <= k); wrap panics before any node proves.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pools = PoolSet::new(n_pools, 2);
            let (tx, rx) = std::sync::mpsc::channel::<(usize, usize)>();
            for i in 0..n_leaves {
                tx.send((i, i)).unwrap();
            }
            drop(tx);
            // Every `wrap` panics — a worker panic must re-panic on the coordinator's
            // `thread::scope` join (not hang, not be silently dropped). The panic fires inside
            // `wrap`, before any level1/fold node runs, so no real proving happens (laptop-safe).
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
    eprintln!("gate-air: streaming wrap-panic propagation OK (re-panics, no hang; n_pools 1+2)");
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
        ProgramTableEval {
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
        BoundaryTableEval {
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
        RcTableEval {
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
/// Promotes the `gpu_tracegen::k1_byte_identity` harness (formerly reachable only via the removed
/// `GATE_AIR_GPU_TEST=k1` prove-path flag) to a real test.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[serial]
fn k1_trace_identity() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let rc_lo = build_rc_lo();
    gpu_tracegen::k1_byte_identity(&gates, &cases, k, &rc_lo, &rc_lo)
        .expect("GPU K1 main trace != CPU recompute");
}

/// (T1b) `k4_interaction_identity` — GPU K4 LogUp interaction == CPU recompute. Promotes the
/// `gpu_tracegen::k4_byte_identity` harness (formerly `GATE_AIR_GPU_TEST=k4`) to a real test.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[serial]
fn k4_interaction_identity() {
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let rc_lo = build_rc_lo();
    gpu_tracegen::k4_byte_identity(&gates, &cases, k, &rc_lo, &rc_lo)
        .expect("GPU K4 interaction != CPU recompute");
}

/// (T2) `base_precompute_identity` — a base shard proved with the shard-invariant precompute
/// (`Some(pc)`) is BYTE-IDENTICAL to one proved rebuild-per-shard (`None`). Replaces the removed
/// `GATE_AIR_NO_BASE_PRECOMPUTE` A/B arm + `GATE_AIR_BASE_PROOF_HASH` compare; extends
/// `tree0_precompute_matches_rebuild` (tree0 only) to the full base proof via
/// `diag::base_proof_fingerprint`.
#[cfg(all(feature = "cuda", feature = "diag"))]
#[test]
#[serial]
fn base_precompute_identity() {
    use base::{prove_base_shard, BaseProverPrecompute};
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!(
            "base_precompute_identity: SKIPPED (GPU base prove). Set HEAVY_RECURSION=1 on the box."
        );
        return;
    }
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let n_gates = gates.len();
    let topo = TopologyConfig::default();
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
    let with_pc = prove_base_shard(Some(&pc), &cases, &gates, k, n_gates, &topo, &rc_lo)
        .expect("prove_base_shard(Some)");
    let rebuild = prove_base_shard(None, &cases, &gates, k, n_gates, &topo, &rc_lo)
        .expect("prove_base_shard(None)");
    assert_eq!(
        diag::base_proof_fingerprint(std::slice::from_ref(&with_pc)),
        diag::base_proof_fingerprint(std::slice::from_ref(&rebuild)),
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
#[serial]
fn incircuit_self_verify() {
    use circuit_statement::{gate_air_components, GateAirStatement};
    use circuits::blake::HashValue;
    use circuits::context::{Context, TraceContext};
    use circuits::ivalue::NoValue;
    use circuits::ops::Guess;
    use circuits_stark_verifier::proof::empty_proof;
    use circuits_stark_verifier::verify::verify as circuit_verify;
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!("incircuit_self_verify: SKIPPED (in-circuit verify build). Set HEAVY_RECURSION=1 on the box.");
        return;
    }
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
#[serial]
fn gpu_vs_host_constraints_identity() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!("gpu_vs_host_constraints_identity: SKIPPED (GPU base prove). Set HEAVY_RECURSION=1 on the box.");
        return;
    }
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
#[serial]
fn full_proof_gpu_vs_simd_identity() {
    if std::env::var("HEAVY_RECURSION").is_err() {
        eprintln!("full_proof_gpu_vs_simd_identity: SKIPPED (GPU base prove). Set HEAVY_RECURSION=1 on the box.");
        return;
    }
    let (gates, cases, k) = nop_fixture(4, 2, 1);
    let (simd_proof, _p, _cfg, _r) = prove_tiny_base(&gates, &cases, k, TEST_RC_LOG);
    let (gpu_proof, _p2) = prove_tiny_base_on_gpu(&gates, &cases, k, TEST_RC_LOG);
    assert_eq!(
        format!("{:?}", simd_proof),
        format!("{:?}", gpu_proof),
        "full proof GPU (CudaBackend) != SimdBackend"
    );
}
