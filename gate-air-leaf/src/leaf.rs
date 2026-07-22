//! Turn the gate_air verification circuit into a foldable leaf and fold N of them. A leaf verifies
//! one gate_air stwo proof (`circuits_stark_verifier::verify` + `GateAirStatement`) and sets its
//! reserved outputs to `H_i`, making it a circuit-prover proof the multiverifier folds.
//!
//! Output: `H_i = blake( H_P ‖ x_limbs ‖ y_limbs )`, where `H_P = blake( program_table ‖ nonce )` is
//! the hiding secret-circuit commitment. Program and x/y are guessed witness AND bound to the base
//! proof via `public_logup_sum` (TAG_PROGRAM_PUB + boundary public terms), so `H_i` commits to a
//! genuine `x→y` execution of the committed hidden circuit; the verifier recomputes every `H_i` from
//! the one published `H_P`, enforcing same-program across shards.

use circuits::blake::{blake2s, HashValue};
use circuits::context::{Context, FinalizedContext, Var};
use circuits::ivalue::{IValue, NoValue};
use circuits::ops::Guess;
use circuits_stark_verifier::proof::{empty_proof, Proof, ProofConfig};
use circuits_stark_verifier::verify::verify;

use circuit_common::finalize::{compute_padded_sizes, pad_to_targets, ComponentSizes};
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_common::N_RESERVED;
use circuit_prover::prover::{
    prepare_circuit_proof_for_circuit_verifier, prove_circuit_with_precompute,
};

use circuit_multiverifier::verify::SharedConfig;
use recursive_aggregate::pinned_configs::assemble_aggregate_config;
use recursive_aggregate::precomputes::{
    fold_used_arities, node_preprocessed_from_shared, RecursionPrecompute, TreeSpec,
};
use recursive_aggregate::{AggregateConfig, TreeProof};
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;

use crate::recursion_consts::OperatingPoint;
use crate::topology::{TopologyConfig, FOLD_ARITY, RECURSION_LOG_BLOWUP};

use crate::circuit_statement::GateAirStatement;
use crate::N_LIMBS;

/// Public parameters of the gate_air proof the leaf verifies (everything `GateAirStatement::new`
/// needs). Identical for the NoValue shape pass and the real QM31 assignment.
#[derive(Clone)]
pub struct GateAirLeafParams {
    pub main_log_size: u32,
    pub program_log_size: u32,
    pub boundary_log_size: u32,
    /// The rc supply-table log-size `R` this base was proved with — a TRUSTED construction value the
    /// leaf statement must reuse (production: `RC_LOG`; tests: the test's chosen value). The base
    /// prover's `R` and this MUST be equal (see `GateAirStatement::new`); NEVER read from the proof.
    pub rc_log: u32,
    pub preprocessed_root: HashValue<QM31>,
    pub boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    pub total_pc: u32,
    /// H_P program commitment (OPEN #3, Fork A). The committed program table this leaf's base proof
    /// ran: one entry per padded slot `(slot, opcode_scalar, target, ctrl_a, ctrl_b, multiplicity)`.
    /// The leaf GUESSES these, BINDS them to the base's committed program via the TAG_PROGRAM_PUB
    /// public term (`public_logup_sum`), and hashes them into `H_P`.
    pub program: ProgramRows,
    /// One shared hiding nonce (2 M31 words), IDENTICAL across all leaves of a run, folded into
    /// `H_P = blake(program ‖ nonce)`. Binding-inert (only blinds the program preimage); its
    /// consistency across leaves is required so every leaf yields the SAME H_P for the same program.
    pub nonce: [u32; 2],
}

/// Program-table rows the leaf hashes into H_P and binds to the base. One entry per padded slot,
/// mirroring `main.rs::ProgramTable`.
#[derive(Clone)]
pub struct ProgramRows {
    pub slot: Vec<u32>,
    pub opcode_scalar: Vec<u32>,
    pub target: Vec<u32>,
    pub ctrl_a: Vec<u32>,
    pub ctrl_b: Vec<u32>,
    /// per-slot multiplicity = samples*k (real) / 0 (padding). PINNED (not free) in the leaf.
    pub multiplicity: Vec<u32>,
}

/// PCS config for the leaf circuit's own (outer) proof. Mirrors stwo-circuits' `get_pcs_config`.
pub fn leaf_pcs_config(trace_log_size: u32, log_blowup_factor: u32) -> PcsConfig {
    let (pow_bits, n_queries) = match log_blowup_factor {
        1 => (26, 70),
        2 => (26, 35),
        3 => (27, 23),
        _ => panic!("unsupported log blowup factor"),
    };
    PcsConfig {
        pow_bits,
        fri_config: FriConfig {
            log_blowup_factor,
            log_last_layer_degree_bound: 0,
            n_queries,
            fold_step: 4,
        },
        lifting_log_size: Some(trace_log_size + log_blowup_factor),
    }
}

/// Verifies ONE gate_air base proof in-circuit and returns the eight `H_i` digest words the leaf
/// sets as outputs. The whole security-relevant binding lives here.
pub fn emit_one_base<Value: IValue>(
    context: &mut Context<Value>,
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> Vec<Var> {
    let statement = GateAirStatement::<Value>::new(
        context,
        params.main_log_size,
        params.program_log_size,
        params.boundary_log_size,
        params.rc_log,
        params.preprocessed_root.clone(),
        params.boundary.clone(),
        params.total_pc,
        params.program.clone(),
        params.nonce,
    );
    let proof_vars = proof.guess(context);
    verify(context, &proof_vars, cfg, &statement);

    // H_i = blake2s( H_P ‖ x ‖ y ) as eight words (#1425 full unreduced digest). H_P (eight words
    // from `compute_h_p`) leads the preimage, followed by the guessed x/y limb Vars. `public_logup_sum`
    // binds both the program (feeding H_P) and x/y to the base proof, so H_i commits to the executed
    // secret circuit. Everything stays witness (not constant) so `leaf_preprocessed_root` is shard-invariant.
    let h_p = statement.compute_h_p(context);
    let mut preimage: Vec<_> = h_p.iter().map(|w| *w.get()).collect();
    for (x, y) in statement.boundary_vars() {
        preimage.extend(x.iter().chain(y.iter()).copied());
    }
    let output_hash: HashValue<_> = blake2s(context, &preimage, 16 * preimage.len());
    let h_i_vars: Vec<Var> = output_hash.iter().map(|w| *w.get()).collect();
    h_i_vars
}

/// Builds the gate_air single-base LEAF circuit: verify one base proof (via [`emit_one_base`]) and
/// set the reserved outputs to its `H_i`. Generic over `Value` (NoValue shape / QM31 assignment).
pub fn build_gate_air_leaf_circuit<Value: IValue>(
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> FinalizedContext<Value> {
    let mut context = Context::new(N_RESERVED);
    let h_i_vars = emit_one_base(&mut context, proof, cfg, params);
    context.set_outputs(&h_i_vars);
    context.finalize(false)
}

/// Assembles the gate_air `AggregateConfig` for pinned operating point `op` entirely from the pinned
/// verifier consts (no derivation), via the same `assemble_aggregate_config` the fresh cascade uses,
/// so a pinned config is byte-identical to a derived one. The PRODUCTION config source. The pinned
/// points fix `fold_arity == FOLD_ARITY` and the default node/leaf blowup (asserted); `cfg`/`params`
/// only compute the leaf's own natural padding target.
pub fn pinned_aggregate_config(
    op: OperatingPoint,
    topo: &TopologyConfig,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> AggregateConfig {
    assert_eq!(
        topo.fold_arity, FOLD_ARITY,
        "pinned operating points fix fold_arity = {FOLD_ARITY}"
    );
    assert_eq!(
        (topo.recursion_log_blowup, topo.leaf_log_blowup),
        (RECURSION_LOG_BLOWUP, RECURSION_LOG_BLOWUP),
        "pinned operating points fix the default node/leaf blowup"
    );
    let derived = op
        .pinned()
        .to_derived(topo.leaf_log_blowup, topo.recursion_log_blowup);
    assemble_aggregate_config(&derived, leaf_target_sizes(cfg, params), FOLD_ARITY)
}

/// Builds the flat leaf/level1/fold [`RecursionPrecompute`] by rebuilding each layer's preprocessed
/// shape from `config` — no fixed-point loop. Every tree asserts its committed root equals `config`'s
/// root (the load-bearing soundness check: a drifted pinned config fails it loudly).
///
/// `build_all_arities`: production passes `false` → build shapes only for the arities this point's
/// fold uses ([`fold_used_arities`] of `op.n()`), since committing every unused `2..=k` arity's ~2^22
/// tree overruns the base-precompute overlap window. Tests pass `true` (they fold a small N differing
/// from the placeholder `op.n()`, so they need every `2..=k` arity).
pub fn build_recursion_precompute(
    config: &AggregateConfig,
    op: OperatingPoint,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    build_all_arities: bool,
) -> RecursionPrecompute {
    let k = config.fold_arity;
    let node_pcs = config.node_pcs_config;
    let node_target = &config.node_target_padding_sizes;

    // Leaf tree: rebuild the leaf preprocessed circuit padded to the config's leaf target.
    let leaf_pp = {
        let mut leaf_ctx = build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params);
        pad_to_targets(&mut leaf_ctx, config.leaf_target_padding_sizes.clone());
        PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx)
    };
    let leaf = TreeSpec {
        preprocessed: leaf_pp,
        pcs_config: config.leaf_pcs_config,
        expected_root: config.leaf_preprocessed_root.clone(),
    };

    // Node shapes: level1 verifies leaves (`leaf_shared_config`), fold verifies nodes
    // (`fold_shared_config`); both pad to the common `node_target`, rebuilt (not re-derived).
    let (level1_arities, fold_arities): (Vec<usize>, Vec<usize>) = if build_all_arities {
        ((2..=k).collect(), (2..=k).collect())
    } else {
        let (l, f) = fold_used_arities(op.n(), k);
        (l.into_iter().collect(), f.into_iter().collect())
    };
    let level1 = level1_arities
        .into_iter()
        .map(|a| {
            (
                a,
                node_spec(
                    &config.leaf_shared_config,
                    node_pcs,
                    node_target,
                    a,
                    config.level1_root(a),
                ),
            )
        })
        .collect();
    let fold = fold_arities
        .into_iter()
        .map(|a| {
            (
                a,
                node_spec(
                    &config.fold_shared_config,
                    node_pcs,
                    node_target,
                    a,
                    config.fold_root(a),
                ),
            )
        })
        .collect();
    RecursionPrecompute::new(leaf, level1, fold)
}

/// Builds + proves one gate_air single-base LEAF, padded to its OWN target so `t_leaf` stays pinned
/// independent of `fold_arity`. `real_proof` is the gate_air proof as circuit values.
pub fn prove_gate_air_leaf(
    real_proof: Proof<QM31>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    config: &AggregateConfig,
    pre: &RecursionPrecompute,
) -> TreeProof {
    let leaf_target = config.leaf_target_padding_sizes.clone();
    let leaf_preprocessed_root = config.leaf_preprocessed_root.clone();
    let mut context = build_gate_air_leaf_circuit::<QM31>(real_proof, cfg, params);
    pad_to_targets(&mut context, leaf_target);
    let leaf_tree = &pre.leaf;
    let circuit_proof = prove_circuit_with_precompute::<Blake2sM31MerkleChannel>(
        &pre.base_column_pool,
        &pre.twiddles,
        &leaf_tree.preprocessed,
        MaybeOwned::Borrowed(&leaf_tree.tree),
        context.values(),
        leaf_tree.pcs_config,
    )
    .expect("gate_air leaf prove failed");
    let (proof, public_data) = prepare_circuit_proof_for_circuit_verifier(circuit_proof);
    let output_values = public_data
        .output_values
        .try_into()
        .expect("leaf emits N_RESERVED outputs");

    TreeProof {
        proof,
        preprocessed_root: leaf_preprocessed_root,
        output_values,
    }
}

/// The leaf's OWN natural padding target, decoupled from the node target so `t_leaf` is pinned
/// independent of `fold_arity`. The pinned leaf root is the soundness anchor; this padding only sets
/// the prover's own leaf trace shape (a drift fails the leaf root assert).
fn leaf_target_sizes(cfg: &ProofConfig, params: &GateAirLeafParams) -> ComponentSizes {
    compute_padded_sizes(&build_gate_air_leaf_circuit::<NoValue>(
        empty_proof(cfg),
        cfg,
        params,
    ))
}

/// One node layer's [`TreeSpec`] for `arity`: the node preprocessed circuit verifying
/// `child_shared`-configured children (padded to `node_target`), committed at `node_pcs`, asserting
/// `expected_root`. Shared by the level1 (leaf child) and fold (node child) tiers.
fn node_spec(
    child_shared: &SharedConfig,
    node_pcs: PcsConfig,
    node_target: &ComponentSizes,
    arity: usize,
    expected_root: HashValue<QM31>,
) -> TreeSpec {
    TreeSpec {
        preprocessed: node_preprocessed_from_shared(child_shared, node_target.clone(), arity),
        pcs_config: node_pcs,
        expected_root,
    }
}
