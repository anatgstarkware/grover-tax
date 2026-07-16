//! Milestone 3: turn the gate_air verification circuit into a foldable leaf and fold N of them.
//!
//! A leaf = the circuit that verifies the gate_air stwo proof (via `circuits_stark_verifier::verify` +
//! `GateAirStatement`), with its 2 reserved outputs set to a commitment, proved with the circuit
//! prover. That makes it a circuit-prover proof the multiverifier folds (`recursive_aggregate`).
//!
//! Output encoding (OPEN #3, Fork A): `H_i = blake( H_P ‖ x_limbs ‖ y_limbs )` per shot, where
//! `H_P = blake( program_table ‖ nonce )` is the hiding secret-circuit commitment. Both the program
//! (feeding H_P) and x/y are guessed witness AND bound to the base proof via `public_logup_sum` (the
//! TAG_PROGRAM_PUB and boundary public terms), so H_i commits to a genuine `x→y` execution of the
//! committed hidden circuit. The native verifier recomputes every `H_i` from the ONE published `H_P`,
//! enforcing same-program across shards for free.

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
    prepare_circuit_proof_for_circuit_verifier, prove_circuit_assignment,
    prove_circuit_with_precompute,
};
use std::sync::Arc;

use recursive_aggregate::{
    multiverifier_node_preprocessed, node_preprocessed_from_shared, preprocessed_root,
    shared_config_for_leaf, AggregateConfig, CircuitPrecompute, RecursionPrecompute, TreeProof,
};
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::mempool::BaseColumnPool;

use crate::circuit_statement::GateAirStatement;
use crate::N_LIMBS;

/// Public parameters of the gate_air proof the leaf verifies (everything `GateAirStatement::new`
/// needs). Identical for the NoValue shape pass and the real QM31 assignment.
#[derive(Clone)]
pub struct GateAirLeafParams {
    pub main_log_size: u32,
    pub program_log_size: u32,
    pub boundary_log_size: u32,
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

fn max_sizes(a: &ComponentSizes, b: &ComponentSizes) -> ComponentSizes {
    ComponentSizes {
        eq: a.eq.max(b.eq),
        qm31_ops: a.qm31_ops.max(b.qm31_ops),
        m31_to_u32: a.m31_to_u32.max(b.m31_to_u32),
        triple_xor: a.triple_xor.max(b.triple_xor),
        blake_g_gate: a.blake_g_gate.max(b.blake_g_gate),
    }
}

/// Verifies ONE gate_air base proof in-circuit and emits its output digest `H_i`. This is the body of
/// the standalone single-base leaf ([`build_gate_air_leaf_circuit`]) — the whole security-relevant
/// binding lives here.
///
/// Returns `h_i_vars`: the eight `H_i` digest words the leaf sets as its outputs.
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
        params.preprocessed_root.clone(),
        params.boundary.clone(),
        params.total_pc,
        params.program.clone(),
        params.nonce,
    );
    let proof_vars = proof.guess(context);
    verify(context, &proof_vars, cfg, &statement);

    // Leaf output (OPEN #3, Fork A): H_i = blake2s( H_P ‖ x ‖ y ), where
    //   H_P = blake2s( program_table ‖ nonce )   — the hiding program commitment.
    // BOTH the program Vars (feeding H_P) and the x/y limbs are GUESSED witness AND bound to the base
    // proof by `GateAirStatement::public_logup_sum`:
    //   - x/y via the boundary's public dangling term B (main carries x at ts=0, boundary re-emits y
    //     at TS_FINAL);
    //   - the program via the program table's public dangling term P_pub (TAG_PROGRAM_PUB), so the
    //     guessed (slot, op, t, a, b, mult) are forced equal to the base's committed, LogUp-bound
    //     program — H_P therefore commits to the EXECUTED secret circuit, not a free guess.
    // `verify` enforces `public_logup_sum + Σ claimed_sums == 0`, discharging both bindings. Keeping
    // everything witness (not `context.constant`) keeps `leaf_preprocessed_root` identical across shards.
    // #1425: outputs are now the full unreduced eight-word Blake2s digest (`N_RESERVED == 8`), so the
    // leaf emits `H_i = blake2s( H_P ‖ x ‖ y )` as eight words. `H_P` is itself the eight-word digest
    // from `compute_h_p`; its words (each a QM31 `(lo, hi, 0, 0)` message word) lead the preimage,
    // followed by the guessed x/y limb Vars, unchanged.
    let h_p = statement.compute_h_p(context);
    let mut preimage: Vec<_> = h_p.iter().map(|w| *w.get()).collect();
    for (x, y) in statement.boundary_vars() {
        preimage.extend(x.iter().chain(y.iter()).copied());
    }
    let output_hash: HashValue<_> = blake2s(context, &preimage, 16 * preimage.len());
    let h_i_vars: Vec<Var> = output_hash.iter().map(|w| *w.get()).collect();
    h_i_vars
}

// =================================================================================================
// LEAF/R1/R2 topology — a standalone single-base LEAF (one gate_air base per leaf), a level-1
// leaf-verifying (R1) node layer, and the shared R2 up-tree fold.
// =================================================================================================

/// Builds the gate_air single-base LEAF circuit: verify ONE gate_air base proof in-circuit (via
/// [`emit_one_base`]) and set the reserved outputs to its `H_i`. The multiverifier's level-0 R1 nodes
/// verify these leaves. Generic over `Value` (NoValue shape / QM31 assignment).
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

/// Derives the leaf/R1/R2 `AggregateConfig` for gate_air leaves of this proof's shape.
/// Resolves the leaf↔node padding decoupling: the leaf pads to its OWN
/// natural target (~2^20), while the two full-`fold_arity` node variants (R1 verifies leaves, R2
/// verifies nodes) pad to a COMMON `node_target` fixed point. Populates the config's LeafR1R2 extras
/// (`leaf_shared_config`, R1, `leaf_preprocessed_root`, leaf target/PCS, leaf/level1 precomputes).
/// The preprocessed circuits + their (pcs, root) for the leaf/R1/R2 shapes, carried out of
/// [`derive_aggregate_config`] so [`build_recursion_precompute`] can build the heavy
/// [`CircuitPrecompute`]s WITHOUT recomputing the (expensive) node fixed-point loop.
pub struct AggregateShapes {
    pub leaf_pp: PreprocessedCircuit,
    pub pcs: PcsConfig,
    pub leaf_root: HashValue<QM31>,
    pub level1_pp: PreprocessedCircuit,
    pub node_pcs: PcsConfig,
    pub level1_root: HashValue<QM31>,
    pub node_pp: PreprocessedCircuit,
    pub node_root: HashValue<QM31>,
}

pub fn derive_aggregate_config(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    fold_arity: usize,
    // Node (R1/R2/root) FRI blowup.
    log_blowup_factor: u32,
    // Leaf-wrap FRI blowup, decoupled from the node blowup (equal to it unless `LEAF_BLOWUP` is set).
    // Only the leaf shape's own PCS + preprocessed root take this; R1 verifies leaves at this blowup
    // automatically because its child config is `shared_config_for_leaf(&leaf_pp, pcs)`.
    leaf_log_blowup: u32,
) -> (AggregateConfig, AggregateShapes) {
    assert!(fold_arity >= 2, "fold_arity k must be >= 2");
    let shape = || build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params);
    let leaf_sizes = compute_padded_sizes(&shape());

    // LEAF↔NODE PADDING DECOUPLING. Pad the leaf to its OWN target (natural ~2^20), NOT max(leaf,node),
    // so `t_leaf` is pinned independent of `fold_arity`.
    let leaf_target = leaf_sizes.clone();
    let (leaf_pp, pcs) = {
        let mut leaf_ctx = shape();
        pad_to_targets(&mut leaf_ctx, leaf_target.clone());
        let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
        let pcs = leaf_pcs_config(leaf_pp.trace_log_size, leaf_log_blowup);
        (leaf_pp, pcs)
    };
    let leaf_preprocessed_root = preprocessed_root(&leaf_pp, leaf_log_blowup);
    let leaf_shared_config = shared_config_for_leaf(&leaf_pp, pcs);

    // The two node variants both pad to a COMMON `node_target` (self-verification fixed point):
    //   - level-1 node (R1): verifies `fold_arity` LEAVES → sized with the leaf `pcs`.
    //   - level-≥2 node (R2): verifies `fold_arity` NODES → sized with a NODE pcs (node blowup +
    //     the node's OWN trace_log = `level1_pp.trace_log_size`, which is padded to `node_target`).
    // Sizing R2's child as a node (not with the leaf `pcs`) makes the fixed point exact for ANY
    // leaf/node blowup pair. This only sets witness padding; the pinned pp-roots are independent of
    // blowup/trace_size, so it does not touch the trust anchors.
    let (_, node1_seed_sizes) = multiverifier_node_preprocessed(&leaf_pp, pcs, None, fold_arity);
    let mut node_target = node1_seed_sizes;
    let (level1_pp, node_pp) = loop {
        let (level1_pp, level1_unpadded) =
            multiverifier_node_preprocessed(&leaf_pp, pcs, Some(node_target.clone()), fold_arity);
        // R2 verifies NODES: size its child with the node blowup + the node's own trace_log.
        let node_child_pcs = leaf_pcs_config(level1_pp.trace_log_size, log_blowup_factor);
        let (node_pp, node2_unpadded) = multiverifier_node_preprocessed(
            &level1_pp,
            node_child_pcs,
            Some(node_target.clone()),
            fold_arity,
        );
        let new_target = max_sizes(&level1_unpadded, &node2_unpadded);
        if new_target == node_target {
            break (level1_pp, node_pp);
        }
        node_target = new_target;
    };
    let level1_preprocessed_root = preprocessed_root(&level1_pp, log_blowup_factor);

    // LEAF↔NODE PCS DECOUPLING. A node proof's Merkle auth-path height (node trace ~2^22 + blowup ~25)
    // is larger than a leaf's (~24), so describe a NODE's own proof (its prove + the R2 child-verify +
    // the root verify) with a separate node PCS from the node trace size.
    let node_pcs = leaf_pcs_config(node_pp.trace_log_size, log_blowup_factor);
    let node_shared_config = shared_config_for_leaf(&level1_pp, node_pcs);
    // Rebuild the R2 (node-verifying) node shape with the NODE pcs for its children (matches
    // `build_node_context` at prove time).
    let node_pp =
        node_preprocessed_from_shared(&node_shared_config, node_target.clone(), fold_arity);
    let node_preprocessed_root = preprocessed_root(&node_pp, log_blowup_factor);

    let roots_collapse = level1_preprocessed_root == node_preprocessed_root;
    eprintln!(
        "gate-air: leaf↔node decoupling: leaf trace 2^{} (pcs lifting {:?}), node trace 2^{} (pcs lifting {:?}); R1{}R2 (collapse={})",
        leaf_pp.trace_log_size,
        pcs.lifting_log_size,
        node_pp.trace_log_size,
        node_pcs.lifting_log_size,
        if roots_collapse { "==" } else { "!=" },
        roots_collapse,
    );

    // The recursion precomputes are built up front by `build_recursion_precompute` from the shapes
    // returned below — NOT here, so the fixed-point loop is not on the precompute-build path.
    let shapes = AggregateShapes {
        leaf_pp,
        pcs,
        leaf_root: leaf_preprocessed_root.clone(),
        level1_pp,
        node_pcs,
        level1_root: level1_preprocessed_root.clone(),
        node_pp,
        node_root: node_preprocessed_root.clone(),
    };

    let agg = AggregateConfig {
        // Shared / R2 (also used by the shared up-tree fold).
        node_shared_config,
        node_preprocessed_root,
        node_target_padding_sizes: node_target,
        node_pcs_config: node_pcs,
        fold_arity,
        // LeafR1R2 tier.
        leaf_shared_config: Some(leaf_shared_config),
        level1_preprocessed_root: Some(level1_preprocessed_root),
        leaf_preprocessed_root: Some(leaf_preprocessed_root),
        leaf_target_padding_sizes: Some(leaf_target),
        leaf_pcs_config: Some(pcs),
    };
    (agg, shapes)
}

/// Builds the leaf/R1/R2 [`RecursionPrecompute`] from the shapes carried out of
/// [`derive_aggregate_config`] (so the node fixed-point loop is NOT recomputed). Honors
/// `GATE_AIR_NO_PRECOMPUTE=1` -> all `None` (rebuild-per-prove).
pub fn build_recursion_precompute(shapes: AggregateShapes) -> RecursionPrecompute {
    let AggregateShapes {
        leaf_pp,
        pcs,
        leaf_root,
        level1_pp,
        node_pcs,
        level1_root,
        node_pp,
        node_root,
    } = shapes;
    let no_precompute = std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok();
    if no_precompute {
        return RecursionPrecompute {
            node_precompute: None,
            level1_precompute: None,
            leaf_precompute: None,
        };
    }
    RecursionPrecompute {
        leaf_precompute: Some(Arc::new(CircuitPrecompute::new(leaf_pp, pcs, leaf_root))),
        level1_precompute: Some(Arc::new(CircuitPrecompute::new(
            level1_pp,
            node_pcs,
            level1_root,
        ))),
        node_precompute: Some(Arc::new(CircuitPrecompute::new(
            node_pp, node_pcs, node_root,
        ))),
    }
}

/// Builds + proves one gate_air single-base LEAF ([`FoldMode::LeafR1R2`]), padded to its OWN target so
/// `t_leaf` stays pinned independent of `fold_arity`. `real_proof` is the gate_air proof as circuit
/// values. Requires a `LeafR1R2` config.
pub fn prove_gate_air_leaf(
    real_proof: Proof<QM31>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    config: &AggregateConfig,
    pre: &RecursionPrecompute,
) -> TreeProof {
    let leaf_target = config.leaf_target_padding_sizes.clone().expect(
        "prove_gate_air_leaf requires a LeafR1R2 config (leaf_target_padding_sizes present)",
    );
    let leaf_pcs = config
        .leaf_pcs_config
        .expect("prove_gate_air_leaf requires a LeafR1R2 config (leaf_pcs_config present)");
    let leaf_preprocessed_root = config
        .leaf_preprocessed_root
        .clone()
        .expect("prove_gate_air_leaf requires a LeafR1R2 config (leaf_preprocessed_root present)");
    let mut context = build_gate_air_leaf_circuit::<QM31>(real_proof, cfg, params);
    pad_to_targets(&mut context, leaf_target);
    let circuit_proof = match &pre.leaf_precompute {
        Some(pc) => prove_circuit_with_precompute::<Blake2sM31MerkleChannel>(
            &pc.base_column_pool,
            &pc.twiddles,
            &pc.preprocessed,
            MaybeOwned::Borrowed(&pc.tree),
            context.values(),
            pc.pcs_config,
        ),
        None => {
            let preprocessed = PreprocessedCircuit::preprocess_circuit(&mut context);
            prove_circuit_assignment(
                context.values(),
                &preprocessed,
                &BaseColumnPool::<SimdBackend>::new(),
                leaf_pcs,
            )
        }
    }
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
