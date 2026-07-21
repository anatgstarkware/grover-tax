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
    prepare_circuit_proof_for_circuit_verifier, prove_circuit_with_precompute,
};
use std::collections::BTreeMap;

use recursive_aggregate::precomputes::{RecursionPrecompute, TreeSpec};
use recursive_aggregate::{
    multiverifier_node_preprocessed, node_preprocessed_from_shared, shared_config_for_leaf,
    AggregateConfig, TreeProof,
};
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;

use crate::recursion_consts::OperatingPoint;

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
        params.rc_log,
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
// Leaf topology — a standalone single-base LEAF (one gate_air base per leaf), a level-1
// leaf-verifying node layer, and the shared fold-node up-tree fold.
// =================================================================================================

/// Builds the gate_air single-base LEAF circuit: verify ONE gate_air base proof in-circuit (via
/// [`emit_one_base`]) and set the reserved outputs to its `H_i`. The multiverifier's level-0
/// level1-nodes verify these leaves. Generic over `Value` (NoValue shape / QM31 assignment).
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

/// The witness-independent preprocessed SHAPES the recursion needs, carried out of
/// [`derive_aggregate_config`] so [`build_recursion_precompute`] can commit the per-shape trees WITHOUT
/// rerunning the (expensive) node fixed-point loop. Roots are NOT recomputed here — they are the pinned
/// consts of `op` (each tree asserts against them at commit time). Per-arity `level1`/`fold` shapes are
/// keyed by child count `2..=fold_arity`.
pub struct AggregateShapes {
    /// Operating point whose pinned root table the trees assert against.
    pub op: OperatingPoint,
    pub leaf_pp: PreprocessedCircuit,
    /// Leaf-wrap PCS (leaf lifting).
    pub leaf_pcs: PcsConfig,
    /// Node PCS (node lifting) — used to commit every level1/fold node tree.
    pub node_pcs: PcsConfig,
    pub level1_pp: BTreeMap<usize, PreprocessedCircuit>,
    pub fold_pp: BTreeMap<usize, PreprocessedCircuit>,
}

/// Assembles the gate_air `AggregateConfig` + its per-arity root table for the pinned operating point
/// `op` (roots read from the const table — NOT recomputed) and builds the witness-independent
/// preprocessed shapes for the node arities the trees need. Resolves the leaf↔node padding decoupling:
/// the leaf pads to its OWN natural target (~2^20), while every node variant (level1 verifies leaves,
/// fold verifies nodes) pads to a COMMON `node_target` fixed point.
///
/// `build_all_shapes`: production passes `false` → build shapes ONLY for the arities THIS point's fold
/// actually uses ([`recursive_aggregate::fold_used_arities`] of `op.n()`), since committing every
/// unused `2..=k` arity's ~2^22 tree overruns the base-precompute overlap window. Tests pass `true` →
/// build every `2..=k` arity (the drift test checks all pinned consts; roundtrip tests fold a small N
/// that differs from the placeholder `op.n()`, so they need the full set).
pub fn derive_aggregate_config(
    op: OperatingPoint,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    fold_arity: usize,
    // Node (level1-/fold-node/root) FRI blowup.
    log_blowup_factor: u32,
    // Leaf-wrap FRI blowup, decoupled from the node blowup (equal to it unless `LEAF_BLOWUP` is set).
    // Only the leaf shape's own PCS takes this; the level1-node verifies leaves at this blowup
    // automatically because its child config is `shared_config_for_leaf(&leaf_pp, leaf_pcs)`.
    leaf_log_blowup: u32,
    build_all_shapes: bool,
) -> (AggregateConfig, AggregateShapes) {
    assert!(fold_arity >= 2, "fold_arity k must be >= 2");
    let shape = || build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params);
    let leaf_sizes = compute_padded_sizes(&shape());

    // LEAF↔NODE PADDING DECOUPLING. Pad the leaf to its OWN target (natural ~2^20), NOT max(leaf,node),
    // so `t_leaf` is pinned independent of `fold_arity`.
    let leaf_target = leaf_sizes.clone();
    let (leaf_pp, leaf_pcs) = {
        let mut leaf_ctx = shape();
        pad_to_targets(&mut leaf_ctx, leaf_target.clone());
        let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
        let leaf_pcs = leaf_pcs_config(leaf_pp.trace_log_size, leaf_log_blowup);
        (leaf_pp, leaf_pcs)
    };
    let leaf_shared_config = shared_config_for_leaf(&leaf_pp, leaf_pcs);

    // Node-size fixed point over the two full-`fold_arity` variants (level1 verifies leaves, fold
    // verifies nodes), both padded to a COMMON `node_target`. Sizing the fold-node's child as a node
    // (node blowup + the node's own trace_log) makes the fixed point exact for any leaf/node blowup
    // pair. This only sets witness padding; the pinned roots are independent of blowup/trace_size.
    let (_, node1_seed_sizes) =
        multiverifier_node_preprocessed(&leaf_pp, leaf_pcs, None, fold_arity);
    let mut node_target = node1_seed_sizes;
    let level1_k_pp = loop {
        let (level1_pp, level1_unpadded) = multiverifier_node_preprocessed(
            &leaf_pp,
            leaf_pcs,
            Some(node_target.clone()),
            fold_arity,
        );
        let node_child_pcs = leaf_pcs_config(level1_pp.trace_log_size, log_blowup_factor);
        let (_node_pp, node2_unpadded) = multiverifier_node_preprocessed(
            &level1_pp,
            node_child_pcs,
            Some(node_target.clone()),
            fold_arity,
        );
        let new_target = max_sizes(&level1_unpadded, &node2_unpadded);
        if new_target == node_target {
            break level1_pp;
        }
        node_target = new_target;
    };

    // LEAF↔NODE PCS DECOUPLING. A node proof's Merkle auth-path height (node trace ~2^22 + blowup ~25)
    // is larger than a leaf's (~24), so a NODE's own proof (its prove + the fold-node child-verify +
    // the root verify) uses a separate node PCS from the node trace size.
    let node_child_trace_log = level1_k_pp.trace_log_size;
    let node_pcs = leaf_pcs_config(node_child_trace_log, log_blowup_factor);
    let node_shared_config = shared_config_for_leaf(&level1_k_pp, node_pcs);

    // Per-arity node shapes: level1 verifies LEAVES (leaf shared config), fold verifies NODES (node
    // shared config); both pad to the common `node_target`. Production builds only the arities this
    // point's fold uses (the rest would be dead ~2^22 commits on the critical path); tests build all.
    let (level1_arities, fold_arities): (Vec<usize>, Vec<usize>) = if build_all_shapes {
        ((2..=fold_arity).collect(), (2..=fold_arity).collect())
    } else {
        let (l, f) = recursive_aggregate::fold_used_arities(op.n(), fold_arity);
        (l.into_iter().collect(), f.into_iter().collect())
    };
    let mut level1_pp: BTreeMap<usize, PreprocessedCircuit> = BTreeMap::new();
    let mut fold_pp: BTreeMap<usize, PreprocessedCircuit> = BTreeMap::new();
    for arity in level1_arities {
        level1_pp.insert(
            arity,
            node_preprocessed_from_shared(&leaf_shared_config, node_target.clone(), arity),
        );
    }
    for arity in fold_arities {
        fold_pp.insert(
            arity,
            node_preprocessed_from_shared(&node_shared_config, node_target.clone(), arity),
        );
    }

    // Per-arity pinned root tables (from the operating-point consts — NOT recomputed).
    let level1_roots: BTreeMap<usize, HashValue<QM31>> =
        (2..=fold_arity).map(|a| (a, op.level1_root(a))).collect();
    let fold_roots: BTreeMap<usize, HashValue<QM31>> =
        (2..=fold_arity).map(|a| (a, op.fold_root(a))).collect();

    eprintln!(
        "gate-air: leaf↔node decoupling: leaf trace 2^{} (pcs lifting {:?}), node trace 2^{} (pcs lifting {:?})",
        leaf_pp.trace_log_size,
        leaf_pcs.lifting_log_size,
        node_child_trace_log,
        node_pcs.lifting_log_size,
    );

    let shapes = AggregateShapes {
        op,
        leaf_pp,
        leaf_pcs,
        node_pcs,
        level1_pp,
        fold_pp,
    };

    let agg = AggregateConfig {
        // Shared / fold-node tier (also used by the shared up-tree fold).
        fold_shared_config: node_shared_config,
        node_target_padding_sizes: node_target,
        node_pcs_config: node_pcs,
        fold_arity,
        // Leaf / level1 tier.
        leaf_shared_config,
        level1_roots,
        fold_roots,
        leaf_preprocessed_root: op.leaf_root(),
        leaf_target_padding_sizes: leaf_target,
        leaf_pcs_config: leaf_pcs,
    };
    (agg, shapes)
}

/// Builds the flat leaf/level1/fold [`RecursionPrecompute`] from the shapes carried out of
/// [`derive_aggregate_config`] (so the node fixed-point loop is NOT recomputed). Every tree asserts its
/// committed root equals the pinned const of `shapes.op` (the load-bearing soundness gate). Precompute
/// is UNCONDITIONAL in production.
pub fn build_recursion_precompute(shapes: AggregateShapes) -> RecursionPrecompute {
    let AggregateShapes {
        op,
        leaf_pp,
        leaf_pcs,
        node_pcs,
        level1_pp,
        fold_pp,
    } = shapes;
    // Commit a tree for every shape present. `derive_aggregate_config` already restricts the shapes to
    // the arities this point's fold uses (production), so this builds exactly the used trees; a
    // build-all-shapes caller (tests) builds all, which is safe (extra trees are just never proved).
    let leaf = TreeSpec {
        preprocessed: leaf_pp,
        pcs_config: leaf_pcs,
        expected_root: op.leaf_root(),
    };
    let level1 = level1_pp
        .into_iter()
        .map(|(arity, pp)| {
            (
                arity,
                TreeSpec {
                    preprocessed: pp,
                    pcs_config: node_pcs,
                    expected_root: op.level1_root(arity),
                },
            )
        })
        .collect();
    let fold = fold_pp
        .into_iter()
        .map(|(arity, pp)| {
            (
                arity,
                TreeSpec {
                    preprocessed: pp,
                    pcs_config: node_pcs,
                    expected_root: op.fold_root(arity),
                },
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
