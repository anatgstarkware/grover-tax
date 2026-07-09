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

use circuits::blake::{HashValue, blake2s, unpack_qm31s_to_u32_words};
use circuits::context::{Context, FinalizedContext, Var};
use circuits::ivalue::{IValue, NoValue};
use circuits::ops::Guess;
use circuits::wrappers::U32Wrapper;
use circuits_stark_verifier::proof::{Proof, ProofConfig, empty_proof};
use circuits_stark_verifier::verify::verify;

use circuit_common::N_RESERVED;
use circuit_common::finalize::{ComponentSizes, compute_padded_sizes, pad_to_targets};
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_prover::prover::{
    prepare_circuit_proof_for_circuit_verifier, prove_circuit_assignment,
    prove_circuit_with_precompute,
};
use std::sync::Arc;

use circuit_multiverifier::verify::{ChildVerifier, build_fanning_circuit};
use recursive_aggregate::{
    AggregateConfig, BaseOutput, CircuitPrecompute, TreeProof, multiverifier_node_preprocessed,
    node_preprocessed_from_shared, preprocessed_root, shared_config_for_leaf,
};
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::mempool::BaseColumnPool;

use crate::N_LIMBS;
use crate::circuit_statement::GateAirStatement;

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

/// Verifies ONE gate_air base proof in-circuit and emits its output digest `H_i`. This is the
/// per-child body of the base-fanning node ([`build_gate_air_base_node_circuit`]) — the whole
/// security-relevant binding lives here, so folding `b` bases is exactly `b` independent copies of
/// this call (each its own statement / verify / channel). (Pre-base-fanning this was also the body of
/// the standalone single-base leaf; base-fanning subsumes that — a b=1 base-node IS a single-base
/// leaf.)
///
/// Returns `(preprocessed_root_vars, h_i_vars)`: the eight guessed base-proof preprocessed-root words
/// and the eight `H_i` digest words. The single-base leaf sets `h_i_vars` as its outputs; the
/// base-node folds `[preprocessed_root_vars, unpack(h_i_vars)]` into the shared node hash.
pub fn emit_one_base<Value: IValue>(
    context: &mut Context<Value>,
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> (HashValue<Var>, Vec<Var>) {
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
    (statement.preprocessed_root_vars().clone(), h_i_vars)
}

/// Per-base input to a base-fanning node: one gate_air base proof + everything `emit_one_base` needs
/// to verify it (its own [`ProofConfig`] and [`GateAirLeafParams`]). Each base carries its own params
/// (its own preprocessed root, (x,y) boundary, program, nonce), so the `b` children are independent.
pub struct GateAirBaseInput<Value: IValue> {
    pub proof: Proof<Value>,
    pub cfg: ProofConfig,
    pub params: GateAirLeafParams,
}

/// [`ChildVerifier`] impl that verifies ONE gate_air base proof (via [`emit_one_base`]) and returns
/// its fold-hash preimage chunk `[preprocessed_root (8 words), H_i unpacked words]` — byte-identical
/// to what a level-1 multiverifier node emits for a leaf child (`preprocessed_root` words then
/// `unpack_qm31s_to_u32_words(output_values)`), so a base-node's aggregate is consumable by an R2
/// node + the unpacker unchanged.
pub struct GateAirChildVerifier;

impl<Value: IValue> ChildVerifier<Value> for GateAirChildVerifier {
    type Input = GateAirBaseInput<Value>;

    fn verify_child(
        &self,
        context: &mut Context<Value>,
        input: Self::Input,
    ) -> Vec<U32Wrapper<Var>> {
        let GateAirBaseInput { proof, cfg, params } = input;
        let (pp_root, h_i_vars) = emit_one_base(context, proof, &cfg, &params);
        // Mirror the multiverifier per-child preimage: eight preprocessed-root words, then the output
        // digest unpacked into u32 words. `H_i` (eight QM31 `(lo, hi, 0, 0)` words) unpacks the same
        // way a leaf's `output_values` do, so the byte-identity contract with the unpacker holds.
        let output_words = unpack_qm31s_to_u32_words(context, h_i_vars);
        pp_root.into_iter().chain(output_words).collect()
    }
}

/// Builds a base-fanning node: verify `b` gate_air base proofs directly (each via [`emit_one_base`])
/// and fold them with the shared multiverifier fold-hash. This does the work of `b` leaves + one
/// level-1 (leaf-verifying) node in a SINGLE circuit — replacing the standalone leaf layer and the R1
/// node layer.
///
/// Its aggregate output is byte-identical to what a level-1 multiverifier node over `b` leaves emits
/// (per child `[preprocessed_root words, H_i words]`, children left-to-right, `blake2s_u32s`), so an
/// R2 node and the out-of-circuit unpacker consume a base-node exactly as they consume an R1 node.
/// Generic over `Value` so the same topology builds the NoValue shape (config derivation) and the
/// real QM31 assignment (proving).
pub fn build_gate_air_base_node_circuit<Value: IValue>(
    bases: Vec<GateAirBaseInput<Value>>,
) -> FinalizedContext<Value> {
    build_fanning_circuit(bases, &GateAirChildVerifier)
}

// =================================================================================================
// LEAF/R1/R2 topology ([`FoldMode::LeafR1R2`], PARALLEL to base-fanning) — restored from the
// pre-base-fanning code (grover-tax-v02 f7a0668). A standalone single-base LEAF (one gate_air base
// per leaf), a level-1 leaf-verifying (R1) node layer, and the shared R2 up-tree fold. Selected at the
// main.rs call sites by `FoldMode::LeafR1R2`; the base-fanning helpers above (`GateAirChildVerifier`,
// `build_gate_air_base_node_circuit`, `emit_one_base`, `derive_base_fanning_config(_ex)`,
// `prove_base_node`, ...) are UNTOUCHED and remain the default.
// =================================================================================================

/// Builds the gate_air single-base LEAF circuit ([`FoldMode::LeafR1R2`]): verify ONE gate_air base
/// proof in-circuit (via [`emit_one_base`]) and set the reserved outputs to its `H_i`. A b=1 leaf. The
/// multiverifier's level-0 R1 nodes verify these leaves. Generic over `Value` (NoValue shape / QM31
/// assignment).
pub fn build_gate_air_leaf_circuit<Value: IValue>(
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> FinalizedContext<Value> {
    let mut context = Context::new(N_RESERVED);
    let (_, h_i_vars) = emit_one_base(&mut context, proof, cfg, params);
    context.set_outputs(&h_i_vars);
    context.finalize(false)
}

/// Derives the leaf/R1/R2 `AggregateConfig` for gate_air leaves of this proof's shape
/// ([`FoldMode::LeafR1R2`]). Resolves the leaf↔node padding decoupling: the leaf pads to its OWN
/// natural target (~2^20), while the two full-`fold_arity` node variants (R1 verifies leaves, R2
/// verifies nodes) pad to a COMMON `node_target` fixed point. Populates the coexisting config's
/// LeafR1R2 extras (`leaf_shared_config`, R1, `leaf_preprocessed_root`, leaf target/PCS, leaf/level1
/// precomputes); the base-fanning-only fields (`base_node_preprocessed_root`, `base_preprocessed_root`)
/// are set to the leaf root (unused under this mode — the leaf unpacker never reads them).
pub fn derive_aggregate_config(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    fold_arity: usize,
    log_blowup_factor: u32,
) -> AggregateConfig {
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
        let pcs = leaf_pcs_config(leaf_pp.trace_log_size, log_blowup_factor);
        (leaf_pp, pcs)
    };
    let leaf_preprocessed_root = preprocessed_root(&leaf_pp, log_blowup_factor);
    let leaf_shared_config = shared_config_for_leaf(&leaf_pp, pcs);

    // The two node variants both pad to a COMMON `node_target` (self-verification fixed point):
    //   - level-1 node (R1): verifies `fold_arity` LEAVES (child config = leaf shape).
    //   - level-≥2 node (R2): verifies `fold_arity` NODES  (child config = node shape).
    let (_, node1_seed_sizes) = multiverifier_node_preprocessed(&leaf_pp, pcs, None, fold_arity);
    let mut node_target = node1_seed_sizes;
    let (level1_pp, node_pp) = loop {
        let (level1_pp, level1_unpadded) =
            multiverifier_node_preprocessed(&leaf_pp, pcs, Some(node_target.clone()), fold_arity);
        let (node_pp, node2_unpadded) =
            multiverifier_node_preprocessed(&level1_pp, pcs, Some(node_target.clone()), fold_arity);
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
    let node_pp = node_preprocessed_from_shared(&node_shared_config, node_target.clone(), fold_arity);
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

    // Witness-independent precomputes (GATE_AIR_NO_PRECOMPUTE=1 -> all None).
    let no_precompute = std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok();
    let (leaf_precompute, level1_precompute, node_precompute) = if no_precompute {
        (None, None, None)
    } else {
        (
            Some(Arc::new(CircuitPrecompute::new(
                leaf_pp,
                pcs,
                leaf_preprocessed_root.clone(),
            ))),
            Some(Arc::new(CircuitPrecompute::new(
                level1_pp,
                node_pcs,
                level1_preprocessed_root.clone(),
            ))),
            Some(Arc::new(CircuitPrecompute::new(
                node_pp,
                node_pcs,
                node_preprocessed_root.clone(),
            ))),
        )
    };

    AggregateConfig {
        // Shared / R2 (also used by the shared up-tree fold).
        node_shared_config,
        node_preprocessed_root,
        node_target_padding_sizes: node_target,
        node_pcs_config: node_pcs,
        node_precompute,
        fold_arity,
        // Base-fanning-only fields — unused under LeafR1R2; set to the leaf root (well-formed, unread).
        base_node_preprocessed_root: leaf_preprocessed_root.clone(),
        base_preprocessed_root: leaf_preprocessed_root.clone(),
        // LeafR1R2 tier.
        leaf_shared_config: Some(leaf_shared_config),
        level1_preprocessed_root: Some(level1_preprocessed_root),
        leaf_preprocessed_root: Some(leaf_preprocessed_root),
        leaf_target_padding_sizes: Some(leaf_target),
        leaf_pcs_config: Some(pcs),
        level1_precompute,
        leaf_precompute,
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
) -> TreeProof {
    let leaf_target = config
        .leaf_target_padding_sizes
        .clone()
        .expect("prove_gate_air_leaf requires a LeafR1R2 config (leaf_target_padding_sizes present)");
    let leaf_pcs = config
        .leaf_pcs_config
        .expect("prove_gate_air_leaf requires a LeafR1R2 config (leaf_pcs_config present)");
    let leaf_preprocessed_root = config
        .leaf_preprocessed_root
        .clone()
        .expect("prove_gate_air_leaf requires a LeafR1R2 config (leaf_preprocessed_root present)");
    let mut context = build_gate_air_leaf_circuit::<QM31>(real_proof, cfg, params);
    pad_to_targets(&mut context, leaf_target);
    let circuit_proof = match &config.leaf_precompute {
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

/// Host computation of one base's output digest `H_i = blake2s(H_P ‖ x ‖ y)`, the value the base-node
/// circuit emits per base and the unpacker binds per base (`BaseOutput.output_values`). Runs the SAME
/// circuit ops as `emit_one_base` — build `GateAirStatement`, `compute_h_p`, hash `H_P ‖ x ‖ y` — in a
/// throwaway QM31 context WITHOUT the STARK `verify` (H_i depends only on the guessed program /
/// boundary / nonce, not on verify's constraints), so it is byte-identical to what the base-node
/// proved, at negligible cost. Prover-side: a wrong `H_i` makes the unpacker's reconstruction miss the
/// verified root ⇒ the final-proof sanity check rejects (correctness, not soundness).
pub fn base_h_i_host(params: &GateAirLeafParams) -> [QM31; N_RESERVED] {
    let mut ctx = Context::<QM31>::new(N_RESERVED);
    let statement = GateAirStatement::<QM31>::new(
        &mut ctx,
        params.main_log_size,
        params.program_log_size,
        params.boundary_log_size,
        params.preprocessed_root.clone(),
        params.boundary.clone(),
        params.total_pc,
        params.program.clone(),
        params.nonce,
    );
    let h_p = statement.compute_h_p(&mut ctx);
    let mut preimage: Vec<_> = h_p.iter().map(|w| *w.get()).collect();
    for (x, y) in statement.boundary_vars() {
        preimage.extend(x.iter().chain(y.iter()).copied());
    }
    let output_hash: HashValue<_> = blake2s(&mut ctx, &preimage, 16 * preimage.len());
    std::array::from_fn(|i| ctx.get(*output_hash[i].get()))
}

/// The gate_air base-fanning config: an [`AggregateConfig`] (for the fold + unpacker) plus the
/// base-node-specific proving data recursive_aggregate is leaf-agnostic about.
///
/// The base-node (`R_base`) verifies `b` gate_air BASE proofs and folds them; its OWN proof is a
/// circuit-prover proof padded to the common `node_target` so R2 verifies it exactly like any node.
pub struct BaseFanConfig {
    /// Fold + unpack config (R2 machinery + the two trusted bottom roots).
    pub agg: AggregateConfig,
    /// Base-fanning arity `b` (from `recursive_aggregate::base_fan_arity`).
    pub b: usize,
    /// The gate_air base proof config each base-node child is verified against.
    pub base_cfg: ProofConfig,
    /// The shard-invariant base params shape (for rebuilding short-base-node shapes / roots). Its
    /// `preprocessed_root` is the shared base tree0 root.
    pub base_params_shape: GateAirLeafParams,
    /// The base-node PCS (== `agg.node_pcs_config`; a base-node proof is a node-sized proof).
    pub node_pcs_config: PcsConfig,
    /// Common node target padding (== `agg.node_target_padding_sizes`).
    pub node_target_padding_sizes: ComponentSizes,
    /// Witness-independent precompute for the FULL-`b` base-node circuit. `None` under
    /// GATE_AIR_NO_PRECOMPUTE. Short base-node groups (arity `< b`) rebuild tree0 per prove.
    pub base_node_precompute: Option<Arc<CircuitPrecompute>>,
}

/// Builds one NoValue base-node shape verifying `arity` empty gate_air bases of `params`' shape.
/// Returns the finalized context (for `compute_padded_sizes` / preprocessing). The shape is always
/// `NoValue` (`empty_proof` yields a `Proof<NoValue>`): the preprocessed trace + component sizes are
/// witness-independent, which is all the config derivation needs.
fn base_node_shape(
    arity: usize,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> FinalizedContext<NoValue> {
    let bases: Vec<GateAirBaseInput<NoValue>> = (0..arity)
        .map(|_| GateAirBaseInput {
            proof: empty_proof(cfg),
            cfg: cfg.clone(),
            params: params.clone(),
        })
        .collect();
    build_gate_air_base_node_circuit(bases)
}

/// Builds + preprocesses the NoValue full-`b` base-node circuit padded to `target`, returning its
/// preprocessed circuit + its UNPADDED component sizes (to grow the common target).
fn base_node_preprocessed(
    b: usize,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    target: Option<ComponentSizes>,
) -> (PreprocessedCircuit, ComponentSizes) {
    let mut ctx = base_node_shape(b, cfg, params);
    let unpadded = compute_padded_sizes(&ctx);
    if let Some(t) = target {
        pad_to_targets(&mut ctx, t);
    }
    (PreprocessedCircuit::preprocess_circuit(&mut ctx), unpadded)
}

/// Derives the base-fanning config for gate_air bases of `params`' shape and base-fan arity `b`
/// (production fold: base-nodes folded up by R2). See [`derive_base_fanning_config_ex`].
pub fn derive_base_fanning_config(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    b: usize,
    fold_arity: usize,
    log_blowup_factor: u32,
    canonical_base_preprocessed_root: HashValue<QM31>,
) -> BaseFanConfig {
    derive_base_fanning_config_ex(
        cfg,
        params,
        b,
        fold_arity,
        log_blowup_factor,
        false,
        canonical_base_preprocessed_root,
    )
}

/// Derives the base-fanning config for gate_air bases of `params`' shape and base-fan arity `b`.
///
/// Two modes selected by `single_base_node`:
///   - `false` (PRODUCTION fold): resolves the self-verification fixed point — the base-node (verifies
///     `b` bases) AND the R2 node (verifies FOLD_ARITY node-proofs of the common shape) both pad to a
///     COMMON `node_target`, so a base-node's own proof and an R2 node's proof share one shape (one
///     `node_shared_config`, one node PCS). Iterate `node_target = max(base_node, node2)` until it
///     stops growing. The base-node is thus padded UP to R2's size (~2^22 at FOLD_ARITY=8).
///   - `true` (SINGLE-BASE-NODE ROOT, no R2): used when the topology is one base-node that IS the root
///     (`b >= n_bases`, so `recursive_aggregate_prove` never builds an R2 fold). The base-node is
///     padded to its OWN natural target (~2^17 for a toy base) — mirroring the old leaf↔node padding
///     decoupling — and `node_shared_config` / `node_pcs_config` / `node_target_padding_sizes` are
///     derived from THAT so `prove_root_verification` verifies the lone base-node-as-root at its small
///     size. No R2 fixed point, no R2 proof — this is what makes the full roundtrip laptop-safe. The
///     `node_preprocessed_root` (R2) field is unused in this topology; it is set to `R_base` (harmless
///     — no R2 node is ever built or bound). Config-derivation ONLY: no verifier/constraint change,
///     so the base-node + root-verify circuits (hence soundness) are byte-identical to the production
///     path for the same padded shape; only the chosen padding target differs.
pub fn derive_base_fanning_config_ex(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    b: usize,
    fold_arity: usize,
    log_blowup_factor: u32,
    single_base_node: bool,
    canonical_base_preprocessed_root: HashValue<QM31>,
) -> BaseFanConfig {
    assert!(b >= 1, "base_fan_arity b must be >= 1");
    assert!(fold_arity >= 2, "fold_arity k must be >= 2");
    // Base tree0 root: shard-invariant, the single trusted base preprocessed root the unpacker BAKES
    // as a constant for EVERY base. SOUNDNESS (step 1): this MUST be the CANONICAL value recomputed at
    // build time from the trusted PUBLIC config (caller's `canonical_base_preprocessed_root`, via
    // `canonical_base_preprocessed_root(..)` — the tree0-root recompute), NOT `params.preprocessed_root`
    // (which is the prover's own `commitments[0]`, a forgeable value). Baking a non-canonical root
    // would leave the base preprocessed trace (positional pc, in-range rc table) unpinned.
    let base_preprocessed_root = canonical_base_preprocessed_root;

    let (node_target, node_pp_for_r2) = if single_base_node {
        // Pad the base-node to its OWN natural size — no R2, so no common fixed point. The base-node
        // IS the root here; `prove_root_verification` verifies it with the config derived below.
        let (_, base_node_own_sizes) = base_node_preprocessed(b, cfg, params, None);
        (base_node_own_sizes, None)
    } else {
        // PRODUCTION fixed point: pad the base-node + the R2 node to a common `node_target`.
        // A PCS sized from the base-node's own trace, used only to seed the fixed point.
        let (seed_base_node_pp, base_node_seed_sizes) = base_node_preprocessed(b, cfg, params, None);
        let seed_pcs = leaf_pcs_config(seed_base_node_pp.trace_log_size, log_blowup_factor);
        // R2 (verifies FOLD_ARITY base-node proofs) is typically LARGER than the base-node; seed the
        // common target from max(base-node, R2) so the FIRST iteration's padding never shrinks a
        // shape below its own size (which would trip the `<= target` pad assert).
        let (_, node2_seed_sizes) =
            multiverifier_node_preprocessed(&seed_base_node_pp, seed_pcs, None, fold_arity);
        let mut node_target = max_sizes(&base_node_seed_sizes, &node2_seed_sizes);
        let node_pp = loop {
            let (base_node_pp, base_node_unpadded) =
                base_node_preprocessed(b, cfg, params, Some(node_target.clone()));
            // R2 verifies `fold_arity` node-proofs of the CURRENT common node shape, padded to it.
            let (node_pp, node2_unpadded) = multiverifier_node_preprocessed(
                &base_node_pp,
                seed_pcs,
                Some(node_target.clone()),
                fold_arity,
            );
            let new_target = max_sizes(&base_node_unpadded, &node2_unpadded);
            if new_target == node_target {
                break node_pp;
            }
            node_target = new_target;
        };
        (node_target, Some(node_pp))
    };

    // R_base: the base-node's preprocessed root, padded to `node_target` (its own target in
    // single-base-node mode, else the common R2 target). This is also the shape a NODE proof has.
    let base_node_pp = {
        let mut ctx = base_node_shape(b, cfg, params);
        pad_to_targets(&mut ctx, node_target.clone());
        PreprocessedCircuit::preprocess_circuit(&mut ctx)
    };
    // NODE PCS from the (base-node = node) trace size (auth-path height = trace_log_size + blowup).
    let node_pcs = leaf_pcs_config(base_node_pp.trace_log_size, log_blowup_factor);
    // Config for verifying a NODE proof (R2's children + the root, or the lone base-node root).
    let node_shared_config = shared_config_for_leaf(&base_node_pp, node_pcs);
    let base_node_preprocessed_root = preprocessed_root(&base_node_pp, log_blowup_factor);

    let _ = node_pp_for_r2; // measured during the fixed point; the final R2 shape is rebuilt below.

    // R2 node shape — only in PRODUCTION mode (single-base-node mode builds no R2). Rebuild with the
    // NODE pcs for its children (matches `build_node_context` at prove time).
    let node_pp_r2: Option<PreprocessedCircuit> = if single_base_node {
        None
    } else {
        Some(node_preprocessed_from_shared(&node_shared_config, node_target.clone(), fold_arity))
    };
    // `node_preprocessed_root` (R2) — the real R2 root in production; in single-base-node mode there
    // is no R2, so this field is unused and set to `R_base` (harmless: no R2 node is built or bound).
    let node_preprocessed_root = match &node_pp_r2 {
        Some(pp) => preprocessed_root(pp, log_blowup_factor),
        None => base_node_preprocessed_root.clone(),
    };

    eprintln!(
        "gate-air: base-fanning: b={b}, base-node trace 2^{} (R_base){}, base tree0 root guessed per base",
        base_node_pp.trace_log_size,
        match &node_pp_r2 {
            Some(pp) => format!(", node trace 2^{} (R2)", pp.trace_log_size),
            None => " [single-base-node root, no R2]".to_string(),
        },
    );

    // Witness-independent precomputes. GATE_AIR_NO_PRECOMPUTE=1 leaves them None (rebuild-per-prove).
    // In single-base-node mode `node_precompute` (R2) is never used at prove time (no R2 node built).
    let no_precompute = std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok();
    let base_node_precompute = if no_precompute {
        None
    } else {
        Some(Arc::new(CircuitPrecompute::new(
            base_node_pp,
            node_pcs,
            base_node_preprocessed_root.clone(),
        )))
    };
    let node_precompute = match (no_precompute, node_pp_r2) {
        (false, Some(node_pp)) => Some(Arc::new(CircuitPrecompute::new(
            node_pp,
            node_pcs,
            node_preprocessed_root.clone(),
        ))),
        _ => None,
    };

    BaseFanConfig {
        agg: AggregateConfig {
            node_shared_config,
            node_preprocessed_root,
            base_node_preprocessed_root,
            base_preprocessed_root,
            node_target_padding_sizes: node_target.clone(),
            node_pcs_config: node_pcs,
            node_precompute,
            fold_arity,
            // Base-fanning mode carries no leaf/R1/R2 tier; those fields are `None`.
            leaf_shared_config: None,
            level1_preprocessed_root: None,
            leaf_preprocessed_root: None,
            leaf_target_padding_sizes: None,
            leaf_pcs_config: None,
            level1_precompute: None,
            leaf_precompute: None,
        },
        b,
        base_cfg: cfg.clone(),
        base_params_shape: params.clone(),
        node_pcs_config: node_pcs,
        node_target_padding_sizes: node_target,
        base_node_precompute,
    }
}

/// The trusted preprocessed root a base-node of `arity` bases reports — full-`b` groups return
/// `R_base` (cached in `config.agg`); a short trailing group (`2..=b-1`) recomputes its own
/// `R_base'(m)` witness-independently (byte-identical to what `prove_base_node` reports for that
/// shape). Public + arity-derived; feeds `BaseFanBottom.base_node_roots` for the unpacker.
pub fn base_node_root_for_arity(config: &BaseFanConfig, arity: usize) -> HashValue<QM31> {
    if arity == config.b {
        return config.agg.base_node_preprocessed_root.clone();
    }
    let mut ctx = base_node_shape(arity, &config.base_cfg, &config.base_params_shape);
    pad_to_targets(&mut ctx, config.node_target_padding_sizes.clone());
    let pp = PreprocessedCircuit::preprocess_circuit(&mut ctx);
    preprocessed_root(&pp, config.node_pcs_config.fri_config.log_blowup_factor)
}

/// Builds + proves one BASE-NODE over `bases` (`m = bases.len()` gate_air base proofs, `1..=b`),
/// padded/configured to match `config` so an R2 node + the unpacker consume it. Each base carries its
/// own [`GateAirLeafParams`] (own preprocessed root, boundary, program, nonce); all share
/// `config.base_cfg`.
///
/// Returns the base-node [`TreeProof`] (its reported root is `R_base` for a full-`b` group, else the
/// recomputed `R_base'(m)` — [`base_node_root_for_arity`]) AND the per-base [`BaseOutput`] hints the
/// unpacker needs (each base's `(preprocessed_root, H_i)`), in base order.
///
/// Reuses `config.base_node_precompute` for full-`b` groups; a short group has a distinct shape, so it
/// rebuilds tree0 per call (rare — at most one short group per run).
pub fn prove_base_node(
    bases: Vec<(Proof<QM31>, GateAirLeafParams)>,
    config: &BaseFanConfig,
) -> (TreeProof, Vec<BaseOutput>) {
    let m = bases.len();
    assert!((1..=config.b).contains(&m), "base-node arity must be 1..=b (got {m})");

    // Per-base unpacker hints: each base's own preprocessed root + host-computed H_i (byte-identical
    // to the H_i the base-node circuit emits for that base).
    let base_outputs: Vec<BaseOutput> = bases
        .iter()
        .map(|(_, params)| BaseOutput {
            preprocessed_root: params.preprocessed_root.clone(),
            output_values: base_h_i_host(params),
        })
        .collect();

    let inputs: Vec<GateAirBaseInput<QM31>> = bases
        .into_iter()
        .map(|(proof, params)| GateAirBaseInput { proof, cfg: config.base_cfg.clone(), params })
        .collect();
    let mut context = build_gate_air_base_node_circuit::<QM31>(inputs);
    // Pad the base-node to the common node target so R2 verifies it identically to any node.
    pad_to_targets(&mut context, config.node_target_padding_sizes.clone());

    // Full-`b` groups reuse the witness-independent precompute; short groups rebuild tree0 per call.
    let is_full = m == config.b;
    let circuit_proof = match (is_full, &config.base_node_precompute) {
        (true, Some(pc)) => prove_circuit_with_precompute::<Blake2sM31MerkleChannel>(
            &pc.base_column_pool,
            &pc.twiddles,
            &pc.preprocessed,
            MaybeOwned::Borrowed(&pc.tree),
            context.values(),
            pc.pcs_config,
        ),
        _ => {
            let preprocessed = PreprocessedCircuit::preprocess_circuit(&mut context);
            prove_circuit_assignment(
                context.values(),
                &preprocessed,
                &BaseColumnPool::<SimdBackend>::new(),
                config.node_pcs_config,
            )
        }
    }
    .expect("gate_air base-node prove failed");
    let (proof, public_data) = prepare_circuit_proof_for_circuit_verifier(circuit_proof);
    let output_values = public_data
        .output_values
        .try_into()
        .expect("base-node emits N_RESERVED outputs");

    let tree_proof = TreeProof {
        proof,
        preprocessed_root: base_node_root_for_arity(config, m),
        output_values,
    };
    (tree_proof, base_outputs)
}
