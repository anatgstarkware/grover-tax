//! Milestone 3: turn the gate_air verification circuit into a foldable leaf and fold N of them.
//!
//! A leaf = the circuit that verifies the gate_air stwo proof (via `circuits_stark_verifier::verify`
//! + `GateAirStatement`), with its 2 reserved outputs set to a commitment, proved with the circuit
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
    AggregateConfig, CircuitPrecompute, FOLD_ARITY, TreeProof, multiverifier_node_preprocessed,
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

/// Builds the gate_air leaf circuit: verify the gate_air proof in-circuit and set the 2 reserved
/// outputs to `H_i = blake( H_P ‖ x_limbs ‖ y_limbs )`, with `H_P = blake( program_table ‖ nonce )`.
/// Generic over `Value` so the same topology builds the NoValue shape (config derivation) and the
/// real QM31 assignment (proving).
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

/// Verifies ONE gate_air base proof in-circuit and emits its output digest `H_i`. This is the
/// per-child body shared by the single-base leaf ([`build_gate_air_leaf_circuit`]) and the
/// base-fanning node ([`build_gate_air_base_node_circuit`]) — the whole security-relevant binding
/// lives here, so folding `b` bases is exactly `b` independent copies of this call (each its own
/// statement / verify / channel).
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
        let output_words = unpack_qm31s_to_u32_words(context, h_i_vars.into_iter());
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

/// Derives the multiverifier `AggregateConfig` for gate_air leaves of this proof's shape. Resolves
/// the target_padding/PCS/node-size fixed point exactly like the cairo leaf path.
pub fn derive_aggregate_config(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    log_blowup_factor: u32,
) -> AggregateConfig {
    let shape = || build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params);
    let leaf_sizes = compute_padded_sizes(&shape());

    // LEAF↔NODE PADDING DECOUPLING. Pad the leaf to its OWN target (its natural ~2^20), NOT
    // `max(leaf, node)`, so `t_leaf` is pinned independent of `FOLD_ARITY` (the k-ary salvage). The
    // leaf shape has no self-verification fixed point (a leaf verifies a base gate_air STARK, not a
    // recursion proof), so its target is simply `leaf_sizes` (the padded component sizes).
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

    // The two node variants both pad to a COMMON `node_target` so their output proofs share one shape
    // (one `node_shared_config`, one PCS) — only their gate structure (hence R1 vs R2) differs.
    //   - level-1 node (R1): verifies FOLD_ARITY LEAVES  (child config = leaf shape).
    //   - level-≥2 node (R2): verifies FOLD_ARITY NODES   (child config = node shape).
    // `node_target = max(node1_sizes, node2_sizes)`. R2's own shape is the child shape it verifies,
    // so this is a self-verification fixed point: pad both variants to `node_target`, recompute each
    // one's sizes verifying children of the (node_target-padded) node shape, and iterate until the
    // common target stops growing. Seed the node shape from the level-1 (leaf-verifying) node.
    // `multiverifier_node_preprocessed` returns the node's UNPADDED component sizes (measured before
    // it pads), so a single call yields both the padded preprocessed circuit and the sizes needed to
    // grow the common target.
    let (_, node1_seed_sizes) = multiverifier_node_preprocessed(&leaf_pp, pcs, None);
    let mut node_target = node1_seed_sizes;
    let (level1_pp, node_pp) = loop {
        // level-1 node: verifies leaves (leaf_pp config), padded to the common node_target.
        let (level1_pp, level1_unpadded) =
            multiverifier_node_preprocessed(&leaf_pp, pcs, Some(node_target.clone()));
        // level-≥2 node: verifies nodes of the CURRENT common node shape (level1_pp is a node proof
        // of that shape), padded to the common node_target.
        let (node_pp, node2_unpadded) =
            multiverifier_node_preprocessed(&level1_pp, pcs, Some(node_target.clone()));
        let new_target = max_sizes(&level1_unpadded, &node2_unpadded);
        if new_target == node_target {
            break (level1_pp, node_pp);
        }
        node_target = new_target;
    };
    let level1_preprocessed_root = preprocessed_root(&level1_pp, log_blowup_factor);

    // LEAF↔NODE PCS DECOUPLING. A node proves a 2^22 trace, so a node proof's Merkle auth-path
    // height is `node_trace_log_size + log_blowup` (~25) — larger than the leaf's (~24). The single
    // PCS derived from the LEAF size above (`pcs`, lifting ~24) correctly describes a leaf proof
    // (verified by a level-1/R1 node) but MIS-SIZES a node proof: verifying a node child (R2 node)
    // or the root with a leaf-sized PCS makes the Merkle check assert `path.len()(25) != height(24)`.
    // Derive a separate node PCS from the node trace size for everything that describes a NODE's own
    // proof (its own prove + the R2 child-verify + the root verify).
    let node_pcs = leaf_pcs_config(node_pp.trace_log_size, log_blowup_factor);

    // Config for verifying a NODE proof (level-≥2 nodes' children + the root). Level-independent
    // because both variants share the common node_target shape; derive it from `level1_pp` — but with
    // the NODE pcs (lifting ~25), since the proof it describes is a 2^22 node proof.
    let node_shared_config = shared_config_for_leaf(&level1_pp, node_pcs);

    // Rebuild the level-≥2 (node-verifying) node shape with the NODE pcs for its children. The loop
    // above sized the common `node_target` fixed point with the leaf pcs; the +1 auth-path step from
    // the node child's larger lifting is absorbed by the 2^22 target padding, so `node_target` is
    // unchanged. But the level-≥2 node's preprocessed circuit (and hence its R2 root and the
    // precompute it proves against) MUST verify lifting-25 node children — exactly what
    // `build_node_context` builds at prove time from `node_shared_config`. Rebuild `node_pp` from
    // that config so the cached precompute matches the proved context.
    let node_pp = node_preprocessed_from_shared(&node_shared_config, node_target.clone(), FOLD_ARITY);
    let node_preprocessed_root = preprocessed_root(&node_pp, log_blowup_factor);

    // Whether the two padded node shapes coincide (R1 == R2, a 1-root collapse). Reported by the
    // topology test; the code path is identical either way.
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

    // Build the witness-independent proving precompute (committed tree0 + twiddles) once for the leaf,
    // the level-1 node, and the level-≥2 node shapes; every prove reuses it instead of rebuilding
    // tree0. `new` asserts the cached tree's root equals the already-trusted root before any proof.
    // GATE_AIR_NO_PRECOMPUTE=1 leaves all caches `None` so the same build runs the fallback
    // (rebuild-tree0-per-prove) path, for byte-identity validation.
    let no_precompute = std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok();
    let (leaf_precompute, level1_precompute, node_precompute) = if no_precompute {
        (None, None, None)
    } else {
        (
            // The leaf precompute PROVES a leaf (2^21) → leaf pcs.
            Some(Arc::new(CircuitPrecompute::new(
                leaf_pp,
                pcs,
                leaf_preprocessed_root.clone(),
            ))),
            // Both node precomputes PROVE a 2^22 node trace → node pcs (they differ only in the child
            // config they verify, not in their own proof shape).
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
        leaf_shared_config,
        node_shared_config,
        node_preprocessed_root,
        level1_preprocessed_root,
        leaf_preprocessed_root,
        node_target_padding_sizes: node_target,
        leaf_target_padding_sizes: leaf_target,
        leaf_pcs_config: pcs,
        node_pcs_config: node_pcs,
        node_precompute,
        level1_precompute,
        leaf_precompute,
    }
}

/// Builds + proves one gate_air leaf, padded/configured to match `config` (so the multiverifier can
/// verify it). `real_proof` is the gate_air proof as circuit values (from `proof_from_stark_proof`).
pub fn prove_gate_air_leaf(
    real_proof: Proof<QM31>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    config: &AggregateConfig,
) -> TreeProof {
    let mut context = build_gate_air_leaf_circuit::<QM31>(real_proof, cfg, params);
    // Pad the leaf to its OWN target (~2^20), decoupled from the k-child node size, so `t_leaf`
    // stays pinned independent of FOLD_ARITY.
    pad_to_targets(&mut context, config.leaf_target_padding_sizes.clone());
    // Reuse the witness-independent precompute (committed tree0 + twiddles) when present, otherwise
    // fall back to the self-contained path that rebuilds tree0 per call.
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
                config.leaf_pcs_config,
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
        preprocessed_root: config.leaf_preprocessed_root.clone(),
        output_values,
    }
}
