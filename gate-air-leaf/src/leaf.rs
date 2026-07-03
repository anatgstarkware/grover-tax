//! Milestone 3: turn the gate_air verification circuit into a foldable leaf and fold N of them.
//!
//! A leaf = the circuit that verifies the gate_air stwo proof (via `circuits_stark_verifier::verify`
//! + `GateAirStatement`), with its 2 reserved outputs set to a commitment, proved with the circuit
//! prover. That makes it a circuit-prover proof the multiverifier folds (`recursive_aggregate`).
//!
//! Output encoding (M3a stepping stone): `blake(preprocessed_root ‖ x_limbs ‖ y_limbs)` per shot —
//! binds the verified state boundary (x→y). The full secret-circuit `H_i = blake(H_P ‖ x ‖ y)`
//! (folding in the program commitment `H_P`) is the next refinement.

use circuits::blake::{ReducedHashValue, blake2s_m31};
use circuits::context::{Context, FinalizedContext};
use circuits::ivalue::{IValue, NoValue};
use circuits::ops::Guess;
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

use recursive_aggregate::{
    AggregateConfig, CircuitPrecompute, TreeProof, multiverifier_node_preprocessed,
    preprocessed_root, shared_config_for_leaf,
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
    pub preprocessed_root: ReducedHashValue<QM31>,
    pub boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    pub total_pc: u32,
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
/// outputs to `blake(preprocessed_root ‖ x_limbs ‖ y_limbs)`. Generic over `Value` so the same
/// topology builds the NoValue shape (config derivation) and the real QM31 assignment (proving).
pub fn build_gate_air_leaf_circuit<Value: IValue>(
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> FinalizedContext<Value> {
    let mut context = Context::new(N_RESERVED);
    let statement = GateAirStatement::<Value>::new(
        &mut context,
        params.main_log_size,
        params.program_log_size,
        params.preprocessed_root.clone(),
        params.boundary.clone(),
        params.total_pc,
    );
    let proof_vars = proof.guess(&mut context);
    verify(&mut context, &proof_vars, cfg, &statement);

    // Leaf output: hash the (now GUESSED, witness) preprocessed root and verified state boundary, so
    // the 2 reserved outputs commit to (x, y) for every shot. Preimage = [ppR.0, ppR.1, x.., y..].
    // These are the SAME guessed Vars that `verify` bound: the root via the preprocessed-trace
    // Merkle decommitment, and the x/y via the boundary LogUp sum. So the commitment is over the
    // bound values, and keeping them witness (not `context.constant`) keeps the leaf's preprocessed
    // trace — and `leaf_preprocessed_root` — identical across shards.
    let pp_root = statement.preprocessed_root_vars();
    let mut preimage = vec![pp_root.0, pp_root.1];
    for (x, y) in statement.boundary_vars() {
        preimage.extend(x.iter().chain(y.iter()).copied());
    }
    let output_hash = blake2s_m31(&mut context, &preimage, 16 * preimage.len());
    context.set_outputs(&[output_hash.0, output_hash.1]);

    context.finalize(false)
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

    let mut target = leaf_sizes.clone();
    let (leaf_pp, pcs) = loop {
        let mut leaf_ctx = shape();
        pad_to_targets(&mut leaf_ctx, target.clone());
        let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
        let pcs = leaf_pcs_config(leaf_pp.trace_log_size, log_blowup_factor);
        let (_, node_sizes) = multiverifier_node_preprocessed(&leaf_pp, pcs, None);
        let new_target = max_sizes(&leaf_sizes, &node_sizes);
        if new_target == target {
            break (leaf_pp, pcs);
        }
        target = new_target;
    };

    let leaf_preprocessed_root = preprocessed_root(&leaf_pp, log_blowup_factor);
    let (node_pp, _) = multiverifier_node_preprocessed(&leaf_pp, pcs, Some(target.clone()));
    let node_preprocessed_root = preprocessed_root(&node_pp, log_blowup_factor);

    // Build the shared config before the precompute moves `leaf_pp`.
    let shared_config = shared_config_for_leaf(&leaf_pp, pcs);

    // Build the witness-independent proving precompute (committed tree0 + twiddles) once for the leaf
    // and node shapes; every leaf/node prove reuses it instead of rebuilding tree0. `new` asserts the
    // cached tree's root equals the already-trusted root before any proof is produced.
    // GATE_AIR_NO_PRECOMPUTE=1 leaves both caches `None` so the same build runs the fallback
    // (rebuild-tree0-per-prove) path, for byte-identity validation.
    let no_precompute = std::env::var("GATE_AIR_NO_PRECOMPUTE").is_ok();
    let (leaf_precompute, node_precompute) = if no_precompute {
        (None, None)
    } else {
        (
            Some(Arc::new(CircuitPrecompute::new(
                leaf_pp,
                pcs,
                leaf_preprocessed_root,
            ))),
            Some(Arc::new(CircuitPrecompute::new(
                node_pp,
                pcs,
                node_preprocessed_root,
            ))),
        )
    };

    AggregateConfig {
        shared_config,
        node_preprocessed_root,
        leaf_preprocessed_root,
        target_padding_sizes: target,
        pcs_config: pcs,
        node_precompute,
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
    pad_to_targets(&mut context, config.target_padding_sizes.clone());
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
                config.pcs_config,
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
        preprocessed_root: config.leaf_preprocessed_root,
        output_values,
    }
}
