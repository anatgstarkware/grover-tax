//! Proof-fingerprint observation helpers behind the prove path's env-gated hooks
//! (`GATE_AIR_PROOF_HASH`, `GATE_AIR_BASE_PROOF_HASH`, `GATE_AIR_RECURSION_FP`). READ-ONLY taps: each
//! is a stable SHA over an already-produced proof, feeding no committed value, so leaving them off is
//! byte-neutral. Compiled unconditionally so `GATE_AIR_*` can be flipped at run time on the
//! production-fast binary. Distinct from the `diag` Cargo FEATURE (which gates the separate
//! `cuda,diag` byte-identity test harness): this module is always compiled and runtime env-gated.

use recursive_aggregate::root_prover::RootVerificationOutput;
use recursive_aggregate::{AggregateOutput, TreeProof};
use sha2::{Digest, Sha256};

use crate::prover::BaseShardOutput;

/// Deterministic SHA-256 over the serde-serialized `StarkProof` — the full-proof byte-identity
/// fingerprint (`GATE_AIR_PROOF_HASH`). Cross-run/-backend stable, so the backend under test is the
/// only possible source of divergence. Prints `gate-air: proof_fingerprint=<hex>`.
pub fn emit_proof_fingerprint<H>(extended: &stwo::core::proof::ExtendedStarkProof<H>)
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    stwo::core::proof::StarkProof<H>: serde::Serialize,
{
    // Fingerprint ONLY `proof` (StarkProof), not `aux`: aux's HashMaps have per-process-randomized
    // serde order, which would make the fingerprint differ run-to-run without any proof divergence.
    let mut hasher = Sha256::new();
    match serde_json::to_vec(&extended.proof) {
        Ok(bytes) => {
            hasher.update(b"gate-air/stark-proof/serde/v1");
            hasher.update(&bytes);
        }
        Err(e) => {
            eprintln!("gate-air: proof serde failed ({e}); falling back to Debug fingerprint");
            hasher.update(b"gate-air/stark-proof/debug/v1");
            hasher.update(format!("{:?}", extended.proof).as_bytes());
        }
    }
    let digest = hasher.finalize();
    println!("gate-air: proof_fingerprint={}", hex::encode(digest));
}

/// Deterministic SHA-256 over the proved base shards (`GATE_AIR_BASE_PROOF_HASH`). Returns the hex.
/// Base precompute must not change any shard proof, so `base_precompute_identity` compares this
/// value precompute-ON vs rebuild-per-shard.
pub fn base_proof_fingerprint(shard_bases: &[BaseShardOutput]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"gate-air/base-shard-proofs/serde/v1");
    hasher.update(format!("n_shards={}", shard_bases.len()).as_bytes());
    for (s, base) in shard_bases.iter().enumerate() {
        let bytes = serde_json::to_vec(&base.0.proof).expect("serialize base shard StarkProof");
        hasher.update(format!("shard[{s}].proof=").as_bytes());
        hasher.update(&bytes);
        hasher.update(format!("shard[{s}].claim={:?}", base.1).as_bytes());
        hasher.update(format!("shard[{s}].nonce={}", base.2).as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Deterministic SHA-256 over the whole recursion (`GATE_AIR_RECURSION_FP`) — the per-run
/// byte-identity anchor. Folds every leaf/node proof + outputs, the root proof + outputs, and the
/// unpacked leaf outputs. k=500 reference: `32d827a2`.
pub fn recursion_fingerprint(
    base_nodes: &[TreeProof],
    out: &AggregateOutput,
    rv: &RootVerificationOutput,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"gate-air/recursion-proofs/debug/v1");
    hasher.update(
        format!(
            "n_base_nodes={} n_levels={}",
            base_nodes.len(),
            out.n_levels
        )
        .as_bytes(),
    );
    for (i, bn) in base_nodes.iter().enumerate() {
        hasher.update(format!("base_node[{i}].proof={:?}", bn.proof).as_bytes());
        hasher.update(format!("base_node[{i}].pp_root={:?}", bn.preprocessed_root).as_bytes());
        hasher.update(format!("base_node[{i}].outs={:?}", bn.output_values).as_bytes());
    }
    hasher.update(format!("root.proof={:?}", out.root.proof).as_bytes());
    hasher.update(format!("root.pp_root={:?}", out.root.preprocessed_root).as_bytes());
    hasher.update(format!("root.outs={:?}", out.root.output_values).as_bytes());
    hasher.update(format!("rv.proof={:?}", rv.proof).as_bytes());
    hasher.update(format!("rv.leaf_outputs={:?}", rv.leaf_outputs).as_bytes());
    hex::encode(hasher.finalize())
}
