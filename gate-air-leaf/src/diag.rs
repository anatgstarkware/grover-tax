//! Diagnostic / observation helpers — the bodies behind the prove path's light, env-gated
//! observation hooks (`GATE_AIR_PROOF_HASH`, `GATE_AIR_BASE_PROOF_HASH`, `GATE_AIR_RECURSION_FP`).
//!
//! These are READ-ONLY taps: each is a stable SHA over an already-produced proof / proof set,
//! touching no constraint-eval / prover / verifier math and computing nothing that feeds a
//! committed value. Removing the hooks or leaving them off is byte-neutral to the proof. They live
//! here (out of the prove functions) so the prove path keeps only a one-line env-gated call; the
//! `diag` feature exists for parity with the test-support harnesses but these helpers are compiled
//! unconditionally so `GATE_AIR_*` can be flipped at RUN time on the production-fast binary.

use recursive_aggregate::{AggregateOutput, RootVerificationOutput, TreeProof};
use sha2::{Digest, Sha256};

use crate::base::BaseShardOutput;

/// STABLE, deterministic SHA-256 over the serde-serialized `ExtendedStarkProof` — the full-proof
/// byte-identity fingerprint (`GATE_AIR_PROOF_HASH`). Prints `gate-air: proof_fingerprint=<hex>`.
///
/// Determinism: the proof is a pure function of (fixture, samples, canonical Fiat-Shamir
/// transcript). serde_json serializes struct fields in declaration order and field elements /
/// hashes as plain numbers / byte arrays, so the byte stream is identical across runs and across
/// backends (SimdBackend vs CudaBackend). The backend under test is therefore the ONLY possible
/// source of divergence — the point of the cross-backend comparison (T7).
///
/// We hash the serde form (not `format!("{:?}", ..)`) because it is a canonical, version-stable
/// encoding; the Debug form is kept as a fallback if serialization were ever to fail.
pub fn emit_proof_fingerprint<H>(extended: &stwo::core::proof::ExtendedStarkProof<H>)
where
    H: stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted,
    stwo::core::proof::StarkProof<H>: serde::Serialize,
{
    // Fingerprint ONLY the verifier-consumed `proof` (StarkProof): it is entirely Vec/struct-based
    // and therefore serializes deterministically. The `aux` (in-circuit-verifier helper data)
    // contains HashMaps whose serde iteration order is randomized per process — including it makes
    // the fingerprint differ run-to-run even on the SAME backend, which is NOT a proof divergence.
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
///
/// The base-precompute optimization only changes HOW each shard's tree0/twiddles/program/N3 are
/// built, never WHAT — so every shard's base proof must be byte-identical precompute-ON vs
/// rebuild-per-shard. This fingerprints the base proofs directly; `base_precompute_identity` (T2)
/// drives the two paths and compares this value.
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
/// byte-identity anchor. Returns the hex. Folds every leaf/node proof (the fold's height-1
/// `base_nodes`) + their outputs, the root proof + outputs, and the unpacked leaf outputs.
///
/// `Proof<QM31>` is purely Vec/array/struct of QM31 (no maps), so its `{:?}` Debug form is a
/// deterministic, cross-process canonical encoding. `recursion_precompute_identity` (T3) compares
/// precompute-ON vs OFF via this value; the k=500 gate is `32d827a2`.
pub fn recursion_fingerprint(
    base_nodes: &[TreeProof],
    out: &AggregateOutput,
    rv: &RootVerificationOutput,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"gate-air/recursion-proofs/debug/v1");
    hasher.update(
        format!("n_base_nodes={} n_levels={}", base_nodes.len(), out.n_levels).as_bytes(),
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
