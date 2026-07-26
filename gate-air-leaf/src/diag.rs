//! Debug/diag-only runtime self-checks moved off the release prover hot path. The whole module is
//! `#[cfg(any(debug_assertions, test))]` (declared in main.rs); items keep their own cfg so they
//! stay callable from the `#[cfg(debug_assertions)]` call sites. Pure relocation — no logic change.

use crate::prover::BaseProverPrecompute;
use crate::tracegen::{build_tree0_columns, to_prover, Row};
use crate::tree0_max_log_size;
use stwo::core::channel::Blake2sM31Channel;
#[cfg(debug_assertions)]
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::CommitmentSchemeProver;

/// Load-bearing soundness check for the base precompute: independently rebuilds shard 0's tree-0 via a
/// fresh throwaway `CommitmentSchemeProver` and asserts the cached root, column count, and per-column
/// domain sizes match. A mismatch (wrong column order / blowup / lifting / sort) aborts before any
/// reused proof is built. Debug/test only (a full duplicate tree0 build), so release pays nothing.
#[cfg(any(debug_assertions, test))]
pub(crate) fn assert_tree0_matches_rebuild(
    pc: &BaseProverPrecompute,
    rows0: &[Row],
    n_gates: usize,
) {
    // Rebuild via a fresh scheme/channel; columns from the same builder.
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(
            tree0_max_log_size(
                pc.log_n_rows,
                pc.rc_log,
                pc.program.log_size,
                pc.qubitmem.log_size,
            ) + 1
                + pc.config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );
    let mut scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(pc.config, &twiddles);
    let cols = build_tree0_columns(
        &pc.program,
        rows0,
        pc.padded_rows,
        pc.log_n_rows,
        n_gates,
        pc.rc_log,
        &pc.qubitmem,
    );
    let n_cols = cols.len();
    let mut tb = scheme.tree_builder();
    tb.extend_evals(to_prover(cols));
    let mut throwaway_channel = Blake2sM31Channel::default();
    tb.commit(&mut throwaway_channel);

    let rebuilt = &scheme.trees[0];
    // Root equality (the value mixed into every shard's transcript).
    assert_eq!(
        pc.tree0.commitment.root(),
        rebuilt.commitment.root(),
        "base-precompute tree0 root != rebuilt shard-0 root (column order/blowup/lifting mismatch)"
    );
    // Column count.
    assert_eq!(
        pc.tree0.polynomials.len(),
        n_cols,
        "base-precompute tree0 column count != rebuilt"
    );
    assert_eq!(
        rebuilt.polynomials.len(),
        n_cols,
        "rebuilt tree0 column count != expected"
    );
    assert_eq!(
        n_cols,
        crate::preprocessed::N_PREPROCESSED_COLS,
        "tree0 column count != N_PREPROCESSED_COLS"
    );
    // Per-column committed domain sizes (the lifted-Merkle sort order).
    for (i, (a, b)) in pc
        .tree0
        .polynomials
        .iter()
        .zip(rebuilt.polynomials.iter())
        .enumerate()
    {
        assert_eq!(
            a.evals.domain.log_size(),
            b.evals.domain.log_size(),
            "base-precompute tree0 column {i} size != rebuilt"
        );
    }
    eprintln!(
        "gate-air: base-precompute tree0 root-equality OK ({} cols, root matches rebuilt shard-0)",
        n_cols
    );
}

/// Debug-only prover self-check: the base shard's claimed LogUp sums must net to the public terms
/// `B + P_pub` (qubitmem + program-public). Pure tripwire — the recomputed publics feed nothing
/// downstream. Returns `Err` (not a panic) to match the prover's error path.
#[cfg(debug_assertions)]
pub(crate) fn assert_claimed_sums_net(
    qubitmem: &crate::tracegen::QubitMemTable,
    program: &crate::tracegen::ProgramTable,
    elements: &crate::air::LookupElements,
    main_sum: SecureField,
    program_sum: SecureField,
    qubitmem_sum: SecureField,
    rc_sum: SecureField,
) -> anyhow::Result<()> {
    let b_public = crate::tracegen::qubitmem_public_term(qubitmem, &elements.qubitmem);
    let p_pub = crate::tracegen::program_public_term(program, &elements.program);
    if main_sum + program_sum + qubitmem_sum + rc_sum != b_public + p_pub {
        anyhow::bail!("shard claimed sums do not net to the public terms B + P_pub");
    }
    Ok(())
}
