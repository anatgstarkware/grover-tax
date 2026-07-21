//! Free topology parameters and the env-var knob layer for the recursion + base-proof pipeline.
//!
//! This is the env-parsing home for the whole binary: [`TopologyConfig::from_env`] reads the sweep
//! knobs (`BASE_BLOWUP`, `LEAF_BLOWUP`, `RECURSION_FOLD_ARITY`, `RECURSION_SHARD_SHOTS`) and produces
//! a plain [`TopologyConfig`] that the `recursive_aggregate` library then consumes as pure derived
//! config (it does no env parsing of its own).

/// Fold arity `k`: each internal node verifies exactly this many children (`k`-to-1 fold).
///
/// The single source of truth for the arity across the recursion pipeline — the tree/streaming fold,
/// the topology, `prove_fold_node`, and the unpacker's per-node hash preimage all read it, so the
/// out-of-circuit unpacker and the in-circuit node hash agree. Re-sweep the arity by changing only
/// this constant; nothing else hard-codes the child count.
///
/// A level's `len() % FOLD_ARITY` (< k) remainder is carried up unchanged, so nodes are always
/// exactly `k` children — never variable-child.
pub const FOLD_ARITY: usize = 8;

/// Default recursion (node-node / root) FRI blowup factor. Feeds the ~96-bit-secure `(pow_bits,
/// n_queries)` table via `get_pcs_config`; the value that makes production node/root proofs.
pub const RECURSION_LOG_BLOWUP: u32 = 3;

/// Default base (shard / "leaf") proof FRI blowup factor. `(pow_bits, n_queries)` and lifting
/// are derived from it via `leaf_pcs_config` to a ~96-bit-secure config. Sweep knob: 1/2/3.
pub const BASE_LOG_BLOWUP: u32 = 1;

/// Default shots (iadd256 executions) per base shard — the manual partition knob. `n_shards =
/// ceil(samples / shots_per_shard)`.
pub const SHOTS_PER_SHARD: usize = 2;

/// All FREE topology parameters, in one place, threaded through the recursion + base-proof pipeline.
///
/// Every field is a *free knob*; everything else (the `(pow_bits, n_queries)` at 96-bit, the trusted
/// roots, the PCS/padding targets, `n_shards`, the base-shard trace log) is DERIVED from these.
/// Security params (`fold_step`, `log_last_layer`, the 96-bit floor, `INTERACTION_POW_BITS`) are
/// pinned, NOT exposed here — they must never be swept below the security floor.
///
/// Construct once (via [`TopologyConfig::from_env`] on the production path, or a literal in a test)
/// and thread it: `fold_arity` rides on the `AggregateConfig` (so every config-carrying fold fn
/// reads `config.fold_arity`), while the blowups / `shots_per_shard` are read at the construction /
/// derivation sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyConfig {
    /// Base (shard) proof FRI blowup factor. Default [`BASE_LOG_BLOWUP`] (env `BASE_BLOWUP`).
    pub base_log_blowup: u32,
    /// Recursion (node-node / root) FRI blowup factor. Default [`RECURSION_LOG_BLOWUP`].
    pub recursion_log_blowup: u32,
    /// Leaf-wrap FRI blowup factor — the multiverifier leaf that verifies a base proof. Default
    /// [`RECURSION_LOG_BLOWUP`] (env `LEAF_BLOWUP`), i.e. equal to `recursion_log_blowup` unless
    /// overridden. Decoupled from the level1-/fold-node blowup so the leaf-wrap trace/lift (which
    /// scales with shots/shard) can be tuned independently; the level1-node verifies leaves at
    /// whatever this is.
    pub leaf_log_blowup: u32,
    /// Node-node fold arity `k` (each internal fold-node verifies exactly `k` children). Default
    /// [`FOLD_ARITY`].
    pub fold_arity: usize,
    /// Shots per base shard (partition knob). Default [`SHOTS_PER_SHARD`] (env `RECURSION_SHARD_SHOTS`).
    pub shots_per_shard: usize,
}

impl Default for TopologyConfig {
    /// Production values. Bottom-layer topology is the standalone-leaf → level-0 leaf-verifying
    /// (level1-node) → shared fold-node up-tree fold.
    fn default() -> Self {
        TopologyConfig {
            base_log_blowup: BASE_LOG_BLOWUP,
            recursion_log_blowup: RECURSION_LOG_BLOWUP,
            leaf_log_blowup: RECURSION_LOG_BLOWUP,
            fold_arity: FOLD_ARITY,
            shots_per_shard: SHOTS_PER_SHARD,
        }
    }
}

impl TopologyConfig {
    /// The topology config in effect, applying the env overrides existing sweep scripts rely on:
    /// `BASE_BLOWUP` → `base_log_blowup`, `RECURSION_SHARD_SHOTS` → `shots_per_shard` (`> 0`),
    /// `RECURSION_FOLD_ARITY` → `fold_arity` (clamped `>= 2`), `LEAF_BLOWUP` → `leaf_log_blowup`.
    /// `recursion_log_blowup` keeps its [`Default`] value (no env knob today). Unset /
    /// unparseable env vars fall back to the default, so with a clean environment this equals
    /// [`TopologyConfig::default`] exactly (in particular `fold_arity` stays 8 unless
    /// `RECURSION_FOLD_ARITY` is explicitly set, e.g. `=4` for the a2 sweep).
    pub fn from_env() -> Self {
        fn parse_env<T: std::str::FromStr>(k: &str) -> Option<T> {
            std::env::var(k).ok().and_then(|s| s.parse().ok())
        }
        let d = TopologyConfig::default();
        TopologyConfig {
            base_log_blowup: parse_env("BASE_BLOWUP").unwrap_or(d.base_log_blowup),
            recursion_log_blowup: d.recursion_log_blowup,
            leaf_log_blowup: parse_env("LEAF_BLOWUP").unwrap_or(d.leaf_log_blowup),
            fold_arity: parse_env::<usize>("RECURSION_FOLD_ARITY")
                .unwrap_or(d.fold_arity)
                .max(2),
            shots_per_shard: parse_env::<usize>("RECURSION_SHARD_SHOTS")
                .filter(|&n| n > 0)
                .unwrap_or(d.shots_per_shard),
        }
    }
}
