//! Free topology parameters and the env-var knob layer for the recursion + base-proof pipeline.
//! [`TopologyConfig::from_env`] is the binary's sole env-parsing site; `recursive_aggregate` consumes
//! the resulting config as pure derived config.

/// Fold arity `k`: each internal node verifies exactly this many children (`k`-to-1 fold). The single
/// source of truth for the arity across the pipeline (fold, unpacker preimage, in-circuit node hash),
/// so out-of-circuit and in-circuit agree. A level's `len() % FOLD_ARITY` remainder is carried up
/// unchanged, so nodes are always exactly `k` children.
pub const FOLD_ARITY: usize = 8;

/// Default recursion (node-node / root) FRI blowup factor. Feeds the ~96-bit-secure `(pow_bits,
/// n_queries)` table via `get_pcs_config`.
pub const RECURSION_LOG_BLOWUP: u32 = 3;

/// Default base (shard) proof FRI blowup factor; derives the ~96-bit config via `leaf_pcs_config`.
pub const BASE_LOG_BLOWUP: u32 = 1;

/// Default shots per base shard (partition knob). `n_shards = ceil(samples / shots_per_shard)`.
pub const SHOTS_PER_SHARD: usize = 2;

/// All FREE topology parameters. Every field is a free knob; everything else (96-bit `(pow_bits,
/// n_queries)`, trusted roots, PCS/padding targets, `n_shards`, base-shard trace log) is DERIVED.
/// Security params (`fold_step`, `log_last_layer`, the 96-bit floor, `INTERACTION_POW_BITS`) are
/// pinned, deliberately NOT exposed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyConfig {
    /// Base (shard) proof FRI blowup factor. Default [`BASE_LOG_BLOWUP`] (env `BASE_BLOWUP`).
    pub base_log_blowup: u32,
    /// Recursion (node-node / root) FRI blowup factor. Default [`RECURSION_LOG_BLOWUP`].
    pub recursion_log_blowup: u32,
    /// Leaf-wrap FRI blowup factor (env `LEAF_BLOWUP`). Default [`RECURSION_LOG_BLOWUP`]. Decoupled
    /// from the node blowup so the leaf-wrap trace/lift (scales with shots/shard) tunes independently.
    pub leaf_log_blowup: u32,
    /// Node-node fold arity `k` (each internal fold-node verifies exactly `k` children). Default
    /// [`FOLD_ARITY`].
    pub fold_arity: usize,
    /// Shots per base shard (partition knob). Default [`SHOTS_PER_SHARD`] (env `RECURSION_SHARD_SHOTS`).
    pub shots_per_shard: usize,
}

impl Default for TopologyConfig {
    /// Production values (standalone-leaf → level1-node → shared fold-node up-tree fold).
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
    /// The topology config with sweep-script env overrides applied: `BASE_BLOWUP`, `LEAF_BLOWUP`,
    /// `RECURSION_SHARD_SHOTS` (`> 0`), `RECURSION_FOLD_ARITY` (clamped `>= 2`). `recursion_log_blowup`
    /// has no env knob. Unset/unparseable falls back to the default, so a clean environment equals
    /// [`TopologyConfig::default`] exactly.
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
