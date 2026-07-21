//! PINNED recursion constants, keyed PER OPERATING POINT (the 3 Tanuj curve points). The
//! leaf/level1/fold preprocessed roots + the per-N unpacker verify-config are FIXED functions of the
//! public base-shard shape (which differs per curve point), so they are pinned here rather than
//! recomputed at prove/verify time. `gate-air-leaf` assembles the `AggregateConfig` root table and the
//! trusted verifier's unpacker config from these consts; `recursive_aggregate` stays generic.
//!
//! !!! PLACEHOLDER VALUES !!! Every root / config field below is a PLACEHOLDER (all-zero roots, empty
//! shapes) marked `// PLACEHOLDER — capture on box`. The real values are captured ONCE on the box via
//! the `#[ignore]`d `recursion_consts_capture` drift test (see `main.rs`), which rebuilds the real
//! config per operating point and prints paste-able literals. Until captured, any real gate_air run at
//! a curve point will assert-fail at tree build (root mismatch) — expected.
//!
//! The 3 curve operating points (samples=9024, RC_LOG=25, base_blowup=1, fold_arity=8, node/leaf
//! blowup=3):
//!   - k=500,  shots_per_shard=52 → N=174
//!   - k=1000, shots_per_shard=26 → N=348
//!   - k=2000, shots_per_shard=13 → N=695
//!
//! Any other operating point (other k / shots / N / arity / blowup) → panic (unsupported).

use circuit_verifier::verify::CircuitConfig;
use circuits::blake::HashValue;
use circuits_stark_verifier::order_hash_map::OrderedHashMap;
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::topology::FOLD_ARITY;

/// The 3 pinned Tanuj curve operating points. Selected from the public `(k, shots_per_shard)` (which
/// fix `N` and the base-shard shape). Any other point is unsupported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatingPoint {
    /// k=500, shots_per_shard=52, N=174.
    K500N174,
    /// k=1000, shots_per_shard=26, N=348.
    K1000N348,
    /// k=2000, shots_per_shard=13, N=695.
    K2000N695,
}

/// The pinned per-arity roots + unpacker config for ONE operating point. Roots are the eight raw
/// preprocessed-tree words (`HashValue::from`); `level1`/`fold` are indexed by arity `2..=FOLD_ARITY`
/// (slot `arity - 2`).
struct PointConsts {
    /// PLACEHOLDER — capture on box. Leaf circuit's preprocessed root.
    leaf: [u32; 8],
    /// PLACEHOLDER — capture on box. Level1 (leaf-verifying) node root per arity `2..=FOLD_ARITY`.
    level1: [[u32; 8]; FOLD_ARITY - 1],
    /// PLACEHOLDER — capture on box. Fold (node-verifying) node root per arity `2..=FOLD_ARITY`.
    fold: [[u32; 8]; FOLD_ARITY - 1],
    /// PLACEHOLDER — capture on box. The trusted per-N unpacker verify-config.
    unpacker: UnpackerConfigConst,
}

/// The pinned unpacker [`CircuitConfig`] fields for one operating point (all captured on box).
struct UnpackerConfigConst {
    /// PLACEHOLDER — capture on box. `pcs_config`.
    pcs: PcsConfig,
    /// PLACEHOLDER — capture on box. `n_outputs` (= N * N_RESERVED).
    n_outputs: usize,
    /// PLACEHOLDER — capture on box. `(preprocessed column id, log_size)` pairs, in canonical order.
    preprocessed_column_log_sizes: &'static [(&'static str, u32)],
    /// PLACEHOLDER — capture on box. `preprocessed_root` (the canonical unpacker root).
    root: [u32; 8],
}

// ---- PLACEHOLDER const tables (all-zero roots / empty shapes) — capture on box. ----

/// A placeholder PCS config (never valid at runtime; overwritten by the captured literals). Marked so
/// the capture test's printed literal drops straight in.
const PLACEHOLDER_PCS: PcsConfig = PcsConfig {
    pow_bits: 0, // PLACEHOLDER — capture on box
    fri_config: FriConfig {
        log_blowup_factor: 0, // PLACEHOLDER — capture on box
        log_last_layer_degree_bound: 0,
        n_queries: 0, // PLACEHOLDER — capture on box
        fold_step: 0, // PLACEHOLDER — capture on box
    },
    lifting_log_size: None, // PLACEHOLDER — capture on box
};

const PLACEHOLDER_UNPACKER: UnpackerConfigConst = UnpackerConfigConst {
    pcs: PLACEHOLDER_PCS,
    n_outputs: 0,                       // PLACEHOLDER — capture on box
    preprocessed_column_log_sizes: &[], // PLACEHOLDER — capture on box
    root: [0; 8],                       // PLACEHOLDER — capture on box
};

const K500N174_UNPACKER: UnpackerConfigConst = UnpackerConfigConst {
    pcs: PcsConfig {
        pow_bits: 27,
        fri_config: FriConfig {
            log_blowup_factor: 3,
            log_last_layer_degree_bound: 0,
            n_queries: 23,
            fold_step: 4,
        },
        lifting_log_size: Some(23),
    },
    n_outputs: 1392,
    preprocessed_column_log_sizes: &[
        ("bitwise_xor_4_0", 8),
        ("bitwise_xor_4_1", 8),
        ("bitwise_xor_4_2", 8),
        ("eq_in0_address", 14),
        ("eq_in1_address", 14),
        ("bitwise_xor_7_0", 14),
        ("bitwise_xor_7_1", 14),
        ("bitwise_xor_7_2", 14),
        ("triple_xor_input_addr_0", 16),
        ("triple_xor_input_addr_1", 16),
        ("triple_xor_input_addr_2", 16),
        ("triple_xor_output_addr", 16),
        ("triple_xor_multiplicity", 16),
        ("seq_16", 16),
        ("bitwise_xor_8_0", 16),
        ("bitwise_xor_8_1", 16),
        ("bitwise_xor_8_2", 16),
        ("m31_to_u32_input_addr", 17),
        ("m31_to_u32_output_addr", 17),
        ("m31_to_u32_multiplicity", 17),
        ("bitwise_xor_9_0", 18),
        ("bitwise_xor_9_1", 18),
        ("bitwise_xor_9_2", 18),
        ("qm31_ops_add_flag", 19),
        ("qm31_ops_sub_flag", 19),
        ("qm31_ops_mul_flag", 19),
        ("qm31_ops_pointwise_mul_flag", 19),
        ("qm31_ops_in0_address", 19),
        ("qm31_ops_in1_address", 19),
        ("qm31_ops_out_address", 19),
        ("qm31_ops_mults", 19),
        ("blake_g_gate_input_addr_a", 19),
        ("blake_g_gate_input_addr_b", 19),
        ("blake_g_gate_input_addr_c", 19),
        ("blake_g_gate_input_addr_d", 19),
        ("blake_g_gate_input_addr_f0", 19),
        ("blake_g_gate_input_addr_f1", 19),
        ("blake_g_gate_output_addr_a", 19),
        ("blake_g_gate_output_addr_b", 19),
        ("blake_g_gate_output_addr_c", 19),
        ("blake_g_gate_output_addr_d", 19),
        ("blake_g_gate_multiplicity", 19),
        ("bitwise_xor_10_0", 20),
        ("bitwise_xor_10_1", 20),
        ("bitwise_xor_10_2", 20),
    ],
    root: [
        1458995347, 1562440200, 1202961514, 2987559716, 1526432770, 3682326847, 1943084383,
        2712873628,
    ],
};

const K500N174_CONSTS: PointConsts = PointConsts {
    leaf: [
        793966613, 3148827868, 371244120, 1131643778, 2398270055, 3940816135, 3922883541,
        3837070059,
    ],
    level1: [
        [
            3913237992, 1328931292, 3600783642, 4160425405, 2292402306, 1372795367, 3344825380,
            3307260737,
        ],
        [
            794020276, 1138354557, 2342314457, 2738058620, 836541475, 572545005, 2133473276,
            3732366911,
        ],
        [
            3981184752, 2869148245, 4050452879, 1549695810, 1413560986, 1526183906, 3280509150,
            3056814724,
        ],
        [
            1772291197, 3779180227, 1829603982, 1960869279, 572129748, 4004918105, 3172614091,
            1985563048,
        ],
        [
            2933855741, 1000136023, 76643639, 3340313634, 2460463927, 2514012950, 2141552836,
            3742645957,
        ],
        [
            3714377982, 4143563905, 2695071820, 3749142775, 1753261140, 1751655539, 1139804928,
            3047062049,
        ],
        [
            348310115, 2823893169, 3301704946, 2792065221, 3585730242, 3904259419, 3672672604,
            3724957741,
        ],
    ],
    fold: [
        [
            2035091519, 501705985, 166047172, 1123045752, 3568305265, 815574077, 3278034143,
            243510273,
        ],
        [
            1205184393, 1768923835, 1951434286, 589634634, 1735120075, 2299869867, 1464856437,
            2015570009,
        ],
        [
            697049043, 4174700279, 879818794, 100854944, 3626372037, 2285919184, 1035665903,
            640249968,
        ],
        [
            3815544182, 1429541617, 1568851281, 503105286, 4267192809, 4058400151, 1111718947,
            579912664,
        ],
        [
            4078638528, 3540322400, 1438635665, 1114065282, 1562653426, 1810740466, 1059032829,
            4099962379,
        ],
        [
            1954138946, 1856729758, 1748132318, 875692421, 2985527280, 2727902775, 397876993,
            3419958345,
        ],
        [
            821842364, 2771518357, 3999513413, 3904203972, 436240095, 516791821, 817970496,
            1661073348,
        ],
    ],
    unpacker: K500N174_UNPACKER,
};

const K1000N348_CONSTS: PointConsts = PointConsts {
    leaf: [0; 8],                     // PLACEHOLDER — capture on box
    level1: [[0; 8]; FOLD_ARITY - 1], // PLACEHOLDER — capture on box
    fold: [[0; 8]; FOLD_ARITY - 1],   // PLACEHOLDER — capture on box
    unpacker: PLACEHOLDER_UNPACKER,
};

const K2000N695_CONSTS: PointConsts = PointConsts {
    leaf: [0; 8],                     // PLACEHOLDER — capture on box
    level1: [[0; 8]; FOLD_ARITY - 1], // PLACEHOLDER — capture on box
    fold: [[0; 8]; FOLD_ARITY - 1],   // PLACEHOLDER — capture on box
    unpacker: PLACEHOLDER_UNPACKER,
};

impl OperatingPoint {
    /// Selects the operating point for the public `(k, shots_per_shard)`. Panics for any other point
    /// (unsupported): only the 3 pinned Tanuj curve points are proved.
    pub fn from_params(k: usize, shots_per_shard: usize) -> Self {
        match (k, shots_per_shard) {
            (500, 52) => OperatingPoint::K500N174,
            (1000, 26) => OperatingPoint::K1000N348,
            (2000, 13) => OperatingPoint::K2000N695,
            _ => panic!(
                "unsupported operating point (k={k}, shots_per_shard={shots_per_shard}); \
                 only the 3 pinned Tanuj curve points are supported"
            ),
        }
    }

    /// Number of shards `N` for this point (leaf count).
    pub fn n(self) -> usize {
        match self {
            OperatingPoint::K500N174 => 174,
            OperatingPoint::K1000N348 => 348,
            OperatingPoint::K2000N695 => 695,
        }
    }

    fn consts(self) -> &'static PointConsts {
        match self {
            OperatingPoint::K500N174 => &K500N174_CONSTS,
            OperatingPoint::K1000N348 => &K1000N348_CONSTS,
            OperatingPoint::K2000N695 => &K2000N695_CONSTS,
        }
    }

    /// The pinned leaf preprocessed root.
    pub fn leaf_root(self) -> HashValue<QM31> {
        HashValue::from(self.consts().leaf)
    }

    /// The pinned level1 (leaf-verifying) node root for `arity` (`2..=FOLD_ARITY`). Panics otherwise.
    pub fn level1_root(self, arity: usize) -> HashValue<QM31> {
        HashValue::from(self.consts().level1[arity_slot(arity)])
    }

    /// The pinned fold (node-verifying) node root for `arity` (`2..=FOLD_ARITY`). Panics otherwise.
    pub fn fold_root(self, arity: usize) -> HashValue<QM31> {
        HashValue::from(self.consts().fold[arity_slot(arity)])
    }

    /// The pinned per-N unpacker verify-config for `n` leaves. Panics if `n` != this point's `N`.
    pub fn unpacker_config(self, n: usize) -> CircuitConfig {
        assert_eq!(
            n,
            self.n(),
            "unpacker config requested for n={n} but this operating point has N={}",
            self.n()
        );
        let u = &self.consts().unpacker;
        CircuitConfig {
            config: u.pcs,
            n_outputs: u.n_outputs,
            preprocessed_column_log_sizes: u
                .preprocessed_column_log_sizes
                .iter()
                .map(|(id, log_size)| {
                    (
                        PreProcessedColumnId {
                            id: (*id).to_owned(),
                        },
                        *log_size,
                    )
                })
                .collect::<OrderedHashMap<PreProcessedColumnId, u32>>(),
            preprocessed_root: HashValue::from(u.root),
        }
    }
}

/// The slot index for `arity` in a `[_; FOLD_ARITY - 1]` per-arity table. Panics for arity outside
/// `2..=FOLD_ARITY` (unsupported).
fn arity_slot(arity: usize) -> usize {
    assert!(
        (2..=FOLD_ARITY).contains(&arity),
        "unsupported arity {arity} (must be 2..={FOLD_ARITY})"
    );
    arity - 2
}
