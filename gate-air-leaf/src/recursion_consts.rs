//! PINNED recursion VERIFIER configs, keyed PER OPERATING POINT (the 3 Tanuj curve points). Every
//! config the recursion verifies against is a fixed constant here — leaf / level1 / fold / unpacker —
//! so nothing security-relevant (n_queries / pow_bits / blowup / fold_step / lifting / column shapes /
//! preprocessed roots) is derived at prove or verify time. `gate-air-leaf` assembles the
//! `AggregateConfig` (leaf/node shared configs, PCS, node_target, roots) and the trusted verifier's
//! unpacker config from these consts; `recursive_aggregate` stays generic.
//!
//! The genuinely-COMPUTED fields pinned here (vs the fixed-by-blowup scalars) are, per (kind, point):
//! `trace_log_size` (→ the PCS `lifting_log_size`), `preprocessed_column_log_sizes`, `node_target`
//! (`ComponentSizes`, shared by level1 + fold), and the per-(kind, arity) `preprocessed_root`. The PCS
//! scalars (`pow_bits`, `n_queries`, `fold_step`, `log_last_layer_degree_bound`, `log_blowup_factor`)
//! and `n_outputs` are FIXED functions of the fixed blowups (see [`leaf_pcs`] / [`node_pcs`]).
//!
//! !!! PLACEHOLDER VALUES !!! Most fields below (all roots, all `trace_log_size`/`cols`, `node_target`,
//! the k=1000/k=2000 unpackers) are PLACEHOLDERs marked `// PLACEHOLDER — capture on box`. The real
//! values are captured ONCE on the box via the `#[ignore]`d per-layer drift tests
//! (`recursion_consts_tests.rs`), which rebuild the real config per point and print paste-able
//! literals. Until captured, any real gate_air run at a curve point asserts-fail at tree build (root
//! mismatch) — expected. k=500's roots + unpacker are already filled.
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

use circuit_common::finalize::ComponentSizes;
use circuit_common::N_RESERVED;

use crate::topology::{FOLD_ARITY, RECURSION_LOG_BLOWUP};

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

/// The pinned full verifier config for ONE operating point: the leaf/level1/fold per-layer shapes +
/// roots + `node_target`, and the trusted per-N unpacker config. Roots are the eight raw
/// preprocessed-tree words (`HashValue::from`); `level1`/`fold` roots are indexed by arity
/// `2..=FOLD_ARITY` (slot `arity - 2`).
struct PointConsts {
    /// The leaf verifier shape (single; leaves share one AIR).
    leaf: LayerShape,
    /// The level-1 (leaf-verifying) node shape + per-arity roots.
    level1: NodeLayer,
    /// The fold (node-verifying) node shape + per-arity roots.
    fold: NodeLayer,
    /// The common padding target every level1/fold node pads to (cross-child uniformity).
    node_target: ComponentSizes,
    /// The trusted per-N unpacker verify-config.
    unpacker: UnpackerConfigConst,
}

/// One verifier layer's computed shape: the trace log-size (→ PCS lifting) + preprocessed columns,
/// plus the single preprocessed root (used by the leaf, whose root is not per-arity).
struct LayerShape {
    /// PLACEHOLDER — capture on box. `trace_log_size` (the padded preprocessed trace log-size).
    trace_log_size: u32,
    /// PLACEHOLDER — capture on box. `(preprocessed column id, log_size)` pairs, in canonical order.
    preprocessed_column_log_sizes: &'static [(&'static str, u32)],
    /// PLACEHOLDER — capture on box. `preprocessed_root`.
    root: [u32; 8],
}

/// A node verifier layer (level1 or fold): the layer's own shape (`trace_log_size` + columns) plus a
/// per-arity `preprocessed_root` (`2..=FOLD_ARITY`). All arities pad to the point's `node_target`, so
/// the shape is shared across arities; only the root content differs per arity.
struct NodeLayer {
    /// PLACEHOLDER — capture on box. `trace_log_size` (the padded preprocessed trace log-size).
    trace_log_size: u32,
    /// PLACEHOLDER — capture on box. `(preprocessed column id, log_size)` pairs, in canonical order.
    preprocessed_column_log_sizes: &'static [(&'static str, u32)],
    /// PLACEHOLDER — capture on box. `preprocessed_root` per arity `2..=FOLD_ARITY` (slot `arity-2`).
    roots: [[u32; 8]; FOLD_ARITY - 1],
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

/// Leaf PCS: the fixed config for the leaf FRI blowup, `lifting_log_size` pinned to the leaf's
/// `trace_log_size`. Mirrors stwo-circuits' `get_pcs_config`; the `(pow_bits, n_queries)` are fixed
/// by the blowup (96-bit secure), so only `lifting_log_size` is point-dependent. The pinned points
/// use the default leaf blowup (`= RECURSION_LOG_BLOWUP`); a `LEAF_BLOWUP` env override is not a
/// pinned point (its shape would differ and fail the tree-build root assert).
pub fn leaf_pcs(trace_log_size: u32) -> PcsConfig {
    pcs_for_blowup(trace_log_size, RECURSION_LOG_BLOWUP)
}

/// Node PCS (level1 / fold / root): the fixed config for the node FRI blowup, `lifting_log_size`
/// pinned to the node's `trace_log_size`.
pub fn node_pcs(trace_log_size: u32) -> PcsConfig {
    pcs_for_blowup(trace_log_size, RECURSION_LOG_BLOWUP)
}

/// The fixed `(pow_bits, n_queries)` for `log_blowup_factor`, as a full [`PcsConfig`] with lifting
/// pinned to `trace_log_size + log_blowup_factor`. Panics for an unsupported blowup.
fn pcs_for_blowup(trace_log_size: u32, log_blowup_factor: u32) -> PcsConfig {
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

// <<GENERATED CONSTS BEGIN — this region is regenerated by scripts/gen_recursion_consts.py from the
// `capture_all` test output; the placeholders below hold until the first box capture. Do not hand-edit.>>

const K500N174_CONSTS: PointConsts = PointConsts {
    leaf: LayerShape {
        trace_log_size: 21,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("triple_xor_input_addr_0", 18),
            ("triple_xor_input_addr_1", 18),
            ("triple_xor_input_addr_2", 18),
            ("triple_xor_output_addr", 18),
            ("triple_xor_multiplicity", 18),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("m31_to_u32_input_addr", 19),
            ("m31_to_u32_output_addr", 19),
            ("m31_to_u32_multiplicity", 19),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 21),
            ("qm31_ops_sub_flag", 21),
            ("qm31_ops_mul_flag", 21),
            ("qm31_ops_pointwise_mul_flag", 21),
            ("qm31_ops_in0_address", 21),
            ("qm31_ops_in1_address", 21),
            ("qm31_ops_out_address", 21),
            ("qm31_ops_mults", 21),
            ("blake_g_gate_input_addr_a", 21),
            ("blake_g_gate_input_addr_b", 21),
            ("blake_g_gate_input_addr_c", 21),
            ("blake_g_gate_input_addr_d", 21),
            ("blake_g_gate_input_addr_f0", 21),
            ("blake_g_gate_input_addr_f1", 21),
            ("blake_g_gate_output_addr_a", 21),
            ("blake_g_gate_output_addr_b", 21),
            ("blake_g_gate_output_addr_c", 21),
            ("blake_g_gate_output_addr_d", 21),
            ("blake_g_gate_multiplicity", 21),
        ],
        root: [
            793966613, 3148827868, 371244120, 1131643778, 2398270055, 3940816135, 3922883541,
            3837070059,
        ],
    },
    level1: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
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
    },
    fold: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
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
    },
    node_target: ComponentSizes {
        eq: 131072,
        qm31_ops: 4194304,
        m31_to_u32: 1048576,
        triple_xor: 524288,
        blake_g_gate: 4194304,
    },
    unpacker: UnpackerConfigConst {
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
    },
};

const K1000N348_CONSTS: PointConsts = PointConsts {
    leaf: LayerShape {
        trace_log_size: 21,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("triple_xor_input_addr_0", 18),
            ("triple_xor_input_addr_1", 18),
            ("triple_xor_input_addr_2", 18),
            ("triple_xor_output_addr", 18),
            ("triple_xor_multiplicity", 18),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("m31_to_u32_input_addr", 19),
            ("m31_to_u32_output_addr", 19),
            ("m31_to_u32_multiplicity", 19),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 21),
            ("qm31_ops_sub_flag", 21),
            ("qm31_ops_mul_flag", 21),
            ("qm31_ops_pointwise_mul_flag", 21),
            ("qm31_ops_in0_address", 21),
            ("qm31_ops_in1_address", 21),
            ("qm31_ops_out_address", 21),
            ("qm31_ops_mults", 21),
            ("blake_g_gate_input_addr_a", 21),
            ("blake_g_gate_input_addr_b", 21),
            ("blake_g_gate_input_addr_c", 21),
            ("blake_g_gate_input_addr_d", 21),
            ("blake_g_gate_input_addr_f0", 21),
            ("blake_g_gate_input_addr_f1", 21),
            ("blake_g_gate_output_addr_a", 21),
            ("blake_g_gate_output_addr_b", 21),
            ("blake_g_gate_output_addr_c", 21),
            ("blake_g_gate_output_addr_d", 21),
            ("blake_g_gate_multiplicity", 21),
        ],
        root: [
            1850900959, 3293234304, 2928656745, 3090560660, 3612317591, 3430403496, 233314597,
            2442885335,
        ],
    },
    level1: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
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
    },
    fold: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
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
    },
    node_target: ComponentSizes {
        eq: 131072,
        qm31_ops: 4194304,
        m31_to_u32: 1048576,
        triple_xor: 524288,
        blake_g_gate: 4194304,
    },
    unpacker: UnpackerConfigConst {
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
        n_outputs: 2784,
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
            3728195187, 1170503658, 3767219659, 4041647410, 1441671135, 3615336523, 211715135,
            2181328451,
        ],
    },
};

const K2000N695_CONSTS: PointConsts = PointConsts {
    leaf: LayerShape {
        trace_log_size: 21,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("eq_in0_address", 16),
            ("eq_in1_address", 16),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("triple_xor_input_addr_0", 18),
            ("triple_xor_input_addr_1", 18),
            ("triple_xor_input_addr_2", 18),
            ("triple_xor_output_addr", 18),
            ("triple_xor_multiplicity", 18),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("m31_to_u32_input_addr", 19),
            ("m31_to_u32_output_addr", 19),
            ("m31_to_u32_multiplicity", 19),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 21),
            ("qm31_ops_sub_flag", 21),
            ("qm31_ops_mul_flag", 21),
            ("qm31_ops_pointwise_mul_flag", 21),
            ("qm31_ops_in0_address", 21),
            ("qm31_ops_in1_address", 21),
            ("qm31_ops_out_address", 21),
            ("qm31_ops_mults", 21),
            ("blake_g_gate_input_addr_a", 21),
            ("blake_g_gate_input_addr_b", 21),
            ("blake_g_gate_input_addr_c", 21),
            ("blake_g_gate_input_addr_d", 21),
            ("blake_g_gate_input_addr_f0", 21),
            ("blake_g_gate_input_addr_f1", 21),
            ("blake_g_gate_output_addr_a", 21),
            ("blake_g_gate_output_addr_b", 21),
            ("blake_g_gate_output_addr_c", 21),
            ("blake_g_gate_output_addr_d", 21),
            ("blake_g_gate_multiplicity", 21),
        ],
        root: [
            1671726266, 3079719153, 3369376636, 2214560819, 1590190948, 784830243, 1338047948,
            4269289402,
        ],
    },
    level1: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
            [
                1990614252, 494231903, 872668398, 1645072460, 3559864537, 1840174976, 1840493561,
                460461879,
            ],
            [
                1350136369, 613818072, 2389158623, 623563511, 762762034, 2941096340, 1306234309,
                781402198,
            ],
            [
                2826914180, 4169723737, 2593954969, 618739703, 3531112443, 1652306850, 3697541633,
                1374203952,
            ],
            [
                3535923161, 4048113930, 1160487281, 3545319893, 1011023218, 3048918468, 1762680830,
                1350636447,
            ],
            [
                2048057097, 2473932466, 1545952202, 94647821, 2688644939, 385238271, 686837108,
                3861452986,
            ],
            [
                3560555931, 2113360241, 2576210057, 3620827623, 2396287476, 2033049081, 3307246211,
                2834975683,
            ],
            [
                1066398071, 957931730, 2617037336, 4031838851, 626823415, 1208599359, 3889888060,
                3816771445,
            ],
        ],
    },
    fold: NodeLayer {
        trace_log_size: 22,
        preprocessed_column_log_sizes: &[
            ("bitwise_xor_4_0", 8),
            ("bitwise_xor_4_1", 8),
            ("bitwise_xor_4_2", 8),
            ("bitwise_xor_7_0", 14),
            ("bitwise_xor_7_1", 14),
            ("bitwise_xor_7_2", 14),
            ("seq_16", 16),
            ("bitwise_xor_8_0", 16),
            ("bitwise_xor_8_1", 16),
            ("bitwise_xor_8_2", 16),
            ("eq_in0_address", 17),
            ("eq_in1_address", 17),
            ("bitwise_xor_9_0", 18),
            ("bitwise_xor_9_1", 18),
            ("bitwise_xor_9_2", 18),
            ("triple_xor_input_addr_0", 19),
            ("triple_xor_input_addr_1", 19),
            ("triple_xor_input_addr_2", 19),
            ("triple_xor_output_addr", 19),
            ("triple_xor_multiplicity", 19),
            ("m31_to_u32_input_addr", 20),
            ("m31_to_u32_output_addr", 20),
            ("m31_to_u32_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
            ("qm31_ops_add_flag", 22),
            ("qm31_ops_sub_flag", 22),
            ("qm31_ops_mul_flag", 22),
            ("qm31_ops_pointwise_mul_flag", 22),
            ("qm31_ops_in0_address", 22),
            ("qm31_ops_in1_address", 22),
            ("qm31_ops_out_address", 22),
            ("qm31_ops_mults", 22),
            ("blake_g_gate_input_addr_a", 22),
            ("blake_g_gate_input_addr_b", 22),
            ("blake_g_gate_input_addr_c", 22),
            ("blake_g_gate_input_addr_d", 22),
            ("blake_g_gate_input_addr_f0", 22),
            ("blake_g_gate_input_addr_f1", 22),
            ("blake_g_gate_output_addr_a", 22),
            ("blake_g_gate_output_addr_b", 22),
            ("blake_g_gate_output_addr_c", 22),
            ("blake_g_gate_output_addr_d", 22),
            ("blake_g_gate_multiplicity", 22),
        ],
        roots: [
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
    },
    node_target: ComponentSizes {
        eq: 131072,
        qm31_ops: 4194304,
        m31_to_u32: 1048576,
        triple_xor: 524288,
        blake_g_gate: 4194304,
    },
    unpacker: UnpackerConfigConst {
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
        n_outputs: 5560,
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
            ("blake_g_gate_input_addr_a", 20),
            ("blake_g_gate_input_addr_b", 20),
            ("blake_g_gate_input_addr_c", 20),
            ("blake_g_gate_input_addr_d", 20),
            ("blake_g_gate_input_addr_f0", 20),
            ("blake_g_gate_input_addr_f1", 20),
            ("blake_g_gate_output_addr_a", 20),
            ("blake_g_gate_output_addr_b", 20),
            ("blake_g_gate_output_addr_c", 20),
            ("blake_g_gate_output_addr_d", 20),
            ("blake_g_gate_multiplicity", 20),
            ("bitwise_xor_10_0", 20),
            ("bitwise_xor_10_1", 20),
            ("bitwise_xor_10_2", 20),
        ],
        root: [
            3703613476, 3482821825, 2797341073, 3706339720, 760043064, 2712616523, 1917170430,
            2259292315,
        ],
    },
};

// <<GENERATED CONSTS END>>

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

    /// The pinned leaf verifier [`CircuitConfig`]: leaf PCS (lifting at the pinned leaf trace-log),
    /// the pinned leaf columns, and the pinned leaf preprocessed root.
    pub fn leaf_config(self) -> CircuitConfig {
        let l = &self.consts().leaf;
        circuit_config(
            leaf_pcs(l.trace_log_size),
            l.preprocessed_column_log_sizes,
            l.root,
        )
    }

    /// The pinned level1 (leaf-verifying) node verifier [`CircuitConfig`] for `arity` (`2..=FOLD_ARITY`):
    /// node PCS (lifting at the pinned level1 trace-log), the pinned level1 columns, and the arity's
    /// pinned root. Panics for arity outside `2..=FOLD_ARITY`.
    pub fn level1_config(self, arity: usize) -> CircuitConfig {
        let l = &self.consts().level1;
        circuit_config(
            node_pcs(l.trace_log_size),
            l.preprocessed_column_log_sizes,
            l.roots[arity_slot(arity)],
        )
    }

    /// The pinned fold (node-verifying) node verifier [`CircuitConfig`] for `arity` (`2..=FOLD_ARITY`):
    /// node PCS (lifting at the pinned fold trace-log), the pinned fold columns, and the arity's pinned
    /// root. Used only by the drift test (the prove path folds nodes against the level1 child config).
    #[cfg(test)]
    pub fn fold_config(self, arity: usize) -> CircuitConfig {
        let f = &self.consts().fold;
        circuit_config(
            node_pcs(f.trace_log_size),
            f.preprocessed_column_log_sizes,
            f.roots[arity_slot(arity)],
        )
    }

    /// The common `node_target` padding sizes every level1/fold node pads to.
    pub fn node_target(self) -> ComponentSizes {
        self.consts().node_target.clone()
    }

    /// The pinned leaf preprocessed root.
    pub fn leaf_root(self) -> HashValue<QM31> {
        HashValue::from(self.consts().leaf.root)
    }

    /// The pinned level1 (leaf-verifying) node root for `arity` (`2..=FOLD_ARITY`). Panics otherwise.
    pub fn level1_root(self, arity: usize) -> HashValue<QM31> {
        HashValue::from(self.consts().level1.roots[arity_slot(arity)])
    }

    /// The pinned fold (node-verifying) node root for `arity` (`2..=FOLD_ARITY`). Panics otherwise.
    pub fn fold_root(self, arity: usize) -> HashValue<QM31> {
        HashValue::from(self.consts().fold.roots[arity_slot(arity)])
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
            preprocessed_column_log_sizes: cols(u.preprocessed_column_log_sizes),
            preprocessed_root: HashValue::from(u.root),
        }
    }
}

/// Assembles a node/leaf verifier [`CircuitConfig`] from a pinned PCS + columns + root. `n_outputs`
/// is `N_RESERVED` (the reserved-output count every recursion circuit emits).
fn circuit_config(
    pcs: PcsConfig,
    preprocessed_column_log_sizes: &'static [(&'static str, u32)],
    root: [u32; 8],
) -> CircuitConfig {
    CircuitConfig {
        config: pcs,
        n_outputs: N_RESERVED,
        preprocessed_column_log_sizes: cols(preprocessed_column_log_sizes),
        preprocessed_root: HashValue::from(root),
    }
}

/// Builds an [`OrderedHashMap`] of preprocessed column id → log_size from the pinned literal pairs,
/// preserving their (canonical committed) order.
fn cols(pairs: &'static [(&'static str, u32)]) -> OrderedHashMap<PreProcessedColumnId, u32> {
    pairs
        .iter()
        .map(|(id, log_size)| {
            (
                PreProcessedColumnId {
                    id: (*id).to_owned(),
                },
                *log_size,
            )
        })
        .collect()
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
