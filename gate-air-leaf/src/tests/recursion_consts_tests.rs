//! PINNED recursion-const capture + DRIFT tests (box-only, `#[ignore]`d). `capture_all` recomputes
//! the whole pinned table FRESH from the fixtures in one pass (emitting `@@` lines the generator
//! parses); the per-operating-point drift tests rebuild the real leaf preprocessed circuit and call
//! `check_configs`, which runs the SAME fresh cascade and asserts each layer/arity equals the pinned
//! table (`op.pinned().to_derived(..)`). One test per point (3); node_target is a field checked
//! inside `check_configs`.
//!
//! No env flags: `#[ignore]` is the laptop guard (these build real ~2^22 node preprocessed circuits —
//! GBs, box-only). The per-k fixture is resolved from `CARGO_MANIFEST_DIR/../../grover-tax/fixtures`.
//!
//! Run e.g.: `cargo test --release k500_drift -- --ignored --nocapture --test-threads=1`.

use super::*;

use circuit_common::finalize::{compute_padded_sizes, pad_to_targets, ComponentSizes};
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_verifier::verify::CircuitConfig;
use circuits::blake::HashValue;
use circuits::ivalue::NoValue;
use circuits_stark_verifier::proof::{empty_proof, ProofConfig};
use recursion_consts::OperatingPoint;
use recursive_aggregate::pinned_configs::{assemble_aggregate_config, DerivedConfigs};
use recursive_aggregate::test_utils::derive_configs;
use recursive_aggregate::AggregateConfig;

use leaf::{build_gate_air_leaf_circuit, GateAirLeafParams};

/// COMPUTES (not pins) the gate_air `AggregateConfig` for a fixture — a THIN wrapper over the shared
/// `derive_configs` + `assemble_aggregate_config`, for the tiny non-pinned wiring roundtrips in
/// `main`'s `mod tests` (a tiny fixture is not a pinned point). Builds the leaf preprocessed circuit
/// (padded to its OWN natural target, decoupled from the node target), derives the full cascade, and
/// assembles the runtime config. The `n` fed to `derive_configs` only affects its (discarded)
/// unpacker field — the returned `AggregateConfig` carries no unpacker — so a placeholder `n = 1`
/// is used; callers recompute the real per-N unpacker via `unpacker_verify_config`. NOT on the
/// production path — production reads `leaf::pinned_aggregate_config`.
pub(super) fn derive_aggregate_config(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    fold_arity: usize,
    // Node (level1-/fold-node/root) FRI blowup.
    log_blowup_factor: u32,
    // Leaf-wrap FRI blowup, decoupled from the node blowup (equal to it unless `LEAF_BLOWUP` is set).
    leaf_log_blowup: u32,
) -> AggregateConfig {
    let (leaf_pp, leaf_pcs_cfg, leaf_target) = build_leaf_pp(cfg, params, leaf_log_blowup);
    let derived = derive_configs(
        &leaf_pp,
        leaf_pcs_cfg,
        leaf_log_blowup,
        log_blowup_factor,
        fold_arity,
        1,
    );
    assemble_aggregate_config(&derived, leaf_target, fold_arity)
}

/// Builds the leaf preprocessed circuit padded to its OWN natural target (the leaf↔node padding
/// decoupling), returning it with its PCS and that natural target.
fn build_leaf_pp(
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    leaf_log_blowup: u32,
) -> (
    PreprocessedCircuit,
    stwo::core::pcs::PcsConfig,
    ComponentSizes,
) {
    let mut leaf_ctx = build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params);
    let leaf_target = compute_padded_sizes(&leaf_ctx);
    pad_to_targets(&mut leaf_ctx, leaf_target.clone());
    let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
    let leaf_pcs_cfg = leaf::leaf_pcs_config(leaf_pp.trace_log_size, leaf_log_blowup);
    (leaf_pp, leaf_pcs_cfg, leaf_target)
}

/// The 3 pinned Tanuj curve points: `(operating point, k, shots_per_shard)`.
const POINTS: [(OperatingPoint, usize, usize); 3] = [
    (OperatingPoint::K500N174, 500, 52),
    (OperatingPoint::K1000N348, 1000, 26),
    (OperatingPoint::K2000N695, 2000, 13),
];

/// The k-fixture directory: `CARGO_MANIFEST_DIR/../../grover-tax/fixtures`.
fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../grover-tax/fixtures")
}

/// Loads this point's k-matching fixture and builds the leaf circuit's public shape params (real
/// shard-0 shape at production `RC_LOG`). Each point needs its OWN k-matching fixture: `build_rows`
/// re-simulates the circuit and checks the final qubit state against the fixture's `y`.
fn leaf_shape(k: usize, shots: usize) -> (ProofConfig, GateAirLeafParams) {
    use circuit_statement::gate_air_components;
    use circuits::wrappers::U32Wrapper;

    let fx_path = fixtures_dir().join(format!("v0.3-iadd256-k{k}-n9024.json"));
    let fixture: Fixture = {
        let f = File::open(&fx_path).unwrap_or_else(|e| panic!("open {fx_path:?}: {e}"));
        serde_json::from_reader(f).expect("parse fixture")
    };
    let gates = parse_gtv1(&fixture.circuit_byte_serialisation_hex).expect("parse gtv1");
    let n_gates = gates.len();
    let cases: Vec<TestCase> = fixture.test_cases[..shots].to_vec();
    let (_rows, boundary, program, _padded, log_n_rows, _mls, base0_config) =
        shard0_shape(&gates, &cases, k, RC_LOG);
    let cfg = ProofConfig::new(
        &gate_air_components::<NoValue>(),
        N_PREPROCESSED_COLS,
        &base0_config,
        INTERACTION_POW_BITS,
    );
    let boundary_xy: Vec<([u32; N_LIMBS], [u32; N_LIMBS])> = cases
        .iter()
        .map(|c| {
            let x = state_to_limbs(&hex::decode(&c.x_hex).unwrap());
            let y = state_to_limbs(&hex::decode(&c.y_hex).unwrap());
            (x, y)
        })
        .collect();
    let placeholder_root: HashValue<SecureField> = HashValue(std::array::from_fn(|_| {
        U32Wrapper::new_unsafe(SecureField::zero())
    }));
    let params = GateAirLeafParams {
        main_log_size: log_n_rows,
        program_log_size: program.log_size,
        boundary_log_size: boundary.log_size,
        rc_log: RC_LOG,
        preprocessed_root: placeholder_root,
        boundary: boundary_xy,
        total_pc: (k * n_gates) as u32,
        program: program_rows_from_table(&program),
        nonce: hiding_nonce(),
    };
    (cfg, params)
}

/// Renders a `HashValue<QM31>`'s eight raw words for a paste-able `[u32; 8]` literal.
fn hv_words(h: &HashValue<SecureField>) -> [u32; 8] {
    std::array::from_fn(|i| {
        let [lo, hi, 0, 0] = h[i].get().to_m31_array().map(|m| m.0) else {
            return 0;
        };
        lo | (hi << 16)
    })
}

/// Runs the whole fresh cascade for one point and asserts it equals the pinned `PinnedConfigs` table
/// (via `check_configs`, which re-derives + per-field/per-arity `assert_eq`). Builds the real
/// ~2^22 node preprocessed circuits, so box-only.
fn drift(op: OperatingPoint, k: usize, shots: usize) {
    let topo = TopologyConfig::default();
    let (cfg, params) = leaf_shape(k, shots);
    let (leaf_pp, leaf_pcs_cfg, _leaf_target) = build_leaf_pp(&cfg, &params, topo.leaf_log_blowup);
    let expected = op
        .pinned()
        .to_derived(topo.leaf_log_blowup, topo.recursion_log_blowup);
    recursive_aggregate::test_utils::check_configs(
        &leaf_pp,
        leaf_pcs_cfg,
        topo.leaf_log_blowup,
        topo.recursion_log_blowup,
        topo.fold_arity,
        op.n(),
        &expected,
    );
}

#[test]
#[ignore = "box-only: builds the real gate_air leaf + ~2^22 node preprocessed circuits"]
fn k500_drift() {
    drift(OperatingPoint::K500N174, 500, 52);
}

#[test]
#[ignore = "box-only: builds the real gate_air leaf + ~2^22 node preprocessed circuits"]
fn k1000_drift() {
    drift(OperatingPoint::K1000N348, 1000, 26);
}

#[test]
#[ignore = "box-only: builds the real gate_air leaf + ~2^22 node preprocessed circuits"]
fn k2000_drift() {
    drift(OperatingPoint::K2000N695, 2000, 13);
}

/// SINGLE-PASS capture of the ENTIRE pinned-const table (all 3 points, all layers) — the from-scratch
/// capture tool feeding `scripts/gen_recursion_consts.py`. Recomputes every layer FRESH from the
/// fixture in one pass via the shared `derive_configs`, so a whole-table (re)capture is a single box
/// run. Emits `@@`-prefixed lines the generator parses. Run:
/// `cargo test --release --features cuda capture_all -- --ignored --nocapture`.
#[test]
#[ignore = "box-only: single-pass full-table capture (real per-point cascade)"]
fn capture_all() {
    let topo = TopologyConfig::default();
    for (op, k, shots) in POINTS {
        println!("@@POINT {op:?}");
        let (cfg, params) = leaf_shape(k, shots);
        let (leaf_pp, leaf_pcs_cfg, _leaf_target) =
            build_leaf_pp(&cfg, &params, topo.leaf_log_blowup);
        let derived = derive_configs(
            &leaf_pp,
            leaf_pcs_cfg,
            topo.leaf_log_blowup,
            topo.recursion_log_blowup,
            topo.fold_arity,
            op.n(),
        );
        emit_point(&derived, topo.recursion_log_blowup);
    }
    println!("@@CAPTURE_DONE");
}

/// Emits every `@@` line for one point's [`DerivedConfigs`] (leaf shape/root, node_target, per-arity
/// level1/fold shape+roots, the unpacker) — the format `gen_recursion_consts.py` parses. The layer
/// shape (`@@{tag}_TRACE`/`@@{tag}_COLS`) is emitted once per node layer (arity 2), roots per arity.
fn emit_point(d: &DerivedConfigs, node_blowup: u32) {
    let leaf_root = HashValue::from(hv_words(&d.leaf.preprocessed_root));
    emit_layer("LEAF", trace_of(&d.leaf, node_blowup), &d.leaf);
    emit_root("LEAF_ROOT", &leaf_root);

    let nt = &d.node_target;
    println!(
        "@@NODE_TARGET {} {} {} {} {}",
        nt.eq, nt.qm31_ops, nt.m31_to_u32, nt.triple_xor, nt.blake_g_gate
    );

    for (tag, layer) in [("LEVEL1", &d.level1), ("FOLD", &d.fold)] {
        for (i, cfg) in layer.iter().enumerate() {
            let arity = i + 2;
            if arity == 2 {
                emit_layer(tag, trace_of(cfg, node_blowup), cfg);
            }
            emit_root(&format!("{tag}_ROOT_{arity}"), &cfg.preprocessed_root);
        }
    }

    let u = &d.unpacker;
    let p = &u.config;
    println!(
        "@@UNPACKER_PCS {} {} {} {} {} {}",
        p.pow_bits,
        p.fri_config.log_blowup_factor,
        p.fri_config.log_last_layer_degree_bound,
        p.fri_config.n_queries,
        p.fri_config.fold_step,
        p.lifting_log_size.expect("unpacker lifting"),
    );
    println!("@@UNPACKER_NOUT {}", u.n_outputs);
    println!("@@UNPACKER_COLS {}", cols_str(u));
    emit_root("UNPACKER_ROOT", &u.preprocessed_root);
}

/// A config's trace log-size = its PCS lifting minus the FRI blowup.
fn trace_of(cfg: &CircuitConfig, blowup: u32) -> u32 {
    cfg.config.lifting_log_size.expect("lifting") - blowup
}

/// Emits a layer's `@@{tag}_TRACE` + `@@{tag}_COLS` lines for the fill generator.
fn emit_layer(tag: &str, trace_log_size: u32, cfg: &CircuitConfig) {
    println!("@@{tag}_TRACE {trace_log_size}");
    println!("@@{tag}_COLS {}", cols_str(cfg));
}

/// Space-separated `id:log_size` preprocessed-column pairs, in canonical committed order.
fn cols_str(cfg: &CircuitConfig) -> String {
    cfg.preprocessed_column_log_sizes
        .iter()
        .map(|(id, ls)| format!("{}:{}", id.id, ls))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Emits `@@{tag} w0 w1 .. w7` for an eight-word root.
fn emit_root(tag: &str, root: &HashValue<SecureField>) {
    let words: Vec<String> = hv_words(root).iter().map(|w| w.to_string()).collect();
    println!("@@{tag} {}", words.join(" "));
}
