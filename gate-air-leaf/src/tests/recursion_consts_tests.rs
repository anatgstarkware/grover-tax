//! PINNED recursion-const capture + per-layer DRIFT tests (box-only, `#[ignore]`d). Each test rebuilds
//! ONE verifier layer's real config and asserts it equals the pinned `recursion_consts` table, printing
//! paste-able literals so a fresh capture (or a stwo-rev drift) is a copy-paste. The layers are
//! INDEPENDENT: `level1_drift`/`fold_drift`/`unpacker_drift` feed the helper the PINNED child config
//! (not a rebuilt one), so a fold drift never rebuilds a level1 and a level1 drift never rebuilds the
//! leaf. `node_target_drift` is the single test that runs the fixed point (its output feeds the node
//! tests' pinned input).
//!
//! No env flags: `#[ignore]` is the laptop guard (these build real ~2^22 node preprocessed circuits —
//! GBs, box-only). The per-k fixture is resolved from `CARGO_MANIFEST_DIR/../../grover-tax/fixtures`.
//!
//! Capture (fresh table) order on the box, since the node tests read pinned inputs:
//!   1. `leaf_drift` + `node_target_drift`  → paste leaf shape + node_target.
//!   2. `level1_drift` + `fold_drift`        → paste level1/fold shapes + roots.
//!   3. `unpacker_drift`                     → paste the per-N unpacker config.
//!
//! Run e.g.: `cargo test --release leaf_drift -- --ignored --nocapture --test-threads=1`.

use super::*;

use circuit_common::finalize::{compute_padded_sizes, pad_to_targets, ComponentSizes};
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_verifier::verify::CircuitConfig;
use circuits::blake::HashValue;
use circuits::ivalue::NoValue;
use circuits_stark_verifier::proof::{empty_proof, ProofConfig};
use recursion_consts::{leaf_pcs, OperatingPoint};
use recursive_aggregate::{preprocessed_root, recompute_node};

use leaf::{build_gate_air_leaf_circuit, derive_aggregate_config, GateAirLeafParams};

use crate::topology::FOLD_ARITY;

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

/// Prints a captured layer shape (`trace_log_size` + preprocessed columns) as paste-able literals.
fn print_shape(label: &str, cfg: &CircuitConfig, lifting: u32) {
    println!("// ---- {label} shape (paste into recursion_consts.rs) ----");
    println!(
        "//   trace_log_size: {},",
        lifting - cfg.config.fri_config.log_blowup_factor
    );
    println!("//   preprocessed_column_log_sizes: &[");
    for (id, log_size) in cfg.preprocessed_column_log_sizes.iter() {
        println!("//       ({:?}, {}),", id.id, log_size);
    }
    println!("//   ],");
}

/// `leaf_drift` — gate_air-specific: build the leaf circuit, compare to the pinned leaf config/root.
#[test]
#[ignore = "box-only: builds the real gate_air leaf preprocessed circuit per point"]
fn leaf_drift() {
    let leaf_blowup = TopologyConfig::default().leaf_log_blowup;
    let mut drifts: Vec<String> = Vec::new();
    for (op, k, shots) in POINTS {
        let (cfg, params) = leaf_shape(k, shots);
        // Rebuild the leaf preprocessed circuit (gate_air-specific), padded to its own natural target.
        let mut leaf_ctx = build_gate_air_leaf_circuit::<NoValue>(empty_proof(&cfg), &cfg, &params);
        let target = compute_padded_sizes(&leaf_ctx);
        pad_to_targets(&mut leaf_ctx, target);
        let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
        let root = preprocessed_root(&leaf_pp, leaf_blowup);
        let real = CircuitConfig {
            config: leaf_pcs(leaf_pp.trace_log_size),
            n_outputs: circuit_common::N_RESERVED,
            preprocessed_column_log_sizes: leaf_pp.preprocessed_trace.log_sizes(),
            preprocessed_root: root.clone(),
        };
        print_shape(
            &format!("{op:?} leaf"),
            &real,
            leaf_pp.trace_log_size + leaf_blowup,
        );
        println!("//   root: {:?},", hv_words(&root));
        if real != op.leaf_config() {
            drifts.push(format!("{op:?} leaf config"));
        }
    }
    assert!(drifts.is_empty(), "leaf drift:\n  {}", drifts.join("\n  "));
}

/// `node_target_drift` — runs the fixed point (the ONE recompute of the cross-child padding target)
/// and compares to the pinned `node_target`.
#[test]
#[ignore = "box-only: runs the node-target fixed point (real ~2^22 node shapes)"]
fn node_target_drift() {
    let topo = TopologyConfig::default();
    let mut drifts: Vec<String> = Vec::new();
    for (op, k, shots) in POINTS {
        let (cfg, params) = leaf_shape(k, shots);
        let config = derive_aggregate_config(
            &cfg,
            &params,
            topo.fold_arity,
            topo.recursion_log_blowup,
            topo.leaf_log_blowup,
        );
        let real = config.node_target_padding_sizes;
        print_node_target(op, &real);
        if real != op.node_target() {
            drifts.push(format!("{op:?} node_target"));
        }
    }
    assert!(
        drifts.is_empty(),
        "node_target drift:\n  {}",
        drifts.join("\n  ")
    );
}

/// Prints a captured `ComponentSizes` node_target as a paste-able literal.
fn print_node_target(op: OperatingPoint, s: &ComponentSizes) {
    println!("// ---- {op:?} node_target (paste into recursion_consts.rs) ----");
    println!(
        "//   ComponentSizes {{ eq: {}, qm31_ops: {}, m31_to_u32: {}, triple_xor: {}, blake_g_gate: {} }},",
        s.eq, s.qm31_ops, s.m31_to_u32, s.triple_xor, s.blake_g_gate,
    );
}

/// `level1_drift` — feeds the helper the PINNED LEAF config + PINNED node_target (no leaf rebuild),
/// recomputes each arity's level1 (leaf-verifying) node, and compares to the pinned level1 config/root.
#[test]
#[ignore = "box-only: builds the real level1 node preprocessed circuits (~2^22)"]
fn level1_drift() {
    let node_blowup = TopologyConfig::default().recursion_log_blowup;
    let mut drifts: Vec<String> = Vec::new();
    for (op, _k, _shots) in POINTS {
        let child = op.leaf_config();
        let node_target = op.node_target();
        for arity in 2..=FOLD_ARITY {
            let (real, root) = recompute_node(&child, arity, node_target.clone(), node_blowup);
            if arity == 2 {
                let lifting = real.config.lifting_log_size.expect("node lifting set");
                print_shape(&format!("{op:?} level1"), &real, lifting);
            }
            println!("// {op:?} level1[{arity}] root: {:?}", hv_words(&root));
            if real != op.level1_config(arity) {
                drifts.push(format!("{op:?} level1[{arity}] config"));
            }
        }
    }
    assert!(
        drifts.is_empty(),
        "level1 drift:\n  {}",
        drifts.join("\n  ")
    );
}

/// `fold_drift` — feeds the helper the PINNED LEVEL1 (node) config + PINNED node_target (no level1
/// rebuild), recomputes each arity's fold (node-verifying) node, and compares to the pinned fold
/// config/root.
#[test]
#[ignore = "box-only: builds the real fold node preprocessed circuits (~2^22)"]
fn fold_drift() {
    let node_blowup = TopologyConfig::default().recursion_log_blowup;
    let mut drifts: Vec<String> = Vec::new();
    for (op, _k, _shots) in POINTS {
        // The fold node verifies level1-shaped NODES; its child is the pinned level1 config.
        let child = op.level1_config(FOLD_ARITY);
        let node_target = op.node_target();
        for arity in 2..=FOLD_ARITY {
            let (real, root) = recompute_node(&child, arity, node_target.clone(), node_blowup);
            if arity == 2 {
                let lifting = real.config.lifting_log_size.expect("node lifting set");
                print_shape(&format!("{op:?} fold"), &real, lifting);
            }
            println!("// {op:?} fold[{arity}] root: {:?}", hv_words(&root));
            if real != op.fold_config(arity) {
                drifts.push(format!("{op:?} fold[{arity}] config"));
            }
        }
    }
    assert!(drifts.is_empty(), "fold drift:\n  {}", drifts.join("\n  "));
}

/// `unpacker_drift` — recomputes the trusted per-N unpacker config from the PINNED configs + roots
/// (via `pinned_aggregate_config`) and compares to the pinned unpacker config.
#[test]
#[ignore = "box-only: builds the real per-N unpacker preprocessed circuit"]
fn unpacker_drift() {
    use recursive_aggregate::root_prover::unpacker_verify_config;

    let topo = TopologyConfig::default();
    let node_blowup = topo.recursion_log_blowup;
    let mut drifts: Vec<String> = Vec::new();
    for (op, k, shots) in POINTS {
        let (cfg, params) = leaf_shape(k, shots);
        // The unpacker bakes every child root, so it needs the full pinned config (roots + shared
        // configs). Once captured, this is a pure function of the pinned consts + N.
        let config = leaf::pinned_aggregate_config(op, &topo, &cfg, &params);
        let n = op.n();
        let n_queries = config.node_pcs_config.fri_config.n_queries;
        let unpacker = unpacker_verify_config(n, &config, node_blowup, Some(n_queries));
        print_unpacker(op, &unpacker);
        if unpacker != op.unpacker_config(n) {
            drifts.push(format!("{op:?} unpacker config"));
        }
    }
    assert!(
        drifts.is_empty(),
        "unpacker drift:\n  {}",
        drifts.join("\n  ")
    );
}

/// Prints a captured unpacker [`CircuitConfig`] as paste-able `UnpackerConfigConst` literals.
fn print_unpacker(op: OperatingPoint, u: &CircuitConfig) {
    let pcs = &u.config;
    println!("// ===== {op:?} unpacker (paste into recursion_consts.rs UnpackerConfigConst) =====");
    println!(
        "//   pcs: PcsConfig {{ pow_bits: {}, fri_config: FriConfig {{ log_blowup_factor: {}, \
         log_last_layer_degree_bound: {}, n_queries: {}, fold_step: {} }}, lifting_log_size: {:?} }},",
        pcs.pow_bits,
        pcs.fri_config.log_blowup_factor,
        pcs.fri_config.log_last_layer_degree_bound,
        pcs.fri_config.n_queries,
        pcs.fri_config.fold_step,
        pcs.lifting_log_size,
    );
    println!("//   n_outputs: {},", u.n_outputs);
    println!("//   root: {:?},", hv_words(&u.preprocessed_root));
    println!("//   preprocessed_column_log_sizes: &[");
    for (id, log_size) in u.preprocessed_column_log_sizes.iter() {
        println!("//       ({:?}, {}),", id.id, log_size);
    }
    println!("//   ],");
}

/// SINGLE-PASS capture of the ENTIRE pinned-const table (all 3 points, all layers) — the from-scratch
/// capture tool feeding `scripts/gen_recursion_consts.py`. Unlike the per-layer drift tests (which feed
/// PINNED child configs, so they capture only AFTER a fill), this recomputes every layer FRESH from the
/// fixture in one pass, so a whole-table (re)capture is a single box run. Emits `@@`-prefixed lines the
/// generator parses. Run: `cargo test --release --features cuda capture_all -- --ignored --nocapture`.
#[test]
#[ignore = "box-only: single-pass full-table capture (real per-point cascade)"]
fn capture_all() {
    use recursive_aggregate::root_prover::unpacker_verify_config;

    let topo = TopologyConfig::default();
    let node_blowup = topo.recursion_log_blowup;
    let leaf_blowup = topo.leaf_log_blowup;
    for (op, k, shots) in POINTS {
        println!("@@POINT {op:?}");
        let (cfg, params) = leaf_shape(k, shots);

        // Leaf: fresh gate_air leaf preprocessed circuit (padded to its own natural target).
        let mut leaf_ctx = build_gate_air_leaf_circuit::<NoValue>(empty_proof(&cfg), &cfg, &params);
        let leaf_pad = compute_padded_sizes(&leaf_ctx);
        pad_to_targets(&mut leaf_ctx, leaf_pad);
        let leaf_pp = PreprocessedCircuit::preprocess_circuit(&mut leaf_ctx);
        let leaf_root = preprocessed_root(&leaf_pp, leaf_blowup);
        let leaf_cfg = CircuitConfig {
            config: leaf_pcs(leaf_pp.trace_log_size),
            n_outputs: circuit_common::N_RESERVED,
            preprocessed_column_log_sizes: leaf_pp.preprocessed_trace.log_sizes(),
            preprocessed_root: leaf_root.clone(),
        };
        emit_layer("LEAF", leaf_pp.trace_log_size, &leaf_cfg);
        emit_root("LEAF_ROOT", &leaf_root);

        // node_target: the fixed point (via the full derive).
        let config =
            derive_aggregate_config(&cfg, &params, topo.fold_arity, node_blowup, leaf_blowup);
        let nt = config.node_target_padding_sizes.clone();
        println!(
            "@@NODE_TARGET {} {} {} {} {}",
            nt.eq, nt.qm31_ops, nt.m31_to_u32, nt.triple_xor, nt.blake_g_gate
        );

        // level1 (child = fresh leaf config) + fold (child = fresh level1[k] config).
        let (level1_k, _) = recompute_node(&leaf_cfg, FOLD_ARITY, nt.clone(), node_blowup);
        for (tag, child) in [("LEVEL1", &leaf_cfg), ("FOLD", &level1_k)] {
            for arity in 2..=FOLD_ARITY {
                let (node_cfg, root) = recompute_node(child, arity, nt.clone(), node_blowup);
                if arity == 2 {
                    let trace =
                        node_cfg.config.lifting_log_size.expect("node lifting") - node_blowup;
                    emit_layer(tag, trace, &node_cfg);
                }
                emit_root(&format!("{tag}_ROOT_{arity}"), &root);
            }
        }

        // unpacker: from the fresh-derived config (real child roots baked).
        let n = op.n();
        let n_queries = config.node_pcs_config.fri_config.n_queries;
        let u = unpacker_verify_config(n, &config, node_blowup, Some(n_queries));
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
        println!("@@UNPACKER_COLS {}", cols_str(&u));
        emit_root("UNPACKER_ROOT", &u.preprocessed_root);
    }
    println!("@@CAPTURE_DONE");
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
