//! PINNED recursion-const capture + DRIFT tests (box-only, `#[ignore]`d — they build real ~2^22 node
//! preprocessed circuits). `capture_all` recomputes the whole pinned table fresh in one pass (emitting
//! `@@` lines `gen_recursion_consts.py` parses); the per-point drift tests rebuild the real cascade and
//! `check_configs` asserts each layer/arity equals `op.pinned().to_derived(..)`.

// Parent module (`recursion_consts`).
use super::{OperatingPoint, FOLD_ARITY, RECURSION_LOG_BLOWUP};
// Crate-root items (ancestor privates: fixture parsing + shape/nonce helpers) and sibling modules.
use crate::air::{INTERACTION_POW_BITS, N_LIMBS, RC_LOG};
use crate::leaf::{self, build_gate_air_leaf_circuit, GateAirLeafParams};
use crate::preprocessed::N_PREPROCESSED_COLS;
use crate::test_utils::shard0_shape;
use crate::tracegen::state_to_limbs;
use crate::{hiding_nonce, parse_gtv1, program_rows_from_table, Fixture, TestCase};

use circuit_common::finalize::{compute_padded_sizes, pad_to_targets, ComponentSizes};
use circuit_common::preprocessed::PreprocessedCircuit;
use circuits::blake::HashValue;
use circuits::ivalue::NoValue;
use circuits_stark_verifier::proof::{empty_proof, ProofConfig};
use num_traits::Zero;
use std::fs::File;
use stwo::core::fields::qm31::SecureField;

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
    use crate::circuit_statement::gate_air_components;
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

/// Runs the whole fresh cascade for one point and asserts it equals the pinned `PinnedConfigs` table
/// (via `check_configs`, which re-derives + per-field/per-arity `assert_eq`). Builds the real
/// ~2^22 node preprocessed circuits, so box-only.
fn drift(op: OperatingPoint, k: usize, shots: usize) {
    let (cfg, params) = leaf_shape(k, shots);
    let (leaf_pp, leaf_pcs_cfg, _leaf_target) = build_leaf_pp(&cfg, &params, RECURSION_LOG_BLOWUP);
    recursive_aggregate::test_utils::check_configs(&leaf_pp, leaf_pcs_cfg, op.pinned());
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
/// fixture in one pass via the shared `capture_point`, so a whole-table (re)capture is a single box
/// run. Emits `@@`-prefixed lines the generator parses. Run:
/// `cargo test --release --features cuda capture_all -- --ignored --nocapture`.
#[test]
#[ignore = "box-only: single-pass full-table capture (real per-point cascade)"]
fn capture_all() {
    for (op, k, shots) in POINTS {
        println!("@@POINT {op:?}");
        let (cfg, params) = leaf_shape(k, shots);
        let (leaf_pp, leaf_pcs_cfg, _leaf_target) =
            build_leaf_pp(&cfg, &params, RECURSION_LOG_BLOWUP);
        recursive_aggregate::test_utils::capture_point(
            &leaf_pp,
            leaf_pcs_cfg,
            RECURSION_LOG_BLOWUP,
            RECURSION_LOG_BLOWUP,
            FOLD_ARITY,
            op.n(),
        );
    }
    println!("@@CAPTURE_DONE");
}
