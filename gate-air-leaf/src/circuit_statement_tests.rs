//! Laptop diagnostic: `MainGate`'s explicit (non-LogUp) constraints must be zero on every valid
//! trace row, cross-checking the `FrameworkEval` → `CircuitEval` constraint translation.

use super::*;
use crate::{build_rows, cell_at, parse_gtv1, Fixture};
use circuits::context::Context;
use circuits_stark_verifier::test_utils::TestComponentData;
use std::collections::HashMap;

// The LogUp `add_to_relation` terms don't touch `acc.accumulation` until `finalize_logup_in_pairs`
// (which we skip), so `acc.finalize() == 0` on a valid row iff the constraint translation is correct.
#[test]
fn main_explicit_constraints_zero_on_valid_rows() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../grover-tax/fixtures/v0.3-iadd256-k4-n16.json"
    );
    let fx: Fixture = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let gates = parse_gtv1(&fx.circuit_byte_serialisation_hex).unwrap();
    let n_gates = gates.len() as u32;
    let k = fx.repetitions;
    let cases = &fx.test_cases[..1];
    let (rows, _boundary) = build_rows(&gates, cases, k).unwrap();

    let dummy_interaction = vec![qm31_from_u32s(0, 0, 0, 0); 16];
    for (ri, row) in rows.iter().enumerate() {
        let mut ctx = Context::<QM31>::default();
        let trace: Vec<QM31> = (0..TRACE_COLUMNS)
            .map(|c| qm31_from_u32s(cell_at(row, c), 0, 0, 0))
            .collect();
        let comp = TestComponentData::from_values(
            &mut ctx,
            &trace,
            &dummy_interaction,
            qm31_from_u32s(0, 0, 0, 0),
            1 << 14,
        );
        // enabler / shot_id / pc / pc_in_prog are PREPROCESSED; feed them via the map.
        let pc = row.pc;
        let pp = HashMap::from([
            (
                pp_id("gate_enabler"),
                ctx.constant(qm31_from_u32s(row.enabler, 0, 0, 0)),
            ),
            (
                pp_id("gate_shot_id"),
                ctx.constant(qm31_from_u32s(row.shot_id, 0, 0, 0)),
            ),
            (pp_id("gate_pc"), ctx.constant(qm31_from_u32s(pc, 0, 0, 0))),
            (
                pp_id("gate_pc_in_prog"),
                ctx.constant(qm31_from_u32s(pc % n_gates, 0, 0, 0)),
            ),
        ]);
        let coeff = ctx.constant(qm31_from_u32s(7, 11, 13, 17));
        let ie = [
            ctx.constant(qm31_from_u32s(2, 3, 5, 7)),
            ctx.constant(qm31_from_u32s(19, 23, 29, 31)),
        ];
        let mut acc =
            CompositionConstraintAccumulator::new(&mut ctx, pp, HashMap::new(), coeff, ie);
        MainGate.evaluate(&mut ctx, &comp, &mut acc);
        let result = acc.finalize();
        assert_eq!(
            ctx.get(result),
            qm31_from_u32s(0, 0, 0, 0),
            "MainGate explicit constraints nonzero at row {ri} (pc={pc})"
        );
    }
    eprintln!("OK: explicit constraints zero on all {} rows", rows.len());
}
