//! In-circuit verifier of the gate_air STARK proof (Design A, Milestone 2).
//!
//! Mirrors gate_air's `FrameworkEval` components as `CircuitEval` components for the generic
//! `circuits_stark_verifier`. gate_air's LogUp uses ONE shared relation (`GateRel`) with a constant
//! id tag prepended to each tuple (see `TAG_*` in main.rs), exactly matching this verifier's single
//! `acc.interaction_elements` model — so each `add_to_relation` here prepends the same tag the
//! prover used, e.g. cairo-style `&[eval!(context, TAG_QDECODE), …payload]`.
//!
//! This file: the 4 table (supply-side) components. The main gate component + `GateAirStatement`
//! (with the public boundary `public_logup_sum`) + the verify glue follow.
#![allow(dead_code)]

use circuits::blake::HashValue;
use circuits::context::{Context, Var};
use circuits::eval;
use circuits::ivalue::{IValue, qm31_from_u32s};
use circuits::ops::{Guess, eq, inv};
use circuits::simd::Simd;
use circuits_stark_verifier::constraint_eval::{
    CircuitEval, ComponentDataTrait, CompositionConstraintAccumulator, RelationUse,
};
use circuits_stark_verifier::logup::combine_term;
use circuits_stark_verifier::order_hash_map::OrderedHashMap;
use circuits_stark_verifier::proof_from_stark_proof::pack_into_qm31s;
use circuits_stark_verifier::statement::Statement;
use indexmap::IndexMap;
use stwo::core::fields::qm31::QM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::{
    N_LIMBS, N_QUBITS, RC_LOG_SIZE, TAG_PROGRAM, TAG_QDECODE, TAG_RC_HI, TAG_RC_LO, TAG_STATE,
    TRACE_COLUMNS, pp_id, preprocessed_column_ids,
};

/// QM31 constant Var of base-field value `v` (= `(v,0,0,0)`), used for tags and literals.
fn konst<Value: IValue>(context: &mut Context<Value>, v: u32) -> Var {
    context.constant(qm31_from_u32s(v, 0, 0, 0))
}

// Table components emit only their supply term (-multiplicity / tagged tuple); they consume no
// relation, so `relation_uses_per_row` is empty (matches cairo's range_check_12). One LogUp term =>
// one batch => SECURE_EXTENSION_DEGREE (4) interaction columns.
const NO_RELATION_USES: [RelationUse; 0] = [];
const ONE_TERM_INTERACTION_COLUMNS: usize = 4;

/// q-decode membership table (supply): -mult / [TAG_QDECODE, q, limb_idx, bit_pos, mask].
/// Fixed 2^9 rows (N_QUBITS = 512).
pub struct QdecodeTable;
impl<Value: IValue> CircuitEval<Value> for QdecodeTable {
    fn name(&self) -> String {
        "gate_qdecode_table".to_string()
    }
    fn trace_columns(&self) -> usize {
        1
    }
    fn interaction_columns(&self) -> usize {
        ONE_TERM_INTERACTION_COLUMNS
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        &NO_RELATION_USES
    }
    fn log_size(&self, _: &OrderedHashMap<PreProcessedColumnId, u32>) -> Option<u32> {
        Some(N_QUBITS.ilog2())
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let [mult] = *component_data.trace_columns() else {
            panic!("qdecode table: expected 1 trace column")
        };
        let q = acc.get_preprocessed_column(&pp_id("gate_qdecode_q"));
        let limb = acc.get_preprocessed_column(&pp_id("gate_qdecode_limb"));
        let pos = acc.get_preprocessed_column(&pp_id("gate_qdecode_pos"));
        let mask = acc.get_preprocessed_column(&pp_id("gate_qdecode_mask"));
        let tag = context.constant(qm31_from_u32s(TAG_QDECODE, 0, 0, 0));
        let num = eval!(context, -(mult));
        acc.add_to_relation(context, num, &[tag, q, limb, pos, mask]);

        let size_bit = component_data.get_n_instances_bit(context, N_QUBITS.ilog2() as usize);
        eq(context, size_bit, context.one());
    }
}

/// Dynamic range-check low table (supply): -mult / [TAG_RC_LO, pos, val]. Fixed 2^RC_LOG_SIZE rows.
pub struct RcLoTable;
impl<Value: IValue> CircuitEval<Value> for RcLoTable {
    fn name(&self) -> String {
        "gate_rc_lo_table".to_string()
    }
    fn trace_columns(&self) -> usize {
        1
    }
    fn interaction_columns(&self) -> usize {
        ONE_TERM_INTERACTION_COLUMNS
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        &NO_RELATION_USES
    }
    fn log_size(&self, _: &OrderedHashMap<PreProcessedColumnId, u32>) -> Option<u32> {
        Some(RC_LOG_SIZE)
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let [mult] = *component_data.trace_columns() else {
            panic!("rc_lo table: expected 1 trace column")
        };
        let pos = acc.get_preprocessed_column(&pp_id("gate_rc_lo_pos"));
        let val = acc.get_preprocessed_column(&pp_id("gate_rc_lo_val"));
        let tag = context.constant(qm31_from_u32s(TAG_RC_LO, 0, 0, 0));
        let num = eval!(context, -(mult));
        acc.add_to_relation(context, num, &[tag, pos, val]);

        let size_bit = component_data.get_n_instances_bit(context, RC_LOG_SIZE as usize);
        eq(context, size_bit, context.one());
    }
}

/// Dynamic range-check high table (supply): -mult / [TAG_RC_HI, pos, val]. Fixed 2^RC_LOG_SIZE rows.
pub struct RcHiTable;
impl<Value: IValue> CircuitEval<Value> for RcHiTable {
    fn name(&self) -> String {
        "gate_rc_hi_table".to_string()
    }
    fn trace_columns(&self) -> usize {
        1
    }
    fn interaction_columns(&self) -> usize {
        ONE_TERM_INTERACTION_COLUMNS
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        &NO_RELATION_USES
    }
    fn log_size(&self, _: &OrderedHashMap<PreProcessedColumnId, u32>) -> Option<u32> {
        Some(RC_LOG_SIZE)
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let [mult] = *component_data.trace_columns() else {
            panic!("rc_hi table: expected 1 trace column")
        };
        let pos = acc.get_preprocessed_column(&pp_id("gate_rc_hi_pos"));
        let val = acc.get_preprocessed_column(&pp_id("gate_rc_hi_val"));
        let tag = context.constant(qm31_from_u32s(TAG_RC_HI, 0, 0, 0));
        let num = eval!(context, -(mult));
        acc.add_to_relation(context, num, &[tag, pos, val]);

        let size_bit = component_data.get_n_instances_bit(context, RC_LOG_SIZE as usize);
        eq(context, size_bit, context.one());
    }
}

/// Program-consistency table (supply): -mult / [TAG_PROGRAM, slot, opcode, target, ctrl_a, ctrl_b].
/// Variable log_size (set per proof), so no fixed size assert here.
pub struct ProgramTable;
impl<Value: IValue> CircuitEval<Value> for ProgramTable {
    fn name(&self) -> String {
        "gate_program_table".to_string()
    }
    fn trace_columns(&self) -> usize {
        5
    }
    fn interaction_columns(&self) -> usize {
        ONE_TERM_INTERACTION_COLUMNS
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        &NO_RELATION_USES
    }
    fn log_size(&self, _: &OrderedHashMap<PreProcessedColumnId, u32>) -> Option<u32> {
        None
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let [opcode_scalar, target, ctrl_a, ctrl_b, mult] = *component_data.trace_columns() else {
            panic!("program table: expected 5 trace columns")
        };
        let slot = acc.get_preprocessed_column(&pp_id("gate_prog_slot"));
        let tag = context.constant(qm31_from_u32s(TAG_PROGRAM, 0, 0, 0));
        let num = eval!(context, -(mult));
        acc.add_to_relation(context, num, &[tag, slot, opcode_scalar, target, ctrl_a, ctrl_b]);
    }
}

// ----------------------------------------------------------------------------
// Main gate component (the big one): mirrors gate_air's GateEval::evaluate exactly
// (same trace-column order, same constraint order, same 12 relation-term emission order).
// ----------------------------------------------------------------------------

/// Parsed read-block Vars: q, limb_idx, bit_pos, mask, lsel[N_LIMBS], lo, hi, bit (4+N_LIMBS+3 = 39).
struct ReadVars {
    q: Var,
    limb_idx: Var,
    bit_pos: Var,
    mask: Var,
    lsel: Vec<Var>,
    lo: Var,
    hi: Var,
    bit: Var,
}

fn parse_read(cols: &[Var], off: usize) -> ReadVars {
    ReadVars {
        q: cols[off],
        limb_idx: cols[off + 1],
        bit_pos: cols[off + 2],
        mask: cols[off + 3],
        lsel: cols[off + 4..off + 4 + N_LIMBS].to_vec(),
        lo: cols[off + 4 + N_LIMBS],
        hi: cols[off + 5 + N_LIMBS],
        bit: cols[off + 6 + N_LIMBS],
    }
}

/// In-circuit twin of gate_air::read_constraints — SAME constraint order.
fn read_constraints_circuit<Value: IValue>(
    context: &mut Context<Value>,
    acc: &mut CompositionConstraintAccumulator,
    r: &ReadVars,
    active: Var,
    in_limb: &[Var],
) {
    let one = context.one();
    // lsel booleanity.
    for &s in &r.lsel {
        let c = eval!(context, (s) * ((s) - (one)));
        acc.add_constraint(context, c);
    }
    // sum lsel = active; sum j*lsel_j = limb_idx.
    let mut sum = context.zero();
    let mut weighted = context.zero();
    for (j, &s) in r.lsel.iter().enumerate() {
        sum = eval!(context, (sum) + (s));
        let jc = konst(context, j as u32);
        weighted = eval!(context, (weighted) + ((s) * (jc)));
    }
    let c = eval!(context, (sum) - (active));
    acc.add_constraint(context, c);
    let limb_idx = r.limb_idx;
    let c = eval!(context, (weighted) - (limb_idx));
    acc.add_constraint(context, c);
    // selected limb L = sum lsel_j * in_limb_j; split L = hi*2*mask + bit*mask + lo.
    let mut l = context.zero();
    for (j, &s) in r.lsel.iter().enumerate() {
        let il = in_limb[j];
        l = eval!(context, (l) + ((s) * (il)));
    }
    let two = konst(context, 2);
    let (mask, hi, bit, lo) = (r.mask, r.hi, r.bit, r.lo);
    let c = eval!(context, (((l) - (((hi) * (two)) * (mask))) - ((bit) * (mask))) - (lo));
    acc.add_constraint(context, c);
    // bit booleanity.
    let c = eval!(context, (bit) * ((bit) - (one)));
    acc.add_constraint(context, c);
    // inactive => bit forced 0.
    let c = eval!(context, ((one) - (active)) * (bit));
    acc.add_constraint(context, c);
}

/// Main gate component. Emits 12 relation terms => 6 batches => 24 interaction columns.
pub struct MainGate;
impl<Value: IValue> CircuitEval<Value> for MainGate {
    fn name(&self) -> String {
        "gate_main".to_string()
    }
    fn trace_columns(&self) -> usize {
        TRACE_COLUMNS
    }
    fn interaction_columns(&self) -> usize {
        24
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        &[
            RelationUse { relation_id: "gate_state", uses: 2 },
            RelationUse { relation_id: "gate_qdecode", uses: 3 },
            RelationUse { relation_id: "gate_rc_lo", uses: 3 },
            RelationUse { relation_id: "gate_rc_hi", uses: 3 },
            RelationUse { relation_id: "gate_program", uses: 1 },
        ]
    }
    fn log_size(&self, _: &OrderedHashMap<PreProcessedColumnId, u32>) -> Option<u32> {
        None
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let cols = component_data.trace_columns();
        assert_eq!(cols.len(), TRACE_COLUMNS, "main: unexpected trace column count");
        let enabler = cols[0];
        let is_nop = cols[1];
        let is_not = cols[2];
        let is_cnot = cols[3];
        let is_toffoli = cols[4];
        let shot_id = cols[5];
        let pc = cols[6];
        let in_limb: Vec<Var> = cols[7..7 + N_LIMBS].to_vec();
        let out_limb: Vec<Var> = cols[7 + N_LIMBS..7 + 2 * N_LIMBS].to_vec();
        let base = 7 + 2 * N_LIMBS;
        let read_w = 4 + N_LIMBS + 3;
        let target = parse_read(cols, base);
        let ctrl_a = parse_read(cols, base + read_w);
        let ctrl_b = parse_read(cols, base + 2 * read_w);
        let ab = cols[base + 3 * read_w];
        let fire = cols[base + 3 * read_w + 1];
        let delta = cols[base + 3 * read_w + 2];

        let pc_in_prog = acc.get_preprocessed_column(&pp_id("gate_pc_in_prog"));
        let one = context.one();
        let two = konst(context, 2);
        let three = konst(context, 3);

        // --- Opcode booleanity + one-hot sum = enabler. ---
        for op in [is_nop, is_not, is_cnot, is_toffoli] {
            let c = eval!(context, (op) * ((op) - (one)));
            acc.add_constraint(context, c);
        }
        let c = eval!(context, ((((enabler) - (is_nop)) - (is_not)) - (is_cnot)) - (is_toffoli));
        acc.add_constraint(context, c);

        let a_active = eval!(context, (is_cnot) + (is_toffoli));
        let b_active = is_toffoli;

        // --- Per-read structural constraints (target, ctrl_a, ctrl_b). ---
        read_constraints_circuit(context, acc, &target, enabler, &in_limb);
        read_constraints_circuit(context, acc, &ctrl_a, a_active, &in_limb);
        read_constraints_circuit(context, acc, &ctrl_b, b_active, &in_limb);

        // --- Fire / delta. ---
        let a_bit = ctrl_a.bit;
        let b_bit = ctrl_b.bit;
        let t_bit = target.bit;
        let c = eval!(context, (ab) - ((a_bit) * (b_bit)));
        acc.add_constraint(context, c);
        let c = eval!(context, (((fire) - (is_not)) - ((is_cnot) * (a_bit))) - ((is_toffoli) * (ab)));
        acc.add_constraint(context, c);
        let c = eval!(context, ((delta) - (fire)) + (((t_bit) * (fire)) * (two)));
        acc.add_constraint(context, c);

        // --- Write: out_limb[j] = in_limb[j] + target.lsel[j]*delta*target.mask. ---
        let tmask = target.mask;
        for j in 0..N_LIMBS {
            let (o, i, ls) = (out_limb[j], in_limb[j], target.lsel[j]);
            let c = eval!(context, ((o) - (i)) - (((ls) * (delta)) * (tmask)));
            acc.add_constraint(context, c);
        }

        // --- Relations (state in/out, qdecode x3, rc(lo,hi) x3, program). ---
        let tag_state = konst(context, TAG_STATE);
        let tag_qd = konst(context, TAG_QDECODE);
        let tag_lo = konst(context, TAG_RC_LO);
        let tag_hi = konst(context, TAG_RC_HI);
        let tag_prog = konst(context, TAG_PROGRAM);

        let mut s_in = vec![tag_state, shot_id, pc];
        s_in.extend_from_slice(&in_limb);
        acc.add_to_relation(context, enabler, &s_in);
        let pc1 = eval!(context, (pc) + (one));
        let neg_enabler = eval!(context, -(enabler));
        let mut s_out = vec![tag_state, shot_id, pc1];
        s_out.extend_from_slice(&out_limb);
        acc.add_to_relation(context, neg_enabler, &s_out);

        for (r, active) in [(&target, enabler), (&ctrl_a, a_active), (&ctrl_b, b_active)] {
            acc.add_to_relation(context, active, &[tag_qd, r.q, r.limb_idx, r.bit_pos, r.mask]);
        }
        for (r, active) in [(&target, enabler), (&ctrl_a, a_active), (&ctrl_b, b_active)] {
            acc.add_to_relation(context, active, &[tag_lo, r.bit_pos, r.lo]);
            acc.add_to_relation(context, active, &[tag_hi, r.bit_pos, r.hi]);
        }
        let opcode_scalar =
            eval!(context, ((is_not) + ((is_cnot) * (two))) + ((is_toffoli) * (three)));
        acc.add_to_relation(
            context,
            enabler,
            &[tag_prog, pc_in_prog, opcode_scalar, target.q, ctrl_a.q, ctrl_b.q],
        );
    }
}

// ----------------------------------------------------------------------------
// GateAirStatement: the 5-component statement (order matches the prover:
// main, qdecode, rc_lo, rc_hi, program).
// ----------------------------------------------------------------------------

pub fn gate_air_components<Value: IValue>() -> IndexMap<&'static str, Box<dyn CircuitEval<Value>>> {
    IndexMap::from([
        ("gate_main", Box::new(MainGate) as Box<dyn CircuitEval<Value>>),
        ("gate_qdecode", Box::new(QdecodeTable) as Box<dyn CircuitEval<Value>>),
        ("gate_rc_lo", Box::new(RcLoTable) as Box<dyn CircuitEval<Value>>),
        ("gate_rc_hi", Box::new(RcHiTable) as Box<dyn CircuitEval<Value>>),
        ("gate_program", Box::new(ProgramTable) as Box<dyn CircuitEval<Value>>),
    ])
}

pub struct GateAirStatement<Value: IValue> {
    components: IndexMap<&'static str, Box<dyn CircuitEval<Value>>>,
    component_log_sizes: Simd,
    /// Preprocessed-trace Merkle root (from the proof's commitments[0], via `.into()`).
    preprocessed_root: HashValue<QM31>,
    /// (x_limbs, y_limbs) per shot (shot_id = index) for the public state boundary.
    boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    total_pc: u32,
    /// Main-trace + program-table log sizes — needed to reproduce the DYNAMIC preprocessed column
    /// order (pc_in_prog is sized with the main trace, so the size-sorted order depends on it).
    main_log_size: u32,
    program_log_size: u32,
}

impl<Value: IValue> GateAirStatement<Value> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &mut Context<Value>,
        main_log_size: u32,
        program_log_size: u32,
        preprocessed_root: HashValue<QM31>,
        boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
        total_pc: u32,
    ) -> Self {
        let log_sizes = [main_log_size, N_QUBITS.ilog2(), RC_LOG_SIZE, RC_LOG_SIZE, program_log_size];
        let n_components = log_sizes.len();
        let packed = pack_into_qm31s(log_sizes.iter().cloned())
            .into_iter()
            .map(|qm31| Value::from_qm31(qm31).guess(context))
            .collect::<Vec<_>>();
        let component_log_sizes = Simd::from_packed(packed, n_components);
        Self {
            components: gate_air_components(),
            component_log_sizes,
            preprocessed_root,
            boundary,
            total_pc,
            main_log_size,
            program_log_size,
        }
    }
}

impl<Value: IValue> Statement<Value> for GateAirStatement<Value> {
    fn claims_to_mix(&self, _context: &mut Context<Value>) -> Vec<Vec<Var>> {
        // Empty public claim (per-component claimed sums are mixed by the verify machinery; the
        // boundary is reconstructed in public_logup_sum).
        vec![vec![]]
    }
    fn get_components(&self) -> &IndexMap<&'static str, Box<dyn CircuitEval<Value>>> {
        &self.components
    }
    fn get_component_log_sizes(&self) -> &Simd {
        &self.component_log_sizes
    }
    fn get_preprocessed_column_ids(&self) -> Vec<PreProcessedColumnId> {
        preprocessed_column_ids(self.main_log_size, self.program_log_size)
    }
    fn get_preprocessed_root(&self, context: &mut Context<Value>) -> HashValue<Var> {
        HashValue(
            context.constant(self.preprocessed_root.0),
            context.constant(self.preprocessed_root.1),
        )
    }
    fn public_logup_sum(&self, context: &mut Context<Value>, interaction_elements: [Var; 2]) -> Var {
        // verify checks `public_logup_sum + Σ claimed_sums == 0`, and gate_air's Σ claimed_sums =
        // v_boundary = Σ_shots(1/ci − 1/cf). So public_logup_sum = −v_boundary = Σ_shots(1/cf − 1/ci)
        // with ci = combine([TAG_STATE, shot, 0, x]), cf = combine([TAG_STATE, shot, total_pc, y]).
        let tag = konst(context, TAG_STATE);
        let total = konst(context, self.total_pc);
        let zero_pc = context.zero();
        let mut sum = context.zero();
        for (shot_id, (x_limbs, y_limbs)) in self.boundary.iter().enumerate() {
            let shot = konst(context, shot_id as u32);
            let mut e_in = vec![tag, shot, zero_pc];
            e_in.extend(x_limbs.iter().map(|&v| konst(context, v)));
            let mut e_out = vec![tag, shot, total];
            e_out.extend(y_limbs.iter().map(|&v| konst(context, v)));
            let ci = combine_term(context, &e_in, interaction_elements);
            let cf = combine_term(context, &e_out, interaction_elements);
            let ci_inv = inv(context, ci);
            let cf_inv = inv(context, cf);
            sum = eval!(context, ((sum) + (cf_inv)) - (ci_inv));
        }
        sum
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;
    use crate::{build_rc_hi, build_rc_lo, build_rows, cell_at, parse_gtv1, Fixture};
    use circuits::context::Context;
    use circuits_stark_verifier::test_utils::TestComponentData;
    use std::collections::HashMap;

    // Laptop diagnostic: every valid trace row must satisfy MainGate's EXPLICIT constraints (the
    // logup add_to_relation terms don't touch acc.accumulation until finalize_logup_in_pairs, which
    // we skip), so acc.finalize() == 0 on a valid row iff the constraint translation is correct.
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
        let rc_lo = build_rc_lo();
        let rc_hi = build_rc_hi();
        let (rows, _counts) = build_rows(&gates, cases, k, &rc_lo, &rc_hi).unwrap();

        let dummy_interaction = vec![qm31_from_u32s(0, 0, 0, 0); 24];
        for (ri, row) in rows.iter().enumerate() {
            let mut ctx = Context::<QM31>::default();
            let trace: Vec<QM31> =
                (0..TRACE_COLUMNS).map(|c| qm31_from_u32s(cell_at(row, c), 0, 0, 0)).collect();
            let comp = TestComponentData::from_values(
                &mut ctx,
                &trace,
                &dummy_interaction,
                qm31_from_u32s(0, 0, 0, 0),
                1 << 14,
            );
            let pc = cell_at(row, 6);
            let pp = HashMap::from([(
                pp_id("gate_pc_in_prog"),
                ctx.constant(qm31_from_u32s(pc % n_gates, 0, 0, 0)),
            )]);
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
}
