//! In-circuit verifier of the gate_air STARK proof (Design A, Milestone 2).
//!
//! Mirrors gate_air's `FrameworkEval` components as `CircuitEval` components for the generic
//! `circuits_stark_verifier`. gate_air's LogUp uses ONE shared relation (`GateRel`) with a constant
//! id tag prepended to each tuple (see `TAG_*` in main.rs), exactly matching this verifier's single
//! `acc.interaction_elements` model — so each `add_to_relation` here prepends the same tag the
//! prover used, e.g. cairo-style `&[eval!(context, TAG_PROGRAM), …payload]`.
//!
//! Qubit-memory encoding (branch anatg/gate-air-qubit-mem): the whole-state TAG_STATE chain is
//! replaced by a per-qubit TAG_QUBITMEM chain lookup. Components: main gate, program (hidden-program
//! consistency), boundary (per (shot,addr) init/final anchoring). Same-row. Timestamp ordering is a
//! degree-1 equality `flag*(ts - prev_ts - 1) = 0` on a per-address +1 counter (no rc_lo table).
#![allow(dead_code)]

use circuits::blake::ReducedHashValue;
use circuits::context::{Context, Var};
use circuits::eval;
use circuits::ivalue::{IValue, qm31_from_u32s};
use circuits::ops::Guess;
use circuits::simd::Simd;
use circuits_stark_verifier::constraint_eval::{
    CircuitEval, ComponentDataTrait, CompositionConstraintAccumulator, RelationUse,
};
use circuits_stark_verifier::order_hash_map::OrderedHashMap;
use circuits_stark_verifier::proof_from_stark_proof::pack_into_qm31s;
use circuits_stark_verifier::statement::Statement;
use indexmap::IndexMap;
use stwo::core::fields::qm31::QM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::{
    ACCESS_BLOCK, ACCESS_COLS, LIMB_BITS, N_LIMBS, RC_LOG_SIZE, RC_LO_BITS, RC_POS_HI, RC_POS_LO,
    TAG_PROGRAM, TAG_QUBITMEM, TAG_RC, TRACE_COLUMNS, TS_FINAL,
    pp_id, preprocessed_column_ids,
};

/// QM31 constant Var of base-field value `v` (= `(v,0,0,0)`), used for tags and literals.
fn konst<Value: IValue>(context: &mut Context<Value>, v: u32) -> Var {
    context.constant(qm31_from_u32s(v, 0, 0, 0))
}

// Table components emit only their supply term(s); they consume no relation, so
// `relation_uses_per_row` is empty (matches cairo's range_check_12).
const NO_RELATION_USES: [RelationUse; 0] = [];
const ONE_TERM_INTERACTION_COLUMNS: usize = 4;

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

/// Qubit-memory boundary table (supply). `shot`/`addr` preprocessed; `x`/`y`/`ts_last` witness.
/// PHASE-3 re-keyed boundary (mirrors main.rs `BoundaryTableEval`): emits on TAG_QUBITMEM the internal
/// final Use[+1](shot,addr,ts_last,y) + the PUBLIC final Yield[-1](shot,addr,TS_FINAL,y). `x` is
/// booleanity-checked only (main carries x publicly at ts=0). Two terms/row => 1 batch => 4 cols.
pub struct BoundaryTable;
impl<Value: IValue> CircuitEval<Value> for BoundaryTable {
    fn name(&self) -> String {
        "gate_boundary_table".to_string()
    }
    fn trace_columns(&self) -> usize {
        3
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
        let [x, y, ts_last] = *component_data.trace_columns() else {
            panic!("boundary table: expected 3 trace columns")
        };
        let shot = acc.get_preprocessed_column(&pp_id("gate_bnd_shot"));
        let addr = acc.get_preprocessed_column(&pp_id("gate_bnd_addr"));
        // Real-row enabler (1 real, 0 padding): gates the emission so non-power-of-two n_shots*512
        // padding rows inject no unmatched LogUp terms. Mirrors main.rs BoundaryTableEval.
        let bnd_enabler = acc.get_preprocessed_column(&pp_id("gate_bnd_enabler"));
        let one = context.one();
        // Booleanity of the boundary values.
        let c = eval!(context, (x) * ((x) - (one)));
        acc.add_constraint(context, c);
        let c = eval!(context, (y) * ((y) - (one)));
        acc.add_constraint(context, c);

        let tag = context.constant(qm31_from_u32s(TAG_QUBITMEM, 0, 0, 0));
        let ts_final = context.constant(qm31_from_u32s(TS_FINAL, 0, 0, 0));
        let _ = x; // booleanity-checked above; not emitted by the boundary (main carries x at ts=0).
        // (B) internal final Use[+bnd_enabler] / (shot, addr, ts_last, y).
        acc.add_to_relation(context, bnd_enabler, &[tag, shot, addr, ts_last, y]);
        // (D) public final Yield[-bnd_enabler] / (shot, addr, TS_FINAL, y).
        let neg_enabler = eval!(context, -(bnd_enabler));
        acc.add_to_relation(context, neg_enabler, &[tag, shot, addr, ts_final, y]);
    }
}

/// ts-ordering range-check table (supply). `pos`/`val` preprocessed; `multiplicity` witness.
/// Mirrors main.rs `RcTableEval`: emits -multiplicity / (TAG_RC, pos, val). One term/row => 4 cols.
pub struct RcTable;
impl<Value: IValue> CircuitEval<Value> for RcTable {
    fn name(&self) -> String {
        "gate_rc_table".to_string()
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
        None
    }
    fn evaluate(
        &self,
        context: &mut Context<Value>,
        component_data: &dyn ComponentDataTrait<Value>,
        acc: &mut CompositionConstraintAccumulator,
    ) {
        let [mult] = *component_data.trace_columns() else {
            panic!("rc table: expected 1 trace column")
        };
        let pos = acc.get_preprocessed_column(&pp_id("gate_rc_pos"));
        let val = acc.get_preprocessed_column(&pp_id("gate_rc_val"));
        let tag = context.constant(qm31_from_u32s(TAG_RC, 0, 0, 0));
        let num = eval!(context, -(mult));
        acc.add_to_relation(context, num, &[tag, pos, val]);
    }
}

// ----------------------------------------------------------------------------
// Main gate component: mirrors gate_air's GateEval::evaluate exactly (same 22-col trace-column
// order, same constraint order, same 13 relation-term emission order => 6 pairs + 1 singleton => 28
// interaction cols). ts (= pc+1) and the target's v_after (= v_before+delta) are inlined, not
// columns. Timestamp ordering is a degree-1 range-check reconstruction + a limb lookup.
// ----------------------------------------------------------------------------

/// Parsed access-block Vars: addr, prev_ts, v, then the two range-check limbs rc_lo, rc_hi.
/// The access timestamp `ts` is NOT a column — it is the inlined `pc + 1` expression.
struct AccessVars {
    addr: Var,
    prev_ts: Var,
    v: Var,
    rc_lo: Var,
    rc_hi: Var,
}

fn parse_access(cols: &[Var], off: usize) -> AccessVars {
    AccessVars {
        addr: cols[off],
        prev_ts: cols[off + 1],
        v: cols[off + 2],
        rc_lo: cols[off + ACCESS_COLS],
        rc_hi: cols[off + ACCESS_COLS + 1],
    }
}

/// Main gate component. Emits 13 relation terms => 6 pairs + 1 singleton => 28 interaction columns
/// (3 qubitmem pairs + 3 rc-limb pairs + 1 program singleton).
pub struct MainGate;
impl<Value: IValue> CircuitEval<Value> for MainGate {
    fn name(&self) -> String {
        "gate_main".to_string()
    }
    fn trace_columns(&self) -> usize {
        TRACE_COLUMNS
    }
    fn interaction_columns(&self) -> usize {
        28
    }
    fn relation_uses_per_row(&self) -> &[RelationUse] {
        // Positive USE terms per row: 3 chain USEs (target + 2 controls) + 6 rc-limb range checks
        // (2 limbs * 3 accesses) + 1 program = 10. The 3 chain YIELDs are negative. (One shared
        // relation id.) The ts-ordering PIN + limb-reconstruction stay degree-1 algebraic.
        &[
            RelationUse { relation_id: "gate_qubitmem_use", uses: 3 },
            RelationUse { relation_id: "gate_rc_use", uses: 6 },
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
        // enabler, shot_id, pc, pc_in_prog are preprocessed (tree0).
        let enabler = acc.get_preprocessed_column(&pp_id("gate_enabler"));
        let shot_id = acc.get_preprocessed_column(&pp_id("gate_shot_id"));
        let pc = acc.get_preprocessed_column(&pp_id("gate_pc"));
        let pc_in_prog = acc.get_preprocessed_column(&pp_id("gate_pc_in_prog"));

        let is_nop = cols[0];
        let is_not = cols[1];
        let is_cnot = cols[2];
        let is_toffoli = cols[3];
        // target/ctrl blocks: ACCESS_BLOCK access cols each. ts (= pc+1) and the target's v_after
        // (= v_before + delta) are NOT columns — they are inlined below.
        let target = parse_access(cols, 4);
        let ctrl_a = parse_access(cols, 4 + ACCESS_BLOCK);
        let ctrl_b = parse_access(cols, 4 + 2 * ACCESS_BLOCK);
        let ab = cols[4 + 3 * ACCESS_BLOCK];
        let fire = cols[4 + 3 * ACCESS_BLOCK + 1];
        let delta = cols[4 + 3 * ACCESS_BLOCK + 2];
        let one = context.one();
        let two = konst(context, 2);
        let three = konst(context, 3);
        // ts = pc + 1 (inlined affine of the preprocessed pc, shared by all accesses of the step).
        let ts = eval!(context, (pc) + (one));
        // v_after = v_before + delta (inlined; target.v is v_before).
        let v_after = eval!(context, (target.v) + (delta));

        // --- Opcode booleanity + one-hot sum = enabler. ---
        for op in [is_nop, is_not, is_cnot, is_toffoli] {
            let c = eval!(context, (op) * ((op) - (one)));
            acc.add_constraint(context, c);
        }
        let c = eval!(context, ((((enabler) - (is_nop)) - (is_not)) - (is_cnot)) - (is_toffoli));
        acc.add_constraint(context, c);

        let a_active = eval!(context, (is_cnot) + (is_toffoli));
        let b_active = is_toffoli;

        // --- Value booleanity (memory values are 1 bit). The target's written value is the derived
        // v_after = v_before + delta; booleanity on it keeps the memory value a bit. ---
        for v in [target.v, v_after, ctrl_a.v, ctrl_b.v] {
            let c = eval!(context, (v) * ((v) - (one)));
            acc.add_constraint(context, c);
        }

        // --- Gate-apply on the memory values. ---
        let a_bit = ctrl_a.v;
        let b_bit = ctrl_b.v;
        let t_bit = target.v; // v_before
        let c = eval!(context, (ab) - ((a_bit) * (b_bit)));
        acc.add_constraint(context, c);
        let c = eval!(context, (((fire) - (is_not)) - ((is_cnot) * (a_bit))) - ((is_toffoli) * (ab)));
        acc.add_constraint(context, c);
        // delta = fire*(1 - 2*v_before). (v_after = v_before + delta is now inlined, so the old
        // `v_after - v_before - delta = 0` equality is vacuous and removed.)
        let c = eval!(context, ((delta) - (fire)) + (((t_bit) * (fire)) * (two)));
        acc.add_constraint(context, c);

        // --- Qubit-memory chain: per active access Use(predecessor) + Yield(successor). ---
        let tag_qm = konst(context, TAG_QUBITMEM);
        let tag_rc = konst(context, TAG_RC);
        let pos_lo = konst(context, RC_POS_LO);
        let pos_hi = konst(context, RC_POS_HI);
        let tag_prog = konst(context, TAG_PROGRAM);

        // pair0: target Use (+enabler), Yield (-enabler). ts = pc+1 (inlined); v_after = v_before+delta.
        acc.add_to_relation(context, enabler, &[tag_qm, shot_id, target.addr, target.prev_ts, target.v]);
        let neg_enabler = eval!(context, -(enabler));
        acc.add_to_relation(context, neg_enabler, &[tag_qm, shot_id, target.addr, ts, v_after]);
        // pair1: ctrl_a Use (+a_active), Yield (-a_active) (read: value propagates). ts = pc+1.
        acc.add_to_relation(context, a_active, &[tag_qm, shot_id, ctrl_a.addr, ctrl_a.prev_ts, ctrl_a.v]);
        let neg_a = eval!(context, -(a_active));
        acc.add_to_relation(context, neg_a, &[tag_qm, shot_id, ctrl_a.addr, ts, ctrl_a.v]);
        // pair2: ctrl_b Use (+b_active), Yield (-b_active). ts = pc+1.
        acc.add_to_relation(context, b_active, &[tag_qm, shot_id, ctrl_b.addr, ctrl_b.prev_ts, ctrl_b.v]);
        let neg_b = eval!(context, -(b_active));
        acc.add_to_relation(context, neg_b, &[tag_qm, shot_id, ctrl_b.addr, ts, ctrl_b.v]);

        // --- ts-ordering RANGE-CHECK LOOKUPs: per active access, look up its two limbs into the rc
        // table (mirrors main.rs `add_rc_lookup`). Emitted AFTER the qubitmem pairs and BEFORE the
        // program emit so the relation-batch order is qubitmem-pairs, rc-pairs (target lo/hi,
        // ctrl_a lo/hi, ctrl_b lo/hi), program singleton. ---
        let add_rc = |context: &mut Context<Value>,
                      acc: &mut CompositionConstraintAccumulator,
                      a: &AccessVars,
                      active: Var| {
            acc.add_to_relation(context, active, &[tag_rc, pos_lo, a.rc_lo]);
            acc.add_to_relation(context, active, &[tag_rc, pos_hi, a.rc_hi]);
        };
        add_rc(context, acc, &target, enabler);
        add_rc(context, acc, &ctrl_a, a_active);
        add_rc(context, acc, &ctrl_b, b_active);

        // --- ts-ordering: RANGE-CHECK reconstruction (mirrors main.rs `add_ts_range`,
        // target/ctrl_a/ctrl_b in order). Per active access:
        //   RANGE: active*(ts - prev_ts - 1 - rc_lo - 2^RC_LO_BITS*rc_hi) = 0 (the limbs are
        //       range-checked by the rc-table lookup above), with ts = pc+1 inlined. The old PIN
        //       constraint is gone (ts is structurally pc+1). pc is the preprocessed per-shot program
        //       counter (verifier-pinned); the structural program-ordered ts + the forward-DAG
        //       range-check defeat the reorder.
        let pow_lo = konst(context, 1u32 << RC_LO_BITS);
        let add_ts_range = |context: &mut Context<Value>,
                            acc: &mut CompositionConstraintAccumulator,
                            a: &AccessVars,
                            active: Var| {
            // RANGE reconstruction: d = rc_lo + 2^RC_LO_BITS * rc_hi, d = ts - prev_ts - 1 = pc - prev_ts.
            let recon = eval!(context, (a.rc_lo) + ((a.rc_hi) * (pow_lo)));
            let d = eval!(context, ((ts) - (a.prev_ts)) - (one));
            let c = eval!(context, (active) * ((d) - (recon)));
            acc.add_constraint(context, c);
        };
        add_ts_range(context, acc, &target, enabler);
        add_ts_range(context, acc, &ctrl_a, a_active);
        add_ts_range(context, acc, &ctrl_b, b_active);

        // --- Program-consistency (use side, +enabler). ---
        let opcode_scalar =
            eval!(context, ((is_not) + ((is_cnot) * (two))) + ((is_toffoli) * (three)));
        acc.add_to_relation(
            context,
            enabler,
            &[tag_prog, pc_in_prog, opcode_scalar, target.addr, ctrl_a.addr, ctrl_b.addr],
        );
    }
}

// ----------------------------------------------------------------------------
// GateAirStatement: the 4-component statement (order matches the prover:
// main, program, boundary, rc). The rc component is the ts-ordering range-check
// supply table (its LogUp lookups replace the algebraic bit-decomposition).
// ----------------------------------------------------------------------------

pub fn gate_air_components<Value: IValue>() -> IndexMap<&'static str, Box<dyn CircuitEval<Value>>> {
    IndexMap::from([
        ("gate_main", Box::new(MainGate) as Box<dyn CircuitEval<Value>>),
        ("gate_program", Box::new(ProgramTable) as Box<dyn CircuitEval<Value>>),
        ("gate_boundary", Box::new(BoundaryTable) as Box<dyn CircuitEval<Value>>),
        ("gate_rc", Box::new(RcTable) as Box<dyn CircuitEval<Value>>),
    ])
}

pub struct GateAirStatement<Value: IValue> {
    components: IndexMap<&'static str, Box<dyn CircuitEval<Value>>>,
    component_log_sizes: Simd,
    /// Base-proof preprocessed-trace Merkle root, GUESSED as a witness Var pair.
    preprocessed_root: ReducedHashValue<Var>,
    /// (x_limbs, y_limbs) per shot (shot_id = index) for the leaf's output-hash preimage, GUESSED as
    /// witness Vars.
    ///
    /// PHASE-3: these guessed limbs are now BOUND to the verified base proof. `public_logup_sum`
    /// bit-decomposes them (addr = limb*16 + bit, LSB-first) and forms the matching qubit-memory
    /// LogUp term over (shot, addr, ts, bit); the base's re-keyed boundary leaves a public term B and
    /// the in-circuit verifier balance `public_logup_sum + Σ claimed_sums == 0` forces these guessed
    /// x/y to equal the base's committed x/y. The output hash then commits to the BOUND values.
    boundary: Vec<([Var; N_LIMBS], [Var; N_LIMBS])>,
    /// Raw u32 limbs (same values as `boundary`, pre-guess) — the host source the bit-decomposition in
    /// `public_logup_sum` guesses from. For NoValue (shape) the bit values are irrelevant.
    boundary_u32: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    total_pc: u32,
    /// Log sizes needed to reproduce the DYNAMIC preprocessed column order.
    main_log_size: u32,
    program_log_size: u32,
    boundary_log_size: u32,
}

impl<Value: IValue> GateAirStatement<Value> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &mut Context<Value>,
        main_log_size: u32,
        program_log_size: u32,
        boundary_log_size: u32,
        preprocessed_root: ReducedHashValue<QM31>,
        boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
        total_pc: u32,
    ) -> Self {
        // Component order: main, program, boundary, rc (rc fixed at RC_LOG_SIZE).
        let log_sizes = [main_log_size, program_log_size, boundary_log_size, RC_LOG_SIZE];
        let n_components = log_sizes.len();
        let packed = pack_into_qm31s(log_sizes.iter().cloned())
            .into_iter()
            .map(|qm31| Value::from_qm31(qm31).guess(context))
            .collect::<Vec<_>>();
        let component_log_sizes = Simd::from_packed(packed, n_components);

        let preprocessed_root = ReducedHashValue(
            Value::from_qm31(preprocessed_root.0).guess(context),
            Value::from_qm31(preprocessed_root.1).guess(context),
        );
        let boundary_vars = boundary
            .iter()
            .map(|(x_limbs, y_limbs)| {
                let x = x_limbs.map(|v| Value::from_qm31(qm31_from_u32s(v, 0, 0, 0)).guess(context));
                let y = y_limbs.map(|v| Value::from_qm31(qm31_from_u32s(v, 0, 0, 0)).guess(context));
                (x, y)
            })
            .collect::<Vec<_>>();
        Self {
            components: gate_air_components(),
            component_log_sizes,
            preprocessed_root,
            boundary: boundary_vars,
            boundary_u32: boundary,
            total_pc,
            main_log_size,
            program_log_size,
            boundary_log_size,
        }
    }

    /// The guessed base-proof preprocessed root Vars (for building the leaf's output-hash preimage).
    pub fn preprocessed_root_vars(&self) -> &ReducedHashValue<Var> {
        &self.preprocessed_root
    }

    /// The guessed boundary (x, y) limb Vars per shot (for the leaf's output-hash preimage).
    pub fn boundary_vars(&self) -> &[([Var; N_LIMBS], [Var; N_LIMBS])] {
        &self.boundary
    }
}

impl<Value: IValue> Statement<Value> for GateAirStatement<Value> {
    fn claims_to_mix(&self, _context: &mut Context<Value>) -> Vec<Vec<Var>> {
        vec![vec![]]
    }
    fn get_components(&self) -> &IndexMap<&'static str, Box<dyn CircuitEval<Value>>> {
        &self.components
    }
    fn get_component_log_sizes(&self) -> &Simd {
        &self.component_log_sizes
    }
    fn get_preprocessed_column_ids(&self) -> Vec<PreProcessedColumnId> {
        preprocessed_column_ids(self.main_log_size, self.program_log_size, self.boundary_log_size)
    }
    fn get_preprocessed_root(&self, _context: &mut Context<Value>) -> ReducedHashValue<Var> {
        ReducedHashValue(self.preprocessed_root.0, self.preprocessed_root.1)
    }
    fn public_logup_sum(&self, context: &mut Context<Value>, interaction_elements: [Var; 2]) -> Var {
        // PHASE-3 x/y binding. The base is no longer internally balanced: its re-keyed boundary leaves
        // a PUBLIC dangling term B = Σ_{shot,addr} ( +[shot,addr,0,x] − [shot,addr,TS_FINAL,y] ) in the
        // committed claimed sums. `verify` enforces `public_logup_sum + Σ claimed_sums == 0`, so this
        // function must return −B computed over the GUESSED x/y:
        //     Σ_{shot,addr} ( −1/combine(TAG,shot,addr,0,x_bit) + 1/combine(TAG,shot,addr,TS_FINAL,y_bit) ).
        // Distinct tuples (ts=0 vs ts=TS_FINAL, value in {0,1}) => at random (z,α) the ONLY way the
        // balance holds is x_bit==committed x and y_bit==committed y per (shot,addr) — binding the
        // guessed limbs (which also feed the output hash) to the proven boundary. addr = limb*16 + bit
        // (LSB-first), matching the base's boundary addr encoding.
        use circuits::ops::{eq, inv};
        use circuits_stark_verifier::logup::combine_term;

        let tag = konst(context, TAG_QUBITMEM);
        let ts0 = context.zero();
        let ts_final = konst(context, TS_FINAL);
        // 2^p constants for the LSB-first limb reconstruction check.
        let pow2: Vec<Var> = (0..LIMB_BITS).map(|p| konst(context, 1u32 << p)).collect();

        let mut acc_sum = context.zero();
        for (shot, ((x_limbs, y_limbs), (x_u32, y_u32))) in
            self.boundary.iter().zip(self.boundary_u32.iter()).enumerate()
        {
            let shot_c = konst(context, shot as u32);
            for limb in 0..N_LIMBS {
                // Bit-decompose this limb; constrain the bits are boolean AND reconstruct the guessed
                // limb Var (so the bits are pinned to the SAME limbs that feed the output hash).
                let mut x_recon = context.zero();
                let mut y_recon = context.zero();
                for p in 0..LIMB_BITS {
                    let addr = (limb * LIMB_BITS + p) as u32;
                    let addr_c = konst(context, addr);
                    // Guess the bit from the host u32 (irrelevant for the NoValue shape pass).
                    let xb_val = (x_u32[limb] >> p) & 1;
                    let yb_val = (y_u32[limb] >> p) & 1;
                    let x_bit = Value::from_qm31(qm31_from_u32s(xb_val, 0, 0, 0)).guess(context);
                    let y_bit = Value::from_qm31(qm31_from_u32s(yb_val, 0, 0, 0)).guess(context);
                    // Booleanity: bit^2 == bit.
                    let xsq = eval!(context, (x_bit) * (x_bit));
                    eq(context, xsq, x_bit);
                    let ysq = eval!(context, (y_bit) * (y_bit));
                    eq(context, ysq, y_bit);
                    // Accumulate reconstruction Σ bit_p * 2^p.
                    let xt = eval!(context, (x_bit) * (pow2[p]));
                    x_recon = eval!(context, (x_recon) + (xt));
                    let yt = eval!(context, (y_bit) * (pow2[p]));
                    y_recon = eval!(context, (y_recon) + (yt));
                    // −1/combine(TAG, shot, addr, 0, x_bit).
                    let dx = combine_term(context, &[tag, shot_c, addr_c, ts0, x_bit], interaction_elements);
                    let ix = inv(context, dx);
                    acc_sum = eval!(context, (acc_sum) - (ix));
                    // +1/combine(TAG, shot, addr, TS_FINAL, y_bit).
                    let dy = combine_term(context, &[tag, shot_c, addr_c, ts_final, y_bit], interaction_elements);
                    let iy = inv(context, dy);
                    acc_sum = eval!(context, (acc_sum) + (iy));
                }
                // Reconstruction: Σ bit_p 2^p == guessed limb Var (pins bits to the hashed limbs).
                eq(context, x_recon, x_limbs[limb]);
                eq(context, y_recon, y_limbs[limb]);
            }
        }
        acc_sum
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;
    use crate::{build_rows, cell_at, parse_gtv1, Fixture};
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
        let (rows, _boundary) = build_rows(&gates, cases, k).unwrap();

        let dummy_interaction = vec![qm31_from_u32s(0, 0, 0, 0); 16];
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
            // enabler / shot_id / pc / pc_in_prog are PREPROCESSED; feed them via the map.
            let pc = row.pc;
            let pp = HashMap::from([
                (pp_id("gate_enabler"), ctx.constant(qm31_from_u32s(row.enabler, 0, 0, 0))),
                (pp_id("gate_shot_id"), ctx.constant(qm31_from_u32s(row.shot_id, 0, 0, 0))),
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
}
