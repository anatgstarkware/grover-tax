//! Main gate execution component (demand side). `GateEval` drives the qubit-memory chain, the
//! ts-ordering range-check, and program-consistency lookups; it consumes all three relation ids,
//! imported from the supply-component files. Relocated verbatim from `air.rs` (byte-identical
//! constraint code); its private constraint helpers live alongside it here.

use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};

use crate::air::{pp_id, GateRel, LookupElements};
use crate::components::program::TAG_PROGRAM;
use crate::components::qubitmem::TAG_QUBITMEM;
use crate::components::range_check::TAG_RC;

/// Main gate constraint. One trace row per gate; per-row accesses (target + up to two controls)
/// drive the qubit-memory chain, the ts-ordering range-check, and program-consistency lookups.
#[derive(Clone)]
pub(crate) struct GateEval {
    pub(crate) log_n_rows: u32,
    pub(crate) elements: LookupElements,
}

impl FrameworkEval for GateEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::one();
        let two = BaseField::from_u32_unchecked(2);

        // Preprocessed (tree0) columns: shard-invariant positional values (verifier-pinned). enabler =
        // real-row indicator; shot_id = row/(k*n_gates); pc = row%(k*n_gates) (per-shot program counter,
        // whose verifier-pinned order forbids reordering an address's accesses — ts = pc+1 is inlined);
        // pc_in_prog = pc mod n_gates (the program slot each row addresses).
        let enabler = eval.get_preprocessed_column(pp_id("gate_enabler"));
        let shot_id = eval.get_preprocessed_column(pp_id("gate_shot_id"));
        let pc = eval.get_preprocessed_column(pp_id("gate_pc"));
        let pc_in_prog = eval.get_preprocessed_column(pp_id("gate_pc_in_prog"));

        let is_nop = eval.next_trace_mask();
        let is_not = eval.next_trace_mask();
        let is_cnot = eval.next_trace_mask();
        let is_toffoli = eval.next_trace_mask();

        // target/control accesses: addr, prev_ts, v_before, d (ts = pc+1 inlined; v_after inlined).
        let target = access_masks(&mut eval);
        let ctrl_a = access_masks(&mut eval);
        let ctrl_b = access_masks(&mut eval);

        let ab = eval.next_trace_mask();
        let fire = eval.next_trace_mask();
        let delta = eval.next_trace_mask();

        // Opcode booleanity + one-hot sum = enabler.
        for op in [&is_nop, &is_not, &is_cnot, &is_toffoli] {
            eval.add_constraint(op.clone() * (op.clone() - one.clone()));
        }
        eval.add_constraint(
            enabler.clone()
                - is_nop.clone()
                - is_not.clone()
                - is_cnot.clone()
                - is_toffoli.clone(),
        );

        let a_active = is_cnot.clone() + is_toffoli.clone();
        let b_active = is_toffoli.clone();

        // v_after is inlined (= v_before + delta), not a witness column; delta = fire*(1-2*v_before)
        // is enforced below, so v_after stays a bit and carries the write forward.
        let t_bit = target.v.clone(); // v_before
        let v_after = t_bit.clone() + delta.clone();

        // Value booleanity (memory values are 1 bit), including the derived v_after.
        for v in [&target.v, &v_after, &ctrl_a.v, &ctrl_b.v] {
            eval.add_constraint(v.clone() * (v.clone() - one.clone()));
        }

        // Gate-apply on the memory values: ab = v_a*v_b; fire = is_not + is_cnot*v_a + is_toffoli*ab;
        // delta = v_after - v_before = fire*(1 - 2*v_before) (v_after = v_before XOR fire).
        let a_bit = ctrl_a.v.clone();
        let b_bit = ctrl_b.v.clone();
        eval.add_constraint(ab.clone() - a_bit.clone() * b_bit.clone());
        eval.add_constraint(
            fire.clone()
                - is_not.clone()
                - is_cnot.clone() * a_bit.clone()
                - is_toffoli.clone() * ab.clone(),
        );
        eval.add_constraint(delta.clone() - fire.clone() + t_bit.clone() * fire.clone() * two);

        // ts = pc + 1 (inlined), shared by all three accesses of the step.
        let ts = pc.clone() + one.clone();

        // Qubit-memory chain: per active access Use(predecessor) + Yield(successor).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &target,
            &ts,
            &v_after,
            enabler.clone(),
        );
        // Control C1 (active iff is_cnot+is_toffoli): read propagates value (v_after = v).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &ctrl_a,
            &ts,
            &ctrl_a.v,
            a_active.clone(),
        );
        // Control C2 (active iff is_toffoli).
        add_qubitmem_pair(
            &mut eval,
            &self.elements.qubitmem,
            &shot_id,
            &ctrl_b,
            &ts,
            &ctrl_b.v,
            b_active.clone(),
        );

        // ts-ordering rc lookups, emitted here so the relation-batch order is qubitmem pairs (6 terms),
        // then the 3 single-`d` rc lookups, then program — folded by finalize-in-pairs into (rc_t, rc_a)
        // and (rc_b, program). Mirrored exactly by `gen_main_interaction` and MainGate.
        add_rc_lookup(&mut eval, &self.elements.rc, &target, enabler.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_a, a_active.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_b, b_active.clone());

        // ts-ordering range-check prev_ts < ts (soundness-critical): active*((pc+1) - prev_ts - 1 - d)
        // = 0 pins d = pc - prev_ts, range-checked into [0,2^RC_LOG) by the rc lookup above. With the
        // program-ordered ts and the LogUp chain balance this forces a forward DAG (no stale-read
        // cycle) — each read observes the program-order-last write. Inactive accesses unconstrained.
        add_ts_range(&mut eval, &ts, &target, enabler.clone());
        add_ts_range(&mut eval, &ts, &ctrl_a, a_active);
        add_ts_range(&mut eval, &ts, &ctrl_b, b_active);

        // Program-consistency (use side, +enabler): opcode_scalar = is_not + 2*is_cnot + 3*is_toffoli
        // (NOP -> 0); addresses are the access addr columns (0 for inactive controls, matching the
        // program table's canonical zero).
        let opcode_scalar = is_not.clone()
            + is_cnot.clone() * two
            + is_toffoli.clone() * BaseField::from_u32_unchecked(3);
        let prog_entry = [
            one.clone() * BaseField::from_u32_unchecked(TAG_PROGRAM),
            pc_in_prog.clone(),
            opcode_scalar,
            target.addr.clone(),
            ctrl_a.addr.clone(),
            ctrl_b.addr.clone(),
        ];
        eval.add_to_relation(RelationEntry::new(
            &self.elements.program,
            E::EF::from(enabler.clone()),
            &prog_entry,
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

// AIR-internal constraint helpers (used only by `GateEval` above).

struct AccessMasks<F> {
    addr: F,
    prev_ts: F,
    v: F,
    d: F, // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LOG_SIZE))
}

fn access_masks<E: EvalAtRow>(eval: &mut E) -> AccessMasks<E::F> {
    // Per-access columns (matches `cell_at` order): addr, prev_ts, v, d. ts is not a column (= pc+1).
    let addr = eval.next_trace_mask();
    let prev_ts = eval.next_trace_mask();
    let v = eval.next_trace_mask();
    let d = eval.next_trace_mask();
    AccessMasks {
        addr,
        prev_ts,
        v,
        d,
    }
}

/// Emit the chain Use(predecessor) + Yield(successor) pair for one access, gated by `active`. `v_out`
/// is the value written forward (v_after for the target, v for a control read).
fn add_qubitmem_pair<E: EvalAtRow>(
    eval: &mut E,
    elements: &GateRel,
    shot_id: &E::F,
    a: &AccessMasks<E::F>,
    ts: &E::F,
    v_out: &E::F,
    active: E::F,
) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_QUBITMEM);
    // Use predecessor: +active / (shot, addr, prev_ts, v_before).
    let use_entry = [
        tag.clone(),
        shot_id.clone(),
        a.addr.clone(),
        a.prev_ts.clone(),
        a.v.clone(),
    ];
    eval.add_to_relation(RelationEntry::new(
        elements,
        E::EF::from(active.clone()),
        &use_entry,
    ));
    // Yield successor: -active / (shot, addr, ts, v_after).
    let yield_entry = [
        tag,
        shot_id.clone(),
        a.addr.clone(),
        ts.clone(),
        v_out.clone(),
    ];
    eval.add_to_relation(RelationEntry::new(
        elements,
        -E::EF::from(active),
        &yield_entry,
    ));
}

/// Range-check reconstruction for one access, gated by `active`: active*(ts - prev_ts - 1 - d) = 0
/// pins the witness `d = pc - prev_ts`. `d` is range-checked into [0,2^RC_LOG) by `add_rc_lookup`, not
/// here, giving prev_ts < ts. Inactive accesses leave prev_ts/d free.
fn add_ts_range<E: EvalAtRow>(eval: &mut E, ts: &E::F, a: &AccessMasks<E::F>, active: E::F) {
    let one = E::F::one();
    let d = ts.clone() - a.prev_ts.clone() - one;
    eval.add_constraint(active * (d - a.d.clone()));
}

/// Single rc-table range-check lookup for one access, gated by `active`: `d` is looked up as (TAG_RC,
/// d). One term/access (mirrored by `gen_main_interaction` and MainGate).
fn add_rc_lookup<E: EvalAtRow>(eval: &mut E, rc: &GateRel, a: &AccessMasks<E::F>, active: E::F) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
    eval.add_to_relation(RelationEntry::new(
        rc,
        E::EF::from(active),
        &[tag, a.d.clone()],
    ));
}
