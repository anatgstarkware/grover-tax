//! Program-consistency supply component. Owns the program relation ids and the `ProgramEval` supply
//! table. Relocated verbatim from `air.rs` (byte-identical constraint code).

use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};

use crate::air::{pp_id, GateRel};

pub(crate) const TAG_PROGRAM: u32 = 5;
// H_P program binding. The program table emits `-mult` on TAG_PROGRAM (internal, cancels main's
// demand) and `+mult` on TAG_PROGRAM_PUB (public dangling term P_pub in `program_sum`). The leaf
// supplies −P_pub over its guessed program Vars — which it also hashes into H_P — binding H_P to the
// executed program. A distinct tag is required (reusing TAG_PROGRAM would make the +mult cancel the
// −mult, vacuous). CPU-side (`gen_program_interaction`), not in the GPU K4 (MAIN-only) kernel.
pub(crate) const TAG_PROGRAM_PUB: u32 = 6;

/// Program-consistency table (supply side). Slot index is preprocessed; the op
/// fields (opcode_scalar, target, ctrl_a, ctrl_b) are WITNESS (the hidden
/// program) and the multiplicity column counts executions of that slot (K*N on
/// real slots, 0 on padding). Emits -multiplicity / combine(slot, op...).
#[derive(Clone)]
pub(crate) struct ProgramEval {
    pub(crate) log_size: u32,
    pub(crate) elements: GateRel,
}

impl FrameworkEval for ProgramEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let slot = eval.get_preprocessed_column(pp_id("gate_prog_slot"));
        let opcode_scalar = eval.next_trace_mask();
        let target = eval.next_trace_mask();
        let ctrl_a = eval.next_trace_mask();
        let ctrl_b = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();
        // H_P binding (Fork A): two supply terms, paired into ONE batch (so the program interaction
        // stays 4 columns — no tree2 layout shift, no GPU K4 change):
        //   (internal, -mult) / combine(TAG_PROGRAM,     slot, op, t, a, b)  -> cancels main's demand
        //   (public,   +mult) / combine(TAG_PROGRAM_PUB, slot, op, t, a, b)  -> dangling P_pub
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_PROGRAM);
        let tag_pub = E::F::one() * BaseField::from_u32_unchecked(TAG_PROGRAM_PUB);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity.clone()),
            &[
                tag,
                slot.clone(),
                opcode_scalar.clone(),
                target.clone(),
                ctrl_a.clone(),
                ctrl_b.clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            E::EF::from(multiplicity),
            &[tag_pub, slot, opcode_scalar, target, ctrl_a, ctrl_b],
        ));
        eval.finalize_logup_in_pairs();
        eval
    }
}
