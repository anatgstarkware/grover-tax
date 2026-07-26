//! ts-ordering range-check supply component. Owns the rc relation id and the `RangeCheckEval` supply
//! table. Relocated verbatim from `air.rs` (byte-identical constraint code).

use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};

use crate::air::GateRel;
use crate::preprocessed::pp_id;

// Relation id tag (distinct constant; prover and in-circuit verifier must agree). TAG_RC =
// ts-ordering range-check (main looks up `d` as (TAG_RC, d), the rc table supplies (TAG_RC, value)
// for value in [0, 2^RC_LOG)).
pub(crate) const TAG_RC: u32 = 2;

/// ts-ordering range-check table (supply side). `val` is preprocessed (the table membership,
/// val[i]=i over [0,2^log_size)); `multiplicity` is witness (count of real `d` lookups landing on
/// this row). Emits -multiplicity / combine(TAG_RC, val) — one term/row => 1 batch => 4 interaction
/// columns. `log_size` is the trusted construction input `R = rc_log` (production: RC_LOG = 25).
#[derive(Clone)]
pub(crate) struct RangeCheckEval {
    pub(crate) log_size: u32,
    pub(crate) elements: GateRel,
}

impl FrameworkEval for RangeCheckEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let val = eval.get_preprocessed_column(pp_id("gate_rc_val"));
        let multiplicity = eval.next_trace_mask();
        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[tag, val],
        ));
        eval.finalize_logup();
        eval
    }
}
