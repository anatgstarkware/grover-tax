//! Qubit-memory boundary supply component. Owns the qubit-memory relation id and the `QubitMemEval`
//! boundary table. Relocated verbatim from `air.rs` (byte-identical constraint code).

use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};

use crate::air::{GateRel, TS_FINAL};
use crate::preprocessed::pp_id;

// Relation id tag (distinct constant; prover and in-circuit verifier must agree). TAG_QUBITMEM =
// per-qubit chain-lookup relation.
pub(crate) const TAG_QUBITMEM: u32 = 1;

/// Qubit-memory boundary table (supply side). Per (shot, addr) — `shot`/`addr` preprocessed,
/// `x`/`y`/`ts_last` witness — emits on TAG_QUBITMEM the chain head + tail:
///   init  Yield[-1](shot, addr, 0, x)
///   final Use [+1](shot, addr, ts_last, y)
/// Two terms/row => 1 batch. Booleanity on x, y (1-bit values).
#[derive(Clone)]
pub(crate) struct QubitMemEval {
    pub(crate) log_size: u32,
    pub(crate) elements: GateRel,
}

impl FrameworkEval for QubitMemEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::one();
        let shot = eval.get_preprocessed_column(pp_id("gate_bnd_shot"));
        let addr = eval.get_preprocessed_column(pp_id("gate_bnd_addr"));
        // Real-row enabler (1 on real (shot,addr), 0 on padding). Gates the boundary emission so
        // padding rows (non-power-of-two n_shots*N_QUBITS) inject no unmatched LogUp terms.
        let bnd_enabler = eval.get_preprocessed_column(pp_id("gate_bnd_enabler"));
        let x = eval.next_trace_mask();
        let y = eval.next_trace_mask();
        let ts_last = eval.next_trace_mask();
        // Booleanity of the boundary values.
        eval.add_constraint(x.clone() * (x.clone() - one.clone()));
        eval.add_constraint(y.clone() * (y.clone() - one.clone()));

        let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_QUBITMEM);
        let ts_final = E::F::one() * BaseField::from_u32_unchecked(TS_FINAL);
        // Phase-3 x/y binding (re-keyed boundary). The init anchor is NO LONGER consumed here:
        // main's first Use +1/(shot,addr,0,x) is left DANGLING as a PUBLIC term. The boundary now
        //   (B) INTERNAL final Use  +1 / (shot, addr, ts_last, y)  -> cancels main's last Yield.
        //   (D) PUBLIC  final Yield -1 / (shot, addr, TS_FINAL, y)  -> re-keys y to a fixed public ts.
        // Net base claimed_sum (per shot,addr) = +[0,x] − [TS_FINAL,y]; the leaf's public_logup_sum
        // supplies −that over its guessed x/y, so the verifier balance forces guessed == committed.
        // (`x` is retained as a witness column with its booleanity constraint above; it is no longer
        // emitted by the boundary — it lives only on main's dangling init term.)
        // (B) internal final Use[+bnd_enabler]: +bnd_enabler / (shot, addr, ts_last, y).
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            E::EF::from(bnd_enabler.clone()),
            &[tag.clone(), shot.clone(), addr.clone(), ts_last, y.clone()],
        ));
        // (D) public final Yield[-bnd_enabler]: -bnd_enabler / (shot, addr, TS_FINAL, y).
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(bnd_enabler),
            &[tag, shot, addr, ts_final, y],
        ));
        eval.finalize_logup_in_pairs();
        eval
    }
}
