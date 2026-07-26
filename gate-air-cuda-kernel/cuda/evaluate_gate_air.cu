// ============================================================================
// gate_air MAIN component constraint kernel — PHASE 1 (algebraic core).
//
// !!! BOX-UNVALIDATED CUDA — cannot be compiled on the laptop (no nvcc). !!!
// Transcribed line-for-line from `impl FrameworkEval for GateEval`
// (gate-air-leaf/src/main.rs, QUBIT-MEMORY + ts=pc+1 (inlined) + single-`d` rc-table encoding).
// Build + validate on the GPU box.
//
// QUBIT-MEMORY LAYOUT (main.rs, 19 cols, ACCESS_BLOCK=4): the main (witness) trace is:
//   [0..4) opcode one-hots; [4..8) target addr/prev_ts/v/d;
//   [8..12) ctrl_a access; [12..16) ctrl_b access; [16..19) ab/fire/delta.
// ts (= pc+1) and the target's v_after (= v_before+delta) are INLINED, NOT columns.
// `enabler`, `shot_id`, `pc`, `pc_in_prog` live in the PREPROCESSED tree (tree0), read via
// eval0.get_preprocessed_column() in that call order (GATE_AIR_N_PREPROCESSED = 4). `pc` feeds
// the inlined ts = pc + 1.
//
// Pipeline (mirror of evaluate_memory_address_to_id.cu):
//   1. pre_kernel  : per eval-domain row, run the 15 ALGEBRAIC add_constraints
//                    (-> numerators[row] = row_res) and emit the 10 LogUp
//                    relation entries (-> intermediate_fractions). PHASE 1.
//                    (ts=pc+1 inlined + single-`d` rc-table: per active access 1 ts algebraic
//                    constraint (RANGE recon only; PIN dropped) + 1 rc LOOKUP (TAG_RC, d);
//                    the v_after equality is dropped (inlined). Algebraic 19->15,
//                    LogUp entries 10 (3 qubitmem pairs + 3 rc-d + 1 program).)
//   2. post_kernel : generic_constraint_post_kernel (evaluate_common.cuh) folds
//                    the 10 fractions into 5 batches (all pairs)
//                    and adds the 5 LogUp cumsum constraints. PHASE 2 SOUNDNESS
//                    GATE — wired here but NOT trusted until box accumulator-diff
//                    is zero.
//   3. finalize    : generic_constraint_quotients_finalize_kernel — quotient =
//                    numerators[row] * denom_inv[row >> trace_log_size], written
//                    into the 4 accumulator coord columns honoring
//                    should_accumulate. Identical to accumulate_pointwise_cpu.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_common.cuh"
#include "evaluate_gate_air.cuh"

#define GATE_AIR_THREAD_COUNT_MAX 256

// ----------------------------------------------------------------------------
// Relation slicing helper.
//
// gate_air draws ONE width-6 relation (`relation!(GateRel, 6)`, main.rs) shared by all three
// logical relations (qubitmem / rc / program); they are kept distinct only by the integer TAG
// prepended as values[0] (main.rs). Every emit combines against the SAME (z, alpha, alpha_powers),
// using only the first N alpha_powers for an N-wide tuple — see the Rust `combine`
// (constraint-framework logup.rs), which folds `alpha_powers[0..values.len()]` and subtracts z.
//
// The `CudaEvaluator::add_to_relation<N>(RelationEntry<N>)` overload requires a `RelationEntry<N>`
// whose `relation` field is a `LookupElementsBasic<N>`. The gate relation is
// `LookupElementsBasic<6>`, so for the N<6 entries (qubitmem N=5, rc N=3) we build the matching
// `LookupElementsBasic<N>` by copying z, alpha, and the first N alpha_powers. This yields
// bit-identical `combine` (the math only touches alpha_powers[0..N]). N==6 (program) needs no slice.
template<int N>
DEVICE_FORCEINLINE LookupElementsBasic<N> gate_relation_slice(
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &rel
) {
    LookupElementsBasic<N> sliced;
    sliced.z = rel.z;
    sliced.alpha = rel.alpha;
    for (int i = 0; i < N; ++i) {
        sliced.alpha_powers[i] = rel.alpha_powers[i];
    }
    return sliced;
}

// ----------------------------------------------------------------------------
// Per-access masks (qubit-memory), in the EXACT order `access_masks` (main.rs) consumes columns
// from the main trace: addr, prev_ts, v, d. 4 columns per access (ACCESS_BLOCK).
// ts is NOT a column — it is the inlined `pc + 1`.
// ----------------------------------------------------------------------------
struct GateAccessMasks {
    m31 addr;
    m31 prev_ts;
    m31 v;
    m31 d;        // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^rc_log))
};

// Mirror of `access_masks` (main.rs): pull 4 consecutive main-trace masks (addr,prev_ts,v then
// d — matching cell_at's per-access column order).
template<typename EvaluatorT>
DEVICE_FORCEINLINE GateAccessMasks gate_access_masks(EvaluatorT &eval) {
    GateAccessMasks a;
    a.addr    = eval.next_trace_mask();
    a.prev_ts = eval.next_trace_mask();
    a.v       = eval.next_trace_mask();
    a.d       = eval.next_trace_mask();
    return a;
}

// Mirror of `add_qubitmem_pair` (main.rs): emit the chain Use(predecessor) + Yield(successor) pair
// for one access, gated by `active`. `ts` = the inlined timestamp (pc+1, shared across the step);
// `v_out` = value written forward (v_after = v_before+delta for the target, v for a control read).
//   Use   [+active] : (TAG_QUBITMEM, shot, addr, prev_ts, v_before)
//   Yield [-active] : (TAG_QUBITMEM, shot, addr, ts=pc+1, v_out)
// Order is Use then Yield — load-bearing (fraction index order).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_qubitmem_pair(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    m31 shot_id,
    const GateAccessMasks &a,
    m31 ts,
    m31 v_out,
    m31 active
) {
    qm31 mult = { { active, 0 }, { 0, 0 } };            // E::EF::from(active)
    m31 use_values[5] = { m31(GATE_AIR_TAG_QUBITMEM), shot_id, a.addr, a.prev_ts, a.v };
    RelationEntry<5> use_entry(gate_relation_slice<5>(relation), mult, use_values);
    eval.template add_to_relation<5>(use_entry);

    m31 neg_active = neg(active);
    qm31 neg_mult = { { neg_active, 0 }, { 0, 0 } };
    m31 yield_values[5] = { m31(GATE_AIR_TAG_QUBITMEM), shot_id, a.addr, ts, v_out };
    RelationEntry<5> yield_entry(gate_relation_slice<5>(relation), neg_mult, yield_values);
    eval.template add_to_relation<5>(yield_entry);
}

// Mirror of `add_rc_lookup` (main.rs): emit the SINGLE rc-table range-check LOOKUP for one access,
// gated by `active`. `d` is looked up as (TAG_RC, d); the rc supply table supplies each in-range
// value. One term/access (mirrored by gen_main_interaction and the in-circuit MainGate).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_rc_lookup(
    EvaluatorT &eval,
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &relation,
    const GateAccessMasks &a,
    m31 active
) {
    qm31 mult = { { active, 0 }, { 0, 0 } };            // E::EF::from(active)
    m31 d_values[2] = { m31(GATE_AIR_TAG_RC), a.d };
    RelationEntry<2> d_entry(gate_relation_slice<2>(relation), mult, d_values);
    eval.template add_to_relation<2>(d_entry);
}

// Mirror of `add_ts_range` (main.rs): emit the ONE flag-gated degree-1 ts-ordering ALGEBRAIC
// constraint for one access:
//   RANGE: active * ((pc+1) - prev_ts - 1 - d) = 0 with d = ts - prev_ts - 1 = pc - prev_ts —
//          pins the witness `d` column to the diff (d is range-checked by gate_add_rc_lookup, not
//          here). `ts` is the inlined `pc + 1`. The old PIN constraint is GONE (ts is structurally
//          pc+1, so the pin is vacuous).
template<typename EvaluatorT>
DEVICE_FORCEINLINE void gate_add_ts_range(
    EvaluatorT &eval,
    m31 ts,
    const GateAccessMasks &a,
    m31 active
) {
    // RANGE recon: active * ((ts - prev_ts - 1) - d), d = ts - prev_ts - 1 = pc - prev_ts.
    m31 recon = sub(sub(ts, a.prev_ts), m31(1));
    eval.add_constraint(mul(active, sub(recon, a.d)));
}

// ----------------------------------------------------------------------------
// pre_kernel: one thread per eval-domain row. Transcribes GateEval::evaluate.
// ----------------------------------------------------------------------------
template<typename EvaluatorT>
__launch_bounds__(GATE_AIR_THREAD_COUNT_MAX, 2)
__global__ void evaluate_gate_air_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    GateAirEval *gate_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array,
    unsigned row_offset,
    unsigned tile_rows
) {
    // Candidate 2 (row-tiling): this launch processes the eval-domain rows
    // [row_offset, row_offset + tile_rows). The thread's GLOBAL eval-domain row
    // is row_offset + local index — this MUST stay global because every trace
    // read (trace_evaluations[col][row]) and every mask-offset
    // (offset_bit_reversed_circle_domain_index(row, ...)) indexes by global row.
    // Only the `intermediate_fractions` buffer is tiled: the caller passes a
    // pointer biased back by row_offset*logup_counts, so the evaluator's
    // global-row index (row*logup_counts) lands at the correct tile-local slot
    // d_fractions[(row-row_offset)*logup_counts + i]. numerators[] /
    // constraint_index_array[] stay full-size and are indexed by global row.
    const unsigned local = threadIdx.x + blockDim.x * blockIdx.x;
    if (local >= tile_rows) {
        return;
    }
    const unsigned row = row_offset + local;

    // Two evaluators sharing the SAME row_res / constraint_index / fraction
    // index accumulation. Each evaluator tracks its own col_index per
    // interaction. The PREPROCESSED columns (tree0) and the MAIN columns (tree1)
    // are two separate device pointer tables, so we mirror
    // memory_address_to_id: use eval0 for trace0 reads (preprocessed) and eval
    // for trace1 reads (main), carrying row_res / constraint_index /
    // fraction_index forward by hand.
    //
    // PREPROCESSED (main.rs GateEval::evaluate). The Rust `evaluate()` reads FOUR
    // preprocessed columns up front, in this EXACT call order:
    //     enabler, shot_id, pc, pc_in_prog
    // (= GateEval's `preprocessed_column_indices` order, which is the order the
    // InfoEvaluator records `get_preprocessed_column` calls; the CUDA component
    // prover gathers `trace0_evaluations` in precisely that index order — see
    // constraint-framework component.rs / component_prover.rs). eval0 therefore
    // reads them sequentially via col_index[0] in the SAME order. The main trace is
    // the 19-column qubit-memory layout (4 opcode masks + target(4) + 2*ctrl(4) + 3;
    // ts = pc+1 and v_after = v_before+delta are inlined, not columns).

    EvaluatorT eval0(
        trace0_evaluations, random_coeff_powers,
        0, row, qm31{{0, 0}, {0, 0}},
        0, cumsum_shift, domain_log_size, eval_domain_log_size,
        intermediate_fractions, logup_counts
    );
    // Preprocessed (tree0) reads, in GateEval call order (main.rs:778-781):
    //   enabler, shot_id, pc, pc_in_prog.
    m31 enabler    = eval0.get_preprocessed_column();  // gate_enabler   (main.rs:778)
    m31 shot_id    = eval0.get_preprocessed_column();  // gate_shot_id   (main.rs:779)
    m31 pc         = eval0.get_preprocessed_column();  // gate_pc        (main.rs:780)
    m31 pc_in_prog = eval0.get_preprocessed_column();  // gate_pc_in_prog(main.rs:781)

    EvaluatorT eval(
        trace1_evaluations, random_coeff_powers,
        0, row, qm31{{0, 0}, {0, 0}},
        0, cumsum_shift, domain_log_size, eval_domain_log_size,
        intermediate_fractions, logup_counts
    );

    // --- Main-trace masks (19 cols), in declaration order (main.rs GateEval::evaluate). ---
    // Header is ONLY the 4 opcode masks (enabler/shot_id/pc/pc_in_prog are tree0).
    m31 is_nop     = eval.next_trace_mask();   // col 0
    m31 is_not     = eval.next_trace_mask();   // col 1
    m31 is_cnot    = eval.next_trace_mask();   // col 2
    m31 is_toffoli = eval.next_trace_mask();   // col 3

    // target access (addr,prev_ts,v,d) cols 4..8. ts (=pc+1) and v_after (=v_before+delta)
    // are NOT columns — inlined below.
    GateAccessMasks target = gate_access_masks(eval);   // cols 4..8
    GateAccessMasks ctrl_a = gate_access_masks(eval);   // cols 8..12
    GateAccessMasks ctrl_b = gate_access_masks(eval);   // cols 12..16

    m31 ab    = eval.next_trace_mask();   // col 16
    m31 fire  = eval.next_trace_mask();   // col 17
    m31 delta = eval.next_trace_mask();   // col 18

    // ts = pc + 1 (inlined affine of the preprocessed pc, shared by all accesses of the step).
    m31 ts = add(pc, m31(1));
    // v_after = v_before + delta (inlined; target.v is v_before).
    m31 v_after = add(target.v, delta);

    // ===================== ALGEBRAIC CONSTRAINTS (part 1: 12) =====================
    // Order MUST match GateEval::evaluate exactly. (The 3 ts-ordering accesses add 3 more algebraic
    // constraints — ONE RANGE recon per access — emitted below in `add_ts_range` position, for 15.
    // The old PIN per access and the v_after equality are dropped.)
    // [1-4] opcode booleanity op*(op-1) for is_nop/is_not/is_cnot/is_toffoli.
    eval.add_constraint(mul(is_nop,     sub(is_nop,     m31(1))));
    eval.add_constraint(mul(is_not,     sub(is_not,     m31(1))));
    eval.add_constraint(mul(is_cnot,    sub(is_cnot,    m31(1))));
    eval.add_constraint(mul(is_toffoli, sub(is_toffoli, m31(1))));
    // [5] one-hot sum = enabler: enabler - is_nop - is_not - is_cnot - is_toffoli.
    eval.add_constraint(
        sub(sub(sub(sub(enabler, is_nop), is_not), is_cnot), is_toffoli)
    );

    // active sub-expressions. NOT columns — computed inline.
    m31 a_active = add(is_cnot, is_toffoli);
    m31 b_active = is_toffoli;

    // [6-9] value booleanity v*(v-1) for target.v, v_after(=v_before+delta), ctrl_a.v, ctrl_b.v (in
    // this order). Booleanity on the derived v_after keeps the target's written memory value a bit.
    eval.add_constraint(mul(target.v, sub(target.v, m31(1))));
    eval.add_constraint(mul(v_after,  sub(v_after,  m31(1))));
    eval.add_constraint(mul(ctrl_a.v, sub(ctrl_a.v, m31(1))));
    eval.add_constraint(mul(ctrl_b.v, sub(ctrl_b.v, m31(1))));

    // --- Gate-apply on the memory values. ---
    m31 a_bit = ctrl_a.v;
    m31 b_bit = ctrl_b.v;
    m31 t_bit = target.v;   // v_before

    // [10] ab - a_bit*b_bit.
    eval.add_constraint(sub(ab, mul(a_bit, b_bit)));
    // [11] fire - is_not - is_cnot*a_bit - is_toffoli*ab.
    eval.add_constraint(
        sub(sub(sub(fire, is_not), mul(is_cnot, a_bit)), mul(is_toffoli, ab))
    );
    // [12] delta - fire + 2*t_bit*fire  (delta = v_after - v_before = fire*(1 - 2*v_before)).
    // (The old [13] v_after - v_before - delta = 0 equality is dropped — v_after is now inlined
    // as v_before + delta.)
    eval.add_constraint(
        add(sub(delta, fire), mul(mul(t_bit, fire), m31(2)))
    );

    // ===================== LOGUP RELATION ENTRIES (10) =====================
    // PHASE 1 emits the 10 entries (the post_kernel turns them into 5 LogUp batch constraints in
    // PHASE 2: all pairs). Order MUST match `evaluate()` / gen_main_interaction exactly:
    //   qubitmem target/ctrl_a/ctrl_b Use/Yield (6), rc target/ctrl_a/ctrl_b d (3), program (1).
    // The 10-entry stream folds into 5 pairs: (t_use,t_yield)(a_use,a_yield)(b_use,b_yield)
    //   (rc_t, rc_a)(rc_b, program).
    // The ts-ordering RANGE recon is a degree-1 algebraic add_constraint (gate_add_ts_range), NOT a
    // relation entry; it is emitted AFTER the rc LOOKUPs and BEFORE the program emit (the same
    // interleave as main.rs `evaluate()`), so the relation-batch order stays qubitmem, rc, program.
    const LookupElementsBasic<GATE_AIR_REL_WIDTH> &rel = gate_eval->relation;

    // 1,2. qubitmem target: Use(+enabler)/Yield(-enabler); ts = pc+1, v_out = v_after = v_before+delta.
    gate_add_qubitmem_pair(eval, rel, shot_id, target, ts, v_after, enabler);
    // 3,4. qubitmem ctrl_a: Use(+a_active)/Yield(-a_active); ts = pc+1, read propagates value (v_out = v).
    gate_add_qubitmem_pair(eval, rel, shot_id, ctrl_a, ts, ctrl_a.v, a_active);
    // 5,6. qubitmem ctrl_b: Use(+b_active)/Yield(-b_active); ts = pc+1.
    gate_add_qubitmem_pair(eval, rel, shot_id, ctrl_b, ts, ctrl_b.v, b_active);

    // 7 / 8 / 9. rc range-check LOOKUPs (single `d`) for target / ctrl_a / ctrl_b — emitted
    // as a group AFTER the qubitmem pairs (mirrors main.rs `add_rc_lookup` × 3 order).
    gate_add_rc_lookup(eval, rel, target, enabler);
    gate_add_rc_lookup(eval, rel, ctrl_a, a_active);
    gate_add_rc_lookup(eval, rel, ctrl_b, b_active);

    // ts-ordering RANGE recon: 1 degree-1 algebraic add_constraint per access (#13..15), NOT relation
    // entries. ts = pc+1 inlined; the old PIN per access is gone (main.rs `add_ts_range` order).
    gate_add_ts_range(eval, ts, target, enabler);
    gate_add_ts_range(eval, ts, ctrl_a, a_active);
    gate_add_ts_range(eval, ts, ctrl_b, b_active);

    // 10. program (+enabler).
    //   opcode_scalar = is_not*1 + is_cnot*2 + is_toffoli*3.
    //   [TAG_PROGRAM, pc_in_prog, opcode_scalar, target.addr, ctrl_a.addr, ctrl_b.addr]
    {
        m31 opcode_scalar = add(add(is_not, mul(is_cnot, m31(2))), mul(is_toffoli, m31(3)));
        m31 prog[6] = {
            m31(GATE_AIR_TAG_PROGRAM), pc_in_prog, opcode_scalar,
            target.addr, ctrl_a.addr, ctrl_b.addr
        };
        qm31 mult = { { enabler, 0 }, { 0, 0 } };
        RelationEntry<6> entry(gate_relation_slice<6>(rel), mult, prog);
        eval.template add_to_relation<6>(entry);
    }

    // Persist the algebraic count so the post_kernel resumes the random-coeff
    // index at the right spot (mirror memory_address_to_id.cu:91-92).
    constraint_index_array[row] = eval.constraint_index;
    numerators[row] = eval.row_res;
}

// ----------------------------------------------------------------------------
// Tiled post_kernel (Candidate 2 row-tiling).
//
// BYTE-IDENTICAL copy of generic_constraint_post_kernel (evaluate_common.cuh
// :45-117) with TWO mechanical changes for row-tiling, nothing else:
//   * the global eval-domain row is row_offset + local thread index, and the
//     bound check is `local >= tile_rows` (instead of row >= eval_domain_size);
//   * `intermediate_fractions` is the tile-biased pointer supplied by the host
//     (d_fractions - row_offset*logup_counts), so the row*logup_counts indexing
//     reads the tile-local slots written by the tiled pre_kernel.
// The per-row math (Fraction::sum, the cumsum masks via next_extension_
// interaction_mask, diff/fixed_diff, add_constraint_ext, finalize_logup_in_pairs
// folding) is copied verbatim — do NOT alter it. We keep a private copy here so
// the shared generic_constraint_post_kernel (89 callers) is left untouched.
template<typename EvaluatorT>
__launch_bounds__(GATE_AIR_THREAD_COUNT_MAX, 2)
__global__ void evaluate_gate_air_post_kernel_tiled(
    qm31 *numerators,
    Fraction *intermediate_fractions,
    unsigned *constraint_index_array,
    m31 **trace2_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int logup_counts,
    unsigned int last_batch,
    qm31 cumsum_shift,
    unsigned row_offset,
    unsigned tile_rows,
    // F2-b / Option B: when true, the 4 shifted last-LogUp cumsum coords are appended
    // to trace2_evaluations at indices [logup_cols*4 .. logup_cols*4 + 4) and the last
    // batch reads prev_row_cumsum from them at OFFSET 0 (both tileable) instead of via
    // the scattered {0,-1} mask on the source coords. When false, the byte-for-byte
    // legacy resident path (scattered -1 read on the source coords) runs unchanged.
    bool tree2_shifted
) {
    const unsigned local = threadIdx.x + blockDim.x * blockIdx.x;
    if (local >= tile_rows) return;
    const unsigned row = row_offset + local;

    EvaluatorT evaluator(
        trace2_evaluations,
        random_coeff_powers,
        constraint_index_array[row],
        row,
        numerators[row],
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    const unsigned logup_interaction = 2;
    qm31 prev_col_cumsum = { {0, 0}, {0, 0} };

    // Process complete batches
    for (unsigned i = 0; i < last_batch; ++i) {
        const Fraction cur_frac = Fraction::sum(&intermediate_fractions[2 * i + row * logup_counts], 2);

        qm31 cur_cumsum_arr[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        int offsets[2] = { 0, 0 };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets, 1, cur_cumsum_arr);

        const qm31 cur_cumsum = cur_cumsum_arr[0];
        const qm31 diff = sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;

        const qm31 constraint_val = sub(mul(diff, cur_frac.denominator), cur_frac.numerator);
        evaluator.add_constraint_ext(constraint_val);
    }

    // Process remaining fractions
    {
        unsigned remaining_fractions = logup_counts - last_batch * 2;
        const Fraction frac_sum = Fraction::sum(&intermediate_fractions[last_batch * 2 + row * logup_counts], remaining_fractions);

        qm31 cur_cumsum;
        qm31 prev_row_cumsum;
        if (tree2_shifted) {
            // F2-b / Option B: cur_cumsum from the source coords (cols [16..20)) at
            // offset 0, prev_row_cumsum from the appended shifted coords (cols [20..24))
            // at offset 0. Both offset-0 pointwise reads over the row-tiled trace2
            // pointer table — no scattered `-1`, so the tile slice is self-contained.
            // The two consecutive next_extension_interaction_mask calls advance
            // col_index[2]: 16->20 (source) then 20->24 (shifted). BYTE-IDENTICAL to
            // the scattered read below because shifted[row] == src[obr_index(row,-1)].
            int off0[1] = { 0 };
            qm31 cur_arr[1] = { { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, off0, 1, cur_arr);
            cur_cumsum = cur_arr[0];

            qm31 prev_arr[1] = { { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, off0, 1, prev_arr);
            prev_row_cumsum = prev_arr[0];
        } else {
            // Legacy resident path: scattered `-1` read on the source coords (cols
            // [16..20)) — BYTE-FOR-BYTE the pre-tiling behavior.
            int offsets2[2] = { 0, -1 };
            qm31 cumsum2[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
            evaluator.next_extension_interaction_mask(logup_interaction, offsets2, 2, cumsum2);
            cur_cumsum = cumsum2[0];
            prev_row_cumsum = cumsum2[1];
        }

        const qm31 diff = sub(sub(cur_cumsum, prev_row_cumsum), prev_col_cumsum);
        const qm31 fixed_diff = add(diff, cumsum_shift);

        const qm31 constraint_val = sub(mul(fixed_diff, frac_sum.denominator), frac_sum.numerator);

        evaluator.add_constraint_ext(constraint_val);
    }

    numerators[row] = evaluator.row_res;
}

// ----------------------------------------------------------------------------
// Host entry: launch pre -> post -> finalize (mirror evaluate_memory_address_to_id).
// ----------------------------------------------------------------------------
// Resolve the row-tile size (rows/block). GATE_AIR_TILE_ROWS is the preferred
// knob (default 2^22); GATE_AIR_COMP_TILE is kept as a back-compat alias for the
// fraction-only tiling. Clamped to eval_domain_size by the caller.
static unsigned gate_air_resolve_tile_rows() {
    // The tiling only bounds the d_fractions transient (numerators is full-domain
    // regardless), and with no per-tile sync fewer/larger tiles are strictly better
    // (fewer launch waves, longer kernels to pipeline). Default 2^22 so a 2^27 eval
    // domain runs ~32 tiles instead of ~128; this only enlarges d_fractions
    // (tile_rows*logup_counts*sizeof(Fraction) = 2^22*10*32B ~= 1.34 GiB), which the
    // resident shard has VRAM headroom for. An explicit GATE_AIR_TILE_ROWS /
    // GATE_AIR_COMP_TILE env override wins and is unchanged.
    unsigned tile_rows = 1u << 22;
    if (const char *env = std::getenv("GATE_AIR_TILE_ROWS")) {
        unsigned long parsed = std::strtoul(env, nullptr, 10);
        if (parsed > 0) tile_rows = (unsigned)parsed;
    } else if (const char *env2 = std::getenv("GATE_AIR_COMP_TILE")) {
        unsigned long parsed = std::strtoul(env2, nullptr, 10);
        if (parsed > 0) tile_rows = (unsigned)parsed;
    }
    return tile_rows;
}

extern "C"
void evaluate_gate_air(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
) {
    (void)number_of_columns;

    GateAirEval *gate_eval = (GateAirEval *) eval;
    const unsigned eval_domain_size = 1u << eval_domain_log_size;

    // All trace columns are RESIDENT on device (the full-eval-domain buffers). The
    // per-tile kernels are enqueued on the same `stream`, so stream ordering already
    // serializes tile b's pre before its post before tile b+1's kernels (the only
    // shared per-tile scratch is d_fractions, reused via frac_biased). We therefore
    // drop any mid-loop sync (keeping a non-blocking cudaGetLastError() launch-error
    // check per tile) and issue ONE cudaStreamSynchronize after the loop + finalize,
    // before any host read of results.

    // Resident device pointer tables: whole-clone each tree's live device pointers.
    m31 **d_trace0 = clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **d_trace1 = clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **d_trace2 = clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    // NOTE: cuda_alloc_zeroes_uint32_t's argument is a COUNT OF uint32_t, not a
    // byte count. A qm31 is 4 uint32_t, so `eval_domain_size` qm31 accumulators
    // need `4 * eval_domain_size` u32 (2.0 GiB at 2^26). The previous
    // `sizeof(qm31) * eval_domain_size` passed 16 * eval_domain_size — a 4x
    // over-allocation (8.0 GiB at 2^26) whose 32-bit value (2^31) additionally
    // truncated in the old `int` helper param. Both are fixed here: correct u32
    // count + 64-bit clean multiply into the now-size_t helper. Byte-identical:
    // numerators[row] is only ever indexed for row < eval_domain_size.
    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t((size_t)4 * eval_domain_size);

    GateAirEval *d_gate_eval = cuda_malloc<GateAirEval>(1);
    cuda_mem_copy_host_to_device<GateAirEval>(gate_eval, d_gate_eval, 1);

    // ----- Row-tiling of the d_fractions INTERMEDIATE (a ~12 GB @2^24 transient). -----
    // The kernels index every trace read by GLOBAL row and only the fraction pointer is
    // biased; numerators is full-domain regardless. tree0/tree1/tree2 are read whole from
    // their resident device buffers.
    unsigned tile_rows = gate_air_resolve_tile_rows();
    if (tile_rows > eval_domain_size) tile_rows = eval_domain_size;

    Fraction *d_fractions =
        cuda_malloc<Fraction>((size_t)tile_rows * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_gate_air");

    // gate_air uses `finalize_logup_in_pairs` => batching[i] = i/2. For logup_counts=10 that is
    // 5 full pairs (batches 0..4) = 5 batches (no singleton tail).
    // The post_kernel folds each pair, reads the interaction cumsum mask (offset
    // 0 for full batches; offsets {0,-1} + cumsum_shift for the last), and adds
    // `diff*denom - num`. This is EXACTLY finalize_logup_in_pairs (lib.rs:185-221).
    // It is wired now but is NOT trusted until the box accumulator-diff is zero
    // (Phase 2 validation).
    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) batching[i] = i / 2;
    unsigned last_batch = batching[logup_counts - 1];

    // ----- Tiled pre_kernel + post_kernel over eval-domain row tiles -----
    for (unsigned tile_start = 0; tile_start < eval_domain_size; tile_start += tile_rows) {
        unsigned this_tile = eval_domain_size - tile_start;
        if (this_tile > tile_rows) this_tile = tile_rows;

        // Bias the fraction pointer back by tile_start*logup_counts so the
        // evaluator's GLOBAL-row index (row*logup_counts) writes/reads into the
        // tile-local buffer slot (row-tile_start)*logup_counts. The kernels never
        // touch d_fractions outside [0, this_tile*logup_counts) because every
        // active thread has row in [tile_start, tile_start+this_tile).
        Fraction *frac_biased = d_fractions - (size_t)tile_start * logup_counts;

        // tree0/tree1 are read whole from their resident device pointer tables; the
        // kernel indexes each by GLOBAL row.
        m31 **pre_trace0 = d_trace0;
        m31 **pre_trace1 = d_trace1;

        int block_dim = this_tile < GATE_AIR_THREAD_COUNT_MAX
            ? (int)this_tile : GATE_AIR_THREAD_COUNT_MAX;
        int num_blocks = (this_tile + block_dim - 1) / block_dim;

        // ----- pre_kernel (PHASE 1: algebraic + relation emits) -----
        if (use_assert_evaluator) {
            evaluate_gate_air_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, pre_trace0, pre_trace1, random_coeff_powers,
                domain_log_size, eval_domain_log_size, d_gate_eval, cumsum_shift,
                frac_biased, logup_counts, constraint_index_array,
                tile_start, this_tile);
        } else {
            evaluate_gate_air_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, pre_trace0, pre_trace1, random_coeff_powers,
                domain_log_size, eval_domain_log_size, d_gate_eval, cumsum_shift,
                frac_biased, logup_counts, constraint_index_array,
                tile_start, this_tile);
        }
        // Same-stream ordering serializes this tile's pre before its post; the single
        // post-loop sync covers all kernels before any host read. Keep the non-blocking
        // launch-error check.
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // tree2 is read whole from its resident device pointer table.
        m31 **post_trace2 = d_trace2;

        // ----- post_kernel (PHASE 2 SOUNDNESS GATE: LogUp pair-batches) -----
        // tree2 is resident: the post_kernel takes the scattered `-1` cumsum read
        // (tree2_shifted=false).
        if (use_assert_evaluator) {
            evaluate_gate_air_post_kernel_tiled<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, post_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile, /* tree2_shifted */ false);
        } else {
            evaluate_gate_air_post_kernel_tiled<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
                numerators, frac_biased, constraint_index_array, post_trace2,
                random_coeff_powers, domain_log_size, eval_domain_log_size,
                logup_counts, last_batch, cumsum_shift,
                tile_start, this_tile, /* tree2_shifted */ false);
        }
        // STAGED path keeps the per-tile sync (its ping-pong H2D depends on it).
        // RESIDENT path drops the sync (same-stream ordering serializes this tile's
        // post before the next tile's pre / d_fractions reuse) but keeps the
        // The single post-loop sync below covers all kernels before any host read.
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    // ----- finalize: quotient = numerators[row] * denom_inv[row>>trace_log] -----
    // Pointwise per row over the FULL eval domain (the tiling above only bounded
    // the d_fractions transient; numerators[] is now fully populated). Identical to
    // accumulate_pointwise_cpu (component_prover.rs:288-289).
    int finalize_block_dim = eval_domain_size < GATE_AIR_THREAD_COUNT_MAX
        ? (int)eval_domain_size : GATE_AIR_THREAD_COUNT_MAX;
    int finalize_num_blocks = (eval_domain_size + finalize_block_dim - 1) / finalize_block_dim;
    generic_constraint_quotients_finalize_kernel<<<finalize_num_blocks, finalize_block_dim, 0, stream>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        numerators, denominator_inverses,
        domain_log_size, eval_domain_log_size, should_accumulate);

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_gate_air");

    cuda_free_memory(d_trace0);
    cuda_free_memory(d_trace1);
    cuda_free_memory(d_trace2);
    cuda_free_memory(numerators);
    cuda_free_memory(d_gate_eval);
    cuda_free_memory(d_fractions);
    cuda_free_memory(constraint_index_array);
}
