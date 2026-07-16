//! Base (per-shard) gate_air prover, extracted from `main.rs`.
//!
//! Holds the base-proof pipeline (trace-gen + commit + `prove_ex`) that produces one
//! `ExtendedStarkProof` per shard, plus the shard-invariant precompute and the tree-0 program /
//! boundary / rc supply tables the base proof commits. Mirrors how `leaf.rs` holds the leaf prover.
//! Byte-identical to the pre-extraction inline code (pure code move).

use crate::*;

use anyhow::{bail, Result};
#[cfg(feature = "cuda")]
use anyhow::Context;
use num_traits::One;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::proof::ExtendedStarkProof;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove_ex, CommitmentSchemeProver};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof_of_work::GrindOps;
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkEval, RelationEntry, Relation,
};

// ----------------------------------------------------------------------------
// Program-consistency table (the single hidden program)
// ----------------------------------------------------------------------------
//
// One row per program slot (gate) i in 0..n_gates, padded to a power of two.
// Row i stores the canonical op tuple (opcode_scalar, target, ctrl_a, ctrl_b)
// as WITNESS (the hidden program) plus a multiplicity = number of executions of
// slot i = samples*K (every shot runs the program K times). Inactive controls
// canonicalise to 0, matching ReadCols::inactive().q == 0 on the use side. Slot
// index is a PREPROCESSED column (public row layout). Padding rows carry
// multiplicity 0, so their op contents are inert (never addressed: pc_in_prog
// stays in 0..n_gates).
pub(crate) struct ProgramTable {
    pub(crate) log_size: u32,
    pub(crate) slot: Vec<u32>,          // preprocessed slot index 0..size
    pub(crate) opcode_scalar: Vec<u32>, // witness
    pub(crate) target: Vec<u32>,        // witness
    pub(crate) ctrl_a: Vec<u32>,        // witness
    pub(crate) ctrl_b: Vec<u32>,        // witness
    pub(crate) multiplicity: Vec<u32>,  // witness: samples*K on real slots, 0 on padding
}

/// Log-size of the program table (one row per gate, padded to a power of two, floored at LANE_COUNT).
/// Pure function of `n_gates` (shard-invariant), so it can be recovered without the table itself.
pub(crate) fn program_log_size(n_gates: usize) -> u32 {
    n_gates.next_power_of_two().max(LANE_COUNT).ilog2()
}

pub(crate) fn build_program_table(gates: &[Gate], samples: usize, k: usize) -> ProgramTable {
    let n_gates = gates.len();
    let padded = n_gates.next_power_of_two().max(LANE_COUNT);
    let log_size = padded.ilog2();
    let mut slot = vec![0u32; padded];
    let mut opcode_scalar = vec![0u32; padded];
    let mut target = vec![0u32; padded];
    let mut ctrl_a = vec![0u32; padded];
    let mut ctrl_b = vec![0u32; padded];
    let mut multiplicity = vec![0u32; padded];
    let mult = (samples * k) as u32;
    for (i, g) in gates.iter().enumerate() {
        slot[i] = i as u32;
        let (sc, a_active, b_active) = match g.opcode {
            OP_NOP => (0u32, false, false),
            OP_NOT => (1u32, false, false),
            OP_CNOT => (2u32, true, false),
            OP_TOFFOLI => (3u32, true, true),
            _ => (0u32, false, false),
        };
        opcode_scalar[i] = sc;
        target[i] = g.target as u32;
        ctrl_a[i] = if a_active { g.ctrl_a as u32 } else { 0 };
        ctrl_b[i] = if b_active { g.ctrl_b as u32 } else { 0 };
        multiplicity[i] = mult;
    }
    // Padding slots keep an in-range index sequence (inert; multiplicity 0).
    for (i, s) in slot.iter_mut().enumerate().skip(n_gates) {
        *s = i as u32;
    }
    ProgramTable {
        log_size,
        slot,
        opcode_scalar,
        target,
        ctrl_a,
        ctrl_b,
        multiplicity,
    }
}
#[derive(Clone, Copy, Default)]
pub(crate) struct BoundaryRow {
    pub(crate) shot_id: u32, // preprocessed
    pub(crate) addr: u32,    // preprocessed (Seq 0..511, repeating per shot)
    pub(crate) x: u32,       // witness (init value, 1 bit)
    pub(crate) y: u32,       // witness (final value, 1 bit)
    pub(crate) ts_last: u32, // witness (last ts at this addr this shot; 0 if untouched)
}

/// Flat list of `n_shots * N_QUBITS` boundary rows, plus the padded power-of-two size.
pub(crate) struct BoundaryTable {
    pub(crate) rows: Vec<BoundaryRow>,
    pub(crate) log_size: u32,
    pub(crate) n_shots: usize,
}

impl BoundaryTable {
    pub(crate) fn new(n_shots: usize) -> Self {
        let real = n_shots * N_QUBITS;
        let padded = real.next_power_of_two().max(LANE_COUNT);
        let mut rows = vec![BoundaryRow::default(); padded];
        // Pre-fill positional (shot, addr) for real rows so untouched-simulation shots still
        // carry a valid tuple; simulate_shot overwrites the witness fields (x, y, ts_last).
        for (i, r) in rows.iter_mut().enumerate().take(real) {
            r.shot_id = (i / N_QUBITS) as u32;
            r.addr = (i % N_QUBITS) as u32;
        }
        Self {
            rows,
            log_size: padded.ilog2(),
            n_shots,
        }
    }

    /// Mutable per-shot chunks of exactly `N_QUBITS` rows (for parallel simulation fill).
    pub(crate) fn per_shot_mut(&mut self) -> Vec<&mut [BoundaryRow]> {
        let real = self.n_shots * N_QUBITS;
        self.rows[..real].chunks_mut(N_QUBITS).collect()
    }
}

/// Supply table for the ts-ordering range-check. `val` is PREPROCESSED (the table membership,
/// shard-invariant, `val[i] = i`); `multiplicity` is WITNESS (count of real `d` lookups landing on
/// that row). The table has `2^log_size` rows enumerating exactly `[0, 2^log_size)`.
pub(crate) struct RcTable {
    pub(crate) log_size: u32,
    pub(crate) val: Vec<u32>,          // preprocessed: val[i] = i for i in [0, 2^log_size)
    pub(crate) multiplicity: Vec<u32>, // witness
}

impl RcTable {
    pub(crate) fn new(rc_log: u32) -> Self {
        let size = 1usize << rc_log;
        // Single block enumerating exactly [0, 2^rc_log): val[i] = i (every row a genuine member).
        let val = (0..size as u32).collect();
        Self {
            log_size: rc_log,
            val,
            multiplicity: vec![0u32; size],
        }
    }

    /// Count the single `d` lookup of one active access into the multiplicity column. `d` indexes the
    /// table directly since `val[i] = i` (row_of(d) == d).
    #[inline]
    fn count_access(&mut self, a: &AccessCols) {
        self.multiplicity[a.d as usize] += 1;
    }
}

/// Build the rc supply table and its multiplicity column by counting every ACTIVE access's single
/// `d` lookup. An access is active iff its owning gate fires it: the target on every real row, a
/// control iff the opcode uses it. Padding rows (enabler = 0) emit no lookup, so they are skipped.
pub(crate) fn build_rc_table(rows: &[Row], rc_log: u32) -> RcTable {
    let mut table = RcTable::new(rc_log);
    for r in rows {
        if r.enabler == 0 {
            continue;
        }
        // target is active on every real row.
        table.count_access(&r.target);
        // controls active per opcode (a_active = is_cnot + is_toffoli, b_active = is_toffoli).
        if r.is_cnot + r.is_toffoli == 1 {
            table.count_access(&r.ctrl_a);
        }
        if r.is_toffoli == 1 {
            table.count_access(&r.ctrl_b);
        }
    }
    table
}

// ----------------------------------------------------------------------------
// Dynamic range-check tables (preprocessed)
// ----------------------------------------------------------------------------
//
// T_lo = {(pos, v): 0 <= v < 2^pos,        pos in 0..16}
// T_hi = {(pos, v): 0 <= v < 2^(15-pos),   pos in 0..16}
// Each has sum_pos 2^pos / 2^(15-pos) = 2^16 - 1 entries; padded to 2^16 rows.
// Padding rows reuse the (pos=0, v=0) tuple (a genuine table member) so the
// LogUp membership math stays sound: extra supply of an existing tuple is fine
// as long as the multiplicity-trace counts only real reads (which it does).

/// Maps (pos, value) -> a row index in the flattened range-check table. The per-address +1 counter
/// fix (and later the single-`d` rc lookup) removed the range-check LOOKUP; this now survives ONLY so
/// the CUDA trace-gen glue (`gpu_flat_inputs` -> `off_lo`/`off_hi`) can keep its device-buffer layout
/// unchanged (the K1/K4 kernels IGNORE those offsets). The pos/val columns and `row()` accessor are
/// unused by the CPU path (hence `dead_code`). Size is `2^LIMB_BITS`, independent of the now-dynamic
/// rc supply table (this index is per-`pos`-block, sum_pos 2^pos = 2^16 - 1 <= 2^LIMB_BITS rows).
#[allow(dead_code)]
pub(crate) struct RcIndex {
    pub(crate) pos_col: Vec<u32>,
    pub(crate) val_col: Vec<u32>,
    // offset[pos] = first row index for this pos block.
    pub(crate) offset: [usize; LIMB_BITS + 1],
}

impl RcIndex {
    /// `bound(pos)` returns the exclusive value bound for this pos.
    fn build(bound: impl Fn(usize) -> u32) -> Self {
        let size = 1usize << LIMB_BITS;
        let mut pos_col = vec![0u32; size];
        let mut val_col = vec![0u32; size];
        let mut offset = [0usize; LIMB_BITS + 1];
        let mut row = 0usize;
        // `pos` indexes `offset` AND drives `bound(pos)` / the written `pos` value, so the range loop
        // is intentional (no slice to iterate).
        #[allow(clippy::needless_range_loop)]
        for pos in 0..LIMB_BITS {
            offset[pos] = row;
            let b = bound(pos);
            for v in 0..b {
                pos_col[row] = pos as u32;
                val_col[row] = v;
                row += 1;
            }
        }
        offset[LIMB_BITS] = row;
        debug_assert!(row <= size);
        // Remaining rows stay (pos=0, v=0): a valid member, inert padding.
        Self {
            pos_col,
            val_col,
            offset,
        }
    }
}

pub(crate) fn build_rc_lo() -> RcIndex {
    RcIndex::build(|pos| 1u32 << pos)
}
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

        // Preprocessed (tree0) columns: SHARD-INVARIANT POSITIONAL values (verifier-pinned).
        //   enabler   = real-row indicator (1 on real rows, 0 on padding)
        //   shot_id   = row / (k*n_gates)   (partitions the per-qubit chains per shot)
        //   pc        = row % (k*n_gates)   (per-shot PROGRAM COUNTER, program order) — the access
        //               timestamp is the inlined affine `ts = pc + 1` (verifier-pinned program order
        //               is what forbids reordering an address's accesses).
        //   pc_in_prog= pc mod n_gates      (the program slot each execution row addresses)
        let enabler = eval.get_preprocessed_column(pp_id("gate_enabler"));
        let shot_id = eval.get_preprocessed_column(pp_id("gate_shot_id"));
        let pc = eval.get_preprocessed_column(pp_id("gate_pc"));
        let pc_in_prog = eval.get_preprocessed_column(pp_id("gate_pc_in_prog"));

        let is_nop = eval.next_trace_mask();
        let is_not = eval.next_trace_mask();
        let is_cnot = eval.next_trace_mask();
        let is_toffoli = eval.next_trace_mask();

        // target access: addr, prev_ts, v_before (ts inlined = pc+1; v_after inlined = v_before+delta).
        let target = access_masks(&mut eval);
        let ctrl_a = access_masks(&mut eval);
        let ctrl_b = access_masks(&mut eval);

        let ab = eval.next_trace_mask();
        let fire = eval.next_trace_mask();
        let delta = eval.next_trace_mask();

        // --- Opcode booleanity + one-hot sum = enabler. ---
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

        // The target's post-gate value `v_after` is NOT a witness column: it equals
        // `v_before + delta` (the pinned gate-apply equality, now inlined). `delta = fire*(1-2*v_before)`
        // is enforced below, so `v_after = v_before + delta` remains a bit and carries the write forward.
        let t_bit = target.v.clone(); // v_before
        let v_after = t_bit.clone() + delta.clone();

        // --- Value booleanity (memory values are 1 bit). The target's written value is the derived
        // `v_after = v_before + delta`; booleanity on it keeps the memory value a bit. ---
        for v in [&target.v, &v_after, &ctrl_a.v, &ctrl_b.v] {
            eval.add_constraint(v.clone() * (v.clone() - one.clone()));
        }

        // --- Gate-apply on the memory values. ---
        let a_bit = ctrl_a.v.clone();
        let b_bit = ctrl_b.v.clone();
        // ab = v_a * v_b.
        eval.add_constraint(ab.clone() - a_bit.clone() * b_bit.clone());
        // fire = is_not + is_cnot*v_a + is_toffoli*ab.
        eval.add_constraint(
            fire.clone()
                - is_not.clone()
                - is_cnot.clone() * a_bit.clone()
                - is_toffoli.clone() * ab.clone(),
        );
        // v_after = v_before XOR fire; delta = v_after - v_before = fire*(1 - 2*v_before). (The
        // `v_after - v_before - delta = 0` equality is now vacuous — v_after is defined as v_before+delta.)
        eval.add_constraint(delta.clone() - fire.clone() + t_bit.clone() * fire.clone() * two);

        // The access timestamp is the affine `ts = pc + 1` of the preprocessed pc, shared by all three
        // accesses of the step (no witness column, no per-access slot). Computed once, inlined below.
        let ts = pc.clone() + one.clone();

        // --- Qubit-memory chain: per active access Use(predecessor) + Yield(successor). ---
        // Target (always active iff enabler): Use(prev_ts, v_before), Yield(ts=pc+1, v_after=v_before+delta).
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

        // --- ts-ordering RANGE-CHECK LOOKUPs (emitted here so their relation-batch order is
        // qubitmem-pairs (6 terms), then the 3 single-`d` rc lookups (target, ctrl_a, ctrl_b), then
        // program; finalize-in-pairs folds the trailing 3 rc + 1 program into 2 batches:
        // (rc_t, rc_a) and (rc_b, program). Mirrored exactly by `gen_main_interaction` and MainGate. ---
        add_rc_lookup(&mut eval, &self.elements.rc, &target, enabler.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_a, a_active.clone());
        add_rc_lookup(&mut eval, &self.elements.rc, &ctrl_b, b_active.clone());

        // --- ts-ordering: RANGE-CHECK prev_ts < ts (soundness-critical). ---
        // The old PIN constraint `active*(ts - (pc*TS_STRIDE + slot)) = 0` is GONE: ts is now
        // structurally `pc + 1` (inlined), so the pin is vacuous. Only the RANGE reconstruction
        // remains: active*((pc+1) - prev_ts - 1 - d) = 0 pins the witness `d` to
        // pc - prev_ts, range-checked by the single rc-table lookup above =>
        // d ∈ [0, 2^TS_RC_BITS), i.e. prev_ts < ts. Together with the structurally program-ordered ts
        // and the LogUp chain balance this forces a FORWARD DAG (no stale-read cycle): each read
        // observes the program-order-last write. Inactive accesses (active=0) unconstrained.
        add_ts_range(&mut eval, &ts, &target, enabler.clone());
        add_ts_range(&mut eval, &ts, &ctrl_a, a_active);
        add_ts_range(&mut eval, &ts, &ctrl_b, b_active);

        // --- Program-consistency (use side, +enabler). ---
        // opcode_scalar = is_not + 2*is_cnot + 3*is_toffoli (NOP -> 0). Addresses are the access
        // addr columns (0 for inactive controls, matching the program table's canonical zero).
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

struct AccessMasks<F> {
    addr: F,
    prev_ts: F,
    v: F,
    d: F, // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LOG_SIZE))
}

fn access_masks<E: EvalAtRow>(eval: &mut E) -> AccessMasks<E::F> {
    // ts is NOT a column — it is the inlined `pc + 1`. Per-access columns: addr, prev_ts, v, d.
    let addr = eval.next_trace_mask();
    let prev_ts = eval.next_trace_mask();
    let v = eval.next_trace_mask();
    // d follows v (matches `cell_at`'s per-access column order).
    let d = eval.next_trace_mask();
    AccessMasks {
        addr,
        prev_ts,
        v,
        d,
    }
}

/// Emit the chain Use(predecessor) + Yield(successor) pair for one access, gated by `active`.
/// `ts` is the access's inlined timestamp expression (`pc + 1`, shared across the step); `v_out` is
/// the value written forward (v_after = v_before+delta for the target, v for a control read).
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
    // Yield successor: -active / (shot, addr, ts=pc+1, v_after).
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

/// Emit the range-check reconstruction for one access, gated by `active`:
///   RANGE: active*(ts - prev_ts - 1 - d) = 0 — pins the witness `d` column to the diff
///          d = ts - prev_ts - 1 = pc - prev_ts. `d` itself is range-checked by the rc-table LOOKUP
///          (`add_rc_lookup`), not here, so d ∈ [0, 2^RC_LOG_SIZE) (prev_ts < ts). `ts` is the inlined
///          `pc + 1` expression.
/// The old PIN constraint is removed (ts is structurally pc+1, so the pin is vacuous). The
/// reconstruction is gated by `active`; the `d` LOOKUP is also gated by `active` (an inactive access
/// emits no rc term). Inactive accesses (active = 0) leave prev_ts/d free.
fn add_ts_range<E: EvalAtRow>(eval: &mut E, ts: &E::F, a: &AccessMasks<E::F>, active: E::F) {
    let one = E::F::one();
    let d = ts.clone() - a.prev_ts.clone() - one;
    eval.add_constraint(active * (d - a.d.clone()));
}

/// Emit the single rc-table range-check LOOKUP for one access, gated by `active`: `d` is looked up as
/// (TAG_RC, d); the rc supply table supplies each in-range value. One term/access (mirrored by
/// `gen_main_interaction` and the in-circuit MainGate).
fn add_rc_lookup<E: EvalAtRow>(eval: &mut E, rc: &GateRel, a: &AccessMasks<E::F>, active: E::F) {
    let tag = E::F::one() * BaseField::from_u32_unchecked(TAG_RC);
    eval.add_to_relation(RelationEntry::new(
        rc,
        E::EF::from(active),
        &[tag, a.d.clone()],
    ));
}

/// Phase-2 packing of the main trace. The scalar `Vec<Row>` (filled in parallel
/// over shots in `build_rows`) is packed into `PackedM31` columns. Columns are
/// fully independent, so we pack IN PARALLEL OVER COLUMNS: each task owns one
/// column's whole `Vec<PackedM31>` and fills every packed word for it. No two
/// threads ever touch the same column or the same packed word, so a shot block
/// (k*n_gates rows) being non-16-aligned can never cause a packed-word race. The
/// cell→column mapping is identical to the old serial `Col::set(row_idx, ..)`
/// fill, so the trace is bit-identical.
pub(crate) fn generate_main_trace(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    use rayon::prelude::*;
    use stwo::prover::backend::simd::column::BaseColumn;

    let n_vec = padded_rows / LANE_COUNT;
    let pad = Row::padding();
    let domain = CanonicCoset::new(log_n_rows).circle_domain();

    (0..TRACE_COLUMNS)
        .into_par_iter()
        .map(|col| {
            let col_data: Vec<PackedM31> = (0..n_vec)
                .map(|vec_row| {
                    PackedM31::from_array(std::array::from_fn(|lane| {
                        let idx = vec_row * LANE_COUNT + lane;
                        let row = rows.get(idx).unwrap_or(&pad);
                        BaseField::from_u32_unchecked(cell_at(row, col))
                    }))
                })
                .collect();
            CircleEvaluation::<TraceBackend, _, BitReversedOrder>::new(
                domain,
                BaseColumn::from_simd(col_data),
            )
        })
        .collect()
}
/// Shard-invariant base-proof precompute. Built ONCE before the shard loop and shared (by `Arc`)
/// across every shard's base proof, so the work that does not depend on the shard's secret shots is
/// done exactly once instead of N times:
///
/// 1. tree-0 (the 13-column preprocessed commitment) — interpolated + LDE + Merkle-committed ONCE
///    and reused via `CommitmentSchemeProver::commit_tree(MaybeOwned::Borrowed(..))` (re-mixes the
///    SAME root into each shard's fresh channel — transcript unchanged).
/// 2. twiddles — `precompute_twiddles` once, shared by reference.
/// 3. the N1 program table — identical across shards (multiplicity = shots_per_shard*k constant).
/// 4. (cuda) the N3 device-resident gate-list / RcIndex-offset buffers — uploaded ONCE.
///
/// N4 (the GATE_SIM + INTERACTION PTX modules) is a process-level OnceLock cache in gpu_tracegen,
/// not part of this struct.
// `config`/`boundary`/`padded_rows`/`log_n_rows` are read by the cuda `build_device_parts` and by the
// debug/test-only `assert_tree0_matches_rebuild`; in a NON-cuda RELEASE build (assert compiled out)
// they are populated-but-unread, so allow dead_code in exactly that config (warning stays live
// everywhere else to catch genuine dead fields).
#[cfg_attr(not(any(debug_assertions, test, feature = "cuda")), allow(dead_code))]
pub(crate) struct BaseProverPrecompute {
    config: stwo::core::pcs::PcsConfig,
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    /// Shared N1 program table (constant multiplicity across shards).
    program: ProgramTable,
    /// Shard-invariant boundary table SHAPE (positional (shot, addr); witness x/y/ts_last are
    /// per-shard, but the preprocessed columns depend only on the shape, which is shard-invariant).
    boundary: BoundaryTable,
    /// Fixed shard shape (every shard holds `shots_per_shard` shots → same row count).
    padded_rows: usize,
    log_n_rows: u32,
    /// Dynamic rc-table log-size (= ceil(log2(k*n_gates))); shard-invariant. Used to REBUILD tree0
    /// (device n != 0 replica / the debug rebuild check) with the same rc membership sizing.
    rc_log: u32,
    /// (cuda) shape needed to REBUILD the device-resident parts on a producer's device (device != 0).
    #[cfg(feature = "cuda")]
    max_log_size: u32,
    #[cfg(feature = "cuda")]
    n_gates: usize,
    /// (cuda) device-resident N3 inputs uploaded once ON DEVICE 0: gate list + RcIndex lo/hi offsets.
    /// Only the per-shard `x_states` upload remains in `prove_base_shard`. For devices != 0 the
    /// equivalent buffers live in `device_parts` (built lazily per device from the owned inputs).
    #[cfg(feature = "cuda")]
    d_gates: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_hi: cudarc::driver::CudaSlice<u32>,
    // MULTI-GPU ("option A"): the device-resident precompute (tree0 + twiddles + N3 buffers) is bound
    // to the device it was built on. Device 0's copy is the eager fields above (built + soundness-
    // asserted in `new`). For a producer thread on device n != 0, `device_parts()` lazily REBUILDS
    // the same device-resident parts on device n (from the owned host inputs below) and caches them
    // in slot n. tree0 is shard-invariant, so a device-n rebuild is byte-identical to device 0's —
    // this is just per-device REPLICATION of the same precompute, not different data. Slot 0 stays
    // empty (device 0 uses the eager fields); slots 1..MAX filled on demand.
    #[cfg(feature = "cuda")]
    rows0: Vec<Row>,
    #[cfg(feature = "cuda")]
    gates_flat: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_lo: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_hi: Vec<u32>,
    #[cfg(feature = "cuda")]
    device_parts: [std::sync::OnceLock<DevicePrecompute>; MAX_BASE_GPUS],
}

/// Number of base GPUs supported by the per-device precompute cache (matches gpu_tracegen's cap).
#[cfg(feature = "cuda")]
const MAX_BASE_GPUS: usize = 16;

/// (cuda) The device-resident half of the base precompute, bound to ONE device. Built once per
/// device (device 0 eagerly in `BaseProverPrecompute::new`, devices != 0 lazily in `device_parts`).
/// tree0/twiddles are re-derived identically on each device (shard-invariant inputs), so replicating
/// them per device does not change any committed value.
#[cfg(feature = "cuda")]
struct DevicePrecompute {
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: cudarc::driver::CudaSlice<u32>,
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    d_off_hi: cudarc::driver::CudaSlice<u32>,
}

impl BaseProverPrecompute {
    /// Build the precompute from shard 0's shape (`program0`, `rows0`). `rows0`/`program0` are
    /// shard-invariant (see [`build_tree0_columns`]), so the cached tree-0 is valid for every shard.
    ///
    /// SOUNDNESS GUARD: the committed tree-0 root is exactly what each shard's transcript mixes (via
    /// `commit_tree`), so the caller asserts it equals an independently rebuilt shard-0 root + matches
    /// column count/sizes (the load-bearing check in `assert_tree0_matches_rebuild`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: stwo::core::pcs::PcsConfig,
        max_log_size: u32,
        program0: ProgramTable,
        rows0: &[Row],
        boundary0: BoundaryTable,
        padded_rows: usize,
        log_n_rows: u32,
        n_gates: usize,
        rc_log: u32,
        #[cfg(feature = "cuda")] gates_flat: &[u32],
        #[cfg(feature = "cuda")] off_lo: &[u32],
        #[cfg(feature = "cuda")] off_hi: &[u32],
    ) -> Result<Self> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        // Scratch pool used only for tree-0's build (the committed tree owns its own polynomials, so
        // the pool can be dropped afterwards; per-shard witness trees use their scheme's own pool).
        let pool = BaseColumnPool::<ProverBackend>::new();

        // Build + commit tree-0 ONCE. `lifting_log_size = None` lets CommitmentTreeProver derive it
        // from the columns' max domain size, exactly as the per-shard `tree_builder().commit()` path
        // does (it uses the same scheme config, which has lifting_log_size from leaf_pcs_config). The
        // store flag is FALSE to match the base proof's barycentric-OODS path (no stored coeffs).
        let cols = build_tree0_columns(
            &program0,
            rows0,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            &boundary0,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            config.fri_config.log_blowup_factor,
            &twiddles,
            false, // store_polynomials_coefficients: barycentric OODS path (matches base proof)
            config.lifting_log_size,
            &pool,
        );

        #[cfg(feature = "cuda")]
        let dev = gpu_tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        #[cfg(feature = "cuda")]
        let d_gates = dev
            .htod_copy(gates_flat.to_vec())
            .map_err(|e| anyhow::anyhow!("htod gates (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_lo = dev
            .htod_copy(off_lo.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_lo (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_hi = dev
            .htod_copy(off_hi.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_hi (precompute): {e}"))?;

        Ok(Self {
            config,
            twiddles,
            tree0,
            program: program0,
            boundary: boundary0,
            padded_rows,
            log_n_rows,
            rc_log,
            #[cfg(feature = "cuda")]
            max_log_size,
            #[cfg(feature = "cuda")]
            n_gates,
            #[cfg(feature = "cuda")]
            d_gates,
            #[cfg(feature = "cuda")]
            d_off_lo,
            #[cfg(feature = "cuda")]
            d_off_hi,
            // Owned rebuild inputs so `device_parts` can replicate the device-resident parts on a
            // producer thread's device (n != 0). Cheap: shard-0 rows + the small N3 flat arrays.
            #[cfg(feature = "cuda")]
            rows0: rows0.to_vec(),
            #[cfg(feature = "cuda")]
            gates_flat: gates_flat.to_vec(),
            #[cfg(feature = "cuda")]
            off_lo: off_lo.to_vec(),
            #[cfg(feature = "cuda")]
            off_hi: off_hi.to_vec(),
            // Slot 0 stays empty (device 0 uses the eager `twiddles`/`tree0`/`d_*` fields above);
            // slots 1..MAX are filled lazily by `device_parts` on first use from each device's thread.
            #[cfg(feature = "cuda")]
            device_parts: [const { std::sync::OnceLock::new() }; MAX_BASE_GPUS],
        })
    }

    /// (cuda) Build the DEVICE-RESIDENT precompute parts (twiddles + tree0 + N3 buffers) on the
    /// CALLING thread's current device, from the shard-invariant host inputs. Same construction as
    /// `new` (byte-identical tree0), factored so device-0 (`new`) and device-n (`device_parts`) share
    /// it. The caller must have already bound its device (via gpu_tracegen::set_base_gpu / cuda_device).
    #[cfg(feature = "cuda")]
    fn build_device_parts(&self) -> Result<DevicePrecompute> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(self.max_log_size + 1 + self.config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let pool = BaseColumnPool::<ProverBackend>::new();
        let cols = build_tree0_columns(
            &self.program,
            &self.rows0,
            self.padded_rows,
            self.log_n_rows,
            self.n_gates,
            self.rc_log,
            &self.boundary,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            self.config.fri_config.log_blowup_factor,
            &twiddles,
            false,
            self.config.lifting_log_size,
            &pool,
        );
        let dev = gpu_tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        let d_gates = dev
            .htod_copy(self.gates_flat.clone())
            .map_err(|e| anyhow::anyhow!("htod gates (device precompute): {e}"))?;
        let d_off_lo = dev
            .htod_copy(self.off_lo.clone())
            .map_err(|e| anyhow::anyhow!("htod off_lo (device precompute): {e}"))?;
        let d_off_hi = dev
            .htod_copy(self.off_hi.clone())
            .map_err(|e| anyhow::anyhow!("htod off_hi (device precompute): {e}"))?;
        Ok(DevicePrecompute {
            twiddles,
            tree0,
            d_gates,
            d_off_lo,
            d_off_hi,
        })
    }

    /// (cuda) The device-resident precompute for the CALLING thread's base GPU ordinal. Device 0
    /// returns the eager fields built in `new` (byte-identical to the single-GPU path). Devices != 0
    /// lazily build + cache their own replica on FIRST use from that device's producer thread. tree0
    /// is shard-invariant, so every device's replica commits the identical root.
    #[cfg(feature = "cuda")]
    fn device_parts(&self) -> DevicePartsRef<'_> {
        let ord = gpu_tracegen::base_gpu_ordinal();
        if ord == 0 {
            return DevicePartsRef {
                twiddles: &self.twiddles,
                tree0: &self.tree0,
                d_gates: &self.d_gates,
                d_off_lo: &self.d_off_lo,
                d_off_hi: &self.d_off_hi,
            };
        }
        let slot = self
            .device_parts
            .get(ord)
            .unwrap_or_else(|| panic!("base gpu ordinal {ord} >= {MAX_BASE_GPUS}"));
        // Build once per device; the build runs on THIS producer thread (already bound to device
        // `ord`). Fatal on failure (matches `new`'s `?` — a broken precompute cannot proceed).
        let parts = slot.get_or_init(|| {
            self.build_device_parts()
                .unwrap_or_else(|e| panic!("device {ord} precompute build failed: {e}"))
        });
        DevicePartsRef {
            twiddles: &parts.twiddles,
            tree0: &parts.tree0,
            d_gates: &parts.d_gates,
            d_off_lo: &parts.d_off_lo,
            d_off_hi: &parts.d_off_hi,
        }
    }
}

/// (cuda) Borrowed view of the device-resident precompute parts (device 0's eager fields or a
/// device-n cached replica), so `prove_base_shard` reads them uniformly regardless of ordinal.
#[cfg(feature = "cuda")]
struct DevicePartsRef<'a> {
    twiddles: &'a stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: &'a stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: &'a cudarc::driver::CudaSlice<u32>,
    d_off_lo: &'a cudarc::driver::CudaSlice<u32>,
    d_off_hi: &'a cudarc::driver::CudaSlice<u32>,
}

/// LOAD-BEARING SOUNDNESS CHECK for the base precompute. Independently rebuilds shard 0's tree-0 the
/// OLD way (a fresh throwaway `CommitmentSchemeProver` + `tree_builder().commit()`) and asserts:
///
/// - the cached tree-0 root == the freshly-rebuilt root (the value mixed into each shard channel),
/// - the cached tree-0 column count == the rebuilt column count,
/// - each cached column's committed domain log_size == the rebuilt column's.
///
/// A mismatch (wrong column order / blowup / lifting / sort) aborts before any reused proof is built.
/// Run on shard 0 only (all shards share the shape). `GATE_AIR_NO_BASE_PRECOMPUTE` skips reuse, so
/// this check is a no-op there (the rebuild path is exercised directly per shard).
///
/// Compiled ONLY in debug or test builds: the runtime caller is `#[cfg(debug_assertions)]` and the CI
/// coverage is `tests::tree0_precompute_matches_rebuild`. Absent from the release binary (its cost is
/// a full duplicate tree0 build), so `--release` pays nothing and stays byte-identical.
#[cfg(any(debug_assertions, test))]
pub(crate) fn assert_tree0_matches_rebuild(pc: &BaseProverPrecompute, rows0: &[Row], n_gates: usize) {
    // Rebuild via the exact old path (fresh scheme/channel; columns from the same builder).
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(
            tree0_max_log_size(
                pc.log_n_rows,
                pc.rc_log,
                pc.program.log_size,
                pc.boundary.log_size,
            ) + 1
                + pc.config.fri_config.log_blowup_factor,
        )
        .circle_domain()
        .half_coset,
    );
    let mut scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(pc.config, &twiddles);
    let cols = build_tree0_columns(
        &pc.program,
        rows0,
        pc.padded_rows,
        pc.log_n_rows,
        n_gates,
        pc.rc_log,
        &pc.boundary,
    );
    let n_cols = cols.len();
    let mut tb = scheme.tree_builder();
    tb.extend_evals(to_prover(cols));
    let mut throwaway_channel = Blake2sM31Channel::default();
    tb.commit(&mut throwaway_channel);

    let rebuilt = &scheme.trees[0];
    // 1. Root equality (the value mixed into every shard's transcript).
    assert_eq!(
        pc.tree0.commitment.root(),
        rebuilt.commitment.root(),
        "base-precompute tree0 root != rebuilt shard-0 root (column order/blowup/lifting mismatch)"
    );
    // 2. Column count.
    assert_eq!(
        pc.tree0.polynomials.len(),
        n_cols,
        "base-precompute tree0 column count != rebuilt"
    );
    assert_eq!(
        rebuilt.polynomials.len(),
        n_cols,
        "rebuilt tree0 column count != expected"
    );
    assert_eq!(
        n_cols, N_PREPROCESSED_COLS,
        "tree0 column count != N_PREPROCESSED_COLS"
    );
    // 3. Per-column committed domain sizes (the lifted-Merkle sort order).
    for (i, (a, b)) in pc
        .tree0
        .polynomials
        .iter()
        .zip(rebuilt.polynomials.iter())
        .enumerate()
    {
        assert_eq!(
            a.evals.domain.log_size(),
            b.evals.domain.log_size(),
            "base-precompute tree0 column {i} size != rebuilt"
        );
    }
    eprintln!(
        "gate-air: base-precompute tree0 root-equality OK ({} cols, root matches rebuilt shard-0)",
        n_cols
    );
}
/// SOUNDNESS (base pp-root pin, step 1): recompute the CANONICAL base gate_air preprocessed (tree0)
/// root at BUILD TIME purely from the trusted PUBLIC config — the program table, `k`, `n_gates`,
/// `shots_per_shard`, the shard-invariant row shape, the boundary layout, `rc_log = rc_log_size(k *
/// n_gates)`, and the base blowup. The canonical base preprocessed (tree0) root a base proof commits,
/// recomputed from the trusted PUBLIC shape so it can be compared against a forgeable proof value.
///
/// It must NOT read `base_extended.proof` (the prover's `commitments[0]` — a forgeable value). tree0
/// is SHARD-INVARIANT (every preprocessed column is positional / shape-derived, see
/// [`build_tree0_columns`]), so ANY shard's rows recompute the same root; the caller passes shard 0's
/// rows (already materialized for the base proof). The build mirrors [`BaseProverPrecompute::new`]'s
/// tree0 path exactly (same columns, blowup, lifting, `store=false`), so the recomputed root equals
/// the honest prover's committed base preprocessed root by construction.
///
/// REBUILD-ASSERT GUARD (debug/test only, mirrors [`assert_tree0_matches_rebuild`]): the tree0 root is
/// re-derived via a fresh `CommitmentSchemeProver` + `tree_builder().commit()` and asserted equal, so
/// a column-order / blowup / lifting / sort divergence aborts before the canonical constant is baked.
/// Compiled out of `--release` (byte-identical, pays nothing).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn canonical_base_preprocessed_root(
    program: &ProgramTable,
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    rc_log: u32,
    boundary: &BoundaryTable,
    pcs_config: stwo::core::pcs::PcsConfig,
) -> circuits::blake::HashValue<SecureField> {
    use circuits::blake::HashValue;
    use stwo::prover::mempool::BaseColumnPool;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::CommitmentTreeProver;

    let max_log_size = tree0_max_log_size(log_n_rows, rc_log, program.log_size, boundary.log_size);
    let twiddles = ProverBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let pool = BaseColumnPool::<ProverBackend>::new();
    // Same tree0 build as `BaseProverPrecompute::new` (barycentric-OODS path: store=false).
    let cols = build_tree0_columns(
        program,
        rows,
        padded_rows,
        log_n_rows,
        n_gates,
        rc_log,
        boundary,
    );
    let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
    let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
        polys,
        pcs_config.fri_config.log_blowup_factor,
        &twiddles,
        false,
        pcs_config.lifting_log_size,
        &pool,
    );
    let canonical: HashValue<SecureField> = tree0.commitment.root().into();

    // Rebuild-assert (debug/test only): independent fresh-scheme rebuild must give the same root.
    #[cfg(any(debug_assertions, test))]
    {
        let twiddles_r = ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + pcs_config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let mut scheme = CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            pcs_config,
            &twiddles_r,
        );
        let cols_r = build_tree0_columns(
            program,
            rows,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            boundary,
        );
        let mut tb = scheme.tree_builder();
        tb.extend_evals(to_prover(cols_r));
        let mut throwaway_channel = Blake2sM31Channel::default();
        tb.commit(&mut throwaway_channel);
        assert_eq!(
            tree0.commitment.root(),
            scheme.trees[0].commitment.root(),
            "canonical base tree0 root != independent rebuild (column order/blowup/lifting mismatch)"
        );
    }

    canonical
}

// The base-proof tuple `prove_base_shard` returns. Named so the pipeline producer can send
// it over a channel; `prove_ex` yields `ExtendedStarkProof<MC::H>` with
// `MC::H = Blake2sMerkleHasher`, so this is backend-independent (cuda vs simd).
pub(crate) type BaseShardOutput = (
    ExtendedStarkProof<Blake2sMerkleHasher>,
    Vec<SecureField>,
    u64,
    u32,
    u32,
    u32,
    Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    u32,
);

// Per-shard base proof: same trace-gen + commit + prove_ex pipeline as the single proof, but over
// this shard's shots. Returns the distinct ExtendedStarkProof plus the claim / nonce / log_n_rows the
// leaf needs. Extracted verbatim from the former `prove_base_shard` closure in `main.rs`; the closure
// captures (`gates`, `k`, `n_gates`, `topo`, `rc_lo_index`) are now explicit params. `rc_lo_index` is
// used only under `#[cfg(feature = "cuda")]`, hence the unused-variable allowance on the CPU build.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(crate) fn prove_base_shard(
    precompute: Option<&BaseProverPrecompute>,
    shard_cases: &[TestCase],
    gates: &[Gate],
    k: usize,
    n_gates: usize,
    topo: &recursive_aggregate::TopologyConfig,
    rc_lo_index: &RcIndex,
) -> Result<BaseShardOutput> {
            let shard_samples = shard_cases.len();
            let (rows, boundary) = build_rows(gates, shard_cases, k)?;
            let real_rows = rows.len();
            let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
            let log_n_rows = padded_rows.ilog2();
            // Dynamic rc-table log-size = ceil(log2(k*n_gates)); <= log_n_rows (never raises the floor).
            let rc_log = rc_log_size(k * gates.len());
            let max_log_size = tree0_max_log_size(
                log_n_rows,
                rc_log,
                program_log_size(gates.len()),
                boundary.log_size,
            );
            let base_blowup: u32 = topo.base_log_blowup;
            let config = leaf::leaf_pcs_config(max_log_size, base_blowup);

            // Twiddles: shared by reference from the precompute, else built fresh per shard.
            let owned_twiddles = if precompute.is_none() {
                Some(ProverBackend::precompute_twiddles(
                    CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                        .circle_domain()
                        .half_coset,
                ))
            } else {
                None
            };
            // MULTI-GPU: the device-resident precompute parts (twiddles/tree0/N3) for THIS thread's
            // device. On device 0 these are the eager fields (byte-identical to before); on device
            // n != 0 they are the lazily-built per-device replica. `None` (no-precompute fallback)
            // leaves `dp` None and uses `owned_twiddles` / the per-shard rebuild, unchanged.
            #[cfg(feature = "cuda")]
            let dp = precompute.map(|pc| pc.device_parts());
            #[cfg(feature = "cuda")]
            let twiddles = match &dp {
                Some(dp) => dp.twiddles,
                None => owned_twiddles.as_ref().unwrap(),
            };
            #[cfg(not(feature = "cuda"))]
            let twiddles = match precompute {
                Some(pc) => &pc.twiddles,
                None => owned_twiddles.as_ref().unwrap(),
            };
            // N1 program table: shared from the precompute (constant multiplicity across shards), else
            // rebuilt. `shard_samples == shots_per_shard` for every shard, so the multiplicity matches.
            let owned_program = if precompute.is_none() {
                Some(build_program_table(gates, shard_samples, k))
            } else {
                None
            };
            let program: &ProgramTable = match precompute {
                Some(pc) => &pc.program,
                None => owned_program.as_ref().unwrap(),
            };

            let prover_channel = &mut Blake2sM31Channel::default();
            let channel_salt = 0u32;
            prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
            config.mix_into(prover_channel);
            let mut commitment_scheme = CommitmentSchemeProver::<
                ProverBackend,
                Blake2sM31MerkleChannel,
            >::new(config, twiddles);
            // commitment_scheme.set_store_polynomials_coefficients();  // disabled: barycentric OODS path

            // Tree 0: reuse the precomputed commitment (re-mix the SAME root into THIS shard's
            // channel via `commit_tree` — no NTT/Merkle rebuild), else rebuild it the old way. Under
            // multi-GPU the reused tree0 is THIS device's replica (`dp.tree0`); its root is identical
            // to device 0's (shard-invariant), so the transcript mix is unchanged.
            #[cfg(feature = "cuda")]
            match &dp {
                Some(dp) => {
                    commitment_scheme.commit_tree(MaybeOwned::Borrowed(dp.tree0), prover_channel);
                }
                None => {
                    let pp = build_tree0_columns(
                        program,
                        &rows,
                        padded_rows,
                        log_n_rows,
                        n_gates,
                        rc_log,
                        &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }
            #[cfg(not(feature = "cuda"))]
            match precompute {
                Some(pc) => {
                    commitment_scheme.commit_tree(MaybeOwned::Borrowed(&pc.tree0), prover_channel);
                }
                None => {
                    // Old path: build the (size-sorted) preprocessed columns, then interpolate + LDE +
                    // Merkle-commit them inline (the shard-invariant work this precompute eliminates).
                    let pp = build_tree0_columns(
                        program,
                        &rows,
                        padded_rows,
                        log_n_rows,
                        n_gates,
                        rc_log,
                        &boundary,
                    );
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(to_prover(pp));
                    tree_builder.commit(prover_channel);
                }
            }

            let public_claim = pack_public_claim(&[]);
            prover_channel.mix_felts(&public_claim);

            #[cfg(feature = "cuda")]
            let gpu_tracegen = std::env::var("GATE_AIR_CPU_TRACEGEN").is_err();

            // ts-ordering range-check supply table (multiplicity counted from active-access lookups).
            let rc_table = build_rc_table(&rows, rc_log);

            // Tree 1: main trace + program witness + boundary witness + rc multiplicity.
            let small_main = {
                let mut v = generate_program_witness(program);
                v.extend(generate_boundary_witness(&boundary));
                v.extend(generate_rc_witness(&rc_table));
                v
            };
            let mut tree_builder = commitment_scheme.tree_builder();
            // Holds K1's column-major main-trace device buffer so K4 (interaction) can
            // reuse it instead of re-running K0/K1. `None` on the CPU path.
            #[cfg(feature = "cuda")]
            let mut d_main_cols: Option<cudarc::driver::CudaSlice<u32>> = None;
            #[cfg(feature = "cuda")]
            if gpu_tracegen {
                // N3: gate list + RcIndex offsets are shard-invariant. On the reuse path they are
                // already device-resident in the precompute (uploaded once); only this shard's
                // `x_states` is uploaded here. On the fallback path they're uploaded per shard.
                let mut x_states = Vec::with_capacity(shard_cases.len() * N_LIMBS);
                for c in shard_cases {
                    let bytes =
                        hex::decode(&c.x_hex).context("decoding x_hex for GPU trace-gen")?;
                    x_states.extend_from_slice(&state_to_limbs(&bytes));
                }
                let (main_dev, _qd, _lo, _hi, d_cols) = match &dp {
                    // Multi-GPU: use THIS device's N3 buffers (device 0's eager d_*, or the per-device
                    // replica) — feeding device-0 buffers to a device-n kernel would be an illegal
                    // cross-device access.
                    Some(dp) => gpu_tracegen::gpu_gen_main_trace_device_d(
                        dp.d_gates,
                        &x_states,
                        dp.d_off_lo,
                        dp.d_off_hi,
                        k as u32,
                        n_gates as u32,
                        shard_samples as u32,
                        padded_rows,
                        log_n_rows,
                    ),
                    None => {
                        let (gates_flat, _x, off_lo, off_hi) =
                            gpu_flat_inputs(gates, shard_cases, rc_lo_index, rc_lo_index)?;
                        gpu_tracegen::gpu_gen_main_trace_device(
                            &gates_flat,
                            &x_states,
                            &off_lo,
                            &off_hi,
                            k as u32,
                            n_gates as u32,
                            shard_samples as u32,
                            padded_rows,
                            log_n_rows,
                        )
                    }
                }
                .map_err(|e| anyhow::anyhow!(e))?;
                let mut main_dev = main_dev;
                main_dev.extend(to_prover(small_main));
                tree_builder.extend_evals(main_dev);
                d_main_cols = Some(d_cols);
            } else {
                let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
                main_trace.extend(small_main);
                tree_builder.extend_evals(to_prover(main_trace));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
                main_trace.extend(small_main);
                tree_builder.extend_evals(to_prover(main_trace));
            }
            tree_builder.commit(prover_channel);

            // Hold the ~24 GB main-trace device buffer resident from the tree1 commit through K4.
            // See `MainTrace::from_k1`.
            #[cfg(feature = "cuda")]
            let mut main_k1: Option<gpu_tracegen::MainTrace> = match d_main_cols.take() {
                Some(d_cols) => {
                    Some(gpu_tracegen::MainTrace::from_k1(d_cols).map_err(|e| anyhow::anyhow!(e))?)
                }
                None => None,
            };

            let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
            prover_channel.mix_u64(interaction_pow_nonce);
            let elements = LookupElements::draw(prover_channel);

            #[cfg(feature = "cuda")]
            {
                gate_air_cuda_kernel::register();
                let (z, alpha_powers) = gpu_tracegen::gate_air_relation_m31x4(&elements.qubitmem);
                gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
            }

            // Interaction traces.
            #[cfg(feature = "cuda")]
            let main_interaction_device = if gpu_tracegen {
                let main = main_k1
                    .as_ref()
                    .expect("K1 main-trace buffer must exist on the GPU path");
                let (cols, claimed) = gpu_tracegen::gpu_gen_interaction_device(
                    main,
                    n_gates as u32,
                    padded_rows,
                    log_n_rows,
                    real_rows as u64,
                    (k * n_gates) as u64,
                    &elements,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                Some((cols, claimed))
            } else {
                None
            };
            // K4 done: FREE the ~24 GB resident `d_cols` DEVICE buffer NOW (before tree2), not at
            // end-of-shard, and synchronize so tree2's pool can reserve it. `free_after_k4` consumes
            // the buffer explicitly.
            #[cfg(feature = "cuda")]
            if let Some(m) = main_k1.take() {
                m.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
            }
            #[cfg(feature = "cuda")]
            let (main_interaction, main_sum) = if let Some((_, claimed)) = &main_interaction_device
            {
                (Vec::new(), *claimed)
            } else {
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements)
            };
            #[cfg(not(feature = "cuda"))]
            let (main_interaction, main_sum) =
                gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
            // H_P binding (Fork A): program supply now carries an internal (-mult, TAG_PROGRAM) AND a
            // public (+mult, TAG_PROGRAM_PUB) term, paired into one batch => still 4 interaction cols.
            let (program_interaction, program_sum) =
                gen_program_interaction(program, &elements.program);
            let (boundary_interaction, boundary_sum) =
                gen_boundary_interaction(&boundary, &elements.qubitmem);
            // rc supply: -multiplicity / combine(TAG_RC, val).
            let (rc_interaction, rc_sum) = {
                let el = elements.rc.clone();
                gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
                    el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
                })
            };

            // Phase-3 x/y binding + H_P program binding (Fork A): the base is NOT internally balanced.
            //   - boundary re-keys y to TS_FINAL, leaving B = Σ(+[0,x] − [TS_FINAL,y]);
            //   - program supply adds a public P_pub = Σ mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b)
            //     (its internal -mult/TAG_PROGRAM term cancels main's program demand).
            // So the base's claimed sums net to B + P_pub (not 0). The leaf's public_logup_sum supplies
            // −B (over guessed x/y) AND −P_pub (over guessed program Vars), so the verifier balance
            // forces guessed x/y == committed AND guessed program == committed. rc demand (main) and rc
            // supply (rc_sum) cancel, contributing 0. (stwo's native verify does NOT require
            // Σ claimed_sums == 0; this is a prover self-check.)
            // Prover self-check (DEBUG-ONLY, compiled out in --release): the base's claimed LogUp sums
            // must net to the public terms B + P_pub. Pure tripwire — `b_public`/`p_pub` feed nothing
            // downstream (only `claimed_sums` below is mixed), so gating changes no committed value
            // (release byte-identical). Per-shard cost (two small-table scans) is thus paid only in
            // debug. CI coverage: `tests::shard_claimed_sums_net_to_public` (CPU/Simd fixture); this
            // runtime check additionally guards each run's real secret shot data in debug builds.
            #[cfg(debug_assertions)]
            {
                let b_public = boundary_public_term(&boundary, &elements.qubitmem);
                let p_pub = program_public_term(program, &elements.program);
                if main_sum + program_sum + boundary_sum + rc_sum != b_public + p_pub {
                    bail!("shard claimed sums do not net to the public terms B + P_pub");
                }
            }

            let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
            prover_channel.mix_felts(&claimed_sums);

            // Tree 2: interaction (same component order as the claimed sums): main, program, boundary, rc.
            let small_interaction = {
                let mut v = program_interaction;
                v.extend(boundary_interaction);
                v.extend(rc_interaction);
                v
            };
            let mut tree_builder = commitment_scheme.tree_builder();
            #[cfg(feature = "cuda")]
            if let Some((main_dev, _)) = main_interaction_device {
                let mut interaction = main_dev;
                interaction.extend(to_prover(small_interaction));
                tree_builder.extend_evals(interaction);
            } else {
                let mut interaction = main_interaction;
                interaction.extend(small_interaction);
                tree_builder.extend_evals(to_prover(interaction));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mut interaction = main_interaction;
                interaction.extend(small_interaction);
                tree_builder.extend_evals(to_prover(interaction));
            }
            tree_builder.commit(prover_channel);

            let components = build_components(
                log_n_rows,
                program.log_size,
                boundary.log_size,
                rc_log,
                &elements,
                main_sum,
                program_sum,
                boundary_sum,
                rc_sum,
            );
            let prover_refs = components.prover_refs();
            let extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(
                &prover_refs,
                prover_channel,
                commitment_scheme,
                false,
            )?;

            // Per-shard boundary (x->y per shard shot) for the leaf's GateAirStatement + output hash.
            let mut shard_boundary = Vec::with_capacity(shard_cases.len());
            for case in shard_cases {
                let x = state_to_limbs(&hex::decode(&case.x_hex)?);
                let y = state_to_limbs(&hex::decode(&case.y_hex)?);
                shard_boundary.push((x, y));
            }
            let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];
            Ok((
                extended,
                claim,
                interaction_pow_nonce,
                channel_salt,
                log_n_rows,
                program.log_size,
                shard_boundary,
                (n_gates * k) as u32,
            ))
}
