//! Trace / witness generation for the gate_air base prover (CPU column builders + interaction
//! LogUp generators) plus the on-device CUDA gate-sim + interaction trace-gen (absorbed from the
//! former `gpu_tracegen` module). Pure trace-data production/transformation; the AIR constraints,
//! layout, and orchestration stay in `main.rs`. Shared AIR consts/types (`N_QUBITS`, `TAG_*`,
//! `LookupElements`, `GateRel`, ...) live in `air.rs`; `Gate`/`TestCase` at the crate root; imported
//! here explicitly.

use anyhow::{bail, Context, Result};
use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use crate::air::{
    ptag, GateRel, LookupElements, ACCESS_BLOCK, LANE_COUNT, LIMB_BITS, M31_MODULUS_U32, NO_CTRL,
    N_LIMBS, N_QUBITS, OP_CNOT, OP_NOP, OP_NOT, OP_TOFFOLI, STATE_BYTES, TS_FINAL, TS_RC_BITS,
};
use crate::components::program::{TAG_PROGRAM, TAG_PROGRAM_PUB};
use crate::components::qubitmem::TAG_QUBITMEM;
use crate::components::range_check::TAG_RC;
use crate::preprocessed::{
    col_from_values, generate_boundary_preprocessed, generate_enabler_preprocessed,
    generate_pc_in_prog_preprocessed, generate_pc_preprocessed, generate_prog_slot_preprocessed,
    generate_rc_preprocessed, generate_shot_id_preprocessed,
};
use crate::prover::ProgramTable;
use crate::{Gate, TestCase};

// ==== CPU trace / witness generation (moved from main.rs) ====

// Witness row
/// One qubit-memory access (chain lookup) for target / ctrl_a / ctrl_b.
/// The access timestamp is NOT a witness column: it is the affine function `ts = pc + 1` of the
/// verifier-pinned preprocessed `pc`, inlined at every use site. `prev_ts` is the ts of the previous
/// access to this addr (0 = the init boundary node). `d` is the ordering diff
/// `d = ts - prev_ts - 1 = (pc+1) - prev_ts - 1 = pc - prev_ts`, range-checked by a SINGLE LogUp
/// lookup into the dynamic rc supply table (`d ∈ [0, 2^RC_LOG_SIZE)`), proving `prev_ts < ts`
/// (forward-DAG / no-stale-read). `active` gates the terms.
#[derive(Clone, Copy)]
pub(crate) struct AccessCols {
    pub(crate) addr: u32, // qubit index 0..511 (0 when inactive, matches program canon)
    pub(crate) prev_ts: u32, // predecessor's ts at this addr (0 if this is the first access)
    pub(crate) v: u32,    // v_before (the value read); for a control this is also v_after
    pub(crate) d: u32, // ts-ordering diff d = ts - prev_ts - 1 = pc - prev_ts (range-checked into [0,2^RC_LOG_SIZE))
}

impl AccessCols {
    pub(crate) fn inactive() -> Self {
        Self {
            addr: 0,
            prev_ts: 0,
            v: 0,
            d: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Row {
    pub(crate) enabler: u32,
    pub(crate) is_nop: u32,
    pub(crate) is_not: u32,
    pub(crate) is_cnot: u32,
    pub(crate) is_toffoli: u32,
    pub(crate) shot_id: u32,
    pub(crate) pc: u32,
    pub(crate) target: AccessCols,
    // NOTE: the target's post-gate value `v_after` is NOT a witness column — it equals
    // `v_before + delta` (a pinned equality), inlined at every use site.
    pub(crate) ctrl_a: AccessCols,
    pub(crate) ctrl_b: AccessCols,
    pub(crate) ab: u32,
    pub(crate) fire: u32,
    pub(crate) delta: u32, // v_after - v_before, signed in {-1,0,1}; stored as M31.
}

impl Row {
    pub(crate) fn padding() -> Self {
        Self {
            enabler: 0,
            is_nop: 0,
            is_not: 0,
            is_cnot: 0,
            is_toffoli: 0,
            shot_id: 0,
            pc: 0,
            target: AccessCols::inactive(),
            ctrl_a: AccessCols::inactive(),
            ctrl_b: AccessCols::inactive(),
            ab: 0,
            fire: 0,
            delta: 0,
        }
    }
}

// Data tables (trace-side): the boundary / rc supply tables the base proof commits. These are built and
// consumed by `tracegen` / the base prover (they hold NO constraint logic); they sit here beside
// `AccessCols`/`Row` (the trace data they count over) and the GPU trace glue.

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
    pub(crate) val: Vec<u32>, // preprocessed: val[i] = i for i in [0, 2^log_size)
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

pub(crate) fn state_to_limbs(bytes: &[u8]) -> [u32; N_LIMBS] {
    debug_assert_eq!(bytes.len(), STATE_BYTES);
    let mut limbs = [0u32; N_LIMBS];
    for (j, limb) in limbs.iter_mut().enumerate() {
        let lo = bytes[2 * j] as u32;
        let hi = bytes[2 * j + 1] as u32;
        *limb = lo | (hi << 8);
    }
    limbs
}

// Witness generation + self-check
/// Build all witness rows for the selected shots, asserting each shot's final state matches y_hex.
/// Parallelizes over SHOTS (each owns a disjoint contiguous row block, chained in order within the
/// shot), trace-bit-identical to serial; packing into `PackedM31` happens later single-threaded/word.
pub(crate) fn build_rows(
    gates: &[Gate],
    cases: &[TestCase],
    k: usize,
) -> Result<(Vec<Row>, BoundaryTable)> {
    use rayon::prelude::*;

    let n_gates = gates.len();
    let shot_rows = k * n_gates;
    let total_rows = cases.len() * shot_rows;

    // LOUD completeness guard (not debug_assert): the max honest ts diff d_max = k*n_gates - 1 must
    // stay below 2^TS_RC_BITS, else the rc-table range-check would silently REJECT honest proofs.
    let d_max: u64 = (k as u64) * (n_gates as u64) - 1;
    if d_max >= (1u64 << TS_RC_BITS) {
        bail!(
            "ts-ordering range-check budget exceeded: k*n_gates = {}*{} makes the max ts diff d_max = {} >= 2^{} \
             (the rc-table bound); honest proofs would silently fail the range-check. Widen TS_RC_BITS / the rc-limb split.",
            k, n_gates, d_max, TS_RC_BITS
        );
    }

    // Pre-allocate the full scalar row buffer; each shot fills a disjoint block.
    let mut rows = vec![Row::padding(); total_rows];
    // Per-shot boundary rows: 512 entries per shot, holding init/final (x, y, ts_last).
    let mut boundary = BoundaryTable::new(cases.len());

    // Phase 1: simulate every shot in parallel into its own disjoint row block +
    // boundary block. Returns Err on the first shot whose simulation fails or whose
    // final state mismatches y_hex.
    let per_shot: Vec<Result<()>> = rows
        .par_chunks_mut(shot_rows)
        .zip(boundary.per_shot_mut().par_iter_mut())
        .zip(cases.par_iter())
        .enumerate()
        .map(|(shot_id, ((block, bnd), case))| simulate_shot(gates, k, shot_id, case, block, bnd))
        .collect();

    for result in per_shot {
        result?;
    }

    Ok((rows, boundary))
}

/// Simulate a single shot sequentially, filling its row block: the chain (K reps * n_gates gates) is
/// run strictly in order, threading the 512-bit state from x_s to y_s, checked against y_hex.
pub(crate) fn simulate_shot(
    gates: &[Gate],
    k: usize,
    shot_id: usize,
    case: &TestCase,
    block: &mut [Row],
    bnd: &mut [BoundaryRow],
) -> Result<()> {
    let n_gates = gates.len();
    let x = hex::decode(&case.x_hex).context("decoding x_hex")?;
    let y = hex::decode(&case.y_hex).context("decoding y_hex")?;
    if x.len() != STATE_BYTES || y.len() != STATE_BYTES {
        bail!("state must be {STATE_BYTES} bytes");
    }
    debug_assert_eq!(bnd.len(), N_QUBITS);

    // Per-shot qubit-memory state: last[addr] = (ts, value). Reset each shot.
    // `ts` is the PROGRAM-ORDER timestamp `pc + 1`: an affine function of the preprocessed pc, so an
    // address's accesses are timestamped in program order and cannot be reordered by the prover.
    let x_bit = |addr: usize| -> u32 { qubit_bit(&x, addr) };
    let mut last_ts = vec![0u32; N_QUBITS];
    let mut last_val: Vec<u32> = (0..N_QUBITS).map(x_bit).collect();
    let mut pc: u32 = 0;
    let mut row_idx = 0usize;

    for _rep in 0..k {
        for gate in gates {
            let (is_nop, is_not, is_cnot, is_toffoli) = match gate.opcode {
                OP_NOP => (1, 0, 0, 0),
                OP_NOT => (0, 1, 0, 0),
                OP_CNOT => (0, 0, 1, 0),
                OP_TOFFOLI => (0, 0, 0, 1),
                other => bail!("unknown opcode {other}"),
            };
            let a_active = is_cnot + is_toffoli;
            let b_active = is_toffoli;

            // Control reads first (they feed the gate-apply), then target read+write. ts is the
            // program-order timestamp `pc + 1`, shared by all accesses of this gate step (no slot).
            // The three accesses of a gate touch DISTINCT addrs (a reversible gate can't use its
            // target as a control), so sharing ts within the step never collides two accesses on the
            // same per-address chain; two accesses to the same addr are in different steps (distinct
            // pc), so they still get strictly increasing ts. In iadd the three accesses touch distinct
            // addrs.
            let ts = pc + 1;
            let ctrl_a = if a_active == 1 {
                debug_assert_ne!(gate.ctrl_a, NO_CTRL);
                let ac = do_access(gate.ctrl_a as u32, pc, &last_ts, &last_val);
                last_ts[gate.ctrl_a as usize] = ts;
                // read: value propagates unchanged.
                ac
            } else {
                AccessCols::inactive()
            };
            let ctrl_b = if b_active == 1 {
                debug_assert_ne!(gate.ctrl_b, NO_CTRL);
                let ac = do_access(gate.ctrl_b as u32, pc, &last_ts, &last_val);
                last_ts[gate.ctrl_b as usize] = ts;
                ac
            } else {
                AccessCols::inactive()
            };
            let target = do_access(gate.target as u32, pc, &last_ts, &last_val);

            let a_bit = ctrl_a.v;
            let b_bit = ctrl_b.v;
            let t_bit = target.v;
            let ab = a_bit * b_bit;
            let fire = is_not + is_cnot * a_bit + is_toffoli * ab;
            debug_assert!(fire <= 1);
            let v_after = t_bit ^ fire;
            let delta_signed = v_after as i64 - t_bit as i64;

            // Commit the target write to memory (ts = pc+1, inlined).
            last_ts[gate.target as usize] = ts;
            last_val[gate.target as usize] = v_after;

            block[row_idx] = Row {
                enabler: 1,
                is_nop,
                is_not,
                is_cnot,
                is_toffoli,
                shot_id: shot_id as u32,
                pc,
                target,
                ctrl_a,
                ctrl_b,
                ab,
                fire,
                delta: delta_to_m31(delta_signed),
            };

            pc += 1;
            row_idx += 1;
        }
    }
    debug_assert_eq!(row_idx, k * n_gates);

    // Boundary rows: init x, final y (== last_val), ts_last (0 if untouched).
    for addr in 0..N_QUBITS {
        let y_bit = qubit_bit(&y, addr);
        // Self-check: simulated final value equals y's bit at this addr.
        if last_val[addr] != y_bit {
            bail!(
                "shot {shot_id}: simulated final qubit {addr} = {} != y bit {}",
                last_val[addr],
                y_bit
            );
        }
        bnd[addr] = BoundaryRow {
            shot_id: shot_id as u32,
            addr: addr as u32,
            x: x_bit(addr),
            y: y_bit,
            ts_last: last_ts[addr],
        };
    }

    Ok(())
}

// Trace generation (column-major)
/// Single scalar cell of `row` at canonical column index `col`. The column order
/// matches the order `evaluate` reads masks (each access block = ACCESS_COLS core + 1 rc diff col):
///   is_{nop,not,cnot,toffoli}(4),
///   target(addr,prev_ts,v, d),
///   ctrl_a(addr,prev_ts,v, d), ctrl_b(addr,prev_ts,v, d),
///   ab, fire, delta (3).
/// NOTE: enabler, shot_id, pc are NOT here — they are preprocessed (tree0). ts (= pc+1) and the
/// target's v_after (= v_before+delta) are NOT columns either — they are inlined in `evaluate`.
#[inline]
pub(crate) fn cell_at(row: &Row, col: usize) -> u32 {
    debug_assert!(col < crate::TRACE_COLUMNS);
    #[inline]
    fn access_cell(a: &AccessCols, i: usize) -> u32 {
        // 0..ACCESS_COLS: addr, prev_ts, v ; then d.
        match i {
            0 => a.addr,
            1 => a.prev_ts,
            2 => a.v,
            _ => a.d,
        }
    }
    let mut c = col;
    // Header (4 cols).
    const HEADER: [fn(&Row) -> u32; 4] =
        [|r| r.is_nop, |r| r.is_not, |r| r.is_cnot, |r| r.is_toffoli];
    if c < HEADER.len() {
        return HEADER[c](row);
    }
    c -= HEADER.len();
    // target block: ACCESS_BLOCK access cols (no v_after column — inlined as v_before+delta).
    if c < ACCESS_BLOCK {
        return access_cell(&row.target, c);
    }
    c -= ACCESS_BLOCK;
    // ctrl_a block.
    if c < ACCESS_BLOCK {
        return access_cell(&row.ctrl_a, c);
    }
    c -= ACCESS_BLOCK;
    // ctrl_b block.
    if c < ACCESS_BLOCK {
        return access_cell(&row.ctrl_b, c);
    }
    c -= ACCESS_BLOCK;
    // Tail: ab, fire, delta.
    match c {
        0 => row.ab,
        1 => row.fire,
        _ => row.delta,
    }
}

/// Converts trace-gen (SimdBackend) columns to the prover backend at the `extend_evals` boundary.
///
/// * default / `gpu`: `ProverBackend == SimdBackend`, or obelyzk `GpuBackend` whose columns are
///   layout-identical to SimdBackend — a cheap rewrap (`CircleEvaluation::new(domain, values)`).
/// * `cuda`: `ProverBackend == CudaBackend` with device-resident `BaseFieldVec` columns; copy each
///   column's host values into a device column via `FromIterator<BaseField> for BaseFieldVec`
///   (`to_cpu()` is a no-op on SimdBackend host data, then `.collect()` uploads to the device).
#[cfg(not(feature = "cuda"))]
pub(crate) fn to_prover(
    cols: Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
) -> Vec<CircleEvaluation<ProverBackend, BaseField, BitReversedOrder>> {
    cols.into_iter()
        .map(|e| CircleEvaluation::new(e.domain, e.values))
        .collect()
}

#[cfg(feature = "cuda")]
pub(crate) fn to_prover(
    cols: Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
) -> Vec<CircleEvaluation<ProverBackend, BaseField, BitReversedOrder>> {
    cols.into_iter()
        .map(|e| {
            let domain = e.domain;
            let values: Col<ProverBackend, BaseField> = e.values.to_cpu().into_iter().collect();
            CircleEvaluation::new(domain, values)
        })
        .collect()
}

/// Boundary-table witness (x, y, ts_last), in the order QubitMemEval reads them.
pub(crate) fn generate_boundary_witness(
    bnd: &BoundaryTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let x: Vec<u32> = bnd.rows.iter().map(|r| r.x).collect();
    let y: Vec<u32> = bnd.rows.iter().map(|r| r.y).collect();
    let ts_last: Vec<u32> = bnd.rows.iter().map(|r| r.ts_last).collect();
    vec![
        col_from_values(&x),
        col_from_values(&y),
        col_from_values(&ts_last),
    ]
}

/// Program-table witness (multiplicity tree): op columns then multiplicity, in
/// the order ProgramEval reads them.
pub(crate) fn generate_program_witness(
    prog: &ProgramTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![
        col_from_values(&prog.opcode_scalar),
        col_from_values(&prog.target),
        col_from_values(&prog.ctrl_a),
        col_from_values(&prog.ctrl_b),
        col_from_values(&prog.multiplicity),
    ]
}

/// rc-table witness (multiplicity tree): a single multiplicity column.
pub(crate) fn generate_rc_witness(
    rc: &RcTable,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    vec![col_from_values(&rc.multiplicity)]
}

/// Main component interaction trace. Logup batches mirror the relation entries
/// emitted in `evaluate`, finalized in pairs. Order must match (5 batches -> 4 columns each):
///   pair0: qubitmem target Use (+enabler), target Yield (-enabler)
///   pair1: qubitmem ctrl_a Use (+a_active), ctrl_a Yield (-a_active)
///   pair2: qubitmem ctrl_b Use (+b_active), ctrl_b Yield (-b_active)
///   pair3: rc target d (+enabler), rc ctrl_a d (+a_active)
///   pair4: rc ctrl_b d (+b_active), program (+enabler)
/// 10 relation entries -> 5 pairs => 5 batches => 20 interaction columns. The rc terms are the
/// ts-ordering range-check single-`d` lookups (TAG_RC, d) mirroring `add_rc_lookup`.
pub(crate) fn gen_main_interaction(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    el: &LookupElements,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(log_n_rows);

    fn sel_t(r: &Row) -> &AccessCols {
        &r.target
    }
    fn sel_a(r: &Row) -> &AccessCols {
        &r.ctrl_a
    }
    fn sel_b(r: &Row) -> &AccessCols {
        &r.ctrl_b
    }

    // Chain USE (predecessor): (shot, addr, prev_ts, v_before).
    let qm_use = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.qubitmem.combine(&[
            ptag(TAG_QUBITMEM),
            pack(lane, |r| r.shot_id),
            pack(lane, |r| sel(r).addr),
            pack(lane, |r| sel(r).prev_ts),
            pack(lane, |r| sel(r).v),
        ])
    };
    // Chain YIELD (successor): (shot, addr, ts=pc+1, v_after). ts is the inlined `pc + 1` (not a
    // column). For the target, v_after = v_before + delta; for controls the value propagates (== v).
    let qm_yield_t = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.qubitmem.combine(&[
            ptag(TAG_QUBITMEM),
            pack(lane, |r| r.shot_id),
            pack(lane, |r| r.target.addr),
            pack(lane, |r| r.pc + 1),
            // v_after = v_before + delta, computed as the exact bit v_before ^ fire (canonical M31,
            // avoids the non-canonical p that a raw `v_before + delta_to_m31(-1)` would produce).
            pack(lane, |r| r.target.v ^ r.fire),
        ])
    };
    let qm_yield_ctrl =
        |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
            el.qubitmem.combine(&[
                ptag(TAG_QUBITMEM),
                pack(lane, |r| r.shot_id),
                pack(lane, |r| sel(r).addr),
                pack(lane, |r| r.pc + 1),
                pack(lane, |r| sel(r).v),
            ])
        };
    let enabler = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.enabler);
    let a_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_cnot + r.is_toffoli);
    let b_active = |lane: &[&Row; LANE_COUNT]| pack(lane, |r| r.is_toffoli);

    // ts-ordering range-check use-side denominator: (TAG_RC, d) — single `d` per access.
    let rc_d = |lane: &[&Row; LANE_COUNT], sel: fn(&Row) -> &AccessCols| -> PackedSecureField {
        el.rc.combine(&[ptag(TAG_RC), pack(lane, |r| sel(r).d)])
    };

    // Program use-side denominator. pc_in_prog = pc mod n_gates (preprocessed in
    // the AIR; recomputed here for the prover). opcode_scalar from the one-hot.
    let ng = n_gates as u32;
    let program = |lane: &[&Row; LANE_COUNT]| -> PackedSecureField {
        el.program.combine(&[
            ptag(TAG_PROGRAM),
            pack(lane, |r| r.pc % ng),
            pack(lane, |r| r.is_not + 2 * r.is_cnot + 3 * r.is_toffoli),
            pack(lane, |r| r.target.addr),
            pack(lane, |r| r.ctrl_a.addr),
            pack(lane, |r| r.ctrl_b.addr),
        ])
    };

    // Write one logup column for a pair of relation entries: fraction = m0/d0 + m1/d1.
    // PARALLEL over vec_rows: the per-row combine (35-element dot products) is the cost; computing
    // the (num, den) fractions in a rayon par_iter and feeding `col_from_par_iter` fans it over all
    // cores (the old sequential packed_rows loop ran this on one core — the trace-gen bottleneck).
    // Generic over the closure types (not &dyn) so the Sync bound holds without lifetime grief.
    #[allow(clippy::too_many_arguments)]
    fn write_pair_par<N0, D0, N1, D1>(
        gen: &mut LogupTraceGenerator,
        rows: &[Row],
        n_vec: usize,
        num0: N0,
        den0: D0,
        sign0: i32,
        num1: N1,
        den1: D1,
        sign1: i32,
    ) where
        N0: Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync,
        D0: Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync,
        N1: Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync,
        D1: Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync,
    {
        use rayon::prelude::*;
        let pad = Row::padding();
        let col_iter = (0..n_vec).into_par_iter().map(|vec_row| {
            let lane: [&Row; LANE_COUNT] =
                std::array::from_fn(|l| rows.get(vec_row * LANE_COUNT + l).unwrap_or(&pad));
            let m0 = PackedSecureField::from(num0(&lane));
            let m0 = if sign0 < 0 { -m0 } else { m0 };
            let d0 = den0(&lane);
            let m1 = PackedSecureField::from(num1(&lane));
            let m1 = if sign1 < 0 { -m1 } else { m1 };
            let d1 = den1(&lane);
            (m0 * d1 + m1 * d0, d0 * d1)
        });
        gen.col_from_par_iter(col_iter);
    }
    let n_vec = padded_rows / LANE_COUNT;
    let write_pair = |gen: &mut LogupTraceGenerator,
                      num0: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync),
                      den0: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync),
                      sign0: i32,
                      num1: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedM31 + Sync),
                      den1: &(dyn Fn(&[&Row; LANE_COUNT]) -> PackedSecureField + Sync),
                      sign1: i32| {
        write_pair_par(gen, rows, n_vec, num0, den0, sign0, num1, den1, sign1);
    };

    // pair0: qubitmem target Use (+enabler), target Yield (-enabler).
    write_pair(
        &mut gen,
        &enabler,
        &|l| qm_use(l, sel_t),
        1,
        &enabler,
        &qm_yield_t,
        -1,
    );
    // pair1: qubitmem ctrl_a Use (+a_active), ctrl_a Yield (-a_active).
    write_pair(
        &mut gen,
        &a_active,
        &|l| qm_use(l, sel_a),
        1,
        &a_active,
        &|l| qm_yield_ctrl(l, sel_a),
        -1,
    );
    // pair2: qubitmem ctrl_b Use (+b_active), ctrl_b Yield (-b_active).
    write_pair(
        &mut gen,
        &b_active,
        &|l| qm_use(l, sel_b),
        1,
        &b_active,
        &|l| qm_yield_ctrl(l, sel_b),
        -1,
    );
    // pair3: rc target d (+enabler), rc ctrl_a d (+a_active).
    write_pair(
        &mut gen,
        &enabler,
        &|l| rc_d(l, sel_t),
        1,
        &a_active,
        &|l| rc_d(l, sel_a),
        1,
    );
    // pair4: rc ctrl_b d (+b_active), program (+enabler). The trailing 3 rc + 1 program fold into
    // these two pairs (matches `finalize_logup_in_pairs` on the 10-entry stream).
    write_pair(
        &mut gen,
        &b_active,
        &|l| rc_d(l, sel_b),
        1,
        &enabler,
        &program,
        1,
    );

    // LogupTraceGenerator already emits SimdBackend (== TraceBackend) columns; conversion to the
    // prover backend happens later via `to_prover` at the `extend_evals` boundary.
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

/// Supply-side interaction trace for a multi-column table looked up with a
/// single relation. `combine_row(i)` returns the combined denominator for row i.
pub(crate) fn gen_table_interaction(
    counts: &[u32],
    log_size: u32,
    combine_row: impl Fn(usize) -> PackedSecureField,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(log_size);
    let mut col = gen.new_col();
    for vec_row in 0..(1usize << (log_size - LOG_N_LANES)) {
        let denom = combine_row(vec_row);
        let packed_counts = PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(counts[(vec_row << LOG_N_LANES) + lane])
        }));
        col.write_frac(vec_row, -PackedSecureField::from(packed_counts), denom);
    }
    col.finalize_col();
    // LogupTraceGenerator already emits SimdBackend (== TraceBackend) columns; conversion to the
    // prover backend happens later via `to_prover` at the `extend_evals` boundary.
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

/// Boundary component interaction trace: per (shot, addr) row emit the internal final
/// Use[+1](shot, addr, ts_last, y) and the PUBLIC final Yield[-1](shot, addr, TS_FINAL, y) on
/// TAG_QUBITMEM. Two terms per row -> one batch (paired), matching `QubitMemEval`. CPU-only in
/// both the CPU and cuda paths (the CUDA kernel covers only the gate_air MAIN component).
pub(crate) fn gen_boundary_interaction(
    bnd: &BoundaryTable,
    el: &GateRel,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(bnd.log_size);
    let mut col = gen.new_col();
    let n_vec = 1usize << (bnd.log_size - LOG_N_LANES);
    let pack_f = |vec_row: usize, f: &dyn Fn(&BoundaryRow) -> u32| pack_boundary(bnd, vec_row, f);
    let ts_final = PackedM31::broadcast(BaseField::from_u32_unchecked(TS_FINAL));
    let real = bnd.n_shots * N_QUBITS;
    for vec_row in 0..n_vec {
        let tag = ptag(TAG_QUBITMEM);
        let shot = pack_f(vec_row, &|r| r.shot_id);
        let addr = pack_f(vec_row, &|r| r.addr);
        let y = pack_f(vec_row, &|r| r.y);
        let ts_last = pack_f(vec_row, &|r| r.ts_last);
        // Real-row enabler per lane (mirrors the gate_bnd_enabler preprocessed column).
        let enabler = PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(((vec_row << LOG_N_LANES) + lane < real) as u32)
        }));
        let enabler = PackedSecureField::from(enabler);
        // Phase-3 re-keyed boundary (mirrors QubitMemEval), gated by the real-row enabler:
        //   (B) internal final Use[+enabler] / (shot, addr, ts_last, y).
        let d_use: PackedSecureField = el.combine(&[tag, shot, addr, ts_last, y]);
        //   (D) public   final Yield[-enabler] / (shot, addr, TS_FINAL, y).
        let d_pub: PackedSecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        // fraction = (+enabler)/d_use + (-enabler)/d_pub.
        col.write_frac(vec_row, enabler * d_pub + (-enabler) * d_use, d_use * d_pub);
    }
    col.finalize_col();
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
}

/// Program-table interaction (H_P binding, Fork A). Mirrors `gen_boundary_interaction`'s two-term/
/// one-batch shape so the program component stays 4 interaction columns. Per real slot row emits:
///   (internal, -mult) / combine(TAG_PROGRAM,     slot, op, t, a, b)  — cancels main's demand,
///   (public,   +mult) / combine(TAG_PROGRAM_PUB, slot, op, t, a, b)  — the dangling P_pub.
/// Padding rows carry multiplicity 0, so both fractions vanish (numerator 0). The returned claimed
/// sum is `program_sum` = Σ_slot [ -mult/d_int + mult/d_pub ] = P_pub (the internal part is cancelled
/// by main's demand only in the GLOBAL sum, not within this component; `program_sum` itself carries
/// BOTH terms, and the global identity becomes main + program + boundary + rc == B + P_pub, where the
/// `-mult/d_int` inside program_sum cancels main's `+enabler/d_int`, leaving net P_pub).
pub(crate) fn gen_program_interaction(
    prog: &ProgramTable,
    el: &GateRel,
) -> (
    ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut gen = LogupTraceGenerator::new(prog.log_size);
    let mut col = gen.new_col();
    let n_vec = 1usize << (prog.log_size - LOG_N_LANES);
    let tag = ptag(TAG_PROGRAM);
    let tag_pub = ptag(TAG_PROGRAM_PUB);
    for vec_row in 0..n_vec {
        let slot = pack_seq(&prog.slot, vec_row);
        let op = pack_seq(&prog.opcode_scalar, vec_row);
        let t = pack_seq(&prog.target, vec_row);
        let a = pack_seq(&prog.ctrl_a, vec_row);
        let b = pack_seq(&prog.ctrl_b, vec_row);
        let mult = PackedSecureField::from(pack_seq(&prog.multiplicity, vec_row));
        let d_int: PackedSecureField = el.combine(&[tag, slot, op, t, a, b]);
        let d_pub: PackedSecureField = el.combine(&[tag_pub, slot, op, t, a, b]);
        // fraction = (-mult)/d_int + (+mult)/d_pub.
        col.write_frac(vec_row, (-mult) * d_pub + mult * d_int, d_int * d_pub);
    }
    col.finalize_col();
    let (cols, sum) = gen.finalize_last();
    (cols, sum)
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

pub(crate) fn build_rc_lo() -> RcIndex {
    RcIndex::build(|pos| 1u32 << pos)
}

/// Program-table PUBLIC term P_pub = Σ_slot mult/combine(TAG_PROGRAM_PUB, slot, op, t, a, b) — the
/// dangling public part the leaf's `public_logup_sum` supplies the negation of (binding the guessed
/// program to the committed one). Recomputed from the committed program witness.
pub(crate) fn program_public_term(prog: &ProgramTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag_pub = BaseField::from_u32_unchecked(TAG_PROGRAM_PUB);
    for i in 0..prog.multiplicity.len() {
        let count = prog.multiplicity[i];
        if count == 0 {
            continue;
        }
        let denom: SecureField = el.combine(&[
            tag_pub,
            BaseField::from_u32_unchecked(prog.slot[i]),
            BaseField::from_u32_unchecked(prog.opcode_scalar[i]),
            BaseField::from_u32_unchecked(prog.target[i]),
            BaseField::from_u32_unchecked(prog.ctrl_a[i]),
            BaseField::from_u32_unchecked(prog.ctrl_b[i]),
        ]);
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    sum
}

/// Program component claimed sum (H_P binding): Σ_slot [ -mult/combine(TAG_PROGRAM,...) +
/// mult/combine(TAG_PROGRAM_PUB,...) ]. Must equal `program_sum`; recomputed from the committed
/// program witness so a mistranscribed term is caught before FRI.
pub(crate) fn program_claimed_sum(prog: &ProgramTable, el: &GateRel) -> SecureField {
    let internal = table_public_sum(&prog.multiplicity, el, TAG_PROGRAM, |i| {
        vec![
            BaseField::from_u32_unchecked(prog.slot[i]),
            BaseField::from_u32_unchecked(prog.opcode_scalar[i]),
            BaseField::from_u32_unchecked(prog.target[i]),
            BaseField::from_u32_unchecked(prog.ctrl_a[i]),
            BaseField::from_u32_unchecked(prog.ctrl_b[i]),
        ]
    });
    // `table_public_sum` returns the NEGATED supply (-Σ count/denom), i.e. exactly the -mult internal
    // term. The public term adds +Σ mult/denom_pub = program_public_term.
    internal + program_public_term(prog, el)
}

/// Boundary component claimed sum (supply side): Σ_rows [ +1/combine(ts_last,y) − 1/combine(TS_FINAL,y) ]
/// — the re-keyed final terms (B)+(D). Must equal `boundary_sum`; recomputed from the committed
/// boundary table (y/ts_last witness) so a mistranscribed term is caught before FRI.
pub(crate) fn boundary_public_sum(bnd: &BoundaryTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag = BaseField::from_u32_unchecked(TAG_QUBITMEM);
    let ts_final = BaseField::from_u32_unchecked(TS_FINAL);
    // Only REAL rows emit (gated by gate_bnd_enabler); padding rows contribute nothing.
    let real = bnd.n_shots * N_QUBITS;
    for r in &bnd.rows[..real] {
        let shot = BaseField::from_u32_unchecked(r.shot_id);
        let addr = BaseField::from_u32_unchecked(r.addr);
        let y = BaseField::from_u32_unchecked(r.y);
        let ts_last = BaseField::from_u32_unchecked(r.ts_last);
        // (B) internal final Use[+1] and (D) public final Yield[-1]: +1/d_use − 1/d_pub.
        let d_use: SecureField = el.combine(&[tag, shot, addr, ts_last, y]);
        let d_pub: SecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        sum += d_use.inverse() - d_pub.inverse();
    }
    sum
}

/// Phase-3 PUBLIC boundary term B = Σ_rows [ +1/combine(shot,addr,0,x) − 1/combine(shot,addr,TS_FINAL,y) ].
/// This is exactly the base's total dangling (unconsumed) LogUp sum: main leaves +[0,x] & −[ts_last,y]
/// per touched (shot,addr), the boundary consumes [ts_last,y] and re-emits −[TS_FINAL,y], so the net is
/// +[0,x] − [TS_FINAL,y]. The base's committed claimed sums must satisfy
/// `main_sum + program_sum + boundary_sum == B` (was `== 0`), and the LEAF's `public_logup_sum` equals
/// `−B` over its GUESSED x/y — so the verifier balance `public_logup_sum + Σ claimed_sums == 0` forces
/// guessed == committed. For UNTOUCHED addrs (ts_last=0) the prover's `x == y`, so the ts=0 term here
/// (using `x`) faithfully matches the actual dangling +[0,y]; the equality thus also checks x==y there.
pub(crate) fn boundary_public_term(bnd: &BoundaryTable, el: &GateRel) -> SecureField {
    let mut sum = SecureField::zero();
    let tag = BaseField::from_u32_unchecked(TAG_QUBITMEM);
    let zero = BaseField::zero();
    let ts_final = BaseField::from_u32_unchecked(TS_FINAL);
    // Only REAL rows are dangling; padding rows emit no main NOR boundary term (gated by enablers).
    let real = bnd.n_shots * N_QUBITS;
    for r in &bnd.rows[..real] {
        let shot = BaseField::from_u32_unchecked(r.shot_id);
        let addr = BaseField::from_u32_unchecked(r.addr);
        let x = BaseField::from_u32_unchecked(r.x);
        let y = BaseField::from_u32_unchecked(r.y);
        let d_init: SecureField = el.combine(&[tag, shot, addr, zero, x]);
        let d_pub: SecureField = el.combine(&[tag, shot, addr, ts_final, y]);
        sum += d_init.inverse() - d_pub.inverse();
    }
    sum
}

/// Supply (table-side) public sum for a single-relation table: sum_i count_i / denom_i.
/// Returns the negated value (matching the supply emission of -multiplicity).
pub(crate) fn table_public_sum<R: Relation<BaseField, SecureField>>(
    counts: &[u32],
    elements: &R,
    tag: u32,
    row_tuple: impl Fn(usize) -> Vec<BaseField>,
) -> SecureField {
    let mut sum = SecureField::zero();
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let mut tuple = vec![BaseField::from_u32_unchecked(tag)];
        tuple.extend(row_tuple(i));
        let denom: SecureField = elements.combine(&tuple);
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    -sum
}

// GPU trace-gen inputs (device path)
/// Flatten the host-side inputs the K1/K4 device kernels consume: the gate list
/// (opcode, target, ctrl_a, ctrl_b per gate), each shot's initial 32-limb state,
/// and the RcIndex lo/hi offsets. Mirrors the prep in `gpu_tracegen::k1_byte_identity`.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_flat_inputs(
    gates: &[Gate],
    cases: &[TestCase],
    rc_lo_index: &RcIndex,
    rc_hi_index: &RcIndex,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let mut gates_flat = Vec::with_capacity(gates.len() * 4);
    for g in gates {
        gates_flat.push(g.opcode as u32);
        gates_flat.push(g.target as u32);
        gates_flat.push(g.ctrl_a as u32);
        gates_flat.push(g.ctrl_b as u32);
    }
    let mut x_states = Vec::with_capacity(cases.len() * N_LIMBS);
    for c in cases {
        let bytes = hex::decode(&c.x_hex).context("decoding x_hex for GPU trace-gen")?;
        x_states.extend_from_slice(&state_to_limbs(&bytes));
    }
    let off_lo: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_lo_index.offset[p] as u32)
        .collect();
    let off_hi: Vec<u32> = (0..LIMB_BITS)
        .map(|p| rc_hi_index.offset[p] as u32)
        .collect();
    Ok((gates_flat, x_states, off_lo, off_hi))
}

/// Build the (size-sorted) preprocessed tree-0 columns for one shard shape. Shared by the
/// per-shard rebuild path AND the precompute build, so the committed column order/sizes are
/// IDENTICAL by construction. tree-0 is SHARD-INVARIANT: every column is POSITIONAL
/// (enabler/shot_id/pc/pc_in_prog from row index, prog_slot/program witness from the shared program
/// with constant multiplicity = shots_per_shard*k, bnd_shot/bnd_addr from the boundary layout), so
/// for a fixed (k, n_gates, shots_per_shard) shape these columns are the same for every shard. The
/// rc-table membership columns (pos, val) are also shard-invariant (fixed [0,range) table).
pub(crate) fn build_tree0_columns(
    program: &ProgramTable,
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
    n_gates: usize,
    rc_log: u32,
    boundary: &BoundaryTable,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    // Shape scalars extracted from the staging data (kept out of `preprocessed`, which sees only
    // shape params): real main rows, shot count, and `k` (= real rows / (n_shots * n_gates)).
    let n_real = rows.len();
    let n_shots = boundary.n_shots;
    let k = n_real / (n_shots.max(1) * n_gates);
    let mut tagged: Vec<(
        u32,
        CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>,
    )> = vec![
        (
            program.log_size,
            generate_prog_slot_preprocessed(program.log_size),
        ),
        (
            log_n_rows,
            generate_enabler_preprocessed(n_real, padded_rows),
        ),
        (
            log_n_rows,
            generate_shot_id_preprocessed(n_real, padded_rows, k, n_gates),
        ),
        (
            log_n_rows,
            generate_pc_preprocessed(n_real, padded_rows, k, n_gates),
        ),
        (
            log_n_rows,
            generate_pc_in_prog_preprocessed(n_real, padded_rows, k, n_gates),
        ),
    ];
    tagged.extend(
        generate_boundary_preprocessed(n_shots, boundary.log_size)
            .into_iter()
            .map(|c| (boundary.log_size, c)),
    );
    // rc membership table sized at the DYNAMIC rc_log (single [0,2^rc_log) val column).
    tagged.extend(
        generate_rc_preprocessed(rc_log)
            .into_iter()
            .map(|c| (rc_log, c)),
    );
    tagged.sort_by_key(|(s, _)| *s); // stable: identical key+listing order as preprocessed_columns_sorted
    tagged.into_iter().map(|(_, c)| c).collect()
}

// Pack a flat sequence column for a vec_row.
pub(crate) fn pack_seq(values: &[u32], vec_row: usize) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(values[(vec_row << LOG_N_LANES) + lane])
    }))
}

#[allow(dead_code)] // used by the cuda/gpu-cuda trace-gen + byte-identity paths
fn limbs_to_state(limbs: &[u32; N_LIMBS]) -> [u8; STATE_BYTES] {
    let mut out = [0u8; STATE_BYTES];
    for (j, &limb) in limbs.iter().enumerate() {
        out[2 * j] = (limb & 0xFF) as u8;
        out[2 * j + 1] = ((limb >> 8) & 0xFF) as u8;
    }
    out
}

#[inline]
#[allow(dead_code)] // used by the cuda/gpu-cuda trace-gen + byte-identity paths
fn qubit_decode(q: u16) -> (u32, u32, u32) {
    let limb_idx = (q as u32) / LIMB_BITS as u32;
    let bit_pos = (q as u32) % LIMB_BITS as u32;
    let mask = 1u32 << bit_pos;
    (limb_idx, bit_pos, mask)
}

fn delta_to_m31(delta: i64) -> u32 {
    // delta in {-1,0,1}, represented in M31.
    if delta >= 0 {
        delta as u32
    } else {
        (M31_MODULUS_U32 as i64 + delta) as u32
    }
}

/// One memory access: reads (prev_ts, v_before) from `last_*[addr]`; the access ts is the inlined
/// program-order `pc + 1`. Returns the `AccessCols` with `v = v_before` and the single rc diff
/// `d = ts - prev_ts - 1 = pc - prev_ts` (>= 0) that the rc-table lookup range-checks. The caller
/// updates `last_*[addr]` to the post-access ts/value.
fn do_access(addr: u32, pc: u32, last_ts: &[u32], last_val: &[u32]) -> AccessCols {
    let a = addr as usize;
    let prev_ts = last_ts[a];
    let v_before = last_val[a];
    let ts = pc + 1;
    debug_assert!(
        ts > prev_ts,
        "ts {ts} must exceed prev_ts {prev_ts} (program order)"
    );
    let d = ts - prev_ts - 1; // = pc - prev_ts; completeness (d < 2^TS_RC_BITS) checked in build_rows.
    debug_assert!(
        (d as u64) < (1u64 << TS_RC_BITS),
        "diff {d} exceeds range-check bound"
    );
    AccessCols {
        addr,
        prev_ts,
        v: v_before,
        d,
    }
}

/// Bit `addr` of a little-endian byte state (bit `addr` = byte `addr/8`, bit `addr%8`).
#[inline]
fn qubit_bit(bytes: &[u8], addr: usize) -> u32 {
    ((bytes[addr / 8] >> (addr % 8)) & 1) as u32
}

// Interaction traces
#[inline]
fn pack(lane: &[&Row; LANE_COUNT], get: impl Fn(&Row) -> u32) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|l| {
        BaseField::from_u32_unchecked(get(lane[l]))
    }))
}

/// Packs one boundary field over a vec_row's lanes.
fn pack_boundary(
    bnd: &BoundaryTable,
    vec_row: usize,
    f: &dyn Fn(&BoundaryRow) -> u32,
) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(f(&bnd.rows[(vec_row << LOG_N_LANES) + lane]))
    }))
}

// Pack a decoded value computed from the qubit index for the qdecode table.
#[allow(dead_code)] // used by the cuda/gpu-cuda interaction paths
fn pack_decode(vec_row: usize, f: impl Fn(usize) -> u32) -> PackedM31 {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(f((vec_row << LOG_N_LANES) + lane))
    }))
}

// ==== GPU (CUDA) trace-gen (absorbed from gpu_tracegen.rs) ====

// The absorbed `gpu_tracegen` module was wholly gated by `gpu-cuda` (it pulls in `cudarc`), so it is
// kept as a single `gpu-cuda`-gated inline module here and re-exported so the callers' `tracegen::<fn>`
// paths (and the previously `gpu_tracegen::<fn>` paths, now `tracegen::<fn>`) resolve unchanged.
#[cfg(feature = "gpu-cuda")]
pub(crate) use gpu::*;

#[cfg(feature = "gpu-cuda")]
mod gpu {
    use super::*;
    use std::sync::{Arc, OnceLock};

    // Multi-GPU: the base GPU ordinal this host thread proves on. Producer threads call `set_base_gpu(n)`;
    // every `cuda_device()` / module-cache access keys off it so tracegen lands on the same device as the
    // backend commit. Default 0 (single-GPU path is byte-identical to before).
    thread_local! {
        static BASE_GPU_ORDINAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Set this host thread's base GPU ordinal (multi-GPU producer setup); call ONCE before any GPU work.
    /// Records the ordinal for tracegen (`cuda_device`) and, under the device-resident backend, binds the
    /// backend's CUDA runtime current-device for this thread.
    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn set_base_gpu(ordinal: usize) {
        BASE_GPU_ORDINAL.with(|c| c.set(ordinal));
        // Explicitly bind the backend runtime current-device: cudarc binds the primary context lazily, but
        // the backend may issue a runtime-API call before that, so set it here to remove the ordering dep.
        #[cfg(feature = "cuda")]
        {
            let rc = unsafe { stwo::stwo_cuda::bindings::cuda_set_device(ordinal as i32) };
            assert_eq!(
                rc, 0,
                "cuda_set_device({ordinal}) failed on producer thread"
            );
        }
    }

    /// Visible CUDA device count (0 on error / no backend). Used to clamp the GATE_AIR_BASE_GPUS knob.
    #[cfg(all(feature = "gpu-cuda", feature = "cuda"))]
    pub(crate) fn backend_device_count() -> usize {
        let n = unsafe { stwo::stwo_cuda::bindings::cuda_device_count() };
        n.max(0) as usize
    }

    /// The current host thread's base GPU ordinal (0 unless a producer set it).
    pub(crate) fn base_gpu_ordinal() -> usize {
        BASE_GPU_ORDINAL.with(|c| c.get())
    }

    /// Build a private rayon `ThreadPool` whose every worker is bound to CUDA device `gpu`. The producer
    /// proves its shard inside this pool (`pool.install`) so the OODS-phase rayon fan-outs dispatch to
    /// device-bound workers, not the global pool. SIGSEGV fix: global-pool workers stay on device 0, so on
    /// device N they dereference device-N pointers while current-device-0 => illegal address; binding each
    /// worker via `set_base_gpu(gpu)` makes the fan-out run on device N. Thread count is a tuning knob
    /// (default 4) supplied by the caller (main reads `GATE_AIR_OODS_POOL_THREADS`) —
    /// correctness-independent, since the fan-outs are order-independent reductions/maps (byte-identical
    /// to the global pool).
    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn build_device_bound_pool(gpu: usize, num_threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("gate-air-oods-gpu{gpu}-{i}"))
            .start_handler(move |_| set_base_gpu(gpu))
            .build()
            .unwrap_or_else(|e| panic!("failed to build device-bound OODS pool for gpu {gpu}: {e}"))
    }

    /// Shared cudarc handle on the calling thread's target device primary CUDA context, cached
    /// process-wide per ordinal. The backend (stwo_cuda) targets the SAME device's primary context, so
    /// device pointers from these cudarc K1/K4 kernels interoperate with the backend's commit on that
    /// device. Single-GPU path (ordinal 0) is byte-identical to the previous single-`OnceLock` behavior.
    pub(crate) fn cuda_device() -> Result<Arc<cudarc::driver::CudaDevice>, String> {
        const MAX_BASE_GPUS: usize = 16;
        static DEVS: [OnceLock<Arc<cudarc::driver::CudaDevice>>; MAX_BASE_GPUS] =
            [const { OnceLock::new() }; MAX_BASE_GPUS];
        let ord = base_gpu_ordinal();
        let slot = DEVS
            .get(ord)
            .ok_or_else(|| format!("base gpu ordinal {ord} >= {MAX_BASE_GPUS}"))?;
        if let Some(d) = slot.get() {
            // Bind this thread to the device's primary context (needed when the cached handle is first
            // touched from a new thread — cudarc bind_to_thread contract).
            d.bind_to_thread()
                .map_err(|e| format!("bind_to_thread(dev {ord}): {e}"))?;
            return Ok(d.clone());
        }
        let d = cudarc::driver::CudaDevice::new(ord)
            .map_err(|e| format!("CudaDevice::new({ord}): {e}"))?;
        let _ = slot.set(d.clone());
        Ok(d)
    }

    /// Process-level module cache: load the AOT fatbin (from cuda/*.cu) ONCE per process instead of per
    /// shard, guarded by a `OnceLock`. Returns the module name + a `bool` (true on the first load).
    #[cfg(feature = "gpu-cuda")]
    fn gate_sim_module(
        dev: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(&'static str, bool), String> {
        // Per-ordinal guard: cudarc registers a loaded module on the specific CudaDevice's CUcontext, so
        // each device must load the fatbin once (a shared guard would leave other devices' `get_func`
        // unresolved). Keyed by ordinal; slot 0 => same one-load behavior as the single-GPU path.
        const MAX_BASE_GPUS: usize = 16;
        static LOADED: [OnceLock<Result<(), String>>; MAX_BASE_GPUS] =
            [const { OnceLock::new() }; MAX_BASE_GPUS];
        let ord = dev.ordinal();
        let guard = LOADED
            .get(ord)
            .ok_or_else(|| format!("gate_sim_module: ordinal {ord} >= {MAX_BASE_GPUS}"))?;
        let mut first = false;
        let res = guard.get_or_init(|| {
            first = true;
            // AOT: build.rs compiled cuda/gate_sim.cu -> $OUT_DIR/gate_sim.fatbin. Load it via
            // `Ptx::from_file` (cudarc's `PtxKind::File` -> `cuModuleLoad`, which the driver resolves for
            // a fatbin). Note: the byte-string `Ptx` paths NUL-terminate their input, so a binary fatbin
            // must go through the file path, not `from_src`/an image byte vec (PtxKind::Image is private).
            let ptx = cudarc::nvrtc::Ptx::from_file(concat!(env!("OUT_DIR"), "/gate_sim.fatbin"));
            dev.load_ptx(
                ptx,
                "gate_sim_mod",
                &["prog_slot_meta", "gate_sim_states", "gate_sim"],
            )
            .map_err(|e| format!("load_ptx: {e}"))?;
            Ok(())
        });
        res.clone().map(|()| ("gate_sim_mod", first))
    }

    /// N4 — process-level cache for the K4 INTERACTION kernel module. See [`gate_sim_module`].
    #[cfg(feature = "gpu-cuda")]
    fn interaction_module(
        dev: &Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(&'static str, bool), String> {
        // PER-ORDINAL load guard (see gate_sim_module): the K4 module must load once per device.
        const MAX_BASE_GPUS: usize = 16;
        static LOADED: [OnceLock<Result<(), String>>; MAX_BASE_GPUS] =
            [const { OnceLock::new() }; MAX_BASE_GPUS];
        let ord = dev.ordinal();
        let guard = LOADED
            .get(ord)
            .ok_or_else(|| format!("interaction_module: ordinal {ord} >= {MAX_BASE_GPUS}"))?;
        let names = [
            "logup_col_gen",
            "logup_finalize_col",
            "logup_cumsum_reduce",
            "logup_cumsum_shift",
            "ps_bit_reverse",
            "ps_circle_to_coset",
            "ps_coset_to_circle",
            "ps_block_scan",
            "ps_add_offsets",
        ];
        let mut first = false;
        let res = guard.get_or_init(|| {
            first = true;
            let ptx =
                cudarc::nvrtc::Ptx::from_file(concat!(env!("OUT_DIR"), "/interaction.fatbin"));
            dev.load_ptx(ptx, "logup_mod", &names)
                .map_err(|e| format!("load_ptx (K4): {e}"))?;
            Ok(())
        });
        res.clone().map(|()| ("logup_mod", first))
    }

    /// Allocate the thread-per-execution scratch buffers shared by every K1 launch wrapper: `d_rep`
    /// (n_shots*k*N_LIMBS rep-boundary states, K0→K1) and `d_slot` (n_gates*9 cyclic-predecessor constants
    /// per gate — the (gate, slot, wrap) of the previous access to the same addr, from which K1 derives
    /// `prev_ts`). `d_slot` is filled once here by the single-thread `prog_slot_meta` pass (n_gates tiny).
    #[cfg(feature = "gpu-cuda")]
    fn alloc_rep_and_slot(
        dev: &Arc<cudarc::driver::CudaDevice>,
        d_gates: &cudarc::driver::CudaSlice<u32>,
        k: u32,
        n_gates: u32,
        n_shots: u32,
    ) -> Result<
        (
            cudarc::driver::CudaSlice<u32>,
            cudarc::driver::CudaSlice<u32>,
        ),
        String,
    > {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let n_exec = (n_shots as u64) * (k as u64);
        let d_rep = dev
            .alloc_zeros::<u32>((n_exec as usize) * crate::N_LIMBS)
            .map_err(|e| format!("alloc rep_states: {e}"))?;
        let mut d_slot = dev
            .alloc_zeros::<u32>((n_gates as usize) * 9)
            .map_err(|e| format!("alloc slot_meta: {e}"))?;
        // Closed-form ts prepass: one thread walks the program once (n_gates tiny).
        let meta = dev
            .get_func("gate_sim_mod", "prog_slot_meta")
            .ok_or_else(|| "get_func prog_slot_meta".to_string())?;
        let one = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            meta.launch(one, (d_gates, n_gates, &mut d_slot))
                .map_err(|e| format!("launch prog_slot_meta: {e}"))?;
        }
        Ok((d_rep, d_slot))
    }

    /// Launch K0 `gate_sim_states`: thread-per-shot value-only pass filling `d_rep` with per-rep-boundary
    /// 512-qubit states (32 limbs each). Seeds K1's per-execution value chain.
    #[cfg(feature = "gpu-cuda")]
    fn launch_k0_states(
        dev: &Arc<cudarc::driver::CudaDevice>,
        d_gates: &cudarc::driver::CudaSlice<u32>,
        d_x: &cudarc::driver::CudaSlice<u32>,
        d_rep: &mut cudarc::driver::CudaSlice<u32>,
        k: u32,
        n_gates: u32,
        n_shots: u32,
    ) -> Result<(), String> {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let states = dev
            .get_func("gate_sim_mod", "gate_sim_states")
            .ok_or_else(|| "get_func gate_sim_states".to_string())?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n_shots.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            states
                .launch(cfg, (d_gates, d_x, d_rep, k, n_gates, n_shots))
                .map_err(|e| format!("launch gate_sim_states: {e}"))?;
        }
        Ok(())
    }

    // K1 gate-sim kernel source lives in `cuda/gate_sim.cu` (compiled AOT into `$OUT_DIR/gate_sim.fatbin`,
    // loaded in `gate_sim_module`); layout/constants mirror `main.rs`. Launch order:
    // prog_slot_meta -> K0 gate_sim_states -> K1 gate_sim.

    /// Number of gate_air main-trace columns the K1 kernel emits (see cuda/gate_sim.cu cell_at layout).
    pub const TRACE_COLUMNS: usize = 19;

    /// Run K1 on the GPU and copy the trace + histogram back to the host (byte-identity diagnostic; the
    /// production device-resident path is `gpu_gen_main_trace_device`). Returns (cols [column-major,
    /// TRACE_COLUMNS*padded_rows], rc_hist[2^rc_log]). Padding rows stay zero here.
    #[cfg(all(feature = "gpu-cuda", feature = "diag"))]
    pub fn gpu_gen_main_trace(
        gates_flat: &[u32],
        x_states: &[u32],
        off_lo: &[u32],
        off_hi: &[u32],
        k: u32,
        n_gates: u32,
        n_shots: u32,
        padded_rows: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        use cudarc::driver::{LaunchAsync, LaunchConfig};

        let dev = cuda_device()?;

        gate_sim_module(&dev)?;
        let func = dev
            .get_func("gate_sim_mod", "gate_sim")
            .ok_or_else(|| "get_func gate_sim".to_string())?;

        let _ = (off_lo, off_hi); // RcIndex offsets unused (arg slots repurposed for rep_states/slot_meta).
        let d_gates = dev
            .htod_copy(gates_flat.to_vec())
            .map_err(|e| format!("htod gates: {e}"))?;
        let d_x = dev
            .htod_copy(x_states.to_vec())
            .map_err(|e| format!("htod x_states: {e}"))?;
        let mut d_cols = dev
            .alloc_zeros::<u32>(TRACE_COLUMNS * padded_rows)
            .map_err(|e| format!("alloc cols: {e}"))?;
        // rc multiplicity histogram over the single diff `d`, sized to the fixed `1 << RC_LOG`: the kernel
        // bumps `rc_hist[d]` with d up to total_pc-1, so the buffer must cover [0,2^RC_LOG).
        let rc_hist_len = 1usize << crate::RC_LOG;
        let mut d_lo = dev
            .alloc_zeros::<u32>(rc_hist_len)
            .map_err(|e| format!("alloc rc_hist: {e}"))?;

        // Thread-per-execution scratch: rep-boundary states (K0 → K1) + closed-form ts constants.
        let (mut d_rep, d_slot) = alloc_rep_and_slot(&dev, &d_gates, k, n_gates, n_shots)?;

        let block = 256u32;
        // K0: thread-per-shot — fill rep-boundary states (value chain, linear in k).
        launch_k0_states(&dev, &d_gates, &d_x, &mut d_rep, k, n_gates, n_shots)?;
        // K1: thread-per-execution — one thread per (shot, rep). rep_states seeds the value chain,
        // slot_meta gives the closed-form ts; d_x is unused (arg compat).
        let n_exec = (n_shots as u64) * (k as u64);
        let grid = (n_exec.div_ceil(block as u64)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &d_gates,
                    &d_x,
                    &d_rep,
                    &d_slot,
                    &mut d_cols,
                    &mut d_lo,
                    k,
                    n_gates,
                    n_shots,
                    padded_rows as u64,
                ),
            )
            .map_err(|e| format!("launch gate_sim: {e}"))?;
        }

        dev.synchronize().map_err(|e| format!("sync: {e}"))?;

        let mut cols = vec![0u32; TRACE_COLUMNS * padded_rows];
        let mut lo = vec![0u32; rc_hist_len];
        dev.dtoh_sync_copy_into(&d_cols, &mut cols)
            .map_err(|e| format!("dtoh cols: {e}"))?;
        dev.dtoh_sync_copy_into(&d_lo, &mut lo)
            .map_err(|e| format!("dtoh rc_hist: {e}"))?;
        Ok((cols, lo))
    }

    /// Assert the GPU K1 trace (real rows + rc multiplicity histogram) is byte-identical to the CPU
    /// reference (`build_rows` + `cell_at` + `build_rc_table`), comparing the 19 main columns cell-by-cell
    /// and the histogram row-by-row. Run on a small fixture via `GATE_AIR_GPU_TEST=k1`.
    #[cfg(all(feature = "gpu-cuda", feature = "diag"))]
    pub fn k1_byte_identity(
        gates: &[crate::Gate],
        cases: &[crate::TestCase],
        k: usize,
        rc_lo: &super::RcIndex,
        _rc_hi: &super::RcIndex, // unused (single-`d` rc); kept for call-site compat.
    ) -> Result<(), String> {
        let n_gates = gates.len();
        let n_shots = cases.len();
        let real_rows = n_shots * k * n_gates;
        let padded_rows = real_rows
            .next_power_of_two()
            .max(1 << (crate::LOG_N_LANES + 2)); // matches main.rs

        // CPU reference; boundary is a separate component not covered by K1, so ignore it. `build_rc_table`
        // gives the CPU rc multiplicity histogram the K1 device histogram must match.
        let (rows, _boundary) = build_rows(gates, cases, k).map_err(|e| e.to_string())?;
        if rows.len() != real_rows {
            return Err(format!(
                "rows.len()={} != real_rows={}",
                rows.len(),
                real_rows
            ));
        }
        let rc_log = crate::RC_LOG;
        let cpu_rc = super::build_rc_table(&rows, rc_log);

        // Host-prep the flat GPU inputs (mirror state_to_limbs / the gate fields / RcIndex offsets).
        let mut gates_flat = Vec::with_capacity(n_gates * 4);
        for g in gates {
            gates_flat.push(g.opcode as u32);
            gates_flat.push(g.target as u32);
            gates_flat.push(g.ctrl_a as u32);
            gates_flat.push(g.ctrl_b as u32);
        }
        let mut x_states = Vec::with_capacity(n_shots * crate::N_LIMBS);
        for c in cases {
            let bytes = hex::decode(&c.x_hex).map_err(|e| format!("decode x_hex: {e}"))?;
            x_states.extend_from_slice(&state_to_limbs(&bytes));
        }
        // off_lo/off_hi are unused by the kernel now (arg-compat only).
        let off_lo: Vec<u32> = (0..crate::LIMB_BITS)
            .map(|p| rc_lo.offset[p] as u32)
            .collect();
        let off_hi = off_lo.clone();

        let (cols, hist) = gpu_gen_main_trace(
            &gates_flat,
            &x_states,
            &off_lo,
            &off_hi,
            k as u32,
            n_gates as u32,
            n_shots as u32,
            padded_rows,
        )?;

        // Compare the 19 main columns over the real rows.
        let mut mismatches = 0usize;
        let mut samples = Vec::new();
        for r in 0..real_rows {
            for c in 0..TRACE_COLUMNS {
                let got = cols[c * padded_rows + r];
                let want = cell_at(&rows[r], c);
                if got != want {
                    mismatches += 1;
                    if samples.len() < 10 {
                        samples.push(format!("row {r} col {c}: gpu={got} cpu={want}"));
                    }
                }
            }
        }

        // Compare the rc multiplicity histogram; the device histogram is indexed directly by `d`
        // (row_of(d) == d), matching RcTable's row order.
        let mut hist_mismatches = 0usize;
        for i in 0..cpu_rc.multiplicity.len() {
            if hist[i] != cpu_rc.multiplicity[i] {
                hist_mismatches += 1;
                if samples.len() < 20 {
                    samples.push(format!(
                        "hist row {i}: gpu={} cpu={}",
                        hist[i], cpu_rc.multiplicity[i]
                    ));
                }
            }
        }

        eprintln!(
            "[K1 byte-identity] real_rows={real_rows} padded_rows={padded_rows} \
         col_mismatches={mismatches} hist_mismatches={hist_mismatches}"
        );
        for s in &samples {
            eprintln!("    {s}");
        }

        if mismatches == 0 && hist_mismatches == 0 {
            eprintln!(
                "[K1 byte-identity] PASS — GPU trace == CPU trace (19 main columns + rc histogram)"
            );
            Ok(())
        } else {
            Err(format!(
            "K1 byte-identity FAILED: {mismatches} column + {hist_mismatches} histogram mismatches"
        ))
        }
    }

    // K4: CUDA LogUp interaction trace (full on-device). Generates gate_air's main-component interaction
    // M31 columns (N_LOGUP_COLS=5 batches × 4 coords = 20) + claimed_sum, byte-identical to the CPU
    // `gen_main_interaction` / `LogupTraceGenerator`, consuming K1's main-trace columns (no re-simulation).
    // Per LogUp column (sequential, col k accumulates onto k-1): logup_col_gen combines the batch tuples
    // into (num, denom), logup_finalize_col computes value = num·denom^{-1} + running sum. On the last
    // column: cumsum_reduce -> claimed_sum, cumsum_shift subtracts claimed_sum/2^log_size, then a per-coord
    // inclusive prefix sum (bit-reverse -> circle→coset -> scan -> coset→circle -> bit-reverse) matching
    // stwo's `inclusive_prefix_sum`. Field math is transcribed exactly from stwo (qm31.rs/cm31.rs: CM31
    // i^2=-1, QM31 R=(2,1)) — NOT obelyzk's u^2=2 convention. Validated by `k4_byte_identity`.

    /// Number of LogUp columns (batches) gate_air's main component emits: 10 relation entries (3 qubitmem
    /// Use+Yield pairs + 3 rc single-`d` terms + 1 program singleton) folded by `finalize_logup_in_pairs`
    /// into 5 batches. Order MUST match `gen_main_interaction` (main.rs) exactly.
    pub const N_LOGUP_COLS: usize = 5;
    /// Number of M31 interaction columns committed = 5 × 4 = 20.
    pub const N_INTERACTION_COLS: usize = N_LOGUP_COLS * 4;
    /// GateRel width (= `relation!(GateRel, 6)`): number of alpha powers uploaded (widest tuple =
    /// program = tag + 5 payload = 6).
    pub const GATE_REL_WIDTH: usize = 6;

    // K4 interaction kernel source lives in `cuda/interaction.cu` (compiled AOT into
    // `$OUT_DIR/interaction.fatbin`, loaded in `interaction_module`).

    /// Run the full K4 interaction pipeline on the GPU and copy the 20 interaction columns + claimed_sum
    /// back to the host. Inputs: host-side main trace `cols` (column-major), drawn `z`/`alpha_powers`, dims.
    /// Returns (interaction_cols [column-major], claimed_sum [4 M31]).
    #[cfg(all(feature = "gpu-cuda", feature = "diag"))]
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_gen_interaction(
        cols: &[u32],
        z: [u32; 4],
        alpha_powers: &[[u32; 4]],
        padded_rows: usize,
        n_gates: u32,
        real_rows: u64,
        shot_stride: u64, // = k * n_gates
    ) -> Result<(Vec<u32>, [u32; 4]), String> {
        use cudarc::driver::{LaunchAsync, LaunchConfig};

        assert_eq!(alpha_powers.len(), GATE_REL_WIDTH, "alpha_powers width");
        assert!(padded_rows.is_power_of_two(), "padded_rows must be 2^k");

        let dev = cuda_device()?;

        interaction_module(&dev)?;
        let get = |n: &str| {
            dev.get_func("logup_mod", n)
                .ok_or_else(|| format!("get_func {n}"))
        };

        // Upload main trace + challenges.
        let d_cols = dev
            .htod_copy(cols.to_vec())
            .map_err(|e| format!("htod cols: {e}"))?;
        let mut ap_flat = Vec::with_capacity(GATE_REL_WIDTH * 4);
        for p in alpha_powers {
            ap_flat.extend_from_slice(p);
        }
        let d_ap = dev
            .htod_copy(ap_flat)
            .map_err(|e| format!("htod ap: {e}"))?;
        // Positional dims for K4's enabler/shot_id/pc recompute; one pointer arg keeps the launch tuple
        // within cudarc's LaunchAsync arity cap.
        let d_dims = dev
            .htod_copy(vec![real_rows, shot_stride])
            .map_err(|e| format!("htod dims: {e}"))?;

        let mut d_inter = dev
            .alloc_zeros::<u32>(N_INTERACTION_COLS * padded_rows)
            .map_err(|e| format!("alloc inter: {e}"))?;
        let mut d_num = dev
            .alloc_zeros::<u32>(4 * padded_rows)
            .map_err(|e| format!("alloc num: {e}"))?;
        let mut d_denom = dev
            .alloc_zeros::<u32>(4 * padded_rows)
            .map_err(|e| format!("alloc denom: {e}"))?;

        let block = 256u32;
        let grid = (padded_rows as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Per LogUp column: col_gen(pair) then finalize_col(pair) (sequential — k accumulates onto k-1).
        for k in 0..N_LOGUP_COLS as u32 {
            unsafe {
                get("logup_col_gen")?
                    .launch(
                        cfg,
                        (
                            &d_cols,
                            padded_rows as u64,
                            z[0],
                            z[1],
                            z[2],
                            z[3],
                            &d_ap,
                            k,
                            n_gates,
                            &d_dims,
                            &mut d_num,
                            &mut d_denom,
                        ),
                    )
                    .map_err(|e| format!("launch col_gen[{k}]: {e}"))?;
                get("logup_finalize_col")?
                    .launch(cfg, (k, padded_rows as u64, &d_num, &d_denom, &mut d_inter))
                    .map_err(|e| format!("launch finalize[{k}]: {e}"))?;
            }
        }

        // claimed_sum = Σ last column (per coord), then subtract cumsum_shift.
        let last_k = (N_LOGUP_COLS - 1) as u32;
        let mut d_sums = dev
            .alloc_zeros::<u32>(4)
            .map_err(|e| format!("alloc sums: {e}"))?;
        let red_block = 256u32;
        let red_grid = ((padded_rows as u32).div_ceil(red_block)).min(1024);
        let red_cfg = LaunchConfig {
            grid_dim: (red_grid, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: 4 * red_block * 4, // 4 coords × blockDim × u32
        };
        unsafe {
            get("logup_cumsum_reduce")?
                .launch(red_cfg, (padded_rows as u64, last_k, &d_inter, &mut d_sums))
                .map_err(|e| format!("launch cumsum_reduce: {e}"))?;
        }
        let mut claimed_sum = [0u32; 4];
        dev.dtoh_sync_copy_into(&d_sums, &mut claimed_sum)
            .map_err(|e| format!("dtoh claimed_sum: {e}"))?;

        unsafe {
            get("logup_cumsum_shift")?
                .launch(
                    cfg,
                    (
                        padded_rows as u64,
                        last_k,
                        padded_rows as u32,
                        &d_sums,
                        &mut d_inter,
                    ),
                )
                .map_err(|e| format!("launch cumsum_shift: {e}"))?;
        }

        // Inclusive prefix sum on each of the 4 coordinate columns of the last LogUp column.
        let bits = padded_rows.trailing_zeros();
        let mut d_tmp = dev
            .alloc_zeros::<u32>(padded_rows)
            .map_err(|e| format!("alloc ps tmp: {e}"))?;
        for j in 0..4u64 {
            let offset = ((last_k as u64) * 4 + j) * padded_rows as u64;
            prefix_sum_column(
                &dev,
                &mut d_inter,
                offset,
                padded_rows,
                bits,
                &mut d_tmp,
                &get,
            )?;
        }

        dev.synchronize().map_err(|e| format!("sync: {e}"))?;

        let mut inter = vec![0u32; N_INTERACTION_COLS * padded_rows];
        dev.dtoh_sync_copy_into(&d_inter, &mut inter)
            .map_err(|e| format!("dtoh inter: {e}"))?;
        Ok((inter, claimed_sum))
    }

    /// stwo-semantics inclusive prefix sum of one column slice `col[offset .. offset+n)`:
    /// bit-reverse → CircleDomain→Coset → inclusive scan → Coset→CircleDomain → bit-reverse.
    #[cfg(feature = "gpu-cuda")]
    fn prefix_sum_column<F>(
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        d_col: &mut cudarc::driver::CudaSlice<u32>,
        offset: u64,
        n: usize,
        bits: u32,
        d_tmp: &mut cudarc::driver::CudaSlice<u32>,
        get: &F,
    ) -> Result<(), String>
    where
        F: Fn(&str) -> Result<cudarc::driver::CudaFunction, String>,
    {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let block = 256u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            get("ps_bit_reverse")?
                .launch(cfg, (&mut *d_col, offset, n as u32, bits))
                .map_err(|e| format!("ps_bit_reverse: {e}"))?;
            get("ps_circle_to_coset")?
                .launch(cfg, (&*d_col, offset, &mut *d_tmp, n as u32))
                .map_err(|e| format!("ps_circle_to_coset: {e}"))?;
        }
        // Inclusive scan over d_tmp[0..n].
        scan_inplace(dev, d_tmp, n, get)?;
        unsafe {
            get("ps_coset_to_circle")?
                .launch(cfg, (&*d_tmp, &mut *d_col, offset, n as u32))
                .map_err(|e| format!("ps_coset_to_circle: {e}"))?;
            get("ps_bit_reverse")?
                .launch(cfg, (&mut *d_col, offset, n as u32, bits))
                .map_err(|e| format!("ps_bit_reverse(2): {e}"))?;
        }
        Ok(())
    }

    /// In-place multi-block inclusive scan of `d[0..n]` (M31 add). Recurses on block sums.
    #[cfg(feature = "gpu-cuda")]
    fn scan_inplace<F>(
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        d: &mut cudarc::driver::CudaSlice<u32>,
        n: usize,
        get: &F,
    ) -> Result<(), String>
    where
        F: Fn(&str) -> Result<cudarc::driver::CudaFunction, String>,
    {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        let block = 256usize;
        let blocks = n.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (block * 4) as u32,
        };
        let mut d_block_sums = dev
            .alloc_zeros::<u32>(blocks)
            .map_err(|e| format!("alloc block_sums: {e}"))?;
        unsafe {
            get("ps_block_scan")?
                .launch(cfg, (&mut *d, &mut d_block_sums, n as u32))
                .map_err(|e| format!("ps_block_scan: {e}"))?;
        }
        if blocks > 1 {
            scan_inplace(dev, &mut d_block_sums, blocks, get)?;
            let add_cfg = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                get("ps_add_offsets")?
                    .launch(add_cfg, (&mut *d, &d_block_sums, n as u32))
                    .map_err(|e| format!("ps_add_offsets: {e}"))?;
            }
        }
        Ok(())
    }

    /// Assert the GPU K4 interaction trace (20 M31 columns + claimed_sum) is byte-identical to the CPU
    /// `gen_main_interaction`, using a fixed `GateRel::dummy()` (z, alpha) so both sides see the same
    /// challenges. Run via `GATE_AIR_GPU_TEST=k4`.
    #[cfg(all(feature = "gpu-cuda", feature = "diag"))]
    pub fn k4_byte_identity(
        gates: &[crate::Gate],
        cases: &[crate::TestCase],
        k: usize,
        _rc_lo: &super::RcIndex, // unused (rc diff `d` comes from the K1 main trace); call-site compat.
        _rc_hi: &super::RcIndex, // unused (single-`d` rc); call-site compat.
    ) -> Result<(), String> {
        use stwo::prover::backend::Column;

        let n_gates = gates.len();
        let n_shots = cases.len();
        let real_rows = n_shots * k * n_gates;
        let padded_rows = real_rows
            .next_power_of_two()
            .max(1 << (crate::LOG_N_LANES + 2));
        let log_n_rows = padded_rows.ilog2();

        // CPU reference: build rows + fixed dummy elements + gen_main_interaction.
        let (rows, _boundary) = build_rows(gates, cases, k).map_err(|e| e.to_string())?;
        let elements = crate::LookupElements::dummy();
        let (cpu_cols, cpu_sum) =
            gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);

        // GPU main trace (host) — reuse K1.
        let mut gates_flat = Vec::with_capacity(n_gates * 4);
        for g in gates {
            gates_flat.push(g.opcode as u32);
            gates_flat.push(g.target as u32);
            gates_flat.push(g.ctrl_a as u32);
            gates_flat.push(g.ctrl_b as u32);
        }
        let mut x_states = Vec::with_capacity(n_shots * crate::N_LIMBS);
        for c in cases {
            let bytes = hex::decode(&c.x_hex).map_err(|e| format!("decode x_hex: {e}"))?;
            x_states.extend_from_slice(&state_to_limbs(&bytes));
        }
        let off_lo: Vec<u32> = (0..crate::LIMB_BITS)
            .map(|p| _rc_lo.offset[p] as u32)
            .collect();
        let off_hi = off_lo.clone(); // unused by the kernel; arg-list compat.
        let (main_cols, _lo) = gpu_gen_main_trace(
            &gates_flat,
            &x_states,
            &off_lo,
            &off_hi,
            k as u32,
            n_gates as u32,
            n_shots as u32,
            padded_rows,
        )?;

        let (z_qm, alpha_powers_qm) = extract_z_alpha(&elements.qubitmem);
        let z = secure_to_m31x4(z_qm);
        let alpha_powers: Vec<[u32; 4]> = alpha_powers_qm
            .iter()
            .map(|p| secure_to_m31x4(*p))
            .collect();

        let (gpu_inter, gpu_sum_arr) = gpu_gen_interaction(
            &main_cols,
            z,
            &alpha_powers,
            padded_rows,
            n_gates as u32,
            real_rows as u64,
            (k * n_gates) as u64,
        )?;

        // Compare the 20 interaction columns over all padded rows (the prefix sum spans them).
        let mut mismatches = 0usize;
        let mut samples: Vec<String> = Vec::new();
        for c in 0..N_INTERACTION_COLS {
            let cpu_c = cpu_cols[c].values.to_cpu();
            for r in 0..padded_rows {
                let got = gpu_inter[c * padded_rows + r];
                let want = cpu_c[r].0;
                if got != want {
                    mismatches += 1;
                    if samples.len() < 10 {
                        samples.push(format!("col {c} row {r}: gpu={got} cpu={want}"));
                    }
                }
            }
        }
        let cpu_sum_arr = secure_to_m31x4(cpu_sum);
        let sum_ok = cpu_sum_arr == gpu_sum_arr;

        eprintln!(
        "[K4 byte-identity] real_rows={real_rows} padded_rows={padded_rows} log_rows={log_n_rows} \
         col_mismatches={mismatches} claimed_sum gpu={gpu_sum_arr:?} cpu={cpu_sum_arr:?} sum_ok={sum_ok}"
    );
        for s in &samples {
            eprintln!("    {s}");
        }

        if mismatches == 0 && sum_ok {
            eprintln!(
            "[K4 byte-identity] PASS — GPU interaction == CPU interaction (20 cols + claimed_sum)"
        );
            Ok(())
        } else {
            Err(format!(
                "K4 byte-identity FAILED: {mismatches} col mismatches; sum_ok={sum_ok}"
            ))
        }
    }

    /// gate_air's drawn LogUp relation as M31x4 coords: (z, alpha_powers[0..GATE_REL_WIDTH]).
    /// Fed to the GPU constraint kernel via `stwo_constraint_framework::set_gate_air_relation`.
    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn gate_air_relation_m31x4(rel: &crate::air::GateRel) -> ([u32; 4], Vec<[u32; 4]>) {
        let (z, alpha_powers) = extract_z_alpha(rel);
        (
            secure_to_m31x4(z),
            alpha_powers.into_iter().map(secure_to_m31x4).collect(),
        )
    }

    /// SecureField (QM31) → [a, b, c, d] M31 coordinates (matching from_m31_array order).
    #[cfg(feature = "gpu-cuda")]
    fn secure_to_m31x4(x: stwo::core::fields::qm31::SecureField) -> [u32; 4] {
        let arr = x.to_m31_array();
        [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
    }

    /// Recover (z, alpha_powers[0..GATE_REL_WIDTH]) from a `GateRel` using only its public
    /// `Relation::combine` (the inner `LookupElements` is private): combine(v) = Σ α^i·v[i] − z, so
    /// z = −combine([0]) and α^i = combine(unit_i) + z.
    #[cfg(feature = "gpu-cuda")]
    fn extract_z_alpha(
        rel: &crate::air::GateRel,
    ) -> (
        stwo::core::fields::qm31::SecureField,
        Vec<stwo::core::fields::qm31::SecureField>,
    ) {
        use stwo::core::fields::m31::BaseField;
        use stwo::core::fields::qm31::SecureField;
        use stwo_constraint_framework::Relation;

        let zero = BaseField::from_u32_unchecked(0);
        let one = BaseField::from_u32_unchecked(1);
        let zero_vec = [zero];
        let z: SecureField = -Relation::<BaseField, SecureField>::combine(rel, &zero_vec);
        let alpha_powers = (0..GATE_REL_WIDTH)
            .map(|i| {
                let mut v = vec![zero; GATE_REL_WIDTH];
                v[i] = one;
                Relation::<BaseField, SecureField>::combine(rel, &v) + z
            })
            .collect();
        (z, alpha_powers)
    }

    // Device-resident output: feed K1/K4's GPU columns straight into the CudaBackend commit with no host
    // round-trip. Interop between two allocators: K1/K4 write cudarc `CudaSlice<u32>`, the CudaBackend
    // columns are stwo_cuda `BaseFieldVec`. Both run on the same primary CUDA context, so a device pointer
    // from one is a valid device-to-device copy operand for the other. Each committed column is a single
    // per-column D2D copy into a fresh `BaseFieldVec` wrapped as `CircleEvaluation<CudaBackend>` — chosen
    // over wrapping the cudarc pointer directly, which would put two owning allocators behind one Drop
    // (double-free / dangling once the kernel scope ends).

    /// cudarc `CudaSlice<u32>` device address as a raw `*const u32` (the integer
    /// `CUdeviceptr`, valid in the shared primary context).
    #[cfg(feature = "cuda")]
    fn cudarc_dptr(slice: &cudarc::driver::CudaSlice<u32>) -> *const u32 {
        use cudarc::driver::DevicePtr;
        (*slice.device_ptr()) as usize as *const u32
    }

    /// Copy one column (`padded_rows` u32s at element offset `col_off`) from a cudarc device buffer into a
    /// fresh `BaseFieldVec` via a device-to-device copy (no host round-trip), wrapped as a device-resident
    /// `CircleEvaluation<CudaBackend>`.
    #[cfg(feature = "gpu-cuda")]
    fn d2d_column(
        src: &cudarc::driver::CudaSlice<u32>,
        col_off: usize,
        padded_rows: usize,
        domain: stwo::core::poly::circle::CircleDomain,
    ) -> stwo::prover::poly::circle::CircleEvaluation<
        stwo::prover::backend::CudaBackend,
        stwo::core::fields::m31::BaseField,
        stwo::prover::poly::BitReversedOrder,
    > {
        use stwo::prover::poly::circle::CircleEvaluation;
        use stwo::prover::poly::BitReversedOrder;
        use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
        use stwo::stwo_cuda::bindings;

        let dst = BaseFieldVec::new_uninitialized(padded_rows);
        // SAFETY: `src` lives on the device-0 primary context (cudarc), `dst.device_ptr`
        // was allocated by NitrooZK on the same context; both spans are `padded_rows`
        // u32s and disjoint, so the D2D copy is in-bounds and non-overlapping.
        unsafe {
            let src_ptr = cudarc_dptr(src).add(col_off);
            bindings::copy_uint32_t_vec_from_device_to_device(
                src_ptr,
                dst.device_ptr,
                padded_rows as u32,
            );
        }
        CircleEvaluation::<_, _, BitReversedOrder>::new(domain, dst)
    }

    /// Device-resident K1: run `gpu_gen_main_trace`'s kernels and return the 19 main columns as
    /// device-resident `CircleEvaluation<CudaBackend>` (no host upload), the rc histogram (host, tiny),
    /// and the raw column-major device buffer `d_cols` so K4 can reuse it (no K0/K1 re-run). The returned
    /// `CircleEvaluation`s are independent D2D copies of `d_cols`'s columns, so returning `d_cols` does
    /// not alias them.
    #[cfg(feature = "gpu-cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_gen_main_trace_device(
        gates_flat: &[u32],
        x_states: &[u32],
        off_lo: &[u32],
        off_hi: &[u32],
        k: u32,
        n_gates: u32,
        n_shots: u32,
        padded_rows: usize,
        log_n_rows: u32,
    ) -> Result<
        (
            Vec<
                stwo::prover::poly::circle::CircleEvaluation<
                    stwo::prover::backend::CudaBackend,
                    stwo::core::fields::m31::BaseField,
                    stwo::prover::poly::BitReversedOrder,
                >,
            >,
            Vec<u32>,
            cudarc::driver::CudaSlice<u32>,
        ),
        String,
    > {
        let dev = cuda_device()?;
        // Upload the shard-invariant inputs here, then delegate to the device-buffer body. The base
        // precompute path uploads these once and calls the `_d` body directly; only `x_states` is per-shard.
        let d_gates = dev
            .htod_copy(gates_flat.to_vec())
            .map_err(|e| format!("htod gates: {e}"))?;
        let d_off_lo = dev
            .htod_copy(off_lo.to_vec())
            .map_err(|e| format!("htod off_lo: {e}"))?;
        let d_off_hi = dev
            .htod_copy(off_hi.to_vec())
            .map_err(|e| format!("htod off_hi: {e}"))?;
        gpu_gen_main_trace_device_d(
            &d_gates,
            x_states,
            &d_off_lo,
            &d_off_hi,
            k,
            n_gates,
            n_shots,
            padded_rows,
            log_n_rows,
        )
    }

    /// Device-buffer entry point for the device-resident K1. Identical to [`gpu_gen_main_trace_device`]
    /// except the shard-invariant `gates`/`off_lo`/`off_hi` are passed as ALREADY-UPLOADED device
    /// buffers (so the base precompute uploads them once and reuses them across every shard). Only
    /// `x_states` (per shard) is uploaded here.
    #[cfg(feature = "gpu-cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_gen_main_trace_device_d(
        d_gates: &cudarc::driver::CudaSlice<u32>,
        x_states: &[u32],
        d_off_lo: &cudarc::driver::CudaSlice<u32>,
        d_off_hi: &cudarc::driver::CudaSlice<u32>,
        k: u32,
        n_gates: u32,
        n_shots: u32,
        padded_rows: usize,
        log_n_rows: u32,
    ) -> Result<
        (
            Vec<
                stwo::prover::poly::circle::CircleEvaluation<
                    stwo::prover::backend::CudaBackend,
                    stwo::core::fields::m31::BaseField,
                    stwo::prover::poly::BitReversedOrder,
                >,
            >,
            Vec<u32>,
            cudarc::driver::CudaSlice<u32>,
        ),
        String,
    > {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        use stwo::core::poly::circle::CanonicCoset;

        let dev = cuda_device()?;

        gate_sim_module(&dev)?;
        let func = dev
            .get_func("gate_sim_mod", "gate_sim")
            .ok_or_else(|| "get_func gate_sim".to_string())?;

        let d_x = dev
            .htod_copy(x_states.to_vec())
            .map_err(|e| format!("htod x_states: {e}"))?;
        let cols_len = TRACE_COLUMNS * padded_rows;
        let mut d_cols = dev
            .alloc_zeros::<u32>(cols_len)
            .map_err(|e| format!("alloc cols: {e}"))?;
        // rc histogram sized to the fixed `1 << RC_LOG`: the kernel bumps `rc_hist[d]` with d up to
        // total_pc-1, so the buffer must cover [0,2^RC_LOG) to keep the atomic bumps in bounds on large-k
        // runs. Production ignores the returned histogram (the multiplicity witness is host-built).
        let rc_hist_len = 1usize << crate::RC_LOG;
        let mut d_lo = dev
            .alloc_zeros::<u32>(rc_hist_len)
            .map_err(|e| format!("alloc rc_hist: {e}"))?;
        let _ = (d_off_lo, d_off_hi); // RcIndex offsets unused (arg slots repurposed for rep_states/slot_meta).

        // Thread-per-execution scratch: rep-boundary states (K0 → K1) + closed-form ts constants.
        let (mut d_rep, d_slot) = alloc_rep_and_slot(&dev, d_gates, k, n_gates, n_shots)?;

        let block = 256u32;
        // K0: thread-per-shot — fill rep-boundary states (value chain, linear in k).
        launch_k0_states(&dev, d_gates, &d_x, &mut d_rep, k, n_gates, n_shots)?;
        // K1: thread-per-execution — one thread per (shot, rep); rep_states seeds the value chain, slot_meta
        // gives the closed-form ts. d_x is unused (compat).
        let n_exec = (n_shots as u64) * (k as u64);
        let grid = (n_exec.div_ceil(block as u64)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    d_gates,
                    &d_x,
                    &d_rep,
                    &d_slot,
                    &mut d_cols,
                    &mut d_lo,
                    k,
                    n_gates,
                    n_shots,
                    padded_rows as u64,
                ),
            )
            .map_err(|e| format!("launch gate_sim: {e}"))?;
        }

        dev.synchronize().map_err(|e| format!("sync: {e}"))?;

        // Device handoff: build the 19 CudaBackend columns from `d_cols` (column-major, column c at offset
        // c*padded_rows). No host copy.
        let domain = CanonicCoset::new(log_n_rows).circle_domain();
        let cols: Vec<_> = (0..TRACE_COLUMNS)
            .map(|c| d2d_column(&d_cols, c * padded_rows, padded_rows, domain))
            .collect();

        let mut lo = vec![0u32; rc_hist_len];
        dev.dtoh_sync_copy_into(&d_lo, &mut lo)
            .map_err(|e| format!("dtoh rc_hist: {e}"))?;
        dev.synchronize().map_err(|e| format!("sync hist: {e}"))?;
        // Return the resident `d_cols` buffer so K4 reads it directly (no K0/K1 re-run); it frees when the
        // caller drops it.
        Ok((cols, lo, d_cols))
    }

    /// The K1 main-trace buffer as consumed by K4: RESIDENT on the device — the `CudaSlice` K1 returned,
    /// held live through tree1 + K4. `gpu_gen_interaction_device` reads it in place for its kernel loop;
    /// the caller frees it via `free_after_k4` after K4 and before tree2.
    #[cfg(feature = "gpu-cuda")]
    pub struct MainTrace(cudarc::driver::CudaSlice<u32>);

    #[cfg(feature = "gpu-cuda")]
    impl MainTrace {
        /// Wrap K1's resident device buffer. Call AFTER the tree1 commit (which borrows into the resident
        /// buffer) and BEFORE the K4 interaction allocs / tree2 commit.
        pub fn from_k1(d_cols: cudarc::driver::CudaSlice<u32>) -> Result<Self, String> {
            Ok(MainTrace(d_cols))
        }

        /// Free the underlying device main-trace buffer now (after K4, before the tree2 commit), then
        /// synchronize so the free settles before tree2's first alloc — instead of relying on end-of-prove
        /// drop.
        pub fn free_after_k4(self) -> Result<(), String> {
            drop(self);
            let dev = cuda_device()?;
            dev.synchronize()
                .map_err(|e| format!("sync after main-trace free: {e}"))?;
            Ok(())
        }
    }

    /// Device-resident K4: run `gpu_gen_interaction`'s pipeline using the real drawn `LookupElements`
    /// (z/alpha recovered via `extract_z_alpha`) and return the 20 interaction columns as device-resident
    /// `CircleEvaluation<CudaBackend>` plus `claimed_sum`. `main` is the resident buffer K1 produced
    /// earlier in this base proof, read in place (no K0/K1 re-run).
    #[cfg(feature = "gpu-cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_gen_interaction_device(
        main: &MainTrace,
        n_gates: u32,
        padded_rows: usize,
        log_n_rows: u32,
        real_rows: u64,
        shot_stride: u64, // = k * n_gates
        elements: &crate::LookupElements,
    ) -> Result<
        (
            Vec<
                stwo::prover::poly::circle::CircleEvaluation<
                    stwo::prover::backend::CudaBackend,
                    stwo::core::fields::m31::BaseField,
                    stwo::prover::poly::BitReversedOrder,
                >,
            >,
            stwo::core::fields::qm31::SecureField,
        ),
        String,
    > {
        use cudarc::driver::{LaunchAsync, LaunchConfig};
        use stwo::core::fields::qm31::SecureField;
        use stwo::core::poly::circle::CanonicCoset;

        assert!(padded_rows.is_power_of_two(), "padded_rows must be 2^k");

        // Recover the real drawn (z, alpha_powers) from the relation's public combine.
        let (z_qm, alpha_powers_qm) = extract_z_alpha(&elements.qubitmem);
        let z = secure_to_m31x4(z_qm);
        let alpha_powers: Vec<[u32; 4]> = alpha_powers_qm
            .iter()
            .map(|p| secure_to_m31x4(*p))
            .collect();
        assert_eq!(alpha_powers.len(), GATE_REL_WIDTH, "alpha_powers width");

        let dev = cuda_device()?;

        // K4 reads the main-trace columns from the buffer K1 already produced (`main`); no K0/K1 re-run,
        // no re-upload, no histogram/rep scratch.
        let block = 256u32;

        interaction_module(&dev)?;
        let get = |n: &str| {
            dev.get_func("logup_mod", n)
                .ok_or_else(|| format!("get_func {n}"))
        };

        let mut ap_flat = Vec::with_capacity(GATE_REL_WIDTH * 4);
        for p in &alpha_powers {
            ap_flat.extend_from_slice(p);
        }
        let d_ap = dev
            .htod_copy(ap_flat)
            .map_err(|e| format!("htod ap: {e}"))?;
        // Positional dims for K4's enabler/shot_id/pc recompute.
        let d_dims = dev
            .htod_copy(vec![real_rows, shot_stride])
            .map_err(|e| format!("htod dims: {e}"))?;

        let mut d_inter = dev
            .alloc_zeros::<u32>(N_INTERACTION_COLS * padded_rows)
            .map_err(|e| format!("alloc inter: {e}"))?;
        let mut d_num = dev
            .alloc_zeros::<u32>(4 * padded_rows)
            .map_err(|e| format!("alloc num: {e}"))?;
        let mut d_denom = dev
            .alloc_zeros::<u32>(4 * padded_rows)
            .map_err(|e| format!("alloc denom: {e}"))?;

        let d_cols: &cudarc::driver::CudaSlice<u32> = &main.0;

        let grid = (padded_rows as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        for kk in 0..N_LOGUP_COLS as u32 {
            unsafe {
                get("logup_col_gen")?
                    .launch(
                        cfg,
                        (
                            d_cols,
                            padded_rows as u64,
                            z[0],
                            z[1],
                            z[2],
                            z[3],
                            &d_ap,
                            kk,
                            n_gates,
                            &d_dims,
                            &mut d_num,
                            &mut d_denom,
                        ),
                    )
                    .map_err(|e| format!("launch col_gen[{kk}]: {e}"))?;
                get("logup_finalize_col")?
                    .launch(
                        cfg,
                        (kk, padded_rows as u64, &d_num, &d_denom, &mut d_inter),
                    )
                    .map_err(|e| format!("launch finalize[{kk}]: {e}"))?;
            }
        }

        let last_k = (N_LOGUP_COLS - 1) as u32;
        let mut d_sums = dev
            .alloc_zeros::<u32>(4)
            .map_err(|e| format!("alloc sums: {e}"))?;
        let red_block = 256u32;
        let red_grid = ((padded_rows as u32).div_ceil(red_block)).min(1024);
        let red_cfg = LaunchConfig {
            grid_dim: (red_grid, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: 4 * red_block * 4,
        };
        unsafe {
            get("logup_cumsum_reduce")?
                .launch(red_cfg, (padded_rows as u64, last_k, &d_inter, &mut d_sums))
                .map_err(|e| format!("launch cumsum_reduce: {e}"))?;
        }
        let mut claimed_sum = [0u32; 4];
        dev.dtoh_sync_copy_into(&d_sums, &mut claimed_sum)
            .map_err(|e| format!("dtoh claimed_sum: {e}"))?;

        unsafe {
            get("logup_cumsum_shift")?
                .launch(
                    cfg,
                    (
                        padded_rows as u64,
                        last_k,
                        padded_rows as u32,
                        &d_sums,
                        &mut d_inter,
                    ),
                )
                .map_err(|e| format!("launch cumsum_shift: {e}"))?;
        }

        let bits = padded_rows.trailing_zeros();
        let mut d_tmp = dev
            .alloc_zeros::<u32>(padded_rows)
            .map_err(|e| format!("alloc ps tmp: {e}"))?;
        for j in 0..4u64 {
            let offset = ((last_k as u64) * 4 + j) * padded_rows as u64;
            prefix_sum_column(
                &dev,
                &mut d_inter,
                offset,
                padded_rows,
                bits,
                &mut d_tmp,
                &get,
            )?;
        }

        dev.synchronize().map_err(|e| format!("sync (K4): {e}"))?;

        // Device-to-device handoff: 20 interaction columns straight to CudaBackend.
        let domain = CanonicCoset::new(log_n_rows).circle_domain();
        let cols: Vec<_> = (0..N_INTERACTION_COLS)
            .map(|c| d2d_column(&d_inter, c * padded_rows, padded_rows, domain))
            .collect();

        let claimed = SecureField::from_m31_array([
            stwo::core::fields::m31::BaseField::from_u32_unchecked(claimed_sum[0]),
            stwo::core::fields::m31::BaseField::from_u32_unchecked(claimed_sum[1]),
            stwo::core::fields::m31::BaseField::from_u32_unchecked(claimed_sum[2]),
            stwo::core::fields::m31::BaseField::from_u32_unchecked(claimed_sum[3]),
        ]);
        Ok((cols, claimed))
    }
}
