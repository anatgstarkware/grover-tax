//! The AIR's PUBLIC preprocessed columns: verifier-known, purely POSITIONAL / SHAPE-derived tree-0
//! data (column identity + order + the value generators). This is the lower layer — it depends only
//! on `air` (encoding consts + the `PreProcessedColumnId` type), NEVER on `tracegen`/`prover` (no
//! staging structs). `tracegen` and the prover depend on THIS module to build the committed tree-0.
//!
//! Every generator takes PUBLIC SHAPE scalars (row counts / log-sizes) and computes the column
//! directly from the same positional formula the prover's staging structs used to fill their
//! preprocessed fields, so the produced columns are byte-identical to the staging-based ones.

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::air::N_QUBITS;

/// Number of preprocessed columns (count-only uses; the order is `preprocessed_column_ids`).
/// prog_slot + (enabler/shot_id/pc/pc_in_prog) + (bnd_shot/bnd_addr/bnd_enabler) + rc_val = 9.
pub(crate) const N_PREPROCESSED_COLS: usize = 9;

// Column identity + order

pub(crate) fn pp_id(id: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: id.to_owned() }
}

/// Each preprocessed column paired with its log_size, in canonical order then STABLE-sorted ascending
/// by size — the committed tree MUST be size-sorted (stwo's lifted Merkle sorts by length; the
/// in-circuit verifier does not re-sort). `gate_rc_val` is sized at the trusted `rc_log` (see RC_LOG),
/// never read from the proof — it sizes the [0,2^rc_log) table pinned by the preprocessed root.
pub(crate) fn preprocessed_columns_sorted(
    main_log_size: u32,
    program_log_size: u32,
    qubitmem_log_size: u32,
    rc_log: u32,
) -> Vec<(PreProcessedColumnId, u32)> {
    let mut cols = vec![
        (pp_id("gate_prog_slot"), program_log_size),
        // Shard-invariant positional main-trace columns (tree0). Sized with the main trace.
        (pp_id("gate_enabler"), main_log_size),
        (pp_id("gate_shot_id"), main_log_size),
        (pp_id("gate_pc"), main_log_size),
        (pp_id("gate_pc_in_prog"), main_log_size),
        // Qubit-memory positional columns. Sized with the qubitmem table.
        (pp_id("gate_bnd_shot"), qubitmem_log_size),
        (pp_id("gate_bnd_addr"), qubitmem_log_size),
        (pp_id("gate_bnd_enabler"), qubitmem_log_size),
        // ts-ordering range-check table membership (val[i]=i). Sized at the DYNAMIC rc_log.
        (pp_id("gate_rc_val"), rc_log),
    ];
    cols.sort_by_key(|&(_, s)| s); // stable: ties keep the listing order above
    cols
}

pub(crate) fn preprocessed_column_ids(
    main_log_size: u32,
    program_log_size: u32,
    qubitmem_log_size: u32,
    rc_log: u32,
) -> Vec<PreProcessedColumnId> {
    preprocessed_columns_sorted(main_log_size, program_log_size, qubitmem_log_size, rc_log)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

// Column generators (shape-derived)

/// Preprocessed slot-index column for the program table: `slot[i] = i` for every row in
/// `[0, 2^program_log_size)` (real gates and padding alike carry an in-range index sequence).
pub(crate) fn generate_prog_slot_preprocessed(
    program_log_size: u32,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let padded = 1usize << program_log_size;
    let vals: Vec<u32> = (0..padded as u32).collect();
    col_from_values(&vals)
}

/// Preprocessed `enabler` column: 1 on real rows, 0 on padding. SHARD-INVARIANT and POSITIONAL —
/// depends only on how many real rows (`n_real`) the (k, n_gates, n_shots) shape produces, never on
/// secret content. Used by the AIR both in the opcode one-hot constraint and as the LogUp numerator.
pub(crate) fn generate_enabler_preprocessed(
    n_real: usize,
    padded_rows: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let vals: Vec<u32> = (0..padded_rows).map(|i| (i < n_real) as u32).collect();
    col_from_values(&vals)
}

/// Preprocessed `shot_id` column: shot index of each row (= row / (k*n_gates)) on real rows, 0 on
/// padding. SHARD-INVARIANT and POSITIONAL (positional payload in the state-relation tuple).
pub(crate) fn generate_shot_id_preprocessed(
    n_real: usize,
    padded_rows: usize,
    k: usize,
    n_gates: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let shot_rows = (k * n_gates) as u32;
    let vals: Vec<u32> = (0..padded_rows)
        .map(|i| if i < n_real { i as u32 / shot_rows } else { 0 })
        .collect();
    col_from_values(&vals)
}

/// Preprocessed `pc` column: monotonic per-shot program counter (= row % (k*n_gates)) on real rows,
/// 0 on padding. SHARD-INVARIANT and POSITIONAL (positional payload in the state-relation tuple).
pub(crate) fn generate_pc_preprocessed(
    n_real: usize,
    padded_rows: usize,
    k: usize,
    n_gates: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let shot_rows = (k * n_gates) as u32;
    let vals: Vec<u32> = (0..padded_rows)
        .map(|i| if i < n_real { i as u32 % shot_rows } else { 0 })
        .collect();
    col_from_values(&vals)
}

/// Preprocessed pc_in_prog column for the main trace: `pc mod n_gates` on real rows, 0 on padding
/// (inert: padding has enabler 0).
pub(crate) fn generate_pc_in_prog_preprocessed(
    n_real: usize,
    padded_rows: usize,
    k: usize,
    n_gates: usize,
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let shot_rows = (k * n_gates) as u32;
    let ng = n_gates as u32;
    let vals: Vec<u32> = (0..padded_rows)
        .map(|i| {
            if i < n_real {
                (i as u32 % shot_rows) % ng
            } else {
                0
            }
        })
        .collect();
    col_from_values(&vals)
}

/// Preprocessed positional columns for the qubitmem table: (shot, addr, enabler) per row. Real rows
/// are the first `n_shots * N_QUBITS`; each real row `i` carries `shot = i / N_QUBITS`,
/// `addr = i % N_QUBITS`, `enabler = 1`; padding rows carry (0, 0, 0).
pub(crate) fn generate_qubitmem_preprocessed(
    n_shots: usize,
    qubitmem_log_size: u32,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let padded = 1usize << qubitmem_log_size;
    let real = n_shots * N_QUBITS;
    let nq = N_QUBITS as u32;
    let shot: Vec<u32> = (0..padded)
        .map(|i| if i < real { i as u32 / nq } else { 0 })
        .collect();
    let addr: Vec<u32> = (0..padded)
        .map(|i| if i < real { i as u32 % nq } else { 0 })
        .collect();
    // Real-row enabler: 1 for the first `n_shots*N_QUBITS` rows, 0 on padding. POSITIONAL /
    // shard-invariant (depends only on the shot count). Gates the qubitmem emission so a non-power-of-
    // two `n_shots*N_QUBITS` (e.g. 9024 shots) does not inject unmatched LogUp terms on padding rows.
    let enabler: Vec<u32> = (0..padded).map(|i| (i < real) as u32).collect();
    vec![
        col_from_values(&shot),
        col_from_values(&addr),
        col_from_values(&enabler),
    ]
}

/// Preprocessed rc-table membership column (val), the single column RangeCheckEval reads: a single
/// block enumerating exactly `[0, 2^rc_log)` with `val[i] = i` (every row a genuine member).
pub(crate) fn generate_rc_preprocessed(
    rc_log: u32,
) -> Vec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    let size = 1usize << rc_log;
    let val: Vec<u32> = (0..size as u32).collect();
    vec![col_from_values(&val)]
}

pub(crate) fn col_from_values(
    values: &[u32],
) -> CircleEvaluation<TraceBackend, BaseField, BitReversedOrder> {
    let log_size = values.len().ilog2();
    let mut col = Col::<TraceBackend, BaseField>::zeros(values.len());
    for (i, &v) in values.iter().enumerate() {
        col.set(i, BaseField::from_u32_unchecked(v));
    }
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}
