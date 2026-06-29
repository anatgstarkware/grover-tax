#![cfg(feature = "cuda")]
//! gate_air-specific device-resident GPU constraint kernel + its registration hook.
//!
//! The generic stwo `CudaBackend` prover (stwo-cuda-backend) is circuit-agnostic: it takes the
//! audited host-delegate by default and offers each component to a registered GPU kernel only when
//! `CUDA_GPU_CONSTRAINTS=1`. This crate IS that kernel for the gate_air MAIN component. Call
//! [`register`] once before proving to install it; the generic prover then routes the structurally-
//! matching gate_air component to [`gate_air_gpu_kernel`], which runs the bespoke CUDA kernel
//! (`cuda/evaluate_gate_air.cu`) on the device-resident trace columns. Everything here is a SEPARATE
//! path to the audited CPU delegate — no host constraint-eval math is touched.
//!
//! # Soundness status (READ THIS)
//! The CUDA kernel is BOX-UNVALIDATED (cannot compile without nvcc). Its algebraic core (151
//! constraints) is transcribed line-for-line from `GateEval::evaluate`; the LogUp pair-batch part
//! is the Phase-2 soundness gate. The kernel only ever runs behind the explicit `CUDA_GPU_CONSTRAINTS=1`
//! opt-in, and falls back to the host delegate (returns `false`) if the drawn relation is not
//! installed via [`set_gate_air_relation`].

use std::ffi::c_void;
use std::sync::Mutex;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::bindings::CudaSecureField;

use stwo_constraint_framework::{
    set_gpu_constraint_kernel, ConstraintQuotientInputs, GpuConstraintDispatch,
    INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
};

// Standalone FFI entry for the gate_air kernel (cuda/entry.cu). Mirrors the generic
// `evaluate_constraint_quotients_on_domain` signature minus the (default) stream arg.
extern "C" {
    fn evaluate_gate_air_entry(
        quotients_0: *const u32,
        quotients_1: *const u32,
        quotients_2: *const u32,
        quotients_3: *const u32,
        trace0_evaluations: *const *const u32,
        trace0_evaluations_len: u32,
        trace1_evaluations: *const *const u32,
        trace1_evaluations_len: u32,
        trace2_evaluations: *const *const u32,
        trace2_evaluations_len: u32,
        random_coeff_powers: *const u32,
        denominator_inverses: *const u32,
        domain_log_size: u32,
        eval_domain_log_size: u32,
        number_of_columns: u32,
        logup_counts: u32,
        eval: *mut c_void,
        cumsum_shift: CudaSecureField,
        should_accumulate: bool,
        use_assert_evaluator: bool,
    ) -> bool;
}

/// gate_air structural constants (must equal the Rust constants in gate-air-leaf src/main.rs).
const GATE_AIR_TRACE_COLUMNS: usize = 191; // main columns (trace1)
const GATE_AIR_INTERACTION_COLUMNS: usize = 24; // 6 LogUp cols * 4 QM31 coords (trace2)
const GATE_AIR_PREPROCESSED_COLUMNS: usize = 1; // gate_pc_in_prog (trace0)
const GATE_AIR_N_CONSTRAINTS: usize = 151 + 6; // 151 algebraic + 6 LogUp pair-batch constraints
const GATE_AIR_LOGUP_COUNTS: u32 = 12; // 12 relation entries -> 6 pairs

/// `fnv1a("gate_air_main")` — kept for parity with the eval struct's first field (`CommonEval`).
const fn fnv1a(s: &[u8]) -> u32 {
    let mut hash: u32 = 0x811C9DC5;
    let prime: u32 = 0x0100_0193;
    let mut i = 0;
    while i < s.len() {
        hash ^= s[i] as u32;
        hash = hash.wrapping_mul(prime);
        i += 1;
    }
    hash
}
const GATE_AIR_EVAL_ID: u32 = fnv1a(b"gate_air_main");

/// Structural fingerprint test for the gate_air MAIN component. Used ONLY to decide whether the
/// opt-in GPU kernel applies; it never changes default behavior. The four small table components
/// have a single relation entry and far fewer constraints, so they do not match and stay on the
/// host delegate.
fn is_gate_air_main(
    n_constraints: usize,
    n_main_columns: usize,
    n_interaction_columns: usize,
    n_preprocessed_columns: usize,
) -> bool {
    n_constraints == GATE_AIR_N_CONSTRAINTS
        && n_main_columns == GATE_AIR_TRACE_COLUMNS
        && n_interaction_columns == GATE_AIR_INTERACTION_COLUMNS
        && n_preprocessed_columns == GATE_AIR_PREPROCESSED_COLUMNS
}

/// The `GateAirEval` struct passed through the FFI `void *eval` arg. Layout MUST match the CUDA
/// `struct GateAirEval` (evaluate_gate_air.cuh): `{ unsigned eval_id; unsigned log_n_rows;
/// LookupElementsBasic<35> relation; }` where `LookupElementsBasic<35>` is
/// `{ qm31 z; qm31 alpha; qm31 alpha_powers[35]; }`. All `qm31` are 4xu32.
#[repr(C)]
struct GateAirEvalFfi {
    eval_id: u32,
    log_n_rows: u32,
    z: [u32; 4],
    alpha: [u32; 4],
    alpha_powers: [[u32; 4]; 35],
}

impl GateAirEvalFfi {
    fn new_with_relation(log_n_rows: u32, z: [u32; 4], alpha_powers: [[u32; 4]; 35]) -> Self {
        Self {
            eval_id: GATE_AIR_EVAL_ID,
            log_n_rows,
            z,
            // `alpha` = alpha^1 = alpha_powers[1]; combine() reads `z` + `alpha_powers` but we
            // populate `alpha` faithfully for parity with LookupElementsBasic.
            alpha: alpha_powers[1],
            alpha_powers,
        }
    }
}

/// gate_air's drawn LogUp challenges (z, alpha_powers[0..35]), each an M31x4 (QM31 coord) value.
/// Set by the leaf prover AFTER drawing the LogUp relation and BEFORE `prove_ex` — the challenges
/// live inside the opaque `GateEval`, so this gate_air-specific hook threads them to the kernel.
/// `None` until set → [`run_gate_air_kernel`] returns `false` (host-delegate fallback).
static GATE_AIR_RELATION: Mutex<Option<([u32; 4], [[u32; 4]; 35])>> = Mutex::new(None);

/// Install gate_air's drawn LogUp relation challenges for the GPU constraint kernel. `alpha_powers`
/// must contain 35 entries (alpha^0..alpha^34), each as M31x4 coords. No effect unless the kernel
/// gate (`CUDA_GPU_CONSTRAINTS=1` + gate_air main) fires.
pub fn set_gate_air_relation(z: [u32; 4], alpha_powers: Vec<[u32; 4]>) {
    assert_eq!(alpha_powers.len(), 35, "gate_air relation needs 35 alpha powers");
    let mut ap = [[0u32; 4]; 35];
    ap.copy_from_slice(&alpha_powers[..35]);
    *GATE_AIR_RELATION.lock().unwrap() = Some((z, ap));
}

/// Inputs to the gate_air GPU constraint kernel; trace columns are device-resident `BaseFieldVec`s.
struct GateAirKernelInputs<'a> {
    trace0: Vec<&'a BaseFieldVec>,
    trace1: Vec<&'a BaseFieldVec>,
    trace2: Vec<&'a BaseFieldVec>,
    denom_inv: &'a [BaseField],
    /// Reversed slice (same as the host delegate's `accum.random_coeff_powers` after `.reverse()`).
    random_coeff_powers: &'a [SecureField],
    domain_log_size: u32,
    eval_domain_log_size: u32,
    log_n_rows: u32,
    cumsum_shift: SecureField,
    should_accumulate: bool,
    use_assert_evaluator: bool,
}

/// Call the gate_air GPU constraint kernel, writing into the 4 device accumulator coordinate
/// columns. Returns `false` (host-delegate fallback) if the drawn relation was not installed.
fn run_gate_air_kernel(inputs: GateAirKernelInputs<'_>, accum_cols: [&BaseFieldVec; 4]) -> bool {
    let trace0_ptrs: Vec<*const u32> = inputs.trace0.iter().map(|c| c.device_ptr).collect();
    let trace1_ptrs: Vec<*const u32> = inputs.trace1.iter().map(|c| c.device_ptr).collect();
    let trace2_ptrs: Vec<*const u32> = inputs.trace2.iter().map(|c| c.device_ptr).collect();

    // Upload denom_inv to device (small: 2^log_expand entries).
    let denom_inv_dev = BaseFieldVec::from_vec(inputs.denom_inv.to_vec());

    let q0 = accum_cols[0].device_ptr;
    let q1 = accum_cols[1].device_ptr;
    let q2 = accum_cols[2].device_ptr;
    let q3 = accum_cols[3].device_ptr;

    // Fall back to the host delegate (rather than emit a zeroed-relation proof) if the leaf prover
    // did not install the drawn (z, alpha_powers).
    let mut gate_eval = match GATE_AIR_RELATION.lock().unwrap().clone() {
        Some((z, ap)) => GateAirEvalFfi::new_with_relation(inputs.log_n_rows, z, ap),
        None => return false,
    };

    let number_of_columns =
        (inputs.trace0.len() + inputs.trace1.len() + inputs.trace2.len()) as u32;
    let random_coeff_powers_ptr = inputs.random_coeff_powers.as_ptr() as *const u32;

    unsafe {
        evaluate_gate_air_entry(
            q0,
            q1,
            q2,
            q3,
            trace0_ptrs.as_ptr(),
            trace0_ptrs.len() as u32,
            trace1_ptrs.as_ptr(),
            trace1_ptrs.len() as u32,
            trace2_ptrs.as_ptr(),
            trace2_ptrs.len() as u32,
            random_coeff_powers_ptr,
            denom_inv_dev.device_ptr,
            inputs.domain_log_size,
            inputs.eval_domain_log_size,
            number_of_columns,
            GATE_AIR_LOGUP_COUNTS,
            &mut gate_eval as *mut GateAirEvalFfi as *mut c_void,
            CudaSecureField::from(inputs.cumsum_shift),
            inputs.should_accumulate,
            inputs.use_assert_evaluator,
        )
    }
}

/// `cumsum_shift` for `finalize_logup_in_pairs`: `claimed_sum / n_rows` (the `LogupAtRow` shift).
fn gate_air_cumsum_shift(claimed_sum: SecureField, log_n_rows: u32) -> SecureField {
    let n_rows = BaseField::from_u32_unchecked(1u32 << log_n_rows);
    claimed_sum * n_rows.inverse()
}

/// The registered GPU constraint kernel. Returns `true` if it handled the gate_air MAIN component
/// (device accumulator written), `false` otherwise (the generic prover then falls back to the
/// audited host delegate). Reads trace columns DEVICE-RESIDENT (no D2H).
fn gate_air_gpu_kernel(d: GpuConstraintDispatch<'_, '_>) -> bool {
    let ConstraintQuotientInputs {
        eval_domain,
        trace_domain,
        trace: trace_cols,
        denom_inv,
    } = d.inputs;

    let n_preprocessed = trace_cols[PREPROCESSED_TRACE_IDX].len();
    let n_main = trace_cols[ORIGINAL_TRACE_IDX].len();
    let n_interaction = trace_cols[INTERACTION_TRACE_IDX].len();
    if !is_gate_air_main(d.n_constraints, n_main, n_interaction, n_preprocessed) {
        return false;
    }

    // Accumulator column + its slice of random_coeff_powers (SAME split + `.reverse()` as the host
    // delegate, guaranteeing identical coefficients / ordering).
    let [mut accum] = d
        .accumulator
        .columns([(eval_domain.log_size(), d.n_constraints)]);
    accum.random_coeff_powers.reverse();

    // Device column pointer tables (borrow the device `BaseFieldVec`s; no host copy).
    let trace0: Vec<&BaseFieldVec> = trace_cols[PREPROCESSED_TRACE_IDX]
        .iter()
        .map(|c| &c.as_ref().values)
        .collect();
    let trace1: Vec<&BaseFieldVec> = trace_cols[ORIGINAL_TRACE_IDX]
        .iter()
        .map(|c| &c.as_ref().values)
        .collect();
    let trace2: Vec<&BaseFieldVec> = trace_cols[INTERACTION_TRACE_IDX]
        .iter()
        .map(|c| &c.as_ref().values)
        .collect();

    let log_n_rows = d.log_n_rows;
    let cumsum_shift = gate_air_cumsum_shift(d.claimed_sum, log_n_rows);

    let accum_cols: [&BaseFieldVec; 4] = [
        &accum.col.columns[0],
        &accum.col.columns[1],
        &accum.col.columns[2],
        &accum.col.columns[3],
    ];

    let inputs = GateAirKernelInputs {
        trace0,
        trace1,
        trace2,
        denom_inv: denom_inv.as_slice(),
        random_coeff_powers: &accum.random_coeff_powers,
        domain_log_size: trace_domain.log_size(),
        eval_domain_log_size: eval_domain.log_size(),
        log_n_rows,
        cumsum_shift,
        // Seed-and-accumulate, matching the host delegate semantics.
        should_accumulate: true,
        use_assert_evaluator: false,
    };

    run_gate_air_kernel(inputs, accum_cols)
}

/// Install the gate_air GPU constraint kernel into the generic stwo `CudaBackend` prover. Call once
/// before proving. No-op effect unless `CUDA_GPU_CONSTRAINTS=1` and the gate_air component matches.
pub fn register() {
    set_gpu_constraint_kernel(gate_air_gpu_kernel);
}
