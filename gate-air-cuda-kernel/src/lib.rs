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
//! The CUDA kernel is BOX-UNVALIDATED (cannot compile without nvcc). Its algebraic core (19
//! constraints, qubit-memory + pc-pinned ts + rc-table encoding) is transcribed line-for-line from
//! `GateEval::evaluate`; the LogUp pair-batch part
//! is the Phase-2 soundness gate. The kernel only ever runs behind the explicit `CUDA_GPU_CONSTRAINTS=1`
//! opt-in, and falls back to the host delegate (returns `false`) if the drawn relation is not
//! installed via [`set_gate_air_relation`].

use std::ffi::c_void;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cuda::fused_commit;
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
        // COMPOSITION_TILING_SCOPE (route c) + FULL (A): PER-COLUMN host-tile-source tables for
        // tree0/tree1. A non-null table has one entry per column: the column's committed host bytes
        // if that column is STAGED (row-tiled, H2D per block), or NULL if that column is RESIDENT
        // (the kernel then uses its live device pointer whole — the resident sentinel). A wholly-
        // null TABLE means the whole tree is resident. The kernel row-tiles iff EITHER table is
        // non-null and dispatches each column independently (mixed supply). Both tables null =>
        // legacy resident path (BYTE-FOR-BYTE). MUST match the CUDA `evaluate_gate_air_entry` arg
        // list in stwo-cuda-backend gate_air_entry.cu — a mismatch is UB.
        host_trace0: *const *const u32,
        host_trace1: *const *const u32,
        // F2-b / Option B: per-column host-tile-source table for tree2 (interaction).
        // Non-null => the kernel row-tiles tree2 like tree0/1 and precomputes the 4
        // shifted last-LogUp cumsum coords (interaction_shift_neg1), so the composition
        // no longer holds all 28 interaction cols resident (~14 GiB @2^26). Null =>
        // tree2 resident + legacy scattered `-1` read (BYTE-FOR-BYTE). Entry c = the
        // staged host stash base pointer for tree2 col c, or NULL if that col is
        // resident. MUST match gate_air_entry.cu's arg list — a mismatch is UB.
        host_trace2: *const *const u32,
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
/// QUBIT-MEMORY + pc-pinned ts + single-`d` rc-table encoding (branch anatg/gate-air-qubit-mem): the
/// whole-state 188-col TAG_STATE encoding is replaced by the 19-col per-qubit chain-lookup qubit-memory
/// with the ts-ordering rc-table range-check (ACCESS_BLOCK = 4 = addr,prev_ts,v,d; ts=pc+1 inlined; the
/// two rc limbs collapsed to a single 25-bit diff column `d`).
/// These MUST match the actual component's shape — `is_gate_air_main` uses them to decide whether the
/// GPU kernel applies, so a stale value silently FALLS BACK to the host delegate (no fail-fast).
const GATE_AIR_TRACE_COLUMNS: usize = 19; // main cols (trace1): 4 opcode + 3*ACCESS_BLOCK(4) + 3 (ts=pc+1 and v_after=v_before+delta inlined)
const GATE_AIR_INTERACTION_COLUMNS: usize = 20; // 5 LogUp cols * 4 QM31 coords (trace2)
const GATE_AIR_PREPROCESSED_COLUMNS: usize = 4; // enabler, shot_id, pc, pc_in_prog (trace0), in get_preprocessed_column call order
const GATE_AIR_N_CONSTRAINTS: usize = 15 + 5; // 15 algebraic (19 -3 PIN -1 v_after eq) + 5 LogUp pair-batch constraints
const GATE_AIR_LOGUP_COUNTS: u32 = 10; // 10 relation entries -> 5 pairs = 5 batches
const GATE_AIR_REL_WIDTH: usize = 6; // relation!(GateRel, 6): tag + widest payload (program = 5)

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
/// LookupElementsBasic<6> relation; }` where `LookupElementsBasic<6>` is
/// `{ qm31 z; qm31 alpha; qm31 alpha_powers[6]; }`. All `qm31` are 4xu32.
#[repr(C)]
struct GateAirEvalFfi {
    eval_id: u32,
    log_n_rows: u32,
    z: [u32; 4],
    alpha: [u32; 4],
    alpha_powers: [[u32; 4]; GATE_AIR_REL_WIDTH],
}

impl GateAirEvalFfi {
    fn new_with_relation(
        log_n_rows: u32,
        z: [u32; 4],
        alpha_powers: [[u32; 4]; GATE_AIR_REL_WIDTH],
    ) -> Self {
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

// gate_air's drawn LogUp challenges (z, alpha_powers[0..6]), each an M31x4 (QM31 coord) value.
// Set by the leaf prover AFTER drawing the LogUp relation and BEFORE `prove_ex` — the challenges
// live inside the opaque `GateEval`, so this gate_air-specific hook threads them to the kernel.
// `None` until set → [`run_gate_air_kernel`] returns `false` (host-delegate fallback).
//
// THREAD_LOCAL for in-process multi-GPU base proving ("option A"). `set_gate_air_relation` is called
// PER SHARD, on that shard's producer thread, right before that shard's `prove_ex`; the value is
// then READ during the composition kernel launch on the SAME thread. A single process-global (the
// old `Mutex<Option<..>>`) is a genuine SET-then-READ RACE under concurrent producers: producer B's
// `set` (its own z/alpha, drawn from B's transcript) could overwrite the relation between producer
// A's `set` and A's kernel read → A commits the WRONG composition polynomial = a soundness bug (the
// same race class as `g_should_accumulate_host`, and squarely on the CUDA_GPU_CONSTRAINTS critical
// path). thread_local gives each producer thread its own relation; the set→read pair stays
// same-thread, so single-GPU behavior is byte-identical (one thread only ever sees its own value).
thread_local! {
    static GATE_AIR_RELATION: std::cell::Cell<Option<([u32; 4], [[u32; 4]; GATE_AIR_REL_WIDTH])>> =
        const { std::cell::Cell::new(None) };
}

/// Install gate_air's drawn LogUp relation challenges for the GPU constraint kernel, FOR THE CALLING
/// THREAD. `alpha_powers` must contain GATE_AIR_REL_WIDTH (6) entries (alpha^0..alpha^5), each as
/// M31x4 coords. No effect unless the kernel gate (`CUDA_GPU_CONSTRAINTS=1` + gate_air main) fires.
/// Must be called on the SAME thread that then drives `prove_ex` for this shard (it is — the
/// producer thread calls this then proves).
pub fn set_gate_air_relation(z: [u32; 4], alpha_powers: Vec<[u32; 4]>) {
    assert_eq!(
        alpha_powers.len(),
        GATE_AIR_REL_WIDTH,
        "gate_air relation needs {GATE_AIR_REL_WIDTH} alpha powers"
    );
    let mut ap = [[0u32; 4]; GATE_AIR_REL_WIDTH];
    ap.copy_from_slice(&alpha_powers[..GATE_AIR_REL_WIDTH]);
    GATE_AIR_RELATION.with(|r| r.set(Some((z, ap))));
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
    // DIAGNOSTIC (GATE_AIR_DIAG_FULL_REHYDRATE=1): bisect the A-side (composition) staged read.
    // When set, every STAGED tree0/tree1 column is `rehydrate_owned` into a fresh OWNED device
    // buffer here and fed to the kernel as RESIDENT (its fresh device ptr in traceX_ptrs; host
    // table entry NULL), so the kernel reads it WHOLE/live and NEVER takes the per-block
    // H2D-from-host-table (tile-buffer) path. Interpretation of a both-flags @2^22 run:
    //   * now PASSES  => the bug is in the composition kernel's per-block staged read mechanics
    //     (host-table / tile-buffer / bias), NOT the stashed bytes.
    //   * still FAILS => `rehydrate_owned` reads the SAME wrong bytes the per-block path would,
    //     so the bug is in the STASHED BYTES themselves (dehydrate captured wrong/stale data).
    // Scoped strictly to this composition dispatch; the keepalive Vec below owns the rehydrated
    // buffers for the FFI-call lifetime and frees them on return. Diagnostic only (not a perf path;
    // full rehydrate is cheap at 2^22).
    let diag_full_rehydrate = std::env::var("GATE_AIR_DIAG_FULL_REHYDRATE").is_ok();

    // Rehydrated owned buffers (diagnostic only). Held for the whole FFI-call lifetime; on
    // `diag_full_rehydrate` the traceX_ptrs entry for a staged column points into one of these.
    let mut diag_rehydrated0: Vec<BaseFieldVec> = Vec::new();
    let mut diag_rehydrated1: Vec<BaseFieldVec> = Vec::new();
    let mut diag_rehydrated2: Vec<BaseFieldVec> = Vec::new();

    let mut trace0_ptrs: Vec<*const u32> = inputs.trace0.iter().map(|c| c.device_ptr).collect();
    let mut trace1_ptrs: Vec<*const u32> = inputs.trace1.iter().map(|c| c.device_ptr).collect();
    // F2-b / Option B: tree2 is now staged-passthrough (like tree0/1) so its device
    // pointers may be stash-key sentinels; the kernel resolves the staged bytes via
    // the host_trace2 table below and row-tiles per block. Under the diagnostic
    // (GATE_AIR_DIAG_FULL_REHYDRATE) tree2 is rehydrated WHOLE to resident and the host
    // table nulled, so the kernel keeps tree2 resident + the scattered `-1` read.
    let mut trace2_ptrs: Vec<*const u32> = inputs.trace2.iter().map(|c| c.device_ptr).collect();

    if diag_full_rehydrate {
        for (c, col) in inputs.trace0.iter().enumerate() {
            if fused_commit::is_staged(col) {
                let owned = fused_commit::rehydrate_owned(col);
                trace0_ptrs[c] = owned.device_ptr;
                diag_rehydrated0.push(owned);
            }
        }
        for (c, col) in inputs.trace1.iter().enumerate() {
            if fused_commit::is_staged(col) {
                let owned = fused_commit::rehydrate_owned(col);
                trace1_ptrs[c] = owned.device_ptr;
                diag_rehydrated1.push(owned);
            }
        }
        for (c, col) in inputs.trace2.iter().enumerate() {
            if fused_commit::is_staged(col) {
                let owned = fused_commit::rehydrate_owned(col);
                trace2_ptrs[c] = owned.device_ptr;
                diag_rehydrated2.push(owned);
            }
        }
        eprintln!(
            "[GATE_AIR_DIAG] full-rehydrate ON: rehydrated {} tree0 + {} tree1 + {} tree2 staged \
             cols to resident for the composition kernel",
            diag_rehydrated0.len(),
            diag_rehydrated1.len(),
            diag_rehydrated2.len()
        );
    }

    // Upload denom_inv to device (small: 2^log_expand entries).
    let denom_inv_dev = BaseFieldVec::from_vec(inputs.denom_inv.to_vec());

    let q0 = accum_cols[0].device_ptr;
    let q1 = accum_cols[1].device_ptr;
    let q2 = accum_cols[2].device_ptr;
    let q3 = accum_cols[3].device_ptr;

    // Fall back to the host delegate (rather than emit a zeroed-relation proof) if the leaf prover
    // did not install the drawn (z, alpha_powers).
    // Read THIS thread's installed relation (thread_local — see the definition). `Cell::get` needs
    // `Copy`; the payload is `([u32;4], [[u32;4];6])` which is `Copy`, so this is a cheap copy-out.
    let mut gate_eval = match GATE_AIR_RELATION.with(|r| r.get()) {
        Some((z, ap)) => GateAirEvalFfi::new_with_relation(inputs.log_n_rows, z, ap),
        None => return false,
    };

    let number_of_columns =
        (inputs.trace0.len() + inputs.trace1.len() + inputs.trace2.len()) as u32;
    let random_coeff_powers_ptr = inputs.random_coeff_powers.as_ptr() as *const u32;

    // COMPOSITION_TILING_SCOPE (route c) + FULL (A) per-column mixed supply: under
    // GATE_AIR_STREAM_COMMIT only the 188 LARGE tree1 eval columns are host-staged (their device
    // buffers freed, bytes in the fused_commit stash); the 4 tree0 preprocessed columns and the
    // small tree1 (multiplicity/witness/program) columns stay RESIDENT on device.
    // `build_scoped_device_trace` leaves a staged column as a NON-OWNING passthrough carrying the
    // stash-key device pointer (so `staged_host_ptr` resolves here) and keeps a resident column as
    // its live device buffer. So tree1 is a MIX and tree0 is fully resident.
    //
    // Build PER-COLUMN host tables: entry c = the staged host stash base pointer if column c is
    // staged, else NULL (the kernel's resident sentinel — it then uses the column's live device
    // pointer, whole, exactly as the resident path). Pass a table pointer iff AT LEAST ONE column
    // of that tree is staged; a wholly-resident tree (tree0 here) passes a NULL table. The kernel's
    // `tiled_input` is true iff EITHER table is non-null, and it dispatches each column
    // independently — so a resident tree0 does NOT force the whole eval-set resident (which would
    // dereference the freed stash-key pointers of the staged tree1 cols — the 2^24 illegal address).
    // When NOTHING is staged (legacy/resident shard) both tables are null and the kernel takes the
    // byte-for-byte resident path. tree2 is always resident (scope §1.3) — no host supply.
    // Under the diagnostic, force EVERY host-table entry NULL: the staged columns were rehydrated
    // to resident above (their fresh device ptr is in traceX_ptrs), so the kernel must read them
    // whole/live, not via the per-block host table. `null_table` on both trees makes `tiled_input`
    // false in the kernel => byte-for-byte the resident path over the rehydrated buffers.
    let null_table = |c: &&BaseFieldVec| -> *const u32 {
        let _ = c;
        std::ptr::null()
    };
    let host0: Vec<*const u32> = inputs
        .trace0
        .iter()
        .map(|c| {
            if diag_full_rehydrate {
                null_table(&c)
            } else {
                fused_commit::staged_host_ptr(c).map_or(std::ptr::null(), |(p, _)| p)
            }
        })
        .collect();
    let host1: Vec<*const u32> = inputs
        .trace1
        .iter()
        .map(|c| {
            if diag_full_rehydrate {
                null_table(&c)
            } else {
                fused_commit::staged_host_ptr(c).map_or(std::ptr::null(), |(p, _)| p)
            }
        })
        .collect();
    // F2-b / Option B: per-column tree2 host table. Entry c = the staged host stash
    // base pointer if tree2 col c is staged, else NULL (resident sentinel). Under the
    // diagnostic, force NULL (tree2 was rehydrated whole to resident above) so the
    // kernel keeps tree2 resident + the scattered `-1` read (byte-for-byte). Passed as
    // a table iff AT LEAST ONE tree2 col is staged; a wholly-resident tree2 passes NULL
    // (kernel then keeps tree2 resident + legacy read — byte-for-byte).
    let host2: Vec<*const u32> = inputs
        .trace2
        .iter()
        .map(|c| {
            if diag_full_rehydrate {
                null_table(&c)
            } else {
                fused_commit::staged_host_ptr(c).map_or(std::ptr::null(), |(p, _)| p)
            }
        })
        .collect();

    // DIAGNOSTIC INSTRUMENTATION (printed regardless of the flag): how many tree0/tree1 columns
    // resolve STAGED vs RESIDENT at composition-kernel dispatch time. `is_staged` reflects the
    // owns_memory-guarded truth; under the diagnostic flag the host tables are nulled AFTER this
    // count, so this count is the pre-diagnostic (real) staged/resident split.
    {
        let staged0 = inputs.trace0.iter().filter(|c| fused_commit::is_staged(c)).count();
        let staged1 = inputs.trace1.iter().filter(|c| fused_commit::is_staged(c)).count();
        eprintln!(
            "[GATE_AIR_DIAG] composition dispatch: tree0 {}/{} staged, tree1 {}/{} staged \
             (diag_full_rehydrate={})",
            staged0,
            inputs.trace0.len(),
            staged1,
            inputs.trace1.len(),
            diag_full_rehydrate,
        );
        // Print + reset the dehydrate counters accumulated during the tree1 commit (which ran
        // before this composition dispatch). Shows how many large-col dehydrations early-returned
        // on a reused stash key.
        fused_commit::diag_report_dehydrate("tree1-commit -> composition");
    }
    // A tree with NO staged column passes a null table (fully resident); otherwise pass the
    // per-column table (staged entries carry the host ptr, resident entries are null).
    let any0_staged = host0.iter().any(|p| !p.is_null());
    let any1_staged = host1.iter().any(|p| !p.is_null());
    let any2_staged = host2.iter().any(|p| !p.is_null());
    // Keepalives own the tables for the FFI-call lifetime; take pointers AFTER binding so they
    // reference the surviving allocation (never a moved/dropped Vec).
    let host0_keepalive = any0_staged.then_some(host0);
    let host1_keepalive = any1_staged.then_some(host1);
    // F2-b / Option B: pass the tree2 host table iff ANY tree2 col is staged. A wholly-
    // resident tree2 passes NULL -> kernel keeps tree2 resident + scattered `-1` read
    // (byte-for-byte). Any staged tree2 col -> kernel row-tiles tree2 + shifts (the
    // resident source cols are handled via D2D inside the kernel's shift build).
    let host2_keepalive = any2_staged.then_some(host2);
    let host_trace0_ptr = host0_keepalive
        .as_ref()
        .map_or(std::ptr::null(), |h| h.as_ptr());
    let host_trace1_ptr = host1_keepalive
        .as_ref()
        .map_or(std::ptr::null(), |h| h.as_ptr());
    let host_trace2_ptr = host2_keepalive
        .as_ref()
        .map_or(std::ptr::null(), |h| h.as_ptr());

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
            host_trace0_ptr,
            host_trace1_ptr,
            host_trace2_ptr,
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
