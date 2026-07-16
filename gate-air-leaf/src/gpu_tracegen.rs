//! P3.1 — K1: CUDA gate-sim + main-trace kernel for gate_air (on-device "model B").
//!
//! ============================================================================================
//! RE-SYNCED to the CURRENT (sound, final) CPU design: ts = pc+1 (inlined) + single-`d` rc-table
//! range-check, 19-col.
//! --------------------------------------------------------------------------------------------
//! This file mirrors the FINAL CPU encoding (`main.rs` `GateEval` / `cell_at` / `gen_main_interaction`
//! / `build_rc_table`): the access timestamp is the affine `ts = pc + 1` of the preprocessed `pc`
//! (NOT a witness column — inlined in K4 and the AIR), plus a range-check on the SINGLE diff
//! `d = ts - prev_ts - 1 = pc - prev_ts` looked up into a dynamic rc supply table. The target's
//! `v_after` is likewise NOT a column (= v_before + delta, inlined). This dropped the 3 per-access
//! `ts` columns + the target `v_after` column, and (this change) collapsed the two rc limbs to a
//! single `d` column: 22 -> 19.
//!   * ts closed form (thread-per-execution SURVIVES): `ts = pc + 1`, `pc = rep*n_gates + gate_idx` —
//!     known per execution, shared by all accesses of the step (no per-gate slot).
//!     `prev_ts` is the ts of the previous access to this addr (0 = init), so it is NO LONGER
//!     `ts-1` in general; K1 reconstructs it from `prog_slot_meta`'s cyclic-predecessor constants
//!     (predecessor pc + 1; see `prev_ts_of`).
//!   * rc DIFF: a SINGLE range-check column per access — `d = pc - prev_ts` (25-bit, no limb split).
//!     Layout is now ACCESS_BLOCK = 4 (addr,prev_ts,v,d), TRACE_COLUMNS = 19 (see `cell_at`/
//!     `ACCESS_BLOCK` in main.rs).
//!   * rc HISTOGRAM: a single `2^rc_log` multiplicity histogram over `d` (rc_log = rc_log_size(
//!     k*n_gates), DYNAMIC per run). Each ACTIVE access bumps `hist[d] += 1` (one bump/access).
//!     Emitted into the (formerly unused) `rc_lo` device arg (`rc_hi` arg stays inert). NOTE: on the
//!     PRODUCTION path the rc multiplicity WITNESS column is built ON THE HOST from the always-present
//!     CPU `rows` (`build_rc_table` in main.rs), independent of this kernel — so the GPU histogram is
//!     used ONLY by the `k1_byte_identity` diagnostic to cross-check the device histogram against CPU.
//!   * boundary: nothing on the MAIN kernel (the `gate_bnd_enabler`-gated (B)/(D) is host-only).
//!
//! QUBIT-MEMORY ENCODING (branch anatg/gate-air-qubit-mem).
//! -------------------------------------------------------
//! Generates the main-trace columns ENTIRELY on the GPU (trace never leaves device memory). The old
//! whole-state (188/191-col TAG_STATE) encoding + its qdecode histograms are replaced by the
//! per-qubit chain-lookup qubit-memory (TAG_QUBITMEM) + the ts-ordering rc-table lookup (TAG_RC),
//! byte-identical to the CPU `simulate_shot` / `cell_at` / `build_rc_table`.
//!
//! THREAD-PER-EXECUTION (survives via a closed-form ts).
//! -----------------------------------------------------
//! `ts` is a CLOSED FORM in the pc: for an access to addr `a` in rep `r` at gate `g`, the pc is
//! `pc = r*n_gates + g` and `ts = pc*TS_STRIDE + slot` (slot from the access role). This depends only
//! on (r, g, slot) — NOT on any running per-address counter — so the thread-per-EXECUTION split holds:
//!   * `prog_slot_meta` — one-thread prepass: per gate-slot the program constant `prev_gate` = the
//!     gate index of the PREVIOUS access to this addr within one pass (or a sentinel if none), so K1
//!     can compute `prev_ts` (the predecessor's closed-form ts) without a serial history. See the
//!     kernel doc for the exact prev_ts recovery (intra-rep predecessor vs. cross-rep / init).
//!   * K0 `gate_sim_states` — thread-per-SHOT, VALUE-ONLY: snapshots the 512-qubit state (32 limbs)
//!     at every (shot, rep) boundary (linear-in-k value chain). Seeds K1's per-execution value chain.
//!   * K1 `gate_sim` — thread-per-EXECUTION: thread `(shot, rep)` loads its rep-boundary state, walks
//!     the rep's n_gates gates for v_before/v_after, and fills ts (closed form) + prev_ts + the two
//!     rc limbs, and bumps the rc histogram.
//! Parallelism is n_shots·k. Row(shot,rep,g) = (shot*k+rep)*n_gates+g is the same contiguous per-shot
//! block; only how ts/prev_ts/rc are produced changed.
//!
//! SOUNDNESS / BYTE-IDENTITY: K1's per-gate body mirrors `simulate_shot` (the ctrl_a/ctrl_b/target
//! access order, the value gate-apply, delta_to_m31), the `cell_at` 19-column layout, the closed-form
//! `ts = pc + 1`, `prev_ts` (per-address chain), and the single rc diff `d` + histogram.
//! Validated by GATE_AIR_GPU_TEST=k1 / k4 column-by-column (+ histogram) vs CPU.

use std::sync::{Arc, OnceLock};

// MULTI-GPU ("option A"): the base GPU ORDINAL this host thread proves on. A producer thread proving
// shard set S on GPU n calls `set_base_gpu(n)` once at its start; every `cuda_device()` /
// module-cache access below then keys off THIS thread's ordinal, so the harness tracegen lands on
// the SAME device (n) as the backend commit for that shard. DEFAULT 0 for every un-set thread, so
// the single-GPU path (one producer, ordinal 0) is byte-identical to before (device 0 throughout).
thread_local! {
    static BASE_GPU_ORDINAL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Set the current host thread's base GPU ordinal (multi-GPU producer setup). Idempotent per thread.
/// Records the ordinal for the harness tracegen (`cuda_device`) AND, under the device-resident
/// backend, binds the backend's CUDA runtime current-device for this thread so its pool/commit calls
/// target the same device. Call ONCE at the top of each producer thread, before any GPU work.
#[cfg(feature = "gpu-cuda")]
pub(crate) fn set_base_gpu(ordinal: usize) {
    BASE_GPU_ORDINAL.with(|c| c.set(ordinal));
    // Bind the backend (stwo_cuda) runtime current-device on THIS thread. cudarc's cuda_device()
    // also binds the primary context (== runtime device) on first use, but the backend may issue a
    // runtime-API call before that; setting it explicitly here removes the ordering dependency.
    // Only under the device-resident backend (feature = "cuda"); no-op on the K1/K4-only build.
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

/// Build a PRIVATE rayon `ThreadPool` whose every worker is bound to CUDA device `gpu` (multi-GPU
/// SIGSEGV fix, Class-1). The producer thread proves its shard INSIDE this pool via `pool.install(..)`,
/// so the OODS-phase rayon fan-outs (`build_weights_hash_map`'s `par_iter`, OODS `par_map_cols` in
/// `pcs/mod.rs`) dispatch to THESE workers instead of rayon's GLOBAL pool. The global pool's workers
/// were never device-bound (default device 0), so on device N != 0 they dereferenced device-N pointers
/// while current-device-0 => illegal address => SIGSEGV. Binding each worker via `set_base_gpu(gpu)`
/// (which sets BOTH the driver `cudaSetDevice` AND the cudarc thread_local ordinal, exactly like the
/// producer) makes the fan-out run on device N, matching the pointers.
///
/// `num_threads` (mechanical choice): the fan-out is light CPU glue that only LAUNCHES kernels — the
/// heavy compute is on-GPU and serializes on device N's stream regardless — so a small pool suffices
/// and avoids over-subscribing cores across the G concurrent producers. Default 4, overridable via
/// `GATE_AIR_OODS_POOL_THREADS` for box tuning. (Correctness is independent of the count; it only
/// affects fan-out parallelism.)
///
/// BYTE-IDENTITY: a private pool changes only WHICH threads run the fan-out and WHICH device they are
/// bound to — never the work, the order of commits, or any Fiat-Shamir draw. rayon's `par_iter`/
/// `par_map_cols` are already order-independent reductions/maps; running them on a 4-thread private
/// pool vs. the global pool yields identical results. At N=1 (device 0) the workers bind device 0,
/// identical to today's global-pool-on-device-0 behavior.
#[cfg(feature = "gpu-cuda")]
pub(crate) fn build_device_bound_pool(gpu: usize) -> rayon::ThreadPool {
    let num_threads = std::env::var("GATE_AIR_OODS_POOL_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(move |i| format!("gate-air-oods-gpu{gpu}-{i}"))
        .start_handler(move |_| set_base_gpu(gpu))
        .build()
        .unwrap_or_else(|e| panic!("failed to build device-bound OODS pool for gpu {gpu}: {e}"))
}

/// Shared cudarc handle on the CURRENT THREAD's target device primary CUDA context, cached
/// process-wide PER ORDINAL. The NitrooZK `CudaBackend` (stwo_cuda) targets the same device's
/// primary context via the CUDA runtime API, and cudarc `CudaDevice::new(n)` retains + binds that
/// SAME primary context (`ctx::set_current`), so once a thread has used this handle its runtime
/// current-device is also `n` — device pointers produced by these cudarc K1/K4 kernels interoperate
/// with the backend's commit on the same device. For the single-GPU path the ordinal is 0 and this
/// is byte-identical to the previous single-`OnceLock` behavior. Replaces the obelyzk
/// `get_cuda_executor`. [box-verified for n=0]
pub(crate) fn cuda_device() -> Result<Arc<cudarc::driver::CudaDevice>, String> {
    // One cached device handle per ordinal. `MAX_BASE_GPUS` slots is plenty (GPUs 0..7 on the box).
    const MAX_BASE_GPUS: usize = 16;
    static DEVS: [OnceLock<Arc<cudarc::driver::CudaDevice>>; MAX_BASE_GPUS] =
        [const { OnceLock::new() }; MAX_BASE_GPUS];
    let ord = base_gpu_ordinal();
    let slot = DEVS
        .get(ord)
        .ok_or_else(|| format!("base gpu ordinal {ord} >= {MAX_BASE_GPUS}"))?;
    if let Some(d) = slot.get() {
        // Ensure THIS thread has the device's primary context current (cheap; needed when the same
        // cached handle is first touched from a new thread — see cudarc bind_to_thread contract).
        d.bind_to_thread()
            .map_err(|e| format!("bind_to_thread(dev {ord}): {e}"))?;
        return Ok(d.clone());
    }
    let d =
        cudarc::driver::CudaDevice::new(ord).map_err(|e| format!("CudaDevice::new({ord}): {e}"))?;
    let _ = slot.set(d.clone());
    Ok(d)
}

/// N4 — process-level PTX MODULE CACHE. `compile_ptx` (NVRTC) + `load_ptx` are expensive and were
/// previously run on EVERY `gpu_gen_main_trace*` / `gpu_gen_interaction*` call (once per shard). Each
/// kernel module should compile + load ONCE per process. cudarc registers a loaded module on the
/// device under its name (`get_func` then retrieves functions cheaply), so we guard the compile+load
/// with a `OnceLock` (like `cuda_device()`); after the first call only `get_func` runs.
///
/// Returns the module name + a `bool` (true on the first load) for callers that want to log.
#[cfg(feature = "gpu-cuda")]
fn gate_sim_module(dev: &Arc<cudarc::driver::CudaDevice>) -> Result<(&'static str, bool), String> {
    use cudarc::nvrtc::compile_ptx;
    // PER-ORDINAL load guard: cudarc registers a loaded module on the SPECIFIC CudaDevice (its
    // CUcontext), so each device must load the PTX once. A single shared guard would load only on
    // the first device and leave the others' `get_func` unresolved. Keyed by the caller's ordinal
    // (== dev.ordinal()); slot [0] for the single-GPU path => same one-load behavior as before.
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
        let ptx = compile_ptx(GATE_SIM_KERNEL).map_err(|e| format!("nvrtc compile: {e}"))?;
        // Qubit-memory encoding, thread-per-EXECUTION (recovered): `prog_slot_meta` precomputes the
        // closed-form ts constants, K0 `gate_sim_states` snapshots per-rep-boundary values, K1
        // `gate_sim` fills each (shot, rep)'s rows independently.
        dev.load_ptx(
            ptx,
            "gate_sim_mod",
            &[
                "prog_slot_meta",
                "gate_sim_states",
                "gate_sim",
                "fill_padding",
            ],
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
    use cudarc::nvrtc::compile_ptx;
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
            compile_ptx(INTERACTION_KERNEL).map_err(|e| format!("nvrtc compile (K4): {e}"))?;
        dev.load_ptx(ptx, "logup_mod", &names)
            .map_err(|e| format!("load_ptx (K4): {e}"))?;
        Ok(())
    });
    res.clone().map(|()| ("logup_mod", first))
}

/// Allocate the thread-per-execution scratch buffers shared by every K1 launch wrapper:
/// - `d_rep`: n_shots*k*N_LIMBS rep-boundary states (written by K0, read by K1).
/// - `d_slot`: n_gates*9 predecessor constants ([a_pg,a_ps,a_wrap, b_pg,b_ps,b_wrap, t_pg,t_ps,t_wrap]
///   per gate) — for each active access slot, the (gate, slot) of the previous access to the SAME addr
///   in cyclic program order + a `wrap` flag (1 = predecessor is in the previous rep, 0 = same rep).
///   K1 turns these into `prev_ts` (see the `gate_sim` doc). Filled ONCE here by `prog_slot_meta`.
/// `prog_slot_meta` is a single-thread program pass (n_gates is tiny), launched here so the constants
/// are ready before K0/K1. Returns (`d_rep`, `d_slot`).
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

/// NVRTC-compiled CUDA source for the gate-sim kernels. Layout/constants mirror gate_air `main.rs`:
/// N_QUBITS=512, N_LIMBS=32, LIMB_BITS=16, TRACE_COLUMNS=19, M31 modulus 2^31-1,
/// opcodes NOP=0/NOT=1/CNOT=2/TOFFOLI=3, ts = pc+1 (no slot); the ts-ordering diff is the SINGLE
/// column `d = pc - prev_ts` (rc histogram: one bump per active access at row `d ∈ [0,2^rc_log)`).
///
/// Buffers (all device):
/// - `gates`:   n_gates * 4  (opcode, target_q, ctrl_a_q, ctrl_b_q), u32
/// - `x_states`: n_shots * N_LIMBS  (initial state limbs per shot), u32
/// - `rep_states`: n_shots * k * N_LIMBS — rep-boundary states K0 produces and K1 consumes:
///   rep_states[(shot*k + rep)*N_LIMBS + i] = limb i of the state at the START of (shot, rep).
/// - `slot_meta`: n_gates * 9 — per-gate predecessor constants (see the `prog_slot_meta` prepass):
///   [a_pg,a_ps,a_wrap, b_pg,b_ps,b_wrap, t_pg,t_ps,t_wrap] for ctrl_a / ctrl_b / target. For each
///   active slot, (pg, ps) = (gate index, slot) of the PREVIOUS access to the same addr in cyclic
///   program order, and wrap = 1 iff that predecessor is in the PREVIOUS rep (0 iff the same rep). K1
///   turns them into `prev_ts` (the predecessor's closed-form ts, or 0 at the program-wide first access).
/// - `cols`:    TRACE_COLUMNS * padded_rows, column-major (col c at cols[c*padded_rows + row]), u32
/// - `rc_lo` (repurposed): the 2^rc_log rc-table MULTIPLICITY HISTOGRAM over `d` (K1 atomically bumps
///   hist[d] once per active access). Caller must zero it and size it to at least `1 << rc_log`
///   (rc_log = rc_log_size(k*n_gates)). `rc_hi` (repurposed) is INERT (kept for arg-list
///   compatibility). NB: production uses the HOST-built rc multiplicity witness (main.rs
///   build_rc_table); this histogram feeds only the k1 diagnostic.
/// Scalars: k, n_gates, n_shots, padded_rows (shot_rows = k*n_gates computed in-kernel).
/// NOTE: caller must zero `cols` first; padding rows [n_shots*shot_rows, padded_rows) stay 0
/// (`Row::padding()` is all-zero — `fill_padding` is a no-op stub).
/// Launch order: `prog_slot_meta` (fills slot_meta, once) → K0 `gate_sim_states` (fills rep_states)
/// → K1 `gate_sim` (consumes both).
pub const GATE_SIM_KERNEL: &str = r#"
#define N_QUBITS 512u
#define N_LIMBS 32u
#define LIMB_BITS 16u
#define M31_MOD 2147483647u
#define OP_NOP 0u
#define OP_NOT 1u
#define OP_CNOT 2u
#define OP_TOFFOLI 3u
#define TS_STRIDE 3u
#define SLOT_CTRL_A 1u
#define SLOT_CTRL_B 2u
#define SLOT_TARGET 3u

// QUBIT-MEMORY ENCODING (branch anatg/gate-air-qubit-mem) — THREAD-PER-EXECUTION, pc-pinned ts.
// -------------------------------------------------------------------------------------------
// The old whole-state (TAG_STATE, 188/191-col) chain is replaced by a per-qubit chain-lookup
// qubit-memory (TAG_QUBITMEM, 19-col). Each row is ONE gate execution. Per row the CPU
// `simulate_shot` (main.rs) maintains, PER SHOT (reset each shot, threaded across ALL k reps):
//   last_ts[addr], last_val[addr]  — the (ts, value) of the most recent access at each qubit
// An access reads (prev_ts = last_ts[addr], v_before = last_val[addr]), sets the PROGRAM-ORDER
// timestamp ts = pc + 1 (pc = rep*n_gates + gate; SHARED by all accesses of a step, no slot), then
// writes last_ts[addr] = ts (and last_val[addr] = v_after for the target, unchanged for controls). The
// diff d = ts - prev_ts - 1 = pc - prev_ts (>= 0) is a SINGLE 25-bit column (no limb split).
//
// TIMESTAMP-ORDERING (pc-pinned ts + single-`d` rc-table range-check, the FINAL sound design):
//   * ts is PINNED to the preprocessed pc: ts == pc + 1 (AIR constraint). Closed form in (rep, gate) —
//     no running counter.
//   * prev_ts < ts is proved by range-checking the single diff `d = ts - prev_ts - 1` looked up as
//     (TAG_RC, d) into the DYNAMIC rc supply table (d ∈ [0, 2^rc_log_size(k*n_gates))).
//
// CLOSED-FORM ts + prev_ts (thread-per-EXECUTION survives): ts is closed form in the pc. prev_ts is
// the closed-form ts of the PREVIOUS access to the same addr in cyclic program order. `prog_slot_meta`
// precomputes, per active slot, (prev_gate, prev_slot, wrap): the (gate, slot) of that predecessor in
// one pass and wrap=1 iff it lies in the PREVIOUS rep. K1 then sets
//     prev_rep = wrap ? rep-1 : rep;  prev_ts = (wrap && rep==0) ? 0 : (prev_rep*n_gates+prev_gate) + 1
// (prev_ts = 0 is the init boundary node — the program-wide first access to that addr). This needs
// only (rep, program constants), NO serial per-address history, so K1 stays embarrassingly parallel.
//
// K0 / K1 SPLIT (thread-per-EXECUTION):
//   K0 `gate_sim_states` — thread-per-SHOT, VALUE-ONLY. Walks the full k*n_gates chain doing only the
//     cheap 32-limb bit-flip (apply_gate, no column I/O) and snapshots the 512-qubit state (as 32
//     limbs) at the START of every (shot, rep) into rep_states. Seeds K1's per-execution value chain.
//   K1 `gate_sim` — thread-per-EXECUTION. Thread `exec = shot*k + rep` seeds a local 512-state from
//     rep_states[exec], processes the rep's n_gates gates updating that local state for v_before/
//     v_after, fills ts (closed form) + prev_ts + the single diff `d` from `slot_meta`+`rep`, and
//     atomically bumps the rc histogram (one bump per active access, over `d`). Grid = n_shots*k.
//
// 19-col cell_at layout (main.rs cell_at, all WITNESS; enabler/shot_id/pc/pc_in_prog are tree0;
// ts = pc+1 and the target's v_after = v_before+delta are INLINED, not columns):
//   [0..4)   is_nop, is_not, is_cnot, is_toffoli
//   [4..8)   target:  addr, prev_ts, v_before, d   (ACCESS_BLOCK = 4)
//   [8..12)  ctrl_a:  addr, prev_ts, v, d
//   [12..16) ctrl_b:  addr, prev_ts, v, d
//   [16..19) ab, fire, delta
//
// Boundary init/final is a SEPARATE CPU-built component (BoundaryTable); the GPU main-trace kernel
// does not emit it.

// prev_ts for one access from its (prev_gate, prev_slot, wrap) predecessor triple + this rep.
// wrap=1 => the predecessor is in the PREVIOUS rep; at rep 0 that predecessor doesn't exist, so the
// init boundary node prev_ts = 0 (matches simulate_shot's last_ts init of 0). Else prev_ts is the
// predecessor's ts = prev_pc + 1 = (prev_rep*n_gates + prev_gate) + 1 (ts = pc+1, no slot; prev_slot
// is unused now — the prepass still records it but the shared ts per step makes it irrelevant, since
// a same-address predecessor is always in a DIFFERENT gate step / pc).
__device__ __forceinline__ unsigned prev_ts_of(
    unsigned prev_gate, unsigned prev_slot, unsigned wrap, unsigned rep, unsigned n_gates)
{
    (void)prev_slot;
    if (wrap && rep == 0u) return 0u;
    unsigned prev_rep = wrap ? (rep - 1u) : rep;
    return (prev_rep * n_gates + prev_gate) + 1u;
}

// Apply one gate's target-bit flip to `limbs` in place. Pure state update (no column I/O) — the part
// the K1 per-gate body and K0 share. Mirrors simulate_shot's value math exactly (fire in {0,1}).
__device__ __forceinline__ void apply_gate(
    unsigned* limbs, const unsigned* __restrict__ gates, unsigned g)
{
    unsigned opcode = gates[g * 4u + 0u];
    unsigned tq     = gates[g * 4u + 1u];
    unsigned aq     = gates[g * 4u + 2u];
    unsigned bq     = gates[g * 4u + 3u];
    unsigned is_not = (opcode == OP_NOT) ? 1u : 0u;
    unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
    unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
    unsigned a_active = is_cnot + is_tof;
    unsigned b_active = is_tof;
    unsigned tl = tq / LIMB_BITS, tbp = tq % LIMB_BITS;
    unsigned t_bit = (limbs[tl] >> tbp) & 1u;
    unsigned a_bit = a_active ? ((limbs[aq / LIMB_BITS] >> (aq % LIMB_BITS)) & 1u) : 0u;
    unsigned b_bit = b_active ? ((limbs[bq / LIMB_BITS] >> (bq % LIMB_BITS)) & 1u) : 0u;
    unsigned fire = is_not + is_cnot * a_bit + is_tof * (a_bit * b_bit);  // in {0,1}
    unsigned new_t = t_bit ^ fire;
    int delta_signed = (int)new_t - (int)t_bit;                          // {-1,0,1}
    unsigned mask = 1u << tbp;
    if (delta_signed > 0) limbs[tl] += mask;
    else if (delta_signed < 0) limbs[tl] -= mask;
}

// PREPASS: per-active-slot cyclic PREDECESSOR constants (gate, slot, wrap) — the program constants K1
// needs to reconstruct `prev_ts` (the predecessor's closed-form ts) with NO serial per-address chain.
// ONE thread walks the program ONCE in ctrl_a→ctrl_b→target order (matching simulate_shot's do_access
// sequencing), maintaining per address the (gate, slot) of its MOST RECENT access seen so far in this
// pass (`lg[a]`, `ls[a]`, valid iff `seen[a]`). For each active slot it records:
//   pg, ps = the (gate, slot) of the previous access to that addr; wrap = 0 (predecessor is EARLIER in
//   THIS pass, so same rep). If the addr has NOT been seen yet this pass (first access this pass), the
//   predecessor is the LAST access to that addr in the pass — from the PREVIOUS rep — so we leave a
//   marker (wrap=1) and backfill (pg, ps) after the full pass from the finished lg/ls (= the last
//   access to the addr in one pass). At the program-wide first access (rep 0), K1 maps wrap&&rep==0 to
//   prev_ts = 0 (the init boundary node), matching simulate_shot's `last_ts` init of 0.
// slot_meta layout per gate g: [a_pg,a_ps,a_wrap, b_pg,b_ps,b_wrap, t_pg,t_ps,t_wrap] (9 u32).
// Inactive controls leave their triple = 0 (unused: K1 zeroes an inactive ctrl AccessCols). Single-
// thread is fine — n_gates is tiny (~2547 for iadd) and this runs ONCE per process-shard, dwarfed by K1.
extern "C" __global__ void prog_slot_meta(
    const unsigned* __restrict__ gates,
    unsigned n_gates,
    unsigned* __restrict__ slot_meta)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    unsigned lg[N_QUBITS];    // gate index of the addr's last access seen this pass
    unsigned ls[N_QUBITS];    // its slot
    unsigned char seen[N_QUBITS];
    for (unsigned a = 0; a < N_QUBITS; a++) { lg[a] = 0u; ls[a] = 0u; seen[a] = 0u; }
    // A record: for each active slot whose predecessor is a FIRST-in-pass wrap, remember where to
    // backfill (slot_meta index for pg) and which addr, so the second pass can fill from lg/ls.
    for (unsigned g = 0; g < n_gates; g++) {
        unsigned opcode = gates[g * 4u + 0u];
        unsigned tq     = gates[g * 4u + 1u];
        unsigned aq     = gates[g * 4u + 2u];
        unsigned bq     = gates[g * 4u + 3u];
        unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
        unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
        unsigned a_active = is_cnot + is_tof;
        unsigned b_active = is_tof;
        unsigned base = g * 9u;
        // Emit predecessor for one active access to addr q at slot `slot`; then record (g, slot) as the
        // addr's latest access. `off` is the slot's base offset in slot_meta (0/3/6 for a/b/t).
        #define PRED(q, slot, off) do {                                             \
            if (seen[(q)]) {                                                         \
                slot_meta[base + (off) + 0u] = lg[(q)];                             \
                slot_meta[base + (off) + 1u] = ls[(q)];                             \
                slot_meta[base + (off) + 2u] = 0u;   /* same rep */                 \
            } else {                                                                \
                /* first access to q this pass: predecessor is the pass's LAST     \
                   access to q, from the previous rep. Marked wrap=1; pg/ps        \
                   backfilled below. Stash the addr in the pg slot temporarily. */ \
                slot_meta[base + (off) + 0u] = (q);   /* temp: addr, backfilled */  \
                slot_meta[base + (off) + 1u] = 0u;                                  \
                slot_meta[base + (off) + 2u] = 1u;   /* wrap */                     \
            }                                                                       \
            lg[(q)] = g; ls[(q)] = (slot); seen[(q)] = 1u;                          \
        } while (0)
        if (a_active) PRED(aq, SLOT_CTRL_A, 0u);
        if (b_active) PRED(bq, SLOT_CTRL_B, 3u);
        PRED(tq, SLOT_TARGET, 6u);   // target always active
        #undef PRED
    }
    // Backfill every wrap=1 predecessor with the addr's LAST access in the pass (lg/ls now final).
    for (unsigned g = 0; g < n_gates; g++) {
        unsigned base = g * 9u;
        #pragma unroll
        for (unsigned off = 0u; off < 9u; off += 3u) {
            if (slot_meta[base + off + 2u] == 1u) {
                unsigned q = slot_meta[base + off + 0u];  // temp addr stashed above
                slot_meta[base + off + 0u] = lg[q];
                slot_meta[base + off + 1u] = ls[q];
            }
        }
    }
}

// K0: thread-per-SHOT, VALUE-ONLY. Walk the full k*n_gates chain (cheap apply_gate limb updates, NO
// column stores) and snapshot the 512-qubit state (32 limbs) at the START of every (shot, rep) into
// rep_states. rep 0's snapshot is exactly x_states[shot]; rep r's is the state after r full passes.
// Keeps total value-chain work LINEAR in k while letting K1 start each (shot, rep) independently.
extern "C" __global__ void gate_sim_states(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ x_states,
    unsigned* __restrict__ rep_states,
    unsigned k,
    unsigned n_gates,
    unsigned n_shots)
{
    unsigned shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= n_shots) return;

    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = x_states[shot * N_LIMBS + i];

    for (unsigned rep = 0; rep < k; rep++) {
        unsigned long base = ((unsigned long)shot * (unsigned long)k + (unsigned long)rep) * N_LIMBS;
        #pragma unroll
        for (unsigned i = 0; i < N_LIMBS; i++) rep_states[base + i] = limbs[i];
        for (unsigned g = 0; g < n_gates; g++) apply_gate(limbs, gates, g);
    }
}

// K1: THREAD-PER-EXECUTION. One thread per (shot, rep). exec = shot*k + rep. Seeds a local 512-state
// from K0's rep-boundary snapshot (so this execution starts exactly where simulate_shot was entering
// rep `rep`), processes the rep's n_gates gates for v_before/v_after, and fills ts (closed form
// pc*TS_STRIDE+slot), prev_ts (from slot_meta's cyclic-predecessor constants), and the two rc limbs.
// `off_lo` is repurposed to carry rep_states and `off_hi` to carry slot_meta (both formerly-unused arg
// slots — keeps the launch tuple at 8 pointers + 4 scalars, cudarc's cap). `qdecode` stays UNUSED;
// `rc_lo` is repurposed as the rc-table MULTIPLICITY HISTOGRAM over `d` (atomic bumps); `rc_hi` INERT.
extern "C" __global__ void gate_sim(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ x_states,      // UNUSED by K1 (state comes from rep_states); compat
    const unsigned* __restrict__ rep_states,    // (was off_lo) n_shots*k*N_LIMBS rep-boundary states
    const unsigned* __restrict__ slot_meta,     // (was off_hi) n_gates*9 predecessor constants
    unsigned* __restrict__ cols,
    unsigned* __restrict__ qdecode,             // UNUSED (kept for arg-list compatibility)
    unsigned* __restrict__ rc_hist,             // (was rc_lo) 2^rc_log rc multiplicity histogram over d
    unsigned* __restrict__ rc_hi,               // INERT (kept for arg-list compatibility)
    unsigned k,
    unsigned n_gates,
    unsigned n_shots,
    unsigned long padded_rows)
{
    (void)x_states; (void)qdecode; (void)rc_hi;

    // THREAD-PER-EXECUTION: one thread per (shot, rep) = n_shots*k threads.
    unsigned long exec = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long n_exec = (unsigned long)n_shots * (unsigned long)k;
    if (exec >= n_exec) return;
    unsigned rep  = (unsigned)(exec % (unsigned long)k);

    // Local 512-qubit value state, seeded from K0's rep-boundary snapshot (32 limbs). This is the
    // value of every qubit at the START of (shot, rep) — exactly last_val entering this rep. ts/prev_ts
    // are CLOSED FORM (below) — no running per-address counter is threaded.
    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = rep_states[exec * N_LIMBS + i];

    // Rows this execution owns: [exec*n_gates, (exec+1)*n_gates). row(shot,rep,g)=exec*n_gates+g.
    unsigned long row = exec * (unsigned long)n_gates;

    for (unsigned g = 0; g < n_gates; g++) {
        unsigned opcode = gates[g * 4u + 0u];
        unsigned tq     = gates[g * 4u + 1u];
        unsigned aq     = gates[g * 4u + 2u];
        unsigned bq     = gates[g * 4u + 3u];

        unsigned is_nop = (opcode == OP_NOP) ? 1u : 0u;
        unsigned is_not = (opcode == OP_NOT) ? 1u : 0u;
        unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
        unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
        unsigned a_active = is_cnot + is_tof;
        unsigned b_active = is_tof;

        unsigned meta = g * 9u;
        // pc for this execution's gate g = rep*n_gates + g (verifier-pinned per-shot program counter).
        unsigned pc = rep * n_gates + g;

        // Read a qubit's current value from the local limb state.
        #define QVAL(q) ((limbs[(q) / LIMB_BITS] >> ((q) % LIMB_BITS)) & 1u)
        // prev_ts from a slot's (prev_gate, prev_slot, wrap) predecessor triple at slot_meta[meta+off].
        #define PREVTS(off) prev_ts_of(slot_meta[meta+(off)+0u], slot_meta[meta+(off)+1u], \
                                       slot_meta[meta+(off)+2u], rep, n_gates)

        // Access order matches simulate_shot: ctrl_a, ctrl_b, then target. ts = pc + 1 (program-order
        // timestamp), SHARED by all accesses of this step (no slot — ts is not emitted). Values are
        // read from the local state (v_before); the target write updates the local state so LATER
        // gates in this rep read the new value. d = ts - prev_ts - 1 = pc - prev_ts.
        unsigned ts = pc + 1u;
        // ctrl_a (active iff a_active): read propagates value (v unchanged).
        unsigned a_addr = 0u, a_prev = 0u, a_v = 0u, a_d = 0u;
        if (a_active) {
            a_addr = aq;
            a_v    = QVAL(aq);
            a_prev = PREVTS(0u);
            a_d    = ts - a_prev - 1u;   // = pc - prev_ts, >= 0 (prev_ts is an earlier program-order ts or 0)
            atomicAdd(&rc_hist[a_d], 1u);
        }
        // ctrl_b (active iff b_active).
        unsigned b_addr = 0u, b_prev = 0u, b_v = 0u, b_d = 0u;
        if (b_active) {
            b_addr = bq;
            b_v    = QVAL(bq);
            b_prev = PREVTS(3u);
            b_d    = ts - b_prev - 1u;
            atomicAdd(&rc_hist[b_d], 1u);
        }
        // target (always active): read+write.
        unsigned t_addr = tq;
        unsigned t_v    = QVAL(tq);     // v_before
        unsigned t_prev = PREVTS(6u);
        unsigned t_d    = ts - t_prev - 1u;
        atomicAdd(&rc_hist[t_d], 1u);

        unsigned ab = a_v * b_v;
        unsigned fire = is_not + is_cnot * a_v + is_tof * ab;   // in {0,1}
        unsigned v_after = t_v ^ fire;
        int delta_signed = (int)v_after - (int)t_v;             // {-1,0,1}

        // Commit target write to the LOCAL state (so subsequent gates in this rep see the new value).
        {
            unsigned tl = tq / LIMB_BITS, tbp = tq % LIMB_BITS, mask = 1u << tbp;
            if (delta_signed > 0) limbs[tl] += mask;
            else if (delta_signed < 0) limbs[tl] -= mask;
        }
        #undef QVAL
        #undef PREVTS

        // Emit the 19 cells in cell_at order (ACCESS_BLOCK = 4: addr,prev_ts,v,d). ts (=pc+1) and the
        // target's v_after (=v_before+delta) are NOT emitted — they are inlined in the AIR / K4
        // interaction kernel. (void v_after / delta arithmetic still updates the local state above.)
        unsigned cc = 0u;
        #define EMIT(v) cols[(unsigned long)(cc++) * padded_rows + row] = (v)
        EMIT(is_nop); EMIT(is_not); EMIT(is_cnot); EMIT(is_tof);              // 0..4
        EMIT(t_addr); EMIT(t_prev); EMIT(t_v); EMIT(t_d);                     // 4..8  target
        EMIT(a_addr); EMIT(a_prev); EMIT(a_v); EMIT(a_d);                     // 8..12 ctrl_a
        EMIT(b_addr); EMIT(b_prev); EMIT(b_v); EMIT(b_d);                     // 12..16 ctrl_b
        EMIT(ab); EMIT(fire);                                                 // 16..18
        EMIT(delta_signed >= 0 ? (unsigned)delta_signed
                               : (unsigned)((int)M31_MOD + delta_signed));    // 18 delta_to_m31
        #undef EMIT

        row += 1u;
    }
}

// Padding rows [real_rows, padded_rows) match `Row::padding()` = ALL ZERO (AccessCols::inactive()
// is addr=ts=prev_ts=v=0; is_* = 0; ab=fire=delta=0). `cols` is pre-zeroed by the caller, so there
// is nothing to write. Kept as a no-op stub so the Rust launch glue (get_func "fill_padding") and
// its shard-loop call site do not need to change. (Old encoding wrote mask=1 on 3 columns; the new
// AccessCols::inactive() has no such non-zero field.)
extern "C" __global__ void fill_padding(
    unsigned* __restrict__ cols,
    unsigned long padded_rows,
    unsigned long real_rows)
{
    (void)cols; (void)padded_rows; (void)real_rows;
}
"#;

pub const TRACE_COLUMNS: usize = 19;

/// Run K1 on the GPU and copy the trace + histograms back to the host (for the P3.1 byte-identity
/// test). The production GPU-resident path (return BaseColumns, no D2H) is P3.4.
///
/// Inputs (host, already flattened by the caller):
/// - `gates_flat`: n_gates*4 (opcode, target_q, ctrl_a_q, ctrl_b_q); inactive controls -> any value
///   (kernel guards on a_active/b_active).
/// - `x_states`: n_shots*32 initial limbs (state_to_limbs of each shot's x_hex).
/// - `off_lo`/`off_hi`: 16 each (RcIndex offsets).
/// Returns (cols [column-major, TRACE_COLUMNS*padded_rows], qdecode[512], rc_hist[2^rc_log], rc_hi[inert,1]).
/// NOTE (padding): real rows [0, n_shots*k*n_gates) are written by the kernel; padding rows stay
/// zero here — the caller must set the 3 read-block `mask` columns = 1 for padding rows to match
/// `Row::padding` (TODO; the byte-identity test compares real rows + histograms first).
#[cfg(feature = "gpu-cuda")]
pub fn gpu_gen_main_trace(
    gates_flat: &[u32],
    x_states: &[u32],
    off_lo: &[u32],
    off_hi: &[u32],
    k: u32,
    n_gates: u32,
    n_shots: u32,
    padded_rows: usize,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>), String> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    let dev = cuda_device()?;

    // N4: compile+load once per process (cached); just get_func afterwards.
    gate_sim_module(&dev)?;
    let func = dev
        .get_func("gate_sim_mod", "gate_sim")
        .ok_or_else(|| "get_func gate_sim".to_string())?;

    let _ = (off_lo, off_hi); // RcIndex offsets no longer used (repurposed arg slots carry rep_states/slot_meta).
    let d_gates = dev
        .htod_copy(gates_flat.to_vec())
        .map_err(|e| format!("htod gates: {e}"))?;
    let d_x = dev
        .htod_copy(x_states.to_vec())
        .map_err(|e| format!("htod x_states: {e}"))?;
    let mut d_cols = dev
        .alloc_zeros::<u32>(TRACE_COLUMNS * padded_rows)
        .map_err(|e| format!("alloc cols: {e}"))?;
    let mut d_qd = dev
        .alloc_zeros::<u32>(512)
        .map_err(|e| format!("alloc qdecode: {e}"))?;
    // rc multiplicity histogram over the single diff `d ∈ [0, 2^rc_log)`. Sized DYNAMICALLY to
    // `1 << rc_log_size(k*n_gates)` (NOT a fixed 2^16) — the kernel bumps `rc_hist[d]` with d up to
    // total_pc-1, so the buffer must cover [0,2^rc_log). `d_hi` stays inert (arg-list compat).
    let rc_hist_len = 1usize << crate::rc_log_size((k as usize) * (n_gates as usize));
    let mut d_lo = dev
        .alloc_zeros::<u32>(rc_hist_len)
        .map_err(|e| format!("alloc rc_hist: {e}"))?;
    let mut d_hi = dev
        .alloc_zeros::<u32>(1)
        .map_err(|e| format!("alloc rc_hi (inert): {e}"))?;

    // Thread-per-execution scratch: rep-boundary states (K0 → K1) + closed-form ts constants.
    let (mut d_rep, d_slot) = alloc_rep_and_slot(&dev, &d_gates, k, n_gates, n_shots)?;

    let block = 256u32;
    // K0: thread-per-SHOT — fill rep-boundary states (value chain, linear in k).
    launch_k0_states(&dev, &d_gates, &d_x, &mut d_rep, k, n_gates, n_shots)?;
    // K1: thread-per-EXECUTION — one thread per (shot, rep) = n_shots*k threads. rep_states seeds the
    // value chain, slot_meta gives the closed-form ts; d_x/d_qd/d_lo/d_hi are UNUSED (arg compat).
    let n_exec = (n_shots as u64) * (k as u64);
    let grid = (n_exec.div_ceil(block as u64)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // 8 device pointers + 4 scalars = 12 args (cudarc launch tuple cap). off_lo slot = rep_states,
    // off_hi slot = slot_meta.
    unsafe {
        func.launch(
            cfg,
            (
                &d_gates,
                &d_x,
                &d_rep,
                &d_slot,
                &mut d_cols,
                &mut d_qd,
                &mut d_lo,
                &mut d_hi,
                k,
                n_gates,
                n_shots,
                padded_rows as u64,
            ),
        )
        .map_err(|e| format!("launch gate_sim: {e}"))?;
    }

    // Populate padding rows (mask columns = 1) to match Row::padding().
    let real_rows = (n_shots as u64) * (k as u64) * (n_gates as u64);
    let n_pad = (padded_rows as u64).saturating_sub(real_rows);
    if n_pad > 0 {
        let fill = dev
            .get_func("gate_sim_mod", "fill_padding")
            .ok_or_else(|| "get_func fill_padding".to_string())?;
        let pad_block = 256u32;
        let pad_grid = (n_pad as u32).div_ceil(pad_block);
        let pad_cfg = LaunchConfig {
            grid_dim: (pad_grid, 1, 1),
            block_dim: (pad_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            fill.launch(pad_cfg, (&mut d_cols, padded_rows as u64, real_rows))
                .map_err(|e| format!("launch fill_padding: {e}"))?;
        }
    }
    dev.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut cols = vec![0u32; TRACE_COLUMNS * padded_rows];
    let mut qd = vec![0u32; 512];
    let mut lo = vec![0u32; rc_hist_len]; // rc multiplicity histogram over d, length 2^rc_log
    let mut hi = vec![0u32; 1]; // inert
    dev.dtoh_sync_copy_into(&d_cols, &mut cols)
        .map_err(|e| format!("dtoh cols: {e}"))?;
    dev.dtoh_sync_copy_into(&d_qd, &mut qd)
        .map_err(|e| format!("dtoh qdecode: {e}"))?;
    dev.dtoh_sync_copy_into(&d_lo, &mut lo)
        .map_err(|e| format!("dtoh rc_hist: {e}"))?;
    dev.dtoh_sync_copy_into(&d_hi, &mut hi)
        .map_err(|e| format!("dtoh rc_hi (inert): {e}"))?;
    Ok((cols, qd, lo, hi))
}

/// P3.1 soundness gate: assert the GPU K1 trace (real rows + rc multiplicity histogram) is
/// BYTE-IDENTICAL to the CPU reference (`build_rows` + `cell_at` + `build_rc_table`). Run on a small
/// fixture (k1-n4) via `GATE_AIR_GPU_TEST=k1` (hooked in main). Compares the 19 main columns
/// cell-by-cell over the real rows AND the rc-table multiplicity histogram (2^rc_log rows over the
/// single diff `d`, val[i]=i) against `build_rc_table(&rows, rc_log).multiplicity`; padding rows are all-zero.
#[cfg(all(feature = "gpu-cuda", feature = "diag"))]
pub fn k1_byte_identity(
    gates: &[crate::Gate],
    cases: &[crate::TestCase],
    k: usize,
    rc_lo: &crate::RcIndex,
    _rc_hi: &crate::RcIndex, // UNUSED (qubit-memory encoding dropped rc_hi); kept for call-site compat.
) -> Result<(), String> {
    let n_gates = gates.len();
    let n_shots = cases.len();
    let real_rows = n_shots * k * n_gates;
    let padded_rows = real_rows
        .next_power_of_two()
        .max(1 << (crate::LOG_N_LANES + 2)); // matches main.rs

    // CPU reference (qubit-memory `build_rows`: (rows, boundary)); boundary is a separate component
    // not covered by K1, so it is ignored here. `build_rc_table` gives the CPU rc multiplicity
    // histogram (one row per row_of(pos,limb)) that the K1 device histogram must match.
    let (rows, _boundary) = crate::build_rows(gates, cases, k).map_err(|e| e.to_string())?;
    if rows.len() != real_rows {
        return Err(format!(
            "rows.len()={} != real_rows={}",
            rows.len(),
            real_rows
        ));
    }
    // Dynamic rc supply-table log-size: rc_log_size(total_pc) with total_pc = k*n_gates.
    let rc_log = crate::rc_log_size(k * n_gates);
    let cpu_rc = crate::build_rc_table(&rows, rc_log);

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
        x_states.extend_from_slice(&crate::state_to_limbs(&bytes));
    }
    // off_lo/off_hi are unused by the kernel now (arg-compat only); pass the RcIndex offsets.
    let off_lo: Vec<u32> = (0..crate::LIMB_BITS)
        .map(|p| rc_lo.offset[p] as u32)
        .collect();
    let off_hi = off_lo.clone();

    // GPU. `hist` (2nd histogram return) is the rc multiplicity histogram; `qd`/`hi` are unused.
    let (cols, _qd, hist, _hi) = gpu_gen_main_trace(
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
            let want = crate::cell_at(&rows[r], c);
            if got != want {
                mismatches += 1;
                if samples.len() < 10 {
                    samples.push(format!("row {r} col {c}: gpu={got} cpu={want}"));
                }
            }
        }
    }

    // Compare the rc multiplicity histogram (GPU device histogram vs CPU build_rc_table). The device
    // histogram is indexed directly by the single diff `d` (row [0,2^rc_log), val[i]=i) — exactly
    // RcTable's flattened row order (row_of(d) == d).
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

// ============================================================================
// P3.2 — K4: CUDA LogUp interaction trace (full on-device).
// ============================================================================
//
// Generates gate_air's main-component interaction M31 columns (N_LOGUP_COLS=5 LogUp columns ×
// 4 coords = 20) + claimed_sum on the GPU, byte-identical to the CPU `gen_main_interaction` /
// `LogupTraceGenerator`. Consumes K1's main-trace columns (no re-simulation). The last LogUp column
// (k = N_LOGUP_COLS-1 = 4) is the (rc ctrl_b d + program) PAIR batch and carries the cumsum_shift.
//
// Pipeline (per LogUp column k=0..N_LOGUP_COLS, sequential — col k accumulates onto col k-1):
//   1. logup_col_gen[batch k]: per row, combine the batch's relation tuple(s) -> (num, denom)
//      where d = (Σ_i alpha^i · values[i]) − z   (QM31), num = m0·d1 + m1·d0, denom = d0·d1.
//   2. logup_finalize_col: value = num · denom^{-1} (per-element QM31 inverse — byte-identical
//      to the CPU batch inverse, since the field inverse is unique); running sum across columns.
// Then once, on the last column (k = N_LOGUP_COLS-1 = 4):
//   3. logup_cumsum_reduce  -> coordinate_sums = claimed_sum (Σ rows of last col, per coord).
//   4. logup_cumsum_shift   -> subtract cumsum_shift = claimed_sum / 2^log_size.
//   5. inclusive_prefix_sum (per coord): bit-reverse -> circle→coset -> scan -> coset→circle ->
//      bit-reverse. Matches stwo's `inclusive_prefix_sum` (coset-order inclusive scan of
//      bit-reversed-CircleDomain evals). Hand-rolled NVRTC scan (block_scan + add_offsets).
//
// FIELD MATH is transcribed exactly from stwo (qm31.rs/cm31.rs): CM31 is i^2 = -1
// (mul = (a.r·b.r − a.i·b.i, a.r·b.i + a.i·b.r)); QM31 = CM31[j]/(j^2 − (2+i)), R = (2,1).
// (NB: obelyzk fft.rs uses a different u^2=2 convention — NOT used here.)
//
// SOUNDNESS: validated by `k4_byte_identity` (20 cols + claimed_sum) vs the CPU reference
// using a FIXED `GateRel::dummy()` (z,alpha) before it is trusted.
//
// OCCUPANCY: K4 is ALREADY thread-per-EXECUTION-instance — every per-row kernel here
// (logup_col_gen / logup_finalize_col / logup_cumsum_shift / the prefix-sum stages) maps one
// thread per ROW with `row = blockIdx*blockDim + threadIdx; if (row >= padded_rows) return;`, and
// the launch grid is `padded_rows.div_ceil(block)`. Since padded_rows ≈ n_shots*k*n_gates rounded
// to a power of two, parallelism is already millions of threads at the Tanuj benchmark and scales
// with total work — it never had K1's thread-per-shot pathology, so K4's mapping is unchanged here.
// (logup_cumsum_reduce is a grid-stride block reduction capped at 1024 blocks, which is correct and
// fully occupied.) The K4 column writes are coordinate-major (4 coords × padded_rows); each thread
// writes its row's 4 coords at stride padded_rows, the same layout K1 uses.

/// Number of LogUp columns (batches) gate_air's main component emits. The pc-pinned ts + single-`d`
/// rc-table range-check encoding emits 10 relation entries — 3 qubitmem pairs (target/ctrl_a/ctrl_b
/// Use+Yield), 3 rc single-`d` terms (target/ctrl_a/ctrl_b), and 1 program singleton — folded by
/// `finalize_logup_in_pairs` into 5 batches (all pairs). Each batch is a SecureColumnByCoords (4 M31).
/// Order MUST match `gen_main_interaction` (main.rs) exactly.
pub const N_LOGUP_COLS: usize = 5;
/// Number of M31 interaction columns committed = 5 × 4 = 20.
pub const N_INTERACTION_COLS: usize = N_LOGUP_COLS * 4;
/// GateRel width (= `relation!(GateRel, 6)`): number of alpha powers uploaded (widest tuple =
/// program = tag + 5 payload = 6).
pub const GATE_REL_WIDTH: usize = 6;

/// NVRTC source: M31/CM31/QM31 device arithmetic + the K4 interaction kernels.
pub const INTERACTION_KERNEL: &str = r#"
#define M31_P 0x7FFFFFFFu

// ---- M31 (p = 2^31 - 1) ----
__device__ __forceinline__ unsigned m31_add(unsigned a, unsigned b) {
    unsigned r = a + b;
    unsigned reduced = (r & M31_P) + (r >> 31);
    return reduced == M31_P ? 0u : reduced;
}
__device__ __forceinline__ unsigned m31_sub(unsigned a, unsigned b) {
    unsigned r = a - b;
    return r + (M31_P & -(r >> 31));
}
__device__ __forceinline__ unsigned m31_neg(unsigned a) {
    return (M31_P - a) * (a != 0u);
}
__device__ __forceinline__ unsigned m31_mul(unsigned a, unsigned b) {
    unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
    unsigned lo = (unsigned)(prod & M31_P);
    unsigned hi = (unsigned)(prod >> 31);
    return m31_add(lo, hi);
}
__device__ __forceinline__ unsigned m31_sqr(unsigned a) { return m31_mul(a, a); }
__device__ unsigned m31_pow(unsigned base, unsigned exp) {
    unsigned result = 1u, b = base;
    while (exp > 0u) { if (exp & 1u) result = m31_mul(result, b); b = m31_sqr(b); exp >>= 1; }
    return result;
}
// a^(p-2) = a^(2^31 - 3).
__device__ __forceinline__ unsigned m31_inv(unsigned a) { return m31_pow(a, 0x7FFFFFFDu); }

// ---- CM31 = M31[i]/(i^2 + 1) ----
struct cm31 { unsigned a; unsigned b; };
__device__ __forceinline__ cm31 cm31_add(cm31 x, cm31 y) { return {m31_add(x.a,y.a), m31_add(x.b,y.b)}; }
__device__ __forceinline__ cm31 cm31_sub(cm31 x, cm31 y) { return {m31_sub(x.a,y.a), m31_sub(x.b,y.b)}; }
__device__ __forceinline__ cm31 cm31_mul(cm31 x, cm31 y) {
    return { m31_sub(m31_mul(x.a,y.a), m31_mul(x.b,y.b)),
             m31_add(m31_mul(x.a,y.b), m31_mul(x.b,y.a)) };
}
__device__ __forceinline__ cm31 cm31_inv(cm31 t) {
    unsigned factor = m31_inv(m31_add(m31_mul(t.a,t.a), m31_mul(t.b,t.b)));
    return { m31_mul(t.a, factor), m31_mul(m31_neg(t.b), factor) };
}

// ---- QM31 = CM31[j]/(j^2 - (2+i)), R = (2,1) ----
struct qm31 { cm31 a; cm31 b; };
__device__ __forceinline__ qm31 qm31_add(qm31 x, qm31 y) { return {cm31_add(x.a,y.a), cm31_add(x.b,y.b)}; }
__device__ __forceinline__ qm31 qm31_sub(qm31 x, qm31 y) { return {cm31_sub(x.a,y.a), cm31_sub(x.b,y.b)}; }
__device__ __forceinline__ qm31 qm31_mul(qm31 x, qm31 y) {
    // (a + b·j)(c + d·j) = (a·c + R·b·d) + (a·d + b·c)·j
    cm31 R = {2u, 1u};
    cm31 ac = cm31_mul(x.a, y.a);
    cm31 bd = cm31_mul(x.b, y.b);
    cm31 ad = cm31_mul(x.a, y.b);
    cm31 bc = cm31_mul(x.b, y.a);
    return { cm31_add(ac, cm31_mul(R, bd)), cm31_add(ad, bc) };
}
__device__ __forceinline__ qm31 qm31_mul_m31(qm31 x, unsigned s) {
    return { { m31_mul(x.a.a,s), m31_mul(x.a.b,s) }, { m31_mul(x.b.a,s), m31_mul(x.b.b,s) } };
}
__device__ __forceinline__ qm31 qm31_inv(qm31 t) {
    // (a + b·j)^{-1} = (a − b·j) / (a^2 − (2+i)·b^2).
    cm31 b2 = cm31_mul(t.b, t.b);
    cm31 ib2 = { m31_neg(b2.b), b2.a };                       // i · b2
    cm31 denom = cm31_sub(cm31_mul(t.a, t.a), cm31_add(cm31_add(b2, b2), ib2));
    cm31 di = cm31_inv(denom);
    cm31 nb = { m31_neg(t.b.a), m31_neg(t.b.b) };
    return { cm31_mul(t.a, di), cm31_mul(nb, di) };
}
__device__ __forceinline__ unsigned qm31_zero_get() { return 0u; }

// combine(values) = ( Σ_i alpha_powers[i] · values[i] ) − z   (QM31).
// `ap` holds GATE_REL_WIDTH QM31s flattened as 4 unsigned each.
__device__ __forceinline__ qm31 logup_combine(
    const unsigned* vals, int n,
    unsigned z0, unsigned z1, unsigned z2, unsigned z3,
    const unsigned* ap)
{
    qm31 acc = { {0u,0u}, {0u,0u} };
    for (int i = 0; i < n; i++) {
        qm31 apw = { { ap[i*4+0], ap[i*4+1] }, { ap[i*4+2], ap[i*4+3] } };
        acc = qm31_add(acc, qm31_mul_m31(apw, vals[i]));
    }
    qm31 z = { {z0,z1}, {z2,z3} };
    return qm31_sub(acc, z);
}

// K4a: per-row (num, denom) for one of the 5 batches (3 qubitmem pairs + 2 rc/program pairs). Reads
// K1's 19 main-trace columns (column-major in `cols`). Outputs interleaved per row:
// num[row*4+j], denom[row*4+j].
//
// Column layout (cell_at, 19 cols, ACCESS_BLOCK=4): 0..4 opcode one-hots;
//   target addr=4,prev_ts=5,v=6,d=7;
//   ctrl_a addr=8,prev_ts=9,v=10,d=11;
//   ctrl_b addr=12,prev_ts=13,v=14,d=15; ab=16,fire=17,delta=18.
// ts (= pc+1) and the target's v_after (= v_before+delta) are INLINED here (recomputed from pc / the
// v+delta columns), NOT read as columns.
// Relation tags: QUBITMEM=1, RC=2, PROGRAM=5. Widths: qubitmem tuple=5, rc tuple=2 (TAG_RC, d),
// program tuple=6 (rel width = GATE_REL_WIDTH = 6). Batch order MUST match gen_main_interaction /
// circuit_statement EXACTLY (the 10-entry stream folded into 5 pairs by finalize_logup_in_pairs):
//   case 0..2 : 3 qubitmem pairs   (target / ctrl_a / ctrl_b : Use[+active] / Yield[-active])
//   case 3    : rc target d [+enabler]   / rc ctrl_a d [+a_active]
//   case 4    : rc ctrl_b d [+b_active]  / program SINGLETON [+enabler]
// A batch is a SINGLETON when n1 == 0 (only v0/m0 contribute: num = m0, den = d0); else a pair. (No
// singleton batches remain in this layout — the program now pairs with rc ctrl_b d in case 4.)
extern "C" __global__ void logup_col_gen(
    const unsigned* __restrict__ cols,
    unsigned long padded_rows,
    unsigned z0, unsigned z1, unsigned z2, unsigned z3,
    const unsigned* __restrict__ ap,
    unsigned pair_id,
    unsigned n_gates,
    const unsigned long* __restrict__ dims, // dims[0]=real_rows, dims[1]=shot_stride (=k*n_gates)
    unsigned* __restrict__ num,
    unsigned* __restrict__ denom)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    unsigned long real_rows   = dims[0];
    unsigned long shot_stride = dims[1];
    #define COL(c) cols[(unsigned long)(c) * padded_rows + row]

    unsigned v0[6]; int n0 = 0;
    unsigned v1[6]; int n1 = 0;
    unsigned m0 = 0u, m1 = 0u; int s0 = 1, s1 = 1;

    // enabler / shot_id / pc are tree0 (POSITIONAL). K4 recomputes them from `row` — identical
    // values to the preprocessed columns (enabler = real-row indicator, shot_id = row / shot_stride,
    // pc = row % shot_stride). pc feeds only the program term's pc_in_prog = pc % n_gates.
    unsigned enabler  = (row < real_rows) ? 1u : 0u;
    unsigned shot_id  = (row < real_rows) ? (unsigned)(row / shot_stride) : 0u;
    unsigned pc       = (row < real_rows) ? (unsigned)(row % shot_stride) : 0u;
    unsigned a_active = m31_add(COL(2), COL(3));   // is_cnot + is_toffoli
    unsigned b_active = COL(3);                     // is_toffoli
    // ts = pc + 1 (INLINED, shared by all accesses of the step; not a column).
    unsigned ts = m31_add(pc, 1u);
    // v_after = v_before + delta = t.v(col6) + delta(col18) (INLINED target write value; not a column).
    unsigned v_after = m31_add(COL(6), COL(18));

    switch (pair_id) {
    case 0: // qubitmem target Use (+enabler): [1, shot, t.addr, t.prev_ts, t.v_before]
            //          target Yield (-enabler): [1, shot, t.addr, ts=pc+1,  v_after]
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(4); v0[3]=COL(5); v0[4]=COL(6); n0=5;   // t.addr,t.prev_ts,t.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(4); v1[3]=ts;     v1[4]=v_after; n1=5;   // ts=pc+1, v_after=v+delta
        m0=enabler; s0=1; m1=enabler; s1=-1; break;
    case 1: // qubitmem ctrl_a Use (+a_active): [1, shot, a.addr, a.prev_ts, a.v]
            //          ctrl_a Yield (-a_active): [1, shot, a.addr, ts=pc+1,  a.v]  (read propagates)
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(8); v0[3]=COL(9);  v0[4]=COL(10); n0=5; // a.addr,a.prev_ts,a.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(8); v1[3]=ts;      v1[4]=COL(10); n1=5;
        m0=a_active; s0=1; m1=a_active; s1=-1; break;
    case 2: // qubitmem ctrl_b Use (+b_active): [1, shot, b.addr, b.prev_ts, b.v]
            //          ctrl_b Yield (-b_active): [1, shot, b.addr, ts=pc+1,  b.v]
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(12); v0[3]=COL(13); v0[4]=COL(14); n0=5; // b.addr,b.prev_ts,b.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(12); v1[3]=ts;      v1[4]=COL(14); n1=5;
        m0=b_active; s0=1; m1=b_active; s1=-1; break;
    case 3: // rc target d (+enabler): [2, t.d] / rc ctrl_a d (+a_active): [2, a.d]
        v0[0]=2u; v0[1]=COL(7);  n0=2;                                             // t.d=col7
        v1[0]=2u; v1[1]=COL(11); n1=2;                                             // a.d=col11
        m0=enabler; s0=1; m1=a_active; s1=1; break;
    case 4: // rc ctrl_b d (+b_active): [2, b.d] / program (+enabler): [5, pc%ng, opcode_scalar, t/a/b addr]
        v0[0]=2u; v0[1]=COL(15); n0=2;                                             // b.d=col15
        v1[0]=5u; v1[1]=(unsigned)((unsigned long)pc % (unsigned long)n_gates);
        v1[2]=m31_add(m31_add(COL(1), m31_mul(2u,COL(2))), m31_mul(3u,COL(3))); // opcode_scalar
        v1[3]=COL(4); v1[4]=COL(8); v1[5]=COL(12); n1=6;                        // t/a/b addr
        m0=b_active; s0=1; m1=enabler; s1=1; break;
    }

    // SINGLETON (n1 == 0): num = m0, den = d0. PAIR: num = m0*d1 + m1*d0, den = d0*d1.
    qm31 d0 = logup_combine(v0, n0, z0,z1,z2,z3, ap);
    unsigned mm0 = (s0 < 0) ? m31_neg(m0) : m0;
    qm31 qm0 = { {mm0,0u}, {0u,0u} };
    qm31 nume, den;
    if (n1 == 0) {
        nume = qm0;
        den  = d0;
    } else {
        qm31 d1 = logup_combine(v1, n1, z0,z1,z2,z3, ap);
        unsigned mm1 = (s1 < 0) ? m31_neg(m1) : m1;
        qm31 qm1 = { {mm1,0u}, {0u,0u} };
        nume = qm31_add(qm31_mul(qm0, d1), qm31_mul(qm1, d0));
        den  = qm31_mul(d0, d1);
    }

    num[row*4+0]=nume.a.a; num[row*4+1]=nume.a.b; num[row*4+2]=nume.b.a; num[row*4+3]=nume.b.b;
    denom[row*4+0]=den.a.a; denom[row*4+1]=den.a.b; denom[row*4+2]=den.b.a; denom[row*4+3]=den.b.b;
    #undef COL
}

// K4b: value = num · denom^{-1}; running sum onto previous logup column.
// `inter` holds the N_INTERACTION_COLS (20) interaction columns, column-major (logup col k coord j = (k*4+j)).
extern "C" __global__ void logup_finalize_col(
    unsigned rep_index,
    unsigned long padded_rows,
    const unsigned* __restrict__ num,
    const unsigned* __restrict__ denom,
    unsigned* __restrict__ inter)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    qm31 nume = { { num[row*4+0], num[row*4+1] }, { num[row*4+2], num[row*4+3] } };
    qm31 den  = { { denom[row*4+0], denom[row*4+1] }, { denom[row*4+2], denom[row*4+3] } };
    qm31 value = qm31_mul(nume, qm31_inv(den));
    qm31 prev = { {0u,0u}, {0u,0u} };
    if (rep_index > 0u) {
        unsigned long b = (unsigned long)(rep_index - 1u) * 4u;
        prev.a.a = inter[(b+0)*padded_rows+row];
        prev.a.b = inter[(b+1)*padded_rows+row];
        prev.b.a = inter[(b+2)*padded_rows+row];
        prev.b.b = inter[(b+3)*padded_rows+row];
    }
    qm31 acc = qm31_add(value, prev);
    unsigned long b = (unsigned long)rep_index * 4u;
    inter[(b+0)*padded_rows+row] = acc.a.a;
    inter[(b+1)*padded_rows+row] = acc.a.b;
    inter[(b+2)*padded_rows+row] = acc.b.a;
    inter[(b+3)*padded_rows+row] = acc.b.b;
}

__device__ __forceinline__ unsigned m31_atomic_add(unsigned* addr, unsigned val) {
    unsigned old = *addr, assumed;
    do { assumed = old; old = atomicCAS(addr, assumed, m31_add(assumed, val)); } while (assumed != old);
    return old;
}

// K4c.1: claimed_sum = Σ_row last_col (per coordinate). Block-reduce + atomic into sums[4].
extern "C" __global__ void logup_cumsum_reduce(
    unsigned long padded_rows,
    unsigned last_k,
    const unsigned* __restrict__ inter,
    unsigned* __restrict__ sums)
{
    extern __shared__ unsigned sh[];   // 4 * blockDim.x
    unsigned* s0 = &sh[0];
    unsigned* s1 = &sh[blockDim.x];
    unsigned* s2 = &sh[2*blockDim.x];
    unsigned* s3 = &sh[3*blockDim.x];
    unsigned long b = (unsigned long)last_k * 4u;
    unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
    unsigned a0=0u,a1=0u,a2=0u,a3=0u;
    for (unsigned long i = tid; i < padded_rows; i += stride) {
        a0 = m31_add(a0, inter[(b+0)*padded_rows+i]);
        a1 = m31_add(a1, inter[(b+1)*padded_rows+i]);
        a2 = m31_add(a2, inter[(b+2)*padded_rows+i]);
        a3 = m31_add(a3, inter[(b+3)*padded_rows+i]);
    }
    s0[threadIdx.x]=a0; s1[threadIdx.x]=a1; s2[threadIdx.x]=a2; s3[threadIdx.x]=a3;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (threadIdx.x < s) {
            s0[threadIdx.x]=m31_add(s0[threadIdx.x],s0[threadIdx.x+s]);
            s1[threadIdx.x]=m31_add(s1[threadIdx.x],s1[threadIdx.x+s]);
            s2[threadIdx.x]=m31_add(s2[threadIdx.x],s2[threadIdx.x+s]);
            s3[threadIdx.x]=m31_add(s3[threadIdx.x],s3[threadIdx.x+s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        m31_atomic_add(&sums[0], s0[0]);
        m31_atomic_add(&sums[1], s1[0]);
        m31_atomic_add(&sums[2], s2[0]);
        m31_atomic_add(&sums[3], s3[0]);
    }
}

// K4c.2: subtract cumsum_shift = claimed_sum / trace_size from the last column.
extern "C" __global__ void logup_cumsum_shift(
    unsigned long padded_rows,
    unsigned last_k,
    unsigned trace_size,
    const unsigned* __restrict__ sums,
    unsigned* __restrict__ inter)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    qm31 claimed = { { sums[0], sums[1] }, { sums[2], sums[3] } };
    qm31 shift = qm31_mul_m31(claimed, m31_inv(trace_size));
    unsigned long b = (unsigned long)last_k * 4u;
    inter[(b+0)*padded_rows+row] = m31_sub(inter[(b+0)*padded_rows+row], shift.a.a);
    inter[(b+1)*padded_rows+row] = m31_sub(inter[(b+1)*padded_rows+row], shift.a.b);
    inter[(b+2)*padded_rows+row] = m31_sub(inter[(b+2)*padded_rows+row], shift.b.a);
    inter[(b+3)*padded_rows+row] = m31_sub(inter[(b+3)*padded_rows+row], shift.b.b);
}

// ---- inclusive prefix sum (matches stwo inclusive_prefix_sum semantics) ----
__device__ __forceinline__ unsigned bitrev(unsigned x, unsigned bits) {
    unsigned r = 0u;
    for (unsigned i = 0u; i < bits; i++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}
// In-place bit-reverse permutation of one column slice [offset, offset+n).
extern "C" __global__ void ps_bit_reverse(unsigned* col, unsigned long offset, unsigned n, unsigned bits) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned r = bitrev(idx, bits);
    if (r > idx) {
        unsigned t = col[offset+idx];
        col[offset+idx] = col[offset+r];
        col[offset+r] = t;
    }
}
// CircleDomain order -> Coset order: out[2i]=in[i], out[2i+1]=in[n-1-i].
extern "C" __global__ void ps_circle_to_coset(const unsigned* col, unsigned long offset, unsigned* tmp, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2u) { tmp[2u*i] = col[offset+i]; tmp[2u*i+1u] = col[offset + n - 1u - i]; }
}
// Coset order -> CircleDomain order.
extern "C" __global__ void ps_coset_to_circle(const unsigned* tmp, unsigned* col, unsigned long offset, unsigned n) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned half = n/2u;
    if (tid < half) col[offset+tid] = tmp[tid << 1];
    else { unsigned i = tid - half; col[offset+tid] = tmp[n - 1u - (i << 1)]; }
}
// Per-block inclusive scan (Hillis–Steele), in place; writes block totals to block_sums.
extern "C" __global__ void ps_block_scan(unsigned* data, unsigned* block_sums, unsigned n) {
    extern __shared__ unsigned sh[];
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;
    sh[tid] = (gid < n) ? data[gid] : 0u;
    __syncthreads();
    for (unsigned off = 1u; off < blockDim.x; off <<= 1) {
        unsigned t = (tid >= off) ? sh[tid - off] : 0u;
        __syncthreads();
        sh[tid] = m31_add(sh[tid], t);
        __syncthreads();
    }
    if (gid < n) data[gid] = sh[tid];
    if (tid == blockDim.x - 1u) block_sums[blockIdx.x] = sh[tid];
}
// Add each block's exclusive offset (inclusive-scanned block sums of prior blocks).
extern "C" __global__ void ps_add_offsets(unsigned* out, const unsigned* scanned_block_sums, unsigned n) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0u) out[gid] = m31_add(out[gid], scanned_block_sums[blockIdx.x - 1u]);
}
"#;

/// Run the full K4 interaction pipeline on the GPU and copy the N_INTERACTION_COLS (20) interaction columns +
/// claimed_sum back to the host. Inputs: the host-side main trace `cols` (column-major,
/// TRACE_COLUMNS × padded_rows), the drawn `z` and `alpha_powers` (each QM31 → 4 M31, length
/// GATE_REL_WIDTH), and dims. Returns (interaction_cols [N_INTERACTION_COLS × padded_rows,
/// column-major], claimed_sum [4 M31]).
#[cfg(feature = "gpu-cuda")]
#[allow(clippy::too_many_arguments)]
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

    // N4: compile+load the interaction module once per process (cached); just get_func afterwards.
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
    // Positional dims for K4's enabler/shot_id/pc recompute (moved to tree0). One pointer arg keeps
    // the launch tuple within cudarc's LaunchAsync arity cap.
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

/// P3.2 soundness gate: assert the GPU K4 interaction trace (N_INTERACTION_COLS=20 M31 columns + claimed_sum) is
/// byte-identical to the CPU `gen_main_interaction` / `LogupTraceGenerator`, using a FIXED
/// `GateRel::dummy()` (z, alpha) so both sides see the same challenges. Run via
/// `GATE_AIR_GPU_TEST=k4` on a small fixture (k1-n4).
#[cfg(all(feature = "gpu-cuda", feature = "diag"))]
pub fn k4_byte_identity(
    gates: &[crate::Gate],
    cases: &[crate::TestCase],
    k: usize,
    _rc_lo: &crate::RcIndex, // UNUSED (arg-compat); the rc diff `d` comes from the K1 main trace columns.
    _rc_hi: &crate::RcIndex, // UNUSED (single-`d` rc encoding); kept for call-site compat.
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
    let (rows, _boundary) = crate::build_rows(gates, cases, k).map_err(|e| e.to_string())?;
    let elements = crate::LookupElements::dummy();
    let (cpu_cols, cpu_sum) =
        crate::gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);

    // GPU main trace (host) — reuse K1 (includes padding rows now).
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
        x_states.extend_from_slice(&crate::state_to_limbs(&bytes));
    }
    let off_lo: Vec<u32> = (0..crate::LIMB_BITS)
        .map(|p| _rc_lo.offset[p] as u32)
        .collect();
    let off_hi = off_lo.clone(); // unused by the kernel; passed for arg-list compat.
    let (main_cols, _qd, _lo, _hi) = gpu_gen_main_trace(
        &gates_flat,
        &x_states,
        &off_lo,
        &off_hi,
        k as u32,
        n_gates as u32,
        n_shots as u32,
        padded_rows,
    )?;

    // Extract (z, alpha_powers) from the relation via the PUBLIC `Relation::combine`
    // (the inner LookupElements is private to constraint-framework). Works for any
    // challenges (dummy here, real-drawn in P3.4): combine(values) = Σ α^i·v[i] − z, so
    //   combine([0])      = −z                  → z      = −combine([0])
    //   combine(unit_i)   = α^i − z             → α^i    = combine(unit_i) + z
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

    // Compare the N_INTERACTION_COLS (20) interaction columns over ALL padded rows (prefix sum spans them).
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
pub(crate) fn gate_air_relation_m31x4(rel: &crate::GateRel) -> ([u32; 4], Vec<[u32; 4]>) {
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
/// `Relation::combine` (the inner `LookupElements` is private to constraint-framework).
#[cfg(feature = "gpu-cuda")]
fn extract_z_alpha(
    rel: &crate::GateRel,
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

// ============================================================================
// P3.4 — device-resident output: feed K1/K4's GPU columns straight into the
// CudaBackend commit with NO host round-trip.
// ============================================================================
//
// INTEROP (the "two CUDA memory worlds" problem): K1/K4 write into cudarc
// `CudaSlice<u32>` (obelyzk executor, `get_cuda_executor`), while the CudaBackend
// columns are NitrooZK `BaseFieldVec { device_ptr, size }` (raw `cuda_malloc` via
// the stwo cuda FFI). Both allocators run on the SAME device-0 primary CUDA
// context (cudarc `CudaDevice::new(0)` and NitrooZK's driver/runtime `cuMalloc`),
// so a device pointer from one is a valid `cudaMemcpy DeviceToDevice` operand for
// the other. We use approach (B) — a single per-column device-to-device copy:
//   1. K1/K4 run exactly as validated, producing the column-major cudarc buffer.
//   2. For each committed column we allocate a `BaseFieldVec::new_uninitialized`
//      and `copy_uint32_t_vec_from_device_to_device(src_ptr + col*padded_rows,
//      dst.device_ptr, padded_rows)` (D2D — no D2H+H2D).
//   3. Wrap each `BaseFieldVec` in `CircleEvaluation::<CudaBackend>::new(domain, _)`.
// (B) is chosen over (A) wrap-the-cudarc-pointer-as-a-BaseFieldVec because mixing
// two owning allocators behind one `Drop` is fragile (double-free / lifetime
// hazards): cudarc frees its `CudaSlice` on drop, and a borrowed BaseFieldVec
// would dangle once the kernel scope ends. D2D keeps each side owning its own
// memory while still removing the host upload — the actual win.

/// cudarc `CudaSlice<u32>` device address as a raw `*const u32` (the integer
/// `CUdeviceptr`, valid in the shared primary context).
#[cfg(feature = "cuda")]
fn cudarc_dptr(slice: &cudarc::driver::CudaSlice<u32>) -> *const u32 {
    use cudarc::driver::DevicePtr;
    (*slice.device_ptr()) as usize as *const u32
}

/// Copy ONE column (`padded_rows` u32s starting at element offset `col_off`) from a
/// cudarc device buffer into a freshly allocated NitrooZK `BaseFieldVec`, via a
/// device-to-device copy (no host round-trip), and wrap it as a device-resident
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

/// Device-resident K1: run `gpu_gen_main_trace`'s kernels and return the 19 main
/// columns as `CircleEvaluation<CudaBackend>` (device-resident, no host upload),
/// plus the qdecode/rc histogram copied to the host (tiny; the
/// multiplicity columns are still built + uploaded on the CPU path in main.rs),
/// plus the raw column-major main-trace device buffer `d_cols` so the interaction
/// path (K4) can REUSE it instead of re-running K0/K1 (no re-simulation, no
/// re-upload). The 19 returned `CircleEvaluation`s are independent D2D copies of
/// `d_cols`'s columns (see `d2d_column`), so handing `d_cols` back to the caller
/// does not alias or mutate them.
///
/// Reuses the EXACT validated `GATE_SIM_KERNEL` source; only the output handoff
/// changes (D2D into BaseFieldVecs instead of D2H into `Vec<u32>`).
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
        Vec<u32>,
        Vec<u32>,
        cudarc::driver::CudaSlice<u32>,
    ),
    String,
> {
    let dev = cuda_device()?;
    // Upload the shard-INVARIANT inputs (gate list + RcIndex offsets) here, then delegate to the
    // device-buffer body. The base precompute path uploads these ONCE and calls the `_d` body
    // directly (skipping this per-shard upload); only `x_states` is per-shard.
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
        Vec<u32>,
        Vec<u32>,
        cudarc::driver::CudaSlice<u32>,
    ),
    String,
> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    use stwo::core::poly::circle::CanonicCoset;

    let dev = cuda_device()?;

    // N4: compile+load the GATE_SIM module once per process (cached); just get_func afterwards.
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
    let mut d_qd = dev
        .alloc_zeros::<u32>(512)
        .map_err(|e| format!("alloc qdecode: {e}"))?;
    // rc multiplicity histogram over the single diff `d`. Sized DYNAMICALLY to `1 << rc_log_size(
    // k*n_gates)` — the K1 kernel bumps `rc_hist[d]` with d up to total_pc-1, so the buffer must
    // cover [0,2^rc_log) or the kernel writes out of bounds on real (large-k) runs. Production ignores
    // the returned histogram (the multiplicity witness is host-built), but the device buffer must
    // still be correctly sized so the atomic bumps stay in bounds. `d_hi` stays inert (arg compat).
    let rc_hist_len = 1usize << crate::rc_log_size((k as usize) * (n_gates as usize));
    let mut d_lo = dev
        .alloc_zeros::<u32>(rc_hist_len)
        .map_err(|e| format!("alloc rc_hist: {e}"))?;
    let mut d_hi = dev
        .alloc_zeros::<u32>(1)
        .map_err(|e| format!("alloc rc_hi (inert): {e}"))?;
    let _ = (d_off_lo, d_off_hi); // RcIndex offsets unused (arg slots repurposed for rep_states/slot_meta).

    // Thread-per-execution scratch: rep-boundary states (K0 → K1) + closed-form ts constants.
    let (mut d_rep, d_slot) = alloc_rep_and_slot(&dev, d_gates, k, n_gates, n_shots)?;

    let block = 256u32;
    // K0: thread-per-SHOT — fill rep-boundary states (value chain, linear in k).
    launch_k0_states(&dev, d_gates, &d_x, &mut d_rep, k, n_gates, n_shots)?;
    // K1: thread-per-EXECUTION — one thread per (shot, rep). rep_states (off_lo slot) seeds the value
    // chain; slot_meta (off_hi slot) gives the closed-form ts. d_x/d_qd/d_lo/d_hi are UNUSED (compat).
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
                &mut d_qd,
                &mut d_lo,
                &mut d_hi,
                k,
                n_gates,
                n_shots,
                padded_rows as u64,
            ),
        )
        .map_err(|e| format!("launch gate_sim: {e}"))?;
    }

    // Populate padding rows (mask columns = 1) to match Row::padding().
    let real_rows = (n_shots as u64) * (k as u64) * (n_gates as u64);
    let n_pad = (padded_rows as u64).saturating_sub(real_rows);
    if n_pad > 0 {
        let fill = dev
            .get_func("gate_sim_mod", "fill_padding")
            .ok_or_else(|| "get_func fill_padding".to_string())?;
        let pad_block = 256u32;
        let pad_grid = (n_pad as u32).div_ceil(pad_block);
        let pad_cfg = LaunchConfig {
            grid_dim: (pad_grid, 1, 1),
            block_dim: (pad_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            fill.launch(pad_cfg, (&mut d_cols, padded_rows as u64, real_rows))
                .map_err(|e| format!("launch fill_padding: {e}"))?;
        }
    }
    dev.synchronize().map_err(|e| format!("sync: {e}"))?;

    // Device handoff: build the 19 CudaBackend columns from `d_cols` (column-major; column c at
    // element offset c*padded_rows). No host copy.
    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    let cols: Vec<_> = (0..TRACE_COLUMNS)
        .map(|c| d2d_column(&d_cols, c * padded_rows, padded_rows, domain))
        .collect();

    // Histograms are tiny → keep them on host (the multiplicity columns are built
    // + uploaded on the existing CPU path in main.rs).
    let mut qd = vec![0u32; 512];
    let mut lo = vec![0u32; rc_hist_len]; // rc multiplicity histogram over d, length 2^rc_log
    let mut hi = vec![0u32; 1]; // inert
    dev.dtoh_sync_copy_into(&d_qd, &mut qd)
        .map_err(|e| format!("dtoh qdecode: {e}"))?;
    dev.dtoh_sync_copy_into(&d_lo, &mut lo)
        .map_err(|e| format!("dtoh rc_hist: {e}"))?;
    dev.dtoh_sync_copy_into(&d_hi, &mut hi)
        .map_err(|e| format!("dtoh rc_hi (inert): {e}"))?;
    dev.synchronize().map_err(|e| format!("sync hist: {e}"))?;
    // SUCCESS handoff: return the resident `d_cols` device buffer so K4 reads it directly (no K0/K1
    // re-run); it is no longer touched here after the column build above. The buffer frees via cudarc
    // `cuMemFree` when the caller eventually drops it.
    Ok((cols, qd, lo, hi, d_cols))
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

    /// Free the underlying device main-trace buffer NOW, after K4 has consumed it and BEFORE the
    /// tree2 commit. Call this instead of relying on `drop` at end-of-prove. The resident `CudaSlice`
    /// (`cuMemFree`) is dropped at the end of this scope; we then SYNCHRONIZE so the free settles
    /// before tree2's first alloc.
    pub fn free_after_k4(self) -> Result<(), String> {
        drop(self);
        let dev = cuda_device()?;
        dev.synchronize()
            .map_err(|e| format!("sync after main-trace free: {e}"))?;
        Ok(())
    }
}

/// Device-resident K4: run `gpu_gen_interaction`'s pipeline using the REAL drawn
/// `LookupElements` (z/alpha recovered via `extract_z_alpha`) and return the 24
/// interaction columns as `CircleEvaluation<CudaBackend>` (device-resident, no host
/// upload) plus the `claimed_sum` (mixed into the channel in main.rs exactly as the
/// CPU `gen_main_interaction` sum was).
///
/// `main` is the SAME column-major main trace K1 produced earlier in this base proof — the resident
/// device buffer, read here via `COL(c)` (no re-simulation; K0/K1 are NOT re-run). The buffer is
/// only read here (the K4 `logup_col_gen` kernel reads it and writes its own `d_num`/`d_denom`/
/// `d_inter` scratch). Reuses the EXACT validated `INTERACTION_KERNEL` source.
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

    // K4 reads the main-trace columns from the buffer K1 already produced (`main`, passed in by the
    // caller). No K0/K1 re-run, no re-upload of gates/x_states/off_lo/off_hi, no histogram/rep
    // scratch — those were only needed to repopulate the main trace, which now lives in `main`.
    let block = 256u32;

    // ---- K4 interaction pipeline (verbatim launches from gpu_gen_interaction) ----
    // N4: compile+load the interaction module once per process (cached); just get_func afterwards.
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
    // Positional dims for K4's enabler/shot_id/pc recompute (moved to tree0).
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

    // K4 reads K1's resident device buffer in place (byte-for-byte the previous behavior).
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

    // Device-to-device handoff: 28 interaction columns straight to CudaBackend.
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
