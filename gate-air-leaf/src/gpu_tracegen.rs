use crate::tracegen::*;
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
    let d =
        cudarc::driver::CudaDevice::new(ord).map_err(|e| format!("CudaDevice::new({ord}): {e}"))?;
    let _ = slot.set(d.clone());
    Ok(d)
}

/// Process-level module cache: load the AOT fatbin (from cuda/*.cu) ONCE per process instead of per
/// shard, guarded by a `OnceLock`. Returns the module name + a `bool` (true on the first load).
#[cfg(feature = "gpu-cuda")]
fn gate_sim_module(dev: &Arc<cudarc::driver::CudaDevice>) -> Result<(&'static str, bool), String> {
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
        let ptx = cudarc::nvrtc::Ptx::from_file(concat!(env!("OUT_DIR"), "/interaction.fatbin"));
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
    rc_lo: &crate::tracegen::RcIndex,
    _rc_hi: &crate::tracegen::RcIndex, // unused (single-`d` rc); kept for call-site compat.
) -> Result<(), String> {
    let n_gates = gates.len();
    let n_shots = cases.len();
    let real_rows = n_shots * k * n_gates;
    let padded_rows = real_rows
        .next_power_of_two()
        .max(1 << (crate::LOG_N_LANES + 2)); // matches main.rs

    // CPU reference; qubitmem is a separate component not covered by K1, so ignore it. `build_rc_table`
    // gives the CPU rc multiplicity histogram the K1 device histogram must match.
    let (rows, _qubitmem) = build_rows(gates, cases, k).map_err(|e| e.to_string())?;
    if rows.len() != real_rows {
        return Err(format!(
            "rows.len()={} != real_rows={}",
            rows.len(),
            real_rows
        ));
    }
    let rc_log = crate::RC_LOG;
    let cpu_rc = crate::tracegen::build_rc_table(&rows, rc_log);

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
    _rc_lo: &crate::tracegen::RcIndex, // unused (rc diff `d` comes from the K1 main trace); call-site compat.
    _rc_hi: &crate::tracegen::RcIndex, // unused (single-`d` rc); call-site compat.
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
    let (rows, _qubitmem) = build_rows(gates, cases, k).map_err(|e| e.to_string())?;
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
