//! P3.1 — K1: CUDA gate-sim + main-trace kernel for gate_air (on-device "model B").
//!
//! Generates the 191 main-trace columns + the 3 lookup histograms (qdecode / rc_lo / rc_hi)
//! ENTIRELY on the GPU, so the trace never leaves device memory (avoiding the model-A transfer
//! wall).
//!
//! THREAD-PER-EXECUTION (occupancy redesign, 2026-06-29).
//! ------------------------------------------------------
//! The original kernel was thread-per-SHOT: each of `n_shots` threads ran its shot's full
//! `k * n_gates` chain. Parallelism was therefore fixed at `n_shots` regardless of total work,
//! so at the real 9024-shot Tanuj benchmark only ~36 of an A100's 108 SMs were ever touched, and
//! sharding across 8 GPUs (~1128 shots/GPU) lit only ~5 SMs/GPU (~95% idle) — worst at high k.
//!
//! The fix maps ONE THREAD PER EXECUTION = per (shot, rep) pair, so parallelism rises from
//! `n_shots` to `n_shots * k` (~9M at k=1000) and saturates the device. The catch is that the
//! `k` reps are a SEQUENTIAL state chain: rep r+1 of a shot starts from the state rep r ended on
//! (Grover-style; the CPU `simulate_shot` never resets `limbs` between reps). So per-rep threads
//! are NOT independent unless each knows its starting state.
//!
//! We resolve this with a TWO-KERNEL split that keeps total work LINEAR in k (a per-rep
//! fast-forward would be O(k^2) — fatal at k=2000) and is NOT the forbidden per-gate-instance
//! two-pass:
//!   K0 `gate_sim_states` — thread-per-SHOT, but does ONLY the cheap limb update (no column
//!       stores, no histogram atomics). It walks the full `k * n_gates` chain and records the
//!       rep-BOUNDARY state of each shot: `rep_states[shot*k*N_LIMBS + rep*N_LIMBS + i]` = the
//!       32-limb state at the START of (shot, rep) (rep 0 == x_states[shot]). This is k small
//!       state snapshots per shot — not k*n_gates rows.
//!   K1 `gate_sim` — thread-per-EXECUTION. Thread `exec = shot*k + rep` loads its start state
//!       from `rep_states`, sets `pc = rep*n_gates` and `row = exec*n_gates`, then walks its own
//!       `n_gates` gates writing the 191 columns + the 3 histograms exactly as before.
//! The expensive phase (191 stores + 3 atomics per row) now runs with `n_shots*k` threads; the
//! cheap sequential chain stays in K0 with `n_shots` threads (a small fraction of the old cost,
//! since K0 omits all the per-row I/O that dominated the old kernel).
//!
//! SOUNDNESS / BYTE-IDENTITY: the output is unchanged. K1's per-gate body is the SAME line-for-line
//! translation of `simulate_shot` / `ReadCols::live`/`inactive` / `qubit_decode` / `count_read` /
//! `delta_to_m31` / the `cell_at` 191-column layout. The only change is the thread→(shot,rep,row,pc)
//! mapping: row(shot,rep,g) = shot*k*n_gates + rep*n_gates + g is EXACTLY the row the old kernel
//! wrote (old: row started at shot*k*n_gates and incremented through rep,g in order); pc = rep*n_gates
//! + g matches the old monotonic `pc` (which also ran 0..k*n_gates per shot); and each thread's start
//! state equals what the old kernel held entering rep r, because K0 reproduces the same chain. The
//! histogram atomicAdds are the same set of increments, just issued by more threads — integer add is
//! commutative/associative so the totals are identical. Validated by the GATE_AIR_GPU_TEST=k1 / k4
//! column-by-column + histogram equality harness vs CPU before it is trusted.

use std::sync::{Arc, OnceLock};

// LOWMEM (`GATE_AIR_STREAM_MAIN_LOWMEM`) pool alloc/free for the ~24 GB main-trace buffer.
//
// These are NitrooZK's `size_t`-based pool wrappers (crates/stwo/.../cuda_mem_pool.cu), NOT the
// u32-based `BaseFieldVec::new_zeroes` path. CRITICAL: at 2^25 the buffer is
// `TRACE_COLUMNS * padded_rows = 188 * 2^25 = 6,308,233,216` u32s, which OVERFLOWS the `u32`/`int`
// argument of `cuda_alloc_zeroes_uint32_t` (BaseFieldVec::new_zeroes) — that truncation allocated an
// ~8 GB buffer and the K1 kernel then wrote 24 GB into it → out-of-bounds device write → SIGABRT at
// the K1 sync (exit=134, box-confirmed). `cuda_mem_pool_allocate_zeroes_uint32` takes `size_t`, so
// the full 6.3e9-element (24 GB) request is passed intact. Both alloc + free route through
// NitrooZK's `g_mem_pool` (`cudaMallocFromPoolAsync` / `cudaFreeAsync`) — the same pool tree2 draws
// from — so the freed buffer is directly reusable by tree2. Linked from the stwo cuda static lib.
#[cfg(feature = "gpu-cuda")]
extern "C" {
    fn cuda_mem_pool_allocate_zeroes_uint32(count: usize) -> *mut u32;
    fn cuda_mem_pool_free_uint32(ptr: *mut u32);
}

/// Shared cudarc handle on device-0's PRIMARY CUDA context, cached process-wide. The NitrooZK
/// `CudaBackend` (stwo_cuda) also targets device-0's primary context (CUDA runtime-API default), so
/// device pointers produced by these cudarc K1/K4 kernels interoperate with the backend's commit
/// (the device-to-device bridge). Replaces the obelyzk `get_cuda_executor`. [box-verified]
pub(crate) fn cuda_device() -> Result<Arc<cudarc::driver::CudaDevice>, String> {
    static DEV: OnceLock<Arc<cudarc::driver::CudaDevice>> = OnceLock::new();
    if let Some(d) = DEV.get() {
        return Ok(d.clone());
    }
    let d = cudarc::driver::CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e}"))?;
    let _ = DEV.set(d.clone());
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
fn gate_sim_module(
    dev: &Arc<cudarc::driver::CudaDevice>,
) -> Result<(&'static str, bool), String> {
    use cudarc::nvrtc::compile_ptx;
    static LOADED: OnceLock<Result<(), String>> = OnceLock::new();
    let mut first = false;
    let res = LOADED.get_or_init(|| {
        first = true;
        let ptx = compile_ptx(GATE_SIM_KERNEL).map_err(|e| format!("nvrtc compile: {e}"))?;
        dev.load_ptx(ptx, "gate_sim_mod", &["gate_sim_states", "gate_sim", "fill_padding"])
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
    static LOADED: OnceLock<Result<(), String>> = OnceLock::new();
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
    let res = LOADED.get_or_init(|| {
        first = true;
        let ptx =
            compile_ptx(INTERACTION_KERNEL).map_err(|e| format!("nvrtc compile (K4): {e}"))?;
        dev.load_ptx(ptx, "logup_mod", &names)
            .map_err(|e| format!("load_ptx (K4): {e}"))?;
        Ok(())
    });
    res.clone().map(|()| ("logup_mod", first))
}

/// NVRTC-compiled CUDA source for the gate-sim kernel. Layout/constants mirror gate_air `main.rs`:
/// N_LIMBS=32, LIMB_BITS=16, TRACE_COLUMNS=191, READ_COLS=39, M31 modulus 2^31-1,
/// opcodes NOP=0/NOT=1/CNOT=2/TOFFOLI=3.
///
/// Buffers (all device):
/// - `gates`:   n_gates * 4  (opcode, target_q, ctrl_a_q, ctrl_b_q), u32
/// - `x_states`: n_shots * N_LIMBS  (initial state limbs per shot), u32
/// - `rep_states`: n_shots * k * N_LIMBS — rep-boundary states K0 produces and K1 consumes:
///   rep_states[(shot*k + rep)*N_LIMBS + i] = limb i of the state at the START of (shot, rep).
/// - `off_lo`/`off_hi`: 16 each — RcIndex offsets: off_lo[p]=2^p-1, off_hi[p]=2^16-2^(16-p)
/// - `cols`:    TRACE_COLUMNS * padded_rows, column-major (col c at cols[c*padded_rows + row]), u32
/// - `qdecode`(512), `rc_lo`(65536), `rc_hi`(65536): histograms, u32, zero-initialized
/// Scalars: k, n_gates, n_shots, padded_rows (shot_rows = k*n_gates computed in-kernel).
/// NOTE: caller must zero `cols` + histograms first, and write padding rows (enabler=0, the 3
/// read-block `mask` columns = 1, rest 0) for rows in [n_shots*shot_rows, padded_rows).
/// Launch order: K0 `gate_sim_states` (fills rep_states), then K1 `gate_sim` (consumes it).
pub const GATE_SIM_KERNEL: &str = r#"
#define N_LIMBS 32u
#define LIMB_BITS 16u
#define READ_COLS 39u
#define M31_MOD 2147483647u
#define OP_NOP 0u
#define OP_NOT 1u
#define OP_CNOT 2u
#define OP_TOFFOLI 3u

// Apply one gate's target-bit flip to `limbs` in place. Pure state update (no column/histogram
// I/O) — the part the K1 per-gate body and K0 share. Mirrors simulate_shot's limb math exactly.
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

// K0: thread-per-SHOT, state-only. Walk the full k*n_gates chain (cheap limb updates, NO column
// stores, NO histogram atomics) and snapshot the state at the START of every (shot, rep) into
// rep_states. rep 0's snapshot is exactly x_states[shot]; rep r's is the state after r full passes.
// This keeps total chain work LINEAR in k while letting K1 start each (shot,rep) independently.
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
        // Snapshot the state entering this rep.
        unsigned long base = ((unsigned long)shot * (unsigned long)k + (unsigned long)rep) * N_LIMBS;
        #pragma unroll
        for (unsigned i = 0; i < N_LIMBS; i++) rep_states[base + i] = limbs[i];
        // Advance through one full pass of the program.
        for (unsigned g = 0; g < n_gates; g++) apply_gate(limbs, gates, g);
    }
}

extern "C" __global__ void gate_sim(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ rep_states,
    const unsigned* __restrict__ off_lo,
    const unsigned* __restrict__ off_hi,
    unsigned* __restrict__ cols,
    unsigned* __restrict__ qdecode,
    unsigned* __restrict__ rc_lo,
    unsigned* __restrict__ rc_hi,
    unsigned k,
    unsigned n_gates,
    unsigned n_shots,
    unsigned long padded_rows)
{
    // THREAD-PER-EXECUTION: one thread per (shot, rep). exec = shot*k + rep.
    unsigned long exec = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long n_exec = (unsigned long)n_shots * (unsigned long)k;
    if (exec >= n_exec) return;
    unsigned shot = (unsigned)(exec / (unsigned long)k);
    unsigned rep  = (unsigned)(exec % (unsigned long)k);

    // Per-thread state: the 32-limb register file, seeded from K0's rep-boundary snapshot so this
    // (shot, rep) starts exactly where the old thread-per-shot kernel was when it entered rep.
    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = rep_states[exec * N_LIMBS + i];

    // Row range + pc this execution owns. row(shot,rep,g) = (shot*k + rep)*n_gates + g = exec*n_gates+g.
    // pc is the monotonic per-shot counter the old kernel emitted: pc(rep,g) = rep*n_gates + g.
    unsigned long row = exec * (unsigned long)n_gates;
    unsigned pc = rep * n_gates;

    // Per-read decoded fields (mirror ReadCols).
    unsigned r_active[3], r_q[3], r_limb[3], r_bitpos[3], r_mask[3], r_lo[3], r_hi[3], r_bit[3];

    for (unsigned g = 0; g < n_gates; g++) {
            unsigned opcode   = gates[g * 4u + 0u];
            unsigned tq       = gates[g * 4u + 1u];
            unsigned aq       = gates[g * 4u + 2u];
            unsigned bq       = gates[g * 4u + 3u];

            unsigned is_nop = (opcode == OP_NOP) ? 1u : 0u;
            unsigned is_not = (opcode == OP_NOT) ? 1u : 0u;
            unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
            unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
            unsigned a_active = is_cnot + is_tof;
            unsigned b_active = is_tof;

            // qubits per read slot: 0=target (always active), 1=ctrl_a (a_active), 2=ctrl_b (b_active).
            unsigned qslot[3] = { tq, aq, bq };
            unsigned act[3]   = { 1u, a_active, b_active };

            #pragma unroll
            for (unsigned s = 0; s < 3u; s++) {
                if (act[s]) {
                    unsigned q = qslot[s];
                    unsigned li = q / LIMB_BITS;
                    unsigned bp = q % LIMB_BITS;
                    unsigned l = limbs[li];
                    r_active[s] = 1u; r_q[s] = q; r_limb[s] = li; r_bitpos[s] = bp;
                    r_mask[s] = 1u << bp;
                    r_lo[s] = l & ((1u << bp) - 1u);
                    r_bit[s] = (l >> bp) & 1u;
                    r_hi[s] = l >> (bp + 1u);
                } else {
                    // ReadCols::inactive(): mask=1, everything else 0.
                    r_active[s] = 0u; r_q[s] = 0u; r_limb[s] = 0u; r_bitpos[s] = 0u;
                    r_mask[s] = 1u; r_lo[s] = 0u; r_hi[s] = 0u; r_bit[s] = 0u;
                }
            }

            unsigned t_bit = r_bit[0], a_bit = r_bit[1], b_bit = r_bit[2];
            unsigned ab = a_bit * b_bit;
            unsigned fire = is_not + is_cnot * a_bit + is_tof * ab;   // in {0,1}
            unsigned new_t = t_bit ^ fire;
            int delta_signed = (int)new_t - (int)t_bit;              // {-1,0,1}

            // out_limb = in_limb with the target bit flipped (±mask on the target limb).
            unsigned out_limb[N_LIMBS];
            #pragma unroll
            for (unsigned i = 0; i < N_LIMBS; i++) out_limb[i] = limbs[i];
            if (delta_signed > 0) out_limb[r_limb[0]] += r_mask[0];
            else if (delta_signed < 0) out_limb[r_limb[0]] -= r_mask[0];

            // Lookup multiplicities (count_read for each active read).
            #pragma unroll
            for (unsigned s = 0; s < 3u; s++) {
                if (r_active[s]) {
                    atomicAdd(&qdecode[r_q[s]], 1u);
                    atomicAdd(&rc_lo[off_lo[r_bitpos[s]] + r_lo[s]], 1u);
                    atomicAdd(&rc_hi[off_hi[r_bitpos[s]] + r_hi[s]], 1u);
                }
            }

            // Emit the 188 cells in cell_at order (running column counter cc).
            // enabler / shot_id / pc were moved OUT of the main trace into the preprocessed tree
            // (tree0); they are positional and are NOT written here. (`shot`/`pc` are still computed
            // above to drive the simulation; they just no longer go into the committed trace.)
            unsigned cc = 0u;
            #define EMIT(v) cols[(unsigned long)(cc++) * padded_rows + row] = (v)
            EMIT(is_nop); EMIT(is_not); EMIT(is_cnot); EMIT(is_tof);
            #pragma unroll
            for (unsigned i = 0; i < N_LIMBS; i++) EMIT(limbs[i]);     // in_limb
            #pragma unroll
            for (unsigned i = 0; i < N_LIMBS; i++) EMIT(out_limb[i]);  // out_limb
            // 3 read blocks: q, limb_idx, bit_pos, mask, lsel[32], lo, hi, bit
            #pragma unroll
            for (unsigned s = 0; s < 3u; s++) {
                EMIT(r_q[s]); EMIT(r_limb[s]); EMIT(r_bitpos[s]); EMIT(r_mask[s]);
                #pragma unroll
                for (unsigned i = 0; i < N_LIMBS; i++) EMIT(r_active[s] && (i == r_limb[s]) ? 1u : 0u); // lsel
                EMIT(r_lo[s]); EMIT(r_hi[s]); EMIT(r_bit[s]);
            }
            EMIT(ab); EMIT(fire);
            EMIT(delta_signed >= 0 ? (unsigned)delta_signed : (unsigned)((int)M31_MOD + delta_signed)); // delta_to_m31
            #undef EMIT

            // Advance: state becomes out_limb for the next gate.
            #pragma unroll
            for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = out_limb[i];
            pc += 1u;
            row += 1u;
    }
}

// Populate padding rows [real_rows, padded_rows) to match `Row::padding()`:
// everything 0 except the 3 read-block `mask` columns (target=71, ctrl_a=110, ctrl_b=149),
// which `ReadCols::inactive()` sets to 1. `cols` is pre-zeroed, so only the masks need writing.
// (Indices shifted down by 3 from the old layout after enabler/shot_id/pc moved to tree0.)
extern "C" __global__ void fill_padding(
    unsigned* __restrict__ cols,
    unsigned long padded_rows,
    unsigned long real_rows)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x + real_rows;
    if (row >= padded_rows) return;
    cols[(unsigned long)71u  * padded_rows + row] = 1u;
    cols[(unsigned long)110u * padded_rows + row] = 1u;
    cols[(unsigned long)149u * padded_rows + row] = 1u;
}
"#;

pub const TRACE_COLUMNS: usize = 188;

/// Run K1 on the GPU and copy the trace + histograms back to the host (for the P3.1 byte-identity
/// test). The production GPU-resident path (return BaseColumns, no D2H) is P3.4.
///
/// Inputs (host, already flattened by the caller):
/// - `gates_flat`: n_gates*4 (opcode, target_q, ctrl_a_q, ctrl_b_q); inactive controls -> any value
///   (kernel guards on a_active/b_active).
/// - `x_states`: n_shots*32 initial limbs (state_to_limbs of each shot's x_hex).
/// - `off_lo`/`off_hi`: 16 each (RcIndex offsets).
/// Returns (cols [column-major, TRACE_COLUMNS*padded_rows], qdecode[512], rc_lo[2^16], rc_hi[2^16]).
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

    let d_gates = dev.htod_copy(gates_flat.to_vec()).map_err(|e| format!("htod gates: {e}"))?;
    let d_x = dev.htod_copy(x_states.to_vec()).map_err(|e| format!("htod x_states: {e}"))?;
    let d_off_lo = dev.htod_copy(off_lo.to_vec()).map_err(|e| format!("htod off_lo: {e}"))?;
    let d_off_hi = dev.htod_copy(off_hi.to_vec()).map_err(|e| format!("htod off_hi: {e}"))?;
    let mut d_cols = dev
        .alloc_zeros::<u32>(TRACE_COLUMNS * padded_rows)
        .map_err(|e| format!("alloc cols: {e}"))?;
    let mut d_qd = dev.alloc_zeros::<u32>(512).map_err(|e| format!("alloc qdecode: {e}"))?;
    let mut d_lo = dev.alloc_zeros::<u32>(1 << 16).map_err(|e| format!("alloc rc_lo: {e}"))?;
    let mut d_hi = dev.alloc_zeros::<u32>(1 << 16).map_err(|e| format!("alloc rc_hi: {e}"))?;
    // K0 boundary-state buffer: n_shots * k * N_LIMBS (rep-start state per execution).
    let n_exec = (n_shots as u64) * (k as u64);
    let mut d_rep = dev
        .alloc_zeros::<u32>((n_exec as usize) * crate::N_LIMBS)
        .map_err(|e| format!("alloc rep_states: {e}"))?;

    let block = 256u32;
    // K0: thread-per-shot — fill rep-boundary states.
    let k0_cfg = LaunchConfig {
        grid_dim: (n_shots.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.get_func("gate_sim_mod", "gate_sim_states")
            .ok_or_else(|| "get_func gate_sim_states".to_string())?
            .launch(k0_cfg, (&d_gates, &d_x, &mut d_rep, k, n_gates, n_shots))
            .map_err(|e| format!("launch gate_sim_states: {e}"))?;
    }
    // K1: thread-per-execution — one thread per (shot, rep) = n_shots*k threads.
    let grid = (n_exec.div_ceil(block as u64)) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // 8 device pointers + 4 scalars = 12 args (cudarc launch tuple cap).
    unsafe {
        func.launch(
            cfg,
            (
                &d_gates, &d_rep, &d_off_lo, &d_off_hi, &mut d_cols, &mut d_qd, &mut d_lo, &mut d_hi,
                k, n_gates, n_shots, padded_rows as u64,
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
    let mut lo = vec![0u32; 1 << 16];
    let mut hi = vec![0u32; 1 << 16];
    dev.dtoh_sync_copy_into(&d_cols, &mut cols).map_err(|e| format!("dtoh cols: {e}"))?;
    dev.dtoh_sync_copy_into(&d_qd, &mut qd).map_err(|e| format!("dtoh qdecode: {e}"))?;
    dev.dtoh_sync_copy_into(&d_lo, &mut lo).map_err(|e| format!("dtoh rc_lo: {e}"))?;
    dev.dtoh_sync_copy_into(&d_hi, &mut hi).map_err(|e| format!("dtoh rc_hi: {e}"))?;
    Ok((cols, qd, lo, hi))
}

/// P3.1 soundness gate: assert the GPU K1 trace (real rows + lookup histograms) is BYTE-IDENTICAL
/// to the CPU reference (`build_rows` + `cell_at` + `LookupCounts`). Run on a small fixture (k1-n4)
/// via `GATE_AIR_GPU_TEST=1` (hooked in main). Compares the 191 main columns cell-by-cell over the
/// real rows and the qdecode/rc_lo/rc_hi histograms; padding rows are handled separately (TODO).
#[cfg(all(feature = "gpu-cuda", feature = "diag"))]
pub fn k1_byte_identity(
    gates: &[crate::Gate],
    cases: &[crate::TestCase],
    k: usize,
    rc_lo: &crate::RcIndex,
    rc_hi: &crate::RcIndex,
) -> Result<(), String> {
    let n_gates = gates.len();
    let n_shots = cases.len();
    let real_rows = n_shots * k * n_gates;
    let padded_rows = real_rows
        .next_power_of_two()
        .max(1 << (crate::LOG_N_LANES + 2)); // matches main.rs

    // CPU reference.
    let (rows, counts) =
        crate::build_rows(gates, cases, k, rc_lo, rc_hi).map_err(|e| e.to_string())?;
    if rows.len() != real_rows {
        return Err(format!("rows.len()={} != real_rows={}", rows.len(), real_rows));
    }

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
    let off_lo: Vec<u32> = (0..crate::LIMB_BITS).map(|p| rc_lo.offset[p] as u32).collect();
    let off_hi: Vec<u32> = (0..crate::LIMB_BITS).map(|p| rc_hi.offset[p] as u32).collect();

    // GPU.
    let (cols, qd, lo, hi) = gpu_gen_main_trace(
        &gates_flat,
        &x_states,
        &off_lo,
        &off_hi,
        k as u32,
        n_gates as u32,
        n_shots as u32,
        padded_rows,
    )?;

    // Compare the 191 main columns over the real rows.
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

    // Compare lookup histograms.
    let cmp = |name: &str, g: &[u32], c: &[u32]| -> Option<String> {
        if g.len() < c.len() {
            return Some(format!("{name}: gpu len {} < cpu len {}", g.len(), c.len()));
        }
        (0..c.len())
            .find(|&i| g[i] != c[i])
            .map(|i| format!("{name}[{i}]: gpu={} cpu={}", g[i], c[i]))
    };
    let q_e = cmp("qdecode", &qd, &counts.qdecode);
    let lo_e = cmp("rc_lo", &lo, &counts.rc_lo);
    let hi_e = cmp("rc_hi", &hi, &counts.rc_hi);

    eprintln!(
        "[K1 byte-identity] real_rows={real_rows} padded_rows={padded_rows} col_mismatches={mismatches}"
    );
    for s in &samples {
        eprintln!("    {s}");
    }
    for e in [&q_e, &lo_e, &hi_e].into_iter().flatten() {
        eprintln!("    HIST {e}");
    }

    if mismatches == 0 && q_e.is_none() && lo_e.is_none() && hi_e.is_none() {
        eprintln!("[K1 byte-identity] PASS — GPU trace == CPU trace (real rows + histograms)");
        Ok(())
    } else {
        Err(format!(
            "K1 byte-identity FAILED: {mismatches} column mismatches; hist q={q_e:?} lo={lo_e:?} hi={hi_e:?}"
        ))
    }
}

// ============================================================================
// P3.2 — K4: CUDA LogUp interaction trace (full on-device).
// ============================================================================
//
// Generates gate_air's 24 interaction M31 columns (6 LogUp columns × 4 coords) +
// claimed_sum on the GPU, byte-identical to the CPU `gen_main_interaction` /
// `LogupTraceGenerator`. Consumes K1's main-trace columns (no re-simulation).
//
// Pipeline (per LogUp column k=0..5, sequential — col k accumulates onto col k-1):
//   1. logup_col_gen[pair k]: per row, combine the two relation tuples -> (num, denom)
//      where d = (Σ_i alpha^i · values[i]) − z   (QM31), num = m0·d1 + m1·d0, denom = d0·d1.
//   2. logup_finalize_col: value = num · denom^{-1} (per-element QM31 inverse — byte-identical
//      to the CPU batch inverse, since the field inverse is unique); running sum across columns.
// Then once, on the last column (k=5):
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
// SOUNDNESS: validated by `k4_byte_identity` (24 cols + claimed_sum) vs the CPU reference
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

/// Number of LogUp columns (pairs) gate_air emits. Each is a SecureColumnByCoords (4 M31).
pub const N_LOGUP_COLS: usize = 6;
/// Number of M31 interaction columns committed = 6 × 4.
pub const N_INTERACTION_COLS: usize = N_LOGUP_COLS * 4;
/// GateRel width (= `relation!(GateRel, 35)`): number of alpha powers uploaded.
pub const GATE_REL_WIDTH: usize = 35;

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

// K4a: per-row (num, denom) for one of the 6 pairs. Reads K1's main-trace columns
// (column-major in `cols`). Outputs interleaved per row: num[row*4+j], denom[row*4+j].
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

    unsigned v0[35]; int n0 = 0;
    unsigned v1[35]; int n1 = 0;
    unsigned m0 = 0u, m1 = 0u; int s0 = 1, s1 = 1;

    // enabler / shot_id / pc moved to the preprocessed tree (tree0). They are POSITIONAL, so K4
    // recomputes them from `row` — identical values to the old in-trace columns (enabler = real-row
    // indicator, shot_id = row / shot_stride, pc = row % shot_stride). All-column indices below
    // shifted down by 3 (header is now just the 4 opcode one-hots at COL 0..3).
    unsigned enabler  = (row < real_rows) ? 1u : 0u;
    unsigned shot_id  = (row < real_rows) ? (unsigned)(row / shot_stride) : 0u;
    unsigned pc       = (row < real_rows) ? (unsigned)(row % shot_stride) : 0u;
    unsigned a_active = m31_add(COL(2), COL(3));   // is_cnot + is_toffoli
    unsigned b_active = COL(3);                     // is_toffoli

    switch (pair_id) {
    case 0: // state_in (+enabler), state_out (-enabler)
        v0[0]=1u; v0[1]=shot_id; v0[2]=pc;
        for (int i=0;i<32;i++) v0[3+i]=COL(4+i);
        n0=35;
        v1[0]=1u; v1[1]=shot_id; v1[2]=pc+1u;
        for (int i=0;i<32;i++) v1[3+i]=COL(36+i);
        n1=35;
        m0=enabler; s0=1; m1=enabler; s1=-1; break;
    case 1: // qdecode target (+enabler), qdecode ctrl_a (+a_active)
        v0[0]=2u; v0[1]=COL(68);  v0[2]=COL(69);  v0[3]=COL(70);  v0[4]=COL(71);  n0=5;
        v1[0]=2u; v1[1]=COL(107); v1[2]=COL(108); v1[3]=COL(109); v1[4]=COL(110); n1=5;
        m0=enabler; m1=a_active; break;
    case 2: // qdecode ctrl_b (+b_active), rc_lo target (+enabler)
        v0[0]=2u; v0[1]=COL(146); v0[2]=COL(147); v0[3]=COL(148); v0[4]=COL(149); n0=5;
        v1[0]=3u; v1[1]=COL(70);  v1[2]=COL(104); n1=3;
        m0=b_active; m1=enabler; break;
    case 3: // rc_hi target (+enabler), rc_lo ctrl_a (+a_active)
        v0[0]=4u; v0[1]=COL(70);  v0[2]=COL(105); n0=3;
        v1[0]=3u; v1[1]=COL(109); v1[2]=COL(143); n1=3;
        m0=enabler; m1=a_active; break;
    case 4: // rc_hi ctrl_a (+a_active), rc_lo ctrl_b (+b_active)
        v0[0]=4u; v0[1]=COL(109); v0[2]=COL(144); n0=3;
        v1[0]=3u; v1[1]=COL(148); v1[2]=COL(182); n1=3;
        m0=a_active; m1=b_active; break;
    case 5: // rc_hi ctrl_b (+b_active), program (+enabler)
        v0[0]=4u; v0[1]=COL(148); v0[2]=COL(183); n0=3;
        v1[0]=5u; v1[1]=(unsigned)((unsigned long)pc % (unsigned long)n_gates);
        v1[2]=m31_add(m31_add(COL(1), m31_mul(2u,COL(2))), m31_mul(3u,COL(3))); // opcode_scalar
        v1[3]=COL(68); v1[4]=COL(107); v1[5]=COL(146); n1=6;
        m0=b_active; m1=enabler; break;
    }

    qm31 d0 = logup_combine(v0, n0, z0,z1,z2,z3, ap);
    qm31 d1 = logup_combine(v1, n1, z0,z1,z2,z3, ap);
    unsigned mm0 = (s0 < 0) ? m31_neg(m0) : m0;
    unsigned mm1 = (s1 < 0) ? m31_neg(m1) : m1;
    qm31 qm0 = { {mm0,0u}, {0u,0u} };
    qm31 qm1 = { {mm1,0u}, {0u,0u} };
    qm31 nume = qm31_add(qm31_mul(qm0, d1), qm31_mul(qm1, d0));
    qm31 den  = qm31_mul(d0, d1);

    num[row*4+0]=nume.a.a; num[row*4+1]=nume.a.b; num[row*4+2]=nume.b.a; num[row*4+3]=nume.b.b;
    denom[row*4+0]=den.a.a; denom[row*4+1]=den.a.b; denom[row*4+2]=den.b.a; denom[row*4+3]=den.b.b;
    #undef COL
}

// K4b: value = num · denom^{-1}; running sum onto previous logup column.
// `inter` holds the 24 interaction columns, column-major (logup col k coord j = (k*4+j)).
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

/// Run the full K4 interaction pipeline on the GPU and copy the 24 interaction columns +
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
    let d_cols = dev.htod_copy(cols.to_vec()).map_err(|e| format!("htod cols: {e}"))?;
    let mut ap_flat = Vec::with_capacity(GATE_REL_WIDTH * 4);
    for p in alpha_powers {
        ap_flat.extend_from_slice(p);
    }
    let d_ap = dev.htod_copy(ap_flat).map_err(|e| format!("htod ap: {e}"))?;
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
                        z[0], z[1], z[2], z[3],
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
    let mut d_sums = dev.alloc_zeros::<u32>(4).map_err(|e| format!("alloc sums: {e}"))?;
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
                (padded_rows as u64, last_k, padded_rows as u32, &d_sums, &mut d_inter),
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
        prefix_sum_column(&dev, &mut d_inter, offset, padded_rows, bits, &mut d_tmp, &get)?;
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

/// P3.2 soundness gate: assert the GPU K4 interaction trace (24 M31 columns + claimed_sum) is
/// byte-identical to the CPU `gen_main_interaction` / `LogupTraceGenerator`, using a FIXED
/// `GateRel::dummy()` (z, alpha) so both sides see the same challenges. Run via
/// `GATE_AIR_GPU_TEST=k4` on a small fixture (k1-n4).
#[cfg(all(feature = "gpu-cuda", feature = "diag"))]
pub fn k4_byte_identity(
    gates: &[crate::Gate],
    cases: &[crate::TestCase],
    k: usize,
    rc_lo: &crate::RcIndex,
    rc_hi: &crate::RcIndex,
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
    let (rows, _counts) =
        crate::build_rows(gates, cases, k, rc_lo, rc_hi).map_err(|e| e.to_string())?;
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
    let off_lo: Vec<u32> = (0..crate::LIMB_BITS).map(|p| rc_lo.offset[p] as u32).collect();
    let off_hi: Vec<u32> = (0..crate::LIMB_BITS).map(|p| rc_hi.offset[p] as u32).collect();
    let (main_cols, _qd, _lo, _hi) = gpu_gen_main_trace(
        &gates_flat, &x_states, &off_lo, &off_hi,
        k as u32, n_gates as u32, n_shots as u32, padded_rows,
    )?;

    // Extract (z, alpha_powers) from the relation via the PUBLIC `Relation::combine`
    // (the inner LookupElements is private to constraint-framework). Works for any
    // challenges (dummy here, real-drawn in P3.4): combine(values) = Σ α^i·v[i] − z, so
    //   combine([0])      = −z                  → z      = −combine([0])
    //   combine(unit_i)   = α^i − z             → α^i    = combine(unit_i) + z
    let (z_qm, alpha_powers_qm) = extract_z_alpha(&elements.state);
    let z = secure_to_m31x4(z_qm);
    let alpha_powers: Vec<[u32; 4]> =
        alpha_powers_qm.iter().map(|p| secure_to_m31x4(*p)).collect();

    let (gpu_inter, gpu_sum_arr) = gpu_gen_interaction(
        &main_cols,
        z,
        &alpha_powers,
        padded_rows,
        n_gates as u32,
        real_rows as u64,
        (k * n_gates) as u64,
    )?;

    // Compare the 24 interaction columns over ALL padded rows (prefix sum spans them).
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
        eprintln!("[K4 byte-identity] PASS — GPU interaction == CPU interaction (24 cols + claimed_sum)");
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

/// Fix (b) (`GATE_AIR_FUSED_INTERP`): wrap ONE column (`padded_rows` u32s at element offset
/// `col_off`) of `src` as a BORROWED (non-owning) `CircleEvaluation<CudaBackend>` — NO device
/// allocation, NO copy. The returned column aliases `src` (== K1's `d_cols`); `src` MUST outlive it
/// (it does: `d_cols` is held by the caller through tree1 commit + K4). Drop will NOT free `src`,
/// and the commit's per-column interpolate copies each view into a reused temp before the in-place
/// b2n, so `d_cols` is only READ here and stays intact for K4 reuse. This is the memory-saving
/// replacement for `d2d_column` under the flag: it removes the second full main-trace resident copy.
#[cfg(feature = "gpu-cuda")]
fn borrowed_column(
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

    // SAFETY: `src` lives on the device-0 primary context; the [col_off, col_off+padded_rows) span
    // is in-bounds (column-major layout). The view is read-only downstream and does not own `src`.
    let src_ptr = unsafe { cudarc_dptr(src).add(col_off) };
    let view = BaseFieldVec::from_borrowed_ptr(src_ptr, padded_rows);
    CircleEvaluation::<_, _, BitReversedOrder>::new(domain, view)
}

/// Device-resident K1: run `gpu_gen_main_trace`'s kernels and return the 191 main
/// columns as `CircleEvaluation<CudaBackend>` (device-resident, no host upload),
/// plus the qdecode/rc_lo/rc_hi histograms copied to the host (tiny; the
/// multiplicity columns are still built + uploaded on the CPU path in main.rs),
/// plus the raw column-major main-trace device buffer `d_cols` so the interaction
/// path (K4) can REUSE it instead of re-running K0/K1 (no re-simulation, no
/// re-upload). The 191 returned `CircleEvaluation`s are independent D2D copies of
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
    let d_gates = dev.htod_copy(gates_flat.to_vec()).map_err(|e| format!("htod gates: {e}"))?;
    let d_off_lo = dev.htod_copy(off_lo.to_vec()).map_err(|e| format!("htod off_lo: {e}"))?;
    let d_off_hi = dev.htod_copy(off_hi.to_vec()).map_err(|e| format!("htod off_hi: {e}"))?;
    gpu_gen_main_trace_device_d(
        &d_gates, x_states, &d_off_lo, &d_off_hi, k, n_gates, n_shots, padded_rows, log_n_rows,
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

    let d_x = dev.htod_copy(x_states.to_vec()).map_err(|e| format!("htod x_states: {e}"))?;
    // LOWMEM device-OOM fix (approach a): allocate the ~24 GB main-trace buffer from NitrooZK's
    // `cudaMemPool_t` (`BaseFieldVec::new_zeroes` -> `cudaMallocFromPoolAsync`), NOT cudarc's
    // `cuMemAlloc`. That is the SAME pool tree2's per-column `cudaMallocFromPoolAsync` draws from, so
    // when `free_after_k4` frees this buffer (`cuda_free_memory` -> `cudaFreeAsync` into the pool) the
    // 24 GB lands back on the pool free-list and tree2 can immediately reuse it. Freeing via cudarc's
    // `cuMemFree` (the default `alloc_zeros` path) instead returns the block to the DRIVER, which the
    // pool cannot reuse -> tree2 OOMs (box-confirmed). We wrap the pool pointer as a cudarc `CudaSlice`
    // (`upgrade_device_ptr`) purely so the K1/K4 kernel launches can address it, and `from_k1` /
    // `gpu_gen_interaction_device` `leak()` those wrappers so cudarc NEVER frees the pool pointer (the
    // owning `PooledBuf` inside `MainTrace::ResidentPooled` is the sole owner + pool-frees on drop).
    // Flag OFF: the exact previous cudarc `alloc_zeros` path (byte-for-byte).
    let lowmem = stream_main_lowmem_enabled();
    let cols_len = TRACE_COLUMNS * padded_rows;
    let d_cols = if lowmem {
        // FAIL-FAST pool alloc via the size_t-safe wrapper (see the extern block up top). Panics/aborts
        // are replaced by an explicit error so a box run reports the precise failing step, not SIGABRT.
        let raw = unsafe { cuda_mem_pool_allocate_zeroes_uint32(cols_len) };
        if raw.is_null() {
            return Err(format!(
                "[LOWMEM] cuda_mem_pool_allocate_zeroes_uint32 returned NULL for the {}-u32 (~{} GiB) \
                 main-trace buffer at 2^{} — pool could not grow. Aborting before K1.",
                cols_len,
                (cols_len as u64 * 4) >> 30,
                log_n_rows
            ));
        }
        // Wrap the pool pointer as a cudarc `CudaSlice` (`upgrade_device_ptr`) so the K1/K4 kernel
        // launches can address it. This cudarc wrapper is ALWAYS `leak()`ed (never dropped) — cudarc's
        // Drop would `cuMemFree` a POOL pointer, which is invalid → abort. The sole owner/freer is the
        // pool free in `free_after_k4` (`cuda_mem_pool_free_uint32`), called exactly once.
        // SAFETY: `raw` is a valid `cols_len`-u32 pool allocation on device-0's primary context (shared
        // with cudarc), zero-initialized on stream 0 (ordered before the K1 launch on the same stream).
        unsafe { dev.upgrade_device_ptr::<u32>(raw as cudarc::driver::sys::CUdeviceptr, cols_len) }
    } else {
        dev.alloc_zeros::<u32>(cols_len)
            .map_err(|e| format!("alloc cols: {e}"))?
    };
    // LOWMEM raw pool pointer, captured so the SUCCESS path (end of this fn) can `leak()` the cudarc
    // wrapper and re-wrap this same pointer into a fresh leaked `CudaSlice` for the caller — so cudarc's
    // Drop (which would `cuMemFree` a pool pointer and abort) NEVER runs on it. `None` on the flag-off
    // cudarc path (its `d_cols` frees correctly via cudarc `cuMemFree`).
    // ERROR-PATH SAFETY (LOWMEM): capture the pool pointer, then wrap `d_cols` in `ManuallyDrop` so
    // cudarc's Drop NEVER `cuMemFree`s the POOL pointer (which aborts). On the SUCCESS path (end of
    // this fn) we extract the pointer and hand back a fresh cudarc slice; on ANY early `?` return (a
    // CUDA/launch fault below) the `ManuallyDrop` is simply forgotten — the buffer leaks on that
    // already-failed run instead of aborting, so the real error surfaces. Flag OFF: identical handling
    // (the buffer is a normal cudarc allocation; we still extract it on success and it leaks only on a
    // failed run — no behavior change for successful proofs).
    let lowmem_raw: Option<*mut u32> = if lowmem {
        use cudarc::driver::DevicePtr;
        Some((*d_cols.device_ptr()) as usize as *mut u32)
    } else {
        None
    };
    let mut d_cols = std::mem::ManuallyDrop::new(d_cols);
    let mut d_qd = dev.alloc_zeros::<u32>(512).map_err(|e| format!("alloc qdecode: {e}"))?;
    let mut d_lo = dev.alloc_zeros::<u32>(1 << 16).map_err(|e| format!("alloc rc_lo: {e}"))?;
    let mut d_hi = dev.alloc_zeros::<u32>(1 << 16).map_err(|e| format!("alloc rc_hi: {e}"))?;
    let n_exec = (n_shots as u64) * (k as u64);
    let mut d_rep = dev
        .alloc_zeros::<u32>((n_exec as usize) * crate::N_LIMBS)
        .map_err(|e| format!("alloc rep_states: {e}"))?;

    let block = 256u32;
    // K0: thread-per-shot — fill rep-boundary states (cheap, state-only chain).
    let k0_cfg = LaunchConfig {
        grid_dim: (n_shots.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        dev.get_func("gate_sim_mod", "gate_sim_states")
            .ok_or_else(|| "get_func gate_sim_states".to_string())?
            .launch(k0_cfg, (d_gates, &d_x, &mut d_rep, k, n_gates, n_shots))
            .map_err(|e| format!("launch gate_sim_states: {e}"))?;
    }
    // K1: thread-per-execution — n_shots*k threads.
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
                d_gates, &d_rep, d_off_lo, d_off_hi, &mut *d_cols, &mut d_qd, &mut d_lo, &mut d_hi,
                k, n_gates, n_shots, padded_rows as u64,
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
            fill.launch(pad_cfg, (&mut *d_cols, padded_rows as u64, real_rows))
                .map_err(|e| format!("launch fill_padding: {e}"))?;
        }
    }
    dev.synchronize().map_err(|e| format!("sync: {e}"))?;

    // Device handoff: build the 188 CudaBackend columns from `d_cols` (column-major; column c at
    // element offset c*padded_rows). No host copy.
    //
    // Fix (b) (`GATE_AIR_FUSED_INTERP`): when set, build BORROWED views into `d_cols` (zero extra
    // device memory) instead of 188 D2D copies. This drops the second full main-trace resident copy
    // so a 2^25 base proof fits a 40 GB A100. The views hold UN-interpolated base-domain evals; the
    // CudaBackend commit's fused per-column-interpolate path (poly.rs) interpolates each into a
    // reused temp (never touching `d_cols`), so K4's reuse of `d_cols` stays correct. When unset,
    // the legacy D2D-copy path is byte-for-byte unchanged.
    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    let interp_in_commit = std::env::var("GATE_AIR_FUSED_INTERP").is_ok();
    let cols: Vec<_> = (0..TRACE_COLUMNS)
        .map(|c| {
            if interp_in_commit {
                borrowed_column(&d_cols, c * padded_rows, padded_rows, domain)
            } else {
                d2d_column(&d_cols, c * padded_rows, padded_rows, domain)
            }
        })
        .collect();

    // Histograms are tiny → keep them on host (the multiplicity columns are built
    // + uploaded on the existing CPU path in main.rs).
    let mut qd = vec![0u32; 512];
    let mut lo = vec![0u32; 1 << 16];
    let mut hi = vec![0u32; 1 << 16];
    dev.dtoh_sync_copy_into(&d_qd, &mut qd).map_err(|e| format!("dtoh qdecode: {e}"))?;
    dev.dtoh_sync_copy_into(&d_lo, &mut lo).map_err(|e| format!("dtoh rc_lo: {e}"))?;
    dev.dtoh_sync_copy_into(&d_hi, &mut hi).map_err(|e| format!("dtoh rc_hi: {e}"))?;
    dev.synchronize().map_err(|e| format!("sync hist: {e}"))?;
    // SUCCESS handoff. `d_cols` is `ManuallyDrop`, so we must extract the inner `CudaSlice` to return
    // it (otherwise the buffer would leak). K4 reads it directly (no K0/K1 re-run); it is no longer
    // touched here after the column build above.
    //
    // LOWMEM: the inner slice wraps a POOL pointer that cudarc must NEVER `cuMemFree` (abort). We take
    // the raw pointer out and hand back a FRESH cudarc wrapper over it — identical bytes, and `from_k1`
    // re-`leak()`s it into a `PooledBuf` (the sole owner that pool-frees). The original `ManuallyDrop`
    // is forgotten (never dropped), so no cudarc `cuMemFree` ever hits the pool pointer.
    if let Some(raw) = lowmem_raw {
        // SAFETY: `raw` is the same valid `cols_len`-u32 pool allocation, still live. The ManuallyDrop
        // `d_cols` is left un-dropped (forgotten), so the pointer has exactly one live wrapper again.
        let fresh = unsafe {
            dev.upgrade_device_ptr::<u32>(raw as cudarc::driver::sys::CUdeviceptr, cols_len)
        };
        Ok((cols, qd, lo, hi, fresh))
    } else {
        // Flag OFF: extract the owning cudarc slice from ManuallyDrop and return it (frees via cudarc
        // `cuMemFree` when the caller eventually drops it, exactly as before this fix).
        let inner = unsafe { std::mem::ManuallyDrop::take(&mut d_cols) };
        Ok((cols, qd, lo, hi, inner))
    }
}

/// Reads `GATE_AIR_STREAM_MAIN` (main-trace DEVICE-capacity fix, opt-in, DEFAULT OFF). When set, the
/// gate_air driver DEHYDRATES K1's column-major main-trace device buffer (`d_cols`, 188 columns ×
/// `padded_rows` — ~24 GB at 2^25) to the HOST right after the tree1 commit and FREES the device
/// buffer, so it is no longer resident during the K4 interaction-scratch allocations (`d_inter` etc.)
/// or the tree2 commit — the two phases that OOM at 2^25 on a 40 GB A100 with the buffer pinned.
/// `gpu_gen_interaction_device` then REHYDRATES the buffer (H2D) only for the duration of its kernel
/// loop (after the small scratch allocs, so the peak is scratch + one rehydrated copy, ~27 GB) and
/// FREES it again before returning, so the pin never overlaps tree2.
///
/// Mirrors the tree1 EVAL streaming (fused_commit stash) at the leaf/`cudarc` layer: only the byte
/// SOURCE of the main trace moves (device → host `Vec<u32>` → device); the committed values and the
/// K4 interaction columns / `claimed_sum` are byte-identical. Composes with (does not require)
/// `GATE_AIR_BOUNDARY_TRIM` (option-0). The host copy is PAGEABLE (`dtoh_sync_copy_into` /
/// `htod_copy`), independent of `GATE_AIR_ASYNC_STASH` (which pins the SEPARATE tree1 eval stash),
/// so the ~72 GB host high-water at 2^25 stays PAGEABLE and does not consume pinnable memory.
#[cfg(feature = "gpu-cuda")]
pub fn stream_main_enabled() -> bool {
    std::env::var("GATE_AIR_STREAM_MAIN").is_ok()
}

/// Reads `GATE_AIR_STREAM_MAIN_LOWMEM` (HOST-memory peak fix, opt-in, DEFAULT OFF). Attacks the
/// ~72 GB host high-water at 2^25 that OOM-kills the leaf on the ~85 GB box, which is the SUM of two
/// full-size host copies of the main trace that coexist through K4:
///   * the ~48 GB tree1 EVAL STASH (188 LDE columns on the 2^26 domain, held on host from tree1
///     commit through FRI decommit — needed by OODS/quotient/build_leaves/decommit), and
///   * the ~24 GB DEHYDRATED d_main host `Vec` that plain `GATE_AIR_STREAM_MAIN` creates (a SECOND
///     host copy of the base-domain main trace, made at the tree1->K4 boundary and consumed by K4).
///
/// Under LOWMEM we KEEP K1's `d_cols` RESIDENT on the DEVICE across K4 (device has headroom once the
/// eval stash has streamed tree1's evals off the device: `d_cols` 24 GB + K4 scratch ~3 GB < 40 GB)
/// instead of dehydrating it to a duplicate host `Vec`, so that ~24 GB host copy NEVER EXISTS. The
/// caller then FREES the device buffer immediately after K4 (before tree2 — the phase that would
/// OOM the DEVICE with `d_cols` still pinned), so `d_cols` resident never overlaps tree2. Net host
/// peak at 2^25 falls to ~48 GB (the eval stash alone). BYTE-IDENTICAL: K4 reads the exact resident
/// `d_cols` K1 produced (the `Resident` path), so the interaction columns / `claimed_sum` are
/// bit-for-bit the flag-off result. Composes with `GATE_AIR_STREAM_COMMIT`/`GATE_AIR_FUSED_INTERP`
/// (the eval-stash streaming that frees tree1 evals off the device — REQUIRED for the device to hold
/// `d_cols` through K4) and with `GATE_AIR_ASYNC_STASH`. Takes precedence over `GATE_AIR_STREAM_MAIN`
/// (whose host dehydrate is exactly the copy this removes).
#[cfg(feature = "gpu-cuda")]
pub fn stream_main_lowmem_enabled() -> bool {
    std::env::var("GATE_AIR_STREAM_MAIN_LOWMEM").is_ok()
}

/// The K1 main-trace buffer as consumed by K4: either RESIDENT on the device (the default — the
/// `CudaSlice` K1 returned, held live through tree1 + K4) or DEHYDRATED to a host `Vec<u32>` (under
/// `GATE_AIR_STREAM_MAIN`, so the ~24 GB device buffer is freed across the K4-alloc / tree2 phases).
/// `gpu_gen_interaction_device` resolves either into a live device buffer for its kernel loop.
/// LOWMEM sole owner of the ~24 GB pool-allocated main-trace buffer. Holds the raw pool `*mut u32`
/// (from `cuda_mem_pool_allocate_zeroes_uint32`) and frees it EXACTLY ONCE via
/// `cuda_mem_pool_free_uint32` (→ `cudaFreeAsync` into `g_mem_pool`) on Drop — the pool tree2 draws
/// from, so the freed buffer is directly reusable. The cudarc `CudaSlice` wrappers used by K1/K4 are
/// always leaked (never dropped), so this is the one and only owner; there is no cudarc `cuMemFree`
/// on the pool pointer (which would abort).
#[cfg(feature = "gpu-cuda")]
pub struct PooledBuf {
    ptr: *mut u32,
    len: usize,
}
#[cfg(feature = "gpu-cuda")]
unsafe impl Send for PooledBuf {}
#[cfg(feature = "gpu-cuda")]
unsafe impl Sync for PooledBuf {}
#[cfg(feature = "gpu-cuda")]
impl Drop for PooledBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Pool free (size-safe: takes only the pointer). Returns the block to `g_mem_pool`.
            unsafe { cuda_mem_pool_free_uint32(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(feature = "gpu-cuda")]
pub enum MainTrace {
    /// The K1 device buffer, still resident (flag OFF). K4 reads it in place; nothing is freed here.
    /// cudarc-owned (`cuMemAlloc`/`cuMemFree`).
    Resident(cudarc::driver::CudaSlice<u32>),
    /// LOWMEM (`GATE_AIR_STREAM_MAIN_LOWMEM`): the K1 device buffer allocated from NitrooZK's
    /// `cudaMemPool_t` and owned by a `PooledBuf` so `free_after_k4` returns it to that pool
    /// (`cudaFreeAsync`) — the pool tree2 allocates from, so the freed 24 GB is directly reusable by
    /// tree2 (unlike a cudarc `cuMemFree`, which returns the block to the driver, not the pool). K4
    /// reads it in place via a leaked cudarc wrapper; the `PooledBuf` stays the sole owner.
    ResidentPooled(PooledBuf),
    /// The K1 main trace D2H-copied to the host, its device buffer freed (flag ON,
    /// `GATE_AIR_STREAM_MAIN`). K4 rehydrates a fresh device copy for its kernel loop, then frees it.
    Dehydrated(Vec<u32>),
}

#[cfg(feature = "gpu-cuda")]
impl MainTrace {
    /// Wrap K1's device buffer, DEHYDRATING it to the host + freeing the device buffer when
    /// `GATE_AIR_STREAM_MAIN` is set, else keeping it resident. Call AFTER the tree1 commit (which
    /// borrows into the resident buffer) and BEFORE the K4 interaction allocs / tree2 commit.
    pub fn from_k1(d_cols: cudarc::driver::CudaSlice<u32>) -> Result<Self, String> {
        // LOWMEM (host-peak + device-OOM fix): keep `d_cols` RESIDENT on the device across K4 rather
        // than dehydrating a duplicate ~24 GB host `Vec`. `d_cols` is a cudarc wrapper over a NitrooZK
        // POOL allocation (see `gpu_gen_main_trace_device_d`); `leak()` it (cudarc forgets the pointer,
        // never `cuMemFree`s it — that would abort on a pool pointer) and re-own the raw pointer as a
        // `PooledBuf`, whose Drop (`free_after_k4`) pool-frees via `cuda_mem_pool_free_uint32` ->
        // `cudaFreeAsync`, the pool tree2 reuses. Takes precedence over `GATE_AIR_STREAM_MAIN`.
        if stream_main_lowmem_enabled() {
            let len = {
                use cudarc::driver::DeviceSlice;
                d_cols.len()
            };
            let ptr = d_cols.leak() as *mut u32; // cudarc no longer owns/frees this pool pointer
            return Ok(MainTrace::ResidentPooled(PooledBuf { ptr, len }));
        }
        if stream_main_enabled() {
            use cudarc::driver::DeviceSlice;
            let dev = cuda_device()?;
            let mut host = vec![0u32; d_cols.len()];
            // PAGEABLE D2H of the whole main trace, then drop the CudaSlice to free the device
            // buffer (~24 GB at 2^25). The eval columns are byte-identical to the resident buffer.
            dev.dtoh_sync_copy_into(&d_cols, &mut host)
                .map_err(|e| format!("dtoh stream-main dehydrate: {e}"))?;
            drop(d_cols); // free the ~24 GB device buffer — no longer resident for K4-alloc / tree2
            Ok(MainTrace::Dehydrated(host))
        } else {
            Ok(MainTrace::Resident(d_cols))
        }
    }

    /// Free the underlying device/host main-trace buffer NOW, after K4 has consumed it and BEFORE the
    /// tree2 commit. Call this instead of relying on `drop` at end-of-prove.
    ///
    /// LOWMEM device-OOM fix: under `GATE_AIR_STREAM_MAIN_LOWMEM` the `ResidentPooled` variant OWNS the
    /// sole K1 `d_cols` buffer as a `PooledBuf` allocated from NitrooZK's `cudaMemPool_t` (~24 GB at
    /// 2^25, kept resident across K4). Consuming `self` here drops that `PooledBuf`, whose Drop calls
    /// `cuda_mem_pool_free_uint32` -> `cudaFreeAsync` INTO the pool, so the freed 24 GB lands on the
    /// pool free-list and tree2's `cudaMallocFromPoolAsync` can immediately reuse it. (An earlier
    /// attempt that used cudarc's `cuMemFree` returned the block to the DRIVER, which the pool cannot
    /// reuse -> tree2 OOMed; box-confirmed.) We then SYNCHRONIZE the device so the deferred
    /// `cudaFreeAsync` completes and the block is on the pool free-list BEFORE tree2's first alloc.
    /// BYTE-IDENTICAL: the free happens only AFTER K4 has read the resident buffer, so the interaction
    /// columns / `claimed_sum` are unaffected. `Dehydrated` frees the host `Vec`; the default flag-off
    /// `Resident` `cuMemFree`s the cudarc buffer (unchanged from before this fix).
    pub fn free_after_k4(self) -> Result<(), String> {
        // `self` is consumed here: `ResidentPooled`'s `PooledBuf` (pool free), `Resident`'s
        // `CudaSlice` (`cuMemFree`), or `Dehydrated`'s host `Vec`, is dropped at the end of this scope.
        drop(self);
        // Settle the free before tree2's first pool alloc so the freed 24 GB is on the pool free-list.
        let dev = cuda_device()?;
        dev.synchronize().map_err(|e| format!("sync after main-trace free: {e}"))?;
        Ok(())
    }
}

/// Device-resident K4: run `gpu_gen_interaction`'s pipeline using the REAL drawn
/// `LookupElements` (z/alpha recovered via `extract_z_alpha`) and return the 24
/// interaction columns as `CircleEvaluation<CudaBackend>` (device-resident, no host
/// upload) plus the `claimed_sum` (mixed into the channel in main.rs exactly as the
/// CPU `gen_main_interaction` sum was).
///
/// `main` is the SAME column-major main trace K1 produced earlier in this base proof — either the
/// resident device buffer (default) or, under `GATE_AIR_STREAM_MAIN`, the host-dehydrated copy that
/// K4 rehydrates here (H2D into a fresh device buffer AFTER the small scratch allocs, so the peak is
/// scratch + one rehydrated copy; the buffer is freed again before returning so the pin never
/// overlaps tree2). Either way K4 reads it via `COL(c)` — no re-simulation (K0/K1 are NOT re-run).
/// The buffer is only read here (the K4 `logup_col_gen` kernel reads it and writes its own
/// `d_num`/`d_denom`/`d_inter` scratch), so the interaction columns + `claimed_sum` are
/// byte-identical to the resident path. Reuses the EXACT validated `INTERACTION_KERNEL` source.
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
    let (z_qm, alpha_powers_qm) = extract_z_alpha(&elements.state);
    let z = secure_to_m31x4(z_qm);
    let alpha_powers: Vec<[u32; 4]> =
        alpha_powers_qm.iter().map(|p| secure_to_m31x4(*p)).collect();
    assert_eq!(alpha_powers.len(), GATE_REL_WIDTH, "alpha_powers width");

    let dev = cuda_device()?;

    // K4 reads the main-trace columns from the buffer K1 already produced (`main`, passed in by the
    // caller). No K0/K1 re-run, no re-upload of gates/x_states/off_lo/off_hi, no histogram/rep
    // scratch — those were only needed to repopulate the main trace, which now lives in `main`.
    // Under `GATE_AIR_STREAM_MAIN` the trace is host-dehydrated; it is REHYDRATED below, AFTER the
    // small interaction-scratch allocs, so the device peak is scratch + one rehydrated main copy.
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
    let d_ap = dev.htod_copy(ap_flat).map_err(|e| format!("htod ap: {e}"))?;
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

    // Resolve the main trace into a live device buffer for the kernel loop. Resident: borrow K1's
    // buffer in place (byte-for-byte the previous behavior). Dehydrated (`GATE_AIR_STREAM_MAIN`):
    // REHYDRATE a fresh device copy via H2D NOW — after the small scratch allocs above (`d_inter` /
    // `d_num` / `d_denom`, ~3 GB), so those never contended with the ~24 GB main buffer that was
    // freed at the tree1->K4 boundary. The rehydrated buffer holds the exact committed main-trace
    // bytes, so `COL(c)` reads and the resulting interaction columns are byte-identical. `_rehydrated`
    // OWNS the fresh buffer: it is dropped (freed) at end of scope, BEFORE tree2 commits, so the
    // ~24 GB pin never overlaps tree2.
    // LOWMEM `ResidentPooled`: build a NON-OWNING cudarc `CudaSlice` view over the pool pointer so the
    // K4 launches can address it, then `leak()` it (below) so cudarc never frees the pool buffer — the
    // `PooledBuf` inside `MainTrace::ResidentPooled` stays the sole owner (freed to the pool by
    // `free_after_k4`). The view reads the exact resident bytes K1 produced, so K4 output is
    // byte-identical to the `Resident` path.
    let _pooled_view: Option<cudarc::driver::CudaSlice<u32>> = match main {
        MainTrace::ResidentPooled(pb) => Some(unsafe {
            // SAFETY: `pb.ptr` is a valid `pb.len`-u32 pool allocation on the shared primary context,
            // live for the whole K4 call (owned by `main`, dropped only after K4 returns).
            dev.upgrade_device_ptr::<u32>(
                pb.ptr as cudarc::driver::sys::CUdeviceptr,
                pb.len,
            )
        }),
        _ => None,
    };
    let _rehydrated: Option<cudarc::driver::CudaSlice<u32>> = match main {
        MainTrace::Dehydrated(host) => Some(
            // `htod_sync_copy` copies from the `&[u32]` slice directly (no host-side clone of the
            // ~24 GB buffer), synchronously, into a fresh device allocation.
            dev.htod_sync_copy(&host[..])
                .map_err(|e| format!("htod stream-main rehydrate: {e}"))?,
        ),
        _ => None,
    };
    let d_cols: &cudarc::driver::CudaSlice<u32> = match main {
        MainTrace::Resident(d) => d,
        MainTrace::ResidentPooled(_) => _pooled_view.as_ref().unwrap(),
        MainTrace::Dehydrated(_) => _rehydrated.as_ref().unwrap(),
    };

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
                        z[0], z[1], z[2], z[3],
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
                .launch(cfg, (kk, padded_rows as u64, &d_num, &d_denom, &mut d_inter))
                .map_err(|e| format!("launch finalize[{kk}]: {e}"))?;
        }
    }

    let last_k = (N_LOGUP_COLS - 1) as u32;
    let mut d_sums = dev.alloc_zeros::<u32>(4).map_err(|e| format!("alloc sums: {e}"))?;
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
                (padded_rows as u64, last_k, padded_rows as u32, &d_sums, &mut d_inter),
            )
            .map_err(|e| format!("launch cumsum_shift: {e}"))?;
    }

    let bits = padded_rows.trailing_zeros();
    let mut d_tmp = dev
        .alloc_zeros::<u32>(padded_rows)
        .map_err(|e| format!("alloc ps tmp: {e}"))?;
    for j in 0..4u64 {
        let offset = ((last_k as u64) * 4 + j) * padded_rows as u64;
        prefix_sum_column(&dev, &mut d_inter, offset, padded_rows, bits, &mut d_tmp, &get)?;
    }

    dev.synchronize().map_err(|e| format!("sync (K4): {e}"))?;

    // LOWMEM: the K4 kernels are done reading `d_cols`. `_pooled_view` is a cudarc `CudaSlice` built
    // via `upgrade_device_ptr` over the POOL-owned buffer; `leak()` it so cudarc's Drop does NOT
    // `cuMemFree` the pool pointer (the `PooledBuf` in `MainTrace::ResidentPooled` is the sole owner
    // and pool-frees it in `free_after_k4`). Without this leak we'd double-free the pool buffer.
    if let Some(view) = _pooled_view {
        let _ = view.leak();
    }

    // Device-to-device handoff: 24 interaction columns straight to CudaBackend.
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
