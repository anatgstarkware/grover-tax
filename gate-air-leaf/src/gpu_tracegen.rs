//! P3.1 — K1: CUDA gate-sim + main-trace kernel for gate_air (on-device "model B").
//!
//! Generates the 191 main-trace columns + the 3 lookup histograms (qdecode / rc_lo / rc_hi)
//! ENTIRELY on the GPU, so the trace never leaves device memory (avoiding the model-A transfer
//! wall). Thread-per-shot: each of the `n_shots` threads runs its shot's full `k * n_gates`
//! sequential gate chain, threading the 32-limb state, and writes its rows directly into
//! device-resident column-major M31 columns.
//!
//! SOUNDNESS: this kernel must produce a trace BYTE-IDENTICAL to the CPU `build_rows` +
//! `generate_main_trace`. It is a line-for-line translation of `simulate_shot` (per-gate compute),
//! `ReadCols::live`/`inactive`, `qubit_decode`, `count_read` (histogram indices), `delta_to_m31`,
//! and the `cell_at` 191-column layout. Validated by a column-by-column M31 equality test vs CPU
//! before it is trusted. No security parameters change — this only moves WHERE the witness is built.
//!
//! Status: kernel source complete (this file). Launch glue (cudarc, sharing stwo's CUDA executor +
//! producing GpuBackend-resident columns) + the byte-identity test are the next steps (see TODOs).

use std::sync::{Arc, OnceLock};

/// Shared cudarc handle on device-0's PRIMARY CUDA context, cached process-wide. The NitrooZK
/// `CudaBackend` (stwo_cuda) also targets device-0's primary context (CUDA runtime-API default), so
/// device pointers produced by these cudarc K1/K4 kernels interoperate with the backend's commit
/// (the device-to-device bridge). Replaces the obelyzk `get_cuda_executor`. [box-verified]
fn cuda_device() -> Result<Arc<cudarc::driver::CudaDevice>, String> {
    static DEV: OnceLock<Arc<cudarc::driver::CudaDevice>> = OnceLock::new();
    if let Some(d) = DEV.get() {
        return Ok(d.clone());
    }
    let d = cudarc::driver::CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e}"))?;
    let _ = DEV.set(d.clone());
    Ok(d)
}

/// NVRTC-compiled CUDA source for the gate-sim kernel. Layout/constants mirror gate_air `main.rs`:
/// N_LIMBS=32, LIMB_BITS=16, TRACE_COLUMNS=191, READ_COLS=39, M31 modulus 2^31-1,
/// opcodes NOP=0/NOT=1/CNOT=2/TOFFOLI=3.
///
/// Buffers (all device):
/// - `gates`:   n_gates * 4  (opcode, target_q, ctrl_a_q, ctrl_b_q), u32
/// - `x_states`: n_shots * N_LIMBS  (initial state limbs per shot), u32
/// - `off_lo`/`off_hi`: 16 each — RcIndex offsets: off_lo[p]=2^p-1, off_hi[p]=2^16-2^(16-p)
/// - `cols`:    TRACE_COLUMNS * padded_rows, column-major (col c at cols[c*padded_rows + row]), u32
/// - `qdecode`(512), `rc_lo`(65536), `rc_hi`(65536): histograms, u32, zero-initialized
/// Scalars: k, n_gates, n_shots, padded_rows (shot_rows = k*n_gates computed in-kernel).
/// NOTE: caller must zero `cols` + histograms first, and write padding rows (enabler=0, the 3
/// read-block `mask` columns = 1, rest 0) for rows in [n_shots*shot_rows, padded_rows).
pub const GATE_SIM_KERNEL: &str = r#"
#define N_LIMBS 32u
#define LIMB_BITS 16u
#define READ_COLS 39u
#define M31_MOD 2147483647u
#define OP_NOP 0u
#define OP_NOT 1u
#define OP_CNOT 2u
#define OP_TOFFOLI 3u

extern "C" __global__ void gate_sim(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ x_states,
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
    unsigned shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= n_shots) return;
    unsigned long shot_rows = (unsigned long)k * (unsigned long)n_gates;

    // Per-thread state: the 32-limb register file, threaded across the whole shot.
    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = x_states[shot * N_LIMBS + i];

    unsigned long row = (unsigned long)shot * shot_rows;
    unsigned pc = 0u;

    // Per-read decoded fields (mirror ReadCols).
    unsigned r_active[3], r_q[3], r_limb[3], r_bitpos[3], r_mask[3], r_lo[3], r_hi[3], r_bit[3];

    for (unsigned rep = 0; rep < k; rep++) {
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

            // Emit the 191 cells in cell_at order (running column counter cc).
            unsigned cc = 0u;
            #define EMIT(v) cols[(unsigned long)(cc++) * padded_rows + row] = (v)
            EMIT(1u);                 // enabler
            EMIT(is_nop); EMIT(is_not); EMIT(is_cnot); EMIT(is_tof);
            EMIT(shot);               // shot_id
            EMIT(pc);
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
}

// Populate padding rows [real_rows, padded_rows) to match `Row::padding()`:
// everything 0 except the 3 read-block `mask` columns (target=74, ctrl_a=113, ctrl_b=152),
// which `ReadCols::inactive()` sets to 1. `cols` is pre-zeroed, so only the masks need writing.
extern "C" __global__ void fill_padding(
    unsigned* __restrict__ cols,
    unsigned long padded_rows,
    unsigned long real_rows)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x + real_rows;
    if (row >= padded_rows) return;
    cols[(unsigned long)74u  * padded_rows + row] = 1u;
    cols[(unsigned long)113u * padded_rows + row] = 1u;
    cols[(unsigned long)152u * padded_rows + row] = 1u;
}
"#;

pub const TRACE_COLUMNS: usize = 191;

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
    use cudarc::nvrtc::compile_ptx;

    let dev = cuda_device()?;

    let ptx = compile_ptx(GATE_SIM_KERNEL).map_err(|e| format!("nvrtc compile: {e}"))?;
    dev.load_ptx(ptx, "gate_sim_mod", &["gate_sim", "fill_padding"])
        .map_err(|e| format!("load_ptx: {e}"))?;
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

    let block = 256u32;
    let grid = n_shots.div_ceil(block);
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
                &d_gates, &d_x, &d_off_lo, &d_off_hi, &mut d_cols, &mut d_qd, &mut d_lo, &mut d_hi,
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
#[cfg(feature = "gpu-cuda")]
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
    unsigned* __restrict__ num,
    unsigned* __restrict__ denom)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    #define COL(c) cols[(unsigned long)(c) * padded_rows + row]

    unsigned v0[35]; int n0 = 0;
    unsigned v1[35]; int n1 = 0;
    unsigned m0 = 0u, m1 = 0u; int s0 = 1, s1 = 1;

    unsigned enabler  = COL(0);
    unsigned a_active = m31_add(COL(3), COL(4));   // is_cnot + is_toffoli
    unsigned b_active = COL(4);                     // is_toffoli

    switch (pair_id) {
    case 0: // state_in (+enabler), state_out (-enabler)
        v0[0]=1u; v0[1]=COL(5); v0[2]=COL(6);
        for (int i=0;i<32;i++) v0[3+i]=COL(7+i);
        n0=35;
        v1[0]=1u; v1[1]=COL(5); v1[2]=COL(6)+1u;
        for (int i=0;i<32;i++) v1[3+i]=COL(39+i);
        n1=35;
        m0=enabler; s0=1; m1=enabler; s1=-1; break;
    case 1: // qdecode target (+enabler), qdecode ctrl_a (+a_active)
        v0[0]=2u; v0[1]=COL(71);  v0[2]=COL(72);  v0[3]=COL(73);  v0[4]=COL(74);  n0=5;
        v1[0]=2u; v1[1]=COL(110); v1[2]=COL(111); v1[3]=COL(112); v1[4]=COL(113); n1=5;
        m0=enabler; m1=a_active; break;
    case 2: // qdecode ctrl_b (+b_active), rc_lo target (+enabler)
        v0[0]=2u; v0[1]=COL(149); v0[2]=COL(150); v0[3]=COL(151); v0[4]=COL(152); n0=5;
        v1[0]=3u; v1[1]=COL(73);  v1[2]=COL(107); n1=3;
        m0=b_active; m1=enabler; break;
    case 3: // rc_hi target (+enabler), rc_lo ctrl_a (+a_active)
        v0[0]=4u; v0[1]=COL(73);  v0[2]=COL(108); n0=3;
        v1[0]=3u; v1[1]=COL(112); v1[2]=COL(146); n1=3;
        m0=enabler; m1=a_active; break;
    case 4: // rc_hi ctrl_a (+a_active), rc_lo ctrl_b (+b_active)
        v0[0]=4u; v0[1]=COL(112); v0[2]=COL(147); n0=3;
        v1[0]=3u; v1[1]=COL(151); v1[2]=COL(185); n1=3;
        m0=a_active; m1=b_active; break;
    case 5: // rc_hi ctrl_b (+b_active), program (+enabler)
        v0[0]=4u; v0[1]=COL(151); v0[2]=COL(186); n0=3;
        v1[0]=5u; v1[1]=(unsigned)((unsigned long)COL(6) % (unsigned long)n_gates);
        v1[2]=m31_add(m31_add(COL(2), m31_mul(2u,COL(3))), m31_mul(3u,COL(4))); // opcode_scalar
        v1[3]=COL(71); v1[4]=COL(110); v1[5]=COL(149); n1=6;
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
pub fn gpu_gen_interaction(
    cols: &[u32],
    z: [u32; 4],
    alpha_powers: &[[u32; 4]],
    padded_rows: usize,
    n_gates: u32,
) -> Result<(Vec<u32>, [u32; 4]), String> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;

    assert_eq!(alpha_powers.len(), GATE_REL_WIDTH, "alpha_powers width");
    assert!(padded_rows.is_power_of_two(), "padded_rows must be 2^k");

    let dev = cuda_device()?;

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
    let ptx = compile_ptx(INTERACTION_KERNEL).map_err(|e| format!("nvrtc compile (K4): {e}"))?;
    dev.load_ptx(ptx, "logup_mod", &names)
        .map_err(|e| format!("load_ptx (K4): {e}"))?;
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
#[cfg(feature = "gpu-cuda")]
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

    let (gpu_inter, gpu_sum_arr) =
        gpu_gen_interaction(&main_cols, z, &alpha_powers, padded_rows, n_gates as u32)?;

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

/// Device-resident K1: run `gpu_gen_main_trace`'s kernels and return the 191 main
/// columns as `CircleEvaluation<CudaBackend>` (device-resident, no host upload),
/// plus the qdecode/rc_lo/rc_hi histograms copied to the host (tiny; the
/// multiplicity columns are still built + uploaded on the CPU path in main.rs).
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
    ),
    String,
> {
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use stwo::core::poly::circle::CanonicCoset;

    let dev = cuda_device()?;

    let ptx = compile_ptx(GATE_SIM_KERNEL).map_err(|e| format!("nvrtc compile: {e}"))?;
    dev.load_ptx(ptx, "gate_sim_mod", &["gate_sim", "fill_padding"])
        .map_err(|e| format!("load_ptx: {e}"))?;
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

    let block = 256u32;
    let grid = n_shots.div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &d_gates, &d_x, &d_off_lo, &d_off_hi, &mut d_cols, &mut d_qd, &mut d_lo, &mut d_hi,
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

    // Device-to-device handoff: build the 191 CudaBackend columns from `d_cols`
    // (column-major; column c at element offset c*padded_rows). No host copy.
    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    let cols: Vec<_> = (0..TRACE_COLUMNS)
        .map(|c| d2d_column(&d_cols, c * padded_rows, padded_rows, domain))
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
    Ok((cols, qd, lo, hi))
}

/// Device-resident K4: run `gpu_gen_interaction`'s pipeline using the REAL drawn
/// `LookupElements` (z/alpha recovered via `extract_z_alpha`) and return the 24
/// interaction columns as `CircleEvaluation<CudaBackend>` (device-resident, no host
/// upload) plus the `claimed_sum` (mixed into the channel in main.rs exactly as the
/// CPU `gen_main_interaction` sum was).
///
/// `d_main` is the same column-major main-trace device buffer K1 produced; we keep
/// it on-device and re-run K4 against it (no re-simulation, no upload). Reuses the
/// EXACT validated `INTERACTION_KERNEL` source.
#[cfg(feature = "gpu-cuda")]
pub fn gpu_gen_interaction_device(
    gates_flat: &[u32],
    x_states: &[u32],
    off_lo: &[u32],
    off_hi: &[u32],
    k: u32,
    n_gates: u32,
    n_shots: u32,
    padded_rows: usize,
    log_n_rows: u32,
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
    use cudarc::nvrtc::compile_ptx;
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

    // Re-run K1's kernels to produce the main-trace device buffer (column-major),
    // kept entirely on-device for K4 to consume. (Same buffer the device main path
    // uses; reproduced here so K4 owns a cudarc buffer it can read with `COL`.)
    let mtx = compile_ptx(GATE_SIM_KERNEL).map_err(|e| format!("nvrtc compile (K1 for K4): {e}"))?;
    dev.load_ptx(mtx, "gate_sim_mod", &["gate_sim", "fill_padding"])
        .map_err(|e| format!("load_ptx (K1 for K4): {e}"))?;
    let gate_sim = dev
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
    let block = 256u32;
    let grid = n_shots.div_ceil(block);
    let cfg_shots = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        gate_sim
            .launch(
                cfg_shots,
                (
                    &d_gates, &d_x, &d_off_lo, &d_off_hi, &mut d_cols, &mut d_qd, &mut d_lo,
                    &mut d_hi, k, n_gates, n_shots, padded_rows as u64,
                ),
            )
            .map_err(|e| format!("launch gate_sim (for K4): {e}"))?;
    }
    let real_rows = (n_shots as u64) * (k as u64) * (n_gates as u64);
    let n_pad = (padded_rows as u64).saturating_sub(real_rows);
    if n_pad > 0 {
        let fill = dev
            .get_func("gate_sim_mod", "fill_padding")
            .ok_or_else(|| "get_func fill_padding".to_string())?;
        let pad_grid = (n_pad as u32).div_ceil(256);
        let pad_cfg = LaunchConfig {
            grid_dim: (pad_grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            fill.launch(pad_cfg, (&mut d_cols, padded_rows as u64, real_rows))
                .map_err(|e| format!("launch fill_padding (for K4): {e}"))?;
        }
    }
    dev.synchronize().map_err(|e| format!("sync (K1 for K4): {e}"))?;

    // ---- K4 interaction pipeline (verbatim launches from gpu_gen_interaction) ----
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
    let ptx = compile_ptx(INTERACTION_KERNEL).map_err(|e| format!("nvrtc compile (K4): {e}"))?;
    dev.load_ptx(ptx, "logup_mod", &names)
        .map_err(|e| format!("load_ptx (K4): {e}"))?;
    let get = |n: &str| {
        dev.get_func("logup_mod", n)
            .ok_or_else(|| format!("get_func {n}"))
    };

    let mut ap_flat = Vec::with_capacity(GATE_REL_WIDTH * 4);
    for p in &alpha_powers {
        ap_flat.extend_from_slice(p);
    }
    let d_ap = dev.htod_copy(ap_flat).map_err(|e| format!("htod ap: {e}"))?;

    let mut d_inter = dev
        .alloc_zeros::<u32>(N_INTERACTION_COLS * padded_rows)
        .map_err(|e| format!("alloc inter: {e}"))?;
    let mut d_num = dev
        .alloc_zeros::<u32>(4 * padded_rows)
        .map_err(|e| format!("alloc num: {e}"))?;
    let mut d_denom = dev
        .alloc_zeros::<u32>(4 * padded_rows)
        .map_err(|e| format!("alloc denom: {e}"))?;

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
                        &d_cols,
                        padded_rows as u64,
                        z[0], z[1], z[2], z[3],
                        &d_ap,
                        kk,
                        n_gates,
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
