#!/usr/bin/env python3
"""Extrapolate a2-highgpu-1g measurements -> a2-highgpu-8g curve (1g-only, no 2g calibration).

The a2-1g IS Cascade Lake = the SAME CPU microarch as the 8g host -> measuring the CPU bucket on
the 1g needs NO microarch discount (this is why dropping the 2g is OK). The only thing the 1g can't
give is how 8 concurrent CPU slices contend for the shared host memory bus -> we BOUND that with an
efficiency range eta and report a RANGE.

Model = PIPELINE BOTTLENECK (see EXTRAPOLATION_PLAN.md):
    GPU_8g = 8 * shard_rows / t_gpu_pure / 1e6              # 8 independent A100s (firm)
    CPU_8g in [ 8*r_slice*ETA_LOW , 8*r_slice*ETA_HIGH ]    # 8 Cascade-Lake slices, contention eta
    throughput_8g = min(GPU_8g, CPU_8g)   (per eta end)
    T(k) = total_rows(k) / (throughput_8g*1e6) + tree_tail
    total_rows(k) = 22.984e6 * k                            # iadd256 x 9024 shots, per rep k
"""

# ============================ INPUTS ============================
SHARD_LOG   = 24                 # rows/shard = 2^SHARD_LOG. MEASURED cap: 191 cols x 2^25 x4B x2(eval)
SHARD_ROWS  = 1 << SHARD_LOG     #   ~51GB > 40GB A100 -> 2^25 OOMs; 2^24 fits. So shard = 2^24.

# --- GPU part: MEASURED on the 1g A100 (NitrooZK wide_fib width=191, full prove, 2026-06-23) ---
# 2^20=0.133s, 2^22=0.480s, 2^24=1.83s. Upper bound for obelyzk (NitrooZK does merkle ON GPU;
# obelyzk lifted-merkle is on CPU -> its GPU time would be LESS, that merkle moves to r_slice).
T_GPU_PURE  = 1.83               # [s/shard] at 2^24, A100
GPU_MEASURED = True

# --- CPU part: MEASURED on the 1g (REAL Cascade Lake, 12 vCPU, 2026-06-23) ---
# 2^24 phases: sim4.6+preproc1.3+main8.0+interaction1.9 = 15.7s trace-gen (0.65 Mrows/s real);
# commits(NTT+merkle) 11.7s; prove_ex 16.7s. CPU bucket = trace-gen + lifted-merkle(CPU) part of
# commits (NTT->GPU). r_slice central ~0.50 (range 0.37 if commits stay CPU .. 0.65 trace-gen only).
R_SLICE     = 0.50               # [Mrows/s] per-12-vCPU CPU bucket (trace-gen + lifted-merkle)
CPU_MEASURED = True
# eta = efficiency of 8 concurrent slices vs 8x a single (memory-bus contention on the shared host).
#   MEASURED 1g contention probe (1x12c vs 2x6c) = 0.695 at 2 jobs -> 8 jobs likely worse -> 0.3 floor.
ETA_LOW     = 0.30             # 8-slice heavy saturation (DDR4 bus) — pessimistic
ETA_HIGH    = 0.695           # measured 2-job probe (optimistic ceiling for 8 jobs)

# --- recursion tail (fold-tree top levels + root wrapper; ~const for large N) ---
TREE_TAIL   = 6.0

# --- benchmark ---
ROWS_PER_K  = 22.984e6
K_POINTS    = [1, 10, 50, 100, 500, 1000, 2000]
SP1_8xA100  = {1:18.5, 10:30.6, 50:88.0, 100:157.0, 500:705.0, 1000:1387.0, 2000:2753.0}
# ===============================================================


def fmt(s):
    if s < 90:   return f"{s:.0f}s"
    if s < 5400: return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def curve(thru, label):
    print(f"  [{label}]  throughput {thru:.2f} Mrows/s")
    print(f"  {'k':>5} | {'T(8g)':>9} | {'SP1':>9} | ratio")
    for k in K_POINTS:
        t = ROWS_PER_K * k / (thru * 1e6) + TREE_TAIL
        sp1 = SP1_8xA100.get(k)
        print(f"  {k:>5} | {fmt(t):>9} | {fmt(sp1):>9} | {t/sp1:>4.1f}x")


def main():
    gpu_8g  = 8 * SHARD_ROWS / T_GPU_PURE / 1e6
    cpu_lo  = 8 * R_SLICE * ETA_LOW
    cpu_hi  = 8 * R_SLICE * ETA_HIGH
    thru_lo = min(gpu_8g, cpu_lo)        # worst case (lowest throughput)
    thru_hi = min(gpu_8g, cpu_hi)        # best case

    print("=" * 60)
    print("a2-highgpu-8g projection (1g-only, pipeline-bottleneck, RANGE)")
    print("=" * 60)
    print(f"shard           : 2^{SHARD_LOG} = {SHARD_ROWS:,} rows")
    print(f"GPU_8g (8xA100) : {gpu_8g:8.1f} Mrows/s  (t_gpu_pure={T_GPU_PURE}s)")
    print(f"CPU_8g (8 slice): {cpu_lo:6.2f} - {cpu_hi:.2f} Mrows/s  "
          f"(r_slice={R_SLICE} x8 x eta[{ETA_LOW}-{ETA_HIGH}])")
    binder_lo = "CPU" if cpu_lo <= gpu_8g else "GPU"
    binder_hi = "CPU" if cpu_hi <= gpu_8g else "GPU"
    print(f"bottleneck      : {binder_lo} (worst) .. {binder_hi} (best)")
    print("-" * 60)
    curve(thru_hi, "BEST  eta=%.2f" % ETA_HIGH)
    print("-" * 60)
    curve(thru_lo, "WORST eta=%.2f" % ETA_LOW)
    print("-" * 60)
    if binder_lo == binder_hi == "CPU":
        print("ROBUST: CPU-bound across the whole eta range -> 8 GPUs starve.")
        print("        The lever is trace-gen + (GPU) lifted-Merkle, NOT GPU speed.")
    for w, ok in [("t_gpu_pure", GPU_MEASURED), ("r_slice", CPU_MEASURED)]:
        if not ok: print(f"  [!] {w} is a PLACEHOLDER (measure on a2-1g)")
    print("  [!] eta range is bounded, not measured (no 2g) -> the T(8g) SPREAD is the residual risk")


if __name__ == "__main__":
    main()
