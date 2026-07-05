#!/usr/bin/env python3
"""Tanuj-benchmark curve for the OPTIMIZED gate_air GPU pipeline: a2-highgpu-1g -> 8g, vs SP1 8xA100.

This SUPERSEDES extrapolate.py's pipeline-bottleneck/eta model. That model assumed the GPU prove was
fast but the CPU bucket (trace-gen + lifted-Merkle) starved it -> throughput was min(GPU_8g, CPU_8g)
with a contention range. After the trace-gen GPU port (K0/K1 device kernels) + the D2H scoped-copy
fix, the WHOLE base proof (device trace-gen + CUDA constraints + commit) is fast and MEASURED end to
end on one A100 -> we no longer model a CPU bucket separately. The base proof is now a single measured
per-shard anchor t_base(shard), and the workload is SHARD-PARALLEL across the 8 GPUs.

MODEL (shard-parallel, per the session methodology):
    total_rows(k) = 22.984e6 * k                 # iadd256 x 9024 shots, per rep k
    shard_real_rows = largest measured shard that fits 40GB A100 (target ~2^25 padded)
    N_shards(k)     = ceil(total_rows(k) / shard_real_rows)

    1 GPU (sequential base, sequential recursion):
        time_1g(k) = N_shards * t_base
                   + (2*N_shards) * (t_leaf + t_node) / REC_CONCURRENCY   # ~2N recursion proofs
                   + t_root
      (On 1 GPU base proofs are sequential; recursion is CPU SimdBackend and can pool a little.)

    8 GPUs (base + leaves spread over the GPUs/host cores):
        par = min(8, N_shards)
        time_8g(k) = (N_shards*t_base + 2*N_shards*(t_leaf+t_node)) / par
                   + t_root                                                # serial root tail (Amdahl)
      KEY SHAPE: the 8x speedup only materializes once N_shards >= 8. At low k a few-shard (even
      1-shard) workload cannot use all 8 GPUs -> time_8g ~ time_1g -> SP1's structural edge at low k.

CAVEATS (all flagged): the GPU anchors are MEASURED on ONE A100; the 8x is a PROJECTION assuming
perfect shard independence (separate HBM per A100). Real 8g loses some to: host-feed/PCIe bandwidth
when 8 GPUs pull witnesses at once, the 40GB-per-GPU memory ceiling (sets max shard), and the SERIAL
root tail (Amdahl cap -> 8g never beats t_root at any k). Recursion stays CPU SimdBackend.
"""

# ============================ MEASURED GPU BASE ANCHORS (a2-highgpu-1g, 1x A100-40GB) ============
# MEASURED 2026-06-29 on anat-ganor-instance (A100-SXM4-40GB, 12 vCPU, 83GB host). Each: padded
# shard 2^L, samples S on the k1000 fixture (S shots * 1000 reps * 2547 gates ~ 2^L real rows),
# FULL GPU path (device trace-gen K0/K1 + CUDA_GPU_CONSTRAINTS=1 + D2H scoped-copy fix).
# t_base = trace_gen_s + prove_s (trace_gen spans build_rows..commit; prove_s = final prove_ex).
# Proof byte-identity CONFIRMED at 2^22: fingerprint 0448237288...  (matches the golden CPU oracle).
#
# *** GPU MEMORY CEILING = 2^23 on a 40GB A100. 2^24 OOMs *** (rfft.cu pool fails to alloc 128MB
# scratch after the 191-col eval domain + twiddles + STORED polynomial coefficients, which the base
# proof keeps for the in-circuit-verifier aux). This is BELOW the CPU sweet spot (2^25-2^27) — the
# device-resident witness + 2x blowup eval + coeff store don't fit 40GB past 2^23. So the curve uses
# the LARGEST shard that fits: 2^23.
#
#   shard_log | samples | trace_gen_s | prove_s | t_base | real_rows | note
ANCHORS = {
    22: dict(samples=1, trace_gen_s=4.492, prove_s=1.861, real_rows=2_547_000, note="MEASURED, fp 0448237288 OK"),
    23: dict(samples=2, trace_gen_s=5.669, prove_s=1.890, real_rows=5_094_000, note="MEASURED — largest that FITS 40GB"),
    24: dict(samples=4, trace_gen_s=None,  prove_s=None,  real_rows=10_188_000, note="OOM @40GB (ceiling is 2^23)"),
}

# Which measured shard to USE for the curve = largest that fit the 40GB A100.
USE_SHARD_LOG = 23

# --- recursion anchors (CPU SimdBackend, secure recursion config blowup3/nq23/pow27) ---
# REUSED from project_session_goal_tanuj_curve.md (secure-base validated, ~size-independent: the
# multiverifier node ~2^21 dominates). Re-measure on the box CPU if time permits.
T_LEAF = 2.9     # s  (secure-config CPU leaf, ~flat 2^22..2^27)
T_NODE = 3.7     # s
T_ROOT = 2.0     # s
REC_CONCURRENCY = 3.0   # CPU pool concurrency for the ~2N recursion proofs (measured ~3x on the VM)
RECURSION_MEASURED = "reused-from-memory (secure CPU anchors; 8g host=96 vCPU matches the VM these came from)"

# NOTE on t_base structure (from the two anchors): trace_gen has a ~3.3s FLAT per-shard floor
# (shot sim + CUDA setup) that does NOT scale with rows; the MARGINAL trace_gen rate is ~2.16
# Mreal-rows/s and prove_ex is ~flat ~1.9s (NTT/FRI/commit-bound at these sizes). So t_base is
# dominated by FIXED per-shard cost, and the 2^23 GPU ceiling forces MANY shards -> we pay that
# floor N_shards times. With a bigger shard (if it fit) t_base/row would drop sharply. This makes
# the curve CONSERVATIVE: a fixture packing more shots/shard (better sim parallelism) or a >40GB
# GPU (H100-80GB lets shard -> 2^25, ~4x fewer shards) would close much of the gap.

# --- benchmark / SP1 reference ---
ROWS_PER_K  = 22.984e6
K_POINTS    = [1, 10, 50, 100, 500, 1000, 2000]
SP1_8xA100  = {1:18.5, 10:30.6, 50:88.0, 100:157.0, 500:705.0, 1000:1387.0, 2000:2753.0}
# ===============================================================================================
import math


def fmt(s):
    if s < 90:   return f"{s:.0f}s"
    if s < 5400: return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def t_base_for(shard_log):
    a = ANCHORS[shard_log]
    if a["trace_gen_s"] is None:
        raise SystemExit(f"shard 2^{shard_log} anchor not measured yet — fill ANCHORS")
    return a["trace_gen_s"] + a["prove_s"]


def main():
    a = ANCHORS[USE_SHARD_LOG]
    t_base = t_base_for(USE_SHARD_LOG)
    shard_real = a["real_rows"] if a["real_rows"] else (1 << USE_SHARD_LOG)
    padded = 1 << USE_SHARD_LOG

    print("=" * 78)
    print("Tanuj curve — OPTIMIZED gate_air GPU pipeline (shard-parallel), vs SP1 8xA100")
    print("=" * 78)
    print(f"shard            : 2^{USE_SHARD_LOG} padded ({padded:,}); real rows/shard {shard_real:,}")
    print(f"t_base (MEASURED): trace_gen {a['trace_gen_s']:.2f}s + prove {a['prove_s']:.2f}s "
          f"= {t_base:.2f}s  ({shard_real/t_base/1e6:.2f} Mreal-rows/s, "
          f"{padded/t_base/1e6:.2f} Mpadded-rows/s)")
    print(f"recursion        : leaf {T_LEAF}s node {T_NODE}s root {T_ROOT}s "
          f"[{RECURSION_MEASURED}], rec_concurrency {REC_CONCURRENCY}")
    print("-" * 78)
    print(f"  {'k':>5} | {'N_sh':>6} | {'ours_1g':>9} | {'ours_8g':>9} | {'SP1_8g':>9} | {'ratio_8g':>8}")
    rows = []
    for k in K_POINTS:
        total = ROWS_PER_K * k
        n = max(1, math.ceil(total / shard_real))
        rec_total = 2 * n * (T_LEAF + T_NODE)
        t1g = n * t_base + rec_total / REC_CONCURRENCY + T_ROOT
        par = min(8, n)
        t8g = (n * t_base + rec_total) / par + T_ROOT
        sp1 = SP1_8xA100[k]
        ratio = t8g / sp1
        rows.append((k, n, t1g, t8g, sp1, ratio))
        print(f"  {k:>5} | {n:>6} | {fmt(t1g):>9} | {fmt(t8g):>9} | {fmt(sp1):>9} | {ratio:>7.2f}x")
    print("-" * 78)
    beats = [k for (k, n, t1, t8, sp1, r) in rows if r < 1.0]
    print(f"ours_8g BEATS SP1 at k in: {beats if beats else 'none'}")
    worst = max(rows, key=lambda r: r[5])
    print(f"widest gap: k={worst[0]} -> {worst[5]:.2f}x behind SP1")
    print("SHAPE: at low k, N_shards < 8 -> 8g cannot fill all GPUs -> time_8g ~ time_1g "
          "(SP1's structural edge); 8x only kicks in once N_shards >= 8.")
    print("FLAGS: t_base MEASURED (1 A100); 8x is PROJECTED (perfect shard independence); "
          "recursion REUSED-from-memory; root tail serial = Amdahl cap.")


if __name__ == "__main__":
    main()
