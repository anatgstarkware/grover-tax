#!/usr/bin/env python3
"""1g -> a2-highgpu-8g extrapolation, RECURSION-BOUND regime (post-#14, GPU base is fast).

Supersedes the latency formula in box-ops/curve.py, which was invalid because it (1) SUMMED base +
recursion instead of OVERLAPPING them, (2) capped recursion parallelism at the GPU count (min(8,N)),
and (3) ignored CPU memory-contention scaling. This script uses the TWO BALANCES:

  Balance A (on-machine overlap): base(GPU) || leaf-gen(CPU) || tree-fold(CPU) run concurrently in the
    streaming pipeline; makespan = max of the streams (NOT a sum). On 12 vCPU the best recursion split
    was 2 pools x 6 threads => P_1g = 2 concurrent recursion pools.
  Balance B (1g->8g scaling): machine = exactly 8x 1g. GPU base throughput scales ~x8 (8 A100s);
    CPU recursion-pool count scales 8 * P_1g but is memory-bandwidth limited => discount by eta in
    [0.30, 0.695] (measured contention probe). throughput binds on whichever resource is slower.

Pipeline makespan over N shards:
    base_wall = ceil(N/8) * t_base                          # 8 A100s, firm x8
    P_8g(eta) = 8 * P_1g * eta                              # concurrent recursion pools on 96 vCPU
    rec_wall  = N * (t_leaf + t_node) / P_8g                # N leaf-wraps + ~N node-folds over pools
    T(k)      = max(base_wall, rec_wall) + tail             # overlap; tail = top-of-tree + root (latency)
"""
import math

# ============================ INPUTS ============================
SHARD_LOG = 23
SHARD_ROWS = 1 << SHARD_LOG
ROWS_PER_K = 22.984e6
T_BASE = 6.29                     # s/shard, GPU base @2^23, MEASURED #14 (precompute amortized)

# Recursion anchors — MEASURED by the stwo-vm 96-vCPU sweep (box-ops/stwo-vm/SWEEP_RESULTS.md,
# 2026-07-01). KEY LESSON: P_8g and per-fold latency are COUPLED by memory bandwidth — you cannot
# combine the optimal pool count (K=16) with the LOW-CONTENTION per-fold times (#10's 6.0/6.2, taken
# at K=2). At the throughput-optimal K=16/T=6 split, the sweep measured t_leaf=15.8s, t_node=10.7s on
# c4 (DDR5) — ~2.6x the low-contention values, because 16 concurrent folds saturate the memory bus.
# So the earlier "2.2x best case" (16 pools x 6.0/6.2) was internally inconsistent / too optimistic.
# c4/DDR5 is OPTIMISTIC for the a2-8g host (Cascade Lake + DDR4, less bandwidth) -> add a DDR4 scenario.
ANCHORS = {
    "c4 sweep @K=16 (DDR5 — optimistic for a2)": dict(t_leaf=15.8, t_node=10.7),
    "a2 DDR4-adjusted (~1.5x saturated per-fold, EST — bounded, not measured)": dict(t_leaf=23.7, t_node=16.0),
}
TAIL = 6.0                        # top-of-tree levels + root wrapper (latency-bound, ~const for large N)

P_1G = 2                          # Balance A: best recursion pools on 12 vCPU (2x6) — superseded by the sweep
ETA = {"best eta=0.695": 0.695, "worst eta=0.30": 0.30}   # (only used if P_8G_MEASURED is None)

# --- MEASURED by the stwo-vm sweep -----------------------------------------------------------------
# Optimal split on 96 vCPU = K=16, T=6 (all leaves in one wave). Throughput peak 0.865 leaves/s;
# saturation: memory BW plateaus at K>=8 (t_node ~10-11s). So P_8g = 16 (validates the 8*P_1G structural
# guess). NOTE the per-fold times in ANCHORS are already the K=16-contended values — do NOT also apply
# an eta discount (that would double-count the contention now baked into t_leaf/t_node). Remaining
# uncertainty = the DDR4 absolute (bounded, not measured) — captured by the two ANCHORS rows.
P_8G_MEASURED = {"K=16 (sweep optimum, 96 vCPU)": 16.0}
# -------------------------------------------------------------------------------------------------

SP1 = {1: 18.5, 10: 30.6, 50: 88.0, 100: 157.0, 500: 705.0, 1000: 1387.0, 2000: 2753.0}
K_POINTS = [1, 10, 50, 100, 500, 1000, 2000]
# ===============================================================


def fmt(s):
    if s < 90:   return f"{s:.0f}s"
    if s < 5400: return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def run(label, t_leaf, t_node, eta_label, P_8g, shard_rows=SHARD_ROWS, t_base=T_BASE, t_precompute=0.0):
    # t_base is the FOLD per-shard base (preprocessed amortized); t_precompute is the one-time
    # shard-invariant precompute paid ONCE before the shard loop (matters only at tiny N).
    print(f"\n## {label} | {eta_label}  -> P_8g={P_8g:.1f} recursion pools")
    print(f"   {'k':>5} | {'N':>5} | {'base_wall':>9} | {'rec_wall':>9} | {'binds':>5} | {'T(8g)':>8} | {'SP1':>7} | ratio")
    for k in K_POINTS:
        N = max(1, math.ceil(ROWS_PER_K * k / shard_rows))
        base_wall = math.ceil(N / 8) * t_base + t_precompute
        rec_wall = N * (t_leaf + t_node) / P_8g
        binds = "rec" if rec_wall >= base_wall else "GPU"
        T = max(base_wall, rec_wall) + TAIL
        sp1 = SP1[k]
        print(f"   {k:>5} | {N:>5} | {fmt(base_wall):>9} | {fmt(rec_wall):>9} | {binds:>5} | "
              f"{fmt(T):>8} | {fmt(sp1):>7} | {T/sp1:>4.1f}x")


def main():
    print("=" * 78)
    print(f"a2-highgpu-8g projection (TWO-BALANCE overlap model) | shard=2^{SHARD_LOG} | t_base={T_BASE}s")
    print("=" * 78)
    # Recursion-concurrency scenarios: measured (from the stwo-vm sweep) if available, else the
    # modeled 8*P_1G*eta range.
    if P_8G_MEASURED:
        pools = P_8G_MEASURED
        print("(P_8g source: stwo-vm sweep, measured)")
    else:
        pools = {lbl: 8 * P_1G * eta for lbl, eta in ETA.items()}
        print(f"(P_8g source: MODELED 8*P_1G*eta, P_1G={P_1G}, eta={list(ETA.values())} — refine via box-ops/stwo-vm/sweep.sh)")
    for label, a in ANCHORS.items():
        for pool_label, P_8g in pools.items():
            run(label, a["t_leaf"], a["t_node"], pool_label, P_8g)

    # ---- 2^25 curve — FINAL, a2-highgpu-8g (8× A100-40 GB), all MEASURED (2026-07-05) ----
    # Stack: migration(stwo 5ea05973, 8-word root) + DECOUPLED k-ary(k=8) + H_P + L3. (plot_k_sweep.py is the plotted twin.)
    # base_wall = ceil(N/8)·t_base_fold + one-time precompute. t_base_fold = FOLD-amortized per-shard base @2^25 WITH L3,
    #   MEASURED on the A100 = 9.53s (preprocessed 4.64 amortized to ~0/shard; single-shot was 14.53). 2^26 DEAD on 40 GB.
    # Recursion (DECOUPLED k=8, CPU-VM proxy for a2-8g, cross-calib caveat): t_leaf PINNED 17.30 (not ballooned) +
    #   t_node_eff 1.68 (t_node1 13.4 amortized /8 leaves) = per-shard rec 18.98s = 0.681× vs coupled k=2 (27.86).
    # TAIL 2.07 MEASURED (root-verification/unpacker; was assumed 6). RESULT: base_wall ≈ rec_wall @k2000 (~1640 vs 1625)
    # ⇒ BALANCED/base-bound ⇒ ~0.60× vs SP1 across k≥50 (k=1 → 0.88×). Next lever = base (L3 full ncu-tune / faster t_base).
    T_BASE_2P25 = 9.53          # MEASURED fold-amortized per-shard base @2^25, with L3 (A100)
    LEAF_2P25, NODE_2P25 = 17.30, 1.68   # decoupled k=8: t_leaf pinned + effective per-shard node term (t_node1/8)
    TAIL_2P25 = 2.07            # MEASURED root-verification/unpacker (flat; +0.0021·N negligible)
    print("\n" + "=" * 78)
    print(f"2^25 FINAL curve — a2-8g decoupled k=8 (t_base_fold={T_BASE_2P25}s, t_leaf={LEAF_2P25}, "
          f"t_node_eff={NODE_2P25}, TAIL={TAIL_2P25}, K=16)")
    print("=" * 78)
    global TAIL; TAIL = TAIL_2P25
    run("2^25 FINAL (a2-8g, decoupled k=8)", LEAF_2P25, NODE_2P25, "K=16",
        list(P_8G_MEASURED.values())[0], shard_rows=1 << 25, t_base=T_BASE_2P25, t_precompute=4.64)


if __name__ == "__main__":
    main()
