#!/usr/bin/env python3
"""k-sweep: single a2-highgpu-8g (estimated) vs SP1 8xA100 (Tanuj), incl k=8000.

Overlap model (extrapolate_8g.py), measured 2^25 anchors (A100, 22-col gate_air):
  N = ceil(ROWS_PER_K*k / 2^25);  base_wall = ceil(N/8)*t_base;  rec_wall = N*(t_leaf+t_node)/P
  T(8g) = max(base_wall, rec_wall) + tail
SP1: published to k=2000; linear extrapolation (k=1000->2000 slope) beyond, drawn dashed.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- model + measured 2^25 anchors ----
SHARD_ROWS = 1 << 25
ROWS_PER_K = 22.984e6
# MEASURED anchors (2026-07-05), final stack = migration(5ea05973) + DECOUPLED k-ary(k=8) + H_P + L3.
# T_BASE_FOLD 9.53s = fold-amortized per-shard base @2^25 WITH L3, MEASURED on the A100 (anat-ganor). T_PRECOMPUTE 4.64s
#   = one-time shard-invariant precompute (matters only at tiny N). Recursion (DECOUPLED k=8, CPU VM proxy for a2-8g,
#   cross-calib caveat): T_LEAF 17.30 (leaf PINNED ~2^21, not ballooned) + T_NODE 1.68 = effective per-shard node term
#   (t_node1 13.4 amortized over 8 leaves; level-≥2 negligible) ⇒ per-shard rec 18.98s = 0.681× vs coupled k=2 (27.86).
# P_8G 16 (K=16 pools fit ~680 GB). TAIL 2.07 = MEASURED root-verification/unpacker (was assumed 6; N-scaling +0.0021·N negligible).
T_BASE_FOLD, T_PRECOMPUTE, T_LEAF, T_NODE, P_8G, TAIL = 9.53, 4.64, 17.30, 1.68, 16, 2.07

def T_ours(k):
    N = max(1, math.ceil(ROWS_PER_K * k / SHARD_ROWS))
    base_wall = math.ceil(N / 8) * T_BASE_FOLD + T_PRECOMPUTE
    rec_wall = N * (T_LEAF + T_NODE) / P_8G
    return max(base_wall, rec_wall) + TAIL

SP1 = {1: 18.5, 10: 30.6, 50: 88.0, 100: 157.0, 500: 705.0, 1000: 1387.0, 2000: 2753.0}
SP1_SLOPE = (2753.0 - 1387.0) / (2000 - 1000)
def sp1(k):
    return SP1[k] if k in SP1 else SP1[2000] + SP1_SLOPE * (k - 2000)

K = [1, 10, 50, 100, 500, 1000, 2000, 4000, 8000]
ours = [T_ours(k) for k in K]
theirs = [sp1(k) for k in K]
ratio = [o / t for o, t in zip(ours, theirs)]
K_SPLIT = 2000  # measured/published up to here; extrapolated beyond

def hms(s):
    if s < 90: return f"{s:.0f}s"
    if s < 5400: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"

OURS_C, SP1_C = "#0f9d8a", "#e2622f"

fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.2, 8.0), height_ratios=[3, 1.15], sharex=True)

# shade the extrapolated region (k > 2000)
for a in (ax, axr):
    a.axvspan(K_SPLIT, K[-1] * 1.15, color="0.92", zorder=0)

def split_plot(a, x, y, color, label, marker):
    x, y = np.array(x, float), np.array(y, float)
    m = x <= K_SPLIT
    # solid over measured/published range (include the split point in the dashed leg too)
    a.plot(x[m], y[m], color=color, lw=2.4, marker=marker, ms=6, label=label, zorder=4)
    idx = np.where(x >= K_SPLIT)[0]
    a.plot(x[idx], y[idx], color=color, lw=2.4, ls="--", marker=marker, ms=6, zorder=4)

split_plot(ax, K, ours, OURS_C, "This work — 1× a2-highgpu-8g (decoupled k=8, measured anchors)", "o")
split_plot(ax, K, theirs, SP1_C, "SP1 — 8× A100 (Tanuj)", "s")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel("wall-clock time")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: hms(v)))
ax.set_title("iadd256 secret gate-circuit ZKP — single a2-highgpu-8g vs SP1 8×A100\n"
             "(9024 shots × k reps; shaded = extrapolated beyond published SP1 k=2000)",
             fontsize=12)
ax.grid(True, which="both", ls=":", alpha=0.45)
ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

# annotate ratio + time at k=2000 and k=8000
for k in (2000, 8000):
    i = K.index(k)
    ax.annotate(f"k={k}\n{hms(ours[i])} vs {hms(theirs[i])}\n{ratio[i]:.3f}×",
                xy=(k, ours[i]), xytext=(k*0.30, ours[i]*0.34),
                fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OURS_C, alpha=0.95))

# ratio panel
axr.plot(K[:K.index(K_SPLIT)+1], ratio[:K.index(K_SPLIT)+1], color=OURS_C, lw=2.2, marker="o", ms=5)
idx = [i for i, k in enumerate(K) if k >= K_SPLIT]
axr.plot([K[i] for i in idx], [ratio[i] for i in idx], color=OURS_C, lw=2.2, ls="--", marker="o", ms=5)
axr.axhline(1.0, color="0.5", lw=1, ls="-")
axr.set_ylabel("ratio\n(ours / SP1)", fontsize=10)
axr.set_xlabel("k  (repetitions per shot)")
axr.set_xscale("log")
axr.set_ylim(0.5, 1.25)
axr.grid(True, which="both", ls=":", alpha=0.45)
axr.set_xticks(K); axr.set_xticklabels([str(k) for k in K])
axr.text(0.99, 0.90, "below 1.0 = faster than SP1", transform=axr.transAxes,
         ha="right", va="top", fontsize=8.5, color="0.35")
for k in (1, 8000):
    i = K.index(k)
    axr.annotate(f"{ratio[i]:.2f}×", (k, ratio[i]),
                 textcoords="offset points", xytext=(0, 8 if k == 1 else -14),
                 ha="center", fontsize=8.5, color=OURS_C)

fig.tight_layout()
out = "/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/k_sweep_vs_sp1.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print(f"\n{'k':>6} {'ours':>9} {'SP1':>9} {'ratio':>7}")
for k, o, t, r in zip(K, ours, theirs, ratio):
    tag = "" if k <= K_SPLIT else "  (extrap)"
    print(f"{k:>6} {hms(o):>9} {hms(t):>9} {r:>6.3f}x{tag}")
