#!/usr/bin/env python3
"""gate-sim STWO AIR (this work) vs SP1 — iadd256, 9024 shots, sweep reps (K).

Same axes as Tanuj's SP1 graph. SP1 points are his measured a100-8gpu values.
gate_air: K=1 is MEASURED (N=9024, 22.98M rows, prove_s=10.96s on c4-highmem-96,
AVX-512); K>1 is EXTRAPOLATED from gate_air's linear scaling (time = 10.96*K s,
since rows = 9024*K*2547 and prove is ~linear at ~0.48 us/row). K>=10 at 9024
shots is NOT single-proof-feasible (2^28+ rows > 732 GB) — extrapolation only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

reps = [1, 10, 50, 100, 500, 1000, 2000]

# SP1, measured, 8x A100 GPU, "total session time" (from Tanuj's graph).
sp1 = [18.5, 30.6, 88, 157, 705, 1387, 2753]

# gate_air: prove_s. K=1 measured; rest = 10.96 * K (linear, anchored at K=1 N=9024).
GATE_K1 = 10.96
gate = [GATE_K1 * k for k in reps]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(reps, sp1, "o-", color="#1f77b4", label="SP1 — 8×A100 GPU (measured, total session)")
# gate_air: measured K=1 filled, extrapolated open/dashed.
ax.plot(reps[1:], gate[1:], "s--", color="#d62728", mfc="white",
        label="gate-sim STWO — 96-core CPU/AVX-512 (prove, EXTRAPOLATED)")
ax.plot([1], [gate[0]], "s", color="#d62728", markersize=9,
        label="gate-sim STWO — K=1 (MEASURED, 22.98M rows)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of Reps (K), log scale")
ax.set_ylabel("Time (seconds, log scale)")
ax.set_title("Proof Generation for iadd256, 9024 shots:\ngate-sim STWO AIR (this work) vs SP1")
ax.set_xticks(reps); ax.set_xticklabels([str(r) for r in reps])
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=8, loc="upper left")

caveat = ("Caveats: gate_air = prove-only on CPU; SP1 = full session on 8×A100 GPU.\n"
          "Only gate_air K=1 is measured; K≥10 at 9024 shots exceeds 732 GB (extrapolated).\n"
          "gate_air proves gate-exec + program-consistency (commitment still TODO).")
ax.text(0.5, -0.30, caveat, transform=ax.transAxes, fontsize=7,
        ha="center", va="top", color="#444")

fig.tight_layout()
out = "gate_air_vs_sp1.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
print("\nK     SP1(s)   gate_air(s)   note")
for k, s, g in zip(reps, sp1, gate):
    print(f"{k:<5} {s:<8} {g:<13.1f} {'MEASURED' if k==1 else 'extrapolated'}")
