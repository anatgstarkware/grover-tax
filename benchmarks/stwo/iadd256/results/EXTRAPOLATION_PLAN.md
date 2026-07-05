# Extrapolating a2-highgpu-1g → a2-highgpu-8g (gate_air GPU pipeline)

Goal: predict the Tanuj-benchmark running time on **a2-highgpu-8g** (8× A100-40GB, 96 vCPU,
680 GB — the apples-to-apples HW for SP1's curve) from measurements on a **single a2-highgpu-1g**
(both GPU and CPU parts — the 1g is Cascade Lake, same CPU as the 8g), combined as a pipeline.
(No a2-2g calibration — the 1g supplies the absolute CPU rate; only the 8-slice scaling efficiency
η is bounded rather than measured. See "η — bounded, not calibrated".)

## Why decompose (and the one rule that matters)
a2-highgpu-8g = EXACTLY 8× a2-highgpu-1g (8 A100 / 96 vCPU / 680 GB = 8×[1 A100 / 12 vCPU / 85 GB]).
But **8 GPUs give 8× THROUGHPUT, not 8× speed-per-item.** The two halves of our pipeline scale
very differently from 1g → 8g:

| part | runs on | scales 1g→8g | why |
|---|---|---|---|
| **GPU prove** (NTT/eval, FRI, quotients, composition) | A100 | **~×8** | 8 independent A100s, separate HBM, no shared bottleneck |
| **CPU work** (trace-gen + lifted-Merkle build) | host vCPU | **~2-3× only** | memory-bandwidth-bound; 8 concurrent shards contend for one host mem bus |

Evidence for the CPU wall: `run_gate_air_satsweep2.sh` on a 96-vCPU node — 8 concurrent CPU proves
gave only **~2.1×** the single-prove throughput (0.78→1.63 Mrows/s, plateau K≈3-4). More cores
don't help once bandwidth-saturated.

**Consequence:** on the GPU pipeline the prove is fast (GPU), so **trace-gen + lifted-Merkle become
the bottleneck**; because they only do ~2-3× per node, the 8 GPUs can STARVE. The 8g total is
likely **CPU-feed-limited, not 8× GPU-limited.** The whole point of this estimate is to find which
side binds.

## The model — PIPELINE BOTTLENECK (1g-only, RANGE; not a sum, not CPU×8)

**Key enabler for 1g-only: the a2-1g IS Cascade Lake = the SAME CPU microarch as the 8g host.** So
measuring the CPU bucket on the 1g needs NO microarch discount (this is why dropping the 2g is fine).
The 1g gives both halves directly; the only thing it can't give is how 8 concurrent CPU slices
contend for the shared host memory bus → we BOUND that with an efficiency range η and report a RANGE.

```
GPU_8g  = 8 · shard_rows / t_gpu_pure / 1e6              # 8 independent A100s (firm)
CPU_8g  in [ 8·r_slice·η_low , 8·r_slice·η_high ]        # 8 Cascade-Lake slices, contention η
throughput_8g = min(GPU_8g, CPU_8g)   (per η end)
T(k) = total_rows(k) / (throughput_8g · 1e6) + tree_tail
total_rows(k) = 22.984e6 · k       # iadd256 × 9024 shots, per rep k
```

- `t_gpu_pure` — pure-A100 per-shard time, MEASURED on a2-1g. **Must isolate from the CPU
  lifted-Merkle** (which the GPU pipeline still runs on CPU): take only the GPU phases (NTT/eval +
  FRI + quotients + composition) from the backend's own phase timers.
- `r_slice` — per-12-vCPU **CPU-bucket** rate (trace-gen + lifted-Merkle ONLY; prove_ex is on GPU),
  MEASURED on a2-1g = REAL Cascade Lake, no discount. Each 8g GPU also gets exactly 12 vCPU, so this
  IS the per-GPU CPU feed rate on the 8g.
- `η` (slice-scaling efficiency) — how 8 concurrent slices share the 8g host memory bus. The one
  thing a single 1g can't measure. BOUND it: η_low ≈ 0.3 (heavy DDR4 saturation; stwo-vm shape gave
  ~2.1×@K=8 → ~0.26), η_high ≤ 1.0 from the 1g **contention probe** (1×12-vCPU vs 2×6-vCPU: ratio<2
  ⇒ bus saturates ⇒ pull η_high down). Report the resulting T(8g) as a range.

## Biases — status with 1g-only
1. **WRONG CPU microarch — ELIMINATED.** We no longer use stwo-vm's c4 for the absolute CPU rate;
   the 1g IS Cascade Lake. (stwo-vm now only informs the η *shape*, not the absolute number.)
2. **Isolation — still applies.** lifted-Merkle is on CPU in the GPU pipeline; split the 1g "prove
   time": A100-only → `t_gpu_pure` (×8); trace-gen+Merkle → `r_slice`.
3. **NEW residual = η (scaling) is bounded, not measured** (no 2g). This is the T(8g) SPREAD. The
   qualitative result (CPU-bound, GPUs starve) is expected to hold across the whole η range → robust;
   only the exact 8g time carries the η uncertainty.

## What scales / what is INVARIANT 1g↔8g
- INVARIANT (same on both): single-shard GPU latency, single-shard CPU latency, **max shard size
  (40 GB VRAM/GPU on both — 8g does NOT allow a bigger single shard)**, the fold-tree top-level
  tail + root wrapper (latency-bound; small fraction for large N → folded into `tree_tail`).
- SCALES: GPU throughput (×8), CPU aggregate (~2-3×, machine-bound — captured by r_cpu).

## Measurement checklist (inputs the combiner needs — ALL from the a2-1g)
| input | how (run_gpu_1g.sh step) | status |
|---|---|---|
| `t_gpu_pure` (s/shard) + GPU/CPU phase split | step (1)+(2): warm GPU prove, take GPU-only phases | PENDING (box + obelyzk port) |
| `r_slice` (Mrows/s, CPU bucket, Cascade Lake) | step (4): CPU-backend [phase] timers, rows / (sim+preprocessed+main+interaction+Merkle) | PENDING (cleanest needs a `--commit-only` flag; else sum CPU phases) |
| `η` bound | step (5) contention probe (1×12c vs 2×6c) → η_high; η_low ~0.3 from stwo-vm shape | PENDING |
| GPU under-feed signal | step (3): 1-GPU K=1 vs K=2 (does a 2nd prove help?) | PENDING |

## η — bounded, not calibrated (no 2g)
The 2g would have measured the slice-scaling efficiency directly; without it we bound η:
- **η_high** from the 1g contention probe (step 5): if 2×6-core buckets ≈ 2× a single → bus has
  headroom → η_high≈1; if they saturate (ratio≪2) → pull η_high down toward that ratio.
- **η_low** from the stwo-vm saturation shape (~2.1× at K=8 on a fixed node → ~0.3) as a pessimistic
  floor (note: c4 DDR5 has MORE headroom than a2 DDR4, so a2 could be worse → η_low is not a hard
  floor, just a guide).
Report T(8g) as the [η_low, η_high] range. If a 2g/8g becomes available later, one e2e run collapses
the range to a point; until then the spread IS the stated uncertainty.

## Use
`python3 extrapolate.py` — edit the INPUTS block with measured values (placeholders flagged), prints
the 8g curve vs SP1 and which side binds (GPU vs CPU).
