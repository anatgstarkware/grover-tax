# P3 — CUDA gate_air trace-gen (on-device "model B") — scope

GOAL: generate gate_air's trace ENTIRELY on the GPU so it never transfers host↔device. This removes
(a) the CPU trace-gen bottleneck (the measured wall — ~0.5 Mrows/s/slice, memory-bound) and (b) the
model-A transfer overhead (measured: GPU commit 31s vs SIMD 9.8s @2^24 — transfer-bound). On-device
trace → commit/FRI with no transfer is the only path to beat SP1 (per extrapolate.py: CPU-feed-bound).

## gate_air dimensions (from main.rs)
- N_QUBITS=512, LIMB_BITS=16 → **N_LIMBS=32** (state = 32×u16 packed as u32 limbs), STATE_BYTES=64.
- **n_gates=2547**, **n_shots=9024** (Tanuj benchmark; k=reps varies). total rows = n_shots·k·n_gates.
- **TRACE_COLUMNS=191** main (=1 enabler +4 opcode-onehot +2 shot/pc +32 in_limb +32 out_limb
  +3·READ_COLS(39) +3 ab/fire/delta). READ_COLS=39 = 4 + 32 lsel-onehot + 3 (lo/hi/bit).
- Lookup histograms: **qdecode[512], rc_lo[2^16], rc_hi[2^16]**.
- Trees: 0=preprocessed (qdecode table 512, rc_lo/rc_hi tables 2^16 each, program n_gates, pc_in_prog
  — MIXED SIZES), 1=main(191)+multiplicity(3), 2=interaction (LogUp).

## Per-gate compute (simulate_shot — the K1 kernel core, must match EXACTLY for soundness)
Sequential within a shot, threading `limbs[32]` + `pc`:
1. opcode → (is_nop,is_not,is_cnot,is_toffoli) one-hot; a_active=cnot+tof, b_active=tof.
2. 3 reads (target, ctrl_a if a_active, ctrl_b if b_active): ReadCols::live(limbs, qubit) via
   qubit_decode(q)→(limb_idx,bit_pos,mask); bit = (limbs[limb_idx]>>bit_pos)&1; lo/hi split.
3. ab=a_bit·b_bit; fire=is_not + is_cnot·a_bit + is_toffoli·ab; new_t=t_bit^fire; delta=new_t−t_bit.
4. write: out_limb=in_limb; out_limb[target.limb]±=mask per delta sign; limbs=out_limb.
5. count_read ×3 → atomicAdd qdecode[q], rc_lo[idx(bit_pos,lo)], rc_hi[idx(bit_pos,hi)].
6. emit Row → 191 cells (cell_at mapping) into device column-major arrays at row = shot·shot_rows +
   rep·n_gates + gate.
Self-check: final limbs == y (per shot; can do on GPU + flag, or skip in prod).

## Kernel decomposition
- **K1 — sim + main-trace (THE big/novel kernel).** thread-per-shot (9024 threads), each runs its
  k·2547 gate chain, writes 191 columns (device column-major u32) + atomicAdds the 3 histograms.
  Gate program (2547×{opcode,target,ctrl_a,ctrl_b}) in constant/global mem. State 32 u32/thread.
  RISK: occupancy — 9024 threads ≈ 4% of A100 capacity (221K); long per-thread chains. OK for v1
  (still 9024× vs CPU 12-96×); future: split a shot across threads via state checkpoints (re-sim
  prefix) if occupancy bound. Scalar column writes (not packed) → NO packed-word race (the CPU fuse
  problem disappears on GPU); pack→PackedM31 after, or commit consumes scalar M31 cols directly.
- **K2 — preprocessed (small, deterministic, MIXED sizes).** qdecode table(512), rc_lo/rc_hi(2^16),
  program(n_gates), pc_in_prog. Not shot-dependent. RECOMMEND: CPU-generate once + upload (tiny,
  fixed per circuit) for v1; GPU-kernel later. (Also dodges the GPU mixed-size NTT bug that blocked
  P1 — keep preprocessed CPU-built, only its commit on GPU.)
- **K3 — multiplicity cols.** the K1 histograms ARE the multiplicity values (indexed by table row) →
  scatter/copy to columns on device. trivial.
- **K4 — LogUp interaction (2nd big piece).** per row, per the 5 relation-pairs (state, qdecode×3,
  rc_lo×3, rc_hi×3, program): denominator = el.<rel>.combine(tagged fields) — a QM31 linear combo of
  M31 fields with alpha-powers + z (challenges from the channel, uploaded post-draw); numerator =
  multiplicity / ±1. column = num0/den0 + num1/den1 → batch-INVERSE the denoms (obelyzk batch_inverse)
  + per-row fraction (parallel) + PREFIX-SUM scan (running LogUp sum) → interaction cols + claimed_sum.
  Must match LogupTraceGenerator EXACTLY (claimed_sum identical). QM31 arith + scan on GPU.

## Reusable from obelyzk vs net-new
- Reuse: GpuBackend column types, batch_inverse, FFT/commit (consume device columns, no transfer),
  prefix-sum primitive (check obelyzk has a device scan; simd has prefix_sum.rs).
- NET-NEW CUDA: K1 gate-sim kernel (gate_air-specific; NitrooZK's cairo trace-gen is NOT reusable),
  K4 LogUp trace-gen (the LogupTraceGenerator is SimdBackend-only).

## Phasing
- P3.1: K1 sim/main-trace kernel → validate the 191 main columns + 3 histograms BYTE-IDENTICAL to the
  CPU build_rows/generate_main_trace (soundness gate; a column-by-column M31 assert test).
- P3.2: K4 LogUp interaction → validate interaction cols + claimed_sum identical to CPU.
- P3.3: K2/K3 preprocessed+multiplicity (or keep CPU-upload).
- P3.4: wire end-to-end (trace GPU-resident → commit → prove, NO transfer) → measure real t_gpu_pure
  on the actual AIR (the number model A couldn't give) → overlay on the SP1/Tanuj curve.

## Soundness (stwo CLAUDE.md): the GPU trace MUST equal the CPU trace bit-for-bit. Every kernel
## (K1 sim, K4 LogUp) gets a byte-identity test vs the CPU reference before it's trusted. No security
## params change; this only moves WHERE the witness is computed.

## Effort: K1 + K4 are the substantial new CUDA (multi-session). K2/K3 small. Validation-gated.
## Prereq: P1's GPU NTT mixed-size bug (evaluate_polynomials_gpu) still needs fixing for the commit
## of mixed-size preprocessed cols — OR keep preprocessed commit on the uniform path. Track that.
