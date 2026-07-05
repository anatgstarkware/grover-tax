# t_base DECOMPOSITION ANALYSIS — gate_air CUDA base proof (2^24 streaming)

Read-only analysis. No code changed, no box spent. All timer-semantics claims cite file:line.
Where a number needs a box sub-timer that does not exist, it is called out as "NEEDS SUB-TIMER",
not fabricated.

---

## Q1 — What t_base actually decomposes into (CODE EVIDENCE)

### 1a. What does JSON `trace_gen_s` time? — IT IS MISLEADINGLY NAMED. **CONFIRMED.**

`trace_gen_s` is `trace_gen_elapsed`, measured from a single `Instant`:
- **START** `trace_gen_start` at `gate-air-leaf/src/main.rs:2281` — "beginning of the full witness
  build (shot simulation + every column fill + interaction traces)".
- **STOP** `trace_gen_elapsed = trace_gen_start.elapsed()` at `main.rs:3521`, explicitly "up to (not
  including) the FRI prove" (comment main.rs:3518-3520). `prove_s` starts right after at `main.rs:3527`
  (`prove_start`) and wraps only the `prove_ex::<…>()` call (main.rs:3532).

Everything between those two `Instant`s is inside `trace_gen_s`. Walking the single-shard base path
(main.rs:3107→3521), that window contains, IN ORDER, each with its own `[phase]` eprintln:

| sub-phase | file:line | what it is | GPU/CPU/PCIe |
|---|---|---|---|
| build_rows (shot SIM → witness rows) | main.rs:2284, timer `build_elapsed` printed 2300-2302 | **CPU gate simulation** + self-check + LogUp count reduce | CPU |
| preprocessed gen+commit | main.rs:3116-3123 (tree0) | preprocessed cols + tree0 NTT+Merkle | GPU + a little CPU |
| main_trace witness gen (GPU K1) | main.rs:3155-3160 | 188 main cols generated **on GPU** (`gpu_gen_main_trace_device`), `d_cols` device-resident | GPU |
| **tree1 commit (NTT+Merkle)** | main.rs:3198-3200 | interpolate/extend/n2b + build_leaves Merkle over the 188 main cols | **GPU compute + PCIe (streaming) + a little CPU** |
| interaction (K4) gen | main.rs:3222-… | 24 interaction cols on GPU reusing `d_cols` | GPU |
| tree2 commit (NTT+Merkle) | main.rs:3490-3503 | interaction tree | GPU |
| build_components | main.rs:3505-3516 | cheap CPU bookkeeping | CPU |

**So the answer to your first suspicion is YES, confirmed with two independent pieces of evidence:**

1. **`trace_gen_s` INCLUDES the tree1 commit (the NTT+Merkle).** The `[phase] tree1 commit` eprintln
   (main.rs:3200) fires strictly between `trace_gen_start` (2281) and `trace_gen_elapsed` (3521). The
   27.7 s tree1 commit @2^24 is a SUB-COMPONENT of the 36.3 s `trace_gen_s`, not part of `prove_s`.

2. **The CPU gate SIMULATION → witness columns is ALSO inside `trace_gen_s`, and it is SMALL — and it
   HAS a timer.** `build_rows` (main.rs:2284) is the CPU sim; its cost is `build_elapsed`, printed as
   `"shots simulated and self-checked (final state == y) in {:.3}s"` (main.rs:2300-2302). On the CUDA
   path the sim→columns witness fill also runs on GPU (K1, main.rs:3155), so the CPU sim is only the
   `build_rows` portion.

**Arithmetic check @2^24** (your phase numbers): preprocessed 2.2 + main K1 1.3 + tree1 27.7 +
interaction 0.1 + tree2 0.3 = **31.6 s of named phases**, vs `trace_gen_s` = 36.3 s. The **~4.7 s
residual = build_rows (CPU sim) + build_components + per-phase glue**. So the CPU sim is *at most*
~4.7 s of the 36.3 s `trace_gen_s` — i.e. **≤ ~13% of trace_gen_s, ≤ ~9% of t_base**. It is NOT the
dominant term. The dominant term of `trace_gen_s` is the **tree1 commit (27.7 s = 76% of trace_gen_s,
54% of t_base)**.

> **VERDICT Q1a: `trace_gen_s` is misleadingly named — it is dominated by the GPU tree1 commit
> (NTT + PCIe rehydrate + Merkle), NOT by CPU simulation.** The `build_rows` CPU sim is ≤~4.7 s. The
> exact split of the ~4.7 s residual (build_rows vs build_components vs glue) NEEDS A SUB-TIMER: the
> `build_elapsed` eprintln already exists (main.rs:2300) — the next 2^24 box run should just capture
> that "shots simulated … in Xs" line. That pins the CPU sim exactly.

**IMPORTANT caveat for downstream reasoning:** the OPTIMIZATION_PLAN line "trace_gen ≈72% of t_base
(CPU memory-bound)" conflates two different things. `trace_gen_s` here is 36.3/51.5 = 70% of t_base —
BUT that 70% is mostly the GPU tree1 commit, not CPU. The genuinely-CPU part (build_rows) is ≤~4.7 s.
See Q4.

### 1b. Breaking the 2^24 tree1 commit (27.7 s) into sub-parts

Data-flow of the streamed tree1 commit (all in `stwo-cuda-backend`):
- **NTT** — per eval column: in-place interpolate → extend (2× blowup) → `ntt_n2b_columns`
  (poly.rs:711). GPU compute. Happens in the fused producer loop (poly.rs:489, guarded by
  `stream_tree1`).
- **dehydrate D2H** — `dehydrate_column` (fused_commit.rs:135): `col.to_vec()` = **synchronous
  pageable whole-column D2H** (fused_commit.rs:138) → `cuda_free_memory` → `cuda_stream_reclaim_freed(0)`
  (fused_commit.rs:149) = **`cudaStreamSynchronize(0)` + `cudaMemPoolTrimTo` PER COLUMN** (a full
  host/device barrier ×188). This is PCIe D2H + a hard serialization barrier.
- **build_leaves rehydrate H2D + Merkle hash** — `build_leaves` heterogeneous path (blake2s.rs:104,
  268-293): for each staged col, `rehydrate_owned` (whole-column **H2D**, blake2s.rs:284) → ONE
  `blake2s_update_columns` absorb (blake2s.rs:286) → free → `cuda_stream_reclaim_freed(0)` again
  (blake2s.rs:288). **This is a SINGLE incremental absorb pass** (not two hash passes) — but it
  rehydrates every one of the 188 cols back H2D that dehydrate just sent D2H.

So one 256-MiB main column @2^24 makes a **round trip**: D2H at dehydrate (commit-input staging), then
H2D at build_leaves (to hash). Plus 188×2 reclaim barriers.

**Bounding the transfer portion from the resident→streaming delta** (resident numbers use no
stash, all device-resident, so their commit is pure GPU NTT+Merkle):
- @2^23: tree1 commit 0.82 s resident → 13.9 s streaming ⇒ **streaming tax +13.1 s @2^23**.
- @2^24: resident tree1 not measured directly. The resident 2^23 was 0.82 s; NTT+Merkle is ~linear×log,
  so resident 2^24 ≈ **~1.6–1.8 s (ESTIMATE, NEEDS the resident-2^24 sub-run to confirm — 2^24 was
  only ever run streaming because resident 2^24 OOMs, which is WHY streaming exists)**. Streaming 27.7 s
  ⇒ **streaming tax ≈ +26 s @2^24**.

Raw-bandwidth floor for the transfers @2^24: 188 cols × 256 MiB ≈ 47 GB each direction; PCIe gen4 ×16
≈ 25 GB/s ⇒ ~1.9 s per direction ⇒ **~3.8 s of unavoidable raw D2H+H2D bytes** (round trip). That is
only ~15% of the +26 s tax. **Therefore the tree1 streaming tax is NOT bandwidth-bound — it is
dominated by SERIALIZATION**: the 188×2 `cudaStreamSynchronize(0)` reclaim barriers + pageable copies
that cannot overlap compute + per-column single-column absorbs (vs one fused multi-column absorb in the
resident path). The pageable `to_vec()` (fused_commit.rs:138) forces the driver through a bounce buffer
and blocks the host.

**Sub-split I can bound vs. what NEEDS a sub-timer:**
- transfers (PCIe raw bytes): **~3.8 s @2^24** (round-trip, computed from byte volume + PCIe BW — a
  physical floor, not a measured split).
- GPU compute (NTT interpolate/extend/n2b + Merkle): **≈ resident tree1 ≈ ~1.6–1.8 s** (est).
- the rest (~22 s): the reclaim barriers + non-overlap + per-column-absorb serialization.

> **NEEDS SUB-TIMERS (specify exact runs):**
> - (T1) A resident-2^24 tree1 commit number — impossible today (2^24 resident OOMs). Proxy:
>   instrument the streamed producer loop to sum (a) NTT kernel time, (b) D2H time, (c) reclaim-barrier
>   time, (d) build_leaves H2D time, (e) absorb time — five `Instant` accumulators around
>   poly.rs:711 / fused_commit.rs:138 / :149 / blake2s.rs:284 / :286. One 2^24 streaming run prints the
>   real split. THIS is the run to do — it converts the "~22 s serialization" bucket from inference to
>   measurement.

### 1c. prove_ex rehydrate tax (composition 5.9 / OODS 5.6 / quotient 2.8 @2^24)

prove_ex staged consumers, all re-reading the stash (H2D), all on the default stream:
- **composition** — staged cols supplied per-tile to `evaluate_gate_air.cu` (tiled_input=true); the tile
  is H2D'd per row-block per column.
- **OODS** — `barycentric_eval_at_point` (poly.rs:397): `if is_staged → rehydrate_owned` (poly.rs:410-411)
  = **WHOLE-column H2D per OODS eval**, run on rayon workers (par_map_cols). 188 whole-column H2D.
- **quotient** — `rehydrate_block(col, off, this_block)` (quotient.rs:309-310): per row-block per
  column H2D into a reused tile buffer.

Bounding from resident→streaming: @2^23 prove_ex 2.30 s resident → 8.93 s streaming ⇒ **+6.6 s
rehydrate tax @2^23**. @2^24 the streaming prove_s is 17.0 s (composition 5.9 + OODS 5.6 + quotient 2.8
+ FRI 2.0 + PoW 0.2 + decommit 0.5). Scaling the +6.6 s @2^23 by the ~2× row growth and the ~2× column
byte growth gives a **rehydrate tax @2^24 in the ~12–14 s band** — i.e. **most of composition+OODS+
quotient (~14 s) is the staged H2D + non-overlap, NOT the arithmetic**. The resident 2^23 prove_ex of
2.30 s is the "true compute" reference; resident 2^24 compute ≈ ~4–5 s (est, ~2×). So of the 14.3 s
(comp+OODS+quot) @2^24, **~4–5 s is real GPU compute and ~9–10 s is rehydrate H2D + serialization
(band)**.

> **NEEDS SUB-TIMER (T2):** per-consumer H2D-vs-kernel accumulators — wrap the H2D in
> `rehydrate_owned`/`rehydrate_block` and the kernel launch separately (poly.rs:411, quotient.rs:310,
> the composition tile H2D in gate_air_entry.cu). One 2^24 streaming run then gives the exact
> compute/copy split for each of composition/OODS/quotient. The ~9–10 s band above is inferred from
> the 2^23 resident-vs-streaming delta scaled to 2^24 — NOT measured at 2^24.

### DECOMPOSITION TABLE — t_base @2^24 (51.5 s), best current attribution

| bucket | seconds @2^24 | basis | class |
|---|---|---|---|
| CPU gate sim (build_rows) + glue | ≤ ~4.7 | residual: trace_gen_s 36.3 − named phases 31.6 | **CPU** (memory-bound) |
| preprocessed (tree0) | 2.2 | [phase] | GPU |
| main_trace K1 (GPU witness gen) | 1.3 | [phase] | GPU |
| **tree1 commit** | **27.7** | [phase] | GPU compute ~1.7 + PCIe raw ~3.8 + **~22 serialization/non-overlap** (est) |
| interaction K4 | 0.1 | [phase] | GPU |
| tree2 commit | 0.3 | [phase] | GPU |
| composition | 5.9 | [prove_ex] | ~compute + rehydrate (split NEEDS T2) |
| OODS | 5.6 | [prove_ex] | ~compute + whole-col rehydrate |
| quotient | 2.8 | [prove_ex] | ~compute + block rehydrate |
| FRI + PoW + decommit | 2.7 | [prove_ex] | resident (not staged) |
| **t_base** | **~51.5** | wall (JSON) | |

Streaming-tax split (what (ii)/fuse can attack): **tree1 commit ~+26 s** (rehydrate H2D + reclaim
barriers, only ~3.8 s is raw bytes) **+ prove_ex ~+9–10 s** (rehydrate). CPU sim (~≤4.7 s) and
resident FRI/tree2 are untouchable by fuse/(ii).

---

## Q2 — Fused-1a-proper saving estimate

**What Fused-1a does:** absorb each large main column into the leaf hash **as it is produced** (inside
the NTT loop, right after n2b, BEFORE the dehydrate D2H). Result: `build_leaves` no longer has to
rehydrate the 188 cols back H2D just to hash them — the hash already consumed the live device column.
The dehydrate D2H for the LATER consumers (composition/OODS/quotient) still happens, because those still
need the bytes.

**What it removes from tree1 commit:**
1. The **188 whole-column build_leaves rehydrate H2D** (blake2s.rs:284). @2^24 that is ~47 GB of H2D =
   ~1.9 s raw PCIe **plus** the far larger serialization/non-overlap around it.
2. The **188 build_leaves reclaim barriers** (`cuda_stream_reclaim_freed(0)`, blake2s.rs:288) — a
   `cudaStreamSynchronize(0)` per column, pure serialization.
3. It does NOT add a second hash pass — the current path is already a single incremental absorb
   (blake2s.rs:286, confirmed). Fused-1a just moves that same single absorb earlier (into the producer
   loop) so the H2D+barrier that fed it disappears.

**What it does NOT remove:** the dehydrate D2H (fused_commit.rs:138) + its reclaim barrier
(fused_commit.rs:149) stay, because composition/OODS/quotient still rehydrate from the stash.

**Estimate.** The tree1 streaming tax @2^24 is ~+26 s over the ~1.7 s resident compute. Structurally
that tax is roughly **half dehydrate-side (D2H + 188 reclaim barriers)** and **half build_leaves-side
(H2D + 188 reclaim barriers + per-col absorb)** — symmetric round trip, symmetric barrier count.
Fused-1a removes the build_leaves-side entirely.

- Projected **tree1 commit post-fuse @2^24: ~13–17 s** (band): resident compute ~1.7 s + dehydrate D2H
  ~1.9 s raw + the ~188 dehydrate reclaim barriers + non-overlap that remain (~10–13 s). Center ~15 s.
- Projected **t_base@2^24 post-fuse: ~51.5 − (27.7 − 15) ≈ ~39 s** (band ~37–42 s). Only tree1 changes;
  prove_ex still pays its ~9–10 s rehydrate.

> **Assumptions / what NEEDS a sub-timer:** the "half/half" split of the +26 s tax between dehydrate-side
> and build_leaves-side is INFERRED from the symmetric round-trip structure, NOT measured. The T1
> sub-timer run (Q1b) is exactly what confirms it: it directly measures the build_leaves H2D + its
> barrier time, which is precisely the slice Fused-1a deletes. Do T1 before banking the ~39 s.

---

## Q3 — Fuse + (ii) stacking, and breakeven

### Are fuse and (ii) complementary? — **YES, confirmed from the data-flow.**

- **Fuse ELIMINATES** the build_leaves rehydrate H2D + its 188 barriers (removes work entirely — it
  never happens).
- **(ii) HIDES** the copies that REMAIN behind compute (makes them `max(compute, copy)` not
  `compute + copy`), via pinned buffers + a copy stream + CUDA events (per ASYNC_OVERLAP_SCOPE §3). The
  remaining copies after fuse are: the **dehydrate D2H** (fused_commit.rs:138) and the **prove_ex
  rehydrate H2D** (composition/OODS/quotient). (ii) also replaces the per-column
  `cuda_stream_reclaim_freed(0)` barrier (Part A) which fuse does NOT touch on the dehydrate side.

They attack disjoint slices: fuse deletes build_leaves-side; (ii) hides dehydrate-side + prove_ex-side.
No double-counting. Stacking is real.

### Projected t_base@2^24 (bands, not points)

| config | tree1 commit | prove_ex extra (comp+OODS+quot over resident) | t_base@2^24 | basis |
|---|---|---|---|---|
| streaming as-is | 27.7 | ~9–10 tax | **51.5** | measured |
| **fuse-only** | ~13–17 | ~9–10 tax | **~37–42** | Q2 |
| **(ii)-only** | ~10–14 | ~2–4 (most hidden) | **~28–36** | ASYNC_OVERLAP_SCOPE §5 band |
| **fuse + (ii)** | ~4–8 | ~2–4 | **~24–31** | fuse removes build_leaves work; (ii) hides the rest |

Reasoning for fuse+(ii) tree1: fuse drops it to ~13–17 s (build_leaves work gone); (ii) then hides the
remaining dehydrate D2H + reclaim behind the NTT compute, collapsing toward `max(NTT compute, D2H)` ≈
resident compute ~1.7 s + a small exposed remainder ⇒ **~4–8 s**. prove_ex: (ii) hides the rehydrate ⇒
comp+OODS+quot fall toward their resident ~4–5 s (tax ~2–4 s exposed).

Floors: irreducible = CPU sim ~4.7 + preprocessed 2.2 + K1 1.3 + interaction/tree2 0.4 + resident
prove_ex compute ~4–5 + FRI/PoW/decommit 2.7 + fused tree1 ~4–8 ≈ **~19–25 s hard floor**. So ~24 s is
near the physical floor of this architecture; below that needs cutting compute or CPU sim, not copies.

### Breakeven verdict (t_base < 26.5 s so base_wall beats 2^23-resident's rec-bound 2.52 h)

| config | t_base@2^24 band | clears < 26.5 s? |
|---|---|---|
| streaming as-is | ~51.5 | NO (2× over) |
| fuse-only | ~37–42 | NO |
| (ii)-only | ~28–36 | NO at the center; only the extreme-optimistic ~28 edge is close, still > 26.5 |
| **fuse + (ii)** | **~24–31** | **PLAUSIBLY YES** — the lower ~24–26 s part of the band clears; the upper part does not |

> **VERDICT Q3:** Neither fuse alone nor (ii) alone plausibly clears 26.5 s. **Only fuse + (ii)
> stacked has a realistic shot**, and even then only the optimistic half of its ~24–31 s band gets
> under breakeven. Call it **~50/50 on clearing 26.5 s with fuse+(ii)** — not a comfortable margin.
> The band's lower edge (~24 s) is close to the ~19–25 s architectural floor, so there is little room
> to spare, and the CPU sim (~4.7 s) + resident compute are hard limits neither optimization touches.

---

## Q4 — The trace-gen-feed question, re-examined (which framing is right)

OPTIMIZATION_PLAN's GPU MEASURED CURVE claims: *"trace-gen (memory-bound CPU) is the wall; 8 GPUs
STARVE (CPU feeds ~0.5 Mrows/s)"* → lever = trace-gen throughput.

**This must be split into your two framings, and they have DIFFERENT bottlenecks:**

**(a) Per-shard single-GPU t_base composition (this box's numbers).** From Q1, the single-shard t_base
is dominated by the **GPU/streaming commit** (tree1 27.7 s = 54% of t_base), NOT CPU sim. The genuinely
CPU-bound piece (build_rows) is **≤ ~4.7 s, ≤ ~9% of t_base**. So for a *single* base proof on *one*
GPU box, **"pivot to trace-gen" is a RED HERRING** — the CPU sim is a rounding error; the wall is the
streaming PCIe/serialization tax (fuse + (ii)). The `trace_gen_s` = 70%-of-t_base figure is REAL but is
mostly the GPU tree1 commit hiding inside a badly-named JSON field (Q1a), not CPU work.

**(b) Aggregate CPU-feed rate to keep 8 GPUs busy (the 8×A100 projection).** This is a DIFFERENT
machine and a DIFFERENT question. The a2-highgpu-8g projection measured (OPTIMIZATION_PLAN GPU MEASURED
CURVE): GPU_8g ≈ 73 Mrows/s but the 12-vCPU CPU on that box feeds only ~0.5 Mrows/s (up to ~1.2–2.8 in
the eta band). There, 8 GPUs genuinely STARVE because ONE small CPU must generate witness for all 8.
That is where "trace-gen is the wall" is TRUE — but it is an **aggregate-provisioning** statement about
the 8-GPU box's CPU:GPU ratio, NOT a statement that CPU sim dominates a single base proof.

> **VERDICT Q4: your framing (b) is right, framing (a) is the correct correction.**
> - For the **single-machine base-proof latency** (the t_base this analysis decomposes), CPU sim is
>   NOT the dominant lever — the streaming commit is. Dominant lever = **fuse + (ii)** (Q3).
> - "trace-gen is the wall" is only true in the **8-GPU aggregate throughput** projection, and even
>   there it is a CPU-*provisioning* problem (too few vCPUs per GPU), solvable by more/faster CPU cores
>   or CUDA trace-gen, NOT by touching the single-shard base path.
> - These were being conflated because `trace_gen_s` (JSON) reads as "trace gen" but is 76% GPU tree1
>   commit. **Do NOT pivot the single-machine effort to CPU trace-gen** — it buys ≤ ~4.7 s. Pivot to
>   trace-gen (CPU sim / CUDA sim) ONLY when scaling to a many-GPU box where the CPU:GPU feed ratio
>   binds.

---

## RECOMMENDATION — dominant lever

1. **For single-machine t_base:** the dominant lever is the **streaming commit tax**, addressed by
   **fuse (Q2) + (ii) async-overlap (ASYNC_OVERLAP_SCOPE)**, in that priority. Fuse is the simpler,
   lower-risk win (deletes build_leaves rehydrate outright); (ii) is needed on top to clear breakeven.
2. **Breakeven (< 26.5 s @2^24) is only plausibly reached by fuse + (ii) stacked, and even then only
   the optimistic ~24–26 s half of the band clears — ~50/50.** Do NOT bank 2^24 until a real build
   confirms t_base lands under 26.5 s; keep 2^23-resident in production meanwhile (matches the
   RECURSION_PLAN CURVE DECISION).
3. **CPU trace-gen is NOT the single-machine lever** (≤ ~9% of t_base) — it only becomes the wall in
   the 8-GPU aggregate projection where CPU:GPU provisioning binds.
4. **Before banking any projection, run the two sub-timers** (they are cheap, one 2^24 streaming run
   each): **T1** (five accumulators in the streamed producer loop — NTT / D2H / dehydrate-reclaim /
   build_leaves-H2D / absorb) confirms the fuse saving and the ~22 s serialization bucket; **T2**
   (H2D-vs-kernel split in rehydrate_owned/rehydrate_block/composition-tile) confirms the prove_ex
   rehydrate tax. The `build_elapsed` "shots simulated … in Xs" line (main.rs:2300, already printed)
   pins the CPU sim. Every band above that rests on an inferred split is flagged NEEDS-SUB-TIMER, not
   presented as measured.
