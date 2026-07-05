# Box validation log — gate_air base-shard ceiling (2^24/2^25) + streaming

Durable results + terse run outcomes for the base-shard ceiling-lift work. **Design, decisions, and implemented work
(marked done) live in `../RECURSION_PLAN.md`; this file is measurements + run outcomes + operational facts only —
no maintained status prose.** Newest first.

## ★★ Decoupling carry-shape fix — RED→GREEN validated (2026-07-06, CPU-VM stwo-development-server)
- **Bug (correctness, not soundness):** under leaf<->node decoupling (leaf lifting 2^24, node lifting 2^25; R1!=R2)
  the k-ary fold (FOLD_ARITY=8) carried the level-0 leaf remainder UP UNCHANGED, so for every non-power-of-8 N a
  lifting-24 leaf reached a lifting-25 node-level fold. Fails at the FOLD PROVER (not the unpacker, not the final verify).
- **RED (pre-fix, box's frozen decoupled copy), N=9:** panic at `stark_verifier/merkle.rs:55`
  `assertion left==right failed  left:24 right:25`, EXIT=101. Confirmed decoupled regime (`R1!=R2 collapse=false`,
  leaf 2^21->lift24, node 2^22->lift25). The 9 leaves + the arity-8 leaf node folded fine (`t_node(h=1)`); the root
  fold (h2, verifying the carried leaf) panicked.
- **Fix (two-phase topology):** level 0 consumes ALL leaves into height-1 leaf-verifying nodes (a short arity-2..=8 node
  absorbs `N % k`; the `r==1` case splits the trailing `k+1` leaves as `k-1,2`), then the group+carry loop folds NODES
  only. No leaf is ever above height 1. Roots selected by public (height, arity): R1/R2 for full-8, recomputed for shorts.
- **GREEN (fix synced+rebuilt), sequential `GATE_AIR_FOLD`, k2000 fixture (2^23 leaves), `SHARD_SHOTS=1`, `POOL_THREADS=8`:**
  all EXIT=0, 0 panics, `fold OK` + `root verification OK` + `self-verify (fold+root) OK`.
  - N=9  (2 lvl) `3963b5c0…` | N=35 (2 lvl) `d8020ef7…` | N=64 (2 lvl) `8a4b96ba…` | N=69 (3 lvl) `24bfbfca…`
  - N=9 log shows the new topology: `t_short_node(arity=7,h=1)` + `(arity=2,h=1)` -> root `(arity=2,h=2)`.
- **Streaming == sequential (byte-identical `recursion_fingerprint`), `GATE_AIR_FOLD`+`GATE_AIR_PIPELINE`:**
  N=9 streaming `3963b5c0…` == sequential ✓ ; N=64 streaming `8a4b96ba…` == sequential ✓
  (N=64's phase-B root is arity 8 -> confirms dispatch parity: streaming routes the full-8 root through the same
  `prove_short_node` path as sequential).
- **Closes:** the carry cases that panicked now prove+self-verify; clean powers (64) no regression; the production
  streaming/overlap path produces the identical proof as sequential. Commits: proving-utils `0ab871b` (amended),
  grover-tax-v02 `790aa71` (amended). Unpacker-vs-prover soundness review spun separately.

## ★★ BANKED RESULT — best verified single-machine time (2026-07-03, LOCAL, extrapolate_8g.py conventions)
- **2^24 steady-state t_base = 20.0s** (fold-measured: 4-shard fold shards 1–3 = 19.99/20.12/20.02s; shard0=31.9s pays
  the one-time page-lock, amortized). Byte-identical (2^24 fp `44fec71ad82a3e6c3f9c441d2a416521dfe082c7ddde515bf44534ec44ecabbb`
  == oracle).
- **-> 2.49× vs SP1 Tanuj curve @k=2000** (per-k: k=1 1.41× | k=10 1.50× | k=100 2.33× | k=1000 2.48× | k=2000 2.49×).
  Base-bound at every k. This is the PRECISE number replacing the "~2.5×" estimate.
- **BEATS 2^23-resident (3.3×)** — 2^24-streaming is now the better config and the best verified single-machine result.
- **Still >1× (SLOWER than SP1):** 2.49× does NOT beat the Tanuj curve outright (that needs <1×).
- **⚠ CAVEAT — 2.49× is MODELED, assumes base‖recursion OVERLAP; NOT measured end-to-end.** The `max(base_wall, rec_wall)`
  form assumes GPU base proving and CPU recursion run CONCURRENTLY (fold-pipelined), and `rec_wall=N·(t_leaf+t_node)/16`
  assumes leaves+nodes share one 16-wide pool (simultaneous across the tree). If base and recursion do NOT overlap it is
  ADDITIVE (base_wall+rec_wall): at k=2000, N≈2741, base_wall≈6860s + rec_wall≈4539s ⇒ **~4.14×**, not 2.49×. So the
  overlap assumption alone is worth **~1.65×** on the multiple. Because steady-state t_base fell to 20s, **recursion is now
  ~66% of the base wall** (was ~38% at t_base=35s) — a lot to hide, so imperfect overlap pushes the real number toward ~4×.
  ⇒ **OVERLAP is the correct model, and MEASURED 2026-07-03 CONFIRMS it.** base runs on the GPU and recursion on the CPU —
  SEPARATE resources — so in a deployed pipeline they overlap; `max(base_wall, rec_wall)` is right, giving **2.49×**. The
  2-shard 2^24 fold + GPU-util timeline showed the current SINGLE-PROCESS harness runs base(GPU) then recursion(CPU)
  serially (GPU 100% during base, flat 0% for the ~42s recursion phase) — but that is a MEASUREMENT/HARNESS artifact (one
  process does one thing at a time), NOT a hardware limit; realizing the overlap in a deployed run is fold-pipelining (impl
  work). The measurement's value: it CONFIRMS the overlap projection — measured recursion anchors t_leaf≈9.8s / t_node=8.8s
  (even CHEAPER than the documented 15.8/10.7), and the per-k split is base-bound at EVERY k (base_wall≈2×rec_wall), so
  recursion fully hides under base ⇒ 2.49× holds robustly (with either anchor set). Split calc: scratchpad/split.py.
  (Serial single-process harness, reference only: ~3.65× @k2000 — the fold-pipelining headroom, GPU idle ~42s/shard-batch.)
- **Method (apples-to-apples with the prior 4.4× RECOMPUTED CURVE below):** reused its EXACT anchors — shard=2^24,
  t_leaf/t_node=15.8/10.7 (c4 K=16 DDR5-optimistic), P_8g=16, tail=6.0s, SP1 8×A100 curve; changed ONLY t_base 35.2→20.0.
  Model `T=max(⌈N/8⌉·t_base, N·(t_leaf+t_node)/16)+tail`. Cross-check: same script with t_base=35.2 reproduces the prior
  2.23/2.50/4.07/4.37/4.39× exactly. Command: `python3 recompute.py` (extrapolate_8g conventions, SHARD_LOG=24; the two
  swapped-t_base runs).

## ★★ Fold-pipeline validated: byte-id + scaling fix + overlap measured (2026-07-03)
### RESOLVES the overlap caveat above: the fold-pipeline EXISTS, is byte-identical, SCALES (flat host mem), and OVERLAPS (measured). The 2.49× overlap model is now MEASUREMENT-BACKED, not assumed.
- **The fold-pipeline ALREADY EXISTED (GATE_AIR_PIPELINE):** the GPU producer streams base proofs over a depth-1
  channel; the CPU consumer wraps leaves + runs a STATIC-tree worker-pool fold (`build_fold_topology`, known N, NO
  dynamic unpacker). Topology byte-identity is unit-proven (N<=130). So the concurrency the 2.49× model assumes is
  implemented, not hypothetical.
- **BYTE-IDENTITY:** N=2 pipeline `recursion_fingerprint` == sequential ==
  `8b4115f2c6ffac5fa8f06ab07f2f701bbbe36fee31204e71d2fd82745bc9503e`. The pipeline is byte-identical to the sequential
  path.
- **SCALABILITY FIX (#1 — dead build_rows in fold mode):** the fold OOM'd the host at N>=4 (exit=137, >83GB) due to a
  DEAD top-level `build_rows` over ALL samples (O(N·samples): ~6.5–13GB @2^24, ~TB at real N) held on `main()`'s frame
  even though it is UNUSED in the fold path (each shard rebuilds its own rows). Fix: skip that `build_rows` in fold mode
  (re-derive the shape scalars identically), in `main.rs`.
  - RESULT (measured, all REAL_EXIT=0): N=2 wall 104s / peak_host 42GB ; N=4 wall 152s / peak_host 46GB (was OOM) ;
    N=8 wall 253s / peak_host 47GB.
  - Peak host is FLAT ~42–47GB across N=2/4/8 ⇒ witness is now O(shots_per_shard), NOT O(N) ⇒ real-benchmark large N
    (1128–2256 shards) is FEASIBLE on the 83GB host.
- **OVERLAP MEASURED (N=8, 1-GPU box):** Sum t_base ~189s ; Sum recursion(leaf+node) ~136s ; serial would be ~325s ;
  measured wall = 253s ⇒ **~22% faster than serial, ~70% of the ideal overlap realized.** Recursion largely hides under
  base.
  - Caveat: the 1-GPU box has `n_pools=1` (a SINGLE fold worker); a2-8g has 16 ⇒ a2-8g overlap is BETTER, so this 1-GPU
    number is a CONSERVATIVE LOWER BOUND. The residual gap to full overlap is the pipeline TAIL (the last shard's
    recursion can't hide), which shrinks as a fraction of the wall at large N.
- **Per-shard steady anchors reconfirmed:** t_base ~22s (shards 1+; shard0 ~30s pays the one-time page-lock),
  t_leaf ~8–11s, t_node ~8–11s.

## ★ 2^25 shards — PARKED (2026-07-03)
- **Blocked:** device tree2-commit OOM persists after 5 fix attempts (streaming, LOWMEM, boundary-trim, cuMemFree
  free_after_k4, pool-alloc). The u32-overflow fix advanced past K1 but tree2 still OOMs (128 MiB allocs fail; d_cols
  apparently not freed for tree2). The 2^25 fold was also OOM-killed (exit=137) early.
- **Strategic (why deprioritized, not just blocked):** the curve is BASE-BOUND, so bigger shards buy only modest
  per-shard fixed-overhead amortization (~10–20%, est ~2.1× @k2000), NOT enough to beat SP1. The ~2.5× gap to SP1 is
  ALGORITHMIC (8×A100 vs 8×A100), not shard-size. Cracking 2^25 needs probe-driven investigation.
- **The real levers to challenge SP1 (<1×) are elsewhere** (GPU port of trace-gen + lifted-Merkle, or cutting recursion
  cost) — reference only.

## ★ Box session: nsys root-cause + fold amortization — dehydrate is one-time page-lock (2026-07-03)
### THE dehydrate cost is a ONE-TIME pinned-buffer page-lock that AMORTIZES across shards — NOT a per-copy defect. Steady-state per-2^24-shard t_base = ~20s (not the single-shot 35s).
- **nsys profile of the 2^24 batched producer (cuda_api_sum):** `cudaHostAlloc = 46.1% / 11.78 s / 188 calls / avg
  62.7 ms each`. The ~11.6s "dehydrate_d2h" is NOT the D2H copy — it is 188 per-column cudaHostAlloc PAGE-LOCKS (128 MiB
  each). The actual D2H copy is fast (~12 GB/s, matching the microbench). (Also visible: cudaStreamSynchronize 34%, but
  the dominant single cost is the page-lock.)
- **The page-lock is a ONE-TIME cost:** the pinned pool is process-global and RECYCLES buffers across proofs/shards
  (clear_stash → free-list), so the page-lock is paid on the FIRST shard only; later shards reuse the already-pinned
  buffers.
- **Fold measurement (GATE_AIR_FOLD, 2^24 shards, SHARD_SHOTS=4, batched):**
  - 2-shard: t_base[shard0]=31.81s, t_base[shard1]=19.81s.
  - 4-shard: t_base[shard0]=31.87s, [1]=19.99s, [2]=20.12s, [3]=20.02s. t_leaf ~8.4–10.6s.
  - ⇒ STEADY-STATE per-2^24-shard t_base = **~20.0s** (stable shards 1–3); shard0 pays ~12s one-time page-lock.
  - (Caveat: the 4-shard run exit=137 OOM at the END, in the multiverifier fold/aggregation AFTER all base+leaf timings
    completed — a SEPARATE 4-shard aggregation memory issue to investigate; base/leaf t_base numbers are clean.)
- **IMPLICATION:** the single-shot t_base (35s → 4.4×) OVERSTATED the real cost. Steady-state 2^24 t_base ~20s →
  **~2.5× vs SP1** (approx; needs curve.py) — BEATS 2^23-resident (3.3×). NOTE ~2.5× still means 2.5× SLOWER than SP1
  (not beating the Tanuj curve, which needs <1×). **2^25 shards** (bigger → fewer recursion nodes → lower multiple) are
  the real SP1-challenge config but are currently BLOCKED by a pool-alloc crash (exit=134), fix in progress.
- **REFRAMES the earlier D2H-microbench "pageable-copy defect" story (below):** the copy was never the bottleneck and is
  NOT effectively pageable — the ~2 GB/s "dehydrate_d2h" wall-time was the 188 one-time page-locks charged to that span.
  The prior FREE_ON_STREAM0 / Option-A-batched / "~2 GB/s fixable-copy defect" framing targeted the wrong thing; the
  copy is fast (~12 GB/s) and the page-lock amortizes. Keep FREE_ON_STREAM0 and BATCHED as default-OFF flags (harmless)
  but note they are NOT the lever.

## ★ Box session: D2H bandwidth microbenchmark (2026-07-03) — SUPERSEDED framing; see nsys section above
### NOTE (CORRECTED 2026-07-03 by the nsys+fold section above): the "~2 GB/s = effectively PAGEABLE copy defect" reading of this microbench is WRONG. The ~2 GB/s dehydrate_d2h wall-time is 188 ONE-TIME cudaHostAlloc page-locks charged to that span, not a slow copy; the copy itself is ~12 GB/s. The page-lock is one-time and amortizes across shards (steady-state 2^24 t_base ~20s). The microbench numbers below are still valid; only the "prover copies to pageable" interpretation is retracted.
### The prover's ~2 GB/s dehydrate is NOT a hardware ceiling — it is effectively copying to PAGEABLE memory (a FIXABLE prover-path defect).
- **Box health:** A100-SXM4-40GB, PCIe gen4 x16 (current == max), single NUMA node (CPU affinity 0-11). Hardware healthy.
- **cuda-samples bandwidthTest:** PINNED D2H 12.88 GB/s ; PAGEABLE D2H 3.61 GB/s.
- **Custom microbench (2 GiB payload, 16 x 128 MiB):**
  - A) single 2 GiB pinned one-shot = 13.20 GB/s (ceiling)
  - B) pinned 16x128MiB async, 1 final sync = 13.20 GB/s
  - C) pinned 16x128MiB sync-each = 13.19 GB/s
  - D) pageable 16x128MiB async = 1.28 GB/s
  - E) pool alloc + memset + async D2H to pinned + freeAsync (the prover's EXACT pattern) = 12.31 GB/s
- **INTERPRETATION:** true pinned D2H = ~13 GB/s, and the prover's own pool+async+pinned PATTERN (case E) hits 12.3 GB/s.
  But the prover's ACTUAL dehydrate measures ~2 GB/s (11.6s/24GiB @2^24; 23.2s/48GiB @2^25) — squarely in the PAGEABLE
  range (1.3-3.6 GB/s, cases D/pageable-bandwidthTest). So the prover's dehydrate is effectively copying to PAGEABLE
  memory despite the pinned pool — a FIXABLE prover-path defect, NOT a hardware ceiling.
- **IMPACT:** fixing it is a ~6x dehydrate win (11.6 -> ~2s @2^24) -> tree1 commit ~17 -> ~7s -> t_base ~35 -> ~25s ->
  plausibly BEATS 2^23-resident's 3.3x and makes 2^24/2^25 streaming competitive. Root-cause + fix pending.

## ★ Box session: Option A batched dehydrate + 2^25 timing attempt (2026-07-03)
### Option A (GATE_AIR_ASYNC_STASH_BATCHED) is a NO-OP for dehydrate — refutes the earlier serialization theory.
2^24 A/B (samples=4, all fp == oracle `44fec71ad82a3e6c3f9c441d2a416521dfe082c7ddde515bf44534ec44ecabbb`):
- async (B1/B2):  dehydrate_d2h 11.628s | tree1_commit 16.799s | wall 42.2s | trace_gen_s 26.14 | prove_s 8.56
- batched (Opt A): dehydrate_d2h 11.626s | tree1_commit 16.789s | wall 41.6s | trace_gen_s 25.59 | prove_s 8.57

Flag engaged (dehydrate_reclaim 0.021 → 0.000s) but dehydrate_d2h IDENTICAL. The batched timer measures the true
pipeline drain, so this REFUTES the "per-column host serialization" diagnosis: batching pageable-speed copies can't
help. WHY it can't help (CORRECTED 2026-07-03 by the D2H microbench above): the copies are running at PAGEABLE speed
(~2 GB/s, 24 GiB / 11.6s) regardless, so scheduling has nothing to hide — this is a prover-path defect (effectively
copying to PAGEABLE memory despite the pinned pool), NOT a REAL hardware bandwidth wall (true pinned D2H = ~13 GB/s
measured; the prover's own pool+async+pinned PATTERN hits 12.3 GB/s in microbench case E). The fix is to make the
copies ACTUALLY pinned (a ~6x win), not to schedule them differently. Batched does NOT beat 3.3× (t_base ~35s → ~4.4×,
unchanged) — but that is because the copies are pageable-speed, a FIXABLE defect, not a ceiling.
### 2^25 STILL OOMs at tree2 — free_after_k4 did NOT resolve it.
Both p25_ref (LOWMEM+BOUNDARY_TRIM+ASYNC_STASH) and p25_batched (…+BATCHED): exit=1, peak host ~40 GB (host FINE).
Sequence: preprocessed 4.7s, K1 1.2s, tree1 commit 33.5s (dehydrate_d2h 23.2s == ~2 GB/s for 48 GiB, batched identical),
"interaction witness gen+sumcheck 0.29s" printed, then repeated `Failed to allocate 134217728 bytes (128 MiB) from
pool: out of memory` = TREE2 commit. So the host ceiling is solved (LOWMEM) but the DEVICE tree2 OOM persists: freeing
d_cols via cuMemFree returns 24 GB to the driver but tree2's cudaMallocFromPoolAsync pool cannot reuse it (residual
pool/fragmentation issue flagged by the fix author). No completing 2^25, no t_base@2^25 yet.
### KEY FINDING (CORRECTED 2026-07-03): streaming's dehydrate runs at ~2 GB/s (both 2^24 and 2^25) — but this is
EFFECTIVELY PAGEABLE-speed copying, a FIXABLE prover-path defect, NOT a hardware wall. The D2H microbench (above)
measured true pinned D2H = ~13 GB/s and the prover's own pool+async+pinned pattern at 12.3 GB/s, so ~2 GB/s is in the
pageable range (1.3-3.6 GB/s). No SCHEDULING change touches it (async/batched can't speed up pageable copies) — but the
FIX (make the copies actually pinned) is a ~6x win. This is WHY streaming (2^24/2^25) currently loses to 2^23-resident;
root-cause + fix pending, plausibly beats 3.3× once fixed.

## ★ Box session: B1/B2 async + d_main streaming, A100-40GB, FX_BIG k1000-n9024 (2026-07-03)
Measured B1/B2 (reusable pinned pool + async double-buffer + OODS no-memset fix) and the 2^25 d_main-streaming/LOWMEM path.
- **Byte-identity — ALL PASS.** 2^22 (baseline, ASYNC_STASH, STREAM_MAIN, ASYNC+STREAM_MAIN) all ==
  oracle `ab1e75b53d49645c09982ed5c890d8880366b0197b554ea5300dcd0f66838ba4`. 2^24 (baseline, PIN_STASH, ASYNC_STASH,
  STREAM_MAIN) all == oracle `44fec71ad82a3e6c3f9c441d2a416521dfe082c7ddde515bf44534ec44ecabbb`.
- **2^24 A/B (samples=4, GPU, k1000-n9024).** Columns: variant | wall | tree1_commit | prove_ex | dehydrate_d2h |
  oods_h2d | oods_kernel | trace_gen_s | prove_s
  - baseline(pageable): 58.3s | 27.49 | 17.49 | 19.88 | 11.82 |  7.60 | 36.34 | 17.49
  - B0 PIN_STASH:       43.6s | 17.89 |  9.34 | 11.69 | 10.93 | 12.21 | 26.74 |  9.34  (dehydrate_reclaim 1.925s = per-col page-lock)
  - B1/B2 ASYNC_STASH:  42.5s | 17.03 |  9.09 | 11.79 |  4.74 |  2.24 | 25.88 |  9.09  (dehydrate_reclaim 0.021s = page-lock GONE via reusable pinned pool)
  - STREAM_MAIN @2^24:  71.4s | 27.39 | 17.46 | 19.82 | 11.49 |  8.31 | 49.38 | 17.46  (overhead: dehydrate d_main to host + rehydrate for K4; only for 2^25 feasibility, not a 2^24 speedup)
  Note: the [T2] oods_h2d/oods_kernel are SUMMED across rayon workers (parallel), so they exceed prove_ex wall; B1/B2's
  OODS no-memset fix cut aggregate oods work ~23s→7s but wall impact is small (OODS was already parallel).
- **VERDICT:** B1/B2 wall ≈ B0 (42.5 vs 43.6s); t_base ≈ trace_gen_s + prove_s ≈ 35.0s → ~4.4× SP1 @k2000. B1/B2 does
  NOT beat 2^23-resident (3.3×). OODS fix + page-lock removal are byte-identical wins but wall-neutral because
  dehydrate_d2h (~11.8s) dominates and is not hidden by the async double-buffer.
- **ANOMALY (now DIAGNOSED 2026-07-03, see D2H microbench section):** dehydrate_d2h is only ~1-2 GB/s even though the
  stash is nominally pinned (12-24 GB moved in ~11.8s @2^24 / 23.4s @2^25) — far below pinned PCIe (~13 GB/s measured).
  RESOLUTION: the copy is EFFECTIVELY PAGEABLE despite the pinned pool (a FIXABLE prover-path defect), NOT a hardware
  wall — a ~6x t_base lever; root-cause + fix pending.
- **2^25 (LOWMEM host fix):** GATE_AIR_STREAM_MAIN_LOWMEM dropped peak HOST RAM 72→46 GB (no host OOM) — host ceiling
  SOLVED. But run still exit=1: DEVICE OOM at tree2 commit. PROBE3 (tree1→K4): device free=834 MiB (d_main 24 GB
  resident + pool ~15 GB). Interaction fit (d_inter OK via boundary-trim), then tree2's 128 MiB per-col allocs failed.
  Root cause: LOWMEM keeps d_main resident and the `drop(main_k1.take())` after K4 did NOT free the 24 GB d_cols
  CudaSlice (Resident was a marker, not owner) → d_main survives into tree2. Fix in progress (move d_cols ownership so
  it frees before tree2); one more box session to confirm 2^25 completes.

## ★ PART B0 — pinned host stash (flag GATE_AIR_PIN_STASH), A100-40GB, FX_BIG k1000-n9024 (2026-07-02)
Measured A/B of the first pinning increment (per-column cudaHostAlloc of the dehydrate stash).
- **Byte-identity PASS:** 2^24 pin-OFF fp == pin-ON fp == `44fec71ad82a3e6c3f9c441d2a416521dfe082c7ddde515bf44534ec44ecabbb`;
  2^22 pin-ON fp == oracle `ab1e75b53d49645c09982ed5c890d8880366b0197b554ea5300dcd0f66838ba4`. Pinning is correctness-safe.
- **2^24 A/B (pin OFF -> pin ON):** wall 57.2s -> 42.6s (-25%); tree1 commit 27.06 -> 17.77s; prove_ex 16.9 -> 8.8s.
  Copy sub-timers: dehydrate_d2h 19.57 -> 11.64s; build_leaves_h2d 5.35 -> 2.09s; quotient_h2d 2.73 -> 1.08s;
  oods_h2d 10.69 -> 10.72s (NO CHANGE); dehydrate_reclaim 0.004 -> 1.915s.
- **Interpretation:** B0 UNDER-delivers vs a pinned copy's ceiling — dehydrate only ~40% faster (not ~10×) because
  per-column cudaHostAlloc RE-PAGE-LOCKS ~12 GB every run (the reclaim delta 0.004 -> 1.915s is that page-lock cost).
  The OODS path did NOT benefit (oods_h2d unchanged) — OODS rehydrate is not routed through the pinned buffers. Both
  are headroom for B1/B2 (reusable pinned double-buffer + async overlap).

## ★ RECOMPUTED CURVE — 2^24 + Part B0 pin-ON (2026-07-02, LOCAL, extrapolate_8g.py conventions)
- **t_base used = 35.2s** (definition: `trace_gen_s + prove_s`, the prior fold per-shard convention — the SAME t_base
  the 6.4x curve used). This is the DIRECT pin-ON measurement (trace_gen_s 26.4 + prove_s 8.8) from the 2^24 pin-ON run
  — HIGHER-CONFIDENCE than the earlier value derived-from-51.5s (34.1s = 51.5 as-is − 9.29 tree1 − 8.1 prove_ex; the
  direct 35.2s and derived 34.1s agree to ~3%, cross-validating the derivation).
- **Anchors REUSED (no new anchors measured):** recursion t_leaf/t_node = c4 K=16 sweep 15.8/10.7 (DDR5-optimistic)
  and a2 DDR4-adjusted 23.7/16.0; P_8g=16; tail=6s. SP1 8×A100 curve unchanged.
- **Ratio vs SP1 8×A100 (base-bound at every k>=1):** k=1 2.23× | k=10 2.50× | k=100 4.07× | k=1000 4.37× | k=2000 4.39×.
  (a2 DDR4-adj identical at k where base binds, which is all of them.)
- **COMPARISON:** prior 2^24-streaming-as-is = **6.4×** @k=2000 -> B0 pin-ON = **4.4×** (recovers ~1/3 of the streaming
  PCIe tax). Still WORSE than 2^23-resident's **3.3×**, and above the async-overlap EST target of **2.0×**. So B0 alone
  does not yet make 2^24 bankable; the remaining PCIe (dehydrate re-page-lock + un-routed OODS + no overlap) needs
  B1/B2 (reusable pinned double-buffer + async). Base still binds at every k -> next lever after B1/B2 = base throughput.

## ★ 2^25 OOM VERDICT — CAPACITY-bound (2026-07-02, corrects the fragmentation story)
- **VERDICT:** 2^25 is CAPACITY-bound — NOT allocator hoarding, NOT fragmentation. d_main (188 cols × 128 MiB ≈ ~24 GB)
  is genuinely-live NON-POOL memory resident through K4; only 4.5 GB free after K1 (vs 22.2 GB free at 2^24).
- **Probe runs (streaming ON: GATE_AIR_STREAM_COMMIT=1 GATE_AIR_FUSED_INTERP=1):**
  - trim-off -> OOM at `alloc inter` (d_inter 3 GB).
  - trim-after-K1 (frees 5.6 GB pool hoard) -> STILL OOM at `alloc inter` (tree1 refills the pool).
  - option-0 boundary-trim (GATE_AIR_BOUNDARY_TRIM, one-shot tree1->K4) -> CLEARS d_inter (interaction runs) but dies
    ONE PHASE LATER at tree2 commit (ifft.cu:839).
- **option-0 status (CORRECTED):** option-0 is CORRECTLY-PLACED and NOT SUFFICIENT on its own — it runs clean, clears
  d_inter, and lets interaction complete; 2^25 then OOMs at tree2. The true cause is the ~24 GB d_main capacity pin,
  not option-0's placement. (trim-after-K1 is mis-placed and can be dropped.)
- **Fix direction (chosen for 2^25):** STREAM d_main too — dehydrate the main columns to host after consumption,
  rehydrate per-column for K4 — frees ~24 GB. option-0 may become UNNECESSARY once d_main streaming frees ~24 GB; keep
  it as a CANDIDATE companion PENDING the next box run (do not assume it is needed).
- **Diagnostic probes added (default-off / read-only, KEEP):** utils.cu `cuda_mem_probe` (~:741); bindings.rs (~:173);
  mod.rs (~:27); gate-air-leaf/src/main.rs PROBE1 (~:3162), GATE_AIR_TRIM_AFTER_K1 (~:3166), PROBE2 (~:3210).

## RUN OUTCOMES (append-only, terse)
- 2026-07-02  STREAM only            @2^22  PASS  fp==oracle
- 2026-07-02  FUSED_INTERP only      @2^22  PASS  fp==oracle
- 2026-07-02  run1 STREAM (pre-A)    @2^22  FAIL  quotient.rs:300 not-host-staged
- 2026-07-02  run2 (B) uniform-stage @2^22  FAIL  quotient.rs:300 (tree0/tree2 resident too) → full-(A)
- 2026-07-02  run3 full-(A)          @2^22  FAIL  ConstraintsNotSatisfied (comp 1.63s/quot 0.70s/TOTAL 3.48s); crash gone
- 2026-07-02  run4 +unbias .cu       @2^22  FAIL  same (confirmed .cu recompiled)
- 2026-07-02  BOTH + single-tile     @2^22  FAIL  same (GATE_AIR_TILE_ROWS=2^24, one tile, tile_start≡0)
- 2026-07-02  +owns_memory alias fix  @2^22  FAIL  STILL ConstraintsNotSatisfied, BUT timings dropped
    (comp 1.63→0.97s, quot 0.70→0.37s, tree1 4.48→3.70s, TOTAL 3.48→2.29s) ⇒ aliasing LIKELY fixed (no more wrongful
    H2D of falsely-staged resident tree0/tree2 cols), but a SECOND independent staged-path correctness bug remains.
    The A/B mismatch (committed-composition-OODS vs sampled-trace-recombination, prover/mod.rs:186) persists.
- **Diagnostic conclusion:** each flag ALONE passes; BOTH fail; BOTH+single-tile STILL fails ⇒ staging only happens
  with both flags, and single-tile failing RULES OUT tiling boundaries/biasing → the bug is in the STAGED DATA PATH
  (dehydrate/stash/rehydrate/keying), not row-tiling. This localized the aliasing root cause (see plan).
- 2026-07-02  DIAG_FULL_REHYDRATE bisect @2^22 — DECISIVE. Counters: `dehydrate_column: 188 calls, 94 EARLY-RETURNED
  (colliding/reused key), 94 stashed`; `composition dispatch: tree0 0/4 staged, tree1 94/188 staged`. Run A (baseline)
  FAIL; Run B (force composition to read staged cols via full rehydrate_owned, bypassing per-block H2D) STILL FAIL.
  ⇒ (1) the cudaMemPoolTrimTo(pool,0) address reuse is massively live — exactly HALF (94/188) the tree1 large cols get
  a just-freed sibling's address and early-return (stay resident). (2) Full-rehydrate failing = bug is in the DATA
  (stashed bytes and/or resident early-return cols), NOT the kernel per-block staged-read mechanics. The owns_memory
  routing fix sends each col to the right SOURCE but the collision still corrupts the DATA. Fix direction = stable
  per-column identity for the stash (not the reusable device_ptr) OR eliminate the reuse — NOT a kernel-read tweak.
- 2026-07-02  fix (A) alloc-next-before-free-prev  @2^22  FAIL — counters UNCHANGED (188 calls, 94 EARLY-RETURNED,
    94 stashed), still ConstraintsNotSatisfied. Why A failed: deferring the free by ONE iteration only stops col N+1
    colliding with col N; but the stash keeps EVERY prior address as a key and the pool reuses freed addresses, so
    col N+2 reuses col N's address (freed 1 iter before N+2 allocs) and collides with col N's still-present stash
    entry → still every-other-col (94/94). ⇒ the collision is FUNDAMENTAL to keying the stash by a reusable
    device_ptr while freeing buffers mid-loop (required for 2^25 memory). Robust fix = STABLE per-column stash
    identity (monotonic id, not device_ptr) — the design fork; routed to user 2026-07-02.
- 2026-07-02  sentinel stash-key fix  @2^22  COLLISION ELIMINATED but NEW crash. Counters: `188 calls, 0
    EARLY-RETURNED, 188 stashed`; `tree1 188/188 staged` — the sentinel (1<<63|id, disjoint from real VAs, written
    into staged col.device_ptr) killed the reuse collision as designed. BUT new hard crash: `CUDA error at
    barycentric.cu:241 cudaErrorIllegalAddress` during 3_oods_eval_at_point (crashes BEFORE the constraints check, so
    a DISTINCT bug from the old ConstraintsNotSatisfied). Cause hypothesis: with all 188 now staged (vs 94 before), OODS
    barycentric_eval_at_point rehydrates EVERY col; one of the 94 newly-staged cols' OODS path hits an illegal address
    — either a rehydrate-at-full-scale issue or a site that derefs the sentinel device_ptr that the audit missed. tree1
    commit 3.7→7.0s (all 188 rehydrate through build_leaves now — expected). Agent adc5cfb6 investigating barycentric.cu:241.
- 2026-07-02  thread-local→global stash fix  — ★ CORRECTNESS CLOSED + CEILING LIFTED TO 2^24. Root cause of the
    barycentric crash: HOST_STASH was thread_local!, but OODS eval_at_points runs on rayon WORKER threads (par_map_cols,
    parallel enabled by cuda) → worker's stash empty → is_staged false → sentinel device_ptr dereferenced → illegal
    address. Fix: HOST_STASH → static Mutex<HashMap> (process-global) + AtomicUsize id/counters. Battery results:
    [A] 2^22 PASS tiled fp==oracle ab1e75b5…; [B] 2^23 tiled==resident f304b5ee…; [D] 2^24 exit=0 NO OOM fp=44fec71a
    (prove TOTAL 17.0s: tree1 27.7s, comp 5.9s, OODS 5.6s, quot 2.8s; wall 58s incl one-time). [E] 2^25 OOM — but NOT
    at tree1 (that COMPLETED at 55.5s, streaming fixed the original tree1 OOM); OOM'd LATER at `alloc inter`
    (DriverError CUDA_ERROR_OUT_OF_MEMORY) = interaction/tree2 gen on top of the resident #1 d_cols (~23.5 GB floor).
    ⇒ streaming base-proof CORRECTNESS is solved; WORKING CEILING = 2^24 (fits + byte-correct); 2^25 has a NEW, LATER
    blocker (interaction-phase memory + the d_cols floor), distinct from the tree1 OOM we fixed.

## ★ MEASUREMENT MATRIX RESULTS (2026-07-02) — box now TERMINATED. OVERTURNS the sync-barrier theory.
8-run matrix (build + 2^22/2^24/2^25 × {notrim=Part A, RECLAIM_TRIM=per-col, BOUNDARY_TRIM=option-0}); full log was
`/tmp/measure_all.log`. Key results:
- **Byte-identity ✓:** 2^22 default AND BOUNDARY_TRIM both fp == oracle ab1e75b5… (option-0 byte-safe).
- **2^24 [T1] decomposition (THE correction):** tree1 commit ≈ 27.1s in ALL THREE modes — Part A recovered ~NOTHING.
  `[T1] ntt 0.9 | dehydrate_d2h 19.7 | dehydrate_reclaim 0.003 | build_leaves_h2d 5.4 | absorb 1.1`. The reclaim/sync/trim
  was NEVER the cost (0.003s notrim / 0.118s per-col trim). The real tax is the PAGEABLE BLOCKING **dehydrate_d2h copy
  = 19.7s** (~25 GB D2H ≈ ~1.3 GB/s — slow because pageable + per-column + blocking) + build_leaves_h2d 5.4s.
  `[T2] oods_h2d 10.8 | oods_kernel 7.4 | quotient_h2d 2.7`. So the streaming tax = the actual PCIe COPIES, not sync.
- **2^25 (RESOLVED — see the "2^25 OOM VERDICT" section above):** CAPACITY-bound on the ~24 GB d_main pin, NOT
  fragmentation. notrim → OOM at `alloc inter`; trim-after-K1 → STILL OOM at `alloc inter` (tree1 refills the pool);
  option-0 boundary-trim → CLEARS d_inter (interaction runs) but OOMs ONE PHASE LATER at tree2 commit (ifft.cu:839).
  ⇒ option-0 is correctly-placed + necessary but NOT sufficient; the fix is to stream d_main too. (The earlier
  "option-0 test was contaminated / must re-test isolated" note is SUPERSEDED — the isolated re-run above gives the
  clean tree2-commit signature.)

**CORRECTED LEVERS (supersede the earlier sync-barrier / fragmentation write-ups):**
1. Part A (reclaim without per-col sync) = PERF NO-OP — reclaim was ~0. Drop it as a lever (harmless to keep). Part A
   is NOT the async lever; the real lever is Part B (pinning + async).
2. **(ii) Part B (pinned + async D2H/H2D) is THE lever** — now MEASURED at B0: pin-ON @2^24 recovers -25% wall / t_base
   51.5->35.2s direct (6.4×->4.4× @k2000). B0 is only the first increment (per-col re-page-lock + un-routed OODS + no overlap
   remain); B1/B2 (reusable pinned double-buffer + async overlap) target the rest toward the ~2.0× EST.
3. **2^25 OOM RESOLVED:** CAPACITY-bound (~24 GB d_main), NOT fragmentation and NOT tree1 commit. option-0 clears the
   interaction phase but 2^25 then OOMs at tree2 commit; the fix direction is streaming d_main. (The
   OOM_2P25_INTERACTION_ANALYSIS.md fragmentation diagnosis is SUPERSEDED by the capacity verdict above.)
Flags in the binary (all compile-clean, synced, byte-safe): default per-col NOTRIM (Part A); GATE_AIR_RECLAIM_TRIM=1
(per-col trim); GATE_AIR_BOUNDARY_TRIM=1 (option-0 one-shot trim at main.rs:3244). Sub-timers GATE_AIR_T1_TIMERS/PROVE_EX_TIMERS.
Agent `adc5cfb6013af84d0` owns the streaming/stash/reclaim code. NOTE: the earlier "~22s = 188×2 sync barriers"
(TBASE_DECOMP_ANALYSIS.md Q1b) and the frag diagnosis were REFUTED by these [T1] numbers.

## 2^23/2^24 anchors + curve (2026-07-02, banked ceiling analysis)
Clean same-accounting anchors (A100 box, 12 vCPU; JSON trace_gen_s + prove_s; fold per-shard t_base/t_leaf):
| config            | trace_gen | prove | t_base (fold) | tree1 commit | note |
|-------------------|-----------|-------|---------------|--------------|------|
| 2^23 RESIDENT     | 5.76s     | 2.30s | 8.06s         | 0.82s        | old "6.6s" ≈ this (resident) |
| 2^23 STREAMING    | 18.85s    | 8.93s | 27.78s        | 13.9s        | streaming tax +19.7s (~3.4×) |
| 2^24 STREAMING    | 36.3s     | 17.0s | 51.5s (51–52) | 27.7s        | t_leaf@2^24=9.1–10.4s |
Streaming tax hits BOTH tree1 commit (rehydrate H2D) AND prove_ex (composition/OODS rehydrate). CORRECTION (2026-07-02,
TBASE_DECOMP_ANALYSIS.md): the JSON `trace_gen_s` field is MISNAMED — it's one span (main.rs:2281→3521) that INCLUDES
the tree1 commit. CPU gate sim is only ≤~4.7s (≤9% of t_base); tree1 commit (27.7s @2^24) is 76% of it. CORRECTION
(matrix, above): that 27.7s is NOT a synchronization cost — the "~22s = 188×2 cudaStreamSynchronize barriers" theory
is REFUTED (reclaim measured 0.003s). It is the PAGEABLE BLOCKING PCIe COPIES (dehydrate_d2h 19.7s + build_leaves_h2d
5.4s). ⇒ the lever is (ii) Part B (pinning + async), NOT Part A (sync-removal, a measured no-op). "trace-gen
is the CPU wall" is a RED HERRING for single-machine latency (CPU sim only binds in the 8-GPU aggregate feed ratio). Curve (model = extrapolate_8g.py conventions: P_8g=16, contended recursion
anchors t_leaf/t_node=15.8/10.7, tail=6s), ratio vs SP1 8×A100 @k=2000:
  2^23 RESIDENT (current)          → 3.3× (rec-bound, 2.52h)   [matches prior recorded baseline]
  2^24 STREAMING as-is             → 6.4× (base-bound, 4.91h)  ← ~2× WORSE than 2^23
  2^24 + Part B0 pin-ON (MEASURED) → 4.4× (base-bound, 3.36h)  ← t_base 51.5->35.2s (direct); recovers ~1/3 of PCIe tax
  2^24 + (ii) async-overlap (EST)  → 2.0× (base-bound, 1.53h)  ← best; t_base~16s est (2× the 2^23-resident, soft)
CONCLUSION: banking 2^24 as-is REGRESSES the curve (streaming PCIe tax 6.4×/shard dwarfs the 2× shard reduction) →
KEEP 2^23-resident in production for now. The 2^24 payoff (3.3×→2.0×, ~1.6×) is ENTIRELY contingent on (ii) hiding the
streaming PCIe; real (ii) t_base is between 16s and 51.5s (only a real build pins it). After (ii) the bind is still
BASE (trace-gen) → next lever is trace-gen throughput. The session's correctness work is the necessary unlock (2^24
now correct + possible; 2^25 tree1-OOM solved) but the perf win is gated behind (ii).
- Repro of the failing config: `CUDA_GPU_CONSTRAINTS=1 GATE_AIR_FUSED_INTERP=1 GATE_AIR_STREAM_COMMIT=1
  GATE_AIR_PROOF_HASH=1 <bin> --fixture <big fx> --samples 1`. Note: tree1 commit (build_leaves) is byte-clean @2^22.

## BOX BATTERY FACTS (`~/tiled2_validate.sh` on A100)
- Flags: `CUDA_GPU_CONSTRAINTS=1 GATE_AIR_FUSED_INTERP=1 GATE_AIR_STREAM_COMMIT=1`.
- Fixture: `~/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json`; samples 1/2/4/8 → 2^22/2^23/2^24/2^25.
- Battery = clean `cuda,diag` build (recompiles the `.cu`) then [A] 2^22 byte-identity vs oracle (EARLY GATE) →
  [B] 2^23 tiled==resident → [D] 2^24 → [E] 2^25.
- **WIN = [E] 2^25 exit=0 (no OOM) + fp stable + [A]/[B] byte-identical.** Battery EARLY-STOPS at [A] if tiled fp ≠
  oracle (short log ending VALIDATE_DONE right after [A] = failed-at-A).
- Ops: `box-ops/start.sh` (A100 anat-ganor-instance, ~1 attempt lately) → `box-ops/sync.sh` (git-diff-driven, catches
  untracked source) → battery. Run the poll from the MAIN loop (sonnet runners bail on the poll loop). ALWAYS
  force-stop the box after (it bills). The gate_air `.cu` is co-compiled by stwo's CMake into the gate-air-leaf target
  tree (NOT `cargo clean -p stwo`); the battery's full clean build recompiles it.

## FLAGS / FINGERPRINTS / ANCHORS
- **Flags** (all opt-in, default OFF = legacy byte-identical): `GATE_AIR_STREAM_COMMIT` (route c: fused stream commit +
  free-after-absorb + host-stage + row-tiled composition/quotient), `GATE_AIR_FUSED_COMMIT` (step-1 fused-but-resident),
  `GATE_AIR_FUSED_INTERP` (fused per-column interpolate producer), `GATE_AIR_TILE_ROWS` (tile size, default 2^20, alias
  `GATE_AIR_COMP_TILE`), `GATE_AIR_NTT_SUBBATCH` (extend batch, default 48), `CUDA_GPU_CONSTRAINTS=1` (GPU constraint
  kernel — must be ON), `GATE_AIR_PROOF_HASH=1` (print fingerprint), `PROVE_EX_TIMERS=1`.
- **Fingerprints:** base oracle @2^22 = `ab1e75b53d49645c09982ed5c890d8880366b0197b554ea5300dcd0f66838ba4`;
  @2^23 (samples=2) = `f304b5ee…`; whole-tree recursion_fingerprint = `07272a7a…`.
- **Anchors:** t_base@2^23 = 6.29–6.68s; t_leaf@2^23 = 9.16s (A100 box, 12 vCPU). (i)-stopgap clean anchors:
  2^23 t_base=6.630s, t_leaf=9.008s (uncontended A100). Recursion sweep optimum (stwo-vm 96-vCPU c4): K=16/T=6,
  t_leaf=15.8s/t_node=10.7s (contended, DDR5-optimistic). Curve model = `extrapolate_8g.py` (two-balance overlap:
  `T=max(⌈N/8⌉·t_base, N·(t_leaf+t_node)/16)+tail`).

## CORRECTED CPU-CURVE NUMBERS (don't re-derive wrong)
Per-shard base is FAST-code ~11–13s @2^24 (satsweep2 `S=4 K=1 wall=13s`) / ~23s @2^25 — NOT the stale 37.6s (that was
2^25 OLD slow-interaction-gen). The ~11× vs SP1 at k=2000 (~8.3h vs 45m53s) comes from `total_rows(k)=22.984M·k`
(k=2000 → 46G rows) ÷ SATURATED base throughput ~1.6 M rows/s (memory-bandwidth plateau, satsweep2 K=8=1.63), NOT
per-shard latency. Curve in `OPTIMIZATION_PLAN.md` is correct (uses 1.6 M/s); only the verbal per-shard walkthrough was wrong.

## NEXT-BOX MEASUREMENT CHECKLIST (once the ceiling fits)
Build `--features cuda,diag` on the A100, re-run the battery:
1. Re-confirm correctness: [A] fp==`ab1e75b5`@2^22 both paths, [B] streamed==resident@2^23, [C] tile-invariance.
2. Does 2^24/2^25 FIT? [D] samples=4, [E] samples=8: exit=0, fp printed, no OOM, warn=0 (if [E] OOMs but [D] fits,
   2^24 is the ceiling — resident witness+tree2 wall).
3. t_base@2^24 and @2^25 = wall + [prove_ex] composition_eval/TOTAL + phase breakdown (the BASE ANCHOR; un-overlapped
   composition_eval includes the un-hidden PCIe = what (ii) would recover).
4. Leaf-scaling: t_leaf@2^25 (or 2^24) vs t_leaf@2^23 ([F] fold, 1 shard each). Fewer shards helps ONLY if t_leaf grows
   slower than the shard-count drop (4× at 2^25). @2^23: t_base=6.68s, t_leaf=9.16s.
5. Compute the real curve: swap measured t_base@2^25 + leaf-scaling into `extrapolate_8g.py`; compare to the ~2.1×
   projection and to 2^23's ~3.3×. If real ≈ 2.1× and beats 2^23, (ii) is justified (→ ~1.6×). If 2^25 doesn't fit or
   doesn't beat 2^23, stop ceiling work and pivot to GPU recursion (`GPU_RECURSION_SCOPE.md`).

---

## CHECKPOINT (2026-07-01) — LDE-streaming ceiling: row-tiling correct; stopgap (i) FAILED; ceiling is STRUCTURAL

**Where we were:** route (c) — stream the tree1 commit (D2H each eval col to host stash, free device) + ROW-TILE the
composition/quotient kernels. Box-validated: [A] byte-identity @2^22 (`ab1e75b5`, both paths), [B] streamed==resident
@2^23 (`f304b5ee`), [C] tile-invariance across TILE_ROWS 2^18/2^20/2^22 — ALL PASS. So the tiling is CORRECT. BUT
[D] 2^24 and [E] 2^25 OOM during the tree1 commit (before composition).

**The blocker (re-diagnosed by full residency accounting):** NOT the tiling, NOT an allocator bug, NOT an un-freed
coeff leak (`store_polynomials_coefficients=false` → coeff polys dropped+freed at poly.rs:699). The cause is a ~2×
duplication of the whole main trace in the gate-air GPU driver:
- **#1 `d_cols`** — K1 column-major output `alloc_zeros(188*padded_rows)` (gpu_tracegen.rs:1452). 11.75 GB @2^24 /
  23.5 GB @2^25. Held via `d_main_cols` for the whole prove fn (K4 reuses it, main.rs:3192).
- **#2 188 `d2d_column` copies** — per-column copies of d_cols (gpu_tracegen.rs:1518-1520), the tree1 commit inputs.
  11.75 GB @2^24 / 23.5 GB @2^25. Freed at drain (poly.rs:691). THESE are the duplication.
- **#3 188 extended 2×-blowup evals** — `new_zeroes(2*rows)` per col (poly.rs:441). 23.5 GB @2^24 / 47 GB @2^25.
  Collapsed to ~1 col by GATE_AIR_STREAM_COMMIT (the streaming built+validated, [A]/[B]/[C] pass).
- Ruled out: NTT in-place (no scratch), no interpolate 2nd buffer, no coeff retention. → 2^25 OOMs in K1 (#1+#2 =
  47 GB, before commit); 2^24 OOMs in tree1 (#1+#2+#3). CORRECTNESS not soundness.

**(i) STOPGAP — box-validated, FAILED:** `cuda_stream_reclaim_freed` in `dehydrate_column`; [A]/[B]/[C] PASS but
[D]/[E] STILL OOM on the same 128 MiB alloc. Streaming only collapses #3; the floor is #1+#2 (driver-side
duplication), which streaming never frees. ⇒ the STRUCTURAL fix (b) — see RECURSION_PLAN.md.

---

## CHECKPOINT (2026-06-30, box session #14) — GPU FAST PATH RESTORED

- **#10's curve was an ARTIFACT — fixed + re-measured.** Root cause: the Rust decline guard `is_gate_air_main`
  (gate-air-cuda-kernel/src/lib.rs) still demanded 191 main / 1 preprocessed, so the 188-col main component failed it
  → kernel DECLINED → silent host-delegate. Fix: `GATE_AIR_TRACE_COLUMNS 191→188`, `GATE_AIR_PREPROCESSED_COLUMNS
  1→4` (N_CONSTRAINTS stays 157, interaction 24). A cross-cutting layout change must touch THREE places: K1/K4
  trace-gen, the `.cu/.cuh`, AND this guard.
- **Box result:** composition_eval 53.4s → 0.271s @2^22 (197×); prove TOTAL 1.36s@2^22 / 2.27s@2^23; host-delegate
  WARNING now SILENT under CUDA_GPU_CONSTRAINTS=1; fp==`ab1e75b5`. Single-shot GPU @2^23 phase-sum ≈ 5.0s
  (tracegen+commit ~2.7s + prove_ex 2.27s; wall 11.2s incl one-time fixture/precompute).
- **Recursion anchors (CPU, base-size-independent):** t_leaf 6.0s, t_node 6.2s, t_root 4.8s. Balance B
  `W = 8·(t_leaf+t_node)/t_base` FLIPS ≫1 (recursion-dominated) — base is single-digit seconds, lever moves to CPU
  recursion / GPU-recursion.
- **Corrected 8g curve (`extrapolate_8g.py` — TWO-BALANCE overlap; `box-ops/curve.py` SUPERSEDED).** curve.py's serial
  sum (no overlap, recursion capped at 8-way) gave a bogus 10.2×@k2000. Correct: `base_wall=⌈N/8⌉·t_base` (8 A100s) ‖
  `rec_wall=N·(t_leaf+t_node)/P_8g`; `T=max(base_wall,rec_wall)+tail`.
- **RECURSION POOL SWEEP (stwo-vm 96-vCPU c4, box-ops/stwo-vm/SWEEP_RESULTS.md):** optimal K=16, T=6 (all leaves one
  wave; peak 0.865 leaves/s); memory BW saturates at K≥8 (t_node plateaus ~10-11s). P_8g and per-fold latency are
  COUPLED by memory BW — at K=16 folds are saturated (t_leaf=15.8s, t_node=10.7s on c4, ~2.6× the K=2 low-contention
  6.0/6.2). **Corrected curve (P_8g=16 + coupled anchors): @k=2000 = 3.3× (c4/DDR5, optimistic) … 4.9× (a2 DDR4-adj,
  est ×1.5).** k=1 WINS (0.7×); recursion-bound at every k≥10. Lever: escape the CPU memory-BW wall → GPU recursion (R3).

## CHECKPOINT (superseded) — box session #10
- New oracles (188-col witness-shrunk AIR): base fp @2^22 = `ab1e75b5…`; recursion_fingerprint = `07272a7a…`. Old
  `0448237288…` retired.
- VALIDATED byte-identical: precompute A/B, K1/K4 at 188 cols, distinct-shard fold verifies + self-verifies.
- t_base@2^23 = 115.6s was the HOST-DELEGATE path (INVALID) — composition_eval 107.85s = 93-98% of t_base; kernel was
  stale for 188 cols → silent host-delegate. #14 restored it (the 35× artifact).
- Phase memory-peak: 2^24 OOMs at TREE COMMIT (main-trace LDE+Merkle, rfft.cu), flat through composition_eval — NOT
  decommit. So the ceiling work must stream/tile the main-trace LDE, not the Merkle/decommit.
