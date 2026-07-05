# 2^25 interaction-phase OOM — diagnosis + fit options (CANDIDATE ANALYSIS, pending user decision)

**Status:** SCOPE/diagnosis only. Read-only investigation, no code touched. 2^25 is currently
DEFERRED; working ceiling = 2^24. This informs whether 2^25 is worth pursuing and how.
**Lane:** interaction / K4 / tree2 + residency accounting. Does NOT touch the streaming reclaim /
tree1 / fused_commit code (Part A of async-overlap, actively edited by another agent).
**Tree analyzed:** `/home/anat/workspace/grover-tax-v02` (the active tree; `alloc inter` = the
`gpu_gen_interaction_device` path, gpu_tracegen.rs:1657). stwo CUDA backend =
`/home/anat/workspace/stwo-cuda-backend` (patched via Cargo `[patch]`, gate-air-leaf/Cargo.toml:75).

---

## 1. What `alloc inter` allocates, and how big (@2^25, padded_rows = 2^25 = 33,554,432)

The OOM string `"alloc inter: {e}"` is emitted at **`gpu_tracegen.rs:1657`**, inside
`gpu_gen_interaction_device` (the d_main-reuse K4 path, called from **main.rs:3232**, BEFORE tree2
commit — so the OOM is in **interaction WITNESS GEN**, not the tree2 Merkle commit).

The K4 interaction phase allocates (all base-domain size = padded_rows, u32=4B):

| Buffer | site | size formula | @2^25 |
|---|---|---|---|
| `d_inter` (24 M31 interaction cols = 6 QM31 × 4) | gpu_tracegen.rs:1655-1657 | `N_INTERACTION_COLS(24) * padded_rows * 4` | **3.00 GiB** |
| `d_num` (LogUp numerator scratch, 4 = one QM31) | :1658-1660 | `4 * padded_rows * 4` | 0.50 GiB |
| `d_denom` (LogUp denominator scratch, 4) | :1661-1663 | `4 * padded_rows * 4` | 0.50 GiB |
| `d_tmp` (prefix-sum scratch, 1 col) | :1723-1725 | `padded_rows * 4` | 0.12 GiB |
| `d_ap` / `d_dims` / `d_sums` | :1649/1651/1696 | tiny (≤ 140 words) | ~0 |
| **Interaction new-alloc total** | | `(24+4+4+1) * padded_rows * 4` | **≈ 4.12 GiB** |

`N_INTERACTION_COLS = 24`, `N_LOGUP_COLS = 6` (gpu_tracegen.rs:595-597). Note these are
**base-domain** (2^25), NOT eval-domain — the LDE blowup happens later in the tree2 commit, not here.
The `d_inter` alloc (**one contiguous 3 GiB block**) is the failing `alloc inter`.

There is **no sumcheck buffer** on this path (LogUp is done by cumsum kernels: `logup_col_gen` →
`logup_finalize_col` → `logup_cumsum_reduce`/`shift` → `prefix_sum_column`, gpu_tracegen.rs:1671-1729;
K4 reads `d_cols` and writes its own num/denom/inter scratch). The 4 small table interactions
(qdecode / rc_lo / rc_hi / program, main.rs:3257-3302) are built on the **CPU (host)** and only
uploaded at tree2 commit — they are NOT part of this device peak.

---

## 2. Full residency accounting at the interaction phase (@2^25, 40 GB budget)

What is LIVE on-device the moment `alloc inter` fires:

| Live buffer | why resident here | size @2^25 |
|---|---|---|
| **#1 `d_cols`** (188 base-domain main cols, K1 column-major) | held via `d_main_cols` for the whole prove fn; K4 reads it directly (`d_cols = d_main`, gpu_tracegen.rs:1634). ONE contiguous `alloc_zeros(188*padded_rows)` (gpu_tracegen.rs:1481-1482) | **23.5 GiB** |
| tree0 (4 preprocessed) — resident committed polys | committed in cached precompute; small | ~0.5–1.0 GiB |
| small tree1 (mult 2^9/2^16 + program witness) — resident | FULL(A): small tree1 stays resident (poly.rs:779-788) | < 0.1 GiB |
| twiddles (base + eval domain) | precompute_twiddles, main.rs:2093 | ~0.25–0.5 GiB |
| **interaction new allocs** (`d_inter`+num+denom+tmp) | this phase (§1) | **≈ 4.12 GiB** |
| **188 freed tree1 eval segments cached in the CUDA pool** | see §2a — the smoking gun | **up to ~47 GiB *reserved*** |

**Explicit-buffer sum WITHOUT the pool reserve:** `d_cols (23.5) + interaction (4.12) + tree0 (~1) +
twiddles (~0.5) ≈ 29.2 GiB` — **comfortably under 40 GB.** So the *logical* working set fits. The OOM
is therefore **not** a raw live-buffer overflow; it is driven by pool reservation / fragmentation.

### 2a. The smoking gun — the CUDA mem-pool holds the 188 freed tree1 eval segments (no-trim)

- The 188 large tree1 columns are LDE'd to eval domain (base+`BASE_LOG_BLOWUP_FACTOR`=1 → **2^26**,
  main.rs:91), i.e. **256 MiB each, ~47 GiB total**, produced and dehydrated **one at a time**
  (poly.rs:668-750; `dehydrate_column` D2H-copies to `HOST_STASH`, then frees, fused_commit.rs:135-159).
- The allocator is CUDA's built-in **memory pool** (`cudaMallocFromPoolAsync`/`cudaFreeAsync`,
  cuda_mem_pool.cuh:41/80) with **`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`**
  (cuda_mem_pool.cu:31-32) — i.e. **freed segments are CACHED in the pool, never released to the OS.**
- The streaming path's default reclaim is **`cuda_stream_reclaim_freed_notrim()`** (Part A;
  fused_commit.rs:361-370 → utils.cu:443-445): it only `cudaStreamSynchronize(0)` and **deliberately
  does NOT `cudaMemPoolTrimTo`** (the trim is the ~22 s per-column churn Part A removes). Legacy
  sync+trim is only under `GATE_AIR_RECLAIM_TRIM` (utils.cu:422-427).

**Consequence:** after tree1 commit, the pool is holding **~23.5 GiB+ of cached 256-MiB freed
segments** (bounded by peak live during the loop, but with no-trim it is NOT handed back to the OS)
**on top of** the 23.5 GiB still-live contiguous `d_cols`. When K4 requests a **fresh 3 GiB contiguous
`d_inter`**, the pool cannot carve one contiguous 3 GiB span out of its fragmented free-list of
256-MiB blocks that are interleaved in the arena with the live 23.5 GiB `d_cols`, so it grows the
pool — and total reservation exceeds the 40 GB device. → `CUDA_ERROR_OUT_OF_MEMORY`.

**This is a POOL-RESERVATION / FRAGMENTATION OOM, not a working-set OOM.** The logical working set is
~29 GiB. That distinction drives the options in §4 (the cheapest fixes are pool-hygiene, not
algorithmic).

> CONFIDENCE / what needs a box measurement: the "pool holds ~23.5+ GiB of cached freed segments and
> can't serve a contiguous 3 GiB" mechanism is inferred from the code (UINT64_MAX threshold + no-trim
> default + one contiguous d_inter). It is **not yet directly measured.** See §6 for the exact
> instrumentation to confirm before committing to a fix.

---

## 3. Is `d_cols` (23.5 GiB) actually needed resident DURING interaction?

**Yes, as currently written** — K4 reads `d_cols` directly (`d_cols = d_main`, gpu_tracegen.rs:1634;
`logup_col_gen` reads `d_cols`, :1677). It is held precisely so K4 does not re-run K0/K1
(gpu_tracegen.rs:1585-1591, main.rs:3146-3149).

**Re-examining the earlier ruling.** The plan RULED OUT option (a) "regenerate d_cols for K4" — but
that ruling was about the **tree1/K1 peak** where #1 `d_cols` + #2 the 188 `d2d_column` copies
coexisted (47 GB), RECURSION_PLAN.md:270-280. That peak is now solved by streaming/fix(b). **For THIS
interaction peak the calculus is different:**

- The interaction allocations ALONE (without d_cols) are only ~4.12 GiB — they trivially fit.
- The problem is that d_cols (23.5 GiB, contiguous) coexists with a fragmented ~23.5 GiB pool reserve.
- So there are two independent levers: (i) **stop the pool from hoarding the freed tree1 segments**
  (don't touch d_cols at all), or (ii) **free/stream d_cols and regenerate it for K4** (removes the
  23.5 GiB live floor, leaving the pool reserve + interaction alloc, ~ fits).

Both are viable *for the interaction peak specifically*. (i) is cheaper and lower-risk (see §4).

---

## 4. Options to fit 2^25 (each: memory effect / effort / risk)

Ordered cheapest→most-invasive. **The first two directly attack the §2a fragmentation root cause and
are by far the most promising.**

### (0) Trim the pool once, after tree1 commit, before K4 — CHEAPEST, HIGHEST-CONFIDENCE
Call `cudaMemPoolTrimTo(g_mem_pool, keep_bytes)` **once** at the tree1→interaction boundary (or set
`GATE_AIR_RECLAIM_TRIM` for the run), releasing the ~23.5 GiB of cached freed tree1 segments back to
the OS so the fresh contiguous 3 GiB `d_inter` has room.
- **Memory:** removes ~23.5 GiB of pool reserve → peak ≈ d_cols(23.5) + interaction(4.12) + tree0 +
  twiddles ≈ **29–30 GiB. Fits 40 GB with margin.**
- **Effort:** tiny (one FFI call at a phase boundary, or just run with `GATE_AIR_RECLAIM_TRIM=1`).
- **Risk:** LOW. Pure allocator hygiene; byte-identical proof. **Caveat: the per-column TrimTo in the
  loop is exactly the ~22 s churn Part A removed — so do a ONE-SHOT trim at the boundary, NOT
  per-column.** This is a ONE-LINE probe: the `GATE_AIR_RECLAIM_TRIM` path ALREADY exists
  (fused_commit.rs:361-367) — running [E] with that env var set is a **zero-code experiment** to
  confirm the diagnosis and possibly close 2^25 outright (at the ~22 s streaming-tax cost, acceptable
  for a first "does it fit" datapoint).
- **File touchpoint if made permanent:** a boundary trim would live in the streaming/reclaim lane
  (fused_commit.rs / utils.cu) — coordinate with the Part A agent; do NOT edit concurrently.

### (i) Stream the interaction gen (dehydrate the 24 cols like tree1) — LOW value here
Host-stage `d_inter` columns as produced.
- **Memory:** saves only ~3 GiB (interaction cols are base-domain 2^25, not eval 2^26). Does NOT
  touch the 23.5 GiB fragmentation reserve → **would NOT fix the OOM by itself.**
- **Effort:** medium. **Risk:** medium (K4 num/denom/prefix-sum are cross-row; streaming complicates
  the cumsum). **Verdict: not worth it** — attacks the small term, misses the root cause.

### (ii) Free/stream `d_cols` before K4, regenerate for interaction — removes the 23.5 GiB floor
Dehydrate d_cols (or drop it and re-run K0/K1) so it is not live during K4; K4 rehydrates/regenerates.
- **Memory:** removes the 23.5 GiB live floor. Peak during K4 ≈ interaction(4.12) + regenerated main
  input + pool reserve. **Fits.**
- **Effort:** HIGH (re-run K0/K1 sim + re-upload, or a 23.5 GiB D2H/H2D round trip per shard).
- **Risk:** medium (correctness: regenerated d_cols must be byte-identical to K1's; perf: re-sim or
  47 GB×2 PCIe is a large per-shard tax at 9024 shards). **Verdict: heavier than (0); only needed if
  the pool-hygiene fix proves insufficient (e.g. if the arena still can't coalesce).**

### (iii) Reduce the interaction working set (fuse LogUp num/denom, tile K4)
Fuse `d_num`+`d_denom` (share a buffer) and/or tile K4 over row-blocks.
- **Memory:** saves ≤ ~1 GiB (num+denom = 1 GiB combined). Marginal; does NOT fix the OOM.
- **Effort:** medium-high; **Risk:** HIGH — **touches LogUp numerator/denominator accumulation, which
  is [SOUNDNESS-CRITICAL]** (see §5). **Verdict: avoid** unless forced; wrong risk/reward.

### (iv) Use a per-column d_inter (24 separate 128 MiB allocs instead of one 3 GiB block)
Make `d_inter` 24 separate `padded_rows` allocations so each can be served from a cached 256-MiB
freed segment (no contiguous 3 GiB request).
- **Memory:** no change to totals, but **sidesteps the contiguity requirement** — 24 × 128 MiB
  requests each fit a cached freed segment.
- **Effort:** medium (touches `d2d_column` handoff + kernel indexing `kk*padded_rows`, gpu_tracegen.rs:
  1671-1737). **Risk:** medium (kernel currently assumes one contiguous `d_inter`; per-col changes the
  `offset` math and the d2d handoff). **Verdict: a fallback if (0) can't fully coalesce**, but (0) is
  simpler.

**Recommendation:** try **(0) first** — it is a zero-code experiment (`GATE_AIR_RECLAIM_TRIM=1`) that
both CONFIRMS the diagnosis and likely FITS 2^25. If it fits but the ~22 s trim tax is unacceptable,
promote to a **one-shot boundary trim** (still option 0, but trim once at tree1→K4 rather than
per-column). Only escalate to (ii)/(iv) if (0) does not fit (i.e. fragmentation persists even after a
full trim, which would point at the live d_cols itself blocking coalescing).

---

## 5. Soundness-critical vs private-backend surface

- **LogUp = [SOUNDNESS-CRITICAL]** per CLAUDE.md (`constraint-framework/src/logup.rs`). Option (iii)
  (fuse num/denom, tile K4) touches LogUp numerator/denominator accumulation semantics → **would need
  the supervised-change protocol** (load skill, cite paper section, state invariant, human approval).
  **Avoid.**
- The gate_air GPU K4 driver (`gpu_gen_interaction_device`, gpu_tracegen.rs) and the
  `logup_col_gen`/`finalize`/`cumsum` **CUDA kernels** are **private gate_air backend** — changing
  d_inter's allocation shape (option iv) or freeing d_cols (option ii) stays in the private driver and
  does NOT touch the shared soundness-critical LogUp constraint code. Still guarded by byte-identity +
  self-verify (correctness, not soundness — verifier untouched).
- Option (0) is pure **allocator hygiene** (pool trim) — no soundness surface at all; the
  `GATE_AIR_RECLAIM_TRIM` path already exists and is a legacy-equivalent code path.
- **Lane note:** a *permanent* option-(0) trim (or option i) would live in the streaming/reclaim files
  (fused_commit.rs / utils.cu) that the Part A agent is editing — coordinate; the *experiment* (env
  var) needs no code change and no coordination.

---

## 6. Exact instrumentation to confirm before implementing (needs the box; box currently STOPPED)

Numbers that are code-derived (sizes in §1/§2) are solid. The **mechanism** in §2a should be confirmed
by ONE box run before any permanent code change:

1. **Zero-code confirm:** re-run the `[E]` 2^25 battery with **`GATE_AIR_RECLAIM_TRIM=1`** (all other
   streaming flags as-is). If it clears `alloc inter` and reaches tree2/prove, the pool-reserve
   diagnosis is confirmed AND option (0) is validated. (Expect ~+22 s streaming tax.)
2. **Measure the pool reservation** at the tree1→K4 boundary: add a temporary
   `cudaMemPoolGetAttribute(g_mem_pool, cudaMemPoolAttrReservedMemCurrent, …)` +
   `cudaMemGetInfo(&free,&total)` print right before `alloc inter` (gpu_tracegen.rs:1655). This gives
   the exact reserved-vs-live split and confirms whether the ~23.5 GiB reserve + 3 GiB request is what
   tips it over 40 GB. (This print is a temporary diagnostic, private-driver only.)
3. If (1) fits but the tax is unwanted: prototype a **one-shot** `cudaMemPoolTrimTo` at the boundary
   (not per-column) and re-measure both fit and t_base.

Do NOT fabricate the exact reserved-bytes figure — it needs run (2). The 4.12 GiB interaction total,
23.5 GiB d_cols, and 47 GiB total-streamed figures ARE code-exact (§1, blowup=1 → eval 2^26).

---

## Bottom line

- The 2^25 interaction OOM is a **CUDA mem-pool fragmentation/reservation** problem, not an
  algorithmic working-set overflow: the logical working set is ~29 GiB (fits 40 GB), but the pool
  hoards ~23.5 GiB of cached freed 256-MiB tree1 eval segments (no-trim, ReleaseThreshold=UINT64_MAX)
  which, alongside the live contiguous 23.5 GiB `d_cols`, blocks a fresh **contiguous 3 GiB** `d_inter`.
- **Cheapest fix = a one-shot pool trim at the tree1→interaction boundary** (option 0), testable
  TODAY with **zero code** via `GATE_AIR_RECLAIM_TRIM=1`. This likely closes 2^25.
- Streaming the interaction (i) or fusing/tiling LogUp (iii) attack the wrong (small) term and (iii)
  touches soundness-critical LogUp — **not recommended.**
- Whether 2^25 is worth pursuing at all is still gated by the CURVE DECISION (RECURSION_PLAN.md:
  361-374): 2^24 already regresses the curve without async-overlap; 2^25's payoff is likewise
  contingent on (ii) async-overlap hiding the streaming PCIe tax. **This memory fix is a necessary
  unlock, not sufficient for a curve win** — recommend confirming the cheap fit (option 0) as a
  datapoint, but sequencing the actual 2^25 perf push behind async-overlap.
