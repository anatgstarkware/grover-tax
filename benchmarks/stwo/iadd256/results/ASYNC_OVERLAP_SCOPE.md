# ASYNC-OVERLAP SCOPE — task #20, gate_air CUDA base prover

> **SCOPE — pending user greenlight, NOT implemented.** This is a candidate design for the
> "(ii) async-overlapped streaming" optimization. No code, no `RECURSION_PLAN.md` edit, no box spend
> was done producing it. Line numbers and file paths are the working streaming tree
> `~/workspace/stwo-cuda-backend/crates/stwo/src/...` (this is where the streaming stash + fused
> commit actually live; the `~/workspace/stwo-gpu-port` tree is the base/upstream port and does NOT
> carry the streaming stash).

## 0. TL;DR / feasibility read

**Feasible, moderate risk, and the substrate already exists.** The streaming path currently pays a
fully-serialized PCIe tax because (a) every dehydrate does a per-column `cudaStreamSynchronize(0)` +
`cudaMemPoolTrimTo` (anti-overlap by construction), (b) the stash is a pageable `Vec<u32>` (pageable
D2H/H2D cannot truly overlap compute), and (c) all copies + kernels are on the default stream 0
(nothing to overlap against). The async-copy + pinned-host bindings are **already landed but
unwired** (`cuda_alloc_pinned_host_uint32_t`, `cuda_free_pinned_host`,
`copy_uint32_t_vec_from_{device_to_host,host_to_device}_async` — all in `stwo_cuda/cuda/utils.cu` +
`bindings.rs`, each taking an explicit `cudaStream_t`). **What is NOT landed** is the piece that
makes overlap *correct*: **stream creation and CUDA events** — there is zero `cudaStreamCreate` /
`cudaEventCreate` / `cudaEventRecord` / `cudaStreamWaitEvent` anywhere in the backend. Those are the
new FFI surface, and they are the correctness core (an under-synchronized cross-stream read = stale
bytes = a WRONG-but-REJECTED proof — a **correctness** bug caught by byte-identity, not a soundness
bug; the verifier is untouched).

Phased so each part lands + validates independently:
- **Part A** (allocator reclaim without per-column host sync) — unblocks overlap, is the highest
  ratio of "removes a hard serialization point" to LOC.
- **Part B1** (commit-side async D2H, pinned + copy stream) — hides the tree1-commit D2H tax.
- **Part B2** (consumer-side async H2D, pinned + copy stream) — hides the build_leaves /
  composition / OODS / quotient rehydrate tax.

Realistic post-(ii) `t_base@2^24` estimate: **~28–36 s** (see §5), NOT the optimistic 16 s. Overlap
hides the *copy behind compute* per phase, bounded by `max(compute, PCIe)` not the sum; but a large
fraction of the streaming tax is compute that *waits on the reclaim sync* and on serialized
per-column NTT/absorb, plus trace-gen (~72% of `t_base`, CPU-bound) is untouched by (ii).

---

## 1. Current data-flow map (every dehydrate D2H / rehydrate H2D, and which are synchronous)

All copies today go through the pageable path on **default stream 0**. Pageable host memory forces
D2H to effectively block the host until the DMA completes, and H2D to stage synchronously — so every
one of these is serialized against compute.

### Producer (dehydrate, D2H) — `poly.rs`
- `evaluate_polynomials` (poly.rs:489) fused-interpolate group: per eval column, in-place
  interpolate → extend → `ntt_n2b_columns` (poly.rs:711) → **`fused_commit::dehydrate_column(&mut
  col)`** at **poly.rs:727**, guarded by `if stream_tree1`. `clear_stash()` at poly.rs:532 at the
  start of the streamed commit.
- `dehydrate_column` (fused_commit.rs:~120): `col.to_vec()` (**synchronous pageable D2H** of the
  whole eval column) → `cuda_free_memory` → **`cuda_stream_reclaim_freed(0)`** →
  re-key by monotonic sentinel → insert into `HOST_STASH`. **This is the un-overlapped D2H #1.**
- `cuda_stream_reclaim_freed(0)` (utils.cu:422) = **`cudaStreamSynchronize(0)` +
  `cudaMemPoolTrimTo(g_mem_pool, keep_bytes)`** — a full host/device barrier **per column**. This is
  the fundamental anti-overlap point (Part A target).

### Consumers (rehydrate, H2D) — all currently synchronous whole-column or block H2D
- **build_leaves rehydrate** — `blake2s.rs:104` `build_leaves`, heterogeneous path (blake2s.rs:~125,
  270): staged large cols are rehydrated (`rehydrate_owned`, whole column) into a fresh device buffer,
  absorbed, freed — **per-column, serialized**. This is the biggest single tax at commit (2^23:
  tree1 commit 0.82 s resident → 13.9 s streaming).
- **OODS barycentric** — `poly.rs:410` `is_staged` → `rehydrate_owned` (whole column) before
  `barycentric_eval_at_point`. Runs on **rayon worker threads** under `par_map_cols` (why the stash is
  process-global) — H2D is synchronous per column.
- **quotient** — `quotient.rs:180` (`any_staged`), `quotient.rs:309` `rehydrate_block(col, off,
  this_block)` per row-block per column, run the SAME kernel over `[0, block)`, free. **Per
  block-per-column synchronous H2D.**
- **composition** — `evaluate_gate_air.cu` staged path: staged cols supplied via the per-column host
  table with `tiled_input=true`; the tile is H2D'd, resident cols read live UNBIASED. Staged block
  H2D is the composition rehydrate. (`gate_air_entry.cu` + `gate-air-cuda-kernel/lib.rs` orchestrate.)
- **decommit** — `column.rs:102` `is_staged` → `host_batch_get(self, &[index])` (host-side read of
  the stash, not a device copy) — negligible, not on the overlap critical path.

**Summary of synchronous copies to hide:** commit-side = 188 whole-column D2H (dehydrate) + 188
whole-column H2D (build_leaves rehydrate); consumer-side = whole-column H2D ×188 (OODS) + per-block
H2D (quotient) + per-tile H2D (composition). Plus 188× the reclaim barrier.

---

## 2. The stash today, and the pinned + event-gated design it needs

`HOST_STASH` (fused_commit.rs:~70): `static LazyLock<Mutex<HashMap<usize, Vec<u32>>>>`, keyed by a
monotonic sentinel `SENTINEL_TAG(1<<63) | id` written into the staged column's `device_ptr`
(disjoint from any real 48-bit GPU VA). `owns_memory=false` gates `is_staged`. Process-global (not
thread-local) because OODS/quotient read it from rayon workers.

**Why it can't overlap as-is:** the value is a pageable `Vec<u32>`. For a copy to overlap compute on
another stream, the host side must be **page-locked (pinned)**; pageable memory forces the driver to
stage through an internal pinned bounce buffer and, in practice, serializes.

**Pinned + event-gated redesign (the correctness core for async D2H INTO / H2D OUT OF the stash):**
- Store the stash value as a **pinned buffer** (via `cuda_alloc_pinned_host_uint32_t` /
  `cuda_free_pinned_host`), not `Vec<u32>`. Wrap it in an RAII type so `Drop` calls
  `cuda_free_pinned_host` (mirrors the base backend's `PinnedBuffer`/`Drop` in
  `stwo-gpu-port .../gpu/optimizations.rs` — but that lives in the *other* (cudarc) backend and is
  not reachable from the C-FFI `stwo_cuda` backend, so a small analogous wrapper is added here).
- **Each stash entry carries a completion `cudaEvent_t`.** For a D2H into the stash: the copy is
  issued on a copy stream, an event is recorded on that stream right after the copy, and **no reader
  may read the pinned bytes until that event has completed**. For an H2D out of the stash: the H2D is
  issued on a copy stream, an event is recorded, and **the consuming kernel must
  `cudaStreamWaitEvent` on that event before it runs** (see §3).
- **The Mutex is fine and does NOT block overlap.** It serializes only the CPU-side HashMap lookup +
  the pinned-ptr/event handles (microseconds); it does not hold across the DMA. The DMA proceeds on
  the copy stream after the lock is released. Keep the lock scope tight: look up ptr+event, drop the
  guard, then issue/wait on the copy. (Do NOT hold the guard across a synchronize.)
- **Lifetime rule:** a pinned buffer must not be freed until its last D2H (fill) AND every
  outstanding H2D (drain) reading it have completed. `clear_stash()` (called only at the next
  streamed-commit boundary) must first ensure all outstanding events on entries are complete before
  freeing pinned buffers — else a free races an in-flight copy.

---

## 3. CUDA events / cross-stream ordering plan (the correctness core)

New FFI (must be ADDED — none exists today): `cuda_create_stream() -> cudaStream_t`,
`cuda_destroy_stream`, `cuda_create_event() / cuda_destroy_event`, `cuda_record_event(event,
stream)`, `cuda_stream_wait_event(stream, event)`, `cuda_event_synchronize(event)`. All thin
`extern "C"` wrappers over the runtime API, alongside the existing async-copy wrappers in utils.cu.

Ordering contract (this is what byte-identity protects):

1. **Commit-side D2H (fill):** on copy stream `S_copy`, issue D2H of eval col `c` into pinned
   `stash[c]`; `cuda_record_event(E_fill[c], S_copy)`. The producer's NEXT column's NTT/absorb runs
   on the compute stream `S_comp` (or stream 0) and does NOT depend on `E_fill[c]`, so col `c+1`'s
   compute overlaps col `c`'s D2H. **`E_fill[c]` gates any reader** (build_leaves absorb, OODS,
   quotient, composition) that touches `stash[c]`: that reader must `cuda_event_synchronize(E_fill[c])`
   (host-side, if the reader reads pinned bytes on CPU) or, if it H2D's back, chain the H2D behind it.
2. **Consumer-side H2D (drain):** on `S_copy`, issue H2D of `stash[c]` (or block) → fresh device
   buffer `d`; `cuda_record_event(E_h2d, S_copy)`. Before the consuming kernel launches on `S_comp`,
   **`cuda_stream_wait_event(S_comp, E_h2d)`** so the kernel cannot read `d` before the copy lands.
   The PRIOR block's kernel on `S_comp` overlaps this block's H2D on `S_copy`.
3. **Buffer free waits on BOTH:** a device buffer `d`'s free must wait on (i) its last-consumer
   kernel on `S_comp` AND (ii) any copy that wrote it. With stream-ordered `cudaFreeAsync` on
   `S_comp`, (i) is automatic; (ii) needs `cuda_stream_wait_event(S_comp, E_h2d)` before the free is
   enqueued (already implied by the kernel dependency in step 2). A pinned host buffer's free waits
   on its last D2H fill AND every H2D drain (`E_fill` + all outstanding drain events).

**Double-buffering:** use 2 (small N) rotating device tiles + 2 pinned staging slots per pipeline so
block `k`'s copy fills slot `k%2` while block `k-1`'s kernel consumes slot `(k-1)%2`. Events gate the
handoff. Ring-buffer index arithmetic is the only new state.

**The async-race risk is a correctness bug, not soundness.** An under-synchronized read produces
stale/garbage committed bytes → the proof is REJECTED (or fingerprint differs), caught by:
- **[A]** 2^22 tiled fp == oracle `ab1e75b5…`
- **[B]** 2^23 tiled == resident `f304b5ee…`
- **[C]** tile-invariance across TILE_ROWS 2^18/2^20/2^22
- proof self-verify (`GATE_AIR_PROOF_HASH=1`, `--samples 1`).

The verifier is never touched, so this cannot be a soundness (accept-invalid) defect — only a
correctness (reject-valid / wrong-fp) one. (Terminology per project memory: this is correctness,
caught by byte-identity.)

---

## 4. Primitives that exist vs. must be added

**Exist (unwired) — `stwo-cuda-backend` C-FFI backend (the gate_air path):**
- `cuda_alloc_pinned_host_uint32_t` / `cuda_free_pinned_host` (utils.cu:446/457; `cudaHostAlloc` /
  `cudaFreeHost`) — real pinned host alloc.
- `copy_uint32_t_vec_from_device_to_host_async(dev, host, size, stream)` (utils.cu:467) and
  `copy_uint32_t_vec_from_host_to_device_async(host, size, stream) -> dev` (utils.cu:476) — real
  `cudaMemcpyAsync`, take an explicit `cudaStream_t`, do NOT synchronize. Bound in bindings.rs:141-158.
- `cuda_stream_reclaim_freed(keep_bytes)` (utils.cu:422) — the sync+trim to be REPLACED in Part A.
- Mem pool: `cudaMallocFromPoolAsync` / `cudaFreeAsync` on stream 0, `cudaMemPoolAttrReleaseThreshold
  = UINT64_MAX` (never auto-trims — this is precisely why the stopgap must manually `TrimTo`).
- fused_commit.rs comment (lines ~44-54) already spells out this exact deferred wiring: "the
  pinned-host + async-copy primitives are landed … but the stash is NOT yet wired onto them."

**Must be ADDED:**
- **Stream creation FFI** — a dedicated copy stream `S_copy` (and optionally a compute stream, or
  reuse stream 0 as compute). None exists. `cuda_create_stream/destroy_stream`.
- **Event FFI** — `cuda_create_event/record_event/stream_wait_event/event_synchronize/destroy_event`.
  None exists (`grep cudaEventCreate` → 0 hits). This is the correctness core.
- **A pinned RAII wrapper** in the Rust stash (analogous to `PinnedBuffer` in the *other* backend's
  `optimizations.rs`, which is NOT reachable here — that `AsyncTransfer` is also a stub: its
  `start_h2d` comment says "cudarc's htod_sync_copy is synchronous; for true async we'd need raw CUDA
  calls" — so do NOT try to reuse it; wire the C-FFI async copies directly).
- **Part A reclaim replacement** (see §5).
- Wiring: `dehydrate_column` → pinned + async D2H on `S_copy`; `rehydrate_owned` / `rehydrate_block` →
  async H2D on `S_copy` + return the gating event; consumers → `stream_wait_event` before kernel.

---

## 5. Expected payoff — what's hideable and a realistic `t_base@2^24`

Overlap hides copy behind compute per phase: the phase cost becomes `max(compute, PCIe)`, not
`compute + PCIe`. Measured tax (A100, 12 vCPU):

| config | trace_gen | prove | t_base | tree1 commit |
|---|---|---|---|---|
| 2^23 RESIDENT | 5.76 | 2.30 | 8.06 | 0.82 |
| 2^23 STREAMING | 18.85 | 8.93 | 27.78 | 13.9 |
| 2^24 STREAMING | 36.3 | 17.0 | 51.5 | 27.7 |

Streaming tax @2^23 = +19.7 s, split across (a) the per-column reclaim barrier (188×
`cudaStreamSynchronize(0)` + `TrimTo`), (b) tree1 commit rehydrate (0.82→13.9 s, +13.1 s), (c)
prove_ex rehydrate (composition/OODS/quotient, 2.30→8.93 s, +6.6 s).

**Hideable:**
- **Part A** removes the 188× reclaim barrier — this is pure serialization (host waits on device with
  no useful overlap), so its removal is a near-direct win AND is the precondition for B to overlap at
  all. Hard to attribute in isolation but likely several seconds @2^24.
- **Part B1** (commit D2H): the D2H of col `c` overlaps col `c+1`'s NTT/absorb. Bounded by
  `max(Σcompute, Σ D2H)`. At 2^24, 188 × 256 MiB ≈ 47 GB D2H; PCIe gen4 ×16 ≈ 25 GB/s ⇒ ~1.9 s of
  raw D2H — most of it hideable behind NTT/absorb compute if compute ≥ that. The 13.1 s tree1-commit
  swing is dominated by the serialized rehydrate + reclaim, NOT raw bandwidth, so B1+A should recover
  most of it (commit → a few seconds over resident, not +13).
- **Part B2** (consumer H2D): the ~47 GB H2D read-set for composition/quotient overlaps the prior
  block's kernel. Recovers most of the +6.6 s prove_ex swing IF composition/quotient compute per
  block ≥ the block H2D time (it is, at TILE_ROWS ≥ 2^20).

**NOT hideable:**
- **trace_gen (~72% of t_base, CPU memory-bound)** — untouched by (ii). @2^24 that's ~36 s in the
  streaming number, but the *resident* trace_gen was 5.76 s @2^23 (streaming inflates trace_gen too,
  because the fused-interpolate-in-commit restructure moved NTT into the streamed loop). The real
  question is how much of streaming's trace_gen inflation (5.76→18.85 @2^23) is the reclaim barrier
  (recoverable by A) vs. genuine per-column serialization of NTT (partially recoverable by B1
  overlap) vs. irreducible CPU work.
- One-time setup, twiddle gen, FRI (reads resident quotient), tree2 (resident, read whole).

**Realistic `t_base@2^24` post-(ii): ~28–36 s** (band, not a point):
- Optimistic edge (~28 s / ~2.5× SP1 @k=2000): A recovers the reclaim barrier, B1 collapses commit to
  ~resident+PCIe, B2 collapses prove_ex to ~resident+PCIe, trace_gen falls back toward ~2× the
  resident trace_gen once the reclaim serialization is gone.
- Conservative edge (~36 s / ~3.3× SP1): trace_gen inflation is mostly irreducible per-column NTT
  serialization that overlap only partially hides, and PCIe (~2 s/phase) is exposed where compute per
  phase is thin.
- The plan's "~16 s" is the *floor if trace_gen were the resident 5.76 s and all copy fully hidden* —
  treat it as a lower bound, not a target. **Only a real build + re-measure pins it** (§7).

**Curve consequence:** even the conservative ~36 s keeps 2^24 roughly at parity with 2^23-resident
(3.3× SP1); the optimistic ~28 s is where 2^24 starts to *beat* 2^23 and the ceiling lift pays off.
So (ii) is necessary-but-may-not-be-sufficient on its own; trace-gen throughput is the next lever
regardless (already flagged in the plan).

---

## 6. Scope boundaries

- **In scope:** private CUDA backend only — `stwo_cuda/cuda/utils.cu` (+ new stream/event FFI),
  `stwo_cuda/bindings.rs`, `prover/backend/cuda/{fused_commit,poly,blake2s,quotient,column}.rs`,
  `evaluate_gate_air.cu` / `gate_air_entry.cu` / `gate-air-cuda-kernel/lib.rs` (composition
  stream-wait wiring). gate_air path only; all behind the existing `GATE_AIR_STREAM_COMMIT` /
  `GATE_AIR_FUSED_INTERP` flags (default OFF = legacy byte-identical).
- **Out of scope / MUST NOT touch:** the shared verifier, `core/` (SOUNDNESS-CRITICAL), FRI/PCS
  params, Fiat-Shamir, `vcs`/`vcs_lifted` commit logic, constraint definitions. Committed bytes must
  be **byte-identical** to the resident/oracle path. Fail-loud preserved (`is_staged`/`rehydrate_*`
  panic on a missing key — must remain; add event-not-ready fail-loud too). No silent CPU/pageable
  fallback when the async path can't run — panic loudly (per project memory).
- **2^25 interaction/tree2-alloc OOM is explicitly out of scope** (a separate, later blocker).
- **Flag anything needing a design fork:** (1) whether the compute path stays on stream 0 or gets a
  dedicated `S_comp` (stream-0 legacy interactions with other kernels — safer to keep compute on 0
  and add only `S_copy` + events; present both). (2) The **stash-concurrency caveat**: the global
  `clear_stash()` wipes the whole map; async adds *outstanding events* to that hazard — `clear_stash`
  must now also drain events before freeing pinned buffers. Single-proof-per-process holds today so
  this is not blocking, but note it. (3) Whether double-buffering depth N=2 suffices or the reclaim
  needs a bounded release threshold instead of manual trim (Part A alternatives below) — measure.

---

## 7. Validation plan

Reuse the existing battery verbatim (it is what makes the async-race safe):
1. **[A]** 2^22 tiled fp == oracle `ab1e75b53d49645c…` (BOTH paths; battery early-stops here if ≠).
2. **[B]** 2^23 streamed == resident `f304b5ee…`.
3. **[C]** tile-invariance across TILE_ROWS 2^18 / 2^20 / 2^22.
4. Proof self-verify: `GATE_AIR_PROOF_HASH=1 <bin> --fixture <fx> --samples 1`.
5. **[D]** 2^24 exit=0, no OOM, fp stable (`44fec71a…` baseline for the current streaming path — must
   match, since committed bytes are unchanged).
6. **New async-specific checks:**
   - Run [A]/[B] under **`compute-sanitizer --tool racecheck`** (and `--tool memcheck` /
     `synccheck`) if available on the box — catches missing `stream_wait_event` / use-before-copy /
     use-after-free that a single lucky run would miss.
   - Run [A]/[B] **repeated N× in a loop in one process** to shake out event/pinned-lifetime races
     that a single run hides (timing-dependent).
   - Assert the pipeline actually overlapped: re-measure and confirm `tree1 commit` and `prove_ex`
     shrank vs. the streaming anchors (13.9→? @2^23 commit; 27.7→? @2^24 commit). If they don't
     shrink, overlap isn't happening (events over-serializing or copies still pageable) — fail the
     experiment, don't bank it.
7. **Re-measure `t_base@2^24`** (JSON `trace_gen_s` + `prove_s`, same accounting as the anchor table)
   and overlay on the SP1 curve to confirm the tax shrank into the §5 band. This is the WIN
   criterion for (ii).

---

## 8. Risk register, change surface, phased plan

### Risk register
| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Missing/incorrect `stream_wait_event` → consumer reads pre-copy bytes | High (correctness) | [A]/[B]/[C] byte-identity + racecheck + loop-repeat; event-not-ready fail-loud |
| R2 | Pinned buffer freed while a copy is in-flight (esp. `clear_stash`) | High (correctness/UB) | Drain `E_fill`+drain events before free; RAII Drop only after event-sync |
| R3 | Part A reclaim change → OOM returns (blocks freed too late) | High (regression) | Keep a bounded release threshold + stream-ordered free; re-run [D] 2^24 no-OOM |
| R4 | Overlap doesn't materialize (still serialized) → no payoff, wasted box | Med | §7.6 assert commit/prove shrank; racecheck confirms multi-stream |
| R5 | Dedicated compute stream interacts badly with legacy stream-0 kernels | Med | Keep compute on stream 0; add only `S_copy`; present the fork |
| R6 | Concurrent-proof stash hazard worsened by events | Low (not hit today) | Single-proof-per-process holds; note in caveat |
| R7 | New unsafe FFI in a soundness-adjacent crate | Med | Thin runtime-API wrappers only; committed bytes unchanged; document; no `core/` touch |

### Change surface (rough)
| File | Change | ~LOC |
|---|---|---|
| `stwo_cuda/cuda/utils.cu` | + stream/event FFI (create/destroy/record/wait/sync) | ~60 |
| `stwo_cuda/bindings.rs` | + extern decls for the above | ~25 |
| `prover/backend/cuda/fused_commit.rs` | pinned RAII stash value + event handles; A reclaim; pinned+async D2H in `dehydrate_column`; async H2D + event in `rehydrate_owned`/`rehydrate_block`; event-aware `clear_stash` | ~150 |
| `prover/backend/cuda/poly.rs` | producer loop: issue D2H on `S_copy`, don't sync per-col (Part A); OODS consumer `stream_wait_event` | ~40 |
| `prover/backend/cuda/blake2s.rs` | build_leaves: double-buffered rehydrate + event gate before absorb | ~60 |
| `prover/backend/cuda/quotient.rs` | `rehydrate_block` double-buffer + `stream_wait_event` before kernel | ~40 |
| `evaluate_gate_air.cu` / `gate_air_entry.cu` / `gate-air-cuda-kernel/lib.rs` | composition staged-tile H2D on `S_copy` + `stream_wait_event` before kernel | ~50 |
| **Total** | | **~425** |

### Phased plan (land + validate each before the next)
- **Phase A — reclaim without per-column host sync.** Replace `cuda_stream_reclaim_freed(0)`'s
  `cudaStreamSynchronize(0) + TrimTo` per-column pattern. Options to present/measure:
  (a) set `cudaMemPoolAttrReleaseThreshold` to a **bounded** value (e.g. hold ~1–2 segments) so
  stream-ordered `cudaFreeAsync` blocks get reused without a manual trim/sync; (b) a **dedicated copy
  stream** with stream-ordered free/alloc so freed blocks return to the pool without draining stream
  0; (c) trim only every K columns instead of every column. Validate: [A]/[B]/[C]/[D] (esp. 2^24
  no-OOM — A must not reintroduce the OOM). **This alone should remove several seconds and is the
  precondition for B to overlap.**
- **Phase B1 — commit-side async D2H (pinned + copy stream).** Switch stash value to pinned; issue
  D2H on `S_copy` in `dehydrate_column`, record `E_fill`; don't sync per column; gate readers on
  `E_fill`. Validate battery + re-measure tree1 commit shrank.
- **Phase B2 — consumer-side async H2D (pinned + copy stream, double-buffered).** Wire
  `rehydrate_owned`/`rehydrate_block` to async H2D on `S_copy` + return gating event; add
  `stream_wait_event` before build_leaves absorb, OODS, quotient, composition kernels; double-buffer
  the tiles. Validate battery + racecheck + re-measure prove_ex shrank + `t_base@2^24` in the §5 band.

Each phase is independently revertable and independently byte-validated; B2 is the largest and
riskiest (multiple consumers), so it lands last.
