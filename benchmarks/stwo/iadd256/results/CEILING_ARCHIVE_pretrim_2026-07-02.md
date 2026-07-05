# ARCHIVE — verbatim pre-trim content of RECURSION_PLAN.md (lines 261–501), 2026-07-02

This is the EXACT text removed from `../RECURSION_PLAN.md` when it was slimmed from 508→312 lines on 2026-07-02
(decisions/understandings kept in the plan; measurements moved to `BOX_VALIDATION_LOG.md`). Preserved here VERBATIM
so nothing is lost. Nothing in this file should be edited — it is a frozen snapshot. Current authoritative homes:
decisions+understandings → RECURSION_PLAN.md; measurements+run outcomes → BOX_VALIDATION_LOG.md.

---

## Latest measured state — CHECKPOINT (2026-07-01, LDE-STREAMING CEILING: row-tiling correct; stopgap (i) FAILED on box — ceiling is STRUCTURAL, not an allocator bug)

### Where we are (one paragraph)
Goal: lift the gate_air base-shard ceiling 2^23 → 2^24/2^25 so the recursion-bound curve improves (4× fewer
shards at 2^25). Approach = route (c): stream the tree1 commit (D2H each eval col to a host stash, free device)
+ ROW-TILE the composition/quotient kernels (they re-read all 188 eval cols → would re-OOM otherwise). Steps 1+2
(fused streamed commit + host-stage) and step 3 (row-tiling) are IMPLEMENTED behind `GATE_AIR_STREAM_COMMIT`
(default OFF; legacy byte-identical). Box-validated 2026-07-01: **[A] byte-identity @2^22 (`ab1e75b5`, both
paths), [B] streamed==resident @2^23 (`f304b5ee`), [C] tile-invariance across TILE_ROWS 2^18/2^20/2^22 — ALL
PASS.** So the tiling is CORRECT. BUT **[D] 2^24 and [E] 2^25 OOM** during the tree1 commit (before composition).

### The blocker (RE-DIAGNOSED 2026-07-01 by full residency accounting — file:line) — earlier ROOT-#1/#2 story was WRONG
NOT the tiling, NOT an allocator-plumbing bug (plumbing verified correct), and NOT an un-freed-coeff leak (the
earlier "ROOT #2 coeffs never freed" was WRONG — `store_polynomials_coefficients=false` so each coeff poly is
dropped+device-freed at poly.rs:699). The real cause is a **~2× duplication of the whole main trace, in the
gate-air GPU driver (not shared stwo)**. Complete accounting of trace-sized residents (K1 → tree2 commit):
- **#1 `d_cols`** — K1 column-major output, `alloc_zeros(188*padded_rows)` (gpu_tracegen.rs:1452). **11.75 GB @2^24
  / 23.5 GB @2^25.** Held via `d_main_cols` for the WHOLE prove fn (K4/interaction reuses it, main.rs:3192) — never
  dropped through tree1.
- **#2 188 `d2d_column` copies** — per-column device copies of d_cols (gpu_tracegen.rs:1518-1520), the tree1 commit
  inputs. **11.75 GB @2^24 / 23.5 GB @2^25.** Freed at drain (poly.rs:691, end of tree1). THESE are the duplication.
- **#3 188 extended 2×-blowup evals** — `new_zeroes(2*rows)` per col (poly.rs:441). **23.5 GB @2^24 / 47 GB @2^25.**
  Collapsed to ~1 col by GATE_AIR_STREAM_COMMIT (dehydrate) — the streaming we built+validated ([A]/[B]/[C] pass).
- Ruled out (checked): NTT is IN-PLACE (no scratch, rfft/ifft.cu), no interpolate 2nd buffer, no coeff retention.
  Nothing else is trace-sized. → **2^25 OOMs in K1** (#1+#2 = 47 GB, before any commit); **2^24 OOMs in tree1**
  (#1+#2+#3). CORRECTNESS not soundness (verifier untouched → caught by [A]/[B]/[C] + K4 claimed-sum cross-check).

### (i) STOPGAP — BOX-VALIDATED 2026-07-01: FAILED (freed #3 only; the OOM is #1+#2, which streaming can't touch)
Implemented `cuda_stream_reclaim_freed` + called in `dehydrate_column`; builds; scoped to streamed path (legacy
byte-exact). Box: **[A]/[B]/[C] PASS** (streaming correct), but **[D] 2^24 & [E] 2^25 STILL OOM** on the same 128
MiB alloc. Clean anchors banked: **2^23 t_base=6.630s, t_leaf=9.008s** (uncontended A100). Why it failed: streaming
only collapses #3; the residency floor is #1+#2 (the driver-side duplication), which streaming never frees. **(i)
and (ii) are BOTH orthogonal to the ceiling — the OOM is the #2 duplication, not the free mechanism.**

### STRUCTURAL FIX = (b): eliminate the #2 duplication (per-column commit off d_cols) — task #22 — GREENLIT 2026-07-01
(a) [regenerate d_cols for K4] is RULED OUT — the 2^25 OOM is in K1 where #1 and #2 coexist during the copy loop;
(a) only removes #1 during tree1, so it does nothing for 2^25. **(b) is the fix; NOT (a), NOT (a)+(b).**
Design: keep #1 `d_cols` resident for K4; STOP materializing #2 (drop the d2d_column loop, gpu_tracegen.rs:1518-1520);
tree1 commit processes ONE column at a time — borrowed view into `d_cols[i*padded_rows]` (from_borrowed_ptr,
owns_memory=false) → copy to 1 reused temp (128 MiB) → **per-column b2n interpolate IN-PLACE on the temp** → extend
→ n2b → blake absorb → dehydrate (existing streaming) → reuse temp. **KEY:** the current interpolate is a SINGLE
BATCHED NTT over all 188 cols (poly.rs:282-291) = a hidden #2-equivalent (needs all 188 coeff bufs resident) — so
interpolate MUST be folded into the per-column loop (per-col b2n == batched b2n, byte-identical; batching is only
launch grouping). No out-of-place ifft exists → the transient temp copy is mandatory to protect d_cols (never write
into d_cols → K4 unchanged, NO reorder, FS order preserved). d_cols is column-major contiguous → view is offset+len.
The (b) fix is TWO parts:

**Part 1 — copy-removal (the design above): sound, DONE modulo the Defect-1 fix.** Borrowed d_cols views +
per-column interpolate off a reused temp. Kills the #2 K1 duplication. poly.rs (per-col interpolate) + fused_commit.rs
(borrowed-view/temp helper) + gpu_tracegen.rs (d2d loop → borrowed views) + main.rs (route tree1 under flag). Fold
in **Defect-1** (`interp_base_log`: drop `- log_blowup_factor`, poly.rs:618). Under the TILED Part-2 (below), also
**remove the fused absorb/stash** from the `interp_in_commit` path (poly.rs:605-737): evaluate_polynomials should
only per-column interpolate+extend+n2b+**dehydrate** into values_list — the leaf build moves entirely to build_leaves
(the fused stash is unconsumable on the mixed tree). Byte-identity-guarded and flag-gated (flag OFF = legacy).

**Part 2 — streaming-aware leaf build: DECIDED 2026-07-02 = TILED, after CONFIRMING tree1 is MIXED-SIZE.**
- CONFIRMED (main.rs:3177-3178): the FUSED_INTERP path feeds the 188 large columns (`extend_polys`) AND `small_main`
  (multiplicity/witness — 2^9 / 2^16 / program) (`extend_evals`) into the SAME `tree_builder` → **tree1 is MIXED-SIZE.**
  So `build_leaves` takes the HETEROGENEOUS path (`all_same_size=false`, blake2s.rs:119) — it does NOT consume the
  fused stash, and the current fused block stashes a large-columns-ONLY layer anyway. **⇒ the fused-stash "1a" as
  coded is DEAD for tree1: build_leaves rebuilds → reads the dehydrated (freed) large columns → use-after-free**
  (the real Defect-2). The earlier "1a is nearly free" scope was WRONG — it conflated the same-size 188-group with
  the whole tree.
- "live-lifting" stands: the lift is an on-the-fly index remap ALREADY in upstream stwo (`vcs_lifted`) + our CUDA
  `build_leaves` (`blake2s_lift_states_kernel`, byte-identical to CPU/SIMD `merkle_lifted.rs:59`); small columns are
  absorbed at native size and lifted into the STATE. build_leaves' heterogeneous orchestration (small groups → lift
  → absorb large → finalize) is ALREADY CORRECT for resident columns. Options 1(materialize)/2(tile-state-lift) stay DROPPED.
- **PRIMARY = TILED leaf build.** Reuse build_leaves' existing correct heterogeneous orchestration; make its LARGE
  (staged) column absorb REHYDRATE the dehydrated columns one-at-a-time (or in row-blocks) via the existing
  `rehydrate_owned` → absorb → free (split the large group's single multi-column `blake2s_update_columns` into
  per-column single-column absorbs — byte-identical by absorb-associativity). Small columns stay resident (native,
  tiny). LOCALIZED change to build_leaves' staged-column absorb; NO producer restructure; reuses the tested lift
  logic. **State size RESOLVED = ~108 B/row (`Blake2sState`: h[8]+t+buf[64]+buflen) → ~7 GB @2^26 (NOT 2 GB).**
  Peak ≈ **34 GB** (d_cols 23.5 + state ~7 + 1 rehydrated col ~0.25 + leaf output ~2), ~6 GB headroom on 40 GB.
  The ~7 GB state array is the dominant swing term (whole-column rehydrate does NOT shrink it) — if the box OOMs,
  the next lever is per-block states (bigger security-critical change): measure-then-decide. Cost: the ~47 GB H2D
  rehydrate — recoverable later via async-overlap (ii).
- **[SECURITY-CRITICAL]** shared CUDA `build_leaves` (blake2s.rs) → supervised protocol. INVARIANT: the tree1 root is
  byte-identical to the flag-off resident build; the change only moves WHERE the large-column bytes come from
  (rehydrated vs resident device ptr) — same values, order, hash. Gated on `is_staged` → flag-off/SIMD/other callers
  unchanged (empty stash → no-op). Mirror the CPU-vs-CUDA equality tests (blake2s_lifted.rs:422-450) as a local
  small-size byte-identity check BEFORE the box.
- **Validation: SAME battery.** Success = [E] 2^25 exit=0 + stable fp; [A]/[B]/[C] byte-identical (flag ON ==
  resident == flag-off). Add a small-size (2^10) MIXED-SIZE resident-vs-staged byte-identity test (mirror
  blake2s_lifted.rs:422-450) as the FIRST, cheapest box check before [E] 2^25.
Note: #1 `d_cols` (23.5 GB) is the floor — scale past ~2^26 would need #1 too (out of scope for 2^25). Fused-1a-proper
(faster — no rehydrate — but re-implements build_leaves' heterogeneous accumulation in the producer → code
duplication) and Option-2 (tile the state-lift) are both parked in **Deferred micro-optimizations**.

**BOX VALIDATION STATUS (2026-07-02, run 3+4):** Full-(A) FIXED THE CRASH — the tiled/streaming path now runs
end-to-end at 2^22 (quotient/composition/FRI/decommit all complete, no quotient.rs:300 panic, no illegal address).
BUT it now fails a CORRECTNESS check: `Error: Constraints not satisfied` (prover/mod.rs:186-198 — the committed
composition-poly OODS value (A) != CPU re-derivation from sampled TRACE OODS values (B); they see INCONSISTENT data).
This is a rejected proof (correctness), NOT soundness. **Isolation (box, 2^22 samples=1):** FUSED_INTERP alone →
PASS (fp==oracle); STREAM_COMMIT alone → PASS; BOTH → FAIL; BOTH+single-tile (GATE_AIR_TILE_ROWS=2^24) → STILL FAIL.
⇒ (1) staging (dehydrate→host stash→rehydrate/H2D) only actually happens with BOTH flags (dehydrate_column sits in the
fused-interp branch, gated by stream_tree1); each flag alone hits only RESIDENT branches (correct). (2) single-tile
failing RULES OUT tiling boundaries/biasing — bug is in the STAGED DATA PATH (dehydrate/stash/rehydrate/keying), not
row-tiling. My composition resident-unbias .cu fix (biased{0,1}[c] = trace{0,1}_evaluations[c], no -tile_start; lines
~692/711) was for a NON-bug (single-tile still fails) but is a legit correctness improvement for multi-tile — KEPT.
**Leading hypothesis: stash-key POINTER ALIASING.** HOST_STASH (fused_commit.rs) is keyed by the FREED
`col.device_ptr as usize`; is_staged/staged_host_ptr match key + `h.len()==col.size` size guard; clear_stash is
once-at-top. A resident tree0(preproc)/tree2(interaction) col — same EVAL-DOMAIN size, passes the size guard — reusing
one of the 188 freed tree1 addresses would make staged_host_ptr(resident_col) return STALE tree1 bytes → composition
(lib.rs:213-222 host0/host1) and/or OODS sampling fed wrong bytes → A!=B. **Agent adc5cfb6013af84d0** investigating +
minimal fix (candidate: robust stash key / explicit staged flag vs raw reusable device_ptr; or tighter stash scope),
laptop compile only, box held. HISTORY: earlier crash was quotient.rs:300 "not host-staged" (pre-full-(A)); full-(A)
resident-branch fixed that (Site 1 quotient + Site 2 composition per-column staged/resident + Site 3 tree2 resident).

### RESUME-AFTER-COMPACT (2026-07-02) — full-(A) DONE, awaiting box
**Target design (FULL (A), IMPLEMENTED):** under FUSED_INTERP+STREAM, ONLY the 188 large tree1 eval columns are
dehydrated (host-staged); small tree1 (mult/witness 2^9/2^16) + tree0 + tree2 stay RESIDENT. Every column consumer
does `is_staged(col) ? rehydrate/slice-from-stash : slice/read-live-device-ptr`.
**Verification finding (load-bearing):** the composition kernel reads `trace_evaluations[col][global_row]` with a
SINGLE shared `eval_domain_log_size` for all 188 tree1 + 4 tree0 + 24 tree2 cols (`eval_at_row.cuh:159`, global row
= `row_offset+local` set at `evaluate_gate_air.cu:222`) — NO per-column lift remap in the read. So a resident col
just biases its live buffer by `-tile_start`, identical to a staged tile's bias → resident-vs-staged is a data-SOURCE
difference only, never a value difference → byte-identical by construction. (Live-lifting is a Merkle-COMMIT
mechanism — lift on the hash-state array, native-size col data — and doesn't bear on the composition read.)
**Implemented (5 files, compiles clean both trees, laptop):**
  - `quotient.rs:~306-325` (Site 1): staged→`rehydrate_block` / resident→`from_borrowed_ptr(device_ptr.add(off))` +
    fail-loud bounds assert.
  - `evaluate_gate_air.cu` (Site 2 kernel): `tiled_input = host0||host1` (was all-or-nothing); tile buffer alloc'd
    ONLY for staged cols; per-block loop branches per-column staged→H2D+bias / resident→`traceX_evaluations[c]-tile_start`
    live buffer; free loop guards null. `gate_air_entry.cu:13-20` FFI doc updated.
  - `gate-air-cuda-kernel/src/lib.rs:~204-232` (Site 2 caller): per-column `Vec<*const u32>` (staged ptr or `null()`
    resident sentinel) replacing `Option<Vec>::collect()`; table passed iff any non-null; tree0-resident no longer
    forces `tiled_input=false`.
  - `poly.rs` (Site 3 / Fix-B revert): `dehydrate_column` fires ONLY for the 188 large main cols; small tree1 +
    tree0 + tree2 never dehydrated → resident.
  - tree2 (24 LogUp-cumsum cols, scattered bit-reversed `-1` offset) stays resident + read whole (never tiled).
  Kept: large-col staging + build_leaves per-group rehydrate + Defect-1 (interp_base_log) + blake2s.rs rehydrate-loop
  `cuda_stream_reclaim_freed`. No shared `evaluate_common.cuh`/verifier touched.
**NEXT (box, when user greenlights):** `box-ops/start.sh` (A100 anat-ganor-instance, ~1 attempt) → `box-ops/sync.sh`
→ build `--features cuda,diag` (nvcc compiles the `.cu` — NOT compiled on laptop) → run `~/tiled2_validate.sh`;
drive the poll from the MAIN loop.
**Box battery facts:** flags `CUDA_GPU_CONSTRAINTS=1 GATE_AIR_FUSED_INTERP=1 GATE_AIR_STREAM_COMMIT=1`; fixture
`~/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json`; oracle fp `ab1e75b53d49645c09982ed5c890d8880366b0197b554ea5300dcd0f66838ba4`
(@2^22 samples=1); samples 1/2/4/8 → 2^22/2^23/2^24/2^25. **WIN = [E] 2^25 exit=0 (no OOM) + fp stable + [A]/[B]
byte-identical.** Battery EARLY-STOPS at [A] if the tiled fp ≠ oracle (a short log ending VALIDATE_DONE after [A]
= failed-at-A). NOTE: run the battery poll from the MAIN loop (sonnet runners bail on the poll loop).
**Box run history (both FAILED at quotient.rs:300 "not host-staged"):** run1 (STREAM only) small cols resident →
panic; run2 (after (B) uniform-staged tree1) STILL panicked because quotient/composition also reference RESIDENT
tree0/tree2 cols → hence full-(A). tree1 commit (build_leaves) itself is byte-clean at 2^22 (completes, not the culprit).
**Corrected CPU-curve numbers (don't re-derive wrong):** per-shard base is FAST-code ~11-13s @2^24 (satsweep2
`S=4 K=1 wall=13s`) / ~23s @2^25 — NOT the stale 37.6s (that was 2^25 OLD slow-interaction-gen). The ~11× vs SP1 at
k=2000 (~8.3h vs 45m53s) comes from `total_rows(k)=22.984M·k` (k=2000 → 46G rows) ÷ SATURATED base throughput
~1.6 M rows/s (memory-bandwidth plateau, satsweep2 K=8=1.63), NOT per-shard latency. Curve in OPTIMIZATION_PLAN.md
is correct (uses 1.6 M/s); only my verbal per-shard walkthrough was wrong.

### (ii) DEFERRED — the async-overlapped version — NOTE: ORTHOGONAL to the ceiling, does NOT fix the OOM
- **Orthogonality (2026-07-01):** (ii) refines the SAME free-the-eval-outputs (#3) mechanism as (i) — it does NOT
  touch the #1+#2 main-trace duplication that actually OOMs, so on its own it OOMs at the SAME place. (ii) is a
  THROUGHPUT win (hide PCIe) to layer on AFTER (b) fits 2^25, never a substitute for (b).
- **Why:** at 2^25 the PCIe (~6-9s) is ~25-30% of t_base, so hiding it is the difference between ~2.1× (i,
  un-hidden) and **~1.6×** (ii, hidden). Also (i)'s per-column `cudaStreamSynchronize` is fundamentally
  ANTI-overlap → it must be ripped out to overlap, so (ii) is not built on (i).
- **What:** (A) fix the allocator so freed blocks reuse WITHOUT a per-column host sync (bound the release
  threshold and/or dedicated copy-stream stream-ordered free/alloc) — so the pipeline CAN overlap; (B) wire the
  PINNED + async DOUBLE-BUFFERED copies (primitives exist unwired): commit-side D2H each eval col on a copy
  stream overlapping the next col's NTT/absorb; composition/quotient-side H2D each row-block on a copy stream
  overlapping the prior block's kernel. Target `t_base@2^25` ≈ compute (~25s, PCIe hidden) → ~1.6×.
- **The async-race risk (CORRECTNESS, not soundness):** if a kernel reads a row-block before its H2D finished, or
  a buffer is freed before its D2H copied the bytes, the prover computes over stale bytes → a WRONG proof that
  FAILS verification (rejected) — NOT a wrong proof that verifies (verifier/constraints/FRI/Fiat-Shamir all
  untouched). Caught by byte-identity ([A]/[B]/[C]) + the proof self-verify. Fix needs CUDA EVENTS for cross-stream
  ordering (H2D-complete event gates the kernel; free waits on both absorb AND D2H). Task #20.

### NEXT STEP AFTER (i) lands — box session: what to measure and WHY
Build `--features cuda,diag` on the A100 (box-ops/), re-run the battery `~/box_tile_validate.sh`:
1. **Re-confirm correctness holds:** [A] fp==`ab1e75b5`@2^22 both paths, [B] streamed==resident@2^23, [C]
   tile-invariance. (The stopgap must not have perturbed byte-identity.) — WHY: the reclaim change is in the
   streamed path; confirm it's still byte-exact.
2. **Does 2^24 / 2^25 now FIT?** [D] samples=4 (2^24), [E] samples=8 (2^25): exit=0, fp printed, NO out-of-memory,
   warn=0. — WHY: this is the go/no-go on whether the ceiling actually lifts (and whether the resident
   witness+tree2 wall bites at 2^25 — if [E] OOMs but [D] fits, 2^24 is the ceiling).
3. **t_base@2^24 and t_base@2^25** = wall + [prove_ex] composition_eval/TOTAL + the phase breakdown. — WHY: this is
   the BASE ANCHOR for the curve. Un-overlapped, so composition_eval will include the un-hidden PCIe (that excess
   over the ~1.1s/2.2s pure compute IS the PCIe tax — quantifies what (ii) would recover).
4. **Leaf-scaling: t_leaf@2^25 (or 2^24) vs t_leaf@2^23** ([F] fold, 1 shard each). — WHY: the recursion win from
   bigger shards depends on how t_leaf grows with base size. Fewer shards helps ONLY if t_leaf grows slower than
   the shard-count drop (4× at 2^25). Measured @2^23 already: t_base=6.68s, t_leaf=9.16s.
5. **Compute the real curve:** swap measured `t_base@2^25` (or 2^24) + the leaf-scaling into
   `results/extrapolate_8g.py` (SHARD_LOG + T_BASE_2P25 + LEAF/NODE placeholders it already has, marked TBD-from-box).
   — WHY: compare the REAL 2^25 curve to the ~2.1× projection AND to 2^23's ~3.3× → decide whether the ceiling
   lift helps enough to justify (ii)'s async work. If real ≈ 2.1× and beats 2^23, (ii) is justified (→ ~1.6×). If
   2^25 doesn't fit or doesn't beat 2^23, stop the ceiling work and pivot to GPU recursion (GPU_RECURSION_SCOPE.md).

### Key state for continuity (flags / anchors / tools)
- **Flags** (all opt-in, default OFF = legacy byte-identical): `GATE_AIR_STREAM_COMMIT` (route c: fused stream
  commit + free-after-absorb + host-stage + row-tiled composition/quotient), `GATE_AIR_FUSED_COMMIT` (step-1
  fused-but-resident, the intermediate A/B state), `GATE_AIR_TILE_ROWS` (tile size, default 2^20, alias
  `GATE_AIR_COMP_TILE`), `GATE_AIR_NTT_SUBBATCH` (extend batch, default 48), `CUDA_GPU_CONSTRAINTS=1` (GPU
  constraint kernel — must be ON; #14), `GATE_AIR_PROOF_HASH=1` (print fingerprint), `PROVE_EX_TIMERS=1`.
- **Fingerprints:** base oracle @2^22 = `ab1e75b5…`; @2^23 (samples=2) = `f304b5ee…`.
- **Anchors:** t_base@2^23=6.29-6.68s; t_leaf@2^23=9.16s (A100 box, 12 vCPU); recursion sweep optimum (stwo-vm
  96-vCPU c4) K=16/T=6, t_leaf=15.8/t_node=10.7 (contended, DDR5-optimistic). Curve model = results/extrapolate_8g.py
  (two-balance overlap: `T=max(⌈N/8⌉·t_base, N·(t_leaf+t_node)/16)+tail`).
- **Docs:** results/LDE_STREAMING_SCOPE.md (the design, §3/§4 corrected: route(a) ruled out, ceiling = route(c) +
  row-tiling), results/COMPOSITION_TILING_SCOPE.md (the kernel row-tiling), results/GPU_RECURSION_SCOPE.md (the
  pivot if the ceiling stalls). Tooling: box-ops/ (A100) + box-ops/stwo-vm/ (96-vCPU CPU sweeps).
- **Sync note:** WIP = 1 new file (fused_commit.rs) + 34 modified across 3 repos (much is cargo-fmt noise);
  box-ops/sync.sh is git-diff-driven + catches untracked source files. Both boxes currently TERMINATED.

## Latest measured state — CHECKPOINT (2026-06-30, box session #14: GPU FAST PATH RESTORED)
- **#10's curve was an ARTIFACT — now fixed and re-measured.** Root cause was NOT the `.cu`: the Rust
  structural decline guard `is_gate_air_main` in `gate-air-cuda-kernel/src/lib.rs` still demanded 191 main /
  1 preprocessed, so the 188-col main component failed it → kernel DECLINED → silent host-delegate. Fix:
  `GATE_AIR_TRACE_COLUMNS 191→188`, `GATE_AIR_PREPROCESSED_COLUMNS 1→4` (N_CONSTRAINTS stays 157, interaction 24).
  A cross-cutting layout change must touch THREE places: K1/K4 trace-gen, the `.cu/.cuh`, AND this guard.
- **Box result:** composition_eval **53.4s → 0.271s @2^22 (197x)**; prove TOTAL 1.36s@2^22 / 2.27s@2^23;
  host-delegate WARNING now SILENT under CUDA_GPU_CONSTRAINTS=1; fp==`ab1e75b5` (byte-identical to SIMD oracle).
  Single-shot GPU @2^23 phase-sum ≈ 5.0s (tracegen+commit ~2.7s + prove_ex 2.27s; wall 11.2s incl one-time fixture/precompute).
- **Recursion anchors (CPU, base-size-independent, reused from #10):** t_leaf 6.0s, t_node 6.2s, t_root 4.8s.
  Balance B `W = 8·(t_leaf+t_node)/t_base` now FLIPS ≫1 (recursion-dominated) — base is single-digit seconds,
  so the lever moves to CPU recursion / GPU-recursion (C2/C1, R3).
- **Tooling:** durable `box-ops/` toolkit (start/sync/build/validate/measure + force-stop + README + LAST_STATE).
- **Corrected 8g curve (results/extrapolate_8g.py — TWO-BALANCE overlap model; box-ops/curve.py is SUPERSEDED).**
  curve.py's serial `(N·t_base+2N·(t_leaf+t_node))/min(8,N)` was invalid: it SUMS base+recursion (no
  overlap) and caps recursion at 8-way (it gave a bogus 10.2x@k2000). Correct model: `base_wall=⌈N/8⌉·t_base`
  (8 A100s) ‖ `rec_wall=N·(t_leaf+t_node)/P_8g` where `P_8g=8·P_1g·η` (Balance A pools=2 on 12 vCPU,
  Balance B η∈[0.30,0.695]); `T=max(base_wall,rec_wall)+tail`.
  (older run @k=2000 gave 2.2x–7.2x — but that combined P_8g=16 with LOW-CONTENTION anchors, now known inconsistent.)
- **RECURSION POOL SWEEP (stwo-vm 96-vCPU c4, 2026-07-01, box-ops/stwo-vm/SWEEP_RESULTS.md):** optimal split
  **K=16, T=6** (all leaves one wave; peak 0.865 leaves/s); memory BW saturates at K≥8 (t_node plateaus ~10-11s).
  KEY: P_8g and per-fold latency are COUPLED by memory BW — at K=16 the folds are saturated (**t_leaf=15.8s,
  t_node=10.7s on c4**, ~2.6x the K=2 low-contention 6.0/6.2). So the old 2.2x was too optimistic (16 pools ×
  uncontended folds is impossible). **Corrected curve (extrapolate_8g.py, P_8g=16 + coupled anchors):
  @k=2000 = 3.3x (c4/DDR5, optimistic) … 4.9x (a2 DDR4-adjusted, est ×1.5).** k=1 WINS (0.7x); recursion-bound
  at every k≥10. Remaining uncertainty = the DDR4 absolute (bounded-not-measured; needs a 96-vCPU Cascade-Lake
  box we don't have). **Lever confirmed: escape the CPU memory-BW wall → GPU recursion (R3); more CPU pools
  just saturate DDR.** Base ceiling: LDE-streaming scope done (results/LDE_STREAMING_SCOPE.md) — go straight to
  host-stage route (c) for 2^25 (route (a) is a fallback).

### (superseded) CHECKPOINT — box session #10
- **New oracles (188-col witness-shrunk AIR):** base proof_fingerprint @2^22 = `ab1e75b5…`; whole-tree recursion_fingerprint = `07272a7a…`. Old `0448237288…` retired.
- **VALIDATED byte-identical:** precompute A/B (GATE_AIR_NO_BASE_PRECOMPUTE on==off), K1/K4 at 188 cols, distinct-shard fold verifies + self-verifies.
- **t_base@2^23 = 115.6s — BUT THIS IS THE HOST-DELEGATE PATH (INVALID for the curve).** composition_eval = 107.85s = 93-98% of t_base. Cause: the GPU constraint kernel `evaluate_gate_air.cu` was stale for 188 cols → silent host-delegate fallback. **#14 restores it (kernel fixed; expected composition_eval ~108s→~1.8s, prove ~1.8s @2^22 per the prior anchor).** The 35x-loss curve from #10 is an ARTIFACT — RE-MEASURE after #14.
- **Phase memory-peak (the #12-decider):** 2^24 OOMs at TREE COMMIT (main-trace LDE+Merkle, rfft.cu), held flat through composition_eval — NOT decommit. So #12 must stream/tile the main-trace LDE, not the Merkle/decommit (the option-1 stage-before-decommit does not help).
- **Recursion anchors (12-vCPU box):** t_leaf 6.0s, t_node 6.2s, t_root 4.8s (~base-size-independent). Balance A: per-fold scales ~linearly with threads up to 12 (no saturation), 2×6 pools best on 12 vCPU. Balance B on the (wrong) host-delegate t_base: W≈0.84 (GPU-bound) — this FLIPS to CPU-bound/recursion-dominated once t_base is corrected to ~6s.
- **NEXT:** #14 (kernel fix done + the (a) host-delegate warning) → box re-measure (GPU vs host A/B: composition ~1.8s, fp==ab1e75b5, warning fires only on host) → real curve vs SP1 → then recursion levers (C2/C1, GPU-recursion) become primary; #12 (LDE ceiling) lower-priority once base is fast.
