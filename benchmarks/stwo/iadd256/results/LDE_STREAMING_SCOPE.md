# Main-Trace LDE-Streaming SCOPE — lifting the gate_air base ceiling 2^23 → 2^24 (→2^25)

**Status:** scope only (read-only; no source edits, no box). 2026-07-01.
**Target:** raise the gate_air GPU base-shard ceiling from 2^23 to **2^24** (stretch **2^25**)
on a 40GB A100, in `stwo-cuda-backend`.
**The peak this scope attacks:** the OOM at the #10 phase-peak is at **TREE COMMIT of the main
trace** — the forward LDE (extend→NTT to the eval domain) **plus** the lifted-Merkle leaf build.
The peak is reached at commit and held flat through composition_eval. It is **NOT** the
decommit/query phase.

> **Blowup = 1 for base shards.** `BASE_LOG_BLOWUP_FACTOR = 1`
> (`grover-tax-v02/gate-air-leaf/src/main.rs:91`); `leaf_pcs_config` at blowup 1 =
> `pow_bits 26, n_queries 70, fold_step 4, lifting_log_size = Some(trace_log + 1)`
> (`gate-air-leaf/src/leaf.rs:54-71`). The eval/lifted domain is therefore **`2^(log_n + 1)`**
> (blowup 3 in `main.rs:2375` is the *recursion* layer, not the base shard). **Dispositive check:**
> 2^23 base shards demonstrably fit 40 GB and prove (fingerprint `ab1e75b5`); at blowup 3 the
> resident eval set @2^23 would be `188 × 2^26 × 4B ≈ 50 GB > 40 GB` → would OOM at 2^23,
> contradicting reality. So blowup 1 is the only value consistent with 2^23-fits / 2^24-OOMs. AIR =
> **188 main** (tree1) + preprocessed (tree0) + 24 interaction (tree2).

> **How this differs from / composes with the prior P-S* scope (`STREAMING_MERKLE_SCOPE.md`).** That
> scope attacked the *Merkle-hash* side of commit (row-wise leaf appears to need all columns resident)
> and the *decommit residency* (eval columns must survive to `decommit`); its P-S0/1/2 stream the
> **hash absorb**. This scope attacks the **forward LDE** that *produces* the eval columns — the
> `evaluate_polynomials` extend+NTT — and shows the LDE and the Merkle build can be **fused** into one
> per-column pass so no column set is ever fully resident. The two compose: the LDE-stream feeds
> columns one at a time into P-S0's already-landed streamed absorb (blake2s.rs:140-165), and
> **P-S1 (keep-coeffs+re-LDE) / P-S2 (host-stage)** supply the post-commit + decommit recovery. The
> prior scope's blowup-1 accounting was **correct**; this doc reuses it and adds the fused-forward-LDE
> lever.

---

## 1. Memory accounting — where 40 GB goes at the ceiling (blowup 1)

### Per-column physical sizes (M31 = u32 = 4 B; `base_field_vec.rs`)

| Trace log-size L | trace-domain col `2^L·4B` | **eval/lifted col `2^(L+1)·4B`** (blowup 1, lift +1) |
|---|---|---|
| 2^22 (anchor) | 16 MiB | **32 MiB** |
| 2^23 (current ceiling) | 32 MiB | **128 MiB** |
| 2^24 (OOMs) | 64 MiB | **256 MiB** |
| 2^25 (stretch) | 128 MiB | **512 MiB** |

`extend` (poly.rs:416-427) allocates the **full eval-sized** buffer `BaseFieldVec::new_zeroes(2^(L+1))`
**per column** (not one monolithic block), zero-filled, copies the `2^L` coeffs in; the NTT then runs
**in-place** on that buffer (rfft.cu:658,666 write back into `device_values`; only extra alloc is the
`num_poly·8B` pointer array, rfft.cu:680). So the main-trace resident cost after
`evaluate_polynomials` is exactly **188 × (eval-domain col)**, and the transient during extend is
bounded by the current sub-batch (`GATE_AIR_NTT_SUBBATCH`, default 48, poly.rs:540) worth of eval-sized
buffers — not a separate trace copy.

### The main-trace resident set at tree1 commit — the wall

| Block | code path | 2^23 (fits) | **2^24 (OOMs)** | 2^25 |
|---|---|---|---|---|
| **188 main eval cols** (blowup 1) | `evaluate_polynomials` → held in `CommitmentTreeProver.polynomials` (pcs/mod.rs:412-438) | **~24 GB** | **~47 GB** | ~94 GB |
| trace-domain coeffs (if `store_coeffs`) `188·2^L·4B` | `evaluate_polynomials` (store_polynomials_coefficients) | ~6 GB | ~12 GB | ~24 GB |
| Merkle leaf layer `2^(L+1)·32B` | `build_leaves` (vcs_lifted/prover.rs:68) | 0.25 GB | 0.5 GB | 1.0 GB |
| Merkle upper layers (~1× leaf) | `build_next_layer` chain (prover.rs:71-73) | 0.25 GB | 0.5 GB | 1.0 GB |
| twiddles (fwd+inv, `2^(max_log+1)`, shared) | `precompute_twiddles` (poly.rs:594) | ~0.13 GB | ~0.25 GB | ~0.5 GB |

**Verdict:** at 2^23 the eval set (~24 GB) + coeffs (~6 GB) + tree/twiddles ≈ **~30 GB → fits 40 GB**.
At 2^24 the **188-column eval set alone is ~47 GB > 40 GB** — it OOMs on a 256 MiB-class per-column
allocation inside the sub-batched extend+NTT in `evaluate_polynomials` (poly.rs:526-584), because each
freshly extended column is **appended to `values_list` and never freed** (poly.rs:569): the subsequent
`MerkleProverLifted::commit` (pcs/mod.rs:428) reads **all** of them, and they are then retained in
`CommitmentTreeProver.polynomials` for OODS/quotient and `decommit` (pcs/mod.rs:412-459).
`GATE_AIR_NTT_SUBBATCH` bounds the extend *spike* but not the *accumulated resident set*. **The
188-column resident EVAL set is the allocation that blows the budget at 2^24 (~47 GB).** (If
`store_coeffs` is on, the +12 GB of coeffs makes it worse still; base runs it off — barycentric OODS.)

---

## 2. Options to stream/tile the main-trace LDE+Merkle

> **CORRECTION (2026-07-01, verified by the step-2 refactor):** the peak reductions below are for the
> **COMMIT** only. They do NOT lift the ceiling on their own, because **composition + quotient are a
> second full-residency consumer** of all 188 eval columns (§3). Bounding *that* requires ROW-TILING the
> constraint/quotient kernels (a kernel-signature change to `evaluate_gate_air.cu`), fed per-row-block by
> the route-(c) host stash. Route (a) below is **ruled out** for composition (global NTT can't emit a
> row-block). Read the per-option numbers as commit-peak reductions, and §3/§4 for the real path.

Enabling CUDA facts (confirmed in rfft.cu):
- **NTT is in-place and per-column-independent** — `ntt_n2b_columns(ptr, log_n, num_poly=1, …)` is
  **byte-identical** to that column inside a `num_poly=188` batch (columns are a grid dimension, no
  cross-column mixing; rfft.cu:647,166; asserted at poly.rs:535-537). The LDE can be driven one column
  (or small block) at a time with **zero** change to the transform result.
- **Twiddles allocated once and shared** across all columns (rfft.cu:642-643) — streaming does not
  duplicate them (~256 MiB @2^24, small vs columns).
- **The streamed Merkle absorb already exists and is validated** — P-S0's
  `blake2s_alloc_init_states → blake2s_update_columns(per col) → blake2s_finalize_all`
  (blake2s.rs:140-165, gated `GATE_AIR_STREAM_MERKLE`), proven byte-identical
  (`test_build_leaves_streamed_same_size_vs_cpu`, blake2s.rs:480). **But P-S0 does NOT free after
  absorb** → its peak is still all-188 eval cols + state array (~0.75 GB@2^24 for `2^(L+1)·~48B`
  states) + leaf buffer. **P-S0 gives byte-identity only, no ceiling change.** This scope's lever is
  adding the **free-after-absorb + forward-LDE fusion** on top of P-S0.

### (a) **Fused per-column LDE → Merkle-absorb → free** — RECOMMENDED lever

Mechanism: allocate the per-row Blake2s state array once; loop columns in the exact `commit`-sorted
order (prover.rs:65) — for each column: `extend`→`ntt_n2b_columns(num_poly=1)`→
`blake2s_update_columns(states, evals_c)`→**free `evals_c`**. `finalize_all` → leaf layer; the
`build_next_layer` chain is unchanged.

- **Peak reduction:** commit residency drops from `188 × eval-col` to
  `state_array + O(block)·eval-col + Merkle tree (+ kept coeffs, see §3)`. State array ≈
  `2^(L+1) × ~48B` ≈ 0.75 GB @2^24 (1.5 GB @2^25). With a block of B columns in flight,
  transient = `B × 256 MiB` @2^24. At B=4: ~1 GB block + 0.75 GB states + ~1 GB tree ≈ **~3 GB @2^24**
  commit peak, vs ~47 GB. **2^25 commit fits** (~1.5 GB states + 2 GB block + 2 GB tree ≈ 5.5 GB).
- **Effort: M.** The absorb primitive and its byte-identity check are landed (P-S0). New work = the
  fused *forward-LDE* loop in `evaluate_polynomials`/`CommitmentTreeProver::new` and freeing each
  column — crosses the pcs↔poly↔vcs_lifted seam; SECURITY-CRITICAL (`vcs_lifted`).
- **Risk: Med** — column **order** is load-bearing (must replay `sorted_by_key` exactly, prover.rs:65);
  freeing evals collides with post-commit OODS/quotient + decommit (§3).
- **CUDA specifics:** already-supported — `ntt_subbatch_size()` proves per-block extend is
  byte-identical; set the LDE block = the absorb block; reuse P-S0 kernels verbatim.

### (b) **Tile the NTT/rfft itself (sub-column)** — NOT viable

The circle NTT is **globally coupled across all rows** of a column (butterfly layers span the whole
`2^(L+1)` buffer; twiddle indexing is by global position, rfft.cu:206,344). A single column cannot be
transformed in isolated row-tiles without a full out-of-core FFT. And one column (256 MiB–512 MiB) is
**not** the wall — the wall is *number of columns resident*. Option (a) already reduces to O(one
column). **Effort L, Risk High, no ceiling benefit** until L ≥ ~2^27. Listed only to rule out.

### (c) **Host-stage eval columns to pinned host memory during commit**

Mechanism: same fused loop as (a), but after absorbing column c, D2H-copy `evals_c` to a pinned host
buffer, then free device. Host has 80+ GB. Same device commit peak as (a). The **difference from (a)**
is on the post-commit side (§3): (c) keeps exact eval bytes on host for cheap H2D of columns/queries;
(a) frees them and must recompute.

- **Effort: M** (reuses P-S0 absorb + the landed `query_to_buffer_index` host-recovery, prover.rs:347).
- **Risk: Med** — PCIe throughput, pinned-mem lifecycle. D2H of 188 × 256 MiB ≈ 47 GB @2^24 over
  ~16 GB/s ≈ 3 s/tree (hidden behind NTT compute if async). This is the **P-S2 route** and the clean
  path to 2^25.

---

## 3. The hard dependency — OODS/quotient/composition + decommit read the eval columns AFTER commit

This is the crux prior P-S2 hit. Freeing eval columns during a streamed commit breaks two later
readers — **different fixes:**

1. **OODS / quotient / composition (the #10 peak plateau, held flat through composition_eval).** With
   `store_polynomials_coefficients = false` (base default), `Poly::eval_at_point` runs
   `barycentric_eval_at_point` over the **FULL eval column** on device (poly.rs:400-414;
   component_prover path pcs/mod.rs:200-249) — a **dense read of every row of every one of the 188
   columns**, *after* tree1 commit, *before* decommit. Eval buffers are freed only **after all
   decommits** (pcs/mod.rs). **You therefore CANNOT free the main eval columns before OODS/quotient
   on a naive recompute** — that would force an immediate full 188-column re-LDE with no residency win
   during composition. This is exactly why prior **P-S2 host-staging was stopped**: OODS/quotient read
   all eval columns after commit but before decommit.

   **Resolution — freeing at commit is NECESSARY BUT NOT SUFFICIENT; composition + quotient must be
   ROW-TILED (the real ceiling-lift work).** Composition (constraint eval) and the quotient are
   row-parallel but read ALL ~188 columns SIMULTANEOUSLY per row, so any recovery that re-materializes
   the full eval set — re-LDE-all OR H2D-all — simply RE-CREATES the ~47 GB OOM at composition (the #10
   peak was held flat *through* composition_eval for this reason). The only fix that bounds residency is
   to tile composition + quotient BY ROWS: process a row-block holding only `O(tile_rows × 188)`, then
   the next. (VERIFIED 2026-07-01 by the step-2 perf refactor.)
   - **The landed `GATE_AIR_COMP_TILE` does NOT do this.** It tiles only the composition OUTPUT
     fraction; the gate_air constraint kernel still reads its INPUT columns at GLOBAL row indices
     (evaluate_gate_air.cu). Bounding INPUT residency needs a KERNEL-SIGNATURE change — the kernel must
     accept a row-range and index into a per-block tile buffer. **This is the actual 2^24/2^25 work**,
     and it applies to the quotient (`accumulate_numerators`) too.
   - **Route (c)/host-stage is the ONLY supplier that composes with row-tiling:** D2H each eval column to
     pinned host during the fused commit; then for each composition/quotient row-block, H2D just that
     block's slice of all 188 columns, run the row-tiled kernel, free. Residency = `O(tile_rows × 188)`.
   - **Route (a)/re-LDE is RULED OUT** — it cannot bound composition residency. The NTT is global, so
     re-LDE emits WHOLE columns (never a row-block); to feed a row-tiled composition it would re-LDE
     every column once PER row-block (compute-prohibitive) or hold all 47 GB resident (OOM). So
     keep-coeffs+re-LDE does **NOT** reach 2^24 for composition and is not a viable fallback.

2. **decommit (FRI queries — small, sparse read).** `decommit` reads each column at only
   `n_queries = 70` positions via `col.at((pos>>(shift+1)<<1)+(pos&1))` (prover.rs:114) — tiny. Recover
   from the **host-staged** column (route c; landed `query_to_buffer_index` +
   `test_host_stage_recovery_roundtrip_matches_decommit`, prover.rs:347-449 prove byte-exact host
   recovery) or re-LDE the queried columns from kept coeffs (route a).

### PCIe is the unavoidable cost of the ceiling lift (route c + row-tiling)

- Row-tiled composition/quotient must H2D the full eval set once per proof, streamed in row-blocks
  (~47 GB @2^24 / ~94 GB @2^25 total), overlappable with GPU compute via pinned buffers + a copy stream.
- **There is NO CPU-free path that fits:** the constraint kernel inherently needs all columns per row,
  and re-LDE (route a) can't tile (global NTT). So the ceiling lift = **kernel row-tiling + host-stage +
  PCIe**, full stop.
- **The one open question is whether that tiled H2D OVERLAPS compute** (pinned + async → hidden) **or
  bottlenecks** (t_base@2^25 ≈ compute + PCIe). This is the go/no-go the box must measure. If PCIe
  cannot be hidden, the ceiling lift may not pay off vs. its recursion-curve benefit — and there is no
  route-(a) escape, so the alternative is simply to STAY at 2^23 and pursue the recursion lever instead.

---

## 4. Recommendation + byte-identity validation

### Recommendation

**The ceiling lift = route (c) host-stage + ROW-TILING the composition/quotient kernels. Route (a) is
ruled out** (§3: re-LDE can't bound composition residency). Streaming the *commit* (steps 1–2, done +
validated) is necessary but NOT sufficient — composition re-materializes all 188 columns, so the OOM
just moves from commit to composition. The remaining work is a kernel-level change, not a recovery-path
choice.

- **Sequencing (smallest-validated-step-first, on top of landed P-S0 + query-map + step-1/2 commit stream):**
  1. **DONE + box-validated:** fused per-column LDE→streamed-absorb, commit-only, evals kept resident
     (`GATE_AIR_FUSED_COMMIT`) — byte-identical (`ab1e75b5` @2^22, on==off @2^23). No ceiling move.
  2. **DONE (code):** step-2 free-after-absorb + host-stash (`GATE_AIR_STREAM_COMMIT`); composition/
     quotient/OODS rehydrate to GPU (no CPU delegate). BUT rehydrates the WHOLE read-set → still OOMs at
     2^24 until step 3. Host stash lives in the streaming layer (BaseFieldVec reverted).
  3. **THE CEILING LIFT — row-tile composition + quotient (kernel work, PENDING).** Change
     `evaluate_gate_air.cu` (+ `accumulate_numerators`) to accept a ROW-RANGE and index into a per-block
     tile buffer; for each row-block, H2D just that block's slice of all 188 columns from the host stash,
     run the kernel, free. Residency → `O(tile_rows × 188)`. Pinned host + async copy stream so the
     per-block H2D overlaps compute. THIS is what makes 2^24/2^25 fit.
  4. **Validate + measure (one box trip):** `root(streamed)==root(resident)` + fingerprint on==off @2^22
     & @2^23 (both fit); decommit-equivalence @2^23; then 2^24 & 2^25 streamed-only prove + self-verify.
     **Go/no-go perf check:** is `t_base@2^25` ≈ compute-only (H2D overlapped) or ≈ compute+PCIe
     (bottlenecked)? If bottlenecked and it can't be hidden, there is NO route-(a) escape — the call is
     to STAY at 2^23 and pursue the recursion lever instead.

### Byte-identity validation plan (the check — `vcs_lifted` is SECURITY-CRITICAL)

- **Invariant:** the lifted Blake2sM31 **root** and the full **proof_fingerprint** must be identical.
  Base oracle **@2^22 = `ab1e75b5…`** (RECURSION_PLAN.md:261).
- **Unit (on-device):** the fused forward-LDE→streamed-absorb must equal `CpuBackend::build_leaves` on
  the same eval columns at 2^16/2^20 — extends `test_build_leaves_streamed_same_size_vs_cpu`
  (blake2s.rs:480) to drive the LDE, not pre-extended columns. Assert **column absorption order ==
  `sorted_by_key(|c| c.len())`** (prover.rs:65).
- **e2e fingerprint:** rebuild gate-air-leaf with the streamed commit; assert `ab1e75b5…` @2^22.
- **Cross-check at a both-fit size:** root(streamed) == root(non-streamed) at 2^23; then trust 2^24/2^25
  via per-column-NTT-independence (rfft.cu, confirmed) + absorb-associativity + a `decommit` round-trip
  self-verify (`verify` passes on the produced proof).
- **Decommit-equivalence (MANDATORY before trusting any new-ceiling proof):** queried values from the
  streamed decommit (recompute or host-recover) == current decommit at 2^23 over a fixed query set — the
  load-bearing soundness check (a wrong `query_to_buffer_index` or non-bit-exact re-LDE would verify
  against a *different* committed value; the new-size fingerprint alone won't catch it).

---

## Key files (read-only references)
- Forward LDE (the OOM site): `crates/stwo/src/prover/backend/cuda/poly.rs` — `evaluate_polynomials`
  (475-592; extend 550, in-place NTT via `ntt_n2b_columns` 558-567, evals appended 569 & never freed),
  `extend` (416-427, allocates full eval-sized buffer per column), `barycentric_eval_at_point`
  (400-414, dense full-column OODS read); NTT kernels `crates/stwo/src/stwo_cuda/cuda/rfft.cu`
  (`ntt_n2b_columns` 672, in-place `ntt_n2b_native_batch` 637; per-column independence via grid dim
  647; shared twiddles 642-643).
- Commit lifecycle & residency: `crates/stwo/src/prover/pcs/mod.rs` — `CommitmentTreeProver::new`
  (402-441; evals held in `.polynomials`), OODS/quotient read (200-249), `decommit` (447-459; queried
  positions only), evals freed after all decommits.
- Lifted Merkle: `crates/stwo/src/prover/vcs_lifted/prover.rs` — `commit` (34-77; sort 65, leaves 68,
  layer chain 71-73), `decommit` (94-165; query map 114), + landed host-recovery tests (347-449).
- Streamed absorb (landed P-S0): `crates/stwo/src/prover/backend/cuda/blake2s.rs` — `build_leaves`
  streamed path (140-165; does NOT free → byte-identity only), `test_build_leaves_streamed_same_size_vs_cpu` (480).
- Base config: `grover-tax-v02/gate-air-leaf/src/main.rs:91` (`BASE_LOG_BLOWUP_FACTOR = 1`),
  `gate-air-leaf/src/leaf.rs:54-71` (`leaf_pcs_config` blowup 1 → pow_bits 26, n_queries 70, fold_step 4,
  `lifting_log_size = trace_log + 1`). Recursion-layer blowup 3 = main.rs:2375 (not the base shard).
- Landed composing knobs: `GATE_AIR_NTT_SUBBATCH` (poly.rs:540, default 48), `GATE_AIR_COMP_TILE`
  (evaluate_gate_air.cu:504-590), `GATE_AIR_STREAM_MERKLE` (blake2s.rs:140), store-coeffs-off/barycentric
  OODS (pcs/mod.rs:71, poly.rs:400-414).
- Prior scope this composes with: `results/STREAMING_MERKLE_SCOPE.md` (P-S0/1/2, blowup-1 accounting was
  correct); measured state `results/RECURSION_PLAN.md:239-263`.
