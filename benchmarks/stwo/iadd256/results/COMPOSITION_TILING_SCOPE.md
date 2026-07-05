# Composition + Quotient ROW-TILING SCOPE — the ceiling-lift work (2^24/2^25)

**Status:** scope only (read-only; no source edits, no box). 2026-07-01.
**Target:** row-tile the gate_air composition (`evaluate_gate_air.cu`) + quotient (`accumulate_numerators`)
so their residency is `O(tile_rows × 188)` instead of the full ~47 GB @2^24 eval set — the actual work
that lifts the base-shard ceiling from 2^23 to **2^24** (stretch **2^25**) on a 40 GB A100.
**Reads first:** `LDE_STREAMING_SCOPE.md` §3/§4 (established: freeing eval cols at commit is necessary
but NOT sufficient; composition + quotient re-materialize all 188 columns → still OOM; route (a)/re-LDE
ruled out because a global NTT can't emit a row-block; only route (c)/host-stash feeds a row-tiled kernel).

> **The single most important finding, which reshapes the halo analysis (§1):** the gate_air composition
> kernel reads **every one of its 188 main-trace columns and its 4 preprocessed columns at offset 0** —
> i.e. `trace_evaluations[col][row]` at the thread's own global row, pointwise, via `next_trace_mask()` /
> `get_preprocessed_column()` (both hard-wire `offsets = {0}`). **There is NO next-row (offset≠0) mask
> anywhere on tree0 or tree1.** The ONLY nonzero offset in the whole gate_air eval is a single `-1` on the
> **tree2 interaction trace** (24 cols) in the post_kernel's LogUp cumsum read (`offsets2 = {0, -1}`,
> `interaction = 2`). So the halo/edge-row problem does **not** touch the 188 big columns at all — it is
> confined to the 24 tiny interaction columns, and even there it is a *bit-reversed-domain* index, not a
> physical `row-1` (§1.4). This makes the 188-column supply path a **clean pointwise row-slice** and the
> halo a bounded, cheap problem.

---

## 1. The kernel change

### 1.1 Where global-row indexing lives today (cite sites)

All trace reads funnel through `CudaEvaluator::next_interaction_mask` (`eval_at_row.cuh:330-353`; the
assert twin at `:148-170`):

```
if (off == 0) result[i] = trace_evaluations[current_col_index][this->row];          // :345
else          result[i] = trace_evaluations[current_col_index][target_row];          // :350
   target_row = offset_bit_reversed_circle_domain_index(row, dom_log, eval_log, off) // :347
```

`row` is the evaluator's global eval-domain row, set in the kernel from `row = row_offset + local`
(`evaluate_gate_air.cu:222`). The pre_kernel (`:265-281`, `gate_read_masks :96-105`, preprocessed reads
`:251-254`) issues **only** off==0 masks → every one of the 188+4 reads is `trace_evaluations[col][row]`.
The post_kernel (`:447`, `:464`) issues the tree2 masks with `offsets={0}` and `offsets2={0,-1}`.

The finalize kernel (`generic_constraint_quotients_finalize_kernel`, launched `:613`) is pointwise on
`numerators[row]` / `denom_inv[row>>trace_log]` — already trivially per-row.

**What the landed `GATE_AIR_COMP_TILE` does NOT do (confirmed):** the existing loop (`:556-604`) tiles
only the `d_fractions` INTERMEDIATE (biasing the fraction pointer by `tile_start*logup_counts`, `:565`)
and `numerators`/`constraint_index` stay full-size and are indexed by global row. **The trace column
pointers `d_trace0/1/2` still point at FULL eval-domain columns** — `trace_evaluations[col][row]`
dereferences the global buffer. So `GATE_AIR_COMP_TILE` bounds a ~12 GB fraction transient but leaves the
~47 GB input residency untouched. Lifting the input ceiling requires the column buffers themselves to
hold only a tile.

### 1.2 The change to `evaluate_gate_air.cu` (the 188 big columns — the ceiling win)

Because every tree0/tree1 read is `col[row]` with the SAME `row` for all 188 columns, a row-block
`[row_offset, row_offset+tile_rows)` needs, per column, exactly the **contiguous physical slice
`col[row_offset .. row_offset+tile_rows]`** — the eval columns are stored in the same bit-reversed
eval-domain order the kernel indexes, so a global-row range is a contiguous byte range. Change:

- **Supply per-block tile buffers, not full columns.** The host builds `d_trace1` (and `d_trace0`) so each
  entry points at a **tile-sized device buffer of length `tile_rows`** holding that column's slice for the
  current block (H2D'd from the host stash — §2). The device pointer table is rebuilt per block.
- **Index into the tile buffer with a LOCAL row.** The one-line kernel change: in `next_interaction_mask`,
  the off==0 read becomes `trace_evaluations[col][local]` where `local = row - row_offset`. Cleanest
  mechanically: keep the evaluator's `row` global (it is still needed for `offset_bit_reversed_...` and for
  `numerators[row]`) but add a `trace_row_base` (= `row_offset`) subtracted at the tree0/tree1 dereference,
  OR (matching the existing `frac_biased` trick, `:565`) pass **tile-biased column pointers**
  `d_trace1[c] - row_offset` so `[row]` lands at tile-local slot `row-row_offset`. The biased-pointer form
  is a zero-kernel-change option for tree0/tree1 (same trick already used and validated for `d_fractions`),
  but note it only works when the read is a plain `[row]` — which for tree0/tree1 it always is.
- **numerators / constraint_index_array** stay full-size, indexed by global `row` (already correct,
  `:386-387`, `:476`) — they are per-row scalars (`qm31` + `u32`), 188× smaller than the columns, so their
  full-size residency (`2^(L+1)·(16B+4B)` ≈ 0.6 GB @2^24) is affordable and needs no tiling.

### 1.3 The tree2 (interaction) columns — the ONLY halo, and it is small

The post_kernel reads the 24 interaction columns (6 LogUp cumsum cols × 4 QM31 coords) at offset `0` and
offset `-1`. Two facts bound this:
1. **24 columns, not 188.** Full-resident tree2 = `24 × 2^(L+1) × 4B` ≈ **6 GB @2^24 / 12 GB @2^25**.
   That alone still busts 40 GB at 2^25 if held whole alongside a 188-col tile, so tree2 must also be
   tiled — but its halo is trivial to satisfy (below), and it is a quarter the size.
2. **The `-1` offset is NOT a physical `row-1`.** `offset_bit_reversed_circle_domain_index`
   (`utils.cuh:120-155`) maps a bit-reversed circle-domain row to another bit-reversed row that is
   *adjacent in coset order*, i.e. its physical position can be **anywhere** in the column
   (`bit_reverse(result_index, eval_log_size)`, `:154`). So the "previous row" for tile row `r` is not
   `r-1`; it is a scattered index. **You cannot satisfy this halo with a ±1 edge row.**

### 1.4 Halo resolution for tree2 (the load-bearing correctness item)

Three viable options, in preference order:
- **(H1) Keep the 24 interaction columns fully resident; tile only tree0/tree1.** At 2^24 that is ~6 GB
  resident tree2 + a 188-col tile + numerators; fits comfortably (see §3). At 2^25 tree2 is ~12 GB and a
  small 188-col tile still fits under 40 GB (§3 shows B up to ~2^19 leaves headroom). **This sidesteps the
  scattered-index halo entirely** — the interaction reads see the whole resident column, so
  `offset_bit_reversed_...` indexes are always valid. Given tree2 is only ~13% of the eval set, this is the
  **recommended** answer: tile the 188 (the actual wall), leave the 24 whole.
- **(H2) Split composition into pre-pass (tree0/tree1, tiled) and post-pass (tree2, tiled separately).**
  The pre_kernel writes `numerators[row]` + `intermediate_fractions` for every global row (fractions must
  then be full-size or spilled — that is the ~12 GB the current `GATE_AIR_COMP_TILE` already streams, so
  they'd need host-staging too). The post_kernel then runs over tree2 with tree2 fully resident. More
  moving parts; only needed if (H1)'s ~6–12 GB resident tree2 is itself too big — it isn't at the target
  sizes.
- **(H3) True halo residency for tree2** — precompute, per tile, the set of scattered source indices the
  `-1` masks touch and gather them. Correct but complex (irregular gather) and unnecessary given tree2 fits
  whole. **Do not do this unless 2^25 tree2 (~12 GB) is shown to not fit.**

**Verdict on row-parallelism:** the constraint eval is cleanly row-parallel for the 188 main + 4
preprocessed columns (pure pointwise, no cross-row dependency within a block). The LogUp cumsum on tree2
has a bit-reversed-domain "previous" read, but it is on 24 small columns and is resolved by keeping tree2
resident (H1), so **there is no cross-row dependency among the 188-column tile** and no boundary/edge-row
handling is needed on the big columns. This is the crux that makes the scheme sound and simple.

### 1.5 The change to `accumulate_numerators` (quotient) — the easy one

`accumulate_numerators_kernel` (`quotients.cu:186-207`) reads `columns[column_index][row]` at global `row`,
**offset 0, pointwise, no next-row mask at all**. It runs over the **subdomain** (first `size>>blowup`
bit-reversed rows — a contiguous prefix, `quotient.rs:156,200`), which at blowup 1 is **half** the eval
size (~23.5 GB @2^24). It already dereferences an arbitrary `column_index` per sample-term, so it needs all
referenced columns resident *for the rows it touches* — but since it is pure `[row]` pointwise over a
prefix, row-tiling is a **pointer-bias-only change** identical to §1.2: for each subdomain row-block, H2D
that block's slice of the referenced columns, run the kernel over `[0,tile)` with biased pointers, write
`result_*[row]` at global row. No kernel-body change beyond the row/pointer bias. The Rust dispatch in
`quotient.rs:169-275` currently `rehydrate_owned`s each staged column **whole** (`:185-198`) before one
kernel launch — that is the site to replace with a per-block H2D loop.

---

## 2. The supply path (per-row-block H2D from the host stash)

The host stash already exists: `fused_commit.rs` `HOST_STASH: HashMap<device_ptr → Vec<u32>>`
(`:58-61`), populated by `dehydrate_column` (`:75-88`) at streamed commit; today the readers call
`rehydrate_owned` (`:106-119`) which H2Ds the **whole** column. The change:

- **Add a per-row-block slice H2D.** New helper alongside `rehydrate_owned`, e.g.
  `rehydrate_block(col, row_offset, tile_rows) -> BaseFieldVec` that H2Ds
  `host[row_offset .. row_offset+tile_rows]` into a tile-sized device buffer. Because the stash holds the
  exact committed bytes in eval-domain order and tree0/tree1 reads are pure `[row]`, the physical slice IS
  the logical row-block — no reindexing.
- **Pinned + async double-buffering (the overlap).** The pinned primitives are landed but unwired
  (`bindings.rs:135-150`: `cuda_alloc_pinned_host_uint32_t`, `copy_uint32_t_vec_from_host_to_device_async`,
  `..._device_to_host_async`, `cuda_free_pinned_host`; noted "not yet wired" in `fused_commit.rs:49-57`).
  Wire the stash entries onto pinned host memory at dehydrate time (so H2D can be async), then run a
  **two-buffer ping-pong**: while the kernel computes block `k` from device buffer A, the copy stream H2Ds
  block `k+1`'s 188-column slice into device buffer B; swap. This overlaps the ~47 GB (2^24) / ~94 GB
  (2^25) of H2D with the composition compute (§4). Double-buffer device cost = `2 × B × 188 × 4B`.
- **Ordering / correctness:** the per-block H2D reads the same stash keyed by the same device-pointer the
  commit stashed under, so bytes are the exact committed bytes (`is_staged` size-guard, `:98-101`). The
  column→pointer-table mapping the kernel consumes must be rebuilt each block to point at the current
  device tile buffer (mirror `quotient.rs:205-222` / `build_scoped_device_trace` but per-block).

---

## 3. Memory accounting at tile granularity

Per-column eval slice for a tile of **B rows** = `B × 4B`. Full read set = 188 (tree1) + 4 (tree0, tiny
preprocessed) ≈ **192 columns** for composition. With (H1) tree2 stays resident.

Composition residency at tile B (2^24; eval domain 2^25 = 33.6M rows):

| Item | formula | @2^24 | @2^25 |
|---|---|---|---|
| 192-col tile (double-buffered) | `2 × B × 192 × 4B` | depends B | depends B |
| tree2 interaction resident (H1) | `24 × 2^(L+1) × 4B` | ~6.4 GB | ~12.9 GB |
| numerators (full, qm31) | `2^(L+1) × 16B` | ~0.5 GB | ~1.1 GB |
| constraint_index (full, u32) | `2^(L+1) × 4B` | ~0.13 GB | ~0.27 GB |
| d_fractions tile (`GATE_AIR_COMP_TILE`) | `B × 12 × 32B` | depends B | depends B |
| twiddles + tree | ~ | ~0.4 GB | ~0.8 GB |

Solve for B under 40 GB. Take **B = 2^20 (≈1.05M rows)**:
- tile double-buffer: `2 × 2^20 × 192 × 4B` ≈ **1.6 GB**
- d_fractions tile: `2^20 × 12 × 32B` ≈ 0.4 GB
- @2^24: 1.6 + 6.4 + 0.5 + 0.13 + 0.4 + 0.4 ≈ **~9.4 GB** → blocks = `2^25 / 2^20` = **32 blocks**.
- @2^25: 1.6 + 12.9 + 1.1 + 0.27 + 0.4 + 0.8 ≈ **~17 GB** → blocks = `2^26 / 2^20` = **64 blocks**.

Both comfortably under 40 GB with B=2^20; you can push B to 2^21–2^22 (fewer, larger blocks → better PCIe
efficiency) and still clear 2^25 (at B=2^22: tile double-buffer ≈ 6.4 GB, @2^25 total ≈ 22 GB). **Verdict:
B in [2^20, 2^22] keeps BOTH 2^24 and 2^25 well under 40 GB.** The quotient (subdomain, half the rows,
tree2 not involved) is strictly easier — same B, half the blocks.

---

## 4. PCIe go/no-go

Total H2D per proof = the full eval set streamed once for composition (~47 GB @2^24 / ~94 GB @2^25) plus
the subdomain prefix for the quotient (~23.5 GB / ~47 GB) ⇒ **~70 GB @2^24 / ~141 GB @2^25** of H2D over
PCIe per base proof. At ~16–20 GB/s pinned H2D that is **~3.5–4.4 s @2^24 / ~7–9 s @2^25** of raw transfer.

**Can compute hide it?** The measured GPU composition_eval was **~0.27 s @2^22**. Constraint eval is
~linear in rows, so scaling 2^22→2^24 (×4) → **~1.1 s**, 2^22→2^25 (×8) → **~2.2 s** of compute. **Compute
is 3–4× SMALLER than the PCIe H2D it must hide.** Even perfectly overlapped, the composition+quotient
phase is **PCIe-BOUND**: wall ≈ `max(compute, H2D) ≈ H2D` ≈ **~3.5–4.4 s @2^24 / ~7–9 s @2^25**, plus the
D2H staging already paid at commit (~3 s @2^24 / ~6 s @2^25, also overlappable there).

**Honest verdict:** row-tiling + host-stage **makes 2^24/2^25 FIT** (the ceiling lifts — that part is
solid), but the base proof becomes **PCIe-transfer-bound at composition**, adding on the order of
**~4 s @2^24 / ~8 s @2^25** that is NOT hidden by compute. Whether that pays off is a **recursion-curve
question, not a local one:** a 2^24 base shard halves the shard count vs 2^23 (fewer leaves → shallower/
cheaper recursion). The ~4 s PCIe tax per base proof must be weighed against the recursion-layer saving
from fewer, bigger shards. **This is exactly the go/no-go `LDE_STREAMING_SCOPE.md §3/§4 flagged for the box
to measure** — and the arithmetic here says the tax is real and non-trivial (compute cannot hide it), so
the decision genuinely hinges on the recursion saving. If the recursion curve does not clearly favor
bigger shards, the honest recommendation is to **STAY at 2^23** (route (a) escape does not exist).
Recommend: implement the tiling (it is the only path to even *measure* 2^24/2^25 end-to-end), then let the
box's Tanuj-curve overlay decide 2^23-vs-2^24.

---

## 5. Soundness / byte-identity + effort / risk

### The load-bearing risk: tree2 halo (§1.4)
The 188 big columns are pure pointwise → tiling them is byte-trivially correct (a row-block is a contiguous
committed-byte slice; same bytes, same `col[row]` math). **The entire correctness risk sits on the 24
interaction columns** because their `-1` mask is a scattered bit-reversed index. Resolving it by keeping
tree2 **resident (H1)** removes the risk: the interaction reads see the identical full column they see
today, so the post_kernel is byte-unchanged. **Do NOT attempt a ±1 physical-edge halo — it would read the
wrong bytes** (the "previous row" is not physically adjacent). This is the one place a naive implementation
silently produces a wrong-but-plausible composition polynomial → an unsound proof.

### Byte-identity plan (mirrors LDE_STREAMING_SCOPE §4)
- **Fingerprint** `ab1e75b5…` @2^22 must reproduce with tiling ON (tiled path must equal resident path
  bit-for-bit at a size that fits both).
- **Cross-check at a both-fit size:** composition-poly / committed root **streamed+tiled == resident** at
  2^23 (both fit 40 GB) — the primary equivalence gate. Then trust 2^24/2^25 via: (a) per-column pointwise
  read is order-independent and slice-exact; (b) tree2 resident ⇒ post_kernel unchanged; (c) a `decommit`
  round-trip self-verify (`verify` passes on the produced 2^24 proof).
- **Tile-invariance unit check:** composition output for tile size B must be independent of B (run B=2^18,
  2^20, 2^22 at 2^22, assert identical `numerators`/quotient) — catches any residual global-vs-local index
  bug and any halo mistake, cheaply, on the box.
- **Quotient equivalence:** `accumulate_numerators` per-block vs whole-column result byte-identical at 2^23.

### Not the forbidden shared verifier
`evaluate_gate_air.cu` / `.cuh` (the gate-air-cuda-kernel) and `quotient.rs`'s CudaBackend impl ARE
modifiable — they were authored/changed in this effort (#14, witness-shrink + Candidate-2 tiling
scaffolding). This is downstream circuit-specific prover code, **not** `src/core` verifier logic. The one
shared file to leave alone is `evaluate_common.cuh`'s `generic_constraint_post_kernel` (89 callers) — the
gate_air path already keeps a private tiled copy (`evaluate_gate_air_post_kernel_tiled`, `:404-477`)
precisely to avoid touching it; continue that pattern.

### Effort
| Piece | effort | note |
|---|---|---|
| `evaluate_gate_air.cu` kernel: local-row/biased-pointer for tree0/tree1; keep tree2 resident (H1) | **M** | biased-pointer trick already proven for `d_fractions`; the pre_kernel is the change, post_kernel unchanged under H1 |
| `accumulate_numerators` (quotient) per-block H2D loop | **S** | pure pointwise; pointer-bias only; replace the whole-column rehydrate at `quotient.rs:185-198` |
| Supply path: `rehydrate_block` slice-H2D helper in `fused_commit.rs` | **S** | slice of existing stash Vec |
| Pinned + async double-buffer overlap (wire the landed primitives) | **M** | primitives exist (`bindings.rs:135-150`), unwired; ping-pong buffers + copy stream |
| Dispatch: per-block device-trace rebuild in `cuda_component_prover.rs` (`build_scoped_device_trace` → per-block loop) | **M** | today rehydrates whole read-set once; becomes a block loop feeding the kernel |
| Validation (fingerprint, 2^23 equivalence, tile-invariance, decommit) | **M** | one box trip |
| **Total** | **M–L** | no XL piece; the async overlap and the dispatch loop are the bulk |

**Risk: Medium**, concentrated entirely in the tree2 halo (mitigated to Low by H1 = keep tree2 resident)
and in getting the per-block pointer table correct (mitigated by the tile-invariance unit check). The
188-column tiling itself is Low risk (byte-trivial pointwise slice).
</content>
</invoke>
