# Streaming-Merkle / Column-Streamed LDE Redesign — SCOPING

**Status:** scope only (read-only investigation, no code/box). 2026-06-30.
**Goal:** lift the gate_air GPU base-shard ceiling past 2^23 on a 40GB A100 by bounding
peak device residency to O(few columns + tree) instead of O(all 191 extended columns).
This is the **only fair lever** to truly beat SP1 on 40GB hardware (Tanuj's a2-highgpu-8g
= 8× A100-SXM4-**40GB**; a2-ultragpu-80GB is a different family and is off the table).

Lever A (memory ceiling) has been declared a dead-end *without* this redesign: candidates
1/2/3 (store-coeffs off, d_fractions row-tiling `GATE_AIR_COMP_TILE`, 191-col NTT
sub-batching `GATE_AIR_NTT_SUBBATCH=48`) all validated byte-identical and lowered peaks but
**did not move the ceiling off 2^23** — because what they free is not what is resident at the
binding wall.

---

## 1. Device-memory budget at the ceiling (where the ~38GB goes at 2^24)

### Column physical size
A trace column is a `BaseFieldVec` = a device array of `u32` (M31), 4 bytes/element
(`crates/stwo/src/stwo_cuda/base_field_vec.rs`). At trace log size `L` the extended
(eval-domain) column is `2^(L+blowup)` elements. With `log_blowup_factor = 1` (the gate_air
config), the eval domain is `2× ` the trace domain.

| Quantity | Formula | at 2^23 | at 2^24 |
|---|---|---|---|
| trace-domain column | `2^L × 4B` | 32 MiB | 64 MiB |
| eval-domain column (blowup 1) | `2^(L+1) × 4B` | 64 MiB | 128 MiB |
| 191 eval-domain columns (main trace, resident for commit) | `191 × 2^(L+1) × 4B` | ~11.9 GB | ~23.9 GB |
| 191 trace-domain inputs (coexist *during* extend) | `191 × 2^L × 4B` | ~6.0 GB | ~11.9 GB |

### The binding wall (confirmed on the box: combined 1+2+3 still OOMs at 2^24)
The OOM occurs during **main-trace generation + LDE**, on a 128 MiB allocation, inside the
batched extend+NTT in `evaluate_polynomials` (`crates/stwo/src/prover/backend/cuda/poly.rs`,
the else-branch ~lines 526-584; downstream the NTT itself is `rfft.cu`). At 2^24 the
**eval-domain column set (~23.9 GB) coexists with the trace-domain input (~11.9 GB)** during
the extend ⇒ ≈ 35.8 GB just for the witness LDE, plus NTT/pool/Merkle overhead → > 40 GB.
The "~38GB at 2^24" figure is this coexistence (`191 × (2^24 + 2^25) × 4B = 35.8 GB` + a few
GB of twiddles/scratch/output-staging).

The full persistent breakdown measured earlier (lever-A investigation) at 2^23 ≈ 28GB total:

| Resident block | code path | 2^23 | scales to 2^24 |
|---|---|---|---|
| main-trace EVALS (191 cols, blowup 1) | `evaluate_polynomials` output → held in `CommitmentTreeProver.polynomials` (pcs/mod.rs:398, 422-431) | 11.9 GB | 23.9 GB |
| stored COEFFS (cand 1 removes) | `store_polynomials_coefficients` | 7.0 GB | 14.0 GB → **freed** |
| interaction trace evals | tree2 | ~1.5 GB | ~3.0 GB |
| Merkle tree layers | `MerkleProverLifted.layers` (vcs_lifted/prover.rs:20) | ~1.5 GB | ~3.0 GB |
| composition evals | tree (split halves) | ~0.5 GB | ~1.0 GB |
| twiddles | `precompute_twiddles` | ~0.12 GB | ~0.24 GB |
| **TRANSIENT** d_fractions (cand 2 tiles) | `evaluate_gate_air.cu` composition | 6.0 GB | 12.0 GB → **tiled to 0.4 GB** |
| numerators / FRI / constraint_index | composition + FRI | ~1.0 GB | ~2.0 GB |

**Merkle tree layers** (`build_next_layer` chain in `vcs_lifted/prover.rs:71-73`): the leaf
layer is `2^lifting × 32B` (Blake2sHash = `u32 s[8]` = 32B, `utils.cuh`), and the geometric
sum of all upper layers adds another `~1×` → ~2 × `2^lifting × 32B` ≈ 1.07 GB at 2^24
(lifting=2^24, blowup-1 ⇒ leaf layer = `2^24 × 32B = 512 MiB`; tree ≈ 1 GB). Small vs the LDE.

**Histograms** (qdecode[512], rc_lo[2^16], rc_hi[2^16]) are trace-gen-phase, ~MB-scale —
negligible to the ceiling.

### Why the existing knobs do NOT lift the ceiling

The three landed mitigations each attack a block that is **not** the binding wall:

- **Candidate 1 (store_polynomials_coefficients off → barycentric OODS):** frees the 14 GB of
  stored coeffs, but those are consumed at **OODS time**, *after* tree1 is built. The OOM is
  *before* OODS, during tree1's extend. Byte-identical, kept, but ceiling unchanged.
- **Candidate 2 (`GATE_AIR_COMP_TILE` row-tiling of d_fractions):** d_fractions is a
  **composition-phase** transient (`evaluate_gate_air.cu:504-590`). At 2^24 the prover never
  reaches composition — it dies in tree1 commit. Byte-identical, kept, irrelevant to ceiling.
- **Candidate 3 (`GATE_AIR_NTT_SUBBATCH=48`):** sub-batches the *extend* so the transient
  spike is one chunk (~6 GB) not all 191 at once (~24 GB). This relieves the **instantaneous
  extend spike**, but the **eval outputs accumulate** — every one of the 191 extended columns
  is appended to `values_list` and stays resident because the subsequent
  `MerkleProverLifted::commit` (pcs/mod.rs:428) reads **all** of them, and they are then
  retained in `CommitmentTreeProver.polynomials` for `decommit` (pcs/mod.rs:447-459). So
  sub-batching bounds the *peak during extend* but not the *resident set after extend* — the
  191-col eval set (~23.9 GB) plus the trace inputs being freed still leaves ~24 GB pinned,
  and the next phases can't fit. **The resident 191 eval columns are the wall.**

**One-line crux:** what stays resident *regardless of any tiling knob* is the full set of 191
extended eval columns, because (a) the lifted-Merkle commit hashes a leaf **row across all
columns at once**, and (b) those same eval columns must survive to `decommit`. Neither can be
freed mid-commit today.

---

## 2. The redesign — column-streamed LDE + incremental Merkle leaf hashing

### The row-wise-leaf vs column-streaming tension (the crux)

The lifted-Merkle leaf is **row-wise across all columns**. From the fused CUDA kernel
`blake2s_build_leaves_fused_kernel` (`crates/stwo/src/stwo_cuda/cuda/blake2s.cu:403-436`),
thread `index` (one leaf row) does:

```
blake2s_init(&state);
for (col = 0; col < number_of_columns; ++col) {
    val = data[col][index];          // M31 at (row=index, col)
    blake2s_update(&state, le_bytes(val), 4);   // absorb 4 LE bytes
}
blake2s_finalize(&state, &result[index]);       // (+ optional M31 reduce)
```

So leaf `i` = `Blake2s( col0[i] ‖ col1[i] ‖ … ‖ col190[i] )`, no domain-separation prefix
(lifted hasher). The CPU/SIMD references (`backend/cpu/merkle_lifted.rs`,
`backend/simd/blake2s_lifted.rs`) do the identical absorption in **column-sorted order**,
maintaining one running Blake2s state per leaf row (SIMD: `prev_layer_states` =
`[u32x16; 8]` per packed group). **This is the tension:** building a leaf needs every
column's value *for that row*, which appears to demand all 191 columns resident
simultaneously — the exact opposite of column-wise streaming.

### Resolution — exploit the incrementality of the absorb

The key observation: **Blake2s leaf hashing is an incremental, order-fixed absorption.** The
loop above absorbs columns one at a time into a per-row running state. We do **not** need all
191 columns resident at once — we need all 191 **running states** resident (one per leaf row)
plus **one column at a time**. This is exactly what the heterogeneous GPU path already does
piecewise (`blake2s_alloc_init_states` → `blake2s_update_columns` per group → `blake2s_lift_states`
→ `blake2s_finalize_all`, blake2s.cu:275-393): a persistent device state array updated by
successive column groups.

**Proposed streamed pipeline (per tree):**

```
states = alloc_init_states(2^lifting)          // one Blake2s state per leaf row; resident
for each column c (in the canonical sorted order build_leaves uses):
    poly_c   = interpolate/extend column c    (already on device, trace-domain coeffs)
    evals_c  = NTT-extend poly_c → eval domain (one column, ~128 MiB at 2^24)
    blake2s_update_columns(states, evals_c)    // absorb this one column into all leaf states
    free(evals_c)                              // <-- the win: column freed immediately
leaves = finalize_all(states)                  // → leaf layer
tree   = build_next_layer chain (unchanged)
```

Peak residency becomes:
**O(states array) + O(1 eval column) + O(Merkle tree)**, i.e.
`2^lifting × sizeof(Blake2sState) + 1×128 MiB + ~1 GB tree`, instead of
`191 × 128 MiB`.

`sizeof(Blake2sState)` (h[8] + buf[64] + t + buflen, see blake2s.cu:323-343) ≈ 88–96 bytes →
at 2^24 the state array is `2^24 × ~96B ≈ 1.5 GB`. So a streamed commit at 2^24 needs roughly
`1.5 GB (states) + 0.13 GB (one column) + 1 GB (tree) ≈ 2.7 GB` for the commit, versus
~23.9 GB today. **That is the entire ceiling lever.**

### The two real obstacles (where the difficulty actually is)

1. **`decommit` needs the eval columns later.** Today the eval columns survive past commit
   because `CommitmentTreeProver.polynomials` keeps them and `decommit` (pcs/mod.rs:447-459)
   re-reads `poly.evals.values` at FRI query positions. If we free each column right after
   absorbing it, decommit has nothing to read. **Resolution options (phase choice):**
   - **(a) Two-pass / recompute-on-decommit.** Free columns during commit; at decommit, the
     query set is tiny (`fri_config.n_queries`, e.g. ≤ ~100 positions). Re-extend only the
     queried rows per column — but NTT is global, so per-row re-eval = `eval_at_point`-style
     or a kept *trace-domain* (coeff) copy that is ~½ the eval size. Keeping the **coeffs**
     (trace-domain, `191 × 2^L × 4B` = 11.9 GB at 2^24) and re-extending queried columns is
     cheaper than keeping evals, but 11.9 GB is still large. Better: keep coeffs, and at
     decommit re-extend a column → read its query rows → free. Coeffs are ~½ the evals, so
     this alone lowers the resident set from ~24 GB to ~12 GB — a 2^24 enabler, maybe not
     2^25.
   - **(b) Stage evals to host pinned memory.** After absorbing column `c` into states, D2H
     the eval column to a pinned host buffer (host has 80+ GB), free device. At decommit,
     read the queried rows from host (or H2D just the queried column). Keeps device peak at
     O(few columns) and is the cleanest route to 2^25. Cost = 191 × 128 MiB D2H per tree
     (PCIe-bound; ~24 GB over ~16 GB/s ≈ 1.5 s/tree — acceptable, hidden behind compute).
   - **(c) Partial-leaf accumulation with transposed tiling.** Instead of one-column-at-a-time
     across all rows, tile by *row blocks*: hold a row-block of all 191 columns (transpose to
     row-major), hash that block's leaves to completion, free, advance. Peak =
     `191 × row_block × 4B`. This keeps full column eval residency *per block* and is closer
     to the current kernel, but requires a **transpose** (column-major device layout →
     row-major leaf access) and reintroduces "all columns for these rows" — only viable if
     the block is small. Less attractive than (a)/(b) because the LDE/NTT is inherently
     column-global (can't NTT a row-block in isolation), so you'd still extend all columns
     first. **(c) does not actually stream the LDE** and is listed only for completeness.

   **Recommended: (b) host-staging** (cleanest, reaches 2^25), with **(a) keep-coeffs +
   re-extend** as the conservative fallback if D2H staging proves too slow or fiddly.

2. **The NTT extend must be per-column (or small-group), not the 191-wide batch.** Candidate 3
   already proved per-chunk extend is byte-identical (each column's NTT is independent;
   batching is launch-grouping only). Streaming to chunk=1 (or a small group sized to the
   state-absorb granularity) is the same transform, so byte-identity carries over for the LDE
   half for free.

---

## 3. Soundness — byte-identical root and proof

The redesign must produce the **exact same lifted Blake2sM31 root and the same proof bytes**
as today. This holds because:

- **Leaf hash is associative over the column absorption sequence.** `blake2s_update` is an
  incremental absorb; `H(a‖b‖c) == update(update(update(init,a),b),c)`. Feeding columns
  one-at-a-time into a persistent per-row state yields **bit-identical** leaf bytes to the
  fused all-at-once loop, **provided the column order is identical**. The order today is the
  sorted-by-length order (`build_leaves` / `commit` sorts: `sorted_by_key(|c| c.len())`,
  prover.rs:65; and within the same-size group, the input column order). The streamed loop
  must replay **exactly that order**. (For the gate_air main tree all 191 columns are the
  same size = lifting_log_size, so it takes the fused same-size path — order = the input
  column order; the streamed loop iterates columns in that same input order.)
- **Lifting / next-layer logic is unchanged.** `blake2s_lift_states` (blake2s.cu:275-284) and
  `build_next_layer` (blake2s.cu:237-252) are reused verbatim. The redesign only changes
  *when* a column's evals exist on-device, not the hash inputs or the tree shape.
- **No verifier / vcs / FRI change.** This is a prover-side residency reorganization. Per the
  stwo CLAUDE.md, `vcs_lifted/` is SECURITY-CRITICAL; the change must **add a streamed
  `build_leaves` path** that is selected by a flag and is provably the same hash, **not**
  alter the hasher, the leaf format, or `MerkleHasherLifted`.

### Byte-identity validation harness (the gate, reused from candidates 1/2/3)

The existing `GATE_AIR_PROOF_HASH` proof-fingerprint (sha2 over the full proof,
`emit_proof_fingerprint`, gate-air-leaf L3) is the regression guard. Validation steps:

1. **Unit, on-device:** extend `crates/stwo/src/prover/backend/cuda/blake2s.rs` tests
   (`test_build_leaves_same_size_vs_cpu`, `..._heterogeneous_vs_cpu`,
   `..._with_extra_lifting_vs_cpu`) with a **streamed** variant asserting
   `streamed_build_leaves(cols, lifting) == CpuBackend::build_leaves(cols, lifting)` element
   for element — same-size, heterogeneous, and extra-lifting cases, at 2^16 and 2^20.
2. **End-to-end fingerprint:** rebuild gate-air-leaf with the streamed commit, run the
   established anchors and assert the fingerprints match the frozen oracles:
   `0448237288…` @2^22 and `a7a8261c…` @2^14 (the same oracles candidates 1/2/3 matched).
3. **New-ceiling correctness:** at 2^24 (now fits), there is no CPU/SIMD oracle that fits in
   one machine cheaply, so cross-check **root equality** between the streamed CudaBackend and
   the *non-streamed* CudaBackend at a size both fit (2^23), then trust the streamed path at
   2^24 via the per-column-independence + absorb-associativity argument plus a `decommit`
   round-trip self-verify (`verify` passes on the produced proof).
4. **Decommit equivalence** (critical for routes a/b): assert the queried values returned by
   the streamed decommit equal those from the current decommit at 2^23 over a fixed query set.

---

## 4. Unlocked ceiling and Tanuj-curve impact

### Ceiling
- **Today:** 2^23 (2^24 OOMs in tree1 LDE; persistent ~38 GB).
- **With route (a) keep-coeffs + re-extend-on-decommit:** resident set during commit drops
  from ~24 GB (evals) to ~12 GB (coeffs) + O(1 column) → **2^24 fits** (≈ 12 + 1.5 states +
  1 tree + transients < 40 GB). 2^25 likely still tight (coeffs alone = 24 GB at 2^25).
- **With route (b) host-staging:** device commit peak = O(states + 1 column + tree) ≈ 2.7 GB
  at 2^24, ≈ 5 GB at 2^25 → **2^25 fits** on 40GB (the composition/FRI phases, with
  candidates 1/2/3 already lowering their peaks, become the next wall to check — d_fractions
  tiling and barycentric OODS were built precisely for this and are now relevant again once
  the commit no longer dominates).

Target: **2^24 (route a, conservative) to 2^25 (route b, full)** per base shard on 40GB.

### Tanuj-curve impact
From the curve memory: SP1 8×A100-40GB is the baseline (k=1:18.5s … k=2000:45m53s). Our first
full GPU result (shard=2^23) was ~tied at k=1 and **~8.5× behind at scale**. The root cause of
the 8.5× was named explicitly: the 2^23 ceiling forces few shots/shard at high k, so the
thread-per-shot K0/K1 trace-gen floor (~3.3 s flat) dominates and per-shard GPU throughput
(0.67 M rows/s) falls below optimized CPU.

Lifting the ceiling helps the curve on **two compounding axes**:
1. **Amortize the K0 trace-gen floor:** a 2^25 shard holds ~4× the shots of a 2^23 shard, so
   the flat thread-per-shot floor is paid over 4× more useful work → per-shard throughput
   rises toward the LDE/commit-bound regime instead of the sim-bound floor.
2. **Fewer, larger shards → less recursion:** N_shards drops ~4× (2^23→2^25). The fold tree
   has ~4× fewer leaves and internal nodes ⇒ recursion overhead (today ~half the total,
   CPU-bound, node≈3.7 s) shrinks proportionally, and the wrapper's O(N) in-circuit work
   (RECURSION_PLAN.md leaf-output encoding) shrinks too. This is the direct path from "~8.5×
   behind" toward parity, because it attacks both the trace-gen floor and the recursion tax
   that the pipelining/GPU-recursion levers attack only partially.

Honest framing (carried from the plan): this redesign is **necessary but possibly not
sufficient alone** to beat SP1 — SP1 fits more-efficient work per 40GB GPU (narrower/deeper
chunks vs our 191-wide columns). But it is the **only lever that removes the residency wall**,
and it is a prerequisite for the larger-shard regime in which our throughput becomes
competitive. Combined with pipelining (~2×) and GPU-accelerated recursion (~half the total),
2^25 shards are the realistic route to closing the gap.

---

## 5. Effort / risk and phasing

This touches `vcs_lifted` (SECURITY-CRITICAL per CLAUDE.md) and CUDA kernels — supervised
work. Phase smallest-validated-step-first.

| Phase | Work | Effort | Risk | Validation gate |
|---|---|---|---|---|
| **P-S0** | Streamed `build_leaves` for the **same-size** case only (the gate_air main tree): persistent state array + per-column `blake2s_update_columns` loop + finalize, replacing the fused all-at-once kernel. **Commit only; keep current decommit (evals still resident).** This proves byte-identity of the *streamed hash* in isolation, with **no ceiling change yet**. | Small–Med | **Low** (absorb-associativity is provable; kernels already exist) | Unit `streamed == cpu` at 2^16/2^20; e2e fingerprint `0448237288@2^22`. |
| **P-S1** | Free each eval column right after absorb (route **a**: keep trace-domain coeffs, re-extend queried columns at decommit). This is the **first real ceiling move (→2^24)**. | Med | **Med** (decommit re-extend correctness; coeffs lifecycle) | Decommit-equivalence at 2^23 (queried values identical); fingerprint at 2^22; **2^24 now proves + self-verifies**. |
| **P-S2** | Route **b** host-staging of eval columns (D2H to pinned, H2D queried column at decommit). Replaces P-S1's coeff-keep if/when 2^25 is the target. | Med–High | **Med–High** (PCIe perf; pinned-mem lifecycle; async overlap) | Same gates as P-S1 at 2^24; **2^25 proves + self-verifies**; perf: D2H not on critical path. |
| **P-S3** | Re-validate composition + FRI peaks at the new ceiling (candidates 1/2/3 become load-bearing again; confirm d_fractions tiling + barycentric OODS keep 2^25 under 40 GB). | Small | Low | 2^25 end-to-end under 40 GB; fingerprint stable. |

### Riskiest part — flag it explicitly
**The decommit reconciliation (P-S1/P-S2).** Freeing the eval columns during commit means the
data the FRI/Merkle decommit needs is gone. Getting decommit to return **byte-identical
queried values** via re-extend (route a) or host round-trip (route b) is the soundness-load-
bearing step: a subtle mismatch (wrong query-position mapping, bit-reversed-order confusion in
`decommit`'s `(pos >> (shift+1) << 1) + (pos & 1)` indexing, prover.rs:114, or a re-extend that
isn't bit-exact to the original) would produce a proof that *verifies against a different
committed value* — a catastrophic soundness failure that the fingerprint gate at the *new*
size won't catch unless we cross-check decommit equivalence at a size both paths fit (2^23).
**Mandatory gate before trusting any new-ceiling proof: decommit-equivalence at 2^23.**

Secondary risk: the **column absorption order** must match `build_leaves`'s sort exactly
(sorted-by-length, then input order within a size group). A reorder silently changes every
leaf. Pin the order in the streamed loop to the identical `sorted_by_key` the current
`commit`/`build_leaves` uses, and assert it in the unit test.

---

## Key files (read-only references)
- LDE / extend+NTT (the OOM site, where streaming originates):
  `crates/stwo/src/prover/backend/cuda/poly.rs` — `evaluate_polynomials` (~475-592),
  `evaluate` (429-462); NTT kernel `rfft.cu`.
- Lifted-Merkle commit (the row-wise-leaf crux):
  `crates/stwo/src/prover/vcs_lifted/ops.rs` (`MerkleOpsLifted::build_leaves`, `PackLeavesOps`),
  `crates/stwo/src/prover/vcs_lifted/prover.rs` (`commit`, `decommit`, layers),
  `crates/stwo/src/prover/backend/cuda/blake2s.rs` (`build_leaves` fused + heterogeneous paths),
  `crates/stwo/src/stwo_cuda/cuda/blake2s.cu` (`blake2s_build_leaves_fused_kernel`:403-436,
  `blake2s_alloc_init_states`/`lift_states`/`update_columns`/`finalize_all`:275-393).
- SIMD/CPU references (byte-exact absorb): `backend/simd/blake2s_lifted.rs`,
  `backend/cpu/merkle_lifted.rs`.
- PCS commit + decommit (residency lifecycle): `crates/stwo/src/prover/pcs/mod.rs`
  (`CommitmentTreeProver::new`:402-441, `decommit`:447-459, `commit`:77-89).
- prove order: `crates/stwo/src/prover/mod.rs` (`prove_ex`:67-).
- Landed knobs (why they don't lift it): `GATE_AIR_NTT_SUBBATCH` (poly.rs:21-32),
  `GATE_AIR_COMP_TILE` (`stwo_cuda/cuda/constraints/evaluate_gate_air.cu`:504-590),
  store-coeffs (pcs/mod.rs:71, barycentric OODS poly.rs:358-414).
