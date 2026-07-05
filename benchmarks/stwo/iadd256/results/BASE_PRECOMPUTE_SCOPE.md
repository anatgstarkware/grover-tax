# Base-side precompute — scope

**Goal.** Reuse the shard-invariant parts of the gate_air BASE proof across all shards instead of
rebuilding them per shard. Each `prove_base_shard` call (`main.rs:2126-2361`) currently rebuilds
witness-independent data from scratch even though it is byte-identical across equal-shaped shards
(diagnosed 2026-06-30, RECURSION_PLAN.md "Distinct-shard fold"). The CPU recursion already does the
analogous thing via `prove_circuit_with_precompute` (`stwo-circuits_2/crates/circuit_prover/src/prover.rs:105`)
which consumes a prebuilt `CommitmentTreeProver` through `commitment_scheme.commit_tree(...)`
(prover.rs:145), and via the `privacy_prove` `RecursiveProverPrecomputes` struct
(`proving-utils/crates/privacy_prove/src/lib.rs:56`). This doc scopes the same pattern for the GPU
base proof.

READ-ONLY scope. No build/run. All line numbers are from
`grover-tax-v02/gate-air-leaf/src/{main.rs,gpu_tracegen.rs,leaf.rs}` on branch
`anatg/gate-air-leaf-cuda-tests`, and from the cuda backend
`stwo-cuda-backend/crates/stwo/src/prover/pcs/mod.rs`.

---

## What `prove_base_shard` recomputes per shard, and what is shard-invariant

`prove_base_shard(shard_cases)` is called once per shard in a loop (`main.rs:2389-2394`, sequential)
or for shard 0 + a producer thread (`main.rs:2382-2385`, pipeline). For N=9024 shards of equal shape
(equal `shots_per_shard`, same program, same `k`), the only thing that genuinely differs between
calls is `shard_cases` — and within that, only each shot's `x_hex`/`y_hex` boundary. Concretely:

| local | built at | function of | shard-invariant? |
|---|---|---|---|
| `program` (`build_program_table`) | 2128 | `gates, shard_samples, k` | YES (equal shards) |
| `(rows, counts)` (`build_rows`) | 2129 | `gates, shard_cases, k, rc_*` | **NO** — `rows` depend on x; `counts` (histograms) depend on x |
| `real_rows / padded_rows / log_n_rows / max_log_size` | 2130-2133 | `rows.len()` only (= `shard_samples*k*n_gates`) | YES (equal shards — diagnosed) |
| `config` (`leaf_pcs_config`) | 2138 | `max_log_size, base_blowup` | YES |
| **`twiddles`** (`precompute_twiddles`) | 2139-2143 | `max_log_size + 1 + log_blowup` | **YES** — item #2 |
| `prover_channel` + salt + `config.mix_into` | 2144-2147 | constants | YES (re-derivable; cheap) |
| **tree0** preprocessed cols + commit | 2154-2166 | `program`(shape), `rows`→`pc_in_prog`, `n_gates`, `rc_*` | **YES** — item #1 (see note) |
| `public_claim` mix | 2168-2169 | empty | YES |
| **GPU constant uploads** `gates_flat/off_lo/off_hi` | via `gpu_flat_inputs` 2186, 2222 | `gates, rc_*` | **YES** — item #3 |
| GPU `x_states` | via `gpu_flat_inputs` 2186, 2222 | `shard_cases` (x_hex) | **NO** — per shard |
| tree1 main trace (GPU K0/K1 or CPU) | 2183-2205 | per-shot qubit states | **NO** |
| `small_main` (multiplicity + program witness) | 2175-2181 | `counts`(x) + `program`(shape) | program witness invariant; multiplicities depend on x |
| interaction (K4 / `gen_*_interaction`) + `prove_ex` | 2207-2340 | witness + drawn challenges | **NO** |

**tree0 byte-identity (item #1 soundness crux).** The 10 preprocessed columns
(`N_PREPROCESSED_COLS = 10`, `main.rs:1153`) are: `generate_qdecode_preprocessed()` (2158,
constant), `generate_prog_slot_preprocessed(&program)` (2159, program-shape only),
`generate_pc_in_prog_preprocessed(&rows, padded_rows, n_gates)` (2160), and the four
`generate_rc_preprocessed` columns (2161-2162, table-only). The one that *reads `rows`* is
`pc_in_prog`. RECURSION_PLAN.md diagnosis (A) established `pc_in_prog = pc % n_gates` is a pure
**positional/program-shape** function (the row counter, not the witness x), so for equal shards the
ten columns — and therefore the committed tree0 root — are **byte-identical** across shards. This is
the same fact opt#1 (`leaf_preprocessed_root`) relies on. **Validation must still confirm it** (see
item #1 validation below); `pc_in_prog` is the only column with a `rows` argument and is the one to
watch.

---

## Item #1 — Base tree0 (preprocessed commitment), reuse + device-resident

### Recomputed today
`main.rs:2154-2166`: a fresh `tree_builder` is built per shard, the 10 preprocessed
`CircleEvaluation`s are regenerated (2156-2164), `extend_evals(to_prover(pp))` (2165), and
`tree_builder.commit(prover_channel)` (2166) re-interpolates + lifts + Merkle-commits them. On GPU
this is a full evaluate-polynomials + lifted-Merkle commit over 10 columns at the base domain — paid
9024×.

### Hoist mechanism / API (CONFIRMED feasible, mirrors the CPU template)
The cuda `CommitmentSchemeProver` already exposes the exact injection API the CPU path uses:
- `CommitmentTreeProver::new(polynomials, log_blowup, twiddles, store_coeffs, lifting_log_size, base_column_pool)` — `pcs/mod.rs:403`, **backend-generic** over `B` (so `B = CudaBackend`). It does the evaluate + lifted-Merkle commit ONCE and stores `{ polynomials, commitment }` (pcs/mod.rs:397-400) — both **device-resident** for CudaBackend.
- `commitment_scheme.commit_tree(tree: MaybeOwned<'a, CommitmentTreeProver<B,MC>>, channel)` — `pcs/mod.rs:93-100` — appends the prebuilt tree and mixes its root into the channel, instead of `commit()` (pcs/mod.rs:77-89) which builds a new one. `MaybeOwned` accepts a **borrow**, so one tree0 can be injected by reference into every shard's scheme.

This is byte-for-byte what `prove_circuit_with_precompute` does:
`commitment_scheme.commit_tree(preprocessed_tree, channel)` (circuit_prover prover.rs:145), with the
tree built once in `prove_circuit` / `RecursiveProverPrecomputes`
(privacy_prove lib.rs:137 `CommitmentTreeProver::new`, lib.rs:60 stored in the precompute struct).

Inside `prove_base_shard`, replace lines 2148-2166 with:
1. `CommitmentSchemeProver::new(config, &pre.twiddles)` (config + twiddles from the precompute), then
2. `commitment_scheme.commit_tree(MaybeOwned::Borrowed(&pre.committed_tree0), prover_channel)`.

The channel transcript is unchanged because `commit_tree` mixes the same root the per-shard
`commit()` would have produced (given byte-identical tree0). Everything downstream (tree1, tree2,
`prove_ex`) is untouched.

**Lifetime note.** `CommitmentSchemeProver<'a, B, MC>` borrows `twiddles` and the injected tree for
`'a`. The precompute struct must therefore outlive every per-shard scheme (built before the shard
loop, dropped after) — trivially satisfied by holding it in a local above the loop. `prove_ex`
consumes the scheme **by value** (`prover/mod.rs:70 mut commitment_scheme`), so a *fresh scheme* is
still made per shard; only the contained tree0 (+ twiddles) are reused — exactly the CPU model.

### Byte-identity validation
Mirror opt#1's root-equality assert (`derive_aggregate_config` / `leaf.rs:255-280` test). Add a
`GATE_AIR_NO_BASE_PRECOMPUTE` env toggle that forces the old per-shard rebuild path. Two gates:
- **(laptop, cheap) root-equality assert:** build tree0 once; for each shard also rebuild it the old
  way and `assert_eq!` the two `commitment.root()` values (and, defensively, the 10 column ids/sizes)
  before injecting — the analog of opt#1's `leaf_preprocessed_root` equality. A mismatch means
  `pc_in_prog` (or padding) is not actually shape-only for that fixture → abort, do not inject.
- **(box) full proof fingerprint:** at the single-proof path, `GATE_AIR_PROOF_HASH=1`
  (`main.rs:3073`, `emit_proof_fingerprint` 3254) must print an identical `proof_fingerprint` with
  base-precompute ON vs OFF at 2^22. For the recursion path, `recursion_fingerprint[PRECOMPUTE_ON]`
  must equal `[PRECOMPUTE_OFF]` (`main.rs:2586-2618`) — note that fingerprint is gated by the
  separate `GATE_AIR_NO_PRECOMPUTE` flag; the base toggle should fold into the same A/B harness.

### Effort / risk / expected saving
- **Effort:** Medium. The injection API exists and is proven on CPU; the work is (a) factor the 10-col
  build out of the closure, (b) build one `CommitmentTreeProver<CudaBackend>` before the loop, (c)
  swap `tree_builder/commit` for `commit_tree`, (d) wire the toggle + assert. ~½–1 day.
- **Risk:** Low–Medium. Soundness rests entirely on tree0 byte-identity, which the root-equality
  assert enforces at runtime (cannot silently inject a wrong tree). The lifetime plumbing of a
  borrowed device-resident tree across a (possibly threaded, pipeline-path) shard loop is the only
  real engineering hazard — see item #3 residency note; if the pipeline producer thread needs it,
  share via `Arc` (privacy_prove wraps the precompute in `Arc`, lib.rs:103).
- **Expected saving:** Second-order per shard (tree0 = 10 cols vs tree1's 191 + tree2's 24; main-gen
  + `prove_ex` dominate the ~113s/shard). But it removes 9024× a full evaluate+lifted-Merkle commit
  of 10 base-domain columns, and on GPU it also removes 9024× re-upload/re-evaluate of those columns.
  Estimate low-single-digit % of base wall, larger in absolute terms at scale; pairs with #2/#3.

---

## Item #2 — Base twiddles, compute once

### Recomputed today
`main.rs:2139-2143`: `ProverBackend::precompute_twiddles(CanonicCoset::new(max_log_size + 1 +
log_blowup).circle_domain().half_coset)`. Pure function of `max_log_size` (= `log_n_rows.max(
RC_LOG_SIZE)`) and the blowup — both shard-invariant for equal shards. No `cases`/`x_states`
dependency. On GPU this is an FFT-twiddle precompute over the base domain, paid 9024×.

### Hoist mechanism / API
Compute `twiddles` once before the shard loop and store it in the precompute struct. Both
`CommitmentSchemeProver::new(config, &twiddles)` and `CommitmentTreeProver::new(..., &twiddles, ...)`
take `&TwiddleTree<B>` by reference, so a single `TwiddleTree<CudaBackend>` is shared by every
shard's scheme and by the tree0 build. This is precisely what the CPU template does
(privacy_prove lib.rs:123 `SimdBackend::precompute_twiddles(...)` once, then reused for both
preprocessed trees and threaded through `prove_circuit_with_precompute(... twiddles ...)`,
prover.rs:107).

### Byte-identity validation
Folded into item #1's A/B: twiddles are deterministic in `max_log_size`, so reuse cannot change any
bytes as long as `max_log_size` is identical across shards (which the equal-shard invariant
guarantees and the tree0 root-equality assert indirectly confirms — a different domain would change
the tree0 root). No separate assert strictly needed; the `GATE_AIR_NO_BASE_PRECOMPUTE` fingerprint
gate covers it.

### Effort / risk / expected saving
- **Effort:** Low. Move one line out of the closure into the precompute struct; thread the borrow.
  Naturally bundled with #1 (the tree0 build needs the twiddles anyway).
- **Risk:** Low. Pure deterministic function; reference-shared, immutable.
- **Expected saving:** Second-order (one twiddle precompute vs the FFT/Merkle of the full witness),
  but removes 9024× a base-domain twiddle precompute. Cheap to claim once #1's struct exists.

---

## Item #3 — GPU constant uploads, upload once + keep device-resident

### Recomputed today
Per shard, `gpu_flat_inputs(&gates, shard_cases, &rc_lo_index, &rc_hi_index)` (`main.rs:2186` for
K1, `2222` for K4) rebuilds `gates_flat` (`gates`-only, `main.rs:1926-1932`), `off_lo`/`off_hi`
(`rc_*`-only, 1938-1939), and `x_states` (`shard_cases` x_hex, 1933-1937). Of these, **only
`x_states` is per-shard**; `gates_flat`/`off_lo`/`off_hi` are shard-invariant.

Worse, inside the kernels these host slices are re-uploaded H2D **every call**:
- `gpu_gen_main_trace_device` (`gpu_tracegen.rs:1300`): `htod_copy` of `gates`, `x`, `off_lo`,
  `off_hi` at 1338-1341, then K0/K1 launch.
- `gpu_gen_interaction_device` (`gpu_tracegen.rs:1434`): **re-runs all of K0/K1 from scratch** (its
  own `htod_copy` at 1483-1486, alloc + K0/K1 launch at 1487-1545) before doing K4 — so the gates +
  offsets are uploaded a *second* time per shard, and the entire main trace is regenerated on-device
  rather than reusing the buffer K1 already produced in `gpu_gen_main_trace_device`. The rc/qdecode
  lookup *tables* themselves are not uploaded (the offset arrays are); histograms are device-alloc'd
  zeroed per call.
- Both also `compile_ptx(GATE_SIM_KERNEL)` + `load_ptx` every call (1331-1333, 1477-1479) — NVRTC
  recompilation of the kernel source per shard, also hoistable.

### Device-residency feasibility (cudarc lifetime via shared `cuda_device()`)
**Verdict: FEASIBLE and clean.** `cuda_device()` (`gpu_tracegen.rs:52-60`) returns a process-wide
`OnceLock<Arc<CudaDevice>>` on device-0's primary context. A `CudaSlice<u32>` allocated from it stays
valid for as long as the `CudaSlice` is held (cudarc frees on `Drop`). So `d_gates`/`d_off_lo`/
`d_off_hi` can be uploaded once into long-lived `CudaSlice`s held by the precompute struct and passed
into every kernel call by reference. The compiled PTX module (`load_ptx` into the named module
`"gate_sim_mod"`) is also cached on the device and can be loaded once. Only `x_states` (and the
per-shard `d_cols`/`d_rep`/histogram scratch) are allocated per call.

### Signature change to `gpu_gen_main_trace_device` / `gpu_gen_interaction_device`
Replace the four `&[u32]` host inputs (`gates_flat, x_states, off_lo, off_hi`) with:
- device-resident constants: `d_gates: &CudaSlice<u32>`, `d_off_lo: &CudaSlice<u32>`,
  `d_off_hi: &CudaSlice<u32>` (from the precompute struct), and
- per-shard `x_states: &[u32]` (or pre-uploaded `d_x: &CudaSlice<u32>`).

Then delete the `htod_copy` of gates/off_* (1338-1341, 1483-1486) and the `compile_ptx`/`load_ptx`
(1331-1333, 1477-1479), reading the cached funcs via `dev.get_func("gate_sim_mod", ...)`.

**Bigger structural win available (flag, do not require for v1):** `gpu_gen_interaction_device`
currently *re-simulates* the entire main trace (re-runs K0/K1, gpu_tracegen.rs:1497-1545) instead of
reusing the `d_cols` buffer `gpu_gen_main_trace_device` already built. The K4 kernels read `d_cols`
via `COL(c)` and never need re-simulation. Hoisting `d_cols` to be produced once per shard (in
main-trace gen) and passed into K4 would remove a full per-shard K0/K1 re-run — likely a larger GPU
saving than the constant-upload removal itself. This is a per-shard (not cross-shard) optimization
but is naturally adjacent; scope it as a stretch sub-item of #3.

### Byte-identity validation
The K1/K4 byte-identity harnesses (`k1_byte_identity` gpu_tracegen.rs:406, `k4_byte_identity` 1086,
via `GATE_AIR_GPU_TEST=k1|k4`) already assert GPU == CPU cell-by-cell. They must still PASS after the
signature change (constants from a reused device buffer must equal constants from a fresh upload —
trivially true since the bytes are identical). The end-to-end `GATE_AIR_PROOF_HASH` /
`recursion_fingerprint` ON-vs-OFF gate (item #1) is the final word.

### Effort / risk / expected saving
- **Effort:** Medium (constant-upload + PTX-cache hoist) → Medium-High if the `d_cols`-reuse stretch
  is included (touches the K4 entry shape). Constant hoist alone ~½ day.
- **Risk:** Medium. Soundness is unchanged (same bytes), so the risk is purely engineering: cudarc
  `CudaSlice` lifetime across the (threaded, in pipeline mode) shard loop, and ensuring the cached
  PTX module is loaded exactly once. Keep the constants behind an `Arc`/`OnceLock` if the pipeline
  producer thread touches them.
- **Expected saving:** The constant re-upload is tiny in bytes (gates = n_gates*4 u32, off_* = 16
  each). The real GPU wins are (a) avoiding NVRTC recompilation per shard, and (b) — if the stretch
  lands — eliminating the per-shard K0/K1 re-run inside K4. (b) is plausibly the single biggest
  per-shard GPU saving in this whole doc.

---

## Proposed precompute struct

Built ONCE before the shard loop (mirrors `RecursiveProverPrecomputes`, privacy_prove lib.rs:56;
wrap in `Arc` if the pipeline producer thread needs it, lib.rs:103):

```rust
struct BaseProverPrecompute<'a> {
    config: PcsConfig,                                         // leaf_pcs_config(max_log_size, blowup)
    twiddles: TwiddleTree<CudaBackend>,                        // item #2
    committed_tree0: CommitmentTreeProver<CudaBackend, Blake2sM31MerkleChannel>, // item #1
    // item #3 — device-resident constants (cudarc), kept alive for the loop:
    #[cfg(feature = "cuda")] d_gates: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")] d_off_lo: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")] d_off_hi: cudarc::driver::CudaSlice<u32>,
    // shard-invariant CPU-side reuse (program shape; program witness for small_main):
    program: ProgramTable,
    // 'a ties twiddles/tree0 borrows to the scheme; or make fields owned + borrow at call site.
}
```

Built once via a `build_base_precompute(&gates, shots_per_shard, k, &rc_*)` that runs the current
2128-2166 + 2139 + `gpu_flat_inputs`(constants only) logic with a representative shard's shape (shard
0's `shots_per_shard`, since all shards share shape). **Consumed** inside `prove_base_shard` by:
replacing 2139 (twiddles) with `&pre.twiddles`; replacing 2148-2166 (scheme new + tree0 build/commit)
with `CommitmentSchemeProver::new(pre.config, &pre.twiddles)` + `commit_tree(Borrowed(&pre.committed_tree0), ch)`;
replacing the two `gpu_flat_inputs(...)` constant fields with `pre.d_gates/d_off_lo/d_off_hi` and
passing only per-shard `x_states`.

---

## Phased plan (smallest validated step first)

**Phase 0 — laptop tree0 byte-identity gate (no box, no perf change).** Add the
`GATE_AIR_NO_BASE_PRECOMPUTE` toggle and the **root-equality assert**: build tree0 once and, for each
shard, also rebuild it the old way and `assert_eq!` roots (+ column ids/sizes). This is the
load-bearing soundness fact; prove it on representative fixtures (k1-n4, plus a >2^16 case so the
RC_LOG_SIZE sort order is exercised — see main.rs:1160 note) BEFORE changing any prove path. Mirrors
opt#1's `leaf_preprocessed_root_identical_across_per_shard_data` test (leaf.rs:266).

**Phase 1 — item #2 (twiddles) + item #1 (tree0) reuse, CPU+SIMD first.** Build the precompute
struct (twiddles + `committed_tree0` + `config`); swap the per-shard build for
`new(...) + commit_tree(...)`. Validate with `GATE_AIR_PROOF_HASH` ON-vs-OFF on the single-proof
path, then `recursion_fingerprint` ON-vs-OFF. (SIMD validation is laptop-runnable for small
fixtures; the cuda backend uses the identical API so it carries over.)

**Phase 2 — item #3 GPU constant residency.** Hoist `d_gates`/`d_off_lo`/`d_off_hi` + the cached PTX
module into the precompute; change `gpu_gen_main_trace_device` / `gpu_gen_interaction_device`
signatures to take device-resident constants + per-shard `x_states`. Re-run K1/K4 byte-identity
(`GATE_AIR_GPU_TEST`) then the box `GATE_AIR_PROOF_HASH`/`recursion_fingerprint` gate.

**Phase 3 (stretch) — K4 reuses K1's `d_cols`.** Eliminate the per-shard K0/K1 re-run inside
`gpu_gen_interaction_device` by threading the main-trace device buffer from main-trace gen into K4.
Per-shard (not cross-shard) but likely the biggest GPU lever here. Same byte-identity gates.

**Synergy note.** If `enabler`/`shot_id`/`pc` later move into tree0 (witness-shrink, RECURSION_PLAN.md
line 107), they become part of `committed_tree0` for free.

---

## Feasibility summary

| Item | Feasible? | API exists? | Soundness gate |
|---|---|---|---|
| #1 tree0 reuse (device-resident) | YES | `commit_tree` + `CommitmentTreeProver::new` (cuda, backend-generic, pcs/mod.rs:93/403) — same as CPU `prove_circuit_with_precompute` | tree0 root-equality assert + ON/OFF fingerprint |
| #2 twiddles once | YES | `&TwiddleTree<B>` borrowed by scheme + tree | folded into #1 fingerprint |
| #3 GPU constant residency | YES | cudarc `CudaSlice` held in precompute via `Arc<CudaDevice>` from `cuda_device()` | K1/K4 byte-identity + ON/OFF fingerprint |

**Device-residency verdict (#3):** feasible and clean — the shared process-wide `cuda_device()`
primary context means constant `CudaSlice`s and the compiled PTX module can be created once and reused
across all shards by reference; only `x_states` + per-shard scratch are per-call. The committed tree0
(item #1) likewise stays device-resident inside `CommitmentTreeProver { polynomials, commitment }`
for CudaBackend and is injected by borrow.
