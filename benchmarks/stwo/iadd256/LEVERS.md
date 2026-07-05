# gate_air pipeline — optimization levers (ranked)

Single source of truth for "what could still make the gate_air + multiverifier-recursion pipeline faster,"
reconciled against `RECURSION_PLAN.md` and `MEMORY_AIRFN_PLAN.md`. Maintained in place.

_Last updated: 2026-07-05._

## ★ FINAL RESULT (2026-07-05) — ~0.60× vs SP1, all anchors MEASURED
Stack: migration (stwo 5ea05973, 8-word root) + **decoupled k-ary (k=8)** + H_P + L3. Curve (`results/plot_k_sweep.py`
→ `k_sweep_vs_sp1.png`, `extrapolate_8g.py`): **~0.60× vs SP1 across k≥50** (k=1 → 0.88×, k=10 → 0.53×), from the prior
0.915×. Now BASE/GPU-bound (base_wall ≈ rec_wall @k2000 ~1640s vs 1625s). Measured anchors: `t_base_fold`@2^25 **9.53s**
(A100, L3, fold-amortized), decoupled recursion per-shard **18.98s** (t_leaf pinned 17.3 + t_node1/8; = 0.681× vs coupled
k=2), `P_8g`=16, `TAIL`≈2.07. Correctness: base/H_P GPU byte-identity ALL PASS, decoupled-fold fingerprint/streaming/
verify PASS, H_P + k-ary soundness-REVIEWED. **All WIP UNCOMMITTED** (migration + decoupling + H_P + L3 + vendored-#1425).
Next lever (base-bound now): base — L3 full ncu occupancy tune, or the per-layer-blowup lever (needs prove_ex timers).
### Loose ends → laptop follow-ups (none block the result)
- L3 `vcs_lifted` 1/5 test fails: CUDA resident lifted-build diverges from CPU on a specific multi-col-per-size mixed
  layout — **gate_air base UNAFFECTED** (self-verifies byte-identical); fix the CUDA lifted-build.
- H_P negative-test harness doesn't exist — author it (soundness already reviewed SOUND; this is automated confirmation).
- decline-guard `warn_if_main_host_delegate` threshold (`n_constraints>50`) is DEAD for the 22-col AIR (22 constraints) —
  lower to ~≥20 to re-arm (fast-path currently confirmed via the 20× prove-gap + DIAG line, not the guard).
- L3 measured win modest (~5% on fri_commit, word-path only); the ncu occupancy/block-size sweep (sudo) is undone.
- `privacy-circuit-verify` broke from the migration (unrelated to the recursion path) — Option-A migration debt to close.
- prove_ex timing instrumentation absent on the recursion path — needed to scope the per-layer-blowup lever.

## Fixed constraints (do not re-litigate)
- **Machine is GIVEN: 1× a2-highgpu-8g = 8× A100-40 GB.** No 80 GB card, no more GPUs, no different host.
- **blake2s is the base commitment hash and stays** — it is the cheapest in-circuit hash for the recursion leaf.
- **96-bit security** — blowup is at the floor (2× domain, pow 26 + 70 queries).
- **gate_air is 22 cols, sound, GPU-validated** (chain-lookup memory; committed). Column reduction is exhausted.
- **The preprocessed trace (tree0) is amortized** — `BaseProverPrecompute` (main.rs:2309) builds it ONCE before
  the shard loop and reuses it across every shard via `commit_tree(Borrowed)`. So the ~4.5s "preprocessed" phase
  in a single-shot prove is a one-time cost, NOT per-shard.

## The binding model (why the ranking is shaped this way)
```
T(k) = max(base_wall, rec_wall) + tail
  base_wall = ceil(N/8) · t_base_fold        (8 fixed GPUs; t_base_fold EXCLUDES the amortized preprocessed)
  rec_wall  = N · (t_leaf + t_node) / P_8g   (P_8g = 16, at the measured sweep optimum)
  N = ceil(ROWS_PER_K · k / 2^25)            (shard count — FIXED; 2^26 is dead ⇒ N cannot shrink)
```
Anchors: single-shot `t_base ≈ 13.95s` **minus the amortized ~4.5s preprocessed ⇒ `t_base_fold ≈ 9.4s`**;
`t_leaf ≈ 18.24s`, `t_node ≈ 11.10s` (both fold-measured at 2^25). At k=2000 (N=1370):
- `base_wall ≈ ceil(1370/8)·9.4 ≈ 1620s`  (NOT the 2460s an un-amortized 14.30 would give)
- `rec_wall ≈ 1370·29.34/16 ≈ 2512s`
- ⇒ **strongly rec-bound (rec/base ≈ 1.55×)**; current ratio **0.915×** vs SP1 (set by `rec_wall`, unchanged).

Consequences that drive the ranking:
1. **`rec_wall` is the binding term with lots of room under it.** The base floor is `base_wall + tail ≈ 1626s ⇒
   ~0.59× vs SP1` — NOT ~0.9×. So recursion levers have a large ceiling: cutting `rec_wall` toward `base_wall`
   takes the ratio from 0.915× down toward ~0.59×.
2. **`base_wall` is the ultimate floor** (`T ≥ base_wall`). To go BELOW ~0.59× you must also cut `t_base_fold`
   (base-kernel tuning) — but that only bites after `rec_wall` is already pushed down to the floor.

> The exact `t_base_fold` is derived (single-shot minus preprocessed); a `measure.sh` fold run would confirm it
> to a few percent. It sets both the recursion-lever ceiling and the k-ary verdict below, so it's the one number
> worth nailing before committing to a large effort.

---

## TIER 1 — the material levers

### L1. GPU-accelerate the recursion circuits (leaf + node)  ⟶ rec_wall  — ceiling ~0.59×
- **Attacks:** `rec_wall` (the binding term). `t_leaf` is 62% of it; `t_node` the rest. Cuts BOTH.
- **Mechanism:** the leaf-wrap (in-circuit stwo_verifier of a base proof) and the node fold run on **CPU /
  SimdBackend** today. Port them to the GPU backend the base already uses (plan: "GPU-port the recursion").
- **Impact:** the biggest prize — drives `rec_wall` down to `base_wall` ⇒ **~0.59×**. Below that needs L3.
- **Effort/risk:** high (a full GPU port of the recursion prover). The largest engineering item.
- **Status:** live NOW — the plan gated it on "base cut enough to be recursion-bound," which is true.
- **Source:** RECURSION_PLAN "GPU-port the RECURSION (leaf/node/fold)."

### L2. k-ary (k-to-1) fold  ⟶ rec_wall  — SCOPED → GO, **~0.73× @ k=8**, the cheaper win
- **Attacks:** `rec_wall` via the `t_node` term — fold k children per node ⇒ fewer nodes.
- **Impact (corrected model, rec-bound at every arity):** `t_node(k) ≈ 2.59 + 4.25·k`; amortized per-leaf
  `(fixed+k·c)/(k−1)`. @k2000: k=2 → 0.915× · k=3 → 0.808× · k=4 → **0.773×** · **k=8 → 0.732×** · k=16 → 0.716× ·
  k=32 → 0.709× (asymptote ~0.70×). Strongly concave: **k=8 captures ~85% of the gain**; k>16 not worth the
  node-circuit/RAM growth. Can't reach L1's ~0.59× (doesn't touch `t_leaf`; `leaf_part` ≈ 1562s ≈ base floor).
- **Recommended arity: k=8** (or k=4 for a smaller node circuit at 0.773×).
- **Effort/risk:** ~3–5 days. Sites: node circuit `build_multiverifier_circuit` (verify.rs:61 — inputs→Vec, low
  risk); tree `chunks(2)→chunks(k)` (lib.rs:324) + `build_fold_topology` FoldTask→Vec<Child> (:370, medium);
  streaming scheduler per-task slots 2→k (:431, medium, concurrency); **the UNPACKER k-child hash preimage
  (:666, HIGH risk — must be byte-identical to the in-circuit node hash)**. Carry the `len()%k` (<k) remainder
  up unchanged; test across all `N mod k` residues.
- **Soundness:** SOUND — node verifies all k children under the trusted preprocessed_root and hashes
  `blake([ppR_i,outs_i] for i<k)`; the only requirement is unpacker↔circuit preimage byte-identity (a
  correctness gate caught by `recursion_fingerprint`, not a silent hole).
- **Gating:** mandatory on-box byte-identity re-validation + new golden fingerprint (existing harness,
  main.rs:~3360). Robust to error in `t_base_fold` (rec_wall exceeds base_wall by a wide margin at all k).
- **Status:** IMPLEMENTED (k=8, Fork A: internal nodes exactly-k on the shared precompute; short root, arity =
  `root_arity(N)` from PUBLIC N, real preprocessed_root via the rebuild path). stwo-circuits COMPILES + 7 tests pass
  (k=2 cairo goldens preserved). proving-utils logic-validated (exhaustive topology N=1..2000, all 8 residues:
  streaming==sequential shape, node-count/height, internal exactly-8, root 2..8; unpacker↔node preimage ordering
  cross-checked 2..8). +2 coupled files found (node_preprocessed_from_shared, multiverifier_node_preprocessed).
  ★ UPDATE 2026-07-05: the "stwo-rev bump" was actually a ROOT-FORMAT MIGRATION (stwo #1425/#597: multiverifier root
  2-QM31 `ReducedHashValue` → 8-word `HashValue`, N_RESERVED 2→8, Blake2sM31MerkleHasher→Blake2sMerkleHasher). Chose
  **Option A** (migrate the whole stack to `5ea05973`). DONE + **CPU-VALIDATED**: workspace COMPILES clean (CPU, no
  cuda); driver/unpacker/AggregateConfig/smoke-mirror + leaf migrated to the 8-word root; unpacker preimage now
  byte-identical to `build_multiverifier_circuit` (`chain!(pp_root[8], outs)`+`blake2s_u32s`). PASS: L2 topology 4/4
  (k=8 all residues), the off-circuit 8-word twin `mv_tree_root_output_kary_carry` (N=9 carry+short-root), OPEN #3
  (constraints-zero + `==B+P_pub` + in-circuit verify + proved:true). ⚠ **k=8 node pads to 2^22 → ~35 GB/prove** ⇒
  recursion-PROVE checks (verify sanity check, recursion_fingerprint ON==OFF, streaming==sequential N=9) OOM the
  laptop → must run on the **CPU VM (stwo-vm)**; also a concurrency-fit flag (K=16×35GB=560GB on the a2-8g ~680GB —
  measure whether it caps P_8g<16, which would erode k-ary's gain). BOX-only remaining: those recursion-prove checks
  (CPU VM) + [A100] vendored-stwo #1425 absorption + L3 + base/H_P GPU byte-identity/fingerprint/negative-tests.
  UNCOMMITTED.
  ★★ REV-BUMP = NEUTRAL (CPU VM, 2026-07-05, `tasks/af871dfa…`): measured at FOLD_ARITY=**2** to isolate the migration
  from k-ary. Correctness ALL PASS (fingerprint ON==OFF, streaming==sequential, verify sanity). Perf t_leaf **17.53s** /
  t_node **10.33s** = ~3% FASTER than old @2^23 (18.14/10.69) → within noise, **NO regression** from the 8-word root /
  N_RESERVED=8 / non-reduced-lifted-Merkle migration. ⇒ Option A is safe on the recursion side; A100 (Track 2) greenlit.
  Vendored-#1425 port DONE + clean (`tasks/a53a3292…`; 2 verbatim files + 1 mechanical cuda hand-port; L3 composes;
  restore the `[patch]`/cuda comment-outs in gate-air-leaf+gate-air-cuda-kernel Cargo.toml for the box).
  Infra flag: `sync.sh` REPOS list doesn't cover the local `stwo-circuits` checkout the [patch] needs — update sync.sh.
  ★★★ k=8 BENEFIT MEASURED (CPU VM, 2026-07-05, `tasks/adeedefe…`) → **NET LOSS ~1.36× WORSE — L2 GO REVERSED.** k=8 is
  NOT ~0.73×; it regresses recursion. MECHANISM: the leaf-wrap is padded to the SAME target_padding_sizes as the node
  (`prove_gate_air_leaf` leaf.rs:234 ← the k-child multiverifier target leaf.rs:185), so a k=8 node growing 2^20→2^22
  balloons EVERY leaf-wrap to 2^22 ⇒ **t_leaf 17.53→~36s (2.05×)**. Leaf work N·t_leaf dominates + is k-independent in
  count ⇒ the 2× t_leaf swamps the fewer-nodes win. rec_wall(k=8)/rec_wall(k=2) = **1.36×** at both k2000/k8000. 35 GB/node
  was NOT the cap (K=16 fits, 518 GB). The L2 scope + all prior k-ary analysis WRONGLY assumed t_leaf fixed under k.
  ⇒ **STAY at FOLD_ARITY=2** (migrated k=2 ≈ 0.915×, neutral). SALVAGE (the real k-ary lever): **decouple the leaf-wrap
  padding from the node target** — prove leaves at their natural 2^20 independent of the k-child node size; only then do
  the node-count savings materialize without the t_leaf penalty.
  ★★★ DECOUPLING SCOPED (2026-07-05, `tasks/abf18b7e…`) → **GO.** Only **2 distinct node roots** needed (NOT per-level):
  the multiverifier SELF-VERIFIES ⇒ level-1 nodes verify leaves (shape S_leaf, root R1), level-≥2 nodes verify nodes
  (shape S_node, root R2) — a fixed point, independent of N. A node's cost is polylog in the child (linear in child's
  log-size, not 2^L), so shapes bucket to 2. **Both are TRUSTED CONSTANTS selected by tree level** (public via
  FoldTask.height) — NOT authenticated inputs, NOT the generalized unpacker (user was right). Changes: MODERATE, ~1-2
  days, LOCALIZED to the 2 wrapper crates (leaf.rs derive_aggregate_config: split leaf vs node padding targets + 2 node
  precomputes; recursive_aggregate AggregateConfig: 2 roots/targets/precomputes; prove_node/build_node_context: select by
  `task.height==1`; unpacker: R1 for the leaf-level pass, R2 above). NO new circuit, NO shared-verifier change (verify.rs
  build_multiverifier_circuit is already child-config-parameterized). SOUNDNESS low-risk (2 trusted constants w/ the
  existing CircuitPrecompute::new root-assert guard; a per-level-root mismatch → REJECTED proof caught by the sanity
  check + byte-identity, not accepted-invalid). BENEFIT confirmed: t_leaf PINNED ~17.53 across k ⇒ rec_wall(k=8) ≈
  (N/P)(17.53+1.9) = **~0.70× vs coupled k=2** ⇒ overall ratio ~**0.6–0.65×** @k2000 (from the current ~0.87–0.915×) —
  the material win k-ary was supposed to deliver; the coupling was the bug. Optimal k=8 (past that marginal + RAM). GO.
  ★ DECOUPLING IMPLEMENTED (2026-07-05) → CPU-compiles + topology 5/5 (R1/R2-per-level) + leaf pinning CONFIRMED on VM
  (t_leaf ~17.6s, NOT ballooned; R1≠R2 distinct; leaf 2^21 / node 2^22). BUT VM measurement found a BUG: the PADDING was
  decoupled but not the PCS — one `pcs` built from the LEAF size (`derive_aggregate_config` leaf.rs:~179) reused for the
  node ⇒ a 2^22 node proof (Merkle height 25) can't be verified by node_shared_config (leaf-PCS, height 24) ⇒ R2 root
  fold PANICS (merkle.rs:55, 25≠24). Topology tests missed it (they don't prove). FIX IN FLIGHT (`tasks/a165fa15…`):
  add `node_pcs = leaf_pcs_config(node_pp.trace_log_size, …)` (lifting 25) for node_shared_config + node prove PCS +
  prove_root_verification; leaf/R1 already correct. Payoff (~0.70×) + full anchors (t_node1/t_node2/P_8g/TAIL) UNMEASURED
  — the R2 fold is VM-only (35GB/node), re-measure after the fix.
  ★★★★ DECOUPLED k-ary CONFIRMED (CPU VM, post-PCS-fix, 2026-07-05, `tasks/a5beb06c…`) → **WORKS, MATERIAL WIN.**
  Part A PASS: R2 root fold COMPLETES (no panic); fingerprint ON==OFF, streaming==sequential, verify OK; leaf 2^21/
  lift24, node 2^22/lift25, R1≠R2. Part B: per-shard rec = t_leaf 17.30 + t_node1/8 1.68 = **18.98s = 0.681× vs k=2**
  (27.86s) ⇒ **overall ~0.59× @k2000** (rec_wall ~1626 ≈ base_wall ~1620 → NOW BALANCED/base-bound; from the migrated-k2
  ~0.87× / orig 0.915×). Anchors: t_leaf PINNED ~17.3 (not 36); t_node1 13.4@N16 / 21.2@N64 (bandwidth-sat); t_node2
  ~9–10; **P_8g K=16 fits** (275 GB@N16, 422@N64 < a2-8g ~680); **TAIL(N)≈2.07+0.0021·N (flat ~2s, NOT 6)**. Commit-vs-
  query split NOT captured (recursion uses upstream stwo 5ea05973 = no prove_ex timers) ⇒ the per-layer-blowup lever
  needs deliberate prove_ex instrumentation (separate task). ⇒ **A100 (Track 2) GO** (base floor + L3 + H_P GPU byte-id).
- **Soundness: SOUND** (reviewed 2026-07-05, `tasks/ae3fbae7…`) — Fork A adds NO new hole. Short-root pp_root is a
  deterministic PUBLIC function of N (root_arity(N)→topology→pp_root), trust discharged by the same honest-outer-verifier
  reconstruction as baseline; padding-domination is a hard assert (violation ⇒ prover abort, never a malformed accept);
  carry/child binding is positional + one-slot. Robustness nit: precompute+assert the root pp_root as a checked constant.
- **Source:** L2 scope 2026-07-05 (`tasks/af49b552…`); supersedes the earlier k-to-1 NO-GO.

### L3. GPU base-kernel tuning (raise occupancy)  ⟶ base_wall  — SCOPED → GO (sequencing-gated)
- **Attacks:** `t_base_fold` (the floor). **Sequencing-gated: only converts to wall-clock AFTER L1/L2 make the
  pipeline base-bound** — do NOT start before then (you'd tune off the critical path).
- **THE lever = the blake2s lifted-Merkle commit** (`fri_commit` = 4.09s = ~43% of `t_base_fold`; ~⅔ of the
  achievable gain). It is genuinely low-occupancy (12%, **memory-*pipe* not DRAM-capped** — 16% DRAM — so tuning
  helps, not past the bus). Cause: 88-byte per-thread `Blake2sState` + byte-streaming churn + no `__launch_bounds__`
  ⇒ high register pressure. Fixes (order): (1) add `__launch_bounds__` + block-size sweep (one-line, cheap
  validation); (2) drop the byte-streaming state for a fixed-size **word-path `compress`** (leaf=22 words, node=16
  words known at compile time) ⇒ state 88B→32B, big register relief. Est ~1.4–1.8× on blake2s.
- **Impact (validated vs the fri_commit-dominated breakdown):** `t_base_fold` 9.4 → ~7.4s **mid (~1.27×)**
  [conservative ~8.2s/1.15×; aggressive ~6.8s/1.38×] — NOT 1.3–1.5× aggregate, since ~3s (oods/quotient/pow/
  fri_query/host-FRI) is untouched by kernel tuning. ⇒ base floor **0.59× → ~0.47× mid** (~0.51× cons / ~0.43× aggr).
- **DEFER within L3:** K1 gate_sim (only 0.61s wall despite 21% nsys-share — nsys measures GPU-active, not
  wall) and the bulk NTT (already tiled/tuned; only the tiny residual stages at 6% occupancy have headroom).
- **Effort/risk:** moderate (multi-session CUDA; the hash output must stay byte-identical — soundness-critical,
  check vcs_lifted roots). Needs a fresh on-box `ncu` (sudo) to pin register/occupancy + post-rewrite re-profile.
- **Files:** `stwo-cuda-backend/.../cuda/blake2s.cu` (lifted kernels ~210–452).
- **Status:** IMPLEMENTED on laptop (blake2s.cu/.cuh; change 1 = `__launch_bounds__` + `BLAKE2S_LIFTED_BLK`/
  `_MINBLOCKS` knobs; change 2 = word-path rewrite, per-thread state 88B→32B, hottest kernels off `Blake2sState`).
  Byte-identity numerically verified across 13 sizes incl the 16-word lazy-flush boundary + vs Python blake2s;
  NOT yet nvcc-compiled. PENDING BOX: nvcc/nvrtc compile + `vcs_lifted` Merkle-root byte-identity + ncu
  occupancy/block-size sweep + fri_commit/t_base_fold timing. UNCOMMITTED. (`tasks/ac2f1f93…`)
- **Source:** L3 scope 2026-07-05 (`tasks/ab099a50…`).

---

## TIER 2 — real but smaller

### L6. ~~Make the unpacker O(1)~~ — **NOT PURSUED (design decision 2026-07-05)**  ⟶ tail
**DECISION: keep the O(N) binding IN the unpacker/wrapper (in-circuit).** The proof must be SELF-CONTAINED — it binds
the exposed public data (leaf outputs; eventually x,y,H_P) to the verified root R *within the proof*, so the
native/on-chain verifier is **O(1)** (verify ONE proof + read the outputs; no O(N) native re-hashing). L6 *is* sound
(the review showed the O(N) check could move native), but that trades verifier simplicity for prover in-circuit
gate-count on the tail — the wrong trade for a deployable proof. So the current in-circuit reconstruction is the
INTENDED design, NOT redundancy. **L6 not pursued.** (Also settles the "must the unpacker be O(N)?" thread: yes, by design.)
Reference — the mechanics (if ever revisited):
State (verified in the COMMITTED `prove_root_verification`, recursive_aggregate/lib.rs):
- The fold ALREADY builds the commitment into the recursion: each node hashes `[ppR_L,outs_L,ppR_R,outs_R]`, so
  the root output **R is a Merkle commitment to all leaf outputs**. (The "IO is folded into recursion" — correct.)
- BUT the final unpacker does NOT exploit it. It (a) guesses each leaf output, (b) **reconstructs the whole tree
  IN-CIRCUIT** via `blake2s_m31` (lib.rs:666) to bind to R, (c) **reveals all leaf outputs** (lib.rs:679). Its own
  doc: *"The unpack is O(N) — it touches every leaf"* (lib.rs:592). So a residual O(N) in-circuit cost remains.
- **k-behavior:** N = fold-tree leaves = base-proof **shards** = `n_shards`, which **GROWS with k** (~1370 @k2000,
  ~5480 @k8000). (NOT the fixed 9024 shots — that was the old leaf=shot count before sharding made a leaf=a shard.)
- **The rejected O(1) idea (for the record):** switch reveal-all-O(N) → commit-only-O(1) and check `(x,y)` NATIVELY.
  Sound, but REJECTED by the design decision above (would push O(N) hashing onto the verifier). Not doing it.
- **Weight/UNKNOWN:** absolute cost UNMEASURED (`TAIL=6s` is assumed; ~n_shards in-circuit blakes is a real proof).
  MEASURE the unpack proof time before ranking. Grows with k ⇒ matters more at large k.
- **Also:** the committed unpacker reveals leaf-output *hashes*; the gate_air `(x,y)/H_P` exposure is NOT yet built
  (module doc: "once gate_air leaves exist"). Confirm the actual end-to-end public output before designing L6.
- **Effort/risk:** medium; touches the unpacker + output contract (soundness-adjacent). Orthogonal to L2's k-ary
  work (same function, but reveal-all→commit-only is a separate change).
- **Source:** committed recursive_aggregate/lib.rs `prove_root_verification` (:588–679); RECURSION_PLAN wrapper note.

---

## TIER 3 — speculative / minor

### L7. Multi-gate-per-row AIR (fewer rows)  — SPECULATIVE, LOW CONFIDENCE
Pack g gates per trace row to cut rows. But the shard is **memory-capped**, and g gates widen the trace ~g×
(more accesses + LogUp entries), so total cells and N are ~unchanged when memory-bound. Only helps if it lowers
*total cells*. Major AIR redesign + fresh soundness analysis. Do not pursue without a cell-count model.

### L8. Minor cleanups (low payoff)
- **Option-2** — tile the heterogeneous Merkle state-lift per block (~1 GB, ~1–2% commit); large complexity jump.
- **C3/C4** — invariant component wiring rebuilt per prove; redundant `validate_circuit()` per node (gate off).
- **C5** — build base + recursion precompute concurrently at startup (tiny, one-time).
- **Consume K1's on-device histograms** and drop the CPU sim to a lean self-check — marginal (CPU pass overlaps).

---

## DEAD (ruled out on this machine)
- **2^26 / bigger shards — WHY:** at 2^26 the *composition* phase needs, all resident simultaneously, the K4
  interaction trace (28 cols × 2^26 × 4 B = **7 GiB just to allocate**), tree2 (~14 GiB), tree0 (~6.5 GiB), plus
  the composition working set — projecting to **~53 GiB > the 40 GiB card**. It cannot be streamed: the LogUp
  `-1` cumsum term is a **bit-reversed scattered index**, so a row-tile's halo would read the wrong bytes ⇒
  tree2/interaction must stay fully resident through composition (measured: 2^26 OOMs at the 7 GiB interaction
  alloc, before composition even starts). The machine is fixed at 40 GiB ⇒ 2^26 physically cannot fit. (It
  would have cut `rec_wall` via fewer leaves on an 80 GiB card — moot here.)
- **Recursion-friendly hash swap** — blake2s is already the cheapest in-circuit hash.
- **Blowup reduction** — already at the floor (blowup=1, 96-bit).
- **Further column reduction** — exhausted at 22 (`prev_ts` + rc limbs are genuine per-address witness).

## NON-LEVERS (assessed — no wall-clock effect here)
- **Drain the ~24 GB pinned host pool at base→leaf** — reclaims host RAM, but the a2-8g has ~680 GB and the leaf
  pools are CPU-throughput/bandwidth-bound, not capacity-bound. Irrelevant on this host.
- **CPU `build_rows`/`simulate_shot` ("the two sims")** — not consumed on the GPU path except for `LookupCounts`,
  and it **overlaps GPU work, off the critical path**. Removing it doesn't cut `t_base_fold`.
- **`cudaMemPoolTrimTo` at tree1→interaction** — recovers the ~10 GB device-pool hoard but composition OOMs
  downstream anyway; only relevant to 2^26, which is dead.
- **RAM-aware single-in-flight-leaf cap** — robustness hardening, not a perf lever.
- **Fold-tree dynamic topology** (as a *load-balancing* change) — we're CPU-fold-bound with a saturated backlog,
  so the fixed balanced+carry tree already keeps pools full. (Its *unpacker generalization* is the L2 mechanism.)
- **Base CPU-phase overlap / async double-buffered D2H/H2D** (was "L4") — SCOPED NO-GO: the async machinery is
  ALREADY implemented (default-OFF flags in `fused_commit.rs`) and box-measured a **wall no-op**. The dominant
  serial term `fri_commit` (4.09s, host FRI) runs after the GPU phases with nothing to overlap it; the D2H copies
  it targets are a one-time page-lock already amortized across shards. (`tasks/adaf25fa…`)
- **Fused-1a-proper (absorb-as-produced)** (was "L5") — SCOPED NO-GO: the ~2s rehydrate was a single-shot number;
  at fold steady-state the whole tree1 commit is 0.46s and the remaining rehydrate (<0.3s) is already hidden by
  the async path. L4 subsumes it. High code-dup + byte-identity risk for ≲3% of `t_base_fold`.

---

## Recommended sequence
1. **Confirm `t_base_fold`** — comes FREE as the "before" reading in L3's box validation (which measures
   fri_commit/t_base_fold before/after); NO standalone session. Pins the base floor (~0.59×). Note: L3 *reduces* the
   absolute (~9.4→~7.4); L2 doesn't touch the base; OPEN #3 adds a hair. (Derived 9.4 = 13.95 single-shot − 4.52
   amortized preprocessed; the fold run just confirms the amortization is real end-to-end.)
2. **L2 (k-ary fold, k=8)** — SCOPED GO, the cheapest material win (**~0.73×**), ~3–5 days; unpacker byte-identity
   is the linchpin (on-box golden re-validation mandatory).
3. **L1 (GPU-accel recursion)** — the biggest prize (**~0.59×**); the full GPU port of the leaf/node prover.
4. **L3 (base-kernel tuning, blake2s commit)** — SCOPED GO but sequencing-gated: run only after L1/L2 make the
   pipeline base-bound; then it lowers the floor to **~0.47×**.
5. **L6** — the O(N) wrapper tail, only if it grows at large N (re-measure first).

(L4/L5 dropped — scoped NO-GO, see NON-LEVERS. L7/L8 speculative/minor.)

## FINAL-CURVE ANCHOR CHECKLIST (measure ALL on-box; do NOT reuse derived/proxy/stale values)
The final SP1-overlay curve (`results/extrapolate_8g.py` + `plot_k_sweep.py`, incl k=8000) must be rebuilt from measured
anchors on the FINAL stack (migration + k-ary decoupled k=8 + H_P + L3). Capture:
- **t_base_fold @2^25** — the FOLD-amortized per-shard base (preprocessed built once, reused), measured BEFORE and AFTER
  L3, on the A100. (Not the 13.95 single-shot; not the derived 9.4.)
- **t_leaf @2^25** — leaf-wrap of a real 2^25 base proof, DECOUPLED (leaf pinned ~2^20 shape). Best measured on the A100
  host (needs a 2^25 base proof); CPU VM @2^23 gives the k-scaling shape + cross-checks the ~flat 2^23→2^25 scaling.
- **t_node split** — `t_node1` (level-1 node: verify k LEAVES, R1 shape) AND `t_node2` (level-≥2 node: verify k NODES,
  R2 shape). Most nodes are level-1 (≈N/k of them), so t_node1 dominates the node term — the model must use both, not one
  averaged t_node. (Level-1 nodes verify 2^20 leaves ⇒ may be cheaper ⇒ factor possibly <0.70×.)
- **P_8g** — the ACHIEVED concurrent pool count at k=8 decoupled, RAM-checked on ~680 GB (node ~2^22/~35 GB). Confirm it
  stays at the K=16 vCPU optimum, not RAM-capped.
- **TAIL** — the root-verification/unpacker prove time AND its N-scaling (the O(N) in-circuit reconstruction — we keep it
  in-circuit by design, L6-rejected). MEASURE it; do NOT keep assuming the ~6s constant (it grows with N ⇒ matters at
  large k/N).
- shard=2^25, N=ceil(ROWS_PER_K·k/2^25), ROWS_PER_K — fixed (workload); SP1 curve from Tanuj (extrapolate >k2000).
- **Per-proof COMMIT-vs-QUERY cost split** for the leaf / level-1 node / level-≥2 node (NTT+lifted-Merkle commit time
  vs FRI-query/decommit time) — needed to evaluate the per-layer-blowup lever below.
Then update extrapolate_8g.py with ALL measured anchors, re-plot vs SP1 (k=1…8000), and mark measured-vs-extrapolated.

## CANDIDATE LEVER — per-layer blowup tuning (recursion), constant 96-bit
Blowup is a FREE COST knob, NOT a security constraint: `bits ≈ n_queries·log_blowup + pow`, so you can lower
`LOG_BLOWUP_FACTOR` and raise `n_queries` at constant 96-bit. The recursion currently runs at **log_blowup=3** (8×
lifted domain) vs the base at 1. Lowering it shrinks the (likely-dominant) commit ~4× per −2 log, at ~3× more queries.
- **Highest leverage = the LEAF's blowup** (leaf = ~90% of rec_wall, N of them). Lower leaf blowup ⇒ cheaper leaf commit
  ⇒ lower t_leaf (the dominant term).
- TRADEOFF (not security): a proof's blowup↓ ⇒ its commit cheaper BUT more queries ⇒ its PARENT's in-circuit verify of
  it gets bigger (more query-paths to decommit). Leaves are N, level-1 nodes N/k ⇒ cheapening leaves at the cost of
  (fewer) bigger level-1 nodes is a net win ⇒ per-layer optimal blowup differs (leaf / level-1 / higher).
- SCOPE after the decoupling baseline + the commit-vs-query split are measured (need the cost structure to pick the
  per-layer optimum + model the child↔parent coupling). Potentially targets the DOMINANT t_leaf ⇒ could stack with
  decoupling toward the base floor. (Complements, not replaces, L1 GPU-accel.)
