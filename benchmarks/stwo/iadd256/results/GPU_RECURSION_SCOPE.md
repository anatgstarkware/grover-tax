# GPU-Accelerating the Multiverifier Recursion — Scope

**Status:** READ-ONLY scoping. No code built/run/edited; no GCP box touched.
**Date:** 2026-06-30.
**Subject:** Move the recursion layer (leaf_prover + 2-to-1 node folds + root verification),
today on CPU `SimdBackend`, onto the device-resident `CudaBackend` (`/home/anat/workspace/stwo-cuda-backend`,
branch `anatg/cuda-backend`). Recursion is ~half the pipeline wall-clock per the Tanuj-curve work,
so it is now worth a GPU pass in its own right.

---

## 0. TL;DR

- **What recursion proves today:** N leaf proofs + (N-1) 2-to-1 multiverifier node proofs + 1 root-verification proof,
  every one a stwo-circuits `circuit_prover` proof, **hardcoded to `SimdBackend`** and the lifted
  Blake2s channel (`Blake2sM31MerkleChannel`). The multiverifier node trace is fixed at ~2^21.
- **The crux (Q3):** the recursion does NOT prove gate_air's `FrameworkComponent`. It proves an
  arbitrary stwo-circuits `CircuitEval` — the 11 fixed circuit components (Eq, QM31Ops, TripleXor,
  M31ToU32, BlakeGGate, 5× VerifyBitwiseXor, RangeCheck16). The `CudaBackend` constraint path is
  **per-AIR** (NitrooZK = ~100 hand-written Cairo kernels; gate_air = one custom `evaluate_gate_air.cu`).
  **Neither covers the multiverifier's 11 components.** So a GPU prove of the recursion needs EITHER
  (a) ~11 new circuit-component CUDA constraint kernels, OR (b) the audited CPU-delegated
  `ComponentProver<CudaBackend>` for constraint eval — which on gate_air at 2^22 was measured at
  **94.8% of prove time** and made the whole prove ~8.5× slower than SIMD. Constraint-eval is the
  entire ballgame here, exactly as it is for the base gate_air GPU prove.
- **Memory (Q4):** a 2^21 node trace is tiny; it fits comfortably on a 40 GB A100, with room to
  overlap with the streaming base pipeline.
- **Soundness gate (Q5):** any GPU recursion output MUST be byte-identical to the CPU `SimdBackend`
  proof — same per-proof `StarkProof` bytes, same `recursion_fingerprint` over the whole tree. The
  `GATE_AIR_PROOF_HASH` (sha256 of `.proof`, aux excluded) harness already exists and is the model.
- **Recommendation:** the **smallest de-risking step is the CPU `circuit_prover` precompute reuse**
  (no GPU) which is a clean few-% win and is already scoped. The GPU node-prove is a substantial
  project whose payoff is gated on a per-component CUDA constraint evaluator that does not exist —
  treat it as a research bet, NOT a quick win, and phase it node-only-first with a byte-identity gate.

---

## 1. What the recursion proves today, and on what backend (Q1)

### 1.1 The three proof shapes

All three are produced by the stwo-circuits **`circuit_prover`** (pinned rev `041ec6101b…` in
`proving-utils/crates/recursive_aggregate/Cargo.toml:11-19`, on stwo `74951f79`), wrapping the
DSL-built `circuits::context::Context` of a circuit:

1. **Leaf** — `gate-air-leaf/src/leaf.rs:186 prove_gate_air_leaf`. Builds the circuit that runs the
   `circuits_stark_verifier::verify` of the gate_air base proof in-circuit + emits 2 QM31 outputs
   (`blake(ppRoot ‖ x ‖ y)`), then proves it. Path:
   `prove_circuit_with_precompute::<Blake2sM31MerkleChannel>` (leaf.rs:197) when a precompute is
   present, else `prove_circuit_assignment` (leaf.rs:207).
2. **Node** — `recursive_aggregate/src/lib.rs:566 prove_node`. Builds a 2-to-1
   `build_multiverifier_circuit` (verifies two child proofs, emits `blake([ppR_L, outs_L, ppR_R, outs_R]`)
   then proves it via `prove_with_precompute` → `prove_circuit_with_precompute` (lib.rs:551-563) or
   `prove_circuit_assignment` (lib.rs:582). Self-verifying shape ⇒ one `SharedConfig` for every level.
3. **Root verification** — `recursive_aggregate/src/lib.rs:439 prove_root_verification`. Verifies the
   root node in-circuit + unpacks the tree (O(N) blakes), optionally `add_zk_blinding`, pads, derives
   PCS config from the actual trace size, then `prove_circuit_assignment` (lib.rs:533). This is the
   only published, only zk-blinded proof; its trace size varies per call so it stays on the
   non-precompute path.

### 1.2 The backend is `SimdBackend`, hardcoded (not generic)

`circuit_prover/src/prover.rs` is **not** generic over the prover backend. The proving entry points
pin `SimdBackend` at every step:

- `to_component_provers(...) -> Vec<&dyn ComponentProver<SimdBackend>>` (prover.rs:40-57) — the 11
  circuit components, all `ComponentProver<SimdBackend>`.
- `CommitmentSchemeProver::<SimdBackend, MC>::with_memory_pool(...)` (prover.rs:152).
- `SimdBackend::precompute_twiddles` / `interpolate_columns` / `grind` (prover.rs:90-103, 178).
- `CommitmentTreeProver::<SimdBackend, MC>` (prover.rs:106).
- `prove_ex::<SimdBackend, _>(&components, channel, commitment_scheme, true)` (prover.rs:210).

The channel `MC` is generic but every caller in the recursion uses `Blake2sM31MerkleChannel` (the
**lifted** Blake2s channel — `vcs_lifted::blake2_merkle`), the same channel the `CudaBackend` already
implements `BackendForChannel` for and was validated byte-identical on (see Q2). `recursive_aggregate`
and `gate-air-leaf` both consume `SimdBackend` directly (lib.rs:65, leaf.rs:37) and have **no backend
type-alias seam** today (unlike gate-air-leaf's base path, which has the `ProverBackend`/`TraceBackend`
seam from the GPU-port work).

### 1.3 Per-proof critical path

`prove_circuit_with_precompute` (prover.rs:125-219), in order:

1. commit preprocessed tree0 (precomputed/reused, prover.rs:161);
2. `write_trace` → **interpolate + Merkle-commit** the base (witness) trace, tree1 (prover.rs:165-175);
3. `grind` interaction PoW + draw interaction elements (prover.rs:178-180);
4. `write_interaction_trace` → LogUp interaction trace + commit, tree2 (prover.rs:183-199);
   `assert_eq!(lookup_sum, 0)` soundness check (prover.rs:193);
5. `prove_ex::<SimdBackend,_>` (prover.rs:210) = **composition (constraint) eval → composition commit →
   OODS → quotient → FRI commit → PoW grind → FRI query decommit**.

### 1.4 Which phases dominate at the ~2^21 multiverifier trace

There is **no per-phase profile of the recursion at 2^21 yet** — the recursion anchors are whole-proof
wall-clock only (leaf ~2.9s, node ~3.7s, root ~2.0s; secure config blowup3/nq23/pow27, ~size-independent
in 2^14–2^27). But the analogous gate_air GPU profile (PROVE_EX_TIMERS, 2^22, CudaBackend) is a
**strong proxy** for what a GPU node prove would look like, because both run `prove_ex` over the lifted
channel:

> prove_s 38.2s = **1_composition_eval 36.23s (94.8%)** + pow_grind 0.95 + fri_query_decommit 0.64 +
> fri_commit 0.27 + oods 0.06 + quotient 0.03 + composition_commit 0.02.

On the **CPU/SimdBackend** the composition eval is the same constraint evaluation but on SIMD lanes
(fast), so the CPU recursion's ~3.7s/node is spread across constraint-eval + the two trace commits +
FRI. The decisive point for GPU porting: **on a GPU, composition/constraint eval is the dominant phase
unless it has a native kernel**; commit/quotient/FRI/OODS are already sub-second GPU wins. This makes
Q3 (the constraint path) the entire question.

---

## 2. Running the circuit-prover on `CudaBackend` — what exists vs needed (Q2)

### 2.1 The trait surface the circuit-prover needs

From §1.2, a GPU `circuit_prover` needs, on `CudaBackend`:
`BackendForChannel<Blake2sM31MerkleChannel>`, `PolyOps` (precompute_twiddles / interpolate_columns /
evaluate), `MerkleOpsLifted` + `PackLeavesOps` (the lifted Blake2s commit), `QuotientOps`, `FriOps`,
`GrindOps`, and `ComponentProver<CudaBackend>` for each of the **11 circuit components**.

### 2.2 What the `CudaBackend` already has (P0/P4, validated byte-identical)

Per `project_gpu_port.md` (P4, 2026-06-28/29), the device-resident `CudaBackend` at rev 74951f79
is **proven byte-identical to SimdBackend** on the lifted channel, for a PCS proof and end-to-end:

- **`cuda_byte_identity` test passes** — lifting_log_size=16 PCS proof on the lifted Blake2s channel is
  byte-identical to SimdBackend and both verify. Validates precompute_twiddles, NTT (interpolate/evaluate),
  lifted Merkle commit, **native** lifted quotient (8.49× kernel), and FRI fold all at once.
- **`PolyOps`** — interpolate/evaluate on GPU (native NTT).
- **`MerkleOpsLifted` + `PackLeavesOps`** — lifted Blake2s commit on device (build_leaves takes
  lifting_log_size; PackLeavesOps host-delegated v1; both validated in the byte-identity test).
- **`QuotientOps`** — NATIVE device lifted quotient (validated byte-identical, 8.49× the SIMD-delegate).
- **`FriOps`** — looped single-step fold_line + zero-dst fold_circle (validated; FRI pack_leaves still
  host/SIMD-delegate, a later perf opt, byte-identical now).
- **`GrindOps`**, `BackendForChannel<Blake2sM31MerkleChannel>` — present.
- **`ComponentProver<CudaBackend>`** — EXISTS, but **CPU-delegated** (constraint-framework
  `cuda_component_prover.rs`): D2H the committed trace → CPU constraint eval → upload accumulator.
  This is the audited default. There is also a gate_air-specific GPU kernel (`evaluate_gate_air.cu`,
  opt-in `CUDA_GPU_CONSTRAINTS=1`) — but that kernel is for gate_air's `GateEval`, **not** the
  multiverifier's components (see Q3).

### 2.3 What is missing for the recursion specifically

The PCS/commit/FRI/quotient half is **done and validated** for the lifted channel. The gap is:

1. **The circuit-prover is hardcoded to `SimdBackend`** (§1.2). Making it prove on `CudaBackend`
   requires either a backend-generic `circuit_prover` (touching a SOUNDNESS-CRITICAL upstream crate —
   needs a `to_component_provers::<CudaBackend>` and generic `prove_ex` call), or a forked/aliased
   recursion-side copy of the proving fn. This is the **gate-air-leaf P1b seam pattern** applied to
   the recursion crates — mechanical but invasive across `circuit_prover` + `recursive_aggregate` +
   `gate-air-leaf`.
2. **`ComponentProver<CudaBackend>` for the 11 circuit components** = the real cost (Q3).

---

## 3. THE CRUX — the multiverifier `CircuitEval` constraint path (Q3)

The recursion proves an arbitrary stwo-circuits **`CircuitEval`**, i.e. the fixed set of **11 circuit
components** (`to_component_provers`, prover.rs:43-55): Eq, QM31Ops, TripleXor, M31ToU32, BlakeGGate,
VerifyBitwiseXor {4,7,8,9,12}, RangeCheck16. It does **not** prove gate_air's `FrameworkComponent`.

The `CudaBackend` constraint coverage, from the P4 investigation:

- **No generic `FrameworkEval`→GPU path exists and one is infeasible/unsafe.** obelyzk's
  `GpuDomainEvaluator` was found UNSOUND on device columns (transmutes device memory as host SIMD
  lanes — UB for `BaseFieldVec`) and wrong even on host layout. REJECTED. NitrooZK has **no** generic
  path — its `evaluate_constraints.cu` is a switch over ~100 hand-written **per-Cairo-component**
  kernels keyed by FNV-1a of the component name.
- **The only GPU constraint kernel built so far is gate_air-specific** (`evaluate_gate_air.cu`, 151
  algebraic constraints + 6 LogUp pairs). It does NOT contain the 11 circuit components.

**Consequence — two routes, both costly:**

- **(a) Native GPU constraint kernels for the 11 circuit components.** This is the speed path but it
  is **~11 new soundness-critical CUDA kernels** (one per `CircuitEval` component, including the Blake
  G-gate and the 5 bitwise-XOR LogUp components), each byte-identity-gated vs SIMD. Large, box-only
  to validate, and exactly the class of work `stwo-circuits/CLAUDE.md` marks SOUNDNESS-CRITICAL.
  Note these components are AIR-generated upstream — there is no codegen for their CUDA equivalents.
- **(b) CPU-delegated `ComponentProver<CudaBackend>`** (the audited default). Correct and byte-identical
  immediately — but the gate_air measurement shows constraint eval is **94.8% of GPU prove time** and
  makes the whole prove ~8.5× SLOWER than SIMD at 2^22. The D2H of the committed trace + CPU eval +
  accumulator merge-back dwarfs the GPU commit/FRI wins. There is **no reason to expect the multiverifier
  node to behave differently** — it runs the same `prove_ex`, and its trace (2^21, ~215+ columns across
  11 components) is comparable in scale.

**Bottom line for Q3:** porting the recursion to GPU with route (b) is a guaranteed **regression**
(slower than the CPU SimdBackend it replaces). Route (a) — native per-component kernels — is the only
path to an actual speedup, and it is a multi-kernel, multi-session, soundness-critical CUDA effort with
no existing scaffold. This is the same wall the base gate_air GPU prove hit; the recursion does not get
to reuse the gate_air kernel because its components are different.

---

## 4. Memory (Q4)

- **Node trace ~2^21**, ~215+ M31 columns (11 components' main + interaction). At ~2^21 rows ×
  ~256 columns × 4 bytes ≈ **~2 GB** of base-field column data, times the constant factor for the
  blowup-1+ eval domain (recursion uses blowup 3 → eval domain 2^24), interaction (QM31 = 4×), tree0,
  twiddles, and FRI layers. Even with the blowup-3 multiplier this is **well under 40 GB** — order
  ~10–15 GB peak, comfortable on a 40 GB A100, very comfortable on 80 GB.
- **vs the 2^25 base:** the node is 16× smaller in rows than a 2^25 base shard, so residency is a
  non-issue relative to base proving.
- **Overlap with the streaming base pipeline (the GPU producer):** the fixed-balanced-tree streaming
  fold (`recursive_aggregate_prove_streaming`, lib.rs:372) is designed to overlap a GPU base producer
  with a CPU fold consumer. If the fold moves to GPU on the **same device** as the base producer, the
  base proof (2^25, much larger) is the residency constraint, not the node. A node fold can be
  scheduled in the gaps between base proofs or on a second device. No residency blocker; the scheduling
  (not the memory) is the open design point, and only matters once GPU folds are real.

---

## 5. Soundness gate / byte-identity validation plan (Q5)

**Invariant (non-negotiable, per `stwo-circuits/CLAUDE.md` Priority Contract #1):** every GPU recursion
proof MUST be **byte-identical** to the CPU `SimdBackend` proof — same `StarkProof` bytes per node, and
the same whole-tree `recursion_fingerprint` (the cheap "streaming == sequential" gate the fixed-tree
design relies on, per `project_iadd_recursion…` / RECURSION_PLAN.md #2).

Harness (model already exists in `gate-air-leaf`):

1. **Per-proof fingerprint** — reuse `GATE_AIR_PROOF_HASH` = sha256 of `.proof` only (StarkProof; aux
   EXCLUDED — its HashMaps serialize in non-deterministic order, the bug already found+fixed). Add the
   equivalent emit at `prove_node` / `prove_gate_air_leaf` / `prove_root_verification` outputs.
2. **Node-level A/B** — prove one node on SimdBackend and on CudaBackend from identical child proofs;
   assert fingerprint equality AND both verify. (Mirrors `cuda_byte_identity.rs` / the gate_air leaf A/B.)
3. **Whole-tree** — `recursion_fingerprint` (hash over all node proofs + root in canonical shard order)
   must match SIMD oracle, validating that the fixed balanced+carry topology is reproduced bit-for-bit.
4. **Toggle parity** — a `GATE_AIR_NO_PRECOMPUTE`-style env (already in leaf.rs:155) and a CPU-fallback
   env must leave the fingerprint constant across all combinations (precompute on/off, GPU/CPU constraint).
5. **CudaBackend foundation already passes** `cuda_byte_identity` for the lifted channel — so the PCS/
   commit/FRI/quotient half is pre-validated; the new validation surface is purely the **constraint eval**
   for the 11 components (route a) or the delegation correctness (route b).

---

## 6. Phased plan (smallest de-risking step first)

| Phase | What | Backend | Effort | Risk | Payoff |
|---|---|---|---|---|---|
| **R0** | **CPU precompute reuse** for node + leaf: extend `AggregateConfig` to hold the committed tree0 + twiddles + pool, built once, reuse via `prove_circuit_with_precompute` (already half-wired: `node_precompute`/`leaf_precompute` exist, lib.rs:114-117; leaf already uses it). Verify all 2N-1 proves skip the per-proof tree0 build. | SimdBackend (no GPU) | **S** | **low** (byte-identity guard already in `CircuitPrecompute::new`, lib.rs:191) | few-% constant factor; minutes off a multi-min fold at large N. **Ship now.** |
| **R1** | **Backend seam** for the recursion: add a `ProverBackend` type alias + a `to_component_provers::<B>` generic (or a CudaBackend twin) across `circuit_prover` + `recursive_aggregate` + `gate-air-leaf`. Type-checks with `B=SimdBackend` default; `B=CudaBackend` under a `cuda` feature. NO behavior change yet. | both (compile) | **M** | low–med (touches SOUNDNESS-CRITICAL `circuit_prover`; mechanical, mirrors gate-air-leaf P1b) | enables R2; no speedup itself. |
| **R2** | **Node-only GPU prove, route (b) CPU-delegated constraints** + per-proof byte-identity gate. The smallest end-to-end GPU recursion proof. EXPECT a REGRESSION (slower than SIMD) — its ONLY purpose is to validate the plumbing byte-identically and quantify the constraint-eval cost on the multiverifier `CircuitEval`. | CudaBackend | **M** | med (cross-device D2H; validated pattern from gate_air) | de-risks the integration; measures the real gap. **Decision gate:** if route-(b) node is ≥ several× slower (highly likely, by the gate_air analogue), STOP and do NOT ship GPU recursion until R3. |
| **R3** | **Native per-component GPU constraint kernels** for the 11 `CircuitEval` components (Eq/QM31Ops/TripleXor/M31ToU32/BlakeGGate/5×XOR/RC16), each byte-identity-gated vs SIMD; opt-in like `CUDA_GPU_CONSTRAINTS`. This is the only path to an actual GPU recursion speedup. | CudaBackend | **XL** | **high** (11 soundness-critical CUDA kernels, box-only validation, no codegen) | the win — but big, research-grade. Sequence AFTER the base gate_air constraint kernel is fully optimized (shared QM31/LogUp device prelude can be reused). |
| **R4** | **Streaming overlap + scheduling**: fold on GPU in the gaps of the base producer, or a 2nd device; re-confirm `recursion_fingerprint` byte-identity under the streaming path. | CudaBackend | M | med | only relevant once R3 makes folds cheap; ties to the "#3 dynamic topology" trigger in RECURSION_PLAN.md (revisit only when fold-starved). |

---

## 7. Expected speedup and where it helps the Tanuj curve

- **Recursion is ~half the wall-clock today** (per the prompt; the Tanuj-curve memory's "~4% of total"
  figure is from the toy/early measurement and predates the secure-config + at-scale accounting — take
  the ~half framing as the operative one for this scope). Anchors: leaf ~2.9s, node ~3.7s, root ~2.0s,
  ×(2N-1) proves over N≈0.69k shards.
- **Realistic GPU recursion ceiling:** the `cuda_byte_identity` / `cuda_pcs_timing` data shows the GPU
  wins the commit/FRI/quotient/OODS half outright (sub-second, already faster than SIMD). IF a native
  constraint kernel (R3) brings composition eval to parity-or-better with SIMD lanes, a node prove could
  plausibly land **2–4× faster** than the CPU `SimdBackend` node (same regime as the gate_air PCS-only
  8.49× before constraint eval dominated; the realistic full-prove number once constraints are native is
  lower, ~2–4×). Cutting recursion 2–4× cuts ~half the wall by ~25–37%.
- **Without R3 (route b only): NEGATIVE.** The gate_air analogue is ~8.5× slower prove / ~4.2× slower
  total on GPU because CPU-delegated constraint eval is 94.8% of GPU prove time. A route-(b) GPU
  recursion would be slower than the CPU recursion it replaces.
- **Where it helps the curve:** recursion is a fixed per-shard tax × N_shards. At high k (right end of
  the Tanuj curve, N_shards large) the recursion tax is largest in absolute terms, so a 2–4× recursion
  speedup helps most at high k — the same end of the curve where we're ~16–17× behind SP1. BUT the base
  gate_air prove is the larger lever; GPU recursion is a complementary ~quarter-of-the-wall win, not the
  decisive one. It is worth doing **after** the base GPU constraint kernel is optimized (so R3 reuses the
  device QM31/LogUp prelude), not before.

---

## 8. Open risks

1. **R3 is the whole payoff and it's the hardest part.** 11 soundness-critical CUDA constraint kernels
   with no codegen and box-only validation. Underestimating this turns "GPU recursion" into a route-(b)
   regression. The base gate_air kernel (`evaluate_gate_air.cu`) is one kernel and already consumed a
   multi-session effort; 11 is qualitatively bigger.
2. **Touching `circuit_prover` (SOUNDNESS-CRITICAL).** The R1 backend seam edits an upstream crate the
   CLAUDE.md marks supervised/approval-required. Prefer a recursion-side generic wrapper over editing
   the pinned crate, or get explicit approval + a byte-identity gate before merging.
3. **Constraint-eval D2H is structural, not incidental.** The CPU-delegated `ComponentProver<CudaBackend>`
   does a full-trace D2H + accumulator upload per prove. At 2^21 × ~256 cols this is real bandwidth on
   every node. Route (b) cannot be tuned around this — it is the kernel or nothing.
4. **No 2^21 recursion GPU profile exists yet.** The 94.8% composition-eval figure is from gate_air at
   2^22; the multiverifier's component mix (heavy Blake/XOR LogUp) could shift the breakdown. R2's
   measurement is needed before committing to R3 — but R2 itself is a non-trivial M-effort.
5. **Same-device contention with the base GPU producer** (R4). If base and recursion share one A100, the
   2^25 base proof's residency + the fold's launches contend; scheduling is unsolved (and only matters
   post-R3).
6. **Channel/lifted-proof drift.** Everything depends on the lifted `Blake2sM31MerkleChannel` proof being
   unchanged (recursion validity rests on it). The `CudaBackend` is validated for it today, but any FRI
   pack_leaves "device-native" follow-up must re-pass `cuda_byte_identity`.

---

## 9. Key file:line references

- Backend hardcoding: `stwo-circuits …/circuit_prover/src/prover.rs:40-57` (`to_component_provers`),
  `:90-113` (twiddles/interpolate/tree0), `:152` (`CommitmentSchemeProver::<SimdBackend,…>`),
  `:210` (`prove_ex::<SimdBackend,_>`).
- Node prove: `proving-utils/crates/recursive_aggregate/src/lib.rs:566 prove_node`, `:551 prove_with_precompute`.
- Root prove: `…/recursive_aggregate/src/lib.rs:439 prove_root_verification` (`:533 prove_circuit_assignment`).
- Precompute (R0): `…/recursive_aggregate/src/lib.rs:114-117` (`node_precompute`/`leaf_precompute`),
  `:128-205 CircuitPrecompute` (`:191` byte-identity root assert), pattern in
  `proving-utils/crates/privacy_prove/src/lib.rs:116-189 prepare_recursive_prover_precomputes`.
- Leaf prove + config derivation: `gate-air-leaf/src/leaf.rs:186 prove_gate_air_leaf`, `:121 derive_aggregate_config`.
- CudaBackend status: `memory/project_gpu_port.md` P4 entries (byte-identity 2026-06-28/29, native quotient
  8.49×, GPU constraint-eval 94.8% / 8.5× slower, generic-FrameworkEval deferred).
- Byte-identity harness: `gate-air-leaf/src/main.rs` (`GATE_AIR_PROOF_HASH`, hashes `.proof` only),
  `src/accumulator_diff.rs`, `bin/compare_proof_fingerprint.sh`.
