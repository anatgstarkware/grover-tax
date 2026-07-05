# P5 — GPU Constraint Kernel + Byte-Identity Harness: Scope

Status: DESIGN ONLY. No source modified. This document scopes two linked deliverables
for the device-resident CUDA backend (`feature = "cuda"`) of the gate_air leaf prover.

- Deliverable 1: a gate_air-specific GPU constraint-evaluation (composition-polynomial) kernel,
  added as a SEPARATE, feature/env-gated path; the existing CPU constraint-eval delegate stays
  UNTOUCHED as the reference oracle.
- Deliverable 2: a byte-identity-at-every-step validation harness comparing CPU-only vs CudaBackend.

Recommended build order is at the end: **harness first, kernel second (phased).**

---

## 0. The problem, restated with evidence

Today `ComponentProver<CudaBackend>::evaluate_constraint_quotients_on_domain`
(`stwo-gpu-port/crates/constraint-framework/src/prover/cuda_component_prover.rs:92-152`)
copies the WHOLE committed device trace to host via `poly_to_cpu` (D2H, line 109), runs the
audited CPU `accumulate_pointwise_cpu` (line 136), then uploads the result back to the device
accumulator (line 148). This D2H + CPU-eval + H2D round trip is THE measured bottleneck
(prove ~95s vs SIMD ~4.5s at 2^22; ~21x). The module header (lines 20-46) documents that
two prior GPU paths were rejected on soundness grounds and that the env var
`CUDA_CONSTRAINT_CPU_FALLBACK` is already wired (`cpu_fallback_forced()`, line 71) as the
A/B switch for a future verified kernel.

The kernel must reproduce, **byte-identically**, the math in
`accumulate_pointwise_cpu` (`crates/constraint-framework/src/prover/component_prover.rs:264-292`):
for each eval-domain row, run `GateEval::evaluate` via an `EvalAtRow`, sum
`random_coeff_powers[i] * constraint_i` into `row_res`, then
`res[row] = accum[row] + row_res * denom_inv[row >> trace_log_size]`.

---

## DELIVERABLE 1 — gate_air-specific GPU constraint kernel

### 1.1 The constraint set to reproduce (`gate-air-leaf/src/main.rs`, `impl FrameworkEval for GateEval`, lines 760-899)

The MAIN component `GateEval` is the only nontrivial one; the four table components
(QDecode/RcLo/RcHi/Program) each emit a single `-multiplicity` relation entry + `finalize_logup`
(lines 1033-1143) and are cheap. The kernel scope is the MAIN component; the table components can
stay on the host delegate initially (they are tiny: 2^9 and 2^16 rows).

Column layout read by `evaluate`, in mask order (this order is load-bearing — it is the FFI
trace-column order):
- Preprocessed (trace0): 1 col, `gate_pc_in_prog` (`get_preprocessed_column`, line 766).
  Note global preprocessed reindex: the component reads its single pp column via
  `preprocessed_column_indices` (`component_prover.rs:85-89`); the kernel receives only that
  one column in trace0.
- Main / base (trace1): `TRACE_COLUMNS = 191` columns (`main.rs:396`), read in this exact order
  (lines 768-786): `enabler, is_nop, is_not, is_cnot, is_toffoli, shot_id, pc`,
  `in_limb[0..32]`, `out_limb[0..32]`, then three READ blocks (`read_masks`, lines 952-971) of
  `q, limb_idx, bit_pos, mask, lsel[0..32], lo, hi, bit` each (39 cols × 3 = 117), then `ab, fire, delta`.
- Interaction (trace2): 6 LogUp columns × 4 QM31 coords = 24 cols, read by `finalize_logup_in_pairs`
  (line 897) as the cumsum mask (offsets `[0]` per-batch and `[0, -1]` for the last batch — see §1.4).

All masks are offset 0 EXCEPT the last LogUp batch's previous-row cumsum (offset `-1`), so the
kernel's `next_interaction_mask` only needs the `off==0` fast path plus one `off==-1`
bit-reversed lookup via `offset_bit_reversed_circle_domain_index` (already implemented in
`eval_at_row.cuh:164` and `utils.cuh`).

#### Algebraic constraints (all added via `eval.add_constraint`, degree ≤ 3), enumerated

| # | Constraint (source line) | Degree | Columns touched |
|---|---|---|---|
| 1-4 | opcode booleanity `op*(op-1)` for is_nop/is_not/is_cnot/is_toffoli (789) | 2 | one-hot |
| 5 | one-hot sum = enabler (791) | 1 | enabler, 4 one-hot |
| — | target read block (`read_constraints`, 804 → 914-950): | | |
| 6-37 | lsel booleanity ×32 (923) | 2 | target.lsel[j] |
| 38 | Σ lsel = active(=enabler) (932) | 1 | target.lsel, enabler |
| 39 | Σ j·lsel = limb_idx (933) | 1 | target.lsel, limb_idx |
| 40 | L − hi·2·mask − bit·mask − lo = 0 (941) | 2 | lsel, in_limb, hi, mask, bit, lo |
| 41 | bit booleanity (947) | 2 | target.bit |
| 42 | (1−active)·bit = 0 (949) | 2 | enabler, target.bit |
| 43-79 | ctrl_a read block (same 37 constraints, active = is_cnot+is_toffoli) | ≤3 | ctrl_a block |
| 80-116 | ctrl_b read block (active = is_toffoli) | ≤3 | ctrl_b block |
| 117 | ab − a_bit·b_bit (813) | 2 | ab, ctrl_a.bit, ctrl_b.bit |
| 118 | fire − is_not − is_cnot·a_bit − is_toffoli·ab (815) | 2 | fire, one-hot, a_bit, ab |
| 119 | delta − fire + 2·t_bit·fire (823) | 2 | delta, fire, target.bit |
| 120-151 | out_limb[j] − in_limb[j] − lsel_t[j]·delta·mask_t ×32 (828) | 3 | out_limb, in_limb, target.lsel/mask, delta |

Total algebraic constraints ≈ **151**. Note: the per-read `active` is a sub-expression
(`is_cnot+is_toffoli` etc.), NOT a column — the kernel computes it inline exactly as Rust does.
Degrees: 1, 2, and 3 (the out_limb write and any `active·bit`-style terms). Max degree 3, hence
`max_constraint_log_degree_bound = log_n_rows + 1` (line 756). The constraint COUNT and ORDER
must match exactly because each consumes the next `random_coeff_powers[constraint_index]`
(`cpu_domain.rs:88-94`).

#### LogUp / interaction constraints (the trickiest part — §1.4)

`evaluate` emits **12 relation entries** then `finalize_logup_in_pairs` (= 6 pairs → 6 batches),
in this exact order (mirrored by the prover's `gen_main_interaction`, `main.rs:1533-1733`):

1. state_in `+enabler` / state_out `−enabler` — `state` relation, 35-wide tuple
   `[TAG_STATE, shot_id, pc(+1 on out), in/out_limb[0..32]]` (lines 837-859)
2. qdecode target `+enabler` / qdecode ctrl_a `+a_active` — `[TAG_QDECODE, q, limb_idx, bit_pos, mask]` (862-863, `add_qdecode_lookup` 973-993)
3. qdecode ctrl_b `+b_active` / rc_lo target `+enabler`
4. rc_hi target `+enabler` / rc_lo ctrl_a `+a_active`
5. rc_hi ctrl_a `+a_active` / rc_lo ctrl_b `+b_active`
6. rc_hi ctrl_b `+b_active` / program `+enabler` — `[TAG_PROGRAM, pc_in_prog, opcode_scalar, target.q, ctrl_a.q, ctrl_b.q]` (883-895)

All five logical relations share ONE drawn `(z, α)` (`LookupElements::draw`, lines 130-140); they
are separated only by the integer TAG as tuple element 0. So the kernel needs ONE
`LookupElementsBasic<35>` (z, α, α_powers[35]); each `combine(values, N)` computes
`Σ α^i·values[i] − z` (`logup.cuh:32-43`), matching the Rust `GateRel::combine`.

### 1.2 The reference math the kernel must reproduce byte-identically

- Per-row driver: `CpuDomainEvaluator` (`cpu_domain.rs:57-101`) / `SimdDomainEvaluator`
  (`simd_domain.rs`) — `next_trace_mask`/`next_interaction_mask` pull columns by a per-interaction
  running `col_index`; `add_constraint` does `row_res += random_coeff_powers[constraint_index] * c`;
  `finalize_logup*` is provided by `crate::logup_proxy!()`.
- Accumulation: `accumulate_pointwise_cpu` (`component_prover.rs:264-292`):
  `res[row] = accum[row] + row_res * denom_inv[row >> trace_log_size]`. The CudaBackend delegate
  seeds `accum` with the device accumulator's current contents (`cuda_component_prover.rs:143`)
  and splits its `random_coeff_powers` then `.reverse()`s them (lines 125-127) — the kernel MUST
  use the SAME reversed slice and the SAME seed-and-accumulate semantics (`should_accumulate=true`).
- `denom_inv` is computed host-side and bit-reversed (`component_prover.rs:101-106`); pass it to the
  kernel as a small device array indexed by `row >> trace_log_size` (NitrooZK already does this:
  `evaluate_wide_fibonacci.cu:51`).

### 1.3 The NitrooZK pattern to mirror

The device EvalAtRow mirror is `CudaEvaluator` in
`stwo-gpu-port/crates/stwo/src/stwo_cuda/cuda/eval_at_row.cuh:280-434`:
`next_trace_mask`/`next_interaction_mask` (330-353), `add_constraint` (370-375,
`row_res += constraint * random_coeff_powers[constraint_index++]`), `add_to_relation` (320-328,
writes a `Fraction{numerator=multiplicity, denominator=relation.combine(values)}` into
`intermediate_fractions[fraction_index + row*logup_counts]`), `combine_ef`/`next_extension_interaction_mask`
(390-419). There is also a `CudaAssertEvaluator` twin (84-277) that ASSERTS each constraint == 0 —
this is the device-side analogue of `assert_constraints_on_trace` and is invaluable for the harness
(use_assert_evaluator flag).

The canonical 3-kernel pipeline for a LogUp component (mirror of `memory_address_to_id`,
`cuda/constraints/evaluate_memory_address_to_id.cu`):
1. **pre_kernel** (per-row, one thread per eval-domain row): read trace0/trace1 masks, run all
   ALGEBRAIC `add_constraint`s into `row_res` (→ `numerators[row]`), emit every relation entry into
   `intermediate_fractions[...]`, and store `constraint_index_array[row]` (the algebraic count, so
   post_kernel resumes the random-coeff index).
2. **generic_constraint_post_kernel** (`evaluate_common.cuh:54-...`): for each pair-batch,
   `Fraction::sum` the 2 fractions, read the interaction-trace cumsum mask via
   `next_extension_interaction_mask(interaction=2, [0])`, and add the LogUp constraint
   `diff*denom − num` (the last batch additionally reads offset `-1` for the previous-row cumsum
   and adds `cumsum_shift`). This is EXACTLY `finalize_logup_in_pairs`. The batching array
   `batching[i] = i/2` and `last_batch` give the 6-pair structure.
3. **generic_constraint_quotients_finalize_kernel**: `quotient = numerators[row] * denom_inv[...]`,
   written to the 4 accumulator coord buffers, honoring `should_accumulate` (accumulate, not
   overwrite, for gate_air).

FFI: the binding `bindings::evaluate_constraint_quotients_on_domain`
(`stwo_cuda/bindings.rs:469-490`) already has the right SHAPE: `quotients_0..3` (the 4 device
accumulator coord pointers via `CudaSecureColumn::device_ptr`, `cuda/secure_column.rs:23`),
`trace0/1/2_evaluations` (arrays of device column `device_ptr`s, `cuda/column.rs:78` BaseFieldVec),
`random_coeff_powers`, `denominator_inverses`, `domain_log_size`, `eval_domain_log_size`,
`number_of_columns`, `logup_counts`, `eval: *mut c_void`, `cumsum_shift: CudaSecureField`,
`should_accumulate: bool`, `use_assert_evaluator: bool` → returns `bool` (false = unsupported).
The dispatch (`cuda/evaluate_constraints.cu:112-325`) `switch`es on `eval_id` (FNV-1a of a
component-name string). gate_air is not in that switch, so today it returns false. We ADD a new arm.

### 1.4 Integration design

Two ways to wire this in; **Option A is recommended** because it keeps the gate_air kernel local to
the gate-air-leaf repo and avoids editing the shared NitrooZK dispatch beyond one generic seam.

**Option A — gate_air kernel + a thin generic FFI seam (recommended).**
1. Author `evaluate_gate_air.cu` / `.cuh` next to the other component kernels
   (`stwo-gpu-port/crates/stwo/src/stwo_cuda/cuda/`), implementing the 3-kernel pipeline above using
   `CudaEvaluator` + the generic post/finalize kernels. The pre_kernel hard-codes GateEval's
   constraint sequence (151 algebraic adds + 12 relation emits) by transcribing `evaluate` /
   `read_constraints` / `add_*_lookup` line-for-line. Pass z/α/α_powers in a small `GateAirEval`
   struct (mirror `MemoryAddressToId_Eval`) via the `eval: *mut c_void` arg.
2. Register it in the dispatch switch (`evaluate_constraints.cu`) under a new tag, e.g.
   `DISPATCH_EVAL_WITH_BOOLS("gate_air_main", ...)`. This is a small, additive edit to a build file,
   not to soundness-critical Rust verifier logic.
3. In a NEW Rust file `cuda_gate_air_kernel.rs` (gate-air-leaf side OR a new module in
   constraint-framework guarded by a cfg), implement the FFI call: build the three device-column
   pointer arrays from the `Trace<CudaBackend>` (reusing `get_constraint_quotient_inputs` to get the
   eval-domain-extended device columns + `denom_inv`), pull the accumulator's 4 coord device ptrs +
   the reversed `random_coeff_powers` slice (same split as lines 125-127), pass the gate_air
   `eval_id`, set `should_accumulate=true`, and call `bindings::evaluate_constraint_quotients_on_domain`.
4. Gate it inside `ComponentProver<CudaBackend>::evaluate_constraint_quotients_on_domain`
   (`cuda_component_prover.rs:92`): when `!cpu_fallback_forced()` AND the component is the gate_air
   MAIN component, call the new GPU path; otherwise (tables, or fallback forced, or FFI returns
   false) take the EXISTING host delegate unchanged. The CPU delegate code is not modified — only a
   new branch is added in front of it. (The header at lines 42-46 already specifies this exact
   contract: "gate the new path on `!cpu_fallback_forced()` and leave this delegate reachable via
   `CUDA_CONSTRAINT_CPU_FALLBACK=1`".)

**Trace column sourcing.** `get_constraint_quotient_inputs(self, trace, mode)` already does the
device-side eval-domain extension for SubDomain/ExtendToEvalDomain (`component_prover.rs:76-114`,
`get_trace_columns` 38-71, `B::precompute_twiddles` + `get_evaluation_on_domain` run on CudaBackend).
So the device columns handed to the kernel are already the eval-domain evaluations — no D2H. Each
`Cow<CircleEvaluation<CudaBackend>>` exposes `.values.device_ptr`. The preprocessed reindex
(lines 85-89) gives the single `gate_pc_in_prog` column for trace0.

**Mask offset → bit-reversed index.** Only the last LogUp batch uses offset `-1`; the device side
already has `offset_bit_reversed_circle_domain_index` (`eval_at_row.cuh:347`, `utils.cuh`). All other
masks are offset 0 (direct `trace_evaluations[col][row]`).

**LogUp is the trickiest part — scope carefully:**
- The 5 logical relations share one `(z, α)`; tags 1..5 are tuple element 0. The kernel uses ONE
  `LookupElementsBasic<35>`; padding shorter tuples is unnecessary because `combine(values, N)` takes
  the actual N per entry (state N=35, qdecode N=5, rc N=3, program N=6).
- α_powers must be the SAME powers the prover used in `combine` (Rust `GateRel::combine`). The
  existing `gpu_tracegen.rs` (`extract_z_alpha`, line 1060; `secure_to_m31x4`, 1052) already extracts
  z and α_powers from the drawn `GateRel` for the K4 interaction kernel — reuse that extraction to
  populate the `GateAirEval` struct, guaranteeing the constraint kernel sees identical challenges.
- `finalize_logup_in_pairs` semantics: 6 pairs, `cumsum_shift = claimed_sum / n_rows`
  (the LogupAtRow shift). The generic post_kernel already encodes this (the
  `diff = cur − prev_row − prev_col + cumsum_shift` on the last batch, `evaluate_common.cuh`). Pass
  `cumsum_shift` = the same value LogupAtRow uses (`logup.rs`); the per-row claimed_sum/log_size.
  This is the single highest-soundness-risk arithmetic — it must be validated by the assert-evaluator
  AND the accumulator diff (Deliverable 2) before trust.

### 1.5 Effort / risk / phasing

Soundness-critical, box-only-validatable (needs the GPU box + nvcc build of libstwo_cuda; cannot be
fully validated on the laptop). Riskiest sub-parts, in order:
1. **LogUp interaction constraints** (the 6-pair cumsum diff + `cumsum_shift`) — most arithmetic,
   most coupling to interaction-trace masks and Fiat-Shamir-drawn challenges.
2. **Constraint ORDER/COUNT exactness** — any off-by-one in the 151 `add_constraint` sequence
   silently corrupts `row_res` (wrong random-coeff power). The `CudaAssertEvaluator` catches this
   per-constraint.
3. **out_limb degree-3 write + the lsel/limb selection** — 32-wide loops that must transcribe the
   Rust `read_constraints` selected-limb sum exactly.
4. **Eval-domain extension correctness on device** (already exercised by other components; lower risk).

Recommended phasing:
- **Phase 1 (algebraic only):** kernel emits ONLY the 151 algebraic constraints; LogUp constraints
  stay on the host delegate is NOT possible per-component (one accumulator column), so instead:
  Phase 1 runs the algebraic part on GPU and the LogUp part... cannot be split within one component
  cleanly. Therefore Phase 1 = full pre_kernel algebraic + `use_assert_evaluator=true` against a
  known-good fixture (asserts every constraint == 0 on the real trace) — validates the algebraic
  transcription and column-order WITHOUT yet trusting the accumulated numerator.
- **Phase 2 (LogUp + accumulation):** add the relation emits + post_kernel + finalize, then validate
  the full accumulator against the CPU oracle (Deliverable 2 §2.2). Only after the accumulator diff
  is zero on multiple fixtures (c=3 demo AND blake/fibonacci/cairo graphs per the
  general-solutions rule) is the path trusted and made default.
- **Phase 3:** optionally move the 4 table components onto GPU too (low value — they are tiny).

---

## DELIVERABLE 2 — byte-identity-at-every-step harness

The CPU-only leaf proof is the deterministic golden oracle: given a fixed fixture and the canonical
transcript (`channel_salt=0`, then `config.mix_into`, `main.rs` prover_channel setup ~line 2044),
the Fiat-Shamir channel is fully determined, so a CPU proof and a CudaBackend proof of the same
fixture MUST be byte-identical. The existing `gpu_tracegen.rs` already established this discipline for
trace/interaction (`k1_byte_identity` line 284, `k4_byte_identity` line 956, both using the FIXED
`LookupElements::dummy()` / `GateRel::dummy()` challenges, `main.rs:142-154`). Deliverable 2 extends
that discipline to the constraint accumulator and the full proof. All taps are READ-ONLY — the CPU
path is the oracle and is not modified.

### 2.1 Full-proof byte-identity

The prover returns `extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(...)`
(`main.rs` ~line 2443). `extended.proof` is a `StarkProof` carrying `.commitments` (Vec of per-tree
Merkle roots), `.sampled_values`, the FRI proof (layer commitments + folded evals), and the PoW
nonce. Plan:
- Add a small, env-gated emit (e.g. `GATE_AIR_EMIT_PROOF=path`): serialize `extended.proof` (it is
  `serde`-serializable in stwo; otherwise serialize a stable structural hash — Blake2s over the
  canonical field-element flattening already used by `pack_public_claim` / the in-circuit verifier).
- Run twice: `cargo run` (default SimdBackend) and `cargo run --features cuda`, same fixture/samples,
  diff the artifacts. Byte-equal proof ⇒ end-to-end backend equivalence.
- Nondeterminism to control: none expected given fixed fixture + `channel_salt=0` + fixed
  `INTERACTION_POW_BITS`. The grind nonce is deterministic for a fixed transcript. Rayon parallelism
  in trace-gen is documented bit-identical (`build_rows` header, `generate_main_trace` header). The
  ONLY divergence source is the backend under test — which is the point.

### 2.2 Per-step diffs for localization (oracle = the kept CPU path)

Run CPU and Cuda on the same small fixture; assert equality at each pipeline stage, in order:

| Step | What to compare | Where to tap |
|---|---|---|
| a. Preprocessed tree root | `proof.commitments[0]` | already extracted as `pp_root` (`main.rs:166`); compare across backends |
| b. Main + interaction trace columns | device→host (`Column::to_cpu`) vs CPU `generate_main_trace`/`gen_main_interaction` | ALREADY DONE by `k1_byte_identity`/`k4_byte_identity` (`gpu_tracegen.rs`). Reuse. |
| c. Trace tree roots | `proof.commitments[1]` (main+interaction) | tap after `tree_builder.commit` |
| d. **Composition / DomainEvaluationAccumulator** (THE localized gate for the kernel) | run CPU `accumulate_pointwise_cpu` and the new GPU kernel on the SAME committed device trace + same `random_coeff_powers` + same `denom_inv`, diff the resulting 4 accumulator coord columns (device→host vs CPU) | new test harness; reuse `get_constraint_quotient_inputs` for both; this is the FIRST place a kernel bug shows up |
| e. claimed_sum | `main_sum` / `claimed_sums` (`main.rs` ~line 2382) | compare the QM31 across backends |
| f. Composition tree root | `proof.commitments[2]` | `main.rs:230,371` |
| g. FRI layer commitments + folded values | `proof.fri_proof` | compare across backends |

Step (d) is the decisive, localized check for Deliverable 1: it isolates the constraint kernel from
all commit/FRI machinery. Implement it as: take a real committed device `Trace<CudaBackend>`, call
BOTH `accumulate_pointwise_cpu` (via the host delegate, seeded with zeros) and the GPU kernel
(seeded with zeros, `should_accumulate=false` for a clean compare, or true with a zeroed seed),
copy both accumulator columns to host, assert equal QM31-for-QM31. Additionally run the GPU kernel
with `use_assert_evaluator=true` to get per-constraint, per-row failure localization on device.

### 2.3 Concrete test form + box command

- A dedicated env-gated bin/test in `gate-air-leaf`, e.g.
  `gate-air-leaf/src/bin/byte_identity.rs` (or a `#[cfg(feature="cuda")] #[test]`), that:
  1. Builds rows + components for a small fixture (the K4 path already wires this).
  2. Runs the CPU oracle and the Cuda path, comparing steps a–g, asserting equality and printing the
     FIRST diverging step + index (mirror `k4_byte_identity`'s PASS/FAIL printing, `gpu_tracegen.rs:1041`).
  3. Uses FIXED challenges (`LookupElements::dummy()`) for the isolated constraint/accumulator check
     (step d) so both sides see identical `(z, α)`; uses the REAL drawn transcript for the full-proof
     check (step a–g, §2.1).
- Box command (build on box with native CPU + cuda, never on laptop, never auto-run gcloud here):
  `RUSTFLAGS="-C target-cpu=native" cargo run --release --features cuda --bin byte_identity -- --fixture <fix> --samples N`
  and the CPU oracle `... cargo run --release --bin byte_identity ...` (no `cuda`), then diff the
  emitted proof artifacts from §2.1.
- Keep CPU code untouched: all taps are reads of existing structs (`extended.proof.commitments`,
  `Column::to_cpu`, the host-delegate accumulator); the harness adds only new code, no edits to the
  CPU constraint path or the shared verifier.

---

## RECOMMENDED BUILD ORDER

1. **Byte-identity harness FIRST (Deliverable 2).** It closes the existing "proof-bytes diff not yet
   done" gap, gives the full-proof oracle (§2.1) and — critically — the localized accumulator diff
   (§2.2 step d) that is the ONLY trustworthy validator for the kernel. Building it first means the
   kernel has a red/green oracle from line 1.
2. **Kernel Phase 1 (algebraic + assert-evaluator).** Transcribe the 151 algebraic constraints into
   `evaluate_gate_air.cu`'s pre_kernel; validate with `use_assert_evaluator=true` (every constraint
   == 0 on a good trace) — catches column-order/count bugs without trusting the numerator yet.
3. **Kernel Phase 2 (LogUp + accumulation).** Add the 12 relation emits + generic post/finalize
   kernels; validate via the accumulator diff (step d) on the c=3 demo AND a non-demo graph
   (blake / 44k fibonacci / cairo) per the general-solutions requirement. Only then flip the default
   on (`!cpu_fallback_forced()`), leaving the host delegate reachable via `CUDA_CONSTRAINT_CPU_FALLBACK=1`.
4. **Optional Phase 3:** move the 4 table components to GPU (low value).

### Soundness-critical / box-only flags
- New CUDA constraint arithmetic (LogUp cumsum + `cumsum_shift`, constraint order) is
  soundness-critical and box-only-validatable (needs nvcc/libstwo_cuda + a GPU). Do NOT trust until
  step-d accumulator diff is zero on multiple graph families.
- The shared NitrooZK verifier / constraint-framework CPU path MUST NOT be modified; the kernel is an
  additive branch + an additive dispatch arm + a new .cu, gated behind the existing env switch.
