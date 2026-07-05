# P4 (Bridge B) — port NitrooZK device-resident CudaBackend → stwo-gpu-port @74951f79

GOAL: add a device-resident `CudaBackend` to our tree so `prove_ex::<CudaBackend, Blake2sM31MerkleChannel>`
emits OUR lifted proof on-device (recursion verifier untouched). Reuse NitrooZK's CUDA ops; reconcile
signatures to 74951f79 (the same delta-fixing pattern as P0, where obelyzk was rebased).

REQUIRED surface = the 24 trait impls our `GpuBackend` (obelyzk) satisfies at 74951f79.
HAVE = NitrooZK CudaBackend (v2.1.0). Below = trait-by-trait reconcile.

## Build/infra (precondition)
- NitrooZK CUDA = real `.cu` compiled by **nvcc via CMake** (build.rs → CMakeLists → `libstwo_cuda.a`,
  link `-lcudart -lstdc++`) + an FFI `stwo_cuda` module (bindings.rs). Our tree currently uses
  **cudarc/NVRTC** (no nvcc/cmake). → Adopt NitrooZK's nvcc/cmake build for the CUDA lib in
  stwo-gpu-port. Transplant the `stwo_cuda/` module (FFI + .cu + build.rs + CMakeLists). Drop the
  ~150 cairo-specific `evaluate_*.cu` (not needed; gate_air constraint-eval = CPU fallback v1). Keep
  AIR-agnostic kernels: ifft/rfft/twiddles/bit_reverse/poly_utils, blake2s (incl lifted), fold_line,
  fold_circle_into_line, fri_utils, quotients, accumulate, batch_inverse, prefix_sum, eval_at_point,
  barycentric, grind_blake2s, mle/gkr. K1/K4 kernels join this set (port from NVRTC string → .cu).
- Box has CUDA 12.x + the box build already does nvcc-capable toolchain? (verify nvcc on box.)

## Trait reconcile table

GREEN = match / trivial. YELLOW = mechanical signature drift. RED = new code / new GPU crypto.

| # | Trait | HAVE (NitrooZK) | REQUIRED (74951f79) | Delta | Rating |
|---|-------|-----------------|---------------------|-------|--------|
| 1 | ColumnOps<BaseField>/<SecureField> | device BaseFieldVec/SecureFieldVec, bit_reverse_column | same + Column trait methods | device-resident IS the win; NitrooZK has extra batch_at (fine) | GREEN |
| 2 | AccumulationOps | accumulate/generate_secure_powers(CPU)/lift_and_accumulate | same 3 | match | GREEN |
| 3 | MerkleOps (Blake2s, Blake2sM31, Poseidon) | commit_on_layer | same | match | GREEN |
| 4 | **MerkleOpsLifted** (Blake2s generic, Poseidon) | build_leaves(cols, **lifting_log_size**), build_next_layer — DEVICE lifted Merkle | identical sig (lifting_log_size present!) | **match — the P2 win, already device-resident** | GREEN |
| 5 | GrindOps<Blake2s M31/Poseidon> | Blake2s GPU; M31/Poseidon CPU-delegate | same (channel = Blake2sChannelGeneric<true>) | match (CPU grind OK) | GREEN |
| 6 | MleOps<BaseField>/<SecureField> | fix_first_variable | same | match | GREEN |
| 7 | Backend / BackendForChannel(3) | marker impls present | same | match | GREEN |
| 8 | PolyOps | 11 methods; Twiddles=BaseFieldVec(device); has batch_eval_at_point | + `evaluate_into` (NEW), `evaluate_polynomials(..., pool: &BaseColumnPool)` extra param | add evaluate_into; add pool param; map Twiddles | YELLOW |
| 9 | QuotientOps | accumulate_numerators(no log_blowup); compute_quotients_and_combine(accs, _lifting_log_size) 2-param | accumulate_numerators(**+log_blowup_factor**); compute_quotients_and_combine(accs, lifting_log_size, **log_blowup_factor, twiddles**) | add params + lifted/blowup semantics (kernel exists) | YELLOW |
| 10 | **FriOps** | fold_line(eval, **alpha:single**, tw, **fold_step**); fold_circle_into_line(**dst:&mut**, src, alpha, tw); decompose=**todo!()** | fold_line(eval, **alphas:&[]**, tw) returns; fold_circle_into_line(src, alpha, tw) **returns**; decompose impl'd | batch alphas; return-style; impl decompose; **+pack_leaves (ABSENT)** | RED |
| 11 | **PackLeavesOps** | **ABSENT** | pack_leaves_input(...) lifted leaf packing | implement on device (lifted-specific) | RED |
| 12 | **ComponentProver<CudaBackend>** | present but CUDA kernels are **cairo-per-component**; generic FrameworkEval → CPU fallback (CUDA_CPU_FALLBACK env) | evaluate gate_air's FrameworkEval constraints | no generic GPU evaluator for arbitrary FrameworkEval → **CPU/SIMD fallback v1** (like obelyzk did) | RED→deferred |
| 13 | GkrOps | gen_eq_evals only; next_layer/sum_as_poly = **todo!()** | 3 methods (Backend supertrait) | gate_air uses LogUp, NOT GKR → stubs compile, never called (VERIFY) | GREEN* |

## The real work (RED items)
1. **FRI reconcile + pack_leaves on device** — the hardest. Our lifted FRI packs leaves (pack_leaves)
   and batches fold alphas; NitrooZK's CUDA FRI predates both (single-alpha, fold_step, no pack_leaves,
   decompose=todo). Must: batch fold_line over alphas, switch fold_circle to return-style, implement
   decompose, and ADD pack_leaves to the device FRI/Merkle path. This is the one genuinely-new GPU
   crypto piece → byte-identity-gate it hard.
2. **PackLeavesOps on device** — lifted leaf packing kernel (pairs with #1).
3. **QuotientOps lifted/blowup params** — add log_blowup_factor + twiddles; verify the device quotient
   matches our lifted semantics.
4. **PolyOps drift** — add evaluate_into + the pool param (mechanical).
5. **ComponentProver<CudaBackend>** — for v1, SIMD/CPU-fallback gate_air's constraint eval (composition
   poly), matching obelyzk's minimal approach. Generic GPU FrameworkEval evaluator = later optimization.

## Strategy
- Transplant NitrooZK `stwo_cuda` + `backend/cuda` wholesale into stwo-gpu-port (new branch/worktree),
  drop cairo evaluate_*.cu, wire the nvcc/cmake build.
- Reconcile signatures to 74951f79 trait defs (GREEN/YELLOW first → compiles), CPU-fallback the RED
  constraint-eval, then tackle FRI pack_leaves + PackLeavesOps + QuotientOps.
- Gate EACH op byte-identical vs SimdBackend (CLAUDE.md soundness), reusing the existing
  tests/gpu_byte_identity.rs harness pattern.
- Port K1/K4 NVRTC kernels → .cu in the same build; switch their I/O to device BaseFieldVec (no host
  round-trip). K4 largely collapses into NitrooZK's existing logup/prefix-sum/batch_inverse machinery.
- THEN: gate_air-leaf prove on CudaBackend → byte-identity vs SIMD proof → measure t_gpu.

## Open questions to resolve early
- Confirm gate_air never invokes GkrOps (LogUp-only) so the todo!() stubs are safe.
- Confirm nvcc/CMake toolchain on the box; arch flags (A100 = sm_80; NitrooZK defaults 89/100/120 — ADD 80).
- Decide constraint-eval v1: pure CPU fallback vs a generic FrameworkEval GPU evaluator (cost: composition
  poly is non-trivial; but trace-gen + Merkle dominate — measure first).

## P4 STATUS (2026-06-28): backend VALIDATED; gate_air end-to-end wiring remains

DONE: NitrooZK device-resident CudaBackend transplanted into stwo-gpu-port @74951f79, reconciled
(`cargo check --features cuda` clean), and BYTE-IDENTITY VALIDATED on the A100 — a lifted-channel
PCS proof on CudaBackend == SimdBackend (crates/stwo/tests/cuda_byte_identity.rs). Branch
`cuda-backend-port`, uncommitted.

REMAINING for the gate_air GPU number (the Tanuj-curve deliverable):

1. **ComponentProver<CudaBackend> in constraint-framework.** The cuda_byte_identity test used PCS
   prove (no components); gate_air's prove_ex needs ComponentProver<CudaBackend> for
   FrameworkComponent<E>. obelyzk's mem::transmute-to-SIMD delegation does NOT work (CudaBackend
   columns = device BaseFieldVec, not layout-identical to SIMD). Options:
   - v1: convert Trace<Cuda>→Trace<Cpu/Simd> (to_cpu all committed cols) + run the audited
     SIMD/CPU constraint eval + merge the DomainEvaluationAccumulator back to Cuda. Correct but a
     D2H of the committed trace on the constraint-eval step. Need to study Trace +
     DomainEvaluationAccumulator structure for the merge-back.
   - later: native CUDA generic FrameworkEval evaluator (big; NitrooZK's is cairo-per-component).

2. **gate-air-leaf trace-backend split.** main.rs is monolithically typed to one ProverBackend for
   BOTH trace-gen (cheap per-element Col::set — col_from_values etc.) AND prove. Device columns
   (BaseFieldVec) have no cheap .set(). Cleanest: `TraceBackend=SimdBackend` (always) for trace-gen,
   `ProverBackend=Cuda` under a new `cuda` feature, + a `to_prover(Vec<CircleEvaluation<Simd>>) ->
   Vec<CircleEvaluation<ProverBackend>>` conversion at the 3 tree_builder.extend_evals points
   (identity for Simd; transmute-rewrap for obelyzk Gpu; to_cpu→BaseFieldVec::from_vec for Cuda).
   Add gate-air-leaf `cuda` feature → stwo/cuda + ProverBackend=CudaBackend.

3. Box build (--features cuda) + prove gate_air + byte-identity vs SIMD proof + measure prove_s
   (commit/NTT/Merkle/FRI on device; constraint-eval host v1) → first gate_air GPU number.

4. THEN: port K1/K4 to device-resident BaseFieldVec I/O (drop host round-trip) for the true
   on-device trace-gen + measure t_gpu_pure for the Tanuj curve.

INTERMEDIATE OPTION (quick, no ComponentProver): a TIMING bench proving a gate_air-SIZED PCS
(191 main + 24 interaction cols at gate_air's log_size) on CudaBackend vs SimdBackend — measures the
GPU-accelerated commit+NTT+Merkle+FRI at the right scale without the component machinery. A lower
bound on the gate_air prove speedup; useful while #1/#2 are built.
