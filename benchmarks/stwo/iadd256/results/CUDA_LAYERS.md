# gate_air CUDA — layered branch stack

Two-layer split of the device-resident CUDA prover: a **generic, circuit-agnostic backend** in the
stwo repo, and the **gate_air-specific** kernel + prove-path + tests in grover-tax-v02, layered as a
stack of single-commit branches. Replaces the monolithic `stwo-gpu-port` fork.

## Repos / worktrees

| Worktree | Repo | Branch | Role |
|---|---|---|---|
| `/home/anat/workspace/stwo-cuda-backend` | stwo (`74951f79`) | `anatg/cuda-backend` | generic device-resident `CudaBackend` |
| `/home/anat/workspace/grover-tax-v02` | AbdelStark/grover-tax | `anatg/gate-air-leaf{,-cuda,-cuda-tests}` | leaf prover stack |
| `/home/anat/workspace/proving-utils` | starkware-libs/proving-utils | `anatg/multiverifier-recursion` | recursion (path dep) |
| stwo-circuits | starkware-libs/stwo-circuits | git rev `0a6351e` | in-circuit verifier (cargo cache) |

> **Box build (A100, nvcc 12.9): GREEN.** `cargo build --release --features cuda` succeeds
> end-to-end (libstwo_cuda + gate_air kernel + cairo/circuits + leaf, 3m05s). See "Box build" below.

## Generic backend — `anatg/cuda-backend` (stwo-cuda-backend)
Commits: `78452c6e` (generic device-resident CudaBackend) + `01d1d525` (co-compile the gate_air
kernel `.cu` into libstwo_cuda — see "CUDA device-link" below).

Vendored NitrooZK device-resident `CudaBackend` into the stwo workspace under a new `cuda` feature:
`backend/cuda` + `stwo_cuda` (nvcc/CMake `libstwo_cuda`) + host-delegate `ComponentProver<CudaBackend>`.
Kept **circuit-agnostic**: no per-AIR kernel here. A downstream crate installs its own kernel via the
registration hook `set_gpu_constraint_kernel` / `GpuConstraintDispatch` (`cuda_constraint_kernel.rs`);
the generic prover offers each component to it (behind `CUDA_GPU_CONSTRAINTS=1`) and falls back to the
audited host-delegate when the kernel declines.

- `pub` bump: `get_constraint_quotient_inputs` / `ConstraintQuotientInputs`.
- `links = "stwo_cuda"` + build.rs exports `DEP_STWO_CUDA_INCLUDE` / `DEP_STWO_CUDA_LIB_DIR`.
- Verified: `cargo check --features cuda` (with `STWO_CUDA_SKIP_BUILD=1`).

## Leaf prover stack — grover-tax-v02 (each branch = parent + exactly 1 commit)

```
2a9eae5  base (parallel LogUp)
 └─ 0adb665  anatg/gate-air-leaf            L1  gate_air + backend seam, NO cuda/test
     └─ 26d2bb0  anatg/gate-air-leaf-cuda        L2  device CUDA prove path + Rust-only kernel crate
         └─ 1d95610  anatg/gate-air-leaf-cuda-tests  L3  CPU-vs-GPU byte-identity harness
```

**L1 `0adb665`** — pluggable prover-backend seam: `TraceBackend`/`ProverBackend` aliases (both
`SimdBackend`), `to_prover` rewrap, `prover_refs`. Builds against upstream stwo `74951f79`, no
`[patch]`. Zero behavior change. ✓ `cargo check`.

**L2 `ccd1af2`** — under `--features cuda`:
- `ProverBackend -> CudaBackend`; `[patch]` redirects stwo to `../../stwo-cuda-backend`.
- `gpu_tracegen.rs`: K1/K4 NitrooZK trace-gen (cudarc/NVRTC) → device-resident `BaseFieldVec`
  columns fed straight into commit; `gate_air_relation_m31x4`. Shared device via a local cached
  `cuda_device()` (cudarc `CudaDevice::new(0)`, device-0 primary context) — replaces obelyzk's
  `get_cuda_executor`.
- new crate **`gate-air-cuda-kernel`** (Rust-only): `src/lib.rs` = the
  `fn(GpuConstraintDispatch)->bool` hook + `set_gate_air_relation` + `register()` +
  the FFI to `evaluate_gate_air_entry`. No `.cu`, no `build.rs` — the kernel
  (`evaluate_gate_air.cu` + standalone `extern "C" gate_air_entry.cu`) is **co-compiled into
  libstwo_cuda** (see "CUDA device-link"); the entry symbol resolves at the leaf's final link.
- ✓ Builds on the A100 with `--features cuda` (and type-checks on laptop via `STWO_CUDA_SKIP_BUILD=1`).

## CUDA device-link (why the kernel `.cu` lives in libstwo_cuda)
`libstwo_cuda` is built `CUDA_RESOLVE_DEVICE_SYMBOLS ON` → self-contained, not externally
device-linkable. A separate downstream `.cu` can't device-link against its `fields` ops + finalize
kernel, and compiling those into a second lib duplicates host symbols. Resolution (chosen): add
`evaluate_gate_air.cu` + `gate_air_entry.cu` to the generic cmake build (one device-link unit, like
the existing cairo/blake/poseidon kernels). NOT wired into the generic eval_id dispatch — reached
only via `evaluate_gate_air_entry`. The **Rust** backend stays circuit-agnostic; only the compiled
artifact carries the extra kernel.

**L3 `0f0167b`** — CPU-vs-GPU comparison, all behind runtime env gates:
- `GATE_AIR_PROOF_HASH`: SHA-256 over the serde StarkProof (oracle = SimdBackend run).
- `GATE_AIR_GPU_TEST=k1|k4`: K1 trace-gen / K4 LogUp byte-identity vs CPU, then exit.
- `accumulator_diff.rs` (composition-poly diff scaffold), `bin/compare_proof_fingerprint.sh`.
- byte-identity FNs live in L2 (`gpu_tracegen.rs`); their entry points are here.
- ✓ `cargo check` (default) and `cargo check --features cuda`.

## Box build (A100) — DONE ✓

Built on `anat-ganor-instance` (us-west1-b, A100-40GB, nvcc 12.9). Layout on box:
`~/workspace/{stwo-cuda-backend, grover-tax-v02, proving-utils}` (path-dep siblings).
```
cd ~/workspace/grover-tax-v02/gate-air-leaf
export PATH=/usr/local/cuda/bin:$PATH
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cuda
```
This transitively builds everything: libstwo_cuda (cmake/nvcc, incl. evaluate_gate_air.cu +
gate_air_entry.cu), cairo/circuits, the Rust-only kernel crate, and the leaf. Result:
`Finished release [optimized] in 3m05s` (BUILD_EXIT=0). The device-link resolved and
`evaluate_gate_air_entry` linked into the leaf. (After a CMakeLists change, `cargo clean -p stwo`
forces the libstwo_cuda rebuild.)

### Still box-pending (validation, not build) — P6
- `GATE_AIR_PROOF_HASH=1`: confirm the `--features cuda` proof fingerprint == the SimdBackend oracle.
- `GATE_AIR_GPU_TEST=k1|k4`: confirm K1/K4 trace-gen byte-identity vs CPU.
- The gate_air constraint kernel itself is BOX-UNVALIDATED for soundness (Phase-2 LogUp gate).
