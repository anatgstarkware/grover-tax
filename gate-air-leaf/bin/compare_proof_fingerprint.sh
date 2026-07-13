#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Byte-identity validation harness for the gate_air leaf prover (Deliverable 2,
# P5_GPU_CONSTRAINT_SCOPE.md §2.1). Proves the SAME fixture+samples twice:
#   1. default features  -> SimdBackend  (the deterministic golden CPU oracle)
#   2. --features cuda    -> CudaBackend (device-resident GPU prover)
# Both emit `gate-air: proof_fingerprint=<hex>` via GATE_AIR_PROOF_HASH=1.
# The two hex fingerprints MUST be EQUAL: a byte-identical ExtendedStarkProof
# proves end-to-end backend equivalence under the fixed Fiat-Shamir transcript.
#
# IMPORTANT: the --features cuda run is GPU/box-only (needs nvcc + libstwo_cuda +
# a CUDA device). Run it on the GPU box, never on the laptop. The default
# (SimdBackend) run is the oracle and runs anywhere.
# ---------------------------------------------------------------------------
set -euo pipefail

FIXTURE="${FIXTURE:-../../../../fixtures/v0.3-iadd256-k4-n16.json}"
SAMPLES="${SAMPLES:-1}"
TOOLCHAIN="${TOOLCHAIN:-+nightly-2026-01-15}"
# Use native codegen for representative numbers (per project policy).
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

cd "$(dirname "$0")/.."

run() {
  # $1 = label, $2.. = extra cargo args (e.g. --features cuda)
  local label="$1"; shift
  echo "==> [$label] proving fixture=$FIXTURE samples=$SAMPLES" >&2
  # This harness validates a SINGLE non-fold proof (SimdBackend vs CudaBackend
  # byte-identity). GATE_AIR_FOLD / GATE_AIR_PIPELINE are now default-ON, so pin
  # them OFF to keep the single-proof (non-fold, sequential) path under test.
  GATE_AIR_FOLD=0 GATE_AIR_PIPELINE=0 \
  GATE_AIR_PROOF_HASH=1 cargo "$TOOLCHAIN" run --release "$@" -- \
    --fixture "$FIXTURE" --samples "$SAMPLES" 2>&1 \
    | grep 'gate-air: proof_fingerprint=' | sed 's/.*proof_fingerprint=//'
}

CPU_FP="$(run cpu-simd)"
GPU_FP="$(run gpu-cuda --features cuda)"

echo "cpu  (SimdBackend) fingerprint: $CPU_FP"
echo "cuda (CudaBackend) fingerprint: $GPU_FP"

if [[ "$CPU_FP" == "$GPU_FP" && -n "$CPU_FP" ]]; then
  echo "PASS: CudaBackend proof is BYTE-IDENTICAL to the SimdBackend oracle."
  exit 0
else
  echo "FAIL: fingerprints differ (or empty). The backend under test diverged from the oracle." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Sub-path A/B toggles (set in the environment before invoking, or export and
# re-run individual `cargo run` commands by hand). All are read-only switches
# that route between equivalent paths; a correct port keeps the fingerprint
# CONSTANT across every toggle setting:
#
#   GATE_AIR_CPU_TRACEGEN=1
#       Forces the legacy CPU-build-then-upload trace-gen path instead of the
#       device-resident K1/K4 NitrooZK kernels (cuda feature only). A/B this to
#       isolate trace-gen from commit+prove: the fingerprint must not change.
#
#   CUDA_CONSTRAINT_CPU_FALLBACK=1
#       Routes the CudaBackend constraint-quotient evaluation through the audited
#       host delegate (accumulate_pointwise_cpu). Today both settings already use
#       the host delegate; once the GPU constraint kernel lands (Deliverable 1),
#       =1 keeps the CPU reference path reachable and =0 selects the kernel. The
#       fingerprint must be identical under both — that is the kernel's acceptance
#       test. cf. cuda_component_prover.rs cpu_fallback_forced().
#
# Example A/B (on the box):
#   FIXTURE=... SAMPLES=... CUDA_CONSTRAINT_CPU_FALLBACK=1 bin/compare_proof_fingerprint.sh
#   FIXTURE=... SAMPLES=... GATE_AIR_CPU_TRACEGEN=1        bin/compare_proof_fingerprint.sh
# ---------------------------------------------------------------------------
