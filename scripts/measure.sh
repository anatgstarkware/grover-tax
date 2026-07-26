#!/usr/bin/env bash
#
# measure.sh — one prover's measured run series (RFC-0008 §"`scripts/measure.sh`").
#
# Usage:
#   scripts/measure.sh <prover> <run_id>
#
# Where `<prover>` ∈ {sp1, stwo} and `<run_id>` is the per-series
# identifier the harness picks (typically `<epoch_ts>-<short_repo_sha>`).
#
# Drives `hyperfine` (M1, M5) and `gnu-time` (M2, M3, M4) on the prover's
# wrapper (`bin/run_<prover>.sh`) and verifier (`bin/verify_<prover>.sh`),
# captures the proof file size (M6), the prover log (M7), and runs the
# post-series discard check.
#
# Exit codes (per docs/spec/04-error-model.md):
#   0 — full series captured.
#   5 — `MEASUREMENT.*` precondition violation (preflight fail, missing
#       hyperfine / gnu-time, GPU residency mid-run, etc.).
#   2 — usage / argv shape.
#   propagates wrapper exit codes otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/locale_env.sh"

if [[ $# -ne 2 ]]; then
  echo "MEASUREMENT.ENV_VAR_MISS: measure.sh expects <prover> <run_id>, got $#" >&2
  echo "Usage: scripts/measure.sh <sp1|stwo> <run_id>" >&2
  exit 2
fi

PROVER="$1"
RUN_ID="$2"

case "${PROVER}" in
  sp1|stwo) ;;
  *) echo "MEASUREMENT.ENV_VAR_MISS: prover must be 'sp1' or 'stwo', got '${PROVER}'" >&2; exit 2 ;;
esac

# RFC-0008 hard-locked sample counts.
HYPERFINE_PROVE_WARMUP="${HYPERFINE_PROVE_WARMUP:-1}"
HYPERFINE_PROVE_RUNS="${HYPERFINE_PROVE_RUNS:-10}"
HYPERFINE_VERIFY_WARMUP="${HYPERFINE_VERIFY_WARMUP:-3}"
HYPERFINE_VERIFY_RUNS="${HYPERFINE_VERIFY_RUNS:-50}"

# Env caps the prover wrappers require. measure.sh re-exports them here so
# its child invocations inherit a clean baseline.
export CUDA_VISIBLE_DEVICES=""
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export OMP_NUM_THREADS=1
export RUST_LOG="${RUST_LOG:-info}"

# Locate prover wrappers.
PROVER_WRAPPER="${REPO_ROOT}/bin/run_${PROVER}.sh"
VERIFIER_WRAPPER="${REPO_ROOT}/bin/verify_${PROVER}.sh"
if [[ ! -x "${PROVER_WRAPPER}" || ! -x "${VERIFIER_WRAPPER}" ]]; then
  echo "BUILD.${PROVER^^}_PATCH_FAIL: wrappers not found at ${PROVER_WRAPPER} / ${VERIFIER_WRAPPER}" >&2
  exit 3
fi

# Toolchain dependencies.
require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MEASUREMENT.ENV_VAR_MISS: ${1} not on PATH (required by RFC-0008)" >&2
    exit 5
  fi
}
require_tool hyperfine
if command -v gtime >/dev/null 2>&1; then
  GNU_TIME="gtime"
elif [[ -x /usr/bin/time ]]; then
  GNU_TIME="/usr/bin/time"
else
  echo "MEASUREMENT.ENV_VAR_MISS: GNU time (gtime or /usr/bin/time) not on PATH" >&2
  exit 5
fi

# 1. Preflight (skip-controllable for tests via SKIP_PREFLIGHT=1).
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if ! bash "${REPO_ROOT}/scripts/preflight.sh"; then
    echo "MEASUREMENT.* series-level precondition failed; aborting series" >&2
    exit 5
  fi
fi

# 2. Output paths.
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}"
BASE="${RESULTS_DIR}/${PROVER}_v0.1_${RUN_ID}"
FIXTURE="${REPO_ROOT}/fixtures/v0.2.json"
if [[ ! -f "${FIXTURE}" ]]; then
  echo "MEASUREMENT.ENV_VAR_MISS: ${FIXTURE} missing; run gen-fixtures first" >&2
  exit 5
fi

TIMING_JSON="${BASE}.timing.json"
TIME_TXT="${BASE}.time.txt"
VERIFY_JSON="${BASE}.verify.json"
PROVERLOG_TXT="${BASE}.proverlog.txt"
PROOF_SIZE_TXT="${BASE}.proof_size.txt"
PROOF_BIN="${BASE}.proof.bin"

cd "${REPO_ROOT}"

# 3. M1 — proof gen wall-clock via hyperfine.
echo "measure.sh: capturing M1 (proof gen wall-clock, ${HYPERFINE_PROVE_RUNS} runs)..."
hyperfine \
  --warmup "${HYPERFINE_PROVE_WARMUP}" \
  --runs "${HYPERFINE_PROVE_RUNS}" \
  --export-json "${TIMING_JSON}" \
  --shell=bash \
  "bash ${PROVER_WRAPPER} ${FIXTURE} ${PROOF_BIN}"

# 4. M2 / M3 / M4 — peak RSS + user/sys CPU via gnu-time on one canonical run.
echo "measure.sh: capturing M2/M3/M4 (peak RSS, user CPU, sys CPU)..."
"${GNU_TIME}" -v -o "${TIME_TXT}" \
  bash "${PROVER_WRAPPER}" "${FIXTURE}" "${PROOF_BIN}" > "${PROVERLOG_TXT}" 2>&1

# 5. M6 — proof file size in bytes.
echo "measure.sh: capturing M6 (proof file size)..."
if stat -f%z "${PROOF_BIN}" >/dev/null 2>&1; then
  stat -f%z "${PROOF_BIN}" > "${PROOF_SIZE_TXT}"   # macOS BSD stat
else
  stat -c%s "${PROOF_BIN}" > "${PROOF_SIZE_TXT}"   # GNU stat
fi

# 6. M5 — verifier wall-clock via hyperfine.
echo "measure.sh: capturing M5 (verifier wall-clock, ${HYPERFINE_VERIFY_RUNS} runs)..."
hyperfine \
  --warmup "${HYPERFINE_VERIFY_WARMUP}" \
  --runs "${HYPERFINE_VERIFY_RUNS}" \
  --export-json "${VERIFY_JSON}" \
  --shell=bash \
  "bash ${VERIFIER_WRAPPER} ${PROOF_BIN}"

# 7. Optional M10 — iostat capture (lands with #34); soft-skip if missing.
if [[ -x "${REPO_ROOT}/scripts/iostat_capture.sh" ]]; then
  echo "measure.sh: M10 (iostat) capture (informational)..."
  bash "${REPO_ROOT}/scripts/iostat_capture.sh" "${PROVER}" "${RUN_ID}" || true
fi

# 8. Post-run discard check (lands with #34); soft-skip if missing.
if [[ -x "${REPO_ROOT}/scripts/post_run_discard_check.sh" ]]; then
  echo "measure.sh: running post-run discard check..."
  bash "${REPO_ROOT}/scripts/post_run_discard_check.sh" "${PROVER}" "${RUN_ID}" || {
    rc=$?
    echo "measure.sh: post-run discard check exited ${rc}" >&2
    exit ${rc}
  }
fi

echo "measure.sh: ${PROVER}/${RUN_ID} series captured. Artifacts:"
ls -la "${BASE}".*
exit 0
