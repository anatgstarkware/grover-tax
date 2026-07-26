#!/usr/bin/env bash
# Regenerate gate-air-leaf/src/recursion_consts.rs (the pinned RECURSION_CONFIG) for repetition k.
#
# Runs the `capture_all` test for k (recomputes the leaf/level1/fold/unpacker verifier config fresh
# from the k-matching fixture), rewrites the `<<GENERATED CONSTS>>` region via
# scripts/gen_recursion_consts.py, then `cargo fmt`. The capture builds only preprocessed circuits +
# verifier configs (no GPU prover) — CPU-only, no `--features cuda`; STWO_CUDA_SKIP_BUILD=1 skips nvcc.
# Needs the k fixture first (scripts/gen_fixture.sh <k>). Usage (from repo root): scripts/regen_recursion_consts.sh <k>
#
# Shard size defaults to a 2^26-row resident trace. On a smaller-memory GPU, export RECURSION_SHARD_SHOTS
# to the shot count you'll run with (e.g. floor(2^25/(k*2547)) for a 40GB GPU); the capture, the generated
# consts, and the run all read the same value, so they stay in sync. Use that SAME value when you run the binary.
set -euo pipefail
k="${1:?usage: $0 <k>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$root/gate-air-leaf"
n_gates=2547
shots="${RECURSION_SHARD_SHOTS:-$(( (1 << 26) / (k * n_gates) ))}"
fixture="$root/fixtures/v0.3-iadd256-k${k}-n9024.json"

[[ -f "$fixture" ]] || { echo "missing fixture for k=$k: $fixture (run scripts/gen_fixture.sh $k)" >&2; exit 1; }

echo "regen recursion_consts for k=$k (shots_per_shard=$shots); fixture $fixture"
log="$(mktemp)"; trap 'rm -f "$log"' EXIT

cd "$crate"  # so .cargo/config.toml (target-cpu=native) applies
CAPTURE_K="$k" STWO_CUDA_SKIP_BUILD=1 cargo test --release \
    capture_all -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"

python3 "$root/scripts/gen_recursion_consts.py" "$log" "$crate/src/recursion_consts.rs"
cargo fmt  # reflow the generated consts to repo style

echo
echo "done. review: git diff gate-air-leaf/src/recursion_consts.rs ; then scripts/build.sh"
