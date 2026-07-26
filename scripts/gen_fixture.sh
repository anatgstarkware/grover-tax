#!/usr/bin/env bash
# Generate the iadd256 benchmark fixture for repetition count k (circuit + samples fixed at the
# benchmark defaults: iadd256.kmx, n=9024). Usage (from repo root): scripts/gen_fixture.sh <k>
#   -> fixtures/v0.3-iadd256-k<k>-n9024.json
set -euo pipefail
k="${1:?usage: $0 <k>}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root (the uv project)
exec uv run gen-iadd-fixtures --circuit iadd256.kmx --repetitions "$k" --samples 9024
