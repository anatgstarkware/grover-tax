#!/usr/bin/env bash
# Build the gate-air-leaf CUDA prover. target-cpu=native comes from gate-air-leaf/.cargo/config.toml.
# CUDA toolkit path defaults to the box image's cuda-12.9; export CUDA_HOME first to override.
# Usage (from repo root): scripts/build.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CUDA_HOME:=/usr/local/cuda-12.9}"
export CUDA_HOME CUDADIR="$CUDA_HOME" PATH="$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
cd "$root/gate-air-leaf"
exec cargo build --release --features cuda
