#!/usr/bin/env bash
# Background: N=4 cairo-leaf multiverifier-tree smoke (recursive_aggregate) on stwo-vm, then stop VM.
# Builds/runs the prover ONLY on the VM, optimized (target-cpu=native, release).
# Paths that must resolve on the REMOTE are single-quoted '$HOME/...' (vm home=/home/anatg_starkware_co).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/recursion_smoke.txt
PU='$HOME/workspace/proving-utils'
TC=nightly-2026-01-15

echo "=== recursion smoke (N=4 cairo leaves) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# Sync my branch working tree (recursive_aggregate + circuit_unpacker + Cargo.toml/lock) over the VM's proving-utils.
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

# Ensure the toolchain the new rev needs is present (no-op if already installed).
timeout 300 $SSH "~/.cargo/bin/rustup toolchain install $TC --profile minimal 2>&1 | tail -1" >>"$OUT" 2>&1

echo "=== build + run smoke (target-cpu=native, release) $(date -u) ===" >> "$OUT"
timeout 3000 $SSH "cd $PU; \
  RUSTFLAGS='-C target-cpu=native' RUST_LOG=info RUST_BACKTRACE=1 \
  ~/.cargo/bin/cargo +$TC test -p recursive-aggregate --release \
  --test smoke_cairo_tree -- --nocapture --test-threads=1 > /tmp/rsmoke.log 2>&1; echo EXIT=\$?" >>"$OUT" 2>&1

echo "=== result / timing ===" >> "$OUT"
$SSH "grep -inE 'SMOKE OK|WRAPPER|ROOT-VERIFY|test result|FAILED|EXIT=' /tmp/rsmoke.log | tail -20" >>"$OUT" 2>&1
echo "=== full panic / error region (last 70 lines) ===" >> "$OUT"
$SSH "tail -70 /tmp/rsmoke.log" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
