#!/usr/bin/env bash
# Background: validate the privacy prove<->verify round-trip under the bumped revs on stwo-vm.
# Runs the two slow-tests (privacy_prove->verify_cairo, privacy_recursive_prove->verify_recursive_circuit),
# which is the semantic check that the new M31 public-data serialization round-trips. Then stops the VM.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/privacy_roundtrip.txt
PU='$HOME/workspace/proving-utils'
TC=nightly-2026-01-15

echo "=== privacy prove<->verify round-trip $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# Sync the recursion branch working tree (carries the bump + the privacy crates) onto the VM.
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"
timeout 300 $SSH "~/.cargo/bin/rustup toolchain install $TC --profile minimal 2>&1 | tail -1" >>"$OUT" 2>&1

echo "=== build + run privacy slow-tests (target-cpu=native, release) $(date -u) ===" >> "$OUT"
# Generous timeout: first build compiles the full cairo prover (stwo-cairo b0b74007) from scratch.
timeout 5400 $SSH "cd $PU; \
  RUSTFLAGS='-C target-cpu=native' RUST_LOG=info RUST_BACKTRACE=1 \
  ~/.cargo/bin/cargo +$TC test -p privacy-prove -p privacy-circuit-verify --release --features slow-tests \
  -- --nocapture --test-threads=1 > /tmp/privrt.log 2>&1; echo EXIT=\$?" >>"$OUT" 2>&1

echo "=== result / timing ===" >> "$OUT"
$SSH "grep -inE 'running [0-9]+ test|test .* \.\.\. (ok|FAILED)|test result|FAILED|panicked|assertion|Update .*consts|EXIT=' /tmp/privrt.log | tail -40" >>"$OUT" 2>&1
echo "=== last 60 lines (panic/error region if any) ===" >> "$OUT"
$SSH "tail -60 /tmp/privrt.log" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
