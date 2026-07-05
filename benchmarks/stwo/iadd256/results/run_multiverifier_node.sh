#!/usr/bin/env bash
# Background: measure one 2-to-1 multiverifier node prove on stwo-vm, then stop VM.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/multiverifier_node.txt
SC='$HOME/workspace/sc-pr507'

echo "=== multiverifier node measure $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/sc-pr507 stwo-vm:~/workspace/ >>"$OUT" 2>&1
echo "rsynced; building + running multiverifier prove test (target-cpu=native) $(date -u)" >> "$OUT"

# rust-toolchain.toml pins nightly-2026-01-15 (already installed). cargo test builds then runs.
timeout 3000 $SSH "cd $SC; RUSTFLAGS='-C target-cpu=native' RUST_LOG=info ~/.cargo/bin/cargo test --release -p circuit-multiverifier test_prove_multiverifier_of_two_cairo_subcircuits -- --nocapture --test-threads=1 > /tmp/mv.log 2>&1; echo EXIT=\$?" >>"$OUT" 2>&1

echo "=== prove spans / test result ===" >> "$OUT"
$SSH "grep -inE 'test result|running [0-9]+ test|prove.*close time.busy|prove_circuit|Num steps|panicked|error\\[|FAILED|ok\\.' /tmp/mv.log | tail -50" >>"$OUT" 2>&1
echo "=== full prove-related tail ===" >> "$OUT"
$SSH "grep -iE 'prove|commitment|fri|merkle|time.busy' /tmp/mv.log | tail -40" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
