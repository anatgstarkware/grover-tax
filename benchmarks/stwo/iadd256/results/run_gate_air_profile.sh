#!/usr/bin/env bash
# Profile the "trace_gen" 30s @ 2^25 into phases: sim / preprocessed / main_trace witness / tree1
# commit / interaction witness / tree2 commit / prove_ex. Separates MY witness gen from stwo commits.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_profile.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== trace_gen phase profile @ 2^25 $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_pf.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

echo "=== samples=8 (2^25), base prove only ===" >> "$OUT"
timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'phase|simulated|trace generation|report'" >>"$OUT" 2>&1
echo "--- repeat (noise check) ---" >> "$OUT"
timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'phase|simulated|trace generation'" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
