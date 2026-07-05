#!/usr/bin/env bash
# Verify sim (shot simulation, par over shots) scales with shot count. Hold trace ~2^25 (20.4M rows
# = samples*k*2547) fixed, vary shots: k1000 s=8 (8 shots), k100 s=80 (80), k10 s=800 (800),
# k1 s=8000 (8000). sim time should fall until ~96 cores saturate (~>=96 shots).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_simscale.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== sim scaling vs shot count (~2^25 fixed) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  $FXDIR/v0.3-iadd256-k1-n9024.json $FXDIR/v0.3-iadd256-k10-n9024.json \
  $FXDIR/v0.3-iadd256-k100-n9024.json $FXDIR/v0.3-iadd256-k1000-n9024.json \
  stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_sc.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

run() { # <fixture-k> <samples> <shots-label>
  echo ">>> $3 shots (fixture k=$1, samples=$2)" >> "$OUT"
  timeout 600 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FXDIR/v0.3-iadd256-k$1-n9024.json --samples $2 2>&1 | grep -iE 'simulated|real_rows=|\[phase\] interaction'" >>"$OUT" 2>&1
}
run 1000 8 8
run 100 80 80
run 10 800 800
run 1 8000 8000

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
