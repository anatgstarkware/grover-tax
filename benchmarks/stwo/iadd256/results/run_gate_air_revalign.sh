#!/usr/bin/env bash
# Milestone 1 validation: confirm the rev-aligned gate_air (stwo 74951f79 / nightly-2026-01-15)
# still PROVES + VERIFIES natively, at small and larger N. <1h budget.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_revalign.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"

echo "=== gate_air rev-align validation $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k4-n16.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build (target-cpu=native) $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_build.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -3 /tmp/gal_build.log" >>"$OUT" 2>&1

# k4-n16, 16 samples (the doc's validated case) + k1-n9024 at 1024 samples (2^22) for scale.
echo "=== run k4-n16 samples=16 $(date -u) ===" >> "$OUT"
timeout 600 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_LOG=warn \
  ./target/release/gate-air-leaf --fixture $FXDIR/v0.3-iadd256-k4-n16.json --samples 16 2>&1 | tail -15; echo RUN=\$?" >>"$OUT" 2>&1
echo "=== run k1-n9024 samples=1024 $(date -u) ===" >> "$OUT"
timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_LOG=warn \
  ./target/release/gate-air-leaf --fixture $FXDIR/v0.3-iadd256-k1-n9024.json --samples 1024 2>&1 | tail -15; echo RUN=\$?" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
