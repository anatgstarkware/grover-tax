#!/usr/bin/env bash
# M2d validation: prove gate_air (canonical transcript) AND verify it IN-CIRCUIT
# (circuits_stark_verifier) on stwo-vm. Smallest case first (k4-n16, samples=1 -> ~2^14 proof).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_incircuit.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"

echo "=== gate_air in-circuit verify (M2d) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k4-n16.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_b.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -3 /tmp/gal_b.log" >>"$OUT" 2>&1

echo "=== run k4-n16 samples=1, GATE_AIR_INCIRCUIT=1 $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_BACKTRACE=full GATE_AIR_INCIRCUIT=1 \
  ./target/release/gate-air-leaf --fixture $FXDIR/v0.3-iadd256-k4-n16.json --samples 1 > /tmp/gal_ic.log 2>&1; echo RUN=\$?" >>"$OUT" 2>&1
$SSH "grep -iE 'DEBUG|in-circuit verify|proved|panicked|FAILED|assertion|lookup_sum|Lifting|error' /tmp/gal_ic.log | head -30; echo '--- tail ---'; tail -45 /tmp/gal_ic.log" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
