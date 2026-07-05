#!/usr/bin/env bash
# F1 validation: secure base config (blowup 1, n_queries 70, pow 26 = 96-bit). Measure:
#  - base prove_s at 2^22/2^25 (expect ~unchanged: prover cost ~ blowup, not n_queries)
#  - REALISTIC leaf cost + config target (no longer the toy 2^20 floor) via FOLD at 2^25
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_secure.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== F1 secure base config $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git/ \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_sec.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -2 /tmp/gal_sec.log" >>"$OUT" 2>&1

echo "=== base prove_s (secure) at 2^22 / 2^25 ===" >> "$OUT"
for S in 1 8; do
  timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FX --samples $S 2>/dev/null | grep gate-air-report" >>"$OUT" 2>&1
done

echo "=== REALISTIC leaf: FOLD=4 at 2^25 (secure base) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' GATE_AIR_FOLD=4 $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'gate-air:|config derived|leaves proved|folded|root verification|panicked|prover.rs|FAILED' | head -12" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
