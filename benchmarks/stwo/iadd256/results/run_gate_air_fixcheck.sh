#!/usr/bin/env bash
# Validate the DYNAMIC preprocessed-order fix: the configs that failed (main>16) should now pass
# both circuits_stark_verifier (INCIRCUIT) and the leaf prove + fold + root (FOLD).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_fixcheck.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== gate_air dynamic-preprocessed-order fix check $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git/ \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_fxb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -2 /tmp/gal_fxb.log" >>"$OUT" 2>&1

echo ">>> INCIRCUIT 2^22 (k1000 s=1) — expect verify OK" >> "$OUT"
timeout 600 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_BACKTRACE=1 GATE_AIR_INCIRCUIT=1 $BIN --fixture $FX --samples 1 2>&1 | grep -iE 'verify OK|verify FAILED|panicked|merkle|left:|right:' | head -5" >>"$OUT" 2>&1

echo ">>> FOLD=2 2^24 (k1000 s=4) — expect leaves proved + root verification OK" >> "$OUT"
timeout 1200 $SSH "cd $GAL; RUSTFLAGS='$RF' GATE_AIR_FOLD=2 $BIN --fixture $FX --samples 4 2>&1 | grep -iE 'gate-air:|leaves proved|folded|root verification|panicked|prover.rs|FAILED' | head -10" >>"$OUT" 2>&1

echo ">>> FOLD=4 2^25 (k1000 s=8) — the original campaign failure; expect full pipeline OK" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' GATE_AIR_FOLD=4 $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'gate-air:|leaves proved|folded|root verification|panicked|prover.rs|FAILED' | head -10" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
