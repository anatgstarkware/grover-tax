#!/usr/bin/env bash
# Base-blowup sweep: BASE_BLOWUP in {1,2,3} at a fixed 2^25 shard. Measures (a) base prove_s (expect
# ~2x per blowup step: prover cost ~ 2^blowup eval domain) and (b) leaf-wrapper cost + config target
# via FOLD (does higher base blowup -> fewer base queries -> cheaper leaf, or is it node-floored?).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_basebuf.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== base-blowup sweep (BASE_BLOWUP 1/2/3 @ 2^25) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git/ \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_bb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

for B in 1 2 3; do
  echo ">>> BASE_BLOWUP=$B (2^25 shard) — base prove_s + leaf cost" >> "$OUT"
  timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' BASE_BLOWUP=$B GATE_AIR_FOLD=2 $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'report|config derived|leaves proved|folded|root verification|panicked|FAILED' | head -8" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
