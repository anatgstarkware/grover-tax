#!/usr/bin/env bash
# Localize the leaf-prove lookup_sum panic (circuit_prover/prover.rs:177). Disambiguate the axis:
#   trace SIZE vs SHOT COUNT vs combination. Works at 2^14/1-shot, fails at 2^25/8-shot.
# For each config, run BOTH the circuits_stark_verifier check (INCIRCUIT) and the leaf prove (FOLD).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_leafdiag.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"
K1000="$FXDIR/v0.3-iadd256-k1000-n9024.json"
K4="$FXDIR/v0.3-iadd256-k4-n16.json"

echo "=== gate_air leaf-prove panic localization $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k4-n16.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_db.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

run() { # <label> <env> <fixture> <samples>
  echo ">>> $1" >> "$OUT"
  timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $2 $BIN --fixture $3 --samples $4 2>&1 | grep -iE 'gate-air:|verify OK|leaves proved|fold OK|panicked|left == right|prover.rs|FAILED|log_rows' | head -8" >>"$OUT" 2>&1
}

echo "=== baseline (known good): 2^14, 1 shot ===" >> "$OUT"
run "INCIRCUIT k4 s=1 (2^16,1shot)"  "GATE_AIR_INCIRCUIT=1" "$K4" 1
run "FOLD=2 k4 s=1 (2^16,1shot)"     "GATE_AIR_FOLD=2"      "$K4" 1
echo "=== axis: SHOT COUNT (small trace, many shots) ===" >> "$OUT"
run "INCIRCUIT k4 s=4 (2^16,4shot)"  "GATE_AIR_INCIRCUIT=1" "$K4" 4
run "FOLD=2 k4 s=4 (2^16,4shot)"     "GATE_AIR_FOLD=2"      "$K4" 4
echo "=== axis: TRACE SIZE (1 shot, bigger trace) ===" >> "$OUT"
run "INCIRCUIT k1000 s=1 (2^22,1shot)" "GATE_AIR_INCIRCUIT=1" "$K1000" 1
run "FOLD=2 k1000 s=1 (2^22,1shot)"    "GATE_AIR_FOLD=2"      "$K1000" 1
echo "=== combination: bigger trace + multishot ===" >> "$OUT"
run "FOLD=2 k1000 s=4 (2^24,4shot)"    "GATE_AIR_FOLD=2"      "$K1000" 4

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
