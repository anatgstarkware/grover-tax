#!/usr/bin/env bash
# Localize WHICH verify.rs eq fails at 2^22 (main>rc crossover). enable_assert_eq_on_eval makes the
# first failing eq panic with a backtrace (composition vs merkle decommit vs size-assert).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_loc22.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== loc22: which eq fails at 2^22 $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_lb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -2 /tmp/gal_lb.log" >>"$OUT" 2>&1

# Also probe the exact threshold: samples=1 on k10 (2^15) / k50 (~2^17) / k100 (~2^18) — but k1000 s=1
# = 2^22 is the cheap known-fail. Run it with full backtrace.
echo "=== 2^22 (k1000 s=1) with backtrace $(date -u) ===" >> "$OUT"
timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_BACKTRACE=full GATE_AIR_INCIRCUIT=1 $BIN --fixture $FX --samples 1 > /tmp/gal_l.log 2>&1; echo RUN=\$?" >>"$OUT" 2>&1
$SSH "grep -nE 'panicked|verify_merkle_path|decommit|::verify::|composition|merkle|FAILED|left:|right:|circuits_stark_verifier|fri_|ops.rs' /tmp/gal_l.log | head -30" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
