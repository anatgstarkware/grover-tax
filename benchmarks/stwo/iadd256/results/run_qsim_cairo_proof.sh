#!/usr/bin/env bash
# Background: generate ONE qsim cairo proof on stwo-vm under the bumped revs, then stop the VM.
# Builds stwo-run-and-prove (bump: nightly-2026-01-15, stwo-cairo b0b74007) and runs iadd-sim-prove
# (simple bootloader + qsim executable) → a serialized CairoProof (stage 1 of the qsim-leaf pipeline).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/qsim_cairo_proof.txt
PU='$HOME/workspace/proving-utils'
GTV='$HOME/workspace/grover-tax-v02'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1-n9024.json'
TC=nightly-2026-01-15

echo "=== qsim cairo proof gen $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# Sync bumped proving-utils + the qsim assets + the fixture.
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
$SSH "mkdir -p ~/workspace/grover-tax-v02/bin ~/workspace/grover-tax-v02/stwo-side/cairo/target/dev ~/workspace/grover-tax-v02/results ~/workspace/grover-tax/fixtures" >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax-v02/bin/iadd-sim-prove \
  stwo-vm:~/workspace/grover-tax-v02/bin/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax-v02/stwo-side/cairo/target/dev/iadd_sim_executable.executable.json \
  stwo-vm:~/workspace/grover-tax-v02/stwo-side/cairo/target/dev/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1-n9024.json \
  stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build stwo-run-and-prove (target-cpu=native, release) $(date -u) ===" >> "$OUT"
timeout 5400 $SSH "cd $PU; RUSTFLAGS='-C target-cpu=native' \
  ~/.cargo/bin/cargo +$TC build --release -p stwo-run-and-prove 2>&1 | tail -3" >>"$OUT" 2>&1

echo "=== run iadd-sim-prove (k1 fixture, N=1) $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GTV; chmod +x bin/iadd-sim-prove; \
  RUST_LOG=info PROVING_UTILS_ROOT=$PU bash bin/iadd-sim-prove $FX 1 > /tmp/qsimproof.log 2>&1; echo EXIT=\$?" >>"$OUT" 2>&1

echo "=== result ===" >> "$OUT"
$SSH "grep -inE 'iadd-sim-prove|proved|verified|Num steps|panicked|error|EXIT=' /tmp/qsimproof.log | tail -25" >>"$OUT" 2>&1
$SSH "ls -la $GTV/results/iadd_sim.proof.bin 2>&1 | tail -1" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
