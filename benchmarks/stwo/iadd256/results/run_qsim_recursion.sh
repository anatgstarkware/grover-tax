#!/usr/bin/env bash
# Background: full qsim recursion pipeline on stwo-vm, then stop the VM.
# Regenerates the qsim cairo proof in BINARY, then builds+runs the iadd-recursion harness:
#   qsim cairo proof -> N cairo-verifier leaves -> multiverifier fold -> zk-blinded root verification.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/qsim_recursion.txt
PU='$HOME/workspace/proving-utils'
GTV='$HOME/workspace/grover-tax-v02'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1-n9024.json'
PARAMS='$HOME/workspace/grover-tax-v02/iadd-recursion/cairo_prover_params.json'
TC=nightly-2026-01-15

echo "=== qsim recursion pipeline $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# Sync proving-utils (recursion branch, w/ recursive_aggregate helpers) + the iadd-recursion crate.
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/iadd-recursion/ stwo-vm:~/workspace/grover-tax-v02/iadd-recursion/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax-v02/bin/iadd-sim-prove stwo-vm:~/workspace/grover-tax-v02/bin/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

# Regenerate the qsim cairo proof in BINARY format with Blake2sM31 channel (what the cairo-verifier
# circuit consumes).
echo "=== regenerate qsim cairo proof (binary, Blake2sM31) $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GTV; rm -f results/iadd_sim.proof.bin; \
  RUST_LOG=warn PROVING_UTILS_ROOT=$PU PROOF_FORMAT=extended-binary PROVER_PARAMS=$PARAMS \
  bash bin/iadd-sim-prove $FX 1 > /tmp/qsimregen.log 2>&1; echo REGEN_EXIT=\$?" >>"$OUT" 2>&1
$SSH "tail -20 /tmp/qsimregen.log; ls -la $GTV/results/iadd_sim.proof.bin 2>&1 | tail -1" >>"$OUT" 2>&1
$SSH "ls -la $GTV/results/iadd_sim.proof.bin 2>&1 | tail -1" >>"$OUT" 2>&1

# Build + run the recursion harness.
echo "=== build + run iadd-recursion (N=4, target-cpu=native) $(date -u) ===" >> "$OUT"
timeout 5400 $SSH "cd $GTV/iadd-recursion; \
  RUSTFLAGS='-C target-cpu=native' RUST_LOG=warn RUST_BACKTRACE=1 \
  ~/.cargo/bin/cargo +$TC run --release -- $GTV/results/iadd_sim.proof.bin 4 > /tmp/qsimrec.log 2>&1; echo EXIT=\$?" >>"$OUT" 2>&1

echo "=== result ===" >> "$OUT"
$SSH "grep -inE 'Deserializing|config derived|leaves proved|folded to root|ROOT-VERIFY|pipeline complete|panicked|assertion|error\\[|EXIT=' /tmp/qsimrec.log | tail -30" >>"$OUT" 2>&1
echo "=== tail (panic/error region) ===" >> "$OUT"
$SSH "tail -40 /tmp/qsimrec.log" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
