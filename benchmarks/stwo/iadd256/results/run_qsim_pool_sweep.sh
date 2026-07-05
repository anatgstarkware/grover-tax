#!/usr/bin/env bash
# Pool-partitioning sweep for the qsim recursion pipeline on stwo-vm: find the best single-machine
# wall-clock by varying POOL_THREADS (=> n_pools = cores/POOL_THREADS). Reuses the existing
# extended-binary qsim cairo proof on disk (regenerates only if missing). Builds the harness once,
# then runs N=4 at each partitioning. Stops the VM at the end.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/qsim_pool_sweep.txt
PU='$HOME/workspace/proving-utils'
GTV='$HOME/workspace/grover-tax-v02'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1-n9024.json'
PARAMS='$HOME/workspace/grover-tax-v02/iadd-recursion/cairo_prover_params.json'
PROOF='$HOME/workspace/grover-tax-v02/results/iadd_sim.proof.bin'
TC=nightly-2026-01-15

echo "=== qsim pool sweep $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/iadd-recursion/ stwo-vm:~/workspace/grover-tax-v02/iadd-recursion/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax-v02/bin/iadd-sim-prove stwo-vm:~/workspace/grover-tax-v02/bin/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

# Regenerate the qsim proof only if it is missing (it persists on the disk across VM stop/start).
echo "=== ensure qsim cairo proof (extended-binary, Blake2sM31) $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "if [ -f $PROOF ]; then echo PROOF_PRESENT; ls -la $PROOF; else \
  cd $GTV; RUST_LOG=warn PROVING_UTILS_ROOT=$PU PROOF_FORMAT=extended-binary PROVER_PARAMS=$PARAMS \
  bash bin/iadd-sim-prove $FX 1 > /tmp/qsimregen.log 2>&1; echo REGEN_EXIT=\$?; ls -la $PROOF; fi" >>"$OUT" 2>&1

# Build the harness once (optimized).
echo "=== build iadd-recursion (target-cpu=native) $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GTV/iadd-recursion; RUSTFLAGS='-C target-cpu=native' \
  ~/.cargo/bin/cargo +$TC build --release > /tmp/qsimbuild.log 2>&1; echo BUILD_EXIT=\$?" >>"$OUT" 2>&1
$SSH "tail -3 /tmp/qsimbuild.log" >>"$OUT" 2>&1

# Sweep partitionings. POOL_THREADS=96 => 1 pool (sequential baseline).
for PT in 96 48 32 24; do
  echo "=== POOL_THREADS=$PT $(date -u) ===" >> "$OUT"
  timeout 3600 $SSH "cd $GTV/iadd-recursion; \
    RUSTFLAGS='-C target-cpu=native' RUST_LOG=warn RUST_BACKTRACE=1 POOL_THREADS=$PT \
    ./target/release/iadd-recursion $GTV/results/iadd_sim.proof.bin 4 > /tmp/qsim_pt_$PT.log 2>&1; echo RUN_EXIT=\$?" >>"$OUT" 2>&1
  $SSH "grep -E 'Pools:|config derived|leaves proved|folded to root|ROOT-VERIFY|pipeline complete|panicked' /tmp/qsim_pt_$PT.log" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
