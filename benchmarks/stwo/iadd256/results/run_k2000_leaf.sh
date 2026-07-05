#!/usr/bin/env bash
# Measure the k=2000 cairo-verifier LEAF (and node) cost to replace the extrapolated multiplier.
# Generates a k=2000 qsim cairo proof with EXPLICIT lifting (fixes the null-lifting panic), then
# runs the recursion harness N=4 at POOL_THREADS=96 (per-leaf @ full machine, vs k=1's 10.2s) and
# POOL_THREADS=48 (partition throughput). Stops the VM. <1h budget.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/k2000_leaf.txt
PU='$HOME/workspace/proving-utils'
GTV='$HOME/workspace/grover-tax-v02'
FX2000='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k2000-n9024.json'
PARAMS='$HOME/workspace/grover-tax-v02/iadd-recursion/cairo_prover_params_k2000.json'
K2000='$HOME/workspace/grover-tax-v02/results/iadd_sim_k2000.proof.bin'
TC=nightly-2026-01-15
RF="-C target-cpu=native"

echo "=== k2000 leaf measurement $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/iadd-recursion/ stwo-vm:~/workspace/grover-tax-v02/iadd-recursion/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax-v02/bin/iadd-sim-prove stwo-vm:~/workspace/grover-tax-v02/bin/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k2000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $PU; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release -p stwo-run-and-prove > /tmp/bp.log 2>&1; echo PROVE_BUILD=\$?" >>"$OUT" 2>&1
timeout 1500 $SSH "cd $GTV/iadd-recursion; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/bh.log 2>&1; echo HARNESS_BUILD=\$?" >>"$OUT" 2>&1

echo "=== gen k=2000 cairo proof (explicit lifting=28) $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $GTV; rm -f results/iadd_sim.proof.bin; \
  S=\$(date +%s); RUST_LOG=info PROVING_UTILS_ROOT=$PU PROOF_FORMAT=extended-binary PROVER_PARAMS=$PARAMS \
  bash bin/iadd-sim-prove $FX2000 1 > /tmp/gen_k2000.log 2>&1; GEN=\$?; E=\$(date +%s); \
  echo GEN_EXIT=\$GEN CAIRO_GEN_SEC=\$((E-S)); \
  if [ -f results/iadd_sim.proof.bin ]; then cp results/iadd_sim.proof.bin $K2000; ls -la $K2000; fi" >>"$OUT" 2>&1
$SSH "grep -iE 'num steps' /tmp/gen_k2000.log | tail -1; grep -iE 'lifting|Proving failed' /tmp/gen_k2000.log | tail -3" >>"$OUT" 2>&1

# Recursion harness over the k2000 leaf: 96 (per-leaf baseline) then 48 (partition).
for PT in 96 48; do
  echo "=== harness over k2000 leaf, POOL_THREADS=$PT, N=4 $(date -u) ===" >> "$OUT"
  timeout 2400 $SSH "cd $GTV/iadd-recursion; \
    RUSTFLAGS='$RF' RUST_LOG=warn RUST_BACKTRACE=1 POOL_THREADS=$PT \
    ./target/release/iadd-recursion $K2000 4 > /tmp/rec_k2000_pt$PT.log 2>&1; echo RUN_EXIT=\$?" >>"$OUT" 2>&1
  $SSH "grep -E 'Pools:|config derived|leaves proved|folded to root|ROOT-VERIFY|pipeline complete|panicked|error\\[' /tmp/rec_k2000_pt$PT.log" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
