#!/usr/bin/env bash
# Bounded (<1h) measurement campaign on stwo-vm to anchor a k=8000/N=9024 cairo-leaf estimate.
# Measures: (1) marginal cairo steps/gate + steady-state cairo prove throughput, from a k=100
# qsim proof (vs the known k=1 = 2.05M steps); (2) recursion leaf/node throughput over that large
# leaf at POOL_THREADS=48 and 24. Everything else is extrapolated off-VM.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/qsim_cost_campaign.txt
PU='$HOME/workspace/proving-utils'
GTV='$HOME/workspace/grover-tax-v02'
FX1000='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
PARAMS='$HOME/workspace/grover-tax-v02/iadd-recursion/cairo_prover_params_auto.json'
K1000='$HOME/workspace/grover-tax-v02/results/iadd_sim_k1000.proof.bin'
TC=nightly-2026-01-15
RF="-C target-cpu=native"

echo "=== qsim cost campaign $(date -u) ===" > "$OUT"
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
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

# Build the cairo prover + the recursion harness (optimized).
echo "=== build stwo-run-and-prove + harness $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $PU; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release -p stwo-run-and-prove > /tmp/build_prove.log 2>&1; echo PROVE_BUILD=\$?" >>"$OUT" 2>&1
timeout 1500 $SSH "cd $GTV/iadd-recursion; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/build_harness.log 2>&1; echo HARNESS_BUILD=\$?" >>"$OUT" 2>&1

# (1) Generate the k=100 qsim cairo proof; capture step count + prove wall time.
echo "=== gen k=1000 cairo proof $(date -u) ===" >> "$OUT"
timeout 1500 $SSH "cd $GTV; rm -f results/iadd_sim.proof.bin; \
  S=\$(date +%s); RUST_LOG=info PROVING_UTILS_ROOT=$PU PROOF_FORMAT=extended-binary PROVER_PARAMS=$PARAMS \
  bash bin/iadd-sim-prove $FX1000 1 > /tmp/gen_k1000.log 2>&1; GEN=\$?; E=\$(date +%s); \
  echo GEN_EXIT=\$GEN CAIRO_GEN_SEC=\$((E-S)); \
  if [ -f results/iadd_sim.proof.bin ]; then cp results/iadd_sim.proof.bin $K1000; ls -la $K1000; fi" >>"$OUT" 2>&1
echo "--- k1000 gen: steps / prove-time / lifting ---" >> "$OUT"
$SSH "grep -iE 'num steps|n_steps|steps =' /tmp/gen_k1000.log | tail -3; \
  grep -iE 'prove_cairo.* close time.busy|lifting|Proving failed' /tmp/gen_k1000.log | tail -5; \
  grep -iE 'Log size [0-9]+:' /tmp/gen_k1000.log | sort -t: -k1 | tail -8" >>"$OUT" 2>&1

# (2) Recursion over the large leaf at two partitionings (N=4).
for PT in 48 24; do
  echo "=== recursion over k1000 leaf, POOL_THREADS=$PT, N=4 $(date -u) ===" >> "$OUT"
  timeout 1800 $SSH "cd $GTV/iadd-recursion; \
    RUSTFLAGS='$RF' RUST_LOG=warn RUST_BACKTRACE=1 POOL_THREADS=$PT \
    ./target/release/iadd-recursion $K1000 4 > /tmp/rec_k1000_pt$PT.log 2>&1; echo RUN_EXIT=\$?" >>"$OUT" 2>&1
  $SSH "grep -E 'Pools:|config derived|leaves proved|folded to root|ROOT-VERIFY|pipeline complete|panicked|error\\[' /tmp/rec_k1000_pt$PT.log" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
