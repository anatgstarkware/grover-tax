#!/usr/bin/env bash
# Background: qsim N=1 K-sweep on stwo-vm (feasible K range), then stop VM.
# Paths that must resolve on the REMOTE are single-quoted '$HOME/...' so they
# expand in the ssh-side shell (laptop home=/home/anat, vm=/home/anatg_starkware_co).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/qsim_ksweep.txt
PU='$HOME/workspace/proving-utils'
FX='$HOME/workspace/grover-tax/fixtures'

echo "=== qsim N=1 K-sweep $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k{1,10,50,100}-n9024.json \
  stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
$SSH "ls -la $PU/target/release/stwo-run-and-prove 2>&1 | tail -1" >>"$OUT" 2>&1

echo "=== qsim N=1 K-sweep (feasible range) ===" >> "$OUT"
for K in 1 10 50 100; do
  timeout 1200 $SSH "cd ~/workspace/grover-tax-v02; \
    RUST_LOG=info PROVING_UTILS_ROOT=$PU /usr/bin/time -v bash bin/iadd-sim-prove $FX/v0.3-iadd256-k${K}-n9024.json 1 > /tmp/qs.$K 2>&1; \
    steps=\$(grep -oE 'Num steps: [0-9]+' /tmp/qs.$K | head -1); \
    pc=\$(grep -E 'prove_cairo: stwo_cairo_prover::prover: close' /tmp/qs.$K | grep -oE 'time.busy=[0-9.]+[a-z]+' | head -1); \
    wall=\$(grep 'Elapsed' /tmp/qs.$K | grep -oE '[0-9:.]+\$'); \
    ok=\$(grep -c 'verified successfully' /tmp/qs.$K); \
    echo \"qsim K=$K  \$steps  prove_cairo=\$pc  wall=\$wall  verified=\$ok\"" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
