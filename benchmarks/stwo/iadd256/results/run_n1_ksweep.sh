#!/usr/bin/env bash
# Background orchestration: gate_air + qsim N=1 K-sweeps on stwo-vm, then stop VM.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/n1_ksweep.txt
# NOTE: single-quoted so $HOME stays literal and expands on the REMOTE (home differs: laptop=/home/anat, vm=/home/anatg_starkware_co).
GA='$HOME/workspace/grover-tax/benchmarks/stwo/iadd256/native-air'
PU='$HOME/workspace/proving-utils'
FX_REMOTE='$HOME/workspace/grover-tax/fixtures'

echo "=== n1 K-sweep run $(date -u) ===" > "$OUT"

timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
echo "started; waiting for ssh" >> "$OUT"
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# Make sure the kK-n9024 fixtures are present on the VM.
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k{1,10,50,100,500,1000,2000}-n9024.json \
  stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1

# Refresh gate_air.rs (parallel trace-gen) and rebuild optimized.
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/native-air/src/bin/gate_air.rs \
  stwo-vm:~/workspace/grover-tax/benchmarks/stwo/iadd256/native-air/src/bin/ >>"$OUT" 2>&1
echo "=== rebuild gate_air (target-cpu=native) ===" >> "$OUT"
timeout 600 $SSH "cd $GA; RUSTFLAGS=\"-C target-cpu=native\" ~/.cargo/bin/cargo +nightly-2025-07-14 build --release --bin gate_air 2>&1 | tail -2" >>"$OUT" 2>&1

echo "=== gate_air N=1 K-sweep ===" >> "$OUT"
for K in 1 10 50 100 500 1000 2000; do
  timeout 300 $SSH "cd $GA; ./target/release/gate_air --fixture $FX_REMOTE/v0.3-iadd256-k${K}-n9024.json --samples 1 2>/dev/null | grep '^{'" 2>/dev/null \
   | python3 -c "import sys,json
d=json.load(sys.stdin)
print(f\"gate_air K={d['repetitions']:<5} rows={d['real_rows']:<10} padded=2^{d['log_rows']:<3} prove_s={d['prove_s']:<7} trace_gen_s={d.get('trace_gen_s','?')} verify_s={d['verify_s']}\")" >>"$OUT" 2>&1
done

echo "=== qsim N=1 K-sweep (feasible range) ===" >> "$OUT"
for K in 1 10 50 100; do
  timeout 600 $SSH "cd ~/workspace/grover-tax-v02; \
     RUST_LOG=info PROVING_UTILS_ROOT=$PU /usr/bin/time -v bash bin/iadd-sim-prove $FX_REMOTE/v0.3-iadd256-k${K}-n9024.json 1 > /tmp/qs.$K 2>&1; \
     steps=\$(grep -oE 'Num steps: [0-9]+' /tmp/qs.$K | head -1); \
     pc=\$(grep -E 'prove:prove_cairo:.*close time.busy' /tmp/qs.$K | grep -oE 'time.busy=[0-9.]+[a-z]+' | tail -1); \
     wall=\$(grep 'Elapsed' /tmp/qs.$K | grep -oE '[0-9:.]+\$'); \
     ok=\$(grep -c 'verified successfully' /tmp/qs.$K); \
     echo \"qsim K=$K  \$steps  prove_cairo=\$pc  wall=\$wall  verified=\$ok\"" >>"$OUT" 2>&1
done

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
