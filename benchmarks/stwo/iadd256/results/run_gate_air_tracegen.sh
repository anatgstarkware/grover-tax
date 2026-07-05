#!/usr/bin/env bash
# Re-measure trace_gen + prove at REALISTIC shot parallelism (campaign 1 under-parallelized trace_gen
# by using only 8-32 shots). Use k1-n9024 fixture with high --samples (thousands of shots), so trace
# gen fans over all 96 cores like the real benchmark. real_rows = samples * 1 * 2547.
#   samples=2256 -> 2^23 ; 4512 -> 2^24 ; 9024 -> 2^25 (the full k=1 benchmark point).
# Also k10 at samples=1316 -> ~2^25 (1316-way) to check trace_gen per-row vs reps.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_tracegen.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== gate_air trace_gen-at-scale re-measure $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1-n9024.json \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k10-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_tb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -2 /tmp/gal_tb.log" >>"$OUT" 2>&1

echo "=== trace_gen+prove at full shot parallelism (k1-n9024) $(date -u) ===" >> "$OUT"
for S in 2256 4512 9024; do
  timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FXDIR/v0.3-iadd256-k1-n9024.json --samples $S 2>/dev/null | grep gate-air-report" >>"$OUT" 2>&1
done
echo "--- k10 (reps=10) at ~2^25, 1316 shots ---" >> "$OUT"
timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FXDIR/v0.3-iadd256-k10-n9024.json --samples 1316 2>/dev/null | grep gate-air-report" >>"$OUT" 2>&1

echo "=== base concurrency (fixed timing): 1x/2x/3x concurrent 2^24 (k1 s=4512) $(date -u) ===" >> "$OUT"
$SSH "cd $GAL
  for K in 1 2 3; do
    t0=\$(date +%s)
    for i in \$(seq 1 \$K); do RUSTFLAGS='$RF' $BIN --fixture $FXDIR/v0.3-iadd256-k1-n9024.json --samples 4512 >/tmp/gal_cc\$i.log 2>/dev/null & done
    wait
    t1=\$(date +%s)
    echo \"CONCURRENCY K=\$K wall=\$((t1-t0))s for \$K x 2^24 proves (=> per-prove \$(( (t1-t0) ))s, throughput x\$K)\"
  done" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
