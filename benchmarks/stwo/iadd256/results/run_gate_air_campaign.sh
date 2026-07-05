#!/usr/bin/env bash
# High-confidence single-machine measurement campaign for the Tanuj curve.
# Measures: (A) base gate_air prove throughput vs shard size 2^22..2^26 (single all-core prove),
# (B) memory ceiling, (C) base-prove concurrency gain, (D) leaf/node cost at a real 2^25 shard.
# Uses the k1000-n9024 fixture; --samples S sets shard size (real_rows = S*1000*2547).
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_campaign.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== gate_air single-machine campaign $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"
$SSH 'nproc; free -g | head -2' >>"$OUT" 2>&1

rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git/ \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_cb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -2 /tmp/gal_cb.log" >>"$OUT" 2>&1

echo "=== (A) base throughput sweep (single all-core prove) $(date -u) ===" >> "$OUT"
for S in 1 2 4 8 16; do
  timeout 900 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FX --samples $S 2>/dev/null | grep gate-air-report" >>"$OUT" 2>&1
done

echo "=== (B) memory ceiling: samples=32 (2^27) $(date -u) ===" >> "$OUT"
timeout 1200 $SSH "cd $GAL; RUSTFLAGS='$RF' $BIN --fixture $FX --samples 32 2>/tmp/gal_s32.err | grep gate-air-report; echo EXIT=\$?; tail -2 /tmp/gal_s32.err" >>"$OUT" 2>&1

echo "=== (C) base concurrency: 1x vs 2x vs 3x concurrent samples=4 (2^24) $(date -u) ===" >> "$OUT"
$SSH "cd $GAL
  for K in 1 2 3; do
    t0=\$(date +%s.%N)
    pids=()
    for i in \$(seq 1 \$K); do RUSTFLAGS='$RF' $BIN --fixture $FX --samples 4 >/tmp/gal_c\$i.log 2>/dev/null & pids+=(\$!); done
    for p in \"\${pids[@]}\"; do wait \$p; done
    t1=\$(date +%s.%N)
    echo \"CONCURRENCY K=\$K wall=\$(echo \"\$t1 - \$t0\" | bc)s for \$K x 2^24 proves\"
  done" >>"$OUT" 2>&1

echo "=== (D) recursion at scale: GATE_AIR_FOLD=4, samples=8 (2^25 shard) $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' GATE_AIR_FOLD=4 $BIN --fixture $FX --samples 8 2>&1 | grep -iE 'gate-air:|config derived|leaves proved|folded|root verification|panicked|FAILED'" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
