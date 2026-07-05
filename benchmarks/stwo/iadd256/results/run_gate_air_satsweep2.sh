#!/usr/bin/env bash
# Re-measure base saturation throughput with the FASTER base (parallel interaction-gen). Push K
# higher to find the plateau: 2^24 K=1..8, 2^25 K=1..5 (memory-capped). aggregate Mrows/s.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_satsweep2.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== base saturation v2 (fast interaction) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
$SSH 'free -g | head -2' >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_s2.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

$SSH "cd $GAL
  run() { S=\$1; MAXK=\$2; RR=\$((S*1000*2547));
    for K in \$(seq 1 \$MAXK); do
      t0=\$(date +%s)
      for i in \$(seq 1 \$K); do RUSTFLAGS='$RF' $BIN --fixture $FX --samples \$S >/tmp/s2_\${S}_\${i}.log 2>/dev/null & done
      wait; t1=\$(date +%s); W=\$((t1-t0)); TOT=\$((K*RR))
      echo \"SAT2 S=\$S K=\$K wall=\${W}s aggMrows_per_s=\$(python3 -c \"print(round(\$TOT/\$W/1e6,3))\")\"
    done
  }
  run 4 8
  run 8 5" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
