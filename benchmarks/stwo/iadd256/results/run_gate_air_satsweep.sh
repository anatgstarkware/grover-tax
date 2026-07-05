#!/usr/bin/env bash
# Base SATURATION sweep: run K concurrent base gate_air proves (trace_gen+prove+verify, no fold) at
# shard sizes 2^24/2^25/2^26, K=1..4 (memory-capped), measure WALL → aggregate rows/s. Finds the
# machine's max base throughput and how it depends on (shard size, concurrency) = "pools given shard".
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_satsweep.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"

echo "=== base saturation sweep (concurrency x shard) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
$SSH 'nproc; free -g | head -2' >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_ss.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

# real_rows = S*1000*2547 : S=4 ->10.188M (2^24), S=8 ->20.376M (2^25), S=16 ->40.752M (2^26)
$SSH "cd $GAL
  run() { # <S> <maxK>
    S=\$1; MAXK=\$2; RR=\$((S*1000*2547))
    for K in \$(seq 1 \$MAXK); do
      t0=\$(date +%s)
      for i in \$(seq 1 \$K); do RUSTFLAGS='$RF' $BIN --fixture $FX --samples \$S >/tmp/ss_\${S}_\${i}.log 2>/dev/null & done
      wait
      t1=\$(date +%s); W=\$((t1-t0)); TOT=\$((K*RR))
      ok=\$(grep -l 'proved\":true' /tmp/ss_\${S}_*.log 2>/dev/null | wc -l)
      echo \"SAT S=\$S(2^\$(python3 -c \"import math;print(int(math.log2(\$RR))+1)\")) K=\$K wall=\${W}s rows=\$TOT ok=\$ok aggMrows_per_s=\$(python3 -c \"print(round(\$TOT/\$W/1e6,3) if \$W>0 else 0)\")\"
    done
  }
  run 4 4
  run 8 4
  run 16 3" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
