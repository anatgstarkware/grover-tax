#!/usr/bin/env bash
# Measure the GPU part on a2-highgpu-1g (1x A100-40GB, 12 vCPU) to feed extrapolate.py.
# TEMPLATE: fill HOST/USER once the box is provisioned AND the obelyzk-ported gate-air-leaf
# (GpuBackend, lifted channel) builds. Until then this documents exactly what to capture.
#
# Produces, per candidate shard size:
#   (1) t_gpu_pure  = warm pure-A100 per-shard time  -> extrapolate.py T_GPU_PURE
#   (2) GPU/CPU phase split  (so t_gpu_pure EXCLUDES the CPU lifted-Merkle build)
#   (3) a 1-GPU concurrency check (does a 2nd concurrent prove help? = are 12 vCPU keeping the
#       single A100 fed, or is even one GPU CPU-starved -> early signal of the 8g CPU wall)
set -uo pipefail

HOST=""                 # <-- a2-highgpu-1g external IP / hostname (set when provisioned)
USER="${USER:-ubuntu}"  # <-- GCP image default user
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 ${USER}@${HOST}"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gpu_1g.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FX='$HOME/workspace/grover-tax/fixtures/v0.3-iadd256-k1000-n9024.json'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
BIN="./target/release/gate-air-leaf"
# ASSUMPTION (wire during the obelyzk port): GATE_AIR_BACKEND=gpu selects GpuBackend (lifted channel);
# the binary prints per-phase [phase] timers incl a GPU-vs-CPU breakdown of each commit
# (NTT/eval on GPU vs lifted-Merkle build on CPU). WARMUP=1 discards the cold CUDA-ctx/JIT run.

if [ -z "$HOST" ]; then echo "set HOST (a2-highgpu-1g) first"; exit 1; fi
echo "=== GPU 1g measurement $(date -u) ===" > "$OUT"
$SSH 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv; nproc; free -g | head -2' >>"$OUT" 2>&1

# sync source + fixture (assumes repo already cloned + deps on the box from the port setup)
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ ${USER}@${HOST}:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
timeout 2400 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release --features gpu > /tmp/gal_gpu.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1

# (1)+(2): warm per-shard GPU prove + phase split, at shard sizes that fit 40GB (2^22, 2^24).
#   samples S s.t. S*1000*2547 ~ 2^shardlog: 2^22~1646, 2^24~6585 (k1000 fixture).
$SSH "cd $GAL
  for S in 1646 6585; do
    echo \">>> shard ~ S=\$S samples (rows~\$((S*1000*2547)))\"
    # 1 cold (discard) + 3 warm; report warm phases + total
    for r in 0 1 2 3; do
      RUSTFLAGS='$RF' GATE_AIR_BACKEND=gpu $BIN --fixture $FX --samples \$S 2>&1 \
        | grep -iE '\[phase\]|real_rows=|prove total|gpu |merkle|commit' | sed \"s/^/run\$r: /\"
    done
  done" >>"$OUT" 2>&1

# (3): 1-GPU concurrency check at the smaller shard (does a 2nd concurrent prove raise throughput?
#   if yes -> 12 vCPU under-feed the A100; foreshadows the 8g CPU-feed wall).
$SSH "cd $GAL; S=1646; RR=\$((S*1000*2547))
  for K in 1 2; do
    t0=\$(date +%s)
    for i in \$(seq 1 \$K); do RUSTFLAGS='$RF' GATE_AIR_BACKEND=gpu $BIN --fixture $FX --samples \$S >/tmp/g_\$i.log 2>&1 & done
    wait; t1=\$(date +%s); W=\$((t1-t0)); TOT=\$((K*RR))
    echo \"GPU1g K=\$K wall=\${W}s aggMrows_per_s=\$(python3 -c \"print(round(\$TOT/\$W/1e6,3))\")\"
  done" >>"$OUT" 2>&1

# (4) CPU-BUCKET rate on the 1g = REAL Cascade Lake (no microarch discount). This feeds extrapolate.py
#   R_SLICE. CPU bucket = trace-gen + lifted-Merkle ONLY (no prove_ex). Best via a CPU-only build with
#   a --commit-only flag (skip prove_ex); until that flag exists, sum the CPU phases (sim + preprocessed
#   + main_trace + interaction + the lifted-Merkle part of the tree commits) from the [phase] timers of
#   a CPU-backend run, and divide rows by that.
$SSH "cd $GAL
  RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_cpu.log 2>&1; echo CPU_BUILD=\$?
  echo '>>> CPU-bucket phase timers (Cascade Lake, 12 vCPU), shard 2^24 (S=6585):'
  RUSTFLAGS='$RF' $BIN --fixture $FX --samples 6585 2>&1 | grep -iE '\[phase\]|real_rows='" >>"$OUT" 2>&1

# (5) CONTENTION PROBE (bounds eta without a 2g): at fixed total cores, does splitting into 2 concurrent
#   half-core buckets keep scaling, or saturate the bus? 1x(12 vCPU) vs 2x(6 vCPU). ratio < 2 => the
#   Cascade-Lake bus saturates => eta_high should be pulled below 1.0 for the 8-slice 8g case.
$SSH "cd $GAL; S=3293; RR=\$((S*1000*2547))   # ~2^23 so 2 fit in 85GB
  echo '>>> contention probe (eta bound): 1x full-cores vs 2x half-cores'
  t0=\$(date +%s); RUSTFLAGS='$RF' $BIN --fixture $FX --samples \$S >/tmp/c1.log 2>&1; t1=\$(date +%s)
  echo \"PROBE K=1 wall=\$((t1-t0))s Mrows/s=\$(python3 -c \"print(round(\$RR/(\$t1-\$t0)/1e6,3))\")\"
  t0=\$(date +%s)
  for i in 1 2; do RUSTFLAGS='$RF' taskset -c \$(( (i-1)*6 ))-\$(( (i-1)*6+5 )) $BIN --fixture $FX --samples \$S >/tmp/c2_\$i.log 2>&1 & done
  wait; t1=\$(date +%s); echo \"PROBE K=2x6c wall=\$((t1-t0))s aggMrows/s=\$(python3 -c \"print(round(2*\$RR/(\$t1-\$t0)/1e6,3))\")\"" >>"$OUT" 2>&1

echo "=== DONE $(date -u) ===" >> "$OUT"
echo "NEXT in extrapolate.py:" >> "$OUT"
echo "  T_GPU_PURE = warm GPU-only phase sum (NTT+FRI+quotients+composition, EXCLUDING CPU Merkle); GPU_MEASURED=True" >> "$OUT"
echo "  R_SLICE    = rows / CPU-bucket-phase-sum from (4); CPU_MEASURED=True" >> "$OUT"
echo "  ETA_HIGH   = (probe K=2x6c agg)/(2 x probe K=1-on-6c) clamped <=1; ETA_LOW ~ stwo-vm shape (~0.3)" >> "$OUT"
