#!/usr/bin/env bash
# M3 validation: prove the gate_air verification circuit as a foldable leaf and fold N of them into
# a multiverifier-tree root, on stwo-vm. Smallest case (k4-n16, samples=1), N=4 leaves.
set -uo pipefail
ZONE=us-central1-a; PROJ=starkware-dev; VM=stwo-development-server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 stwo-vm"
OUT=/home/anat/workspace/grover-tax/benchmarks/stwo/iadd256/results/gate_air_fold.txt
GAL='$HOME/workspace/grover-tax-v02/gate-air-leaf'
FXDIR='$HOME/workspace/grover-tax/fixtures'
TC=nightly-2026-01-15
RF="-C target-cpu=native"
N=${1:-4}

echo "=== gate_air multiverifier fold (M3) $(date -u) ===" > "$OUT"
timeout 200 gcloud compute instances start $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
for i in $(seq 1 40); do timeout 25 $SSH 'echo UP' 2>/dev/null | grep -q UP && break; sleep 8; done
echo "ssh up $(date -u)" >> "$OUT"

# gate-air-leaf path-deps proving-utils' recursive_aggregate (workspace inheritance) — sync both.
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ \
  /home/anat/workspace/grover-tax-v02/gate-air-leaf/ stwo-vm:~/workspace/grover-tax-v02/gate-air-leaf/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" --exclude target/ --exclude .git/ \
  /home/anat/workspace/proving-utils/ stwo-vm:~/workspace/proving-utils/ >>"$OUT" 2>&1
rsync -az -e "ssh -o BatchMode=yes" \
  /home/anat/workspace/grover-tax/fixtures/v0.3-iadd256-k4-n16.json stwo-vm:~/workspace/grover-tax/fixtures/ >>"$OUT" 2>&1
echo "rsynced $(date -u)" >> "$OUT"

echo "=== build $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' ~/.cargo/bin/cargo +$TC build --release > /tmp/gal_fb.log 2>&1; echo BUILD=\$?" >>"$OUT" 2>&1
$SSH "tail -3 /tmp/gal_fb.log" >>"$OUT" 2>&1

echo "=== run k4-n16 samples=1, GATE_AIR_FOLD=$N $(date -u) ===" >> "$OUT"
timeout 1800 $SSH "cd $GAL; RUSTFLAGS='$RF' RUST_BACKTRACE=1 GATE_AIR_FOLD=$N \
  ./target/release/gate-air-leaf --fixture $FXDIR/v0.3-iadd256-k4-n16.json --samples 1 > /tmp/gal_f.log 2>&1; echo RUN=\$?" >>"$OUT" 2>&1
$SSH "grep -iE 'gate-air:|leaves proved|folded|config derived|fold OK|panicked|FAILED|error\[|assertion' /tmp/gal_f.log | head -40; echo '--- tail ---'; tail -40 /tmp/gal_f.log" >>"$OUT" 2>&1

echo "=== stopping VM $(date -u) ===" >> "$OUT"
timeout 200 gcloud compute instances stop $VM --zone=$ZONE --project=$PROJ >>"$OUT" 2>&1
timeout 40 gcloud compute instances describe $VM --zone=$ZONE --project=$PROJ --format="value(status)" >>"$OUT" 2>&1
echo "=== DONE $(date -u) ===" >> "$OUT"
