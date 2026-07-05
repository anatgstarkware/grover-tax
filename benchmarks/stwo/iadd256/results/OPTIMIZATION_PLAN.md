# gate_air single-machine optimization — config changelist + sweep plan

Goal: best single-machine time for the Tanuj benchmark (iadd256.kmx × 9024 shots, per k), with
configs that PASS the privacy-verifier-style tests.

## Hard invariants (every stage's PcsConfig must satisfy — privacy_circuit_verify/tests.rs)
- Security: `pow_bits + n_queries · log_blowup_factor >= 96` (CONJECTURED_SECURITY_BITS).
- `lifting_log_size = trace_log_size + log_blowup_factor`.
- `fold_step = 4`.
- pow_bits + n_queries are NOT free: canonical secure pairs per blowup (≈96 bits):
  - blowup 1 → n_queries 70, pow 26
  - blowup 2 → n_queries 35, pow 26
  - blowup 3 → n_queries 23, pow 27
  (matches CAIRO_PCS_CONFIG blowup3/23/27 + CIRCUIT_PCS_CONFIG blowup2/35/26; encoded in leaf.rs `leaf_pcs_config`.)
- shared-config padding size is NOT free: determined by the shard (leaf) size.

## Stages and their independent blowup knobs
1. BASE = gate_air STARK proof (the shard / "leaves"). blowup_base. INDEPENDENT.
2. RECURSION = leaf-wrapper proof + ALL multiverifier tree-node proofs. blowup_rec. ONE shared
   config (the node verifies leaf- and node-proofs under one `shared_config` → leaf-wrapper and tree
   blowups are COUPLED, cannot differ). (User's "leaf wrappers" + "tree" = this one knob.)
3. ROOT WRAPPER = prove_root_verification. blowup_root. INDEPENDENT (its own published proof;
   verifies the root node under the shared config).

## Optimization targets (user, 2026-06-22)
- SHARD SIZE: NOT max. Set so T(leaves + leaf-wrappers) ≈ T(tree + root wrapper). Knob = shots/shard.
- POOLS: separate best pool config for (leaves+wrappers) [depends on shard size], for the tree, and
  for the root wrapper. Define given the chosen shard size.
- NO const padding: shared config sized to the actual (secure) leaf, not a fixed floor.
- BLOWUP sweep: after shard+pools fixed, try blowup_base ∈ {1,2,3}, blowup_rec ∈ {1,2,3},
  blowup_root ∈ {1,2,3}; pick best (each changes pow/n_queries via the table → new config).

## CHANGELIST (what changes, how) — update as we go
- [F1 FOUNDATIONAL] BASE config: main.rs `PcsConfig::default()` (TOY, 13-bit, FAILS security) →
  secure `leaf_pcs_config(max_log_size, BASE_LOG_BLOWUP_FACTOR)`. Makes base 96-bit; leaf becomes
  realistic size (verifies n_queries≈70 not 3). STATUS: in progress.
- [F2] Add `BASE_LOG_BLOWUP_FACTOR` const (the base blowup knob).
- [F3] Confirm derive_aggregate_config tracks the real leaf size (no hardcoded floor) once base is
  secure; the fixed-point max(leaf,node) should now reflect the shard. Verify no literal const.
- [K1] recursion blowup = `LOG_BLOWUP_FACTOR` in the GATE_AIR_FOLD block (already a const) — sweep.
- [K2] root blowup = arg to `prove_root_verification` (already independent) — sweep.
- [P1] Replace the single PoolSet with per-phase pools: leaves+wrappers / tree / root.
- [S1] shots-per-shard knob to set shard size for the balance target.

## Sweep results table (fill in; time = single-machine, secure configs)
| attempt | shard(log) | blowup base/rec/root | pools L/T/R | leaves+wrap s | tree+root s | balanced? | notes |
|---|---|---|---|---|---|---|---|
| (toy baseline) | 2^25 | def/3/3 | 3x / 2x48 / — | leaf~1.7s (TOY nq=3) | — | n/a | insecure; leaf unrealistically small |
| F1 secure base | 2^25 | 1/3/3 | seq | base 2^25=7.67s; leaf~2.9s (4 in 11.5s) | node~3.7s (fold 11s/3); root 2.0s | n/a | SECURE 96-bit; base prove UNCHANGED vs toy (cost~blowup not nq); config target grew 2^20→2^21 (tracks real leaf); recursion ~2× toy |

## Measured building blocks (secure, k1000 fixture, 96-core)
- BASE prove (blowup1): ~2.5M real-rows/s (0.94s@2^22 … 7.67s@2^25), + trace_gen ~0.79M/s (3.6× prove). UNCHANGED by security.
- LEAF wrapper (blowup3): ~2.9s @ 2^25-shard (padded 2^21). Node ~3.7s. Root ~2.0s (trace 2^20).
- LEAF vs SHARD SIZE (secure, measured 2^22/2^24/2^25/2^27): **FLAT** — config target = 2^21 and
  leaf ~2.8s at ALL sizes. WHY: the shared padding target = max(leaf_verify_circuit, multiverifier
  NODE). The node verifies a blowup-3/nq-23 leaf PROOF (recursion config, shard-INDEPENDENT) → node
  ~2^21 dominates, and the leaf-verify circuit stays below it even at a 2^27 base (n_queries=70 is
  the only growth term, FRI layers add ~log). So leaf/node/recursion are ~size-independent here; the
  "const 2^21" is the node's REAL intrinsic size, not toy waste. (Leaf would only start growing if
  the base proof got big enough that verifying it exceeds the node ~2^21 — beyond 2^27 / higher nq.)
- BASE single-prove combined (trace_gen+prove) ~0.5M rows/s, ~flat 2^22-2^27 (2^27 trace_gen
  superlinear). T_base wall is concurrency-bound, NOT per-prove: see note below.
- BASE SATURATION (measured: K concurrent full base sessions = trace_gen+prove+verify, aggregate Mrows/s):
  - 2^24: K=1 0.44, K=2 0.78, K=3 0.93, K=4 1.07
  - 2^25: K=1 0.45, K=2 0.80, K=3 0.96, K=4 1.09
  - FINDINGS: (1) aggregate throughput is IDENTICAL for 2^24 vs 2^25 at each K → shard size doesn't
    change base throughput, only CONCURRENCY does. (2) STILL CLIMBING at K=4 (not saturated) → push
    K higher; smaller shards fit more in 732GB → can reach higher K → higher aggregate. (3) single
    prove (K=1) only ~0.45M/s — the machine is far from saturated by one prove (trace_gen/prove idle
    phases). 2^26 K=1..3 pending. So "pools given shard" = run as many concurrent base proves as
    memory allows; smaller shards may win on base throughput (but cost more recursion).
  - TODO: push K to 6-8 on 2^23/2^24 to find the saturation plateau.
- BASE-BLOWUP sweep (2^25, BASE_BLOWUP 1/2/3): base combined 37.6/46.3/62.6s; leaf 2.8/1.6/1.6s;
  target 2^21/2^20/2^20. **VERDICT: base blowup 1 (base prove dominates ∝~1.5×/step; the leaf+node
  saving at b2/3 never repays the base increase).** Per-shard: b1 ~45s vs b2 ~50s vs b3 ~67s. Higher
  base blowup shrinks the leaf (70→23 queries) but that's minor vs the base. KEEP base = 1.
- **DOMINANT COST = trace_gen (79% of base, 29.9 of 37.6s @ 2^25), ~blowup-independent.** The config
  knobs (shard/pool/blowup) tune the remaining ~21% + recursion; the big lever is trace_gen itself.

## REFRESHED base throughput (FAST interaction-gen, run_gate_air_satsweep2.sh, 2^24)
- K=1 0.78 / K=2 1.07 / K=3 1.33 / K=4 1.41 / K=5 1.50 / K=6 1.53 / K=7 1.59 / K=8 1.63 Mrows/s.
- vs old slow-code (0.44/0.78/0.93/1.09) → ~1.5x better aggregate, ~1.8x single-prove. Plateau ~1.6-1.7.
- Saturated base throughput ≈ **1.6 Mrows/s** (use for the curve).

## REFRESHED CURVE (single CPU machine, base@1.6Mrows/s + recursion), total_rows(k)=22.984M·k:
| k | base | +recursion | total | SP1 8×A100 | ratio |
|---|---|---|---|---|---|
| 1 | 14s | +~6s | ~20s | 18.5s | ~1.1× |
| 10 | 2.4min | +~6s | ~2.5min | 30.6s | ~5× |
| 100 | 24min | +~1min | ~25min | 2m37s | ~10× |
| 1000 | 4.0h | +~9min | ~4.2h | 23m7s | ~11× |
| 2000 | 8.0h | +~19min | ~8.3h | 45m53s | ~11× |
- vs PRE-optimization curve (16-17× behind): now ~11× behind at scale. The ~1.5× base-throughput win
  (interaction-gen parallelization) shifted the whole curve down. Recursion (leaf~2.9s, node~3.7s,
  N_shards≈0.686k, ~8× concurrency) stays ~4% of total. Still CPU vs 8 A100; GPU(~20× on commit+
  prove_ex, the remaining ~13s/shard) is the lever to actually beat SP1.

## STATUS of knobs
- base blowup = 1 (CONFIRMED best).
- shard ~2^25 (recursion-min; base throughput is shard-independent so this is fine; could go smaller
  if higher-K base concurrency wins, pending K-plateau).
- pools: base concurrency not saturated at K=4 (push to 6-8); recursion pools TBD.
- recursion blowup (leaf-wrap+tree) = 3, root = 3 — NOT yet swept (minor: recursion << base). Optional.

## GPU MEASURED CURVE (2026-06-23, a2-highgpu-1g A100 + Cascade Lake; cheap-measurement, NO rebase)
- t_gpu_pure (A100, NitrooZK wide_fib w=191, full prove): 2^20 .133s / 2^22 .480s / **2^24 1.83s**.
  Shard caps at 2^24 on 40GB (2^25 x191cols OOMs ~51GB). GPU_8g = 8x2^24/1.83 = **73 Mrows/s**.
- r_slice (CPU bucket, 12 vCPU Cascade Lake, 2^24): trace-gen 15.7s = 0.65 Mrows/s; +lifted-merkle
  -> **~0.50 Mrows/s** (range .37-.65). 0.65@12vCPU vs 0.79@96vCPU = memory-bound saturation CONFIRMED.
- eta (1x12c vs 2x6c probe) = **0.695** (8-job worse -> range [0.3, 0.695]).
- **a2-highgpu-8g PROJECTION (CPU-BOUND across all eta): vs SP1 8xA100 ~tied@k1; ~6x behind (best eta)
  to ~14x (worst) at k>=1000.** GPU 73 >> CPU-feed 1.2-2.8 Mrows/s -> 8 GPUs STARVE.
- **CONCLUSION: GPU port alone does NOT beat SP1 — trace-gen (memory-bound CPU) is the wall (as the
  pre-measurement model predicted). Lever = trace-gen throughput (sim->columns fuse / CUDA trace-gen),
  NOT GPU speed.** Full curve: results/extrapolate.py (run it) + EXTRAPOLATION_PLAN.md.

## Balance model (MY interpretation — confirm): run leaf-gen and tree-fold CONCURRENTLY on the one
## machine (partition cores into pools), STREAM leaves into the fold; makespan = max(leaf-gen-phase,
## tree-phase), minimized when balanced. Shard size + pool split are the knobs. (Needs a streaming
## fold; current recursive_aggregate_prove takes all leaves upfront → sequential.) If instead the
## phases are sequential, total = sum and bigger shards always win (base fixed) — so balance only
## makes sense under concurrent/pipelined execution.
