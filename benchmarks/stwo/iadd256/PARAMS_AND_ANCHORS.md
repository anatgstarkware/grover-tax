# gate_air pipeline — measured anchors & configuration parameters

Reference sheet for the final result (single a2-highgpu-8g vs SP1 8×A100 on the iadd256
9024-shot × k-rep benchmark). Stack = migration (stwo 5ea05973, 8-word HashValue) + decoupled
k-ary fold (k=8) + H_P program commitment + L3 word-path Merkle kernel.

Source of truth: `results/extrapolate_8g.py`, `results/plot_k_sweep.py` (anchors), and the leaf /
recursion code (`grover-tax-v02/gate-air-leaf/src/{leaf.rs,main.rs}`,
`proving-utils/crates/recursive_aggregate/src/lib.rs`).

---

## 1. Measured anchors (2^25 final stack)

### Timing anchors (the overlap-model inputs)

| anchor | value | what it is | measured on |
|---|---|---|---|
| `t_base_fold` | **9.53 s/shard** | fold-amortized per-shard GPU base @2^25, with L3 | A100 (anat-ganor) |
| `t_precompute` | **4.64 s** | one-time shard-invariant precompute (tree0 built once, `Borrowed`); paid before the shard loop, matters only at tiny N | A100 |
| `t_leaf` | **17.30 s** | leaf wrap, decoupled k=8 (leaf padded to its own ~2^21/lift24 size, not ballooned) | c4 CPU-VM (a2-8g proxy) |
| `t_node1` | **13.4 s** | one level-1 node fold (verifies 8 leaves) | c4 CPU-VM |
| `t_node` (effective) | **1.68 s** | per-shard node term = t_node1 / 8; level-≥2 negligible | derived |
| `P_8g` | **16** | recursion pool count (K=16, throughput optimum, mem-BW saturated) | stwo-vm 96-vCPU sweep |
| `TAIL` | **2.07 s** | root-verification / in-circuit unpacker (top-of-tree + root); +0.0021·N negligible | c4 CPU-VM |

Derived combined: per-shard recursion `t_leaf + t_node = 18.98 s` = **0.681×** the coupled-k=2
baseline (27.86 s) — the decoupling win. Base single-shot was 14.53 s (precompute 4.64 amortizes
to ~0/shard in the fold).

### Correctness anchors (byte-identity, all on the box)

- base + H_P GPU byte-identity: **ALL PASS**; base fingerprint `9638…` (re-baselines pre-H_P `661decf9`)
- decoupled fold: fingerprint **ON == OFF**, **streaming == sequential**, verify sanity check **PASS**, R1 ≠ R2

### Confidence caveat

Measurement is **split across two machines**: base anchors on a real A100 (target-class GPU),
recursion anchors (`t_leaf`, `t_node1`, `TAIL`) on the **c4 CPU-VM used as an a2-8g CPU proxy**
(cross-calibration, not the actual a2-8g host — the A100 host CPU measured t_leaf = 8.7 s; 17.30 is
the conservative c4 proxy). This is why the ~1% base-vs-rec margin at large k sits inside the
measurement uncertainty; "balanced, marginally base-bound" is the honest read. A single a2-8g
end-to-end run would tighten it.

---

## 2. Base-bound vs recursion-bound

**Base-bound at every k on the sweep**, but the character changes sharply:

| regime | k | margin (base over rec) | reading |
|---|---|---|---|
| small k | 1–50 | 92% → 21% | strongly base-bound — recursion trivial (1–35 shards), GPU base per-shard cost is ~the whole wall |
| large k | 500–8000 | 1.8% → 0.5% | balanced, tipping marginally to base |

At k=2000: base_wall 1644 s vs rec_wall 1625 s (~1.1%). **Implication:** the marginal resource is
the base (GPU) path — every second off `t_base_fold` comes straight off the wall; recursion has ~1%
slack at large k before it would bind, so it's no longer where to spend effort. (The ~1% large-k
margin is inside the cross-calibration uncertainty above.)

---

## 3. Configuration parameters

### Machine (fixed — the machine is given)

| param | value |
|---|---|
| target host | a2-highgpu-8g = 8× A100-40GB, 96 vCPU, 680 GB RAM |
| structure | exactly 8× a2-1g (1 A100 / 12 vCPU / 85 GB per unit) |

### Shard & trace

| param | value | where |
|---|---|---|
| shard size | 2^25 rows | `SHARD_LOG=25` |
| gate_air trace | 22 columns (per-qubit chain-lookup memory AIR) | — |
| hash output width | N_RESERVED = 8 words (non-reduced Blake2s, post-#1425) | `leaf.rs` |
| workload | 22.984e6 rows/rep → N = ⌈22.984e6·k / 2^25⌉ shards | model |

### Base proof (the shard / "leaves") — GPU

| param | value | note |
|---|---|---|
| `BASE_LOG_BLOWUP_FACTOR` | 1 | `main.rs` |
| n_queries | 70 | from blowup=1 |
| pow_bits | 26 | → security = 26 + 70·1 = 96 bit |
| fold_step | 4 | |
| log_last_layer_degree_bound | 0 | |
| lifting (LDE) | 2^25 + 1 = 2^26 | |
| backend | CudaBackend (device-resident commit + prove_ex) | |

### Recursion (leaf wrap + node folds) — CPU

| param | value | note |
|---|---|---|
| `LOG_BLOWUP_FACTOR` | 3 | `main.rs` |
| n_queries | 23 | from blowup=3 |
| pow_bits | 27 | → security = 27 + 23·3 = 96 bit |
| fold_step | 4 | |
| `FOLD_ARITY` (k) | 8 | every INTERNAL node; `lib.rs` |
| root arity m | derived per k (see table below); the single root fold's terminal size | `root_arity(N)` |
| leaf trace / lifting | ~2^21 → 2^24 ("lift24", `leaf_pcs_config`) | |
| node trace / lifting | ~2^22 → 2^25 ("lift25", `node_pcs_config`) | decoupled from leaf |
| trusted roots | R1 (level-1 nodes verify k leaves), R2 (level-≥2 nodes verify k nodes, self-verifying fixed point) | 2 roots, selected by `FoldTask.height` |

### Concrete fold topology per benchmark point

Every internal node is arity 8; the root fold's arity `m = root_arity(N)` and the tree shape are a
deterministic function of the shard count N alone (never prover-chosen).

| k | N (shards) | root arity m | tree levels | internal nodes |
|---|---|---|---|---|
| 1 | 1 | 1 (no root fold) | 0 | 0 |
| 10 | 7 | 7 | 1 | 1 |
| 50 | 35 | 7 | 2 | 5 |
| 100 | 69 | 6 | 3 | 10 |
| 500 | 343 | 7 | 3 | 49 |
| 1000 | 685 | 6 | 4 | 98 |
| 2000 | 1370 | 5 | 4 | 196 |
| 4000 | 2740 | 3 | 5 | 392 |
| 8000 | 5480 | 6 | 5 | 783 |

### Concurrency / balances (the tuned knobs)

| param | value | source |
|---|---|---|
| recursion pools `P_8g` | 16 (K=16 pools × T=6 threads = 96 vCPU) | stwo-vm sweep — throughput optimum, mem-BW saturated |
| `P_1g` | 2 (2×6 on 12 vCPU), scaled ×8 | Balance A |
| GPU concurrency | 8 (⌈N/8⌉ base waves) | 8 A100s, firm ×8 |
| overlap model | `T(k) = max(base_wall, rec_wall) + tail` | base(GPU) ∥ recursion(CPU) |
| contention η | folded into measured t_leaf/t_node (K=16-contended values) — not double-counted | sweep |

### Program commitment (H_P) & hash

| param | value |
|---|---|
| tags | `TAG_PROGRAM` (internal, cancels main's demand), `TAG_PROGRAM_PUB = 6` (dangling public) |
| per-shard hash | H_P = blake2s(program_table ‖ nonce); output H_i = blake2s(H_P ‖ x ‖ y) |
| Merkle hasher | `Blake2sMerkleHasher` = `Blake2sHasherGeneric<false>` (standard, non-reduced 8-word) |
| Fiat–Shamir channel | `Blake2sM31MerkleChannel` (M31 challenge/query derivation, unreduced root) |

### The invariant tying it together

security = pow_bits + n_queries·log_blowup = **96 bit** at every blowup setting — base runs
blowup=1 (70 queries / pow 26), recursion runs blowup=3 (23 queries / pow 27); both hit 96. The
blowup↔n_queries trade is the free cost knob at fixed 96-bit. The leaf↔node lifting decoupling
(lift24 vs lift25) pins t_leaf independent of FOLD_ARITY — the single change that took the k-ary
fold from a net loss to the 0.681× recursion win.
