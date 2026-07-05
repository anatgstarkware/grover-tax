# iadd recursion plan — gate-level, secret circuit

## Load-bearing constraint
Prove **gate-level** execution of a reversible {NOP,NOT,CNOT,TOFFOLI} circuit, where the
**circuit is SECRET (must be hidden)**. ⇒ the fast arithmetic native-AIR (~140s for the
whole K=8000/N=9024, ~75× faster than SP1) is **ruled out** — it proves the addition
result, not that a secret gate circuit ran. Gate-level (gate_air, program=witness) is
mandatory; hiding requires zk-blinding (a plain STARK leaks the trace → the circuit).

## Committed pipeline

| component | what | status | risk |
|---|---|---|---|
| gate_air base shards | gate-level, program=witness; K=8000, N=1/shard → 2^25; 1 shard/shot | **done** (proves+verifies; program-consistency binds) | low |
| read-write Memory AirFn | qubit state as cheap witness-indexed read/write; subsumes dynamic addressing + chain-lookup | to build | medium (air_infra expressiveness) |
| **leaf adapter** | gate_air → circuit-verifier the multiverifier consumes (via air_infra `AirFn` codegen) | to build | **highest** |
| multiverifier 2-to-1 tree | verify 2 proofs → hash [rootA,outA,rootB,outB] → 2-QM31 out; self-verifies → any depth | **exists** (stwo-circuits PR 506 leaf / 507 self-verify) | low (reuse) |
| zk-blinding (hiding) | `circuit_common::finalize::add_zk_blinding(context, seed, n_queries)` — general, context-level | **exists** (proving-utils `privacy_prove`) | medium (wire to multiverifier ctx) |
| unpacker | final node: open hash-tree root → claimed {y_i} + Blake2s program commitment | to build | low–medium |
| GPU port | NitrooZK-stwo CUDA fork, ~20× on Blake2s/M31 (wide-fib log23: 11s→0.49s, RTX4090) | to build | medium (supports complex FrameworkComponent+LogUp?) |

## Key facts learned
- **Per-node cost:** multiverifier circuit is fixed `trace_log_size = 2^21`, blowup 8 (PR506/507 tests); prove-time still unmeasured (the test only does *assignment* ~0.33s; `prove_circuit_assignment` ≠ full FRI prove — that's `prove_circuit_with_precompute`/`prove_ex`). Est. ~few s/node.
- **Tree economics:** 9024 base (2^25, ~11s CPU each) + ~9023 nodes (~2^21). Per-node < base ⇒ recursion is ~30–40% overhead, not a blow-up. CPU wall: ~37h naive / ~10–12h concurrent-1-VM / ~2–4h on 8 VMs. **GPU (~20×): ~15–20 min on 8 GPUs ⇒ competitive with SP1's ~3h/8×A100.**
- **No codegen for hand-written FrameworkEval:** `CircuitEval` mirrors are generated from `AirFn` in `stwo-air-infra/air_code_gen` (Opcode/Gate/ChainRound/Memory trace types + `add_lookup_term`/`range_check`). `Felt252IdMemory` is read-only/Cairo-specific (29-bit addr, felt252, id-compressed) — does NOT fit our read-write 512×1-bit qubit state ⇒ custom Memory AirFn.
- **DSL indexed access** (`select_by_index`) is a linear mux (~n ops) — too expensive for witness-indexed qubit reads; cheap access needs the memory argument (air_infra).
- **Memory-sort check (at N=1):** permutation (multiset) + range-checked consecutive-key diffs (9-bit addr; ts ≤ 2^25 fits M31 → single range-check, no multi-limb) + consistency (read carries / write sets / boundary to x,y).

## STEP 1 (do first): de-risk recursion + hiding + unpacker on EXISTING pieces
**Goal:** prove the full chain works *and hides*, using existing code (cairo leaves), before the gate_air port.
**Experiment:** small multiverifier tree over N=4 cairo leaf-proofs (reuse `test_data/circuit_multiverifier/proof.bin`): 4 leaves → 2 leaf-multiverifiers (PR506) → 1 internal multiverifier (PR507) → final. Apply `add_zk_blinding` to node context(s). Verify final + unpacker check (output == hash-tree root over leaves' (root,out); placeholder commitment).
**New code:** (a) tree harness = loop `build_multiverifier_circuit` over levels (node exists; the level-loop is new); (b) unpacker = recompute expected root from leaf outputs, assert == final output; (c) wire `add_zk_blinding` into the multiverifier prove.
**Reuse:** multiverifier (506/507), `add_zk_blinding` (general), `prove_circuit_with_precompute`, existing cairo test proof as leaves.
**Validation:** tree composes to depth ≥2 (507 shows self-verify); final verifies; output == expected root; zk-blinding applied (hiding rests on `add_zk_blinding`'s ZK).
**Open Qs:** depth>2 composition w/ consistent SharedConfig; `add_zk_blinding` on multiverifier ctx composes; unpacker root-opening shape.

## Leaf output encoding (PINNED 2026-06-19)
Each gate_air leaf (one shot) exposes exactly **2 QM31** to the multiverifier (`N_RESERVED=2`,
≈ one Blake2s digest). gate_air state = 512 qubits = 32×16-bit limbs; per leaf: input `x_i`
(512-bit), output `y_i` (512-bit), hidden **witness** program table `P` (n_gates, run K times;
program-consistency LogUp already binds all K reps to `P`). Encoding:

```
H_P  = Blake2s( program_table ‖ nonce )     // hiding program commitment (one shared nonce, all leaves)
H_i  = Blake2s( H_P ‖ x_i ‖ y_i )           // the leaf's 2-QM31 output
```

- **Program-equality across leaves is enforced for free:** the wrapper/verifier recomputes every
  `H_i` with ONE `H_P`; a leaf that ran a different program has a different real `H_P`, so its
  committed `H_i` won't match → unsatisfiable. No separate "all program-commitments equal" gadget.
- **`pp_root` equality** (same AIR) is a *separate*, cheap check: wrapper uses one trusted
  `leaf_pp_root` constant in the opener. NOTE: equal `pp_root` does NOT imply equal program (program
  is witness; pp_root is program-independent) — that's why `H_P` is needed.
- **Inputs/outputs pinned:** `x_i, y_i` are in the `H_i` preimage.
- gate_air already binds the hashed program table to execution (program-consistency LogUp), so
  `H_P` commits to the *executed* program.

### Wrapper (final) — public outputs = `[H_P, IO_root]` (4 QM31), then zk-blinded
1. verify root MV proof → root output `R`.
2. guess `(H_P, {x_i, y_i})`; derive each `H_i = Blake2s(H_P ‖ x_i ‖ y_i)` in-circuit.
3. opener: tree-hash `{H_i}` with `leaf_pp_root`/`node_pp` constants → computed root; `eq` to `R`.
4. `IO_root = Blake-Merkle over {(x_i, y_i)}` in-circuit; expose `[H_P, IO_root]`.
5. `add_zk_blinding` (hiding).
Verifier is later given `{(x_i,y_i)}` + checks against `IO_root`. **Cost note:** wrapper is **O(N)**
in-circuit blakes (derive H_i + tree + IO_root) — for N=9024 it may dominate; an optimization is to
fold the IO commitment into the recursion (each node combines children) making the wrapper O(1).
Decisions: hiding nonce = YES; publish `H_P` = YES; IO = commit (root), not reveal-all.

### Distinct-shard fold BLOCKER + fix (diagnosed 2026-06-30) — make leaf per-shard data WITNESS, not constants
First end-to-end test of folding genuinely-distinct shards failed (precompute mismatch `leaf.rs:215`;
node binding eq `1 != 0`), in BOTH sequential and streamed paths (so not the pipeline).

**Diagnosis = Verdict B (measured, laptop, no box, no full prove):**
- (A) the **base** `preprocessed_root` (gate_air shard proof tree0) is **byte-identical** across equal
  shards (`real_rows/padded_rows/log` all equal; `pc_in_prog = pc % n_gates` is program-shape-only).
  So it is NOT a shape/padding artifact — canonical padding fixes nothing.
- (b1) the **leaf_prover** `leaf_preprocessed_root` **changes when only the boundary (x/y) differs**;
  (b2) it **changes when only the baked base pp_root differs**.

**Mechanism:** the leaf_prover circuit bakes per-shard data as circuit constants — base pp_root
(`leaf.rs:97`) and each shot's x/y limbs (`circuit_statement.rs:483-494`, `konst`/`context.constant`)
— which land in the leaf's PREPROCESSED trace ⇒ a different `leaf_preprocessed_root` per shard. But
the pipeline derives ONE `AggregateConfig` from shard 0 and reuses its cached tree0 + single trusted
`leaf_preprocessed_root` for every leaf (`main.rs:2440`); later shards no longer match ⇒ the failure.

**Why NOT just allow a different `leaf_pp_root` per shard:** it would force the node + root unpacker
to take each child's root as an AUTHENTICATED input instead of a trusted constant = the deferred #3
generalized-unpacker (modifies the soundness-critical shared verifier ⇒ re-audit) AND voids the
tree0 precompute reuse. And it is the WRONG fix: every shard runs the SAME program / SAME AIR / SAME
shape — the leaf_prover circuit is genuinely identical; only the WITNESS (the base proof + x/y)
differs. The per-shard root is purely the constant-baking artifact. (Per-shard roots are only right
for heterogeneous leaves — different shapes/programs — which is not our case.)

**FIX (gate-air-leaf ONLY; does NOT touch the shared circuits_stark_verifier / node):** feed boundary
x/y and the base pp_root into the leaf as **guessed witness / public-claim Vars**, not
`context.constant(...)`, so they stay out of the leaf preprocessed trace and all leaves share ONE
`leaf_preprocessed_root`. Build the output-hash preimage (`leaf.rs:106-114`) from those Vars. This is
exactly the M3 leaf-encoding pinned above (`H_i = blake(H_P ‖ x ‖ y)` with the per-shot data
*guessed*); the current `context.constant` version was an M3a stepping-stone.
SOUNDNESS CRUX: the guessed pp_root must be the one `verify` uses to check the base proof (a wrong
guess must fail verification), and the boundary Vars must be constrained to the base proof's actual
x/y (via the existing logup/boundary constraint) — not left free.
LAPTOP success gate (no box): after the fix, `leaf_preprocessed_root` from `derive_aggregate_config`
must be IDENTICAL across shards with different boundary (b1) and different base pp_root (b2). BOX
gate (later): full distinct-shard fold completes + `recursion_fingerprint` byte-identical + verify.

## Witness-shrink optimization — move shard-invariant base columns to the shared pp-trace (2026-06-30)
Of the **191 main (witness, tree1) columns** of the gate_air base AIR, some are constant across
shards and could move to the **preprocessed (tree0) trace shared by all shards**, shrinking tree1.
A column can move ONLY IF **(1) position/shape-determined** (so identical every shard) **AND (2) not
secret** (the pp-trace is PUBLIC — the verifier sees it). Condition (2) is load-bearing: the gate
circuit is hidden. Column layout = `cell_at` (`main.rs:1314`), `TRACE_COLUMNS = 191`.

- **MOVABLE (positional, leak nothing new — shape is already public via `pc_in_prog`/`prog_slot`):**
  `enabler` (padding pattern, fixed by `padded_rows`), `shot_id` (local index 0..shots_per_shard),
  `pc` (positional counter; `pc_in_prog = pc % n_gates` is ALREADY preprocessed).
- **NOT movable — position-determined but SECRET (public pp would leak the circuit):**
  `is_nop/is_not/is_cnot/is_toffoli` (the opcode = hidden program), and per read block (×3)
  `q`/`limb_idx`/`bit_pos`/`mask`/`lsel[32]` (the qubit each gate touches = hidden wiring). These
  stay witness, bound to the secret program table via the program-consistency LogUp.
- **NOT movable — per-shot CONTENT:** `in_limb[32]`/`out_limb[32]` (qubit states), `lo`/`hi`/`bit`
  (range-check + extracted bit of a value), `ab`/`fire`/`delta` (control logic + state update).

Caveats: (a) this is a witness-size/perf optimization, NOT a correctness fix; (b) before moving,
verify the constraint framework allows `enabler`/`shot_id`/`pc` as preprocessed (pp columns CAN be
used in constraints — `pc_in_prog`/`prog_slot` already are — but confirm the LogUp enabler/selectors
aren't required in the main tree). Note this is the OPPOSITE direction from the distinct-shard fix
above (there, shard-VARYING leaf data moves OUT of the leaf pp-trace into witness; here,
shard-INVARIANT base columns move INTO the shared base pp-trace).

## Base-side precompute — reuse shard-invariant base work across all shards (2026-06-30)
The CPU recursion already reuses witness-independent data (opt#1: leaf/node tree0 + twiddles via
`prove_circuit_with_precompute`, valid across shards once the distinct-shard fix lands). The SAME
pattern (the `privacy_prove` `RecursiveProverPrecomputes` template) applies to the **base gate_air
proof on GPU**, where each shard's `prove_base_shard` currently REBUILDS these from scratch even
though the diagnostic proved they are byte-identical across shards:

1. **Base tree0 (preprocessed commitment).** The 10 preprocessed columns (qdecode/prog_slot/
   pc_in_prog/rc) + their Merkle commit are shard-invariant. Compute + commit ONCE, keep
   **device-resident**, reuse for every base proof (at 9024 shards this removes 9024× redundant
   tree0 build+commit). GPU/base analog of opt#1.
2. **Base twiddles.** `precompute_twiddles` over the base domain (`main.rs:~2139`) depends only on
   `max_log_size` (shard-invariant for equal shards) → compute once.
3. **GPU constant uploads.** `gates_flat`/`off_lo`/`off_hi` + rc/qdecode lookup tables are
   shard-invariant (only `x_states` differ per shard) → upload once, keep device-resident, instead
   of `gpu_flat_inputs(&gates, …)` per shard.
4. **Program-witness + multiplicity columns** are shard-invariant but live in **tree1** (mixed with
   the per-shard main trace) → can share their *generation*, not the tree1 *commit*.

Synergy: if `enabler`/`shot_id`/`pc` move into base tree0 (witness-shrink above), they become part
of this precomputed shared tree0 automatically.

INHERENTLY per-shard (cannot move): main-trace gen (per-shot qubit states), base `prove_ex`
(composition/OODS/FRI over the witness), root verification (variable size).

CALIBRATION: these are REAL but second-order — each is a fraction of the ~113s/shard base cost
(dominated by main-trace gen + prove_ex, both per-shard). They pay off most at SCALE (9024×
redundancy) and for GPU residency (no re-upload of constants per shard). Headline levers remain
streaming-Merkle (2^23 ceiling) and GPU recursion. Scope doc: results/BASE_PRECOMPUTE_SCOPE.md.

**Pipeline-walk addendum (2026-06-30, results/BASE_PROOF_WALK.md) — the movable set, completed:**
the known set holds, plus two under-specified items. **N1:** `build_program_table` (main.rs:2128) is
rebuilt every shard but is identical across shards (multiplicity = shard_samples*k, constant) — fold
into Phase 1 (the top-level `program` at :1976 can't be reused: it uses full `samples` → wrong
multiplicity). **N4 (highest-confidence GPU win, easy):** there is NO PTX module cache — `compile_ptx`
+ `load_ptx` runs **3*n_shards** times (gpu_tracegen.rs:1331/1477/1559) where **2 total** suffice; the
device handle is already `OnceLock`-cached (:53), the modules are not. At 9024 shards that's ~27k NVRTC
compiles vs 2 → likely a large, cheap saving; cache the modules in a `OnceLock` like the device. N3 =
gates_flat/off_lo/off_hi re-flattened 2x/shard (Phase 2). Confirmed: `pc_in_prog` is shard-invariant
(correctly in tree0); `generate_program_witness` is invariant but trapped in the per-shard tree1 commit
(partial — gen reusable, commit not). Per-shard-inherent (do NOT touch): build_rows/sim, multiplicity
gen, main-trace gen, grind/draw, interaction gen, tree1/tree2 commits, prove_ex, leaf wrap, fold, root.

## CPU recursion precompute — additional candidates (2026-06-30, results/CPU_RECURSION_WALK.md)
opt#1 (leaf/node tree0 + twiddles, reused via `prove_circuit_with_precompute`) captured the LARGEST
invariant per-proof cost but is not the complete set. Further proof-invariant work redone per
leaf/node/root, priority order:
- **C2 (highest):** circuit_prover `write_trace` rebuilds the `input_to_row` HashMaps for the 5 fixed
  bitwise-xor tables (xor_4/7/8/9/12) + range_check_16 on EVERY prove — global constants (not even
  shape-dependent), identical across every leaf, every node, AND the root, ×~2N proves (~2^8+2^14+
  2^16+2^18+… entries). Fix = a process-wide `OnceLock`/thread-local cache keyed on the constant
  tables — but it lives in `write_trace` with no injection hook, so it requires a (byte-identical,
  safe) change to the SHARED `circuit_prover` crate (upstream stwo-circuits), not just an
  `AggregateConfig` field. Likely sub-second/prove but ×~2N. Backend-independent.
- **C1:** `build_gate_air_leaf_circuit`/`build_multiverifier_circuit` re-emit the invariant op graph
  per prove (only `context.values()` is per-proof). Modest direct gain; enables a clean
  structure-once/assign-per-proof split. Needs a circuits-crate refactor.
(C3/C4 and C5 were minor/setup-only → moved to the **Deferred micro-optimizations** section below.)
Per-proof-INHERENT (correctly not precomputed): witness trace-gen/interpolation, grind/interaction
draw, composition/quotient/FRI/query openings (the dominant cost). `prove_root_verification` stays
uncached — its trace size grows with N (O(N) unpack + zk-blinding), so its shape varies.
CALIBRATION: smaller than the base-side wins (recursion's dominant cost is per-proof FRI/quotient).
DECISION DEFERRED to post-#10 — the measured curve shows how much of the wall the recursion actually
is (post-opt#1); C2 is the cleanest but touches the shared circuit_prover crate.

## Deferred micro-optimizations (skipped for now)
Small opts noted but NOT pursued — low payoff and/or off the critical path. Parked here so they aren't lost.
- **Fused-1a-proper (base commit / leaf-build — POSSIBLY FASTER than the chosen TILED Part-2):** absorb each large
  column into the leaf state AS PRODUCED in the NTT loop (before dehydrating), so the leaf build needs NO rehydrate
  — saving TILED's ~47 GB H2D round-trip (~2 s per 2^25 base proof, until async-overlap (ii) hides TILED's rehydrate
  anyway). BUT on the MIXED-SIZE tree1 the fused producer must first absorb the small groups + lift the state to
  max_log_size, then absorb the large columns as-produced, then finalize — i.e. **re-implement build_leaves'
  heterogeneous accumulation inside evaluate_polynomials' group loop (CODE DUPLICATION of the tested leaf logic +
  extra soundness risk).** Peak ≈ 26-33 GB. Chosen TILED instead for simplicity/reuse-of-tested-logic; revisit if
  the rehydrate cost proves material and async-overlap doesn't fully hide it.
- **Option-2 (base commit / leaf-build):** instead of Fix(b)/Option-1's lift-smalls-to-full-size, tile the
  heterogeneous lifted-Merkle STATE-lift itself per row-block — absorb small columns at their native size into
  each block's `src`-preimage sub-states, lift up, then absorb the 188 large columns' rehydrated block. Saves
  ~1 GB peak + ~1-2% commit compute (smalls are 2^9/2^16 — a rounding error vs the 188 large cols) at a large
  jump in complexity + soundness risk (per-block multi-level `src`-map). Revisit only if the #1 `d_cols` floor
  is later addressed and every GB matters.
- **C3/C4 (recursion, minor):** `CircuitComponents::new` + `TraceLocationAllocator` rebuild invariant component
  wiring per prove; `prove_node` runs a redundant `validate_circuit()` per node (debug-class, cheap to gate off).
- **CPU build_rows trace materialization (verified 2026-07-04, marginal):** on the GPU base path the CPU `rows`
  (Vec<Row>, 776 B ea) are NOT consumed — main trace = K1 `d_cols` (main.rs:2528-2561), interaction main = K4
  `gpu_gen_interaction_device` (main.rs:2606-2611); `gen_main_interaction(&rows,..)` (main.rs:2628) is the CPU
  fallback only, and `rows.len()` (main.rs:2433) is the only GPU-path read. BUT `build_rows` (main.rs:2432) also
  returns `counts` (LookupCounts), which IS needed: it feeds the tree1 multiplicity cols (main.rs:2506-2509) +
  tree2 table interactions (main.rs:2636/2648/2658). counts + the final-state self-check (main.rs:600-608) + the
  full Row write are all ONE fused per-shot pass (simulate_shot), so you cannot skip the trace build to keep only
  counts+self-check without splitting that loop. Note K1 ALREADY computes byte-identical histograms on-device
  (gpu_tracegen.rs:1636-1641 → d_qd/d_lo/d_hi) but the caller DISCARDS them (`_qd,_lo,_hi`, main.rs:2528) and uses
  the CPU counts instead. Real opt = consume the GPU histograms + drop the CPU sim to a lean final-state-only self-
  check (no Row materialization, no count_read). Payoff marginal: it is a light CPU pass that overlaps GPU work,
  off the GPU critical path; matches the line-171 "build_rows: do NOT touch" call. NOT a lever — leave it.
- **[deferred, LOW] Drain the ~24 GB pinned host pool at the base->leaf transition:** add a `cudaFreeHost` drain of
  PINNED_POOL's free-list (today it only recycles via `give()`, never returns page-locked RAM to the OS); call it
  when crossing from GPU base-proving to CPU recursion. Reclaims a fixed ~24 GB of page-locked host RAM (does NOT
  scale with N), lowering the floor the ~25 GB SimdBackend leaf-prove climbs from. NOT needed after fix #1 (fold now
  fits ~47 GB peak on the 83 GB box with margin); the a2-8g deployment has ~680 GB host RAM so it's irrelevant there.
  Becomes relevant only for a tighter-RAM host or higher fold concurrency (n_pools>1 running multiple concurrent
  leaf-proves). Needs a new pool-drain API (fused_commit.rs).
- **[deferred, LOW] Robust RAM-aware single-in-flight-leaf cap:** fold-fix #2 ("one ~25 GB SimdBackend transient at a
  time") is EFFECTIVELY satisfied on target hardware but NOT robustly guaranteed. Current sizing (main.rs:2945-2954) is
  a core-count heuristic: n_pools = (cores/pool_threads).max(1) → 1 on the 12-vCPU box (measured N=8 peak 47 GB, fits
  83 GB), but ≥2 on a ≥96-core host regardless of RAM. Two gaps: (i) not RAM-aware (many-core → multiple concurrent
  ~25 GB node-proves); (ii) n_pools bounds only NODE-prove concurrency — the leaf-wrap runs on the separate consumer
  thread, so even at n_pools=1 a leaf-prove + a node-prove can coexist (~2×25 GB). Does NOT bite in practice (measured
  fit; a2-8g has ~680 GB). Robust fix = cap (n_pools + leaf/node coexistence) by available RAM. Low priority.
- **C5 (recursion, setup-only):** base (GPU) precompute and recursion (CPU) precompute are built SERIALLY, and
  the CPU one is data-coupled to the GPU one — `BaseProverPrecompute::new` (main.rs:2733) runs first, then shard 0
  is proved, then `derive_aggregate_config` (main.rs:2871) builds the leaf/node `CircuitPrecompute` from
  `shape_params` carrying shard 0's `pp_root0` (main.rs:2860). They use DIFFERENT resources (GPU device vs CPU
  host) and the recursion precompute is witness-independent + SHAPE-only (log sizes known upfront; base pp_root is
  a guessed Var post distinct-shard-fix, not baked) → both could build CONCURRENTLY at startup (thread::scope,
  split the shape-only `CircuitPrecompute::new` from the pp_root0 binding). PAYOFF: tiny — both are one-time setup
  (base ~sub-second; recursion a few s), amortized over a fold of thousands of shards. Cheap cleanup, NOT a lever.

## Fold-tree pipelining across GPUs — #2 now, #3 deferred (decided 2026-06-30)
How the N base-shard proofs feed the 2-to-1 multiverifier fold tree on 8 GPUs.

**#2 (CHOSEN) — fixed balanced+carry tree.** Topology fixed up front from N (balanced tree +
carry for non-power-of-2). Correct on 8 GPUs because topology ⊥ completion order and we are
**CPU-fold-bound**: one CPU can't keep up with 8 GPUs' base output ⇒ a backlog always exists ⇒
fold pools stay saturated regardless of arrival order. Wins: deterministic + byte-identical
`recursion_fingerprint` (the cheap "streaming == sequential" validation gate holds), no
verifier-side change, canonical leaf-output (shard) order for free, no re-audit.

**#3 (DEFERRED) — dynamic topology + generalized unpacker.** Fold siblings as GPUs finish;
topology unknown at circuit-build time. SOUND with the generalized unpacker: it recomputes the
root hash from `(topology, leaf outputs, trusted leaf/node pp-roots)` and `eq`-binds it to the
verified root proof's output. Blake2s-Merkle collision resistance means the prover can't supply a
different topology/leaf-set hashing to the same root, so topology becomes authenticated input —
safe as long as the replay recomputes the same per-node `blake([ppR_L, outs_L, ppR_R, outs_R])`
and keeps the one-trusted-leaf-root invariant. (lib.rs module doc already flags "arbitrary tree
shape unknown at circuit-build time" as a future optimization, so the unpacker may get generalized
anyway.)

Three costs #2 doesn't pay:
1. Touches the soundness-critical verifier-side unpacker (generalize `prove_root_verification` to
   take an explicit fold-tree) — the one piece otherwise kept byte-for-byte frozen ⇒ re-audit.
2. Published proof becomes **nondeterministic** (topology + leaf-output order depend on GPU arrival,
   jitters run-to-run): breaks the `recursion_fingerprint` byte-identity gate (need a weaker
   verify+content-match gate), and leaf outputs arrive in arrival-order not shard-order (need an
   in-circuit permutation back to canonical, itself bound, or a content-matching verifier). #2 gets
   canonical shard order for free.
3. ~Zero throughput gain in our regime: we're fold-*bound*, not fold-*starved*. #3 only avoids
   idling a fold worker on a slow sibling, but a CPU that can't keep up with 8 GPUs never idles its
   pools. #3 reorders work that was never going to idle.

**TRIGGER TO REVISIT #3 — when the bottleneck changes.** #3's value is contingent on the recursion
NOT being the bottleneck. If/when GPU-accelerated recursion lands, folds get cheap, the CPU stops
being the wall, and the system can become **fold-starved**. *Then* #3 + generalized unpacker
matters (and is cheaper to adopt if the unpacker was already generalized). Building #3 before then
pays the soundness/nondeterminism bill before the benefit exists.


## Base-shard ceiling lift (2^24/2^25) + streaming — decided design & status
_Raw results (fingerprints, anchors, box run outcomes, curve numbers) → `results/BOX_VALIDATION_LOG.md`._
_Full verbatim pre-trim history of this section (all detail, frozen) → `results/CEILING_ARCHIVE_pretrim_2026-07-02.md`._
_Scope docs: `results/LDE_STREAMING_SCOPE.md` (route(a) ruled out; ceiling = route(c) + row-tiling),
`results/COMPOSITION_TILING_SCOPE.md` (the kernel row-tiling), `results/GPU_RECURSION_SCOPE.md` (pivot if ceiling stalls)._

**Diagnosis (decided):** 2^25 OOMs in K1 from a ~2× main-trace duplication — #1 `d_cols` (23.5 GB, K1 column-major,
held for K4) + #2 the 188 `d2d_column` copies (23.5 GB) coexisting before commit (47 GB); 2^24 adds #3 the 188
extended 2×-blowup evals. Streaming (route c) collapses #3 only; the structural fix must remove #2. CORRECTNESS not
soundness (verifier untouched; caught by byte-identity + self-verify).

**STRUCTURAL FIX = (b) [GREENLIT]:** keep #1 `d_cols` resident for K4; stop materializing #2 (drop the d2d_column
loop, gpu_tracegen.rs:1518-1520); tree1 commit processes ONE column at a time — borrowed view into
`d_cols[i*padded_rows]` → reused 128 MiB temp → per-column IN-PLACE interpolate → extend → n2b → absorb → dehydrate.
(a) [regenerate d_cols for K4] RULED OUT (the 2^25 OOM is in K1 where #1+#2 coexist). Per-col interpolate == batched,
byte-identical; never write into d_cols (K4 unchanged, FS order preserved). Why interpolate MUST be folded into the
per-column loop: the current interpolate is a SINGLE BATCHED NTT over all 188 cols (poly.rs:282-291) that is itself a
hidden #2-equivalent (needs all 188 coeff buffers resident).

**FULL (A) — resident/staged model [DONE, pending box]:** under FUSED_INTERP+STREAM only the 188 LARGE tree1 eval
columns are host-staged (dehydrated); small tree1 (mult/witness 2^9/2^16) + tree0 (preproc) + tree2 (interaction)
stay RESIDENT. Every consumer branches `is_staged(col) ? rehydrate/slice-from-stash : read/slice-live-device-ptr`.
Consumer set (confirmed bounded): quotient, composition, OODS-barycentric, decommit `col.at`, build_leaves (+
defensive dispatcher); FRI reads the resident quotient, not committed cols. tree2 (24 LogUp-cumsum cols, scattered
bit-reversed `-1` offset) stays resident + read whole (never tiled). Files: `quotient.rs` (Site 1: staged→
`rehydrate_block` / resident→`from_borrowed_ptr(+off)`); `evaluate_gate_air.cu` + `gate_air_entry.cu` +
`gate-air-cuda-kernel/lib.rs` (Site 2: per-column staged/resident composition tiling, `tiled_input=host0||host1`,
resident cols read UNBIASED); `poly.rs` (Site 3: dehydrate ONLY the 188 large cols); `blake2s.rs` build_leaves
per-group rehydrate; Defect-1 (`interp_base_log`, poly.rs:618).

**Staged-path aliasing fix [DONE, pending box] (2026-07-02):** ROOT CAUSE — `HOST_STASH` is keyed by the FREED
`col.device_ptr`; `cuda_free_memory` returns the block to a recycling CUDA pool; the `h.len()==col.size` guard
doesn't discriminate (tree1 main & tree2 interaction eval cols are both eval-domain size). tree0 is committed first
in the cached precompute → safe; tree2 commit (doesn't clear the stash) reallocates interaction cols from the pool
that JUST freed the tree1 blocks → a resident tree2 col matches a STALE tree1 key. Composition reads tree2 from LIVE
pointers (correct); OODS trace-sampling (`barycentric_eval_at_point` → `is_staged`) rehydrated STALE tree1 bytes →
committed-composition ≠ sampled-trace at OODS (prover/mod.rs:186) → `ConstraintsNotSatisfied`. FIX (`fused_commit.rs`
ONLY): gate `is_staged` / `staged_host_ptr` on `!col.owns_memory` — a resident committed col always owns its buffer;
`dehydrate_column` always sets a staged col `owns_memory=false` → the gate excludes aliasing resident cols and never
misses a genuine staged col. Fail-loud preserved; byte-identical; private backend only.

**Key conclusions (durable understandings — the "why" behind the design):**
- **live-lifting** = an on-the-fly index remap ALREADY in upstream stwo `vcs_lifted` + our CUDA `build_leaves`
  (`blake2s_lift_states_kernel`, byte-identical to CPU/SIMD `merkle_lifted.rs:59`): mixed-size columns commit in ONE
  tree by lifting the SMALLER columns into the hash STATE — small cols are absorbed at NATIVE size, NOT materialized
  to full size. So build_leaves' heterogeneous orchestration (small groups → lift → absorb large → finalize) is
  ALREADY CORRECT for resident columns. It is a MERKLE-COMMIT mechanism ONLY and does NOT bear on the composition
  read. ⇒ Options 1 (materialize small cols to full size) / 2 (tile the state-lift) are DROPPED as unnecessary.
- **Composition-read model:** the gate_air composition kernel reads `trace_evaluations[col][global_row]` with a
  SINGLE shared `eval_domain_log_size` for all 188 tree1 + 4 tree0 + 24 tree2 cols (`eval_at_row.cuh:159`; global row
  = `row_offset + local`, set at `evaluate_gate_air.cu:222`) — NO per-column lift remap in the read (unlike the Merkle
  commit). ⇒ every referenced col is a full-eval-domain buffer; a RESIDENT col is read UNBIASED, while a STAGED tile
  buffer (local rows [0,tile)) is biased `−tile_start`. Resident-vs-staged is a data-SOURCE difference only, never a
  value difference → byte-identical by construction. (A −tile_start bias on a resident full buffer = the multi-tile
  bug I first chased; the real bug was the aliasing above. The `.cu` unbias fix — `biased{0,1}[c] =
  trace{0,1}_evaluations[c]`, no `-tile_start`, evaluate_gate_air.cu ~692/711 — was NOT what fixed the failure
  (single-tile still failed) but is KEPT as a legit correctness fix for the multi-tile case.)
- **tree1 is MIXED-SIZE / Defect-2:** the FUSED_INTERP path feeds the 188 large cols (`extend_polys`) AND `small_main`
  (mult/witness 2^9/2^16 + program, `extend_evals`) into the SAME `tree_builder` (main.rs:3177-3178) → build_leaves
  takes the HETEROGENEOUS path (`all_same_size=false`, blake2s.rs:119): it does NOT consume the fused large-only stash
  and REBUILDS from the columns. So a fused-stash-then-free "1a" would read the freed large cols = use-after-free
  (Defect-2). ⇒ the leaf build must REHYDRATE the staged large cols (the DONE design), not rely on a fused stash.
- **Memory floor / 2^26:** #1 `d_cols` (23.5 GB @2^25) is the residency FLOOR — scaling past ~2^26 would also need to
  stream #1 (out of scope for 2^25). The blake2s STATE array is ~108 B/row (`Blake2sState` = h[8]+t+buf[64]+buflen) →
  ~7 GB @2^26, the dominant non-d_cols swing term (whole-column rehydrate does NOT shrink it); if 2^26 OOMs the next
  lever is per-block states (bigger, security-critical — measure-then-decide).
- **Ruled out (residency accounting):** NTT is IN-PLACE (no scratch, rfft/ifft.cu); no interpolate 2nd buffer; coeffs
  ARE freed (`store_polynomials_coefficients=false` → dropped at poly.rs:699). Nothing else is trace-sized.

**Stash concurrency caveat (2026-07-02, current impl — pending box validation):** the host-stage stash
(`HOST_STASH`, fused_commit.rs) is now PROCESS-GLOBAL (`static Mutex<HashMap>`), keyed by a monotonic SENTINEL id
(`1<<63|id`) written into the staged column's `device_ptr` — this replaced the old reusable-device_ptr key that
caused a 94/188 stash collision (the CUDA pool recycles freed addresses, so freshly-allocated columns collided with
older columns' still-present stash keys). It was made GLOBAL (not thread-local) because the stash is read on rayon
WORKER threads — OODS `eval_at_points` runs under `par_map_cols` (the `parallel` feature, which `cuda` enables), and a
thread-local stash is empty on a worker → `is_staged` false → the sentinel `device_ptr` gets dereferenced → illegal
address (barycentric.cu:241). CAVEAT: `clear_stash()` wipes the WHOLE map, so the global stash is UNSAFE if multiple
CUDA base proofs ever run CONCURRENTLY in ONE PROCESS (one proof's clear_stash would wipe another's staged entries).
NOT a problem today: CUDA base = one proof per process/GPU; the CPU saturation-sweep + recursion fold are SimdBackend
and never touch this stash; async-overlap (#20) is single-proof intra-proof pipelining. IF we ever prove multiple CUDA
shards concurrently in one process, namespace by proof-id (the monotonic id already makes keys unique — just scope
clear_stash to a proof's own keys instead of wiping the map).

**DEFERRED:** (ii) async-overlapped 2^25 [task #20] — pinned double-buffered D2H/H2D + allocator reclaim without a
per-column host sync, layered AFTER (b) fits 2^25 (throughput ~2.1×→~1.6× — SUPERSEDED: this IS the B1/B2 work, now
MEASURED at ~4.4×, see BOX_VALIDATION_LOG; async overlap did NOT reduce wall; ORTHOGONAL to the OOM, not a substitute).
Async-race is a CORRECTNESS risk (stale bytes → rejected proof), caught by byte-identity + self-verify; needs CUDA
events for cross-stream ordering. Fused-1a-proper + Option-2 parked in "Deferred micro-optimizations".

**STATUS (2026-07-02, VALIDATED on box):** streaming base-proof CORRECTNESS is CLOSED and the ceiling is LIFTED to
2^24. The staged-path bug chain — (1) resident/staged aliasing (owns_memory guard), (2) reusable-device_ptr stash-key
collision 94/188 (monotonic SENTINEL key), (3) thread-local stash empty on rayon OODS workers (HOST_STASH →
process-global Mutex) — is all FIXED and box-confirmed: [A] 2^22 tiled fp == oracle `ab1e75b5…`, [B] 2^23
tiled==resident `f304b5ee…`, [D] 2^24 exit=0 NO OOM (prove TOTAL 17.0s; tree1 27.7s). **Working ceiling = 2^24.**
**2^25 = CAPACITY-bound (decided, corrects the fragmentation story):** streaming fixed the tree1 OOM (tree1 commit
COMPLETES at 55.5s); 2^25 then OOMs LATER because d_main (188×128 MiB ≈ ~24 GB) is genuinely-live non-pool memory held
through K4 (only 4.5 GB free after K1 vs 22.2 GB @2^24). NOT fragmentation, NOT hoarding: notrim and trim-after-K1 both
OOM at `alloc inter`; option-0 boundary-trim (GATE_AIR_BOUNDARY_TRIM) is correctly-placed + NOT SUFFICIENT on its own —
it clears d_inter (interaction runs) yet 2^25 dies one phase later at tree2 commit (ifft.cu:839).
**FIX DIRECTION (chosen):** stream d_main too — dehydrate main columns to host after consumption, rehydrate per-column
for K4 (frees ~24 GB). option-0 may become UNNECESSARY once d_main streaming frees ~24 GB; keep it as a CANDIDATE
companion PENDING the next box run (do not assume it is needed). trim-after-K1 is mis-placed, drop it.
**LEVER (understanding, MEASURED + CORRECTED 2026-07-03):** the tree1-commit tax @2^24 (~27s) is dominated by the
dehydrate span (dehydrate_d2h ~11.6-19.7s + build_leaves_h2d 5.4s; prove_ex adds oods_h2d 10.8s + quotient_h2d 2.7s) —
NOT CPU trace-gen (≤9%) and NOT sync barriers (that theory REFUTED — reclaim 0.003s; Part A is a measured NO-OP). The
dehydrate span was FIRST mis-read as ~1.3-2 GB/s "pageable copies," then RE-DIAGNOSED (nsys, below) as 188 one-time
page-locks, not a copy at all.
CORRECTED AGAIN by nsys + fold amortization (BOX_VALIDATION_LOG 2026-07-03): the "dehydrate_d2h ~2 GB/s" is NOT a slow
or pageable copy — it is 188 ONE-TIME per-column cudaHostAlloc PAGE-LOCKS (nsys: cudaHostAlloc 11.78s / 188 calls /
62.7ms each) charged to that span; the actual copy is ~12 GB/s. The page-lock is paid on the FIRST shard only (pinned
pool recycles across shards) and AMORTIZES: steady-state 2^24 t_base ≈ 20s (fold shards 1–3), not 35s. ⇒ there is no
"~6x pinning win" to chase — the earlier "make the copies actually pinned" lever is RETRACTED; the copy was already fast.
Part B0 (GATE_AIR_PIN_STASH, byte-identical, -25% wall single-shot) helped only by starting to reuse pinned buffers;
its residual per-shot cost IS the one-time page-lock, which simply amortizes across shards. NEXT lever = unblock 2^25
shards (the SP1-challenge config), NOT more pinning.
**CURVE DECISION (measured, CORRECTED 2026-07-03 by fold amortization):** the single-shot 2^24 numbers (6.4× as-is,
4.4× with B0/B1/B2) OVERSTATED the cost because they charged the ONE-TIME page-lock to every shard. Fold measurement
(GATE_AIR_FOLD, 2^24) shows steady-state per-shard t_base ≈ 20s (shard0 31.8s, shards 1–3 ~20s) → **~2.5× SP1 @k=2000
(approx, needs curve.py), which BEATS 2^23-resident's 3.3×.** ⇒ 2^24-STREAMING is now the better config than
2^23-resident. (~2.5× still means 2.5× SLOWER than SP1; Tanuj curve needs <1×.) FREE_ON_STREAM0 / Option A (BATCHED) /
B1/B2 async overlap remain measured NO-OPs — but because they targeted the copy, not the one-time page-lock (which
amortizes on its own), NOT because of a hardware wall. NEXT lever = unblock 2^25 shards (bigger → fewer recursion nodes
→ lower multiple; the real SP1-challenge config), currently blocked by a pool-alloc crash (exit=134). The correctness +
ceiling + B0/B1/B2 work this session was the necessary UNLOCK.
**Latest measured state (2026-07-03, nsys + fold amortization — dehydrate CLOSED):** the dehydrate cost is a ONE-TIME
per-column cudaHostAlloc PAGE-LOCK, not a slow copy and not a pageable defect. nsys (2^24 batched): cudaHostAlloc =
46.1% / 11.78s / 188 calls / 62.7ms each; the "dehydrate_d2h ~11.6s / ~2 GB/s" wall-time WAS those 188 page-locks
charged to that span — the actual D2H copy is ~12 GB/s. The pinned pool is process-global and recycles buffers across
shards (clear_stash → free-list), so the page-lock is paid on the FIRST shard ONLY and AMORTIZES: fold measurement
(GATE_AIR_FOLD, 2^24, SHARD_SHOTS=4) gives t_base[shard0]≈31.8s then STEADY-STATE ~20.0s for shards 1–3 (t_leaf
~8.4–10.6s). ⇒ steady-state 2^24 t_base ≈ **20s**. **BANKED RESULT (curve.py, apples-to-apples with the 4.4× curve —
only t_base 35.2→20.0, same anchors: 2^24, t_leaf/t_node 15.8/10.7, P_8g=16, tail 6): 2.49× vs SP1 Tanuj curve @k=2000**
(k=1 1.41× | k=10 1.50× | k=100 2.33× | k=1000 2.48× | k=2000 2.49×; base-bound at every k). **BEATS 2^23-resident
(3.3×)** — 2^24-streaming is the best VERIFIED single-machine result (byte-identical fp `44fec71a…` == oracle). Still >1×
(2.49× SLOWER than SP1; does NOT beat the Tanuj curve, which needs <1×).
**2.49× overlap is now MEASUREMENT-BACKED (2026-07-03), not an assumption:** the model is
`T = max(base_wall=⌈N/8⌉·t_base, rec_wall=N·(t_leaf+t_node)/16) + tail` — the `max` assumes GPU base ‖ CPU recursion.
The FOLD-PIPELINE THAT REALIZES THIS OVERLAP ALREADY EXISTS AND IS DONE (validate+harden complete): GATE_AIR_PIPELINE —
GPU producer streams base proofs over a depth-1 channel; CPU consumer wraps leaves + a static-tree worker-pool fold
(`build_fold_topology`, known N, NO dynamic unpacker). It is (a) BYTE-IDENTICAL (N=2 pipeline fingerprint ==
sequential), (b) SCALES — after the dead-`build_rows` fix (main.rs, skip in fold mode) host mem is FLAT ~47GB at
N=2/4/8, i.e. O(shots_per_shard) NOT O(N), so real large N (1128–2256 shards) fits the 83GB host, and (c) OVERLAPS —
measured N=8 wall 253s vs serial ~325s ⇒ ~22% faster, ~70% of ideal on a 1-GPU box with a SINGLE fold worker (a2-8g's
16 workers do better, so this is a conservative lower bound). So 2.49× is ASYMPTOTICALLY SOUND (large N ⇒ tail
negligible ⇒ base-bound), and the earlier "serial ~3.65× / real between 2.49 and 3.65 pending measurement" is RESOLVED
toward 2.49× — the overlap is real; the only residual is the finite-N pipeline TAIL (last shard's recursion can't hide),
which shrinks as a fraction at large N. Full numbers → `results/BOX_VALIDATION_LOG.md`. (For context: if base and
recursion did NOT overlap it would be additive `base_wall+rec_wall` ≈ 4.14× @k2000, so the overlap is worth ~1.65× on
the multiple — hence why measuring it mattered.) The single-shot 35s → 4.4× OVERSTATED it (it
included the one-time page-lock). This RETRACTS the earlier "pageable-copy defect / ~6x pinning win" framing: the copy was
never the bottleneck, so FREE_ON_STREAM0, Option A (BATCHED) and B1/B2 async overlap were all NO-OPs — keep them as
harmless default-OFF flags but they are NOT the lever.
**Committed & pushed (2026-07-03) — the banked streaming set (3 coordinated branches):** the code behind the 2.49× result is committed + pushed:
- stwo-cuda-backend `anatg/cuda-backend` @ `f06c5763` → `starkware-libs/stwo` (LDE/host streaming + pinned stash + fused_commit.rs).
- grover-tax-v02 `anatg/gate-air-leaf-cuda-tests` @ `1affbcf` → `anatgstarkware/grover-tax` (fork; `origin`=AbdelStark/grover-tax) (d_main streaming + fold timing).
- proving-utils `anatg/multiverifier-recursion` @ `5652c76` → `starkware-libs/proving-utils` (streaming multiverifier fold).
Wiring: leaf `[patch]` → local `../../stwo-cuda-backend/crates/stwo`; leaf path-dep → `../../proving-utils/crates/recursive_aggregate` (RELATIVE paths ⇒ the 3 repos must sit side-by-side on these branches, else Cargo falls back to pinned rev 74951f79 = upstream, no streaming). Streaming is flag-gated default-OFF, so this ONE set carries BOTH results: resident 2^23 (flags off, byte-identical) and streaming 2^24 (2.49×). No PRs opened.
**2^25 shards — PARKED.** Blocked: device tree2-commit OOM after 5 fix attempts (streaming, LOWMEM, boundary-trim,
cuMemFree free_after_k4, pool-alloc); the u32-overflow fix advanced past K1 but tree2 still OOMs (128 MiB allocs; d_cols
not freed for tree2); 2^25 fold also OOM-killed (exit=137). Deprioritized on STRATEGY not just the block: the curve is
base-bound, so bigger shards buy only ~10–20% fixed-overhead amortization (est ~2.1×), NOT enough to beat SP1 — the ~2.5×
gap is ALGORITHMIC (8×A100 vs 8×A100), not shard-size. NOTE: the base proof's trace-gen (K1/K4) and lifted-Merkle
(blake2s) are ALREADY on GPU — so the levers to challenge SP1 (<1×) are NOT "port to GPU" but: (1) base-proof GPU KERNEL
efficiency (nsys hot kernels: blake2s lifted-Merkle 31.6%, gate_sim/gate_sim_states ~21%, NTT ~20%); (2) a smaller/cheaper
base AIR (fewer than 188 cols, lower blowup, fewer constraints) — the bigger algorithmic swing; (3) GPU-port the RECURSION
(leaf/node/fold, still CPU/SimdBackend) — only matters once the base is cut enough to become recursion-bound (today
t_base 20s > (t_leaf+t_node)/2 ≈ 13s, so base-bound). Fuse-1a is a ~10% base-kernel polish, not a game-changer. (2^25 host ceiling was SOLVED via GATE_AIR_STREAM_MAIN_LOWMEM,
72→46 GB; DEVICE tree2 OOM is the open item if ever resumed.) Full numbers → `results/BOX_VALIDATION_LOG.md`.

## Path to beat SP1 (<1×) — scoped 2026-07-04 (both levers profiled)
**SP1 comparison is FAIR (verified):** the Tanuj/SP1 benchmark (github tanujkhattar/zkp_ecc @update_examples) ALSO HIDES the
circuit — it's injected as PRIVATE stdin input, only its SHA256 hash is public; 9024 FS-selected test cases (= our 9024
shots). Same hidden-circuit task as ours (H_P hash-commit, program=witness). So the ~2.49× gap is a genuine, fair,
ALGORITHMIC gap — NO "we prove a harder thing" discount. Beating SP1 (<1×) is the right goal.

**LEVER A — Memory AirFn (~2×, BIGGEST, DUAL-PURPOSE):** replace the `lsel[N_LIMBS]` one-hot limb selector (96 of 188
main cols, ~half the trace) with a read-write memory argument ⇒ 188→~92 cols ⇒ t_base ~20→~10s. ALSO required to build
the **leaf adapter** (the in-circuit gate_air verifier the multiverifier consumes — needs the gate_air `AirFn` via
air_infra codegen; Cairo's `Felt252IdMemory` is read-only ⇒ custom Memory `AirFn`). So one effort shrinks the base AIR
AND unlocks the leaf adapter. Research/deep design; NOT blocked by hiding (limb/memory structure is a public encoding
choice; opcodes/`q`/states stay witness). No-new-infra alternative: wider limbs 16→31-bit (`N_LIMBS` 32→17, 188→~118,
~20→~13s, ~1.5×) — subsumed by the AirFn on `lsel`. **Full project plan + Phase 1 concrete scope →
`MEMORY_AIRFN_PLAN.md`.** NOTE (2026-07-04 correction): the mechanism above ("gate_air AirFn via air_infra codegen") is
SUPERSEDED — `blake_gate` is deprecated; gate_air is authored in the **stwo-circuits circuits DSL** (no air_infra dep),
as an address+timestamp RW-memory circuit (F2; F1/topology-based is ruled out by hiding). Leaf adapter still falls out
(the `circuit_verifier` `CircuitEval` IS the in-circuit verifier). Shrink refined to ~180→~3 addr cols (GateStep ~11–13
+ MemSort ~6–8). See the plan doc.

**LEVER B — kernel tuning (~1.3–1.5×, moderate):** ncu (2^22, sudo — perf-counters are admin-gated, `RmProfilingAdminOnly=1`)
shows the hot kernels are LOW-OCCUPANCY and MEMORY-pipe-bound, not compute-bound: `blake2s_update_columns` big launch =
72% memory / 9% SM / 16% DRAM / **12% occupancy**; NTT small stages = **6% occupancy**. Systemic low occupancy ⇒ raise
it (register/shared pressure, larger/batched launches, coalescing) ⇒ ~1.3–1.5×, capped by memory bandwidth. Multi-session
CUDA occupancy work.

**BLOWUP:** at the floor (2× domain; 96-bit FRI = pow 26 + n_queries 70×1) — no swing.

**COMBINED:** ~2× (AirFn) × ~1.4× (kernel) ≈ 2.8× ⇒ 2.49× → ~0.9× ⇒ beating SP1 (<1×) is PLAUSIBLE but needs BOTH deep
pieces; neither alone crosses <1× (AirFn→~1.25×, kernel→~1.7×). Natural order: **Memory AirFn first** (biggest lever +
required for the leaf adapter), then kernel tuning. (Current leaf `leaf::prove_gate_air_leaf` works/validated — confirm
whether it's the codegen adapter or a stopgap before the AirFn work.)

## Sequencing
1. Step 1 (above) — cheapest risk-reduction.
2. gate_air leaf via air_infra (+ read-write Memory AirFn) — the crux; emits `H_i` as its 2 outputs.
3. GPU port (NitrooZK).
4. Fold pipelining: #2 (fixed tree) is DONE — GATE_AIR_PIPELINE validated 2026-07-03 (byte-identical, scales flat host
   mem after the dead-build_rows fix, overlap measured ~22% faster than serial at N=8). Revisit #3 + generalized
   unpacker only after GPU-recursion shifts the bottleneck to fold-starved (see section above).
