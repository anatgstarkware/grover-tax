# gate_air BASE-proof pipeline walk — shard-invariant vs per-shard inventory

READ-ONLY completeness pass over the `prove_base_shard` closure
(`src/main.rs:2126`-`2361`) and every function it calls. Branch
`anatg/gate-air-leaf-cuda-tests`, laptop source
`/home/anat/workspace/grover-tax-v02/gate-air-leaf`.

## Sharding invariant that makes the analysis work

All shards have EQUAL shape. `shard_case_sets` (`main.rs:2110`) cuts `cases` into
`n_shards` slices of EXACTLY `shots_per_shard` shots each; the final ragged shard
repeats its last real shot up to `shots_per_shard` (`main.rs:2115`). So inside the
closure `shard_samples == shots_per_shard` for every shard, and therefore:

- `real_rows = shard_samples * k * n_gates` is equal across shards,
- `padded_rows`, `log_n_rows`, `max_log_size` are equal across shards,
- `program` (built with `shard_samples`) is byte-identical across shards,
- the PcsConfig, the preprocessed (tree0) root, and all target sizes are equal.

The ONLY things that differ shard-to-shard are the witness/cases-derived values:
each shot's `x`/`y` state, the `Row` cells, the `LookupCounts`, the multiplicity
columns, the drawn `LookupElements` + grind nonce (because the per-shard tree1 root
seeds the channel), the claimed sums, the interaction trace, and the FRI proof.

The captured setup (`gates`, `n_gates`, `k`, `rc_lo_index`, `rc_hi_index`, the
top-level `program` at `main.rs:1976`) is built ONCE before the closure and shared
by reference — none of it is recomputed per shard. The recomputation that DOES
happen per shard is everything listed inside the closure below.

## Tagged step table

| step (fn + file:line) | inputs it depends on | tag | if invariant: recomputed per shard? (movable) | notes |
|---|---|---|---|---|
| `build_program_table` (`main.rs:702`, called `:2128`) | `gates`, `shard_samples`, `k` | SHARD-INVARIANT | YES — movable | `shard_samples` equal across shards; `multiplicity = shard_samples*k` is the same constant for every shard. Whole table (slot/op cols/multiplicity) identical. Already built once at top-level `:1976` with full `samples` (different multiplicity) — NOT reused in closure. |
| `build_rows` / `simulate_shot` (`main.rs:458` / `:501`, called `:2129`) | per-shot `x_hex`/`y_hex`, gates, k | PER-SHARD | n/a | Threads 512-bit state per shot; fills `Row` cells; self-checks final == y. Inherent per-shard. |
| `counts` (`LookupCounts`) reduce inside `build_rows` (`:488`) | reads emitted by sim | PER-SHARD | n/a | qdecode/rc_lo/rc_hi histograms depend on actual read values. |
| `padded_rows`/`log_n_rows`/`max_log_size` (`:2131`-`2133`) | `real_rows` (= shape) | SHARD-INVARIANT | YES — trivially movable | Pure ilog2 of equal shapes; cheap, but recomputed. |
| `leaf::leaf_pcs_config` (`leaf.rs:54`, called `:2138`) | `max_log_size`, blowup | SHARD-INVARIANT | YES — trivially movable | Pure match on blowup + log size. Cheap. |
| `ProverBackend::precompute_twiddles` (`:2139`) | `max_log_size + 1 + log_blowup` | SHARD-INVARIANT | YES — **movable (in plan)** | Domain size identical across shards. Currently rebuilt per shard. Non-trivial cost. |
| channel init + salt mix + `config.mix_into` (`:2144`-`2147`) | fixed salt 0, config | SHARD-INVARIANT (initial channel state) | partial — see notes | Channel STARTS identical for every shard; diverges only after tree1 commit. Could clone a pre-mixed channel, but cost is negligible. |
| `CommitmentSchemeProver::new` (`:2148`) | config, `&twiddles` | SHARD-INVARIANT (construction) | borderline | Just wraps config+twiddles; the per-shard state is added by the tree builders. Cheap to construct; depends on movable twiddles. |
| **Tree 0 build**: `generate_qdecode_preprocessed` (`:1455`) | none (pure `qubit_decode` constants) | SHARD-INVARIANT | YES — **movable (in plan: tree0)** | Pure constant columns. |
| `generate_prog_slot_preprocessed(program)` (`:1498`) | `program.slot` (positional 0..size) | SHARD-INVARIANT | YES — **movable (in plan: tree0)** | Slot index is positional, not witness. |
| `generate_pc_in_prog_preprocessed(&rows, padded_rows, n_gates)` (`:1484`) | `rows[i].pc` (positional), n_gates | **SHARD-INVARIANT (looks per-shard)** | YES — **movable (in plan: tree0)** | Takes `&rows` but only reads `pc`, which is the positional counter `0..k*n_gates-1` per shot (set by `row_idx`, independent of witness). `pc % n_gates` is a fixed per-row pattern → identical across shards. The flagged "looks per-shard but isn't" case. |
| `generate_rc_preprocessed(rc_lo_index)` / `(rc_hi_index)` (`:1476`, called `:2161`-`2162`) | rc indices (built once) | SHARD-INVARIANT | YES — **movable (in plan: tree0)** | Pure copy of pos/val columns. |
| `tagged.sort_by_key` + `tree_builder.commit` (tree0) (`:2163`-`2166`) | the pp columns above | SHARD-INVARIANT | YES — **movable (in plan: tree0)** | The ENTIRE tree0 commit (Merkle over preprocessed) is identical across shards; this is the known "base tree0 / preprocessed root reuse". |
| `pack_public_claim(&[])` + mix (`:2168`-`2169`) | empty | SHARD-INVARIANT | YES — trivially movable | Empty public claim; constant mix. |
| **Tree 1 small_main pt.1**: `generate_multiplicity_trace(counts.qdecode/rc_lo/rc_hi)` (`:1518`, called `:2176`-`2178`) | per-shard `counts` | PER-SHARD | n/a | Multiplicity columns depend on reads. Inherent per-shard. |
| **Tree 1 small_main pt.2**: `generate_program_witness(&program)` (`:1506`, called `:2179`) | `program` op/multiplicity cols | **SHARD-INVARIANT (trapped)** | YES — **movable-partial (in plan: program-witness gen)** | The 5 program columns are identical across shards. But they are appended into `small_main` and committed inside the PER-SHARD tree1 Merkle commit, so only the column *generation* is reusable; the commit is not. |
| `gpu_flat_inputs` for main (`main.rs:1920`, called `:2185`) | gates / cases / rc indices | SPLIT | YES (partial) — see notes | Returns `(gates_flat, x_states, off_lo, off_hi)`. `gates_flat`, `off_lo`, `off_hi` are SHARD-INVARIANT (gates + rc offsets). `x_states` is PER-SHARD. **Called AGAIN at `:2222` for interaction** → invariant parts flattened TWICE per shard on host. Cheap host work but redundant. |
| `gpu_gen_main_trace_device` (`gpu_tracegen.rs:1300`, called `:2187`) | gates_flat, x_states, off_*, shape | SPLIT — mostly per-shard | YES (the constant subset) — **movable (in plan: GPU constants + PTX)** | `compile_ptx(GATE_SIM_KERNEL)` + `load_ptx` (`:1331`-`1332`) is SHARD-INVARIANT, recompiled+reloaded EVERY call. `htod_copy(d_gates/d_off_lo/d_off_hi)` (`:1338`,`1340`,`1341`) SHARD-INVARIANT, re-uploaded every call. The K0/K1 sim + `d_cols` build + histograms are PER-SHARD (x_states). |
| `generate_main_trace(&rows,…)` (CPU path) (`main.rs:1385`, called `:2195`/`:2201`) | `rows` cells | PER-SHARD | n/a | Packs witness cells into columns. Inherent per-shard (CPU fallback). |
| `tree_builder.extend_evals` + `commit` (tree1) (`:2205`) | main trace + multiplicities + program witness | PER-SHARD | n/a | Tree1 Merkle root depends on the per-shard main trace → per-shard. (Program-witness columns are invariant but trapped here.) |
| `ProverBackend::grind(INTERACTION_POW_BITS)` (`:2207`) | post-tree1 channel state | PER-SHARD | n/a | Channel diverges after per-shard tree1 commit → nonce per-shard. Inherent. |
| `LookupElements::draw` (`main.rs:132`, called `:2209`) | post-grind channel state | PER-SHARD | n/a | Drawn from per-shard channel → per-shard challenges. Inherent. |
| `gate_air_cuda_kernel::register()` (`:2213`) | none (external constraint-kernel registration) | SHARD-INVARIANT | YES — movable (minor) | External crate. Called once per shard; registration is process-global and almost certainly idempotent → only first call does real work, but flag to confirm it's not re-doing PTX/module work internally. |
| `gpu_tracegen::gate_air_relation_m31x4(&elements.state)` + `set_gate_air_relation` (`:2214`-`2215`) | drawn `elements` | PER-SHARD | n/a | z/alpha derived from per-shard challenges. Inherent. |
| `gpu_flat_inputs` for interaction (`:2222`) | gates / cases / rc indices | SPLIT (redundant) | YES — see notes | SECOND call per shard. Invariant parts (`gates_flat`/`off_*`) recomputed; could share with the `:2185` call within a shard. |
| `gpu_gen_interaction_device` (`gpu_tracegen.rs:1434`, called `:2223`) | gates_flat, x_states, off_*, shape, `elements` | SPLIT — mostly per-shard | YES (the constant subset) — **movable (in plan: GPU constants + PTX)** | RE-COMPILES BOTH kernels: `compile_ptx(GATE_SIM_KERNEL)`+load (`:1477`-`1478`) AND `compile_ptx(INTERACTION_KERNEL)`+load (`:1559`-`1560`). RE-UPLOADS `d_gates/d_off_lo/d_off_hi` (`:1483`,`1485`,`1486`). **Also RE-RUNS the full K0+K1 gate simulation** (`:1504`-`1545`) to rebuild `d_cols`, duplicating the main-trace sim already done in `gpu_gen_main_trace_device` for the SAME shard. The K4 logup pass itself is per-shard (uses `elements`). |
| `gen_main_interaction` (CPU path) (`main.rs:1544`, called `:2236`/`:2240`) | `rows`, `elements`, n_gates | PER-SHARD | n/a | Combines witness rows with per-shard challenges. Inherent (CPU fallback). |
| `gen_table_interaction` ×4 (qdecode/rc_lo/rc_hi/program) (`main.rs:1821`, called `:2244`-`2286`) | per-shard `counts`/`program.multiplicity`, `elements` | PER-SHARD | n/a | Denominators use per-shard `elements`; multiplicities are per-shard (program.multiplicity is invariant but its interaction denom uses per-shard `elements`). Inherent. |
| `public_boundary_sum(shard_cases,…,&elements.state)` (`main.rs:1849`, called `:2289`) | shard cases x/y, per-shard `elements` | PER-SHARD | n/a | Per-shard boundary; sanity cross-check. Inherent. |
| claimed-sums mix (`:2294`-`2295`) | per-shard sums | PER-SHARD | n/a | Inherent. |
| **Tree 2** extend + `commit` (`:2305`-`2322`) | per-shard interaction columns | PER-SHARD | n/a | Inherent. |
| `build_components` (`main.rs:1245`, called `:2324`) | shapes + `elements` + 5 sums | PER-SHARD | n/a | `TraceLocationAllocator`/`preprocessed_column_ids` part is shape-only (invariant), but every component embeds `elements.clone()` + its per-shard claimed sum → per-shard overall. Cheap. |
| `prove_ex` (called `:2335`) | per-shard commitment scheme, components, channel | PER-SHARD | n/a | The FRI/STARK prove. The dominant inherent per-shard cost. |
| shard_boundary build (`:2343`-`2348`) | shard cases x/y | PER-SHARD | n/a | For the leaf statement. Inherent. |

### After the closure (per-shard-wrap / once-only aggregate)

| step | tag | notes |
|---|---|---|
| `derive_aggregate_config` (`main.rs:2440`; `leaf.rs:122`) | SHARD-INVARIANT | ALREADY correctly factored out — called ONCE for all shards using shard 0's shape (`cfg`/`shape_params`). Not in the closure. |
| `build_gate_air_leaf_circuit::<NoValue>` shape (`leaf.rs:86`) | SHARD-INVARIANT | Used only inside `derive_aggregate_config` (already once-only). |
| `cfg` = `ProofConfig::new(...)` (`:2423`) | SHARD-INVARIANT | Built once. |
| `wrap_leaf` → `proof_from_stark_proof` + `prove_gate_air_leaf` (`:2464`-`2477`) | PER-SHARD | Each shard's distinct base proof → distinct leaf (commits its own x/y). Inherent. |
| `recursive_aggregate_prove[_streaming]` / `prove_root_verification` (`:2516`,`:2578`) | PER-RUN (folds all leaves) | Inherent recursion work. Root verification runs once over all leaves. |

## Movable set — the COMPLETE list of shard-invariant-but-currently-per-shard work

### Already in the plan
1. **Base tree0 (preprocessed commit + root).** All of `generate_qdecode_preprocessed`,
   `generate_prog_slot_preprocessed`, `generate_pc_in_prog_preprocessed`,
   `generate_rc_preprocessed`×2, the sort, and the tree0 Merkle `commit`
   (`main.rs:2155`-`2166`). One commit reusable for every shard; gives the single
   trusted `preprocessed_root`.
2. **Twiddles** (`precompute_twiddles`, `:2139`). Same domain every shard.
3. **GPU constant uploads + PTX** — but the walk shows this is BIGGER than a single
   upload (see "amplified" note below): per shard the code does
   - `compile_ptx(GATE_SIM_KERNEL)` + `load_ptx` **twice** (main `:1331`, interaction `:1477`),
   - `compile_ptx(INTERACTION_KERNEL)` + `load_ptx` **once** (`:1559`),
   - `htod_copy` of `d_gates`/`d_off_lo`/`d_off_hi` **twice** (`:1338`/`:1340`/`:1341` and `:1483`/`:1485`/`:1486`).
   All shard-invariant; all redone every shard (and twice within a shard).
4. **Program-witness column generation** (`generate_program_witness`, `:2179`) — invariant
   but PARTIAL: the columns are reusable, the tree1 commit they feed is not.

### NEW movable computations found by the walk (not previously enumerated)
- **N1 — `build_program_table` per shard (`:2128`).** The full program table
  (slot/op cols/multiplicity) is identical across shards (`multiplicity = shard_samples*k`,
  constant). Currently rebuilt every shard. *Full precompute* (build once, capture by
  ref). Saving: small CPU (table is `n_gates`-sized) but feeds both tree0 (slot) and the
  program witness — building it once removes redundant work for both. Note the top-level
  `program` at `:1976` uses full `samples` (wrong multiplicity), so it can't be reused as-is;
  a `shard_samples`-multiplicity table must be the one cached.
- **N2 — Redundant intra-shard GPU main-trace simulation.** `gpu_gen_interaction_device`
  re-runs the ENTIRE K0+K1 gate sim (`gpu_tracegen.rs:1504`-`1545`) to rebuild `d_cols`,
  even though `gpu_gen_main_trace_device` already produced exactly that buffer for the same
  shard moments earlier. This is PER-SHARD data (so not movable *across* shards), but it is a
  duplicated heavy GPU computation *within* each shard. Saving: ~one full main-trace sim pass
  per shard (likely the second-largest GPU cost after FRI). Fix is to hand the device buffer
  from the main-trace call to the interaction call instead of regenerating. Flagging because
  the precompute phase touches exactly this code and should not miss it.
- **N3 — `gpu_flat_inputs` invariant outputs flattened twice per shard** (`:2185`, `:2222`),
  and the invariant subset (`gates_flat`, `off_lo`, `off_hi`) re-flattened on every shard.
  Cheap host work, but trivially hoistable to a once-built struct. *Full precompute* for the
  invariant fields; `x_states` stays per-shard.
- **N4 — PTX compile amplification (sub-item of #3 but worth its own line).** NVRTC
  compilation of `GATE_SIM_KERNEL` and `INTERACTION_KERNEL` is genuinely expensive
  (hundreds of ms each) and there is NO module cache: every device-function call recompiles
  and reloads. With `n_shards` shards that is `3 * n_shards` NVRTC compiles + module loads
  where 2 (one per distinct kernel) would suffice for the whole run. Movable to a process-once
  `OnceLock<CudaModule>` (the device handle is already cached this way at `gpu_tracegen.rs:53`,
  the modules are not). Highest-confidence pure win.
- **N5 — minor: shape scalars + `leaf_pcs_config` + initial channel mix** (`:2131`-`2147`)
  recomputed per shard. Negligible cost; list for completeness, not worth a dedicated phase.
- **N6 — `gate_air_cuda_kernel::register()` per shard (`:2213`).** External; almost
  certainly idempotent (process-global registration) so likely a no-op after first call —
  but confirm it does not internally recompile/reload on each call (if it does, it joins N4).

### Per-shard-inherent (NOT movable — listed so the precompute phase doesn't touch them)
- `build_rows`/`simulate_shot` and the `LookupCounts` reduce.
- `generate_multiplicity_trace` (qdecode/rc_lo/rc_hi).
- `generate_main_trace` (CPU) / the K0+K1 sim half of `gpu_gen_main_trace_device` (GPU).
- `grind` nonce, `LookupElements::draw`, `set_gate_air_relation`.
- All interaction-trace gen (`gen_main_interaction`, `gen_table_interaction`×4, GPU K4),
  claimed sums, `public_boundary_sum`.
- Tree1 + Tree2 Merkle commits.
- `build_components`, `prove_ex`, shard boundary.
- Leaf wrap (`proof_from_stark_proof` + `prove_gate_air_leaf`), recursion fold, root verification.

## Bottom line
The known movable set (tree0 / twiddles / GPU constants+PTX / program-witness gen) is
**directionally complete but under-specified**, and the walk surfaces concrete additions:
- NEW full-precompute items: **N1** (`build_program_table` once), **N3** (`gpu_flat_inputs`
  invariant fields once), **N4** (PTX/module compile+load process-once — the cleanest GPU win).
- NEW intra-shard dedup (not cross-shard, but a large redundant GPU pass): **N2** (stop
  re-running K0/K1 in `gpu_gen_interaction_device`; reuse the main-trace device buffer).
- Confirm **N6** (`register()`) is idempotent.

The "GPU constants + PTX" line in the plan must be expanded: it is not one upload but
**3 PTX compiles + 6 constant uploads per shard** (because each of the two device functions
redoes them, and one does it for two kernels), all reducible to a once-per-process setup.
