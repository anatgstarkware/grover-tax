# CPU recursion pipeline — proof-invariant vs per-proof walk

READ-ONLY completeness pass over the whole CPU recursion pipeline (leaf_prover + multiverifier
node + root verification), all on `SimdBackend`. Goal: is opt#1's precompute (leaf/node tree0 +
twiddles) complete, or are there more invariant computations redone per leaf/node?

Sources walked (laptop, pinned to the exact build):
- `gate-air-leaf/src/leaf.rs` — `derive_aggregate_config`, `build_gate_air_leaf_circuit`, `prove_gate_air_leaf`.
- `gate-air-leaf/src/circuit_statement.rs` — `GateAirStatement` / `MainGate`.
- `proving-utils/crates/recursive_aggregate/src/lib.rs` — `recursive_aggregate_prove`, `prove_node`, `prove_root_verification`, `AggregateConfig`/`CircuitPrecompute`, `preprocessed_root`, `shared_config_for_leaf`.
- circuit_prover @ stwo-circuits rev `041ec61`:
  `~/.cargo/git/checkouts/stwo-circuits-acca305c7c7af9d7/041ec61/crates/circuit_prover/src/prover.rs`
  (`prove_circuit_assignment`, `prove_circuit_with_precompute`), `.../witness/trace.rs`
  (`write_trace`, `write_interaction_trace`), `.../witness/components/{verify_bitwise_xor_*,range_check_16}.rs`,
  `.../circuit_air/circuit_components.rs` (`CircuitComponents::new`).
- `circuit_common/src/preprocessed.rs` — `PreprocessedCircuit::preprocess_circuit`, `add_fixed_preprocessed_columns`.
- `privacy_prove/src/lib.rs` — the `RecursiveProverPrecomputes` precedent opt#1 was modelled on.

Legend for the "movable?" column:
- **OPT#1** = already precomputed by `CircuitPrecompute` (tree0 + twiddles) — do not re-flag.
- **MOVABLE** = proof-invariant but still recomputed per leaf/node → candidate.
- **INHERENT** = per-proof by nature (depends on the witness), cannot be hoisted.

---

## A. Config derivation (`derive_aggregate_config`, leaf.rs:122) — runs ONCE per campaign

| step (fn + file:line) | inputs | tag | recomputed per leaf/node? | notes |
|---|---|---|---|---|
| `build_gate_air_leaf_circuit::<NoValue>` shape (leaf.rs:127) | cfg, params (no witness) | PROOF-INVARIANT | no — once | Circuit STRUCTURE (topology) of the leaf. Built only here for sizing. |
| fixed-point loop `pad_to_targets`+`preprocess_circuit`+`multiverifier_node_preprocessed` (leaf.rs:131-142) | NoValue shapes | PROOF-INVARIANT | no — once | Resolves `target_padding_sizes`/PCS/node sizes. Already one-shot. |
| `preprocessed_root(leaf_pp)` / `(node_pp)` (leaf.rs:144-146, lib.rs:780) | leaf/node `PreprocessedCircuit` | PROOF-INVARIANT | no — once | Interpolate + commit tree0 to get the root. Folded into OPT#1's tree (built again inside `CircuitPrecompute::new`, asserted equal). |
| `shared_config_for_leaf` (leaf.rs:149, lib.rs:804) | leaf_pp, pcs | PROOF-INVARIANT | no — once | `ProofConfig`/`SharedConfig` (incl. `preprocessed_column_log_sizes`). Lives in `AggregateConfig`. |
| `CircuitPrecompute::new` ×2 (leaf.rs:161-171, lib.rs:157) | leaf_pp/node_pp, pcs, root | PROOF-INVARIANT | no — once | **OPT#1**: tree0 (interpolate + Merkle-commit of preprocessed cols) + twiddles + base_column_pool, for leaf and node shapes. |

Config derivation is fully one-shot already; nothing per-proof here.

---

## B. Per-leaf prove (`prove_gate_air_leaf`, leaf.rs:187) — runs ~N times

| step (fn + file:line) | inputs | tag | movable? | notes |
|---|---|---|---|---|
| `build_gate_air_leaf_circuit::<QM31>` (leaf.rs:193) | real_proof (witness), cfg, params | **split** | MOVABLE (structure) / INHERENT (assignment) | The circuit is rebuilt from scratch per leaf, but its TOPOLOGY (gate list / `Context` ops emitted by `GateAirStatement` + `verify`) is identical across leaves — only the GUESSED witness values (`preprocessed_root`, `boundary`, the base proof Vars) differ. Today structure+assignment are fused (`Context` records ops and values together); only the `context.values()` slice is genuinely per-proof. **Caching the op-list and re-emitting only values is a candidate (see C1).** |
| `pad_to_targets` (leaf.rs:194) | context, target sizes | PROOF-INVARIANT (amount) | folds into C1 | Padding targets are constant; the padding work rides on the structure rebuild. |
| `prove_circuit_with_precompute` (leaf.rs:198, prover.rs:105) | values, OPT#1 precompute, pcs | per-proof core | — | Entry to the prover. Sub-steps below. |
| ↳ channel setup + `pcs_config.mix_into` + `commit_tree(preprocessed_tree)` (prover.rs:130-145) | salt, pcs, **cached tree0** | mixed | tree0 = OPT#1 | The expensive tree0 commit is skipped (borrowed from precompute); only the cheap channel mix of the already-built root remains — INHERENT but trivial. |
| ↳ `write_trace` (prover.rs:149, trace.rs:39) | `context_values` (witness), preprocessed_trace, twiddles | **mostly INHERENT** | partial (see C2) | Generates + interpolates all 11 component base traces. Witness-dependent. **But** the per-component `ClaimGenerator::new(...)` for the 5 const-size xor tables + range_check_16 build a witness-independent `input_to_row` map from the FIXED preprocessed XOR/seq columns — see C2. |
| ↳ `verify_bitwise_xor_{4,7,8,9,12}::ClaimGenerator::new` → `make_input_to_row` (trace.rs:96-119, prelude.rs:102) | FIXED bitwise-xor preprocessed columns only | **PROOF-INVARIANT** | **MOVABLE (C2)** | HashMaps keyed by the constant XOR tables: sizes ~2^8 (xor_4) + 2^14 (xor_7) + 2^16 (xor_8) + 2^18 (xor_9) + 2^((ELEM_BITS-EXPAND_BITS)*2) (xor_12). Rebuilt on EVERY prove inside `write_trace`. Identical for leaf AND node AND root (the fixed tables don't even depend on the circuit shape). The `mults` columns they accompany ARE per-proof; only the index map is invariant. |
| ↳ `interpolate_columns` of each base-trace component (trace.rs:84-216) | per-proof traces, twiddles | INHERENT | — | Witness traces → polys. Twiddles already OPT#1. |
| ↳ `grind(INTERACTION_POW_BITS)` + draw interaction elements (prover.rs:162-164) | channel state | INHERENT | — | Fiat-Shamir; depends on committed base trace. |
| ↳ `write_interaction_trace` (prover.rs:168, trace.rs:322) | lookup data (witness), interaction_elements, twiddles | INHERENT | — | LogUp interaction columns + interpolation. Witness + transcript dependent. |
| ↳ `CircuitComponents::new` + `TraceLocationAllocator` (prover.rs:182, circuit_components.rs:18) | interaction_elements, claimed_sums, log_sizes, `preprocessed_column_ids` | **split** | partial (C3) | Builds the 11 component provers. The `TraceLocationAllocator::new_with_preprocessed_columns(ids)` + the per-component `Eval` structs are keyed on the INVARIANT preprocessed-column id list and constant log sizes; only `claimed_sums` / drawn `common_lookup_elements` are per-proof. Cheap relative to NTT/FRI but structurally invariant — minor candidate C3. |
| ↳ `prove_ex` (composition poly, quotients, FRI, query openings) (prover.rs:191) | committed trees, channel | INHERENT | — | The cryptographic core: composition polynomial eval over the witness, DEEP/quotient, FRI folding, Merkle query paths. All witness/transcript dependent. Dominant cost. |
| `prepare_circuit_proof_for_circuit_verifier` (leaf.rs:217, prover.rs:202) | circuit_proof | INHERENT | — | Repackages the proof; `ProofConfig::new(all_circuit_components, …)` here is invariant but ~free. |

---

## C. Per-node prove (`prove_node`, lib.rs:722) — runs ~N-1 times

Identical prover path to B, so the same rows apply. Node-specific structural steps:

| step (fn + file:line) | inputs | tag | movable? | notes |
|---|---|---|---|---|
| `build_multiverifier_circuit::<QM31>(input(a), input(b), shared_config)` (lib.rs:730) | child proofs a,b (witness), shared_config | **split** | MOVABLE (structure) / INHERENT (assignment) | Same as the leaf: the node circuit TOPOLOGY is fixed (`shared_config` + the multiverifier verifying two children of fixed shape); only the two children's proof Vars are per-node witness. Rebuilt every node ⇒ same C1 candidate, node-side. |
| `pad_to_targets` + `validate_circuit` (lib.rs:732-733) | context | invariant amount | folds into C1 | `validate_circuit` is a per-build O(circuit) sanity pass over the (invariant-shape) graph — redundant work per node once the shape is trusted (C4). |
| `prove_with_precompute` → `prove_circuit_with_precompute` (lib.rs:735/707) | values, **node** OPT#1 precompute | per-proof core | — | Same sub-steps as B (tree0 skip = OPT#1; `write_trace` xor maps = C2; etc.). |

---

## D. Root verification (`prove_root_verification`, lib.rs:595) — runs ONCE per campaign

| step (fn + file:line) | inputs | tag | movable? | notes |
|---|---|---|---|---|
| build verify-root circuit (lib.rs:608-622) | root proof (witness), shared_config | per-proof | INHERENT | One-shot; runs once total, so hoisting buys nothing. |
| O(N) unpack loop reconstructing the tree root (lib.rs:636-671) | guessed per-leaf outputs (witness) | INHERENT | — | Inherently O(N) and witness-dependent (the leaf outputs). |
| `pad_context` + `preprocess_circuit` + `get_pcs_config(trace_log_size,…)` (lib.rs:685-688) | finalized context | per-proof | INHERENT (intentional) | **Confirmed: deliberately NOT precomputed.** The root circuit's trace size GROWS with N (the unpack adds ~N blake nodes + N·N_RESERVED guessed outputs + optional zk-blinding rows), so `trace_log_size` — and therefore the preprocessed trace, tree0, twiddles, and PCS config — varies per campaign. A fixed `CircuitPrecompute` cannot cover a variable shape, and it runs only once, so there is nothing to amortize. The `prove_circuit_assignment` fallback (rebuild tree0) is correct here. |
| `add_zk_blinding` (lib.rs:681) | context, seed | INHERENT | — | Only the root is blinded; one-shot. |

---

## Additional movable set (BEYOND opt#1's tree0 + twiddles)

Recursion is ~half wall-clock; leaf ~2.9 s, node ~3.7 s, ~2N proves total (N leaves + N-1 nodes).
"Saving" estimates are per-prove and multiply by ~2N.

### C2 — `input_to_row` HashMaps for the const-size xor / range-check tables  **[both sides; full; HIGH-confidence invariant, prover-API change needed]**
- **What:** `write_trace` calls `verify_bitwise_xor_{4,7,8,9,12}::ClaimGenerator::new` (and `range_check_16`), each of which calls `make_input_to_row` (prelude.rs:102) to build a HashMap from the FIXED bitwise-xor preprocessed columns: total ~2^8 + 2^14 + 2^16 + 2^18 + 2^((ELEM_BITS-EXPAND_BITS)*2) entries. These tables are global constants — identical across every leaf, every node, AND the root; they do not even depend on the circuit shape.
- **Now:** rebuilt from scratch on **every** `prove_circuit_with_precompute` / `prove_circuit_assignment` call (~2N times).
- **Saving:** building several million-entry hashmaps per prove. Likely a small but non-trivial fraction of the ~2.9 s / ~3.7 s (sub-second, but ×2N it adds up); needs a profile to size precisely.
- **Catch:** this cost lives INSIDE circuit_prover's `write_trace`, which today exposes no hook to inject prebuilt `ClaimGenerator`s — `prove_circuit_with_precompute` only accepts tree0 + twiddles + pool. Realizing C2 requires either (a) a thread-local/`OnceLock` cache inside circuit_prover keyed on the fixed table contents, or (b) threading a precomputed-maps handle through `write_trace`. Since the maps are global constants, a process-wide `OnceLock` (built once, shared read-only) is the least-invasive lever and is backend-independent. This is the **single clearest invariant-but-per-proof cost the prover redoes**.

### C1 — Circuit STRUCTURE rebuild (op-list) per leaf / per node  **[both sides; partial; structurally invariant, large refactor]**
- **What:** `build_gate_air_leaf_circuit` (leaf) and `build_multiverifier_circuit` (node) re-emit the entire `Context` op graph on every prove, then `preprocess`/pad it. The topology is identical across all leaves (resp. all nodes) — only the GUESSED witness values differ (`GateAirStatement` deliberately keeps per-shard data as guessed Vars precisely so the SHAPE is constant, leaf.rs:454-456; comment confirms `leaf_preprocessed_root` is identical across shards).
- **Now:** full rebuild per prove (~2N times). The `context.values()` vector is the only genuinely per-proof output.
- **Saving:** the circuit-build itself is cheap vs. NTT/FRI, so the win is modest; the real point is it would also make the witness-assignment path the only per-proof CPU work, enabling a future "structure once, assign per proof" prover. Lower priority than C2.
- **Catch:** `Context` fuses op-recording and value-recording; separating them (build the gate graph + multiplicities once, run only the value-assignment pass per proof) is a circuits-crate refactor, not a config addition. Tagged partial.

### C3 — Component-prover construction (`CircuitComponents::new` + `TraceLocationAllocator`)  **[both sides; partial; minor]**
- **What:** `CircuitComponents::new` (prover.rs:182) rebuilds 11 `Component` provers with a fresh `TraceLocationAllocator::new_with_preprocessed_columns(ids)`; the id list, log sizes, and per-component `Eval` wiring are invariant — only `claimed_sums` and the drawn `common_lookup_elements` are per-proof.
- **Now:** rebuilt per prove. Cost is small relative to the NTT/FRI core.
- **Saving:** marginal. Lower priority; likely not worth the API churn unless C2/C1 are done and it shows in a profile.

### C4 — `validate_circuit()` per node (lib.rs:733)  **[node-side; debug-class; trivial]**
- **What:** `prove_node` calls `context.validate_circuit()` (an O(circuit) consistency pass over the invariant-shape graph) on every node build; the leaf path does not. Once the node shape is trusted (it is — `node_preprocessed_root` is a fixed point), this is redundant per-prove work.
- **Saving:** small; gate it behind debug/once. Cheap to drop.

### Confirmed NOT movable (per-proof inherent)
- `write_trace` / `write_interaction_trace` base + interaction trace generation and their `interpolate_columns` (depend on `context_values`).
- `grind` + interaction-element draw, composition poly, quotients, FRI, query Merkle openings inside `prove_ex` — the dominant cost, fully witness/transcript dependent.
- The entire root verification (`prove_root_verification`): runs ONCE, and its trace size varies with N, so precompute is both impossible (variable shape) and pointless (no amortization). Correctly left on the `prove_circuit_assignment` rebuild path.

---

## Verdict

Opt#1 (tree0 + twiddles cached in `CircuitPrecompute` for the fixed leaf and node shapes) covers the
**single largest** invariant per-proof cost — the preprocessed-trace interpolate + Merkle commit — and
is correct and complete for what it targets. It is **not the complete set** of proof-invariant work
still redone per prove. The remaining invariant-but-per-proof computations, in priority order:

1. **C2 (highest):** the const-size xor / range-check `input_to_row` HashMaps (~2^8…2^18+ entries each),
   rebuilt inside `write_trace` on all ~2N proves, are global constants identical across leaf/node/root.
   Movable via a process-wide `OnceLock` cache in circuit_prover (backend-independent; needs a small
   prover-side change, not just an `AggregateConfig` field). Profile to confirm magnitude.
2. **C1:** circuit-structure (op-list) rebuild per leaf/node — invariant topology, per-proof values only;
   modest direct saving, enables a cleaner structure/assignment split. Larger refactor.
3. **C3 / C4:** component-prover construction and the per-node `validate_circuit` — minor / debug-class.

Everything else (trace gen, interpolation of witness traces, composition/quotient/FRI, and the whole
once-only root verification) is genuinely per-proof and correctly not precomputed.
