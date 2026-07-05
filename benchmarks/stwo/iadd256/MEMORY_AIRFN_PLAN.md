# gate_air RW-memory redesign — project plan (base shrink ~2× + leaf adapter)

> Filename is legacy ("AirFn"); v2 is authored in the **stwo-circuits circuits DSL**, NOT air_infra. See "Correction"
> below. Branch: `anatg/qubit-mem-gate-air` in /home/anat/workspace/stwo-circuits (off origin/main).
> **Implementable component spec → `QUBIT_MEM_SPEC.md`** (chain-lookup, same-row).

**Goal:** replace gate_air's qubit-read encoding (32×16-bit limbs + 32-wide `lsel` one-hot + bit-extraction, ~180 of
188 cols) with a **read-write qubit-memory argument** (address+timestamp offline-memory-checking). Two payoffs:
1. **~2× base-AIR shrink** (collapse ~180 cols → ~3 address columns + memory lookups) → t_base ~20→~10s → the biggest
   algorithmic lever toward beating SP1 (<1×). See RECURSION_PLAN.md "Path to beat SP1" Lever A.
2. **The leaf adapter, free-by-construction** — the `circuit_verifier` `CircuitEval` component you author IS the
   in-circuit gate_air verifier the multiverifier consumes (not free-by-codegen — hand-authored — but it falls out of
   the same work).

**Fair-comparison basis:** SP1's Tanuj benchmark (tanujkhattar/zkp_ecc) also hides the circuit (private stdin, SHA256
public, 9024 FS cases) — so closing the ~2.49× gap is a fair, real goal.

## Correction (2026-07-04): v1 (air_infra AirFn + codegen) was WRONG
- **`blake_gate` in stwo-air-infra is DEPRECATED.** The efficient blake was rewritten in the **circuits DSL in
  stwo-circuits** (`crates/circuits/src/blake.rs`, landed PR #492 "Switch to the new blake circuit"). See
  [[reference_circuits_dsl_locus]].
- **The air components in stwo-circuits ARE code_gen'd from AirFns in stwo-air-infra** (`crates/airs/src/circuit/`) via
  `air_code_gen` (`generate_stwo_circuits`). The "created by the AIR team" header = generated file. Confirmed
  correspondences: stwo-circuits `circuit_verifier/components/qm31_ops.rs` ← air_infra `airs/src/circuit/qm31_ops.rs`;
  same for `range_check_16`, `m_31_to_u_32`, `blake_g_gate`, `triple_xor`, bitwise_xor subroutines. No Cargo dep because
  the generated code is committed INTO stwo-circuits, but air_infra IS the source. (Also uses stwo-cairo pieces:
  `BLAKE_SIGMA`, `Seq`/preprocessed plumbing, `relation!`.)
- **The leaf adapter is free-by-CODEGEN.** `air_code_gen` emits the `circuit_verifier` `CircuitEval` (the in-circuit
  verifier the multiverifier consumes) alongside the prover components — so authoring the gate_air air component(s) as
  AirFn(s) yields the leaf. The generator is `air_code_gen` in stwo-air-infra, NOT external.
- **Division of labor (blake pattern):** the high-level circuit (gate program) is a hand-written **circuits-DSL** circuit
  in `stwo-circuits/crates/circuits/src/` (like `blake.rs`) that EMITS gates; each gate TYPE is backed by a code_gen'd
  air component. So gate_air = a DSL driver (`gate.rs`) + whatever air components the memory argument needs (existing:
  arithmetic gates, `range_check_16`, `Permutation`; possibly NEW: a memory AirFn in air_infra, code_gen'd). "Mostly
  circuits DSL, a few airs from air_infra." **How much of F2 is existing-gate-only vs needs a new memory AirFn =
  the key remaining unknown (Phase 1.1/1.2).**

## Mechanism (v2 — the circuits DSL)
- **The circuits DSL = a gate-list circuit builder** (gnark/Circom-style: wires + typed gates), NOT an AIR eDSL. Atom =
  a wire (`Var{idx}`, `context.rs:29`) holding a QM31. `struct Circuit { n_vars, add, sub, mul, pointwise_mul, eq,
  triple_xor, m31_to_u32, blake_g_gate, permutation, output }` (`circuit.rs:369`). Every gate declares `uses()`/
  `yields()` (`circuit.rs:19`) driving a LogUp bus; each wire yielded exactly once (`check_yields`, `circuit.rs:462`).
- **Authoring = a builder over `Context`** (`context.rs:60`) via `ops::{add,sub,mul,pointwise_mul,eq,permute,guess,
  output}` (`ops.rs`) + the `eval!` macro (`ops.rs:26`). `guess`/`U16Wrapper::guess` = prover nondeterminism; a guessed
  `U16` is auto range-checked to [0,2^16) (`context.rs:230`). One builder fn both defines constraints and generates the
  witness (two instantiations via `IValue`: `Context<QM31>` witness, `Context<NoValue>` topology).
- **Template = `crates/circuits/src/blake.rs`** (typed wrappers + `fn foo<Value: IValue>(ctx, …) -> …` emitters pushing
  gate structs; `blake_test.rs` = the build→finalize→prove→verify harness).
- **LogUp / range-check / permutation all exist:** GATE relation is already an `(address, value)` bus with per-wire
  multiplicity (`stark_verifier/constraint_eval.rs:144` `add_to_relation`; read/write example =
  `circuit_verifier/components/qm31_ops.rs:161` — positive uses on operand addrs + negative-mult yield on dst).
  `RangeCheck_16` component (`circuit_verifier/components/range_check_16.rs`). First-class `Permutation` gate
  (`circuit.rs:239`, `ops::permute`) with a bus arithmetization (`circuit_common/preprocessed.rs:73`
  `fill_permutation_columns`) — the template for a sorted companion. Offline memory checking precedent =
  `cairo_verifier/components/memory_address_to_id.rs`.

## Design decision: F2 (address+timestamp RW memory), addresses/opcodes as WITNESS — CONFIRMED 2026-07-04
Two levels were considered; **F1 is ruled out by the secret-circuit constraint** (same reason arithmetic AIR was ruled
out — [[project_iadd_recursion_gate_level_secret]]):
- **F1 (rejected):** state = 512 wires, each gate = pure arithmetic on specific wires (NOT `t'=1−t`, CNOT `t+c−2tc`,
  TOFFOLI `t+c1c2−2t·c1·c2`). 100% DSL-native, zero new components — BUT the gate program lives in the circuit
  **topology**, which is public → **leaks the secret gate circuit**. F1-with-hiding collapses into F2 (a universal
  circuit selecting ops from witness).
- **F2 (chosen):** target/control selection is a runtime, **witness-driven** memory access, so the gate program stays
  hidden (only H_P committed, matching the SP1/Tanuj model). Hits the ~180→~3-column collapse.

## Design (RESOLVED 2026-07-04) — 3 components sharing a new `QubitMem` relation
Template = air_infra `crates/airs/src/circuit/qm31_ops.rs` (`TraceType::Gate`; self-managed `add_lookup_term`
Use/Yield). `felt252_id_memory` is OFF the circuit/leaf path (it's the cairo casm codegen job) — conceptual precedent
only. Two brief-assumptions corrected: (a) the folding challenge already exists — `add_to_relation` in codegen folds the
whole tuple `(relation_id, addr, ts, value)` with the Fiat-Shamir challenge from `common_lookup_elements`, so NO manual
packing (the AIR-level fix that the DSL couldn't do); (b) `felt252_id_memory` is not reachable from the leaf path.
- **`QubitMem` relation** — new relation (distinct from the shared `Gate` relation), tuple `(addr, ts, value)`.
- **`QubitMemAccess` air** (`TraceType::Gate`, ~12 trace cols/row, 1 row/step) — the per-step read/write emitter:
  witness cols `addr_t,addr_c1,addr_c2, ts_r,ts_w, v_t_old,v_c1,v_c2,v_t_new, is_not,is_cnot,is_tof`. Emits `QubitMem`
  Use(read-old)/Yield(write-new) for target + read-then-rewrite for controls (keeps them in the multiset). Booleanity
  only. Gate-apply arithmetic is NOT here (it's DSL). Depart from qm31_ops: **addresses are TRACE (witness) cols, not
  preprocessed** (hiding) — see OPEN Q1.
- **`QubitMemSort` companion air** (`TraceType::Gate`, ~6 cols, 4N rows) — the sorted-by-(addr,ts) side: matching
  `QubitMem` terms so the two multisets balance (permutation totality replaces the DSL `Permutation` gate, now with the
  challenge folding the tuple); range-checked strict `(addr,ts)` ordering (`d−1`, multi-limb `range_check_variant` sized
  for total accesses); read-after-write consistency on same-addr runs. NO in-repo precedent on the circuit path — the
  novel piece — see OPEN Q2.
- **DSL `gate.rs`** (new, stwo-circuits `crates/circuits/src/`) — orchestration: gate-apply XOR arithmetic
  (`flip=is_not+is_cnot·c1+is_tof·c1·c2`, `t'=t+flip−2·t·flip`, ~5–7 gates/step, sparse for NOP/NOT) + boundary wiring
  (seed all 512 addrs: init Yield `(addr,0,x_i)`, final Use `(addr,MAX,y_i)`, x_i/y_i as public wires) + pushing the
  `QubitMemAccess`/`QubitMemSort` structs (mirrors `blake.rs` pushing `blake_g_gate`). Untouched-addr `x_i=y_i` falls out
  of RAW on single-value runs (seed all 512).
- **Trace cost:** hot per-step row ~12 cols vs 188 (~15× on the dominant component); + the 4N-row sort table (the main
  new cost). Net **~1.5–2× total cell shrink** (~188 → ~90–130 cells/step worst-case TOFFOLI, far less for NOP/NOT).
  Confirms Lever A is worthwhile but at the lower end — combined with kernel tuning (~1.4×) → ~2.1–2.8× vs the 2.49× gap
  (beating SP1 <1× needs both levers, and this is borderline; a firm number needs the GPU port + measurement).

## DECISION (2026-07-04) — hand-write the components (no codegen); optimize for smallest trace
Q1/Q2 resolved: **stwo-air-infra codegen does NOT support these (trace-keyed relation + neighbor-row/range-check sorted
companion), so write the components MANUALLY** — the stwo AIR `Eval` (prover), the witness gen, AND the `circuit_verifier`
`CircuitEval` (the leaf, hand-written like cairo's `memory_address_to_id`). Not free-by-codegen; the trade is full
control over the trace, which we USE to shrink it below the ~2×-only estimate.

### Trace-shrink levers (the previous ~1.5–2× was inflated; these push well past it)
1. **Gate-apply as AIR CONSTRAINTS, not DSL gates (the biggest lever).** The prior design put ~10 arithmetic ops/step
   (flip, XOR, booleanity) into the DSL as separate `qm31_ops` rows ≈ 40–120 cells/step. Hand-written, gate-apply lives
   as a degree-≤4 CONSTRAINT in the access component (`v_t_new = v_t_old ⊕ flip`, flip from same-row cols) — **free in
   columns** (constraints add degree, not trace). This alone removes the dominant per-step cost.
2. **Timestamp representation (user, 2026-07-04): a >31-bit ts is held in TWO M31 limbs** (`ts_hi`,`ts_lo`), compared
   lexicographically — M31 is 31 bits so a ~37-bit global ts cannot fit one limb. The LogUp folds the whole tuple
   `(addr, ts_hi, ts_lo, value)` into ONE field element via the challenge regardless of tuple length, so the interaction
   cost is unchanged (1 term). **Shot separation = a PREPROCESSED `shot_id` in the tuple** (`QubitMem(shot_id, addr,
   ts_local, value)`) — matches the baseline gate_air, which already has a preprocessed `gate_shot_id` and already
   reveals the circuit SIZE (via `gate_pc_in_prog`/`gate_shot_id`; SP1-fair) while keeping CONTENT witness. With `shot_id`
   partitioning tuples, per-shot `ts_local` (reset each shot, ~14 bits) is **1 M31 limb for ANY shard size** — no
   global counter, no 2-limb case. Each shot re-seeds its 512 boundary rows. (Per-shot ts WITHOUT a discriminator would
   be unsound — chains mix; `shot_id` is the discriminator.) See `QUBIT_MEM_SPEC.md` §2/§6/§10.
3. **Narrow, packed sort row.** Sort row = `addr`, `ts_hi`, `ts_lo`, `value` (foldable into the lookup) + `is_write` +
   ~1–2 delta range-check limbs ≈ **~5 cols** (was ~6; ~4 with per-shot 1-limb ts). Ordering = range-check the
   lexicographic `(addr, ts_hi, ts_lo)` delta.
4. **Variable access count.** The sort table is a FLAT list of all accesses, so NOT/CNOT/NOP steps emit fewer rows
   (`2 + #controls`), not padded to 4. Worst case (all-TOFFOLI) is 4N; typical mixes are less.

**Revised estimate:** access component ~N rows × ~12 cols (gate-apply free) + sort component ≤4N rows × ~3–4 cols. In
committed col-rows: ~12N + ~16N ≈ **~28N vs the wide AIR's 188N ⇒ ~5–7× on the memory/gate part** (to be confirmed by a
real count; the 4N-row sort table is the floor). Combined with kernel tuning this comfortably clears the 2.49× gap —
IF the count holds. Still leaves OPEN items below.

## BLOCKER + PIVOT (2026-07-04, found during spec-writing) — leaf is SAME-ROW only
The leaf verifier trait `CircuitEval`/`ComponentDataTrait` (`stark_verifier/src/constraint_eval.rs:39,217`) exposes ONLY
`trace_columns() -> &[Var]` — one row's columns, NO neighbor-row / row-offset access (confirmed: no `next_trace_mask`/
mask-offset anywhere in the circuit/leaf path; cairo's `memory_address_to_id` is READ-ONLY/single-assignment, address=
`Seq`, so it never compares consecutive rows either). **A sorted-companion offline-memory-checking argument REQUIRES
consecutive-row comparison (ordering + read-after-write) ⇒ NOT expressible in the leaf, and we won't patch the shared
verifier** ([[feedback_no_patching_shared_verifier]]). So the sort-companion design is infeasible on the recursion path.

**PIVOT — CHAIN LOOKUP (confirmed 2026-07-04; the established name for this predecessor-referencing pattern).** Already
used + proven-sound in production cairo components: **pedersen, poseidon, and the cairo blake OPCODE** (`blake_compress_
opcode` — NOT the blake circuit). So it's not a novel/risky construction — mirror those. air_infra has chain-lookup
primitives (`chain_lookup_call`, `chain_id_intermediate`, the `anatg/chain_lu*` branches) but we HAND-WRITE the
generated-component shape. Mechanism (predecessor-referencing): each memory access carries an explicit
pointer to its predecessor at that address — columns `(addr, ts, prev_ts, value_before, value_after)`. Ties via ONE
`QubitMem` LogUp: every access YIELDS `(addr, ts, value_after)` (mult 1) and USES its predecessor `(addr, prev_ts,
value_before)` (mult 1); multiset balance ⇒ every write used exactly once ⇒ forces the real linear chain (skipping a
write leaves it unused → imbalance). A SAME-ROW range-check `ts − prev_ts − 1 ≥ 0` enforces per-address ordering / breaks
cycles. Boundary: init yields `(addr,0,x_i)` (used by first access), final uses `(addr, ts_last, y_i)`. **Removes the
separate 4N-row sort companion entirely** — one component, N step-rows, each emitting the ≤4 access Use/Yield pairs +
gate-apply constraint, all same-row ⇒ works in the leaf AND is smaller than the sort-companion design. Needs soundness
review (chain linearity from mult-1 balance + ts ordering; boundary; per-address predecessor correctness).

## OPEN QUESTIONS (resolve in soundness pre-review / before the build)
- **Q3 — value tie.** The gate-apply result `v_t_new` must be the SAME cell the access component Yields to memory (now
  trivial — both are in one hand-written component, so it's one column; no DSL/air boundary to bridge).
- **Q4 (soundness) — permutation totality + RAW + boundary.** access-Yields ↔ sort-Uses must balance exactly (all ±1,
  every addr seeded init+final so untouched ⇒ x_i=y_i); RAW on same-addr runs; strict `(addr,ts)` ordering. The offline-
  memory-checking core — the #1 review target. (ts-sizing now bounded per-shot by lever 2.)
- **Q5 — is a separate access component needed, or can gate-apply + access be ONE component?** Likely yes (one component,
  N rows, emits the ≤4 access lookup terms AND holds the gate-apply constraint) + the sort component. Confirm in design.

## ★ PHASE 1 — DONE & INDEPENDENTLY VERIFIED (2026-07-04)
Chain-lookup qubit-memory re-encoding implemented in grover-tax-v02/gate-air-leaf (branch
`anatg/gate-air-qubit-mem`, NOT committed): base `GateEval` (`main.rs`) + CPU witness + leaf `CircuitEval`
(`circuit_statement.rs`) + new `BoundaryTable` component. **`TRACE_COLUMNS` 188 → 20** (dropped 64 in/out limbs +
117 ReadCols + `TAG_STATE`; added per-qubit `TAG_QUBITMEM` chain). Verified on the laptop (debug, small fixtures):
self-check `final state == y`, `"proved":true`, verify Ok — for k1-n4 (1 shot, log_rows=12) AND k2-n4 (2 shots,
log_rows=14, exercises per-shot boundary). Both base + leaf mirrors updated; leaf-mirror unit test passes.
SOUNDNESS REVIEW (2026-07-04): mostly SOUND (shot separation, boundary balance/completeness, flag-gating, booleanity/XOR,
base↔leaf term-for-term equivalence, hiding — all verified sound), with TWO holes:
- ★ HOLE #1 — MUST-FIX (blocks representative scale, both soundness AND completeness): `RC_TS_POS=15` caps the ts-diff
  range-check at `[0,2^15)`, but `ts_local` resets per SHOT and increments across all k reps (`main.rs:484,509`) ⇒ ts
  reaches ~`3·k·n_gates` (k=1000 → ~7.6M ≫ 2^15). So **the new encoding only works for k≲4**: at representative k the
  HONEST range-check fails on large same-addr gaps (completeness), AND with no absolute ts bound a malicious prover with
  ≥~2^16 same-addr accesses can forge a wraparound (mod p) balanced cycle detached from init/final (soundness — a wrong
  proof that verifies). FIX (DECIDED 2026-07-04) — **per-address `+1` counter** (better than the range-check considered):
  `ts` becomes a PER-ADDRESS sequential index (`ts = last_ts[addr]+1`); replace the ts range-check with a flag-gated
  degree-1 equality `flag·(ts − prev_ts − 1) = 0`. A detached cycle then needs `Σ1 ≡ 0 (mod p)` ⇒ loop length p ≈ 2^31 ⇒
  infeasible at ANY k [WRONG — see verdict below]. REMOVES the ts-rc lookups + rc_lo. BONUS: closed-form ts ⇒ thread-
  per-EXECUTION. Applies to CPU `main.rs` + leaf + GPU.
  ★★ SOUNDNESS VERDICT (adversarial review 2026-07-04) — **THE `+1` COUNTER IS UNSOUND for general circuits. My "uncondi-
  tionally sound" claim was WRONG.** The `+1` counter is only PER-ADDRESS and is NOT tied to program order, so it lost the
  CROSS-ADDRESS ordering the removed range-check provided. Concrete verified attack: program `NOT B; CNOT ctrl=B target=A`
  — a prover gives B's control-read a LOWER per-address ts than B's NOT-write, so the read observes B's PRE-NOT value ⇒
  publishes `A_final = xA⊕xB` (wrong-order result, a DIFFERENT function) and the multiset balances for all inputs. Also
  HOLE: untouched qubits — their `y` is unconstrained witness (forge (0,0) or (1,1)); not triggered by iadd256 (touches all
  512) but general. Also a padding completeness concern (boundary emits unconditionally). NOTE: the x/y-BINDING (base
  change) IS sound and TS_FINAL aliasing is DEFENDED — the break is in the memory argument UNDER it. FIX: replace the `+1`
  counter with a timestamp TIED TO PROGRAM ORDER (derive ts from the preprocessed `pc` + slot, so per-qubit access order
  follows program order — a free/per-address ts is reorderable) + the multi-limb range-check for chain ordering; ANCHOR
  untouched qubits (init `+[s,a,0,x]` tied to a committed input); gate padding out of the boundary emission. Still closed-
  form (ts=pc-derived) ⇒ thread-per-execution survives. This fix is itself SOUNDNESS-CRITICAL ⇒ needs its own review.
  ⇒ The base proof is UNSOUND as implemented; steps 2-3 (recursion) on it are premature — fix the memory argument FIRST.
  (Earlier byte-identity/self-check "validation" only covered the HONEST witness; it doesn't catch the malicious forgery.)
  ★★ FIX IMPLEMENTED + RE-REVIEWED SOUND (2026-07-04). `ts = pc*3 + slot` (PIN `active·(ts−(pc·3+slot))=0`, pc
  preprocessed/verifier-pinned) + RANGE `d=ts−prev_ts−1 ∈ [0,2^25)` ⇒ PIN+RANGE force a UNIQUE forward linear chain per
  (shot,addr) ⇒ reads observe the program-order-last write. Adversarial re-review attacked all 6 surfaces (reorder,
  2-cycle/backward-edge, prev_ts-skip, inactive-inject, untouched, cross-shot) — NO passing-but-false proof found; the
  `+1` hole is CLOSED. Untouched qubits anchored (x/y tied to public-input-bound bits); padding gated by preprocessed
  `gate_bnd_enabler`; composes with the x/y binding unchanged. Verified box-free: build/test/self-check/GATE_AIR_ASSERT
  (all constraints + LogUp balance) PASS. CAVEATS (operational, not holes): (1) completeness ceiling honest `d<2^25` ⇒
  `k·n_gates < ~11.2M`/shot (Tanuj k=2000→5.09M, ~2.2× margin) — adding a LOUD release `bail!` guard; (2) boundary `Use`
  has no range-check (sound today; re-verify if boundary encoding changes). COST: range-check as bit-decomp ⇒ TRACE_COLS
  20→95 (188→95 ≈ 2× shrink, NOT the 9× of the unsound 20-col).
  ★ rc-table refactor DONE (2026-07-04): range-check via LogUp rc-table (d = rc_lo[15]+2^15·rc_hi[10], table enumerates
  EXACT ranges [0,2^15)+[0,2^10) ⇒ NO slack ⇒ d∈[0,2^25) exactly, <p). **TRACE_COLUMNS 95→26** (188→26 ≈ 7.2× shrink; the
  2 limb cols/access are inherent). New relation `TAG_RC`; `RcTable` component wired base+leaf (list now [main,program,
  boundary,rc]); loud release `bail!` guard (fires k≥5000, not k=4000; Tanuj k=2000 fine). Verified box-free: build/test/
  GATE_AIR_ASSERT(main+program+boundary+rc)+cross-check PASS, full prove+native-verify OK, and **in-circuit verify OK ⇒
  the "Variable N unused" panic is CLEARED, leaf/recursion path functional.** Ordering re-review (SOUND) carries over
  (impl swap only). Fingerprint changed ⇒ Phase-2 re-validation on box.
  ★ SOUND t_base RE-MEASURED (26-col, stwo-vm 96-core CPU release, 2026-07-05): **2^22 = 1.94s** (tracegen 0.999 + prove
  0.936 + verify 0.008), 2^23 = 4.11s (~2.1×), proved+verify, cols=26. Only ~10% above the unsound 20-col (1.76s) — the
  soundness fix was nearly free on perf. 188→26 ≈ 7× shrink holds; vs old 188-col ~11-14s tracegen+prove @2^22 on A100.
  CPU curve vs SP1 materially unchanged (preliminary 1-CPU-vs-8-A100 signal); decisive curve = GPU/a2-8g via
  extrapolate_8g.py (A100 anchors). This is the current BANKED sound base result.
  ⇒ STEPS 2-3 UNBLOCKED. GPU K1 TODO: emit the 2 limb cols + rc histogram (26-col layout) before A100 byte-identity re-val.
  ★ SHRINK MEMORY OPPORTUNITY (user, 2026-07-05) — add to the A100 session: the 26-col device footprint is ~7× smaller,
  so (a) **2^24 should fit ALL-RESIDENT (streaming OFF)** — main trace 26×2^24×4 ≈ 1.7GB vs 188-col ~12.6GB (the old 2^24
  NEEDED streaming); resident = simpler + faster. (b) **2^25 likely FITS now** (was PARKED at 188-col): footprint 26×2^25×4
  ≈ 3.5GB, AND the `u32` overflow is gone (`26·2^25≈872M < u32::MAX`; was `188·2^25≈6.3B`). Bigger resident shards ⇒
  fewer shards/folds + better base-bound ⇒ better SP1 curve. A100 session: byte-identity + t_base @2^22 + **@2^24
  streaming-OFF** + **2^25 fit probe** + fold anchors.
  ★★ A100 VALIDATION RESULT (2026-07-05, box force-stopped): CUDA build (cuda,diag) CLEAN first try (transcription
  compiled, 0 fixes). **Byte-identity ALL PASS** — K1 (26 cols + rc hist, 0 mm), K4 (28 interaction, sum_ok), overall
  GPU-tracegen==CPU-tracegen (identical fingerprint, no host-delegate). New 26-col oracles: 2^22=`65f2e97…`, 2^24=
  `38b2b20…`, 2^25=`9e5ba7b…`, small/2^14=`8db7d15…` (replace stale `041394b3`/`ORACLE_FP_2P22`). **FIT (streaming OFF/
  all-resident): 2^24 FITS 18.5GB/40 (streaming now UNNECESSARY); 2^25 FITS 39.6GB/40 (TIGHT, no OOM — UNPARKED, both
  188-col blockers gone).** Fold driver WIRED+runnable (no offline work). ⚠ TIMINGS NOT REPRESENTATIVE — built with
  `diag` ⇒ prove inflated (2^22: tracegen 1.96s good, but prove 22s vs CPU 0.94s / old-188col GPU ~2.8s). NEEDS a clean
  NO-DIAG re-measure of t_base(@2^22/2^24) + fold anchors before the curve (disambiguates diag-artifact vs unoptimized
  transcription kernel). Practical shard = 2^24 (comfortable resident); 2^25 too tight to overlap the fold.
  ★★ CLEAN (no-diag) A100 RESULT + CURVE (2026-07-05): diag WAS the inflation — clean GPU t_base @2^22 = 4.19s (tracegen
  3.15 + **prove 1.03** vs diag's 22s), @2^24 = 7.75s (resident, peak 20.8GB), **@2^25 = 14.30s (MAX shard, resident,
  peak 40264/40960 MiB)**. **2^26 does NOT fit** even with the full streaming set (OOMs at tree2/ifft — device-capacity
  wall, not a missing flag); 2^27 skipped, 2^28 = u32 wall. So MAX = 2^25 resident (vs old 188-col's 2^24-STREAMING —
  bigger shard, no streaming). Fold anchors (A100 host, LOW-contention, 2^23 shards): t_leaf 8.22 / t_node 8.20 / t_root
  ~5-6. NOTABLE: for the narrow 26-col trace, GPU tracegen (3.15s@2^22) ≈/slower than the 96-core CPU (0.999s) — the
  shrink reduced the GPU's advantage (base is now cheap on both). **CURVE (extrapolate_8g, shard 2^25, t_base 14.3s ÷8
  A100):** base_wall @k2000 = 2460s < SP1 2753s ⇒ **base is GPU-bound and BELOW SP1**. Outcome now HINGES on RECURSION
  throughput (leaf/node at the a2-8g's K=16 memory-saturated concurrency): optimistic (low-contention 8.22/8.20) ⇒
  **~0.66–0.90× SP1 (BEAT across the curve)**; realistic (~2.6× saturated 21.4/21.3) ⇒ ~1.1–1.33× (LOSE at high k).
  ⇒ **Beating SP1 (<1×) is PLAUSIBLE but now bottlenecked on the FOLD/recursion, not the base.**
  ★★★ RECURSION SWEEP RESULT (stwo-vm K=16, 2026-07-05): **t_leaf=18.14s / t_node=10.69s @2^23** (K=16/T=6 optimum,
  0.856 leaves/s; K=24 over-saturates t_node→14.8s). KEY FINDING: **the 26-col shrink did NOT help the recursion** —
  leaf/node are FRI/Merkle-verification-bound (~independent of base column count); new 18.14/10.69 ≈ old 15.8/10.7.
  Base_wall ≈ rec_wall at K=16 ⇒ **base and recursion are near-perfectly BALANCED** (~41min each @k2000). BOTTOM LINE
  @k2000 (t_base@2^25=14.3÷8, N=1370, tail=6, SP1=2753s): **[A] leaf/node as-measured (18.14/10.69) ⇒ T=41.2min, 0.90×
  = BEAT SP1 by ~10%. [B] ×1.2 for the 2^25-base leaf (21.8/12.8) ⇒ T=49.5min, 1.08× = LOSE by ~8%.** KNIFE'S EDGE — the
  sole open variable is the 2^23→2^25 leaf/node scaling (FRI query count is ~fixed by security ⇒ real scaling likely
  ~log/×1.05-1.1 ⇒ LEANS BEAT, but UNCONFIRMED). SETTLE via a direct 2^25-shard fold measurement. STRATEGIC: the base
  is solved; the co-bottleneck is now the RECURSION (not helped by the shrink) — a decisive <1× margin would come from
  REDUCING recursion cost (fewer FRI queries / cheaper leaf / fold topology), OR bigger shards (but 2^25 is the max).
  ★★★ RECURSION-COST LEVERS (investigation 2026-07-05): leaf/node are **blake2s Merkle/FRI-decommit-in-circuit bound,
  LINEAR in the FRI query count** of the verified proof (only log in trace size — why the col-shrink didn't help). Two
  query knobs, set ASYMMETRICALLY: **base `BASE_LOG_BLOWUP_FACTOR=1` ⇒ 70 queries (drives t_leaf); outer
  `LOG_BLOWUP_FACTOR=3` ⇒ 23 queries (drives t_node).** The base's 70q is the fat one.
  **#1 (HIGHEST, near-zero-code, soundness-NEUTRAL): raise BASE_LOG_BLOWUP_FACTOR 1→2** (→ base needs only 35q for 96-bit;
  security identity `pow+q·bw≥96` preserved) ⇒ **t_leaf ×0.53** (~18.14→~9.6s @K16). Trade: base eval-domain 2× (base_wall
  up) — but the pipeline is RECURSION-bound (W=22), so trading cheap ×8-GPU base for expensive CPU leaf is strictly
  correct; modeled **W 22→~9**. Already wired (`leaf_pcs_config` bw 1/2/3). CAVEAT: bigger blowup ⇒ 2× device footprint ⇒
  max resident shard drops 2^25→2^24 (more shards) — net needs measuring. Does NOT help t_node.
  #2 outer bw 3→4 (23→18q): marginal t_node cut (diminishing). #3 k-to-1 fold: ~15-25% node wall, medium-high risk
  (topology+unpacker+fingerprint). #4 poseidon-vs-blake: FORBIDDEN (shared verifier, [[feedback_no_patching_shared_verifier]]).
  #5 opening-batch: no headroom (already optimal). ⇒ **Decisive-<1× path = Lever #1** (± #3 for t_node).
  ★★★★ RESULT — **BEAT SP1: 0.915× @k=2000 (CONFIRMED, 2026-07-05).** Direct 2^25 fold measured: leaf/node barely grew
  2^23→2^25 (t_leaf 15.8→18.24, t_node 10.7→11.10 — leaf cost is #queries-bound on base PROOF size, not trace size), so
  the knife's edge landed on BEAT. Curve @k2000: base_wall 2460s ‖ rec_wall 2512s (well balanced), T=2518s < SP1 2753s ⇒
  **0.915×.** The gate_air-shrink + recursion pipeline beats SP1-8×A100 by ~8.5% at k=2000, on the A100-40GB.
  ★ LEVER #1 CORRECTION (the sub-measurement's positive read was wrong — it missed the shard-cap): bw2 DOES halve the
  recursion arm (measured t_leaf 18.96→9.48 ×0.50, t_node ~halves), BUT on the 40GB A100 bw2 = 2× device footprint ⇒
  2^25-bw2 (eval 2^27) does NOT fit even streaming (same tree2 wall as 2^26-bw1) ⇒ max shard drops 2^25→2^24 ⇒ **2× more
  shards (N 1370→2740) ⇒ base_wall ~doubles (~3950s) ⇒ NET WORSE (~1.44×).** So Lever #1 is NOT a win on 40GB — the
  shard-cap penalty dominates the recursion halving. It WOULD win on an 80GB GPU (bw2-2^25 fits ⇒ N unchanged ⇒ recursion
  halves ⇒ decisive). ⇒ **On the 40GB A100, bw1-2^25 is optimal and already BEATS SP1.** Further margin needs either
  bigger GPU memory (unlocks Lever #1) or a footprint-neutral recursion-cost lever (#3 k-to-1 fold reduces node count
  without shrinking the shard).
  ★ #3 k-to-1 FOLD — SCOPED → **NO-GO (2026-07-05).** Ceiling is only ~**0.896×** (from 0.915×, ~2% marginal): k-to-1
  cuts only t_NODE, but **t_leaf DOMINATES rec_wall** (leaf-wrap ≈1562s vs t_node ≈951s), and `base_wall`=2460s is the
  hard floor — so at any k≥3 rec_wall drops UNDER base_wall and T pins at 2460s (0.896×). k=3 already hits the floor;
  larger k is pointless. Effort medium-high (node generalizes cleanly in `circuit_multiverifier/verify.rs:61`, but the
  UNPACKER `recursive_aggregate/lib.rs:666` hard-codes the 2-child 8-word preimage → k-ary re-derivation + streaming
  scheduler + byte-identity fingerprint re-validation, ~2-4 days). Disproportionate to ~2%. **DEFER.** (Recursion code
  lives on `stwo-circuits@anatg/qubit-mem-gate-air` + `proving-utils/recursive_aggregate`, not the circuit-gui tree.)
  ★ REAL HEADROOM (where the levers actually are, from the scope): **(1) t_leaf** — 62% of rec_wall, the true recursion
  bottleneck (the leaf-wrap is a full STARK-verify circuit) ⇒ GPU-accelerate the leaf-wrap, or reduce N (fewer/fatter
  shards). **(2) base_wall** — the 2460s hard floor that caps everything ⇒ faster base proving, or (on 80GB GPU) Lever #1.
  k-to-1 optimizes the smaller half (t_node) that's about to become irrelevant under the floor.
  ★ k-to-1 PRESSURE-TESTED across full k-sweep incl k=8000 (2026-07-05) → **CONFIRM NO-GO.** Ratio is k-INVARIANT
  (base_wall & rec_wall both linear in N ⇒ rec/base≈1.02 at every k≥500; no favorable large-k regime; k=8000 = same
  0.896× ceiling as k=2000). DECISIVE **floor test:** set t_node→0 (best k-to-1 could ever do) ⇒ rec_wall = leaf_part
  only = 1562s @k2000, STILL below base_wall 2460s ⇒ T pins at base_wall regardless. The one lever that unlocks a
  rec-bound regime (cut t_base) simultaneously makes **t_leaf** the binding term — which k-to-1 can't touch. Effort
  confirmed medium-high (arity-2 hardcoded at recursive_aggregate/lib.rs:325 fold, :384 topology, :663 unpacker 8-word
  preimage → byte-identity re-derivation vs circuit_multiverifier/verify.rs node hash). NEEDLE-MOVER = a **t_leaf lever**.
  ★★★ CORRECTION (2026-07-05) — the k-to-1 NO-GO and the "0.896× floor" above are **WRONG**: they used the
  un-amortized single-shot t_base=14.30 ⇒ base_wall 2460. But the preprocessed trace IS amortized
  (`BaseProverPrecompute` built ONCE, tree0 reused via commit_tree(Borrowed) across shards, main.rs:2309), so
  **t_base_fold ≈ 9.4s and base_wall ≈ 1620s @k2000**. ⇒ the pipeline is STRONGLY rec-bound (rec/base ≈ 1.55×, not
  2%), the base FLOOR is **≈ 0.59×** (not 0.896×), and **k-to-1 is REVIVED (~0.70×)**. The recursion-lever ceiling
  is far higher than the entries above imply. SOURCE OF TRUTH for levers = **LEVERS.md** (this block supersedes the
  k-to-1 verdict above). (0.915× headline unchanged — it's rec-bound, set by rec_wall which was measured correctly.)
  ★ COL-REDUCTION SCOPED (2026-07-05) → **GO, sound.** ts insight CORRECT: `ts = pc·3+slot` is fully determined by
  preprocessed `pc` (zero prover freedom) ⇒ DROP the 3 `ts` witness cols, inline `ts = pc+1` (the +1 preserves the
  pc=0/init-node ts=0 distinctness). Also DROP `v_after` (= v_before+delta, pinned) ⇒ **26 → 22 cols** (safer staged
  first cut = ts-only 26→23). KEEP `ab/fire/delta` (inlining raises degree→blowup penalty); prev_ts + rc_lo/rc_hi stay
  (genuine per-address witness). CORRECTNESS-class change (committed relation equivalent; verifier untouched — leaf
  MainGate is our own DSL mirror, NOT the shared verifier). Files in lockstep: main.rs, circuit_statement.rs,
  gpu_tracegen.rs (K1/K4 COL() remap), evaluate_gate_air.cu (drop PIN, 19→15 algebraic). ~10-12h; needs on-box
  byte-identity re-measure (CUDA laptop-uncompilable). ★ IMPACT NUANCE: est ~8-12% t_base — but base_wall is NOT the
  binding term at k≥500 (rec-bound), so the DIRECT large-k ratio move is only via t_leaf, which is FRI/Merkle-bound
  (~column-insensitive) ⇒ small. REAL value = (a) small-k / crossover regime (GPU-bound there), (b) ~15% less device
  footprint → may help the 2^26-shard lever fit (bigger shard halves N → the actual rec_wall win). Do CPU edit +
  assert_main_constraints locally FIRST, then box for kernels+diff+timing.
  ★ COL-REDUCTION SOUNDNESS-REVIEWED (2026-07-05, adversarial) → **SOUND** (all 5 probes; no wrong-proof-that-verifies;
  not weaker than baseline). (1) same-ts collision (target==addr==ctrl, NO distinctness constraint exists — prover CAN
  set them equal) is SAFE: a stray Yield@ts=pc+1 has no valid consumer (a Use needs prev_ts=pc+1 ⇒ d=−1 ⇒ rejected by
  exact-range rc, no field wrap; else it dangles ⇒ LogUp balance fails). (2) cross-address reorder STILL closed (ts
  strictly increasing in pc; slot only separated distinct addrs within a step). (3) v_after inlining equivalent (was
  pinned). (4) **the `+1` is LOAD-BEARING** — `ts=pc` (no +1) WOULD be unsound (init-node ts=0 collision); `ts=pc+1` is
  correct. (5) program-table + shot_id isolation unchanged. CAVEAT (completeness, not soundness — relayed to impl
  agent): the `build_rows` guard at main.rs:471 (d_max = TS_STRIDE·k·n_gates−1) must switch to the new stride=1.
  ★ 2^26 OOM DIAGNOSED (2026-07-05): the OOM is **tree2 (interaction commit), NOT tree1**. The 28 interaction eval
  cols (each 2^27·4B = 512 MiB @2^26 = ~14 GiB) are held **RESIDENT — streaming is gated to tree1 only** (`poly.rs:532`
  `stream_tree1` fires only for BORROWED cols; tree2's owned `extend_evals` ⇒ false). ~14 GiB tree2 + ~6.5 GiB tree0
  (13 preproc cols resident) + twiddles/Merkle → OOM at 40 GB. Fixes: **Fix A = stream tree2** (extend the proven
  tree1 dehydrate path to the interaction group) — FEASIBLE, effort M, commit-side low-risk; the OPEN item is the
  COMPOSITION-phase peak (tree2 must be resident there ~14 GiB + tiled tree1 + numerators) ⇒ **needs ONE on-box
  measure** to confirm <40 GB + that the extra D2H/H2D doesn't blow t_base. **Fix B = col-reduction** trims ~2 GiB off
  tree2 — ⚠️CORRECTION: col-reduction does NOT reduce tree2 (interaction col count = f(LogUp entry count=13), UNCHANGED
  by ts→pc+1 / v_after drop — those remove no entries; confirmed by soundness reviewer). So Fix B helps t_base (tree1
  + constraint kernel) but does NOT help the 2^26 OOM, which is purely a tree2 problem. Fix A (stream tree2) is the
  only code lever for 2^26 on 40 GB. Fix C (lower blowup) RULED OUT (already at floor =1, 96-bit). **80 GB GPU unlocks 2^26 with NO
  code change** (probably 2^27 with A/B). [CAVEAT to verify: the 2^26 agent cited "188 main cols for tree1" — contradicts
  the col-reduction agent's code-verified 26 cols; likely a misread, doesn't change the tree2 diagnosis since tree1 is
  host-staged ~0-resident at tree2 time.]
  ★★ SYNTHESIS — the levers COMPOUND: 2^26 shards HALVE N ⇒ rec_wall drops ~40% (t_leaf grows sub-linearly, FRI-bound)
  ⇒ pipeline flips back to GPU/base-bound ⇒ THEN col-reduction's ~10% t_base cut moves the ratio DIRECTLY (base_wall
  2460→~2200). Path to push k≥500 ratio below 0.90×: [2^26 via Fix A+B] + [col-reduction], BOTH gated on t_base@2^26 ≈
  2×t_base@2^25 (not PCIe-blown) — the single box-measurement that decides it. This, not k-to-1, is the real lever set.
  ★★★ BOX-MEASURED 2026-07-05 (A100-40GB, 22-col build) — SYNTHESIS ABOVE IS NOW PARTLY REFUTED:
  (a) COL-REDUCTION 22-col VALIDATED on GPU: CUDA compiles, byte-identity ALL PASS (K1/K4/CPU==GPU composition), fast
  path fires, prove+verify OK @2^22/2^23. Fresh 22-col fp = 661decf963d8cfbbf81387476bd8f6bc2cff5c7bedf21c2d9b5a3ed517
  1b1018. **BLOCKER the impl agent MISSED: the CUDA decline-guard `gate-air-cuda-kernel/src/lib.rs` was NOT updated**
  (still 26 cols / 19+7 constraints) → GPU declines → 60× host-delegate; box agent fixed it to 22/15+7 (on laptop WIP +
  synced, NOT committed). Any future AIR shape change MUST update this guard too (5th file).
  (b) t_base@2^25 = **13.95s** (vs 14.30 baseline) = only **~2-3% faster, NOT ~8-12%** — REVISE the estimate DOWN. Reason:
  t_base is dominated by `preprocessed` (4.52s, CPU-side) + `fri_commit` (4.09s), both COLUMN-COUNT-INDEPENDENT; the 4
  dropped witness cols are a small fraction. So col-reduction's t_base value is marginal; its real worth is the ~15%
  device-footprint cut (still relevant only if it helped a fit — but see (c) it doesn't save 2^26).
  (c) **2^26 STREAMING = NO-GO on 40 GB (DECISIVE).** 2^26 OOMs even EARLIER than composition — at the K4 interaction
  alloc `d_inter = 28·2^26 = 7168 MiB` (only 3766 free; pool also hoards ~10 GiB cached-freed after tree1). Measured
  per-phase @2^25: after-tree2 (composition entry) = 26628 MiB used / 5836 free; smi peak 39240/40960 (already at the
  edge). ×2 projection to 2^26: composition-entry resident ≈ **53 GiB (12 GiB OVER the card)**; whole-prove peak ≈ 78 GiB.
  The interaction+tree2+composition floor is UN-STREAMABLE (tree2 can't tile — bit-reversed scattered LogUp −1 index), so
  **Fix A cannot rescue 2^26 on 40 GB.** ⇒ 2^26 (and the halve-N → rec_wall-drop lever) **REQUIRES an 80 GB card**
  (A100-80GB/H100), where it works with ~no code change. On the 40 GB A100 we are near the ceiling at 0.915×.
  ★ NET after box: on 40 GB, the campaign is near-optimal (0.915×). Material further gains need EITHER an 80 GB card
  (unlocks 2^26 → rec_wall drop) OR a **t_leaf lever** (GPU-accelerate the in-circuit leaf-wrap — 62% of rec_wall, the
  one term that dominates every regime). Col-reduction is a sound, correct, small (~2-3% t_base) win to keep, not a
  game-changer. k-to-1 NO-GO. The 22-col WIP + guard fix are UNCOMMITTED on the laptop working tree.
- ~~OPEN #2~~ **RESOLVED — verified in code 2026-07-05.** The x/y ARE bound: leaf `public_logup_sum`
  (`circuit_statement.rs:468-529`) bit-decomposes each 16-bit x/y limb (addr=limb*16+bit, LSB-first), booleanity +
  reconstruction (`Σ bitₚ·2ᵖ == guessed limb`) pin the bits to the SAME limbs that feed the output hash, and returns −B
  over the guessed x/y so `verify`'s `public_logup_sum + Σ claimed_sums == 0` forces guessed==committed per (shot,addr).
  This IS the "Phase-3 fix" the old entry called for — implemented (PHASE-3 re-keyed boundary) + soundness-reviewed. (The
  old line ref 336-341 was stale; code shifted.) H_P (OPEN #3, Fork A) extends this exact mechanism with a program term.
- ~~OPEN #3~~ **H_P PROGRAM COMMITMENT — IMPLEMENTED (Fork A), CPU-validated 2026-07-05; pending box re-baseline +
  soundness review (in flight).** Realization: the program table is a SINGLE term (not two like the boundary), so a
  literal re-key fails; instead it emits TWO paired terms in ONE batch (still 4 interaction cols) — INTERNAL
  `−mult/TAG_PROGRAM` (unchanged, cancels main's demand → program-consistency intact) + PUBLIC `+mult/TAG_PROGRAM_PUB=6`
  (the dangling P_pub). Identity → `main+program+boundary+rc == B + P_pub`; leaf `public_logup_sum` supplies `−P_pub`
  over guessed program → forces guessed==committed; same guessed Vars feed `compute_h_p = blake(program‖nonce)`; output
  `H_i = blake(H_P‖x‖y)`. `mult`/`slot`/padding-op/addr are pinned constants (no forgery); nonce guessed/shard-invariant
  (hiding, binding-inert). CPU: 4/4 tests + ASSERT_ONLY cross-check (`==B+P_pub`) + in-circuit verify + `proved:true`.
  **★ NO GPU CHANGES NEEDED** — the program supply is generated CPU-side (`gen_program_interaction`), NOT in K4 (K4 case 6
  = main's demand, unchanged); main component / kernels / decline-guard untouched → only the base FINGERPRINT changes
  (box re-baseline, no kernel edit). Files: main.rs + circuit_statement.rs + leaf.rs (+388/−100). UNCOMMITTED. Flag to
  soundness review: the two-term/new-tag `TAG_PROGRAM_PUB` deviation from a literal re-key. [ORIGINAL GAP, for context:]
  the committed
  gate_air leaf output is `blake(pp_root ‖ x ‖ y)` (`leaf.rs:117`) — `pp_root` is the base PREPROCESSED root, which is
  **program-INDEPENDENT** (equal pp_root ⇏ equal program; program is witness). ⇒ the end-to-end pipeline does NOT yet
  commit to / bind the secret circuit, and same-program-across-shards is NOT enforced. So the current pipeline proves
  "each shard ran a self-consistent program," NOT "all shots ran the ONE committed secret circuit P" (the benchmark's
  actual claim). H_P is the pinned M3 design (current leaf = M3a stepping-stone). FIX + SOUNDNESS CRUX: leaf output must
  be `H_i = blake(H_P ‖ x ‖ y)` with **H_P CONSTRAINED inside the leaf** to `blake(program_table ‖ nonce)` over the
  program-consistency-LogUp-bound program witness — NOT a free guess (a free H_P is vacuous: prover picks any value).
  Then the native verifier recomputes H_i from the ONE published H_P → same-program for free. Cost: one extra hash/leaf
  (negligible; does not change the 0.915× perf headline, which is a proving-TIME result). This is the load-bearing gap
  for the deliverable's soundness — bigger than any lever.
- NOTE (unpacker): the in-circuit tree reconstruction (`recursive_aggregate/lib.rs:665`) is REDUNDANT (re-derives R,
  which the proof already outputs) ⇒ the unpacker can be O(1) (commit-only; native verifier does the free O(N) recompute)
  = L6, SOUND. But the unpacker↔node preimage byte-identity is a MANUAL contract — keep the golden-fingerprint gate
  mandatory on the k-ary change.
- OPEN #3 SOUNDNESS → **SOUND** (reviewed 2026-07-05, `tasks/a6a61e61…`): all 5 probes pass — internal −mult/TAG_PROGRAM
  still cancels main's demand (program-consistency intact); TAG_PROGRAM_PUB distinct (no self-cancel/double-count);
  guessed program forced==committed by −P_pub (collision arg) and the SAME Vars feed compute_h_p; nonce binding-inert.
  FRAGILE LINE (needs a regression test): the `slot`/`mult`/**padding op·addr** pinning as `context.constant`
  (circuit_statement.rs:468-499) — if padding fields ever became free/shard-varying, a prover could forge H_P without
  touching the balance. Add a test asserting they're constants + TAG distinctness.
- OPEN #A3 — **add a final-proof verify SANITY CHECK** (reframed 2026-07-05 per design decision): the gate-air-leaf
  recursion path PROVES the final root-verification proof (`main.rs:3484`) + checks the fold `recursion_fingerprint`,
  then `return Ok(())` at `main.rs:3530` WITHOUT verifying the aggregated proof. Per-level verification IS embedded
  in-circuit (leaf verifies base; nodes verify children; root-verification verifies the root MV proof) and the mechanics
  have verify tests (`smoke_root_verification`) — but the full pipeline's final proof is never verified in-process.
  → ACTION = **(a) add a `verify(rv)` on the final aggregated proof to the recursion tests** — a sanity check that what
  we produce actually verifies (small). NOT needed: (b) runtime re-derivation of leaf/node/root preprocessed_roots —
  they are verifier-APPROVED CONSTANTS (computed once in setup/tests). Already correct by design: (c) the (x,y,H_P)↔R
  binding STAYS in the unpacker/wrapper (in-circuit) so the proof is self-contained / verifier O(1) — which is exactly
  why L6 is NOT pursued.
PHASE 2 GPU PORT — DONE offline (unvalidated), branches `anatg/gate-air-qubit-mem` (gate-air-leaf + gate-air-cuda-kernel)
+ `anatg/cuda-backend` (stwo-cuda-backend), no commits. Ported K1 trace-gen + K4 interaction (5 pairs) + constraint
kernel (`evaluate_gate_air.cu/.cuh`) to the 20-col encoding. `cargo check` (default) passes; `--features cuda` NOT yet
compilable (no toolkit locally). Issues:
- GPU-BLOCKER (2-line): `main.rs:2532,3222` (cuda-gated) still use removed `&elements.state` → change to `&elements.
  qubitmem` (all 3 relations clone one drawn `GateRel`). Needed for the `--features cuda` build.
- ★ PERF REGRESSION (flag): K1 went thread-per-EXECUTION → thread-per-SHOT (the new per-row ts/prev_ts needs the full
  per-shot running history `last_ts[512]`/`last_val[512]`/`ts_local` threaded across all k reps; the old K0 rep-snapshot
  can't seed it). Parallelism drops `n_shots·k → n_shots` — a HIGH-k regression (exactly the Tanuj-curve regime). Byte-
  correct. Fix = a linear-in-k reseed (K0 snapshots per-rep 512-wide (ts,val)+ts_local); DEFERRED (correctness first).
  MUST address before final SP1-curve numbers.

PRE-BOX FIX LIST: (1) ✓ DONE — ts fix via per-address `+1` counter (CPU main.rs + leaf + GPU gpu_tracegen + evaluate_
gate_air.cu); rc_lo/RcLoTable/TAG_RC_LO/RC_TS_POS all REMOVED (only used for ts). Component list now `[main, program,
boundary]`; trace_columns 20 (unchanged), main interaction 20→16, preprocessed 9→7. (2) ✓ DONE — 2-line `state→qubitmem`.
(3) ✓ VERIFIED (fast checks only, per [[feedback-prover-on-vm]]): build clean, cargo test 4/4 (incl leaf-mirror), self-
check `final state==y` on k1-n4 & k2-n4, cols=20; agent also saw `proved:true`+verify on both (one debug run, pre-nudge).
NEW RISK (box item): **in-circuit leaf verify `GATE_AIR_INCIRCUIT=1` PANICS** `"Variable N is unused but not marked as
unused"` — a proof-SHAPE/wiring issue (component-count/log-size packing after removing rc_lo + the pre-existing UNBOUND
boundary x/y Vars), NOT per-row constraints (unit test passes). Couldn't A/B locally (HEAD is old encoding). Likely
overlaps the Phase-3 x/y-rebind wiring; investigate on box / when wiring Phase-3.
THEN: (4) ✓ DONE — CPU t_base measurement (release, stwo-vm 96-core c4, 2026-07-04): 20-col encoding, **2^22 (k1000,
1 shot) t_base = 1.76s** (tracegen 0.843 + prove 0.910 + verify 0.004); **2^23 = 3.51s** (~2× scaling); rate ~0.42 µs/
row; both proved:true + verify + cols=20. vs OLD 188-col which was ~11–14s tracegen+prove @2^22 on the A100 → big
per-shard shrink from the column reduction. CPU curve vs Tanuj (single 96-core machine, base-only lower bound):
k=1 9.6s (0.52× SP1, faster) but k≥100 ~6–7× SP1 (1 CPU vs 8×A100 + base-only) — a PRELIMINARY/relative signal, NOT the
decider; the real comparison is the GPU/a2-8g curve (tiny t_base × 8-way × recursion overlap) after box validation. VM
force-stopped.
(5) ✓ DONE — A100 box validation (2026-07-04, accumulator-diff SKIPPED per user). Build clean after 2 trivial name
fixes now in laptop source (`gpu_tracegen.rs:1334` `_rc_lo`; `main.rs` 5 call-sites `rc_hi_index`→`rc_lo_index`; the
2 remaining `rc_hi_index` @1819/1834 are the self-consistent fn def, benign — rc_hi dead). **K1 byte-identity PASS**
(col_mismatches=0), **K4 PASS** (col_mismatches=0, sum_ok=true, 16 interaction cols), **overall GPU-tracegen ==
CPU-tracegen** fingerprint `041394b3bdb2c0225276e2b0f295620cbb643825d14882cf8575bfa17e9f26e6` (new-encoding oracle,
small fixture k1-n4 s=1; NOT the retired 188-col `ORACLE_FP_2P22`), proved+verify+cols=20, no host-delegate fallback,
no GPU panics. ⇒ **PHASE 2 GPU VALIDATED** (trace-gen K1/K4 + thread-per-execution rewrite + constraint kernel all
byte-identical to CPU on the 20-col encoding). Box force-stopped.
(6) ✓ DONE + VALIDATED — thread-per-EXECUTION K1 rewrite (K0 rep-boundary states + cnt/ord prepass + closed-form
ts=r·cnt+ord+1; grid n_shots·k). Confirmed by K1 byte-identity PASS on the box.
(7) Phase-3 x/y rebind — CONFIRMED a genuine unbound gap, and (2026-07-04 accounting) **the leaf-only fix is UNSOUND —
proven impossible.** The recursion binding (`statement.rs:109-148`, `validate_logup_sum` `logup.rs:36`) requires
`public_logup_sum(guessed) + Σ claimed_sums == 0`, and it forces guessed==committed ONLY when the bound value is
**yielded-but-UNCONSUMED** in the base (cairo outputs, `eq.rs`). But gate_air's boundary x/y are already CONSUMED inside
the base's zero-sum telescope (`main.rs:2491` `main+program+boundary==0`), so `Σ claimed_sums==0` is fixed ⇒ any boundary
term the leaf adds is a NET addition ⇒ breaks honest completeness (double-count). Also the leaf never sees committed x/y
as field values (only OODS masks), so no leaf-only `public_logup_sum` can reference them. **QED: no leaf-only binding
exists.** REAL FIX (needs a DECISION + soundness re-review): a **base-prover structural change** — leave the boundary
x/y term UNCONSUMED in the base internal sum (like cairo reserved outputs) so `Σ claimed_sums == −(boundary public
term)`, then the leaf's `public_logup_sum` supplies the matching `+` term over the guessed bit-decomposed limbs
(limb=addr/16, bit=addr%16 LSB-first; `ts_last` promoted to reserved/guessed), per (shot,addr). This CHANGES the base
proof (new fingerprint ⇒ Phase-2 byte-identity must be re-established) and touches the GPU boundary-interaction kernel
(`gen_boundary_interaction` `main.rs:1730`). In-circuit `"Variable N unused"` panic clears only once x/y are actually
consumed by a sound binding (box in-circuit run to confirm). DONE safely: corrected the stale/wrong comment
`leaf.rs:105-116` (was claiming x/y bound — deleted whole-state encoding). ⇒ **BLOCKS steps 2/3 until the base-prover
change is decided.**
DECISION (2026-07-04): DO THE BASE CHANGE. Soundness review is NON-BLOCKING (user): spin it once the base change lands,
but run it IN PARALLEL — do NOT gate Phase-2 re-validation or steps 2-3 on it. Commit is the real gate (commit only when
asked), so nothing unreviewed ships. Base change, once done, UNBLOCKS steps 2-3. The chain leaves x/y as DANGLING endpoints (use(x)+yield(y) per addr) — NOT
consumed by the chain; the base's boundary component currently cancels them (net 0). Relocate: make the boundary x/y a
yielded-but-UNCONSUMED public term so it surfaces in the base's claimed_sum/public claim (cairo reserved-outputs style),
then the leaf `public_logup_sum` supplies the matching term over the GUESSED x/y ⇒ balance forces guessed==committed.
Must avoid double-count (the leaf's `BoundaryTable` CircuitEval `circuit_statement.rs:100-125` already VERIFIES the base
boundary over committed OODS values). Implementing CPU base+leaf accounting-first (STOP if unsound); GPU boundary-
interaction kernel (`gen_boundary_interaction` `main.rs:1730`) + box re-validation (fingerprint CHANGES from
`041394b3…`) + a soundness review FOLLOW.

## Phases
1. **Build gate_air F2 ENTIRELY in the circuits DSL + CPU soundness-validate** (stwo-circuits only; no GPU; no new
   AirFns unless a part escalates). One `gate.rs`-style DSL circuit expressing the whole F2 memory argument + gate-apply,
   using existing gate types; prove+verify on SimdBackend. Leaf adapter is the existing components' `CircuitEval`s (free).
   ← scoped concretely below.
2. **GPU port** (stwo-cuda-backend). The ~2× t_base win still needs the CUDA trace-gen + constraint kernels re-ported
   for the new memory components (circuit_prover has SIMD witness gen; the CUDA backend is separate).
3. **Wire the leaf into recursion** — partly free (the `circuit_verifier` `CircuitEval` IS the leaf); replaces the
   current hand-wired `leaf::prove_gate_air_leaf` in grover-tax-v02.

## Risks & decisions
- **DECISION (2026-07-04) — start ALL-in-circuits-DSL, no airs.** The DSL-vs-AirFn split is a trace-EFFICIENCY question
  specific to gate_air. Write F2 in the circuits DSL; escalate a part to a new AirFn ONLY if it's genuinely hard/
  inefficient. `anatg/qubit-mem-airfn` (air_infra) on standby.
- **1.2 FINDING (2026-07-04) — ESCALATION RAISED: the memory-checking core is trace-inefficient in raw DSL.** The
  design IS fully expressible in the DSL with no new AirFn (see full design in conversation/this doc), BUT the offline-
  memory-checking machinery is expensive as raw gates: the `Permutation` gate is SINGLE-WIRE with no Fiat-Shamir
  challenge (`circuit_common/preprocessed.rs:64-107`) ⇒ must pack `(addr,ts,val,kind)` into one wire (Horner) AND re-
  decompose every sorted row via guessed components + per-component range-checks (no wire-decomposition primitive) +
  2× replay of the whole access log through the permutation. Estimate **~60-70 QM31Ops rows + ~45 RangeCheck_16 uses per
  gate step**. Everything collapses into the shared 12-col `qm31_ops` + 1-col `range_check_16` + 4-col `m31_to_u32`, so
  the trace goes from **~188 wide cols × N rows → ~16 cols × ~65·N rows** — i.e. **likely MORE total trace cells than
  the current 188-col AIR (~3-4×), which DEFEATS Lever A's purpose (trace shrink for t_base).** KEY INSIGHT reconciling
  this with "blake is efficient in the DSL": blake's heavy work is done by PURPOSE-BUILT air components (`blake_g_gate`,
  `triple_xor`, code_gen'd AirFns) invoked as single gates — the DSL is the orchestration layer, efficiency comes from
  the right air components. **The analogous move for gate_air = a purpose-built offline-memory-checking AirFn** (single-
  challenge tuple bus + native ts range-check — folds `(addr,ts,val)` into ONE lookup, eliminates packing/re-
  decomposition/2×-replay), with the DSL doing orchestration + gate-apply (~10 cheap gates/step). DECISION (2026-07-04):
  **(A) chosen — escalate the memory core to an AirFn now.** Design the offline-memory-checking component as AirFn(s) in
  air_infra `crates/airs/src/circuit/` (template = `qm31_ops.rs`, NOT the deprecated `blake_gate`; registered in
  `circuit_registry.rs`; code_gen'd into stwo-circuits + yields the leaf `CircuitEval`). A thin `gate.rs` circuits-DSL
  driver orchestrates: emits the per-step accesses (Use the QubitMem relation) + gate-apply (~10 gates/step) + boundary
  wiring to public x/y. "Mostly circuits DSL, a few airs from air_infra." At the AIR level the LogUp challenge IS
  available, so `(addr,ts,val)` folds into ONE lookup (no packing/re-decomposition/2×-replay). Branch:
  `anatg/qubit-mem-airfn` (air_infra) + `anatg/qubit-mem-gate-air` (stwo-circuits).
- **Breaks byte-identity** (new AIR → new proof). Confidence shifts to **soundness review of the memory argument** +
  correctness/verify + a NEW oracle baseline. THE key risk.
- **Hiding constraint:** addresses/opcodes MUST be witness (trace) columns, NOT preprocessed (public) — else the gate
  program leaks. A validation criterion, not just correctness.
- **~2× is not immediate** — Phase 1 gives leaf adapter + CPU AIR + soundness; the t_base win lands after Phase 2 (GPU).
- **Repos:** stwo-circuits (design + components, source of truth) → grover-tax-v02 (leaf wire, Phase 3) →
  stwo-cuda-backend (GPU, Phase 2). (stwo-air-infra is NOT involved.)

## Phase 1 — concrete scope (circuits DSL + CPU soundness-validate; NO box, NO GPU)
Deliverable: a soundness-reviewed gate_air F2 memory circuit that proves+verifies on SimdBackend, plus its
`circuit_verifier` `CircuitEval` components (the leaf adapter). Ordered steps:

- **1.1 DSL primitives study (branches created: `anatg/qubit-mem-gate-air` in stwo-circuits, `anatg/qubit-mem-airfn` in
  air_infra on standby).** Anchors: `circuits/src/context.rs`/`ops.rs`/`circuit.rs` (wires+gates, `guess`, `output`),
  the `Permutation` gate (`circuit.rs:239`, `ops::permute`, `fill_permutation_columns`), `range_check_16`, the GATE bus
  (`add_to_relation`, per-wire multiplicity). Focus: which existing gate types express memory read/write, a sorted
  permutation, range-checked ordering, and 1-bit values — and their trace cost.
- **1.2 Design F2 in circuits-DSL constructs + soundness pre-review.** Express in DSL wires/gates: qubit RW memory (reads
  `+1`/writes `−1` on the address/value bus), the sorted companion via the `Permutation` gate, range-checked strictly-
  increasing `(addr,ts)`, read-after-write, boundary to `x_i`/`y_i` via `output`, and gate-apply (`bt'=bt⊕flip`).
  Estimate the trace cost; **flag any part that is hard/inefficient in the DSL as an escalation candidate.** Soundness
  pre-review BEFORE the expensive build. Top-3: (1) ts strict-monotonic (`d_ts−1` range-checked) + TS_BITS bounds total
  accesses; (2) permutation totality + untouched-address `x_i=y_i`; (3) booleanity + XOR gating. PLUS (4) addresses/
  opcodes are WITNESS (guessed wires), never preprocessed/public (hiding).
- **1.3 Implement all-in-DSL.** New `crates/circuits/src/gate.rs` (`apply_gates` driver + memory argument + boundary) +
  a test harness like `blake_test.rs`. No new air components unless 1.2 flags an escalation. If a part escalates: author
  the minimal AirFn on `anatg/qubit-mem-airfn`, code_gen it in, and note it here.
- **1.4 CPU prove+verify.** Small gate circuit (a few shots) on SimdBackend; confirm correct x→y + passing verify.
- **1.5 Soundness review** (route to the recursion-soundness owner) — the check that replaces byte-identity.
- **1.6 New oracle.** Record the new proof fingerprint as the byte-identity baseline for Phases 2/3.

Next action: 1.2 — design F2 in circuits-DSL constructs (all-DSL), judged by trace efficiency, flagging any part that
warrants escalation to an AirFn; then soundness pre-review before implementation.
