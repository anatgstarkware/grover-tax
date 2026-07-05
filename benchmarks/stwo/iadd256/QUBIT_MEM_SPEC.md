# gate_air qubit-memory — component spec (chain-lookup, same-row)

Companion to `MEMORY_AIRFN_PLAN.md` (design/decisions). This is the implementable spec for the hand-written components.
Locus: stwo-circuits (branch `anatg/qubit-mem-gate-air`). Same-row only (leaf `CircuitEval` has no neighbor-row access);
memory consistency via **chain lookup** (proven pattern — cairo pedersen/poseidon/blake-opcode). NOT air_infra codegen.

## 1. Summary
gate_air applies a SECRET witness program of {NOP,NOT,CNOT,TOFFOLI} to 512 qubits (512 addr × 1 bit). Represent the
qubit state as a **read-write memory checked by a chain lookup**: every memory access carries an explicit pointer to its
predecessor at that address; a single LogUp relation ties each access to its predecessor, and a same-row range-check
enforces per-address time ordering (which forbids detached cycles). Two components:
- **`qubit_mem_step`** — one row per gate step (N rows). Holds the step's ≤3 reads + 1 write as chain accesses AND the
  gate-apply constraint. This is the hot component.
- **`qubit_mem_boundary`** — 512 rows/shot: seeds each address's chain head (init) and tail (final), tying to public
  x/y. Small, fixed.
No sorted companion (the same-row blocker), no separate access/sort split.

## 2. Relation `QubitMem`
Tuple `(addr, ts, value)` (folded with the Fiat-Shamir challenge by `add_to_relation`, so tuple length is free):
Tuple = `(shot_id, addr, ts_local, value)`:
- `shot_id` — **PREPROCESSED** (positional), exactly like today's gate_air `gate_shot_id` (= `row/(k·n_gates)`).
  Partitions tuples per shot so chains cannot mix; a `Use` carries the current row's `shot_id` ⇒ cross-shot references
  are structurally impossible. Reveals nothing beyond the circuit SIZE, which the baseline gate_air ALREADY makes public
  (via preprocessed `gate_pc_in_prog`/`gate_shot_id`; content — opcodes/addresses — stays witness). SP1-fair (SP1 also
  reveals program size via its public cycle count).
- `addr` — qubit index 0..511 (9 bits), witness.
- `ts_local` — per-shot strictly-increasing access counter (resets each shot). ~4·n_gates ≈ 14 bits ⇒ **ONE M31 limb,
  any shard size** (shot_id, not ts, separates shots ⇒ no global counter, no 2-limb case). witness.
- `value` — 1 bit, witness.
- `value` — 1 bit.
Register `QubitMem` alongside the other relation ids (mirror `qm31_ops`'s `m31_gate_relation_id`; distinct from `Gate`).

## 3. `qubit_mem_step` — trace columns (one row = one gate step)
Convention: **all trace columns (addresses, opcodes, values, timestamps included) — required for hiding the program.**
Per row, 3 access slots (target T + control C1 + control C2) and the opcode:

| group | columns |
|---|---|
| opcode | `is_not, is_cnot, is_tof` (NOP = all 0) |
| target T (read+write) | `addr_t, ts_t, prev_ts_t, v_before, v_after` |
| control C1 (read) | `addr_c1, ts_c1, prev_ts_c1, v_c1` |
| control C2 (read) | `addr_c2, ts_c2, prev_ts_c2, v_c2` |
| ordering witnesses | per active access: range-check limbs of `ts − prev_ts − 1` (1 limb/access if per-shot ts, else 2) |

≈ 3 + 5 + 4 + 4 + (~3–6 rc limbs) = **~19–22 trace cols**. (vs 188.)

## 4. Constraints (all same-row, degree ≤ 4) — via `acc.add_constraint(context, eval!(...))`
- **Booleanity:** `x·(x−1)=0` for `is_not, is_cnot, is_tof, v_before, v_after, v_c1, v_c2`.
- **Opcode exclusivity:** `is_not + is_cnot + is_tof ≤ 1` (e.g. pairwise products = 0; NOP = all 0).
- **Gate-apply:** `flip = is_not + is_cnot·v_c1 + is_tof·v_c1·v_c2`; `v_after − (v_before + flip − 2·v_before·flip) = 0`
  (= `v_before ⊕ flip`; NOP ⇒ flip=0 ⇒ `v_after=v_before`).
- **Ordering (acyclicity):** for each active access, `range_check(ts − prev_ts − 1)` ≥ 0 (the `d−1` trick ⇒ strictly
  increasing ⇒ no cycles). Range-check via the shared `range_check_16` bus (multi-limb for a 2-limb ts).

## 5. Lookup terms (chain use/yield) — via `acc.add_to_relation(context, numerator, &[QubitMem_id, addr, ts, value])`
Per access: **USE predecessor (positive numerator) + YIELD successor (negative numerator)**, mirroring poseidon
(`add_to_relation(context, numerator, tuple)`; Use = `+mult`, Yield = `-(mult)`), each **gated by the access's active
flag** so inactive controls contribute 0:
- Target (always active): `Use (addr_t, prev_ts_t, v_before)` mult `1`; `Yield (addr_t, ts_t, v_after)` mult `-1`.
- C1 (active iff `is_cnot+is_tof`): `Use (addr_c1, prev_ts_c1, v_c1)` mult `(is_cnot+is_tof)`; `Yield (addr_c1, ts_c1,
  v_c1)` mult `-(is_cnot+is_tof)` (read propagates the same value forward).
- C2 (active iff `is_tof`): `Use (addr_c2, prev_ts_c2, v_c2)` mult `is_tof`; `Yield (addr_c2, ts_c2, v_c2)` mult
  `-is_tof`.
`relation_uses_per_row = [RelationUse{ relation_id:"QubitMem", uses: 3 }]` (3 positive uses; the 3 yields are negative).

## 6. `qubit_mem_boundary` — per-shot init/final anchoring (512 rows PER SHOT)
Re-seeded EVERY shot (each shot's memory anchored to its own x/y; with the preprocessed `shot_id` partitioning tuples,
this isolates shots). Per (shot s, address i): preprocessed `shot_id=s`, `addr_i` (Seq 0..511); witness `x_{s,i}`,
`y_{s,i}`, `ts_last_{s,i}`. Emits on `QubitMem`:
- **init:** `Yield (s, addr_i, 0, x_{s,i})` mult `-1` (consumed by shot s's first access to addr i; ts_local=0).
- **final:** `Use (s, addr_i, ts_last_{s,i}, y_{s,i})` mult `1` (consumes shot s's last Yield to addr i; untouched ⇒
  `ts_last=0` ⇒ forces `y=x`). `shot_id=s` in the tuple ⇒ a different shot cannot consume this shot's Yields.
`x_i`/`y_i` are public (or committed via a blake2s hash of the 512-bit x and y, made public via `output`). This is the
only place x/y touch the relation.

## 7. Witness generation (prover)
`shot_id` is preprocessed (positional). For each shot s, reset `ts_local := 0` and `last[addr] := (0, x_{s,addr})`:
1. At shot start: for all 512 addr, emit init `Yield (s, addr, 0, x_{s,addr})`.
2. For each step: for each active access to `addr`, read `(prev_ts, prev_val) = last[addr]`, set `ts_local += 1`, fill
   `prev_ts, v_before=prev_val`; for the write compute `v_after` per gate-apply; set `last[addr] := (ts_local, v_after)`
   (a read sets `last := (ts_local, prev_val)` — propagates the value). Fill the range-check limb for
   `ts_local − prev_ts − 1` (1 limb).
3. At shot end: for each addr, `ts_last, y := last[addr]`; emit final `Use (s, addr, ts_last, y)`.
`shot_id` separates shots ⇒ `ts_local` resets safely; ~14 bits ⇒ 1 limb, any shard size.

## 8. LOCUS + edits (CORRECTED 2026-07-04) — evolve the EXISTING gate_air, not fresh stwo-circuits components
The base gate_air is a standalone `FrameworkEval` already in **grover-tax-v02/gate-air-leaf** (`main.rs` base +
`circuit_statement.rs` leaf + CPU witness in `main.rs` + GPU witness in `gpu_tracegen.rs`). It ALREADY uses a chain
lookup: `TAG_STATE` "telescoping" threads the WHOLE 512-bit state (32 limbs) row→row keyed by `(shot_id, pc)`
(`main.rs:848-871`, relation width `GATE_REL_WIDTH=35`). The 188 cols = 4 opcode + 32 `in_limb` + 32 `out_limb` +
3×`READ_COLS`(=39: `q,limb_idx,bit_pos,mask,lsel[32],lo,hi,bit`) + 3 (`ab,fire,delta`). **The redesign REFINES the
granularity: whole-state chain → per-qubit chain.** Branch `anatg/gate-air-qubit-mem` (off the working gate_air). Edits:
- **`main.rs` `GateEval::evaluate` (base):** DELETE `in_limb`/`out_limb` (64), the 3×`ReadCols` lsel/qdecode/rc bit-
  extraction (117), the `TAG_STATE` whole-state Use/Yield, and the qdecode + dynamic-rc(bit) lookups. ADD per access
  (target + 2 controls) chain columns `addr, ts, prev_ts, value` (+ target `v_after`) and a new **`TAG_QUBITMEM`** sub-
  relation (tuple `[TAG, shot_id, addr, ts, value]`, width 5): per active access `Use[+flag](shot,addr,prev_ts,v_before)`
  + `Yield[-flag](shot,addr,ts,v_after)` (controls: `v_after=v_before`). Keep opcode booleanity + gate-apply
  (`fire`/`delta` = XOR) but on the memory `value`s. ADD ts-ordering rc `ts-prev_ts-1` (reuse the RC tables). New
  `TRACE_COLUMNS ≈ 4 + (5+4+4) + 3(ab,fire,delta) + 3(rc) ≈ 23`.
- **New boundary supply** (like the existing Qdecode/RC/Program supply-table components): per (shot, addr) emit init
  `Yield[-1](shot,addr,0,x)` + final `Use[+1](shot,addr,ts_last,y)` on `TAG_QUBITMEM`. `shot_id` preprocessed;
  `x/y/ts_last` witness. Size = n_shots×512 (public).
- **CPU witness (`main.rs` build_* fns):** fill new columns per §7 (per-shot `ts_local`, `prev_ts`, values via
  simulation; `last[addr]`). Retire the limb/lsel/qdecode fill.
- **Leaf `circuit_statement.rs`:** mirror ALL of the above in the `CircuitEval` (the tagged `add_to_relation` for
  `TAG_QUBITMEM`, the gate-apply + ts-rc constraints, the boundary component's supply term). Same-row only (holds).
- **GPU `gpu_tracegen.rs`: DEFER to Phase 2** (Phase 1 is CPU prove/verify only).
- Retire `TAG_STATE` (and qdecode/bit-rc if fully unused); add `TAG_QUBITMEM`. `GATE_REL_WIDTH` 35→ (max tuple width).
- **Test:** a SMALL CPU prove/verify (few shots, small k) — laptop OK (NOT the big shard). Confirm correct x→y + verify.

## 9. Trace cost
`qubit_mem_step` ≈ ~20 trace cols × N rows + interaction cols. Interaction: 6 `QubitMem` terms + ~3–4 range-check terms
≈ 5 logup pairs → ~20 interaction cols. So ≈ **~40 cols × N** vs the wide AIR's **188 × N ⇒ ~4–5×** on the hot part;
`qubit_mem_boundary` is 512 rows/shot × ~5 cols (negligible). No 4N sort table. Firm number pending a real count; the
interaction columns are the main dilutant of the ~8× trace-narrowing.

## 10. Soundness checklist (route to the recursion-soundness owner before/at build)
1. **Chain acyclicity** — the ts strict-ordering range-check (`d−1`) is what forbids detached balanced cycles; mult-1
   balance alone does NOT (a cycle disjoint from init/final balances). ts ordering is load-bearing, not optional.
2. **Shot separation via preprocessed `shot_id`** — `shot_id` is in the tuple and preprocessed (positional), so chains
   of different shots CANNOT mix (a `Use` carries the current row's `shot_id`). This is why per-shot `ts_local` (reset
   each shot, ~14 bits, 1 limb) is safe here. Confirm `ts_local` doesn't wrap over `~4·n_gates` accesses (it won't). The
   preprocessed `shot_id` reveals only the circuit SIZE — already public in the baseline gate_air, SP1-fair; the program
   CONTENT (opcodes/addresses) stays witness.
3. **Boundary completeness** — every one of the 512 addresses seeded init+final each shot (untouched ⇒ `x_i=y_i` falls
   out); mults are exactly ±1; the first access Uses the init Yield, the final Use consumes the last Yield.
4. **Flag-gated multiplicities** — inactive controls contribute 0 (numerator = flag); confirm a gated-off access cannot
   inject a phantom `QubitMem` term, and `v_c*` don't-cares can't affect gate-apply (they're multiplied by 0 flags).
5. **Booleanity + XOR** — all bits constrained; `v_after` fed to the Yield is the SAME column as the gate-apply output
   (one component ⇒ trivially tied; no DSL/air boundary).
6. **Hiding** — addr/opcode/ts are trace (witness) columns, never preprocessed/public (except boundary `addr_i=Seq`).

## Q. Sub-decisions (LOCKED 2026-07-04)
- **Discriminator = preprocessed `shot_id` (LOCKED)** — matches baseline gate_air `gate_shot_id`; per-shot `ts_local` is
  1 M31 limb for any shard. Circuit SIZE is public (baseline already reveals it; SP1-fair); CONTENT stays witness.
- **`ts_local` range-check** = single `range_check_16` limb (~14 bits) on `ts_local − prev_ts − 1`.
- **FUTURE (optional): hide circuit size too** — would need `shot_id`/`pc` as WITNESS with a consistency argument, but
  monotonicity is a neighbor-row constraint (same-row blocker). NOT needed now (baseline reveals size). Separate effort.

## Templates / file anchors (stwo-circuits @ origin/main)
- Leaf `CircuitEval` shape: `crates/circuit_verifier/src/components/qm31_ops.rs`.
- Chain use/yield (gated numerators): `crates/cairo_verifier/src/components/poseidon_full_round_chain.rs`
  (`add_to_relation(context, numerator, tuple)`, `numerator = -(enabler)` for yields; `relation_uses_per_row`).
- Prover `FrameworkEval`: `crates/circuit_prover/src/circuit_air/components/qm_31_ops.rs` (`next_trace_mask` offset-0).
- Leaf trait (same-row): `crates/stark_verifier/src/constraint_eval.rs:39,217` (`trace_columns()->&[Var]`, no neighbor).
- DSL driver template: `crates/circuits/src/blake.rs` + `crates/circuits/src/circuit.rs` (gate-list struct).
