# gate_air recursive-proof pipeline — design

A zero-knowledge proof system for a **secret reversible quantum gate circuit**. It proves, over many
input/output test cases, that a hidden circuit maps each input state to its output state — revealing only a
hash commitment to the circuit and the (input, output) pairs. The base statement is a Circle-STARK AIR
(`gate_air`) proved on GPU; many base proofs are aggregated by an in-circuit multiverifier recursion into a
single proof.

---

## 1. The statement

- **The circuit** `P` is a sequence of reversible gates over `{NOP, NOT, CNOT, TOFFOLI}` acting on
  `N_QUBITS = 512` single-bit qubits. It is a **witness** — only a hash commitment `H_P` is public.
- Gate semantics on a target bit `t` with controls `c1, c2`:
  - `NOP`: `t' = t`
  - `NOT`: `t' = 1 − t`
  - `CNOT(c1)`: `t' = t ⊕ c1`
  - `TOFFOLI(c1,c2)`: `t' = t ⊕ (c1 · c2)`
- **The workload** is a set of test cases ("shots"), each a `(x, y)` pair where `x` is an initial 512-qubit
  state and `y` the final state; each shot runs the full circuit `k` times in sequence (the state threads
  across the `k` repetitions within a shot).
- **What is proved:** for every shot, applying `P` to `x` yields `y`. The public output commits to `H_P` and
  the per-shot `(x, y)`.
- **Hidden vs public:** the gate opcodes and the target/control **addresses** are witness (the circuit is
  secret). The circuit **size** and the per-shot `(x, y)` are public.

---

## 2. Pipeline architecture

```
   workload (shots × k)
        │  shard
        ▼
   ┌──────────┐   base proof (gate_air, §3)        ┌──────────┐
   │ shard 0  │ ─────────────  GPU  ──────────────▶ │ leaf 0   │  in-circuit verify of the base proof (§5)
   │ shard 1  │                                     │ leaf 1   │  → emits H_i = blake(H_P ‖ x ‖ y)
   │   ...    │                                     │   ...    │
   └──────────┘                                     └────┬─────┘
                                                          │  k-to-1 (k=8) multiverifier fold (§6)
                                                          ▼
                                                    ┌──────────┐
                                                    │  tree    │ → root → unpacker
                                                    └──────────┘   (per-shot (x,y) + H_P, zk-hiding)
```

- **Shard.** The `shots × k` executions are partitioned into base proofs of up to `2^25` trace rows each.
- **Base proof.** Each shard is proved by `gate_air` (§3) on the GPU (§4).
- **Leaf.** Each base proof is verified *in-circuit* by a leaf circuit (§5) that re-proves the base proof's
  validity and emits a digest `H_i = blake2s(H_P ‖ x ‖ y)` binding the shard's `(x, y)` and the program
  commitment `H_P` (§5).
- **Fold.** A **k-to-1 (k = 8)** tree of multiverifier nodes aggregates the leaf digests up to a single root (§6).
- **Root / unpacker.** The root proof is verified and unpacked into the public output — the per-shot `(x, y)`
  and `H_P` — with zero-knowledge hiding.
- **Overlap.** Base proving (GPU) and leaf-generation + tree-folding (CPU) run **concurrently** in a streaming
  pipeline; the makespan is the max of the two streams, not their sum.

---

## 3. gate_air — the base AIR (in detail)

`gate_air` proves that one shard's executions apply the (witness) gate program to each shot's `x` and reach
`y`. The 512-qubit state is modeled as a **read-write memory** (512 addresses × 1 bit) whose consistency is
enforced by an **offline-memory-checking chain lookup**.

### 3.1 Trace layout — 22 columns

One trace row per **gate step** (one gate applied to the current state). Each step performs up to three memory
accesses — read target, read control-a, read control-b — and one write (target).

| group | cols | columns |
|---|---|---|
| opcode one-hots | 4 | `is_nop, is_not, is_cnot, is_toffoli` |
| target access | 5 | `addr, prev_ts, v_before, rc_lo, rc_hi` |
| ctrl_a access | 5 | `addr, prev_ts, v, rc_lo, rc_hi` |
| ctrl_b access | 5 | `addr, prev_ts, v, rc_lo, rc_hi` |
| gate-apply | 3 | `ab, fire, delta` |

Per-access fields:
- **`addr`** — qubit index in `[0, 512)` (witness; the secret program selects it).
- **`prev_ts`** — timestamp of the previous access to `addr` (the chain predecessor; §3.5).
- **`v`** / **`v_before`** — the bit value read.
- **`rc_lo, rc_hi`** — two limbs of the range-check witness for the timestamp ordering (§3.5).

The access **timestamp** `ts` and the target's **written value** `v_after` are not stored columns — each is a
fixed expression of preprocessed/committed data, used inline: `ts = pc + 1` (§3.4) and
`v_after = v_before ⊕ fire` (§3.6).

### 3.2 Preprocessed (verifier-pinned) columns

Structural, program-order data — public and fixed by the circuit size, not chosen by the prover:
- **`gate_enabler`** — real-row indicator (`1` on real gate rows, `0` on padding).
- **`gate_shot_id`** — `row / (k · n_gates)`; identifies the shot a row belongs to.
- **`gate_pc`** — `row mod (k · n_gates)`; the per-shot monotone program counter.
- **`gate_pc_in_prog`** — `gate_pc mod n_gates`; the program slot the row addresses.
- **`gate_bnd_shot`, `gate_bnd_addr`, `gate_bnd_enabler`** — the boundary component's positional
  `(shot, addr)` and its real-row enabler.
- **`gate_rc_{pos,val}`** — the range-check supply table (§3.5).
- the program-table slot index (§3.7).

### 3.3 The shared LogUp relation

All lookups run over a **single LogUp relation**; each tuple carries a leading constant **tag** that namespaces
its use without cross-talk:
- **`TAG_QUBITMEM`** — the qubit-memory chain, tuple `(shot, addr, ts, value)`.
- **`TAG_RC`** — the range-check table, tuple `(pos, limb)`.
- **`TAG_PROGRAM`** — the internal program table, tuple `(slot, opcode, target, ctrl_a, ctrl_b)`.
- **`TAG_PROGRAM_PUB`** — a *public* copy of the program table (same tuple, distinct tag) — the dangling program
  term `P_pub` the recursion binds the program commitment `H_P` against (§3.7).
A proof is valid only if the relation's multiset balances (every yield consumed by a matching use), after
adding the base proof's public terms — the boundary `B` (§3.8) and the program `P_pub` (§3.7).

### 3.4 Timestamps tie access order to program order

Each access's timestamp is a fixed expression of the **preprocessed** program counter:

```
ts = pc + 1
```

All three accesses of a step share this one value; `ts` is not a witness column — it is substituted inline
wherever it appears (the chain tuples of §3.5). Because `pc` is preprocessed (verifier-pinned) and strictly
increasing across steps, every address's accesses receive strictly increasing timestamps in program order,
which the prover cannot permute. The `+ 1` keeps the minimum real timestamp at `1`, distinct from the
boundary's initial entry at `ts = 0` (§3.8). Within a step the three accesses target distinct addresses, so a
shared `ts` never collides within any single address's chain.

### 3.5 The chain lookup (memory consistency)

Each memory access references its predecessor at the same address. Per active access the main component emits,
on `TAG_QUBITMEM`:
- a **Use** of the predecessor: `+ (shot, addr, prev_ts, v_before)`,
- a **Yield** of the successor: `− (shot, addr, ts, v_after)` (for reads, `v_after = v_before`, propagating the
  value forward).

Multiplicity `±1` and multiset balance force each yield to be consumed exactly once, so every address's
accesses form a single linear chain. Ordering is enforced by a **range check** on the timestamp gap:

```
d = ts − prev_ts − 1 = pc − prev_ts,   d = rc_lo + 2^15 · rc_hi,   rc_lo ∈ [0, 2^15),  rc_hi ∈ [0, 2^10)   ⇒  d ∈ [0, 2^25)
```

`rc_lo`/`rc_hi` are looked up (`TAG_RC`) against a supply table whose two `pos`-blocks enumerate **exactly**
`[0, 2^15)` and `[0, 2^10)` (no padding slack), so `d ∈ [0, 2^25)` with no over-range values, and `2^25 < p`
(the M31 modulus) so no field wrap. This forces `prev_ts < ts` — a strictly increasing chain. Combined with
the program-ordered `ts` (§3.4), each read observes the **program-order-last** write to its address.

### 3.6 Gate application

On the same row, the target's written value is computed from the read values and the opcode:

```
ab   = v_a · v_b
fire = is_not + is_cnot · v_a + is_toffoli · ab
v_after = v_before ⊕ fire      (via delta = fire · (1 − 2·v_before))
```

`v_after` is used directly in the target's chain Yield (§3.5) rather than stored; a booleanity constraint keeps
it a bit. With booleanity on every read value and opcode flag, and the one-hot constraint
`enabler = is_nop + is_not + is_cnot + is_toffoli`, the gate semantics are fully pinned.

### 3.7 Program table (the secret circuit) + program commitment

The gate program is a **witness** supply table: per slot `(opcode_scalar, target, ctrl_a, ctrl_b)` with a
constant multiplicity `mult = k · n_shots`. It emits **two** terms per slot, paired into one interaction batch:
- an **internal** supply on `TAG_PROGRAM` that cancels the main component's per-execution program *demand* (each
  execution row uses `(pc_in_prog, opcode, target.addr, ctrl_a.addr, ctrl_b.addr)`). Balance forces every
  execution to equal `program[pc_in_prog]` — i.e. all `k · n_shots` executions run the **same** hidden program
  in cyclic order. Program-consistency, closed internally.
- a **public** supply on `TAG_PROGRAM_PUB` (same tuple, distinct tag so it does not self-cancel) that surfaces in
  the committed claimed sums as a dangling term `P_pub = Σ_slot mult · (slot, opcode, target, ctrl_a, ctrl_b)`.

Only the slot index and `mult` are preprocessed/pinned; the opcodes and addresses are witness (hidden). `P_pub`
is what the recursion leaf binds the program commitment `H_P` against (§5) — it exposes the *executed* program
(through the LogUp binding) without revealing it.

### 3.8 Boundary — input/output anchoring and public commitment

A boundary component anchors every `(shot, addr)`:
- the memory chain begins with a use of `(shot, addr, 0, x)` — the initial bit `x`,
- the boundary consumes the chain's final yield `(shot, addr, ts_last, y)` and **re-emits** it at a fixed public
  timestamp `TS_FINAL = 2^30` (chosen above every real `ts`).

The net effect is that the base proof's LogUp sum does not close to zero on its own; it closes to a **public
term** `B = Σ_{shot,addr} ( +(shot, addr, 0, x) − (shot, addr, TS_FINAL, y) )` carried in the base proof's
committed claimed sums. Every one of the 512 addresses is anchored each shot; an untouched address has only
its initial and final entries, forcing `y = x`. Together with the program term `P_pub` (§3.7), this `B` makes
the base proof's total public claim `Σ claimed_sums = B + P_pub` — the terms the recursion leaf binds `(x, y)`
and `H_P` against (§5).

### 3.9 Shot isolation

`shot_id` (preprocessed) is the first element of every `TAG_QUBITMEM` tuple, so a use in one shot can only match
a yield in the same shot. Each shot's memory is therefore an independent chain, even though many shots share one
committed trace.

### 3.10 Components

The AIR is four components over the shared relation: **main** (the per-step gate + chain + range-check + program
terms), **program** (the witness gate-program supply table), **boundary** (the per-`(shot,addr)` anchoring +
public term), and **rc** (the range-check supply table).

### 3.11 PCS / FRI

Circle STARK over M31 with a lifted-Merkle (blake2s) commitment. The base proof uses a low-blowup PCS config at
the target 96-bit security (proof-of-work + FRI query count chosen so `pow_bits + n_queries · blowup ≥ 96`).

### 3.12 Soundness summary

- **Ordering:** program-pinned `ts` (§3.4) + the range-checked strictly-increasing chain (§3.5) + multiset
  balance ⇒ each read returns the program-order-last write; no access reordering is possible.
- **Program fidelity + commitment:** the internal program term (§3.7) forces every execution to run one hidden
  program; the public term `P_pub` binds that *executed* program to the published commitment `H_P` in the leaf
  (§5), so all shards provably run the **same** committed circuit.
- **Boundary:** every address is anchored to the committed input and output (§3.8); the public term `B`
  exposes `(x, y)` for binding.
- **Isolation:** `shot_id` (§3.9) keeps shots independent.

---

## 4. GPU base proving

The base proof is produced on the CUDA backend, resident (no host streaming) up to `2^25` trace rows:
- **K0** — per-shot state simulation producing the 512-qubit state at each repetition boundary.
- **K1** — the main trace generator (thread-per-execution): each thread seeds from its repetition-boundary
  state and fills the 22-column layout, deriving the memory-chain values (with `ts = pc + 1` used inline).
- **K4** — the interaction (LogUp) trace: per-row numerator/denominator fractions for the chain, range-check,
  and program terms.
- **Constraint kernel** — evaluates the composition polynomial (the AIR constraints of §3) on device.
- **Commit** — NTT + lifted-Merkle (blake2s) commitment of the trace and interaction columns; then FRI.

---

## 5. Leaf adapter (in-circuit verifier)

Each base proof is verified by an **in-circuit STARK verifier** expressed in the circuits DSL — the recursion
leaf. It re-runs the base proof's verification (mirroring gate_air's components, the composition check, and the
FRI/Merkle decommitment) inside a proof circuit, and produces the digest the fold consumes.

The leaf guesses the shard's `(x, y)` **and its program table**, and its `public_logup_sum` supplies the matching
public terms so the recursion's balance holds **only if** the guesses equal the base proof's committed claims:
- **`(x, y)`** — bit-decomposed and supplied against the boundary term `B` (§3.8); balance forces guessed `(x, y)`
  == the committed boundary bits per `(shot, addr)`.
- **the program** — supplied against `P_pub` (§3.7) over the guessed `(slot, opcode, addresses)` (with `mult` and
  slot pinned as constants); balance forces the guessed program == the *executed* committed program.

The leaf then forms the program commitment and the output digest over those bound witnesses:

```
H_P = blake2s( program_table ‖ nonce )     (hiding; one shared nonce across all leaves)
H_i = blake2s( H_P ‖ x ‖ y )               (the leaf's public output — an 8-word digest)
```

Because the program is bound to `P_pub` (not a free guess), `H_P` commits to the *executed* circuit — a leaf that
ran a different program yields a different `H_P`. The shared nonce hides the program while keeping `H_P` identical
across shards. So `H_i` provably commits to a genuine `x → y` execution of the one committed circuit `H_P`.

The leaf is padded to its **own natural size**, independent of the fold arity — which keeps the dominant
leaf-wrap cost fixed as `k` grows (§6).

---

## 6. Recursion — k-to-1 multiverifier fold

The leaf digests are aggregated by a **k-to-1 (k = 8) multiverifier tree**. Each node verifies its `k` child
proofs in-circuit and hashes `[pp_root, outs]` for all children (left to right) into its own output — an
**8-word `HashValue`** (the multiverifier root/output format). The tree binds each node's output to its
children's committed outputs, so the leaf digests propagate faithfully to the root.

**Leaf/node size decoupling.** A node verifying `k` children is larger than a leaf, so leaves and nodes are
proved at **separate padding sizes** (each with a PCS sized to its own trace) — the leaf stays small and its
cost is fixed independent of `k`, so raising the fold arity shrinks the node *count* without inflating the
dominant leaf term. A node's cost is polylog in its child, so there are exactly **two node shapes**, each with
its own trusted preprocessed root:
- **R1** — level-1 nodes, which verify *leaves*;
- **R2** — level-≥2 nodes, which verify *nodes* (a fixed point: a node proof has the multiverifier's own shape).

The topology is fixed from the public shard count `N`, so each node's level — hence whether it carries R1 or R2
— is public. R1/R2 are **trusted constants selected by tree level** (not prover-chosen), checked via the same
per-child config the node already uses.

**Root / unpacker.** The root proof is verified in-circuit; the **unpacker** then reconstructs the tree's hash
from the guessed per-leaf outputs (selecting R1/R2 per level, byte-identically to what the nodes emitted), binds
it to the verified root, and exposes the aggregated public data — the per-shot `(x, y)` and `H_P` — with
zero-knowledge blinding. Keeping this binding **in-circuit** makes the final proof self-contained: a verifier
checks one proof and reads its public output, without re-hashing the fold itself.

The result is a single proof whose public output is `(H_P, {(x_i, y_i)})`: the hidden circuit, committed by
`H_P`, maps each public input `x_i` to its output `y_i`.
