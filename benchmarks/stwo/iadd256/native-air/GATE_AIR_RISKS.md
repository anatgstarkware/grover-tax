# gate_air.rs — native gate-simulator AIR: status & flagged risks

Reversible-gate-circuit simulator AIR (proves gate-by-gate execution of a
{NOP,NOT,CNOT,TOFFOLI} circuit over a 512-qubit / 16-bit-limb state). Built as
`src/bin/gate_air.rs` (~1150 lines, self-contained); the existing
`native-iadd-air` binary is untouched.

## Status (as of first implementation)
- Builds clean: `cargo +nightly-2025-07-14 build --release --bin gate_air` (zero warnings).
- **Rust trace self-check PASSES** (no proving): k4-n16, parses 2547-gate GTV1,
  simulates K=4 over 16-bit limbs, final state == fixture `y_hex` for all 16 samples.
  Confirmed load-bearing (comparing vs `x` correctly errors). → encoding, limb
  state, GTV1 parse, and gate semantics are correct.
- **Proving NOT yet run** (laptop rule → run on stwo-vm). The risks below surface
  only at prove time.
- Layout: 191 columns/row. 4 LogUp relations: `StateElements` (chain-lookup state
  threading, width 34 = shot_id+pc+32 limbs), `QDecodeElements` (q→(limb,pos,mask)
  512-row table), `RangeLo`/`RangeHi` (combined (pos,value) dynamic-RC tables, 2^16).
  191 = enabler(1)+opcode one-hot(4)+shot_id,pc(2)+in/out limbs(64)+3 read blocks(117)+helpers ab,fire,delta(3).

## Flagged risks (verify before / during VM proving)

1. **Dynamic-RC table padding (most important).** T_lo and T_hi are flattened to
   2^16-row (pos,value) preprocessed columns; each has exactly 2^16−1 valid tuples,
   and the one remaining row is padded with a genuine member `(pos=0, value=0)`.
   Sound only if LogUp soundness tolerates a duplicate/extra supply row (multiplicity
   counts real reads vs table rows; the extra (0,0) is an unused term). Review if
   stwo LogUp expects distinct table rows. (iadd's SEQ tables pad with sequential
   values, but those are single-column identity tables — different case.)

2. **Main-component LogUp batch pairing order.** `evaluate` emits 11 relation
   entries (2 state + 3 qdecode + 3 rc_lo + 3 rc_hi) via `finalize_logup_in_pairs()`;
   `gen_main_interaction` hand-builds the matching 6 columns (5 pairs + 1 tail). The
   pairing order MUST match emission order exactly, else claimed-sum mismatch at
   prove time. Most likely failure point. Consider a relation-tracker assertion run.

3. **Constraint degrees.** Highest-degree: write constraint
   `out = in + lsel_t·delta·mask` and split `L = hi·2·mask + bit·mask + lo` are
   degree 3 (mask is a witness column, not a constant); fire/delta XOR ≤3.
   `max_constraint_log_degree_bound = log_n_rows + 1` (blowup 2) matches iadd;
   degree-3 fits blowup 2. If the composition-poly degree check trips, bump blowup.

4. **`delta` representation.** `delta = new_t − t_bit ∈ {−1,0,1}` stored as M31
   (−1 → p−1); pinned algebraically by `delta = fire·(1−2·t_bit)`. No range check
   needed. Believed sound.

## RESOLVED — proves + verifies on stwo-vm (AVX-512)
k4-n16 N=1: proved=true, prove_s≈0.04s, verify_s≈0.001s (2^14 rows, 191 cols).
- Risks 1 (RC pad), 2 (pairing order), 3 (degrees), 4 (delta): all checked sound;
  `assert_constraints_on_trace` (env `GATE_AIR_ASSERT=1`, cheap localizer left in)
  confirms every component's constraints hold.
- Two real bugs found & fixed, both prover/verifier-wiring (NOT constraints):
  (a) pre-flight cross-check must be `main_sum + qdecode_sum + rc_lo_sum + rc_hi_sum
      == boundary` (the main interaction carries the lookup *use* terms; tables carry
      the supplies; they cancel, leaving the state boundary);
  (b) verifier must reconstruct the main claimed sum as
      `v_main = v_boundary − v_qdecode − v_rc_lo − v_rc_hi` (it had used v_boundary,
      diverging Fiat-Shamir + the OODS composition value → DEEP-ALI).

## Scale measurements (stwo-vm, AVX-512; k1-n9024, K=1, sweep N; rows = N*2547, 191 cols)
| N | rows | padded | prove_s | verify_s |
|---|---|---|---|---|
| 16 | 40752 | 2^16 | 0.038 | 0.003 |
| 64 | 163008 | 2^18 | 0.090 | 0.006 |
| 256 | 652032 | 2^20 | 0.258 | 0.013 |
| 1024 | 2608128 | 2^22 | 1.243 | 0.018 |
| 4096 | 10432512 | 2^24 | 5.289 | 0.024 |
~linear in rows (~0.5 us/gate-sim @ 2^24); verify ≤24ms. Gate-level (1 row/gate)
vs Cairo qsim's ~70 steps/gate → ~70x fewer rows; 10.4M gate-sims proved in 5.3s.
(N=9024/2^25 not captured — likely mid-run or memory at that size.)

## Still open for full apples-to-apples
- Circuit COMMITMENT (Blake2s(program)=public H) — binds *which* program; not yet added.
- Program-CONSISTENCY: op columns are free witness per row; nothing forces the K passes
  (and N shots) to use the SAME single program, nor to match a committed program.
  Fix couples with the commitment: commit the single n_gates program + constrain
  executed op[pc] == program[pc mod n_gates] via a lookup.
- Scale measurement: chart rows/gate cost vs the Cairo qsim (~70 steps/gate) and SP1.
