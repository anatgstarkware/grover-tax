# P3.2 — K4: CUDA LogUp interaction — complete spec + plan

GOAL: generate gate_air's interaction (LogUp) trace on the GPU — the 6 logup columns + claimed_sum
— byte-identical to the CPU `gen_main_interaction` / `LogupTraceGenerator`. Second (and last) big
net-new kernel for the on-device pipeline (consumes K1's GPU-resident main trace).

## combine (exact, from constraint-framework logup.rs)
`d = combine(values) = ( Σ_i alpha^i · values[i] ) − z`   in QM31.
- One `GateRel` drawn from the channel: `z, alpha` (2 secure felts); `alpha_powers[i]=alpha^i`, i=0..34.
- ALL relations (state/qdecode/rc_lo/rc_hi/program) share the SAME (z, alpha). values[0] is the TAG.

## The 6 pairs (each → one logup column = (num, denom), num=m0·d1+m1·d0, denom=d0·d1)
m = numerator (an M31 multiplicity → QM31, negated if sign<0); d = combine(values):
- pair0: (m=enabler, d=state_in, +) , (m=enabler, d=state_out, −)
  - state_in  values = [TAG_STATE(1), shot_id, pc,   in_limb[0..31]]   (35)
  - state_out values = [TAG_STATE(1), shot_id, pc+1, out_limb[0..31]]  (35)
- pair1: (enabler, qdecode(target), +), (a_active, qdecode(ctrl_a), +)
- pair2: (b_active, qdecode(ctrl_b), +), (enabler, rc_lo(target), +)
- pair3: (enabler, rc_hi(target), +),    (a_active, rc_lo(ctrl_a), +)
- pair4: (a_active, rc_hi(ctrl_a), +),   (b_active, rc_lo(ctrl_b), +)
- pair5: (b_active, rc_hi(ctrl_b), +),   (enabler, program, +)
  - qdecode(s) values = [TAG_QDECODE(2), s.q, s.limb_idx, s.bit_pos, s.mask]   (5)
  - rc_lo(s)   values = [TAG_RC_LO(3), s.bit_pos, s.lo]                         (3)
  - rc_hi(s)   values = [TAG_RC_HI(4), s.bit_pos, s.hi]                         (3)
  - program    values = [TAG_PROGRAM(5), pc%n_gates, opcode_scalar, target.q, ctrl_a.q, ctrl_b.q] (6)
  - opcode_scalar = is_not + 2·is_cnot + 3·is_toffoli;  enabler/a_active(=cnot+tof)/b_active(=tof).
  All these fields are ALREADY in K1's main-trace columns (in_limb/out_limb/shot_id/pc/the 3 read
  blocks/opcode one-hot) → K4 reads K1's GPU-resident columns; no re-sim.

## accumulation (LogupTraceGenerator, exact)
- per column k (sequential k=0..5, each row parallel): batch-inverse the denoms; frac = num·denom_inv;
  `col_k[row] = col_{k-1}[row] + frac` (running sum ACROSS columns; col_{-1}=0).
- finalize_last (on the LAST column only): claimed_sum = Σ_row col_5[row] (coord-wise → QM31);
  cumsum_shift = claimed_sum / 2^log_size; subtract cumsum_shift from every element; then
  inclusive_prefix_sum ACROSS rows → final col_5. Return all 6 cols (each = SecureColumnByCoords =
  4 M31 cols → 24 M31 interaction columns) + claimed_sum (QM31).
- NOTE only the last column is prefix-summed + shifted; cols 0..4 are the plain running sums.

## kernel decomposition (reuse obelyzk where possible)
- REUSE: M31 device fns (constraints.rs: m31_mul/add/sub/inv), QM31/CM31 fns (quotients.rs:
  qm31_mul_cm31 etc. — extract a qm31 add/mul/sub/from_m31), `batch_inverse_packed_qm31`
  (constraint-framework), a device inclusive-prefix-sum (simd has prefix_sum.rs; check gpu/).
- NEW kernel **K4a (per-row denom/num)**: thread-per-(row or vec_row); for each of the 6 pairs read
  the needed M31 fields from K1's cols, compute d0,d1 (the combine dot products, ≤35-wide using
  alpha_powers in constant mem + z), m0,m1, then (num,denom)=(m0·d1+m1·d0, d0·d1). Output 6×(num,denom)
  QM31 columns. (denom dot product is the cost; alpha_powers/z uploaded after the channel draw.)
- **K4b accumulate**: for k=0..5: batch_inverse(denom_k); col_k = col_{k-1} + num_k·denom_inv_k.
- **K4c finalize**: reduce claimed_sum on col_5; subtract cumsum_shift; inclusive_prefix_sum.
- Output: 24 M31 interaction columns (GPU-resident) + claimed_sum.

## inputs / glue
- z (QM31), alpha_powers[0..34] (QM31) from LookupElements (drawn during prove) — upload to device.
- K1's main-trace columns (GPU-resident) — or, for the standalone byte-identity test, regenerate.
- padded_rows, n_gates (for pc%n_gates), real rows vs padding (padding row: enabler=0 → num=0; d
  still computed from padding fields — matches CPU which runs write_pair over padded rows w/ Row::padding).

## validation (soundness gate)
Byte-identity test: run K4 with a FIXED (z, alpha) (use GateRel::dummy() or a seeded channel) +
compare the 24 interaction M31 columns AND claimed_sum to CPU gen_main_interaction with the same
elements, on k1-n4. Must be exactly equal.

## effort: K4a (combine, QM31 dot products) is the novel core; K4b reuses batch_inverse; K4c needs a
## QM31 reduce + prefix-sum. Mirror K1's structure (kernel string in gpu_tracegen.rs + cudarc glue +
## env-gated byte-identity test). Multi-step but fully specified above.
