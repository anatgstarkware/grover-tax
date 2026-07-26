
#define N_QUBITS 512u
#define N_LIMBS 32u
#define LIMB_BITS 16u
#define M31_MOD 2147483647u
#define OP_NOP 0u
#define OP_NOT 1u
#define OP_CNOT 2u
#define OP_TOFFOLI 3u
#define SLOT_CTRL_A 1u
#define SLOT_CTRL_B 2u
#define SLOT_TARGET 3u

// QUBIT-MEMORY ENCODING (branch anatg/gate-air-qubit-mem) — THREAD-PER-EXECUTION, pc-pinned ts.
// -------------------------------------------------------------------------------------------
// The old whole-state (TAG_STATE, 188/191-col) chain is replaced by a per-qubit chain-lookup
// qubit-memory (TAG_QUBITMEM, 19-col). Each row is ONE gate execution. Per row the CPU
// `simulate_shot` (main.rs) maintains, PER SHOT (reset each shot, threaded across ALL k reps):
//   last_ts[addr], last_val[addr]  — the (ts, value) of the most recent access at each qubit
// An access reads (prev_ts = last_ts[addr], v_before = last_val[addr]), sets the PROGRAM-ORDER
// timestamp ts = pc + 1 (pc = rep*n_gates + gate; SHARED by all accesses of a step, no slot), then
// writes last_ts[addr] = ts (and last_val[addr] = v_after for the target, unchanged for controls). The
// diff d = ts - prev_ts - 1 = pc - prev_ts (>= 0) is a SINGLE 25-bit column (no limb split).
//
// TIMESTAMP-ORDERING (pc-pinned ts + single-`d` rc-table range-check, the FINAL sound design):
//   * ts is PINNED to the preprocessed pc: ts == pc + 1 (AIR constraint). Closed form in (rep, gate) —
//     no running counter.
//   * prev_ts < ts is proved by range-checking the single diff `d = ts - prev_ts - 1` looked up as
//     (TAG_RC, d) into the FIXED rc supply table (d ∈ [0, 2^RC_LOG)).
//
// CLOSED-FORM ts + prev_ts (thread-per-EXECUTION survives): ts is closed form in the pc. prev_ts is
// the closed-form ts of the PREVIOUS access to the same addr in cyclic program order. `prog_slot_meta`
// precomputes, per active slot, (prev_gate, prev_slot, wrap): the (gate, slot) of that predecessor in
// one pass and wrap=1 iff it lies in the PREVIOUS rep. K1 then sets
//     prev_rep = wrap ? rep-1 : rep;  prev_ts = (wrap && rep==0) ? 0 : (prev_rep*n_gates+prev_gate) + 1
// (prev_ts = 0 is the init boundary node — the program-wide first access to that addr). This needs
// only (rep, program constants), NO serial per-address history, so K1 stays embarrassingly parallel.
//
// K0 / K1 SPLIT (thread-per-EXECUTION):
//   K0 `gate_sim_states` — thread-per-SHOT, VALUE-ONLY. Walks the full k*n_gates chain doing only the
//     cheap 32-limb bit-flip (apply_gate, no column I/O) and snapshots the 512-qubit state (as 32
//     limbs) at the START of every (shot, rep) into rep_states. Seeds K1's per-execution value chain.
//   K1 `gate_sim` — thread-per-EXECUTION. Thread `exec = shot*k + rep` seeds a local 512-state from
//     rep_states[exec], processes the rep's n_gates gates updating that local state for v_before/
//     v_after, fills ts (closed form) + prev_ts + the single diff `d` from `slot_meta`+`rep`, and
//     atomically bumps the rc histogram (one bump per active access, over `d`). Grid = n_shots*k.
//
// 19-col cell_at layout (main.rs cell_at, all WITNESS; enabler/shot_id/pc/pc_in_prog are tree0;
// ts = pc+1 and the target's v_after = v_before+delta are INLINED, not columns):
//   [0..4)   is_nop, is_not, is_cnot, is_toffoli
//   [4..8)   target:  addr, prev_ts, v_before, d   (ACCESS_BLOCK = 4)
//   [8..12)  ctrl_a:  addr, prev_ts, v, d
//   [12..16) ctrl_b:  addr, prev_ts, v, d
//   [16..19) ab, fire, delta
//
// Boundary init/final is a SEPARATE CPU-built component (BoundaryTable); the GPU main-trace kernel
// does not emit it.

// prev_ts for one access from its (prev_gate, prev_slot, wrap) predecessor triple + this rep.
// wrap=1 => the predecessor is in the PREVIOUS rep; at rep 0 that predecessor doesn't exist, so the
// init boundary node prev_ts = 0 (matches simulate_shot's last_ts init of 0). Else prev_ts is the
// predecessor's ts = prev_pc + 1 = (prev_rep*n_gates + prev_gate) + 1 (ts = pc+1, no slot; prev_slot
// is unused now — the prepass still records it but the shared ts per step makes it irrelevant, since
// a same-address predecessor is always in a DIFFERENT gate step / pc).
__device__ __forceinline__ unsigned prev_ts_of(
    unsigned prev_gate, unsigned prev_slot, unsigned wrap, unsigned rep, unsigned n_gates)
{
    (void)prev_slot;
    if (wrap && rep == 0u) return 0u;
    unsigned prev_rep = wrap ? (rep - 1u) : rep;
    return (prev_rep * n_gates + prev_gate) + 1u;
}

// Apply one gate's target-bit flip to `limbs` in place. Pure state update (no column I/O) — the part
// the K1 per-gate body and K0 share. Mirrors simulate_shot's value math exactly (fire in {0,1}).
__device__ __forceinline__ void apply_gate(
    unsigned* limbs, const unsigned* __restrict__ gates, unsigned g)
{
    unsigned opcode = gates[g * 4u + 0u];
    unsigned tq     = gates[g * 4u + 1u];
    unsigned aq     = gates[g * 4u + 2u];
    unsigned bq     = gates[g * 4u + 3u];
    unsigned is_not = (opcode == OP_NOT) ? 1u : 0u;
    unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
    unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
    unsigned a_active = is_cnot + is_tof;
    unsigned b_active = is_tof;
    unsigned tl = tq / LIMB_BITS, tbp = tq % LIMB_BITS;
    unsigned t_bit = (limbs[tl] >> tbp) & 1u;
    unsigned a_bit = a_active ? ((limbs[aq / LIMB_BITS] >> (aq % LIMB_BITS)) & 1u) : 0u;
    unsigned b_bit = b_active ? ((limbs[bq / LIMB_BITS] >> (bq % LIMB_BITS)) & 1u) : 0u;
    unsigned fire = is_not + is_cnot * a_bit + is_tof * (a_bit * b_bit);  // in {0,1}
    unsigned new_t = t_bit ^ fire;
    int delta_signed = (int)new_t - (int)t_bit;                          // {-1,0,1}
    unsigned mask = 1u << tbp;
    if (delta_signed > 0) limbs[tl] += mask;
    else if (delta_signed < 0) limbs[tl] -= mask;
}

// PREPASS: per-active-slot cyclic PREDECESSOR constants (gate, slot, wrap) — the program constants K1
// needs to reconstruct `prev_ts` (the predecessor's closed-form ts) with NO serial per-address chain.
// ONE thread walks the program ONCE in ctrl_a→ctrl_b→target order (matching simulate_shot's do_access
// sequencing), maintaining per address the (gate, slot) of its MOST RECENT access seen so far in this
// pass (`lg[a]`, `ls[a]`, valid iff `seen[a]`). For each active slot it records:
//   pg, ps = the (gate, slot) of the previous access to that addr; wrap = 0 (predecessor is EARLIER in
//   THIS pass, so same rep). If the addr has NOT been seen yet this pass (first access this pass), the
//   predecessor is the LAST access to that addr in the pass — from the PREVIOUS rep — so we leave a
//   marker (wrap=1) and backfill (pg, ps) after the full pass from the finished lg/ls (= the last
//   access to the addr in one pass). At the program-wide first access (rep 0), K1 maps wrap&&rep==0 to
//   prev_ts = 0 (the init boundary node), matching simulate_shot's `last_ts` init of 0.
// slot_meta layout per gate g: [a_pg,a_ps,a_wrap, b_pg,b_ps,b_wrap, t_pg,t_ps,t_wrap] (9 u32).
// Inactive controls leave their triple = 0 (unused: K1 zeroes an inactive ctrl AccessCols). Single-
// thread is fine — n_gates is tiny (~2547 for iadd) and this runs ONCE per process-shard, dwarfed by K1.
extern "C" __global__ void prog_slot_meta(
    const unsigned* __restrict__ gates,
    unsigned n_gates,
    unsigned* __restrict__ slot_meta)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    unsigned lg[N_QUBITS];    // gate index of the addr's last access seen this pass
    unsigned ls[N_QUBITS];    // its slot
    unsigned char seen[N_QUBITS];
    for (unsigned a = 0; a < N_QUBITS; a++) { lg[a] = 0u; ls[a] = 0u; seen[a] = 0u; }
    // A record: for each active slot whose predecessor is a FIRST-in-pass wrap, remember where to
    // backfill (slot_meta index for pg) and which addr, so the second pass can fill from lg/ls.
    for (unsigned g = 0; g < n_gates; g++) {
        unsigned opcode = gates[g * 4u + 0u];
        unsigned tq     = gates[g * 4u + 1u];
        unsigned aq     = gates[g * 4u + 2u];
        unsigned bq     = gates[g * 4u + 3u];
        unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
        unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
        unsigned a_active = is_cnot + is_tof;
        unsigned b_active = is_tof;
        unsigned base = g * 9u;
        // Emit predecessor for one active access to addr q at slot `slot`; then record (g, slot) as the
        // addr's latest access. `off` is the slot's base offset in slot_meta (0/3/6 for a/b/t).
        #define PRED(q, slot, off) do {                                             \
            if (seen[(q)]) {                                                         \
                slot_meta[base + (off) + 0u] = lg[(q)];                             \
                slot_meta[base + (off) + 1u] = ls[(q)];                             \
                slot_meta[base + (off) + 2u] = 0u;   /* same rep */                 \
            } else {                                                                \
                /* first access to q this pass: predecessor is the pass's LAST     \
                   access to q, from the previous rep. Marked wrap=1; pg/ps        \
                   backfilled below. Stash the addr in the pg slot temporarily. */ \
                slot_meta[base + (off) + 0u] = (q);   /* temp: addr, backfilled */  \
                slot_meta[base + (off) + 1u] = 0u;                                  \
                slot_meta[base + (off) + 2u] = 1u;   /* wrap */                     \
            }                                                                       \
            lg[(q)] = g; ls[(q)] = (slot); seen[(q)] = 1u;                          \
        } while (0)
        if (a_active) PRED(aq, SLOT_CTRL_A, 0u);
        if (b_active) PRED(bq, SLOT_CTRL_B, 3u);
        PRED(tq, SLOT_TARGET, 6u);   // target always active
        #undef PRED
    }
    // Backfill every wrap=1 predecessor with the addr's LAST access in the pass (lg/ls now final).
    for (unsigned g = 0; g < n_gates; g++) {
        unsigned base = g * 9u;
        #pragma unroll
        for (unsigned off = 0u; off < 9u; off += 3u) {
            if (slot_meta[base + off + 2u] == 1u) {
                unsigned q = slot_meta[base + off + 0u];  // temp addr stashed above
                slot_meta[base + off + 0u] = lg[q];
                slot_meta[base + off + 1u] = ls[q];
            }
        }
    }
}

// K0: thread-per-SHOT, VALUE-ONLY. Walk the full k*n_gates chain (cheap apply_gate limb updates, NO
// column stores) and snapshot the 512-qubit state (32 limbs) at the START of every (shot, rep) into
// rep_states. rep 0's snapshot is exactly x_states[shot]; rep r's is the state after r full passes.
// Keeps total value-chain work LINEAR in k while letting K1 start each (shot, rep) independently.
extern "C" __global__ void gate_sim_states(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ x_states,
    unsigned* __restrict__ rep_states,
    unsigned k,
    unsigned n_gates,
    unsigned n_shots)
{
    unsigned shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= n_shots) return;

    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = x_states[shot * N_LIMBS + i];

    for (unsigned rep = 0; rep < k; rep++) {
        unsigned long base = ((unsigned long)shot * (unsigned long)k + (unsigned long)rep) * N_LIMBS;
        #pragma unroll
        for (unsigned i = 0; i < N_LIMBS; i++) rep_states[base + i] = limbs[i];
        for (unsigned g = 0; g < n_gates; g++) apply_gate(limbs, gates, g);
    }
}

// K1: THREAD-PER-EXECUTION. One thread per (shot, rep). exec = shot*k + rep. Seeds a local 512-state
// from K0's rep-boundary snapshot (so this execution starts exactly where simulate_shot was entering
// rep `rep`), processes the rep's n_gates gates for v_before/v_after, and fills prev_ts (from
// slot_meta's cyclic-predecessor constants) and the per-access rc value `d`. ts is the closed form
// `pc + 1` (shared by all accesses of a step, not emitted); `d = ts - prev_ts - 1 = pc - prev_ts`.
// `off_lo` is repurposed to carry rep_states and `off_hi` to carry slot_meta (both formerly-unused arg
// slots). `rc_lo` is repurposed as the rc-table MULTIPLICITY HISTOGRAM over `d` (atomic bumps).
extern "C" __global__ void gate_sim(
    const unsigned* __restrict__ gates,
    const unsigned* __restrict__ x_states,      // UNUSED by K1 (state comes from rep_states); compat
    const unsigned* __restrict__ rep_states,    // (was off_lo) n_shots*k*N_LIMBS rep-boundary states
    const unsigned* __restrict__ slot_meta,     // (was off_hi) n_gates*9 predecessor constants
    unsigned* __restrict__ cols,
    unsigned* __restrict__ rc_hist,             // (was rc_lo) 2^rc_log rc multiplicity histogram over d
    unsigned k,
    unsigned n_gates,
    unsigned n_shots,
    unsigned long padded_rows)
{
    (void)x_states;

    // THREAD-PER-EXECUTION: one thread per (shot, rep) = n_shots*k threads.
    unsigned long exec = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long n_exec = (unsigned long)n_shots * (unsigned long)k;
    if (exec >= n_exec) return;
    unsigned rep  = (unsigned)(exec % (unsigned long)k);

    // Local 512-qubit value state, seeded from K0's rep-boundary snapshot (32 limbs). This is the
    // value of every qubit at the START of (shot, rep) — exactly last_val entering this rep. ts/prev_ts
    // are CLOSED FORM (below) — no running per-address counter is threaded.
    unsigned limbs[N_LIMBS];
    #pragma unroll
    for (unsigned i = 0; i < N_LIMBS; i++) limbs[i] = rep_states[exec * N_LIMBS + i];

    // Rows this execution owns: [exec*n_gates, (exec+1)*n_gates). row(shot,rep,g)=exec*n_gates+g.
    unsigned long row = exec * (unsigned long)n_gates;

    for (unsigned g = 0; g < n_gates; g++) {
        unsigned opcode = gates[g * 4u + 0u];
        unsigned tq     = gates[g * 4u + 1u];
        unsigned aq     = gates[g * 4u + 2u];
        unsigned bq     = gates[g * 4u + 3u];

        unsigned is_nop = (opcode == OP_NOP) ? 1u : 0u;
        unsigned is_not = (opcode == OP_NOT) ? 1u : 0u;
        unsigned is_cnot = (opcode == OP_CNOT) ? 1u : 0u;
        unsigned is_tof = (opcode == OP_TOFFOLI) ? 1u : 0u;
        unsigned a_active = is_cnot + is_tof;
        unsigned b_active = is_tof;

        unsigned meta = g * 9u;
        // pc for this execution's gate g = rep*n_gates + g (verifier-pinned per-shot program counter).
        unsigned pc = rep * n_gates + g;

        // Read a qubit's current value from the local limb state.
        #define QVAL(q) ((limbs[(q) / LIMB_BITS] >> ((q) % LIMB_BITS)) & 1u)
        // prev_ts from a slot's (prev_gate, prev_slot, wrap) predecessor triple at slot_meta[meta+off].
        #define PREVTS(off) prev_ts_of(slot_meta[meta+(off)+0u], slot_meta[meta+(off)+1u], \
                                       slot_meta[meta+(off)+2u], rep, n_gates)

        // Access order matches simulate_shot: ctrl_a, ctrl_b, then target. ts = pc + 1 (program-order
        // timestamp), SHARED by all accesses of this step (no slot — ts is not emitted). Values are
        // read from the local state (v_before); the target write updates the local state so LATER
        // gates in this rep read the new value. d = ts - prev_ts - 1 = pc - prev_ts.
        unsigned ts = pc + 1u;
        // ctrl_a (active iff a_active): read propagates value (v unchanged).
        unsigned a_addr = 0u, a_prev = 0u, a_v = 0u, a_d = 0u;
        if (a_active) {
            a_addr = aq;
            a_v    = QVAL(aq);
            a_prev = PREVTS(0u);
            a_d    = ts - a_prev - 1u;   // = pc - prev_ts, >= 0 (prev_ts is an earlier program-order ts or 0)
            atomicAdd(&rc_hist[a_d], 1u);
        }
        // ctrl_b (active iff b_active).
        unsigned b_addr = 0u, b_prev = 0u, b_v = 0u, b_d = 0u;
        if (b_active) {
            b_addr = bq;
            b_v    = QVAL(bq);
            b_prev = PREVTS(3u);
            b_d    = ts - b_prev - 1u;
            atomicAdd(&rc_hist[b_d], 1u);
        }
        // target (always active): read+write.
        unsigned t_addr = tq;
        unsigned t_v    = QVAL(tq);     // v_before
        unsigned t_prev = PREVTS(6u);
        unsigned t_d    = ts - t_prev - 1u;
        atomicAdd(&rc_hist[t_d], 1u);

        unsigned ab = a_v * b_v;
        unsigned fire = is_not + is_cnot * a_v + is_tof * ab;   // in {0,1}
        unsigned v_after = t_v ^ fire;
        int delta_signed = (int)v_after - (int)t_v;             // {-1,0,1}

        // Commit target write to the LOCAL state (so subsequent gates in this rep see the new value).
        {
            unsigned tl = tq / LIMB_BITS, tbp = tq % LIMB_BITS, mask = 1u << tbp;
            if (delta_signed > 0) limbs[tl] += mask;
            else if (delta_signed < 0) limbs[tl] -= mask;
        }
        #undef QVAL
        #undef PREVTS

        // Emit the 19 cells in cell_at order (ACCESS_BLOCK = 4: addr,prev_ts,v,d). ts (=pc+1) and the
        // target's v_after (=v_before+delta) are NOT emitted — they are inlined in the AIR / K4
        // interaction kernel. (void v_after / delta arithmetic still updates the local state above.)
        unsigned cc = 0u;
        #define EMIT(v) cols[(unsigned long)(cc++) * padded_rows + row] = (v)
        EMIT(is_nop); EMIT(is_not); EMIT(is_cnot); EMIT(is_tof);              // 0..4
        EMIT(t_addr); EMIT(t_prev); EMIT(t_v); EMIT(t_d);                     // 4..8  target
        EMIT(a_addr); EMIT(a_prev); EMIT(a_v); EMIT(a_d);                     // 8..12 ctrl_a
        EMIT(b_addr); EMIT(b_prev); EMIT(b_v); EMIT(b_d);                     // 12..16 ctrl_b
        EMIT(ab); EMIT(fire);                                                 // 16..18
        EMIT(delta_signed >= 0 ? (unsigned)delta_signed
                               : (unsigned)((int)M31_MOD + delta_signed));    // 18 delta_to_m31
        #undef EMIT

        row += 1u;
    }
}

// Padding rows [real_rows, padded_rows) match `Row::padding()` = ALL ZERO (AccessCols::inactive()
// is addr=ts=prev_ts=v=0; is_* = 0; ab=fire=delta=0). `cols` is pre-zeroed by the caller, so there
// is nothing to write (no padding kernel needed).
