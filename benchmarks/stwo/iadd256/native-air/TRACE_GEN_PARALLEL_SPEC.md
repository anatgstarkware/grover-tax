# gate_air — parallelize trace generation across shots (spec)

## Why
Measured on the VM (2^24-row workload, RAYON sweep):
- The **prover** parallelizes well (~19× to ~64 threads — stwo uses rayon internally).
- **Trace generation does not**: at 96 threads the full wall is ~54 s while the prove
  is only ~5.6 s — i.e. **~48 s is trace-gen, and it barely parallelizes (~2.8×)**.
So trace-gen is now the end-to-end bottleneck. This spec makes it parallel.

All changes are in **gate_air.rs (+ native-air/Cargo.toml)** — NO stwo changes.
(`rayon` is already added to Cargo.toml.)

## The parallelism unit is the SHOT
- Within a shot the gates and the K reps are **strictly sequential**: gate `t`'s input
  is gate `t-1`'s output (state threads). You CANNOT parallelize inside a chain.
- Shots are **fully independent**: shot `s` owns the contiguous row block
  `[s·K·n_gates, (s+1)·K·n_gates)`, has its own initial state `x_s`, and its own chain
  to `y_s`, with no shared mutable state. With N up to 9024, that's ample parallelism.

## What to change
1. **Par-over-shots simulation.** Replace the serial shot loop (the
   `for (shot_id, case) in cases.iter().enumerate()` that builds rows) with a rayon
   **parallel iterator over shots**. Each thread simulates one shot sequentially (load
   `x_s`, run K·n_gates gates threading its 512-bit state) and fills that shot's block.

2. **SIMD packing — preserve the EXACT trace layout (soundness-critical).**
   Columns are `PackedM31` (16 logical rows per word = `LANE_COUNT`), and a shot block
   (2547·K rows) is NOT 16-aligned, so two shots can share a packed word → data race.
   **Use the two-phase approach:**
   - Phase 1 (parallel over shots): write per-row **scalar** column data into disjoint
     regions (one shot = one disjoint scalar range).
   - Phase 2 (parallel over packed rows): pack scalars → `PackedM31` columns with
     `par_iter_mut` over packed rows (embarrassingly parallel — just reads finished
     scalars).
   This keeps the trace **bit-identical** to the current serial version (same proof).
   Do NOT lane-pad shots with extra rows unless you can prove it leaves `pc_in_prog`,
   the chain-lookup boundary, the program-consistency lookup, and all multiplicities
   exactly correct — changing row indices is dangerous; two-phase avoids it.

3. **Shot-independent columns** (the op columns, `pc_in_prog`, the program table) do
   not depend on the shot — `op[s, t] = program[t mod n_gates]` for every `s`. Fill
   them once / by tiling the program, not per-shot.

4. **Lookup multiplicities are global aggregates** (qdecode / rc_lo / rc_hi / program
   use-counts): accumulate **per-shot local count arrays**, then **reduce (sum) across
   shots** at the end. Arrays are small (512, 2^16, n_gates).

5. **Add `trace_gen_s` to the JSON report** — time the full witness build (simulation
   + column fill, everything before the prove call). Keep `prove_s` / `verify_s`.
   This is the metric we'll sweep to measure trace-gen scaling.

## Correctness bar (must hold)
The parallel trace MUST be **bit-identical** to the serial one — it's just filling the
same cells concurrently. Verify on the laptop (cheap, allowed — NOT the full FRI prover):
- `GATE_AIR_ASSERT=1` passes on all components for k4-n16 **N=4** (exercises multiple
  shots in parallel) and N=1, and the prover cross-check passes.
- The Rust self-check (final state == y) passes for all shots.
- `cargo +nightly-2025-07-14 build --release --bin gate_air` compiles clean.
Do NOT run the full prove+verify — that's VM-only (run separately).

## Measurement (done after, on the VM)
Rebuild with `RUSTFLAGS="-C target-cpu=native"`; sweep `RAYON_NUM_THREADS ∈
{1,2,4,8,16,32,64,96}` at a fixed 2^24-row workload (k1-n9024, --samples 4096) and read
`trace_gen_s` at each → compare against the current ~2.8× to see the new scaling.
