# P3.4 — wire gate_air prove GPU-resident + measure t_gpu (scope)

## The finding that reframes P3.4

P3.4's goal was "wire GPU trace-gen into the prove with NO host round-trip, then measure
t_gpu_pure for the Tanuj curve". Investigating the integration surfaced two decisive facts:

1. **obelyzk's GpuBackend is host-column.** `GpuBackend::Column = simd::BaseColumn`
   (gpu/column.rs:40,66). Trace columns live in HOST memory; every NTT/Merkle op offloads to
   the GPU per-call (H2D + D2H). "Keep the trace device-resident through the prove" is therefore
   *impossible* on this backend — it is **structurally transfer-bound**. This is exactly why P1
   (model A) measured ~31s vs SIMD's 9.8s: a net LOSS. GPU trace-gen feeding obelyzk's prover
   re-incurs that same D2H wall. **The obelyzk path cannot beat SIMD/SP1.**

2. **NitrooZK-stwo (dev, v2.1.0) is the device-resident architecture we need — and the recorded
   "no vcs_lifted" blocker is STALE.** The current dev branch has, all at once:
   - **Lifted Merkle on GPU**: device-resident `MerkleOpsLifted<Blake2sMerkleHasherGeneric>`,
     heterogeneous column sizes, GPU lifting kernels (backend/cuda/blake2s.rs:100-253). (= P2, done.)
   - **Device-resident columns**: `BaseFieldVec { device_ptr }`, device-to-device ops, host copy
     only on explicit `to_cpu()`.
   - **ExtendedStarkProof + set_store_polynomials_coefficients** (core/proof.rs:20) — the exact
     proof-aux form gate_air's leaf_prover emits for the in-circuit verifier.

So the K1+K4 NVRTC kernels (gate-sim, logup — correct & byte-identity-validated) are the right
gate_air-specific trace-gen, but to PAY OFF they must plug into a device-resident backend.
That backend is NitrooZK's, not obelyzk's.

## The fork

- **Path O (obelyzk, current tree):** finish wiring K1/K4 into the obelyzk prove. Right rev
  (74951f79) already integrated; K1/K4 done. BUT host-column → transfer-bound → cannot win.
  Measuring it just confirms a ~31s loss. Dead end for the Tanuj number.
- **Path N (NitrooZK device-resident):** target gate_air's prove at NitrooZK's CudaBackend.
  The only path that can produce a WINNING measured number. Lifted-Merkle-on-GPU already done.
  Cost/risk:
  - REV-ALIGN: gate_air + recursion need stwo 74951f79 + circuits 0a6351e. NitrooZK is stwo
    v2.1.0. Must verify the EXACT proof serialization NitrooZK emits deserializes in our
    circuits_stark_verifier (the historical integration risk — verify, don't assume).
  - PORT K1/K4 to NitrooZK's build model (nvcc/cmake/cub .cu files, device-resident BaseFieldVec
    inputs/outputs) instead of NVRTC/cudarc + host Vec<u32>. The kernel MATH is identical and
    validated; the glue + I/O change. (NitrooZK already has the logup/prefix-sum/batch-inverse
    primitives K4 mirrors — so K4 largely collapses into their existing machinery.)
  - INTEGRATE the gate_air component (FrameworkEval) + preprocessed/mult/table-interactions +
    leaf_prover proof emission on NitrooZK's prover.

## Recommendation

Pivot the GPU prove to **Path N**. obelyzk gave us the right-rev integration learning + validated
K1/K4 kernels, but it cannot produce a winning number. Suggested first concrete step (cheap,
decisive, NO new kernels): stand up NitrooZK locally and **prove a small gate_air (or a
FrameworkEval stand-in) on its CudaBackend at v2.1.0, then check whether the emitted
ExtendedStarkProof deserializes/verifies under circuits_stark_verifier 0a6351e** — i.e. settle
the rev-align risk BEFORE investing in the K1/K4 port. If it deserializes, the path is green and
we port K1/K4 + integrate. If not, scope the serialization delta.

## DE-RISK VERDICT (2026-06-26) — proof-format compatibility checked

NOT directly compatible, but BRIDGEABLE via the shared lifted scheme.

- OURS (stwo 74951f79): the WHOLE proof path is LIFTED. `CommitmentSchemeProof<H: MerkleHasherLifted>`,
  `MerkleDecommitmentLifted { hash_witness }` (NO column_witness), `queried_values:
  TreeVec<ColumnVec<Vec<BaseField>>>` (column-nested), `FriConfig.fold_step` + FRI `pack_leaves`
  logic, channel = `Blake2sM31MerkleChannel`. Recursion/circuits_stark_verifier consume THIS.
- NITROO (v2.1.0) DEFAULT prove_ex: NON-LIFTED. `StarkProof<H: MerkleHasher>`, `MerkleDecommitment
  { hash_witness, column_witness }`, `queried_values: TreeVec<Vec<BaseField>>` (flat),
  `FriConfig.line_fold_step` (no pack_leaves in first-layer verifier), channel = `Blake2sMerkleChannel`.
- BUT NITROO ALSO HAS the lifted primitives, and they MATCH ours: `vcs_lifted::MerkleDecommitmentLifted
  { hash_witness }` only (core/vcs_lifted/verifier.rs:12) — identical; lifted prover ops
  (prover/vcs_lifted/), and a DEVICE-RESIDENT CUDA `MerkleOpsLifted` (backend/cuda/blake2s.rs:100-253).
  These are present but NOT wired into NitrooZK's CommitmentSchemeProver/prove_ex/FRI (which are non-lifted).

### Recommended bridge = **Bridge B**: port NitrooZK's device-resident CudaBackend INTO our tree
Add a device-resident `CudaBackend` to stwo-gpu-port (@ 74951f79) by porting NitrooZK's CUDA kernels
(device columns `BaseFieldVec`, NTT, **lifted** Merkle, FRI, batch-inverse, prefix-sum) + implementing
OUR backend traits (`ColumnOps`, `PolyOps`, `MerkleOpsLifted`, `FriOps`, `BackendForChannel<Blake2sM31MerkleChannel>`).
Then `prove_ex::<CudaBackend, Blake2sM31MerkleChannel>` emits OUR correct lifted proof, ON-DEVICE.
- WHY B over "wire lifted in NitrooZK (Bridge A)": our tree's lifted PCS/FRI/verifier is already correct
  AND recursion-validated — keep it untouched (no recursion re-validation). We only borrow NitrooZK's
  device-resident CUDA OPS, not its (older, non-lifted) proof/FRI structure.
- SCALE: a P0-style rebase but bigger — nvcc/cmake/cub build, device column types, real CUDA kernels.
  K1/K4 (validated) port to device-resident I/O (BaseFieldVec in/out instead of host Vec<u32>).
- KEY RISK: FRI `pack_leaves` on device. Our FRI uses pack_leaves (tied to lifting); NitrooZK's CUDA
  FRI appears to predate it (line_fold_step, no pack_leaves). Porting/adding pack_leaves to the device
  FRI is the one genuinely-new-GPU-crypto piece — scope it explicitly + byte-identity-gate it.
- NON-GOAL: do NOT adopt NitrooZK's proof serialization or non-lifted FRI.

## Deferred regardless
GPU table-interactions / multiplicity / preprocessed (small, table-sized — CPU-generate + upload
v1); these are not the cost-dominant pieces (K1 main 191×rows + K4 interaction 24×rows are).
