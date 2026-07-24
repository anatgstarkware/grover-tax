//! Base (per-shard) gate_air prover, extracted from `main.rs`.
//!
//! Holds the base-proof pipeline (trace-gen + commit + `prove_ex`) that produces one
//! `ExtendedStarkProof` per shard, plus the shard-invariant precompute and the tree-0 program /
//! boundary / rc supply tables the base proof commits. Mirrors how `leaf.rs` holds the leaf prover.
//! Byte-identical to the pre-extraction inline code (pure code move).

use crate::*;
// AIR assembly items (LookupElements, GateRel, build_components, consts, pp/ptag) live in `air`;
// glob-imported so this module's bare references keep resolving.
use crate::air::*;
// Relation ids now live with their owning components (module reorg). Only `TAG_RC` is used bare here.
use crate::components::range_check::TAG_RC;
// Trace/witness-gen items relocated to `tracegen` (Step-1 reorg). Imported explicitly rather than
// glob so `tracegen`'s gate-local `TRACE_COLUMNS`/`GATE_REL_WIDTH` gpu consts don't shadow the
// crate-root shared consts this module uses via `use crate::*`.
use crate::tracegen::{
    build_rc_table, build_rows, build_tree0_columns, cell_at, gen_boundary_interaction,
    gen_program_interaction, gen_table_interaction, generate_boundary_witness,
    generate_program_witness, generate_rc_witness, pack_seq, state_to_limbs, to_prover,
    BoundaryTable, RcIndex, Row,
};
// Only the non-cuda base path calls the CPU main-interaction generator (the cuda path uses the GPU
// K4 kernel), so gate its import to match.
#[cfg(not(feature = "cuda"))]
use crate::tracegen::gen_main_interaction;
#[cfg(feature = "cuda")]
use crate::tracegen::gpu_flat_inputs;

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use circuits_stark_verifier::proof_from_stark_proof::pack_public_claim;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::ExtendedStarkProof;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::SimdBackend as TraceBackend;
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove_ex, CommitmentSchemeProver};
use stwo_constraint_framework::Relation;

/// Default base (shard) proof FRI blowup factor; derives the ~96-bit config via `leaf_pcs_config`.
/// Env override `BASE_BLOWUP` is parsed in `main`.
pub const BASE_LOG_BLOWUP: u32 = 1;

/// Program-consistency table (the single hidden program): one row per program slot (gate) i in
/// 0..n_gates, padded to a power of two. Row i stores the canonical op tuple as WITNESS plus a
/// multiplicity = samples*K executions of that slot. Inactive controls canonicalise to 0 (matching the
/// use side). Slot index is PREPROCESSED. Padding rows carry multiplicity 0, so their op contents are
/// inert (never addressed: pc_in_prog stays in 0..n_gates).
pub(crate) struct ProgramTable {
    pub(crate) log_size: u32,
    pub(crate) slot: Vec<u32>,          // preprocessed slot index 0..size
    pub(crate) opcode_scalar: Vec<u32>, // witness
    pub(crate) target: Vec<u32>,        // witness
    pub(crate) ctrl_a: Vec<u32>,        // witness
    pub(crate) ctrl_b: Vec<u32>,        // witness
    pub(crate) multiplicity: Vec<u32>,  // witness: samples*K on real slots, 0 on padding
}

/// Log-size of the program table (one row per gate, padded to a power of two, floored at LANE_COUNT).
/// Pure function of `n_gates` (shard-invariant), so it can be recovered without the table itself.
pub(crate) fn program_log_size(n_gates: usize) -> u32 {
    n_gates.next_power_of_two().max(LANE_COUNT).ilog2()
}

pub(crate) fn build_program_table(gates: &[Gate], samples: usize, k: usize) -> ProgramTable {
    let n_gates = gates.len();
    let padded = n_gates.next_power_of_two().max(LANE_COUNT);
    let log_size = padded.ilog2();
    let mut slot = vec![0u32; padded];
    let mut opcode_scalar = vec![0u32; padded];
    let mut target = vec![0u32; padded];
    let mut ctrl_a = vec![0u32; padded];
    let mut ctrl_b = vec![0u32; padded];
    let mut multiplicity = vec![0u32; padded];
    let mult = (samples * k) as u32;
    for (i, g) in gates.iter().enumerate() {
        slot[i] = i as u32;
        let (sc, a_active, b_active) = match g.opcode {
            OP_NOP => (0u32, false, false),
            OP_NOT => (1u32, false, false),
            OP_CNOT => (2u32, true, false),
            OP_TOFFOLI => (3u32, true, true),
            _ => (0u32, false, false),
        };
        opcode_scalar[i] = sc;
        target[i] = g.target as u32;
        ctrl_a[i] = if a_active { g.ctrl_a as u32 } else { 0 };
        ctrl_b[i] = if b_active { g.ctrl_b as u32 } else { 0 };
        multiplicity[i] = mult;
    }
    // Padding slots keep an in-range index sequence (inert; multiplicity 0).
    for (i, s) in slot.iter_mut().enumerate().skip(n_gates) {
        *s = i as u32;
    }
    ProgramTable {
        log_size,
        slot,
        opcode_scalar,
        target,
        ctrl_a,
        ctrl_b,
        multiplicity,
    }
}
/// Pack the scalar `Vec<Row>` main trace into `PackedM31` columns, in parallel over columns: each task
/// owns one column's whole buffer, so no two threads touch the same packed word (a non-16-aligned shot
/// block can't race). Bit-identical to a serial per-cell fill.
pub(crate) fn generate_main_trace(
    rows: &[Row],
    padded_rows: usize,
    log_n_rows: u32,
) -> ColumnVec<CircleEvaluation<TraceBackend, BaseField, BitReversedOrder>> {
    use rayon::prelude::*;
    use stwo::prover::backend::simd::column::BaseColumn;

    let n_vec = padded_rows / LANE_COUNT;
    let pad = Row::padding();
    let domain = CanonicCoset::new(log_n_rows).circle_domain();

    (0..TRACE_COLUMNS)
        .into_par_iter()
        .map(|col| {
            let col_data: Vec<PackedM31> = (0..n_vec)
                .map(|vec_row| {
                    PackedM31::from_array(std::array::from_fn(|lane| {
                        let idx = vec_row * LANE_COUNT + lane;
                        let row = rows.get(idx).unwrap_or(&pad);
                        BaseField::from_u32_unchecked(cell_at(row, col))
                    }))
                })
                .collect();
            CircleEvaluation::<TraceBackend, _, BitReversedOrder>::new(
                domain,
                BaseColumn::from_simd(col_data),
            )
        })
        .collect()
}
/// Shard-invariant base-proof precompute, built once before the shard loop and shared by `Arc` across
/// every shard so the shot-independent work runs once instead of N times: tree-0 (interpolated + LDE +
/// Merkle-committed once, reused via `commit_tree` which re-mixes the same root into each shard's
/// channel), twiddles, the N1 program table (constant multiplicity across shards), and (cuda) the N3
/// device-resident gate-list / RcIndex-offset buffers.
// `config`/`boundary`/`padded_rows`/`log_n_rows` are read only by the cuda `build_device_parts` and
// the debug/test `assert_tree0_matches_rebuild`; in a non-cuda release build they are populated-but-
// unread, so allow dead_code in exactly that config.
#[cfg_attr(not(any(debug_assertions, test, feature = "cuda")), allow(dead_code))]
pub(crate) struct BaseProverPrecompute {
    // `config`/`tree0`/`program`/`boundary`/`padded_rows`/`log_n_rows`/`rc_log` are read by the
    // debug/test `diag::assert_tree0_matches_rebuild` (a sibling module), hence `pub(crate)`.
    pub(crate) config: stwo::core::pcs::PcsConfig,
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    pub(crate) tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    /// Shared N1 program table (constant multiplicity across shards).
    pub(crate) program: ProgramTable,
    /// Shard-invariant boundary table shape: the preprocessed columns depend only on the (shot, addr)
    /// shape (witness x/y/ts_last are per-shard).
    pub(crate) boundary: BoundaryTable,
    /// Fixed shard shape (every shard holds `shots_per_shard` shots → same row count).
    pub(crate) padded_rows: usize,
    pub(crate) log_n_rows: u32,
    /// Dynamic rc-table log-size (= ceil(log2(k*n_gates))); shard-invariant. Used to rebuild tree0
    /// (device-n replica / the debug rebuild check) with the same rc membership sizing.
    pub(crate) rc_log: u32,
    /// (cuda) shape needed to rebuild the device-resident parts on a producer's device (device != 0).
    #[cfg(feature = "cuda")]
    max_log_size: u32,
    #[cfg(feature = "cuda")]
    n_gates: usize,
    /// (cuda) device-resident N3 inputs uploaded once on device 0 (gate list + RcIndex offsets); for
    /// devices != 0 the equivalent buffers live in `device_parts`.
    #[cfg(feature = "cuda")]
    d_gates: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    d_off_hi: cudarc::driver::CudaSlice<u32>,
    // Multi-GPU: the device-resident precompute is bound to the device it was built on. Device 0 uses
    // the eager fields above; for a producer on device n != 0, `device_parts()` lazily rebuilds the
    // same parts on device n from these owned host inputs and caches them in slot n. tree0 is
    // shard-invariant, so a device-n rebuild is byte-identical — pure per-device replication.
    #[cfg(feature = "cuda")]
    rows0: Vec<Row>,
    #[cfg(feature = "cuda")]
    gates_flat: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_lo: Vec<u32>,
    #[cfg(feature = "cuda")]
    off_hi: Vec<u32>,
    #[cfg(feature = "cuda")]
    device_parts: [std::sync::OnceLock<DevicePrecompute>; MAX_BASE_GPUS],
}

/// Number of base GPUs supported by the per-device precompute cache (matches tracegen's cap).
#[cfg(feature = "cuda")]
const MAX_BASE_GPUS: usize = 16;

/// (cuda) The device-resident half of the base precompute, bound to one device (device 0 eagerly in
/// `new`, devices != 0 lazily in `device_parts`). tree0/twiddles are re-derived identically per device
/// (shard-invariant), so replication changes no committed value.
#[cfg(feature = "cuda")]
struct DevicePrecompute {
    twiddles: stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: cudarc::driver::CudaSlice<u32>,
    d_off_lo: cudarc::driver::CudaSlice<u32>,
    d_off_hi: cudarc::driver::CudaSlice<u32>,
}

impl BaseProverPrecompute {
    /// Build the precompute from shard 0's shape (`program0`, `rows0`), which is shard-invariant (see
    /// [`build_tree0_columns`]), so the cached tree-0 is valid for every shard. Soundness guard: the
    /// caller asserts the committed root equals an independently rebuilt shard-0 root
    /// (`assert_tree0_matches_rebuild`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: stwo::core::pcs::PcsConfig,
        max_log_size: u32,
        program0: ProgramTable,
        rows0: &[Row],
        boundary0: BoundaryTable,
        padded_rows: usize,
        log_n_rows: u32,
        n_gates: usize,
        rc_log: u32,
        #[cfg(feature = "cuda")] gates_flat: &[u32],
        #[cfg(feature = "cuda")] off_lo: &[u32],
        #[cfg(feature = "cuda")] off_hi: &[u32],
    ) -> Result<Self> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        // Scratch pool for tree-0's build only (the committed tree owns its polynomials).
        let pool = BaseColumnPool::<ProverBackend>::new();

        // Build + commit tree-0 once, matching the per-shard `tree_builder().commit()` path (same
        // scheme config; store=false for the barycentric-OODS path, no stored coeffs).
        let cols = build_tree0_columns(
            &program0,
            rows0,
            padded_rows,
            log_n_rows,
            n_gates,
            rc_log,
            &boundary0,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            config.fri_config.log_blowup_factor,
            &twiddles,
            false, // store_polynomials_coefficients: barycentric OODS path
            config.lifting_log_size,
            &pool,
        );

        #[cfg(feature = "cuda")]
        let dev = tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        #[cfg(feature = "cuda")]
        let d_gates = dev
            .htod_copy(gates_flat.to_vec())
            .map_err(|e| anyhow::anyhow!("htod gates (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_lo = dev
            .htod_copy(off_lo.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_lo (precompute): {e}"))?;
        #[cfg(feature = "cuda")]
        let d_off_hi = dev
            .htod_copy(off_hi.to_vec())
            .map_err(|e| anyhow::anyhow!("htod off_hi (precompute): {e}"))?;

        Ok(Self {
            config,
            twiddles,
            tree0,
            program: program0,
            boundary: boundary0,
            padded_rows,
            log_n_rows,
            rc_log,
            #[cfg(feature = "cuda")]
            max_log_size,
            #[cfg(feature = "cuda")]
            n_gates,
            #[cfg(feature = "cuda")]
            d_gates,
            #[cfg(feature = "cuda")]
            d_off_lo,
            #[cfg(feature = "cuda")]
            d_off_hi,
            // Owned rebuild inputs so `device_parts` can replicate on a producer's device (n != 0).
            #[cfg(feature = "cuda")]
            rows0: rows0.to_vec(),
            #[cfg(feature = "cuda")]
            gates_flat: gates_flat.to_vec(),
            #[cfg(feature = "cuda")]
            off_lo: off_lo.to_vec(),
            #[cfg(feature = "cuda")]
            off_hi: off_hi.to_vec(),
            // Slot 0 stays empty (device 0 uses the eager fields); slots 1..MAX filled lazily.
            #[cfg(feature = "cuda")]
            device_parts: [const { std::sync::OnceLock::new() }; MAX_BASE_GPUS],
        })
    }

    /// (cuda) Build the device-resident precompute parts (twiddles + tree0 + N3 buffers) on the calling
    /// thread's current device from the shard-invariant host inputs — byte-identical tree0 to `new`.
    /// The caller must have already bound its device.
    #[cfg(feature = "cuda")]
    fn build_device_parts(&self) -> Result<DevicePrecompute> {
        use stwo::prover::mempool::BaseColumnPool;
        use stwo::prover::poly::circle::PolyOps;
        use stwo::prover::CommitmentTreeProver;

        let twiddles = ProverBackend::precompute_twiddles(
            CanonicCoset::new(self.max_log_size + 1 + self.config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );
        let pool = BaseColumnPool::<ProverBackend>::new();
        let cols = build_tree0_columns(
            &self.program,
            &self.rows0,
            self.padded_rows,
            self.log_n_rows,
            self.n_gates,
            self.rc_log,
            &self.boundary,
        );
        let polys = ProverBackend::interpolate_columns(to_prover(cols), &twiddles);
        let tree0 = CommitmentTreeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(
            polys,
            self.config.fri_config.log_blowup_factor,
            &twiddles,
            false,
            self.config.lifting_log_size,
            &pool,
        );
        let dev = tracegen::cuda_device().map_err(|e| anyhow::anyhow!(e))?;
        let d_gates = dev
            .htod_copy(self.gates_flat.clone())
            .map_err(|e| anyhow::anyhow!("htod gates (device precompute): {e}"))?;
        let d_off_lo = dev
            .htod_copy(self.off_lo.clone())
            .map_err(|e| anyhow::anyhow!("htod off_lo (device precompute): {e}"))?;
        let d_off_hi = dev
            .htod_copy(self.off_hi.clone())
            .map_err(|e| anyhow::anyhow!("htod off_hi (device precompute): {e}"))?;
        Ok(DevicePrecompute {
            twiddles,
            tree0,
            d_gates,
            d_off_lo,
            d_off_hi,
        })
    }

    /// (cuda) The device-resident precompute for the calling thread's base GPU ordinal. Device 0
    /// returns the eager fields from `new`; devices != 0 lazily build + cache their own replica on
    /// first use. tree0 is shard-invariant, so every replica commits the identical root.
    #[cfg(feature = "cuda")]
    fn device_parts(&self) -> DevicePartsRef<'_> {
        let ord = tracegen::base_gpu_ordinal();
        if ord == 0 {
            return DevicePartsRef {
                twiddles: &self.twiddles,
                tree0: &self.tree0,
                d_gates: &self.d_gates,
                d_off_lo: &self.d_off_lo,
                d_off_hi: &self.d_off_hi,
            };
        }
        let slot = self
            .device_parts
            .get(ord)
            .unwrap_or_else(|| panic!("base gpu ordinal {ord} >= {MAX_BASE_GPUS}"));
        // Build once per device, on this producer thread (already bound to device `ord`). Fatal on
        // failure — a broken precompute cannot proceed.
        let parts = slot.get_or_init(|| {
            self.build_device_parts()
                .unwrap_or_else(|e| panic!("device {ord} precompute build failed: {e}"))
        });
        DevicePartsRef {
            twiddles: &parts.twiddles,
            tree0: &parts.tree0,
            d_gates: &parts.d_gates,
            d_off_lo: &parts.d_off_lo,
            d_off_hi: &parts.d_off_hi,
        }
    }
}

/// (cuda) Borrowed view of the device-resident precompute parts, so `prove_base_shard` reads them
/// uniformly regardless of ordinal.
#[cfg(feature = "cuda")]
struct DevicePartsRef<'a> {
    twiddles: &'a stwo::prover::poly::twiddles::TwiddleTree<ProverBackend>,
    tree0: &'a stwo::prover::CommitmentTreeProver<ProverBackend, Blake2sM31MerkleChannel>,
    d_gates: &'a cudarc::driver::CudaSlice<u32>,
    d_off_lo: &'a cudarc::driver::CudaSlice<u32>,
    d_off_hi: &'a cudarc::driver::CudaSlice<u32>,
}

/// The base-proof tuple `prove_base_shard` returns (named so the pipeline producer can send it over a
/// channel). Backend-independent: `prove_ex` yields `ExtendedStarkProof<Blake2sMerkleHasher>`.
pub(crate) type BaseShardOutput = (
    ExtendedStarkProof<Blake2sMerkleHasher>,
    Vec<SecureField>,
    u64,
    u32,
    u32,
    u32,
    Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    u32,
);

/// Per-shard base proof: the trace-gen + commit + prove_ex pipeline over this shard's shots, returning
/// the ExtendedStarkProof plus the claim / nonce / log_n_rows the leaf needs. `rc_lo_index` is used
/// only under `cuda`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(crate) fn prove_base_shard(
    precompute: Option<&BaseProverPrecompute>,
    shard_cases: &[TestCase],
    gates: &[Gate],
    k: usize,
    n_gates: usize,
    base_log_blowup: u32,
    rc_lo_index: &RcIndex,
) -> Result<BaseShardOutput> {
    let shard_samples = shard_cases.len();
    let (rows, boundary) = build_rows(gates, shard_cases, k)?;
    let real_rows = rows.len();
    let padded_rows = real_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let log_n_rows = padded_rows.ilog2();
    // Fixed rc-table log-size RC_LOG. Prove-entry tripwire: the fixed [0,2^RC_LOG) table must contain
    // every honest `d = pc - prev_ts` (max k*n_gates - 1); k ≳ 8000 would overflow it.
    debug_assert!(
        (k * gates.len()).next_power_of_two().ilog2() <= RC_LOG,
        "rc: k·n_gates log2 exceeds RC_LOG={RC_LOG}; k≳8000 needs a wider rc table"
    );
    let rc_log = RC_LOG;
    let max_log_size = tree0_max_log_size(
        log_n_rows,
        rc_log,
        program_log_size(gates.len()),
        boundary.log_size,
    );
    let base_blowup: u32 = base_log_blowup;
    let config = leaf::leaf_pcs_config(max_log_size, base_blowup);

    // Twiddles: shared by reference from the precompute, else built fresh per shard.
    let owned_twiddles = if precompute.is_none() {
        Some(ProverBackend::precompute_twiddles(
            CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        ))
    } else {
        None
    };
    // Multi-GPU: the device-resident precompute parts for this thread's device (device 0's eager
    // fields, or a per-device replica). `None` (no-precompute fallback) leaves `dp` None and uses
    // `owned_twiddles` / the per-shard rebuild.
    #[cfg(feature = "cuda")]
    let dp = precompute.map(|pc| pc.device_parts());
    #[cfg(feature = "cuda")]
    let twiddles = match &dp {
        Some(dp) => dp.twiddles,
        None => owned_twiddles.as_ref().unwrap(),
    };
    #[cfg(not(feature = "cuda"))]
    let twiddles = match precompute {
        Some(pc) => &pc.twiddles,
        None => owned_twiddles.as_ref().unwrap(),
    };
    // N1 program table: shared from the precompute (constant multiplicity across shards), else rebuilt.
    let owned_program = if precompute.is_none() {
        Some(build_program_table(gates, shard_samples, k))
    } else {
        None
    };
    let program: &ProgramTable = match precompute {
        Some(pc) => &pc.program,
        None => owned_program.as_ref().unwrap(),
    };

    let prover_channel = &mut Blake2sM31Channel::default();
    let channel_salt = 0u32;
    prover_channel.mix_felts(&[BaseField::from_u32_unchecked(channel_salt).into()]);
    config.mix_into(prover_channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<ProverBackend, Blake2sM31MerkleChannel>::new(config, twiddles);

    // Tree 0: reuse the precomputed commitment (re-mix the same root via `commit_tree`, no NTT/Merkle
    // rebuild), else rebuild it. Under multi-GPU the reused tree0 is this device's replica, whose root
    // is identical to device 0's, so the transcript mix is unchanged.
    #[cfg(feature = "cuda")]
    match &dp {
        Some(dp) => {
            commitment_scheme.commit_tree(MaybeOwned::Borrowed(dp.tree0), prover_channel);
        }
        None => {
            let pp = build_tree0_columns(
                program,
                &rows,
                padded_rows,
                log_n_rows,
                n_gates,
                rc_log,
                &boundary,
            );
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(to_prover(pp));
            tree_builder.commit(prover_channel);
        }
    }
    #[cfg(not(feature = "cuda"))]
    match precompute {
        Some(pc) => {
            commitment_scheme.commit_tree(MaybeOwned::Borrowed(&pc.tree0), prover_channel);
        }
        None => {
            // Fallback: build + interpolate + LDE + Merkle-commit the preprocessed columns inline
            // (the shard-invariant work the precompute eliminates).
            let pp = build_tree0_columns(
                program,
                &rows,
                padded_rows,
                log_n_rows,
                n_gates,
                rc_log,
                &boundary,
            );
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(to_prover(pp));
            tree_builder.commit(prover_channel);
        }
    }

    let public_claim = pack_public_claim(&[]);
    prover_channel.mix_felts(&public_claim);

    // ts-ordering range-check supply table (multiplicity counted from active-access lookups).
    let rc_table = build_rc_table(&rows, rc_log);

    // Tree 1: main trace + program witness + boundary witness + rc multiplicity.
    let small_main = {
        let mut v = generate_program_witness(program);
        v.extend(generate_boundary_witness(&boundary));
        v.extend(generate_rc_witness(&rc_table));
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    // Holds K1's column-major main-trace device buffer so K4 can reuse it (no K0/K1 re-run). GPU
    // trace-gen is unconditional under `cuda`; the CPU arm below survives only on the non-cuda build.
    #[cfg(feature = "cuda")]
    let d_main_cols: cudarc::driver::CudaSlice<u32> = {
        // N3 inputs (gate list + RcIndex offsets) are device-resident in the precompute on the reuse
        // path; only this shard's `x_states` is uploaded here (per-shard on the fallback path).
        let mut x_states = Vec::with_capacity(shard_cases.len() * N_LIMBS);
        for c in shard_cases {
            let bytes = hex::decode(&c.x_hex).context("decoding x_hex for GPU trace-gen")?;
            x_states.extend_from_slice(&state_to_limbs(&bytes));
        }
        let (main_dev, _lo, d_cols) = match &dp {
            // Multi-GPU: use THIS device's N3 buffers (feeding device-0 buffers to a device-n kernel
            // would be an illegal cross-device access).
            Some(dp) => tracegen::gpu_gen_main_trace_device_d(
                dp.d_gates,
                &x_states,
                dp.d_off_lo,
                dp.d_off_hi,
                k as u32,
                n_gates as u32,
                shard_samples as u32,
                padded_rows,
                log_n_rows,
            ),
            None => {
                let (gates_flat, _x, off_lo, off_hi) =
                    gpu_flat_inputs(gates, shard_cases, rc_lo_index, rc_lo_index)?;
                tracegen::gpu_gen_main_trace_device(
                    &gates_flat,
                    &x_states,
                    &off_lo,
                    &off_hi,
                    k as u32,
                    n_gates as u32,
                    shard_samples as u32,
                    padded_rows,
                    log_n_rows,
                )
            }
        }
        .map_err(|e| anyhow::anyhow!(e))?;
        let mut main_dev = main_dev;
        main_dev.extend(to_prover(small_main));
        tree_builder.extend_evals(main_dev);
        d_cols
    };
    #[cfg(not(feature = "cuda"))]
    {
        let mut main_trace = generate_main_trace(&rows, padded_rows, log_n_rows);
        main_trace.extend(small_main);
        tree_builder.extend_evals(to_prover(main_trace));
    }
    tree_builder.commit(prover_channel);

    // Hold the ~24 GB main-trace device buffer resident from the tree1 commit through K4.
    #[cfg(feature = "cuda")]
    let main_k1: tracegen::MainTrace =
        tracegen::MainTrace::from_k1(d_main_cols).map_err(|e| anyhow::anyhow!(e))?;

    let interaction_pow_nonce = ProverBackend::grind(prover_channel, INTERACTION_POW_BITS);
    prover_channel.mix_u64(interaction_pow_nonce);
    let elements = LookupElements::draw(prover_channel);

    #[cfg(feature = "cuda")]
    {
        gate_air_cuda_kernel::register();
        let (z, alpha_powers) = tracegen::gate_air_relation_m31x4(&elements.qubitmem);
        gate_air_cuda_kernel::set_gate_air_relation(z, alpha_powers);
    }

    // Interaction traces (device K4 unconditional under `cuda`; CPU arm survives only on non-cuda).
    #[cfg(feature = "cuda")]
    let main_interaction_device = {
        let (cols, claimed) = tracegen::gpu_gen_interaction_device(
            &main_k1,
            n_gates as u32,
            padded_rows,
            log_n_rows,
            real_rows as u64,
            (k * n_gates) as u64,
            &elements,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        (cols, claimed)
    };
    // Free the ~24 GB resident buffer now (before tree2, not end-of-shard) so tree2's pool can reserve
    // it.
    #[cfg(feature = "cuda")]
    main_k1.free_after_k4().map_err(|e| anyhow::anyhow!(e))?;
    // GPU path: skip CPU interaction gen (the dominant cost); claimed_sum == CPU main_sum.
    #[cfg(feature = "cuda")]
    let main_sum = main_interaction_device.1;
    #[cfg(not(feature = "cuda"))]
    let (main_interaction, main_sum) =
        gen_main_interaction(&rows, padded_rows, log_n_rows, n_gates, &elements);
    // H_P binding: program supply carries both an internal (-mult, TAG_PROGRAM) and a public
    // (+mult, TAG_PROGRAM_PUB) term, paired into one batch (still 4 interaction cols).
    let (program_interaction, program_sum) = gen_program_interaction(program, &elements.program);
    let (boundary_interaction, boundary_sum) =
        gen_boundary_interaction(&boundary, &elements.qubitmem);
    // rc supply: -multiplicity / combine(TAG_RC, val).
    let (rc_interaction, rc_sum) = {
        let el = elements.rc.clone();
        gen_table_interaction(&rc_table.multiplicity, rc_table.log_size, |vec_row| {
            el.combine(&[ptag(TAG_RC), pack_seq(&rc_table.val, vec_row)])
        })
    };

    // x/y binding + H_P program binding: the base is NOT internally balanced. Boundary re-keys y to
    // TS_FINAL, leaving B = Σ(+[0,x] − [TS_FINAL,y]); program supply adds a public P_pub (its internal
    // -mult/TAG_PROGRAM term cancels main's demand). So the base's claimed sums net to B + P_pub, and
    // the leaf's public_logup_sum supplies −B and −P_pub over guessed values, forcing guessed ==
    // committed. rc demand and supply cancel. (stwo verify does not require Σ = 0; this is a self-check.)
    // Debug-only prover self-check that the claimed sums net to B + P_pub (see `diag`).
    #[cfg(debug_assertions)]
    crate::diag::assert_claimed_sums_net(
        &boundary,
        program,
        &elements,
        main_sum,
        program_sum,
        boundary_sum,
        rc_sum,
    )?;

    let claimed_sums = vec![main_sum, program_sum, boundary_sum, rc_sum];
    prover_channel.mix_felts(&claimed_sums);

    // Tree 2: interaction (same component order as the claimed sums): main, program, boundary, rc.
    let small_interaction = {
        let mut v = program_interaction;
        v.extend(boundary_interaction);
        v.extend(rc_interaction);
        v
    };
    let mut tree_builder = commitment_scheme.tree_builder();
    #[cfg(feature = "cuda")]
    {
        let mut interaction = main_interaction_device.0;
        interaction.extend(to_prover(small_interaction));
        tree_builder.extend_evals(interaction);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let mut interaction = main_interaction;
        interaction.extend(small_interaction);
        tree_builder.extend_evals(to_prover(interaction));
    }
    tree_builder.commit(prover_channel);

    let components = build_components(
        log_n_rows,
        program.log_size,
        boundary.log_size,
        rc_log,
        &elements,
        main_sum,
        program_sum,
        boundary_sum,
        rc_sum,
    );
    let prover_refs = components.prover_refs();
    let extended = prove_ex::<ProverBackend, Blake2sM31MerkleChannel>(
        &prover_refs,
        prover_channel,
        commitment_scheme,
        false,
    )?;

    // Per-shard boundary (x->y per shard shot) for the leaf's GateAirStatement + output hash.
    let mut shard_boundary = Vec::with_capacity(shard_cases.len());
    for case in shard_cases {
        let x = state_to_limbs(&hex::decode(&case.x_hex)?);
        let y = state_to_limbs(&hex::decode(&case.y_hex)?);
        shard_boundary.push((x, y));
    }
    let claim: Vec<SecureField> = vec![main_sum, program_sum, boundary_sum, rc_sum];
    Ok((
        extended,
        claim,
        interaction_pow_nonce,
        channel_salt,
        log_n_rows,
        program.log_size,
        shard_boundary,
        (n_gates * k) as u32,
    ))
}
