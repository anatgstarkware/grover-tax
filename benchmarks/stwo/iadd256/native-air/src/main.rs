use std::fs::{self, File};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use ruint::aliases::U256;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest as Sha2Digest, Sha256};
use sha3::digest::{ExtendableOutput, Update as XofUpdate};
use sha3::Shake256;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver, ComponentProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation, RelationEntry,
    TraceLocationAllocator,
};
use zkp_ecc_lib::{Circuit, Simulator};

const LIMBS: usize = 9;
const LIMB_BITS: usize = 29;
const TOP_LIMB_BITS: usize = 256 - LIMB_BITS * (LIMBS - 1);
const COMPACT_TRACE_COLUMNS: usize = 3 + LIMBS * 4;
const LOOKUP_CHUNK_BITS: usize = 18;
const LOOKUP_TRACE_COLUMNS: usize = COMPACT_TRACE_COLUMNS + LIMBS * 2;
const LOOKUP_MAIN_BATCHES: usize = 1 + LIMBS;
const LOOKUP_SEQ18_LOG_SIZE: u32 = LOOKUP_CHUNK_BITS as u32;
const LOOKUP_SEQ11_LOG_SIZE: u32 = (LIMB_BITS - LOOKUP_CHUNK_BITS) as u32;
const LOOKUP_SEQ6_LOG_SIZE: u32 = (TOP_LIMB_BITS - LOOKUP_CHUNK_BITS) as u32;
const IADD_STATE_WIDTH: usize = 2 + 2 * LIMBS;
const M31_MODULUS_U32: u32 = (1 << 31) - 1;
const LANE_COUNT: usize = 1 << LOG_N_LANES;
const KMX_REPEAT_CHECK_LIMIT: usize = 16;

type NativeIaddComponent = FrameworkComponent<NativeIaddEval>;
type Seq18LookupComponent = FrameworkComponent<Seq18LookupEval>;
type Seq11LookupComponent = FrameworkComponent<Seq11LookupEval>;
type Seq6LookupComponent = FrameworkComponent<Seq6LookupEval>;

stwo_constraint_framework::relation!(IaddStateElements, IADD_STATE_WIDTH);
stwo_constraint_framework::relation!(RangeSeq18Elements, 1);
stwo_constraint_framework::relation!(RangeSeq11Elements, 1);
stwo_constraint_framework::relation!(RangeSeq6Elements, 1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum RangeCheckMode {
    /// Compact benchmarking AIR. Assumes the witness generator emits canonical limbs.
    Off,
    /// Sound AIR mode. Proves every acc/addend/next limb by bit decomposition.
    Bits,
    /// Sound AIR mode. Proves canonical next limbs via lookup-backed chunk range checks.
    Lookup,
}

impl RangeCheckMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Bits => "bits",
            Self::Lookup => "lookup",
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Grover-tax v0.3-iadd fixture JSON.
    #[arg(long, default_value = "../../../../fixtures/v0.3-iadd256-k4-n16.json")]
    fixture: PathBuf,

    /// Limit samples from the fixture. Defaults to all fixture samples.
    #[arg(long)]
    samples: Option<usize>,

    /// Override repetitions. Defaults to fixture repetitions.
    #[arg(long)]
    repetitions: Option<usize>,

    /// Generate and check the native witness but skip STWO proof generation.
    #[arg(long)]
    no_prove: bool,

    /// Use a lower-memory STWO decommit path.
    #[arg(long)]
    low_memory: bool,

    /// Store polynomial coefficients in the commitment scheme. Faster decommitment, higher RSS.
    #[arg(long)]
    store_coefficients: bool,

    /// Limb range-check strategy.
    #[arg(long, value_enum, default_value_t = RangeCheckMode::Off)]
    range_check: RangeCheckMode,

    /// Run the upstream Kickmix parser/simulator against selected fixture samples.
    #[arg(long)]
    check_kmx: bool,

    /// Kickmix circuit path used by --check-kmx and artifact metadata.
    #[arg(
        long,
        default_value = "../external/zkp_ecc/docs/example_data/iadd256.kmx"
    )]
    kmx: PathBuf,

    /// Maximum selected samples to run through the KMX simulator.
    #[arg(long, default_value_t = 64)]
    kmx_check_samples: usize,

    /// Write a pretty JSON benchmark artifact to this path.
    #[arg(long)]
    artifact: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    version: String,
    circuit_source: String,
    repetitions: usize,
    n_samples: usize,
    register_width: usize,
    test_cases: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    #[serde(deserialize_with = "deserialize_biguint")]
    r0_in: BigUint,
    #[serde(deserialize_with = "deserialize_biguint")]
    r1_in: BigUint,
    y_hex: String,
}

#[derive(Serialize)]
struct RunReport {
    schema: &'static str,
    metadata: RunMetadata,
    config: RunConfig,
    metrics: RunMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    kmx_equivalence: Option<KmxEquivalenceReport>,
}

#[derive(Serialize)]
struct RunConfig {
    samples: usize,
    repetitions: usize,
    range_check: RangeCheckMode,
    low_memory: bool,
    store_coefficients: bool,
    no_prove: bool,
}

#[derive(Serialize)]
struct RunMetrics {
    real_rows: usize,
    padded_rows: usize,
    log_rows: u32,
    trace_columns: usize,
    main_trace_cells: u64,
    main_trace_bytes: u64,
    trace_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    twiddle_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prove_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verify_s: Option<f64>,
    proved: bool,
}

#[derive(Serialize)]
struct RunMetadata {
    generated_unix_s: u64,
    executable: String,
    host_arch: String,
    host_os: String,
    rustc: Option<String>,
    fixture_path: String,
    fixture_sha256: String,
    fixture_version: String,
    circuit_source: String,
    kmx_path: String,
    kmx_sha256: Option<String>,
    stwo_commit: Option<String>,
    stwo_cairo_commit: Option<String>,
    zkp_ecc_commit: Option<String>,
    grover_tax_commit: Option<String>,
}

#[derive(Serialize)]
struct KmxEquivalenceReport {
    kmx_path: String,
    kmx_sha256: String,
    circuit_operations: usize,
    circuit_num_qubits: u64,
    circuit_num_bits: u64,
    circuit_registers: usize,
    checked_samples: usize,
    one_step_matches_native_add: bool,
    repeated_fixture_checks: usize,
    repeated_check_limit: usize,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    is_real: u32,
    sample_id: u32,
    rep_idx: u32,
    acc: [u32; LIMBS],
    addend: [u32; LIMBS],
    next: [u32; LIMBS],
    carry: [u32; LIMBS],
}

#[derive(Clone)]
struct RangeLookupElements {
    seq18: RangeSeq18Elements,
    seq11: RangeSeq11Elements,
    seq6: RangeSeq6Elements,
}

impl RangeLookupElements {
    fn draw(channel: &mut impl Channel) -> Self {
        Self {
            seq18: RangeSeq18Elements::draw(channel),
            seq11: RangeSeq11Elements::draw(channel),
            seq6: RangeSeq6Elements::draw(channel),
        }
    }
}

#[derive(Default)]
struct LookupCounts {
    seq18: Vec<u32>,
    seq11: Vec<u32>,
    seq6: Vec<u32>,
}

impl LookupCounts {
    fn new() -> Self {
        Self {
            seq18: vec![0; 1 << LOOKUP_SEQ18_LOG_SIZE],
            seq11: vec![0; 1 << LOOKUP_SEQ11_LOG_SIZE],
            seq6: vec![0; 1 << LOOKUP_SEQ6_LOG_SIZE],
        }
    }

    fn add_next_limbs(&mut self, next: &[u32; LIMBS]) {
        for (limb_idx, &value) in next.iter().enumerate() {
            let (lo, hi) = split_lookup_chunks(limb_idx, value);
            self.seq18[lo as usize] += 1;
            if limb_idx + 1 == LIMBS {
                self.seq6[hi as usize] += 1;
            } else {
                self.seq11[hi as usize] += 1;
            }
        }
    }
}

struct NativeComponents {
    main: NativeIaddComponent,
    seq18: Option<Seq18LookupComponent>,
    seq11: Option<Seq11LookupComponent>,
    seq6: Option<Seq6LookupComponent>,
}

impl NativeComponents {
    fn component_refs(&self) -> Vec<&dyn Component> {
        let mut components = vec![&self.main as &dyn Component];
        if let Some(component) = &self.seq18 {
            components.push(component as &dyn Component);
        }
        if let Some(component) = &self.seq11 {
            components.push(component as &dyn Component);
        }
        if let Some(component) = &self.seq6 {
            components.push(component as &dyn Component);
        }
        components
    }

    fn prover_refs(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        let mut components = vec![&self.main as &dyn ComponentProver<SimdBackend>];
        if let Some(component) = &self.seq18 {
            components.push(component as &dyn ComponentProver<SimdBackend>);
        }
        if let Some(component) = &self.seq11 {
            components.push(component as &dyn ComponentProver<SimdBackend>);
        }
        if let Some(component) = &self.seq6 {
            components.push(component as &dyn ComponentProver<SimdBackend>);
        }
        components
    }

    fn trace_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::concat_cols(
            self.component_refs()
                .into_iter()
                .map(|c| c.trace_log_degree_bounds()),
        )
    }
}

impl Row {
    fn zero() -> Self {
        Self {
            is_real: 0,
            sample_id: 0,
            rep_idx: 0,
            acc: [0; LIMBS],
            addend: [0; LIMBS],
            next: [0; LIMBS],
            carry: [0; LIMBS],
        }
    }
}

#[derive(Clone)]
struct NativeIaddEval {
    log_n_rows: u32,
    state_elements: IaddStateElements,
    range_check: RangeCheckMode,
    range_lookup_elements: Option<RangeLookupElements>,
}

impl FrameworkEval for NativeIaddEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real = eval.next_trace_mask();
        let sample_id = eval.next_trace_mask();
        let rep_idx = eval.next_trace_mask();

        let mut acc = Vec::with_capacity(LIMBS);
        for _ in 0..LIMBS {
            acc.push(eval.next_trace_mask());
        }

        let mut addend = Vec::with_capacity(LIMBS);
        for _ in 0..LIMBS {
            addend.push(eval.next_trace_mask());
        }

        let next = (0..LIMBS).map(|_| eval.next_trace_mask()).collect_vec();
        let carry = (0..LIMBS).map(|_| eval.next_trace_mask()).collect_vec();

        let one = E::F::one();
        eval.add_constraint(is_real.clone() * (is_real.clone() - one.clone()));
        for i in 0..LIMBS {
            eval.add_constraint(
                is_real.clone() * carry[i].clone() * (carry[i].clone() - one.clone()),
            );
        }

        for i in 0..LIMBS {
            let limb_base = BaseField::from_u32_unchecked(limb_base(i));
            let carry_in = if i == 0 {
                E::F::zero()
            } else {
                carry[i - 1].clone()
            };
            let limb_relation = acc[i].clone() + addend[i].clone() + carry_in
                - next[i].clone()
                - carry[i].clone() * limb_base;
            eval.add_constraint(is_real.clone() * limb_relation);
        }

        if self.range_check == RangeCheckMode::Bits {
            add_limb_bit_range_constraints(&mut eval, &is_real, &acc);
            add_limb_bit_range_constraints(&mut eval, &is_real, &addend);
            add_limb_bit_range_constraints(&mut eval, &is_real, &next);
        }

        let mut input_state = Vec::with_capacity(IADD_STATE_WIDTH);
        input_state.push(sample_id.clone());
        input_state.push(rep_idx.clone());
        input_state.extend(acc.iter().cloned());
        input_state.extend(addend.iter().cloned());

        let mut output_state = Vec::with_capacity(IADD_STATE_WIDTH);
        output_state.push(sample_id);
        output_state.push(rep_idx + one);
        output_state.extend(next.iter().cloned());
        output_state.extend(addend.iter().cloned());

        let multiplicity = E::EF::from(is_real.clone());
        eval.add_to_relation(RelationEntry::new(
            &self.state_elements,
            multiplicity.clone(),
            &input_state,
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.state_elements,
            -multiplicity,
            &output_state,
        ));
        if self.range_check == RangeCheckMode::Lookup {
            let lookup_elements = self
                .range_lookup_elements
                .as_ref()
                .expect("lookup mode requires lookup elements");
            add_lookup_range_constraints(&mut eval, &is_real, &next, lookup_elements);
        }
        eval.finalize_logup_in_pairs();

        eval
    }
}

#[derive(Clone)]
struct Seq18LookupEval {
    elements: RangeSeq18Elements,
}

impl FrameworkEval for Seq18LookupEval {
    fn log_size(&self) -> u32 {
        LOOKUP_SEQ18_LOG_SIZE
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let value = eval.get_preprocessed_column(range_seq18_column_id());
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[value],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct Seq11LookupEval {
    elements: RangeSeq11Elements,
}

impl FrameworkEval for Seq11LookupEval {
    fn log_size(&self) -> u32 {
        LOOKUP_SEQ11_LOG_SIZE
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let value = eval.get_preprocessed_column(range_seq11_column_id());
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[value],
        ));
        eval.finalize_logup();
        eval
    }
}

#[derive(Clone)]
struct Seq6LookupEval {
    elements: RangeSeq6Elements,
}

impl FrameworkEval for Seq6LookupEval {
    fn log_size(&self) -> u32 {
        LOOKUP_SEQ6_LOG_SIZE
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size() + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let value = eval.get_preprocessed_column(range_seq6_column_id());
        let multiplicity = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(
            &self.elements,
            -E::EF::from(multiplicity),
            &[value],
        ));
        eval.finalize_logup();
        eval
    }
}

fn add_limb_bit_range_constraints<E: EvalAtRow>(eval: &mut E, is_real: &E::F, limbs: &[E::F]) {
    let one = E::F::one();
    for (limb_idx, limb) in limbs.iter().enumerate() {
        let mut reconstructed = E::F::zero();
        for bit_idx in 0..limb_bits(limb_idx) {
            let bit = eval.next_trace_mask();
            eval.add_constraint(is_real.clone() * bit.clone() * (bit.clone() - one.clone()));
            reconstructed += bit * BaseField::from_u32_unchecked(1u32 << bit_idx);
        }
        eval.add_constraint(is_real.clone() * (limb.clone() - reconstructed));
    }
}

fn add_lookup_range_constraints<E: EvalAtRow>(
    eval: &mut E,
    is_real: &E::F,
    next: &[E::F],
    lookup_elements: &RangeLookupElements,
) {
    let lookup_base = BaseField::from_u32_unchecked(1u32 << LOOKUP_CHUNK_BITS);
    let multiplicity = E::EF::from(is_real.clone());
    for (limb_idx, limb) in next.iter().enumerate() {
        let lo = eval.next_trace_mask();
        let hi = eval.next_trace_mask();
        eval.add_constraint(
            is_real.clone() * (limb.clone() - lo.clone() - hi.clone() * lookup_base),
        );
        eval.add_to_relation(RelationEntry::new(
            &lookup_elements.seq18,
            multiplicity.clone(),
            &[lo],
        ));
        if limb_idx + 1 == LIMBS {
            eval.add_to_relation(RelationEntry::new(
                &lookup_elements.seq6,
                multiplicity.clone(),
                &[hi],
            ));
        } else {
            eval.add_to_relation(RelationEntry::new(
                &lookup_elements.seq11,
                multiplicity.clone(),
                &[hi],
            ));
        }
    }
}

fn build_components(
    log_n_rows: u32,
    range_check: RangeCheckMode,
    state_elements: IaddStateElements,
    range_lookup_elements: Option<RangeLookupElements>,
    main_claimed_sum: SecureField,
    seq18_claimed_sum: Option<SecureField>,
    seq11_claimed_sum: Option<SecureField>,
    seq6_claimed_sum: Option<SecureField>,
) -> NativeComponents {
    let mut allocator = if range_check == RangeCheckMode::Lookup {
        TraceLocationAllocator::new_with_preprocessed_columns(&lookup_preprocessed_columns())
    } else {
        TraceLocationAllocator::default()
    };

    let main = NativeIaddComponent::new(
        &mut allocator,
        NativeIaddEval {
            log_n_rows,
            state_elements,
            range_check,
            range_lookup_elements: range_lookup_elements.clone(),
        },
        main_claimed_sum,
    );

    let (seq18, seq11, seq6) = if range_check == RangeCheckMode::Lookup {
        (
            Some(Seq18LookupComponent::new(
                &mut allocator,
                Seq18LookupEval {
                    elements: range_lookup_elements
                        .as_ref()
                        .expect("lookup mode requires seq18 elements")
                        .seq18
                        .clone(),
                },
                seq18_claimed_sum.expect("lookup mode requires seq18 claimed sum"),
            )),
            Some(Seq11LookupComponent::new(
                &mut allocator,
                Seq11LookupEval {
                    elements: range_lookup_elements
                        .as_ref()
                        .expect("lookup mode requires seq11 elements")
                        .seq11
                        .clone(),
                },
                seq11_claimed_sum.expect("lookup mode requires seq11 claimed sum"),
            )),
            Some(Seq6LookupComponent::new(
                &mut allocator,
                Seq6LookupEval {
                    elements: range_lookup_elements
                        .as_ref()
                        .expect("lookup mode requires seq6 elements")
                        .seq6
                        .clone(),
                },
                seq6_claimed_sum.expect("lookup mode requires seq6 claimed sum"),
            )),
        )
    } else {
        (None, None, None)
    };

    NativeComponents {
        main,
        seq18,
        seq11,
        seq6,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fixture_path = normalize_fixture_path(args.fixture);
    let kmx_path = normalize_fixture_path(args.kmx);
    let fixture = load_fixture(&fixture_path)?;

    if fixture.version != "v0.3-iadd" {
        bail!("expected v0.3-iadd fixture, got {}", fixture.version);
    }
    if fixture.register_width != 256 {
        bail!("this prototype expects 256-bit iadd fixtures");
    }

    let samples = args
        .samples
        .unwrap_or(fixture.n_samples)
        .min(fixture.test_cases.len());
    let repetitions = args.repetitions.unwrap_or(fixture.repetitions);
    if samples == 0 || repetitions == 0 {
        bail!("samples and repetitions must be non-zero");
    }
    if samples >= M31_MODULUS_U32 as usize || repetitions >= M31_MODULUS_U32 as usize {
        bail!("samples and repetitions must fit in M31 without wrapping");
    }

    let selected = &fixture.test_cases[..samples];
    check_fixture_outputs(selected, repetitions)?;

    let target_rows = samples
        .checked_mul(repetitions)
        .ok_or_else(|| anyhow!("row count overflow"))?;
    let padded_rows = target_rows.next_power_of_two().max(1 << (LOG_N_LANES + 2));
    let log_n_rows = padded_rows.ilog2();
    let trace_columns = trace_columns(args.range_check);

    eprintln!(
        "native-iadd-air: circuit={} samples={} repetitions={} real_rows={} padded_rows={} log_rows={} columns={} range_check={}",
        fixture.circuit_source,
        samples,
        repetitions,
        target_rows,
        padded_rows,
        log_n_rows,
        trace_columns,
        args.range_check.as_str()
    );

    let trace_start = Instant::now();
    let (main_trace, lookup_counts) = generate_main_trace::<SimdBackend>(
        selected,
        repetitions,
        padded_rows,
        log_n_rows,
        args.range_check,
    );
    let trace_elapsed = trace_start.elapsed();
    eprintln!(
        "native-iadd-air: trace generated in {:.3}s",
        trace_elapsed.as_secs_f64()
    );

    let kmx_equivalence = if args.check_kmx {
        Some(check_kmx_equivalence(
            selected,
            repetitions,
            &kmx_path,
            args.kmx_check_samples,
        )?)
    } else {
        None
    };

    let metadata = RunMetadata::collect(&fixture_path, &kmx_path, &fixture)?;

    if args.no_prove {
        let report = RunReport {
            schema: "native-iadd-air-report/v1",
            metadata,
            config: RunConfig {
                samples,
                repetitions,
                range_check: args.range_check,
                low_memory: args.low_memory,
                store_coefficients: args.store_coefficients,
                no_prove: args.no_prove,
            },
            metrics: RunMetrics {
                real_rows: target_rows,
                padded_rows,
                log_rows: log_n_rows,
                trace_columns,
                main_trace_cells: padded_rows as u64 * trace_columns as u64,
                main_trace_bytes: padded_rows as u64 * trace_columns as u64 * 4,
                trace_s: trace_elapsed.as_secs_f64(),
                twiddle_s: None,
                prove_s: None,
                verify_s: None,
                proved: false,
            },
            kmx_equivalence,
        };
        emit_report(&report, args.artifact.as_ref())?;
        return Ok(());
    }

    let config = PcsConfig::default();
    let max_log_size = if args.range_check == RangeCheckMode::Lookup {
        log_n_rows.max(LOOKUP_SEQ18_LOG_SIZE)
    } else {
        log_n_rows
    };
    let twiddle_start = Instant::now();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let twiddle_elapsed = twiddle_start.elapsed();

    let prove_start = Instant::now();
    let prover_channel = &mut Blake2sM31Channel::default();
    config.mix_into(prover_channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
    if args.store_coefficients {
        commitment_scheme.set_store_polynomials_coefficients();
    }
    if args.low_memory {
        anyhow::bail!(
            "--low-memory requires a stwo build exposing set_low_memory(); \
             unavailable in the pinned upstream stwo"
        );
    }

    let mut tree_builder = commitment_scheme.tree_builder();
    if args.range_check == RangeCheckMode::Lookup {
        tree_builder.extend_evals(vec![
            generate_preprocessed_seq_trace::<SimdBackend>(LOOKUP_SEQ18_LOG_SIZE),
            generate_preprocessed_seq_trace::<SimdBackend>(LOOKUP_SEQ11_LOG_SIZE),
            generate_preprocessed_seq_trace::<SimdBackend>(LOOKUP_SEQ6_LOG_SIZE),
        ]);
    } else {
        tree_builder.extend_evals(vec![]);
    }
    tree_builder.commit(prover_channel);

    let mut original_trace = main_trace;
    if let Some(counts) = lookup_counts.as_ref() {
        original_trace.extend(generate_lookup_multiplicity_trace::<SimdBackend>(
            &counts.seq18,
        ));
        original_trace.extend(generate_lookup_multiplicity_trace::<SimdBackend>(
            &counts.seq11,
        ));
        original_trace.extend(generate_lookup_multiplicity_trace::<SimdBackend>(
            &counts.seq6,
        ));
    }
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(original_trace);
    tree_builder.commit(prover_channel);

    let state_elements = IaddStateElements::draw(prover_channel);
    let range_lookup_elements = (args.range_check == RangeCheckMode::Lookup)
        .then(|| RangeLookupElements::draw(prover_channel));
    let (main_interaction_trace, main_claimed_sum) = gen_main_interaction_trace(
        selected,
        repetitions,
        padded_rows,
        log_n_rows,
        &state_elements,
        args.range_check,
        range_lookup_elements.as_ref(),
    );
    let boundary_sum = public_boundary_sum(selected, repetitions, &state_elements)?;
    let lookup_sum = if let (Some(counts), Some(elements)) =
        (lookup_counts.as_ref(), range_lookup_elements.as_ref())
    {
        lookup_public_sum(counts, elements)
    } else {
        SecureField::zero()
    };
    let expected_main_sum = boundary_sum + lookup_sum;
    if main_claimed_sum != expected_main_sum {
        bail!("native transition claimed sum does not match fixture boundary sum");
    }
    let (seq18_interaction_trace, seq18_claimed_sum) = if let (Some(counts), Some(elements)) =
        (lookup_counts.as_ref(), range_lookup_elements.as_ref())
    {
        let (trace, claimed_sum) = gen_seq_lookup_interaction_trace::<LOOKUP_SEQ18_LOG_SIZE, _>(
            &counts.seq18,
            &elements.seq18,
        );
        let expected =
            -seq_lookup_public_sum::<LOOKUP_SEQ18_LOG_SIZE, _>(&counts.seq18, &elements.seq18);
        if claimed_sum != expected {
            bail!("seq18 lookup claimed sum mismatch");
        }
        (Some(trace), Some(claimed_sum))
    } else {
        (None, None)
    };
    let (seq11_interaction_trace, seq11_claimed_sum) = if let (Some(counts), Some(elements)) =
        (lookup_counts.as_ref(), range_lookup_elements.as_ref())
    {
        let (trace, claimed_sum) = gen_seq_lookup_interaction_trace::<LOOKUP_SEQ11_LOG_SIZE, _>(
            &counts.seq11,
            &elements.seq11,
        );
        let expected =
            -seq_lookup_public_sum::<LOOKUP_SEQ11_LOG_SIZE, _>(&counts.seq11, &elements.seq11);
        if claimed_sum != expected {
            bail!("seq11 lookup claimed sum mismatch");
        }
        (Some(trace), Some(claimed_sum))
    } else {
        (None, None)
    };
    let (seq6_interaction_trace, seq6_claimed_sum) = if let (Some(counts), Some(elements)) =
        (lookup_counts.as_ref(), range_lookup_elements.as_ref())
    {
        let (trace, claimed_sum) = gen_seq_lookup_interaction_trace::<LOOKUP_SEQ6_LOG_SIZE, _>(
            &counts.seq6,
            &elements.seq6,
        );
        let expected =
            -seq_lookup_public_sum::<LOOKUP_SEQ6_LOG_SIZE, _>(&counts.seq6, &elements.seq6);
        if claimed_sum != expected {
            bail!("seq6 lookup claimed sum mismatch");
        }
        (Some(trace), Some(claimed_sum))
    } else {
        (None, None)
    };
    let claimed_sums = std::iter::once(main_claimed_sum)
        .chain(seq18_claimed_sum)
        .chain(seq11_claimed_sum)
        .chain(seq6_claimed_sum)
        .collect_vec();
    prover_channel.mix_felts(&claimed_sums);

    let mut interaction_trace = main_interaction_trace;
    if let Some(trace) = seq18_interaction_trace {
        interaction_trace.extend(trace);
    }
    if let Some(trace) = seq11_interaction_trace {
        interaction_trace.extend(trace);
    }
    if let Some(trace) = seq6_interaction_trace {
        interaction_trace.extend(trace);
    }
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(prover_channel);

    let prover_components = build_components(
        log_n_rows,
        args.range_check,
        state_elements.clone(),
        range_lookup_elements.clone(),
        main_claimed_sum,
        seq18_claimed_sum,
        seq11_claimed_sum,
        seq6_claimed_sum,
    );
    let sizes = prover_components.trace_log_sizes();
    let prover_refs = prover_components.prover_refs();
    let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
        &prover_refs,
        prover_channel,
        commitment_scheme,
    )?;
    let prove_elapsed = prove_start.elapsed();

    let verify_start = Instant::now();
    let verifier_channel = &mut Blake2sM31Channel::default();
    config.mix_into(verifier_channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
    commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
    commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
    let verifier_state_elements = IaddStateElements::draw(verifier_channel);
    let verifier_range_lookup_elements = (args.range_check == RangeCheckMode::Lookup)
        .then(|| RangeLookupElements::draw(verifier_channel));
    let verifier_boundary_sum =
        public_boundary_sum(selected, repetitions, &verifier_state_elements)?;
    let verifier_lookup_sum = if let (Some(counts), Some(elements)) = (
        lookup_counts.as_ref(),
        verifier_range_lookup_elements.as_ref(),
    ) {
        lookup_public_sum(counts, elements)
    } else {
        SecureField::zero()
    };
    let verifier_main_sum = verifier_boundary_sum + verifier_lookup_sum;
    let verifier_claimed_sums = std::iter::once(verifier_main_sum)
        .chain(
            lookup_counts
                .as_ref()
                .zip(verifier_range_lookup_elements.as_ref())
                .map(|(counts, elements)| {
                    -seq_lookup_public_sum::<LOOKUP_SEQ18_LOG_SIZE, _>(
                        &counts.seq18,
                        &elements.seq18,
                    )
                }),
        )
        .chain(
            lookup_counts
                .as_ref()
                .zip(verifier_range_lookup_elements.as_ref())
                .map(|(counts, elements)| {
                    -seq_lookup_public_sum::<LOOKUP_SEQ11_LOG_SIZE, _>(
                        &counts.seq11,
                        &elements.seq11,
                    )
                }),
        )
        .chain(
            lookup_counts
                .as_ref()
                .zip(verifier_range_lookup_elements.as_ref())
                .map(|(counts, elements)| {
                    -seq_lookup_public_sum::<LOOKUP_SEQ6_LOG_SIZE, _>(&counts.seq6, &elements.seq6)
                }),
        )
        .collect_vec();
    verifier_channel.mix_felts(&verifier_claimed_sums);
    let verifier_components = build_components(
        log_n_rows,
        args.range_check,
        verifier_state_elements,
        verifier_range_lookup_elements,
        verifier_main_sum,
        verifier_claimed_sums.get(1).copied(),
        verifier_claimed_sums.get(2).copied(),
        verifier_claimed_sums.get(3).copied(),
    );
    commitment_scheme.commit(proof.commitments[2], &sizes[2], verifier_channel);
    verify(
        &verifier_components.component_refs(),
        verifier_channel,
        commitment_scheme,
        proof,
    )?;
    let verify_elapsed = verify_start.elapsed();

    let report = RunReport {
        schema: "native-iadd-air-report/v1",
        metadata,
        config: RunConfig {
            samples,
            repetitions,
            range_check: args.range_check,
            low_memory: args.low_memory,
            store_coefficients: args.store_coefficients,
            no_prove: args.no_prove,
        },
        metrics: RunMetrics {
            real_rows: target_rows,
            padded_rows,
            log_rows: log_n_rows,
            trace_columns,
            main_trace_cells: padded_rows as u64 * trace_columns as u64,
            main_trace_bytes: padded_rows as u64 * trace_columns as u64 * 4,
            trace_s: trace_elapsed.as_secs_f64(),
            twiddle_s: Some(twiddle_elapsed.as_secs_f64()),
            prove_s: Some(prove_elapsed.as_secs_f64()),
            verify_s: Some(verify_elapsed.as_secs_f64()),
            proved: true,
        },
        kmx_equivalence,
    };
    emit_report(&report, args.artifact.as_ref())?;

    Ok(())
}

fn normalize_fixture_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

fn load_fixture(path: &PathBuf) -> Result<Fixture> {
    let file = File::open(path).with_context(|| format!("opening fixture {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parsing fixture {}", path.display()))
}

impl RunMetadata {
    fn collect(fixture_path: &PathBuf, kmx_path: &PathBuf, fixture: &Fixture) -> Result<Self> {
        Ok(Self {
            generated_unix_s: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            executable: std::env::current_exe()
                .ok()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            host_arch: std::env::consts::ARCH.to_string(),
            host_os: std::env::consts::OS.to_string(),
            rustc: command_output("rustc", &["-Vv"]),
            fixture_path: fixture_path.display().to_string(),
            fixture_sha256: sha256_file(fixture_path)?,
            fixture_version: fixture.version.clone(),
            circuit_source: fixture.circuit_source.clone(),
            kmx_path: kmx_path.display().to_string(),
            kmx_sha256: sha256_file(kmx_path).ok(),
            stwo_commit: git_commit("../../../../../stwo"),
            stwo_cairo_commit: git_commit("../../../../../stwo-cairo"),
            zkp_ecc_commit: git_commit("../external/zkp_ecc"),
            grover_tax_commit: git_commit("../../../.."),
        })
    }
}

fn emit_report(report: &RunReport, artifact_path: Option<&PathBuf>) -> Result<()> {
    println!("{}", serde_json::to_string(report)?);
    if let Some(path) = artifact_path {
        let normalized = normalize_fixture_path(path.clone());
        if let Some(parent) = normalized.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating artifact directory {}", parent.display()))?;
        }
        fs::write(&normalized, serde_json::to_string_pretty(report)?)
            .with_context(|| format!("writing artifact {}", normalized.display()))?;
    }
    Ok(())
}

fn sha256_file(path: &PathBuf) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_commit(repo: &str) -> Option<String> {
    let repo_path = normalize_fixture_path(PathBuf::from(repo));
    command_output(
        "git",
        &["-C", repo_path.to_str()?, "rev-parse", "--verify", "HEAD"],
    )
}

fn deserialize_biguint<'de, D>(deserializer: D) -> std::result::Result<BigUint, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let s = match value {
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s,
        other => {
            return Err(serde::de::Error::custom(format!(
                "expected decimal integer, got {other}"
            )))
        }
    };
    BigUint::parse_bytes(s.as_bytes(), 10)
        .ok_or_else(|| serde::de::Error::custom(format!("invalid decimal integer {s}")))
}

fn check_fixture_outputs(cases: &[TestCase], repetitions: usize) -> Result<()> {
    let modulus = BigUint::one() << 256usize;
    let reps = BigUint::from(repetitions as u64);
    for (i, case) in cases.iter().enumerate() {
        let expected = (&case.r0_in + &case.r1_in * &reps) % &modulus;
        let actual = low_register_from_state_hex(&case.y_hex)?;
        if expected != actual {
            bail!("fixture output mismatch at case {i}");
        }
    }
    Ok(())
}

fn check_kmx_equivalence(
    cases: &[TestCase],
    repetitions: usize,
    kmx_path: &PathBuf,
    sample_limit: usize,
) -> Result<KmxEquivalenceReport> {
    let circuit =
        Circuit::from_kmx(kmx_path).with_context(|| format!("parsing {}", kmx_path.display()))?;
    if circuit.registers.len() < 2 {
        bail!(
            "expected the KMX circuit to expose at least 2 registers, got {}",
            circuit.registers.len()
        );
    }
    if circuit.registers[0].len() != 256 || circuit.registers[1].len() != 256 {
        bail!(
            "expected 256-bit r0/r1 registers, got {}/{} bits",
            circuit.registers[0].len(),
            circuit.registers[1].len()
        );
    }

    let checked_samples = sample_limit.min(cases.len());
    let modulus = BigUint::one() << 256usize;
    for (sample_idx, case) in cases.iter().take(checked_samples).enumerate() {
        let (out0, out1) = simulate_kmx_repetitions(&circuit, case, 1, sample_idx as u64)?;
        let expected_once = (&case.r0_in + &case.r1_in) % &modulus;
        if u256_to_biguint(out0) != expected_once {
            bail!("KMX one-step output mismatch for sample {sample_idx}");
        }
        if u256_to_biguint(out1) != case.r1_in {
            bail!("KMX one-step addend register mismatch for sample {sample_idx}");
        }
    }

    let repeated_fixture_checks = if repetitions <= KMX_REPEAT_CHECK_LIMIT {
        for (sample_idx, case) in cases.iter().take(checked_samples).enumerate() {
            let (out0, out1) =
                simulate_kmx_repetitions(&circuit, case, repetitions, sample_idx as u64)?;
            let expected_final = low_register_from_state_hex(&case.y_hex)?;
            if u256_to_biguint(out0) != expected_final {
                bail!("KMX repeated output mismatch for sample {sample_idx}");
            }
            if u256_to_biguint(out1) != case.r1_in {
                bail!("KMX repeated addend register mismatch for sample {sample_idx}");
            }
        }
        checked_samples
    } else {
        0
    };

    Ok(KmxEquivalenceReport {
        kmx_path: kmx_path.display().to_string(),
        kmx_sha256: sha256_file(kmx_path)?,
        circuit_operations: circuit.operations.len(),
        circuit_num_qubits: circuit.num_qubits,
        circuit_num_bits: circuit.num_bits,
        circuit_registers: circuit.registers.len(),
        checked_samples,
        one_step_matches_native_add: true,
        repeated_fixture_checks,
        repeated_check_limit: KMX_REPEAT_CHECK_LIMIT,
    })
}

fn simulate_kmx_repetitions(
    circuit: &Circuit,
    case: &TestCase,
    repetitions: usize,
    seed_suffix: u64,
) -> Result<(U256, U256)> {
    let mut hasher = Shake256::default();
    hasher.update(b"native-iadd-air kmx equivalence");
    hasher.update(&seed_suffix.to_le_bytes());
    hasher.update(&repetitions.to_le_bytes());
    let mut xof = hasher.finalize_xof();
    let mut sim = Simulator::new(
        circuit.num_qubits as usize,
        circuit.num_bits as usize,
        &mut xof,
    );
    sim.set_register(&circuit.registers[0], biguint_to_u256(&case.r0_in)?, 0);
    sim.set_register(&circuit.registers[1], biguint_to_u256(&case.r1_in)?, 0);
    for _ in 0..repetitions {
        sim.apply_iter(circuit.operations.iter());
    }
    Ok((
        sim.get_register(&circuit.registers[0], 0),
        sim.get_register(&circuit.registers[1], 0),
    ))
}

fn biguint_to_u256(value: &BigUint) -> Result<U256> {
    U256::from_str_radix(&value.to_str_radix(10), 10)
        .map_err(|e| anyhow!("converting BigUint to U256: {e}"))
}

fn u256_to_biguint(value: U256) -> BigUint {
    BigUint::parse_bytes(value.to_string().as_bytes(), 10).expect("U256 decimal parses as BigUint")
}

fn generate_main_trace<B: stwo::prover::backend::Backend>(
    cases: &[TestCase],
    repetitions: usize,
    padded_rows: usize,
    log_n_rows: u32,
    range_check: RangeCheckMode,
) -> (
    ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    Option<LookupCounts>,
) {
    let mut cols = (0..trace_columns(range_check))
        .map(|_| Col::<B, BaseField>::zeros(padded_rows))
        .collect_vec();
    let mut lookup_counts = (range_check == RangeCheckMode::Lookup).then(LookupCounts::new);

    let mut row_idx = 0;
    for (sample_idx, case) in cases.iter().enumerate() {
        let addend = biguint_to_limbs(&case.r1_in);
        let mut acc = biguint_to_limbs(&case.r0_in);
        for rep_idx in 0..repetitions {
            let (next, carry) = add_limbs(acc, addend);
            let row = Row {
                is_real: 1,
                sample_id: sample_idx as u32,
                rep_idx: rep_idx as u32,
                acc,
                addend,
                next,
                carry,
            };
            write_row_to_cols::<B>(&mut cols, row_idx, &row, range_check);
            if let Some(counts) = lookup_counts.as_mut() {
                counts.add_next_limbs(&row.next);
            }
            row_idx += 1;
            acc = next;
        }
    }
    debug_assert_eq!(row_idx, cases.len() * repetitions);

    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    (
        cols.into_iter()
            .map(|eval| CircleEvaluation::<B, _, BitReversedOrder>::new(domain, eval))
            .collect_vec(),
        lookup_counts,
    )
}

fn write_row_to_cols<B: stwo::prover::backend::Backend>(
    cols: &mut [Col<B, BaseField>],
    row_idx: usize,
    row: &Row,
    range_check: RangeCheckMode,
) {
    let mut col = 0;
    set_col::<B>(cols, col, row_idx, row.is_real);
    col += 1;
    set_col::<B>(cols, col, row_idx, row.sample_id);
    col += 1;
    set_col::<B>(cols, col, row_idx, row.rep_idx);
    col += 1;
    for value in row.acc {
        set_col::<B>(cols, col, row_idx, value);
        col += 1;
    }
    for value in row.addend {
        set_col::<B>(cols, col, row_idx, value);
        col += 1;
    }
    for value in row.next {
        set_col::<B>(cols, col, row_idx, value);
        col += 1;
    }
    for value in row.carry {
        set_col::<B>(cols, col, row_idx, value);
        col += 1;
    }
    if range_check == RangeCheckMode::Bits {
        write_limb_bits::<B>(cols, &mut col, row_idx, &row.acc);
        write_limb_bits::<B>(cols, &mut col, row_idx, &row.addend);
        write_limb_bits::<B>(cols, &mut col, row_idx, &row.next);
    } else if range_check == RangeCheckMode::Lookup {
        write_lookup_chunks::<B>(cols, &mut col, row_idx, &row.next);
    }
    debug_assert_eq!(col, trace_columns(range_check));
}

fn write_limb_bits<B: stwo::prover::backend::Backend>(
    cols: &mut [Col<B, BaseField>],
    col: &mut usize,
    row_idx: usize,
    limbs: &[u32; LIMBS],
) {
    for (limb_idx, limb) in limbs.iter().enumerate() {
        for bit_idx in 0..limb_bits(limb_idx) {
            set_col::<B>(cols, *col, row_idx, (limb >> bit_idx) & 1);
            *col += 1;
        }
    }
}

fn write_lookup_chunks<B: stwo::prover::backend::Backend>(
    cols: &mut [Col<B, BaseField>],
    col: &mut usize,
    row_idx: usize,
    next: &[u32; LIMBS],
) {
    for (limb_idx, &value) in next.iter().enumerate() {
        let (lo, hi) = split_lookup_chunks(limb_idx, value);
        set_col::<B>(cols, *col, row_idx, lo);
        *col += 1;
        set_col::<B>(cols, *col, row_idx, hi);
        *col += 1;
    }
}

fn set_col<B: stwo::prover::backend::Backend>(
    cols: &mut [Col<B, BaseField>],
    col: usize,
    row: usize,
    value: u32,
) {
    cols[col].set(row, BaseField::from_u32_unchecked(value));
}

fn range_seq18_column_id() -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: "native_iadd_range_seq18".to_owned(),
    }
}

fn range_seq11_column_id() -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: "native_iadd_range_seq11".to_owned(),
    }
}

fn range_seq6_column_id() -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: "native_iadd_range_seq6".to_owned(),
    }
}

fn lookup_preprocessed_columns() -> [PreProcessedColumnId; 3] {
    [
        range_seq18_column_id(),
        range_seq11_column_id(),
        range_seq6_column_id(),
    ]
}

fn split_lookup_chunks(limb_idx: usize, value: u32) -> (u32, u32) {
    debug_assert!(value < limb_base(limb_idx));
    let lo_mask = (1u32 << LOOKUP_CHUNK_BITS) - 1;
    let lo = value & lo_mask;
    let hi = value >> LOOKUP_CHUNK_BITS;
    debug_assert!(hi < (1u32 << lookup_high_bits(limb_idx)));
    (lo, hi)
}

fn lookup_high_bits(limb_idx: usize) -> usize {
    limb_bits(limb_idx) - LOOKUP_CHUNK_BITS
}

fn generate_preprocessed_seq_trace<B: stwo::prover::backend::Backend>(
    log_size: u32,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);
    for i in 0..(1 << log_size) {
        col.set(i, BaseField::from_u32_unchecked(i as u32));
    }
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

fn generate_lookup_multiplicity_trace<B: stwo::prover::backend::Backend>(
    counts: &[u32],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let log_size = counts.len().ilog2();
    let mut col = Col::<B, BaseField>::zeros(counts.len());
    for (row, &count) in counts.iter().enumerate() {
        col.set(row, BaseField::from_u32_unchecked(count));
    }
    vec![CircleEvaluation::new(
        CanonicCoset::new(log_size).circle_domain(),
        col,
    )]
}

fn add_limbs(a: [u32; LIMBS], b: [u32; LIMBS]) -> ([u32; LIMBS], [u32; LIMBS]) {
    let mut out = [0u32; LIMBS];
    let mut carry = [0u32; LIMBS];
    let mut carry_in = 0u32;
    for i in 0..LIMBS {
        let total = a[i] as u64 + b[i] as u64 + carry_in as u64;
        out[i] = (total & (limb_base(i) as u64 - 1)) as u32;
        carry[i] = (total >> limb_bits(i)) as u32;
        carry_in = carry[i];
    }
    (out, carry)
}

fn biguint_to_limbs(value: &BigUint) -> [u32; LIMBS] {
    let mut limbs = [0u32; LIMBS];
    let mut offset = 0usize;
    for i in 0..LIMBS {
        let bits = limb_bits(i);
        let mask = (BigUint::one() << bits) - BigUint::one();
        let limb = (value >> offset) & mask;
        limbs[i] = limb.to_u32_digits().first().copied().unwrap_or(0);
        offset += bits;
    }
    limbs
}

fn low_register_from_state_hex(hex: &str) -> Result<BigUint> {
    let bytes = hex::decode(hex).context("decoding y_hex")?;
    if bytes.len() < 32 {
        bail!("expected at least 32 state bytes, got {}", bytes.len());
    }
    Ok(BigUint::from_bytes_le(&bytes[..32]))
}

#[derive(Clone, Copy)]
struct PackedFraction {
    numerator: PackedSecureField,
    denominator: PackedSecureField,
}

fn gen_main_interaction_trace(
    cases: &[TestCase],
    repetitions: usize,
    padded_rows: usize,
    log_n_rows: u32,
    state_elements: &IaddStateElements,
    range_check: RangeCheckMode,
    range_lookup_elements: Option<&RangeLookupElements>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let n_batches = main_interaction_batches(range_check);
    let mut logup_gen = LogupTraceGenerator::new(log_n_rows);
    for batch_idx in 0..n_batches {
        let mut col_gen = logup_gen.new_col();
        for_each_packed_row(cases, repetitions, padded_rows, |vec_row, lane_rows| {
            let frac = main_batched_fraction(
                batch_idx,
                lane_rows,
                state_elements,
                range_check,
                range_lookup_elements,
            );
            col_gen.write_frac(vec_row, frac.numerator, frac.denominator);
        });
        col_gen.finalize_col();
    }
    logup_gen.finalize_last()
}

fn main_interaction_batches(range_check: RangeCheckMode) -> usize {
    match range_check {
        RangeCheckMode::Off | RangeCheckMode::Bits => 1,
        RangeCheckMode::Lookup => LOOKUP_MAIN_BATCHES,
    }
}

fn main_batched_fraction(
    batch_idx: usize,
    lane_rows: &[Row; LANE_COUNT],
    state_elements: &IaddStateElements,
    range_check: RangeCheckMode,
    range_lookup_elements: Option<&RangeLookupElements>,
) -> PackedFraction {
    if batch_idx == 0 {
        return state_transition_fraction(lane_rows, state_elements);
    }
    let lookup_elements = range_lookup_elements.expect("lookup mode requires lookup elements");
    match range_check {
        RangeCheckMode::Lookup => lookup_range_fraction(batch_idx - 1, lane_rows, lookup_elements),
        RangeCheckMode::Off | RangeCheckMode::Bits => unreachable!("unexpected extra logup batch"),
    }
}

fn state_transition_fraction(
    lane_rows: &[Row; LANE_COUNT],
    state_elements: &IaddStateElements,
) -> PackedFraction {
    let input_state = packed_state(lane_rows, |row| {
        (row.sample_id, row.rep_idx, &row.acc, &row.addend)
    });
    let output_state = packed_state(lane_rows, |row| {
        (row.sample_id, row.rep_idx + 1, &row.next, &row.addend)
    });
    let input_denom: PackedSecureField = state_elements.combine(&input_state);
    let output_denom: PackedSecureField = state_elements.combine(&output_state);
    let multiplicity = packed_is_real(lane_rows);
    PackedFraction {
        numerator: multiplicity * (output_denom - input_denom),
        denominator: input_denom * output_denom,
    }
}

fn lookup_range_fraction(
    limb_idx: usize,
    lane_rows: &[Row; LANE_COUNT],
    lookup_elements: &RangeLookupElements,
) -> PackedFraction {
    let lo = packed_lookup_chunk(lane_rows, limb_idx, false);
    let hi = packed_lookup_chunk(lane_rows, limb_idx, true);
    let lo_denom: PackedSecureField = lookup_elements.seq18.combine(&[lo]);
    let multiplicity = packed_is_real(lane_rows);
    if limb_idx + 1 == LIMBS {
        let hi_denom: PackedSecureField = lookup_elements.seq6.combine(&[hi]);
        PackedFraction {
            numerator: multiplicity * (hi_denom + lo_denom),
            denominator: lo_denom * hi_denom,
        }
    } else {
        let hi_denom: PackedSecureField = lookup_elements.seq11.combine(&[hi]);
        PackedFraction {
            numerator: multiplicity * (hi_denom + lo_denom),
            denominator: lo_denom * hi_denom,
        }
    }
}

fn gen_seq_lookup_interaction_trace<const LOG_SIZE: u32, R>(
    counts: &[u32],
    elements: &R,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
)
where
    R: Relation<PackedBaseField, PackedSecureField> + Sync,
{
    let mut logup_gen = LogupTraceGenerator::new(LOG_SIZE);
    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (LOG_SIZE - LOG_N_LANES)) {
        let values = packed_seq_values::<LOG_SIZE>(vec_row);
        let packed_counts = PackedM31::from_array(std::array::from_fn(|lane| {
            BaseField::from_u32_unchecked(counts[(vec_row << LOG_N_LANES) + lane])
        }));
        let denominator: PackedSecureField = elements.combine(&[values]);
        col_gen.write_frac(
            vec_row,
            -PackedSecureField::from(packed_counts),
            denominator,
        );
    }
    col_gen.finalize_col();
    logup_gen.finalize_last()
}

fn for_each_packed_row(
    cases: &[TestCase],
    repetitions: usize,
    padded_rows: usize,
    mut f: impl FnMut(usize, &[Row; LANE_COUNT]),
) {
    debug_assert_eq!(padded_rows % LANE_COUNT, 0);
    let mut lane_rows = [Row::zero(); LANE_COUNT];
    let mut row_idx = 0;

    for (sample_idx, case) in cases.iter().enumerate() {
        let addend = biguint_to_limbs(&case.r1_in);
        let mut acc = biguint_to_limbs(&case.r0_in);
        for rep_idx in 0..repetitions {
            let (next, carry) = add_limbs(acc, addend);
            lane_rows[row_idx % LANE_COUNT] = Row {
                is_real: 1,
                sample_id: sample_idx as u32,
                rep_idx: rep_idx as u32,
                acc,
                addend,
                next,
                carry,
            };
            if row_idx % LANE_COUNT == LANE_COUNT - 1 {
                f(row_idx / LANE_COUNT, &lane_rows);
                lane_rows = [Row::zero(); LANE_COUNT];
            }
            row_idx += 1;
            acc = next;
        }
    }

    while row_idx < padded_rows {
        lane_rows[row_idx % LANE_COUNT] = Row::zero();
        if row_idx % LANE_COUNT == LANE_COUNT - 1 {
            f(row_idx / LANE_COUNT, &lane_rows);
            lane_rows = [Row::zero(); LANE_COUNT];
        }
        row_idx += 1;
    }
    debug_assert_eq!(row_idx, padded_rows);
}

fn public_boundary_sum(
    cases: &[TestCase],
    repetitions: usize,
    state_elements: &IaddStateElements,
) -> Result<SecureField> {
    let mut sum = SecureField::zero();
    for (sample_idx, case) in cases.iter().enumerate() {
        let initial_state = state_from_limbs(
            sample_idx as u32,
            0,
            &biguint_to_limbs(&case.r0_in),
            &biguint_to_limbs(&case.r1_in),
        );
        let final_acc = biguint_to_limbs(&low_register_from_state_hex(&case.y_hex)?);
        let final_state = state_from_limbs(
            sample_idx as u32,
            repetitions as u32,
            &final_acc,
            &biguint_to_limbs(&case.r1_in),
        );
        let initial_comb: SecureField = state_elements.combine(&initial_state);
        let final_comb: SecureField = state_elements.combine(&final_state);
        sum += initial_comb.inverse() - final_comb.inverse();
    }
    Ok(sum)
}

fn lookup_public_sum(counts: &LookupCounts, elements: &RangeLookupElements) -> SecureField {
    seq_lookup_public_sum::<LOOKUP_SEQ18_LOG_SIZE, _>(&counts.seq18, &elements.seq18)
        + seq_lookup_public_sum::<LOOKUP_SEQ11_LOG_SIZE, _>(&counts.seq11, &elements.seq11)
        + seq_lookup_public_sum::<LOOKUP_SEQ6_LOG_SIZE, _>(&counts.seq6, &elements.seq6)
}

fn seq_lookup_public_sum<const LOG_SIZE: u32, R>(counts: &[u32], elements: &R) -> SecureField
where
    R: Relation<BaseField, SecureField>,
{
    let mut sum = SecureField::zero();
    for (value, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let denom: SecureField = elements.combine(&[BaseField::from_u32_unchecked(value as u32)]);
        sum += SecureField::from(BaseField::from_u32_unchecked(count)) * denom.inverse();
    }
    sum
}

fn packed_state(
    rows: &[Row; LANE_COUNT],
    state: impl Fn(&Row) -> (u32, u32, &[u32; LIMBS], &[u32; LIMBS]),
) -> [PackedM31; IADD_STATE_WIDTH] {
    std::array::from_fn(|i| {
        PackedM31::from_array(std::array::from_fn(|lane| {
            let (sample_id, rep_idx, lhs, rhs) = state(&rows[lane]);
            let value = if i == 0 {
                sample_id
            } else if i == 1 {
                rep_idx
            } else if i < 2 + LIMBS {
                lhs[i - 2]
            } else {
                rhs[i - 2 - LIMBS]
            };
            BaseField::from_u32_unchecked(value)
        }))
    })
}

fn packed_is_real(rows: &[Row; LANE_COUNT]) -> PackedSecureField {
    PackedSecureField::from(PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(rows[lane].is_real)
    })))
}

fn packed_lookup_chunk(rows: &[Row; LANE_COUNT], limb_idx: usize, high: bool) -> PackedBaseField {
    PackedM31::from_array(std::array::from_fn(|lane| {
        let (lo, hi) = split_lookup_chunks(limb_idx, rows[lane].next[limb_idx]);
        BaseField::from_u32_unchecked(if high { hi } else { lo })
    }))
}

fn packed_seq_values<const LOG_SIZE: u32>(vec_row: usize) -> PackedBaseField {
    PackedM31::from_array(std::array::from_fn(|lane| {
        BaseField::from_u32_unchecked(((vec_row << LOG_N_LANES) + lane) as u32)
    }))
}

fn state_from_limbs(
    sample_id: u32,
    rep_idx: u32,
    acc: &[u32; LIMBS],
    addend: &[u32; LIMBS],
) -> [BaseField; IADD_STATE_WIDTH] {
    std::array::from_fn(|i| {
        let value = if i == 0 {
            sample_id
        } else if i == 1 {
            rep_idx
        } else if i < 2 + LIMBS {
            acc[i - 2]
        } else {
            addend[i - 2 - LIMBS]
        };
        BaseField::from_u32_unchecked(value)
    })
}

fn limb_bits(i: usize) -> usize {
    if i + 1 == LIMBS {
        TOP_LIMB_BITS
    } else {
        LIMB_BITS
    }
}

fn limb_base(i: usize) -> u32 {
    1u32 << limb_bits(i)
}

fn range_bit_columns() -> usize {
    3 * (0..LIMBS).map(limb_bits).sum::<usize>()
}

fn trace_columns(range_check: RangeCheckMode) -> usize {
    COMPACT_TRACE_COLUMNS
        + match range_check {
            RangeCheckMode::Off => 0,
            RangeCheckMode::Bits => range_bit_columns(),
            RangeCheckMode::Lookup => LOOKUP_TRACE_COLUMNS - COMPACT_TRACE_COLUMNS,
        }
}
