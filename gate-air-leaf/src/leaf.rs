//! Turn the gate_air verification circuit into a foldable leaf and fold N of them. A leaf verifies
//! one gate_air stwo proof (`circuits_stark_verifier::verify` + `GateAirStatement`) and sets its
//! reserved outputs to `H_i`, making it a circuit-prover proof the multiverifier folds.
//!
//! Output: `H_i = blake( H_P ‖ x_limbs ‖ y_limbs )`, where `H_P = blake( program_table ‖ nonce )` is
//! the hiding secret-circuit commitment. Program and x/y are guessed witness AND bound to the base
//! proof via `public_logup_sum` (TAG_PROGRAM_PUB + boundary public terms), so `H_i` commits to a
//! genuine `x→y` execution of the committed hidden circuit; the verifier recomputes every `H_i` from
//! the one published `H_P`, enforcing same-program across shards.

use circuits::blake::{blake2s, HashValue};
use circuits::context::{Context, FinalizedContext, Var};
use circuits::ivalue::{IValue, NoValue};
use circuits::ops::Guess;
use circuits_stark_verifier::proof::{empty_proof, Proof, ProofConfig};
use circuits_stark_verifier::verify::verify;

use circuit_common::finalize::{compute_padded_sizes, ComponentSizes};
use circuit_common::N_RESERVED;

use recursive_aggregate::precomputes::RecursionPrecompute;
use recursive_aggregate::AggregateConfig;
use stwo::core::fields::qm31::QM31;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;

use crate::recursion_consts::OperatingPoint;
use crate::topology::{TopologyConfig, FOLD_ARITY, RECURSION_LOG_BLOWUP};

use crate::circuit_statement::GateAirStatement;
use crate::N_LIMBS;

/// Public parameters of the gate_air proof the leaf verifies (everything `GateAirStatement::new`
/// needs). Identical for the NoValue shape pass and the real QM31 assignment.
#[derive(Clone)]
pub struct GateAirLeafParams {
    pub main_log_size: u32,
    pub program_log_size: u32,
    pub boundary_log_size: u32,
    /// The rc supply-table log-size `R` this base was proved with — a TRUSTED construction value the
    /// leaf statement must reuse (production: `RC_LOG`; tests: the test's chosen value). The base
    /// prover's `R` and this MUST be equal (see `GateAirStatement::new`); NEVER read from the proof.
    pub rc_log: u32,
    pub preprocessed_root: HashValue<QM31>,
    pub boundary: Vec<([u32; N_LIMBS], [u32; N_LIMBS])>,
    pub total_pc: u32,
    /// H_P program commitment (OPEN #3, Fork A). The committed program table this leaf's base proof
    /// ran: one entry per padded slot `(slot, opcode_scalar, target, ctrl_a, ctrl_b, multiplicity)`.
    /// The leaf GUESSES these, BINDS them to the base's committed program via the TAG_PROGRAM_PUB
    /// public term (`public_logup_sum`), and hashes them into `H_P`.
    pub program: ProgramRows,
    /// One shared hiding nonce (2 M31 words), IDENTICAL across all leaves of a run, folded into
    /// `H_P = blake(program ‖ nonce)`. Binding-inert (only blinds the program preimage); its
    /// consistency across leaves is required so every leaf yields the SAME H_P for the same program.
    pub nonce: [u32; 2],
}

/// Program-table rows the leaf hashes into H_P and binds to the base. One entry per padded slot,
/// mirroring `main.rs::ProgramTable`.
#[derive(Clone)]
pub struct ProgramRows {
    pub slot: Vec<u32>,
    pub opcode_scalar: Vec<u32>,
    pub target: Vec<u32>,
    pub ctrl_a: Vec<u32>,
    pub ctrl_b: Vec<u32>,
    /// per-slot multiplicity = samples*k (real) / 0 (padding). PINNED (not free) in the leaf.
    pub multiplicity: Vec<u32>,
}

/// PCS config for the leaf circuit's own (outer) proof. Mirrors stwo-circuits' `get_pcs_config`.
pub fn leaf_pcs_config(trace_log_size: u32, log_blowup_factor: u32) -> PcsConfig {
    let (pow_bits, n_queries) = match log_blowup_factor {
        1 => (26, 70),
        2 => (26, 35),
        3 => (27, 23),
        _ => panic!("unsupported log blowup factor"),
    };
    PcsConfig {
        pow_bits,
        fri_config: FriConfig {
            log_blowup_factor,
            log_last_layer_degree_bound: 0,
            n_queries,
            fold_step: 4,
        },
        lifting_log_size: Some(trace_log_size + log_blowup_factor),
    }
}

/// Verifies ONE gate_air base proof in-circuit and returns the eight `H_i` digest words the leaf
/// sets as outputs. The whole security-relevant binding lives here.
fn emit_one_base<Value: IValue>(
    context: &mut Context<Value>,
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> Vec<Var> {
    let statement = GateAirStatement::<Value>::new(
        context,
        params.main_log_size,
        params.program_log_size,
        params.boundary_log_size,
        params.rc_log,
        params.preprocessed_root.clone(),
        params.boundary.clone(),
        params.total_pc,
        params.program.clone(),
        params.nonce,
    );
    let proof_vars = proof.guess(context);
    verify(context, &proof_vars, cfg, &statement);

    // H_i = blake2s( H_P ‖ x ‖ y ) as eight words (#1425 full unreduced digest). H_P (eight words
    // from `compute_h_p`) leads the preimage, followed by the guessed x/y limb Vars. `public_logup_sum`
    // binds both the program (feeding H_P) and x/y to the base proof, so H_i commits to the executed
    // secret circuit. Everything stays witness (not constant) so `leaf_preprocessed_root` is shard-invariant.
    let h_p = statement.compute_h_p(context);
    let mut preimage: Vec<_> = h_p.iter().map(|w| *w.get()).collect();
    for (x, y) in statement.boundary_vars() {
        preimage.extend(x.iter().chain(y.iter()).copied());
    }
    let output_hash: HashValue<_> = blake2s(context, &preimage, 16 * preimage.len());
    let h_i_vars: Vec<Var> = output_hash.iter().map(|w| *w.get()).collect();
    h_i_vars
}

/// Builds the gate_air single-base LEAF circuit: verify one base proof (via [`emit_one_base`]) and
/// set the reserved outputs to its `H_i`. Generic over `Value` (NoValue shape / QM31 assignment).
pub fn build_gate_air_leaf_circuit<Value: IValue>(
    proof: Proof<Value>,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> FinalizedContext<Value> {
    let mut context = Context::new(N_RESERVED);
    let h_i_vars = emit_one_base(&mut context, proof, cfg, params);
    context.set_outputs(&h_i_vars);
    context.finalize(false)
}

/// Assembles the gate_air `AggregateConfig` for pinned operating point `op` entirely from the pinned
/// verifier consts (no derivation), via the same `assemble_aggregate_config` the fresh cascade uses,
/// so a pinned config is byte-identical to a derived one. The PRODUCTION config source. The pinned
/// points fix `fold_arity == FOLD_ARITY` and the default node/leaf blowup (asserted); `cfg`/`params`
/// only compute the leaf's own natural padding target.
pub fn pinned_aggregate_config(
    op: OperatingPoint,
    topo: &TopologyConfig,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
) -> AggregateConfig {
    assert_eq!(
        topo.fold_arity, FOLD_ARITY,
        "pinned operating points fix fold_arity = {FOLD_ARITY}"
    );
    assert_eq!(
        (topo.recursion_log_blowup, topo.leaf_log_blowup),
        (RECURSION_LOG_BLOWUP, RECURSION_LOG_BLOWUP),
        "pinned operating points fix the default node/leaf blowup"
    );
    op.pinned().to_aggregate_config(
        leaf_target_sizes(cfg, params),
        topo.leaf_log_blowup,
        topo.recursion_log_blowup,
        FOLD_ARITY,
    )
}

/// Builds the flat leaf/level1/fold [`RecursionPrecompute`]: builds the AIR-specific leaf circuit
/// (the only gate_air-specific ingredient) and hands it to the generic
/// [`recursive_aggregate::precomputes::build_recursion_precompute`], which rebuilds every held tree
/// from `config` and asserts each committed root equals `config`'s pinned root.
///
/// `build_all_arities`: production passes `false` → build shapes only for the arities this point's
/// fold uses ([`op.n()`] leaves); tests pass `true`. See the proving-utils fn for details.
pub fn build_recursion_precompute(
    config: &AggregateConfig,
    op: OperatingPoint,
    cfg: &ProofConfig,
    params: &GateAirLeafParams,
    build_all_arities: bool,
) -> RecursionPrecompute {
    recursive_aggregate::precomputes::build_recursion_precompute(
        build_gate_air_leaf_circuit::<NoValue>(empty_proof(cfg), cfg, params),
        config,
        op.n(),
        build_all_arities,
    )
}

/// The leaf's OWN natural padding target, decoupled from the node target so `t_leaf` is pinned
/// independent of `fold_arity`. The pinned leaf root is the soundness anchor; this padding only sets
/// the prover's own leaf trace shape (a drift fails the leaf root assert).
fn leaf_target_sizes(cfg: &ProofConfig, params: &GateAirLeafParams) -> ComponentSizes {
    compute_padded_sizes(&build_gate_air_leaf_circuit::<NoValue>(
        empty_proof(cfg),
        cfg,
        params,
    ))
}
