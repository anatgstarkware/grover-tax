//! gate_air AIR assembly: the shared `LookupElements` + `GateRel` relation, the `Components` bundle
//! (type aliases + `build_components` + component/prover refs + trace_log_sizes), and the AIR
//! constants (widths, encoding sizes). The preprocessed column layout + generators live in
//! `crate::preprocessed`. The per-component `FrameworkEval`s
//! and their relation ids live in `crate::air::components::{gate,program,qubitmem,range_check}`; this file
//! is the verifier-facing assembly they are wired into.
//!
//! `GateRel` (the single drawn LogUp relation, `relation!(GateRel, 6)`) lives here (the AIR is its
//! near-sole user); the prover pipeline stays in `prover.rs`, trace/witness generation in `tracegen.rs`.

pub(crate) mod components;

use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
#[cfg(not(feature = "cuda"))]
use stwo::prover::backend::simd::SimdBackend as ProverBackend;
#[cfg(feature = "cuda")]
use stwo::prover::backend::CudaBackend as ProverBackend;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::air::components::gate::GateEval;
use crate::air::components::program::ProgramEval;
use crate::air::components::qubitmem::QubitMemEval;
use crate::air::components::range_check::RangeCheckEval;
use crate::preprocessed::{preprocessed_column_ids, N_PREPROCESSED_COLS};

// The single drawn LogUp relation (shared by qubitmem / rc / program via a prepended tag; see
// `LookupElements`). The `relation!` macro emits a `pub struct`; inside this crate-local `air` module
// that is effectively crate-visible, referenced by the prover (`prover.rs`) and tracegen as
// `crate::air::GateRel`.
stwo_constraint_framework::relation!(GateRel, 6);

// Encoding constants

pub(crate) const N_QUBITS: usize = 512;
pub(crate) const LIMB_BITS: usize = 16;
pub(crate) const N_LIMBS: usize = N_QUBITS / LIMB_BITS; // 32
pub(crate) const STATE_BYTES: usize = N_QUBITS / 8; // 64

/// Production log-size `R` of the ts-ordering range-check (rc) supply table: a single block
/// enumerating `[0, 2^RC_LOG)` with `val[i] = i`, so one lookup per access checks
/// `d = pc - prev_ts ∈ [0, 2^RC_LOG)`. FIXED at 2^25 (sized for the k≈8000 target). The PUBLIC,
/// trusted `R` (never read from a proof), threaded into the base prover + `GateAirStatement`; tests
/// pass their own small `R`. Sound while every honest `d` fits: `d_max = k*n_gates - 1 < 2^RC_LOG`
/// (k ≲ 8000; base prove-entry `debug_assert` guards it). `LOG_N_LANES <= rc_log <= log_n_rows` and
/// `2^RC_LOG < p`, so no SIMD underflow, field wrap, or raised FRI floor.
pub(crate) const RC_LOG: u32 = 25;

// Interaction-trace proof-of-work bits (canonical transcript; matches the in-circuit verifier's
// ProofConfig). Tiny grind (~2^8), present so the in-circuit verifier can replay the transcript.
pub(crate) const INTERACTION_POW_BITS: u32 = 8;

pub(crate) const NO_CTRL: u16 = 0xFFFF;

// Timestamp/range-check soundness (authoritative spot). `ts = pc + 1`, `pc` the PREPROCESSED,
// verifier-pinned per-shot program counter — so ts is a fixed affine function of pc the prover cannot
// reorder, and ts is inlined (not a witness column). The `+1` keeps the smallest real ts = 1 > 0 =
// the init boundary node's ts (plain `pc` would collide pc=0's accesses with init). Within a gate step
// the (<=3) accesses hit distinct addresses, so sharing ts = pc+1 never collides two on one chain;
// same-address accesses are in different steps (distinct pc), so their ts strictly increases.
//
// The diff `d = pc - prev_ts` is a SINGLE 25-bit column, proven in [0, 2^RC_LOG) by one LogUp lookup
// into the EXACT-range rc table (no slack). Honest `d <= k*n_gates - 1 < 2^RC_LOG` (completeness); the
// bound < p (RC_LOG <= TS_RC_BITS = 25) means the field subtraction cannot wrap, so a cyclic stale-read
// chain is impossible. pc-pinned ts (program order) + `prev_ts < ts` on every access ⇒ forward DAG.
pub(crate) const TS_RC_BITS: usize = 25;

pub(crate) const M31_MODULUS_U32: u32 = (1 << 31) - 1;
pub(crate) const LANE_COUNT: usize = 1 << LOG_N_LANES;

// ONE shared LogUp relation (single drawn (z,α)); logical relations are distinguished by a distinct id
// TAG prepended as the first tuple element, matching the in-circuit verifier's single-relation model
// one acc.interaction_elements, relation id as a constant in the tuple). Width =
// widest payload (program = slot,opcode,target,ctrl_a,ctrl_b = 5) + 1 tag = 6.
#[allow(dead_code)]
const GATE_REL_WIDTH: usize = 6;

// Relation id tags live in the per-component files (`components::{qubitmem,range_check,program}`):
// TAG_QUBITMEM = per-qubit chain-lookup relation; TAG_RC = ts-ordering range-check; TAG_PROGRAM /
// TAG_PROGRAM_PUB = program-consistency (H_P binding). Each supply component owns its own id.

// x/y binding: the boundary's final `y` is re-keyed to a FIXED public ts `TS_FINAL` so it surfaces as
// an unconsumed public LogUp term (the leaf supplies the matching term over its guessed x/y, forcing
// guessed == committed). TS_FINAL must exceed every real ts (0..=k*n_gates) and be a valid M31, so the
// public tuples never alias an interior chain node: 2^30 < p and >> any real ts.
pub(crate) const TS_FINAL: u32 = 1 << 30;

// Circuit parser (GTV1) opcodes.
pub(crate) const OP_NOP: u8 = 0;
pub(crate) const OP_NOT: u8 = 1;
pub(crate) const OP_CNOT: u8 = 2;
pub(crate) const OP_TOFFOLI: u8 = 3;

// Column layout. Per row, one access block = ACCESS_COLS core + 1 rc diff col:
//   is_nop,is_not,is_cnot,is_toffoli                                 (4)
//   target access: addr,prev_ts,v_before, d                         (ACCESS_BLOCK)
//   ctrl_a access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ctrl_b access: addr,prev_ts,v, d                                 (ACCESS_BLOCK)
//   ab, fire, delta                                                  (3)
// `enabler`/`shot_id`/`pc` are shard-invariant positional values in the preprocessed tree0. `ts =
// pc + 1` and the target `v_after = v_before + delta` are NOT columns — inlined (see TS_RC_BITS above).
pub(crate) const ACCESS_COLS: usize = 3; // addr, prev_ts, v (core access cols; ts inlined = pc+1)
pub(crate) const ACCESS_BLOCK: usize = ACCESS_COLS + 1; // core cols + the single rc diff col `d`
pub(crate) const TRACE_COLUMNS: usize = 4 + ACCESS_BLOCK + ACCESS_BLOCK + ACCESS_BLOCK + 3; // 4 + 3*4 + 3 = 19

// rc supply table: a single 2^R-row block enumerating exactly [0, 2^R) with `val[i] = i` (R = rc_log;
// no pos selector, no split). Main looks up (TAG_RC, d) per active access; the table supplies
// -multiplicity / (TAG_RC, value), pinning d < 2^R with no slack (see RC_LOG / TS_RC_BITS above).

// Boundary table: per (shot, addr) emits on TAG_QUBITMEM the INTERNAL final Use[+1](shot,addr,ts_last,y)
// (cancels main's last chain Yield) and the PUBLIC final Yield[-1](shot,addr,TS_FINAL,y) (re-keys y to
// the fixed public ts). `shot`/`addr` preprocessed; `x`/`y`/`ts_last` witness (`x` booleanity-only —
// main carries x publicly via its ts=0 init Use). Nets to the public term B = Σ(+[0,x] − [TS_FINAL,y]),
// which the leaf's public_logup_sum matches over guessed x/y.

/// The logical relations share the SAME drawn `(z,α)` (clones of one `GateRel`); the tag prepended at
/// each combine is what keeps them separate.
#[derive(Clone)]
pub(crate) struct LookupElements {
    pub(crate) qubitmem: GateRel,
    pub(crate) rc: GateRel,
    pub(crate) program: GateRel,
}

impl LookupElements {
    pub(crate) fn draw(channel: &mut impl Channel) -> Self {
        let rel = GateRel::draw(channel);
        Self {
            qubitmem: rel.clone(),
            rc: rel.clone(),
            program: rel,
        }
    }

    /// Fixed challenges for byte-identity tests. Used by the K4 GPU interaction validation.
    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn dummy() -> Self {
        let rel = GateRel::dummy();
        Self {
            qubitmem: rel.clone(),
            rc: rel.clone(),
            program: rel,
        }
    }
}

// Components bundle

type GateComponent = FrameworkComponent<GateEval>;
type ProgramComponent = FrameworkComponent<ProgramEval>;
type QubitMemComponent = FrameworkComponent<QubitMemEval>;
type RangeCheckComponent = FrameworkComponent<RangeCheckEval>;

pub(crate) struct Components {
    pub(crate) main: GateComponent,
    pub(crate) program: ProgramComponent,
    pub(crate) qubitmem: QubitMemComponent,
    pub(crate) range_check: RangeCheckComponent,
}

impl Components {
    pub(crate) fn component_refs(&self) -> Vec<&dyn Component> {
        vec![
            &self.main as &dyn Component,
            &self.program as &dyn Component,
            &self.qubitmem as &dyn Component,
            &self.range_check as &dyn Component,
        ]
    }

    pub(crate) fn prover_refs(&self) -> Vec<&dyn stwo::prover::ComponentProver<ProverBackend>> {
        vec![
            &self.main as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.program as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.qubitmem as &dyn stwo::prover::ComponentProver<ProverBackend>,
            &self.range_check as &dyn stwo::prover::ComponentProver<ProverBackend>,
        ]
    }

    pub(crate) fn trace_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        // Use stwo's CANONICAL column sizes (not a plain component-concat). stwo's verifier
        // reindexes the PREPROCESSED tree GLOBALLY by each component's preprocessed_column_indices
        // (i.e. into preprocessed_column_ids() order), so the preprocessed sizes land in the
        // committed order — which the lifted Merkle commits sorted by size and the in-circuit
        // verifier (circuits_stark_verifier) does NOT re-sort. A naive concat would order the
        // preprocessed sizes by component instead, mismatching the committed tree ("Root mismatch").
        stwo::core::air::Components {
            components: self.component_refs(),
            n_preprocessed_columns: N_PREPROCESSED_COLS,
        }
        .column_log_sizes()
    }
}

// Free functions

/// Packed tag constant for prover-side `combine` tuples.
pub(crate) fn ptag(tag: u32) -> PackedM31 {
    PackedM31::broadcast(BaseField::from_u32_unchecked(tag))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_components(
    log_n_rows: u32,
    program_log_size: u32,
    boundary_log_size: u32,
    rc_log: u32,
    elements: &LookupElements,
    main_sum: SecureField,
    program_sum: SecureField,
    boundary_sum: SecureField,
    rc_sum: SecureField,
) -> Components {
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(
        &preprocessed_column_ids(log_n_rows, program_log_size, boundary_log_size, rc_log),
    );
    let main = GateComponent::new(
        &mut allocator,
        GateEval {
            log_n_rows,
            elements: elements.clone(),
        },
        main_sum,
    );
    let program = ProgramComponent::new(
        &mut allocator,
        ProgramEval {
            log_size: program_log_size,
            elements: elements.program.clone(),
        },
        program_sum,
    );
    let qubitmem = QubitMemComponent::new(
        &mut allocator,
        QubitMemEval {
            log_size: boundary_log_size,
            elements: elements.qubitmem.clone(),
        },
        boundary_sum,
    );
    let range_check = RangeCheckComponent::new(
        &mut allocator,
        RangeCheckEval {
            log_size: rc_log,
            elements: elements.rc.clone(),
        },
        rc_sum,
    );
    Components {
        main,
        program,
        qubitmem,
        range_check,
    }
}
