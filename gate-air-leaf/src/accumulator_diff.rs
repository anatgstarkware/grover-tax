//! Accumulator-diff scaffold for the future gate_air GPU constraint kernel
//! (P5_GPU_CONSTRAINT_SCOPE.md, Deliverable 2 §2.2 step d + Deliverable 1).
//!
//! This is the DECISIVE, localized validator for the upcoming device-resident CUDA constraint
//! kernel: it isolates the composition-polynomial (constraint-quotient) accumulation from all
//! commit / FRI machinery. The kernel is trusted only once this diff is ZERO, coordinate-by-
//! coordinate, on multiple graph families (c=3 demo AND blake / 44k fibonacci / cairo, per the
//! general-solutions rule).
//!
//! STATUS: scaffold only. The GPU constraint kernel does NOT exist yet (the CudaBackend currently
//! always routes constraint eval through the audited host delegate — see
//! `stwo-gpu-port/crates/constraint-framework/src/prover/cuda_component_prover.rs`). This module
//! therefore provides:
//!
//! 1. The comparison PRIMITIVE that operates on the two host-side results (works today).
//! 2. A precise, line-cited plan for WHERE the CPU oracle column and the GPU kernel column are
//!    produced and HOW they are brought to host for the diff.
//!
//! It does NOT modify any CPU constraint-eval / prover / verifier math (HARD CONSTRAINT): the diff
//! is a pure read of two already-computed `SecureColumnByCoords` results.
//!
//! ===========================================================================================
//! WHERE THE TWO SIDES COME FROM (the plug-in point)
//! ===========================================================================================
//!
//! The single function that both sides must agree on is
//! `ComponentProver::<B>::evaluate_constraint_quotients_on_domain`, which writes the
//! `DomainEvaluationAccumulator<B>`'s 4 QM31-coordinate columns for one committed trace.
//!
//! CPU ORACLE (exists, audited, READ-ONLY reference):
//!   `accumulate_pointwise_cpu`
//!     (constraint-framework/src/prover/component_prover.rs:264-292)
//!   driven from the CudaBackend delegate
//!     (constraint-framework/src/prover/cuda_component_prover.rs:116-144)
//!   which already:
//!     - calls `get_constraint_quotient_inputs` (component_prover.rs:76-114) to obtain the
//!       eval-domain-extended trace columns + `denom_inv`,
//!     - splits + `.reverse()`s `random_coeff_powers` off the accumulator (cuda_component_prover.rs:125-127),
//!     - runs `accumulate_pointwise_cpu` seeded with the device accumulator's current contents
//!       (`accum.col.to_cpu()`, line 143), producing a `SecureColumnByCoords<CpuBackend>`.
//!   => The CPU oracle column is exactly `host_result` at cuda_component_prover.rs:136.
//!
//! GPU KERNEL (future, Deliverable 1):
//!   The new `evaluate_constraint_quotients_on_domain` GPU branch (gated on `!cpu_fallback_forced()`
//!   AND component == gate_air MAIN) will call the FFI
//!     `bindings::evaluate_constraint_quotients_on_domain`
//!     (stwo-gpu-port/.../stwo_cuda/bindings.rs:469-490)
//!   writing the 4 device accumulator coord buffers (`CudaSecureColumn::device_ptr`). To diff, copy
//!   those 4 device columns to host via `Column::to_cpu()` (the same primitive
//!   `accum.col.to_cpu()` already uses), yielding a second `SecureColumnByCoords<CpuBackend>`.
//!
//! THE DIFF (this module): `diff_secure_columns` compares the two host coordinate arrays
//! QM31-for-QM31 and reports the FIRST diverging (row, coord). For a clean compare, run BOTH sides
//! seeded with a ZEROED accumulator (`should_accumulate=false`, or `=true` with a zeroed seed) and
//! with FIXED challenges (`LookupElements::dummy()`, main.rs:142-154) so both see identical (z, α).
//!
//! ===========================================================================================
//! REQUIRED ADDITIVE SEAM (read-only; not yet added — documented for the implementer)
//! ===========================================================================================
//!
//! `accumulate_pointwise_cpu` and `get_constraint_quotient_inputs` are `pub(crate)` in
//! constraint-framework, so a clean tap CANNOT be reached from gate-air-leaf today. The minimal,
//! NON-math, READ-ONLY seam (to be added INSIDE cuda_component_prover.rs, where the data already
//! lives) is a function that, for a given committed `Trace<CudaBackend>` + component, returns BOTH:
//!   (a) the CPU-oracle accumulator column (re-running the existing delegate with a zeroed seed), and
//!   (b) the GPU-kernel accumulator column (the new FFI path, zeroed seed),
//! both already `.to_cpu()`'d, then calls `diff_secure_columns` here. This adds a new gated branch
//! only; it does not touch the existing `accumulate_pointwise_cpu` body or any verifier math.
//! Until that seam + the kernel exist, this module's `diff_secure_columns` stands ready and the
//! plan above is the contract.

#![allow(dead_code)]

use stwo::core::fields::qm31::SecureField;

/// Result of a coordinate-by-coordinate accumulator comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccumulatorDiff {
    /// Both accumulator columns are byte-identical (QM31-for-QM31). The kernel matches the oracle.
    Equal { rows: usize },
    /// Length mismatch — the two columns have different row counts.
    LenMismatch { cpu_rows: usize, gpu_rows: usize },
    /// First diverging row, with the oracle vs kernel values at that row.
    FirstDiff {
        row: usize,
        cpu: SecureField,
        gpu: SecureField,
    },
}

/// Compare two host-side composition-polynomial accumulator columns, each given as its flat
/// sequence of `SecureField` (QM31) row values (i.e. `SecureColumnByCoords::to_vec()` on both
/// sides). Returns the FIRST diverging row, mirroring the PASS/FAIL localization of
/// `gpu_tracegen.rs`'s k1/k4 byte-identity checks.
///
/// `cpu` is the golden oracle (`accumulate_pointwise_cpu` result), `gpu` is the kernel output.
pub fn diff_secure_columns(cpu: &[SecureField], gpu: &[SecureField]) -> AccumulatorDiff {
    if cpu.len() != gpu.len() {
        return AccumulatorDiff::LenMismatch {
            cpu_rows: cpu.len(),
            gpu_rows: gpu.len(),
        };
    }
    for (row, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        if c != g {
            return AccumulatorDiff::FirstDiff {
                row,
                cpu: c,
                gpu: g,
            };
        }
    }
    AccumulatorDiff::Equal { rows: cpu.len() }
}

/// Convenience: assert equality and print a PASS line, or print the first diff and return false.
/// (Caller decides whether to panic; the harness prints rather than aborting mid-pipeline.)
pub fn report_accumulator_diff(label: &str, cpu: &[SecureField], gpu: &[SecureField]) -> bool {
    match diff_secure_columns(cpu, gpu) {
        AccumulatorDiff::Equal { rows } => {
            println!("gate-air: accumulator_diff[{label}]=PASS rows={rows}");
            true
        }
        AccumulatorDiff::LenMismatch { cpu_rows, gpu_rows } => {
            println!("gate-air: accumulator_diff[{label}]=FAIL len cpu={cpu_rows} gpu={gpu_rows}");
            false
        }
        AccumulatorDiff::FirstDiff { row, cpu, gpu } => {
            println!(
                "gate-air: accumulator_diff[{label}]=FAIL first_diff row={row} cpu={cpu:?} gpu={gpu:?}"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn equal_columns() {
        let a = vec![SecureField::one(), SecureField::zero(), SecureField::one()];
        assert_eq!(
            diff_secure_columns(&a, &a),
            AccumulatorDiff::Equal { rows: 3 }
        );
    }

    #[test]
    fn first_diff_localized() {
        let a = vec![SecureField::one(), SecureField::one()];
        let b = vec![SecureField::one(), SecureField::zero()];
        match diff_secure_columns(&a, &b) {
            AccumulatorDiff::FirstDiff { row, .. } => assert_eq!(row, 1),
            other => panic!("expected FirstDiff, got {other:?}"),
        }
    }

    #[test]
    fn len_mismatch() {
        let a = vec![SecureField::one()];
        let b = vec![SecureField::one(), SecureField::one()];
        assert!(matches!(
            diff_secure_columns(&a, &b),
            AccumulatorDiff::LenMismatch { .. }
        ));
    }
}
