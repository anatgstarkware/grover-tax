"""Error taxonomy for `grover_tax`.

The error model is contractually defined in `docs/spec/04-error-model.md`:
five disjoint categories (`BUILD`, `FIXTURE`, `PROVER`, `MEASUREMENT`,
`REPORT`), each with its own exit code and a stable set of subcodes.

Every harness-emitted error carries a subcode of the form
``<CATEGORY>.<SPECIFIC>`` (e.g. ``FIXTURE.WORKLOAD_NOT_PINNED``). Subcodes are
stable identifiers; the human message after the colon is free-form and may
evolve. The full line format on stderr is::

    <subcode>: <message> | run_id=<id> prover=<sp1|stwo> path=<file>

This module provides the typed exception hierarchy. The string form of the
exception is always ``<subcode>: <message>`` — context fields are appended by
the harness (`bin/run_*.sh`, `scripts/run_all.sh`) and are not the
responsibility of the exception itself.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "BUILD_EXIT_CODE",
    "FIXTURE_EXIT_CODE",
    "MEASUREMENT_SERIES_EXIT_CODE",
    "MEASUREMENT_WRAPPER_EXIT_CODE",
    "PROVER_EXIT_CODE",
    "REPORT_EXIT_CODE",
    "BuildError",
    "BuildSubcode",
    "FixtureError",
    "FixtureSubcode",
    "GroverTaxError",
    "MeasurementError",
    "MeasurementSubcode",
    "ProverError",
    "ProverSubcode",
    "ReportError",
    "ReportSubcode",
]

# Exit codes per `docs/spec/04-error-model.md`. Exit code `0` is reserved for
# success; any code outside the set below indicates a defect.
BUILD_EXIT_CODE = 3
FIXTURE_EXIT_CODE = 4
PROVER_EXIT_CODE = 1
MEASUREMENT_WRAPPER_EXIT_CODE = 2
MEASUREMENT_SERIES_EXIT_CODE = 5
REPORT_EXIT_CODE = 6


class BuildSubcode(str, Enum):
    """Subcodes for `BUILD` (exit 3) errors."""

    RUSTC_MISMATCH = "BUILD.RUSTC_MISMATCH"
    SP1_PATCH_FAIL = "BUILD.SP1_PATCH_FAIL"
    STWO_SHA_DRIFT = "BUILD.STWO_SHA_DRIFT"
    CARGO_FAIL = "BUILD.CARGO_FAIL"
    UV_SYNC_FAIL = "BUILD.UV_SYNC_FAIL"
    LICENSE_CHECK_FAIL = "BUILD.LICENSE_CHECK_FAIL"


class FixtureSubcode(str, Enum):
    """Subcodes for `FIXTURE` (exit 4) errors."""

    CROSS_VALIDATION_FAIL = "FIXTURE.CROSS_VALIDATION_FAIL"
    COMMITMENT_MISMATCH = "FIXTURE.COMMITMENT_MISMATCH"
    SCHEMA_INVALID = "FIXTURE.SCHEMA_INVALID"
    WORKLOAD_NOT_PINNED = "FIXTURE.WORKLOAD_NOT_PINNED"
    SEED_DRIFT = "FIXTURE.SEED_DRIFT"
    DRIFT = "FIXTURE.DRIFT"
    # KB-1 (#113): the .kmx → GTV1 transpiler rejects out-of-subset input.
    UNSUPPORTED_INSTRUCTION = "FIXTURE.UNSUPPORTED_INSTRUCTION"
    KMX_PARSE_ERROR = "FIXTURE.KMX_PARSE_ERROR"


class ProverSubcode(str, Enum):
    """Subcodes for `PROVER` (exit 1) errors."""

    WITNESS_REJECTED = "PROVER.WITNESS_REJECTED"
    VERIFIER_REJECTED = "PROVER.VERIFIER_REJECTED"
    OOM = "PROVER.OOM"
    TIMEOUT = "PROVER.TIMEOUT"
    STDOUT_GRAMMAR_VIOLATION = "PROVER.STDOUT_GRAMMAR_VIOLATION"


class MeasurementSubcode(str, Enum):
    """Subcodes for `MEASUREMENT` errors (exit 2 from wrappers, exit 5 from `measure.sh`)."""

    ENV_VAR_MISS = "MEASUREMENT.ENV_VAR_MISS"
    AFFINITY_MISS = "MEASUREMENT.AFFINITY_MISS"
    GPU_RESIDENT = "MEASUREMENT.GPU_RESIDENT"
    THERMAL_EXCEEDED = "MEASUREMENT.THERMAL_EXCEEDED"
    SWAP_ACTIVE = "MEASUREMENT.SWAP_ACTIVE"
    AC_POWER_MISS = "MEASUREMENT.AC_POWER_MISS"
    LOWPOWER_ENABLED = "MEASUREMENT.LOWPOWER_ENABLED"
    GOVERNOR_MISS = "MEASUREMENT.GOVERNOR_MISS"
    VERSIONS_DRIFT = "MEASUREMENT.VERSIONS_DRIFT"


class ReportSubcode(str, Enum):
    """Subcodes for `REPORT` (exit 6) errors."""

    INSUFFICIENT_SAMPLES = "REPORT.INSUFFICIENT_SAMPLES"
    STABILITY_BREACH = "REPORT.STABILITY_BREACH"
    MISSING_ARTIFACT = "REPORT.MISSING_ARTIFACT"
    SCHEMA_INVALID = "REPORT.SCHEMA_INVALID"


class GroverTaxError(Exception):
    """Base class for every typed error in the harness.

    Subclasses bind to a single error category and a single subcode enum;
    callers always pass a category-specific subcode, which produces a stable
    ``<SUBCODE>: <message>`` string form.
    """

    exit_code: int

    def __init__(self, subcode: str, message: str = "") -> None:
        self.subcode = subcode
        self.message = message
        super().__init__(self._render())

    def _render(self) -> str:
        return f"{self.subcode}: {self.message}" if self.message else self.subcode

    def __str__(self) -> str:
        return self._render()


class BuildError(GroverTaxError):
    """`BUILD` category — toolchain / submodule / cargo / uv failures (exit 3)."""

    exit_code = BUILD_EXIT_CODE

    def __init__(self, subcode: BuildSubcode | str, message: str = "") -> None:
        super().__init__(str(subcode.value if isinstance(subcode, BuildSubcode) else subcode), message)


class FixtureError(GroverTaxError):
    """`FIXTURE` category — generator, schema, or workload-pin failures (exit 4)."""

    exit_code = FIXTURE_EXIT_CODE

    def __init__(self, subcode: FixtureSubcode | str, message: str = "") -> None:
        super().__init__(
            str(subcode.value if isinstance(subcode, FixtureSubcode) else subcode), message
        )


class ProverError(GroverTaxError):
    """`PROVER` category — the prover binary itself failed (exit 1)."""

    exit_code = PROVER_EXIT_CODE

    def __init__(self, subcode: ProverSubcode | str, message: str = "") -> None:
        super().__init__(
            str(subcode.value if isinstance(subcode, ProverSubcode) else subcode), message
        )


class MeasurementError(GroverTaxError):
    """`MEASUREMENT` category — environmental or precondition failures.

    Default `exit_code` is `2` (per-run wrapper exit). Set
    ``exit_code = MEASUREMENT_SERIES_EXIT_CODE`` (`5`) for series-level failures
    raised from `measure.sh`.
    """

    exit_code = MEASUREMENT_WRAPPER_EXIT_CODE

    def __init__(
        self,
        subcode: MeasurementSubcode | str,
        message: str = "",
        *,
        series_level: bool = False,
    ) -> None:
        super().__init__(
            str(subcode.value if isinstance(subcode, MeasurementSubcode) else subcode), message
        )
        # Override the class-level exit code for series-level failures.
        # Per-instance shadowing — `BaseException.exit_code` stays at the
        # class default for callers that don't care.
        if series_level:
            self.exit_code = MEASUREMENT_SERIES_EXIT_CODE


class ReportError(GroverTaxError):
    """`REPORT` category — `analyze.py` or `plot.py` failures (exit 6)."""

    exit_code = REPORT_EXIT_CODE

    def __init__(self, subcode: ReportSubcode | str, message: str = "") -> None:
        super().__init__(
            str(subcode.value if isinstance(subcode, ReportSubcode) else subcode), message
        )
