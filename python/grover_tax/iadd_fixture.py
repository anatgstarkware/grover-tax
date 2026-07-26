"""`uv run gen-iadd-fixtures` — repeated-addition fixture generator (KB-4 #116, KB-15 #127).

Builds the `v0.3-iadd` fixtures for the Khattar/Google benchmark: an upstream
kickmix in-place adder (`iadd64.kmx`, `iadd256.kmx`, …) executed `K` times
(the scaling knob, KB-7/#119), transpiled to GTV1 via `grover_tax.kmx`, with
register-aware two-register test cases from `grover_tax.registers`.

**Storage model:** the fixture stores the GTV1 bytes of **one** adder
repetition; `repetitions` records `K` and the prover sides loop the gate list
`K` times per test case in-proof. This mirrors upstream, whose guest receives
the circuit once plus `num_repetitions` and commits `K` as a public value
(`example_zkp_fuzzer.rs`), and it keeps the fixture a constant ~20 KB at any
`K` — at the reference scale (`iadd256`, K≈8000, Tanuj's 8xA100 SP1 curve)
embedded concatenation would mean ~163 MB of circuit bytes.

Expected outputs come from independent integer math
(`r0' = r0 + K·r1 mod 2^width`, the relation the upstream fuzzer asserts); the
reference simulator replays as many full K-repetition cases as fit a fixed
gate-step budget (all of them at small scales), so cross-validation never makes
large-K generation infeasible. The supplied-case XOF is seeded by
`sha256(SEED ‖ sha256(kmx bytes) ‖ K ‖ N)`, binding the case stream to the
exact circuit and benchmark point — one step shy of KB-9's (#121) in-proof
Fiat-Shamir derivation.

The emitted file (`fixtures/v0.3-<circuit>-k<K>-n<N>.json`) is validated
in-process against `docs/spec/schemas/fixture-v0.3-iadd.schema.json`; a shape
violation is a generator defect (`FIXTURE.SCHEMA_INVALID`). `--check`
regenerates and compares byte-for-byte (modulo `generator_commit`), so CI can
prove the fixture is reproducible.

Commitment policy (`docs/spec/v0.3/COMMITMENT-POLICY.md`): the fixture commits
the **raw single-repetition `.kmx` SHA-256** (`kmx_source_sha256`,
cross-comparable with upstream) *and* the GTV1 SHA-256/Blake2s over the stored
single-repetition bytes (internal integrity).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from grover_tax import logging as gt_logging
from grover_tax.errors import FIXTURE_EXIT_CODE, FixtureError, FixtureSubcode
from grover_tax.kmx import KmxCircuit, kmx_source_sha256, opcode_histogram, transpile_file
from grover_tax.paths import fixtures_dir, repo_root
from grover_tax.registers import encode_registers, iadd_test_cases, min_state_bytes
from grover_tax.serialise import Opcode, serialise
from grover_tax.sim_reference import run
from grover_tax.validate_schemas import validate_artifact

__all__ = ["build_iadd_fixture", "main"]

# Byte-stable seed prefix for the supplied-case stream; mixed with the kmx
# hash and the (K, N) benchmark point before seeding the XOF. Changing it
# changes every v0.3-iadd fixture — by design.
SEED: bytes = b"grover-tax-v0.3-iadd-2026"

FIXTURE_VERSION = "v0.3-iadd"
CIRCUIT_SERIALISATION_FORMAT_VERSION = 1
SCHEMA_FILENAME = "fixture-v0.3-iadd.schema.json"
_BIT_STRIPE_WIDTH = 64  # verbatim from upstream batching (WORKLOAD.md)

# `tanujkhattar/zkp_ecc` commit that introduced `iadd256.kmx` and the
# generalized `run_proofs.sh` benchmark driver (branch `update_examples`).
# The vendored example_data circuits are byte-identical at this commit.
# Supersedes WORKLOAD.md's v0.1 pin for the iadd workload until KB-5 (#117)
# re-pins WORKLOAD.md itself.
UPSTREAM_PIN_COMMIT = "fc8dc785dee9aa1045e440ed42ba56942c458124"

# Reference-simulator replay budget for cross-validation, in gate steps
# (gates-per-repetition x K x cases). At iadd256/K=4 this covers ~2400 cases;
# at the reference point (K=8000) it still replays one full case. The replayed
# count is a pure function of the fixture inputs, so fixtures stay
# byte-reproducible.
_SIM_BUDGET_GATE_STEPS = 25_000_000

_ACC, _OFFSET = "r0", "r1"

_log = gt_logging.get_logger("grover_tax.iadd_fixture")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _example_data_dir() -> Path:
    """Directory of vendored upstream kickmix circuits."""
    return repo_root() / "third_party" / "sp1" / "docs" / "example_data"


def build_iadd_fixture(
    *,
    repetitions: int,
    n_samples: int,
    tier: str,
    circuit_source: str = "iadd64.kmx",
    pin_commit: str = UPSTREAM_PIN_COMMIT,
) -> dict[str, object]:
    """Assemble a `v0.3-iadd` fixture dict: one stored adder copy, K-loop semantics.

    Raises:
        FixtureError(FIXTURE.SCHEMA_INVALID): if the assembled fixture fails its
            own JSON Schema (a generator defect).
        FixtureError(FIXTURE.CROSS_VALIDATION_FAIL): if the reference simulator
            disagrees with the integer-math expectation on a replayed case.
        FixtureError: propagated from the transpiler if the source circuit is
            outside the classical subset.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    base = transpile_file(_example_data_dir() / circuit_source)
    register_width = _adder_register_width(base, circuit_source)
    num_bytes = min_state_bytes(base)
    layout = base.register_layout

    seed_bytes = hashlib.sha256(
        SEED
        + bytes.fromhex(kmx_source_sha256(base.source_bytes))
        + repetitions.to_bytes(8, "little")
        + n_samples.to_bytes(8, "little")
    ).digest()
    cases = iadd_test_cases(width=register_width, count=n_samples, seed=seed_bytes)

    modulus = 1 << register_width
    test_cases: list[dict[str, object]] = []
    for case in cases:
        acc_out = (case.x + repetitions * case.y) % modulus
        x_state = encode_registers({_ACC: case.x, _OFFSET: case.y}, layout, num_bytes)
        y_state = encode_registers({_ACC: acc_out, _OFFSET: case.y}, layout, num_bytes)
        test_cases.append(
            {
                "r0_in": case.x,
                "r1_in": case.y,
                "x_hex": x_state.hex(),
                "y_hex": y_state.hex(),
            }
        )

    _sim_cross_validate(base, test_cases, repetitions=repetitions)

    circuit_bytes = serialise(base.gates)
    hist = opcode_histogram(base.gates)
    # CCX is the only non-Clifford here; upstream scales its demanded bounds by
    # the repetition count (`run_proofs.sh`: SCALED_TOFFOLI = TOFFOLI * nrep).
    non_clifford = hist[Opcode.TOFFOLI.name] * repetitions
    instruction_count = len(base.gates) * repetitions

    fixture: dict[str, object] = {
        "version": FIXTURE_VERSION,
        "generator_commit": _git_head_sha(),
        "workload_pin_commit": pin_commit,
        "seed_hex": seed_bytes.hex(),
        "tier": tier,
        "circuit_source": circuit_source,
        "kmx_source_sha256": kmx_source_sha256(base.source_bytes),
        "repetitions": repetitions,
        "register_layout": layout,
        "register_width": register_width,
        "num_qubits": base.num_qubits,
        "n_samples": n_samples,
        "bit_stripe_width": _BIT_STRIPE_WIDTH,
        "circuit_serialisation_format_version": CIRCUIT_SERIALISATION_FORMAT_VERSION,
        "circuit_byte_serialisation_hex": circuit_bytes.hex(),
        "circuit_commitment_sha256_hex": hashlib.sha256(circuit_bytes).hexdigest(),
        "circuit_commitment_blake2s_hex": hashlib.blake2s(circuit_bytes).hexdigest(),
        "demanded_max_qubit_count": base.num_qubits,
        "demanded_max_non_clifford_count": non_clifford,
        "demanded_max_circuit_instructions": instruction_count,
        "demanded_num_samples": n_samples,
        "test_cases": test_cases,
    }

    errors = validate_artifact(document=fixture, schema_filename=SCHEMA_FILENAME)
    if errors:
        joined = "; ".join(e.message for e in errors[:5])
        raise FixtureError(
            FixtureSubcode.SCHEMA_INVALID,
            f"assembled v0.3-iadd fixture failed its schema: {joined}",
        )
    return fixture


def _adder_register_width(circuit: KmxCircuit, circuit_source: str) -> int:
    """The shared bit width of the adder's two registers (r0, r1)."""
    layout = circuit.register_layout
    if "r0" not in layout or "r1" not in layout:
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"{circuit_source}: expected two registers r0,r1, got {sorted(layout)}",
        )
    width = len(layout["r0"])
    if len(layout["r1"]) != width:
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"{circuit_source}: register widths differ ({width} vs {len(layout['r1'])})",
        )
    return width


def _sim_cross_validate(
    base: KmxCircuit,
    test_cases: list[dict[str, object]],
    *,
    repetitions: int,
) -> None:
    """Replay full K-repetition cases under the reference simulator (F-INV-4).

    Validates the first `n` cases where `n` is the largest count fitting the
    fixed gate-step budget — at least one case is always replayed, whatever
    the cost.
    """
    gates = list(base.gates)
    steps_per_case = len(gates) * repetitions
    n_validate = min(len(test_cases), max(1, _SIM_BUDGET_GATE_STEPS // steps_per_case))
    for i in range(n_validate):
        state = bytes.fromhex(str(test_cases[i]["x_hex"]))
        for _ in range(repetitions):
            state = run(gates, state)
        if state != bytes.fromhex(str(test_cases[i]["y_hex"])):
            raise FixtureError(
                FixtureSubcode.CROSS_VALIDATION_FAIL,
                f"test case {i}: simulated K={repetitions} output {state.hex()} "
                f"!= integer-math expectation {test_cases[i]['y_hex']}",
            )
    _log.info(
        "sim cross-validated %d/%d cases (%d gate steps each)",
        n_validate,
        len(test_cases),
        steps_per_case,
    )


# -- CLI + I/O ---------------------------------------------------------------


def _default_out_path(circuit_source: str, repetitions: int, n_samples: int) -> Path:
    stem = Path(circuit_source).stem
    return fixtures_dir() / f"v0.3-{stem}-k{repetitions}-n{n_samples}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gen-iadd-fixtures", description=__doc__)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="K — adder repetitions; expected output is r0 + K*r1 mod 2^width (default: 1)",
    )
    parser.add_argument(
        "--samples", type=int, default=16, help="number of supplied test cases (default: 16)"
    )
    parser.add_argument("--tier", default="T0", help="scale-tier label (default: T0)")
    parser.add_argument(
        "--circuit", default="iadd64.kmx", help="source .kmx circuit (default: iadd64.kmx)"
    )
    parser.add_argument(
        "--pin-commit",
        default=UPSTREAM_PIN_COMMIT,
        help="upstream zkp_ecc commit recorded as workload_pin_commit",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: fixtures/v0.3-<circuit>-k<K>-n<N>.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare against the on-disk fixture (FIXTURE.DRIFT on mismatch)",
    )
    args = parser.parse_args(argv)

    out_path = (
        args.out
        if args.out is not None
        else _default_out_path(args.circuit, args.repetitions, args.samples)
    )

    try:
        fixture = build_iadd_fixture(
            repetitions=args.repetitions,
            n_samples=args.samples,
            tier=args.tier,
            circuit_source=args.circuit,
            pin_commit=args.pin_commit,
        )
    except FixtureError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    if args.check:
        return _check_against_disk(fixture, out_path)
    _write_atomic(fixture, out_path)
    _log.info("wrote iadd fixture to %s", out_path)
    return 0


def _write_atomic(fixture: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(fixture, indent=2, sort_keys=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".partial.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialised)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _check_against_disk(fixture: dict[str, object], path: Path) -> int:
    if not path.is_file():
        print(
            f"{FixtureSubcode.DRIFT.value}: no fixture on disk at {path}; "
            "run `uv run gen-iadd-fixtures` first",
            file=sys.stderr,
        )
        return FIXTURE_EXIT_CODE
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    if _normalise(fixture) != _normalise(on_disk):
        print(f"{FixtureSubcode.DRIFT.value}: {path} differs from regenerated bytes", file=sys.stderr)
        return FIXTURE_EXIT_CODE
    return 0


def _normalise(fixture: dict[str, object]) -> dict[str, object]:
    """Strip fields not relevant to byte-stable comparison (generator_commit)."""
    return {k: v for k, v in fixture.items() if k != "generator_commit"}


def _git_head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and _SHA_RE.match(result.stdout.strip()):
            return result.stdout.strip()
    except OSError:
        pass
    return "0" * 40


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
