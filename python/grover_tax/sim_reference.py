"""Python reference simulator for the gate list `C` (RFC-0003).

The simulator is the cross-validation oracle for `gen_fixtures.py`
(`FIXTURE.CROSS_VALIDATION_FAIL` is raised when `run(C, x) != y`). It is
also the third independent implementation alongside the Rust and Cairo
mirrors — see RFC-0003 §"Motivation". The semantics here are normative
relative to the gate-list contract; if the two prover-side
implementations disagree on a witness, Python is the tie-breaker.

Bit ordering: bit `i` lives at byte `i // 8`, bit-position `i % 8`,
little-endian within each byte. This layout is shared with the Cairo and
Rust mirrors; the cross-impl test (#18 → `T-bit-layout`) asserts agreement.

Gate semantics (normative):

- `NOP`     — no effect.
- `NOT`     — `s[target] ^= 1`.
- `CNOT`    — `s[target] ^= s[ctrl_a]`.
- `TOFFOLI` — `s[target] ^= s[ctrl_a] & s[ctrl_b]`.

Any opcode outside `{0, 1, 2, 3}` raises `ValueError`. Gate field bounds
are checked by `grover_tax.serialise.deserialise`; this module trusts
its inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from grover_tax.errors import FIXTURE_EXIT_CODE, FixtureError, FixtureSubcode
from grover_tax.paths import fixture_path
from grover_tax.serialise import Gate, Opcode, deserialise

__all__ = ["BitVector", "main", "run", "step"]

_BITS_PER_BYTE: Final[int] = 8


class BitVector:
    """A mutable bit-vector backed by a `bytearray`.

    Bit `i` lives at byte `i // 8`, bit-position `i % 8` (LSB-first within
    each byte). Length is fixed at construction time and equal to the
    underlying byte count times 8 — `BitVector(b"\\x00")` has 8 bits.

    Out-of-range indices raise `IndexError`; out-of-range values raise
    `ValueError` (`v` must be `0` or `1`).
    """

    __slots__ = ("_data",)

    def __init__(self, bits: bytes) -> None:
        self._data = bytearray(bits)

    def __len__(self) -> int:
        return len(self._data) * _BITS_PER_BYTE

    def get(self, i: int) -> int:
        """Return bit `i` as `0` or `1`."""
        if not 0 <= i < len(self):
            raise IndexError(f"BitVector.get: index {i} out of range [0, {len(self)})")
        return (self._data[i // _BITS_PER_BYTE] >> (i % _BITS_PER_BYTE)) & 1

    def set(self, i: int, v: int) -> None:
        """Write bit `i` to `0` or `1`."""
        if not 0 <= i < len(self):
            raise IndexError(f"BitVector.set: index {i} out of range [0, {len(self)})")
        if v not in (0, 1):
            raise ValueError(f"BitVector.set: value must be 0 or 1, got {v}")
        byte_idx = i // _BITS_PER_BYTE
        bit_idx = i % _BITS_PER_BYTE
        if v == 1:
            self._data[byte_idx] |= 1 << bit_idx
        else:
            self._data[byte_idx] &= ~(1 << bit_idx) & 0xFF

    def to_bytes(self) -> bytes:
        return bytes(self._data)


def step(state: BitVector, gate: Gate) -> None:
    """Apply one gate, mutating `state`.

    The four supported opcodes are bound by the GateListV1 contract
    (`grover_tax.serialise.Opcode`). Unknown opcodes raise `ValueError`.
    """
    op = gate.opcode
    if op == Opcode.NOP:
        return
    if op == Opcode.NOT:
        state.set(gate.target, state.get(gate.target) ^ 1)
        return
    if op == Opcode.CNOT:
        c = state.get(gate.ctrl_a)
        state.set(gate.target, state.get(gate.target) ^ c)
        return
    if op == Opcode.TOFFOLI:
        c = state.get(gate.ctrl_a) & state.get(gate.ctrl_b)
        state.set(gate.target, state.get(gate.target) ^ c)
        return
    raise ValueError(f"step: unknown opcode {op}")


def run(circuit: Sequence[Gate], x: bytes) -> bytes:
    """Execute `circuit` on initial state `x`, return the final state.

    Pure-functional: no mutation of `x`; deterministic given `(circuit, x)`.
    """
    state = BitVector(x)
    for gate in circuit:
        step(state, gate)
    return state.to_bytes()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `uv run sim-check`.

    Loads `fixtures/v0.1.json` (or `--fixture` if passed), decodes the
    serialised gate list, and verifies that every test case satisfies
    `run(C, x_i_bytes) == y_i_bytes`. Returns `0` on full agreement,
    exit-code `4` (`FIXTURE.CROSS_VALIDATION_FAIL`) on any mismatch.
    """
    parser = argparse.ArgumentParser(prog="sim-check", description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to the fixture JSON (default: repo-root fixtures/v0.2.json).",
    )
    args = parser.parse_args(argv)

    fixture_file = args.fixture if args.fixture is not None else fixture_path("v0.2")
    if not fixture_file.is_file():
        print(
            f"FIXTURE.MISSING: no fixture at {fixture_file}; "
            "run `uv run gen-fixtures` first",
            file=sys.stderr,
        )
        return FIXTURE_EXIT_CODE

    try:
        _verify_fixture(fixture_file)
    except FixtureError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code
    return 0


def _verify_fixture(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    version = fixture.get("version", "v0.1")

    if version == "v0.1":
        # v0.1: all-NOP circuit; F-INV-4 (sim_reference cross-validation)
        # is intentionally skipped — run(C, x) == x != y (EC add result).
        return

    if version == "v0.3-iadd":
        # The adopted canonical workload (KB-2, #114): K-repeated iadd64.
        # x_hex / y_hex are the *full* register-encoded states (KB-3), so the
        # simulator runs over the whole state — no [:32] truncation.
        _verify_iadd_fixture(fixture)
        return

    # v0.2: random GTV1 circuit; the prover consumes only x_hex[:32] (P.X).
    # Retained for regression / T0 continuity (KB-2 keeps the random path).
    circuit = deserialise(bytes.fromhex(fixture["circuit_byte_serialisation_hex"]))
    for i, case in enumerate(fixture["test_cases"]):
        x_bytes = bytes.fromhex(case["x_hex"])  # 64 bytes = P.X || Q.X
        y_expected = bytes.fromhex(case["y_hex"])  # 32 bytes = circuit output
        y_got = run(circuit, x_bytes[:32])  # run on first 32 bytes (P.X)
        if y_got != y_expected:
            raise FixtureError(
                FixtureSubcode.CROSS_VALIDATION_FAIL,
                f"test case {i}: run(C, x[:32]) = {y_got.hex()} != y = {y_expected.hex()}",
            )


def _verify_iadd_fixture(fixture: dict[str, object]) -> None:
    """Cross-validate a `v0.3-iadd` fixture: ``run(C^K, x_state) == y_state``.

    Each test case carries the full register-encoded input/output state, so the
    reference simulator runs over the entire state and must reproduce the
    fixture's `y_hex` exactly — the F-INV-4 oracle for the adopted adder.
    The serialised circuit stores ONE adder repetition; `repetitions` (K)
    scales the workload, so the gate list is applied K times per case (KB-15).
    """
    circuit = deserialise(bytes.fromhex(str(fixture["circuit_byte_serialisation_hex"])))
    reps_raw = fixture.get("repetitions", 1)
    repetitions = reps_raw if isinstance(reps_raw, int) else int(str(reps_raw))
    cases = fixture["test_cases"]
    assert isinstance(cases, list)
    for i, case in enumerate(cases):
        x_bytes = bytes.fromhex(case["x_hex"])
        y_expected = bytes.fromhex(case["y_hex"])
        y_got = x_bytes
        for _ in range(repetitions):
            y_got = run(circuit, y_got)
        if y_got != y_expected:
            raise FixtureError(
                FixtureSubcode.CROSS_VALIDATION_FAIL,
                f"test case {i} (r0_in={case.get('r0_in')}, r1_in={case.get('r1_in')}): "
                f"run(C^{repetitions}, x) = {y_got.hex()} != y = {y_expected.hex()}",
            )


if __name__ == "__main__":  # pragma: no cover - thin __main__ wrapper
    raise SystemExit(main())
