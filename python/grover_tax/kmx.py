"""Kickmix (`.kmx`) → GTV1 transpiler — classical reversible subset (KB-1, #113).

The upstream Khattar/Google benchmark (`tanujkhattar/zkp_ecc`) ships its
circuits in the human-readable **kickmix** assembly format (`.kmx`, see
`third_party/sp1/docs/kickmix_file_format.md`). grover-tax's prover side speaks
**GTV1** — a four-opcode binary gate list (`grover_tax.serialise`). This module
bridges the two for the *classical reversible subset* of kickmix so the
existing `{NOP, NOT, CNOT, TOFFOLI}` simulator can run the real addition
circuits (`iadd64.kmx`, `iadd8.kmx`) that Tanuj asked us to benchmark.

Scope (Tier 1 of the alignment plan,
`docs/spec/v0.3/KHATTAR-BENCHMARK-ALIGNMENT.md` §5):

- Parse the kickmix text grammar: instruction name, space-separated
  qubit/bit/register targets, optional ``if bN`` condition, ``#`` comments,
  and arbitrary indentation (comments and indentation are decorative).
- Map the reversible classical gates ``X→NOT``, ``CX→CNOT``, ``CCX→TOFFOLI``.
- Honour ``REGISTER`` / ``APPEND_TO_REGISTER`` metadata to build the
  register → qubit-index layout (consumed by KB-3 / #115 for register I/O).
- **Reject**, with a clear ``FIXTURE.UNSUPPORTED_INSTRUCTION`` error, any
  circuit using a phase/measurement/classical/control-flow instruction
  (``Z CZ CCZ HMR NEG SWAP R BIT_INVERT BIT_STORE0 BIT_STORE1
  PUSH_CONDITION POP_CONDITION DEBUG_PRINT``) or any ``if`` condition. Those
  belong to the full kickmix simulator (Tier 2 / KB-8, #120).

Wire-id convention: a kickmix qubit ``qN`` maps to GTV1 wire index ``N`` and to
bit ``N`` of `grover_tax.sim_reference.BitVector` — a direct identity, so a
transpiled circuit run under the reference simulator reproduces the upstream
register semantics once the registers are loaded (KB-3).

Determinism: parsing is a pure function of the input bytes; gate order is the
file's instruction order; register members are recorded in
``APPEND_TO_REGISTER`` order (least-significant first), matching the kickmix
2's-complement little-endian register convention.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from grover_tax.errors import FixtureError, FixtureSubcode
from grover_tax.serialise import UNUSED_CTRL, Gate, Opcode, serialise

__all__ = [
    "KmxCircuit",
    "kmx_source_sha256",
    "opcode_histogram",
    "parse_kmx",
    "transpile_file",
]

# The kickmix grammar: a target is a type-prefixed non-negative integer.
_TARGET_RE = re.compile(r"^(?P<kind>[qbr])(?P<id>[0-9]+)$")
# An instruction name is upper-case, starting with a letter.
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# GTV1 wire ids are u16; a kickmix qubit id beyond this cannot be represented.
_U16_MAX = 0xFFFF

# The reversible classical subset reachable by the GTV1 simulator. Anything
# else is a Tier-2 instruction and is rejected here (KB-8 / #120 covers them).
_CLASSICAL_GATES = frozenset({"X", "CX", "CCX"})
_METADATA_GATES = frozenset({"REGISTER", "APPEND_TO_REGISTER"})

# `APPEND_TO_REGISTER <member> <register>` has exactly two operands.
_APPEND_TO_REGISTER_OPERANDS = 2


@dataclass(frozen=True)
class KmxCircuit:
    """A kickmix circuit transpiled to the GTV1 gate-list contract.

    Attributes:
        gates: The transpiled gate list, in source order. Only ``NOT``,
            ``CNOT`` and ``TOFFOLI`` opcodes appear (no ``NOP`` padding — the
            fixture builder pads to a power of two if it wants to).
        registers: Maps each register id to its qubit indices in
            least-significant-first order (``APPEND_TO_REGISTER`` order). This
            is the layout KB-3 (#115) uses to encode/decode 2's-complement
            little-endian register values.
        num_qubits: One past the largest qubit index referenced anywhere in
            the circuit (gates *or* register metadata). The reference state
            must be at least this wide.
        source_bytes: The exact UTF-8 bytes of the parsed ``.kmx`` source.
            Carried so KB-4 (#116) can commit ``sha256`` over the *raw* kmx
            bytes for cross-comparability with upstream's ``sha256(iadd64.kmx)``
            (our GTV1 bytes necessarily differ).
    """

    gates: tuple[Gate, ...]
    registers: dict[int, tuple[int, ...]]
    num_qubits: int
    source_bytes: bytes

    @property
    def register_layout(self) -> dict[str, list[int]]:
        """JSON-friendly view of `registers` (``"r0"`` → ``[0, 1, …]``)."""
        return {f"r{rid}": list(members) for rid, members in sorted(self.registers.items())}


def opcode_histogram(gates: tuple[Gate, ...]) -> dict[str, int]:
    """Count gates by opcode name — the golden-test fingerprint of a circuit."""
    counts = {op.name: 0 for op in Opcode}
    for g in gates:
        counts[Opcode(g.opcode).name] += 1
    return counts


def kmx_source_sha256(source_bytes: bytes) -> str:
    """SHA-256 over the raw ``.kmx`` bytes — the cross-comparable commitment.

    Upstream commits ``sha256(circuit_kmx_bytes)``; our GTV1 serialisation has
    different bytes, so KB-4's commitment policy records *this* hash for direct
    comparison with the reference.
    """
    return hashlib.sha256(source_bytes).hexdigest()


def parse_kmx(text: str, *, source_bytes: bytes | None = None) -> KmxCircuit:
    """Parse kickmix `text` into a `KmxCircuit`, classical subset only.

    Args:
        text: The kickmix circuit source.
        source_bytes: The original file bytes for the raw-kmx commitment. When
            omitted, ``text.encode("utf-8")`` is used (round-trips for files
            read as UTF-8 text).

    Raises:
        FixtureError(FIXTURE.UNSUPPORTED_INSTRUCTION): on any phase /
            measurement / classical-bit / control-flow instruction, or any
            ``if`` condition.
        FixtureError(FIXTURE.KMX_PARSE_ERROR): on malformed syntax — bad
            instruction name, unparseable target, wrong gate arity,
            non-qubit operand to a quantum gate, or a qubit id over the GTV1
            u16 wire-id limit.
    """
    gates: list[Gate] = []
    registers: dict[int, list[int]] = {}
    max_qubit = -1

    for lineno, raw in enumerate(text.splitlines(), start=1):
        name, targets, has_condition = _tokenise(raw, lineno)
        if name is None:
            continue  # blank or comment-only line

        if has_condition:
            raise FixtureError(
                FixtureSubcode.UNSUPPORTED_INSTRUCTION,
                f"line {lineno}: `if` conditions are Tier-2 (KB-8); {raw.strip()!r}",
            )

        if name in _METADATA_GATES:
            _apply_metadata(name, targets, registers, lineno)
            for kind, idx in targets:
                if kind == "q":
                    max_qubit = max(max_qubit, idx)
            continue

        if name in _CLASSICAL_GATES:
            gate, gate_max_qubit = _to_gate(name, targets, lineno)
            gates.append(gate)
            max_qubit = max(max_qubit, gate_max_qubit)
            continue

        raise FixtureError(
            FixtureSubcode.UNSUPPORTED_INSTRUCTION,
            f"line {lineno}: instruction {name!r} is outside the classical "
            f"reversible subset {sorted(_CLASSICAL_GATES | _METADATA_GATES)}; "
            "it belongs to the full kickmix simulator (KB-8, #120)",
        )

    return KmxCircuit(
        gates=tuple(gates),
        registers={rid: tuple(members) for rid, members in registers.items()},
        num_qubits=max_qubit + 1,
        source_bytes=source_bytes if source_bytes is not None else text.encode("utf-8"),
    )


def transpile_file(path: str | Path) -> KmxCircuit:
    """Read a ``.kmx`` file and transpile it (classical subset)."""
    raw = Path(path).read_bytes()
    return parse_kmx(raw.decode("utf-8"), source_bytes=raw)


# -- parsing internals -------------------------------------------------------


def _tokenise(raw: str, lineno: int) -> tuple[str | None, list[tuple[str, int]], bool]:
    """Split one source line into ``(name, targets, has_condition)``.

    Strips the comment and indentation (both decorative). Returns
    ``(None, [], False)`` for a blank/comment-only line. ``targets`` is a list
    of ``(kind, id)`` pairs where kind is ``"q"``, ``"b"`` or ``"r"``.
    """
    # Drop the comment: everything from the first '#'. The format permits
    # non-ASCII only inside comments, so we slice before validating ASCII.
    code = raw.split("#", 1)[0].strip()
    if not code:
        return None, [], False

    tokens = code.split()
    name = tokens[0]
    if not _NAME_RE.match(name):
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"line {lineno}: malformed instruction name {name!r}",
        )

    rest = tokens[1:]
    has_condition = False
    if "if" in rest:
        idx = rest.index("if")
        # Validate the condition has exactly one bit operand after `if`, so a
        # malformed condition is a parse error rather than a silent drop — even
        # though we ultimately reject all conditions as unsupported.
        cond = rest[idx + 1 :]
        if len(cond) != 1 or not _TARGET_RE.match(cond[0]):
            raise FixtureError(
                FixtureSubcode.KMX_PARSE_ERROR,
                f"line {lineno}: malformed `if` condition {' '.join(cond)!r}",
            )
        has_condition = True
        rest = rest[:idx]

    targets = [_parse_target(tok, lineno) for tok in rest]
    return name, targets, has_condition


def _parse_target(token: str, lineno: int) -> tuple[str, int]:
    m = _TARGET_RE.match(token)
    if m is None:
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"line {lineno}: malformed target {token!r} (want q<N>/b<N>/r<N>)",
        )
    return m.group("kind"), int(m.group("id"))


def _apply_metadata(
    name: str,
    targets: list[tuple[str, int]],
    registers: dict[int, list[int]],
    lineno: int,
) -> None:
    """Update `registers` for a ``REGISTER`` / ``APPEND_TO_REGISTER`` line."""
    if name == "REGISTER":
        # `REGISTER rN` declares (and ensures the existence of) register N.
        if len(targets) != 1 or targets[0][0] != "r":
            raise FixtureError(
                FixtureSubcode.KMX_PARSE_ERROR,
                f"line {lineno}: REGISTER takes exactly one register operand",
            )
        registers.setdefault(targets[0][1], [])
        return

    # APPEND_TO_REGISTER <qubit|bit> rN — append a member in increasing
    # significance. The classical subset only loads qubit registers (the
    # adders use no classical-bit registers), so a bit member is rejected.
    if len(targets) != _APPEND_TO_REGISTER_OPERANDS or targets[1][0] != "r":
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"line {lineno}: APPEND_TO_REGISTER takes <qubit> <register>",
        )
    member_kind, member_id = targets[0]
    reg_id = targets[1][1]
    if member_kind != "q":
        raise FixtureError(
            FixtureSubcode.UNSUPPORTED_INSTRUCTION,
            f"line {lineno}: classical-subset registers hold qubits only; "
            f"bit member b{member_id} requires the full simulator (KB-8, #120)",
        )
    _check_wire_id(member_id, lineno)
    registers.setdefault(reg_id, []).append(member_id)


def _to_gate(name: str, targets: list[tuple[str, int]], lineno: int) -> tuple[Gate, int]:
    """Map a classical gate to a `Gate`, returning the max qubit index touched."""
    arity = {"X": 1, "CX": 2, "CCX": 3}[name]
    if len(targets) != arity:
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"line {lineno}: {name} expects {arity} qubit target(s), got {len(targets)}",
        )
    qubits: list[int] = []
    for kind, idx in targets:
        if kind != "q":
            raise FixtureError(
                FixtureSubcode.KMX_PARSE_ERROR,
                f"line {lineno}: {name} operands must be qubits, got {kind}{idx}",
            )
        _check_wire_id(idx, lineno)
        qubits.append(idx)

    if name == "X":
        (t,) = qubits
        return Gate(Opcode.NOT, t, UNUSED_CTRL, UNUSED_CTRL), t
    if name == "CX":
        # kickmix `CX qC qT`: control first, target second (flip qT where qC on).
        c, t = qubits
        return Gate(Opcode.CNOT, t, c, UNUSED_CTRL), max(c, t)
    # CCX qC1 qC2 qT: two controls, then target.
    c1, c2, t = qubits
    return Gate(Opcode.TOFFOLI, t, c1, c2), max(c1, c2, t)


def _check_wire_id(idx: int, lineno: int) -> None:
    if idx > _U16_MAX:
        raise FixtureError(
            FixtureSubcode.KMX_PARSE_ERROR,
            f"line {lineno}: qubit id {idx} exceeds the GTV1 u16 wire limit {_U16_MAX}",
        )


# -- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``uv run kmx-transpile <file.kmx>`` — transpile and report.

    Prints the opcode histogram, qubit count, register layout, and the raw-kmx
    SHA-256. With ``--out PATH`` also writes the canonical GTV1 bytes.
    """
    parser = argparse.ArgumentParser(prog="kmx-transpile", description=__doc__)
    parser.add_argument("kmx", type=Path, help="path to the .kmx circuit file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write canonical GTV1 bytes to this path",
    )
    args = parser.parse_args(argv)

    try:
        circuit = transpile_file(args.kmx)
    except FixtureError as e:
        print(str(e), file=sys.stderr)
        return e.exit_code

    hist = opcode_histogram(circuit.gates)
    print(f"gates: {len(circuit.gates)}")
    print(f"  histogram: {hist}")
    print(f"qubits: {circuit.num_qubits}")
    print(f"registers: {circuit.register_layout}")
    print(f"kmx_source_sha256: {kmx_source_sha256(circuit.source_bytes)}")

    if args.out is not None:
        args.out.write_bytes(serialise(circuit.gates))
        print(f"wrote GTV1 bytes to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin __main__ wrapper
    raise SystemExit(main())
