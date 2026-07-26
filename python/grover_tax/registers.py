"""Register-aware test-case I/O for kickmix adders (KB-3, #115).

Kickmix adders are *two-register* circuits: `iadd64.kmx` transforms a pair
``(x, y)`` into ``((x + y) mod 2⁶⁴, y)``, where each register's value is laid
out across its qubits in **2's-complement little-endian** order
(`third_party/sp1/docs/kickmix_file_format.md`, `DEBUG_PRINT` semantics). The
grover-tax reference simulator (`grover_tax.sim_reference`) works on a flat
bit-vector keyed by qubit index, so to run a real adder we must encode register
*values* into qubit-indexed state and decode the result back.

This module supplies that bridge on top of the KB-1 transpiler
(`grover_tax.kmx`):

- `encode_registers` / `decode_registers` — map ``{register: value}`` to and
  from the flat `BitVector` state, using the register → qubit-index layout the
  transpiler recovered from `APPEND_TO_REGISTER` metadata (member ``j`` carries
  bit ``j``, least-significant first).
- `run_adder_case` — encode an input pair, run the transpiled gate list, decode
  the output pair.
- `iadd_test_cases` — a deterministic port of upstream's
  `docs/example_tools/print_iadd_cases.py` (`a b -> (a+b)%2ⁿ b`), seeded by the
  `XOF` primitive so a third party can re-derive the exact same cases.

Tier-1 note: these test cases are **supplied** (deterministically generated
here), *not* in-proof Fiat-Shamir-derived. Closing that divergence is KB-9
(#121); see `docs/spec/v0.3/KHATTAR-BENCHMARK-ALIGNMENT.md` §4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from grover_tax.kmx import KmxCircuit
from grover_tax.sim_reference import BitVector, run
from grover_tax.xof import XOF

__all__ = [
    "AdderCase",
    "decode_registers",
    "encode_registers",
    "iadd_test_cases",
    "min_state_bytes",
    "run_adder_case",
]

_BITS_PER_BYTE = 8

# The two registers of an in-place adder, by convention: r0 is the accumulator
# (overwritten with the sum), r1 is the offset (preserved).
_ACC_REGISTER = "r0"
_OFFSET_REGISTER = "r1"


@dataclass(frozen=True)
class AdderCase:
    """One adder test case: ``(x, y) -> ((x + y) mod 2^width, y)``.

    `x` and `y` are unsigned residues in ``[0, 2^width)`` — the same bit
    patterns a 2's-complement reading would use; only the interpretation of the
    top bit differs, and an in-place adder is agnostic to it (`x + y mod 2ⁿ`).
    """

    x: int
    y: int
    sum_out: int
    width: int


def min_state_bytes(circuit: KmxCircuit) -> int:
    """Smallest byte width whose bit-vector covers every qubit in `circuit`."""
    return (circuit.num_qubits + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE


def encode_registers(
    values: Mapping[str, int],
    layout: Mapping[str, Sequence[int]],
    num_bytes: int,
) -> bytes:
    """Encode ``{register: value}`` into a flat `num_bytes`-wide state.

    Each register's value is written little-endian across its qubit indices:
    layout member ``j`` (0 = least significant) carries bit ``j`` of the value.
    The value is reduced modulo ``2^width`` first, where ``width`` is the
    register's qubit count. Qubits not belonging to any named register are left
    at 0 (ancillae are expected to start and end cleared).

    Raises:
        KeyError: if `values` names a register absent from `layout`.
        IndexError: if a layout qubit index falls outside the `num_bytes` state.
    """
    bv = BitVector(bytes(num_bytes))
    for register, value in values.items():
        members = layout[register]
        width = len(members)
        residue = value & ((1 << width) - 1) if width else 0
        for j, qubit_index in enumerate(members):
            bv.set(qubit_index, (residue >> j) & 1)
    return bv.to_bytes()


def decode_registers(
    state: bytes,
    layout: Mapping[str, Sequence[int]],
) -> dict[str, int]:
    """Decode a flat state back into ``{register: unsigned value}``.

    Inverse of `encode_registers`: reads each register's qubits little-endian
    (member ``j`` → bit ``j``) and returns the unsigned integer in
    ``[0, 2^width)``.
    """
    bv = BitVector(state)
    decoded: dict[str, int] = {}
    for register, members in layout.items():
        value = 0
        for j, qubit_index in enumerate(members):
            value |= bv.get(qubit_index) << j
        decoded[register] = value
    return decoded


def run_adder_case(
    circuit: KmxCircuit,
    x: int,
    y: int,
    *,
    num_bytes: int | None = None,
) -> tuple[int, int]:
    """Run `circuit` on the input pair ``(x, y)``, returning the decoded output.

    Encodes the accumulator register ``r0 := x`` and the offset register
    ``r1 := y`` (the in-place-adder convention), executes the transpiled gate
    list under the reference simulator, then decodes both registers. For a
    correct in-place adder the result is ``((x + y) mod 2^width, y)``.
    """
    width = num_bytes if num_bytes is not None else min_state_bytes(circuit)
    layout = circuit.register_layout
    state = encode_registers({_ACC_REGISTER: x, _OFFSET_REGISTER: y}, layout, width)
    out_state = run(list(circuit.gates), state)
    decoded = decode_registers(out_state, layout)
    return decoded[_ACC_REGISTER], decoded[_OFFSET_REGISTER]


def iadd_test_cases(width: int, count: int, *, seed: bytes) -> list[AdderCase]:
    """Deterministically generate `count` adder cases of register `width` bits.

    A port of upstream's `print_iadd_cases.py` (`a b -> (a+b)%2ⁿ b`), but seeded
    by the `XOF` primitive so the stream is reproducible: any consumer that
    re-derives ``XOF(seed)`` and reads the same bytes obtains identical cases.
    Two width-bit operands are drawn per case (`x` then `y`), each from
    ``ceil(width/8)`` XOF bytes masked to `width` bits.

    Raises:
        ValueError: if `width` or `count` is non-positive.
    """
    if width <= 0:
        raise ValueError(f"iadd_test_cases: width must be positive, got {width}")
    if count <= 0:
        raise ValueError(f"iadd_test_cases: count must be positive, got {count}")

    xof = XOF(seed)
    nbytes = (width + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
    mask = (1 << width) - 1
    modulus = 1 << width

    cases: list[AdderCase] = []
    for _ in range(count):
        x = int.from_bytes(xof.read(nbytes), "little") & mask
        y = int.from_bytes(xof.read(nbytes), "little") & mask
        cases.append(AdderCase(x=x, y=y, sum_out=(x + y) % modulus, width=width))
    return cases
