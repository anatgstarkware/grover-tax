//! Lean qubit-circuit simulator — proves exactly what SP1 proves for the
//! iadd benchmark: simulate the GTV1 circuit K times on N input states and
//! check each output, plus a Blake2s commitment binding the circuit.
//!
//! State = `Felt252Dict<felt252>` keyed by qubit index 0..=511 (an unset
//! qubit reads 0). Branchy / non-oblivious by design (lean), no NOP padding.
//!
//! Qubit/state encoding (verified against `grover_tax.registers` /
//! `sim_reference`): qubits 0..255 = register r0, 256..511 = r1; in the
//! 64-byte state, qubit `q` is bit `q % 8` of byte `q / 8` (LSB-first).
//! Output after K reps: r0' = r0 + K·r1 (mod 2^256), r1 unchanged — checked
//! as full 64-byte state equality against the fixture's `y_hex`.

use core::array::ArrayTrait;
use core::dict::Felt252Dict;
use grover_tax_circuit::Gate;
use grover_tax_circuit::{NO_CTRL, OP_NOP, OP_NOT, OP_CNOT, OP_TOFFOLI};
use grover_tax_circuit::serialise::deserialise;
use grover_tax_circuit::commit::commit_blake2s;
use grover_tax_circuit::digest_to_u256_be;

pub const NUM_QUBITS: u32 = 512;
pub const STATE_BYTES: u32 = 64;

/// `2^k` for small `k` (bit-in-byte, 0..=7).
fn pow2_u32(k: u32) -> u32 {
    let mut acc: u32 = 1;
    let mut i: u32 = 0;
    loop {
        if i == k {
            break acc;
        }
        acc = acc * 2;
        i = i + 1;
    }
}

/// Read control qubit `q`, treating `NO_CTRL` as literal 0.
fn read_ctrl(ref state: Felt252Dict<felt252>, q: u32) -> felt252 {
    if q == NO_CTRL {
        0
    } else {
        state.get(q.into())
    }
}

/// Apply one gate in place — branchy, only the work the op needs.
fn apply_gate(ref state: Felt252Dict<felt252>, gate: Gate) {
    if gate.opcode == OP_NOP {
        return;
    }
    if gate.opcode == OP_NOT {
        let t = state.get(gate.target.into());
        state.insert(gate.target.into(), 1 - t);
        return;
    }
    if gate.opcode == OP_CNOT {
        let a = read_ctrl(ref state, gate.ctrl_a);
        if a == 1 {
            let t = state.get(gate.target.into());
            state.insert(gate.target.into(), 1 - t);
        }
        return;
    }
    if gate.opcode == OP_TOFFOLI {
        let a = read_ctrl(ref state, gate.ctrl_a);
        let b = read_ctrl(ref state, gate.ctrl_b);
        if a == 1 && b == 1 {
            let t = state.get(gate.target.into());
            state.insert(gate.target.into(), 1 - t);
        }
        return;
    }
    panic!("qsim: bad opcode {}", gate.opcode);
}

/// Run `gates` over `state`, `k` times.
fn run_reps(ref state: Felt252Dict<felt252>, gates: @Array<Gate>, k: u32) {
    let n = gates.len();
    let mut rep: u32 = 0;
    loop {
        if rep == k {
            break;
        }
        let mut gi: u32 = 0;
        loop {
            if gi == n {
                break;
            }
            apply_gate(ref state, *gates.at(gi));
            gi = gi + 1;
        };
        rep = rep + 1;
    };
}

/// Load a 64-byte state (next STATE_BYTES felts of `span`) into the qubit dict.
fn load_state(ref state: Felt252Dict<felt252>, ref span: Span<felt252>) {
    let mut byte_idx: u32 = 0;
    loop {
        if byte_idx == STATE_BYTES {
            break;
        }
        let bv: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
        let mut bit: u32 = 0;
        loop {
            if bit == 8 {
                break;
            }
            let v: u32 = (bv / pow2_u32(bit)) & 1;
            let q: felt252 = (byte_idx * 8 + bit).into();
            state.insert(q, v.into());
            bit = bit + 1;
        };
        byte_idx = byte_idx + 1;
    };
}

/// Re-pack the qubit dict into 64 bytes and assert equality against the next
/// STATE_BYTES felts of `span` (the expected output state).
fn check_output(ref state: Felt252Dict<felt252>, ref span: Span<felt252>) {
    let mut byte_idx: u32 = 0;
    loop {
        if byte_idx == STATE_BYTES {
            break;
        }
        let expected: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
        let mut got: u32 = 0;
        let mut bit: u32 = 0;
        loop {
            if bit == 8 {
                break;
            }
            let q: felt252 = (byte_idx * 8 + bit).into();
            let vu: u32 = state.get(q).try_into().unwrap();
            got = got + vu * pow2_u32(bit);
            bit = bit + 1;
        };
        assert!(got == expected, "qsim: output byte {} mismatch", byte_idx);
        byte_idx = byte_idx + 1;
    };
}

/// Bootloader `Cairo1Executable` entry point.
///
/// Input layout (flat `Array<felt252>`):
///   [0]            n_cb : u32
///   [1 .. n_cb]    circuit bytes
///   [n_cb+1]       commitment_lo : u128  (Blake2s digest bytes[16:32] big-endian)
///   [n_cb+2]       commitment_hi : u128  (Blake2s digest bytes[0:16]  big-endian)
///   [n_cb+3]       k : u32   — repetitions
///   [n_cb+4]       n : u32   — number of input states
///   then per input: 64 x-bytes (input state) then 64 y-bytes (expected output)
#[executable]
pub fn iadd_sim_executable(input: Array<felt252>) -> felt252 {
    let mut span = input.span();

    let n_cb: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
    let mut circuit_bytes: Array<u8> = ArrayTrait::new();
    let mut i: u32 = 0;
    loop {
        if i == n_cb {
            break;
        }
        let b: u8 = (*span.pop_front().unwrap()).try_into().unwrap();
        circuit_bytes.append(b);
        i = i + 1;
    };
    let comm_lo: u128 = (*span.pop_front().unwrap()).try_into().unwrap();
    let comm_hi: u128 = (*span.pop_front().unwrap()).try_into().unwrap();
    let k: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
    let n: u32 = (*span.pop_front().unwrap()).try_into().unwrap();

    // Bind the circuit: Blake2s(circuit_bytes) == public commitment.
    let digest = commit_blake2s(@circuit_bytes);
    let computed = digest_to_u256_be(digest);
    assert!(computed == u256 { low: comm_lo, high: comm_hi }, "qsim: commitment mismatch");

    let gates = deserialise(@circuit_bytes);

    let mut shot: u32 = 0;
    loop {
        if shot == n {
            break;
        }
        let mut state: Felt252Dict<felt252> = Default::default();
        load_state(ref state, ref span);
        run_reps(ref state, @gates, k);
        check_output(ref state, ref span);
        shot = shot + 1;
    };
    1
}
