//! grover-tax Cairo program (apples-to-apples v0.1 / A2).
//!
//! Mirrors the SP1 side at `third_party/sp1/program/src/main.rs`:
//!
//! ```text
//! p        = 2^256 − 2^32 − 977            (secp256k1 prime)
//! σ₀       = Blake2s(circuit_bytes) reduced mod p   (Stwo side — RFC-0005)
//! σ_{i+1}  = (σ_i + (i + 1)) mod p          for i ∈ [0, gate_count)
//! ```
//!
//! Public outputs: `(commitment: u256 from Blake2s, sigma_n: u256)`.
//!
//! The program is callable both:
//!   * as a Cairo unit test entry point (via `apples_to_apples()`), and
//!   * as a cairo-vm-executable function for `stwo-cairo`'s `run_and_prove`.

pub mod limbs;
pub mod gates;
pub mod serialise;
pub mod commit;
pub mod c_tests;
pub mod io;
pub mod qsim;

use core::array::ArrayTrait;
use grover_tax_circuit::commit::commit_blake2s;

#[derive(Drop, Copy)]
pub struct Gate {
    pub opcode: u32,
    pub target: u32,
    pub ctrl_a: u32,
    pub ctrl_b: u32,
}

#[derive(Drop, Copy)]
pub struct TestCase {
    pub x_lo: u256,
    pub x_hi: u256,
    pub y: u256,
}

pub const NO_CTRL: u32 = 0xFFFF;
pub const OP_NOP: u32 = 0;
pub const OP_NOT: u32 = 1;
pub const OP_CNOT: u32 = 2;
pub const OP_TOFFOLI: u32 = 3;

/// secp256k1 prime `p = 2^256 − 2^32 − 977`.
///
/// Hex: `0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`.
/// Cairo's `u256` is `{ low: u128, high: u128 }` where `low` holds bits 0..127
/// and `high` holds bits 128..255.
fn secp256k1_p() -> u256 {
    u256 {
        low: 0xFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F_u128,
        high: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_u128,
    }
}

/// Convert 8 little-endian u32 words (= Blake2s output) to a big-endian
/// `u256`. The 32-byte digest is interpreted exactly as SP1's side does
/// with `U256::from_be_bytes` of the raw digest bytes.
pub fn digest_to_u256_be(d: [u32; 8]) -> u256 {
    let [d0, d1, d2, d3, d4, d5, d6, d7] = d;
    // d0..d7 are little-endian u32 words of the digest bytes b0..b31.
    // The big-endian u256 reads bytes b0 (MSB) through b31 (LSB).
    // Word d_i contains bytes b_{4i}..b_{4i+3} as the low-to-high byte.
    // u256 layout in Cairo: `low` = bytes 16..31 (low 128 bits),
    //                       `high` = bytes 0..15 (high 128 bits).
    // Each u32 word reversed back to its 4 BE bytes:
    let be_w0: u128 = swap_u32_endian(d0).into();
    let be_w1: u128 = swap_u32_endian(d1).into();
    let be_w2: u128 = swap_u32_endian(d2).into();
    let be_w3: u128 = swap_u32_endian(d3).into();
    let be_w4: u128 = swap_u32_endian(d4).into();
    let be_w5: u128 = swap_u32_endian(d5).into();
    let be_w6: u128 = swap_u32_endian(d6).into();
    let be_w7: u128 = swap_u32_endian(d7).into();
    let high = be_w0 * 0x1000000000000000000000000_u128
             + be_w1 * 0x10000000000000000_u128
             + be_w2 * 0x100000000_u128
             + be_w3;
    let low = be_w4 * 0x1000000000000000000000000_u128
            + be_w5 * 0x10000000000000000_u128
            + be_w6 * 0x100000000_u128
            + be_w7;
    u256 { low, high }
}

/// Byte-swap a u32 (little-endian → big-endian word).
fn swap_u32_endian(w: u32) -> u32 {
    let b0 = w / 0x1000000_u32;
    let b1 = (w / 0x10000_u32) & 0xFF_u32;
    let b2 = (w / 0x100_u32) & 0xFF_u32;
    let b3 = w & 0xFF_u32;
    b3 * 0x1000000_u32 + b2 * 0x10000_u32 + b1 * 0x100_u32 + b0
}

/// Modular addition `(a + b) mod p`. Safe when `a < p` and `b` is small
/// (≪ 2^64): `a + b < 2^256` always, so Cairo's u256 `+` does not
/// overflow. We then conditionally subtract `p`.
fn mod_add_p(a: u256, b: u256, p: u256) -> u256 {
    let sum = a + b;
    if sum >= p { sum - p } else { sum }
}

/// Apples-to-apples kernel: N modular additions over secp256k1's prime.
///
/// Returns `(commitment, sigma_n)`. The caller is expected to publish
/// both as cairo-vm output for `stwo-cairo` to lift into public inputs.
pub fn apples_to_apples(circuit_bytes: @Array<u8>, n: u64) -> (u256, u256) {
    let digest = commit_blake2s(circuit_bytes);
    let commitment = digest_to_u256_be(digest);
    let p = secp256k1_p();
    let mut sigma = commitment % p;
    let mut i: u64 = 0;
    loop {
        if i == n { break; }
        let step: u256 = (i + 1).into();
        sigma = mod_add_p(sigma, step, p);
        i = i + 1;
    };
    (commitment, sigma)
}

/// Skeleton main retained for backward compatibility with existing
/// tests. Real execution goes through `apples_to_apples_executable`
/// driven by `scarb execute` → `stwo-cairo`.
fn main(
    public_test_cases: Array<TestCase>,
    public_h_c: u256,
    secret_c: Array<Gate>,
) {
    let _ = public_test_cases.len();
    let _ = public_h_c;
    let _ = secret_c.len();
}

/// Cairo-vm-executable entry point for `scarb execute` + `stwo-cairo`.
///
/// Argument convention (passed via `--arguments-file` or `--arguments`):
///   * `circuit_bytes`: `Array<u8>` (the fixture's
///     `circuit_byte_serialisation_hex` decoded).
///   * `n`: `u64` — the loop iteration count (= fixture's `gate_count`).
///   * `expected_commitment`: `u256` — the verifier-provided expected
///     `Blake2s(circuit_bytes)` reduced mod p (= σ₀).
///   * `expected_sigma`: `u256` — the verifier-provided expected σ_N.
///
/// The function asserts that the computed `(commitment, sigma_n)` equals
/// the provided expected values. The proof of "I ran this and the
/// asserts didn't fire" is equivalent to "I performed the N mod-adds
/// and arrived at (expected_commitment, expected_sigma)". This
/// avoids returning large u128 values via the AP region — stwo-cairo's
/// `--program_type executable` extract_public_segments path needs all
/// AP slots to fit in u32.
///
/// Returns `()` so cairo-vm writes nothing to the output segment;
/// public commitments are the input arguments themselves.
/// Cairo-vm-executable entry point for `scarb execute` + `stwo-cairo`.
///
/// Single-argument signature: a flat `Array<felt252>` matching
/// stwo-cairo's `user_args: vec![vec![Arg::Array(args)]]` convention.
/// Cairo 1's executable runtime stuffs *positional* args into the AP
/// region — and u128/u64 values exceed the u32 cast inside
/// stwo-cairo's `extract_public_segments`. A single `Array<felt252>`
/// arg lives behind a *pointer* (small u32), which the AP layout
/// can hold safely.
///
/// Layout of `input`:
///   [0] seed_lo (felt; low 128 bits of σ₀)
///   [1] seed_hi (felt; high 128 bits of σ₀)
///   [2] n       (felt; loop iteration count)
///
/// Returns `()` to keep the program tail's AP slots empty.
/// The "public" output is the σ_N value computed in the loop — we
/// `assert!` it against the caller-provided `expected_sigma_*` in
/// the input array, so the proof attests both to the loop body and
/// to the agreed σ_N. To preserve this constraint without complicating
/// the AP layout, we encode expected σ_N in input slots [3] and [4].
/// Cairo-vm-executable entry point for v0.2 gate circuit execution.
///
/// Proves: Blake2s(circuit_bytes) == expected_commitment, and for each
/// test case (x, y): simulate(circuit, x_bytes[:32]) == y_bytes[:32].
///
/// Input layout (flat `Array<felt252>`):
///   [0]              n_cb: u32   — number of circuit bytes
///   [1 .. n_cb]      circuit bytes, one byte per felt252
///   [n_cb+1]         commitment_lo: u128 — low  128 bits of Blake2s digest
///   [n_cb+2]         commitment_hi: u128 — high 128 bits of Blake2s digest
///   [n_cb+3]         n_tc: u32   — number of test cases
///   per test case (64 felts):
///     [0..31]  x_bytes: u8 — first 32 bytes of P.X (256-bit initial state)
///     [32..63] y_bytes: u8 — 32-byte expected circuit output state
#[executable]
pub fn apples_to_apples_executable(input: Array<felt252>) -> felt252 {
    let mut span = input.span();

    // 1. Read circuit bytes.
    let n_cb: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
    let mut circuit_bytes: Array<u8> = ArrayTrait::new();
    let mut ci: u32 = 0_u32;
    loop {
        if ci == n_cb { break; }
        let b: u8 = (*span.pop_front().unwrap()).try_into().unwrap();
        circuit_bytes.append(b);
        ci = ci + 1_u32;
    };

    // 2. Read expected Blake2s commitment.
    let expected_lo: u128 = (*span.pop_front().unwrap()).try_into().unwrap();
    let expected_hi: u128 = (*span.pop_front().unwrap()).try_into().unwrap();

    // 3. Verify commitment: Blake2s(circuit_bytes) == expected.
    let digest = commit_blake2s(@circuit_bytes);
    let computed = digest_to_u256_be(digest);
    let expected_commitment = u256 { low: expected_lo, high: expected_hi };
    assert!(computed == expected_commitment, "blake2s commitment mismatch");

    // 4. Deserialise circuit bytes into gates.
    let gates = grover_tax_circuit::serialise::deserialise(@circuit_bytes);

    // 5. Read n_tc and run each test case.
    let n_tc: u32 = (*span.pop_front().unwrap()).try_into().unwrap();
    let mut tc: u32 = 0_u32;
    loop {
        if tc == n_tc { break; }
        // Read 32 x-bytes.
        let mut x_bytes: Array<u8> = ArrayTrait::new();
        let mut bi: u32 = 0_u32;
        loop {
            if bi == 32_u32 { break; }
            let b: u8 = (*span.pop_front().unwrap()).try_into().unwrap();
            x_bytes.append(b);
            bi = bi + 1_u32;
        };
        // Read 32 y-bytes.
        let mut y_bytes: Array<u8> = ArrayTrait::new();
        let mut bi: u32 = 0_u32;
        loop {
            if bi == 32_u32 { break; }
            let b: u8 = (*span.pop_front().unwrap()).try_into().unwrap();
            y_bytes.append(b);
            bi = bi + 1_u32;
        };
        // Convert to states.
        let x_state = grover_tax_circuit::io::bytes_to_state(@x_bytes);
        let y_state = grover_tax_circuit::io::bytes_to_state(@y_bytes);
        // Run gate loop.
        let mut s = x_state;
        let mut gi: u32 = 0_u32;
        loop {
            if gi == gates.len() { break; }
            let gate = *gates.at(gi);
            grover_tax_circuit::gates::range_check_gate(gate);
            s = grover_tax_circuit::gates::step(s, gate);
            gi = gi + 1_u32;
        };
        assert!(s == y_state, "test case simulation mismatch");
        tc = tc + 1_u32;
    };

    1  // success
}

#[cfg(test)]
mod tests {
    use core::array::ArrayTrait;
    use super::{apples_to_apples, secp256k1_p, mod_add_p};

    #[test]
    fn mod_add_p_basic() {
        let p = secp256k1_p();
        let a: u256 = 1_u256;
        let b: u256 = 2_u256;
        let r = mod_add_p(a, b, p);
        assert!(r == 3_u256);
    }

    #[test]
    fn mod_add_p_wrap() {
        // (p - 1) + 2 = p + 1 ≡ 1  (mod p)
        let p = secp256k1_p();
        let r = mod_add_p(p - 1_u256, 2_u256, p);
        assert!(r == 1_u256);
    }

    #[test]
    fn apples_empty_circuit_n_zero() {
        let bytes = ArrayTrait::<u8>::new();
        let (commitment, sigma) = apples_to_apples(@bytes, 0_u64);
        // n=0 → sigma = commitment mod p (Blake2s digest of empty input,
        // reduced mod secp256k1's prime).
        // We don't pin the exact value here — `apples_to_apples_python_match`
        // (Python-side cross-check) is the authoritative gate.
        let p = secp256k1_p();
        assert!(sigma == commitment % p);
    }

    #[test]
    fn apples_short_input_n_one() {
        // Single-step loop adds 1.
        let mut bytes = ArrayTrait::<u8>::new();
        bytes.append('a');
        bytes.append('b');
        bytes.append('c');
        let (_, sigma_0) = apples_to_apples(@bytes, 0_u64);
        let (_, sigma_1) = apples_to_apples(@bytes, 1_u64);
        let p = secp256k1_p();
        assert!(sigma_1 == mod_add_p(sigma_0, 1_u256, p));
    }
}
