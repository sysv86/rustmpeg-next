// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
// Copyright 2026 The rustmpeg-next Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bit-level bitstream writer for AV2 OBUs, Sequence Headers, Frame Headers, and Tile Groups.
//!
//! Every method in [`PutBits`] is the exact mathematical inverse of the corresponding
//! reader method in [`crate::av2::getbits::GetBits`] and the AV2 v1.0.0 specification (§4.11 & §8.1).

use crate::av2::intops::{recenter, ulog2};

#[derive(Debug, Clone, Default)]
pub struct PutBits {
    buf: Vec<u8>,
    cur_byte: u8,
    bits_used: u8,
}

impl PutBits {
    /// Creates a new empty bit writer.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(256),
            cur_byte: 0,
            bits_used: 0,
        }
    }

    /// Creates a new bit writer with pre-allocated byte capacity.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            cur_byte: 0,
            bits_used: 0,
        }
    }

    /// Total number of bits written so far.
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.buf.len() * 8 + self.bits_used as usize
    }

    /// Returns `true` if the bit writer is currently aligned on a byte boundary.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_used == 0
    }

    /// Writes a single bit (`0` or `1`) MSB-first.
    #[inline]
    pub fn put_bit(&mut self, bit: u32) {
        let b = (bit & 1) as u8;
        self.cur_byte |= b << (7 - self.bits_used);
        self.bits_used += 1;
        if self.bits_used == 8 {
            self.buf.push(self.cur_byte);
            self.cur_byte = 0;
            self.bits_used = 0;
        }
    }

    /// Writes `n` unsigned bits (`0..=32`) from `val` MSB-first (`f(n)` in AV2 §4.11).
    #[inline]
    pub fn put_bits(&mut self, n: i32, val: u32) {
        debug_assert!((0..=32).contains(&n));
        if n == 0 {
            return;
        }
        for i in (0..n).rev() {
            self.put_bit((val >> i) & 1);
        }
    }

    /// Writes `n` signed two's-complement bits (`0..=32`) from `val` MSB-first (`su(n)` in AV2 §4.11).
    #[inline]
    pub fn put_sbits(&mut self, n: i32, val: i32) {
        debug_assert!((0..=32).contains(&n));
        if n == 0 {
            return;
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.put_bits(n, (val as u32) & mask);
    }

    /// Writes an unsigned LEB128 integer (`leb128()` in AV2 §4.11).
    pub fn put_uleb128(&mut self, mut val: u32) {
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            self.put_bits(8, byte as u32);
            if val == 0 {
                break;
            }
        }
    }

    /// Writes an unsigned LEB128 integer padded to a fixed `n_bytes` width (`1..=8`).
    pub fn put_uleb128_fixed(&mut self, mut val: u32, n_bytes: usize) {
        debug_assert!((1..=8).contains(&n_bytes));
        for i in 0..n_bytes {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if i + 1 < n_bytes {
                byte |= 0x80;
            }
            self.put_bits(8, byte as u32);
        }
    }

    /// Writes a variable-length unsigned integer (`uvlc()` in AV2 §4.11, inverse of `GetBits::get_vlc`).
    pub fn put_vlc(&mut self, val: u32) {
        if val == 0 {
            self.put_bit(1);
            return;
        }
        let v = val as u64 + 1;
        let n_bits = (63 ^ v.leading_zeros()) as i32;
        for _ in 0..n_bits {
            self.put_bit(0);
        }
        self.put_bit(1);
        let rem = (v - (1u64 << n_bits)) as u32;
        self.put_bits(n_bits, rem);
    }

    /// Writes a Rice/Golomb code of order `k` (inverse of `GetBits::get_golomb`).
    pub fn put_golomb(&mut self, val: u32, k: u32) {
        debug_assert!(k < 32);
        let q = val >> k;
        let rem = val & ((1u32 << k) - 1);
        for _ in 0..q {
            self.put_bit(1);
        }
        self.put_bit(0);
        self.put_bits(k as i32, rem);
    }

    /// Writes a non-symmetric uniform value in `0..max` (`ns(max)` in AV2 §4.11, inverse of `GetBits::get_uniform`).
    pub fn put_uniform(&mut self, max: u32, val: u32) {
        if max <= 1 {
            return;
        }
        debug_assert!(val < max);
        let l = ulog2(max) + 1;
        let m = (1u32 << l) - max;
        if val < m {
            self.put_bits(l - 1, val);
        } else {
            let v = val + m;
            self.put_bits(l - 1, v >> 1);
            self.put_bit(v & 1);
        }
    }

    /// Writes a reference-biased uniform value in `0..max` with default `def` (inverse of `GetBits::get_ref_uniform`).
    pub fn put_ref_uniform(&mut self, max: u32, def: u32, val: u32) {
        debug_assert!(val < max && def < max);
        if val == def {
            self.put_bit(0);
        } else {
            self.put_bit(1);
            let res = if val > def { val - 1 } else { val };
            self.put_uniform(max - 1, res);
        }
    }

    /// Writes an unsigned sub-exponentially coded value in `0..n` centered around `ref_val`
    /// (inverse of `GetBits::get_bits_subexp_u`).
    pub fn put_bits_subexp_u(&mut self, ref_val: u32, n: u32, k: i32, val: u32) {
        debug_assert!(val < n);
        let mut v = if ref_val * 2 <= n {
            recenter(ref_val, val)
        } else {
            recenter(n - 1 - ref_val, n - 1 - val)
        };

        let mut base = 0u32;
        let mut i = 0i32;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b;
            if n <= base + 3 * a {
                self.put_uniform(n - base, v);
                break;
            }
            if v < a {
                self.put_bit(0);
                self.put_bits(b, v);
                break;
            }
            self.put_bit(1);
            v -= a;
            base += a;
            i += 1;
        }
    }

    /// Writes a signed sub-exponentially coded value in `[-n+1, n-1]` centered around `ref_val`
    /// (inverse of `GetBits::get_bits_subexp`).
    pub fn put_bits_subexp(&mut self, ref_val: i32, n: u32, val: i32) {
        let off = n as i32 - 1;
        let n2 = n + off as u32;
        self.put_bits_subexp_u((ref_val + off) as u32, n2, 3, (val + off) as u32);
    }

    /// Pads with zero bits to the next byte boundary (`byte_alignment()` in AV2 §5.2).
    #[inline]
    pub fn bytealign(&mut self) {
        while self.bits_used != 0 {
            self.put_bit(0);
        }
    }

    /// Writes OBU trailing bits (`1` followed by `0`s to byte alignment, `trailing_bits()` in AV2 §5.3.4).
    #[inline]
    pub fn put_trailing_bits(&mut self) {
        self.put_bit(1);
        self.bytealign();
    }

    /// Appends raw bytes when already byte-aligned, or bit-by-bit if unaligned.
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        if self.bits_used == 0 {
            self.buf.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.put_bits(8, b as u32);
            }
        }
    }

    /// Flushes any partial byte (padding with zero bits) and returns the finished byte buffer.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.bytealign();
        self.buf
    }

    /// Borrows the completed byte slice (panics in debug mode if not byte-aligned).
    pub fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(self.bits_used, 0);
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_putbits_primitives_and_uleb128() {
        let mut pb = PutBits::new();
        pb.put_uleb128(300);
        let bytes = pb.into_bytes();
        assert_eq!(bytes, vec![0xAC, 0x02]);

        let mut pb2 = PutBits::new();
        pb2.put_bit(1);
        pb2.put_bits(5, 0b10110);
        pb2.put_bits(2, 0b01);
        assert_eq!(pb2.into_bytes(), vec![0b11011001]);
    }
}
