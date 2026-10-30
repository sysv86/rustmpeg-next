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

//! MSB-first bit reader for AV2 OBU payloads.
//!
//! Every method is the inverse of the corresponding [`crate::av2::putbits::PutBits`]
//! writer method, which follows AV2 Specification v1.0.0 §4.11.

use crate::av2::error::{Av2Error, Result};
use crate::av2::intops::ulog2;

/// Bounds-checked MSB-first bit reader.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Creates a reader positioned at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current absolute bit position.
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Total number of bits in the backing slice.
    #[inline]
    pub fn len_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Number of unread bits.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        self.len_bits().saturating_sub(self.pos)
    }

    /// Returns `true` once the reader has consumed every input bit.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.len_bits()
    }

    /// Reads a single bit (`f(1)`), returning an error at end of data.
    #[inline]
    pub fn get_bit(&mut self) -> Result<u32> {
        if self.pos >= self.len_bits() {
            return Err(Av2Error::BitstreamError(
                "unexpected end of bitstream".to_string(),
            ));
        }
        let byte = self.data[self.pos >> 3];
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit as u32)
    }

    /// Reads `n` (`0..=32`) unsigned bits MSB-first (`f(n)`).
    pub fn get_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.pos + n as usize > self.len_bits() {
            return Err(Av2Error::BitstreamError(
                "unexpected end of bitstream".to_string(),
            ));
        }
        let mut v = 0u32;
        let mut left = n;
        while left > 0 {
            let byte_idx = self.pos >> 3;
            let bit_off = self.pos & 7;
            let avail = 8 - bit_off;
            let take = avail.min(left as usize);
            let byte = self.data[byte_idx] as u32;
            let chunk = (byte >> (avail - take)) & ((1u32 << take) - 1);
            v = (v << take) | chunk;
            self.pos += take;
            left -= take as u32;
        }
        Ok(v)
    }

    /// Reads `n` signed bits (`su(n)`).
    #[inline]
    pub fn get_sbits(&mut self, n: u32) -> Result<i32> {
        let v = self.get_bits(n)?;
        if n == 0 {
            return Ok(0);
        }
        let sign = 1i32 << (n - 1);
        Ok(if (v & sign as u32) != 0 {
            v as i32 - (1i32 << n)
        } else {
            v as i32
        })
    }

    /// Reads an unsigned variable-length code (`uvlc()`), the inverse of
    /// [`crate::av2::putbits::PutBits::put_vlc`].
    pub fn get_vlc(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        while self.get_bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(Av2Error::BitstreamError(
                    "uvlc() prefix too long".to_string(),
                ));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let rem = self.get_bits(zeros)?;
        Ok((1u32 << zeros) + rem - 1)
    }

    /// Reads a non-symmetric uniform value in `0..max` (`ns(max)`), the inverse of
    /// [`crate::av2::putbits::PutBits::put_uniform`].
    pub fn get_uniform(&mut self, max: u32) -> Result<u32> {
        if max <= 1 {
            return Ok(0);
        }
        let l = (ulog2(max) + 1) as u32;
        let m = (1u32 << l) - max;
        let v = self.get_bits(l - 1)?;
        if v < m {
            Ok(v)
        } else {
            let extra = self.get_bit()?;
            Ok(((v << 1) | extra) - m)
        }
    }

    /// Reads an unsigned sub-exponentially coded value in `0..n` centred around
    /// `ref_val` (`get_bits_subexp_u`, AV2 §4.11 / dav2d `getbits.c`), the
    /// inverse of [`crate::av2::putbits::PutBits::put_bits_subexp_u`].
    pub fn get_bits_subexp_u(&mut self, ref_val: u32, n: u32, k: i32) -> Result<u32> {
        if n == 0 {
            return Err(Av2Error::BitstreamError(
                "sub-exponential value has an empty range".to_string(),
            ));
        }
        let mut v = 0u32;
        let mut i = 0i32;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            if !(0..=31).contains(&b) {
                return Err(Av2Error::BitstreamError(
                    "sub-exponential code length is out of range".to_string(),
                ));
            }
            let a = 1u32 << b;
            if n <= v + 3 * a {
                v += self.get_uniform(n - v)?;
                break;
            }
            if self.get_bit()? == 0 {
                v += self.get_bits(b as u32)?;
                break;
            }
            v += a;
            i += 1;
        }
        Ok(if ref_val * 2 <= n {
            crate::av2::intops::inv_recenter(ref_val, v)
        } else {
            n - 1 - crate::av2::intops::inv_recenter(n - 1 - ref_val, v)
        })
    }

    /// Reads a value in `0..max` biased towards `def` (`ns_ref(max)`), the inverse of
    /// [`crate::av2::putbits::PutBits::put_ref_uniform`].
    pub fn get_ref_uniform(&mut self, max: u32, def: u32) -> Result<u32> {
        if self.get_bit()? == 0 {
            Ok(def)
        } else {
            let v = self.get_uniform(max - 1)?;
            Ok(if v >= def { v + 1 } else { v })
        }
    }

    /// Advances to the next byte boundary (`byte_alignment()`).
    #[inline]
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// Reads `trailing_bits()`: a `1` bit followed by zero bits up to byte alignment.
    pub fn trailing_bits(&mut self) -> Result<()> {
        if self.get_bit()? != 1 {
            return Err(Av2Error::BitstreamError(
                "missing trailing_bits() marker".to_string(),
            ));
        }
        while (self.pos & 7) != 0 {
            if self.get_bit()? != 0 {
                return Err(Av2Error::BitstreamError(
                    "non-zero trailing_bits() padding".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Skips `n` bits.
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.len_bits() {
            return Err(Av2Error::BitstreamError(
                "unexpected end of bitstream".to_string(),
            ));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::putbits::PutBits;

    #[test]
    fn test_bits_vlc_uniform_roundtrip() {
        let mut pb = PutBits::new();
        pb.put_bits(3, 5);
        pb.put_sbits(5, -7);
        pb.put_vlc(0);
        pb.put_vlc(1);
        pb.put_vlc(37);
        pb.put_uniform(5, 3);
        pb.put_uniform(3, 1);
        pb.put_ref_uniform(8, 6, 2);
        pb.put_trailing_bits();
        let bytes = pb.into_bytes();

        let mut br = BitReader::new(&bytes);
        assert_eq!(br.get_bits(3).unwrap(), 5);
        assert_eq!(br.get_sbits(5).unwrap(), -7);
        assert_eq!(br.get_vlc().unwrap(), 0);
        assert_eq!(br.get_vlc().unwrap(), 1);
        assert_eq!(br.get_vlc().unwrap(), 37);
        assert_eq!(br.get_uniform(5).unwrap(), 3);
        assert_eq!(br.get_uniform(3).unwrap(), 1);
        assert_eq!(br.get_ref_uniform(8, 6).unwrap(), 2);
        br.trailing_bits().unwrap();
        assert!(br.is_exhausted());
    }

    #[test]
    fn test_subexp_u_roundtrip() {
        let mut pb = PutBits::new();
        let cases = [
            (0u32, 64u32, 3i32, 0u32),
            (5, 64, 3, 19),
            (31, 64, 3, 63),
            (0, 6, 3, 5),
            (3, 16, 1, 9),
        ];
        for &(r, n, k, v) in &cases {
            pb.put_bits_subexp_u(r, n, k, v);
        }
        pb.put_trailing_bits();
        let bytes = pb.into_bytes();
        let mut br = BitReader::new(&bytes);
        for &(r, n, k, v) in &cases {
            assert_eq!(br.get_bits_subexp_u(r, n, k).unwrap(), v);
        }
        br.trailing_bits().unwrap();
    }

    #[test]
    fn test_reader_rejects_truncated_input() {
        let mut br = BitReader::new(&[0u8]);
        assert_eq!(br.get_bits(8).unwrap(), 0);
        assert!(br.get_bit().is_err());
        assert!(br.get_bits(1).is_err());
    }
}
