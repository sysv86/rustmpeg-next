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

//! Multi-symbol arithmetic decoder (MSAC) for AV2 tile payloads.
//!
//! The decoder is the exact inverse of [`crate::av2::msac::MsacEncoder`] and
//! follows AV2 Specification v1.0.0 §8.2. The encoder stores every emitted bit
//! inverted and terminates the stream with a single `1` bit; this reader
//! inverts the bits back into the normative `SymbolValue` domain and consumes
//! the implied `1` padding past the end of the tile payload.

use crate::av2::error::{Av2Error, Result};
use crate::av2::intops::ulog2;
use crate::av2::msac::MsacEncoder;

/// Number of code-value bits held in the arithmetic window.
const WINDOW_BITS: u32 = 15;

/// AV2 multi-symbol arithmetic decoder.
#[derive(Debug, Clone)]
pub struct MsacDecoder<'a> {
    data: &'a [u8],
    /// Absolute bit index into `data`.
    pos: usize,
    /// Current value inside `[0, rng)`.
    value: u32,
    /// Current interval width (`0x8000..=0xFFFF`).
    rng: u32,
    /// `true` when CDF adaptation is enabled (`disable_cdf_update == 0`).
    allow_update_cdf: bool,
    /// Number of symbols decoded so far (diagnostics).
    symbols: usize,
}

impl<'a> MsacDecoder<'a> {
    /// Creates a decoder over a tile payload of `data`.
    pub fn new(data: &'a [u8], allow_update_cdf: bool) -> Self {
        let mut dec = Self {
            data,
            pos: 0,
            value: 0,
            rng: 0x8000,
            allow_update_cdf,
            symbols: 0,
        };
        let mut value = 0u32;
        for _ in 0..WINDOW_BITS {
            value = (value << 1) | dec.read_symbol_bit();
        }
        dec.value = value;
        dec
    }

    /// Returns the number of symbols decoded through this decoder.
    #[inline]
    pub fn symbols_decoded(&self) -> usize {
        self.symbols
    }

    /// Current interval width (`0x8000..=0xFFFF`).
    #[inline]
    pub fn rng(&self) -> u32 {
        self.rng
    }

    /// Current bit position.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Current window value.
    #[inline]
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Reads one stream bit in the `SymbolValue` domain.
    ///
    /// Stored bits are inverted and the stream is logically zero-extended, so a
    /// bit past the end of the payload reads as `1`.
    #[inline]
    fn read_symbol_bit(&mut self) -> u32 {
        if self.pos >= self.data.len() * 8 {
            self.pos += 1;
            return 1;
        }
        let byte = self.data[self.pos >> 3];
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        (bit ^ 1) as u32
    }

    /// Reads `n` (`0..=31`) stream bits in the `SymbolValue` domain, MSB first.
    #[inline]
    fn read_symbol_bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_symbol_bit();
        }
        v
    }

    /// Renormalizes the interval back to `[0x8000, 0xFFFF]`.
    #[inline]
    fn renormalize(&mut self, mut value: u32, mut rng: u32) {
        let bits = (WINDOW_BITS as i32 - ulog2(rng)) as u32;
        if bits > 0 {
            value = (value << bits) | self.read_symbol_bits(bits);
            rng <<= bits;
        }
        self.value = value;
        self.rng = rng;
    }

    /// Decodes an adaptive multi-symbol value in `0..=n_symbols` and updates `cdf`.
    pub fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> Result<usize> {
        let c = self.value;
        let mut prev = self.rng;
        let mut sym = 0usize;
        let mut cur;
        loop {
            if sym > n_symbols {
                return Err(Av2Error::EntropyError(
                    "CDF symbol search left the alphabet".to_string(),
                ));
            }
            cur = MsacEncoder::compute_cur(self.rng, cdf, sym, n_symbols);
            if c >= cur {
                break;
            }
            prev = cur;
            sym += 1;
        }
        let new_value = c - cur;
        let new_rng = prev - cur;
        debug_assert!(new_rng > 0);
        self.renormalize(new_value, new_rng);

        if self.allow_update_cdf {
            MsacEncoder::update_cdf(cdf, sym, n_symbols);
        }
        self.symbols += 1;
        Ok(sym)
    }

    /// Decodes an adaptive boolean and updates its two-entry CDF.
    #[inline]
    pub fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> Result<u32> {
        let c = self.value;
        let rng = self.rng;
        let f = 32768u32.saturating_sub(cdf[0] as u32);
        let pp = ((f >> 7) << 4) + 8;
        let cur = (((rng >> 8) * pp) >> 7) << 3;
        let (sym, new_value, new_rng) = if c >= cur {
            (0usize, c - cur, rng - cur)
        } else {
            (1usize, c, cur)
        };
        self.renormalize(new_value, new_rng);
        if self.allow_update_cdf {
            MsacEncoder::update_cdf(cdf, sym, 1);
        }
        self.symbols += 1;
        Ok(sym as u32)
    }

    /// Decodes one equiprobable bypass bit (§8.2.3).
    #[inline]
    pub fn decode_bool_bypass(&mut self) -> u32 {
        let cur = self.rng >> 1;
        let bit = if self.value >= cur {
            self.value -= cur;
            0
        } else {
            1
        };
        self.value = (self.value << 1) | self.read_symbol_bit();
        self.symbols += 1;
        bit
    }

    /// Decodes `n` (`0..=32`) equiprobable bypass bits, MSB first.
    pub fn decode_bools_bypass(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.decode_bool_bypass();
        }
        v
    }

    /// Decodes a value in `0..n` using non-symmetric uniform bypass coding (`NS(n)`).
    pub fn decode_uniform(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = (ulog2(n) + 1) as u32;
        let m = (1u32 << w) - n;
        let v = self.decode_bools_bypass(w - 1);
        if v < m {
            v
        } else {
            let extra = self.decode_bool_bypass();
            ((v << 1) | extra) - m
        }
    }

    /// Decodes a value in `0..=max_len` coded with truncated unary bypass coding.
    pub fn decode_unary_bypass(&mut self, max_len: u32) -> u32 {
        let mut v = 0;
        while v < max_len && self.decode_bool_bypass() != 0 {
            v += 1;
        }
        v
    }

    /// Decodes an order-0 Exp-Golomb bypass value.
    pub fn decode_exp_golomb(&mut self) -> u32 {
        let mut zeros = 0u32;
        while self.decode_bool_bypass() == 0 {
            zeros += 1;
            if zeros > 31 {
                return u32::MAX;
            }
        }
        if zeros == 0 {
            return 0;
        }
        let rem = self.decode_bools_bypass(zeros);
        (1u32 << zeros) + rem - 1
    }

    /// Decodes a high-range coefficient residual (`read_quant`) and updates `hr_level_avg`.
    ///
    /// This is the exact inverse of
    /// [`crate::av2::msac::MsacEncoder::encode_quant_hr`].
    pub fn decode_quant_hr(&mut self, lvl_shift: u32, hr_level_avg: &mut u32) -> u32 {
        let pred_level = *hr_level_avg >> lvl_shift;
        let msb = if pred_level == 0 {
            0i32
        } else {
            ulog2(pred_level)
        };
        let m = msb.clamp(1, 6) as u32;
        let k = m + 1;
        let c_max = (m + 4).min(6);
        let unary_limit = c_max << m;

        let mut zeros = 0u32;
        while zeros < c_max && self.decode_bool_bypass() == 0 {
            zeros += 1;
        }

        let x = if zeros < c_max {
            let rem = self.decode_bools_bypass(m);
            (zeros << m) | rem
        } else {
            let mut golomb_zeros = 0u32;
            while self.decode_bool_bypass() == 0 {
                golomb_zeros += 1;
                if golomb_zeros > 31 {
                    return 0;
                }
            }
            let length = k + golomb_zeros;
            let x_base = unary_limit + (1u32 << length) - (1u32 << k);
            let coeff_rem = self.decode_bools_bypass(length);
            x_base + coeff_rem
        };

        *hr_level_avg = ((x << lvl_shift) + *hr_level_avg) >> 1;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::cdf::CdfContext;
    use crate::av2::msac::MsacEncoder;

    fn encode_symbols(cdf_init: &[u16], syms: &[usize], n: usize) -> Vec<u8> {
        let mut enc = MsacEncoder::new();
        let mut cdf = cdf_init.to_vec();
        for &s in syms {
            enc.encode_symbol_adapt(s, &mut cdf, n);
        }
        enc.finalize()
    }

    #[test]
    fn test_symbol_roundtrip_with_adaptation() {
        let mut ctx = CdfContext::new(100);
        let mut cdf_init = *ctx.m.y_mode_set();
        let syms: Vec<usize> = (0..500).map(|i| (i * 7 + 3) % 4).collect();
        let bytes = encode_symbols(&cdf_init, &syms, 3);

        let mut dec = MsacDecoder::new(&bytes, true);
        let mut cdf = cdf_init;
        for &s in &syms {
            assert_eq!(dec.decode_symbol_adapt(&mut cdf, 3).unwrap(), s);
        }
        // CDFs must stay in lockstep with the encoder.
        let mut enc = MsacEncoder::new();
        for &s in &syms {
            enc.encode_symbol_adapt(s, &mut cdf_init, 3);
        }
        assert_eq!(cdf, cdf_init);
    }

    #[test]
    fn test_bool_and_bypass_roundtrip() {
        let mut enc = MsacEncoder::new();
        let mut cdf = [16384u16, 0, 0];
        let mut expect = Vec::new();
        for i in 0..300 {
            let b = (i * 5 + 1) % 2;
            enc.encode_bool_adapt(b, &mut cdf);
            expect.push(b);
        }
        let mut cdf_dec = [16384u16, 0, 0];
        let bytes = enc.finalize();
        let mut dec = MsacDecoder::new(&bytes, true);
        for &b in &expect {
            assert_eq!(dec.decode_bool_adapt(&mut cdf_dec).unwrap(), b);
        }
        assert_eq!(cdf, cdf_dec);
    }

    #[test]
    fn test_bypass_uniform_and_hr_roundtrip() {
        let mut enc = MsacEncoder::new();
        let mut hr_avg = 0u32;
        let mut hr_expect = Vec::new();
        for i in 0..64u32 {
            enc.encode_bools_bypass(7, (i * 37) & 0x7F);
            enc.encode_uniform(13, (i * 5) % 13);
            let val = (i * i) % 200;
            enc.encode_quant_hr(val, 0, &mut hr_avg);
            hr_expect.push(val);
        }
        let bytes = enc.finalize();
        let mut dec = MsacDecoder::new(&bytes, true);
        let mut hr_avg_dec = 0u32;
        for i in 0..64u32 {
            assert_eq!(dec.decode_bools_bypass(7), (i * 37) & 0x7F);
            assert_eq!(dec.decode_uniform(13), (i * 5) % 13);
            assert_eq!(
                dec.decode_quant_hr(0, &mut hr_avg_dec),
                hr_expect[i as usize]
            );
        }
    }

    #[test]
    fn test_empty_and_padded_payloads_stay_in_range() {
        for payload in [vec![], vec![0u8], vec![0xFFu8], vec![0x00, 0x22]] {
            let mut dec = MsacDecoder::new(&payload, true);
            let mut cdf = [16384u16, 0, 0];
            for _ in 0..16 {
                let _ = dec.decode_bool_adapt(&mut cdf).unwrap();
                assert!(dec.rng() >= 0x8000);
            }
        }
    }
}
