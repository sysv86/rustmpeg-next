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

//! Normative AV2 Multi-Symbol Arithmetic Encoder (`MsacEncoder`) and RDO Fractional
//! Bit-Rate Estimator (in 1/256-bit `Q8` fixed-point units), implementing AV2 Specification
//! v1.0.0 §8.2 (`init_symbol`, `read_bool`, `exit_symbol`, `read_literal`, `read_symbol`) and
//! §5.20.7.28 (`read_quant`).

use crate::av2::intops::{clz, ulog2};

/// Normative probability increment floor table `Prob_Inc[7][8]` from AV2 Specification v1.0.0
/// §8.2.6 (`attachments/prob_inc.h`).
pub static PROB_INC: [[u16; 8]; 7] = [
    [8, 0, 0, 0, 0, 0, 0, 0],
    [10, 5, 0, 0, 0, 0, 0, 0],
    [12, 8, 4, 0, 0, 0, 0, 0],
    [12, 9, 6, 3, 0, 0, 0, 0],
    [13, 10, 8, 5, 2, 0, 0, 0],
    [13, 11, 9, 6, 4, 2, 0, 0],
    [14, 12, 10, 8, 6, 4, 2, 0],
];

/// Normative adaptation rate adjustment table `Para_Adjustment_List[125][3]` from AV2
/// Specification v1.0.0 §8.2.6 (`attachments/para_adjustment_list.h`).
pub static PARA_ADJUSTMENT_LIST: [[i8; 3]; 125] = [
    [0, 0, 0],
    [0, 0, -1],
    [0, 0, -2],
    [0, 0, 1],
    [0, 0, 1],
    [0, -1, 0],
    [0, -1, -1],
    [0, -1, -2],
    [0, -1, 1],
    [0, -1, 1],
    [0, -2, 0],
    [0, -2, -1],
    [0, -2, -2],
    [0, -2, 1],
    [0, -2, 1],
    [0, 1, 0],
    [0, 1, -1],
    [0, 1, -2],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, -1],
    [0, 1, -2],
    [0, 1, 1],
    [0, 1, 1],
    [-1, 0, 0],
    [-1, 0, -1],
    [-1, 0, -2],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, -1, 0],
    [-1, -1, -1],
    [-1, -1, -2],
    [-1, -1, 1],
    [-1, -1, 1],
    [-1, -2, 0],
    [-1, -2, -1],
    [-1, -2, -2],
    [-1, -2, 1],
    [-1, -2, 1],
    [-1, 1, 0],
    [-1, 1, -1],
    [-1, 1, -2],
    [-1, 1, 1],
    [-1, 1, 1],
    [-1, 1, 0],
    [-1, 1, -1],
    [-1, 1, -2],
    [-1, 1, 1],
    [-1, 1, 1],
    [-2, 0, 0],
    [-2, 0, -1],
    [-2, 0, -2],
    [-2, 0, 1],
    [-2, 0, 1],
    [-2, -1, 0],
    [-2, -1, -1],
    [-2, -1, -2],
    [-2, -1, 1],
    [-2, -1, 1],
    [-2, -2, 0],
    [-2, -2, -1],
    [-2, -2, -2],
    [-2, -2, 1],
    [-2, -2, 1],
    [-2, 1, 0],
    [-2, 1, -1],
    [-2, 1, -2],
    [-2, 1, 1],
    [-2, 1, 1],
    [-2, 1, 0],
    [-2, 1, -1],
    [-2, 1, -2],
    [-2, 1, 1],
    [-2, 1, 1],
    [1, 0, 0],
    [1, 0, -1],
    [1, 0, -2],
    [1, 0, 1],
    [1, 0, 1],
    [1, -1, 0],
    [1, -1, -1],
    [1, -1, -2],
    [1, -1, 1],
    [1, -1, 1],
    [1, -2, 0],
    [1, -2, -1],
    [1, -2, -2],
    [1, -2, 1],
    [1, -2, 1],
    [1, 1, 0],
    [1, 1, -1],
    [1, 1, -2],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
    [1, 1, -1],
    [1, 1, -2],
    [1, 1, 1],
    [1, 1, 1],
    [1, 0, 0],
    [1, 0, -1],
    [1, 0, -2],
    [1, 0, 1],
    [1, 0, 1],
    [1, -1, 0],
    [1, -1, -1],
    [1, -1, -2],
    [1, -1, 1],
    [1, -1, 1],
    [1, -2, 0],
    [1, -2, -1],
    [1, -2, -2],
    [1, -2, 1],
    [1, -2, 1],
    [1, 1, 0],
    [1, 1, -1],
    [1, 1, -2],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 0],
    [1, 1, -1],
    [1, 1, -2],
    [1, 1, 1],
    [1, 1, 1],
];

/// Lightweight snapshot of `MsacEncoder` state for RDO trial encoding and rollback.
#[derive(Debug, Clone)]
pub struct MsacSnapshot {
    pub low: u64,
    pub rng: u32,
    pub bits_len: usize,
}

/// Pure-Rust AV2 Multi-Symbol Arithmetic Encoder (`MsacEncoder`).
///
/// Tracks `low` directly in the AV2 Spec §8.2 `SymbolValue` coordinate system (`[0, 0x8000)`)
/// and inverts emitted bits (`bit ^ 1`) upon byte-packing in [`MsacEncoder::finalize`],
/// guaranteeing a 100% bit-exact inverse of `init_symbol`, `read_symbol`, `read_bool`, and
/// `exit_symbol` in AV2 Specification v1.0.0 §8.2.
#[derive(Debug, Clone)]
pub struct MsacEncoder {
    /// Current lower bound in `SymbolValue` coordinates (`0 <= low < 0x8000`).
    pub low: u64,
    /// Current interval width (`0x8000 <= rng <= 0xFFFF`).
    pub rng: u32,
    /// Uncommitted bits (`0` or `1` in `SymbolValue` space) buffered to support carry propagation.
    pub bits: Vec<u8>,
    /// Whether CDF adaptation (`disable_cdf_update == 0`) is enabled.
    pub allow_update_cdf: bool,
}

impl Default for MsacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MsacEncoder {
    /// Creates a new `MsacEncoder` with initial interval `[0, 0x8000)`.
    pub fn new() -> Self {
        Self {
            low: 0,
            rng: 0x8000,
            bits: Vec::with_capacity(8192),
            allow_update_cdf: true,
        }
    }

    /// Creates a dry-run `MsacEncoder` for RDO bit-counting.
    pub fn new_rdo_counter(allow_update_cdf: bool) -> Self {
        Self {
            low: 0,
            rng: 0x8000,
            bits: Vec::new(),
            allow_update_cdf,
        }
    }

    /// Takes a snapshot of the current arithmetic coder state.
    #[inline]
    pub fn snapshot(&self) -> MsacSnapshot {
        MsacSnapshot {
            low: self.low,
            rng: self.rng,
            bits_len: self.bits.len(),
        }
    }

    /// Restores the arithmetic coder state to a previous [`MsacSnapshot`].
    #[inline]
    pub fn restore(&mut self, snap: &MsacSnapshot) {
        self.low = snap.low;
        self.rng = snap.rng;
        self.bits.truncate(snap.bits_len);
    }

    /// Returns the number of bits emitted so far (for RDO rate measurement).
    #[inline]
    pub fn tell_bits(&self) -> usize {
        self.bits.len()
    }

    /// Propagates an integer carry `c` backward through previously emitted bits.
    #[inline]
    fn propagate_carry(&mut self, mut c: u8) {
        let mut idx = self.bits.len();
        while c > 0 && idx > 0 {
            idx -= 1;
            let s = self.bits[idx] + c;
            self.bits[idx] = s & 1;
            c = s >> 1;
        }
    }

    /// Shifts `d` bits out of `self.low >> 15` into `self.bits`, propagating any carries.
    #[inline]
    fn emit_shifted_bits(&mut self, d: u32) {
        for _ in 0..d {
            self.low <<= 1;
            let bit = (self.low >> 15) as u8;
            self.low &= 0x7FFF;
            if bit > 1 {
                self.propagate_carry(bit >> 1);
            }
            self.bits.push(bit & 1);
        }
    }

    /// Estimates the fractional bit cost (in `Q8`, i.e., `1/256` bit units) of coding `symbol`
    /// with the given adaptive `cdf` without mutating state.
    #[inline]
    pub fn estimate_symbol_bits_q8(symbol: usize, cdf: &[u16], n_symbols: usize) -> u32 {
        let icdf_prev = if symbol > 0 {
            32768u32.saturating_sub(cdf[symbol - 1] as u32)
        } else {
            32768u32
        };
        let icdf_cur = if symbol < n_symbols {
            32768u32.saturating_sub(cdf[symbol] as u32)
        } else {
            0u32
        };
        let p = icdf_prev.saturating_sub(icdf_cur).max(1);
        let lg = ulog2(p).max(0) as u32;
        let frac = ((p - (1u32 << lg)) << 8) >> lg;
        ((15u32 << 8).saturating_sub((lg << 8) + frac)).max(32)
    }

    /// Estimates the fractional bit cost (in `Q8`, i.e., `1/256` bit units) of coding a boolean
    /// `bit` (`0` or `1`) against `cdf0` without mutating state.
    #[inline]
    pub fn estimate_bool_bits_q8(bit: u32, cdf0: u16) -> u32 {
        let p0 = (cdf0 as u32).clamp(1, 32767);
        let p = if bit != 0 { 32768 - p0 } else { p0 };
        let lg = ulog2(p).max(0) as u32;
        let frac = ((p - (1u32 << lg)) << 8) >> lg;
        ((15u32 << 8).saturating_sub((lg << 8) + frac)).max(32)
    }

    /// Computes the normative range split point `cur` for `symbol` in an alphabet of size
    /// `N = n_symbols + 1` per AV2 Specification v1.0.0 §8.2.6:
    /// ```text
    /// f = (symbol == N - 1) ? 0 : (1 << 15) - cdf[symbol]
    /// pp = ((f >> 7) << 4) + Prob_Inc[N - 2][symbol]
    /// cur = (((SymbolRange >> 8) * pp) >> 7) << 3
    /// ```
    #[inline]
    pub(crate) fn compute_cur(rng: u32, cdf: &[u16], symbol: usize, n_symbols: usize) -> u32 {
        if symbol >= n_symbols {
            return 0;
        }
        let f = 32768u32.saturating_sub(cdf[symbol] as u32);
        let inc = if n_symbols <= 7 {
            PROB_INC[n_symbols - 1][symbol] as u32
        } else {
            ((n_symbols - symbol) as u32) << 1
        };
        let pp = ((f >> 7) << 4) + inc;
        (((rng >> 8) * pp) >> 7) << 3
    }

    /// Updates `cdf` after coding `symbol` in `0..=n_symbols` (`N = n_symbols + 1`) per
    /// AV2 Specification v1.0.0 §8.2.6.
    #[inline]
    pub(crate) fn update_cdf(cdf: &mut [u16], symbol: usize, n_symbols: usize) {
        let (para, count, separate_count) = if cdf.len() >= n_symbols + 2 && cdf[n_symbols] <= 124 {
            (cdf[n_symbols] as usize, cdf[n_symbols + 1] as usize, true)
        } else {
            let meta = cdf[n_symbols] as usize;
            ((meta >> 8).min(124), meta & 0xFF, false)
        };
        let time_interval = if count > 31 {
            2usize
        } else if count > 15 {
            1usize
        } else {
            0usize
        };
        let rate = (4
            + (time_interval as i32)
            + if n_symbols > 2 { 1 } else { 0 }
            + (PARA_ADJUSTMENT_LIST[para][time_interval] as i32)) as u32;
        if n_symbols == 1 {
            let c0 = cdf[0];
            cdf[0] = if symbol == 0 {
                c0 + ((32768u16 - c0) >> rate)
            } else {
                c0 - (c0 >> rate)
            };
        } else {
            let sym_clamped = symbol.min(n_symbols);
            for c in &mut cdf[..sym_clamped] {
                *c -= *c >> rate;
            }
            for c in &mut cdf[sym_clamped..n_symbols] {
                *c += (32768u16 - *c) >> rate;
            }
        }
        if separate_count {
            if count < 32 {
                cdf[n_symbols + 1] = (count + 1) as u16;
            }
        } else if count < 32 {
            cdf[n_symbols] += 1;
        }
    }

    /// Encodes a symbol in `0..=n_symbols` (`N = n_symbols + 1`) using a forward 15-bit CDF table
    /// and updates the CDF per AV2 Specification v1.0.0 §8.2.6 (`read_symbol`).
    pub fn encode_symbol_adapt(&mut self, symbol: usize, cdf: &mut [u16], n_symbols: usize) {
        let sym = symbol.min(n_symbols);
        let prev = if sym == 0 {
            self.rng
        } else {
            Self::compute_cur(self.rng, cdf, sym - 1, n_symbols)
        };
        let cur = Self::compute_cur(self.rng, cdf, sym, n_symbols);
        let new_rng = prev.saturating_sub(cur).max(8);
        self.low += cur as u64;
        let bits = 15u32.saturating_sub(31 ^ clz(new_rng));
        self.rng = new_rng << bits;
        self.emit_shifted_bits(bits);

        if self.allow_update_cdf {
            Self::update_cdf(cdf, sym, n_symbols);
        }
    }

    /// Encodes a boolean symbol (`0` or `1`) with raw forward probability `cdf0` (without CDF adaptation).
    pub fn encode_bool_raw(&mut self, bit: u32, cdf0: u16) {
        let f = 32768u32.saturating_sub(cdf0 as u32);
        let pp = ((f >> 7) << 4) + 8;
        let cur = (((self.rng >> 8) * pp) >> 7) << 3;
        let (low_inc, new_rng) = if bit != 0 {
            (0u64, cur.max(8))
        } else {
            (cur as u64, self.rng.saturating_sub(cur).max(8))
        };
        self.low += low_inc;
        let bits = 15u32.saturating_sub(31 ^ clz(new_rng));
        self.rng = new_rng << bits;
        self.emit_shifted_bits(bits);
    }

    /// Encodes an equiprobable bypass bit (`0` or `1`) per AV2 Specification v1.0.0 §8.2.3 (`read_bool`).
    ///
    /// Note: Per §8.2.3, `SymbolRange` (`self.rng`) is NOT modified during `read_bool`;
    /// `cur = SymbolRange >> 1`, `SymbolValue` is reduced by `cur` when `symbol == 0`, and
    /// 1 bit is shifted into `SymbolValue`.
    pub fn encode_bool_bypass(&mut self, bit: u32) {
        let cur = self.rng >> 1;
        if bit == 0 {
            self.low += cur as u64;
        }
        self.emit_shifted_bits(1);
    }

    /// Encodes an adaptive boolean symbol (`0` or `1`) and updates `cdf` per AV2 Spec §8.2.6.
    pub fn encode_bool_adapt(&mut self, bit: u32, cdf: &mut [u16]) {
        self.encode_symbol_adapt((bit != 0) as usize, cdf, 1);
    }

    /// Encodes `n` equiprobable bypass bits from `val` (MSB first), matching AV2 Spec §8.2.5 (`read_literal(n)`).
    pub fn encode_bools_bypass(&mut self, n: u32, val: u32) {
        for i in (0..n).rev() {
            self.encode_bool_bypass((val >> i) & 1);
        }
    }

    /// Encodes `val` (`0..=max_len`) using truncated unary bypass coding.
    pub fn encode_unary_bypass(&mut self, val: u32, max_len: u32) {
        debug_assert!(val <= max_len);
        for _ in 0..val {
            self.encode_bool_bypass(1);
        }
        if val < max_len {
            self.encode_bool_bypass(0);
        }
    }

    /// Encodes `val` in `0..n` using AV2 non-symmetric uniform bypass coding (`NS(n)`).
    pub fn encode_uniform(&mut self, n: u32, val: u32) {
        if n <= 1 {
            return;
        }
        let val = val.min(n - 1);
        let w = (ulog2(n) as u32) + 1;
        let m = (1u32 << w) - n;
        if val < m {
            self.encode_bools_bypass(w - 1, val);
        } else {
            let v = val + m;
            self.encode_bools_bypass(w - 1, v >> 1);
            self.encode_bool_bypass(v & 1);
        }
    }

    /// Encodes `val` using order-0 Exp-Golomb bypass coding.
    pub fn encode_exp_golomb(&mut self, val: u32) {
        let v_plus_1 = val + 1;
        let l = ulog2(v_plus_1).max(0) as u32;
        for _ in 0..l {
            self.encode_bool_bypass(0);
        }
        self.encode_bool_bypass(1);
        if l > 0 {
            let rem = v_plus_1 - (1u32 << l);
            self.encode_bools_bypass(l, rem);
        }
    }

    /// Encodes a high-range quantized coefficient residual `x = quant - max_level` and updates
    /// `hr_level_avg` per AV2 Specification v1.0.0 §5.20.7.28 (`read_quant`).
    pub fn encode_quant_hr(&mut self, x: u32, lvl_shift: u32, hr_level_avg: &mut u32) {
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

        if x < unary_limit {
            let q = x >> m;
            for _ in 0..q {
                self.encode_bool_bypass(0);
            }
            self.encode_bool_bypass(1);
            let rem = x & ((1u32 << m) - 1);
            self.encode_bools_bypass(m, rem);
        } else {
            for _ in 0..c_max {
                self.encode_bool_bypass(0);
            }
            let excess = x - unary_limit + (1u32 << k);
            let length = ulog2(excess).max(k as i32) as u32;
            let golomb_zeros = length - k;
            for _ in 0..golomb_zeros {
                self.encode_bool_bypass(0);
            }
            self.encode_bool_bypass(1);
            let x_base = unary_limit + (1u32 << length) - (1u32 << k);
            let coeff_rem = x - x_base;
            self.encode_bools_bypass(length, coeff_rem);
        }

        *hr_level_avg = ((x << lvl_shift) + *hr_level_avg) >> 1;
    }

    /// Legacy single-parameter wrapper around [`MsacEncoder::encode_quant_hr`] for `read_quant`.
    pub fn encode_hr(&mut self, val: u32, k: u32) {
        let mut avg = (1u32 << k.clamp(1, 6)) >> 1;
        self.encode_quant_hr(val, 0, &mut avg);
    }

    /// Estimates the bit cost (in `Q8`, i.e., `1/256` bit units) of `encode_quant_hr`.
    #[inline]
    pub fn estimate_hr_bits_q8(val: u32, k: u32) -> u32 {
        let m = k.clamp(1, 6);
        let c_max = (m + 4).min(6);
        let limit = c_max << m;
        let n_bits = if val < limit {
            (val >> m) + 1 + m
        } else {
            let excess = val - limit + (1u32 << (m + 1));
            let length = ulog2(excess).max((m + 1) as i32) as u32;
            c_max + (length - (m + 1)) + 1 + length
        };
        n_bits << 8
    }

    /// Finalizes the arithmetic bitstream and emits the normative AV2 §8.2.4 trailing bit (`1`
    /// at `trailingBitPosition = M`, followed by `0` padding bits to byte alignment).
    ///
    /// Because `self.rng >= 0x8000` (`32768`) after every renormalization step, the interval
    /// `[self.low, self.low + self.rng)` is guaranteed to contain an integer whose low 15 bits
    /// in `SymbolValue` space equal `0x3FFF` (`0b0111_1111_1111_111`, which inverts to
    /// `0b1000_0000_0000_000` in the bitstream). When `self.low > 0x3FFF`, choosing `0x8000 + 0x3FFF`
    /// propagates a carry of `1` into `self.bits`.
    pub fn finalize(mut self) -> Vec<u8> {
        if self.low > 0x3FFF {
            self.propagate_carry(1);
        }
        let total_bits = self.bits.len() + 1;
        let num_bytes = (total_bits + 7) >> 3;
        let mut out = vec![0u8; num_bytes];
        for (i, &b) in self.bits.iter().enumerate() {
            let inv_bit = (b ^ 1) & 1;
            out[i >> 3] |= inv_bit << (7 - (i & 7));
        }
        // Set the normative trailing 1 bit at bit position M = self.bits.len()
        let m = self.bits.len();
        out[m >> 3] |= 1u8 << (7 - (m & 7));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msac_encoder_snapshot_and_rate_estimation() {
        let mut enc = MsacEncoder::new();
        let mut cdf = [16384u16, 0, 0];
        let snap = enc.snapshot();
        enc.encode_bool_adapt(1, &mut cdf);
        assert!(enc.tell_bits() >= snap.bits_len);
        enc.restore(&snap);
        assert_eq!(enc.tell_bits(), snap.bits_len);

        let cost_equi = MsacEncoder::estimate_bool_bits_q8(0, 16384);
        assert!((cost_equi as i32 - 256).abs() <= 4);
    }
}
