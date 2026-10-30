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

//! Integer and bitwise operations for AV2 encoding and decoding.

#[inline(always)]
pub fn imax(a: i32, b: i32) -> i32 {
    a.max(b)
}

#[inline(always)]
pub fn imin(a: i32, b: i32) -> i32 {
    a.min(b)
}

#[inline(always)]
pub fn umax(a: u32, b: u32) -> u32 {
    a.max(b)
}

#[inline(always)]
pub fn umin(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[inline(always)]
pub fn iclip(v: i32, min: i32, max: i32) -> i32 {
    v.clamp(min, max)
}

#[inline(always)]
pub fn iclip64to32(v: i64, min: i32, max: i32) -> i32 {
    if v < min as i64 {
        min
    } else if v > max as i64 {
        max
    } else {
        v as i32
    }
}

#[inline(always)]
pub fn iclip_u8(v: i32) -> i32 {
    iclip(v, 0, 255)
}

#[inline(always)]
pub fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 { -v } else { v }
}

#[inline(always)]
pub fn apply_sign64(v: i64, s: i64) -> i32 {
    if s < 0 { -(v as i32) } else { v as i32 }
}

#[inline(always)]
pub fn clz(v: u32) -> u32 {
    v.leading_zeros()
}

#[inline(always)]
pub fn clzll(v: u64) -> u32 {
    v.leading_zeros()
}

#[inline(always)]
pub fn ulog2(v: u32) -> i32 {
    debug_assert!(v > 0);
    31 ^ clz(v) as i32
}

/// `Div_Lut` from AV2 Specification v1.0.0 §7.13.3.22 (`DIV_LUT_NUM = 129`,
/// `DIV_LUT_BITS = 7`, `DIV_LUT_PREC_BITS = 9`).
pub static DIV_LUT: [u16; 129] = [
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462, 458, 455, 452, 449,
    446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417, 415, 412, 410, 407, 405, 402, 400, 397,
    395, 392, 390, 388, 386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356,
    354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324, 323,
    321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295,
    294, 293, 291, 290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274, 273, 272,
    271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256,
];

/// `resolve_divisor(d)` (AV2 Spec v1.0.0 §7.13.3.22): returns `(divShift, divFactor)`.
pub fn resolve_divisor(d: i32) -> (u32, i32) {
    let ad = d.unsigned_abs();
    let n = ulog2(ad);
    let e = ad - (1 << n);
    let f = if n > 7 {
        ((e + (1 << (n - 7 - 1))) >> (n - 7)) as usize
    } else {
        (e << (7 - n)) as usize
    };
    let div_shift = (n + 9) as u32;
    let div_factor = DIV_LUT[f.min(DIV_LUT.len() - 1)] as i32;
    (div_shift, if d < 0 { -div_factor } else { div_factor })
}

/// `approx_divide(num, den)` (AV2 Spec v1.0.0 §7.13.2.10): `Round2(num * divFactor, divShift)`.
///
/// The result is an approximation of `num / den`, so callers must clip it.
pub fn approx_divide(num: i32, den: i32) -> i32 {
    let (shift, factor) = resolve_divisor(den);
    let scaled = num as i64 * factor as i64;
    ((scaled + (1i64 << (shift - 1))) >> shift) as i32
}

#[inline(always)]
pub fn u64log2(v: u64) -> i32 {
    debug_assert!(v > 0);
    63 ^ clzll(v) as i32
}

#[inline(always)]
pub fn ctz(v: u32) -> u32 {
    v.trailing_zeros()
}

#[inline(always)]
pub fn popcnt(x: u32) -> u32 {
    x.count_ones()
}

/// Inverse recenter around `r` (AV2 §4.10 `inv_recenter`).
#[inline(always)]
pub fn inv_recenter(r: u32, v: u32) -> u32 {
    if v > r << 1 {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

/// Forward recenter of non-negative integer `v` around `r` (exact inverse of `inv_recenter`).
#[inline(always)]
pub fn recenter(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

/// Decode-side negative deinterleave for segment IDs and deltas (`neg_deinterleave` in AV2).
#[inline(always)]
pub fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        diff
    } else if r >= max - 1 {
        max - diff - 1
    } else if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                r + ((diff + 1) >> 1)
            } else {
                r - (diff >> 1)
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - r - 1) {
        if diff & 1 != 0 {
            r + ((diff + 1) >> 1)
        } else {
            r - (diff >> 1)
        }
    } else {
        max - (diff + 1)
    }
}

/// Encode-side interleave for segment IDs and deltas (exact inverse of `neg_deinterleave`).
#[inline(always)]
pub fn deinterleave(val: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        val
    } else if r >= max - 1 {
        max - val - 1
    } else if 2 * r < max {
        if val > 2 * r {
            val
        } else if val >= r {
            ((val - r) << 1) - (val > r) as i32
        } else {
            (r - val) << 1
        }
    } else if val < 2 * r - max + 1 {
        max - val - 1
    } else if val >= r {
        ((val - r) << 1) - (val > r) as i32
    } else {
        (r - val) << 1
    }
}

/// Smallest `k >= 0` such that `(sz << k) >= tgt`.
#[inline(always)]
pub fn tile_log2(sz: i32, tgt: i32) -> i32 {
    let mut k = 0;
    while (sz << k) < tgt {
        k += 1;
    }
    k
}

/// Signed modular POC difference on `order_hint_n_bits` bits.
#[inline(always)]
pub fn get_poc_diff(order_hint_n_bits: i32, poc0: i32, poc1: i32) -> i32 {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1 << (order_hint_n_bits - 1);
    let diff = poc0 - poc1;
    (diff & (mask - 1)) - (diff & mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recenter_roundtrip() {
        for r in 0..32 {
            for v in 0..64 {
                let enc = recenter(r, v);
                let dec = inv_recenter(r, enc);
                assert_eq!(dec, v, "r={r}, v={v}");
            }
        }
    }

    #[test]
    fn test_deinterleave_roundtrip() {
        for max in 1..=16 {
            for r in 0..max {
                for v in 0..max {
                    let diff = deinterleave(v, r, max);
                    let rec = neg_deinterleave(diff, r, max);
                    assert_eq!(rec, v, "max={max}, r={r}, v={v}, diff={diff}");
                }
            }
        }
    }

    #[test]
    fn test_resolve_divisor_matches_the_specification() {
        // Power-of-two divisors reduce to an exact scaling by 512.
        assert_eq!(resolve_divisor(16), (4 + 9, 512));
        assert_eq!(resolve_divisor(64), (6 + 9, 512));
        assert_eq!(DIV_LUT[0], 512);
        assert_eq!(DIV_LUT[128], 256);
        // `24 = 16 + 8` maps to `f = 8 << 3 = 64` (AV2 Spec v1.0.0 §7.13.3.22).
        assert_eq!(resolve_divisor(24), (4 + 9, 341));
    }

    #[test]
    fn test_approx_divide_is_not_a_rounded_division() {
        // `approx_divide` is a lookup-based approximation, so it can differ from
        // the exact rounded division by one. DC prediction must follow it.
        assert_eq!(approx_divide(5076, 24), 211);
        assert_eq!((5076 + 12) / 24, 212);
        assert_eq!(approx_divide(5088, 24), 212);
        // Power-of-two divisors match Round2 exactly over the whole range.
        for sum in 0..=(16 * 255) {
            assert_eq!(approx_divide(sum, 16), (sum + 8) >> 4);
        }
    }
}
