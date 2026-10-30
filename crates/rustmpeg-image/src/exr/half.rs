// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! IEEE 754 binary16 (`half`), PXR24 24-bit float, and B44 perceptual
//! floating-point conversions for OpenEXR.
//!
//! All formulas in this module are derived directly from the IEEE 754-2008 binary16
//! / binary32 specification and the OpenEXR `PXR24` / `B44` mathematical specifications
//! (`docs/05-compression-codecs.md` and `OpenEXRCore/internal_pxr24.c`, `internal_b44_table_init.c`).

/// Converts an IEEE 754 binary16 bit pattern (`u16`) to an IEEE 754 binary32 (`f32`).
///
/// Exact mathematical mapping:
/// - Sign bit `s = (h >> 15) & 1`
/// - Exponent `e = (h >> 10) & 0x1f`
/// - Mantissa `m = h & 0x03ff`
/// - `e == 0, m == 0` -> signed zero `(s << 31)`
/// - `e == 0, m != 0` -> subnormal: normalize `m` by shifting left until bit 10 is 1,
///   decreasing exponent from `127 - 14 = 113`.
/// - `0 < e < 31` -> normal: `f32` exponent is `e + 112`, mantissa is `m << 13`.
/// - `e == 31, m == 0` -> signed infinity `(s << 31) | 0x7f80_0000`.
/// - `e == 31, m != 0` -> NaN: preserve payload and set quiet NaN bit `(s << 31) | 0x7fc0_0000 | (m << 13)`.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    f32::from_bits(f16_to_f32_bits(h))
}

/// Converts an IEEE 754 binary16 bit pattern (`u16`) to the raw 32-bit IEEE 754 binary32 bit pattern (`u32`).
#[inline]
pub const fn f16_to_f32_bits(h: u16) -> u32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x03ff) as u32;

    if e == 0 {
        if m == 0 {
            s
        } else {
            let mut mant = m;
            let mut exp = 113u32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            s | (exp << 23) | (mant << 13)
        }
    } else if e == 31 {
        if m == 0 {
            s | 0x7f80_0000
        } else {
            s | 0x7fc0_0000 | (m << 13)
        }
    } else {
        s | ((e + 112) << 23) | (m << 13)
    }
}

/// Converts an IEEE 754 binary32 (`f32`) to an IEEE 754 binary16 bit pattern (`u16`)
/// using round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    f32_bits_to_f16(val.to_bits())
}

/// Converts raw 32-bit IEEE 754 binary32 bits (`u32`) to an IEEE 754 binary16 bit pattern (`u16`)
/// using round-to-nearest-even.
#[inline]
pub const fn f32_bits_to_f16(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        if mant == 0 {
            sign | 0x7c00
        } else {
            let nan_payload = (mant >> 13) as u16;
            sign | 0x7e00 | (if nan_payload == 0 { 1 } else { nan_payload })
        }
    } else {
        let unbiased = exp - 127 + 15;
        if unbiased >= 31 {
            sign | 0x7c00
        } else if unbiased <= 0 {
            if unbiased < -10 {
                sign
            } else {
                let full_mant = mant | 0x0080_0000;
                let shift = (14 - unbiased) as u32;
                let half_mant = (full_mant >> shift) as u16;
                let rem = full_mant & ((1u32 << shift) - 1);
                let halfway = 1u32 << (shift - 1);
                let round_up = if rem > halfway {
                    1u16
                } else if rem == halfway {
                    half_mant & 1
                } else {
                    0
                };
                sign | (half_mant + round_up)
            }
        } else {
            let half_exp = (unbiased as u16) << 10;
            let half_mant = (mant >> 13) as u16;
            let rem = mant & 0x1fff;
            let round_up = if rem > 0x1000 {
                1u16
            } else if rem == 0x1000 {
                half_mant & 1
            } else {
                0
            };
            sign + ((half_exp | half_mant) + round_up)
        }
    }
}

/// Converts an OpenEXR `FLOAT` (`f32`) to the 24-bit floating-point representation
/// used by `PXR24` (`OpenEXRCore/internal_pxr24.c: float_to_float24`).
///
/// Exact reference rule:
/// - Sign bit `s = bits & 0x8000_0000`
/// - Exponent `e = bits & 0x7f80_0000`
/// - Mantissa `m = bits & 0x007f_ffff`
/// - If `e == 0x7f80_0000`:
///   - If `m != 0` (NaN): shift `m >>= 8`, and set `(e >> 8) | m | (if m == 0 { 1 } else { 0 })`
///   - Else (Infinity): `e >> 8`
/// - Else (finite):
///   - Round-to-nearest-even on the dropped 8 LSBs:
///     `i = ((e | m) + (m & 0x0000_0080)) >> 8`
///   - If `i >= 0x7f8000`, clamp to `(e | m) >> 8`
/// - Combine with `(s >> 8) | i`.
#[inline]
pub fn float_to_float24(f: f32) -> u32 {
    let bits = f.to_bits();
    let s = bits & 0x8000_0000;
    let e = bits & 0x7f80_0000;
    let mut m = bits & 0x007f_ffff;

    let i = if e == 0x7f80_0000 {
        if m != 0 {
            m >>= 8;
            (e >> 8) | m | u32::from(m == 0)
        } else {
            e >> 8
        }
    } else {
        let mut rounded = ((e | m) + (m & 0x0000_0080)) >> 8;
        if rounded >= 0x7f8000 {
            rounded = (e | m) >> 8;
        }
        rounded
    };

    (s >> 8) | i
}

/// Expands a 24-bit `PXR24` floating-point integer back into an IEEE 754 `f32`
/// by shifting left 8 bits (`f32::from_bits(f24 << 8)`).
#[inline]
pub fn float24_to_float(f24: u32) -> f32 {
    f32::from_bits(f24 << 8)
}

/// Converts a 16-bit half-float (`u16`) with `pLinear == 1` using `exp(f / 8.0)`
/// before `B44` / `B44A` packing (`OpenEXRCore/internal_b44_table_init.c: b44_convertFromLinear`).
///
/// Exact reference rule:
/// - If `(x & 0x7c00) == 0x7c00` (Infinity or NaN) -> `0`
/// - If `x >= 0x558c && x < 0x8000` (`>= 8 * ln(HALF_MAX)`) -> `0x7bff` (`HALF_MAX`)
/// - Else `f32_to_f16((f16_to_f32(x) / 8.0).exp())`
#[inline]
pub fn b44_convert_from_linear(x: u16) -> u16 {
    if (x & 0x7c00) == 0x7c00 {
        return 0;
    }
    if (0x558c..0x8000).contains(&x) {
        return 0x7bff;
    }
    let f = (f16_to_f32(x) / 8.0).exp();
    f32_to_f16(f)
}

/// Converts a `B44` / `B44A` unpacked 16-bit half-float (`u16`) back to linear when `pLinear == 1`
/// using `8.0 * ln(f)` (`OpenEXRCore/internal_b44_table_init.c: b44_convertToLinear`).
///
/// Exact reference rule:
/// - If `(x & 0x7c00) == 0x7c00` (Infinity or NaN) -> `0`
/// - If `x > 0x8000` (strictly negative, excluding `-0.0`) -> `0`
/// - Else `f32_to_f16(8.0 * f16_to_f32(x).ln())`
#[inline]
pub fn b44_convert_to_linear(x: u16) -> u16 {
    if (x & 0x7c00) == 0x7c00 {
        return 0;
    }
    if x > 0x8000 {
        return 0;
    }
    let f = 8.0 * f16_to_f32(x).ln();
    f32_to_f16(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_roundtrip_special_and_normals() {
        // +0.0 and -0.0
        assert_eq!(f16_to_f32_bits(0x0000), 0x0000_0000);
        assert_eq!(f16_to_f32_bits(0x8000), 0x8000_0000);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);

        // 1.0 and -1.0
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);

        // All finite 16-bit half values must roundtrip cleanly through f32
        for h in 0..=u16::MAX {
            if (h & 0x7c00) == 0x7c00 && (h & 0x03ff) != 0 {
                // NaN
                let f = f16_to_f32(h);
                assert!(f.is_nan());
                let h2 = f32_to_f16(f);
                assert_eq!(h2 & 0x7c00, 0x7c00);
                assert_ne!(h2 & 0x03ff, 0);
            } else {
                let f = f16_to_f32(h);
                let h2 = f32_to_f16(f);
                assert_eq!(h, h2, "mismatch for half {h:#06x}");
            }
        }
    }

    #[test]
    fn test_b44_linear_conversion_guards_and_accuracy() {
        // Infinity / NaN -> 0
        assert_eq!(b44_convert_from_linear(0x7c00), 0);
        assert_eq!(b44_convert_to_linear(0x7c00), 0);
        // Clamp >= 0x558c to HALF_MAX (0x7bff)
        assert_eq!(b44_convert_from_linear(0x558c), 0x7bff);
        // Negative inputs to b44_convert_to_linear (> 0x8000) -> 0
        assert_eq!(b44_convert_to_linear(0xbc00), 0);
        // 0.0 -> exp(0) = 1.0 (0x3c00) -> 8 * ln(1.0) = 0.0 (0x0000)
        assert_eq!(b44_convert_from_linear(0x0000), 0x3c00);
        assert_eq!(b44_convert_to_linear(0x3c00), 0x0000);
    }

    #[test]
    fn test_pxr24_float24_conversion() {
        assert_eq!(float_to_float24(0.0), 0);
        assert_eq!(float_to_float24(1.0), 0x3f8000);
        assert_eq!(float24_to_float(float_to_float24(1.0)), 1.0);
        let nan24 = float_to_float24(f32::from_bits(0x7f80_0001));
        assert!(float24_to_float(nan24).is_nan());
    }
}
