// SPDX-License-Identifier: Apache-2.0
//! APV 8x8 integer transform, scaling, and zig-zag scan (RFC 9924, Sections
//! 4.4.1 and 6.3).

use super::ApvError;

/// Normative 8x8 transform matrix (RFC 9924, Figure 25).
const TRANSFORM: [[i64; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [84, 35, -35, -84, -84, -35, 35, 84],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [35, -84, 84, -35, -35, 84, -84, 35],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

/// Level scale by `qP % 6` (RFC 9924, Section 6.3.1).
pub(crate) const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 71];

/// Default quantization matrix entry (RFC 9924, Section 5.3.7).
pub(crate) const DEFAULT_QMATRIX: [u8; 64] = [16; 64];

/// Zig-zag scan positions for an 8x8 block (RFC 9924, Figure 5).
pub(crate) const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse 8x8 transform of one block.
///
/// `coefficients` is indexed `x * 8 + y`, where `x` is the horizontal
/// frequency and `y` the vertical frequency.  The result is indexed the same
/// way in the spatial domain.
pub(crate) fn inverse_8x8(coefficients: &[i32; 64]) -> [i32; 64] {
    // B is the RFC matrix (rows: frequency, columns: position).  The inverse
    // is R = B^T * D * B with the normative rounding between the two passes.
    let mut columns = [0_i64; 64];
    for u in 0..8 {
        for y in 0..8 {
            let mut sum = 0_i64;
            for v in 0..8 {
                sum += TRANSFORM[v][y] * i64::from(coefficients[u * 8 + v]);
            }
            columns[u * 8 + y] = sum;
        }
    }
    let mut scaled = [0_i64; 64];
    for index in 0..64 {
        scaled[index] = (columns[index] + 64) >> 7;
    }
    let mut output = [0_i32; 64];
    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0_i64;
            for u in 0..8 {
                sum += TRANSFORM[u][x] * scaled[u * 8 + y];
            }
            output[x * 8 + y] = sum as i32;
        }
    }
    output
}

/// Forward 8x8 transform used by the encoder.
///
/// `residual` is the spatial block in the same scale as the decoder's
/// pre-shift output (`sample << (20 - bit_depth)`).  The result is the
/// unquantized transform coefficient block.
pub(crate) fn forward_8x8(residual: &[i32; 64]) -> [i32; 64] {
    // D = B * R * B^T / 8388608 (see the inverse above).
    let mut temp = [0_i64; 64];
    for u in 0..8 {
        for j in 0..8 {
            let mut sum = 0_i64;
            for i in 0..8 {
                sum += TRANSFORM[u][i] * i64::from(residual[i * 8 + j]);
            }
            temp[u * 8 + j] = sum;
        }
    }
    let mut output = [0_i32; 64];
    for u in 0..8 {
        for v in 0..8 {
            let mut sum = 0_i64;
            for j in 0..8 {
                sum += temp[u * 8 + j] * TRANSFORM[v][j];
            }
            output[u * 8 + v] = div_round(sum, 8_388_608) as i32;
        }
    }
    output
}

/// Applies the normative scaling process to one 8x8 coefficient block.
pub(crate) fn dequantize(
    coefficients: &[i32; 64],
    tile_qp: u8,
    qmatrix: &[u8; 64],
    bit_depth: u8,
) -> Result<[i32; 64], ApvError> {
    let qp = i64::from(tile_qp);
    let scale = LEVEL_SCALE[(tile_qp % 6) as usize];
    let shift = qp / 6;
    let bd_shift = i32::from(bit_depth) - 2;
    let offset = 1_i128 << (bd_shift - 1);
    let mut output = [0_i32; 64];
    for index in 0..64 {
        let product =
            i128::from(coefficients[index]) * i128::from(qmatrix[index]) * i128::from(scale);
        let scaled = (product << shift) + offset;
        let value = scaled >> bd_shift;
        output[index] = value.clamp(i128::from(i16::MIN), i128::from(i16::MAX)) as i32;
    }
    Ok(output)
}

/// Quantizes one transform block for the encoder.
pub(crate) fn quantize(
    transform: &[i32; 64],
    tile_qp: u8,
    qmatrix: &[u8; 64],
    bit_depth: u8,
) -> Result<[i16; 64], ApvError> {
    let scale = LEVEL_SCALE[(tile_qp % 6) as usize];
    let shift = i64::from(tile_qp / 6);
    let bd_shift = i32::from(bit_depth) - 2;
    let mut output = [0_i16; 64];
    for index in 0..64 {
        let qmat = i64::from(qmatrix[index]);
        if qmat == 0 {
            return Err(ApvError::Invalid("quantization matrix entry is zero"));
        }
        let denominator = i128::from(qmat) * i128::from(scale) * (1_i128 << shift);
        let numerator = i128::from(transform[index]) * (1_i128 << bd_shift);
        let value = div_round_i128(numerator, denominator)
            .clamp(i128::from(i16::MIN), i128::from(i16::MAX));
        output[index] = value as i16;
    }
    Ok(output)
}

/// Converts the inverse-transform output to clipped samples.
pub(crate) fn output_samples(block: &[i32; 64], bit_depth: u8) -> [u16; 64] {
    let shift = 20 - i32::from(bit_depth);
    let offset = 1_i32 << (shift - 1);
    let center = 1_i32 << (bit_depth - 1);
    let maximum = (1_i32 << bit_depth) - 1;
    let mut output = [0_u16; 64];
    for index in 0..64 {
        let value = ((block[index] + offset) >> shift) + center;
        output[index] = value.clamp(0, maximum) as u16;
    }
    output
}

fn div_round(value: i64, denominator: i64) -> i64 {
    if value >= 0 {
        (value + denominator / 2) / denominator
    } else {
        (value - denominator / 2) / denominator
    }
}

fn div_round_i128(value: i128, denominator: i128) -> i128 {
    if value >= 0 {
        (value + denominator / 2) / denominator
    } else {
        (value - denominator / 2) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_matches_the_rfc_order() {
        assert_eq!(ZIGZAG[..8], [0, 1, 8, 16, 9, 2, 3, 10]);
        assert_eq!(ZIGZAG[56..], [53, 60, 61, 54, 47, 55, 62, 63]);
        let mut seen = [false; 64];
        for position in ZIGZAG {
            assert!(!seen[position]);
            seen[position] = true;
        }
    }

    #[test]
    fn dc_coefficient_reconstructs_a_constant_block() {
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 32;
        let block = inverse_8x8(&coefficients);
        assert!(block.iter().all(|sample| *sample == 1024));
    }

    #[test]
    fn forward_and_inverse_are_inverse_for_dc() {
        let mut residual = [0_i32; 64];
        residual.fill(1024);
        let transform = forward_8x8(&residual);
        assert_eq!(transform[0], 32);
        assert!(transform[1..].iter().all(|value| value.abs() <= 1));
        let back = inverse_8x8(&transform);
        assert!(back.iter().all(|sample| (*sample - 1024).abs() <= 3));
    }

    #[test]
    fn output_is_clipped_and_centered() {
        let mut block = [0_i32; 64];
        block[0] = 1 << 20;
        block[1] = -(1 << 20);
        let samples = output_samples(&block, 10);
        assert_eq!(samples[0], 1023);
        assert_eq!(samples[1], 0);
        assert_eq!(samples[2], 512);
    }
}
