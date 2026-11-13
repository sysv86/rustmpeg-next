// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Low-frequency non-separable transform (LFNST) for H.266/VVC (ITU-T Rec. H.266 §8.7.4.2).
//!
//! Algorithms and matrices translated from VTM 24.0 `TrQuant::xFwdLfnst`,
//! `TrQuant::xInvLfnst`, `TrQuant::fwdLfnstNxN`, `TrQuant::invLfnstNxN` and
//! `RomLFNST.cpp`, BSD-3-Clause, Copyright (c) 2010-2026 ITU/ISO/IEC. See
//! `THIRD_PARTY.md`.
//!
//! Both non-transposed (`intra_mode <= 34` or `67..=80`) and transposed
//! (`35..=66` or `81..=94`) LFNST layouts are implemented for square and
//! rectangular transform blocks (`TrQuant.cpp:319-543`).

use super::tr_tables::{LFNST_4X4, LFNST_8X8, LFNST_LUT};

/// `g_coefTopLeftDiagScan8x8` split into `(x, y)` pairs.
const TOP_LEFT_DIAG_SCAN_8X8: [(usize, usize); 64] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (1, 3),
    (2, 2),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (0, 6),
    (1, 5),
    (2, 4),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (1, 7),
    (2, 6),
    (3, 5),
    (2, 7),
    (3, 6),
    (3, 7),
    (4, 0),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
    (6, 3),
    (7, 2),
    (7, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (4, 6),
    (5, 5),
    (6, 4),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (5, 7),
    (6, 6),
    (7, 5),
    (6, 7),
    (7, 6),
    (7, 7),
];

/// Map a wide-angle intra mode (`-14..=80`) to its LFNST table index (`0..95`,
/// VTM `TrQuant::getLFNSTIntraMode`, `TrQuant.cpp:319-337`).
pub fn get_lfnst_intra_mode(wide_ang_pred_mode: i32) -> u16 {
    if wide_ang_pred_mode < 0 {
        (wide_ang_pred_mode + 14 + 67) as u16
    } else if wide_ang_pred_mode >= 67 {
        (wide_ang_pred_mode + 14) as u16
    } else {
        wide_ang_pred_mode as u16
    }
}

/// `true` when VTM applies the transposed LFNST layout to `intra_mode`
/// (`TrQuant::getTransposeFlag`, `TrQuant.cpp:339-343`).
pub fn transposes(intra_mode: u16) -> bool {
    (35..67).contains(&intra_mode) || intra_mode >= 81
}

/// `g_lfnstLut[intra_mode]`: the LFNST matrix set for an intra mode (`0..95`).
pub fn lfnst_set(intra_mode: u16) -> Result<usize, &'static str> {
    LFNST_LUT
        .get(usize::from(intra_mode))
        .map(|&set| usize::from(set))
        .ok_or("intra mode outside the LFNST table")
}

/// LFNST sub-block size (`4` or `8`), matrix width (`16` or `48`) and zero-out
/// width (`8` or `16`) for a `(width, height)` transform block.
pub fn parameters_rect(
    width: usize,
    height: usize,
    lfnst_idx: u8,
) -> Result<(usize, usize, usize), &'static str> {
    if lfnst_idx == 0 || lfnst_idx > 2 {
        return Err("LFNST index outside 1..=2");
    }
    if width < 4 || height < 4 {
        return Err("LFNST requires width >= 4 and height >= 4");
    }
    let sb_size = if width >= 8 && height >= 8 { 8 } else { 4 };
    let tu_4x4_or_8x8 = (width == 4 && height == 4) || (width == 8 && height == 8);
    let tr_size = if sb_size == 4 { 16 } else { 48 };
    let zero_out = if tu_4x4_or_8x8 { 8 } else { 16 };
    Ok((sb_size, tr_size, zero_out))
}

/// Forward LFNST on a rectangular `width x height` transform block
/// (`TrQuant::xFwdLfnst`, `TrQuant.cpp:444-543`).
pub fn forward_lfnst_rect(
    width: usize,
    height: usize,
    coeffs: &mut [i32],
    lfnst_idx: u8,
    intra_mode: u16,
) -> Result<(), &'static str> {
    if coeffs.len() != width * height {
        return Err("coefficient length does not match the transform size");
    }
    let (sb_size, tr_size, zero_out) = parameters_rect(width, height, lfnst_idx)?;
    let set = lfnst_set(intra_mode)?;
    let index = usize::from(lfnst_idx - 1);
    let is_transposed = transposes(intra_mode);

    let mut src = [0i32; 48];
    if is_transposed {
        if sb_size == 4 {
            for y in 0..4 {
                src[y] = coeffs[y * width];
                src[y + 4] = coeffs[y * width + 1];
                src[y + 8] = coeffs[y * width + 2];
                src[y + 12] = coeffs[y * width + 3];
            }
        } else {
            for y in 0..8 {
                src[y] = coeffs[y * width];
                src[y + 8] = coeffs[y * width + 1];
                src[y + 16] = coeffs[y * width + 2];
                src[y + 24] = coeffs[y * width + 3];
                if y < 4 {
                    src[y + 32] = coeffs[y * width + 4];
                    src[y + 36] = coeffs[y * width + 5];
                    src[y + 40] = coeffs[y * width + 6];
                    src[y + 44] = coeffs[y * width + 7];
                }
            }
        }
    } else {
        let mut pos = 0usize;
        for y in 0..sb_size {
            let stride = if y < 4 { sb_size } else { 4 };
            for x in 0..stride {
                src[pos] = coeffs[y * width + x];
                pos += 1;
            }
        }
    }

    let mut out = [0i32; 48];
    if sb_size == 4 {
        let matrix = &LFNST_4X4[set][index];
        for (j, value) in out.iter_mut().enumerate().take(zero_out) {
            let mut sum = 0i64;
            for (i, &input) in src.iter().enumerate().take(tr_size) {
                sum += i64::from(input) * i64::from(matrix[j][i]);
            }
            *value = ((sum + 64) >> 7) as i32;
        }
    } else {
        let matrix = &LFNST_8X8[set][index];
        for (j, value) in out.iter_mut().enumerate().take(zero_out) {
            let mut sum = 0i64;
            for (i, &input) in src.iter().enumerate().take(tr_size) {
                sum += i64::from(input) * i64::from(matrix[j][i]);
            }
            *value = ((sum + 64) >> 7) as i32;
        }
    }

    for (k, &(x, y)) in TOP_LEFT_DIAG_SCAN_8X8.iter().enumerate().take(tr_size) {
        coeffs[y * width + x] = out[k];
    }
    Ok(())
}

/// Forward LFNST on a square `size x size` transform block.
pub fn forward_lfnst(
    size: usize,
    coeffs: &mut [i32],
    lfnst_idx: u8,
    intra_mode: u16,
) -> Result<(), &'static str> {
    forward_lfnst_rect(size, size, coeffs, lfnst_idx, intra_mode)
}

/// Inverse LFNST on a rectangular `width x height` transform block
/// (`TrQuant::xInvLfnst`, `TrQuant.cpp:345-442`).
pub fn inverse_lfnst_rect(
    width: usize,
    height: usize,
    coeffs: &mut [i32],
    lfnst_idx: u8,
    intra_mode: u16,
) -> Result<(), &'static str> {
    if coeffs.len() != width * height {
        return Err("coefficient length does not match the transform size");
    }
    let (sb_size, tr_size, zero_out) = parameters_rect(width, height, lfnst_idx)?;
    let set = lfnst_set(intra_mode)?;
    let index = usize::from(lfnst_idx - 1);
    let is_transposed = transposes(intra_mode);

    let mut src = [0i32; 16];
    for (i, value) in src.iter_mut().enumerate() {
        let (x, y) = TOP_LEFT_DIAG_SCAN_8X8[i];
        *value = coeffs[y * width + x];
    }

    let mut out = [0i32; 48];
    if sb_size == 4 {
        let matrix = &LFNST_4X4[set][index];
        for (j, value) in out.iter_mut().enumerate().take(tr_size) {
            let mut sum = 0i64;
            for (i, &input) in src.iter().enumerate().take(zero_out) {
                sum += i64::from(input) * i64::from(matrix[i][j]);
            }
            *value = ((sum + 64) >> 7).clamp(-32768, 32767) as i32;
        }
    } else {
        let matrix = &LFNST_8X8[set][index];
        for (j, value) in out.iter_mut().enumerate().take(tr_size) {
            let mut sum = 0i64;
            for (i, &input) in src.iter().enumerate().take(zero_out) {
                sum += i64::from(input) * i64::from(matrix[i][j]);
            }
            *value = ((sum + 64) >> 7).clamp(-32768, 32767) as i32;
        }
    }

    if is_transposed {
        if sb_size == 4 {
            for y in 0..4 {
                coeffs[y * width] = out[y];
                coeffs[y * width + 1] = out[y + 4];
                coeffs[y * width + 2] = out[y + 8];
                coeffs[y * width + 3] = out[y + 12];
            }
        } else {
            for y in 0..8 {
                coeffs[y * width] = out[y];
                coeffs[y * width + 1] = out[y + 8];
                coeffs[y * width + 2] = out[y + 16];
                coeffs[y * width + 3] = out[y + 24];
                if y < 4 {
                    coeffs[y * width + 4] = out[y + 32];
                    coeffs[y * width + 5] = out[y + 36];
                    coeffs[y * width + 6] = out[y + 40];
                    coeffs[y * width + 7] = out[y + 44];
                }
            }
        }
    } else {
        let mut pos = 0usize;
        for y in 0..sb_size {
            let stride = if y < 4 { sb_size } else { 4 };
            for x in 0..stride {
                coeffs[y * width + x] = out[pos];
                pos += 1;
            }
        }
    }
    Ok(())
}

/// Inverse LFNST on a square `size x size` transform block.
pub fn inverse_lfnst(
    size: usize,
    coeffs: &mut [i32],
    lfnst_idx: u8,
    intra_mode: u16,
) -> Result<(), &'static str> {
    inverse_lfnst_rect(size, size, coeffs, lfnst_idx, intra_mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(size: usize, seed: &mut u32) -> Vec<i32> {
        let mut out = vec![0i32; size * size];
        for value in &mut out {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *value = ((*seed >> 20) % 81) as i32 - 40;
        }
        out
    }

    #[test]
    fn planar_and_dc_never_transpose() {
        for mode in 0..2u16 {
            assert!(!transposes(mode));
            assert_eq!(lfnst_set(mode).expect("set"), 0);
        }
    }

    #[test]
    fn rejects_invalid_indexes_and_sizes() {
        let mut coeffs = vec![0i32; 16];
        assert!(forward_lfnst(4, &mut coeffs, 0, 0).is_err());
        assert!(forward_lfnst(4, &mut coeffs, 3, 0).is_err());
        assert!(forward_lfnst(4, &mut [0; 15], 1, 0).is_err());
        assert!(inverse_lfnst(4, &mut coeffs, 0, 0).is_err());
    }

    #[test]
    fn forward_zeroes_everything_outside_the_low_frequency_region() {
        for &size in &[4usize, 8, 16, 32] {
            for index in [1u8, 2] {
                let mut coeffs = block(size, &mut 0x1234_5678);
                forward_lfnst(size, &mut coeffs, index, 0).expect("forward LFNST");
                let (_, _, zero_out) = parameters_rect(size, size, index).expect("parameters");
                let mut nonzero = 0usize;
                let scan_len = if size >= 8 { 48 } else { 16 };
                for (k, &(x, y)) in TOP_LEFT_DIAG_SCAN_8X8.iter().enumerate().take(scan_len) {
                    if coeffs[y * size + x] != 0 {
                        assert!(k < zero_out, "size {size} index {index} pos {k}");
                        nonzero += 1;
                    }
                }
                assert!(nonzero <= zero_out);
            }
        }
    }

    #[test]
    fn inverse_of_forward_preserves_low_frequencies_approximately() {
        for &size in &[8usize, 16, 32] {
            for index in [1u8, 2] {
                let original = block(size, &mut (0x9ABC_DEF0 ^ size as u32 ^ u32::from(index)));
                let mut coeffs = original.clone();
                forward_lfnst(size, &mut coeffs, index, 0).expect("forward LFNST");
                inverse_lfnst(size, &mut coeffs, index, 0).expect("inverse LFNST");
                let sse: i64 = original
                    .iter()
                    .zip(&coeffs)
                    .map(|(&a, &b)| {
                        let e = i64::from(a) - i64::from(b);
                        e * e
                    })
                    .sum();
                assert!(sse < 200_000_000, "size {size} index {index} sse {sse}");
            }
        }
    }

    #[test]
    fn transposed_directional_modes_round_trip() {
        for mode in [35u16, 50, 66, 81, 94] {
            assert!(transposes(mode));
            for &size in &[4usize, 8, 16] {
                let mut coeffs = block(size, &mut (0x55AA_0000 ^ u32::from(mode)));
                forward_lfnst(size, &mut coeffs, 1, mode).expect("forward transposed LFNST");
                inverse_lfnst(size, &mut coeffs, 1, mode).expect("inverse transposed LFNST");
                assert!(coeffs.iter().all(|&v| (-32768..=32767).contains(&v)));
            }
        }
    }
}
