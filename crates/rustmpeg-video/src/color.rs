// SPDX-License-Identifier: Apache-2.0
//! RGBA -> YUV 4:2:0 (I420), BT.709 limited range.
//!
//! This module is the canonical CPU reference for the Vulkan Compute
//! conversion kernel: both paths use the same integer coefficients and the
//! same premultiplied-alpha handling, so their output is identical byte for
//! byte.
//!
//! Only the forward conversion used by the encoder is included.  It uses
//! 16-bit fixed-point integer BT.709 arithmetic (`+32768 >> 16`) with
//! multi-threaded row chunking on frames >= 128x128, so no per-pixel
//! floating-point division or rounding occurs in the hot path and the result is
//! independent of the thread count.

use crate::I420Frame;

/// 16-bit fixed-point BT.709 limited-range 8-bit coefficients:
///
/// `Y'  = 16  + ((11966 * R + 40254 * G +  4063 * B + 32768) >> 16)`
/// `Cb' = 128 + ((-6596 * R - 22188 * G + 28784 * B + 32768) >> 16)`
/// `Cr' = 128 + ((28784 * R - 26149 * G -  2635 * B + 32768) >> 16)`
///
/// Note: `11966 + 40254 + 4063 = 56283` (`(56283 * 255 + 32768) >> 16 == 219`),
/// and chroma coefficients sum to 0 on neutral greys (`R == G == B` => `Cb = Cr = 128`).
#[inline(always)]
fn luma_i32(r: i32, g: i32, b: i32) -> u8 {
    let y = 16 + ((11_966 * r + 40_254 * g + 4_063 * b + 32_768) >> 16);
    y.clamp(0, 255) as u8
}

#[inline(always)]
fn chroma_b_i32(r: i32, g: i32, b: i32) -> u8 {
    let cb = 128 + ((-6_596 * r - 22_188 * g + 28_784 * b + 32_768) >> 16);
    cb.clamp(0, 255) as u8
}

#[inline(always)]
fn chroma_r_i32(r: i32, g: i32, b: i32) -> u8 {
    let cr = 128 + ((28_784 * r - 26_149 * g - 2_635 * b + 32_768) >> 16);
    cr.clamp(0, 255) as u8
}

/// Read one RGBA8 pixel and undo premultiplication.
///
/// The alpha channel is never coded.  Opaque input (alpha 255) is used as-is;
/// partially transparent input is unpremultiplied before conversion so that
/// the conversion matches the reference encoder for non-opaque overlays.
#[inline(always)]
fn unpremul_rgb_i32(rgba: &[u8], i: usize) -> (i32, i32, i32) {
    let a = i32::from(rgba[i + 3]);
    if a == 255 {
        (
            i32::from(rgba[i]),
            i32::from(rgba[i + 1]),
            i32::from(rgba[i + 2]),
        )
    } else if a == 0 {
        (0, 0, 0)
    } else {
        let half = a >> 1;
        let r = ((i32::from(rgba[i]) * 255 + half) / a).min(255);
        let g = ((i32::from(rgba[i + 1]) * 255 + half) / a).min(255);
        let b = ((i32::from(rgba[i + 2]) * 255 + half) / a).min(255);
        (r, g, b)
    }
}

#[inline(always)]
fn sample_box_2x2_rgb(rgba: &[u8], w: usize, h: usize, cx: usize, cy: usize) -> (i32, i32, i32) {
    let mut sum_r = 0i32;
    let mut sum_g = 0i32;
    let mut sum_b = 0i32;
    let mut count = 0i32;
    for dy in 0..2 {
        let py = cy * 2 + dy;
        if py >= h {
            continue;
        }
        let row_base = py * w;
        for dx in 0..2 {
            let px = cx * 2 + dx;
            if px >= w {
                continue;
            }
            let (r, g, b) = unpremul_rgb_i32(rgba, (row_base + px) * 4);
            sum_r += r;
            sum_g += g;
            sum_b += b;
            count += 1;
        }
    }
    if count == 4 {
        ((sum_r + 2) >> 2, (sum_g + 2) >> 2, (sum_b + 2) >> 2)
    } else if count > 0 {
        let half = count >> 1;
        (
            (sum_r + half) / count,
            (sum_g + half) / count,
            (sum_b + half) / count,
        )
    } else {
        (0, 0, 0)
    }
}

/// Converts premultiplied RGBA8 pixels to a BT.709 limited-range I420 frame.
///
/// `rgba` must contain exactly `width * height * 4` bytes; the caller validates
/// the length. A short buffer would index out of bounds, so the length is a
/// caller invariant rather than a return value.
///
/// Alpha is undone before the colour matrix is applied: fully opaque input is
/// used as is, fully transparent input becomes black, and partially
/// transparent input is unpremultiplied with rounding.
pub fn rgba_to_i420(rgba: &[u8], width: u32, height: u32) -> I420Frame {
    let (w, h) = (width as usize, height as usize);
    debug_assert_eq!(rgba.len(), w * h * 4);

    let mut y_plane = vec![0u8; w * h];
    let (cw, ch) = (width.div_ceil(2) as usize, height.div_ceil(2) as usize);
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    if w * h >= 128 * 128 {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(ch.max(1));
        let y_rows_per_chunk = h.div_ceil(threads).max(1);
        let c_rows_per_chunk = ch.div_ceil(threads).max(1);

        std::thread::scope(|s| {
            for (chunk_idx, y_chunk) in y_plane.chunks_mut(y_rows_per_chunk * w).enumerate() {
                let start_row = chunk_idx * y_rows_per_chunk;
                s.spawn(move || {
                    let rows = y_chunk.len() / w;
                    for r in 0..rows {
                        let row = start_row + r;
                        let out_row = &mut y_chunk[r * w..(r + 1) * w];
                        let row_base = row * w;
                        for (col, out) in out_row.iter_mut().enumerate() {
                            let (red, green, blue) = unpremul_rgb_i32(rgba, (row_base + col) * 4);
                            *out = luma_i32(red, green, blue);
                        }
                    }
                });
            }

            for (chunk_idx, (u_chunk, v_chunk)) in u_plane
                .chunks_mut(c_rows_per_chunk * cw)
                .zip(v_plane.chunks_mut(c_rows_per_chunk * cw))
                .enumerate()
            {
                let start_cy = chunk_idx * c_rows_per_chunk;
                s.spawn(move || {
                    let rows = u_chunk.len() / cw;
                    for r in 0..rows {
                        let cy = start_cy + r;
                        for cx in 0..cw {
                            let (red, green, blue) = sample_box_2x2_rgb(rgba, w, h, cx, cy);
                            u_chunk[r * cw + cx] = chroma_b_i32(red, green, blue);
                            v_chunk[r * cw + cx] = chroma_r_i32(red, green, blue);
                        }
                    }
                });
            }
        });
    } else {
        for row in 0..h {
            let row_base = row * w;
            for col in 0..w {
                let (r, g, b) = unpremul_rgb_i32(rgba, (row_base + col) * 4);
                y_plane[row_base + col] = luma_i32(r, g, b);
            }
        }

        for cy in 0..ch {
            for cx in 0..cw {
                let (r, g, b) = sample_box_2x2_rgb(rgba, w, h, cx, cy);
                u_plane[cy * cw + cx] = chroma_b_i32(r, g, b);
                v_plane[cy * cw + cx] = chroma_r_i32(r, g, b);
            }
        }
    }

    I420Frame {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    #[test]
    fn black_and_white_land_on_limited_range_anchors() {
        let black = rgba_to_i420(&solid(4, 4, 0, 0, 0), 4, 4);
        assert_eq!(black.y[0], 16, "black luma must be 16 in limited range");

        let white = rgba_to_i420(&solid(4, 4, 255, 255, 255), 4, 4);
        assert_eq!(white.y[0], 235, "white luma must be 235 in limited range");
    }

    #[test]
    fn neutral_grey_has_neutral_chroma() {
        let grey = rgba_to_i420(&solid(2, 2, 128, 128, 128), 2, 2);
        assert!(
            (i32::from(grey.u[0]) - 128).abs() <= 1,
            "Cb should be ~128, got {}",
            grey.u[0]
        );
        assert!(
            (i32::from(grey.v[0]) - 128).abs() <= 1,
            "Cr should be ~128, got {}",
            grey.v[0]
        );
    }

    #[test]
    fn primaries_shift_chroma_in_the_expected_direction() {
        let red = rgba_to_i420(&solid(2, 2, 255, 0, 0), 2, 2);
        assert!(red.v[0] > 230, "red should raise Cr, got {}", red.v[0]);
        assert!(red.u[0] < 110, "red should lower Cb, got {}", red.u[0]);

        let blue = rgba_to_i420(&solid(2, 2, 0, 0, 255), 2, 2);
        assert!(blue.u[0] > 230, "blue should raise Cb, got {}", blue.u[0]);
        assert!(
            blue.v[0] < 128,
            "blue should lower Cr below neutral, got {}",
            blue.v[0]
        );
        assert!(blue.v[0] > 110, "…but only slightly, got {}", blue.v[0]);
    }

    #[test]
    fn premultiplied_colour_is_unpremultiplied_before_conversion() {
        let mut half_transparent = Vec::new();
        for _ in 0..4 {
            half_transparent.extend_from_slice(&[128, 0, 0, 128]);
        }
        let soft = rgba_to_i420(&half_transparent, 2, 2);
        let opaque = rgba_to_i420(&solid(2, 2, 255, 0, 0), 2, 2);

        assert_eq!(
            soft.v[0], opaque.v[0],
            "a half-transparent red must carry the same chroma as opaque red"
        );
        assert_eq!(soft.u[0], opaque.u[0]);
    }

    #[test]
    fn odd_dimensions_round_chroma_up() {
        let frame = rgba_to_i420(&solid(3, 3, 10, 20, 30), 3, 3);
        assert_eq!(frame.chroma_size(), (2, 2));
        assert_eq!(frame.u.len(), 4);
        assert_eq!(frame.v.len(), 4);
        assert_eq!(frame.y.len(), 9);
    }

    #[test]
    fn parallel_path_matches_small_path() {
        let large = solid(256, 256, 180, 90, 40);
        let i420 = rgba_to_i420(&large, 256, 256);
        let small = rgba_to_i420(&solid(4, 4, 180, 90, 40), 4, 4);
        assert_eq!(i420.y[0], small.y[0]);
        assert_eq!(i420.u[0], small.u[0]);
        assert_eq!(i420.v[0], small.v[0]);
    }
}
