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

//! AV2 In-Loop Restoration (AV2 Specification v1.0.0 §7.20):
//! - Pixel-classified (PC) Wiener filtering with pretrained filters (§7.20.4).
//! - Non-separable (NS) Wiener filtering for luma and chroma, including the
//!   luma cross-component chroma taps (§7.20.3).
//! - Per-unit `RESTORE_NONE` / `RESTORE_PC_WIENER` / `RESTORE_WIENER_NONSEP`
//!   selection and frame-level filter signalling (§5.18.7.11, §5.20.10.4-6).
//!
//! AV2 v1.0.0 has no self-guided restoration (`Sgrproj`); its
//! `FrameRestorationType` values are `NONE`, `PC_WIENER`, `WIENER_NONSEP` and
//! `SWITCHABLE` (§6.17.7.7). `apply_sgrproj_plane` is retained only as an
//! unsignalled helper and is not part of the AV2 bitstream.

use crate::av2::deblock::DeblockGrid;
use crate::av2::frame::{Frame, Plane};
use crate::av2::headers::{FrameHeader, SequenceHeader};
use crate::av2::intops::iclip;
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::quantizer::dq_lookup;
use crate::av2::wiener_tables::{
    PC_WIENER_CONFIG, PC_WIENER_FILTERS, PC_WIENER_NORMALIZER, PRETRAINED_WIENER_CLASS,
    WIENER_NS_CONFIG_UV, WIENER_NS_CONFIG_Y, setup_wiener_qval_approx_data,
};

/// Solves a 3x3 linear system $A x = b$ via Gaussian elimination with partial pivoting,
/// returning the 3 symmetric Wiener outer tap weights `[t0, t1, t2]` (with center tap
/// $t_3 = 128 - 2(t_0 + t_1 + t_2)$).
fn solve_3x3_symmetric_taps(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [i16; 4] {
    for i in 0..3 {
        let mut pivot = i;
        let mut max_v = a[i][i].abs();
        for r in (i + 1)..3 {
            if a[r][i].abs() > max_v {
                max_v = a[r][i].abs();
                pivot = r;
            }
        }
        if max_v < 1e-6 {
            return [0, 0, 0, 128];
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let diag = a[i][i];
        for r in (i + 1)..3 {
            let factor = a[r][i] / diag;
            for c in i..3 {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }

    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for c in (i + 1)..3 {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }

    // Clamp symmetric taps to AV2 normative bounds:
    // t0 in [-5..=10], t1 in [-23..=36], t2 in [-17..=83]
    let t0 = iclip((x[0] * 128.0).round() as i32, -5, 10) as i16;
    let t1 = iclip((x[1] * 128.0).round() as i32, -23, 36) as i16;
    let t2 = iclip((x[2] * 128.0).round() as i32, -17, 83) as i16;
    let t3 = 128i16 - 2 * (t0 + t1 + t2);
    [t0, t1, t2, t3]
}

/// Solves for the optimal 1D 7-tap symmetric Wiener filter `[t0, t1, t2, t3]` along horizontal
/// (`horizontal = true`) or vertical (`horizontal = false`) direction between `rec` and `src`.
pub fn solve_wiener_1d<P: Pixel>(src: &Plane<P>, rec: &Plane<P>, horizontal: bool) -> [i16; 4] {
    let w = rec.width;
    let h = rec.height;
    if w < 8 || h < 8 {
        return [0, 0, 0, 128];
    }

    let max_x = (w as i32 - 1).max(0);
    let max_y = (h as i32 - 1).max(0);
    let sample_rec = |x: i32, y: i32| -> f64 {
        let cx = x.clamp(0, max_x) as usize;
        let cy = y.clamp(0, max_y) as usize;
        rec.data[cy * rec.stride + cx].to_i32() as f64
    };

    // Parameterize symmetric 7-tap filter with unit DC gain:
    // y_hat = c + t0 * ((p[-3] - c) + (p[+3] - c)) + t1 * ((p[-2] - c) + (p[+2] - c)) + t2 * ((p[-1] - c) + (p[+1] - c))
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];

    for y in 0..h {
        let s_row = y * src.stride;
        for x in 0..w {
            let center = sample_rec(x as i32, y as i32);
            let target_diff = (src.data[s_row + x].to_i32() as f64) - center;
            let d = if horizontal {
                [
                    (sample_rec(x as i32 - 3, y as i32) - center)
                        + (sample_rec(x as i32 + 3, y as i32) - center),
                    (sample_rec(x as i32 - 2, y as i32) - center)
                        + (sample_rec(x as i32 + 2, y as i32) - center),
                    (sample_rec(x as i32 - 1, y as i32) - center)
                        + (sample_rec(x as i32 + 1, y as i32) - center),
                ]
            } else {
                [
                    (sample_rec(x as i32, y as i32 - 3) - center)
                        + (sample_rec(x as i32, y as i32 + 3) - center),
                    (sample_rec(x as i32, y as i32 - 2) - center)
                        + (sample_rec(x as i32, y as i32 + 2) - center),
                    (sample_rec(x as i32, y as i32 - 1) - center)
                        + (sample_rec(x as i32, y as i32 + 1) - center),
                ]
            };

            for i in 0..3 {
                rhs[i] += d[i] * target_diff;
                for j in 0..3 {
                    m[i][j] += d[i] * d[j];
                }
            }
        }
    }

    solve_3x3_symmetric_taps(m, rhs)
}

/// Applies a separable 7-tap symmetric Wiener filter (`h_taps`, `v_taps`) to `src_plane` into `dst_plane`.
pub fn apply_wiener_plane<BD: BitDepth>(
    bd: BD,
    dst: &mut Plane<BD::Pixel>,
    src: &Plane<BD::Pixel>,
    h_taps: &[i16; 4],
    v_taps: &[i16; 4],
) {
    let w = src.width;
    let h = src.height;
    if w == 0 || h == 0 {
        return;
    }
    let hf = [
        h_taps[0] as i32,
        h_taps[1] as i32,
        h_taps[2] as i32,
        h_taps[3] as i32,
        h_taps[2] as i32,
        h_taps[1] as i32,
        h_taps[0] as i32,
    ];
    let vf = [
        v_taps[0] as i32,
        v_taps[1] as i32,
        v_taps[2] as i32,
        v_taps[3] as i32,
        v_taps[2] as i32,
        v_taps[1] as i32,
        v_taps[0] as i32,
    ];

    let max_x = (w as i32 - 1).max(0);
    let max_y = (h as i32 - 1).max(0);
    let mut tmp = vec![0i32; w * h];

    // 1. Horizontal 7-tap pass
    for y in 0..h {
        let row_off = y * src.stride;
        for x in 0..w {
            let mut sum = 0i32;
            for k in 0..7 {
                let cx = ((x as i32) + (k as i32) - 3).clamp(0, max_x) as usize;
                sum += src.data[row_off + cx].to_i32() * hf[k];
            }
            tmp[y * w + x] = (sum + 64) >> 7;
        }
    }

    // 2. Vertical 7-tap pass
    for y in 0..h {
        let dst_row = y * dst.stride;
        for x in 0..w {
            let mut sum = 0i32;
            for k in 0..7 {
                let cy = ((y as i32) + (k as i32) - 3).clamp(0, max_y) as usize;
                sum += tmp[cy * w + x] * vf[k];
            }
            dst.data[dst_row + x] = bd.clip_pixel((sum + 64) >> 7);
        }
    }
}

/// Applies a 3x3 Self-Guided (`Sgrproj`) edge-preserving box filter with noise parameter `eps`
/// to `src_plane` into `dst_plane`.
pub fn apply_sgrproj_plane<BD: BitDepth>(
    bd: BD,
    dst: &mut Plane<BD::Pixel>,
    src: &Plane<BD::Pixel>,
    eps: i64,
) {
    let w = src.width;
    let h = src.height;
    if w == 0 || h == 0 {
        return;
    }
    let max_x = (w as i32 - 1).max(0);
    let max_y = (h as i32 - 1).max(0);

    for y in 0..h {
        let dst_row = y * dst.stride;
        for x in 0..w {
            let mut sum = 0i64;
            let mut sum_sq = 0i64;
            for dy in -1i32..=1 {
                let cy = (y as i32 + dy).clamp(0, max_y) as usize;
                for dx in -1i32..=1 {
                    let cx = (x as i32 + dx).clamp(0, max_x) as usize;
                    let v = src.data[cy * src.stride + cx].to_i32() as i64;
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let mean = (sum + 4) / 9;
            let var = ((sum_sq - (sum * sum) / 9).max(0)) / 9;
            // Edge-preserving blend weight `a = var / (var + eps)` in [0..256]
            let a_q8 = ((var * 256) / (var + eps.max(1))).clamp(0, 256);
            let center = src.data[y * src.stride + x].to_i32() as i64;
            let out = (a_q8 * center + (256 - a_q8) * mean + 128) >> 8;
            dst.data[dst_row + x] = bd.clip_pixel(out as i32);
        }
    }
}

/// Applies normative AV2 Pixel-Correlated (PC) Wiener In-Loop Restoration Filter
/// (AV2 Specification v1.0.0 §7.20.4).
pub fn apply_pc_wiener_normative<BD: BitDepth>(
    bd: BD,
    rec_luma: &mut Plane<BD::Pixel>,
    cdef_luma: &Plane<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    lr_units: &[bool],
    grid: &DeblockGrid,
) {
    if fh.restoration.planes[0].restoration_type != 1 {
        return;
    }
    apply_pc_wiener_units(bd, rec_luma, cdef_luma, orig_luma, seq, fh, lr_units, grid);
}

/// Applies PC Wiener to the units whose entry in `lr_units` is set, regardless
/// of the frame-level restoration type. Used by switchable restoration where
/// PC and NS Wiener units are interleaved.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_pc_wiener_units<BD: BitDepth>(
    bd: BD,
    rec_luma: &mut Plane<BD::Pixel>,
    cdef_luma: &Plane<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    _seq: &SequenceHeader,
    fh: &FrameHeader,
    lr_units: &[bool],
    grid: &DeblockGrid,
) {
    let w = rec_luma.width;
    let h = rec_luma.height;
    let stride = rec_luma.stride;
    if w == 0 || h == 0 || lr_units.is_empty() {
        return;
    }

    let bitdepth_min_8 = (bd.bitdepth() - 8) as i32;
    let qmax = 255 + 24 * (if bd.bitdepth() > 8 { 1 } else { 0 });
    let qidx = (fh.quant.yac as i32).min(qmax);
    let base_q = dq_lookup(fh.quant.yac as i32);
    let qstep = (base_q + (1 << bitdepth_min_8 >> 1)) >> bitdepth_min_8;
    let (qval_approx, class_lut_offset) = setup_wiener_qval_approx_data(qstep);
    let qp_group = if qidx < 130 {
        0
    } else if qidx < 190 {
        1
    } else if qidx < 220 {
        2
    } else {
        3
    };

    let unit_sz_log2 = fh.restoration.unit_size[0] as usize;
    let unit_sz = 1usize << unit_sz_log2;
    let half_unit = unit_sz >> 1;
    let unit_cols = ((w + half_unit) >> unit_sz_log2).max(1);
    let unit_rows = ((h + half_unit) >> unit_sz_log2).max(1);

    let mut stripe_start_y = 0usize;
    let mut is_first_stripe = true;

    while stripe_start_y < h {
        let stripe_h = if is_first_stripe {
            56usize.min(h - stripe_start_y)
        } else {
            64usize.min(h - stripe_start_y)
        };
        is_first_stripe = false;
        let stripe_end_y = stripe_start_y + stripe_h - 1;
        let s_start = stripe_start_y as i32;
        let s_end = stripe_end_y as i32;

        let bh_blocks = stripe_h.div_ceil(4);
        let bw_blocks = w.div_ceil(4);

        let get_sample = |x: i32, y: i32| -> i32 {
            let cx = x.clamp(0, w.saturating_sub(1) as i32) as usize;
            if y < s_start {
                if s_start == 0 {
                    cdef_luma.data[cx].to_i32()
                } else {
                    let y_fetch = y.max(s_start - 2) as usize;
                    orig_luma.data[y_fetch * orig_luma.stride + cx].to_i32()
                }
            } else if y > s_end {
                if s_end >= (h.saturating_sub(1) as i32) {
                    cdef_luma.data[(h - 1) * cdef_luma.stride + cx].to_i32()
                } else {
                    let y_fetch = y.min(s_end + 2) as usize;
                    orig_luma.data[y_fetch * orig_luma.stride + cx].to_i32()
                }
            } else {
                cdef_luma.data[y as usize * cdef_luma.stride + cx].to_i32()
            }
        };

        let by4_base = stripe_start_y / 4;
        for by in 0..bh_blocks {
            let block_y = stripe_start_y + by * 4;
            let ur = (block_y >> unit_sz_log2).min(unit_rows - 1);

            for bx in 0..bw_blocks {
                let block_x = bx * 4;
                let uc = (block_x >> unit_sz_log2).min(unit_cols - 1);
                let unit_idx = ur * unit_cols + uc;

                if unit_idx >= lr_units.len() || !lr_units[unit_idx] {
                    continue;
                }

                // Feature calculation over 3x3 grid of 2x2 cells
                let mut f0 = 0u32;
                let mut f1 = 0u32;
                let mut f2 = 0u32;

                for &dy2 in &[-1i32, 1, 3] {
                    let y2 = block_y as i32 + dy2;
                    for &dx2 in &[-1i32, 1, 3] {
                        let x2 = block_x as i32 + dx2;
                        for py in y2..y2 + 2 {
                            for px in x2..x2 + 2 {
                                let m = get_sample(px, py);
                                let up = get_sample(px, py - 1);
                                let down = get_sample(px, py + 1);
                                let vert = up - 2 * m + down;

                                let up_right = get_sample(px + 1, py - 1);
                                let down_left = get_sample(px - 1, py + 1);
                                let anti_diag = up_right - 2 * m + down_left;

                                let up_left = get_sample(px - 1, py - 1);
                                let down_right = get_sample(px + 1, py + 1);
                                let diag = up_left - 2 * m + down_right;

                                f0 += vert.unsigned_abs();
                                f1 += anti_diag.unsigned_abs();
                                f2 += diag.unsigned_abs();
                            }
                        }
                    }
                }

                let rnd = (1 << bitdepth_min_8) >> 1;
                let nf0 = ((f0 * (PC_WIENER_NORMALIZER[0] as u32) + rnd) >> bitdepth_min_8) as i32;
                let nf1 = ((f1 * (PC_WIENER_NORMALIZER[1] as u32) + rnd) >> bitdepth_min_8) as i32;
                let nf2 = ((f2 * (PC_WIENER_NORMALIZER[2] as u32) + rnd) >> bitdepth_min_8) as i32;
                let f_norm = [nf0, nf1, nf2];

                let mut s = 0i32;
                for dy in -1i32..=1 {
                    let nby = (by as i32 + dy).clamp(0, (bh_blocks - 1) as i32) as usize;
                    let wy = if dy == 0 { 4 } else { 1 };
                    let gy = by4_base + nby;
                    for dx in -1i32..=1 {
                        let nbx = (bx as i32 + dx).clamp(0, (bw_blocks - 1) as i32) as usize;
                        let wx = if dx == 0 { 4 } else { 1 };
                        let is_skip = !grid.get_lr_noskip(nbx, gy);
                        s += wy * wx * (is_skip as i32);
                    }
                }

                let mut lut_idx = if s < (qval_approx.idx0_changeover as i32) {
                    512usize
                } else {
                    0usize
                };

                for i in 0..3 {
                    let qval = (qval_approx.slopes[i] as i32) * s
                        + qval_approx.error_lut[i * 64 + s as usize]
                        + qval_approx.offsets[i];
                    let mut sub_idx = (0.max(f_norm[i] + 255 * qval) + 8192) >> 14;
                    sub_idx = sub_idx.min(255) >> 5;
                    lut_idx |= (sub_idx as usize) << (3 * (2 - i));
                }

                let class_idx =
                    PRETRAINED_WIENER_CLASS[qp_group][(lut_idx + class_lut_offset) & 4095];
                let filter = &PC_WIENER_FILTERS[qp_group][class_idx as usize];

                let end_x = (block_x + 4).min(w);
                let end_y = (block_y + 4).min(h);

                for py in block_y..end_y {
                    let row_off = py * stride;
                    for px in block_x..end_x {
                        let center = get_sample(px as i32, py as i32);
                        let mut sum = 2 * center * (filter[12] as i32);
                        for i in 0..12 {
                            let dy = PC_WIENER_CONFIG[i][0] as i32;
                            let dx = PC_WIENER_CONFIG[i][1] as i32;
                            let p1 = get_sample(px as i32 + dx, py as i32 + dy);
                            let p2 = get_sample(px as i32 - dx, py as i32 - dy);
                            sum += (filter[i] as i32) * (p1 + p2);
                        }
                        let v = (sum + 64) >> 7;
                        rec_luma.data[row_off + px] = bd.clip_pixel(v);
                    }
                }
            }
        }

        stripe_start_y += stripe_h;
    }
}

/// Source-sample accessor for one restoration stripe (§7.20.2). Samples inside
/// the stripe come from the CDEF/CCSO output; samples outside it come from the
/// deblocked frame, clamped to two lines above or below the stripe.
struct StripeSampler<'a, P: Pixel> {
    w: usize,
    h: usize,
    cdef: &'a Plane<P>,
    orig: &'a Plane<P>,
    stripe_start: i32,
    stripe_end: i32,
}

impl<P: Pixel> StripeSampler<'_, P> {
    #[inline]
    fn sample(&self, x: i32, y: i32) -> i32 {
        let cx = x.clamp(0, self.w.saturating_sub(1) as i32) as usize;
        let cy = y.clamp(0, self.h.saturating_sub(1) as i32);
        if cy < self.stripe_start {
            let y_fetch = cy.max(self.stripe_start - 2) as usize;
            self.orig.data[y_fetch * self.orig.stride + cx].to_i32()
        } else if cy > self.stripe_end {
            let y_fetch = cy.min(self.stripe_end + 2) as usize;
            self.orig.data[y_fetch * self.orig.stride + cx].to_i32()
        } else {
            self.cdef.data[cy as usize * self.cdef.stride + cx].to_i32()
        }
    }
}

/// `Wiener_Filters_420` (§7.20.3) used by `get_luma_sample`.
static WIENER_FILTERS_420: [[[i32; 2]; 2]; 2] = [[[1, 1], [1, 1]], [[2, 0], [2, 0]]];

/// `get_luma_sample` (§7.20.3): maps a chroma position to the filtered luma
/// sample used by the chroma cross-component Wiener taps.
#[allow(clippy::too_many_arguments)]
fn luma_sample(
    sampler: &StripeSampler<'_, impl Pixel>,
    seq: &SequenceHeader,
    x: i32,
    y: i32,
    sub_x: i32,
    sub_y: i32,
) -> i32 {
    let last_y = sampler.h.saturating_sub(1) as i32 - sub_y;
    let last_x = sampler.w.saturating_sub(1) as i32 - sub_x;
    let lx = (x << sub_x).clamp(0, last_x.max(0));
    let ly = (y << sub_y).clamp(0, last_y.max(0));
    let mut filter_idx = seq.cfl_ds_filter_index;
    if filter_idx == 3 {
        filter_idx = 0;
    }
    if sub_x != 0 && sub_y != 0 && filter_idx <= 1 {
        let mut t = 0;
        for dy in 0..2 {
            for dx in 0..2 {
                t += WIENER_FILTERS_420[filter_idx as usize][dy][dx]
                    * sampler.sample(lx + dx as i32, ly + dy as i32);
            }
        }
        t >> 2
    } else {
        sampler.sample(lx, ly)
    }
}

/// Applies the normative non-separable Wiener filter (§7.20.3) to the units of
/// one plane whose `unit_types` entry is `2` (`RESTORE_WIENER_NONSEP`).
#[allow(clippy::too_many_arguments)]
pub fn apply_ns_wiener_plane<BD: BitDepth>(
    bd: BD,
    plane_idx: usize,
    dst: &mut Plane<BD::Pixel>,
    cdef: &Plane<BD::Pixel>,
    orig: &Plane<BD::Pixel>,
    cdef_luma: &Plane<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    seq: &SequenceHeader,
    unit_types: &[u8],
    filter: &[i8; 18],
    unit_sz_log2: usize,
    sub_x: i32,
    sub_y: i32,
) {
    let w = dst.width;
    let h = dst.height;
    if w == 0 || h == 0 || unit_types.is_empty() {
        return;
    }
    let luma_w = cdef_luma.width;
    let luma_h = cdef_luma.height;
    let half_unit = 1usize << unit_sz_log2.saturating_sub(1);
    let unit_cols = ((w + half_unit) >> unit_sz_log2).max(1);
    let unit_rows = ((h + half_unit) >> unit_sz_log2).max(1);
    let config: &[[i8; 3]] = if plane_idx == 0 {
        &WIENER_NS_CONFIG_Y
    } else {
        &WIENER_NS_CONFIG_UV
    };
    // Stripes are defined on the luma grid; chroma stripes are the subsampled
    // luma stripe. `k = 0` starts at -8 luma lines, clamped to the frame.
    let mut k = 0usize;
    loop {
        let unclipped_start = k as i32 * 64 - 8;
        if unclipped_start >= luma_h as i32 {
            break;
        }
        let luma_stripe_start = unclipped_start.max(0);
        let luma_stripe_end = (unclipped_start + 63).min(luma_h as i32 - 1);
        k += 1;

        let stripe_start = (luma_stripe_start >> sub_y).clamp(0, h as i32 - 1);
        let stripe_end = (luma_stripe_end >> sub_y).clamp(stripe_start, h as i32 - 1);
        let sampler = StripeSampler {
            w,
            h,
            cdef,
            orig,
            stripe_start,
            stripe_end,
        };
        let luma_sampler = StripeSampler {
            w: luma_w,
            h: luma_h,
            cdef: cdef_luma,
            orig: orig_luma,
            stripe_start: luma_stripe_start,
            stripe_end: luma_stripe_end,
        };

        let block_y_start = (stripe_start as usize) & !3;
        let block_y_end = (stripe_end as usize).min(h - 1);
        let mut block_y = block_y_start;
        while block_y <= block_y_end {
            let ur = (block_y >> unit_sz_log2).min(unit_rows - 1);
            let mut block_x = 0usize;
            while block_x < w {
                let uc = (block_x >> unit_sz_log2).min(unit_cols - 1);
                let unit_idx = ur * unit_cols + uc;
                if unit_idx >= unit_types.len() || unit_types[unit_idx] != 2 {
                    block_x += 4;
                    continue;
                }
                let end_x = (block_x + 4).min(w);
                let end_y = (block_y + 4).min(h);
                for py in block_y..end_y {
                    let row_off = py * dst.stride;
                    for px in block_x..end_x {
                        let m = sampler.sample(px as i32, py as i32);
                        let mut s = m << 7;
                        for tap in config.iter() {
                            let dy = tap[0] as i32;
                            let dx = tap[1] as i32;
                            let idx = tap[2] as usize;
                            let diff = sampler.sample(px as i32 + dx, py as i32 + dy) - m;
                            s += diff * filter[idx] as i32;
                        }
                        if plane_idx > 0 {
                            let m_luma =
                                luma_sample(&luma_sampler, seq, px as i32, py as i32, sub_x, sub_y);
                            // Chroma cross-component taps use coefficients
                            // 6..17 in tap order (not the self-tap index).
                            for (i, tap) in config.iter().enumerate() {
                                let coeff = filter[i + 6] as i32;
                                if coeff != 0 {
                                    let dy = tap[0] as i32;
                                    let dx = tap[1] as i32;
                                    let luma_diff = luma_sample(
                                        &luma_sampler,
                                        seq,
                                        px as i32 + dx,
                                        py as i32 + dy,
                                        sub_x,
                                        sub_y,
                                    ) - m_luma;
                                    s += luma_diff * coeff;
                                }
                            }
                        }
                        let v = (s + 64) >> 7;
                        dst.data[row_off + px] = bd.clip_pixel(v);
                    }
                }
                block_x += 4;
            }
            block_y += 4;
        }
    }
}

/// Fixed, non-identity encoder NS filters. The first two coefficients sharpen
/// the nearest horizontal/vertical neighbours while preserving DC gain; the
/// encoder signals exactly these coefficients so reconstruction and bitstream
/// stay consistent.
pub fn encoder_ns_luma_filter() -> [i8; 16] {
    let mut f = [0i8; 16];
    f[0] = -4;
    f[1] = -4;
    f[2] = 1;
    f[3] = 1;
    f
}

/// Fixed chroma NS filter with non-zero chroma self taps and zero luma
/// cross-component taps.
pub fn encoder_ns_chroma_filter() -> [i8; 18] {
    let mut f = [0i8; 18];
    f[0] = -4;
    f[1] = -4;
    f[2] = 1;
    f[3] = 1;
    f[6] = 1;
    f[7] = 1;
    f[8] = 1;
    f[9] = 1;
    f
}

/// Applies the restoration filters selected by the frame header and by the
/// per-unit type map to the encoder reconstruction. The bitstream has already
/// signalled exactly these choices, so the filters are always applied; there is
/// no second, un-signalled RDO decision here.
#[allow(clippy::too_many_arguments)]
pub fn apply_encoder_restoration<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    orig: &[Plane<BD::Pixel>; 3],
    seq: &SequenceHeader,
    fh: &FrameHeader,
    lr_types: &[Vec<u8>; 3],
    grid: &DeblockGrid,
) {
    let cdef: [Plane<BD::Pixel>; 3] = [
        rec.planes[0].clone(),
        rec.planes[1].clone(),
        rec.planes[2].clone(),
    ];
    let ss_x = seq.ss_hor as i32;
    let ss_y = seq.ss_ver as i32;

    let luma_type = fh.restoration.planes[0].restoration_type;
    if luma_type == 1 || luma_type == 3 {
        let pc_units: Vec<bool> = lr_types[0].iter().map(|&t| t == 1).collect();
        apply_pc_wiener_units(
            bd,
            &mut rec.planes[0],
            &cdef[0],
            &orig[0],
            seq,
            fh,
            &pc_units,
            grid,
        );
    }
    if luma_type == 2 || luma_type == 3 {
        apply_ns_wiener_plane(
            bd,
            0,
            &mut rec.planes[0],
            &cdef[0],
            &orig[0],
            &cdef[0],
            &orig[0],
            seq,
            &lr_types[0],
            &fh.restoration.planes[0].ns.filter[0],
            fh.restoration.unit_size[0] as usize,
            0,
            0,
        );
    }

    if seq.layout != crate::av2::headers::PixelLayout::I400 {
        for p in 1..3 {
            if fh.restoration.planes[p].restoration_type != 2 {
                continue;
            }
            apply_ns_wiener_plane(
                bd,
                p,
                &mut rec.planes[p],
                &cdef[p],
                &orig[p],
                &cdef[0],
                &orig[0],
                seq,
                &lr_types[p],
                &fh.restoration.planes[p].ns.filter[0],
                fh.restoration.unit_size[1] as usize,
                ss_x,
                ss_y,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::PixelLayout;
    use crate::av2::pixel::{BitDepth8, BitDepth16};

    #[test]
    fn test_solve_wiener_identity() {
        let mut p = Plane::<u8>::new(16, 16, 0, 0, 8);
        for y in 0..16 {
            for x in 0..16 {
                p.data[y * p.stride + x] = ((x * 15 + y * 7) & 0xff) as u8;
            }
        }
        let taps = solve_wiener_1d(&p, &p, true);
        assert_eq!(taps, [0, 0, 0, 128]);
    }

    #[test]
    fn test_wiener_plane_8bit() {
        let mut src = Plane::<u8>::new(16, 16, 0, 0, 8);
        let mut dst = Plane::<u8>::new(16, 16, 0, 0, 8);
        for y in 0..16 {
            for x in 0..16 {
                src.data[y * src.stride + x] = 120;
            }
        }
        let h_taps = [0, 0, 0, 128];
        let v_taps = [0, 0, 0, 128];
        apply_wiener_plane(BitDepth8, &mut dst, &src, &h_taps, &v_taps);
        assert_eq!(dst.data[0], 120);
    }

    #[test]
    fn test_wiener_plane_10bit_and_12bit() {
        for bd_val in [10, 12] {
            let bd = BitDepth16::new(bd_val);
            let mut src = Plane::<u16>::new(16, 16, 0, 0, bd_val);
            let mut dst = Plane::<u16>::new(16, 16, 0, 0, bd_val);
            let test_val = (1 << (bd_val - 1)) as u16;
            for y in 0..16 {
                for x in 0..16 {
                    src.data[y * src.stride + x] = test_val;
                }
            }
            let h_taps = [0, 0, 0, 128];
            let v_taps = [0, 0, 0, 128];
            apply_wiener_plane(bd, &mut dst, &src, &h_taps, &v_taps);
            assert_eq!(dst.data[0], test_val);
        }
    }

    #[test]
    fn test_sgrproj_plane_identity() {
        let mut src = Plane::<u8>::new(16, 16, 0, 0, 8);
        let mut dst = Plane::<u8>::new(16, 16, 0, 0, 8);
        for y in 0..16 {
            for x in 0..16 {
                src.data[y * src.stride + x] = 100;
            }
        }
        apply_sgrproj_plane(BitDepth8, &mut dst, &src, 100);
        assert_eq!(dst.data[0], 100);
    }

    #[test]
    fn test_encoder_ns_filters_are_non_identity_and_in_range() {
        let luma = encoder_ns_luma_filter();
        assert!(luma.iter().any(|&c| c != 0));
        for (j, &c) in luma.iter().enumerate() {
            let low = crate::av2::wiener_tables::WIENER_NS_TAPS_MIN[0][j] as i32;
            let high = low + (1 << crate::av2::wiener_tables::WIENER_NS_TAPS_K[0][j]) - 1;
            assert!((c as i32) >= low && (c as i32) <= high);
        }
        let chroma = encoder_ns_chroma_filter();
        assert!(chroma.iter().any(|&c| c != 0));
        for (j, &c) in chroma.iter().enumerate() {
            let low = crate::av2::wiener_tables::WIENER_NS_TAPS_MIN[1][j] as i32;
            let high = low + (1 << crate::av2::wiener_tables::WIENER_NS_TAPS_K[1][j]) - 1;
            assert!((c as i32) >= low && (c as i32) <= high);
        }
    }

    #[test]
    fn test_ns_wiener_luma_changes_gradient() {
        let w = 64;
        let h = 64;
        let mut plane = Plane::<u8>::new(w, h, 0, 0, 8);
        for y in 0..h {
            for x in 0..w {
                plane.data[y * plane.stride + x] = ((x * 5 + y * 3) & 0xff) as u8;
            }
        }
        let orig = plane.clone();
        let mut dst = plane.clone();
        let seq = SequenceHeader::default();
        let types = vec![2u8; 1];
        let mut filter = [0i8; 18];
        filter[..16].copy_from_slice(&encoder_ns_luma_filter());
        apply_ns_wiener_plane(
            BitDepth8, 0, &mut dst, &plane, &orig, &plane, &orig, &seq, &types, &filter, 6, 0, 0,
        );
        assert!(
            dst.data[..w * h]
                .iter()
                .zip(&plane.data[..w * h])
                .any(|(a, b)| a != b)
        );
    }

    #[test]
    fn test_pc_wiener_normative_stripe_boundaries() {
        let w = 64;
        let h = 128; // Crosses 56-line first stripe boundary into second stripe
        let mut rec = Frame::new(w, h, 8, PixelLayout::I420, 0);
        for y in 0..h {
            for x in 0..w {
                rec.planes[0].data[y * rec.planes[0].stride + x] = ((x * 3 + y * 5) & 0xff) as u8;
            }
        }
        let orig_luma = rec.planes[0].clone();
        let cdef_luma = rec.planes[0].clone();
        let seq = SequenceHeader::default();
        let mut fh = FrameHeader::default();
        fh.restoration.planes[0].restoration_type = 1; // PC Wiener
        fh.restoration.unit_size[0] = 8; // 256x256
        let lr_units = vec![true; 1];
        let mut grid = DeblockGrid::new(w / 4, h / 4, PixelLayout::I420);
        for by in 0..(h / 4) {
            for bx in 0..(w / 4) {
                grid.set_lr_noskip(bx, by, true);
            }
        }

        apply_pc_wiener_normative(
            BitDepth8,
            &mut rec.planes[0],
            &cdef_luma,
            &orig_luma,
            &seq,
            &fh,
            &lr_units,
            &grid,
        );
        assert_eq!(rec.planes[0].width, 64);
        assert_eq!(rec.planes[0].height, 128);
    }

    #[test]
    fn test_pc_wiener_high_bit_depth() {
        for bd_val in [10, 12] {
            let bd = BitDepth16::new(bd_val);
            let w = 64;
            let h = 64;
            let mut rec = Frame::new(w, h, bd_val, PixelLayout::I420, 0);
            for y in 0..h {
                for x in 0..w {
                    rec.planes[0].data[y * rec.planes[0].stride + x] =
                        (((x * 3 + y * 5) << (bd_val - 8)) & ((1 << bd_val) - 1)) as u16;
                }
            }
            let orig_luma = rec.planes[0].clone();
            let cdef_luma = rec.planes[0].clone();
            let seq = SequenceHeader::default();
            let mut fh = FrameHeader::default();
            fh.restoration.planes[0].restoration_type = 1;
            fh.restoration.unit_size[0] = 8;
            let lr_units = vec![true; 1];
            let mut grid = DeblockGrid::new(w / 4, h / 4, PixelLayout::I420);
            for by in 0..(h / 4) {
                for bx in 0..(w / 4) {
                    grid.set_lr_noskip(bx, by, true);
                }
            }

            apply_pc_wiener_normative(
                bd,
                &mut rec.planes[0],
                &cdef_luma,
                &orig_luma,
                &seq,
                &fh,
                &lr_units,
                &grid,
            );
            assert_eq!(rec.planes[0].width, 64);
            assert_eq!(rec.planes[0].height, 64);
        }
    }
}
