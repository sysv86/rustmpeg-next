// SPDX-License-Identifier: Apache-2.0
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

//! AV2 film grain synthesis and signalling (AV2 Spec v1.0.0 §5.14, §5.18.10,
//! §6.13, §6.17.10, §7.21.7).
//!
//! Grain is applied to the output picture only; reference frames stay clean
//! (§7.21.1). The module provides the normative PRNG and template generation,
//! the piecewise-linear scaling lookup, the add-noise process, the film grain
//! OBU writer, and a simple encoder-side estimator.

use crate::av2::frame::Frame;
use crate::av2::headers::{FilmGrainData, ObuType, PixelLayout};
use crate::av2::intops::iclip;
use crate::av2::obu::write_obu;
use crate::av2::pixel::Pixel;
use crate::av2::putbits::PutBits;
use crate::av2::warp::round2;

pub static GAUSSIAN_SEQUENCE: [i32; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388,
    -732, -104, -188, 800, 112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500,
    -228, 12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192, 1000, -332, 652, -136,
    -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124, -24, 396, 540, -12, -104, 640,
    464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40, -156, 252,
    -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60,
    -264, -96, -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792,
    304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548,
    488, -344, -136, 488, -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168, -108,
    464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116, 436, 504, -232, 328,
    844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4,
    196, 44, -36, -648, 932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448,
    208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540, 28, 136, -88, -436, 256, 296,
    -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768,
    388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648, -136, 372,
    -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112,
    -220, -100, 996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412,
    -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240,
    -208, -40, 100, -592, 344, -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752, -100, 560, -1020, 180,
    -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164,
    -376, -684, 592, -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864,
    424, -728, 988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776,
    -824, -884, 524, -548, -564, -68, -164, -96, 692, 364, -692, -1012, -68, 260, -480, 876, -1116,
    452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112, -440,
    -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276,
    -508, 940, 24, -400, -120, 756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892, -888, 60, 164, 68, -760, 216,
    -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776,
    1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272,
    460, -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560,
    1040, -472, 116, -848, -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972,
    -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828, 524, 212, -212, 52, 12, 200, 268,
    -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436, 316, 372,
    -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4,
    -1168, 368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272,
    -256, -420, 320, 208, -144, -156, 156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360,
    32, -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840, 784, -460, 656, -544,
    -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308, 308, -112, -636, -760,
    280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636, -76,
    392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24,
    -1044, -32, 40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424,
    -1008, -324, 588, -1496, 576, 460, -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52,
    -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164, -304, -540,
    120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148, 716, -604, 404,
    -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508,
    132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384,
    -188, -612, -576, 736, 508, 892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508,
    -144, -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472, -204, 908, -188, -1000,
    -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760, -176, 376, -524, -452,
    -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832,
    148, 248, 652, 616, 1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64,
    156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524, -268, -720, 540,
    -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468,
    -28, -120, -32, -44, 1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156,
    36, 492, -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808, -416, 592, 316, -480,
    56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324, -416, -600, 768, 268, -248, -88, -132,
    -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480, -216, -92, 1688,
    -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332, 204,
    760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728,
    -520, 276, -188, 80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272,
    -404, 180, 816, -796, 48, 152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316, 476,
    12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576, 260, -352, 120, -16, -136, -260,
    -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84,
    192, 852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848,
    -344, 224, 208, -200, 484, 128, -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112,
    -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84, 200, -152, 0, -144, 584,
    280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300,
    24, -676, -596, 132, 1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488,
    424, 24, 756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424,
    888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628, -348, -228, -380, 1012, -452,
    -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344, 428, 548,
    -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284,
    -12, 824, 164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648,
    -620, -684, -24, -376, -384, -108, -920, -1032, 768, 180, -264, -508, -1268, -260, -60, 300,
    -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4, -216, -32,
    836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228,
    132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376,
    -108, -92, 788, 368, -56, 544, 400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200, -56, 276, -88, 484, -324,
    948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116,
    260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676,
    -396, -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212,
    252, -660, -708, 484, -444, -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196, 76, 388, 232, 412, -1168,
    -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168, -608, 448, 160, -532, -272,
    28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524, -572, -556,
    -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488,
    -192, -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12,
    -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716,
    592, -80, 436, 360, 4, -248, 160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300, -212, -1144, 32, -724,
    800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552,
    -52, -100, -656, 208, -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272,
    -876, -664, -648, -332, -136, 16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92, 356, 404, -408, 252, 196, -176,
    -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508,
    -476, 324, -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348,
    -288, -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12,
    168, 1020, 512, -552, 924, -148, 716, 188, 164, -340, -520, -184, 880, -152, -680, -208, -1156,
    -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280, -224, 96,
    568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428, -484,
];
/// Returns the AV2 `CHROMA_FORMAT_*` value for a pixel layout
/// (§5.14: `fgm_chroma_idc`, Table in §6.4.2).
pub fn chroma_idc(layout: PixelLayout) -> u32 {
    match layout {
        PixelLayout::I420 => 0,
        PixelLayout::I400 => 1,
        PixelLayout::I444 => 2,
        PixelLayout::I422 => 3,
    }
}

/// `get_random_number(bits)` (§7.21.7.2).
pub fn get_random_number(register: &mut u16, bits: u32) -> i32 {
    let r = *register;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    let r = (r >> 1) | (bit << 15);
    *register = r;
    ((r >> (16 - bits)) & ((1u16 << bits) - 1)) as i32
}

/// The three grain templates generated by §7.21.7.3.
struct GrainTemplates {
    luma: Vec<i32>,
    cb: Vec<i32>,
    cr: Vec<i32>,
    chroma_w: usize,
}

const LUMA_GRAIN_W: usize = 82;
const LUMA_GRAIN_H: usize = 73;

/// `Generate grain process` (§7.21.7.3).
fn generate_grain(
    data: &FilmGrainData,
    seed: u16,
    sub_x: u32,
    sub_y: u32,
    bit_depth: u8,
) -> GrainTemplates {
    let mut register = seed;
    let grain_min = -(1i32 << (bit_depth - 1));
    let grain_max = (1i32 << (bit_depth - 1)) - 1;
    let noise_shift = 12 - bit_depth as u32 + data.grain_scale_shift as u32;
    let ar_shift = data.ar_coeff_shift as u32;

    let mut luma = vec![0i32; LUMA_GRAIN_W * LUMA_GRAIN_H];
    for y in 0..LUMA_GRAIN_H {
        for x in 0..LUMA_GRAIN_W {
            let g = if data.num_points[0] > 0 {
                GAUSSIAN_SEQUENCE[get_random_number(&mut register, 11) as usize]
            } else {
                0
            };
            luma[y * LUMA_GRAIN_W + x] = round2(g as i64, noise_shift) as i32;
        }
    }

    // Luma auto-regressive filter.
    let lag = data.ar_coeff_lag;
    for y in 3..LUMA_GRAIN_H {
        for x in 3..LUMA_GRAIN_W - 3 {
            let mut s: i64 = 0;
            let mut pos = 0usize;
            for delta_row in -lag..=0 {
                for delta_col in -lag..=lag {
                    if delta_row == 0 && delta_col == 0 {
                        break;
                    }
                    let c = data.ar_coeffs[0][pos] as i64;
                    let yy = (y as i32 + delta_row) as usize;
                    let xx = (x as i32 + delta_col) as usize;
                    s += luma[yy * LUMA_GRAIN_W + xx] as i64 * c;
                    pos += 1;
                }
            }
            let idx = y * LUMA_GRAIN_W + x;
            luma[idx] = iclip(luma[idx] + round2(s, ar_shift) as i32, grain_min, grain_max);
        }
    }

    let chroma_w = if sub_x != 0 { 44 } else { LUMA_GRAIN_W };
    let chroma_h = if sub_y != 0 { 38 } else { LUMA_GRAIN_H };
    let mut cb = vec![0i32; chroma_w * chroma_h];
    let mut cr = vec![0i32; chroma_w * chroma_h];

    register = seed ^ 0xb524;
    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let g = if data.num_points[1] > 0 || data.chroma_scaling_from_luma {
                GAUSSIAN_SEQUENCE[get_random_number(&mut register, 11) as usize]
            } else {
                0
            };
            cb[y * chroma_w + x] = round2(g as i64, noise_shift) as i32;
        }
    }
    register = seed ^ 0x49d8;
    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let g = if data.num_points[2] > 0 || data.chroma_scaling_from_luma {
                GAUSSIAN_SEQUENCE[get_random_number(&mut register, 11) as usize]
            } else {
                0
            };
            cr[y * chroma_w + x] = round2(g as i64, noise_shift) as i32;
        }
    }

    // Chroma auto-regressive filter (with the optional luma term at (0, 0)).
    for y in 3..chroma_h {
        for x in 3..chroma_w - 3 {
            let mut s0: i64 = 0;
            let mut s1: i64 = 0;
            let mut pos = 0usize;
            for delta_row in -lag..=0 {
                for delta_col in -lag..=lag {
                    let c0 = data.ar_coeffs[1][pos] as i64;
                    let c1 = data.ar_coeffs[2][pos] as i64;
                    if delta_row == 0 && delta_col == 0 {
                        if data.num_points[0] > 0 {
                            let mut luma_sum = 0i32;
                            let luma_x = ((x as i32 - 3) << sub_x) + 3;
                            let luma_y = ((y as i32 - 3) << sub_y) + 3;
                            for i in 0..=sub_y as i32 {
                                for j in 0..=sub_x as i32 {
                                    luma_sum += luma[((luma_y + i) as usize) * LUMA_GRAIN_W
                                        + (luma_x + j) as usize];
                                }
                            }
                            let luma_val = round2(luma_sum as i64, sub_x + sub_y);
                            s0 += luma_val * c0;
                            s1 += luma_val * c1;
                        }
                        break;
                    }
                    let yy = (y as i32 + delta_row) as usize;
                    let xx = (x as i32 + delta_col) as usize;
                    s0 += cb[yy * chroma_w + xx] as i64 * c0;
                    s1 += cr[yy * chroma_w + xx] as i64 * c1;
                    pos += 1;
                }
            }
            let idx = y * chroma_w + x;
            cb[idx] = iclip(cb[idx] + round2(s0, ar_shift) as i32, grain_min, grain_max);
            cr[idx] = iclip(cr[idx] + round2(s1, ar_shift) as i32, grain_min, grain_max);
        }
    }

    GrainTemplates {
        luma,
        cb,
        cr,
        chroma_w,
    }
}

/// Piecewise-linear scaling lookup tables (§7.21.7.4).
struct ScalingLuts {
    lut: [[u8; 256]; 3],
}

impl ScalingLuts {
    fn build(data: &FilmGrainData, num_planes: usize) -> Self {
        let mut lut = [[0u8; 256]; 3];
        for (plane, table) in lut.iter_mut().enumerate().take(num_planes) {
            let num_points = if plane == 0 || data.chroma_scaling_from_luma {
                data.num_points[0]
            } else {
                data.num_points[plane]
            } as usize;
            if num_points == 0 {
                continue;
            }
            let get_x = |i: usize| -> usize {
                if plane == 0 || data.chroma_scaling_from_luma {
                    data.points[0][i][0] as usize
                } else {
                    data.points[plane][i][0] as usize
                }
            };
            let get_y = |i: usize| -> i32 {
                if plane == 0 || data.chroma_scaling_from_luma {
                    data.points[0][i][1] as i32
                } else {
                    data.points[plane][i][1] as i32
                }
            };

            let first_x = get_x(0);
            for x in 0..first_x.min(256) {
                table[x] = get_y(0) as u8;
            }
            for i in 0..num_points - 1 {
                let delta_y = get_y(i + 1) - get_y(i);
                let delta_x = get_x(i + 1) - get_x(i);
                if delta_x == 0 {
                    continue;
                }
                let delta = delta_y * ((65536 + (delta_x >> 1)) as i32 / delta_x as i32);
                for x in 0..delta_x {
                    let v = get_y(i) + ((x as i32 * delta + 32768) >> 16);
                    let dst = get_x(i) + x;
                    if dst < 256 {
                        table[dst] = v.clamp(0, 255) as u8;
                    }
                }
            }
            let last_x = get_x(num_points - 1);
            for x in last_x..256 {
                table[x] = get_y(num_points - 1) as u8;
            }
        }
        Self { lut }
    }

    /// `scale_lut(plane, index, bitDepth)` (§7.21.7.5).
    fn scale(&self, plane: usize, index: i32, bit_depth: u8) -> i32 {
        let shift = (bit_depth - 8) as u32;
        let x = (index >> shift).clamp(0, 255) as usize;
        let rem = index - ((x as i32) << shift);
        if x == 255 || shift == 0 {
            return self.lut[plane][x] as i32;
        }
        let start = self.lut[plane][x] as i32;
        let end = self.lut[plane][x + 1] as i32;
        start + round2(((end - start) * rem) as i64, shift) as i32
    }
}

/// Applies film grain to the output picture (§7.21.7).
///
/// `frame` must be the cropped output picture. Reference pictures must not be
/// passed to this function; grain is never stored in the DPB (§7.21.1).
pub fn apply_film_grain<P: Pixel>(frame: &mut Frame<P>, data: &FilmGrainData, seed: u16) {
    let w = frame.width;
    let h = frame.height;
    if w == 0 || h == 0 {
        return;
    }
    let bit_depth = frame.bit_depth;
    let (sub_x, sub_y, num_planes) = match frame.layout {
        PixelLayout::I400 => (1u32, 1u32, 1usize),
        PixelLayout::I420 => (1, 1, 3),
        PixelLayout::I422 => (1, 0, 3),
        PixelLayout::I444 => (0, 0, 3),
    };
    let has_grain = data.num_points[0] > 0
        || data.num_points[1] > 0
        || data.num_points[2] > 0
        || data.chroma_scaling_from_luma;
    if !has_grain {
        return;
    }

    let templates = generate_grain(data, seed, sub_x, sub_y, bit_depth);
    let luts = ScalingLuts::build(data, num_planes);
    let grain_min = -(1i32 << (bit_depth - 1));
    let grain_max = (1i32 << (bit_depth - 1)) - 1;

    let luma_size = if data.block_size != 0 { 32usize } else { 16 };
    let step = luma_size >> 1;
    let half_w = w.div_ceil(2);
    let half_h = h.div_ceil(2);
    let n_stripes = half_h.div_ceil(step).max(1);
    let stripe_w = w + 2 * luma_size + 4;
    let stripe_h = luma_size + 2;

    // noiseStripe[lumaNum][plane][row][col]
    let mut stripes = vec![vec![0i32; stripe_h * stripe_w]; n_stripes * num_planes];

    for luma_num in 0..n_stripes {
        let y = luma_num * step;
        let mut register = seed;
        let luma_rand = (y >> 3) as u32;
        register ^= (((luma_rand * 37 + 178) & 255) << 8) as u16;
        register ^= ((luma_rand * 173 + 105) & 255) as u16;
        let mut x = 0usize;
        while x < half_w {
            let offset_y = (get_random_number(&mut register, 9) * (3 - data.block_size)) >> 6;
            let _ = get_random_number(&mut register, 1);
            let _ = get_random_number(&mut register, 1);
            let _ = get_random_number(&mut register, 1);
            let offset_x = (get_random_number(&mut register, 9) * (3 - data.block_size)) >> 6;
            let _ = get_random_number(&mut register, 1);
            let _ = get_random_number(&mut register, 1);
            let _ = get_random_number(&mut register, 1);

            for plane in 0..num_planes {
                let plane_sub_x = if plane > 0 { sub_x } else { 0 };
                let plane_sub_y = if plane > 0 { sub_y } else { 0 };
                let plane_offset_x = if plane_sub_x != 0 {
                    6 + offset_x
                } else {
                    9 + offset_x * 2
                };
                let plane_offset_y = if plane_sub_y != 0 {
                    6 + offset_y
                } else {
                    9 + offset_y * 2
                };
                let rows = (luma_size + 2) >> plane_sub_y;
                let cols = (luma_size + 2) >> plane_sub_x;
                let stripe = &mut stripes[luma_num * num_planes + plane];
                for i in 0..rows {
                    for j in 0..cols {
                        let src_x = (plane_offset_x + j as i32) as usize;
                        let src_y = (plane_offset_y + i as i32) as usize;
                        let g = match plane {
                            0 => templates.luma[src_y * LUMA_GRAIN_W + src_x],
                            1 => templates.cb[src_y * templates.chroma_w + src_x],
                            _ => templates.cr[src_y * templates.chroma_w + src_x],
                        };
                        if plane_sub_x == 0 {
                            if j < 2 && data.overlap_flag && x > 0 {
                                let old = stripe[i * stripe_w + x * 2 + j];
                                let blended = if j == 0 {
                                    old * 27 + g * 17
                                } else {
                                    old * 17 + g * 27
                                };
                                let v =
                                    iclip(round2(blended as i64, 5) as i32, grain_min, grain_max);
                                stripe[i * stripe_w + x * 2 + j] = v;
                            } else {
                                stripe[i * stripe_w + x * 2 + j] = g;
                            }
                        } else {
                            if j == 0 && data.overlap_flag && x > 0 {
                                let old = stripe[i * stripe_w + x + j];
                                let blended = old * 23 + g * 22;
                                let v =
                                    iclip(round2(blended as i64, 5) as i32, grain_min, grain_max);
                                stripe[i * stripe_w + x + j] = v;
                            } else {
                                stripe[i * stripe_w + x + j] = g;
                            }
                        }
                    }
                }
            }
            x += step;
        }
    }

    // Build noiseImage per plane, blending vertically between stripes.
    let mut noise: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        let plane_sub_x = if plane > 0 { sub_x } else { 0 };
        let plane_sub_y = if plane > 0 { sub_y } else { 0 };
        let pw = ((w as u32 + plane_sub_x) >> plane_sub_x) as usize;
        let ph = ((h as u32 + plane_sub_y) >> plane_sub_y) as usize;
        let mut image = vec![0i32; pw * ph];
        for y in 0..ph {
            let luma_num = y >> (4 + data.block_size as usize - plane_sub_y as usize);
            let luma_num = luma_num.min(n_stripes - 1);
            let i = y - (luma_num << (4 + data.block_size as usize - plane_sub_y as usize));
            for x in 0..pw {
                let stripe = &stripes[luma_num * num_planes + plane];
                let mut g = stripe[i * stripe_w + x];
                if plane_sub_y == 0 {
                    if i < 2 && luma_num > 0 && data.overlap_flag {
                        let old = stripes[(luma_num - 1) * num_planes + plane]
                            [(i + luma_size) * stripe_w + x];
                        let blended = if i == 0 {
                            old * 27 + g * 17
                        } else {
                            old * 17 + g * 27
                        };
                        g = iclip(round2(blended as i64, 5) as i32, grain_min, grain_max);
                    }
                } else if i < 1 && luma_num > 0 && data.overlap_flag {
                    let old = stripes[(luma_num - 1) * num_planes + plane]
                        [(i + (luma_size >> 1)) * stripe_w + x];
                    let blended = old * 23 + g * 22;
                    g = iclip(round2(blended as i64, 5) as i32, grain_min, grain_max);
                }
                image[y * pw + x] = g;
            }
        }
        noise.push(image);
    }

    let (min_value, max_luma, max_chroma) = if data.clip_to_restricted_range {
        let min_value = 16i32 << (bit_depth - 8);
        let max_luma = 235i32 << (bit_depth - 8);
        let max_chroma = if data.mc_identity {
            max_luma
        } else {
            240i32 << (bit_depth - 8)
        };
        (min_value, max_luma, max_chroma)
    } else {
        let max = (256i32 << (bit_depth - 8)) - 1;
        (0, max, max)
    };
    let scaling_shift = data.scaling_shift as u32;

    if num_planes > 1 {
        let cw = ((w as u32 + sub_x) >> sub_x) as usize;
        let ch = ((h as u32 + sub_y) >> sub_y) as usize;
        for y in 0..ch {
            let luma_y = y << sub_y;
            for x in 0..cw {
                let luma_x = x << sub_x;
                let luma_next_x = (luma_x + 1).min(w - 1);
                let average_luma = if sub_x != 0 {
                    round2(
                        (frame.planes[0].data[luma_y * frame.planes[0].stride + luma_x].to_i32()
                            + frame.planes[0].data[luma_y * frame.planes[0].stride + luma_next_x]
                                .to_i32()) as i64,
                        1,
                    ) as i32
                } else {
                    frame.planes[0].data[luma_y * frame.planes[0].stride + luma_x].to_i32()
                };
                for plane in 1..3 {
                    if data.num_points[plane] == 0 && !data.chroma_scaling_from_luma {
                        continue;
                    }
                    let idx = y * cw + x;
                    let orig =
                        frame.planes[plane].data[y * frame.planes[plane].stride + x].to_i32();
                    let merged = if data.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined = average_luma * (data.uv_luma_mult[plane - 1])
                            + orig * (data.uv_mult[plane - 1]);
                        iclip(
                            (combined >> 6) + ((data.uv_offset[plane - 1]) << (bit_depth - 8)),
                            0,
                            (1 << bit_depth) - 1,
                        )
                    };
                    let noise_val = round2(
                        (luts.scale(plane, merged, bit_depth) * noise[plane][idx]) as i64,
                        scaling_shift,
                    ) as i32;
                    let out = iclip(orig + noise_val, min_value, max_chroma);
                    frame.planes[plane].data[y * frame.planes[plane].stride + x] = P::from_i32(out);
                }
            }
        }
    }

    if data.num_points[0] > 0 {
        for y in 0..h {
            for x in 0..w {
                let orig = frame.planes[0].data[y * frame.planes[0].stride + x].to_i32();
                let noise_val = round2(
                    (luts.scale(0, orig, bit_depth) * noise[0][y * w + x]) as i64,
                    scaling_shift,
                ) as i32;
                let out = iclip(orig + noise_val, min_value, max_luma);
                frame.planes[0].data[y * frame.planes[0].stride + x] = P::from_i32(out);
            }
        }
    }
}

/// Writes a complete `OBU_FILM_GRAIN` OBU carrying one model in `slot`
/// (§5.14 `film_grain_obu()`, §5.18.10.2 `film_grain_model()`).
pub fn write_film_grain_obu(
    data: &FilmGrainData,
    layout: PixelLayout,
    slot: u8,
    tlayer_id: u8,
    mlayer_id: u8,
    xlayer_id: u8,
) -> Vec<u8> {
    let mut pb = PutBits::new();
    let slot = slot & 7;
    pb.put_bits(8, 1u32 << slot); // fgm_update_flags
    pb.put_vlc(chroma_idc(layout)); // fgm_chroma_idc

    let monochrome = layout == PixelLayout::I400;
    let chroma_scaling_from_luma = data.chroma_scaling_from_luma && !monochrome;
    if !monochrome {
        pb.put_bit(chroma_scaling_from_luma as u32);
    }

    let num_planes = if monochrome { 1 } else { 3 };
    let write_points = |pb: &mut PutBits, plane: usize| {
        let num_points = data.num_points[plane].clamp(0, 14) as u32;
        pb.put_bits(4, num_points);
        if num_points == 0 {
            return;
        }
        pb.put_bits(3, 7); // point_value_increment_bits_minus_1 -> 8 bits
        pb.put_bits(2, 3); // point_scaling_bits_minus_5 -> 8 bits
        let mut previous = 0u32;
        for i in 0..num_points as usize {
            let value = data.points[plane][i][0] as u32;
            pb.put_bits(8, value.saturating_sub(previous));
            previous = value;
            pb.put_bits(8, data.points[plane][i][1] as u32);
        }
    };
    write_points(&mut pb, 0);
    if !monochrome && !chroma_scaling_from_luma {
        // §5.18.10.2: with chroma_scaling_from_luma both chroma point
        // counts are inferred to zero and are not present.
        write_points(&mut pb, 1);
        write_points(&mut pb, 2);
    }

    pb.put_bits(2, (data.scaling_shift.clamp(8, 11) - 8) as u32);
    let lag = data.ar_coeff_lag.clamp(0, 3) as u32;
    pb.put_bits(2, lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = num_pos_luma + u32::from(data.num_points[0] > 0);

    if data.num_points[0] > 0 {
        pb.put_bits(2, 3); // bits_per_ar_coeff_y_minus_5 -> 8 bits
        for i in 0..num_pos_luma as usize {
            pb.put_bits(8, (data.ar_coeffs[0][i] as i32 + 128) as u32);
        }
    }
    if chroma_scaling_from_luma || data.num_points[1] > 0 {
        pb.put_bits(2, 3);
        for i in 0..num_pos_chroma as usize {
            pb.put_bits(8, (data.ar_coeffs[1][i] as i32 + 128) as u32);
        }
    }
    if chroma_scaling_from_luma || data.num_points[2] > 0 {
        pb.put_bits(2, 3);
        for i in 0..num_pos_chroma as usize {
            pb.put_bits(8, (data.ar_coeffs[2][i] as i32 + 128) as u32);
        }
    }

    pb.put_bits(2, (data.ar_coeff_shift.clamp(6, 9) - 6) as u32);
    pb.put_bits(2, data.grain_scale_shift.clamp(0, 3) as u32);

    for plane in 1..num_planes {
        if data.num_points[plane] == 0 {
            continue;
        }
        pb.put_bits(8, (data.uv_mult[plane - 1] + 128) as u32);
        pb.put_bits(8, (data.uv_luma_mult[plane - 1] + 128) as u32);
        pb.put_bits(9, (data.uv_offset[plane - 1] + 256) as u32);
    }

    pb.put_bit(data.overlap_flag as u32);
    pb.put_bit(data.clip_to_restricted_range as u32);
    if data.clip_to_restricted_range {
        pb.put_bit(data.mc_identity as u32);
    }
    pb.put_bit(data.block_size as u32);
    pb.put_trailing_bits();

    write_obu(
        ObuType::Fgm,
        tlayer_id,
        mlayer_id,
        xlayer_id,
        &pb.into_bytes(),
    )
}

/// Fits four lag-1 auto-regressive coefficients from a residual plane.
///
/// The fit solves the normal equations for the four causal neighbours
/// `(-1,-1), (-1,0), (-1,1), (0,-1)` of §7.21.7.3. It is a simple encoder-side
/// estimate; the decoder always uses the signalled coefficients.
fn fit_ar4(residual: &[i32], width: usize, height: usize) -> [i8; 4] {
    if width < 3 || height < 2 {
        return [0; 4];
    }
    let mut a = [[0f64; 4]; 4];
    let mut b = [0f64; 4];
    // A 4x4 subsample is plenty for a four-coefficient fit and keeps the
    // encoder-side estimate cheap on large frames.
    for y in (1..height).step_by(4) {
        for x in (1..width - 1).step_by(4) {
            let e = residual[y * width + x] as f64;
            let n = [
                residual[(y - 1) * width + x - 1] as f64,
                residual[(y - 1) * width + x] as f64,
                residual[(y - 1) * width + x + 1] as f64,
                residual[y * width + x - 1] as f64,
            ];
            for i in 0..4 {
                b[i] += n[i] * e;
                for j in 0..4 {
                    a[i][j] += n[i] * n[j];
                }
            }
        }
    }
    let trace: f64 = (0..4).map(|i| a[i][i]).sum();
    let regularization = (trace * 1e-3 + 1e-6).max(1e-9);
    for i in 0..4 {
        a[i][i] += regularization;
    }

    // Gaussian elimination with partial pivoting.
    for col in 0..4 {
        let mut pivot = col;
        for row in col + 1..4 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return [0; 4];
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let diag = a[col][col];
        for row in col + 1..4 {
            let factor = a[row][col] / diag;
            for k in col..4 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut c = [0f64; 4];
    for row in (0..4).rev() {
        let mut sum = b[row];
        for k in row + 1..4 {
            sum -= a[row][k] * c[k];
        }
        c[row] = sum / a[row][row];
    }

    let mut q = [0i8; 4];
    for i in 0..4 {
        let v = ((c[i].clamp(-1.5, 1.5) * 64.0).round() as i32).clamp(-127, 127);
        q[i] = v as i8;
    }
    if q.iter().all(|&v| v == 0) {
        // A flat residual has no measurable correlation; use a mild vertical
        // predictor so the signalled model still shapes the noise.
        q[1] = 32;
    }
    q
}

/// Estimates a film grain model from the difference between the source and the
/// encoder reconstruction. Returns an inactive model when the residual is flat.
pub fn estimate_film_grain_params<P: Pixel>(
    src: &Frame<P>,
    rec: &Frame<P>,
    _seed: u16,
) -> FilmGrainData {
    let w = src.width.min(rec.width);
    let h = src.height.min(rec.height);
    let mut data = FilmGrainData::default();
    if w < 16 || h < 16 {
        return data;
    }

    // Luma residual and per-intensity-bin standard deviation.
    let mut residual = vec![0i32; w * h];
    let mut bin_sum = [0i64; 5];
    let mut bin_sq = [0i64; 5];
    let mut bin_count = [0u64; 5];
    let mut total_sq = 0i64;
    for y in 0..h {
        for x in 0..w {
            let s = src.planes[0].data[y * src.planes[0].stride + x].to_i32();
            let r = rec.planes[0].data[y * rec.planes[0].stride + x].to_i32();
            let e = s - r;
            residual[y * w + x] = e;
            total_sq += (e * e) as i64;
            let bin = ((s.max(0) as usize * 5) / 256).min(4);
            bin_sum[bin] += e as i64;
            bin_sq[bin] += (e * e) as i64;
            bin_count[bin] += 1;
        }
    }
    let variance = total_sq as f64 / (w * h) as f64;
    if variance < 0.01 {
        return data;
    }

    // Piecewise-linear scaling points at 0, 64, 128, 192 and 255.
    const POINT_X: [u8; 5] = [0, 64, 128, 192, 255];
    let mut points = [[0u8; 2]; 14];
    let mut previous_scaling = 0u8;
    for i in 0..5 {
        let std_dev = if bin_count[i] > 0 {
            let n = bin_count[i] as f64;
            let mean = bin_sum[i] as f64 / n;
            let var = (bin_sq[i] as f64 / n - mean * mean).max(0.0);
            var.sqrt()
        } else {
            0.0
        };
        // A gain of four keeps the synthetic grain visible after quantisation.
        let mut scaling = (8.0 * std_dev * 4.0).round() as i32;
        scaling = scaling.clamp(1, 255);
        let scaling = scaling.max(previous_scaling as i32) as u8;
        previous_scaling = scaling;
        points[i] = [POINT_X[i], scaling];
    }
    data.num_points[0] = 5;
    data.points[0] = points;

    data.chroma_scaling_from_luma = src.layout != PixelLayout::I400;
    data.num_points[1] = 0;
    data.num_points[2] = 0;
    data.scaling_shift = 8;
    data.ar_coeff_lag = 1;
    data.ar_coeffs[0][..4].copy_from_slice(&fit_ar4(&residual, w, h));
    data.ar_coeff_shift = 6;
    data.grain_scale_shift = 0;
    data.overlap_flag = true;
    data.clip_to_restricted_range = true;
    data.mc_identity = false;
    data.block_size = 0;

    // Chroma coefficients shape the (scaled-from-luma) chroma templates.
    if src.layout != PixelLayout::I400 {
        for plane in 1..3 {
            let pw = src.planes[plane].width;
            let ph = src.planes[plane].height;
            let mut chroma = vec![0i32; pw * ph];
            for y in 0..ph {
                for x in 0..pw {
                    let s = src.planes[plane].data[y * src.planes[plane].stride + x].to_i32();
                    let r = rec.planes[plane].data[y * rec.planes[plane].stride + x].to_i32();
                    chroma[y * pw + x] = s - r;
                }
            }
            let q = fit_ar4(&chroma, pw, ph);
            // The fifth coefficient multiplies the luma template; leave it at
            // zero because chroma scaling is taken from luma.
            data.ar_coeffs[plane][..4].copy_from_slice(&q);
            data.ar_coeffs[plane][4] = 0;
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::PixelLayout;

    fn test_model() -> FilmGrainData {
        let mut data = FilmGrainData::default();
        data.num_points[0] = 2;
        data.points[0][0] = [0, 96];
        data.points[0][1] = [255, 96];
        data.chroma_scaling_from_luma = true;
        data.scaling_shift = 8;
        data.ar_coeff_lag = 1;
        data.ar_coeffs[0][0] = 16;
        data.ar_coeffs[0][1] = 40;
        data.ar_coeffs[0][2] = 16;
        data.ar_coeffs[0][3] = 8;
        data.ar_coeffs[1][1] = 24;
        data.ar_coeffs[2][1] = 24;
        data.ar_coeff_shift = 6;
        data.overlap_flag = true;
        data.clip_to_restricted_range = true;
        data
    }

    fn test_frame() -> Frame<u8> {
        let mut frame = Frame::new(64, 48, 8, PixelLayout::I420, 0);
        for y in 0..frame.height {
            for x in 0..frame.width {
                frame.planes[0].data[y * frame.planes[0].stride + x] =
                    ((x * 3 + y * 5) & 0xFF) as u8;
            }
        }
        for plane in 1..3 {
            for y in 0..frame.planes[plane].height {
                for x in 0..frame.planes[plane].width {
                    frame.planes[plane].data[y * frame.planes[plane].stride + x] =
                        (128 + x as i32 - y as i32) as u8;
                }
            }
        }
        frame
    }

    #[test]
    fn test_prng_is_deterministic() {
        let mut a = 0x1234u16;
        let mut b = 0x1234u16;
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..64 {
            let va = get_random_number(&mut a, 11);
            let vb = get_random_number(&mut b, 11);
            assert_eq!(va, vb);
            assert!((0..2048).contains(&va));
            distinct.insert(va);
        }
        assert!(distinct.len() > 8, "PRNG produced too few distinct values");
    }

    #[test]
    fn test_grain_template_same_seed_is_identical() {
        let data = test_model();
        let a = generate_grain(&data, 7391, 1, 1, 8);
        let b = generate_grain(&data, 7391, 1, 1, 8);
        let c = generate_grain(&data, 7392, 1, 1, 8);
        assert_eq!(a.luma, b.luma);
        assert_eq!(a.cb, b.cb);
        assert_eq!(a.cr, b.cr);
        assert_ne!(a.luma, c.luma);
        assert!(a.luma.iter().any(|&v| v != 0));
    }

    #[test]
    fn test_apply_film_grain_is_seed_deterministic() {
        let data = test_model();
        let mut a = test_frame();
        let mut b = test_frame();
        let mut c = test_frame();
        apply_film_grain(&mut a, &data, 7391);
        apply_film_grain(&mut b, &data, 7391);
        apply_film_grain(&mut c, &data, 7392);
        assert_eq!(a.planes[0].data, b.planes[0].data);
        assert_eq!(a.planes[1].data, b.planes[1].data);
        assert_ne!(a.planes[0].data, c.planes[0].data);
    }

    #[test]
    fn test_apply_changes_pixels() {
        let data = test_model();
        let mut frame = test_frame();
        let before = frame.planes[0].data.clone();
        apply_film_grain(&mut frame, &data, 7391);
        assert_ne!(before, frame.planes[0].data);
    }

    #[test]
    fn test_film_grain_obu_is_written() {
        let data = test_model();
        let obu = write_film_grain_obu(&data, PixelLayout::I420, 0, 0, 0, 0);
        let obus = crate::av2::dec::hls::split_obus(&obu).expect("split");
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, ObuType::Fgm);
        assert!(obus[0].payload.len() > 2);
    }
}
