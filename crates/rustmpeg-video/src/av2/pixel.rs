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

//! Clean-room bit-depth and pixel sample abstraction (`u8` for 8-bit, `u16` for 10/12-bit).

use std::io::{Read, Write};
use std::simd::prelude::*;
use std::simd::{Simd, i32x8, i32x16, simd_swizzle};

/// Storage type for one video sample (`u8` for 8bpc, `u16` for 10/12bpc).
pub trait Pixel: Copy + Default + PartialEq + Eq + Send + Sync + Into<i32> + 'static {
    /// Storage width in bits (`8` for `u8`, `16` for `u16`).
    const BITDEPTH: u8;
    /// Maximum representable storage value.
    const MAX: Self;

    fn from_i32(v: i32) -> Self;
    fn to_i32(self) -> i32;
    fn as_u16(self) -> u16;

    /// Sum of absolute differences over a single row of length `w`.
    fn row_sad(a: &[Self], b: &[Self], w: usize) -> u64;

    /// Sum of squared differences over a single row of length `w`.
    fn row_sse(a: &[Self], b: &[Self], w: usize) -> u64;

    /// Loads 4 samples from `a` and `b` and returns `a - b` as `[i32; 4]`.
    fn diff4(a: &[Self], b: &[Self]) -> [i32; 4];

    /// Loads 16 samples from `a` and `b` and returns `a - b` as `i32x16`.
    fn diff16(a: &[Self], b: &[Self]) -> i32x16;

    /// Adds a constant `dc` to each sample in `dst[..n]` and clips to `0..=max_val`.
    fn add_dc_clip(dst: &mut [Self], dc: i32, max_val: i32);

    /// Adds `((c[i] + rnd) >> shift)` to `dst[i]` and clips to `0..=max_val`.
    fn add_residual_clip(dst: &mut [Self], c: &[i32], rnd: i32, shift: i32, max_val: i32);

    /// Upsamples `c[..w/2]` by 2x horizontally and adds `((c[i] + rnd) >> shift)` to `dst[..w]`.
    fn add_residual_upsample2x_row_clip(
        dst: &mut [Self],
        c: &[i32],
        rnd: i32,
        shift: i32,
        max_val: i32,
    );

    /// Reads raw little-endian Y4M plane samples into `dst`.
    fn read_le_samples<R: Read>(
        reader: &mut R,
        dst: &mut [Self],
        bps: usize,
        bit_depth: u8,
    ) -> std::io::Result<()>;

    /// Writes raw little-endian Y4M plane samples from `src`.
    fn write_le_samples<W: Write>(writer: &mut W, src: &[Self], bps: usize) -> std::io::Result<()>;
}

impl Pixel for u8 {
    const BITDEPTH: u8 = 8;
    const MAX: u8 = 0xFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self as u16
    }

    #[inline(always)]
    fn row_sad(a: &[Self], b: &[Self], w: usize) -> u64 {
        rustmpeg_core::sad_u8(&a[..w], &b[..w])
    }

    #[inline(always)]
    fn row_sse(a: &[Self], b: &[Self], w: usize) -> u64 {
        rustmpeg_core::sse_u8(&a[..w], &b[..w])
    }

    #[inline(always)]
    fn diff4(a: &[Self], b: &[Self]) -> [i32; 4] {
        [
            a[0] as i32 - b[0] as i32,
            a[1] as i32 - b[1] as i32,
            a[2] as i32 - b[2] as i32,
            a[3] as i32 - b[3] as i32,
        ]
    }

    #[inline(always)]
    fn diff16(a: &[Self], b: &[Self]) -> i32x16 {
        let va = Simd::<u8, 16>::from_slice(&a[..16]).cast::<i32>();
        let vb = Simd::<u8, 16>::from_slice(&b[..16]).cast::<i32>();
        va - vb
    }

    #[inline(always)]
    fn add_dc_clip(dst: &mut [Self], dc: i32, max_val: i32) {
        let v_dc = i32x16::splat(dc);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut x = 0;
        while x + 16 <= dst.len() {
            let p = Simd::<u8, 16>::from_slice(&dst[x..x + 16]).cast::<i32>();
            let out = (p + v_dc).simd_clamp(v_zero, v_max).cast::<u8>();
            dst[x..x + 16].copy_from_slice(out.as_array());
            x += 16;
        }
        while x < dst.len() {
            dst[x] = (dst[x] as i32 + dc).clamp(0, max_val) as u8;
            x += 1;
        }
    }

    #[inline(always)]
    fn add_residual_clip(dst: &mut [Self], c: &[i32], rnd: i32, shift: i32, max_val: i32) {
        let n = dst.len().min(c.len());
        let v_rnd = i32x16::splat(rnd);
        let v_sh = i32x16::splat(shift);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut x = 0;
        while x + 16 <= n {
            let p = Simd::<u8, 16>::from_slice(&dst[x..x + 16]).cast::<i32>();
            let vc = i32x16::from_slice(&c[x..x + 16]);
            let res = (vc + v_rnd) >> v_sh;
            let out = (p + res).simd_clamp(v_zero, v_max).cast::<u8>();
            dst[x..x + 16].copy_from_slice(out.as_array());
            x += 16;
        }
        while x < n {
            let res = (c[x] + rnd) >> shift;
            dst[x] = (dst[x] as i32 + res).clamp(0, max_val) as u8;
            x += 1;
        }
    }

    #[inline(always)]
    fn add_residual_upsample2x_row_clip(
        dst: &mut [Self],
        c: &[i32],
        rnd: i32,
        shift: i32,
        max_val: i32,
    ) {
        let half = (dst.len() / 2).min(c.len());
        let v_rnd = i32x8::splat(rnd);
        let v_sh = i32x8::splat(shift);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut ci = 0;
        while ci + 8 <= half {
            let vc = i32x8::from_slice(&c[ci..ci + 8]);
            let res8 = (vc + v_rnd) >> v_sh;
            let res16 = simd_swizzle!(res8, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]);
            let dx = ci * 2;
            let p = Simd::<u8, 16>::from_slice(&dst[dx..dx + 16]).cast::<i32>();
            let out = (p + res16).simd_clamp(v_zero, v_max).cast::<u8>();
            dst[dx..dx + 16].copy_from_slice(out.as_array());
            ci += 8;
        }
        while ci < half {
            let cf = (c[ci] + rnd) >> shift;
            let dx = ci * 2;
            dst[dx] = (dst[dx] as i32 + cf).clamp(0, max_val) as u8;
            dst[dx + 1] = (dst[dx + 1] as i32 + cf).clamp(0, max_val) as u8;
            ci += 1;
        }
    }

    #[inline]
    fn read_le_samples<R: Read>(
        reader: &mut R,
        dst: &mut [Self],
        bps: usize,
        bit_depth: u8,
    ) -> std::io::Result<()> {
        if bps == 1 {
            reader.read_exact(dst)?;
            if bit_depth < 8 {
                let max_val = (1u8 << bit_depth).saturating_sub(1);
                for v in dst.iter_mut() {
                    *v = (*v).min(max_val);
                }
            }
            Ok(())
        } else {
            let max_val = ((1u32 << bit_depth) - 1).min(255) as u16;
            let mut buf = vec![0u8; dst.len() * 2];
            reader.read_exact(&mut buf)?;
            for (o, chunk) in dst.iter_mut().zip(buf.chunks_exact(2)) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]).min(max_val);
                *o = v as u8;
            }
            Ok(())
        }
    }

    #[inline]
    fn write_le_samples<W: Write>(writer: &mut W, src: &[Self], bps: usize) -> std::io::Result<()> {
        if bps == 1 {
            writer.write_all(src)
        } else {
            let mut buf = Vec::with_capacity(src.len() * 2);
            for &p in src {
                buf.extend_from_slice(&(p as u16).to_le_bytes());
            }
            writer.write_all(&buf)
        }
    }
}

impl Pixel for u16 {
    const BITDEPTH: u8 = u16::BITS as u8;
    const MAX: u16 = 0xFFFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self
    }

    #[inline(always)]
    fn row_sad(a: &[Self], b: &[Self], w: usize) -> u64 {
        let mut sum = 0u64;
        let mut i = 0;
        while i + 16 <= w {
            let va = Simd::<u16, 16>::from_slice(&a[i..i + 16]);
            let vb = Simd::<u16, 16>::from_slice(&b[i..i + 16]);
            let diff = (va.simd_max(vb) - va.simd_min(vb)).cast::<u32>();
            sum += diff.reduce_sum() as u64;
            i += 16;
        }
        while i < w {
            sum += a[i].abs_diff(b[i]) as u64;
            i += 1;
        }
        sum
    }

    #[inline(always)]
    fn row_sse(a: &[Self], b: &[Self], w: usize) -> u64 {
        let mut sum = 0u64;
        let mut i = 0;
        while i + 8 <= w {
            let va = Simd::<u16, 8>::from_slice(&a[i..i + 8]);
            let vb = Simd::<u16, 8>::from_slice(&b[i..i + 8]);
            let diff = (va.simd_max(vb) - va.simd_min(vb)).cast::<u64>();
            sum += (diff * diff).reduce_sum();
            i += 8;
        }
        while i < w {
            let d = a[i].abs_diff(b[i]) as u64;
            sum += d * d;
            i += 1;
        }
        sum
    }

    #[inline(always)]
    fn diff4(a: &[Self], b: &[Self]) -> [i32; 4] {
        [
            a[0] as i32 - b[0] as i32,
            a[1] as i32 - b[1] as i32,
            a[2] as i32 - b[2] as i32,
            a[3] as i32 - b[3] as i32,
        ]
    }

    #[inline(always)]
    fn diff16(a: &[Self], b: &[Self]) -> i32x16 {
        let va = Simd::<u16, 16>::from_slice(&a[..16]).cast::<i32>();
        let vb = Simd::<u16, 16>::from_slice(&b[..16]).cast::<i32>();
        va - vb
    }

    #[inline(always)]
    fn add_dc_clip(dst: &mut [Self], dc: i32, max_val: i32) {
        let v_dc = i32x16::splat(dc);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut x = 0;
        while x + 16 <= dst.len() {
            let p = Simd::<u16, 16>::from_slice(&dst[x..x + 16]).cast::<i32>();
            let out = (p + v_dc).simd_clamp(v_zero, v_max).cast::<u16>();
            dst[x..x + 16].copy_from_slice(out.as_array());
            x += 16;
        }
        while x < dst.len() {
            dst[x] = (dst[x] as i32 + dc).clamp(0, max_val) as u16;
            x += 1;
        }
    }

    #[inline(always)]
    fn add_residual_clip(dst: &mut [Self], c: &[i32], rnd: i32, shift: i32, max_val: i32) {
        let n = dst.len().min(c.len());
        let v_rnd = i32x16::splat(rnd);
        let v_sh = i32x16::splat(shift);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut x = 0;
        while x + 16 <= n {
            let p = Simd::<u16, 16>::from_slice(&dst[x..x + 16]).cast::<i32>();
            let vc = i32x16::from_slice(&c[x..x + 16]);
            let res = (vc + v_rnd) >> v_sh;
            let out = (p + res).simd_clamp(v_zero, v_max).cast::<u16>();
            dst[x..x + 16].copy_from_slice(out.as_array());
            x += 16;
        }
        while x < n {
            let res = (c[x] + rnd) >> shift;
            dst[x] = (dst[x] as i32 + res).clamp(0, max_val) as u16;
            x += 1;
        }
    }

    #[inline(always)]
    fn add_residual_upsample2x_row_clip(
        dst: &mut [Self],
        c: &[i32],
        rnd: i32,
        shift: i32,
        max_val: i32,
    ) {
        let half = (dst.len() / 2).min(c.len());
        let v_rnd = i32x8::splat(rnd);
        let v_sh = i32x8::splat(shift);
        let v_zero = i32x16::splat(0);
        let v_max = i32x16::splat(max_val);
        let mut ci = 0;
        while ci + 8 <= half {
            let vc = i32x8::from_slice(&c[ci..ci + 8]);
            let res8 = (vc + v_rnd) >> v_sh;
            let res16 = simd_swizzle!(res8, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]);
            let dx = ci * 2;
            let p = Simd::<u16, 16>::from_slice(&dst[dx..dx + 16]).cast::<i32>();
            let out = (p + res16).simd_clamp(v_zero, v_max).cast::<u16>();
            dst[dx..dx + 16].copy_from_slice(out.as_array());
            ci += 8;
        }
        while ci < half {
            let cf = (c[ci] + rnd) >> shift;
            let dx = ci * 2;
            dst[dx] = (dst[dx] as i32 + cf).clamp(0, max_val) as u16;
            dst[dx + 1] = (dst[dx + 1] as i32 + cf).clamp(0, max_val) as u16;
            ci += 1;
        }
    }

    #[inline]
    fn read_le_samples<R: Read>(
        reader: &mut R,
        dst: &mut [Self],
        bps: usize,
        bit_depth: u8,
    ) -> std::io::Result<()> {
        let max_val = ((1u32 << bit_depth) - 1) as u16;
        if bps == 1 {
            let mut buf = vec![0u8; dst.len()];
            reader.read_exact(&mut buf)?;
            for (o, &b) in dst.iter_mut().zip(buf.iter()) {
                *o = (b as u16).min(max_val);
            }
        } else {
            let mut buf = vec![0u8; dst.len() * 2];
            reader.read_exact(&mut buf)?;
            for (o, chunk) in dst.iter_mut().zip(buf.chunks_exact(2)) {
                *o = u16::from_le_bytes([chunk[0], chunk[1]]).min(max_val);
            }
        }
        Ok(())
    }

    #[inline]
    fn write_le_samples<W: Write>(writer: &mut W, src: &[Self], bps: usize) -> std::io::Result<()> {
        if bps == 1 {
            let mut buf = Vec::with_capacity(src.len());
            for &p in src {
                buf.push(p.min(255) as u8);
            }
            writer.write_all(&buf)
        } else {
            let mut buf = Vec::with_capacity(src.len() * 2);
            for &p in src {
                buf.extend_from_slice(&p.to_le_bytes());
            }
            writer.write_all(&buf)
        }
    }
}

/// Bit-depth dispatch trait for 8-bit (`BitDepth8`) and 10/12-bit (`BitDepth16`) pipelines.
pub trait BitDepth: Clone + Copy + Send + Sync + 'static {
    /// Sample storage type (`u8` or `u16`).
    type Pixel: Pixel;
    /// Intermediate signed coefficient type (`i16` for 8bpc, `i32` for 10/12bpc).
    type Coef: Copy;

    /// Storage width in bits (`8` or `16`).
    const BPC: u8;

    /// Constructs from the coded bit depth (`8`, `10`, or `12`).
    fn new(bitdepth: u8) -> Self;

    /// Coded bit depth (`8`, `10`, or `12`).
    fn bitdepth(&self) -> u8;

    /// `(1 << bitdepth) - 1` — maximum legal sample value.
    fn bitdepth_max(&self) -> i32;

    /// `bitdepth - 8` — extra precision shift for 10/12-bit modes.
    #[inline(always)]
    fn bitdepth_min_8(&self) -> i32 {
        self.bitdepth() as i32 - 8
    }

    /// Clips a reconstructed integer sample into `[0, (1 << bitdepth) - 1]`.
    #[inline(always)]
    fn pixel_clip(&self, v: i32) -> Self::Pixel {
        let max = self.bitdepth_max();
        Self::Pixel::from_i32(v.clamp(0, max))
    }

    /// Alias for [`BitDepth::pixel_clip`].
    #[inline(always)]
    fn clip_pixel(&self, v: i32) -> Self::Pixel {
        self.pixel_clip(v)
    }

    /// Alias for [`BitDepth::pixel_clip`] used by the in-loop filter and MC paths.
    #[inline(always)]
    fn iclip_pixel(&self, v: i32) -> Self::Pixel {
        self.pixel_clip(v)
    }
}

/// 8-bit sample pipeline (`u8` storage, `bitdepth_max = 255`).
#[derive(Clone, Copy, Default, Debug)]
pub struct BitDepth8;

impl BitDepth for BitDepth8 {
    type Pixel = u8;
    type Coef = i16;
    const BPC: u8 = 8;

    #[inline(always)]
    fn new(_bitdepth: u8) -> Self {
        BitDepth8
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        8
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        255
    }
}

/// High-bit-depth sample pipeline (`u16` storage, runtime 10-bit `1023` or 12-bit `4095` ceiling).
#[derive(Clone, Copy, Debug)]
pub struct BitDepth16 {
    bitdepth: u8,
}

impl BitDepth16 {
    #[inline(always)]
    pub const fn new(bitdepth: u8) -> Self {
        Self { bitdepth }
    }
}

impl BitDepth for BitDepth16 {
    type Pixel = u16;
    type Coef = i32;
    const BPC: u8 = 16;

    #[inline(always)]
    fn new(bitdepth: u8) -> Self {
        debug_assert!(bitdepth == 10 || bitdepth == 12);
        BitDepth16 { bitdepth }
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        self.bitdepth
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        (1 << self.bitdepth) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitdepth_clip_and_conversion() {
        let bd8 = BitDepth8::new(8);
        assert_eq!(bd8.bitdepth(), 8);
        assert_eq!(bd8.bitdepth_max(), 255);
        assert_eq!(bd8.clip_pixel(-10), 0u8);
        assert_eq!(bd8.clip_pixel(300), 255u8);
        assert_eq!(bd8.clip_pixel(128).to_i32(), 128);

        let bd10 = BitDepth16::new(10);
        assert_eq!(bd10.bitdepth(), 10);
        assert_eq!(bd10.bitdepth_max(), 1023);
        assert_eq!(bd10.clip_pixel(-1), 0u16);
        assert_eq!(bd10.clip_pixel(2048), 1023u16);
        assert_eq!(bd10.clip_pixel(512).to_i32(), 512);
    }
}
