// SPDX-License-Identifier: Apache-2.0

//! JPEG XL VarDCT sub-bitstream decoder, clean-room encoder, 2D DCT/IDCT transforms,
//! XYB color space conversions, and in-loop restoration filters (ISO/IEC 18181-1 Annex I, J, K).

use super::JxlError;
use super::bits::{BitReader, BitWriter, U32Specifier, pack_signed, unpack_signed};
use super::entropy::{
    EntropyDecoder, EntropyEncoder, HybridUintConfig, JxlEntropyMode, read_clusters,
    read_permutation,
};
use super::headers::{
    ColourEncoding, EdgePreservingFilter, EpfParams, FrameHeader, ImageHeader, OpsinInverseMatrix,
    Primaries, TransferFunctionHeader, WhitePoint,
};
use super::modular::{
    ChannelShift, MaConfig, ModularChannelInfo, ModularEncodeOptions, ModularGrid, Predictor,
    decode_modular_subimage, encode_modular_subimage, encode_modular_subimage_v2,
};

/// Dequantization weights for the 3 XYB LF channels (ISO/IEC 18181-1 Annex I.2).
#[derive(Debug, Clone, PartialEq)]
pub struct LfChannelDequantization {
    /// Whether all weights use specification defaults.
    pub all_default: bool,
    /// LF dequantization weight for X (`default = 1/32`).
    pub m_x_lf: f32,
    /// LF dequantization weight for Y (`default = 1/4`).
    pub m_y_lf: f32,
    /// LF dequantization weight for B (`default = 1/2`).
    pub m_b_lf: f32,
}

impl Default for LfChannelDequantization {
    fn default() -> Self {
        Self {
            all_default: true,
            m_x_lf: 1.0 / 32.0,
            m_y_lf: 1.0 / 4.0,
            m_b_lf: 1.0 / 2.0,
        }
    }
}

impl LfChannelDequantization {
    /// Parses `LfChannelDequantization` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        let m_x_lf = br.read_f16()?;
        let m_y_lf = br.read_f16()?;
        let m_b_lf = br.read_f16()?;
        if [m_x_lf / 128.0, m_y_lf / 128.0, m_b_lf / 128.0]
            .into_iter()
            .any(|v| v < 1e-8)
        {
            return Err(JxlError::InvalidBitstream(
                "Modular dequant weight is too small",
            ));
        }
        Ok(Self {
            all_default: false,
            m_x_lf,
            m_y_lf,
            m_b_lf,
        })
    }

    /// Writes `LfChannelDequantization` to the bitstream.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_bool(self.all_default);
        if !self.all_default {
            bw.write_f16(self.m_x_lf)?;
            bw.write_f16(self.m_y_lf)?;
            bw.write_f16(self.m_b_lf)?;
        }
        Ok(())
    }
}

/// Global quantizer scale parameters for VarDCT frames (ISO/IEC 18181-1 Annex I.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantizer {
    /// Global quantizer scale (`1..=73728`).
    pub global_scale: u32,
    /// LF quantizer multiplier (`1..=65536`).
    pub quant_lf: u32,
}

impl Quantizer {
    /// Parses `Quantizer` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        Ok(Self {
            global_scale: br.read_u32(
                U32Specifier::u(1, 11),
                U32Specifier::u(2049, 11),
                U32Specifier::u(4097, 12),
                U32Specifier::u(8193, 16),
            )?,
            quant_lf: br.read_u32(
                U32Specifier::c(16),
                U32Specifier::u(1, 5),
                U32Specifier::u(1, 8),
                U32Specifier::u(1, 16),
            )?,
        })
    }

    /// Writes `Quantizer` to the bitstream.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_u32(
            self.global_scale,
            U32Specifier::u(1, 11),
            U32Specifier::u(2049, 11),
            U32Specifier::u(4097, 12),
            U32Specifier::u(8193, 16),
        )?;
        bw.write_u32(
            self.quant_lf,
            U32Specifier::c(16),
            U32Specifier::u(1, 5),
            U32Specifier::u(1, 8),
            U32Specifier::u(1, 16),
        )?;
        Ok(())
    }
}

/// Chroma-from-luma (CfL) channel correlation parameters (ISO/IEC 18181-1 Annex I.2).
#[derive(Debug, Clone, PartialEq)]
pub struct LfChannelCorrelation {
    /// Whether default correlation parameters are used.
    pub all_default: bool,
    /// Denominator scale factor (`default = 84`).
    pub colour_factor: u32,
    /// Base X-from-Y correlation (`default = 0.0`).
    pub base_correlation_x: f32,
    /// Base B-from-Y correlation (`default = 1.0`).
    pub base_correlation_b: f32,
    /// LF X correlation factor (`default = 128`).
    pub x_factor_lf: u32,
    /// LF B correlation factor (`default = 128`).
    pub b_factor_lf: u32,
}

impl Default for LfChannelCorrelation {
    fn default() -> Self {
        Self {
            all_default: true,
            colour_factor: 84,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        }
    }
}

impl LfChannelCorrelation {
    /// Parses `LfChannelCorrelation` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let all_default = br.read_bool()?;
        if all_default {
            return Ok(Self::default());
        }
        Ok(Self {
            all_default: false,
            colour_factor: br.read_u32(
                U32Specifier::c(84),
                U32Specifier::c(256),
                U32Specifier::u(2, 8),
                U32Specifier::u(258, 16),
            )?,
            base_correlation_x: br.read_f16()?,
            base_correlation_b: br.read_f16()?,
            x_factor_lf: br.read_bits(8)?,
            b_factor_lf: br.read_bits(8)?,
        })
    }

    /// Writes `LfChannelCorrelation` to the bitstream.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_bool(self.all_default);
        if !self.all_default {
            bw.write_u32(
                self.colour_factor,
                U32Specifier::c(84),
                U32Specifier::c(256),
                U32Specifier::u(2, 8),
                U32Specifier::u(258, 16),
            )?;
            bw.write_f16(self.base_correlation_x)?;
            bw.write_f16(self.base_correlation_b)?;
            bw.write_bits(8, self.x_factor_lf);
            bw.write_bits(8, self.b_factor_lf);
        }
        Ok(())
    }
}

/// Block context clustering configuration for HF entropy coding (ISO/IEC 18181-1 Annex I.3).
#[derive(Debug, Clone)]
pub struct HfBlockContext {
    /// Quantizer factor thresholds.
    pub qf_thresholds: Vec<u32>,
    /// LF coefficient thresholds per channel.
    pub lf_thresholds: [Vec<i32>; 3],
    /// Mapping from `(ch * 13 + order_id, hf_idx, lf_idx)` to block cluster index.
    pub block_ctx_map: Vec<u8>,
    /// Total number of distinct block clusters (`<= 16`).
    pub num_block_clusters: u32,
}

impl Default for HfBlockContext {
    fn default() -> Self {
        Self {
            qf_thresholds: Vec::new(),
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            block_ctx_map: vec![
                0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14,
                14, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
            ],
            num_block_clusters: 15,
        }
    }
}

impl HfBlockContext {
    /// Parses `HfBlockContext` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if br.read_bool()? {
            return Ok(Self::default());
        }
        let mut qf_thresholds = Vec::new();
        let mut lf_thresholds = [Vec::new(), Vec::new(), Vec::new()];
        let mut bsize = 1u32;
        for thr in &mut lf_thresholds {
            let n = br.read_bits(4)?;
            bsize *= n + 1;
            for _ in 0..n {
                let t = unpack_signed(br.read_u32(
                    U32Specifier::u(0, 4),
                    U32Specifier::u(16, 8),
                    U32Specifier::u(272, 16),
                    U32Specifier::u(65808, 32),
                )?);
                thr.push(t);
            }
        }
        let n_qf = br.read_bits(4)?;
        bsize *= n_qf + 1;
        for _ in 0..n_qf {
            let t = br.read_u32(
                U32Specifier::u(0, 2),
                U32Specifier::u(4, 3),
                U32Specifier::u(12, 5),
                U32Specifier::u(44, 8),
            )?;
            qf_thresholds.push(1 + t);
        }
        let (num_clusters, block_ctx_map) = read_clusters(br, bsize * 39)?;
        Ok(Self {
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters: num_clusters,
        })
    }

    /// Writes a single-cluster `HfBlockContext` (`num_block_clusters = 1`) for compact encoding.
    pub fn write_single_cluster(bw: &mut BitWriter) {
        bw.write_bool(false); // use_default = false
        for _ in 0..3 {
            bw.write_bits(4, 0); // num_lf_thresholds = 0
        }
        bw.write_bits(4, 0); // num_qf_thresholds = 0
        // Simple cluster map with nbits = 0 -> all 39 contexts map to cluster 0
        bw.write_bool(true);
        bw.write_bits(2, 0);
    }
}

/// Varblock transform types (ISO/IEC 18181-1 Table I.1).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum TransformType {
    /// Standard 8x8 DCT.
    #[default]
    Dct8 = 0,
    /// Hornuss 8x8 transform.
    Hornuss = 1,
    /// 2x2 DCT sub-blocks inside 8x8.
    Dct2 = 2,
    /// 4x4 DCT sub-blocks inside 8x8.
    Dct4 = 3,
    /// 16x16 DCT.
    Dct16 = 4,
    /// 32x32 DCT.
    Dct32 = 5,
    /// 16x8 DCT (width 8, height 16).
    Dct16x8 = 6,
    /// 8x16 DCT (width 16, height 8).
    Dct8x16 = 7,
    /// 32x8 DCT (width 8, height 32).
    Dct32x8 = 8,
    /// 8x32 DCT (width 32, height 8).
    Dct8x32 = 9,
    /// 32x16 DCT (width 16, height 32).
    Dct32x16 = 10,
    /// 16x32 DCT (width 32, height 16).
    Dct16x32 = 11,
    /// 4x8 DCT inside 8x8.
    Dct4x8 = 12,
    /// 8x4 DCT inside 8x8.
    Dct8x4 = 13,
    /// Asymmetric Frequency Variant corner 0.
    Afv0 = 14,
    /// Asymmetric Frequency Variant corner 1.
    Afv1 = 15,
    /// Asymmetric Frequency Variant corner 2.
    Afv2 = 16,
    /// Asymmetric Frequency Variant corner 3.
    Afv3 = 17,
    /// 64x64 DCT.
    Dct64 = 18,
    /// 64x32 DCT.
    Dct64x32 = 19,
    /// 32x64 DCT.
    Dct32x64 = 20,
    /// 128x128 DCT.
    Dct128 = 21,
    /// 128x64 DCT.
    Dct128x64 = 22,
    /// 64x128 DCT.
    Dct64x128 = 23,
    /// 256x256 DCT.
    Dct256 = 24,
    /// 256x128 DCT.
    Dct256x128 = 25,
    /// 128x256 DCT.
    Dct128x256 = 26,
}

impl TransformType {
    /// Converts a raw byte (`0..=26`) into a [`TransformType`].
    pub fn from_u8(value: u8) -> Result<Self, JxlError> {
        Ok(match value {
            0 => Self::Dct8,
            1 => Self::Hornuss,
            2 => Self::Dct2,
            3 => Self::Dct4,
            4 => Self::Dct16,
            5 => Self::Dct32,
            6 => Self::Dct16x8,
            7 => Self::Dct8x16,
            8 => Self::Dct32x8,
            9 => Self::Dct8x32,
            10 => Self::Dct32x16,
            11 => Self::Dct16x32,
            12 => Self::Dct4x8,
            13 => Self::Dct8x4,
            14 => Self::Afv0,
            15 => Self::Afv1,
            16 => Self::Afv2,
            17 => Self::Afv3,
            18 => Self::Dct64,
            19 => Self::Dct64x32,
            20 => Self::Dct32x64,
            21 => Self::Dct128,
            22 => Self::Dct128x64,
            23 => Self::Dct64x128,
            24 => Self::Dct256,
            25 => Self::Dct256x128,
            26 => Self::Dct128x256,
            _ => {
                return Err(JxlError::InvalidBitstream("invalid TransformType enum"));
            }
        })
    }

    /// Size of the varblock in 8x8 blocks `(bw, bh)`.
    #[must_use]
    #[inline]
    pub const fn dct_select_size(self) -> (u32, u32) {
        use TransformType::*;
        match self {
            Dct8 | Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => (1, 1),
            Dct16 => (2, 2),
            Dct32 => (4, 4),
            Dct16x8 => (1, 2),
            Dct8x16 => (2, 1),
            Dct32x8 => (1, 4),
            Dct8x32 => (4, 1),
            Dct32x16 => (2, 4),
            Dct16x32 => (4, 2),
            Dct64 => (8, 8),
            Dct64x32 => (4, 8),
            Dct32x64 => (8, 4),
            Dct128 => (16, 16),
            Dct128x64 => (8, 16),
            Dct64x128 => (16, 8),
            Dct256 => (32, 32),
            Dct256x128 => (16, 32),
            Dct128x256 => (32, 16),
        }
    }

    /// Dequantization matrix parameter index (`0..=16`).
    #[must_use]
    #[inline]
    pub const fn dequant_matrix_param_index(self) -> usize {
        use TransformType::*;
        match self {
            Dct8 => 0,
            Hornuss => 1,
            Dct2 => 2,
            Dct4 => 3,
            Dct16 => 4,
            Dct32 => 5,
            Dct16x8 | Dct8x16 => 6,
            Dct32x8 | Dct8x32 => 7,
            Dct32x16 | Dct16x32 => 8,
            Dct4x8 | Dct8x4 => 9,
            Afv0 | Afv1 | Afv2 | Afv3 => 10,
            Dct64 => 11,
            Dct64x32 | Dct32x64 => 12,
            Dct128 => 13,
            Dct128x64 | Dct64x128 => 14,
            Dct256 => 15,
            Dct256x128 | Dct128x256 => 16,
        }
    }

    /// Canonical dequantization matrix dimensions `(width, height)` in samples.
    #[must_use]
    #[inline]
    pub const fn dequant_matrix_size(self) -> (u32, u32) {
        use TransformType::*;
        match self {
            Dct8 | Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => (8, 8),
            Dct16 => (16, 16),
            Dct32 => (32, 32),
            Dct16x8 | Dct8x16 => (16, 8),
            Dct32x8 | Dct8x32 => (32, 8),
            Dct32x16 | Dct16x32 => (32, 16),
            Dct64 => (64, 64),
            Dct64x32 | Dct32x64 => (64, 32),
            Dct128 => (128, 128),
            Dct128x64 | Dct64x128 => (128, 64),
            Dct256 => (256, 256),
            Dct256x128 | Dct128x256 => (256, 128),
        }
    }

    /// Coefficient scan order ID (`0..=12`).
    #[must_use]
    #[inline]
    pub const fn order_id(self) -> usize {
        use TransformType::*;
        match self {
            Dct8 => 0,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => 1,
            Dct16 => 2,
            Dct32 => 3,
            Dct16x8 | Dct8x16 => 4,
            Dct32x8 | Dct8x32 => 5,
            Dct32x16 | Dct16x32 => 6,
            Dct64 => 7,
            Dct64x32 | Dct32x64 => 8,
            Dct128 => 9,
            Dct128x64 | Dct64x128 => 10,
            Dct256 => 11,
            Dct256x128 | Dct128x256 => 12,
        }
    }

    /// Whether natural-order coordinates and dequantization matrices are transposed.
    #[must_use]
    #[inline]
    pub const fn need_transpose(self) -> bool {
        use TransformType::*;
        if matches!(
            self,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3
        ) {
            false
        } else {
            let (w, h) = self.dct_select_size();
            h >= w
        }
    }
}

/// Set of dequantization matrices for all 17 transform families (ISO/IEC 18181-1 Annex I.4).
#[derive(Debug, Clone)]
pub struct DequantMatrixSet {
    matrices: Vec<[Vec<f32>; 3]>,
    matrices_tr: Vec<[Vec<f32>; 3]>,
}

const DCT_SELECT_LIST: [TransformType; 17] = [
    TransformType::Dct8,
    TransformType::Hornuss,
    TransformType::Dct2,
    TransformType::Dct4,
    TransformType::Dct16,
    TransformType::Dct32,
    TransformType::Dct8x16,
    TransformType::Dct8x32,
    TransformType::Dct16x32,
    TransformType::Dct4x8,
    TransformType::Afv0,
    TransformType::Dct64,
    TransformType::Dct32x64,
    TransformType::Dct128,
    TransformType::Dct64x128,
    TransformType::Dct256,
    TransformType::Dct128x256,
];

impl Default for DequantMatrixSet {
    fn default() -> Self {
        let matrices: Vec<[Vec<f32>; 3]> = DCT_SELECT_LIST
            .into_iter()
            .map(|t| default_dequant_matrix(t).expect("default dequant matrix is valid"))
            .collect();
        let matrices_tr = transpose_matrices(&matrices);
        Self {
            matrices,
            matrices_tr,
        }
    }
}

fn transpose_matrices(matrices: &[[Vec<f32>; 3]]) -> Vec<[Vec<f32>; 3]> {
    matrices
        .iter()
        .zip(DCT_SELECT_LIST)
        .map(|(matrix, dct_select)| {
            std::array::from_fn(|c| {
                let m = &matrix[c];
                let (w, h) = dct_select.dequant_matrix_size();
                let w = w as usize;
                let h = h as usize;
                let mut out = vec![0.0f32; m.len()];
                for (idx, val) in out.iter_mut().enumerate() {
                    let mx = idx % h;
                    let my = idx / h;
                    *val = m[mx * w + my];
                }
                out
            })
        })
        .collect()
}

impl DequantMatrixSet {
    /// Parses `DequantMatrixSet` from the bitstream.
    pub fn parse(
        br: &mut BitReader<'_>,
        bit_depth: u32,
        num_lf_groups: u32,
        global_ma_config: Option<&MaConfig>,
    ) -> Result<Self, JxlError> {
        if br.read_bool()? {
            return Ok(Self::default());
        }
        let stream_base = 1 + num_lf_groups * 3;
        let mut matrices = Vec::with_capacity(17);
        for (idx, dct_select) in DCT_SELECT_LIST.into_iter().enumerate() {
            matrices.push(parse_one_dequant_matrix(
                br,
                dct_select,
                bit_depth,
                stream_base + idx as u32,
                global_ma_config,
            )?);
        }
        let matrices_tr = transpose_matrices(&matrices);
        Ok(Self {
            matrices,
            matrices_tr,
        })
    }

    /// Returns the untransposed dequantization matrix for `(channel, dct_select)`.
    #[must_use]
    #[inline]
    pub fn get(&self, channel: usize, dct_select: TransformType) -> &[f32] {
        &self.matrices[dct_select.dequant_matrix_param_index()][channel]
    }

    /// Returns the transposed dequantization matrix for `(channel, dct_select)`.
    #[must_use]
    #[inline]
    pub fn get_transposed(&self, channel: usize, dct_select: TransformType) -> &[f32] {
        &self.matrices_tr[dct_select.dequant_matrix_param_index()][channel]
    }
}

fn interpolate_bands(pos: f32, max: f32, bands: &[f32]) -> f32 {
    if let &[val] = bands {
        return val;
    }
    let scaled_pos = pos * (bands.len() - 1) as f32 / max;
    let scaled_idx = (scaled_pos as usize).min(bands.len() - 2);
    let frac = scaled_pos - scaled_idx as f32;
    let a = bands[scaled_idx];
    let b = bands[scaled_idx + 1];
    a * (b / a).powf(frac)
}

fn band_mult(x: f32) -> f32 {
    if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
}

fn dct_quant_weights(params: &[f32], width: u32, height: u32) -> Result<Vec<f32>, JxlError> {
    let mut bands = Vec::with_capacity(params.len());
    let mut last_band = params[0];
    bands.push(last_band);
    for &val in &params[1..] {
        let band = last_band * band_mult(val);
        if band <= 0.0 {
            return Err(JxlError::InvalidBitstream("DCT dequant matrix band <= 0"));
        }
        bands.push(band);
        last_band = band;
    }
    let mut ret = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            let dist = (dx * dx + dy * dy).sqrt();
            ret.push(interpolate_bands(
                dist,
                std::f32::consts::SQRT_2 + 1e-6,
                &bands,
            ));
        }
    }
    Ok(ret)
}

#[rustfmt::skip]
fn default_dequant_matrix(dct_select: TransformType) -> Result<[Vec<f32>; 3], JxlError> {
    const SEQ_A: [f32; 7] = [-1.025, -0.78, -0.65012, -0.19041574, -0.20819396, -0.421064, -0.32733846];
    const SEQ_B: [f32; 7] = [-0.30419582, -0.36330363, -0.3566038, -0.34430745, -0.33699593, -0.30180866, -0.27321684];
    const SEQ_C: [f32; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
    const DCT4X8_PARAMS: [[f32; 4]; 3] = [
        [2198.0505, -0.96269625, -0.7619425, -0.65511405],
        [764.36554, -0.926302, -0.967523, -0.2784529],
        [527.10754, -1.4594386, -1.4500821, -1.5843723],
    ];
    const DCT4_PARAMS: [[f32; 4]; 3] = [
        [2200.0, 0.0, 0.0, 0.0],
        [392.0, 0.0, 0.0, 0.0],
        [112.0, -0.25, -0.25, -0.5],
    ];

    let make_common = |a: f32, b: f32, c: f32| -> [Vec<f32>; 3] {
        let mut va = vec![a];
        va.extend_from_slice(&SEQ_A);
        let mut vb = vec![b];
        vb.extend_from_slice(&SEQ_B);
        let mut vc = vec![c];
        vc.extend_from_slice(&SEQ_C);
        [va, vb, vc]
    };

    use TransformType::*;
    let mut weights = match dct_select {
        Dct8 => {
            let p = [
                vec![3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
                vec![560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
                vec![512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
            ];
            [
                dct_quant_weights(&p[0], 8, 8)?,
                dct_quant_weights(&p[1], 8, 8)?,
                dct_quant_weights(&p[2], 8, 8)?,
            ]
        }
        Hornuss => {
            let p = [
                [280.0, 3160.0, 3160.0],
                [60.0, 864.0, 864.0],
                [18.0, 200.0, 200.0],
            ];
            p.map(|params| {
                let mut ret = vec![params[0]; 64];
                ret[0] = 1.0;
                ret[1] = params[1];
                ret[8] = params[1];
                ret[9] = params[2];
                ret
            })
        }
        Dct2 => {
            let p = [
                [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
                [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
                [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
            ];
            p.map(|params| {
                let mut ret = vec![0.0f32; 64];
                ret[0] = 1.0;
                for (idx, val) in params.into_iter().enumerate() {
                    let shift = idx / 2;
                    let dim = 1usize << shift;
                    if idx % 2 == 0 {
                        for y in 0..dim {
                            for x in dim..dim * 2 {
                                ret[y * 8 + x] = val;
                                ret[x * 8 + y] = val;
                            }
                        }
                    } else {
                        for y in dim..dim * 2 {
                            for x in dim..dim * 2 {
                                ret[y * 8 + x] = val;
                            }
                        }
                    }
                }
                ret
            })
        }
        Dct4 => build_dct4_weights([[1.0; 2]; 3], DCT4_PARAMS.map(|v| v.to_vec()))?,
        Dct4x8 | Dct8x4 => build_dct4x8_weights([[1.0]; 3], DCT4X8_PARAMS.map(|v| v.to_vec()))?,
        Afv0 | Afv1 | Afv2 | Afv3 => build_afv_weights(
            [
                [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
                [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
                [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
            ],
            DCT4X8_PARAMS.map(|v| v.to_vec()),
            DCT4_PARAMS.map(|v| v.to_vec()),
        )?,
        other => {
            let p = match other {
                Dct16 => [
                    vec![8996.873, -1.3000778, -0.4942453, -0.43909377, -0.6350102, -0.9017726, -1.6162099],
                    vec![3191.4836, -0.67424583, -0.80745816, -0.4492584, -0.3586544, -0.3132239, -0.37615025],
                    vec![1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914],
                ],
                Dct32 => [
                    vec![15718.408, -1.025, -0.98, -0.9012, -0.4, -0.48819396, -0.421064, -0.27],
                    vec![7305.7637, -0.8041958, -0.76330364, -0.5566038, -0.49785304, -0.43699592, -0.40180868, -0.27321684],
                    vec![3803.5317, -3.0607336, -2.041327, -2.023565, -0.54953897, -0.4, -0.4, -0.3],
                ],
                Dct8x16 | Dct16x8 => [
                    vec![7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
                    vec![1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
                    vec![506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
                ],
                Dct8x32 | Dct32x8 => [
                    vec![16283.249, -1.7812846, -1.6309059, -1.0382179, -0.85, -0.7, -0.9, -1.2360638],
                    vec![5089.1577, -0.3200494, -0.3536285, -0.3034, -0.61, -0.5, -0.5, -0.6],
                    vec![3397.7761, -0.32132736, -0.3450762, -0.7034, -0.9, -1.0, -1.0, -1.1754606],
                ],
                Dct16x32 | Dct32x16 => [
                    vec![13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6],
                    vec![4798.964, -0.6112531, -0.8377079, -0.7901486, -0.26927274, -0.38272768, -0.22924222, -0.20719099],
                    vec![1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5],
                ],
                Dct64 => make_common(23966.166, 8380.191, 4493.024),
                Dct32x64 | Dct64x32 => make_common(15358.898, 5597.3604, 2919.9617),
                Dct128 => make_common(47932.332, 16760.383, 8986.048),
                Dct64x128 | Dct128x64 => make_common(30717.797, 11194.721, 5839.9233),
                Dct256 => make_common(95864.664, 33520.766, 17972.096),
                Dct128x256 | Dct256x128 => make_common(61435.594, 24209.441, 12979.847),
                _ => unreachable!(),
            };
            let (w, h) = other.dequant_matrix_size();
            [
                dct_quant_weights(&p[0], w, h)?,
                dct_quant_weights(&p[1], w, h)?,
                dct_quant_weights(&p[2], w, h)?,
            ]
        }
    };

    for w in weights.iter_mut().flatten() {
        *w = 1.0 / *w;
        if *w >= 1e8 || *w <= 0.0 {
            return Err(JxlError::InvalidBitstream("Invalid dequant weight"));
        }
    }
    Ok(weights)
}

fn build_dct4_weights(
    params: [[f32; 2]; 3],
    dct_params: [Vec<f32>; 3],
) -> Result<[Vec<f32>; 3], JxlError> {
    let mut ret = [Vec::new(), Vec::new(), Vec::new()];
    for (out, (p, dp)) in ret.iter_mut().zip(params.into_iter().zip(dct_params)) {
        let mat = dct_quant_weights(&dp, 4, 4)?;
        *out = vec![0.0f32; 64];
        for y in 0..4 {
            for x in 0..4 {
                let v = mat[y * 4 + x];
                out[y * 16 + x * 2] = v;
                out[y * 16 + x * 2 + 1] = v;
                out[(y * 2 + 1) * 8 + x * 2] = v;
                out[(y * 2 + 1) * 8 + x * 2 + 1] = v;
            }
        }
        out[1] /= p[0];
        out[8] /= p[0];
        out[9] /= p[1];
    }
    Ok(ret)
}

fn build_dct4x8_weights(
    params: [[f32; 1]; 3],
    dct_params: [Vec<f32>; 3],
) -> Result<[Vec<f32>; 3], JxlError> {
    let mut ret = [Vec::new(), Vec::new(), Vec::new()];
    for (out, (p, dp)) in ret.iter_mut().zip(params.into_iter().zip(dct_params)) {
        let mat = dct_quant_weights(&dp, 8, 4)?;
        *out = mat
            .as_chunks::<8>()
            .0
            .iter()
            .flat_map(|v| [v, v])
            .flatten()
            .copied()
            .collect();
        out[8] /= p[0];
    }
    Ok(ret)
}

fn build_afv_weights(
    params: [[f32; 9]; 3],
    dct_params: [Vec<f32>; 3],
    dct4x4_params: [Vec<f32>; 3],
) -> Result<[Vec<f32>; 3], JxlError> {
    const FREQS: [f32; 16] = [
        0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0, 7.275749,
        10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
    ];
    const FREQ_LO: f32 = FREQS[2];
    const FREQ_HI: f32 = FREQS[15];

    let mut ret = [Vec::new(), Vec::new(), Vec::new()];
    for (out, ((p, dp8), dp4)) in ret
        .iter_mut()
        .zip(params.into_iter().zip(dct_params).zip(dct4x4_params))
    {
        let w4x8 = dct_quant_weights(&dp8, 8, 4)?;
        let w4x4 = dct_quant_weights(&dp4, 4, 4)?;
        let mut bands = [p[5], 0.0, 0.0, 0.0];
        let mut prev = bands[0];
        for (b, &param) in bands[1..].iter_mut().zip(&p[6..]) {
            *b = prev * band_mult(param);
            prev = *b;
        }
        *out = vec![0.0f32; 64];
        for y in 0..4 {
            for x in 0..4 {
                out[16 * y + 2 * x] = match (x, y) {
                    (0, 0) => 1.0,
                    (0, 1) => p[2],
                    (1, 0) => p[3],
                    (1, 1) => p[4],
                    (x, y) => interpolate_bands(
                        FREQS[y * 4 + x] - FREQ_LO,
                        FREQ_HI - FREQ_LO + 1e-6,
                        &bands,
                    ),
                };
            }
        }
        for (y, ((rows, r8), r4)) in out
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(w4x8.as_chunks::<8>().0)
            .zip(w4x4.as_chunks::<4>().0)
            .enumerate()
        {
            let (row0, row1) = rows.split_at_mut(8);
            for (x, (w, &dw)) in row1.iter_mut().zip(r8).enumerate() {
                *w = if y == 0 && x == 0 { p[0] } else { dw };
            }
            for (x, (pair, &dw)) in row0.as_chunks_mut::<2>().0.iter_mut().zip(r4).enumerate() {
                pair[1] = if y == 0 && x == 0 { p[1] } else { dw };
            }
        }
    }
    Ok(ret)
}

fn parse_one_dequant_matrix(
    br: &mut BitReader<'_>,
    dct_select: TransformType,
    bit_depth: u32,
    stream_index: u32,
    global_ma_config: Option<&MaConfig>,
) -> Result<[Vec<f32>; 3], JxlError> {
    fn read_fixed<const N: usize>(br: &mut BitReader<'_>) -> Result<[[f32; N]; 3], JxlError> {
        let mut out = [[0.0f32; N]; 3];
        for v in out.iter_mut().flatten() {
            *v = br.read_f16()?;
        }
        Ok(out)
    }

    fn read_dct_params(br: &mut BitReader<'_>) -> Result<[Vec<f32>; 3], JxlError> {
        let n = br.read_bits(4)? as usize + 1;
        let mut p = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        for v in p.iter_mut().flatten() {
            *v = br.read_f16()?;
        }
        for c in &mut p {
            c[0] *= 64.0;
        }
        Ok(p)
    }

    let mode = br.read_bits(3)?;
    if mode == 0 {
        return default_dequant_matrix(dct_select);
    }
    if (1..=5).contains(&mode)
        && !matches!(
            dct_select.dequant_matrix_param_index(),
            0 | 1 | 2 | 3 | 9 | 10
        )
    {
        return Err(JxlError::InvalidBitstream(
            "Invalid encoding mode for DctSelect in DequantMatrixSet",
        ));
    }

    let need_recip = mode != 7;
    let mut weights = match mode {
        1 => {
            let p = read_fixed::<3>(br)?;
            p.map(|params| {
                let mut ret = vec![params[0]; 64];
                ret[0] = 1.0;
                ret[1] = params[1];
                ret[8] = params[1];
                ret[9] = params[2];
                ret
            })
        }
        2 => {
            let p = read_fixed::<6>(br)?;
            p.map(|params| {
                let mut ret = vec![0.0f32; 64];
                ret[0] = 1.0;
                for (idx, val) in params.into_iter().enumerate() {
                    let shift = idx / 2;
                    let dim = 1usize << shift;
                    if idx % 2 == 0 {
                        for y in 0..dim {
                            for x in dim..dim * 2 {
                                ret[y * 8 + x] = val;
                                ret[x * 8 + y] = val;
                            }
                        }
                    } else {
                        for y in dim..dim * 2 {
                            for x in dim..dim * 2 {
                                ret[y * 8 + x] = val;
                            }
                        }
                    }
                }
                ret
            })
        }
        3 => build_dct4_weights(read_fixed::<2>(br)?, read_dct_params(br)?)?,
        4 => build_dct4x8_weights(read_fixed::<1>(br)?, read_dct_params(br)?)?,
        5 => {
            let mut p = read_fixed::<9>(br)?;
            for row in &mut p {
                for v in &mut row[..6] {
                    *v *= 64.0;
                }
            }
            build_afv_weights(p, read_dct_params(br)?, read_dct_params(br)?)?
        }
        6 => {
            let p = read_dct_params(br)?;
            let (w, h) = dct_select.dequant_matrix_size();
            [
                dct_quant_weights(&p[0], w, h)?,
                dct_quant_weights(&p[1], w, h)?,
                dct_quant_weights(&p[2], w, h)?,
            ]
        }
        7 => {
            let (w, h) = dct_select.dequant_matrix_size();
            let denom = br.read_f16()?;
            let ch_info = vec![ModularChannelInfo::new(w, h, ChannelShift::from_shift(0)); 3];
            let grids =
                decode_modular_subimage(br, ch_info, bit_depth, global_ma_config, stream_index)?;
            [0usize, 1, 2].map(|c| grids[c].data.iter().map(|&v| v as f32 * denom).collect())
        }
        _ => unreachable!(),
    };

    if need_recip {
        for w in weights.iter_mut().flatten() {
            *w = 1.0 / *w;
        }
    }
    for w in weights.iter().flatten() {
        if *w >= 1e8 || *w <= 0.0 {
            return Err(JxlError::InvalidBitstream(
                "Dequant matrix element out of range",
            ));
        }
    }
    Ok(weights)
}

/// Block placement entry inside an `HfMetadata` block map.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BlockInfo {
    /// Unassigned block slot.
    #[default]
    Uninit,
    /// Non-root block covered by a multi-8x8 varblock.
    Occupied,
    /// Top-left 8x8 block of a varblock.
    Data {
        /// Transform type for this varblock.
        dct_select: TransformType,
        /// Per-varblock quantizer multiplier (`>= 1`).
        hf_mul: i32,
    },
}

impl BlockInfo {
    /// Returns true if this 8x8 block slot has already been assigned.
    #[must_use]
    #[inline]
    pub const fn is_occupied(self) -> bool {
        !matches!(self, Self::Uninit)
    }
}

/// Decoded `LfCoeff` and `HfMetadata` for a single LF group.
#[derive(Debug, Clone)]
pub struct DecodedLfGroup {
    /// Extra precision shift (`0..=3`) for quantized LF coefficients.
    pub extra_precision: u8,
    /// Quantized LF channels in `[Y, X, B]` order (`channel 0 = Y, 1 = X, 2 = B`).
    pub lf_quant: Option<[ModularGrid; 3]>,
    /// Chroma-from-luma correlation grid for X (`64x64` pixel blocks).
    pub x_from_y: ModularGrid,
    /// Chroma-from-luma correlation grid for B (`64x64` pixel blocks).
    pub b_from_y: ModularGrid,
    /// Width of `block_info` in 8x8 blocks.
    pub bw: usize,
    /// Height of `block_info` in 8x8 blocks.
    pub bh: usize,
    /// Row-major `bw * bh` varblock map.
    pub block_info: Vec<BlockInfo>,
    /// Row-major `bw * bh` EPF sigma grid.
    pub epf_sigma: Vec<f32>,
}

impl DecodedLfGroup {
    /// Parses `LfCoeff` and `HfMetadata` for `lf_group_idx` (and decodes any LF-group Modular tiles).
    #[allow(clippy::too_many_arguments)]
    pub fn parse(
        br: &mut BitReader<'_>,
        frame_header: &FrameHeader,
        quantizer: &Quantizer,
        global_ma_config: Option<&MaConfig>,
        gmodular: &mut super::modular::GlobalModularState,
        lf_group_idx: u32,
    ) -> Result<Self, JxlError> {
        let (lf_width, lf_height) = frame_header.lf_group_size_for(lf_group_idx);
        let bits_per_sample = gmodular.bit_depth;

        let mut extra_precision = 0u8;
        let lf_quant = if !frame_header.flags.use_lf_frame() {
            extra_precision = br.read_bits(2)? as u8;
            let w = lf_width.div_ceil(8);
            let h = lf_height.div_ceil(8);
            let channels = [1usize, 0, 2]
                .into_iter()
                .map(|idx| {
                    let shift =
                        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx);
                    ModularChannelInfo::new(w, h, shift)
                })
                .collect();
            let mut grids = decode_modular_subimage(
                br,
                channels,
                bits_per_sample,
                global_ma_config,
                1 + lf_group_idx,
            )?;
            let g2 = grids.pop().unwrap();
            let g1 = grids.pop().unwrap();
            let g0 = grids.pop().unwrap();
            Some([g0, g1, g2])
        } else {
            None
        };

        gmodular.decode_lf_group(br, lf_group_idx, frame_header.num_lf_groups())?;

        let mut bw = lf_width.div_ceil(8) as usize;
        let mut bh = lf_height.div_ceil(8) as usize;
        let h_upsample = frame_header
            .jpeg_upsampling
            .into_iter()
            .any(|j| j == 1 || j == 2);
        let v_upsample = frame_header
            .jpeg_upsampling
            .into_iter()
            .any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = bw.div_ceil(2) * 2;
        }
        if v_upsample {
            bh = bh.div_ceil(2) * 2;
        }

        let nb_blocks_bits = (bw * bh).next_power_of_two().trailing_zeros() as usize;
        let nb_blocks = 1 + br.read_bits(nb_blocks_bits)?;

        let hf_meta_channels = vec![
            ModularChannelInfo::new(
                lf_width.div_ceil(64),
                lf_height.div_ceil(64),
                ChannelShift::from_shift(0),
            ),
            ModularChannelInfo::new(
                lf_width.div_ceil(64),
                lf_height.div_ceil(64),
                ChannelShift::from_shift(0),
            ),
            ModularChannelInfo::new(nb_blocks, 2, ChannelShift::from_shift(0)),
            ModularChannelInfo::new(bw as u32, bh as u32, ChannelShift::from_shift(0)),
        ];
        let mut meta_grids = decode_modular_subimage(
            br,
            hf_meta_channels,
            bits_per_sample,
            global_ma_config,
            1 + 2 * frame_header.num_lf_groups() + lf_group_idx,
        )?;
        let sharpness = meta_grids.pop().unwrap();
        let block_info_raw = meta_grids.pop().unwrap();
        let b_from_y = meta_grids.pop().unwrap();
        let x_from_y = meta_grids.pop().unwrap();

        let epf_cfg = match &frame_header.restoration_filter.epf {
            EdgePreservingFilter::Disabled => None,
            EdgePreservingFilter::Enabled(EpfParams {
                sharp_lut, sigma, ..
            }) => Some((
                sigma.quant_mul * 65536.0 / quantizer.global_scale as f32,
                *sharp_lut,
            )),
        };

        let mut block_info = vec![BlockInfo::Uninit; bw * bh];
        let mut epf_sigma = vec![0.0f32; bw * bh];
        let mut data_idx = 0usize;
        for y in 0..bh {
            let mut x = 0usize;
            while x < bw {
                if !block_info[y * bw + x].is_occupied() {
                    if data_idx >= block_info_raw.width {
                        return Err(JxlError::InvalidBitstream(
                            "BlockInfo does not fill LF group",
                        ));
                    }
                    let dct_select = TransformType::from_u8(block_info_raw.get(data_idx, 0) as u8)?;
                    let hf_mul = block_info_raw.get(data_idx, 1) + 1;
                    if hf_mul <= 0 {
                        return Err(JxlError::InvalidBitstream("non-positive HfMul"));
                    }
                    let (dw, dh) = dct_select.dct_select_size();
                    let dw = dw as usize;
                    let dh = dh as usize;
                    if (x % 32) + dw > 32 || (y % 32) + dh > 32 || x + dw > bw || y + dh > bh {
                        return Err(JxlError::InvalidBitstream(
                            "varblock crosses pass group or LF group border",
                        ));
                    }
                    for dy in 0..dh {
                        for dx in 0..dw {
                            let idx = (y + dy) * bw + (x + dx);
                            if block_info[idx].is_occupied() {
                                return Err(JxlError::InvalidBitstream("varblocks overlap"));
                            }
                            block_info[idx] = if dx == 0 && dy == 0 {
                                BlockInfo::Data { dct_select, hf_mul }
                            } else {
                                BlockInfo::Occupied
                            };
                            if let Some((base_sigma, sharp_lut)) = epf_cfg {
                                let s = sharpness.get(x + dx, y + dy);
                                if !(0..8).contains(&s) {
                                    return Err(JxlError::InvalidBitstream(
                                        "EPF sharpness out of 0..8",
                                    ));
                                }
                                epf_sigma[idx] =
                                    (base_sigma / hf_mul as f32) * sharp_lut[s as usize];
                            }
                        }
                    }
                    data_idx += 1;
                    x += dw;
                } else {
                    x += 1;
                }
            }
        }

        Ok(Self {
            extra_precision,
            lf_quant,
            x_from_y,
            b_from_y,
            bw,
            bh,
            block_info,
            epf_sigma,
        })
    }
}

const BLOCK_SIZES: [(usize, usize); 13] = [
    (8, 8),
    (8, 8),
    (16, 16),
    (32, 32),
    (16, 8),
    (32, 8),
    (32, 16),
    (64, 64),
    (64, 32),
    (128, 128),
    (128, 64),
    (256, 256),
    (256, 128),
];

/// Computes the natural zig-zag scan order for transform block size `(bw, bh)`.
#[must_use]
pub fn compute_natural_order((bw, bh): (usize, usize)) -> Vec<(u16, u16)> {
    let y_scale = bw / bh;
    let mut ret = vec![(0u16, 0u16); bw * bh];
    let mut idx = 0usize;
    let lbw = bw / 8;
    let lbh = bh / 8;

    while idx < lbw * lbh {
        let x = idx % lbw;
        let y = idx / lbw;
        ret[idx] = (x as u16, y as u16);
        idx += 1;
    }

    for dist in 1..(2 * bw) {
        let margin = dist.saturating_sub(bw);
        for order in margin..(dist - margin) {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            if x < lbw && y < lbw {
                continue;
            }
            if y % y_scale != 0 {
                continue;
            }
            ret[idx] = (x as u16, (y / y_scale) as u16);
            idx += 1;
        }
    }
    ret
}

/// Natural scan order for 8x8 DCT blocks (`64` coordinates).
pub const DCT8_NATURAL_ORDER: [(u16, u16); 64] = {
    let mut ret = [(0u16, 0u16); 64];
    ret[0] = (0, 0);
    let mut idx = 1usize;
    let mut dist = 1usize;
    while dist < 16 {
        let margin = dist.saturating_sub(8);
        let mut order = margin;
        while order < dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            order += 1;
            if x == 0 && y == 0 {
                continue;
            }
            ret[idx] = (x as u16, y as u16);
            idx += 1;
        }
        dist += 1;
    }
    ret
};

/// HF pass configuration (coefficient permutation orders and entropy distribution).
#[derive(Debug, Clone)]
pub struct HfPass {
    /// Custom or natural scan order for each `(order_id, channel)`.
    pub orders: [[Vec<(u16, u16)>; 3]; 13],
    /// Entropy decoder template for HF coefficients in this pass.
    pub hf_dist: EntropyDecoder,
}

impl HfPass {
    /// Parses an `HfPass` from the bitstream.
    pub fn parse(
        br: &mut BitReader<'_>,
        hf_block_ctx: &HfBlockContext,
        num_hf_presets: u32,
    ) -> Result<Self, JxlError> {
        let mut used_orders = br.read_u32(
            U32Specifier::c(0x5F),
            U32Specifier::c(0x13),
            U32Specifier::c(0x00),
            U32Specifier::u(0, 13),
        )?;
        let mut perm_dec = if used_orders != 0 {
            let mut d = EntropyDecoder::parse(br, 8)?;
            d.begin(br)?;
            Some(d)
        } else {
            None
        };

        let mut orders: [[Vec<(u16, u16)>; 3]; 13] =
            std::array::from_fn(|_| [Vec::new(), Vec::new(), Vec::new()]);
        for (idx, (order_ch, &block_size)) in orders.iter_mut().zip(&BLOCK_SIZES).enumerate() {
            let nat = if idx <= 1 {
                DCT8_NATURAL_ORDER.to_vec()
            } else {
                compute_natural_order(block_size)
            };
            if used_orders & 1 != 0 {
                let dec = perm_dec.as_mut().unwrap();
                let size = (block_size.0 * block_size.1) as u32;
                let skip = size / 64;
                for ch_ord in order_ch.iter_mut() {
                    let perm = read_permutation(br, dec, size, skip)?;
                    *ch_ord = perm.into_iter().map(|i| nat[i]).collect();
                }
            } else {
                for ch_ord in order_ch.iter_mut() {
                    *ch_ord = nat.clone();
                }
            }
            used_orders >>= 1;
        }
        if let Some(dec) = perm_dec {
            dec.finalize()?;
        }

        let num_contexts = 495 * num_hf_presets * hf_block_ctx.num_block_clusters;
        let hf_dist = EntropyDecoder::parse(br, num_contexts)?;
        Ok(Self { orders, hf_dist })
    }

    /// Returns the scan order for `(order_id, channel)`.
    #[must_use]
    #[inline]
    pub fn order(&self, order_id: usize, channel: usize) -> &[(u16, u16)] {
        &self.orders[order_id][channel]
    }
}

/// Global HF data (`HfGlobal`) for a VarDCT frame.
#[derive(Debug, Clone)]
pub struct HfGlobal {
    /// Dequantization matrix set.
    pub dequant_matrices: DequantMatrixSet,
    /// Number of HF preset tables.
    pub num_hf_presets: u32,
    /// Per-pass HF decoder configurations.
    pub hf_passes: Vec<HfPass>,
}

impl HfGlobal {
    /// Parses `HfGlobal` from the bitstream.
    pub fn parse(
        br: &mut BitReader<'_>,
        bit_depth: u32,
        frame_header: &FrameHeader,
        global_ma_config: Option<&MaConfig>,
        hf_block_ctx: &HfBlockContext,
    ) -> Result<Self, JxlError> {
        let dequant_matrices = DequantMatrixSet::parse(
            br,
            bit_depth,
            frame_header.num_lf_groups(),
            global_ma_config,
        )?;
        let num_groups = frame_header.num_groups();
        let preset_bits = num_groups.next_power_of_two().trailing_zeros() as usize;
        let num_hf_presets = br.read_bits(preset_bits)? + 1;
        let mut hf_passes = Vec::with_capacity(frame_header.passes.num_passes as usize);
        for _ in 0..frame_header.passes.num_passes {
            hf_passes.push(HfPass::parse(br, hf_block_ctx, num_hf_presets)?);
        }
        Ok(Self {
            dequant_matrices,
            num_hf_presets,
            hf_passes,
        })
    }
}

const COEFF_FREQ_CONTEXT: [u32; 63] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
    20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27,
    27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152, 180,
    180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206,
];

/// Decodes quantized HF coefficients for a single `(pass_idx, group_idx)` and accumulates them into
/// `hf_coeff_grid` (`[X, Y, B]`, each of size `(width_rounded, height_rounded)`).
#[allow(clippy::too_many_arguments)]
pub fn decode_hf_pass_group(
    br: &mut BitReader<'_>,
    frame_header: &FrameHeader,
    hf_block_ctx: &HfBlockContext,
    hf_global: &HfGlobal,
    lf_group: &DecodedLfGroup,
    pass_idx: u32,
    group_idx: u32,
    hf_coeff_grid: &mut [ModularGrid; 3],
) -> Result<(), JxlError> {
    let hf_pass = &hf_global.hf_passes[pass_idx as usize];
    let coeff_shift = frame_header
        .passes
        .shift
        .get(pass_idx as usize)
        .copied()
        .unwrap_or(0);

    let group_col = group_idx % frame_header.groups_per_row();
    let group_row = group_idx / frame_header.groups_per_row();
    let lf_col = (group_col % 8) as usize;
    let lf_row = (group_row % 8) as usize;
    let group_dim_blocks = (frame_header.group_dim() / 8) as usize;

    let block_left = lf_col * group_dim_blocks;
    let block_top = lf_row * group_dim_blocks;
    let block_width = lf_group.bw.saturating_sub(block_left).min(group_dim_blocks);
    let block_height = lf_group.bh.saturating_sub(block_top).min(group_dim_blocks);

    let frame_block_left = (group_col as usize) * group_dim_blocks;
    let frame_block_top = (group_row as usize) * group_dim_blocks;

    let mut dist = hf_pass.hf_dist.clone();
    let hfp_bits = hf_global
        .num_hf_presets
        .next_power_of_two()
        .trailing_zeros() as usize;
    let hfp = br.read_bits(hfp_bits)?;
    if hfp >= hf_global.num_hf_presets {
        return Err(JxlError::InvalidBitstream(
            "selected HF preset out of bounds",
        ));
    }

    let num_block_clusters = hf_block_ctx.num_block_clusters;
    let ctx_size = (495 * num_block_clusters) as usize;
    let cluster_map = dist.cluster_map()[(ctx_size * hfp as usize)..][..ctx_size].to_vec();

    dist.begin(br)?;

    let upsampling_shifts: [ChannelShift; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let hshifts = upsampling_shifts.map(|s| s.hshift());
    let vshifts = upsampling_shifts.map(|s| s.vshift());
    let non_zeros_lens = upsampling_shifts
        .map(|s| s.shift_size((block_width as u32, block_height as u32)).0 as usize);
    let mut non_zeros_grid_row = [
        vec![0u32; non_zeros_lens[0]],
        vec![0u32; non_zeros_lens[1]],
        vec![0u32; non_zeros_lens[2]],
    ];

    let lf_thresholds = &hf_block_ctx.lf_thresholds;
    let qf_thresholds = &hf_block_ctx.qf_thresholds;
    let lf_idx_mul =
        (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);
    let hf_idx_mul = qf_thresholds.len() + 1;

    for by in 0..block_height {
        for bx in 0..block_width {
            let BlockInfo::Data {
                dct_select,
                hf_mul: qf,
            } = lf_group.block_info[(block_top + by) * lf_group.bw + (block_left + bx)]
            else {
                continue;
            };
            let (w8, h8) = dct_select.dct_select_size();
            let num_blocks = w8 * h8;
            let num_blocks_log = num_blocks.trailing_zeros();
            let order_id = dct_select.order_id();

            let lf_idx = if let Some(ref lf_quant) = lf_group.lf_quant {
                let lf_xyb = [&lf_quant[1], &lf_quant[0], &lf_quant[2]];
                let mut idx = 0usize;
                for c in [0usize, 2, 1] {
                    idx *= lf_thresholds[c].len() + 1;
                    let lx = (block_left + bx) >> hshifts[c];
                    let ly = (block_top + by) >> vshifts[c];
                    let q = lf_xyb[c].get(lx, ly);
                    for &thr in &lf_thresholds[c] {
                        if q > thr {
                            idx += 1;
                        }
                    }
                }
                idx
            } else {
                0
            };

            let mut hf_idx = 0usize;
            for &thr in qf_thresholds {
                if qf > thr as i32 {
                    hf_idx += 1;
                }
            }

            for c_iter in 0..3 {
                let ch_idx = c_iter * 13 + order_id;
                let c = [1usize, 0, 2][c_iter]; // Y, X, B

                let hs = hshifts[c];
                let vs = vshifts[c];
                let sx = bx >> hs;
                let sy = by >> vs;
                if hs != 0 || vs != 0 {
                    if (sx << hs) != bx || (sy << vs) != by {
                        continue;
                    }
                    if !matches!(
                        lf_group.block_info[(block_top + sy) * lf_group.bw + (block_left + sx)],
                        BlockInfo::Data { .. }
                    ) {
                        continue;
                    }
                }

                let map_idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = hf_block_ctx.block_ctx_map[map_idx] as u32;
                let predicted = if sy == 0 {
                    if sx == 0 {
                        32
                    } else {
                        non_zeros_grid_row[c][sx - 1]
                    }
                } else if sx == 0 {
                    non_zeros_grid_row[c][sx]
                } else {
                    (non_zeros_grid_row[c][sx] + non_zeros_grid_row[c][sx - 1] + 1) >> 1
                };
                let nz_bucket = if predicted >= 8 {
                    4 + predicted / 2
                } else {
                    predicted
                };
                let non_zeros_ctx = block_ctx + nz_bucket * num_block_clusters;
                let mut non_zeros = dist.read_varint_with_multiplier_clustered(
                    br,
                    cluster_map[non_zeros_ctx as usize],
                    0,
                )?;
                if non_zeros > (63 << num_blocks_log) {
                    return Err(JxlError::InvalidBitstream("HF non_zeros > 63 * B"));
                }

                let nz_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for dx in 0..w8 as usize {
                    non_zeros_grid_row[c][sx + dx] = nz_val;
                }
                if non_zeros == 0 {
                    continue;
                }

                let mut is_prev_nonzero = (non_zeros <= num_blocks * 4) as u32;
                let order = hf_pass.order(order_id, c);
                let coeff_ctx_base = (block_ctx * 458 + 37 * num_block_clusters) as usize;
                let coeff_clusters = &cluster_map[coeff_ctx_base..coeff_ctx_base + 458];

                let dst_bx = (frame_block_left >> hs) + sx;
                let dst_by = (frame_block_top >> vs) + sy;

                for (idx, &coord) in order[num_blocks as usize..].iter().enumerate() {
                    let nz_norm = ((non_zeros - 1) >> num_blocks_log) as usize;
                    let idx_norm = idx >> num_blocks_log;
                    let coeff_ctx =
                        ((COEFF_NUM_NONZERO_CONTEXT[nz_norm] + COEFF_FREQ_CONTEXT[idx_norm]) * 2
                            + is_prev_nonzero) as usize;
                    let cluster = *coeff_clusters
                        .get(coeff_ctx)
                        .ok_or(JxlError::InvalidBitstream("HF coeff_ctx out of bounds"))?;
                    let ucoeff = dist.read_varint_with_multiplier_clustered(br, cluster, 0)?;
                    if ucoeff == 0 {
                        is_prev_nonzero = 0;
                        continue;
                    }
                    let coeff = unpack_signed(ucoeff) << coeff_shift;
                    let (mut dx, mut dy) = coord;
                    if dct_select.need_transpose() {
                        std::mem::swap(&mut dx, &mut dy);
                    }
                    let px = dst_bx * 8 + dx as usize;
                    let py = dst_by * 8 + dy as usize;
                    let prev = hf_coeff_grid[c].get(px, py);
                    hf_coeff_grid[c].set(px, py, prev + coeff);

                    is_prev_nonzero = 1;
                    non_zeros -= 1;
                    if non_zeros == 0 {
                        break;
                    }
                }
            }
        }
    }

    dist.finalize()?;
    Ok(())
}

/// Direction of the 2D DCT.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DctDirection {
    /// Forward spatial-to-frequency transform.
    Forward,
    /// Inverse frequency-to-spatial transform.
    Inverse,
}

#[allow(clippy::excessive_precision)]
const SEC_HALF_4: [f32; 2] = [0.541196100146197, 1.3065629648763764];
#[allow(clippy::excessive_precision)]
const SEC_HALF_8: [f32; 4] = [
    0.5097955791041592,
    0.6013448869350453,
    0.8999762231364156,
    2.5629154477415055,
];
#[allow(clippy::excessive_precision)]
const SEC_HALF_16: [f32; 8] = [
    0.5024192861881557,
    0.5224986149396889,
    0.5669440348163577,
    0.6468217833599901,
    0.7881546234512502,
    1.060677685990347,
    1.7224470982383342,
    5.101148618689155,
];
#[allow(clippy::excessive_precision)]
const SEC_HALF_32: [f32; 16] = [
    0.5006029982351963,
    0.5054709598975436,
    0.5154473099226246,
    0.5310425910897841,
    0.5531038960344445,
    0.5829349682061339,
    0.6225041230356648,
    0.6748083414550057,
    0.7445362710022984,
    0.8393496454155268,
    0.9725682378619608,
    1.1694399334328847,
    1.4841646163141662,
    2.057781009953411,
    3.407608418468719,
    10.190008123548033,
];

#[allow(clippy::excessive_precision)]
const SCALE_F: [f32; 32] = [
    1.0000000000000000,
    0.9996047255830407,
    0.9984194528776054,
    0.9964458326264695,
    0.9936866130906366,
    0.9901456355893141,
    0.9858278282666936,
    0.9807391980963174,
    0.9748868211368796,
    0.9682788310563117,
    0.9609244059440204,
    0.9528337534340876,
    0.9440180941651672,
    0.9344896436056892,
    0.9242615922757944,
    0.9133480844001980,
    0.9017641950288744,
    0.8895259056651056,
    0.8766500784429904,
    0.8631544288990163,
    0.8490574973847023,
    0.8343786191696513,
    0.8191378932865928,
    0.8033561501721485,
    0.7870549181591013,
    0.7702563888779096,
    0.7529833816270532,
    0.7352593067735488,
    0.7171081282466044,
    0.6985543251889097,
    0.6796228528314652,
    0.6603391026591464,
];

#[inline]
const fn scale_f(c: usize, logb: usize) -> f32 {
    SCALE_F[c << logb]
}

fn sec_half_vec(n: usize) -> Vec<f32> {
    match n {
        4 => SEC_HALF_4.to_vec(),
        8 => SEC_HALF_8.to_vec(),
        16 => SEC_HALF_16.to_vec(),
        32 => SEC_HALF_32.to_vec(),
        _ => (0..n / 2)
            .map(|k| {
                let theta = (2 * k + 1) as f32 / (2 * n) as f32 * std::f32::consts::PI;
                theta.cos().recip() / 2.0
            })
            .collect(),
    }
}

#[inline]
fn dct4_1d(input: [f32; 4], direction: DctDirection) -> [f32; 4] {
    let sec0 = 0.5411961f32;
    let sec1 = 1.306563f32;
    if direction == DctDirection::Forward {
        let sum03 = input[0] + input[3];
        let sum12 = input[1] + input[2];
        let tmp0 = (input[0] - input[3]) * sec0;
        let tmp1 = (input[1] - input[2]) * sec1;
        let out0 = (tmp0 + tmp1) / 4.0;
        let out1 = (tmp0 - tmp1) / 4.0;
        [
            (sum03 + sum12) / 4.0,
            out0 * std::f32::consts::SQRT_2 + out1,
            (sum03 - sum12) / 4.0,
            out1,
        ]
    } else {
        let tmp0 = input[1] * std::f32::consts::SQRT_2;
        let tmp1 = input[1] + input[3];
        let out0 = (tmp0 + tmp1) * sec0;
        let out1 = (tmp0 - tmp1) * sec1;
        let sum02 = input[0] + input[2];
        let sub02 = input[0] - input[2];
        [sum02 + out0, sub02 + out1, sub02 - out1, sum02 - out0]
    }
}

/// Computes an in-place 1D DCT or IDCT on a power-of-two slice `io`.
pub fn dct_1d(io: &mut [f32], scratch: &mut [f32], direction: DctDirection) {
    let n = io.len();
    if n <= 1 {
        return;
    }
    if n == 2 {
        let t0 = io[0] + io[1];
        let t1 = io[0] - io[1];
        if direction == DctDirection::Forward {
            io[0] = t0 * 0.5;
            io[1] = t1 * 0.5;
        } else {
            io[0] = t0;
            io[1] = t1;
        }
        return;
    }
    if n == 4 {
        io.copy_from_slice(&dct4_1d([io[0], io[1], io[2], io[3]], direction));
        return;
    }
    if n == 8 {
        if direction == DctDirection::Forward {
            let in0 = [
                (io[0] + io[7]) * 0.5,
                (io[1] + io[6]) * 0.5,
                (io[2] + io[5]) * 0.5,
                (io[3] + io[4]) * 0.5,
            ];
            let in1 = [
                (io[0] - io[7]) * SEC_HALF_8[0] * 0.5,
                (io[1] - io[6]) * SEC_HALF_8[1] * 0.5,
                (io[2] - io[5]) * SEC_HALF_8[2] * 0.5,
                (io[3] - io[4]) * SEC_HALF_8[3] * 0.5,
            ];
            let out0 = dct4_1d(in0, DctDirection::Forward);
            for (i, v) in out0.into_iter().enumerate() {
                io[i * 2] = v;
            }
            let mut out1 = dct4_1d(in1, DctDirection::Forward);
            out1[0] *= std::f32::consts::SQRT_2;
            for i in 0..3 {
                io[i * 2 + 1] = out1[i] + out1[i + 1];
            }
            io[7] = out1[3];
        } else {
            let in0 = [io[0], io[2], io[4], io[6]];
            let in1 = [
                io[1] * std::f32::consts::SQRT_2,
                io[3] + io[1],
                io[5] + io[3],
                io[7] + io[5],
            ];
            let out0 = dct4_1d(in0, DctDirection::Inverse);
            let out1 = dct4_1d(in1, DctDirection::Inverse);
            for (i, &sec) in SEC_HALF_8.iter().enumerate() {
                let r = out1[i] * sec;
                io[i] = out0[i] + r;
                io[7 - i] = out0[i] - r;
            }
        }
        return;
    }

    let sec = sec_half_vec(n);
    if direction == DctDirection::Forward {
        let (in0, in1) = scratch[..n].split_at_mut(n / 2);
        for i in 0..(n / 2) {
            in0[i] = (io[i] + io[n - i - 1]) * 0.5;
            in1[i] = (io[i] - io[n - i - 1]) * 0.5 * sec[i];
        }
        let (out0, out1) = io.split_at_mut(n / 2);
        dct_1d(in0, out0, DctDirection::Forward);
        dct_1d(in1, out1, DctDirection::Forward);
        in1[0] *= std::f32::consts::SQRT_2;
        for i in 0..(n / 2 - 1) {
            in1[i] += in1[i + 1];
        }
        for i in 0..(n / 2) {
            io[i * 2] = in0[i];
            io[i * 2 + 1] = in1[i];
        }
    } else {
        let (in0, in1) = scratch[..n].split_at_mut(n / 2);
        for i in 0..(n / 2) {
            in0[i] = io[i * 2];
            in1[i] = io[i * 2 + 1];
        }
        for i in 1..(n / 2) {
            in1[n / 2 - i] += in1[n / 2 - i - 1];
        }
        in1[0] *= std::f32::consts::SQRT_2;
        let (out0, out1) = input_output_split(io, n);
        dct_1d(in0, out0, DctDirection::Inverse);
        dct_1d(in1, out1, DctDirection::Inverse);
        for (v, &s) in in1.iter_mut().zip(&sec) {
            *v *= s;
        }
        for i in 0..(n / 2) {
            out0[i] = scratch[i] + scratch[i + n / 2];
            out1[n / 2 - i - 1] = scratch[i] - scratch[i + n / 2];
        }
    }
}

#[inline]
fn input_output_split(io: &mut [f32], n: usize) -> (&mut [f32], &mut [f32]) {
    io.split_at_mut(n / 2)
}

/// Computes an in-place 2D DCT or IDCT on a row-major `width x height` buffer `buf`.
pub fn dct_2d(buf: &mut [f32], width: usize, height: usize, direction: DctDirection) {
    if width * height <= 1 {
        return;
    }
    let mut scratch = vec![0.0f32; width.max(height)];
    for y in 0..height {
        dct_1d(
            &mut buf[y * width..(y + 1) * width],
            &mut scratch[..width],
            direction,
        );
    }
    let mut col = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col[y] = buf[y * width + x];
        }
        dct_1d(&mut col, &mut scratch[..height], direction);
        for y in 0..height {
            buf[y * width + x] = col[y];
        }
    }
}

#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const AFV_BASIS: [[f32; 16]; 16] = [
    [
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
    ],
    [
        0.876902929799142, 0.2206518106944235, -0.10140050393753763, -0.1014005039375375,
        0.2206518106944236, -0.10140050393753777, -0.10140050393753772, -0.10140050393753763,
        -0.10140050393753758, -0.10140050393753769, -0.1014005039375375, -0.10140050393753768,
        -0.10140050393753768, -0.10140050393753759, -0.10140050393753763, -0.10140050393753741,
    ],
    [
        0.0, 0.0, 0.40670075830260755, 0.44444816619734445,
        0.0, 0.0, 0.19574399372042936, 0.2929100136981264,
        -0.40670075830260716, -0.19574399372042872, 0.0, 0.11379074460448091,
        -0.44444816619734384, -0.29291001369812636, -0.1137907446044814, 0.0,
    ],
    [
        0.0, 0.0, -0.21255748058288748, 0.3085497062849767,
        0.0, 0.4706702258572536, -0.1621205195722993, 0.0,
        -0.21255748058287047, -0.16212051957228327, -0.47067022585725277, -0.1464291867126764,
        0.3085497062849487, 0.0, -0.14642918671266536, 0.4251149611657548,
    ],
    [
        0.0, -0.7071067811865474, 0.0, 0.0,
        std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
    [
        -0.4105377591765233, 0.6235485373547691, -0.06435071657946274, -0.06435071657946266,
        0.6235485373547694, -0.06435071657946284, -0.0643507165794628, -0.06435071657946274,
        -0.06435071657946272, -0.06435071657946279, -0.06435071657946266, -0.06435071657946277,
        -0.06435071657946277, -0.06435071657946273, -0.06435071657946274, -0.0643507165794626,
    ],
    [
        0.0, 0.0, -0.4517556589999482, 0.15854503551840063,
        0.0, -0.04038515160822202, 0.0074182263792423875, 0.39351034269210167,
        -0.45175565899994635, 0.007418226379244351, 0.1107416575309343, 0.08298163094882051,
        0.15854503551839705, 0.3935103426921022, 0.0829816309488214, -0.45175565899994796,
    ],
    [
        0.0, 0.0, -0.304684750724869, 0.5112616136591823,
        0.0, 0.0, -0.290480129728998, -0.06578701549142804,
        0.304684750724884, 0.2904801297290076, 0.0, -0.23889773523344604,
        -0.5112616136592012, 0.06578701549142545, 0.23889773523345467, 0.0,
    ],
    [
        0.0, 0.0, 0.3017929516615495, 0.25792362796341184,
        0.0, 0.16272340142866204, 0.09520022653475037, 0.0,
        0.3017929516615503, 0.09520022653475055, -0.16272340142866173, -0.35312385449816297,
        0.25792362796341295, 0.0, -0.3531238544981624, -0.6035859033230976,
    ],
    [
        0.0, 0.0, 0.40824829046386274, 0.0,
        0.0, 0.0, 0.0, -0.4082482904638628,
        -0.4082482904638635, 0.0, 0.0, -0.40824829046386296,
        0.0, 0.4082482904638634, 0.408248290463863, 0.0,
    ],
    [
        0.0, 0.0, 0.1747866975480809, 0.0812611176717539,
        0.0, 0.0, -0.3675398009862027, -0.307882213957909,
        -0.17478669754808135, 0.3675398009862011, 0.0, 0.4826689115059883,
        -0.08126111767175039, 0.30788221395790305, -0.48266891150598584, 0.0,
    ],
    [
        0.0, 0.0, -0.21105601049335784, 0.18567180916109802,
        0.0, 0.0, 0.49215859013738733, -0.38525013709251915,
        0.21105601049335806, -0.49215859013738905, 0.0, 0.17419412659916217,
        -0.18567180916109904, 0.3852501370925211, -0.1741941265991621, 0.0,
    ],
    [
        0.0, 0.0, -0.14266084808807264, -0.3416446842253372,
        0.0, 0.7367497537172237, 0.24627107722075148, -0.08574019035519306,
        -0.14266084808807344, 0.24627107722075137, 0.14883399227113567, -0.04768680350229251,
        -0.3416446842253373, -0.08574019035519267, -0.047686803502292804, -0.14266084808807242,
    ],
    [
        0.0, 0.0, -0.13813540350758585, 0.3302282550303788,
        0.0, 0.08755115000587084, -0.07946706605909573, -0.4613374887461511,
        -0.13813540350758294, -0.07946706605910261, 0.49724647109535086, 0.12538059448563663,
        0.3302282550303805, -0.4613374887461554, 0.12538059448564315, -0.13813540350758452,
    ],
    [
        0.0, 0.0, -0.17437602599651067, 0.0702790691196284,
        0.0, -0.2921026642334881, 0.3623817333531167, 0.0,
        -0.1743760259965108, 0.36238173335311646, 0.29210266423348785, -0.4326608024727445,
        0.07027906911962818, 0.0, -0.4326608024727457, 0.34875205199302267,
    ],
    [
        0.0, 0.0, 0.11354987314994337, -0.07417504595810355,
        0.0, 0.19402893032594343, -0.435190496523228, 0.21918684838857466,
        0.11354987314994257, -0.4351904965232251, 0.5550443808910661, -0.25468277124066463,
        -0.07417504595810233, 0.2191868483885728, -0.25468277124066413, 0.1135498731499429,
    ],
];

fn aux_idct2_in_place<const SIZE: usize>(block: &mut [f32; 64]) {
    let half = SIZE / 2;
    let mut scratch = [[0.0f32; SIZE]; SIZE];
    for y in 0..half {
        for x in 0..half {
            let c00 = block[y * 8 + x];
            let c01 = block[y * 8 + x + half];
            let c10 = block[(y + half) * 8 + x];
            let c11 = block[(y + half) * 8 + x + half];
            scratch[2 * y][2 * x] = c00 + c01 + c10 + c11;
            scratch[2 * y][2 * x + 1] = c00 + c01 - c10 - c11;
            scratch[2 * y + 1][2 * x] = c00 - c01 + c10 - c11;
            scratch[2 * y + 1][2 * x + 1] = c00 - c01 - c10 + c11;
        }
    }
    for (y, row) in scratch.iter().enumerate() {
        block[y * 8..y * 8 + SIZE].copy_from_slice(row);
    }
}

/// Applies the inverse varblock transform for `dct_select` on a `bw*8 x bh*8` coefficient block.
pub fn inverse_varblock_transform(
    block: &mut [f32],
    bw: usize,
    bh: usize,
    dct_select: TransformType,
) {
    use TransformType::*;
    match dct_select {
        Dct2 => {
            let b: &mut [f32; 64] = block.try_into().unwrap();
            aux_idct2_in_place::<2>(b);
            aux_idct2_in_place::<4>(b);
            aux_idct2_in_place::<8>(b);
        }
        Dct4 => {
            let b: &mut [f32; 64] = block.try_into().unwrap();
            aux_idct2_in_place::<2>(b);
            let mut scratch = [0.0f32; 64];
            for y in 0..2 {
                for x in 0..2 {
                    let sub = &mut scratch[(y * 2 + x) * 16..(y * 2 + x + 1) * 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            sub[ix * 4 + iy] = b[(y + iy * 2) * 8 + (x + ix * 2)];
                        }
                    }
                    dct_2d(sub, 4, 4, DctDirection::Inverse);
                }
            }
            for y in 0..2 {
                for x in 0..2 {
                    let sub = &scratch[(y * 2 + x) * 16..(y * 2 + x + 1) * 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            b[(y * 4 + iy) * 8 + (x * 4 + ix)] = sub[iy * 4 + ix];
                        }
                    }
                }
            }
        }
        Hornuss => {
            let b: &mut [f32; 64] = block.try_into().unwrap();
            aux_idct2_in_place::<2>(b);
            let mut scratch = [0.0f32; 64];
            for y in 0..2 {
                for x in 0..2 {
                    let sub = &mut scratch[(y * 2 + x) * 16..(y * 2 + x + 1) * 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            sub[iy * 4 + ix] = b[(y + iy * 2) * 8 + (x + ix * 2)];
                        }
                    }
                    let res_sum: f32 = sub[1..].iter().copied().sum();
                    let avg = sub[0] - res_sum / 16.0;
                    sub[0] = sub[5];
                    sub[5] = 0.0;
                    for s in sub.iter_mut() {
                        *s += avg;
                    }
                }
            }
            for y in 0..2 {
                for x in 0..2 {
                    let sub = &scratch[(y * 2 + x) * 16..(y * 2 + x + 1) * 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            b[(y * 4 + iy) * 8 + (x * 4 + ix)] = sub[iy * 4 + ix];
                        }
                    }
                }
            }
        }
        Dct4x8 | Dct8x4 => {
            let tr = dct_select == Dct8x4;
            let b: &mut [f32; 64] = block.try_into().unwrap();
            let c0 = b[0];
            let c1 = b[8];
            b[0] = c0 + c1;
            b[8] = c0 - c1;
            let mut scratch = [0.0f32; 64];
            for idx in [0usize, 1] {
                let sub = &mut scratch[idx * 32..(idx + 1) * 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        sub[iy * 8 + ix] = b[(iy * 2 + idx) * 8 + ix];
                    }
                }
                dct_2d(sub, 8, 4, DctDirection::Inverse);
            }
            if tr {
                for y in 0..8 {
                    for x in 0..8 {
                        b[x * 8 + y] = scratch[y * 8 + x];
                    }
                }
            } else {
                b.copy_from_slice(&scratch);
            }
        }
        Afv0 | Afv1 | Afv2 | Afv3 => {
            let n = (dct_select as usize) - (Afv0 as usize);
            let flip_x = n % 2;
            let flip_y = n / 2;
            let b: &mut [f32; 64] = block.try_into().unwrap();
            let mut coeff_afv = [0.0f32; 16];
            coeff_afv[0] = (b[0] + b[1] + b[8]) * 4.0;
            for (idx, v) in coeff_afv.iter_mut().enumerate().skip(1) {
                let iy = idx / 4;
                let ix = idx % 4;
                *v = b[(2 * iy) * 8 + 2 * ix];
            }
            let mut samples_afv = [0.0f32; 16];
            for (coeff, basis) in coeff_afv.into_iter().zip(AFV_BASIS) {
                for (s, bv) in samples_afv.iter_mut().zip(basis) {
                    *s = coeff.mul_add(bv, *s);
                }
            }
            let mut s4x4 = [0.0f32; 16];
            let mut s4x8 = [0.0f32; 32];
            s4x4[0] = b[0] - b[1] + b[8];
            for iy in 0..4 {
                for ix in 0..4 {
                    if ix | iy == 0 {
                        continue;
                    }
                    s4x4[ix * 4 + iy] = b[(2 * iy) * 8 + (2 * ix + 1)];
                }
            }
            dct_2d(&mut s4x4, 4, 4, DctDirection::Inverse);

            s4x8[0] = b[0] - b[8];
            for iy in 0..4 {
                for ix in 0..8 {
                    if ix | iy == 0 {
                        continue;
                    }
                    s4x8[iy * 8 + ix] = b[(2 * iy + 1) * 8 + ix];
                }
            }
            dct_2d(&mut s4x8, 8, 4, DctDirection::Inverse);

            for iy in 0..4 {
                let ay = if flip_y == 0 { iy } else { 3 - iy };
                for ix in 0..4 {
                    let ax = if flip_x == 0 { ix } else { 3 - ix };
                    b[(flip_y * 4 + iy) * 8 + (flip_x * 4 + ix)] = samples_afv[ay * 4 + ax];
                }
            }
            for iy in 0..4 {
                let y = flip_y * 4 + iy;
                for ix in 0..4 {
                    let x = (1 - flip_x) * 4 + ix;
                    b[y * 8 + x] = s4x4[iy * 4 + ix];
                }
            }
            for iy in 0..4 {
                let y = (1 - flip_y) * 4 + iy;
                b[y * 8..y * 8 + 8].copy_from_slice(&s4x8[iy * 8..iy * 8 + 8]);
            }
        }
        _ => {
            dct_2d(block, bw * 8, bh * 8, DctDirection::Inverse);
        }
    }
}

/// 2D floating-point planar image grid.
#[derive(Debug, Clone)]
pub struct FloatGrid {
    /// Grid width in samples.
    pub width: usize,
    /// Grid height in samples.
    pub height: usize,
    /// Row-major `width * height` sample buffer.
    pub data: Vec<f32>,
}

impl FloatGrid {
    /// Allocates a zero-initialized `FloatGrid` of `width x height`.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0f32; width * height],
        }
    }

    /// Reads sample at `(x, y)`.
    #[must_use]
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Sets sample at `(x, y)`.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        self.data[y * self.width + x] = val;
    }

    /// Immutable slice for row `y`.
    #[must_use]
    #[inline]
    pub fn row(&self, y: usize) -> &[f32] {
        let start = y * self.width;
        &self.data[start..start + self.width]
    }

    /// Mutable slice for row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.width;
        &mut self.data[start..start + self.width]
    }
}

/// Applies Chroma-from-Luma (CfL) on the dequantized LF grids `[X, Y, B]`.
pub fn chroma_from_luma_lf(lf_xyb: &mut [FloatGrid; 3], corr: &LfChannelCorrelation) {
    let x_factor = corr.x_factor_lf as i32 - 128;
    let b_factor = corr.b_factor_lf as i32 - 128;
    let kx = corr.base_correlation_x + (x_factor as f32 / corr.colour_factor as f32);
    let kb = corr.base_correlation_b + (b_factor as f32 / corr.colour_factor as f32);

    let [lx, ly, lb] = lf_xyb;
    for ((x, &y), b) in lx.data.iter_mut().zip(&ly.data).zip(&mut lb.data) {
        *x += kx * y;
        *b += kb * y;
    }
}

/// Applies adaptive LF smoothing on the dequantized LF grids `[X, Y, B]` (ISO/IEC 18181-1 Annex I.5).
pub fn adaptive_lf_smoothing(
    lf_xyb: &mut [FloatGrid; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    let width = lf_xyb[0].width;
    let height = lf_xyb[0].height;
    if width <= 2 || height <= 2 || lf_xyb[1].width != width || lf_xyb[2].width != width {
        return;
    }

    let scale_inv = (quantizer.global_scale as u64 * quantizer.quant_lf as u64) as f64;
    let lf_step = [
        (512.0 * lf_dequant.m_x_lf as f64 / scale_inv) as f32,
        (512.0 * lf_dequant.m_y_lf as f64 / scale_inv) as f32,
        (512.0 * lf_dequant.m_b_lf as f64 / scale_inv) as f32,
    ];

    let orig = lf_xyb.clone();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut wa = [0.0f32; 3];
            let mut gap = 0.5f32;
            for c in 0..3 {
                let g = &orig[c];
                let me = g.get(x, y);
                let side = g.get(x - 1, y) + g.get(x + 1, y) + g.get(x, y - 1) + g.get(x, y + 1);
                let diag = g.get(x - 1, y - 1)
                    + g.get(x + 1, y - 1)
                    + g.get(x - 1, y + 1)
                    + g.get(x + 1, y + 1);
                wa[c] = me * SCALE_SELF + side * SCALE_SIDE + diag * SCALE_DIAG;
                gap = gap.max((wa[c] - me).abs() / lf_step[c]);
            }
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);
            for c in 0..3 {
                let me = orig[c].get(x, y);
                lf_xyb[c].set(x, y, (wa[c] - me) * gap_scale + me);
            }
        }
    }
}

/// Reconstructs the spatial XYB image grids `[X, Y, B]` from quantized LF and HF coefficients.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_vardct_xyb(
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    hf_global: &HfGlobal,
    lf_groups: &[DecodedLfGroup],
    hf_coeff_grid: &[ModularGrid; 3],
) -> [FloatGrid; 3] {
    reconstruct_vardct_xyb_with_lf_frame(
        image_header,
        frame_header,
        lf_dequant,
        quantizer,
        lf_chan_corr,
        hf_global,
        lf_groups,
        hf_coeff_grid,
        None,
    )
}

/// Reconstructs the spatial XYB image grids `[X, Y, B]` from quantized LF and HF coefficients,
/// optionally reading DC from a hierarchical `lf_frame` when `flags.use_lf_frame()` is set.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_vardct_xyb_with_lf_frame(
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_chan_corr: &LfChannelCorrelation,
    hf_global: &HfGlobal,
    lf_groups: &[DecodedLfGroup],
    hf_coeff_grid: &[ModularGrid; 3],
    lf_frame: Option<&[FloatGrid; 3]>,
) -> [FloatGrid; 3] {
    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let mut bw_total = width.div_ceil(8);
    let mut bh_total = height.div_ceil(8);
    let h_upsample = frame_header
        .jpeg_upsampling
        .into_iter()
        .any(|j| j == 1 || j == 2);
    let v_upsample = frame_header
        .jpeg_upsampling
        .into_iter()
        .any(|j| j == 1 || j == 3);
    if h_upsample {
        bw_total = bw_total.div_ceil(2) * 2;
    }
    if v_upsample {
        bh_total = bh_total.div_ceil(2) * 2;
    }

    let shifts: [ChannelShift; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let subsampled = frame_header.jpeg_upsampling.into_iter().any(|x| x != 0);

    let mut lf_xyb: [FloatGrid; 3] = std::array::from_fn(|c| {
        let (sw, sh) = shifts[c].shift_size((bw_total as u32, bh_total as u32));
        FloatGrid::new(sw as usize, sh as usize)
    });

    let group_dim_blocks = (frame_header.group_dim() / 8) as usize;
    let lf_dim_blocks = group_dim_blocks * 8;
    let lf_cols = (frame_header.lf_groups_per_row()) as usize;
    let m_lfs = [lf_dequant.m_x_lf, lf_dequant.m_y_lf, lf_dequant.m_b_lf];
    let scale_inv = (quantizer.global_scale as u64 * quantizer.quant_lf as u64) as f64;

    for (lf_idx, lfg) in lf_groups.iter().enumerate() {
        let Some(ref lf_quant) = lfg.lf_quant else {
            continue;
        };
        let lf_gx = lf_idx % lf_cols;
        let lf_gy = lf_idx / lf_cols;
        let base_bx = lf_gx * lf_dim_blocks;
        let base_by = lf_gy * lf_dim_blocks;
        let prec_scale = (1i32 << (9 - lfg.extra_precision)) as f64;

        let src_channels = [&lf_quant[1], &lf_quant[0], &lf_quant[2]];
        for c in 0..3 {
            let scale = (m_lfs[c] as f64 * prec_scale / scale_inv) as f32;
            let hs = shifts[c].hshift();
            let vs = shifts[c].vshift();
            let sbx = base_bx >> hs;
            let sby = base_by >> vs;
            let src = src_channels[c];
            for y in 0..src.height {
                if sby + y >= lf_xyb[c].height {
                    break;
                }
                for x in 0..src.width {
                    if sbx + x >= lf_xyb[c].width {
                        break;
                    }
                    lf_xyb[c].set(sbx + x, sby + y, src.get(x, y) as f32 * scale);
                }
            }
        }
    }

    if let Some(ext_lf) = lf_frame
        && frame_header.flags.use_lf_frame()
    {
        for c in 0..3 {
            let w_lim = lf_xyb[c].width.min(ext_lf[c].width);
            let h_lim = lf_xyb[c].height.min(ext_lf[c].height);
            for y in 0..h_lim {
                for x in 0..w_lim {
                    lf_xyb[c].set(x, y, ext_lf[c].get(x, y));
                }
            }
        }
    } else {
        if !subsampled {
            chroma_from_luma_lf(&mut lf_xyb, lf_chan_corr);
        }
        if !frame_header.flags.skip_adaptive_lf_smoothing() {
            adaptive_lf_smoothing(&mut lf_xyb, lf_dequant, quantizer);
        }
    }

    let mut out_xyb: [FloatGrid; 3] = std::array::from_fn(|c| {
        let (sw, sh) = shifts[c].shift_size((bw_total as u32, bh_total as u32));
        FloatGrid::new(sw as usize * 8, sh as usize * 8)
    });

    let oim = &image_header.metadata.opsin_inverse_matrix;
    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];

    for (lf_idx, lfg) in lf_groups.iter().enumerate() {
        let lf_gx = lf_idx % lf_cols;
        let lf_gy = lf_idx / lf_cols;
        let base_bx = lf_gx * lf_dim_blocks;
        let base_by = lf_gy * lf_dim_blocks;

        for by in 0..lfg.bh {
            for bx in 0..lfg.bw {
                let BlockInfo::Data { dct_select, hf_mul } = lfg.block_info[by * lfg.bw + bx]
                else {
                    continue;
                };
                let (bw, bh) = dct_select.dct_select_size();
                let bw = bw as usize;
                let bh = bh as usize;
                let need_tr = dct_select.need_transpose();

                for c in 0..3 {
                    let hs = shifts[c].hshift();
                    let vs = shifts[c].vshift();
                    let sbx = (base_bx + bx) >> hs;
                    let sby = (base_by + by) >> vs;
                    if hs != 0 || vs != 0 {
                        if ((bx >> hs) << hs) != bx || ((by >> vs) << vs) != by {
                            continue;
                        }
                        if !matches!(
                            lfg.block_info[(by >> vs) * lfg.bw + (bx >> hs)],
                            BlockInfo::Data { .. }
                        ) {
                            continue;
                        }
                    }

                    let quant_bias = oim.quant_bias[c];
                    let mul =
                        65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[c];
                    let mat = if need_tr {
                        hf_global.dequant_matrices.get_transposed(c, dct_select)
                    } else {
                        hf_global.dequant_matrices.get(c, dct_select)
                    };

                    let w_px = bw * 8;
                    let h_px = bh * 8;
                    for dy in 0..h_px {
                        let mat_row = &mat[dy * w_px..(dy + 1) * w_px];
                        for (dx, &m) in mat_row.iter().enumerate() {
                            let px = sbx * 8 + dx;
                            let py = sby * 8 + dy;
                            let qn = hf_coeff_grid[c].get(px, py);
                            let mut q = qn as f32;
                            if q.abs() <= 1.0 {
                                q *= quant_bias;
                            } else {
                                q -= oim.quant_bias_numerator / q;
                            }
                            out_xyb[c].set(px, py, q * m * mul);
                        }
                    }
                }
            }
        }

        if !subsampled {
            let px_left = base_bx * 8;
            let px_top = base_by * 8;
            let px_w = lfg.bw * 8;
            let px_h = lfg.bh * 8;
            for dy in 0..px_h {
                let cfl_y = dy / 64;
                for dx in 0..px_w {
                    let cfl_x = dx / 64;
                    let kx = lf_chan_corr.base_correlation_x
                        + (lfg.x_from_y.get(cfl_x, cfl_y) as f32
                            / lf_chan_corr.colour_factor as f32);
                    let kb = lf_chan_corr.base_correlation_b
                        + (lfg.b_from_y.get(cfl_x, cfl_y) as f32
                            / lf_chan_corr.colour_factor as f32);
                    let px = px_left + dx;
                    let py = px_top + dy;
                    let cy = out_xyb[1].get(px, py);
                    let cx = out_xyb[0].get(px, py) + kx * cy;
                    let cb = out_xyb[2].get(px, py) + kb * cy;
                    out_xyb[0].set(px, py, cx);
                    out_xyb[2].set(px, py, cb);
                }
            }
        }

        for by in 0..lfg.bh {
            for bx in 0..lfg.bw {
                let BlockInfo::Data { dct_select, .. } = lfg.block_info[by * lfg.bw + bx] else {
                    continue;
                };
                let (bw, bh) = dct_select.dct_select_size();
                let bw = bw as usize;
                let bh = bh as usize;
                let logbw = bw.trailing_zeros() as usize;
                let logbh = bh.trailing_zeros() as usize;

                for c in 0..3 {
                    let hs = shifts[c].hshift();
                    let vs = shifts[c].vshift();
                    let sbx = (base_bx + bx) >> hs;
                    let sby = (base_by + by) >> vs;
                    if hs != 0 || vs != 0 {
                        if ((bx >> hs) << hs) != bx || ((by >> vs) << vs) != by {
                            continue;
                        }
                        if !matches!(
                            lfg.block_info[(by >> vs) * lfg.bw + (bx >> hs)],
                            BlockInfo::Data { .. }
                        ) {
                            continue;
                        }
                    }

                    if bw * bh == 1 {
                        out_xyb[c].set(sbx * 8, sby * 8, lf_xyb[c].get(sbx, sby));
                    } else {
                        let mut llf = vec![0.0f32; bw * bh];
                        for y in 0..bh {
                            for x in 0..bw {
                                llf[y * bw + x] = lf_xyb[c].get(sbx + x, sby + y);
                            }
                        }
                        dct_2d(&mut llf, bw, bh, DctDirection::Forward);
                        for y in 0..bh {
                            for x in 0..bw {
                                let scaled = llf[y * bw + x]
                                    / (scale_f(y, 5 - logbh) * scale_f(x, 5 - logbw));
                                out_xyb[c].set(sbx * 8 + x, sby * 8 + y, scaled);
                            }
                        }
                    }

                    let w_px = bw * 8;
                    let h_px = bh * 8;
                    let mut block_buf = vec![0.0f32; w_px * h_px];
                    for dy in 0..h_px {
                        for dx in 0..w_px {
                            block_buf[dy * w_px + dx] = out_xyb[c].get(sbx * 8 + dx, sby * 8 + dy);
                        }
                    }
                    inverse_varblock_transform(&mut block_buf, bw, bh, dct_select);
                    for dy in 0..h_px {
                        for dx in 0..w_px {
                            out_xyb[c].set(sbx * 8 + dx, sby * 8 + dy, block_buf[dy * w_px + dx]);
                        }
                    }
                }
            }
        }
    }

    for c in 0..3 {
        let hs = shifts[c].hshift() as usize;
        let vs = shifts[c].vshift() as usize;
        if hs != 0 || vs != 0 || out_xyb[c].width != width || out_xyb[c].height != height {
            let mut cropped = FloatGrid::new(width, height);
            for y in 0..height {
                let sy = (y >> vs).min(out_xyb[c].height.saturating_sub(1));
                for x in 0..width {
                    let sx = (x >> hs).min(out_xyb[c].width.saturating_sub(1));
                    cropped.set(x, y, out_xyb[c].get(sx, sy));
                }
            }
            out_xyb[c] = cropped;
        }
    }

    out_xyb
}

#[inline]
fn mirror_coord(mut offset: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    loop {
        if offset < 0 {
            offset = -(offset + 1);
        } else if (offset as usize) >= len {
            offset = (-(offset + 1)).wrapping_add_unsigned(len * 2);
        } else {
            return offset as usize;
        }
    }
}

/// Applies the 3x3 Gabor-like ("Gaborish") filter on `channels` (ISO/IEC 18181-1 Annex J.1).
pub fn apply_gaborish(channels: &mut [FloatGrid; 3], weights: [[f32; 2]; 3]) {
    use std::simd::f32x4;
    for (grid, [w0, w1]) in channels.iter_mut().zip(weights) {
        let w = grid.width;
        let h = grid.height;
        if w == 0 || h == 0 {
            continue;
        }
        let orig = grid.clone();
        let inv = (1.0 + w0 * 4.0 + w1 * 4.0).recip();
        let vw0 = f32x4::splat(w0);
        let vw1 = f32x4::splat(w1);
        let vinv = f32x4::splat(inv);
        for y in 0..h {
            let ym1 = mirror_coord(y as isize - 1, h);
            let yp1 = mirror_coord(y as isize + 1, h);
            let rt = orig.row(ym1);
            let rc = orig.row(y);
            let rb = orig.row(yp1);
            let out_row = grid.row_mut(y);
            let mut x = 0usize;
            if w >= 6 {
                x = 1;
                while x + 4 < w {
                    let c = f32x4::from_slice(&rc[x..x + 4]);
                    let side = f32x4::from_slice(&rt[x..x + 4])
                        + f32x4::from_slice(&rc[x - 1..x + 3])
                        + f32x4::from_slice(&rc[x + 1..x + 5])
                        + f32x4::from_slice(&rb[x..x + 4]);
                    let diag = f32x4::from_slice(&rt[x - 1..x + 3])
                        + f32x4::from_slice(&rt[x + 1..x + 5])
                        + f32x4::from_slice(&rb[x - 1..x + 3])
                        + f32x4::from_slice(&rb[x + 1..x + 5]);
                    let res = (c + side * vw0 + diag * vw1) * vinv;
                    out_row[x..x + 4].copy_from_slice(&res.to_array());
                    x += 4;
                }
                let xm1 = mirror_coord(-1, w);
                let xp1 = mirror_coord(1, w);
                let side0 = rt[0] + rc[xm1] + rc[xp1] + rb[0];
                let diag0 = rt[xm1] + rt[xp1] + rb[xm1] + rb[xp1];
                out_row[0] = (rc[0] + side0 * w0 + diag0 * w1) * inv;
            }
            for xi in x..w {
                let xm1 = mirror_coord(xi as isize - 1, w);
                let xp1 = mirror_coord(xi as isize + 1, w);
                let side = rt[xi] + rc[xm1] + rc[xp1] + rb[xi];
                let diag = rt[xm1] + rt[xp1] + rb[xm1] + rb[xp1];
                out_row[xi] = (rc[xi] + side * w0 + diag * w1) * inv;
            }
        }
    }
}

/// Applies a 3x3 inverse-Gaborish pre-sharpening filter in XYB space before VarDCT encoding
/// when the decoder's Gaborish deblocking filter is enabled.
pub fn apply_inverse_gaborish(channels: &mut [FloatGrid; 3], weights: [[f32; 2]; 3]) {
    for (grid, [w0, w1]) in channels.iter_mut().zip(weights) {
        let w = grid.width;
        let h = grid.height;
        if w == 0 || h == 0 {
            continue;
        }
        let orig = grid.clone();
        // Damped sharpening factor (0.55 of full inverse) to prevent ringing on step edges.
        let s0 = w0 * 0.55;
        let s1 = w1 * 0.55;
        let center_w = 1.0 + 4.0 * (s0 + s1);
        for y in 0..h {
            let ym1 = mirror_coord(y as isize - 1, h);
            let yp1 = mirror_coord(y as isize + 1, h);
            let rt = orig.row(ym1);
            let rc = orig.row(y);
            let rb = orig.row(yp1);
            let out_row = grid.row_mut(y);
            for (x, out) in out_row.iter_mut().enumerate() {
                let xm1 = mirror_coord(x as isize - 1, w);
                let xp1 = mirror_coord(x as isize + 1, w);
                let side = rt[x] + rc[xm1] + rc[xp1] + rb[x];
                let diag = rt[xm1] + rt[xp1] + rb[xm1] + rb[xp1];
                *out = rc[x] * center_w - side * s0 - diag * s1;
            }
        }
    }
}

/// Applies the Edge-Preserving Filter (EPF stages 0, 1, 2) on `channels` (ISO/IEC 18181-1 Annex J.2).
pub fn apply_epf(
    channels: &mut [FloatGrid; 3],
    frame_header: &FrameHeader,
    epf_params: &EpfParams,
    lf_groups: Option<&[DecodedLfGroup]>,
) {
    let width = channels[0].width;
    let height = channels[0].height;
    if width == 0 || height == 0 {
        return;
    }

    let bw = width.div_ceil(8);
    let bh = height.div_ceil(8);
    let mut sigma_grid = vec![epf_params.sigma_for_modular; bw * bh];
    if let Some(lfg_list) = lf_groups {
        let lf_dim_blocks = (frame_header.group_dim() as usize / 8) * 8;
        let lf_cols = frame_header.lf_groups_per_row() as usize;
        for (lf_idx, lfg) in lfg_list.iter().enumerate() {
            let gx = lf_idx % lf_cols;
            let gy = lf_idx / lf_cols;
            let base_bx = gx * lf_dim_blocks;
            let base_by = gy * lf_dim_blocks;
            for by in 0..lfg.bh {
                if base_by + by >= bh {
                    break;
                }
                for bx in 0..lfg.bw {
                    if base_bx + bx >= bw {
                        break;
                    }
                    sigma_grid[(base_by + by) * bw + (base_bx + bx)] =
                        lfg.epf_sigma[by * lfg.bw + bx];
                }
            }
        }
    }

    let iters = epf_params.iters;
    if iters == 3 {
        run_epf_stage::<0>(channels, &sigma_grid, bw, epf_params);
    }
    if iters >= 1 {
        run_epf_stage::<1>(channels, &sigma_grid, bw, epf_params);
    }
    if iters >= 2 {
        run_epf_stage::<2>(channels, &sigma_grid, bw, epf_params);
    }
}

fn run_epf_stage<const STEP: usize>(
    channels: &mut [FloatGrid; 3],
    sigma_grid: &[f32],
    sigma_bw: usize,
    epf_params: &EpfParams,
) {
    const KERNEL_1: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    #[rustfmt::skip]
    const KERNEL_2: [(isize, isize); 12] = [
        (0, -2), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (0, 2),
    ];
    let kernel: &[(isize, isize)] = if STEP == 0 { &KERNEL_2 } else { &KERNEL_1 };
    let dist_offsets: &[(isize, isize)] = if STEP == 0 {
        &[(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)]
    } else if STEP == 1 {
        &[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)]
    } else {
        &[(0, 0)]
    };

    let step_mul = if STEP == 0 {
        epf_params.sigma.pass0_sigma_scale
    } else if STEP == 2 {
        epf_params.sigma.pass2_sigma_scale
    } else {
        1.0
    };
    let border_sad_mul = epf_params.sigma.border_sad_mul;
    let channel_scale = epf_params.channel_scale;

    let width = channels[0].width;
    let height = channels[0].height;
    let orig = channels.clone();

    for y in 0..height {
        let is_y_border = (y + 1) & 0b110 == 0;
        let sm_row = if is_y_border {
            [step_mul * border_sad_mul; 8]
        } else {
            let mut sm = [step_mul; 8];
            sm[0] *= border_sad_mul;
            sm[7] *= border_sad_mul;
            sm
        };
        let sigma_row = &sigma_grid[(y / 8) * sigma_bw..(y / 8 + 1) * sigma_bw];

        for x in 0..width {
            let sigma_val = sigma_row[x / 8];
            if sigma_val < 0.3 {
                continue;
            }
            let sm = sm_row[x & 7];
            let neg_inv_sigma = 6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma_val * sm;

            let mut sum_weights = 1.0f32;
            let mut sum_ch = [orig[0].get(x, y), orig[1].get(x, y), orig[2].get(x, y)];

            for &(kx, ky) in kernel {
                let mut dist = 0.0f32;
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for &(ix, iy) in dist_offsets {
                        let kx_i = mirror_coord(x as isize + kx + ix, width);
                        let ky_i = mirror_coord(y as isize + ky + iy, height);
                        let bx_i = mirror_coord(x as isize + ix, width);
                        let by_i = mirror_coord(y as isize + iy, height);
                        acc += (orig[c].get(kx_i, ky_i) - orig[c].get(bx_i, by_i)).abs();
                    }
                    dist += channel_scale[c] * acc;
                }
                let w = (1.0 + dist * neg_inv_sigma).max(0.0);
                sum_weights += w;
                let tx = mirror_coord(x as isize + kx, width);
                let ty = mirror_coord(y as isize + ky, height);
                for c in 0..3 {
                    sum_ch[c] += w * orig[c].get(tx, ty);
                }
            }

            for c in 0..3 {
                channels[c].set(x, y, sum_ch[c] / sum_weights);
            }
        }
    }
}

/// Converts a single linear sRGB sample `[0.0, 1.0]` to non-linear sRGB `[0.0, 1.0]`.
#[must_use]
#[inline]
pub fn linear_to_srgb_f32(x: f32) -> f32 {
    let a = x.abs();
    let out = if a <= 0.0031308 {
        a * 12.92
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    out.copysign(x)
}

/// Converts a single non-linear sRGB sample `[0.0, 1.0]` to linear sRGB `[0.0, 1.0]`.
#[must_use]
#[inline]
pub fn srgb_to_linear_f32(x: f32) -> f32 {
    let a = x.abs();
    let out = if a <= 0.04045 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    out.copysign(x)
}

/// Applies the forward transfer function (`Linear -> target TF`) to `x`.
#[must_use]
#[inline]
pub fn apply_forward_tf(x: f32, tf: TransferFunctionHeader, intensity_target: f32) -> f32 {
    match tf {
        TransferFunctionHeader::Linear | TransferFunctionHeader::Unknown => x,
        TransferFunctionHeader::Srgb => linear_to_srgb_f32(x),
        TransferFunctionHeader::Gamma(g) => {
            if x <= 1e-7 {
                0.0
            } else {
                x.powf(1e7 / g as f32)
            }
        }
        TransferFunctionHeader::Bt709 => {
            let a = x.abs();
            let v = if a < 0.018_053_97 {
                4.5 * a
            } else {
                1.0992968 * a.powf(0.45) - 0.09929682
            };
            v.copysign(x)
        }
        TransferFunctionHeader::Dci => {
            if x <= 1e-7 {
                0.0
            } else {
                x.powf(1.0 / 2.6)
            }
        }
        TransferFunctionHeader::Pq => {
            let y = (x * (intensity_target / 10000.0)).max(0.0);
            let m1 = 2610.0 / 16384.0;
            let m2 = 2523.0 / 4096.0 * 128.0;
            let c1 = 3424.0 / 4096.0;
            let c2 = 2413.0 / 4096.0 * 32.0;
            let c3 = 2392.0 / 4096.0 * 32.0;
            let ym1 = y.powf(m1);
            ((c1 + c2 * ym1) / (1.0 + c3 * ym1)).powf(m2)
        }
        TransferFunctionHeader::Hlg => {
            let a = x.abs();
            let v = if a <= 1.0 / 12.0 {
                (3.0 * a).sqrt()
            } else {
                0.17883277 * (12.0 * a - 0.28466892).ln() + 0.5599107
            };
            v.copysign(x)
        }
    }
}

/// Applies the inverse transfer function (`target TF -> Linear`) to `x`.
#[must_use]
#[inline]
pub fn apply_inverse_tf(x: f32, tf: TransferFunctionHeader, intensity_target: f32) -> f32 {
    match tf {
        TransferFunctionHeader::Linear | TransferFunctionHeader::Unknown => x,
        TransferFunctionHeader::Srgb => srgb_to_linear_f32(x),
        TransferFunctionHeader::Gamma(g) => {
            let a = x.abs();
            if a <= 1e-7 {
                0.0
            } else {
                a.powf(1e7 / (g as f32)).copysign(x)
            }
        }
        TransferFunctionHeader::Bt709 => {
            let a = x.abs();
            let v = if a < 0.081_242_86 {
                a / 4.5
            } else {
                ((a + 0.09929682) / 1.0992968).powf(1.0 / 0.45)
            };
            v.copysign(x)
        }
        TransferFunctionHeader::Dci => {
            let a = x.abs();
            a.powf(2.6).copysign(x)
        }
        TransferFunctionHeader::Pq => {
            let p = x.clamp(0.0, 1.0);
            let m1 = 2610.0 / 16384.0;
            let m2 = 2523.0 / 4096.0 * 128.0;
            let c1 = 3424.0 / 4096.0;
            let c2 = 2413.0 / 4096.0 * 32.0;
            let c3 = 2392.0 / 4096.0 * 32.0;
            let pm = p.powf(1.0 / m2);
            let num = (pm - c1).max(0.0);
            let den = c2 - c3 * pm;
            if den <= 1e-7 {
                1.0
            } else {
                (num / den).powf(1.0 / m1) * (10000.0 / intensity_target.max(1.0))
            }
        }
        TransferFunctionHeader::Hlg => {
            let a = x.abs();
            let v = if a <= 0.5 {
                (a * a) / 3.0
            } else {
                (((a - 0.5599107) / 0.17883277).exp() + 0.28466892) / 12.0
            };
            v.copysign(x)
        }
    }
}

const MAT_BRADFORD: [f32; 9] = [
    0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296,
];

const MAT_BRADFORD_INV: [f32; 9] = [
    0.9869929, -0.1470543, 0.1599627, 0.4323053, 0.5183603, 0.0492912, -0.0085287, 0.0400428,
    0.9684867,
];

#[inline]
fn matmul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]
}

#[inline]
fn matmul3vec(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matinv3(m: &[f32; 9]) -> [f32; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        + m[1] * (m[5] * m[6] - m[3] * m[8])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    [
        (m[4] * m[8] - m[5] * m[7]) / det,
        (m[7] * m[2] - m[8] * m[1]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        (m[5] * m[6] - m[3] * m[8]) / det,
        (m[8] * m[0] - m[6] * m[2]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        (m[3] * m[7] - m[4] * m[6]) / det,
        (m[6] * m[1] - m[7] * m[0]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ]
}

fn white_point_xy(wp: WhitePoint) -> [f32; 2] {
    match wp {
        WhitePoint::D65 => [0.3127, 0.329],
        WhitePoint::E => [1.0 / 3.0, 1.0 / 3.0],
        WhitePoint::Dci => [0.314, 0.351],
        WhitePoint::Custom(xy) => [xy.x as f32 / 1_000_000.0, xy.y as f32 / 1_000_000.0],
    }
}

#[allow(clippy::excessive_precision)]
fn primaries_xy(primaries: Primaries) -> [[f32; 2]; 3] {
    match primaries {
        Primaries::Srgb => [
            [0.639998686, 0.330010138],
            [0.300003784, 0.600003357],
            [0.150002046, 0.059997204],
        ],
        Primaries::Bt2100 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        Primaries::P3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        Primaries::Custom { red, green, blue } => [
            [red.x as f32 / 1_000_000.0, red.y as f32 / 1_000_000.0],
            [green.x as f32 / 1_000_000.0, green.y as f32 / 1_000_000.0],
            [blue.x as f32 / 1_000_000.0, blue.y as f32 / 1_000_000.0],
        ],
    }
}

fn illuminant_to_xyz(xy: [f32; 2]) -> [f32; 3] {
    let y = xy[1].max(1e-6);
    [xy[0] / y, 1.0, (1.0 - xy[0]) / y - 1.0]
}

fn adapt_mat(from_xy: [f32; 2], to_xy: [f32; 2]) -> [f32; 9] {
    if (from_xy[0] - to_xy[0]).abs() < 1e-7 && (from_xy[1] - to_xy[1]).abs() < 1e-7 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let from_w = illuminant_to_xyz(from_xy);
    let to_w = illuminant_to_xyz(to_xy);
    let from_lms = matmul3vec(&MAT_BRADFORD, &from_w);
    let to_lms = matmul3vec(&MAT_BRADFORD, &to_w);
    let mul = [
        to_lms[0] / from_lms[0],
        to_lms[1] / from_lms[1],
        to_lms[2] / from_lms[2],
    ];
    let multiplied = std::array::from_fn(|idx| MAT_BRADFORD[idx] * mul[idx / 3]);
    matmul3(&MAT_BRADFORD_INV, &multiplied)
}

fn primaries_to_xyz_mat(p: [[f32; 2]; 3], wp: [f32; 2]) -> [f32; 9] {
    let mut m = [
        p[0][0],
        p[1][0],
        p[2][0],
        p[0][1],
        p[1][1],
        p[2][1],
        1.0 - p[0][0] - p[0][1],
        1.0 - p[1][0] - p[1][1],
        1.0 - p[2][0] - p[2][1],
    ];
    let inv = matinv3(&m);
    let w_xyz = illuminant_to_xyz(wp);
    let mul = matmul3vec(&inv, &w_xyz);
    for (idx, val) in m.iter_mut().enumerate() {
        *val *= mul[idx % 3];
    }
    m
}

fn xyz_to_primaries_mat(p: [[f32; 2]; 3], wp: [f32; 2]) -> [f32; 9] {
    let m = [
        p[0][0],
        p[1][0],
        p[2][0],
        p[0][1],
        p[1][1],
        p[2][1],
        1.0 - p[0][0] - p[0][1],
        1.0 - p[1][0] - p[1][1],
        1.0 - p[2][0] - p[2][1],
    ];
    let mut inv = matinv3(&m);
    let w_xyz = illuminant_to_xyz(wp);
    let mul = matmul3vec(&inv, &w_xyz);
    for (idx, val) in inv.iter_mut().enumerate() {
        *val /= mul[idx % 3];
    }
    inv
}

/// Computes the 3x3 linear gamut conversion matrix from D65 sRGB to `(target_primaries, target_wp)`.
#[must_use]
pub fn srgb_to_target_gamut_matrix(target_primaries: Primaries, target_wp: WhitePoint) -> [f32; 9] {
    if target_primaries == Primaries::Srgb && target_wp == WhitePoint::D65 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let srgb_p = primaries_xy(Primaries::Srgb);
    let d65 = white_point_xy(WhitePoint::D65);
    let dst_p = primaries_xy(target_primaries);
    let dst_w = white_point_xy(target_wp);
    let to_xyz = primaries_to_xyz_mat(srgb_p, d65);
    let adapt = adapt_mat(d65, dst_w);
    let from_xyz = xyz_to_primaries_mat(dst_p, dst_w);
    matmul3(&from_xyz, &matmul3(&adapt, &to_xyz))
}

/// Computes the 3x3 linear gamut conversion matrix from `(src_primaries, src_wp)` to D65 sRGB.
#[must_use]
pub fn target_to_srgb_gamut_matrix(src_primaries: Primaries, src_wp: WhitePoint) -> [f32; 9] {
    if src_primaries == Primaries::Srgb && src_wp == WhitePoint::D65 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let srgb_p = primaries_xy(Primaries::Srgb);
    let d65 = white_point_xy(WhitePoint::D65);
    let src_p = primaries_xy(src_primaries);
    let src_w = white_point_xy(src_wp);
    let to_xyz = primaries_to_xyz_mat(src_p, src_w);
    let adapt = adapt_mat(src_w, d65);
    let from_xyz = xyz_to_primaries_mat(srgb_p, d65);
    matmul3(&from_xyz, &matmul3(&adapt, &to_xyz))
}

/// Converts planar XYB grids `[X, Y, B]` in-place to target RGB (or Grey in channel 0).
pub fn xyb_to_target_rgb(
    xyb: &mut [FloatGrid; 3],
    oim: &OpsinInverseMatrix,
    intensity_target: f32,
    colour_encoding: &ColourEncoding,
) {
    let itscale = 255.0 / intensity_target;
    let ob = oim.opsin_bias;
    let cbrt_ob = ob.map(|v| v.cbrt());
    let inv = oim.inv_mat;

    let (tf, gamut_mat) = match colour_encoding {
        ColourEncoding::Enum(e) => (
            e.tf,
            srgb_to_target_gamut_matrix(e.primaries, e.white_point),
        ),
        ColourEncoding::IccProfile(_) => (
            TransferFunctionHeader::Srgb,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ),
    };
    let is_identity_gamut = gamut_mat == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let len = xyb[0].data.len();
    for i in 0..len {
        let x = xyb[0].data[i];
        let y = xyb[1].data[i];
        let b = xyb[2].data[i];

        let gl = (y + x) - cbrt_ob[0];
        let gm = (y - x) - cbrt_ob[1];
        let gs = b - cbrt_ob[2];

        let ml = (gl * gl).mul_add(gl, ob[0]) * itscale;
        let mm = (gm * gm).mul_add(gm, ob[1]) * itscale;
        let ms = (gs * gs).mul_add(gs, ob[2]) * itscale;

        let mut r_lin = inv[0][0] * ml + inv[0][1] * mm + inv[0][2] * ms;
        let mut g_lin = inv[1][0] * ml + inv[1][1] * mm + inv[1][2] * ms;
        let mut b_lin = inv[2][0] * ml + inv[2][1] * mm + inv[2][2] * ms;

        if !is_identity_gamut {
            let [rt, gt, bt] = matmul3vec(&gamut_mat, &[r_lin, g_lin, b_lin]);
            r_lin = rt;
            g_lin = gt;
            b_lin = bt;
        }

        xyb[0].data[i] = apply_forward_tf(r_lin, tf, intensity_target);
        xyb[1].data[i] = apply_forward_tf(g_lin, tf, intensity_target);
        xyb[2].data[i] = apply_forward_tf(b_lin, tf, intensity_target);
    }
}

/// Converts a single linear sRGB triplet `[r, g, b]` (`[0.0, 1.0]`) to XYB `[x, y, b]`
/// using the default ISO/IEC 18181-1 opsin absorbance matrix.
#[must_use]
#[inline]
pub fn linear_srgb_to_xyb_pixel(r_lin: f32, g_lin: f32, b_lin: f32) -> [f32; 3] {
    const K: [[f32; 3]; 3] = [
        [0.30, 0.622, 0.078],
        [0.23, 0.692, 0.078],
        [0.24342269, 0.20476744, 0.55180987],
    ];
    const POS_BIAS: f32 = 0.0037930733;
    let bias_cbrt = POS_BIAS.cbrt();

    let ml = (K[0][0] * r_lin + K[0][1] * g_lin + K[0][2] * b_lin + POS_BIAS).max(0.0);
    let mm = (K[1][0] * r_lin + K[1][1] * g_lin + K[1][2] * b_lin + POS_BIAS).max(0.0);
    let ms = (K[2][0] * r_lin + K[2][1] * g_lin + K[2][2] * b_lin + POS_BIAS).max(0.0);

    let lg = ml.cbrt() - bias_cbrt;
    let mg = mm.cbrt() - bias_cbrt;
    let sg = ms.cbrt() - bias_cbrt;

    [0.5 * (lg - mg), 0.5 * (lg + mg), sg]
}

/// Configuration options for the v2 clean-room VarDCT frame encoder.
#[derive(Debug, Clone, Copy)]
pub struct VarDctEncodeOptions {
    /// Quality factor in `1.0..=100.0`.
    pub quality: f32,
    /// Estimate Chroma-from-Luma (`x_from_y`, `b_from_y`) correlation in `HfMetadata`.
    pub enable_cfl: bool,
    /// Enable activity-masked per-block `hf_mul` quantization (`1..=2`) and EPF sharpness map.
    pub enable_adaptive_quant: bool,
    /// Enable adaptive `Dct16` vs `Dct8` varblock transform selection.
    pub enable_adaptive_dct: bool,
    /// Entropy coding backend (`Prefix` or `Ans`).
    pub entropy_mode: JxlEntropyMode,
    /// Enable 2D LZ77 in Modular sub-streams (`LfCoeff`, `HfMetadata`, `ExtraChannels`).
    pub lz77: bool,
}

impl Default for VarDctEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 90.0,
            enable_cfl: true,
            enable_adaptive_quant: true,
            enable_adaptive_dct: true,
            entropy_mode: JxlEntropyMode::Prefix,
            lz77: false,
        }
    }
}

/// Clean-room VarDCT v2 frame encoder supporting `extra_precision` DC, Chroma-from-Luma (`CfL`),
/// adaptive per-block `hf_mul` quantization, `Dct16`/`Dct8` varblock selection, EPF sharpness
/// estimation, and `Prefix` / `12-bit rANS` entropy coding.
pub fn encode_vardct_frame_sections_v2(
    xyb: &[FloatGrid; 3],
    extra_channels: &[ModularGrid],
    frame_header: &FrameHeader,
    opts: &VarDctEncodeOptions,
) -> Result<Vec<Vec<u8>>, JxlError> {
    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let bw_total = width.div_ceil(8);
    let bh_total = height.div_ceil(8);

    let q_clamped = opts.quality.clamp(1.0, 100.0);
    let global_scale =
        ((2048.0 + (q_clamped / 100.0).powf(2.0) * 63487.0).round() as u32).clamp(2048, 65535);
    let quantizer = Quantizer {
        global_scale,
        quant_lf: 32,
    };
    let extra_precision: u8 = if q_clamped >= 80.0 { 1 } else { 0 };
    let prec_scale = (1i32 << (9 - extra_precision)) as f64;

    let lf_dequant = LfChannelDequantization::default();
    let lf_corr = LfChannelCorrelation {
        all_default: false,
        colour_factor: 84,
        base_correlation_x: 0.0,
        base_correlation_b: 0.0,
        x_factor_lf: 128,
        b_factor_lf: 128,
    };
    let dequant_set = DequantMatrixSet::default();
    let oim = OpsinInverseMatrix::default();

    let mut dc_grids = [
        ModularGrid::new(bw_total, bh_total),
        ModularGrid::new(bw_total, bh_total),
        ModularGrid::new(bw_total, bh_total),
    ];
    let mut hf_q_grids = [
        ModularGrid::new(bw_total * 8, bh_total * 8),
        ModularGrid::new(bw_total * 8, bh_total * 8),
        ModularGrid::new(bw_total * 8, bh_total * 8),
    ];

    let scale_inv = (quantizer.global_scale as u64 * quantizer.quant_lf as u64) as f64;
    let m_lfs = [lf_dequant.m_x_lf, lf_dequant.m_y_lf, lf_dequant.m_b_lf];
    let lf_scales: [f32; 3] =
        std::array::from_fn(|c| (m_lfs[c] as f64 * prec_scale / scale_inv) as f32);
    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];
    let hf_mul_base = 65536.0 / (quantizer.global_scale as f32);

    // Per-8x8-block variance and transform/quant decision map
    let mut block_var = vec![0.0f32; bw_total * bh_total];
    for by in 0..bh_total {
        for bx in 0..bw_total {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for dy in 0..8 {
                let sy = (by * 8 + dy).min(height - 1);
                for dx in 0..8 {
                    let sx = (bx * 8 + dx).min(width - 1);
                    let v = xyb[1].get(sx, sy);
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let mean = sum / 64.0;
            block_var[by * bw_total + bx] = (sum_sq / 64.0 - mean * mean).max(0.0);
        }
    }

    let mut block_info_grid = vec![BlockInfo::Uninit; bw_total * bh_total];
    let mut sharpness_grid = ModularGrid::new(bw_total, bh_total);
    let epf_enabled = frame_header.restoration_filter.epf.enabled();

    for by in 0..bh_total {
        for bx in 0..bw_total {
            if block_info_grid[by * bw_total + bx].is_occupied() {
                continue;
            }
            let can_dct16 = opts.enable_adaptive_dct
                && bx % 2 == 0
                && by % 2 == 0
                && (bx % 32) + 2 <= 32
                && (by % 32) + 2 <= 32
                && bx + 2 <= bw_total
                && by + 2 <= bh_total
                && !block_info_grid[by * bw_total + bx + 1].is_occupied()
                && !block_info_grid[(by + 1) * bw_total + bx].is_occupied()
                && !block_info_grid[(by + 1) * bw_total + bx + 1].is_occupied();

            let use_dct16 = if can_dct16 {
                let v00 = block_var[by * bw_total + bx];
                let v10 = block_var[by * bw_total + bx + 1];
                let v01 = block_var[(by + 1) * bw_total + bx];
                let v11 = block_var[(by + 1) * bw_total + bx + 1];
                let vmax = v00.max(v10).max(v01).max(v11);
                vmax < 0.0012
            } else {
                false
            };

            let dct_select = if use_dct16 {
                TransformType::Dct16
            } else {
                TransformType::Dct8
            };
            let (dw, dh) = dct_select.dct_select_size();
            let dw = dw as usize;
            let dh = dh as usize;

            let v_local = block_var[by * bw_total + bx];
            let hf_mul = if opts.enable_adaptive_quant && v_local > 0.008 && q_clamped >= 50.0 {
                2i32
            } else {
                1i32
            };
            let sharp_val = if epf_enabled {
                ((v_local.sqrt() * 28.0).round() as i32).clamp(0, 7)
            } else {
                0
            };

            for dy in 0..dh {
                for dx in 0..dw {
                    let idx = (by + dy) * bw_total + (bx + dx);
                    block_info_grid[idx] = if dx == 0 && dy == 0 {
                        BlockInfo::Data { dct_select, hf_mul }
                    } else {
                        BlockInfo::Occupied
                    };
                    sharpness_grid.set(bx + dx, by + dy, sharp_val);
                }
            }
        }
    }

    // Forward DCT per varblock (storing unquantized frequency coefficients in `freq_grids`)
    let mut freq_grids = [
        FloatGrid::new(bw_total * 8, bh_total * 8),
        FloatGrid::new(bw_total * 8, bh_total * 8),
        FloatGrid::new(bw_total * 8, bh_total * 8),
    ];

    for by in 0..bh_total {
        for bx in 0..bw_total {
            let BlockInfo::Data { dct_select, .. } = block_info_grid[by * bw_total + bx] else {
                continue;
            };
            let (bw, bh) = dct_select.dct_select_size();
            let bw = bw as usize;
            let bh = bh as usize;
            let w_px = bw * 8;
            let h_px = bh * 8;
            let logbw = bw.trailing_zeros() as usize;
            let logbh = bh.trailing_zeros() as usize;

            for c in 0..3 {
                let lf_step = lf_scales[c];
                let mut block = vec![0.0f32; w_px * h_px];
                for dy in 0..h_px {
                    let sy = (by * 8 + dy).min(height - 1);
                    for dx in 0..w_px {
                        let sx = (bx * 8 + dx).min(width - 1);
                        block[dy * w_px + dx] = xyb[c].get(sx, sy);
                    }
                }
                dct_2d(&mut block, w_px, h_px, DctDirection::Forward);
                for dy in 0..h_px {
                    for dx in 0..w_px {
                        freq_grids[c].set(bx * 8 + dx, by * 8 + dy, block[dy * w_px + dx]);
                    }
                }

                if bw * bh == 1 {
                    let q_dc = (block[0] / lf_step).round() as i32;
                    dc_grids[c].set(bx, by, q_dc);
                } else {
                    let mut llf = vec![0.0f32; bw * bh];
                    for y in 0..bh {
                        for x in 0..bw {
                            llf[y * bw + x] = block[y * w_px + x]
                                * (scale_f(y, 5 - logbh) * scale_f(x, 5 - logbw));
                        }
                    }
                    dct_2d(&mut llf, bw, bh, DctDirection::Inverse);
                    for y in 0..bh {
                        for x in 0..bw {
                            let q_dc = (llf[y * bw + x] / lf_step).round() as i32;
                            dc_grids[c].set(bx + x, by + y, q_dc);
                        }
                    }
                }
            }
        }
    }

    // First quantize Y channel (`c == 1`) and reconstruct dequantized Y AC in `deq_y_grid`
    let mut deq_y_grid = FloatGrid::new(bw_total * 8, bh_total * 8);
    for by in 0..bh_total {
        for bx in 0..bw_total {
            let BlockInfo::Data { dct_select, hf_mul } = block_info_grid[by * bw_total + bx] else {
                continue;
            };
            let (bw, bh) = dct_select.dct_select_size();
            let bw = bw as usize;
            let bh = bh as usize;
            let w_px = bw * 8;
            let h_px = bh * 8;
            let mat = dequant_set.get_transposed(1, dct_select);
            let mul = (hf_mul_base / (hf_mul as f32)) * qm_scale[1];
            let quant_bias = oim.quant_bias[1];

            for dy in 0..h_px {
                for dx in 0..w_px {
                    if dx < bw && dy < bh {
                        continue;
                    }
                    let px = bx * 8 + dx;
                    let py = by * 8 + dy;
                    let step = mat[dy * w_px + dx] * mul;
                    let coeff = freq_grids[1].get(px, py);
                    let raw = coeff / step;
                    let q_ac = if raw.abs() < 0.62 {
                        0
                    } else {
                        raw.round() as i32
                    };
                    hf_q_grids[1].set(px, py, q_ac);
                    let mut qf = q_ac as f32;
                    if qf.abs() <= 1.0 {
                        qf *= quant_bias;
                    } else {
                        qf -= oim.quant_bias_numerator / qf;
                    }
                    deq_y_grid.set(px, py, qf * step);
                }
            }
        }
    }

    // Estimate Chroma-from-Luma (`x_from_y_full`, `b_from_y_full`) per 64x64 tile (`8x8` blocks)
    let cfl_w_total = bw_total.div_ceil(8);
    let cfl_h_total = bh_total.div_ceil(8);
    let mut x_from_y_full = ModularGrid::new(cfl_w_total, cfl_h_total);
    let mut b_from_y_full = ModularGrid::new(cfl_w_total, cfl_h_total);

    if opts.enable_cfl {
        for cy_tile in 0..cfl_h_total {
            for cx_tile in 0..cfl_w_total {
                let mut sum_yy = 0.0f64;
                let mut sum_xy = 0.0f64;
                let mut sum_by = 0.0f64;
                let bx_start = cx_tile * 8;
                let by_start = cy_tile * 8;
                let bx_end = (bx_start + 8).min(bw_total);
                let by_end = (by_start + 8).min(bh_total);

                for by in by_start..by_end {
                    for bx in bx_start..bx_end {
                        let BlockInfo::Data { dct_select, .. } =
                            block_info_grid[by * bw_total + bx]
                        else {
                            continue;
                        };
                        let (bw, bh) = dct_select.dct_select_size();
                        let bw = bw as usize;
                        let bh = bh as usize;
                        for dy in 0..(bh * 8) {
                            for dx in 0..(bw * 8) {
                                if dx < bw && dy < bh {
                                    continue;
                                }
                                let px = bx * 8 + dx;
                                let py = by * 8 + dy;
                                let yv = f64::from(deq_y_grid.get(px, py));
                                let xv = f64::from(freq_grids[0].get(px, py));
                                let bv = f64::from(freq_grids[2].get(px, py));
                                sum_yy += yv * yv;
                                sum_xy += xv * yv;
                                sum_by += bv * yv;
                            }
                        }
                    }
                }
                if sum_yy > 1e-5 {
                    let kx = ((sum_xy / sum_yy) * 84.0).round().clamp(-64.0, 64.0) as i32;
                    let kb = ((sum_by / sum_yy) * 84.0).round().clamp(-64.0, 64.0) as i32;
                    x_from_y_full.set(cx_tile, cy_tile, kx);
                    b_from_y_full.set(cx_tile, cy_tile, kb);
                }
            }
        }
    }

    // Quantize X (`c == 0`) and B (`c == 2`) after subtracting CfL prediction `k * deq_y`
    for c in [0usize, 2] {
        for by in 0..bh_total {
            for bx in 0..bw_total {
                let BlockInfo::Data { dct_select, hf_mul } = block_info_grid[by * bw_total + bx]
                else {
                    continue;
                };
                let (bw, bh) = dct_select.dct_select_size();
                let bw = bw as usize;
                let bh = bh as usize;
                let w_px = bw * 8;
                let h_px = bh * 8;
                let mat = dequant_set.get_transposed(c, dct_select);
                let mul = (hf_mul_base / (hf_mul as f32)) * qm_scale[c];

                for dy in 0..h_px {
                    for dx in 0..w_px {
                        if dx < bw && dy < bh {
                            continue;
                        }
                        let px = bx * 8 + dx;
                        let py = by * 8 + dy;
                        let cfl_k = if c == 0 {
                            x_from_y_full.get(px / 64, py / 64) as f32 / 84.0
                        } else {
                            b_from_y_full.get(px / 64, py / 64) as f32 / 84.0
                        };
                        let step = mat[dy * w_px + dx] * mul;
                        let coeff = freq_grids[c].get(px, py) - cfl_k * deq_y_grid.get(px, py);
                        let raw = coeff / step;
                        let q_ac = if raw.abs() < 0.62 {
                            0
                        } else {
                            raw.round() as i32
                        };
                        hf_q_grids[c].set(px, py, q_ac);
                    }
                }
            }
        }
    }

    // 1. Encode `LfGlobal` bitstream.
    let mut lf_global_bw = BitWriter::new();
    lf_dequant.write(&mut lf_global_bw)?;
    quantizer.write(&mut lf_global_bw)?;
    HfBlockContext::write_single_cluster(&mut lf_global_bw);
    lf_corr.write(&mut lf_global_bw)?;
    lf_global_bw.write_bool(false); // GlobalModular: has_global_tree = false

    let group_dim = frame_header.group_dim() as usize;
    let group_dim_blocks = group_dim / 8;
    let lf_dim_blocks = group_dim_blocks * 8;
    let num_lf_cols = bw_total.div_ceil(lf_dim_blocks);
    let num_lf_rows = bh_total.div_ceil(lf_dim_blocks);
    let num_lf_groups = num_lf_cols * num_lf_rows;
    let num_cols = bw_total.div_ceil(group_dim_blocks);
    let num_rows = bh_total.div_ceil(group_dim_blocks);
    let num_groups = num_cols * num_rows;

    let mod_opts_grad = ModularEncodeOptions {
        predictor: Predictor::Gradient,
        use_rct_ycocg: false,
        enable_palette: false,
        enable_squeeze: false,
        adaptive_tree: false,
        entropy_mode: opts.entropy_mode,
        lz77: opts.lz77,
        residual_multiplier: 1,
    };
    let mod_opts_zero = ModularEncodeOptions {
        predictor: Predictor::Zero,
        ..mod_opts_grad
    };

    let gmod_ec_count = extra_channels
        .iter()
        .take_while(|g| g.width <= group_dim && g.height <= group_dim)
        .count();
    if !extra_channels.is_empty() {
        let header = super::modular::ModularHeader {
            use_global_tree: false,
            wp_params: super::modular::WpHeader::default(),
            transform: Vec::new(),
        };
        header.write(&mut lf_global_bw)?;
        MaConfig::write_single_leaf_tree(&mut lf_global_bw, Predictor::Gradient)?;
        let mut enc = EntropyEncoder::single_cluster_with_options(
            1,
            HybridUintConfig::DEFAULT_4_2_0,
            opts.entropy_mode,
            opts.lz77,
        );
        let dist_multiplier = extra_channels[..gmod_ec_count]
            .iter()
            .map(|g| g.width as u32)
            .max()
            .unwrap_or(0);
        for grid in &extra_channels[..gmod_ec_count] {
            record_gradient_channel(&mut enc, grid, dist_multiplier);
        }
        enc.finish(&mut lf_global_bw)?;
    }

    // 2. Encode each `LfGroup[lf_idx]`.
    let mut lf_group_bws = Vec::with_capacity(num_lf_groups);
    for lf_idx in 0..num_lf_groups {
        let mut bw = BitWriter::new();
        let gx = lf_idx % num_lf_cols;
        let gy = lf_idx / num_lf_cols;
        let bx0 = gx * lf_dim_blocks;
        let by0 = gy * lf_dim_blocks;
        let lfw_b = bw_total.saturating_sub(bx0).min(lf_dim_blocks);
        let lfh_b = bh_total.saturating_sub(by0).min(lf_dim_blocks);

        // `LfCoeff`: extra_precision (2 bits), followed by `[Y, X, B]` Modular subimage.
        bw.write_bits(2, u32::from(extra_precision));
        let y_sub = dc_grids[1].extract_rect(bx0, by0, lfw_b, lfh_b);
        let x_sub = dc_grids[0].extract_rect(bx0, by0, lfw_b, lfh_b);
        let b_sub = dc_grids[2].extract_rect(bx0, by0, lfw_b, lfh_b);
        encode_modular_subimage_v2(&mut bw, &[y_sub, x_sub, b_sub], &mod_opts_grad)?;

        // Collect `block_info_raw` entries for this `LfGroup` in raster order.
        let mut block_entries: Vec<(i32, i32)> = Vec::with_capacity(lfw_b * lfh_b);
        for by in 0..lfh_b {
            for bx in 0..lfw_b {
                if let BlockInfo::Data { dct_select, hf_mul } =
                    block_info_grid[(by0 + by) * bw_total + (bx0 + bx)]
                {
                    block_entries.push((dct_select as i32, hf_mul - 1));
                }
            }
        }
        let nb_blocks = block_entries.len().max(1) as u32;
        let nb_bits = (lfw_b * lfh_b).next_power_of_two().trailing_zeros() as usize;
        bw.write_bits(nb_bits, nb_blocks - 1);

        let cfl_w = lfw_b.div_ceil(8);
        let cfl_h = lfh_b.div_ceil(8);
        let x_from_y = x_from_y_full.extract_rect(bx0 / 8, by0 / 8, cfl_w, cfl_h);
        let b_from_y = b_from_y_full.extract_rect(bx0 / 8, by0 / 8, cfl_w, cfl_h);
        let mut block_info_raw = ModularGrid::new(nb_blocks as usize, 2);
        for (idx, &(ds, qm)) in block_entries.iter().enumerate() {
            block_info_raw.set(idx, 0, ds);
            block_info_raw.set(idx, 1, qm);
        }
        let sharpness = sharpness_grid.extract_rect(bx0, by0, lfw_b, lfh_b);
        encode_modular_subimage_v2(
            &mut bw,
            &[x_from_y, b_from_y, block_info_raw, sharpness],
            &mod_opts_zero,
        )?;
        lf_group_bws.push(bw);
    }

    // 3. Collect HF coefficient tokens for all `num_groups` pass groups using 1 block cluster (`495` contexts).
    let dct16_natural_order = compute_natural_order((16, 16));
    let mut hf_enc = EntropyEncoder::single_cluster_with_options(
        495,
        HybridUintConfig::DEFAULT_4_2_0,
        opts.entropy_mode,
        false,
    );
    let mut group_token_counts = Vec::with_capacity(num_groups);

    for g_idx in 0..num_groups {
        let gx = g_idx % num_cols;
        let gy = g_idx / num_cols;
        let bx0 = gx * group_dim_blocks;
        let by0 = gy * group_dim_blocks;
        let gw_b = bw_total.saturating_sub(bx0).min(group_dim_blocks);
        let gh_b = bh_total.saturating_sub(by0).min(group_dim_blocks);

        let mut token_count = 0usize;
        let mut non_zeros_row = [vec![0u32; gw_b], vec![0u32; gw_b], vec![0u32; gw_b]];

        for by in 0..gh_b {
            for bx in 0..gw_b {
                let BlockInfo::Data { dct_select, .. } =
                    block_info_grid[(by0 + by) * bw_total + (bx0 + bx)]
                else {
                    continue;
                };
                let (w8, h8) = dct_select.dct_select_size();
                let num_blocks = w8 * h8;
                let num_blocks_log = num_blocks.trailing_zeros();
                let order_slice: &[(u16, u16)] = if dct_select == TransformType::Dct16 {
                    &dct16_natural_order
                } else {
                    &DCT8_NATURAL_ORDER
                };

                for c_iter in 0..3 {
                    let c = [1usize, 0, 2][c_iter]; // Y, X, B
                    let predicted = if by == 0 {
                        if bx == 0 {
                            32
                        } else {
                            non_zeros_row[c][bx - 1]
                        }
                    } else if bx == 0 {
                        non_zeros_row[c][bx]
                    } else {
                        (non_zeros_row[c][bx] + non_zeros_row[c][bx - 1] + 1) >> 1
                    };
                    let nz_bucket = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    let non_zeros_ctx = nz_bucket;

                    let ac_order = &order_slice[num_blocks as usize..];
                    let mut ac_vals = Vec::with_capacity(ac_order.len());
                    let mut non_zeros = 0u32;
                    for &(mut dx, mut dy) in ac_order {
                        if dct_select.need_transpose() {
                            std::mem::swap(&mut dx, &mut dy);
                        }
                        let px = (bx0 + bx) * 8 + dx as usize;
                        let py = (by0 + by) * 8 + dy as usize;
                        let v = hf_q_grids[c].get(px, py);
                        ac_vals.push(v);
                        if v != 0 {
                            non_zeros += 1;
                        }
                    }

                    hf_enc.record_varint(non_zeros_ctx, non_zeros);
                    token_count += 1;
                    let nz_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                    for dx in 0..w8 as usize {
                        if bx + dx < gw_b {
                            non_zeros_row[c][bx + dx] = nz_val;
                        }
                    }

                    if non_zeros == 0 {
                        continue;
                    }

                    let mut is_prev_nonzero = (non_zeros <= num_blocks * 4) as u32;
                    let coeff_ctx_base = 37u32;
                    let mut rem_nz = non_zeros;
                    for (idx, &v) in ac_vals.iter().enumerate() {
                        let nz_norm = ((rem_nz - 1) >> num_blocks_log) as usize;
                        let idx_norm = idx >> num_blocks_log;
                        let coeff_ctx = coeff_ctx_base
                            + (COEFF_NUM_NONZERO_CONTEXT[nz_norm] + COEFF_FREQ_CONTEXT[idx_norm])
                                * 2
                            + is_prev_nonzero;
                        let ucoeff = if v == 0 { 0 } else { pack_signed(v) };
                        hf_enc.record_varint(coeff_ctx, ucoeff);
                        token_count += 1;
                        if v == 0 {
                            is_prev_nonzero = 0;
                        } else {
                            is_prev_nonzero = 1;
                            rem_nz -= 1;
                            if rem_nz == 0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        group_token_counts.push(token_count);
    }

    let mut hf_global_bw = BitWriter::new();
    hf_global_bw.write_bool(true); // `DequantMatrixSet`: all_default = true
    let preset_bits = (num_groups as u32).next_power_of_two().trailing_zeros() as usize;
    hf_global_bw.write_bits(preset_bits, 0);
    hf_global_bw.write_u32(
        0,
        U32Specifier::c(0x5F),
        U32Specifier::c(0x13),
        U32Specifier::c(0x00),
        U32Specifier::u(0, 13),
    )?;

    let (hf_header_bits, mut pass_group_bws) = hf_enc.encode_split_groups(&group_token_counts)?;
    hf_global_bw.append_bits_from(&hf_header_bits);

    // Since `num_hf_presets == 1`, `ceil(log2(num_hf_presets)) == 0`, so `hfp` takes 0 bits.

    if gmod_ec_count < extra_channels.len() {
        for (g_idx, pg_bw) in pass_group_bws.iter_mut().enumerate() {
            let gx = g_idx % num_cols;
            let gy = g_idx / num_cols;
            let x0 = gx * group_dim;
            let y0 = gy * group_dim;
            let mut tiles = Vec::new();
            for ec in &extra_channels[gmod_ec_count..] {
                if x0 < ec.width && y0 < ec.height {
                    tiles.push(ec.extract_rect(x0, y0, group_dim, group_dim));
                }
            }
            if !tiles.is_empty() {
                encode_modular_subimage_v2(pg_bw, &tiles, &mod_opts_grad)?;
            }
        }
    }

    if num_groups == 1 {
        let mut combined = lf_global_bw;
        for b in &lf_group_bws {
            combined.append_bits_from(b);
        }
        combined.append_bits_from(&hf_global_bw);
        for b in &pass_group_bws {
            combined.append_bits_from(b);
        }
        Ok(vec![combined.into_bytes()])
    } else {
        let mut sections = Vec::with_capacity(2 + num_lf_groups + num_groups);
        sections.push(lf_global_bw.into_bytes());
        for b in lf_group_bws {
            sections.push(b.into_bytes());
        }
        sections.push(hf_global_bw.into_bytes());
        for b in pass_group_bws {
            sections.push(b.into_bytes());
        }
        Ok(sections)
    }
}

/// Clean-room VarDCT frame encoder: encodes planar XYB `FloatGrid`s (plus optional Modular extra
/// channels) into frame TOC section payloads.
pub fn encode_vardct_frame_sections(
    xyb: &[FloatGrid; 3],
    extra_channels: &[ModularGrid],
    frame_header: &FrameHeader,
    quality: f32,
) -> Result<Vec<Vec<u8>>, JxlError> {
    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let bw_total = width.div_ceil(8);
    let bh_total = height.div_ceil(8);

    let q_clamped = quality.clamp(1.0, 100.0);
    let global_scale =
        ((2048.0 + (q_clamped / 100.0).powf(2.0) * 63487.0).round() as u32).clamp(2048, 65535);
    let quantizer = Quantizer {
        global_scale,
        quant_lf: 32,
    };
    let lf_dequant = LfChannelDequantization::default();
    let lf_corr = LfChannelCorrelation {
        all_default: false,
        colour_factor: 84,
        base_correlation_x: 0.0,
        base_correlation_b: 0.0,
        x_factor_lf: 128,
        b_factor_lf: 128,
    };
    let dequant_set = DequantMatrixSet::default();

    let mut dc_grids = [
        ModularGrid::new(bw_total, bh_total),
        ModularGrid::new(bw_total, bh_total),
        ModularGrid::new(bw_total, bh_total),
    ];
    let mut hf_q_grids = [
        ModularGrid::new(bw_total * 8, bh_total * 8),
        ModularGrid::new(bw_total * 8, bh_total * 8),
        ModularGrid::new(bw_total * 8, bh_total * 8),
    ];

    let scale_inv = (quantizer.global_scale as u64 * quantizer.quant_lf as u64) as f64;
    let m_lfs = [lf_dequant.m_x_lf, lf_dequant.m_y_lf, lf_dequant.m_b_lf];
    let lf_scales: [f32; 3] = std::array::from_fn(|c| (m_lfs[c] as f64 * 512.0 / scale_inv) as f32);
    let hf_mul_base = 65536.0 / (quantizer.global_scale as f32);

    for c in 0..3 {
        let mat = dequant_set.get_transposed(c, TransformType::Dct8);
        let lf_step = lf_scales[c];
        let mut block = [0.0f32; 64];

        for by in 0..bh_total {
            for bx in 0..bw_total {
                for dy in 0..8 {
                    let sy = (by * 8 + dy).min(height - 1);
                    for dx in 0..8 {
                        let sx = (bx * 8 + dx).min(width - 1);
                        block[dy * 8 + dx] = xyb[c].get(sx, sy);
                    }
                }
                dct_2d(&mut block, 8, 8, DctDirection::Forward);
                let q_dc = (block[0] / lf_step).round() as i32;
                dc_grids[c].set(bx, by, q_dc);

                for dy in 0..8 {
                    for dx in 0..8 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let step = mat[dy * 8 + dx] * hf_mul_base;
                        let coeff = block[dy * 8 + dx];
                        let raw = coeff / step;
                        let q_ac = if raw.abs() < 0.65 {
                            0
                        } else {
                            raw.round() as i32
                        };
                        hf_q_grids[c].set(bx * 8 + dx, by * 8 + dy, q_ac);
                    }
                }
            }
        }
    }

    // 1. Encode `LfGlobal` bitstream.
    let mut lf_global_bw = BitWriter::new();
    lf_dequant.write(&mut lf_global_bw)?;
    quantizer.write(&mut lf_global_bw)?;
    HfBlockContext::write_single_cluster(&mut lf_global_bw);
    lf_corr.write(&mut lf_global_bw)?;
    lf_global_bw.write_bool(false); // GlobalModular: has_global_tree = false

    let group_dim = frame_header.group_dim() as usize;
    let group_dim_blocks = group_dim / 8;
    let lf_dim_blocks = group_dim_blocks * 8;
    let num_lf_cols = bw_total.div_ceil(lf_dim_blocks);
    let num_lf_rows = bh_total.div_ceil(lf_dim_blocks);
    let num_lf_groups = num_lf_cols * num_lf_rows;
    let num_cols = bw_total.div_ceil(group_dim_blocks);
    let num_rows = bh_total.div_ceil(group_dim_blocks);
    let num_groups = num_cols * num_rows;

    let gmod_ec_count = extra_channels
        .iter()
        .take_while(|g| g.width <= group_dim && g.height <= group_dim)
        .count();
    if !extra_channels.is_empty() {
        let header = super::modular::ModularHeader {
            use_global_tree: false,
            wp_params: super::modular::WpHeader::default(),
            transform: Vec::new(),
        };
        header.write(&mut lf_global_bw)?;
        MaConfig::write_single_leaf_tree(&mut lf_global_bw, Predictor::Gradient)?;
        let mut enc = EntropyEncoder::single_cluster(1, HybridUintConfig::DEFAULT_4_2_0);
        let dist_multiplier = extra_channels[..gmod_ec_count]
            .iter()
            .map(|g| g.width as u32)
            .max()
            .unwrap_or(0);
        for grid in &extra_channels[..gmod_ec_count] {
            record_gradient_channel(&mut enc, grid, dist_multiplier);
        }
        enc.finish(&mut lf_global_bw)?;
    }

    // 2. Encode each `LfGroup[lf_idx]`.
    let mut lf_group_bws = Vec::with_capacity(num_lf_groups);
    for lf_idx in 0..num_lf_groups {
        let mut bw = BitWriter::new();
        let gx = lf_idx % num_lf_cols;
        let gy = lf_idx / num_lf_cols;
        let bx0 = gx * lf_dim_blocks;
        let by0 = gy * lf_dim_blocks;
        let lfw_b = bw_total.saturating_sub(bx0).min(lf_dim_blocks);
        let lfh_b = bh_total.saturating_sub(by0).min(lf_dim_blocks);

        // `LfCoeff`: extra_precision = 0 (2 bits), followed by `[Y, X, B]` Modular subimage.
        bw.write_bits(2, 0);
        let y_sub = dc_grids[1].extract_rect(bx0, by0, lfw_b, lfh_b);
        let x_sub = dc_grids[0].extract_rect(bx0, by0, lfw_b, lfh_b);
        let b_sub = dc_grids[2].extract_rect(bx0, by0, lfw_b, lfh_b);
        encode_modular_subimage(&mut bw, &[y_sub, x_sub, b_sub], Predictor::Gradient, false)?;

        // `HfMetadata`:
        let nb_blocks = (lfw_b * lfh_b) as u32;
        let nb_bits = (nb_blocks as usize).next_power_of_two().trailing_zeros() as usize;
        bw.write_bits(nb_bits, nb_blocks - 1);

        let cfl_w = lfw_b.div_ceil(8);
        let cfl_h = lfh_b.div_ceil(8);
        let x_from_y = ModularGrid::new(cfl_w, cfl_h);
        let b_from_y = ModularGrid::new(cfl_w, cfl_h);
        let block_info_raw = ModularGrid::new(nb_blocks as usize, 2);
        let sharpness = ModularGrid::new(lfw_b, lfh_b);
        encode_modular_subimage(
            &mut bw,
            &[x_from_y, b_from_y, block_info_raw, sharpness],
            Predictor::Zero,
            false,
        )?;
        lf_group_bws.push(bw);
    }

    // 3. Collect HF coefficient tokens for all `num_groups` pass groups using 1 block cluster (`495` contexts).
    let mut hf_enc = EntropyEncoder::single_cluster(495, HybridUintConfig::DEFAULT_4_2_0);
    let mut group_token_counts = Vec::with_capacity(num_groups);

    for g_idx in 0..num_groups {
        let gx = g_idx % num_cols;
        let gy = g_idx / num_cols;
        let bx0 = gx * group_dim_blocks;
        let by0 = gy * group_dim_blocks;
        let gw_b = bw_total.saturating_sub(bx0).min(group_dim_blocks);
        let gh_b = bh_total.saturating_sub(by0).min(group_dim_blocks);

        let mut token_count = 0usize;
        let mut non_zeros_row = [vec![0u32; gw_b], vec![0u32; gw_b], vec![0u32; gw_b]];

        for by in 0..gh_b {
            for bx in 0..gw_b {
                for c_iter in 0..3 {
                    let c = [1usize, 0, 2][c_iter]; // Y, X, B
                    let predicted = if by == 0 {
                        if bx == 0 {
                            32
                        } else {
                            non_zeros_row[c][bx - 1]
                        }
                    } else if bx == 0 {
                        non_zeros_row[c][bx]
                    } else {
                        (non_zeros_row[c][bx] + non_zeros_row[c][bx - 1] + 1) >> 1
                    };
                    let nz_bucket = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    let non_zeros_ctx = nz_bucket; // block_ctx = 0, num_block_clusters = 1

                    let mut ac_vals = [0i32; 63];
                    let mut non_zeros = 0u32;
                    for (idx, &(dx, dy)) in DCT8_NATURAL_ORDER[1..].iter().enumerate() {
                        let px = (bx0 + bx) * 8 + dy as usize;
                        let py = (by0 + by) * 8 + dx as usize;
                        let v = hf_q_grids[c].get(px, py);
                        ac_vals[idx] = v;
                        if v != 0 {
                            non_zeros += 1;
                        }
                    }

                    hf_enc.record_varint(non_zeros_ctx, non_zeros);
                    token_count += 1;
                    non_zeros_row[c][bx] = non_zeros;

                    if non_zeros == 0 {
                        continue;
                    }

                    let mut is_prev_nonzero = (non_zeros <= 4) as u32;
                    let coeff_ctx_base = 37u32;
                    let mut rem_nz = non_zeros;
                    for (idx, &v) in ac_vals.iter().enumerate() {
                        let nz_norm = (rem_nz - 1) as usize;
                        let coeff_ctx = coeff_ctx_base
                            + (COEFF_NUM_NONZERO_CONTEXT[nz_norm] + COEFF_FREQ_CONTEXT[idx]) * 2
                            + is_prev_nonzero;
                        let ucoeff = if v == 0 { 0 } else { pack_signed(v) };
                        hf_enc.record_varint(coeff_ctx, ucoeff);
                        token_count += 1;
                        if v == 0 {
                            is_prev_nonzero = 0;
                        } else {
                            is_prev_nonzero = 1;
                            rem_nz -= 1;
                            if rem_nz == 0 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        group_token_counts.push(token_count);
    }

    let mut hf_global_bw = BitWriter::new();
    // `DequantMatrixSet`: all_default = true (1 bit)
    hf_global_bw.write_bool(true);
    // `num_hf_presets - 1 = 0` (`preset_bits` bits)
    let preset_bits = (num_groups as u32).next_power_of_two().trailing_zeros() as usize;
    hf_global_bw.write_bits(preset_bits, 0);
    // `HfPass[0]`: `used_orders = 0` -> selector 2 (`10` in 2 bits)
    hf_global_bw.write_u32(
        0,
        U32Specifier::c(0x5F),
        U32Specifier::c(0x13),
        U32Specifier::c(0x00),
        U32Specifier::u(0, 13),
    )?;

    let (hf_header_bits, mut pass_group_bws) = hf_enc.encode_split_groups(&group_token_counts)?;
    hf_global_bw.append_bits_from(&hf_header_bits);

    // Note: each pass group starts with `hfp = 0` (`preset_bits` bits), followed by the HF tokens.
    // Since `num_hf_presets == 1`, `ceil(log2(num_hf_presets)) == 0`, so `hfp` takes 0 bits.

    if gmod_ec_count < extra_channels.len() {
        for (g_idx, pg_bw) in pass_group_bws.iter_mut().enumerate() {
            let gx = g_idx % num_cols;
            let gy = g_idx / num_cols;
            let x0 = gx * group_dim;
            let y0 = gy * group_dim;
            let mut tiles = Vec::new();
            for ec in &extra_channels[gmod_ec_count..] {
                if x0 < ec.width && y0 < ec.height {
                    tiles.push(ec.extract_rect(x0, y0, group_dim, group_dim));
                }
            }
            if !tiles.is_empty() {
                encode_modular_subimage(pg_bw, &tiles, Predictor::Gradient, false)?;
            }
        }
    }

    if num_groups == 1 {
        let mut combined = lf_global_bw;
        for b in &lf_group_bws {
            combined.append_bits_from(b);
        }
        combined.append_bits_from(&hf_global_bw);
        for b in &pass_group_bws {
            combined.append_bits_from(b);
        }
        Ok(vec![combined.into_bytes()])
    } else {
        let mut sections = Vec::with_capacity(2 + num_lf_groups + num_groups);
        sections.push(lf_global_bw.into_bytes());
        for b in lf_group_bws {
            sections.push(b.into_bytes());
        }
        sections.push(hf_global_bw.into_bytes());
        for b in pass_group_bws {
            sections.push(b.into_bytes());
        }
        Ok(sections)
    }
}

fn record_gradient_channel(enc: &mut EntropyEncoder, grid: &ModularGrid, dist_multiplier: u32) {
    let width = grid.width;
    let height = grid.height;
    if width == 0 || height == 0 {
        return;
    }
    let mut w = 0i32;
    for x in 0..width {
        let s = grid.get(x, 0);
        enc.record_varint_with_multiplier(0, pack_signed(s.wrapping_sub(w)), dist_multiplier);
        w = s;
    }
    for y in 1..height {
        let n0 = grid.get(0, y - 1);
        let s0 = grid.get(0, y);
        enc.record_varint_with_multiplier(0, pack_signed(s0.wrapping_sub(n0)), dist_multiplier);
        w = s0;
        for x in 1..width {
            let nw = grid.get(x - 1, y - 1);
            let n = grid.get(x, y - 1);
            let pred = super::modular::grad_clamped_i32(n, w, nw);
            let s = grid.get(x, y);
            enc.record_varint_with_multiplier(
                0,
                pack_signed(s.wrapping_sub(pred)),
                dist_multiplier,
            );
            w = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jxl::headers::EnumColourEncoding;

    #[test]
    fn test_dct_2d_roundtrip_8x8() {
        let mut orig = [0.0f32; 64];
        for (i, v) in orig.iter_mut().enumerate() {
            *v = ((i * 17 + 5) % 256) as f32 / 255.0;
        }
        let mut buf = orig;
        dct_2d(&mut buf, 8, 8, DctDirection::Forward);
        dct_2d(&mut buf, 8, 8, DctDirection::Inverse);
        for (a, b) in orig.iter().zip(&buf) {
            assert!((a - b).abs() < 1e-5, "DCT 8x8 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_xyb_srgb_roundtrip() {
        let oim = OpsinInverseMatrix::default();
        let enc = ColourEncoding::Enum(EnumColourEncoding::srgb());
        for r in (0..=255).step_by(51) {
            for g in (0..=255).step_by(51) {
                for b in (0..=255).step_by(51) {
                    let rf = srgb_to_linear_f32(r as f32 / 255.0);
                    let gf = srgb_to_linear_f32(g as f32 / 255.0);
                    let bf = srgb_to_linear_f32(b as f32 / 255.0);
                    let [x, y, b_xyb] = linear_srgb_to_xyb_pixel(rf, gf, bf);
                    let mut grids = [
                        FloatGrid {
                            width: 1,
                            height: 1,
                            data: vec![x],
                        },
                        FloatGrid {
                            width: 1,
                            height: 1,
                            data: vec![y],
                        },
                        FloatGrid {
                            width: 1,
                            height: 1,
                            data: vec![b_xyb],
                        },
                    ];
                    xyb_to_target_rgb(&mut grids, &oim, 255.0, &enc);
                    let ro = (grids[0].data[0] * 255.0).round() as i32;
                    let go = (grids[1].data[0] * 255.0).round() as i32;
                    let bo = (grids[2].data[0] * 255.0).round() as i32;
                    assert!((ro - r).abs() <= 1, "R mismatch: {r} vs {ro}");
                    assert!((go - g).abs() <= 1, "G mismatch: {g} vs {go}");
                    assert!((bo - b).abs() <= 1, "B mismatch: {b} vs {bo}");
                }
            }
        }
    }
}
