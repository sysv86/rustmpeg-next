// SPDX-License-Identifier: Apache-2.0
//! APV frame, tile, and access-unit syntax (RFC 9924, Sections 5.3 and 9).

use rustmpeg_core::{
    ChromaLocation, ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients,
    TransferFunction,
};

use super::bits::BitReader;
use super::transform::DEFAULT_QMATRIX;
use super::{
    ApvChromaFormat, ApvError, MAX_DIMENSION, MAX_LUMA_SAMPLES, MAX_TILES, max_tile_qp,
    profile_allows, qp_bd_offset, valid_level_idc,
};

/// Frame information shared by `frame_info` and `apvC` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameInfo {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub band_idc: u8,
    pub width: u32,
    pub height: u32,
    pub chroma_format: ApvChromaFormat,
    pub bit_depth: u8,
    pub capture_time_distance: u8,
}

impl FrameInfo {
    pub(crate) fn parse(reader: &mut BitReader<'_>) -> Result<Self, ApvError> {
        let profile_idc = reader.read_bits(8)? as u8;
        let level_idc = reader.read_bits(8)? as u8;
        let band_idc = reader.read_bits(3)? as u8;
        if reader.read_bits(5)? != 0 {
            return Err(ApvError::Invalid("reserved frame-info bits are set"));
        }
        let width = reader.read_bits(24)?;
        let height = reader.read_bits(24)?;
        let chroma_idc = reader.read_bits(4)? as u8;
        let bit_depth = reader.read_bits(4)? as u8 + 8;
        let capture_time_distance = reader.read_bits(8)? as u8;
        if reader.read_bits(8)? != 0 {
            return Err(ApvError::Unsupported("APV companding"));
        }
        let info = Self {
            profile_idc,
            level_idc,
            band_idc,
            width,
            height,
            chroma_format: ApvChromaFormat::from_idc(chroma_idc)?,
            bit_depth,
            capture_time_distance,
        };
        info.validate()?;
        Ok(info)
    }

    pub(crate) fn validate(&self) -> Result<(), ApvError> {
        if !profile_allows(self.profile_idc, self.chroma_format, self.bit_depth) {
            return Err(ApvError::Unsupported(
                "profile/chroma/bit-depth combination",
            ));
        }
        if !valid_level_idc(self.level_idc) {
            return Err(ApvError::Unsupported("APV level_idc"));
        }
        if self.band_idc > 3 {
            return Err(ApvError::Unsupported("APV band_idc"));
        }
        if self.width == 0 || self.height == 0 {
            return Err(ApvError::Invalid("zero frame dimension"));
        }
        if self.width > MAX_DIMENSION || self.height > MAX_DIMENSION {
            return Err(ApvError::Limit("frame dimension"));
        }
        if u64::from(self.width) * u64::from(self.height) > MAX_LUMA_SAMPLES {
            return Err(ApvError::Limit("luma samples"));
        }
        if self.chroma_format == ApvChromaFormat::Yuv422 && !self.width.is_multiple_of(2) {
            return Err(ApvError::Invalid("4:2:2 width is odd"));
        }
        Ok(())
    }

    pub(crate) fn width_in_mbs(&self) -> u32 {
        self.width.div_ceil(16)
    }

    pub(crate) fn height_in_mbs(&self) -> u32 {
        self.height.div_ceil(16)
    }
}

/// Parsed frame header, including tile geometry.
#[derive(Debug, Clone)]
pub(crate) struct FrameHeader {
    pub info: FrameInfo,
    pub color: Option<ColorDescription>,
    pub qmatrix: Vec<[u8; 64]>,
    pub col_starts: Vec<u32>,
    pub row_starts: Vec<u32>,
    pub tile_cols: usize,
    pub num_tiles: usize,
    pub tile_sizes_in_fh: Option<Vec<u32>>,
}

impl FrameHeader {
    pub(crate) fn parse(reader: &mut BitReader<'_>) -> Result<Self, ApvError> {
        let info = FrameInfo::parse(reader)?;
        if reader.read_bits(8)? != 0 {
            return Err(ApvError::Invalid("reserved frame-header bits are set"));
        }
        let color = if reader.read_bit()? != 0 {
            let primaries = reader.read_bits(8)? as u8;
            let transfer = reader.read_bits(8)? as u8;
            let matrix = reader.read_bits(8)? as u8;
            let full_range = reader.read_bit()? != 0;
            Some(color_from_codes(primaries, transfer, matrix, full_range))
        } else {
            None
        };
        let use_q_matrix = reader.read_bit()? != 0;
        let mut qmatrix = vec![DEFAULT_QMATRIX; info.chroma_format.components()];
        if use_q_matrix {
            for matrix in &mut qmatrix {
                for entry in matrix {
                    let value = reader.read_bits(8)? as u8;
                    if value == 0 {
                        return Err(ApvError::Invalid("zero quantization matrix entry"));
                    }
                    *entry = value;
                }
            }
        }

        let tile_width_in_mbs = reader.read_bits(20)?;
        let tile_height_in_mbs = reader.read_bits(20)?;
        if tile_width_in_mbs < 16 || tile_height_in_mbs < 8 {
            return Err(ApvError::Invalid("tile is smaller than the level minimum"));
        }
        let width_in_mbs = info.width_in_mbs();
        let height_in_mbs = info.height_in_mbs();
        let mut col_starts = Vec::new();
        let mut start = 0_u32;
        while start < width_in_mbs {
            col_starts.push(start * 16);
            start = start
                .checked_add(tile_width_in_mbs)
                .ok_or(ApvError::Limit("tile column"))?;
        }
        col_starts.push(width_in_mbs * 16);
        let mut row_starts = Vec::new();
        let mut start = 0_u32;
        while start < height_in_mbs {
            row_starts.push(start * 16);
            start = start
                .checked_add(tile_height_in_mbs)
                .ok_or(ApvError::Limit("tile row"))?;
        }
        row_starts.push(height_in_mbs * 16);
        let tile_cols = col_starts.len() - 1;
        let tile_rows = row_starts.len() - 1;
        if tile_cols > 20 || tile_rows > 20 || tile_cols * tile_rows > MAX_TILES {
            return Err(ApvError::Limit("tile count"));
        }
        let tile_sizes_in_fh = if reader.read_bit()? != 0 {
            let mut sizes = Vec::with_capacity(tile_cols * tile_rows);
            for _ in 0..tile_cols * tile_rows {
                let size = reader.read_bits(32)?;
                if size == 0 {
                    return Err(ApvError::Invalid("zero tile size in frame header"));
                }
                sizes.push(size);
            }
            Some(sizes)
        } else {
            None
        };
        if reader.read_bits(8)? != 0 {
            return Err(ApvError::Invalid("reserved tile-info bits are set"));
        }
        reader.align_zero()?;

        Ok(Self {
            info,
            color,
            qmatrix,
            col_starts,
            row_starts,
            tile_cols,
            num_tiles: tile_cols * tile_rows,
            tile_sizes_in_fh,
        })
    }

    pub(crate) fn tile_columns(&self, tile_index: usize) -> Result<(usize, usize), ApvError> {
        if tile_index >= self.num_tiles {
            return Err(ApvError::Invalid("tile index is out of range"));
        }
        Ok((tile_index % self.tile_cols, tile_index / self.tile_cols))
    }

    pub(crate) fn tile_mb_span(&self, tile_index: usize) -> Result<(u32, u32, u32, u32), ApvError> {
        let (column, row) = self.tile_columns(tile_index)?;
        let x0 = self.col_starts[column];
        let y0 = self.row_starts[row];
        let width_mbs = (self.col_starts[column + 1] - x0) / 16;
        let height_mbs = (self.row_starts[row + 1] - y0) / 16;
        Ok((x0, y0, width_mbs, height_mbs))
    }
}

/// Parsed tile header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileHeader {
    pub data_sizes: [u32; 4],
    pub qp: [u8; 4],
    pub header_size: usize,
}

impl TileHeader {
    pub(crate) fn parse(
        tile: &[u8],
        tile_index: usize,
        components: usize,
        bit_depth: u8,
    ) -> Result<Self, ApvError> {
        let mut reader = BitReader::new(tile);
        let header_size = usize::try_from(reader.read_bits(16)?)
            .map_err(|_| ApvError::Limit("tile header size"))?;
        let expected = 5 + components * 5;
        if header_size != expected {
            return Err(ApvError::Invalid(
                "tile_header_size does not match the component count",
            ));
        }
        let index =
            u16::try_from(reader.read_bits(16)?).map_err(|_| ApvError::Invalid("tile index"))?;
        if usize::from(index) != tile_index {
            return Err(ApvError::Invalid("tile_index does not match raster order"));
        }
        let mut data_sizes = [0_u32; 4];
        let mut total = header_size as u64;
        for size in data_sizes.iter_mut().take(components) {
            let value = reader.read_bits(32)?;
            if value == 0 {
                return Err(ApvError::Invalid("zero tile_data_size"));
            }
            *size = value;
            total = total
                .checked_add(u64::from(value))
                .ok_or(ApvError::Limit("tile size"))?;
        }
        if total > tile.len() as u64 {
            return Err(ApvError::Invalid("tile data exceeds the tile size"));
        }
        let mut qp = [0_u8; 4];
        for value in qp.iter_mut().take(components) {
            let q = reader.read_bits(8)? as u8;
            if q > max_tile_qp(bit_depth) {
                return Err(ApvError::Invalid("tile_qp is out of range"));
            }
            *value = q;
        }
        if reader.read_bits(8)? != 0 {
            return Err(ApvError::Invalid("reserved tile-header bits are set"));
        }
        reader.align_zero()?;
        if reader.byte_position() != header_size {
            return Err(ApvError::Invalid(
                "tile header size does not match its fields",
            ));
        }
        Ok(Self {
            data_sizes,
            qp,
            header_size,
        })
    }
}

pub(crate) fn color_from_codes(
    primaries: u8,
    transfer: u8,
    matrix: u8,
    full_range: bool,
) -> ColorDescription {
    ColorDescription {
        range: if full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
        primaries: match primaries {
            1 => ColorPrimaries::Bt709,
            9 => ColorPrimaries::Bt2020,
            12 => ColorPrimaries::DisplayP3,
            _ => ColorPrimaries::Unknown,
        },
        transfer: match transfer {
            1 => TransferFunction::Sdr,
            16 => TransferFunction::Pq,
            18 => TransferFunction::Hlg,
            8 => TransferFunction::Linear,
            13 => TransferFunction::Srgb,
            _ => TransferFunction::Unknown,
        },
        matrix: match matrix {
            1 => MatrixCoefficients::Bt709,
            9 => MatrixCoefficients::Bt2020NonConstant,
            0 => MatrixCoefficients::Identity,
            _ => MatrixCoefficients::Unknown,
        },
        chroma_location: ChromaLocation::Left,
    }
}

pub(crate) fn color_codes(color: &ColorDescription) -> (u8, u8, u8, bool) {
    let primaries = match color.primaries {
        ColorPrimaries::Bt709 => 1,
        ColorPrimaries::Bt2020 => 9,
        ColorPrimaries::DisplayP3 => 12,
        ColorPrimaries::Unknown => 2,
    };
    let transfer = match color.transfer {
        TransferFunction::Sdr => 1,
        TransferFunction::Pq => 16,
        TransferFunction::Hlg => 18,
        TransferFunction::Linear => 8,
        TransferFunction::Srgb => 13,
        TransferFunction::Unknown => 2,
    };
    let matrix = match color.matrix {
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Bt2020NonConstant => 9,
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::Unknown => 2,
    };
    (primaries, transfer, matrix, color.range == ColorRange::Full)
}

pub(crate) fn qp_offsets(bit_depth: u8) -> u8 {
    qp_bd_offset(bit_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_geometry_covers_padded_macroblocks() {
        let header = FrameHeader {
            info: FrameInfo {
                profile_idc: 33,
                level_idc: 123,
                band_idc: 3,
                width: 300,
                height: 140,
                chroma_format: ApvChromaFormat::Yuv422,
                bit_depth: 10,
                capture_time_distance: 0,
            },
            color: None,
            qmatrix: vec![DEFAULT_QMATRIX; 3],
            col_starts: vec![0, 256, 304],
            row_starts: vec![0, 128, 144],
            tile_cols: 2,
            num_tiles: 4,
            tile_sizes_in_fh: None,
        };
        assert_eq!(header.tile_mb_span(0).expect("span"), (0, 0, 16, 8));
        assert_eq!(header.tile_mb_span(1).expect("span"), (256, 0, 3, 8));
        assert_eq!(header.tile_mb_span(2).expect("span"), (0, 128, 16, 1));
        assert_eq!(header.tile_mb_span(3).expect("span"), (256, 128, 3, 1));
    }

    #[test]
    fn colour_codes_round_trip() {
        let color = ColorDescription::BT2020_PQ;
        let (primaries, transfer, matrix, full) = color_codes(&color);
        let decoded = color_from_codes(primaries, transfer, matrix, full);
        assert_eq!(decoded.primaries, color.primaries);
        assert_eq!(decoded.transfer, color.transfer);
        assert_eq!(decoded.matrix, color.matrix);
        assert_eq!(decoded.range, color.range);
    }
}
