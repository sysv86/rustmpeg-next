// SPDX-License-Identifier: Apache-2.0
//! APV intra encoder (RFC 9924).

use super::bits::BitWriter;
use super::syntax::{color_codes, qp_offsets};
use super::transform::{DEFAULT_QMATRIX, ZIGZAG, forward_8x8, quantize};
use super::vlc::write_hv;
use super::{ApvConfig, ApvError, ApvFrame, MAX_TILES, max_tile_qp, profile_for, valid_level_idc};

/// APV access-unit signature (`aPv1`).
const SIGNATURE: u32 = 0x6150_7631;

/// APV intra encoder.
#[derive(Debug, Clone)]
pub struct ApvEncoder {
    config: ApvConfig,
}

impl ApvEncoder {
    /// Creates an encoder after validating the level, band, QP, and tile size.
    ///
    /// # Errors
    ///
    /// Rejects an unknown level, a band above 3, a QP above 51, or a tile
    /// smaller than the level minimum.
    pub fn new(config: ApvConfig) -> Result<Self, ApvError> {
        if !valid_level_idc(config.level_idc) {
            return Err(ApvError::Frame("unknown APV level"));
        }
        if config.band_idc > 3 {
            return Err(ApvError::Frame("APV band must be 0..=3"));
        }
        if config.qp > 51 {
            return Err(ApvError::Frame("APV QP must be 0..=51"));
        }
        if config.tile_width_in_mbs < 16 || config.tile_height_in_mbs < 8 {
            return Err(ApvError::Frame("APV tile is smaller than 16x8 MBs"));
        }
        Ok(Self { config })
    }

    /// Returns the encoder configuration.
    #[must_use]
    pub fn config(&self) -> ApvConfig {
        self.config
    }

    /// Encodes one frame as an APV access unit (starting with `aPv1`).
    ///
    /// # Errors
    ///
    /// Rejects invalid frames, unsupported profile/layout combinations, and
    /// syntax-size overflows.
    pub fn encode_access_unit(&self, frame: &ApvFrame) -> Result<Vec<u8>, ApvError> {
        if !frame.is_valid() {
            return Err(ApvError::Frame(
                "frame plane sizes do not match its geometry",
            ));
        }
        let profile_idc = profile_for(frame.chroma_format, frame.bit_depth)?;
        let width_in_mbs = frame.width.div_ceil(16);
        let height_in_mbs = frame.height.div_ceil(16);
        let tile_width = u32::from(self.config.tile_width_in_mbs)
            .max(16)
            .max(width_in_mbs.div_ceil(20));
        let tile_height = u32::from(self.config.tile_height_in_mbs)
            .max(8)
            .max(height_in_mbs.div_ceil(20));
        let tile_cols = width_in_mbs.div_ceil(tile_width);
        let tile_rows = height_in_mbs.div_ceil(tile_height);
        if tile_cols > 20 || tile_rows > 20 || tile_cols * tile_rows > MAX_TILES as u32 {
            return Err(ApvError::Limit("tile count"));
        }
        let mut col_starts = Vec::with_capacity(tile_cols as usize + 1);
        for column in 0..tile_cols {
            col_starts.push(column * tile_width * 16);
        }
        col_starts.push(width_in_mbs * 16);
        let mut row_starts = Vec::with_capacity(tile_rows as usize + 1);
        for row in 0..tile_rows {
            row_starts.push(row * tile_height * 16);
        }
        row_starts.push(height_in_mbs * 16);

        let qp = self
            .config
            .qp
            .checked_add(qp_offsets(frame.bit_depth))
            .ok_or(ApvError::Limit("tile QP"))?;
        if qp > max_tile_qp(frame.bit_depth) {
            return Err(ApvError::Frame("tile QP is out of range"));
        }

        let width = frame.width;
        let height = frame.height;
        let mut header = BitWriter::new();
        header.write_bits(u32::from(profile_idc), 8);
        header.write_bits(u32::from(self.config.level_idc), 8);
        header.write_bits(u32::from(self.config.band_idc), 3);
        header.write_bits(0, 5);
        header.write_bits(width, 24);
        header.write_bits(height, 24);
        header.write_bits(u32::from(frame.chroma_format.idc()), 4);
        header.write_bits(u32::from(frame.bit_depth - 8), 4);
        header.write_bits(u32::from(frame.capture_time_distance), 8);
        header.write_bits(0, 8); // reserved / companding
        header.write_bits(0, 8); // frame-header reserved
        if let Some(color) = &frame.color {
            header.write_bit(1);
            let (primaries, transfer, matrix, full_range) = color_codes(color);
            header.write_bits(u32::from(primaries), 8);
            header.write_bits(u32::from(transfer), 8);
            header.write_bits(u32::from(matrix), 8);
            header.write_bit(u32::from(full_range));
        } else {
            header.write_bit(0);
        }
        header.write_bit(0); // use_q_matrix
        header.write_bits(tile_width, 20);
        header.write_bits(tile_height, 20);
        header.write_bit(0); // tile_size_present_in_fh_flag
        header.write_bits(0, 8); // tile-info reserved
        header.align_zero();
        let mut frame_bytes = header.into_bytes();

        for tile_index in 0..(tile_cols * tile_rows) as usize {
            let column = tile_index % tile_cols as usize;
            let row = tile_index / tile_cols as usize;
            let x0 = col_starts[column];
            let y0 = row_starts[row];
            let mb_cols = (col_starts[column + 1] - x0) / 16;
            let mb_rows = (row_starts[row + 1] - y0) / 16;
            let tile = self.encode_tile(
                frame,
                tile_index,
                x0,
                y0,
                mb_cols,
                mb_rows,
                qp,
                frame.bit_depth,
            )?;
            let size = u32::try_from(tile.len()).map_err(|_| ApvError::Limit("tile size"))?;
            if size == 0 {
                return Err(ApvError::Invalid("empty tile"));
            }
            frame_bytes.extend_from_slice(&size.to_be_bytes());
            frame_bytes.extend_from_slice(&tile);
        }

        let mut payload = Vec::with_capacity(frame_bytes.len() + 4);
        payload.extend_from_slice(&[1, 0, 1, 0]); // primary frame, group 1
        payload.extend_from_slice(&frame_bytes);
        let pbu_size = u32::try_from(payload.len()).map_err(|_| ApvError::Limit("PBU size"))?;
        let mut access_unit = Vec::with_capacity(payload.len() + 8);
        access_unit.extend_from_slice(&SIGNATURE.to_be_bytes());
        access_unit.extend_from_slice(&pbu_size.to_be_bytes());
        access_unit.extend_from_slice(&payload);
        Ok(access_unit)
    }

    /// Builds the `apvC` decoder configuration record for one frame.
    ///
    /// The payload is the `APVDecoderConfigurationRecord` from the APV
    /// ISOBMFF binding, ready to be stored in an `apvC` box by a container
    /// writer.  It carries one configuration entry for a primary frame.
    ///
    /// # Errors
    ///
    /// Rejects an unsupported profile or an invalid level/band.
    pub fn isobmff_config_record(&self, frame: &ApvFrame) -> Result<Vec<u8>, ApvError> {
        if !frame.is_valid() {
            return Err(ApvError::Frame(
                "frame plane sizes do not match its geometry",
            ));
        }
        if !valid_level_idc(self.config.level_idc) || self.config.band_idc > 3 {
            return Err(ApvError::Frame("invalid APV level or band"));
        }
        let profile = profile_for(frame.chroma_format, frame.bit_depth)?;
        let (primaries, transfer, matrix, full_range) = frame
            .color
            .as_ref()
            .map(color_codes)
            .unwrap_or((2, 2, 2, false));
        let mut record = Vec::with_capacity(22);
        record.push(1); // configurationVersion
        record.push(1); // number_of_configuration_entry
        record.push(1); // pbu_type: primary frame
        record.push(1); // number_of_frame_info
        record.push(u8::from(frame.color.is_some()) << 1);
        record.push(profile);
        record.push(self.config.level_idc);
        record.push(self.config.band_idc);
        record.extend_from_slice(&frame.width.to_be_bytes());
        record.extend_from_slice(&frame.height.to_be_bytes());
        record.push((frame.chroma_format.idc() << 4) | (frame.bit_depth - 8));
        record.push(frame.capture_time_distance);
        if frame.color.is_some() {
            record.push(primaries);
            record.push(transfer);
            record.push(matrix);
            record.push(u8::from(full_range) << 7);
        }
        Ok(record)
    }

    /// Encodes one frame as a raw APV bitstream (`au_size` + access unit).
    ///
    /// # Errors
    ///
    /// Propagates access-unit errors and rejects an access unit above 4 GiB.
    pub fn encode_raw_bitstream(&self, frame: &ApvFrame) -> Result<Vec<u8>, ApvError> {
        let access_unit = self.encode_access_unit(frame)?;
        let size = u32::try_from(access_unit.len()).map_err(|_| ApvError::Limit("access unit"))?;
        if size == 0 {
            return Err(ApvError::Invalid("empty access unit"));
        }
        let mut output = Vec::with_capacity(access_unit.len() + 4);
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(&access_unit);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_tile(
        &self,
        frame: &ApvFrame,
        tile_index: usize,
        x0: u32,
        y0: u32,
        mb_cols: u32,
        mb_rows: u32,
        qp: u8,
        bit_depth: u8,
    ) -> Result<Vec<u8>, ApvError> {
        let components = frame.chroma_format.components();
        let mut component_data = Vec::with_capacity(components);
        for component in 0..components {
            let data = encode_component(frame, component, x0, y0, mb_cols, mb_rows, qp, bit_depth)?;
            component_data.push(data);
        }
        let mut header = Vec::with_capacity(5 + components * 5);
        let header_size = 5 + components * 5;
        header.extend_from_slice(&(header_size as u16).to_be_bytes());
        header.extend_from_slice(
            &u16::try_from(tile_index)
                .map_err(|_| ApvError::Limit("tile index"))?
                .to_be_bytes(),
        );
        for data in &component_data {
            let size = u32::try_from(data.len()).map_err(|_| ApvError::Limit("tile data"))?;
            if size == 0 {
                return Err(ApvError::Invalid("empty component data"));
            }
            header.extend_from_slice(&size.to_be_bytes());
        }
        for _ in 0..components {
            header.push(qp);
        }
        header.push(0);
        let mut tile = header;
        for data in component_data {
            tile.extend_from_slice(&data);
        }
        Ok(tile)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_component(
    frame: &ApvFrame,
    component: usize,
    x0: u32,
    y0: u32,
    mb_cols: u32,
    mb_rows: u32,
    qp: u8,
    bit_depth: u8,
) -> Result<Vec<u8>, ApvError> {
    let chroma = frame.chroma_format;
    let sub_w = if component == 0 {
        0
    } else {
        chroma.sub_width()
    };
    let sub_h = if component == 0 {
        0
    } else {
        chroma.sub_height()
    };
    let mb_width = 16_u32 >> sub_w;
    let mb_height = 16_u32 >> sub_h;
    let (plane_width, plane_height) = frame
        .plane_size(component)
        .ok_or(ApvError::Invalid("component index"))?;
    let x0 = x0 >> sub_w;
    let y0 = y0 >> sub_h;
    let center = 1_i32 << (bit_depth - 1);
    let shift = 20 - i32::from(bit_depth);
    let mut writer = BitWriter::new();
    let mut previous_dc = 0_i32;
    let mut previous_dc_diff = 20_i32;
    let mut previous_first_ac = 0_i32;

    for mb_y in 0..mb_rows {
        for mb_x in 0..mb_cols {
            let block_x = x0 + mb_x * mb_width;
            let block_y = y0 + mb_y * mb_height;
            for y in (0..mb_height).step_by(8) {
                for x in (0..mb_width).step_by(8) {
                    let mut residual = [0_i32; 64];
                    for row in 0..8_u32 {
                        for column in 0..8_u32 {
                            let source_x = (block_x + x + column).min(plane_width - 1);
                            let source_y = (block_y + y + row).min(plane_height - 1);
                            let sample = i32::from(
                                frame.planes[component]
                                    [(source_y * plane_width + source_x) as usize],
                            );
                            residual[(column * 8 + row) as usize] = (sample - center) << shift;
                        }
                    }
                    let transformed = forward_8x8(&residual);
                    let coefficients = quantize(&transformed, qp, &DEFAULT_QMATRIX, bit_depth)?;
                    write_block_coefficients(
                        &mut writer,
                        &coefficients,
                        &mut previous_dc,
                        &mut previous_dc_diff,
                        &mut previous_first_ac,
                    )?;
                }
            }
        }
    }
    writer.align_zero();
    Ok(writer.into_bytes())
}

fn write_block_coefficients(
    writer: &mut BitWriter,
    coefficients: &[i16; 64],
    previous_dc: &mut i32,
    previous_dc_diff: &mut i32,
    previous_first_ac: &mut i32,
) -> Result<(), ApvError> {
    let dc = i32::from(coefficients[0]);
    let difference = dc - *previous_dc;
    let absolute = difference.unsigned_abs();
    let k_dc = ((*previous_dc_diff >> 1).clamp(0, 5)) as u32;
    write_hv(writer, absolute, k_dc)?;
    if absolute != 0 {
        writer.write_bit(u32::from(difference < 0));
    }
    *previous_dc = dc;
    *previous_dc_diff = absolute as i32;

    let mut scan_position = 1_usize;
    let mut previous_level = *previous_first_ac;
    let mut previous_run = 0_i32;
    let mut first_ac = true;
    while scan_position < 64 {
        let mut run = 0_usize;
        while scan_position + run < 64 {
            let raster = ZIGZAG[scan_position + run];
            let coefficient = i32::from(coefficients[(raster & 7) * 8 + (raster >> 3)]);
            if coefficient != 0 {
                break;
            }
            run += 1;
        }
        let k_run = ((previous_run >> 2).clamp(0, 2)) as u32;
        write_hv(writer, run as u32, k_run)?;
        scan_position += run;
        previous_run = run as i32;
        if scan_position >= 64 {
            break;
        }
        let raster = ZIGZAG[scan_position];
        let coefficient = i32::from(coefficients[(raster & 7) * 8 + (raster >> 3)]);
        let level = coefficient.unsigned_abs();
        let k_level = ((previous_level >> 2).clamp(0, 4)) as u32;
        write_hv(writer, level - 1, k_level)?;
        writer.write_bit(u32::from(coefficient < 0));
        previous_level = level as i32;
        if first_ac {
            first_ac = false;
            *previous_first_ac = previous_level;
        }
        scan_position += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apv::ApvChromaFormat;

    #[test]
    fn config_record_matches_the_apv_isobmff_layout() {
        let frame = ApvFrame::new(ApvChromaFormat::Yuv422, 10, 320, 240).expect("frame");
        let encoder = ApvEncoder::new(ApvConfig {
            level_idc: 123,
            band_idc: 3,
            ..ApvConfig::default()
        })
        .expect("encoder");
        let record = encoder.isobmff_config_record(&frame).expect("record");
        assert_eq!(
            record,
            vec![
                1, 1, 1, 1, 0, 33, 123, 3, 0, 0, 1, 64, 0, 0, 0, 240, 0x22, 0
            ]
        );
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(
            ApvEncoder::new(ApvConfig {
                tile_width_in_mbs: 8,
                ..ApvConfig::default()
            })
            .is_err()
        );
        assert!(
            ApvEncoder::new(ApvConfig {
                qp: 52,
                ..ApvConfig::default()
            })
            .is_err()
        );
    }
}
