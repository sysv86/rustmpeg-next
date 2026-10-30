// SPDX-License-Identifier: Apache-2.0
//! APV decoder (RFC 9924, Sections 5 and 6).

use super::bits::BitReader;
use super::syntax::{FrameHeader, TileHeader};
use super::transform::{ZIGZAG, dequantize, inverse_8x8, output_samples};
use super::vlc::read_hv;
use super::{ApvError, ApvFrame, MAX_TILES};

/// APV access-unit signature (`aPv1`).
const SIGNATURE: u32 = 0x6150_7631;
/// Largest access unit accepted by the decoder (256 MiB).
const MAX_AU_BYTES: usize = 256 * 1024 * 1024;

/// Stateless APV decoder.  APV frames are intra-only and independent.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApvDecoder;

impl ApvDecoder {
    /// Creates a decoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decodes one access unit and returns its primary frame.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a bad signature, a truncated or malformed
    /// PBU/frame/tile, an unsupported profile, or an exceeded limit.
    pub fn decode_access_unit(&self, data: &[u8]) -> Result<ApvFrame, ApvError> {
        if data.len() > MAX_AU_BYTES {
            return Err(ApvError::Limit("access unit"));
        }
        if data.len() < 4 || read_u32(data, 0)? != SIGNATURE {
            return Err(ApvError::InvalidSignature);
        }
        let mut position = 4;
        let mut primary = None;
        let mut pbu_count = 0_usize;
        while position < data.len() {
            pbu_count += 1;
            if pbu_count > 1_000_000 {
                return Err(ApvError::Limit("PBU count"));
            }
            let pbu_size = read_u32(data, position)? as usize;
            position += 4;
            if pbu_size < 4 || pbu_size == usize::MAX {
                return Err(ApvError::Invalid("invalid pbu_size"));
            }
            let end = position
                .checked_add(pbu_size)
                .ok_or(ApvError::Limit("PBU size"))?;
            if end > data.len() {
                return Err(ApvError::Invalid("PBU exceeds the access unit"));
            }
            let pbu_type = *data
                .get(position)
                .ok_or(ApvError::Invalid("truncated PBU header"))?;
            let group_id = u16::from_be_bytes([
                *data
                    .get(position + 1)
                    .ok_or(ApvError::Invalid("truncated PBU"))?,
                *data
                    .get(position + 2)
                    .ok_or(ApvError::Invalid("truncated PBU"))?,
            ]);
            let reserved = *data
                .get(position + 3)
                .ok_or(ApvError::Invalid("truncated PBU header"))?;
            let payload = &data[position + 4..end];
            position = end;
            if group_id == 0 && pbu_type <= 64 {
                return Err(ApvError::Invalid("group_id zero on a frame PBU"));
            }
            if reserved != 0 {
                // RFC 9924: decoders must ignore PBUs with non-zero reserved
                // header bits.
                continue;
            }
            match pbu_type {
                1 => {
                    if primary.is_some() {
                        return Err(ApvError::Invalid("multiple primary frames"));
                    }
                    primary = Some(decode_frame(payload)?);
                }
                2 | 25 | 26 | 27 => {
                    // Non-primary, preview, depth, and alpha frames are not
                    // returned by this entry point.
                }
                65..=67 => {}
                _ => return Err(ApvError::Unsupported("reserved PBU type")),
            }
        }
        primary.ok_or(ApvError::Invalid("access unit has no primary frame"))
    }

    /// Decodes a raw APV bitstream (a sequence of `au_size` + access unit).
    ///
    /// # Errors
    ///
    /// Propagates access-unit errors and rejects truncated `au_size` fields.
    pub fn decode_raw_bitstream(&self, data: &[u8]) -> Result<Vec<ApvFrame>, ApvError> {
        let mut frames = Vec::new();
        let mut position = 0;
        while position < data.len() {
            let au_size = read_u32(data, position)? as usize;
            position += 4;
            if au_size == 0 || au_size == usize::MAX {
                return Err(ApvError::Invalid("invalid au_size"));
            }
            let end = position
                .checked_add(au_size)
                .ok_or(ApvError::Limit("access unit size"))?;
            if end > data.len() {
                return Err(ApvError::Invalid("access unit exceeds the buffer"));
            }
            frames.push(self.decode_access_unit(&data[position..end])?);
            position = end;
        }
        Ok(frames)
    }
}

fn decode_frame(payload: &[u8]) -> Result<ApvFrame, ApvError> {
    let mut reader = BitReader::new(payload);
    let header = FrameHeader::parse(&mut reader)?;
    let info = header.info;
    let mut frame = ApvFrame::new(info.chroma_format, info.bit_depth, info.width, info.height)?;
    frame.capture_time_distance = info.capture_time_distance;
    frame.color = header.color;

    for tile_index in 0..header.num_tiles {
        let tile_size = reader.read_bits(32)? as usize;
        if tile_size == 0 || tile_size == u32::MAX as usize {
            return Err(ApvError::Invalid("invalid tile size"));
        }
        if let Some(sizes) = &header.tile_sizes_in_fh
            && sizes[tile_index] as usize != tile_size
        {
            return Err(ApvError::Invalid("tile size differs from the frame header"));
        }
        if !reader.is_aligned() {
            return Err(ApvError::Invalid("tile is not byte-aligned"));
        }
        let start = reader.byte_position();
        let end = start
            .checked_add(tile_size)
            .ok_or(ApvError::Limit("tile size"))?;
        let tile = payload
            .get(start..end)
            .ok_or(ApvError::Invalid("tile exceeds the frame"))?;
        decode_tile(tile, tile_index, &header, &mut frame)?;
        reader.skip_bits(tile_size * 8)?;
    }

    // The remainder of the frame is `filler()`, which is a run of 0xFF bytes.
    let rest = &payload[reader.byte_position()..];
    if rest.iter().any(|byte| *byte != 0xFF) {
        return Err(ApvError::Invalid("non-filler byte after the last tile"));
    }
    Ok(frame)
}

fn decode_tile(
    tile: &[u8],
    tile_index: usize,
    header: &FrameHeader,
    frame: &mut ApvFrame,
) -> Result<(), ApvError> {
    if header.num_tiles > MAX_TILES {
        return Err(ApvError::Limit("tile count"));
    }
    let components = header.info.chroma_format.components();
    let tile_header = TileHeader::parse(tile, tile_index, components, header.info.bit_depth)?;
    let (x0, y0, mb_cols, mb_rows) = header.tile_mb_span(tile_index)?;
    let mut offset = tile_header.header_size;
    for component in 0..components {
        let size = tile_header.data_sizes[component] as usize;
        let end = offset
            .checked_add(size)
            .ok_or(ApvError::Limit("tile data size"))?;
        let data = tile
            .get(offset..end)
            .ok_or(ApvError::Invalid("tile data exceeds the tile"))?;
        decode_component(
            data,
            component,
            header,
            &tile_header,
            frame,
            x0,
            y0,
            mb_cols,
            mb_rows,
        )?;
        offset = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_component(
    data: &[u8],
    component: usize,
    header: &FrameHeader,
    tile: &TileHeader,
    frame: &mut ApvFrame,
    x0: u32,
    y0: u32,
    mb_cols: u32,
    mb_rows: u32,
) -> Result<(), ApvError> {
    let chroma = header.info.chroma_format;
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
    let mut reader = BitReader::new(data);
    let mut previous_dc = 0_i32;
    let mut previous_dc_diff = 20_i32;
    let mut previous_first_ac = 0_i32;

    let x0 = x0 >> sub_w;
    let y0 = y0 >> sub_h;
    for mb_y in 0..mb_rows {
        for mb_x in 0..mb_cols {
            let block_x = x0 + mb_x * mb_width;
            let block_y = y0 + mb_y * mb_height;
            for y in (0..mb_height).step_by(8) {
                for x in (0..mb_width).step_by(8) {
                    let mut coefficients = [0_i32; 64];
                    let dc = read_dc(&mut reader, &mut previous_dc, &mut previous_dc_diff)?;
                    coefficients[0] = dc;
                    read_ac(&mut reader, &mut coefficients, &mut previous_first_ac)?;
                    let scaled = dequantize(
                        &coefficients,
                        tile.qp[component],
                        &header.qmatrix[component],
                        header.info.bit_depth,
                    )?;
                    let block = inverse_8x8(&scaled);
                    let samples = output_samples(&block, header.info.bit_depth);
                    write_block(
                        frame,
                        component,
                        plane_width,
                        plane_height,
                        block_x + x,
                        block_y + y,
                        &samples,
                    );
                }
            }
        }
    }
    reader.align_zero()?;
    Ok(())
}

fn read_dc(
    reader: &mut BitReader<'_>,
    previous_dc: &mut i32,
    previous_dc_diff: &mut i32,
) -> Result<i32, ApvError> {
    let k_param = ((*previous_dc_diff >> 1).clamp(0, 5)) as u32;
    let absolute = read_hv(reader, k_param)?;
    if absolute > u16::MAX as u32 {
        return Err(ApvError::Invalid("DC difference is too large"));
    }
    let dc = if absolute == 0 {
        *previous_dc
    } else {
        let sign = reader.read_bit()?;
        let delta = absolute as i32;
        if sign == 0 {
            previous_dc
                .checked_add(delta)
                .ok_or(ApvError::Invalid("DC coefficient overflow"))?
        } else {
            previous_dc
                .checked_sub(delta)
                .ok_or(ApvError::Invalid("DC coefficient overflow"))?
        }
    };
    if !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&dc) {
        return Err(ApvError::Invalid("DC coefficient is out of range"));
    }
    *previous_dc = dc;
    *previous_dc_diff = absolute as i32;
    Ok(dc)
}

fn read_ac(
    reader: &mut BitReader<'_>,
    coefficients: &mut [i32; 64],
    previous_first_ac: &mut i32,
) -> Result<(), ApvError> {
    let mut scan_position = 1_usize;
    let mut previous_level = *previous_first_ac;
    let mut previous_run = 0_i32;
    let mut first_ac = true;
    while scan_position < 64 {
        let k_run = ((previous_run >> 2).clamp(0, 2)) as u32;
        let run = read_hv(reader, k_run)? as usize;
        if run > 64 - scan_position {
            return Err(ApvError::Invalid("AC zero run exceeds the block"));
        }
        scan_position += run;
        previous_run = run as i32;
        if scan_position >= 64 {
            break;
        }
        let k_level = ((previous_level >> 2).clamp(0, 4)) as u32;
        let magnitude = read_hv(reader, k_level)?;
        if magnitude >= i16::MAX as u32 {
            return Err(ApvError::Invalid("AC level is too large"));
        }
        let level = magnitude as i32 + 1;
        let signed = if reader.read_bit()? == 0 {
            level
        } else {
            -level
        };
        let raster = ZIGZAG[scan_position];
        let x = raster & 7;
        let y = raster >> 3;
        coefficients[x * 8 + y] = signed;
        scan_position += 1;
        previous_level = level;
        if first_ac {
            first_ac = false;
            *previous_first_ac = previous_level;
        }
    }
    Ok(())
}

fn write_block(
    frame: &mut ApvFrame,
    component: usize,
    plane_width: u32,
    plane_height: u32,
    x: u32,
    y: u32,
    samples: &[u16; 64],
) {
    let Some(plane) = frame.planes.get_mut(component) else {
        return;
    };
    for row in 0..8_u32 {
        let destination_y = y + row;
        if destination_y >= plane_height {
            continue;
        }
        for column in 0..8_u32 {
            let destination_x = x + column;
            if destination_x >= plane_width {
                continue;
            }
            let destination = (destination_y * plane_width + destination_x) as usize;
            let source = (column * 8 + row) as usize;
            plane[destination] = samples[source];
        }
    }
}

fn read_u32(data: &[u8], position: usize) -> Result<u32, ApvError> {
    let bytes = data.get(position..position + 4).ok_or(ApvError::Eof)?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_signature_and_truncation() {
        let decoder = ApvDecoder::new();
        assert!(decoder.decode_access_unit(&[]).is_err());
        assert!(decoder.decode_access_unit(b"nope").is_err());
        assert!(decoder.decode_access_unit(&[0, 0, 0, 0]).is_err());
        assert!(decoder.decode_raw_bitstream(&[0, 0, 0]).is_err());
    }
}
