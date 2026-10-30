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

//! Container parsers and muxers for AV2 video streams:
//! - **Y4M (`YUV4MPEG2`)**: Uncompressed YUV frame reader and writer (8/10/12-bit, I400/I420/I422/I444).
//! - **IVF (`DKIF` / `AV02`)**: Standard AOMedia IVF packetized bitstream container reader and writer.
//! - **Raw OBU (`.obu` / `.av2`)**: Raw length-delimited AV2 Open Bitstream Unit stream reader and writer.

use std::io::{BufRead, Read, Write};

use crate::av2::error::{Av2Error, Result};
use crate::av2::headers::PixelLayout;
use crate::av2::pixel::Pixel;

/// Metadata extracted from a Y4M (`YUV4MPEG2`) stream header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Y4mHeader {
    pub width: usize,
    pub height: usize,
    pub framerate_num: u32,
    pub framerate_den: u32,
    pub aspect_num: u32,
    pub aspect_den: u32,
    pub bit_depth: u8,
    pub layout: PixelLayout,
}

impl Default for Y4mHeader {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            framerate_num: 30,
            framerate_den: 1,
            aspect_num: 1,
            aspect_den: 1,
            bit_depth: 8,
            layout: PixelLayout::I420,
        }
    }
}

impl Y4mHeader {
    /// Returns the dimensions `(width, height)` of plane `pl` (`0 = Y`, `1 = U`, `2 = V`).
    #[inline]
    pub fn plane_dimensions(&self, pl: usize) -> (usize, usize) {
        if pl == 0 {
            (self.width, self.height)
        } else {
            match self.layout {
                PixelLayout::I400 => (0, 0),
                PixelLayout::I420 => ((self.width + 1) >> 1, (self.height + 1) >> 1),
                PixelLayout::I422 => ((self.width + 1) >> 1, self.height),
                PixelLayout::I444 => (self.width, self.height),
            }
        }
    }

    /// Number of bytes per sample in the raw Y4M stream (`1` for 8-bit, `2` for 10/12-bit little-endian).
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 { 2 } else { 1 }
    }
}

/// Raw uncompressed frame buffer read from or written to a Y4M stream.
#[derive(Debug, Clone)]
pub struct RawFrame<P: Pixel> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
    pub layout: PixelLayout,
    pub y: Vec<P>,
    pub u: Vec<P>,
    pub v: Vec<P>,
}

impl<P: Pixel> RawFrame<P> {
    /// Allocates a new blank `RawFrame<P>` initialized with neutral mid-gray (`1 << (bit_depth - 1)`).
    pub fn new(width: usize, height: usize, bit_depth: u8, layout: PixelLayout) -> Self {
        let mid = P::from_i32(1 << (bit_depth - 1));
        let y_len = width * height;
        let (uv_w, uv_h) = match layout {
            PixelLayout::I400 => (0, 0),
            PixelLayout::I420 => ((width + 1) >> 1, (height + 1) >> 1),
            PixelLayout::I422 => ((width + 1) >> 1, height),
            PixelLayout::I444 => (width, height),
        };
        let uv_len = uv_w * uv_h;
        Self {
            width,
            height,
            bit_depth,
            layout,
            y: vec![mid; y_len],
            u: vec![mid; uv_len],
            v: vec![mid; uv_len],
        }
    }

    /// Generates a deterministic synthetic test video frame (`frame_idx`) with spatial
    /// edges, diagonal gradients, frequency detail, and moving blocks for testing intra and inter coding.
    pub fn generate_synthetic(
        width: usize,
        height: usize,
        bit_depth: u8,
        layout: PixelLayout,
        frame_idx: usize,
    ) -> Self {
        let mut frame = Self::new(width, height, bit_depth, layout);
        let max_val = (1i32 << bit_depth) - 1;
        let shift = (bit_depth - 8) as i32;
        let dx = (frame_idx as i32) * 2;
        let dy = frame_idx as i32;

        for r in 0..height {
            for c in 0..width {
                let x = (c as i32 + dx).rem_euclid(width.max(1) as i32);
                let y = (r as i32 + dy).rem_euclid(height.max(1) as i32);

                let grad = ((x * 160 / (width.max(1) as i32)) + (y * 64 / (height.max(1) as i32)))
                    .clamp(16, 235);
                let box_hit = (x >= (width as i32) / 4 && x < (width as i32) * 3 / 4)
                    && (y >= (height as i32) / 4 && y < (height as i32) * 3 / 4);
                let detail = if ((x >> 3) ^ (y >> 3)) & 1 != 0 {
                    12
                } else {
                    -12
                };
                let base_8 = if box_hit {
                    (180 + detail).clamp(16, 235)
                } else {
                    (grad + (detail / 2)).clamp(16, 235)
                };
                let sample = (base_8 << shift).clamp(0, max_val);
                frame.y[r * width + c] = P::from_i32(sample);
            }
        }

        if layout != PixelLayout::I400 {
            let (uv_w, uv_h) = match layout {
                PixelLayout::I400 => (0, 0),
                PixelLayout::I420 => ((width + 1) >> 1, (height + 1) >> 1),
                PixelLayout::I422 => ((width + 1) >> 1, height),
                PixelLayout::I444 => (width, height),
            };
            for r in 0..uv_h {
                for c in 0..uv_w {
                    let u_8 =
                        (96 + ((c as i32 + dx / 2) * 64 / (uv_w.max(1) as i32))).clamp(16, 240);
                    let v_8 =
                        (160 - ((r as i32 + dy / 2) * 64 / (uv_h.max(1) as i32))).clamp(16, 240);
                    frame.u[r * uv_w + c] = P::from_i32((u_8 << shift).clamp(0, max_val));
                    frame.v[r * uv_w + c] = P::from_i32((v_8 << shift).clamp(0, max_val));
                }
            }
        }

        frame
    }
}

/// Reader for `YUV4MPEG2` (`.y4m`) uncompressed video streams.
pub struct Y4mReader<R: BufRead> {
    reader: R,
    pub header: Y4mHeader,
}

impl<R: BufRead> Y4mReader<R> {
    /// Opens a Y4M stream and parses its initial `YUV4MPEG2` header line.
    pub fn new(mut reader: R) -> Result<Self> {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        if bytes_read == 0 || !line.starts_with("YUV4MPEG2") {
            return Err(Av2Error::InvalidHeader(
                "Stream does not start with YUV4MPEG2 magic".into(),
            ));
        }

        let mut header = Y4mHeader::default();
        for token in line.split_whitespace().skip(1) {
            if token.is_empty() {
                continue;
            }
            let (tag, val) = token.split_at(1);
            match tag {
                "W" => {
                    header.width = val.parse::<usize>().map_err(|_| {
                        Av2Error::InvalidHeader(format!("Invalid Y4M width: {}", val))
                    })?;
                }
                "H" => {
                    header.height = val.parse::<usize>().map_err(|_| {
                        Av2Error::InvalidHeader(format!("Invalid Y4M height: {}", val))
                    })?;
                }
                "F" => {
                    if let Some((n, d)) = val.split_once(':') {
                        header.framerate_num = n.parse().unwrap_or(30);
                        header.framerate_den = d.parse().unwrap_or(1).max(1);
                    }
                }
                "A" => {
                    if let Some((n, d)) = val.split_once(':') {
                        header.aspect_num = n.parse().unwrap_or(1);
                        header.aspect_den = d.parse().unwrap_or(1).max(1);
                    }
                }
                "C" => {
                    let (layout, bit_depth) = parse_y4m_colorspace(val)?;
                    header.layout = layout;
                    header.bit_depth = bit_depth;
                }
                _ => {}
            }
        }

        if header.width == 0 || header.height == 0 {
            return Err(Av2Error::InvalidHeader(
                "Y4M header has zero width or height".into(),
            ));
        }

        Ok(Self { reader, header })
    }

    /// Reads the next uncompressed frame from the Y4M stream, or returns `Ok(None)` at EOF.
    pub fn read_frame<P: Pixel>(&mut self) -> Result<Option<RawFrame<P>>> {
        let mut frame_line = String::new();
        loop {
            frame_line.clear();
            let n = self
                .reader
                .read_line(&mut frame_line)
                .map_err(|e| Av2Error::IoError(e.to_string()))?;
            if n == 0 {
                return Ok(None);
            }
            if !frame_line.trim().is_empty() {
                break;
            }
        }

        if !frame_line.starts_with("FRAME") {
            return Err(Av2Error::InvalidHeader(format!(
                "Expected Y4M FRAME marker, got: {}",
                frame_line.trim()
            )));
        }

        let mut raw = RawFrame::<P>::new(
            self.header.width,
            self.header.height,
            self.header.bit_depth,
            self.header.layout,
        );

        let bps = self.header.bytes_per_sample();
        read_plane_samples(&mut self.reader, &mut raw.y, bps, self.header.bit_depth)?;
        if self.header.layout != PixelLayout::I400 {
            read_plane_samples(&mut self.reader, &mut raw.u, bps, self.header.bit_depth)?;
            read_plane_samples(&mut self.reader, &mut raw.v, bps, self.header.bit_depth)?;
        }

        Ok(Some(raw))
    }
}

fn parse_y4m_colorspace(val: &str) -> Result<(PixelLayout, u8)> {
    if val.starts_with("mono") {
        let bd = if val.ends_with("12") {
            12
        } else if val.ends_with("10") {
            10
        } else {
            8
        };
        return Ok((PixelLayout::I400, bd));
    }
    let layout = if val.starts_with("444") {
        PixelLayout::I444
    } else if val.starts_with("422") {
        PixelLayout::I422
    } else if val.starts_with("420") {
        PixelLayout::I420
    } else {
        return Err(Av2Error::InvalidHeader(format!(
            "Unsupported Y4M colorspace: {}",
            val
        )));
    };
    let bit_depth = if val.contains("p12") {
        12
    } else if val.contains("p10") {
        10
    } else {
        8
    };
    Ok((layout, bit_depth))
}

fn read_plane_samples<R: Read, P: Pixel>(
    reader: &mut R,
    dst: &mut [P],
    bps: usize,
    bit_depth: u8,
) -> Result<()> {
    P::read_le_samples(reader, dst, bps, bit_depth).map_err(|e| Av2Error::IoError(e.to_string()))
}

/// Writer for `YUV4MPEG2` (`.y4m`) uncompressed video streams.
pub struct Y4mWriter<W: Write> {
    writer: W,
    header: Y4mHeader,
}

impl<W: Write> Y4mWriter<W> {
    /// Creates a new Y4M writer and immediately writes the `YUV4MPEG2` stream header.
    pub fn new(mut writer: W, header: Y4mHeader) -> Result<Self> {
        let cs = match (header.layout, header.bit_depth) {
            (PixelLayout::I400, 12) => "mono12",
            (PixelLayout::I400, 10) => "mono10",
            (PixelLayout::I400, _) => "mono",
            (PixelLayout::I420, 8) => "420jpeg",
            (PixelLayout::I420, 10) => "420p10",
            (PixelLayout::I420, 12) => "420p12",
            (PixelLayout::I422, 8) => "422",
            (PixelLayout::I422, 10) => "422p10",
            (PixelLayout::I422, 12) => "422p12",
            (PixelLayout::I444, 8) => "444",
            (PixelLayout::I444, 10) => "444p10",
            (PixelLayout::I444, 12) => "444p12",
            _ => "420jpeg",
        };
        let hdr_line = format!(
            "YUV4MPEG2 W{} H{} F{}:{} Ip A{}:{} C{}\n",
            header.width,
            header.height,
            header.framerate_num,
            header.framerate_den,
            header.aspect_num,
            header.aspect_den,
            cs
        );
        writer
            .write_all(hdr_line.as_bytes())
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        Ok(Self { writer, header })
    }

    /// Writes a single uncompressed frame to the Y4M stream.
    pub fn write_frame<P: Pixel>(&mut self, frame: &RawFrame<P>) -> Result<()> {
        self.writer
            .write_all(b"FRAME\n")
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        let bps = self.header.bytes_per_sample();
        write_plane_samples(&mut self.writer, &frame.y, bps)?;
        if self.header.layout != PixelLayout::I400 {
            write_plane_samples(&mut self.writer, &frame.u, bps)?;
            write_plane_samples(&mut self.writer, &frame.v, bps)?;
        }
        Ok(())
    }

    /// Flushes the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| Av2Error::IoError(e.to_string()))
    }
}

fn write_plane_samples<W: Write, P: Pixel>(writer: &mut W, src: &[P], bps: usize) -> Result<()> {
    P::write_le_samples(writer, src, bps).map_err(|e| Av2Error::IoError(e.to_string()))
}

/// Standard 32-byte IVF container header (`DKIF` signature, `AV02` FourCC).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfHeader {
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub timebase_num: u32,
    pub timebase_den: u32,
    pub num_frames: u32,
}

/// IVF container writer for AV2 packetized bitstreams (`AV02` FourCC).
pub struct IvfWriter<W: Write> {
    writer: W,
    pub header: IvfHeader,
    pub frames_written: u32,
}

impl<W: Write> IvfWriter<W> {
    /// Creates an IVF writer and writes the 32-byte `DKIF` file header with FourCC `AV02`.
    pub fn new(
        mut writer: W,
        width: u16,
        height: u16,
        framerate_num: u32,
        framerate_den: u32,
        num_frames_hint: u32,
    ) -> Result<Self> {
        let header = IvfHeader {
            fourcc: *b"AV02",
            width,
            height,
            timebase_den: framerate_num.max(1),
            timebase_num: framerate_den.max(1),
            num_frames: num_frames_hint,
        };

        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(b"DKIF");
        hdr[4..6].copy_from_slice(&0u16.to_le_bytes());
        hdr[6..8].copy_from_slice(&32u16.to_le_bytes());
        hdr[8..12].copy_from_slice(&header.fourcc);
        hdr[12..14].copy_from_slice(&header.width.to_le_bytes());
        hdr[14..16].copy_from_slice(&header.height.to_le_bytes());
        hdr[16..20].copy_from_slice(&header.timebase_den.to_le_bytes());
        hdr[20..24].copy_from_slice(&header.timebase_num.to_le_bytes());
        hdr[24..28].copy_from_slice(&header.num_frames.to_le_bytes());
        hdr[28..32].copy_from_slice(&0u32.to_le_bytes());

        writer
            .write_all(&hdr)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;

        Ok(Self {
            writer,
            header,
            frames_written: 0,
        })
    }

    /// Writes a single temporal unit packet with its presentation timestamp (`pts`) and 12-byte IVF frame header.
    pub fn write_packet(&mut self, pts: u64, payload: &[u8]) -> Result<()> {
        let mut frame_hdr = [0u8; 12];
        let size = payload.len() as u32;
        frame_hdr[0..4].copy_from_slice(&size.to_le_bytes());
        frame_hdr[4..12].copy_from_slice(&pts.to_le_bytes());
        self.writer
            .write_all(&frame_hdr)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        self.writer
            .write_all(payload)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        self.frames_written += 1;
        Ok(())
    }

    /// Flushes the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| Av2Error::IoError(e.to_string()))
    }
}

/// IVF packet read from an `IvfReader`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfPacket {
    pub pts: u64,
    pub data: Vec<u8>,
}

/// IVF container reader for verifying packetized `AV02` bitstreams.
pub struct IvfReader<R: Read> {
    reader: R,
    pub header: IvfHeader,
}

impl<R: Read> IvfReader<R> {
    /// Opens an IVF stream and validates its 32-byte `DKIF` header.
    pub fn new(mut reader: R) -> Result<Self> {
        let mut hdr = [0u8; 32];
        reader
            .read_exact(&mut hdr)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        if &hdr[0..4] != b"DKIF" {
            return Err(Av2Error::InvalidHeader(
                "IVF stream does not start with DKIF magic".into(),
            ));
        }
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&hdr[8..12]);
        let width = u16::from_le_bytes([hdr[12], hdr[13]]);
        let height = u16::from_le_bytes([hdr[14], hdr[15]]);
        let timebase_den = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
        let timebase_num = u32::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]);
        let num_frames = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);

        Ok(Self {
            reader,
            header: IvfHeader {
                fourcc,
                width,
                height,
                timebase_num,
                timebase_den,
                num_frames,
            },
        })
    }

    /// Reads the next IVF packet, or returns `Ok(None)` at EOF.
    pub fn read_packet(&mut self) -> Result<Option<IvfPacket>> {
        let mut frame_hdr = [0u8; 12];
        match self.reader.read_exact(&mut frame_hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(Av2Error::IoError(e.to_string())),
        }
        let size =
            u32::from_le_bytes([frame_hdr[0], frame_hdr[1], frame_hdr[2], frame_hdr[3]]) as usize;
        let pts = u64::from_le_bytes([
            frame_hdr[4],
            frame_hdr[5],
            frame_hdr[6],
            frame_hdr[7],
            frame_hdr[8],
            frame_hdr[9],
            frame_hdr[10],
            frame_hdr[11],
        ]);
        let mut data = vec![0u8; size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| Av2Error::IoError(e.to_string()))?;
        Ok(Some(IvfPacket { pts, data }))
    }
}
