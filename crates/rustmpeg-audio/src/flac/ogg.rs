// SPDX-License-Identifier: Apache-2.0
//! Ogg FLAC muxing.
//!
//! The mapping combines the native `fLaC` marker and STREAMINFO block into the
//! first Ogg packet and stores every audio frame in its own packet, with the
//! page granule position set to the end sample of the last completed frame.
//! See the Ogg FLAC mapping (Xiph.Org) and RFC 3533.

use std::io::Write;

use super::encoder::{FlacEncoderConfig, PreparedFlac, prepare_stream, streaminfo_payload};
use super::{FlacError, StreamInfo};

const OGG_SERIAL: u32 = 0x5253_4d50; // "RSMP"
const CRC_POLYNOMIAL: u32 = 0x04c1_1db7;

/// Computes the Ogg page checksum (CRC-32, unreflected, initial value zero).
fn ogg_crc(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ CRC_POLYNOMIAL
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Writes Ogg pages for FLAC packets.
struct PageWriter<W: Write> {
    writer: W,
    sequence: u32,
    first_page: bool,
}

impl<W: Write> PageWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            sequence: 0,
            first_page: true,
        }
    }

    /// Writes one packet, split across as many pages as its lacing needs.
    fn write_packet(&mut self, packet: &[u8], granule: u64, eos: bool) -> Result<(), FlacError> {
        // Build the lacing values: full 255-byte segments plus a final short
        // segment (a zero-length terminator when the length is a multiple of
        // 255).
        let mut lacing = Vec::with_capacity(packet.len() / 255 + 1);
        let mut remaining = packet.len();
        while remaining >= 255 {
            lacing.push(255u8);
            remaining -= 255;
        }
        lacing.push(remaining as u8);

        let mut segment_index = 0usize;
        let mut body_offset = 0usize;
        let mut continuation = false;
        while segment_index < lacing.len() {
            let end = (segment_index + 255).min(lacing.len());
            let segments = &lacing[segment_index..end];
            let body_len: usize = segments.iter().map(|&value| usize::from(value)).sum();
            let body = &packet[body_offset..body_offset + body_len];
            let packet_complete = end == lacing.len();

            let mut header_type = 0u8;
            if self.first_page {
                header_type |= 0x02; // beginning of stream
            }
            if continuation {
                header_type |= 0x01;
            }
            if packet_complete && eos {
                header_type |= 0x04; // end of stream
            }
            let page_granule = if packet_complete { granule } else { u64::MAX };

            let mut header = Vec::with_capacity(27 + segments.len());
            header.extend_from_slice(b"OggS");
            header.push(0); // stream structure version
            header.push(header_type);
            header.extend_from_slice(&page_granule.to_le_bytes());
            header.extend_from_slice(&OGG_SERIAL.to_le_bytes());
            header.extend_from_slice(&self.sequence.to_le_bytes());
            header.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
            header.push(segments.len() as u8);
            header.extend_from_slice(segments);
            let mut page = header;
            page.extend_from_slice(body);
            let crc = ogg_crc(&page);
            let crc_offset = 22;
            page[crc_offset..crc_offset + 4].copy_from_slice(&crc.to_le_bytes());
            self.writer.write_all(&page)?;

            self.sequence = self.sequence.wrapping_add(1);
            self.first_page = false;
            segment_index = end;
            body_offset += body_len;
            continuation = true;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), FlacError> {
        self.writer.flush()?;
        Ok(())
    }
}

fn first_packet(info: &StreamInfo) -> Vec<u8> {
    let mut packet = Vec::with_capacity(9 + 4 + 4 + 34);
    packet.push(0x7f); // first header packet type
    packet.extend_from_slice(b"FLAC");
    packet.push(1); // mapping major version
    packet.push(0); // mapping minor version
    packet.extend_from_slice(&1u16.to_be_bytes()); // one comment header packet follows
    packet.extend_from_slice(b"fLaC");
    packet.extend_from_slice(&[0, 0, 0, 34]); // STREAMINFO, not the last block
    packet.extend_from_slice(&streaminfo_payload(info));
    packet
}

/// Builds a minimal VORBIS_COMMENT metadata packet (vendor string, no comments).
fn comment_packet() -> Vec<u8> {
    let vendor = b"rustmpeg-next";
    let mut payload = Vec::with_capacity(4 + vendor.len() + 4);
    payload.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    payload.extend_from_slice(vendor);
    payload.extend_from_slice(&0u32.to_le_bytes());
    let length = payload.len();
    let mut packet = Vec::with_capacity(4 + length);
    packet.push(0x84); // last metadata block, type VORBIS_COMMENT
    packet.extend_from_slice(&[(length >> 16) as u8, (length >> 8) as u8, length as u8]);
    packet.extend_from_slice(&payload);
    packet
}

/// Encodes planar PCM into an Ogg FLAC stream.
pub fn encode_ogg_flac<W: Write>(
    writer: W,
    config: &FlacEncoderConfig,
    planes: &[Vec<i32>],
) -> Result<StreamInfo, FlacError> {
    let prepared: PreparedFlac = prepare_stream(config, planes)?;
    let mut pages = PageWriter::new(writer);
    pages.write_packet(&first_packet(&prepared.info), 0, false)?;
    pages.write_packet(&comment_packet(), 0, false)?;
    let mut total = 0u64;
    for (index, frame) in prepared.frames.iter().enumerate() {
        total += u64::from(prepared.frame_meta[index].2);
        let eos = index + 1 == prepared.frames.len();
        pages.write_packet(frame, total, eos)?;
    }
    if prepared.frames.is_empty() {
        // Emit an empty EOS page so players observe a clean end of stream.
        pages.write_packet(&[], total, true)?;
    }
    pages.flush()?;
    Ok(prepared.info)
}

/// Encodes planar PCM into an in-memory Ogg FLAC stream.
pub fn encode_ogg_flac_to_vec(
    config: &FlacEncoderConfig,
    planes: &[Vec<i32>],
) -> Result<(Vec<u8>, StreamInfo), FlacError> {
    let mut out = Vec::new();
    let info = encode_ogg_flac(&mut out, config, planes)?;
    Ok((out, info))
}

/// Streaming decoder for Ogg FLAC files.
///
/// The demuxer validates Ogg page CRCs, reassembles packets, extracts the
/// mapping header and metadata, and feeds the native FLAC frames to
/// [`FlacDecoder`].  MD5 verification is inherited from the native decoder.
///
/// The first beginning-of-stream page must belong to the FLAC logical stream;
/// a leading Skeleton stream, chained links, and multiplexed non-FLAC streams
/// before the FLAC stream are not supported.  Pages belonging to other serial
/// numbers after the FLAC stream is selected are skipped.
pub struct OggFlacDecoder<R: std::io::Read> {
    decoder: super::FlacDecoder<OggStream<R>>,
    info: StreamInfo,
}

struct PageReader<R: std::io::Read> {
    reader: R,
    serial: Option<u32>,
    segments: Vec<u8>,
    body: Vec<u8>,
    body_pos: usize,
    segment_pos: usize,
    packet: Vec<u8>,
    eof: bool,
}

impl<R: std::io::Read> PageReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            serial: None,
            segments: Vec::new(),
            body: Vec::new(),
            body_pos: 0,
            segment_pos: 0,
            packet: Vec::new(),
            eof: false,
        }
    }

    fn read_exact_or_eof(&mut self, buffer: &mut [u8]) -> Result<bool, FlacError> {
        let mut filled = 0usize;
        while filled < buffer.len() {
            match self.reader.read(&mut buffer[filled..]) {
                Ok(0) => {
                    if filled == 0 {
                        return Ok(false);
                    }
                    return Err(FlacError::InvalidOgg("truncated page"));
                }
                Ok(count) => filled += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(FlacError::Io(error)),
            }
        }
        Ok(true)
    }

    /// Reads the next page, skipping logical streams other than the FLAC one.
    fn read_page(&mut self) -> Result<bool, FlacError> {
        if self.eof {
            return Ok(false);
        }
        loop {
            let mut header = [0u8; 27];
            if !self.read_exact_or_eof(&mut header)? {
                self.eof = true;
                return Ok(false);
            }
            if &header[..4] != b"OggS" {
                return Err(FlacError::InvalidOgg("missing OggS capture pattern"));
            }
            if header[4] != 0 {
                return Err(FlacError::InvalidOgg("unsupported Ogg stream version"));
            }
            let header_type = header[5];
            let serial = u32::from_le_bytes(header[14..18].try_into().unwrap());
            let segment_count = usize::from(header[26]);
            let mut segments = vec![0u8; segment_count];
            if !self.read_exact_or_eof(&mut segments)? {
                return Err(FlacError::InvalidOgg("truncated segment table"));
            }
            let body_len: usize = segments.iter().map(|&value| usize::from(value)).sum();
            let mut body = vec![0u8; body_len];
            if !self.read_exact_or_eof(&mut body)? {
                return Err(FlacError::InvalidOgg("truncated page body"));
            }
            let stored_crc = u32::from_le_bytes(header[22..26].try_into().unwrap());
            let mut check = Vec::with_capacity(27 + segment_count + body_len);
            check.extend_from_slice(&header);
            check[22..26].fill(0);
            check.extend_from_slice(&segments);
            check.extend_from_slice(&body);
            if ogg_crc(&check) != stored_crc {
                return Err(FlacError::InvalidOgg("page CRC mismatch"));
            }

            match self.serial {
                None => {
                    if header_type & 0x02 == 0 {
                        return Err(FlacError::InvalidOgg(
                            "first page is not a beginning-of-stream page",
                        ));
                    }
                    self.serial = Some(serial);
                }
                Some(current) => {
                    if serial != current {
                        // Another multiplexed logical stream; skip its pages.
                        continue;
                    }
                }
            }
            self.segments = segments;
            self.body = body;
            self.body_pos = 0;
            self.segment_pos = 0;
            return Ok(true);
        }
    }

    fn next_packet(&mut self) -> Result<Option<Vec<u8>>, FlacError> {
        loop {
            while self.segment_pos < self.segments.len() {
                let length = usize::from(self.segments[self.segment_pos]);
                self.segment_pos += 1;
                let end = self.body_pos + length;
                if end > self.body.len() {
                    return Err(FlacError::InvalidOgg("segment exceeds page body"));
                }
                self.packet
                    .extend_from_slice(&self.body[self.body_pos..end]);
                self.body_pos = end;
                if length < 255 {
                    return Ok(Some(std::mem::take(&mut self.packet)));
                }
            }
            if !self.read_page()? {
                if self.packet.is_empty() {
                    return Ok(None);
                }
                return Err(FlacError::InvalidOgg(
                    "unterminated packet at end of stream",
                ));
            }
        }
    }
}

/// `Read` adapter that exposes the reconstructed native FLAC header followed by
/// the audio packets of the Ogg stream.
struct OggStream<R: std::io::Read> {
    prefix: std::io::Cursor<Vec<u8>>,
    pages: PageReader<R>,
    pending: Vec<u8>,
    pending_pos: usize,
    eof: bool,
}

impl<R: std::io::Read> std::io::Read for OggStream<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.prefix.position() < self.prefix.get_ref().len() as u64 {
            let read = self.prefix.read(buffer)?;
            if read > 0 {
                return Ok(read);
            }
        }
        while self.pending_pos >= self.pending.len() {
            if self.eof {
                return Ok(0);
            }
            match self.pages.next_packet() {
                Ok(Some(packet)) => {
                    self.pending = packet;
                    self.pending_pos = 0;
                    if self.pending.is_empty() {
                        continue;
                    }
                }
                Ok(None) => {
                    self.eof = true;
                    return Ok(0);
                }
                Err(error) => {
                    return Err(std::io::Error::other(error.to_string()));
                }
            }
        }
        let count = buffer.len().min(self.pending.len() - self.pending_pos);
        buffer[..count].copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + count]);
        self.pending_pos += count;
        Ok(count)
    }
}

/// Parses a native FLAC metadata block header and payload.
fn split_metadata_block(data: &[u8]) -> Result<(bool, u8, Vec<u8>), FlacError> {
    if data.len() < 4 {
        return Err(FlacError::InvalidOgg("truncated metadata block"));
    }
    let last = data[0] & 0x80 != 0;
    let kind = data[0] & 0x7f;
    let length = (u32::from(data[1]) << 16) | (u32::from(data[2]) << 8) | u32::from(data[3]);
    let length = length as usize;
    if data.len() < 4 + length {
        return Err(FlacError::InvalidOgg("truncated metadata payload"));
    }
    Ok((last, kind, data[4..4 + length].to_vec()))
}

impl<R: std::io::Read> OggFlacDecoder<R> {
    /// Creates an Ogg FLAC decoder and parses all header packets.
    pub fn new(reader: R) -> Result<Self, FlacError> {
        let mut pages = PageReader::new(reader);
        let first = pages
            .next_packet()?
            .ok_or(FlacError::InvalidOgg("no mapping header packet"))?;
        if first.len() < 13 || first[0] != 0x7f || &first[1..5] != b"FLAC" {
            return Err(FlacError::InvalidOgg("missing Ogg FLAC mapping header"));
        }
        if first[5] != 1 {
            return Err(FlacError::InvalidOgg(
                "unsupported Ogg FLAC mapping version",
            ));
        }
        // The header count is advisory: libFLAC's decoder walks the metadata
        // chain until its last-block flag instead of trusting the count, and
        // files in the wild disagree about whether the first packet is counted.
        if &first[9..13] != b"fLaC" {
            return Err(FlacError::InvalidOgg("missing native FLAC marker"));
        }
        let (streaminfo_last, kind, streaminfo) = split_metadata_block(&first[13..])?;
        if kind != 0 || streaminfo.len() != 34 {
            return Err(FlacError::InvalidOgg(
                "first metadata block is not STREAMINFO",
            ));
        }
        let mut streaminfo_bytes = [0u8; 34];
        streaminfo_bytes.copy_from_slice(&streaminfo);
        let info = StreamInfo::parse(&streaminfo_bytes)?;

        let mut blocks: Vec<(u8, Vec<u8>)> = vec![(0, streaminfo)];
        let mut last = streaminfo_last;
        while !last {
            let packet = pages
                .next_packet()?
                .ok_or(FlacError::InvalidOgg("missing metadata header packet"))?;
            if packet.len() >= 2 && packet[0] == 0xff && packet[1] & 0xfe == 0xf8 {
                return Err(FlacError::InvalidOgg(
                    "metadata chain has no last-block flag",
                ));
            }
            let (block_last, kind, payload) = split_metadata_block(&packet)?;
            blocks.push((kind, payload));
            last = block_last;
        }

        let mut prefix = b"fLaC".to_vec();
        let block_count = blocks.len();
        for (index, (kind, payload)) in blocks.into_iter().enumerate() {
            let last = index + 1 == block_count;
            let length = payload.len() as u32;
            prefix.push((u8::from(last) << 7) | (kind & 0x7f));
            prefix.push((length >> 16) as u8);
            prefix.push((length >> 8) as u8);
            prefix.push(length as u8);
            prefix.extend_from_slice(&payload);
        }

        let stream = OggStream {
            prefix: std::io::Cursor::new(prefix),
            pages,
            pending: Vec::new(),
            pending_pos: 0,
            eof: false,
        };
        let decoder = super::FlacDecoder::new(stream)?;
        Ok(Self { decoder, info })
    }

    /// Returns the STREAMINFO metadata block.
    #[must_use]
    pub fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    /// Decodes the next audio frame.
    pub fn next_frame(&mut self) -> Result<Option<crate::PcmFrame>, FlacError> {
        self.decoder.next_frame()
    }

    /// Enables or disables MD5 verification.
    pub fn set_verify_md5(&mut self, enabled: bool) {
        self.decoder.set_verify_md5(enabled);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::ogg_crc;
    use crate::flac::{FlacDecoder, FlacEncoderConfig, encode_ogg_flac_to_vec};

    #[derive(Debug)]
    struct Page {
        header_type: u8,
        granule: u64,
        serial: u32,
        sequence: u32,
        crc_ok: bool,
        segments: Vec<u8>,
        body: Vec<u8>,
    }

    fn parse_pages(mut data: &[u8]) -> Vec<Page> {
        let mut pages = Vec::new();
        while !data.is_empty() {
            assert!(data.len() >= 27);
            assert_eq!(&data[..4], b"OggS");
            assert_eq!(data[4], 0);
            let header_type = data[5];
            let granule = u64::from_le_bytes(data[6..14].try_into().unwrap());
            let serial = u32::from_le_bytes(data[14..18].try_into().unwrap());
            let sequence = u32::from_le_bytes(data[18..22].try_into().unwrap());
            let stored_crc = u32::from_le_bytes(data[22..26].try_into().unwrap());
            let count = usize::from(data[26]);
            let segments = data[27..27 + count].to_vec();
            let body_len: usize = segments.iter().map(|&value| usize::from(value)).sum();
            let body_start = 27 + count;
            let body = data[body_start..body_start + body_len].to_vec();
            let mut check = data[..body_start + body_len].to_vec();
            check[22..26].fill(0);
            pages.push(Page {
                header_type,
                granule,
                serial,
                sequence,
                crc_ok: ogg_crc(&check) == stored_crc,
                segments,
                body,
            });
            data = &data[body_start + body_len..];
        }
        pages
    }

    fn packets(pages: &[Page]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut current = Vec::new();
        for page in pages {
            let mut offset = 0usize;
            for &lacing in &page.segments {
                let count = usize::from(lacing);
                current.extend_from_slice(&page.body[offset..offset + count]);
                offset += count;
                if lacing < 255 {
                    out.push(std::mem::take(&mut current));
                }
            }
        }
        assert!(current.is_empty(), "unterminated Ogg packet");
        out
    }

    fn signal(channels: usize, bits: u8, len: usize) -> Vec<Vec<i32>> {
        let limit = 1i64 << (bits - 1);
        let mut state = 0x1234_5678_9abc_def0u64;
        (0..channels)
            .map(|channel| {
                (0..len)
                    .map(|index| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let noise = ((state >> 33) as i64) % (limit / 3).max(1);
                        ((index as i64 * 13 + channel as i64 * 7) % limit - limit / 2 + noise / 3)
                            .clamp(-limit, limit - 1) as i32
                    })
                    .collect()
            })
            .collect()
    }

    fn roundtrip(bits: u8, channels: u8, block_size: u32, samples: usize) {
        let config = FlacEncoderConfig {
            sample_rate: 48_000,
            channels,
            bits_per_sample: bits,
            block_size,
            compression_level: 5,
        };
        let planes = signal(usize::from(channels), bits, samples);
        let (bytes, info) = encode_ogg_flac_to_vec(&config, &planes).unwrap();
        let pages = parse_pages(&bytes);
        assert!(!pages.is_empty());
        for (index, page) in pages.iter().enumerate() {
            assert!(page.crc_ok, "page {index} CRC mismatch");
            assert_eq!(page.serial, 0x5253_4d50);
            assert_eq!(page.sequence, index as u32);
            if index == 0 {
                assert_eq!(page.header_type & 0x02, 0x02);
                assert_eq!(page.granule, 0);
            } else {
                assert_eq!(page.header_type & 0x02, 0);
            }
        }
        assert_eq!(pages.last().unwrap().header_type & 0x04, 0x04);
        assert_eq!(pages.last().unwrap().granule, samples as u64);

        let packets = packets(&pages);
        assert_eq!(packets[0][0], 0x7f);
        assert_eq!(&packets[0][1..5], b"FLAC");
        assert_eq!(packets[0][5], 1);
        assert_eq!(&packets[0][9..13], b"fLaC");
        assert_eq!(packets[1][0] & 0x7f, 4); // VORBIS_COMMENT
        assert_eq!(packets[1][0] & 0x80, 0x80); // last metadata block
        assert_eq!(
            packets.len() - 2,
            info.total_samples.unwrap().div_ceil(u64::from(block_size)) as usize
        );

        // Rebuild the equivalent native FLAC stream and decode it with the
        // native decoder, proving the Ogg payload carries the same frames.
        let mut native = b"fLaC".to_vec();
        native.extend_from_slice(&packets[0][13..]); // STREAMINFO metadata block
        native.extend_from_slice(&packets[1]); // comment metadata block
        for packet in &packets[2..] {
            native.extend_from_slice(packet);
        }
        let mut decoder = FlacDecoder::new(Cursor::new(native)).unwrap();
        let mut decoded = vec![Vec::new(); usize::from(channels)];
        let mut decoded_samples = 0u64;
        while let Some(frame) = decoder.next_frame().unwrap() {
            for (dst, src) in decoded.iter_mut().zip(&frame.planes) {
                dst.extend_from_slice(src);
            }
            decoded_samples += frame.block_size() as u64;
        }
        assert_eq!(decoded_samples, samples as u64);
        assert_eq!(decoded, planes);
    }

    #[test]
    fn crc_of_empty_page_header_is_stable() {
        // Ogg CRC-32 is unreflected with initial value zero; an empty header
        // therefore hashes to a deterministic value.
        assert_eq!(ogg_crc(b"OggS"), 0x5fb0_a94f);
        assert_eq!(ogg_crc(b""), 0);
    }

    #[test]
    fn ogg_roundtrip_stereo() {
        roundtrip(16, 2, 4096, 10_000);
        roundtrip(24, 2, 1024, 3_000);
    }

    #[test]
    fn ogg_roundtrip_multichannel_and_large_packets() {
        // A 32-bit 8-channel 16384-sample frame needs more than 255 lacing
        // values, exercising continuation pages.
        roundtrip(32, 8, 16_384, 16_384);
        roundtrip(8, 1, 192, 1_000);
    }

    #[test]
    fn demuxes_our_own_ogg_stream() {
        let config = FlacEncoderConfig {
            sample_rate: 48_000,
            channels: 2,
            bits_per_sample: 16,
            block_size: 1024,
            compression_level: 5,
        };
        let planes = signal(2, 16, 5_000);
        let (bytes, info) = encode_ogg_flac_to_vec(&config, &planes).unwrap();
        let mut decoder = super::OggFlacDecoder::new(Cursor::new(bytes)).unwrap();
        assert_eq!(decoder.stream_info().total_samples, Some(5_000));
        let mut decoded = vec![Vec::new(); 2];
        let mut samples = 0u64;
        while let Some(frame) = decoder.next_frame().unwrap() {
            for (dst, src) in decoded.iter_mut().zip(&frame.planes) {
                dst.extend_from_slice(src);
            }
            samples += frame.block_size() as u64;
        }
        assert_eq!(samples, 5_000);
        assert_eq!(decoded, planes);
        assert_eq!(info.md5.len(), 16);
    }

    #[test]
    fn demuxes_a_libflac_ogg_fixture() {
        let ogg = include_bytes!("../../tests/data/ogg_stereo16_44k_1024.oga");
        let native = include_bytes!("../../tests/data/stereo16_44k_1024_l8.flac");
        let mut ogg_decoder = super::OggFlacDecoder::new(Cursor::new(ogg)).unwrap();
        let mut native_decoder = FlacDecoder::new(Cursor::new(native)).unwrap();
        let mut ogg_planes = vec![Vec::new(); 2];
        let mut native_planes = vec![Vec::new(); 2];
        let mut ogg_samples = 0u64;
        while let Some(frame) = ogg_decoder.next_frame().unwrap() {
            for (dst, src) in ogg_planes.iter_mut().zip(&frame.planes) {
                dst.extend_from_slice(src);
            }
            ogg_samples += frame.block_size() as u64;
        }
        while let Some(frame) = native_decoder.next_frame().unwrap() {
            for (dst, src) in native_planes.iter_mut().zip(&frame.planes) {
                dst.extend_from_slice(src);
            }
        }
        assert!(ogg_samples > 0);
        assert_eq!(ogg_planes, native_planes);
    }
}
