// SPDX-License-Identifier: Apache-2.0
//! ISO base media file format box primitives (ISO/IEC 14496-12, Section 4).
//!
//! The parser is a strict, zero-copy cursor over an in-memory file.  Every
//! size, offset, and count is bounds checked; unknown boxes are skipped by
//! the caller, never guessed.

use crate::FormatError;

/// Maximum number of boxes accepted in one document.
pub(crate) const MAX_BOXES: usize = 1_000_000;

/// One box header resolved against the backing buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoxHeader {
    /// Four-character type.
    pub kind: [u8; 4],
    /// Offset of the first byte of the box.
    pub start: usize,
    /// Offset of the first payload byte.
    pub payload_start: usize,
    /// Offset one past the last payload byte.
    pub payload_end: usize,
}

impl BoxHeader {
    /// Payload bytes, excluding the header (and a `uuid` extended type).
    pub(crate) fn payload<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        &data[self.payload_start..self.payload_end]
    }
}

/// Iterator over the boxes in a byte range.
#[derive(Debug)]
pub(crate) struct Boxes<'a> {
    data: &'a [u8],
    position: usize,
    end: usize,
    count: usize,
}

impl<'a> Boxes<'a> {
    /// Creates an iterator over `data[start..end]`.
    pub(crate) fn new(data: &'a [u8], start: usize, end: usize) -> Result<Self, FormatError> {
        if start > end || end > data.len() {
            return Err(FormatError::Isobmff("box range is outside the file"));
        }
        Ok(Self {
            data,
            position: start,
            end,
            count: 0,
        })
    }
}

impl Iterator for Boxes<'_> {
    type Item = Result<BoxHeader, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.end {
            return None;
        }
        self.count += 1;
        if self.count > MAX_BOXES {
            return Some(Err(FormatError::IsobmffLimit("box count")));
        }
        match next_box(self.data, &mut self.position, self.end) {
            Ok(header) => Some(Ok(header)),
            Err(error) => {
                self.position = self.end;
                Some(Err(error))
            }
        }
    }
}

/// Reads one box header and advances `position` past the box.
pub(crate) fn next_box(
    data: &[u8],
    position: &mut usize,
    end: usize,
) -> Result<BoxHeader, FormatError> {
    let start = *position;
    if start.checked_add(8).is_none_or(|value| value > end) {
        return Err(FormatError::Isobmff("truncated box header"));
    }
    let size32 = read_u32(data, start)?;
    let mut kind = [0_u8; 4];
    kind.copy_from_slice(
        data.get(start + 4..start + 8)
            .ok_or(FormatError::Isobmff("truncated box type"))?,
    );
    let (size, header_len) = match size32 {
        0 => {
            // A zero size extends to the end of the enclosing container.
            (end - start, 8)
        }
        1 => {
            if start.checked_add(16).is_none_or(|value| value > end) {
                return Err(FormatError::Isobmff("truncated large box size"));
            }
            let large = read_u64(data, start + 8)?;
            let size = usize::try_from(large).map_err(|_| FormatError::IsobmffLimit("box size"))?;
            (size, 16)
        }
        _ => (size32 as usize, 8),
    };
    if size < header_len {
        return Err(FormatError::Isobmff("box size is smaller than its header"));
    }
    let payload_end = start
        .checked_add(size)
        .ok_or(FormatError::IsobmffLimit("box size"))?;
    if payload_end > end {
        return Err(FormatError::Isobmff("box exceeds its container"));
    }
    let mut payload_start = start + header_len;
    if kind == *b"uuid" {
        // The extended type belongs to the payload but is not interpreted.
        if payload_start
            .checked_add(16)
            .is_none_or(|value| value > payload_end)
        {
            return Err(FormatError::Isobmff("truncated uuid box"));
        }
        payload_start += 16;
    }
    *position = payload_end;
    Ok(BoxHeader {
        kind,
        start,
        payload_start,
        payload_end,
    })
}

/// Splits a full box payload into version, flags, and the remaining bytes.
pub(crate) fn full_box<'a>(
    data: &'a [u8],
    header: &BoxHeader,
) -> Result<(u8, u32, &'a [u8]), FormatError> {
    let payload = header.payload(data);
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated full box header"));
    }
    let version = payload[0];
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    Ok((version, flags, &payload[4..]))
}

/// Reads a big-endian `u16`.
pub(crate) fn read_u16(data: &[u8], position: usize) -> Result<u16, FormatError> {
    let bytes = data
        .get(position..position + 2)
        .ok_or(FormatError::Isobmff("truncated u16"))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// Reads a big-endian `u32`.
pub(crate) fn read_u32(data: &[u8], position: usize) -> Result<u32, FormatError> {
    let bytes = data
        .get(position..position + 4)
        .ok_or(FormatError::Isobmff("truncated u32"))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Reads a big-endian `i32`.
pub(crate) fn read_i32(data: &[u8], position: usize) -> Result<i32, FormatError> {
    Ok(read_u32(data, position)? as i32)
}

/// Reads a big-endian `u64`.
pub(crate) fn read_u64(data: &[u8], position: usize) -> Result<u64, FormatError> {
    let bytes = data
        .get(position..position + 8)
        .ok_or(FormatError::Isobmff("truncated u64"))?;
    Ok(u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Appends a box with a 32-bit size, or a 64-bit size when required.
pub(crate) fn push_box(out: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    let size = payload.len() as u64 + 8;
    if size <= u64::from(u32::MAX) {
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(payload);
    } else {
        out.extend_from_slice(&1_u32.to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(&(size + 8).to_be_bytes());
        out.extend_from_slice(payload);
    }
}

/// Builds one box.
pub(crate) fn box_bytes(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    push_box(&mut out, kind, payload);
    out
}

/// Appends a full box with version and flags.
pub(crate) fn push_full_box(
    out: &mut Vec<u8>,
    kind: [u8; 4],
    version: u8,
    flags: u32,
    payload: &[u8],
) {
    let mut body = Vec::with_capacity(payload.len() + 4);
    body.push(version);
    body.extend_from_slice(&flags.to_be_bytes()[1..]);
    body.extend_from_slice(payload);
    push_box(out, kind, &body);
}

/// Builds one full box.
pub(crate) fn full_box_bytes(kind: [u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 20);
    push_full_box(&mut out, kind, version, flags, payload);
    out
}

/// Appends a big-endian `u16`.
pub(crate) fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends a big-endian `u32`.
pub(crate) fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Appends a big-endian `u64`.
pub(crate) fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Encodes a standard 15-bit packed ISO-639-2/T language code.
pub(crate) fn encode_language(text: &str) -> u16 {
    let mut letters = text.bytes().filter(|byte| byte.is_ascii_lowercase());
    let mut value = 0_u16;
    for _ in 0..3 {
        let letter = letters.next().unwrap_or(b'u').saturating_sub(0x60);
        value = (value << 5) | u16::from(letter & 0x1f);
    }
    value
}

/// Decodes a standard 15-bit packed ISO-639-2/T language code.
pub(crate) fn decode_language(value: u16) -> String {
    if value == 0 {
        return "und".into();
    }
    let mut text = String::with_capacity(3);
    for shift in [10, 5, 0] {
        let letter = ((value >> shift) & 0x1f) as u8;
        text.push(char::from(letter + 0x60));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_cover_32_bit_and_large_sizes() {
        let mut data = Vec::new();
        data.extend_from_slice(&16_u32.to_be_bytes());
        data.extend_from_slice(b"free");
        data.extend_from_slice(&[0; 8]);

        let mut position = 0;
        let header = next_box(&data, &mut position, data.len()).expect("header");
        assert_eq!(header.kind, *b"free");
        assert_eq!(header.payload(&data), &[0; 8]);
        assert_eq!(position, data.len());

        let mut large = Vec::new();
        large.extend_from_slice(&1_u32.to_be_bytes());
        large.extend_from_slice(b"mdat");
        large.extend_from_slice(&24_u64.to_be_bytes());
        large.extend_from_slice(&[0; 8]);
        let mut position = 0;
        let header = next_box(&large, &mut position, large.len()).expect("large");
        assert_eq!(header.kind, *b"mdat");
        assert_eq!(header.payload_start, 16);
    }

    #[test]
    fn truncated_and_impossible_boxes_are_rejected() {
        let mut position = 0;
        assert!(next_box(&[0, 0, 0], &mut position, 3).is_err());
        let mut data = Vec::new();
        data.extend_from_slice(&7_u32.to_be_bytes());
        data.extend_from_slice(b"free");
        let mut position = 0;
        assert!(next_box(&data, &mut position, data.len()).is_err());
        let mut data = Vec::new();
        data.extend_from_slice(&32_u32.to_be_bytes());
        data.extend_from_slice(b"free");
        let mut position = 0;
        assert!(next_box(&data, &mut position, data.len()).is_err());
    }

    #[test]
    fn language_codes_round_trip() {
        assert_eq!(decode_language(encode_language("eng")), "eng");
        assert_eq!(decode_language(encode_language("und")), "und");
    }

    #[test]
    fn full_boxes_have_version_and_flags() {
        let boxed = full_box_bytes(*b"test", 1, 0x123456, &[9, 8]);
        let mut position = 0;
        let header = next_box(&boxed, &mut position, boxed.len()).expect("header");
        let (version, flags, payload) = full_box(&boxed, &header).expect("full");
        assert_eq!(version, 1);
        assert_eq!(flags, 0x123456);
        assert_eq!(payload, &[9, 8]);
    }
}
