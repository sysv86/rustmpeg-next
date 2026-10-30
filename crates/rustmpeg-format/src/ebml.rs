// SPDX-License-Identifier: Apache-2.0
//! Strict EBML primitive parser (IETF RFC 8794).
//!
//! The parser works on an in-memory buffer and returns zero-copy byte ranges.
//! Every read is bounds checked; unknown-size elements are reported as such and
//! resolved by the Matroska layer, which knows the allowed children.

use thiserror::Error;

/// Largest element ID accepted by the Matroska profile (RFC 8794 §4).
const MAX_ID_LENGTH: u8 = 4;
/// Largest element size descriptor accepted (RFC 8794 §4).
const MAX_SIZE_LENGTH: u8 = 8;

/// Errors from EBML or Matroska parsing.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The input ends before the requested element.
    #[error("unexpected end of data")]
    Eof,
    /// A variable-length integer is malformed.
    #[error("invalid EBML variable-length integer")]
    InvalidVint,
    /// An element header is malformed or violates the Matroska profile.
    #[error("invalid EBML element header")]
    InvalidElement,
    /// An integer element is wider than eight octets or empty where forbidden.
    #[error("invalid EBML integer")]
    InvalidInteger,
    /// A float element is neither four nor eight octets.
    #[error("invalid EBML float")]
    InvalidFloat,
    /// A string element is not valid UTF-8.
    #[error("invalid EBML string")]
    InvalidString,
    /// The document type is not Matroska or WebM.
    #[error("unsupported EBML document type")]
    UnsupportedDocType,
    /// The codec is outside the project's rolling support window.
    #[error("unsupported codec {0}")]
    UnsupportedCodec(String),
    /// The Matroska structure violates the specification.
    #[error("Matroska structure error: {0}")]
    Structure(&'static str),
    /// A block has an invalid header, lacing layout, or frame sizes.
    #[error("Matroska block error: {0}")]
    Block(&'static str),
    /// A block references a track that is not declared in `Tracks`.
    #[error("unknown Matroska track {0}")]
    UnknownTrack(u64),
    /// A configured implementation limit was exceeded.
    #[error("Matroska limit exceeded: {0}")]
    Limit(&'static str),
    /// Seeking was requested but the file has no usable `Cues`.
    #[error("Matroska file has no seek index")]
    NoSeekIndex,
    /// The ISO base media file structure violates ISO/IEC 14496-12.
    #[error("ISOBMFF structure error: {0}")]
    Isobmff(&'static str),
    /// An ISO base media implementation limit was exceeded.
    #[error("ISOBMFF limit exceeded: {0}")]
    IsobmffLimit(&'static str),
    /// A sample entry references an unsupported decoder configuration.
    #[error("unsupported ISOBMFF sample entry {0}")]
    UnsupportedSampleEntry(String),
}

/// Decodes one EBML variable-length integer.
///
/// `keep_marker` selects element ID semantics: the marker bit stays part of
/// the value. Sizes strip the marker and report the all-ones pattern as
/// "unknown" through `None`.
pub(crate) fn read_vint(
    data: &[u8],
    position: &mut usize,
    keep_marker: bool,
) -> Result<(u64, Option<u8>), FormatError> {
    let first = *data.get(*position).ok_or(FormatError::Eof)?;
    if first == 0 {
        return Err(FormatError::InvalidVint);
    }
    let length = first.leading_zeros() as u8 + 1;
    if length > MAX_SIZE_LENGTH {
        return Err(FormatError::InvalidVint);
    }
    let end = position
        .checked_add(usize::from(length))
        .ok_or(FormatError::InvalidVint)?;
    if end > data.len() {
        return Err(FormatError::Eof);
    }
    let mut value = if keep_marker {
        u64::from(first)
    } else {
        let mask = if length == 8 { 0 } else { 0xff >> length };
        u64::from(first & mask)
    };
    for &byte in &data[*position + 1..end] {
        value = (value << 8) | u64::from(byte);
    }
    *position = end;

    let unknown = if keep_marker {
        None
    } else {
        let all_ones = (1u64 << (7 * u32::from(length))) - 1;
        (value == all_ones).then_some(length)
    };
    Ok((value, unknown))
}

/// One parsed EBML element header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ElementHeader {
    /// Element ID with its marker bits.
    pub id: u64,
    /// Payload size in octets, or `None` for an unknown-size element.
    pub size: Option<u64>,
    /// Offset of the payload from the start of the buffer.
    pub data_start: usize,
}

impl ElementHeader {
    /// Reads an element header at `position` and advances past it.
    pub(crate) fn read(data: &[u8], position: &mut usize) -> Result<Self, FormatError> {
        let id_position = *position;
        let (id, _) = read_vint(data, position, true)?;
        let id_length = position
            .checked_sub(id_position)
            .ok_or(FormatError::InvalidElement)?;
        if id_length > usize::from(MAX_ID_LENGTH) || id == 0 {
            return Err(FormatError::InvalidElement);
        }
        let (size, unknown) = read_vint(data, position, false)?;
        let data_start = *position;
        if unknown.is_none() {
            let end = data_start
                .checked_add(usize::try_from(size).map_err(|_| FormatError::Limit("element size"))?)
                .ok_or(FormatError::Limit("element size"))?;
            if end > data.len() {
                return Err(FormatError::Eof);
            }
        }
        Ok(Self {
            id,
            size: unknown.is_none().then_some(size),
            data_start,
        })
    }

    /// Returns the payload end offset for a known-size element.
    pub(crate) fn data_end(&self) -> Result<usize, FormatError> {
        let size = self.size.ok_or(FormatError::InvalidElement)?;
        self.data_start
            .checked_add(usize::try_from(size).map_err(|_| FormatError::Limit("element size"))?)
            .ok_or(FormatError::Limit("element size"))
    }
}

/// Reads an unsigned EBML integer (`0` to `8` octets, big endian).
pub(crate) fn read_uinteger(data: &[u8]) -> Result<u64, FormatError> {
    if data.len() > 8 {
        return Err(FormatError::InvalidInteger);
    }
    let mut value = 0u64;
    for &byte in data {
        value = (value << 8) | u64::from(byte);
    }
    Ok(value)
}

/// Reads a signed EBML integer (`1` to `8` octets, two's complement).
pub(crate) fn read_sinteger(data: &[u8]) -> Result<i64, FormatError> {
    if data.is_empty() || data.len() > 8 {
        return Err(FormatError::InvalidInteger);
    }
    let negative = data[0] & 0x80 != 0;
    let mut value = if negative { !0u64 } else { 0u64 };
    for &byte in data {
        value = (value << 8) | u64::from(byte);
    }
    Ok(value as i64)
}

/// Reads an EBML float (`4` or `8` octets, big endian IEEE 754).
pub(crate) fn read_float(data: &[u8]) -> Result<f64, FormatError> {
    match data.len() {
        4 => Ok(f64::from(f32::from_bits(u32::from_be_bytes(
            data.try_into().map_err(|_| FormatError::InvalidFloat)?,
        )))),
        8 => Ok(f64::from_bits(u64::from_be_bytes(
            data.try_into().map_err(|_| FormatError::InvalidFloat)?,
        ))),
        _ => Err(FormatError::InvalidFloat),
    }
}

/// Reads an EBML string and strips the zero padding allowed by the profile.
pub(crate) fn read_string(data: &[u8]) -> Result<&str, FormatError> {
    let end = data
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |index| index + 1);
    std::str::from_utf8(&data[..end]).map_err(|_| FormatError::InvalidString)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vints_decode_ids_and_sizes() {
        let mut position = 0;
        assert_eq!(
            read_vint(&[0xa3], &mut position, true).unwrap(),
            (0xa3, None)
        );
        assert_eq!(position, 1);

        position = 0;
        assert_eq!(
            read_vint(&[0x1f, 0x43, 0xb6, 0x75], &mut position, true).unwrap(),
            (0x1f43_b675, None)
        );
        assert_eq!(position, 4);

        position = 0;
        assert_eq!(read_vint(&[0x81], &mut position, false).unwrap(), (1, None));
        position = 0;
        assert_eq!(
            read_vint(&[0x40, 0x02], &mut position, false).unwrap(),
            (2, None)
        );
    }

    #[test]
    fn vint_unknown_size_is_reported() {
        let mut position = 0;
        assert_eq!(
            read_vint(&[0xff], &mut position, false).unwrap(),
            (0x7f, Some(1))
        );
        position = 0;
        assert_eq!(
            read_vint(
                &[0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                &mut position,
                false
            )
            .unwrap(),
            (0x00ff_ffff_ffff_ffff, Some(8))
        );
    }

    #[test]
    fn vint_rejects_zero_and_truncated_input() {
        let mut position = 0;
        assert_eq!(
            read_vint(&[0x00], &mut position, false),
            Err(FormatError::InvalidVint)
        );
        let mut position = 0;
        assert_eq!(
            read_vint(&[0x40], &mut position, false),
            Err(FormatError::Eof)
        );
    }

    #[test]
    fn element_header_reads_known_and_unknown_sizes() {
        let data = [
            0x1f, 0x43, 0xb6, 0x75, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let mut position = 0;
        let header = ElementHeader::read(&data, &mut position).unwrap();
        assert_eq!(header.id, 0x1f43_b675);
        assert_eq!(header.size, None);
        assert_eq!(header.data_start, 12);

        let data = [0xa3, 0x84, 1, 2, 3, 4];
        let mut position = 0;
        let header = ElementHeader::read(&data, &mut position).unwrap();
        assert_eq!(header.id, 0xa3);
        assert_eq!(header.size, Some(4));
        assert_eq!(header.data_end().unwrap(), 6);
    }

    #[test]
    fn element_header_rejects_truncated_payload() {
        let data = [0xa3, 0x84, 1, 2];
        let mut position = 0;
        assert_eq!(
            ElementHeader::read(&data, &mut position),
            Err(FormatError::Eof)
        );
    }

    #[test]
    fn integers_and_floats_follow_ebml_rules() {
        assert_eq!(read_uinteger(&[]).unwrap(), 0);
        assert_eq!(read_uinteger(&[0x01, 0x02]).unwrap(), 0x102);
        assert_eq!(read_sinteger(&[0xff]).unwrap(), -1);
        assert_eq!(read_sinteger(&[0xfe]).unwrap(), -2);
        assert_eq!(read_sinteger(&[0x80]).unwrap(), -128);
        assert!(read_uinteger(&[0; 9]).is_err());
        assert!(read_sinteger(&[]).is_err());
        assert_eq!(read_float(&[0x3f, 0x80, 0x00, 0x00]).unwrap(), 1.0);
        assert!(read_float(&[0; 3]).is_err());
    }

    #[test]
    fn strings_strip_zero_padding_and_validate_utf8() {
        assert_eq!(read_string(b"matroska").unwrap(), "matroska");
        assert_eq!(read_string(b"webm\0\0").unwrap(), "webm");
        assert_eq!(read_string(b"\0\0").unwrap(), "");
        assert_eq!(read_string(&[0xff]), Err(FormatError::InvalidString));
    }
}
