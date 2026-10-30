// SPDX-License-Identifier: Apache-2.0
//! ISO/IEC 18181-2 container parser and writer for JPEG XL.
//!
//! A JPEG XL file is either:
//! * a bare codestream starting with the 2-byte signature `0xFF 0x0A`, or
//! * an ISOBMFF-based box container starting with the 12-byte `JXL ` signature
//!   box `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A` followed by `ftyp` (`jxl `) and
//!   either a single `jxlc` box or a sequence of `jxlp` partial codestream boxes.
//!
//! Per `docs/CONCEPT.md`, the legacy JPEG bitstream reconstruction (`jbrd`)
//! feature is forcibly excluded and rejected with a typed [`JxlError::Unsupported`].

use std::borrow::Cow;

use super::JxlError;

/// Bare JPEG XL codestream magic bytes (`0xFF 0x0A`).
pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];

/// ISO/IEC 18181-2 container signature box (`JXL ` box, 12 bytes).
pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
];

/// Standard 20-byte `ftyp` box for ISO/IEC 18181-2 containers.
pub const CONTAINER_FTYP_BOX: [u8; 20] = [
    0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00, 0x00,
    b'j', b'x', b'l', b' ',
];

/// Maximum number of container boxes allowed in a single file to prevent DoS.
pub const MAX_CONTAINER_BOXES: usize = 16_384;

/// Maximum assembled codestream size (256 MiB).
pub const MAX_CODESTREAM_BYTES: usize = 256 * 1024 * 1024;

/// Format framing of a JPEG XL bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum JxlContainerKind {
    /// Naked codestream starting with `0xFF 0x0A`.
    #[default]
    BareCodestream,
    /// ISO/IEC 18181-2 box container (`JXL ` + `ftyp` + `jxlc`/`jxlp`).
    Container,
}

/// Parsed JPEG XL container or bare codestream.
#[derive(Debug, Clone)]
pub struct ParsedContainer<'a> {
    /// Framing kind detected from the leading signature.
    pub kind: JxlContainerKind,
    /// Assembled codestream bytes (zero-copy borrowed for bare codestreams and
    /// single-`jxlc` containers; owned when reassembled from multiple `jxlp` boxes).
    pub codestream: Cow<'a, [u8]>,
    /// Optional codestream level from a `jxll` box (`5` or `10`).
    pub level: Option<u8>,
    /// Optional raw Exif payload (excluding the 4-byte TIFF header offset).
    pub exif: Option<Vec<u8>>,
    /// Optional raw XML/XMP metadata payload.
    pub xmp: Option<Vec<u8>>,
}

/// Parses a bare JPEG XL codestream or an ISO/IEC 18181-2 box container.
///
/// # Errors
///
/// Returns:
/// * [`JxlError::InvalidSignature`] if `input` starts with neither `0xFF 0x0A`
///   nor the 12-byte ISO/IEC 18181-2 container signature;
/// * [`JxlError::Unsupported`] if the container carries a `jbrd` legacy JPEG
///   reconstruction box (excluded by `docs/CONCEPT.md`);
/// * [`JxlError::InvalidContainer`] if box headers or `jxlc`/`jxlp` sequences
///   violate ISO/IEC 18181-2.
pub fn parse_container(input: &[u8]) -> Result<ParsedContainer<'_>, JxlError> {
    if input.len() < 2 {
        return Err(JxlError::UnexpectedEof);
    }
    if input.starts_with(&CODESTREAM_SIGNATURE) {
        if input.len() > MAX_CODESTREAM_BYTES {
            return Err(JxlError::LimitExceeded("codestream exceeds size limit"));
        }
        return Ok(ParsedContainer {
            kind: JxlContainerKind::BareCodestream,
            codestream: Cow::Borrowed(input),
            level: None,
            exif: None,
            xmp: None,
        });
    }
    if !input.starts_with(&CONTAINER_SIGNATURE) {
        return Err(JxlError::InvalidSignature);
    }

    let mut pos = CONTAINER_SIGNATURE.len();
    let mut box_count = 0usize;
    let mut seen_ftyp = false;
    let mut single_jxlc: Option<&[u8]> = None;
    let mut partial_codestream: Vec<u8> = Vec::new();
    let mut next_jxlp_index: Option<u32> = None;
    let mut jxlp_finished = false;
    let mut level: Option<u8> = None;
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;

    while pos < input.len() {
        box_count += 1;
        if box_count > MAX_CONTAINER_BOXES {
            return Err(JxlError::LimitExceeded("too many container boxes"));
        }
        let rem = &input[pos..];
        if rem.len() < 8 {
            return Err(JxlError::InvalidContainer("truncated box header"));
        }
        let size32 = u32::from_be_bytes([rem[0], rem[1], rem[2], rem[3]]);
        let box_type: [u8; 4] = [rem[4], rem[5], rem[6], rem[7]];
        let (header_len, payload_len) = match size32 {
            0 => (8usize, rem.len() - 8),
            1 => {
                if rem.len() < 16 {
                    return Err(JxlError::InvalidContainer("truncated 64-bit box header"));
                }
                let size64 = u64::from_be_bytes([
                    rem[8], rem[9], rem[10], rem[11], rem[12], rem[13], rem[14], rem[15],
                ]);
                if size64 < 16 {
                    return Err(JxlError::InvalidContainer(
                        "64-bit box size smaller than 16",
                    ));
                }
                let total = usize::try_from(size64)
                    .map_err(|_| JxlError::LimitExceeded("box size overflows usize"))?;
                (16usize, total - 16)
            }
            2..=7 => {
                return Err(JxlError::InvalidContainer("box size in 2..=7 is invalid"));
            }
            n => (8usize, (n as usize) - 8),
        };

        let box_end = pos
            .checked_add(header_len)
            .and_then(|v| v.checked_add(payload_len))
            .ok_or(JxlError::InvalidContainer("box size overflow"))?;
        if box_end > input.len() {
            return Err(JxlError::InvalidContainer("box extends past end of file"));
        }
        let payload = &input[(pos + header_len)..box_end];
        pos = box_end;

        match &box_type {
            b"ftyp" => {
                if box_count != 1 {
                    return Err(JxlError::InvalidContainer(
                        "ftyp box must immediately follow the JXL signature box",
                    ));
                }
                if payload.len() < 8 || &payload[0..4] != b"jxl " {
                    return Err(JxlError::InvalidContainer("invalid ftyp brand"));
                }
                seen_ftyp = true;
            }
            b"jxll" => {
                if payload.len() != 1 {
                    return Err(JxlError::InvalidContainer(
                        "jxll box payload must be 1 byte",
                    ));
                }
                level = Some(payload[0]);
            }
            b"jxlc" => {
                if !seen_ftyp {
                    return Err(JxlError::InvalidContainer("jxlc box precedes ftyp"));
                }
                if single_jxlc.is_some() || next_jxlp_index.is_some() {
                    return Err(JxlError::InvalidContainer(
                        "duplicate jxlc or mixed jxlc/jxlp boxes",
                    ));
                }
                if payload.len() > MAX_CODESTREAM_BYTES {
                    return Err(JxlError::LimitExceeded("codestream exceeds size limit"));
                }
                single_jxlc = Some(payload);
            }
            b"jxlp" => {
                if !seen_ftyp {
                    return Err(JxlError::InvalidContainer("jxlp box precedes ftyp"));
                }
                if single_jxlc.is_some() || jxlp_finished {
                    return Err(JxlError::InvalidContainer(
                        "invalid jxlp box after jxlc or final jxlp",
                    ));
                }
                if payload.len() < 4 {
                    return Err(JxlError::InvalidContainer("jxlp box smaller than 4 bytes"));
                }
                let raw_idx = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
                let is_last = (raw_idx & 0x8000_0000) != 0;
                let idx = raw_idx & 0x7fff_ffff;
                let expected = next_jxlp_index.unwrap_or(0);
                if idx != expected {
                    return Err(JxlError::InvalidContainer("out-of-order jxlp box index"));
                }
                next_jxlp_index = Some(idx + 1);
                if is_last {
                    jxlp_finished = true;
                }
                let chunk = &payload[4..];
                if partial_codestream.len().saturating_add(chunk.len()) > MAX_CODESTREAM_BYTES {
                    return Err(JxlError::LimitExceeded("codestream exceeds size limit"));
                }
                partial_codestream.extend_from_slice(chunk);
            }
            b"jbrd" => {
                return Err(JxlError::Unsupported(
                    "legacy JPEG bitstream reconstruction (jbrd) is excluded by architecture policy",
                ));
            }
            b"brob" => {
                if payload.len() < 4 {
                    return Err(JxlError::InvalidContainer("brob box smaller than 4 bytes"));
                }
                if &payload[0..4] == b"jbrd" {
                    return Err(JxlError::Unsupported(
                        "legacy JPEG bitstream reconstruction (jbrd) is excluded by architecture policy",
                    ));
                }
                tracing::debug!(
                    inner_type = ?std::str::from_utf8(&payload[0..4]),
                    "skipping Brotli-compressed auxiliary metadata box"
                );
            }
            b"Exif" => {
                if payload.len() < 4 {
                    return Err(JxlError::InvalidContainer("Exif box smaller than 4 bytes"));
                }
                let tiff_offset =
                    u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
                let exif_body = &payload[4..];
                if tiff_offset > exif_body.len() {
                    return Err(JxlError::InvalidContainer(
                        "Exif tiff_header_offset exceeds box payload",
                    ));
                }
                if exif.is_none() {
                    exif = Some(exif_body[tiff_offset..].to_vec());
                }
            }
            b"xml " => {
                if xmp.is_none() {
                    xmp = Some(payload.to_vec());
                }
            }
            _ => {
                tracing::trace!(?box_type, len = payload.len(), "ignoring auxiliary box");
            }
        }
    }

    let codestream = if let Some(cs) = single_jxlc {
        Cow::Borrowed(cs)
    } else if !partial_codestream.is_empty() {
        Cow::Owned(partial_codestream)
    } else {
        return Err(JxlError::InvalidContainer(
            "container has no jxlc or jxlp codestream box",
        ));
    };

    if !codestream.starts_with(&CODESTREAM_SIGNATURE) {
        return Err(JxlError::InvalidSignature);
    }

    Ok(ParsedContainer {
        kind: JxlContainerKind::Container,
        codestream,
        level,
        exif,
        xmp,
    })
}

/// Wraps a raw JPEG XL codestream (`0xFF 0x0A ...`) inside an ISO/IEC 18181-2
/// box container (`JXL ` signature + `ftyp` + optional `jxll` + `jxlc`).
///
/// # Errors
///
/// Returns [`JxlError::InvalidSignature`] if `codestream` does not begin with `0xFF 0x0A`.
pub fn write_container(
    codestream: &[u8],
    level: Option<u8>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Result<Vec<u8>, JxlError> {
    if !codestream.starts_with(&CODESTREAM_SIGNATURE) {
        return Err(JxlError::InvalidSignature);
    }
    let mut out = Vec::with_capacity(
        CONTAINER_SIGNATURE.len() + CONTAINER_FTYP_BOX.len() + 16 + codestream.len(),
    );
    out.extend_from_slice(&CONTAINER_SIGNATURE);
    out.extend_from_slice(&CONTAINER_FTYP_BOX);

    if let Some(lvl) = level {
        write_box(&mut out, *b"jxll", &[lvl]);
    }
    write_box(&mut out, *b"jxlc", codestream);

    if let Some(exif_bytes) = exif {
        let mut payload = Vec::with_capacity(4 + exif_bytes.len());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(exif_bytes);
        write_box(&mut out, *b"Exif", &payload);
    }
    if let Some(xmp_bytes) = xmp {
        write_box(&mut out, *b"xml ", xmp_bytes);
    }
    Ok(out)
}

fn write_box(out: &mut Vec<u8>, box_type: [u8; 4], payload: &[u8]) {
    let total = 8u64 + (payload.len() as u64);
    if let Ok(size32) = u32::try_from(total) {
        out.extend_from_slice(&size32.to_be_bytes());
        out.extend_from_slice(&box_type);
        out.extend_from_slice(payload);
    } else {
        let total64 = 16u64 + (payload.len() as u64);
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(&box_type);
        out.extend_from_slice(&total64.to_be_bytes());
        out.extend_from_slice(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_codestream_and_container_round_trip() {
        let dummy_cs = [0xff, 0x0a, 0x00, 0x01, 0x02, 0x03];
        let bare = parse_container(&dummy_cs).expect("bare");
        assert_eq!(bare.kind, JxlContainerKind::BareCodestream);
        assert_eq!(&*bare.codestream, &dummy_cs);

        let wrapped =
            write_container(&dummy_cs, Some(5), Some(b"II*\0"), Some(b"<xmp/>")).expect("wrap");
        let parsed = parse_container(&wrapped).expect("parse container");
        assert_eq!(parsed.kind, JxlContainerKind::Container);
        assert_eq!(&*parsed.codestream, &dummy_cs);
        assert_eq!(parsed.level, Some(5));
        assert_eq!(parsed.exif.as_deref(), Some(&b"II*\0"[..]));
        assert_eq!(parsed.xmp.as_deref(), Some(&b"<xmp/>"[..]));
    }

    #[test]
    fn jbrd_box_is_rejected_by_policy() {
        let dummy_cs = [0xff, 0x0a, 0x00];
        let mut bytes = write_container(&dummy_cs, None, None, None).expect("wrap");
        write_box(&mut bytes, *b"jbrd", &[1, 2, 3, 4]);
        let err = parse_container(&bytes).expect_err("jbrd must be rejected");
        assert!(matches!(err, JxlError::Unsupported(_)));
    }
}
