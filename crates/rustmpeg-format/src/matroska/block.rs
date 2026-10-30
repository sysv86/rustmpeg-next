// SPDX-License-Identifier: Apache-2.0
//! Matroska block header and lacing (RFC 9559 §10.1–§10.3).

use bytes::Bytes;

use super::FormatError;
use crate::ebml::read_vint;

/// Lacing mode of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lacing {
    None,
    Xiph,
    Fixed,
    Ebml,
}

/// One parsed Matroska block, split into its laced frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    /// Track number from the block header.
    pub track: u64,
    /// Signed timestamp relative to the cluster, in Segment Ticks.
    pub relative_ticks: i16,
    /// Keyframe flag (SimpleBlock only; BlockGroup derives it from references).
    pub keyframe: bool,
    /// Invisible flag.
    pub invisible: bool,
    /// Discardable flag.
    pub discardable: bool,
    /// Zero-copy frame data, one entry per laced frame.
    pub frames: Vec<Bytes>,
}

/// Parses a `SimpleBlock` or `Block` payload.
///
/// The block is expected to occupy the whole `data` slice; the caller passes
/// the element payload, so trailing side data cannot be mistaken for frames.
pub(crate) fn parse_block(data: &Bytes, simple: bool) -> Result<Block, FormatError> {
    if data.len() < 4 {
        return Err(FormatError::Block("block header is truncated"));
    }
    let mut position = 0usize;
    let (track, _) = read_vint(data, &mut position, false)?;
    if track == 0 {
        return Err(FormatError::Block("block track number is zero"));
    }
    let relative = data
        .get(position..position + 2)
        .ok_or(FormatError::Block("block timestamp is truncated"))?;
    let relative_ticks = i16::from_be_bytes([relative[0], relative[1]]);
    position += 2;
    let flags = *data
        .get(position)
        .ok_or(FormatError::Block("block flags are truncated"))?;
    position += 1;

    let keyframe = simple && flags & 0x80 != 0;
    let invisible = flags & 0x08 != 0;
    let discardable = flags & 0x01 != 0;
    let lacing = match (flags >> 1) & 0x03 {
        0 => Lacing::None,
        1 => Lacing::Xiph,
        3 => Lacing::Ebml,
        2 => Lacing::Fixed,
        _ => unreachable!(),
    };

    let frames = match lacing {
        Lacing::None => vec![data.slice(position..)],
        Lacing::Xiph => lace_xiph(data, position)?,
        Lacing::Fixed => lace_fixed(data, position)?,
        Lacing::Ebml => lace_ebml(data, position)?,
    };
    if frames.iter().any(Bytes::is_empty) {
        return Err(FormatError::Block("empty laced frame"));
    }
    Ok(Block {
        track,
        relative_ticks,
        keyframe,
        invisible,
        discardable,
        frames,
    })
}

/// Reads the lace frame count (`frames - 1`) and returns the count.
fn lace_count(data: &Bytes, position: &mut usize) -> Result<usize, FormatError> {
    let count = *data
        .get(*position)
        .ok_or(FormatError::Block("lacing header is truncated"))?;
    *position += 1;
    Ok(usize::from(count) + 1)
}

fn lace_xiph(data: &Bytes, mut position: usize) -> Result<Vec<Bytes>, FormatError> {
    let count = lace_count(data, &mut position)?;
    let mut sizes = Vec::with_capacity(count.saturating_sub(1));
    let mut total = 0usize;
    for _ in 0..count - 1 {
        let mut size = 0usize;
        loop {
            let byte = *data
                .get(position)
                .ok_or(FormatError::Block("Xiph lacing size is truncated"))?;
            position += 1;
            size = size
                .checked_add(usize::from(byte))
                .ok_or(FormatError::Limit("frame size"))?;
            if byte != 255 {
                break;
            }
        }
        total = total
            .checked_add(size)
            .ok_or(FormatError::Limit("frame size"))?;
        sizes.push(size);
    }
    split_frames(data, position, &sizes, total)
}

fn lace_fixed(data: &Bytes, mut position: usize) -> Result<Vec<Bytes>, FormatError> {
    let count = lace_count(data, &mut position)?;
    let available = data.len() - position;
    if !available.is_multiple_of(count) {
        return Err(FormatError::Block(
            "fixed lacing does not divide the block evenly",
        ));
    }
    let size = available / count;
    if size == 0 {
        return Err(FormatError::Block("empty laced frame"));
    }
    Ok((0..count)
        .map(|index| {
            let start = position + index * size;
            data.slice(start..start + size)
        })
        .collect())
}

fn lace_ebml(data: &Bytes, mut position: usize) -> Result<Vec<Bytes>, FormatError> {
    let count = lace_count(data, &mut position)?;
    let mut sizes = Vec::with_capacity(count.saturating_sub(1));
    let mut total = 0usize;
    let (first, _) = read_vint(data, &mut position, false)?;
    let first = usize::try_from(first).map_err(|_| FormatError::Limit("frame size"))?;
    sizes.push(first);
    total = total
        .checked_add(first)
        .ok_or(FormatError::Limit("frame size"))?;
    let mut previous = first;
    for _ in 1..count - 1 {
        let start = position;
        let (value, _) = read_vint(data, &mut position, false)?;
        let length = position
            .checked_sub(start)
            .ok_or(FormatError::InvalidVint)?;
        let bias = (1i64 << (7 * length as u32 - 1)) - 1;
        let delta = i64::try_from(value)
            .map_err(|_| FormatError::Limit("frame size"))?
            .checked_sub(bias)
            .ok_or(FormatError::Limit("frame size"))?;
        let size = i64::try_from(previous)
            .map_err(|_| FormatError::Limit("frame size"))?
            .checked_add(delta)
            .filter(|size| *size > 0)
            .ok_or(FormatError::Block("invalid EBML lacing delta"))?;
        let size = usize::try_from(size).map_err(|_| FormatError::Limit("frame size"))?;
        total = total
            .checked_add(size)
            .ok_or(FormatError::Limit("frame size"))?;
        sizes.push(size);
        previous = size;
    }
    split_frames(data, position, &sizes, total)
}

/// Splits the frame area after the lacing header and derives the last size.
fn split_frames(
    data: &Bytes,
    position: usize,
    sizes: &[usize],
    total: usize,
) -> Result<Vec<Bytes>, FormatError> {
    if position + total > data.len() {
        return Err(FormatError::Block("laced frames exceed the block"));
    }
    let mut frames = Vec::with_capacity(sizes.len() + 1);
    let mut start = position;
    for &size in sizes {
        frames.push(data.slice(start..start + size));
        start += size;
    }
    frames.push(data.slice(start..));
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(payload: Vec<u8>) -> Bytes {
        Bytes::from(payload)
    }

    #[test]
    fn parses_a_simple_block_without_lacing() {
        let data = block(vec![0x81, 0x00, 0x01, 0x80, 1, 2, 3]);
        let parsed = parse_block(&data, true).unwrap();
        assert_eq!(parsed.track, 1);
        assert_eq!(parsed.relative_ticks, 1);
        assert!(parsed.keyframe);
        assert!(!parsed.invisible);
        assert!(!parsed.discardable);
        assert_eq!(parsed.frames, [Bytes::from_static(&[1, 2, 3])]);
    }

    #[test]
    fn parses_xiph_lacing() {
        // Track 1, timestamp 0, flags: keyframe + Xiph lacing, 2 frames.
        let mut payload = vec![0x81, 0x00, 0x00, 0x82, 0x01, 0xff, 0x2d];
        payload.extend_from_slice(&[0xaa; 300]);
        payload.extend_from_slice(&[0xbb; 10]);
        let parsed = parse_block(&block(payload), true).unwrap();
        assert_eq!(parsed.frames.len(), 2);
        assert_eq!(parsed.frames[0].len(), 300);
        assert_eq!(parsed.frames[1].len(), 10);
        assert_eq!(parsed.frames[0][0], 0xaa);
        assert_eq!(parsed.frames[1][0], 0xbb);
    }

    #[test]
    fn parses_ebml_lacing_with_signed_deltas() {
        // RFC 9559 Table 38: 800, 500, 1000 octets.
        let mut payload = vec![0x81, 0x00, 0x00, 0x86, 0x02, 0x43, 0x20, 0x5e, 0xd3];
        payload.extend(std::iter::repeat_n(0x11, 800));
        payload.extend(std::iter::repeat_n(0x22, 500));
        payload.extend(std::iter::repeat_n(0x33, 1000));
        let parsed = parse_block(&block(payload), true).unwrap();
        assert_eq!(parsed.frames.len(), 3);
        assert_eq!(parsed.frames[0].len(), 800);
        assert_eq!(parsed.frames[1].len(), 500);
        assert_eq!(parsed.frames[2].len(), 1000);
    }

    #[test]
    fn parses_fixed_lacing() {
        let mut payload = vec![0x81, 0x00, 0x00, 0x84, 0x02];
        payload.extend_from_slice(&[1; 30]);
        let parsed = parse_block(&block(payload), true).unwrap();
        assert_eq!(parsed.frames.len(), 3);
        assert!(parsed.frames.iter().all(|frame| frame.len() == 10));
    }

    #[test]
    fn fixed_lacing_rejects_a_ragged_payload() {
        let mut payload = vec![0x81, 0x00, 0x00, 0x84, 0x02];
        payload.extend_from_slice(&[1; 31]);
        assert!(parse_block(&block(payload), true).is_err());
    }

    #[test]
    fn block_group_blocks_are_not_keyframes_by_default() {
        let data = block(vec![0x81, 0x00, 0x00, 0x00, 9]);
        let parsed = parse_block(&data, false).unwrap();
        assert!(!parsed.keyframe);
        assert!(!parsed.discardable);
    }

    #[test]
    fn lacing_rejects_truncated_sizes_and_empty_frames() {
        let data = block(vec![0x81, 0x00, 0x00, 0x82, 0x01, 0xff]);
        assert_eq!(
            parse_block(&data, true),
            Err(FormatError::Block("Xiph lacing size is truncated"))
        );

        let data = block(vec![0x81, 0x00, 0x00, 0x82, 0x01, 0x00]);
        assert_eq!(
            parse_block(&data, true),
            Err(FormatError::Block("empty laced frame"))
        );
    }
}
