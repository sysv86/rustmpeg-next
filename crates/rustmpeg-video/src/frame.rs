// SPDX-License-Identifier: Apache-2.0
//! Planar 4:2:0 (I420) frame buffers.

/// One planar 4:2:0 frame with 8-bit BT.709 limited-range samples.
///
/// Chroma planes are stored at half resolution, rounded up for odd
/// dimensions: `ceil(width / 2) * ceil(height / 2)` bytes each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct I420Frame {
    /// Frame width in luma samples.
    pub width: u32,
    /// Frame height in luma samples.
    pub height: u32,
    /// Full-resolution luma plane (`width * height` bytes).
    pub y: Vec<u8>,
    /// Half-resolution Cb plane, in raster order.
    pub u: Vec<u8>,
    /// Half-resolution Cr plane, in raster order.
    pub v: Vec<u8>,
}

impl I420Frame {
    /// Chroma plane dimensions (rounded up for odd sizes).
    pub fn chroma_size(&self) -> (u32, u32) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// `true` when the three plane lengths match the declared dimensions.
    ///
    /// Zero-sized frames are rejected; size arithmetic is performed in `u64`
    /// so that hostile dimensions cannot overflow the length check.
    pub fn is_valid(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }
        let luma = u64::from(self.width) * u64::from(self.height);
        let (chroma_width, chroma_height) = self.chroma_size();
        let chroma = u64::from(chroma_width) * u64::from(chroma_height);
        usize::try_from(luma).is_ok_and(|len| self.y.len() == len)
            && usize::try_from(chroma).is_ok_and(|len| self.u.len() == len && self.v.len() == len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_size_rounds_up_for_odd_dimensions() {
        let frame = I420Frame {
            width: 3,
            height: 3,
            y: vec![0; 9],
            u: vec![0; 4],
            v: vec![0; 4],
        };
        assert_eq!(frame.chroma_size(), (2, 2));
        assert!(frame.is_valid());
    }

    #[test]
    fn mismatched_plane_lengths_are_invalid() {
        let frame = I420Frame {
            width: 4,
            height: 4,
            y: vec![0; 16],
            u: vec![0; 3],
            v: vec![0; 4],
        };
        assert!(!frame.is_valid());
    }

    #[test]
    fn zero_and_oversized_dimensions_are_invalid() {
        let zero = I420Frame {
            width: 0,
            height: 0,
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
        };
        assert!(!zero.is_valid());

        let overflow = I420Frame {
            width: u32::MAX,
            height: u32::MAX,
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
        };
        assert!(!overflow.is_valid());
    }
}
