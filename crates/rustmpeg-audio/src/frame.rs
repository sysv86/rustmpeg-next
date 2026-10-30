// SPDX-License-Identifier: Apache-2.0
//! Decoded audio frames shared by the audio codecs.

/// One decoded block of planar integer PCM.
///
/// Samples are stored in their native integer representation: a frame with
/// `bits_per_sample == 24` contains values in the signed 24-bit range, not
/// values shifted into the most significant bits of an [`i32`].  Consumers that
/// need a fixed-width interleaved layout must shift the samples themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmFrame {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of channels, which is also the number of planes.
    pub channels: u8,
    /// Bit depth of the samples in this frame.
    pub bits_per_sample: u8,
    /// Frame number as coded in the bitstream.
    ///
    /// For variable block size streams, which carry no frame number, this is
    /// the zero-based index of the frame in decode order.
    pub frame_number: u64,
    /// Number of the first sample of this frame in decode order.
    pub sample_number: u64,
    /// Planar samples, one plane per channel, each containing `block_size`
    /// native integer values.
    pub planes: Vec<Vec<i32>>,
}

impl PcmFrame {
    /// Returns the number of samples per channel in this frame.
    ///
    /// This is zero only for a hand-built frame with no planes.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.planes.first().map_or(0, Vec::len)
    }

    /// Returns the samples of one channel, or [`None`] if `channel` is out of
    /// range.
    #[must_use]
    pub fn plane(&self, channel: usize) -> Option<&[i32]> {
        self.planes.get(channel).map(Vec::as_slice)
    }
}
