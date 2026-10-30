// SPDX-License-Identifier: Apache-2.0
//! Audio sample formats and interleaved frames.

use crate::buffer::Buffer;
use crate::error::{CoreError, Result};
use crate::limits;

/// Largest channel count accepted by the core container (IAMF 28-channel
/// scenes fit comfortably).
pub const MAX_CHANNELS: u16 = 64;

/// Storage format of one audio sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit.
    U8,
    /// Signed 16-bit little-endian.
    S16,
    /// Signed 24-bit little-endian, packed in three bytes.
    S24,
    /// Signed 32-bit little-endian.
    S32,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

impl SampleFormat {
    /// Bytes occupied by one sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16 => 2,
            Self::S24 => 3,
            Self::S32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// `true` for floating-point samples.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

/// Interleaved audio format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    /// Per-sample storage.
    pub sample_format: SampleFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Interleaved channel count.
    pub channels: u16,
}

impl AudioFormat {
    /// Creates a validated audio format.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] for a zero sample rate or channel
    /// count, and [`CoreError::LimitExceeded`] above [`MAX_CHANNELS`].
    pub fn new(sample_format: SampleFormat, sample_rate: u32, channels: u16) -> Result<Self> {
        if sample_rate == 0 {
            return Err(CoreError::InvalidConfig(
                "sample rate must be non-zero".into(),
            ));
        }
        if channels == 0 {
            return Err(CoreError::InvalidConfig(
                "channel count must be non-zero".into(),
            ));
        }
        if channels > MAX_CHANNELS {
            return Err(CoreError::limit(
                "audio channels",
                u64::from(channels),
                u64::from(MAX_CHANNELS),
            ));
        }
        Ok(Self {
            sample_format,
            sample_rate,
            channels,
        })
    }

    /// Bytes occupied by one interleaved frame (one sample per channel).
    pub fn bytes_per_frame(self) -> Result<usize> {
        limits::checked_mul(
            self.sample_format.bytes_per_sample(),
            usize::from(self.channels),
            "audio frame size",
        )
    }

    /// Bytes required for `samples` interleaved frames.
    ///
    /// # Errors
    ///
    /// Propagates overflow.
    pub fn bytes_for_samples(self, samples: usize) -> Result<usize> {
        limits::checked_mul(self.bytes_per_frame()?, samples, "audio buffer size")
    }
}

/// One interleaved audio frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// Sample layout.
    pub format: AudioFormat,
    /// Interleaved sample bytes.
    pub data: Buffer,
}

impl AudioFrame {
    /// Wraps an interleaved buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidData`] when the buffer is not a whole number
    /// of interleaved frames.
    pub fn new(format: AudioFormat, data: Buffer) -> Result<Self> {
        let frame_bytes = format.bytes_per_frame()?;
        if !data.len().is_multiple_of(frame_bytes) {
            return Err(CoreError::InvalidData(
                "audio buffer is not a whole number of interleaved frames".into(),
            ));
        }
        Ok(Self { format, data })
    }

    /// Number of interleaved frames (one sample per channel).
    ///
    /// # Errors
    ///
    /// Propagates frame-size overflow.
    pub fn sample_count(&self) -> Result<usize> {
        let frame_bytes = self.format.bytes_per_frame()?;
        Ok(self.data.len() / frame_bytes)
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> Result<f64> {
        Ok(self.sample_count()? as f64 / f64::from(self.format.sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn sample_widths_match_the_formats() {
        assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
        assert_eq!(SampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::S24.bytes_per_sample(), 3);
        assert_eq!(SampleFormat::F64.bytes_per_sample(), 8);
        assert!(SampleFormat::F32.is_float());
        assert!(!SampleFormat::S32.is_float());
    }

    #[test]
    fn audio_formats_validate_and_size() {
        let format = AudioFormat::new(SampleFormat::S16, 48_000, 2).expect("format");
        assert_eq!(format.bytes_per_frame().expect("frame"), 4);
        assert_eq!(format.bytes_for_samples(480).expect("bytes"), 1920);
        assert!(AudioFormat::new(SampleFormat::S16, 0, 2).is_err());
        assert!(AudioFormat::new(SampleFormat::S16, 48_000, 0).is_err());
        assert!(AudioFormat::new(SampleFormat::S16, 48_000, MAX_CHANNELS + 1).is_err());
    }

    #[test]
    fn frames_must_contain_whole_samples() {
        let format = AudioFormat::new(SampleFormat::S16, 48_000, 2).expect("format");
        let frame = AudioFrame::new(format, Buffer::from_vec(vec![0; 8])).expect("frame");
        assert_eq!(frame.sample_count().expect("samples"), 2);
        assert!((frame.duration_seconds().expect("duration") - 2.0 / 48_000.0).abs() < 1e-12);
        assert!(AudioFrame::new(format, Buffer::from_vec(vec![0; 7])).is_err());
    }
}
