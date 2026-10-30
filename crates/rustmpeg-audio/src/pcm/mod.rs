// SPDX-License-Identifier: Apache-2.0
//! Raw linear PCM codecs (`pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
//! `pcm_f64le`).
//!
//! Integer formats decode to [`PcmFrame`] with native-width signed samples, so a
//! 24-bit stream produces values in `-8388608..=8388607` rather than values
//! shifted into the high bits of an [`i32`].  Floating-point formats decode to
//! [`FloatFrame`] with [`f64`] planes; `f32` samples widen exactly.
//!
//! The module also provides loss-preserving conversion helpers between the two
//! representations ([`PcmFrame::to_float`] and [`FloatFrame::to_integer`]).
//!
//! Only the little-endian interleaved layouts are supported, matching the
//! formats listed in the project codec matrix.

use thiserror::Error;

use crate::PcmFrame;

/// Maximum number of bytes accepted by one raw PCM decode call (1 GiB).
pub const MAX_PCM_BYTES: usize = 1 << 30;

/// Raw PCM sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    /// Signed 16-bit little-endian.
    S16Le,
    /// Signed 24-bit little-endian.
    S24Le,
    /// Signed 32-bit little-endian.
    S32Le,
    /// IEEE 754 binary32 little-endian.
    F32Le,
    /// IEEE 754 binary64 little-endian.
    F64Le,
}

impl PcmFormat {
    /// Returns the number of stored bits per sample.
    #[must_use]
    pub const fn bits_per_sample(self) -> u8 {
        match self {
            Self::S16Le => 16,
            Self::S24Le => 24,
            Self::S32Le => 32,
            Self::F32Le => 32,
            Self::F64Le => 64,
        }
    }

    /// Returns the number of stored bytes per sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16Le => 2,
            Self::S24Le => 3,
            Self::S32Le => 4,
            Self::F32Le => 4,
            Self::F64Le => 8,
        }
    }

    /// Returns `true` for the IEEE 754 formats.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32Le | Self::F64Le)
    }
}

/// Stream parameters for raw PCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmParams {
    /// Sample format.
    pub format: PcmFormat,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Channel count (`1..=8`).
    pub channels: u8,
}

impl PcmParams {
    /// Validates the parameters.
    pub fn validate(&self) -> Result<(), PcmError> {
        if self.sample_rate == 0 {
            return Err(PcmError::InvalidParams("sample rate must be non-zero"));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(PcmError::InvalidParams(
                "channel count must be between 1 and 8",
            ));
        }
        Ok(())
    }

    /// Returns the size in bytes of one interleaved sample frame.
    #[must_use]
    pub const fn block_align(&self) -> usize {
        self.format.bytes_per_sample() * self.channels as usize
    }
}

/// Errors produced while decoding or encoding raw PCM.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PcmError {
    /// The stream parameters are invalid.
    #[error("invalid PCM parameters: {0}")]
    InvalidParams(&'static str),
    /// The input length is not a whole number of sample frames.
    #[error("PCM byte length {length} is not a multiple of the {block_align}-byte frame size")]
    Truncated {
        /// Supplied byte length.
        length: usize,
        /// Size of one interleaved sample frame.
        block_align: usize,
    },
    /// The input exceeds [`MAX_PCM_BYTES`].
    #[error("PCM input exceeds the 1 GiB limit")]
    LimitExceeded,
    /// A conversion cannot represent the source bit depth.
    #[error("PCM conversion error: {0}")]
    Conversion(&'static str),
}

/// One block of planar floating-point PCM.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatFrame {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of channels, which is also the number of planes.
    pub channels: u8,
    /// Number of the first sample in decode order.
    pub sample_number: u64,
    /// Planar samples, one plane per channel.
    pub planes: Vec<Vec<f64>>,
}

impl FloatFrame {
    /// Returns the number of samples per channel.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.planes.first().map_or(0, Vec::len)
    }

    /// Converts to native integer PCM at `bits_per_sample`.
    ///
    /// Values are scaled by `2^(bits-1)`, rounded to the nearest integer, and
    /// clamped to the symmetric integer range.  `bits_per_sample` must be in
    /// `4..=32`.
    pub fn to_integer(&self, bits_per_sample: u8) -> Result<PcmFrame, PcmError> {
        if !(4..=32).contains(&bits_per_sample) {
            return Err(PcmError::Conversion("bit depth outside 4..=32"));
        }
        let scale = f64::from(1u32 << (bits_per_sample - 1));
        let min = -(1i64 << (bits_per_sample - 1));
        let max = (1i64 << (bits_per_sample - 1)) - 1;
        let mut planes = Vec::with_capacity(self.planes.len());
        for plane in &self.planes {
            let mut out = Vec::with_capacity(plane.len());
            for &sample in plane {
                let scaled = (sample * scale).round();
                out.push(scaled.clamp(min as f64, max as f64) as i32);
            }
            planes.push(out);
        }
        Ok(PcmFrame {
            sample_rate: self.sample_rate,
            channels: self.channels,
            bits_per_sample,
            frame_number: 0,
            sample_number: self.sample_number,
            planes,
        })
    }
}

impl PcmFrame {
    /// Converts native integer samples to normalized floating point.
    ///
    /// Samples are divided by `2^(bits-1)`, so the minimum value is `-1.0` and
    /// the maximum is `1 - 2^(1-bits)`.
    #[must_use]
    pub fn to_float(&self) -> FloatFrame {
        let scale = 1.0 / f64::from(1u32 << (self.bits_per_sample - 1));
        FloatFrame {
            sample_rate: self.sample_rate,
            channels: self.channels,
            sample_number: self.sample_number,
            planes: self
                .planes
                .iter()
                .map(|plane| {
                    plane
                        .iter()
                        .map(|&sample| f64::from(sample) * scale)
                        .collect()
                })
                .collect(),
        }
    }
}

/// Decoded raw PCM: integer or floating point.
#[derive(Debug, Clone, PartialEq)]
pub enum PcmData {
    /// Native integer samples.
    Integer(PcmFrame),
    /// Normalized floating-point samples.
    Float(FloatFrame),
}

impl PcmData {
    /// Returns the number of channels.
    #[must_use]
    pub fn channels(&self) -> u8 {
        match self {
            Self::Integer(frame) => frame.channels,
            Self::Float(frame) => frame.channels,
        }
    }

    /// Returns the number of samples per channel.
    #[must_use]
    pub fn block_size(&self) -> usize {
        match self {
            Self::Integer(frame) => frame.block_size(),
            Self::Float(frame) => frame.block_size(),
        }
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Integer(frame) => frame.sample_rate,
            Self::Float(frame) => frame.sample_rate,
        }
    }
}

/// Decodes a buffer of interleaved little-endian PCM.
pub fn decode_pcm(bytes: &[u8], params: &PcmParams) -> Result<PcmData, PcmError> {
    params.validate()?;
    if bytes.len() > MAX_PCM_BYTES {
        return Err(PcmError::LimitExceeded);
    }
    let block_align = params.block_align();
    if !bytes.len().is_multiple_of(block_align) {
        return Err(PcmError::Truncated {
            length: bytes.len(),
            block_align,
        });
    }
    let samples = bytes.len() / block_align;
    let channels = usize::from(params.channels);
    match params.format {
        PcmFormat::S16Le | PcmFormat::S24Le | PcmFormat::S32Le => {
            let bytes_per_sample = params.format.bytes_per_sample();
            let mut planes = vec![Vec::with_capacity(samples); channels];
            for frame in bytes.chunks_exact(block_align) {
                for (channel, plane) in planes.iter_mut().enumerate() {
                    let offset = channel * bytes_per_sample;
                    let mut value = 0i32;
                    for byte in 0..bytes_per_sample {
                        value |= i32::from(frame[offset + byte]) << (8 * byte);
                    }
                    if bytes_per_sample < 4 {
                        let sign = 8 * bytes_per_sample as u32;
                        if value & (1 << (sign - 1)) != 0 {
                            value |= -1 << sign;
                        }
                    }
                    plane.push(value);
                }
            }
            Ok(PcmData::Integer(PcmFrame {
                sample_rate: params.sample_rate,
                channels: params.channels,
                bits_per_sample: params.format.bits_per_sample(),
                frame_number: 0,
                sample_number: 0,
                planes,
            }))
        }
        PcmFormat::F32Le | PcmFormat::F64Le => {
            let bytes_per_sample = params.format.bytes_per_sample();
            let mut planes = vec![Vec::with_capacity(samples); channels];
            for frame in bytes.chunks_exact(block_align) {
                for (channel, plane) in planes.iter_mut().enumerate() {
                    let offset = channel * bytes_per_sample;
                    let value = if bytes_per_sample == 4 {
                        f64::from(f32::from_le_bytes(
                            frame[offset..offset + 4].try_into().unwrap(),
                        ))
                    } else {
                        f64::from_le_bytes(frame[offset..offset + 8].try_into().unwrap())
                    };
                    plane.push(value);
                }
            }
            Ok(PcmData::Float(FloatFrame {
                sample_rate: params.sample_rate,
                channels: params.channels,
                sample_number: 0,
                planes,
            }))
        }
    }
}

/// Encodes decoded PCM into an interleaved little-endian buffer.
///
/// Integer data can only be encoded into an integer format of the same bit
/// depth; use [`PcmFrame::to_float`] or [`FloatFrame::to_integer`] to convert
/// first.  Floating-point data can be encoded into either float format.
pub fn encode_pcm(data: &PcmData, format: PcmFormat) -> Result<Vec<u8>, PcmError> {
    match data {
        PcmData::Integer(frame) => {
            if format.is_float() {
                return Err(PcmError::Conversion(
                    "integer PCM requires an integer format",
                ));
            }
            if frame.bits_per_sample != format.bits_per_sample() {
                return Err(PcmError::Conversion(
                    "integer PCM bit depth does not match the output format",
                ));
            }
            let channels = usize::from(frame.channels);
            if frame.planes.len() != channels {
                return Err(PcmError::Conversion("channel count mismatch"));
            }
            let samples = frame.block_size();
            if frame.planes.iter().any(|plane| plane.len() != samples) {
                return Err(PcmError::Conversion("plane lengths differ"));
            }
            let bytes_per_sample = format.bytes_per_sample();
            if samples
                .saturating_mul(channels)
                .saturating_mul(bytes_per_sample)
                > MAX_PCM_BYTES
            {
                return Err(PcmError::LimitExceeded);
            }
            let mut out = Vec::with_capacity(samples * channels * bytes_per_sample);
            for index in 0..samples {
                for plane in &frame.planes {
                    let sample = plane[index];
                    for byte in 0..bytes_per_sample {
                        out.push((sample >> (8 * byte)) as u8);
                    }
                }
            }
            Ok(out)
        }
        PcmData::Float(frame) => {
            if !format.is_float() {
                return Err(PcmError::Conversion(
                    "floating-point PCM requires a floating-point format",
                ));
            }
            let channels = usize::from(frame.channels);
            if frame.planes.len() != channels {
                return Err(PcmError::Conversion("channel count mismatch"));
            }
            let samples = frame.block_size();
            if frame.planes.iter().any(|plane| plane.len() != samples) {
                return Err(PcmError::Conversion("plane lengths differ"));
            }
            if samples
                .saturating_mul(channels)
                .saturating_mul(format.bytes_per_sample())
                > MAX_PCM_BYTES
            {
                return Err(PcmError::LimitExceeded);
            }
            let mut out = Vec::with_capacity(samples * channels * format.bytes_per_sample());
            for index in 0..samples {
                for plane in &frame.planes {
                    if format == PcmFormat::F32Le {
                        out.extend_from_slice(&(plane[index] as f32).to_le_bytes());
                    } else {
                        out.extend_from_slice(&plane[index].to_le_bytes());
                    }
                }
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(format: PcmFormat, channels: u8) -> PcmParams {
        PcmParams {
            format,
            sample_rate: 48_000,
            channels,
        }
    }

    #[test]
    fn decodes_s16le_byte_order_and_sign() {
        let bytes = [0x00, 0x80, 0xff, 0xff, 0x00, 0x00, 0x01, 0x00, 0xff, 0x7f];
        let data = decode_pcm(&bytes, &params(PcmFormat::S16Le, 1)).unwrap();
        match data {
            PcmData::Integer(frame) => {
                assert_eq!(frame.bits_per_sample, 16);
                assert_eq!(frame.planes[0], vec![-32768, -1, 0, 1, 32767]);
            }
            PcmData::Float(_) => panic!("expected integer PCM"),
        }
    }

    #[test]
    fn decodes_s24le_and_s32le() {
        let bytes = [0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00];
        let data = decode_pcm(&bytes, &params(PcmFormat::S24Le, 1)).unwrap();
        match data {
            PcmData::Integer(frame) => {
                assert_eq!(frame.planes[0], vec![-8_388_608, -1, 0]);
            }
            PcmData::Float(_) => panic!("expected integer PCM"),
        }
        let bytes = [0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0xff];
        let data = decode_pcm(&bytes, &params(PcmFormat::S32Le, 1)).unwrap();
        match data {
            PcmData::Integer(frame) => {
                assert_eq!(frame.planes[0], vec![i32::MIN, -1]);
            }
            PcmData::Float(_) => panic!("expected integer PCM"),
        }
    }

    #[test]
    fn decodes_interleaved_stereo() {
        let bytes = [0x01, 0x00, 0xff, 0xff, 0x02, 0x00, 0xfe, 0xff];
        let data = decode_pcm(&bytes, &params(PcmFormat::S16Le, 2)).unwrap();
        match data {
            PcmData::Integer(frame) => {
                assert_eq!(frame.planes[0], vec![1, 2]);
                assert_eq!(frame.planes[1], vec![-1, -2]);
            }
            PcmData::Float(_) => panic!("expected integer PCM"),
        }
    }

    #[test]
    fn decodes_and_encodes_floats() {
        let samples = [1.0f64, -0.5, 0.0];
        let mut bytes = Vec::new();
        for sample in samples {
            bytes.extend_from_slice(&(sample as f32).to_le_bytes());
        }
        let data = decode_pcm(&bytes, &params(PcmFormat::F32Le, 1)).unwrap();
        match &data {
            PcmData::Float(frame) => {
                assert_eq!(frame.planes[0], samples.to_vec());
            }
            PcmData::Integer(_) => panic!("expected float PCM"),
        }
        let encoded = encode_pcm(&data, PcmFormat::F32Le).unwrap();
        assert_eq!(encoded, bytes);
        let encoded64 = encode_pcm(&data, PcmFormat::F64Le).unwrap();
        assert_eq!(encoded64.len(), samples.len() * 8);
        let decoded64 = decode_pcm(&encoded64, &params(PcmFormat::F64Le, 1)).unwrap();
        assert_eq!(decoded64, data);
    }

    #[test]
    fn integer_float_conversions_round_trip_common_depths() {
        for bits in [16u8, 24, 32] {
            let max = (1i64 << (bits - 1)) - 1;
            let min = -(1i64 << (bits - 1));
            let frame = PcmFrame {
                sample_rate: 48_000,
                channels: 1,
                bits_per_sample: bits,
                frame_number: 0,
                sample_number: 0,
                planes: vec![vec![min as i32, -1, 0, 1, max as i32]],
            };
            let float = frame.to_float();
            let back = float.to_integer(bits).unwrap();
            assert_eq!(back.planes, frame.planes, "bits {bits}");
        }
    }

    #[test]
    fn rejects_truncated_and_invalid_input() {
        let error = decode_pcm(&[0u8; 3], &params(PcmFormat::S16Le, 2)).unwrap_err();
        assert!(matches!(error, PcmError::Truncated { .. }));
        let error = decode_pcm(
            &[],
            &PcmParams {
                format: PcmFormat::S16Le,
                sample_rate: 0,
                channels: 2,
            },
        )
        .unwrap_err();
        assert!(matches!(error, PcmError::InvalidParams(_)));
        let error = decode_pcm(
            &[],
            &PcmParams {
                format: PcmFormat::S16Le,
                sample_rate: 48_000,
                channels: 9,
            },
        )
        .unwrap_err();
        assert!(matches!(error, PcmError::InvalidParams(_)));
    }

    #[test]
    fn rejects_mismatched_encode_requests() {
        let frame = PcmFrame {
            sample_rate: 48_000,
            channels: 1,
            bits_per_sample: 16,
            frame_number: 0,
            sample_number: 0,
            planes: vec![vec![0]],
        };
        assert!(matches!(
            encode_pcm(&PcmData::Integer(frame.clone()), PcmFormat::S24Le),
            Err(PcmError::Conversion(_))
        ));
        assert!(matches!(
            encode_pcm(&PcmData::Integer(frame), PcmFormat::F32Le),
            Err(PcmError::Conversion(_))
        ));
    }

    #[test]
    fn empty_input_decodes_to_empty_planes() {
        let data = decode_pcm(&[], &params(PcmFormat::S24Le, 2)).unwrap();
        match data {
            PcmData::Integer(frame) => {
                assert_eq!(frame.block_size(), 0);
                assert_eq!(frame.planes.len(), 2);
            }
            PcmData::Float(_) => panic!("expected integer PCM"),
        }
    }
}
