// SPDX-License-Identifier: Apache-2.0
//! SILK decoder: packet header, frame loop, stereo, and resampling.
//!
//! The decoder implements the SILK layer of RFC 6716 for SILK-only Opus
//! frames.  It decodes 10, 20, 40, and 60 ms Opus frames at NB, MB, or WB,
//! mono or stereo, and produces PCM at 8, 12, 16, 24, or 48 kHz.  CELT,
//! hybrid mode, packet-loss concealment, and in-band FEC are not part of this
//! stage.

#[allow(dead_code)]
mod define;
mod nlsf;
mod pulses;
mod resample;
mod stereo;
mod structs;
mod synthesis;
#[allow(dead_code)]
mod tables;
#[allow(dead_code)]
mod tables_nlsf;

use std::vec::Vec;

use crate::opus::{OpusBandwidth, OpusError, RangeDecoder};

use resample::Resampler;
use synthesis::ChannelDecoder;
use tables::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};

/// Maximum number of SILK subframes in one frame.
pub const MAX_NB_SUBFR: usize = 4;
/// Maximum LPC order.
pub const MAX_LPC_ORDER: usize = 16;
/// Number of NLSF quantization levels around the prediction.
pub const NLSF_QUANT_MAX_AMPLITUDE: i8 = 4;

/// Conditional-coding context for a frame (RFC 6716, Section 4.2.7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CondCoding {
    /// First frame of a packet, or the preceding frame of the same type was
    /// not coded.
    Independent,
    /// Later frame with previous-frame prediction.
    Conditionally,
    /// Independent coding after an uncoded side frame; the LTP scaling
    /// parameter is omitted.
    IndependentNoLtpScaling,
}

/// Decoded SILK frame indices, exposed for tests and future concealment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SilkFrameIndices {
    /// Signal type (0 no voice activity, 1 unvoiced, 2 voiced).
    pub signal_type: i8,
    /// Quantization offset type.
    pub quant_offset_type: i8,
    /// Gain index per subframe.
    pub gain_indices: [i8; MAX_NB_SUBFR],
    /// Stage-1 and stage-2 NLSF indices.
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    /// NLSF interpolation coefficient in Q2.
    pub nlsf_interp_coef_q2: i8,
    /// Primary pitch lag offset in samples (without the minimum lag).
    pub lag_index: i16,
    /// Pitch contour index.
    pub contour_index: i8,
    /// LTP periodicity index.
    pub per_index: i8,
    /// LTP gain index per subframe.
    pub ltp_index: [i8; MAX_NB_SUBFR],
    /// LTP scaling index.
    pub ltp_scale_index: i8,
    /// LCG seed.
    pub seed: i8,
}

impl Default for SilkFrameIndices {
    fn default() -> Self {
        Self {
            signal_type: 0,
            quant_offset_type: 0,
            gain_indices: [0; MAX_NB_SUBFR],
            nlsf_indices: [0; MAX_LPC_ORDER + 1],
            nlsf_interp_coef_q2: 4,
            lag_index: 0,
            contour_index: 0,
            per_index: 0,
            ltp_index: [0; MAX_NB_SUBFR],
            ltp_scale_index: 0,
            seed: 0,
        }
    }
}

/// Stateful SILK decoder.
///
/// One instance decodes one logical stream.  The output rate and channel
/// count are fixed at construction; the internal SILK rate and frame size
/// follow the packet.  A change of the internal rate resets the synthesis
/// history and the resamplers.
#[derive(Debug, Clone)]
pub struct SilkDecoder {
    output_rate: u32,
    output_channels: usize,
    fs_khz: u8,
    channels: [ChannelDecoder; 2],
    resamplers: [Option<Resampler>; 2],
    previous_weights: [i32; 2],
    mid_history: [i16; 2],
    side_history: [i16; 2],
    previous_stereo: bool,
    side_uncoded: bool,
    last_range: u32,
    mid_frame: Vec<i16>,
    side_frame: Vec<i16>,
    left: Vec<i16>,
    right: Vec<i16>,
    scratch: Vec<i16>,
}

impl SilkDecoder {
    /// Creates a decoder for a fixed output sample rate and channel count.
    ///
    /// Supported rates are 8, 12, 16, 24, and 48 kHz.  `channels` must be 1
    /// or 2.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::Malformed`] for an unsupported rate or channel
    /// count.
    pub fn new(output_rate: u32, channels: usize) -> Result<Self, OpusError> {
        if !matches!(output_rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
            return Err(OpusError::Malformed(
                "SILK output rate must be 8, 12, 16, 24, or 48 kHz",
            ));
        }
        if channels != 1 && channels != 2 {
            return Err(OpusError::Malformed("SILK output must be mono or stereo"));
        }
        Ok(Self {
            output_rate,
            output_channels: channels,
            fs_khz: 8,
            channels: [ChannelDecoder::new(8, 4), ChannelDecoder::new(8, 4)],
            resamplers: [None, None],
            previous_weights: [0; 2],
            mid_history: [0; 2],
            side_history: [0; 2],
            previous_stereo: false,
            side_uncoded: true,
            last_range: 0,
            mid_frame: Vec::new(),
            side_frame: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// Returns the configured output sample rate in Hz.
    #[must_use]
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    /// Returns the configured output channel count.
    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Returns the range-decoder state after the most recently decoded frame.
    ///
    /// This is the value the reference encoder stores next to each packet in
    /// the RFC 6716 test vectors.
    #[must_use]
    pub fn last_range(&self) -> u32 {
        self.last_range
    }

    /// Decodes one SILK-only Opus frame.
    ///
    /// `bandwidth` and `duration_us` come from the packet's TOC byte and
    /// `stereo` from its stereo bit.  `pcm` receives interleaved signed
    /// 16-bit samples.  The return value is the number of samples per
    /// channel written.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an unsupported bandwidth or duration, a
    /// short output buffer, or a corrupt range-coded payload.
    pub fn decode_frame(
        &mut self,
        data: &[u8],
        bandwidth: OpusBandwidth,
        duration_us: u32,
        stereo: bool,
        pcm: &mut [i16],
    ) -> Result<usize, OpusError> {
        let fs_khz = match bandwidth {
            OpusBandwidth::Narrowband => 8,
            OpusBandwidth::Mediumband => 12,
            OpusBandwidth::Wideband => 16,
            OpusBandwidth::SuperWideband | OpusBandwidth::Fullband => {
                return Err(OpusError::Malformed(
                    "SILK does not support super-wideband or fullband",
                ));
            }
        };
        let (frames, nb_subfr) = match duration_us {
            10_000 => (1_usize, 2_usize),
            20_000 => (1, 4),
            40_000 => (2, 4),
            60_000 => (3, 4),
            _ => {
                return Err(OpusError::Malformed(
                    "SILK frame duration must be 10, 20, 40, or 60 ms",
                ));
            }
        };
        let internal_channels = usize::from(stereo) + 1;
        let total = duration_us as usize * self.output_rate as usize / 1_000_000;
        let needed = total * self.output_channels;
        if pcm.len() < needed {
            return Err(OpusError::OutputTooSmall {
                needed,
                available: pcm.len(),
            });
        }

        self.configure(fs_khz, nb_subfr);
        if stereo && !self.previous_stereo {
            // The prediction weights start at zero on a mono-to-stereo
            // transition, but the mid delay line stays continuous.
            self.previous_weights = [0; 2];
            self.side_history = [0; 2];
        }

        let mut decoder = RangeDecoder::new(data);
        let mut lbrr = [[false; 3]; 2];
        let mut lbrr_flag = [false; 2];
        // Per channel: one VAD bit per 20 ms interval, then that channel's
        // LBRR flag (RFC 6716, Section 4.2.3).  The side channel repeats the
        // whole group.
        for channel in 0..internal_channels {
            for vad in self.channels[channel].vad_flags.iter_mut().take(frames) {
                *vad = decoder.decode_bit_logp(1)?;
            }
            lbrr_flag[channel] = decoder.decode_bit_logp(1)?;
        }
        for channel in 0..internal_channels {
            if lbrr_flag[channel] {
                if frames == 1 {
                    lbrr[channel][0] = true;
                } else {
                    let icdf: &[u8] = if frames == 2 {
                        &SILK_LBRR_FLAGS_2_ICDF
                    } else {
                        &SILK_LBRR_FLAGS_3_ICDF
                    };
                    let symbol = decoder.decode_icdf(icdf, 8)? + 1;
                    for (frame, flag) in lbrr[channel].iter_mut().enumerate().take(frames) {
                        *flag = (symbol >> frame) & 1 != 0;
                    }
                }
            }
        }

        self.skip_lbrr(&mut decoder, &lbrr, frames, internal_channels)?;

        let frame_length = self.channels[0].frame_length();
        let frame_out = frame_length * self.output_rate as usize / (usize::from(fs_khz) * 1000);
        let input_rate = u32::from(fs_khz) * 1000;
        let mut written = 0;

        for frame in 0..frames {
            let mut mid_only = false;
            let mut weights = [0_i32; 2];
            if internal_channels == 2 {
                weights = stereo::decode_weights(&mut decoder)?;
                if !self.channels[1].vad_flags[frame] {
                    mid_only = stereo::decode_mid_only(&mut decoder)?;
                }
            }

            let mid_condition = if frame == 0 {
                CondCoding::Independent
            } else {
                CondCoding::Conditionally
            };
            self.channels[0].decode_frame(&mut decoder, frame, false, mid_condition, true)?;

            if internal_channels == 2 {
                if mid_only {
                    self.side_frame.clear();
                    self.side_frame.resize(frame_length, 0);
                    self.side_uncoded = true;
                } else {
                    if self.side_uncoded {
                        self.channels[1].reset();
                    }
                    let side_condition = if frame == 0 {
                        CondCoding::Independent
                    } else if self.side_uncoded {
                        CondCoding::IndependentNoLtpScaling
                    } else {
                        CondCoding::Conditionally
                    };
                    self.channels[1].decode_frame(
                        &mut decoder,
                        frame,
                        false,
                        side_condition,
                        true,
                    )?;
                    self.side_frame.clear();
                    self.side_frame.extend_from_slice(&self.channels[1].pcm);
                    self.side_uncoded = false;
                }
            }

            self.mid_frame.clear();
            self.mid_frame.extend_from_slice(&self.channels[0].pcm);

            if internal_channels == 2 {
                if self.output_channels == 2 {
                    self.left.resize(frame_length, 0);
                    self.right.resize(frame_length, 0);
                    stereo::mid_side_to_left_right(
                        &mut self.previous_weights,
                        &mut self.mid_history,
                        &mut self.side_history,
                        weights,
                        &self.mid_frame,
                        &self.side_frame,
                        fs_khz,
                        &mut self.left,
                        &mut self.right,
                    );
                } else {
                    self.previous_weights = weights;
                    self.update_mid_history();
                    self.left.clear();
                    self.left.extend_from_slice(&self.mid_frame);
                }
            } else {
                self.update_mid_history();
                self.left.clear();
                self.left.extend_from_slice(&self.mid_frame);
                if self.output_channels == 2 {
                    self.right.clear();
                    self.right.extend_from_slice(&self.mid_frame);
                }
            }

            self.write_channel(0, written, pcm, input_rate)?;
            if self.output_channels == 2 {
                self.write_channel(1, written, pcm, input_rate)?;
            }
            written += frame_out;
        }

        debug_assert_eq!(written, total);
        self.last_range = decoder.range();
        self.previous_stereo = stereo;
        if internal_channels == 1 {
            self.side_uncoded = true;
        }
        Ok(written)
    }

    fn configure(&mut self, fs_khz: u8, nb_subfr: usize) {
        if self.fs_khz != fs_khz {
            self.fs_khz = fs_khz;
            self.channels = [
                ChannelDecoder::new(fs_khz, nb_subfr),
                ChannelDecoder::new(fs_khz, nb_subfr),
            ];
            self.resamplers = [None, None];
            self.weights_reset();
            self.side_uncoded = true;
            self.previous_stereo = false;
        } else {
            self.channels[0].configure(fs_khz, nb_subfr);
            self.channels[1].configure(fs_khz, nb_subfr);
        }
    }

    fn weights_reset(&mut self) {
        self.previous_weights = [0; 2];
        self.mid_history = [0; 2];
        self.side_history = [0; 2];
    }

    fn update_mid_history(&mut self) {
        let length = self.mid_frame.len();
        if length >= 2 {
            self.mid_history = [self.mid_frame[length - 2], self.mid_frame[length - 1]];
        }
    }

    fn skip_lbrr(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        lbrr: &[[bool; 3]; 2],
        frames: usize,
        channels: usize,
    ) -> Result<(), OpusError> {
        for frame in 0..frames {
            for channel in 0..channels {
                if !lbrr[channel][frame] {
                    continue;
                }
                if channels == 2 && channel == 0 {
                    let _ = stereo::decode_weights(decoder)?;
                    if !lbrr[1][frame] {
                        let _ = stereo::decode_mid_only(decoder)?;
                    }
                }
                let condition = if frame > 0 && lbrr[channel][frame - 1] {
                    CondCoding::Conditionally
                } else {
                    CondCoding::Independent
                };
                self.channels[channel].decode_frame(decoder, frame, true, condition, false)?;
            }
        }
        Ok(())
    }

    fn write_channel(
        &mut self,
        channel: usize,
        base: usize,
        pcm: &mut [i16],
        input_rate: u32,
    ) -> Result<(), OpusError> {
        let block = if channel == 0 {
            &self.left
        } else {
            &self.right
        };
        let count = block.len() * self.output_rate as usize / input_rate as usize;
        self.scratch.resize(count, 0);
        let resampler = self.resamplers[channel]
            .get_or_insert_with(|| Resampler::new(input_rate, self.output_rate));
        resampler.process(block, &mut self.scratch[..count]);
        let channels = self.output_channels;
        for (index, &sample) in self.scratch[..count].iter().enumerate() {
            pcm[(base + index) * channels + channel] = sample;
        }
        Ok(())
    }
}
