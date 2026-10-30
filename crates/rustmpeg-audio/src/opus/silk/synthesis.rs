// SPDX-License-Identifier: Apache-2.0
//! SILK excitation reconstruction and LTP/LPC synthesis.
//!
//! The module follows RFC 6716, Sections 4.2.7.4 through 4.2.7.9.2.  Samples
//! are kept in Q23 (1.0 == 1 << 23) for the whole synthesis chain so the
//! excitation quantization offset stays exact even at the lowest gains.

use crate::opus::{OpusError, RangeDecoder};

use super::nlsf;
use super::pulses;
use super::structs::NLSFCodebook;
use super::tables::{
    SILK_GAIN_ICDF, SILK_LTP_GAIN_ICDF_PTRS, SILK_LTP_SCALES_TABLE_Q14, SILK_LTP_VQ_PTRS_Q7,
    SILK_PITCH_CONTOUR_NB_10_MS, SILK_PITCH_CONTOUR_NB_20_MS, SILK_PITCH_CONTOUR_WB_10_MS,
    SILK_PITCH_CONTOUR_WB_20_MS, SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF,
};
use super::{CondCoding, MAX_LPC_ORDER, MAX_NB_SUBFR, NLSF_QUANT_MAX_AMPLITUDE, SilkFrameIndices};

/// One in Q23.
const ONE_Q23: i32 = 1 << 23;

/// Samples retained between frames: maximum pitch lag (18 ms at 16 kHz) plus
/// the LPC order and the two-sample LTP width, rounded up.
const HISTORY_SAMPLES: usize = 320;

/// Quantization offset in Q23 per signal type and offset type (RFC 6716,
/// Table 53).  Index 0 selects inactive/unvoiced, index 1 voiced.
const QUANT_OFFSET_Q23: [[i32; 2]; 2] = [[25, 60], [8, 25]];

/// Appends samples by absolute index and retains a bounded window.
#[derive(Debug, Clone)]
struct History {
    samples: Vec<i32>,
    base: i64,
}

impl History {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            base: 0,
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.base = 0;
    }

    fn end(&self) -> i64 {
        self.base + self.samples.len() as i64
    }

    fn extend(&mut self, count: usize) {
        self.samples.resize(self.samples.len() + count, 0);
    }

    fn get(&self, index: i64) -> i32 {
        if index < self.base {
            return 0;
        }
        self.samples
            .get((index - self.base) as usize)
            .copied()
            .unwrap_or(0)
    }

    fn set(&mut self, index: i64, value: i32) {
        let position = (index - self.base) as usize;
        debug_assert!(position < self.samples.len());
        self.samples[position] = value;
    }

    fn retain(&mut self, count: usize) {
        if self.samples.len() > count {
            let drop = self.samples.len() - count;
            self.samples.drain(..drop);
            self.base += drop as i64;
        }
    }
}

/// One channel's persistent decoder state.
#[derive(Debug, Clone)]
pub(crate) struct ChannelDecoder {
    fs_khz: u8,
    nb_subfr: usize,
    /// Previous signal type for conditional pitch coding.
    ec_prev_signal_type: i8,
    /// Previous primary lag offset for conditional pitch coding.
    ec_prev_lag_index: i16,
    /// Per-20-ms VAD flags of the current Opus frame.
    pub vad_flags: [bool; 3],
    nlsf_cb: &'static NLSFCodebook,
    pitch_lag_low_bits_icdf: &'static [u8],
    pitch_contour_icdf: &'static [u8],
    out: History,
    lpc: History,
    residual: History,
    prev_nlsf: [i16; MAX_LPC_ORDER],
    prev_log_gain: i32,
    first_frame: bool,
    /// PCM of the most recently decoded frame, at the internal rate.
    pub pcm: Vec<i16>,
}

impl ChannelDecoder {
    pub(crate) fn new(fs_khz: u8, nb_subfr: usize) -> Self {
        let (nlsf_cb, pitch_lag_low_bits_icdf) = match fs_khz {
            8 => (
                &super::tables_nlsf::SILK_NLSF_CB_NB_MB,
                &SILK_UNIFORM4_ICDF[..],
            ),
            12 => (
                &super::tables_nlsf::SILK_NLSF_CB_NB_MB,
                &SILK_UNIFORM6_ICDF[..],
            ),
            _ => (
                &super::tables_nlsf::SILK_NLSF_CB_WB,
                &SILK_UNIFORM8_ICDF[..],
            ),
        };
        let pitch_contour_icdf = pitch_contour_icdf(fs_khz, nb_subfr);
        let mut decoder = Self {
            fs_khz,
            nb_subfr,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [false; 3],
            nlsf_cb,
            pitch_lag_low_bits_icdf,
            pitch_contour_icdf,
            out: History::new(),
            lpc: History::new(),
            residual: History::new(),
            prev_nlsf: [0; MAX_LPC_ORDER],
            prev_log_gain: 10,
            first_frame: true,
            pcm: Vec::new(),
        };
        decoder.reset();
        decoder
    }

    /// Changes the internal configuration.  A change of the internal rate
    /// resets the synthesis history; a change of frame size alone does not,
    /// mirroring the reference decoder.
    pub(crate) fn configure(&mut self, fs_khz: u8, nb_subfr: usize) {
        if self.fs_khz != fs_khz {
            let fresh = Self::new(fs_khz, nb_subfr);
            *self = fresh;
            return;
        }
        self.nb_subfr = nb_subfr;
        self.pitch_contour_icdf = pitch_contour_icdf(fs_khz, nb_subfr);
    }

    /// Clears all synthesis and prediction state.
    pub(crate) fn reset(&mut self) {
        self.out.clear();
        self.lpc.clear();
        self.residual.clear();
        self.prev_nlsf = [0; MAX_LPC_ORDER];
        self.prev_log_gain = 10;
        self.ec_prev_signal_type = 0;
        self.ec_prev_lag_index = 0;
        self.first_frame = true;
        self.pcm.clear();
    }

    /// Number of samples in one SILK frame at the internal rate.
    pub(crate) fn frame_length(&self) -> usize {
        usize::from(self.fs_khz) * 5 * self.nb_subfr
    }

    /// Decodes one SILK frame: indices, excitation, and (for regular frames)
    /// PCM synthesis.  LBRR frames are parsed and discarded because in-band
    /// FEC is not implemented yet.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_frame(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        frame_index: usize,
        decode_lbrr: bool,
        cond_coding: CondCoding,
        synthesize: bool,
    ) -> Result<(), OpusError> {
        let indices = self.decode_indices(decoder, frame_index, decode_lbrr, cond_coding)?;
        let frame_length = self.frame_length();
        let pulses = pulses::decode_pulses(
            decoder,
            frame_length,
            indices.signal_type,
            indices.quant_offset_type,
        )?;
        if synthesize {
            self.synthesize(&indices, &pulses, cond_coding);
        }
        Ok(())
    }

    /// Decodes the frame's side information (RFC 6716, Sections 4.2.7.3
    /// through 4.2.7.8).
    fn decode_indices(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        frame_index: usize,
        decode_lbrr: bool,
        cond_coding: CondCoding,
    ) -> Result<SilkFrameIndices, OpusError> {
        let mut indices = SilkFrameIndices::default();

        let symbol = if decode_lbrr || self.vad_flags[frame_index.min(2)] {
            decoder.decode_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8)? + 2
        } else {
            decoder.decode_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8)?
        };
        indices.signal_type = (symbol >> 1) as i8;
        indices.quant_offset_type = (symbol & 1) as i8;
        if indices.signal_type > 2 {
            return Err(OpusError::Malformed("SILK signal type out of range"));
        }

        if cond_coding == CondCoding::Conditionally {
            indices.gain_indices[0] =
                decoder.decode_icdf(&super::tables::SILK_DELTA_GAIN_ICDF, 8)? as i8;
        } else {
            indices.gain_indices[0] =
                (decoder.decode_icdf(&SILK_GAIN_ICDF[indices.signal_type as usize], 8)? << 3) as i8;
            indices.gain_indices[0] +=
                decoder.decode_icdf(&super::tables::SILK_UNIFORM8_ICDF, 8)? as i8;
        }
        for gain in indices.gain_indices.iter_mut().take(self.nb_subfr).skip(1) {
            *gain = decoder.decode_icdf(&super::tables::SILK_DELTA_GAIN_ICDF, 8)? as i8;
        }

        let codebook = self.nlsf_cb;
        let stage1_offset = ((indices.signal_type >> 1) as usize) * (codebook.n_vectors as usize);
        indices.nlsf_indices[0] =
            decoder.decode_icdf(&codebook.cb1_icdf[stage1_offset..], 8)? as i8;

        let mut ec_ix = [0_i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0_u8; MAX_LPC_ORDER];
        nlsf::unpack(
            &mut ec_ix,
            &mut pred_q8,
            codebook,
            indices.nlsf_indices[0] as usize,
        );

        for index in 0..(codebook.order as usize) {
            let mut value =
                decoder.decode_icdf(&codebook.ec_icdf[ec_ix[index] as usize..], 8)? as i8;
            if value == 0 {
                value -= decoder.decode_icdf(&super::tables::SILK_NLSF_EXT_ICDF, 8)? as i8;
            } else if value == 2 * NLSF_QUANT_MAX_AMPLITUDE {
                value += decoder.decode_icdf(&super::tables::SILK_NLSF_EXT_ICDF, 8)? as i8;
            }
            indices.nlsf_indices[index + 1] = value - NLSF_QUANT_MAX_AMPLITUDE;
        }

        indices.nlsf_interp_coef_q2 = if self.nb_subfr == 4 {
            decoder.decode_icdf(&super::tables::SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8)? as i8
        } else {
            4
        };

        if indices.signal_type == 2 {
            let mut absolute = true;
            if cond_coding == CondCoding::Conditionally && self.ec_prev_signal_type == 2 {
                let delta = decoder.decode_icdf(&SILK_PITCH_DELTA_ICDF, 8)? as i16;
                if delta > 0 {
                    indices.lag_index = self.ec_prev_lag_index + delta - 9;
                    absolute = false;
                }
            }
            if absolute {
                indices.lag_index = (decoder.decode_icdf(&SILK_PITCH_LAG_ICDF, 8)? as i16)
                    * (i16::from(self.fs_khz) >> 1);
                indices.lag_index += decoder.decode_icdf(self.pitch_lag_low_bits_icdf, 8)? as i16;
            }
            self.ec_prev_lag_index = indices.lag_index;

            indices.contour_index = decoder.decode_icdf(self.pitch_contour_icdf, 8)? as i8;
            indices.per_index =
                decoder.decode_icdf(&super::tables::SILK_LTP_PER_INDEX_ICDF, 8)? as i8;
            if indices.per_index > 2 {
                return Err(OpusError::Malformed(
                    "SILK LTP periodicity index out of range",
                ));
            }
            for gain in indices.ltp_index.iter_mut().take(self.nb_subfr) {
                *gain = decoder
                    .decode_icdf(SILK_LTP_GAIN_ICDF_PTRS[indices.per_index as usize], 8)?
                    as i8;
            }
            if cond_coding == CondCoding::Independent {
                indices.ltp_scale_index =
                    decoder.decode_icdf(&super::tables::SILK_LTPSCALE_ICDF, 8)? as i8;
            }
        }
        self.ec_prev_signal_type = indices.signal_type;
        indices.seed = decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8)? as i8;
        Ok(indices)
    }

    /// Dequantizes the subframe gains (RFC 6716, Section 4.2.7.4).
    fn dequantize_gains(
        &mut self,
        indices: &SilkFrameIndices,
        cond_coding: CondCoding,
    ) -> [i32; MAX_NB_SUBFR] {
        let mut gains = [0_i32; MAX_NB_SUBFR];
        let mut log_gain = self.prev_log_gain;
        for (index, gain) in gains.iter_mut().enumerate().take(self.nb_subfr) {
            let value = i32::from(indices.gain_indices[index]);
            if index == 0 && cond_coding != CondCoding::Conditionally {
                log_gain = value.max(log_gain - 16);
            } else {
                log_gain = (log_gain + value - 4).max(2 * value - 16);
            }
            log_gain = log_gain.clamp(0, 63);
            *gain = log2lin(((1_907_825 * log_gain) >> 16) + 2090);
        }
        self.prev_log_gain = log_gain;
        gains
    }

    /// Computes the per-subframe pitch lags (RFC 6716, Section 4.2.7.6.1).
    fn pitch_lags(&self, indices: &SilkFrameIndices) -> [i16; MAX_NB_SUBFR] {
        let min_lag = i32::from(self.fs_khz) * 2;
        let max_lag = i32::from(self.fs_khz) * 18;
        let primary = min_lag + i32::from(indices.lag_index);
        let contour = self.contour(indices.contour_index as usize);
        let mut lags = [0_i16; MAX_NB_SUBFR];
        for (index, lag) in lags.iter_mut().enumerate().take(self.nb_subfr) {
            *lag = (primary + i32::from(contour[index])).clamp(min_lag, max_lag) as i16;
        }
        lags
    }

    fn contour(&self, index: usize) -> &'static [i8] {
        if self.fs_khz == 8 {
            if self.nb_subfr == 2 {
                &SILK_PITCH_CONTOUR_NB_10_MS[index]
            } else {
                &SILK_PITCH_CONTOUR_NB_20_MS[index]
            }
        } else if self.nb_subfr == 2 {
            &SILK_PITCH_CONTOUR_WB_10_MS[index]
        } else {
            &SILK_PITCH_CONTOUR_WB_20_MS[index]
        }
    }

    /// Synthesizes one frame from its indices and excitation pulses.
    fn synthesize(&mut self, indices: &SilkFrameIndices, pulses: &[i16], cond_coding: CondCoding) {
        let order = if self.fs_khz == 16 { 16 } else { 10 };
        let frame_length = self.frame_length();
        let subframe_length = frame_length / self.nb_subfr;
        let start = self.out.end();
        self.out.extend(frame_length);
        self.lpc.extend(frame_length);
        self.residual.extend(frame_length);

        let current_nlsf = nlsf::decode(&indices.nlsf_indices, self.nlsf_cb);
        let interpolate =
            self.nb_subfr == 4 && !self.first_frame && indices.nlsf_interp_coef_q2 < 4;
        let first_nlsf = if interpolate {
            nlsf::interpolate(
                &self.prev_nlsf,
                &current_nlsf,
                indices.nlsf_interp_coef_q2,
                order,
            )
        } else {
            current_nlsf
        };
        let first_lpc = nlsf::to_lpc(&first_nlsf, order);
        let second_lpc = if interpolate {
            nlsf::to_lpc(&current_nlsf, order)
        } else {
            first_lpc
        };
        self.prev_nlsf = current_nlsf;

        let gains = self.dequantize_gains(indices, cond_coding);
        let excitation = reconstruct_excitation(indices, pulses);
        let voiced = indices.signal_type == 2;
        let pitch_lags = if voiced {
            self.pitch_lags(indices)
        } else {
            [0_i16; MAX_NB_SUBFR]
        };
        let ltp_scale =
            i32::from(SILK_LTP_SCALES_TABLE_Q14[indices.ltp_scale_index.clamp(0, 2) as usize]);

        self.pcm.resize(frame_length, 0);
        for subframe in 0..self.nb_subfr {
            let coefficients = if interpolate && subframe < 2 {
                &first_lpc
            } else {
                &second_lpc
            };
            let gain = gains[subframe];
            let subframe_start = start + (subframe * subframe_length) as i64;

            if voiced {
                let lag = i64::from(pitch_lags[subframe]);
                let (out_end, scale) = if interpolate && subframe >= 2 {
                    (start + ((subframe - 2) * subframe_length) as i64, 16384_i64)
                } else {
                    (start, i64::from(ltp_scale))
                };
                self.rewhiten(
                    subframe_start - lag - 2,
                    subframe_start,
                    out_end,
                    scale,
                    gain,
                    coefficients,
                    order,
                );
            }

            for offset in 0..subframe_length {
                let position = subframe_start + offset as i64;
                let mut residual = i64::from(excitation[(position - start) as usize]);
                if voiced {
                    let lag = i64::from(pitch_lags[subframe]);
                    let taps = SILK_LTP_VQ_PTRS_Q7[indices.per_index as usize]
                        [indices.ltp_index[subframe] as usize];
                    for (tap, &coefficient) in taps.iter().enumerate() {
                        residual += (i64::from(coefficient)
                            * i64::from(self.residual.get(position - lag + 2 - tap as i64)))
                            >> 7;
                    }
                }
                let residual = clamp_i32(residual);
                self.residual.set(position, residual);

                let mut value = (i64::from(gain) * i64::from(residual)) >> 16;
                for (index, &coefficient) in coefficients.iter().enumerate().take(order) {
                    value += (i64::from(coefficient)
                        * i64::from(self.lpc.get(position - index as i64 - 1)))
                        >> 12;
                }
                let value = clamp_i32(value);
                self.lpc.set(position, value);
                let clamped = value.clamp(-ONE_Q23, ONE_Q23);
                self.out.set(position, clamped);
                self.pcm[subframe * subframe_length + offset] =
                    (clamped >> 8).clamp(-32768, 32767) as i16;
            }
        }

        self.first_frame = false;
        self.out.retain(HISTORY_SAMPLES);
        self.lpc.retain(HISTORY_SAMPLES);
        self.residual.retain(HISTORY_SAMPLES);
    }

    /// Reconstructs the gain-normalized LPC residual for `[from, to)` by
    /// rewhitening the output or the unclamped LPC signal (RFC 6716, Section
    /// 4.2.7.9.1).
    #[allow(clippy::too_many_arguments)]
    fn rewhiten(
        &mut self,
        from: i64,
        to: i64,
        out_end: i64,
        scale_q14: i64,
        gain_q16: i32,
        coefficients: &[i16; MAX_LPC_ORDER],
        order: usize,
    ) {
        let gain = i64::from(gain_q16);
        for position in from..to {
            if position < 0 {
                continue;
            }
            let use_output = position < out_end;
            let source = if use_output { &self.out } else { &self.lpc };
            let mut whitened = i64::from(source.get(position));
            for (index, &coefficient) in coefficients.iter().enumerate().take(order) {
                whitened -= (i64::from(coefficient)
                    * i64::from(source.get(position - index as i64 - 1)))
                    >> 12;
            }
            let scaled = if use_output {
                (4 * scale_q14 * whitened + gain / 2) / gain
            } else {
                (65536 * whitened + gain / 2) / gain
            };
            self.residual.set(position, clamp_i32(scaled));
        }
    }
}

/// Reconstructs the excitation with the LCG and quantization offset
/// (RFC 6716, Section 4.2.7.8.6).
fn reconstruct_excitation(indices: &SilkFrameIndices, pulses: &[i16]) -> Vec<i32> {
    let offset =
        QUANT_OFFSET_Q23[(indices.signal_type >> 1) as usize][indices.quant_offset_type as usize];
    let mut seed = u32::from(indices.seed as u8);
    let mut excitation = Vec::with_capacity(pulses.len());
    for &pulse in pulses {
        let raw = i32::from(pulse);
        let mut value = (raw << 8) + offset;
        if raw > 0 {
            value -= 20;
        } else if raw < 0 {
            value += 20;
        }
        seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
        if seed & 0x8000_0000 != 0 {
            value = -value;
        }
        excitation.push(value);
        seed = seed.wrapping_add(raw as u32);
    }
    excitation
}

/// `silk_log2lin`: approximates 2**(value / 128) (RFC 6716, Section 4.2.7.4).
fn log2lin(value_q7: i32) -> i32 {
    if value_q7 < 0 {
        return 0;
    }
    if value_q7 >= 3967 {
        return i32::MAX;
    }
    let integer = 1_i64 << (value_q7 >> 7);
    let fraction = i64::from(value_q7 & 127);
    let correction = ((-174 * fraction * (128 - fraction)) >> 16) + fraction;
    let add = if value_q7 < 2048 {
        (integer * correction) >> 7
    } else {
        (integer >> 7) * correction
    };
    (integer + add) as i32
}

fn clamp_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn pitch_contour_icdf(fs_khz: u8, nb_subfr: usize) -> &'static [u8] {
    use super::tables::{
        SILK_PITCH_CONTOUR_10_MS_ICDF, SILK_PITCH_CONTOUR_10_MS_NB_ICDF, SILK_PITCH_CONTOUR_ICDF,
        SILK_PITCH_CONTOUR_NB_ICDF,
    };
    if fs_khz == 8 {
        if nb_subfr == 2 {
            &SILK_PITCH_CONTOUR_10_MS_NB_ICDF
        } else {
            &SILK_PITCH_CONTOUR_NB_ICDF
        }
    } else if nb_subfr == 2 {
        &SILK_PITCH_CONTOUR_10_MS_ICDF
    } else {
        &SILK_PITCH_CONTOUR_ICDF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2lin_matches_known_gain_values() {
        // Reference values from RFC 6716, Section 4.2.7.4.
        assert_eq!(log2lin(2090), 81_920);
        assert_eq!(log2lin(((1_907_825 * 63) >> 16) + 2090), 1_686_110_208);
    }

    #[test]
    fn excitation_seed_is_deterministic() {
        let indices = SilkFrameIndices {
            signal_type: 2,
            quant_offset_type: 1,
            seed: 3,
            ..SilkFrameIndices::default()
        };
        let pulses = [1_i16, -2, 0, 3];
        let first = reconstruct_excitation(&indices, &pulses);
        let second = reconstruct_excitation(&indices, &pulses);
        assert_eq!(first, second);
        assert_eq!(first.len(), pulses.len());
        assert!(first.iter().any(|value| *value != 0));
    }
}
