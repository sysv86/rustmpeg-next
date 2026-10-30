// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
// Copyright 2026 The rustmpeg-next Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AV2 encoder rate control (`CQP`, `VBR`, `CBR`) and Lagrangian multiplier
//! selection.
//!
//! The controller is encoder-only: it never appears in the bitstream. It uses a
//! frame budget in bits, a dead-banded logarithmic error term, and a bounded
//! per-frame quantizer step. The dead band and the small gain are what keep the
//! quantizer from oscillating between consecutive frames; a fixed proportional
//! gain on the raw bit ratio (the previous behaviour) overshoots because one
//! frame's coded size is a poor predictor of the next for real content.
//!
//! Quantizer changes produced by this module are what the segmentation
//! `SEG_LVL_ALT_Q` and per-superblock `delta_q` layers start from; the encoder
//! reports the effective average quantizer back through `end_frame()`, so the
//! controller sees the quantizer AQ actually used rather than only the frame
//! base.

use crate::av2::headers::FrameType;
use crate::av2::intops::iclip;
use crate::av2::quantizer::dq_lookup;

/// Rate control operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant quantizer index (`base_qidx` fixed, with the existing keyframe
    /// and inter-frame offsets). Bitstream-compatible with the pre-AQ encoder.
    ConstantQp { base_qidx: u16 },
    /// Variable bitrate targeting `target_kbps` kilobits per second over the sequence.
    Vbr { target_kbps: u32, max_qidx: u16 },
    /// Constant bitrate targeting `target_kbps` with a virtual leaky-bucket buffer of `buffer_ms`.
    Cbr { target_kbps: u32, buffer_ms: u32 },
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::ConstantQp { base_qidx: 100 }
    }
}

impl RateControlMode {
    /// Target bits per frame for the bitrate modes, or `None` for `ConstantQp`.
    pub fn target_bits_per_frame(&self, fps: f64) -> Option<f64> {
        match *self {
            Self::ConstantQp { .. } => None,
            Self::Vbr { target_kbps, .. } | Self::Cbr { target_kbps, .. } => {
                Some((target_kbps as f64) * 1000.0 / fps.max(1.0))
            }
        }
    }
}

/// Stateful frame-level rate controller and Lagrangian multiplier (`lambda`) calculator.
#[derive(Debug, Clone)]
pub struct RateController {
    pub mode: RateControlMode,
    pub fps_num: u32,
    pub fps_den: u32,
    pub width: usize,
    pub height: usize,
    pub keyint: usize,
    /// Virtual buffer fullness in bits (for VBR/CBR).
    pub buffer_fullness_bits: i64,
    /// Virtual buffer capacity in bits.
    pub buffer_capacity_bits: i64,
    /// Exponentially smoothed bits per frame for keyframes and inter frames.
    pub avg_frame_bits_key: f64,
    pub avg_frame_bits_inter: f64,
    pub current_qidx: u16,
    pub frames_coded: u64,
    /// Bits emitted by the most recently completed frame.
    pub last_frame_bits: i64,
    /// Quantizer used by the most recently completed frame (effective average,
    /// including segmentation and per-superblock delta-Q).
    pub last_qidx: u16,
    /// Consecutive frames whose bit ratio stayed inside the dead band.
    pub stable_frames: u32,
}

impl RateController {
    /// Creates a new `RateController` for the given stream geometry and framerate.
    pub fn new(
        mode: RateControlMode,
        width: usize,
        height: usize,
        fps_num: u32,
        fps_den: u32,
        keyint: usize,
    ) -> Self {
        let fps = (fps_num.max(1) as f64) / (fps_den.max(1) as f64);
        let (target_bps, buf_ms, init_q) = match mode {
            RateControlMode::ConstantQp { base_qidx } => (0.0, 1000, base_qidx),
            RateControlMode::Vbr { target_kbps, .. } => (target_kbps as f64 * 1000.0, 2000, 110),
            RateControlMode::Cbr {
                target_kbps,
                buffer_ms,
            } => (target_kbps as f64 * 1000.0, buffer_ms.max(200), 110),
        };
        let avg_bits = (target_bps / fps.max(1.0)).max(1024.0);
        let cap_bits = ((target_bps * (buf_ms as f64)) / 1000.0) as i64;

        Self {
            mode,
            fps_num: fps_num.max(1),
            fps_den: fps_den.max(1),
            width,
            height,
            keyint: keyint.max(1),
            buffer_fullness_bits: cap_bits / 2,
            buffer_capacity_bits: cap_bits.max(16384),
            avg_frame_bits_key: avg_bits * 2.5,
            avg_frame_bits_inter: avg_bits * 0.85,
            current_qidx: init_q,
            frames_coded: 0,
            last_frame_bits: 0,
            last_qidx: init_q,
            stable_frames: 0,
        }
    }

    /// Target bits for the next frame after accounting for the virtual buffer.
    ///
    /// A leaky-bucket target is `target_bits_per_frame` corrected by a fraction
    /// of the buffer error. The correction is deliberately bounded so one
    /// oversized frame cannot swing the quantizer by tens of steps.
    pub fn frame_budget_bits(&self, frame_type: FrameType) -> Option<f64> {
        let fps = (self.fps_num as f64) / (self.fps_den as f64);
        let base = self.mode.target_bits_per_frame(fps)?;
        let cap = self.buffer_capacity_bits.max(1) as f64;
        let fill = self.buffer_fullness_bits as f64 / cap;
        let correction = ((fill - 0.5) * 0.25).clamp(-0.25, 0.25);
        let type_scale = if matches!(frame_type, FrameType::Key | FrameType::Intra) {
            2.0
        } else {
            1.0
        };
        Some((base * (1.0 - correction) * type_scale).max(256.0))
    }

    /// Selects the frame base quantizer index `qidx` (`0..=255`) and Lagrangian
    /// multipliers `(qidx, lambda_sad, lambda_rdoq)` before encoding a frame of
    /// `frame_type`.
    pub fn begin_frame(&self, frame_type: FrameType, bit_depth: u8) -> (u16, u64, u64) {
        let qidx = match self.mode {
            // ConstantQp is intentionally unchanged so enabling the rate
            // controller refactor cannot alter existing streams.
            RateControlMode::ConstantQp { base_qidx } => match frame_type {
                FrameType::Key | FrameType::Intra => base_qidx.saturating_sub(8),
                FrameType::Inter | FrameType::Switch => (base_qidx + 4).min(255),
            },
            RateControlMode::Vbr { max_qidx, .. } => {
                let fill =
                    self.buffer_fullness_bits as f64 / self.buffer_capacity_bits.max(1) as f64;
                // Small buffer correction + keyframe quality preference.
                let buffer_step = (-(fill - 0.5) * 8.0).round() as i32;
                let type_off = if matches!(frame_type, FrameType::Key | FrameType::Intra) {
                    -6
                } else {
                    0
                };
                iclip(
                    self.current_qidx as i32 + buffer_step + type_off,
                    8,
                    max_qidx as i32,
                ) as u16
            }
            RateControlMode::Cbr { .. } => {
                let fill =
                    self.buffer_fullness_bits as f64 / self.buffer_capacity_bits.max(1) as f64;
                let buffer_step = (-(fill - 0.5) * 10.0).round() as i32;
                let type_off = if matches!(frame_type, FrameType::Key | FrameType::Intra) {
                    -4
                } else {
                    0
                };
                iclip(self.current_qidx as i32 + buffer_step + type_off, 12, 248) as u16
            }
        };

        let shift = (bit_depth - 8) as u32;
        let q_step = dq_lookup(qidx as i32).max(4) as u64;
        let lambda_sad = ((q_step >> 2).max(1)) << shift;
        let lambda_rdoq = ((q_step * q_step) >> 4).max(16);
        (qidx, lambda_sad, lambda_rdoq)
    }

    /// Updates the rate controller state with the number of `bytes_emitted` by
    /// the frame and the effective quantizer `qidx_used` (which may include
    /// segmentation ALT_Q and per-SB delta-Q).
    pub fn end_frame(&mut self, frame_type: FrameType, qidx_used: u16, bytes_emitted: usize) {
        let bits_emitted = (bytes_emitted as i64) * 8;
        self.frames_coded += 1;
        self.last_frame_bits = bits_emitted;
        self.last_qidx = qidx_used;

        match self.mode {
            RateControlMode::ConstantQp { .. } => {
                // No feedback: the bitstream must stay identical to the
                // pre-rate-control encoder for constant-QP operation.
                self.current_qidx = qidx_used;
            }
            RateControlMode::Vbr { target_kbps, .. } | RateControlMode::Cbr { target_kbps, .. } => {
                let fps = (self.fps_num as f64) / (self.fps_den as f64);
                let target_bits_per_frame =
                    ((target_kbps as f64) * 1000.0 / fps.max(1.0)).round() as i64;
                self.buffer_fullness_bits = (self.buffer_fullness_bits + bits_emitted
                    - target_bits_per_frame)
                    .clamp(0, self.buffer_capacity_bits * 2);

                if matches!(frame_type, FrameType::Key | FrameType::Intra) {
                    self.avg_frame_bits_key =
                        0.7 * self.avg_frame_bits_key + 0.3 * (bits_emitted as f64);
                } else {
                    self.avg_frame_bits_inter =
                        0.8 * self.avg_frame_bits_inter + 0.2 * (bits_emitted as f64);
                }

                // Logarithmic error with a dead band: the coded size of one
                // frame is noisy, so correcting small errors only injects
                // quantizer oscillation. The gain is 4 quantizer steps per
                // octave of bitrate error and is clamped to 4 steps per frame.
                let target = (target_bits_per_frame.max(256)) as f64;
                let ratio = bits_emitted as f64 / target;
                if (0.85..=1.18).contains(&ratio) {
                    self.stable_frames = self.stable_frames.saturating_add(1);
                } else {
                    self.stable_frames = 0;
                    let step = ((ratio.log2() * 4.0).round() as i32).clamp(-4, 4);
                    let qmin = if matches!(self.mode, RateControlMode::Cbr { .. }) {
                        12
                    } else {
                        8
                    };
                    self.current_qidx = iclip(qidx_used as i32 + step, qmin, 250) as u16;
                }
            }
        }
    }

    /// Requested quantizer step per bitrate octave, exposed for the unit tests
    /// and for callers that log rate-control tuning.
    pub const QIDX_PER_OCTAVE: f64 = 4.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_qp_is_unchanged() {
        let rc = RateController::new(
            RateControlMode::ConstantQp { base_qidx: 100 },
            64,
            64,
            30,
            1,
            30,
        );
        assert_eq!(rc.begin_frame(FrameType::Key, 8).0, 92);
        assert_eq!(rc.begin_frame(FrameType::Inter, 8).0, 104);
    }

    #[test]
    fn vbr_quantizer_is_stable_around_the_target() {
        let mut rc = RateController::new(
            RateControlMode::Vbr {
                target_kbps: 300,
                max_qidx: 220,
            },
            320,
            192,
            30,
            1,
            30,
        );
        let target = 300.0 * 1000.0 / 30.0;
        let mut q = rc.begin_frame(FrameType::Inter, 8).0;
        let mut history = Vec::new();
        // Alternate a content-sized frame around the target; a stable
        // controller must not walk the quantizer to a rail.
        for i in 0..120 {
            let bytes = if i % 3 == 0 {
                (target * 1.35 / 8.0) as usize
            } else if i % 3 == 1 {
                (target * 0.75 / 8.0) as usize
            } else {
                (target / 8.0) as usize
            };
            rc.end_frame(FrameType::Inter, q, bytes);
            q = rc.begin_frame(FrameType::Inter, 8).0;
            history.push(q);
        }
        let min = *history.iter().min().unwrap();
        let max = *history.iter().max().unwrap();
        assert!(
            max - min <= 12,
            "quantizer oscillated over {min}..{max}: {history:?}"
        );
        // The last third is within a few steps of the running average.
        let tail = &history[80..];
        let mean = tail.iter().map(|&v| v as i32).sum::<i32>() / tail.len() as i32;
        for &v in tail {
            assert!((v as i32 - mean).abs() <= 8, "{history:?}");
        }
    }

    #[test]
    fn cbr_buffer_feedback_is_bounded() {
        let mut rc = RateController::new(
            RateControlMode::Cbr {
                target_kbps: 400,
                buffer_ms: 1000,
            },
            320,
            192,
            30,
            1,
            30,
        );
        let mut prev = rc.begin_frame(FrameType::Inter, 8).0;
        for _ in 0..30 {
            // A grossly oversized frame must move the quantizer by at most a
            // few steps, not by the full buffer error.
            rc.end_frame(FrameType::Inter, prev, 60_000);
            let q = rc.begin_frame(FrameType::Inter, 8).0;
            assert!((q as i32 - prev as i32).abs() <= 12, "{prev} -> {q}");
            prev = q;
        }
    }

    #[test]
    fn frame_budget_tracks_target_and_keyframes() {
        let rc = RateController::new(
            RateControlMode::Vbr {
                target_kbps: 300,
                max_qidx: 220,
            },
            320,
            192,
            30,
            1,
            30,
        );
        let inter = rc.frame_budget_bits(FrameType::Inter).unwrap();
        let key = rc.frame_budget_bits(FrameType::Key).unwrap();
        assert!(key > inter * 1.5);
        assert!((inter - 10_000.0).abs() < 4_000.0);
    }
}
