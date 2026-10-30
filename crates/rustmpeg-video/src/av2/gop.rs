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

//! Clean-room Wave 3 Lookahead Scene-Cut Detector & Hierarchical Pyramid GOP Scheduler (`gop.rs`).
//!
//! - **Scene-Cut Detector (`detect_scenecut`)**: Combines 16-bin luma histogram L1 distance and
//!   downsampled 8x8 block SAD to detect abrupt scene transitions and trigger `FrameType::Key`.
//! - **Hierarchical Pyramid GOP (`GopDecision`, `plan_gop_frame`)**: Schedules temporal layers
//!   (`tlayer_id = 0..=2`), multi-slot reference frame refresh masks (`LAST`, `GOLDEN`, `ALTREF`),
//!   and per-layer quantizer offsets (`delta_qidx`).

use crate::av2::container::RawFrame;
use crate::av2::headers::FrameType;
use crate::av2::intops::iclip;
use crate::av2::pixel::Pixel;

/// Scheduling decision for a single frame in a hierarchical GOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GopFramePlan {
    pub frame_type: FrameType,
    pub tlayer_id: u8,
    pub qp_offset: i16,
    pub refresh_frame_flags: u16,
    pub ref_slots: [i8; 2],
    pub is_scenecut: bool,
}

/// Computes the normalized 16-bin luma histogram L1 difference and mean absolute pixel difference
/// between `prev` and `cur` frames, returning `true` if `cur` is a scene cut.
pub fn detect_scenecut<P: Pixel>(prev: &RawFrame<P>, cur: &RawFrame<P>) -> bool {
    if prev.width != cur.width || prev.height != cur.height || prev.y.is_empty() {
        return true;
    }

    let shift = (cur.bit_depth - 4) as i32;
    let mut hist_prev = [0u64; 16];
    let mut hist_cur = [0u64; 16];
    let mut total_sad = 0u64;

    let step = (cur.y.len() / 4096).max(1);
    let mut sampled = 0u64;
    let mut idx = 0usize;
    while idx < cur.y.len() {
        let p = prev.y[idx].to_i32();
        let c = cur.y[idx].to_i32();
        let bp = ((p >> shift) & 15) as usize;
        let bc = ((c >> shift) & 15) as usize;
        hist_prev[bp] += 1;
        hist_cur[bc] += 1;
        total_sad += (p - c).unsigned_abs() as u64;
        sampled += 1;
        idx += step;
    }

    if sampled == 0 {
        return false;
    }

    let mut hist_l1 = 0u64;
    for b in 0..16 {
        hist_l1 += hist_prev[b].abs_diff(hist_cur[b]);
    }

    let bd_shift = (cur.bit_depth - 8) as u64;
    let avg_sad_8bit = (total_sad >> bd_shift) / sampled;
    // Trigger scene cut if > 55% of histogram mass shifted AND mean pixel delta > 28 (in 8-bit scale)
    hist_l1 * 100 > sampled * 110 && avg_sad_8bit > 28
}

/// Plans the [`GopFramePlan`] (frame type, temporal layer `0..=2`, QP offset, DPB refresh mask,
/// and active reference slots) for frame `frame_idx` within a hierarchical pyramid GOP.
pub fn plan_gop_frame(
    frame_idx: u64,
    keyint: usize,
    hierarchical_gop: bool,
    is_scenecut: bool,
    num_ref_slots: u8,
) -> GopFramePlan {
    let n_slots = num_ref_slots.clamp(2, 8) as u16;
    let is_key =
        is_scenecut || keyint <= 1 || frame_idx == 0 || (frame_idx as usize).is_multiple_of(keyint);

    if is_key {
        return GopFramePlan {
            frame_type: FrameType::Key,
            tlayer_id: 0,
            qp_offset: -10,
            refresh_frame_flags: (1u16 << n_slots) - 1,
            ref_slots: [0, 0],
            is_scenecut,
        };
    }

    if !hierarchical_gop {
        let inter_slots = (n_slots - 1).max(1);
        let slot = 1 + ((frame_idx - 1) as u16 % inter_slots);
        let prev_slot = if frame_idx == 1 {
            0
        } else {
            1 + ((frame_idx - 2) as u16 % inter_slots)
        };
        return GopFramePlan {
            frame_type: FrameType::Inter,
            tlayer_id: 0,
            qp_offset: 2,
            refresh_frame_flags: 1u16 << slot,
            ref_slots: [prev_slot as i8, 0],
            is_scenecut: false,
        };
    }

    // 4-frame hierarchical mini-GOP pyramid:
    // - pos % 4 == 0: Temporal Layer 0 (Base anchor, updates slot 1, references slot 0 & 1, qp_offset = -2)
    // - pos % 2 == 0: Temporal Layer 1 (Mid anchor, updates slot 2, references slot 1 & 0, qp_offset = +2)
    // - pos % 2 == 1: Temporal Layer 2 (Disposable leaf, updates slot 3, references nearest anchors, qp_offset = +6)
    let sub_pos = (frame_idx % 4) as usize;
    match sub_pos {
        0 => GopFramePlan {
            frame_type: FrameType::Inter,
            tlayer_id: 0,
            qp_offset: -2,
            refresh_frame_flags: 1u16 << 1,
            ref_slots: [1, 0],
            is_scenecut: false,
        },
        2 => GopFramePlan {
            frame_type: FrameType::Inter,
            tlayer_id: 1,
            qp_offset: 2,
            refresh_frame_flags: 1u16 << 2,
            ref_slots: [1, 0],
            is_scenecut: false,
        },
        _ => GopFramePlan {
            frame_type: FrameType::Inter,
            tlayer_id: 2,
            qp_offset: iclip(6, -32, 32) as i16,
            refresh_frame_flags: 1u16 << 3,
            ref_slots: [2, 1],
            is_scenecut: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::PixelLayout;

    #[test]
    fn test_scenecut_detection_and_hierarchical_gop_schedule() {
        let f_dark = RawFrame::<u8>::new(64, 64, 8, PixelLayout::I420);
        let mut f_bright = RawFrame::<u8>::new(64, 64, 8, PixelLayout::I420);
        f_bright.y.fill(235);

        assert!(detect_scenecut(&f_dark, &f_bright));
        assert!(!detect_scenecut(&f_dark, &f_dark));

        let p0 = plan_gop_frame(0, 30, true, false, 8);
        let p1 = plan_gop_frame(1, 30, true, false, 8);
        let p2 = plan_gop_frame(2, 30, true, false, 8);
        let p4 = plan_gop_frame(4, 30, true, false, 8);
        assert_eq!(p0.frame_type, FrameType::Key);
        assert_eq!(p1.tlayer_id, 2);
        assert_eq!(p2.tlayer_id, 1);
        assert_eq!(p4.tlayer_id, 0);
        assert!(p4.qp_offset < p2.qp_offset && p2.qp_offset < p1.qp_offset);
    }
}
