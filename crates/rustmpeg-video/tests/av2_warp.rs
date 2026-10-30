// SPDX-License-Identifier: Apache-2.0
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

//! End-to-end `WARP_NEWMV` coverage: the encoder must select at least one
//! warped block on zooming content, and the decoder must reproduce the
//! encoder reconstruction byte-for-byte.

use rustmpeg_video::Av2Decoder;
use rustmpeg_video::Av2Encoder;
use rustmpeg_video::Av2EncoderConfig;
use rustmpeg_video::av2::container::RawFrame;
use rustmpeg_video::av2::frame::ToolConfig;
use rustmpeg_video::av2::headers::PixelLayout;

fn zoom_frame(width: usize, height: usize, step: usize) -> RawFrame<u8> {
    let mut frame = RawFrame::new(width, height, 8, PixelLayout::I420);
    let scale = 1.0 + 0.06 * step as f64;
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    for y in 0..height {
        for x in 0..width {
            let sx = ((x as f64 - cx) / scale + cx).clamp(0.0, (width - 1) as f64) as i32;
            let sy = ((y as f64 - cy) / scale + cy).clamp(0.0, (height - 1) as f64) as i32;
            frame.y[y * width + x] = ((sx * 5 + sy * 3) % 180 + 40) as u8;
        }
    }
    frame
}

#[test]
fn warped_motion_is_selected_and_decodes_exactly() {
    let (width, height) = (64usize, 64usize);
    let config = Av2EncoderConfig {
        width,
        height,
        layout: PixelLayout::I420,
        bit_depth: 8,
        base_qidx: 100,
        tools: ToolConfig {
            enable_warped_motion: true,
            ..Default::default()
        },
        ..Av2EncoderConfig::default()
    };
    let mut encoder = Av2Encoder::<u8>::new(config).expect("encoder");
    let mut decoder = Av2Decoder::<u8>::new();
    let mut warped = 0usize;
    let mut packets = Vec::new();
    for step in 0..3 {
        let frame = zoom_frame(width, height, step);
        let packet = encoder.encode_raw_frame(&frame).expect("encode");
        warped += packet.stats.warped_blocks;
        packets.push(packet);
    }
    assert!(
        warped > 0,
        "warped_motion enabled but no WARP_NEWMV block was selected"
    );
    for packet in &packets {
        let decoded = decoder
            .decode_packet(&packet.data)
            .expect("decode")
            .expect("decoded frame");
        let raw = packet.rec_frame.to_raw();
        assert_eq!(raw.y, decoded.frame.y, "decoder luma differs");
        assert_eq!(raw.u, decoded.frame.u, "decoder U differs");
        assert_eq!(raw.v, decoded.frame.v, "decoder V differs");
    }
}
