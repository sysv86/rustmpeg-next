// SPDX-License-Identifier: Apache-2.0
//! Public API integration tests for `rustmpeg-core`.

use rustmpeg_core::{
    AudioFormat, AudioFrame, Buffer, ColorDescription, CoreError, FrameRate, MediaDuration,
    MediaTime, Packet, PacketFlags, PixelFormat, Plane, Rational, SampleFormat, VideoFrame,
    VideoFrameView, sad_u8, sse_u8,
};

#[test]
fn zero_copy_buffers_keep_the_same_allocation() {
    let source = Buffer::from_vec((0..32).collect());
    let view = source.slice(8..24).expect("slice");
    assert_eq!(view.len(), 16);
    assert_eq!(
        view.as_ptr(),
        source.as_ptr().wrapping_add(8),
        "slices must not copy"
    );
}

#[test]
fn video_frames_expose_plane_geometry_and_rows() {
    let frame = VideoFrame::new(
        PixelFormat::I420,
        8,
        65,
        49,
        ColorDescription::BT709_LIMITED,
    )
    .expect("frame");
    assert_eq!(frame.plane_count(), 3);
    assert_eq!(frame.width(), 65);
    assert_eq!(frame.height(), 49);
    let chroma = frame.plane(1).expect("chroma");
    assert_eq!((chroma.width, chroma.height), (33, 25));
    assert_eq!(chroma.row(0).expect("row").len(), 33);
    assert!(frame.is_valid());
}

#[test]
fn timeline_converts_between_timebases_without_floats() {
    let timebase = Rational::new(1, 90_000).expect("timebase");
    let one_second = MediaTime::new(1, Rational::new(1, 1).expect("seconds"));
    assert_eq!(
        one_second.rescale(timebase).expect("rescale").ticks(),
        90_000
    );
    let duration = MediaDuration::new(2, Rational::new(1, 1).expect("seconds"));
    assert_eq!(
        duration.rescale(timebase).expect("rescale").ticks(),
        180_000
    );
    assert_eq!(
        FrameRate::new(60, 1).expect("rate").timebase(),
        Rational::new(1, 60).expect("frame timebase")
    );
}

#[test]
fn packets_carry_payload_timing_and_flags() {
    let timebase = Rational::new(1, 48_000).expect("timebase");
    let packet = Packet::new(1, Buffer::from_slice(b"payload"))
        .with_pts(MediaTime::new(0, timebase))
        .with_dts(MediaTime::new(0, timebase))
        .with_duration(MediaDuration::new(960, timebase))
        .with_flags(PacketFlags {
            keyframe: true,
            ..PacketFlags::default()
        });
    assert_eq!(packet.len(), 7);
    assert!(packet.is_keyframe());
    assert_eq!(packet.stream_index, 1);
}

#[test]
fn audio_frames_are_interleaved_and_sized() {
    let format = AudioFormat::new(SampleFormat::F32, 48_000, 2).expect("format");
    let frame = AudioFrame::new(format, Buffer::from_vec(vec![0; 16])).expect("audio");
    assert_eq!(frame.sample_count().expect("samples"), 2);
    assert!((frame.duration_seconds().expect("duration") - 2.0 / 48_000.0).abs() < 1e-12);
}

#[test]
fn plane_helper_rejects_wrong_lengths() {
    let plane = Plane::new(Buffer::from_vec(vec![0; 12]), 3, 4, 3);
    let error = VideoFrame::from_planes(
        PixelFormat::I400,
        8,
        4,
        3,
        ColorDescription::default(),
        [plane, Plane::empty(), Plane::empty()],
    )
    .expect_err("stride is too small for the frame");
    assert!(matches!(error, CoreError::InvalidConfig(_)));
}

#[test]
fn simd_primitives_match_the_public_scalar_definition() {
    let left: Vec<u8> = (0..100).map(|value| value as u8).collect();
    let right: Vec<u8> = (0..100)
        .map(|value| (value as u8).wrapping_add(3))
        .collect();
    let scalar_sad: u64 = left
        .iter()
        .zip(&right)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum();
    let scalar_sse: u64 = left
        .iter()
        .zip(&right)
        .map(|(a, b)| {
            let difference = u64::from(a.abs_diff(*b));
            difference * difference
        })
        .sum();
    assert_eq!(sad_u8(&left, &right), scalar_sad);
    assert_eq!(sse_u8(&left, &right), scalar_sse);
}
