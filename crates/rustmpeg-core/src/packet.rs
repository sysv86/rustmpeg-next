// SPDX-License-Identifier: Apache-2.0
//! Encoded packets and their timing/flags metadata.

use crate::buffer::Buffer;
use crate::time::{MediaDuration, MediaTime};

/// Boolean flags attached to an encoded packet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PacketFlags {
    /// The packet is a random-access point.
    pub keyframe: bool,
    /// The decoder may discard this packet.
    pub discard: bool,
    /// The packet marks the end of a stream.
    pub end_of_stream: bool,
    /// The payload is known to be damaged.
    pub corrupted: bool,
}

/// One encoded packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    /// Stream index inside the container.
    pub stream_index: u32,
    /// Presentation timestamp.
    pub pts: Option<MediaTime>,
    /// Decode timestamp.
    pub dts: Option<MediaTime>,
    /// Packet duration.
    pub duration: Option<MediaDuration>,
    /// Encoded payload.
    pub data: Buffer,
    /// Packet flags.
    pub flags: PacketFlags,
}

impl Packet {
    /// Creates a packet with no timing information and default flags.
    pub fn new(stream_index: u32, data: Buffer) -> Self {
        Self {
            stream_index,
            pts: None,
            dts: None,
            duration: None,
            data,
            flags: PacketFlags::default(),
        }
    }

    /// Sets the presentation timestamp.
    pub fn with_pts(mut self, pts: MediaTime) -> Self {
        self.pts = Some(pts);
        self
    }

    /// Sets the decode timestamp.
    pub fn with_dts(mut self, dts: MediaTime) -> Self {
        self.dts = Some(dts);
        self
    }

    /// Sets the packet duration.
    pub fn with_duration(mut self, duration: MediaDuration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Replaces the packet flags.
    pub fn with_flags(mut self, flags: PacketFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Replaces the stream index.
    pub fn with_stream_index(mut self, stream_index: u32) -> Self {
        self.stream_index = stream_index;
        self
    }

    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` when the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `true` when the keyframe flag is set.
    pub fn is_keyframe(&self) -> bool {
        self.flags.keyframe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::time::{MediaTime, Rational};

    fn timebase() -> Rational {
        Rational::new(1, 90_000).expect("timebase")
    }

    #[test]
    fn builders_set_timing_without_losing_payload() {
        let packet = Packet::new(2, Buffer::from_slice(b"frame"))
            .with_pts(MediaTime::new(180_000, timebase()))
            .with_dts(MediaTime::new(90_000, timebase()))
            .with_duration(MediaDuration::new(3_000, timebase()))
            .with_flags(PacketFlags {
                keyframe: true,
                ..PacketFlags::default()
            });
        assert_eq!(packet.stream_index, 2);
        assert_eq!(packet.len(), 5);
        assert!(!packet.is_empty());
        assert!(packet.is_keyframe());
        assert_eq!(packet.pts.expect("pts").ticks(), 180_000);
        assert_eq!(packet.dts.expect("dts").ticks(), 90_000);
        assert_eq!(packet.duration.expect("duration").ticks(), 3_000);
    }

    #[test]
    fn packet_defaults_are_empty_and_unflagged() {
        let packet = Packet::new(0, Buffer::empty());
        assert!(packet.is_empty());
        assert!(!packet.is_keyframe());
        assert!(packet.pts.is_none());
        assert!(packet.dts.is_none());
        assert!(packet.duration.is_none());
    }
}
