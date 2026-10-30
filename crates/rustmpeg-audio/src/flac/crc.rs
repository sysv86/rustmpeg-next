// SPDX-License-Identifier: Apache-2.0
//! CRC-8 and CRC-16 as used by FLAC frames (RFC 9639, Sections 9.1.8 and 9.3).
//!
//! Both checksums are MSB-first, unreflected, and initialized to zero.  The
//! frame header uses the polynomial `x^8 + x^2 + x + 1` (`0x07`); the complete
//! frame uses `x^16 + x^15 + x^2 + 1` (`0x8005`).

const CRC8_POLYNOMIAL: u8 = 0x07;
const CRC16_POLYNOMIAL: u16 = 0x8005;

/// Updates the FLAC CRC-8 with one byte.
pub(crate) fn crc8_update(mut crc: u8, byte: u8) -> u8 {
    crc ^= byte;
    for _ in 0..8 {
        crc = if crc & 0x80 != 0 {
            (crc << 1) ^ CRC8_POLYNOMIAL
        } else {
            crc << 1
        };
    }
    crc
}

/// Updates the FLAC CRC-16 with one byte.
pub(crate) fn crc16_update(mut crc: u16, byte: u8) -> u16 {
    crc ^= u16::from(byte) << 8;
    for _ in 0..8 {
        crc = if crc & 0x8000 != 0 {
            (crc << 1) ^ CRC16_POLYNOMIAL
        } else {
            crc << 1
        };
    }
    crc
}

/// Computes the FLAC CRC-8 of a byte slice.
#[cfg(test)]
pub(crate) fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0, |crc, &byte| crc8_update(crc, byte))
}

/// Computes the FLAC CRC-16 of a byte slice.
#[cfg(test)]
pub(crate) fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0, |crc, &byte| crc16_update(crc, byte))
}

#[cfg(test)]
mod tests {
    use super::{crc8, crc16};

    // Check values from RFC 9639, Appendix D.1.
    #[test]
    fn rfc9639_example_1_header_crc8() {
        let header = [0xff, 0xf8, 0x69, 0x18, 0x00, 0x00];
        assert_eq!(crc8(&header), 0xbf);
    }

    #[test]
    fn rfc9639_example_1_frame_crc16() {
        let frame = [
            0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b,
        ];
        assert_eq!(crc16(&frame), 0xaa9a);
    }

    // Check values from RFC 9639, Appendix D.2 and D.3.
    #[test]
    fn rfc9639_example_2_header_crc8() {
        let header = [0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f];
        assert_eq!(crc8(&header), 0x99);
    }

    #[test]
    fn rfc9639_example_3_header_crc8() {
        let header = [0xff, 0xf8, 0x68, 0x02, 0x00, 0x17];
        assert_eq!(crc8(&header), 0xe9);
    }

    #[test]
    fn empty_input_keeps_the_initial_value() {
        assert_eq!(crc8(&[]), 0);
        assert_eq!(crc16(&[]), 0);
    }
}
