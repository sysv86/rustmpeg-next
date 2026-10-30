// SPDX-License-Identifier: Apache-2.0
//! Range decoder tests for RFC 6716, Section 4.1.
//!
//! The decoder is round-tripped against a test-only encoder implementing the
//! arithmetic of RFC 6716, Section 5.1.  The RFC states that the encoder and
//! decoder must hold exactly the same `rng` after each symbol, so every
//! round-trip also compares the range.

#[path = "support/range_encoder.rs"]
mod range_encoder;

use range_encoder::RangeEncoder;
use rustmpeg_audio::{OpusError, RangeDecoder};

/// Small deterministic xorshift generator, local to the tests.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next() % bound
    }
}

fn assert_valid(decoder: &RangeDecoder<'_>) {
    assert!(
        decoder.range() > 1 << 23,
        "range {} is not renormalized",
        decoder.range()
    );
}

#[test]
fn initialization_matches_rfc() {
    let decoder = RangeDecoder::new(&[]);
    assert_eq!(decoder.range(), 1 << 31);
    assert_eq!(decoder.tell(), 1);

    // A one-byte payload still renormalizes to the full initial range.
    let decoder = RangeDecoder::new(&[0x12]);
    assert_eq!(decoder.range(), 1 << 31);
}

#[test]
fn binary_symbols_round_trip() {
    for logp in [1, 2, 3, 5, 8, 15, 16] {
        let mut rng = Rng(0x1234_5678 + u64::from(logp));
        let bits: Vec<bool> = (0..400).map(|_| rng.next() & 1 != 0).collect();
        let mut encoder = RangeEncoder::new();
        let mut ranges = Vec::new();
        let mut tells = Vec::new();
        for &bit in &bits {
            encoder.encode_bit_logp(logp, bit);
            ranges.push(encoder.range());
            tells.push(encoder.tell());
        }
        let data = encoder.finish();

        let mut decoder = RangeDecoder::new(&data);
        assert_eq!(decoder.tell(), 1, "logp {logp}");
        for (index, &expected) in bits.iter().enumerate() {
            let decoded = decoder.decode_bit_logp(logp).unwrap();
            assert_eq!(decoded, expected, "logp {logp}, symbol {index}");
            assert_eq!(decoder.range(), ranges[index], "logp {logp}");
            assert_eq!(decoder.tell(), tells[index], "logp {logp}, tell");
            assert_valid(&decoder);
        }
    }
}

#[test]
fn icdf_symbols_round_trip() {
    const TABLES: [(&[u8], u32); 4] = [
        (&[192, 128, 64, 0], 8),
        (&[200, 120, 60, 20, 0], 8),
        (&[14, 11, 7, 4, 1, 0], 4),
        (&[1, 0], 1),
    ];
    for (table, ftb) in TABLES {
        let mut rng = Rng(0xdead_beef ^ u64::from(ftb));
        let symbols: Vec<u8> = (0..300)
            .map(|_| u8::try_from(rng.below(table.len() as u32)).unwrap())
            .collect();
        let mut encoder = RangeEncoder::new();
        let mut ranges = Vec::new();
        let mut tells = Vec::new();
        for &symbol in &symbols {
            encoder.encode_icdf(table, ftb, usize::from(symbol));
            ranges.push(encoder.range());
            tells.push(encoder.tell());
        }
        let data = encoder.finish();

        let mut decoder = RangeDecoder::new(&data);
        for (index, &expected) in symbols.iter().enumerate() {
            let decoded = decoder.decode_icdf(table, ftb).unwrap();
            assert_eq!(decoded, expected, "table {table:?}, symbol {index}");
            assert_eq!(decoder.range(), ranges[index], "table {table:?}");
            assert_eq!(decoder.tell(), tells[index], "table {table:?}, tell");
            assert_valid(&decoder);
        }
    }
}

#[test]
fn full_8_bit_icdf_table_round_trips() {
    // 256 equiprobable symbols: icdf[k] = 255 - k, terminated by zero.
    let table: Vec<u8> = (0..256).rev().map(|value| value as u8).collect();
    let mut rng = Rng(0x0bad_f00d);
    let symbols: Vec<u8> = (0..300).map(|_| rng.next() as u8).collect();
    let mut encoder = RangeEncoder::new();
    let mut ranges = Vec::new();
    let mut tells = Vec::new();
    for &symbol in &symbols {
        encoder.encode_icdf(&table, 8, usize::from(symbol));
        ranges.push(encoder.range());
        tells.push(encoder.tell());
    }
    let data = encoder.finish();
    let mut decoder = RangeDecoder::new(&data);
    for (index, &expected) in symbols.iter().enumerate() {
        let decoded = decoder.decode_icdf(&table, 8).unwrap();
        assert_eq!(decoded, expected);
        assert_eq!(decoder.range(), ranges[index]);
        assert_eq!(decoder.tell(), tells[index]);
        assert_valid(&decoder);
    }
}

#[test]
fn unavailable_icdf_symbols_are_never_returned() {
    // Symbols 0 and 1 share a cumulative boundary: symbol 1 has zero
    // probability and must never be decoded.
    let table = [128, 128, 64, 0];
    let mut encoder = RangeEncoder::new();
    for symbol in [0usize, 2, 3, 0, 3, 2, 0, 0, 3, 2] {
        encoder.encode_icdf(&table, 8, symbol);
    }
    let data = encoder.finish();
    let mut decoder = RangeDecoder::new(&data);
    for _ in 0..10 {
        let symbol = decoder.decode_icdf(&table, 8).unwrap();
        assert_ne!(symbol, 1);
    }
}

#[test]
fn uniformly_distributed_integers_round_trip() {
    let totals = [
        1,
        2,
        3,
        5,
        17,
        255,
        256,
        257,
        1_000,
        65_535,
        65_536,
        1 << 20,
        u32::MAX,
    ];
    for ft in totals {
        let mut rng = Rng(0xfeed_face ^ u64::from(ft));
        let values: Vec<u32> = (0..120)
            .map(|_| {
                let value = rng.next() % ft;
                // Include the boundaries explicitly.
                value
            })
            .collect();
        let mut encoder = RangeEncoder::new();
        let mut ranges = Vec::new();
        let mut tells = Vec::new();
        for &value in &values {
            encoder.encode_uint(ft, value);
            ranges.push(encoder.range());
            tells.push(encoder.tell());
        }
        let data = encoder.finish();

        let mut decoder = RangeDecoder::new(&data);
        for (index, &expected) in values.iter().enumerate() {
            let decoded = decoder.decode_uint(ft).unwrap();
            assert_eq!(decoded, expected, "ft {ft}, symbol {index}");
            assert_eq!(decoder.range(), ranges[index], "ft {ft}");
            assert_eq!(decoder.tell(), tells[index], "ft {ft}, tell");
            assert_valid(&decoder);
        }
    }
}

#[test]
fn uint_boundaries_round_trip() {
    let mut encoder = RangeEncoder::new();
    let values = [(3u32, 0u32), (3, 2), (257, 0), (257, 256), (257, 100)];
    let mut tells = Vec::new();
    for &(ft, value) in &values {
        encoder.encode_uint(ft, value);
        tells.push(encoder.tell());
    }
    let data = encoder.finish();
    let mut decoder = RangeDecoder::new(&data);
    for (index, &(ft, expected)) in values.iter().enumerate() {
        assert_eq!(decoder.decode_uint(ft).unwrap(), expected);
        assert_eq!(decoder.tell(), tells[index]);
        assert_valid(&decoder);
    }
}

#[test]
fn decode_bin_and_update_round_trip() {
    let mut rng = Rng(0x5555_aaaa);
    let mut encoder = RangeEncoder::new();
    let mut ranges = Vec::new();
    let mut tells = Vec::new();
    let symbols: Vec<u32> = (0..200).map(|_| rng.below(16)).collect();
    for &symbol in &symbols {
        encoder.encode(symbol, symbol + 1, 16);
        ranges.push(encoder.range());
        tells.push(encoder.tell());
    }
    let data = encoder.finish();

    let mut decoder = RangeDecoder::new(&data);
    for (index, &expected) in symbols.iter().enumerate() {
        let fs = decoder.decode_bin(4).unwrap();
        assert_eq!(fs, expected);
        decoder.update(expected, expected + 1, 16).unwrap();
        assert_eq!(decoder.range(), ranges[index]);
        assert_eq!(decoder.tell(), tells[index]);
        assert_valid(&decoder);
    }
}

#[test]
fn raw_bits_round_trip() {
    let mut encoder = RangeEncoder::new();
    let mut tells = Vec::new();
    for bit in [false, true, false, false, true] {
        encoder.encode_bit_logp(2, bit);
        tells.push(encoder.tell());
    }
    encoder.write_raw_bits(1, 1);
    tells.push(encoder.tell());
    encoder.write_raw_bits(8, 0xa5);
    tells.push(encoder.tell());
    encoder.write_raw_bits(10, 0x2aa);
    tells.push(encoder.tell());
    encoder.write_raw_bits(16, 0xbeef);
    tells.push(encoder.tell());
    let data = encoder.finish();

    let mut decoder = RangeDecoder::new(&data);
    for (index, expected) in [false, true, false, false, true].into_iter().enumerate() {
        assert_eq!(decoder.decode_bit_logp(2).unwrap(), expected);
        assert_eq!(decoder.tell(), tells[index]);
        assert_valid(&decoder);
    }
    let raw = [(1, 1), (8, 0xa5), (10, 0x2aa), (16, 0xbeef)];
    for (index, (bits, expected)) in raw.into_iter().enumerate() {
        assert_eq!(decoder.decode_raw_bits(bits).unwrap(), expected);
        assert_eq!(decoder.tell(), tells[5 + index]);
    }
}

#[test]
fn raw_bits_of_32_bits_round_trip() {
    let mut encoder = RangeEncoder::new();
    encoder.encode_bit_logp(1, true);
    encoder.encode_bit_logp(2, false);
    encoder.write_raw_bits(32, 0xdead_beef);
    let tell = encoder.tell();
    let data = encoder.finish();

    let mut decoder = RangeDecoder::new(&data);
    assert!(decoder.decode_bit_logp(1).unwrap());
    assert!(!decoder.decode_bit_logp(2).unwrap());
    assert_eq!(decoder.decode_raw_bits(32).unwrap(), 0xdead_beef);
    assert_eq!(decoder.tell(), tell);
}

#[test]
fn raw_bits_past_the_frame_are_zero() {
    let mut decoder = RangeDecoder::new(&[]);
    assert_eq!(decoder.decode_raw_bits(8).unwrap(), 0);
    assert_eq!(decoder.decode_raw_bits(24).unwrap(), 0);
    // Reading beyond the start of the buffer keeps returning zero.
    assert_eq!(decoder.decode_raw_bits(32).unwrap(), 0);
}

#[test]
fn short_frame_normalizes_past_the_end() {
    // A one-byte frame must be treated as zero bytes after its end.  Decoding
    // enough symbols forces several renormalizations past the frame.
    let mut decoder = RangeDecoder::new(&[0x00]);
    for _ in 0..64 {
        decoder.decode_bit_logp(1).unwrap();
        assert_valid(&decoder);
    }
    // The range decoder reads ahead into the buffer, but raw bits beyond the
    // one-byte frame still come back as zero.
    assert_eq!(decoder.decode_raw_bits(8).unwrap(), 0x00);
    assert_eq!(decoder.decode_raw_bits(32).unwrap(), 0);
}

#[test]
fn tell_is_monotonic_and_bounded() {
    let mut encoder = RangeEncoder::new();
    let mut rng = Rng(0x0102_0304);
    let mut encoder_tells = Vec::new();
    for _ in 0..100 {
        encoder.encode_bit_logp(3, rng.next() & 1 != 0);
        encoder_tells.push(encoder.tell());
        let symbol = rng.below(5);
        encoder.encode_icdf(&[200, 120, 60, 20, 0], 8, symbol as usize);
        encoder_tells.push(encoder.tell());
        encoder.encode_uint(1_000, rng.below(1_000));
        encoder_tells.push(encoder.tell());
    }
    encoder.write_raw_bits(8, 0x5a);
    encoder_tells.push(encoder.tell());
    let data = encoder.finish();

    let mut decoder = RangeDecoder::new(&data);
    let mut tell = decoder.tell();
    assert_eq!(tell, 1);
    let mut symbol_index = 0;
    for _ in 0..100 {
        decoder.decode_bit_logp(3).unwrap();
        let next = decoder.tell();
        assert!(next >= tell, "tell went backwards: {tell} -> {next}");
        assert_eq!(next, encoder_tells[symbol_index]);
        symbol_index += 1;
        tell = next;
        decoder.decode_icdf(&[200, 120, 60, 20, 0], 8).unwrap();
        let next = decoder.tell();
        assert!(next >= tell);
        assert_eq!(next, encoder_tells[symbol_index]);
        symbol_index += 1;
        tell = next;
        decoder.decode_uint(1_000).unwrap();
        let next = decoder.tell();
        assert!(next >= tell);
        assert_eq!(next, encoder_tells[symbol_index]);
        symbol_index += 1;
        tell = next;
    }
    assert_eq!(decoder.decode_raw_bits(8).unwrap(), 0x5a);
    assert_eq!(decoder.tell(), encoder_tells[symbol_index]);
    assert!(decoder.tell() <= 8 * decoder_payload_bits(&data) + 64);
}

fn decoder_payload_bits(data: &[u8]) -> u32 {
    u32::try_from(data.len() * 8).unwrap()
}

#[test]
fn corrupt_uint_values_are_reported() {
    // ft = 257 leaves one value in the coded space outside 0..257, so some
    // inputs must be reported as corrupt rather than silently decoded.
    let mut rng = Rng(0x7777_8888);
    let mut corrupt = 0;
    for _ in 0..10_000 {
        let mut data = [0u8; 4];
        for byte in &mut data {
            *byte = rng.next() as u8;
        }
        let mut decoder = RangeDecoder::new(&data);
        match decoder.decode_uint(257) {
            Ok(value) => assert!(value < 257),
            Err(OpusError::RangeValue { value, ft }) => {
                assert!(value >= ft);
                assert_eq!(ft, 257);
                corrupt += 1;
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert!(corrupt > 0, "no corrupt uint value was detected");
}

#[test]
fn arbitrary_payloads_never_panic() {
    let mut rng = Rng(0xabcd_ef01);
    for iteration in 0..2_000 {
        let length = (rng.next() % 65) as usize;
        let mut data = vec![0u8; length];
        for byte in &mut data {
            *byte = rng.next() as u8;
        }
        let mut decoder = RangeDecoder::new(&data);
        for _ in 0..32 {
            match rng.next() % 5 {
                0 => {
                    let _ = decoder.decode_bit_logp(1 + rng.below(15));
                }
                1 => {
                    let _ = decoder.decode_uint(1 + rng.below(70_000));
                }
                2 => {
                    let _ = decoder.decode_icdf(&[128, 64, 32, 16, 0], 8);
                }
                3 => {
                    let _ = decoder.decode_bin(rng.below(9));
                }
                _ => {
                    let _ = decoder.decode_raw_bits(rng.below(25));
                }
            }
            assert!(decoder.range() > 1 << 23, "iteration {iteration}");
        }
    }
}
