// SPDX-License-Identifier: Apache-2.0
//! Variable-length code of RFC 9924, Sections 7.1.4 and 7.2.4.

use super::ApvError;
use super::bits::{BitReader, BitWriter};

/// Largest symbol accepted by the generic VLC reader.  The APV syntax uses
/// values far below this bound; the cap keeps a hostile all-zero stream from
/// running away.
const MAX_SYMBOL: u32 = 1 << 20;

/// Reads one `h(v)` symbol with the supplied `kParam`.
pub(crate) fn read_hv(reader: &mut BitReader<'_>, k_param: u32) -> Result<u32, ApvError> {
    if k_param > 31 {
        return Err(ApvError::Invalid("kParam exceeds 31"));
    }
    let mut symbol = 0_u32;
    let parse_exp_golomb;
    let mut k = k_param;

    if reader.read_bit()? == 1 {
        parse_exp_golomb = false;
    } else if reader.read_bit()? == 0 {
        symbol = symbol
            .checked_add(1_u32 << k)
            .ok_or(ApvError::Invalid("VLC symbol overflow"))?;
        parse_exp_golomb = false;
    } else {
        symbol = symbol
            .checked_add(2_u32 << k)
            .ok_or(ApvError::Invalid("VLC symbol overflow"))?;
        parse_exp_golomb = true;
    }

    if parse_exp_golomb {
        loop {
            if reader.read_bit()? == 1 {
                break;
            }
            symbol = symbol
                .checked_add(1_u32 << k)
                .ok_or(ApvError::Invalid("VLC symbol overflow"))?;
            k += 1;
            if k > 31 {
                return Err(ApvError::Invalid("VLC exponent is too large"));
            }
            if symbol > MAX_SYMBOL {
                return Err(ApvError::Limit("VLC symbol"));
            }
        }
    }

    if k > 0 {
        symbol = symbol
            .checked_add(reader.read_bits(k)?)
            .ok_or(ApvError::Invalid("VLC symbol overflow"))?;
    }
    if symbol > MAX_SYMBOL {
        return Err(ApvError::Limit("VLC symbol"));
    }
    Ok(symbol)
}

/// Writes one `h(v)` symbol with the supplied `kParam`.
pub(crate) fn write_hv(writer: &mut BitWriter, symbol: u32, k_param: u32) -> Result<(), ApvError> {
    if k_param > 31 || symbol > MAX_SYMBOL {
        return Err(ApvError::Limit("VLC symbol"));
    }
    const PREFIX: [[u32; 2]; 3] = [[1, 0], [0, 0], [0, 1]];
    let prefix = ((symbol >> k_param).min(2)) as usize;
    let mut value = u64::from(symbol);
    let mut k = k_param;
    let mut bit_count = 0_usize;

    while value >= (1_u64 << k) {
        value -= 1_u64 << k;
        if bit_count < 2 {
            writer.write_bit(PREFIX[prefix][bit_count]);
        } else {
            writer.write_bit(0);
        }
        if bit_count >= 2 {
            k += 1;
            if k > 31 {
                return Err(ApvError::Limit("VLC exponent"));
            }
        }
        bit_count += 1;
        if bit_count > 64 {
            return Err(ApvError::Limit("VLC code length"));
        }
    }

    if bit_count < 2 {
        writer.write_bit(PREFIX[prefix][bit_count]);
    } else {
        writer.write_bit(1);
    }
    if k > 0 {
        writer.write_bits(value as u32, k);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip_for_every_context() {
        for k_param in 0..=5 {
            for symbol in [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 100, 255, 1024, 32767] {
                let mut writer = BitWriter::new();
                write_hv(&mut writer, symbol, k_param).expect("write");
                let bytes = writer.into_bytes();
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    read_hv(&mut reader, k_param).expect("read"),
                    symbol,
                    "kParam {k_param}, symbol {symbol}"
                );
                reader.align_zero().expect("padding");
                assert!(reader.read_bit().is_err());
            }
        }
    }

    #[test]
    fn truncated_symbol_fails() {
        let mut reader = BitReader::new(&[]);
        assert!(read_hv(&mut reader, 0).is_err());
    }
}
