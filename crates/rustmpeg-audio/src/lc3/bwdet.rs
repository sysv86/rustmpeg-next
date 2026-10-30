// SPDX-License-Identifier: Apache-2.0
//! LC3 bandwidth signalling reader (Bluetooth LC3 v1.0.1 §3.4.2.3).

use super::Lc3Error;
use super::bits::BitReader;

/// Reads the transmitted bandwidth index (`0..=sample_rate_index`).
pub(crate) fn read_bandwidth(bits: &mut BitReader<'_>, sr: usize) -> Result<usize, Lc3Error> {
    if sr >= 5 {
        return Err(Lc3Error::InvalidConfig("LC3 sample rate index"));
    }
    let nbits = if sr == 0 {
        0
    } else {
        1 + (31 - (sr as u32).leading_zeros())
    };
    let bw = bits.read_uint(nbits)? as usize;
    if bw > sr {
        return Err(Lc3Error::Bitstream("bandwidth exceeds sample rate"));
    }
    Ok(bw)
}
