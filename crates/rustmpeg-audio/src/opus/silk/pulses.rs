// SPDX-License-Identifier: Apache-2.0
//! SILK excitation pulse decoding (RFC 6716 §4.2.7.8).

use std::vec;
use std::vec::Vec;

use crate::opus::{OpusError, RangeDecoder};

use super::define::{N_RATE_LEVELS, SHELL_CODEC_FRAME_LENGTH, SILK_MAX_PULSES};
use super::tables::{
    SILK_LSB_ICDF, SILK_PULSES_PER_BLOCK_ICDF, SILK_RATE_LEVELS_ICDF,
    SILK_SHELL_CODE_TABLE_OFFSETS, SILK_SHELL_CODE_TABLE0, SILK_SHELL_CODE_TABLE1,
    SILK_SHELL_CODE_TABLE2, SILK_SHELL_CODE_TABLE3, SILK_SIGN_ICDF,
};

const MAX_NB_SHELL_BLOCKS: usize = 20;

/// Decodes the excitation pulse amplitudes for one SILK frame.
///
/// The returned vector contains `frame_length` signed pulse amplitudes.  The
/// caller reconstructs the residual and applies LTP/LPC synthesis.
///
/// # Errors
///
/// Propagates range-decoder and symbol-range failures.
pub fn decode_pulses(
    decoder: &mut RangeDecoder<'_>,
    frame_length: usize,
    signal_type: i8,
    quant_offset_type: i8,
) -> Result<Vec<i16>, OpusError> {
    if !(0..=2).contains(&signal_type) {
        return Err(OpusError::Malformed("SILK signal type out of range"));
    }
    let rate_level =
        decoder.decode_icdf(&SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8)? as usize;
    if rate_level >= N_RATE_LEVELS {
        return Err(OpusError::Malformed("SILK rate level out of range"));
    }

    let blocks = frame_length.div_ceil(SHELL_CODEC_FRAME_LENGTH);
    if blocks > MAX_NB_SHELL_BLOCKS {
        return Err(OpusError::Malformed("SILK excitation frame is too long"));
    }
    // MB 10 ms frames nominally contain 120 samples but code 8 full shell
    // blocks; the trailing 8 samples are parsed and then discarded
    // (RFC 6716, Section 4.2.7.8, Table 44).
    let coded_length = blocks * SHELL_CODEC_FRAME_LENGTH;
    let mut pulses = vec![0_i16; coded_length];
    let mut sum_pulses = [0_i32; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts = [0_i32; MAX_NB_SHELL_BLOCKS];

    let count_icdf = &SILK_PULSES_PER_BLOCK_ICDF[rate_level];
    for (block, sum) in sum_pulses.iter_mut().enumerate().take(blocks) {
        let mut count = i32::from(decoder.decode_icdf(count_icdf, 8)?);
        while count == SILK_MAX_PULSES as i32 + 1 {
            n_lshifts[block] += 1;
            if n_lshifts[block] == 10 {
                count = i32::from(
                    decoder.decode_icdf(&SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1][1..], 8)?,
                );
            } else {
                count = i32::from(
                    decoder.decode_icdf(&SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1], 8)?,
                );
            }
        }
        *sum = count;
    }

    for (block, &count) in sum_pulses.iter().take(blocks).enumerate() {
        let start = block * SHELL_CODEC_FRAME_LENGTH;
        if count > 0 {
            shell_decode(
                &mut pulses[start..start + SHELL_CODEC_FRAME_LENGTH],
                decoder,
                count,
            )?;
        }
    }

    for (block, &shift) in n_lshifts.iter().take(blocks).enumerate() {
        if shift > 0 {
            let start = block * SHELL_CODEC_FRAME_LENGTH;
            for sample in &mut pulses[start..start + SHELL_CODEC_FRAME_LENGTH] {
                let mut value = i32::from(*sample);
                for _ in 0..shift {
                    value <<= 1;
                    value += i32::from(decoder.decode_icdf(&SILK_LSB_ICDF, 8)?);
                }
                *sample = value as i16;
            }
            sum_pulses[block] |= shift << 5;
        }
    }

    decode_signs(
        decoder,
        &mut pulses,
        coded_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
        blocks,
    )?;
    pulses.truncate(frame_length);
    Ok(pulses)
}

fn decode_split(
    decoder: &mut RangeDecoder<'_>,
    count: i32,
    table: &[u8],
) -> Result<(i16, i16), OpusError> {
    if count <= 0 {
        return Ok((0, 0));
    }
    let offset = usize::from(SILK_SHELL_CODE_TABLE_OFFSETS[count as usize]);
    let left = i16::from(decoder.decode_icdf(&table[offset..], 8)?);
    Ok((left, count as i16 - left))
}

fn shell_decode(
    output: &mut [i16],
    decoder: &mut RangeDecoder<'_>,
    pulses4: i32,
) -> Result<(), OpusError> {
    if output.len() != SHELL_CODEC_FRAME_LENGTH {
        return Err(OpusError::Malformed("SILK shell block has wrong length"));
    }
    // The tree is traversed depth-first, left subtree before right, because
    // the split symbols are interleaved in exactly that order by the encoder
    // (RFC 6716, Section 4.2.7.8.3).
    let (left, right) = decode_split(decoder, pulses4, &SILK_SHELL_CODE_TABLE3)?;
    let level3 = [left, right];

    let (left, right) = decode_split(decoder, i32::from(level3[0]), &SILK_SHELL_CODE_TABLE2)?;
    let mut level2 = [left, right, 0, 0];
    let (left, right) = decode_split(decoder, i32::from(level2[0]), &SILK_SHELL_CODE_TABLE1)?;
    let mut level1 = [left, right, 0, 0, 0, 0, 0, 0];
    let (left, right) = decode_split(decoder, i32::from(level1[0]), &SILK_SHELL_CODE_TABLE0)?;
    output[0] = left;
    output[1] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[1]), &SILK_SHELL_CODE_TABLE0)?;
    output[2] = left;
    output[3] = right;

    let (left, right) = decode_split(decoder, i32::from(level2[1]), &SILK_SHELL_CODE_TABLE1)?;
    level1[2] = left;
    level1[3] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[2]), &SILK_SHELL_CODE_TABLE0)?;
    output[4] = left;
    output[5] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[3]), &SILK_SHELL_CODE_TABLE0)?;
    output[6] = left;
    output[7] = right;

    let (left, right) = decode_split(decoder, i32::from(level3[1]), &SILK_SHELL_CODE_TABLE2)?;
    level2[2] = left;
    level2[3] = right;
    let (left, right) = decode_split(decoder, i32::from(level2[2]), &SILK_SHELL_CODE_TABLE1)?;
    level1[4] = left;
    level1[5] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[4]), &SILK_SHELL_CODE_TABLE0)?;
    output[8] = left;
    output[9] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[5]), &SILK_SHELL_CODE_TABLE0)?;
    output[10] = left;
    output[11] = right;

    let (left, right) = decode_split(decoder, i32::from(level2[3]), &SILK_SHELL_CODE_TABLE1)?;
    level1[6] = left;
    level1[7] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[6]), &SILK_SHELL_CODE_TABLE0)?;
    output[12] = left;
    output[13] = right;
    let (left, right) = decode_split(decoder, i32::from(level1[7]), &SILK_SHELL_CODE_TABLE0)?;
    output[14] = left;
    output[15] = right;
    Ok(())
}

fn decode_signs(
    decoder: &mut RangeDecoder<'_>,
    pulses: &mut [i16],
    frame_length: usize,
    signal_type: i8,
    quant_offset_type: i8,
    sum_pulses: &[i32],
    blocks: usize,
) -> Result<(), OpusError> {
    let base = 7 * (i32::from(quant_offset_type) + (i32::from(signal_type) << 1));
    for (block, &count) in sum_pulses.iter().take(blocks).enumerate() {
        if count <= 0 {
            continue;
        }
        let index = ((count & 0x1f).min(6)) as usize;
        let icdf = [SILK_SIGN_ICDF[base as usize + index], 0];
        let start = block * SHELL_CODEC_FRAME_LENGTH;
        let end = (start + SHELL_CODEC_FRAME_LENGTH).min(frame_length);
        for sample in &mut pulses[start..end] {
            if *sample > 0 && decoder.decode_icdf(&icdf, 8)? == 0 {
                *sample = -*sample;
            }
        }
    }
    Ok(())
}
