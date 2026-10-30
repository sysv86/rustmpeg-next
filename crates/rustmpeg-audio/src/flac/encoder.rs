// SPDX-License-Identifier: Apache-2.0
//! Native FLAC encoding (RFC 9639).
//!
//! The encoder produces standard native FLAC streams: a `fLaC` marker, a
//! STREAMINFO block with the MD5 of the unencoded audio, an optional SEEKTABLE,
//! and frames using constant, verbatim, fixed-predictor, or LPC subframes with
//! partitioned Rice residuals.
//!
//! Two entry points are provided:
//!
//! - [`encode_flac`] buffers the whole input, computes exact frame sizes and the
//!   MD5 signature, and writes a complete file with a seek table;
//! - [`FlacEncoder`] is a streaming encoder for writers that cannot seek.  Its
//!   STREAMINFO leaves the MD5, frame sizes, and total sample count unknown
//!   (zero), which RFC 9639 permits.
//!
//! Note that [`FlacEncoder`] also offers [`FlacEncoder::finish_seekable`] for a
//! [`Write`] + [`Seek`] sink, which back-patches the complete STREAMINFO.

use std::io::{Seek, SeekFrom, Write};

use super::bitwriter::BitWriter;
use super::crc::{crc8_update, crc16_update};
use super::md5::{Md5, interleaved_bytes};
use super::{FlacError, StreamInfo};
use crate::PcmFrame;

/// Maximum LPC order used by the encoder.
const MAX_LPC_ORDER: usize = 12;

/// Maximum predictor order for a fixed subframe.
const MAX_FIXED_ORDER: usize = 4;

/// Maximum Rice parameter in the 4-bit parameter form.
const MAX_RICE_PARAM_4: u32 = 14;

/// Maximum Rice parameter in the 5-bit parameter form.
const MAX_RICE_PARAM_5: u32 = 30;

/// Maximum residual partition order the encoder tries.
const MAX_PARTITION_ORDER: u32 = 8;

/// Encoder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacEncoderConfig {
    /// Sample rate in hertz (`1..=1048575`).
    pub sample_rate: u32,
    /// Number of channels (`1..=8`).
    pub channels: u8,
    /// Bits per sample (`4..=32`).
    pub bits_per_sample: u8,
    /// Block size in samples (`16..=65535`).
    pub block_size: u32,
    /// Compression level (`0..=8`). Higher levels search more LPC orders and
    /// residual partitions; the output is always lossless.
    pub compression_level: u8,
}

impl Default for FlacEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 2,
            bits_per_sample: 16,
            block_size: 4096,
            compression_level: 5,
        }
    }
}

impl FlacEncoderConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), FlacError> {
        if self.sample_rate == 0 || self.sample_rate > 1_048_575 {
            return Err(FlacError::InvalidConfig(
                "sample rate must be between 1 and 1048575 Hz",
            ));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(FlacError::InvalidConfig(
                "channel count must be between 1 and 8",
            ));
        }
        if !(4..=32).contains(&self.bits_per_sample) {
            return Err(FlacError::InvalidConfig(
                "bit depth must be between 4 and 32",
            ));
        }
        if !(16..=65_535).contains(&self.block_size) {
            return Err(FlacError::InvalidConfig(
                "block size must be between 16 and 65535 samples",
            ));
        }
        if self.compression_level > 8 {
            return Err(FlacError::InvalidConfig(
                "compression level must be between 0 and 8",
            ));
        }
        Ok(())
    }

    fn max_lpc_order(&self) -> usize {
        match self.compression_level {
            0 => 0,
            1 => 2,
            2 | 3 => 4,
            4 | 5 => 6,
            6 | 7 => 8,
            _ => MAX_LPC_ORDER,
        }
    }

    fn max_partition_order(&self) -> u32 {
        match self.compression_level {
            0 => 0,
            1 => 2,
            2 | 3 => 3,
            4 | 5 => 5,
            _ => MAX_PARTITION_ORDER,
        }
    }
}

/// One planned subframe.
#[derive(Debug, Clone)]
struct SubframePlan {
    kind: SubframeKind,
    wasted_bits: u32,
    coded_bits: u32,
    /// Samples after removing wasted bits.
    samples: Vec<i64>,
    /// Residual values for fixed/LPC subframes.
    residual: Vec<i64>,
    /// Quantized LPC coefficients.
    coefficients: [i32; MAX_LPC_ORDER],
    precision: u32,
    shift: u32,
    /// Residual coding plan for fixed/LPC subframes.
    residual_plan: Option<ResidualPlan>,
    /// Estimated size in bits, used to choose between subframe types.
    cost: u64,
}

/// Subframe prediction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubframeKind {
    Constant(i64),
    Verbatim,
    Fixed(u32),
    Lpc(u32),
}

/// Residual partition coding plan.
#[derive(Debug, Clone)]
struct ResidualPlan {
    /// `0` = 4-bit Rice parameters, `1` = 5-bit Rice parameters.
    method: u32,
    partition_order: u32,
    parameters: Vec<PartitionParameter>,
    cost: u64,
}

/// Coding choice for one residual partition.
#[derive(Debug, Clone, Copy)]
enum PartitionParameter {
    Rice(u32),
    Escape(u32),
}

/// Writes the STREAMINFO payload for `info`.
pub(crate) fn streaminfo_payload(info: &StreamInfo) -> [u8; 34] {
    let mut payload = [0u8; 34];
    payload[0..2].copy_from_slice(&info.min_block_size.to_be_bytes());
    payload[2..4].copy_from_slice(&info.max_block_size.to_be_bytes());
    payload[4..7].copy_from_slice(&info.min_frame_size.to_be_bytes()[1..]);
    payload[7..10].copy_from_slice(&info.max_frame_size.to_be_bytes()[1..]);
    let total = info.total_samples.unwrap_or(0) & 0x0000_000f_ffff_ffff;
    let packed = (u64::from(info.sample_rate) << 44)
        | (u64::from(info.channels.saturating_sub(1)) << 41)
        | (u64::from(info.bits_per_sample.saturating_sub(1)) << 36)
        | total;
    payload[10..18].copy_from_slice(&packed.to_be_bytes());
    payload[18..34].copy_from_slice(&info.md5);
    payload
}

/// Writes one metadata block header and its payload.
fn write_metadata_block<W: Write>(
    out: &mut W,
    kind: u8,
    payload: &[u8],
    last: bool,
) -> std::io::Result<()> {
    let length = payload.len() as u32;
    let header = [
        (u8::from(last) << 7) | (kind & 0x7f),
        (length >> 16) as u8,
        (length >> 8) as u8,
        length as u8,
    ];
    out.write_all(&header)?;
    out.write_all(payload)
}

/// Returns the coded-number bytes for a fixed-blocksize frame number
/// (RFC 9639, Section 9.1.5).
fn coded_number_bytes(value: u64) -> Vec<u8> {
    if value < 0x80 {
        return vec![value as u8];
    }
    let (length, prefix) = if value < 0x800 {
        (2usize, 0xc0u8)
    } else if value < 0x1_0000 {
        (3, 0xe0)
    } else if value < 0x20_0000 {
        (4, 0xf0)
    } else if value < 0x0400_0000 {
        (5, 0xf8)
    } else if value < 0x8000_0000 {
        (6, 0xfc)
    } else {
        (7, 0xfe)
    };
    let mut bytes = vec![0u8; length];
    let mut remaining = value;
    for index in (1..length).rev() {
        bytes[index] = 0x80 | (remaining & 0x3f) as u8;
        remaining >>= 6;
    }
    bytes[0] = prefix | remaining as u8;
    bytes
}

/// Chooses the frame-header block size code and optional extra bytes.
fn block_size_code(block_size: u32) -> (u8, Option<Vec<u8>>) {
    let table = [
        (192, 1u8),
        (576, 2),
        (1152, 3),
        (2304, 4),
        (4608, 5),
        (256, 8),
        (512, 9),
        (1024, 10),
        (2048, 11),
        (4096, 12),
        (8192, 13),
        (16_384, 14),
        (32_768, 15),
    ];
    for (size, code) in table {
        if block_size == size {
            return (code, None);
        }
    }
    if block_size <= 256 {
        (6, Some(vec![(block_size - 1) as u8]))
    } else {
        let stored = (block_size - 1) as u16;
        (7, Some(stored.to_be_bytes().to_vec()))
    }
}

/// Builds a frame header with CRC-8.
fn frame_header(
    block_size: u32,
    channel_assignment: u8,
    frame_number: u64,
) -> Result<Vec<u8>, FlacError> {
    let (size_code, size_extra) = block_size_code(block_size);
    let mut header = Vec::with_capacity(16);
    header.push(0xff);
    header.push(0xf8); // sync code, reserved 0, fixed block size
    header.push(size_code << 4); // sample-rate code 0 = STREAMINFO
    header.push(channel_assignment << 4); // bit depth 0 = STREAMINFO
    header.extend_from_slice(&coded_number_bytes(frame_number));
    if let Some(extra) = size_extra {
        header.extend_from_slice(&extra);
    }
    let crc = header.iter().fold(0u8, |crc, &byte| crc8_update(crc, byte));
    header.push(crc);
    Ok(header)
}

/// Appends a frame footer CRC-16.
fn finish_frame(mut frame: Vec<u8>) -> Vec<u8> {
    debug_assert!(frame.len() >= 6);
    let crc = frame
        .iter()
        .fold(0u16, |crc, &byte| crc16_update(crc, byte));
    frame.extend_from_slice(&crc.to_be_bytes());
    frame
}

/// Computes the fixed-predictor residual for `order`.
fn fixed_residual(samples: &[i64], order: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(samples.len().saturating_sub(order));
    for index in order..samples.len() {
        let prediction = match order {
            0 => 0,
            1 => samples[index - 1],
            2 => 2 * samples[index - 1] - samples[index - 2],
            3 => 3 * samples[index - 1] - 3 * samples[index - 2] + samples[index - 3],
            4 => {
                4 * samples[index - 1] - 6 * samples[index - 2] + 4 * samples[index - 3]
                    - samples[index - 4]
            }
            _ => unreachable!(),
        };
        out.push(samples[index] - prediction);
    }
    out
}

/// Computes autocorrelation values `r[0..=order]`.
fn autocorrelation(samples: &[i64], order: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0f64;
        for index in lag..samples.len() {
            sum += samples[index] as f64 * samples[index - lag] as f64;
        }
        r[lag] = sum;
    }
    r
}

/// Solves the normal equations with Levinson-Durbin recursion.
fn levinson_durbin(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r[0] <= 0.0 || !r[0].is_finite() {
        return None;
    }
    let mut a = vec![0.0f64; order + 1];
    let mut previous = vec![0.0f64; order + 1];
    let mut error = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let reflection = acc / error;
        if !reflection.is_finite() {
            return None;
        }
        previous[..=i].copy_from_slice(&a[..=i]);
        a[i] = reflection;
        for j in 1..i {
            a[j] = previous[j] - reflection * previous[i - j];
        }
        error *= 1.0 - reflection * reflection;
        if error <= 0.0 {
            return None;
        }
    }
    Some(a[1..=order].to_vec())
}

/// Quantizes LPC coefficients and returns `(coefficients, precision, shift)`.
fn quantize_lpc(coefficients: &[f64], base_bits: u32) -> Option<(Vec<i32>, u32, u32)> {
    let precision = if base_bits <= 16 { 12 } else { 15 };
    let limit = (1i64 << (precision - 1)) - 1;
    let max_abs = coefficients
        .iter()
        .fold(0.0f64, |max, value| max.max(value.abs()));
    if !max_abs.is_finite() || max_abs == 0.0 {
        return None;
    }
    let mut shift = 0u32;
    while shift < 15 && max_abs * f64::from(1u32 << (shift + 1)) <= limit as f64 {
        shift += 1;
    }
    let scale = f64::from(1u32 << shift);
    let mut quantized = Vec::with_capacity(coefficients.len());
    for &value in coefficients {
        let scaled = (value * scale).round();
        if !scaled.is_finite() {
            return None;
        }
        quantized.push((scaled as i64).clamp(-limit, limit) as i32);
    }
    if quantized.iter().all(|&value| value == 0) {
        return None;
    }
    Some((quantized, precision, shift))
}

/// Computes the LPC residual for quantized coefficients.
fn lpc_residual(samples: &[i64], coefficients: &[i32], shift: u32) -> Vec<i64> {
    let order = coefficients.len();
    let mut out = Vec::with_capacity(samples.len().saturating_sub(order));
    for index in order..samples.len() {
        let mut sum = 0i128;
        for (offset, &coefficient) in coefficients.iter().enumerate() {
            sum += i128::from(coefficient) * i128::from(samples[index - 1 - offset]);
        }
        let prediction = (sum >> shift) as i64;
        out.push(samples[index] - prediction);
    }
    out
}

/// Zigzag-folds a residual into an unsigned value.
#[inline]
fn fold(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Minimum number of signed bits needed to represent `value`.
#[inline]
fn signed_bits(value: i64) -> u32 {
    let mask = value ^ (value >> 63);
    64 - mask.leading_zeros() + 1
}

/// Plans one residual partition.
fn plan_partition(values: &[i64], method: u32) -> (PartitionParameter, u64) {
    let max_parameter = if method == 0 {
        MAX_RICE_PARAM_4
    } else {
        MAX_RICE_PARAM_5
    };
    let mut best = PartitionParameter::Rice(0);
    let mut best_cost = u64::MAX;
    for parameter in 0..=max_parameter {
        let mut cost = 0u64;
        for &value in values {
            let folded = fold(value);
            cost = cost
                .saturating_add(folded >> parameter)
                .saturating_add(1 + u64::from(parameter));
        }
        if cost < best_cost {
            best_cost = cost;
            best = PartitionParameter::Rice(parameter);
        }
    }
    let raw_bits = values
        .iter()
        .map(|&value| signed_bits(value))
        .max()
        .unwrap_or(1);
    if raw_bits <= 31 {
        let escape_cost = 5 + values.len() as u64 * u64::from(raw_bits);
        if escape_cost < best_cost {
            best_cost = escape_cost;
            best = PartitionParameter::Escape(raw_bits);
        }
    }
    (best, best_cost)
}

/// Chooses the best Rice method, partition order, and per-partition parameters.
fn plan_residual(
    residual: &[i64],
    block_size: usize,
    predictor_order: usize,
    max_partition_order: u32,
) -> Option<ResidualPlan> {
    let max_order = max_partition_order.min(MAX_PARTITION_ORDER);
    let mut best: Option<ResidualPlan> = None;
    for method in 0..=1u32 {
        let parameter_bits = if method == 0 { 4u32 } else { 5 };
        for partition_order in 0..=max_order {
            let partitions = 1usize << partition_order;
            if !block_size.is_multiple_of(partitions) {
                continue;
            }
            let samples_per_partition = block_size / partitions;
            if samples_per_partition <= predictor_order {
                continue;
            }
            let mut cost = 2 + 4 + partitions as u64 * u64::from(parameter_bits);
            let mut parameters = Vec::with_capacity(partitions);
            let mut offset = 0usize;
            let mut valid = true;
            for partition in 0..partitions {
                let count = if partition == 0 {
                    samples_per_partition - predictor_order
                } else {
                    samples_per_partition
                };
                let end = offset + count;
                if end > residual.len() {
                    valid = false;
                    break;
                }
                let (parameter, partition_cost) = plan_partition(&residual[offset..end], method);
                cost = cost.saturating_add(partition_cost);
                parameters.push(parameter);
                offset = end;
            }
            if !valid || offset != residual.len() {
                continue;
            }
            if best.as_ref().is_none_or(|current| cost < current.cost) {
                best = Some(ResidualPlan {
                    method,
                    partition_order,
                    parameters,
                    cost,
                });
            }
        }
    }
    best
}

/// Plans the best subframe representation for one channel.
fn plan_subframe(
    samples: &[i64],
    base_bits: u32,
    max_lpc_order: usize,
    max_partition_order: u32,
) -> Result<SubframePlan, FlacError> {
    let block_size = samples.len();
    if block_size == 0 {
        return Err(FlacError::EncodeError("empty block"));
    }

    // Wasted low bits: common trailing zeros of all nonzero samples.
    let mut wasted_bits = u32::MAX;
    for &sample in samples {
        if sample != 0 {
            wasted_bits = wasted_bits.min(sample.trailing_zeros());
        }
    }
    if wasted_bits == u32::MAX {
        wasted_bits = 0;
    }
    wasted_bits = wasted_bits.min(base_bits.saturating_sub(1));
    let coded_bits = base_bits - wasted_bits;
    let shifted: Vec<i64> = samples
        .iter()
        .map(|&sample| sample >> wasted_bits)
        .collect();

    let header_bits = 8 + u64::from(wasted_bits);
    let limit = 1i64 << (coded_bits - 1);
    if shifted
        .iter()
        .any(|&sample| sample < -limit || sample >= limit)
    {
        return Err(FlacError::EncodeError(
            "sample does not fit the subframe bit depth",
        ));
    }

    let constant = shifted.iter().all(|&sample| sample == shifted[0]);
    if constant {
        return Ok(SubframePlan {
            kind: SubframeKind::Constant(shifted[0]),
            wasted_bits,
            coded_bits,
            samples: shifted,
            residual: Vec::new(),
            coefficients: [0; MAX_LPC_ORDER],
            precision: 0,
            shift: 0,
            residual_plan: None,
            cost: header_bits + u64::from(coded_bits),
        });
    }

    let mut best = SubframePlan {
        kind: SubframeKind::Verbatim,
        wasted_bits,
        coded_bits,
        samples: shifted.clone(),
        residual: Vec::new(),
        coefficients: [0; MAX_LPC_ORDER],
        precision: 0,
        shift: 0,
        residual_plan: None,
        cost: header_bits + block_size as u64 * u64::from(coded_bits),
    };

    for order in 0..=MAX_FIXED_ORDER {
        if order >= block_size {
            break;
        }
        let residual = fixed_residual(&shifted, order);
        if let Some(plan) = plan_residual(&residual, block_size, order, max_partition_order) {
            let cost = header_bits + order as u64 * u64::from(coded_bits) + plan.cost;
            if cost < best.cost {
                best = SubframePlan {
                    kind: SubframeKind::Fixed(order as u32),
                    wasted_bits,
                    coded_bits,
                    samples: shifted.clone(),
                    residual,
                    coefficients: [0; MAX_LPC_ORDER],
                    precision: 0,
                    shift: 0,
                    residual_plan: Some(plan),
                    cost,
                };
            }
        }
    }

    if max_lpc_order > 0 && block_size > 1 {
        let max_order = max_lpc_order.min(MAX_LPC_ORDER).min(block_size - 1);
        let autocorrelation = autocorrelation(&shifted, max_order);
        if let Some(coefficients) = levinson_durbin(&autocorrelation, max_order) {
            for order in 1..=max_order {
                let coefficients = &coefficients[..order];
                let Some((quantized, precision, shift)) = quantize_lpc(coefficients, base_bits)
                else {
                    continue;
                };
                let residual = lpc_residual(&shifted, &quantized, shift);
                let Some(plan) = plan_residual(&residual, block_size, order, max_partition_order)
                else {
                    continue;
                };
                let cost = header_bits
                    + order as u64 * u64::from(coded_bits)
                    + 4
                    + 5
                    + order as u64 * u64::from(precision)
                    + plan.cost;
                if cost < best.cost {
                    let mut array = [0i32; MAX_LPC_ORDER];
                    array[..order].copy_from_slice(&quantized);
                    best = SubframePlan {
                        kind: SubframeKind::Lpc(order as u32),
                        wasted_bits,
                        coded_bits,
                        samples: shifted.clone(),
                        residual,
                        coefficients: array,
                        precision,
                        shift,
                        residual_plan: Some(plan),
                        cost,
                    };
                }
            }
        }
    }

    Ok(best)
}

/// Writes one planned subframe.
fn write_subframe(writer: &mut BitWriter, plan: &SubframePlan) -> Result<(), FlacError> {
    writer.write_bit(false); // zero padding bit
    let type_code = match plan.kind {
        SubframeKind::Constant(_) => 0,
        SubframeKind::Verbatim => 1,
        SubframeKind::Fixed(order) => 8 + order,
        SubframeKind::Lpc(order) => 31 + order,
    };
    if type_code > 63 {
        return Err(FlacError::EncodeError("subframe type out of range"));
    }
    writer.write_bits(6, u64::from(type_code));
    if plan.wasted_bits > 0 {
        writer.write_bit(true);
        writer.write_unary(plan.wasted_bits - 1);
    } else {
        writer.write_bit(false);
    }

    match plan.kind {
        SubframeKind::Constant(value) => {
            writer.write_signed(plan.coded_bits, value);
        }
        SubframeKind::Verbatim => {
            for &sample in &plan.samples {
                writer.write_signed(plan.coded_bits, sample);
            }
        }
        SubframeKind::Fixed(order) => {
            for &sample in &plan.samples[..order as usize] {
                writer.write_signed(plan.coded_bits, sample);
            }
            write_residual(writer, &plan.residual, plan)?;
        }
        SubframeKind::Lpc(order) => {
            for &sample in &plan.samples[..order as usize] {
                writer.write_signed(plan.coded_bits, sample);
            }
            writer.write_bits(4, u64::from(plan.precision - 1));
            writer.write_signed(5, i64::from(plan.shift));
            for &coefficient in &plan.coefficients[..order as usize] {
                writer.write_signed(plan.precision, i64::from(coefficient));
            }
            write_residual(writer, &plan.residual, plan)?;
        }
    }
    Ok(())
}

/// Writes the planned partitioned Rice residual.
fn write_residual(
    writer: &mut BitWriter,
    residual: &[i64],
    plan: &SubframePlan,
) -> Result<(), FlacError> {
    let residual_plan = plan
        .residual_plan
        .as_ref()
        .ok_or(FlacError::EncodeError("missing residual plan"))?;
    writer.write_bits(2, u64::from(residual_plan.method));
    writer.write_bits(4, u64::from(residual_plan.partition_order));
    let parameter_bits = if residual_plan.method == 0 { 4 } else { 5 };
    let escape = (1u32 << parameter_bits) - 1;
    let partitions = 1usize << residual_plan.partition_order;
    let samples_per_partition = plan.samples.len() / partitions;
    let predictor_order = match plan.kind {
        SubframeKind::Fixed(order) | SubframeKind::Lpc(order) => order as usize,
        _ => 0,
    };
    let mut offset = 0usize;
    for (partition, parameter) in residual_plan.parameters.iter().enumerate() {
        let count = if partition == 0 {
            samples_per_partition - predictor_order
        } else {
            samples_per_partition
        };
        let end = offset + count;
        let values = residual
            .get(offset..end)
            .ok_or(FlacError::EncodeError("residual partition overflow"))?;
        match *parameter {
            PartitionParameter::Rice(parameter) => {
                if parameter >= escape {
                    return Err(FlacError::EncodeError("Rice parameter out of range"));
                }
                writer.write_bits(parameter_bits, u64::from(parameter));
                let mask = (1u64 << parameter) - 1;
                for &value in values {
                    let folded = fold(value);
                    writer.write_unary((folded >> parameter) as u32);
                    writer.write_bits(parameter, folded & mask);
                }
            }
            PartitionParameter::Escape(raw_bits) => {
                writer.write_bits(parameter_bits, u64::from(escape));
                writer.write_bits(5, u64::from(raw_bits));
                for &value in values {
                    writer.write_signed(raw_bits, value);
                }
            }
        }
        offset = end;
    }
    Ok(())
}

/// Channel assignment chosen for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assignment {
    Independent,
    LeftSide,
    RightSide,
    MidSide,
}

impl Assignment {
    fn code(self, channels: u8) -> u8 {
        match self {
            Self::Independent => channels - 1,
            Self::LeftSide => 8,
            Self::RightSide => 9,
            Self::MidSide => 10,
        }
    }
}

/// Encodes one block of planar samples into a complete FLAC frame.
fn encode_frame(
    config: &FlacEncoderConfig,
    planes: &[&[i32]],
    frame_number: u64,
) -> Result<Vec<u8>, FlacError> {
    if planes.len() != usize::from(config.channels) {
        return Err(FlacError::EncodeError("channel count mismatch"));
    }
    let block_size = planes[0].len();
    if block_size == 0 {
        return Err(FlacError::EncodeError("empty frame"));
    }
    if planes.iter().any(|plane| plane.len() != block_size) {
        return Err(FlacError::EncodeError(
            "plane lengths differ within a frame",
        ));
    }

    let bits = u32::from(config.bits_per_sample);
    let side_bits = bits + 1;
    type ChannelData = (Vec<i64>, u32);
    type Candidate = (Assignment, Vec<ChannelData>);
    let mut candidates: Vec<Candidate> = Vec::new();

    if config.channels == 2 {
        let left: Vec<i64> = planes[0].iter().map(|&sample| i64::from(sample)).collect();
        let right: Vec<i64> = planes[1].iter().map(|&sample| i64::from(sample)).collect();
        let side: Vec<i64> = left.iter().zip(&right).map(|(&l, &r)| l - r).collect();
        let mid: Vec<i64> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l + r) >> 1)
            .collect();

        candidates.push((
            Assignment::Independent,
            vec![(left.clone(), bits), (right.clone(), bits)],
        ));
        candidates.push((
            Assignment::LeftSide,
            vec![(left.clone(), bits), (side.clone(), side_bits)],
        ));
        candidates.push((
            Assignment::RightSide,
            vec![(side.clone(), side_bits), (right.clone(), bits)],
        ));
        candidates.push((Assignment::MidSide, vec![(mid, bits), (side, side_bits)]));
    } else {
        let mut channels = Vec::with_capacity(planes.len());
        for plane in planes {
            channels.push((
                plane.iter().map(|&sample| i64::from(sample)).collect(),
                bits,
            ));
        }
        candidates.push((Assignment::Independent, channels));
    }

    let mut best: Option<(u64, Assignment, Vec<SubframePlan>)> = None;
    for (assignment, channels) in candidates {
        let mut total = 0u64;
        let mut plans = Vec::with_capacity(channels.len());
        for (samples, channel_bits) in channels {
            let plan = plan_subframe(
                &samples,
                channel_bits,
                config.max_lpc_order(),
                config.max_partition_order(),
            )?;
            total = total.saturating_add(plan.cost);
            plans.push(plan);
        }
        if best.as_ref().is_none_or(|current| total < current.0) {
            best = Some((total, assignment, plans));
        }
    }
    let (_, assignment, plans) = best.ok_or(FlacError::EncodeError(
        "failed to choose a channel assignment",
    ))?;

    let header = frame_header(
        block_size as u32,
        assignment.code(config.channels),
        frame_number,
    )?;
    let mut writer = BitWriter::with_capacity(block_size * 2 + header.len());
    writer.write_bytes(&header);
    for plan in &plans {
        write_subframe(&mut writer, plan)?;
    }
    writer.align_zero();
    Ok(finish_frame(writer.into_bytes()))
}

/// Validates that every sample fits the declared bit depth.
fn validate_samples(planes: &[Vec<i32>], bits_per_sample: u8) -> Result<(), FlacError> {
    let limit = 1i64 << (bits_per_sample - 1);
    for plane in planes {
        for &sample in plane {
            if i64::from(sample) < -limit || i64::from(sample) >= limit {
                return Err(FlacError::EncodeError(
                    "sample is outside the declared bit depth",
                ));
            }
        }
    }
    Ok(())
}

/// Builds a SEEKTABLE payload for the frames in `frames`.
///
/// `frames` contains `(first_sample, stream_offset, block_size)`, with offsets
/// relative to the first frame byte.
fn seektable_payload(frames: &[(u64, u64, u32)], sample_rate: u32) -> Vec<u8> {
    let mut entries: Vec<(u64, u64, u16)> = Vec::new();
    let interval = u64::from(sample_rate.max(1));
    let mut next_target = 0u64;
    for &(sample, offset, block) in frames {
        if sample >= next_target {
            entries.push((sample, offset, block as u16));
            next_target = sample.saturating_add(interval);
        }
    }
    let mut payload = Vec::with_capacity(entries.len() * 18);
    for (sample, offset, block) in entries {
        payload.extend_from_slice(&sample.to_be_bytes());
        payload.extend_from_slice(&offset.to_be_bytes());
        payload.extend_from_slice(&block.to_be_bytes());
    }
    payload
}

fn block_size_bounds(frames: &[u32]) -> (u16, u16) {
    if frames.is_empty() {
        return (16, 16);
    }
    let last = frames[frames.len() - 1];
    let min = if frames.len() > 1 {
        frames[..frames.len() - 1]
            .iter()
            .copied()
            .min()
            .unwrap_or(last)
    } else {
        last
    };
    let max = frames.iter().copied().max().unwrap_or(last);
    (min.max(16) as u16, max.max(16) as u16)
}

/// Fully prepared stream: metadata plus encoded frames.
pub(crate) struct PreparedFlac {
    /// STREAMINFO values for the stream.
    pub info: StreamInfo,
    /// Encoded frames in stream order.
    pub frames: Vec<Vec<u8>>,
    /// `(first_sample, stream_offset, block_size)` for every frame.
    pub frame_meta: Vec<(u64, u64, u32)>,
}

/// Validates the input, computes the MD5 signature, and encodes every frame.
pub(crate) fn prepare_stream(
    config: &FlacEncoderConfig,
    planes: &[Vec<i32>],
) -> Result<PreparedFlac, FlacError> {
    config.validate()?;
    if planes.len() != usize::from(config.channels) {
        return Err(FlacError::EncodeError("channel count mismatch"));
    }
    let total_samples = planes.first().map_or(0, Vec::len);
    if planes.iter().any(|plane| plane.len() != total_samples) {
        return Err(FlacError::EncodeError(
            "plane lengths differ within the stream",
        ));
    }
    if total_samples as u64 > 0x0000_000f_ffff_ffff {
        return Err(FlacError::EncodeError(
            "sample count exceeds the STREAMINFO limit",
        ));
    }
    validate_samples(planes, config.bits_per_sample)?;

    let mut md5 = Md5::new();
    md5.update(&interleaved_bytes(planes, config.bits_per_sample));
    let md5 = md5.finalize();

    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut frame_meta: Vec<(u64, u64, u32)> = Vec::new();
    let mut block_sizes: Vec<u32> = Vec::new();
    let block_size = config.block_size as usize;
    let mut offset = 0u64;
    let mut start = 0usize;
    while start < total_samples {
        let count = block_size.min(total_samples - start);
        let refs: Vec<&[i32]> = planes
            .iter()
            .map(|plane| &plane[start..start + count])
            .collect();
        let frame = encode_frame(config, &refs, frames.len() as u64)?;
        frame_meta.push((start as u64, offset, count as u32));
        offset += frame.len() as u64;
        block_sizes.push(count as u32);
        frames.push(frame);
        start += count;
    }

    let (min_block_size, max_block_size) = block_size_bounds(&block_sizes);
    let min_frame_size = frames.iter().map(Vec::len).min().unwrap_or(0) as u32;
    let max_frame_size = frames.iter().map(Vec::len).max().unwrap_or(0) as u32;
    if max_frame_size > 0x00ff_ffff {
        return Err(FlacError::EncodeError("frame exceeds 24-bit size field"));
    }
    let info = StreamInfo {
        min_block_size,
        max_block_size,
        min_frame_size,
        max_frame_size,
        sample_rate: config.sample_rate,
        channels: config.channels,
        bits_per_sample: config.bits_per_sample,
        total_samples: (total_samples != 0).then_some(total_samples as u64),
        md5,
    };
    Ok(PreparedFlac {
        info,
        frames,
        frame_meta,
    })
}

/// Encodes planar PCM into a complete native FLAC stream.
///
/// The whole input is buffered internally so that STREAMINFO can carry exact
/// frame sizes, the total sample count, and the MD5 signature, and so that a
/// SEEKTABLE can be written.  `planes` must contain exactly `config.channels`
/// planes of equal length.
pub fn encode_flac<W: Write>(
    mut writer: W,
    config: &FlacEncoderConfig,
    planes: &[Vec<i32>],
) -> Result<StreamInfo, FlacError> {
    let prepared = prepare_stream(config, planes)?;
    let info = prepared.info.clone();
    writer.write_all(b"fLaC")?;
    let seektable = seektable_payload(&prepared.frame_meta, config.sample_rate);
    if seektable.is_empty() {
        write_metadata_block(&mut writer, 0, &streaminfo_payload(&info), true)?;
    } else {
        // STREAMINFO must be first; SEEKTABLE follows.
        write_metadata_block(&mut writer, 0, &streaminfo_payload(&info), false)?;
        write_metadata_block(&mut writer, 3, &seektable, true)?;
    }
    for frame in &prepared.frames {
        writer.write_all(frame)?;
    }
    writer.flush()?;
    Ok(info)
}

/// Encodes planar PCM into an in-memory native FLAC stream.
pub fn encode_flac_to_vec(
    config: &FlacEncoderConfig,
    planes: &[Vec<i32>],
) -> Result<(Vec<u8>, StreamInfo), FlacError> {
    let mut out = Vec::new();
    let info = encode_flac(&mut out, config, planes)?;
    Ok((out, info))
}

/// Streaming FLAC encoder.
///
/// The stream header is written when the encoder is created, so a non-seekable
/// sink receives a valid stream whose STREAMINFO leaves the MD5, frame sizes,
/// and total sample count unknown (all zero).  Use
/// [`FlacEncoder::finish_seekable`] on a seekable sink to back-patch those
/// fields, or [`encode_flac`] for a fully populated in-memory/file encode.
pub struct FlacEncoder<W: Write> {
    writer: W,
    config: FlacEncoderConfig,
    pending: Vec<Vec<i32>>,
    buffered: usize,
    frame_index: u64,
    next_sample: u64,
    md5: Md5,
    min_frame_size: u32,
    max_frame_size: u32,
    min_block_size: u16,
    max_block_size: u16,
    finished: bool,
}

impl<W: Write> FlacEncoder<W> {
    /// Creates an encoder and writes the stream marker and a placeholder
    /// STREAMINFO block.
    pub fn new(mut writer: W, config: FlacEncoderConfig) -> Result<Self, FlacError> {
        config.validate()?;
        let info = StreamInfo {
            min_block_size: config.block_size.min(65_535) as u16,
            max_block_size: config.block_size.min(65_535) as u16,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate: config.sample_rate,
            channels: config.channels,
            bits_per_sample: config.bits_per_sample,
            total_samples: None,
            md5: [0; 16],
        };
        writer.write_all(b"fLaC")?;
        write_metadata_block(&mut writer, 0, &streaminfo_payload(&info), true)?;
        Ok(Self {
            writer,
            pending: vec![Vec::new(); usize::from(config.channels)],
            config,
            buffered: 0,
            frame_index: 0,
            next_sample: 0,
            md5: Md5::new(),
            min_frame_size: u32::MAX,
            max_frame_size: 0,
            min_block_size: u16::MAX,
            max_block_size: 0,
            finished: false,
        })
    }

    /// Number of samples written so far.
    #[must_use]
    pub fn samples_written(&self) -> u64 {
        self.next_sample
    }

    /// Adds one PCM frame to the encoder.
    pub fn write_frame(&mut self, frame: &PcmFrame) -> Result<(), FlacError> {
        if self.finished {
            return Err(FlacError::EncodeError("encoder is already finished"));
        }
        if frame.sample_rate != self.config.sample_rate
            || frame.channels != self.config.channels
            || frame.bits_per_sample != self.config.bits_per_sample
        {
            return Err(FlacError::EncodeError(
                "frame stream properties do not match the encoder configuration",
            ));
        }
        if frame.planes.len() != usize::from(self.config.channels) {
            return Err(FlacError::EncodeError("frame channel count mismatch"));
        }
        validate_samples(&frame.planes, self.config.bits_per_sample)?;
        self.md5.update(&interleaved_bytes(
            &frame.planes,
            self.config.bits_per_sample,
        ));
        self.next_sample = self
            .next_sample
            .checked_add(frame.block_size() as u64)
            .ok_or(FlacError::EncodeError("sample count overflow"))?;

        let mut offset = 0usize;
        let block_size = self.config.block_size as usize;
        while offset < frame.block_size() {
            let space = block_size - self.buffered;
            let take = space.min(frame.block_size() - offset);
            for (channel, plane) in frame.planes.iter().enumerate() {
                self.pending[channel].extend_from_slice(&plane[offset..offset + take]);
            }
            self.buffered += take;
            offset += take;
            if self.buffered == block_size {
                self.flush_block(block_size)?;
            }
        }
        Ok(())
    }

    /// Flushes a complete block from the pending buffer.
    fn flush_block(&mut self, count: usize) -> Result<(), FlacError> {
        let refs: Vec<&[i32]> = self.pending.iter().map(|plane| &plane[..count]).collect();
        let frame = encode_frame(&self.config, &refs, self.frame_index)?;
        self.writer.write_all(&frame)?;
        self.min_frame_size = self.min_frame_size.min(frame.len() as u32);
        self.max_frame_size = self.max_frame_size.max(frame.len() as u32);
        self.min_block_size = self.min_block_size.min(count as u16);
        self.max_block_size = self.max_block_size.max(count as u16);
        for plane in &mut self.pending {
            plane.clear();
        }
        self.buffered = 0;
        self.frame_index += 1;
        Ok(())
    }

    /// Finishes the stream and returns the underlying writer.
    ///
    /// The STREAMINFO block keeps zero MD5/frame-size/total fields.
    pub fn finish(mut self) -> Result<W, FlacError> {
        self.finish_inner()?;
        self.writer.flush()?;
        Ok(self.writer)
    }

    fn finish_inner(&mut self) -> Result<(), FlacError> {
        if self.finished {
            return Ok(());
        }
        if self.buffered > 0 {
            let count = self.buffered;
            self.flush_block(count)?;
        }
        self.finished = true;
        Ok(())
    }
}

impl<W: Write + Seek> FlacEncoder<W> {
    /// Finishes the stream, back-patches STREAMINFO, and returns the writer.
    ///
    /// The MD5 signature is only known once every sample has been written, so
    /// this method seeks back to the fixed STREAMINFO payload offset.
    pub fn finish_seekable(mut self) -> Result<W, FlacError> {
        self.finish_inner()?;
        let md5 = self.md5.clone().finalize();
        let (min_block_size, max_block_size) = if self.max_block_size == 0 {
            (16, 16)
        } else {
            (self.min_block_size.max(16), self.max_block_size.max(16))
        };
        let info = StreamInfo {
            min_block_size,
            max_block_size,
            min_frame_size: if self.min_frame_size == u32::MAX {
                0
            } else {
                self.min_frame_size
            },
            max_frame_size: self.max_frame_size,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            total_samples: (self.next_sample != 0).then_some(self.next_sample),
            md5,
        };
        self.writer.seek(SeekFrom::Start(8))?;
        self.writer.write_all(&streaminfo_payload(&info))?;
        self.writer.flush()?;
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::flac::{FlacDecoder, FlacEncoder, FlacEncoderConfig};

    fn signal(channels: usize, bits: u8, len: usize, seed: u64) -> Vec<Vec<i32>> {
        let limit = 1i64 << (bits - 1);
        let mut state = seed;
        (0..channels)
            .map(|channel| {
                (0..len)
                    .map(|index| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let noise = ((state >> 33) as i64) % (limit / 4).max(1);
                        let base = (index as i64 * 37 + channel as i64 * 11) % limit - limit / 2;
                        (base / 2 + noise / 2).clamp(-limit, limit - 1) as i32
                    })
                    .collect()
            })
            .collect()
    }

    fn config(channels: u8, bits: u8, block_size: u32, level: u8) -> FlacEncoderConfig {
        FlacEncoderConfig {
            sample_rate: 48_000,
            channels,
            bits_per_sample: bits,
            block_size,
            compression_level: level,
        }
    }

    fn decode_all(bytes: &[u8]) -> (Vec<Vec<i32>>, u64, [u8; 16]) {
        let mut decoder = FlacDecoder::new(Cursor::new(bytes.to_vec())).unwrap();
        let info = decoder.stream_info().clone();
        let channels = usize::from(info.channels);
        let mut planes = vec![Vec::new(); channels];
        let mut samples = 0u64;
        while let Some(frame) = decoder.next_frame().unwrap() {
            assert_eq!(frame.planes.len(), channels);
            assert_eq!(frame.sample_number, samples);
            for (dst, src) in planes.iter_mut().zip(&frame.planes) {
                dst.extend_from_slice(src);
            }
            samples += frame.block_size() as u64;
        }
        (planes, samples, info.md5)
    }

    fn md5_of(planes: &[Vec<i32>], bits: u8) -> [u8; 16] {
        let mut hasher = super::Md5::new();
        hasher.update(&super::interleaved_bytes(planes, bits));
        hasher.finalize()
    }

    fn roundtrip(bits: u8, channels: u8, block_size: u32, level: u8, len: usize) {
        let planes = signal(usize::from(channels), bits, len, 0x1234 + u64::from(bits));
        let cfg = config(channels, bits, block_size, level);
        let (bytes, info) = encode_flac_to_vec(&cfg, &planes).unwrap();
        assert_eq!(info.total_samples, Some(len as u64));
        assert_eq!(info.channels, channels);
        assert_eq!(info.bits_per_sample, bits);
        assert_eq!(info.sample_rate, 48_000);
        assert!(info.max_frame_size > 0);
        assert!(info.min_frame_size > 0);
        assert_eq!(info.md5, md5_of(&planes, bits));
        let (decoded, samples, md5) = decode_all(&bytes);
        assert_eq!(samples, len as u64);
        assert_eq!(md5, info.md5);
        assert_eq!(
            decoded, planes,
            "bits={bits} channels={channels} block={block_size} level={level} len={len}"
        );
    }

    #[test]
    fn roundtrips_common_depths_and_layouts() {
        for bits in [8u8, 12, 16, 20, 24, 32] {
            roundtrip(bits, 1, 1024, 5, 4_000);
        }
        for (bits, channels) in [(8u8, 2u8), (16, 2), (24, 2), (32, 2)] {
            roundtrip(bits, channels, 4096, 5, 9_000);
        }
        roundtrip(16, 3, 576, 5, 2_000);
        roundtrip(16, 6, 1024, 5, 3_000);
        roundtrip(24, 8, 192, 5, 1_500);
    }

    #[test]
    fn roundtrips_all_compression_levels() {
        for level in 0..=8u8 {
            roundtrip(16, 2, 1024, level, 3_000);
        }
    }

    #[test]
    fn roundtrips_constant_silence_and_short_streams() {
        for len in [1usize, 2, 15, 16, 17, 100] {
            roundtrip(16, 1, 4096, 5, len);
        }
        for bits in [8u8, 16, 24, 32] {
            let channels = 2usize;
            let len = 1_000;
            let limit = 1i64 << (bits - 1);
            let planes: Vec<Vec<i32>> = (0..channels)
                .map(|channel| vec![if channel == 0 { 0 } else { (limit - 1) as i32 }; len])
                .collect();
            let cfg = config(2, bits, 512, 5);
            let (bytes, _) = encode_flac_to_vec(&cfg, &planes).unwrap();
            let (decoded, _, _) = decode_all(&bytes);
            assert_eq!(decoded, planes);
        }
        // All-zero stereo.
        let planes = vec![vec![0i32; 777], vec![0i32; 777]];
        let (bytes, _) = encode_flac_to_vec(&config(2, 16, 256, 5), &planes).unwrap();
        assert_eq!(decode_all(&bytes).0, planes);
    }

    #[test]
    fn streaming_seekable_writer_backpatches_streaminfo() {
        let planes = signal(2, 16, 5_000, 42);
        let cfg = config(2, 16, 512, 5);
        let mut encoder = FlacEncoder::new(Cursor::new(Vec::new()), cfg).unwrap();
        let mut offset = 0usize;
        for chunk in [97usize, 300, 512, 1_000] {
            let end = (offset + chunk).min(planes[0].len());
            let frame = crate::PcmFrame {
                sample_rate: 48_000,
                channels: 2,
                bits_per_sample: 16,
                frame_number: 0,
                sample_number: offset as u64,
                planes: vec![
                    planes[0][offset..end].to_vec(),
                    planes[1][offset..end].to_vec(),
                ],
            };
            encoder.write_frame(&frame).unwrap();
            offset = end;
            if offset == planes[0].len() {
                break;
            }
        }
        let remaining = planes[0].len() - offset;
        if remaining > 0 {
            let frame = crate::PcmFrame {
                sample_rate: 48_000,
                channels: 2,
                bits_per_sample: 16,
                frame_number: 0,
                sample_number: offset as u64,
                planes: vec![planes[0][offset..].to_vec(), planes[1][offset..].to_vec()],
            };
            encoder.write_frame(&frame).unwrap();
        }
        let cursor = encoder.finish_seekable().unwrap();
        let bytes = cursor.into_inner();
        let (decoded, samples, md5) = decode_all(&bytes);
        assert_eq!(samples, planes[0].len() as u64);
        assert_eq!(decoded, planes);
        assert_eq!(md5, md5_of(&planes, 16));
        let info = FlacDecoder::new(Cursor::new(bytes)).unwrap();
        assert_eq!(
            info.stream_info().total_samples,
            Some(planes[0].len() as u64)
        );
    }

    #[test]
    fn streaming_pipe_writer_stays_decodable() {
        let planes = signal(1, 24, 2_000, 7);
        let mut encoder = FlacEncoder::new(Vec::new(), config(1, 24, 333, 4)).unwrap();
        let frame = crate::PcmFrame {
            sample_rate: 48_000,
            channels: 1,
            bits_per_sample: 24,
            frame_number: 0,
            sample_number: 0,
            planes: planes.clone(),
        };
        encoder.write_frame(&frame).unwrap();
        let bytes = encoder.finish().unwrap();
        let (decoded, samples, _) = decode_all(&bytes);
        assert_eq!(samples, planes[0].len() as u64);
        assert_eq!(decoded, planes);
    }

    #[test]
    fn decoder_verifies_md5_and_allows_opt_out() {
        let planes = signal(2, 16, 1_000, 99);
        let (mut bytes, _) = encode_flac_to_vec(&config(2, 16, 256, 5), &planes).unwrap();
        // STREAMINFO payload starts at byte 8; the MD5 field is at payload
        // offset 18, so the first digest byte is at file offset 26.
        bytes[26] ^= 0x01;
        let mut decoder = FlacDecoder::new(Cursor::new(bytes.clone())).unwrap();
        let mut error = None;
        loop {
            match decoder.next_frame() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(value) => {
                    error = Some(value);
                    break;
                }
            }
        }
        assert!(matches!(error, Some(FlacError::Md5Mismatch { .. })));

        let mut decoder = FlacDecoder::new(Cursor::new(bytes)).unwrap();
        decoder.set_verify_md5(false);
        while decoder.next_frame().unwrap().is_some() {}
    }

    #[test]
    fn decoder_skips_verification_for_an_unknown_md5() {
        let planes = signal(1, 16, 500, 3);
        let (mut bytes, _) = encode_flac_to_vec(&config(1, 16, 128, 5), &planes).unwrap();
        bytes[26..42].fill(0);
        let (decoded, samples, md5) = decode_all(&bytes);
        assert_eq!(samples, 500);
        assert_eq!(decoded, planes);
        assert_eq!(md5, [0; 16]);
    }

    #[test]
    fn seeking_lands_on_the_requested_sample() {
        let planes = signal(2, 16, 144_000, 77);
        let (bytes, _) = encode_flac_to_vec(&config(2, 16, 4096, 5), &planes).unwrap();
        let (reference, total, _) = decode_all(&bytes);
        assert_eq!(total, 144_000);
        let mut decoder = FlacDecoder::new(Cursor::new(bytes.clone())).unwrap();
        for target in [0u64, 1, 4_095, 4_096, 48_000, 48_001, 100_000, 143_999] {
            let start = decoder.seek_to_sample(target).unwrap();
            assert!(start <= target, "target {target} start {start}");
            let frame = decoder.next_frame().unwrap().unwrap();
            assert_eq!(frame.sample_number, start);
            for (channel, plane) in frame.planes.iter().enumerate() {
                let range = start as usize..start as usize + frame.block_size();
                assert_eq!(plane, &reference[channel][range]);
            }
        }

        // Seeking past the end clamps to the last sample and yields the last
        // frame, then a clean end of stream.
        let start = decoder.seek_to_sample(1_000_000).unwrap();
        let frame = decoder.next_frame().unwrap().unwrap();
        assert_eq!(frame.sample_number, start);
        assert!(start + frame.block_size() as u64 > 143_999);
        assert!(decoder.next_frame().unwrap().is_none());

        // A short stream has only the initial seek point; seeking still works
        // by decoding forward.
        let (short, _) =
            encode_flac_to_vec(&config(1, 16, 512, 5), &signal(1, 16, 4_000, 5)).unwrap();
        let mut decoder = FlacDecoder::new(Cursor::new(short)).unwrap();
        let start = decoder.seek_to_sample(2_000).unwrap();
        assert!(start <= 2_000);
        assert!(decoder.next_frame().unwrap().is_some());
    }

    #[test]
    fn rejects_samples_outside_the_bit_depth() {
        let cfg = config(1, 8, 256, 5);
        let planes = vec![vec![128i32; 10]];
        assert!(matches!(
            encode_flac_to_vec(&cfg, &planes),
            Err(FlacError::EncodeError(_))
        ));
    }

    #[test]
    fn rejects_invalid_configuration() {
        let mut cfg = config(2, 16, 4096, 5);
        cfg.block_size = 15;
        assert!(matches!(cfg.validate(), Err(FlacError::InvalidConfig(_))));
        cfg.block_size = 4096;
        cfg.channels = 0;
        assert!(matches!(cfg.validate(), Err(FlacError::InvalidConfig(_))));
        cfg.channels = 2;
        cfg.bits_per_sample = 33;
        assert!(matches!(cfg.validate(), Err(FlacError::InvalidConfig(_))));
    }
}
