// SPDX-License-Identifier: Apache-2.0
//! Minimal command-line bridge for every `rustmpeg-core` primitive.
//!
//! The binary is intentionally a thin, dependency-free wrapper: it parses
//! arguments, calls exactly one core API per command, and prints the result.
//! It is meant for inspection, scripting, and smoke testing, not as a media
//! player.

#![forbid(unsafe_code)]

use std::fs;
use std::io::Read;
use std::process::ExitCode;
use std::str::FromStr;

use rustmpeg_core::{
    AudioFormat, AudioFrame, Buffer, ChromaLocation, ColorDescription, ColorPrimaries, ColorRange,
    FrameRate, MAX_CHANNELS, MAX_DIMENSION, MAX_LUMA_SAMPLES, MatrixCoefficients, MediaDuration,
    MediaTime, Packet, PacketFlags, PixelFormat, Rational, SampleFormat, Timebase,
    TransferFunction, VideoFrame, clamp_u8, limits, sad_u8, sse_u8,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const FRAME_LIMIT: usize = 64 * 1024 * 1024;
const AUDIO_LIMIT: usize = 64 * 1024 * 1024;

const USAGE: &str = "\
rustmpeg-core — minimal CLI bridge for rustmpeg-core

Usage: rustmpeg-core COMMAND [ARGS...]

Commands:
  version
  buffer <hex|@file|-> [--slice START:END] [--advance N] [--truncate N] [--split AT]
  time rational <NUM> <DEN>
  time rescale <TICKS> <FROM_NUM> <FROM_DEN> <TO_NUM> <TO_DEN>
  time seconds <SECONDS> <NUM> <DEN>
  time add <TICKS_A> <NUM_A> <DEN_A> <TICKS_B> <NUM_B> <DEN_B>
  time duration <TICKS> <NUM> <DEN> [TO_NUM TO_DEN]
  time frame-rate <NUM> <DEN>
  pixel list
  pixel info <FORMAT> <WIDTH> <HEIGHT> <BIT_DEPTH>
  pixel stride <STRIDE> <WIDTH> <BYTES_PER_SAMPLE>
  color list
  color show <PRESET>
  color parse <RANGE> <PRIMARIES> <TRANSFER> <MATRIX> <CHROMA>
  frame info <FORMAT> <BIT_DEPTH> <WIDTH> <HEIGHT> [PRESET]
  audio list
  audio info <FORMAT> <RATE> <CHANNELS> <SAMPLES>
  audio frame <FORMAT> <RATE> <CHANNELS> <BYTES>
  packet <hex|@file|-> [--stream N] [--timebase NUM DEN]
         [--pts T] [--dts T] [--duration T] [--flags key,discard,eos,corrupt]
  simd sad <hex|@file|-> <hex|@file|->
  simd sse <hex|@file|-> <hex|@file|->
  simd clamp <I16>
  limits show
  limits check <WIDTH> <HEIGHT>
  limits plane <WIDTH> <HEIGHT> <BYTES_PER_SAMPLE>
  limits stride <STRIDE> <WIDTH> <BYTES_PER_SAMPLE>

Formats: i400, i420, i422, i444, rgb8, rgba8, bgra8
Samples: u8, s16, s24, s32, f32, f64
Presets: bt709-limited, bt2020-pq, bt2020-hlg, default
Data arguments accept hexadecimal (0x prefix and ':' separators allowed),
@path for a file, or '-' for standard input.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let result = dispatch(&args);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), String> {
    match args[0].as_str() {
        "version" | "--version" | "-V" => {
            println!("rustmpeg-core {VERSION}");
            Ok(())
        }
        "buffer" => cmd_buffer(&args[1..]),
        "time" => cmd_time(&args[1..]),
        "pixel" => cmd_pixel(&args[1..]),
        "color" => cmd_color(&args[1..]),
        "frame" => cmd_frame(&args[1..]),
        "audio" => cmd_audio(&args[1..]),
        "packet" => cmd_packet(&args[1..]),
        "simd" => cmd_simd(&args[1..]),
        "limits" => cmd_limits(&args[1..]),
        other => Err(format!(
            "unknown command '{other}'; run 'rustmpeg-core help'"
        )),
    }
}

fn cmd_buffer(args: &[String]) -> Result<(), String> {
    let data = args
        .first()
        .ok_or("usage: rustmpeg-core buffer <hex|@file|-> [options]")?;
    let mut buffer = Buffer::from_vec(load(data)?);
    let mut slice: Option<(usize, usize)> = None;
    let mut split: Option<usize> = None;
    let mut advance: Option<usize> = None;
    let mut truncate: Option<usize> = None;

    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--slice" => {
                let value = value(args, index + 1, option)?;
                slice = Some(parse_byte_range(value)?);
                index += 2;
            }
            "--split" => {
                split = Some(parse(value(args, index + 1, option)?, "split offset")?);
                index += 2;
            }
            "--advance" => {
                advance = Some(parse(value(args, index + 1, option)?, "advance count")?);
                index += 2;
            }
            "--truncate" => {
                truncate = Some(parse(value(args, index + 1, option)?, "truncate length")?);
                index += 2;
            }
            other => return Err(format!("unknown buffer option '{other}'")),
        }
    }

    if let Some((start, end)) = slice {
        buffer = core(buffer.slice(start..end))?;
    }
    if let Some(count) = advance {
        core(buffer.advance(count))?;
    }
    if let Some(length) = truncate {
        core(buffer.truncate(length))?;
    }
    if let Some(at) = split {
        let right = core(buffer.split_off(at))?;
        print_buffer("left", &buffer);
        print_buffer("right", &right);
    } else {
        print_buffer("buffer", &buffer);
    }
    Ok(())
}

fn cmd_time(args: &[String]) -> Result<(), String> {
    let subcommand = args.first().ok_or(
        "usage: rustmpeg-core time <rational|rescale|seconds|add|duration|frame-rate> ...",
    )?;
    match subcommand.as_str() {
        "rational" => {
            let numerator = parse(value(args, 1, "rational")?, "numerator")?;
            let denominator = parse(value(args, 2, "rational")?, "denominator")?;
            let rational = core(Rational::new(numerator, denominator))?;
            println!("value {}", rational_text(rational));
            println!("reduced {}", rational_text(rational.reduced()));
            println!("inverse {}", rational_text(core(rational.invert())?));
            println!("seconds {:.9}", rational.to_f64());
            Ok(())
        }
        "rescale" => {
            let ticks = parse(value(args, 1, "rescale")?, "ticks")?;
            let from = timebase(args, 2)?;
            let to = timebase(args, 4)?;
            let time = core(MediaTime::new(ticks, from).rescale(to))?;
            print_time("time", time);
            Ok(())
        }
        "seconds" => {
            let seconds = parse(value(args, 1, "seconds")?, "seconds")?;
            let base = timebase(args, 2)?;
            let time = core(MediaTime::from_seconds(seconds, base))?;
            print_time("time", time);
            Ok(())
        }
        "add" => {
            let ticks_a = parse(value(args, 1, "add")?, "first ticks")?;
            let base_a = timebase(args, 2)?;
            let ticks_b = parse(value(args, 4, "add")?, "second ticks")?;
            let base_b = timebase(args, 5)?;
            let a = MediaTime::new(ticks_a, base_a);
            let b = MediaTime::new(ticks_b, base_b);
            print_time("sum", core(a.checked_add(b))?);
            Ok(())
        }
        "duration" => {
            let ticks = parse(value(args, 1, "duration")?, "ticks")?;
            let base = timebase(args, 2)?;
            let duration = if args.len() > 4 {
                let target = timebase(args, 4)?;
                core(MediaDuration::new(ticks, base).rescale(target))?
            } else {
                MediaDuration::new(ticks, base)
            };
            print_duration("duration", duration);
            Ok(())
        }
        "frame-rate" => {
            let numerator = parse(value(args, 1, "frame-rate")?, "numerator")?;
            let denominator = parse(value(args, 2, "frame-rate")?, "denominator")?;
            let rate = core(FrameRate::new(numerator, denominator))?;
            println!("value {}/{}", rate.numerator(), rate.denominator());
            println!("fps {:.6}", rate.to_f64());
            println!("timebase {}", rational_text(rate.timebase()));
            Ok(())
        }
        other => Err(format!("unknown time subcommand '{other}'")),
    }
}

fn cmd_pixel(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core pixel <list|info|stride> ...")?;
    match subcommand.as_str() {
        "list" => {
            for format in PIXEL_FORMATS {
                let (sub_x, sub_y) = format.subsampling();
                println!(
                    "{:<6} planes={} subsampling={}x{} planar={} packed={}",
                    pixel_name(format),
                    format.plane_count(),
                    1 << sub_x,
                    1 << sub_y,
                    format.is_planar(),
                    format.is_packed()
                );
            }
            Ok(())
        }
        "info" => {
            let format = parse_pixel(value(args, 1, "info")?)?;
            let width: u32 = parse(value(args, 2, "info")?, "width")?;
            let height: u32 = parse(value(args, 3, "info")?, "height")?;
            let bit_depth: u8 = parse(value(args, 4, "info")?, "bit depth")?;
            print_pixel_info(format, width, height, bit_depth)
        }
        "stride" => {
            let stride: usize = parse(value(args, 1, "stride")?, "stride")?;
            let width: u32 = parse(value(args, 2, "stride")?, "width")?;
            let bytes: usize = parse(value(args, 3, "stride")?, "bytes per sample")?;
            core(limits::validate_stride(stride, width, bytes))?;
            println!("ok stride {stride} holds {width} samples of {bytes} bytes");
            Ok(())
        }
        other => Err(format!("unknown pixel subcommand '{other}'")),
    }
}

fn cmd_color(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core color <list|show|parse> ...")?;
    match subcommand.as_str() {
        "list" => {
            for (name, color) in COLOR_PRESETS {
                println!("{name}");
                print_color(&color);
                println!();
            }
            Ok(())
        }
        "show" => {
            let preset = value(args, 1, "show")?;
            print_color(&color_preset(preset)?);
            Ok(())
        }
        "parse" => {
            let color = ColorDescription {
                range: parse_color_range(value(args, 1, "parse")?)?,
                primaries: parse_primaries(value(args, 2, "parse")?)?,
                transfer: parse_transfer(value(args, 3, "parse")?)?,
                matrix: parse_matrix(value(args, 4, "parse")?)?,
                chroma_location: parse_chroma(value(args, 5, "parse")?)?,
            };
            print_color(&color);
            Ok(())
        }
        other => Err(format!("unknown color subcommand '{other}'")),
    }
}

fn cmd_frame(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core frame info <FORMAT> <BIT_DEPTH> <WIDTH> <HEIGHT> [PRESET]")?;
    if subcommand != "info" {
        return Err(format!("unknown frame subcommand '{subcommand}'"));
    }
    let format = parse_pixel(value(args, 1, "info")?)?;
    let bit_depth: u8 = parse(value(args, 2, "info")?, "bit depth")?;
    let width: u32 = parse(value(args, 3, "info")?, "width")?;
    let height: u32 = parse(value(args, 4, "info")?, "height")?;
    let color = match args.get(5) {
        Some(preset) => color_preset(preset)?,
        None => ColorDescription::BT709_LIMITED,
    };

    let total = pixel_total(format, width, height, bit_depth)?;
    if total > FRAME_LIMIT {
        return Err(format!(
            "refusing to allocate {total} bytes (CLI cap is {FRAME_LIMIT})"
        ));
    }
    let frame = core(VideoFrame::new(format, bit_depth, width, height, color))?;

    println!("format {}", pixel_name(format));
    println!("bit_depth {bit_depth}");
    println!("width {width}");
    println!("height {height}");
    print_color(&color);
    println!("valid {}", frame.is_valid());
    for index in 0..frame.plane_count() {
        let plane = &frame.planes[index];
        println!(
            "plane{index} {}x{} stride={} row_bytes={} bytes={}",
            plane.width,
            plane.height,
            plane.stride,
            core(frame.row_bytes(index))?,
            plane.data.len()
        );
    }
    println!("total_bytes {total}");
    Ok(())
}

fn cmd_audio(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core audio <list|info|frame> ...")?;
    match subcommand.as_str() {
        "list" => {
            for format in SAMPLE_FORMATS {
                println!(
                    "{:<3} bytes={} float={}",
                    sample_name(format),
                    format.bytes_per_sample(),
                    format.is_float()
                );
            }
            Ok(())
        }
        "info" => {
            let format = parse_audio(args, 1)?;
            let samples: usize = parse(value(args, 4, "info")?, "sample count")?;
            println!("format {}", sample_name(format.sample_format));
            println!("sample_rate {}", format.sample_rate);
            println!("channels {}", format.channels);
            println!(
                "bytes_per_sample {}",
                format.sample_format.bytes_per_sample()
            );
            println!("bytes_per_frame {}", core(format.bytes_per_frame())?);
            let bytes = core(format.bytes_for_samples(samples))?;
            println!("samples {samples}");
            println!("bytes {bytes}");
            println!(
                "duration {:.6}",
                samples as f64 / f64::from(format.sample_rate)
            );
            Ok(())
        }
        "frame" => {
            let format = parse_audio(args, 1)?;
            let bytes: usize = parse(value(args, 4, "frame")?, "byte count")?;
            if bytes > AUDIO_LIMIT {
                return Err(format!(
                    "refusing to allocate {bytes} bytes (CLI cap is {AUDIO_LIMIT})"
                ));
            }
            let frame = core(AudioFrame::new(format, Buffer::from_vec(vec![0; bytes])))?;
            println!("format {}", sample_name(format.sample_format));
            println!("bytes {}", frame.data.len());
            println!("sample_count {}", core(frame.sample_count())?);
            println!("duration {:.6}", core(frame.duration_seconds())?);
            Ok(())
        }
        other => Err(format!("unknown audio subcommand '{other}'")),
    }
}

fn cmd_packet(args: &[String]) -> Result<(), String> {
    let data = args
        .first()
        .ok_or("usage: rustmpeg-core packet <hex|@file|-> [options]")?;
    let mut packet = Packet::new(0, Buffer::from_vec(load(data)?));
    let mut base = core(Rational::new(1, 1))?;
    let mut pts = None;
    let mut dts = None;
    let mut duration = None;
    let mut flags = PacketFlags::default();

    let mut index = 1;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--stream" => {
                packet =
                    packet.with_stream_index(parse(value(args, index + 1, option)?, "stream")?);
                index += 2;
            }
            "--timebase" => {
                let numerator = parse(value(args, index + 1, option)?, "timebase numerator")?;
                let denominator = parse(value(args, index + 2, option)?, "timebase denominator")?;
                base = core(Rational::new(numerator, denominator))?;
                index += 3;
            }
            "--pts" => {
                pts = Some(parse(value(args, index + 1, option)?, "pts")?);
                index += 2;
            }
            "--dts" => {
                dts = Some(parse(value(args, index + 1, option)?, "dts")?);
                index += 2;
            }
            "--duration" => {
                duration = Some(parse(value(args, index + 1, option)?, "duration")?);
                index += 2;
            }
            "--flags" => {
                flags = parse_flags(value(args, index + 1, option)?)?;
                index += 2;
            }
            other => return Err(format!("unknown packet option '{other}'")),
        }
    }

    if let Some(ticks) = pts {
        packet = packet.with_pts(MediaTime::new(ticks, base));
    }
    if let Some(ticks) = dts {
        packet = packet.with_dts(MediaTime::new(ticks, base));
    }
    if let Some(ticks) = duration {
        packet = packet.with_duration(MediaDuration::new(ticks, base));
    }
    packet = packet.with_flags(flags);

    println!("stream_index {}", packet.stream_index);
    println!("size {}", packet.len());
    println!("empty {}", packet.is_empty());
    println!("keyframe {}", packet.is_keyframe());
    print_optional_time("pts", packet.pts);
    print_optional_time("dts", packet.dts);
    print_optional_duration("duration", packet.duration);
    println!("flags {}", flags_text(packet.flags));
    println!("data.hex {}", hex_preview(packet.data.as_slice()));
    Ok(())
}

fn cmd_simd(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core simd <sad|sse|clamp> ...")?;
    match subcommand.as_str() {
        "sad" | "sse" => {
            let left = load(value(args, 1, subcommand)?)?;
            let right = load(value(args, 2, subcommand)?)?;
            let compared = left.len().min(right.len());
            let value = if subcommand == "sad" {
                sad_u8(&left, &right)
            } else {
                sse_u8(&left, &right)
            };
            println!("{subcommand} {value}");
            println!("compared {compared}");
            Ok(())
        }
        "clamp" => {
            let value: i16 = parse(value(args, 1, "clamp")?, "i16")?;
            println!("clamped {}", clamp_u8(value));
            Ok(())
        }
        other => Err(format!("unknown simd subcommand '{other}'")),
    }
}

fn cmd_limits(args: &[String]) -> Result<(), String> {
    let subcommand = args
        .first()
        .ok_or("usage: rustmpeg-core limits <show|check|plane|stride> ...")?;
    match subcommand.as_str() {
        "show" => {
            println!("max_dimension {MAX_DIMENSION}");
            println!("max_luma_samples {MAX_LUMA_SAMPLES}");
            println!("max_channels {MAX_CHANNELS}");
            Ok(())
        }
        "check" => {
            let width: u32 = parse(value(args, 1, "check")?, "width")?;
            let height: u32 = parse(value(args, 2, "check")?, "height")?;
            core(limits::validate_dimensions(width, height))?;
            println!("ok {width}x{height}");
            Ok(())
        }
        "plane" => {
            let width: u32 = parse(value(args, 1, "plane")?, "width")?;
            let height: u32 = parse(value(args, 2, "plane")?, "height")?;
            let bytes: usize = parse(value(args, 3, "plane")?, "bytes per sample")?;
            println!("bytes {}", core(limits::plane_bytes(width, height, bytes))?);
            Ok(())
        }
        "stride" => {
            let stride: usize = parse(value(args, 1, "stride")?, "stride")?;
            let width: u32 = parse(value(args, 2, "stride")?, "width")?;
            let bytes: usize = parse(value(args, 3, "stride")?, "bytes per sample")?;
            core(limits::validate_stride(stride, width, bytes))?;
            println!("ok stride {stride} holds {width} samples of {bytes} bytes");
            Ok(())
        }
        other => Err(format!("unknown limits subcommand '{other}'")),
    }
}

fn print_buffer(label: &str, buffer: &Buffer) {
    println!("{label}.len {}", buffer.len());
    println!("{label}.empty {}", buffer.is_empty());
    println!("{label}.hex {}", hex_preview(buffer.as_slice()));
}

fn print_time(label: &str, time: MediaTime) {
    println!("{label}.ticks {}", time.ticks());
    println!("{label}.timebase {}", rational_text(time.timebase()));
    println!("{label}.seconds {:.6}", time.to_seconds());
}

fn print_duration(label: &str, duration: MediaDuration) {
    println!("{label}.ticks {}", duration.ticks());
    println!("{label}.timebase {}", rational_text(duration.timebase()));
    println!("{label}.seconds {:.6}", duration.to_seconds());
}

fn print_optional_time(label: &str, time: Option<MediaTime>) {
    match time {
        Some(time) => print_time(label, time),
        None => println!("{label} -"),
    }
}

fn print_optional_duration(label: &str, duration: Option<MediaDuration>) {
    match duration {
        Some(duration) => print_duration(label, duration),
        None => println!("{label} -"),
    }
}

fn print_pixel_info(
    format: PixelFormat,
    width: u32,
    height: u32,
    bit_depth: u8,
) -> Result<(), String> {
    let (sub_x, sub_y) = format.subsampling();
    println!("format {}", pixel_name(format));
    println!("planes {}", format.plane_count());
    println!("subsampling {}x{}", 1 << sub_x, 1 << sub_y);
    println!("planar {}", format.is_planar());
    println!("packed {}", format.is_packed());
    if let Ok(bytes) = format.bytes_per_sample(bit_depth) {
        println!("bytes_per_sample {bytes}");
    }
    if let Ok(bytes) = format.packed_bytes_per_pixel() {
        println!("packed_bytes_per_pixel {bytes}");
    }
    for plane in 0..format.plane_count() {
        let (plane_width, plane_height) = core(format.plane_dimensions(width, height, plane))?;
        let bytes = core(format.plane_bytes(width, height, plane, bit_depth))?;
        println!("plane{plane}.width {plane_width}");
        println!("plane{plane}.height {plane_height}");
        println!("plane{plane}.bytes {bytes}");
    }
    println!(
        "total_bytes {}",
        pixel_total(format, width, height, bit_depth)?
    );
    Ok(())
}

fn pixel_total(
    format: PixelFormat,
    width: u32,
    height: u32,
    bit_depth: u8,
) -> Result<usize, String> {
    let mut total = 0_usize;
    for plane in 0..format.plane_count() {
        let bytes = core(format.plane_bytes(width, height, plane, bit_depth))?;
        total = total
            .checked_add(bytes)
            .ok_or("pixel byte count overflowed")?;
    }
    Ok(total)
}

fn print_color(color: &ColorDescription) {
    println!("range {}", range_name(color.range));
    println!("primaries {}", primaries_name(color.primaries));
    println!("transfer {}", transfer_name(color.transfer));
    println!("matrix {}", matrix_name(color.matrix));
    println!("chroma_location {}", chroma_name(color.chroma_location));
}

fn timebase(args: &[String], index: usize) -> Result<Timebase, String> {
    let numerator = parse(value(args, index, "timebase")?, "timebase numerator")?;
    let denominator = parse(value(args, index + 1, "timebase")?, "timebase denominator")?;
    core(Rational::new(numerator, denominator))
}

fn parse_audio(args: &[String], index: usize) -> Result<AudioFormat, String> {
    let format = parse_sample(value(args, index, "audio")?)?;
    let rate: u32 = parse(value(args, index + 1, "audio")?, "sample rate")?;
    let channels: u16 = parse(value(args, index + 2, "audio")?, "channel count")?;
    core(AudioFormat::new(format, rate, channels))
}

fn value<'a>(args: &'a [String], index: usize, option: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{option} needs a value"))
}

fn parse<T: FromStr>(text: &str, what: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("invalid {what}: {text}"))
}

fn parse_byte_range(text: &str) -> Result<(usize, usize), String> {
    let (start, end) = text
        .split_once(':')
        .ok_or_else(|| format!("range must be START:END, got {text}"))?;
    Ok((parse(start, "range start")?, parse(end, "range end")?))
}

fn core<T>(result: rustmpeg_core::Result<T>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn load(argument: &str) -> Result<Vec<u8>, String> {
    if argument == "-" {
        let mut data = Vec::new();
        std::io::stdin()
            .read_to_end(&mut data)
            .map_err(|error| format!("cannot read stdin: {error}"))?;
        Ok(data)
    } else if let Some(path) = argument.strip_prefix('@') {
        fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))
    } else {
        decode_hex(argument)
    }
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = text
        .chars()
        .filter(|character| !character.is_whitespace() && *character != ':' && *character != '_')
        .collect();
    let digits = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
        .unwrap_or(&cleaned);
    if !digits.len().is_multiple_of(2) {
        return Err("hex input must have an even number of digits".into());
    }
    let mut bytes = Vec::with_capacity(digits.len() / 2);
    for index in (0..digits.len()).step_by(2) {
        let byte = u8::from_str_radix(&digits[index..index + 2], 16)
            .map_err(|_| format!("invalid hex byte: {}", &digits[index..index + 2]))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn hex_preview(bytes: &[u8]) -> String {
    if bytes.len() <= 128 {
        hex(bytes)
    } else {
        format!("{}... ({} bytes)", hex(&bytes[..64]), bytes.len())
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut text, "{byte:02x}");
    }
    text
}

fn rational_text(rational: Rational) -> String {
    format!("{}/{}", rational.numerator(), rational.denominator())
}

const PIXEL_FORMATS: [PixelFormat; 7] = [
    PixelFormat::I400,
    PixelFormat::I420,
    PixelFormat::I422,
    PixelFormat::I444,
    PixelFormat::Rgb8,
    PixelFormat::Rgba8,
    PixelFormat::Bgra8,
];

const SAMPLE_FORMATS: [SampleFormat; 6] = [
    SampleFormat::U8,
    SampleFormat::S16,
    SampleFormat::S24,
    SampleFormat::S32,
    SampleFormat::F32,
    SampleFormat::F64,
];

const COLOR_PRESETS: [(&str, ColorDescription); 3] = [
    ("bt709-limited", ColorDescription::BT709_LIMITED),
    ("bt2020-pq", ColorDescription::BT2020_PQ),
    ("bt2020-hlg", ColorDescription::BT2020_HLG),
];

fn pixel_name(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::I400 => "i400",
        PixelFormat::I420 => "i420",
        PixelFormat::I422 => "i422",
        PixelFormat::I444 => "i444",
        PixelFormat::Rgb8 => "rgb8",
        PixelFormat::Rgba8 => "rgba8",
        PixelFormat::Bgra8 => "bgra8",
    }
}

fn parse_pixel(text: &str) -> Result<PixelFormat, String> {
    match text.to_ascii_lowercase().as_str() {
        "i400" | "gray" | "grey" => Ok(PixelFormat::I400),
        "i420" | "yuv420p" | "yuv420" => Ok(PixelFormat::I420),
        "i422" | "yuv422p" | "yuv422" => Ok(PixelFormat::I422),
        "i444" | "yuv444p" | "yuv444" => Ok(PixelFormat::I444),
        "rgb8" | "rgb" => Ok(PixelFormat::Rgb8),
        "rgba8" | "rgba" => Ok(PixelFormat::Rgba8),
        "bgra8" | "bgra" => Ok(PixelFormat::Bgra8),
        other => Err(format!("unknown pixel format '{other}'")),
    }
}

fn sample_name(format: SampleFormat) -> &'static str {
    match format {
        SampleFormat::U8 => "u8",
        SampleFormat::S16 => "s16",
        SampleFormat::S24 => "s24",
        SampleFormat::S32 => "s32",
        SampleFormat::F32 => "f32",
        SampleFormat::F64 => "f64",
    }
}

fn parse_sample(text: &str) -> Result<SampleFormat, String> {
    match text.to_ascii_lowercase().as_str() {
        "u8" => Ok(SampleFormat::U8),
        "s16" | "i16" => Ok(SampleFormat::S16),
        "s24" | "i24" => Ok(SampleFormat::S24),
        "s32" | "i32" => Ok(SampleFormat::S32),
        "f32" => Ok(SampleFormat::F32),
        "f64" => Ok(SampleFormat::F64),
        other => Err(format!("unknown sample format '{other}'")),
    }
}

fn color_preset(text: &str) -> Result<ColorDescription, String> {
    match text.to_ascii_lowercase().as_str() {
        "bt709-limited" | "bt709" | "default" => Ok(ColorDescription::BT709_LIMITED),
        "bt2020-pq" | "pq" => Ok(ColorDescription::BT2020_PQ),
        "bt2020-hlg" | "hlg" => Ok(ColorDescription::BT2020_HLG),
        other => Err(format!("unknown colour preset '{other}'")),
    }
}

fn range_name(range: ColorRange) -> &'static str {
    match range {
        ColorRange::Limited => "limited",
        ColorRange::Full => "full",
    }
}

fn primaries_name(primaries: ColorPrimaries) -> &'static str {
    match primaries {
        ColorPrimaries::Bt709 => "bt709",
        ColorPrimaries::Bt2020 => "bt2020",
        ColorPrimaries::DisplayP3 => "display-p3",
        ColorPrimaries::Unknown => "unknown",
    }
}

fn transfer_name(transfer: TransferFunction) -> &'static str {
    match transfer {
        TransferFunction::Sdr => "sdr",
        TransferFunction::Pq => "pq",
        TransferFunction::Hlg => "hlg",
        TransferFunction::Linear => "linear",
        TransferFunction::Srgb => "srgb",
        TransferFunction::Unknown => "unknown",
    }
}

fn matrix_name(matrix: MatrixCoefficients) -> &'static str {
    match matrix {
        MatrixCoefficients::Bt709 => "bt709",
        MatrixCoefficients::Bt2020NonConstant => "bt2020-ncl",
        MatrixCoefficients::Identity => "identity",
        MatrixCoefficients::Unknown => "unknown",
    }
}

fn chroma_name(chroma: ChromaLocation) -> &'static str {
    match chroma {
        ChromaLocation::Left => "left",
        ChromaLocation::Center => "center",
        ChromaLocation::TopLeft => "top-left",
    }
}

fn parse_color_range(text: &str) -> Result<ColorRange, String> {
    match text.to_ascii_lowercase().as_str() {
        "limited" | "tv" => Ok(ColorRange::Limited),
        "full" | "pc" => Ok(ColorRange::Full),
        other => Err(format!("unknown colour range '{other}'")),
    }
}

fn parse_primaries(text: &str) -> Result<ColorPrimaries, String> {
    match text.to_ascii_lowercase().as_str() {
        "bt709" => Ok(ColorPrimaries::Bt709),
        "bt2020" => Ok(ColorPrimaries::Bt2020),
        "display-p3" | "p3" => Ok(ColorPrimaries::DisplayP3),
        "unknown" => Ok(ColorPrimaries::Unknown),
        other => Err(format!("unknown colour primaries '{other}'")),
    }
}

fn parse_transfer(text: &str) -> Result<TransferFunction, String> {
    match text.to_ascii_lowercase().as_str() {
        "sdr" | "bt709" => Ok(TransferFunction::Sdr),
        "pq" => Ok(TransferFunction::Pq),
        "hlg" => Ok(TransferFunction::Hlg),
        "linear" => Ok(TransferFunction::Linear),
        "srgb" => Ok(TransferFunction::Srgb),
        "unknown" => Ok(TransferFunction::Unknown),
        other => Err(format!("unknown transfer function '{other}'")),
    }
}

fn parse_matrix(text: &str) -> Result<MatrixCoefficients, String> {
    match text.to_ascii_lowercase().as_str() {
        "bt709" | "bt601" => Ok(MatrixCoefficients::Bt709),
        "bt2020" | "bt2020-ncl" => Ok(MatrixCoefficients::Bt2020NonConstant),
        "identity" | "rgb" => Ok(MatrixCoefficients::Identity),
        "unknown" => Ok(MatrixCoefficients::Unknown),
        other => Err(format!("unknown matrix coefficients '{other}'")),
    }
}

fn parse_chroma(text: &str) -> Result<ChromaLocation, String> {
    match text.to_ascii_lowercase().as_str() {
        "left" => Ok(ChromaLocation::Left),
        "center" | "centre" => Ok(ChromaLocation::Center),
        "top-left" | "topleft" => Ok(ChromaLocation::TopLeft),
        other => Err(format!("unknown chroma location '{other}'")),
    }
}

fn parse_flags(text: &str) -> Result<PacketFlags, String> {
    let mut flags = PacketFlags::default();
    for name in text.split(',').filter(|name| !name.is_empty()) {
        match name.to_ascii_lowercase().as_str() {
            "keyframe" | "key" => flags.keyframe = true,
            "discard" => flags.discard = true,
            "eos" | "end-of-stream" => flags.end_of_stream = true,
            "corrupt" | "corrupted" => flags.corrupted = true,
            other => return Err(format!("unknown packet flag '{other}'")),
        }
    }
    Ok(flags)
}

fn flags_text(flags: PacketFlags) -> String {
    let mut names = Vec::new();
    if flags.keyframe {
        names.push("keyframe");
    }
    if flags.discard {
        names.push("discard");
    }
    if flags.end_of_stream {
        names.push("end-of-stream");
    }
    if flags.corrupted {
        names.push("corrupted");
    }
    if names.is_empty() {
        "-".into()
    } else {
        names.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_accepts_prefixes_and_separators() {
        assert_eq!(decode_hex("0x48:65_6c 6c6f").expect("hex"), b"Hello");
        assert!(decode_hex("abc").is_err());
        assert!(decode_hex("zz").is_err());
    }

    #[test]
    fn pixel_and_sample_aliases_resolve() {
        assert_eq!(parse_pixel("yuv420p").expect("pixel"), PixelFormat::I420);
        assert_eq!(parse_sample("i16").expect("sample"), SampleFormat::S16);
        assert!(parse_pixel("nv12").is_err());
    }

    #[test]
    fn colour_presets_and_names_round_trip() {
        let color = color_preset("bt2020-pq").expect("preset");
        assert_eq!(range_name(color.range), "full");
        assert_eq!(transfer_name(color.transfer), "pq");
        assert_eq!(matrix_name(color.matrix), "bt2020-ncl");
        assert_eq!(
            parse_chroma("top-left").expect("chroma"),
            ChromaLocation::TopLeft
        );
    }

    #[test]
    fn pixel_total_matches_plane_sum() {
        let total = pixel_total(PixelFormat::I420, 1920, 1080, 8).expect("total");
        assert_eq!(total, 3_110_400);
    }

    #[test]
    fn packet_flags_parse_and_print() {
        let flags = parse_flags("key,discard,eos,corrupted").expect("flags");
        assert!(flags.keyframe && flags.discard && flags.end_of_stream && flags.corrupted);
        assert_eq!(
            flags_text(flags),
            "keyframe,discard,end-of-stream,corrupted"
        );
        assert_eq!(flags_text(PacketFlags::default()), "-");
    }
}
