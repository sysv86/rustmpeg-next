// SPDX-License-Identifier: Apache-2.0
//! End-to-end Vulkan Compute tests.
//!
//! Every test skips cleanly when no Vulkan adapter is present, so the suite
//! passes on headless CI and on machines without a driver. On a machine with a
//! device, the GPU output is compared byte for byte with the CPU reference.

#![recursion_limit = "1024"]

use std::sync::Arc;

use rustmpeg_video::I420Frame;
use rustmpeg_video::accelerator::{
    CpuFrameMetrics, CpuRgbaToI420, FrameMetricsProvider, RgbaToI420Converter, i420_metrics,
};
use rustmpeg_video::av2::container::RawFrame;
use rustmpeg_video::av2::headers::PixelLayout;
use rustmpeg_video::av2::{Encoder as Av2Encoder, EncoderConfig as Av2EncoderConfig};
use rustmpeg_video::vvc::{VvcEncoder, VvcEncoderConfig};
use rustmpeg_vulkan::DeblockKernel;
use rustmpeg_vulkan::{
    Accelerator, Compute, FrameSize, I420Converter, Placement, PlaneMetricsKernel, RgbaConverter,
    Scheduler, Stage, i420_to_rgba, i420_to_rgba_cpu,
};

fn adapter() -> Option<Compute> {
    match Compute::new() {
        Ok(compute) => Some(compute),
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            None
        }
    }
}

fn rgba_gradient(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for index in 0..width * height {
        rgba.push(u8::try_from((index * 11 + seed) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 29 + 7 + seed) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 47 + 13 + seed) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 61 + 3 + seed) % 256).unwrap_or(0));
    }
    rgba
}

fn i420_gradient(width: u32, height: u32, seed: u32) -> I420Frame {
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let y = (0..width * height)
        .map(|index| u8::try_from((index * 7 + seed) % 256).unwrap_or(0))
        .collect();
    let u = (0..chroma_width * chroma_height)
        .map(|index| u8::try_from((index * 13 + seed) % 256).unwrap_or(0))
        .collect();
    let v = (0..chroma_width * chroma_height)
        .map(|index| u8::try_from((index * 19 + seed) % 256).unwrap_or(0))
        .collect();
    I420Frame {
        width,
        height,
        y,
        u,
        v,
    }
}

#[test]
fn rgba_to_i420_and_back_is_bit_exact() {
    let Some(compute) = adapter() else {
        return;
    };
    let to_i420 = RgbaConverter::new(&compute).expect("RGBA kernel compiles");
    let to_rgba = I420Converter::new(&compute).expect("RGBA kernel compiles");

    for (width, height) in [(64, 48), (65, 49), (128, 72), (17, 9)] {
        let rgba = rgba_gradient(width, height, 3);
        let gpu_i420 = to_i420.convert(&rgba, width, height).expect("dispatch");
        let cpu_i420 = CpuRgbaToI420
            .rgba_to_i420(&rgba, width, height)
            .expect("CPU reference");
        assert_eq!(
            gpu_i420, cpu_i420,
            "RGBA->I420 mismatch at {width}x{height}"
        );

        let gpu_rgba = to_rgba.convert(&gpu_i420).expect("dispatch");
        let cpu_rgba = i420_to_rgba_cpu(&gpu_i420).expect("reference");
        assert_eq!(
            gpu_rgba, cpu_rgba,
            "I420->RGBA mismatch at {width}x{height}"
        );

        let direct = i420_to_rgba(&compute, &gpu_i420).expect("convenience path");
        assert_eq!(direct, cpu_rgba);
    }
}

#[test]
fn plane_metrics_match_the_cpu_reference() {
    let Some(compute) = adapter() else {
        return;
    };
    let kernel = PlaneMetricsKernel::new(&compute).expect("metrics kernel compiles");
    for (width, height) in [(64, 48), (65, 49), (128, 72)] {
        let reference = i420_gradient(width, height, 5);
        let distorted = i420_gradient(width, height, 91);
        let gpu = i420_metrics(&kernel, &reference, &distorted).expect("GPU metrics");
        let cpu = i420_metrics(&CpuFrameMetrics, &reference, &distorted).expect("CPU metrics");
        assert_eq!(gpu, cpu, "metrics mismatch at {width}x{height}");
    }
}

#[test]
fn vvc_encoder_gpu_conversion_preserves_the_bitstream() {
    let Some(compute) = adapter() else {
        return;
    };
    let width = 96;
    let height = 64;
    let config = VvcEncoderConfig {
        width,
        height,
        qp: 30,
    };
    let converter = RgbaConverter::new(&compute).expect("kernel compiles");

    let mut cpu_encoder = VvcEncoder::new(config).expect("encoder");
    let mut gpu_encoder = VvcEncoder::new(config).expect("encoder");

    for frame in 0..3 {
        let rgba = rgba_gradient(width, height, frame * 31);
        let cpu = cpu_encoder.encode_rgba(&rgba).expect("CPU encode");
        let gpu = gpu_encoder
            .encode_rgba_with(&rgba, &converter)
            .expect("GPU encode");
        assert_eq!(cpu, gpu, "access unit {frame} differs");
    }
}

fn raw_frame(width: usize, height: usize, seed: u8) -> RawFrame<u8> {
    let mut frame = RawFrame::new(width, height, 8, PixelLayout::I420);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    for (index, sample) in frame.y.iter_mut().enumerate() {
        *sample = u8::try_from((index * 5 + usize::from(seed) * 17) % 256).unwrap_or(0);
    }
    for (index, sample) in frame.u.iter_mut().enumerate() {
        *sample = u8::try_from((index * 7 + usize::from(seed) * 11) % 256).unwrap_or(0);
    }
    for (index, sample) in frame.v.iter_mut().enumerate() {
        *sample = u8::try_from((index * 11 + usize::from(seed) * 3) % 256).unwrap_or(0);
    }
    assert_eq!(frame.u.len(), chroma_width * chroma_height);
    frame
}

#[test]
fn av2_gpu_metrics_preserve_the_bitstream_and_match_cpu_statistics() {
    let Some(compute) = adapter() else {
        return;
    };
    let config = Av2EncoderConfig {
        width: 64,
        height: 64,
        bit_depth: 8,
        layout: PixelLayout::I420,
        keyint: 1,
        ..Av2EncoderConfig::default()
    };

    let mut cpu_encoder = Av2Encoder::<u8>::new(config.clone()).expect("encoder");
    let mut gpu_encoder = Av2Encoder::<u8>::new(config).expect("encoder");
    let kernel = Arc::new(PlaneMetricsKernel::new(&compute).expect("metrics kernel compiles"));
    gpu_encoder.set_frame_metrics(kernel);

    for index in 0..2 {
        let raw = raw_frame(64, 64, index);
        let cpu = cpu_encoder.encode_raw_frame(&raw).expect("CPU encode");
        let gpu = gpu_encoder.encode_raw_frame(&raw).expect("GPU encode");
        assert_eq!(cpu.data, gpu.data, "packet {index} differs");
        assert_eq!(cpu.stats.psnr_y, gpu.stats.psnr_y);
        assert_eq!(cpu.stats.psnr_u, gpu.stats.psnr_u);
        assert_eq!(cpu.stats.psnr_v, gpu.stats.psnr_v);
        assert_eq!(cpu.stats.psnr_avg, gpu.stats.psnr_avg);
        assert_eq!(cpu.stats.mse_y, gpu.stats.mse_y);
    }
}

#[test]
fn metrics_provider_trait_is_object_safe_and_send_sync() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    let Some(compute) = adapter() else {
        return;
    };
    let kernel = PlaneMetricsKernel::new(&compute).expect("metrics kernel compiles");
    assert_send_sync(&kernel);
    let provider: Arc<dyn FrameMetricsProvider> = Arc::new(kernel);
    let a = i420_gradient(16, 16, 0);
    let b = i420_gradient(16, 16, 1);
    let metrics = i420_metrics(provider.as_ref(), &a, &b).expect("metrics");
    assert!(metrics.planes[0].samples > 0);
}

#[test]
fn scheduler_calibrates_against_the_cpu_reference() {
    let Some(compute) = adapter() else {
        return;
    };
    let mut scheduler = Scheduler::with_compute(&compute);
    assert!(!scheduler.is_calibrated());
    let large = FrameSize::new(1920, 1080);
    assert_eq!(
        scheduler.place(Stage::RgbaToI420, large),
        Placement::Cpu,
        "an uncalibrated scheduler must stay on the CPU"
    );

    scheduler.calibrate(&compute).expect("calibration");
    assert!(scheduler.is_calibrated());
    assert!(scheduler.gpu_enabled());
    let costs = scheduler.costs().expect("calibrated costs");
    for cost in [costs.rgba_to_i420, costs.i420_to_rgba, costs.plane_metrics] {
        assert!(cost.cpu_ns_per_pixel > 0.0 && cost.cpu_ns_per_pixel < 1.0e6);
        assert!(cost.cpu_fixed_ns >= 0.0 && cost.cpu_fixed_ns < 1.0e9);
        assert!(cost.gpu_ns_per_pixel > 0.0 && cost.gpu_ns_per_pixel < 1.0e6);
        assert!(cost.gpu_fixed_ns >= 0.0 && cost.gpu_fixed_ns < 1.0e9);
    }

    for stage in [Stage::RgbaToI420, Stage::I420ToRgba, Stage::PlaneMetrics] {
        let Some((cpu_ns, gpu_ns)) = scheduler.estimate(stage, large) else {
            panic!("calibration must produce an estimate");
        };
        assert!(cpu_ns.is_finite() && cpu_ns > 0.0);
        assert!(gpu_ns.is_finite() && gpu_ns >= 0.0);
        let expected = if gpu_ns * 1.1 < cpu_ns {
            Placement::Gpu
        } else {
            Placement::Cpu
        };
        assert_eq!(scheduler.place(stage, large), expected);
    }

    scheduler.disable_gpu();
    assert_eq!(scheduler.place(Stage::RgbaToI420, large), Placement::Cpu);
    assert!(scheduler.estimate(Stage::RgbaToI420, large).is_some());
}

#[test]
fn codecs_can_use_the_smart_accelerator_directly() {
    let Some(compute) = adapter() else {
        return;
    };
    let accelerator = Accelerator::with_compute(compute).expect("accelerator");
    assert!(accelerator.scheduler().is_calibrated());

    let width = 96;
    let height = 64;
    let config = VvcEncoderConfig {
        width,
        height,
        qp: 30,
    };
    let mut cpu_encoder = VvcEncoder::new(config).expect("encoder");
    let mut smart_encoder = VvcEncoder::new(config).expect("encoder");
    for frame in 0..2 {
        let rgba = rgba_gradient(width, height, frame * 17);
        let cpu = cpu_encoder.encode_rgba(&rgba).expect("CPU encode");
        let smart = smart_encoder
            .encode_rgba_with(&rgba, &accelerator)
            .expect("accelerator encode");
        assert_eq!(cpu, smart);
    }

    let av2_config = Av2EncoderConfig {
        width: 64,
        height: 64,
        bit_depth: 8,
        layout: PixelLayout::I420,
        keyint: 1,
        ..Av2EncoderConfig::default()
    };
    let mut cpu_encoder = Av2Encoder::<u8>::new(av2_config.clone()).expect("encoder");
    let mut smart_encoder = Av2Encoder::<u8>::new(av2_config).expect("encoder");
    smart_encoder.set_frame_metrics(Arc::new(accelerator));
    let raw = raw_frame(64, 64, 2);
    let cpu = cpu_encoder.encode_raw_frame(&raw).expect("CPU encode");
    let smart = smart_encoder.encode_raw_frame(&raw).expect("smart encode");
    assert_eq!(cpu.data, smart.data);
    assert_eq!(cpu.stats.psnr_y, smart.stats.psnr_y);
}

#[test]
fn av2_deblock_accelerator_is_bit_exact_and_preserves_the_bitstream() {
    let Some(compute) = adapter() else {
        return;
    };
    let kernel = Arc::new(DeblockKernel::new(&compute).expect("deblock kernel compiles"));
    let config = Av2EncoderConfig {
        width: 64,
        height: 64,
        bit_depth: 8,
        layout: PixelLayout::I420,
        keyint: 1,
        tools: rustmpeg_video::av2::frame::ToolConfig {
            enable_deblock: true,
            ..rustmpeg_video::av2::frame::ToolConfig::default()
        },
        ..Av2EncoderConfig::default()
    };

    let mut cpu_encoder = Av2Encoder::<u8>::new(config.clone()).expect("encoder");
    let mut gpu_encoder = Av2Encoder::<u8>::new(config).expect("encoder");
    gpu_encoder.set_deblock_filter(kernel);

    for index in 0..2 {
        let raw = raw_frame(64, 64, index);
        let cpu = cpu_encoder.encode_raw_frame(&raw).expect("CPU encode");
        let gpu = gpu_encoder.encode_raw_frame(&raw).expect("GPU encode");
        if cpu.data != gpu.data {
            let position = cpu
                .data
                .iter()
                .zip(&gpu.data)
                .position(|(a, b)| a != b)
                .unwrap_or(cpu.data.len().min(gpu.data.len()));
            panic!(
                "packet {index} differs at {position}: cpu={} gpu={} lengths {} {}",
                cpu.data.get(position).copied().unwrap_or(0),
                gpu.data.get(position).copied().unwrap_or(0),
                cpu.data.len(),
                gpu.data.len()
            );
        }
        for plane in 0..3 {
            if cpu.rec_frame.planes[plane].data != gpu.rec_frame.planes[plane].data {
                let position = cpu.rec_frame.planes[plane]
                    .data
                    .iter()
                    .zip(&gpu.rec_frame.planes[plane].data)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                panic!(
                    "reconstruction plane {plane} differs at {position}: cpu={} gpu={}",
                    cpu.rec_frame.planes[plane].data[position],
                    gpu.rec_frame.planes[plane].data[position]
                );
            }
        }
    }
}
