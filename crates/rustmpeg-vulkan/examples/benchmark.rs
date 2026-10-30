// SPDX-License-Identifier: Apache-2.0
//! CPU versus Vulkan Compute benchmark for the frame-level stages.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p rustmpeg-vulkan --example benchmark
//! ```
//!
//! The GPU numbers include the upload, dispatch, and read-back that a real
//! pipeline pays, so they are directly comparable with the CPU path.

#![recursion_limit = "1024"]

use std::hint::black_box;
use std::time::{Duration, Instant};

use rustmpeg_video::I420Frame;
use rustmpeg_video::accelerator::{
    CpuFrameMetrics, CpuRgbaToI420, FrameMetricsProvider, RgbaToI420Converter, i420_plane_views,
};
use rustmpeg_video::av2::deblock::{DeblockFilter, deblock_frame};
use rustmpeg_video::av2::frame::Frame;
use rustmpeg_video::av2::headers::{FhDeblock, PixelLayout};
use rustmpeg_video::av2::pixel::BitDepth8;
use rustmpeg_video::vvc::{VvcEncoder, VvcEncoderConfig};
use rustmpeg_vulkan::{
    Accelerator, Compute, DeblockKernel, FrameSize, I420Converter, Placement, PlaneMetricsKernel,
    RgbaConverter, Scheduler, Stage, i420_to_rgba_cpu,
};

const SIZES: [(u32, u32); 4] = [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)];
const ITERATIONS: usize = 7;

fn rgba_gradient(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for index in 0..width * height {
        rgba.push(u8::try_from((index * 11) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 29 + 7) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 47 + 13) % 256).unwrap_or(0));
        rgba.push(u8::try_from((index * 61 + 3) % 256).unwrap_or(0));
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

/// Runs `function` once as a warm-up and then `ITERATIONS` timed times.
fn measure(function: impl FnMut()) -> Duration {
    measure_with(ITERATIONS, function)
}

fn measure_with(iterations: usize, mut function: impl FnMut()) -> Duration {
    function();
    let mut samples: Vec<Duration> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        function();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn speedup(cpu: Duration, gpu: Duration) -> f64 {
    cpu.as_secs_f64() / gpu.as_secs_f64()
}

fn placement(scheduler: &Scheduler, stage: Stage, width: u32, height: u32) -> &'static str {
    match scheduler.place(stage, FrameSize::new(width, height)) {
        Placement::Cpu => "CPU",
        Placement::Gpu => "GPU",
    }
}

fn main() {
    let compute = match Compute::new() {
        Ok(compute) => compute,
        Err(error) => {
            eprintln!("no Vulkan compute device: {error}");
            return;
        }
    };
    let info = compute.info();
    println!(
        "adapter: {} | {} | {} | {} MB buffer limit",
        info.name,
        info.backend,
        info.driver,
        info.max_buffer_size / (1024 * 1024)
    );

    let rgba_to_i420 = RgbaConverter::new(&compute).expect("RGBA kernel");
    let i420_to_rgba = I420Converter::new(&compute).expect("I420 kernel");
    let metrics_kernel = PlaneMetricsKernel::new(&compute).expect("metrics kernel");
    let cpu_rgba = CpuRgbaToI420;
    let cpu_metrics = CpuFrameMetrics;

    // Calibrate once, before the heavy benchmark loops. A laptop GPU that has
    // been running at full load for a while can throttle, and re-calibrating
    // afterwards would report a pessimistic rate.
    let accelerator = Accelerator::with_compute(compute.clone()).expect("accelerator");
    let scheduler = accelerator.scheduler();
    if let Some(costs) = scheduler.costs() {
        println!("\n== Calibrated cost model ==");
        for (name, stage, cost) in [
            ("rgba_to_i420", Stage::RgbaToI420, costs.rgba_to_i420),
            ("i420_to_rgba", Stage::I420ToRgba, costs.i420_to_rgba),
            ("plane_metrics", Stage::PlaneMetrics, costs.plane_metrics),
        ] {
            println!(
                "{name:<14} CPU {:>9.4} ns/px + {:>9.1} us | GPU {:>9.4} ns/px + {:>9.1} us",
                cost.cpu_ns_per_pixel,
                cost.cpu_fixed_ns / 1e3,
                cost.gpu_ns_per_pixel,
                cost.gpu_fixed_ns / 1e3
            );
            if let Some((cpu_ns, gpu_ns)) = scheduler.estimate(stage, FrameSize::new(1920, 1080)) {
                println!(
                    "{:<14} 1080p predicted: CPU {:.3} ms, GPU {:.3} ms",
                    "",
                    cpu_ns / 1e6,
                    gpu_ns / 1e6
                );
            }
        }
    }

    println!("\n== RGBA8 -> I420 (upload + convert + read-back) ==");
    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>14} {:>14} {:>8}",
        "size", "CPU", "GPU", "speedup", "CPU Mpix/s", "GPU Mpix/s", "scheduler"
    );
    for (width, height) in SIZES {
        let rgba = rgba_gradient(width, height);
        let pixels = f64::from(width) * f64::from(height);
        let cpu = measure(|| {
            black_box(
                cpu_rgba
                    .rgba_to_i420(black_box(&rgba), width, height)
                    .expect("CPU conversion"),
            );
        });
        let gpu = measure(|| {
            black_box(
                rgba_to_i420
                    .convert(black_box(&rgba), width, height)
                    .expect("GPU conversion"),
            );
        });
        println!(
            "{:<12} {:>9.3} ms {:>9.3} ms {:>9.2}x {:>14.1} {:>14.1} {:>8}",
            format!("{width}x{height}"),
            millis(cpu),
            millis(gpu),
            speedup(cpu, gpu),
            pixels / millis(cpu) / 1e3,
            pixels / millis(gpu) / 1e3,
            placement(scheduler, Stage::RgbaToI420, width, height),
        );
    }

    println!("\n== I420 -> RGBA8 (upload + convert + read-back) ==");
    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>14} {:>14} {:>8}",
        "size", "CPU", "GPU", "speedup", "CPU Mpix/s", "GPU Mpix/s", "scheduler"
    );
    for (width, height) in SIZES {
        let frame = i420_gradient(width, height, 5);
        let pixels = f64::from(width) * f64::from(height);
        let cpu = measure(|| {
            black_box(i420_to_rgba_cpu(black_box(&frame)).expect("CPU conversion"));
        });
        let gpu = measure(|| {
            black_box(
                i420_to_rgba
                    .convert(black_box(&frame))
                    .expect("GPU conversion"),
            );
        });
        println!(
            "{:<12} {:>9.3} ms {:>9.3} ms {:>9.2}x {:>14.1} {:>14.1} {:>8}",
            format!("{width}x{height}"),
            millis(cpu),
            millis(gpu),
            speedup(cpu, gpu),
            pixels / millis(cpu) / 1e3,
            pixels / millis(gpu) / 1e3,
            placement(scheduler, Stage::I420ToRgba, width, height),
        );
    }

    println!("\n== Frame metrics: 3-plane SSE + SAD (Y, U, V) ==");
    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>8}",
        "size", "CPU", "GPU", "speedup", "scheduler"
    );
    for (width, height) in SIZES {
        let reference = i420_gradient(width, height, 7);
        let distorted = i420_gradient(width, height, 91);
        let reference_views = i420_plane_views(&reference);
        let distorted_views = i420_plane_views(&distorted);
        let cpu = measure(|| {
            black_box(
                cpu_metrics
                    .frame_metrics(reference_views, distorted_views)
                    .expect("CPU metrics"),
            );
        });
        let gpu = measure(|| {
            black_box(
                metrics_kernel
                    .frame_metrics(reference_views, distorted_views)
                    .expect("GPU metrics"),
            );
        });
        println!(
            "{:<12} {:>9.3} ms {:>9.3} ms {:>9.2}x {:>8}",
            format!("{width}x{height}"),
            millis(cpu),
            millis(gpu),
            speedup(cpu, gpu),
            placement(scheduler, Stage::PlaneMetrics, width, height),
        );
    }

    println!("\n== AV2 deblocking (vertical + horizontal + read-back) ==");
    println!(
        "{:<12} {:>12} {:>12} {:>10}",
        "size", "CPU", "GPU", "speedup"
    );
    let deblock = DeblockKernel::new(&compute).expect("deblock kernel");
    let params = FhDeblock {
        level_y: [24, 24],
        level_u: 18,
        level_v: 18,
        sharpness: 1,
        ..FhDeblock::default()
    };
    for (width, height) in [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)] {
        let size = (width as usize, height as usize);
        let make_frame = || {
            let mut frame = Frame::<u8>::new(size.0, size.1, 8, PixelLayout::I420, 0);
            for (plane_index, plane) in frame.planes.iter_mut().enumerate() {
                for row in 0..plane.height {
                    for column in 0..plane.width {
                        plane.data[row * plane.stride + column] =
                            u8::try_from((row * 31 + column * 17 + plane_index * 53) % 256)
                                .unwrap_or(0);
                    }
                }
            }
            frame
        };
        let mut cpu_frame = make_frame();
        let mut gpu_frame = make_frame();
        // Repeated filtering is intentional: the work per pass is identical,
        // and it avoids charging a frame copy to either side.
        let cpu = measure(|| {
            deblock_frame(BitDepth8, &mut cpu_frame, &params);
            black_box(&cpu_frame);
        });
        let gpu = measure(|| {
            let [y, u, v] = &mut gpu_frame.planes;
            deblock.deblock([y, u, v], &params).expect("GPU deblock");
            black_box(&gpu_frame);
        });
        println!(
            "{:<12} {:>9.3} ms {:>9.3} ms {:>9.2}x",
            format!("{width}x{height}"),
            millis(cpu),
            millis(gpu),
            speedup(cpu, gpu),
        );
    }

    println!("\n== VVC encode_rgba end-to-end (conversion inside the encoder) ==");
    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>14}",
        "size", "CPU total", "GPU total", "speedup", "GPU conversion"
    );
    for (width, height) in [(320, 180), (640, 360), (1280, 720)] {
        let config = VvcEncoderConfig {
            width,
            height,
            qp: 30,
        };
        let rgba = rgba_gradient(width, height);

        let mut cpu_encoder = VvcEncoder::new(config).expect("encoder");
        let cpu = measure_with(3, || {
            black_box(
                cpu_encoder
                    .encode_rgba(black_box(&rgba))
                    .expect("CPU encode"),
            );
        });

        let mut gpu_encoder = VvcEncoder::new(config).expect("encoder");
        let gpu = measure_with(3, || {
            black_box(
                gpu_encoder
                    .encode_rgba_with(black_box(&rgba), &accelerator)
                    .expect("GPU encode"),
            );
        });

        let conversion = measure(|| {
            black_box(
                accelerator
                    .convert_rgba(black_box(&rgba), width, height)
                    .expect("accelerator conversion"),
            );
        });

        println!(
            "{:<12} {:>9.3} ms {:>9.3} ms {:>9.2}x {:>11.3} ms",
            format!("{width}x{height}"),
            millis(cpu),
            millis(gpu),
            speedup(cpu, gpu),
            millis(conversion),
        );
    }
}
