// SPDX-License-Identifier: Apache-2.0
//! AV2 in-loop deblocking filter on Vulkan Compute.
//!
//! The kernel mirrors `rustmpeg_video::av2::deblock` for 8-bit planes. Each
//! invocation owns one 32-bit word of the output, so the vertical and horizontal
//! passes are race-free without atomics. The vertical pass writes the plane
//! in place; the horizontal pass reads the vertically filtered result in a
//! second dispatch. Both are recorded into one [`Batch`](crate::compute::Batch),
//! so a frame costs one submission and one read-back per plane.

use rustmpeg_video::accelerator::AcceleratorError;
use rustmpeg_video::av2::deblock::DeblockFilter;
use rustmpeg_video::av2::frame::Plane;
use rustmpeg_video::av2::headers::FhDeblock;

use crate::compute::{Compute, GpuBuffer, Kernel};
use crate::error::VulkanError;

const SHADER: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    stride: u32,
    words_per_row: u32,
    level: u32,
    sharpness: u32,
    step: u32,
    edges: u32,
};

@group(0) @binding(0) var<storage, read_write> plane: array<u32>;
@group(0) @binding(1) var<uniform> dims: Dims;

fn load(x: u32, y: u32) -> i32 {
    let index = y * dims.stride + x;
    return i32((plane[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu);
}

fn filter_edge(
    p2: i32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    level: i32,
    sharpness: u32,
) -> vec4<i32> {
    let limit = max(level >> min(sharpness, 2u), 2);
    let blimit = max(2 * level + limit + 4, 4);
    let thresh = max(level >> 4u, 1);
    let mask = abs(p1 - p0) <= limit
        && abs(q1 - q0) <= limit
        && (abs(p0 - q0) * 2 + (abs(p1 - q1) >> 1u)) <= blimit;
    if (!mask) {
        return vec4<i32>(p1, p0, q0, q1);
    }

    let flat = abs(p2 - p1) <= thresh && abs(q2 - q1) <= thresh;
    let hev = abs(p1 - p0) > thresh || abs(q1 - q0) > thresh;
    if (flat && !hev) {
        let np1 = clamp((p2 * 3 + p1 * 2 + p0 * 2 + q0 + 4) >> 3u, 0, 255);
        let np0 = clamp((p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3u, 0, 255);
        let nq0 = clamp((p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3u, 0, 255);
        let nq1 = clamp((p0 + q0 * 2 + q1 * 2 + q2 * 3 + 4) >> 3u, 0, 255);
        return vec4<i32>(np1, np0, nq0, nq1);
    }

    var delta_raw = 3 * (q0 - p0);
    if (hev) {
        delta_raw = delta_raw + clamp(p1 - q1, -127, 127);
    }
    let d = clamp((delta_raw + 4) >> 3u, -16, 16);
    let np0 = clamp(p0 + d, 0, 255);
    let nq0 = clamp(q0 - d, 0, 255);
    var np1 = p1;
    var nq1 = q1;
    if (!hev) {
        let d2 = (d + 1) >> 1u;
        np1 = clamp(p1 + d2, 0, 255);
        nq1 = clamp(q1 - d2, 0, 255);
    }
    return vec4<i32>(np1, np0, nq0, nq1);
}

@compute @workgroup_size(8, 8, 1)
fn vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let word_x = id.x;
    if (row >= dims.height || word_x >= dims.words_per_row) {
        return;
    }
    let base = word_x * 4u;
    if (base >= dims.width) {
        return;
    }

    let word_index = row * dims.words_per_row + word_x;
    let original = plane[word_index];
    var packed = 0u;
    var mask = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let column = base + k;
        if (column < dims.width) {
            mask = mask | (0xffu << (k * 8u));
            let edge = ((column + 2u) / dims.step) * dims.step;
            if (edge >= dims.step && edge + 3u <= dims.width && column + 2u >= edge && column <= edge + 1u) {
                let out = filter_edge(
                    load(edge - 3u, row),
                    load(edge - 2u, row),
                    load(edge - 1u, row),
                    load(edge, row),
                    load(edge + 1u, row),
                    load(edge + 2u, row),
                    i32(dims.level),
                    dims.sharpness,
                );
                let slot = column - (edge - 2u);
                var value = out.x;
                if (slot == 1u) {
                    value = out.y;
                } else if (slot == 2u) {
                    value = out.z;
                } else if (slot == 3u) {
                    value = out.w;
                }
                packed = packed | (u32(clamp(value, 0, 255)) << (k * 8u));
            } else {
                packed = packed | (u32(load(column, row)) << (k * 8u));
            }
        }
    }
    plane[word_index] = (original & ~mask) | (packed & mask);
}

@compute @workgroup_size(8, 8, 1)
fn horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    let word_x = id.x;
    let edge_index = id.y;
    if (word_x >= dims.words_per_row || edge_index >= dims.edges) {
        return;
    }
    let edge = (edge_index + 1u) * dims.step;
    if (edge + 3u > dims.height) {
        return;
    }
    let base = word_x * 4u;
    if (base >= dims.width) {
        return;
    }

    var rows = array<u32, 4>(0u, 0u, 0u, 0u);
    var mask = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let column = base + k;
        if (column < dims.width) {
            mask = mask | (0xffu << (k * 8u));
            let out = filter_edge(
                load(column, edge - 3u),
                load(column, edge - 2u),
                load(column, edge - 1u),
                load(column, edge),
                load(column, edge + 1u),
                load(column, edge + 2u),
                i32(dims.level),
                dims.sharpness,
            );
            rows[0] = rows[0] | (u32(clamp(out.x, 0, 255)) << (k * 8u));
            rows[1] = rows[1] | (u32(clamp(out.y, 0, 255)) << (k * 8u));
            rows[2] = rows[2] | (u32(clamp(out.z, 0, 255)) << (k * 8u));
            rows[3] = rows[3] | (u32(clamp(out.w, 0, 255)) << (k * 8u));
        } else {
            rows[0] = rows[0] | (u32(load(column, edge - 2u)) << (k * 8u));
            rows[1] = rows[1] | (u32(load(column, edge - 1u)) << (k * 8u));
            rows[2] = rows[2] | (u32(load(column, edge)) << (k * 8u));
            rows[3] = rows[3] | (u32(load(column, edge + 1u)) << (k * 8u));
        }
    }

    for (var r = 0u; r < 4u; r = r + 1u) {
        let row = edge - 2u + r;
        let word_index = row * dims.words_per_row + word_x;
        let original = plane[word_index];
        plane[word_index] = (original & ~mask) | (rows[r] & mask);
    }
}
"#;

/// A compiled AV2 deblocking kernel.
#[derive(Debug)]
pub struct DeblockKernel {
    compute: Compute,
    vertical: Kernel,
    horizontal: Kernel,
}

impl DeblockKernel {
    /// Compiles both deblocking passes.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when a shader does not compile.
    pub fn new(compute: &Compute) -> Result<Self, VulkanError> {
        Ok(Self {
            compute: compute.clone(),
            vertical: compute.kernel(SHADER, "vertical")?,
            horizontal: compute.kernel(SHADER, "horizontal")?,
        })
    }

    fn apply(&self, planes: [&mut Plane<u8>; 3], params: &FhDeblock) -> Result<(), VulkanError> {
        struct Job {
            plane: usize,
            len: usize,
            read: usize,
        }

        let mut batch = self.compute.batch();
        let mut keep_alive: Vec<GpuBuffer> = Vec::new();
        let mut jobs: Vec<Job> = Vec::new();
        let sharpness = u32::from(params.sharpness);

        for (index, plane_ref) in planes.iter().enumerate() {
            let level = match index {
                0 => params.level_y[0].max(params.level_y[1]),
                1 => params.level_u,
                _ => params.level_v,
            };
            let plane = &**plane_ref;
            if plane.width == 0 || plane.height == 0 || level == 0 {
                continue;
            }
            let width = u32::try_from(plane.width).map_err(|_| {
                VulkanError::InvalidConfig("deblock plane width does not fit in u32".to_owned())
            })?;
            let height = u32::try_from(plane.height).map_err(|_| {
                VulkanError::InvalidConfig("deblock plane height does not fit in u32".to_owned())
            })?;
            if plane.stride % 4 != 0 {
                return Err(VulkanError::InvalidConfig(
                    "deblock requires a word-aligned plane stride".to_owned(),
                ));
            }
            let step = if index == 0 {
                8
            } else {
                (8_u32 >> plane.ss_x.min(1)).max(4)
            };
            let len = plane
                .stride
                .checked_mul(plane.height)
                .ok_or_else(|| VulkanError::Execution("deblock plane size overflow".to_owned()))?;
            if len > plane.data.len() {
                return Err(VulkanError::InvalidConfig(
                    "deblock plane exceeds its backing buffer".to_owned(),
                ));
            }
            let words_per_row = u32::try_from(plane.stride / 4).map_err(|_| {
                VulkanError::InvalidConfig("deblock stride does not fit in u32".to_owned())
            })?;
            let edges = height.saturating_sub(3) / step;

            let mut dims = [0_u8; 32];
            dims[0..4].copy_from_slice(&width.to_le_bytes());
            dims[4..8].copy_from_slice(&height.to_le_bytes());
            dims[8..12].copy_from_slice(&(plane.stride as u32).to_le_bytes());
            dims[12..16].copy_from_slice(&words_per_row.to_le_bytes());
            dims[16..20].copy_from_slice(&u32::from(level).to_le_bytes());
            dims[20..24].copy_from_slice(&sharpness.to_le_bytes());
            dims[24..28].copy_from_slice(&step.to_le_bytes());
            dims[28..32].copy_from_slice(&edges.to_le_bytes());

            let buffer = self.compute.upload(&plane.data[..len])?;
            let dims_buffer = self.compute.uniform(&dims)?;
            batch.dispatch(
                &self.vertical,
                &[&buffer, &dims_buffer],
                [words_per_row, height, 1],
            )?;
            if edges > 0 {
                batch.dispatch(
                    &self.horizontal,
                    &[&buffer, &dims_buffer],
                    [words_per_row, edges, 1],
                )?;
            }
            let read = batch.read(&buffer, len)?;
            jobs.push(Job {
                plane: index,
                len,
                read,
            });
            keep_alive.push(buffer);
            keep_alive.push(dims_buffer);
        }

        if jobs.is_empty() {
            return Ok(());
        }
        let results = batch.submit()?;
        for job in jobs {
            planes[job.plane].data[..job.len].copy_from_slice(&results[job.read]);
        }
        let _ = keep_alive;
        Ok(())
    }
}

impl DeblockFilter<u8> for DeblockKernel {
    fn deblock(
        &self,
        planes: [&mut Plane<u8>; 3],
        params: &FhDeblock,
    ) -> Result<(), AcceleratorError> {
        self.apply(planes, params)
            .map_err(crate::compute::accelerator_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmpeg_video::av2::deblock::deblock_frame;
    use rustmpeg_video::av2::frame::Frame;
    use rustmpeg_video::av2::headers::PixelLayout;
    use rustmpeg_video::av2::pixel::BitDepth8;

    fn frame(width: usize, height: usize, layout: PixelLayout) -> Frame<u8> {
        let mut frame = Frame::new(width, height, 8, layout, 0);
        for (index, plane) in frame.planes.iter_mut().enumerate() {
            if plane.width == 0 {
                continue;
            }
            for row in 0..plane.height {
                for column in 0..plane.width {
                    plane.data[row * plane.stride + column] =
                        u8::try_from((row * 31 + column * 17 + index * 53) % 256).unwrap_or(0);
                }
            }
        }
        frame
    }

    fn params() -> FhDeblock {
        FhDeblock {
            level_y: [24, 24],
            level_u: 18,
            level_v: 18,
            sharpness: 1,
            ..FhDeblock::default()
        }
    }

    #[test]
    fn gpu_deblock_matches_the_cpu_filter() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = DeblockKernel::new(&compute).expect("kernel compiles");
        for (width, height, layout) in [
            (64, 48, PixelLayout::I420),
            (48, 48, PixelLayout::I444),
            (64, 48, PixelLayout::I422),
            (65, 49, PixelLayout::I420),
        ] {
            let mut cpu = frame(width, height, layout);
            deblock_frame(BitDepth8, &mut cpu, &params());
            let mut gpu = frame(width, height, layout);
            let [y, u, v] = &mut gpu.planes;
            kernel.deblock([y, u, v], &params()).expect("GPU deblock");
            for index in 0..3 {
                assert_eq!(
                    cpu.planes[index].data, gpu.planes[index].data,
                    "plane {index} mismatch at {width}x{height} {layout:?}"
                );
            }
        }
    }
}
