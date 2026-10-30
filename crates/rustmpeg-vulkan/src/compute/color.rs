// SPDX-License-Identifier: Apache-2.0
//! Colour conversion between premultiplied RGBA8 and BT.709 limited-range
//! I420 on Vulkan Compute.
//!
//! Both directions use the same integer arithmetic as the software reference in
//! `CpuRgbaToI420` and `i420_to_rgba_cpu`, so a GPU frame and a CPU
//! frame are byte-identical. The kernels write packed 32-bit words and are
//! arranged so that exactly one invocation owns each output word, which makes
//! the dispatch race-free without atomics.

use rustmpeg_video::I420Frame;
use rustmpeg_video::accelerator::{AcceleratorError, RgbaToI420Converter as RgbaConverterTrait};

use crate::compute::{Compute, Kernel};
use crate::error::VulkanError;

const Y_SHADER: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
};

@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> luma: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

fn pixel(index: u32) -> vec4<i32> {
    let packed = rgba[index];
    return vec4<i32>(
        i32(packed & 0xffu),
        i32((packed >> 8u) & 0xffu),
        i32((packed >> 16u) & 0xffu),
        i32((packed >> 24u) & 0xffu),
    );
}

fn unpremultiply(c: vec4<i32>) -> vec3<i32> {
    let alpha = c.w;
    if (alpha == 255) {
        return c.xyz;
    }
    if (alpha == 0) {
        return vec3<i32>(0);
    }
    let half = alpha >> 1;
    let scaled = c.xyz * vec3<i32>(255) + vec3<i32>(half);
    return min(scaled / vec3<i32>(alpha), vec3<i32>(255));
}

fn luma_value(rgb: vec3<i32>) -> u32 {
    let value = 16 + ((11966 * rgb.x + 40254 * rgb.y + 4063 * rgb.z + 32768) >> 16);
    return u32(clamp(value, 0, 255));
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let word = id.y * grid.x + id.x;
    let total = dims.width * dims.height;
    let words = (total + 3u) >> 2u;
    if (word >= words) {
        return;
    }

    var packed = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let sample = word * 4u + k;
        if (sample < total) {
            let rgb = unpremultiply(pixel(sample));
            packed = packed | (luma_value(rgb) << (k * 8u));
        }
    }
    luma[word] = packed;
}
"#;

const CHROMA_SHADER: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
};

@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> cb: array<u32>;
@group(0) @binding(2) var<storage, read_write> cr: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

fn pixel(index: u32) -> vec4<i32> {
    let packed = rgba[index];
    return vec4<i32>(
        i32(packed & 0xffu),
        i32((packed >> 8u) & 0xffu),
        i32((packed >> 16u) & 0xffu),
        i32((packed >> 24u) & 0xffu),
    );
}

fn unpremultiply(c: vec4<i32>) -> vec3<i32> {
    let alpha = c.w;
    if (alpha == 255) {
        return c.xyz;
    }
    if (alpha == 0) {
        return vec3<i32>(0);
    }
    let half = alpha >> 1;
    let scaled = c.xyz * vec3<i32>(255) + vec3<i32>(half);
    return min(scaled / vec3<i32>(alpha), vec3<i32>(255));
}

fn box_2x2(cx: u32, cy: u32) -> vec3<i32> {
    var sum = vec3<i32>(0);
    var count = 0;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        let py = cy * 2u + dy;
        if (py < dims.height) {
            for (var dx = 0u; dx < 2u; dx = dx + 1u) {
                let px = cx * 2u + dx;
                if (px < dims.width) {
                    sum = sum + unpremultiply(pixel(py * dims.width + px));
                    count = count + 1;
                }
            }
        }
    }
    if (count == 4) {
        return (sum + vec3<i32>(2)) >> vec3<u32>(2);
    }
    if (count > 0) {
        return (sum + vec3<i32>(count >> 1)) / vec3<i32>(count);
    }
    return vec3<i32>(0);
}

fn chroma_b(rgb: vec3<i32>) -> u32 {
    let value = 128 + ((-6596 * rgb.x - 22188 * rgb.y + 28784 * rgb.z + 32768) >> 16);
    return u32(clamp(value, 0, 255));
}

fn chroma_r(rgb: vec3<i32>) -> u32 {
    let value = 128 + ((28784 * rgb.x - 26149 * rgb.y - 2635 * rgb.z + 32768) >> 16);
    return u32(clamp(value, 0, 255));
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let word = id.y * grid.x + id.x;
    let total = dims.chroma_width * dims.chroma_height;
    let words = (total + 3u) >> 2u;
    if (word >= words) {
        return;
    }

    var packed_b = 0u;
    var packed_r = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let sample = word * 4u + k;
        if (sample < total) {
            let cy = sample / dims.chroma_width;
            let cx = sample - cy * dims.chroma_width;
            let rgb = box_2x2(cx, cy);
            packed_b = packed_b | (chroma_b(rgb) << (k * 8u));
            packed_r = packed_r | (chroma_r(rgb) << (k * 8u));
        }
    }
    cb[word] = packed_b;
    cr[word] = packed_r;
}
"#;

const RGBA_SHADER: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
};

@group(0) @binding(0) var<storage, read> luma: array<u32>;
@group(0) @binding(1) var<storage, read> cb: array<u32>;
@group(0) @binding(2) var<storage, read> cr: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= dims.width || id.y >= dims.height) {
        return;
    }
    let luma_index = id.y * dims.width + id.x;
    let y = i32((luma[luma_index >> 2u] >> ((luma_index & 3u) * 8u)) & 0xffu);

    let chroma_index = (id.y >> 1u) * dims.chroma_width + (id.x >> 1u);
    let u = i32((cb[chroma_index >> 2u] >> ((chroma_index & 3u) * 8u)) & 0xffu);
    let v = i32((cr[chroma_index >> 2u] >> ((chroma_index & 3u) * 8u)) & 0xffu);

    let c = y - 16;
    let d = u - 128;
    let e = v - 128;
    let r = clamp((298 * c + 409 * e + 128) >> 8, 0, 255);
    let g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
    let b = clamp((298 * c + 516 * d + 128) >> 8, 0, 255);

    rgba[luma_index] = u32(r) | (u32(g) << 8u) | (u32(b) << 16u) | (255u << 24u);
}
"#;

/// A compiled I420 to RGBA kernel.
///
/// The converter owns a handle to one [`Compute`] device and can be reused for
/// every frame; compilation happens once in [`I420Converter::new`].
#[derive(Debug)]
pub struct I420Converter {
    compute: Compute,
    kernel: Kernel,
}

impl I420Converter {
    /// Compiles the conversion kernel.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when the shader does not compile.
    pub fn new(compute: &Compute) -> Result<Self, VulkanError> {
        Ok(Self {
            compute: compute.clone(),
            kernel: compute.kernel(RGBA_SHADER, "main")?,
        })
    }

    /// Converts one I420 frame into tightly packed RGBA8.
    ///
    /// The output has `width * height * 4` bytes in row-major order with an
    /// opaque alpha channel.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Execution`] when the frame is malformed and
    /// propagates buffer and dispatch failures.
    pub fn convert(&self, frame: &I420Frame) -> Result<Vec<u8>, VulkanError> {
        if !frame.is_valid() {
            return Err(VulkanError::InvalidConfig(
                "I420 frame dimensions do not match its planes".to_owned(),
            ));
        }
        let width = frame.width;
        let height = frame.height;
        let (chroma_width, chroma_height) = frame.chroma_size();
        let output_len =
            usize::try_from(u64::from(width) * u64::from(height) * 4).map_err(|_| {
                VulkanError::BufferTooLarge {
                    requested: u64::from(width) * u64::from(height) * 4,
                    limit: usize::MAX as u64,
                }
            })?;

        let luma = self.compute.upload(&pad_to_word(&frame.y))?;
        let cb = self.compute.upload(&pad_to_word(&frame.u))?;
        let cr = self.compute.upload(&pad_to_word(&frame.v))?;
        let output = self.compute.allocate(output_len)?;
        let dims = self
            .compute
            .uniform(&dims_bytes(width, height, chroma_width, chroma_height))?;

        let mut batch = self.compute.batch();
        batch.dispatch(
            &self.kernel,
            &[&luma, &cb, &cr, &output, &dims],
            [width.div_ceil(8), height.div_ceil(8), 1],
        )?;
        batch.read(&output, output_len)?;
        let mut results = batch.submit()?;
        Ok(std::mem::take(&mut results[0]))
    }
}

/// A compiled RGBA to I420 kernel.
///
/// The converter implements [`RgbaConverterTrait`], which lets the VVC encoder
/// use it without depending on this crate.
#[derive(Debug)]
pub struct RgbaConverter {
    compute: Compute,
    luma: Kernel,
    chroma: Kernel,
}

impl RgbaConverter {
    /// Compiles both conversion kernels.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when a shader does not compile.
    pub fn new(compute: &Compute) -> Result<Self, VulkanError> {
        Ok(Self {
            compute: compute.clone(),
            luma: compute.kernel(Y_SHADER, "main")?,
            chroma: compute.kernel(CHROMA_SHADER, "main")?,
        })
    }

    /// Converts exactly `width * height * 4` bytes of premultiplied RGBA8.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Execution`] when the input length or the
    /// dimensions are invalid, and propagates buffer and dispatch failures.
    pub fn convert(&self, rgba: &[u8], width: u32, height: u32) -> Result<I420Frame, VulkanError> {
        validate_rgba(rgba, width, height).map_err(VulkanError::InvalidConfig)?;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);

        let luma_len = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| VulkanError::Execution("luma plane is too large".to_owned()))?;
        let chroma_len = usize::try_from(u64::from(chroma_width) * u64::from(chroma_height))
            .map_err(|_| VulkanError::Execution("chroma plane is too large".to_owned()))?;

        let input = self.compute.upload(rgba)?;
        let luma = self.compute.allocate(luma_len)?;
        let cb = self.compute.allocate(chroma_len)?;
        let cr = self.compute.allocate(chroma_len)?;
        let dims = self
            .compute
            .uniform(&dims_bytes(width, height, chroma_width, chroma_height))?;

        let luma_words = luma_len.div_ceil(4) as u32;
        let chroma_words = chroma_len.div_ceil(4) as u32;
        let mut batch = self.compute.batch();
        batch.dispatch(
            &self.luma,
            &[&input, &luma, &dims],
            word_grid(&self.compute, luma_words)?,
        )?;
        batch.dispatch(
            &self.chroma,
            &[&input, &cb, &cr, &dims],
            word_grid(&self.compute, chroma_words)?,
        )?;
        let luma_index = batch.read(&luma, luma_len)?;
        let cb_index = batch.read(&cb, chroma_len)?;
        let cr_index = batch.read(&cr, chroma_len)?;
        let mut results = batch.submit()?;

        Ok(I420Frame {
            width,
            height,
            y: std::mem::take(&mut results[luma_index]),
            u: std::mem::take(&mut results[cb_index]),
            v: std::mem::take(&mut results[cr_index]),
        })
    }
}

impl RgbaConverterTrait for RgbaConverter {
    fn rgba_to_i420(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<I420Frame, AcceleratorError> {
        self.convert(rgba, width, height)
            .map_err(crate::compute::accelerator_error)
    }
}

/// Converts one I420 frame into RGBA8, compiling the kernel for this call.
///
/// Prefer [`I420Converter`] when converting more than one frame.
///
/// # Errors
///
/// Propagates kernel compilation, buffer, and dispatch failures.
pub fn i420_to_rgba(compute: &Compute, frame: &I420Frame) -> Result<Vec<u8>, VulkanError> {
    I420Converter::new(compute)?.convert(frame)
}

/// Converts one RGBA8 frame into I420, compiling the kernels for this call.
///
/// Prefer [`RgbaConverter`] when converting more than one frame.
///
/// # Errors
///
/// Propagates kernel compilation, buffer, and dispatch failures.
pub fn rgba_to_i420(
    compute: &Compute,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<I420Frame, VulkanError> {
    RgbaConverter::new(compute)?.convert(rgba, width, height)
}

/// CPU reference implementation of the I420 to RGBA shader.
///
/// Uses the same integer BT.709 limited-range matrix as the WGSL kernel.
///
/// # Errors
///
/// Returns [`VulkanError::Execution`] when the frame is malformed.
pub fn i420_to_rgba_cpu(frame: &I420Frame) -> Result<Vec<u8>, VulkanError> {
    if !frame.is_valid() {
        return Err(VulkanError::Execution(
            "I420 frame dimensions do not match its planes".to_owned(),
        ));
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let chroma_width = frame.chroma_size().0 as usize;

    let mut rgba = vec![0_u8; width * height * 4];
    for row in 0..height {
        for column in 0..width {
            let chroma_index = (row / 2) * chroma_width + column / 2;
            let luma = i32::from(frame.y[row * width + column]) - 16;
            let cb = i32::from(frame.u[chroma_index]) - 128;
            let cr = i32::from(frame.v[chroma_index]) - 128;

            let r = clamp_u8((298 * luma + 409 * cr + 128) >> 8);
            let g = clamp_u8((298 * luma - 100 * cb - 208 * cr + 128) >> 8);
            let b = clamp_u8((298 * luma + 516 * cb + 128) >> 8);

            let offset = (row * width + column) * 4;
            rgba[offset] = r;
            rgba[offset + 1] = g;
            rgba[offset + 2] = b;
            rgba[offset + 3] = 255;
        }
    }
    Ok(rgba)
}

fn validate_rgba(rgba: &[u8], width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("picture dimensions must be non-zero".to_owned());
    }
    let expected = u64::from(width) * u64::from(height) * 4;
    if expected != rgba.len() as u64 {
        return Err(format!(
            "RGBA input has {} bytes, expected {expected} for {width}x{height}",
            rgba.len()
        ));
    }
    Ok(())
}

fn dims_bytes(width: u32, height: u32, chroma_width: u32, chroma_height: u32) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    bytes[0..4].copy_from_slice(&width.to_le_bytes());
    bytes[4..8].copy_from_slice(&height.to_le_bytes());
    bytes[8..12].copy_from_slice(&chroma_width.to_le_bytes());
    bytes[12..16].copy_from_slice(&chroma_height.to_le_bytes());
    bytes
}

fn word_grid(compute: &Compute, words: u32) -> Result<[u32; 3], VulkanError> {
    let limit = compute.info().max_workgroups_per_dimension;
    let x = words.clamp(1, limit);
    let y = words.div_ceil(x);
    if y > limit {
        return Err(VulkanError::DispatchTooLarge {
            requested: [x, y, 1],
            limit,
        });
    }
    Ok([x, y, 1])
}

fn pad_to_word(data: &[u8]) -> Vec<u8> {
    let mut data = data.to_vec();
    let aligned = (data.len() + 3) & !3;
    data.resize(aligned.max(4), 0);
    data
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmpeg_video::accelerator::{CpuRgbaToI420, RgbaToI420Converter};

    fn gradient(width: u32, height: u32) -> I420Frame {
        let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
        let y = (0..width * height)
            .map(|index| u8::try_from(index % 256).unwrap_or(0))
            .collect();
        let u = (0..chroma_width * chroma_height)
            .map(|index| u8::try_from((index * 7) % 256).unwrap_or(0))
            .collect();
        let v = (0..chroma_width * chroma_height)
            .map(|index| u8::try_from((index * 13) % 256).unwrap_or(0))
            .collect();
        I420Frame {
            width,
            height,
            y,
            u,
            v,
        }
    }

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

    #[test]
    fn cpu_reference_maps_black_and_white() {
        let black = I420Frame {
            width: 2,
            height: 2,
            y: vec![16; 4],
            u: vec![128; 1],
            v: vec![128; 1],
        };
        assert_eq!(i420_to_rgba_cpu(&black).unwrap(), [0, 0, 0, 255].repeat(4));

        let white = I420Frame {
            width: 2,
            height: 2,
            y: vec![235; 4],
            u: vec![128; 1],
            v: vec![128; 1],
        };
        assert_eq!(
            i420_to_rgba_cpu(&white).unwrap(),
            [255, 255, 255, 255].repeat(4)
        );
    }

    #[test]
    fn cpu_reference_rejects_invalid_frames() {
        let frame = I420Frame {
            width: 4,
            height: 4,
            y: vec![0; 16],
            u: vec![0; 3],
            v: vec![0; 4],
        };
        assert!(i420_to_rgba_cpu(&frame).is_err());
    }

    #[test]
    fn rgba_validation_rejects_short_and_empty_input() {
        assert!(validate_rgba(&[0; 15], 2, 2).is_err());
        assert!(validate_rgba(&[], 0, 0).is_err());
        assert!(validate_rgba(&[0; 16], 2, 2).is_ok());
    }

    #[test]
    fn word_grid_stays_inside_the_device_limit() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let limit = compute.info().max_workgroups_per_dimension;
        let grid = word_grid(&compute, 1).expect("grid");
        assert_eq!(grid, [1, 1, 1]);
        let grid = word_grid(&compute, limit + 1).expect("grid");
        assert_eq!(grid[0], limit);
        assert_eq!(grid[1], 2);
    }

    #[test]
    fn gpu_i420_to_rgba_matches_the_cpu_reference() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let converter = I420Converter::new(&compute).expect("kernel compiles");
        for (width, height) in [(16, 16), (17, 9), (64, 48), (31, 7)] {
            let frame = gradient(width, height);
            let gpu = converter.convert(&frame).expect("dispatch");
            let cpu = i420_to_rgba_cpu(&frame).expect("reference");
            assert_eq!(gpu, cpu, "mismatch for {width}x{height}");
        }
    }

    #[test]
    fn gpu_rgba_to_i420_matches_the_cpu_reference_with_alpha() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let converter = RgbaConverter::new(&compute).expect("kernel compiles");
        for (width, height) in [(16, 16), (17, 9), (64, 48), (31, 7), (3, 3)] {
            let rgba = rgba_gradient(width, height);
            let gpu = converter.convert(&rgba, width, height).expect("dispatch");
            let cpu = CpuRgbaToI420
                .rgba_to_i420(&rgba, width, height)
                .expect("reference");
            assert_eq!(gpu, cpu, "mismatch for {width}x{height}");
        }
    }

    #[test]
    fn gpu_rgba_conversion_matches_opaque_and_transparent_pixels() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let converter = RgbaConverter::new(&compute).expect("kernel compiles");
        let rgba = vec![
            0, 0, 0, 0, 255, 255, 255, 255, 128, 0, 0, 128, 0, 0, 255, 64, 10, 200, 30, 200, 1, 2,
            3, 254,
        ];
        let gpu = converter.convert(&rgba, 3, 2).expect("dispatch");
        let cpu = CpuRgbaToI420.rgba_to_i420(&rgba, 3, 2).expect("reference");
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn converter_trait_maps_driver_errors() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let converter = RgbaConverter::new(&compute).expect("kernel compiles");
        let error = RgbaConverterTrait::rgba_to_i420(&converter, &[0; 3], 2, 2)
            .expect_err("short input must fail");
        assert!(matches!(error, AcceleratorError::InvalidInput(_)));
    }
}
