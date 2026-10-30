// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR file header, multipart header, attribute catalog, tile geometry,
//! and normative validation rules (`OpenEXRCore/parse_header.c` and `validation.c`).

use super::ExrError;

/// OpenEXR 4-byte file magic: `0x76, 0x2f, 0x31, 0x01` (`20000630` LE).
pub const EXR_MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

/// Version flag bit 9 (`0x0000_0200`): single-part tiled image.
pub const EXR_FLAG_TILED: u32 = 0x0000_0200;
/// Version flag bit 10 (`0x0000_0400`): long attribute / channel names (up to 255 bytes).
pub const EXR_FLAG_LONG_NAMES: u32 = 0x0000_0400;
/// Version flag bit 11 (`0x0000_0800`): non-image / deep data part(s) present.
pub const EXR_FLAG_DEEP: u32 = 0x0000_0800;
/// Version flag bit 12 (`0x0000_1000`): multi-part file.
pub const EXR_FLAG_MULTIPART: u32 = 0x0000_1000;
/// Mask of all valid version flags (`0x1e00`).
pub const EXR_VALID_FLAGS_MASK: u32 =
    EXR_FLAG_TILED | EXR_FLAG_LONG_NAMES | EXR_FLAG_DEEP | EXR_FLAG_MULTIPART;

/// Maximum coordinate magnitude (`INT_MAX / 2 = 1_073_741_823`) enforced by
/// `OpenEXRCore/validation.c: internal_exr_validate_read_part`.
pub const EXR_MAX_COORD_BOUND: i32 = i32::MAX / 2;

/// Default maximum image width/height (65536) unless overridden by `ExrConfig`.
pub const DEFAULT_MAX_IMAGE_DIM: u32 = 65536;
/// Default maximum tile width/height (16384) unless overridden by `ExrConfig`.
pub const DEFAULT_MAX_TILE_DIM: u32 = 16384;

/// OpenEXR pixel sample storage type (`UINT = 0`, `HALF = 1`, `FLOAT = 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrSampleType {
    /// 32-bit unsigned integer (`0`).
    Uint = 0,
    /// 16-bit IEEE 754 binary16 half-precision float (`1`).
    Half = 1,
    /// 32-bit IEEE 754 binary32 single-precision float (`2`).
    Float = 2,
}

impl ExrSampleType {
    /// Parses a raw 32-bit sample type ID (`0`, `1`, `2`).
    pub fn from_u32(v: u32) -> Result<Self, ExrError> {
        match v {
            0 => Ok(Self::Uint),
            1 => Ok(Self::Half),
            2 => Ok(Self::Float),
            _ => Err(ExrError::InvalidHeader("unknown channel pixel sample type")),
        }
    }

    /// Size in bytes of a single sample (`4` for `Uint`/`Float`, `2` for `Half`).
    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Uint | Self::Float => 4,
            Self::Half => 2,
        }
    }
}

/// OpenEXR compression codec ID (`0..=13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrCompression {
    /// `0` — Uncompressed (`NO_COMPRESSION`).
    None = 0,
    /// `1` — Run-length encoding of delta-predicted interleaved bytes (`RLE_COMPRESSION`, 1 line).
    Rle = 1,
    /// `2` — Zlib compression, 1 scanline per chunk (`ZIPS_COMPRESSION`).
    Zips = 2,
    /// `3` — Zlib compression, 16 scanlines per chunk (`ZIP_COMPRESSION`).
    Zip = 3,
    /// `4` — 2D Haar wavelet + 16-bit canonical Huffman (`PIZ_COMPRESSION`, 32 scanlines).
    Piz = 4,
    /// `5` — Pixar 24-bit float + byte-plane transpose + zlib (`PXR24_COMPRESSION`, 16 scanlines).
    Pxr24 = 5,
    /// `6` — 4x4 fixed-rate half-float block compression (`B44_COMPRESSION`, 32 scanlines).
    B44 = 6,
    /// `7` — 4x4 half-float block compression with flat-field 3-byte mode (`B44A_COMPRESSION`, 32 scanlines).
    B44a = 7,
    /// `8` — DreamWorks DCT lossy/lossless (`DWAA_COMPRESSION`, 32 scanlines).
    Dwaa = 8,
    /// `9` — DreamWorks DCT lossy/lossless (`DWAB_COMPRESSION`, 256 scanlines).
    Dwab = 9,
    /// `10` — High-Throughput JPEG 2000 256 scanlines (`HTJ2K256_COMPRESSION`).
    Htj2k256 = 10,
    /// `11` — High-Throughput JPEG 2000 32 scanlines (`HTJ2K32_COMPRESSION`).
    Htj2k32 = 11,
    /// `12` — Lossy JPEG 2000 256 scanlines (`LJ2K_COMPRESSION`).
    Lj2k = 12,
    /// `13` — Zstandard compression, 1 scanline per chunk (`ZSTD_COMPRESSION`).
    Zstd = 13,
}

impl ExrCompression {
    /// Parses a compression byte (`0..=13`).
    pub fn from_u8(v: u8) -> Result<Self, ExrError> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Rle),
            2 => Ok(Self::Zips),
            3 => Ok(Self::Zip),
            4 => Ok(Self::Piz),
            5 => Ok(Self::Pxr24),
            6 => Ok(Self::B44),
            7 => Ok(Self::B44a),
            8 => Ok(Self::Dwaa),
            9 => Ok(Self::Dwab),
            10 => Ok(Self::Htj2k256),
            11 => Ok(Self::Htj2k32),
            12 => Ok(Self::Lj2k),
            13 => Ok(Self::Zstd),
            _ => Err(ExrError::InvalidHeader("unknown compression enum value")),
        }
    }

    /// Number of scanlines grouped into each scanline chunk (`docs/01-file-layout.md` Section 7.2).
    #[inline]
    pub const fn scanlines_per_chunk(self) -> usize {
        match self {
            Self::None | Self::Rle | Self::Zips | Self::Zstd => 1,
            Self::Zip | Self::Pxr24 => 16,
            Self::Piz | Self::B44 | Self::B44a | Self::Dwaa | Self::Htj2k32 => 32,
            Self::Dwab | Self::Htj2k256 | Self::Lj2k => 256,
        }
    }

    /// Whether this compression method is permitted on deep parts (`None`, `Rle`, `Zips`, `Zstd`).
    #[inline]
    pub const fn is_valid_for_deep(self) -> bool {
        matches!(self, Self::None | Self::Rle | Self::Zips | Self::Zstd)
    }
}

/// OpenEXR chunk line ordering (`0..=2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrLineOrder {
    /// `0` — Increasing Y (`INCREASING_Y`).
    IncreasingY = 0,
    /// `1` — Decreasing Y (`DECREASING_Y`).
    DecreasingY = 1,
    /// `2` — Random Y (`RANDOM_Y`, valid only for tiled parts).
    RandomY = 2,
}

impl ExrLineOrder {
    /// Parses a lineOrder byte (`0..=2`).
    pub fn from_u8(v: u8) -> Result<Self, ExrError> {
        match v {
            0 => Ok(Self::IncreasingY),
            1 => Ok(Self::DecreasingY),
            2 => Ok(Self::RandomY),
            _ => Err(ExrError::InvalidHeader("unknown lineOrder enum value")),
        }
    }
}

/// OpenEXR part storage type (`scanlineimage`, `tiledimage`, `deepscanline`, `deeptile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrPartType {
    /// Flat scanline image (`scanlineimage`).
    ScanlineImage,
    /// Flat tiled image (`tiledimage`).
    TiledImage,
    /// Deep scanline image (`deepscanline`).
    DeepScanline,
    /// Deep tiled image (`deeptile`).
    DeepTile,
}

impl ExrPartType {
    /// Parses a standard part `type` string.
    pub fn from_str_name(s: &str) -> Result<Self, ExrError> {
        match s {
            "scanlineimage" => Ok(Self::ScanlineImage),
            "tiledimage" => Ok(Self::TiledImage),
            "deepscanline" => Ok(Self::DeepScanline),
            "deeptile" => Ok(Self::DeepTile),
            _ => Err(ExrError::InvalidHeader(
                "invalid part type attribute string",
            )),
        }
    }

    /// Canonical wire string for the `type` attribute.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScanlineImage => "scanlineimage",
            Self::TiledImage => "tiledimage",
            Self::DeepScanline => "deepscanline",
            Self::DeepTile => "deeptile",
        }
    }

    /// Whether this part type is tiled (`TiledImage` or `DeepTile`).
    #[inline]
    pub const fn is_tiled(self) -> bool {
        matches!(self, Self::TiledImage | Self::DeepTile)
    }

    /// Whether this part type is deep (`DeepScanline` or `DeepTile`).
    #[inline]
    pub const fn is_deep(self) -> bool {
        matches!(self, Self::DeepScanline | Self::DeepTile)
    }
}

/// Inclusive 2D integer bounding box (`min_x..=max_x`, `min_y..=max_y`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExrBox2i {
    /// Minimum X coordinate (inclusive).
    pub min_x: i32,
    /// Minimum Y coordinate (inclusive).
    pub min_y: i32,
    /// Maximum X coordinate (inclusive).
    pub max_x: i32,
    /// Maximum Y coordinate (inclusive).
    pub max_y: i32,
}

impl ExrBox2i {
    /// Creates an inclusive bounding box `[0, 0]..=[width - 1, height - 1]`.
    pub const fn from_dimensions(width: u32, height: u32) -> Self {
        Self {
            min_x: 0,
            min_y: 0,
            max_x: (width as i32) - 1,
            max_y: (height as i32) - 1,
        }
    }

    /// Width in pixels (`max_x - min_x + 1`).
    #[inline]
    pub fn width(self) -> u32 {
        (i64::from(self.max_x) - i64::from(self.min_x) + 1).max(0) as u32
    }

    /// Height in pixels (`max_y - min_y + 1`).
    #[inline]
    pub fn height(self) -> u32 {
        (i64::from(self.max_y) - i64::from(self.min_y) + 1).max(0) as u32
    }
}

/// Inclusive 2D float bounding box (`box2f`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExrBox2f {
    /// Minimum X coordinate.
    pub min_x: f32,
    /// Minimum Y coordinate.
    pub min_y: f32,
    /// Maximum X coordinate.
    pub max_x: f32,
    /// Maximum Y coordinate.
    pub max_y: f32,
}

/// Tile level mode (`ONE_LEVEL = 0`, `MIPMAP_LEVELS = 1`, `RIPMAP_LEVELS = 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrLevelMode {
    /// Single resolution level (`0`).
    OneLevel = 0,
    /// Square power-of-two mipmap pyramid (`1`).
    MipmapLevels = 1,
    /// Independent X/Y power-of-two ripmap levels (`2`).
    RipmapLevels = 2,
}

/// Level rounding mode (`ROUND_DOWN = 0`, `ROUND_UP = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrLevelRoundingMode {
    /// Floor division when computing level dimensions (`0`).
    RoundDown = 0,
    /// Ceiling division when computing level dimensions (`1`).
    RoundUp = 1,
}

/// OpenEXR `tiledesc` attribute (`x_size`, `y_size`, `level_mode`, `rounding_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExrTileDesc {
    /// Tile width in pixels (`> 0`).
    pub x_size: u32,
    /// Tile height in pixels (`> 0`).
    pub y_size: u32,
    /// Resolution level mode.
    pub level_mode: ExrLevelMode,
    /// Level dimension rounding mode.
    pub rounding_mode: ExrLevelRoundingMode,
}

impl ExrTileDesc {
    /// Computes the number of `(num_x_levels, num_y_levels)` for a given `data_window`
    /// (`docs/06-tiled-and-deep-data.md` Section 1.2).
    pub fn level_counts(&self, dw: ExrBox2i) -> (u32, u32) {
        let w = dw.width();
        let h = dw.height();
        let r = self.rounding_mode;
        match self.level_mode {
            ExrLevelMode::OneLevel => (1, 1),
            ExrLevelMode::MipmapLevels => {
                let n = Self::num_levels_1d(w.max(h), r);
                (n, n)
            }
            ExrLevelMode::RipmapLevels => (Self::num_levels_1d(w, r), Self::num_levels_1d(h, r)),
        }
    }

    fn num_levels_1d(mut size: u32, rounding: ExrLevelRoundingMode) -> u32 {
        if size == 0 {
            return 1;
        }
        let mut count = 1u32;
        while size > 1 {
            size = match rounding {
                ExrLevelRoundingMode::RoundDown => size / 2,
                ExrLevelRoundingMode::RoundUp => size.div_ceil(2),
            };
            count += 1;
        }
        count
    }

    /// Computes a single dimension at level `l` (`0` is full resolution).
    pub fn level_size_1d(full_size: u32, level: u32, rounding: ExrLevelRoundingMode) -> u32 {
        if level == 0 {
            return full_size.max(1);
        }
        if level >= 31 {
            return 1;
        }
        let div = 1u32 << level;
        let res = match rounding {
            ExrLevelRoundingMode::RoundDown => full_size / div,
            ExrLevelRoundingMode::RoundUp => full_size.div_ceil(div),
        };
        res.max(1)
    }

    /// Computes `(level_width, level_height)` for level `(lx, ly)`.
    pub fn level_dimensions(&self, dw: ExrBox2i, lx: u32, ly: u32) -> (u32, u32) {
        let w = Self::level_size_1d(dw.width(), lx, self.rounding_mode);
        let h = Self::level_size_1d(dw.height(), ly, self.rounding_mode);
        (w, h)
    }

    /// Computes `(num_tiles_x, num_tiles_y)` at level `(lx, ly)`.
    pub fn tile_counts(&self, dw: ExrBox2i, lx: u32, ly: u32) -> (u32, u32) {
        let (lw, lh) = self.level_dimensions(dw, lx, ly);
        let tx = lw.div_ceil(self.x_size.max(1));
        let ty = lh.div_ceil(self.y_size.max(1));
        (tx, ty)
    }

    /// Total number of tile chunks across all valid `(lx, ly)` levels for `dw`.
    pub fn total_tiles(&self, dw: ExrBox2i) -> Result<usize, ExrError> {
        let (nlx, nly) = self.level_counts(dw);
        let mut total = 0usize;
        match self.level_mode {
            ExrLevelMode::OneLevel => {
                let (tx, ty) = self.tile_counts(dw, 0, 0);
                total = (tx as usize)
                    .checked_mul(ty as usize)
                    .ok_or(ExrError::DimensionsTooLarge)?;
            }
            ExrLevelMode::MipmapLevels => {
                for l in 0..nlx {
                    let (tx, ty) = self.tile_counts(dw, l, l);
                    let lvl_tiles = (tx as usize)
                        .checked_mul(ty as usize)
                        .ok_or(ExrError::DimensionsTooLarge)?;
                    total = total
                        .checked_add(lvl_tiles)
                        .ok_or(ExrError::DimensionsTooLarge)?;
                }
            }
            ExrLevelMode::RipmapLevels => {
                for ly in 0..nly {
                    for lx in 0..nlx {
                        let (tx, ty) = self.tile_counts(dw, lx, ly);
                        let lvl_tiles = (tx as usize)
                            .checked_mul(ty as usize)
                            .ok_or(ExrError::DimensionsTooLarge)?;
                        total = total
                            .checked_add(lvl_tiles)
                            .ok_or(ExrError::DimensionsTooLarge)?;
                    }
                }
            }
        }
        Ok(total)
    }
}

/// Channel descriptor from the `chlist` attribute (`docs/02-header-and-attributes.md` Section 3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrChannelDesc {
    /// Channel name (e.g. `"A"`, `"B"`, `"G"`, `"R"`, `"Z"`).
    pub name: String,
    /// Sample storage type (`Uint`, `Half`, `Float`).
    pub sample_type: ExrSampleType,
    /// Plausibility / perceptual linearity hint (`0` or `1`, used by `B44` / `B44A`).
    pub p_linear: u8,
    /// Horizontal subsampling factor (`>= 1`).
    pub x_sampling: i32,
    /// Vertical subsampling factor (`>= 1`).
    pub y_sampling: i32,
}

impl ExrChannelDesc {
    /// Creates a standard unsubsampled (`1x1`) channel descriptor.
    pub fn new(name: impl Into<String>, sample_type: ExrSampleType) -> Self {
        Self {
            name: name.into(),
            sample_type,
            p_linear: 0,
            x_sampling: 1,
            y_sampling: 1,
        }
    }

    /// Number of samples for this channel on scanline `y` (in `dataWindow` coordinates)
    /// across a region of width `region_width`.
    #[inline]
    pub fn samples_in_line(&self, y: i32, region_width: usize) -> usize {
        let xs = (self.x_sampling.max(1)) as usize;
        let ys = self.y_sampling.max(1);
        if ys > 1 && y.rem_euclid(ys) != 0 {
            0
        } else if xs > 1 {
            region_width / xs
        } else {
            region_width
        }
    }
}

/// CIE xy chromaticities (`chromaticities` attribute, 32 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExrChromaticities {
    /// Red primary `(x, y)`.
    pub red: (f32, f32),
    /// Green primary `(x, y)`.
    pub green: (f32, f32),
    /// Blue primary `(x, y)`.
    pub blue: (f32, f32),
    /// White point `(x, y)`.
    pub white: (f32, f32),
}

/// Preview thumbnail attribute (`preview`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrPreview {
    /// Thumbnail width in pixels.
    pub width: u32,
    /// Thumbnail height in pixels.
    pub height: u32,
    /// Raw RGBA8 pixels (`4 * width * height` bytes).
    pub rgba: Vec<u8>,
}

/// OpenEXR attribute value covering all standard predefined types plus `Opaque` pass-through.
#[derive(Debug, Clone, PartialEq)]
pub enum ExrAttributeValue {
    /// `box2i` (16 bytes).
    Box2i(ExrBox2i),
    /// `box2f` (16 bytes).
    Box2f(ExrBox2f),
    /// `chlist` (variable).
    Chlist(Vec<ExrChannelDesc>),
    /// `chromaticities` (32 bytes).
    Chromaticities(ExrChromaticities),
    /// `compression` (1 byte).
    Compression(ExrCompression),
    /// `double` (8 bytes).
    Double(f64),
    /// `envmap` (1 byte: `0 = ENVMAP_LATLONG`, `1 = ENVMAP_CUBE`).
    Envmap(u8),
    /// `float` (4 bytes).
    Float(f32),
    /// `floatvector` (variable `4 * N` bytes).
    FloatVector(Vec<f32>),
    /// `int` (4 bytes).
    Int(i32),
    /// `keycode` (28 bytes: `7 * i32`).
    Keycode([i32; 7]),
    /// `lineOrder` (1 byte).
    LineOrder(ExrLineOrder),
    /// `m33f` (36 bytes).
    M33f([f32; 9]),
    /// `m33d` (72 bytes).
    M33d([f64; 9]),
    /// `m44f` (64 bytes).
    M44f([f32; 16]),
    /// `m44d` (128 bytes).
    M44d([f64; 16]),
    /// `preview` (`8 + 4 * w * h` bytes).
    Preview(ExrPreview),
    /// `rational` (8 bytes: `num: i32, den: u32`).
    Rational(i32, u32),
    /// `string` (exact `size` bytes).
    String(String),
    /// `stringvector` (length-prefixed strings).
    StringVector(Vec<String>),
    /// `tiledesc` (9 bytes).
    TileDesc(ExrTileDesc),
    /// `timecode` (8 bytes: `[u32; 2]`).
    Timecode([u32; 2]),
    /// `v2i` (8 bytes).
    V2i([i32; 2]),
    /// `v2f` (8 bytes).
    V2f([f32; 2]),
    /// `v2d` (16 bytes).
    V2d([f64; 2]),
    /// `v3i` (12 bytes).
    V3i([i32; 3]),
    /// `v3f` (12 bytes).
    V3f([f32; 3]),
    /// `v3d` (24 bytes).
    V3d([f64; 3]),
    /// `bytes` or custom/unrecognized attribute preserved verbatim for round-trip.
    Opaque {
        /// Attribute type name as stored in the file.
        type_name: String,
        /// Raw payload bytes.
        data: Vec<u8>,
    },
}

/// Named OpenEXR header attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrAttribute {
    /// Attribute name (e.g. `"channels"`, `"compression"`, `"comments"`).
    pub name: String,
    /// Parsed or opaque attribute value.
    pub value: ExrAttributeValue,
}

/// Parsed and validated header for a single OpenEXR part.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrPartHeader {
    /// Sorted channel list (`channels` attribute, sorted by ASCII byte order).
    pub channels: Vec<ExrChannelDesc>,
    /// Compression method (`compression` attribute).
    pub compression: ExrCompression,
    /// Pixel data bounding box (`dataWindow` attribute).
    pub data_window: ExrBox2i,
    /// Display bounding box (`displayWindow` attribute).
    pub display_window: ExrBox2i,
    /// Scanline / tile ordering (`lineOrder` attribute).
    pub line_order: ExrLineOrder,
    /// Pixel aspect ratio (`pixelAspectRatio` attribute).
    pub pixel_aspect_ratio: f32,
    /// Perspective screen window center (`screenWindowCenter` attribute).
    pub screen_window_center: [f32; 2],
    /// Perspective screen window width (`screenWindowWidth` attribute).
    pub screen_window_width: f32,
    /// Optional part name (`name` attribute, required in multi-part / deep files).
    pub name: Option<String>,
    /// Resolved part storage type (`ScanlineImage`, `TiledImage`, `DeepScanline`, `DeepTile`).
    pub part_type: ExrPartType,
    /// Optional explicit `type` attribute string if present in the header.
    pub explicit_type_attr: Option<ExrPartType>,
    /// Optional tile geometry (`tiles` attribute, required for tiled parts).
    pub tiles: Option<ExrTileDesc>,
    /// Optional chunk count (`chunkCount` attribute).
    pub chunk_count: Option<i32>,
    /// Optional deep format version (`version` attribute, must be `1` for deep parts).
    pub deep_version: Option<i32>,
    /// Optional maximum samples per pixel (`maxSamplesPerPixel` attribute for deep parts).
    pub max_samples_per_pixel: Option<i32>,
    /// Additional / custom attributes preserved in order.
    pub other_attributes: Vec<ExrAttribute>,
}

impl ExrPartHeader {
    /// Creates a default single-part scanline header for a `width x height` image.
    pub fn new_scanline(
        width: u32,
        height: u32,
        compression: ExrCompression,
        mut channels: Vec<ExrChannelDesc>,
    ) -> Self {
        channels.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        let win = ExrBox2i::from_dimensions(width, height);
        Self {
            channels,
            compression,
            data_window: win,
            display_window: win,
            line_order: ExrLineOrder::IncreasingY,
            pixel_aspect_ratio: 1.0,
            screen_window_center: [0.0, 0.0],
            screen_window_width: 1.0,
            name: None,
            part_type: ExrPartType::ScanlineImage,
            explicit_type_attr: None,
            tiles: None,
            chunk_count: None,
            deep_version: None,
            max_samples_per_pixel: None,
            other_attributes: Vec::new(),
        }
    }

    /// Computes the exact number of chunks expected for this part based on its geometry
    /// (`docs/01-file-layout.md` Section 7.2 and `docs/06-tiled-and-deep-data.md`).
    pub fn compute_expected_chunk_count(&self) -> Result<usize, ExrError> {
        match self.part_type {
            ExrPartType::ScanlineImage | ExrPartType::DeepScanline => {
                let h = self.data_window.height() as usize;
                if h == 0 {
                    return Ok(0);
                }
                let lines = self.compression.scanlines_per_chunk();
                Ok(h.div_ceil(lines))
            }
            ExrPartType::TiledImage | ExrPartType::DeepTile => {
                let td = self.tiles.ok_or(ExrError::InvalidHeader(
                    "tiled part missing tiles attribute",
                ))?;
                td.total_tiles(self.data_window)
            }
        }
    }

    /// Returns whether any attribute name or channel name exceeds 31 bytes (requiring flag `0x400`).
    pub fn requires_long_names(&self) -> bool {
        if self.channels.iter().any(|c| c.name.len() > 31) {
            return true;
        }
        self.other_attributes.iter().any(|a| {
            a.name.len() > 31
                || matches!(&a.value, ExrAttributeValue::Opaque { type_name, .. } if type_name.len() > 31)
        })
    }
}

/// Complete parsed OpenEXR file header (version field + 1 or more part headers).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrFileHeader {
    /// File format version number (`1` or `2`).
    pub version: u8,
    /// Raw 32-bit version/flags word (`version | flags`).
    pub flags: u32,
    /// Validated headers for each part (`parts.len() >= 1`).
    pub parts: Vec<ExrPartHeader>,
}

impl ExrFileHeader {
    /// Returns true if the `0x1000` multi-part flag is set.
    #[inline]
    pub const fn is_multipart(&self) -> bool {
        (self.flags & EXR_FLAG_MULTIPART) != 0
    }

    /// Returns true if the `0x0800` deep/non-image flag is set.
    #[inline]
    pub const fn is_deep(&self) -> bool {
        (self.flags & EXR_FLAG_DEEP) != 0
    }

    /// Returns true if the `0x0200` single-part tiled flag is set.
    #[inline]
    pub const fn is_single_tiled(&self) -> bool {
        (self.flags & EXR_FLAG_TILED) != 0
    }
}

/// Parses and validates an OpenEXR file header from the start of `data`, returning
/// `(ExrFileHeader, offset_after_headers)`.
pub fn parse_file_header(
    data: &[u8],
    max_width: u32,
    max_height: u32,
    strict: bool,
) -> Result<(ExrFileHeader, usize), ExrError> {
    if data.len() < 8 {
        return Err(ExrError::UnexpectedEof);
    }
    if data[0..4] != EXR_MAGIC {
        return Err(ExrError::InvalidMagic);
    }

    let version_word = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let version = (version_word & 0xff) as u8;
    let flags = version_word & !0xff;

    if version != 1 && version != 2 {
        return Err(ExrError::UnsupportedVersion(version_word));
    }
    if (flags & !EXR_VALID_FLAGS_MASK) != 0 {
        return Err(ExrError::InvalidHeader("unknown version flag bits set"));
    }

    let is_tiled = (flags & EXR_FLAG_TILED) != 0;
    let is_long_names = (flags & EXR_FLAG_LONG_NAMES) != 0;
    let is_deep = (flags & EXR_FLAG_DEEP) != 0;
    let is_multipart = (flags & EXR_FLAG_MULTIPART) != 0;

    // Normative rule from `OpenEXRCore/parse_header.c`:
    // If single-part tiled bit (0x200) is set, deep (0x800) and multipart (0x1000) MUST be 0.
    if is_tiled && (is_deep || is_multipart) {
        return Err(ExrError::InvalidHeader(
            "single-part tiled flag (0x200) cannot be combined with deep (0x800) or multipart (0x1000)",
        ));
    }

    let max_name_len = if is_long_names { 255usize } else { 31usize };
    let mut pos = 8usize;
    let mut parts = Vec::new();

    if is_multipart {
        loop {
            if pos >= data.len() {
                return Err(ExrError::UnexpectedEof);
            }
            if data[pos] == 0 {
                pos += 1;
                break;
            }
            let part = parse_single_part_header(
                data,
                &mut pos,
                max_name_len,
                is_tiled,
                is_deep,
                is_multipart,
                strict,
            )?;
            parts.push(part);
        }
        if parts.is_empty() {
            return Err(ExrError::InvalidHeader("multipart file has zero parts"));
        }
    } else {
        if pos >= data.len() || data[pos] == 0 {
            return Err(ExrError::InvalidHeader("empty single-part header"));
        }
        let part = parse_single_part_header(
            data,
            &mut pos,
            max_name_len,
            is_tiled,
            is_deep,
            is_multipart,
            strict,
        )?;
        parts.push(part);
    }

    // Validate each part and cross-part constraints
    let mut seen_names = Vec::with_capacity(parts.len());
    for (idx, part) in parts.iter().enumerate() {
        validate_part_header(
            part,
            is_tiled,
            is_deep,
            is_multipart,
            max_width,
            max_height,
            strict,
        )?;
        if is_multipart {
            let pname = part.name.as_deref().ok_or(ExrError::InvalidHeader(
                "multipart header missing required 'name' attribute",
            ))?;
            if seen_names.contains(&pname) {
                return Err(ExrError::InvalidHeader(
                    "duplicate part 'name' in multipart file",
                ));
            }
            seen_names.push(pname);

            if idx > 0 {
                validate_shared_multipart_attributes(&parts[0], part)?;
            }
        }
    }

    Ok((
        ExrFileHeader {
            version,
            flags,
            parts,
        },
        pos,
    ))
}

fn parse_single_part_header(
    data: &[u8],
    pos: &mut usize,
    max_name_len: usize,
    is_single_tiled: bool,
    is_deep_flag: bool,
    is_multipart: bool,
    strict: bool,
) -> Result<ExrPartHeader, ExrError> {
    let mut channels: Option<Vec<ExrChannelDesc>> = None;
    let mut compression: Option<ExrCompression> = None;
    let mut data_window: Option<ExrBox2i> = None;
    let mut display_window: Option<ExrBox2i> = None;
    let mut line_order: Option<ExrLineOrder> = None;
    let mut pixel_aspect_ratio: Option<f32> = None;
    let mut screen_window_center: Option<[f32; 2]> = None;
    let mut screen_window_width: Option<f32> = None;
    let mut name: Option<String> = None;
    let mut explicit_type_attr: Option<ExrPartType> = None;
    let mut tiles: Option<ExrTileDesc> = None;
    let mut chunk_count: Option<i32> = None;
    let mut deep_version: Option<i32> = None;
    let mut max_samples_per_pixel: Option<i32> = None;
    let mut other_attributes: Vec<ExrAttribute> = Vec::new();
    let mut seen_attr_names: Vec<String> = Vec::new();

    loop {
        if *pos >= data.len() {
            return Err(ExrError::UnexpectedEof);
        }
        if data[*pos] == 0 {
            *pos += 1;
            break;
        }

        let attr_name = read_cstring(data, pos, max_name_len, strict)?;
        if attr_name.is_empty() {
            return Err(ExrError::InvalidHeader("empty attribute name"));
        }
        if seen_attr_names.iter().any(|s| s == &attr_name) {
            return Err(ExrError::InvalidHeader(
                "duplicate attribute name in header",
            ));
        }
        seen_attr_names.push(attr_name.clone());

        let type_name = read_cstring(data, pos, max_name_len, strict)?;
        if type_name.is_empty() {
            return Err(ExrError::InvalidHeader("empty attribute type name"));
        }

        if *pos + 4 > data.len() {
            return Err(ExrError::UnexpectedEof);
        }
        let size_i32 =
            i32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos += 4;
        if size_i32 < 0 {
            return Err(ExrError::InvalidHeader("negative attribute payload size"));
        }
        let size = size_i32 as usize;
        if *pos + size > data.len() {
            return Err(ExrError::UnexpectedEof);
        }
        let payload = &data[*pos..*pos + size];
        *pos += size;

        // Core required/standard attributes must have their canonical wire type
        match attr_name.as_str() {
            "channels" => {
                if type_name != "chlist" {
                    return Err(ExrError::InvalidHeader(
                        "channels attribute must have type chlist",
                    ));
                }
                channels = Some(parse_chlist(payload, max_name_len, strict)?);
            }
            "compression" => {
                if type_name != "compression" || payload.len() != 1 {
                    return Err(ExrError::InvalidHeader(
                        "compression attribute must have type compression and size 1",
                    ));
                }
                compression = Some(ExrCompression::from_u8(payload[0])?);
            }
            "dataWindow" => {
                if type_name != "box2i" || payload.len() != 16 {
                    return Err(ExrError::InvalidHeader(
                        "dataWindow attribute must have type box2i and size 16",
                    ));
                }
                data_window = Some(parse_box2i(payload));
            }
            "displayWindow" => {
                if type_name != "box2i" || payload.len() != 16 {
                    return Err(ExrError::InvalidHeader(
                        "displayWindow attribute must have type box2i and size 16",
                    ));
                }
                display_window = Some(parse_box2i(payload));
            }
            "lineOrder" => {
                if type_name != "lineOrder" || payload.len() != 1 {
                    return Err(ExrError::InvalidHeader(
                        "lineOrder attribute must have type lineOrder and size 1",
                    ));
                }
                line_order = Some(ExrLineOrder::from_u8(payload[0])?);
            }
            "pixelAspectRatio" => {
                if type_name != "float" || payload.len() != 4 {
                    return Err(ExrError::InvalidHeader(
                        "pixelAspectRatio attribute must have type float and size 4",
                    ));
                }
                pixel_aspect_ratio = Some(read_f32_le(payload));
            }
            "screenWindowCenter" => {
                if type_name != "v2f" || payload.len() != 8 {
                    return Err(ExrError::InvalidHeader(
                        "screenWindowCenter attribute must have type v2f and size 8",
                    ));
                }
                screen_window_center =
                    Some([read_f32_le(&payload[0..4]), read_f32_le(&payload[4..8])]);
            }
            "screenWindowWidth" => {
                if type_name != "float" || payload.len() != 4 {
                    return Err(ExrError::InvalidHeader(
                        "screenWindowWidth attribute must have type float and size 4",
                    ));
                }
                screen_window_width = Some(read_f32_le(payload));
            }
            "name" => {
                if type_name != "string" {
                    return Err(ExrError::InvalidHeader(
                        "name attribute must have type string",
                    ));
                }
                let s = parse_string_payload(payload)?;
                if s.is_empty() {
                    return Err(ExrError::InvalidHeader("part name cannot be empty"));
                }
                name = Some(s);
            }
            "type" => {
                if type_name != "string" {
                    return Err(ExrError::InvalidHeader(
                        "type attribute must have type string",
                    ));
                }
                let s = parse_string_payload(payload)?;
                explicit_type_attr = Some(ExrPartType::from_str_name(&s)?);
            }
            "tiles" => {
                if type_name != "tiledesc" || payload.len() != 9 {
                    return Err(ExrError::InvalidHeader(
                        "tiles attribute must have type tiledesc and size 9",
                    ));
                }
                tiles = Some(parse_tiledesc(payload)?);
            }
            "chunkCount" => {
                if type_name != "int" || payload.len() != 4 {
                    return Err(ExrError::InvalidHeader(
                        "chunkCount attribute must have type int and size 4",
                    ));
                }
                chunk_count = Some(read_i32_le(payload));
            }
            "version" => {
                if type_name != "int" || payload.len() != 4 {
                    return Err(ExrError::InvalidHeader(
                        "version attribute must have type int and size 4",
                    ));
                }
                deep_version = Some(read_i32_le(payload));
            }
            "maxSamplesPerPixel" => {
                if type_name != "int" || payload.len() != 4 {
                    return Err(ExrError::InvalidHeader(
                        "maxSamplesPerPixel attribute must have type int and size 4",
                    ));
                }
                max_samples_per_pixel = Some(read_i32_le(payload));
            }
            _ => {
                let val = parse_attribute_value(&type_name, payload, max_name_len, strict)?;
                other_attributes.push(ExrAttribute {
                    name: attr_name,
                    value: val,
                });
            }
        }
    }

    let channels = channels.ok_or(ExrError::InvalidHeader("missing channels attribute"))?;
    let compression =
        compression.ok_or(ExrError::InvalidHeader("missing compression attribute"))?;
    let data_window = data_window.ok_or(ExrError::InvalidHeader("missing dataWindow attribute"))?;
    let display_window =
        display_window.ok_or(ExrError::InvalidHeader("missing displayWindow attribute"))?;
    let line_order = line_order.ok_or(ExrError::InvalidHeader("missing lineOrder attribute"))?;
    let pixel_aspect_ratio = pixel_aspect_ratio.ok_or(ExrError::InvalidHeader(
        "missing pixelAspectRatio attribute",
    ))?;
    let screen_window_center = screen_window_center.ok_or(ExrError::InvalidHeader(
        "missing screenWindowCenter attribute",
    ))?;
    let screen_window_width = screen_window_width.ok_or(ExrError::InvalidHeader(
        "missing screenWindowWidth attribute",
    ))?;

    // Resolve part_type from explicit `type` attribute or version flags
    let part_type = if let Some(exp) = explicit_type_attr {
        exp
    } else if is_multipart || is_deep_flag {
        return Err(ExrError::InvalidHeader(
            "multipart or deep header missing required 'type' attribute",
        ));
    } else if is_single_tiled {
        ExrPartType::TiledImage
    } else {
        ExrPartType::ScanlineImage
    };

    Ok(ExrPartHeader {
        channels,
        compression,
        data_window,
        display_window,
        line_order,
        pixel_aspect_ratio,
        screen_window_center,
        screen_window_width,
        name,
        part_type,
        explicit_type_attr,
        tiles,
        chunk_count,
        deep_version,
        max_samples_per_pixel,
        other_attributes,
    })
}

fn validate_part_header(
    part: &ExrPartHeader,
    is_single_tiled: bool,
    is_deep_flag: bool,
    is_multipart: bool,
    max_width: u32,
    max_height: u32,
    strict: bool,
) -> Result<(), ExrError> {
    // 1. Validate part_type vs file flags (`OpenEXRCore/validation.c: internal_exr_validate_shared_attrs`)
    if !is_multipart {
        if is_deep_flag {
            if !part.part_type.is_deep() {
                return Err(ExrError::InvalidHeader(
                    "deep flag (0x800) set on single-part file with non-deep type",
                ));
            }
        } else if is_single_tiled {
            if part.part_type != ExrPartType::TiledImage && strict {
                return Err(ExrError::InvalidHeader(
                    "single-part tiled flag (0x200) conflicts with explicit 'type' attribute",
                ));
            }
        } else if part.part_type != ExrPartType::ScanlineImage && strict {
            return Err(ExrError::InvalidHeader(
                "single-part scanline flags conflict with explicit 'type' attribute",
            ));
        }
    } else if !is_deep_flag && part.part_type.is_deep() && strict {
        return Err(ExrError::InvalidHeader(
            "deep part present in multipart file without deep flag (0x800)",
        ));
    }

    // 2. Validate dataWindow and displayWindow bounds (`OpenEXRCore/validation.c`)
    validate_window_bounds(part.data_window, "dataWindow")?;
    validate_window_bounds(part.display_window, "displayWindow")?;

    let w = part.data_window.width();
    let h = part.data_window.height();
    if w == 0 || h == 0 {
        return Err(ExrError::InvalidHeader(
            "dataWindow has zero width or height",
        ));
    }
    if max_width > 0 && w > max_width {
        return Err(ExrError::DimensionsTooLarge);
    }
    if max_height > 0 && h > max_height {
        return Err(ExrError::DimensionsTooLarge);
    }

    // 3. Validate pixelAspectRatio & screenWindowWidth
    if !part.pixel_aspect_ratio.is_normal()
        || part.pixel_aspect_ratio < 1e-6
        || part.pixel_aspect_ratio > 1e6
    {
        return Err(ExrError::InvalidHeader("invalid pixelAspectRatio"));
    }
    if part.screen_window_width.is_nan() || part.screen_window_width < 0.0 {
        return Err(ExrError::InvalidHeader("invalid screenWindowWidth"));
    }

    // 4. Validate lineOrder (`RANDOM_Y` is only allowed on tiled parts)
    if part.line_order == ExrLineOrder::RandomY && !part.part_type.is_tiled() {
        return Err(ExrError::InvalidHeader(
            "RANDOM_Y lineOrder is only permitted for tiled parts",
        ));
    }

    // 5. Validate tiles attribute presence and geometry
    if part.part_type.is_tiled() {
        let td = part.tiles.ok_or(ExrError::InvalidHeader(
            "tiled part missing required 'tiles' attribute",
        ))?;
        if td.x_size == 0
            || td.y_size == 0
            || td.x_size > (EXR_MAX_COORD_BOUND as u32) / 4
            || td.y_size > (EXR_MAX_COORD_BOUND as u32) / 4
            || td.x_size > DEFAULT_MAX_TILE_DIM
            || td.y_size > DEFAULT_MAX_TILE_DIM
        {
            return Err(ExrError::InvalidHeader("invalid tile dimensions"));
        }
    } else if part.tiles.is_some() && strict {
        return Err(ExrError::InvalidHeader(
            "non-tiled part must not contain 'tiles' attribute",
        ));
    }

    // 6. Validate deep part constraints
    if part.part_type.is_deep() {
        if !part.compression.is_valid_for_deep() {
            return Err(ExrError::InvalidHeader(
                "invalid compression method for deep data part",
            ));
        }
        if let Some(v) = part.deep_version
            && v != 1
        {
            return Err(ExrError::InvalidHeader("deep part 'version' must be 1"));
        }
        if let Some(m) = part.max_samples_per_pixel
            && m < 0
        {
            return Err(ExrError::InvalidHeader(
                "negative maxSamplesPerPixel in deep part",
            ));
        }
    }

    // 7. Validate channels (`chlist`)
    if part.channels.is_empty() {
        return Err(ExrError::InvalidHeader(
            "chlist must contain at least 1 channel",
        ));
    }
    for i in 0..part.channels.len() {
        let ch = &part.channels[i];
        if i > 0 && part.channels[i - 1].name == ch.name {
            return Err(ExrError::InvalidHeader("duplicate channel name in chlist"));
        }
        if ch.p_linear > 1 {
            return Err(ExrError::InvalidHeader("channel pLinear must be 0 or 1"));
        }
        if ch.x_sampling < 1 || ch.y_sampling < 1 {
            return Err(ExrError::InvalidHeader(
                "channel xSampling and ySampling must be >= 1",
            ));
        }
        if (part.part_type.is_tiled() || part.part_type.is_deep())
            && (ch.x_sampling != 1 || ch.y_sampling != 1)
        {
            return Err(ExrError::InvalidHeader(
                "subsampled channels are not permitted in tiled or deep parts",
            ));
        }
        if ch.x_sampling > 1
            && (part.data_window.min_x % ch.x_sampling != 0 || (w as i32) % ch.x_sampling != 0)
        {
            return Err(ExrError::InvalidHeader(
                "dataWindow X bounds are not divisible by channel xSampling",
            ));
        }
        if ch.y_sampling > 1
            && (part.data_window.min_y % ch.y_sampling != 0 || (h as i32) % ch.y_sampling != 0)
        {
            return Err(ExrError::InvalidHeader(
                "dataWindow Y bounds are not divisible by channel ySampling",
            ));
        }
    }

    // 8. Validate chunkCount if present
    let expected_chunks = part.compute_expected_chunk_count()?;
    if let Some(cc) = part.chunk_count
        && (cc < 0 || (cc as usize) != expected_chunks)
    {
        return Err(ExrError::InvalidHeader(
            "chunkCount attribute does not match computed chunk count",
        ));
    }

    Ok(())
}

fn validate_window_bounds(b: ExrBox2i, _name: &'static str) -> Result<(), ExrError> {
    if b.min_x > b.max_x
        || b.min_y > b.max_y
        || b.min_x <= -EXR_MAX_COORD_BOUND
        || b.min_y <= -EXR_MAX_COORD_BOUND
        || b.max_x >= EXR_MAX_COORD_BOUND
        || b.max_y >= EXR_MAX_COORD_BOUND
    {
        return Err(ExrError::InvalidHeader(
            "window coordinates out of valid bounds",
        ));
    }
    Ok(())
}

/// Validates that shared attributes across parts in a multipart file match Part 0
/// (`OpenEXRCore/validation.c: internal_exr_validate_shared_attrs`).
fn validate_shared_multipart_attributes(
    part0: &ExrPartHeader,
    other: &ExrPartHeader,
) -> Result<(), ExrError> {
    if part0.display_window != other.display_window {
        return Err(ExrError::InvalidHeader(
            "multipart displayWindow mismatch across parts",
        ));
    }
    if part0.pixel_aspect_ratio.to_bits() != other.pixel_aspect_ratio.to_bits() {
        return Err(ExrError::InvalidHeader(
            "multipart pixelAspectRatio mismatch across parts",
        ));
    }

    // If `chromaticities` or `timecode` is present in either part, they must match across parts
    for shared_name in ["chromaticities", "timecode"] {
        let a0 = part0
            .other_attributes
            .iter()
            .find(|a| a.name == shared_name)
            .map(|a| &a.value);
        let a1 = other
            .other_attributes
            .iter()
            .find(|a| a.name == shared_name)
            .map(|a| &a.value);
        if a0 != a1 {
            return Err(ExrError::InvalidHeader(
                "multipart shared attribute (chromaticities/timecode) mismatch across parts",
            ));
        }
    }
    Ok(())
}

fn read_cstring(
    data: &[u8],
    pos: &mut usize,
    max_len: usize,
    strict: bool,
) -> Result<String, ExrError> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != 0 {
        *pos += 1;
    }
    if *pos >= data.len() {
        return Err(ExrError::UnexpectedEof);
    }
    let len = *pos - start;
    *pos += 1; // skip NUL
    if strict && len > max_len {
        return Err(ExrError::InvalidHeader(
            "name exceeds maximum allowed length (31 or 255)",
        ));
    }
    String::from_utf8(data[start..start + len].to_vec())
        .map_err(|_| ExrError::InvalidHeader("non-UTF8 identifier in header"))
}

fn parse_chlist(
    payload: &[u8],
    max_name_len: usize,
    strict: bool,
) -> Result<Vec<ExrChannelDesc>, ExrError> {
    let mut pos = 0usize;
    let mut channels = Vec::new();
    loop {
        if pos >= payload.len() {
            return Err(ExrError::InvalidHeader(
                "unterminated chlist attribute payload",
            ));
        }
        if payload[pos] == 0 {
            pos += 1;
            break;
        }
        let name = read_cstring(payload, &mut pos, max_name_len, strict)?;
        if name.is_empty() {
            return Err(ExrError::InvalidHeader("empty channel name in chlist"));
        }
        if pos + 16 > payload.len() {
            return Err(ExrError::InvalidHeader("truncated channel entry in chlist"));
        }
        let sample_type_raw = read_u32_le(&payload[pos..pos + 4]);
        let sample_type = ExrSampleType::from_u32(sample_type_raw)?;
        let p_linear = payload[pos + 4];
        let x_sampling = read_i32_le(&payload[pos + 8..pos + 12]);
        let y_sampling = read_i32_le(&payload[pos + 12..pos + 16]);
        pos += 16;

        channels.push(ExrChannelDesc {
            name,
            sample_type,
            p_linear,
            x_sampling,
            y_sampling,
        });
    }
    if pos != payload.len() {
        return Err(ExrError::InvalidHeader("trailing bytes in chlist payload"));
    }
    // Canonical OpenEXR channel order is ASCII byte order by channel name
    channels.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    Ok(channels)
}

fn parse_box2i(payload: &[u8]) -> ExrBox2i {
    ExrBox2i {
        min_x: read_i32_le(&payload[0..4]),
        min_y: read_i32_le(&payload[4..8]),
        max_x: read_i32_le(&payload[8..12]),
        max_y: read_i32_le(&payload[12..16]),
    }
}

fn parse_box2f(payload: &[u8]) -> ExrBox2f {
    ExrBox2f {
        min_x: read_f32_le(&payload[0..4]),
        min_y: read_f32_le(&payload[4..8]),
        max_x: read_f32_le(&payload[8..12]),
        max_y: read_f32_le(&payload[12..16]),
    }
}

fn parse_tiledesc(payload: &[u8]) -> Result<ExrTileDesc, ExrError> {
    let x_size = read_u32_le(&payload[0..4]);
    let y_size = read_u32_le(&payload[4..8]);
    let mode_byte = payload[8];
    let level_bits = mode_byte & 0x0f;
    let round_bits = (mode_byte >> 4) & 0x0f;

    let level_mode = match level_bits {
        0 => ExrLevelMode::OneLevel,
        1 => ExrLevelMode::MipmapLevels,
        2 => ExrLevelMode::RipmapLevels,
        _ => return Err(ExrError::InvalidHeader("invalid tiledesc level mode")),
    };
    let rounding_mode = match round_bits {
        0 => ExrLevelRoundingMode::RoundDown,
        1 => ExrLevelRoundingMode::RoundUp,
        _ => return Err(ExrError::InvalidHeader("invalid tiledesc rounding mode")),
    };

    Ok(ExrTileDesc {
        x_size,
        y_size,
        level_mode,
        rounding_mode,
    })
}

fn parse_string_payload(payload: &[u8]) -> Result<String, ExrError> {
    String::from_utf8(payload.to_vec())
        .map_err(|_| ExrError::InvalidHeader("invalid UTF-8 string attribute"))
}

fn parse_attribute_value(
    type_name: &str,
    payload: &[u8],
    max_name_len: usize,
    strict: bool,
) -> Result<ExrAttributeValue, ExrError> {
    match type_name {
        "box2i" => {
            if payload.len() != 16 {
                return Err(ExrError::InvalidHeader("box2i attribute size != 16"));
            }
            Ok(ExrAttributeValue::Box2i(parse_box2i(payload)))
        }
        "box2f" => {
            if payload.len() != 16 {
                return Err(ExrError::InvalidHeader("box2f attribute size != 16"));
            }
            Ok(ExrAttributeValue::Box2f(parse_box2f(payload)))
        }
        "chlist" => Ok(ExrAttributeValue::Chlist(parse_chlist(
            payload,
            max_name_len,
            strict,
        )?)),
        "chromaticities" => {
            if payload.len() != 32 {
                return Err(ExrError::InvalidHeader(
                    "chromaticities attribute size != 32",
                ));
            }
            Ok(ExrAttributeValue::Chromaticities(ExrChromaticities {
                red: (read_f32_le(&payload[0..4]), read_f32_le(&payload[4..8])),
                green: (read_f32_le(&payload[8..12]), read_f32_le(&payload[12..16])),
                blue: (read_f32_le(&payload[16..20]), read_f32_le(&payload[20..24])),
                white: (read_f32_le(&payload[24..28]), read_f32_le(&payload[28..32])),
            }))
        }
        "compression" => {
            if payload.len() != 1 {
                return Err(ExrError::InvalidHeader("compression attribute size != 1"));
            }
            Ok(ExrAttributeValue::Compression(ExrCompression::from_u8(
                payload[0],
            )?))
        }
        "double" => {
            if payload.len() != 8 {
                return Err(ExrError::InvalidHeader("double attribute size != 8"));
            }
            Ok(ExrAttributeValue::Double(read_f64_le(payload)))
        }
        "envmap" => {
            if payload.len() != 1 || payload[0] > 1 {
                return Err(ExrError::InvalidHeader("invalid envmap attribute"));
            }
            Ok(ExrAttributeValue::Envmap(payload[0]))
        }
        "float" => {
            if payload.len() != 4 {
                return Err(ExrError::InvalidHeader("float attribute size != 4"));
            }
            Ok(ExrAttributeValue::Float(read_f32_le(payload)))
        }
        "floatvector" => {
            if payload.len() % 4 != 0 {
                return Err(ExrError::InvalidHeader(
                    "floatvector attribute size not divisible by 4",
                ));
            }
            let mut v = Vec::with_capacity(payload.len() / 4);
            for chunk in payload.as_chunks::<4>().0 {
                v.push(read_f32_le(chunk));
            }
            Ok(ExrAttributeValue::FloatVector(v))
        }
        "int" => {
            if payload.len() != 4 {
                return Err(ExrError::InvalidHeader("int attribute size != 4"));
            }
            Ok(ExrAttributeValue::Int(read_i32_le(payload)))
        }
        "keycode" => {
            if payload.len() != 28 {
                return Err(ExrError::InvalidHeader("keycode attribute size != 28"));
            }
            let mut kc = [0i32; 7];
            for (i, chunk) in payload.as_chunks::<4>().0.iter().enumerate() {
                kc[i] = read_i32_le(chunk);
            }
            Ok(ExrAttributeValue::Keycode(kc))
        }
        "lineOrder" => {
            if payload.len() != 1 {
                return Err(ExrError::InvalidHeader("lineOrder attribute size != 1"));
            }
            Ok(ExrAttributeValue::LineOrder(ExrLineOrder::from_u8(
                payload[0],
            )?))
        }
        "m33f" => {
            if payload.len() != 36 {
                return Err(ExrError::InvalidHeader("m33f attribute size != 36"));
            }
            let mut m = [0.0f32; 9];
            for (i, chunk) in payload.as_chunks::<4>().0.iter().enumerate() {
                m[i] = read_f32_le(chunk);
            }
            Ok(ExrAttributeValue::M33f(m))
        }
        "m33d" => {
            if payload.len() != 72 {
                return Err(ExrError::InvalidHeader("m33d attribute size != 72"));
            }
            let mut m = [0.0f64; 9];
            for (i, chunk) in payload.as_chunks::<8>().0.iter().enumerate() {
                m[i] = read_f64_le(chunk);
            }
            Ok(ExrAttributeValue::M33d(m))
        }
        "m44f" => {
            if payload.len() != 64 {
                return Err(ExrError::InvalidHeader("m44f attribute size != 64"));
            }
            let mut m = [0.0f32; 16];
            for (i, chunk) in payload.as_chunks::<4>().0.iter().enumerate() {
                m[i] = read_f32_le(chunk);
            }
            Ok(ExrAttributeValue::M44f(m))
        }
        "m44d" => {
            if payload.len() != 128 {
                return Err(ExrError::InvalidHeader("m44d attribute size != 128"));
            }
            let mut m = [0.0f64; 16];
            for (i, chunk) in payload.as_chunks::<8>().0.iter().enumerate() {
                m[i] = read_f64_le(chunk);
            }
            Ok(ExrAttributeValue::M44d(m))
        }
        "preview" => {
            if payload.len() < 8 {
                return Err(ExrError::InvalidHeader("preview attribute < 8 bytes"));
            }
            let width = read_u32_le(&payload[0..4]);
            let height = read_u32_le(&payload[4..8]);
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|px| px.checked_mul(4))
                .and_then(|b| b.checked_add(8))
                .ok_or(ExrError::InvalidHeader("preview dimensions overflow"))?;
            if payload.len() != expected {
                return Err(ExrError::InvalidHeader("preview payload size mismatch"));
            }
            Ok(ExrAttributeValue::Preview(ExrPreview {
                width,
                height,
                rgba: payload[8..].to_vec(),
            }))
        }
        "rational" => {
            if payload.len() != 8 {
                return Err(ExrError::InvalidHeader("rational attribute size != 8"));
            }
            Ok(ExrAttributeValue::Rational(
                read_i32_le(&payload[0..4]),
                read_u32_le(&payload[4..8]),
            ))
        }
        "string" => {
            let s = String::from_utf8_lossy(payload).into_owned();
            Ok(ExrAttributeValue::String(s))
        }
        "stringvector" => {
            let mut pos = 0usize;
            let mut vec_str = Vec::new();
            while pos < payload.len() {
                if pos + 4 > payload.len() {
                    return Err(ExrError::InvalidHeader("truncated stringvector entry"));
                }
                let slen_i32 = read_i32_le(&payload[pos..pos + 4]);
                pos += 4;
                if slen_i32 < 0 || pos + (slen_i32 as usize) > payload.len() {
                    return Err(ExrError::InvalidHeader("invalid stringvector entry length"));
                }
                let slen = slen_i32 as usize;
                let s = String::from_utf8_lossy(&payload[pos..pos + slen]).into_owned();
                pos += slen;
                vec_str.push(s);
            }
            Ok(ExrAttributeValue::StringVector(vec_str))
        }
        "tiledesc" => {
            if payload.len() != 9 {
                return Err(ExrError::InvalidHeader("tiledesc attribute size != 9"));
            }
            Ok(ExrAttributeValue::TileDesc(parse_tiledesc(payload)?))
        }
        "timecode" => {
            if payload.len() != 8 {
                return Err(ExrError::InvalidHeader("timecode attribute size != 8"));
            }
            Ok(ExrAttributeValue::Timecode([
                read_u32_le(&payload[0..4]),
                read_u32_le(&payload[4..8]),
            ]))
        }
        "v2i" => {
            if payload.len() != 8 {
                return Err(ExrError::InvalidHeader("v2i attribute size != 8"));
            }
            Ok(ExrAttributeValue::V2i([
                read_i32_le(&payload[0..4]),
                read_i32_le(&payload[4..8]),
            ]))
        }
        "v2f" => {
            if payload.len() != 8 {
                return Err(ExrError::InvalidHeader("v2f attribute size != 8"));
            }
            Ok(ExrAttributeValue::V2f([
                read_f32_le(&payload[0..4]),
                read_f32_le(&payload[4..8]),
            ]))
        }
        "v2d" => {
            if payload.len() != 16 {
                return Err(ExrError::InvalidHeader("v2d attribute size != 16"));
            }
            Ok(ExrAttributeValue::V2d([
                read_f64_le(&payload[0..8]),
                read_f64_le(&payload[8..16]),
            ]))
        }
        "v3i" => {
            if payload.len() != 12 {
                return Err(ExrError::InvalidHeader("v3i attribute size != 12"));
            }
            Ok(ExrAttributeValue::V3i([
                read_i32_le(&payload[0..4]),
                read_i32_le(&payload[4..8]),
                read_i32_le(&payload[8..12]),
            ]))
        }
        "v3f" => {
            if payload.len() != 12 {
                return Err(ExrError::InvalidHeader("v3f attribute size != 12"));
            }
            Ok(ExrAttributeValue::V3f([
                read_f32_le(&payload[0..4]),
                read_f32_le(&payload[4..8]),
                read_f32_le(&payload[8..12]),
            ]))
        }
        "v3d" => {
            if payload.len() != 24 {
                return Err(ExrError::InvalidHeader("v3d attribute size != 24"));
            }
            Ok(ExrAttributeValue::V3d([
                read_f64_le(&payload[0..8]),
                read_f64_le(&payload[8..16]),
                read_f64_le(&payload[16..24]),
            ]))
        }
        _ => Ok(ExrAttributeValue::Opaque {
            type_name: type_name.to_string(),
            data: payload.to_vec(),
        }),
    }
}

/// Serializes an OpenEXR file header (magic, version word, and all part headers) into `out`.
pub fn write_file_header(parts: &[ExrPartHeader], out: &mut Vec<u8>) -> Result<(), ExrError> {
    if parts.is_empty() {
        return Err(ExrError::InvalidHeader("cannot write file with 0 parts"));
    }
    let is_multipart = parts.len() > 1;
    let is_deep = parts.iter().any(|p| p.part_type.is_deep());
    let is_single_tiled = !is_multipart && !is_deep && parts[0].part_type.is_tiled();
    let is_long_names = parts.iter().any(ExrPartHeader::requires_long_names);

    let mut flags = 2u32;
    if is_single_tiled {
        flags |= EXR_FLAG_TILED;
    }
    if is_long_names {
        flags |= EXR_FLAG_LONG_NAMES;
    }
    if is_deep {
        flags |= EXR_FLAG_DEEP;
    }
    if is_multipart {
        flags |= EXR_FLAG_MULTIPART;
    }

    out.extend_from_slice(&EXR_MAGIC);
    out.extend_from_slice(&flags.to_le_bytes());

    for (idx, part) in parts.iter().enumerate() {
        write_single_part_header(part, idx, is_multipart, is_deep, out)?;
    }

    if is_multipart {
        // Multi-part end-of-headers empty header (0x00)
        out.push(0);
    }

    Ok(())
}

fn write_single_part_header(
    part: &ExrPartHeader,
    part_idx: usize,
    is_multipart: bool,
    is_deep: bool,
    out: &mut Vec<u8>,
) -> Result<(), ExrError> {
    // 1. channels (chlist)
    let mut sorted_channels = part.channels.clone();
    sorted_channels.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut ch_payload = Vec::new();
    for ch in &sorted_channels {
        ch_payload.extend_from_slice(ch.name.as_bytes());
        ch_payload.push(0);
        ch_payload.extend_from_slice(&(ch.sample_type as u32).to_le_bytes());
        ch_payload.push(ch.p_linear);
        ch_payload.extend_from_slice(&[0, 0, 0]);
        ch_payload.extend_from_slice(&ch.x_sampling.to_le_bytes());
        ch_payload.extend_from_slice(&ch.y_sampling.to_le_bytes());
    }
    ch_payload.push(0);
    write_raw_attr(out, "channels", "chlist", &ch_payload);

    // 2. chunkCount (required for multi-part or deep)
    let expected_chunks = part.compute_expected_chunk_count()? as i32;
    if is_multipart || is_deep || part.chunk_count.is_some() {
        write_raw_attr(out, "chunkCount", "int", &expected_chunks.to_le_bytes());
    }

    // 3. compression
    write_raw_attr(out, "compression", "compression", &[part.compression as u8]);

    // 4. dataWindow
    write_box2i_attr(out, "dataWindow", part.data_window);

    // 5. displayWindow
    write_box2i_attr(out, "displayWindow", part.display_window);

    // 6. lineOrder
    write_raw_attr(out, "lineOrder", "lineOrder", &[part.line_order as u8]);

    // 7. maxSamplesPerPixel & version for deep parts
    if part.part_type.is_deep() {
        let max_s = part.max_samples_per_pixel.unwrap_or(1);
        write_raw_attr(out, "maxSamplesPerPixel", "int", &max_s.to_le_bytes());
        let ver = part.deep_version.unwrap_or(1);
        write_raw_attr(out, "version", "int", &ver.to_le_bytes());
    }

    // 8. name (required for multi-part)
    if is_multipart || part.name.is_some() {
        let default_name = format!("part{part_idx}");
        let pname = part.name.as_deref().unwrap_or(&default_name);
        write_raw_attr(out, "name", "string", pname.as_bytes());
    }

    // 9. pixelAspectRatio
    write_raw_attr(
        out,
        "pixelAspectRatio",
        "float",
        &part.pixel_aspect_ratio.to_le_bytes(),
    );

    // 10. screenWindowCenter
    let mut swc = [0u8; 8];
    swc[0..4].copy_from_slice(&part.screen_window_center[0].to_le_bytes());
    swc[4..8].copy_from_slice(&part.screen_window_center[1].to_le_bytes());
    write_raw_attr(out, "screenWindowCenter", "v2f", &swc);

    // 11. screenWindowWidth
    write_raw_attr(
        out,
        "screenWindowWidth",
        "float",
        &part.screen_window_width.to_le_bytes(),
    );

    // 12. tiles (for tiled parts)
    if let Some(td) = part.tiles {
        let mut td_bytes = [0u8; 9];
        td_bytes[0..4].copy_from_slice(&td.x_size.to_le_bytes());
        td_bytes[4..8].copy_from_slice(&td.y_size.to_le_bytes());
        td_bytes[8] = (td.level_mode as u8) | ((td.rounding_mode as u8) << 4);
        write_raw_attr(out, "tiles", "tiledesc", &td_bytes);
    }

    // 13. type (required for multi-part or deep)
    if is_multipart || is_deep || part.explicit_type_attr.is_some() {
        write_raw_attr(out, "type", "string", part.part_type.as_str().as_bytes());
    }

    // 14. Additional attributes
    for attr in &part.other_attributes {
        write_attribute(out, attr);
    }

    // End of part header
    out.push(0);
    Ok(())
}

fn write_raw_attr(out: &mut Vec<u8>, name: &str, type_name: &str, payload: &[u8]) {
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(type_name.as_bytes());
    out.push(0);
    out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
    out.extend_from_slice(payload);
}

fn write_box2i_attr(out: &mut Vec<u8>, name: &str, b: ExrBox2i) {
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&b.min_x.to_le_bytes());
    buf[4..8].copy_from_slice(&b.min_y.to_le_bytes());
    buf[8..12].copy_from_slice(&b.max_x.to_le_bytes());
    buf[12..16].copy_from_slice(&b.max_y.to_le_bytes());
    write_raw_attr(out, name, "box2i", &buf);
}

fn write_attribute(out: &mut Vec<u8>, attr: &ExrAttribute) {
    match &attr.value {
        ExrAttributeValue::Box2i(b) => write_box2i_attr(out, &attr.name, *b),
        ExrAttributeValue::Box2f(b) => {
            let mut buf = [0u8; 16];
            buf[0..4].copy_from_slice(&b.min_x.to_le_bytes());
            buf[4..8].copy_from_slice(&b.min_y.to_le_bytes());
            buf[8..12].copy_from_slice(&b.max_x.to_le_bytes());
            buf[12..16].copy_from_slice(&b.max_y.to_le_bytes());
            write_raw_attr(out, &attr.name, "box2f", &buf);
        }
        ExrAttributeValue::Chlist(list) => {
            let mut p = Vec::new();
            for ch in list {
                p.extend_from_slice(ch.name.as_bytes());
                p.push(0);
                p.extend_from_slice(&(ch.sample_type as u32).to_le_bytes());
                p.push(ch.p_linear);
                p.extend_from_slice(&[0, 0, 0]);
                p.extend_from_slice(&ch.x_sampling.to_le_bytes());
                p.extend_from_slice(&ch.y_sampling.to_le_bytes());
            }
            p.push(0);
            write_raw_attr(out, &attr.name, "chlist", &p);
        }
        ExrAttributeValue::Chromaticities(c) => {
            let mut buf = [0u8; 32];
            for (i, v) in [
                c.red.0, c.red.1, c.green.0, c.green.1, c.blue.0, c.blue.1, c.white.0, c.white.1,
            ]
            .into_iter()
            .enumerate()
            {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "chromaticities", &buf);
        }
        ExrAttributeValue::Compression(c) => {
            write_raw_attr(out, &attr.name, "compression", &[*c as u8]);
        }
        ExrAttributeValue::Double(d) => {
            write_raw_attr(out, &attr.name, "double", &d.to_le_bytes());
        }
        ExrAttributeValue::Envmap(e) => {
            write_raw_attr(out, &attr.name, "envmap", &[*e]);
        }
        ExrAttributeValue::Float(f) => {
            write_raw_attr(out, &attr.name, "float", &f.to_le_bytes());
        }
        ExrAttributeValue::FloatVector(fv) => {
            let mut p = Vec::with_capacity(fv.len() * 4);
            for f in fv {
                p.extend_from_slice(&f.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "floatvector", &p);
        }
        ExrAttributeValue::Int(v) => {
            write_raw_attr(out, &attr.name, "int", &v.to_le_bytes());
        }
        ExrAttributeValue::Keycode(kc) => {
            let mut buf = [0u8; 28];
            for (i, v) in kc.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "keycode", &buf);
        }
        ExrAttributeValue::LineOrder(lo) => {
            write_raw_attr(out, &attr.name, "lineOrder", &[*lo as u8]);
        }
        ExrAttributeValue::M33f(m) => {
            let mut buf = [0u8; 36];
            for (i, v) in m.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "m33f", &buf);
        }
        ExrAttributeValue::M33d(m) => {
            let mut buf = [0u8; 72];
            for (i, v) in m.iter().enumerate() {
                buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "m33d", &buf);
        }
        ExrAttributeValue::M44f(m) => {
            let mut buf = [0u8; 64];
            for (i, v) in m.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "m44f", &buf);
        }
        ExrAttributeValue::M44d(m) => {
            let mut buf = [0u8; 128];
            for (i, v) in m.iter().enumerate() {
                buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "m44d", &buf);
        }
        ExrAttributeValue::Preview(p) => {
            let mut buf = Vec::with_capacity(8 + p.rgba.len());
            buf.extend_from_slice(&p.width.to_le_bytes());
            buf.extend_from_slice(&p.height.to_le_bytes());
            buf.extend_from_slice(&p.rgba);
            write_raw_attr(out, &attr.name, "preview", &buf);
        }
        ExrAttributeValue::Rational(n, d) => {
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&n.to_le_bytes());
            buf[4..8].copy_from_slice(&d.to_le_bytes());
            write_raw_attr(out, &attr.name, "rational", &buf);
        }
        ExrAttributeValue::String(s) => {
            write_raw_attr(out, &attr.name, "string", s.as_bytes());
        }
        ExrAttributeValue::StringVector(sv) => {
            let mut buf = Vec::new();
            for s in sv {
                buf.extend_from_slice(&(s.len() as i32).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            write_raw_attr(out, &attr.name, "stringvector", &buf);
        }
        ExrAttributeValue::TileDesc(td) => {
            let mut buf = [0u8; 9];
            buf[0..4].copy_from_slice(&td.x_size.to_le_bytes());
            buf[4..8].copy_from_slice(&td.y_size.to_le_bytes());
            buf[8] = (td.level_mode as u8) | ((td.rounding_mode as u8) << 4);
            write_raw_attr(out, &attr.name, "tiledesc", &buf);
        }
        ExrAttributeValue::Timecode(tc) => {
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&tc[0].to_le_bytes());
            buf[4..8].copy_from_slice(&tc[1].to_le_bytes());
            write_raw_attr(out, &attr.name, "timecode", &buf);
        }
        ExrAttributeValue::V2i(v) => {
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&v[0].to_le_bytes());
            buf[4..8].copy_from_slice(&v[1].to_le_bytes());
            write_raw_attr(out, &attr.name, "v2i", &buf);
        }
        ExrAttributeValue::V2f(v) => {
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&v[0].to_le_bytes());
            buf[4..8].copy_from_slice(&v[1].to_le_bytes());
            write_raw_attr(out, &attr.name, "v2f", &buf);
        }
        ExrAttributeValue::V2d(v) => {
            let mut buf = [0u8; 16];
            buf[0..8].copy_from_slice(&v[0].to_le_bytes());
            buf[8..16].copy_from_slice(&v[1].to_le_bytes());
            write_raw_attr(out, &attr.name, "v2d", &buf);
        }
        ExrAttributeValue::V3i(v) => {
            let mut buf = [0u8; 12];
            for (i, x) in v.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "v3i", &buf);
        }
        ExrAttributeValue::V3f(v) => {
            let mut buf = [0u8; 12];
            for (i, x) in v.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "v3f", &buf);
        }
        ExrAttributeValue::V3d(v) => {
            let mut buf = [0u8; 24];
            for (i, x) in v.iter().enumerate() {
                buf[i * 8..i * 8 + 8].copy_from_slice(&x.to_le_bytes());
            }
            write_raw_attr(out, &attr.name, "v3d", &buf);
        }
        ExrAttributeValue::Opaque { type_name, data } => {
            write_raw_attr(out, &attr.name, type_name, data);
        }
    }
}

#[inline]
fn read_i32_le(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_f64_le(b: &[u8]) -> f64 {
    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}
