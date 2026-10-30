// SPDX-License-Identifier: Apache-2.0
//! JPEG XL and OpenEXR image pipeline with pure-Rust color management for `rustmpeg-next`.
//!
//! This crate implements the image and texture subsystem defined in `docs/CONCEPT.md`:
//!
//! * [`jxl`] — ISO/IEC 18181-1 (core codestream) and ISO/IEC 18181-2 (file format)
//!   JPEG XL v2 decoder and clean-room encoder (Modular lossless/near-lossless and VarDCT lossy);
//! * [`exr`] — OpenEXR v2/v3 decoder and encoder (OpenEXRCore-derived algorithms and
//!   tables, BSD-3-Clause) supporting single-part and
//!   multi-part scanline, tiled (`OneLevel`, `MipmapLevels`, `RipmapLevels`), deep
//!   scanline, and deep tiled images across `None`, `Rle`, `Zips`, `Zip`, `Piz`,
//!   `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, and `Zstd` codecs;
//! * [`cms`] — pure-Rust ICCv4 and CICP color management built on `moxcms`.
//!
//! Legacy formats (JPEG/JFIF, PNG, GIF, WebP, HEIC, AVIF) and legacy JPEG bitstream
//! reconstruction (`jbrd`) are excluded by architecture policy.

#![forbid(unsafe_code)]
#![feature(portable_simd)]

pub mod cms;
pub mod exr;
pub mod jxl;

pub use cms::{CmsError, CmsProfile, CmsRenderingIntent, CmsTransform};
pub use exr::{
    ExrAttribute, ExrAttributeValue, ExrBox2f, ExrBox2i, ExrChannelData, ExrChannelDesc,
    ExrChannelPlane, ExrChromaticities, ExrCompression, ExrConfig, ExrDecoder, ExrDeepSamples,
    ExrEncoder, ExrError, ExrFileHeader, ExrImage, ExrLevelImage, ExrLevelMode,
    ExrLevelRoundingMode, ExrLineOrder, ExrPart, ExrPartHeader, ExrPartType, ExrPreview,
    ExrSampleType, ExrTileDesc,
};
pub use jxl::{
    JxlBitDepth, JxlColorEncoding, JxlConfig, JxlContainerKind, JxlDecoder, JxlEncoder,
    JxlEncodingMode, JxlEntropyMode, JxlError, JxlFrame, JxlImage, JxlPixelLayout,
};
