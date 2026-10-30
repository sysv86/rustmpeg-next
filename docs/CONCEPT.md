# Concept and specification of the rustmpeg-next multimedia framework formats

## Architectural doctrine and basic principles

rustmpeg-next is a high-performance, full-cycle multimedia framework (demuxing, decoding, filtering, encoding, muxing, and network transport) developed from scratch in Rust without using the FFmpeg codebase and without unsafe C/C++ wrappers (`*-sys` crates). The FFmpeg toolkit serves exclusively as an algorithmic and reference baseline for cross-validating output streams and verifying specification compliance.

### Rolling-Window Obsolescence policy

Classic multimedia frameworks (FFmpeg, GStreamer) carry a quarter-century of technical debt by supporting hundreds of legacy formats, which critically expands the attack surface and prevents deep optimization of the data pipeline. The rustmpeg-next architecture introduces a strict "one task — one current generation of the standard" doctrine with zero backward compatibility. Formats and codecs from previous generations (MPEG-2, H.264/AVC, H.265/HEVC, AV1, VP8, VP9, MP3, Vorbis, AAC, AC-3, DTS, JPEG, PNG, WebP, AVI, FLV) are forcibly blocked at the architectural level and are never included in the software core. The lifecycle of every supported standard follows the Hard Drop rule: once the next generation of a codec is standardized and stabilized in production, its predecessor is marked Deprecated for exactly one minor release and then physically removed from the source tree. The AOMedia slot is occupied by AV2: AV1 is not implemented and is excluded from the matrix together with the still-image profile that depends on it (AVIF).

### Memory safety, SIMD, and the hardware stack

Container and bitstream parsers have historically formed the primary memory-vulnerability zone (buffer overflow, integer overflow, use-after-free) in C/C++ multimedia libraries. In rustmpeg-next, the `#![forbid(unsafe_code)]` compiler directive is mandatory in demuxers, header parsers, and containers. Packet routing is built on the tokio asynchronous runtime and immutable shared `Arc<[u8]>` slices, which guarantees zero-copy data reads without redundant allocations or thread blocking.
To reach performance parity with low-level C implementations and external assemblers (NASM), the project rejects automatic translation via C2Rust, which produces unsafe code in early translations, relying instead on pure vectorization and a unified GPU stack. Compute cores use compiler autovectorization and the cross-platform `std::simd` module (portable_simd), which provides direct access to LLVM vector intrinsics on x86_64, aarch64, and wasm32 architectures. Reference Rust implementations (symphonia, zune-jpeg, png) demonstrate that this approach delivers decoding speed at parity with or 10–50% higher than FFmpeg's C code while fully preserving memory safety. Hardware acceleration rejects fragmented platform APIs (VA-API, NVDEC/NVENC, VideoToolbox, DXVA2) and fixed-function Vulkan Video in favour of a single cross-platform compute backend: Vulkan Compute through `wgpu`, with WGSL kernels compiled at runtime by `naga`. Vulkan Video is not used because its codec operations cover only AV1, H.264/AVC, H.265/HEVC, and VP9 — none of the codecs in the matrix. Frame-level block-parallel stages run as compute kernels when they are measured to be faster; parsing, entropy coding, block decisions, and per-block work stay on the CPU. The same GPU stack is designed to perform colour correction, scaling, and HDR tone mapping (SMPTE ST 2094-50).

## Video codec matrix

The video stack is divided strictly into three functional domains: mass delivery and web streaming, ultra-high-density broadcasting (UHD/8K), and professional studio production (Mezzanine/Production). Software support for H.264/AVC and H.265/HEVC is completely excluded, including hardware passthrough: a legacy codec does not enter the tree even as a fixed-function path. The VP8, VP9, MPEG-2, ProRes, and DNxHD codecs are rejected definitively.

| Codec | Application domain | Status in rustmpeg-next | Technical rationale and hardware profile | Hard Drop trigger |
| :---- | :---- | :---- | :---- | :---- |
| **AV2** | Next-generation streaming, AR/VR, 8K VOD | Primary (experimental software encoder/decoder; Vulkan Compute metrics) | The AOMedia specification finalized in 2026, providing a fundamental compression-efficiency gain over AV1. Designated as the main video codec of the architecture. The software core is written in safe Rust; frame-level metrics are accelerated and the opt-in deblocking kernel is bit-exact but currently slower than the CPU. Edge sizes that are not superblock-aligned remain non-conformant. | Release of the next AOMedia standard generation. |
| **H.266 / VVC** | Ultra-high resolutions (4K/8K), DVB/OTT broadcasting | Primary (intra-only encoder + strict subset decoder; full VVC pending) | Direct replacement for the outdated H.265/HEVC. Supports 10/12-bit profiles, 4:2:0 and 4:4:4 subsampling, and the vvc1, vvi1, and Annex-B packaging formats in the target design; the implemented profile is Main 10 8-bit 4:2:0. Vulkan Compute accelerates RGBA→I420 and metrics; transforms, prediction, and CABAC stay on the CPU until a whole-frame VRAM-resident batch exists. Fixed-function vendor APIs are not used. | Release and hardware adoption of the ITU-T H.267 (ECM) standard. |
| **APV (RFC 9924)** | Studio editing (NLE), RAW capture, ultra-low latency | Implemented (Mezzanine / Pro Video; intra encoder + decoder, experimental, CPU) | Open SMPTE / IETF RFC 9924 standard (ASWF OpenAPV reference), displacing closed Apple ProRes and Avid DNxHR. Intra-only visually lossless compression with support for 2K/4K/8K, 10–16-bit depth, color layouts from 4:0:0 to 4:4:4:4, and independent frame tiling. Planned acceleration through Vulkan Compute shaders once the core exists. | Standardization of a new generation of SMPTE/IETF intra-frame codecs. |

## Heterogeneous CPU/GPU execution

Acceleration is scheduled per stage, not per codec. At startup the framework builds two capability profiles:

- **CPU:** available SIMD width, core count, cache sizes, and the measured cost of the baseline `std::simd` kernels.
- **GPU:** adapter limits, workgroup sizes, memory heaps, queue capabilities, and the measured cost of dispatch and read-back.

A stage is placed on the GPU only when its total cost, including transfer, dispatch, synchronization, and read-back, is lower than the CPU cost by a measured margin. The default split is:

- **CPU:** bitstream parsing, entropy coding (CABAC, range coding, APV entropy), block-level decisions, reference management, and control flow. These stages are serial or branch-heavy and do not vectorize.
- **GPU:** colour conversion, scaling, tone mapping, distortion metrics, and loop filters whose parameters are already decided, plus transforms/prediction/motion compensation only when they operate on a whole-frame or whole-tile batch that is already resident in VRAM. These stages are block-parallel but must be issued as large batches.

Per-block dispatch is forbidden: at 32x32 or 64x64 the upload, dispatch, and read-back cost more than the arithmetic. The AV2 deblocking filter was implemented on both paths as a reference; it is bit-exact on the GPU, but the transfer-bound path is slower at 640x360 through 4K, so the CPU filter remains the default.

Frames and coefficient planes should stay in VRAM between consecutive GPU stages, so a decode or encode pass performs one upload and one read-back per pass rather than a round trip per kernel. The runtime provides a batched submission recorder for this. Small pictures, short clips, and machines without a usable adapter run the same pipeline on the CPU through the `std::simd` fallback. Both paths must be bit-exact: every GPU kernel is validated against a CPU reference, and a mismatch disables the GPU path rather than changing the output.

The full mandatory rules for future codecs and codec updates are in [`AGENTS.md`](../AGENTS.md#heterogeneous-cpugpu-execution). GPU acceleration is relevant to video codecs today (AV2 and H.266/VVC, with APV when its core lands). Audio codecs, containers, and network transports stay on the CPU.

## Audio codec matrix and spatial audio

The audio subsystem completely breaks with the psychoacoustic models of the late 1990s and with proprietary object-based formats. The MP1, MP2, MP3, Vorbis, all AAC profiles (LC, HE v1/v2), the Dolby family (AC-3, E-AC-3, Atmos), DTS, ALAC, WavPack, ADPCM, and Bluetooth SBC codecs are removed from the architecture. Although the IAMF specification formally permits the use of legacy AAC-LC for backward compatibility, the rustmpeg-next parser will forcibly reject such streams, allowing only combinations with Opus, FLAC, or LPCM inside immersive containers. Implementation status: FLAC, LPCM, LC3, the Opus packet/range layer, and the SILK decoder (NB/MB/WB, 10–60 ms, mono/stereo, resampling) are implemented; CELT, hybrid, PLC, and IAMF/OAR are planned.

| Format / Codec | Application domain | Status in rustmpeg-next | Technical rationale and architectural features |
| :---- | :---- | :---- | :---- |
| **Opus** | Streaming, VoIP, WebRTC, game audio, media containers | Primary (Lossy Universal; packet/range + SILK decoder implemented, CELT/hybrid/PLC pending) | Outperforms AAC-LC, HE-AAC, and Vorbis at all bitrates with minimal algorithmic delay. Serves as the only universal lossy audio codec for stationary and web scenarios. |
| **IAMF (Eclipsa Audio)** | Spatial audio (3D Audio), home theater, VR/AR | Planned (Immersive / Spatial; not started) | Open, royalty-free AOMedia container and metadata standard (v1.1 / v2.0), developed with the participation of Google and Samsung as a direct replacement for Dolby Atmos and MPEG-H 3D Audio. Encapsulates Opus, FLAC, or LPCM streams, combining channel layers (up to 7.1.4) and Ambisonics scenes (FOA/HOA, up to 28 channels in the Base-Enhanced profile) with binaural or speaker rendering according to the Open Audio Renderer (OAR) standard. |
| **FLAC & LPCM** | Archival storage, studio mastering, inter-node exchange | Primary (Lossless / Raw) | Lossless compression (FLAC up to 32 bit / 96+ kHz) and raw streams (pcm_s24le, pcm_s32le, pcm_f32le, pcm_f64le). The pure-Rust FLAC decoder is implemented from RFC 9639 and preserves 100% memory safety; performance parity with FFmpeg is not yet benchmarked. |
| **LC3 / LC3plus** | Wireless audio (Bluetooth LE Audio, Auracast), IoT | Primary (Low-Power Wireless) | The mandatory codec of the Bluetooth 5.2/5.4+ specification (LE Audio), replacing the outdated SBC. Reduces radio latency to ~20 ms (versus 100–200 ms in classic A2DP) and provides native support for isochronous channels and Auracast broadcast mode (CIS/BIS). |

## Image format and texture matrix

Unlike FFmpeg, where image handling has historically been built as a special case of single-frame video with legacy 8-bit color spaces, rustmpeg-next separates graphics into a specialized pipeline with HDR support by default, wide color gamuts (Display P3, Rec.2020), and pure-Rust color management. Implementation status: JPEG XL (`JxlDecoder` + clean-room `JxlEncoder`), OpenEXR v2 (`ExrDecoder` + `ExrEncoder`, OpenEXRCore-derived, BSD-3-Clause), and `moxcms` ICCv4/CICP color management are implemented in `rustmpeg-image`. Classic 8-bit JPEG (JFIF), PNG, GIF, WebP (based on the dead VP8 video codec), and HEIC (based on the patented H.265) are completely excluded. AVIF is excluded as well because it reuses the AV1 core, which is not in the matrix; an AV2-based still format will be considered once AV2-Still is standardized.

| Format | Application domain | Status in rustmpeg-next | Technical rationale and color handling |
| :---- | :---- | :---- | :---- |
| **JPEG XL (JXL)** | Photography, web graphics, textures, medical and scientific imaging | Implemented (v1 decoder + clean-room `ModularLossless`/`VarDctLossy` encoder) | A single replacement for JPEG, PNG, GIF, and TIFF. Supports high bit depth, HDR, progressive decoding, alpha channel, depth maps, animation, and lossless compression. The implementation follows the jxl-oxide / jxl-rs architecture with ICCv4 color profiles handled by the pure-Rust moxcms crate (a complete rejection of the C lcms2 library). The legacy JPEG bitstream reconstruction feature is forcibly removed. |
| **OpenEXR** | VFX, 3D rendering, compositing, studio grading (DI) | Implemented (v2 decoder + encoder; OpenEXRCore-derived, BSD-3-Clause: `None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`, scanline/tiled/deep/multipart) | The industry standard for storing multi-layer HDR floating-point frames (f16 / f32) for direct interaction with Vulkan compute shaders and compositing systems. |

## Containers and network transports

The muxing and network delivery layer is completely cleaned of 1990s containers with fixed offset tables (AVI, MPEG-TS, FLV) and TCP-based streaming protocols (RTMP, classic HLS/DASH over HTTP/1.1) that suffer from head-of-line blocking.

| Layer | Standard | Purpose in rustmpeg-next | Architectural constraints |
| :---- | :---- | :---- | :---- |
| **Media container** | **ISOBMFF / CMAF** (.mp4, .m4s) | Universal storage and fragmented streaming | Strict zero-copy parser and muxer supporting only descriptors of current formats: AV2, vvc1/vvi1 (H.266/VVC), apv1 (APV), iamf (Eclipsa Audio), Opus, and fLaC. |
| **Media container** | **Matroska v4 / WebM** (.mkv, .webm) | Streaming recording, open storage, multi-track | Implemented on top of a strict EBML primitive parser. The WebM profile forcibly allows only AV2 video streams and Opus/IAMF audio streams; the Matroska v4 profile extends support to VVC, APV, and FLAC. |
| **Network transport** | **MoQ (Media over QUIC)** | Real-time streaming, interactive broadcasting, CDN | Native implementation of the IETF moq-transport standard over the QUIC protocol. Completely replaces RTMP, RTSP, SRT, and the complex WebRTC stack, providing independent delivery and prioritization of audio and video chunks with minimal latency. |

Implementation status: EBML primitives, the Matroska v4/WebM demuxer and muxer,
and the ISOBMFF/CMAF reader and writers are implemented. MoQ is planned and not
started. The tables above describe the target scope, not every implemented
detail; limitations are listed in [`ISOBMFF.md`](ISOBMFF.md).

## Architecture comparison: FFmpeg vs rustmpeg-next

| Parameter | FFmpeg (9.x branch) | rustmpeg-next |
| :---- | :---- | :---- |
| **Language and memory model** | C11 + hand-written ASM (NASM); regular memory vulnerabilities in demuxers and decoders. | 100% Rust (`#![forbid(unsafe_code)]` in parsers and containers, portable_simd in DSP cores). |
| **Format support policy** | Infinite legacy accumulation over 25 years of the project's existence. | Rolling window: 3 video codecs, 4 audio standards, 2 image formats, 2 containers, 1 network protocol. |
| **Hardware acceleration** | A fragmented set of a dozen platform backends (VA-API, NVENC, QSV, Vulkan, etc.). | A single cross-platform backend, **Vulkan Compute** (`wgpu` / `naga`) with a capability-aware CPU fallback. Vulkan Video is not used because it has no operation for any codec in the matrix. |
| **Professional video (Intra)** | Legacy ProRes, DNxHD/DNxHR + the new APV. | Only **APV (RFC 9924)** is planned, with acceleration through Vulkan Compute shaders. |
| **Spatial audio** | Partial parsing of closed formats + IAMF integration. | Native **IAMF (Eclipsa Audio)** + the **OAR** renderer exclusively over Opus, FLAC, and LPCM. |
| **Build and distribution** | A complex tree of system C dependencies, CMake/autotools, dynamic-linking conflicts. | A monolithic self-contained binary or modular consumption through cargo add; native compilation to wasm32-unknown-unknown. |

## Architectural decomposition and layer implementation

To bring rustmpeg-next to the level of a fully featured industrial competitor to FFmpeg, the codebase is designed as a set of rigidly isolated modular crates, where each higher layer depends strictly on the layer below it, with no cyclic dependencies. The bottom layer (rustmpeg-core) is implemented as a `no_std` + `alloc`, runtime-agnostic crate: zero-copy `Arc<[u8]>` buffers, exact rational time, pixel/colour/frame/audio descriptions, packets, typed errors, and portable SIMD primitives. The input/output layer (rustmpeg-format) currently implements EBML, the in-memory Matroska v4/WebM demuxer and muxer, and the ISOBMFF/CMAF reader and writers; asynchronous streaming I/O and MoQ are planned. Above it sit the audio and graphics stacks (rustmpeg-audio and rustmpeg-image): FLAC, LPCM, LC3, the Opus packet/range layer, the SILK decoder, JPEG XL (`JxlDecoder` + clean-room `JxlEncoder`), OpenEXR (`ExrDecoder` + `ExrEncoder`, OpenEXRCore-derived), and the `moxcms` ICCv4 pipeline are implemented; IAMF/OAR and CELT/hybrid are planned.
The video layer (rustmpeg-video and rustmpeg-vulkan) implements the AV2 encoder/decoder foundation, the VVC intra profile and subset decoder, codec-level reference picture management, and the Vulkan Compute backend through `wgpu` and `naga`. APV (RFC 9924) is implemented on the same CPU stack as an intra-only encoder and decoder; a GPU stage is deferred until a measured whole-frame win exists. When no usable adapter is present — including serverless and WebAssembly environments — the identical pipeline runs on the CPU with `std::simd` vector primitives. The architecture is completed by an automated lifecycle-control pipeline at the CI/CD and Cargo package-manager level: as soon as a next-generation crate passes all conformance vectors (analogous to the Fluster test framework) and reaches stable-release status, build scripts automatically block compilation of the outdated module and initiate its physical removal from the repository.

| Module (Crate) | Key standards and dependencies | Memory and execution policy |
| :---- | :---- | :---- |
| rustmpeg-core | Zero-copy `Arc<[u8]>` buffers, rational time, pixel/colour/frame/audio formats, packets, typed errors, portable SIMD. | `#![forbid(unsafe_code)]`, `no_std` + alloc, runtime-agnostic, wasm32 support. |
| rustmpeg-format | EBML/Matroska v4/WebM and ISOBMFF/CMAF implemented; moq-transport planned. | `#![forbid(unsafe_code)]`, in-memory zero-copy parsing now; async streaming I/O planned. |
| rustmpeg-audio | Opus SILK, FLAC, LPCM, and Bluetooth LC3 implemented; Opus CELT/hybrid, IAMF, and OAR planned. | `#![forbid(unsafe_code)]`, SIMD autovectorization. |
| rustmpeg-image | JPEG XL (ISO/IEC 18181-1 & 18181-2) v2 and OpenEXR v2 decoder + encoder (OpenEXRCore-derived, BSD-3-Clause) and `moxcms` CMS implemented. | `#![forbid(unsafe_code)]`, `std::simd` Gaborish filter. |
| rustmpeg-video | AV2/VVC/APV software cores implemented (experimental/partial); accelerator stage traits with CPU references. | Safe Rust + portable_simd for matrix transformations. |
| rustmpeg-vulkan | `wgpu`, `naga`, WGSL compute kernels, capability-aware CPU/GPU scheduling. | `#![forbid(unsafe_code)]`, safe `wgpu` bindings, zero-copy device buffers; no Vulkan Video. |

#### Sources

1. oxideav-basic - crates.io: Rust Package Registry, https://crates.io/crates/oxideav-basic
2. Introducing OxiMedia: A Pure Rust Reconstruction of FFmpeg and, https://kitasanio.medium.com/introducing-oximedia-a-pure-rust-reconstruction-of-ffmpeg-and-opencv-255fa6018145
3. symphonia - crates.io: Rust Package Registry, https://crates.io/crates/symphonia/0.4.0
4. Spatial9 Joins the Alliance for Open Media, http://aomedia.org/press%20releases/Spatial9-Joins-the-Alliance-for-Open-Media/
5. CVE-2026-52834 - Exploits & Severity - Feedly, https://feedly.com/cve/CVE-2026-52834
6. GitHub - pathscale/ps-jxl-oxide: Pure Rust JPEG XL decoder, https://github.com/pathscale/ps-jxl-oxide
7. gpu-video - crates.io: Rust Package Registry, https://crates.io/crates/gpu-video/dependencies
8. Progress report on rustc_codegen_cranelift (Oct 2023) - bjorn3, https://bjorn3.github.io/2023/10/31/progress-report-oct-2023.html
9. Symphonia v0.5: ALAC, MKV, & gapless playback support : r/rust, https://www.reddit.com/r/rust/comments/sh27ep/symphonia_v05_alac_mkv_gapless_playback_support/
10. image v0.25: performance improvements, production-ready WebP : r/rust, https://www.reddit.com/r/rust/comments/1cj94va/image_v025_performance_improvements/
11. Vulkan® Video – RasterGrid | Software Consultancy, https://www.rastergrid.com/vulkan-video/
12. FFmpeg 9.1 Preps For Release With APV Vulkan Encoder, NVIDIA, https://www.phoronix.com/news/FFmpeg-9.1-Releasing
13. List of codecs - Wikipedia, https://en.wikipedia.org/wiki/List_of_codecs
14. oxideav_mkv - Rust - Docs.rs, https://docs.rs/oxideav-mkv
15. oxideav-vaapi - GitHub, https://github.com/OxideAV/oxideav-vaapi
16. Eclipsa Audio vs MPEG-H 3D Audio Compared - Spatial9, https://spatial9.ai/blog/eclipsa-audio-vs-mpeg-h-3d-audio
17. GitHub - chaitanyarahalkar/bumble-rs: A Rust port of Google's, https://github.com/chaitanyarahalkar/bumble-rs
18. jxl_oxide - Rust - Docs.rs, https://docs.rs/jxl-oxide
19. tirr-c/jxl-oxide: Pure Rust implementation of JPEG XL decoder - GitHub, https://github.com/tirr-c/jxl-oxide
20. Debian -- Software Packages in "trixie", Subsection libs, https://packages.debian.org/stable/libs/
21. VVC/H.266 in GStreamer 1.26, https://blogs.igalia.com/cadubentzen/vvc-h-266-in-gstreamer-1-26/
22. Video — list of Rust libraries/crates // Lib.rs, https://lib.rs/multimedia/video
23. Immersive Audio Model and Formats, https://aomediacodec.github.io/iamf/
24. moq-transport - crates.io: Rust Package Registry, https://crates.io/crates/moq-transport
25. FFmpeg remade with rust for embedded memory safe encoding and, https://github.com/Remade-With-Rust/remade_ffmpeg_rs
26. Video Coding :: Vulkan Documentation Project, https://docs.vulkan.org/spec/latest/chapters/videocoding.html
