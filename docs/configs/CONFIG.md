# Configuration and commands

This page lists every command-line entry point in the workspace, the environment
variables the code reads, the configuration structs used by library callers, and
the offline build procedure.

There is **no configuration file**. No crate reads a TOML, JSON, INI, or
environment file: behaviour is selected by CLI arguments, by the small set of
environment variables listed below, or by the Rust configuration structs. If a
flag is not listed here, it does not exist — do not guess one.

Everything below is derived from `crates/*/src`, `crates/*/examples`, and
`crates/rustmpeg-cli/src/main.rs` in this tree.

## Toolchain and build

The toolchain is pinned in [`rust-toolchain.toml`](../../rust-toolchain.toml):
`nightly-2026-10-25`, minimal profile, `rustfmt` + `clippy`, target
`wasm32-unknown-unknown`. Nightly is required because the compute cores use
`std::simd`.

```bash
rustup toolchain install nightly-2026-10-25
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo fmt --all -- --check
python3 scripts/check-licenses.py
```

All five gates — `check`, `clippy`, `test`, `fmt`, and the license audit — must
pass before a change is complete. There are no Cargo features: every crate builds
with its full code path. `rustmpeg-core` is `no_std` + `alloc`; the other crates
are `std`.

### Offline build

There is no `build.rs` in the workspace, no C/C++ compilation, and no
`pkg-config` or system library lookup. All dependencies are pure Rust, so an
offline build only needs a populated Cargo cache and the pinned toolchain:

```bash
# Once, while online:
rustup toolchain install nightly-2026-10-25
cargo fetch --locked

# Afterwards, with no network:
cargo check --workspace --locked --offline
cargo test --workspace --locked --offline
cargo run --locked --offline -p rustmpeg-cli -- version
python3 scripts/check-licenses.py
```

- `CARGO_NET_OFFLINE=true` is equivalent to `--offline` for every Cargo call.
- `--locked` is meaningful because `Cargo.lock` is committed.
- A fully self-contained checkout can vendor dependencies: `cargo vendor vendor/`
  writes the sources, and a `.cargo/config.toml` with
  `[source.crates-io] replace-with = "vendored-sources"` plus
  `[source.vendored-sources] directory = "vendor"` makes the build independent of
  the cargo registry. The workspace does not ship that file.
- On a machine without `rustc`/`cargo`, only the static checks are available:
  `python3 scripts/check-licenses.py` and source-level audits. Do not claim a
  compile, lint, or conformance run that did not happen.

The optional GPU path needs a Vulkan 1.1 adapter and a working driver at runtime;
it needs no SDK, shader compiler, or Vulkan Video extension at build time.
Without an adapter the CPU path runs unchanged and `Compute::new()` returns a
typed error.

## Core CLI: `rustmpeg-core`

Package `rustmpeg-cli`, binary `rustmpeg-core`. It is a thin, dependency-free
bridge over `rustmpeg-core` primitives: one core API call per command, printed as
`key value` lines. It does not encode or decode media.

```bash
cargo run -p rustmpeg-cli -- <command> [args...]
# or, after installing:
cargo install --path crates/rustmpeg-cli
rustmpeg-core help
```

Exit status is `0` on success and `2` on any argument or validation error, with
the message on stderr.

### Commands

```text
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
```

`help`, `--help`, and `-h` print usage. `version`, `--version`, and `-V` print
the crate version.

### Data arguments

- hexadecimal, with optional `0x` prefix and `:` or `_` separators;
- `@path` reads a file;
- `-` reads standard input.

### Formats, samples, presets

| Kind | Accepted values |
| :-- | :-- |
| Pixel formats | `i400`, `i420`, `i422`, `i444`, `rgb8`, `rgba8`, `bgra8` (aliases: `gray`/`grey`, `yuv420p`/`yuv420`, `yuv422p`, `yuv444p`, `rgb`, `rgba`, `bgra`) |
| Sample formats | `u8`, `s16`, `s24`, `s32`, `f32`, `f64` (aliases `i16`, `i24`, `i32`) |
| Colour presets | `bt709-limited` (also `bt709`, `default`), `bt2020-pq` (also `pq`), `bt2020-hlg` (also `hlg`) |
| Colour range | `limited`/`tv`, `full`/`pc` |
| Primaries | `bt709`, `bt2020`, `display-p3`/`p3`, `unknown` |
| Transfer | `sdr`/`bt709`, `pq`, `hlg`, `linear`, `srgb`, `unknown` |
| Matrix | `bt709`/`bt601`, `bt2020`/`bt2020-ncl`, `identity`/`rgb`, `unknown` |
| Chroma location | `left`, `center`/`centre`, `top-left`/`topleft` |
| Packet flags | `keyframe`/`key`, `discard`, `eos`/`end-of-stream`, `corrupt`/`corrupted` |

### Examples

```bash
rustmpeg-core time rational 30000 1001
rustmpeg-core pixel info i420 1920 1080 8
rustmpeg-core frame info i420 10 3840 2160 bt2020-pq
rustmpeg-core packet 0xdeadbeef --timebase 1 90000 --pts 90000 --duration 3000
rustmpeg-core buffer @payload.bin --slice 0:64
rustmpeg-core limits check 7680 4320
rustmpeg-core limits plane 3840 2160 2
rustmpeg-core simd sad 0x00ff 0x0f0f
```

`frame info` refuses to allocate more than 64 MiB (`FRAME_LIMIT`); `audio frame`
has the same 64 MiB cap (`AUDIO_LIMIT`).

## Video examples

All examples are run from the workspace root. Positional arguments are in the
order shown; options after a positional slot are still positional, so to set a
late option you must also supply every earlier one.

### VVC

```bash
# Synthetic encode: writes <prefix>.vvc, <prefix>.source.yuv, <prefix>.recon.yuv
cargo run -p rustmpeg-video --example vvc_dump -- <width> <height> <qp> <frames> <prefix> \
    [--bit-depth 8|10|12] [--chroma-format 400|420|422|444] [--ctu-size 32|64|128]

# Strict subset decode of an Annex-B stream to planar YUV
cargo run -p rustmpeg-video --example vvc_decode -- <input.vvc> <output.yuv>

# Stateful decoder: 8/10/12-bit, 4:0:0..4:4:4
cargo run -p rustmpeg-video --example vvc_stateful_decode -- <input.vvc> <output.yuv> \
    [--apply-film-grain 0|1]

# Multi-tool encoder (intra + P/B, coded tools, rate control)
cargo run -p rustmpeg-video --example vvc_multitool_dump -- \
    <width> <height> <qp> <frames> <keyint> <prefix> [options]
```

`vvc_dump`:

- defaults: `--bit-depth 8`, `--chroma-format 420`, `--ctu-size 32`;
- `--ctu-size 16` is rejected with `InvalidConfig` because
  `sps_log2_ctu_size_minus5` cannot signal a 16-sample CTU; use 32, 64, or 128;
- `--bit-depth` accepts 8, 10, or 12; `--chroma-format` accepts 400, 420, 422,
  444 (or 0..3).

`vvc_decode` uses `decode_annexb_to_i420` and therefore writes 8-bit planar YUV.
Use `vvc_stateful_decode` for 10/12-bit output. `--apply-film-grain` is parsed
for CLI compatibility and currently has no effect on the output (the decoder
applies grain only when the stream carries a Film Grain SEI and the decoder is
configured for it).

`vvc_multitool_dump` options:

| Option | Values | Effect |
| :-- | :-- | :-- |
| `--bit-depth` | `8`, `10`, `12` | Sample bit depth |
| `--chroma-format` | `400`, `420`, `422`, `444` | Chroma layout |
| `--ctu-size` | `32`, `64`, `128` | CTU log2 5/6/7; `16` is rejected |
| `--b-frames` | integer | B pictures per mini-GOP (`0` = I/P only) |
| `--rate-control` | `cqp`, `vbr` | `cqp` uses `qp`; `vbr` uses `--bitrate-kbps` |
| `--bitrate-kbps` | integer, default `1000` | VBR target |
| `--tiles`, `--slices`, `--wpp`, `--rpr` | one value each | **Parsed and ignored**; the stream is the plain single-slice stream |
| `--film-grain` | `0`/`1`/`true` | Emit a Film Grain SEI on keyframes |

`vvc_multitool_dump` environment variables:

| Variable | Values | Effect |
| :-- | :-- | :-- |
| `VVC_MT_INTRA_ONLY` | `1` | Force `keyint = 1` and `b_frames = 0` (all-intra) |
| `VVC_MT_TOOLS` | `all`, `none`, or a comma-separated list | Tool preset; default `all` |
| `VVC_MT_STATIC` | `1` | Use frame 0 for every frame (static scene) |

Accepted `VVC_MT_TOOLS` entries: `all`, `none`, `modes`, `cclm`, `mip`, `mrl`,
`isp`, `bdpcm`, `palette`, `ibc`, `act`, `depquant`, `sdh`, `jccr`, `deblock`,
`sao`, `alf`, `ccalf`, `lmcs`. The names `gpm`, `affine`, `prof`, `ciip`, `bdof`,
`dmvr`, `mmvd`, `amvr`, `bcw`, `tmvp`, and `rpr` are accepted but have no effect
in this writer. Unknown names print a warning on stderr and are skipped.

Note: `all` expands to `modes,cclm,mip,depquant,sdh,jccr,deblock,sao,alf` only.
It does **not** enable `mrl`, `isp`, `bdpcm`, `palette`, `ibc`, `act`, `lmcs`, or
`ccalf`; name them explicitly.

Example:

```bash
VVC_MT_TOOLS=modes,cclm,mip,deblock \
  cargo run -p rustmpeg-video --example vvc_multitool_dump -- \
  64 64 26 4 4 $WORK/mt
```

### AV2

```bash
# Encode synthetic frames (or a Y4M input) to IVF
cargo run -p rustmpeg-video --example av2_encode -- <out.ivf> \
    [frames] [width] [height] [qp] [log2_tile_cols] [log2_tile_rows] \
    [deblock 0|1] [420|422|444] [cdef 0|1] [ccso 0|1] [gdf 0|1] [rst 0|1] \
    [newmv 0|1] [switchable 0|1] [compound 0|1] [bit_depth] [vaq 0|1] \
    [delta_q 0|1] [rec.y4m] [vbr_kbps] [warped_motion 0|1] [obmc 0|1] \
    [input.y4m] [palette 0|1] [intrabc 0|1]

# Decode IVF to Y4M
cargo run -p rustmpeg-video --example av2_decode -- <in.ivf> <out.y4m> [bit_depth]
```

`av2_encode` defaults: `frames 3`, `width 64`, `height 64`, `qp 100`,
tiles `0 0`, `deblock 0`, layout `420`, all other tool flags `0`, `bit_depth 8`,
`vbr_kbps 0`, no reconstruction file, synthetic input.

Behaviour:

- `qp` is the AV2 quantizer index (`0` lossless .. `255`); the default `100` is
  the high-quality point. When `vbr_kbps > 0`, constant-QP is replaced by VBR with
  `target_kbps = vbr_kbps` and `max_qidx = 220`.
- `bit_depth` selects the generic encoder (`u8` for 8, `u16` for 10/12).
- `rec.y4m` is optional, but it is positional: pass a real path (for example
  `$WORK/rec.y4m`) when you need to set `vbr_kbps` or any later argument.
  The file receives the encoder's own reconstruction for byte-exact comparison.
- `input.y4m` is positional too, and it must be a real readable Y4M file. To reach
  `palette` or `intrabc` on the command line you must therefore also supply an
  input file; otherwise set those tools through `ToolConfig` in the API.
- `input.y4m` replaces the synthetic generator; the file must match `width`,
  `height`, `bit_depth`, and layout.
- `warped_motion 1` enables `WARPMV`/`DELTAWARP` blocks.
- `obmc 1` is accepted but has no effect: AV2 v1.0.0 has no OBMC tool.
- `palette 1` and `intrabc 1` enable those tools on intra/key frames.
- `av2_decode` takes a third argument `bit_depth`; values above 8 select the
  16-bit decoder path. Output is Y4M.

Examples:

```bash
# 3 synthetic 64x64 frames, QP 100
cargo run -p rustmpeg-video --example av2_encode -- $WORK/out.ivf

# 10 frames of 256x192 4:2:0 8-bit, QP 140, two tile columns, deblocking on,
# VBR at 800 kbit/s, plus a reconstruction file
cargo run -p rustmpeg-video --example av2_encode -- $WORK/out.ivf \
    10 256 192 140 1 0 1 420 0 0 0 0 0 0 0 8 0 0 \
    $WORK/rec.y4m 800

cargo run -p rustmpeg-video --example av2_decode -- $WORK/out.ivf $WORK/out.y4m
```

### APV

```bash
# One synthetic frame -> raw APV access units
cargo run -p rustmpeg-video --example apv_encode -- \
    <out.apv> [width] [height] [400|422|444|4444] [bit_depth] [qp]

# Raw APV bitstream -> little-endian u16 planes, component order, no header
cargo run -p rustmpeg-video --example apv_decode -- <in.apv> <out.raw>
```

Defaults: `64x48`, `422`, `10`-bit, `qp 18`. The decoder output matches FFmpeg's
raw `yuv422p10le`/`yuv422p12le` layout for the same stream. With no arguments the
examples use `$WORK/apv-example.apv` and `$WORK/apv-example.raw`.

## Audio examples

```bash
# Native or Ogg FLAC -> interleaved left-aligned s32le
cargo run -p rustmpeg-audio --example flac_decode -- <input.flac|input.oga> <output.s32le>

# Synthetic PCM -> native FLAC (or Ogg FLAC) plus <out>.raw reference PCM
cargo run -p rustmpeg-audio --example flac_encode -- <out.flac> \
    [bits] [channels] [samples] [block_size] [level] [ogg]

# Interleaved s16le -> liblc3 lc3bin
cargo run -p rustmpeg-audio --example lc3_encode -- \
    <in.s16le> <out.lc3> <sample_rate> <7500|10000> <channels> <nbytes>

# liblc3 lc3bin -> interleaved s16le
cargo run -p rustmpeg-audio --example lc3_decode -- <in.lc3> <out.s16le>
```

`flac_encode` defaults: `bits 16`, `channels 2`, `samples 48000`,
`block_size 4096`, `level 5`; the sample rate is fixed at 48000 Hz in the
example. The seventh argument selects Ogg when it is the literal string `ogg`.
`<out>.raw` uses the same interleaved little-endian layout that `flac -d
--force-raw-format --endian=little --sign=signed` produces.

`lc3_encode` frame duration is `7500` or `10000` microseconds; `nbytes` is the
payload size per frame. The lc3bin header is the liblc3 18-byte little-endian
header.

## Image examples

```bash
# Synthetic 64x64 RGBA -> lossless Modular and lossy VarDCT JXL round-trip
cargo run -p rustmpeg-image --example jxl_encode -- [out.jxl] [out_lossy.jxl]

# Probe and decode a JXL file (or synthetic test stream) into JxlImage and VideoFrame
cargo run -p rustmpeg-image --example jxl_decode -- [in.jxl]

# Encode and decode a synthetic 64x64 RGBA HALF image across all 11 OpenEXR v2 codecs
cargo run -p rustmpeg-image --example exr_encode
```

## Vulkan benchmark

```bash
cargo run --release -p rustmpeg-vulkan --example benchmark
```

No arguments. Without a usable adapter it prints `no Vulkan compute device: ...`
and exits successfully. It calibrates the cost model first, then measures
RGBA→I420, I420→RGBA, frame metrics, AV2 deblocking, and VVC `encode_rgba` at
fixed sizes. See [`../benchmarks/BENCHMARK.md`](../benchmarks/BENCHMARK.md).

## Environment variables

Only the variables below are read. Everything else in the environment is
ignored.

| Variable | Read by | Values | Purpose |
| :-- | :-- | :-- | :-- |
| `VVC_MT_INTRA_ONLY` | `vvc_multitool_dump` example | `1` | Force all-intra |
| `VVC_MT_TOOLS` | `vvc_multitool_dump` example | tool list | Slice tool preset |
| `VVC_MT_STATIC` | `vvc_multitool_dump` example | `1` | Static scene instead of moving content |
| `DAV2D` | AV2 `#[ignore]` conformance tests | path | `dav2d` binary; otherwise looked up on `PATH`, and the test skips when absent |
| `AV2_NO_IBC` | AV2 encoder | any | Disable IntraBC selection even when the tool is enabled (diagnostic) |
| `AV2_IBC_CTX0` | AV2 encoder | any | Force IntraBC context index 0 (diagnostic) |
| `AV2_DEBUG_PAL` | AV2 encoder | any | Print palette block decisions to stderr (diagnostic) |
| `AV2_TRACE` | AV2 decoder | any | Print intra block traces to stderr (diagnostic) |
| `AV2_TRACE_HDR` | AV2 decoder | any | Print sequence/frame header traces (diagnostic) |
| `AV2_TRACE_OBU` | AV2 decoder | any | Print OBU type/size traces (diagnostic) |
| `AV2_TRACE_IBCFLAG` | AV2 decoder | any | Print IntraBC flag context/bit traces (diagnostic) |
| `AV2_DEBUG_IBC` | AV2 decoder | any | Print IntraBC block details (diagnostic) |
| `CARGO_NET_OFFLINE` | Cargo | `true`/`false` | Offline build, equivalent to `--offline` |

The `AV2_*` variables are debugging hooks used while bringing a syntax path into
conformance. They are not a supported configuration surface: they can change the
bitstream or the trace output, they are not covered by tests, and they must not be
used in a release pipeline. The `tracing` crate is used for warnings and debug
events, but the workspace installs no `tracing` subscriber, so `RUST_LOG` has no
effect today.

## Library configuration structs

There is no serialized config type. Callers construct these structs directly.
Defaults are the values in `Default::default()`.

### VVC

```rust
pub struct VvcEncoderConfig { pub width: u32, pub height: u32, pub qp: u8 }
```

- `width`/`height`: even, at least 2.
- `qp`: `SliceQpY` in `4..=63`; QP 4 is the 8-bit transform-skip lossless point.

```rust
pub struct VvcEncoderOptions {
    pub config: VvcEncoderConfig,
    pub keyint: u32,                 // 1 = all-intra; > 1 = IDR + TRAIL P/B
    pub slice_options: SliceCodingOptions,
    pub bit_depth: u8,               // 8, 10, 12
    pub chroma_format: ChromaFormatIdc, // Mono400, Chroma420, Chroma422, Chroma444
    pub ctu_log2: u8,                // 5..=7 (32/64/128); 16 is unsignallable
    pub b_frames: u8,                // 0 = I/P only
    pub rate_control: Option<VvcRateControlMode>,
    pub emit_film_grain_sei: bool,
}
```

`SliceCodingOptions` is a set of `bool` switches, all `false` by default:
`full_intra_modes`, `enable_cclm`, `enable_mip`, `dep_quant`,
`sign_data_hiding`, `enable_jccr`, `enable_deblock`, `enable_sao`, `enable_alf`,
`enable_mrl`, `enable_isp`, `enable_bdpcm`, `enable_palette`, `enable_ibc`,
`enable_act`, `enable_lmcs`, `enable_ccalf`.

`VvcRateControlMode` is either `Cqp { base_qp }` or
`Vbr { target_kbps, fps_num, fps_den, init_qp }`.

### AV2

```rust
pub struct EncoderConfig {
    pub width: usize, pub height: usize,   // 1..=16384
    pub bit_depth: u8,                     // 8, 10, 12
    pub layout: PixelLayout,               // I400, I420, I422, I444
    pub framerate_num: u32, pub framerate_den: u32,
    pub base_qidx: u16,                    // 0..=255
    pub rate_control: RateControlMode,     // ConstantQp, Vbr, Cbr
    pub keyint: usize,                     // 1 = all-intra
    pub speed: u8,                         // 0 (best) ..= 10 (fastest)
    pub use_sb128: bool,                   // 128x128 vs 64x64 superblocks
    pub log2_tile_cols: u8, pub log2_tile_rows: u8, // 0..=6
    pub tools: ToolConfig,
}
```

Defaults: `64x64`, 8-bit, `I420`, `30/1`, `base_qidx 100`, `ConstantQp { 100 }`,
`keyint 30`, `speed 4`, 64x64 superblocks, no tiles, `threads 1`.

`ToolConfig` defaults to the foundation preset: `enable_ext_partitions` and
`enable_uneven_4way_partitions` on; `enable_rect_partitions`, `enable_tx_partition`,
`enable_tx_type_rdo`, `enable_rdoq`, `enable_cfl`, `enable_mrl`,
`enable_filter_intra`, `enable_isp`, `enable_intrabc`, `enable_palette`,
`enable_compound`, `enable_obmc`, `enable_warped_motion`, `enable_vaq`,
`enable_delta_q`, `enable_seg_temporal`, `enable_deblock`, `enable_cdef`,
`enable_ccso`, `enable_gdf`, `enable_restoration`, `enable_newmv`,
`enable_switchable_interp`, `enable_film_grain`, `enable_scenecut`, and
`enable_hierarchical_gop` off. `threads` is `1`.

`enable_isp` and `enable_obmc` are accepted for API/CLI compatibility and have no
effect: neither tool exists in AV2 v1.0.0.

`RateControlMode` variants:

- `ConstantQp { base_qidx }` — fixed quantizer with the keyframe and inter-frame
  offsets;
- `Vbr { target_kbps, max_qidx }` — variable bitrate over the sequence;
- `Cbr { target_kbps, buffer_ms }` — leaky-bucket constant bitrate.

### APV

```rust
pub struct ApvConfig {
    pub level_idc: u8,          // 30 * level number; default 123 (4.1)
    pub band_idc: u8,           // 0..=3; default 3
    pub qp: u8,                 // 0..=51, shared by luma and chroma; default 24
    pub tile_width_in_mbs: u16, // >= 16; default 16
    pub tile_height_in_mbs: u16,// >= 8; default 8
}
```

### FLAC

```rust
pub struct FlacEncoderConfig {
    pub sample_rate: u32,       // 1..=1048575
    pub channels: u8,           // 1..=8
    pub bits_per_sample: u8,    // 4..=32
    pub block_size: u32,        // 16..=65535
    pub compression_level: u8,  // 0..=8, always lossless
}
```

Defaults: 44100 Hz, 2 channels, 16-bit, and level 5.

### LC3

```rust
pub struct Lc3Config {
    pub sample_rate: u32,             // 8000, 16000, 24000, 32000, 48000
    pub frame_duration: Lc3FrameDuration, // Ms7_5, Ms10
    pub channels: u8,                 // 1 or 2
    pub bit_depth: u8,                // 16 or 24
}
```

Defaults: 48 kHz, 10 ms, 2 channels, 16-bit.

### JPEG XL

```rust
pub struct JxlConfig {
    pub mode: JxlEncodingMode,       // ModularLossless { predictor } | VarDctLossy { quality }
    pub use_rct: bool,               // reversible YCoCg (rct_type = 6) in ModularLossless
    pub gaborish: bool,              // 3x3 Gaborish deblocking + pre-sharpening in VarDctLossy
    pub force_container: bool,       // wrap codestream in ISO/IEC 18181-2 container
    pub entropy_mode: JxlEntropyMode,// Prefix | Ans (12-bit rANS)
    pub lz77: bool,                  // 2D LZ77 backward-reference match finding
    pub enable_palette: bool,        // automatic lossless Palette transform (<= 256 colors)
    pub enable_squeeze: bool,        // multi-level Haar-like Squeeze decomposition
    pub modular_ma_tree: bool,       // 4-context adaptive MaConfig decision tree
    pub modular_max_error: u8,       // near-lossless max sample error (0 = strictly lossless)
    pub enable_cfl: bool,            // 64x64 least-squares Chroma-from-Luma in VarDCT
    pub enable_adaptive_dct: bool,   // adaptive Dct16 / Dct8 varblock selection in VarDCT
    pub enable_adaptive_quant: bool, // per-block hf_mul activity-masked quantization in VarDCT
    pub epf_iters: u8,               // Edge-Preserving Filter iterations (0..=3)
}
```

Constructors: `JxlConfig::lossless()`, `JxlConfig::lossless_v2()`, `JxlConfig::lossy(quality)`, and `JxlConfig::lossy_v2(quality)`.

### Matroska muxer

```rust
pub struct MuxerConfig {
    pub doc_type: DocType,        // Matroska, Webm
    pub timestamp_scale_ns: u64,  // default 1_000_000 (1 ms)
}
```

## Codec and format options

| Option | Values | Applies to |
| :-- | :-- | :-- |
| Bit depth | 8, 10, 12 | VVC (`VvcEncoderOptions`, `vvc_dump`), AV2 (`EncoderConfig`) |
| Bit depth | 10, 12 | APV |
| Bit depth | 4..=32 | FLAC |
| Chroma layout | 4:0:0, 4:2:0, 4:2:2, 4:4:4 | VVC and AV2 |
| Chroma layout | 4:0:0, 4:2:2, 4:4:4, 4:4:4:4 | APV |
| CTU size | 32, 64, 128 | VVC multi-tool writer (16 cannot be signalled) |
| Superblock | 64x64, 128x128 | AV2 (`use_sb128`) |
| Quantizer | VVC QP 4..=63; AV2 qidx 0..=255; APV QP 0..=51 | Per codec |
| Rate control | CQP, VBR (VVC); CQP, VBR, CBR (AV2) | Encoders |
| Tiles | VVC: parsed but not written; AV2: `log2_tile_cols/rows` 0..=6 | Encoders |
| Container | IVF + Y4M (AV2 examples); Annex-B (VVC); raw access unit (APV) | Examples |

## Known limitations that affect configuration

These are recorded so a flag is not mistaken for working behaviour:

- VVC multi-tool `--tiles`, `--slices`, `--wpp`, and `--rpr` are parsed and
  ignored; only the plain single-slice stream is written.
- `vvc_stateful_decode --apply-film-grain` is accepted and ignored.
- AV2 `obmc` and `isp` flags are accepted and ignored; neither tool exists in
  AV2 v1.0.0.
- AV2 frames that are not aligned to the superblock grid are not
  `dav2d`-conformant; the encoder writes its own edge-partition syntax and the
  matching decoder follows it.
- The APV GPU stage does not exist yet; APV is CPU only.
