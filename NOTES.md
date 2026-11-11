# NOTES — codec progress

Last updated: 2026-10-27.

Status legend: **done** / **partial** / **in progress** / **not started**.

## rustmpeg-core

Status: **done** — foundational crate implemented from scratch.

- `no_std` + `alloc`, runtime-agnostic, `#![forbid(unsafe_code)]`; no `tokio` or
  `bytes` dependency.
- `Buffer`: immutable `Arc<[u8]>` plus a visible range; `slice`, `split_off`,
  `advance`, and `truncate` are zero-copy.
- `Rational`, `Timebase`, `MediaTime`, `MediaDuration`, and `FrameRate`: checked
  integer rescaling and timestamp arithmetic without floating-point rounding.
- `PixelFormat`, `ColorDescription`, `VideoFrame`, `Plane`, and `PlaneRef`:
  planar/packed geometry, padded strides, odd chroma dimensions, and validated
  backing buffers.
- `AudioFormat`, `AudioFrame`, and `SampleFormat`: interleaved audio with checked
  frame sizing.
- `Packet`/`PacketFlags`: encoded payload plus PTS/DTS/duration/flags.
- `CoreError`: typed base error with limits, overflow, buffer, and index variants.
- `sad_u8`/`sse_u8`: portable `core::simd` primitives with scalar tails and
  exact scalar equivalence.
- Verified on `x86_64-unknown-linux-gnu` and `wasm32-unknown-unknown`; 36 unit
  tests plus 7 public-API integration tests.

### Core CLI

`crates/rustmpeg-cli` builds the `rustmpeg-core` binary, a thin, dependency-free
bridge over every core primitive: zero-copy buffers, rational time and
timestamps, pixel/colour/frame geometry, audio formats, packets, SIMD metrics,
and limit checks.  It never decodes media; each command calls one core API and
prints the result.  `rustmpeg-core help` lists the commands, and data arguments
accept hexadecimal, `@file`, or standard input.

## H.266/VVC

Status: **implemented (Waves 1–14: multi-tool Intra + Inter P, 10-bit/4:2:2/4:4:4/ACT, palette/IBC, RPR/SEI scaffolding); 242 VVC unit tests pass; the 8-bit 4:2:0 strict subset and the recorded multi-tool intra/inter matrix are byte-exact against VTM 24.0 and VVdeC 3.3** — see [`docs/vvc/STATUS.md`](docs/vvc/STATUS.md) and [`VERIFICATION_DEBT.md`](VERIFICATION_DEBT.md). Multi-tool inter outside the checked matrix (QP4/QP10, 200x120 QP40) and B slices remain open.

### Implemented and verified

- Intra-only encoder and strict subset decoder in `crates/rustmpeg-video/src/vvc/`.
- Main 10 signalling, 8-bit 4:2:0, BT.709 limited range, VUI.
- One `IDR_N_LP` per access unit, one slice, one tile, one 32x32 CU == TU per CTU.
- PLANAR and DC intra prediction, selected per CTU by SAD; DM chroma; PDPC.
- Transform skip at QP 4 (lossless); DCT-2 32x32/16x16, explicit intra MTS
  (DST-7/DCT-8), and LFNST for QP > 4, with regular residual coding.
- MTS and LFNST candidates are selected per luma block by reconstruction SSE;
  chroma stays DCT-2.
- Annex-B output with in-band VPS/SPS/PPS/PH, emulation prevention, typed errors,
  checked arithmetic, explicit size limits, and `#![forbid(unsafe_code)]`.
- Decoder validates SPS/PPS IDs, PPS→SPS and PH→PPS linkage, one VCL per AU,
  `IDR_N_LP` only, layer 0, temporal sub-layer 0, and rejects unsupported syntax.
- 84 module tests; `cargo clippy -D warnings` and `cargo fmt --check` pass.
- Byte-exact reconstruction verified with FFmpeg n9.0.2, vvdec 3.3, and VTM 24.0
  `DecoderApp` at QP 4 (lossless), 26, 40, and 50 on 200x120 streams.

Compression at 200x120, 3 frames: 170,473 bytes at QP 4, 24,238 at QP 26, 5,205 at
QP 40, and 1,514 at QP 50. The same clip at QP 26 was 38,737 bytes with transform
skip only.

### Not implemented

- Dependent quantisation and sign-data hiding.
- Rate-distortion model; transform selection uses reconstruction SSE only, without
  a rate term.
- CU partitioning: sizes 4..128, QT/MTT, transform tree, dual tree.
- Multi-slice, tiles, WPP, sub-pictures.
- Inter prediction: P/B, DPB/RPL, motion estimation, merge/AMVP, affine, weighted
  prediction, BCW/CIIP/GPM.
- In-loop filters: deblocking, SAO, ALF, CCALF, LMCS, LADF.
- 10/12-bit, 4:2:2, 4:4:4, non-square transforms, 64-point transforms, scaling lists.
- General H.266 decoder; the current decoder accepts only this encoder's streams.
- ISOBMFF `vvc1`/`vvcC` and Matroska `V_MPEGI/ISO/VVC` muxing.

### Boundaries

The formal scope of the intra profile is in
[`docs/VVC-INTRA-PROFILE.md`](docs/VVC-INTRA-PROFILE.md). Full VVC requires the
items above; they are separate milestones, not configuration switches.

## AV1

Out of scope. The AOMedia slot is occupied by AV2, and AV1 is not implemented.
AVIF is excluded with it because it reuses the AV1 core.

## AV2

Status: **experimental encoder + decoder** in `rustmpeg-video::av2`. Units 1–12
landed on 2026-10-30: deblocking, CDEF, CCSO/GDF, loop restoration,
segmentation/VAQ, NEWMV/switchable subpel, compound/wedge/CWP, warped motion,
transform sets/WHT/DPCM, extended partitions, palette/IntraBC, and film grain.
Release close-out status, the per-unit support table, and the exact verification
commands are in [`docs/AV2-RELEASE.md`](docs/AV2-RELEASE.md).

- Encoder: pixel/bit-depth abstraction, padded planes, OBU/IVF/Y4M containers,
  MSAC/CDF entropy coding, transforms, quantisation, intra/inter prediction, rate
  control, and frame pipeline.
- Decoder (`Av2Decoder`): OBU/IVF parsing, sequence/frame headers, tile groups,
  partition trees, intra modes (DC/directional/smooth/Paeth/MRL/DIP/CfL), single
  reference inter with DRL/NEARMV, coefficient decoding, dequantisation, and the
  inverse transform. 8/10/12-bit; I420/I422/I444/I400; 64/128 superblocks;
  multi-tile streams.
- Shared conformance fixes made while building the decoder: the smooth predictor
  now uses the normative `AboveRow[w]`/`LeftCol[h]` references, intra edge
  availability is superblock- and tile-scoped, and the inter neighbor contexts
  (intra flag, skip, reference list, single-ref mode) use the normative two
  neighbor selection.
- Partition coding follows AV2 v1.0.0 §5.20.3.2 in both directions:
  `init_allowed_partitions` / `is_partition_allowed`, `partition_implied` /
  `partition_implied_at_boundary`, `Partition_Subsize`,
  `Partition_Size_Adjust` / `.._Rect_Type`, the leaf-only neighbour-mask update
  (`AboveMiSizes` / `LeftMiSizes`), and the `do_split` / `do_square_split` /
  `rect_type` CDF contexts derived from the neighbour masks. Partitions the
  syntax forbids are never emitted and implied partitions code no symbols, so
  superblock roots, frame/tile edges and non-aligned sizes are conformant. The
  default encoder still signals `PARTITION_NONE` everywhere except
  frame-boundary partitions; rectangular and extended (`*_3`, `*_4A/4B`)
  partitions are opt-in through `enable_rect_partitions` and
  `enable_ext_partitions`. Unit 10 records 7/7 encoder and 37 decoder
  extended-partition cases byte-exact against `dav2d`.
- Other conformance fixes: `txSzCtx = (Tx_Size_Sqr + Tx_Size_Sqr_Up + 1) >> 1`
  (CDF and dequantizer size class), `Max_Tx_Size_Rect` for every block size,
  chroma transform sizing from the subsampled residual block, intra reference
  reads clamped to `MiCols`/`MiRows` (§7.13.2.1 `maxX`/`maxY`, including the
  chroma subsampling shift), the intra luma mode context and reordering keyed on
  the neighbour joint mode index (`midx`) sampled at the block's last 4x4, and
  intra blocks resetting their reference lists to `-1` in the neighbour context.
- Verified: decoder output is bit-exact against the encoder reconstruction, and
  the baseline driver is **byte-exact against `dav2d`** for QP 20, 60, 100, 140,
  180 and 220 in both 1-frame and 3-frame form: **84/84 OK** across 1920x1080,
  1280x720, 64x64, 96x64, 100x64, 129x64 and 1080x1920. Earlier runs also
  covered 1280x736, 1280x704, 640x720, 256x720, 128x720, 1920x1088, 128x1080
  and 320x176. That includes the partial bottom superblock row at every
  remainder (16, 32 and 56 samples) and unaligned widths.
  The fixes that made 1080p exact were: intra reference reads clamped to
  `MiCols`/`MiRows` (§7.13.2.1 `maxX`/`maxY`), `MiCols = 2*((w+7)>>3)` /
  `MiRows = 2*((h+7)>>3)` (the 8-sample MI grid), coefficient-context updates
  clipped to the frame MI bounds, inter neighbour gathering using absolute MI
  coordinates for the tile-boundary tests, and `Subsampled_Size` accepting
  4x32/32x4 chroma residuals.
- Unit 13 re-ran the close-out on 2026-10-30: `check_av2.py full` is **84/84
  OK**, `quick` is **9/9 OK**, and all **12 ignored external-oracle tests
  pass**. Units 1–7 and 10–12 are complete for their recorded matrices; units
  8 and 9 are complete with the tool limits listed below. The old 102-case
  external group and its 2x2-corner/12-bit-warp failures are superseded:
  `dav2d` now SIGSEGVs on those 2x2-remainder streams (recorded as skipped), and
  12-bit warped motion is exact on the recorded 3-frame source. Per-unit
  evidence is in [`docs/AV2-RELEASE.md`](docs/AV2-RELEASE.md).
- Conformance fixes that closed the last inter and high-QP gaps:
  - Inter luma `transform_type` (§5.20.8.2) now selects the set with the
    normative `get_tx_set` (§5.20.8.3). `TX_SET_WIDE_64` / `TX_SET_HIGH_64`
    (`64x16`, `64x8`, `64x4`, mirrored) code a four-symbol `inter_tx_type` from
    `TileInterTxTypeLongCdf`, `TX_SET_WIDE_32` / `TX_SET_HIGH_32` code
    `is_long_side_dct` first, and only `TX_SET_DCT_IDTX` codes the single
    DCT/IDTX flag. `TX_SET_DCTONLY` (`64x64`, `64x32`, `32x64`) codes nothing.
    The old encoder emitted the DCT/IDTX flag at every size, which desynchronised
    the `dav2d` CDF state inside the last superblock row of heights with a 16-px
    remainder, where the frame-edge partitions produce 64x16 leaves.
  - The `inter_tx_type` context uses the last non-zero scan index
    (`eob - 1` in the specification's one-based `eob`), not one less.
  - Directional intra prediction (§7.13.2.8) sets `enableIdif = plane == 0`:
    luma uses the 4-tap `Dr_Interp_Filter`, chroma uses two-tap linear
    interpolation. The luma edge extension (`LeftCol[h]`, `AboveRow[w]`, the
    `maxBase + 1/2` slots) and the `maxBaseX` / `maxBaseY` clamping are applied
    per §7.13.2.8, and `mrl_index` participates in the projection as specified.
  - DC prediction uses the normative `approx_divide` (§7.13.2.10,
    `resolve_divisor` + `Div_Lut`), which differs from a rounded integer division
    by one whenever `w + h` is not a power of two. The CfL AC average uses the
    same helper. This closed the remaining ±1 differences at high QP on
    non-square blocks.
- Limits outside the recorded matrices: in the default `ToolConfig` the encoder
  still signals `PARTITION_NONE` everywhere except frame-boundary partitions.
  Rectangular and extended (`*_3`, `*_4A/4B`) partitions are opt-in
  (`enable_rect_partitions`, `enable_ext_partitions`) and are now verified by
  the unit 10 matrix; non-DCT inter transform types are opt-in through
  `enable_tx_type_rdo`; transform sets 0/1/3 and non-uniform tiles are rejected
  by the decoder with typed errors; CCSO parameter `reuse` (without block-map
  reuse) and per-unit NS filter banks are rejected.
- Remaining decoder limits (evidence in
  [`docs/AV2-RELEASE.md`](docs/AV2-RELEASE.md)):
  - unit 8: `LOCALWARP`, `EXTENDWARP`, `INTERINTRA`, warp-inter-intra and
    six-parameter warp deltas are rejected, and `WARPMV` is decoded but not
    emitted. Warped motion is byte-exact at 8/10/12-bit on the recorded
    matrices;
  - unit 9: lossless is intra-only, the encoder emits only
    `TX_PARTITION_NONE`, the inter lossless CDF is absent, and lossless
    palette/DIP/CfL are rejected. Full transform sets 0–3, 4x4 WHT and DPCM are
    implemented and verified;
  - sub-pixel and global IntraBC are rejected; the encoder emits at most one
    IntraBC block per frame;
  - the encoder and decoder both use the full AV2 §7.12.2 `find_mv_stack`
    (single/compound, derived and reference-bank candidates, sorting,
    `WarpParamStack`, per-4x4 `Mvs`/`SubMvs`); the encoder emits `LOCALWARP`
    and `EXTENDWARP` and the recorded warp/newmv matrix is byte-exact except
    one out-of-matrix content case;
  - AVM inter frames differ from `dav2d` even with loop filters disabled; AVM
    4:2:2/4:4:4 and 10/12-bit streams are rejected by `dav2d` sequence headers.
    The multi-frame AVM key frame also carries the unrelated 1-LSB intra
    reconstruction difference.
  - 10-frame 2x2-remainder streams (`66x66`, `130x66`, `130x130`) make `dav2d`
    SIGSEGV after one frame; those oracle cells are skipped, and our decoder
    still equals the encoder reconstruction on all frames.
- OBMC is not a gap: AV2 v1.0.0 has no OBMC tool (§7.3 `MOTION_MODES = 5`,
  §5.20.7.14); `ToolConfig::enable_obmc` is a documented no-op.
- Examples:
  `cargo run -p rustmpeg-video --example av2_encode -- /tmp/out.ivf 3`
  and `cargo run -p rustmpeg-video --example av2_decode -- /tmp/out.ivf /tmp/out.y4m`.
- Reference spec: `$DOCS/av2/` (full v1.0.0 PDF,
  tables, syntax browser, and `av2-spec-main` extraction).

### AV2 benchmark (2026-10-27)

Method: 10 frames, 1920x1080 I420 8-bit, Release builds, **one thread**,
identical Y4M input (`testsrc2` graphics and `testsrc2 + noise`). PSNR is
measured against the source with FFmpeg; bitrate is `bytes * 8 * 30 / frames`.
Comparison targets: **SVT-AV1 4.2.0** (production AV1) and **AVM e0e3aa8**
(AV2 reference, built with nasm 2.16.03). This is a cross-codec, asymmetric
comparison: rustmpeg's benchmark preset uses only the baseline intra/inter
tool set; loop filters, segmentation, film grain, warped motion and
palette/IntraBC are opt-in and were not used for these numbers, while SVT-AV1
and AVM are full encoders.

`testsrc2` (graphics content):

| Encoder | Setting | fps (enc) | fps (dec) | kbps | PSNR |
| :-- | :-- | --: | --: | --: | --: |
| rustmpeg AV2 (baseline) | qp 20 | 1.67 | 30.1 | 47,273 | 37.02 |
| rustmpeg AV2 (baseline) | qp 60 | 1.75 | 37.2 | 27,871 | 39.25 |
| rustmpeg AV2 (baseline) | qp 100 | 1.74 | 47.7 | 13,171 | 38.80 |
| rustmpeg AV2 (baseline) | qp 140 | 1.83 | 61.9 | 4,025 | 36.55 |
| **rustmpeg AV2 (optimized)** | **qp 20** | **17.56** | **36.1** | **51,847** | **57.50** |
| **rustmpeg AV2 (optimized)** | **qp 60** | **19.79** | **44.1** | **32,573** | **52.86** |
| **rustmpeg AV2 (optimized)** | **qp 100** | **25.05** | **67.4** | **14,520** | **39.37** |
| **rustmpeg AV2 (optimized)** | **qp 140** | **31.07** | **128.3** | **3,951** | **36.21** |
| SVT-AV1 | p12 / crf 40 | 57.4 | — | 3,280 | 38.48 |
| SVT-AV1 | p12 / crf 50 | 61.6 | — | 1,177 | 36.96 |
| SVT-AV1 | p8 / crf 40 | 24.3 | — | 3,535 | 41.16 |
| AVM | qp 100 | 0.024 | — | — | — |

Key gains from the optimization pass:
- **Encoder throughput**: 10.5x to 17.0x speedup across operating points (17.56–31.07 fps on 1080p, exceeding the 6 fps target and 10 fps stretch goal).
- **Decoder throughput**: 1.41x to 2.07x speedup (up to 128.3 fps on 1080p qp140, exceeding the 100 fps target).
- **BD-Rate / RD efficiency**: Fixed the low-QP non-monotonicity bug (qp 20 is now 57.50 dB vs baseline 37.02 dB, a +20.48 dB improvement). At matched PSNR ~39.3 dB, bitrate dropped from 27,871 kbps (baseline qp60) to 14,520 kbps (optimized qp100), achieving a **48.0% bitrate reduction** at equal quality.
- **Architectural optimizations**:
  - Spatial noise variance classifier (`is_spatial_noise_block`) and variance-guided RDO partitioning: structured areas preserve 32x32 detail while flat/noise blocks stay in 64x64.
  - Zero-allocation inner loops: replaced heap allocations in intra prediction, CfL AC calculation, transform pipelines, and syntax parsing with stack buffers.
  - Portable SIMD acceleration (`core::simd`): vectorized 8-sample and 16-sample pixel diff/accumulate (`sad_u8`, `sse_u8`), predictor blending, and 1D IDCT butterfly stages.
  - Fast-path 2D inverse transforms (`inv_txfm_add`): single 1D IDCT + broadcast when non-zero transform coefficients only lie on the primary axis (x=0 or col=0), reducing butterfly calls from 64 down to 1.
  - Conformance and stability: byte-exact match against `dav2d --strict 1` preserved across the full test matrix; 64-bit safe ULEB128 decoding resists malformed bitstreams.

## APV (RFC 9924)

Status: **implemented (experimental, CPU)** — intra-only encoder and decoder.

- `crates/rustmpeg-video/src/apv/`: strict access-unit/PBU/frame/tile parser,
  8x8 integer transform and scaling, RFC `h(v)` VLC, differential DC and
  run-level AC coding, and a fixed-QP encoder.
- Profiles 400-10, 422-10/12, 444-10/12, and 4444-10/12; 4:0:0, 4:2:2, 4:4:4,
  4:4:4:4; 10/12-bit; arbitrary frame edges; multiple tiles; H.273 colour
  metadata and `capture_time_distance`.
- Decoder input: access units and raw APV bitstreams.  Encoder output: access
  units, raw bitstreams, and `apvC` ISOBMFF configuration records.
- Verified bit-exact against FFmpeg n9.0.2's APV decoder on OpenAPV `qp_F.apv`
  (320x240 4:2:2 12-bit, two frames; committed) and `qp_D.apv` (3840x2160
  4:2:2 10-bit, three frames; not committed due to size).  Encoder output in
  all four layouts was decoded by FFmpeg; a muxed `apv1`/`apvC` MP4 also decoded.
- Not done: non-primary/alpha/depth/preview frames are skipped, metadata PBUs
  are skipped, companding is rejected, level luma-rate limits are not enforced,
  encoder rate control and custom quantization matrices are not implemented,
  and there is no GPU stage yet.
- Reference: RFC 9924; API and limits in [`docs/APV.md`](docs/APV.md).

## Audio

Status: **partial** — FLAC full cycle, LPCM, LC3 encode/decode/PLC, and the
Opus SILK decoder done (CELT, hybrid, and PLC pending).

### FLAC

- Decoder: native and Ogg FLAC; 1–8 channels, 4–32-bit, rates up to 1,048,575 Hz;
  CONSTANT/VERBATIM/FIXED/LPC; wasted bits; partitioned Rice with escape;
  independent/left-side/right-side/mid-side; CRC-8/CRC-16; strict limits and
  typed errors; STREAMINFO MD5 verification; sample-accurate seeking using
  SEEKTABLE points (or forward scanning when absent).
- Encoder: native and Ogg FLAC; fixed-block streams; constant/verbatim/fixed/LPC
  (orders 1–12) subframes; wasted bits; per-partition Rice parameter search with
  4-bit and 5-bit methods plus escape; stereo decorrelation chosen per frame;
  STREAMINFO with exact frame sizes, total samples, and MD5; SEEKTABLE emitted by
  the buffered encoder; seekable streaming encoder back-patches STREAMINFO.
- Verified: `flac -t` (including MD5) and raw PCM equality against libFLAC 1.5.0
  for 8/12/16/20/24/32-bit, mono/stereo/6-channel, block sizes 192–16384, and
  compression levels 0–8; native and Ogg outputs both decode equally with
  `flac`/FFmpeg. On the synthetic test signal the encoder is slightly smaller
  than `flac -5`/`-8`. Ogg demuxing is validated against a libFLAC-produced
  fixture.
- Not done: Ogg FLAC chained links and a leading non-FLAC (Skeleton) stream;
  decoder resynchronisation after corruption; performance tuning.

### LPCM

- Implemented raw `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`,
  `pcm_f64le` encode/decode, plus native integer ↔ normalized float conversion.

### Opus

- Done: TOC/config table for SILK/Hybrid/CELT; packet framing codes 0–3,
  VBR/CBR, padding chains, 1–48 frames, 120 ms and 1275-byte limits; RFC 6716 §4.1
  range decoder; borrowed frame views; typed errors and a 1 MiB packet cap.
- Done: SILK decoder in `crates/rustmpeg-audio/src/opus/silk/`.
  - Header bits: VAD flags, LBRR flag, per-frame LBRR flags; LBRR payloads are
    parsed and skipped (FEC recovery is a later wave).
  - Parameter decoding: frame type, gains (independent/conditional), NLSF
    stage 1/2, stabilization, interpolation, LPC conversion with range and
    prediction-gain limiting, pitch lag/contour, LTP coefficients, LTP scaling,
    LCG seed, and excitation (rate level, shell blocks, LSBs, signs).
  - Synthesis: LCG excitation reconstruction, LTP synthesis with rewhitening,
    LPC synthesis, per-channel history, stereo prediction, MS→LR, and a
    rational polyphase resampler to 8/12/16/24/48 kHz.
  - Layouts: NB/MB/WB, 10/20/40/60 ms, mono/stereo.  Output is interleaved
    signed 16-bit PCM at a fixed caller-selected rate and channel count.
- Verified: 11 real fixtures match `ffprobe` packet sizes/counts/durations
  (2.5/5/10/20/40/60 ms, VoIP/audio/lowdelay, mono/stereo, CBR/VBR, 8–128 kbps).
- Verified offline against RFC 6716 test vectors 02, 03, and 04 (all SILK,
  10/20/40/60 ms, mono and stereo, NB/MB/WB): all 3448 packet final-range
  states match the encoder, so the bitstream layer is bit-exact.  Decoded PCM
  stays within 0.26 dB median (p90 1.2–5.2 dB) of the reference `.dec` spectra.
  The time-domain waveform is not bit-identical because the RFC explicitly
  allows a non-bit-exact synthesis stage (Section 4.2.7.9); the remaining
  difference is fixed-point rounding amplified by the LTP/LPC feedback.
- Decoder plan: [`docs/OPUS-DECODER-PLAN.md`](docs/OPUS-DECODER-PLAN.md).
  Waves 0–2 are complete.  Wave 3 is CELT, followed by hybrid, PLC, in-band
  FEC recovery, multistream, and the public Ogg API.
- Not done: CELT, hybrid, PLC, FEC recovery, multistream and self-delimiting
  framing, public Ogg API.

### LC3

- Encoder and decoder: Bluetooth LC3 v1.0.1 for 7.5 ms and 10 ms frames at
  8/16/24/32/48 kHz, mono or stereo.
- Encoder chain: bandwidth detection, attack detection, low-delay forward MDCT,
  SNS analysis and VQ, TNS analysis, LTPF analysis, global-gain selection,
  quantisation, arithmetic coding with residual or LSB spectrum modes, noise
  factor, and payload multiplexing.
- Decoder chain: side information, arithmetic spectrum, residual and LSB modes,
  noise filling, SNS, TNS, low-delay inverse MDCT, LTPF.
- Packet-loss concealment (Appendix B, low complexity): `Lc3Decoder::conceal_frame`
  sign-scrambles the retained SNS-shaped spectrum with the Appendix B.3 sequence
  (`seed = (16831 + seed*12821) & 0xffff`, initialized to 24,607 once and never
  reset), fades it by 1.0/0.9/0.85 per frame for loss counts below 4, 4–7, and 8
  or more, zeroes the spectrum above `NE`, and forces the LTPF inactive. Good
  frames restart the fade but keep the seed.
- Decoder verified: sample-exact (within one 16-bit LSB) against liblc3 across 160
  configurations (5 rates × 2 frame durations × mono/stereo × 8 bit rates), plus
  fixed fixtures covering TNS, noise filling, LSB mode, and odd-byte stereo
  splits.
- Encoder verified: 12-configuration black-box check against liblc3 `elc3`/`dlc3`
  (48/32/24/16/8 kHz, both frame durations, mono/stereo, 20–400-byte frames).
  liblc3 accepts every stream; steady-state SNR is within 0.02 dB of liblc3 and
  most frames are byte-identical. The remaining differences are the LTPF activity
  bit (the spec-literal `gain_ltpf != 0` gate) and rare ±1 global-gain choices on
  knife-edge frames. A 100-configuration validity sweep (900 frames each) was
  accepted by both the in-tree and liblc3 decoders.
- Encoder round-trip integration test: every fixture is re-encoded at its frame
  size and decoded; output SNR is 12.5–36.4 dB against the input at the codec
  delay (2.5 ms for 10 ms frames, 4 ms for 7.5 ms frames).
- PLC verified against liblc3's Appendix B concealment on six configurations
  (8/16/24/32/48 kHz, 7.5/10 ms, mono/stereo) with a loss pattern that crosses the
  fade thresholds and returns to good frames: worst deviation is one 16-bit LSB.
  A stereo 48 kHz fixture with eight lost frames is checked in
  `tests/data/lc3/`; unit tests cover the seed sequence, the fade schedule, the
  good-frame reset, and retention limited to `NE` coefficients.
- Arithmetic-coder termination fix: a pending carry with a full final byte must
  terminate with `0xff` (§3.3.13.6). The previous `0x00` produced rare invalid
  streams; a dedicated regression test covers the path.
- Not done: LC3plus, 2.5/5 ms frame durations, and multichannel (>2) streams.

### IAMF / OAR

- IAMF + OAR: not started (wave 3).

## Images

Status: **implemented (JPEG XL v2 + OpenEXR v2 + `moxcms` CMS)** in `rustmpeg-image`.

- JPEG XL (`crates/rustmpeg-image/src/jxl/`, see [`docs/JXL.md`](docs/JXL.md)):
  - ISO/IEC 18181-2 container parser and writer (`jxlc`, `jxlp`, `jxll`, `Exif`, `xml `, `brob`) with architectural policy rejection of `jbrd` legacy JPEG reconstruction.
  - ISO/IEC 18181-1 Annex B/C/D/E/F/G/H/I/J bitstream reader/writer, HybridUint, prefix histograms, 12-bit rANS decoder and encoder (`write_ans_histogram`, `encode_ans_slice`), 2D LZ77 special distances and match finder, compressed ICC stream decoding (`read_icc` / `decode_icc`) and encoding (`write_icc` / `encode_icc`), Modular engine (14 predictors, `SelfCorrecting`, 4-context adaptive `MaConfig` trees, 42 RCTs, `Palette` forward/inverse, `Squeeze` forward/inverse), VarDCT engine (27 `TransformType`s, `DequantMatrixSet`, 2D DCT/IDCT, XYB + CIE 1931 XYZ / Bradford gamut matrices for Display P3 and Rec.2020 PQ/HLG, `std::simd::f32x4` Gaborish + inverse Gaborish pre-sharpening, 3-stage EPF), and Annex J synthesis features (`Splines` centripetal Catmull-Rom + 32-point IDCT + 2D `erf`, `Noise` `XorShift128Plus` + 5x5 high-pass kernel, `Patches`, and `LfFrame`).
  - Clean-room Apache-2.0 `JxlEncoder` (v1 & v2 `ModularLossless` and `VarDctLossy`); verified pixel-by-pixel against the standalone `jxl-oxide 0.12.6` reference decoder across all reference files (including `official_splines.jxl` at 100.00% exact over 12,582,912 samples) and encoder outputs (23 tests pass).
- Color management (`crates/rustmpeg-image/src/cms.rs`): pure-Rust ICCv4 and CICP transforms via `moxcms`.
- OpenEXR (`crates/rustmpeg-image/src/exr/`, see [`docs/OPENEXR.md`](docs/OPENEXR.md)):
  - OpenEXR v2/v3 file header and multipart header parser/writer, 29 standard attribute types + `Opaque` round-trip, `TileDesc` (`OneLevel`, `MipmapLevels`, `RipmapLevels`, `RoundDown`, `RoundUp`), and normative header/window/channel/multipart validation.
  - 11 codecs (`None`, `Rle`, `Zips`, `Zip`, `Piz` 16-bit LUT + 2D Haar wavelet `wenc14`/`wdec14`/`wenc16`/`wdec16` + 58-bit canonical `Huf`, `Pxr24`, `B44`, `B44a`, `Dwaa` and `Dwab` v0/v1/v2 with closed-form perceptual LUTs + 8x8 DCT/IDCT + BT.709 CSC, and `Zstd` 24-byte `zstd-exr` container + 2/4-byte channel sorting + row delta + Blosc byte shuffle + pure-Rust RFC 8878 Zstandard frame decoder/encoder) plus self-contained RFC 1950/1951 `zlib`/`DEFLATE` with dynamic/fixed/stored block selection.
  - Single-part and multi-part flat scanline, flat tiled, deep scanline (`deepscanline`), and deep tiled (`deeptile`) decoder (`ExrDecoder`) and encoder (`ExrEncoder`), Luminance/Chroma (`Y` + `RY` + `BY`) subsampled reconstruction/encoding, `displayWindow` vs `dataWindow` framing (`to_display_video_frame`), and automatic `generate_mipmaps()` / `generate_ripmaps()` pyramid generation.
  - Oracle verification against the official OpenEXR 3.5.2 tools and the `openexr-kit` conformance corpus is recorded in [`docs/OPENEXR.md`](docs/OPENEXR.md) §3.
  - Provenance: not clean-room. The container/chunk layout, validation limits, DWA classification rules and perceptual tables, DCT constants, B44 conversions, PIZ/Huffman conventions, and Zstd-EXR container are derived in part from OpenEXR 3.5.2 `OpenEXRCore` (BSD-3-Clause). Affected files carry `Apache-2.0 AND BSD-3-Clause`; `deflate.rs` remains independent RFC 1950/1951 code. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Containers and transport

Status: **partial** — EBML, Matroska/WebM, and ISOBMFF/CMAF are done; MoQ remains.

### Matroska v4 / WebM

- Strict EBML primitive parser (`crates/rustmpeg-format/src/ebml.rs`): variable-
  length IDs and sizes, unknown sizes, unsigned/signed integers, floats, UTF-8
  strings, and explicit element-size limits.
- Demuxer (`crates/rustmpeg-format/src/matroska/`): `Info`, `Tracks`, `Cues`,
  `Cluster`, `SimpleBlock`, and `BlockGroup` with `BlockDuration`,
  `ReferenceBlock`, and `DiscardPadding`; unknown-size clusters; one segment per
  document; zero-copy `Bytes` frame slices.
- Lacing: none, Xiph, fixed, and EBML lacing (RFC 9559 §10.3), one packet per
  laced frame; lacing is rejected when the track clears `FlagLacing`.
- Timestamps: block timestamp is `(cluster + relative) * TimestampScale`, then
  `CodecDelay` is subtracted to produce the presentation timestamp (RFC 9559
  §5.1.4.1.25). The first Opus packet can therefore be negative.
- Seeking: `seek_to` selects the latest cue at or before the target, positions at
  the referenced cluster, and leaves packet filtering and `SeekPreRoll` to the
  caller. `is_seekable` reports whether a `Cues` index exists.
- Codecs classified: Opus, FLAC, LC3, LPCM (integer and float), VVC, AV2,
  APV; other `CodecID` strings are preserved but not decoded.
- Verified with `ffprobe` packet manifests for WebM/Opus (11 packets, pre-skip,
  discard padding) and Matroska/FLAC (10 packets, per-block durations), plus a
  synthetic file covering unknown-size clusters, Xiph-laced blocks, and
  `DiscardPadding` on the last laced frame. `cargo clippy -D warnings` passes.

### Matroska v4 / WebM muxer

- `MatroskaWriter` (`crates/rustmpeg-format/src/matroska/write.rs`) accepts the
  same `Track`/`Packet` types as the demuxer and returns a complete `Bytes`
  document from `finish`.
- Known-size `Segment`; `Info` with `TimestampScale`, exact `Duration`,
  `MuxingApp`, and `WritingApp`; `Tracks` with audio/video settings,
  `CodecPrivate`, `CodecDelay`, `SeekPreRoll`, `DefaultDuration`, language, and
  name; `Cues` at every cluster.
- Clusters start at the first packet and roll after one second or 1 MiB. Block
  timestamps are stored relative to the cluster; the writer rejects packets that
  are out of order, precede the track's codec delay, or reference an unknown
  track.
- `SimpleBlock` is used for ordinary frames. `BlockGroup` is used when a frame
  needs `BlockDuration`, `DiscardPadding`, or a non-keyframe `ReferenceBlock`.
- Presentation timestamps are converted back to block timestamps by adding
  `CodecDelay`; remuxed WebM/Opus and Matroska/FLAC files have the same
  `ffprobe` packet layout as the sources and decode to bit-identical PCM.
- Codec IDs outside the rolling window are rejected with
  `FormatError::UnsupportedCodec`; audio/video track types are checked against the
  codec classification. Lacing is not emitted (one frame per block).

### ISOBMFF / CMAF

- `rustmpeg-format::isobmff` implements ISO/IEC 14496-12 box parsing and writing,
  classic `.mp4` sample tables, and fragmented CMAF.
- Reader: `mvhd`/`trak`/`mdia`/`minf`/`stbl`, `stsd`, `stts`, `ctts` (v0/v1),
  `stsc`, `stsz`, `stco`/`co64`, `stss`, `elst`, `mvex`/`trex`, `moof`, `traf`,
  `tfhd`, `tfdt` (v0/v1), and `trun` with multiple tracks, trafs, and truns.
  Samples are zero-copy `Bytes` slices, validated to lie inside `mdat`.
- Edit lists: leading empty edits and one media edit are applied; this covers
  the Opus pre-skip edit list generated by FFmpeg. Multiple media edits are
  rejected.
- `Mp4Writer` writes `ftyp` + `moov` + `mdat`, interleaved chunks, `stts`,
  `ctts` (v1 for negative composition offsets), `stss`, `stsc`, `stsz`, and
  `stco`/`co64`; video entries carry `colr`/`nclx` and `pasp`.
- `CmafWriter` writes an initialization segment and one-track media segments
  (`styp` + `moof` + `mdat`) with `tfhd`, `tfdt`, and per-sample `trun` fields;
  the first sample of every segment must be a keyframe.
- Verified against FFmpeg n9.0.2 fixtures: non-fragmented Opus (pre-skip edit
  list), fragmented Opus, and two-track variants; packet sizes, decode
  timestamps, and edit shifts match `ffprobe`. Remuxed Opus files written by
  both writers decode with FFmpeg without errors. A B-frame check against an
  FFmpeg-produced MPEG-4 Part 2 file validated `ctts`; that codec is not part of
  the matrix and is not committed.
- Not done: `sidx` seeking, `stz2`, encryption/`pssh`, multiple sample
  descriptions, multi-track CMAF segments, streaming I/O, and writer-side edit
  lists.

### Not done

- MoQ over QUIC.

## GPU

Status: **implemented for Vulkan Compute**. Vulkan Video is not used: Vulkan
1.4.366 defines codec operations only for AV1, H.264/AVC, H.265/HEVC, and VP9,
none of which are in the supported matrix. The implemented GPU stages are
frame-level colour conversion and metrics; the AV2 deblocking kernel is bit-exact
but slower than the CPU under the current transfer model, so it is opt-in. Block
level codec stages stay on the CPU until a whole-frame VRAM-resident batch exists.
APV is CPU-only until a whole-frame GPU stage beats the CPU by the required margin; entropy coding and block decisions stay serial.

### Vulkan Compute (`rustmpeg-vulkan`)

- Device layer (`src/compute/mod.rs`): Vulkan-only headless `wgpu` instance,
  adapter enumeration, selection by exact name or power preference, and a device
  created with the adapter's allocation, binding, workgroup, and dispatch limits.
  `ComputeInfo` reports the limits that the created device enforces, not the
  adapter's advertised values.
- Kernel layer: WGSL is parsed and validated by `naga` before it reaches the
  driver. Entry-point presence, compute stage, workgroup size, and the
  group-0/contiguous-binding contract are checked up front, so shader errors are
  typed `VulkanError::Kernel` values instead of device loss.
- Buffers: storage and uniform allocations are aligned, bounds-checked against
  `max_buffer_size`, and validated again against the per-binding limits at
  dispatch. Read-back uses a bounded timeout and never blocks forever.
- Kernels (`src/compute/color.rs`, `src/compute/metrics.rs`):
  - RGBA8 → I420 matches `rustmpeg_video::accelerator::CpuRgbaToI420` byte for byte,
    including premultiplied-alpha un-premultiplication and the odd-dimension
    2x2 box filter. Output words are owned by exactly one invocation, so no
    atomics or write races are needed.
  - I420 → RGBA8 uses the same integer BT.709 limited-range matrix as
    `i420_to_rgba_cpu`.
  - Plane SSE/SAD uses one workgroup per row, workgroup atomics, and a host-side
    `u64` reduction, with padded strides supported. All three planes are
    recorded into one batch, so a frame costs one queue submission.
  - AV2 deblocking (`src/compute/deblock.rs`) implements the vertical and
    horizontal passes for 8-bit planes, bit-exact with `av2::deblock`, in one
    batch. It is connected through `DeblockFilter<u8>` and
    `Av2Encoder::set_deblock_filter`; the current upload/read-back path makes it
    slower than the CPU filter, so it is opt-in and the default stays on the CPU.
- Batch execution (`src/compute/mod.rs`): `Compute::batch` records dispatches and
  read-backs into one command encoder, submits once, and maps the results. This
  is the pipeline primitive that avoids a queue round trip per kernel.
- Scheduler (`src/scheduler.rs`): builds a CPU profile (cores, SIMD width, cache
  sizes) and a GPU profile (limits, workgroups, device type), calibrates both
  paths at two frame sizes, validates GPU output against the CPU reference, and
  places a stage on the GPU only when `gpu_cost * 1.1 < cpu_cost`. The GPU path
  is disabled permanently on a mismatch. Small pictures and machines without an
  adapter stay on the CPU.
- `Accelerator` (`src/accelerator.rs`): owns the device, kernels, and scheduler;
  implements `RgbaToI420Converter` and `FrameMetricsProvider`; retries on the
  CPU and disables the GPU after a run-time failure. VVC and AV2 accept it
  directly.
- Codec boundary (`rustmpeg-video::accelerator`): the codec crates do not depend
  on `rustmpeg-vulkan`. `RgbaToI420Converter` is consumed by
  `VvcEncoder::encode_rgba_with`; `FrameMetricsProvider` is consumed by
  `Av2Encoder::set_frame_metrics`. The CPU reference implementations always
  exist.
- Verified on NVIDIA RTX 3050 (driver 615.78.08): RGBA→I420 and I420→RGBA are
  byte-exact against the CPU reference at even, odd, and non-multiple-of-four
  dimensions; SSE/SAD are exact with tightly packed and padded strides. A machine
  without a Vulkan driver skips the hardware tests and uses the CPU path.

### Measured throughput

Release build on an NVIDIA RTX 3050 Laptop GPU, 7 iterations, GPU numbers include
upload and read-back:

| Stage | 720p | 1080p | 4K |
| :-- | --: | --: | --: |
| I420 → RGBA | 2.58x | 2.84x | 3.03x |
| RGBA → I420 | 1.43x | 1.27x | 1.00x |
| Metrics (SSE/SAD) | 0.98x | 1.11x | 1.03x |
| AV2 deblock (opt-in) | 0.61x | 0.73x | 0.91x |
| Full VVC `encode_rgba` | ~1.0x | — | — |

Colour conversion is memory-bound at 4K, and metrics are dominated by the
three upload/read-back pairs, so their gain is small. Deblocking is also
transfer-bound and stays on the CPU; the GPU kernel exists for a future
VRAM-resident pipeline but is not enabled by default. The VVC encoder itself
(transform, prediction, CABAC) dominates its runtime; the conversion it saves is
only 0.3–2.4 ms out of 19–250 ms, so end-to-end VVC encoding remains unchanged.
This is why the CPU/GPU split targets whole-frame preprocessing rather than the
serial codec core.

### GPU integration policy

The mandatory contract for future codecs and codec updates is in
[`AGENTS.md`](AGENTS.md#heterogeneous-cpugpu-execution). In short: CPU
reference first, bit-exact GPU output, release-mode measurement including
transfer, a 10% margin before the GPU is selected, one batched submission per
pass, and no per-block dispatch. A new GPU stage without a scheduler entry and a
recorded benchmark is incomplete.

### GPU boundary

The implemented kernels are frame-level: colour conversion and per-plane metrics.
VVC and AV2 transforms, intra/inter prediction, and loop filters are not dispatched
per block. A 32x32 VVC block is too small to amortize an upload, a dispatch, and a
read-back, and the encoder needs the reconstructed result to choose the next mode.
Those stages stay on the CPU until a whole-frame coefficient batching API exists;
the scheduler already has the cost model needed to place them when it does.

### Removed

The Vulkan Video layer (`loader.rs`, `device.rs`, `video/*`, `tests/video.rs`) and
the `ash` / `gpu-allocator` dependencies were removed in this wave. They probed
profiles that the project does not support and could not decode or encode any
codec in the matrix.

## License audit

Status: **clean** (2026-10-27).

- 137 cargo packages and 161 Rust source files audited; all permissive, no
  GPL/LGPL/AGPL/MPL/SSPL/BUSL/CDLA.
- One OpenAPV APV conformance vector (`qp_F.apv`) is committed as test data under
  BSD-3-Clause and attributed in `THIRD_PARTY.md` and `NOTICE`.
- `quinn` removed: it pulled `ring` (C/assembly) and `webpki-root-certs`
  (CDLA-Permissive-2.0). No dependency build script compiles C/C++/assembly now.
- SPDX headers added to every Rust source file. VVC files with VTM-derived
  algorithms/tables record `Apache-2.0 AND BSD-3-Clause`; AV2 files with
  dav2d-derived conformance data record `Apache-2.0 AND BSD-2-Clause`;
  `rustmpeg-video` declares the combined package expression.
- Audit script: `python3 scripts/check-licenses.py`.
- Policy: `LICENSING.md`; audit report: `docs/LICENSE-AUDIT.md`; third-party
  notices: `THIRD_PARTY.md`.

## Verification tools available locally

- Reference library (outside the repo): `$DOCS/INDEX.md`.
  Contains H.266/AV2/Opus/FLAC/LC3/IAMF specs, RFCs, Vulkan docs, and shallow
  reference repos (VTM, VVenC, AVM, rav2d, libopus, libFLAC, liblc3, etc.).
- FFmpeg n9.0.2 with native VVC decoder.
- Original `dav2d` AV2 decoder (built from source):
  `$WORK/dav2d/build/tools/dav2d`.
- vvdec 3.3 (`$WORK/vvc/vvdec/bin/release-static/vvdecapp`).
- VTM 24.0 (`$WORK/vvc/vtm_build/bin/umake/gcc-16.2/x86_64/release/DecoderApp`).
- Pre-change backups under `.git/agent-checkpoints/`.

## Provenance

VVC code is original Rust code whose algorithms and tables are derived from
VTM 24.0 (BSD-3-Clause, © 2010-2020 ITU/ISO/IEC). The AV2 encoder foundation uses
dav2d (BSD-2-Clause, © 2018-2026 VideoLAN and dav2d authors) conformance data and
syntax details. See [`THIRD_PARTY.md`](THIRD_PARTY.md). FFmpeg is used only as an
external behavioral oracle; no FFmpeg code is included.

## AV2 deblocking (normative, CPU)

- Decoder and encoder both use `av2::deblock::deblock_frame_normative` driven by a
  per-4x4 `DeblockGrid` (transform size, skip, block/transform edges, lossless).
  The legacy `DeblockFilter` accelerator trait models the old block-agnostic filter
  only; the encoder no longer routes the normative filter through it (CPU only).
- Verified against dav2d (our encoder with `enable_deblock`, our decoder and dav2d
  byte-identical): 4:2:0 and 4:2:2 and 4:4:4 8-bit, 64x64..320x192, QP 20..220,
  1..3 frames, tiles 2x1 and 2x2. Unit test
  `test_decode_matches_encoder_recon_with_deblocking` covers 8/10/12-bit.
- 1080p release, 3 frames, QP 140 (synthetic content): deblock off 0.263 s /
  61810 B; on 0.265 s / 61026 B (-1.3 % size, PSNR Y +0.14 dB on the inter frames,
  -0.65 dB on the key frame because the level is derived, not searched).
