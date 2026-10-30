# Opus SILK/CELT decoder plan

Status: **waves 0–2 complete** (packet layer, SILK parameters, SILK synthesis).
The decoder is implemented from RFC 6716 and RFC 8251. No libopus source is
transcribed. Normative constant tables are taken from RFC 6716 Appendix A and
recorded as such; algorithms are implemented from the prose and the conformance
requirements, not translated from the reference C files.

## Scope

The decoder must eventually support:

- SILK-only: NB, MB, WB; 10/20/40/60 ms; mono and stereo.
- CELT-only: NB, WB, SWB, FB; 2.5/5/10/20 ms; mono and stereo.
- Hybrid: SILK WB/SWB + CELT; 10/20 ms.
- Output at 8, 12, 16, 24, and 48 kHz, mono or stereo.
- Packet loss concealment, at least the normative fade/pitch extrapolation.
- In-band FEC only after the baseline decoder is conformant.
- Multistream only when the IAMF/Ogg layer needs it; not in the first waves.

Out of scope for the first decoder wave:

- Encoder.
- Multistream/projection.
- DTX comfort noise generation (CNG) beyond what normal SILK requires.
- Custom modes and non-standard frame sizes.

## Layering

`crates/rustmpeg-audio/src/opus/`:

```text
opus/
  mod.rs          packet framing, TOC, public OpusError (exists)
  range.rs        range decoder (exists)
  decoder.rs      public OpusDecoder: packet validation, mode routing, output
  silk/
    mod.rs        public SilkDecoder, packet header, frame loop, LBRR flags
    tables.rs     normative RFC 6716 constant tables
    define.rs     SILK constants
    structs.rs    NLSF codebook descriptor
    pulses.rs     shell/pulse/LSB/sign excitation decoding
    nlsf.rs       NLSF reconstruction, stabilization, interpolation, LPC
    synthesis.rs  gains, pitch, LCG excitation, LTP/LPC synthesis, stereo
    stereo.rs     stereo weights, mid-only flag, MS -> LR reconstruction
    resample.rs   rational polyphase resampler
  celt/
    mod.rs        CELT state and frame loop
    tables.rs     eBands, allocation, PVQ, windows
    energy.rs     coarse/fine energy
    allocation.rs bit allocation
    shape.rs      PVQ, spreading, split, TF change
    mdct.rs       inverse MDCT and overlap-add
    stereo.rs     mid/side and intensity stereo
  hybrid.rs       SILK + CELT combination and transition handling
```

`OpusPacket` remains the single framing authority. `RangeDecoder` remains the
single entropy decoder for SILK and CELT.

## Wave 1 — SILK bitstream parameters (complete)

Goal: decode every SILK-only packet in the fixtures and the RFC test vectors into
a complete parameter set, with the final range state matching the reference.

Deliverables (all implemented):

- SILK header bits: VAD flags, LBRR flags per frame, per-channel ordering.
- LBRR payload skipping (no FEC synthesis yet).
- Frame type, gain indices, NLSF stage 1/2 indices, pitch lag/contour, LTP
  indices, LTP scaling, LCG seed, excitation rate level, pulse counts, shell
  codes, LSBs, and signs.
- Parameter range validation and typed errors.
- Depth-first shell-block traversal, which is required for decoder/encoder bit
  agreement.

Acceptance: all 3448 SILK packets in RFC 6716 test vectors 02, 03, and 04
decode to the stored final range state exactly; every fixture packet parses.

## Wave 2 — SILK synthesis (complete)

Goal: produce PCM at the SILK internal rate and validate it against a reference
oracle.

Deliverables (all implemented):

- NLSF reconstruction, stabilization, interpolation, and fixed-point LPC
  conversion with range and prediction-gain limiting.
- LCG and excitation reconstruction with quantization offsets.
- LTP synthesis with pitch lag, contour, filter taps, scaling, and residual
  rewhitening.
- LPC synthesis with per-channel history and clamping.
- Per-frame state: LPC and LTP history, previous NLSF, previous gain, previous
  lag, and conditional-coding state.
- Stereo prediction weights, mid-only handling, and mid/side to left/right
  reconstruction.
- Rational polyphase resampler from 8/12/16 kHz to 8/12/16/24/48 kHz; exact
  output sample counts for every frame size.
- Signed 16-bit interleaved output at a caller-selected rate and channel count.

Acceptance: test vectors 02, 03, and 04 decode with the correct sample and
channel counts and the stored final range state for every packet.  PCM spectra
are within 0.26 dB (median) and 1.2–5.2 dB (90th percentile) of the reference
`.dec` files for NB, MB, and WB, including stereo.  The waveform is not
bit-identical because RFC 6716, Section 4.2.7.9 explicitly permits a
non-bit-exact synthesis stage; the residual difference comes from fixed-point
rounding in the LTP/LPC feedback loop.

## Wave 3 — CELT (next)

Goal: decode CELT-only packets.

Deliverables:

- Transient decision and short/long MDCT windows.
- Coarse and fine energy, band allocation, PVQ, spreading, split, time-frequency
  change, anti-collapse, denormalization.
- Inverse MDCT and overlap-add.
- Post-filter, de-emphasis, and stereo (mid/side, intensity).
- 2.5/5/10/20 ms and all four bandwidths.

Acceptance: CELT fixtures and test vectors decode within the conformance metric;
range state matches.

## Wave 4 — Hybrid, PLC, FEC

- Combine SILK and CELT frames with the correct transition window.
- Packet loss concealment and fade.
- In-band FEC after the baseline path is stable.
- Multistream only if the IAMF/Ogg layer requires it.

## Testing strategy

- Existing Ogg fixtures are extracted with the test-only Ogg reader; the decoder
  must produce the expected sample count and non-silent output.
- RFC 6716/8251 `.bit` vectors are decoded with a test-only reader and compared
  to `.dec` PCM with an in-repo quality/SNR check.
- Every new table gets a unit test against a small symbolic packet.
- Malformed payloads must return typed errors; no panics, no unbounded
  allocation.
- `cargo test --workspace`, clippy, fmt, and the license audit remain mandatory.

## Licensing and provenance

- Algorithms: RFC 6716 and RFC 8251; no FFmpeg, libopus, or GStreamer source is
  copied or translated.
- Normative tables: RFC 6716 Appendix A (BSD-3-Clause). The table module records
  the RFC section and the BSD-3 notice; this is recorded in `THIRD_PARTY.md` and
  `NOTICE`.
- Test vectors: official Opus test vectors are used only as data; their source
  and license are recorded with the fixtures.
- No C/C++ toolchain, no FFI, no `unsafe`.
