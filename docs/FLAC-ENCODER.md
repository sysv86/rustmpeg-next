# FLAC encoder status

This page records the native FLAC encoder in `rustmpeg-audio` (RFC 9639), how it
is verified, and what is still missing. It is a status page, not a conformance
claim: the numbers below come from one local run on one machine and must be
reproduced before they are quoted as current.

## Scope

`encode_flac`, `encode_flac_to_vec`, and `FlacEncoder` write native FLAC;
`encode_ogg_flac` writes Ogg FLAC. The encoder supports:

- 1 to 8 channels, independent or the three stereo decorrelation modes
  (left/side, right/side, mid/side) chosen per frame by estimated coded size;
- 4, 8, 12, 16, 20, 24, and 32-bit samples;
- block sizes from 16 to 65535 samples, including a short final block;
- constant, verbatim, fixed-order 0 to 4, and LPC order 1 to 12 subframes;
- wasted-bit removal, partitioned Rice residuals with 4-bit or 5-bit
  parameters, and escaped partitions;
- STREAMINFO (block and frame sizes, total samples, MD5 of the unencoded
  audio), an optional one-second SEEKTABLE, and frame CRC-8/CRC-16.

`FlacEncoder` can stream to a non-seekable sink (unknown MD5, frame sizes, and
sample count, which RFC 9639 permits), or back-patch the complete STREAMINFO
through `finish_seekable` when the sink implements `Seek`.

## Encoder changes recorded in this cycle

The LPC analysis had three real compression defects, all fixed without changing
the bitstream syntax:

1. The autocorrelation was computed on the raw block. Periodic material makes
   the unwindowed sample autocorrelation triangle-weighted, so the normal
   equations were badly conditioned. The block is now tapered before analysis
   (half Tukey and Hann, each evaluated, with the actually encoded cost
   deciding).
2. Every LPC order reused a prefix of the final-order Levinson solution.
   Prefixes are not the least-squares predictor for the lower orders. Levinson
   now returns the coefficient set for each order.
3. Coefficient precision was chosen from the declared sample depth instead of
   the effective depth after wasted-bit removal. A 24-bit stream with 8 wasted
   bits therefore paid for 15-bit coefficients on 16-bit data. Precision now
   follows the coded depth.

Measured effect on the 2-second corpus described below (positive means
rustmpeg is larger than `flac -8`):

| Signal | Before | After |
| :-- | --: | --: |
| 440 Hz sine, 16-bit stereo | -5.5% | -27.0% |
| two-tone stereo, 16-bit | +2.3% | -19.0% |
| white noise, 16-bit stereo | -3.5% | -3.5% |
| pink noise, 16-bit stereo | -2.2% | -2.3% |
| sweep, 16-bit stereo | +15.5% | -15.1% |
| 300/500/700/900/1100/1300 Hz, 24-bit 5.1 | +36.7% | -3.3% |
| 24-bit upsampled 440 Hz sine | +16.7% | +1.2% |

The width of the change is the point: the old analysis was not wrong in the
bitstream, only badly chosen, and the loss was concentrated in high bit depth
and correlated material.

## External verification

Every file in the corpus below was checked with the reference tools:

```bash
flac -t file.flac
flac -d --force-raw-format --endian=little --sign=signed file.flac -o out.raw
cmp out.raw reference.raw
```

`flac -t` passed and the decoded bytes were identical to the reference PCM for
16-bit stereo, 24-bit stereo, 24-bit 5.1, 24-bit 7.1, 32-bit stereo, 44.1 kHz,
48 kHz, and the short-final-block cases. The same round trip is covered
in-process by `flac::encoder::tests` and `synth::tests::flac_roundtrip_for_every_family`.

The size comparison against `flac -8` used the same PCM and the same reference
encoder (`flac 1.5.0`). On the generated families (2 seconds, 48 kHz, 16-bit
stereo), `rustmpeg` was smaller in every case:

| Family | rustmpeg bytes | `flac -8` bytes | Delta |
| :-- | --: | --: | --: |
| pad | 72,019 | 80,512 | -10.5% |
| pluck | 126,202 | 136,485 | -7.5% |
| percussion | 230,242 | 237,919 | -3.2% |
| glitch | 118,883 | 139,396 | -14.7% |
| ambient | 265,581 | 273,727 | -3.0% |
| sweep | 62,349 | 84,265 | -26.0% |
| chord | 119,401 | 126,012 | -5.2% |

This does not mean the encoder beats libFLAC in general. The corpus is small,
the reference was given the same block size without its exhaustive modes, and
some signals (24-bit high-frequency content, 24-bit stereo) were still up to
about 2% larger in separate runs. Report a size comparison only with the exact
command and corpus.

## Still missing

- No writer for Vorbis comments, APPLICATION blocks, CUESHEET, PICTURE, or
  PADDING. The encoder writes STREAMINFO and an optional SEEKTABLE only; a
  caller cannot attach tags through the library yet.
- LPC analysis uses two apodization windows and every order up to the selected
  maximum. It does not implement libFLAC's exhaustive model search or its
  newer subdivide-Tukey window schedule.
- The residual planner minimizes an estimated bit count, not the final
  byte-aligned frame length, so the selected partition order can differ from
  the true optimum by a small amount.
- Fixed-block-size streams only. Variable block size and frame-header sample
  rate/bit-depth overrides are not emitted; frames refer to STREAMINFO.
- Ogg FLAC muxing is present, but the streaming `FlacEncoder` has no Ogg sink.
- No fuzz target for the encoder itself; the decoder has one.

These are recorded rather than hidden. A change that closes one of them belongs
in the same commit as its verification, including a fresh `flac -t` and raw
byte comparison.
