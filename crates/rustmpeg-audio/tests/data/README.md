# FLAC test fixtures

These files are small native FLAC streams generated locally from deterministic
`ffmpeg` lavfi sources.  No third-party audio corpus is included.  FFmpeg
(n9.0.2) and the reference `flac` command-line encoder (1.5.0) are used only as
external generation and validation tools.

Each fixture is validated against FFmpeg's reference decode:

```bash
ffmpeg -v error -i <fixture>.flac -f s32le <fixture>.reference.s32
```

`tests/flac_decode.rs` stores the FNV-1a 64 hash of that reference stream
(interleaved, little-endian, left-aligned `s32le`).  The decoder's own
`examples/flac_decode.rs` output is compared byte-for-byte against the same
reference during development.

## Generation

All commands are run from an empty directory.  `-P 0` suppresses the default
8 KiB padding block.

```bash
# 16-bit stereo, 44.1 kHz, block 1024, compression level 8 (LPC)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.55*sin(2*PI*440*t)+0.18*sin(2*PI*1320*t)+0.07*sin(2*PI*5100*t)|0.50*sin(2*PI*441*t)+0.20*sin(2*PI*880*t)+0.06*sin(2*PI*6200*t):s=44100:d=0.25" -ac 2 -c:a pcm_s16le stereo16.wav
flac -f -8 -b 1024 -m -P 0 -o stereo16_44k_1024_l8.flac stereo16.wav

# 24-bit stereo, 96 kHz, block 4096, compression level 5 (LPC, 5-bit Rice)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.7*sin(2*PI*300*t)+0.2*sin(2*PI*3100*t)+0.05*sin(2*PI*19000*t)|0.65*sin(2*PI*301*t)+0.22*sin(2*PI*2200*t)+0.05*sin(2*PI*21000*t):s=96000:d=0.15" -ac 2 -c:a pcm_s24le stereo24.wav
flac -f -5 -b 4096 -m -P 0 -o stereo24_96k_4096_l5.flac stereo24.wav

# 8-bit mono, 8 kHz, block 192, compression level 0 (fixed predictor)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.6*sin(2*PI*300*t)+0.3*sin(2*PI*1100*t):s=8000:d=0.5" -ac 1 -c:a pcm_u8 mono8.wav
flac -f -0 -b 192 -P 0 -o mono8_8k_192_l0.flac mono8.wav

# 5.1, 24-bit, 48 kHz, block 1024, compression level 8 (fixed and LPC)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.5*sin(2*PI*220*t)|0.5*sin(2*PI*330*t)|0.4*sin(2*PI*440*t)|0.25*sin(2*PI*60*t)|0.35*sin(2*PI*550*t)|0.35*sin(2*PI*660*t):s=48000:d=0.1:c=5.1" -c:a pcm_s24le surround24.wav
flac -f -8 -b 1024 -P 0 -o surround24_48k_1024_l8.flac surround24.wav

# 32-bit mono, 48 kHz, block 4096, compression level 5 (LPC, 5-bit Rice)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.7*sin(2*PI*440*t)+0.2*sin(2*PI*1500*t):s=48000:d=0.1" -ac 1 -c:a pcm_s32le mono32.wav
ffmpeg -v error -y -i mono32.wav -f s32le mono32.raw
flac -f -5 -b 4096 -P 0 --force-raw-format --endian=little --sign=signed --channels=1 --bps=32 --sample-rate=48000 -o mono32_48k_4096_l5.flac mono32.raw

# 32-bit stereo, 44.1 kHz, block 1024, compression level 8 (LPC, 5-bit Rice, right/side)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.6*sin(2*PI*330*t)+0.2*sin(2*PI*2700*t)|0.55*sin(2*PI*331*t)+0.2*sin(2*PI*1800*t):s=44100:d=0.05" -ac 2 -c:a pcm_s32le stereo32.wav
ffmpeg -v error -y -i stereo32.wav -f s32le stereo32.raw
flac -f -8 -b 1024 -m -P 0 --force-raw-format --endian=little --sign=signed --channels=2 --bps=32 --sample-rate=44100 -o stereo32_44k_1024_l8.flac stereo32.raw

# Constant subframes (silence)
ffmpeg -v error -y -f lavfi -i "anullsrc=r=44100:cl=mono" -t 0.05 -c:a pcm_s16le silence.wav
flac -f -b 1024 -P 0 -o mono16_44k_1024_silence.flac silence.wav

# Wasted bits: 8-bit content stored in a 16-bit container (8 wasted bits)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.7*sin(2*PI*440*t):s=44100:d=0.1" -ac 1 -c:a pcm_u8 wasted8.wav
ffmpeg -v error -y -i wasted8.wav -c:a pcm_s16le wasted16.wav
flac -f -8 -b 1024 -P 0 -o mono16_44k_1024_wasted.flac wasted16.wav

# Verbatim subframes: full-scale white noise at compression level 0
ffmpeg -v error -y -f lavfi -i "anoisesrc=color=white:seed=7:amplitude=1.0:sample_rate=8000:duration=0.1" -ac 1 -c:a pcm_u8 verbatim8.wav
flac -f -0 -b 192 -P 0 -o verbatim8_8k_192_l0.flac verbatim8.wav

# Forced mid/side and left/side stereo (FFmpeg encoder, block size 1024)
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.6*sin(2*PI*440*t)+0.2*sin(2*PI*1320*t)|0.55*sin(2*PI*440*t)+0.2*sin(2*PI*1320*t):s=44100:d=0.1" -ac 2 -sample_fmt s16 -c:a flac -compression_level 8 -ch_mode mid_side midside16_44k_1024_l8.flac
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.6*sin(2*PI*440*t)+0.2*sin(2*PI*1320*t)|0.55*sin(2*PI*440*t)+0.2*sin(2*PI*1320*t):s=44100:d=0.1" -ac 2 -sample_fmt s16 -c:a flac -compression_level 8 -ch_mode left_side leftside16_44k_1024_l8.flac
```

## Coverage

| Fixture | Channels | Depth | Rate | Block | Notable coding |
| :-- | --: | --: | --: | --: | :-- |
| `stereo16_44k_1024_l8.flac` | 2 | 16 | 44100 | 1024 | LPC order 9, independent |
| `stereo24_96k_4096_l5.flac` | 2 | 24 | 96000 | 4096 | LPC order 8, 5-bit Rice, 16-bit uncommon last block 2112 |
| `mono8_8k_192_l0.flac` | 1 | 8 | 8000 | 192 | fixed order 4, 21 frames |
| `surround24_48k_1024_l8.flac` | 6 | 24 | 48000 | 1024 | independent channels, fixed and LPC |
| `mono32_48k_4096_l5.flac` | 1 | 32 | 48000 | 4096 | LPC, 5-bit Rice |
| `stereo32_44k_1024_l8.flac` | 2 | 32 | 44100 | 1024 | LPC, 5-bit Rice, right/side, partition order 5 |
| `mono16_44k_1024_silence.flac` | 1 | 16 | 44100 | 1024 | constant subframes |
| `mono16_44k_1024_wasted.flac` | 1 | 16 | 44100 | 1024 | 8 wasted bits, fixed order 2 |
| `verbatim8_8k_192_l0.flac` | 1 | 8 | 8000 | 192 | verbatim subframes |
| `midside16_44k_1024_l8.flac` | 2 | 16 | 44100 | 1024 | mid/side, LPC |
| `leftside16_44k_1024_l8.flac` | 2 | 16 | 44100 | 1024 | left/side, LPC |

`tests/flac_decode.rs` additionally contains the three known-answer streams
from RFC 9639 Appendix D (constant/verbatim/fixed/LPC, side-right stereo,
partitioned Rice with an escaped partition), and `tests/flac_synthetic.rs`
covers variable block size streams, 12-bit and 20-bit samples, and malformed
frame headers/footers.

## Ogg FLAC fixture

`ogg_stereo16_44k_1024.oga` was produced from `stereo16_44k_1024_l8.flac` with
`flac --ogg -5` (flac 1.5.0). It is generated test data, used to validate the Ogg
FLAC demuxer against an independently produced file.
