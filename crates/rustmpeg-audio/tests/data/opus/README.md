# Opus test fixtures

These small Ogg Opus files were generated locally with FFmpeg n9.0.2 and its
`libopus` encoder from deterministic `ffmpeg` lavfi sources.  No third-party
audio corpus is included.  FFmpeg is used only as an external generation and
validation oracle.

Ogg is **not** part of the supported container matrix of `rustmpeg-next`.
`tests/support/ogg.rs` is a test-only reader that extracts the OpusHead packet,
the OpusTags packet, and the audio packets; it is not a product API.

## Generation

All commands are run from an empty directory.  Each input is a deterministic
sine or two-tone source at 48 kHz; the durations are short to keep the fixtures
small.

```bash
# SILK only: 20 ms frames, mono, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t)+0.20*sin(2*PI*660*t):s=48000:d=0.6" -ac 1 -c:a libopus -b:a 8k -application voip -frame_duration 20 -vbr on mono_voip_20ms_8k.opus

# SILK only: 10 ms frames, mono, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t)+0.20*sin(2*PI*660*t):s=48000:d=0.4" -ac 1 -c:a libopus -b:a 8k -application voip -frame_duration 10 -vbr on mono_voip_10ms_8k.opus

# SILK only: 60 ms frames, mono, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t)+0.20*sin(2*PI*660*t):s=48000:d=0.6" -ac 1 -c:a libopus -b:a 10k -application voip -frame_duration 60 -vbr on mono_voip_60ms_10k.opus

# Hybrid: 20 ms frames, mono, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t)+0.20*sin(2*PI*660*t):s=48000:d=0.5" -ac 1 -c:a libopus -b:a 16k -application voip -frame_duration 20 -vbr on mono_voip_20ms_16k.opus

# 40 ms input split into two 20 ms frames; code 1 and code 2 packets
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t):s=48000:d=0.8" -ac 1 -c:a libopus -b:a 24k -application voip -frame_duration 40 -vbr on mono_voip_40ms_24k.opus

# 60 ms input packed as three 20 ms frames: code 3 VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*300*t)+0.10*sin(2*PI*1200*t):s=48000:d=0.6" -ac 1 -c:a libopus -b:a 16k -application voip -frame_duration 60 -vbr on mono_voip_60ms_16k_vbr.opus

# 60 ms input packed as three equal 20 ms frames: code 3 CBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*220*t)+0.10*sin(2*PI*880*t):s=48000:d=0.6" -ac 1 -c:a libopus -b:a 32k -application voip -frame_duration 60 -vbr off mono_voip_60ms_32k_cbr.opus

# CELT only: 2.5 ms frames, mono, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.45*sin(2*PI*440*t):s=48000:d=0.2" -ac 1 -c:a libopus -b:a 48k -application lowdelay -frame_duration 2.5 -vbr on mono_lowdelay_2_5ms_48k.opus

# CELT only: 5 ms frames, mono, CBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.50*sin(2*PI*1000*t)+0.10*sin(2*PI*4000*t):s=48000:d=0.25" -ac 1 -c:a libopus -b:a 64k -application lowdelay -frame_duration 5 -vbr off mono_lowdelay_5ms_64k_cbr.opus

# CELT only: 10 ms frames, stereo, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.40*sin(2*PI*330*t)+0.15*sin(2*PI*990*t)|0.40*sin(2*PI*331*t)+0.15*sin(2*PI*1320*t):s=48000:d=0.3" -ac 2 -c:a libopus -b:a 96k -application audio -frame_duration 10 -vbr on stereo_audio_10ms_96k.opus

# CELT only: 20 ms frames, stereo, VBR
ffmpeg -v error -y -f lavfi -i "aevalsrc=0.35*sin(2*PI*150*t)+0.15*sin(2*PI*3000*t)|0.35*sin(2*PI*151*t)+0.15*sin(2*PI*4500*t):s=48000:d=0.3" -ac 2 -c:a libopus -b:a 128k -application audio -frame_duration 20 -vbr on stereo_audio_20ms_128k.opus
```

`gen_manifest.py` reads each file with a minimal Ogg packet extractor, runs
`ffprobe -show_packets -show_streams`, and writes the reference manifest:

```bash
cd crates/rustmpeg-audio/tests/data/opus
python3 gen_manifest.py > manifest.txt
```

The script resolves its fixtures relative to its own location, so the `cd` is
optional.  It asserts that the extracted packet sizes equal ffprobe's `size`
values.  `tests/opus_fixtures.rs` derives each packet's frame duration from its
TOC and compares it with these ffprobe reference values; the script itself
records:

- `channels`, `pre_skip`, and `frame_samples` from the stream;
- `duration_ts` (stream duration in 48 kHz samples) from `-show_streams`;
- `coded_samples` (packets multiplied by the full TOC frame duration);
- the histogram of TOC configuration numbers and frame-count codes, observed
  by parsing each audio packet's TOC byte;
- every packet's `size` and ffprobe `duration` in samples.

The last packet's ffprobe duration is smaller than the coded frame duration
because the final granule position trims encoder padding (RFC 7845, Section
4.4).  `tests/opus_fixtures.rs` checks that relationship explicitly.

## Coverage

| Fixture | Channels | Config(s) | Code(s) | Notable |
| :-- | --: | :-- | :-- | :-- |
| `mono_voip_20ms_8k.opus` | 1 | 1 | 0 | SILK NB, 20 ms, VBR |
| `mono_voip_10ms_8k.opus` | 1 | 0 | 0 | SILK NB, 10 ms, VBR |
| `mono_voip_60ms_10k.opus` | 1 | 11 | 0 | SILK WB, 60 ms, VBR |
| `mono_voip_20ms_16k.opus` | 1 | 15 | 0 | Hybrid FB, 20 ms, VBR |
| `mono_voip_40ms_24k.opus` | 1 | 15, 31 | 1, 2 | two 20 ms frames per packet, mode switch |
| `mono_voip_60ms_16k_vbr.opus` | 1 | 15 | 3 | code 3 VBR, three 20 ms frames |
| `mono_voip_60ms_32k_cbr.opus` | 1 | 31 | 3 | code 3 CBR, three 20 ms frames |
| `mono_lowdelay_2_5ms_48k.opus` | 1 | 28 | 0 | CELT FB, 2.5 ms |
| `mono_lowdelay_5ms_64k_cbr.opus` | 1 | 29 | 0 | CELT FB, 5 ms, CBR |
| `stereo_audio_10ms_96k.opus` | 2 | 30 | 0 | CELT FB, 10 ms, stereo |
| `stereo_audio_20ms_128k.opus` | 2 | 31 | 0 | CELT FB, 20 ms, stereo |

Padding is not produced by `libopus`; Opus padding is covered by synthetic
packets in `tests/opus_packets.rs`, as are all 32 TOC configurations.
