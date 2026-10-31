# rustmpeg-audio

Audio codecs and PCM handling for `rustmpeg-next`: Opus, FLAC, LPCM, and LC3,
plus a deterministic procedural synthesiser for test material.

Part of the [rustmpeg-next](https://github.com/sysv86/rustmpeg-next) framework.
See the repository `README.md` for status, supported formats, and build
instructions.

## FLAC

Native and Ogg FLAC encode and decode are implemented: STREAMINFO with the
unencoded-audio MD5, an optional SEEKTABLE, stereo decorrelation selected per
frame, wasted-bit removal, fixed and LPC subframes, partitioned Rice residuals,
seeking, and 4- to 32-bit samples in 1 to 8 channels.

The encoder is verified with the reference `flac` decoder (`flac -t` plus
decode-to-raw byte equality) and compared against `flac -8` on a small corpus.
Encoder status, the measured sizes, and the features that are still missing
(Vorbis comments and other metadata writers, exhaustive model search, variable
block size, Ogg streaming) are recorded in
[`docs/FLAC-ENCODER.md`](../../docs/FLAC-ENCODER.md).

## Procedural synthesis

`rustmpeg_audio::synth` renders a seed into a short stereo clip. Every parameter
is derived from the seed: root note and scale, chord, tempo, envelopes,
modulation depth, noise colour, and stereo placement. The families are `pad`,
`pluck`, `percussion`, `glitch`, `ambient`, `sweep`, and `chord`.

```rust
use rustmpeg_audio::synth::{Family, SynthConfig, render};

let config = SynthConfig {
    seed: 12345,
    family: Family::Pad,
    sample_rate: 48_000,
    bits_per_sample: 16,
    duration_seconds: 4.0,
    channels: 2,
};
let output = render(&config)?;
std::fs::write("pad.wav", output.wav_bytes())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The example writes FLAC directly through the existing encoder, with optional
WAV and interleaved PCM output:

```bash
cargo run -p rustmpeg-audio --release --example flac_generate -- out.flac \
    --seed 12345 --family pad --duration 4 --rate 48000 --bits 16 \
    --wav out.wav --pcm out.pcm

# A random seed is printed so the clip can be reproduced.
cargo run -p rustmpeg-audio --release --example flac_generate -- out.flac --seed random

cargo run -p rustmpeg-audio --release --example flac_generate -- --list-families
```

`--family auto` (the default) picks a family from the seed. Rendering is
deterministic for a fixed seed, family, sample rate, bit depth, and duration.

A local diversity experiment compares 32 seed-driven clips with FFmpeg `sine`,
`anoisesrc`, and `aevalsrc` corpora. It records the metric definitions, the
exact commands, and the honest conclusion that hand-varied FFmpeg parameters can
match the pairwise feature spread while the seed-driven generator guarantees
reproducible, distinct clips. See
[`docs/benchmarks/AUDIO_DIVERSITY.md`](../../docs/benchmarks/AUDIO_DIVERSITY.md).
