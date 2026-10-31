# Audio diversity experiment (local)

**Status: local experiment, not a CI benchmark and not a conformance claim.**
It measures how much the clips in a corpus differ from each other under three
objective features. It does not measure musical quality, and it does not prove
that one generator is "better" than another.

Recorded 2026-10-31 on one Linux x86_64 machine with `ffmpeg n9.0.2`,
`flac 1.5.0`, Python 3.12 (standard library only), and the repository at the
commit that added `crates/rustmpeg-audio/src/synth.rs`. The scripts use 32
clips per set, 2.0 seconds each, 48 kHz, 16-bit stereo.

## Sets

| Set | Generator | Content |
| :-- | :-- | :-- |
| `ff_fixed` | FFmpeg `sine`, `anoisesrc`, `aevalsrc` | 11 identical 440 Hz sines, 11 identical white-noise clips, 10 identical three-harmonic clips. Every clip of a type uses exactly the same parameters. |
| `ff_varied` | FFmpeg `sine`, `anoisesrc`, `aevalsrc` | 32 clips whose frequency, noise colour, amplitude, or harmonic frequencies are changed by hand per clip (derived from the clip index). |
| `rs` | `examples/flac_generate.rs` | 32 seeds `1000..1031`, family cycling through pad, pluck, percussion, glitch, ambient, sweep, and chord. Every family parameter is derived from the seed. |

`ff_fixed` is deliberately repetitive: it is the baseline that shows what a
fixed FFmpeg command line produces. `ff_varied` is the fair comparison, because
it changes parameters per clip the way a user would have to write a script to do.

## Commands

All paths are placeholders: `$WORK` is a scratch directory, `$REPO` is the
repository root, and `$GEN` is the built example.

```bash
cargo build --release -p rustmpeg-audio --example flac_generate
GEN="$REPO/target/release/examples/flac_generate"
mkdir -p "$WORK"/{ff_fixed,ff_varied,rs}

# ff_fixed: three fixed parameterisations, 32 clips.
for i in $(seq 0 31); do
  if   [ "$i" -lt 11 ]; then
    ffmpeg -loglevel error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=2" -ac 2 -c:a pcm_s16le "$WORK/ff_fixed/$i.wav"
  elif [ "$i" -lt 22 ]; then
    ffmpeg -loglevel error -y -f lavfi -i "anoisesrc=color=white:amplitude=0.3:sample_rate=48000:duration=2" -ac 2 -c:a pcm_s16le "$WORK/ff_fixed/$i.wav"
  else
    ffmpeg -loglevel error -y -f lavfi -i "aevalsrc=0.2*sin(2*PI*220*t)+0.1*sin(2*PI*440*t)+0.05*sin(2*PI*880*t):s=48000:d=2" -ac 2 -c:a pcm_s16le "$WORK/ff_fixed/$i.wav"
  fi
done

# ff_varied: per-clip parameters derived from the index.
for i in $(seq 0 31); do
  case $((i % 3)) in
    0) f=$(python3 -c "print(110*2**((($i*7)%60)/12))")
       ffmpeg -loglevel error -y -f lavfi -i "sine=frequency=$f:sample_rate=48000:duration=2" -ac 2 -c:a pcm_s16le "$WORK/ff_varied/$i.wav" ;;
    1) c=$(echo white pink brown | cut -d' ' -f$((i%3+1)))
       a=$(python3 -c "print(0.12+0.02*($i%8))")
       ffmpeg -loglevel error -y -f lavfi -i "anoisesrc=color=$c:amplitude=$a:sample_rate=48000:duration=2" -ac 2 -c:a pcm_s16le "$WORK/ff_varied/$i.wav" ;;
    *) ffmpeg -loglevel error -y -f lavfi -i "aevalsrc=0.$((2+i%7))*sin(2*PI*$((110+i*13))*t)+0.$((1+i%5))*sin(2*PI*$((220+i*29))*t)+0.0$((1+i%9))*sin(2*PI*$((440+i*41))*t):s=48000:d=2" -ac 2 -c:a pcm_s16le "$WORK/ff_varied/$i.wav" ;;
  esac
done

# rustmpeg: different seed per clip, families cycle.
fams=(pad pluck percussion glitch ambient sweep chord)
for i in $(seq 0 31); do
  fam=${fams[$((i%7))]}
  "$GEN" "$WORK/rs/$i.flac" --seed $((1000+i)) --family "$fam" --duration 2
  ffmpeg -loglevel error -y -i "$WORK/rs/$i.flac" -ac 2 -c:a pcm_s16le "$WORK/rs/$i.wav"
done

python3 "$REPO/scripts/bench/audio_diversity.py" "$WORK"
```

## Features

For each clip the script takes 1024-sample mono-downmixed frames with no overlap
and a Hann window, then computes:

- **Spectral centroid**: power-weighted mean frequency per frame, averaged over
  the clip. The reported distance is the mean pairwise absolute difference in
  hertz.
- **MFCC distance**: 26 triangular mel filters over 20 Hz to 20 kHz, log energy,
  DCT-II coefficients 1 to 13, averaged over the clip; the reported distance is
  the mean pairwise Euclidean distance.
- **Band-energy distance**: mean log energy in eight bands
  (0-200-400-800-1600-3200-6400-12000-24000 Hz), mean pairwise Euclidean
  distance in dB.
- **Unique-frame ratio**: each frame becomes a vector of the 13 MFCCs (rounded
  to 0.001) and eight band energies (rounded to 0.01 dB). `unique/clip` is the
  mean fraction of distinct vectors inside a clip; `unique/global` is the
  fraction of distinct vectors over the whole 32-clip set.

All distances are computed over the 496 unordered pairs in each set. Larger is
more diverse under that feature.

## Results

| Set | Clips | Pairs | Centroid (Hz) | MFCC | Band (dB) | Unique/clip | Unique/global |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `ff_fixed` | 32 | 496 | 5448.9 | 82.92 | 96.00 | 0.873 | 0.394 |
| `ff_varied` | 32 | 496 | 1213.8 | 106.00 | 108.24 | 0.970 | 0.970 |
| `rs` | 32 | 496 | 1554.4 | 72.75 | 84.67 | 0.974 | 0.973 |

## Honest reading

- `ff_fixed` has a large centroid distance only because the corpus mixes a pure
  440 Hz tone with full-band noise; its global unique-frame ratio (0.394) shows
  the repetition directly. The pair distance is not evidence of a varied
  generator.
- `ff_varied` is the set to compare with `rs`. It **matches or exceeds** the
  pairwise MFCC and band-energy distances (106.00 vs 72.75 and 108.24 vs 84.67
  dB) because it deliberately alternates pure tones, noise, and harmonic sums,
  which are far apart in feature space. If the goal were only a large pairwise
  distance, an FFmpeg script with varied parameters can reach it.
- `rs` is slightly ahead on unique-frame ratio (0.974 vs 0.970 per clip, 0.973
  vs 0.970 globally). The difference is small and should not be treated as a
  win by itself.
- What FFmpeg cannot do without extra scripting is turn one seed into one
  reproducible clip: `sine`, `anoisesrc`, and `aevalsrc` take fixed arguments.
  `rs` derives all of its musical parameters from the seed, and
  `--seed random` plus the printed seed makes any clip reproducible.
- The features are blind to musical structure. They do not show whether the pad,
  arpeggio, or ambient material is pleasant, nor whether the FFmpeg sine and
  noise corpus is musically useful. A listening test would be needed for that.
- The corpus is 16-bit only, mono-downmixed for analysis, and one family per
  seed in `rs`. A 24-bit or per-family breakdown would be a different run.

The conclusion is narrow: with hand-varied parameters, FFmpeg's synthetic
generators reach comparable or larger objective feature spread; the rustmpeg
generator's distinct property is seed-deterministic variety across several
families, not a higher score on every distance metric.
