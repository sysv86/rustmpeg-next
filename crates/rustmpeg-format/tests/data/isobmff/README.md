# ISOBMFF test fixtures

These tiny files were generated locally with FFmpeg n9.0.2 and its `libopus`
encoder from a deterministic sine source.  FFmpeg is used only as an external
container oracle; no third-party media corpus is included.

```bash
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
  -ac 1 -c:a libopus -b:a 24k opus-20ms.mp4

ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
  -ac 1 -c:a libopus -b:a 24k \
  -movflags +frag_keyframe+empty_moov+default_base_moof opus-fragmented.mp4
```

A FLAC audio track uses the same source:

```bash
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
  -ac 2 -c:a flac flac.mp4
```

`opus-20ms.mp4` is a non-fragmented file with a `moov` sample table and an edit
list that skips the 312-sample Opus pre-skip.  `ffprobe` reports six audio
packets with sizes `95, 66, 59, 56, 59, 47`; the first decode timestamp is
`-312` after applying the edit list.

The two-track fixtures use two sine inputs mapped to separate Opus tracks:

```bash
ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
  -f lavfi -i "sine=frequency=880:sample_rate=48000:duration=0.1" \
  -map 0:a -map 1:a -c:a libopus -b:a 24k opus-multi-plain.mp4

ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
  -f lavfi -i "sine=frequency=880:sample_rate=48000:duration=0.1" \
  -map 0:a -map 1:a -c:a libopus -b:a 24k \
  -movflags +frag_keyframe+empty_moov+default_base_moof opus-multi.mp4
```

`opus-fragmented.mp4` is a CMAF-style file with `ftyp` + `moov` (`mvex`) and
three `moof`/`mdat` fragments.  It has the same six packets, with decode
timestamps starting at zero.

`tests/isobmff.rs` checks both files against these reference values.
