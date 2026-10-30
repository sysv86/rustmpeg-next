# Matroska/WebM test fixtures

These files are generated media data, not source code. FFmpeg n9.0.2 produced
them and `ffprobe` is the black-box oracle for the expected track metadata and
packet layout. No FFmpeg source is used or linked by the demuxer; see
[`THIRD_PARTY.md`](../../../../../THIRD_PARTY.md).

## `opus.webm`

```bash
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.2" \
    -ac 2 -c:a libopus -b:a 64k -f webm opus.webm
```

- WebM (`DocType` `webm`, version 4), 200 ms, one cluster.
- One audio track: Opus, 48 kHz, stereo, `CodecPrivate` = 19-byte `OpusHead`,
  `CodecDelay` = 6,500,000 ns, `SeekPreRoll` = 80,000,000 ns,
  `DefaultDuration` = 20,000,000 ns.
- 11 packets in milliseconds: 0, 21, 41, …, 201. The demuxer subtracts
  `CodecDelay`, so presentation timestamps start at −6,500,000 ns. The last
  block carries `BlockDuration` = 7 ms and `DiscardPadding` = 13,500,000 ns.
- `FlagLacing` is 0.

## `opus_long.webm`

```bash
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=2" \
    -ac 2 -c:a libopus -b:a 64k -cluster_time_limit 100 -f webm opus_long.webm
```

- Same track settings as `opus.webm`, 2,008 ms, 101 packets.
- `-cluster_time_limit 100` forces multiple clusters and a populated `Cues`
  index, which exercises `MatroskaReader::seek_to`.

## `flac.mka`

```bash
ffmpeg -f lavfi -i "sine=frequency=330:sample_rate=48000:duration=0.2" \
    -ac 1 -c:a flac -f matroska flac.mka
```

- Matroska (`DocType` `matroska`, version 4), 200 ms, one cluster.
- One audio track: FLAC, 48 kHz, mono, `CodecPrivate` starts with `fLaC`,
  `DefaultDuration` = 21,333,333 ns (1024 samples at 48 kHz).
- 10 packets at 0, 21, 43, …, 192 ms. The last block has its own duration
  (8 ms) so the stream ends at 200 ms.

## Expected values

Expected packet sizes, timestamps, durations, and padding were read with:

```bash
ffprobe -show_streams -show_format -of json input
ffprobe -show_packets -of csv=p=0 input
```

Matroska stores block timestamps in Segment Ticks (1 ms here), so the tests use
the nanosecond value derived from the block ticks rather than `ffprobe`'s rounded
millisecond display.

## Muxer validation

`tests/matroska_muxer.rs` remuxes all three fixtures through `MatroskaWriter` and
reads the result back with the in-tree demuxer. During development the remuxed
files were also checked with FFmpeg n9.0.2: `ffprobe` reported the same streams,
packet sizes, timestamps, and durations as the sources, and decoding the remuxed
WebM/Opus and Matroska/FLAC files to `s16le` produced PCM identical to decoding
the originals.
