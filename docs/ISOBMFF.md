# ISOBMFF / CMAF

Status: **wave 1 implemented**. The reader handles non-fragmented `.mp4` files
and fragmented CMAF/`.m4s` files; the writers produce both layouts. The
implementation follows ISO/IEC 14496-12 (ISO base media file format), the
codec-configuration boxes of ISO/IEC 14496-15, and the CMAF constraints of
ISO/IEC 23000-19. No third-party source code is used; FFmpeg/ffprobe are used
only as external black-box oracles.

## Layout

`crates/rustmpeg-format/src/isobmff/`:

```text
isobmff/
  mod.rs          public Track/Sample/SampleEntry types, limits, colour mapping
  boxes.rs        strict box cursor, 32/64-bit sizes, full-box helpers
  sample_entry.rs Visual/Audio sample entries, colr/pasp, decoder config boxes
  reader.rs       non-fragmented sample tables + fragmented CMAF reader
  writer.rs       non-fragmented .mp4 writer
  cmaf.rs         initialization segment + media segment writer
```

## Reader

- Strict top-level and nested box parsing with checked sizes, 32-bit and
  64-bit (`largesize`) box headers, zero-size boxes, and a one-million-box cap.
- Non-fragmented files: `mvhd`, `trak` (`tkhd`, `mdhd`, `hdlr`, `minf`,
  `stbl`), `stsd`, `stts`, `ctts` (versions 0 and 1), `stsc`, `stsz`,
  `stco`/`co64`, `stss`, and `edts`/`elst`.
- Fragmented files: `mvex`/`trex` defaults, `moof`, `mfhd`, `traf`, `tfhd`,
  `tfdt` (versions 0 and 1), and `trun` with per-sample durations, sizes, flags,
  and composition offsets. Multiple tracks, trafs, and truns are supported.
- Samples are returned in file order as zero-copy `Bytes` slices. Every sample
  range is checked to lie inside a top-level `mdat`.
- Edit lists on non-fragmented tracks: leading empty edits and a single media
  edit are applied to timestamps, which covers the Opus pre-skip edit list
  produced by FFmpeg.
  Multiple media edits are rejected instead of silently returning wrong times.
- Audio/video sample entries expose width/height, `colr` (`nclx`), `pasp`,
  channels, sample size, sample rate, and the first decoder configuration box
  (`vvcC`, `apvC`, `av2C`, `dOps`, or `dfLa`).
- Encrypted sample entries (`encv`, `enca`, QuickTime `drms`/`drmi`) are
  rejected with `FormatError::UnsupportedSampleEntry`.

## Writers

### `Mp4Writer`

- Classic `ftyp` + `moov` + `mdat` layout with `stts`, `ctts`, `stss`,
  `stsc`, `stsz`, and `stco` (falling back to `co64` when an offset exceeds
  32 bits).
- Samples may be interleaved across tracks; consecutive samples of one track
  become a chunk, so the chunk tables preserve the caller's mdat order.
- `ctts` uses version 1 when any composition offset is negative, which covers
  B-frame timestamps.
- Video sample entries carry `colr`/`nclx` and `pasp`; audio entries carry the
  interleaved channel count, sample size, and 16.16 sample rate.
- Requires non-negative, non-decreasing decode timestamps per track. It does not
  write an edit list; a pre-skip is expected to be described by the codec
  configuration (for example `dOps`) or by the caller's timestamps.

### `CmafWriter`

- `init_segment()` writes `ftyp` + `moov` (`mvex`/`trex`, empty sample tables).
- `write_segment(track_id, samples)` writes `styp` + `moof` + `mdat` for one
  track, with `tfhd` (`default-base-is-moof`), `tfdt` version 1, and a single
  `trun` that carries durations, sizes, sample flags, and composition offsets.
- The first sample of a segment must be a keyframe. Samples from multiple tracks
  in one call are rejected; a CMAF track segment is the unit.
- `sequence_number()` exposes the next `mfhd` sequence number.

## Limits

- 128 tracks and 16,000,000 samples per document.
- One sample description per track.
- `stz2`, `sidx`-based seeking, encryption, `mfra`/`tfra`, and multi-track CMAF
  segments are not implemented yet.
- The reader is in-memory and eager; streaming I/O is a later wave.
- Decoder configuration records are carried verbatim. The codec crates own
  their parsing and construction; the container does not guess VVC/APV/AV2
  parameter-set layouts.

## Verification

- `crates/rustmpeg-format/tests/isobmff.rs` covers writer/reader round trips,
  interleaved tracks, fragmented segments, edit lists, and error paths.
- FFmpeg n9.0.2 fixtures in `crates/rustmpeg-format/tests/data/isobmff/`:
  non-fragmented Opus, fragmented Opus, and two-track variants. Packet sizes,
  decode timestamps, and pre-skip edit shifts match `ffprobe`.
- Remuxed Opus files written by `Mp4Writer` and `CmafWriter` decode with FFmpeg
  without errors. A separate B-frame check against an FFmpeg-produced MPEG-4
  Part 2 file confirmed `ctts` and edit-list timestamps; that file is not part
  of the repository because the codec is outside the matrix.
