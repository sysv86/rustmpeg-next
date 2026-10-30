# LC3 conformance fixtures

Each `*.lc3` file is an LC3 binary stream (liblc3 `lc3bin` container) produced by
the liblc3 `elc3` encoder from a deterministic synthetic signal (two tones, white
noise, and a short high-frequency burst). The matching `*.s16le` file is the
raw little-endian 16-bit PCM written by the liblc3 `dlc3` decoder when run on the
same stream.

The streams cover the decoder matrix:

- 8, 16, 24, 32, and 48 kHz;
- 7.5 ms and 10 ms frames;
- mono and stereo, including a stereo block with an odd byte count;
- low bit rates (noise filling, TNS) and a high bit rate that enables the LSB
  spectrum mode.

`manifest.txt` lists the configuration of every stream.

The same fixtures drive the encoder round-trip test: the `.s16le` PCM is
re-encoded at the fixture frame size, decoded with the in-tree decoder, and
compared with the input at the codec delay (2.5 ms for 10 ms frames, 4 ms for
7.5 ms frames).

`lc3_48k_10ms_stereo_96k_plc.s16le` is the liblc3 concealment reference for the
`lc3_48k_10ms_stereo_96k` stream with frames 4, 5, 6, 10, 11, 12, 13, and 17
treated as lost. It was produced by the liblc3 decoder API, calling the decode
entry point with no payload for a lost frame, and is compared sample by sample
with `Lc3Decoder::conceal_frame` in the PLC integration test.

## Provenance

liblc3 is the reference implementation used as a black-box oracle; its source is
Apache-2.0, Copyright 2022 Google LLC. No liblc3 source code is copied into this
repository; only the normative LC3 tables were transcribed with attribution. The
fixtures are generated media data and do not inherit the license of the tools
that produced them. See `THIRD_PARTY.md` for details.
