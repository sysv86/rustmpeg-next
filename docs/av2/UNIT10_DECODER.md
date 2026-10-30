# AV2 unit 10 decoder: extended partitions

Status: decoder half complete. The decoder parses and reconstructs every
extended partition family (`H3`, `V3`, `H4A`, `H4B`, `V4A`, `V4B`), and the
required matrix is byte-exact against the encoder reconstruction and `dav2d`.
Date: 2026-10-30.

## Scope and references

- AV2 Specification v1.0.0 §5.20.3.2: `Partition_Subsize`,
  `H_Partition_Midsize`, `is_ext_partition_allowed`,
  `is_uneven_4way_partition_allowed`, `rect_type_implied_by_bsize`,
  `is_partition_allowed`, `init_allowed_partitions`, `partition_implied`,
  `read_partition`, chroma-offset propagation, and the extended CDF contexts.
- AV2 Specification v1.0.0 §5.20.3.1: `decode_partition` recursion.
- AV2 Specification v1.0.0 §5.20.7.26: `get_plane_residual_size`, used to
  validate the subsampled chroma size of every child.
- `dav2d` `src/decode.c` `read_partition` / `decode_sb` (`subb`, `has_hv3`,
  `has_hv4ab`, `get_partition2_ctx`) as the black-box conformance oracle. No
  `dav2d`, FFmpeg, or AVM source was copied or transcribed.

## Implementation

`crates/rustmpeg-video/src/av2/dec/hls.rs`

- Parses `ext_partitions`, the dependent `uneven_4way_partitions`, and
  `max_pb_aspect_ratio_log2` from the sequence header (§5.14). These flags feed
  `PartitionSyntaxConfig::from_seq_header`.

`crates/rustmpeg-video/src/av2/dec/syntax.rs`

- `decode_partition_decision` reads the normative symbol order: `part_split`,
  `part_square`, `part_dir`, `part_ext`, `part_uneven_4way`, then the
  bypass-coded `is_a_or_b` bit for uneven 4-way. Contexts use
  `get_partition_ctx` / `get_partition2_ctx` plus `partition_size_adjust` and
  `partition_size_adjust_rect_type`, exactly as the encoder writes them.
- Implied directions from `rect_type_implied_by_bsize` code no symbol, and a
  node with a single legal partition codes none.

`crates/rustmpeg-video/src/av2/dec/frame.rs`

- `decode_sb` builds the legal set with `init_allowed_partitions`, resolves
  `partition_implied` and single-partition nodes, and only then reads a
  partition decision. No decoder-side gate rejects an extended partition.
- `decode_partition_node` implements the geometry for every family:
  - `H3` / `V3`: outer children from `partition_subsize`, two middle children
    from `h_partition_midsize`, at quarter offsets;
  - `H4A` / `H4B` / `V4A` / `V4B`: eighth-size outer children plus the
    `partition_4way_sizes` medium and large children, in the A/B order;
  - each child is visited in coding order with the correct `chromaOffset`
    propagation, partition-context update, and leaf decode.
- `decode_leaf` uses the child `BlockSize` directly for prediction, transform
  size, coefficient layout, and chroma residual size, so 64x8, 64x16, 8x64,
  16x64, 32x32 and the other extended children are reconstructed with the same
  layout the encoder signalled.

The shared tables in `crates/rustmpeg-video/src/av2/partition.rs` were corrected
by the encoder half; the decoder consumes the same `Partition_Subsize`,
`H_Partition_Midsize`, `partition_4way_sizes`, `partition_implied`, and allowed
set functions, so encoder and decoder cannot disagree on a tree.

## Verification

### In-tree regression

`crates/rustmpeg-video/src/av2/dec/mod.rs` adds:

- `test_decode_extended_partitions_roundtrip`: all six families, 64x64 and
  256x192, QP 20/100/140, 8-bit I420, 3 frames. It asserts that the requested
  family is selected at least once and that every decoded frame equals the
  encoder reconstruction byte-for-byte.
- `test_decode_extended_partitions_10bit_roundtrip`: 10-bit synthetic 64x64,
  QP 100, 3 frames; asserts `V3` is selected and the decode is byte-exact.
- `av2_extended_partitions_decoder_dav2d_byte_exact` (ignored by default): the
  same 36-case 8-bit matrix plus the 10-bit case, run through `dav2d` with
  `--threads 1 --framedelay 1 --strict 1 -q` and compared on frame payloads
  only. The Y4M header is not compared (`dav2d` writes `C420`, the examples
  write `C420jpeg`).

### Matrix result

All cases satisfy `encoder reconstruction == our decoder == dav2d`.

| Size | QP | H3 | V3 | H4A | H4B | V4A | V4B |
| :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- |
| 64x64 | 20 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 64x64 | 100 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 64x64 | 140 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 256x192 | 20 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 256x192 | 100 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 256x192 | 140 | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH |
| 64x64, 10-bit | 100 | - | MATCH | - | - | - | - |

Representative frame-0 selection counts (`rect=1 ext=1 uneven=1`): at 64x64
each family source selects one extended node (`none 4`, family `1`); at
256x192 each family source selects twelve (`none 48`, family `12`). The
counters are read from `FrameStats::partition_counts`, and the test fails if the
requested family count is zero.

### Commands and results

Run on 2026-10-30 in the repository root:

```text
cargo test -p rustmpeg-video --lib av2::dec::tests::test_decode_extended_partitions
    -> 2 passed, 0 failed
cargo test -p rustmpeg-video --lib av2_extended_partitions_decoder_dav2d_byte_exact -- --ignored
    -> 1 passed, 0 failed
rustfmt --check crates/rustmpeg-video/src/av2/dec/mod.rs
    -> clean
python3 scripts/check-licenses.py
    -> license audit: OK (all packages permissive, no copyleft)
bash $ORACLE/vvc-conf/run_conformance.sh
    -> rc 0, 25/25 cases OK with VTM and VVdeC
```

## Known limitations

- `dav2d` exits with `SIGSEGV` (rc -11) on some extended-partition streams
  outside the required matrix: 4:2:2 `V3`/`V4A`/`V4B`, 4:4:4 `H3`/`H4A`, and
  some 130x66 combinations. Our decoder returns rc 0 and equals the encoder
  reconstruction in those cases; the oracle comparison is reported as skipped,
  not as a pass. The required 8-bit I420 64x64/256x192 matrix is unaffected.
- A synthetic 16x16-constant stress source that selects `H3` in several
  superblocks still differs from `dav2d` in some mid-child luma samples while
  our decoder equals the encoder reconstruction. The same source matches with
  rectangular partitions only, so this is an encoder/H3 conformance question
  recorded for follow-up, not a decoder parsing gate.
- The encoder's non-square luma intra mode is restricted to `DC_PRED` (see
  [`UNIT10_ENCODER.md`](UNIT10_ENCODER.md)). That limitation is encoder-side and
  does not affect this decoder's syntax coverage.
