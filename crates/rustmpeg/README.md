# rustmpeg

Convenience facade for the rustmpeg-next multimedia framework.

This crate is a thin entry point, not a second implementation. It re-exports the
workspace crates as `rustmpeg::core`, `rustmpeg::format`, `rustmpeg::audio`,
`rustmpeg::image`, `rustmpeg::video`, and `rustmpeg::vulkan`. The binary-only
`rustmpeg-cli` crate is installed separately.

Part of the [rustmpeg-next](https://github.com/sysv86/rustmpeg-next) framework.
See the repository `README.md` for status, supported formats, and build
instructions.
