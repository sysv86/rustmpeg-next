// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
// Copyright 2026 The rustmpeg-next Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::fmt;

/// Errors returned by the `av2` AV2 encoder and decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Av2Error {
    /// Invalid encoder configuration parameter.
    InvalidConfig(String),
    /// Invalid input frame dimensions.
    InvalidDimension { width: usize, height: usize },
    /// Invalid input frame data, bit depth, or plane layout.
    InvalidInput(String),
    /// Invalid container (Y4M / IVF) header or stream structure.
    InvalidHeader(String),
    /// Bitstream or OBU syntax error during encoding or verification.
    BitstreamError(String),
    /// Entropy coding range or symbol error.
    EntropyError(String),
    /// Syntax or coding tool that this implementation does not decode yet.
    Unsupported(String),
    /// Encoder requires more input frames before producing a packet.
    NeedMoreData,
    /// End of stream reached; no further packets available.
    EndOfStream,
    /// I/O error message when reading or writing container streams.
    IoError(String),
}

impl fmt::Display for Av2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid encoder configuration: {msg}"),
            Self::InvalidDimension { width, height } => {
                write!(f, "invalid frame dimensions: {width}x{height}")
            }
            Self::InvalidInput(msg) => write!(f, "invalid frame input: {msg}"),
            Self::InvalidHeader(msg) => write!(f, "invalid container header: {msg}"),
            Self::BitstreamError(msg) => write!(f, "bitstream error: {msg}"),
            Self::EntropyError(msg) => write!(f, "entropy coding error: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported AV2 feature: {msg}"),
            Self::NeedMoreData => write!(f, "encoder needs more input data"),
            Self::EndOfStream => write!(f, "end of stream"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for Av2Error {}

impl From<std::io::Error> for Av2Error {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Av2Error>;
