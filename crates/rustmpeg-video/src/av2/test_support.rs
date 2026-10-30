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

//! Oracle resolution for tests that compare against external decoders.
//!
//! No oracle path is hard-coded. A test names the environment variable that
//! overrides the binary (for example `DAV2D`) and the executable name to look
//! up on `PATH`; when neither resolves the test prints why and skips.

use std::path::PathBuf;

/// Resolve an oracle binary from `var`, then from `PATH`, by `name`.
///
/// Returns `None` when the variable is unset and `name` is not on `PATH`.
pub(crate) fn oracle_bin(var: &str, name: &str) -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(var) {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Some(path);
        }
        eprintln!("{var} does not point to a file: {}", path.display());
        return None;
    }
    let search = std::env::var_os("PATH")?;
    std::env::split_paths(&search)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Resolve a stream or asset file from `var`.
///
/// Returns `None` when the variable is unset or does not name a file.
pub(crate) fn oracle_file(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(var)?);
    path.is_file().then_some(path)
}
