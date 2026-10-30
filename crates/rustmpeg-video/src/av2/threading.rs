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

//! Clean-room Wave 3 Multi-Threaded Tile-Parallel Execution Engine (`threading.rs`).
//!
//! Uses Rust's safe `std::thread::scope` API to encode independent AV2 tiles in parallel
//! across configurable worker threads (`threads >= 1`) with deterministic tile ordering
//! and zero `unsafe` code.

use std::thread;

/// Descriptor for a single rectangular tile job within a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileJob {
    pub tile_idx: usize,
    pub tile_row: usize,
    pub tile_col: usize,
    pub col_start4: usize,
    pub col_end4: usize,
    pub row_start4: usize,
    pub row_end4: usize,
}

/// Executes `encode_tile_fn` across all `jobs` in parallel using up to `num_threads` scoped
/// OS worker threads and returns the results ordered deterministically by `tile_idx` (`0..jobs.len()`).
pub fn execute_tiles_parallel<R, F>(
    jobs: &[TileJob],
    num_threads: usize,
    encode_tile_fn: F,
) -> Vec<R>
where
    R: Send,
    F: Fn(&TileJob) -> R + Sync,
{
    if jobs.is_empty() {
        return Vec::new();
    }
    let workers = num_threads.max(1).min(jobs.len());
    if workers == 1 {
        return jobs.iter().map(&encode_tile_fn).collect();
    }

    let chunk_size = jobs.len().div_ceil(workers);
    let mut results = Vec::with_capacity(jobs.len());

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for chunk in jobs.chunks(chunk_size) {
            let f_ref = &encode_tile_fn;
            handles.push(s.spawn(move || {
                let mut local_out = Vec::with_capacity(chunk.len());
                for job in chunk {
                    local_out.push(f_ref(job));
                }
                local_out
            }));
        }
        for h in handles {
            let chunk_res = h.join().expect("Tile worker thread panicked");
            results.extend(chunk_res);
        }
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_parallel_tile_execution_preserves_order() {
        let jobs: Vec<TileJob> = (0..8)
            .map(|i| TileJob {
                tile_idx: i,
                tile_row: i / 4,
                tile_col: i % 4,
                col_start4: (i % 4) * 16,
                col_end4: ((i % 4) + 1) * 16,
                row_start4: (i / 4) * 16,
                row_end4: ((i / 4) + 1) * 16,
            })
            .collect();

        let out = execute_tiles_parallel(&jobs, 4, |job| {
            job.tile_idx * 100 + (job.col_end4 - job.col_start4)
        });
        assert_eq!(out.len(), 8);
        for (i, &val) in out.iter().enumerate() {
            assert_eq!(val, i * 100 + 16);
        }
    }
}
