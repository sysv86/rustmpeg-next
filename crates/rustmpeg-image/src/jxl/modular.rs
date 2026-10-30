// SPDX-License-Identifier: Apache-2.0

//! JPEG XL Modular sub-bitstream decoder and clean-room encoder (ISO/IEC 18181-1 Annex G & H).
//!
//! Modular streams represent 2D integer sample grids and are used both for standalone
//! lossless/near-lossless Modular frames and for auxiliary sub-images inside VarDCT frames
//! (such as `GlobalModular`, quantized LF DC coefficients `LfCoeff`, and `HfMetadata`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::num::Wrapping;

use super::JxlError;
use super::bits::{BitReader, BitWriter, U32Specifier, pack_signed, unpack_signed};
use super::entropy::{EntropyDecoder, EntropyEncoder, HybridUintConfig, JxlEntropyMode};

/// Weighted predictor (`SelfCorrecting`) header parameters (ISO/IEC 18181-1 Annex G.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WpHeader {
    /// Whether default weighted predictor parameters are used.
    pub default_wp: bool,
    /// Parameter `p1` (5 bits, default 16).
    pub wp_p1: u32,
    /// Parameter `p2` (5 bits, default 10).
    pub wp_p2: u32,
    /// Parameter `p3a` (5 bits, default 7).
    pub wp_p3a: u32,
    /// Parameter `p3b` (5 bits, default 7).
    pub wp_p3b: u32,
    /// Parameter `p3c` (5 bits, default 7).
    pub wp_p3c: u32,
    /// Parameter `p3d` (5 bits, default 0).
    pub wp_p3d: u32,
    /// Parameter `p3e` (5 bits, default 0).
    pub wp_p3e: u32,
    /// Weight `w0` (4 bits, default 13).
    pub wp_w0: u32,
    /// Weight `w1` (4 bits, default 12).
    pub wp_w1: u32,
    /// Weight `w2` (4 bits, default 12).
    pub wp_w2: u32,
    /// Weight `w3` (4 bits, default 12).
    pub wp_w3: u32,
}

impl Default for WpHeader {
    fn default() -> Self {
        Self {
            default_wp: true,
            wp_p1: 16,
            wp_p2: 10,
            wp_p3a: 7,
            wp_p3b: 7,
            wp_p3c: 7,
            wp_p3d: 0,
            wp_p3e: 0,
            wp_w0: 13,
            wp_w1: 12,
            wp_w2: 12,
            wp_w3: 12,
        }
    }
}

impl WpHeader {
    /// Parses a `WpHeader` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let default_wp = br.read_bool()?;
        if default_wp {
            return Ok(Self::default());
        }
        Ok(Self {
            default_wp: false,
            wp_p1: br.read_bits(5)?,
            wp_p2: br.read_bits(5)?,
            wp_p3a: br.read_bits(5)?,
            wp_p3b: br.read_bits(5)?,
            wp_p3c: br.read_bits(5)?,
            wp_p3d: br.read_bits(5)?,
            wp_p3e: br.read_bits(5)?,
            wp_w0: br.read_bits(4)?,
            wp_w1: br.read_bits(4)?,
            wp_w2: br.read_bits(4)?,
            wp_w3: br.read_bits(4)?,
        })
    }

    /// Writes a `WpHeader` to the bitstream.
    pub fn write(&self, bw: &mut BitWriter) {
        bw.write_bool(self.default_wp);
        if !self.default_wp {
            bw.write_bits(5, self.wp_p1);
            bw.write_bits(5, self.wp_p2);
            bw.write_bits(5, self.wp_p3a);
            bw.write_bits(5, self.wp_p3b);
            bw.write_bits(5, self.wp_p3c);
            bw.write_bits(5, self.wp_p3d);
            bw.write_bits(5, self.wp_p3e);
            bw.write_bits(4, self.wp_w0);
            bw.write_bits(4, self.wp_w1);
            bw.write_bits(4, self.wp_w2);
            bw.write_bits(4, self.wp_w3);
        }
    }
}

/// Modular sample predictor mode (ISO/IEC 18181-1 Table G.3).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum Predictor {
    /// Always predicts 0.
    #[default]
    Zero = 0,
    /// Left neighbor `W`.
    West = 1,
    /// Top neighbor `N`.
    North = 2,
    /// `(W + N) / 2`.
    AvgWestAndNorth = 3,
    /// Select `W` or `N` depending on distance to `NW`.
    Select = 4,
    /// Clamped gradient `clamp(N + W - NW, min(W, N), max(W, N))`.
    Gradient = 5,
    /// Weighted self-correcting predictor.
    SelfCorrecting = 6,
    /// Top-right neighbor `NE`.
    NorthEast = 7,
    /// Top-left neighbor `NW`.
    NorthWest = 8,
    /// Far-left neighbor `WW`.
    WestWest = 9,
    /// `(W + NW) / 2`.
    AvgWestAndNorthWest = 10,
    /// `(N + NW) / 2`.
    AvgNorthAndNorthWest = 11,
    /// `(N + NE) / 2`.
    AvgNorthAndNorthEast = 12,
    /// Weighted average of `N`, `NN`, `W`, `WW`, `NE`, `NEE`.
    AvgAll = 13,
}

impl Predictor {
    /// Converts a raw predictor ID (`0..=13`) into a [`Predictor`].
    pub fn from_u32(value: u32) -> Result<Self, JxlError> {
        Ok(match value {
            0 => Self::Zero,
            1 => Self::West,
            2 => Self::North,
            3 => Self::AvgWestAndNorth,
            4 => Self::Select,
            5 => Self::Gradient,
            6 => Self::SelfCorrecting,
            7 => Self::NorthEast,
            8 => Self::NorthWest,
            9 => Self::WestWest,
            10 => Self::AvgWestAndNorthWest,
            11 => Self::AvgNorthAndNorthWest,
            12 => Self::AvgNorthAndNorthEast,
            13 => Self::AvgAll,
            _ => {
                return Err(JxlError::InvalidBitstream("invalid Modular Predictor enum"));
            }
        })
    }

    /// Computes the predicted sample value for the current pixel.
    #[inline]
    pub fn predict(self, state: &PredictorState<'_>, sc_pred: Option<&PredictionResult>) -> i32 {
        match self {
            Self::Zero => 0,
            Self::West => state.w,
            Self::North => state.n,
            Self::AvgWestAndNorth => ((state.w as i64 + state.n as i64) / 2) as i32,
            Self::Select => {
                let n = state.n;
                let w = state.w;
                let nw = state.nw;
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            }
            Self::Gradient => grad_clamped_i32(state.n, state.w, state.nw),
            Self::SelfCorrecting => {
                let pred = sc_pred.map(|p| p.prediction).unwrap_or(0);
                ((pred + 3) >> 3) as i32
            }
            Self::NorthEast => state.ne(),
            Self::NorthWest => state.nw,
            Self::WestWest => state.ww(),
            Self::AvgWestAndNorthWest => ((state.w as i64 + state.nw as i64) / 2) as i32,
            Self::AvgNorthAndNorthWest => ((state.n as i64 + state.nw as i64) / 2) as i32,
            Self::AvgNorthAndNorthEast => ((state.n as i64 + state.ne() as i64) / 2) as i32,
            Self::AvgAll => {
                let n = state.n as i64;
                let w = state.w as i64;
                let nn = state.nn() as i64;
                let ww = state.ww() as i64;
                let nee = state.nee() as i64;
                let ne = state.ne() as i64;
                ((6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16) as i32
            }
        }
    }
}

/// Computes the clamped gradient `clamp(n + w - nw, min(w, n), max(w, n))`.
#[must_use]
#[inline]
pub fn grad_clamped_i32(n: i32, w: i32, nw: i32) -> i32 {
    let (hi, lo) = if w > n {
        (w as i64, n as i64)
    } else {
        (n as i64, w as i64)
    };
    (w as i64 + n as i64 - nw as i64).clamp(lo, hi) as i32
}

const DIV_LOOKUP: [u32; 65] = {
    let mut out = [0u32; 65];
    let mut i = 1usize;
    while i <= 64 {
        out[i] = ((1 << 24) / i) as u32;
        i += 1;
    }
    out
};

/// Result of the weighted self-correcting predictor for a single sample.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Scaled prediction (`sample << 3`).
    pub prediction: i64,
    /// Maximum signed true error among neighbors.
    pub max_error: i32,
    /// Sub-predictor values (`<< 3`).
    pub subpred: [i64; 4],
}

/// State of the weighted `SelfCorrecting` predictor across a channel grid.
#[derive(Debug, Clone)]
pub struct SelfCorrectingPredictor {
    width: u32,
    x: u32,
    y: u32,
    true_err_row: Vec<i32>,
    subpred_err_row: Vec<[u32; 4]>,
    wp: WpHeader,
    true_err_w: i32,
    true_err_nw: i32,
    true_err_n: i32,
    true_err_ne: i32,
    subpred_err_nw_ww: [u32; 4],
    subpred_err_n_w: [u32; 4],
    subpred_err_ne: [u32; 4],
}

impl SelfCorrectingPredictor {
    /// Creates a new `SelfCorrectingPredictor` for a grid of `width` columns.
    #[must_use]
    pub fn new(width: u32, wp_header: WpHeader) -> Self {
        Self {
            width,
            x: 0,
            y: 0,
            true_err_row: vec![0i32; width as usize],
            subpred_err_row: vec![[0u32; 4]; width as usize],
            wp: wp_header,
            true_err_w: 0,
            true_err_nw: 0,
            true_err_n: 0,
            true_err_ne: 0,
            subpred_err_nw_ww: [0; 4],
            subpred_err_n_w: [0; 4],
            subpred_err_ne: [0; 4],
        }
    }

    /// Predicts the current sample given neighbors `n`, `nw`, `ne`, `w`, `nn`.
    #[must_use]
    #[inline]
    pub fn predict(&self, n: i32, nw: i32, ne: i32, w: i32, nn: i32) -> PredictionResult {
        let true_err_w = self.true_err_w as i64;
        let true_err_nw = self.true_err_nw as i64;
        let true_err_n = self.true_err_n as i64;
        let true_err_ne = self.true_err_ne as i64;

        let n3 = (n as i64) << 3;
        let nw3 = (nw as i64) << 3;
        let ne3 = (ne as i64) << 3;
        let w3 = (w as i64) << 3;
        let nn3 = (nn as i64) << 3;

        let wp = &self.wp;
        let subpred = [
            w3 + ne3 - n3,
            n3 - (((true_err_w + true_err_n + true_err_ne) * wp.wp_p1 as i64) >> 5),
            w3 - (((true_err_w + true_err_n + true_err_nw) * wp.wp_p2 as i64) >> 5),
            n3 - ((true_err_nw * wp.wp_p3a as i64
                + true_err_n * wp.wp_p3b as i64
                + true_err_ne * wp.wp_p3c as i64
                + (nn3 - n3) * wp.wp_p3d as i64
                + (nw3 - w3) * wp.wp_p3e as i64)
                >> 5),
        ];

        let mut subpred_err_sum = [0u32; 4];
        for (i, sum) in subpred_err_sum.iter_mut().enumerate() {
            *sum = self.subpred_err_nw_ww[i]
                .wrapping_add(self.subpred_err_n_w[i])
                .wrapping_add(self.subpred_err_ne[i]);
        }

        let wp_wn = [wp.wp_w0, wp.wp_w1, wp.wp_w2, wp.wp_w3];
        let mut weight = [0u32; 4];
        for ((w_out, err_sum), maxweight) in weight.iter_mut().zip(subpred_err_sum).zip(wp_wn) {
            let shift = ((err_sum as u64 + 1) >> 5).checked_ilog2().unwrap_or(0);
            *w_out = 4 + ((maxweight * DIV_LOOKUP[(err_sum >> shift) as usize + 1]) >> shift);
        }

        let sum_weights: u32 = weight.iter().copied().sum();
        let log_weight = ((sum_weights as u64) >> 4).checked_ilog2().unwrap_or(0);
        for w_out in &mut weight {
            *w_out >>= log_weight;
        }
        let sum_weights: u32 = weight.iter().copied().sum();
        let mut s = (sum_weights as i64 >> 1) - 1;
        for (sp, wt) in subpred.into_iter().zip(weight) {
            s += sp * wt as i64;
        }
        let mut prediction = (s * DIV_LOOKUP[sum_weights as usize] as i64) >> 24;
        if (true_err_n ^ true_err_w) | (true_err_n ^ true_err_nw) <= 0 {
            let min = n3.min(w3).min(ne3);
            let max = n3.max(w3).max(ne3);
            prediction = prediction.clamp(min, max);
        }

        let true_errors = [true_err_n, true_err_nw, true_err_ne];
        let mut max_error = true_err_w;
        for err in true_errors {
            if err.abs() > max_error.abs() {
                max_error = err;
            }
        }

        PredictionResult {
            prediction,
            max_error: max_error as i32,
            subpred,
        }
    }

    /// Records the decoded/encoded `sample` and updates running error rows.
    #[inline]
    pub fn record(&mut self, pred: &PredictionResult, sample: i32) {
        let sample64 = sample as i64;
        let true_err = pred.prediction - (sample64 << 3);
        let mut subpred_err = [0u32; 4];
        for (err, sp) in subpred_err.iter_mut().zip(pred.subpred) {
            *err = ((sp.abs_diff(sample64 << 3) + 3) >> 3) as u32;
        }

        self.true_err_row[self.x as usize] = true_err as i32;
        self.subpred_err_row[self.x as usize] = subpred_err;
        self.x += 1;

        if self.x >= self.width {
            self.y += 1;
            self.x = 0;

            self.true_err_w = 0;
            self.true_err_n = self.true_err_row[0];
            self.true_err_nw = self.true_err_n;
            self.subpred_err_n_w = self.subpred_err_row[0];
            self.subpred_err_nw_ww = self.subpred_err_n_w;
            if self.width <= 1 {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else {
                self.true_err_ne = self.true_err_row[1];
                self.subpred_err_ne = self.subpred_err_row[1];
            }
        } else {
            self.true_err_w = true_err as i32;
            self.true_err_nw = self.true_err_n;
            self.true_err_n = self.true_err_ne;
            self.subpred_err_nw_ww = self.subpred_err_n_w;
            self.subpred_err_n_w = self.subpred_err_ne;
            for (w0, w1) in self.subpred_err_n_w.iter_mut().zip(subpred_err) {
                *w0 = w0.wrapping_add(w1);
            }

            if self.x + 1 >= self.width {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else if self.y != 0 {
                let x = self.x as usize;
                self.true_err_ne = self.true_err_row[x + 1];
                self.subpred_err_ne = self.subpred_err_row[x + 1];
            }
        }
    }
}

/// 2D integer sample grid for a single Modular channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularGrid {
    /// Grid width in samples.
    pub width: usize,
    /// Grid height in samples.
    pub height: usize,
    /// Row-major sample buffer of length `width * height`.
    pub data: Vec<i32>,
}

impl ModularGrid {
    /// Allocates a zero-filled `ModularGrid` of `width x height`.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0i32; width * height],
        }
    }

    /// Reads sample at `(x, y)`.
    #[must_use]
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> i32 {
        self.data[y * self.width + x]
    }

    /// Sets sample at `(x, y)`.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: i32) {
        self.data[y * self.width + x] = val;
    }

    /// Returns an immutable slice for row `y`.
    #[must_use]
    #[inline]
    pub fn row(&self, y: usize) -> &[i32] {
        let start = y * self.width;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable slice for row `y`.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [i32] {
        let start = y * self.width;
        &mut self.data[start..start + self.width]
    }

    /// Copies a sub-rectangle from `src` into `self` at `(dst_x, dst_y)`.
    pub fn copy_from_rect(&mut self, dst_x: usize, dst_y: usize, src: &ModularGrid) {
        for y in 0..src.height {
            if dst_y + y >= self.height {
                break;
            }
            let copy_w = src.width.min(self.width.saturating_sub(dst_x));
            let dst_row = &mut self.row_mut(dst_y + y)[dst_x..dst_x + copy_w];
            let src_row = &src.row(y)[..copy_w];
            dst_row.copy_from_slice(src_row);
        }
    }

    /// Extracts a sub-rectangle of size `(rect_w, rect_h)` starting at `(src_x, src_y)`.
    #[must_use]
    pub fn extract_rect(&self, src_x: usize, src_y: usize, rect_w: usize, rect_h: usize) -> Self {
        let w = rect_w.min(self.width.saturating_sub(src_x));
        let h = rect_h.min(self.height.saturating_sub(src_y));
        let mut out = Self::new(w, h);
        for y in 0..h {
            out.row_mut(y)
                .copy_from_slice(&self.row(src_y + y)[src_x..src_x + w]);
        }
        out
    }
}

/// Spatial predictor and MA property state across a channel scanline.
#[derive(Debug)]
pub struct PredictorState<'a> {
    width: u32,
    prev_row: Vec<i32>,
    curr_row: Vec<i32>,
    prev_channels_rev: Vec<&'a ModularGrid>,
    self_correcting: Option<SelfCorrectingPredictor>,
    y: u32,
    x: u32,
    w: i32,
    n: i32,
    nw: i32,
    prev_grad: i32,
}

impl<'a> PredictorState<'a> {
    /// Creates a new `PredictorState` for a grid of `width` columns.
    #[must_use]
    pub fn new(
        width: u32,
        prev_channels_rev: Vec<&'a ModularGrid>,
        wp_header: Option<&WpHeader>,
    ) -> Self {
        Self {
            width,
            prev_row: Vec::with_capacity(width as usize),
            curr_row: Vec::with_capacity(width as usize),
            prev_channels_rev,
            self_correcting: wp_header.map(|wp| SelfCorrectingPredictor::new(width, wp.clone())),
            y: 0,
            x: 0,
            w: 0,
            n: 0,
            nw: 0,
            prev_grad: 0,
        }
    }

    #[inline]
    fn nn(&self) -> i32 {
        self.curr_row
            .get(self.x as usize)
            .copied()
            .unwrap_or(self.n)
    }

    #[inline]
    fn ne(&self) -> i32 {
        let x = self.x;
        if self.prev_row.is_empty() || x + 1 >= self.width {
            self.n
        } else {
            self.prev_row[x as usize + 1]
        }
    }

    #[inline]
    fn nee(&self) -> i32 {
        let x = self.x;
        if self.prev_row.is_empty() || x + 2 >= self.width {
            self.ne()
        } else {
            self.prev_row[x as usize + 2]
        }
    }

    #[inline]
    fn ww(&self) -> i32 {
        if let Some(x2) = self.x.checked_sub(2) {
            self.curr_row[x2 as usize]
        } else {
            self.w
        }
    }

    /// Runs the self-correcting predictor if enabled.
    #[must_use]
    #[inline]
    pub fn step_sc(&self) -> Option<PredictionResult> {
        self.self_correcting
            .as_ref()
            .map(|sc| sc.predict(self.n, self.nw, self.ne(), self.w, self.nn()))
    }

    /// Evaluates MA property `prop` at the current `(x, y)` position.
    #[must_use]
    #[inline]
    pub fn property(
        &self,
        prop: u32,
        channel: u32,
        stream_idx: u32,
        sc_pred: Option<&PredictionResult>,
    ) -> i32 {
        let w_nw = self.w.wrapping_sub(self.nw);
        match prop {
            0 => channel as i32,
            1 => stream_idx as i32,
            2 => self.y as i32,
            3 => self.x as i32,
            4 => self.n.unsigned_abs() as i32,
            5 => self.w.unsigned_abs() as i32,
            6 => self.n,
            7 => self.w,
            8 => self.w.wrapping_sub(self.prev_grad),
            9 => w_nw.wrapping_add(self.n),
            10 => w_nw,
            11 => self.nw.wrapping_sub(self.n),
            12 => self.n.wrapping_sub(self.ne()),
            13 => self.n.wrapping_sub(self.nn()),
            14 => self.w.wrapping_sub(self.ww()),
            15 => sc_pred.map(|p| p.max_error).unwrap_or(0),
            p => {
                let extra = (p - 16) as usize;
                let prev_idx = extra / 4;
                let kind = extra % 4;
                let Some(prev_ch) = self.prev_channels_rev.get(prev_idx) else {
                    return 0;
                };
                let x = self.x as usize;
                let y = self.y as usize;
                let c = prev_ch.get(x, y);
                match kind {
                    0 => c.abs(),
                    1 => c,
                    2 | 3 => {
                        let g = match (x, y) {
                            (0, 0) => 0,
                            (0, y) => prev_ch.get(0, y - 1),
                            (x, 0) => prev_ch.get(x - 1, 0),
                            (x, y) => {
                                let nw = prev_ch.get(x - 1, y - 1);
                                let n = prev_ch.get(x, y - 1);
                                let w = prev_ch.get(x - 1, y);
                                grad_clamped_i32(n, w, nw)
                            }
                        };
                        if kind == 2 {
                            c.abs_diff(g) as i32
                        } else {
                            c.wrapping_sub(g)
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Records the reconstructed `sample` at `(x, y)` and advances the predictor state.
    #[inline]
    pub fn record(&mut self, sample: i32, sc_pred: Option<&PredictionResult>) {
        let prop9 = self.w.wrapping_sub(self.nw).wrapping_add(self.n);
        if let (Some(sc), Some(pred)) = (&mut self.self_correcting, sc_pred) {
            sc.record(pred, sample);
        }

        if let Some(out) = self.curr_row.get_mut(self.x as usize) {
            *out = sample;
        } else {
            self.curr_row.push(sample);
        }
        self.x += 1;

        if self.x >= self.width {
            self.y += 1;
            self.x = 0;

            std::mem::swap(&mut self.prev_row, &mut self.curr_row);
            self.prev_grad = 0;

            if !self.prev_row.is_empty() {
                let n = self.prev_row[0];
                self.n = n;
                self.w = n;
                self.nw = n;
            }
        } else {
            self.prev_grad = prop9;
            self.w = sample;
            if self.prev_row.is_empty() {
                self.nw = sample;
                self.n = sample;
            } else {
                self.nw = self.n;
                self.n = self.prev_row[self.x as usize];
            }
        }
    }
}

/// Leaf node of a Meta-Adaptive (MA) decision tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaTreeLeaf {
    /// Entropy cluster ID (`cluster_map[ctx]`).
    pub cluster: u8,
    /// Predictor mode for this leaf.
    pub predictor: Predictor,
    /// Additive residual offset.
    pub offset: i32,
    /// Multiplicative residual scale.
    pub multiplier: u32,
}

/// Node in a Meta-Adaptive (MA) decision tree (ISO/IEC 18181-1 Annex G.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaTreeNode {
    /// Decision node branching on `property > value` (`left`) vs `property <= value` (`right`).
    Decision {
        /// Property index (`0..`).
        property: u32,
        /// Threshold value.
        value: i32,
        /// Branch taken when `property_value > value`.
        left: Box<MaTreeNode>,
        /// Branch taken when `property_value <= value`.
        right: Box<MaTreeNode>,
    },
    /// Leaf node specifying predictor and entropy context cluster.
    Leaf(MaTreeLeaf),
}

impl MaTreeNode {
    /// Returns whether any node in the subtree uses `Predictor::SelfCorrecting` or property 15.
    #[must_use]
    pub fn needs_self_correcting(&self) -> bool {
        match self {
            Self::Decision {
                property,
                left,
                right,
                ..
            } => *property == 15 || left.needs_self_correcting() || right.needs_self_correcting(),
            Self::Leaf(leaf) => leaf.predictor == Predictor::SelfCorrecting,
        }
    }

    /// Prunes branches that depend only on constant channel/stream properties `(0, 1)` or
    /// out-of-range previous channels.
    #[must_use]
    pub fn prune(&self, channel: u32, stream_idx: u32, prev_channels: u32) -> MaTreeNode {
        match self {
            Self::Decision {
                property: prop @ (0 | 1),
                value,
                left,
                right,
            } => {
                let target = if *prop == 0 {
                    channel as i32
                } else {
                    stream_idx as i32
                };
                if target > *value {
                    left.prune(channel, stream_idx, prev_channels)
                } else {
                    right.prune(channel, stream_idx, prev_channels)
                }
            }
            Self::Decision {
                property,
                value,
                left,
                right,
            } if *property >= 16 => {
                let prev_idx = (*property - 16) / 4;
                if prev_idx >= prev_channels {
                    if 0 > *value {
                        left.prune(channel, stream_idx, prev_channels)
                    } else {
                        right.prune(channel, stream_idx, prev_channels)
                    }
                } else {
                    Self::Decision {
                        property: *property,
                        value: *value,
                        left: Box::new(left.prune(channel, stream_idx, prev_channels)),
                        right: Box::new(right.prune(channel, stream_idx, prev_channels)),
                    }
                }
            }
            Self::Decision {
                property,
                value,
                left,
                right,
            } => Self::Decision {
                property: *property,
                value: *value,
                left: Box::new(left.prune(channel, stream_idx, prev_channels)),
                right: Box::new(right.prune(channel, stream_idx, prev_channels)),
            },
            Self::Leaf(leaf) => Self::Leaf(leaf.clone()),
        }
    }

    /// Traverses the decision tree to find the active [`MaTreeLeaf`] for the current pixel.
    #[must_use]
    #[inline]
    pub fn eval(
        &self,
        state: &PredictorState<'_>,
        channel: u32,
        stream_idx: u32,
        sc_pred: Option<&PredictionResult>,
    ) -> &MaTreeLeaf {
        let mut cur = self;
        loop {
            match cur {
                Self::Decision {
                    property,
                    value,
                    left,
                    right,
                } => {
                    let prop_val = state.property(*property, channel, stream_idx, sc_pred);
                    if prop_val > *value {
                        cur = left;
                    } else {
                        cur = right;
                    }
                }
                Self::Leaf(leaf) => return leaf,
            }
        }
    }
}

/// Meta-Adaptive (MA) tree and associated entropy decoder configuration.
#[derive(Debug, Clone)]
pub struct MaConfig {
    /// Root of the decoded MA decision tree.
    pub tree: MaTreeNode,
    /// Entropy decoder template for decoding Modular channel residuals.
    pub decoder: EntropyDecoder,
}

impl MaConfig {
    /// Parses a [`MaConfig`] from the bitstream.
    pub fn parse(br: &mut BitReader<'_>, node_limit: usize) -> Result<Self, JxlError> {
        enum FoldingTree {
            Decision(u32, i32),
            Leaf {
                ctx: u32,
                predictor: Predictor,
                offset: i32,
                multiplier: u32,
            },
        }

        let mut tree_decoder = EntropyDecoder::parse(br, 6)?;
        let cluster_1 = tree_decoder.cluster_map()[1];
        if let Some(tok) = tree_decoder.single_token(cluster_1)
            && tok != 0
        {
            return Err(JxlError::InvalidBitstream("infinite MA tree"));
        }

        let mut ctx = 0u32;
        let mut nodes_left = 1usize;
        let mut nodes = Vec::with_capacity(16);

        tree_decoder.begin(br)?;
        while nodes_left > 0 {
            if nodes.len() > node_limit || nodes.len() >= (1 << 22) {
                return Err(JxlError::LimitExceeded("MA tree node limit exceeded"));
            }
            nodes_left -= 1;
            let property = tree_decoder.read_varint(br, 1)?;
            if let Some(prop) = property.checked_sub(1) {
                let value = unpack_signed(tree_decoder.read_varint(br, 0)?);
                nodes.push(FoldingTree::Decision(prop, value));
                nodes_left += 2;
            } else {
                let predictor = Predictor::from_u32(tree_decoder.read_varint(br, 2)?)?;
                let offset = unpack_signed(tree_decoder.read_varint(br, 3)?);
                let mul_log = tree_decoder.read_varint(br, 4)?;
                if mul_log > 30 {
                    return Err(JxlError::InvalidBitstream("MA tree mul_log > 30"));
                }
                let mul_bits = tree_decoder.read_varint(br, 5)?;
                if mul_bits > (1u32 << (31 - mul_log)) - 2 {
                    return Err(JxlError::InvalidBitstream("MA tree mul_bits overflow"));
                }
                let multiplier = (mul_bits + 1) << mul_log;
                nodes.push(FoldingTree::Leaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                });
                ctx += 1;
            }
        }
        tree_decoder.finalize()?;

        let decoder = EntropyDecoder::parse(br, ctx)?;
        let cluster_map = decoder.cluster_map();

        let mut q = VecDeque::<(Box<MaTreeNode>, usize)>::new();
        for node in nodes.into_iter().rev() {
            match node {
                FoldingTree::Decision(property, value) => {
                    let (right, dr) = q
                        .pop_front()
                        .ok_or(JxlError::InvalidBitstream("corrupt MA tree right child"))?;
                    let (left, dl) = q
                        .pop_front()
                        .ok_or(JxlError::InvalidBitstream("corrupt MA tree left child"))?;
                    let depth = dr.max(dl) + 1;
                    if depth > 2048 {
                        return Err(JxlError::LimitExceeded("MA tree depth > 2048"));
                    }
                    q.push_back((
                        Box::new(MaTreeNode::Decision {
                            property,
                            value,
                            left,
                            right,
                        }),
                        depth,
                    ));
                }
                FoldingTree::Leaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                } => {
                    let cluster = cluster_map[ctx as usize];
                    q.push_back((
                        Box::new(MaTreeNode::Leaf(MaTreeLeaf {
                            cluster,
                            predictor,
                            offset,
                            multiplier,
                        })),
                        0,
                    ));
                }
            }
        }

        if q.len() != 1 {
            return Err(JxlError::InvalidBitstream("unbalanced MA tree"));
        }
        let (tree, _) = q.pop_front().unwrap();
        Ok(Self {
            tree: *tree,
            decoder,
        })
    }

    /// Writes a 4-leaf context-adaptive MA tree (`4` contexts) to `bw`.
    ///
    /// - If `has_palette_lead` is true:
    ///   - `channel > 0`: splits on `|W| > 4` into `ctx 0` (`Predictor::Zero`) and `ctx 1` (`Predictor::Zero`).
    ///   - `channel <= 0` (palette table): splits on `|W| > 32` into `ctx 2` (`Predictor::West`) and `ctx 3` (`Predictor::West`).
    /// - Else:
    ///   - `channel > 0`: splits on `|W| > 16` into `ctx 0` (`predictor`) and `ctx 1` (`predictor`).
    ///   - `channel <= 0`: splits on `|W| > 32` into `ctx 2` (`predictor`) and `ctx 3` (`predictor`).
    pub fn write_adaptive_tree(
        bw: &mut BitWriter,
        predictor: Predictor,
        has_palette_lead: bool,
        multiplier: u32,
    ) -> Result<MaTreeNode, JxlError> {
        let mul = multiplier.max(1);
        let mul_log = mul.trailing_zeros();
        let mul_bits = (mul >> mul_log) - 1;
        let (p_left, t_left, p_right, t_right) = if has_palette_lead {
            (Predictor::Zero, 4i32, Predictor::West, 32i32)
        } else {
            (predictor, 16i32, predictor, 32i32)
        };

        let mut tree_enc = EntropyEncoder::single_cluster(6, HybridUintConfig::DEFAULT_4_2_0);
        // Node 0: Decision(prop = 0 [channel], val = 0) -> property + 1 = 1
        tree_enc.record_varint(1, 1);
        tree_enc.record_varint(0, pack_signed(0));
        // Node 1 (left: channel > 0): Decision(prop = 5 [|W|], val = t_left) -> property + 1 = 6
        tree_enc.record_varint(1, 6);
        tree_enc.record_varint(0, pack_signed(t_left));
        // Node 2 (right: channel <= 0): Decision(prop = 5 [|W|], val = t_right) -> property + 1 = 6
        tree_enc.record_varint(1, 6);
        tree_enc.record_varint(0, pack_signed(t_right));

        // 4 Leaves: ctx 0, 1, 2, 3
        for &pred in &[p_left, p_left, p_right, p_right] {
            tree_enc.record_varint(1, 0); // Leaf
            tree_enc.record_varint(2, pred as u32);
            tree_enc.record_varint(3, 0); // offset = 0
            tree_enc.record_varint(4, mul_log);
            tree_enc.record_varint(5, mul_bits);
        }
        tree_enc.finish(bw)?;

        let leaf = |cluster: u8, pred: Predictor| {
            Box::new(MaTreeNode::Leaf(MaTreeLeaf {
                cluster,
                predictor: pred,
                offset: 0,
                multiplier: mul,
            }))
        };
        Ok(MaTreeNode::Decision {
            property: 0,
            value: 0,
            left: Box::new(MaTreeNode::Decision {
                property: 5,
                value: t_left,
                left: leaf(0, p_left),
                right: leaf(1, p_left),
            }),
            right: Box::new(MaTreeNode::Decision {
                property: 5,
                value: t_right,
                left: leaf(2, p_right),
                right: leaf(3, p_right),
            }),
        })
    }

    /// Writes a single-leaf MA tree (`ctx = 1`, `offset = 0`, `multiplier = 1`) to `bw`.
    pub fn write_single_leaf_tree(
        bw: &mut BitWriter,
        predictor: Predictor,
    ) -> Result<(), JxlError> {
        let mut tree_enc = EntropyEncoder::single_cluster(6, HybridUintConfig::DEFAULT_4_2_0);
        tree_enc.record_varint(1, 0); // property == 0 -> Leaf
        tree_enc.record_varint(2, predictor as u32);
        tree_enc.record_varint(3, 0); // offset = 0
        tree_enc.record_varint(4, 0); // mul_log = 0
        tree_enc.record_varint(5, 0); // mul_bits = 0
        tree_enc.finish(bw)
    }
}

/// Subsampling shift specification for a Modular channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelShift {
    /// JPEG chroma subsampling flags.
    JpegUpsampling {
        /// Whether any channel is horizontally subsampled.
        has_h_subsample: bool,
        /// Whether this channel is horizontally subsampled.
        h_subsample: bool,
        /// Whether any channel is vertically subsampled.
        has_v_subsample: bool,
        /// Whether this channel is vertically subsampled.
        v_subsample: bool,
    },
    /// Uniform power-of-two shift in both dimensions.
    Shifts(u32),
    /// Explicit `(hshift, vshift)` power-of-two shifts.
    Raw(i32, i32),
}

impl ChannelShift {
    /// Constructs a uniform shift.
    #[must_use]
    #[inline]
    pub const fn from_shift(shift: u32) -> Self {
        Self::Shifts(shift)
    }

    /// Constructs a `ChannelShift` from `jpeg_upsampling` for channel `idx` (`0..3`).
    #[must_use]
    pub fn from_jpeg_upsampling(jpeg_upsampling: [u32; 3], idx: usize) -> Self {
        let upsampling = jpeg_upsampling[idx];
        let hscale = jpeg_upsampling.into_iter().any(|v| v == 1 || v == 2);
        let vscale = jpeg_upsampling.into_iter().any(|v| v == 1 || v == 3);
        let (h, v) = match upsampling {
            0 => (hscale, vscale),
            1 => (false, false),
            2 => (false, vscale),
            3 => (hscale, false),
            _ => (false, false),
        };
        Self::JpegUpsampling {
            has_h_subsample: hscale,
            h_subsample: h,
            has_v_subsample: vscale,
            v_subsample: v,
        }
    }

    /// Horizontal shift (`>= 0`, or `-1` for meta-channels).
    #[must_use]
    #[inline]
    pub const fn hshift(&self) -> i32 {
        match *self {
            Self::JpegUpsampling { h_subsample, .. } => h_subsample as i32,
            Self::Shifts(s) => s as i32,
            Self::Raw(h, _) => h,
        }
    }

    /// Vertical shift (`>= 0`, or `-1` for meta-channels).
    #[must_use]
    #[inline]
    pub const fn vshift(&self) -> i32 {
        match *self {
            Self::JpegUpsampling { v_subsample, .. } => v_subsample as i32,
            Self::Shifts(s) => s as i32,
            Self::Raw(_, v) => v,
        }
    }

    /// Computes shifted `(width, height)` for an image of `(width, height)`.
    #[must_use]
    pub fn shift_size(&self, (width, height): (u32, u32)) -> (u32, u32) {
        match *self {
            Self::JpegUpsampling {
                has_h_subsample,
                has_v_subsample,
                h_subsample,
                v_subsample,
            } => {
                let w = if has_h_subsample {
                    let s = width.div_ceil(2);
                    if h_subsample { s } else { s * 2 }
                } else {
                    width
                };
                let h = if has_v_subsample {
                    let s = height.div_ceil(2);
                    if v_subsample { s } else { s * 2 }
                } else {
                    height
                };
                (w, h)
            }
            Self::Shifts(s) => {
                let add = (1u32 << s) - 1;
                ((width + add) >> s, (height + add) >> s)
            }
            Self::Raw(h, v) => {
                let hs = h.max(0) as u32;
                let vs = v.max(0) as u32;
                let add_h = (1u32 << hs) - 1;
                let add_v = (1u32 << vs) - 1;
                ((width + add_h) >> hs, (height + add_v) >> vs)
            }
        }
    }
}

/// Dimensions and shift metadata for a single Modular channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularChannelInfo {
    /// Current channel width in samples.
    pub width: u32,
    /// Current channel height in samples.
    pub height: u32,
    /// Original un-squeezed channel width.
    pub original_width: u32,
    /// Original un-squeezed channel height.
    pub original_height: u32,
    /// Horizontal subsampling shift (`-1` for unshiftable meta-channels).
    pub hshift: i32,
    /// Vertical subsampling shift (`-1` for unshiftable meta-channels).
    pub vshift: i32,
    /// Initial channel shift descriptor.
    pub original_shift: ChannelShift,
}

impl ModularChannelInfo {
    /// Creates a new `ModularChannelInfo` from original dimensions and a `ChannelShift`.
    #[must_use]
    pub fn new(original_width: u32, original_height: u32, shift: ChannelShift) -> Self {
        let (width, height) = shift.shift_size((original_width, original_height));
        Self {
            width,
            height,
            original_width,
            original_height,
            hshift: shift.hshift(),
            vshift: shift.vshift(),
            original_shift: shift,
        }
    }

    /// Creates an unshiftable meta-channel (e.g., palette table) of `width x height`.
    #[must_use]
    pub fn new_unshiftable(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            original_width: width,
            original_height: height,
            hshift: -1,
            vshift: -1,
            original_shift: ChannelShift::from_shift(0),
        }
    }
}

/// Collection of Modular channel descriptors and meta-channel count.
#[derive(Debug, Clone)]
pub struct ModularChannels {
    /// Per-channel dimension and shift descriptors.
    pub info: Vec<ModularChannelInfo>,
    /// Number of leading meta-channels.
    pub nb_meta_channels: u32,
}

impl ModularChannels {
    /// Creates `ModularChannels` from a list of `ModularChannelInfo`.
    #[must_use]
    pub fn new(info: Vec<ModularChannelInfo>) -> Self {
        Self {
            info,
            nb_meta_channels: 0,
        }
    }
}

/// Reversible Colour Transform (RCT) parameters (ISO/IEC 18181-1 Annex H.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rct {
    /// Index of the first of the 3 channels to transform.
    pub begin_c: u32,
    /// RCT type (`0..41`: `permutation = rct_type / 7`, `ty = rct_type % 7`).
    pub rct_type: u32,
}

impl Rct {
    /// Parses an `Rct` transform descriptor.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let begin_c = br.read_u32(
            U32Specifier::u(0, 3),
            U32Specifier::u(8, 6),
            U32Specifier::u(72, 10),
            U32Specifier::u(1096, 13),
        )?;
        let rct_type = br.read_u32(
            U32Specifier::c(6),
            U32Specifier::u(0, 2),
            U32Specifier::u(2, 4),
            U32Specifier::u(10, 6),
        )?;
        if rct_type >= 42 {
            return Err(JxlError::InvalidBitstream("rct_type >= 42"));
        }
        Ok(Self { begin_c, rct_type })
    }

    /// Writes an `Rct` transform descriptor.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_u32(
            self.begin_c,
            U32Specifier::u(0, 3),
            U32Specifier::u(8, 6),
            U32Specifier::u(72, 10),
            U32Specifier::u(1096, 13),
        )?;
        bw.write_u32(
            self.rct_type,
            U32Specifier::c(6),
            U32Specifier::u(0, 2),
            U32Specifier::u(2, 4),
            U32Specifier::u(10, 6),
        )?;
        Ok(())
    }

    /// Validates channel dimensions for this RCT.
    pub fn prepare_channels(&self, channels: &mut ModularChannels) -> Result<(), JxlError> {
        let begin = self.begin_c as usize;
        let end = begin
            .checked_add(3)
            .ok_or(JxlError::InvalidBitstream("RCT begin_c overflow"))?;
        if end > channels.info.len() {
            return Err(JxlError::InvalidBitstream("RCT begin_c + 3 out of range"));
        }
        let w = channels.info[begin].width;
        let h = channels.info[begin].height;
        for ch in &channels.info[begin + 1..end] {
            if ch.width != w || ch.height != h {
                return Err(JxlError::InvalidBitstream(
                    "RCT channels have mismatched dimensions",
                ));
            }
        }
        Ok(())
    }

    /// Applies the forward RCT in-place on `grids[begin_c .. begin_c + 3]`.
    pub fn forward(&self, grids: &mut [ModularGrid]) {
        let begin = self.begin_c as usize;
        let permutation = self.rct_type / 7;
        let ty = self.rct_type % 7;
        let len = grids[begin].data.len();
        let (left, right) = grids.split_at_mut(begin + 1);
        let a_slice = &mut left[begin].data;
        let (mid, tail) = right.split_at_mut(1);
        let b_slice = &mut mid[0].data;
        let c_slice = &mut tail[0].data;

        for i in 0..len {
            let mut d = Wrapping(a_slice[i]);
            let mut e = Wrapping(b_slice[i]);
            let mut f = Wrapping(c_slice[i]);
            match permutation {
                1 => {
                    let tmp = d;
                    d = e;
                    e = f;
                    f = tmp;
                }
                2 => {
                    let tmp = d;
                    d = f;
                    f = e;
                    e = tmp;
                }
                3 => std::mem::swap(&mut e, &mut f),
                4 => std::mem::swap(&mut d, &mut e),
                5 => std::mem::swap(&mut d, &mut f),
                _ => {}
            }
            let (a, b, c) = if ty == 6 {
                let b = d - f;
                let tmp = f + (b >> 1);
                let c = e - tmp;
                let a = tmp + (c >> 1);
                (a, b, c)
            } else {
                let a = d;
                let b = if (ty >> 1) == 1 {
                    e - a
                } else if (ty >> 1) == 2 {
                    e - ((a + f) >> 1)
                } else {
                    e
                };
                let c = if (ty & 1) != 0 { f - a } else { f };
                (a, b, c)
            };
            a_slice[i] = a.0;
            b_slice[i] = b.0;
            c_slice[i] = c.0;
        }
    }

    /// Applies the inverse RCT in-place on `grids[begin_c .. begin_c + 3]`.
    pub fn inverse(&self, grids: &mut [ModularGrid]) {
        let begin = self.begin_c as usize;
        let permutation = self.rct_type / 7;
        let ty = self.rct_type % 7;
        let len = grids[begin].data.len();
        let (left, right) = grids.split_at_mut(begin + 1);
        let a_slice = &mut left[begin].data;
        let (mid, tail) = right.split_at_mut(1);
        let b_slice = &mut mid[0].data;
        let c_slice = &mut tail[0].data;

        for i in 0..len {
            let a = Wrapping(a_slice[i]);
            let b = Wrapping(b_slice[i]);
            let c = Wrapping(c_slice[i]);
            let (mut d, mut e, mut f) = if ty == 6 {
                let tmp = a - (c >> 1);
                let e = c + tmp;
                let f = tmp - (b >> 1);
                let d = f + b;
                (d.0, e.0, f.0)
            } else {
                let d = a;
                let f = if (ty & 1) != 0 { c + a } else { c };
                let e = if (ty >> 1) == 1 {
                    b + a
                } else if (ty >> 1) == 2 {
                    b + ((a + f) >> 1)
                } else {
                    b
                };
                (d.0, e.0, f.0)
            };
            match permutation {
                1 => {
                    std::mem::swap(&mut d, &mut e);
                    std::mem::swap(&mut d, &mut f);
                }
                2 => {
                    std::mem::swap(&mut d, &mut e);
                    std::mem::swap(&mut e, &mut f);
                }
                3 => std::mem::swap(&mut e, &mut f),
                4 => std::mem::swap(&mut d, &mut e),
                5 => std::mem::swap(&mut d, &mut f),
                _ => {}
            }
            a_slice[i] = d;
            b_slice[i] = e;
            c_slice[i] = f;
        }
    }
}

#[rustfmt::skip]
const DELTA_PALETTE: [[i16; 3]; 72] = [
    [0, 0, 0], [4, 4, 4], [11, 0, 0], [0, 0, -13], [0, -12, 0], [-10, -10, -10],
    [-18, -18, -18], [-27, -27, -27], [-18, -18, 0], [0, 0, -32], [-32, 0, 0], [-37, -37, -37],
    [0, -32, -32], [24, 24, 45], [50, 50, 50], [-45, -24, -24], [-24, -45, -45], [0, -24, -24],
    [-34, -34, 0], [-24, 0, -24], [-45, -45, -24], [64, 64, 64], [-32, 0, -32], [0, -32, 0],
    [-32, 0, 32], [-24, -45, -24], [45, 24, 45], [24, -24, -45], [-45, -24, 24], [80, 80, 80],
    [64, 0, 0], [0, 0, -64], [0, -64, -64], [-24, -24, 45], [96, 96, 96], [64, 64, 0],
    [45, -24, -24], [34, -34, 0], [112, 112, 112], [24, -45, -45], [45, 45, -24], [0, -32, 32],
    [24, -24, 45], [0, 96, 96], [45, -24, 24], [24, -45, -24], [-24, -45, 24], [0, -64, 0],
    [96, 0, 0], [128, 128, 128], [64, 0, 64], [144, 144, 144], [96, 96, 0], [-36, -36, 36],
    [45, -24, -45], [45, -45, -24], [0, 0, -96], [0, 128, 128], [0, 96, 0], [45, 24, -45],
    [-128, 0, 0], [24, -45, 24], [-45, 24, -45], [64, 0, -64], [64, -64, -64], [96, 0, 96],
    [45, -45, 24], [24, 45, -45], [64, 64, -64], [128, 128, 0], [0, 0, -128], [-24, 45, -45],
];

/// Palette transform parameters (ISO/IEC 18181-1 Annex H.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    /// First channel index covered by the palette.
    pub begin_c: u32,
    /// Number of channels combined into the palette index channel.
    pub num_c: u32,
    /// Number of explicit palette entries.
    pub nb_colours: u32,
    /// Number of delta palette entries.
    pub nb_deltas: u32,
    /// Predictor used for delta palette entries.
    pub d_pred: Predictor,
    /// Optional weighted predictor parameters when `d_pred == Predictor::SelfCorrecting`.
    pub wp_header: Option<WpHeader>,
}

impl Palette {
    /// Parses a `Palette` transform descriptor.
    pub fn parse(br: &mut BitReader<'_>, wp_header: &WpHeader) -> Result<Self, JxlError> {
        let begin_c = br.read_u32(
            U32Specifier::u(0, 3),
            U32Specifier::u(8, 6),
            U32Specifier::u(72, 10),
            U32Specifier::u(1096, 13),
        )?;
        let num_c = br.read_u32(
            U32Specifier::c(1),
            U32Specifier::c(3),
            U32Specifier::c(4),
            U32Specifier::u(1, 13),
        )?;
        let nb_colours = br.read_u32(
            U32Specifier::u(0, 8),
            U32Specifier::u(256, 10),
            U32Specifier::u(1280, 12),
            U32Specifier::u(5376, 16),
        )?;
        let nb_deltas = br.read_u32(
            U32Specifier::c(0),
            U32Specifier::u(1, 8),
            U32Specifier::u(257, 10),
            U32Specifier::u(1281, 16),
        )?;
        let d_pred = Predictor::from_u32(br.read_bits(4)?)?;
        Ok(Self {
            begin_c,
            num_c,
            nb_colours,
            nb_deltas,
            d_pred,
            wp_header: (d_pred == Predictor::SelfCorrecting).then(|| wp_header.clone()),
        })
    }

    /// Writes a `Palette` transform descriptor to `bw`.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_u32(
            self.begin_c,
            U32Specifier::u(0, 3),
            U32Specifier::u(8, 6),
            U32Specifier::u(72, 10),
            U32Specifier::u(1096, 13),
        )?;
        bw.write_u32(
            self.num_c,
            U32Specifier::c(1),
            U32Specifier::c(3),
            U32Specifier::c(4),
            U32Specifier::u(1, 13),
        )?;
        bw.write_u32(
            self.nb_colours,
            U32Specifier::u(0, 8),
            U32Specifier::u(256, 10),
            U32Specifier::u(1280, 12),
            U32Specifier::u(5376, 16),
        )?;
        bw.write_u32(
            self.nb_deltas,
            U32Specifier::c(0),
            U32Specifier::u(1, 8),
            U32Specifier::u(257, 10),
            U32Specifier::u(1281, 16),
        )?;
        bw.write_bits(4, self.d_pred as u32);
        Ok(())
    }

    /// Attempts to apply a lossless forward `Palette` transform on `grids[0..num_c]`
    /// if the number of unique colour tuples is `<= 256` and `< (width * height) / 2`.
    pub fn try_forward(grids: &mut Vec<ModularGrid>) -> Option<Self> {
        let num_c = match grids.len() {
            1 => 1usize,
            3 => 3usize,
            4 => 4usize,
            _ => return None,
        };
        let w = grids[0].width;
        let h = grids[0].height;
        if w * h < 4 {
            return None;
        }
        for g in &grids[1..num_c] {
            if g.width != w || g.height != h {
                return None;
            }
        }
        let max_colors = 256usize.min((w * h) / 2);
        if max_colors == 0 {
            return None;
        }
        let mut unique: Vec<[i32; 4]> = Vec::new();
        for i in 0..(w * h) {
            let mut px = [0i32; 4];
            for (dst, g) in px.iter_mut().zip(grids.iter().take(num_c)) {
                *dst = g.data[i];
            }
            if !unique.contains(&px) {
                if unique.len() >= max_colors {
                    return None;
                }
                unique.push(px);
            }
        }
        unique.sort_unstable();
        let nb_colours = unique.len();
        let mut palette_grid = ModularGrid::new(nb_colours, num_c);
        for (idx, px) in unique.iter().enumerate() {
            for (c, &val) in px.iter().enumerate().take(num_c) {
                palette_grid.set(idx, c, val);
            }
        }
        let mut index_grid = ModularGrid::new(w, h);
        for i in 0..(w * h) {
            let mut px = [0i32; 4];
            for (dst, g) in px.iter_mut().zip(grids.iter().take(num_c)) {
                *dst = g.data[i];
            }
            let idx = unique.binary_search(&px).unwrap();
            index_grid.data[i] = idx as i32;
        }
        grids.drain(0..num_c);
        grids.insert(0, index_grid);
        grids.insert(0, palette_grid);
        Some(Self {
            begin_c: 0,
            num_c: num_c as u32,
            nb_colours: nb_colours as u32,
            nb_deltas: 0,
            d_pred: Predictor::Zero,
            wp_header: None,
        })
    }

    /// Updates channel metadata for this `Palette` transform.
    pub fn prepare_channels(&self, channels: &mut ModularChannels) -> Result<(), JxlError> {
        let begin = self.begin_c;
        let end = begin
            .checked_add(self.num_c)
            .ok_or(JxlError::InvalidBitstream("Palette num_c overflow"))?;
        if end as usize > channels.info.len() || self.num_c == 0 {
            return Err(JxlError::InvalidBitstream(
                "Palette channel range out of bounds",
            ));
        }
        if begin < channels.nb_meta_channels {
            if end > channels.nb_meta_channels {
                return Err(JxlError::InvalidBitstream(
                    "Palette crosses meta-channel boundary",
                ));
            }
            channels.nb_meta_channels = channels.nb_meta_channels + 2 - self.num_c;
        } else {
            channels.nb_meta_channels += 1;
        }

        let w = channels.info[begin as usize].width;
        let h = channels.info[begin as usize].height;
        for ch in &channels.info[(begin as usize + 1)..(end as usize)] {
            if ch.width != w || ch.height != h {
                return Err(JxlError::InvalidBitstream(
                    "Palette channels have mismatched dimensions",
                ));
            }
        }

        channels.info.drain((begin as usize + 1)..(end as usize));
        channels.info.insert(
            0,
            ModularChannelInfo::new_unshiftable(self.nb_colours, self.num_c),
        );
        Ok(())
    }

    /// Applies the inverse `Palette` transform to `grids`.
    pub fn inverse(&self, grids: &mut Vec<ModularGrid>, bit_depth: u32) {
        let begin = self.begin_c as usize;
        let num_c = self.num_c as usize;
        let palette_grid = grids.remove(0);
        let index_grid = &grids[begin];
        let width = index_grid.width;
        let height = index_grid.height;
        let nb_colors = self.nb_colours as i32;
        let nb_deltas = self.nb_deltas as i32;

        let mut targets = vec![ModularGrid::new(width, height); num_c];
        let mut need_delta = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let index = index_grid.get(x, y);
                if index < nb_deltas {
                    need_delta.push((x, y));
                }
                if (0..nb_colors).contains(&index) {
                    for (c, target) in targets.iter_mut().enumerate() {
                        target.set(x, y, palette_grid.get(index as usize, c));
                    }
                } else if index >= nb_colors {
                    let idx = index - nb_colors;
                    if idx < 64 {
                        for (c, target) in targets.iter_mut().enumerate() {
                            let val = ((idx >> (2 * c)) % 4) * ((1i32 << bit_depth.min(24)) - 1)
                                / 4
                                + (1i32 << bit_depth.min(24).saturating_sub(3));
                            target.set(x, y, val);
                        }
                    } else {
                        let mut rem = idx - 64;
                        for target in &mut targets {
                            let val = (rem % 5) * ((1i32 << bit_depth.min(24)) - 1) / 4;
                            target.set(x, y, val);
                            rem /= 5;
                        }
                    }
                } else {
                    for (c, target) in targets.iter_mut().enumerate() {
                        if c >= 3 {
                            target.set(x, y, 0);
                            continue;
                        }
                        let idx = (-(index + 1) % 143) as usize;
                        let mut temp = DELTA_PALETTE[(idx + 1) >> 1][c] as i32;
                        if idx & 1 == 0 {
                            temp = -temp;
                        }
                        if bit_depth > 8 {
                            temp <<= bit_depth.min(24) - 8;
                        }
                        target.set(x, y, temp);
                    }
                }
            }
        }

        if !need_delta.is_empty() {
            let wp = if self.d_pred == Predictor::SelfCorrecting {
                self.wp_header.as_ref()
            } else {
                None
            };
            for target in &mut targets {
                let mut pred_state = PredictorState::new(width as u32, Vec::new(), wp);
                let mut delta_idx = 0usize;
                'row_loop: for y in 0..height {
                    for x in 0..width {
                        let sc = pred_state.step_sc();
                        let mut val = target.get(x, y);
                        if delta_idx < need_delta.len() && need_delta[delta_idx] == (x, y) {
                            let diff = self.d_pred.predict(&pred_state, sc.as_ref());
                            val = val.wrapping_add(diff);
                            target.set(x, y, val);
                            delta_idx += 1;
                            if delta_idx >= need_delta.len() {
                                break 'row_loop;
                            }
                        }
                        pred_state.record(val, sc.as_ref());
                    }
                }
            }
        }

        let mut it = targets.into_iter();
        grids[begin] = it.next().unwrap();
        for (i, g) in it.enumerate() {
            grids.insert(begin + 1 + i, g);
        }
    }
}

/// Parameters for a single Squeeze step (ISO/IEC 18181-1 Annex H.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqueezeParams {
    /// True for horizontal squeeze, false for vertical squeeze.
    pub horizontal: bool,
    /// True if residual channels are placed immediately after `begin_c + num_c`.
    pub in_place: bool,
    /// First channel index to squeeze.
    pub begin_c: u32,
    /// Number of consecutive channels to squeeze.
    pub num_c: u32,
}

impl SqueezeParams {
    /// Parses `SqueezeParams` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        Ok(Self {
            horizontal: br.read_bool()?,
            in_place: br.read_bool()?,
            begin_c: br.read_u32(
                U32Specifier::u(0, 3),
                U32Specifier::u(8, 6),
                U32Specifier::u(72, 10),
                U32Specifier::u(1096, 13),
            )?,
            num_c: br.read_u32(
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(3),
                U32Specifier::u(4, 4),
            )?,
        })
    }
}

/// Squeeze haar-like transform parameters (ISO/IEC 18181-1 Annex H.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Squeeze {
    /// Sequence of squeeze steps.
    pub sp: Vec<SqueezeParams>,
}

impl Squeeze {
    /// Parses a `Squeeze` transform descriptor.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let num_sq = br.read_u32(
            U32Specifier::c(0),
            U32Specifier::u(1, 4),
            U32Specifier::u(9, 6),
            U32Specifier::u(41, 8),
        )?;
        let mut sp = Vec::with_capacity(num_sq as usize);
        for _ in 0..num_sq {
            sp.push(SqueezeParams::parse(br)?);
        }
        Ok(Self { sp })
    }

    /// Populates default `SqueezeParams` when `sp` is empty (ISO/IEC 18181-1 Annex H.4.2).
    pub fn set_default_params(&mut self, channels: &ModularChannels) {
        if !self.sp.is_empty() {
            return;
        }
        let first = channels.nb_meta_channels;
        if first as usize >= channels.info.len() {
            return;
        }
        let first_ch = &channels.info[first as usize];
        let mut w = first_ch.width;
        let mut h = first_ch.height;
        if channels.info.len() as u32 - first >= 3 {
            let next_ch = &channels.info[first as usize + 1];
            if next_ch.width == w && next_ch.height == h {
                let param_base = SqueezeParams {
                    begin_c: first + 1,
                    num_c: 2,
                    in_place: false,
                    horizontal: false,
                };
                self.sp.push(SqueezeParams {
                    horizontal: true,
                    ..param_base
                });
                self.sp.push(param_base);
            }
        }

        let param_base = SqueezeParams {
            begin_c: first,
            num_c: channels.info.len() as u32 - first,
            in_place: true,
            horizontal: false,
        };

        if h >= w && h > 8 {
            self.sp.push(SqueezeParams {
                horizontal: false,
                ..param_base
            });
            h = h.div_ceil(2);
        }
        while w > 8 || h > 8 {
            if w > 8 {
                self.sp.push(SqueezeParams {
                    horizontal: true,
                    ..param_base
                });
                w = w.div_ceil(2);
            }
            if h > 8 {
                self.sp.push(SqueezeParams {
                    horizontal: false,
                    ..param_base
                });
                h = h.div_ceil(2);
            }
        }
    }

    /// Updates channel metadata for all squeeze steps.
    pub fn prepare_channels(&mut self, channels: &mut ModularChannels) -> Result<(), JxlError> {
        self.set_default_params(channels);
        for sp in &self.sp {
            let begin = sp.begin_c;
            let end = begin
                .checked_add(sp.num_c)
                .ok_or(JxlError::InvalidBitstream("Squeeze num_c overflow"))?;
            if end as usize > channels.info.len() {
                return Err(JxlError::InvalidBitstream(
                    "Squeeze channel range out of bounds",
                ));
            }
            if begin < channels.nb_meta_channels {
                if !sp.in_place || end > channels.nb_meta_channels {
                    return Err(JxlError::InvalidBitstream(
                        "Invalid Squeeze on meta-channels",
                    ));
                }
                channels.nb_meta_channels += sp.num_c;
            }

            let mut residu_channels = Vec::with_capacity(sp.num_c as usize);
            for ch in &mut channels.info[(begin as usize)..(end as usize)] {
                if ch.width == 0 || ch.height == 0 || ch.hshift > 30 || ch.vshift > 30 {
                    return Err(JxlError::InvalidBitstream(
                        "Invalid channel dimensions or shift in Squeeze",
                    ));
                }
                let mut residu = ch.clone();
                if sp.horizontal {
                    let len = ch.width;
                    ch.width = len.div_ceil(2);
                    residu.width = len / 2;
                    if ch.hshift >= 0 {
                        ch.hshift += 1;
                        residu.hshift += 1;
                    }
                } else {
                    let len = ch.height;
                    ch.height = len.div_ceil(2);
                    residu.height = len / 2;
                    if ch.vshift >= 0 {
                        ch.vshift += 1;
                        residu.vshift += 1;
                    }
                }
                residu_channels.push(residu);
            }

            if sp.in_place {
                residu_channels.extend(channels.info.drain((end as usize)..));
            }
            channels.info.extend(residu_channels);
        }
        Ok(())
    }

    /// Writes a `Squeeze` transform descriptor to `bw`.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_u32(
            self.sp.len() as u32,
            U32Specifier::c(0),
            U32Specifier::u(1, 4),
            U32Specifier::u(9, 6),
            U32Specifier::u(41, 8),
        )?;
        for sp in &self.sp {
            bw.write_bool(sp.horizontal);
            bw.write_bool(sp.in_place);
            bw.write_u32(
                sp.begin_c,
                U32Specifier::u(0, 3),
                U32Specifier::u(8, 6),
                U32Specifier::u(72, 10),
                U32Specifier::u(1096, 13),
            )?;
            bw.write_u32(
                sp.num_c,
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(3),
                U32Specifier::u(4, 4),
            )?;
        }
        Ok(())
    }

    /// Applies the forward Squeeze steps in order on `grids`.
    pub fn forward(&self, grids: &mut Vec<ModularGrid>) {
        for sp in &self.sp {
            let begin = sp.begin_c as usize;
            let count = sp.num_c as usize;
            let end = begin + count;
            let mut residuals = Vec::with_capacity(count);
            for grid in &mut grids[begin..end] {
                let (avg, res) = if sp.horizontal {
                    forward_squeeze_h(grid)
                } else {
                    forward_squeeze_v(grid)
                };
                *grid = avg;
                residuals.push(res);
            }
            if sp.in_place {
                let tail: Vec<ModularGrid> = grids.drain(end..).collect();
                grids.extend(residuals);
                grids.extend(tail);
            } else {
                grids.extend(residuals);
            }
        }
    }

    /// Applies the inverse Squeeze steps in reverse order on `grids`.
    pub fn inverse(&self, grids: &mut Vec<ModularGrid>) {
        for sp in self.sp.iter().rev() {
            let begin = sp.begin_c as usize;
            let count = sp.num_c as usize;
            let end = begin + count;
            let residuals: Vec<ModularGrid> = if sp.in_place {
                grids.drain(end..(end + count)).collect()
            } else {
                grids.drain((grids.len() - count)..).collect()
            };
            for (avg_grid, res_grid) in grids[begin..end].iter_mut().zip(residuals) {
                if sp.horizontal {
                    *avg_grid = inverse_squeeze_h(avg_grid, &res_grid);
                } else {
                    *avg_grid = inverse_squeeze_v(avg_grid, &res_grid);
                }
            }
        }
    }
}

#[inline]
fn tendency_i32(a: i32, b: i32, c: i32) -> i32 {
    let a = Wrapping(a);
    let b = Wrapping(b);
    let c = Wrapping(c);

    let n1 = Wrapping(1);
    let n2 = Wrapping(2);
    let n3 = Wrapping(3);
    let n4 = Wrapping(4);
    let n6 = Wrapping(6);
    let n12 = Wrapping(12);

    if a >= b && b >= c {
        let mut x = (n4 * a - n3 * c - b + n6) / n12;
        if x - (x & n1) > n2 * (a - b) {
            x = n2 * (a - b) + n1;
        }
        if x + (x & n1) > n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else if a <= b && b <= c {
        let mut x = (n4 * a - n3 * c - b - n6) / n12;
        if x + (x & n1) < n2 * (a - b) {
            x = n2 * (a - b) - n1;
        }
        if x - (x & n1) < n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else {
        0
    }
}

fn inverse_squeeze_h(avg_grid: &ModularGrid, res_grid: &ModularGrid) -> ModularGrid {
    let avg_w = avg_grid.width;
    let res_w = res_grid.width;
    let width = avg_w + res_w;
    let height = avg_grid.height;
    let mut out = ModularGrid::new(width, height);
    if width == 0 || height == 0 {
        return out;
    }

    for y in 0..height {
        let avg_row = avg_grid.row(y);
        let res_row = res_grid.row(y);
        let out_row = out.row_mut(y);

        let mut avg = avg_row[0];
        let mut left = avg;
        for x in 0..res_w {
            let residu = res_row[x];
            let next_avg = avg_row.get(x + 1).copied().unwrap_or(avg);
            let diff = residu.wrapping_add(tendency_i32(left, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            out_row[2 * x] = first;
            out_row[2 * x + 1] = second;
            avg = next_avg;
            left = second;
        }
        if width % 2 == 1 {
            out_row[width - 1] = avg_row[avg_w - 1];
        }
    }
    out
}

fn inverse_squeeze_v(avg_grid: &ModularGrid, res_grid: &ModularGrid) -> ModularGrid {
    let width = avg_grid.width;
    let avg_h = avg_grid.height;
    let res_h = res_grid.height;
    let height = avg_h + res_h;
    let mut out = ModularGrid::new(width, height);
    if width == 0 || height == 0 {
        return out;
    }

    for x in 0..width {
        let mut avg = avg_grid.get(x, 0);
        let mut top = avg;
        for y in 0..res_h {
            let residu = res_grid.get(x, y);
            let next_avg = if y + 1 < avg_h {
                avg_grid.get(x, y + 1)
            } else {
                avg
            };
            let diff = residu.wrapping_add(tendency_i32(top, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            out.set(x, 2 * y, first);
            out.set(x, 2 * y + 1, second);
            avg = next_avg;
            top = second;
        }
        if height % 2 == 1 {
            out.set(x, height - 1, avg_grid.get(x, avg_h - 1));
        }
    }
    out
}

fn forward_squeeze_h(src: &ModularGrid) -> (ModularGrid, ModularGrid) {
    let width = src.width;
    let height = src.height;
    let avg_w = width.div_ceil(2);
    let res_w = width / 2;
    let mut avg_grid = ModularGrid::new(avg_w, height);
    let mut res_grid = ModularGrid::new(res_w, height);
    if width == 0 || height == 0 {
        return (avg_grid, res_grid);
    }

    for y in 0..height {
        let src_row = src.row(y);
        let avg_row = avg_grid.row_mut(y);
        let res_row = res_grid.row_mut(y);

        for x in 0..res_w {
            let first = src_row[2 * x];
            let second = src_row[2 * x + 1];
            let diff = first.wrapping_sub(second);
            avg_row[x] = first.wrapping_sub(diff / 2);
        }
        if width % 2 == 1 {
            avg_row[avg_w - 1] = src_row[width - 1];
        }

        let mut avg = avg_row[0];
        let mut left = avg;
        for x in 0..res_w {
            let first = src_row[2 * x];
            let second = src_row[2 * x + 1];
            let diff = first.wrapping_sub(second);
            let next_avg = avg_row.get(x + 1).copied().unwrap_or(avg);
            let tend = tendency_i32(left, avg, next_avg);
            res_row[x] = diff.wrapping_sub(tend);
            avg = next_avg;
            left = second;
        }
    }
    (avg_grid, res_grid)
}

fn forward_squeeze_v(src: &ModularGrid) -> (ModularGrid, ModularGrid) {
    let width = src.width;
    let height = src.height;
    let avg_h = height.div_ceil(2);
    let res_h = height / 2;
    let mut avg_grid = ModularGrid::new(width, avg_h);
    let mut res_grid = ModularGrid::new(width, res_h);
    if width == 0 || height == 0 {
        return (avg_grid, res_grid);
    }

    for x in 0..width {
        for y in 0..res_h {
            let first = src.get(x, 2 * y);
            let second = src.get(x, 2 * y + 1);
            let diff = first.wrapping_sub(second);
            avg_grid.set(x, y, first.wrapping_sub(diff / 2));
        }
        if height % 2 == 1 {
            avg_grid.set(x, avg_h - 1, src.get(x, height - 1));
        }

        let mut avg = avg_grid.get(x, 0);
        let mut top = avg;
        for y in 0..res_h {
            let first = src.get(x, 2 * y);
            let second = src.get(x, 2 * y + 1);
            let diff = first.wrapping_sub(second);
            let next_avg = if y + 1 < avg_h {
                avg_grid.get(x, y + 1)
            } else {
                avg
            };
            let tend = tendency_i32(top, avg, next_avg);
            res_grid.set(x, y, diff.wrapping_sub(tend));
            avg = next_avg;
            top = second;
        }
    }
    (avg_grid, res_grid)
}

/// Reversible Modular transform step (ISO/IEC 18181-1 Annex H).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformInfo {
    /// Reversible colour transform (`tr == 0`).
    Rct(Rct),
    /// Channel / pixel palette transform (`tr == 1`).
    Palette(Palette),
    /// Haar-like multi-resolution squeeze transform (`tr == 2`).
    Squeeze(Squeeze),
}

impl TransformInfo {
    /// Parses a `TransformInfo` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>, wp_header: &WpHeader) -> Result<Self, JxlError> {
        let tr = br.read_bits(2)?;
        match tr {
            0 => Ok(Self::Rct(Rct::parse(br)?)),
            1 => Ok(Self::Palette(Palette::parse(br, wp_header)?)),
            2 => Ok(Self::Squeeze(Squeeze::parse(br)?)),
            _ => Err(JxlError::InvalidBitstream("invalid TransformId")),
        }
    }

    /// Updates channel metadata for this transform.
    pub fn prepare_channels(&mut self, channels: &mut ModularChannels) -> Result<(), JxlError> {
        match self {
            Self::Rct(rct) => rct.prepare_channels(channels),
            Self::Palette(pal) => pal.prepare_channels(channels),
            Self::Squeeze(sq) => sq.prepare_channels(channels),
        }
    }

    /// Applies the inverse transform to `grids`.
    pub fn inverse(&self, grids: &mut Vec<ModularGrid>, bit_depth: u32) {
        match self {
            Self::Rct(rct) => rct.inverse(grids),
            Self::Palette(pal) => pal.inverse(grids, bit_depth),
            Self::Squeeze(sq) => sq.inverse(grids),
        }
    }
}

/// Header of a Modular sub-bitstream (ISO/IEC 18181-1 Annex G.1).
#[derive(Debug, Clone)]
pub struct ModularHeader {
    /// Whether this stream reuses the frame's global `MaConfig`.
    pub use_global_tree: bool,
    /// Weighted predictor parameters.
    pub wp_params: WpHeader,
    /// Ordered list of forward transforms applied before encoding.
    pub transform: Vec<TransformInfo>,
}

impl ModularHeader {
    /// Parses a `ModularHeader` from the bitstream.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let use_global_tree = br.read_bool()?;
        let wp_params = WpHeader::parse(br)?;
        let nb_transforms = br.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::u(2, 4),
            U32Specifier::u(18, 8),
        )?;
        if nb_transforms > 512 {
            return Err(JxlError::LimitExceeded(
                "nb_transforms > 512 in ModularHeader",
            ));
        }
        let mut transform = Vec::with_capacity(nb_transforms as usize);
        for _ in 0..nb_transforms {
            transform.push(TransformInfo::parse(br, &wp_params)?);
        }
        Ok(Self {
            use_global_tree,
            wp_params,
            transform,
        })
    }

    /// Writes a `ModularHeader` to the bitstream.
    pub fn write(&self, bw: &mut BitWriter) -> Result<(), JxlError> {
        bw.write_bool(self.use_global_tree);
        self.wp_params.write(bw);
        bw.write_u32(
            self.transform.len() as u32,
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::u(2, 4),
            U32Specifier::u(18, 8),
        )?;
        for tr in &self.transform {
            match tr {
                TransformInfo::Rct(rct) => {
                    bw.write_bits(2, 0);
                    rct.write(bw)?;
                }
                TransformInfo::Palette(pal) => {
                    bw.write_bits(2, 1);
                    pal.write(bw)?;
                }
                TransformInfo::Squeeze(sq) => {
                    bw.write_bits(2, 2);
                    sq.write(bw)?;
                }
            }
        }
        Ok(())
    }
}

/// Decodes a sequence of transformed Modular channels using `ma_config` and `wp_header`.
pub fn decode_modular_channels(
    br: &mut BitReader<'_>,
    channel_info: &[ModularChannelInfo],
    ma_config: &MaConfig,
    wp_header: &WpHeader,
    stream_index: u32,
) -> Result<Vec<ModularGrid>, JxlError> {
    let dist_multiplier = channel_info
        .iter()
        .map(|info| info.width)
        .max()
        .unwrap_or(0);

    let mut decoder = ma_config.decoder.clone();
    decoder.begin(br)?;

    let mut grids = Vec::with_capacity(channel_info.len());
    let mut prev_map: HashMap<(u32, u32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, info) in channel_info.iter().enumerate() {
        let width = info.width as usize;
        let height = info.height as usize;
        let mut grid = ModularGrid::new(width, height);
        if width == 0 || height == 0 {
            grids.push(grid);
            continue;
        }

        let key = (info.width, info.height, info.hshift, info.vshift);
        let prev_indices = prev_map.entry(key).or_default().clone();
        let pruned_tree = ma_config
            .tree
            .prune(i as u32, stream_index, prev_indices.len() as u32);

        if let MaTreeNode::Leaf(ref leaf) = pruned_tree {
            decode_single_leaf_channel(
                br,
                &mut decoder,
                dist_multiplier,
                wp_header,
                &mut grid,
                leaf,
            )?;
        } else {
            let prev_refs: Vec<&ModularGrid> =
                prev_indices.iter().map(|&idx| &grids[idx]).collect();
            let need_sc = pruned_tree.needs_self_correcting();
            let mut pred_state =
                PredictorState::new(info.width, prev_refs, need_sc.then_some(wp_header));
            for y in 0..height {
                for x in 0..width {
                    let sc = pred_state.step_sc();
                    let leaf = pruned_tree.eval(&pred_state, i as u32, stream_index, sc.as_ref());
                    let tok = decoder.read_varint_with_multiplier_clustered(
                        br,
                        leaf.cluster,
                        dist_multiplier,
                    )?;
                    let diff = unpack_signed(tok)
                        .wrapping_mul(leaf.multiplier as i32)
                        .wrapping_add(leaf.offset);
                    let pred = leaf.predictor.predict(&pred_state, sc.as_ref());
                    let sample = diff.wrapping_add(pred);
                    grid.set(x, y, sample);
                    pred_state.record(sample, sc.as_ref());
                }
            }
        }

        prev_map.entry(key).or_default().insert(0, i);
        grids.push(grid);
    }

    decoder.finalize()?;
    Ok(grids)
}

fn decode_single_leaf_channel(
    br: &mut BitReader<'_>,
    decoder: &mut EntropyDecoder,
    dist_multiplier: u32,
    wp_header: &WpHeader,
    grid: &mut ModularGrid,
    leaf: &MaTreeLeaf,
) -> Result<(), JxlError> {
    let width = grid.width;
    let height = grid.height;

    if leaf.predictor == Predictor::Zero {
        if let Some(tok) = decoder.single_token(leaf.cluster) {
            let val = unpack_signed(tok)
                .wrapping_mul(leaf.multiplier as i32)
                .wrapping_add(leaf.offset);
            grid.data.fill(val);
            return Ok(());
        }
        for out in &mut grid.data {
            let tok =
                decoder.read_varint_with_multiplier_clustered(br, leaf.cluster, dist_multiplier)?;
            *out = unpack_signed(tok)
                .wrapping_mul(leaf.multiplier as i32)
                .wrapping_add(leaf.offset);
        }
        return Ok(());
    }

    if leaf.predictor == Predictor::Gradient && leaf.offset == 0 && leaf.multiplier == 1 {
        let mut w = 0i32;
        for x in 0..width {
            let tok =
                decoder.read_varint_with_multiplier_clustered(br, leaf.cluster, dist_multiplier)?;
            w = unpack_signed(tok).wrapping_add(w);
            grid.set(x, 0, w);
        }
        for y in 1..height {
            let tok =
                decoder.read_varint_with_multiplier_clustered(br, leaf.cluster, dist_multiplier)?;
            let n0 = grid.get(0, y - 1);
            w = unpack_signed(tok).wrapping_add(n0);
            grid.set(0, y, w);
            for x in 1..width {
                let nw = grid.get(x - 1, y - 1);
                let n = grid.get(x, y - 1);
                let pred = grad_clamped_i32(n, w, nw);
                let tok = decoder.read_varint_with_multiplier_clustered(
                    br,
                    leaf.cluster,
                    dist_multiplier,
                )?;
                w = unpack_signed(tok).wrapping_add(pred);
                grid.set(x, y, w);
            }
        }
        return Ok(());
    }

    let need_sc = leaf.predictor == Predictor::SelfCorrecting;
    let mut pred_state =
        PredictorState::new(width as u32, Vec::new(), need_sc.then_some(wp_header));
    for y in 0..height {
        for x in 0..width {
            let sc = pred_state.step_sc();
            let tok =
                decoder.read_varint_with_multiplier_clustered(br, leaf.cluster, dist_multiplier)?;
            let diff = unpack_signed(tok)
                .wrapping_mul(leaf.multiplier as i32)
                .wrapping_add(leaf.offset);
            let pred = leaf.predictor.predict(&pred_state, sc.as_ref());
            let sample = diff.wrapping_add(pred);
            grid.set(x, y, sample);
            pred_state.record(sample, sc.as_ref());
        }
    }
    Ok(())
}

/// Decodes a standalone Modular sub-image (used for `LfCoeff`, `HfMetadata`, `LfGroup` modular,
/// and `PassGroup` modular) and applies all inverse transforms.
pub fn decode_modular_subimage(
    br: &mut BitReader<'_>,
    initial_channels: Vec<ModularChannelInfo>,
    bit_depth: u32,
    global_ma_config: Option<&MaConfig>,
    stream_index: u32,
) -> Result<Vec<ModularGrid>, JxlError> {
    if initial_channels.is_empty() {
        return Ok(Vec::new());
    }
    let mut header = ModularHeader::parse(br)?;
    let mut tr_channels = ModularChannels::new(initial_channels);
    for tr in &mut header.transform {
        tr.prepare_channels(&mut tr_channels)?;
    }
    if tr_channels.info.len() > (1 << 16) {
        return Err(JxlError::LimitExceeded(
            "nb_channels_tr > 65536 in ModularSubimage",
        ));
    }

    let local_ma;
    let ma_ctx = if header.use_global_tree {
        global_ma_config.ok_or(JxlError::InvalidBitstream(
            "GlobalMaTree requested but not present",
        ))?
    } else {
        let local_samples = tr_channels
            .info
            .iter()
            .fold(0u64, |acc, ch| acc + ch.width as u64 * ch.height as u64);
        let node_limit = (1024 + local_samples).min(1 << 20) as usize;
        local_ma = MaConfig::parse(br, node_limit)?;
        &local_ma
    };

    let mut grids = decode_modular_channels(
        br,
        &tr_channels.info,
        ma_ctx,
        &header.wp_params,
        stream_index,
    )?;

    for tr in header.transform.iter().rev() {
        tr.inverse(&mut grids, bit_depth);
    }

    Ok(grids)
}

/// Encodes a standalone Modular sub-image into `bw` using a clean-room single-leaf MA tree.
///
/// If `use_rct_ycocg` is true and `grids.len() >= 3` with matching dimensions on the first
/// 3 channels, applies the reversible YCoCg (`rct_type = 6`) transform before encoding.
pub fn encode_modular_subimage(
    bw: &mut BitWriter,
    grids: &[ModularGrid],
    predictor: Predictor,
    use_rct_ycocg: bool,
) -> Result<(), JxlError> {
    if grids.is_empty() {
        return Ok(());
    }

    let mut work_grids = grids.to_vec();
    let mut transform = Vec::new();
    if use_rct_ycocg
        && work_grids.len() >= 3
        && work_grids[0].width == work_grids[1].width
        && work_grids[0].width == work_grids[2].width
        && work_grids[0].height == work_grids[1].height
        && work_grids[0].height == work_grids[2].height
    {
        let rct = Rct {
            begin_c: 0,
            rct_type: 6,
        };
        rct.forward(&mut work_grids);
        transform.push(TransformInfo::Rct(rct));
    }

    let header = ModularHeader {
        use_global_tree: false,
        wp_params: WpHeader::default(),
        transform,
    };
    header.write(bw)?;
    MaConfig::write_single_leaf_tree(bw, predictor)?;

    let mut enc = EntropyEncoder::single_cluster(1, HybridUintConfig::DEFAULT_4_2_0);
    for grid in &work_grids {
        let width = grid.width;
        let height = grid.height;
        if width == 0 || height == 0 {
            continue;
        }
        match predictor {
            Predictor::Zero => {
                for &sample in &grid.data {
                    enc.record_varint(0, pack_signed(sample));
                }
            }
            Predictor::Gradient => {
                let mut w = 0i32;
                for x in 0..width {
                    let s = grid.get(x, 0);
                    let diff = s.wrapping_sub(w);
                    enc.record_varint(0, pack_signed(diff));
                    w = s;
                }
                for y in 1..height {
                    let n0 = grid.get(0, y - 1);
                    let s0 = grid.get(0, y);
                    enc.record_varint(0, pack_signed(s0.wrapping_sub(n0)));
                    w = s0;
                    for x in 1..width {
                        let nw = grid.get(x - 1, y - 1);
                        let n = grid.get(x, y - 1);
                        let pred = grad_clamped_i32(n, w, nw);
                        let s = grid.get(x, y);
                        enc.record_varint(0, pack_signed(s.wrapping_sub(pred)));
                        w = s;
                    }
                }
            }
            _ => {
                let need_sc = predictor == Predictor::SelfCorrecting;
                let mut pred_state = PredictorState::new(
                    width as u32,
                    Vec::new(),
                    need_sc.then_some(&header.wp_params),
                );
                for y in 0..height {
                    for x in 0..width {
                        let sc = pred_state.step_sc();
                        let pred = predictor.predict(&pred_state, sc.as_ref());
                        let s = grid.get(x, y);
                        enc.record_varint(0, pack_signed(s.wrapping_sub(pred)));
                        pred_state.record(s, sc.as_ref());
                    }
                }
            }
        }
    }
    enc.finish(bw)
}

/// Configuration options for the v2 Modular sub-image encoder.
#[derive(Debug, Clone, Copy)]
pub struct ModularEncodeOptions {
    /// Spatial predictor mode (`Predictor::Gradient` or `Predictor::SelfCorrecting` recommended).
    pub predictor: Predictor,
    /// Apply reversible YCoCg (`rct_type = 6`) on 3+ channel sub-images when not palettized.
    pub use_rct_ycocg: bool,
    /// Automatically extract a lossless `Palette` transform if unique colors `<= 256`.
    pub enable_palette: bool,
    /// Apply a 1-level horizontal + vertical `Squeeze` transform on channels `>= 16x16`.
    pub enable_squeeze: bool,
    /// Emit a 4-context adaptive `MaConfig` tree instead of a 1-leaf tree.
    pub adaptive_tree: bool,
    /// Entropy coding mode (`Prefix` or `Ans`).
    pub entropy_mode: JxlEntropyMode,
    /// Enable 2D LZ77 backward-reference match finding.
    pub lz77: bool,
    /// Residual quantization step (`0` or `1` = lossless, `> 1` = near-lossless multiplier).
    pub residual_multiplier: u32,
}

impl Default for ModularEncodeOptions {
    fn default() -> Self {
        Self {
            predictor: Predictor::Gradient,
            use_rct_ycocg: true,
            enable_palette: false,
            enable_squeeze: false,
            adaptive_tree: false,
            entropy_mode: JxlEntropyMode::Prefix,
            lz77: false,
            residual_multiplier: 1,
        }
    }
}

/// Encodes a Modular sub-image using v2 transforms (`Palette`, `Rct`, `Squeeze`),
/// optional 4-cluster context-adaptive `MaConfig` tree, `SelfCorrecting` predictor,
/// and `Prefix` / `12-bit rANS` + `2D LZ77` entropy coding.
pub fn encode_modular_subimage_v2(
    bw: &mut BitWriter,
    grids: &[ModularGrid],
    opts: &ModularEncodeOptions,
) -> Result<(), JxlError> {
    if grids.is_empty() {
        return Ok(());
    }
    let mut work_grids = grids.to_vec();
    let mut transform = Vec::new();
    let mut has_palette_lead = false;

    if opts.enable_palette
        && let Some(pal) = Palette::try_forward(&mut work_grids)
    {
        transform.push(TransformInfo::Palette(pal));
        has_palette_lead = true;
    }

    if !has_palette_lead
        && opts.use_rct_ycocg
        && work_grids.len() >= 3
        && work_grids[0].width == work_grids[1].width
        && work_grids[0].width == work_grids[2].width
        && work_grids[0].height == work_grids[1].height
        && work_grids[0].height == work_grids[2].height
    {
        let rct = Rct {
            begin_c: 0,
            rct_type: 6,
        };
        rct.forward(&mut work_grids);
        transform.push(TransformInfo::Rct(rct));
    }

    if !has_palette_lead
        && opts.enable_squeeze
        && work_grids.iter().all(|g| g.width >= 16 && g.height >= 16)
    {
        let num_c = work_grids.len() as u32;
        let sq = Squeeze {
            sp: vec![
                SqueezeParams {
                    horizontal: true,
                    in_place: true,
                    begin_c: 0,
                    num_c,
                },
                SqueezeParams {
                    horizontal: false,
                    in_place: true,
                    begin_c: 0,
                    num_c,
                },
            ],
        };
        sq.forward(&mut work_grids);
        transform.push(TransformInfo::Squeeze(sq));
    }

    let wp_params = WpHeader::default();
    let header = ModularHeader {
        use_global_tree: false,
        wp_params: wp_params.clone(),
        transform,
    };
    header.write(bw)?;

    let dist_multiplier = work_grids.iter().map(|g| g.width as u32).max().unwrap_or(0);
    let mul = opts.residual_multiplier.max(1);

    if opts.adaptive_tree || has_palette_lead || mul > 1 {
        let tree = MaConfig::write_adaptive_tree(bw, opts.predictor, has_palette_lead, mul)?;
        let mut enc = EntropyEncoder::with_options(
            vec![0, 1, 2, 3],
            vec![HybridUintConfig::DEFAULT_4_2_0; 4],
            opts.entropy_mode,
            opts.lz77,
        );
        let need_sc = tree.needs_self_correcting();
        for (ch_idx, grid) in work_grids.iter_mut().enumerate() {
            let width = grid.width;
            let height = grid.height;
            if width == 0 || height == 0 {
                continue;
            }
            let mut pred_state =
                PredictorState::new(width as u32, Vec::new(), need_sc.then_some(&wp_params));
            for y in 0..height {
                for x in 0..width {
                    let sc = pred_state.step_sc();
                    let leaf = tree.eval(&pred_state, ch_idx as u32, 0, sc.as_ref());
                    let pred = leaf.predictor.predict(&pred_state, sc.as_ref());
                    let orig = grid.get(x, y);
                    let raw_diff = orig.wrapping_sub(pred);
                    let q_diff = if mul > 1 {
                        let m = mul as i32;
                        if raw_diff >= 0 {
                            (raw_diff + m / 2) / m
                        } else {
                            (raw_diff - m / 2) / m
                        }
                    } else {
                        raw_diff
                    };
                    let recon = pred.wrapping_add(q_diff.wrapping_mul(mul as i32));
                    grid.set(x, y, recon);
                    enc.record_varint_with_multiplier(
                        u32::from(leaf.cluster),
                        pack_signed(q_diff),
                        dist_multiplier,
                    );
                    pred_state.record(recon, sc.as_ref());
                }
            }
        }
        enc.finish(bw)
    } else {
        MaConfig::write_single_leaf_tree(bw, opts.predictor)?;
        let mut enc = EntropyEncoder::single_cluster_with_options(
            1,
            HybridUintConfig::DEFAULT_4_2_0,
            opts.entropy_mode,
            opts.lz77,
        );
        let need_sc = opts.predictor == Predictor::SelfCorrecting;
        for grid in &work_grids {
            let width = grid.width;
            let height = grid.height;
            if width == 0 || height == 0 {
                continue;
            }
            let mut pred_state =
                PredictorState::new(width as u32, Vec::new(), need_sc.then_some(&wp_params));
            for y in 0..height {
                for x in 0..width {
                    let sc = pred_state.step_sc();
                    let pred = opts.predictor.predict(&pred_state, sc.as_ref());
                    let s = grid.get(x, y);
                    enc.record_varint_with_multiplier(
                        0,
                        pack_signed(s.wrapping_sub(pred)),
                        dist_multiplier,
                    );
                    pred_state.record(s, sc.as_ref());
                }
            }
        }
        enc.finish(bw)
    }
}

/// Placement of a tile belonging to a global Modular channel inside an `LfGroup` or `PassGroup`.
#[derive(Debug, Clone)]
pub struct ModularGroupTile {
    /// Index of the channel in `GlobalModularState::tr_channels`.
    pub tr_channel_idx: usize,
    /// Destination X offset in `tr_grids[tr_channel_idx]`.
    pub dst_x: usize,
    /// Destination Y offset in `tr_grids[tr_channel_idx]`.
    pub dst_y: usize,
    /// Channel descriptor for this tile.
    pub tile_info: ModularChannelInfo,
}

/// State of a frame-level `GlobalModular` image across `LfGlobal`, `LfGroup`, and `PassGroup`.
#[derive(Debug, Clone)]
pub struct GlobalModularState {
    /// Optional global `MaConfig` parsed in `LfGlobal`.
    pub global_ma_config: Option<MaConfig>,
    /// Global Modular header.
    pub header: Option<ModularHeader>,
    /// Active MA configuration for the global Modular image.
    pub ma_ctx: Option<MaConfig>,
    /// Bit depth of the frame.
    pub bit_depth: u32,
    /// Index of the first extra channel in the initial channel list.
    pub extra_channel_from: usize,
    /// Transformed channel descriptors.
    pub tr_channels: Vec<ModularChannelInfo>,
    /// Transformed channel sample grids.
    pub tr_grids: Vec<ModularGrid>,
    /// Number of channels decoded directly in `LfGlobal`.
    pub gmodular_count: usize,
    /// Per-LF-group modular tiles.
    pub lf_group_tiles: Vec<Vec<ModularGroupTile>>,
    /// Per-pass, per-group modular tiles (`pass_group_tiles[pass_idx][group_idx]`).
    pub pass_group_tiles: Vec<Vec<Vec<ModularGroupTile>>>,
}

impl GlobalModularState {
    /// Parses `GlobalModular` from the `LfGlobal` section of a frame and prepares group tile maps.
    #[allow(clippy::too_many_arguments)]
    pub fn parse(
        br: &mut BitReader<'_>,
        initial_channels: Vec<ModularChannelInfo>,
        extra_channel_from: usize,
        group_dim: u32,
        bit_depth: u32,
        frame_width: u32,
        frame_height: u32,
        pass_shifts: &BTreeMap<u32, (i32, i32)>,
    ) -> Result<Self, JxlError> {
        let num_channels = initial_channels.len() as u64;
        let max_global_nodes = (1024 + frame_width as u64 * frame_height as u64 * num_channels / 16)
            .min(1 << 22) as usize;
        let global_ma_config = if br.read_bool()? {
            Some(MaConfig::parse(br, max_global_nodes)?)
        } else {
            None
        };

        if initial_channels.is_empty() {
            return Ok(Self {
                global_ma_config,
                header: None,
                ma_ctx: None,
                bit_depth,
                extra_channel_from,
                tr_channels: Vec::new(),
                tr_grids: Vec::new(),
                gmodular_count: 0,
                lf_group_tiles: Vec::new(),
                pass_group_tiles: Vec::new(),
            });
        }

        let mut header = ModularHeader::parse(br)?;
        let mut tr_channels = ModularChannels::new(initial_channels);
        for tr in &mut header.transform {
            tr.prepare_channels(&mut tr_channels)?;
        }
        if tr_channels.info.len() > (1 << 16) {
            return Err(JxlError::LimitExceeded(
                "nb_channels_tr > 65536 in GlobalModular",
            ));
        }

        let ma_ctx = if header.use_global_tree {
            global_ma_config.clone().ok_or(JxlError::InvalidBitstream(
                "GlobalModular requested global tree when none was present",
            ))?
        } else {
            let local_samples = tr_channels
                .info
                .iter()
                .fold(0u64, |acc, ch| acc + ch.width as u64 * ch.height as u64);
            let node_limit = (1024 + local_samples).min(1 << 20) as usize;
            MaConfig::parse(br, node_limit)?
        };

        let nb_meta = tr_channels.nb_meta_channels as usize;
        let gmodular_count = tr_channels
            .info
            .iter()
            .enumerate()
            .take_while(|&(i, info)| {
                i < nb_meta || (info.width <= group_dim && info.height <= group_dim)
            })
            .count();

        let gmodular_grids = decode_modular_channels(
            br,
            &tr_channels.info[..gmodular_count],
            &ma_ctx,
            &header.wp_params,
            0,
        )?;

        let mut tr_grids = Vec::with_capacity(tr_channels.info.len());
        tr_grids.extend(gmodular_grids);
        for info in &tr_channels.info[gmodular_count..] {
            tr_grids.push(ModularGrid::new(info.width as usize, info.height as usize));
        }

        let num_passes = pass_shifts
            .last_key_value()
            .map(|(&k, _)| k as usize + 1)
            .unwrap_or(1);
        let group_dim_shift = group_dim.trailing_zeros();
        let lf_dim = group_dim << 3;
        let num_lf_cols = frame_width.div_ceil(lf_dim) as usize;
        let num_lf_rows = frame_height.div_ceil(lf_dim) as usize;
        let num_lf_groups = num_lf_cols * num_lf_rows;
        let num_cols = frame_width.div_ceil(group_dim) as usize;
        let num_rows = frame_height.div_ceil(group_dim) as usize;
        let num_groups = num_cols * num_rows;

        let mut lf_group_tiles = vec![Vec::new(); num_lf_groups];
        let mut pass_group_tiles = vec![vec![Vec::new(); num_groups]; num_passes];

        for (i, info) in tr_channels.info.iter().enumerate().skip(gmodular_count) {
            let hshift = info.hshift;
            let vshift = info.vshift;
            if hshift < 0 || vshift < 0 {
                return Err(JxlError::InvalidBitstream(
                    "Negative shift in non-gmodular channel",
                ));
            }

            if hshift < 3 || vshift < 3 {
                let shift = hshift.min(vshift);
                let pass_idx = pass_shifts
                    .iter()
                    .find(|&(_, &(minshift, maxshift))| (minshift..maxshift).contains(&shift))
                    .map(|(&k, _)| k as usize)
                    .unwrap_or(0);
                let gw = (group_dim >> hshift) as usize;
                let gh = (group_dim >> vshift) as usize;
                if gw == 0 || gh == 0 {
                    return Err(JxlError::InvalidBitstream(
                        "Pass group dimension became 0 after shift",
                    ));
                }
                let ch_cols = ((info.original_width + group_dim - 1) >> group_dim_shift) as usize;
                let ch_rows = ((info.original_height + group_dim - 1) >> group_dim_shift) as usize;
                for gy in 0..ch_rows.min(num_rows) {
                    for gx in 0..ch_cols.min(num_cols) {
                        let dst_x = gx * gw;
                        let dst_y = gy * gh;
                        let tw = (info.width as usize).saturating_sub(dst_x).min(gw) as u32;
                        let th = (info.height as usize).saturating_sub(dst_y).min(gh) as u32;
                        if tw == 0 || th == 0 {
                            continue;
                        }
                        let g_idx = gy * num_cols + gx;
                        pass_group_tiles[pass_idx][g_idx].push(ModularGroupTile {
                            tr_channel_idx: i,
                            dst_x,
                            dst_y,
                            tile_info: ModularChannelInfo {
                                width: tw,
                                height: th,
                                original_width: tw << hshift,
                                original_height: th << vshift,
                                hshift,
                                vshift,
                                original_shift: info.original_shift,
                            },
                        });
                    }
                }
            } else {
                let lfw = (group_dim >> (hshift - 3)) as usize;
                let lfh = (group_dim >> (vshift - 3)) as usize;
                if lfw == 0 || lfh == 0 {
                    return Err(JxlError::InvalidBitstream(
                        "LF group dimension became 0 after shift",
                    ));
                }
                let ch_cols =
                    ((info.original_width + lf_dim - 1) >> (group_dim_shift + 3)) as usize;
                let ch_rows =
                    ((info.original_height + lf_dim - 1) >> (group_dim_shift + 3)) as usize;
                for gy in 0..ch_rows.min(num_lf_rows) {
                    for gx in 0..ch_cols.min(num_lf_cols) {
                        let dst_x = gx * lfw;
                        let dst_y = gy * lfh;
                        let tw = (info.width as usize).saturating_sub(dst_x).min(lfw) as u32;
                        let th = (info.height as usize).saturating_sub(dst_y).min(lfh) as u32;
                        if tw == 0 || th == 0 {
                            continue;
                        }
                        let lf_idx = gy * num_lf_cols + gx;
                        lf_group_tiles[lf_idx].push(ModularGroupTile {
                            tr_channel_idx: i,
                            dst_x,
                            dst_y,
                            tile_info: ModularChannelInfo {
                                width: tw,
                                height: th,
                                original_width: tw << hshift,
                                original_height: th << vshift,
                                hshift,
                                vshift,
                                original_shift: info.original_shift,
                            },
                        });
                    }
                }
            }
        }

        Ok(Self {
            global_ma_config,
            header: Some(header),
            ma_ctx: Some(ma_ctx),
            bit_depth,
            extra_channel_from,
            tr_channels: tr_channels.info,
            tr_grids,
            gmodular_count,
            lf_group_tiles,
            pass_group_tiles,
        })
    }

    /// Decodes the Modular sub-image for `lf_group_idx` if non-empty.
    pub fn decode_lf_group(
        &mut self,
        br: &mut BitReader<'_>,
        lf_group_idx: u32,
        num_lf_groups: u32,
    ) -> Result<(), JxlError> {
        let Some(tiles) = self.lf_group_tiles.get(lf_group_idx as usize).cloned() else {
            return Ok(());
        };
        if tiles.is_empty() {
            return Ok(());
        }
        let channels: Vec<ModularChannelInfo> = tiles.iter().map(|t| t.tile_info.clone()).collect();
        let stream_index = 1 + num_lf_groups + lf_group_idx;
        let decoded = decode_modular_subimage(
            br,
            channels,
            self.bit_depth,
            self.global_ma_config.as_ref(),
            stream_index,
        )?;
        for (tile, src_grid) in tiles.iter().zip(decoded) {
            self.tr_grids[tile.tr_channel_idx].copy_from_rect(tile.dst_x, tile.dst_y, &src_grid);
        }
        Ok(())
    }

    /// Decodes the Modular sub-image for `(pass_idx, group_idx)` if non-empty.
    pub fn decode_pass_group(
        &mut self,
        br: &mut BitReader<'_>,
        pass_idx: u32,
        group_idx: u32,
        num_lf_groups: u32,
        num_groups: u32,
    ) -> Result<(), JxlError> {
        let Some(tiles) = self
            .pass_group_tiles
            .get(pass_idx as usize)
            .and_then(|p| p.get(group_idx as usize))
            .cloned()
        else {
            return Ok(());
        };
        if tiles.is_empty() {
            return Ok(());
        }
        let channels: Vec<ModularChannelInfo> = tiles.iter().map(|t| t.tile_info.clone()).collect();
        let stream_index = 1 + 3 * num_lf_groups + 17 + pass_idx * num_groups + group_idx;
        let decoded = decode_modular_subimage(
            br,
            channels,
            self.bit_depth,
            self.global_ma_config.as_ref(),
            stream_index,
        )?;
        for (tile, src_grid) in tiles.iter().zip(decoded) {
            self.tr_grids[tile.tr_channel_idx].copy_from_rect(tile.dst_x, tile.dst_y, &src_grid);
        }
        Ok(())
    }

    /// Applies the global inverse transforms and returns the reconstructed Modular channels.
    pub fn finish(&mut self) -> Vec<ModularGrid> {
        let mut grids = std::mem::take(&mut self.tr_grids);
        if let Some(header) = &self.header {
            for tr in header.transform.iter().rev() {
                tr.inverse(&mut grids, self.bit_depth);
            }
        }
        grids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rct_all_42_roundtrip() {
        for rct_type in 0..42 {
            let rct = Rct {
                begin_c: 0,
                rct_type,
            };
            let mut g0 = ModularGrid::new(8, 8);
            let mut g1 = ModularGrid::new(8, 8);
            let mut g2 = ModularGrid::new(8, 8);
            for i in 0..64 {
                g0.data[i] = (i as i32 * 7 - 50) & 0xFF;
                g1.data[i] = (i as i32 * 13 + 20) & 0xFF;
                g2.data[i] = (i as i32 * 29 - 90) & 0xFF;
            }
            let orig = vec![g0.clone(), g1.clone(), g2.clone()];
            let mut work = orig.clone();
            rct.forward(&mut work);
            rct.inverse(&mut work);
            assert_eq!(work, orig, "RCT roundtrip failed for rct_type={rct_type}");
        }
    }

    #[test]
    fn test_modular_subimage_encode_decode_roundtrip() {
        let mut g0 = ModularGrid::new(16, 12);
        let mut g1 = ModularGrid::new(16, 12);
        let mut g2 = ModularGrid::new(16, 12);
        for y in 0..12 {
            for x in 0..16 {
                g0.set(x, y, (x * 11 + y * 5) as i32);
                g1.set(x, y, (x * 3 + y * 17) as i32);
                g2.set(x, y, (255 - x * 7 - y * 9) as i32);
            }
        }
        let orig = vec![g0, g1, g2];

        for predictor in [
            Predictor::Zero,
            Predictor::West,
            Predictor::North,
            Predictor::Gradient,
            Predictor::Select,
            Predictor::SelfCorrecting,
        ] {
            let mut bw = BitWriter::new();
            encode_modular_subimage(&mut bw, &orig, predictor, true).unwrap();
            let bytes = bw.into_bytes();

            let mut br = BitReader::new(&bytes);
            let channels = vec![
                ModularChannelInfo::new(16, 12, ChannelShift::from_shift(0)),
                ModularChannelInfo::new(16, 12, ChannelShift::from_shift(0)),
                ModularChannelInfo::new(16, 12, ChannelShift::from_shift(0)),
            ];
            let decoded = decode_modular_subimage(&mut br, channels, 8, None, 0).unwrap();
            assert_eq!(decoded, orig, "Modular roundtrip failed for {predictor:?}");
        }
    }

    #[test]
    fn test_modular_v2_palette_squeeze_adaptive_ans_lz77_roundtrip() {
        let w = 32usize;
        let h = 32usize;
        // 1. Indexed 4-color image -> Palette + Ans + LZ77
        let mut pal_grids = vec![
            ModularGrid::new(w, h),
            ModularGrid::new(w, h),
            ModularGrid::new(w, h),
        ];
        for y in 0..h {
            for x in 0..w {
                let idx = (x / 8 + y / 8) % 4;
                pal_grids[0].set(x, y, (idx * 50 + 10) as i32);
                pal_grids[1].set(x, y, (idx * 60 + 20) as i32);
                pal_grids[2].set(x, y, (idx * 70 + 30) as i32);
            }
        }
        let opts_pal = ModularEncodeOptions {
            predictor: Predictor::SelfCorrecting,
            use_rct_ycocg: true,
            enable_palette: true,
            enable_squeeze: false,
            adaptive_tree: true,
            entropy_mode: JxlEntropyMode::Ans,
            lz77: true,
            residual_multiplier: 1,
        };
        let mut bw = BitWriter::new();
        encode_modular_subimage_v2(&mut bw, &pal_grids, &opts_pal).expect("encode pal v2");
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let ch_info = vec![
            ModularChannelInfo::new(w as u32, h as u32, ChannelShift::from_shift(0)),
            ModularChannelInfo::new(w as u32, h as u32, ChannelShift::from_shift(0)),
            ModularChannelInfo::new(w as u32, h as u32, ChannelShift::from_shift(0)),
        ];
        let dec_pal =
            decode_modular_subimage(&mut br, ch_info.clone(), 8, None, 0).expect("decode pal v2");
        assert_eq!(dec_pal, pal_grids);

        // 2. Continuous gradient image -> Rct + Squeeze + SelfCorrecting + Adaptive Tree + Ans
        let mut sq_grids = vec![
            ModularGrid::new(w, h),
            ModularGrid::new(w, h),
            ModularGrid::new(w, h),
        ];
        for y in 0..h {
            for x in 0..w {
                sq_grids[0].set(x, y, ((x * 7 + y * 5) & 0xff) as i32);
                sq_grids[1].set(x, y, ((x * 3 + y * 11) & 0xff) as i32);
                sq_grids[2].set(x, y, ((x * 13 + y * 2) & 0xff) as i32);
            }
        }
        let opts_sq = ModularEncodeOptions {
            predictor: Predictor::SelfCorrecting,
            use_rct_ycocg: true,
            enable_palette: false,
            enable_squeeze: true,
            adaptive_tree: true,
            entropy_mode: JxlEntropyMode::Ans,
            lz77: true,
            residual_multiplier: 1,
        };
        let mut bw_sq = BitWriter::new();
        encode_modular_subimage_v2(&mut bw_sq, &sq_grids, &opts_sq).expect("encode squeeze v2");
        let sq_bytes = bw_sq.into_bytes();
        let mut br_sq = BitReader::new(&sq_bytes);
        let dec_sq =
            decode_modular_subimage(&mut br_sq, ch_info, 8, None, 0).expect("decode squeeze v2");
        assert_eq!(dec_sq, sq_grids);
    }
}
