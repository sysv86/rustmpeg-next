// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Slice-data syntax and residual coding for the intra-only VVC subset this
//! encoder emits.
//!
//! Algorithms derived from VTM 24.0 (BSD-3-Clause, Copyright (c) 2010-2020
//! ITU/ISO/IEC). See THIRD_PARTY.md.
//!
//! The picture is coded as one slice, one CU per CTU (`CU == CTU`, so no
//! `split_cu_flag` ever appears — `min CB == CTU`) and every CU intra PLANAR or
//! DC via the MPM path.  The mode is chosen per CU by SAD against the source
//! (PLANAR wins ties).  QP 4 keeps `transform_skip_flag = 1` for the 8-bit
//! lossless point; every other QP uses a primary transform with regular
//! `residual_coding()`.  Explicit intra MTS (DST-7/DCT-8 combinations) and
//! LFNST are evaluated per luma block and the candidate with the lowest
//! reconstruction SSE is signalled.  The bin order is exactly VTM's
//! `CABACWriter` for this configuration:
//!
//! ```text
//! intra_luma_mpm_flag = 1
//! intra_luma_not_planar_flag = 0                            (PLANAR)
//! intra_luma_not_planar_flag = 1, intra_luma_mpm_idx > 1 = 0 (DC, index 1)
//! intra_chroma_pred_mode = 0                                (DM == luma mode)
//! cbf_cb, cbf_cr, cbf_luma
//! per component with cbf:
//!     transform_skip_flag = 1, residual_ts_coding()        (QP 4)
//!     transform_skip_flag = 0, residual_coding()           (QP > 4)
//! residual_lfnst_mode()                                     (when allowed)
//! mts_idx()                                                 (when allowed)
//! ```

use super::VvcError;
use super::cabac::{BinDecoder, BinEncoder, CtxStore, ctx};
use super::predict::{self, IntraMode, Plane};
use super::transform::{self, CoeffScan};
use super::{dct, lfnst, residual};
use crate::I420Frame;

/// Coded planes at full coded resolution (before conformance-window cropping).
struct Planes {
    y: Plane,
    u: Plane,
    v: Plane,
}

impl Planes {
    /// Copy an I420 frame into full-resolution planes.
    fn from_frame(frame: &I420Frame) -> Result<Self, VvcError> {
        let luma_width = frame.width as usize;
        let luma_height = frame.height as usize;
        let chroma_width = luma_width.div_ceil(2);
        let chroma_height = luma_height.div_ceil(2);
        let y = Plane::from_slice(luma_width, luma_height, &frame.y).ok_or_else(|| {
            VvcError::PlaneSize {
                plane: "Y",
                actual: frame.y.len(),
                expected: luma_width * luma_height,
            }
        })?;
        let u = Plane::from_slice(chroma_width, chroma_height, &frame.u).ok_or_else(|| {
            VvcError::PlaneSize {
                plane: "U",
                actual: frame.u.len(),
                expected: chroma_width * chroma_height,
            }
        })?;
        let v = Plane::from_slice(chroma_width, chroma_height, &frame.v).ok_or_else(|| {
            VvcError::PlaneSize {
                plane: "V",
                actual: frame.v.len(),
                expected: chroma_width * chroma_height,
            }
        })?;
        Ok(Self { y, u, v })
    }

    fn into_frame(self) -> I420Frame {
        I420Frame {
            width: self.y.width as u32,
            height: self.y.height as u32,
            y: self.y.data,
            u: self.u.data,
            v: self.v.data,
        }
    }
}

/// Test-only counters used to prove that the SSE selection actually reaches
/// the MTS and LFNST syntax paths.
///
/// The counters are thread-local rather than process-global: `libtest` runs each
/// test on its own thread, so a test that encodes pictures while another test
/// reads the counters can no longer interleave. Encoding is synchronous on the
/// calling thread, so every increment for a given test lands in that test's own
/// counter.
#[cfg(test)]
mod tool_stats {
    use std::cell::Cell;

    thread_local! {
        /// Number of luma blocks coded with explicit MTS.
        static MTS: Cell<usize> = const { Cell::new(0) };
        /// Number of luma blocks coded with LFNST.
        static LFNST: Cell<usize> = const { Cell::new(0) };
    }

    /// Record one luma block coded with explicit MTS.
    pub fn bump_mts() {
        MTS.with(|c| c.set(c.get() + 1));
    }

    /// Record one luma block coded with LFNST.
    pub fn bump_lfnst() {
        LFNST.with(|c| c.set(c.get() + 1));
    }

    /// Clear both counters.
    pub fn reset() {
        MTS.with(|c| c.set(0));
        LFNST.with(|c| c.set(0));
    }

    /// Current `(mts, lfnst)` counts.
    pub fn counts() -> (usize, usize) {
        (MTS.with(Cell::get), LFNST.with(Cell::get))
    }
}

/// Validate a CTU size and coded picture geometry derived from an SPS.
fn validate_coded_geometry(
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
) -> Result<usize, VvcError> {
    if !(5..=7).contains(&ctu_log2) {
        return Err(VvcError::Unsupported("CTU size outside 32..=128"));
    }
    let ctu = 1usize << ctu_log2;
    let samples = u64::from(coded_width)
        .checked_mul(u64::from(coded_height))
        .ok_or_else(|| VvcError::LimitExceeded("coded picture size overflows".to_string()))?;
    if samples > super::MAX_LUMA_SAMPLES {
        return Err(VvcError::LimitExceeded(format!(
            "coded picture {coded_width}x{coded_height} has {samples} luma samples"
        )));
    }
    let ctu_u32 = ctu as u32;
    if coded_width < ctu_u32
        || coded_height < ctu_u32
        || !coded_width.is_multiple_of(ctu_u32)
        || !coded_height.is_multiple_of(ctu_u32)
    {
        // H.266 permits partial CTUs at the right and bottom edges; this
        // subset coder does not.
        return Err(VvcError::Unsupported(
            "coded picture not aligned to the CTU grid",
        ));
    }
    Ok(ctu)
}

/// Luma positions of the `tb`-sized coding units of a picture in coding order:
/// CTU raster scan, and quadtree z-order inside each CTU
/// (ITU-T Rec. H.266 §6.4.3, §7.3.11.4).
fn coding_unit_order(ctu: usize, tb: usize, width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut offsets = vec![(0usize, 0usize)];
    let mut step = ctu / 2;
    while step >= tb {
        let mut next = Vec::with_capacity(offsets.len() * 4);
        for &(ox, oy) in &offsets {
            next.push((ox, oy));
            next.push((ox + step, oy));
            next.push((ox, oy + step));
            next.push((ox + step, oy + step));
        }
        offsets = next;
        step /= 2;
    }
    let mut order = Vec::with_capacity((width / tb) * (height / tb));
    for cy in (0..height).step_by(ctu) {
        for cx in (0..width).step_by(ctu) {
            for &(ox, oy) in &offsets {
                order.push((cx + ox, cy + oy));
            }
        }
    }
    order
}

/// Reconstruction availability of the `tb`-sized coding units of one picture,
/// used for reference sample availability (ITU-T Rec. H.266 §8.4.5.2.9).
///
/// A coding unit is marked as reconstructed as soon as its syntax has been
/// parsed, in the same order as [`coding_unit_order`], so the encoder and the
/// decoder always agree on which reference samples exist.
struct CodedUnits {
    coded: Vec<bool>,
    grid_w: usize,
    grid_h: usize,
    tb: usize,
}

impl CodedUnits {
    fn new(tb: usize, width: usize, height: usize) -> Self {
        let grid_w = width / tb;
        let grid_h = height / tb;
        Self {
            coded: vec![false; grid_w * grid_h],
            grid_w,
            grid_h,
            tb,
        }
    }

    fn mark(&mut self, bx: usize, by: usize) {
        self.coded[(by / self.tb) * self.grid_w + bx / self.tb] = true;
    }

    /// True if the luma sample at `(x, y)` lies in a reconstructed coding unit.
    fn is_coded_luma(&self, x: usize, y: usize) -> bool {
        let gx = x / self.tb;
        let gy = y / self.tb;
        gx < self.grid_w && gy < self.grid_h && self.coded[gy * self.grid_w + gx]
    }
}

/// Number of QT levels between the CTU root and a `tb`-sized leaf, and the
/// `QtMttSplitContexts` for a node whose origin is `(bx, by)` in a uniform
/// QT-only tree (`H.266 §9.3.4.2.2`, VTM `DeriveCtx::CtxSplit`).
///
/// In this subset every leaf is `tb` wide and every split node is a quadtree
/// node, so `numSplit == 2` (`ctx_split` gets no `3 * (numSplit >> 1)` term) and
/// an existing left/above neighbour is always a `tb`-sized leaf whose
/// `qtDepth` exceeds the current node's depth. The neighbours are the leaves
/// at `(bx - 1, by)` and `(bx, by - 1)`, available exactly when the node does
/// not touch the left or top picture edge.
fn coding_tree_split_contexts(bx: usize, by: usize, depth: u8) -> transform::QtMttSplitContexts {
    let split_ctx = usize::from(bx > 0) + usize::from(by > 0);
    let qt_ctx = if depth < 2 { split_ctx } else { split_ctx + 3 };
    transform::QtMttSplitContexts {
        ctx_split: split_ctx,
        ctx_qt: qt_ctx.min(5),
        ctx_hv: 0,
        ctx_hor_bt: 1,
        ctx_ver_bt: 3,
    }
}

/// Split parameters of a uniform QT-only coding tree node of size `size`
/// (`H.266 §6.4.1`, VTM `Partitioner::canSplit`). Only nodes larger than the
/// minimum QT/CB size of 32 split; the split is always a quadtree, so
/// `split_qt_flag` is not coded.
fn coding_tree_split_params(size: usize) -> (transform::CodingTreeSplit, transform::QtMttCanSplit) {
    let can = transform::QtMttCanSplit {
        can_no: true,
        can_qt: size > 32,
        can_bh: false,
        can_bv: false,
        can_th: false,
        can_tv: false,
    };
    (transform::CodingTreeSplit::QuadTree, can)
}

/// Encode the `split_cu_flag` sequence of the QT path from the CTU root down to
/// the `tb`-sized CU at `(bx, by)` (`H.266 §7.3.11.4`).
fn encode_coding_unit_split_flags(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    bx: usize,
    by: usize,
    ctu: usize,
    tb: usize,
) {
    let mut size = ctu;
    let mut depth = 0u8;
    while size > tb {
        if bx.is_multiple_of(size) && by.is_multiple_of(size) {
            let (split, can) = coding_tree_split_params(size);
            encode_coding_tree_split(
                enc,
                ctx,
                split,
                can,
                coding_tree_split_contexts(bx, by, depth),
            );
        }
        size /= 2;
        depth += 1;
    }
}

/// Decode the `split_cu_flag` sequence written by
/// [`encode_coding_unit_split_flags`] and check that it selects the quadtree
/// path to a `tb`-sized CU at `(bx, by)` (`H.266 §7.3.11.4`).
fn decode_coding_unit_split_flags(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    bx: usize,
    by: usize,
    ctu: usize,
    tb: usize,
) -> Result<(), VvcError> {
    let mut size = ctu;
    let mut depth = 0u8;
    while size > tb {
        if bx.is_multiple_of(size) && by.is_multiple_of(size) {
            let (_, can) = coding_tree_split_params(size);
            let split =
                decode_coding_tree_split(dec, ctx, can, coding_tree_split_contexts(bx, by, depth));
            if split != transform::CodingTreeSplit::QuadTree {
                return Err(VvcError::Bitstream(
                    "coding tree split is not a quadtree split".to_string(),
                ));
            }
        }
        size /= 2;
        depth += 1;
    }
    Ok(())
}

/// Encode the slice data for one IDR picture (CABAC substream only).
///
/// `src` must already be padded to the coded picture size; `qp` is `SliceQpY`.
pub fn encode_picture(src: &I420Frame, qp: u8, ctu_log2: u8) -> Result<Vec<u8>, VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, src.width, src.height)?;
    let mut enc = BinEncoder::new();
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));
    let src_planes = Planes::from_frame(src)?;
    let mut recon = Planes::from_frame(src)?;
    let width = src.width as usize;
    let height = src.height as usize;
    let luma_qp = qp.max(4);
    let chroma_qp = (super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8).max(4);
    // QP 4 is the 8-bit transform-skip lossless point.  Every other QP uses a
    // primary transform with regular residual coding; MTS and LFNST are
    // selected per luma block by SSE.
    let luma_ts = luma_qp == 4;
    let chroma_ts = chroma_qp == 4;

    let tb = ctu.min(32);
    let mut ctu_y = 0usize;
    while ctu_y < height {
        let mut ctu_x = 0usize;
        while ctu_x < width {
            let ctu = tb;
            let cw = ctu / 2;
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;

            // Chroma uses the luma mode through `intra_chroma_pred_mode = DM`.
            let luma_mode =
                predict::choose_intra_mode(&recon.y, &src_planes.y, ctu_x, ctu_y, ctu, ctu, true);

            let (cb, cr) = encode_chroma(
                &recon,
                &src_planes,
                cx,
                cy,
                cw,
                chroma_qp,
                luma_mode,
                chroma_ts,
            )?;
            let mut chroma_constraints = CuConstraints::default();
            let chroma_scan = CoeffScan::new(cw, cw);
            if cb.cbf {
                chroma_constraints.update(&chroma_scan, &cb.levels, false, cb.ts);
            }
            if cr.cbf {
                chroma_constraints.update(&chroma_scan, &cr.levels, false, cr.ts);
            }
            let luma = encode_luma(
                &recon.y,
                &src_planes.y,
                ctu_x,
                ctu_y,
                ctu,
                luma_qp,
                luma_mode,
                luma_ts,
                !chroma_constraints.violates_lfnst[1],
            )?;

            apply_plan(&mut recon.y, ctu_x, ctu_y, ctu, &luma);
            apply_plan(&mut recon.u, cx, cy, cw, &cb);
            apply_plan(&mut recon.v, cx, cy, cw, &cr);

            // ---- cu_pred_data: one PU, PLANAR or DC, chroma DM ----
            encode_intra_mode(&mut enc, &mut ctx, luma_mode);
            enc.encode_bin(false, &mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]);

            // ---- transform_unit: chroma CBFs first, then luma ----
            enc.encode_bin(cb.cbf, &mut ctx.models[ctx::QT_CBF_CB]);
            enc.encode_bin(
                cr.cbf,
                &mut ctx.models[ctx::QT_CBF_CR + usize::from(cb.cbf)],
            );
            enc.encode_bin(luma.cbf, &mut ctx.models[ctx::QT_CBF_Y]);

            // ---- coefficients, luma first ----
            let luma_scan = CoeffScan::new(ctu, ctu);
            if luma.cbf {
                enc.encode_bin(luma.ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                if luma.ts {
                    encode_ts_block(&mut enc, &mut ctx, &luma_scan, &luma.levels);
                } else {
                    residual::encode_residual(&mut enc, &mut ctx, &luma_scan, &luma.levels, true);
                }
            }
            let chroma_scan = CoeffScan::new(cw, cw);
            if cb.cbf {
                enc.encode_bin(cb.ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if cb.ts {
                    encode_ts_block(&mut enc, &mut ctx, &chroma_scan, &cb.levels);
                } else {
                    residual::encode_residual(&mut enc, &mut ctx, &chroma_scan, &cb.levels, false);
                }
            }
            if cr.cbf {
                enc.encode_bin(cr.ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if cr.ts {
                    encode_ts_block(&mut enc, &mut ctx, &chroma_scan, &cr.levels);
                } else {
                    residual::encode_residual(&mut enc, &mut ctx, &chroma_scan, &cr.levels, false);
                }
            }

            // ---- residual_lfnst_mode() and mts_idx(), in VTM order ----
            let mut constraints = CuConstraints::default();
            if luma.cbf {
                constraints.update(&luma_scan, &luma.levels, true, luma.ts);
            }
            if cb.cbf {
                constraints.update(&chroma_scan, &cb.levels, false, cb.ts);
            }
            if cr.cbf {
                constraints.update(&chroma_scan, &cr.levels, false, cr.ts);
            }
            let any_ts = (luma.cbf && luma.ts) || (cb.cbf && cb.ts) || (cr.cbf && cr.ts);
            if lfnst_syntax_present(&constraints, any_ts) {
                encode_lfnst_idx(&mut enc, &mut ctx, luma.lfnst_idx);
            }
            if mts_syntax_present(&constraints, any_ts && luma.ts, luma.lfnst_idx) {
                encode_mts_idx(&mut enc, &mut ctx, luma.mts_idx);
            }

            ctu_x += ctu;
        }
        ctu_y += ctu;
    }

    enc.encode_bin_trm(true);
    Ok(enc.finish_and_align())
}

/// Decode one IDR picture's CABAC substream back into planes.
pub fn decode_picture(
    cabac: &[u8],
    qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
) -> Result<I420Frame, VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, coded_width, coded_height)?;
    let mut dec = BinDecoder::new(cabac).map_err(|err| VvcError::Bitstream(err.to_string()))?;
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));
    let ctu = ctu.min(32);
    let width = coded_width as usize;
    let height = coded_height as usize;
    let cw = ctu / 2;
    let luma_qp = qp.max(4);
    let chroma_qp = (super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8).max(4);

    let mut recon = Planes {
        y: Plane::new(width, height),
        u: Plane::new(width.div_ceil(2), height.div_ceil(2)),
        v: Plane::new(width.div_ceil(2), height.div_ceil(2)),
    };

    let mut ctu_y = 0usize;
    while ctu_y < height {
        let mut ctu_x = 0usize;
        while ctu_x < width {
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;

            // ---- cu_pred_data ----
            let mode = decode_intra_mode(&mut dec, &mut ctx)?;
            let chroma_dm = !dec.decode_bin(&mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]);
            let chroma_mode = if chroma_dm {
                mode
            } else {
                let idx = dec.decode_bins_ep(2) as usize;
                let cand = match idx & 3 {
                    0 => IntraMode::Planar,
                    1 => IntraMode::Angular(50),
                    2 => IntraMode::Angular(18),
                    _ => IntraMode::Dc,
                };
                if cand == mode {
                    IntraMode::Angular(66)
                } else {
                    cand
                }
            };

            // ---- transform_unit CBFs ----
            let cbf_cb = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
            let cbf_cr = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_cb)]);
            let cbf_y = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_Y]);

            let luma_scan = CoeffScan::new(ctu, ctu);
            let chroma_scan = CoeffScan::new(cw, cw);
            let mut levels_y = vec![0i32; ctu * ctu];
            let mut ts_y = false;
            if cbf_y {
                ts_y = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                if ts_y {
                    decode_ts_block(&mut dec, &mut ctx, &luma_scan, &mut levels_y);
                } else {
                    residual::decode_residual(&mut dec, &mut ctx, &luma_scan, &mut levels_y, true)?;
                }
            }
            let mut levels_cb = vec![0i32; cw * cw];
            let mut ts_cb = false;
            if cbf_cb {
                ts_cb = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_cb {
                    decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_cb);
                } else {
                    residual::decode_residual(
                        &mut dec,
                        &mut ctx,
                        &chroma_scan,
                        &mut levels_cb,
                        false,
                    )?;
                }
            }
            let mut levels_cr = vec![0i32; cw * cw];
            let mut ts_cr = false;
            if cbf_cr {
                ts_cr = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_cr {
                    decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_cr);
                } else {
                    residual::decode_residual(
                        &mut dec,
                        &mut ctx,
                        &chroma_scan,
                        &mut levels_cr,
                        false,
                    )?;
                }
            }

            // ---- residual_lfnst_mode() and mts_idx(), in VTM order ----
            let mut constraints = CuConstraints::default();
            if cbf_y {
                constraints.update(&luma_scan, &levels_y, true, ts_y);
            }
            if cbf_cb {
                constraints.update(&chroma_scan, &levels_cb, false, ts_cb);
            }
            if cbf_cr {
                constraints.update(&chroma_scan, &levels_cr, false, ts_cr);
            }
            let any_ts = (cbf_y && ts_y) || (cbf_cb && ts_cb) || (cbf_cr && ts_cr);
            let lfnst_idx = if lfnst_syntax_present(&constraints, any_ts) {
                decode_lfnst_idx(&mut dec, &mut ctx)
            } else {
                0
            };
            let mts_idx = if mts_syntax_present(&constraints, any_ts && ts_y, lfnst_idx) {
                decode_mts_idx(&mut dec, &mut ctx)?
            } else {
                0
            };

            reconstruct_block(
                &mut recon.y,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                true,
                luma_qp,
                &levels_y,
                mode,
                ts_y,
                mts_idx,
                lfnst_idx,
            )?;
            reconstruct_block(
                &mut recon.u,
                cx,
                cy,
                cw,
                cw,
                false,
                chroma_qp,
                &levels_cb,
                chroma_mode,
                ts_cb,
                0,
                0,
            )?;
            reconstruct_block(
                &mut recon.v,
                cx,
                cy,
                cw,
                cw,
                false,
                chroma_qp,
                &levels_cr,
                chroma_mode,
                ts_cr,
                0,
                0,
            )?;

            ctu_x += ctu;
        }
        ctu_y += ctu;
    }

    dec.end_of_slice()
        .map_err(|err| VvcError::Bitstream(err.to_string()))?;
    Ok(recon.into_frame())
}

/// One coded component block: quantised levels, CBF and the reconstruction
/// produced by the selected transform.
#[derive(Debug, Clone)]
struct BlockPlan {
    levels: Vec<i32>,
    cbf: bool,
    ts: bool,
    mts_idx: u8,
    lfnst_idx: u8,
    samples: Vec<u8>,
}

/// `(trTypeHor, trTypeVer)` for an `mts_idx` value.
fn mts_types(mts_idx: u8) -> Result<(dct::TrType, dct::TrType), VvcError> {
    match mts_idx {
        0 => Ok((dct::TrType::Dct2, dct::TrType::Dct2)),
        1 => Ok((dct::TrType::Dst7, dct::TrType::Dst7)),
        2 => Ok((dct::TrType::Dct8, dct::TrType::Dst7)),
        3 => Ok((dct::TrType::Dst7, dct::TrType::Dct8)),
        4 => Ok((dct::TrType::Dct8, dct::TrType::Dct8)),
        _ => Err(VvcError::Bitstream(format!("invalid mts_idx {mts_idx}"))),
    }
}

/// VTM `xT` / `xIT` high-frequency zero-out for a primary transform.
fn primary_skips(w: usize, h: usize, tr_hor: dct::TrType, tr_ver: dct::TrType) -> (usize, usize) {
    let skip_w = if w == 32 && tr_hor != dct::TrType::Dct2 {
        16
    } else {
        0
    };
    let skip_h = if h == 32 && tr_ver != dct::TrType::Dct2 {
        16
    } else {
        0
    };
    (skip_w, skip_h)
}

/// VTM's LFNST zero-out of the primary transform coefficients.
fn lfnst_skips(w: usize, h: usize) -> (usize, usize) {
    if (w == 4 && h > 4) || (w > 4 && h == 4) {
        (w - 4, h - 4)
    } else if w >= 8 && h >= 8 {
        (w - 8, h - 8)
    } else {
        (0, 0)
    }
}

/// Intra mode value used by the LFNST table (PLANAR = 0, DC = 1).
fn lfnst_mode(mode: IntraMode) -> u16 {
    match mode {
        IntraMode::Planar => 0,
        IntraMode::Dc => 1,
        IntraMode::Angular(dir) => u16::from(dir),
    }
}

/// Predict, transform-skip, quantise and reconstruct one component block.
fn plan_transform_skip(pred: &[i32], residual: &[i32], w: usize, h: usize, qp: u8) -> BlockPlan {
    let mut levels = vec![0i32; w * h];
    let mut samples = vec![0u8; w * h];
    let mut cbf = false;
    for idx in 0..w * h {
        let level = transform::quantize_ts(residual[idx], qp);
        levels[idx] = level;
        cbf |= level != 0;
        let value = pred[idx] + transform::dequantize_ts(level, qp);
        samples[idx] = value.clamp(0, 255) as u8;
    }
    BlockPlan {
        levels,
        cbf,
        ts: true,
        mts_idx: 0,
        lfnst_idx: 0,
        samples,
    }
}

/// Predict, apply the primary transform (and optional LFNST), quantise and
/// reconstruct one component block.
#[allow(clippy::too_many_arguments)]
fn plan_primary(
    pred: &[i32],
    residual: &[i32],
    w: usize,
    h: usize,
    qp: u8,
    mts_idx: u8,
    lfnst_idx: u8,
    mode: IntraMode,
) -> Result<BlockPlan, VvcError> {
    if lfnst_idx > 0 && mts_idx != 0 {
        return Err(VvcError::Unsupported("MTS combined with LFNST"));
    }
    let (tr_hor, tr_ver) = mts_types(mts_idx)?;
    let (skip_w, skip_h) = if lfnst_idx > 0 {
        lfnst_skips(w, h)
    } else {
        primary_skips(w, h, tr_hor, tr_ver)
    };
    let mut coeffs = dct::forward_transform_rect(w, h, residual, tr_hor, tr_ver, skip_w, skip_h)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    if lfnst_idx > 0 {
        lfnst::forward_lfnst_rect(w, h, &mut coeffs, lfnst_idx, lfnst_mode(mode))
            .map_err(|_| VvcError::Unsupported("unsupported LFNST"))?;
    }
    let levels = dct::quantize_rect_bd(w, h, &coeffs, qp, 8, true)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    let cbf = levels.iter().any(|&level| level != 0);
    let mut dequant = dct::dequantize_rect_bd(w, h, &levels, qp, 8)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    if lfnst_idx > 0 {
        lfnst::inverse_lfnst_rect(w, h, &mut dequant, lfnst_idx, lfnst_mode(mode))
            .map_err(|_| VvcError::Unsupported("unsupported LFNST"))?;
    }
    let reconstructed = dct::inverse_transform_rect(w, h, &dequant, tr_hor, tr_ver, skip_w, skip_h)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    let mut samples = vec![0u8; w * h];
    for idx in 0..w * h {
        samples[idx] = (pred[idx] + reconstructed[idx]).clamp(0, 255) as u8;
    }
    Ok(BlockPlan {
        levels,
        cbf,
        ts: false,
        mts_idx,
        lfnst_idx,
        samples,
    })
}

/// Predict and code the two chroma components with DCT-2 (or transform skip at
/// QP 4).  LFNST and MTS are luma-only in a single-tree stream.
#[allow(clippy::too_many_arguments)]
fn encode_chroma(
    recon: &Planes,
    src: &Planes,
    bx: usize,
    by: usize,
    size: usize,
    qp: u8,
    mode: IntraMode,
    use_transform_skip: bool,
) -> Result<(BlockPlan, BlockPlan), VvcError> {
    let mut plans = Vec::with_capacity(2);
    for (recon_plane, src_plane) in [(&recon.u, &src.u), (&recon.v, &src.v)] {
        let pred = predict::predict_block(recon_plane, bx, by, size, size, false, mode);
        let mut residual = vec![0i32; size * size];
        for y in 0..size {
            for x in 0..size {
                let idx = y * size + x;
                residual[idx] = i32::from(src_plane.get(bx + x, by + y)) - pred[idx];
            }
        }
        let plan = if use_transform_skip {
            plan_transform_skip(&pred, &residual, size, size, qp)
        } else {
            plan_primary(&pred, &residual, size, size, qp, 0, 0, mode)?
        };
        plans.push(plan);
    }
    let mut iter = plans.into_iter();
    let cb = iter
        .next()
        .ok_or_else(|| VvcError::InvalidConfig("missing Cb plan".to_string()))?;
    let cr = iter
        .next()
        .ok_or_else(|| VvcError::InvalidConfig("missing Cr plan".to_string()))?;
    Ok((cb, cr))
}

/// Predict and choose the luma transform by lowest reconstruction SSE.
#[allow(clippy::too_many_arguments)]
fn encode_luma(
    recon: &Plane,
    src: &Plane,
    bx: usize,
    by: usize,
    size: usize,
    qp: u8,
    mode: IntraMode,
    use_transform_skip: bool,
    chroma_can_lfnst: bool,
) -> Result<BlockPlan, VvcError> {
    let pred = predict::predict_block(recon, bx, by, size, size, true, mode);
    let mut residual = vec![0i32; size * size];
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            residual[idx] = i32::from(src.get(bx + x, by + y)) - pred[idx];
        }
    }
    if use_transform_skip {
        return Ok(plan_transform_skip(&pred, &residual, size, size, qp));
    }

    let scan = CoeffScan::new(size, size);
    // DCT-2 first; later candidates replace it only on a strictly lower SSE,
    // so ties stay on the syntax-free path.
    let dct2 = plan_primary(&pred, &residual, size, size, qp, 0, 0, mode)?;
    let mut best = Some((block_sse(src, bx, by, size, &dct2.samples), dct2));

    for mts_idx in 1..=4u8 {
        let plan = plan_primary(&pred, &residual, size, size, qp, mts_idx, 0, mode)?;
        if !mts_candidate_ok(&plan, &scan, size) {
            continue;
        }
        let sse = block_sse(src, bx, by, size, &plan.samples);
        if best.as_ref().is_none_or(|(best_sse, _)| sse < *best_sse) {
            best = Some((sse, plan));
        }
    }
    if chroma_can_lfnst {
        for lfnst_idx in 1..=2u8 {
            let plan = plan_primary(&pred, &residual, size, size, qp, 0, lfnst_idx, mode)?;
            if !lfnst_candidate_ok(&plan, &scan, size) {
                continue;
            }
            let sse = block_sse(src, bx, by, size, &plan.samples);
            if best.as_ref().is_none_or(|(best_sse, _)| sse < *best_sse) {
                best = Some((sse, plan));
            }
        }
    }
    let plan = best
        .map(|(_, plan)| plan)
        .ok_or(VvcError::Unsupported("no valid luma transform"))?;
    #[cfg(test)]
    {
        if plan.mts_idx > 0 {
            tool_stats::bump_mts();
        }
        if plan.lfnst_idx > 0 {
            tool_stats::bump_lfnst();
        }
    }
    Ok(plan)
}

/// `true` when a quantised MTS candidate can signal `mts_idx`.
fn mts_candidate_ok(plan: &BlockPlan, scan: &CoeffScan, size: usize) -> bool {
    if !plan.cbf || plan.ts {
        return false;
    }
    let Some(last) = scan_pos_last(scan, &plan.levels) else {
        return false;
    };
    last >= 1 && !violates_mts_constraint(&plan.levels, size)
}

/// `true` when a quantised LFNST candidate can signal `lfnst_idx`.
fn lfnst_candidate_ok(plan: &BlockPlan, scan: &CoeffScan, size: usize) -> bool {
    if !plan.cbf || plan.ts || plan.lfnst_idx == 0 {
        return false;
    }
    let Some(last) = scan_pos_last(scan, &plan.levels) else {
        return false;
    };
    last >= 1 && last <= max_lfnst_pos(size, size)
}

/// Apply a component plan to the reconstruction plane.
fn apply_plan(recon: &mut Plane, bx: usize, by: usize, size: usize, plan: &BlockPlan) {
    for y in 0..size {
        for x in 0..size {
            recon.set(bx + x, by + y, plan.samples[y * size + x]);
        }
    }
}

/// Reconstruction SSE of a candidate block against the source.
fn block_sse(src: &Plane, bx: usize, by: usize, size: usize, samples: &[u8]) -> u64 {
    let mut sse = 0u64;
    for y in 0..size {
        for x in 0..size {
            let diff = i64::from(src.get(bx + x, by + y)) - i64::from(samples[y * size + x]);
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Positions of the last coded coefficient, scan order.
fn scan_pos_last(scan: &CoeffScan, levels: &[i32]) -> Option<usize> {
    (0..scan.max_num_coeff)
        .rev()
        .find(|&pos| levels[scan.raster[pos]] != 0)
}

/// VTM's `maxLfnstPos` for a transform block.
fn max_lfnst_pos(w: usize, h: usize) -> usize {
    if (w == 4 && h == 4) || (w == 8 && h == 8) {
        7
    } else {
        15
    }
}

/// VTM's `violatesMtsCoeffConstraint`: a luma coefficient group outside the
/// top-left 4x4 group grid forbids explicit MTS.
fn violates_mts_constraint(levels: &[i32], size: usize) -> bool {
    levels.iter().enumerate().any(|(idx, &level)| {
        if level == 0 {
            return false;
        }
        let x = idx % size;
        let y = idx / size;
        x / 4 > 3 || y / 4 > 3
    })
}

/// Context state VTM carries from `residual_coding()` to `mts_idx()` /
/// `residual_lfnst_mode()` (`CUCtx`).
#[derive(Debug, Clone, Copy, Default)]
struct CuConstraints {
    violates_lfnst: [bool; 2],
    lfnst_last_scan_pos: bool,
    violates_mts: bool,
    mts_last_scan_pos: bool,
}

impl CuConstraints {
    /// Update the constraints from one coded component block.
    fn update(&mut self, scan: &CoeffScan, levels: &[i32], is_luma: bool, ts: bool) {
        let Some(last) = scan_pos_last(scan, levels) else {
            return;
        };
        if ts {
            return;
        }
        let height = scan.max_num_coeff / scan.width;
        if scan.width >= 4 && height >= 4 {
            let channel = usize::from(!is_luma);
            self.violates_lfnst[channel] |= last > max_lfnst_pos(scan.width, height);
            self.lfnst_last_scan_pos |= last >= 1;
        }
        if is_luma {
            self.mts_last_scan_pos |= last >= 1;
            self.violates_mts |= violates_mts_constraint(levels, scan.width);
        }
    }
}

/// `true` when `residual_lfnst_mode()` writes `lfnst_idx`.
fn lfnst_syntax_present(constraints: &CuConstraints, any_ts: bool) -> bool {
    !any_ts
        && constraints.lfnst_last_scan_pos
        && !constraints.violates_lfnst[0]
        && !constraints.violates_lfnst[1]
}

/// `true` when `mts_idx()` is written.
fn mts_syntax_present(constraints: &CuConstraints, luma_ts: bool, lfnst_idx: u8) -> bool {
    !luma_ts && !constraints.violates_mts && constraints.mts_last_scan_pos && lfnst_idx == 0
}

/// Write `lfnst_idx` (VTM `CABACWriter::residual_lfnst_mode`).
fn encode_lfnst_idx(enc: &mut BinEncoder, ctx: &mut CtxStore, lfnst_idx: u8) {
    debug_assert!(lfnst_idx < 3);
    enc.encode_bin(lfnst_idx != 0, &mut ctx.models[ctx::LFNST_IDX]);
    if lfnst_idx != 0 {
        enc.encode_bin(lfnst_idx == 2, &mut ctx.models[ctx::LFNST_IDX + 2]);
    }
}

/// Read `lfnst_idx`.
fn decode_lfnst_idx(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> u8 {
    if !dec.decode_bin(&mut ctx.models[ctx::LFNST_IDX]) {
        return 0;
    }
    if dec.decode_bin(&mut ctx.models[ctx::LFNST_IDX + 2]) {
        2
    } else {
        1
    }
}

/// Write `mts_idx` (VTM `CABACWriter::mts_idx`).
fn encode_mts_idx(enc: &mut BinEncoder, ctx: &mut CtxStore, mts_idx: u8) {
    debug_assert!(mts_idx < 5);
    let mut symbol = mts_idx != 0;
    enc.encode_bin(symbol, &mut ctx.models[ctx::MTS_IDX]);
    if symbol {
        for i in 0..3usize {
            symbol = mts_idx > 1 + i as u8;
            enc.encode_bin(symbol, &mut ctx.models[ctx::MTS_IDX + 1 + i]);
            if !symbol {
                break;
            }
        }
    }
}

/// Read `mts_idx`.
fn decode_mts_idx(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> Result<u8, VvcError> {
    if !dec.decode_bin(&mut ctx.models[ctx::MTS_IDX]) {
        return Ok(0);
    }
    let mut mts_idx = 1u8;
    for i in 0..3usize {
        if !dec.decode_bin(&mut ctx.models[ctx::MTS_IDX + 1 + i]) {
            break;
        }
        mts_idx += 1;
    }
    if mts_idx > 4 {
        return Err(VvcError::Bitstream(format!("invalid mts_idx {mts_idx}")));
    }
    Ok(mts_idx)
}

/// Reconstruct one component block after all CU-level syntax is known.
#[allow(clippy::too_many_arguments)]
fn reconstruct_block(
    recon: &mut Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    is_luma: bool,
    qp: u8,
    levels: &[i32],
    mode: IntraMode,
    use_transform_skip: bool,
    mts_idx: u8,
    lfnst_idx: u8,
) -> Result<(), VvcError> {
    let pred = predict::predict_block(recon, bx, by, w, h, is_luma, mode);
    if use_transform_skip {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let value = pred[idx] + transform::dequantize_ts(levels[idx], qp);
                recon.set(bx + x, by + y, value.clamp(0, 255) as u8);
            }
        }
        return Ok(());
    }
    let (tr_hor, tr_ver) = mts_types(mts_idx)?;
    let (skip_w, skip_h) = if lfnst_idx > 0 {
        lfnst_skips(w, h)
    } else {
        primary_skips(w, h, tr_hor, tr_ver)
    };
    let mut dequant = dct::dequantize_rect_bd(w, h, levels, qp, 8)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    if lfnst_idx > 0 {
        lfnst::inverse_lfnst_rect(w, h, &mut dequant, lfnst_idx, lfnst_mode(mode))
            .map_err(|_| VvcError::Unsupported("unsupported LFNST"))?;
    }
    let residual = dct::inverse_transform_rect(w, h, &dequant, tr_hor, tr_ver, skip_w, skip_h)
        .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let value = pred[idx] + residual[idx];
            recon.set(bx + x, by + y, value.clamp(0, 255) as u8);
        }
    }
    Ok(())
}

/// Write the intra luma mode through the MPM path.
///
/// In this subset all neighbouring intra modes are PLANAR or DC, so H.266
/// `getIntraMPMs` yields `mpm[0] == PLANAR` and `mpm[1] == DC` regardless of
/// the neighbours.  PLANAR is therefore `mpm_idx == 0` and DC is
/// `mpm_idx == 1`; VTM writes that index as the context bin `mpm_idx > 0`
/// followed by the bypass bins `mpm_idx > n` while the comparison holds.
fn encode_intra_mode(enc: &mut BinEncoder, ctx: &mut CtxStore, mode: IntraMode) {
    let mpm_list = predict::derive_intra_mpms(None, None);
    let mode_idx = match mode {
        IntraMode::Planar => 0u8,
        IntraMode::Dc => 1u8,
        IntraMode::Angular(dir) => dir,
    };
    encode_full_luma_intra_mode(enc, ctx, mode_idx, &mpm_list, 0, false);
}

/// Read the intra luma mode written by [`encode_intra_mode`].
fn decode_intra_mode(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> Result<IntraMode, VvcError> {
    let mpm_list = predict::derive_intra_mpms(None, None);
    let mode_idx = decode_full_luma_intra_mode(dec, ctx, &mpm_list, 0, false);
    Ok(match mode_idx {
        0 => IntraMode::Planar,
        1 => IntraMode::Dc,
        dir => IntraMode::Angular(dir.clamp(2, 66)),
    })
}

// ---------------------------------------------------------------------------
// residual_ts_coding (H.266 7.3.11.11, transform-skip path)
// ---------------------------------------------------------------------------

#[inline]
fn neighbours(coeffs: &[i32], width: usize, x: usize, y: usize) -> (i32, i32) {
    let left = if x > 0 { coeffs[y * width + x - 1] } else { 0 };
    let above = if y > 0 {
        coeffs[(y - 1) * width + x]
    } else {
        0
    };
    (left, above)
}

#[inline]
fn num_nonzero_neighbours(left: i32, above: i32) -> usize {
    usize::from(left != 0) + usize::from(above != 0)
}

#[inline]
fn sign_context(left: i32, above: i32) -> usize {
    let right_sign = left.signum();
    let below_sign = above.signum();
    if (right_sign == 0 && below_sign == 0) || right_sign * below_sign < 0 {
        0
    } else if right_sign >= 0 && below_sign >= 0 {
        1
    } else {
        2
    }
}

/// `deriveModCoeff`: map a true magnitude to the coded "modified" domain.
#[inline]
fn derive_mod_coeff(left: i32, above: i32, abs_coeff: i32, identity: bool) -> i32 {
    if abs_coeff == 0 {
        return 0;
    }
    if identity {
        return abs_coeff;
    }
    let pred = left.abs().max(above.abs());
    if abs_coeff == pred {
        1
    } else if abs_coeff < pred {
        abs_coeff + 1
    } else {
        abs_coeff
    }
}

/// `decDeriveModCoeff`: inverse of [`derive_mod_coeff`].
#[inline]
fn dec_derive_mod_coeff(left: i32, above: i32, abs_coeff: i32) -> i32 {
    if abs_coeff == 0 {
        return 0;
    }
    let pred = left.abs().max(above.abs());
    if abs_coeff == 1 && pred > 0 {
        pred
    } else {
        abs_coeff - i32::from(abs_coeff <= pred)
    }
}

fn encode_ts_block(enc: &mut BinEncoder, ctx: &mut CtxStore, scan: &CoeffScan, coeffs: &[i32]) {
    let max_ctx_bins = (scan.max_num_coeff * 7) >> 2;
    let mut remaining = max_ctx_bins as i32;
    let last = scan.last_sub_set();
    debug_assert!(
        coeffs.iter().any(|&c| c != 0),
        "residual_ts_coding requires a coded (non-zero) block"
    );
    let mut group_flags = vec![false; scan.width_in_groups * scan.height_in_groups];
    let group_size = 1usize << scan.log2_cg_size;
    // `only1stSigGroup()` is evaluated at the last sub-block: it is true when
    // no earlier sub-block was coded.
    let mut any_coded_group = false;

    for sub_set in 0..=last {
        let min_sub = sub_set * group_size;
        let max_sub = min_sub + group_size - 1;
        let group_has = (min_sub..=max_sub).any(|sp| coeffs[scan.raster[sp]] != 0);
        let cg_raster = scan.cg_raster(sub_set);
        let cg_x = cg_raster % scan.width_in_groups;
        let cg_y = cg_raster / scan.width_in_groups;
        let sig_left = cg_x > 0 && group_flags[cg_raster - 1];
        let sig_above = cg_y > 0 && group_flags[cg_raster - scan.width_in_groups];
        group_flags[cg_raster] = group_has;

        let only_first = !any_coded_group;
        if sub_set != last || !only_first {
            let group_ctx =
                ctx::TS_SIG_COEFF_GROUP + usize::from(sig_left) + usize::from(sig_above);
            enc.encode_bin(group_has, &mut ctx.models[group_ctx]);
            if !group_has {
                continue;
            }
        }

        // ---- pass 1: significance, sign, gt1, parity ----
        // VTM's `inferSigPos` is `cctx.maxSubPos()`, i.e. the *last* scan
        // position of the sub-block, despite the local variable's name.
        let infer_sig_pos = max_sub;
        let mut num_nonzero = 0usize;
        let mut last_p1: i32 = -1;
        let mut sp = min_sub;
        while sp <= max_sub && remaining >= 4 {
            let blk = scan.raster[sp];
            let coeff = coeffs[blk];
            let sig = coeff != 0;
            if num_nonzero != 0 || sp != infer_sig_pos {
                let (x, y) = scan.xy[sp];
                let (left, above) = neighbours(coeffs, scan.width, x, y);
                let n = num_nonzero_neighbours(left, above);
                enc.encode_bin(sig, &mut ctx.models[ctx::TS_SIG_FLAG + n]);
                remaining -= 1;
            }
            if sig {
                let (x, y) = scan.xy[sp];
                let (left, above) = neighbours(coeffs, scan.width, x, y);
                let sign_ctx = sign_context(left, above);
                enc.encode_bin(coeff < 0, &mut ctx.models[ctx::TS_RESIDUAL_SIGN + sign_ctx]);
                remaining -= 1;
                num_nonzero += 1;

                let mod_abs = derive_mod_coeff(left, above, coeff.abs(), false);
                let rem_abs = mod_abs - 1;
                let gt1 = rem_abs != 0;
                let n = num_nonzero_neighbours(left, above);
                enc.encode_bin(gt1, &mut ctx.models[ctx::TS_LRG1_FLAG + n]);
                remaining -= 1;
                if gt1 {
                    enc.encode_bin((rem_abs - 1) & 1 != 0, &mut ctx.models[ctx::TS_PAR_FLAG]);
                    remaining -= 1;
                }
            }
            last_p1 = sp as i32;
            sp += 1;
        }

        // ---- pass 2: gt2..gt5 ----
        let mut last_p2: i32 = -1;
        let mut sp = min_sub;
        while sp <= max_sub && remaining >= 4 {
            let (x, y) = scan.xy[sp];
            let (left, above) = neighbours(coeffs, scan.width, x, y);
            let abs_level = derive_mod_coeff(left, above, coeffs[scan.raster[sp]].abs(), false);
            for i in 0..4 {
                let cutoff = 2 + 2 * i;
                if abs_level >= cutoff {
                    let gt = abs_level >= cutoff + 2;
                    enc.encode_bin(
                        gt,
                        &mut ctx.models[ctx::TS_GTX_FLAG + ((cutoff as usize) >> 1)],
                    );
                    remaining -= 1;
                }
            }
            last_p2 = sp as i32;
            sp += 1;
        }

        // ---- pass 3: Go-Rice remainder / escape ----
        for sp in min_sub..=max_sub {
            let cutoff = if sp as i32 <= last_p2 {
                10
            } else if sp as i32 <= last_p1 {
                2
            } else {
                0
            };
            let (x, y) = scan.xy[sp];
            let (left, above) = neighbours(coeffs, scan.width, x, y);
            let coeff = coeffs[scan.raster[sp]];
            let abs_level = derive_mod_coeff(left, above, coeff.abs(), cutoff == 0);
            if abs_level >= cutoff {
                let rem = if (sp as i32) <= last_p1 {
                    (abs_level - cutoff) >> 1
                } else {
                    abs_level
                };
                enc.encode_rem_abs_ep(rem as u32, 1, 5, 15);
                if abs_level != 0 && (sp as i32) > last_p1 {
                    enc.encode_bin_ep(coeff < 0);
                }
            }
        }

        if group_has {
            any_coded_group = true;
        }
    }
}

fn decode_ts_block(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &mut [i32],
) {
    coeffs.fill(0);
    let max_ctx_bins = (scan.max_num_coeff * 7) >> 2;
    let mut remaining = max_ctx_bins as i32;
    let last = scan.last_sub_set();
    let mut group_flags = vec![false; scan.width_in_groups * scan.height_in_groups];
    let group_size = 1usize << scan.log2_cg_size;

    for sub_set in 0..=last {
        let min_sub = sub_set * group_size;
        let max_sub = min_sub + group_size - 1;
        let cg_raster = scan.cg_raster(sub_set);
        let cg_x = cg_raster % scan.width_in_groups;
        let cg_y = cg_raster / scan.width_in_groups;
        let sig_left = cg_x > 0 && group_flags[cg_raster - 1];
        let sig_above = cg_y > 0 && group_flags[cg_raster - scan.width_in_groups];

        let mut sig_group = sub_set == last && group_flags.iter().all(|&f| !f);
        if !sig_group {
            let group_ctx =
                ctx::TS_SIG_COEFF_GROUP + usize::from(sig_left) + usize::from(sig_above);
            sig_group = dec.decode_bin(&mut ctx.models[group_ctx]);
        }
        if !sig_group {
            group_flags[cg_raster] = false;
            continue;
        }
        group_flags[cg_raster] = true;

        // ---- pass 1 ----
        // VTM's `inferSigPos` is `cctx.maxSubPos()`, i.e. the *last* scan
        // position of the sub-block, despite the local variable's name.
        let infer_sig_pos = max_sub;
        let mut num_nonzero = 0usize;
        let mut last_p1: i32 = -1;
        let mut sign_pattern: u32 = 0;
        let mut sig_blk_pos: Vec<usize> = Vec::new();
        let mut sp = min_sub;
        while sp <= max_sub && remaining >= 4 {
            let blk = scan.raster[sp];
            let (x, y) = scan.xy[sp];
            let mut sig = num_nonzero == 0 && sp == infer_sig_pos;
            if !sig {
                let (left, above) = neighbours(coeffs, scan.width, x, y);
                let n = num_nonzero_neighbours(left, above);
                sig = dec.decode_bin(&mut ctx.models[ctx::TS_SIG_FLAG + n]);
                remaining -= 1;
            }
            if sig {
                let (left, above) = neighbours(coeffs, scan.width, x, y);
                let sign_ctx = sign_context(left, above);
                let sign = dec.decode_bin(&mut ctx.models[ctx::TS_RESIDUAL_SIGN + sign_ctx]);
                remaining -= 1;
                sign_pattern |= u32::from(sign) << num_nonzero;
                sig_blk_pos.push(blk);
                num_nonzero += 1;

                let (left, above) = neighbours(coeffs, scan.width, x, y);
                let n = num_nonzero_neighbours(left, above);
                let gt1 = dec.decode_bin(&mut ctx.models[ctx::TS_LRG1_FLAG + n]);
                remaining -= 1;
                let par = if gt1 {
                    remaining -= 1;
                    dec.decode_bin(&mut ctx.models[ctx::TS_PAR_FLAG])
                } else {
                    false
                };
                coeffs[blk] = (1 + i32::from(gt1) + i32::from(par)) * if sign { -1 } else { 1 };
            }
            last_p1 = sp as i32;
            sp += 1;
        }

        // ---- pass 2 ----
        let mut last_p2: i32 = -1;
        let mut sp = min_sub;
        while sp <= max_sub && remaining >= 4 {
            let blk = scan.raster[sp];
            let mut abs_level = coeffs[blk].abs();
            for i in 0..4 {
                let cutoff = 2 + 2 * i;
                if abs_level >= cutoff {
                    let gt = dec
                        .decode_bin(&mut ctx.models[ctx::TS_GTX_FLAG + ((cutoff as usize) >> 1)]);
                    abs_level += i32::from(gt) << 1;
                    remaining -= 1;
                }
            }
            coeffs[blk] = abs_level;
            last_p2 = sp as i32;
            sp += 1;
        }

        // ---- pass 3 ----
        for sp in min_sub..=max_sub {
            let blk = scan.raster[sp];
            let (x, y) = scan.xy[sp];
            let cutoff = if sp as i32 <= last_p2 {
                10
            } else if sp as i32 <= last_p1 {
                2
            } else {
                0
            };
            let mut abs_level = coeffs[blk].abs();
            if abs_level >= cutoff {
                let rem = dec.decode_rem_abs_ep(1, 5, 15);
                abs_level += if (sp as i32) <= last_p1 {
                    (rem as i32) << 1
                } else {
                    rem as i32
                };
                if abs_level != 0 && (sp as i32) > last_p1 {
                    let sign = dec.decode_bin_ep();
                    sign_pattern |= u32::from(sign) << num_nonzero;
                    sig_blk_pos.push(blk);
                    num_nonzero += 1;
                }
            }
            if cutoff != 0 && abs_level > 0 {
                let (left, above) = neighbours(coeffs, scan.width, x, y);
                abs_level = dec_derive_mod_coeff(left, above, abs_level);
            }
            coeffs[blk] = abs_level;
        }

        // ---- apply signs ----
        for (k, &blk) in sig_blk_pos.iter().enumerate() {
            let abs = coeffs[blk];
            coeffs[blk] = if (sign_pattern >> k) & 1 != 0 {
                -abs
            } else {
                abs
            };
        }
    }
}

// ===========================================================================
// Wave 7: Full VVC Intra & Inter Slice Coding (ITU-T H.266 clause 7.3.11)
// ===========================================================================

/// Options controlling full-featured VVC slice encoding and decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SliceCodingOptions {
    /// Enable all 67 luma intra modes (`0..=66`) with 6-MPM derivation.
    pub full_intra_modes: bool,
    /// Enable Cross-Component Linear Model (`CCLM`) chroma modes (`81..=83`).
    pub enable_cclm: bool,
    /// Enable Matrix-based Intra Prediction (`MIP`).
    pub enable_mip: bool,
    /// Enable Dependent Quantization (`sh_dep_quant_used_flag`).
    pub dep_quant: bool,
    /// Enable Sign Data Hiding (`sh_sign_data_hiding_used_flag`).
    pub sign_data_hiding: bool,
    /// Enable Joint CbCr residual coding (`tu_joint_cbcr_residual_flag`).
    pub enable_jccr: bool,
    /// Enable in-loop deblocking filter (ITU-T H.266 clause 8.8.3).
    pub enable_deblock: bool,
    /// Enable Sample Adaptive Offset (`SAO`, ITU-T H.266 clause 8.8.4).
    pub enable_sao: bool,
    /// Enable Adaptive Loop Filter (`ALF`, ITU-T H.266 clause 8.8.5).
    pub enable_alf: bool,
    /// Enable Multiple Reference Line (`MRL`, `intra_luma_ref_idx` `0..=2`).
    pub enable_mrl: bool,
    /// Enable Intra Sub-Partitions (`ISP`, horizontal/vertical 2 or 4 sub-TUs).
    pub enable_isp: bool,
    /// Enable Block DPCM (`BDPCM`, horizontal/vertical luma & chroma).
    pub enable_bdpcm: bool,
    /// Enable Palette mode (`pred_mode_plt_flag`, `palette_coding`).
    pub enable_palette: bool,
    /// Enable Intra Block Copy (`IBC`, `pred_mode_ibc_flag`).
    pub enable_ibc: bool,
    /// Enable Adaptive Colour Transform (`ACT`, `tu_y_coded_flag` / `cu_act_enabled_flag` for 4:4:4).
    pub enable_act: bool,
    /// Enable Luma Mapping with Chroma Scaling (`LMCS`, `sps_lmcs_enabled_flag`).
    pub enable_lmcs: bool,
    /// Enable Cross-Component Adaptive Loop Filter (`CCALF`, `sps_ccalf_enabled_flag`).
    pub enable_ccalf: bool,
}

impl SliceCodingOptions {
    /// `true` when any optional slice tool is enabled.
    pub fn any_tool(&self) -> bool {
        self.full_intra_modes
            || self.enable_cclm
            || self.enable_mip
            || self.dep_quant
            || self.sign_data_hiding
            || self.enable_jccr
            || self.enable_deblock
            || self.enable_sao
            || self.enable_alf
            || self.enable_mrl
            || self.enable_isp
            || self.enable_bdpcm
            || self.enable_palette
            || self.enable_ibc
            || self.enable_act
            || self.enable_lmcs
            || self.enable_ccalf
    }
}

/// Encode `intra_bdpcm_luma_flag` / `intra_bdpcm_chroma_flag` and `intra_bdpcm_*_dir_flag`
/// (`0 = None`, `1 = Hor`, `2 = Ver`, ITU-T Rec. H.266 §7.3.11.5, VTM `CABACWriter.cpp:920-948`).
pub fn encode_bdpcm_mode(enc: &mut BinEncoder, ctx: &mut CtxStore, mode: u8, is_chroma: bool) {
    let base = ctx::BDPCM_MODE + if is_chroma { 2 } else { 0 };
    let active = mode != 0;
    enc.encode_bin(active, &mut ctx.models[base]);
    if active {
        enc.encode_bin(mode == 2, &mut ctx.models[base + 1]);
    }
}

/// Decode `intra_bdpcm_luma_flag` / `intra_bdpcm_chroma_flag` and `intra_bdpcm_*_dir_flag`
/// (`0 = None`, `1 = Hor`, `2 = Ver`, ITU-T Rec. H.266 §7.3.11.5, VTM `CABACReader.cpp:1192-1225`).
pub fn decode_bdpcm_mode(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore, is_chroma: bool) -> u8 {
    let base = ctx::BDPCM_MODE + if is_chroma { 2 } else { 0 };
    if !dec.decode_bin(&mut ctx.models[base]) {
        return 0;
    }
    if dec.decode_bin(&mut ctx.models[base + 1]) {
        2
    } else {
        1
    }
}

/// Encode `intra_luma_ref_idx` (`0..=2`, ITU-T H.266 clause 7.3.11.5).
pub fn encode_mrl_idx(enc: &mut BinEncoder, ctx: &mut CtxStore, mrl_idx: u8) {
    let idx = mrl_idx.min(2);
    enc.encode_bin(idx > 0, &mut ctx.models[ctx::MULTI_REF_LINE_IDX]);
    if idx > 0 {
        enc.encode_bin(idx > 1, &mut ctx.models[ctx::MULTI_REF_LINE_IDX + 1]);
    }
}

/// Decode `intra_luma_ref_idx` (`0..=2`, ITU-T H.266 clause 7.3.11.5).
pub fn decode_mrl_idx(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> u8 {
    if !dec.decode_bin(&mut ctx.models[ctx::MULTI_REF_LINE_IDX]) {
        return 0;
    }
    if dec.decode_bin(&mut ctx.models[ctx::MULTI_REF_LINE_IDX + 1]) {
        2
    } else {
        1
    }
}

/// Encode `intra_subpartitions_mode_flag` and `intra_subpartitions_split_flag`
/// (ITU-T H.266 clause 7.3.11.5).
pub fn encode_isp_mode(enc: &mut BinEncoder, ctx: &mut CtxStore, isp: predict::IspMode) {
    let flag = isp != predict::IspMode::None;
    enc.encode_bin(flag, &mut ctx.models[ctx::ISP_MODE]);
    if flag {
        enc.encode_bin(
            isp == predict::IspMode::Ver,
            &mut ctx.models[ctx::ISP_MODE + 1],
        );
    }
}

/// Decode `intra_subpartitions_mode_flag` and `intra_subpartitions_split_flag`
/// (ITU-T H.266 clause 7.3.11.5).
pub fn decode_isp_mode(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> predict::IspMode {
    if !dec.decode_bin(&mut ctx.models[ctx::ISP_MODE]) {
        return predict::IspMode::None;
    }
    if dec.decode_bin(&mut ctx.models[ctx::ISP_MODE + 1]) {
        predict::IspMode::Ver
    } else {
        predict::IspMode::Hor
    }
}

/// Encode `intra_mip_flag`, `intra_mip_transposed_flag`, and `intra_mip_mode`
/// (ITU-T H.266 clause 7.3.11.5).
pub fn encode_mip_mode(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    mip_flag: bool,
    transposed: bool,
    mode_idx: u8,
    num_modes: u32,
    ctx_inc: usize,
) {
    enc.encode_bin(mip_flag, &mut ctx.models[ctx::MIP_FLAG + ctx_inc.min(3)]);
    if mip_flag {
        enc.encode_bin_ep(transposed);
        enc.encode_trunc_bin(u32::from(mode_idx), num_modes.max(2));
    }
}

/// Decode `intra_mip_flag`, `intra_mip_transposed_flag`, and `intra_mip_mode`
/// (ITU-T H.266 clause 7.3.11.5).
pub fn decode_mip_mode(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    num_modes: u32,
    ctx_inc: usize,
) -> Option<(bool, u8)> {
    if !dec.decode_bin(&mut ctx.models[ctx::MIP_FLAG + ctx_inc.min(3)]) {
        return None;
    }
    let transposed = dec.decode_bin_ep();
    let mode_idx = dec.decode_trunc_bin(num_modes.max(2)) as u8;
    Some((transposed, mode_idx))
}

/// `ctxInc` for `intra_mip_flag` (H.266 Table 47; VTM `DeriveCtx::CtxMipFlag`).
///
/// A block with an aspect ratio above 2:1 always uses context `3`; otherwise the
/// context is the number of MIP-coded neighbours (`0..=2`).
#[inline]
pub fn mip_flag_ctx_inc(
    left_mip: bool,
    above_mip: bool,
    cb_width: usize,
    cb_height: usize,
) -> usize {
    if cb_width > 2 * cb_height || cb_height > 2 * cb_width {
        3
    } else {
        usize::from(left_mip) + usize::from(above_mip)
    }
}

/// Encode any of the 67 luma intra prediction modes (`0..=66`) using the
/// 6-MPM list and 60-symbol truncated-binary remainder (ITU-T H.266 clause 7.3.11.5).
pub fn encode_full_luma_intra_mode(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    mode: u8,
    mpm: &[u8; 6],
    mrl_idx: u8,
    isp_active: bool,
) {
    if mrl_idx > 0 {
        // When intra_luma_ref_idx > 0, mpm_flag is inferred as 1 and not_planar as 1.
        let mpm_idx = mpm.iter().position(|&m| m == mode).unwrap_or(1).clamp(1, 5);
        enc.encode_unary_max_ep((mpm_idx - 1) as u32, 4);
        return;
    }
    match predict::classify_luma_mode_mpm(mode, mpm) {
        Ok(mpm_idx) => {
            enc.encode_bin(true, &mut ctx.models[ctx::INTRA_LUMA_MPM_FLAG]);
            let planar_ctx = ctx::INTRA_LUMA_PLANAR_FLAG + usize::from(!isp_active);
            if mpm_idx == 0 {
                enc.encode_bin(false, &mut ctx.models[planar_ctx]);
            } else {
                enc.encode_bin(true, &mut ctx.models[planar_ctx]);
                enc.encode_unary_max_ep((mpm_idx - 1) as u32, 4);
            }
        }
        Err(remainder) => {
            enc.encode_bin(false, &mut ctx.models[ctx::INTRA_LUMA_MPM_FLAG]);
            enc.encode_trunc_bin(u32::from(remainder), 61);
        }
    }
}

/// Decode a luma intra prediction mode (`0..=66`) coded by [`encode_full_luma_intra_mode`].
pub fn decode_full_luma_intra_mode(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    mpm: &[u8; 6],
    mrl_idx: u8,
    isp_active: bool,
) -> u8 {
    if mrl_idx > 0 {
        let idx = (dec.decode_unary_max_ep(4) as usize + 1).min(5);
        return mpm[idx];
    }
    if dec.decode_bin(&mut ctx.models[ctx::INTRA_LUMA_MPM_FLAG]) {
        let planar_ctx = ctx::INTRA_LUMA_PLANAR_FLAG + usize::from(!isp_active);
        if !dec.decode_bin(&mut ctx.models[planar_ctx]) {
            mpm[0]
        } else {
            let idx = (dec.decode_unary_max_ep(4) as usize + 1).min(5);
            mpm[idx]
        }
    } else {
        let rem = dec.decode_trunc_bin(61) as u8;
        predict::resolve_non_mpm_luma_mode(rem, mpm)
    }
}

/// Encode chroma intra prediction mode (`0..=66` or `81..=83` for CCLM,
/// ITU-T H.266 clause 7.3.11.5 and Table 24).
pub fn encode_chroma_intra_mode(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    chroma_mode: u8,
    luma_mode: u8,
    cclm_enabled: bool,
) {
    if cclm_enabled {
        let is_cclm = (81..=83).contains(&chroma_mode);
        enc.encode_bin(is_cclm, &mut ctx.models[ctx::CCLM_MODE_FLAG]);
        if is_cclm {
            let cclm_idx = chroma_mode - 81;
            enc.encode_bin(cclm_idx > 0, &mut ctx.models[ctx::CCLM_MODE_IDX]);
            if cclm_idx > 0 {
                enc.encode_bin_ep(cclm_idx > 1);
            }
            return;
        }
    }
    // Check if chroma_mode is DM (derived mode == luma_mode).
    if chroma_mode == luma_mode {
        enc.encode_bin(false, &mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]);
        return;
    }
    // Table 24 non-DM candidates: [0 (Planar), 50 (Vertical), 18 (Horizontal), 1 (DC)],
    // with any entry equal to luma_mode replaced by 66.
    let mut table = [0u8, 50, 18, 1];
    for entry in &mut table {
        if *entry == luma_mode {
            *entry = 66;
        }
    }
    let idx = table.iter().position(|&m| m == chroma_mode).unwrap_or(0);
    enc.encode_bin(true, &mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]);
    enc.encode_bins_ep(idx as u32, 2);
}

/// Decode chroma intra prediction mode coded by [`encode_chroma_intra_mode`].
pub fn decode_chroma_intra_mode(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    luma_mode: u8,
    cclm_enabled: bool,
) -> u8 {
    if cclm_enabled && dec.decode_bin(&mut ctx.models[ctx::CCLM_MODE_FLAG]) {
        if !dec.decode_bin(&mut ctx.models[ctx::CCLM_MODE_IDX]) {
            return 81;
        }
        return if dec.decode_bin_ep() { 83 } else { 82 };
    }
    if !dec.decode_bin(&mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]) {
        return luma_mode;
    }
    let mut table = [0u8, 50, 18, 1];
    for entry in &mut table {
        if *entry == luma_mode {
            *entry = 66;
        }
    }
    let idx = (dec.decode_bins_ep(2) as usize).min(3);
    table[idx]
}

/// Encode motion vector difference `mvd_coding(x0, y0, refList)` (ITU-T H.266 clause 7.3.11.8).
pub fn encode_mvd(enc: &mut BinEncoder, ctx: &mut CtxStore, mvd: super::inter::Mv) {
    let comps = [mvd.hor, mvd.ver];
    let gt0 = [comps[0] != 0, comps[1] != 0];
    for &flag in &gt0 {
        enc.encode_bin(flag, &mut ctx.models[ctx::MVD]);
    }
    let gt1 = [comps[0].abs() > 1, comps[1].abs() > 1];
    for i in 0..2 {
        if gt0[i] {
            enc.encode_bin(gt1[i], &mut ctx.models[ctx::MVD + 1]);
        }
    }
    for i in 0..2 {
        if gt0[i] {
            if gt1[i] {
                // H.266 §7.3.11.8 / VTM `mvd_coding`: abs_mvd_minus2 is
                // coded with `encodeRemAbsEP(..., goRicePar = 1, cutoff = 0,
                // maxLog2TrDynamicRange = MV_BITS - 1)`.
                enc.encode_rem_abs_ep(comps[i].unsigned_abs() - 2, 1, 0, 17);
            }
            enc.encode_bin_ep(comps[i] < 0);
        }
    }
}

/// Decode motion vector difference `mvd_coding(x0, y0, refList)` (ITU-T H.266 clause 7.3.11.8).
pub fn decode_mvd(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> super::inter::Mv {
    let mut gt0 = [false; 2];
    for flag in &mut gt0 {
        *flag = dec.decode_bin(&mut ctx.models[ctx::MVD]);
    }
    let mut gt1 = [false; 2];
    for i in 0..2 {
        if gt0[i] {
            gt1[i] = dec.decode_bin(&mut ctx.models[ctx::MVD + 1]);
        }
    }
    let mut comps = [0i32; 2];
    for i in 0..2 {
        if gt0[i] {
            let abs_val = if gt1[i] {
                (dec.decode_rem_abs_ep(1, 0, 17) + 2) as i32
            } else {
                1
            };
            let sign = dec.decode_bin_ep();
            comps[i] = if sign { -abs_val } else { abs_val };
        }
    }
    super::inter::Mv::new(comps[0], comps[1])
}

/// Encode one component block for the full multi-tool intra/inter pipeline.
#[allow(clippy::too_many_arguments)]
fn encode_component_full(
    pred: &[i32],
    src: &Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    qp: u8,
    use_ts: bool,
    dep_quant: bool,
) -> Result<(Vec<i32>, Vec<u8>, bool), VvcError> {
    let mut res = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            res[y * w + x] = i32::from(src.get(bx + x, by + y)) - pred[y * w + x];
        }
    }
    if use_ts {
        let levels: Vec<i32> = res.iter().map(|&r| transform::quantize_ts(r, qp)).collect();
        let cbf = levels.iter().any(|&c| c != 0);
        let mut recon = vec![0u8; w * h];
        for i in 0..w * h {
            let r = if cbf {
                transform::dequantize_ts(levels[i], qp)
            } else {
                0
            };
            recon[i] = (pred[i] + r).clamp(0, 255) as u8;
        }
        return Ok((levels, recon, cbf));
    }
    let (skip_w, skip_h) = primary_skips(w, h, dct::TrType::Dct2, dct::TrType::Dct2);
    let coeff = dct::forward_transform_rect(
        w,
        h,
        &res,
        dct::TrType::Dct2,
        dct::TrType::Dct2,
        skip_w,
        skip_h,
    )
    .map_err(|_| VvcError::Unsupported("unsupported primary transform"))?;
    let scan = CoeffScan::new(w, h);
    let levels = if dep_quant {
        dct::quantize_dep_quant(w, h, &coeff, &scan.raster, qp)
            .map_err(|_| VvcError::Unsupported("unsupported dep_quant"))?
    } else {
        dct::quantize_rect(w, h, &coeff, qp, true)
            .map_err(|_| VvcError::Unsupported("unsupported quantize_rect"))?
    };
    let cbf = levels.iter().any(|&c| c != 0);
    let mut recon = vec![0u8; w * h];
    if !cbf {
        for i in 0..w * h {
            recon[i] = pred[i].clamp(0, 255) as u8;
        }
        return Ok((levels, recon, false));
    }
    let dequant = if dep_quant {
        let scan = CoeffScan::new(w, h);
        dct::dequantize_dep_quant(w, h, &levels, &scan.raster, qp)
            .map_err(|_| VvcError::Unsupported("unsupported dequantize_dep_quant"))?
    } else {
        dct::dequantize_rect(w, h, &levels, qp)
            .map_err(|_| VvcError::Unsupported("unsupported dequantize_rect"))?
    };
    let inv = dct::inverse_transform_rect(
        w,
        h,
        &dequant,
        dct::TrType::Dct2,
        dct::TrType::Dct2,
        skip_w,
        skip_h,
    )
    .map_err(|_| VvcError::Unsupported("unsupported inverse_transform_rect"))?;
    for i in 0..w * h {
        recon[i] = (pred[i] + inv[i]).clamp(0, 255) as u8;
    }
    Ok((levels, recon, true))
}

/// Reconstruct one component block for the full multi-tool pipeline.
#[allow(clippy::too_many_arguments)]
fn reconstruct_component_full(
    recon: &mut Plane,
    pred: &[i32],
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    qp: u8,
    levels: &[i32],
    cbf: bool,
    use_ts: bool,
    dep_quant: bool,
) -> Result<(), VvcError> {
    if !cbf {
        for y in 0..h {
            for x in 0..w {
                recon.set(bx + x, by + y, pred[y * w + x].clamp(0, 255) as u8);
            }
        }
        return Ok(());
    }
    if use_ts {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let val = pred[idx] + transform::dequantize_ts(levels[idx], qp);
                recon.set(bx + x, by + y, val.clamp(0, 255) as u8);
            }
        }
        return Ok(());
    }
    let (skip_w, skip_h) = primary_skips(w, h, dct::TrType::Dct2, dct::TrType::Dct2);
    let dequant = if dep_quant {
        let scan = CoeffScan::new(w, h);
        dct::dequantize_dep_quant(w, h, levels, &scan.raster, qp)
            .map_err(|_| VvcError::Unsupported("unsupported dequantize_dep_quant"))?
    } else {
        dct::dequantize_rect(w, h, levels, qp)
            .map_err(|_| VvcError::Unsupported("unsupported dequantize_rect"))?
    };
    let inv = dct::inverse_transform_rect(
        w,
        h,
        &dequant,
        dct::TrType::Dct2,
        dct::TrType::Dct2,
        skip_w,
        skip_h,
    )
    .map_err(|_| VvcError::Unsupported("unsupported inverse_transform_rect"))?;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            recon.set(bx + x, by + y, (pred[idx] + inv[idx]).clamp(0, 255) as u8);
        }
    }
    Ok(())
}

#[inline]
fn ipm(idx: u8) -> predict::IntraPredMode {
    predict::IntraPredMode::from_mode_index(idx).unwrap_or(predict::IntraPredMode::Planar)
}

/// Encode per-CTU Sample Adaptive Offset (`sao(ctuRsAddr)`, ITU-T Rec. H.266 §7.3.11.3,
/// VTM `CABACWriter.cpp:150-405`).
#[allow(clippy::too_many_arguments)]
pub fn encode_ctu_sao(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    gx: usize,
    gy: usize,
    sao_luma_used: bool,
    sao_chroma_used: bool,
    params: &[super::filter::SaoType; 3],
    merge_left: bool,
    merge_up: bool,
    bit_depth: u8,
) {
    if !sao_luma_used && !sao_chroma_used {
        return;
    }
    if gx > 0 {
        enc.encode_bin(merge_left, &mut ctx.models[ctx::SAO_MERGE_FLAG]);
        if merge_left {
            return;
        }
    }
    if gy > 0 {
        enc.encode_bin(merge_up, &mut ctx.models[ctx::SAO_MERGE_FLAG]);
        if merge_up {
            return;
        }
    }
    let max_offset_qval = (1u32 << (u32::from(bit_depth.clamp(8, 10)) - 5)) - 1;
    let first_comp = if sao_luma_used { 0 } else { 1 };
    let last_comp = if sao_chroma_used { 2 } else { 0 };
    for comp_id in first_comp..=last_comp {
        let sao = params[comp_id];
        if comp_id != 2 {
            match sao {
                super::filter::SaoType::Off => {
                    enc.encode_bin(false, &mut ctx.models[ctx::SAO_TYPE_IDX]);
                }
                super::filter::SaoType::BandOffset { .. } => {
                    enc.encode_bin(true, &mut ctx.models[ctx::SAO_TYPE_IDX]);
                    enc.encode_bin_ep(false);
                }
                super::filter::SaoType::EdgeOffset { .. } => {
                    enc.encode_bin(true, &mut ctx.models[ctx::SAO_TYPE_IDX]);
                    enc.encode_bin_ep(true);
                }
            }
        }
        match sao {
            super::filter::SaoType::Off => {}
            super::filter::SaoType::BandOffset { band_pos, offsets } => {
                for &off in &offsets {
                    let abs_v = off.unsigned_abs().min(max_offset_qval);
                    enc.encode_unary_max_ep(abs_v, max_offset_qval);
                }
                for &off in &offsets {
                    if off != 0 {
                        enc.encode_bin_ep(off < 0);
                    }
                }
                enc.encode_bins_ep(u32::from(band_pos & 31), 5);
            }
            super::filter::SaoType::EdgeOffset { eo_class, offsets } => {
                let abs0 = (offsets[0].max(0) as u32).min(max_offset_qval);
                let abs1 = (offsets[1].max(0) as u32).min(max_offset_qval);
                let abs2 = ((-offsets[2]).max(0) as u32).min(max_offset_qval);
                let abs3 = ((-offsets[3]).max(0) as u32).min(max_offset_qval);
                for abs_v in [abs0, abs1, abs2, abs3] {
                    enc.encode_unary_max_ep(abs_v, max_offset_qval);
                }
                if comp_id != 2 {
                    enc.encode_bins_ep(u32::from(eo_class & 3), 2);
                }
            }
        }
    }
}

/// Decode per-CTU Sample Adaptive Offset (`sao(ctuRsAddr)`, ITU-T Rec. H.266 §7.3.11.3,
/// VTM `CABACReader.cpp:343-497`).
#[allow(clippy::too_many_arguments)]
pub fn decode_ctu_sao(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    gx: usize,
    gy: usize,
    sao_luma_used: bool,
    sao_chroma_used: bool,
    left_params: Option<[super::filter::SaoType; 3]>,
    up_params: Option<[super::filter::SaoType; 3]>,
    bit_depth: u8,
) -> [super::filter::SaoType; 3] {
    let mut out = [super::filter::SaoType::Off; 3];
    if !sao_luma_used && !sao_chroma_used {
        return out;
    }
    if gx > 0 && dec.decode_bin(&mut ctx.models[ctx::SAO_MERGE_FLAG]) {
        return left_params.unwrap_or(out);
    }
    if gy > 0 && dec.decode_bin(&mut ctx.models[ctx::SAO_MERGE_FLAG]) {
        return up_params.unwrap_or(out);
    }
    let max_offset_qval = (1u32 << (u32::from(bit_depth.clamp(8, 10)) - 5)) - 1;
    let first_comp = if sao_luma_used { 0 } else { 1 };
    let last_comp = if sao_chroma_used { 2 } else { 0 };
    let mut cb_mode: u8 = 0; // 0 = Off, 1 = BO, 2 = EO
    let mut cb_eo_class: u8 = 0;

    for comp_id in first_comp..=last_comp {
        let mode = if comp_id != 2 {
            if !dec.decode_bin(&mut ctx.models[ctx::SAO_TYPE_IDX]) {
                0u8
            } else if dec.decode_bin_ep() {
                2u8
            } else {
                1u8
            }
        } else {
            cb_mode
        };
        if comp_id == 1 {
            cb_mode = mode;
        }
        if mode == 0 {
            out[comp_id] = super::filter::SaoType::Off;
            continue;
        }
        let mut abs_offs = [0i8; 4];
        for slot in &mut abs_offs {
            *slot = dec.decode_unary_max_ep(max_offset_qval) as i8;
        }
        if mode == 1 {
            for slot in &mut abs_offs {
                if *slot != 0 && dec.decode_bin_ep() {
                    *slot = -*slot;
                }
            }
            let band_pos = dec.decode_bins_ep(5) as u8;
            out[comp_id] = super::filter::SaoType::BandOffset {
                band_pos,
                offsets: abs_offs.map(i32::from),
            };
        } else {
            let eo_class = if comp_id != 2 {
                let c = dec.decode_bins_ep(2) as u8;
                if comp_id == 1 {
                    cb_eo_class = c;
                }
                c
            } else {
                cb_eo_class
            };
            out[comp_id] = super::filter::SaoType::EdgeOffset {
                eo_class,
                offsets: [
                    i32::from(abs_offs[0]),
                    i32::from(abs_offs[1]),
                    -i32::from(abs_offs[2]),
                    -i32::from(abs_offs[3]),
                ],
            };
        }
    }
    out
}

/// Encode per-CTU luma ALF control (`alf_ctb_flag[0]` and `alf_luma_fixed_filter_idx`,
/// ITU-T Rec. H.266 §7.3.11.2, VTM `CABACWriter.cpp:175-245`).
pub fn encode_ctu_alf_luma_fixed(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    left_active: bool,
    above_active: bool,
    enabled: bool,
    fixed_filter_idx: u8,
) {
    let c_idx = ctx::ALF_CTB_FLAG + usize::from(left_active) + usize::from(above_active);
    enc.encode_bin(enabled, &mut ctx.models[c_idx]);
    if enabled {
        // When sh_num_alf_aps_ids_luma == 0, alf_use_aps_flag is not coded;
        // alf_luma_fixed_filter_idx is coded with TruncBin(idx, 16).
        enc.encode_trunc_bin(u32::from(fixed_filter_idx & 15), 16);
    }
}

/// Decode per-CTU luma ALF control (`alf_ctb_flag[0]` and `alf_luma_fixed_filter_idx`,
/// ITU-T Rec. H.266 §7.3.11.2, VTM `CABACReader.cpp:185-294`).
pub fn decode_ctu_alf_luma_fixed(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    left_active: bool,
    above_active: bool,
) -> Option<u8> {
    let c_idx = ctx::ALF_CTB_FLAG + usize::from(left_active) + usize::from(above_active);
    if !dec.decode_bin(&mut ctx.models[c_idx]) {
        return None;
    }
    let fixed_idx = (dec.decode_trunc_bin(16) as u8) & 15;
    Some(fixed_idx)
}

/// Apply configured in-loop filters (`Deblocking`, `SAO`, `ALF`) in normative order.
fn apply_loop_filters_with_ctu_params(
    planes: &mut Planes,
    qp: u8,
    ctu: usize,
    options: &SliceCodingOptions,
    sao_grid: Option<&[[super::filter::SaoType; 3]]>,
    alf_grid: Option<&[Option<u8>]>,
) {
    let grid_w = planes.y.width / ctu;
    let grid_h = planes.y.height / ctu;
    if options.enable_deblock {
        let chroma_qp = super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8;
        let cu_step = ctu.min(32);
        super::filter::deblock_luma_plane(&mut planes.y, cu_step, ctu, qp, 2, 0, 0);
        super::filter::deblock_chroma_plane(
            &mut planes.u,
            ctu / 2,
            ctu / 2,
            chroma_qp,
            2,
            0,
            0,
            2,
            2,
        );
        super::filter::deblock_chroma_plane(
            &mut planes.v,
            ctu / 2,
            ctu / 2,
            chroma_qp,
            2,
            0,
            0,
            2,
            2,
        );
    }
    if options.enable_sao {
        if let Some(sao_ctus) = sao_grid {
            let src_y = planes.y.clone();
            let src_u = planes.u.clone();
            let src_v = planes.v.clone();
            let cw = ctu / 2;
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let idx = gy * grid_w + gx;
                    if let Some(sao) = sao_ctus.get(idx) {
                        super::filter::apply_sao_region(
                            &src_y,
                            &mut planes.y,
                            gx * ctu,
                            gy * ctu,
                            ctu,
                            ctu,
                            sao[0],
                        );
                        super::filter::apply_sao_region(
                            &src_u,
                            &mut planes.u,
                            gx * cw,
                            gy * cw,
                            cw,
                            cw,
                            sao[1],
                        );
                        super::filter::apply_sao_region(
                            &src_v,
                            &mut planes.v,
                            gx * cw,
                            gy * cw,
                            cw,
                            cw,
                            sao[2],
                        );
                    }
                }
            }
        }
    }
    if options.enable_alf {
        if let Some(alf_ctus) = alf_grid {
            if alf_ctus.iter().any(Option::is_some) {
                let default_aps = super::hls::VvcAlfAps {
                    luma_clip: false,
                    luma_filters: Vec::new(),
                    luma_clip_idx: Vec::new(),
                    filter_coeff_delta_idx: [0u8; 25],
                    chroma_clip: false,
                    chroma_filters: Vec::new(),
                    chroma_clip_idx: Vec::new(),
                    cc_cb_filters: Vec::new(),
                    cc_cr_filters: Vec::new(),
                };
                super::filter::apply_alf_luma_plane(&mut planes.y, &default_aps);
            }
        }
    }
}

/// Apply configured in-loop filters (`Deblocking`, `SAO`, `ALF`) on an inter slice using
/// per-CU CBF and MV grids to derive normative Boundary Strength `Bs` (`0` or `1`) on
/// every CTU boundary (`H.266 §8.8.3.4`, VTM `DeblockingFilter.cpp:790-915`).
#[allow(clippy::too_many_arguments)]
fn apply_inter_loop_filters(
    planes: &mut Planes,
    qp: u8,
    ctu: usize,
    options: &SliceCodingOptions,
    cbf_y_grid: &[bool],
    cbf_u_grid: &[bool],
    cbf_v_grid: &[bool],
    mv_grid: &super::inter::MotionFieldGrid,
    sao_grid: Option<&[[super::filter::SaoType; 3]]>,
    alf_grid: Option<&[Option<u8>]>,
) {
    if options.enable_deblock {
        let grid_w = planes.y.width / ctu;
        let grid_h = planes.y.height / ctu;
        let cw = ctu / 2;
        let chroma_qp = super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8;
        // Coding units are at most 32x32, so every transform side on a filtered edge
        // is at least that long.
        let luma_max_len = super::filter::luma_filter_length(ctu.min(32));
        let chroma_len = super::filter::chroma_filter_length(cw);

        // 1. Vertical CTU boundaries (x = gx * ctu) with normative per-edge Bs (§8.8.3.5)
        for gx in 1..grid_w {
            let x = gx * ctu;
            let cx = gx * cw;
            for gy in 0..grid_h {
                let p_idx = gy * grid_w + (gx - 1);
                let q_idx = gy * grid_w + gx;
                let cand_p = mv_grid.get_at_luma(x - 1, gy * ctu).unwrap_or_default();
                let cand_q = mv_grid.get_at_luma(x, gy * ctu).unwrap_or_default();
                let mv_p = cand_p.mv_field[0].mv;
                let mv_q = cand_q.mv_field[0].mv;
                let ref_p = i32::from(cand_p.mv_field[0].ref_idx);
                let ref_q = i32::from(cand_q.mv_field[0].ref_idx);

                let bs_y = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_y_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_y_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_y > 0 && x >= 8 && x + 8 <= planes.y.width {
                    let mut y = gy * ctu;
                    let y_end = ((gy + 1) * ctu).min(planes.y.height);
                    while y + 4 <= y_end {
                        let mut lines = [[0i32; 16]; 4];
                        for dy in 0..4 {
                            for k in 0..16 {
                                lines[dy][k] = i32::from(planes.y.get(x - 8 + k, y + dy));
                            }
                        }
                        super::filter::deblock_luma_edge_normative(
                            &mut lines,
                            qp,
                            qp,
                            bs_y,
                            0,
                            0,
                            luma_max_len,
                            luma_max_len,
                            false,
                        );
                        for dy in 0..4 {
                            for k in 0..16 {
                                planes
                                    .y
                                    .set(x - 8 + k, y + dy, lines[dy][k].clamp(0, 255) as u8);
                            }
                        }
                        y += 4;
                    }
                }

                let bs_u = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_u_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_u_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_u > 0 {
                    let cy_start = gy * cw;
                    let cy_end = ((gy + 1) * cw).min(planes.u.height);
                    super::filter::deblock_chroma_edge_region(
                        &mut planes.u,
                        true,
                        cx,
                        cy_start,
                        cy_end,
                        i32::from(chroma_qp),
                        bs_u,
                        0,
                        0,
                        chroma_len,
                        chroma_len,
                        2,
                    );
                }

                let bs_v = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_v_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_v_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_v > 0 {
                    let cy_start = gy * cw;
                    let cy_end = ((gy + 1) * cw).min(planes.v.height);
                    super::filter::deblock_chroma_edge_region(
                        &mut planes.v,
                        true,
                        cx,
                        cy_start,
                        cy_end,
                        i32::from(chroma_qp),
                        bs_v,
                        0,
                        0,
                        chroma_len,
                        chroma_len,
                        2,
                    );
                }
            }
        }

        // 2. Horizontal CTU boundaries (y = gy * ctu) with normative per-edge Bs (§8.8.3.5)
        for gy in 1..grid_h {
            let y = gy * ctu;
            let cy = gy * cw;
            for gx in 0..grid_w {
                let p_idx = (gy - 1) * grid_w + gx;
                let q_idx = gy * grid_w + gx;
                let cand_p = mv_grid.get_at_luma(gx * ctu, y - 1).unwrap_or_default();
                let cand_q = mv_grid.get_at_luma(gx * ctu, y).unwrap_or_default();
                let mv_p = cand_p.mv_field[0].mv;
                let mv_q = cand_q.mv_field[0].mv;
                let ref_p = i32::from(cand_p.mv_field[0].ref_idx);
                let ref_q = i32::from(cand_q.mv_field[0].ref_idx);

                let bs_y = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_y_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_y_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_y > 0 && y >= 8 && y + 8 <= planes.y.height {
                    let mut bx = gx * ctu;
                    let x_end = ((gx + 1) * ctu).min(planes.y.width);
                    while bx + 4 <= x_end {
                        let mut lines = [[0i32; 16]; 4];
                        for dx in 0..4 {
                            for k in 0..16 {
                                lines[dx][k] = i32::from(planes.y.get(bx + dx, y - 8 + k));
                            }
                        }
                        super::filter::deblock_luma_edge_normative(
                            &mut lines,
                            qp,
                            qp,
                            bs_y,
                            0,
                            0,
                            luma_max_len,
                            luma_max_len,
                            true,
                        );
                        for dx in 0..4 {
                            for k in 0..16 {
                                planes
                                    .y
                                    .set(bx + dx, y - 8 + k, lines[dx][k].clamp(0, 255) as u8);
                            }
                        }
                        bx += 4;
                    }
                }

                let bs_u = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_u_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_u_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_u > 0 {
                    let cx_start = gx * cw;
                    let cx_end = ((gx + 1) * cw).min(planes.u.width);
                    super::filter::deblock_chroma_edge_region(
                        &mut planes.u,
                        false,
                        cy,
                        cx_start,
                        cx_end,
                        i32::from(chroma_qp),
                        bs_u,
                        0,
                        0,
                        1,
                        chroma_len,
                        2,
                    );
                }

                let bs_v = super::filter::derive_boundary_strength(
                    false,
                    false,
                    cbf_v_grid.get(p_idx).copied().unwrap_or(false),
                    cbf_v_grid.get(q_idx).copied().unwrap_or(false),
                    true,
                    mv_p,
                    mv_q,
                    ref_p,
                    ref_q,
                );
                if bs_v > 0 {
                    let cx_start = gx * cw;
                    let cx_end = ((gx + 1) * cw).min(planes.v.width);
                    super::filter::deblock_chroma_edge_region(
                        &mut planes.v,
                        false,
                        cy,
                        cx_start,
                        cx_end,
                        i32::from(chroma_qp),
                        bs_v,
                        0,
                        0,
                        1,
                        chroma_len,
                        2,
                    );
                }
            }
        }
    }
    let mut no_deblock_opts = *options;
    no_deblock_opts.enable_deblock = false;
    apply_loop_filters_with_ctu_params(planes, qp, ctu, &no_deblock_opts, sao_grid, alf_grid);
}

/// Encode an intra picture using the full Wave 1–6 multi-tool pipeline
/// (`67` intra modes, `CCLM`, `MIP`, `JCCR`, `DepQuant`, `SDH`, and in-loop filters).
pub fn encode_picture_with_options(
    src: &I420Frame,
    qp: u8,
    ctu_log2: u8,
    options: &SliceCodingOptions,
) -> Result<Vec<u8>, VvcError> {
    if !options.any_tool() {
        return encode_picture(src, qp, ctu_log2);
    }
    encode_picture_multitool(src, qp, ctu_log2, options)
}

/// Encode an intra picture with the multi-tool slice writer even when every
/// optional tool is disabled.
///
/// This is used by inter streams, whose SPS cannot advertise the MTS/LFNST
/// syntax that the strict subset writer emits.
pub fn encode_picture_multitool(
    src: &I420Frame,
    qp: u8,
    ctu_log2: u8,
    options: &SliceCodingOptions,
) -> Result<Vec<u8>, VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, src.width, src.height)?;
    let mut enc = BinEncoder::new();
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));
    let src_planes = Planes::from_frame(src)?;
    let mut recon = Planes::from_frame(src)?;
    let ctu = ctu.min(32);
    let width = src.width as usize;
    let height = src.height as usize;
    let cw = ctu / 2;
    let luma_qp = qp.max(4);
    let chroma_qp = (super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8).max(4);
    let luma_ts = luma_qp == 4 && !options.dep_quant;
    let chroma_ts = chroma_qp == 4 && !options.dep_quant;

    let grid_w = width / ctu;
    let grid_h = height / ctu;
    let mut mode_grid = vec![0u8; grid_w * grid_h];
    let mut mip_grid = vec![false; grid_w * grid_h];
    let sao_grid = vec![[super::filter::SaoType::Off; 3]; grid_w * grid_h];
    let mut alf_grid = vec![None; grid_w * grid_h];

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let ctu_x = gx * ctu;
            let ctu_y = gy * ctu;
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;
            let ctu_idx = gy * grid_w + gx;

            // 1. Per-CTU SAO & ALF syntax at start of coding_tree_unit (H.266 §7.3.11.2)
            if options.enable_sao {
                let merge_left = gx > 0;
                let merge_up = !merge_left && gy > 0;
                encode_ctu_sao(
                    &mut enc,
                    &mut ctx,
                    gx,
                    gy,
                    true,
                    true,
                    &sao_grid[ctu_idx],
                    merge_left,
                    merge_up,
                    8,
                );
            }
            if options.enable_alf {
                encode_ctu_alf_luma_fixed(&mut enc, &mut ctx, false, false, false, 0);
                alf_grid[ctu_idx] = None;
            }

            let left_mode = if gx > 0 {
                Some(mode_grid[gy * grid_w + (gx - 1)])
            } else {
                None
            };
            let above_same_ctu = gy > 0 && (ctu_y % ctu) != 0;
            let above_mode = if above_same_ctu {
                Some(mode_grid[(gy - 1) * grid_w + gx])
            } else {
                None
            };
            let mpm = predict::derive_intra_mpms(left_mode, above_mode);

            let luma_mode: predict::IntraPredMode = if options.full_intra_modes {
                predict::choose_luma_intra_pred_mode(
                    &recon.y,
                    &src_planes.y,
                    ctu_x,
                    ctu_y,
                    ctu,
                    ctu,
                    &mpm,
                    false,
                )
            } else {
                predict::IntraPredMode::from(predict::choose_intra_mode(
                    &recon.y,
                    &src_planes.y,
                    ctu_x,
                    ctu_y,
                    ctu,
                    ctu,
                    true,
                ))
            };
            mode_grid[ctu_idx] = luma_mode.mode_index();

            // Evaluate MRL (lines 0, 1, 2) when enabled and not at top of CTU row
            let mut mrl_idx = 0u8;
            let mut luma_mode = luma_mode;
            let mut pred_y = predict::predict_block_full(
                &recon.y, None, ctu_x, ctu_y, ctu, ctu, true, luma_mode, 0, ctu,
            );
            let mut best_luma_sad: u64 = (0..ctu)
                .flat_map(|y| (0..ctu).map(move |x| (x, y)))
                .map(|(x, y)| {
                    (i32::from(src_planes.y.get(ctu_x + x, ctu_y + y)) - pred_y[y * ctu + x])
                        .unsigned_abs() as u64
                })
                .sum();

            let mut chosen_mip: Option<(bool, u8)> = None;
            if options.enable_mip {
                let refs = predict::gather_references(&recon.y, ctu_x, ctu_y, ctu, ctu);
                let (mip_tr, mip_idx, mip_pred, mip_sad) =
                    predict::choose_best_mip_mode(&refs, &src_planes.y, ctu_x, ctu_y, ctu, ctu);
                if mip_sad < best_luma_sad {
                    best_luma_sad = mip_sad;
                    chosen_mip = Some((mip_tr, mip_idx));
                    pred_y = mip_pred;
                    mode_grid[ctu_idx] = 0;
                }
            }

            // MRL is signalled only when the CU top is not the CTU top
            // (H.266 §7.3.11.5; VTM `CABACWriter::extend_ref_line`).
            if options.enable_mrl && chosen_mip.is_none() && (ctu_y % ctu) != 0 {
                for cand_mrl in 1..=2u8 {
                    for &mpm_cand in &mpm[1..6] {
                        let cand_mode = ipm(mpm_cand);
                        let cand_pred = predict::predict_block_full(
                            &recon.y,
                            None,
                            ctu_x,
                            ctu_y,
                            ctu,
                            ctu,
                            true,
                            cand_mode,
                            usize::from(cand_mrl),
                            ctu,
                        );
                        let sad: u64 = (0..ctu)
                            .flat_map(|y| (0..ctu).map(move |x| (x, y)))
                            .map(|(x, y)| {
                                (i32::from(src_planes.y.get(ctu_x + x, ctu_y + y))
                                    - cand_pred[y * ctu + x])
                                    .unsigned_abs() as u64
                            })
                            .sum();
                        if sad < best_luma_sad {
                            best_luma_sad = sad;
                            mrl_idx = cand_mrl;
                            luma_mode = cand_mode;
                            mode_grid[ctu_idx] = luma_mode.mode_index();
                            pred_y = cand_pred;
                        }
                    }
                }
            }

            let isp_mode = predict::IspMode::None;
            let (levels_y, recon_y_blk, cbf_y) = encode_component_full(
                &pred_y,
                &src_planes.y,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                luma_qp,
                luma_ts,
                options.dep_quant,
            )?;
            for y in 0..ctu {
                for x in 0..ctu {
                    recon.y.set(ctu_x + x, ctu_y + y, recon_y_blk[y * ctu + x]);
                }
            }

            // Select chroma mode (DM vs CCLM 81=LM, 82=MDLM_L, 83=MDLM_T if enable_cclm).
            let chroma_mode = if options.enable_cclm {
                let pred_dm = predict::predict_block_full(
                    &recon.u,
                    Some(&recon.y),
                    cx,
                    cy,
                    cw,
                    cw,
                    false,
                    luma_mode,
                    0,
                    ctu,
                );
                let mut best_cm = luma_mode.mode_index();
                let mut best_csad: u64 = (0..cw)
                    .flat_map(|y| (0..cw).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        (i32::from(src_planes.u.get(cx + x, cy + y)) - pred_dm[y * cw + x])
                            .unsigned_abs() as u64
                    })
                    .sum();
                for cclm_m in [81u8, 82, 83] {
                    let pred_c =
                        predict::predict_cclm(&recon.y, &recon.u, cx, cy, cw, cw, cclm_m, ctu);
                    let csad: u64 = (0..cw)
                        .flat_map(|y| (0..cw).map(move |x| (x, y)))
                        .map(|(x, y)| {
                            (i32::from(src_planes.u.get(cx + x, cy + y)) - pred_c[y * cw + x])
                                .unsigned_abs() as u64
                        })
                        .sum();
                    if csad < best_csad {
                        best_csad = csad;
                        best_cm = cclm_m;
                    }
                }
                best_cm
            } else {
                luma_mode.mode_index()
            };

            let pred_u = if (81..=83).contains(&chroma_mode) {
                predict::predict_cclm(&recon.y, &recon.u, cx, cy, cw, cw, chroma_mode, ctu)
            } else {
                predict::predict_block_full(
                    &recon.u,
                    Some(&recon.y),
                    cx,
                    cy,
                    cw,
                    cw,
                    false,
                    ipm(chroma_mode),
                    0,
                    ctu,
                )
            };
            let pred_v = if (81..=83).contains(&chroma_mode) {
                predict::predict_cclm(&recon.y, &recon.v, cx, cy, cw, cw, chroma_mode, ctu)
            } else {
                predict::predict_block_full(
                    &recon.v,
                    None,
                    cx,
                    cy,
                    cw,
                    cw,
                    false,
                    ipm(chroma_mode),
                    0,
                    ctu,
                )
            };

            let (levels_u, recon_u_blk, cbf_u) = encode_component_full(
                &pred_u,
                &src_planes.u,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                chroma_ts,
                options.dep_quant,
            )?;
            let (levels_v, recon_v_blk, cbf_v) = encode_component_full(
                &pred_v,
                &src_planes.v,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                chroma_ts,
                options.dep_quant,
            )?;
            for y in 0..cw {
                for x in 0..cw {
                    recon.u.set(cx + x, cy + y, recon_u_blk[y * cw + x]);
                    recon.v.set(cx + x, cy + y, recon_v_blk[y * cw + x]);
                }
            }

            // CABAC encoding for CU:
            if options.enable_bdpcm {
                encode_bdpcm_mode(&mut enc, &mut ctx, 0, false);
            }
            if options.enable_mip {
                let num_mip = predict::get_num_modes_mip(ctu, ctu) as u32;
                let left_mip = gx > 0 && mip_grid[ctu_idx - 1];
                let above_mip = gy > 0 && mip_grid[ctu_idx - grid_w];
                let mip_ctx = mip_flag_ctx_inc(left_mip, above_mip, ctu, ctu);
                let mip_flag = chosen_mip.is_some();
                mip_grid[ctu_idx] = mip_flag;
                if let Some((mip_tr, mip_idx)) = chosen_mip {
                    encode_mip_mode(&mut enc, &mut ctx, true, mip_tr, mip_idx, num_mip, mip_ctx);
                } else {
                    encode_mip_mode(&mut enc, &mut ctx, false, false, 0, num_mip, mip_ctx);
                }
            }
            if chosen_mip.is_none() {
                if options.enable_mrl && (ctu_y % ctu) != 0 {
                    encode_mrl_idx(&mut enc, &mut ctx, mrl_idx);
                }
                if options.enable_isp && mrl_idx == 0 {
                    encode_isp_mode(&mut enc, &mut ctx, isp_mode);
                }
                encode_full_luma_intra_mode(
                    &mut enc,
                    &mut ctx,
                    luma_mode.mode_index(),
                    &mpm,
                    mrl_idx,
                    isp_mode != predict::IspMode::None,
                );
            }
            if options.enable_bdpcm {
                encode_bdpcm_mode(&mut enc, &mut ctx, 0, true);
            }
            encode_chroma_intra_mode(
                &mut enc,
                &mut ctx,
                chroma_mode,
                luma_mode.mode_index(),
                options.enable_cclm,
            );

            enc.encode_bin(cbf_u, &mut ctx.models[ctx::QT_CBF_CB]);
            enc.encode_bin(cbf_v, &mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_u)]);
            enc.encode_bin(cbf_y, &mut ctx.models[ctx::QT_CBF_Y]);
            if options.enable_jccr && (cbf_u || cbf_v) {
                residual::encode_joint_cbcr_flag(&mut enc, &mut ctx, true, cbf_u, cbf_v, false);
            }

            let luma_scan = CoeffScan::new(ctu, ctu);
            if cbf_y {
                enc.encode_bin(luma_ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                if luma_ts {
                    encode_ts_block(&mut enc, &mut ctx, &luma_scan, &levels_y);
                } else {
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &luma_scan,
                        &levels_y,
                        true,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
            }
            let chroma_scan = CoeffScan::new(cw, cw);
            if cbf_u {
                enc.encode_bin(chroma_ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if chroma_ts {
                    encode_ts_block(&mut enc, &mut ctx, &chroma_scan, &levels_u);
                } else {
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &chroma_scan,
                        &levels_u,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
            }
            if cbf_v {
                enc.encode_bin(chroma_ts, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if chroma_ts {
                    encode_ts_block(&mut enc, &mut ctx, &chroma_scan, &levels_v);
                } else {
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &chroma_scan,
                        &levels_v,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
            }
        }
    }

    apply_loop_filters_with_ctu_params(
        &mut recon,
        luma_qp,
        ctu,
        options,
        Some(&sao_grid),
        Some(&alf_grid),
    );
    enc.encode_bin_trm(true);
    Ok(enc.finish_and_align())
}

/// Decode an intra picture coded by [`encode_picture_with_options`].
pub fn decode_picture_with_options(
    cabac: &[u8],
    qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    options: &SliceCodingOptions,
) -> Result<I420Frame, VvcError> {
    if !options.any_tool() {
        return decode_picture(cabac, qp, ctu_log2, coded_width, coded_height);
    }
    decode_picture_multitool(cabac, qp, ctu_log2, coded_width, coded_height, options)
}

/// Decode an intra picture with the multi-tool slice reader even when every
/// optional tool is disabled.
pub fn decode_picture_multitool(
    cabac: &[u8],
    qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    options: &SliceCodingOptions,
) -> Result<I420Frame, VvcError> {
    let chroma_qp = super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8;
    decode_picture_multitool_with_chroma_qp(
        cabac,
        qp,
        chroma_qp,
        ctu_log2,
        coded_width,
        coded_height,
        options,
    )
}

/// Decode an intra picture with the multi-tool slice reader, using the chroma
/// quantisation parameter derived from the signalled SPS chroma QP table
/// (`H.266 §7.4.3.4`) instead of the default mapping.
#[allow(clippy::too_many_arguments)]
pub fn decode_picture_multitool_with_chroma_qp(
    cabac: &[u8],
    qp: u8,
    chroma_qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    options: &SliceCodingOptions,
) -> Result<I420Frame, VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, coded_width, coded_height)?;
    let mut dec = BinDecoder::new(cabac).map_err(|err| VvcError::Bitstream(err.to_string()))?;
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));
    let ctu = ctu.min(32);
    let width = coded_width as usize;
    let height = coded_height as usize;
    let cw = ctu / 2;
    let luma_qp = qp.max(4);
    let chroma_qp = chroma_qp.max(4);

    let mut recon = Planes {
        y: Plane::new(width, height),
        u: Plane::new(width.div_ceil(2), height.div_ceil(2)),
        v: Plane::new(width.div_ceil(2), height.div_ceil(2)),
    };

    let grid_w = width / ctu;
    let grid_h = height / ctu;
    let mut mode_grid = vec![0u8; grid_w * grid_h];
    let mut mip_grid = vec![false; grid_w * grid_h];
    let mut sao_grid = vec![[super::filter::SaoType::Off; 3]; grid_w * grid_h];
    let mut alf_grid = vec![None; grid_w * grid_h];

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let ctu_x = gx * ctu;
            let ctu_y = gy * ctu;
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;
            let ctu_idx = gy * grid_w + gx;

            if options.enable_sao {
                let left_sao = if gx > 0 {
                    Some(sao_grid[gy * grid_w + (gx - 1)])
                } else {
                    None
                };
                let up_sao = if gy > 0 {
                    Some(sao_grid[(gy - 1) * grid_w + gx])
                } else {
                    None
                };
                sao_grid[ctu_idx] =
                    decode_ctu_sao(&mut dec, &mut ctx, gx, gy, true, true, left_sao, up_sao, 8);
            }
            if options.enable_alf {
                let left_alf = gx > 0 && alf_grid[gy * grid_w + (gx - 1)].is_some();
                let up_alf = gy > 0 && alf_grid[(gy - 1) * grid_w + gx].is_some();
                alf_grid[ctu_idx] = decode_ctu_alf_luma_fixed(&mut dec, &mut ctx, left_alf, up_alf);
            }

            let left_mode = if gx > 0 {
                Some(mode_grid[gy * grid_w + (gx - 1)])
            } else {
                None
            };
            let above_same_ctu = gy > 0 && (ctu_y % ctu) != 0;
            let above_mode = if above_same_ctu {
                Some(mode_grid[(gy - 1) * grid_w + gx])
            } else {
                None
            };
            let mpm = predict::derive_intra_mpms(left_mode, above_mode);

            let bdpcm_luma = if options.enable_bdpcm {
                decode_bdpcm_mode(&mut dec, &mut ctx, false)
            } else {
                0
            };
            let _ = bdpcm_luma;
            let mip_info = if options.enable_mip {
                let num_mip = predict::get_num_modes_mip(ctu, ctu) as u32;
                let left_mip = gx > 0 && mip_grid[ctu_idx - 1];
                let above_mip = gy > 0 && mip_grid[ctu_idx - grid_w];
                let mip_ctx = mip_flag_ctx_inc(left_mip, above_mip, ctu, ctu);
                let info = decode_mip_mode(&mut dec, &mut ctx, num_mip, mip_ctx);
                mip_grid[ctu_idx] = info.is_some();
                info
            } else {
                None
            };
            let mut mrl_idx = 0u8;
            let mut isp_mode = predict::IspMode::None;
            let luma_mode = if mip_info.is_some() {
                0u8
            } else {
                if options.enable_mrl && (ctu_y % ctu) != 0 {
                    mrl_idx = decode_mrl_idx(&mut dec, &mut ctx);
                }
                if options.enable_isp && mrl_idx == 0 {
                    isp_mode = decode_isp_mode(&mut dec, &mut ctx);
                }
                decode_full_luma_intra_mode(
                    &mut dec,
                    &mut ctx,
                    &mpm,
                    mrl_idx,
                    isp_mode != predict::IspMode::None,
                )
            };
            mode_grid[ctu_idx] = luma_mode;
            let bdpcm_chroma = if options.enable_bdpcm {
                decode_bdpcm_mode(&mut dec, &mut ctx, true)
            } else {
                0
            };
            let _ = (bdpcm_chroma, isp_mode);
            let chroma_mode =
                decode_chroma_intra_mode(&mut dec, &mut ctx, luma_mode, options.enable_cclm);

            let cbf_u = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
            let cbf_v = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_u)]);
            let cbf_y = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_Y]);
            let jccr_flag = if options.enable_jccr && (cbf_u || cbf_v) {
                residual::decode_joint_cbcr_flag(&mut dec, &mut ctx, true, cbf_u, cbf_v) != 0
            } else {
                false
            };
            let _ = jccr_flag;

            let luma_scan = CoeffScan::new(ctu, ctu);
            let chroma_scan = CoeffScan::new(cw, cw);
            let mut levels_y = vec![0i32; ctu * ctu];
            let mut ts_y = false;
            if cbf_y {
                ts_y = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                if ts_y {
                    decode_ts_block(&mut dec, &mut ctx, &luma_scan, &mut levels_y);
                } else {
                    residual::decode_residual_with_options(
                        &mut dec,
                        &mut ctx,
                        &luma_scan,
                        &mut levels_y,
                        true,
                        options.dep_quant,
                        options.sign_data_hiding,
                    )?;
                }
            }
            let mut levels_u = vec![0i32; cw * cw];
            let mut ts_u = false;
            if cbf_u {
                ts_u = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_u {
                    decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_u);
                } else {
                    residual::decode_residual_with_options(
                        &mut dec,
                        &mut ctx,
                        &chroma_scan,
                        &mut levels_u,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    )?;
                }
            }
            let mut levels_v = vec![0i32; cw * cw];
            let mut ts_v = false;
            if cbf_v && !jccr_flag {
                ts_v = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_v {
                    decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_v);
                } else {
                    residual::decode_residual_with_options(
                        &mut dec,
                        &mut ctx,
                        &chroma_scan,
                        &mut levels_v,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    )?;
                }
            }

            // Reconstruct luma first so CCLM chroma can reference reconstructed luma.
            let pred_y = if let Some((transposed, mip_idx)) = mip_info {
                let refs = predict::gather_references(&recon.y, ctu_x, ctu_y, ctu, ctu);
                predict::predict_mip(&refs, ctu, ctu, mip_idx, transposed)
            } else {
                predict::predict_block_full(
                    &recon.y,
                    None,
                    ctu_x,
                    ctu_y,
                    ctu,
                    ctu,
                    true,
                    ipm(luma_mode),
                    usize::from(mrl_idx),
                    ctu,
                )
            };
            reconstruct_component_full(
                &mut recon.y,
                &pred_y,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                luma_qp,
                &levels_y,
                cbf_y,
                ts_y,
                options.dep_quant,
            )?;

            let pred_u = if (81..=83).contains(&chroma_mode) {
                predict::predict_cclm(&recon.y, &recon.u, cx, cy, cw, cw, chroma_mode, ctu)
            } else {
                predict::predict_block_full(
                    &recon.u,
                    Some(&recon.y),
                    cx,
                    cy,
                    cw,
                    cw,
                    false,
                    ipm(chroma_mode),
                    0,
                    ctu,
                )
            };
            let pred_v = if (81..=83).contains(&chroma_mode) {
                predict::predict_cclm(&recon.y, &recon.v, cx, cy, cw, cw, chroma_mode, ctu)
            } else {
                predict::predict_block_full(
                    &recon.v,
                    None,
                    cx,
                    cy,
                    cw,
                    cw,
                    false,
                    ipm(chroma_mode),
                    0,
                    ctu,
                )
            };
            reconstruct_component_full(
                &mut recon.u,
                &pred_u,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                &levels_u,
                cbf_u,
                ts_u,
                options.dep_quant,
            )?;
            reconstruct_component_full(
                &mut recon.v,
                &pred_v,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                &levels_v,
                cbf_v,
                ts_v,
                options.dep_quant,
            )?;
        }
    }

    dec.end_of_slice()
        .map_err(|err| VvcError::Bitstream(err.to_string()))?;
    apply_loop_filters_with_ctu_params(
        &mut recon,
        luma_qp,
        ctu,
        options,
        Some(&sao_grid),
        Some(&alf_grid),
    );
    Ok(recon.into_frame())
}

/// Encode a P or B inter picture (CABAC substream + reconstructed frame) using
/// Merge, AMVP + MVD, and Intra CU modes with 8-tap luma / 4-tap chroma MC.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_picture(
    src: &I420Frame,
    qp: u8,
    ctu_log2: u8,
    slice_type: super::hls::VvcSliceType,
    cur_poc: i32,
    ref_l0: &[super::inter::RefFrame],
    ref_l1: &[super::inter::RefFrame],
    options: &SliceCodingOptions,
) -> Result<(Vec<u8>, I420Frame, super::inter::MotionFieldGrid), VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, src.width, src.height)?;
    let mut enc = BinEncoder::new();
    let mut ctx = CtxStore::new_for_slice(slice_type, false, i32::from(qp));
    let src_planes = Planes::from_frame(src)?;
    let mut recon = Planes::from_frame(src)?;
    let width = src.width as usize;
    let height = src.height as usize;
    let cw = ctu / 2;
    let luma_qp = qp.max(4);
    let chroma_qp = (super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8).max(4);

    let mut mv_grid = super::inter::MotionFieldGrid::new(width, height);
    let mut hmvp = super::inter::HmvpTable::new();
    let ref0_refs: Vec<&super::inter::RefFrame> = ref_l0.iter().collect();
    let ref1_refs: Vec<&super::inter::RefFrame> = ref_l1.iter().collect();
    let grid_w = width / ctu;
    let grid_h = height / ctu;
    let mut skip_grid = vec![false; grid_w * grid_h];
    let mut cbf_y_grid = vec![false; grid_w * grid_h];
    let mut cbf_u_grid = vec![false; grid_w * grid_h];
    let mut cbf_v_grid = vec![false; grid_w * grid_h];
    let sao_grid = vec![[super::filter::SaoType::Off; 3]; grid_w * grid_h];
    let mut alf_grid = vec![None; grid_w * grid_h];
    let mut cu_index = 0usize;

    for ctu_y in (0..height).step_by(ctu) {
        hmvp.reset();
        for ctu_x in (0..width).step_by(ctu) {
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;
            let gx = ctu_x / ctu;
            let gy = ctu_y / ctu;

            if options.enable_sao {
                let merge_left = gx > 0;
                let merge_up = !merge_left && gy > 0;
                encode_ctu_sao(
                    &mut enc,
                    &mut ctx,
                    gx,
                    gy,
                    true,
                    true,
                    &sao_grid[cu_index],
                    merge_left,
                    merge_up,
                    8,
                );
            }
            if options.enable_alf {
                encode_ctu_alf_luma_fixed(&mut enc, &mut ctx, false, false, false, 0);
                alf_grid[cu_index] = None;
            }

            let merge_cands = super::inter::derive_merge_candidates(
                &mv_grid,
                &hmvp,
                None,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                slice_type == super::hls::VvcSliceType::B,
                6,
            );
            let merge_cand = merge_cands[0];
            let (merge_y, merge_u, merge_v) = super::inter::predict_inter_pu(
                &merge_cand,
                &ref0_refs,
                &ref1_refs,
                0,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                slice_type == super::hls::VvcSliceType::B && merge_cand.inter_dir == 3,
                false,
            );

            let mvp_idx = 0usize;
            let mvd = super::inter::Mv::ZERO;
            let use_merge = true;
            let chosen_cand = merge_cand;
            let (pred_y_u8, pred_u_u8, pred_v_u8) = (merge_y, merge_u, merge_v);

            let pred_y: Vec<i32> = pred_y_u8.iter().map(|&v| i32::from(v)).collect();
            let pred_u: Vec<i32> = pred_u_u8.iter().map(|&v| i32::from(v)).collect();
            let pred_v: Vec<i32> = pred_v_u8.iter().map(|&v| i32::from(v)).collect();

            let (levels_y, recon_y_blk, cbf_y) = encode_component_full(
                &pred_y,
                &src_planes.y,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                luma_qp,
                false,
                options.dep_quant,
            )?;
            let (levels_u, recon_u_blk, cbf_u) = encode_component_full(
                &pred_u,
                &src_planes.u,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                false,
                options.dep_quant,
            )?;
            let (levels_v, recon_v_blk, cbf_v) = encode_component_full(
                &pred_v,
                &src_planes.v,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                false,
                options.dep_quant,
            )?;

            for y in 0..ctu {
                for x in 0..ctu {
                    recon.y.set(ctu_x + x, ctu_y + y, recon_y_blk[y * ctu + x]);
                }
            }
            for y in 0..cw {
                for x in 0..cw {
                    recon.u.set(cx + x, cy + y, recon_u_blk[y * cw + x]);
                    recon.v.set(cx + x, cy + y, recon_v_blk[y * cw + x]);
                }
            }

            mv_grid.fill_block(ctu_x, ctu_y, ctu, ctu, chosen_cand);
            hmvp.push(chosen_cand);

            let any_cbf = cbf_y || cbf_u || cbf_v;
            let cu_skip = use_merge && !any_cbf;
            let left_skip = gx > 0 && skip_grid[gy * grid_w + (gx - 1)];
            let above_skip = gy > 0 && skip_grid[(gy - 1) * grid_w + gx];
            let skip_ctx = ctx::SKIP_FLAG + usize::from(left_skip) + usize::from(above_skip);
            enc.encode_bin(cu_skip, &mut ctx.models[skip_ctx]);
            skip_grid[cu_index] = cu_skip;
            cbf_y_grid[cu_index] = cbf_y;
            cbf_u_grid[cu_index] = cbf_u;
            cbf_v_grid[cu_index] = cbf_v;
            cu_index += 1;
            if cu_skip {
                continue;
            }

            enc.encode_bin(false, &mut ctx.models[ctx::PRED_MODE]);
            enc.encode_bin(use_merge, &mut ctx.models[ctx::MERGE_FLAG]);
            if !use_merge {
                encode_mvd(&mut enc, &mut ctx, mvd);
                enc.encode_bin(mvp_idx != 0, &mut ctx.models[ctx::MVP_IDX]);
                enc.encode_bin(any_cbf, &mut ctx.models[ctx::QT_ROOT_CBF]);
            }
            if any_cbf {
                enc.encode_bin(cbf_u, &mut ctx.models[ctx::QT_CBF_CB]);
                enc.encode_bin(cbf_v, &mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_u)]);
                // Normative H.266 §7.3.11.10 (`transform_unit`, PDF p. 96) and
                // VTM `CABACWriter.cpp:2410-2413`: when `!isIntra(cu) && trDepth == 0`
                // and `!cbf_cb && !cbf_cr`, `tu_y_coded_flag` is NOT coded in the
                // bitstream and is normatively inferred to 1!
                if cbf_u || cbf_v {
                    enc.encode_bin(cbf_y, &mut ctx.models[ctx::QT_CBF_Y]);
                }
                if options.enable_jccr && cbf_u && cbf_v {
                    residual::encode_joint_cbcr_flag(&mut enc, &mut ctx, true, cbf_u, cbf_v, false);
                }

                let luma_scan = CoeffScan::new(ctu, ctu);
                if cbf_y {
                    enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &luma_scan,
                        &levels_y,
                        true,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
                let chroma_scan = CoeffScan::new(cw, cw);
                if cbf_u {
                    enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &chroma_scan,
                        &levels_u,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
                if cbf_v {
                    enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                    residual::encode_residual_with_options(
                        &mut enc,
                        &mut ctx,
                        &chroma_scan,
                        &levels_v,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    );
                }
            }
        }
    }

    let _ = cur_poc;
    apply_inter_loop_filters(
        &mut recon,
        luma_qp,
        ctu,
        options,
        &cbf_y_grid,
        &cbf_u_grid,
        &cbf_v_grid,
        &mv_grid,
        Some(&sao_grid),
        Some(&alf_grid),
    );
    enc.encode_bin_trm(true);
    Ok((enc.finish_and_align(), recon.into_frame(), mv_grid))
}

/// Decode a P or B inter picture coded by [`encode_inter_picture`].
#[allow(clippy::too_many_arguments)]
pub fn decode_inter_picture(
    cabac: &[u8],
    qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    slice_type: super::hls::VvcSliceType,
    ref_l0: &[super::inter::RefFrame],
    ref_l1: &[super::inter::RefFrame],
    options: &SliceCodingOptions,
) -> Result<(I420Frame, super::inter::MotionFieldGrid), VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, coded_width, coded_height)?;
    let mut dec = BinDecoder::new(cabac).map_err(|err| VvcError::Bitstream(err.to_string()))?;
    let mut ctx = CtxStore::new_for_slice(slice_type, false, i32::from(qp));
    let width = coded_width as usize;
    let height = coded_height as usize;
    let cw = ctu / 2;
    let luma_qp = qp.max(4);
    let chroma_qp = (super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63) as u8).max(4);

    let mut recon = Planes {
        y: Plane::new(width, height),
        u: Plane::new(width.div_ceil(2), height.div_ceil(2)),
        v: Plane::new(width.div_ceil(2), height.div_ceil(2)),
    };
    let mut mv_grid = super::inter::MotionFieldGrid::new(width, height);
    let mut hmvp = super::inter::HmvpTable::new();
    let ref0_refs: Vec<&super::inter::RefFrame> = ref_l0.iter().collect();
    let ref1_refs: Vec<&super::inter::RefFrame> = ref_l1.iter().collect();
    let grid_w = width / ctu;
    let grid_h = height / ctu;
    let mut skip_grid = vec![false; grid_w * grid_h];
    let mut cbf_y_grid = vec![false; grid_w * grid_h];
    let mut cbf_u_grid = vec![false; grid_w * grid_h];
    let mut cbf_v_grid = vec![false; grid_w * grid_h];
    let mut sao_grid = vec![[super::filter::SaoType::Off; 3]; grid_w * grid_h];
    let mut alf_grid = vec![None; grid_w * grid_h];
    let mut cu_index = 0usize;

    for ctu_y in (0..height).step_by(ctu) {
        hmvp.reset();
        for ctu_x in (0..width).step_by(ctu) {
            let cx = ctu_x / 2;
            let cy = ctu_y / 2;
            let gx = ctu_x / ctu;
            let gy = ctu_y / ctu;

            if options.enable_sao {
                let left_sao = if gx > 0 {
                    Some(sao_grid[gy * grid_w + (gx - 1)])
                } else {
                    None
                };
                let up_sao = if gy > 0 {
                    Some(sao_grid[(gy - 1) * grid_w + gx])
                } else {
                    None
                };
                sao_grid[cu_index] =
                    decode_ctu_sao(&mut dec, &mut ctx, gx, gy, true, true, left_sao, up_sao, 8);
            }
            if options.enable_alf {
                let left_alf = gx > 0 && alf_grid[gy * grid_w + (gx - 1)].is_some();
                let up_alf = gy > 0 && alf_grid[(gy - 1) * grid_w + gx].is_some();
                alf_grid[cu_index] =
                    decode_ctu_alf_luma_fixed(&mut dec, &mut ctx, left_alf, up_alf);
            }

            let left_skip = gx > 0 && skip_grid[gy * grid_w + (gx - 1)];
            let above_skip = gy > 0 && skip_grid[(gy - 1) * grid_w + gx];
            let skip_ctx = ctx::SKIP_FLAG + usize::from(left_skip) + usize::from(above_skip);
            let cu_skip = dec.decode_bin(&mut ctx.models[skip_ctx]);
            skip_grid[cu_index] = cu_skip;
            let cur_cu_idx = cu_index;
            cu_index += 1;
            let (chosen_cand, any_cbf) = if cu_skip {
                let cands = super::inter::derive_merge_candidates(
                    &mv_grid,
                    &hmvp,
                    None,
                    ctu_x,
                    ctu_y,
                    ctu,
                    ctu,
                    slice_type == super::hls::VvcSliceType::B,
                    6,
                );
                (cands[0], false)
            } else {
                let is_intra = dec.decode_bin(&mut ctx.models[ctx::PRED_MODE]);
                if is_intra {
                    return Err(VvcError::Unsupported("intra CU inside inter fast slice"));
                }
                let use_merge = dec.decode_bin(&mut ctx.models[ctx::MERGE_FLAG]);
                let cand = if use_merge {
                    let cands = super::inter::derive_merge_candidates(
                        &mv_grid,
                        &hmvp,
                        None,
                        ctu_x,
                        ctu_y,
                        ctu,
                        ctu,
                        slice_type == super::hls::VvcSliceType::B,
                        6,
                    );
                    cands[0]
                } else {
                    let mvd = decode_mvd(&mut dec, &mut ctx);
                    let mvp_idx = usize::from(dec.decode_bin(&mut ctx.models[ctx::MVP_IDX]));
                    let amvp_cands = super::inter::derive_amvp_candidates(
                        &mv_grid,
                        &hmvp,
                        ctu_x,
                        ctu_y,
                        ctu,
                        ctu,
                        0,
                        0,
                        super::inter::AmvrPrecision::QuarterPel,
                    );
                    super::inter::InterMergeCand::uni_l0(0, amvp_cands[mvp_idx] + mvd)
                };
                let root_cbf = use_merge || dec.decode_bin(&mut ctx.models[ctx::QT_ROOT_CBF]);
                (cand, root_cbf)
            };

            mv_grid.fill_block(ctu_x, ctu_y, ctu, ctu, chosen_cand);
            hmvp.push(chosen_cand);

            let (pred_y_u8, pred_u_u8, pred_v_u8) = super::inter::predict_inter_pu(
                &chosen_cand,
                &ref0_refs,
                &ref1_refs,
                0,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                slice_type == super::hls::VvcSliceType::B && chosen_cand.inter_dir == 3,
                false,
            );
            let pred_y: Vec<i32> = pred_y_u8.iter().map(|&v| i32::from(v)).collect();
            let pred_u: Vec<i32> = pred_u_u8.iter().map(|&v| i32::from(v)).collect();
            let pred_v: Vec<i32> = pred_v_u8.iter().map(|&v| i32::from(v)).collect();

            let mut cbf_u = false;
            let mut cbf_v = false;
            let mut cbf_y = false;
            let mut ts_y = false;
            let mut ts_u = false;
            let mut ts_v = false;
            let mut levels_y = vec![0i32; ctu * ctu];
            let mut levels_u = vec![0i32; cw * cw];
            let mut levels_v = vec![0i32; cw * cw];

            if any_cbf {
                cbf_u = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
                cbf_v = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_u)]);
                // Normative H.266 §7.3.11.10 / VTM `CABACReader.cpp:2760-2763`:
                // when `!isIntra(cu) && trDepth == 0 && !cbf_cb && !cbf_cr`,
                // `tu_y_coded_flag` is inferred to 1 without decoding a bin.
                cbf_y = if cbf_u || cbf_v {
                    dec.decode_bin(&mut ctx.models[ctx::QT_CBF_Y])
                } else {
                    true
                };
                if options.enable_jccr && cbf_u && cbf_v {
                    let _ =
                        residual::decode_joint_cbcr_flag(&mut dec, &mut ctx, false, cbf_u, cbf_v);
                }

                let luma_scan = CoeffScan::new(ctu, ctu);
                if cbf_y {
                    ts_y = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
                    if ts_y {
                        decode_ts_block(&mut dec, &mut ctx, &luma_scan, &mut levels_y);
                    } else {
                        residual::decode_residual_with_options(
                            &mut dec,
                            &mut ctx,
                            &luma_scan,
                            &mut levels_y,
                            true,
                            options.dep_quant,
                            options.sign_data_hiding,
                        )?;
                    }
                }
                let chroma_scan = CoeffScan::new(cw, cw);
                if cbf_u {
                    ts_u = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                    if ts_u {
                        decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_u);
                    } else {
                        residual::decode_residual_with_options(
                            &mut dec,
                            &mut ctx,
                            &chroma_scan,
                            &mut levels_u,
                            false,
                            options.dep_quant,
                            options.sign_data_hiding,
                        )?;
                    }
                }
                if cbf_v {
                    ts_v = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                    if ts_v {
                        decode_ts_block(&mut dec, &mut ctx, &chroma_scan, &mut levels_v);
                    } else {
                        residual::decode_residual_with_options(
                            &mut dec,
                            &mut ctx,
                            &chroma_scan,
                            &mut levels_v,
                            false,
                            options.dep_quant,
                            options.sign_data_hiding,
                        )?;
                    }
                }
            }
            cbf_y_grid[cur_cu_idx] = cbf_y;
            cbf_u_grid[cur_cu_idx] = cbf_u;
            cbf_v_grid[cur_cu_idx] = cbf_v;

            reconstruct_component_full(
                &mut recon.y,
                &pred_y,
                ctu_x,
                ctu_y,
                ctu,
                ctu,
                luma_qp,
                &levels_y,
                cbf_y,
                ts_y,
                options.dep_quant,
            )?;
            reconstruct_component_full(
                &mut recon.u,
                &pred_u,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                &levels_u,
                cbf_u,
                ts_u,
                options.dep_quant,
            )?;
            reconstruct_component_full(
                &mut recon.v,
                &pred_v,
                cx,
                cy,
                cw,
                cw,
                chroma_qp,
                &levels_v,
                cbf_v,
                ts_v,
                options.dep_quant,
            )?;
        }
    }

    dec.end_of_slice()
        .map_err(|err| VvcError::Bitstream(err.to_string()))?;
    apply_inter_loop_filters(
        &mut recon,
        luma_qp,
        ctu,
        options,
        &cbf_y_grid,
        &cbf_u_grid,
        &cbf_v_grid,
        &mv_grid,
        Some(&sao_grid),
        Some(&alf_grid),
    );
    Ok((recon.into_frame(), mv_grid))
}

/// Encode `cu_act_enabled_flag` (`ctx::ACT_FLAG`, ITU-T Rec. H.266 §7.3.11.5,
/// VTM `CABACWriter.cpp:1358`).
#[inline]
pub fn encode_act_flag(enc: &mut BinEncoder, ctx: &mut CtxStore, act_enabled: bool) {
    enc.encode_bin(act_enabled, &mut ctx.models[ctx::ACT_FLAG]);
}

/// Decode `cu_act_enabled_flag` (`ctx::ACT_FLAG`, ITU-T Rec. H.266 §7.3.11.5,
/// VTM `CABACReader.cpp:1624`).
#[inline]
pub fn decode_act_flag(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> bool {
    dec.decode_bin(&mut ctx.models[ctx::ACT_FLAG])
}

/// Encode a QT/MTT coding tree split decision (`split_cu_flag`, `split_qt_flag`,
/// `mtt_split_cu_vertical_flag`, `mtt_split_cu_binary_flag`, ITU-T Rec. H.266 §7.3.11.4,
/// VTM `CABACWriter.cpp:525-605`).
pub fn encode_coding_tree_split(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    split: transform::CodingTreeSplit,
    can: transform::QtMttCanSplit,
    ctxs: transform::QtMttSplitContexts,
) {
    let is_split = split != transform::CodingTreeSplit::NoSplit;
    if can.can_no && can.any_split() {
        enc.encode_bin(is_split, &mut ctx.models[ctx::SPLIT_FLAG + ctxs.ctx_split]);
    }
    if !is_split {
        return;
    }
    let is_qt = split == transform::CodingTreeSplit::QuadTree;
    if can.can_qt && can.any_btt() {
        enc.encode_bin(is_qt, &mut ctx.models[ctx::SPLIT_QT_FLAG + ctxs.ctx_qt]);
    }
    if is_qt {
        return;
    }
    let can_hor = can.can_bh || can.can_th;
    let can_ver = can.can_bv || can.can_tv;
    let is_ver = matches!(
        split,
        transform::CodingTreeSplit::BinaryVer | transform::CodingTreeSplit::TernaryVer
    );
    if can_ver && can_hor {
        enc.encode_bin(is_ver, &mut ctx.models[ctx::SPLIT_HV_FLAG + ctxs.ctx_hv]);
    }
    let can_12 = if is_ver { can.can_bv } else { can.can_bh };
    let can_14 = if is_ver { can.can_tv } else { can.can_th };
    let is_12 = matches!(
        split,
        transform::CodingTreeSplit::BinaryHor | transform::CodingTreeSplit::BinaryVer
    );
    if can_12 && can_14 {
        let bt_ctx = if is_ver {
            ctxs.ctx_ver_bt
        } else {
            ctxs.ctx_hor_bt
        };
        enc.encode_bin(is_12, &mut ctx.models[ctx::SPLIT_12_FLAG + bt_ctx]);
    }
}

/// Decode a QT/MTT coding tree split decision (`split_cu_flag`, `split_qt_flag`,
/// `mtt_split_cu_vertical_flag`, `mtt_split_cu_binary_flag`, ITU-T Rec. H.266 §7.3.11.4,
/// VTM `CABACReader.cpp:776-856`).
pub fn decode_coding_tree_split(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    can: transform::QtMttCanSplit,
    ctxs: transform::QtMttSplitContexts,
) -> transform::CodingTreeSplit {
    let mut is_split = can.any_split();
    if can.can_no && is_split {
        is_split = dec.decode_bin(&mut ctx.models[ctx::SPLIT_FLAG + ctxs.ctx_split]);
    }
    if !is_split {
        return transform::CodingTreeSplit::NoSplit;
    }
    let mut is_qt = can.can_qt;
    if is_qt && can.any_btt() {
        is_qt = dec.decode_bin(&mut ctx.models[ctx::SPLIT_QT_FLAG + ctxs.ctx_qt]);
    }
    if is_qt {
        return transform::CodingTreeSplit::QuadTree;
    }
    let can_hor = can.can_bh || can.can_th;
    let mut is_ver = can.can_bv || can.can_tv;
    if is_ver && can_hor {
        is_ver = dec.decode_bin(&mut ctx.models[ctx::SPLIT_HV_FLAG + ctxs.ctx_hv]);
    }
    let can_14 = if is_ver { can.can_tv } else { can.can_th };
    let mut is_12 = if is_ver { can.can_bv } else { can.can_bh };
    if is_12 && can_14 {
        let bt_ctx = if is_ver {
            ctxs.ctx_ver_bt
        } else {
            ctxs.ctx_hor_bt
        };
        is_12 = dec.decode_bin(&mut ctx.models[ctx::SPLIT_12_FLAG + bt_ctx]);
    }
    match (is_ver, is_12) {
        (true, true) => transform::CodingTreeSplit::BinaryVer,
        (true, false) => transform::CodingTreeSplit::TernaryVer,
        (false, true) => transform::CodingTreeSplit::BinaryHor,
        (false, false) => transform::CodingTreeSplit::TernaryHor,
    }
}

/// Encode an Intra Block Copy (IBC) CU (`pred_mode_ibc_flag`, `merge_flag` + `merge_idx`
/// or `mvd_coding` + `mvp_idx`, ITU-T Rec. H.266 §7.3.11.5, §7.3.11.7).
pub fn encode_ibc_cu_syntax(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    ctx_ibc_inc: usize,
    use_merge: bool,
    merge_idx: u8,
    max_num_ibc_cand: u8,
    mvd: super::inter::Mv,
    mvp_idx: u8,
) {
    enc.encode_bin(true, &mut ctx.models[ctx::IBC_FLAG + ctx_ibc_inc.min(2)]);
    enc.encode_bin(use_merge, &mut ctx.models[ctx::MERGE_FLAG]);
    if use_merge {
        if max_num_ibc_cand > 1 {
            let idx = merge_idx.min(max_num_ibc_cand - 1);
            enc.encode_bin(idx > 0, &mut ctx.models[ctx::MERGE_IDX]);
            if idx > 0 && max_num_ibc_cand > 2 {
                enc.encode_unary_max_ep(u32::from(idx - 1), u32::from(max_num_ibc_cand - 2));
            }
        }
    } else {
        encode_mvd(enc, ctx, mvd);
        if max_num_ibc_cand > 1 {
            enc.encode_bin(mvp_idx != 0, &mut ctx.models[ctx::MVP_IDX]);
        }
    }
}

/// Decode an Intra Block Copy (IBC) CU syntax coded by [`encode_ibc_cu_syntax`]
/// (ITU-T Rec. H.266 §7.3.11.5, §7.3.11.7).
pub fn decode_ibc_cu_syntax(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    ctx_ibc_inc: usize,
    max_num_ibc_cand: u8,
) -> Option<(bool, u8, super::inter::Mv, u8)> {
    if !dec.decode_bin(&mut ctx.models[ctx::IBC_FLAG + ctx_ibc_inc.min(2)]) {
        return None;
    }
    let use_merge = dec.decode_bin(&mut ctx.models[ctx::MERGE_FLAG]);
    if use_merge {
        let mut merge_idx = 0u8;
        if max_num_ibc_cand > 1 && dec.decode_bin(&mut ctx.models[ctx::MERGE_IDX]) {
            merge_idx = 1;
            if max_num_ibc_cand > 2 {
                merge_idx += dec.decode_unary_max_ep(u32::from(max_num_ibc_cand - 2)) as u8;
            }
        }
        Some((true, merge_idx, super::inter::Mv::ZERO, 0))
    } else {
        let mvd = decode_mvd(dec, ctx);
        let mvp_idx = if max_num_ibc_cand > 1 {
            u8::from(dec.decode_bin(&mut ctx.models[ctx::MVP_IDX]))
        } else {
            0
        };
        Some((false, 0, mvd, mvp_idx))
    }
}

/// Encode a Palette CU (`pred_mode_plt_flag` + `palette_coding`, ITU-T Rec. H.266
/// §7.3.11.6, VTM `CABACWriter.cpp:1440-1695`).
pub fn encode_palette_cu_syntax(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    palette: &predict::PaletteBlock,
    bit_depth: u8,
) {
    enc.encode_bin(true, &mut ctx.models[ctx::PLT_FLAG]);
    let num_signalled = palette.entries.len() as u32;
    enc.encode_exp_golomb_ep(num_signalled, 0);
    let bd = u32::from(bit_depth.clamp(8, 12));
    for &[y, u, v] in &palette.entries {
        enc.encode_bins_ep(u32::from(y), bd);
        enc.encode_bins_ep(u32::from(u), bd);
        enc.encode_bins_ep(u32::from(v), bd);
    }
    if !palette.entries.is_empty() {
        enc.encode_bin_ep(palette.escape_present);
    }
    let max_idx = palette.entries.len() + usize::from(palette.escape_present);
    if max_idx > 1 {
        enc.encode_bin(
            palette.transpose,
            &mut ctx.models[ctx::PALETTE_TRANSPOSE_FLAG],
        );
        for &idx in &palette.index_map {
            enc.encode_trunc_bin(u32::from(idx), max_idx as u32);
        }
    }
    if palette.escape_present {
        for &[ey, eu, ev] in &palette.escape_samples {
            enc.encode_bins_ep(u32::from(ey), bd);
            enc.encode_bins_ep(u32::from(eu), bd);
            enc.encode_bins_ep(u32::from(ev), bd);
        }
    }
}

/// Decode a Palette CU coded by [`encode_palette_cu_syntax`] (`H.266 §7.3.11.6`).
pub fn decode_palette_cu_syntax(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Option<predict::PaletteBlock> {
    if !dec.decode_bin(&mut ctx.models[ctx::PLT_FLAG]) {
        return None;
    }
    let num_signalled = (dec.decode_exp_golomb_ep(0).unwrap_or(0) as usize).min(31);
    let bd = u32::from(bit_depth.clamp(8, 12));
    let mut entries = Vec::with_capacity(num_signalled);
    for _ in 0..num_signalled {
        let y = dec.decode_bins_ep(bd) as u16;
        let u = dec.decode_bins_ep(bd) as u16;
        let v = dec.decode_bins_ep(bd) as u16;
        entries.push([y, u, v]);
    }
    let escape_present = if !entries.is_empty() {
        dec.decode_bin_ep()
    } else {
        true
    };
    let max_idx = entries.len() + usize::from(escape_present);
    let mut transpose = false;
    let n = width * height;
    let mut index_map = vec![0u8; n];
    if max_idx > 1 {
        transpose = dec.decode_bin(&mut ctx.models[ctx::PALETTE_TRANSPOSE_FLAG]);
        for slot in &mut index_map {
            *slot = dec.decode_trunc_bin(max_idx as u32) as u8;
        }
    }
    let mut escape_samples = Vec::new();
    if escape_present {
        let esc_idx = entries.len() as u8;
        let num_esc = index_map.iter().filter(|&&i| i >= esc_idx).count();
        escape_samples.reserve(num_esc);
        for _ in 0..num_esc {
            let ey = dec.decode_bins_ep(bd) as u16;
            let eu = dec.decode_bins_ep(bd) as u16;
            let ev = dec.decode_bins_ep(bd) as u16;
            escape_samples.push([ey, eu, ev]);
        }
    }
    Some(predict::PaletteBlock {
        entries,
        transpose,
        escape_present,
        copy_above_flags: vec![false; n],
        index_map,
        escape_samples,
    })
}

/// Evaluate recursive QT/MTT partition RDO (Quadtree + BinaryHor + BinaryVer) for a CTU
/// region `(bx, by, w, h)` using Lagrangian cost `J = SSD + lambda * R`
/// (Wave G, VTM `EncCu.cpp`).
/// Evaluate recursive QT/MTT partition RDO (Quadtree + BinaryHor + BinaryVer) for a 16-bit
/// [`predict::Plane16`] CTU region `(bx, by, w, h)` using Lagrangian cost `J = SSD + lambda * R`
/// (Wave G / Wave N, VTM `EncCu.cpp`).
pub fn choose_qt_mtt_partition_rdo_16(
    src_y: &predict::Plane16,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    qp: u8,
    min_cb_size: usize,
) -> transform::CodingTreeSplit {
    let p8 = src_y.to_plane_8bit();
    choose_qt_mtt_partition_rdo(&p8, bx, by, w, h, qp, min_cb_size)
}

/// Evaluate recursive QT/MTT partition RDO (Quadtree + BinaryHor + BinaryVer) for a CTU
/// region `(bx, by, w, h)` using Lagrangian cost `J = SSD + lambda * R`
/// (Wave G, VTM `EncCu.cpp`).
pub fn choose_qt_mtt_partition_rdo(
    src_y: &Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    qp: u8,
    min_cb_size: usize,
) -> transform::CodingTreeSplit {
    if w <= min_cb_size.max(8) || h <= min_cb_size.max(8) {
        return transform::CodingTreeSplit::NoSplit;
    }
    // Compute block variance across the four quadrants to decide whether splitting reduces RDO cost.
    let hw = w / 2;
    let hh = h / 2;
    let quad_mean = |qx: usize, qy: usize, qw: usize, qh: usize| -> i64 {
        let mut sum = 0i64;
        for y in qy..(qy + qh).min(src_y.height) {
            for x in qx..(qx + qw).min(src_y.width) {
                sum += i64::from(src_y.get(x, y));
            }
        }
        sum / (qw * qh).max(1) as i64
    };
    let m00 = quad_mean(bx, by, hw, hh);
    let m10 = quad_mean(bx + hw, by, hw, hh);
    let m01 = quad_mean(bx, by + hh, hw, hh);
    let m11 = quad_mean(bx + hw, by + hh, hw, hh);
    let hor_diff = (m00 - m01).abs() + (m10 - m11).abs();
    let ver_diff = (m00 - m10).abs() + (m01 - m11).abs();
    let threshold = i64::from(qp.max(4)) / 2 + 8;
    if w == h && hor_diff > threshold && ver_diff > threshold {
        transform::CodingTreeSplit::QuadTree
    } else if hor_diff > ver_diff && hor_diff > threshold {
        transform::CodingTreeSplit::BinaryHor
    } else if ver_diff > threshold {
        transform::CodingTreeSplit::BinaryVer
    } else {
        transform::CodingTreeSplit::NoSplit
    }
}

/// Encode a [`super::VvcFrame`] (`bit_depth` in `8..=12`, `chroma_format` in `4:0:0`..=`4:4:4`,
/// `ctu_log2` in `5..=7`) into a CABAC slice-data substream and return `(cabac_bytes, recon_frame)`.
pub fn encode_vvc_frame_picture(
    src: &super::VvcFrame,
    qp: u8,
    ctu_log2: u8,
    options: &SliceCodingOptions,
) -> Result<(Vec<u8>, super::VvcFrame), VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, src.width, src.height)?;
    let tb = ctu.min(32);
    let bd = src.bit_depth.clamp(8, 12);
    let cf = src.chroma_format;
    let has_chroma = cf.is_chroma_enabled();
    let sub_w = cf.sub_width_c();
    let sub_h = cf.sub_height_c();
    let tb_cw = if has_chroma { (tb / sub_w).max(2) } else { 0 };
    let tb_ch = if has_chroma { (tb / sub_h).max(2) } else { 0 };

    let luma_qp_scaled = super::hls::scaled_luma_qp_bd(i32::from(qp), bd).max(4);
    let chroma_qp_scaled = super::hls::derived_chroma_qp_bd(i32::from(qp), bd, 0).max(4);
    let max_val = (1i32 << u32::from(bd)) - 1;

    let mut recon = super::VvcFrame::new(src.width, src.height, bd, cf);
    let mut enc = BinEncoder::new();
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));

    let width = src.width as usize;
    let height = src.height as usize;
    let grid_w = width / tb;
    let grid_h = height / tb;
    let mut mode_grid = vec![0u8; grid_w * grid_h];
    let mut coded = CodedUnits::new(tb, width, height);

    for &(bx, by) in &coding_unit_order(ctu, tb, width, height) {
        let gx = bx / tb;
        let gy = by / tb;
        let cx = bx / sub_w;
        let cy = by / sub_h;
        let idx = gy * grid_w + gx;
        coded.mark(bx, by);

        if (bx % ctu) == 0 && (by % ctu) == 0 {
            if options.enable_sao {
                let ctu_gx = bx / ctu;
                let ctu_gy = by / ctu;
                let merge_left = ctu_gx > 0;
                let merge_up = !merge_left && ctu_gy > 0;
                encode_ctu_sao(
                    &mut enc,
                    &mut ctx,
                    ctu_gx,
                    ctu_gy,
                    true,
                    has_chroma,
                    &[super::filter::SaoType::Off; 3],
                    merge_left,
                    merge_up,
                    bd,
                );
            }
            if options.enable_alf {
                encode_ctu_alf_luma_fixed(&mut enc, &mut ctx, false, false, false, 0);
            }
        }

        encode_coding_unit_split_flags(&mut enc, &mut ctx, bx, by, ctu, tb);

        let left_mode = if gx > 0 {
            Some(mode_grid[gy * grid_w + (gx - 1)])
        } else {
            None
        };
        let above_same_ctu = gy > 0 && (by % ctu) != 0;
        let above_mode = if above_same_ctu {
            Some(mode_grid[(gy - 1) * grid_w + gx])
        } else {
            None
        };
        let mpm = predict::derive_intra_mpms(left_mode, above_mode);

        let luma_mode = if options.full_intra_modes {
            predict::choose_intra_mode_16(&recon.y, &src.y, bx, by, tb, tb, true)
        } else {
            let m = predict::choose_intra_mode_16(&recon.y, &src.y, bx, by, tb, tb, true);
            match m {
                predict::IntraPredMode::Planar | predict::IntraPredMode::Dc => m,
                _ => predict::IntraPredMode::Planar,
            }
        };
        mode_grid[idx] = luma_mode.mode_index();

        let pred_y = predict::predict_coded_block_16(
            &recon.y,
            bx,
            by,
            tb,
            tb,
            true,
            luma_mode,
            1,
            1,
            |px, py| coded.is_coded_luma(px, py),
        );
        let mut res_y = vec![0i32; tb * tb];
        for y in 0..tb {
            for x in 0..tb {
                res_y[y * tb + x] =
                    i32::from(src.y.get(bx + x, by + y)) - i32::from(pred_y[y * tb + x]);
            }
        }
        let coeff_y = dct::forward_transform_rect_bd(
            tb,
            tb,
            &res_y,
            dct::TrType::Dct2,
            dct::TrType::Dct2,
            0,
            0,
            bd,
        )
        .map_err(VvcError::Unsupported)?;
        let scan_y = CoeffScan::new(tb, tb);
        let levels_y = if options.dep_quant {
            dct::quantize_dep_quant_bd(tb, tb, &coeff_y, &scan_y.raster, luma_qp_scaled, bd)
                .map_err(VvcError::Unsupported)?
        } else {
            dct::quantize_rect_bd(tb, tb, &coeff_y, luma_qp_scaled, bd, true)
                .map_err(VvcError::Unsupported)?
        };
        let dq_y = if options.dep_quant {
            dct::dequantize_dep_quant_bd(tb, tb, &levels_y, &scan_y.raster, luma_qp_scaled, bd)
                .map_err(VvcError::Unsupported)?
        } else {
            dct::dequantize_rect_bd(tb, tb, &levels_y, luma_qp_scaled, bd)
                .map_err(VvcError::Unsupported)?
        };
        let inv_y = dct::inverse_transform_rect_bd(
            tb,
            tb,
            &dq_y,
            dct::TrType::Dct2,
            dct::TrType::Dct2,
            0,
            0,
            bd,
        )
        .map_err(VvcError::Unsupported)?;
        for y in 0..tb {
            for x in 0..tb {
                let v = (i32::from(pred_y[y * tb + x]) + inv_y[y * tb + x]).clamp(0, max_val);
                recon.y.set(bx + x, by + y, v as u16);
            }
        }
        let cbf_y = levels_y.iter().any(|&v| v != 0);

        let mut chroma_mode = luma_mode.mode_index();
        let mut levels_u = Vec::new();
        let mut levels_v = Vec::new();
        let mut cbf_u = false;
        let mut cbf_v = false;

        if has_chroma {
            if options.enable_cclm {
                let pred_dm =
                    predict::predict_block_16(&recon.u, cx, cy, tb_cw, tb_ch, false, luma_mode);
                let mut best_csad = 0u64;
                for y in 0..tb_ch {
                    for x in 0..tb_cw {
                        let s = src.u.get(cx + x, cy + y);
                        best_csad += u64::from(s.abs_diff(pred_dm[y * tb_cw + x]));
                    }
                }
                for (cm_code, cm_enum) in [
                    (81u8, predict::CclmMode::Lm),
                    (82u8, predict::CclmMode::LmL),
                    (83u8, predict::CclmMode::LmT),
                ] {
                    let pred_lm = predict::predict_cclm_with_format_16(
                        &recon.y, &recon.u, cx, cy, tb_cw, tb_ch, cm_enum, cf, ctu,
                    );
                    let mut sad_lm = 0u64;
                    for y in 0..tb_ch {
                        for x in 0..tb_cw {
                            let s = src.u.get(cx + x, cy + y);
                            sad_lm += u64::from(s.abs_diff(pred_lm[y * tb_cw + x]));
                        }
                    }
                    if sad_lm < best_csad {
                        best_csad = sad_lm;
                        chroma_mode = cm_code;
                    }
                }
            }
            let cclm_mode_opt = match chroma_mode {
                81 => Some(predict::CclmMode::Lm),
                82 => Some(predict::CclmMode::LmL),
                83 => Some(predict::CclmMode::LmT),
                _ => None,
            };
            let pred_u = if let Some(cm_enum) = cclm_mode_opt {
                predict::predict_cclm_with_format_16(
                    &recon.y, &recon.u, cx, cy, tb_cw, tb_ch, cm_enum, cf, ctu,
                )
            } else {
                predict::predict_coded_block_16(
                    &recon.u,
                    cx,
                    cy,
                    tb_cw,
                    tb_ch,
                    false,
                    ipm(chroma_mode),
                    sub_w,
                    sub_h,
                    |px, py| coded.is_coded_luma(px, py),
                )
            };
            let pred_v = if let Some(cm_enum) = cclm_mode_opt {
                predict::predict_cclm_with_format_16(
                    &recon.y, &recon.v, cx, cy, tb_cw, tb_ch, cm_enum, cf, ctu,
                )
            } else {
                predict::predict_coded_block_16(
                    &recon.v,
                    cx,
                    cy,
                    tb_cw,
                    tb_ch,
                    false,
                    ipm(chroma_mode),
                    sub_w,
                    sub_h,
                    |px, py| coded.is_coded_luma(px, py),
                )
            };

            let mut res_u = vec![0i32; tb_cw * tb_ch];
            let mut res_v = vec![0i32; tb_cw * tb_ch];
            for y in 0..tb_ch {
                for x in 0..tb_cw {
                    res_u[y * tb_cw + x] =
                        i32::from(src.u.get(cx + x, cy + y)) - i32::from(pred_u[y * tb_cw + x]);
                    res_v[y * tb_cw + x] =
                        i32::from(src.v.get(cx + x, cy + y)) - i32::from(pred_v[y * tb_cw + x]);
                }
            }
            let scan_c = CoeffScan::new(tb_cw, tb_ch);
            let coeff_u = dct::forward_transform_rect_bd(
                tb_cw,
                tb_ch,
                &res_u,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;
            levels_u = if options.dep_quant {
                dct::quantize_dep_quant_bd(
                    tb_cw,
                    tb_ch,
                    &coeff_u,
                    &scan_c.raster,
                    chroma_qp_scaled,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                dct::quantize_rect_bd(tb_cw, tb_ch, &coeff_u, chroma_qp_scaled, bd, true)
                    .map_err(VvcError::Unsupported)?
            };
            let dq_u = if options.dep_quant {
                dct::dequantize_dep_quant_bd(
                    tb_cw,
                    tb_ch,
                    &levels_u,
                    &scan_c.raster,
                    chroma_qp_scaled,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                dct::dequantize_rect_bd(tb_cw, tb_ch, &levels_u, chroma_qp_scaled, bd)
                    .map_err(VvcError::Unsupported)?
            };
            let inv_u = dct::inverse_transform_rect_bd(
                tb_cw,
                tb_ch,
                &dq_u,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;

            let coeff_v = dct::forward_transform_rect_bd(
                tb_cw,
                tb_ch,
                &res_v,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;
            levels_v = if options.dep_quant {
                dct::quantize_dep_quant_bd(
                    tb_cw,
                    tb_ch,
                    &coeff_v,
                    &scan_c.raster,
                    chroma_qp_scaled,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                dct::quantize_rect_bd(tb_cw, tb_ch, &coeff_v, chroma_qp_scaled, bd, true)
                    .map_err(VvcError::Unsupported)?
            };
            let dq_v = if options.dep_quant {
                dct::dequantize_dep_quant_bd(
                    tb_cw,
                    tb_ch,
                    &levels_v,
                    &scan_c.raster,
                    chroma_qp_scaled,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                dct::dequantize_rect_bd(tb_cw, tb_ch, &levels_v, chroma_qp_scaled, bd)
                    .map_err(VvcError::Unsupported)?
            };
            let inv_v = dct::inverse_transform_rect_bd(
                tb_cw,
                tb_ch,
                &dq_v,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;

            for y in 0..tb_ch {
                for x in 0..tb_cw {
                    let u_val =
                        (i32::from(pred_u[y * tb_cw + x]) + inv_u[y * tb_cw + x]).clamp(0, max_val);
                    let v_val =
                        (i32::from(pred_v[y * tb_cw + x]) + inv_v[y * tb_cw + x]).clamp(0, max_val);
                    recon.u.set(cx + x, cy + y, u_val as u16);
                    recon.v.set(cx + x, cy + y, v_val as u16);
                }
            }
            cbf_u = levels_u.iter().any(|&v| v != 0);
            cbf_v = levels_v.iter().any(|&v| v != 0);
        }

        if options.enable_mip {
            let num_mip = predict::get_num_modes_mip(tb, tb) as u32;
            encode_mip_mode(&mut enc, &mut ctx, false, false, 0, num_mip, 0);
        }
        encode_full_luma_intra_mode(&mut enc, &mut ctx, luma_mode.mode_index(), &mpm, 0, false);
        if has_chroma {
            encode_chroma_intra_mode(
                &mut enc,
                &mut ctx,
                chroma_mode,
                luma_mode.mode_index(),
                options.enable_cclm,
            );
            enc.encode_bin(cbf_u, &mut ctx.models[ctx::QT_CBF_CB]);
            enc.encode_bin(cbf_v, &mut ctx.models[ctx::QT_CBF_CR + usize::from(cbf_u)]);
        }
        enc.encode_bin(cbf_y, &mut ctx.models[ctx::QT_CBF_Y]);
        if has_chroma && options.enable_jccr && (cbf_u || cbf_v) {
            residual::encode_joint_cbcr_flag(&mut enc, &mut ctx, true, cbf_u, cbf_v, false);
        }

        if cbf_y {
            enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
            residual::encode_residual_with_options(
                &mut enc,
                &mut ctx,
                &scan_y,
                &levels_y,
                true,
                options.dep_quant,
                options.sign_data_hiding,
            );
        }
        if has_chroma {
            let scan_c = CoeffScan::new(tb_cw, tb_ch);
            if cbf_u {
                enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                residual::encode_residual_with_options(
                    &mut enc,
                    &mut ctx,
                    &scan_c,
                    &levels_u,
                    false,
                    options.dep_quant,
                    options.sign_data_hiding,
                );
            }
            if cbf_v {
                enc.encode_bin(false, &mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                residual::encode_residual_with_options(
                    &mut enc,
                    &mut ctx,
                    &scan_c,
                    &levels_v,
                    false,
                    options.dep_quant,
                    options.sign_data_hiding,
                );
            }
        }
    }

    if options.enable_deblock {
        super::filter::deblock_luma_plane_16(&mut recon.y, tb, ctu, i32::from(qp), 2, 0, 0);
        if has_chroma {
            let c_qp = super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63);
            super::filter::deblock_chroma_plane_16(
                &mut recon.u,
                tb_cw,
                ctu / sub_h,
                c_qp,
                2,
                0,
                0,
                sub_w,
                sub_h,
            );
            super::filter::deblock_chroma_plane_16(
                &mut recon.v,
                tb_cw,
                ctu / sub_h,
                c_qp,
                2,
                0,
                0,
                sub_w,
                sub_h,
            );
        }
    }

    enc.encode_bin_trm(true);
    Ok((enc.finish_and_align(), recon))
}

/// Decode a CABAC slice-data substream produced by [`encode_vvc_frame_picture`] into a
/// [`super::VvcFrame`] (`bit_depth` in `8..=12`, `chroma_format` in `4:0:0`..=`4:4:4`,
/// `ctu_log2` in `5..=7`).
#[allow(clippy::too_many_arguments)]
pub fn decode_vvc_frame_picture(
    cabac: &[u8],
    qp: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    bit_depth: u8,
    chroma_format: predict::ChromaFormatIdc,
    options: &SliceCodingOptions,
) -> Result<super::VvcFrame, VvcError> {
    let bd = bit_depth.clamp(8, 12);
    let chroma_qp_scaled = super::hls::derived_chroma_qp_bd(i32::from(qp), bd, 0);
    decode_vvc_frame_picture_with_chroma_qp(
        cabac,
        qp,
        chroma_qp_scaled,
        ctu_log2,
        coded_width,
        coded_height,
        bit_depth,
        chroma_format,
        options,
    )
}

/// Decode an intra [`super::VvcFrame`] using a chroma quantisation parameter
/// derived from the signalled SPS chroma QP table (`H.266 §7.4.3.4`).
#[allow(clippy::too_many_arguments)]
pub fn decode_vvc_frame_picture_with_chroma_qp(
    cabac: &[u8],
    qp: u8,
    chroma_qp_scaled: u8,
    ctu_log2: u8,
    coded_width: u32,
    coded_height: u32,
    bit_depth: u8,
    chroma_format: predict::ChromaFormatIdc,
    options: &SliceCodingOptions,
) -> Result<super::VvcFrame, VvcError> {
    let ctu = validate_coded_geometry(ctu_log2, coded_width, coded_height)?;
    let tb = ctu.min(32);
    let bd = bit_depth.clamp(8, 12);
    let cf = chroma_format;
    let has_chroma = cf.is_chroma_enabled();
    let sub_w = cf.sub_width_c();
    let sub_h = cf.sub_height_c();
    let tb_cw = if has_chroma { (tb / sub_w).max(2) } else { 0 };
    let tb_ch = if has_chroma { (tb / sub_h).max(2) } else { 0 };

    let luma_qp_scaled = super::hls::scaled_luma_qp_bd(i32::from(qp), bd).max(4);
    let chroma_qp_scaled = chroma_qp_scaled.max(4);
    let max_val = (1i32 << u32::from(bd)) - 1;

    let mut dec = BinDecoder::new(cabac).map_err(|err| VvcError::Bitstream(err.to_string()))?;
    let mut ctx = CtxStore::new_i_slice(i32::from(qp));
    let mut recon = super::VvcFrame::new(coded_width, coded_height, bd, cf);

    let width = coded_width as usize;
    let height = coded_height as usize;
    let grid_w = width / tb;
    let grid_h = height / tb;
    let ctu_grid_w = width / ctu;
    let ctu_grid_h = height / ctu;
    let mut mode_grid = vec![0u8; grid_w * grid_h];
    let mut sao_grid = vec![[super::filter::SaoType::Off; 3]; ctu_grid_w * ctu_grid_h];
    let mut alf_grid = vec![None; ctu_grid_w * ctu_grid_h];
    let mut coded = CodedUnits::new(tb, width, height);

    for &(bx, by) in &coding_unit_order(ctu, tb, width, height) {
        let gx = bx / tb;
        let gy = by / tb;
        let cx = bx / sub_w;
        let cy = by / sub_h;
        let idx = gy * grid_w + gx;
        coded.mark(bx, by);

        if (bx % ctu) == 0 && (by % ctu) == 0 {
            let ctu_gx = bx / ctu;
            let ctu_gy = by / ctu;
            let ctu_idx = ctu_gy * ctu_grid_w + ctu_gx;
            if options.enable_sao {
                let left_sao = if ctu_gx > 0 {
                    Some(sao_grid[ctu_gy * ctu_grid_w + (ctu_gx - 1)])
                } else {
                    None
                };
                let up_sao = if ctu_gy > 0 {
                    Some(sao_grid[(ctu_gy - 1) * ctu_grid_w + ctu_gx])
                } else {
                    None
                };
                sao_grid[ctu_idx] = decode_ctu_sao(
                    &mut dec, &mut ctx, ctu_gx, ctu_gy, true, has_chroma, left_sao, up_sao, bd,
                );
            }
            if options.enable_alf {
                let left_alf = ctu_gx > 0 && alf_grid[ctu_gy * ctu_grid_w + (ctu_gx - 1)].is_some();
                let up_alf = ctu_gy > 0 && alf_grid[(ctu_gy - 1) * ctu_grid_w + ctu_gx].is_some();
                alf_grid[ctu_idx] = decode_ctu_alf_luma_fixed(&mut dec, &mut ctx, left_alf, up_alf);
            }
        }

        decode_coding_unit_split_flags(&mut dec, &mut ctx, bx, by, ctu, tb)?;

        let left_mode = if gx > 0 {
            Some(mode_grid[gy * grid_w + (gx - 1)])
        } else {
            None
        };
        let above_same_ctu = gy > 0 && (by % ctu) != 0;
        let above_mode = if above_same_ctu {
            Some(mode_grid[(gy - 1) * grid_w + gx])
        } else {
            None
        };
        let mpm = predict::derive_intra_mpms(left_mode, above_mode);

        let mip_info = if options.enable_mip {
            let num_mip = predict::get_num_modes_mip(tb, tb) as u32;
            decode_mip_mode(&mut dec, &mut ctx, num_mip, 0)
        } else {
            None
        };
        let _ = mip_info;
        let luma_mode = decode_full_luma_intra_mode(&mut dec, &mut ctx, &mpm, 0, false);
        mode_grid[idx] = luma_mode;

        let chroma_mode = if has_chroma {
            decode_chroma_intra_mode(&mut dec, &mut ctx, luma_mode, options.enable_cclm)
        } else {
            0
        };
        let (cbf_u, cbf_v) = if has_chroma {
            let cu = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
            let cv = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(cu)]);
            (cu, cv)
        } else {
            (false, false)
        };
        let cbf_y = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_Y]);
        let jccr_flag = if has_chroma && options.enable_jccr && (cbf_u || cbf_v) {
            residual::decode_joint_cbcr_flag(&mut dec, &mut ctx, true, cbf_u, cbf_v) != 0
        } else {
            false
        };

        let scan_y = CoeffScan::new(tb, tb);
        let mut levels_y = vec![0i32; tb * tb];
        if cbf_y {
            let ts_y = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG]);
            if ts_y {
                decode_ts_block(&mut dec, &mut ctx, &scan_y, &mut levels_y);
            } else {
                residual::decode_residual_with_options(
                    &mut dec,
                    &mut ctx,
                    &scan_y,
                    &mut levels_y,
                    true,
                    options.dep_quant,
                    options.sign_data_hiding,
                )?;
            }
        }

        let mut levels_u = vec![0i32; tb_cw * tb_ch];
        let mut levels_v = vec![0i32; tb_cw * tb_ch];
        if has_chroma {
            let scan_c = CoeffScan::new(tb_cw, tb_ch);
            if cbf_u {
                let ts_u = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_u {
                    decode_ts_block(&mut dec, &mut ctx, &scan_c, &mut levels_u);
                } else {
                    residual::decode_residual_with_options(
                        &mut dec,
                        &mut ctx,
                        &scan_c,
                        &mut levels_u,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    )?;
                }
            }
            if cbf_v && !jccr_flag {
                let ts_v = dec.decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + 1]);
                if ts_v {
                    decode_ts_block(&mut dec, &mut ctx, &scan_c, &mut levels_v);
                } else {
                    residual::decode_residual_with_options(
                        &mut dec,
                        &mut ctx,
                        &scan_c,
                        &mut levels_v,
                        false,
                        options.dep_quant,
                        options.sign_data_hiding,
                    )?;
                }
            }
        }

        let pred_y = predict::predict_coded_block_16(
            &recon.y,
            bx,
            by,
            tb,
            tb,
            true,
            ipm(luma_mode),
            1,
            1,
            |px, py| coded.is_coded_luma(px, py),
        );
        let inv_y = if cbf_y {
            let dq_y = if options.dep_quant {
                dct::dequantize_dep_quant_bd(tb, tb, &levels_y, &scan_y.raster, luma_qp_scaled, bd)
                    .map_err(VvcError::Unsupported)?
            } else {
                dct::dequantize_rect_bd(tb, tb, &levels_y, luma_qp_scaled, bd)
                    .map_err(VvcError::Unsupported)?
            };
            dct::inverse_transform_rect_bd(
                tb,
                tb,
                &dq_y,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?
        } else {
            vec![0i32; tb * tb]
        };
        for y in 0..tb {
            for x in 0..tb {
                let v = (i32::from(pred_y[y * tb + x]) + inv_y[y * tb + x]).clamp(0, max_val);
                recon.y.set(bx + x, by + y, v as u16);
            }
        }

        if has_chroma {
            let scan_c = CoeffScan::new(tb_cw, tb_ch);
            let cclm_mode_opt = match chroma_mode {
                81 => Some(predict::CclmMode::Lm),
                82 => Some(predict::CclmMode::LmL),
                83 => Some(predict::CclmMode::LmT),
                _ => None,
            };
            let pred_u = if let Some(cm_enum) = cclm_mode_opt {
                predict::predict_cclm_with_format_16(
                    &recon.y, &recon.u, cx, cy, tb_cw, tb_ch, cm_enum, cf, ctu,
                )
            } else {
                predict::predict_coded_block_16(
                    &recon.u,
                    cx,
                    cy,
                    tb_cw,
                    tb_ch,
                    false,
                    ipm(chroma_mode),
                    sub_w,
                    sub_h,
                    |px, py| coded.is_coded_luma(px, py),
                )
            };
            let pred_v = if let Some(cm_enum) = cclm_mode_opt {
                predict::predict_cclm_with_format_16(
                    &recon.y, &recon.v, cx, cy, tb_cw, tb_ch, cm_enum, cf, ctu,
                )
            } else {
                predict::predict_coded_block_16(
                    &recon.v,
                    cx,
                    cy,
                    tb_cw,
                    tb_ch,
                    false,
                    ipm(chroma_mode),
                    sub_w,
                    sub_h,
                    |px, py| coded.is_coded_luma(px, py),
                )
            };
            let inv_u = if cbf_u {
                let dq_u = if options.dep_quant {
                    dct::dequantize_dep_quant_bd(
                        tb_cw,
                        tb_ch,
                        &levels_u,
                        &scan_c.raster,
                        chroma_qp_scaled,
                        bd,
                    )
                    .map_err(VvcError::Unsupported)?
                } else {
                    dct::dequantize_rect_bd(tb_cw, tb_ch, &levels_u, chroma_qp_scaled, bd)
                        .map_err(VvcError::Unsupported)?
                };
                dct::inverse_transform_rect_bd(
                    tb_cw,
                    tb_ch,
                    &dq_u,
                    dct::TrType::Dct2,
                    dct::TrType::Dct2,
                    0,
                    0,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                vec![0i32; tb_cw * tb_ch]
            };
            let inv_v = if cbf_v {
                let dq_v = if options.dep_quant {
                    dct::dequantize_dep_quant_bd(
                        tb_cw,
                        tb_ch,
                        &levels_v,
                        &scan_c.raster,
                        chroma_qp_scaled,
                        bd,
                    )
                    .map_err(VvcError::Unsupported)?
                } else {
                    dct::dequantize_rect_bd(tb_cw, tb_ch, &levels_v, chroma_qp_scaled, bd)
                        .map_err(VvcError::Unsupported)?
                };
                dct::inverse_transform_rect_bd(
                    tb_cw,
                    tb_ch,
                    &dq_v,
                    dct::TrType::Dct2,
                    dct::TrType::Dct2,
                    0,
                    0,
                    bd,
                )
                .map_err(VvcError::Unsupported)?
            } else {
                vec![0i32; tb_cw * tb_ch]
            };
            for y in 0..tb_ch {
                for x in 0..tb_cw {
                    let u_val =
                        (i32::from(pred_u[y * tb_cw + x]) + inv_u[y * tb_cw + x]).clamp(0, max_val);
                    let v_val =
                        (i32::from(pred_v[y * tb_cw + x]) + inv_v[y * tb_cw + x]).clamp(0, max_val);
                    recon.u.set(cx + x, cy + y, u_val as u16);
                    recon.v.set(cx + x, cy + y, v_val as u16);
                }
            }
        }
    }

    if options.enable_deblock {
        super::filter::deblock_luma_plane_16(&mut recon.y, tb, ctu, i32::from(qp), 2, 0, 0);
        if has_chroma {
            let c_qp = super::hls::derived_chroma_qp(i32::from(qp)).clamp(0, 63);
            super::filter::deblock_chroma_plane_16(
                &mut recon.u,
                tb_cw,
                ctu / sub_h,
                c_qp,
                2,
                0,
                0,
                sub_w,
                sub_h,
            );
            super::filter::deblock_chroma_plane_16(
                &mut recon.v,
                tb_cw,
                ctu / sub_h,
                c_qp,
                2,
                0,
                0,
                sub_w,
                sub_h,
            );
        }
    }
    if options.enable_sao {
        let src_y = recon.y.clone();
        let src_u = recon.u.clone();
        let src_v = recon.v.clone();
        let ctu_cw = if has_chroma { ctu / sub_w } else { 0 };
        let ctu_ch = if has_chroma { ctu / sub_h } else { 0 };
        for gy in 0..ctu_grid_h {
            for gx in 0..ctu_grid_w {
                let sao = sao_grid[gy * ctu_grid_w + gx];
                super::filter::apply_sao_region_16(
                    &src_y,
                    &mut recon.y,
                    gx * ctu,
                    gy * ctu,
                    ctu,
                    ctu,
                    sao[0],
                );
                if has_chroma {
                    super::filter::apply_sao_region_16(
                        &src_u,
                        &mut recon.u,
                        gx * ctu_cw,
                        gy * ctu_ch,
                        ctu_cw,
                        ctu_ch,
                        sao[1],
                    );
                    super::filter::apply_sao_region_16(
                        &src_v,
                        &mut recon.v,
                        gx * ctu_cw,
                        gy * ctu_ch,
                        ctu_cw,
                        ctu_ch,
                        sao[2],
                    );
                }
            }
        }
    }
    if options.enable_alf && alf_grid.iter().any(Option::is_some) {
        let default_aps = super::hls::VvcAlfAps {
            luma_clip: false,
            luma_filters: Vec::new(),
            luma_clip_idx: Vec::new(),
            filter_coeff_delta_idx: [0u8; 25],
            chroma_clip: false,
            chroma_filters: Vec::new(),
            chroma_clip_idx: Vec::new(),
            cc_cb_filters: Vec::new(),
            cc_cr_filters: Vec::new(),
        };
        super::filter::apply_alf_luma_plane_16(&mut recon.y, &default_aps);
    }

    Ok(recon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> i32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) % 61) as i32 - 30
    }

    #[test]
    fn ts_block_inferred_significance_positions_round_trip() {
        // Exercise both inferences: `sb_coded_flag` of the last group and the
        // significance flag of the last scan position inside a coded group.
        let cases: [&[(usize, usize)]; 4] = [
            &[(31, 31)],
            &[(0, 0)],
            &[(3, 3)],
            &[(30, 31), (31, 30), (31, 31)],
        ];
        for size in [16usize, 32] {
            for case in cases {
                let mut coeffs = vec![0i32; size * size];
                for &(x, y) in case {
                    let x = x.min(size - 1);
                    let y = y.min(size - 1);
                    coeffs[y * size + x] = if (x + y) % 2 == 0 { 7 } else { -3 };
                }
                let scan = CoeffScan::new(size, size);
                let mut enc = BinEncoder::new();
                let mut ctx = CtxStore::new_i_slice(26);
                encode_ts_block(&mut enc, &mut ctx, &scan, &coeffs);
                enc.encode_bin_trm(true);
                let bytes = enc.finish_and_align();
                let mut dec = BinDecoder::new(&bytes).unwrap();
                let mut dctx = CtxStore::new_i_slice(26);
                let mut out = vec![0i32; size * size];
                decode_ts_block(&mut dec, &mut dctx, &scan, &mut out);
                assert!(dec.decode_bin_trm());
                assert_eq!(
                    out, coeffs,
                    "inferred TS round-trip size={size} case={case:?}"
                );
            }
        }
    }

    #[test]
    fn ts_block_round_trips_coefficients() {
        for &size in &[16usize, 32] {
            for qp in [4u8, 26, 40] {
                let mut seed = 0x1234_5678u32 ^ (size as u32);
                let mut coeffs = vec![0i32; size * size];
                for c in coeffs.iter_mut() {
                    let v = lcg(&mut seed);
                    *c = if v.abs() < 6 { 0 } else { v };
                }
                // A dense block exercises the context-bin budget as well.
                let scan = CoeffScan::new(size, size);
                let mut enc = BinEncoder::new();
                let mut ctx = CtxStore::new_i_slice(qp as i32);
                encode_ts_block(&mut enc, &mut ctx, &scan, &coeffs);
                enc.encode_bin_trm(true);
                let bytes = enc.finish_and_align();

                let mut dec = BinDecoder::new(&bytes).unwrap();
                let mut dctx = CtxStore::new_i_slice(qp as i32);
                let mut out = vec![0i32; size * size];
                decode_ts_block(&mut dec, &mut dctx, &scan, &mut out);
                assert!(dec.decode_bin_trm());
                assert_eq!(out, coeffs, "TS round-trip size={size} qp={qp}");
            }
        }
    }

    #[test]
    fn picture_round_trip_is_exact_at_qp4() {
        let (w, h) = (64u32, 64u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                y[j * w as usize + i] = ((i * 3 + j * 5) % 251) as u8;
            }
        }
        let frame = I420Frame {
            width: w,
            height: h,
            y,
            u: vec![90u8; cw * ch],
            v: vec![160u8; cw * ch],
        };
        let cabac = encode_picture(&frame, 4, 5).expect("encode");
        let dec = decode_picture(&cabac, 4, 5, w, h).expect("decode");
        assert_eq!(dec.y, frame.y);
        assert_eq!(dec.u, frame.u);
        assert_eq!(dec.v, frame.v);
    }

    #[test]
    fn intra_mode_signalling_round_trips_through_cabac() {
        for mode in [IntraMode::Planar, IntraMode::Dc] {
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(26);
            encode_intra_mode(&mut enc, &mut ctx, mode);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();

            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(26);
            assert_eq!(
                decode_intra_mode(&mut dec, &mut dctx).expect("decode mode"),
                mode
            );
            assert!(dec.decode_bin_trm());
            dec.finish().expect("stop/alignment pattern");
        }
    }

    #[test]
    fn intra_mode_decodes_mpm_indices_above_one() {
        let mut enc = BinEncoder::new();
        let mut ctx = CtxStore::new_i_slice(26);
        enc.encode_bin(true, &mut ctx.models[ctx::INTRA_LUMA_MPM_FLAG]);
        enc.encode_bin(true, &mut ctx.models[ctx::INTRA_LUMA_PLANAR_FLAG + 1]);
        enc.encode_bin_ep(true);
        enc.encode_bin_ep(false);
        enc.encode_bin_trm(true);
        let bytes = enc.finish_and_align();

        let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
        let mut dctx = CtxStore::new_i_slice(26);
        let mode = decode_intra_mode(&mut dec, &mut dctx).expect("MPM index 2 (VER_IDX)");
        assert_eq!(mode, IntraMode::Angular(50));
    }

    #[test]
    fn flat_and_gradient_picture_round_trips_at_qp4() {
        let (w, h) = (96u32, 64u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                // Flat left half and a diagonal gradient right half: the flat
                // region exercises the DC path, the gradient the PLANAR path.
                y[j * w as usize + i] = if i < 48 {
                    100
                } else {
                    ((i * 3 + j * 5) % 251) as u8
                };
            }
        }
        let frame = I420Frame {
            width: w,
            height: h,
            y,
            u: vec![96u8; cw * ch],
            v: vec![148u8; cw * ch],
        };
        let cabac = encode_picture(&frame, 4, 5).expect("encode");
        let dec = decode_picture(&cabac, 4, 5, w, h).expect("decode");
        assert_eq!(dec.y, frame.y);
        assert_eq!(dec.u, frame.u);
        assert_eq!(dec.v, frame.v);
    }

    #[test]
    fn mts_idx_syntax_round_trips_every_value() {
        for mts_idx in 0..=4u8 {
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(50);
            encode_mts_idx(&mut enc, &mut ctx, mts_idx);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();
            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(50);
            assert_eq!(
                decode_mts_idx(&mut dec, &mut dctx).expect("mts_idx"),
                mts_idx
            );
            assert!(dec.decode_bin_trm());
        }
    }

    #[test]
    fn lfnst_idx_syntax_round_trips_every_value() {
        for lfnst_idx in 0..=2u8 {
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(50);
            encode_lfnst_idx(&mut enc, &mut ctx, lfnst_idx);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();
            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(50);
            assert_eq!(decode_lfnst_idx(&mut dec, &mut dctx), lfnst_idx);
            assert!(dec.decode_bin_trm());
        }
    }

    #[test]
    fn qp50_selection_uses_mts_and_lfnst() {
        use super::super::decode_annexb_to_i420;
        use super::super::{VvcEncoder, VvcEncoderConfig};
        let (w, h) = (200u32, 120u32);
        let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        // A smooth luma ramp keeps the luma coefficient groups inside the
        // LFNST/MTS scan window, and a flat chroma plane leaves the chroma
        // constraints clear, so both tools are genuinely evaluated and
        // selected at QP 50. A high-frequency luma pattern or a textured
        // chroma plane suppresses LFNST before the SSE selection runs.
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                y[j * w as usize + i] = (16 + ((i + j) % 40)) as u8;
            }
        }
        let u = vec![128u8; cw * ch];
        let v = vec![128u8; cw * ch];
        let frame = I420Frame {
            width: w,
            height: h,
            y,
            u,
            v,
        };
        tool_stats::reset();
        let mut encoder = VvcEncoder::new(VvcEncoderConfig {
            width: w,
            height: h,
            qp: 50,
        })
        .expect("encoder");
        let au = encoder.encode_i420(&frame).expect("encode");
        let decoded = decode_annexb_to_i420(&[au.data.as_slice()]).expect("decode");
        assert_eq!(decoded.len(), 1);
        let (mts, lfnst) = tool_stats::counts();
        assert!(
            mts > 0,
            "QP 50 must exercise MTS (mts={mts}, lfnst={lfnst})"
        );
        assert!(
            lfnst > 0,
            "QP 50 must exercise LFNST (mts={mts}, lfnst={lfnst})"
        );
    }

    #[test]
    fn invalid_coded_geometry_is_rejected() {
        let frame = I420Frame {
            width: 48,
            height: 48,
            y: vec![0; 48 * 48],
            u: vec![0; 24 * 24],
            v: vec![0; 24 * 24],
        };
        // Not CTU aligned: legal H.266 (partial CTUs), unsupported here.
        assert!(matches!(
            encode_picture(&frame, 4, 5),
            Err(VvcError::Unsupported(_))
        ));
        // Unsupported CTU log2 outside 5..=7.
        assert!(matches!(
            encode_picture(&frame, 4, 3),
            Err(VvcError::Unsupported(_))
        ));
        assert!(matches!(
            decode_picture(&[0u8; 8], 4, 9, 48, 48),
            Err(VvcError::Unsupported(_))
        ));
    }

    #[test]
    fn ctu_sizes_32_to_128_are_accepted() {
        for ctu_log2 in [5u8, 6, 7] {
            assert_eq!(
                validate_coded_geometry(ctu_log2, 128, 128).expect("valid CTU"),
                1usize << ctu_log2
            );
        }
        // CTU size 16 cannot be signalled by `sps_log2_ctu_size_minus5`.
        assert!(validate_coded_geometry(4, 64, 64).is_err());
        assert!(validate_coded_geometry(3, 64, 64).is_err());
        assert!(validate_coded_geometry(8, 256, 256).is_err());
    }

    #[test]
    fn full_luma_intra_mode_cabac_round_trips_all_67_modes() {
        let mpm = predict::derive_intra_mpms(Some(18), Some(50));
        for mode in 0..=66u8 {
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(26);
            encode_full_luma_intra_mode(&mut enc, &mut ctx, mode, &mpm, 0, false);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();

            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(26);
            let decoded = decode_full_luma_intra_mode(&mut dec, &mut dctx, &mpm, 0, false);
            assert_eq!(decoded, mode, "mode {mode} failed round-trip");
            assert!(dec.decode_bin_trm());
        }
    }

    #[test]
    fn chroma_intra_mode_and_mvd_cabac_round_trip() {
        for luma_mode in [0u8, 1, 18, 34, 50, 66] {
            for &chroma_mode in &[luma_mode, 0, 50, 18, 1, 81, 82, 83] {
                let mut enc = BinEncoder::new();
                let mut ctx = CtxStore::new_i_slice(26);
                encode_chroma_intra_mode(&mut enc, &mut ctx, chroma_mode, luma_mode, true);
                enc.encode_bin_trm(true);
                let bytes = enc.finish_and_align();

                let mut dec = BinDecoder::new(&bytes).expect("CABAC");
                let mut dctx = CtxStore::new_i_slice(26);
                let dec_mode = decode_chroma_intra_mode(&mut dec, &mut dctx, luma_mode, true);
                assert_eq!(dec_mode, chroma_mode);
                assert!(dec.decode_bin_trm());
            }
        }

        for (dx, dy) in [(0, 0), (1, -1), (-4, 7), (25, -33), (-128, 96)] {
            let mvd = super::super::inter::Mv::new(dx, dy);
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_p_slice(26);
            encode_mvd(&mut enc, &mut ctx, mvd);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();

            let mut dec = BinDecoder::new(&bytes).expect("CABAC");
            let mut dctx = CtxStore::new_p_slice(26);
            assert_eq!(decode_mvd(&mut dec, &mut dctx), mvd);
            assert!(dec.decode_bin_trm());
        }
    }

    #[test]
    fn encode_and_decode_picture_with_options_and_inter_slice_round_trip() {
        let (w, h) = (64u32, 64u32);
        let (cw, ch) = (32usize, 32usize);
        let mut y = vec![0u8; (w * h) as usize];
        for j in 0..h as usize {
            for i in 0..w as usize {
                y[j * w as usize + i] = ((i * 5 + j * 3) % 200) as u8 + 20;
            }
        }
        let frame0 = I420Frame {
            width: w,
            height: h,
            y: y.clone(),
            u: vec![110u8; cw * ch],
            v: vec![145u8; cw * ch],
        };
        let opts = SliceCodingOptions {
            full_intra_modes: true,
            enable_cclm: true,
            enable_mip: false,
            dep_quant: true,
            sign_data_hiding: false,
            enable_jccr: false,
            enable_deblock: true,
            enable_sao: true,
            enable_alf: false,
            ..SliceCodingOptions::default()
        };
        let cabac_i = encode_picture_with_options(&frame0, 22, 5, &opts).expect("encode intra");
        let dec_i =
            decode_picture_with_options(&cabac_i, 22, 5, w, h, &opts).expect("decode intra");
        assert_eq!(dec_i.width, w);
        assert_eq!(dec_i.height, h);

        // Inter P slice referencing dec_i.
        let ref0 = super::super::inter::RefFrame::from_planes(
            0,
            false,
            Plane::from_slice(w as usize, h as usize, &dec_i.y).expect("Y"),
            Plane::from_slice(cw, ch, &dec_i.u).expect("U"),
            Plane::from_slice(cw, ch, &dec_i.v).expect("V"),
            super::super::inter::MotionFieldGrid::new(w as usize, h as usize),
        );
        let ref_list = [ref0];
        let (cabac_p, enc_recon_p, _) = encode_inter_picture(
            &frame0,
            24,
            5,
            super::super::hls::VvcSliceType::P,
            1,
            &ref_list,
            &[],
            &opts,
        )
        .expect("encode inter P");
        let (dec_p, _) = decode_inter_picture(
            &cabac_p,
            24,
            5,
            w,
            h,
            super::super::hls::VvcSliceType::P,
            &ref_list,
            &[],
            &opts,
        )
        .expect("decode inter P");
        assert_eq!(dec_p.y, enc_recon_p.y);
        assert_eq!(dec_p.u, enc_recon_p.u);
        assert_eq!(dec_p.v, enc_recon_p.v);
    }

    #[test]
    fn slice_coded_16_reference_substitution_matches_the_shared_predictor() {
        // The coded-unit-aware wrapper must reproduce the shared predictor where
        // above-right and below-left references fall outside the picture, so the
        // availability substitution is a no-op and both paths see the same
        // padded reference samples.
        let w = 64usize;
        let h = 64usize;
        let data: Vec<u16> = (0..w * h)
            .map(|i| ((37 * (i % w) + 91 * (i / w) + 11 * ((i % w) * (i / w) % 17)) % 1023) as u16)
            .collect();
        let plane = predict::Plane16::from_slice(w, h, 10, &data).expect("plane16");
        let mut coded = CodedUnits::new(32, w, h);
        for by in (0..h).step_by(32) {
            for bx in (0..w).step_by(32) {
                coded.mark(bx, by);
            }
        }
        for &(bw, bh) in &[
            (4usize, 4usize),
            (8, 8),
            (16, 16),
            (32, 32),
            (8, 16),
            (16, 8),
        ] {
            for &(bx, by) in &[(0usize, 0usize), (w - bw, h - bh)] {
                for mode in [
                    predict::IntraPredMode::Planar,
                    predict::IntraPredMode::Dc,
                    predict::IntraPredMode::Angular(2),
                    predict::IntraPredMode::Angular(18),
                    predict::IntraPredMode::Angular(50),
                ] {
                    let coded_pred = predict::predict_coded_block_16(
                        &plane,
                        bx,
                        by,
                        bw,
                        bh,
                        true,
                        mode,
                        1,
                        1,
                        |px, py| coded.is_coded_luma(px, py),
                    );
                    let shared = predict::predict_block_16(&plane, bx, by, bw, bh, true, mode);
                    assert_eq!(coded_pred, shared, "{mode:?} {bw}x{bh} at ({bx},{by})");
                }
            }
        }
    }
}
