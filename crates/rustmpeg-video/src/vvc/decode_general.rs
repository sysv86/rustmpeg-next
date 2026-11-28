// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! General H.266 coding-tree, coding-unit, and transform-unit decoding for
//! intra pictures.
//!
//! Algorithms derived from VTM 24.0 (BSD-3-Clause, Copyright (c) 2010-2020
//! ITU/ISO/IEC). See `THIRD_PARTY.md`.
//!
//! This module implements the general quadtree / multi-type tree for one
//! intra slice with a single tile: `split_cu_flag`, `split_qt_flag`,
//! `mtt_split_cu_vertical_flag`, `mtt_split_cu_binary_flag`, the implicit
//! boundary splits of `H.266 §6.4.3`, and the dual-tree intra layout of
//! `sps_qtbtt_dual_tree_intra_flag`. Coding-unit and transform-tree syntax is
//! parsed per `H.266 §7.3.11`, coefficients per `§7.3.11.11`, and every block
//! is reconstructed with the shared prediction, transform, and loop-filter
//! primitives.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::cabac::{BinDecoder, CtxStore, ctx};
use super::filter::LmcsReshaper16;
use super::hls::{VvcActiveApsBank, VvcPictureHeader, VvcPps, VvcSliceHeader, VvcSps};
use super::predict::{self, IntraMode, Plane16};
use super::transform::CoeffScan;
use super::{VvcError, dct, filter, lfnst, residual, transform};

const MAX_TB: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TreeType {
    Luma,
    Chroma,
    Joint,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum Split {
    #[default]
    Ctu,
    Qt,
    Bh,
    Bv,
    Th,
    Tv,
    None,
}

/// Coding-unit mode type of a coding-tree node (`H.266 §7.4.12.4`,
/// VTM `ModeType`). `All` is the CTU root of an inter slice; `Intra` and
/// `Inter` are set when `mode_constraint()` signals or infers a mode type.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum ModeType {
    /// `MODE_TYPE_ALL`
    #[default]
    All,
    /// `MODE_TYPE_INTER`
    Inter,
    /// `MODE_TYPE_INTRA`
    Intra,
}

#[derive(Clone, Copy)]
struct Node {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    qt_depth: u8,
    mtt_depth: u8,
    bt_depth: u8,
    part_idx: u8,
    last_split: Split,
    implicit_bt_depth: u8,
    checkd_implicit: bool,
    implicit_split: Split,
    tree: TreeType,
    /// Mode type inherited from the parent split (`mode_constraint()`).
    mode_type: ModeType,
    /// Split series from the CTU root (`H.266 §8.4.4` / VTM `cu.splitSeries`).
    series: [u8; 8],
    series_len: u8,
}

impl Node {
    /// Split recorded at `depth` of the coding-tree split series, or
    /// [`Split::None`] when the node is not split that deep (`CU::getSplitAtDepth`).
    fn split_at(&self, depth: u8) -> Split {
        if depth >= self.series_len {
            return Split::None;
        }
        match self.series[depth as usize] {
            1 => Split::Qt,
            2 => Split::Bh,
            3 => Split::Bv,
            4 => Split::Th,
            5 => Split::Tv,
            _ => Split::None,
        }
    }

    fn push_split(&mut self, split: Split) {
        let idx = usize::from(self.series_len);
        if idx < self.series.len() {
            self.series[idx] = split_code(split);
            self.series_len += 1;
        }
    }
}

fn split_code(split: Split) -> u8 {
    if split == Split::None { 0 } else { split as u8 }
}

#[derive(Clone, Copy, Default)]
struct LumaCell {
    valid: bool,
    cu_w: u16,
    cu_h: u16,
    qt_depth: u8,
    intra: bool,
    mode: u8,
    mip: bool,
    /// Split at the 64x64 dual-tree node of the co-located luma CU (`0` = none).
    split64: u8,
    /// Whether the co-located luma CU uses ISP.
    isp: bool,
    /// `cu_skip_flag` of the co-located CU (merge/skip context derivation).
    skip: bool,
    /// `inter_affine_flag` / `merge_subblock_flag` of the co-located CU.
    affine: bool,
    /// Top-left corner of the owning coding unit (`pu.cu != neighbour.cu`).
    cu_x: u16,
    cu_y: u16,
    /// `inter_pred_idc` (`1` = L0, `2` = L1, `3` = BI).
    inter_dir: u8,
    /// `ref_idx_l0` / `ref_idx_l1` (`-1` = unused).
    ref_idx: [i8; 2],
    /// Motion vectors in 1/16-pel units.
    mv: [super::inter::Mv; 2],
    /// CU-level BCW weight index for bi-prediction.
    bcw: u8,
}

#[derive(Clone, Copy, Default)]
struct ChromaCell {
    valid: bool,
    cu_w: u16,
    cu_h: u16,
    qt_depth: u8,
}

#[derive(Clone, Copy)]
struct Lim {
    min_qt: usize,
    max_bt: usize,
    min_bt: usize,
    max_tt: usize,
    min_tt: usize,
    max_mtt: u8,
    max_tb: usize,
}

fn limits(g: &Gen<'_>, chroma: bool) -> Lim {
    let sps = g.sps;
    let min_cb = 1usize << sps.min_cb_log2;
    // VTM `PreCalcValues` (`Slice.h:1140-1185`, `getValIdx`): an intra slice uses
    // the intra-slice parameters (the chroma-specific ones for the chroma tree
    // of a dual tree), every other slice - including intra subtrees of an inter
    // slice - uses the inter-slice parameters for both channels.
    let inter = g.sh.slice_type != super::hls::VvcSliceType::I;
    // The picture header carries the effective partition limits: the
    // `ph_partition_constraints_override_*` values when the PH signals an
    // override, the SPS values otherwise (`H.266 §7.4.12.4`,
    // `VvcPictureHeader::log2_diff_min_qt_min_cb`).
    let channel = if inter {
        1
    } else if chroma {
        2
    } else {
        0
    };
    let min_qt_diff = g.ph.log2_diff_min_qt_min_cb[channel];
    let bt_diff = g.ph.log2_diff_max_bt_min_qt[channel];
    let tt_diff = g.ph.log2_diff_max_tt_min_qt[channel];
    let max_mtt = g.ph.max_mtt_depth[channel];
    let min_qt = min_cb << min_qt_diff;
    Lim {
        min_qt,
        max_bt: min_qt << bt_diff,
        min_bt: min_cb,
        max_tt: min_qt << tt_diff,
        min_tt: min_cb,
        max_mtt,
        max_tb: if sps.max_tb_size_64 { 64 } else { 32 },
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct CanSplit {
    no: bool,
    qt: bool,
    bh: bool,
    bv: bool,
    th: bool,
    tv: bool,
}

impl CanSplit {
    fn any_split(self) -> bool {
        self.qt || self.bh || self.bv || self.th || self.tv
    }
    fn any_btt(self) -> bool {
        self.bh || self.bv || self.th || self.tv
    }
}

fn implicit_split(node: &Node, lim: &Lim, pic_w: usize, pic_h: usize, dual: bool) -> Split {
    if node.checkd_implicit {
        return node.implicit_split;
    }
    let bl_in_pic = node.y + node.h - 1 < pic_h;
    let tr_in_pic = node.x + node.w - 1 < pic_w;
    let max_bt_depth = u32::from(lim.max_mtt) + u32::from(node.implicit_bt_depth);
    let bt_allowed =
        node.w <= lim.max_bt && node.h <= lim.max_bt && u32::from(node.mtt_depth) < max_bt_depth;
    let qt_allowed = node.w > lim.min_qt && node.h > lim.min_qt && node.bt_depth == 0;
    let mut split = Split::None;
    if !bl_in_pic && !tr_in_pic && qt_allowed {
        split = Split::Qt;
    } else if !bl_in_pic && bt_allowed && node.w <= MAX_TB {
        split = Split::Bh;
    } else if !tr_in_pic && bt_allowed && node.h <= MAX_TB {
        split = Split::Bv;
    } else if !bl_in_pic || !tr_in_pic {
        split = Split::Qt;
    }
    if dual && (node.w > 64 || node.h > 64) {
        split = Split::Qt;
    }
    if (!bl_in_pic || !tr_in_pic) && split == Split::None {
        split = Split::Qt;
    }
    split
}

fn can_split(g: &Gen<'_>, node: &Node, lim: &Lim, dual: bool, can: &mut CanSplit) {
    *can = CanSplit {
        no: true,
        qt: true,
        bh: true,
        bv: true,
        th: true,
        tv: true,
    };
    let area_c = if node.tree == TreeType::Chroma {
        Some((node.w / 2, node.h / 2))
    } else {
        None
    };
    let last = node.last_split;
    let parl = if last == Split::Th {
        Split::Bh
    } else {
        Split::Bv
    };
    if last != Split::Ctu && last != Split::Qt {
        can.qt = false;
    }
    if node.w <= lim.min_qt {
        can.qt = false;
    }
    if let Some((cw, _)) = area_c {
        if cw <= 4 {
            can.qt = false;
        }
    }
    let isplit = implicit_split(node, lim, g.pic_w, g.pic_h, dual);
    if isplit != Split::None {
        can.no = false;
        can.th = false;
        can.tv = false;
        can.bh = isplit == Split::Bh;
        can.bv = isplit == Split::Bv;
        if let Some((cw, _)) = area_c {
            if cw == 4 {
                can.bv = false;
            }
        }
        if !can.bh && !can.bv && !can.qt {
            can.qt = true;
        }
        return;
    }
    if (last == Split::Th || last == Split::Tv) && node.part_idx == 1 {
        can.bh = parl != Split::Bh;
        can.bv = parl != Split::Bv;
    }
    let max_bt_depth = u32::from(lim.max_mtt) + u32::from(node.implicit_bt_depth);
    let mut can_btt = u32::from(node.mtt_depth) < max_bt_depth;
    if can_btt
        && node.w <= lim.min_bt
        && node.h <= lim.min_bt
        && node.w <= lim.min_tt
        && node.h <= lim.min_tt
    {
        can_btt = false;
    }
    if can_btt
        && (node.w > lim.max_bt || node.h > lim.max_bt)
        && (node.w > lim.max_tt || node.h > lim.max_tt)
    {
        can_btt = false;
    }
    if !can_btt {
        can.bh = false;
        can.bv = false;
        can.th = false;
        can.tv = false;
        return;
    }
    if node.w > lim.max_bt || node.h > lim.max_bt {
        can.bh = false;
        can.bv = false;
    }
    if node.h <= lim.min_bt {
        can.bh = false;
    }
    if node.w > MAX_TB && node.h <= MAX_TB {
        can.bh = false;
    }
    if let Some((cw, ch)) = area_c {
        if cw * ch <= 16 {
            can.bh = false;
        }
    }
    if node.w <= lim.min_bt {
        can.bv = false;
    }
    if node.w <= MAX_TB && node.h > MAX_TB {
        can.bv = false;
    }
    if let Some((cw, ch)) = area_c {
        if cw * ch <= 16 || cw == 4 {
            can.bv = false;
        }
    }
    if node.h <= 2 * lim.min_tt || node.h > lim.max_tt || node.w > lim.max_tt {
        can.th = false;
    }
    if node.w > MAX_TB || node.h > MAX_TB {
        can.th = false;
    }
    if let Some((cw, ch)) = area_c {
        if cw * ch <= 32 {
            can.th = false;
        }
    }
    if node.w <= 2 * lim.min_tt || node.w > lim.max_tt || node.h > lim.max_tt {
        can.tv = false;
    }
    if node.w > MAX_TB || node.h > MAX_TB {
        can.tv = false;
    }
    if let Some((cw, ch)) = area_c {
        if cw * ch <= 32 || cw == 8 {
            can.tv = false;
        }
    }
}

struct Gen<'a> {
    sps: &'a VvcSps,
    sh: &'a VvcSliceHeader,
    ph: &'a VvcPictureHeader,
    aps: &'a VvcActiveApsBank,
    pic_w: usize,
    pic_h: usize,
    vis_w: usize,
    vis_h: usize,
    ctu: usize,
    /// Luma sample bit depth (`sps_bit_depth_minus8 + 8`).
    bit_depth: u8,
    lgrid: Vec<LumaCell>,
    lgrid_w: usize,
    cgrid: Vec<ChromaCell>,
    cgrid_w: usize,
    cgrid_h: usize,
    y: Plane16,
    u: Plane16,
    v: Plane16,
    qp_y: i32,
    qp_cb: i32,
    qp_cr: i32,
    /// `Qp′CbCr` (`H.266 §8.7.2`, used when `TuCResMode == 2`).
    qp_joint: i32,
    /// `NumRefIdxActive[0..1]` of the slice.
    num_ref_idx: [u8; 2],
    /// `slice->getBiDirPred()`: symmetric MVD is available for this picture.
    bidir_pred: bool,
    /// HMVP table of the current CTU row (`HmvpCandList`).
    hmvp: Vec<super::inter::InterMergeCand>,
    /// Prediction planes of the inter coding unit being reconstructed.
    pred_y: Plane16,
    pred_u: Plane16,
    pred_v: Plane16,
    /// Active reference pictures of the slice, in list order.
    refs_l0: Vec<super::inter::RefFrame>,
    refs_l1: Vec<super::inter::RefFrame>,
    /// `PicOrderCntVal` of the picture being decoded.
    poc: i32,
    /// Collocated picture for TMVP, with its per-4x4 MV grid and reference POCs.
    collocated: Option<super::inter::RefFrame>,
    /// `m_checkLdc`: every active reference POC is not in the future.
    check_ldc: bool,
    /// Symmetric MVD reference indices (`getSymRefIdx`, DecLib.cpp:3510-3580).
    sym_ref: [i8; 2],
    lmcs: Option<LmcsReshaper16>,
    lmcs_on: bool,
    sao: Vec<[filter::SaoType; 3]>,
    ctu_cols: usize,
    alf_luma: Vec<Option<AlfLumaMode>>,
    alf_cb: Vec<Option<usize>>,
    alf_cr: Vec<Option<usize>>,
    /// Per-CTU CCALF filter index (`0` = off) for Cb and Cr.
    ccalf_cb: Vec<u8>,
    ccalf_cr: Vec<u8>,
    /// Luma transform-block boundaries: per 4x4 cell id, u32::MAX = none.
    tu_y: Vec<u32>,
    /// Whether the luma TU covering each 4x4 cell has a coded residual.
    tu_cbf_y: Vec<bool>,
    tu_u: Vec<u32>,
    tu_v: Vec<u32>,
    tu_y_w: usize,
    tu_y_h: usize,
    tu_c_w: usize,
    tu_c_h: usize,
    tu_wy: Vec<u16>,
    tu_hy: Vec<u16>,
    tu_wc: Vec<u16>,
    tu_hc: Vec<u16>,
    dec_y: Vec<bool>,
    dec_c: Vec<bool>,
    next_tu_id: u32,
    vpdu_scale: std::collections::HashMap<(usize, usize), i32>,
}

#[derive(Clone, Copy)]
enum AlfLumaMode {
    Fixed(usize),
    Aps(usize),
}

impl<'a> Gen<'a> {
    /// Maximum legal sample value `(1 << BitDepth) - 1`.
    fn max_val(&self) -> i32 {
        (1i32 << u32::from(self.bit_depth.clamp(8, 12))) - 1
    }

    /// Unavailable-reference default sample `1 << (BitDepth - 1)`.
    fn dc(&self) -> i32 {
        1i32 << (u32::from(self.bit_depth.clamp(8, 12)) - 1)
    }

    fn lcell(&self, x: usize, y: usize) -> Option<LumaCell> {
        if x >= self.pic_w || y >= self.pic_h {
            return None;
        }
        let c = self.lgrid[(y / 4) * self.lgrid_w + x / 4];
        c.valid.then_some(c)
    }

    /// Motion state of the 4x4 unit at `(x, y)` (`cs.getPU(pos)`).
    fn motion(&self, x: usize, y: usize) -> Option<MotionInfo> {
        let c = self.lcell(x, y)?;
        if c.intra {
            return None;
        }
        Some(MotionInfo {
            valid: true,
            inter_dir: c.inter_dir,
            ref_idx: c.ref_idx,
            mv: c.mv,
            bcw: c.bcw,
            cu_w: c.cu_w,
            cu_h: c.cu_h,
            skip: c.skip,
            affine: c.affine,
        })
    }

    /// `pu.cu != neighbour.cu`.
    fn same_cu(&self, node: &Node, x: usize, y: usize) -> bool {
        self.lcell(x, y)
            .is_some_and(|c| usize::from(c.cu_x) == node.x && usize::from(c.cu_y) == node.y)
    }

    /// POC of the active reference `ref_idx` in `list`.
    fn ref_poc(&self, list: usize, ref_idx: i8) -> Option<i32> {
        let refs = if list == 0 {
            &self.refs_l0
        } else {
            &self.refs_l1
        };
        refs.get(usize::try_from(ref_idx).ok()?).map(|r| r.poc)
    }

    fn ccell(&self, x: usize, y: usize) -> Option<ChromaCell> {
        if x >= self.cgrid_w || y >= self.cgrid_h {
            return None;
        }
        let c = self.cgrid[y * self.cgrid_w + x];
        c.valid.then_some(c)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_lcu(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        qt: u8,
        mode: u8,
        mip: bool,
        split64: u8,
        isp: bool,
    ) {
        for yy in y..(y + h).min(self.pic_h) {
            for xx in x..(x + w).min(self.pic_w) {
                self.lgrid[(yy / 4) * self.lgrid_w + xx / 4] = LumaCell {
                    valid: true,
                    cu_w: w as u16,
                    cu_h: h as u16,
                    qt_depth: qt,
                    intra: true,
                    mode,
                    mip,
                    split64,
                    isp,
                    ..LumaCell::default()
                };
            }
        }
    }

    /// Record the inter syntax state of a coding unit on the luma grid.
    fn set_lcell_inter(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        qt: u8,
        skip: bool,
        affine: bool,
        inter_dir: u8,
        ref_idx: [i8; 2],
        mv: [super::inter::Mv; 2],
        bcw: u8,
    ) {
        for yy in y..(y + h).min(self.pic_h) {
            for xx in x..(x + w).min(self.pic_w) {
                self.lgrid[(yy / 4) * self.lgrid_w + xx / 4] = LumaCell {
                    valid: true,
                    cu_w: w as u16,
                    cu_h: h as u16,
                    qt_depth: qt,
                    intra: false,
                    skip,
                    affine,
                    cu_x: x as u16,
                    cu_y: y as u16,
                    inter_dir,
                    ref_idx,
                    mv,
                    bcw,
                    ..LumaCell::default()
                };
            }
        }
    }

    fn set_ccu(&mut self, x: usize, y: usize, w: usize, h: usize, qt: u8) {
        for yy in y..(y + h).min(self.cgrid_h) {
            for xx in x..(x + w).min(self.cgrid_w) {
                self.cgrid[yy * self.cgrid_w + xx] = ChromaCell {
                    valid: true,
                    cu_w: w as u16,
                    cu_h: h as u16,
                    qt_depth: qt,
                };
            }
        }
    }
}

/// `CclmEnabled` of `H.266 §8.4.4` / VTM `CodingUnit::checkCCLMAllowed`: whether
/// `cclm_mode_flag` may be present for the current chroma coding unit.
fn cclm_enabled_for_chroma(g: &Gen<'_>, node: &Node) -> bool {
    if !g.sps.cclm_enabled {
        return false;
    }
    if !g.sps.dual_tree_intra || g.sh.slice_type != super::hls::VvcSliceType::I {
        return true;
    }
    if g.ctu <= 32 {
        return true;
    }
    let depth = if g.ctu > 64 { 1u8 } else { 0u8 };
    let s1 = node.split_at(depth);
    let s2 = node.split_at(depth + 1);
    let chroma_ok = s1 == Split::Qt
        || (s1 == Split::Bh && s2 == Split::Bv)
        || s1 == Split::None
        || (s1 == Split::Bh && s2 == Split::None);
    if !chroma_ok {
        return false;
    }
    let Some(cell) = g.lcell(node.x, node.y) else {
        return false;
    };
    if cell.cu_w < 64 || cell.cu_h < 64 {
        return cell.split64 == Split::Qt as u8;
    }
    if cell.cu_w == 64 && cell.cu_h == 64 && cell.isp {
        return false;
    }
    true
}

fn split_contexts(g: &Gen<'_>, node: &Node, can: CanSplit) -> (usize, usize, usize, usize, usize) {
    let chroma = node.tree == TreeType::Chroma;
    let (_lw, lh, lqt, aw, _ah, aqt) = if chroma {
        let cx = node.x / 2;
        let cy = node.y / 2;
        let l = g.ccell(cx.saturating_sub(1), cy);
        let a = g.ccell(cx, cy.saturating_sub(1));
        (
            l.map_or(0, |c| usize::from(c.cu_w)),
            l.map_or(0, |c| usize::from(c.cu_h)),
            l.map_or(0, |c| c.qt_depth),
            a.map_or(0, |c| usize::from(c.cu_w)),
            a.map_or(0, |c| usize::from(c.cu_h)),
            a.map_or(0, |c| c.qt_depth),
        )
    } else {
        let l = g.lcell(node.x.saturating_sub(1), node.y);
        let a = g.lcell(node.x, node.y.saturating_sub(1));
        (
            l.map_or(0, |c| usize::from(c.cu_w)),
            l.map_or(0, |c| usize::from(c.cu_h)),
            l.map_or(0, |c| c.qt_depth),
            a.map_or(0, |c| usize::from(c.cu_w)),
            a.map_or(0, |c| usize::from(c.cu_h)),
            a.map_or(0, |c| c.qt_depth),
        )
    };
    let (w_cur, h_cur) = if chroma {
        (node.w / 2, node.h / 2)
    } else {
        (node.w, node.h)
    };
    let mut ctx_split = 0usize;
    if lh > 0 && lh < h_cur {
        ctx_split += 1;
    }
    if aw > 0 && aw < w_cur {
        ctx_split += 1;
    }
    let mut num_split = 0usize;
    if can.qt {
        num_split += 2;
    }
    for b in [can.bh, can.bv, can.th, can.tv] {
        if b {
            num_split += 1;
        }
    }
    if num_split > 0 {
        num_split = num_split.saturating_sub(1);
    }
    ctx_split += 3 * (num_split >> 1);

    let mut ctx_qt = 0usize;
    if lh > 0 && lqt > node.qt_depth {
        ctx_qt += 1;
    }
    if aw > 0 && aqt > node.qt_depth {
        ctx_qt += 1;
    }
    if node.qt_depth >= 2 {
        ctx_qt += 3;
    }

    let num_hor = usize::from(can.bh) + usize::from(can.th);
    let num_ver = usize::from(can.bv) + usize::from(can.tv);
    let ctx_hv = if num_ver == num_hor {
        if lh == 0 || aw == 0 {
            0
        } else {
            let dep_above = w_cur / aw.max(1);
            let dep_left = h_cur / lh.max(1);
            if dep_above == dep_left {
                0
            } else if dep_above < dep_left {
                1
            } else {
                2
            }
        }
    } else if num_ver < num_hor {
        3
    } else {
        4
    };
    (
        ctx_split.min(8),
        ctx_qt.min(5),
        ctx_hv.min(4),
        usize::from(node.mtt_depth <= 1),
        if node.mtt_depth <= 1 { 3 } else { 2 },
    )
}

fn decode_split(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    can: CanSplit,
    c: (usize, usize, usize, usize, usize),
) -> Split {
    let (cs, cq, ch, chb, cvb) = c;
    let mut is_split = can.any_split();
    if can.no && is_split {
        is_split = dec.decode_bin(&mut ctx.models[ctx::SPLIT_FLAG + cs]);
    }
    if !is_split {
        return Split::None;
    }
    let mut is_qt = can.qt;
    if is_qt && can.any_btt() {
        is_qt = dec.decode_bin(&mut ctx.models[ctx::SPLIT_QT_FLAG + cq]);
    }
    if is_qt {
        return Split::Qt;
    }
    let can_hor = can.bh || can.th;
    let mut is_ver = can.bv || can.tv;
    if is_ver && can_hor {
        is_ver = dec.decode_bin(&mut ctx.models[ctx::SPLIT_HV_FLAG + ch]);
    }
    let can14 = if is_ver { can.tv } else { can.th };
    let mut is12 = if is_ver { can.bv } else { can.bh };
    if is12 && can14 {
        let c = if is_ver { cvb } else { chb };
        is12 = dec.decode_bin(&mut ctx.models[ctx::SPLIT_12_FLAG + c]);
    }
    match (is_ver, is12) {
        (true, true) => Split::Bv,
        (true, false) => Split::Tv,
        (false, true) => Split::Bh,
        (false, false) => Split::Th,
    }
}

#[derive(Default, Clone, Copy)]
struct CuCtx {
    violates_lfnst: [bool; 2],
    lfnst_last_scan_pos: bool,
    violates_mts: bool,
    mts_last_scan_pos: bool,
}

#[derive(Clone)]
struct TuData {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    component: usize,
    levels: Vec<i32>,
    cbf: bool,
    ts: bool,
    joint: u8,
}

#[derive(Clone, Copy)]
struct CuPlan {
    tree: TreeType,
    x: usize,
    w: usize,
    h: usize,
    qp_y: i32,
    qp_cb: i32,
    qp_cr: i32,
    mode: u8,
    mip: Option<(bool, usize)>,
    isp: predict::IspMode,
    mrl: u8,
    bdpcm_l: u8,
    bdpcm_c: u8,
    chroma_mode: u8,
    /// Intra mode used by the LFNST set/transpose derivation for a chroma TU.
    lfnst_mode_c: u8,
    lfnst: u8,
    mts: u8,
    implicit_mts: bool,
    dep_quant: bool,
    /// Subblock-transform mode (`0` = none, else `sbt_idx * 2 + pos + 1`).
    sbt: u8,
    /// Whether the coding unit is intra (`CU::isIntra`).
    intra: bool,
}

pub(crate) struct GeneralPicture {
    pub y: Plane16,
    pub u: Plane16,
    pub v: Plane16,
    /// Reconstructed per-4x4 motion field for TMVP lookups by later pictures.
    pub mv_grid: super::inter::MotionFieldGrid,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_general_picture(
    cabac: &[u8],
    sps: &VvcSps,
    _pps: &VvcPps,
    sh: &VvcSliceHeader,
    ph: &VvcPictureHeader,
    aps: &VvcActiveApsBank,
    qp_y: i32,
    qp_cb: i32,
    qp_cr: i32,
    qp_joint: i32,
    num_ref_idx: [u8; 2],
    bidir_pred: bool,
    refs_l0: Vec<super::inter::RefFrame>,
    refs_l1: Vec<super::inter::RefFrame>,
    poc: i32,
) -> Result<GeneralPicture, VvcError> {
    if sps.info.chroma_format_idc != 1 || !matches!(sps.info.bit_depth_luma, 8 | 10) {
        return Err(VvcError::Unsupported(
            "general coding-tree decoding is implemented for 8/10-bit 4:2:0",
        ));
    }
    if sps.palette_enabled || sps.ibc_enabled || sps.act_enabled {
        return Err(VvcError::Unsupported(
            "general coding-tree decoding does not implement palette, IBC, or ACT",
        ));
    }
    let vis_w = sps.info.width as usize;
    let vis_h = sps.info.height as usize;
    let ctu = 1usize << sps.info.ctu_log2;
    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "general: vis={}x{} ctu={} dual={} min_cb={} qt_l={} bt_l={} tt_l={} mtt_l={} qt_c={} bt_c={} tt_c={} mtt_c={} cclm={} cclm_vert={} jccr={} mts={} explicit_mts={} lfnst={} isp={} ts={}",
            vis_w,
            vis_h,
            ctu,
            sps.dual_tree_intra,
            sps.min_cb_log2,
            sps.log2_diff_min_qt_min_cb_intra_luma,
            sps.log2_diff_max_bt_min_qt_intra_luma,
            sps.log2_diff_max_tt_min_qt_intra_luma,
            sps.max_mtt_depth_intra_luma,
            sps.log2_diff_min_qt_min_cb_intra_chroma,
            sps.log2_diff_max_bt_min_qt_intra_chroma,
            sps.log2_diff_max_tt_min_qt_intra_chroma,
            sps.max_mtt_depth_intra_chroma,
            sps.cclm_enabled,
            sps.chroma_vertical_collocated,
            sps.joint_cbcr_enabled,
            sps.mts_enabled,
            sps.explicit_mts_intra,
            sps.lfnst_enabled,
            sps.isp_enabled,
            sps.transform_skip_enabled
        );
    }
    let pic_w = vis_w.div_ceil(ctu) * ctu;
    let pic_h = vis_h.div_ceil(ctu) * ctu;
    let lgrid_w = pic_w.div_ceil(4);
    let lgrid_h = pic_h.div_ceil(4);
    let cgrid_w = pic_w.div_ceil(2);
    let cgrid_h = pic_h.div_ceil(2);
    let ctu_cols = pic_w.div_ceil(ctu);
    let ctu_rows = pic_h.div_ceil(ctu);
    let lmcs_on = sps.lmcs_enabled
        && ph.lmcs_enabled
        && sh.lmcs_used
        && std::env::var_os("VVC_NO_LMCS").is_none();
    let lmcs = if lmcs_on {
        aps.lmcs[usize::from(ph.lmcs_aps_id & 7)]
            .as_ref()
            .map(|a| LmcsReshaper16::from_aps(a, sps.info.bit_depth_luma))
    } else {
        None
    };
    let bit_depth = sps.info.bit_depth_luma;
    // `DecLib.cpp:3480-3505`: low-delay check makes the collocated MV use the
    // current list; otherwise it uses `ph_collocated_from_l0_flag`.
    if std::env::var_os("VVC_MV_TRACE").is_some() {
        eprintln!(
            "sps log2_parallel_merge_level_minus2={} poc={}",
            sps.log2_parallel_merge_level_minus2, poc
        );
    }
    let check_ldc = refs_l0.iter().chain(refs_l1.iter()).all(|r| r.poc <= poc);
    let col_list = if sh.slice_type == super::hls::VvcSliceType::B && !ph.collocated_from_l0 {
        1
    } else {
        0
    };
    let collocated = (if col_list == 0 { &refs_l0 } else { &refs_l1 })
        .get(usize::try_from(ph.collocated_ref_idx).unwrap_or(usize::MAX))
        .cloned();
    // `DecLib.cpp:3510-3580`: nearest forward POC in L0 and nearest backward
    // POC in L1 (with the mirrored fallback) are the symmetric references.
    let mut sym_ref = [-1i8; 2];
    let nearest = |refs: &[super::inter::RefFrame], forward: bool| -> Option<(i32, i8)> {
        let mut best: Option<(i32, i8)> = None;
        for (i, r) in refs.iter().enumerate() {
            let side = if forward { r.poc < poc } else { r.poc > poc };
            if side && best.is_none_or(|(p, _)| if forward { r.poc > p } else { r.poc < p }) {
                best = Some((r.poc, i as i8));
            }
        }
        best
    };
    let f0 = nearest(&refs_l0, true);
    let b1 = nearest(&refs_l1, false);
    if let (Some(f0), Some(b1)) = (f0, b1) {
        sym_ref = [f0.1, b1.1];
    } else {
        let b0 = nearest(&refs_l0, false);
        let f1 = nearest(&refs_l1, true);
        if let (Some(b0), Some(f1)) = (b0, f1) {
            sym_ref = [f1.1, b0.1];
        }
    }
    if std::env::var_os("VVC_MV_TRACE").is_some() {
        eprintln!(
            "colinfo poc={} col_list={} col_ref={} col_from_l0={} check_ldc={} colloc={:?}",
            poc,
            col_list,
            ph.collocated_ref_idx,
            ph.collocated_from_l0,
            check_ldc,
            collocated.as_ref().map(|r| r.poc)
        );
    }
    let mut g = Gen {
        sps,
        sh,
        ph,
        aps,
        pic_w,
        pic_h,
        vis_w,
        vis_h,
        ctu,
        bit_depth,
        lgrid: vec![LumaCell::default(); lgrid_w * lgrid_h],
        lgrid_w,
        cgrid: vec![ChromaCell::default(); cgrid_w * cgrid_h],
        cgrid_w,
        cgrid_h,
        y: Plane16::new(pic_w, pic_h, bit_depth),
        u: Plane16::new(cgrid_w, cgrid_h, bit_depth),
        v: Plane16::new(cgrid_w, cgrid_h, bit_depth),
        qp_y,
        qp_cb,
        qp_cr,
        qp_joint,
        num_ref_idx,
        bidir_pred,
        hmvp: Vec::new(),
        pred_y: Plane16::new(pic_w, pic_h, bit_depth),
        pred_u: Plane16::new(cgrid_w, cgrid_h, bit_depth),
        pred_v: Plane16::new(cgrid_w, cgrid_h, bit_depth),
        refs_l0,
        refs_l1,
        poc,
        collocated,
        check_ldc,
        sym_ref,
        lmcs,
        lmcs_on,
        sao: vec![[filter::SaoType::Off; 3]; ctu_cols * ctu_rows],
        ctu_cols,
        alf_luma: vec![None; ctu_cols * ctu_rows],
        alf_cb: vec![None; ctu_cols * ctu_rows],
        alf_cr: vec![None; ctu_cols * ctu_rows],
        ccalf_cb: vec![0; ctu_cols * ctu_rows],
        ccalf_cr: vec![0; ctu_cols * ctu_rows],
        tu_y_w: lgrid_w,
        tu_y_h: lgrid_h,
        tu_c_w: cgrid_w.div_ceil(2),
        tu_c_h: cgrid_h.div_ceil(2),
        tu_y: vec![u32::MAX; lgrid_w * lgrid_h],
        tu_cbf_y: vec![false; lgrid_w * lgrid_h],
        tu_u: vec![u32::MAX; cgrid_w.div_ceil(2) * cgrid_h.div_ceil(2)],
        tu_v: vec![u32::MAX; cgrid_w.div_ceil(2) * cgrid_h.div_ceil(2)],
        tu_wy: vec![0; lgrid_w * lgrid_h],
        tu_hy: vec![0; lgrid_w * lgrid_h],
        tu_wc: vec![0; cgrid_w.div_ceil(2) * cgrid_h.div_ceil(2)],
        tu_hc: vec![0; cgrid_w.div_ceil(2) * cgrid_h.div_ceil(2)],
        dec_y: vec![false; lgrid_w * lgrid_h],
        dec_c: vec![false; cgrid_w * cgrid_h],
        next_tu_id: 0,
        vpdu_scale: std::collections::HashMap::new(),
    };

    let mut dec = BinDecoder::new(cabac).map_err(|e| VvcError::Bitstream(e.to_string()))?;
    // `initCtxModels()` selects the P/B table from the slice type and swaps it
    // when `sh_cabac_init_flag` is set (`H.266 §9.3.2.2`, VTM
    // `CABACReader.cpp:75-88`); using the I-slice table for an inter picture
    // desynchronises at the first context-coded bin.
    let mut ctx = CtxStore::new_for_slice(sh.slice_type, sh.cabac_init_flag, qp_y);
    let dual = sps.dual_tree_intra && sh.slice_type == super::hls::VvcSliceType::I;

    for ctu_y in (0..pic_h).step_by(ctu) {
        for ctu_x in (0..pic_w).step_by(ctu) {
            let ctu_idx = (ctu_y / ctu) * ctu_cols + ctu_x / ctu;
            if ctu_x == 0 {
                g.hmvp.clear();
            }
            decode_sao_ctu(&mut g, &mut dec, &mut ctx, ctu_x, ctu_y, ctu_idx);
            decode_alf_ctu(&mut g, &mut dec, &mut ctx, ctu_x, ctu_y, ctu_idx);

            let base = Node {
                x: ctu_x,
                y: ctu_y,
                w: ctu,
                h: ctu,
                qt_depth: 0,
                mtt_depth: 0,
                bt_depth: 0,
                part_idx: 0,
                last_split: Split::Ctu,
                implicit_bt_depth: 0,
                checkd_implicit: false,
                implicit_split: Split::None,
                tree: TreeType::Joint,
                mode_type: ModeType::All,
                series: [0; 8],
                series_len: 0,
            };
            if dual {
                if ctu > 64 {
                    let lim = limits(&g, false);
                    let mut can = CanSplit::default();
                    let root = Node {
                        tree: TreeType::Luma,
                        ..base
                    };
                    can_split(&g, &root, &lim, true, &mut can);
                    let c = split_contexts(&g, &root, can);
                    let split = decode_split(&mut dec, &mut ctx, can, c);
                    if split != Split::Qt {
                        return Err(VvcError::Unsupported(
                            "dual-tree intra CTU larger than 64 without a QT split",
                        ));
                    }
                    for (ox, oy) in [(0, 0), (ctu / 2, 0), (0, ctu / 2), (ctu / 2, ctu / 2)] {
                        let x = ctu_x + ox;
                        let y = ctu_y + oy;
                        if x < vis_w && y < vis_h {
                            let region = Node {
                                x,
                                y,
                                w: ctu / 2,
                                h: ctu / 2,
                                qt_depth: 1,
                                mtt_depth: 0,
                                bt_depth: 0,
                                part_idx: 0,
                                last_split: Split::Qt,
                                implicit_bt_depth: 0,
                                checkd_implicit: false,
                                implicit_split: Split::None,
                                tree: TreeType::Luma,
                                mode_type: ModeType::Intra,
                                series: {
                                    let mut series = [0u8; 8];
                                    series[0] = Split::Qt as u8;
                                    series
                                },
                                series_len: 1,
                            };
                            coding_tree(&mut g, &mut dec, &mut ctx, region)?;
                            coding_tree(
                                &mut g,
                                &mut dec,
                                &mut ctx,
                                Node {
                                    tree: TreeType::Chroma,
                                    ..region
                                },
                            )?;
                        }
                    }
                } else {
                    coding_tree(
                        &mut g,
                        &mut dec,
                        &mut ctx,
                        Node {
                            tree: TreeType::Luma,
                            ..base
                        },
                    )?;
                    coding_tree(
                        &mut g,
                        &mut dec,
                        &mut ctx,
                        Node {
                            tree: TreeType::Chroma,
                            ..base
                        },
                    )?;
                }
            } else {
                let joint = Node {
                    tree: TreeType::Joint,
                    ..base
                };
                coding_tree(&mut g, &mut dec, &mut ctx, joint)?;
            }
        }
    }

    dec.end_of_slice()
        .map_err(|e| VvcError::Bitstream(e.to_string()))?;

    if g.lmcs_on {
        if let Some(r) = &g.lmcs {
            r.apply_inverse_plane_16(&mut g.y);
        }
    }
    let no_db = std::env::var_os("VVC_NO_DB").is_some();
    let no_sao = std::env::var_os("VVC_NO_SAO").is_some();
    let no_alf = std::env::var_os("VVC_NO_ALF").is_some();
    if !sh.deblocking_disabled && !no_db {
        apply_deblock(&mut g);
    }
    if g.sps.sao_enabled && (sh.sao_luma_used || sh.sao_chroma_used) && !no_sao {
        apply_sao(&mut g);
    }
    if g.sps.alf_enabled && sh.alf_enabled && !no_alf {
        apply_alf(&mut g);
    }
    let mut mv_grid = super::inter::MotionFieldGrid::new(pic_w, pic_h);
    for gy in 0..g.tu_y_h {
        for gx in 0..g.lgrid_w {
            let c = g.lgrid[gy * g.lgrid_w + gx];
            if !c.valid || c.intra {
                continue;
            }
            mv_grid.fill_region(
                gx * 4,
                gy * 4,
                4,
                4,
                Some(super::inter::InterMergeCand {
                    inter_dir: c.inter_dir,
                    mv_field: [
                        super::inter::MvField::new(c.mv[0], c.ref_idx[0]),
                        super::inter::MvField::new(c.mv[1], c.ref_idx[1]),
                    ],
                    bcw_idx: 2,
                    use_alt_hpel_if: false,
                }),
            );
        }
    }
    Ok(GeneralPicture {
        y: g.y,
        u: g.u,
        v: g.v,
        mv_grid,
    })
}

fn coding_tree(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    let lim = limits(g, node.tree == TreeType::Chroma);
    // The dual-tree rules only apply to intra pictures; an inter picture always
    // uses a single tree even when `sps_qtbtt_dual_tree_intra_flag` is set.
    let dual = g.sps.dual_tree_intra && g.sh.slice_type == super::hls::VvcSliceType::I;
    let mut can = CanSplit::default();
    can_split(g, &node, &lim, dual, &mut can);
    let contexts = split_contexts(g, &node, can);
    let trace = std::env::var_os("VVC_TRACE").is_some();
    let split = decode_split(dec, ctx, can, contexts);
    if trace {
        let any = can.any_split();
        eprintln!(
            "split_cu_mode() ctx={} split={} qt={} ctxHv={} ctx12={} mode={} tree={:?} pos=({},{}) size={}x{}",
            contexts.0,
            u8::from(any && split != Split::None),
            u8::from(split == Split::Qt),
            contexts.2,
            if split == Split::Th || split == Split::Bh {
                contexts.3
            } else {
                contexts.4
            },
            split as u8,
            node.tree,
            node.x,
            node.y,
            node.w,
            node.h
        );
    }
    if split == Split::None {
        return coding_unit(g, dec, ctx, node);
    }
    let is_implicit = implicit_split(&node, &lim, g.pic_w, g.pic_h, dual) != Split::None;
    let child_mode = signal_mode_constraint(g, dec, ctx, &node, split)?;
    let mut next = node;
    next.mode_type = child_mode;
    next.last_split = split;
    next.checkd_implicit = false;
    next.implicit_split = Split::None;
    next.push_split(split);
    match split {
        Split::Qt => {
            next.qt_depth += 1;
            next.mtt_depth = 0;
            next.bt_depth = 0;
            next.w /= 2;
            next.h /= 2;
            next.part_idx = 0;
            for (ox, oy) in [(0, 0), (1usize, 0), (0, 1), (1, 1)] {
                let child = Node {
                    x: node.x + ox * next.w,
                    y: node.y + oy * next.h,
                    ..next
                };
                if child.x < g.vis_w && child.y < g.vis_h {
                    coding_tree(g, dec, ctx, child)?;
                }
            }
        }
        Split::Bh | Split::Bv => {
            next.mtt_depth += 1;
            next.bt_depth += 1;
            if is_implicit {
                next.implicit_bt_depth += 1;
            }
            let parts: [(usize, usize, usize, usize); 2] = if split == Split::Bh {
                [
                    (0, 0, node.w, node.h / 2),
                    (0, node.h / 2, node.w, node.h / 2),
                ]
            } else {
                [
                    (0, 0, node.w / 2, node.h),
                    (node.w / 2, 0, node.w / 2, node.h),
                ]
            };
            for (i, (ox, oy, w, h)) in parts.into_iter().enumerate() {
                let child = Node {
                    x: node.x + ox,
                    y: node.y + oy,
                    w,
                    h,
                    part_idx: i as u8,
                    ..next
                };
                if child.x < g.vis_w && child.y < g.vis_h {
                    coding_tree(g, dec, ctx, child)?;
                }
            }
        }
        Split::Th | Split::Tv => {
            next.mtt_depth += 1;
            next.bt_depth += 2;
            let parts: [(usize, usize, usize, usize); 3] = if split == Split::Th {
                [
                    (0, 0, node.w, node.h / 4),
                    (0, node.h / 4, node.w, node.h / 2),
                    (0, 3 * node.h / 4, node.w, node.h / 4),
                ]
            } else {
                [
                    (0, 0, node.w / 4, node.h),
                    (node.w / 4, 0, node.w / 2, node.h),
                    (3 * node.w / 4, 0, node.w / 4, node.h),
                ]
            };
            for (i, (ox, oy, w, h)) in parts.into_iter().enumerate() {
                let child = Node {
                    x: node.x + ox,
                    y: node.y + oy,
                    w,
                    h,
                    part_idx: i as u8,
                    ..next
                };
                if child.x < g.vis_w && child.y < g.vis_h {
                    coding_tree(g, dec, ctx, child)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// `mode_constraint()` (`H.266 §7.4.12.4`, VTM `CodingStructure::signalModeCons`
/// and `CABACReader::mode_constraint`, `CodingStructure.cpp:1046-1072`,
/// `CABACReader.cpp:757-777`).
///
/// In an inter slice a split can switch the subtree to intra coding when the
/// chroma block would be too small: `LDT_MODE_TYPE_SIGNAL` carries one
/// `mode_cons_flag` bin, `LDT_MODE_TYPE_INFER` forces intra and
/// `LDT_MODE_TYPE_INHERIT` keeps the parent mode type.
fn signal_mode_constraint(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    split: Split,
) -> Result<ModeType, VvcError> {
    if g.sh.slice_type == super::hls::VvcSliceType::I || node.mode_type != ModeType::All {
        return Ok(node.mode_type);
    }
    let mut min_luma_area = node.w * node.h;
    match split {
        Split::Qt | Split::Th | Split::Tv => min_luma_area >>= 2,
        Split::Bh | Split::Bv => min_luma_area >>= 1,
        _ => {}
    }
    // 4:2:0: `getChannelTypeScaleX + getChannelTypeScaleY == 2`.
    let min_chroma_block = min_luma_area >> 2;
    let chroma_w = node.w / 2;
    let is_2xn = (chroma_w == 4 && split == Split::Bv) || (chroma_w == 8 && split == Split::Tv);
    if min_chroma_block >= 16 && !is_2xn {
        return Ok(node.mode_type);
    }
    if min_luma_area < 32 {
        return Ok(ModeType::Intra);
    }
    let left_intra = g
        .lcell(node.x.wrapping_sub(1), node.y)
        .is_some_and(|c| c.intra);
    let above_intra = g
        .lcell(node.x, node.y.wrapping_sub(1))
        .is_some_and(|c| c.intra);
    let flag = dec
        .decode_bin(&mut ctx.models[ctx::MODE_CONS_FLAG + usize::from(left_intra || above_intra)]);
    Ok(if flag {
        ModeType::Intra
    } else {
        ModeType::Inter
    })
}

fn decode_sao_ctu(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    x: usize,
    y: usize,
    idx: usize,
) {
    if !g.sps.sao_enabled {
        return;
    }
    let gx = x / g.ctu;
    let gy = y / g.ctu;
    let left = if gx > 0 { Some(g.sao[idx - 1]) } else { None };
    let up = if gy > 0 {
        Some(g.sao[idx - g.ctu_cols])
    } else {
        None
    };
    let params = super::slice::decode_ctu_sao(
        dec,
        ctx,
        gx,
        gy,
        g.sh.sao_luma_used,
        g.sh.sao_chroma_used,
        left,
        up,
        g.bit_depth,
    );
    g.sao[idx] = params;
}

fn decode_alf_ctu(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    x: usize,
    y: usize,
    idx: usize,
) {
    if !g.sps.alf_enabled || !g.sh.alf_enabled {
        return;
    }
    let gx = x / g.ctu;
    let gy = y / g.ctu;
    let left_on = gx > 0
        && (g.alf_luma[idx - 1].is_some()
            || g.alf_cb[idx - 1].is_some()
            || g.alf_cr[idx - 1].is_some());
    let up_on = gy > 0
        && (g.alf_luma[idx - g.ctu_cols].is_some()
            || g.alf_cb[idx - g.ctu_cols].is_some()
            || g.alf_cr[idx - g.ctu_cols].is_some());
    let ctx_off = usize::from(left_on) + usize::from(up_on);
    let num_aps = g.sh.alf_aps_ids_luma.len();
    let luma_on = dec.decode_bin(&mut ctx.models[ctx::ALF_CTB_FLAG + ctx_off]);
    if luma_on {
        let use_aps = num_aps > 0 && dec.decode_bin(&mut ctx.models[ctx::ALF_USE_APS_FLAG]);
        if use_aps {
            let f = if num_aps > 1 {
                dec.decode_trunc_bin(num_aps as u32) as usize
            } else {
                0
            };
            if std::env::var_os("VVC_TRACE").is_some() {
                eprintln!("alf luma idx={} aps={}", idx, f);
            }
            g.alf_luma[idx] = Some(AlfLumaMode::Aps(f));
        } else {
            let f = dec.decode_trunc_bin(16) as usize;
            if std::env::var_os("VVC_TRACE").is_some() {
                eprintln!("alf luma idx={} fixed={}", idx, f);
            }
            g.alf_luma[idx] = Some(AlfLumaMode::Fixed(f));
        }
    }
    if g.sh.alf_cb_enabled || g.sh.alf_cr_enabled {
        let aps_idx = usize::from(g.sh.alf_aps_id_chroma & 7);
        let num_alts = g.aps.alf[aps_idx]
            .as_ref()
            .map_or(0, |a| a.chroma_filters.len());
        for comp in 1..3usize {
            let on = dec.decode_bin(&mut ctx.models[ctx::ALF_CTB_FLAG + comp * 3 + ctx_off]);
            if !on {
                continue;
            }
            let mut alt = 0usize;
            while alt + 1 < num_alts
                && dec.decode_bin(&mut ctx.models[ctx::CTB_ALF_ALTERNATIVE + comp - 1])
            {
                alt += 1;
            }
            if comp == 1 {
                g.alf_cb[idx] = Some(alt);
            } else {
                g.alf_cr[idx] = Some(alt);
            }
        }
    }
    if g.sps.ccalf_enabled && g.sh.alf_enabled {
        for comp in 1..3usize {
            let aps_id = if comp == 1 {
                g.sh.cc_alf_cb_aps_id
            } else {
                g.sh.cc_alf_cr_aps_id
            };
            let Some(aps_id) = aps_id else { continue };
            let count = g.aps.alf[usize::from(aps_id & 7)].as_ref().map_or(0, |a| {
                if comp == 1 {
                    a.cc_cb_filters.len()
                } else {
                    a.cc_cr_filters.len()
                }
            });
            if count == 0 {
                continue;
            }
            // VTM `CABACReader.cpp:296-330` (`ccAlfFilterControlIdc`).
            let left_on = if gx > 0 {
                let v = if comp == 1 {
                    g.ccalf_cb[idx - 1]
                } else {
                    g.ccalf_cr[idx - 1]
                };
                v != 0
            } else {
                false
            };
            let up_on = if gy > 0 {
                let v = if comp == 1 {
                    g.ccalf_cb[idx - g.ctu_cols]
                } else {
                    g.ccalf_cr[idx - g.ctu_cols]
                };
                v != 0
            } else {
                false
            };
            let ctxt = usize::from(left_on) + usize::from(up_on) + if comp == 2 { 3 } else { 0 };
            let mut idc =
                u8::from(dec.decode_bin(&mut ctx.models[ctx::CC_ALF_FILTER_CONTROL_FLAG + ctxt]));
            if idc != 0 {
                while usize::from(idc) != count && dec.decode_bin_ep() {
                    idc += 1;
                }
            }
            if comp == 1 {
                g.ccalf_cb[idx] = idc;
            } else {
                g.ccalf_cr[idx] = idc;
            }
        }
    }
}

fn coding_unit(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    match node.tree {
        TreeType::Luma | TreeType::Joint => decode_cu_luma(g, dec, ctx, node),
        TreeType::Chroma => decode_cu_chroma(g, dec, ctx, node),
    }
}

fn bit_is_on(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore, model: usize) -> bool {
    dec.decode_bin(&mut ctx.models[model])
}

// ============================================================================
// Inter coding units (`H.266 §7.3.8.7`, VTM `CABACReader::coding_unit`,
// `prediction_unit`, `merge_data`, `cu_residual`)
// ============================================================================

/// Inter syntax state of one coding unit (VTM `CodingUnit` + `PredictionUnit`
/// fields that are signalled or derived while parsing).
#[derive(Clone, Copy)]
struct InterCu {
    skip: bool,
    merge: bool,
    /// `merge_subblock_flag` (called `affine` in VTM).
    affine: bool,
    /// `cu_affine_type_flag` (6-parameter affine model).
    affine_6param: bool,
    mmvd: bool,
    ciip: bool,
    geo: bool,
    /// `geoSplitDir` for a geometric-partitioning merge.
    geo_split_dir: u8,
    /// `geoMergeIdx[0..1]` for a geometric-partitioning merge.
    geo_cand: [u8; 2],
    merge_idx: u8,
    /// `inter_pred_idc`: `1` = L0, `2` = L1, `3` = BI.
    inter_dir: u8,
    ref_idx: [i8; 2],
    /// `mvp_flag` per list.
    mvp_idx: [u8; 2],
    mvd: [super::inter::Mv; 2],
    /// `sym_mvd_flag` + 1 (`0` = absent).
    smvd: u8,
    /// `amvr_flag`/`amvr_precision_idx` result (`0` = 1/4 pel).
    imv: u8,
    /// `bcw_idx` (`BCW_DEFAULT == 2`).
    bcw: u8,
}

impl Default for InterCu {
    fn default() -> Self {
        Self {
            skip: false,
            merge: false,
            affine: false,
            affine_6param: false,
            mmvd: false,
            ciip: false,
            geo: false,
            geo_split_dir: 0,
            geo_cand: [0, 0],
            merge_idx: 0,
            inter_dir: 0,
            ref_idx: [-1, -1],
            mvp_idx: [0, 0],
            mvd: [super::inter::Mv::default(); 2],
            smvd: 0,
            imv: 0,
            bcw: 2,
        }
    }
}

/// `DeriveCtx::CtxSkipFlag` (`ContextModelling.cpp:328-342`).
fn skip_ctx(g: &Gen<'_>, node: &Node) -> usize {
    let left = g
        .lcell(node.x.wrapping_sub(1), node.y)
        .is_some_and(|c| c.skip);
    let above = g
        .lcell(node.x, node.y.wrapping_sub(1))
        .is_some_and(|c| c.skip);
    usize::from(left) + usize::from(above)
}

/// `DeriveCtx::CtxPredModeFlag` (`ContextModelling.cpp:344-352`).
fn pred_mode_ctx(g: &Gen<'_>, node: &Node) -> usize {
    let left = g
        .lcell(node.x.wrapping_sub(1), node.y)
        .is_some_and(|c| c.intra);
    let above = g
        .lcell(node.x, node.y.wrapping_sub(1))
        .is_some_and(|c| c.intra);
    usize::from(left || above)
}

/// `DeriveCtx::CtxAffineFlag` (`ContextModelling.cpp:314-326`).
fn affine_ctx(g: &Gen<'_>, node: &Node) -> usize {
    let left = g
        .lcell(node.x.wrapping_sub(1), node.y)
        .is_some_and(|c| c.affine);
    let above = g
        .lcell(node.x, node.y.wrapping_sub(1))
        .is_some_and(|c| c.affine);
    usize::from(left) + usize::from(above)
}

/// `DeriveCtx::CtxInterDir` (`ContextModelling.cpp:302-305`).
fn inter_dir_ctx(w: usize, h: usize) -> usize {
    7 - ((w.trailing_zeros() as usize + h.trailing_zeros() as usize + 1) >> 1)
}

/// Unary index with a coded first bin (`merge_idx`, `merge_subblock_idx`).
fn unary_index(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore, base: usize, num: usize) -> u8 {
    let num_minus1 = num.saturating_sub(1);
    let mut idx = 0u8;
    if num_minus1 > 0 && dec.decode_bin(&mut ctx.models[base]) {
        idx = 1;
        while usize::from(idx) < num_minus1 && dec.decode_bin_ep() {
            idx += 1;
        }
    }
    idx
}

/// `unary_max_eqprob()` (`H.266 §9.3.3.9`, VTM `CABACReader.cpp:3612-3622`).
fn unary_max_eqprob(dec: &mut BinDecoder<'_>, max_symbol: usize) -> u8 {
    let mut symbol = 0u8;
    while usize::from(symbol) < max_symbol && dec.decode_bin_ep() {
        symbol += 1;
    }
    symbol
}

/// `mmvd_merge_idx()` (`H.266 §7.3.8.9`, VTM `CABACReader.cpp:2420-2451`).
fn mmvd_merge_idx(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    max_merge_cand: u8,
) -> (u8, u8, u8) {
    let base_idx = if max_merge_cand > 1 {
        u8::from(dec.decode_bin(&mut ctx.models[ctx::MMVD_MERGE_IDX]))
    } else {
        0
    };
    let num_step_minus1 = 7u8; // `MmvdIdx::REFINE_STEP - 1`
    let mut step = 0u8;
    if dec.decode_bin(&mut ctx.models[ctx::MMVD_STEP_MVP_IDX]) {
        step = 1;
        while step < num_step_minus1 && dec.decode_bin_ep() {
            step += 1;
        }
    }
    let position = dec.decode_bin_ep() as u8 * 2 + dec.decode_bin_ep() as u8;
    (base_idx, step, position)
}

/// `imv_mode()` / `affine_amvr_mode()` (`H.266 §7.3.8.7`, VTM
/// `CABACReader.cpp:1017-1100`).
fn amvr_mode(
    g: &Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    affine: bool,
    nonzero_mvd: bool,
    nonzero_affine_mvd: bool,
) -> u8 {
    if !g.sps.amvr_enabled {
        return 0;
    }
    if affine {
        if !g.sps.affine_amvr_enabled || !nonzero_affine_mvd {
            return 0;
        }
        let mut value = u8::from(dec.decode_bin(&mut ctx.models[ctx::IMV_FLAG + 2]));
        if value != 0 {
            value = u8::from(dec.decode_bin(&mut ctx.models[ctx::IMV_FLAG + 3]));
            value += 1;
        }
        return value;
    }
    if !nonzero_mvd {
        return 0;
    }
    let value = u8::from(dec.decode_bin(&mut ctx.models[ctx::IMV_FLAG]));
    if value == 0 {
        return 0;
    }
    let hpel = u8::from(dec.decode_bin(&mut ctx.models[ctx::IMV_FLAG + 4]));
    if hpel == 0 {
        return 3; // `IMV_HPEL`
    }
    let four_pel = u8::from(dec.decode_bin(&mut ctx.models[ctx::IMV_FLAG + 1]));
    if four_pel == 0 { 1 } else { 2 } // `IMV_FPEL`, `IMV_4PEL`
}

/// `cu_bcw_flag()` (`H.266 §7.3.8.7`, VTM `CABACReader.cpp:1275-1312`).
fn bcw_flag(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> u8 {
    // `g_BcwParsingOrder` = {2, 3, 1, 4, 0} with `BCW_DEFAULT == 2`.
    const PARSING_ORDER: [u8; 5] = [2, 3, 1, 4, 0];
    let mut idx = 0usize;
    if dec.decode_bin(&mut ctx.models[ctx::BCW_IDX]) {
        idx = 1;
        let num_bcw = 3usize; // `slice->getCheckLDC()` is always 0 in the decoder
        while idx + 1 < num_bcw && dec.decode_bin_ep() {
            idx += 1;
        }
    }
    PARSING_ORDER[idx]
}

/// Per-4x4 motion state (`MotionInfo` in VTM).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct MotionInfo {
    valid: bool,
    inter_dir: u8,
    ref_idx: [i8; 2],
    mv: [super::inter::Mv; 2],
    bcw: u8,
    skip: bool,
    affine: bool,
    cu_w: u16,
    cu_h: u16,
}

impl MotionInfo {
    fn field(&self, list: usize) -> super::inter::MvField {
        super::inter::MvField::new(self.mv[list], self.ref_idx[list])
    }
    fn has(&self, list: usize) -> bool {
        self.valid && (self.inter_dir & (1 << list)) != 0 && self.ref_idx[list] >= 0
    }
    fn same_motion(&self, o: &MotionInfo) -> bool {
        self.inter_dir == o.inter_dir && self.ref_idx == o.ref_idx && self.mv == o.mv
    }
}

/// `PU::isDiffMER()` (`UnitTools.cpp`).
fn is_diff_mer(g: &Gen<'_>, x0: usize, y0: usize, x1: usize, y1: usize) -> bool {
    let p = usize::from(g.sps.log2_parallel_merge_level_minus2) + 2;
    (x0 >> p) != (x1 >> p) || (y0 >> p) != (y1 >> p)
}

/// Right shift from a coded AMVR precision to the internal 1/16-pel precision.
///
/// `Mv::m_amvrPrecision = { QUARTER, ONE, FOUR, HALF }` and
/// `MvPrecision::INTERNAL == ONE + 4` (VTM `Mv.cpp:43-44`, `Mv.h:50-58`).
fn amvr_shift(imv: u8) -> u32 {
    match imv {
        0 => 2, // 1/4 pel
        1 => 4, // 1 pel
        2 => 6, // 4 pel
        3 => 3, // 1/2 pel
        _ => 2,
    }
}

/// `Mv::changeTransPrecAmvr2Internal`: coded MVD (1/4, 1, 4 or 1/2 pel) to
/// internal 1/16-pel units.
fn mvd_to_internal(mv: super::inter::Mv, imv: u8) -> super::inter::Mv {
    let s = amvr_shift(imv);
    super::inter::Mv::new(mv.hor << s, mv.ver << s)
}

/// `Mv::roundTransPrecInternal2Amvr`: truncate an internal MV to the AMVR grid.
fn round_to_amvr(mv: super::inter::Mv, imv: u8) -> super::inter::Mv {
    let s = amvr_shift(imv);
    super::inter::Mv::new((mv.hor >> s) << s, (mv.ver >> s) << s)
}

/// `xGetDistScaleFactor` (`UnitTools.cpp:1703-1717`, H.266 §8.5.2.3).
fn dist_scale_factor(curr_poc: i32, curr_ref: i32, col_poc: i32, col_ref: i32) -> i32 {
    let td = col_poc - col_ref;
    let tb = curr_poc - curr_ref;
    if td == tb {
        return 4096;
    }
    let tdb = tb.clamp(-128, 127);
    let tdd = td.clamp(-128, 127);
    let x = (0x4000 + (tdd / 2).abs()) / tdd;
    ((tdb * x + 32) >> 6).clamp(-4096, 4095)
}

/// `Mv::getScaledMv`: multiply by the fixed-point factor and shift right 8.
fn scale_mv_fixed(mv: super::inter::Mv, scale: i32) -> super::inter::Mv {
    let clip = |v: i64| -> i32 {
        ((v * i64::from(scale)) >> 8).clamp(
            i64::from(super::inter::MV_MIN_STORAGE),
            i64::from(super::inter::MV_MAX_STORAGE),
        ) as i32
    };
    super::inter::Mv::new(clip(i64::from(mv.hor)), clip(i64::from(mv.ver)))
}

/// `Mv::convertMvFixedToFloat` / `convertMvFloatToFixed` pair used by
/// `roundMvComp` (`UnitTools.cpp:1739-1754`).
fn round_mv_comp(x: i32) -> i32 {
    let sign = x >> 31;
    let scale = (31 - ((x ^ sign) | 31).leading_zeros() as i32) - 5;
    if scale < 0 {
        return x;
    }
    let n = (x + ((1i32 << scale) >> 1)) >> scale;
    let exponent = scale + ((n ^ sign) >> 5);
    let mantissa = (n & 31) + sign * 32;
    let encoded = exponent | (mantissa * 16);
    let exp = encoded & 15;
    let mant = encoded >> 4;
    if exp == 0 {
        mant
    } else {
        (mant ^ 32) * (1 << (exp - 1))
    }
}

/// `PU::getColocatedMVP` (`UnitTools.cpp:1808-1928`, H.266 §8.5.2.11).
fn tmvp_candidate(
    g: &Gen<'_>,
    node: &Node,
    list_idx: usize,
    ref_idx: i8,
) -> Option<super::inter::InterMergeCand> {
    if !g.ph.temporal_mvp_enabled || node.w + node.h <= 12 {
        return None;
    }
    let col = g.collocated.as_ref()?;
    let (x, y, w, h) = (node.x, node.y, node.w, node.h);
    let rb = (x + w - 4, y + h - 4);
    let ctu = g.ctu;
    let c0 = rb.0 + 4 < g.pic_w && rb.1 + 4 < g.pic_h && (rb.1 & (ctu - 1)) + 4 < ctu;
    let positions = if c0 {
        [(rb.0 + 4, rb.1 + 4), (x + w / 2, y + h / 2)]
    } else {
        [(x + w / 2, y + h / 2), (x + w / 2, y + h / 2)]
    };
    let mut out = None;
    for (px, py) in positions {
        let Some(field) = col.mv_grid.get_at_luma(px, py) else {
            if std::env::var_os("VVC_MV_TRACE").is_some() && node.x < 128 && node.y < 128 {
                eprintln!(
                    "tmvp-miss poc={} pos=({},{}) size={}x{} at=({},{}) list={}",
                    g.poc, x, y, w, h, px, py, list_idx
                );
            }
            continue;
        };
        let mut col_list = if g.check_ldc {
            list_idx
        } else {
            usize::from(g.ph.collocated_from_l0)
        };
        let mut col_ref = field.mv_field[col_list].ref_idx;
        if col_ref < 0 {
            col_list = 1 - col_list;
            col_ref = field.mv_field[col_list].ref_idx;
        }
        if col_ref < 0 {
            continue;
        }
        let (Some(col_ref_poc), Some(curr_ref_poc)) = (
            col.ref_pocs[col_list].get(col_ref as usize).copied(),
            g.ref_poc(list_idx, ref_idx),
        ) else {
            continue;
        };
        let scale = dist_scale_factor(g.poc, curr_ref_poc, col.poc, col_ref_poc);
        let mv = scale_mv_fixed(
            super::inter::Mv::new(
                round_mv_comp(field.mv_field[col_list].mv.hor),
                round_mv_comp(field.mv_field[col_list].mv.ver),
            ),
            scale,
        );
        if std::env::var_os("VVC_MV_TRACE").is_some() && node.x < 128 && node.y < 128 {
            eprintln!(
                "tmvp-hit poc={} pos=({},{}) size={}x{} at=({},{}) list={} colref={} scale={} mv=({},{})",
                g.poc, x, y, w, h, px, py, list_idx, col_ref, scale, mv.hor, mv.ver
            );
        }
        out = Some(super::inter::InterMergeCand {
            inter_dir: 1 << list_idx,
            mv_field: [
                if list_idx == 0 {
                    super::inter::MvField::new(mv, ref_idx)
                } else {
                    super::inter::MvField::default()
                },
                if list_idx == 1 {
                    super::inter::MvField::new(mv, ref_idx)
                } else {
                    super::inter::MvField::default()
                },
            ],
            bcw_idx: super::inter::BCW_DEFAULT,
            use_alt_hpel_if: false,
        });
        break;
    }
    out
}

/// `CU::saveMotionForHmvp` / `CodingStructure::addMiToLut`
/// (`UnitTools.cpp:367-396`, `CodingStructure.cpp:1385-1408`).
fn push_hmvp(g: &mut Gen<'_>, node: &Node, mut cand: super::inter::InterMergeCand) {
    let p = usize::from(g.sps.log2_parallel_merge_level_minus2) + 2;
    let mask = !((1usize << p) - 1);
    if ((node.x ^ (node.x + node.w)) & mask) == 0 || ((node.y ^ (node.y + node.h)) & mask) == 0 {
        return;
    }
    if cand.inter_dir != 3 {
        cand.bcw_idx = super::inter::BCW_DEFAULT;
    }
    if let Some(pos) = g.hmvp.iter().position(|c| *c == cand) {
        g.hmvp.remove(pos);
    } else if g.hmvp.len() >= super::inter::MAX_NUM_HMVP_CANDS {
        g.hmvp.remove(0);
    }
    g.hmvp.push(cand);
}

/// TMVP entry of the regular Merge candidate list (`UnitTools.cpp:1410-1480`).
fn merge_tmvp(g: &Gen<'_>, node: &Node, is_b: bool) -> Option<super::inter::InterMergeCand> {
    let mut c = super::inter::InterMergeCand {
        inter_dir: 0,
        mv_field: [super::inter::MvField::default(); 2],
        bcw_idx: super::inter::BCW_DEFAULT,
        use_alt_hpel_if: false,
    };
    for l in 0..if is_b { 2 } else { 1 } {
        if let Some(t) = tmvp_candidate(g, node, l, 0) {
            c.mv_field[l] = t.mv_field[l];
            c.inter_dir |= 1 << l;
        }
    }
    (c.inter_dir != 0).then_some(c)
}

/// `PU::getInterMergeCandidates()` (`UnitTools.cpp:1059-1700`).
fn derive_merge_list(
    g: &Gen<'_>,
    node: &Node,
    hmvp: &[super::inter::InterMergeCand],
    col: Option<super::inter::InterMergeCand>,
    is_b: bool,
) -> Vec<super::inter::InterMergeCand> {
    let max_cands = usize::from(g.sps.max_num_merge_cand).clamp(1, 6);
    let (w, h, x, y) = (node.w, node.h, node.x, node.y);
    let mut list: Vec<super::inter::InterMergeCand> = Vec::with_capacity(max_cands);
    let push = |info: MotionInfo, list: &mut Vec<super::inter::InterMergeCand>| {
        if list.len() >= max_cands {
            return;
        }
        let mut c = super::inter::InterMergeCand {
            inter_dir: info.inter_dir,
            mv_field: [info.field(0), info.field(1)],
            bcw_idx: if info.inter_dir == 3 {
                info.bcw
            } else {
                super::inter::BCW_DEFAULT
            },
            use_alt_hpel_if: false,
        };
        if !is_b {
            c.inter_dir &= 1;
            c.mv_field[1] = super::inter::MvField::default();
        }
        if c.inter_dir != 0 {
            list.push(c);
        }
    };

    let b1 = if y > 0 {
        g.motion(x + w - 1, y - 1)
    } else {
        None
    }
    .filter(|m| {
        !m.affine && is_diff_mer(g, x, y, x + w - 1, y - 1) && !g.same_cu(node, x + w - 1, y - 1)
    });
    let a1 = if x > 0 {
        g.motion(x - 1, y + h - 1)
    } else {
        None
    }
    .filter(|m| {
        !m.affine && is_diff_mer(g, x, y, x - 1, y + h - 1) && !g.same_cu(node, x - 1, y + h - 1)
    });
    if let Some(m) = b1 {
        push(m, &mut list);
    }
    if list.len() < max_cands {
        if let Some(m) = a1 {
            if b1.map_or(true, |b| !b.same_motion(&m)) {
                push(m, &mut list);
            }
        }
    }
    if list.len() == max_cands {
        return list;
    }

    let b0 = if y > 0 { g.motion(x + w, y - 1) } else { None }
        .filter(|m| !m.affine && is_diff_mer(g, x, y, x + w, y - 1));
    if let Some(m) = b0 {
        if b1.map_or(true, |b| !b.same_motion(&m)) {
            push(m, &mut list);
        }
    }
    if list.len() == max_cands {
        return list;
    }
    let a0 = if x > 0 { g.motion(x - 1, y + h) } else { None }
        .filter(|m| !m.affine && is_diff_mer(g, x, y, x - 1, y + h));
    if let Some(m) = a0 {
        if a1.map_or(true, |a| !a.same_motion(&m)) {
            push(m, &mut list);
        }
    }
    if list.len() == max_cands {
        return list;
    }

    if list.len() < 4 {
        let b2 = if x > 0 && y > 0 {
            g.motion(x - 1, y - 1)
        } else {
            None
        }
        .filter(|m| !m.affine && is_diff_mer(g, x, y, x - 1, y - 1));
        if let Some(m) = b2 {
            let da = a1.map_or(true, |a| !a.same_motion(&m));
            let db = b1.map_or(true, |b| !b.same_motion(&m));
            if da && db {
                push(m, &mut list);
            }
        }
    }
    if list.len() == max_cands {
        return list;
    }

    // TMVP (`H.266 §8.5.2.11`): not available for 4x8/8x4 units.
    if g.ph.temporal_mvp_enabled && w + h > 12 {
        if let Some(m) = col {
            if list.len() < max_cands {
                list.push(m);
            }
        }
    }
    if list.len() == max_cands {
        return list;
    }

    // `addMergeHmvpCand` (`UnitTools.cpp:838-929`): only the two newest
    // history entries are pruned against A1/B1; older ones are always added.
    for (mrg_idx, c) in hmvp.iter().rev().enumerate() {
        if list.len() >= max_cands.saturating_sub(1) {
            break;
        }
        if mrg_idx < 2 {
            let dup = |n: Option<MotionInfo>| {
                n.is_some_and(|n| {
                    n.inter_dir == c.inter_dir
                        && [n.ref_idx[0], n.ref_idx[1]]
                            == [c.mv_field[0].ref_idx, c.mv_field[1].ref_idx]
                        && n.mv == [c.mv_field[0].mv, c.mv_field[1].mv]
                })
            };
            if dup(a1) || dup(b1) {
                continue;
            }
        }
        list.push(*c);
    }

    if list.len() >= 2 && list.len() < max_cands {
        let (c0, c1) = (list[0], list[1]);
        let mut avg = super::inter::InterMergeCand {
            inter_dir: 0,
            mv_field: [super::inter::MvField::default(); 2],
            bcw_idx: 2,
            use_alt_hpel_if: false,
        };
        for l in 0..if is_b { 2 } else { 1 } {
            let (f0, f1) = (c0.mv_field[l], c1.mv_field[l]);
            if f0.is_valid() && f1.is_valid() && f0.ref_idx == f1.ref_idx {
                avg.mv_field[l] = super::inter::MvField::new(
                    super::inter::Mv::new((f0.mv.hor + f1.mv.hor) / 2, (f0.mv.ver + f1.mv.ver) / 2),
                    f0.ref_idx,
                );
                avg.inter_dir |= 1 << l;
            } else if f0.is_valid() {
                avg.mv_field[l] = f0;
                avg.inter_dir |= 1 << l;
            } else if f1.is_valid() {
                avg.mv_field[l] = f1;
                avg.inter_dir |= 1 << l;
            }
        }
        if avg.inter_dir != 0 {
            list.push(avg);
        }
    }

    while list.len() < max_cands {
        let i = list.len();
        let mut c = super::inter::InterMergeCand {
            inter_dir: if is_b { 3 } else { 1 },
            mv_field: [super::inter::MvField::default(); 2],
            bcw_idx: 2,
            use_alt_hpel_if: false,
        };
        let r0 = i.min(usize::from(g.num_ref_idx[0].saturating_sub(1))) as i8;
        c.mv_field[0] = super::inter::MvField::new(super::inter::Mv::default(), r0);
        if is_b {
            let r1 = i.min(usize::from(g.num_ref_idx[1].saturating_sub(1))) as i8;
            c.mv_field[1] = super::inter::MvField::new(super::inter::Mv::default(), r1);
        }
        list.push(c);
    }
    list
}

/// `PU::fillMvpCand()` (`UnitTools.cpp`): left/above spatial candidates with MV
/// scaling, then HMVP, then zero.
fn derive_amvp_list(
    g: &Gen<'_>,
    node: &Node,
    list_idx: usize,
    ref_idx: usize,
    hmvp: &[super::inter::InterMergeCand],
    imv: u8,
) -> [super::inter::Mv; 2] {
    let (w, h, x, y) = (node.w, node.h, node.x, node.y);
    let target = g.ref_poc(list_idx, ref_idx as i8);
    let mut out: Vec<super::inter::Mv> = Vec::with_capacity(2);
    // `addMVPCandUnscaled` (`UnitTools.cpp:2918-2993`): a spatial neighbour is
    // accepted only when one of its reference POCs equals the target POC; the
    // MV is then copied without temporal scaling.
    let add = |pos: (usize, usize), out: &mut Vec<super::inter::Mv>| {
        let Some(m) = g.motion(pos.0, pos.1) else {
            return false;
        };
        for l in [list_idx, 1 - list_idx] {
            if m.has(l) && g.ref_poc(l, m.ref_idx[l]) == target {
                let mv = m.mv[l];
                if !out.contains(&mv) {
                    out.push(mv);
                }
                return true;
            }
        }
        false
    };
    if x > 0 {
        if !add((x - 1, y + h), &mut out) {
            add((x - 1, y + h - 1), &mut out);
        }
    }
    if out.len() < 2 && y > 0 {
        if !add((x + w, y - 1), &mut out) && !add((x + w - 1, y - 1), &mut out) {
            add((x - 1, y - 1), &mut out);
        }
    }
    // VTM appends TMVP even when it repeats a spatial candidate; only the two
    // spatial candidates are pruned for equality (`UnitTools.cpp:2138-2144`).
    if let Some(t) = tmvp_candidate(g, node, list_idx, ref_idx as i8) {
        if out.len() < 2 {
            out.push(t.mv_field[list_idx].mv);
        }
    }
    for c in hmvp.iter().rev() {
        if out.len() >= 2 {
            break;
        }
        let f = c.mv_field[list_idx];
        if f.is_valid() && !out.contains(&f.mv) {
            out.push(f.mv);
        }
    }
    for mv in &mut out {
        *mv = round_to_amvr(*mv, imv);
    }
    while out.len() < 2 {
        out.push(super::inter::Mv::default());
    }
    if std::env::var_os("VVC_MV_TRACE").is_some() && node.x < 128 && node.y < 128 {
        eprintln!(
            "amvp poc={} pos=({},{}) l={} ref={} imv={} cands=[({},{}) ({},{})]",
            g.poc,
            node.x,
            node.y,
            list_idx,
            ref_idx,
            imv,
            out[0].hor,
            out[0].ver,
            out[1].hor,
            out[1].ver
        );
    }
    [out[0], out[1]]
}

/// Parse one inter coding unit.
///
/// The prediction and residual reconstruction of inter coding units is the
/// next decoder stage: the syntax below is complete for the tools that
/// `d_ra8` uses, and `VVC_INTER_SYNTAX_ONLY` exists so that the CABAC
/// synchronisation of those pictures can be verified while reconstruction is
/// still missing.
fn decode_inter_cu(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
    skip: bool,
) -> Result<(), VvcError> {
    let w = node.w;
    let h = node.h;
    let mut cu = InterCu {
        skip,
        ..InterCu::default()
    };

    // `prediction_unit()`: `merge_flag` unless the CU is a skip CU.
    cu.merge = if skip {
        true
    } else {
        bit_is_on(dec, ctx, ctx::MERGE_FLAG)
    };

    if cu.merge {
        // `merge_data()`
        if g.sps.max_num_affine_merge_cand > 0 && w >= 8 && h >= 8 {
            cu.affine = bit_is_on(dec, ctx, ctx::SUBBLOCK_MERGE_FLAG + affine_ctx(g, &node));
        }
        if cu.affine {
            cu.merge_idx = unary_index(
                dec,
                ctx,
                ctx::AFF_MERGE_IDX,
                usize::from(g.sps.max_num_affine_merge_cand),
            );
        } else {
            let ciip_avail = g.sps.ciip_enabled && !skip && w < 128 && h < 128 && w * h >= 64;
            let geo_avail = g.sps.gpm_enabled
                && g.sh.slice_type == super::hls::VvcSliceType::B
                && g.sps.max_num_gpm_cand > 1
                && (8..=64).contains(&w)
                && (8..=64).contains(&h)
                && w < 8 * h
                && h < 8 * w;
            let regular = if geo_avail || ciip_avail {
                bit_is_on(dec, ctx, ctx::REGULAR_MERGE_FLAG + usize::from(!skip))
            } else {
                true
            };
            if regular {
                cu.mmvd = g.sps.mmvd_enabled && bit_is_on(dec, ctx, ctx::MMVD_FLAG);
                if cu.mmvd {
                    let (base_idx, step, position) =
                        mmvd_merge_idx(dec, ctx, g.sps.max_num_merge_cand);
                    let _ = (base_idx, step, position);
                } else {
                    cu.merge_idx = unary_index(
                        dec,
                        ctx,
                        ctx::MERGE_IDX,
                        usize::from(g.sps.max_num_merge_cand),
                    );
                }
            } else if ciip_avail && geo_avail {
                cu.ciip = bit_is_on(dec, ctx, ctx::CIIP_FLAG);
                cu.geo = !cu.ciip;
            } else if ciip_avail {
                cu.ciip = true;
            } else {
                cu.geo = true;
            }
            if cu.geo {
                // `merge_idx()` for a geometric-partitioning merge: the split
                // direction is a truncated binary code over
                // `GEO_NUM_PARTITION_MODE` and the two merge candidates are
                // equal-probability unary codes (`CABACReader.cpp:2395-2418`).
                cu.geo_split_dir = dec.decode_trunc_bin(64) as u8;
                let num_cand_minus2 = usize::from(g.sps.max_num_gpm_cand).saturating_sub(2);
                let mut cand0 = 0u8;
                let mut cand1 = 0u8;
                if bit_is_on(dec, ctx, ctx::MERGE_IDX) {
                    cand0 += unary_max_eqprob(dec, num_cand_minus2) + 1;
                }
                if num_cand_minus2 > 0 && bit_is_on(dec, ctx, ctx::MERGE_IDX) {
                    cand1 += unary_max_eqprob(dec, num_cand_minus2 - 1) + 1;
                }
                if cand1 >= cand0 {
                    cand1 += 1;
                }
                cu.geo_cand = [cand0, cand1];
            }
        }
    } else {
        // `inter_pred_idc()`
        if g.sh.slice_type == super::hls::VvcSliceType::P {
            cu.inter_dir = 1;
        } else {
            let restricted = (w == 4 && h == 4) || w + h == 12;
            if !restricted && bit_is_on(dec, ctx, ctx::INTER_DIR + inter_dir_ctx(w, h)) {
                cu.inter_dir = 3;
            }
            if cu.inter_dir != 3 {
                cu.inter_dir = if bit_is_on(dec, ctx, ctx::INTER_DIR + 5) {
                    2
                } else {
                    1
                };
            }
        }
        // `affine_flag()` + `affine_type()`
        if g.sps.affine_enabled && w > 8 && h > 8 {
            cu.affine = bit_is_on(dec, ctx, ctx::AFFINE_FLAG + affine_ctx(g, &node));
            if cu.affine && g.sps.affine_type_6param {
                cu.affine_6param = bit_is_on(dec, ctx, ctx::AFFINE_TYPE);
            }
        }
        // `smvd_mode()`
        if cu.inter_dir == 3 && !cu.affine && g.bidir_pred {
            cu.smvd = if bit_is_on(dec, ctx, ctx::SMVD_FLAG) {
                1
            } else {
                0
            };
        }
        // `ref_idx()`, `mvd_coding()`, `mvp_flag()` per list.
        let num_mv = if cu.affine {
            if cu.affine_6param { 3 } else { 2 }
        } else {
            1
        };
        if cu.inter_dir != 2 {
            cu.ref_idx[0] = ref_idx(dec, ctx, g.num_ref_idx[0], cu.smvd != 0, 0);
            for i in 0..num_mv {
                cu.mvd[0] = super::slice::decode_mvd(dec, ctx);
                let _ = i;
            }
            cu.mvp_idx[0] = u8::from(bit_is_on(dec, ctx, ctx::MVP_IDX));
        }
        if cu.inter_dir != 1 {
            if cu.smvd == 0 {
                cu.ref_idx[1] = ref_idx(dec, ctx, g.num_ref_idx[1], false, 1);
                // `ph_mvd_l1_zero_flag` suppresses the list-1 MVD for BI CUs.
                if !(g.ph.mvd_l1_zero && cu.inter_dir == 3) {
                    for i in 0..num_mv {
                        cu.mvd[1] = super::slice::decode_mvd(dec, ctx);
                        let _ = i;
                    }
                }
            }
            cu.mvp_idx[1] = u8::from(bit_is_on(dec, ctx, ctx::MVP_IDX));
        }
        if cu.inter_dir == 3 && ((w == 4 && h == 4) || w + h == 12) {
            cu.inter_dir = 1;
            cu.ref_idx[1] = -1;
            cu.bcw = 2;
        }
        if cu.smvd != 0 {
            let cur = usize::from(cu.smvd - 1);
            cu.mvd[1 - cur] = super::inter::Mv::new(-cu.mvd[cur].hor, -cu.mvd[cur].ver);
            cu.ref_idx = g.sym_ref;
        }
    }

    let nonzero_mvd =
        cu.mvd[0].hor != 0 || cu.mvd[0].ver != 0 || cu.mvd[1].hor != 0 || cu.mvd[1].ver != 0;
    cu.imv = amvr_mode(g, dec, ctx, cu.affine, nonzero_mvd, nonzero_mvd);
    if !cu.merge
        && !cu.affine
        && g.sps.bcw_enabled
        && g.sh.slice_type == super::hls::VvcSliceType::B
        && w * h >= 256
        && cu.inter_dir == 3
    {
        cu.bcw = bcw_flag(dec, ctx);
    }

    // `cu_residual()`: `rqt_root_cbf`, `sbt_mode`, transform tree.
    let mut cuctx = CuCtx::default();
    let mut tus = Vec::new();
    let mut root_cbf = true;
    if !cu.merge {
        root_cbf = bit_is_on(dec, ctx, ctx::QT_ROOT_CBF);
    }
    if !root_cbf {
        if std::env::var_os("VVC_TRACE").is_some() {
            eprintln!(
                "inter_cu pos=({},{}) size={}x{} skip={} merge={} dir={} root_cbf=0",
                node.x, node.y, w, h, cu.skip, cu.merge, cu.inter_dir
            );
        }
        let mut plan = CuPlan {
            tree: node.tree,
            x: node.x,
            w,
            h,
            qp_y: g.qp_y,
            qp_cb: g.qp_cb,
            qp_cr: g.qp_cr,
            mode: 0,
            mip: None,
            isp: predict::IspMode::None,
            mrl: 0,
            bdpcm_l: 0,
            bdpcm_c: 0,
            chroma_mode: 0,
            lfnst_mode_c: 0,
            lfnst: 0,
            mts: 0,
            implicit_mts: false,
            dep_quant: g.sh.dep_quant_used,
            sbt: 0,
            intra: false,
        };
        return reconstruct_inter_cu(g, &node, &cu, &mut plan, &[]);
    }

    // A skip coding unit carries no residual syntax: `coding_unit()` returns
    // right after `prediction_unit()` (`CABACReader.cpp:888-897`), so neither
    // `sbt_mode()` nor the transform tree is coded.
    if cu.skip {
        if std::env::var_os("VVC_TRACE").is_some() {
            eprintln!(
                "inter_cu pos=({},{}) size={}x{} skip=true merge={} subblock={} mmvd={} merge_idx={}",
                node.x, node.y, w, h, cu.merge, cu.affine, cu.mmvd, cu.merge_idx
            );
        }
        let mut plan = CuPlan {
            tree: node.tree,
            x: node.x,
            w,
            h,
            qp_y: g.qp_y,
            qp_cb: g.qp_cb,
            qp_cr: g.qp_cr,
            mode: 0,
            mip: None,
            isp: predict::IspMode::None,
            mrl: 0,
            bdpcm_l: 0,
            bdpcm_c: 0,
            chroma_mode: 0,
            lfnst_mode_c: 0,
            lfnst: 0,
            mts: 0,
            implicit_mts: false,
            dep_quant: g.sh.dep_quant_used,
            sbt: 0,
            intra: false,
        };
        return reconstruct_inter_cu(g, &node, &cu, &mut plan, &[]);
    }

    // `sbt_mode()`
    let mut sbt = 0u8;
    if root_cbf {
        sbt = decode_sbt_mode(g, dec, ctx, &node, cu.affine)?;
    }

    let mut plan = CuPlan {
        tree: node.tree,
        x: node.x,
        w,
        h,
        qp_y: g.qp_y,
        qp_cb: g.qp_cb,
        qp_cr: g.qp_cr,
        mode: 0,
        mip: None,
        isp: predict::IspMode::None,
        mrl: 0,
        bdpcm_l: 0,
        bdpcm_c: 0,
        chroma_mode: 0,
        lfnst_mode_c: 0,
        lfnst: 0,
        mts: 0,
        implicit_mts: false,
        dep_quant: g.sh.dep_quant_used,
        sbt,
        intra: false,
    };
    let mut isp_prev = false;
    let mut isp_any_cbf = false;
    {
        transform_tree(
            g,
            dec,
            ctx,
            &node,
            &mut plan,
            0,
            0,
            &mut tus,
            &mut cuctx,
            &mut isp_prev,
            &mut isp_any_cbf,
            false,
            false,
        )?;
    }
    decode_inter_mts(g, dec, ctx, &node, &mut plan, &tus, &cuctx);

    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "inter_cu pos=({},{}) size={}x{} skip={} merge={} subblock={} mmvd={} ciip={} dir={} merge_idx={} imv={} bcw={} sbt={} root_cbf=1",
            node.x,
            node.y,
            w,
            h,
            cu.skip,
            cu.merge,
            cu.affine,
            cu.mmvd,
            cu.ciip,
            cu.inter_dir,
            cu.merge_idx,
            cu.imv,
            cu.bcw,
            sbt
        );
    }
    reconstruct_inter_cu(g, &node, &cu, &mut plan, &tus)
}

/// Finish an inter coding unit: a unit without residual (skip or
/// `rqt_root_cbf == 0`) has no transform unit, so its prediction must be copied
/// into the picture; the others add their residual on top of it.
fn finish_inter(
    g: &mut Gen<'_>,
    node: &Node,
    plan: &mut CuPlan,
    tus: &[TuData],
) -> Result<(), VvcError> {
    if tus.is_empty() {
        apply_prediction(g, node);
    }
    reconstruct_tus(g, plan, tus)
}

/// Copy the stored prediction of a residual-free inter coding unit into the
/// picture and mark the samples as decoded.
fn apply_prediction(g: &mut Gen<'_>, node: &Node) {
    let (w, h) = (node.w, node.h);
    let (cw, ch) = (w / 2, h / 2);
    for j in 0..h {
        for i in 0..w {
            let v = g.pred_y.get(node.x + i, node.y + j);
            g.y.set(node.x + i, node.y + j, v);
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            let u = g.pred_u.get(node.x / 2 + i, node.y / 2 + j);
            let v = g.pred_v.get(node.x / 2 + i, node.y / 2 + j);
            g.u.set(node.x / 2 + i, node.y / 2 + j, u);
            g.v.set(node.x / 2 + i, node.y / 2 + j, v);
        }
    }
    for yy in node.y..(node.y + h).min(g.pic_h) {
        for xx in node.x..(node.x + w).min(g.pic_w) {
            g.dec_y[(yy / 4) * g.lgrid_w + xx / 4] = true;
        }
    }
    for yy in (node.y / 2)..(node.y / 2 + ch).min(g.cgrid_h) {
        for xx in (node.x / 2)..(node.x / 2 + cw).min(g.cgrid_w) {
            g.dec_c[yy * g.cgrid_w + xx] = true;
        }
    }
}

/// Luma, Cb and Cr prediction samples of one coding unit.
type CuPrediction = (Vec<i32>, Vec<i32>, Vec<i32>);

/// Motion compensation of one candidate into `(y, u, v)` predictions
/// (`InterPrediction::motionCompensation`).
fn mc_block(
    g: &Gen<'_>,
    node: &Node,
    cand: &super::inter::InterMergeCand,
    bdof: bool,
    dmvr: bool,
) -> Result<CuPrediction, VvcError> {
    let r0: Vec<&super::inter::RefFrame> = g.refs_l0.iter().collect();
    let r1: Vec<&super::inter::RefFrame> = g.refs_l1.iter().collect();
    if (cand.inter_dir & 1) != 0
        && cand.mv_field[0].ref_idx >= 0
        && cand.mv_field[0].ref_idx as usize >= r0.len()
    {
        return Err(VvcError::Bitstream(
            "missing L0 reference picture".to_string(),
        ));
    }
    if (cand.inter_dir & 2) != 0
        && cand.mv_field[1].ref_idx >= 0
        && cand.mv_field[1].ref_idx as usize >= r1.len()
    {
        return Err(VvcError::Bitstream(
            "missing L1 reference picture".to_string(),
        ));
    }
    // All motion vectors in the inter reconstruction state are kept in the
    // internal 1/16-pel precision expected by the interpolation kernels; AMVR
    // conversion happens once when the coded MVD is added to the predictor.
    Ok(super::inter::predict_inter_pu(
        cand, &r0, &r1, g.poc, node.x, node.y, node.w, node.h, bdof, dmvr,
    ))
}

/// Store the prediction of the current coding unit in the `pred_*` planes.
fn write_prediction(g: &mut Gen<'_>, node: &Node, py: &[i32], pcb: &[i32], pcr: &[i32]) {
    let max = (1i32 << g.bit_depth) - 1;
    let (cw, ch) = (node.w / 2, node.h / 2);
    for j in 0..node.h {
        for i in 0..node.w {
            g.pred_y.set(
                node.x + i,
                node.y + j,
                py[j * node.w + i].clamp(0, max) as u16,
            );
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            g.pred_u.set(
                node.x / 2 + i,
                node.y / 2 + j,
                pcb[j * cw + i].clamp(0, max) as u16,
            );
            g.pred_v.set(
                node.x / 2 + i,
                node.y / 2 + j,
                pcr[j * cw + i].clamp(0, max) as u16,
            );
        }
    }
}

/// Resolve the motion vector(s) of an inter coding unit, run motion
/// compensation into `pred_{y,u,v}` and reconstruct the residual on top.
fn reconstruct_inter_cu(
    g: &mut Gen<'_>,
    node: &Node,
    cu: &InterCu,
    plan: &mut CuPlan,
    tus: &[TuData],
) -> Result<(), VvcError> {
    let is_b_slice = g.sh.slice_type == super::hls::VvcSliceType::B;
    let trace_cu = std::env::var_os("VVC_MV_TRACE").is_some() && node.y < 128 && node.x < 128;
    if cu.affine {
        // Subblock merge (`H.266 §8.5.3`). The candidate list starts with
        // SbTMVP (requires a collocated *inter* picture) and the inherited
        // affine candidates of the neighbours; when none is available VTM pads
        // with zero MVs and reference index 0 (`UnitTools.cpp`, zero padding in
        // `PU::getAffineMergeCand`).
        let cand = super::inter::InterMergeCand {
            inter_dir: if is_b_slice { 3 } else { 1 },
            mv_field: [
                super::inter::MvField::new(super::inter::Mv::default(), 0),
                super::inter::MvField::new(super::inter::Mv::default(), 0),
            ],
            bcw_idx: 2,
            use_alt_hpel_if: false,
        };
        let (py, pcb, pcr) = mc_block(g, node, &cand, false, false)?;
        write_prediction(g, node, &py, &pcb, &pcr);
        g.set_lcell_inter(
            node.x,
            node.y,
            node.w,
            node.h,
            node.qt_depth,
            cu.skip,
            true,
            cand.inter_dir,
            [0, 0],
            [cand.mv_field[0].mv, cand.mv_field[1].mv],
            cand.bcw_idx,
        );
        return finish_inter(g, node, plan, tus);
    }
    if cu.ciip {
        return Err(VvcError::Unsupported("inter coding unit with CIIP"));
    }
    let is_b = g.sh.slice_type == super::hls::VvcSliceType::B;
    if cu.geo {
        // Geometric partitioning (`H.266 §8.5.7`): blend two uni/bi
        // predictions with the split-direction mask; BDOF and DMVR are
        // disabled for these coding units.
        let hmvp = g.hmvp.clone();
        let col = merge_tmvp(g, node, is_b);
        let list = derive_merge_list(g, node, &hmvp, col, is_b);
        let pick = |idx: u8| {
            list.get(usize::from(idx)).copied().unwrap_or_else(|| {
                *list
                    .first()
                    .unwrap_or(&super::inter::InterMergeCand::default())
            })
        };
        let c0 = pick(cu.geo_cand[0]);
        let c1 = pick(cu.geo_cand[1]);
        let (y0, u0, v0) = mc_block(g, node, &c0, false, false)?;
        let (y1, u1, v1) = mc_block(g, node, &c1, false, false)?;
        let (cw, ch) = (node.w / 2, node.h / 2);
        let py =
            super::inter::predict_gpm_block(&y0, &y1, node.w, node.h, cu.geo_split_dir, true, 1, 1);
        let pcb = super::inter::predict_gpm_block(&u0, &u1, cw, ch, cu.geo_split_dir, false, 2, 2);
        let pcr = super::inter::predict_gpm_block(&v0, &v1, cw, ch, cu.geo_split_dir, false, 2, 2);
        let cand = c0;
        write_prediction(g, node, &py, &pcb, &pcr);
        g.set_lcell_inter(
            node.x,
            node.y,
            node.w,
            node.h,
            node.qt_depth,
            cu.skip,
            cu.affine,
            cand.inter_dir,
            [cand.mv_field[0].ref_idx, cand.mv_field[1].ref_idx],
            [cand.mv_field[0].mv, cand.mv_field[1].mv],
            cand.bcw_idx,
        );
        return finish_inter(g, node, plan, tus);
    }
    let cand = if cu.merge {
        let hmvp = g.hmvp.clone();
        let col = merge_tmvp(g, node, is_b);
        let list = derive_merge_list(g, node, &hmvp, col, is_b);
        list.get(usize::from(cu.merge_idx))
            .copied()
            .unwrap_or_else(|| {
                *list
                    .first()
                    .unwrap_or(&super::inter::InterMergeCand::default())
            })
    } else {
        let mut c = super::inter::InterMergeCand {
            inter_dir: cu.inter_dir,
            mv_field: [super::inter::MvField::default(); 2],
            bcw_idx: cu.bcw,
            use_alt_hpel_if: cu.imv == 3,
        };
        for l in 0..2usize {
            if cu.inter_dir & (1 << l) == 0 {
                continue;
            }
            if l == 1 && cu.smvd != 0 {
                continue;
            }
            let ref_idx = cu.ref_idx[l].max(0);
            let mvp = derive_amvp_list(g, node, l, ref_idx as usize, &g.hmvp, cu.imv)
                [usize::from(cu.mvp_idx[l])];
            if std::env::var_os("VVC_MV_TRACE").is_some() && node.x < 128 && node.y < 128 {
                eprintln!(
                    "mvp pos=({},{}) l={} ref={} mvp=({},{}) mvd=({},{})",
                    node.x, node.y, l, ref_idx, mvp.hor, mvp.ver, cu.mvd[l].hor, cu.mvd[l].ver
                );
            }
            let mvd = mvd_to_internal(cu.mvd[l], cu.imv);
            c.mv_field[l] = super::inter::MvField::new(
                super::inter::Mv::new(mvp.hor + mvd.hor, mvp.ver + mvd.ver),
                ref_idx,
            );
        }
        if cu.smvd != 0 {
            let cur = usize::from(cu.smvd - 1);
            let other = 1 - cur;
            let f = c.mv_field[cur];
            let sym_ref = g.sym_ref[other];
            c.mv_field[other] =
                super::inter::MvField::new(super::inter::Mv::new(-f.mv.hor, -f.mv.ver), sym_ref);
            c.inter_dir = 3;
        }
        c
    };

    if trace_cu {
        eprintln!(
            "hmvp poc={} pos=({},{}) n={} {:?}",
            g.poc,
            node.x,
            node.y,
            g.hmvp.len(),
            g.hmvp
                .iter()
                .map(|c| (c.inter_dir, c.mv_field[0].ref_idx, c.mv_field[0].mv.hor))
                .collect::<Vec<_>>()
        );
        let col = merge_tmvp(g, node, is_b_slice);
        let list = derive_merge_list(g, node, &g.hmvp.clone(), col, is_b_slice);
        eprintln!(
            "mvcu poc={} pos=({},{}) size={}x{} skip={} merge={} midx={} dir={} ref={:?} mv=[({},{}),({},{})] bcw={} cands={:?}",
            g.poc,
            node.x,
            node.y,
            node.w,
            node.h,
            cu.skip,
            cu.merge,
            cu.merge_idx,
            cand.inter_dir,
            [cand.mv_field[0].ref_idx, cand.mv_field[1].ref_idx],
            cand.mv_field[0].mv.hor,
            cand.mv_field[0].mv.ver,
            cand.mv_field[1].mv.hor,
            cand.mv_field[1].mv.ver,
            cand.bcw_idx,
            list.iter()
                .take(6)
                .map(|c| (
                    c.inter_dir,
                    c.mv_field[0].ref_idx,
                    c.mv_field[0].mv.hor,
                    c.mv_field[0].mv.ver
                ))
                .collect::<Vec<_>>()
        );
    }
    // Motion compensation (`InterPrediction::motionCompensation`).
    let bdof = g.sps.bdof_enabled && !g.ph.bdof_disabled;
    let dmvr = g.sps.dmvr_enabled && !g.ph.dmvr_disabled && cu.merge;
    let (py, pcb, pcr) = mc_block(g, node, &cand, bdof, dmvr)?;
    write_prediction(g, node, &py, &pcb, &pcr);

    g.set_lcell_inter(
        node.x,
        node.y,
        node.w,
        node.h,
        node.qt_depth,
        cu.skip,
        cu.affine,
        cand.inter_dir,
        [cand.mv_field[0].ref_idx, cand.mv_field[1].ref_idx],
        [cand.mv_field[0].mv, cand.mv_field[1].mv],
        cand.bcw_idx,
    );
    push_hmvp(g, node, cand);
    // Residual on top of the inter prediction (`reconstruct_tus` reads the
    // prediction from the `pred_*` planes when `plan.intra == false`).
    finish_inter(g, node, plan, tus)
}

/// `ref_idx_l0/l1` (`H.266 §7.3.8.7`, VTM `CABACReader::ref_idx`).
fn ref_idx(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    num_ref: u8,
    smvd: bool,
    _list: usize,
) -> i8 {
    if smvd || num_ref <= 1 {
        return 0;
    }
    if !dec.decode_bin(&mut ctx.models[ctx::REF_PIC]) {
        return 0;
    }
    if num_ref <= 2 || !dec.decode_bin(&mut ctx.models[ctx::REF_PIC + 1]) {
        return 1;
    }
    let mut idx = 3i8;
    loop {
        if i16::from(idx) >= i16::from(num_ref) || !dec.decode_bin_ep() {
            return idx - 1;
        }
        idx += 1;
    }
}

/// `sbt_mode()` (`H.266 §7.3.8.7`, VTM `CABACReader.cpp:1628-1683`).
fn decode_sbt_mode(
    g: &Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    affine: bool,
) -> Result<u8, VvcError> {
    // `CU::checkAllowedSbt()` does not exclude subblock-merge (affine) CUs.
    let _ = affine;
    let allowed = sbt_allowed(g, node);
    if allowed == 0 {
        return Ok(0);
    }
    let ctx_idx = usize::from(node.w * node.h <= 256);
    if !dec.decode_bin(&mut ctx.models[ctx::SBT_FLAG + ctx_idx]) {
        return Ok(0);
    }
    let (ver_half, hor_half) = (allowed & 1 != 0, allowed & 2 != 0);
    let (ver_quad, hor_quad) = (allowed & 4 != 0, allowed & 8 != 0);
    let quad = if (hor_half || ver_half) && (hor_quad || ver_quad) {
        dec.decode_bin(&mut ctx.models[ctx::SBT_QUAD_FLAG])
    } else {
        false
    };
    let hor = if (quad && ver_quad && hor_quad) || (!quad && ver_half && hor_half) {
        let c = if node.w == node.h {
            0
        } else if node.w < node.h {
            1
        } else {
            2
        };
        dec.decode_bin(&mut ctx.models[ctx::SBT_HOR_FLAG + c])
    } else {
        (quad && hor_quad) || (!quad && hor_half)
    };
    let pos = u8::from(dec.decode_bin(&mut ctx.models[ctx::SBT_POS_FLAG]));
    // Encode the mode as `sbtIdx * 2 + pos` so the transform tree can split.
    // `SbtIdx`: `SBT_VER_HALF = 0`, `SBT_HOR_HALF = 1`, `SBT_VER_QUAD = 2`,
    // `SBT_HOR_QUAD = 3` (`TypeDef.h`).
    let idx = match (hor, quad) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    };
    Ok(idx * 2 + pos + 1)
}

/// `CU::checkAllowedSbt` bit mask (`H.266 §7.4.12.7`,
/// VTM `UnitTools.cpp`, `SBT_VER_HALF = 1`, `SBT_HOR_HALF = 2`,
/// `SBT_VER_QUAD = 4`, `SBT_HOR_QUAD = 8`).
fn sbt_allowed(g: &Gen<'_>, node: &Node) -> u8 {
    // `CU::checkAllowedSbt()` (`Unit.cpp:453-495`).
    let (w, h) = (node.w, node.h);
    let max_sbt_cu = if g.sps.max_tb_size_64 { 64 } else { 32 };
    if !g.sps.sbt_enabled || w > max_sbt_cu || h > max_sbt_cu {
        return 0;
    }
    let min_sbt_cu = 8usize; // `1 << (MIN_CU_LOG2 + 1)`
    let mut mask = 0u8;
    if w >= min_sbt_cu {
        mask |= 1; // `SBT_VER_HALF`
    }
    if h >= min_sbt_cu {
        mask |= 2; // `SBT_HOR_HALF`
    }
    if w >= 2 * min_sbt_cu {
        mask |= 4; // `SBT_VER_QUAD`
    }
    if h >= 2 * min_sbt_cu {
        mask |= 8; // `SBT_HOR_QUAD`
    }
    mask
}

/// Inter `mts_idx()` (`H.266 §7.3.11.10`, VTM `CABACReader.cpp:3025-3057`).
fn decode_inter_mts(
    g: &Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    plan: &mut CuPlan,
    tus: &[TuData],
    cuctx: &CuCtx,
) {
    let (w, h) = (node.w, node.h);
    let first_ts = tus
        .iter()
        .find(|t| t.component == 0)
        .map(|t| t.ts)
        .unwrap_or(true);
    let allowed = g.sps.mts_enabled && g.sps.explicit_mts_inter && w <= 32 && h <= 32 && !first_ts;
    if !allowed || cuctx.violates_mts || !cuctx.mts_last_scan_pos || plan.sbt != 0 {
        return;
    }
    let mut v = 0u8;
    if dec.decode_bin(&mut ctx.models[ctx::MTS_IDX]) {
        v = 1;
        for i in 0..3usize {
            if !dec.decode_bin(&mut ctx.models[ctx::MTS_IDX + 1 + i]) {
                break;
            }
            v += 1;
        }
    }
    plan.mts = v;
}

fn decode_cu_luma(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    let w = node.w;
    let h = node.h;
    // `cu_skip_flag` / `pred_mode_flag` (`H.266 §7.3.8.7`). IBC and palette are
    // rejected for this decoder, so a skip or inter prediction mode always
    // selects the inter coding-unit syntax below.
    if g.sh.slice_type != super::hls::VvcSliceType::I && node.mode_type != ModeType::Intra {
        if g.sps.ibc_enabled {
            return Err(VvcError::Unsupported(
                "general coding-tree decoding does not implement IBC",
            ));
        }
        let skip = !(w == 4 && h == 4) && bit_is_on(dec, ctx, ctx::SKIP_FLAG + skip_ctx(g, &node));
        if skip {
            return decode_inter_cu(g, dec, ctx, node, true);
        }
        let intra = if node.mode_type == ModeType::All && !(w == 4 && h == 4) {
            bit_is_on(dec, ctx, ctx::PRED_MODE + pred_mode_ctx(g, &node))
        } else {
            node.mode_type == ModeType::All
        };
        if node.mode_type == ModeType::Inter || !intra {
            return decode_inter_cu(g, dec, ctx, node, false);
        }
    }
    let ts_max = 1usize << g.sps.log2_transform_skip_max_size;
    let bdpcm_l = if g.sps.bdpcm_enabled && w <= ts_max && h <= ts_max {
        decode_bdpcm(dec, ctx, false)
    } else {
        0
    };

    let mut plan = CuPlan {
        tree: node.tree,
        x: node.x,
        w,
        h,
        qp_y: g.qp_y,
        qp_cb: g.qp_cb,
        qp_cr: g.qp_cr,
        mode: 0,
        mip: None,
        isp: predict::IspMode::None,
        mrl: 0,
        bdpcm_l,
        bdpcm_c: 0,
        chroma_mode: 0,
        lfnst_mode_c: 0,
        lfnst: 0,
        mts: 0,
        implicit_mts: g.sps.mts_enabled && !g.sps.explicit_mts_intra,
        dep_quant: g.sh.dep_quant_used,
        sbt: 0,
        intra: true,
    };

    if bdpcm_l != 0 {
        plan.mode = if bdpcm_l == 2 { 50 } else { 18 };
    } else {
        let mip = if g.sps.mip_enabled {
            let (l, a) = mip_neighbours(g, &node);
            let n = predict::get_num_modes_mip(w, h) as u32;
            let c = if w > 2 * h || h > 2 * w {
                3
            } else {
                usize::from(l) + usize::from(a)
            };
            if bit_is_on(dec, ctx, ctx::MIP_FLAG + c) {
                let transposed = dec.decode_bin_ep();
                let mode = dec.decode_trunc_bin(n) as usize;
                Some((transposed, mode))
            } else {
                None
            }
        } else {
            None
        };
        if let Some(m) = mip {
            plan.mip = Some(m);
            plan.mode = 0;
        } else {
            let mrl = if g.sps.mrl_enabled && !node.y.is_multiple_of(g.ctu) {
                decode_mrl(dec, ctx)
            } else {
                0
            };
            plan.mrl = mrl;
            let isp = if g.sps.isp_enabled && mrl == 0 && w * h > 16 && w <= 64 && h <= 64 {
                decode_isp(dec, ctx)
            } else {
                0
            };
            plan.isp = match isp {
                1 => predict::IspMode::Hor,
                2 => predict::IspMode::Ver,
                _ => predict::IspMode::None,
            };
            let (left, above) = luma_mpm_neighbours(g, &node);
            let mpm = predict::derive_intra_mpms(left, above);
            let mpm_flag = if mrl != 0 {
                true
            } else {
                bit_is_on(dec, ctx, ctx::INTRA_LUMA_MPM_FLAG)
            };
            plan.mode = if mpm_flag {
                let planar_ctx = if isp != 0 { 0 } else { 1 };
                let mut idx = if mrl == 0 {
                    u32::from(bit_is_on(
                        dec,
                        ctx,
                        ctx::INTRA_LUMA_PLANAR_FLAG + planar_ctx,
                    ))
                } else {
                    1
                };
                if idx > 0 {
                    idx += dec.decode_bin_ep() as u32;
                }
                if idx > 1 {
                    idx += dec.decode_bin_ep() as u32;
                }
                if idx > 2 {
                    idx += dec.decode_bin_ep() as u32;
                }
                if idx > 3 {
                    idx += dec.decode_bin_ep() as u32;
                }
                mpm[idx as usize]
            } else {
                let rem = dec.decode_trunc_bin(61) as u8;
                predict::resolve_non_mpm_luma_mode(rem, &mpm)
            };
        }
    }

    // H.266 §8.7.4.1: `implicitMtsEnabled` is 1 for an ISP coding unit whenever
    // `sps_mts_enabled_flag` is 1 (independently of
    // `sps_explicit_mts_intra_enabled_flag`).
    if plan.isp != predict::IspMode::None {
        plan.implicit_mts = g.sps.mts_enabled;
    }

    if node.tree == TreeType::Joint {
        let cw = w / 2;
        let ch = h / 2;
        plan.bdpcm_c = if g.sps.bdpcm_enabled && cw <= ts_max && ch <= ts_max {
            decode_bdpcm(dec, ctx, true)
        } else {
            0
        };
        plan.chroma_mode = if plan.bdpcm_c != 0 {
            if plan.bdpcm_c == 2 { 50 } else { 18 }
        } else {
            decode_chroma_mode(dec, ctx, plan.mode, g.sps.cclm_enabled, plan.mip.is_some())
        };
        plan.lfnst_mode_c = chroma_lfnst_mode(plan.mode, plan.chroma_mode);
    }

    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "coding_unit() pos=({},{}) size={}x{} mip={} isp={:?} mrl={} mode={} bdpcm={} chroma={}",
            node.x,
            node.y,
            w,
            h,
            plan.mip.is_some(),
            plan.isp,
            plan.mrl,
            plan.mode,
            plan.bdpcm_l,
            plan.chroma_mode
        );
    }
    let depth64 = if g.ctu > 64 { 1 } else { 0 };
    g.set_lcu(
        node.x,
        node.y,
        w,
        h,
        node.qt_depth,
        plan.mode,
        plan.mip.is_some(),
        split_code(node.split_at(depth64)),
        plan.isp != predict::IspMode::None,
    );

    let mut cuctx = CuCtx::default();
    let mut tus = Vec::new();
    let mut isp_prev = false;
    let mut isp_any_cbf = false;
    transform_tree(
        g,
        dec,
        ctx,
        &node,
        &mut plan,
        0,
        0,
        &mut tus,
        &mut cuctx,
        &mut isp_prev,
        &mut isp_any_cbf,
        false,
        false,
    )?;
    decode_lfnst_mts(g, dec, ctx, &node, &mut plan, &mut cuctx, &tus);
    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "cu_done pos=({},{}) lfnst={} mts={} nz={} bin={}",
            node.x,
            node.y,
            plan.lfnst,
            plan.mts,
            tus.iter()
                .map(|t| t.levels.iter().filter(|&&l| l != 0).count())
                .sum::<usize>(),
            super::cabac::bin_count()
        );
    }
    reconstruct_tus(g, &plan, &tus)?;
    Ok(())
}

fn decode_cu_chroma(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    let cw = node.w / 2;
    let ch = node.h / 2;
    let ts_max = 1usize << g.sps.log2_transform_skip_max_size;
    let bdpcm_c = if g.sps.bdpcm_enabled && cw <= ts_max && ch <= ts_max {
        decode_bdpcm(dec, ctx, true)
    } else {
        0
    };
    let luma_mode = {
        let lx = (node.x + node.w / 2).min(g.pic_w.saturating_sub(1));
        let ly = (node.y + node.h / 2).min(g.pic_h.saturating_sub(1));
        g.lcell(lx, ly)
            .map_or(0, |c| if c.mip { 0 } else { c.mode })
    };
    let chroma_mode = if bdpcm_c != 0 {
        if bdpcm_c == 2 { 50 } else { 18 }
    } else {
        decode_chroma_mode(
            dec,
            ctx,
            luma_mode,
            cclm_enabled_for_chroma(g, &node),
            false,
        )
    };
    let mut plan = CuPlan {
        tree: TreeType::Chroma,
        x: node.x,
        w: node.w,
        h: node.h,
        qp_y: g.qp_y,
        qp_cb: g.qp_cb,
        qp_cr: g.qp_cr,
        mode: luma_mode,
        mip: None,
        isp: predict::IspMode::None,
        mrl: 0,
        bdpcm_l: 0,
        bdpcm_c,
        chroma_mode,
        lfnst_mode_c: chroma_lfnst_mode(luma_mode, chroma_mode),
        lfnst: 0,
        mts: 0,
        implicit_mts: g.sps.mts_enabled && !g.sps.explicit_mts_intra,
        dep_quant: g.sh.dep_quant_used,
        sbt: 0,
        intra: true,
    };
    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "chroma_cu pos=({},{}) size={}x{} mode={} bdpcm={}",
            node.x, node.y, cw, ch, chroma_mode, bdpcm_c
        );
    }
    g.set_ccu(node.x / 2, node.y / 2, cw, ch, node.qt_depth);
    let mut cuctx = CuCtx::default();
    let mut tus = Vec::new();
    let mut isp_prev = false;
    let mut isp_any_cbf = false;
    transform_tree(
        g,
        dec,
        ctx,
        &node,
        &mut plan,
        0,
        0,
        &mut tus,
        &mut cuctx,
        &mut isp_prev,
        &mut isp_any_cbf,
        true,
        false,
    )?;
    decode_lfnst_mts(g, dec, ctx, &node, &mut plan, &mut cuctx, &tus);
    reconstruct_tus(g, &plan, &tus)?;
    Ok(())
}

fn decode_bdpcm(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore, chroma: bool) -> u8 {
    let base = ctx::BDPCM_MODE + if chroma { 2 } else { 0 };
    if !dec.decode_bin(&mut ctx.models[base]) {
        0
    } else if dec.decode_bin(&mut ctx.models[base + 1]) {
        2
    } else {
        1
    }
}

fn decode_mrl(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> u8 {
    if !dec.decode_bin(&mut ctx.models[ctx::MULTI_REF_LINE_IDX]) {
        0
    } else if !dec.decode_bin(&mut ctx.models[ctx::MULTI_REF_LINE_IDX + 1]) {
        1
    } else {
        2
    }
}

fn decode_isp(dec: &mut BinDecoder<'_>, ctx: &mut CtxStore) -> u8 {
    if !dec.decode_bin(&mut ctx.models[ctx::ISP_MODE]) {
        0
    } else if dec.decode_bin(&mut ctx.models[ctx::ISP_MODE + 1]) {
        2
    } else {
        1
    }
}

/// Intra mode whose LFNST set and transpose flag apply to a chroma TU
/// (H.266 §8.7.4.3, VTM `TrQuant::xInvLfnst`): the actual chroma mode, with the
/// co-located luma mode substituted for an INTRA_LT/L/T_CCLM mode.
fn chroma_lfnst_mode(luma_mode: u8, chroma_mode: u8) -> u8 {
    if (81..=83).contains(&chroma_mode) {
        luma_mode
    } else {
        chroma_mode
    }
}

/// LFNST intra mode index (`0..=94`, VTM `TrQuant::getLFNSTIntraMode`) for a
/// transform block, derived from the §8.4.5.2.7 wide-angle mapping of the
/// prediction mode in force for that block.
///
/// `nw`/`nh` are the coding-block dimensions when ISP is active on luma and the
/// transform-block dimensions otherwise (§8.4.5.2.7, Eq. 318-321).
fn lfnst_intra_mode(plan: &CuPlan, luma: bool, cw: usize, ch: usize) -> u16 {
    let (nw, nh, mode) = if luma {
        if plan.isp == predict::IspMode::None {
            (cw, ch, plan.mode)
        } else {
            (plan.w, plan.h, plan.mode)
        }
    } else {
        (cw, ch, plan.lfnst_mode_c)
    };
    lfnst::get_lfnst_intra_mode(predict::get_wide_angle(nw, nh, mode))
}

fn decode_chroma_mode(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    luma_mode: u8,
    cclm: bool,
    luma_mip: bool,
) -> u8 {
    if cclm && dec.decode_bin(&mut ctx.models[ctx::CCLM_MODE_FLAG]) {
        if !dec.decode_bin(&mut ctx.models[ctx::CCLM_MODE_IDX]) {
            return 81;
        }
        return if dec.decode_bin_ep() { 83 } else { 82 };
    }
    if !dec.decode_bin(&mut ctx.models[ctx::INTRA_CHROMA_PRED_MODE]) {
        return if luma_mip { 0 } else { luma_mode };
    }
    let cand = dec.decode_bins_ep(2) as usize;
    let list = predict::derive_chroma_cand_modes(if luma_mip { 0 } else { luma_mode }, false);
    list[cand & 3]
}

fn mip_neighbours(g: &Gen<'_>, node: &Node) -> (bool, bool) {
    let left = g.lcell(node.x.saturating_sub(1), node.y);
    let above = g.lcell(node.x, node.y.saturating_sub(1));
    (left.is_some_and(|c| c.mip), above.is_some_and(|c| c.mip))
}

fn luma_mpm_neighbours(g: &Gen<'_>, node: &Node) -> (Option<u8>, Option<u8>) {
    let left = g.lcell(node.x.saturating_sub(1), node.y + node.h.saturating_sub(1));
    let above = if node.y > 0 && !node.y.is_multiple_of(g.ctu) {
        g.lcell(node.x + node.w.saturating_sub(1), node.y - 1)
    } else {
        None
    };
    (
        left.filter(|c| c.intra)
            .map(|c| if c.mip { 0 } else { c.mode }),
        above
            .filter(|c| c.intra)
            .map(|c| if c.mip { 0 } else { c.mode }),
    )
}

fn transform_tree(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    plan: &mut CuPlan,
    tr_depth: u8,
    sub_tu: i32,
    tus: &mut Vec<TuData>,
    cuctx: &mut CuCtx,
    isp_prev: &mut bool,
    isp_any_cbf: &mut bool,
    chroma_tree: bool,
    no_residual: bool,
) -> Result<(), VvcError> {
    let lim = limits(g, chroma_tree);
    let (w, h) = if chroma_tree {
        (node.w / 2, node.h / 2)
    } else {
        (node.w, node.h)
    };
    // `QTBTPartitioner::canSplit( TU_MAX_TR_SPLIT )` compares the **luma** area
    // against `MaxTbSize`, capped at 64 for areas larger than 64x64
    // (`UnitPartitioner.cpp:528-544`), and `PartitionerImpl::getMaxTuTiling`
    // divides every component by the luma-derived tile counts
    // (`UnitPartitioner.cpp:1155-1195`). Comparing the chroma dimensions against
    // the luma limit desynchronises 4:2:0 streams whose `MaxTbSize` is 32.
    let (lw, lh) = (node.w, node.h);
    let max_tr = if lw > 64 || lh > 64 { 64 } else { lim.max_tb };
    let nx = (lw / max_tr).max(1);
    let ny = (lh / max_tr).max(1);
    let can_max_split = nx > 1 || ny > 1;
    let isp_split = tr_depth == 0 && !chroma_tree && plan.isp != predict::IspMode::None;
    // `cu_sbt_flag` splits the coding unit's transform area into two tiles and
    // leaves one of them without residual (`H.266 §7.3.11.10`,
    // VTM `PartitionerImpl::getSbtTuTiling`, `UnitPartitioner.cpp:1194-1258`).
    let sbt_split = tr_depth == 0 && !chroma_tree && plan.sbt != 0 && !can_max_split;
    if can_max_split || isp_split || sbt_split {
        let parts: Vec<(usize, usize, usize, usize)> = if can_max_split {
            max_tu_tiles(w, h, nx, ny)
        } else if sbt_split {
            sbt_tu_tiles(w, h, plan.sbt)
        } else {
            predict::isp_subpartitions(0, 0, w, h, plan.isp)
        };
        for (i, (ox, oy, pw, ph)) in parts.into_iter().enumerate() {
            let child = Node {
                x: node.x + if chroma_tree { ox * 2 } else { ox },
                y: node.y + if chroma_tree { oy * 2 } else { oy },
                w: if chroma_tree { pw * 2 } else { pw },
                h: if chroma_tree { ph * 2 } else { ph },
                ..*node
            };
            if child.x < g.vis_w && child.y < g.vis_h {
                let no_residual = sbt_split && sbt_tile_has_no_residual(plan.sbt, i);
                transform_tree(
                    g,
                    dec,
                    ctx,
                    &child,
                    plan,
                    tr_depth + 1,
                    sub_tu + i as i32,
                    tus,
                    cuctx,
                    isp_prev,
                    isp_any_cbf,
                    chroma_tree,
                    no_residual,
                )?;
            }
        }
        return Ok(());
    }
    transform_unit(
        g,
        dec,
        ctx,
        node,
        plan,
        no_residual,
        tr_depth,
        sub_tu,
        tus,
        cuctx,
        isp_prev,
        isp_any_cbf,
        chroma_tree,
    )
}

/// Transform-block tiling when the area exceeds `MaxTbSize`
/// (VTM `PartitionerImpl::getMaxTuTiling`, `UnitPartitioner.cpp:1155-1195`).
///
/// `nx`/`ny` are the luma tile counts; the parts are returned in raster order
/// over that grid, which is what the `g_rsScanToZ`/`g_zScanTo*` tables produce.
/// Tile geometry of a subblock transform (`plan.sbt = sbt_idx * 2 + pos + 1`),
/// following VTM `PartitionerImpl::getSbtTuTiling`
/// (`UnitPartitioner.cpp:1194-1258`).
fn sbt_tu_tiles(w: usize, h: usize, sbt: u8) -> Vec<(usize, usize, usize, usize)> {
    let info = sbt.saturating_sub(1);
    let idx = info / 2;
    let pos = info % 2;
    match idx {
        0 => {
            // `SBT_VER_HALF`
            vec![(0, 0, w / 2, h), (w / 2, 0, w / 2, h)]
        }
        1 => {
            // `SBT_HOR_HALF`
            vec![(0, 0, w, h / 2), (0, h / 2, w, h / 2)]
        }
        2 => {
            // `SBT_VER_QUAD`
            if pos == 0 {
                vec![(0, 0, w / 4, h), (w / 4, 0, 3 * w / 4, h)]
            } else {
                vec![(0, 0, 3 * w / 4, h), (3 * w / 4, 0, w / 4, h)]
            }
        }
        _ => {
            // `SBT_HOR_QUAD`
            if pos == 0 {
                vec![(0, 0, w, h / 4), (0, h / 4, w, 3 * h / 4)]
            } else {
                vec![(0, 0, w, 3 * h / 4), (0, 3 * h / 4, w, h / 4)]
            }
        }
    }
}

/// Whether SBT tile `index` carries no residual
/// (`TransformUnit::checkTuNoResidual`).
fn sbt_tile_has_no_residual(sbt: u8, index: usize) -> bool {
    let pos = sbt.saturating_sub(1) % 2;
    (pos == 0 && index == 1) || (pos == 1 && index == 0)
}

fn max_tu_tiles(w: usize, h: usize, nx: usize, ny: usize) -> Vec<(usize, usize, usize, usize)> {
    let tw = w / nx;
    let th = h / ny;
    let mut out = Vec::with_capacity(nx * ny);
    for i in 0..nx * ny {
        out.push(((i % nx) * tw, (i / nx) * th, tw, th));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn transform_unit(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    plan: &mut CuPlan,
    no_residual: bool,
    tr_depth: u8,
    sub_tu: i32,
    tus: &mut Vec<TuData>,
    cuctx: &mut CuCtx,
    isp_prev: &mut bool,
    isp_any_cbf: &mut bool,
    chroma_tree: bool,
) -> Result<(), VvcError> {
    let (lw, lh) = (node.w, node.h);
    let isp = !chroma_tree && plan.isp != predict::IspMode::None;
    let joint = plan.tree == TreeType::Joint && !chroma_tree;
    let mut cbf = [false; 3];
    let mut jccr = 0u8;
    if no_residual {
        // A subblock-transform tile without residual infers every cbf to 0 and
        // codes no coefficient syntax (`TransformUnit::checkTuNoResidual`).
        tus.push(TuData {
            x: node.x,
            y: node.y,
            w: lw,
            h: lh,
            component: 0,
            levels: Vec::new(),
            cbf: false,
            ts: false,
            joint: 0,
        });
        return Ok(());
    }
    if chroma_tree || (joint && !isp) {
        let c0 = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
        let c1 = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(c0)]);
        cbf[1] = c0;
        cbf[2] = c1;
    }
    if !chroma_tree {
        let n_tus = if isp {
            match plan.isp {
                predict::IspMode::Hor => {
                    plan.h / predict::isp_split_dim(plan.w, plan.h, plan.isp).max(1)
                }
                predict::IspMode::Ver => {
                    plan.w / predict::isp_split_dim(plan.w, plan.h, plan.isp).max(1)
                }
                predict::IspMode::None => 1,
            }
        } else {
            1
        };
        let last = sub_tu as usize + 1 == n_tus;
        let prev = *isp_prev;
        let any_cbf = *isp_any_cbf;
        // `H.266 §7.3.11.10` / VTM `CABACReader.cpp:2759-2763`: an inter
        // transform unit at depth 0 with no chroma residual infers
        // `tu_y_coded_flag == 1` instead of coding it.
        cbf[0] = if isp && last && !any_cbf {
            true
        } else if !plan.intra && tr_depth == 0 && !cbf[1] && !cbf[2] {
            true
        } else {
            let c = ctx::QT_CBF_Y + if isp { 2 + usize::from(prev) } else { 0 };
            dec.decode_bin(&mut ctx.models[c])
        };
        *isp_prev = cbf[0];
        *isp_any_cbf = any_cbf || cbf[0];
    }

    // `joint_cb_cr()` follows the luma cbf and precedes the coefficient syntax
    // (VTM `CABACReader.cpp:2838-2840`, `2913-2925`): the flag is read for an
    // intra coding unit with any coded chroma component, and for an inter
    // coding unit only when both are coded.
    if plan.tree != TreeType::Luma && g.sps.joint_cbcr_enabled && (cbf[1] || cbf[2]) {
        let mask = u8::from(cbf[1]) * 2 + u8::from(cbf[2]);
        if (plan.intra && mask != 0) || mask == 3 {
            let on = dec.decode_bin(&mut ctx.models[ctx::JOINT_CB_CR_FLAG + usize::from(mask - 1)]);
            jccr = if on { mask } else { 0 };
        }
    }

    let comps: &[usize] = if chroma_tree {
        if plan.tree == TreeType::Chroma {
            &[1, 2]
        } else {
            &[1, 2]
        }
    } else if joint {
        &[0, 1, 2]
    } else {
        &[0]
    };

    for &comp in comps {
        let (cw, ch) = if comp == 0 {
            (lw, lh)
        } else {
            (lw / 2, lh / 2)
        };
        if cw == 0 || ch == 0 {
            continue;
        }
        let mut levels = Vec::new();
        let mut ts = false;
        if cbf[comp] && !(comp == 2 && jccr == 3) {
            // `TU::isTSAllowed` (`UnitTools.cpp`): transform skip is not
            // signalled in a subblock-transform coding unit.
            let ts_allowed = g.sps.transform_skip_enabled
                && plan.sbt == 0
                && plan.bdpcm_l == 0
                && plan.bdpcm_c == 0
                && (!isp || comp != 0)
                && cw <= (1usize << g.sps.log2_transform_skip_max_size)
                && ch <= (1usize << g.sps.log2_transform_skip_max_size);
            ts = ts_allowed
                && dec
                    .decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + usize::from(comp != 0)]);
            // A transform block larger than 32 samples zeroes the coefficients
            // outside the top-left 32x32 region but keeps the full block stride
            // for `scanPosLast` (`H.266 §7.4.11.6`, VTM `CABACReader.cpp:3142-3206`).
            let scan = CoeffScan::new(cw, ch);
            levels = vec![0i32; cw * ch];
            if ts {
                super::slice::decode_ts_block(dec, ctx, &scan, &mut levels);
            } else {
                residual::decode_residual_with_options(
                    dec,
                    ctx,
                    &scan,
                    &mut levels,
                    comp == 0,
                    g.sh.dep_quant_used,
                    g.sh.sign_data_hiding_used,
                )?;
            }
            if std::env::var_os("VVC_TRACE").is_some() {
                eprintln!(
                    "transform_unit() pos=({},{}) comp={} size={}x{} cbf={} ts={}",
                    node.x, node.y, comp, cw, ch, cbf[comp], ts
                );
            }
            update_constraints(cuctx, &scan, &levels, comp == 0, ts);
        }
        tus.push(TuData {
            x: node.x,
            y: node.y,
            w: lw,
            h: lh,
            component: comp,
            levels,
            cbf: cbf[comp] && !(comp == 2 && jccr == 3),
            ts,
            joint: jccr,
        });
    }
    let _ = tr_depth;
    Ok(())
}

fn update_constraints(cuctx: &mut CuCtx, scan: &CoeffScan, levels: &[i32], luma: bool, ts: bool) {
    if ts {
        return;
    }
    let Some(last) = (0..scan.raster.len())
        .rev()
        .find(|&p| levels.get(scan.raster[p]).is_some_and(|&l| l != 0))
    else {
        if std::env::var_os("VVC_TRACE").is_some() {
            eprintln!("constraints none w={} n={}", scan.width, levels.len());
        }
        return;
    };
    if std::env::var_os("VVC_TRACE").is_some() {
        eprintln!(
            "constraints last={} w={} nz={}",
            last,
            scan.width,
            levels.iter().filter(|&&l| l != 0).count()
        );
    }
    let height = scan.height;
    if scan.width >= 4 && height >= 4 {
        let max_lfnst = if (scan.width == 4 && height == 4) || (scan.width == 8 && height == 8) {
            7
        } else {
            15
        };
        cuctx.violates_lfnst[usize::from(!luma)] |= last > max_lfnst;
        cuctx.lfnst_last_scan_pos |= last >= 1;
    }
    if luma {
        cuctx.mts_last_scan_pos |= last >= 1;
        let width = scan.width;
        cuctx.violates_mts |= levels.iter().enumerate().any(|(i, &l)| {
            l != 0 && {
                let (x, y) = (i % width, i / width);
                x / 4 > 3 || y / 4 > 3
            }
        });
    }
}

fn decode_lfnst_mts(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: &Node,
    plan: &mut CuPlan,
    cuctx: &mut CuCtx,
    tus: &[TuData],
) {
    let chroma_tree = node.tree == TreeType::Chroma;
    let (w, h) = if chroma_tree {
        (node.w / 2, node.h / 2)
    } else {
        (node.w, node.h)
    };
    let luma_cu = !chroma_tree;
    let mut any_ts = false;
    for t in tus {
        if t.ts && ((t.component == 0) == luma_cu) {
            any_ts = true;
        }
    }
    let max_tb = limits(g, chroma_tree).max_tb;
    let luma_size = if chroma_tree {
        (node.w, node.h)
    } else {
        (w, h)
    };
    let mip_ok = plan.mip.map(|_| w >= 16 && h >= 16).unwrap_or(true);
    let isp_ok = plan.isp == predict::IspMode::None || {
        let dim = predict::isp_split_dim(w, h, plan.isp);
        if plan.isp == predict::IspMode::Hor {
            dim >= 4 && w >= 4
        } else {
            dim >= 4 && h >= 4
        }
    };
    if !mip_ok
        || !isp_ok
        || (chroma_tree && w.min(h) < 4)
        || luma_size.0 > max_tb
        || luma_size.1 > max_tb
    {
        plan.lfnst = 0;
    } else if g.sps.lfnst_enabled {
        let constrained =
            (luma_cu && cuctx.violates_lfnst[0]) || (chroma_tree && cuctx.violates_lfnst[1]);
        if std::env::var_os("VVC_TRACE").is_some() {
            eprintln!(
                "lfnst_check pos=({},{}) last={} violates={:?} anysts={} constrained={}",
                node.x,
                node.y,
                cuctx.lfnst_last_scan_pos,
                cuctx.violates_lfnst,
                any_ts,
                constrained
            );
        }
        if (!cuctx.lfnst_last_scan_pos && plan.isp == predict::IspMode::None)
            || constrained
            || any_ts
        {
            plan.lfnst = 0;
        } else {
            let cctx = usize::from(plan.tree != TreeType::Joint);
            let first = dec.decode_bin(&mut ctx.models[ctx::LFNST_IDX + cctx]);
            plan.lfnst = if first {
                1 + u8::from(dec.decode_bin(&mut ctx.models[ctx::LFNST_IDX + 2]))
            } else {
                0
            };
        }
    }
    let mts_allowed = g.sps.mts_enabled
        && g.sps.explicit_mts_intra
        && luma_cu
        && plan.isp == predict::IspMode::None
        && plan.bdpcm_l == 0
        && w <= 32
        && h <= 32;
    let first_ts_skip = tus.iter().find(|t| t.component == 0).is_some_and(|t| t.ts);
    if mts_allowed
        && !cuctx.violates_mts
        && cuctx.mts_last_scan_pos
        && plan.lfnst == 0
        && !first_ts_skip
    {
        let mut v = 0u8;
        if dec.decode_bin(&mut ctx.models[ctx::MTS_IDX]) {
            v = 1;
            for i in 0..3usize {
                if !dec.decode_bin(&mut ctx.models[ctx::MTS_IDX + 1 + i]) {
                    break;
                }
                v += 1;
            }
        }
        plan.mts = v;
    }
}

fn reconstruct_tus(g: &mut Gen<'_>, plan: &CuPlan, tus: &[TuData]) -> Result<(), VvcError> {
    let mut i = 0usize;
    while i < tus.len() {
        let t = &tus[i];
        if t.component == 1
            && t.joint != 0
            && tus
                .get(i + 1)
                .is_some_and(|n| n.component == 2 && n.joint == t.joint && n.x == t.x && n.y == t.y)
        {
            reconstruct_joint_cbcr(g, plan, t, &tus[i + 1])?;
            i += 2;
            continue;
        }
        reconstruct_tu(g, plan, t)?;
        i += 1;
    }
    Ok(())
}

/// Inverse Joint CbCr (ICT) residual reconstruction of one chroma transform
/// block pair (H.266 §8.7.4.2 / §7.4.12.7, VTM `TU::getICTMode` and
/// `TrQuant::invTransformICT`).
fn reconstruct_joint_cbcr(
    g: &mut Gen<'_>,
    plan: &CuPlan,
    cb: &TuData,
    cr: &TuData,
) -> Result<(), VvcError> {
    let cw = cb.w / 2;
    let ch = cb.h / 2;
    if cw == 0 || ch == 0 {
        return Ok(());
    }
    let ix = cb.x / 2;
    let iy = cb.y / 2;
    let mut pred_cb = predict_chroma(g, plan, 1, ix, iy, cw, ch, plan.chroma_mode);
    let mut pred_cr = predict_chroma(g, plan, 2, ix, iy, cw, ch, plan.chroma_mode);
    // H.266 §8.7.2 (1135): TuCResMode == 2 uses Qp′CbCr for both components.
    let (qp_cb, qp_cr) = if cb.joint == 3 {
        (g.qp_joint, g.qp_joint)
    } else {
        (plan.qp_cb, plan.qp_cr)
    };
    let mut res_cb = residual_of(plan, cb, false, cw, ch, qp_cb, g.bit_depth)?;
    let mut res_cr = residual_of(plan, cr, false, cw, ch, qp_cr, g.bit_depth)?;
    let mode = if g.ph.joint_cbcr_sign {
        match cb.joint {
            1 => -3,
            2 => -1,
            3 => -2,
            _ => 0,
        }
    } else {
        match cb.joint {
            1 => 3,
            2 => 1,
            3 => 2,
            _ => 0,
        }
    };
    for i in 0..(cw * ch) {
        match mode {
            3 => res_cb[i] = res_cr[i] >> 1,
            -3 => res_cb[i] = -res_cr[i] >> 1,
            1 => res_cr[i] = res_cb[i] >> 1,
            -1 => res_cr[i] = -res_cb[i] >> 1,
            2 => res_cr[i] = res_cb[i],
            -2 => res_cr[i] = -res_cb[i],
            _ => {}
        }
    }
    if g.lmcs_on && g.ph.chroma_residual_scale && (cb.cbf || cr.cbf) && cw * ch > 4 {
        // VTM `DecCu.cpp:289-292`: with `tu.jointCbCr != 0` both chroma
        // residuals are scaled, including the component derived by ICT whose
        // own CBF may be zero.
        let avg = chroma_adj_luma(g, cb);
        if let Some(r) = &g.lmcs {
            r.scale_chroma_residual(&mut res_cb, avg);
            r.scale_chroma_residual(&mut res_cr, avg);
        }
    }
    let max_val = g.max_val();
    for y in 0..ch {
        for x in 0..cw {
            let v = pred_cb[y * cw + x] + res_cb[y * cw + x];
            g.u.set(ix + x, iy + y, v.clamp(0, max_val) as u16);
            let v = pred_cr[y * cw + x] + res_cr[y * cw + x];
            g.v.set(ix + x, iy + y, v.clamp(0, max_val) as u16);
        }
    }
    mark_tu(g, cb);
    mark_tu(g, cr);
    pred_cb.clear();
    pred_cr.clear();
    Ok(())
}

fn reconstruct_tu(g: &mut Gen<'_>, plan: &CuPlan, t: &TuData) -> Result<(), VvcError> {
    let luma = t.component == 0;
    let (cw, ch) = if luma { (t.w, t.h) } else { (t.w / 2, t.h / 2) };
    if cw == 0 || ch == 0 {
        return Ok(());
    }
    let ix = if luma { t.x } else { t.x / 2 };
    let iy = if luma { t.y } else { t.y / 2 };
    if std::env::var_os("VVC_TRACE").is_some() && t.component == 0 && t.y == 0 && t.x < 12 {
        eprintln!(
            "rt tx={} ty={} tw={} th={} ix={} iy={} cbf={}",
            t.x, t.y, t.w, t.h, ix, iy, t.cbf
        );
    }
    let mode = if luma { plan.mode } else { plan.chroma_mode };
    let pred = if luma {
        if let Some((transposed, mip_idx)) = plan.mip {
            let refs = g.gather_avail(true, ix, iy, cw, ch);
            predict::predict_mip_with_bit_depth(
                &refs,
                cw,
                ch,
                mip_idx as u8,
                transposed,
                g.bit_depth,
            )
        } else if plan.isp == predict::IspMode::Ver && (plan.w == 4 || (plan.w == 8 && plan.h > 4))
        {
            // VTM `CU::isPredRegDiffFromTB` / `adjustPredArea`:
            // an ISP vertical sub-partition narrower than 4 luma samples shares
            // one 4-wide prediction region. The region is predicted once and
            // each sub-partition takes its columns from that prediction.
            let rx = plan.x + ((ix - plan.x) / 4) * 4;
            let region_w = 4;
            let full = predict_luma(g, plan, rx, iy, region_w, ch, mode);
            let mut out = vec![0i32; cw * ch];
            for yy in 0..ch {
                for xx in 0..cw {
                    out[yy * cw + xx] = full[yy * region_w + (ix - rx) + xx];
                }
            }
            out
        } else {
            predict_luma(g, plan, ix, iy, cw, ch, mode)
        }
    } else {
        predict_chroma(g, plan, t.component, ix, iy, cw, ch, mode)
    };
    if std::env::var_os("VVC_TRACE").is_some()
        && t.component == 0
        && iy == 0
        && (8..12).contains(&ix)
    {
        let l = if ix > 0 { g.y.get(ix - 1, 0) } else { 0 };
        eprintln!(
            "predx ix={} mode={} isp={:?} pred={:?} left={}",
            ix,
            plan.mode,
            plan.isp,
            &pred[..pred.len().min(3)],
            l
        );
    }
    let _ = luma;
    let qp = if t.component == 0 {
        plan.qp_y
    } else if t.component == 1 {
        plan.qp_cb
    } else {
        plan.qp_cr
    };
    let mut residual = residual_of(plan, t, t.component == 0, cw, ch, qp, g.bit_depth)?;
    if t.component == 0 && plan.bdpcm_l != 0 {
        residual = predict::apply_bdpcm_inverse(&residual, cw, ch, plan.bdpcm_l == 2);
    }
    if t.component != 0 && plan.bdpcm_c != 0 {
        residual = predict::apply_bdpcm_inverse(&residual, cw, ch, plan.bdpcm_c == 2);
    }
    if std::env::var_os("VVC_TRACE").is_some()
        && t.component == 0
        && iy == 0
        && (10..12).contains(&ix)
    {
        eprintln!(
            "resx ix={} pred={:?} res={:?}",
            ix,
            &pred[..3],
            &residual[..residual.len().min(3)]
        );
    }

    if t.component != 0 && g.lmcs_on && g.ph.chroma_residual_scale && t.cbf && cw * ch > 4 {
        let avg = chroma_adj_luma(g, t);
        if let Some(r) = &g.lmcs {
            r.scale_chroma_residual(&mut residual, avg);
        }
    }
    let max_val = g.max_val();
    let plane = if t.component == 0 {
        &mut g.y
    } else if t.component == 1 {
        &mut g.u
    } else {
        &mut g.v
    };
    for y in 0..ch {
        for x in 0..cw {
            let v = i32::from(pred[y * cw + x]) + residual[y * cw + x];
            plane.set(ix + x, iy + y, v.clamp(0, max_val) as u16);
        }
    }
    mark_tu(g, t);
    Ok(())
}

/// Dequantised residual samples of one transform block, handling transform
/// skip without the inverse transform (`TrQuant::xITransformSkip`).
fn residual_of(
    plan: &CuPlan,
    t: &TuData,
    luma: bool,
    cw: usize,
    ch: usize,
    qp: i32,
    bit_depth: u8,
) -> Result<Vec<i32>, VvcError> {
    if !t.cbf {
        return Ok(vec![0i32; cw * ch]);
    }
    if t.ts {
        // VTM `TrQuant::xITransformSkip` dequantises with `Qp′` (the slice QP
        // plus `QpBdOffset`, H.266 Eq. 29).
        let qp_scaled = (qp + dct::qp_bd_offset(bit_depth)).clamp(0, 63 + 6 * 4) as u8;
        return Ok(t
            .levels
            .iter()
            .map(|&l| transform::dequantize_ts(l, qp_scaled))
            .collect());
    }
    inverse_residual(plan, t, luma, cw, ch, qp, bit_depth)
}

fn inverse_residual(
    plan: &CuPlan,
    t: &TuData,
    luma: bool,
    cw: usize,
    ch: usize,
    qp: i32,
    bit_depth: u8,
) -> Result<Vec<i32>, VvcError> {
    let scan = CoeffScan::new(cw, ch);
    // `Qp′ = Qp + QpBdOffset` (H.266 Eq. 29); the dequantisers expect the
    // scaled value and apply `qp_bd_offset` internally only for the transform
    // shift, not for the QP itself (`Quant.cpp:64-103`).
    let qp = (qp + dct::qp_bd_offset(bit_depth)).clamp(0, 63 + 6 * 4) as u8;
    // `t.levels` is already raster-indexed: `residual::decode_residual_*`
    // writes each coefficient at `scan.raster[scan_pos]`.
    let full = t.levels.clone();
    if std::env::var_os("VVC_TRACE").is_some() && cw == 1 {
        eprintln!(
            "deq lv={:?} full={:?}",
            &t.levels[..t.levels.len().min(4)],
            &full[..full.len().min(4)]
        );
    }
    let used_mts = luma && plan.mts != 0;
    let dequant = if g_dep_quant(plan) {
        dct::dequantize_dep_quant_bd(cw, ch, &full, &scan.raster, qp, bit_depth).map_err(|_e| {
            VvcError::Unsupported("unsupported transform dimensions or transform type")
        })?
    } else {
        dct::dequantize_rect_bd(cw, ch, &full, qp, bit_depth).map_err(|_e| {
            VvcError::Unsupported("unsupported transform dimensions or transform type")
        })?
    };
    let mut dequant = dequant;
    if std::env::var_os("VVC_TRACE").is_some() && cw == 1 {
        eprintln!("deq out={:?}", &dequant[..dequant.len().min(4)]);
    }

    if plan.lfnst > 0 && !used_mts && (luma || plan.tree == TreeType::Chroma) {
        let lfnst_intra_mode = lfnst_intra_mode(plan, luma, cw, ch);
        lfnst::inverse_lfnst_rect(cw, ch, &mut dequant, plan.lfnst, lfnst_intra_mode).map_err(
            |_e| VvcError::Unsupported("unsupported transform dimensions or transform type"),
        )?;
    }
    let (th, tv) = if used_mts {
        mts_types(plan.mts)?
    } else if luma && plan.lfnst == 0 && plan.mip.is_none() && plan.implicit_mts {
        // H.266 §8.7.4.1 / VTM `TrQuant::getTrTypes`: with `implicitMtsEnabled`
        // (ISP, or no explicit intra MTS) a luma dimension in `4..=16` uses
        // DST-7 and DCT-2 otherwise. With `sps_mts_enabled_flag == 0`, or for
        // chroma, both stay DCT-2.
        (
            if (4..=16).contains(&cw) {
                dct::TrType::Dst7
            } else {
                dct::TrType::Dct2
            },
            if (4..=16).contains(&ch) {
                dct::TrType::Dst7
            } else {
                dct::TrType::Dct2
            },
        )
    } else {
        (dct::TrType::Dct2, dct::TrType::Dct2)
    };
    let (sw, sh) = if plan.lfnst > 0 && !used_mts {
        lfnst_skips(cw, ch)
    } else {
        primary_skips(cw, ch, th, tv)
    };
    dct::inverse_transform_rect_bd(cw, ch, &dequant, th, tv, sw, sh, bit_depth)
        .map_err(|_e| VvcError::Unsupported("unsupported transform dimensions or transform type"))
}

fn g_dep_quant(plan: &CuPlan) -> bool {
    plan.dep_quant
}

fn mts_types(idx: u8) -> Result<(dct::TrType, dct::TrType), VvcError> {
    match idx {
        0 => Ok((dct::TrType::Dct2, dct::TrType::Dct2)),
        1 => Ok((dct::TrType::Dst7, dct::TrType::Dst7)),
        2 => Ok((dct::TrType::Dct8, dct::TrType::Dst7)),
        3 => Ok((dct::TrType::Dst7, dct::TrType::Dct8)),
        4 => Ok((dct::TrType::Dct8, dct::TrType::Dct8)),
        _ => Err(VvcError::Bitstream(format!("invalid mts_idx {idx}"))),
    }
}

fn primary_skips(w: usize, h: usize, th: dct::TrType, tv: dct::TrType) -> (usize, usize) {
    let sw = if w == 32 && th != dct::TrType::Dct2 {
        16
    } else {
        w.saturating_sub(32)
    };
    let sh = if h == 32 && tv != dct::TrType::Dct2 {
        16
    } else {
        h.saturating_sub(32)
    };
    (sw, sh)
}

/// Zero-out width/height of the primary transform for a block with LFNST
/// (H.266 §8.7.4.1, Eq. 1169/1170): `nonZeroW` and `nonZeroH` are `4` when either
/// dimension is `4` and `8` otherwise.
fn lfnst_skips(w: usize, h: usize) -> (usize, usize) {
    let nz = if w == 4 || h == 4 { 4 } else { 8 };
    (w - nz.min(w), h - nz.min(h))
}

impl Gen<'_> {
    fn decoded_at(&self, is_luma: bool, x: usize, y: usize) -> bool {
        if is_luma {
            if x >= self.pic_w || y >= self.pic_h {
                return false;
            }
            self.dec_y[(y / 4) * self.lgrid_w + x / 4]
        } else {
            if x >= self.cgrid_w || y >= self.cgrid_h {
                return false;
            }
            self.dec_c[y * self.cgrid_w + x]
        }
    }

    /// Collect `top_len` above and `left_len` left reference samples with the
    /// availability substitution of H.266 §8.4.5.2.8–9 (VTM
    /// `IntraPrediction::xFillReferenceSamples`).
    ///
    /// `top_len`/`left_len` may exceed `2 * w`/`2 * h` for an ISP
    /// sub-partition, where VTM keeps `nCbW + nTbW` above and
    /// `nCbH + nTbH` left samples (`IntraPrediction.cpp:814-829`).
    fn gather_len(
        &self,
        plane: &Plane16,
        is_luma: bool,
        x: usize,
        y: usize,
        top_len: usize,
        left_len: usize,
        dc: i32,
    ) -> predict::RefSamples {
        let pw = plane.width;
        let ph = plane.height;
        let mut above_avail = vec![false; top_len];
        let mut left_avail = vec![false; left_len];
        let mut any = false;
        if y > 0 {
            for k in 0..top_len {
                if x + k < pw && self.decoded_at(is_luma, x + k, y - 1) {
                    above_avail[k] = true;
                    any = true;
                }
            }
        }
        if x > 0 {
            for k in 0..left_len {
                if y + k < ph && self.decoded_at(is_luma, x - 1, y + k) {
                    left_avail[k] = true;
                    any = true;
                }
            }
        }
        let tl_avail = x > 0 && y > 0 && self.decoded_at(is_luma, x - 1, y - 1);
        if tl_avail {
            any = true;
        }

        let first_top = above_avail
            .iter()
            .position(|&a| a)
            .map(|k| i32::from(plane.get(x + k, y - 1)));
        let first_left = left_avail
            .iter()
            .position(|&a| a)
            .map(|k| i32::from(plane.get(x - 1, y + k)));

        let mut above = vec![dc; top_len];
        let mut left = vec![dc; left_len];
        let mut prev: Option<i32> = None;
        for k in 0..top_len {
            if above_avail[k] {
                let v = i32::from(plane.get(x + k, y - 1));
                above[k] = v;
                prev = Some(v);
            } else {
                above[k] = prev.or(first_top).or(first_left).unwrap_or(dc);
            }
        }
        prev = None;
        for k in 0..left_len {
            if left_avail[k] {
                let v = i32::from(plane.get(x - 1, y + k));
                left[k] = v;
                prev = Some(v);
            } else {
                left[k] = prev.or(first_left).or(first_top).unwrap_or(dc);
            }
        }
        let top_left = if tl_avail {
            i32::from(plane.get(x - 1, y - 1))
        } else {
            first_top.or(first_left).unwrap_or(dc)
        };
        if !any {
            above.fill(dc);
            left.fill(dc);
        }
        predict::RefSamples {
            above,
            left,
            top_left,
        }
    }

    fn gather_avail(
        &self,
        is_luma: bool,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> predict::RefSamples {
        let plane = if is_luma { &self.y } else { &self.u };
        self.gather_len(plane, is_luma, x, y, 2 * w, 2 * h, self.dc())
    }

    fn gather_chroma(
        &self,
        plane: &Plane16,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> predict::RefSamples {
        self.gather_len(plane, false, x, y, 2 * w, 2 * h, self.dc())
    }

    /// Cross-component linear model prediction for one chroma transform block
    /// (H.266 §8.4.5.2.14; VTM `IntraPrediction::predIntraChromaLM` /
    /// `xGetLMParameters`). Implements the 4:2:0 down-sampling of (368)/(369),
    /// the CTU-boundary top template of (371)–(373), the left template of
    /// (376)/(377), adaptive template selection (359)–(364), the min/max
    /// group parameter derivation (378)–(404), and the per-sample application.
    fn cclm_predict(
        &self,
        comp: usize,
        mode: predict::CclmMode,
        xc: usize,
        yc: usize,
        w: usize,
        h: usize,
    ) -> Vec<i32> {
        let mid = self.dc();
        let max_val = self.max_val();
        let chroma = if comp == 1 { &self.u } else { &self.v };
        let luma = &self.y;
        let xtb = xc * 2;
        let ytb = yc * 2;
        let avail_l = xc > 0 && self.decoded_at(false, xc - 1, yc);
        let avail_t = yc > 0 && self.decoded_at(false, xc, yc - 1);
        let ctu_h = self.ctu;
        let b_ctu_boundary = ctu_h > 0 && (ytb & (ctu_h - 1)) == 0;
        let vert_collocated = self.sps.chroma_vertical_collocated;

        // `pY[x][y]` with the availability padding of §8.4.5.2.14 step 2. The
        // negative/negative corner is never sampled by formulas (368)–(377).
        let py = |x: isize, y: isize| -> i32 {
            let y = if y < 0 && !avail_t { 0 } else { y };
            let x = if x < 0 && !avail_l { 0 } else { x };
            i32::from(luma.get((xtb as isize + x) as usize, (ytb as isize + y) as usize))
        };

        // numTopRight / numLeftBelow (INTRA_T_CCLM / INTRA_L_CCLM only).
        let mut num_top_right = 0usize;
        if mode == predict::CclmMode::LmT && avail_t {
            for x in w..(2 * w) {
                if !self.decoded_at(false, xc + x, yc - 1) {
                    break;
                }
                num_top_right += 1;
            }
        }
        let mut num_left_below = 0usize;
        if mode == predict::CclmMode::LmL && avail_l {
            for y in h..(2 * h) {
                if !self.decoded_at(false, xc - 1, yc + y) {
                    break;
                }
                num_left_below += 1;
            }
        }

        let (num_t, num_l) = match mode {
            predict::CclmMode::Lm => (if avail_t { w } else { 0 }, if avail_l { h } else { 0 }),
            predict::CclmMode::LmT => (if avail_t { w + num_top_right.min(h) } else { 0 }, 0),
            predict::CclmMode::LmL => (
                0,
                if avail_l {
                    h + num_left_below.min(w)
                } else {
                    0
                },
            ),
        };
        let num_is4n = usize::from(!(avail_t && avail_l && mode == predict::CclmMode::Lm));

        let mut sel_luma = [mid; 4];
        let mut sel_chroma = [mid; 4];
        let mut count = 0usize;

        let top_ds = |x: usize| -> i32 {
            let lx = 2 * x as isize;
            if b_ctu_boundary {
                (py(lx - 1, -1) + 2 * py(lx, -1) + py(lx + 1, -1) + 2) >> 2
            } else if vert_collocated {
                (py(lx, -3) + py(lx - 1, -2) + 4 * py(lx, -2) + py(lx + 1, -2) + py(lx, -1) + 4)
                    >> 3
            } else {
                (py(lx - 1, -1)
                    + py(lx - 1, -2)
                    + 2 * py(lx, -1)
                    + 2 * py(lx, -2)
                    + py(lx + 1, -1)
                    + py(lx + 1, -2)
                    + 4)
                    >> 3
            }
        };
        let left_ds = |y: usize| -> i32 {
            let ly = 2 * y as isize;
            if vert_collocated {
                (py(-2, ly - 1) + py(-3, ly) + 4 * py(-2, ly) + py(-1, ly) + py(-2, ly + 1) + 4)
                    >> 3
            } else {
                (py(-3, ly)
                    + py(-3, ly + 1)
                    + 2 * py(-2, ly)
                    + 2 * py(-2, ly + 1)
                    + py(-1, ly)
                    + py(-1, ly + 1)
                    + 4)
                    >> 3
            }
        };

        if num_t > 0 {
            let start = num_t >> (2 + num_is4n);
            let step = 1.max(num_t >> (1 + num_is4n));
            let cnt = num_t.min((1 + num_is4n) << 1);
            for pos in 0..cnt {
                let x = start + pos * step;
                if x >= num_t {
                    break;
                }
                sel_luma[count] = top_ds(x);
                sel_chroma[count] = i32::from(chroma.get(xc + x, yc - 1));
                count += 1;
            }
        }
        if num_l > 0 {
            let start = num_l >> (2 + num_is4n);
            let step = 1.max(num_l >> (1 + num_is4n));
            let cnt = num_l.min((1 + num_is4n) << 1);
            for pos in 0..cnt {
                let y = start + pos * step;
                if y >= num_l {
                    break;
                }
                sel_luma[count] = left_ds(y);
                sel_chroma[count] = i32::from(chroma.get(xc - 1, yc + y));
                count += 1;
            }
        }

        if count == 0 {
            return vec![mid; w * h];
        }
        let params =
            predict::derive_cclm_params_from_4_samples(sel_luma, sel_chroma, count, self.bit_depth);

        let mut out = vec![0i32; w * h];
        for y in 0..h {
            let ly = 2 * y as isize;
            for x in 0..w {
                let lx = 2 * x as isize;
                let ds = if vert_collocated {
                    (py(lx, ly - 1)
                        + py(lx - 1, ly)
                        + 4 * py(lx, ly)
                        + py(lx + 1, ly)
                        + py(lx, ly + 1)
                        + 4)
                        >> 3
                } else {
                    (py(lx - 1, ly)
                        + py(lx - 1, ly + 1)
                        + 2 * py(lx, ly)
                        + 2 * py(lx, ly + 1)
                        + py(lx + 1, ly)
                        + py(lx + 1, ly + 1)
                        + 4)
                        >> 3
                };
                let pred = ((ds * params.a) >> params.shift) + params.b;
                out[y * w + x] = pred.clamp(0, max_val);
            }
        }
        out
    }

    /// MRL reference lines (`intra_luma_ref_idx > 0`), with the same
    /// availability substitution as [`Self::gather_len`] applied to the
    /// selected reference line.
    fn gather_mrl(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mrl: usize,
    ) -> predict::MrlRefSamples {
        let plane = &self.y;
        let dc = self.dc();
        let offset = 1 + mrl;
        let above_len = 2 * w;
        let left_len = 2 * h;
        let mut row_above = vec![dc; above_len + offset + 4];
        let mut row_left = vec![dc; left_len + offset + 4];

        let tl_avail = x >= offset && y >= offset && self.decoded_at(true, x - offset, y - offset);
        if tl_avail {
            row_above[0] = i32::from(plane.get(x - offset, y - offset));
            row_left[0] = row_above[0];
            for i in 1..offset {
                row_above[i] = i32::from(plane.get(x - offset + i, y - offset));
                row_left[i] = i32::from(plane.get(x - offset, y - offset + i));
            }
        }

        let mut top_avail = vec![false; above_len];
        let mut left_avail = vec![false; left_len];
        if y >= offset {
            for k in 0..above_len {
                if x + k < plane.width && self.decoded_at(true, x + k, y - offset) {
                    top_avail[k] = true;
                }
            }
        }
        if x >= offset {
            for k in 0..left_len {
                if y + k < plane.height && self.decoded_at(true, x - offset, y + k) {
                    left_avail[k] = true;
                }
            }
        }
        let first_top = top_avail
            .iter()
            .position(|&a| a)
            .map(|k| i32::from(plane.get(x + k, y - offset)));
        let first_left = left_avail
            .iter()
            .position(|&a| a)
            .map(|k| i32::from(plane.get(x - offset, y + k)));

        let mut prev: Option<i32> = None;
        for k in 0..above_len {
            if top_avail[k] {
                let v = i32::from(plane.get(x + k, y - offset));
                row_above[offset + k] = v;
                prev = Some(v);
            } else {
                row_above[offset + k] = prev.or(first_top).or(first_left).unwrap_or(dc);
            }
        }
        for k in (offset + above_len)..row_above.len() {
            row_above[k] = row_above[offset + above_len - 1];
        }
        prev = None;
        for k in 0..left_len {
            if left_avail[k] {
                let v = i32::from(plane.get(x - offset, y + k));
                row_left[offset + k] = v;
                prev = Some(v);
            } else {
                row_left[offset + k] = prev.or(first_left).or(first_top).unwrap_or(dc);
            }
        }
        for k in (offset + left_len)..row_left.len() {
            row_left[k] = row_left[offset + left_len - 1];
        }
        if !tl_avail {
            let fallback = first_top.or(first_left).unwrap_or(dc);
            for i in 0..offset {
                row_above[i] = fallback;
                row_left[i] = fallback;
            }
        }
        predict::MrlRefSamples {
            mrl_idx: mrl,
            row_above,
            row_left,
        }
    }
}

fn isp_refs(
    g: &Gen<'_>,
    plan: &CuPlan,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> predict::RefSamples {
    // VTM `IntraPrediction::initIntraPatternChTypeISP` keeps
    // `nCbW + nTbW` above and `nCbH + nTbH` left samples for an ISP
    // sub-partition. The references themselves are the reconstructed samples
    // around the sub-partition, which for later sub-partitions include the
    // previously decoded part of the same CU.
    g.gather_len(&g.y, true, x, y, plan.w + w, plan.h + h, g.dc())
}

fn predict_luma(
    g: &Gen<'_>,
    plan: &CuPlan,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: u8,
) -> Vec<i32> {
    if !plan.intra {
        let mut out = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                out.push(i32::from(g.pred_y.get(x + i, y + j)));
            }
        }
        return out;
    }
    if plan.bdpcm_l != 0 {
        let refs = g.gather_avail(true, x, y, w, h);
        return predict::predict_bdpcm(&refs, w, h, plan.bdpcm_l == 2);
    }
    let isp = plan.isp != predict::IspMode::None;
    let mrl = usize::from(plan.mrl).min(2);
    let mrl_refs = if isp {
        predict::MrlRefSamples::from_ref_samples(&isp_refs(g, plan, x, y, w, h))
    } else if mrl == 0 {
        predict::MrlRefSamples::from_ref_samples(&g.gather_avail(true, x, y, w, h))
    } else {
        g.gather_mrl(x, y, w, h, mrl)
    };

    let off = mrl_refs.mrl_idx;
    let base_refs = predict::RefSamples {
        above: mrl_refs.row_above[off + 1..off + 1 + 2 * w].to_vec(),
        left: mrl_refs.row_left[off + 1..off + 1 + 2 * h].to_vec(),
        top_left: mrl_refs.row_above[off],
    };
    match intra_mode(mode, false, false) {
        IntraMode::Planar => {
            let used = if mrl == 0
                && !isp
                && predict::uses_reference_filter(true, IntraMode::Planar, w, h)
            {
                base_refs.filtered()
            } else {
                base_refs.clone()
            };
            let mut p = predict::predict_planar(&used, w, h);
            if mrl == 0 && w >= 4 && h >= 4 {
                predict::apply_pdpc(&mut p, &used, w, h);
            }
            p
        }
        IntraMode::Dc => {
            let used =
                if mrl == 0 && !isp && predict::uses_reference_filter(true, IntraMode::Dc, w, h) {
                    base_refs.filtered()
                } else {
                    base_refs.clone()
                };
            let mut p = predict::predict_dc(&used, w, h);
            if mrl == 0 && w >= 4 && h >= 4 {
                predict::apply_pdpc(&mut p, &used, w, h);
            }
            p
        }
        IntraMode::Angular(dir) => {
            let params = predict::init_angular_params(dir, w, h, plan.w, plan.h, true, mrl, isp);
            if mrl == 0 && !isp {
                let used = if params.ref_filter_flag {
                    base_refs.filtered()
                } else {
                    base_refs.clone()
                };
                let m = predict::MrlRefSamples::from_ref_samples(&used);
                predict::predict_angular_with_params_bd(&m, w, h, true, &params, g.bit_depth)
            } else {
                predict::predict_angular_with_params_bd(&mrl_refs, w, h, true, &params, g.bit_depth)
            }
        }
    }
}

fn predict_chroma(
    g: &Gen<'_>,
    _plan: &CuPlan,
    component: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: u8,
) -> Vec<i32> {
    if !_plan.intra {
        let src = if component == 1 { &g.pred_u } else { &g.pred_v };
        let mut out = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                out.push(i32::from(src.get(x + i, y + j)));
            }
        }
        return out;
    }
    let plane = if component == 1 { &g.u } else { &g.v };
    if (81..=83).contains(&mode) {
        let cclm = match mode {
            81 => predict::CclmMode::Lm,
            82 => predict::CclmMode::LmL,
            _ => predict::CclmMode::LmT,
        };
        return g.cclm_predict(component, cclm, x, y, w, h);
    }
    let refs = g.gather_chroma(plane, x, y, w, h);
    match intra_mode(mode, false, true) {
        IntraMode::Planar => {
            let used = if predict::uses_reference_filter(false, IntraMode::Planar, w, h) {
                refs.filtered()
            } else {
                refs.clone()
            };
            let mut p = predict::predict_planar(&used, w, h);
            if w >= 4 && h >= 4 {
                predict::apply_pdpc(&mut p, &used, w, h);
            }
            p
        }
        IntraMode::Dc => {
            let used = if predict::uses_reference_filter(false, IntraMode::Dc, w, h) {
                refs.filtered()
            } else {
                refs.clone()
            };
            let mut p = predict::predict_dc(&used, w, h);
            if w >= 4 && h >= 4 {
                predict::apply_pdpc(&mut p, &used, w, h);
            }
            p
        }
        IntraMode::Angular(dir) => {
            predict::predict_angular_bd(&refs, w, h, false, dir, g.bit_depth)
        }
    }
}

fn intra_mode(mode: u8, _mip: bool, _chroma: bool) -> IntraMode {
    match mode {
        0 => IntraMode::Planar,
        1 => IntraMode::Dc,
        m => IntraMode::Angular(m.min(66)),
    }
}

fn chroma_adj_luma(g: &mut Gen<'_>, t: &TuData) -> i32 {
    let vpdu = if g.ctu == 128 { 64 } else { g.ctu };
    let vx = (t.x / vpdu) * vpdu;
    let vy = (t.y / vpdu) * vpdu;
    if let Some(&avg) = g.vpdu_scale.get(&(vx, vy)) {
        return avg;
    }
    let n = 64usize.min(vpdu);
    let mut sum = 0i32;
    let mut count = 0usize;
    if vx > 0 {
        for i in 0..n {
            let y = (vy + i).min(g.pic_h - 1);
            sum += i32::from(g.y.get(vx - 1, y));
            count += 1;
        }
    }
    if vy > 0 {
        for i in 0..n {
            let x = (vx + i).min(g.pic_w - 1);
            sum += i32::from(g.y.get(x, vy - 1));
            count += 1;
        }
    }
    let avg = match count {
        0 => g.dc(),
        c if c == n => (sum + (1 << (n.trailing_zeros() - 1))) >> n.trailing_zeros(),
        c if c == 2 * n => (sum + (1 << n.trailing_zeros())) >> (n.trailing_zeros() + 1),
        _ => g.dc(),
    };
    g.vpdu_scale.insert((vx, vy), avg);
    avg
}

fn mark_tu(g: &mut Gen<'_>, t: &TuData) {
    let id = g.next_tu_id;
    g.next_tu_id += 1;
    let (map, wmap, hmap, mw, mh, x, y, w, h) = if t.component == 0 {
        (
            &mut g.tu_y,
            &mut g.tu_wy,
            &mut g.tu_hy,
            g.tu_y_w,
            g.tu_y_h,
            t.x,
            t.y,
            t.w,
            t.h,
        )
    } else if t.component == 1 {
        (
            &mut g.tu_u,
            &mut g.tu_wc,
            &mut g.tu_hc,
            g.tu_c_w,
            g.tu_c_h,
            t.x / 2,
            t.y / 2,
            t.w / 2,
            t.h / 2,
        )
    } else {
        (
            &mut g.tu_v,
            &mut g.tu_wc,
            &mut g.tu_hc,
            g.tu_c_w,
            g.tu_c_h,
            t.x / 2,
            t.y / 2,
            t.w / 2,
            t.h / 2,
        )
    };
    // Chroma deblocking uses a 2-sample grid (4 luma samples at 4:2:0); the
    // luma grid is 4 samples (`LOG_GRID_SIZE = 2`, VTM `DeblockingFilter.h:55`).
    let cell_size = if t.component == 0 { 4 } else { 2 };
    for yy in y..(y + h).min(mh * cell_size) {
        for xx in x..(x + w).min(mw * cell_size) {
            let cell = (yy / cell_size) * mw + xx / cell_size;
            if cell < map.len() {
                map[cell] = id;
                wmap[cell] = w as u16;
                hmap[cell] = h as u16;
            }
        }
    }
    if t.component == 0 {
        for yy in (y / 4)..((y + h).div_ceil(4)).min(g.tu_y_h) {
            for xx in (x / 4)..((x + w).div_ceil(4)).min(g.tu_y_w) {
                g.tu_cbf_y[yy * g.tu_y_w + xx] = t.cbf;
            }
        }
        for yy in y..(y + h).min(g.pic_h) {
            for xx in x..(x + w).min(g.pic_w) {
                g.dec_y[(yy / 4) * g.lgrid_w + xx / 4] = true;
            }
        }
    } else {
        for yy in y..(y + h).min(g.cgrid_h) {
            for xx in x..(x + w).min(g.cgrid_w) {
                g.dec_c[yy * g.cgrid_w + xx] = true;
            }
        }
    }
}

/// `maxFilterLengthP`/`maxFilterLengthQ` of a luma block edge from the transform
/// block heights or widths on both sides (H.266 §8.8.3.3, Eq. 1227 ff.).
///
/// A side of at most four samples forces *both* lengths to `1`; only then is the
/// per-side `7`/`3` selection applied. Deriving the two sides independently would
/// admit a long filter against a 4-sample transform block.
fn luma_lens(size_p: u16, size_q: u16) -> (usize, usize) {
    if size_p <= 4 || size_q <= 4 {
        return (1, 1);
    }
    let side = |v: u16| if v >= 32 { 7 } else { 3 };
    (side(size_p), side(size_q))
}

/// `maxFilterLengthP`/`maxFilterLengthQ` of a chroma block edge (H.266 §8.8.3.3).
///
/// Both sides must span at least eight chroma samples for the strong filter; on a
/// horizontal edge that coincides with the upper chroma CTB boundary the pair is
/// `1`/`3` (Eq. 1417-1420), otherwise `3`/`3`. Any smaller side forces both to `1`.
fn chroma_lens(size_p: u16, size_q: u16, ctb_boundary: bool) -> (usize, usize) {
    if size_p >= 8 && size_q >= 8 {
        if ctb_boundary { (1, 3) } else { (3, 3) }
    } else {
        (1, 1)
    }
}

/// Inter/intra boundary strength for one 4x4 luma edge
/// (`DeblockingFilter.cpp`, H.266 §8.8.3).
fn deblock_bs(g: &Gen<'_>, x: usize, y: usize, vertical: bool, is_tu_edge: bool) -> u8 {
    let (p, q) = if vertical {
        (g.lcell(x - 1, y), g.lcell(x, y))
    } else {
        (g.lcell(x, y - 1), g.lcell(x, y))
    };
    let p_intra = p.is_some_and(|c| c.intra);
    let q_intra = q.is_some_and(|c| c.intra);
    if p_intra || q_intra {
        return 2;
    }
    let field = |c: LumaCell| -> (super::inter::Mv, i32) {
        for l in 0..2 {
            if c.ref_idx[l] >= 0 {
                return (c.mv[l], g.ref_poc(l, c.ref_idx[l]).unwrap_or(0));
            }
        }
        (super::inter::Mv::default(), 0)
    };
    let (pmv, pref) = p.map_or((super::inter::Mv::default(), 0), field);
    let (qmv, qref) = q.map_or((super::inter::Mv::default(), 0), field);
    let cbf = |cx: usize, cy: usize| {
        if cx / 4 < g.lgrid_w && cy / 4 < g.tu_y_h {
            g.tu_cbf_y[(cy / 4) * g.lgrid_w + cx / 4]
        } else {
            false
        }
    };
    let (pcbf, qcbf) = if vertical {
        (cbf(x - 1, y), cbf(x, y))
    } else {
        (cbf(x, y - 1), cbf(x, y))
    };
    filter::derive_boundary_strength(
        p_intra, q_intra, pcbf, qcbf, is_tu_edge, pmv, qmv, pref, qref,
    )
}

fn apply_deblock(g: &mut Gen<'_>) {
    let max_val = g.max_val();
    let bd = g.bit_depth;
    let qp_y = i32::from(g.qp_y.clamp(0, 63)) as u8;
    let beta = g.sh.luma_beta_offset_div2 as i8;
    let tc = g.sh.luma_tc_offset_div2 as i8;
    let w = g.vis_w;
    let h = g.vis_h;
    // Luma vertical edges.
    let mut x = 4usize;
    while x < w {
        let mut y = 0usize;
        while y < h {
            let lc = ((y / 4) * g.tu_y_w) + (x / 4 - 1);
            let rc = (y / 4) * g.tu_y_w + x / 4;
            if g.tu_y[lc] != u32::MAX && g.tu_y[rc] != u32::MAX && g.tu_y[lc] != g.tu_y[rc] {
                let (lp, lq) = luma_lens(g.tu_wy[lc], g.tu_wy[rc]);
                let mut lines = [[0i32; 16]; 4];
                for (i, line) in lines.iter_mut().enumerate() {
                    let yy = (y + i).min(h - 1);
                    for (k, v) in line.iter_mut().enumerate() {
                        let xx = (x as isize - 8 + k as isize).clamp(0, w as isize - 1) as usize;
                        *v = i32::from(g.y.get(xx, yy));
                    }
                }
                filter::deblock_luma_edge_normative_bd(
                    &mut lines,
                    qp_y,
                    qp_y,
                    deblock_bs(g, x, y, true, true),
                    beta,
                    tc,
                    lp,
                    lq,
                    false,
                    bd,
                );
                for (i, line) in lines.iter().enumerate() {
                    let yy = (y + i).min(h - 1);
                    for k in 0..16 {
                        let xx = (x as isize - 8 + k as isize).clamp(0, w as isize - 1) as usize;
                        g.y.set(xx, yy, line[k].clamp(0, max_val) as u16);
                    }
                }
            }
            y += 4;
        }
        x += 4;
    }
    // Luma horizontal edges.
    let mut y = 4usize;
    while y < h {
        let mut x = 0usize;
        while x < w {
            let tc_cell = ((y / 4 - 1) * g.tu_y_w) + x / 4;
            let bc = (y / 4) * g.tu_y_w + x / 4;
            if g.tu_y[tc_cell] != u32::MAX
                && g.tu_y[bc] != u32::MAX
                && g.tu_y[tc_cell] != g.tu_y[bc]
            {
                let (lp, lq) = luma_lens(g.tu_hy[tc_cell], g.tu_hy[bc]);
                let mut lines = [[0i32; 16]; 4];
                for (i, line) in lines.iter_mut().enumerate() {
                    let xx = (x + i).min(w - 1);
                    for (k, v) in line.iter_mut().enumerate() {
                        let yy = (y as isize - 8 + k as isize).clamp(0, h as isize - 1) as usize;
                        *v = i32::from(g.y.get(xx, yy));
                    }
                }
                filter::deblock_luma_edge_normative_bd(
                    &mut lines,
                    qp_y,
                    qp_y,
                    deblock_bs(g, x, y, false, true),
                    beta,
                    tc,
                    lp,
                    lq,
                    y.is_multiple_of(g.ctu),
                    bd,
                );
                for (i, line) in lines.iter().enumerate() {
                    let xx = (x + i).min(w - 1);
                    for k in 0..16 {
                        let yy = (y as isize - 8 + k as isize).clamp(0, h as isize - 1) as usize;
                        g.y.set(xx, yy, line[k].clamp(0, max_val) as u16);
                    }
                }
            }
            x += 4;
        }
        y += 4;
    }
    // Chroma.
    let cvis_w = g.vis_w.div_ceil(2);
    let cvis_h = g.vis_h.div_ceil(2);
    for comp in 1..3usize {
        let qp = if comp == 1 { g.qp_cb } else { g.qp_cr };
        let qp = qp.clamp(0, 63) as u8;
        let (beta_c, tc_c) = if comp == 1 {
            (g.sh.cb_beta_offset_div2 as i8, g.sh.cb_tc_offset_div2 as i8)
        } else {
            (g.sh.cr_beta_offset_div2 as i8, g.sh.cr_tc_offset_div2 as i8)
        };
        let (plane, wmap, hmap, cmap, cw, ch, tuw, tuh) = if comp == 1 {
            (
                &mut g.u, &g.tu_wc, &g.tu_hc, &g.tu_u, cvis_w, cvis_h, g.tu_c_w, g.tu_c_h,
            )
        } else {
            (
                &mut g.v, &g.tu_wc, &g.tu_hc, &g.tu_v, cvis_w, cvis_h, g.tu_c_w, g.tu_c_h,
            )
        };
        // Chroma edges live on a 2-sample grid (4 luma samples at 4:2:0) and
        // each decision group spans two samples along the edge
        // (VTM `xEdgeFilterChroma`, `loopLength = pelsInPartChromaV = 2`).
        let mut x = 2usize;
        while x < cw {
            let mut y = 0usize;
            // VTM `xEdgeFilterChroma` skips chroma edges that are not on the
            // 8-sample chroma grid when the minimum chroma part is smaller than
            // `GRID_SIZE_CHR` (`DeblockingFilter.cpp:1221-1229`).
            if (2 * x) % 16 != 0 {
                x += 2;
                continue;
            }
            while y + 2 <= ch {
                let lc = (y / 2) * tuw + (x / 2 - 1);
                let rc = (y / 2) * tuw + x / 2;
                if cmap[lc] != u32::MAX && cmap[rc] != u32::MAX && cmap[lc] != cmap[rc] {
                    let (lp, lq) = chroma_lens(wmap[lc], wmap[rc], false);
                    let mut lines = [[0i32; 8]; 2];
                    for (i, line) in lines.iter_mut().enumerate() {
                        let yy = (y + i).min(ch - 1);
                        for (k, v) in line.iter_mut().enumerate() {
                            let xx =
                                (x as isize - 4 + k as isize).clamp(0, cw as isize - 1) as usize;
                            *v = i32::from(plane.get(xx, yy));
                        }
                    }
                    filter::deblock_chroma_edge(
                        &mut lines,
                        i32::from(qp),
                        2,
                        beta_c,
                        tc_c,
                        lp,
                        lq,
                        bd,
                    );
                    for (i, line) in lines.iter().enumerate() {
                        let yy = (y + i).min(ch - 1);
                        for k in 0..8 {
                            let xx =
                                (x as isize - 4 + k as isize).clamp(0, cw as isize - 1) as usize;
                            plane.set(xx, yy, line[k].clamp(0, max_val) as u16);
                        }
                    }
                }
                y += 2;
            }
            x += 2;
        }
        let mut y = 2usize;
        while y < ch {
            if (2 * y) % 16 != 0 {
                y += 2;
                continue;
            }
            let mut x = 0usize;
            while x + 2 <= cw {
                let tc_cell = ((y / 2 - 1) * tuw) + x / 2;
                let bc = (y / 2) * tuw + x / 2;
                if cmap[tc_cell] != u32::MAX && cmap[bc] != u32::MAX && cmap[tc_cell] != cmap[bc] {
                    let (lp, lq) =
                        chroma_lens(hmap[tc_cell], hmap[bc], y.is_multiple_of(g.ctu / 2));
                    let mut lines = [[0i32; 8]; 2];
                    for (i, line) in lines.iter_mut().enumerate() {
                        let xx = (x + i).min(cw - 1);
                        for (k, v) in line.iter_mut().enumerate() {
                            let yy =
                                (y as isize - 4 + k as isize).clamp(0, ch as isize - 1) as usize;
                            *v = i32::from(plane.get(xx, yy));
                        }
                    }
                    filter::deblock_chroma_edge(
                        &mut lines,
                        i32::from(qp),
                        2,
                        beta_c,
                        tc_c,
                        lp,
                        lq,
                        bd,
                    );
                    for (i, line) in lines.iter().enumerate() {
                        let xx = (x + i).min(cw - 1);
                        for k in 0..8 {
                            let yy =
                                (y as isize - 4 + k as isize).clamp(0, ch as isize - 1) as usize;
                            plane.set(xx, yy, line[k].clamp(0, max_val) as u16);
                        }
                    }
                }
                x += 2;
            }
            y += 2;
        }
        let _ = tuh;
    }
}

fn apply_sao(g: &mut Gen<'_>) {
    let ctu = g.ctu;
    let src_y = g.y.clone();
    let src_u = g.u.clone();
    let src_v = g.v.clone();
    let cvis_w = g.vis_w.div_ceil(2);
    let cvis_h = g.vis_h.div_ceil(2);
    for gy in 0..g.vis_h.div_ceil(ctu) {
        for gx in 0..g.vis_w.div_ceil(ctu) {
            let idx = gy * g.ctu_cols + gx;
            let sao = g.sao[idx];
            let x0 = gx * ctu;
            let y0 = gy * ctu;
            let w = ctu.min(g.vis_w - x0);
            let h = ctu.min(g.vis_h - y0);
            filter::apply_sao_region_bounded_16(
                &src_y, &mut g.y, x0, y0, w, h, g.vis_w, g.vis_h, sao[0],
            );
            if g.sh.sao_chroma_used {
                let cx0 = x0 / 2;
                let cy0 = y0 / 2;
                let cw_here = (ctu / 2).min(cvis_w.saturating_sub(cx0));
                let ch_here = (ctu / 2).min(cvis_h.saturating_sub(cy0));
                if cw_here > 0 && ch_here > 0 {
                    filter::apply_sao_region_bounded_16(
                        &src_u, &mut g.u, cx0, cy0, cw_here, ch_here, cvis_w, cvis_h, sao[1],
                    );
                    filter::apply_sao_region_bounded_16(
                        &src_v, &mut g.v, cx0, cy0, cw_here, ch_here, cvis_w, cvis_h, sao[2],
                    );
                }
            }
        }
    }
}

fn apply_alf(g: &mut Gen<'_>) {
    let src_y = g.y.clone();
    let src_u = g.u.clone();
    let src_v = g.v.clone();
    let ctu = g.ctu;
    let cvis_w = g.vis_w.div_ceil(2);
    let cvis_h = g.vis_h.div_ceil(2);
    for gy in 0..g.vis_h.div_ceil(ctu) {
        for gx in 0..g.vis_w.div_ceil(ctu) {
            let idx = gy * g.ctu_cols + gx;
            let x0 = gx * ctu;
            let y0 = gy * ctu;
            let w = ctu.min(g.vis_w - x0);
            let h = ctu.min(g.vis_h - y0);
            if let Some(mode) = g.alf_luma[idx] {
                let aps =
                    g.sh.alf_aps_ids_luma
                        .first()
                        .and_then(|&id| g.aps.alf[usize::from(id & 7)].as_ref());
                alf_luma_block(
                    &src_y,
                    &mut g.y,
                    x0,
                    y0,
                    w,
                    h,
                    g.vis_w,
                    g.vis_h,
                    ctu,
                    ctu.saturating_sub(4),
                    mode,
                    aps,
                    g.bit_depth,
                );
            }
            for comp in 1..3usize {
                let alt = if comp == 1 {
                    g.alf_cb[idx]
                } else {
                    g.alf_cr[idx]
                };
                if let Some(alt) = alt {
                    let aps_idx = usize::from(g.sh.alf_aps_id_chroma & 7);
                    if let Some(aps) = g.aps.alf[aps_idx].as_ref() {
                        let (src, dst) = if comp == 1 {
                            (&src_u, &mut g.u)
                        } else {
                            (&src_v, &mut g.v)
                        };
                        alf_chroma_block(
                            src,
                            dst,
                            aps,
                            alt,
                            x0 / 2,
                            y0 / 2,
                            (w / 2).max(1),
                            (h / 2).max(1),
                            cvis_w,
                            cvis_h,
                            ctu / 2,
                            (ctu / 2).saturating_sub(2),
                            g.bit_depth,
                        );
                    }
                }
            }
            // CCALF: apply after this CTU's chroma ALF, using the pre-ALF luma
            // (`AdaptiveLoopFilter.cpp:505-521`).
            for comp in 1..3usize {
                let idc = if comp == 1 {
                    g.ccalf_cb[idx]
                } else {
                    g.ccalf_cr[idx]
                };
                if idc == 0 {
                    continue;
                }
                let aps_id = if comp == 1 {
                    g.sh.cc_alf_cb_aps_id
                } else {
                    g.sh.cc_alf_cr_aps_id
                };
                let Some(aps_id) = aps_id else { continue };
                let Some(aps) = g.aps.alf[usize::from(aps_id & 7)].as_ref() else {
                    continue;
                };
                let filters = if comp == 1 {
                    &aps.cc_cb_filters
                } else {
                    &aps.cc_cr_filters
                };
                let Some(coeffs) = filters.get(usize::from(idc) - 1) else {
                    continue;
                };
                let (chroma, cw, ch) = if comp == 1 {
                    (&mut g.u, cvis_w, cvis_h)
                } else {
                    (&mut g.v, cvis_w, cvis_h)
                };
                ccalf_block(
                    &src_y,
                    chroma,
                    coeffs,
                    x0 / 2,
                    y0 / 2,
                    (w / 2).max(1),
                    (h / 2).max(1),
                    cw,
                    ch,
                    g.vis_w,
                    g.vis_h,
                    ctu,
                    g.bit_depth,
                );
            }
        }
    }
}

/// Cross-component ALF for one chroma CTU (`AdaptiveLoopFilter.cpp:1333-1448`,
/// `filterBlkCcAlf`). The luma tap positions are fixed for `4:2:0`.
#[allow(clippy::too_many_arguments)]
fn ccalf_block(
    src_y: &Plane16,
    chroma: &mut Plane16,
    coeffs: &[i16; 7],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    vis_lw: usize,
    vis_lh: usize,
    ctu_l: usize,
    bit_depth: u8,
) {
    let bd = bit_depth.clamp(8, 12);
    let max_val = chroma.max_sample();
    let offset = 1i32 << (bd - 1);
    let lw = vis_lw.min(src_y.width) as isize;
    let lh = vis_lh.min(src_y.height) as isize;
    let get = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, lw - 1) as usize;
        let cy = y.clamp(0, lh - 1) as usize;
        i32::from(src_y.get(cx, cy))
    };
    let vb_pos = ctu_l.saturating_sub(4);
    let c = [
        i32::from(coeffs[0]),
        i32::from(coeffs[1]),
        i32::from(coeffs[2]),
        i32::from(coeffs[3]),
        i32::from(coeffs[4]),
        i32::from(coeffs[5]),
        i32::from(coeffs[6]),
    ];
    for y in y0..(y0 + h).min(ch) {
        let ly = (y * 2) as isize;
        let pos = (ly as usize) & (ctu_l - 1);
        let (above, below, below2) = if pos == vb_pos.saturating_sub(2) || pos == vb_pos + 1 {
            (ly - 1, ly + 1, ly + 1)
        } else if pos == vb_pos.saturating_sub(1) || pos == vb_pos {
            (ly, ly, ly)
        } else {
            (ly - 1, ly + 1, ly + 2)
        };
        for x in x0..(x0 + w).min(cw) {
            let lx = (x * 2) as isize;
            let cur = get(lx, ly);
            let mut sum = c[0] * (get(lx, above) - cur);
            sum += c[1] * (get(lx - 1, ly) - cur);
            sum += c[2] * (get(lx + 1, ly) - cur);
            sum += c[3] * (get(lx - 1, below) - cur);
            sum += c[4] * (get(lx, below) - cur);
            sum += c[5] * (get(lx + 1, below) - cur);
            sum += c[6] * (get(lx, below2) - cur);
            let v = ((sum + 64) >> 7).clamp(-offset, offset - 1);
            let out = i32::from(chroma.get(x, y)) + v;
            chroma.set(x, y, out.clamp(0, max_val) as u16);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn alf_luma_block(
    src: &Plane16,
    dst: &mut Plane16,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
    vb_h: usize,
    vb_pos: usize,
    mode: AlfLumaMode,
    aps: Option<&super::hls::VvcAlfAps>,
    bit_depth: u8,
) {
    let bd = bit_depth.clamp(8, 12);
    let max_val = dst.max_sample();
    let clip_table = if bd >= 12 {
        filter::ALF_CLIP_VALS_12BIT
    } else if bd >= 10 {
        filter::ALF_CLIP_VALS_10BIT
    } else {
        filter::ALF_CLIP_VALS_8BIT
    };
    let bw = vis_w.min(src.width) as isize;
    let bh = vis_h.min(src.height) as isize;
    let get_src = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, bw - 1) as usize;
        let cy = y.clamp(0, bh - 1) as usize;
        i32::from(src.get(cx, cy))
    };
    let mut by = y0;
    while by < y0 + h {
        let mut bx = x0;
        while bx < x0 + w {
            let (class_idx, transpose) = filter::derive_alf_class_4x4_vb_16(
                src,
                bx,
                by,
                bw as usize,
                bh as usize,
                vb_h,
                vb_pos,
            );
            let (coeffs, clips): ([i16; 12], [u8; 12]) = match mode {
                AlfLumaMode::Fixed(set) => {
                    let fi = usize::from(
                        filter::ALF_CLASS_TO_FILTER_MAPPING[set.min(15)][class_idx.min(24)],
                    );
                    let mut c = [0i16; 12];
                    c.copy_from_slice(&filter::ALF_FIXED_FILTER_COEFF[fi][..12]);
                    (c, [0u8; 12])
                }
                AlfLumaMode::Aps(k) => {
                    let aps = aps.expect("ALF APS");
                    let fi = usize::from(aps.filter_coeff_delta_idx[class_idx.min(24)])
                        .min(aps.luma_filters.len().saturating_sub(1));
                    let c = aps.luma_filters[fi];
                    let mut cl = [0u8; 12];
                    if aps.luma_clip {
                        if let Some(x) = aps.luma_clip_idx.get(fi) {
                            cl = *x;
                        }
                    }
                    let _ = k;
                    (c, cl)
                }
            };
            let perm = &filter::ALF_TRANSPOSE_PERM[transpose & 3];
            let mut fc = [0i32; 12];
            let mut fclip = [0i32; 12];
            for k in 0..12 {
                fc[k] = i32::from(coeffs[perm[k]]);
                fclip[k] = clip_table[usize::from(clips[perm[k]] & 3)];
            }
            // Tap coordinates: `(column offset, index into the seven row
            // pointers)` where row 0 is the current row, 1/+1, 2/-1, 3/+2,
            // 4/-2, 5/+3, 6/-3 (`AdaptiveLoopFilter.cpp:1230-1243`).
            const TERMS: [(isize, usize, isize, usize); 12] = [
                (0, 5, 0, 6),
                (1, 3, -1, 4),
                (0, 3, 0, 4),
                (-1, 3, 1, 4),
                (2, 1, -2, 2),
                (1, 1, -1, 2),
                (0, 1, 0, 2),
                (-1, 1, 1, 2),
                (-2, 1, 2, 2),
                (3, 0, -3, 0),
                (2, 0, -2, 0),
                (1, 0, -1, 0),
            ];
            let xe = (bx + 4).min(x0 + w);
            let ye = (by + 4).min(y0 + h);
            for y in by..ye {
                let iy = y as isize;
                for x in bx..xe {
                    let ix = x as isize;
                    let cur = get_src(ix, iy);
                    // ALF virtual boundary at `vb_pos` samples from the top of a
                    // `vb_h`-high CTU row (`AdaptiveLoopFilter.cpp:1251-1272`).
                    let yv = if vb_h > 0 { y % vb_h } else { usize::MAX };
                    let mut rows = [iy, iy + 1, iy - 1, iy + 2, iy - 2, iy + 3, iy - 3];
                    if vb_h > 0 && yv < vb_pos && yv + 4 >= vb_pos {
                        if yv == vb_pos - 1 {
                            rows[1] = rows[0];
                            rows[2] = rows[0];
                        }
                        if yv + 2 >= vb_pos {
                            rows[3] = rows[1];
                            rows[4] = rows[2];
                        }
                        if yv + 3 >= vb_pos {
                            rows[5] = rows[3];
                            rows[6] = rows[4];
                        }
                    } else if vb_h > 0 && yv >= vb_pos && yv <= vb_pos + 3 {
                        if yv == vb_pos {
                            rows[2] = rows[0];
                            rows[1] = rows[0];
                        }
                        if yv <= vb_pos + 1 {
                            rows[4] = rows[2];
                            rows[3] = rows[1];
                        }
                        if yv <= vb_pos + 2 {
                            rows[6] = rows[4];
                            rows[5] = rows[3];
                        }
                    }
                    let near = if vb_h > 0 {
                        (yv < vb_pos && yv + 1 >= vb_pos) || yv == vb_pos
                    } else {
                        false
                    };
                    let mut sum = 0i32;
                    for (k, &(cax, ra, cbx, rb)) in TERMS.iter().enumerate() {
                        let a = get_src(ix + cax, rows[ra]);
                        let b = get_src(ix + cbx, rows[rb]);
                        // VTM `AdaptiveLoopFilter.h:72-75` (`clipALF`): both taps are
                        // clipped around the current sample, not around each other.
                        let v = (a - cur).clamp(-fclip[k], fclip[k])
                            + (b - cur).clamp(-fclip[k], fclip[k]);
                        sum += fc[k] * v;
                    }
                    let shift = if near { 10 } else { 7 };
                    let val = cur + ((sum + (1 << (shift - 1))) >> shift);
                    dst.set(x, y, val.clamp(0, max_val) as u16);
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

#[allow(clippy::too_many_arguments)]
fn alf_chroma_block(
    src: &Plane16,
    dst: &mut Plane16,
    aps: &super::hls::VvcAlfAps,
    alt: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
    vb_h: usize,
    vb_pos: usize,
    bit_depth: u8,
) {
    if aps.chroma_filters.is_empty() {
        return;
    }
    let idx = alt.min(aps.chroma_filters.len() - 1);
    let c = aps.chroma_filters[idx];
    let cl = aps.chroma_clip_idx.get(idx).copied().unwrap_or([0u8; 6]);
    let bd = bit_depth.clamp(8, 12);
    let max_val = dst.max_sample();
    let table = if bd >= 12 {
        filter::ALF_CLIP_VALS_12BIT
    } else if bd >= 10 {
        filter::ALF_CLIP_VALS_10BIT
    } else {
        filter::ALF_CLIP_VALS_8BIT
    };
    let bw = vis_w.min(src.width) as isize;
    let bh = vis_h.min(src.height) as isize;
    let get = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, bw - 1) as usize;
        let cy = y.clamp(0, bh - 1) as usize;
        i32::from(src.get(cx, cy))
    };
    // `(column offset, row pointer)` with rows 0 = current, 1/+1, 2/-1,
    // 3/+2, 4/-2 (`AdaptiveLoopFilter.cpp:1276-1289`).
    const TERMS: [(isize, usize, isize, usize); 6] = [
        (0, 3, 0, 4),
        (1, 1, -1, 2),
        (0, 1, 0, 2),
        (-1, 1, 1, 2),
        (2, 0, -2, 0),
        (1, 0, -1, 0),
    ];
    for y in y0..(y0 + h).min(dst.height) {
        for x in x0..(x0 + w).min(dst.width) {
            let (ix, iy) = (x as isize, y as isize);
            let cur = get(ix, iy);
            // Chroma ALF virtual boundary at two samples above each CTU-row
            // boundary (`AdaptiveLoopFilter.cpp:1251-1272`, chroma variant).
            let yv = if vb_h > 0 { y % vb_h } else { usize::MAX };
            let mut rows = [iy, iy + 1, iy - 1, iy + 2, iy - 2];
            if vb_h > 0 && yv < vb_pos && yv + 2 >= vb_pos {
                if yv + 1 == vb_pos {
                    rows[1] = rows[0];
                    rows[2] = rows[0];
                }
                if yv + 2 >= vb_pos {
                    rows[3] = rows[1];
                    rows[4] = rows[2];
                }
            } else if vb_h > 0 && yv >= vb_pos && yv <= vb_pos + 1 {
                if yv == vb_pos {
                    rows[2] = rows[0];
                    rows[1] = rows[0];
                }
                if yv <= vb_pos + 1 {
                    rows[4] = rows[2];
                    rows[3] = rows[1];
                }
            }
            let near = if vb_h > 0 {
                (yv < vb_pos && yv + 1 >= vb_pos) || yv == vb_pos
            } else {
                false
            };
            let mut sum = 0i32;
            for (k, &(cax, ra, cbx, rb)) in TERMS.iter().enumerate() {
                let a = get(ix + cax, rows[ra]);
                let b = get(ix + cbx, rows[rb]);
                let clk = table[usize::from(cl[k] & 3)];
                let v = (a - cur).clamp(-clk, clk) + (b - cur).clamp(-clk, clk);
                sum += i32::from(c[k]) * v;
            }
            let shift = if near { 10 } else { 7 };
            let val = cur + ((sum + (1 << (shift - 1))) >> shift);
            dst.set(x, y, val.clamp(0, max_val) as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(mode: u8, isp: predict::IspMode, w: usize, h: usize) -> CuPlan {
        CuPlan {
            tree: TreeType::Luma,
            x: 0,
            w,
            h,
            qp_y: 26,
            qp_cb: 26,
            qp_cr: 26,
            mode,
            mip: None,
            isp,
            mrl: 0,
            bdpcm_l: 0,
            bdpcm_c: 0,
            chroma_mode: 0,
            lfnst_mode_c: 0,
            lfnst: 1,
            mts: 0,
            implicit_mts: false,
            dep_quant: false,
            sbt: 0,
            intra: true,
        }
    }

    /// H.266 §8.7.4.1 derives the LFNST intra mode from the §8.4.5.2.7
    /// wide-angle mapping, applied to the coding block for a luma ISP
    /// sub-partition (Eq. 320/321) and to the transform block otherwise.
    #[test]
    fn lfnst_mode_uses_the_wide_angle_coding_block_for_isp() {
        // 16x4 coding block, mode 6 -> 71 -> extended index 85: LFNST set 1,
        // transposed.
        let isp = plan(6, predict::IspMode::Ver, 16, 4);
        let idx = lfnst_intra_mode(&isp, true, 4, 4);
        assert_eq!(idx, 85);
        assert_eq!(lfnst::lfnst_set(idx), Ok(1));
        assert!(lfnst::transposes(idx));

        // The same mode without ISP maps 6 -> 71 through the 4x4 block size,
        // where the mapping does not apply: extended index 6, set 1, not
        // transposed.
        let plain = plan(6, predict::IspMode::None, 16, 4);
        let idx = lfnst_intra_mode(&plain, true, 4, 4);
        assert_eq!(idx, 6);
        assert_eq!(lfnst::lfnst_set(idx), Ok(1));
        assert!(!lfnst::transposes(idx));
    }

    /// H.266 §8.8.3.3: a luma transform side of at most four samples forces both
    /// maximum filter lengths to `1`; the `7`/`3` selection is per side otherwise.
    #[test]
    fn luma_max_filter_lengths_are_derived_pairwise() {
        assert_eq!(luma_lens(4, 32), (1, 1));
        assert_eq!(luma_lens(32, 4), (1, 1));
        assert_eq!(luma_lens(8, 8), (3, 3));
        assert_eq!(luma_lens(32, 8), (7, 3));
        assert_eq!(luma_lens(8, 64), (3, 7));
        assert_eq!(luma_lens(16, 16), (3, 3));
    }

    /// H.266 §8.8.3.3: the chroma pair is `3`/`3` only when both sides span at
    /// least eight samples, and `1`/`3` on the upper chroma CTB row boundary.
    #[test]
    fn chroma_max_filter_lengths_are_derived_pairwise() {
        assert_eq!(chroma_lens(8, 8, false), (3, 3));
        assert_eq!(chroma_lens(16, 8, false), (3, 3));
        assert_eq!(chroma_lens(8, 8, true), (1, 3));
        assert_eq!(chroma_lens(4, 16, false), (1, 1));
        assert_eq!(chroma_lens(16, 4, true), (1, 1));
    }

    /// H.266 §8.7.4.1, Eq. 1169/1170: the primary transform only uses the
    /// `nonZeroW x nonZeroH` low-frequency region when LFNST applies.
    #[test]
    fn lfnst_zeroes_the_primary_transform_outside_non_zero_region() {
        assert_eq!(lfnst_skips(4, 4), (0, 0));
        assert_eq!(lfnst_skips(8, 8), (0, 0));
        assert_eq!(lfnst_skips(16, 8), (8, 0));
        assert_eq!(lfnst_skips(8, 4), (4, 0));
        assert_eq!(lfnst_skips(4, 16), (0, 12));
        assert_eq!(lfnst_skips(16, 16), (8, 8));
        assert_eq!(lfnst_skips(64, 64), (56, 56));
    }
}
