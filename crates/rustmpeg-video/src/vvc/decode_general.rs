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
use super::filter::LmcsReshaper;
use super::hls::{VvcActiveApsBank, VvcPictureHeader, VvcPps, VvcSliceHeader, VvcSps};
use super::predict::{self, IntraMode, Plane};
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

fn limits(sps: &VvcSps, chroma: bool) -> Lim {
    let min_cb = 1usize << sps.min_cb_log2;
    let min_qt = if chroma {
        min_cb << sps.log2_diff_min_qt_min_cb_intra_chroma
    } else {
        min_cb << sps.log2_diff_min_qt_min_cb_intra_luma
    };
    let (bt_diff, tt_diff, max_mtt) = if chroma {
        (
            sps.log2_diff_max_bt_min_qt_intra_chroma,
            sps.log2_diff_max_tt_min_qt_intra_chroma,
            sps.max_mtt_depth_intra_chroma,
        )
    } else {
        (
            sps.log2_diff_max_bt_min_qt_intra_luma,
            sps.log2_diff_max_tt_min_qt_intra_luma,
            sps.max_mtt_depth_intra_luma,
        )
    };
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
    lgrid: Vec<LumaCell>,
    lgrid_w: usize,
    cgrid: Vec<ChromaCell>,
    cgrid_w: usize,
    cgrid_h: usize,
    y: Plane,
    u: Plane,
    v: Plane,
    qp_y: i32,
    qp_cb: i32,
    qp_cr: i32,
    /// `Qp′CbCr` (`H.266 §8.7.2`, used when `TuCResMode == 2`).
    qp_joint: i32,
    lmcs: Option<LmcsReshaper>,
    lmcs_on: bool,
    sao: Vec<[filter::SaoType; 3]>,
    ctu_cols: usize,
    alf_luma: Vec<Option<AlfLumaMode>>,
    alf_cb: Vec<Option<usize>>,
    alf_cr: Vec<Option<usize>>,
    /// Luma transform-block boundaries: per 4x4 cell id, u32::MAX = none.
    tu_y: Vec<u32>,
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
    fn lcell(&self, x: usize, y: usize) -> Option<LumaCell> {
        if x >= self.pic_w || y >= self.pic_h {
            return None;
        }
        let c = self.lgrid[(y / 4) * self.lgrid_w + x / 4];
        c.valid.then_some(c)
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
    lfnst_mode_c: u16,
    lfnst: u8,
    mts: u8,
    implicit_mts: bool,
    dep_quant: bool,
}

pub(crate) struct GeneralPicture {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_general_intra_picture(
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
) -> Result<GeneralPicture, VvcError> {
    if sps.info.chroma_format_idc != 1 || sps.info.bit_depth_luma != 8 {
        return Err(VvcError::Unsupported(
            "general coding-tree decoding is implemented for 8-bit 4:2:0",
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
            .map(LmcsReshaper::from_aps_8bit)
    } else {
        None
    };
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
        lgrid: vec![LumaCell::default(); lgrid_w * lgrid_h],
        lgrid_w,
        cgrid: vec![ChromaCell::default(); cgrid_w * cgrid_h],
        cgrid_w,
        cgrid_h,
        y: Plane::new(pic_w, pic_h),
        u: Plane::new(cgrid_w, cgrid_h),
        v: Plane::new(cgrid_w, cgrid_h),
        qp_y,
        qp_cb,
        qp_cr,
        qp_joint,
        lmcs,
        lmcs_on,
        sao: vec![[filter::SaoType::Off; 3]; ctu_cols * ctu_rows],
        ctu_cols,
        alf_luma: vec![None; ctu_cols * ctu_rows],
        alf_cb: vec![None; ctu_cols * ctu_rows],
        alf_cr: vec![None; ctu_cols * ctu_rows],
        tu_y_w: lgrid_w,
        tu_y_h: lgrid_h,
        tu_c_w: cgrid_w.div_ceil(4),
        tu_c_h: cgrid_h.div_ceil(4),
        tu_y: vec![u32::MAX; lgrid_w * lgrid_h],
        tu_u: vec![u32::MAX; cgrid_w.div_ceil(4) * cgrid_h.div_ceil(4)],
        tu_v: vec![u32::MAX; cgrid_w.div_ceil(4) * cgrid_h.div_ceil(4)],
        tu_wy: vec![0; lgrid_w * lgrid_h],
        tu_hy: vec![0; lgrid_w * lgrid_h],
        tu_wc: vec![0; cgrid_w.div_ceil(4) * cgrid_h.div_ceil(4)],
        tu_hc: vec![0; cgrid_w.div_ceil(4) * cgrid_h.div_ceil(4)],
        dec_y: vec![false; lgrid_w * lgrid_h],
        dec_c: vec![false; cgrid_w * cgrid_h],
        next_tu_id: 0,
        vpdu_scale: std::collections::HashMap::new(),
    };

    let mut dec = BinDecoder::new(cabac).map_err(|e| VvcError::Bitstream(e.to_string()))?;
    let mut ctx = CtxStore::new_i_slice(qp_y);
    let dual = sps.dual_tree_intra && sh.slice_type == super::hls::VvcSliceType::I;

    for ctu_y in (0..pic_h).step_by(ctu) {
        for ctu_x in (0..pic_w).step_by(ctu) {
            let ctu_idx = (ctu_y / ctu) * ctu_cols + ctu_x / ctu;
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
                series: [0; 8],
                series_len: 0,
            };
            if dual {
                if ctu > 64 {
                    let lim = limits(sps, false);
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
            r.apply_inverse_plane(&mut g.y);
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
    Ok(GeneralPicture {
        y: g.y,
        u: g.u,
        v: g.v,
    })
}

fn coding_tree(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    let lim = limits(g.sps, node.tree == TreeType::Chroma);
    let dual = g.sps.dual_tree_intra;
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
    let mut next = node;
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
        8,
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

fn decode_cu_luma(
    g: &mut Gen<'_>,
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    node: Node,
) -> Result<(), VvcError> {
    let w = node.w;
    let h = node.h;
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
fn chroma_lfnst_mode(luma_mode: u8, chroma_mode: u8) -> u16 {
    if (81..=83).contains(&chroma_mode) {
        u16::from(luma_mode)
    } else {
        u16::from(chroma_mode)
    }
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
) -> Result<(), VvcError> {
    let lim = limits(g.sps, chroma_tree);
    let (w, h) = if chroma_tree {
        (node.w / 2, node.h / 2)
    } else {
        (node.w, node.h)
    };
    let can_max_split = w > lim.max_tb || h > lim.max_tb;
    let isp_split = tr_depth == 0 && !chroma_tree && plan.isp != predict::IspMode::None;
    if can_max_split || isp_split {
        let parts: Vec<(usize, usize, usize, usize)> = if can_max_split {
            max_tu_tiles(w, h, lim.max_tb)
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
        tr_depth,
        sub_tu,
        tus,
        cuctx,
        isp_prev,
        isp_any_cbf,
        chroma_tree,
    )
}

fn max_tu_tiles(w: usize, h: usize, max_tb: usize) -> Vec<(usize, usize, usize, usize)> {
    let nx = (w / max_tb).max(1);
    let ny = (h / max_tb).max(1);
    let tw = w / nx;
    let th = h / ny;
    let mut out = Vec::new();
    fn rec(
        x: usize,
        y: usize,
        nx: usize,
        ny: usize,
        tw: usize,
        th: usize,
        out: &mut Vec<(usize, usize, usize, usize)>,
    ) {
        if nx == 1 && ny == 1 {
            out.push((x, y, tw, th));
            return;
        }
        if nx >= ny {
            rec(x, y, nx / 2, ny, tw, th, out);
            rec(x + (nx / 2) * tw, y, nx - nx / 2, ny, tw, th, out);
        } else {
            rec(x, y, nx, ny / 2, tw, th, out);
            rec(x, y + (ny / 2) * th, nx, ny - ny / 2, tw, th, out);
        }
    }
    // z-order for a power-of-two grid
    rec(0, 0, nx, ny, tw, th, &mut out);
    out
}

fn transform_unit(
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
) -> Result<(), VvcError> {
    let (lw, lh) = (node.w, node.h);
    let isp = !chroma_tree && plan.isp != predict::IspMode::None;
    let joint = plan.tree == TreeType::Joint && !chroma_tree;
    let mut cbf = [false; 3];
    let mut jccr = 0u8;
    if chroma_tree || (joint && !isp) {
        let c0 = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CB]);
        let c1 = dec.decode_bin(&mut ctx.models[ctx::QT_CBF_CR + usize::from(c0)]);
        cbf[1] = c0;
        cbf[2] = c1;
        if g.sps.joint_cbcr_enabled && (c0 || c1) {
            let mask = u32::from(c0) * 2 + u32::from(c1);
            let on = dec.decode_bin(&mut ctx.models[ctx::JOINT_CB_CR_FLAG + (mask - 1) as usize]);
            jccr = if on { mask as u8 } else { 0 };
        }
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
        cbf[0] = if isp && last && !any_cbf {
            true
        } else {
            let c = ctx::QT_CBF_Y + if isp { 2 + usize::from(prev) } else { 0 };
            dec.decode_bin(&mut ctx.models[c])
        };
        *isp_prev = cbf[0];
        *isp_any_cbf = any_cbf || cbf[0];
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
            let ts_allowed = g.sps.transform_skip_enabled
                && plan.bdpcm_l == 0
                && plan.bdpcm_c == 0
                && (!isp || comp != 0)
                && cw <= (1usize << g.sps.log2_transform_skip_max_size)
                && ch <= (1usize << g.sps.log2_transform_skip_max_size);
            ts = ts_allowed
                && dec
                    .decode_bin(&mut ctx.models[ctx::TRANSFORM_SKIP_FLAG + usize::from(comp != 0)]);
            let cw_coded = cw.min(32);
            let ch_coded = ch.min(32);
            let scan = CoeffScan::new(cw_coded, ch_coded);
            levels = vec![0i32; scan.max_num_coeff];
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
    let height = scan.max_num_coeff / scan.width;
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
    let max_tb = limits(g.sps, chroma_tree).max_tb;
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
    let mut res_cb = residual_of(plan, cb, false, cw, ch, qp_cb)?;
    let mut res_cr = residual_of(plan, cr, false, cw, ch, qp_cr)?;
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
        let avg_cb = if cb.cbf {
            Some(chroma_adj_luma(g, cb))
        } else {
            None
        };
        let avg_cr = if cr.cbf {
            Some(chroma_adj_luma(g, cr))
        } else {
            None
        };
        if let Some(r) = &g.lmcs {
            if let Some(avg) = avg_cb {
                r.scale_chroma_residual(&mut res_cb, avg);
            }
            if let Some(avg) = avg_cr {
                r.scale_chroma_residual(&mut res_cr, avg);
            }
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let v = pred_cb[y * cw + x] + res_cb[y * cw + x];
            g.u.set(ix + x, iy + y, v.clamp(0, 255) as u8);
            let v = pred_cr[y * cw + x] + res_cr[y * cw + x];
            g.v.set(ix + x, iy + y, v.clamp(0, 255) as u8);
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
            predict::predict_mip(&refs, cw, ch, mip_idx as u8, transposed)
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
    let mut residual = residual_of(plan, t, t.component == 0, cw, ch, qp)?;
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
            plane.set(ix + x, iy + y, v.clamp(0, 255) as u8);
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
) -> Result<Vec<i32>, VvcError> {
    if !t.cbf {
        return Ok(vec![0i32; cw * ch]);
    }
    if t.ts {
        return Ok(t
            .levels
            .iter()
            .map(|&l| transform::dequantize_ts(l, qp.clamp(0, 63) as u8))
            .collect());
    }
    inverse_residual(plan, t, luma, cw, ch, qp)
}

fn inverse_residual(
    plan: &CuPlan,
    t: &TuData,
    luma: bool,
    cw: usize,
    ch: usize,
    qp: i32,
) -> Result<Vec<i32>, VvcError> {
    let cw_coded = cw.min(32);
    let ch_coded = ch.min(32);
    let scan = CoeffScan::new(cw_coded, ch_coded);
    let qp = qp.clamp(0, 63) as u8;
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
        dct::dequantize_dep_quant_bd(cw, ch, &full, &scan.raster, qp, 8).map_err(|_e| {
            VvcError::Unsupported("unsupported transform dimensions or transform type")
        })?
    } else {
        dct::dequantize_rect_bd(cw, ch, &full, qp, 8).map_err(|_e| {
            VvcError::Unsupported("unsupported transform dimensions or transform type")
        })?
    };
    let mut dequant = dequant;
    if std::env::var_os("VVC_TRACE").is_some() && cw == 1 {
        eprintln!("deq out={:?}", &dequant[..dequant.len().min(4)]);
    }

    if plan.lfnst > 0 && !used_mts && (luma || plan.tree == TreeType::Chroma) {
        let lfnst_intra_mode = if luma {
            lfnst_mode(plan.mode)
        } else {
            plan.lfnst_mode_c
        };
        lfnst::inverse_lfnst_rect(cw, ch, &mut dequant, plan.lfnst, lfnst_intra_mode).map_err(
            |_e| VvcError::Unsupported("unsupported transform dimensions or transform type"),
        )?;
    }
    let (th, tv) = if used_mts {
        mts_types(plan.mts)?
    } else if plan.lfnst == 0
        && plan.mip.is_none()
        && (plan.isp != predict::IspMode::None || plan.implicit_mts)
    {
        // H.266 §8.7.3 / VTM `TrQuant::getTrTypes`: intra ISP and implicit
        // intra MTS select DST-7 for a dimension in `4..=16`, DCT-2 otherwise.
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
    dct::inverse_transform_rect_bd(cw, ch, &dequant, th, tv, sw, sh, 8)
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

fn lfnst_skips(w: usize, h: usize) -> (usize, usize) {
    (if w > 32 { 16 } else { 0 }, if h > 32 { 16 } else { 0 })
}

fn lfnst_mode(mode: u8) -> u16 {
    match mode {
        0 => 0,
        1 => 1,
        2..=66 => u16::from(mode),
        _ => 0,
    }
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
        plane: &Plane,
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
        self.gather_len(plane, is_luma, x, y, 2 * w, 2 * h, 128)
    }

    fn gather_chroma(
        &self,
        plane: &Plane,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> predict::RefSamples {
        self.gather_len(plane, false, x, y, 2 * w, 2 * h, 128)
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
        let mid = 128i32;
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
        let params = predict::derive_cclm_params_from_4_samples(sel_luma, sel_chroma, count, 8);

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
                out[y * w + x] = pred.clamp(0, 255);
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
        let dc = 128;
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
    g.gather_len(&g.y, true, x, y, plan.w + w, plan.h + h, 128)
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
            if mrl == 0 && !isp {
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
            if mrl == 0 && !isp {
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
                predict::predict_angular_with_params(&m, w, h, true, &params)
            } else {
                predict::predict_angular_with_params(&mrl_refs, w, h, true, &params)
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
        IntraMode::Angular(dir) => predict::predict_angular(&refs, w, h, false, dir),
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
    if let Some(&scale) = g.vpdu_scale.get(&(vx, vy)) {
        return scale;
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
        0 => 128,
        c if c == n => (sum + (1 << (n.trailing_zeros() - 1))) >> n.trailing_zeros(),
        c if c == 2 * n => (sum + (1 << n.trailing_zeros())) >> (n.trailing_zeros() + 1),
        _ => 128,
    };
    let scale = g
        .lmcs
        .as_ref()
        .map_or(1 << 11, |r| r.chroma_scale_for_luma(avg));
    g.vpdu_scale.insert((vx, vy), scale);
    scale
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
    for yy in y..(y + h).min(mh * 4) {
        for xx in x..(x + w).min(mw * 4) {
            let cell = (yy / 4) * mw + xx / 4;
            if cell < map.len() {
                map[cell] = id;
                wmap[cell] = w as u16;
                hmap[cell] = h as u16;
            }
        }
    }
    if t.component == 0 {
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

fn luma_len(v: u16) -> usize {
    if v <= 4 {
        1
    } else if v >= 32 {
        7
    } else {
        3
    }
}

fn chroma_len(v: u16) -> usize {
    if v >= 8 { 3 } else { 1 }
}

fn apply_deblock(g: &mut Gen<'_>) {
    let qp_y = i32::from(g.qp_y.clamp(0, 63)) as u8;
    let beta = g.sh.luma_beta_offset_div2 as i8;
    let tc = g.sh.luma_tc_offset_div2 as i8;
    let w = g.pic_w;
    let h = g.pic_h;
    // Luma vertical edges.
    let mut x = 4usize;
    while x < w {
        let mut y = 0usize;
        while y < h {
            let lc = ((y / 4) * g.tu_y_w) + (x / 4 - 1);
            let rc = (y / 4) * g.tu_y_w + x / 4;
            if g.tu_y[lc] != u32::MAX && g.tu_y[rc] != u32::MAX && g.tu_y[lc] != g.tu_y[rc] {
                let lp = luma_len(g.tu_wy[lc]);
                let lq = luma_len(g.tu_wy[rc]);
                let mut lines = [[0i32; 16]; 4];
                for (i, line) in lines.iter_mut().enumerate() {
                    let yy = (y + i).min(h - 1);
                    for (k, v) in line.iter_mut().enumerate() {
                        let xx = (x as isize - 8 + k as isize).clamp(0, w as isize - 1) as usize;
                        *v = i32::from(g.y.get(xx, yy));
                    }
                }
                filter::deblock_luma_edge_normative(
                    &mut lines, qp_y, qp_y, 2, beta, tc, lp, lq, false,
                );
                for (i, line) in lines.iter().enumerate() {
                    let yy = (y + i).min(h - 1);
                    for k in 0..16 {
                        let xx = (x as isize - 8 + k as isize).clamp(0, w as isize - 1) as usize;
                        g.y.set(xx, yy, line[k].clamp(0, 255) as u8);
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
                let lp = luma_len(g.tu_hy[tc_cell]);
                let lq = luma_len(g.tu_hy[bc]);
                let mut lines = [[0i32; 16]; 4];
                for (i, line) in lines.iter_mut().enumerate() {
                    let xx = (x + i).min(w - 1);
                    for (k, v) in line.iter_mut().enumerate() {
                        let yy = (y as isize - 8 + k as isize).clamp(0, h as isize - 1) as usize;
                        *v = i32::from(g.y.get(xx, yy));
                    }
                }
                filter::deblock_luma_edge_normative(
                    &mut lines,
                    qp_y,
                    qp_y,
                    2,
                    beta,
                    tc,
                    lp,
                    lq,
                    y.is_multiple_of(g.ctu),
                );
                for (i, line) in lines.iter().enumerate() {
                    let xx = (x + i).min(w - 1);
                    for k in 0..16 {
                        let yy = (y as isize - 8 + k as isize).clamp(0, h as isize - 1) as usize;
                        g.y.set(xx, yy, line[k].clamp(0, 255) as u8);
                    }
                }
            }
            x += 4;
        }
        y += 4;
    }
    // Chroma.
    for comp in 1..3usize {
        let qp = if comp == 1 { g.qp_cb } else { g.qp_cr };
        let qp = qp.clamp(0, 63) as u8;
        let (plane, wmap, hmap, cmap, cw, ch, tuw, tuh) = if comp == 1 {
            (
                &mut g.u, &g.tu_wc, &g.tu_hc, &g.tu_u, g.cgrid_w, g.cgrid_h, g.tu_c_w, g.tu_c_h,
            )
        } else {
            (
                &mut g.v, &g.tu_wc, &g.tu_hc, &g.tu_v, g.cgrid_w, g.cgrid_h, g.tu_c_w, g.tu_c_h,
            )
        };
        let mut x = 4usize;
        while x < cw {
            let mut y = 0usize;
            while y < ch {
                let lc = (y / 4) * tuw + (x / 4 - 1);
                let rc = (y / 4) * tuw + x / 4;
                if cmap[lc] != u32::MAX && cmap[rc] != u32::MAX && cmap[lc] != cmap[rc] {
                    let lp = chroma_len(wmap[lc]);
                    let lq = chroma_len(wmap[rc]);
                    let mut lines = [[0i32; 8]; 4];
                    for (i, line) in lines.iter_mut().enumerate() {
                        let yy = (y + i).min(ch - 1);
                        for (k, v) in line.iter_mut().enumerate() {
                            let xx =
                                (x as isize - 4 + k as isize).clamp(0, cw as isize - 1) as usize;
                            *v = i32::from(plane.get(xx, yy));
                        }
                    }
                    filter::deblock_chroma_edge(&mut lines, i32::from(qp), 2, 0, 0, lp, lq, 8);
                    for (i, line) in lines.iter().enumerate() {
                        let yy = (y + i).min(ch - 1);
                        for k in 0..8 {
                            let xx =
                                (x as isize - 4 + k as isize).clamp(0, cw as isize - 1) as usize;
                            plane.set(xx, yy, line[k].clamp(0, 255) as u8);
                        }
                    }
                }
                y += 4;
            }
            x += 4;
        }
        let mut y = 4usize;
        while y < ch {
            let mut x = 0usize;
            while x < cw {
                let tc_cell = ((y / 4 - 1) * tuw) + x / 4;
                let bc = (y / 4) * tuw + x / 4;
                if cmap[tc_cell] != u32::MAX && cmap[bc] != u32::MAX && cmap[tc_cell] != cmap[bc] {
                    let lp = chroma_len(hmap[tc_cell]);
                    let lq = chroma_len(hmap[bc]);
                    let mut lines = [[0i32; 8]; 4];
                    for (i, line) in lines.iter_mut().enumerate() {
                        let xx = (x + i).min(cw - 1);
                        for (k, v) in line.iter_mut().enumerate() {
                            let yy =
                                (y as isize - 4 + k as isize).clamp(0, ch as isize - 1) as usize;
                            *v = i32::from(plane.get(xx, yy));
                        }
                    }
                    filter::deblock_chroma_edge(&mut lines, i32::from(qp), 2, 0, 0, lp, lq, 8);
                    for (i, line) in lines.iter().enumerate() {
                        let xx = (x + i).min(cw - 1);
                        for k in 0..8 {
                            let yy =
                                (y as isize - 4 + k as isize).clamp(0, ch as isize - 1) as usize;
                            plane.set(xx, yy, line[k].clamp(0, 255) as u8);
                        }
                    }
                }
                x += 4;
            }
            y += 4;
        }
        let _ = tuh;
    }
}

fn apply_sao(g: &mut Gen<'_>) {
    let ctu = g.ctu;
    let cw = g.cgrid_w.min(ctu / 2);
    let ch = g.cgrid_h.min(ctu / 2);
    let src_y = g.y.clone();
    let src_u = g.u.clone();
    let src_v = g.v.clone();
    for gy in 0..g.pic_h.div_ceil(ctu) {
        for gx in 0..g.pic_w.div_ceil(ctu) {
            let idx = gy * g.ctu_cols + gx;
            let sao = g.sao[idx];
            filter::apply_sao_region(&src_y, &mut g.y, gx * ctu, gy * ctu, ctu, ctu, sao[0]);
            if g.sh.sao_chroma_used {
                let cw_here = cw.min(g.cgrid_w.saturating_sub(gx * ctu / 2));
                let ch_here = ch.min(g.cgrid_h.saturating_sub(gy * ctu / 2));
                if cw_here > 0 && ch_here > 0 {
                    filter::apply_sao_region(
                        &src_u,
                        &mut g.u,
                        gx * ctu / 2,
                        gy * ctu / 2,
                        cw_here,
                        ch_here,
                        sao[1],
                    );
                    filter::apply_sao_region(
                        &src_v,
                        &mut g.v,
                        gx * ctu / 2,
                        gy * ctu / 2,
                        cw_here,
                        ch_here,
                        sao[2],
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
    for gy in 0..g.pic_h.div_ceil(ctu) {
        for gx in 0..g.pic_w.div_ceil(ctu) {
            let idx = gy * g.ctu_cols + gx;
            let x0 = gx * ctu;
            let y0 = gy * ctu;
            let w = ctu.min(g.pic_w - x0);
            let h = ctu.min(g.pic_h - y0);
            if let Some(mode) = g.alf_luma[idx] {
                let aps_idx = usize::from(g.sh.alf_aps_id_chroma & 7);
                let _ = aps_idx;
                let aps =
                    g.sh.alf_aps_ids_luma
                        .first()
                        .and_then(|&id| g.aps.alf[usize::from(id & 7)].as_ref());
                alf_luma_block(&src_y, &mut g.y, x0, y0, w, h, mode, aps);
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
                        );
                    }
                }
            }
        }
    }
}

fn alf_luma_block(
    src: &Plane,
    dst: &mut Plane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    mode: AlfLumaMode,
    aps: Option<&super::hls::VvcAlfAps>,
) {
    let get_src = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };
    let mut by = y0;
    while by < y0 + h {
        let mut bx = x0;
        while bx < x0 + w {
            let (class_idx, transpose) = filter::derive_alf_class_4x4(src, bx, by);
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
            let clip_table = filter::ALF_CLIP_VALS_8BIT;
            let mut fc = [0i32; 12];
            let mut fclip = [0i32; 12];
            for k in 0..12 {
                fc[k] = i32::from(coeffs[perm[k]]);
                fclip[k] = clip_table[usize::from(clips[perm[k]] & 3)];
            }
            let xe = (bx + 4).min(x0 + w);
            let ye = (by + 4).min(y0 + h);
            for y in by..ye {
                let iy = y as isize;
                for x in bx..xe {
                    let ix = x as isize;
                    let cur = get_src(ix, iy);
                    let mut sum = 0i32;
                    let terms = [
                        (ix, iy + 3, ix, iy - 3),
                        (ix + 1, iy + 2, ix - 1, iy - 2),
                        (ix, iy + 2, ix, iy - 2),
                        (ix - 1, iy + 2, ix + 1, iy - 2),
                        (ix + 2, iy + 1, ix - 2, iy - 1),
                        (ix + 1, iy + 1, ix - 1, iy - 1),
                        (ix, iy + 1, ix, iy - 1),
                        (ix - 1, iy + 1, ix + 1, iy - 1),
                        (ix - 2, iy + 1, ix + 2, iy - 1),
                        (ix + 3, iy, ix - 3, iy),
                        (ix + 2, iy, ix - 2, iy),
                        (ix + 1, iy, ix - 1, iy),
                    ];
                    for (k, &(ax, ay, bx2, by2)) in terms.iter().enumerate() {
                        let a = get_src(ax, ay);
                        let b = get_src(bx2, by2);
                        let v = (a - b).clamp(-fclip[k], fclip[k]);
                        sum += fc[k] * v;
                    }
                    let val = cur + ((sum + 64) >> 7);
                    dst.set(x, y, val.clamp(0, 255) as u8);
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

fn alf_chroma_block(
    src: &Plane,
    dst: &mut Plane,
    aps: &super::hls::VvcAlfAps,
    alt: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    if aps.chroma_filters.is_empty() {
        return;
    }
    let idx = alt.min(aps.chroma_filters.len() - 1);
    let c = aps.chroma_filters[idx];
    let cl = aps.chroma_clip_idx.get(idx).copied().unwrap_or([0u8; 6]);
    let table = filter::ALF_CLIP_VALS_8BIT;
    let get = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };
    for y in y0..(y0 + h).min(dst.height) {
        for x in x0..(x0 + w).min(dst.width) {
            let (ix, iy) = (x as isize, y as isize);
            let cur = get(ix, iy);
            let mut sum = 0i32;
            let terms = [
                (ix, iy + 2, ix, iy - 2),
                (ix + 1, iy + 1, ix - 1, iy - 1),
                (ix, iy + 1, ix, iy - 1),
                (ix - 1, iy + 1, ix + 1, iy - 1),
                (ix + 2, iy, ix - 2, iy),
                (ix + 1, iy, ix - 1, iy),
            ];
            for (k, &(ax, ay, bx, by)) in terms.iter().enumerate() {
                let v = (get(ax, ay) - get(bx, by)).clamp(
                    -table[usize::from(cl[k] & 3)],
                    table[usize::from(cl[k] & 3)],
                );
                sum += i32::from(c[k]) * v;
            }
            let val = cur + ((sum + 64) >> 7);
            dst.set(x, y, val.clamp(0, 255) as u8);
        }
    }
}
