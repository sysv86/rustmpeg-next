// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Bit-exact VVC CABAC engine and complete 379-context initialization tables
//! (H.266 clause 9.3).
//!
//! Algorithms and context initialization tables derived from VTM 24.0
//! (BSD-3-Clause, Copyright (c) 2010-2020 ITU/ISO/IEC):
//! - `source/Lib/CommonLib/Contexts.cpp:180-895`
//! - `source/Lib/CommonLib/ContextModelling.h:60-185`
//! - `source/Lib/EncoderLib/BinEncoder.cpp:50-340`
//! - `source/Lib/DecoderLib/BinDecoder.cpp:48-295`
//! - `source/Lib/EncoderLib/CABACWriter.cpp:54-66, 921-939, 3268-3309`
//! - `source/Lib/DecoderLib/CABACReader.cpp:75-88, 1314-1325, 2666-2720`
//! See THIRD_PARTY.md.

use super::bits::BitWriter;
use super::hls::VvcSliceType;

/// Number of context models in VVC (`NumberOfContexts` in VTM `Contexts.h`).
pub const NUM_CONTEXTS: usize = 379;

/// `m_RenormTable_32`: LPS renormalisation bits indexed by `lps_range >> 3`
/// (`H.266 p. 435, PDF p. 451, Table 127`; VTM `BinEncoder.cpp:52`).
const RENORM_LPS: [u8; 32] = [
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

const MASK_0: u32 = 0x7FE0; // keeps bits 15..5
const MASK_1: u32 = 0x7FFE; // keeps bits 15..1

/// One adaptive context model (`BinProbModel_Std`, H.266 §9.3.2 & §9.3.4).
#[derive(Debug, Clone, Copy)]
pub struct CtxModel {
    state0: u16,
    state1: u16,
    rate0: u8,
    rate1: u8,
}

impl CtxModel {
    /// Initialise from `initId`, the slice QP and the context's log2 window size
    /// (`H.266 p. 399, PDF p. 415, §9.3.2.2 Eq. 1515–1523`).
    pub fn init(qp: i32, init_id: u8, log2_window: u8) -> Self {
        let slope = i32::from(init_id >> 3) - 4;
        let offset = i32::from(init_id & 7) * 18 + 1;
        let inistate = ((slope * (qp - 16)) >> 1) + offset;
        let state = inistate.clamp(1, 127) as u16;
        let p1 = state << 8;
        let rate0 = 2 + ((log2_window >> 2) & 3);
        let rate1 = 3 + rate0 + (log2_window & 3);
        Self {
            state0: (u32::from(p1) & MASK_0) as u16,
            state1: (u32::from(p1) & MASK_1) as u16,
            rate0,
            rate1,
        }
    }

    #[inline]
    fn state(&self) -> u8 {
        ((self.state0 + self.state1) >> 8) as u8
    }

    #[inline]
    fn mps(&self) -> bool {
        self.state() >= 128
    }

    #[inline]
    fn get_lps(&self, range: u32) -> u32 {
        let mut q = self.state();
        if q & 0x80 != 0 {
            q ^= 0xFF;
        }
        (((u32::from(q) >> 2) * (range >> 5)) >> 1) + 4
    }

    #[inline]
    fn update(&mut self, bin: bool) {
        let s0 = u32::from(self.state0);
        let s1 = u32::from(self.state1);
        let r0 = u32::from(self.rate0);
        let r1 = u32::from(self.rate1);
        let mut n0 = s0 - ((s0 >> r0) & MASK_0);
        let mut n1 = s1 - ((s1 >> r1) & MASK_1);
        if bin {
            n0 += (0x7FFF >> r0) & MASK_0;
            n1 += (0x7FFF >> r1) & MASK_1;
        }
        self.state0 = (n0 & 0xFFFF) as u16;
        self.state1 = (n1 & 0xFFFF) as u16;
    }
}

/// Complete `(initId, log2WindowSize)` table for `I_SLICE` (row 2 and row 3 of
/// `ContextSetCfg::addCtxSet` in VTM 24.0 `source/Lib/CommonLib/Contexts.cpp:180-895`;
/// `H.266 p. 400–423, PDF p. 416–439, Tables 54–125`).
#[rustfmt::skip]
pub const CTX_INIT_I: [(u8, u8); NUM_CONTEXTS] = [
    (19, 12), (28, 13), (38, 8), (27, 8), (29, 13), (38, 12), (20, 5), (30, 9),
    (31, 9), (27, 0), (6, 8), (15, 8), (25, 12), (19, 12), (37, 8), (43, 9),
    (42, 8), (29, 9), (27, 8), (44, 5), (36, 12), (45, 13), (36, 12), (45, 13),
    (35, 1), (35, 0), (0, 5), (26, 4), (28, 8), (26, 4), (35, 5), (35, 5),
    (34, 4), (35, 4), (35, 10), (35, 0), (35, 5), (35, 1), (25, 5), (60, 8),
    (45, 6), (13, 1), (28, 5), (59, 4), (27, 9), (34, 5), (33, 9), (49, 10),
    (50, 9), (25, 6), (35, 8), (35, 8), (35, 0), (35, 0), (35, 1), (35, 4),
    (35, 4), (35, 0), (35, 0), (35, 4), (35, 4), (35, 4), (35, 4), (35, 4),
    (35, 0), (35, 0), (35, 4), (35, 0), (35, 1), (14, 9), (45, 5), (19, 1),
    (35, 4), (1, 1), (27, 0), (6, 4), (52, 1), (15, 5), (12, 1), (5, 8),
    (7, 9), (12, 5), (21, 0), (33, 2), (28, 1), (36, 0), (18, 8), (31, 5),
    (25, 5), (15, 8), (25, 12), (19, 9), (28, 9), (14, 10), (25, 9), (20, 9),
    (29, 9), (30, 10), (19, 8), (37, 8), (30, 8), (38, 10), (25, 12), (27, 12),
    (28, 9), (37, 13), (34, 4), (53, 5), (53, 8), (46, 9), (11, 9), (38, 13),
    (46, 8), (54, 8), (27, 8), (39, 8), (39, 8), (39, 5), (44, 8), (39, 0),
    (39, 0), (39, 0), (19, 8), (46, 12), (38, 12), (39, 8), (52, 4), (39, 0),
    (39, 0), (39, 0), (18, 8), (39, 8), (39, 8), (39, 8), (27, 8), (39, 0),
    (39, 4), (39, 4), (0, 0), (39, 0), (39, 0), (39, 0), (11, 8), (39, 8),
    (39, 8), (39, 8), (19, 4), (39, 0), (39, 0), (39, 0), (33, 8), (25, 9),
    (18, 12), (26, 13), (34, 13), (27, 13), (25, 10), (26, 13), (19, 13), (42, 13),
    (35, 13), (33, 13), (19, 13), (27, 13), (35, 13), (35, 13), (34, 10), (42, 13),
    (20, 13), (43, 13), (20, 13), (33, 8), (25, 12), (26, 12), (42, 12), (19, 13),
    (27, 13), (26, 13), (50, 13), (35, 13), (20, 13), (43, 13), (25, 1), (1, 5),
    (40, 9), (25, 9), (33, 9), (11, 6), (17, 5), (25, 9), (25, 10), (18, 10),
    (4, 9), (17, 9), (33, 9), (26, 9), (19, 9), (13, 9), (33, 6), (19, 8),
    (20, 9), (28, 9), (22, 10), (40, 1), (9, 5), (25, 8), (18, 8), (26, 9),
    (35, 6), (25, 6), (26, 9), (35, 8), (28, 8), (37, 9), (25, 9), (25, 5),
    (11, 10), (27, 13), (20, 13), (21, 10), (33, 9), (12, 10), (28, 13), (21, 13),
    (22, 13), (34, 9), (28, 10), (29, 10), (29, 10), (30, 13), (36, 8), (29, 9),
    (45, 10), (30, 10), (23, 13), (40, 8), (33, 8), (27, 9), (28, 12), (21, 12),
    (37, 10), (36, 5), (37, 9), (45, 9), (38, 9), (46, 13), (13, 8), (5, 5),
    (4, 4), (21, 5), (14, 4), (4, 4), (6, 5), (14, 4), (21, 1), (11, 0),
    (14, 4), (7, 1), (14, 0), (5, 0), (11, 0), (21, 0), (30, 1), (22, 0),
    (13, 0), (42, 0), (12, 5), (4, 4), (3, 4), (13, 8), (5, 5), (4, 8),
    (6, 5), (13, 5), (11, 4), (14, 5), (6, 5), (5, 4), (3, 0), (14, 5),
    (22, 4), (6, 1), (4, 0), (3, 0), (6, 1), (22, 4), (29, 0), (20, 0),
    (34, 0), (12, 6), (4, 5), (3, 5), (42, 12), (35, 5), (60, 0), (13, 4),
    (28, 9), (52, 9), (42, 10), (25, 1), (42, 5), (42, 9), (50, 9), (37, 6),
    (45, 9), (30, 10), (46, 5), (45, 0), (38, 9), (46, 5), (25, 1), (9, 1),
    (29, 8), (0, 0), (28, 9), (0, 0), (33, 9), (43, 2), (35, 1), (35, 5),
    (35, 10), (35, 8), (35, 4), (35, 1), (35, 13), (35, 8), (35, 8), (35, 0),
    (34, 5), (35, 0), (35, 0), (35, 4), (62, 0), (39, 0), (39, 0), (54, 4),
    (39, 0), (39, 0), (31, 1), (39, 0), (39, 0), (11, 0), (11, 0), (46, 0),
    (18, 4), (30, 1), (31, 4), (18, 4), (30, 1), (31, 4), (35, 1), (17, 1),
    (42, 5), (36, 8), (12, 1), (21, 1), (35, 0), (18, 5), (20, 8), (38, 8),
    (25, 13), (28, 13), (38, 8), (11, 6), (35, 8), (10, 1), (3, 1), (3, 1),
    (3, 1), (11, 4), (5, 2), (5, 1), (14, 6), (12, 1), (17, 4), (46, 4),
    (28, 5), (25, 8), (46, 8),
];

/// Complete `(initId, log2WindowSize)` table for `P_SLICE` (row 1 and row 3 of
/// `ContextSetCfg::addCtxSet` in VTM 24.0 `source/Lib/CommonLib/Contexts.cpp:180-895`).
#[rustfmt::skip]
pub const CTX_INIT_P: [(u8, u8); NUM_CONTEXTS] = [
    (11, 12), (35, 13), (53, 8), (12, 8), (6, 13), (30, 12), (13, 5), (15, 9),
    (31, 9), (20, 0), (14, 8), (23, 8), (18, 12), (19, 12), (6, 8), (43, 9),
    (35, 8), (37, 9), (34, 8), (52, 5), (43, 12), (37, 13), (21, 12), (22, 13),
    (25, 1), (12, 0), (57, 5), (59, 4), (45, 8), (21, 4), (38, 5), (7, 5),
    (20, 4), (26, 4), (43, 10), (60, 0), (40, 5), (35, 1), (25, 5), (58, 8),
    (36, 6), (12, 1), (20, 5), (34, 4), (27, 9), (25, 5), (41, 9), (57, 10),
    (58, 9), (26, 6), (35, 8), (35, 8), (7, 0), (6, 0), (5, 1), (12, 4),
    (4, 4), (40, 0), (20, 0), (35, 4), (48, 4), (57, 4), (44, 4), (12, 4),
    (13, 0), (14, 0), (35, 4), (5, 0), (4, 1), (44, 9), (43, 5), (40, 1),
    (36, 4), (0, 1), (13, 0), (5, 4), (46, 1), (23, 5), (5, 1), (20, 8),
    (7, 9), (25, 5), (28, 0), (25, 2), (29, 1), (45, 0), (25, 8), (30, 5),
    (25, 5), (45, 8), (17, 12), (41, 9), (42, 9), (29, 10), (25, 9), (49, 9),
    (43, 9), (37, 10), (33, 8), (58, 8), (51, 8), (30, 10), (17, 12), (34, 12),
    (35, 9), (21, 13), (41, 4), (59, 5), (60, 8), (38, 9), (19, 9), (38, 13),
    (38, 8), (46, 8), (34, 8), (54, 8), (54, 8), (39, 5), (6, 8), (39, 0),
    (39, 0), (39, 0), (35, 8), (45, 12), (53, 12), (54, 8), (44, 4), (39, 0),
    (39, 0), (39, 0), (19, 8), (39, 8), (54, 8), (39, 8), (19, 8), (39, 0),
    (39, 4), (39, 4), (56, 0), (39, 0), (39, 0), (39, 0), (34, 8), (38, 8),
    (62, 8), (39, 8), (26, 4), (39, 0), (39, 0), (39, 0), (18, 8), (17, 9),
    (33, 12), (18, 13), (26, 13), (42, 13), (25, 10), (33, 13), (26, 13), (42, 13),
    (27, 13), (25, 13), (34, 13), (42, 13), (42, 13), (35, 13), (26, 10), (27, 13),
    (42, 13), (20, 13), (20, 13), (25, 8), (25, 12), (26, 12), (11, 12), (19, 13),
    (27, 13), (33, 13), (42, 13), (35, 13), (35, 13), (43, 13), (17, 1), (0, 5),
    (1, 9), (17, 9), (25, 9), (18, 6), (0, 5), (9, 9), (25, 10), (33, 10),
    (34, 9), (9, 9), (25, 9), (18, 9), (26, 9), (20, 9), (25, 6), (18, 8),
    (19, 9), (27, 9), (29, 10), (17, 1), (9, 5), (25, 8), (10, 8), (18, 9),
    (4, 6), (17, 6), (33, 9), (19, 8), (20, 8), (29, 9), (0, 9), (17, 5),
    (26, 10), (19, 13), (35, 13), (21, 10), (25, 9), (34, 10), (20, 13), (28, 13),
    (29, 13), (33, 9), (27, 10), (28, 10), (29, 10), (22, 13), (34, 8), (28, 9),
    (44, 10), (37, 10), (38, 13), (0, 8), (25, 8), (19, 9), (20, 12), (13, 12),
    (14, 10), (57, 5), (44, 9), (30, 9), (30, 9), (23, 13), (6, 8), (13, 5),
    (12, 4), (6, 5), (6, 4), (12, 4), (14, 5), (14, 4), (13, 1), (12, 0),
    (29, 4), (7, 1), (6, 0), (13, 0), (36, 0), (28, 0), (14, 1), (13, 0),
    (5, 0), (26, 0), (12, 5), (4, 4), (18, 4), (5, 8), (5, 5), (12, 8),
    (6, 5), (6, 5), (4, 4), (6, 5), (14, 5), (5, 4), (12, 0), (14, 5),
    (7, 4), (13, 1), (5, 0), (13, 0), (21, 1), (14, 4), (20, 0), (12, 0),
    (34, 0), (11, 6), (4, 5), (18, 5), (34, 12), (28, 5), (60, 0), (5, 4),
    (37, 9), (45, 9), (27, 10), (0, 1), (42, 5), (59, 9), (51, 9), (30, 6),
    (30, 9), (38, 10), (23, 5), (38, 0), (53, 9), (46, 5), (25, 1), (9, 1),
    (45, 8), (40, 0), (27, 9), (0, 0), (33, 9), (36, 2), (56, 1), (57, 5),
    (42, 10), (20, 8), (43, 4), (12, 1), (28, 13), (35, 8), (35, 8), (59, 0),
    (48, 5), (58, 0), (60, 0), (60, 4), (13, 0), (23, 0), (46, 0), (4, 4),
    (61, 0), (54, 0), (19, 1), (46, 0), (54, 0), (20, 0), (12, 0), (46, 0),
    (18, 4), (21, 1), (38, 4), (18, 4), (21, 1), (38, 4), (57, 1), (0, 1),
    (57, 5), (44, 8), (27, 1), (36, 1), (45, 0), (18, 5), (12, 8), (29, 8),
    (40, 13), (35, 13), (44, 8), (3, 6), (35, 8), (2, 1), (10, 1), (3, 1),
    (3, 1), (18, 4), (11, 2), (4, 1), (28, 6), (5, 1), (10, 4), (53, 4),
    (43, 5), (25, 8), (46, 8),
];

/// Complete `(initId, log2WindowSize)` table for `B_SLICE` (row 0 and row 3 of
/// `ContextSetCfg::addCtxSet` in VTM 24.0 `source/Lib/CommonLib/Contexts.cpp:180-895`).
#[rustfmt::skip]
pub const CTX_INIT_B: [(u8, u8); NUM_CONTEXTS] = [
    (18, 12), (27, 13), (15, 8), (18, 8), (28, 13), (45, 12), (26, 5), (7, 9),
    (23, 9), (26, 0), (36, 8), (38, 8), (18, 12), (34, 12), (21, 8), (43, 9),
    (42, 8), (37, 9), (42, 8), (44, 5), (28, 12), (29, 13), (28, 12), (29, 13),
    (25, 1), (20, 0), (57, 5), (60, 4), (46, 8), (6, 4), (46, 5), (15, 5),
    (18, 4), (25, 4), (43, 10), (59, 0), (40, 5), (35, 1), (25, 5), (59, 8),
    (44, 6), (13, 1), (6, 5), (26, 4), (27, 9), (25, 5), (56, 9), (57, 10),
    (50, 9), (26, 6), (35, 8), (35, 8), (14, 0), (13, 0), (5, 1), (4, 4),
    (3, 4), (40, 0), (5, 0), (35, 4), (25, 4), (58, 4), (45, 4), (19, 4),
    (13, 0), (6, 0), (35, 4), (4, 0), (5, 1), (51, 9), (36, 5), (19, 1),
    (21, 4), (0, 1), (28, 0), (12, 4), (46, 1), (15, 5), (6, 1), (5, 8),
    (14, 9), (25, 5), (37, 0), (9, 2), (36, 1), (45, 0), (25, 8), (45, 5),
    (25, 5), (14, 8), (17, 12), (41, 9), (49, 9), (36, 10), (1, 9), (49, 9),
    (50, 9), (37, 10), (48, 8), (51, 8), (58, 8), (45, 10), (9, 12), (49, 12),
    (50, 9), (36, 13), (48, 4), (59, 5), (59, 8), (38, 9), (26, 9), (45, 13),
    (53, 8), (46, 8), (49, 8), (54, 8), (61, 8), (39, 5), (35, 8), (39, 0),
    (39, 0), (39, 0), (34, 8), (45, 12), (38, 12), (31, 8), (58, 4), (39, 0),
    (39, 0), (39, 0), (19, 8), (54, 8), (39, 8), (39, 8), (50, 8), (39, 0),
    (39, 4), (39, 4), (0, 0), (39, 0), (39, 0), (39, 0), (34, 8), (38, 8),
    (54, 8), (39, 8), (41, 4), (39, 0), (39, 0), (39, 0), (33, 8), (40, 9),
    (25, 12), (41, 13), (26, 13), (42, 13), (25, 10), (33, 13), (26, 13), (34, 13),
    (27, 13), (25, 13), (41, 13), (42, 13), (42, 13), (35, 13), (33, 10), (27, 13),
    (35, 13), (42, 13), (43, 13), (33, 8), (25, 12), (26, 12), (34, 12), (19, 13),
    (27, 13), (33, 13), (42, 13), (43, 13), (35, 13), (43, 13), (25, 1), (0, 5),
    (0, 9), (17, 9), (25, 9), (26, 6), (0, 5), (9, 9), (25, 10), (33, 10),
    (19, 9), (0, 9), (25, 9), (33, 9), (26, 9), (20, 9), (25, 6), (33, 8),
    (27, 9), (35, 9), (22, 10), (25, 1), (1, 5), (25, 8), (33, 8), (26, 9),
    (12, 6), (25, 6), (33, 9), (27, 8), (28, 8), (37, 9), (0, 9), (0, 5),
    (33, 10), (34, 13), (35, 13), (21, 10), (25, 9), (34, 10), (35, 13), (28, 13),
    (29, 13), (40, 9), (42, 10), (43, 10), (29, 10), (30, 13), (49, 8), (36, 9),
    (37, 10), (45, 10), (38, 13), (0, 8), (40, 8), (34, 9), (43, 12), (36, 12),
    (37, 10), (57, 5), (52, 9), (45, 9), (38, 9), (46, 13), (6, 8), (6, 5),
    (12, 4), (14, 5), (6, 4), (4, 4), (14, 5), (7, 4), (6, 1), (4, 0),
    (29, 4), (7, 1), (6, 0), (6, 0), (12, 0), (28, 0), (7, 1), (13, 0),
    (13, 0), (35, 0), (19, 5), (5, 4), (4, 4), (5, 8), (5, 5), (20, 8),
    (13, 5), (13, 5), (19, 4), (21, 5), (6, 5), (12, 4), (12, 0), (14, 5),
    (14, 4), (5, 1), (4, 0), (12, 0), (13, 1), (7, 4), (13, 0), (12, 0),
    (41, 0), (11, 6), (5, 5), (27, 5), (34, 12), (28, 5), (2, 0), (2, 4),
    (52, 9), (37, 9), (27, 10), (17, 1), (35, 5), (50, 9), (58, 9), (45, 6),
    (45, 9), (30, 10), (38, 5), (45, 0), (38, 9), (46, 5), (25, 1), (17, 1),
    (45, 8), (25, 0), (27, 9), (0, 0), (33, 9), (43, 2), (41, 1), (57, 5),
    (42, 10), (35, 8), (51, 4), (27, 1), (28, 13), (35, 8), (35, 8), (59, 0),
    (26, 5), (50, 0), (60, 0), (38, 4), (33, 0), (52, 0), (46, 0), (25, 4),
    (61, 0), (54, 0), (25, 1), (61, 0), (54, 0), (11, 0), (26, 0), (46, 0),
    (25, 4), (35, 1), (38, 4), (25, 4), (28, 1), (38, 4), (57, 1), (0, 1),
    (43, 5), (45, 8), (42, 1), (43, 1), (52, 0), (18, 5), (35, 8), (45, 8),
    (25, 13), (50, 13), (37, 8), (11, 6), (35, 8), (3, 1), (4, 1), (4, 1),
    (5, 1), (19, 4), (11, 2), (4, 1), (6, 6), (35, 1), (25, 4), (46, 4),
    (28, 5), (33, 8), (38, 8),
];

/// All context models of one slice.
#[derive(Debug, Clone)]
pub struct CtxStore {
    /// Context models indexed by `ctx::*` plus derived neighbour offsets.
    pub models: [CtxModel; NUM_CONTEXTS],
}

impl CtxStore {
    /// Initialise every context model from a slice initialization table at `qp`.
    pub fn from_table(qp: i32, table: &[(u8, u8); NUM_CONTEXTS]) -> Self {
        let qp = qp.clamp(0, 63);
        let mut models = [CtxModel::init(qp, 35, 8); NUM_CONTEXTS];
        for (model, &(init_id, ws)) in models.iter_mut().zip(table.iter()) {
            *model = CtxModel::init(qp, init_id, ws);
        }
        Self { models }
    }

    /// Initialise every context model for an I slice at `qp`.
    pub fn new_i_slice(qp: i32) -> Self {
        Self::from_table(qp, &CTX_INIT_I)
    }

    /// Initialise every context model for a P slice at `qp`.
    pub fn new_p_slice(qp: i32) -> Self {
        Self::from_table(qp, &CTX_INIT_P)
    }

    /// Initialise every context model for a B slice at `qp`.
    pub fn new_b_slice(qp: i32) -> Self {
        Self::from_table(qp, &CTX_INIT_B)
    }

    /// Initialise every context model for `slice_type` and `cabac_init_flag` at `qp`
    /// (`H.266 p. 399, PDF p. 415, §9.3.2.2`; VTM `CABACReader.cpp:75-88`).
    pub fn new_for_slice(slice_type: VvcSliceType, cabac_init_flag: bool, qp: i32) -> Self {
        match (slice_type, cabac_init_flag) {
            (VvcSliceType::I, _) => Self::new_i_slice(qp),
            (VvcSliceType::P, false) | (VvcSliceType::B, true) => Self::new_p_slice(qp),
            (VvcSliceType::B, false) | (VvcSliceType::P, true) => Self::new_b_slice(qp),
        }
    }
}

/// Context id offsets for all 86 VVC syntax element context sets
/// (`reference/VTM-24.0/source/Lib/CommonLib/Contexts.cpp:180-895`).
pub mod ctx {
    /// `split_cu_flag` (9 contexts).
    pub const SPLIT_FLAG: usize = 0;
    /// `split_qt_flag` (6 contexts).
    pub const SPLIT_QT_FLAG: usize = 9;
    /// `mtt_split_cu_vertical_flag` (5 contexts).
    pub const SPLIT_HV_FLAG: usize = 15;
    /// `mtt_split_cu_binary_flag` (4 contexts).
    pub const SPLIT_12_FLAG: usize = 20;
    /// `non_inter_flag` (2 contexts).
    pub const MODE_CONS_FLAG: usize = 24;
    /// `cu_skip_flag` (3 contexts).
    pub const SKIP_FLAG: usize = 26;
    /// `general_merge_flag` (1 context).
    pub const MERGE_FLAG: usize = 29;
    /// `regular_merge_flag` (2 contexts).
    pub const REGULAR_MERGE_FLAG: usize = 30;
    /// `merge_idx` (1 context).
    pub const MERGE_IDX: usize = 32;
    /// `mmvd_merge_flag` (1 context).
    pub const MMVD_FLAG: usize = 33;
    /// `mmvd_cand_flag` (1 context).
    pub const MMVD_MERGE_IDX: usize = 34;
    /// `mmvd_distance_idx` (1 context).
    pub const MMVD_STEP_MVP_IDX: usize = 35;
    /// `pred_mode_flag` (2 contexts).
    pub const PRED_MODE: usize = 36;
    /// `intra_luma_ref_idx` (2 contexts).
    pub const MULTI_REF_LINE_IDX: usize = 38;
    /// `intra_luma_mpm_flag` (1 context).
    pub const INTRA_LUMA_MPM_FLAG: usize = 40;
    /// `intra_luma_not_planar_flag` (2 contexts: 0 = ISP, 1 = non-ISP).
    pub const INTRA_LUMA_PLANAR_FLAG: usize = 41;
    /// `cclm_mode_flag` (1 context).
    pub const CCLM_MODE_FLAG: usize = 43;
    /// `cclm_mode_idx` (1 context).
    pub const CCLM_MODE_IDX: usize = 44;
    /// `intra_chroma_pred_mode` (1 context).
    pub const INTRA_CHROMA_PRED_MODE: usize = 45;
    /// `intra_mip_flag` (4 contexts).
    pub const MIP_FLAG: usize = 46;
    /// `cu_qp_delta_abs` (2 contexts).
    pub const DELTA_QP: usize = 50;
    /// `inter_pred_idc` (6 contexts).
    pub const INTER_DIR: usize = 52;
    /// `ref_idx_l0` / `ref_idx_l1` (2 contexts).
    pub const REF_PIC: usize = 58;
    /// `merge_subblock_flag` (3 contexts).
    pub const SUBBLOCK_MERGE_FLAG: usize = 60;
    /// `inter_affine_flag` (3 contexts).
    pub const AFFINE_FLAG: usize = 63;
    /// `cu_affine_type_flag` (1 context).
    pub const AFFINE_TYPE: usize = 66;
    /// `merge_subblock_idx` (1 context).
    pub const AFF_MERGE_IDX: usize = 67;
    /// `bcw_idx` (1 context).
    pub const BCW_IDX: usize = 68;
    /// `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` (2 contexts).
    pub const MVD: usize = 69;
    /// `intra_bdpcm_luma_flag` / `intra_bdpcm_luma_dir_flag` / chroma counterparts (4 contexts).
    pub const BDPCM_MODE: usize = 71;
    /// `cu_cbf` (1 context).
    pub const QT_ROOT_CBF: usize = 75;
    /// `tu_y_coded_flag` (4 contexts).
    pub const QT_CBF_Y: usize = 77;
    /// `tu_cb_coded_flag` (2 contexts).
    pub const QT_CBF_CB: usize = 81;
    /// `tu_cr_coded_flag` (3 contexts).
    pub const QT_CBF_CR: usize = 83;
    /// `sb_coded_flag` luma (2 contexts).
    pub const SIG_COEFF_GROUP_LUMA: usize = 86;
    /// `sb_coded_flag` chroma (2 contexts).
    pub const SIG_COEFF_GROUP_CHROMA: usize = 88;
    /// `sig_coeff_flag` luma state 0/1 (12 contexts).
    pub const SIG_FLAG_LUMA: usize = 90;
    /// `sig_coeff_flag` chroma state 0/1 (8 contexts).
    pub const SIG_FLAG_CHROMA: usize = 102;
    /// `sig_coeff_flag` luma state 2 (12 contexts).
    pub const SIG_FLAG_LUMA_1: usize = 110;
    /// `sig_coeff_flag` chroma state 2 (8 contexts).
    pub const SIG_FLAG_CHROMA_1: usize = 122;
    /// `sig_coeff_flag` luma state 3 (12 contexts).
    pub const SIG_FLAG_LUMA_2: usize = 130;
    /// `sig_coeff_flag` chroma state 3 (8 contexts).
    pub const SIG_FLAG_CHROMA_2: usize = 142;
    /// `par_level_flag` luma (21 contexts).
    pub const PAR_FLAG_LUMA: usize = 150;
    /// `par_level_flag` chroma (11 contexts).
    pub const PAR_FLAG_CHROMA: usize = 171;
    /// `abs_level_gtx_flag[0]` (gt2) luma (21 contexts).
    pub const GTX2_LUMA: usize = 182;
    /// `abs_level_gtx_flag[0]` (gt2) chroma (11 contexts).
    pub const GTX2_CHROMA: usize = 203;
    /// `abs_level_gtx_flag[1]` (gt1) luma (21 contexts).
    pub const GTX1_LUMA: usize = 214;
    /// `abs_level_gtx_flag[1]` (gt1) chroma (11 contexts).
    pub const GTX1_CHROMA: usize = 235;
    /// `last_sig_coeff_x_prefix` luma (20 contexts).
    pub const LAST_X_LUMA: usize = 246;
    /// `last_sig_coeff_x_prefix` chroma (3 contexts).
    pub const LAST_X_CHROMA: usize = 266;
    /// `last_sig_coeff_y_prefix` luma (20 contexts).
    pub const LAST_Y_LUMA: usize = 269;
    /// `last_sig_coeff_y_prefix` chroma (3 contexts).
    pub const LAST_Y_CHROMA: usize = 289;
    /// `mvp_l0_flag` / `mvp_l1_flag` (1 context).
    pub const MVP_IDX: usize = 292;
    /// `sym_mvd_flag` (1 context).
    pub const SMVD_FLAG: usize = 293;
    /// `sao_merge_left_flag` / `sao_merge_up_flag` (1 context).
    pub const SAO_MERGE_FLAG: usize = 294;
    /// `sao_type_idx_luma` / `sao_type_idx_chroma` (1 context).
    pub const SAO_TYPE_IDX: usize = 295;
    /// `lfnst_idx` (3 contexts).
    pub const LFNST_IDX: usize = 296;
    /// `transform_skip_flag` (2 contexts).
    pub const TRANSFORM_SKIP_FLAG: usize = 310;
    /// `mts_idx` (4 contexts).
    pub const MTS_IDX: usize = 312;
    /// `intra_subpartitions_mode_flag` / `intra_subpartitions_split_flag` (2 contexts).
    pub const ISP_MODE: usize = 316;
    /// `cu_sbt_flag` (2 contexts).
    pub const SBT_FLAG: usize = 318;
    /// `cu_sbt_quad_flag` (1 context).
    pub const SBT_QUAD_FLAG: usize = 320;
    /// `cu_sbt_horizontal_flag` (3 contexts).
    pub const SBT_HOR_FLAG: usize = 321;
    /// `cu_sbt_pos_flag` (1 context).
    pub const SBT_POS_FLAG: usize = 324;
    /// `cu_chroma_qp_offset_flag` (1 context).
    pub const CHROMA_QP_ADJ_FLAG: usize = 325;
    /// `cu_chroma_qp_offset_idx` (1 context).
    pub const CHROMA_QP_ADJ_IDC: usize = 326;
    /// `amvr_flag` / `amvr_precision_idx` (5 contexts).
    pub const IMV_FLAG: usize = 327;
    /// `alf_ctb_flag` (9 contexts).
    pub const ALF_CTB_FLAG: usize = 332;
    /// `alf_ctb_filter_alt_idx` (2 contexts).
    pub const CTB_ALF_ALTERNATIVE: usize = 341;
    /// `alf_use_aps_flag` (1 context).
    pub const ALF_USE_APS_FLAG: usize = 343;
    /// `alf_ctb_cc_cb_idc` / `alf_ctb_cc_cr_idc` (6 contexts).
    pub const CC_ALF_FILTER_CONTROL_FLAG: usize = 344;
    /// `ciip_flag` (1 context).
    pub const CIIP_FLAG: usize = 350;
    /// `pred_mode_ibc_flag` (3 contexts).
    pub const IBC_FLAG: usize = 351;
    /// `tu_joint_cbcr_residual_flag` (3 contexts).
    pub const JOINT_CB_CR_FLAG: usize = 354;
    /// `sb_coded_flag` transform-skip (3 contexts).
    pub const TS_SIG_COEFF_GROUP: usize = 357;
    /// `sig_coeff_flag` transform-skip (3 contexts).
    pub const TS_SIG_FLAG: usize = 360;
    /// `par_level_flag` transform-skip (1 context).
    pub const TS_PAR_FLAG: usize = 363;
    /// `abs_level_gtx_flag` transform-skip (5 contexts).
    pub const TS_GTX_FLAG: usize = 364;
    /// `abs_level_gt1_flag` transform-skip (4 contexts).
    pub const TS_LRG1_FLAG: usize = 369;
    /// `coeff_sign_flag` transform-skip (6 contexts).
    pub const TS_RESIDUAL_SIGN: usize = 373;
    /// `cu_act_enabled_flag` context.
    pub const ACT_FLAG: usize = 374;
    /// `palette_mode_flag` context (luma/chroma).
    pub const PLT_FLAG: usize = 375;
    /// `palette_transpose_flag` context.
    pub const PALETTE_TRANSPOSE_FLAG: usize = 376;
}

/// VVC arithmetic encoder.
pub struct BinEncoder {
    low: u32,
    range: u32,
    buffered_byte: u8,
    num_buffered_bytes: u32,
    bits_left: i32,
    out: BitWriter,
}

impl Default for BinEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BinEncoder {
    /// Create an encoder for one slice.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            buffered_byte: 0xFF,
            num_buffered_bytes: 0,
            bits_left: 23,
            out: BitWriter::with_capacity(1024),
        }
    }

    /// Flush the arithmetic coder and append `byte_alignment()`.
    pub fn finish_and_align(mut self) -> Vec<u8> {
        self.finish();
        self.out.byte_alignment();
        self.out.into_bytes()
    }

    /// Encode one context-coded bin.
    #[inline]
    pub fn encode_bin(&mut self, bin: bool, ctx: &mut CtxModel) {
        let lps_range = ctx.get_lps(self.range);
        let mps = ctx.mps();
        self.range -= lps_range;
        if bin != mps {
            let num_bits = RENORM_LPS[(lps_range >> 3) as usize];
            self.bits_left -= i32::from(num_bits);
            self.low = self.low.wrapping_add(self.range);
            self.low <<= num_bits;
            self.range = lps_range << num_bits;
            if self.bits_left < 12 {
                self.write_out();
            }
        } else if self.range < 256 {
            let num_bits = 1;
            self.bits_left -= num_bits;
            self.low <<= num_bits;
            self.range <<= num_bits;
            if self.bits_left < 12 {
                self.write_out();
            }
        }
        ctx.update(bin);
    }

    /// Encode one equiprobable bin.
    #[inline]
    pub fn encode_bin_ep(&mut self, bin: bool) {
        self.low <<= 1;
        if bin {
            self.low = self.low.wrapping_add(self.range);
        }
        self.bits_left -= 1;
        if self.bits_left < 12 {
            self.write_out();
        }
    }

    /// Equiprobable bins, MSB of `bins` first (H.266 `encode_bins_ep`).
    pub fn encode_bins_ep(&mut self, bins: u32, num_bins: u32) {
        if num_bins == 0 {
            return;
        }
        if self.range == 256 {
            self.encode_aligned_bins_ep(bins, num_bins);
            return;
        }
        let mut rem = num_bins;
        let mut bins = bins;
        while rem > 8 {
            rem -= 8;
            let pattern = (bins >> rem) & 0xFF;
            self.low <<= 8;
            self.low = self.low.wrapping_add(self.range.wrapping_mul(pattern));
            bins -= pattern << rem;
            self.bits_left -= 8;
            if self.bits_left < 12 {
                self.write_out();
            }
        }
        self.low <<= rem;
        self.low = self.low.wrapping_add(self.range.wrapping_mul(bins));
        self.bits_left -= rem as i32;
        if self.bits_left < 12 {
            self.write_out();
        }
    }

    fn encode_aligned_bins_ep(&mut self, bins: u32, num_bins: u32) {
        let mut rem = num_bins;
        while rem > 0 {
            let n = rem.min(8);
            let mask = (1u32 << n) - 1;
            let new_bins = (bins >> (rem - n)) & mask;
            self.low = (self.low << n) + (new_bins << 8);
            rem -= n;
            self.bits_left -= n as i32;
            if self.bits_left < 12 {
                self.write_out();
            }
        }
    }

    /// Encode a truncated binary (`TB`) symbol in `0..num_symbols` using bypass bins
    /// (`H.266 p. 429, PDF p. 445, §9.3.3.5`; VTM `CABACWriter.cpp:921-939`).
    pub fn encode_trunc_bin(&mut self, symbol: u32, num_symbols: u32) {
        if num_symbols <= 1 {
            return;
        }
        // H.266 §9.3.3.5 / VTM `CABACWriter::xWriteTruncBinCode`: the short
        // codes are the values below `val - b`, not below `b`.
        let thresh = 31 - num_symbols.leading_zeros();
        let val = 1u32 << thresh;
        let b = num_symbols - val;
        if symbol < val - b {
            self.encode_bins_ep(symbol, thresh);
        } else {
            self.encode_bins_ep(symbol + val - b, thresh + 1);
        }
    }

    /// Encode a truncated unary (`TU`) symbol in `0..=max_symbol` using bypass bins
    /// (`H.266 p. 426, PDF p. 442, §9.3.3.2`; VTM `CABACWriter.cpp:3268-3280`).
    pub fn encode_unary_max_ep(&mut self, symbol: u32, max_symbol: u32) {
        if max_symbol == 0 {
            return;
        }
        let ones = symbol.min(max_symbol);
        for _ in 0..ones {
            self.encode_bin_ep(true);
        }
        if symbol < max_symbol {
            self.encode_bin_ep(false);
        }
    }

    /// Encode a `k`-th order Exp-Golomb (`EGk`) code using bypass bins
    /// (`H.266 p. 427, PDF p. 443, §9.3.3.4`; VTM `CABACWriter.cpp:3292-3309`).
    pub fn encode_exp_golomb_ep(&mut self, mut symbol: u32, mut count: u32) {
        while symbol >= (1u32 << count) {
            self.encode_bin_ep(true);
            symbol -= 1u32 << count;
            count += 1;
        }
        self.encode_bin_ep(false);
        if count > 0 {
            self.encode_bins_ep(symbol, count);
        }
    }

    /// Terminate the arithmetic coder (`encode_bin_trm`).
    pub fn encode_bin_trm(&mut self, bin: bool) {
        self.range -= 2;
        if bin {
            self.low = self.low.wrapping_add(self.range);
            self.low <<= 7;
            self.range = 2 << 7;
            self.bits_left -= 7;
        } else if self.range >= 256 {
            return;
        } else {
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left -= 1;
        }
        if self.bits_left < 12 {
            self.write_out();
        }
    }

    /// `encode_rem_abs_ep` (H.266 9.3.4.2.3): truncated Rice plus Escape.
    pub fn encode_rem_abs_ep(&mut self, bins: u32, k: u32, cutoff: u32, max_log2_tr: u32) {
        let threshold = cutoff << k;
        if bins < threshold {
            let bit_mask = (1u32 << k) - 1;
            let length = (bins >> k) + 1;
            self.encode_bins_ep((1u32 << length) - 2, length);
            self.encode_bins_ep(bins & bit_mask, k);
        } else {
            let max_prefix_length = 32 - cutoff - max_log2_tr;
            let code_value = (bins >> k) - cutoff;
            let (prefix_length, suffix_length);
            if code_value >= (1u32 << max_prefix_length) - 1 {
                prefix_length = max_prefix_length;
                suffix_length = max_log2_tr;
            } else {
                let mut pl = 0u32;
                while code_value > ((2u32 << pl) - 2) {
                    pl += 1;
                }
                prefix_length = pl;
                suffix_length = pl + k + 1;
            }
            let total_prefix_length = prefix_length + cutoff;
            let bit_mask = (1u32 << k) - 1;
            let prefix = (1u32 << total_prefix_length) - 1;
            let suffix = ((code_value - ((1u32 << prefix_length) - 1)) << k) | (bins & bit_mask);
            self.encode_bins_ep(prefix, total_prefix_length);
            self.encode_bins_ep(suffix, suffix_length);
        }
    }

    fn write_out(&mut self) {
        let lead_byte = self.low >> (24 - self.bits_left);
        self.bits_left += 8;
        let keep = if self.bits_left >= 32 {
            0
        } else {
            0xFFFF_FFFFu32 >> self.bits_left
        };
        self.low &= keep;
        if lead_byte == 0xFF {
            self.num_buffered_bytes += 1;
        } else if self.num_buffered_bytes > 0 {
            let carry = lead_byte >> 8;
            let byte = u32::from(self.buffered_byte) + carry;
            self.buffered_byte = (lead_byte & 0xFF) as u8;
            self.out.write_bits(byte, 8);
            let fill = (0xFF + carry) & 0xFF;
            while self.num_buffered_bytes > 1 {
                self.out.write_bits(fill, 8);
                self.num_buffered_bytes -= 1;
            }
        } else {
            self.num_buffered_bytes = 1;
            self.buffered_byte = lead_byte as u8;
        }
    }

    fn finish(&mut self) {
        if self.bits_left < 32 && (self.low >> (32 - self.bits_left)) != 0 {
            self.out.write_bits(u32::from(self.buffered_byte) + 1, 8);
            while self.num_buffered_bytes > 1 {
                self.out.write_bits(0x00, 8);
                self.num_buffered_bytes -= 1;
            }
            self.low -= 1u32 << (32 - self.bits_left);
        } else {
            if self.num_buffered_bytes > 0 {
                self.out.write_bits(u32::from(self.buffered_byte), 8);
            }
            while self.num_buffered_bytes > 1 {
                self.out.write_bits(0xFF, 8);
                self.num_buffered_bytes -= 1;
            }
        }
        let n = (24 - self.bits_left) as u32;
        self.out.write_bits(self.low >> 8, n);
    }
}

/// VVC arithmetic decoder.
pub struct BinDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    value: u32,
    bits_needed: i32,
    last_byte: u8,
}

impl<'a> BinDecoder<'a> {
    /// Create a decoder over one slice's CABAC substream.
    ///
    /// `init_dec()` reads two bytes (`H.266 §9.3.4.3`). VTM's
    /// `InputBitstream::readByte` returns `0` once the substream is exhausted
    /// (`BinDecoder.cpp:92-99`), so a substream of one or zero bytes after the
    /// slice header is still well defined; it is zero-extended here instead of
    /// being rejected.
    pub fn new(data: &'a [u8]) -> Result<Self, &'static str> {
        let first = data.first().copied().unwrap_or(0);
        let second = data.get(1).copied().unwrap_or(0);
        Ok(Self {
            data,
            pos: data.len().min(2),
            range: 510,
            value: (u32::from(first) << 8) + u32::from(second),
            bits_needed: -8,
            last_byte: second,
        })
    }

    #[inline]
    fn read_byte(&mut self) -> u32 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            self.last_byte = b;
            u32::from(b)
        } else {
            0
        }
    }

    /// Decode one context-coded bin.
    #[inline]
    pub fn decode_bin(&mut self, ctx: &mut CtxModel) -> bool {
        let mps = ctx.mps();
        let lps_range = ctx.get_lps(self.range);
        self.range -= lps_range;
        let scaled_range = self.range << 7;
        let bin;
        if self.value < scaled_range {
            bin = mps;
            if self.range < 256 {
                self.range <<= 1;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed >= 0 {
                    self.value += self.read_byte() << (self.bits_needed as u32);
                    self.bits_needed -= 8;
                }
            }
        } else {
            bin = !mps;
            let num_bits = i32::from(RENORM_LPS[(lps_range >> 3) as usize]);
            self.value -= scaled_range;
            self.value <<= num_bits as u32;
            self.range = lps_range << num_bits;
            self.bits_needed += num_bits;
            if self.bits_needed >= 0 {
                self.value += self.read_byte() << (self.bits_needed as u32);
                self.bits_needed -= 8;
            }
        }
        ctx.update(bin);
        bin
    }

    /// Decode one equiprobable bin.
    #[inline]
    pub fn decode_bin_ep(&mut self) -> bool {
        self.value <<= 1;
        self.bits_needed += 1;
        if self.bits_needed >= 0 {
            self.value += self.read_byte();
            self.bits_needed = -8;
        }
        let scaled_range = self.range << 7;
        if self.value >= scaled_range {
            self.value -= scaled_range;
            true
        } else {
            false
        }
    }

    /// Decode `num_bins` equiprobable bins, MSB first.
    pub fn decode_bins_ep(&mut self, num_bins: u32) -> u32 {
        let mut bins = 0u32;
        for _ in 0..num_bins {
            bins = (bins << 1) | u32::from(self.decode_bin_ep());
        }
        bins
    }

    /// Decode a truncated binary (`TB`) symbol in `0..num_symbols`
    /// (`H.266 p. 429, PDF p. 445, §9.3.3.5`; VTM `CABACReader.cpp:1314-1325`).
    pub fn decode_trunc_bin(&mut self, num_symbols: u32) -> u32 {
        if num_symbols <= 1 {
            return 0;
        }
        let thresh = 31 - num_symbols.leading_zeros();
        let val = 1u32 << thresh;
        let b = num_symbols - val;
        let mut symbol = self.decode_bins_ep(thresh);
        if symbol >= val - b {
            symbol = 2 * symbol - (val - b) + u32::from(self.decode_bin_ep());
        }
        symbol
    }

    /// Decode a truncated unary (`TU`) symbol in `0..=max_symbol` using bypass bins.
    pub fn decode_unary_max_ep(&mut self, max_symbol: u32) -> u32 {
        let mut symbol = 0u32;
        while symbol < max_symbol && self.decode_bin_ep() {
            symbol += 1;
        }
        symbol
    }

    /// Decode a `k`-th order Exp-Golomb (`EGk`) code using bypass bins
    /// (`H.266 p. 427, PDF p. 443, §9.3.3.4`; VTM `CABACReader.cpp:2700-2715`).
    pub fn decode_exp_golomb_ep(&mut self, mut count: u32) -> Result<u32, &'static str> {
        let mut symbol = 0u32;
        let mut prefix_len = 0u32;
        while self.decode_bin_ep() {
            prefix_len += 1;
            if prefix_len > 30 || count >= 31 {
                return Err("EGk prefix overflow in CABAC stream");
            }
            symbol = symbol.checked_add(1u32 << count).ok_or("EGk overflow")?;
            count += 1;
        }
        if count > 0 {
            symbol = symbol
                .checked_add(self.decode_bins_ep(count))
                .ok_or("EGk suffix overflow")?;
        }
        Ok(symbol)
    }

    /// Decode the terminating bin.
    pub fn decode_bin_trm(&mut self) -> bool {
        self.range -= 2;
        let scaled_range = self.range << 7;
        if self.value >= scaled_range {
            true
        } else {
            if self.range < 256 {
                self.range <<= 1;
                self.value <<= 1;
                self.bits_needed += 1;
                if self.bits_needed == 0 {
                    self.value += self.read_byte();
                    self.bits_needed = -8;
                }
            }
            false
        }
    }

    /// Verify the stop/alignment pattern required by VTM's `BinDecoder::finish`.
    pub fn finish(&self) -> Result<(), &'static str> {
        let shift = 8 + self.bits_needed;
        if shift >= 0 && ((u32::from(self.last_byte) << shift) & 0xFF) != 0x80 {
            return Err("no proper stop/alignment pattern at end of CABAC stream");
        }
        Ok(())
    }

    /// End one `slice_data()` substream: decode `end_of_slice_one_bit` (or
    /// `end_of_subset_one_bit` at a tile/WPP substream boundary) followed by
    /// `byte_alignment()` (`H.266 §7.3.11.2`, §9.3.4.3).
    ///
    /// This is exactly VTM's `CABACReader::terminating_bit()`
    /// (`CABACReader.cpp:108-121`, called from `DecSlice.cpp:251` and
    /// `DecSlice.cpp:262`): the terminating bin must be 1 and the stop bit of
    /// the byte-aligned termination must be present. No further bit is
    /// required after the alignment, which is why the check is applied at the
    /// substream end only and never per coding unit.
    pub fn end_of_slice(&mut self) -> Result<(), &'static str> {
        if !self.decode_bin_trm() {
            return Err("missing CABAC end_of_slice_one_bit");
        }
        self.finish()
    }

    /// `decode_rem_abs_ep` (H.266 9.3.4.2.3).
    pub fn decode_rem_abs_ep(&mut self, go_rice_param: u32, cutoff: u32, max_log2_tr: u32) -> u32 {
        let max_prefix = 32 - max_log2_tr;
        let mut prefix = 0u32;
        let mut code_word;
        loop {
            prefix += 1;
            code_word = u32::from(self.decode_bin_ep());
            if !(code_word != 0 && prefix < max_prefix) {
                break;
            }
        }
        prefix -= 1 - code_word;

        let mut length = go_rice_param;
        let offset;
        if prefix < cutoff {
            offset = prefix << go_rice_param;
        } else {
            offset = ((1u32 << (prefix - cutoff)) + cutoff - 1) << go_rice_param;
            length += if prefix == max_prefix {
                max_log2_tr - go_rice_param
            } else {
                prefix - cutoff
            };
        }
        offset + self.decode_bins_ep(length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CtxModel {
        CtxModel::init(26, 25, 13)
    }

    #[test]
    fn round_trips_context_coded_and_bypass_bins() {
        let bins: Vec<bool> = (0..500).map(|i| ((i * 37 + i / 7) % 11) < 5).collect();
        let mut enc = BinEncoder::new();
        let mut enc_ctx = model();
        for &b in &bins {
            enc.encode_bin(b, &mut enc_ctx);
        }
        for &b in &bins[..100] {
            enc.encode_bin_ep(b);
        }
        enc.encode_bin_trm(true);
        let bytes = enc.finish_and_align();

        let mut dec = BinDecoder::new(&bytes).unwrap();
        let mut dec_ctx = model();
        for (i, &b) in bins.iter().enumerate() {
            assert_eq!(dec.decode_bin(&mut dec_ctx), b, "ctx bin {i}");
        }
        for &b in &bins[..100] {
            assert_eq!(dec.decode_bin_ep(), b);
        }
        assert!(dec.decode_bin_trm());
        dec.finish().unwrap();
    }

    #[test]
    fn truncated_binary_matches_vtm_long_code_convention() {
        // `xWriteTruncBinCode`: for numSymbols = 61, thresh = 5, val = 32,
        // b = 29, short codes are symbols < val - b = 3. Symbols >= 3 are
        // written as `symbol + 3` in six bins.
        for (symbol, code) in [(3u32, 6u32), (29, 32), (60, 63)] {
            let mut got = BinEncoder::new();
            got.encode_trunc_bin(symbol, 61);
            got.encode_bin_trm(true);
            let got = got.finish_and_align();

            let mut want = BinEncoder::new();
            want.encode_bins_ep(code, 6);
            want.encode_bin_trm(true);
            let want = want.finish_and_align();
            assert_eq!(got, want, "symbol {symbol} long code");
        }
        // A short code stays in five bins.
        let mut got = BinEncoder::new();
        got.encode_trunc_bin(2, 61);
        got.encode_bin_trm(true);
        let got = got.finish_and_align();
        let mut want = BinEncoder::new();
        want.encode_bins_ep(2, 5);
        want.encode_bin_trm(true);
        assert_eq!(got, want.finish_and_align());
    }

    #[test]
    fn round_trips_remainders() {
        for &value in &[0u32, 1, 7, 9, 10, 42, 1000, 40967] {
            let mut enc = BinEncoder::new();
            enc.encode_rem_abs_ep(value, 1, 5, 15);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();
            let mut dec = BinDecoder::new(&bytes).unwrap();
            assert_eq!(dec.decode_rem_abs_ep(1, 5, 15), value);
            assert!(dec.decode_bin_trm());
        }
    }

    #[test]
    fn initialises_every_i_slice_context() {
        let store = CtxStore::new_i_slice(26);
        assert_eq!(store.models.len(), NUM_CONTEXTS);
        let p_store = CtxStore::new_p_slice(26);
        let b_store = CtxStore::new_b_slice(26);
        assert_eq!(p_store.models.len(), NUM_CONTEXTS);
        assert_eq!(b_store.models.len(), NUM_CONTEXTS);
    }

    #[test]
    fn trunc_bin_unary_and_exp_golomb_round_trip() {
        let mut enc = BinEncoder::new();
        for s in 0..60u32 {
            enc.encode_trunc_bin(s, 60);
        }
        for u in 0..=5u32 {
            enc.encode_unary_max_ep(u, 5);
        }
        for eg in [0u32, 1, 2, 7, 19, 255] {
            enc.encode_exp_golomb_ep(eg, 1);
        }
        enc.encode_bin_trm(true);
        let bytes = enc.finish_and_align();

        let mut dec = BinDecoder::new(&bytes).unwrap();
        for s in 0..60u32 {
            assert_eq!(dec.decode_trunc_bin(60), s);
        }
        for u in 0..=5u32 {
            assert_eq!(dec.decode_unary_max_ep(5), u);
        }
        for eg in [0u32, 1, 2, 7, 19, 255] {
            assert_eq!(dec.decode_exp_golomb_ep(1).unwrap(), eg);
        }
        assert!(dec.decode_bin_trm());
        dec.finish().unwrap();
    }

    #[test]
    fn short_substream_is_zero_extended() {
        // `init_dec()` reads two bytes; VTM's `InputBitstream::readByte`
        // returns 0 past the end of a substream (`BinDecoder.cpp:92-99`), so a
        // one-byte CABAC substream is well defined.
        assert!(BinDecoder::new(&[]).is_ok());
        assert!(BinDecoder::new(&[0x00]).is_ok());
        let mut dec = BinDecoder::new(&[0x00]).expect("one-byte substream");
        // A substream with no coded bin still has to carry the terminating
        // bit, which this pattern does not.
        assert!(dec.end_of_slice().is_err());
        assert!(BinDecoder::new(&[0x00, 0x00]).is_ok());
    }
}
