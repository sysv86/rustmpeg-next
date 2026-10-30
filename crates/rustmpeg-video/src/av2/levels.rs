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

//! Clean-room safe Rust coding unit enumerations, partition geometries, transform sizes/types,
//! prediction modes, and motion vector types for the `av2` AV2 encoder.

pub const TIP_FRAME: usize = 7;
pub const INVALID_MV: i32 = 0x200000;

/// Metadata OBU payload type codes (AV2 Spec §6.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObuMetaType {
    HdrCll = 1,
    HdrMdcv = 2,
    Scalability = 3,
    ItutT35 = 4,
    Timecode = 5,
}

/// Square transform size index (`4x4` up to `64x64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum TxfmSize {
    #[default]
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
    Tx64x64 = 4,
}
pub const N_TX_SIZES: usize = 5;

/// Square and rectangular 2D transform size identifiers (`0..25`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum RectTxfmSize {
    #[default]
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
    Tx64x64 = 4,
    Rtx4x8 = 5,
    Rtx8x4 = 6,
    Rtx8x16 = 7,
    Rtx16x8 = 8,
    Rtx16x32 = 9,
    Rtx32x16 = 10,
    Rtx32x64 = 11,
    Rtx64x32 = 12,
    Rtx4x16 = 13,
    Rtx16x4 = 14,
    Rtx8x32 = 15,
    Rtx32x8 = 16,
    Rtx16x64 = 17,
    Rtx64x16 = 18,
    Rtx4x32 = 19,
    Rtx32x4 = 20,
    Rtx8x64 = 21,
    Rtx64x8 = 22,
    Rtx4x64 = 23,
    Rtx64x4 = 24,
}
pub const N_RECT_TX_SIZES: usize = 25;

impl RectTxfmSize {
    #[inline]
    pub const fn from_usize(v: usize) -> Option<Self> {
        if v < N_RECT_TX_SIZES {
            Some(match v {
                0 => Self::Tx4x4,
                1 => Self::Tx8x8,
                2 => Self::Tx16x16,
                3 => Self::Tx32x32,
                4 => Self::Tx64x64,
                5 => Self::Rtx4x8,
                6 => Self::Rtx8x4,
                7 => Self::Rtx8x16,
                8 => Self::Rtx16x8,
                9 => Self::Rtx16x32,
                10 => Self::Rtx32x16,
                11 => Self::Rtx32x64,
                12 => Self::Rtx64x32,
                13 => Self::Rtx4x16,
                14 => Self::Rtx16x4,
                15 => Self::Rtx8x32,
                16 => Self::Rtx32x8,
                17 => Self::Rtx16x64,
                18 => Self::Rtx64x16,
                19 => Self::Rtx4x32,
                20 => Self::Rtx32x4,
                21 => Self::Rtx8x64,
                22 => Self::Rtx64x8,
                23 => Self::Rtx4x64,
                _ => Self::Rtx64x4,
            })
        } else {
            None
        }
    }
}

/// 1D transform kernel family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tx1dType {
    Dct = 0,
    Adst = 1,
    FlipAdst = 2,
    Ddt = 3,
    FlipDdt = 4,
    Idtx = 5,
    Wht = 6,
}
pub const N_TX_1D_TYPES: usize = 7;

/// 2D transform scan/context class (`TwoD`, horizontal `H`, vertical `V`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxClass {
    TwoD = 0,
    H = 1,
    V = 2,
}

pub mod txtp {
    //! Packed transform-type identifiers:
    //! `horizontal_1d | (tx_class << 3) | (vertical_1d << 5)`, matching the
    //! reference bitstream semantics (`get_tx_class`, `get_scan`, `compute_tx_type`).
    //! The `*_DCT` / `DCT_*` names follow the vertical/horizontal convention:
    //! `ADST_DCT` is vertical ADST and horizontal DCT.
    pub const DCT_DCT: u32 = 0;
    pub const DCT_ADST: u32 = 2; // hor ADST
    pub const DCT_FLIPADST: u32 = 3;
    pub const DCT_DDT: u32 = 4;
    pub const DCT_FLIPDDT: u32 = 5;
    pub const ADST_DCT: u32 = 64; // ver ADST
    pub const ADST_ADST: u32 = 66;
    pub const ADST_FLIPADST: u32 = 67;
    pub const ADST_DDT: u32 = 68;
    pub const ADST_FLIPDDT: u32 = 69;
    pub const FLIPADST_DCT: u32 = 96;
    pub const FLIPADST_FLIPADST: u32 = 99;
    pub const FLIPADST_ADST: u32 = 98;
    pub const FLIPADST_DDT: u32 = 100;
    pub const FLIPADST_FLIPDDT: u32 = 101;
    pub const DDT_DCT: u32 = 128;
    pub const DDT_ADST: u32 = 130;
    pub const DDT_FLIPADST: u32 = 131;
    pub const DDT_DDT: u32 = 132;
    pub const DDT_FLIPDDT: u32 = 133;
    pub const FLIPDDT_DCT: u32 = 160;
    pub const FLIPDDT_ADST: u32 = 162;
    pub const FLIPDDT_FLIPADST: u32 = 163;
    pub const FLIPDDT_DDT: u32 = 164;
    pub const FLIPDDT_FLIPDDT: u32 = 165;
    pub const IDTX: u32 = 33; // hor identity, ver identity, 2D class
    pub const IDTX_INV: u32 = 41;
    pub const DDT_IDTX: u32 = 129;
    pub const IDTX_DDT: u32 = 36;
    pub const FLIPDDT_IDTX: u32 = 161;
    pub const IDTX_FLIPDDT: u32 = 37;
    pub const V_DCT: u32 = 25;
    pub const H_DCT: u32 = 48;
    pub const V_ADST: u32 = 89;
    pub const H_ADST: u32 = 50;
    pub const V_FLIPADST: u32 = 121;
    pub const H_FLIPADST: u32 = 51;
    pub const N_TX_TYPES: usize = 70;
    pub const WHT_WHT: u32 = 198;
    pub const N_TX_TYPES_PLUS_LL: usize = 71;
}

/// Primary luma/chroma spatial intra prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraPredMode {
    #[default]
    Dc = 0,
    Vert = 1,
    Hor = 2,
    DiagDownLeft = 3,
    DiagDownRight = 4,
    VertRight = 5,
    HorDown = 6,
    HorUp = 7,
    VertLeft = 8,
    Smooth = 9,
    SmoothV = 10,
    SmoothH = 11,
    Paeth = 12,
    CflOrDip = 13,
}

pub const N_INTRA_PRED_MODES: usize = 13;
pub const CFL_PRED: u8 = 13;
pub const N_UV_INTRA_PRED_MODES: usize = 14;
pub const LEFT_DC_PRED: u8 = 3;
pub const TOP_DC_PRED: u8 = 4;
pub const DC_128_PRED: u8 = 5;
pub const Z1_PRED: u8 = 6;
pub const Z2_PRED: u8 = 7;
pub const Z3_PRED: u8 = 8;
pub const DIP_PRED: u8 = 13;

/// 5-mode 7-tap Filter Intra modes (AV2 Spec §7.11.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FilterIntraMode {
    #[default]
    Dc = 0,
    Vertical = 1,
    Horizontal = 2,
    Diagonal = 3,
    Paeth = 4,
}
pub const N_FILTER_INTRA_MODES: usize = 5;

/// Superblock/coding-tree partition split modes (AV2 Spec §5.20.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlockPartition {
    #[default]
    None = 0,
    H = 1,
    V = 2,
    H3 = 3,
    V3 = 4,
    H4A = 5,
    H4B = 6,
    V4A = 7,
    V4B = 8,
    Split = 9,
}
pub const N_PARTITIONS: usize = 10;

impl BlockPartition {
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::H),
            2 => Some(Self::V),
            3 => Some(Self::H3),
            4 => Some(Self::V3),
            5 => Some(Self::H4A),
            6 => Some(Self::H4B),
            7 => Some(Self::V4A),
            8 => Some(Self::V4B),
            9 => Some(Self::Split),
            _ => None,
        }
    }

    /// Returns `true` for 3-way and uneven 4-way partitions (`H3`, `V3`, `H4A`, `H4B`, `V4A`, `V4B`).
    #[inline]
    pub const fn is_extended(&self) -> bool {
        matches!(
            self,
            Self::H3 | Self::V3 | Self::H4A | Self::H4B | Self::V4A | Self::V4B
        )
    }

    /// Returns `true` for vertical partitions (`V`, `V3`, `V4A`, `V4B`).
    #[inline]
    pub const fn is_vertical(&self) -> bool {
        matches!(self, Self::V | Self::V3 | Self::V4A | Self::V4B)
    }

    /// Returns `true` for horizontal partitions (`H`, `H3`, `H4A`, `H4B`).
    #[inline]
    pub const fn is_horizontal(&self) -> bool {
        matches!(self, Self::H | Self::H3 | Self::H4A | Self::H4B)
    }

    /// Returns `true` for uneven 4-way partitions (`H4A`, `H4B`, `V4A`, `V4B`).
    #[inline]
    pub const fn is_uneven_4way(&self) -> bool {
        matches!(self, Self::H4A | Self::H4B | Self::V4A | Self::V4B)
    }
}

/// Transform sub-partitioning modes within a coding block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxPartition {
    #[default]
    TxNone = 0,
    TxSplit = 1,
    TxH = 2,
    TxV = 3,
    TxH4 = 4,
    TxV4 = 5,
}

/// All 31 AV2 coding block sizes (`128x128` down to `4x4` including 1:2, 2:1, 1:4, 4:1, 1:8, 8:1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum BlockSize {
    #[default]
    Bs128x128 = 0,
    Bs128x64 = 1,
    Bs64x128 = 2,
    Bs64x64 = 3,
    Bs64x32 = 4,
    Bs32x64 = 5,
    Bs32x32 = 6,
    Bs32x16 = 7,
    Bs16x32 = 8,
    Bs16x16 = 9,
    Bs16x8 = 10,
    Bs8x16 = 11,
    Bs8x8 = 12,
    Bs8x4 = 13,
    Bs4x8 = 14,
    Bs4x4 = 15,
    Bs64x16 = 16,
    Bs16x64 = 17,
    Bs32x8 = 18,
    Bs8x32 = 19,
    Bs16x4 = 20,
    Bs4x16 = 21,
    Bs128x32 = 22,
    Bs32x128 = 23,
    Bs64x8 = 24,
    Bs8x64 = 25,
    Bs32x4 = 26,
    Bs4x32 = 27,
    Bs64x4 = 28,
    Bs4x64 = 29,
    Bs2x2 = 30,
}
pub const N_BS_SIZES: usize = 31;

impl BlockSize {
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Bs128x128),
            1 => Some(Self::Bs128x64),
            2 => Some(Self::Bs64x128),
            3 => Some(Self::Bs64x64),
            4 => Some(Self::Bs64x32),
            5 => Some(Self::Bs32x64),
            6 => Some(Self::Bs32x32),
            7 => Some(Self::Bs32x16),
            8 => Some(Self::Bs16x32),
            9 => Some(Self::Bs16x16),
            10 => Some(Self::Bs16x8),
            11 => Some(Self::Bs8x16),
            12 => Some(Self::Bs8x8),
            13 => Some(Self::Bs8x4),
            14 => Some(Self::Bs4x8),
            15 => Some(Self::Bs4x4),
            16 => Some(Self::Bs64x16),
            17 => Some(Self::Bs16x64),
            18 => Some(Self::Bs32x8),
            19 => Some(Self::Bs8x32),
            20 => Some(Self::Bs16x4),
            21 => Some(Self::Bs4x16),
            22 => Some(Self::Bs128x32),
            23 => Some(Self::Bs32x128),
            24 => Some(Self::Bs64x8),
            25 => Some(Self::Bs8x64),
            26 => Some(Self::Bs32x4),
            27 => Some(Self::Bs4x32),
            28 => Some(Self::Bs64x4),
            29 => Some(Self::Bs4x64),
            30 => Some(Self::Bs2x2),
            _ => None,
        }
    }
}

/// Single-reference inter prediction motion vector modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterPredMode {
    #[default]
    NearestMv = 0,
    NearMv = 1,
    GlobalMv = 2,
    NewMv = 3,
    ZeroMv = 4,
}

/// Compound (bi-predictive) inter prediction blending types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompInterType {
    #[default]
    Average = 0,
    DistanceWeighted = 1,
    DifferenceWeighted = 2,
    Wedge = 3,
}

/// Block motion model (`Translation`, `Obmc`, `Warp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MotionMode {
    #[default]
    Translation = 0,
    Obmc = 1,
    Warp = 2,
}

/// 2D motion vector in 1/8-pel or 1/16-pel units (`y` vertical, `x` horizontal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct MvXY {
    pub y: i16,
    pub x: i16,
}

impl MvXY {
    #[inline]
    pub const fn new(y: i16, x: i16) -> Self {
        Self { y, x }
    }

    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.y == 0 && self.x == 0
    }

    /// Manhattan distance between two motion vectors.
    #[inline]
    pub const fn l1_dist(&self, other: Self) -> u32 {
        (self.y as i32 - other.y as i32).unsigned_abs()
            + (self.x as i32 - other.x as i32).unsigned_abs()
    }
}

/// Safe encoder-side representation of a coded block's mode, transform, and motion decisions.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodingBlockInfo {
    pub bs: BlockSize,
    pub is_intra: bool,
    pub skip: bool,
    pub seg_id: u8,
    pub tx: RectTxfmSize,
    pub uv_tx: RectTxfmSize,
    pub txtp_y: u16,
    pub txtp_uv: u16,
    pub y_mode: u8,
    pub y_angle_delta: i8,
    pub mrl_idx: u8,
    pub filter_intra_mode: Option<FilterIntraMode>,
    pub uv_mode: u8,
    pub uv_angle_delta: i8,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub inter_mode: InterPredMode,
    pub drl_idx: u8,
    pub ref_frames: [i8; 2],
    pub comp_type: Option<CompInterType>,
    pub mv: [MvXY; 2],
    pub interp_filter: u8,
}
