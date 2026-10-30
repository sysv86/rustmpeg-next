// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! H.266 / VVC high-level syntax writers and parsers: AUD, OPI, DCI, VPS, SPS,
//! PPS, APS (ALF, LMCS, ScalingList), picture header, slice header, reference
//! picture list structures, and weighted prediction tables.
//!
//! Algorithms derived from VTM 24.0 (BSD-3-Clause, Copyright (c) 2010-2020
//! ITU/ISO/IEC). See THIRD_PARTY.md.
//!
//! Normative citations:
//! - ITU-T Rec. H.266 (V4) (01/2026) §7.3.1–7.3.10 (p. 37–65, PDF p. 53–81)
//! - ITU-T Rec. H.266 (V4) (01/2026) §7.4.2–7.4.11 (p. 89–163, PDF p. 105–179)
//! - VTM 24.0 `source/Lib/DecoderLib/VLCReader.cpp:315-5255`
//! - VTM 24.0 `source/Lib/EncoderLib/VLCWriter.cpp:228-3180`
//! - VTM 24.0 `source/Lib/CommonLib/SequenceParameterSet.cpp:144-185`

use super::VvcError;
use super::bits::{BitReader, BitWriter, rbsp_to_ebsp};

/// VVC `nal_unit_type` values (H.266 p. 90, PDF p. 106, §7.4.2.2 Table 5).
pub const NAL_TRAIL: u8 = 0;
/// Step-wise temporal sub-layer access picture (`STSA_NUT`).
pub const NAL_STSA: u8 = 1;
/// Random access decodable leading picture (`RADL_NUT`).
pub const NAL_RADL: u8 = 2;
/// Random access skipped leading picture (`RASL_NUT`).
pub const NAL_RASL: u8 = 3;
/// IDR picture with RADL pictures (`IDR_W_RADL`).
pub const NAL_IDR_W_RADL: u8 = 7;
/// IDR picture without leading pictures (`IDR_N_LP`).
pub const NAL_IDR_N_LP: u8 = 8;
/// Clean random access picture (`CRA_NUT`).
pub const NAL_CRA: u8 = 9;
/// Gradual decoding refresh picture (`GDR_NUT`).
pub const NAL_GDR: u8 = 10;
/// Operating point information (`OPI_NUT`).
pub const NAL_OPI: u8 = 12;
/// Decoding capability information (`DCI_NUT`).
pub const NAL_DCI: u8 = 13;
/// Video parameter set (`VPS_NUT`).
pub const NAL_VPS: u8 = 14;
/// Sequence parameter set (`SPS_NUT`).
pub const NAL_SPS: u8 = 15;
/// Picture parameter set (`PPS_NUT`).
pub const NAL_PPS: u8 = 16;
/// Prefix adaptation parameter set (`PREFIX_APS_NUT`).
pub const NAL_PREFIX_APS: u8 = 17;
/// Suffix adaptation parameter set (`SUFFIX_APS_NUT`).
pub const NAL_SUFFIX_APS: u8 = 18;
/// Picture header (`PH_NUT`).
pub const NAL_PH: u8 = 19;
/// Access unit delimiter (`AUD_NUT`).
pub const NAL_AUD: u8 = 20;
/// End of sequence (`EOS_NUT`).
pub const NAL_EOS: u8 = 21;
/// End of bitstream (`EOB_NUT`).
pub const NAL_EOB: u8 = 22;
/// Prefix SEI (`PREFIX_SEI_NUT`).
pub const NAL_PREFIX_SEI: u8 = 23;
/// Suffix SEI (`SUFFIX_SEI_NUT`).
pub const NAL_SUFFIX_SEI: u8 = 24;

/// Colour description carried in the VUI (BT.709, limited/studio range).
pub const BT709_PRIMARIES: u8 = 1;
/// BT.709 transfer characteristics code.
pub const BT709_TRANSFER: u8 = 1;
/// BT.709 matrix coefficients code.
pub const BT709_MATRIX: u8 = 1;

/// Slice coding type (`sh_slice_type`, H.266 p. 154, PDF p. 170, §7.4.8 Table 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcSliceType {
    /// Bi-predictive slice (`sh_slice_type == 0`).
    B = 0,
    /// Predictive slice (`sh_slice_type == 1`).
    P = 1,
    /// Intra slice (`sh_slice_type == 2`).
    I = 2,
}

impl VvcSliceType {
    /// Convert a raw `sh_slice_type` integer into [`VvcSliceType`].
    pub fn from_u32(val: u32) -> Result<Self, VvcError> {
        match val {
            0 => Ok(Self::B),
            1 => Ok(Self::P),
            2 => Ok(Self::I),
            _ => Err(VvcError::Bitstream(format!("invalid sh_slice_type {val}"))),
        }
    }
}

/// Static configuration of a VVC stream produced by this encoder.
#[derive(Debug, Clone)]
pub struct VvcStreamConfig {
    /// Coded picture width in luma samples.  Must be a multiple of the CTU size.
    pub width: u32,
    /// Coded picture height in luma samples.  Must be a multiple of the CTU size.
    pub height: u32,
    /// Right crop in luma samples (must be even); applied via the conformance window.
    pub crop_right: u32,
    /// Bottom crop in luma samples (must be even); applied via the conformance window.
    pub crop_bottom: u32,
    /// Slice QP (`4..=63`).
    pub qp: u8,
    /// `general_level_idc` (`16 * major + 3 * minor`, e.g. 51 = Level 3.1).
    pub level_idc: u8,
    /// `log2` of the CTU size (`5`, `6` or `7`).
    pub ctu_log2: u8,
    /// `log2` of the minimum luma coding block size (`2..=ctu_log2`).
    pub min_cb_log2: u8,
    /// `log2` of the maximum transform-skip block size (`2..=5`).
    pub ts_log2: u8,
}

/// Coding tools whose availability changes SPS/PPS/slice-header syntax.
///
/// Every flag corresponds to a normative enable flag in H.266 §7.3.2.4 (SPS),
/// §7.3.2.6 (PPS) or §7.3.7 (slice header).  A tool that is enabled here is
/// assumed to be signalled by the slice-data writer as well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VvcHlsTools {
    /// Emit inter slices (`gci_intra_only_constraint_flag = 0`, TRAIL allowed).
    pub inter: bool,
    /// `sps_mts_enabled_flag` and explicit intra MTS (H.266 §7.4.3.4).
    pub mts: bool,
    /// `sps_lfnst_enabled_flag` (H.266 §7.4.3.4).
    pub lfnst: bool,
    /// `sps_transform_skip_enabled_flag` (H.266 §7.4.3.4).
    pub transform_skip: bool,
    /// `sps_bdpcm_enabled_flag` (H.266 §7.4.3.4).
    pub bdpcm: bool,
    /// `sps_cclm_enabled_flag` (H.266 §7.4.3.4).
    pub cclm: bool,
    /// `sps_mip_enabled_flag` (H.266 §7.4.3.4).
    pub mip: bool,
    /// `sps_isp_enabled_flag` (H.266 §7.4.3.4).
    pub isp: bool,
    /// `sps_mrl_enabled_flag` (H.266 §7.4.3.4).
    pub mrl: bool,
    /// `sps_joint_cbcr_enabled_flag` (H.266 §7.4.3.4).
    pub joint_cbcr: bool,
    /// `sps_dep_quant_enabled_flag` (H.266 §7.4.3.4).
    pub dep_quant: bool,
    /// `sps_sign_data_hiding_enabled_flag` (H.266 §7.4.3.4).
    pub sign_data_hiding: bool,
    /// `sps_sao_enabled_flag` (H.266 §7.4.3.4).
    pub sao: bool,
    /// `sps_alf_enabled_flag` (H.266 §7.4.3.4).
    pub alf: bool,
    /// Enable the deblocking filter (`pps_deblocking_filter_disabled_flag = 0`).
    pub deblock: bool,
}

impl Default for VvcHlsTools {
    /// Defaults match the strict intra subset: transform skip, MTS, and LFNST
    /// are all advertised and signalled.
    fn default() -> Self {
        Self {
            inter: false,
            mts: true,
            lfnst: true,
            transform_skip: true,
            bdpcm: false,
            cclm: false,
            mip: false,
            isp: false,
            mrl: false,
            joint_cbcr: false,
            dep_quant: false,
            sign_data_hiding: false,
            sao: false,
            alf: false,
            deblock: false,
        }
    }
}

/// Choose the smallest VVC level whose `MaxLumaPs` covers `width * height`.
///
/// Limits from H.266 Annex A, Table A.1 (p. 457, PDF p. 473; also
/// `ProfileTierLevel.cpp` in VTM).
pub fn select_level_idc(width: u32, height: u32) -> u8 {
    let samples = u64::from(width) * u64::from(height);
    // (level_idc, max luma samples per picture)
    const LEVELS: [(u8, u64); 13] = [
        (16, 36_864),      // 1
        (32, 122_880),     // 2
        (35, 245_760),     // 2.1
        (48, 552_960),     // 3
        (51, 983_040),     // 3.1
        (64, 2_228_224),   // 4
        (67, 2_228_224),   // 4.1
        (80, 8_912_896),   // 5
        (83, 8_912_896),   // 5.1
        (86, 8_912_896),   // 5.2
        (96, 35_651_584),  // 6
        (99, 35_651_584),  // 6.1
        (102, 35_651_584), // 6.2
    ];
    for (idc, max) in LEVELS {
        if samples <= max {
            return idc;
        }
    }
    102
}

/// `ceil(log2(x))` for syntax element bit widths (`Ceil(Log2(x))` in H.266 §5.8).
#[inline]
pub fn ceil_log2(val: u32) -> u32 {
    if val <= 1 {
        0
    } else {
        32 - (val - 1).leading_zeros()
    }
}

/// `profile_tier_level(profileTierPresentFlag = 1, maxNumSubLayersMinus1 = 0)`
/// with `general_constraints_info()` present (H.266 p. 55, PDF p. 71, §7.3.3.1).
pub fn write_profile_tier_level(bw: &mut BitWriter, level_idc: u8, ctu_log2: u8) {
    write_profile_tier_level_with_tools(bw, level_idc, ctu_log2, &VvcHlsTools::default());
}

/// `profile_tier_level()` with constraint flags matching `tools`.
fn write_profile_tier_level_with_tools(
    bw: &mut BitWriter,
    level_idc: u8,
    ctu_log2: u8,
    tools: &VvcHlsTools,
) {
    // Main 10 profile, Main tier.
    bw.write_bits(1, 7); // general_profile_idc = 1 (Main 10)
    bw.write_bit(false); // general_tier_flag
    bw.write_bits(u32::from(level_idc), 8); // general_level_idc
    bw.write_bit(true); // ptl_frame_only_constraint_flag
    bw.write_bit(false); // ptl_multilayer_enabled_flag
    write_general_constraints_info(bw, ctu_log2, tools);
    // No sub-layer level flags (maxNumSubLayersMinus1 == 0).
    bw.align_zero(); // ptl_reserved_zero_bit
    bw.write_bits(0, 8); // ptl_num_sub_profiles
}

/// `gci_three_minus_max_log2_ctu_size_constraint_idc` for a CTU `log2`.
///
/// The field is `3 - (log2_ctu_size - 5)` and is only defined for a 32..=128
/// CTU.  Clamping keeps a caller-supplied out-of-range value from panicking;
/// this encoder always writes 5 (32x32).
fn ctu_size_constraint_idc(ctu_log2: u8) -> u32 {
    8 - u32::from(ctu_log2.clamp(5, 7))
}

/// `general_constraints_info()` for the configuration this encoder emits
/// (H.266 p. 56–57, PDF p. 72–73, §7.3.3.2; VTM `VLCWriter.cpp:2655-2770`).
fn write_general_constraints_info(bw: &mut BitWriter, ctu_log2: u8, tools: &VvcHlsTools) {
    bw.write_bit(true); // gci_present_flag

    // general
    bw.write_bit(!tools.inter); // gci_intra_only_constraint_flag
    bw.write_bit(true); // gci_all_layers_independent_constraint_flag
    bw.write_bit(false); // gci_one_au_only_constraint_flag

    // picture format
    bw.write_bits(8, 4); // gci_sixteen_minus_max_bitdepth_constraint_idc (8-bit)
    bw.write_bits(2, 2); // gci_three_minus_max_chroma_format_constraint_idc (4:2:0)

    // NAL unit type related
    bw.write_bit(true); // gci_no_mixed_nalu_types_in_pic_constraint_flag
    bw.write_bit(!tools.inter); // gci_no_trail_constraint_flag
    bw.write_bit(true); // gci_no_stsa_constraint_flag
    bw.write_bit(true); // gci_no_rasl_constraint_flag
    bw.write_bit(true); // gci_no_radl_constraint_flag
    bw.write_bit(false); // gci_no_idr_constraint_flag (we use IDR)
    bw.write_bit(true); // gci_no_cra_constraint_flag
    bw.write_bit(true); // gci_no_gdr_constraint_flag
    bw.write_bit(false); // gci_no_aps_constraint_flag (ALF/LMCS APS may be used)
    bw.write_bit(true); // gci_no_idr_rpl_constraint_flag

    // tile, slice, subpicture partitioning
    bw.write_bit(true); // gci_one_tile_per_pic_constraint_flag
    bw.write_bit(false); // gci_pic_header_in_slice_header_constraint_flag (PH NAL used)
    bw.write_bit(true); // gci_one_slice_per_pic_constraint_flag
    bw.write_bit(false); // gci_no_rectangular_slice_constraint_flag
    bw.write_bit(false); // gci_one_slice_per_subpic_constraint_flag
    bw.write_bit(true); // gci_no_subpic_info_constraint_flag

    // CTU and block partitioning
    bw.write_bits(ctu_size_constraint_idc(ctu_log2), 2); // gci_three_minus_max_log2_ctu_size_constraint_idc
    bw.write_bit(true); // gci_no_partition_constraints_override_constraint_flag
    bw.write_bit(true); // gci_no_mtt_constraint_flag
    bw.write_bit(true); // gci_no_qtbtt_dual_tree_intra_constraint_flag

    // intra
    bw.write_bit(true); // gci_no_palette_constraint_flag
    bw.write_bit(true); // gci_no_ibc_constraint_flag
    bw.write_bit(!tools.isp); // gci_no_isp_constraint_flag
    bw.write_bit(!tools.mrl); // gci_no_mrl_constraint_flag
    bw.write_bit(!tools.mip); // gci_no_mip_constraint_flag
    bw.write_bit(!tools.cclm); // gci_no_cclm_constraint_flag

    // inter
    bw.write_bit(true); // gci_no_ref_pic_resampling_constraint_flag
    bw.write_bit(true); // gci_no_res_change_in_clvs_constraint_flag
    bw.write_bit(true); // gci_no_weighted_prediction_constraint_flag
    bw.write_bit(true); // gci_no_ref_wraparound_constraint_flag
    bw.write_bit(true); // gci_no_temporal_mvp_constraint_flag
    bw.write_bit(true); // gci_no_sbtmvp_constraint_flag
    bw.write_bit(true); // gci_no_amvr_constraint_flag
    bw.write_bit(true); // gci_no_bdof_constraint_flag
    bw.write_bit(true); // gci_no_smvd_constraint_flag
    bw.write_bit(true); // gci_no_dmvr_constraint_flag
    bw.write_bit(true); // gci_no_mmvd_constraint_flag
    bw.write_bit(true); // gci_no_affine_motion_constraint_flag
    bw.write_bit(true); // gci_no_prof_constraint_flag
    bw.write_bit(true); // gci_no_bcw_constraint_flag
    bw.write_bit(true); // gci_no_ciip_constraint_flag
    bw.write_bit(true); // gci_no_gpm_constraint_flag

    // transform, quantisation, residual
    bw.write_bit(true); // gci_no_luma_transform_size_64_constraint_flag
    bw.write_bit(!tools.transform_skip); // gci_no_transform_skip_constraint_flag
    bw.write_bit(true); // gci_no_bdpcm_constraint_flag
    bw.write_bit(!tools.mts); // gci_no_mts_constraint_flag
    bw.write_bit(!tools.lfnst); // gci_no_lfnst_constraint_flag
    bw.write_bit(!tools.joint_cbcr); // gci_no_joint_cbcr_constraint_flag
    bw.write_bit(true); // gci_no_sbt_constraint_flag
    bw.write_bit(true); // gci_no_act_constraint_flag
    bw.write_bit(true); // gci_no_explicit_scaling_list_constraint_flag
    bw.write_bit(!tools.dep_quant); // gci_no_dep_quant_constraint_flag
    bw.write_bit(!tools.sign_data_hiding); // gci_no_sign_data_hiding_constraint_flag
    bw.write_bit(true); // gci_no_cu_qp_delta_constraint_flag
    bw.write_bit(true); // gci_no_chroma_qp_offset_constraint_flag

    // loop filter
    bw.write_bit(!tools.sao); // gci_no_sao_constraint_flag
    bw.write_bit(!tools.alf); // gci_no_alf_constraint_flag
    bw.write_bit(true); // gci_no_ccalf_constraint_flag
    bw.write_bit(true); // gci_no_lmcs_constraint_flag
    bw.write_bit(true); // gci_no_ladf_constraint_flag
    bw.write_bit(true); // gci_no_virtual_boundaries_constraint_flag

    bw.write_bits(0, 8); // gci_num_additional_bits

    bw.align_zero(); // gci_alignment_zero_bit
}

/// Build the VPS RBSP for a single-layer stream with `vps_video_parameter_set_id = 1`
/// (H.266 p. 38, PDF p. 54, §7.3.2.3).
pub fn build_vps_rbsp(cfg: &VvcStreamConfig) -> Vec<u8> {
    build_vps_rbsp_with_tools(cfg, &VvcHlsTools::default())
}

/// Build the VPS RBSP with constraint flags matching `tools`.
pub fn build_vps_rbsp_with_tools(cfg: &VvcStreamConfig, tools: &VvcHlsTools) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(24);
    bw.write_bits(1, 4); // vps_video_parameter_set_id = 1
    bw.write_bits(0, 6); // vps_max_layers_minus1
    bw.write_bits(0, 3); // vps_max_sublayers_minus1
    bw.write_bits(0, 6); // vps_layer_id[0]
    // vps_ptl_reserved_zero_bit: zeros until byte aligned (19 -> 24 bits).
    bw.align_zero();
    write_profile_tier_level_with_tools(&mut bw, cfg.level_idc, cfg.ctu_log2, tools);
    bw.write_bit(false); // vps_extension_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Build the SPS RBSP (H.266 p. 38–42, PDF p. 54–58, §7.3.2.4).
pub fn build_sps_rbsp(cfg: &VvcStreamConfig) -> Vec<u8> {
    build_sps_rbsp_with_tools(cfg, &VvcHlsTools::default())
}

/// Build the SPS RBSP with tool availability flags (H.266 §7.3.2.4).
pub fn build_sps_rbsp_with_tools(cfg: &VvcStreamConfig, tools: &VvcHlsTools) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(48);
    bw.write_bits(0, 4); // sps_seq_parameter_set_id
    bw.write_bits(1, 4); // sps_video_parameter_set_id
    bw.write_bits(0, 3); // sps_max_sub_layers_minus1
    bw.write_bits(1, 2); // sps_chroma_format_idc = 1 (4:2:0)
    assert!(
        (5..=7).contains(&cfg.ctu_log2),
        "sps_log2_ctu_size_minus5 requires ctu_log2 in 5..=7, got {}",
        cfg.ctu_log2
    );
    bw.write_bits(u32::from(cfg.ctu_log2 - 5), 2); // sps_log2_ctu_size_minus5
    bw.write_bit(true); // sps_ptl_dpb_hrd_params_present_flag
    write_profile_tier_level_with_tools(&mut bw, cfg.level_idc, cfg.ctu_log2, tools);

    bw.write_bit(false); // sps_gdr_enabled_flag
    bw.write_bit(false); // sps_ref_pic_resampling_enabled_flag
    bw.write_ue(cfg.width); // sps_pic_width_max_in_luma_samples
    bw.write_ue(cfg.height); // sps_pic_height_max_in_luma_samples
    let crop = cfg.crop_right > 0 || cfg.crop_bottom > 0;
    bw.write_bit(crop); // sps_conformance_window_flag
    if crop {
        assert!(cfg.crop_right.is_multiple_of(2) && cfg.crop_bottom.is_multiple_of(2));
        bw.write_ue(0); // sps_conf_win_left_offset
        bw.write_ue(cfg.crop_right / 2); // sps_conf_win_right_offset
        bw.write_ue(0); // sps_conf_win_top_offset
        bw.write_ue(cfg.crop_bottom / 2); // sps_conf_win_bottom_offset
    }
    bw.write_bit(false); // sps_subpic_info_present_flag
    bw.write_ue(0); // sps_bitdepth_minus8 (8-bit)
    bw.write_bit(false); // sps_entropy_coding_sync_enabled_flag
    bw.write_bit(false); // sps_entry_point_offsets_present_flag
    bw.write_bits(4, 4); // sps_log2_max_pic_order_cnt_lsb_minus4 => 8-bit POC LSB
    bw.write_bit(false); // sps_poc_msb_cycle_flag
    bw.write_bits(0, 2); // sps_num_extra_ph_bytes
    bw.write_bits(0, 2); // sps_num_extra_sh_bytes

    // dpb_parameters(maxSubLayersMinus1 = 0, ...)
    bw.write_ue(0); // dpb_max_dec_pic_buffering_minus1[0]
    bw.write_ue(0); // dpb_max_num_reorder_pics[0]
    bw.write_ue(0); // dpb_max_latency_increase_plus1[0]

    bw.write_ue(u32::from(cfg.min_cb_log2 - 2)); // sps_log2_min_luma_coding_block_size_minus2
    bw.write_bit(false); // sps_partition_constraints_override_enabled_flag
    bw.write_ue(0); // sps_log2_diff_min_qt_min_cb_intra_slice_luma
    bw.write_ue(0); // sps_max_mtt_hierarchy_depth_intra_slice_luma
    bw.write_bit(false); // sps_qtbtt_dual_tree_intra_flag
    bw.write_ue(0); // sps_log2_diff_min_qt_min_cb_inter_slice
    bw.write_ue(0); // sps_max_mtt_hierarchy_depth_inter_slice
    if cfg.ctu_log2 > 5 {
        bw.write_bit(false); // sps_max_luma_transform_size_64_flag (max TB 32x32)
    }

    bw.write_bit(tools.transform_skip); // sps_transform_skip_enabled_flag
    if tools.transform_skip {
        bw.write_ue(u32::from(cfg.ts_log2 - 2)); // sps_log2_transform_skip_max_size_minus2
        bw.write_bit(tools.bdpcm); // sps_bdpcm_enabled_flag
    }
    bw.write_bit(tools.mts); // sps_mts_enabled_flag
    if tools.mts {
        bw.write_bit(tools.mts); // sps_explicit_mts_intra_enabled_flag
        bw.write_bit(false); // sps_explicit_mts_inter_enabled_flag
    }
    bw.write_bit(tools.lfnst); // sps_lfnst_enabled_flag

    bw.write_bit(tools.joint_cbcr); // sps_joint_cbcr_enabled_flag
    bw.write_bit(true); // sps_same_qp_table_for_chroma_flag
    bw.write_se(0); // sps_qp_table_starts_minus26
    bw.write_ue(0); // sps_num_points_in_qp_table_minus1
    bw.write_ue(0); // sps_delta_qp_in_val_minus1[0][0]
    bw.write_ue(0); // sps_delta_qp_diff_val[0][0]

    bw.write_bit(tools.sao); // sps_sao_enabled_flag
    bw.write_bit(tools.alf); // sps_alf_enabled_flag
    if tools.alf {
        bw.write_bit(false); // sps_ccalf_enabled_flag
    }
    bw.write_bit(false); // sps_lmcs_enable_flag
    bw.write_bit(false); // sps_weighted_pred_flag
    bw.write_bit(false); // sps_weighted_bipred_flag
    bw.write_bit(false); // sps_long_term_ref_pics_flag
    bw.write_bit(false); // sps_inter_layer_prediction_enabled_flag (vps_id > 0)
    bw.write_bit(false); // sps_idr_rpl_present_flag
    bw.write_bit(true); // sps_rpl1_same_as_rpl0_flag
    bw.write_ue(0); // sps_num_ref_pic_lists[0]
    bw.write_bit(false); // sps_ref_wraparound_enabled_flag
    bw.write_bit(false); // sps_temporal_mvp_enabled_flag
    bw.write_bit(false); // sps_amvr_enabled_flag
    bw.write_bit(false); // sps_bdof_enabled_flag
    bw.write_bit(false); // sps_smvd_enabled_flag
    bw.write_bit(false); // sps_dmvr_enabled_flag
    bw.write_bit(false); // sps_mmvd_enabled_flag
    bw.write_ue(5); // sps_six_minus_max_num_merge_cand => MaxNumMergeCand == 1
    bw.write_bit(false); // sps_sbt_enabled_flag
    bw.write_bit(false); // sps_affine_enabled_flag
    bw.write_bit(false); // sps_bcw_enabled_flag
    bw.write_bit(false); // sps_ciip_enabled_flag
    bw.write_ue(0); // sps_log2_parallel_merge_level_minus2
    bw.write_bit(tools.isp); // sps_isp_enabled_flag
    bw.write_bit(tools.mrl); // sps_mrl_enabled_flag
    bw.write_bit(tools.mip); // sps_mip_enabled_flag
    bw.write_bit(tools.cclm); // sps_cclm_enabled_flag
    bw.write_bit(false); // sps_chroma_horizontal_collocated_flag
    bw.write_bit(false); // sps_chroma_vertical_collocated_flag
    bw.write_bit(false); // sps_palette_enabled_flag
    if tools.transform_skip {
        bw.write_ue(0); // sps_internal_bit_depth_minus_input_bit_depth
    }
    bw.write_bit(false); // sps_ibc_enabled_flag
    bw.write_bit(false); // sps_ladf_enabled_flag
    bw.write_bit(false); // sps_explicit_scaling_list_enabled_flag
    bw.write_bit(tools.dep_quant); // sps_dep_quant_enabled_flag
    bw.write_bit(tools.sign_data_hiding); // sps_sign_data_hiding_enabled_flag
    bw.write_bit(false); // sps_virtual_boundaries_enabled_flag
    bw.write_bit(false); // sps_timing_hrd_params_present_flag
    bw.write_bit(false); // sps_field_seq_flag
    bw.write_bit(true); // sps_vui_parameters_present_flag
    bw.write_ue(4); // sps_vui_payload_size_minus1 => 5-byte payload
    bw.align_zero(); // sps_vui_alignment_zero_bit
    let vui = build_vui_payload();
    assert_eq!(vui.len(), 5);
    bw.append_bytes(&vui);
    bw.write_bit(false); // sps_extension_present_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// 5-byte `vui_parameters()` payload: progressive BT.709 limited-range 4:2:0.
fn build_vui_payload() -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(8);
    bw.write_bit(true); // vui_progressive_source_flag
    bw.write_bit(false); // vui_interlaced_source_flag
    bw.write_bit(true); // vui_non_packed_constraint_flag
    bw.write_bit(true); // vui_non_projected_constraint_flag
    bw.write_bit(false); // vui_aspect_ratio_info_present_flag
    bw.write_bit(false); // vui_overscan_info_present_flag
    bw.write_bit(true); // vui_colour_description_present_flag
    bw.write_bits(u32::from(BT709_PRIMARIES), 8); // vui_colour_primaries
    bw.write_bits(u32::from(BT709_TRANSFER), 8); // vui_transfer_characteristics
    bw.write_bits(u32::from(BT709_MATRIX), 8); // vui_matrix_coeffs
    bw.write_bit(false); // vui_full_range_flag (studio swing)
    bw.write_bit(true); // vui_chroma_loc_info_present_flag
    bw.write_ue(0); // vui_chroma_sample_loc_type (left / co-sited horizontally)
    bw.write_bit(true);
    while !bw.is_byte_aligned() {
        bw.write_bit(false);
    }
    bw.into_bytes()
}

/// Build the PPS RBSP (`pps_no_pic_partition_flag = 1`, deblocking disabled).
pub fn build_pps_rbsp(cfg: &VvcStreamConfig) -> Vec<u8> {
    build_pps_rbsp_with_tools(cfg, &VvcHlsTools::default())
}

/// Build the PPS RBSP with tool availability flags (H.266 §7.3.2.6).
pub fn build_pps_rbsp_with_tools(cfg: &VvcStreamConfig, tools: &VvcHlsTools) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(24);
    bw.write_bits(0, 6); // pps_pic_parameter_set_id
    bw.write_bits(0, 4); // pps_seq_parameter_set_id
    bw.write_bit(false); // pps_mixed_nalu_types_in_pic_flag
    bw.write_ue(cfg.width); // pps_pic_width_in_luma_samples
    bw.write_ue(cfg.height); // pps_pic_height_in_luma_samples
    bw.write_bit(false); // pps_conformance_window_flag
    bw.write_bit(false); // pps_scaling_window_explicit_signalling_flag
    bw.write_bit(false); // pps_output_flag_present_flag
    bw.write_bit(true); // pps_no_pic_partition_flag
    bw.write_bit(false); // pps_subpic_id_mapping_present_flag
    bw.write_bit(false); // pps_cabac_init_present_flag
    bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[0]
    bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[1]
    bw.write_bit(false); // pps_rpl1_idx_present_flag
    bw.write_bit(false); // pps_weighted_pred_flag
    bw.write_bit(false); // pps_weighted_bipred_flag
    bw.write_bit(false); // pps_ref_wraparound_enabled_flag
    bw.write_se(i32::from(cfg.qp) - 26); // pps_init_qp_minus26
    bw.write_bit(false); // pps_cu_qp_delta_enabled_flag
    bw.write_bit(false); // pps_chroma_tool_offsets_present_flag
    bw.write_bit(true); // pps_deblocking_filter_control_present_flag
    bw.write_bit(false); // pps_deblocking_filter_override_enabled_flag
    bw.write_bit(!tools.deblock); // pps_deblocking_filter_disabled_flag
    if tools.deblock {
        bw.write_se(0); // pps_beta_offset_div2
        bw.write_se(0); // pps_tc_offset_div2
    }
    bw.write_bit(false); // pps_picture_header_extension_present_flag
    bw.write_bit(false); // pps_slice_header_extension_present_flag
    bw.write_bit(false); // pps_extension_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Build the picture header RBSP.
pub fn build_ph_rbsp(poc_lsb: u32) -> Vec<u8> {
    build_ph_rbsp_with_tools(poc_lsb, &VvcHlsTools::default())
}

/// Build the picture header RBSP for an IRAP picture with tool flags (`ph_joint_cbcr_sign_flag`).
pub fn build_ph_rbsp_with_tools(poc_lsb: u32, tools: &VvcHlsTools) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(4);
    bw.write_bit(true); // ph_gdr_or_irap_pic_flag
    bw.write_bit(false); // ph_non_ref_pic_flag
    bw.write_bit(false); // ph_gdr_pic_flag (IRAP, not GDR)
    bw.write_bit(false); // ph_inter_slice_allowed_flag
    bw.write_ue(0); // ph_pic_parameter_set_id
    bw.write_bits(poc_lsb & 0xFF, 8); // ph_pic_order_cnt_lsb
    if tools.joint_cbcr {
        bw.write_bit(false); // ph_joint_cbcr_sign_flag
    }
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Build the slice header bits (without `slice_data()`), including the
/// trailing `byte_alignment()`.
pub fn build_slice_header_bits(slice_qp_delta: i32) -> Vec<u8> {
    build_slice_header_bits_with_tools(slice_qp_delta, &VvcHlsTools::default(), false, false)
}

/// Build an intra-slice header for a stream whose SPS enables `tools`
/// (H.266 §7.3.7).
///
/// `dep_quant` and `sign_data_hiding` are the values actually used by the
/// slice-data writer; they must be consistent with the SPS enable flags carried
/// in `tools`.
pub fn build_slice_header_bits_with_tools(
    slice_qp_delta: i32,
    tools: &VvcHlsTools,
    dep_quant: bool,
    sign_data_hiding: bool,
) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(8);
    bw.write_bit(false); // sh_picture_header_in_slice_header_flag (PH NAL used)
    bw.write_bit(false); // sh_no_output_of_prior_pics_flag (IDR_N_LP)
    if tools.alf {
        // pps_alf_info_in_ph_flag is 0, so the slice carries the ALF flag.
        bw.write_bit(true); // sh_alf_enabled_flag = 1 (uses fixed filter sets, num_aps = 0)
        bw.write_bits(0, 3); // sh_num_alf_aps_ids_luma = 0
        bw.write_bit(false); // sh_alf_cb_enabled_flag = 0
        bw.write_bit(false); // sh_alf_cr_enabled_flag = 0
    }
    bw.write_se(slice_qp_delta); // sh_qp_delta
    if tools.sao {
        // pps_sao_info_in_ph_flag is 0, so the slice carries both SAO flags.
        bw.write_bit(true); // sh_sao_luma_used_flag
        bw.write_bit(true); // sh_sao_chroma_used_flag
    }
    // pps_deblocking_filter_override_enabled_flag is 0, so no slice-level
    // deblocking parameters are present.
    if tools.dep_quant {
        bw.write_bit(dep_quant); // sh_dep_quant_used_flag
    }
    if tools.sign_data_hiding && !dep_quant {
        bw.write_bit(sign_data_hiding); // sh_sign_data_hiding_used_flag
    }
    if !dep_quant && !sign_data_hiding {
        bw.write_bit(false); // sh_ts_residual_coding_disabled_flag = 0
    }
    bw.byte_alignment();
    bw.into_bytes()
}

/// Build the picture header for a non-IRAP `TRAIL` P picture (H.266 §7.3.2.8).
///
/// The stream uses explicit reference picture lists in the slice header, so the
/// picture header carries only the base fields and `ph_mvd_l1_zero_flag`.
pub fn build_ph_rbsp_inter(poc_lsb: u32, tools: &VvcHlsTools) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(4);
    bw.write_bit(false); // ph_gdr_or_irap_pic_flag (TRAIL)
    bw.write_bit(false); // ph_non_ref_pic_flag
    bw.write_bit(true); // ph_inter_slice_allowed_flag
    bw.write_bit(true); // ph_intra_slice_allowed_flag
    bw.write_ue(0); // ph_pic_parameter_set_id
    bw.write_bits(poc_lsb & 0xFF, 8); // ph_pic_order_cnt_lsb
    // No ALF/LMCS/scaling-list/partition-override/SAO/Deblocking fields are
    // present for this configuration.
    // `pps_rpl_info_in_ph_flag` is 0, so ph_mvd_l1_zero_flag is present.
    bw.write_bit(false); // ph_mvd_l1_zero_flag
    if tools.joint_cbcr {
        bw.write_bit(false); // ph_joint_cbcr_sign_flag
    }
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Build the slice header for a `TRAIL` P slice (H.266 §7.3.7).
///
/// L0 contains the previous picture (`DeltaPocSt = -1`); L1 is empty. Both
/// lists are carried explicitly because `sps_num_ref_pic_lists[0] == 0`.
pub fn build_inter_slice_header_bits(
    slice_qp_delta: i32,
    tools: &VvcHlsTools,
    dep_quant: bool,
    sign_data_hiding: bool,
) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(16);
    bw.write_bit(false); // sh_picture_header_in_slice_header_flag
    bw.write_ue(1); // sh_slice_type = P
    // TRAIL is not an IRAP, so sh_no_output_of_prior_pics_flag is absent.
    if tools.alf {
        bw.write_bit(true); // sh_alf_enabled_flag = 1 (uses fixed filter sets, num_aps = 0)
        bw.write_bits(0, 3); // sh_num_alf_aps_ids_luma = 0
        bw.write_bit(false); // sh_alf_cb_enabled_flag = 0
        bw.write_bit(false); // sh_alf_cr_enabled_flag = 0
    }
    // L0: one short-term entry with DeltaPocSt = -1.
    bw.write_ue(1); // num_ref_entries[0][0]
    bw.write_ue(0); // abs_delta_poc_st[0][0][0] (abs(DeltaPocSt) - 1)
    bw.write_bit(true); // strp_entry_sign_flag (negative)
    // L1: no entries.
    bw.write_ue(0); // num_ref_entries[1][0]
    bw.write_se(slice_qp_delta); // sh_qp_delta
    if tools.sao {
        bw.write_bit(true); // sh_sao_luma_used_flag
        bw.write_bit(true); // sh_sao_chroma_used_flag
    }
    if tools.dep_quant {
        bw.write_bit(dep_quant); // sh_dep_quant_used_flag
    }
    if tools.sign_data_hiding && !dep_quant {
        bw.write_bit(sign_data_hiding); // sh_sign_data_hiding_used_flag
    }
    if !dep_quant && !sign_data_hiding {
        bw.write_bit(false); // sh_ts_residual_coding_disabled_flag
    }
    bw.byte_alignment();
    bw.into_bytes()
}

/// Build the slice header for a `TRAIL` B slice (`sh_slice_type = 0`, H.266 §7.3.7,
/// VTM `VLCWriter.cpp:2515-2950`) with one L0 short-term reference (`delta_poc_l0`)
/// and one L1 short-term reference (`delta_poc_l1`).
pub fn build_b_slice_header_bits(
    slice_qp_delta: i32,
    delta_poc_l0: i32,
    delta_poc_l1: i32,
    tools: &VvcHlsTools,
    dep_quant: bool,
    sign_data_hiding: bool,
) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(16);
    bw.write_bit(false); // sh_picture_header_in_slice_header_flag
    bw.write_ue(0); // sh_slice_type = B (0)
    if tools.alf {
        bw.write_bit(true); // sh_alf_enabled_flag
        bw.write_bits(0, 3); // sh_num_alf_aps_ids_luma = 0
        bw.write_bit(false); // sh_alf_cb_enabled_flag
        bw.write_bit(false); // sh_alf_cr_enabled_flag
    }
    // L0: one short-term entry
    let abs_l0 = delta_poc_l0.unsigned_abs().max(1);
    bw.write_ue(1); // num_ref_entries[0]
    bw.write_ue(abs_l0 - 1); // abs_delta_poc_st[0][0]
    bw.write_bit(delta_poc_l0 < 0); // strp_entry_sign_flag
    // L1: one short-term entry
    let abs_l1 = delta_poc_l1.unsigned_abs().max(1);
    bw.write_ue(1); // num_ref_entries[1]
    bw.write_ue(abs_l1 - 1); // abs_delta_poc_st[1][0]
    bw.write_bit(delta_poc_l1 < 0); // strp_entry_sign_flag
    bw.write_se(slice_qp_delta); // sh_qp_delta
    if tools.sao {
        bw.write_bit(true); // sh_sao_luma_used_flag
        bw.write_bit(true); // sh_sao_chroma_used_flag
    }
    if tools.dep_quant {
        bw.write_bit(dep_quant);
    }
    if tools.sign_data_hiding && !dep_quant {
        bw.write_bit(sign_data_hiding);
    }
    if !dep_quant && !sign_data_hiding {
        bw.write_bit(false); // sh_ts_residual_coding_disabled_flag
    }
    bw.byte_alignment();
    bw.into_bytes()
}

/// Build an AUD RBSP (`aud_irap_or_gdr_au_flag = 1`, I picture).
pub fn build_aud_rbsp() -> Vec<u8> {
    build_aud_rbsp_pic_type(0)
}

/// Build an AUD RBSP with an explicit `aud_pic_type` (`0` = I, `1` = P, `2` = B).
pub fn build_aud_rbsp_pic_type(pic_type: u8) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(2);
    bw.write_bit(true); // aud_irap_or_gdr_au_flag
    bw.write_bits(u32::from(pic_type & 0x7), 3); // aud_pic_type
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Append one Annex-B NAL unit: 4-byte start code, 2-byte VVC header, EPB-escaped RBSP.
pub fn push_nal(out: &mut Vec<u8>, nal_type: u8, layer_id: u8, temporal_id_plus1: u8, rbsp: &[u8]) {
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.push(layer_id & 0x3F);
    out.push(((nal_type & 0x1F) << 3) | (temporal_id_plus1 & 0x07));
    out.extend_from_slice(&rbsp_to_ebsp(rbsp));
}

/// The chroma QP mapping table derived from the trivial parameter set this
/// encoder writes (`start = 0`, one point, `delta_qp_in_val_minus1 = 0`,
/// `delta_qp_diff_val = 0`).
///
/// Mirrors `ChromaQpMappingTable::deriveChromaQPMappingTables()`: for 8-bit the
/// derived table is `q -> q` for `q <= 26` and `q -> q - 1` above that.
pub fn derived_chroma_qp(qp: i32) -> i32 {
    let qpi = qp.clamp(0, 63);
    if qpi <= 26 { qpi } else { qpi - 1 }
}

/// Derive the scaled luma quantisation parameter `Qp'_Y = Clip3(0, 63 + QpBdOffset, Qp_Y + QpBdOffset)`
/// for `bit_depth` in `8..=12` (`QpBdOffset = 6 * (bit_depth - 8)`, ITU-T Rec. H.266 §8.7.1,
/// VTM `Quant.cpp:76-92`).
#[inline]
pub fn scaled_luma_qp_bd(qpy: i32, bit_depth: u8) -> u8 {
    let qp_bd = 6 * (i32::from(bit_depth.clamp(8, 12)) - 8);
    (qpy + qp_bd).clamp(0, 63 + qp_bd) as u8
}

/// Derive the scaled chroma quantisation parameter `Qp'_C` for `bit_depth` in `8..=12`
/// (`QpBdOffset = 6 * (bit_depth - 8)`, `0..=63 + QpBdOffset`, up to `87` at 12-bit,
/// ITU-T Rec. H.266 §8.7.1, VTM `Quant.cpp:80-92`).
#[inline]
pub fn derived_chroma_qp_bd(qpy: i32, bit_depth: u8, chroma_qp_offset: i32) -> u8 {
    let qp_bd = 6 * (i32::from(bit_depth.clamp(8, 12)) - 8);
    let qpi = qpy.clamp(-qp_bd, 63);
    let mapped = if qpi <= 26 { qpi } else { qpi - 1 };
    let base_qp = (mapped + chroma_qp_offset).clamp(-qp_bd, 63) + qp_bd;
    base_qp.clamp(0, 63 + qp_bd) as u8
}

/// Information extracted from a sequence parameter set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcSpsInfo {
    /// `sps_seq_parameter_set_id`.
    pub sps_id: u8,
    /// `sps_video_parameter_set_id`.
    pub vps_id: u8,
    /// `sps_chroma_format_idc` (1 = 4:2:0).
    pub chroma_format_idc: u8,
    /// Luma bit depth, derived from `sps_bitdepth_minus8`.
    pub bit_depth_luma: u8,
    /// Visible (conformance-window-cropped) width in luma samples.
    pub width: u32,
    /// Visible (conformance-window-cropped) height in luma samples.
    pub height: u32,
    /// Coded width (`sps_pic_width_max_in_luma_samples`).
    pub coded_width: u32,
    /// Coded height (`sps_pic_height_max_in_luma_samples`).
    pub coded_height: u32,
    /// `log2` of the CTU size.
    pub ctu_log2: u8,
    /// `general_profile_idc`.
    pub profile_idc: u8,
    /// `general_tier_flag`.
    pub tier_flag: bool,
    /// `general_level_idc`.
    pub level_idc: u8,
    /// `ptl_frame_only_constraint_flag`.
    pub frame_only_constraint: bool,
    /// `ptl_multilayer_enabled_flag`.
    pub multilayer_enabled: bool,
    /// `general_constraints_info()` syntax bits, starting with
    /// `gci_present_flag` and excluding its alignment zeros.
    pub gci_bits: Vec<bool>,
    /// `ptl_num_sub_profiles`.
    pub num_sub_profiles: u8,
    /// `ptl_sub_profiles[i]`.
    pub sub_profile_idc: Vec<u32>,
    /// `sps_ref_pic_resampling_enabled_flag`.
    pub ref_pic_resampling: bool,
    /// `sps_res_change_in_clvs_allowed_flag` (RPR only).
    pub res_change_in_clvs_allowed: bool,
}

/// Map a bit-reader failure onto a malformed-bitstream error.
fn sps_bit<T>(value: Result<T, &'static str>) -> Result<T, VvcError> {
    value.map_err(|err| VvcError::Bitstream(format!("SPS: {err}")))
}

/// Map a bit-reader failure onto a malformed-bitstream error with a custom prefix.
fn hls_bit<T>(prefix: &str, value: Result<T, &'static str>) -> Result<T, VvcError> {
    value.map_err(|err| VvcError::Bitstream(format!("{prefix}: {err}")))
}

/// Parse the leading fields of an SPS RBSP (no emulation-prevention bytes).
pub fn parse_sps_info(rbsp: &[u8]) -> Result<VvcSpsInfo, VvcError> {
    let mut br = BitReader::new(rbsp);
    let sps_id = sps_bit(br.read_bits(4))? as u8;
    let vps_id = sps_bit(br.read_bits(4))? as u8;
    let max_sub_layers_minus1 = sps_bit(br.read_bits(3))?;
    let chroma_format_idc = sps_bit(br.read_bits(2))? as u8;
    let log2_ctu_size_minus5 = sps_bit(br.read_bits(2))? as u8;
    if max_sub_layers_minus1 != 0 {
        return Err(VvcError::Unsupported(
            "SPS: more than one temporal sub-layer",
        ));
    }
    let ptl_present = sps_bit(br.read_bit())?;
    if !ptl_present {
        return Err(VvcError::Unsupported("SPS: profile_tier_level is absent"));
    }

    // profile_tier_level(1, max_sub_layers_minus1)
    let profile_idc = sps_bit(br.read_bits(7))? as u8;
    let tier_flag = sps_bit(br.read_bit())?;
    let level_idc = sps_bit(br.read_bits(8))? as u8;
    let frame_only_constraint = sps_bit(br.read_bit())?;
    let multilayer_enabled = sps_bit(br.read_bit())?;

    let mut gci_bits = Vec::with_capacity(96);
    let gci_present = sps_bit(br.read_bit())?;
    gci_bits.push(gci_present);
    if gci_present {
        const GCI_FLAGS: usize = 71;
        for _ in 0..GCI_FLAGS {
            gci_bits.push(sps_bit(br.read_bit())?);
        }
        let additional = sps_bit(br.read_bits(8))?;
        for i in (0..8).rev() {
            gci_bits.push((additional >> i) & 1 != 0);
        }
        if additional > 5 {
            for _ in 0..6 {
                gci_bits.push(sps_bit(br.read_bit())?);
            }
        }
        for _ in 0..additional.saturating_sub(if additional > 5 { 6 } else { 0 }) {
            gci_bits.push(sps_bit(br.read_bit())?);
        }
    }
    while !br.is_byte_aligned() {
        if sps_bit(br.read_bit())? {
            return Err(VvcError::Bitstream(
                "SPS: gci_alignment_zero_bit is set".to_string(),
            ));
        }
    }

    let num_sub_profiles = sps_bit(br.read_bits(8))? as u8;
    let mut sub_profile_idc = Vec::with_capacity(num_sub_profiles as usize);
    for _ in 0..num_sub_profiles {
        sub_profile_idc.push(sps_bit(br.read_bits(32))?);
    }

    let _gdr_enabled = sps_bit(br.read_bit())?;
    let ref_pic_resampling = sps_bit(br.read_bit())?;
    // `sps_res_change_in_clvs_allowed_flag` is present only when RPR is enabled.
    // Same-resolution streams decode without resampling; actual resolution
    // changes are still unsupported and rejected at picture-header time.
    let res_change_in_clvs_allowed = if ref_pic_resampling {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    let coded_width = sps_bit(br.read_ue())?;
    let coded_height = sps_bit(br.read_ue())?;
    let conformance_window = sps_bit(br.read_bit())?;
    let (mut crop_right, mut crop_bottom) = (0u32, 0u32);
    if conformance_window {
        let crop_left = sps_bit(br.read_ue())?;
        crop_right = sps_bit(br.read_ue())?;
        let crop_top = sps_bit(br.read_ue())?;
        crop_bottom = sps_bit(br.read_ue())?;
        if crop_left != 0 || crop_top != 0 {
            return Err(VvcError::Unsupported(
                "SPS: conformance window not anchored at the origin",
            ));
        }
    }
    let subpic_info = sps_bit(br.read_bit())?;
    if subpic_info {
        let num_subpics_minus1 = sps_bit(br.read_ue())?;
        let num_subpics = num_subpics_minus1.saturating_add(1);
        let ctu_size = 1u32 << (log2_ctu_size_minus5.min(3) + 5);
        let max_pic_w_ctu = coded_width.max(ctu_size).div_ceil(ctu_size);
        let max_pic_h_ctu = coded_height.max(ctu_size).div_ceil(ctu_size);
        if num_subpics > 1 {
            let indep = sps_bit(br.read_bit())?;
            let same_size = sps_bit(br.read_bit())?;
            let w_bits = u32::BITS - max_pic_w_ctu.saturating_sub(1).leading_zeros();
            let h_bits = u32::BITS - max_pic_h_ctu.saturating_sub(1).leading_zeros();
            for i in 0..num_subpics {
                if !same_size || i == 0 {
                    if i > 0 && coded_width > ctu_size {
                        let _ = sps_bit(br.read_bits(w_bits))?;
                    }
                    if i > 0 && coded_height > ctu_size {
                        let _ = sps_bit(br.read_bits(h_bits))?;
                    }
                    if i + 1 < num_subpics && coded_width > ctu_size {
                        let _ = sps_bit(br.read_bits(w_bits))?;
                    }
                    if i + 1 < num_subpics && coded_height > ctu_size {
                        let _ = sps_bit(br.read_bits(h_bits))?;
                    }
                }
                if !indep {
                    let _ = sps_bit(br.read_bit())?;
                    let _ = sps_bit(br.read_bit())?;
                }
            }
        }
        let id_len = sps_bit(br.read_ue())?.saturating_add(1) as usize;
        if id_len == 0 || id_len > 16 {
            return Err(VvcError::Bitstream(
                "SPS: invalid sps_subpic_id_len_minus1".to_string(),
            ));
        }
        if sps_bit(br.read_bit())? && sps_bit(br.read_bit())? {
            for _ in 0..num_subpics {
                let _ = sps_bit(br.read_bits(id_len as u32))?;
            }
        }
    }
    let bit_depth_minus8 = sps_bit(br.read_ue())?;
    let bit_depth_luma = bit_depth_minus8
        .checked_add(8)
        .and_then(|depth| u8::try_from(depth).ok())
        .ok_or_else(|| {
            VvcError::Bitstream(format!(
                "SPS: sps_bitdepth_minus8 value {bit_depth_minus8} is out of range"
            ))
        })?;

    let (sub_w, sub_h): (u32, u32) = match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let crop_x = crop_right.checked_mul(sub_w).ok_or_else(|| {
        VvcError::Bitstream("SPS: conformance window right offset overflows".to_string())
    })?;
    let crop_y = crop_bottom.checked_mul(sub_h).ok_or_else(|| {
        VvcError::Bitstream("SPS: conformance window bottom offset overflows".to_string())
    })?;
    let width = coded_width.checked_sub(crop_x).ok_or_else(|| {
        VvcError::Bitstream("SPS: conformance window exceeds picture width".to_string())
    })?;
    let height = coded_height.checked_sub(crop_y).ok_or_else(|| {
        VvcError::Bitstream("SPS: conformance window exceeds picture height".to_string())
    })?;

    Ok(VvcSpsInfo {
        sps_id,
        vps_id,
        chroma_format_idc,
        bit_depth_luma,
        width,
        height,
        coded_width,
        coded_height,
        ctu_log2: log2_ctu_size_minus5 + 5,
        profile_idc,
        tier_flag,
        level_idc,
        frame_only_constraint,
        multilayer_enabled,
        gci_bits,
        num_sub_profiles,
        sub_profile_idc,
        ref_pic_resampling,
        res_change_in_clvs_allowed,
    })
}

// ============================================================================
// Wave 1 — Full High-Level Syntax (OPI, DCI, VPS, RPL, SPS, PPS, APS, PH, SH)
// ============================================================================

/// Operating Point Information (`operating_point_information_rbsp`,
/// H.266 p. 37, PDF p. 53, §7.3.2.1; VTM `VLCReader.cpp:2197`, `VLCWriter.cpp:1375`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcOpi {
    /// Optional `opi_ols_idx` when `opi_ols_info_present_flag == 1`.
    pub ols_idx: Option<u32>,
    /// Optional `opi_htid_plus1` (3-bit) when `opi_htid_info_present_flag == 1`.
    pub htid_plus1: Option<u8>,
}

/// Build an OPI RBSP (`H.266 §7.3.2.1`, `VLCWriter.cpp:1375-1394`).
pub fn build_opi_rbsp(opi: &VvcOpi) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(8);
    bw.write_bit(opi.ols_idx.is_some()); // opi_ols_info_present_flag
    bw.write_bit(opi.htid_plus1.is_some()); // opi_htid_info_present_flag
    if let Some(ols_idx) = opi.ols_idx {
        bw.write_ue(ols_idx); // opi_ols_idx
    }
    if let Some(htid_plus1) = opi.htid_plus1 {
        bw.write_bits(u32::from(htid_plus1 & 7), 3); // opi_htid_plus1
    }
    bw.write_bit(false); // opi_extension_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Parse an OPI RBSP (`H.266 §7.3.2.1`, `VLCReader.cpp:2197-2231`).
pub fn parse_opi_rbsp(rbsp: &[u8]) -> Result<VvcOpi, VvcError> {
    let mut br = BitReader::new(rbsp);
    let ols_present = hls_bit("OPI", br.read_bit())?;
    let htid_present = hls_bit("OPI", br.read_bit())?;
    let ols_idx = if ols_present {
        Some(hls_bit("OPI", br.read_ue())?)
    } else {
        None
    };
    let htid_plus1 = if htid_present {
        Some(hls_bit("OPI", br.read_bits(3))? as u8)
    } else {
        None
    };
    let _ext = hls_bit("OPI", br.read_bit())?;
    Ok(VvcOpi {
        ols_idx,
        htid_plus1,
    })
}

/// Decoding Capability Information (`decoding_capability_information_rbsp`,
/// H.266 p. 37–38, PDF p. 53–54, §7.3.2.2; VTM `VLCReader.cpp:2233`, `VLCWriter.cpp:1354`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcDci {
    /// `(profile_idc, tier_flag, level_idc)` for each signalled PTL (`1..=15`).
    pub ptls: Vec<(u8, bool, u8)>,
}

/// Build a DCI RBSP (`H.266 §7.3.2.2`, `VLCWriter.cpp:1354-1373`).
pub fn build_dci_rbsp(dci: &VvcDci, ctu_log2: u8) -> Result<Vec<u8>, VvcError> {
    if dci.ptls.is_empty() || dci.ptls.len() > 15 {
        return Err(VvcError::InvalidConfig(
            "DCI requires 1..=15 PTL entries".to_string(),
        ));
    }
    let mut bw = BitWriter::with_capacity(32);
    bw.write_bits(0, 4); // dci_reserved_zero_4bits
    bw.write_bits((dci.ptls.len() - 1) as u32, 4); // dci_num_ptls_minus1
    for &(profile_idc, tier_flag, level_idc) in &dci.ptls {
        bw.write_bits(u32::from(profile_idc & 0x7F), 7);
        bw.write_bit(tier_flag);
        bw.write_bits(u32::from(level_idc), 8);
        bw.write_bit(true); // ptl_frame_only_constraint_flag
        bw.write_bit(false); // ptl_multilayer_enabled_flag
        write_general_constraints_info(&mut bw, ctu_log2, &VvcHlsTools::default());
        bw.align_zero();
        bw.write_bits(0, 8); // ptl_num_sub_profiles
    }
    bw.write_bit(false); // dci_extension_flag
    bw.rbsp_trailing_bits();
    Ok(bw.into_bytes())
}

/// Parse a DCI RBSP (`H.266 §7.3.2.2`, `VLCReader.cpp:2233-2263`).
pub fn parse_dci_rbsp(rbsp: &[u8]) -> Result<VvcDci, VvcError> {
    let mut br = BitReader::new(rbsp);
    let reserved = hls_bit("DCI", br.read_bits(4))?;
    if reserved != 0 {
        return Err(VvcError::Bitstream(
            "DCI: dci_reserved_zero_4bits is non-zero".to_string(),
        ));
    }
    let num_ptls = (hls_bit("DCI", br.read_bits(4))? + 1) as usize;
    if num_ptls > 15 {
        return Err(VvcError::Bitstream(
            "DCI: dci_num_ptls_minus1 exceeds 14".to_string(),
        ));
    }
    let mut ptls = Vec::with_capacity(num_ptls);
    for _ in 0..num_ptls {
        let profile_idc = hls_bit("DCI", br.read_bits(7))? as u8;
        let tier_flag = hls_bit("DCI", br.read_bit())?;
        let level_idc = hls_bit("DCI", br.read_bits(8))? as u8;
        let _frame_only = hls_bit("DCI", br.read_bit())?;
        let _multilayer = hls_bit("DCI", br.read_bit())?;
        let gci_present = hls_bit("DCI", br.read_bit())?;
        if gci_present {
            for _ in 0..71 {
                hls_bit("DCI", br.read_bit())?;
            }
            let add = hls_bit("DCI", br.read_bits(8))?;
            for _ in 0..add {
                hls_bit("DCI", br.read_bit())?;
            }
        }
        while !br.is_byte_aligned() {
            if hls_bit("DCI", br.read_bit())? {
                return Err(VvcError::Bitstream(
                    "DCI: non-zero alignment bit".to_string(),
                ));
            }
        }
        let num_sub = hls_bit("DCI", br.read_bits(8))?;
        for _ in 0..num_sub {
            hls_bit("DCI", br.read_bits(32))?;
        }
        ptls.push((profile_idc, tier_flag, level_idc));
    }
    Ok(VvcDci { ptls })
}

/// Parsed Video Parameter Set (`video_parameter_set_rbsp`, H.266 p. 38, PDF p. 54, §7.3.2.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcVps {
    /// `vps_video_parameter_set_id` (`1..=15`).
    pub vps_id: u8,
    /// `vps_max_layers_minus1`.
    pub max_layers_minus1: u8,
    /// `vps_max_sublayers_minus1`.
    pub max_sublayers_minus1: u8,
    /// `vps_layer_id[0]`.
    pub layer_id_0: u8,
    /// `general_profile_idc`.
    pub profile_idc: u8,
    /// `general_tier_flag`.
    pub tier_flag: bool,
    /// `general_level_idc`.
    pub level_idc: u8,
}

/// Parse a single-layer VPS RBSP (`H.266 §7.3.2.3`, `VLCReader.cpp:2265-2350`).
pub fn parse_vps_rbsp(rbsp: &[u8]) -> Result<VvcVps, VvcError> {
    let mut br = BitReader::new(rbsp);
    let vps_id = hls_bit("VPS", br.read_bits(4))? as u8;
    if vps_id == 0 {
        return Err(VvcError::Bitstream(
            "VPS: vps_video_parameter_set_id shall not be 0".to_string(),
        ));
    }
    let max_layers_minus1 = hls_bit("VPS", br.read_bits(6))? as u8;
    let max_sublayers_minus1 = hls_bit("VPS", br.read_bits(3))? as u8;
    if max_layers_minus1 != 0 || max_sublayers_minus1 != 0 {
        return Err(VvcError::Unsupported("VPS: multi-layer or multi-sublayer"));
    }
    let layer_id_0 = hls_bit("VPS", br.read_bits(6))? as u8;
    while !br.is_byte_aligned() {
        if hls_bit("VPS", br.read_bit())? {
            return Err(VvcError::Bitstream(
                "VPS: vps_ptl_reserved_zero_bit is non-zero".to_string(),
            ));
        }
    }
    let profile_idc = hls_bit("VPS", br.read_bits(7))? as u8;
    let tier_flag = hls_bit("VPS", br.read_bit())?;
    let level_idc = hls_bit("VPS", br.read_bits(8))? as u8;
    Ok(VvcVps {
        vps_id,
        max_layers_minus1,
        max_sublayers_minus1,
        layer_id_0,
        profile_idc,
        tier_flag,
        level_idc,
    })
}

/// One entry in a `ref_pic_list_struct` (H.266 p. 63, PDF p. 79, §7.3.10;
/// VTM `VLCReader.cpp:341-436`, `VLCWriter.cpp:228-293`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcRefPicEntry {
    /// Short-term reference picture identified by signed `DeltaPocValSt` relative
    /// to the current picture (or previous STRP entry in `ref_pic_list_struct`).
    ShortTerm {
        /// Signed POC delta from the previous STRP entry (or current picture for first STRP).
        delta_poc: i32,
    },
    /// Long-term reference picture identified by `poc_lsb_lt`.
    LongTerm {
        /// `rpls_poc_lsb_lt`.
        poc_lsb: u32,
        /// Optional `delta_poc_msb_cycle_lt` when `delta_poc_msb_present_flag == 1`.
        delta_poc_msb_cycle: Option<u32>,
    },
}

/// Reference picture list structure (`ref_pic_list_struct(listIdx, rplsIdx)`,
/// H.266 p. 63, PDF p. 79, §7.3.10).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VvcRefPicListStruct {
    /// `ltrp_in_header_flag`.
    pub ltrp_in_header_flag: bool,
    /// Ordered entries (`num_ref_entries`).
    pub entries: Vec<VvcRefPicEntry>,
}

/// Write `ref_pic_list_struct` (`H.266 §7.3.10`, `VLCWriter.cpp:228-293`).
pub fn write_ref_pic_list_struct(
    bw: &mut BitWriter,
    rpl: &VvcRefPicListStruct,
    long_term_present: bool,
    poc_lsb_bits: u32,
    forbidden_zero_delta_poc: bool,
    in_sps: bool,
) {
    bw.write_ue(rpl.entries.len() as u32); // num_ref_entries
    if long_term_present && !rpl.entries.is_empty() && in_sps {
        bw.write_bit(rpl.ltrp_in_header_flag); // ltrp_in_header_flag
    }
    for (i, entry) in rpl.entries.iter().enumerate() {
        match *entry {
            VvcRefPicEntry::ShortTerm { delta_poc } => {
                if long_term_present {
                    bw.write_bit(true); // st_ref_pic_flag = 1
                }
                let abs_delta = delta_poc.unsigned_abs();
                if forbidden_zero_delta_poc || i == 0 {
                    bw.write_ue(abs_delta.saturating_sub(1)); // abs_delta_poc_st
                } else {
                    bw.write_ue(abs_delta);
                }
                if abs_delta > 0 {
                    // Note: in H.266 §7.4.11 `strp_entry_sign_flag == 1` means negative DeltaPocValSt
                    // in VTM `VLCWriter.cpp:277` (`deltaValue < 0 ? 1 : 0`).
                    bw.write_bit(delta_poc < 0);
                }
            }
            VvcRefPicEntry::LongTerm { poc_lsb, .. } => {
                if long_term_present {
                    bw.write_bit(false); // st_ref_pic_flag = 0
                }
                if !rpl.ltrp_in_header_flag {
                    bw.write_bits(poc_lsb, poc_lsb_bits); // rpls_poc_lsb_lt
                }
            }
        }
    }
}

/// Parse `ref_pic_list_struct` (`H.266 §7.3.10`, `VLCReader.cpp:341-436`).
pub fn parse_ref_pic_list_struct(
    br: &mut BitReader<'_>,
    long_term_present: bool,
    poc_lsb_bits: u32,
    forbidden_zero_delta_poc: bool,
    in_sps: bool,
) -> Result<VvcRefPicListStruct, VvcError> {
    let num_entries = hls_bit("RPL", br.read_ue())? as usize;
    if num_entries > 64 {
        return Err(VvcError::LimitExceeded(format!(
            "RPL num_ref_entries {num_entries} exceeds 64"
        )));
    }
    let ltrp_in_header_flag = if long_term_present && num_entries > 0 && in_sps {
        hls_bit("RPL", br.read_bit())?
    } else {
        false
    };
    let mut entries = Vec::with_capacity(num_entries);
    for i in 0..num_entries {
        let is_st = if long_term_present {
            hls_bit("RPL", br.read_bit())?
        } else {
            true
        };
        if is_st {
            let coded = hls_bit("RPL", br.read_ue())?;
            let abs_delta = if forbidden_zero_delta_poc || i == 0 {
                coded.checked_add(1).ok_or_else(|| {
                    VvcError::Bitstream("RPL: abs_delta_poc_st overflow".to_string())
                })?
            } else {
                coded
            };
            let sign = if abs_delta > 0 {
                hls_bit("RPL", br.read_bit())?
            } else {
                false
            };
            let mag = i32::try_from(abs_delta).map_err(|_| {
                VvcError::Bitstream("RPL: abs_delta_poc_st exceeds i32".to_string())
            })?;
            entries.push(VvcRefPicEntry::ShortTerm {
                delta_poc: if sign { -mag } else { mag },
            });
        } else {
            let poc_lsb = if !ltrp_in_header_flag {
                hls_bit("RPL", br.read_bits(poc_lsb_bits))?
            } else {
                0
            };
            entries.push(VvcRefPicEntry::LongTerm {
                poc_lsb,
                delta_poc_msb_cycle: None,
            });
        }
    }
    Ok(VvcRefPicListStruct {
        ltrp_in_header_flag,
        entries,
    })
}

/// Signalled chroma QP mapping table parameters (`H.266 p. 40, PDF p. 56, §7.3.2.4`;
/// `H.266 p. 113–114, PDF p. 129–130, §7.4.3.4 Eq. 36–39`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcChromaQpTable {
    /// `sps_qp_table_start_minus26[i]`.
    pub qp_table_start_minus26: i32,
    /// `(sps_delta_qp_in_val_minus1[i][j], sps_delta_qp_diff_val[i][j])` points.
    pub points: Vec<(u32, u32)>,
}

impl Default for VvcChromaQpTable {
    fn default() -> Self {
        Self {
            qp_table_start_minus26: 0,
            points: vec![(0, 0)],
        }
    }
}

impl VvcChromaQpTable {
    /// Derive the chroma QP mapping table for `qpBdOffset = 6 * (BitDepth - 8)`
    /// following H.266 §7.4.3.4 Eq. 36–39 and VTM
    /// `SequenceParameterSet.cpp:144-185`. Entry `i` is `QpC` for
    /// `qPi = i as i32 - qp_bd`.
    pub fn derive_table(&self, qp_bd: i32) -> Result<Vec<i32>, VvcError> {
        let qp_bd = qp_bd.clamp(0, 36);
        if self.points.is_empty() || self.points.len() > 63 {
            return Err(VvcError::Bitstream(
                "ChromaQpTable: invalid number of points".to_string(),
            ));
        }
        let n = self.points.len();
        let lo = -qp_bd;
        let mut qp_in = vec![0i32; n + 1];
        let mut qp_out = vec![0i32; n + 1];
        qp_in[0] = 26i32.saturating_add(self.qp_table_start_minus26);
        qp_out[0] = qp_in[0];
        if !(lo..=63).contains(&qp_in[0]) {
            return Err(VvcError::Bitstream(format!(
                "ChromaQpTable: start {} outside {}..=63",
                qp_in[0], lo
            )));
        }
        for (j, &(delta_in_minus1, delta_diff)) in self.points.iter().enumerate() {
            let delta_in = i32::try_from(delta_in_minus1)
                .ok()
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    VvcError::Bitstream("ChromaQpTable: delta_qp_in overflow".to_string())
                })?;
            // VTM stores `sps_delta_qp_diff_val` XORed with the matching
            // `sps_delta_qp_in_val_minus1` (`VLCReader.cpp:1749-1751`).
            let delta_out = i32::try_from(delta_in_minus1 ^ delta_diff).map_err(|_| {
                VvcError::Bitstream("ChromaQpTable: delta_qp_out overflow".to_string())
            })?;
            qp_in[j + 1] = qp_in[j].saturating_add(delta_in);
            qp_out[j + 1] = qp_out[j].saturating_add(delta_out);
            if !(lo..=63).contains(&qp_in[j + 1]) || !(lo..=63).contains(&qp_out[j + 1]) {
                return Err(VvcError::Bitstream(
                    "ChromaQpTable: derived pivot outside the valid QP range".to_string(),
                ));
            }
        }
        let mut table = vec![0i32; (64 + qp_bd) as usize];
        let at = |qp: i32| (qp + qp_bd) as usize;
        table[at(qp_in[0])] = qp_out[0];
        for k in (lo..qp_in[0]).rev() {
            table[at(k)] = (table[at(k + 1)] - 1).clamp(lo, 63);
        }
        for j in 0..n {
            let denom = (self.points[j].0 + 1) as i32;
            let sh = denom >> 1;
            let mut m = 1i32;
            for k in (qp_in[j] + 1)..=qp_in[j + 1] {
                table[at(k)] = table[at(qp_in[j])] + ((qp_out[j + 1] - qp_out[j]) * m + sh) / denom;
                m += 1;
            }
        }
        for k in (qp_in[n] + 1)..=63 {
            table[at(k)] = (table[at(k - 1)] + 1).clamp(lo, 63);
        }
        Ok(table)
    }

    /// Derive the 64-entry chroma QP mapping table for 8-bit (`QpBdOffset = 0`)
    /// following H.266 §7.4.3.4 Eq. 36–39 and VTM `SequenceParameterSet.cpp:144-185`.
    pub fn derive_table_8bit(&self) -> Result<[i32; 64], VvcError> {
        let table = self.derive_table(0)?;
        let mut out = [0i32; 64];
        out.copy_from_slice(&table[..64]);
        Ok(out)
    }
}

/// Complete Sequence Parameter Set (`seq_parameter_set_rbsp`, H.266 p. 38–42,
/// PDF p. 54–58, §7.3.2.4; VTM `VLCReader.cpp:1363-2195`, `VLCWriter.cpp:828-1352`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcSps {
    /// Leading identification, geometry, and PTL info.
    pub info: VvcSpsInfo,
    /// `sps_gdr_enabled_flag`.
    pub gdr_enabled: bool,
    /// `sps_entropy_coding_sync_enabled_flag` (WPP).
    pub wpp_enabled: bool,
    /// `sps_entry_point_offsets_present_flag`.
    pub entry_points_present: bool,
    /// `sps_log2_max_pic_order_cnt_lsb_minus4 + 4` (`4..=16`).
    pub poc_lsb_bits: u8,
    /// `sps_log2_min_luma_coding_block_size_minus2 + 2` (`2..=6`).
    pub min_cb_log2: u8,
    /// `sps_partition_constraints_override_enabled_flag`.
    pub partition_constraints_override_enabled: bool,
    /// `sps_log2_diff_min_qt_min_cb_intra_slice_luma`.
    pub log2_diff_min_qt_min_cb_intra_luma: u8,
    /// `sps_max_mtt_hierarchy_depth_intra_slice_luma`.
    pub max_mtt_depth_intra_luma: u8,
    /// `sps_log2_diff_max_bt_min_qt_intra_slice_luma`.
    pub log2_diff_max_bt_min_qt_intra_luma: u8,
    /// `sps_log2_diff_max_tt_min_qt_intra_slice_luma`.
    pub log2_diff_max_tt_min_qt_intra_luma: u8,
    /// `sps_qtbtt_dual_tree_intra_flag`.
    pub dual_tree_intra: bool,
    /// `sps_log2_diff_min_qt_min_cb_inter_slice`.
    pub log2_diff_min_qt_min_cb_inter: u8,
    /// `sps_max_mtt_hierarchy_depth_inter_slice`.
    pub max_mtt_depth_inter: u8,
    /// `sps_log2_diff_max_bt_min_qt_inter_slice`.
    pub log2_diff_max_bt_min_qt_inter: u8,
    /// `sps_log2_diff_max_tt_min_qt_inter_slice`.
    pub log2_diff_max_tt_min_qt_inter: u8,
    /// `sps_max_luma_transform_size_64_flag`.
    pub max_tb_size_64: bool,
    /// `sps_transform_skip_enabled_flag`.
    pub transform_skip_enabled: bool,
    /// `sps_log2_transform_skip_max_size_minus2 + 2`.
    pub log2_transform_skip_max_size: u8,
    /// `sps_bdpcm_enabled_flag`.
    pub bdpcm_enabled: bool,
    /// `sps_mts_enabled_flag`.
    pub mts_enabled: bool,
    /// `sps_explicit_mts_intra_enabled_flag`.
    pub explicit_mts_intra: bool,
    /// `sps_explicit_mts_inter_enabled_flag`.
    pub explicit_mts_inter: bool,
    /// `sps_lfnst_enabled_flag`.
    pub lfnst_enabled: bool,
    /// `sps_joint_cbcr_enabled_flag`.
    pub joint_cbcr_enabled: bool,
    /// `sps_same_qp_table_for_chroma_flag`.
    pub same_qp_table_for_chroma: bool,
    /// Signalled chroma QP mapping tables (`1..=3`).
    pub chroma_qp_tables: Vec<VvcChromaQpTable>,
    /// `sps_sao_enabled_flag`.
    pub sao_enabled: bool,
    /// `sps_alf_enabled_flag`.
    pub alf_enabled: bool,
    /// `sps_ccalf_enabled_flag`.
    pub ccalf_enabled: bool,
    /// `sps_lmcs_enabled_flag`.
    pub lmcs_enabled: bool,
    /// `sps_weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `sps_weighted_bipred_flag`.
    pub weighted_bipred: bool,
    /// `sps_long_term_ref_pics_flag`.
    pub long_term_ref_pics: bool,
    /// `sps_idr_rpl_present_flag`.
    pub idr_rpl_present: bool,
    /// `sps_rpl1_same_as_rpl0_flag`.
    pub rpl1_same_as_rpl0: bool,
    /// Reference picture list structures for L0 and L1.
    pub rpl_lists: [Vec<VvcRefPicListStruct>; 2],
    /// `sps_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `sps_sbtmvp_enabled_flag`.
    pub sbtmvp_enabled: bool,
    /// `sps_amvr_enabled_flag`.
    pub amvr_enabled: bool,
    /// `sps_bdof_enabled_flag`.
    pub bdof_enabled: bool,
    /// `sps_smvd_enabled_flag`.
    pub smvd_enabled: bool,
    /// `sps_dmvr_enabled_flag`.
    pub dmvr_enabled: bool,
    /// `sps_mmvd_enabled_flag`.
    pub mmvd_enabled: bool,
    /// `MaxNumMergeCand` (`6 - sps_six_minus_max_num_merge_cand`, `1..=6`).
    pub max_num_merge_cand: u8,
    /// `sps_sbt_enabled_flag`.
    pub sbt_enabled: bool,
    /// `sps_affine_enabled_flag`.
    pub affine_enabled: bool,
    /// `MaxNumSubblockMergeCand` (`5 - sps_five_minus_max_num_subblock_merge_cand`).
    pub max_num_affine_merge_cand: u8,
    /// `sps_affine_type_flag` (6-parameter affine enabled).
    pub affine_type_6param: bool,
    /// `sps_affine_amvr_enabled_flag`.
    pub affine_amvr_enabled: bool,
    /// `sps_affine_prof_enabled_flag`.
    pub affine_prof_enabled: bool,
    /// `sps_bcw_enabled_flag`.
    pub bcw_enabled: bool,
    /// `sps_ciip_enabled_flag`.
    pub ciip_enabled: bool,
    /// `sps_gpm_enabled_flag`.
    pub gpm_enabled: bool,
    /// `MaxNumGpmMergeCand`.
    pub max_num_gpm_cand: u8,
    /// `sps_isp_enabled_flag`.
    pub isp_enabled: bool,
    /// `sps_mrl_enabled_flag`.
    pub mrl_enabled: bool,
    /// `sps_mip_enabled_flag`.
    pub mip_enabled: bool,
    /// `sps_cclm_enabled_flag`.
    pub cclm_enabled: bool,
    /// `sps_explicit_scaling_list_enabled_flag`.
    pub explicit_scaling_list_enabled: bool,
    /// `sps_scaling_matrix_for_lfnst_disabled_flag`.
    pub scaling_matrix_for_lfnst_disabled: bool,
    /// `sps_dep_quant_enabled_flag`.
    pub dep_quant_enabled: bool,
    /// `sps_sign_data_hiding_enabled_flag`.
    pub sign_data_hiding_enabled: bool,
    /// `sps_palette_enabled_flag`.
    pub palette_enabled: bool,
    /// `sps_act_enabled_flag` (4:4:4 Adaptive Colour Transform).
    pub act_enabled: bool,
    /// `sps_ibc_enabled_flag` (Intra Block Copy).
    pub ibc_enabled: bool,
    /// `MaxNumIbcMergeCand` (`6 - sps_six_minus_max_num_ibc_merge_cand`).
    pub max_num_ibc_merge_cand: u8,
}

impl VvcSps {
    /// Map luma `QpY` to the chroma quantisation parameter
    /// `Qp'C = Clip3(0, 63 + QpBdOffset, Clip3(-QpBdOffset, 63, QpC + offset) + QpBdOffset)`
    /// using the signalled `chroma_qp_mapping_table[component]`
    /// (`H.266 §7.4.3.4`, VTM `Quant.cpp:72-92`, `SequenceParameterSet.cpp:144-185`).
    ///
    /// `component` is `1` for Cb and `2` for Cr.
    pub fn mapped_chroma_qp(&self, component: u8, qpy: i32, offset: i32) -> i32 {
        let qp_bd = 6 * (i32::from(self.info.bit_depth_luma.clamp(8, 12)) - 8);
        let qpi = qpy.clamp(-qp_bd, 63);
        let default = if qpi <= 26 { qpi } else { qpi - 1 };
        let table_index = if self.same_qp_table_for_chroma {
            0
        } else {
            usize::from(component.saturating_sub(1))
        };
        let mapped = self
            .chroma_qp_tables
            .get(table_index)
            .and_then(|table| table.derive_table(qp_bd).ok())
            .and_then(|table| table.get((qpi + qp_bd) as usize).copied())
            .unwrap_or(default);
        ((mapped + offset).clamp(-qp_bd, 63) + qp_bd).clamp(0, 63 + qp_bd)
    }
}

/// Parse a complete SPS RBSP (`H.266 §7.3.2.4`, `VLCReader.cpp:1363-2195`).
pub fn parse_full_sps_rbsp(rbsp: &[u8]) -> Result<VvcSps, VvcError> {
    let info = parse_sps_info(rbsp)?;
    let mut br = BitReader::new(rbsp);
    // Advance past header + PTL using exact bit structure verified in parse_sps_info.
    let _ = sps_bit(br.read_bits(15))?;
    let _ptl_present = sps_bit(br.read_bit())?;
    let _ = sps_bit(br.read_bits(18))?;
    let gci_present = sps_bit(br.read_bit())?;
    if gci_present {
        for _ in 0..71 {
            let _ = sps_bit(br.read_bit())?;
        }
        let add = sps_bit(br.read_bits(8))?;
        for _ in 0..add {
            let _ = sps_bit(br.read_bit())?;
        }
    }
    while !br.is_byte_aligned() {
        let _ = sps_bit(br.read_bit())?;
    }
    let num_sub = sps_bit(br.read_bits(8))?;
    for _ in 0..num_sub {
        let _ = sps_bit(br.read_bits(32))?;
    }

    let gdr_enabled = sps_bit(br.read_bit())?;
    let _rpr = sps_bit(br.read_bit())?;
    if _rpr {
        // `sps_res_change_in_clvs_allowed_flag` (H.266 §7.3.2.4).
        let _ = sps_bit(br.read_bit())?;
    }
    let _w = sps_bit(br.read_ue())?;
    let _h = sps_bit(br.read_ue())?;
    if sps_bit(br.read_bit())? {
        for _ in 0..4 {
            let _ = sps_bit(br.read_ue())?;
        }
    }
    let _subpic = sps_bit(br.read_bit())?;
    let _bd = sps_bit(br.read_ue())?;

    let wpp_enabled = sps_bit(br.read_bit())?;
    let entry_points_present = sps_bit(br.read_bit())?;
    let poc_lsb_bits = (sps_bit(br.read_bits(4))? as u8) + 4;
    let poc_msb_cycle = sps_bit(br.read_bit())?;
    if poc_msb_cycle {
        let _ = sps_bit(br.read_ue())?;
    }
    let extra_ph = sps_bit(br.read_bits(2))?;
    for _ in 0..(extra_ph * 8) {
        let _ = sps_bit(br.read_bit())?;
    }
    let extra_sh = sps_bit(br.read_bits(2))?;
    for _ in 0..(extra_sh * 8) {
        let _ = sps_bit(br.read_bit())?;
    }
    // dpb_parameters (maxSubLayersMinus1 == 0)
    for _ in 0..3 {
        let _ = sps_bit(br.read_ue())?;
    }
    let min_cb_log2 = (sps_bit(br.read_ue())? as u8).saturating_add(2);
    let partition_constraints_override_enabled = sps_bit(br.read_bit())?;
    let log2_diff_min_qt_min_cb_intra_luma = sps_bit(br.read_ue())? as u8;
    let max_mtt_depth_intra_luma = sps_bit(br.read_ue())? as u8;
    let (mut log2_diff_max_bt_min_qt_intra_luma, mut log2_diff_max_tt_min_qt_intra_luma) = (0, 0);
    if max_mtt_depth_intra_luma != 0 {
        log2_diff_max_bt_min_qt_intra_luma = sps_bit(br.read_ue())? as u8;
        log2_diff_max_tt_min_qt_intra_luma = sps_bit(br.read_ue())? as u8;
    }
    let dual_tree_intra = if info.chroma_format_idc != 0 {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    if dual_tree_intra {
        let _min_qt_c = sps_bit(br.read_ue())?;
        let mtt_c = sps_bit(br.read_ue())?;
        if mtt_c != 0 {
            let _ = sps_bit(br.read_ue())?;
            let _ = sps_bit(br.read_ue())?;
        }
    }
    let log2_diff_min_qt_min_cb_inter = sps_bit(br.read_ue())? as u8;
    let max_mtt_depth_inter = sps_bit(br.read_ue())? as u8;
    let (mut log2_diff_max_bt_min_qt_inter, mut log2_diff_max_tt_min_qt_inter) = (0, 0);
    if max_mtt_depth_inter != 0 {
        log2_diff_max_bt_min_qt_inter = sps_bit(br.read_ue())? as u8;
        log2_diff_max_tt_min_qt_inter = sps_bit(br.read_ue())? as u8;
    }
    let max_tb_size_64 = if info.ctu_log2 > 5 {
        sps_bit(br.read_bit())?
    } else {
        false
    };

    let transform_skip_enabled = sps_bit(br.read_bit())?;
    let (mut log2_transform_skip_max_size, mut bdpcm_enabled) = (2u8, false);
    if transform_skip_enabled {
        log2_transform_skip_max_size = (sps_bit(br.read_ue())? as u8).saturating_add(2);
        bdpcm_enabled = sps_bit(br.read_bit())?;
    }
    let mts_enabled = sps_bit(br.read_bit())?;
    let (mut explicit_mts_intra, mut explicit_mts_inter) = (false, false);
    if mts_enabled {
        explicit_mts_intra = sps_bit(br.read_bit())?;
        explicit_mts_inter = sps_bit(br.read_bit())?;
    }
    let lfnst_enabled = sps_bit(br.read_bit())?;

    let mut joint_cbcr_enabled = false;
    let mut same_qp_table_for_chroma = true;
    let mut chroma_qp_tables = Vec::new();
    if info.chroma_format_idc != 0 {
        joint_cbcr_enabled = sps_bit(br.read_bit())?;
        same_qp_table_for_chroma = sps_bit(br.read_bit())?;
        let num_tables = if same_qp_table_for_chroma {
            1
        } else if joint_cbcr_enabled {
            3
        } else {
            2
        };
        for _ in 0..num_tables {
            let start = sps_bit(br.read_se())?;
            let num_pts_minus1 = sps_bit(br.read_ue())? as usize;
            if num_pts_minus1 > 63 {
                return Err(VvcError::Bitstream(
                    "SPS: sps_num_points_in_qp_table_minus1 exceeds 63".to_string(),
                ));
            }
            let mut points = Vec::with_capacity(num_pts_minus1 + 1);
            for _ in 0..=num_pts_minus1 {
                let d_in = sps_bit(br.read_ue())?;
                let d_diff = sps_bit(br.read_ue())?;
                points.push((d_in, d_diff));
            }
            chroma_qp_tables.push(VvcChromaQpTable {
                qp_table_start_minus26: start,
                points,
            });
        }
    }

    let sao_enabled = sps_bit(br.read_bit())?;
    let alf_enabled = sps_bit(br.read_bit())?;
    let ccalf_enabled = if alf_enabled && info.chroma_format_idc != 0 {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    let lmcs_enabled = sps_bit(br.read_bit())?;
    let weighted_pred = sps_bit(br.read_bit())?;
    let weighted_bipred = sps_bit(br.read_bit())?;
    let long_term_ref_pics = sps_bit(br.read_bit())?;
    if info.vps_id > 0 {
        let ilp = sps_bit(br.read_bit())?;
        if ilp {
            return Err(VvcError::Unsupported("SPS: inter-layer prediction"));
        }
    }
    let idr_rpl_present = sps_bit(br.read_bit())?;
    let rpl1_same_as_rpl0 = sps_bit(br.read_bit())?;
    let forbidden_zero_delta_poc = !weighted_pred && !weighted_bipred;
    let num_rpl0 = sps_bit(br.read_ue())? as usize;
    if num_rpl0 > 64 {
        return Err(VvcError::LimitExceeded(
            "SPS: sps_num_ref_pic_lists[0] exceeds 64".to_string(),
        ));
    }
    let mut rpl0 = Vec::with_capacity(num_rpl0);
    for _ in 0..num_rpl0 {
        rpl0.push(parse_ref_pic_list_struct(
            &mut br,
            long_term_ref_pics,
            u32::from(poc_lsb_bits),
            forbidden_zero_delta_poc,
            true,
        )?);
    }
    let rpl1 = if rpl1_same_as_rpl0 {
        rpl0.clone()
    } else {
        let num_rpl1 = sps_bit(br.read_ue())? as usize;
        if num_rpl1 > 64 {
            return Err(VvcError::LimitExceeded(
                "SPS: sps_num_ref_pic_lists[1] exceeds 64".to_string(),
            ));
        }
        let mut list1 = Vec::with_capacity(num_rpl1);
        for _ in 0..num_rpl1 {
            list1.push(parse_ref_pic_list_struct(
                &mut br,
                long_term_ref_pics,
                u32::from(poc_lsb_bits),
                forbidden_zero_delta_poc,
                true,
            )?);
        }
        list1
    };

    let _wraparound = sps_bit(br.read_bit())?;
    let temporal_mvp_enabled = sps_bit(br.read_bit())?;
    let sbtmvp_enabled = if temporal_mvp_enabled {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    let amvr_enabled = sps_bit(br.read_bit())?;
    let bdof_enabled = sps_bit(br.read_bit())?;
    if bdof_enabled {
        let _bdof_ph = sps_bit(br.read_bit())?;
    }
    let smvd_enabled = sps_bit(br.read_bit())?;
    let dmvr_enabled = sps_bit(br.read_bit())?;
    if dmvr_enabled {
        let _dmvr_ph = sps_bit(br.read_bit())?;
    }
    let mmvd_enabled = sps_bit(br.read_bit())?;
    if mmvd_enabled {
        let _fpel_mmvd = sps_bit(br.read_bit())?;
    }
    let six_minus_merge = sps_bit(br.read_ue())?;
    if six_minus_merge > 5 {
        return Err(VvcError::Bitstream(
            "SPS: sps_six_minus_max_num_merge_cand > 5".to_string(),
        ));
    }
    let max_num_merge_cand = (6 - six_minus_merge) as u8;
    let sbt_enabled = sps_bit(br.read_bit())?;
    let affine_enabled = sps_bit(br.read_bit())?;
    let (
        mut max_num_affine_merge_cand,
        mut affine_type_6param,
        mut affine_amvr_enabled,
        mut affine_prof_enabled,
    ) = (if sbtmvp_enabled { 1 } else { 0 }, false, false, false);
    if affine_enabled {
        let five_minus_aff = sps_bit(br.read_ue())?;
        if five_minus_aff > 4 {
            return Err(VvcError::Bitstream(
                "SPS: sps_five_minus_max_num_subblock_merge_cand > 4".to_string(),
            ));
        }
        max_num_affine_merge_cand = (5 - five_minus_aff) as u8;
        affine_type_6param = sps_bit(br.read_bit())?;
        if amvr_enabled {
            affine_amvr_enabled = sps_bit(br.read_bit())?;
        }
        affine_prof_enabled = sps_bit(br.read_bit())?;
        if affine_prof_enabled {
            let _prof_ph = sps_bit(br.read_bit())?;
        }
    }
    let bcw_enabled = sps_bit(br.read_bit())?;
    let ciip_enabled = sps_bit(br.read_bit())?;
    let (mut gpm_enabled, mut max_num_gpm_cand) = (false, 0u8);
    if max_num_merge_cand >= 2 {
        gpm_enabled = sps_bit(br.read_bit())?;
        if gpm_enabled {
            max_num_gpm_cand = if max_num_merge_cand >= 3 {
                let diff = sps_bit(br.read_ue())? as u8;
                max_num_merge_cand.saturating_sub(diff)
            } else {
                2
            };
        }
    }
    let _log2_par_merge = sps_bit(br.read_ue())?;
    let isp_enabled = sps_bit(br.read_bit())?;
    let mrl_enabled = sps_bit(br.read_bit())?;
    let mip_enabled = sps_bit(br.read_bit())?;
    let cclm_enabled = if info.chroma_format_idc != 0 {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    if info.chroma_format_idc == 1 {
        let _hor_col = sps_bit(br.read_bit())?;
        let _ver_col = sps_bit(br.read_bit())?;
    }
    let palette_enabled = sps_bit(br.read_bit())?;
    let act_enabled = if info.chroma_format_idc == 3 && !max_tb_size_64 {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    if transform_skip_enabled || palette_enabled {
        let _internal_bd = sps_bit(br.read_ue())?;
    }
    let ibc_enabled = sps_bit(br.read_bit())?;
    let max_num_ibc_merge_cand = if ibc_enabled {
        let six_minus_ibc = sps_bit(br.read_ue())? as u8;
        6u8.saturating_sub(six_minus_ibc).clamp(1, 6)
    } else {
        1
    };
    let ladf_enabled = sps_bit(br.read_bit())?;
    if ladf_enabled {
        let num_ladf = sps_bit(br.read_bits(2))? + 2;
        let _ = sps_bit(br.read_se())?;
        for _ in 1..num_ladf {
            let _ = sps_bit(br.read_se())?;
            let _ = sps_bit(br.read_ue())?;
        }
    }
    let explicit_scaling_list_enabled = sps_bit(br.read_bit())?;
    let scaling_matrix_for_lfnst_disabled = if lfnst_enabled && explicit_scaling_list_enabled {
        sps_bit(br.read_bit())?
    } else {
        false
    };
    let dep_quant_enabled = sps_bit(br.read_bit())?;
    let sign_data_hiding_enabled = sps_bit(br.read_bit())?;

    Ok(VvcSps {
        info,
        gdr_enabled,
        wpp_enabled,
        entry_points_present,
        poc_lsb_bits,
        min_cb_log2,
        partition_constraints_override_enabled,
        log2_diff_min_qt_min_cb_intra_luma,
        max_mtt_depth_intra_luma,
        log2_diff_max_bt_min_qt_intra_luma,
        log2_diff_max_tt_min_qt_intra_luma,
        dual_tree_intra,
        log2_diff_min_qt_min_cb_inter,
        max_mtt_depth_inter,
        log2_diff_max_bt_min_qt_inter,
        log2_diff_max_tt_min_qt_inter,
        max_tb_size_64,
        transform_skip_enabled,
        log2_transform_skip_max_size,
        bdpcm_enabled,
        mts_enabled,
        explicit_mts_intra,
        explicit_mts_inter,
        lfnst_enabled,
        joint_cbcr_enabled,
        same_qp_table_for_chroma,
        chroma_qp_tables,
        sao_enabled,
        alf_enabled,
        ccalf_enabled,
        lmcs_enabled,
        weighted_pred,
        weighted_bipred,
        long_term_ref_pics,
        idr_rpl_present,
        rpl1_same_as_rpl0,
        rpl_lists: [rpl0, rpl1],
        temporal_mvp_enabled,
        sbtmvp_enabled,
        amvr_enabled,
        bdof_enabled,
        smvd_enabled,
        dmvr_enabled,
        mmvd_enabled,
        max_num_merge_cand,
        sbt_enabled,
        affine_enabled,
        max_num_affine_merge_cand,
        affine_type_6param,
        affine_amvr_enabled,
        affine_prof_enabled,
        bcw_enabled,
        ciip_enabled,
        gpm_enabled,
        max_num_gpm_cand,
        isp_enabled,
        mrl_enabled,
        mip_enabled,
        cclm_enabled,
        explicit_scaling_list_enabled,
        scaling_matrix_for_lfnst_disabled,
        dep_quant_enabled,
        sign_data_hiding_enabled,
        palette_enabled,
        act_enabled,
        ibc_enabled,
        max_num_ibc_merge_cand,
    })
}

/// Complete Picture Parameter Set (`pic_parameter_set_rbsp`, H.266 p. 43–45,
/// PDF p. 59–61, §7.3.2.5; VTM `VLCReader.cpp:438-900`, `VLCWriter.cpp:295-531`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcPps {
    /// `pps_pic_parameter_set_id` (`0..=63`).
    pub pps_id: u8,
    /// `pps_seq_parameter_set_id` (`0..=15`).
    pub sps_id: u8,
    /// `pps_mixed_nalu_types_in_pic_flag`.
    pub mixed_nalu_types: bool,
    /// `pps_pic_width_in_luma_samples`.
    pub width: u32,
    /// `pps_pic_height_in_luma_samples`.
    pub height: u32,
    /// `pps_output_flag_present_flag`.
    pub output_flag_present: bool,
    /// `pps_no_pic_partition_flag`.
    pub no_pic_partition: bool,
    /// `pps_log2_ctu_size_minus5 + 5` when `!no_pic_partition`.
    pub ctu_log2: u8,
    /// Explicit tile column widths in CTUs (`pps_tile_column_width_minus1[i] + 1`).
    pub exp_tile_col_widths: Vec<u32>,
    /// Explicit tile row heights in CTUs (`pps_tile_row_height_minus1[i] + 1`).
    pub exp_tile_row_heights: Vec<u32>,
    /// `pps_loop_filter_across_tiles_enabled_flag`.
    pub loop_filter_across_tiles: bool,
    /// `pps_rect_slice_flag`.
    pub rect_slice_flag: bool,
    /// `pps_single_slice_per_subpic_flag`.
    pub single_slice_per_subpic: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub loop_filter_across_slices: bool,
    /// `pps_cabac_init_present_flag`.
    pub cabac_init_present: bool,
    /// `pps_num_ref_idx_default_active_minus1[i] + 1`.
    pub num_ref_idx_default_active: [u8; 2],
    /// `pps_rpl1_idx_present_flag`.
    pub rpl1_idx_present: bool,
    /// `pps_weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `pps_weighted_bipred_flag`.
    pub weighted_bipred: bool,
    /// `pps_ref_wraparound_enabled_flag`.
    pub ref_wraparound_enabled: bool,
    /// `pps_init_qp_minus26`.
    pub init_qp_minus26: i32,
    /// `pps_cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled: bool,
    /// `pps_chroma_tool_offsets_present_flag`.
    pub chroma_tool_offsets_present: bool,
    /// `pps_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `pps_joint_cbcr_qp_offset_value`.
    pub joint_cbcr_qp_offset: Option<i32>,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub slice_chroma_qp_offsets_present: bool,
    /// `pps_deblocking_filter_control_present_flag`.
    pub deblocking_control_present: bool,
    /// `pps_deblocking_filter_override_enabled_flag`.
    pub deblocking_override_enabled: bool,
    /// `pps_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `pps_dbf_info_in_ph_flag`.
    pub dbf_info_in_ph: bool,
    /// `pps_luma_beta_offset_div2`.
    pub luma_beta_offset_div2: i32,
    /// `pps_luma_tc_offset_div2`.
    pub luma_tc_offset_div2: i32,
    /// `pps_cb_beta_offset_div2`.
    pub cb_beta_offset_div2: i32,
    /// `pps_cb_tc_offset_div2`.
    pub cb_tc_offset_div2: i32,
    /// `pps_cr_beta_offset_div2`.
    pub cr_beta_offset_div2: i32,
    /// `pps_cr_tc_offset_div2`.
    pub cr_tc_offset_div2: i32,
    /// `pps_rpl_info_in_ph_flag`.
    pub rpl_info_in_ph: bool,
    /// `pps_sao_info_in_ph_flag`.
    pub sao_info_in_ph: bool,
    /// `pps_alf_info_in_ph_flag`.
    pub alf_info_in_ph: bool,
    /// `pps_wp_info_in_ph_flag`.
    pub wp_info_in_ph: bool,
    /// `pps_qp_delta_info_in_ph_flag`.
    pub qp_delta_info_in_ph: bool,
}

impl Default for VvcPps {
    fn default() -> Self {
        Self {
            pps_id: 0,
            sps_id: 0,
            mixed_nalu_types: false,
            width: 32,
            height: 32,
            output_flag_present: false,
            no_pic_partition: true,
            ctu_log2: 5,
            exp_tile_col_widths: vec![1],
            exp_tile_row_heights: vec![1],
            loop_filter_across_tiles: true,
            rect_slice_flag: true,
            single_slice_per_subpic: true,
            loop_filter_across_slices: true,
            cabac_init_present: false,
            num_ref_idx_default_active: [1, 1],
            rpl1_idx_present: false,
            weighted_pred: false,
            weighted_bipred: false,
            ref_wraparound_enabled: false,
            init_qp_minus26: 0,
            cu_qp_delta_enabled: false,
            chroma_tool_offsets_present: false,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            joint_cbcr_qp_offset: None,
            slice_chroma_qp_offsets_present: false,
            deblocking_control_present: true,
            deblocking_override_enabled: false,
            deblocking_disabled: true,
            dbf_info_in_ph: false,
            luma_beta_offset_div2: 0,
            luma_tc_offset_div2: 0,
            cb_beta_offset_div2: 0,
            cb_tc_offset_div2: 0,
            cr_beta_offset_div2: 0,
            cr_tc_offset_div2: 0,
            rpl_info_in_ph: false,
            sao_info_in_ph: false,
            alf_info_in_ph: false,
            wp_info_in_ph: false,
            qp_delta_info_in_ph: false,
        }
    }
}

/// Build a complete PPS RBSP (`H.266 §7.3.2.5`, `VLCWriter.cpp:295-531`).
pub fn build_full_pps_rbsp(pps: &VvcPps) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(32);
    bw.write_bits(u32::from(pps.pps_id & 0x3F), 6);
    bw.write_bits(u32::from(pps.sps_id & 0x0F), 4);
    bw.write_bit(pps.mixed_nalu_types);
    bw.write_ue(pps.width);
    bw.write_ue(pps.height);
    bw.write_bit(false); // pps_conformance_window_flag
    bw.write_bit(false); // pps_scaling_window_explicit_signalling_flag
    bw.write_bit(pps.output_flag_present);
    bw.write_bit(pps.no_pic_partition);
    bw.write_bit(false); // pps_subpic_id_mapping_present_flag
    if !pps.no_pic_partition {
        bw.write_bits(u32::from(pps.ctu_log2.clamp(5, 7) - 5), 2);
        let num_cols = pps.exp_tile_col_widths.len().max(1);
        let num_rows = pps.exp_tile_row_heights.len().max(1);
        bw.write_ue((num_cols - 1) as u32);
        bw.write_ue((num_rows - 1) as u32);
        for &w in &pps.exp_tile_col_widths {
            bw.write_ue(w.saturating_sub(1));
        }
        for &h in &pps.exp_tile_row_heights {
            bw.write_ue(h.saturating_sub(1));
        }
        let multi_tile = num_cols * num_rows > 1
            || pps.exp_tile_col_widths.iter().sum::<u32>()
                < pps.width.div_ceil(1u32 << pps.ctu_log2)
            || pps.exp_tile_row_heights.iter().sum::<u32>()
                < pps.height.div_ceil(1u32 << pps.ctu_log2);
        if multi_tile {
            bw.write_bit(pps.loop_filter_across_tiles);
            bw.write_bit(pps.rect_slice_flag);
        }
        if pps.rect_slice_flag {
            bw.write_bit(pps.single_slice_per_subpic);
            if !pps.single_slice_per_subpic {
                bw.write_ue(0); // pps_num_slices_in_pic_minus1 = 0
            }
        }
        if !pps.rect_slice_flag || pps.single_slice_per_subpic {
            bw.write_bit(pps.loop_filter_across_slices);
        }
    }
    bw.write_bit(pps.cabac_init_present);
    bw.write_ue(u32::from(
        pps.num_ref_idx_default_active[0].saturating_sub(1),
    ));
    bw.write_ue(u32::from(
        pps.num_ref_idx_default_active[1].saturating_sub(1),
    ));
    bw.write_bit(pps.rpl1_idx_present);
    bw.write_bit(pps.weighted_pred);
    bw.write_bit(pps.weighted_bipred);
    bw.write_bit(pps.ref_wraparound_enabled);
    bw.write_se(pps.init_qp_minus26);
    bw.write_bit(pps.cu_qp_delta_enabled);
    bw.write_bit(pps.chroma_tool_offsets_present);
    if pps.chroma_tool_offsets_present {
        bw.write_se(pps.cb_qp_offset);
        bw.write_se(pps.cr_qp_offset);
        bw.write_bit(pps.joint_cbcr_qp_offset.is_some());
        if let Some(j_off) = pps.joint_cbcr_qp_offset {
            bw.write_se(j_off);
        }
        bw.write_bit(pps.slice_chroma_qp_offsets_present);
        bw.write_bit(false); // pps_cu_chroma_qp_offset_list_enabled_flag
    }
    bw.write_bit(pps.deblocking_control_present);
    if pps.deblocking_control_present {
        bw.write_bit(pps.deblocking_override_enabled);
        bw.write_bit(pps.deblocking_disabled);
        if !pps.no_pic_partition && pps.deblocking_override_enabled {
            bw.write_bit(pps.dbf_info_in_ph);
        }
        if !pps.deblocking_disabled {
            bw.write_se(pps.luma_beta_offset_div2);
            bw.write_se(pps.luma_tc_offset_div2);
            if pps.chroma_tool_offsets_present {
                bw.write_se(pps.cb_beta_offset_div2);
                bw.write_se(pps.cb_tc_offset_div2);
                bw.write_se(pps.cr_beta_offset_div2);
                bw.write_se(pps.cr_tc_offset_div2);
            }
        }
    }
    if !pps.no_pic_partition {
        bw.write_bit(pps.rpl_info_in_ph);
        bw.write_bit(pps.sao_info_in_ph);
        bw.write_bit(pps.alf_info_in_ph);
        if (pps.weighted_pred || pps.weighted_bipred) && pps.rpl_info_in_ph {
            bw.write_bit(pps.wp_info_in_ph);
        }
        bw.write_bit(pps.qp_delta_info_in_ph);
    }
    bw.write_bit(false); // pps_picture_header_extension_present_flag
    bw.write_bit(false); // pps_slice_header_extension_present_flag
    bw.write_bit(false); // pps_extension_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Parse a complete PPS RBSP (`H.266 §7.3.2.5`, `VLCReader.cpp:438-900`).
pub fn parse_full_pps_rbsp(rbsp: &[u8]) -> Result<VvcPps, VvcError> {
    let mut br = BitReader::new(rbsp);
    let pps_id = hls_bit("PPS", br.read_bits(6))? as u8;
    let sps_id = hls_bit("PPS", br.read_bits(4))? as u8;
    let mixed_nalu_types = hls_bit("PPS", br.read_bit())?;
    let width = hls_bit("PPS", br.read_ue())?;
    let height = hls_bit("PPS", br.read_ue())?;
    if hls_bit("PPS", br.read_bit())? {
        for _ in 0..4 {
            let _ = hls_bit("PPS", br.read_ue())?;
        }
    }
    if hls_bit("PPS", br.read_bit())? {
        for _ in 0..4 {
            let _ = hls_bit("PPS", br.read_se())?;
        }
    }
    let output_flag_present = hls_bit("PPS", br.read_bit())?;
    let no_pic_partition = hls_bit("PPS", br.read_bit())?;
    if hls_bit("PPS", br.read_bit())? {
        let num_subpics = if !no_pic_partition {
            hls_bit("PPS", br.read_ue())?.saturating_add(1)
        } else {
            1
        };
        let id_len = hls_bit("PPS", br.read_ue())?.saturating_add(1) as usize;
        if id_len == 0 || id_len > 16 || num_subpics > 1024 {
            return Err(VvcError::Bitstream(
                "PPS: invalid sub-picture ID mapping".to_string(),
            ));
        }
        for _ in 0..num_subpics {
            let _ = hls_bit("PPS", br.read_bits(id_len as u32))?;
        }
    }
    let mut ctu_log2 = 5u8;
    let mut exp_tile_col_widths = vec![1u32];
    let mut exp_tile_row_heights = vec![1u32];
    let mut loop_filter_across_tiles = true;
    let mut rect_slice_flag = true;
    let mut single_slice_per_subpic = true;
    let mut loop_filter_across_slices = true;

    if !no_pic_partition {
        ctu_log2 = (hls_bit("PPS", br.read_bits(2))? as u8) + 5;
        let num_exp_cols = (hls_bit("PPS", br.read_ue())? as usize) + 1;
        let num_exp_rows = (hls_bit("PPS", br.read_ue())? as usize) + 1;
        if num_exp_cols > 256 || num_exp_rows > 256 {
            return Err(VvcError::LimitExceeded(
                "PPS: explicit tile count exceeds 256".to_string(),
            ));
        }
        exp_tile_col_widths.clear();
        for _ in 0..num_exp_cols {
            exp_tile_col_widths.push(hls_bit("PPS", br.read_ue())? + 1);
        }
        exp_tile_row_heights.clear();
        for _ in 0..num_exp_rows {
            exp_tile_row_heights.push(hls_bit("PPS", br.read_ue())? + 1);
        }
        let ctu_size = 1u32 << ctu_log2;
        let pic_w_ctu = width.div_ceil(ctu_size);
        let pic_h_ctu = height.div_ceil(ctu_size);
        let multi_tile = num_exp_cols * num_exp_rows > 1
            || exp_tile_col_widths.iter().sum::<u32>() < pic_w_ctu
            || exp_tile_row_heights.iter().sum::<u32>() < pic_h_ctu;
        if multi_tile {
            loop_filter_across_tiles = hls_bit("PPS", br.read_bit())?;
            rect_slice_flag = hls_bit("PPS", br.read_bit())?;
        }
        if rect_slice_flag {
            single_slice_per_subpic = hls_bit("PPS", br.read_bit())?;
            if !single_slice_per_subpic {
                let num_slices_minus1 = hls_bit("PPS", br.read_ue())?;
                if num_slices_minus1 > 0 {
                    let tile_idx_delta_present = hls_bit("PPS", br.read_bit())?;
                    for i in 0..num_slices_minus1 {
                        let _ = hls_bit("PPS", br.read_ue())?;
                        let _ = hls_bit("PPS", br.read_ue())?;
                        if tile_idx_delta_present && i + 1 < num_slices_minus1 {
                            let _ = hls_bit("PPS", br.read_se())?;
                        }
                    }
                }
            }
        }
        if !rect_slice_flag || single_slice_per_subpic {
            loop_filter_across_slices = hls_bit("PPS", br.read_bit())?;
        }
    }

    let cabac_init_present = hls_bit("PPS", br.read_bit())?;
    let active0 = (hls_bit("PPS", br.read_ue())? as u8).saturating_add(1);
    let active1 = (hls_bit("PPS", br.read_ue())? as u8).saturating_add(1);
    let rpl1_idx_present = hls_bit("PPS", br.read_bit())?;
    let weighted_pred = hls_bit("PPS", br.read_bit())?;
    let weighted_bipred = hls_bit("PPS", br.read_bit())?;
    let ref_wraparound_enabled = hls_bit("PPS", br.read_bit())?;
    if ref_wraparound_enabled {
        let _ = hls_bit("PPS", br.read_ue())?;
    }
    let init_qp_minus26 = hls_bit("PPS", br.read_se())?;
    let cu_qp_delta_enabled = hls_bit("PPS", br.read_bit())?;
    let chroma_tool_offsets_present = hls_bit("PPS", br.read_bit())?;
    let (mut cb_qp_offset, mut cr_qp_offset) = (0, 0);
    let mut joint_cbcr_qp_offset = None;
    let mut slice_chroma_qp_offsets_present = false;
    if chroma_tool_offsets_present {
        cb_qp_offset = hls_bit("PPS", br.read_se())?;
        cr_qp_offset = hls_bit("PPS", br.read_se())?;
        if hls_bit("PPS", br.read_bit())? {
            joint_cbcr_qp_offset = Some(hls_bit("PPS", br.read_se())?);
        }
        slice_chroma_qp_offsets_present = hls_bit("PPS", br.read_bit())?;
        if hls_bit("PPS", br.read_bit())? {
            let list_len = (hls_bit("PPS", br.read_ue())? as usize) + 1;
            for _ in 0..list_len {
                let _ = hls_bit("PPS", br.read_se())?;
                let _ = hls_bit("PPS", br.read_se())?;
                if joint_cbcr_qp_offset.is_some() {
                    let _ = hls_bit("PPS", br.read_se())?;
                }
            }
        }
    }

    let deblocking_control_present = hls_bit("PPS", br.read_bit())?;
    let mut deblocking_override_enabled = false;
    let mut deblocking_disabled = false;
    let mut dbf_info_in_ph = false;
    let (
        mut luma_beta_offset_div2,
        mut luma_tc_offset_div2,
        mut cb_beta_offset_div2,
        mut cb_tc_offset_div2,
        mut cr_beta_offset_div2,
        mut cr_tc_offset_div2,
    ) = (0, 0, 0, 0, 0, 0);
    if deblocking_control_present {
        deblocking_override_enabled = hls_bit("PPS", br.read_bit())?;
        deblocking_disabled = hls_bit("PPS", br.read_bit())?;
        if !no_pic_partition && deblocking_override_enabled {
            dbf_info_in_ph = hls_bit("PPS", br.read_bit())?;
        }
        if !deblocking_disabled {
            luma_beta_offset_div2 = hls_bit("PPS", br.read_se())?;
            luma_tc_offset_div2 = hls_bit("PPS", br.read_se())?;
            if chroma_tool_offsets_present {
                cb_beta_offset_div2 = hls_bit("PPS", br.read_se())?;
                cb_tc_offset_div2 = hls_bit("PPS", br.read_se())?;
                cr_beta_offset_div2 = hls_bit("PPS", br.read_se())?;
                cr_tc_offset_div2 = hls_bit("PPS", br.read_se())?;
            }
        }
    }

    let (
        mut rpl_info_in_ph,
        mut sao_info_in_ph,
        mut alf_info_in_ph,
        mut wp_info_in_ph,
        mut qp_delta_info_in_ph,
    ) = (false, false, false, false, false);
    if !no_pic_partition {
        rpl_info_in_ph = hls_bit("PPS", br.read_bit())?;
        sao_info_in_ph = hls_bit("PPS", br.read_bit())?;
        alf_info_in_ph = hls_bit("PPS", br.read_bit())?;
        if (weighted_pred || weighted_bipred) && rpl_info_in_ph {
            wp_info_in_ph = hls_bit("PPS", br.read_bit())?;
        }
        qp_delta_info_in_ph = hls_bit("PPS", br.read_bit())?;
    }

    Ok(VvcPps {
        pps_id,
        sps_id,
        mixed_nalu_types,
        width,
        height,
        output_flag_present,
        no_pic_partition,
        ctu_log2,
        exp_tile_col_widths,
        exp_tile_row_heights,
        loop_filter_across_tiles,
        rect_slice_flag,
        single_slice_per_subpic,
        loop_filter_across_slices,
        cabac_init_present,
        num_ref_idx_default_active: [active0, active1],
        rpl1_idx_present,
        weighted_pred,
        weighted_bipred,
        ref_wraparound_enabled,
        init_qp_minus26,
        cu_qp_delta_enabled,
        chroma_tool_offsets_present,
        cb_qp_offset,
        cr_qp_offset,
        joint_cbcr_qp_offset,
        slice_chroma_qp_offsets_present,
        deblocking_control_present,
        deblocking_override_enabled,
        deblocking_disabled,
        dbf_info_in_ph,
        luma_beta_offset_div2,
        luma_tc_offset_div2,
        cb_beta_offset_div2,
        cb_tc_offset_div2,
        cr_beta_offset_div2,
        cr_tc_offset_div2,
        rpl_info_in_ph,
        sao_info_in_ph,
        alf_info_in_ph,
        wp_info_in_ph,
        qp_delta_info_in_ph,
    })
}

/// Adaptation Parameter Set type (`aps_params_type`, H.266 p. 118, PDF p. 134, Table 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcApsType {
    /// Adaptive Loop Filter parameters (`ALF_APS == 0`).
    Alf = 0,
    /// Luma Mapping with Chroma Scaling parameters (`LMCS_APS == 1`).
    Lmcs = 1,
    /// Scaling List parameters (`SCALING_LIST_APS == 2`).
    ScalingList = 2,
}

/// ALF & CCALF parameters carried in an `ALF_APS` (`H.266 p. 52–53, PDF p. 68–69, §7.3.2.18`;
/// VTM `VLCReader.cpp:942-1084, 5203-5255`, `VLCWriter.cpp:559-649, 3141-3180`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcAlfAps {
    /// `alf_luma_clip_flag`.
    pub luma_clip: bool,
    /// Signalled 12-coefficient 7x7 diamond luma filters (`1..=25`).
    pub luma_filters: Vec<[i16; 12]>,
    /// Per-filter 2-bit clipping indices when `luma_clip == true`.
    pub luma_clip_idx: Vec<[u8; 12]>,
    /// Class-to-filter mapping (`0..luma_filters.len()`) for the 25 ALF classes.
    pub filter_coeff_delta_idx: [u8; 25],
    /// `alf_chroma_clip_flag`.
    pub chroma_clip: bool,
    /// Signalled 6-coefficient 5x5 diamond chroma filter alternatives (`0..=8`).
    pub chroma_filters: Vec<[i16; 6]>,
    /// Per-alternative 2-bit chroma clipping indices when `chroma_clip == true`.
    pub chroma_clip_idx: Vec<[u8; 6]>,
    /// Signalled 7-coefficient 3x4 CCALF filters for Cb (`0..=4`).
    pub cc_cb_filters: Vec<[i16; 7]>,
    /// Signalled 7-coefficient 3x4 CCALF filters for Cr (`0..=4`).
    pub cc_cr_filters: Vec<[i16; 7]>,
}

/// LMCS parameters carried in an `LMCS_APS` (`H.266 p. 53, PDF p. 69, §7.3.2.19`;
/// VTM `VLCReader.cpp:1085-1120`, `VLCWriter.cpp:650-680`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcLmcsAps {
    /// `lmcs_min_bin_idx` (`0..=15`).
    pub min_bin_idx: u8,
    /// `15 - lmcs_delta_max_bin_idx` (`min_bin_idx..=15`).
    pub max_bin_idx: u8,
    /// Signed codeword deltas for bins `0..16` (`LmcsDeltaCW[i]`).
    pub bin_cw_delta: [i16; 16],
    /// Signed chroma residual scaling offset (`LmcsDeltaCrs`, `-7..=7`).
    pub chroma_res_scaling_offset: i8,
}

/// Scaling list entry for one `scalingListId` (`0..28`) in a `SCALING_LIST_APS`
/// (`H.266 p. 54, PDF p. 70, §7.3.2.20`; VTM `VLCReader.cpp:5059-5200`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcScalingListEntry {
    /// DC coefficient (`scaling_list_dc_coef` offset from 16) for `id >= 14` (16x16..64x64).
    pub dc_coef: i32,
    /// Flattened row-major matrix (`4`, `16`, or `64` entries).
    pub matrix: Vec<i32>,
}

///Parsed or constructed Adaptation Parameter Set (`adaptation_parameter_set_rbsp`,
/// H.266 p. 46, PDF p. 62, §7.3.2.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VvcApsPayload {
    /// ALF / CCALF APS (`aps_params_type == 0`).
    Alf(VvcAlfAps),
    /// LMCS APS (`aps_params_type == 1`).
    Lmcs(VvcLmcsAps),
    /// Scaling List APS (`aps_params_type == 2`), up to 28 matrices.
    ScalingList(Vec<VvcScalingListEntry>),
}

/// Complete Adaptation Parameter Set header + payload (`H.266 §7.3.2.6`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcAps {
    /// `adaptation_parameter_set_id` (5-bit; 0..=7 for ALF/ScalingList, 0..=3 for LMCS).
    pub aps_id: u8,
    /// `aps_chroma_present_flag`.
    pub chroma_present: bool,
    /// Typed APS payload.
    pub payload: VvcApsPayload,
}

/// Build an APS RBSP (`H.266 §7.3.2.6`, `VLCWriter.cpp:533-687`).
pub fn build_aps_rbsp(aps: &VvcAps) -> Result<Vec<u8>, VvcError> {
    let mut bw = BitWriter::with_capacity(64);
    let aps_type = match &aps.payload {
        VvcApsPayload::Alf(_) => VvcApsType::Alf,
        VvcApsPayload::Lmcs(_) => VvcApsType::Lmcs,
        VvcApsPayload::ScalingList(_) => VvcApsType::ScalingList,
    };
    bw.write_bits(aps_type as u32, 3); // aps_params_type
    bw.write_bits(u32::from(aps.aps_id & 0x1F), 5); // adaptation_parameter_set_id
    bw.write_bit(aps.chroma_present); // aps_chroma_present_flag

    match &aps.payload {
        VvcApsPayload::Alf(alf) => {
            let has_luma = !alf.luma_filters.is_empty();
            let has_chroma = aps.chroma_present && !alf.chroma_filters.is_empty();
            let has_cc_cb = aps.chroma_present && !alf.cc_cb_filters.is_empty();
            let has_cc_cr = aps.chroma_present && !alf.cc_cr_filters.is_empty();
            bw.write_bit(has_luma); // alf_luma_filter_signal_flag
            if aps.chroma_present {
                bw.write_bit(has_chroma); // alf_chroma_filter_signal_flag
                bw.write_bit(has_cc_cb); // alf_cc_cb_filter_signal_flag
                bw.write_bit(has_cc_cr); // alf_cc_cr_filter_signal_flag
            }
            if has_luma {
                if alf.luma_filters.len() > 25 {
                    return Err(VvcError::InvalidConfig(
                        "ALF: at most 25 luma filters allowed".to_string(),
                    ));
                }
                bw.write_bit(alf.luma_clip); // alf_luma_clip_flag
                bw.write_ue((alf.luma_filters.len() - 1) as u32);
                if alf.luma_filters.len() > 1 {
                    let bits = ceil_log2(alf.luma_filters.len() as u32);
                    for &idx in &alf.filter_coeff_delta_idx {
                        bw.write_bits(u32::from(idx), bits);
                    }
                }
                for f in &alf.luma_filters {
                    for &c in f {
                        let abs = i32::from(c).unsigned_abs();
                        bw.write_ue(abs);
                        if abs != 0 {
                            bw.write_bit(c < 0);
                        }
                    }
                }
                if alf.luma_clip {
                    for clip in &alf.luma_clip_idx {
                        for &ci in clip {
                            bw.write_bits(u32::from(ci & 3), 2);
                        }
                    }
                }
            }
            if has_chroma {
                if alf.chroma_filters.len() > 8 {
                    return Err(VvcError::InvalidConfig(
                        "ALF: at most 8 chroma filter alternatives allowed".to_string(),
                    ));
                }
                bw.write_bit(alf.chroma_clip);
                bw.write_ue((alf.chroma_filters.len() - 1) as u32);
                for f in &alf.chroma_filters {
                    for &c in f {
                        let abs = i32::from(c).unsigned_abs();
                        bw.write_ue(abs);
                        if abs != 0 {
                            bw.write_bit(c < 0);
                        }
                    }
                }
                if alf.chroma_clip {
                    for clip in &alf.chroma_clip_idx {
                        for &ci in clip {
                            bw.write_bits(u32::from(ci & 3), 2);
                        }
                    }
                }
            }
            for cc_filters in [&alf.cc_cb_filters, &alf.cc_cr_filters] {
                if aps.chroma_present && !cc_filters.is_empty() {
                    if cc_filters.len() > 4 {
                        return Err(VvcError::InvalidConfig(
                            "CCALF: at most 4 filters per component".to_string(),
                        ));
                    }
                    bw.write_ue((cc_filters.len() - 1) as u32);
                    for f in cc_filters {
                        for &c in f {
                            if c == 0 {
                                bw.write_bits(0, 3);
                            } else {
                                let abs = i32::from(c).unsigned_abs().clamp(1, 64);
                                let mapped = 1 + (31 - abs.leading_zeros());
                                bw.write_bits(mapped, 3);
                                bw.write_bit(c < 0);
                            }
                        }
                    }
                }
            }
        }
        VvcApsPayload::Lmcs(lmcs) => {
            if lmcs.min_bin_idx > lmcs.max_bin_idx || lmcs.max_bin_idx > 15 {
                return Err(VvcError::InvalidConfig(
                    "LMCS: invalid bin range".to_string(),
                ));
            }
            bw.write_ue(u32::from(lmcs.min_bin_idx));
            bw.write_ue(u32::from(15 - lmcs.max_bin_idx));
            let mut max_abs = 0u32;
            for i in usize::from(lmcs.min_bin_idx)..=usize::from(lmcs.max_bin_idx) {
                max_abs = max_abs.max(i32::from(lmcs.bin_cw_delta[i]).unsigned_abs());
            }
            let prec = if max_abs <= 1 {
                1
            } else {
                32 - max_abs.leading_zeros()
            };
            bw.write_ue(prec - 1); // lmcs_delta_cw_prec_minus1
            for i in usize::from(lmcs.min_bin_idx)..=usize::from(lmcs.max_bin_idx) {
                let d = i32::from(lmcs.bin_cw_delta[i]);
                let abs = d.unsigned_abs();
                bw.write_bits(abs, prec);
                if abs > 0 {
                    bw.write_bit(d < 0);
                }
            }
            if aps.chroma_present {
                let d_crs = i32::from(lmcs.chroma_res_scaling_offset);
                let abs_crs = d_crs.unsigned_abs().min(7);
                bw.write_bits(abs_crs, 3);
                if abs_crs > 0 {
                    bw.write_bit(d_crs < 0);
                }
            }
        }
        VvcApsPayload::ScalingList(_lists) => {
            // Write identity copy-mode scaling list flags (`scaling_list_copy_mode_flag = 1`,
            // `scaling_list_pred_matrix_id_delta = 0` => default flat 16 matrices).
            for id in 0..28u32 {
                let is_luma = matches!(id, 2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 | 26 | 27);
                if aps.chroma_present || is_luma {
                    bw.write_bit(true); // scaling_list_copy_mode_flag
                    if id != 0 && id != 2 && id != 8 {
                        bw.write_ue(0); // scaling_list_pred_matrix_id_delta = 0 (default)
                    }
                }
            }
        }
    }
    bw.write_bit(false); // aps_extension_flag
    bw.rbsp_trailing_bits();
    Ok(bw.into_bytes())
}

/// Parse an APS RBSP (`H.266 §7.3.2.6`, `VLCReader.cpp:901-1225`).
pub fn parse_aps_rbsp(rbsp: &[u8]) -> Result<VvcAps, VvcError> {
    let mut br = BitReader::new(rbsp);
    let params_type = hls_bit("APS", br.read_bits(3))?;
    let aps_id = hls_bit("APS", br.read_bits(5))? as u8;
    let chroma_present = hls_bit("APS", br.read_bit())?;

    let payload = match params_type {
        0 => {
            let has_luma = hls_bit("APS", br.read_bit())?;
            let (mut has_chroma, mut has_cc_cb, mut has_cc_cr) = (false, false, false);
            if chroma_present {
                has_chroma = hls_bit("APS", br.read_bit())?;
                has_cc_cb = hls_bit("APS", br.read_bit())?;
                has_cc_cr = hls_bit("APS", br.read_bit())?;
            }
            if !has_luma && !has_chroma && !has_cc_cb && !has_cc_cr {
                return Err(VvcError::Bitstream(
                    "ALF APS: at least one filter signal flag must be 1".to_string(),
                ));
            }
            let mut luma_clip = false;
            let mut luma_filters = Vec::new();
            let mut luma_clip_idx = Vec::new();
            let mut filter_coeff_delta_idx = [0u8; 25];
            if has_luma {
                luma_clip = hls_bit("APS", br.read_bit())?;
                let num_filters = (hls_bit("APS", br.read_ue())? as usize) + 1;
                if num_filters > 25 {
                    return Err(VvcError::Bitstream(
                        "ALF APS: num_luma_filters > 25".to_string(),
                    ));
                }
                if num_filters > 1 {
                    let bits = ceil_log2(num_filters as u32);
                    for slot in &mut filter_coeff_delta_idx {
                        let idx = hls_bit("APS", br.read_bits(bits))? as u8;
                        if usize::from(idx) >= num_filters {
                            return Err(VvcError::Bitstream(
                                "ALF APS: filter_coeff_delta_idx out of range".to_string(),
                            ));
                        }
                        *slot = idx;
                    }
                }
                for _ in 0..num_filters {
                    let mut coeffs = [0i16; 12];
                    for c in &mut coeffs {
                        let abs = hls_bit("APS", br.read_ue())?;
                        if abs > 128 {
                            return Err(VvcError::Bitstream(
                                "ALF APS: luma coeff exceeds 128".to_string(),
                            ));
                        }
                        let mut val = abs as i16;
                        if val != 0 && hls_bit("APS", br.read_bit())? {
                            val = -val;
                        }
                        *c = val;
                    }
                    luma_filters.push(coeffs);
                }
                if luma_clip {
                    for _ in 0..num_filters {
                        let mut clips = [0u8; 12];
                        for ci in &mut clips {
                            *ci = hls_bit("APS", br.read_bits(2))? as u8;
                        }
                        luma_clip_idx.push(clips);
                    }
                } else {
                    luma_clip_idx.resize(num_filters, [0u8; 12]);
                }
            }

            let mut chroma_clip = false;
            let mut chroma_filters = Vec::new();
            let mut chroma_clip_idx = Vec::new();
            if has_chroma {
                chroma_clip = hls_bit("APS", br.read_bit())?;
                let num_alts = (hls_bit("APS", br.read_ue())? as usize) + 1;
                if num_alts > 8 {
                    return Err(VvcError::Bitstream(
                        "ALF APS: num_chroma_alts > 8".to_string(),
                    ));
                }
                for _ in 0..num_alts {
                    let mut coeffs = [0i16; 6];
                    for c in &mut coeffs {
                        let abs = hls_bit("APS", br.read_ue())?;
                        if abs > 128 {
                            return Err(VvcError::Bitstream(
                                "ALF APS: chroma coeff exceeds 128".to_string(),
                            ));
                        }
                        let mut val = abs as i16;
                        if val != 0 && hls_bit("APS", br.read_bit())? {
                            val = -val;
                        }
                        *c = val;
                    }
                    chroma_filters.push(coeffs);
                    if chroma_clip {
                        let mut clips = [0u8; 6];
                        for ci in &mut clips {
                            *ci = hls_bit("APS", br.read_bits(2))? as u8;
                        }
                        chroma_clip_idx.push(clips);
                    } else {
                        chroma_clip_idx.push([0u8; 6]);
                    }
                }
            }

            let mut cc_cb_filters = Vec::new();
            let mut cc_cr_filters = Vec::new();
            for (flag, dst) in [
                (has_cc_cb, &mut cc_cb_filters),
                (has_cc_cr, &mut cc_cr_filters),
            ] {
                if flag {
                    let count = (hls_bit("APS", br.read_ue())? as usize) + 1;
                    if count > 4 {
                        return Err(VvcError::Bitstream(
                            "CCALF APS: filter count > 4".to_string(),
                        ));
                    }
                    for _ in 0..count {
                        let mut coeffs = [0i16; 7];
                        for c in &mut coeffs {
                            let mapped = hls_bit("APS", br.read_bits(3))?;
                            if mapped != 0 {
                                let sign = hls_bit("APS", br.read_bit())?;
                                let mag = 1i16 << (mapped - 1);
                                *c = if sign { -mag } else { mag };
                            }
                        }
                        dst.push(coeffs);
                    }
                }
            }

            VvcApsPayload::Alf(VvcAlfAps {
                luma_clip,
                luma_filters,
                luma_clip_idx,
                filter_coeff_delta_idx,
                chroma_clip,
                chroma_filters,
                chroma_clip_idx,
                cc_cb_filters,
                cc_cr_filters,
            })
        }
        1 => {
            let min_bin_idx = hls_bit("APS", br.read_ue())? as u8;
            let delta_max = hls_bit("APS", br.read_ue())? as u8;
            if min_bin_idx > 15 || delta_max > 15 || min_bin_idx > 15 - delta_max {
                return Err(VvcError::Bitstream(
                    "LMCS APS: invalid bin indices".to_string(),
                ));
            }
            let max_bin_idx = 15 - delta_max;
            let prec = hls_bit("APS", br.read_ue())? + 1;
            if prec > 15 {
                return Err(VvcError::Bitstream(
                    "LMCS APS: lmcs_delta_cw_prec exceeds 15".to_string(),
                ));
            }
            let mut bin_cw_delta = [0i16; 16];
            for i in usize::from(min_bin_idx)..=usize::from(max_bin_idx) {
                let abs = hls_bit("APS", br.read_bits(prec))? as i16;
                let sign = if abs > 0 {
                    hls_bit("APS", br.read_bit())?
                } else {
                    false
                };
                bin_cw_delta[i] = if sign { -abs } else { abs };
            }
            let mut chroma_res_scaling_offset = 0i8;
            if chroma_present {
                let abs_crs = hls_bit("APS", br.read_bits(3))? as i8;
                let sign_crs = if abs_crs > 0 {
                    hls_bit("APS", br.read_bit())?
                } else {
                    false
                };
                chroma_res_scaling_offset = if sign_crs { -abs_crs } else { abs_crs };
            }
            VvcApsPayload::Lmcs(VvcLmcsAps {
                min_bin_idx,
                max_bin_idx,
                bin_cw_delta,
                chroma_res_scaling_offset,
            })
        }
        2 => {
            let mut entries = Vec::with_capacity(28);
            for id in 0..28u32 {
                let mat_size = if id < 2 {
                    2usize
                } else if id < 8 {
                    4
                } else {
                    8
                };
                let is_luma = matches!(id, 2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 | 26 | 27);
                if chroma_present || is_luma {
                    let copy_mode = hls_bit("APS", br.read_bit())?;
                    let pred_mode = if !copy_mode {
                        hls_bit("APS", br.read_bit())?
                    } else {
                        false
                    };
                    let delta_id = if (copy_mode || pred_mode) && id != 0 && id != 2 && id != 8 {
                        hls_bit("APS", br.read_ue())?
                    } else {
                        0
                    };
                    if delta_id > id {
                        return Err(VvcError::Bitstream(
                            "ScalingList APS: pred_matrix_id_delta > scalingListId".to_string(),
                        ));
                    }
                    let ref_id = (id - delta_id) as usize;
                    if copy_mode {
                        if ref_id == id as usize {
                            entries.push(VvcScalingListEntry {
                                dc_coef: 16,
                                matrix: vec![16; mat_size * mat_size],
                            });
                        } else {
                            let cloned = entries[ref_id].clone();
                            entries.push(cloned);
                        }
                    } else {
                        let mut next_coef = if pred_mode { 0i32 } else { 8i32 };
                        let ref_dc = if pred_mode {
                            if ref_id == id as usize || ref_id < 14 {
                                16
                            } else {
                                entries[ref_id].dc_coef
                            }
                        } else {
                            0
                        };
                        let dc_coef = if id >= 14 {
                            let delta = hls_bit("APS", br.read_se())?;
                            next_coef = (next_coef + delta) & 0xFF;
                            (ref_dc + next_coef) & 0xFF
                        } else {
                            16
                        };
                        let coef_num = mat_size * mat_size;
                        let mut matrix = vec![16i32; coef_num];
                        for i in 0..coef_num {
                            if id >= 26 && (i % 8) >= 4 && (i / 8) >= 4 {
                                matrix[i] = 0;
                                continue;
                            }
                            let delta = hls_bit("APS", br.read_se())?;
                            next_coef = (next_coef + delta) & 0xFF;
                            let pred_val = if pred_mode {
                                if ref_id == id as usize {
                                    16
                                } else {
                                    entries[ref_id].matrix.get(i).copied().unwrap_or(16)
                                }
                            } else {
                                0
                            };
                            matrix[i] = (pred_val + next_coef) & 0xFF;
                        }
                        entries.push(VvcScalingListEntry { dc_coef, matrix });
                    }
                } else {
                    entries.push(VvcScalingListEntry {
                        dc_coef: 16,
                        matrix: vec![16; mat_size * mat_size],
                    });
                }
            }
            VvcApsPayload::ScalingList(entries)
        }
        other => {
            return Err(VvcError::Unsupported(match other {
                _ => "APS: unknown aps_params_type",
            }));
        }
    };
    Ok(VvcAps {
        aps_id,
        chroma_present,
        payload,
    })
}

/// Complete Picture Header structure (`picture_header_structure`, H.266 p. 47–50,
/// PDF p. 63–66, §7.3.2.8; VTM `VLCReader.cpp:2609-3549`, `VLCWriter.cpp:1589-2211`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcPictureHeader {
    /// `ph_gdr_or_irap_pic_flag`.
    pub gdr_or_irap_pic: bool,
    /// `ph_non_ref_pic_flag`.
    pub non_ref_pic: bool,
    /// `ph_gdr_pic_flag`.
    pub gdr_pic: bool,
    /// `ph_inter_slice_allowed_flag`.
    pub inter_slice_allowed: bool,
    /// `ph_intra_slice_allowed_flag`.
    pub intra_slice_allowed: bool,
    /// `ph_pic_parameter_set_id`.
    pub pps_id: u8,
    /// `ph_pic_order_cnt_lsb`.
    pub poc_lsb: u32,
    /// `ph_recovery_poc_cnt` when `gdr_pic == true`.
    pub recovery_poc_cnt: u32,
    /// `ph_lmcs_enabled_flag`.
    pub lmcs_enabled: bool,
    /// `ph_lmcs_aps_id`.
    pub lmcs_aps_id: u8,
    /// `ph_chroma_residual_scale_flag`.
    pub chroma_residual_scale: bool,
    /// `ph_Explicit_scaling_list_enabled_flag`.
    pub explicit_scaling_list_enabled: bool,
    /// `ph_scaling_list_aps_id`.
    pub scaling_list_aps_id: u8,
    /// `ph_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `ph_mvd_l1_zero_flag`.
    pub mvd_l1_zero: bool,
    /// `ph_joint_cbcr_sign_flag`.
    pub joint_cbcr_sign: bool,
}

impl Default for VvcPictureHeader {
    fn default() -> Self {
        Self {
            gdr_or_irap_pic: true,
            non_ref_pic: false,
            gdr_pic: false,
            inter_slice_allowed: false,
            intra_slice_allowed: true,
            pps_id: 0,
            poc_lsb: 0,
            recovery_poc_cnt: 0,
            lmcs_enabled: false,
            lmcs_aps_id: 0,
            chroma_residual_scale: false,
            explicit_scaling_list_enabled: false,
            scaling_list_aps_id: 0,
            temporal_mvp_enabled: false,
            mvd_l1_zero: false,
            joint_cbcr_sign: false,
        }
    }
}

/// Write `picture_header_structure` (`H.266 §7.3.2.8`, `VLCWriter.cpp:1589-2211`).
pub fn write_picture_header_structure(
    bw: &mut BitWriter,
    ph: &VvcPictureHeader,
    sps: &VvcSps,
    pps: &VvcPps,
) {
    bw.write_bit(ph.gdr_or_irap_pic);
    bw.write_bit(ph.non_ref_pic);
    if ph.gdr_or_irap_pic {
        bw.write_bit(ph.gdr_pic);
    }
    bw.write_bit(ph.inter_slice_allowed);
    if ph.inter_slice_allowed {
        bw.write_bit(ph.intra_slice_allowed);
    }
    bw.write_ue(u32::from(ph.pps_id));
    bw.write_bits(ph.poc_lsb, u32::from(sps.poc_lsb_bits));
    if ph.gdr_pic {
        bw.write_ue(ph.recovery_poc_cnt);
    }
    if sps.alf_enabled && pps.alf_info_in_ph {
        bw.write_bit(false); // ph_alf_enabled_flag
    }
    if sps.lmcs_enabled {
        bw.write_bit(ph.lmcs_enabled);
        if ph.lmcs_enabled {
            bw.write_bits(u32::from(ph.lmcs_aps_id & 3), 2);
            if sps.info.chroma_format_idc != 0 {
                bw.write_bit(ph.chroma_residual_scale);
            }
        }
    }
    if sps.explicit_scaling_list_enabled {
        bw.write_bit(ph.explicit_scaling_list_enabled);
        if ph.explicit_scaling_list_enabled {
            bw.write_bits(u32::from(ph.scaling_list_aps_id & 7), 3);
        }
    }
    if pps.output_flag_present && !ph.non_ref_pic {
        bw.write_bit(true); // ph_pic_output_flag
    }
    if sps.partition_constraints_override_enabled {
        bw.write_bit(false); // ph_partition_constraints_override_flag
    }
    if ph.intra_slice_allowed && pps.cu_qp_delta_enabled {
        bw.write_ue(0); // ph_cu_qp_delta_subdiv_intra_slice
    }
    if ph.inter_slice_allowed {
        if pps.cu_qp_delta_enabled {
            bw.write_ue(0); // ph_cu_qp_delta_subdiv_inter_slice
        }
        if sps.temporal_mvp_enabled {
            bw.write_bit(ph.temporal_mvp_enabled);
        }
        if !pps.rpl_info_in_ph {
            bw.write_bit(ph.mvd_l1_zero);
        }
    }
    if pps.qp_delta_info_in_ph {
        bw.write_se(0); // ph_qp_delta
    }
    if sps.joint_cbcr_enabled {
        bw.write_bit(ph.joint_cbcr_sign);
    }
    if sps.sao_enabled && pps.sao_info_in_ph {
        bw.write_bit(false);
        if sps.info.chroma_format_idc != 0 {
            bw.write_bit(false);
        }
    }
    if pps.deblocking_control_present && pps.dbf_info_in_ph {
        bw.write_bit(false); // ph_deblocking_params_present_flag
    }
}

/// Build a standalone PH RBSP (`H.266 §7.3.2.7`).
pub fn build_full_ph_rbsp(ph: &VvcPictureHeader, sps: &VvcSps, pps: &VvcPps) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(16);
    write_picture_header_structure(&mut bw, ph, sps, pps);
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

/// Parse `picture_header_structure` (`H.266 §7.3.2.8`, `VLCReader.cpp:2609-3549`).
pub fn parse_picture_header_structure(
    br: &mut BitReader<'_>,
    sps_table: &[Option<VvcSps>; 16],
    pps_table: &[Option<VvcPps>; 64],
) -> Result<VvcPictureHeader, VvcError> {
    let gdr_or_irap_pic = hls_bit("PH", br.read_bit())?;
    let non_ref_pic = hls_bit("PH", br.read_bit())?;
    let gdr_pic = if gdr_or_irap_pic {
        hls_bit("PH", br.read_bit())?
    } else {
        false
    };
    let inter_slice_allowed = hls_bit("PH", br.read_bit())?;
    let intra_slice_allowed = if inter_slice_allowed {
        hls_bit("PH", br.read_bit())?
    } else {
        true
    };
    let pps_id_u32 = hls_bit("PH", br.read_ue())?;
    let pps_id = u8::try_from(pps_id_u32)
        .ok()
        .filter(|&id| id < 64)
        .ok_or_else(|| VvcError::Bitstream(format!("PH: invalid pps_id {pps_id_u32}")))?;
    let pps = pps_table[usize::from(pps_id)]
        .as_ref()
        .ok_or_else(|| VvcError::Bitstream(format!("PH references unknown PPS {pps_id}")))?;
    let sps = sps_table[usize::from(pps.sps_id)]
        .as_ref()
        .ok_or_else(|| VvcError::Bitstream(format!("PPS references unknown SPS {}", pps.sps_id)))?;

    let poc_lsb = hls_bit("PH", br.read_bits(u32::from(sps.poc_lsb_bits)))?;
    let recovery_poc_cnt = if gdr_pic {
        hls_bit("PH", br.read_ue())?
    } else {
        0
    };
    if sps.alf_enabled && pps.alf_info_in_ph {
        let alf_en = hls_bit("PH", br.read_bit())?;
        if alf_en {
            let num_aps = hls_bit("PH", br.read_bits(3))?;
            for _ in 0..num_aps {
                let _ = hls_bit("PH", br.read_bits(3))?;
            }
            if sps.info.chroma_format_idc != 0 {
                let cb = hls_bit("PH", br.read_bit())?;
                let cr = hls_bit("PH", br.read_bit())?;
                if cb || cr {
                    let _ = hls_bit("PH", br.read_bits(3))?;
                }
            }
            if sps.ccalf_enabled {
                if hls_bit("PH", br.read_bit())? {
                    let _ = hls_bit("PH", br.read_bits(3))?;
                }
                if hls_bit("PH", br.read_bit())? {
                    let _ = hls_bit("PH", br.read_bits(3))?;
                }
            }
        }
    }
    let (mut lmcs_enabled, mut lmcs_aps_id, mut chroma_residual_scale) = (false, 0u8, false);
    if sps.lmcs_enabled {
        lmcs_enabled = hls_bit("PH", br.read_bit())?;
        if lmcs_enabled {
            lmcs_aps_id = hls_bit("PH", br.read_bits(2))? as u8;
            if sps.info.chroma_format_idc != 0 {
                chroma_residual_scale = hls_bit("PH", br.read_bit())?;
            }
        }
    }
    let (mut explicit_scaling_list_enabled, mut scaling_list_aps_id) = (false, 0u8);
    if sps.explicit_scaling_list_enabled {
        explicit_scaling_list_enabled = hls_bit("PH", br.read_bit())?;
        if explicit_scaling_list_enabled {
            scaling_list_aps_id = hls_bit("PH", br.read_bits(3))? as u8;
        }
    }
    if pps.output_flag_present && !non_ref_pic {
        let _ = hls_bit("PH", br.read_bit())?;
    }
    if sps.partition_constraints_override_enabled {
        let ovr = hls_bit("PH", br.read_bit())?;
        if ovr {
            if intra_slice_allowed {
                let _ = hls_bit("PH", br.read_ue())?;
                let mtt_y = hls_bit("PH", br.read_ue())?;
                if mtt_y != 0 {
                    let _ = hls_bit("PH", br.read_ue())?;
                    let _ = hls_bit("PH", br.read_ue())?;
                }
                if sps.dual_tree_intra {
                    let _ = hls_bit("PH", br.read_ue())?;
                    let mtt_c = hls_bit("PH", br.read_ue())?;
                    if mtt_c != 0 {
                        let _ = hls_bit("PH", br.read_ue())?;
                        let _ = hls_bit("PH", br.read_ue())?;
                    }
                }
            }
            if inter_slice_allowed {
                let _ = hls_bit("PH", br.read_ue())?;
                let mtt_inter = hls_bit("PH", br.read_ue())?;
                if mtt_inter != 0 {
                    let _ = hls_bit("PH", br.read_ue())?;
                    let _ = hls_bit("PH", br.read_ue())?;
                }
            }
        }
    }
    if intra_slice_allowed && pps.cu_qp_delta_enabled {
        let _ = hls_bit("PH", br.read_ue())?;
    }
    let mut temporal_mvp_enabled = false;
    let mut mvd_l1_zero = false;
    if inter_slice_allowed {
        if pps.cu_qp_delta_enabled {
            let _ = hls_bit("PH", br.read_ue())?;
        }
        if sps.temporal_mvp_enabled {
            temporal_mvp_enabled = hls_bit("PH", br.read_bit())?;
        }
        if !pps.rpl_info_in_ph {
            mvd_l1_zero = hls_bit("PH", br.read_bit())?;
        }
    }
    if pps.qp_delta_info_in_ph {
        let _ = hls_bit("PH", br.read_se())?;
    }
    let joint_cbcr_sign = if sps.joint_cbcr_enabled {
        hls_bit("PH", br.read_bit())?
    } else {
        false
    };
    if sps.sao_enabled && pps.sao_info_in_ph {
        let _ = hls_bit("PH", br.read_bit())?;
        if sps.info.chroma_format_idc != 0 {
            let _ = hls_bit("PH", br.read_bit())?;
        }
    }
    if pps.deblocking_control_present && pps.dbf_info_in_ph {
        let _ = hls_bit("PH", br.read_bit())?;
    }

    Ok(VvcPictureHeader {
        gdr_or_irap_pic,
        non_ref_pic,
        gdr_pic,
        inter_slice_allowed,
        intra_slice_allowed,
        pps_id,
        poc_lsb,
        recovery_poc_cnt,
        lmcs_enabled,
        lmcs_aps_id,
        chroma_residual_scale,
        explicit_scaling_list_enabled,
        scaling_list_aps_id,
        temporal_mvp_enabled,
        mvd_l1_zero,
        joint_cbcr_sign,
    })
}

/// Parse a standalone PH RBSP (`H.266 §7.3.2.7`).
pub fn parse_full_ph_rbsp(
    rbsp: &[u8],
    sps_table: &[Option<VvcSps>; 16],
    pps_table: &[Option<VvcPps>; 64],
) -> Result<VvcPictureHeader, VvcError> {
    let mut br = BitReader::new(rbsp);
    parse_picture_header_structure(&mut br, sps_table, pps_table)
}

/// Complete Slice Header (`slice_header`, H.266 p. 59–62, PDF p. 75–78, §7.3.7;
/// VTM `VLCReader.cpp:3551-4460`, `VLCWriter.cpp:2213-2655`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcSliceHeader {
    /// `sh_picture_header_in_slice_header_flag`.
    pub ph_in_slice_header: bool,
    /// Embedded `picture_header_structure` when `ph_in_slice_header == true`.
    pub embedded_ph: Option<VvcPictureHeader>,
    /// `sh_slice_type`.
    pub slice_type: VvcSliceType,
    /// `sh_no_output_of_prior_pics_flag`.
    pub no_output_of_prior_pics: bool,
    /// `sh_alf_enabled_flag`.
    pub alf_enabled: bool,
    /// `sh_alf_aps_id_luma`.
    pub alf_aps_ids_luma: Vec<u8>,
    /// `sh_alf_cb_enabled_flag`.
    pub alf_cb_enabled: bool,
    /// `sh_alf_cr_enabled_flag`.
    pub alf_cr_enabled: bool,
    /// `sh_alf_aps_id_chroma`.
    pub alf_aps_id_chroma: u8,
    /// `sh_alf_cc_cb_enabled_flag` and `sh_alf_cc_cb_aps_id`.
    pub cc_alf_cb_aps_id: Option<u8>,
    /// `sh_alf_cc_cr_enabled_flag` and `sh_alf_cc_cr_aps_id`.
    pub cc_alf_cr_aps_id: Option<u8>,
    /// `sh_lmcs_used_flag`.
    pub lmcs_used: bool,
    /// `sh_explicit_scaling_list_used_flag`.
    pub explicit_scaling_list_used: bool,
    /// Explicit RPLs signalled in the slice header for L0 and L1.
    pub rpl: [VvcRefPicListStruct; 2],
    /// `NumRefIdxActive[0]` and `NumRefIdxActive[1]`.
    pub num_ref_idx_active: [u8; 2],
    /// `sh_cabac_init_flag`.
    pub cabac_init_flag: bool,
    /// `sh_collocated_from_l0_flag`.
    pub collocated_from_l0: bool,
    /// `sh_collocated_ref_idx`.
    pub collocated_ref_idx: u8,
    /// `sh_qp_delta`.
    pub qp_delta: i32,
    /// `sh_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `sh_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `sh_joint_cbcr_qp_offset`.
    pub joint_cbcr_qp_offset: i32,
    /// `sh_sao_luma_used_flag`.
    pub sao_luma_used: bool,
    /// `sh_sao_chroma_used_flag`.
    pub sao_chroma_used: bool,
    /// `sh_deblocking_filter_disabled_flag`.
    pub deblocking_disabled: bool,
    /// `sh_luma_beta_offset_div2`.
    pub luma_beta_offset_div2: i32,
    /// `sh_luma_tc_offset_div2`.
    pub luma_tc_offset_div2: i32,
    /// `sh_dep_quant_used_flag`.
    pub dep_quant_used: bool,
    /// `sh_sign_data_hiding_used_flag`.
    pub sign_data_hiding_used: bool,
    /// `sh_ts_residual_coding_disabled_flag`.
    pub ts_residual_coding_disabled: bool,
    /// Entry point offsets (`sh_entry_point_offset_minus1[i] + 1`).
    pub entry_point_offsets: Vec<u32>,
}

impl Default for VvcSliceHeader {
    fn default() -> Self {
        Self {
            ph_in_slice_header: false,
            embedded_ph: None,
            slice_type: VvcSliceType::I,
            no_output_of_prior_pics: false,
            alf_enabled: false,
            alf_aps_ids_luma: Vec::new(),
            alf_cb_enabled: false,
            alf_cr_enabled: false,
            alf_aps_id_chroma: 0,
            cc_alf_cb_aps_id: None,
            cc_alf_cr_aps_id: None,
            lmcs_used: false,
            explicit_scaling_list_used: false,
            rpl: [
                VvcRefPicListStruct::default(),
                VvcRefPicListStruct::default(),
            ],
            num_ref_idx_active: [0, 0],
            cabac_init_flag: false,
            collocated_from_l0: true,
            collocated_ref_idx: 0,
            qp_delta: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            joint_cbcr_qp_offset: 0,
            sao_luma_used: false,
            sao_chroma_used: false,
            deblocking_disabled: true,
            luma_beta_offset_div2: 0,
            luma_tc_offset_div2: 0,
            dep_quant_used: false,
            sign_data_hiding_used: false,
            ts_residual_coding_disabled: false,
            entry_point_offsets: Vec::new(),
        }
    }
}

/// Build `slice_header` bytes including trailing `byte_alignment()` (`H.266 §7.3.7`,
/// `VLCWriter.cpp:2213-2655`).
pub fn build_full_slice_header_bits(
    sh: &VvcSliceHeader,
    nal_type: u8,
    ph: &VvcPictureHeader,
    sps: &VvcSps,
    pps: &VvcPps,
) -> Vec<u8> {
    let mut bw = BitWriter::with_capacity(24);
    bw.write_bit(sh.ph_in_slice_header);
    if sh.ph_in_slice_header {
        let hdr = sh.embedded_ph.as_ref().unwrap_or(ph);
        write_picture_header_structure(&mut bw, hdr, sps, pps);
    }
    if ph.inter_slice_allowed {
        bw.write_ue(sh.slice_type as u32);
    }
    if matches!(nal_type, NAL_IDR_W_RADL | NAL_IDR_N_LP | NAL_CRA | NAL_GDR) {
        bw.write_bit(sh.no_output_of_prior_pics);
    }
    if sps.alf_enabled && !pps.alf_info_in_ph {
        bw.write_bit(sh.alf_enabled);
        if sh.alf_enabled {
            let count = sh.alf_aps_ids_luma.len().clamp(1, 7);
            bw.write_bits(count as u32, 3);
            for &id in sh.alf_aps_ids_luma.iter().take(count) {
                bw.write_bits(u32::from(id & 7), 3);
            }
            if sps.info.chroma_format_idc != 0 {
                bw.write_bit(sh.alf_cb_enabled);
                bw.write_bit(sh.alf_cr_enabled);
                if sh.alf_cb_enabled || sh.alf_cr_enabled {
                    bw.write_bits(u32::from(sh.alf_aps_id_chroma & 7), 3);
                }
            }
            if sps.ccalf_enabled {
                bw.write_bit(sh.cc_alf_cb_aps_id.is_some());
                if let Some(id) = sh.cc_alf_cb_aps_id {
                    bw.write_bits(u32::from(id & 7), 3);
                }
                bw.write_bit(sh.cc_alf_cr_aps_id.is_some());
                if let Some(id) = sh.cc_alf_cr_aps_id {
                    bw.write_bits(u32::from(id & 7), 3);
                }
            }
        }
    }
    if ph.lmcs_enabled && !sh.ph_in_slice_header {
        bw.write_bit(sh.lmcs_used);
    }
    if ph.explicit_scaling_list_enabled && !sh.ph_in_slice_header {
        bw.write_bit(sh.explicit_scaling_list_used);
    }
    let is_idr = matches!(nal_type, NAL_IDR_W_RADL | NAL_IDR_N_LP);
    if !pps.rpl_info_in_ph && (!is_idr || sps.idr_rpl_present) {
        let forbidden_zero = !sps.weighted_pred && !sps.weighted_bipred;
        for l in 0..2usize {
            if !sps.rpl_lists[l].is_empty() && (l == 0 || pps.rpl1_idx_present) {
                bw.write_bit(false); // rpl_sps_flag[l] = 0 (inline RPL)
            }
            write_ref_pic_list_struct(
                &mut bw,
                &sh.rpl[l],
                sps.long_term_ref_pics,
                u32::from(sps.poc_lsb_bits),
                forbidden_zero,
                false,
            );
        }
    }
    if (sh.slice_type != VvcSliceType::I && sh.rpl[0].entries.len() > 1)
        || (sh.slice_type == VvcSliceType::B && sh.rpl[1].entries.len() > 1)
    {
        let def0 = pps.num_ref_idx_default_active[0].min(sh.rpl[0].entries.len() as u8);
        let def1 = pps.num_ref_idx_default_active[1].min(sh.rpl[1].entries.len() as u8);
        let ovr = sh.num_ref_idx_active[0] != def0
            || (sh.slice_type == VvcSliceType::B && sh.num_ref_idx_active[1] != def1);
        bw.write_bit(ovr);
        if ovr {
            if sh.rpl[0].entries.len() > 1 {
                bw.write_ue(u32::from(sh.num_ref_idx_active[0].saturating_sub(1)));
            }
            if sh.slice_type == VvcSliceType::B && sh.rpl[1].entries.len() > 1 {
                bw.write_ue(u32::from(sh.num_ref_idx_active[1].saturating_sub(1)));
            }
        }
    }
    if sh.slice_type != VvcSliceType::I {
        if pps.cabac_init_present {
            bw.write_bit(sh.cabac_init_flag);
        }
        if ph.temporal_mvp_enabled && !pps.rpl_info_in_ph {
            if sh.slice_type == VvcSliceType::B {
                bw.write_bit(sh.collocated_from_l0);
            }
            let active_col = if sh.collocated_from_l0 || sh.slice_type == VvcSliceType::P {
                sh.num_ref_idx_active[0]
            } else {
                sh.num_ref_idx_active[1]
            };
            if active_col > 1 {
                bw.write_ue(u32::from(sh.collocated_ref_idx));
            }
        }
    }
    if !pps.qp_delta_info_in_ph {
        bw.write_se(sh.qp_delta);
    }
    if pps.slice_chroma_qp_offsets_present {
        bw.write_se(sh.cb_qp_offset);
        bw.write_se(sh.cr_qp_offset);
        if sps.joint_cbcr_enabled {
            bw.write_se(sh.joint_cbcr_qp_offset);
        }
    }
    if sps.sao_enabled && !pps.sao_info_in_ph {
        bw.write_bit(sh.sao_luma_used);
        if sps.info.chroma_format_idc != 0 {
            bw.write_bit(sh.sao_chroma_used);
        }
    }
    if pps.deblocking_control_present && pps.deblocking_override_enabled && !pps.dbf_info_in_ph {
        bw.write_bit(false); // sh_deblocking_params_present_flag
    }
    if sps.dep_quant_enabled {
        bw.write_bit(sh.dep_quant_used);
    }
    if sps.sign_data_hiding_enabled && !sh.dep_quant_used {
        bw.write_bit(sh.sign_data_hiding_used);
    }
    if sps.transform_skip_enabled && !sh.dep_quant_used && !sh.sign_data_hiding_used {
        bw.write_bit(sh.ts_residual_coding_disabled);
    }
    if sps.entry_points_present && !sh.entry_point_offsets.is_empty() {
        let max_off = sh.entry_point_offsets.iter().copied().max().unwrap_or(1);
        let len_minus1 = (32 - max_off.saturating_sub(1).leading_zeros()).saturating_sub(1);
        bw.write_ue(len_minus1);
        for &off in &sh.entry_point_offsets {
            bw.write_bits(off.saturating_sub(1), len_minus1 + 1);
        }
    }
    bw.byte_alignment();
    bw.into_bytes()
}

/// Parse `slice_header` (`H.266 §7.3.7`, `VLCReader.cpp:3551-4460`), returning
/// `(VvcSliceHeader, byte_offset_of_slice_data)`.
pub fn parse_full_slice_header(
    rbsp: &[u8],
    nal_type: u8,
    active_ph: Option<&VvcPictureHeader>,
    sps_table: &[Option<VvcSps>; 16],
    pps_table: &[Option<VvcPps>; 64],
) -> Result<(VvcSliceHeader, usize), VvcError> {
    let mut br = BitReader::new(rbsp);
    let ph_in_slice_header = hls_bit("SH", br.read_bit())?;
    let embedded_ph = if ph_in_slice_header {
        Some(parse_picture_header_structure(
            &mut br, sps_table, pps_table,
        )?)
    } else {
        None
    };
    let ph = embedded_ph
        .as_ref()
        .or(active_ph)
        .ok_or_else(|| VvcError::Bitstream("SH: missing active Picture Header".to_string()))?;
    let pps = pps_table[usize::from(ph.pps_id)]
        .as_ref()
        .ok_or_else(|| VvcError::Bitstream(format!("SH: unknown PPS {}", ph.pps_id)))?;
    let sps = sps_table[usize::from(pps.sps_id)]
        .as_ref()
        .ok_or_else(|| VvcError::Bitstream(format!("SH: unknown SPS {}", pps.sps_id)))?;

    let slice_type = if ph.inter_slice_allowed {
        VvcSliceType::from_u32(hls_bit("SH", br.read_ue())?)?
    } else {
        VvcSliceType::I
    };
    let no_output_of_prior_pics =
        if matches!(nal_type, NAL_IDR_W_RADL | NAL_IDR_N_LP | NAL_CRA | NAL_GDR) {
            hls_bit("SH", br.read_bit())?
        } else {
            false
        };

    let mut alf_enabled = false;
    let mut alf_aps_ids_luma = Vec::new();
    let (mut alf_cb_enabled, mut alf_cr_enabled, mut alf_aps_id_chroma) = (false, false, 0u8);
    let (mut cc_alf_cb_aps_id, mut cc_alf_cr_aps_id) = (None, None);
    if sps.alf_enabled && !pps.alf_info_in_ph {
        alf_enabled = hls_bit("SH", br.read_bit())?;
        if alf_enabled {
            let num_aps = hls_bit("SH", br.read_bits(3))? as usize;
            for _ in 0..num_aps {
                alf_aps_ids_luma.push(hls_bit("SH", br.read_bits(3))? as u8);
            }
            if sps.info.chroma_format_idc != 0 {
                alf_cb_enabled = hls_bit("SH", br.read_bit())?;
                alf_cr_enabled = hls_bit("SH", br.read_bit())?;
                if alf_cb_enabled || alf_cr_enabled {
                    alf_aps_id_chroma = hls_bit("SH", br.read_bits(3))? as u8;
                }
            }
            if sps.ccalf_enabled {
                if hls_bit("SH", br.read_bit())? {
                    cc_alf_cb_aps_id = Some(hls_bit("SH", br.read_bits(3))? as u8);
                }
                if hls_bit("SH", br.read_bit())? {
                    cc_alf_cr_aps_id = Some(hls_bit("SH", br.read_bits(3))? as u8);
                }
            }
        }
    }

    let lmcs_used = if ph.lmcs_enabled && !ph_in_slice_header {
        hls_bit("SH", br.read_bit())?
    } else {
        ph.lmcs_enabled
    };
    let explicit_scaling_list_used = if ph.explicit_scaling_list_enabled && !ph_in_slice_header {
        hls_bit("SH", br.read_bit())?
    } else {
        ph.explicit_scaling_list_enabled
    };

    let mut rpl = [
        VvcRefPicListStruct::default(),
        VvcRefPicListStruct::default(),
    ];
    let is_idr = matches!(nal_type, NAL_IDR_W_RADL | NAL_IDR_N_LP);
    if !pps.rpl_info_in_ph && (!is_idr || sps.idr_rpl_present) {
        let forbidden_zero = !sps.weighted_pred && !sps.weighted_bipred;
        let mut rpl_sps_flag0 = false;
        let mut rpl_idx0 = 0usize;
        for l in 0..2usize {
            let rpl_sps_flag = if sps.rpl_lists[l].is_empty() {
                false
            } else if l == 0 || pps.rpl1_idx_present {
                hls_bit("SH", br.read_bit())?
            } else {
                rpl_sps_flag0
            };
            if l == 0 {
                rpl_sps_flag0 = rpl_sps_flag;
            }
            if rpl_sps_flag {
                let idx = if sps.rpl_lists[l].len() > 1 && (l == 0 || pps.rpl1_idx_present) {
                    let bits = ceil_log2(sps.rpl_lists[l].len() as u32);
                    hls_bit("SH", br.read_bits(bits))? as usize
                } else if sps.rpl_lists[l].len() == 1 {
                    0
                } else {
                    rpl_idx0
                };
                if l == 0 {
                    rpl_idx0 = idx;
                }
                rpl[l] = sps.rpl_lists[l]
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| VvcError::Bitstream("SH: rpl_idx out of bounds".to_string()))?;
            } else {
                rpl[l] = parse_ref_pic_list_struct(
                    &mut br,
                    sps.long_term_ref_pics,
                    u32::from(sps.poc_lsb_bits),
                    forbidden_zero,
                    false,
                )?;
            }
        }
    }

    let mut num_ref_idx_active = [
        if slice_type == VvcSliceType::I {
            0
        } else {
            pps.num_ref_idx_default_active[0].min(rpl[0].entries.len() as u8)
        },
        if slice_type == VvcSliceType::B {
            pps.num_ref_idx_default_active[1].min(rpl[1].entries.len() as u8)
        } else {
            0
        },
    ];
    if (slice_type != VvcSliceType::I && rpl[0].entries.len() > 1)
        || (slice_type == VvcSliceType::B && rpl[1].entries.len() > 1)
    {
        if hls_bit("SH", br.read_bit())? {
            if rpl[0].entries.len() > 1 {
                num_ref_idx_active[0] = (hls_bit("SH", br.read_ue())? as u8).saturating_add(1);
            }
            if slice_type == VvcSliceType::B && rpl[1].entries.len() > 1 {
                num_ref_idx_active[1] = (hls_bit("SH", br.read_ue())? as u8).saturating_add(1);
            }
        }
    }

    let mut cabac_init_flag = false;
    let mut collocated_from_l0 = true;
    let mut collocated_ref_idx = 0u8;
    if slice_type != VvcSliceType::I {
        if pps.cabac_init_present {
            cabac_init_flag = hls_bit("SH", br.read_bit())?;
        }
        if ph.temporal_mvp_enabled && !pps.rpl_info_in_ph {
            if slice_type == VvcSliceType::B {
                collocated_from_l0 = hls_bit("SH", br.read_bit())?;
            }
            let active_col = if collocated_from_l0 || slice_type == VvcSliceType::P {
                num_ref_idx_active[0]
            } else {
                num_ref_idx_active[1]
            };
            if active_col > 1 {
                collocated_ref_idx = hls_bit("SH", br.read_ue())? as u8;
            }
        }
    }

    let qp_delta = if !pps.qp_delta_info_in_ph {
        hls_bit("SH", br.read_se())?
    } else {
        0
    };
    let (mut cb_qp_offset, mut cr_qp_offset, mut joint_cbcr_qp_offset) = (0, 0, 0);
    if pps.slice_chroma_qp_offsets_present {
        cb_qp_offset = hls_bit("SH", br.read_se())?;
        cr_qp_offset = hls_bit("SH", br.read_se())?;
        if sps.joint_cbcr_enabled {
            joint_cbcr_qp_offset = hls_bit("SH", br.read_se())?;
        }
    }

    let (mut sao_luma_used, mut sao_chroma_used) = (false, false);
    if sps.sao_enabled && !pps.sao_info_in_ph {
        sao_luma_used = hls_bit("SH", br.read_bit())?;
        if sps.info.chroma_format_idc != 0 {
            sao_chroma_used = hls_bit("SH", br.read_bit())?;
        }
    }

    let mut deblocking_disabled = pps.deblocking_disabled;
    let mut luma_beta_offset_div2 = pps.luma_beta_offset_div2;
    let mut luma_tc_offset_div2 = pps.luma_tc_offset_div2;
    if pps.deblocking_control_present && pps.deblocking_override_enabled && !pps.dbf_info_in_ph {
        if hls_bit("SH", br.read_bit())? {
            if !pps.deblocking_disabled {
                deblocking_disabled = hls_bit("SH", br.read_bit())?;
            }
            if !deblocking_disabled {
                luma_beta_offset_div2 = hls_bit("SH", br.read_se())?;
                luma_tc_offset_div2 = hls_bit("SH", br.read_se())?;
                if pps.chroma_tool_offsets_present {
                    let _ = hls_bit("SH", br.read_se())?;
                    let _ = hls_bit("SH", br.read_se())?;
                    let _ = hls_bit("SH", br.read_se())?;
                    let _ = hls_bit("SH", br.read_se())?;
                }
            }
        }
    }

    let dep_quant_used = if sps.dep_quant_enabled {
        hls_bit("SH", br.read_bit())?
    } else {
        false
    };
    let sign_data_hiding_used = if sps.sign_data_hiding_enabled && !dep_quant_used {
        hls_bit("SH", br.read_bit())?
    } else {
        false
    };
    let ts_residual_coding_disabled =
        if sps.transform_skip_enabled && !dep_quant_used && !sign_data_hiding_used {
            hls_bit("SH", br.read_bit())?
        } else {
            false
        };

    let mut entry_point_offsets = Vec::new();
    if sps.entry_points_present {
        let tile_grid = pps.derive_tile_grid();
        let num_ep = if sps.wpp_enabled {
            tile_grid
                .row_heights_ctu
                .iter()
                .map(|&h| h as usize)
                .sum::<usize>()
                .saturating_sub(1)
        } else {
            tile_grid.num_tiles().saturating_sub(1)
        };
        if num_ep > 0 {
            let offset_len_minus1 = hls_bit("SH", br.read_ue())?;
            if offset_len_minus1 >= 32 {
                return Err(VvcError::Bitstream(
                    "SH: sh_entry_offset_len_minus1 >= 32".to_string(),
                ));
            }
            let bits = offset_len_minus1 + 1;
            entry_point_offsets.reserve(num_ep);
            for _ in 0..num_ep {
                let off_minus1 = hls_bit("SH", br.read_bits(bits))?;
                entry_point_offsets.push(off_minus1.saturating_add(1));
            }
        }
    }

    // byte_alignment(): a '1' bit followed by zero bits until aligned (H.266 §7.3.2.17)
    if !hls_bit("SH", br.read_bit())? {
        return Err(VvcError::Bitstream(
            "SH: byte_alignment bit_equal_to_one is 0".to_string(),
        ));
    }
    while !br.is_byte_aligned() {
        if hls_bit("SH", br.read_bit())? {
            return Err(VvcError::Bitstream(
                "SH: byte_alignment bit_equal_to_zero is 1".to_string(),
            ));
        }
    }
    let data_offset = br.bit_pos() / 8;

    Ok((
        VvcSliceHeader {
            ph_in_slice_header,
            embedded_ph,
            slice_type,
            no_output_of_prior_pics,
            alf_enabled,
            alf_aps_ids_luma,
            alf_cb_enabled,
            alf_cr_enabled,
            alf_aps_id_chroma,
            cc_alf_cb_aps_id,
            cc_alf_cr_aps_id,
            lmcs_used,
            explicit_scaling_list_used,
            rpl,
            num_ref_idx_active,
            cabac_init_flag,
            collocated_from_l0,
            collocated_ref_idx,
            qp_delta,
            cb_qp_offset,
            cr_qp_offset,
            joint_cbcr_qp_offset,
            sao_luma_used,
            sao_chroma_used,
            deblocking_disabled,
            luma_beta_offset_div2,
            luma_tc_offset_div2,
            dep_quant_used,
            sign_data_hiding_used,
            ts_residual_coding_disabled,
            entry_point_offsets,
        },
        data_offset,
    ))
}

/// Normative CTU and tile partition grid derived from a [`VvcPps`]
/// (ITU-T Rec. H.266 §6.5.1 Equations 15–21, VTM `PictureParameterSet.cpp:169-235`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcTileGrid {
    /// Picture width in CTUs (`PicWidthInCtbsY`).
    pub pic_width_in_ctus: u32,
    /// Picture height in CTUs (`PicHeightInCtbsY`).
    pub pic_height_in_ctus: u32,
    /// Tile column widths in CTUs (`colWidthVal[i]`, length `NumTileColumns`).
    pub col_widths_ctu: Vec<u32>,
    /// Tile row heights in CTUs (`rowHeightVal[j]`, length `NumTileRows`).
    pub row_heights_ctu: Vec<u32>,
    /// Tile column left boundaries in CTUs (`TileColBdVal[i]`, length `NumTileColumns + 1`).
    pub col_bd_ctu: Vec<u32>,
    /// Tile row top boundaries in CTUs (`TileRowBdVal[j]`, length `NumTileRows + 1`).
    pub row_bd_ctu: Vec<u32>,
    /// Mapping from CTU X coordinate (`0..PicWidthInCtbsY`) to tile column index (`ctbToTileColIdx`).
    pub ctu_to_tile_col: Vec<usize>,
    /// Mapping from CTU Y coordinate (`0..PicHeightInCtbsY`) to tile row index (`ctbToTileRowIdx`).
    pub ctu_to_tile_row: Vec<usize>,
}

impl VvcTileGrid {
    /// Total number of tiles in the picture (`NumTilesInPic = NumTileColumns * NumTileRows`).
    #[inline]
    pub fn num_tiles(&self) -> usize {
        self.col_widths_ctu.len() * self.row_heights_ctu.len()
    }

    /// Return the CTU bounding rectangle `(ctu_x0, ctu_y0, ctu_w, ctu_h)` of tile `tile_idx`.
    pub fn tile_ctu_rect(&self, tile_idx: usize) -> Option<(u32, u32, u32, u32)> {
        let num_cols = self.col_widths_ctu.len();
        if num_cols == 0 || tile_idx >= self.num_tiles() {
            return None;
        }
        let col = tile_idx % num_cols;
        let row = tile_idx / num_cols;
        Some((
            self.col_bd_ctu[col],
            self.row_bd_ctu[row],
            self.col_widths_ctu[col],
            self.row_heights_ctu[row],
        ))
    }

    /// Return the ordered list of `(ctu_x, ctu_y)` coordinates in tile-scan order across the picture.
    pub fn ctu_coords_in_tile_scan(&self) -> Vec<(u32, u32)> {
        let mut out =
            Vec::with_capacity((self.pic_width_in_ctus * self.pic_height_in_ctus) as usize);
        for tile_idx in 0..self.num_tiles() {
            if let Some((x0, y0, w, h)) = self.tile_ctu_rect(tile_idx) {
                for cy in y0..(y0 + h) {
                    for cx in x0..(x0 + w) {
                        out.push((cx, cy));
                    }
                }
            }
        }
        out
    }
}

impl VvcPps {
    /// Derive the normative [`VvcTileGrid`] for this PPS (ITU-T Rec. H.266 §6.5.1 Eq. 15–19,
    /// VTM `PPS::initTiles`, `PictureParameterSet.cpp:169-235`).
    pub fn derive_tile_grid(&self) -> VvcTileGrid {
        let ctu_size = 1u32 << self.ctu_log2.clamp(5, 7);
        let pic_w_ctu = self.width.max(1).div_ceil(ctu_size);
        let pic_h_ctu = self.height.max(1).div_ceil(ctu_size);
        if self.no_pic_partition {
            return VvcTileGrid {
                pic_width_in_ctus: pic_w_ctu,
                pic_height_in_ctus: pic_h_ctu,
                col_widths_ctu: vec![pic_w_ctu],
                row_heights_ctu: vec![pic_h_ctu],
                col_bd_ctu: vec![0, pic_w_ctu],
                row_bd_ctu: vec![0, pic_h_ctu],
                ctu_to_tile_col: vec![0; pic_w_ctu as usize],
                ctu_to_tile_row: vec![0; pic_h_ctu as usize],
            };
        }

        let derive_axis = |total_ctu: u32, exp: &[u32]| -> (Vec<u32>, Vec<u32>, Vec<usize>) {
            let mut widths = Vec::new();
            let mut remaining = total_ctu;
            let last_exp = exp.last().copied().unwrap_or(total_ctu).max(1);
            for &w in exp.iter().take(exp.len().saturating_sub(1)) {
                let step = w.max(1).min(remaining);
                if step > 0 {
                    widths.push(step);
                    remaining -= step;
                }
            }
            while remaining >= last_exp {
                widths.push(last_exp);
                remaining -= last_exp;
            }
            if remaining > 0 || widths.is_empty() {
                widths.push(remaining.max(1));
            }
            let mut bd = Vec::with_capacity(widths.len() + 1);
            bd.push(0);
            let mut map = Vec::with_capacity(total_ctu as usize);
            let mut acc = 0u32;
            for (idx, &w) in widths.iter().enumerate() {
                acc += w;
                bd.push(acc);
                for _ in 0..w {
                    if map.len() < total_ctu as usize {
                        map.push(idx);
                    }
                }
            }
            while map.len() < total_ctu as usize {
                map.push(widths.len() - 1);
            }
            (widths, bd, map)
        };

        let (col_widths_ctu, col_bd_ctu, ctu_to_tile_col) =
            derive_axis(pic_w_ctu, &self.exp_tile_col_widths);
        let (row_heights_ctu, row_bd_ctu, ctu_to_tile_row) =
            derive_axis(pic_h_ctu, &self.exp_tile_row_heights);

        VvcTileGrid {
            pic_width_in_ctus: pic_w_ctu,
            pic_height_in_ctus: pic_h_ctu,
            col_widths_ctu,
            row_heights_ctu,
            col_bd_ctu,
            row_bd_ctu,
            ctu_to_tile_col,
            ctu_to_tile_row,
        }
    }
}

/// Active Adaptation Parameter Set (APS) bank holding up to 8 `ALF_APS`, 8 `LMCS_APS`,
/// and 8 `SCALING_LIST_APS` entries (`H.266 §7.4.3.6`).
#[derive(Debug, Clone, Default)]
pub struct VvcActiveApsBank {
    /// Active `ALF_APS` entries indexed by `aps_id & 7`.
    pub alf: [Option<VvcAlfAps>; 8],
    /// Active `LMCS_APS` entries indexed by `aps_id & 7`.
    pub lmcs: [Option<VvcLmcsAps>; 8],
    /// Active `SCALING_LIST_APS` entries indexed by `aps_id & 7`.
    pub scaling_list: [Option<Vec<VvcScalingListEntry>>; 8],
}

impl VvcActiveApsBank {
    /// Insert or update a parsed [`VvcAps`] (`PREFIX_APS_NUT` or `SUFFIX_APS_NUT`).
    pub fn activate(&mut self, aps: VvcAps) {
        let idx = usize::from(aps.aps_id & 7);
        match aps.payload {
            VvcApsPayload::Alf(a) => self.alf[idx] = Some(a),
            VvcApsPayload::Lmcs(l) => self.lmcs[idx] = Some(l),
            VvcApsPayload::ScalingList(s) => self.scaling_list[idx] = Some(s),
        }
    }
}

/// Supplemental Enhancement Information (SEI) payload type for Buffering Period (`0`).
pub const SEI_PAYLOAD_BUFFERING_PERIOD: u32 = 0;
/// Supplemental Enhancement Information (SEI) payload type for Picture Timing (`1`).
pub const SEI_PAYLOAD_PICTURE_TIMING: u32 = 1;
/// Supplemental Enhancement Information (SEI) payload type for Film Grain Characteristics (`19`).
pub const SEI_PAYLOAD_FILM_GRAIN_CHARACTERISTICS: u32 = 19;
/// Supplemental Enhancement Information (SEI) payload type for Decoding Unit Information (`130`).
pub const SEI_PAYLOAD_DECODING_UNIT_INFO: u32 = 130;
/// Supplemental Enhancement Information (SEI) payload type for Decoded Picture Hash (`132`).
pub const SEI_PAYLOAD_DECODED_PICTURE_HASH: u32 = 132;
/// Supplemental Enhancement Information (SEI) payload type for Scalable Nesting (`133`).
pub const SEI_PAYLOAD_SCALABLE_NESTING: u32 = 133;

/// Parsed Buffering Period SEI message (`payloadType == 0`, ITU-T Rec. H.266 Annex D.2.2,
/// VTM `SEIread.cpp:1754-1909`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcBufferingPeriodSei {
    /// `bp_nal_hrd_params_present_flag`.
    pub nal_hrd_present: bool,
    /// `bp_vcl_hrd_params_present_flag`.
    pub vcl_hrd_present: bool,
    /// `bp_cpb_initial_removal_delay_length_minus1 + 1`.
    pub cpb_initial_removal_delay_length: u8,
    /// `bp_cpb_removal_delay_length_minus1 + 1`.
    pub cpb_removal_delay_length: u8,
    /// `bp_dpb_output_delay_length_minus1 + 1`.
    pub dpb_output_delay_length: u8,
    /// `bp_du_hrd_params_present_flag`.
    pub du_hrd_present: bool,
}

/// Parsed Picture Timing SEI message (`payloadType == 1`, ITU-T Rec. H.266 Annex D.2.3,
/// VTM `SEIread.cpp:1910-2090`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcPictureTimingSei {
    /// `pt_cpb_removal_delay_minus1 + 1` for the highest sublayer.
    pub cpb_removal_delay: u32,
    /// `pt_dpb_output_delay`.
    pub dpb_output_delay: u32,
}

/// Parsed Decoding Unit Information SEI message (`payloadType == 130`, ITU-T Rec. H.266
/// Annex D.2.4, VTM `SEIread.cpp:1695-1752`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcDecodingUnitInfoSei {
    /// `dui_decoding_unit_idx`.
    pub decoding_unit_idx: u32,
    /// `dui_dpb_output_du_delay_present_flag`.
    pub dpb_output_du_delay_present: bool,
}

/// Parsed Decoded Picture Hash SEI message (`payloadType == 132`, Rec. ITU-T H.274 §8.4).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcDecodedPictureHashSei {
    /// `dph_sei_hash_type` (`0` = MD5, `1` = CRC, `2` = Checksum).
    pub hash_type: u8,
    /// Per-component digests (`16` bytes for MD5, `2` for CRC, `4` for Checksum).
    pub digests: Vec<Vec<u8>>,
}

/// Parsed Scalable Nesting SEI message (`payloadType == 133`, ITU-T Rec. H.266 Annex D.2.5,
/// VTM `SEIread.cpp:1062-1165`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcScalableNestingSei {
    /// `sn_ols_flag`.
    pub ols_flag: bool,
    /// `sn_subpic_flag`.
    pub subpic_flag: bool,
    /// Nested [`VvcSeiMessage`] payloads unpacked from the byte-aligned nesting container.
    pub nested_messages: Vec<VvcSeiMessage>,
}

/// Parsed Film Grain Characteristics SEI message (`payloadType == 19`, ITU-T Rec. H.266
/// Annex D.3.10 / SMPTE RDD 5; VTM `SEIread.cpp:2587-2650`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VvcFilmGrainSei {
    /// `fg_characteristics_cancel_flag`.
    pub cancel_flag: bool,
    /// `fg_model_id` (`0` = frequency filtering, `1` = auto-regressive).
    pub model_id: u8,
    /// `fg_separate_colour_description_present_flag`.
    pub separate_colour_desc_present: bool,
    /// `fg_blending_mode_id` (`0` = additive, `1` = multiplicative).
    pub blending_mode_id: u8,
    /// `fg_log2_scale_factor` (`2..=7`).
    pub log2_scale_factor: u8,
    /// `fg_comp_model_present_flag[c]` for `Y, Cb, Cr`.
    pub comp_model_present: [bool; 3],
    /// Per-component intensity intervals `(lower_bound, upper_bound, first_model_value)`.
    pub intervals: [Vec<(u8, u8, i16)>; 3],
    /// `fg_characteristics_persistence_flag`.
    pub persistence_flag: bool,
}

/// Parsed SEI message from a `PREFIX_SEI_NUT` (`23`) or `SUFFIX_SEI_NUT` (`24`) RBSP
/// (`H.266 §7.3.5`, VTM `SEIread.cpp:150-210`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VvcSeiMessage {
    /// Buffering Period SEI (`payloadType == 0`).
    BufferingPeriod(VvcBufferingPeriodSei),
    /// Picture Timing SEI (`payloadType == 1`).
    PictureTiming(VvcPictureTimingSei),
    /// Film Grain Characteristics SEI (`payloadType == 19`).
    FilmGrain(VvcFilmGrainSei),
    /// Decoding Unit Information SEI (`payloadType == 130`).
    DecodingUnitInfo(VvcDecodingUnitInfoSei),
    /// Decoded Picture Hash SEI (`payloadType == 132`).
    DecodedPictureHash(VvcDecodedPictureHashSei),
    /// Scalable Nesting SEI (`payloadType == 133`).
    ScalableNesting(VvcScalableNestingSei),
    /// Any other SEI message preserved raw without desynchronising the decoder.
    Other {
        /// `payloadType`.
        payload_type: u32,
        /// Raw `payloadSize`-byte SEI payload.
        payload: Vec<u8>,
    },
}

/// Parse one or more [`VvcSeiMessage`] payloads from a `PREFIX_SEI_NUT` or `SUFFIX_SEI_NUT`
/// RBSP (`H.266 §7.3.2.16`, VTM `SEIread.cpp:150-210`).
pub fn parse_sei_rbsp(rbsp: &[u8]) -> Result<Vec<VvcSeiMessage>, VvcError> {
    let mut messages = Vec::new();
    let mut pos = 0usize;
    // Strip trailing rbsp_trailing_bits byte if it is a single 0x80 marker
    let end = if rbsp.last() == Some(&0x80) {
        rbsp.len().saturating_sub(1)
    } else {
        rbsp.len()
    };
    while pos < end {
        let mut payload_type = 0u32;
        while pos < end && rbsp[pos] == 0xFF {
            payload_type = payload_type.saturating_add(255);
            pos += 1;
        }
        if pos >= end {
            break;
        }
        payload_type = payload_type.saturating_add(u32::from(rbsp[pos]));
        pos += 1;

        let mut payload_size = 0usize;
        while pos < end && rbsp[pos] == 0xFF {
            payload_size = payload_size.saturating_add(255);
            pos += 1;
        }
        if pos >= end {
            return Err(VvcError::Bitstream(
                "SEI: truncated payload_size header".to_string(),
            ));
        }
        payload_size = payload_size.saturating_add(usize::from(rbsp[pos]));
        pos += 1;

        if pos + payload_size > rbsp.len() {
            return Err(VvcError::Bitstream(
                "SEI: payload_size exceeds RBSP length".to_string(),
            ));
        }
        let payload = &rbsp[pos..pos + payload_size];
        pos += payload_size;

        match payload_type {
            SEI_PAYLOAD_BUFFERING_PERIOD => {
                if let Ok(bp) = parse_buffering_period_sei_payload(payload) {
                    messages.push(VvcSeiMessage::BufferingPeriod(bp));
                    continue;
                }
            }
            SEI_PAYLOAD_PICTURE_TIMING => {
                if let Ok(pt) = parse_picture_timing_sei_payload(payload, 8, 8) {
                    messages.push(VvcSeiMessage::PictureTiming(pt));
                    continue;
                }
            }
            SEI_PAYLOAD_FILM_GRAIN_CHARACTERISTICS => {
                if let Ok(fg) = parse_film_grain_sei_payload(payload) {
                    messages.push(VvcSeiMessage::FilmGrain(fg));
                    continue;
                }
            }
            SEI_PAYLOAD_DECODING_UNIT_INFO => {
                if let Ok(dui) = parse_decoding_unit_info_sei_payload(payload) {
                    messages.push(VvcSeiMessage::DecodingUnitInfo(dui));
                    continue;
                }
            }
            SEI_PAYLOAD_DECODED_PICTURE_HASH => {
                if let Ok(dph) = parse_decoded_picture_hash_sei_payload(payload) {
                    messages.push(VvcSeiMessage::DecodedPictureHash(dph));
                    continue;
                }
            }
            SEI_PAYLOAD_SCALABLE_NESTING => {
                if let Ok(sn) = parse_scalable_nesting_sei_payload(payload) {
                    messages.push(VvcSeiMessage::ScalableNesting(sn));
                    continue;
                }
            }
            _ => {}
        }
        messages.push(VvcSeiMessage::Other {
            payload_type,
            payload: payload.to_vec(),
        });
    }
    Ok(messages)
}

/// Parse a `buffering_period` SEI payload (`payloadType == 0`, ITU-T Rec. H.266 Annex D.2.2,
/// VTM `SEIread.cpp:1754-1909`).
pub fn parse_buffering_period_sei_payload(
    payload: &[u8],
) -> Result<VvcBufferingPeriodSei, VvcError> {
    let mut br = BitReader::new(payload);
    let nal_hrd_present = hls_bit("SEI:BP", br.read_bit())?;
    let vcl_hrd_present = hls_bit("SEI:BP", br.read_bit())?;
    let cpb_initial_removal_delay_length = (hls_bit("SEI:BP", br.read_bits(5))? as u8) + 1;
    let cpb_removal_delay_length = (hls_bit("SEI:BP", br.read_bits(5))? as u8) + 1;
    let dpb_output_delay_length = (hls_bit("SEI:BP", br.read_bits(5))? as u8) + 1;
    let du_hrd_present = hls_bit("SEI:BP", br.read_bit())?;
    Ok(VvcBufferingPeriodSei {
        nal_hrd_present,
        vcl_hrd_present,
        cpb_initial_removal_delay_length,
        cpb_removal_delay_length,
        dpb_output_delay_length,
        du_hrd_present,
    })
}

/// Parse a `picture_timing` SEI payload (`payloadType == 1`, ITU-T Rec. H.266 Annex D.2.3,
/// VTM `SEIread.cpp:1910-2090`).
pub fn parse_picture_timing_sei_payload(
    payload: &[u8],
    cpb_removal_delay_length: u8,
    dpb_output_delay_length: u8,
) -> Result<VvcPictureTimingSei, VvcError> {
    let mut br = BitReader::new(payload);
    let c_len = u32::from(cpb_removal_delay_length.clamp(1, 32));
    let d_len = u32::from(dpb_output_delay_length.clamp(1, 32));
    let cpb_removal_delay = hls_bit("SEI:PT", br.read_bits(c_len))?.saturating_add(1);
    let dpb_output_delay = hls_bit("SEI:PT", br.read_bits(d_len))?;
    Ok(VvcPictureTimingSei {
        cpb_removal_delay,
        dpb_output_delay,
    })
}

/// Parse a `decoding_unit_info` SEI payload (`payloadType == 130`, ITU-T Rec. H.266
/// Annex D.2.4, VTM `SEIread.cpp:1695-1752`).
pub fn parse_decoding_unit_info_sei_payload(
    payload: &[u8],
) -> Result<VvcDecodingUnitInfoSei, VvcError> {
    let mut br = BitReader::new(payload);
    let decoding_unit_idx = hls_bit("SEI:DUI", br.read_ue())?;
    let dpb_output_du_delay_present = br.read_bit().unwrap_or(false);
    Ok(VvcDecodingUnitInfoSei {
        decoding_unit_idx,
        dpb_output_du_delay_present,
    })
}

/// Parse a `decoded_picture_hash` SEI payload (`payloadType == 132`, Rec. ITU-T H.274 §8.4).
pub fn parse_decoded_picture_hash_sei_payload(
    payload: &[u8],
) -> Result<VvcDecodedPictureHashSei, VvcError> {
    if payload.is_empty() {
        return Err(VvcError::Bitstream("SEI:DPH: empty payload".to_string()));
    }
    let hash_type = payload[0] >> 5;
    let single_comp = ((payload[0] >> 4) & 1) != 0;
    let num_comp = if single_comp { 1usize } else { 3usize };
    let digest_len = match hash_type {
        0 => 16usize,
        1 => 2usize,
        2 => 4usize,
        _ => {
            return Err(VvcError::Bitstream(
                "SEI:DPH: invalid hash_type".to_string(),
            ));
        }
    };
    let mut digests = Vec::with_capacity(num_comp);
    let mut off = 1usize;
    for _ in 0..num_comp {
        if off + digest_len <= payload.len() {
            digests.push(payload[off..off + digest_len].to_vec());
            off += digest_len;
        }
    }
    Ok(VvcDecodedPictureHashSei { hash_type, digests })
}

/// Parse a `scalable_nesting` SEI payload (`payloadType == 133`, ITU-T Rec. H.266 Annex D.2.5,
/// VTM `SEIread.cpp:1062-1165`).
pub fn parse_scalable_nesting_sei_payload(
    payload: &[u8],
) -> Result<VvcScalableNestingSei, VvcError> {
    let mut br = BitReader::new(payload);
    let ols_flag = hls_bit("SEI:SN", br.read_bit())?;
    let subpic_flag = hls_bit("SEI:SN", br.read_bit())?;
    if ols_flag {
        let num_olss = hls_bit("SEI:SN", br.read_ue())?.saturating_add(1);
        for _ in 0..num_olss {
            let _ = hls_bit("SEI:SN", br.read_ue())?;
        }
    } else {
        let all_layers = hls_bit("SEI:SN", br.read_bit())?;
        if !all_layers {
            let num_layers = hls_bit("SEI:SN", br.read_ue())?.saturating_add(1);
            for _ in 1..num_layers {
                let _ = hls_bit("SEI:SN", br.read_bits(6))?;
            }
        }
    }
    if subpic_flag {
        let num_subpics = hls_bit("SEI:SN", br.read_ue())?.saturating_add(1);
        let id_len = (hls_bit("SEI:SN", br.read_ue())? as usize).saturating_add(1);
        for _ in 0..num_subpics {
            let _ = hls_bit("SEI:SN", br.read_bits(id_len.min(16) as u32))?;
        }
    }
    let _num_seis = hls_bit("SEI:SN", br.read_ue())?.saturating_add(1);
    br.byte_align();
    let nested_off = br.bit_pos() / 8;
    let nested_messages = if nested_off < payload.len() {
        parse_sei_rbsp(&payload[nested_off..]).unwrap_or_default()
    } else {
        Vec::new()
    };
    Ok(VvcScalableNestingSei {
        ols_flag,
        subpic_flag,
        nested_messages,
    })
}

/// Parse a `film_grain_characteristics` SEI payload (`payloadType == 19`,
/// VTM `SEIread.cpp:2587-2650`).
pub fn parse_film_grain_sei_payload(payload: &[u8]) -> Result<VvcFilmGrainSei, VvcError> {
    let mut br = BitReader::new(payload);
    let cancel_flag = hls_bit("SEI:FG", br.read_bit())?;
    if cancel_flag {
        return Ok(VvcFilmGrainSei {
            cancel_flag: true,
            ..VvcFilmGrainSei::default()
        });
    }
    let model_id = hls_bit("SEI:FG", br.read_bits(2))? as u8;
    let separate_colour_desc_present = hls_bit("SEI:FG", br.read_bit())?;
    if separate_colour_desc_present {
        let _ = hls_bit("SEI:FG", br.read_bits(3))?; // bit_depth_luma_minus8
        let _ = hls_bit("SEI:FG", br.read_bits(3))?; // bit_depth_chroma_minus8
        let _ = hls_bit("SEI:FG", br.read_bit())?; // full_range
        let _ = hls_bit("SEI:FG", br.read_bits(8))?; // primaries
        let _ = hls_bit("SEI:FG", br.read_bits(8))?; // transfer
        let _ = hls_bit("SEI:FG", br.read_bits(8))?; // matrix
    }
    let blending_mode_id = hls_bit("SEI:FG", br.read_bits(2))? as u8;
    let log2_scale_factor = hls_bit("SEI:FG", br.read_bits(4))? as u8;
    let mut comp_model_present = [false; 3];
    for flag in &mut comp_model_present {
        *flag = hls_bit("SEI:FG", br.read_bit())?;
    }
    let mut intervals: [Vec<(u8, u8, i16)>; 3] = Default::default();
    for c in 0..3 {
        if comp_model_present[c] {
            let num_intervals = (hls_bit("SEI:FG", br.read_bits(8))? as usize) + 1;
            let num_model_values = (hls_bit("SEI:FG", br.read_bits(3))? as usize) + 1;
            for _ in 0..num_intervals {
                let lo = hls_bit("SEI:FG", br.read_bits(8))? as u8;
                let hi = hls_bit("SEI:FG", br.read_bits(8))? as u8;
                let mut first_val = 0i16;
                for m in 0..num_model_values {
                    let v = hls_bit("SEI:FG", br.read_se())? as i16;
                    if m == 0 {
                        first_val = v;
                    }
                }
                intervals[c].push((lo, hi, first_val));
            }
        }
    }
    let persistence_flag = hls_bit("SEI:FG", br.read_bit())?;
    Ok(VvcFilmGrainSei {
        cancel_flag,
        model_id,
        separate_colour_desc_present,
        blending_mode_id,
        log2_scale_factor,
        comp_model_present,
        intervals,
        persistence_flag,
    })
}

/// Build a complete `PREFIX_SEI_NUT` RBSP containing a [`VvcFilmGrainSei`] message
/// (`payloadType == 19`, `H.266 §7.3.2.16`).
pub fn build_film_grain_sei_rbsp(fg: &VvcFilmGrainSei) -> Vec<u8> {
    let mut pbw = BitWriter::with_capacity(32);
    pbw.write_bit(fg.cancel_flag);
    if !fg.cancel_flag {
        pbw.write_bits(u32::from(fg.model_id & 3), 2);
        pbw.write_bit(false); // fg_separate_colour_description_present_flag = 0
        pbw.write_bits(u32::from(fg.blending_mode_id & 3), 2);
        pbw.write_bits(u32::from(fg.log2_scale_factor & 0xF), 4);
        for c in 0..3 {
            pbw.write_bit(fg.comp_model_present[c] && !fg.intervals[c].is_empty());
        }
        for c in 0..3 {
            if fg.comp_model_present[c] && !fg.intervals[c].is_empty() {
                pbw.write_bits((fg.intervals[c].len() - 1) as u32, 8);
                pbw.write_bits(0, 3); // num_model_values_minus1 = 0
                for &(lo, hi, val) in &fg.intervals[c] {
                    pbw.write_bits(u32::from(lo), 8);
                    pbw.write_bits(u32::from(hi), 8);
                    pbw.write_se(i32::from(val));
                }
            }
        }
        pbw.write_bit(fg.persistence_flag);
    }
    while !pbw.is_byte_aligned() {
        pbw.write_bit(false);
    }
    let payload = pbw.into_bytes();

    let mut out = Vec::with_capacity(payload.len() + 4);
    out.push(SEI_PAYLOAD_FILM_GRAIN_CHARACTERISTICS as u8);
    let mut rem = payload.len();
    while rem >= 255 {
        out.push(0xFF);
        rem -= 255;
    }
    out.push(rem as u8);
    out.extend_from_slice(&payload);
    out.push(0x80); // rbsp_trailing_bits
    out
}

/// Build an SPS RBSP with explicit `chroma_format_idc` (`1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4`),
/// `bit_depth` (`8` or `10`), and screen-content / ACT / RPR flags (`H.266 §7.3.2.4`).
#[allow(clippy::too_many_arguments)]
pub fn build_sps_rbsp_format_bd(
    cfg: &VvcStreamConfig,
    tools: &VvcHlsTools,
    chroma_format_idc: u8,
    bit_depth: u8,
    act_enabled: bool,
    palette_enabled: bool,
    ibc_enabled: bool,
    rpr_enabled: bool,
) -> Vec<u8> {
    if chroma_format_idc == 1
        && bit_depth == 8
        && !act_enabled
        && !palette_enabled
        && !ibc_enabled
        && !rpr_enabled
    {
        return build_sps_rbsp_with_tools(cfg, tools);
    }
    let mut bw = BitWriter::with_capacity(64);
    let cf = chroma_format_idc.min(3);
    let bd = bit_depth.clamp(8, 12);
    bw.write_bits(0, 4); // sps_seq_parameter_set_id
    bw.write_bits(1, 4); // sps_video_parameter_set_id
    bw.write_bits(0, 3); // sps_max_sub_layers_minus1
    bw.write_bits(u32::from(cf), 2); // sps_chroma_format_idc
    bw.write_bits(u32::from(cfg.ctu_log2.clamp(5, 7) - 5), 2);
    bw.write_bit(true); // sps_ptl_dpb_hrd_params_present_flag
    // Profile tier level (Main 10 = 1, Main 12 = 2, Main 10 4:4:4 = 33, Main 12 4:4:4 = 34)
    let profile_idc: u32 = if bd > 10 {
        if cf == 0 || cf > 1 || act_enabled || palette_enabled || ibc_enabled {
            34
        } else {
            2
        }
    } else if cf == 0 || cf > 1 || act_enabled || palette_enabled || ibc_enabled {
        33
    } else {
        1
    };
    bw.write_bits(profile_idc, 7);
    bw.write_bit(false);
    bw.write_bits(u32::from(cfg.level_idc), 8);
    bw.write_bit(true);
    bw.write_bit(false);
    bw.write_bit(false); // gci_present_flag = 0 for unconstrained Main 10 / 4:4:4
    bw.align_zero();
    bw.write_bits(0, 8); // ptl_num_sub_profiles

    bw.write_bit(false); // sps_gdr_enabled_flag
    bw.write_bit(rpr_enabled); // sps_ref_pic_resampling_enabled_flag
    if rpr_enabled {
        bw.write_bit(true); // sps_res_change_in_clvs_allowed_flag
    }
    bw.write_ue(cfg.width);
    bw.write_ue(cfg.height);
    let (sub_w, sub_h) = match cf {
        1 => (2u32, 2u32),
        2 => (2u32, 1u32),
        _ => (1u32, 1u32),
    };
    let crop = cfg.crop_right > 0 || cfg.crop_bottom > 0;
    bw.write_bit(crop);
    if crop {
        bw.write_ue(0);
        bw.write_ue(cfg.crop_right / sub_w);
        bw.write_ue(0);
        bw.write_ue(cfg.crop_bottom / sub_h);
    }
    bw.write_bit(false); // sps_subpic_info_present_flag
    bw.write_ue(u32::from(bd - 8)); // sps_bitdepth_minus8
    bw.write_bit(false); // sps_entropy_coding_sync_enabled_flag
    bw.write_bit(false); // sps_entry_point_offsets_present_flag
    bw.write_bits(4, 4); // sps_log2_max_pic_order_cnt_lsb_minus4
    bw.write_bit(false);
    bw.write_bits(0, 2);
    bw.write_bits(0, 2);
    bw.write_ue(0);
    bw.write_ue(0);
    bw.write_ue(0);

    bw.write_ue(u32::from(cfg.min_cb_log2 - 2));
    bw.write_bit(false);
    bw.write_ue(0);
    bw.write_ue(0);
    if cf != 0 {
        bw.write_bit(false); // sps_qtbtt_dual_tree_intra_flag
    }
    bw.write_ue(0);
    bw.write_ue(0);
    if cfg.ctu_log2 > 5 {
        bw.write_bit(false);
    }
    bw.write_bit(tools.transform_skip);
    if tools.transform_skip {
        bw.write_ue(u32::from(cfg.ts_log2 - 2));
        bw.write_bit(tools.bdpcm);
    }
    bw.write_bit(tools.mts);
    if tools.mts {
        bw.write_bit(tools.mts);
        bw.write_bit(false);
    }
    bw.write_bit(tools.lfnst);
    if cf != 0 {
        bw.write_bit(tools.joint_cbcr);
        bw.write_bit(true);
        bw.write_se(0);
        bw.write_ue(0);
        bw.write_ue(0);
        bw.write_ue(0);
    }

    bw.write_bit(tools.sao);
    bw.write_bit(tools.alf);
    if tools.alf && cf != 0 {
        bw.write_bit(false); // sps_ccalf_enabled_flag
    }
    bw.write_bit(false); // sps_lmcs_enabled_flag
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(true);
    bw.write_ue(0);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_ue(5);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_ue(0);
    bw.write_bit(tools.isp);
    bw.write_bit(tools.mrl);
    bw.write_bit(tools.mip);
    if cf != 0 {
        bw.write_bit(tools.cclm);
    }
    if cf == 1 {
        bw.write_bit(false);
        bw.write_bit(false);
    }
    bw.write_bit(palette_enabled);
    if cf == 3 {
        bw.write_bit(act_enabled);
    }
    if tools.transform_skip || palette_enabled {
        bw.write_ue(0);
    }
    bw.write_bit(ibc_enabled);
    if ibc_enabled {
        bw.write_ue(0); // sps_six_minus_max_num_ibc_merge_cand = 0 -> 6 candidates
    }
    bw.write_bit(false); // sps_ladf_enabled_flag
    bw.write_bit(false); // sps_explicit_scaling_list_enabled_flag
    bw.write_bit(tools.dep_quant);
    bw.write_bit(tools.sign_data_hiding);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false);
    bw.write_bit(false); // sps_vui_parameters_present_flag
    bw.write_bit(false); // sps_extension_present_flag
    bw.rbsp_trailing_bits();
    bw.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_config() -> VvcStreamConfig {
        VvcStreamConfig {
            width: 128,
            height: 128,
            crop_right: 0,
            crop_bottom: 0,
            qp: 26,
            level_idc: 16,
            ctu_log2: 6,
            min_cb_log2: 3,
            ts_log2: 3,
        }
    }

    #[test]
    fn chroma_qp_mapping_table_matches_vtm() {
        // VTM `QpInValCb 17 27 32 44`, `QpOutValCb 17 29 34 41`
        // (`SameCQPTablesForAllChroma = 1`): the writer signals
        // `sps_delta_qp_diff_val` XORed with `sps_delta_qp_in_val_minus1`
        // (`VLCReader.cpp:1749-1751`).
        let table = VvcChromaQpTable {
            qp_table_start_minus26: 17 - 26,
            points: vec![(9, 5), (4, 1), (11, 12)],
        };
        assert_eq!(table.derive_table(0).expect("8-bit table")[17], 17);
        assert_eq!(table.derive_table(0).expect("8-bit table")[27], 29);
        assert_eq!(table.derive_table(0).expect("8-bit table")[32], 34);
        assert_eq!(table.derive_table(0).expect("8-bit table")[44], 41);
        assert_eq!(table.derive_table(0).expect("8-bit table")[45], 42);

        let mut sps = parse_full_sps_rbsp(&build_sps_rbsp_with_tools(
            &reference_config(),
            &VvcHlsTools::default(),
        ))
        .expect("parse the builder's own SPS");
        sps.same_qp_table_for_chroma = true;
        sps.chroma_qp_tables = vec![table];
        assert_eq!(sps.mapped_chroma_qp(1, 32, 0), 34);
        assert_eq!(sps.mapped_chroma_qp(2, 32, 0), 34);
        // The default mapping (`qPi > 26 ? qPi - 1 : qPi`) still applies when
        // no custom table is signalled.
        sps.chroma_qp_tables.clear();
        assert_eq!(sps.mapped_chroma_qp(1, 32, 0), 31);
        assert_eq!(sps.mapped_chroma_qp(1, 26, 0), 26);
    }

    #[test]
    fn ptl_matches_the_vtm_validated_reference() {
        let expected = [
            0x02u8, 0x10, 0xBA, 0x2F, 0xB6, 0x9B, 0xFF, 0xFF, 0xFF, 0x4F, 0xFF, 0xC0, 0x00, 0x00,
        ];
        let mut bw = BitWriter::with_capacity(16);
        write_profile_tier_level(&mut bw, 16, 6);
        assert_eq!(bw.into_bytes(), expected);
    }

    #[test]
    fn aud_ph_and_slice_header_bytes() {
        assert_eq!(build_aud_rbsp(), vec![0x88]);
        assert_eq!(build_ph_rbsp(0), vec![0x88, 0x04]);
        assert_eq!(build_slice_header_bits(0), vec![0x28]);
    }

    #[test]
    fn vps_and_sps_have_expected_prefix() {
        let cfg = reference_config();
        let vps = build_vps_rbsp(&cfg);
        assert_eq!(&vps[..3], &[0x10, 0x00, 0x00]);
        let sps = build_sps_rbsp(&cfg);
        assert_eq!(&sps[..3], &[0x01, 0x0B, 0x02]);
        let info = parse_sps_info(&sps).expect("parse own SPS");
        assert_eq!((info.width, info.height), (128, 128));
        assert_eq!(info.level_idc, 16);
        assert_eq!(info.profile_idc, 1);
    }

    #[test]
    fn conformance_window_crops_visible_size() {
        let mut cfg = reference_config();
        cfg.width = 128;
        cfg.height = 64;
        cfg.crop_right = 28;
        cfg.crop_bottom = 4;
        let sps = build_sps_rbsp(&cfg);
        let info = parse_sps_info(&sps).expect("parse cropped SPS");
        assert_eq!((info.width, info.height), (100, 60));
    }

    #[test]
    fn chroma_qp_table_matches_reference_derivation() {
        assert_eq!(derived_chroma_qp(0), 0);
        assert_eq!(derived_chroma_qp(26), 26);
        assert_eq!(derived_chroma_qp(27), 26);
        assert_eq!(derived_chroma_qp(63), 62);
        let table = VvcChromaQpTable::default()
            .derive_table_8bit()
            .expect("derive default chroma QP table");
        for qp in 0..=63 {
            assert_eq!(table[qp], derived_chroma_qp(qp as i32));
        }
    }

    #[test]
    fn truncated_sps_is_an_error() {
        let cfg = reference_config();
        let sps = build_sps_rbsp(&cfg);
        for len in 0..=12 {
            assert!(parse_sps_info(&sps[..len]).is_err(), "len {len}");
        }
        assert!(parse_sps_info(&sps).is_ok());
    }

    fn synthetic_sps(
        max_sub_layers_minus1: u32,
        ref_pic_resampling: bool,
        subpic_info: bool,
        crop: (u32, u32, u32, u32),
        chroma_format_idc: u32,
    ) -> Vec<u8> {
        let mut bw = BitWriter::with_capacity(32);
        bw.write_bits(0, 4);
        bw.write_bits(1, 4);
        bw.write_bits(max_sub_layers_minus1, 3);
        bw.write_bits(chroma_format_idc, 2);
        bw.write_bits(0, 2);
        bw.write_bit(true);
        write_profile_tier_level(&mut bw, 16, 5);
        bw.write_bit(false);
        bw.write_bit(ref_pic_resampling);
        if ref_pic_resampling {
            bw.write_bit(false);
        }
        bw.write_ue(128);
        bw.write_ue(128);
        let has_crop = crop != (0, 0, 0, 0);
        bw.write_bit(has_crop);
        if has_crop {
            bw.write_ue(crop.0);
            bw.write_ue(crop.1);
            bw.write_ue(crop.2);
            bw.write_ue(crop.3);
        }
        bw.write_bit(subpic_info);
        bw.write_ue(0);
        bw.rbsp_trailing_bits();
        bw.into_bytes()
    }

    #[test]
    fn unsupported_sps_structures_are_rejected_as_unsupported() {
        let err = parse_sps_info(&synthetic_sps(1, false, false, (0, 0, 0, 0), 1))
            .expect_err("multiple sub-layers");
        assert!(matches!(err, VvcError::Unsupported(_)), "{err}");

        assert!(
            parse_sps_info(&synthetic_sps(0, true, false, (0, 0, 0, 0), 1)).is_ok(),
            "same-resolution RPR streams are accepted"
        );

        let err = parse_sps_info(&synthetic_sps(0, false, true, (0, 0, 0, 0), 1))
            .expect_err("truncated sub-picture information");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");

        for crop in [(1, 0, 0, 0), (0, 0, 1, 0)] {
            let err = parse_sps_info(&synthetic_sps(0, false, false, crop, 1))
                .expect_err("crop not anchored at the origin");
            assert!(matches!(err, VvcError::Unsupported(_)), "{err}");
        }

        assert!(parse_sps_info(&synthetic_sps(0, false, false, (0, 0, 0, 0), 1)).is_ok());
    }

    #[test]
    fn sps_without_profile_tier_level_is_unsupported() {
        let mut bw = BitWriter::with_capacity(8);
        bw.write_bits(0, 4);
        bw.write_bits(1, 4);
        bw.write_bits(0, 3);
        bw.write_bits(1, 2);
        bw.write_bits(0, 2);
        bw.write_bit(false);
        bw.rbsp_trailing_bits();
        let err = parse_sps_info(&bw.into_bytes()).expect_err("PTL absent");
        assert!(matches!(err, VvcError::Unsupported(_)), "{err}");
    }

    #[test]
    fn sps_vertical_crop_uses_sub_height_c() {
        let sps = synthetic_sps(0, false, false, (0, 8, 0, 8), 2);
        let info = parse_sps_info(&sps).expect("parse 4:2:2 cropped SPS");
        assert_eq!(info.chroma_format_idc, 2);
        assert_eq!((info.coded_width, info.coded_height), (128, 128));
        assert_eq!((info.width, info.height), (112, 120));
    }

    #[test]
    fn ctu_constraint_idc_is_guarded_for_out_of_range_log2() {
        assert_eq!(ctu_size_constraint_idc(5), 3);
        assert_eq!(ctu_size_constraint_idc(6), 2);
        assert_eq!(ctu_size_constraint_idc(7), 1);
        for ctu_log2 in 0..=9u8 {
            let mut bw = BitWriter::with_capacity(16);
            write_profile_tier_level(&mut bw, 16, ctu_log2);
            assert!(!bw.into_bytes().is_empty(), "ctu_log2 {ctu_log2}");
        }
    }

    #[test]
    fn oversized_bit_depth_and_crop_are_rejected() {
        let mut bw = BitWriter::with_capacity(16);
        bw.write_bits(0, 4);
        bw.write_bits(1, 4);
        bw.write_bits(0, 3);
        bw.write_bits(1, 2);
        bw.write_bits(0, 2);
        bw.write_bit(true);
        bw.write_bits(1, 7);
        bw.write_bit(false);
        bw.write_bits(16, 8);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(false);
        while !bw.is_byte_aligned() {
            bw.write_bit(false);
        }
        bw.write_bits(0, 8);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_ue(128);
        bw.write_ue(128);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_ue(250);
        bw.align_zero();
        let bytes = bw.into_bytes();
        let err = parse_sps_info(&bytes).expect_err("bit depth must be rejected");
        assert!(matches!(err, VvcError::Bitstream(_)));
    }

    #[test]
    fn opi_dci_and_vps_round_trip() {
        let opi = VvcOpi {
            ols_idx: Some(2),
            htid_plus1: Some(1),
        };
        let opi_rbsp = build_opi_rbsp(&opi);
        assert_eq!(parse_opi_rbsp(&opi_rbsp).expect("parse OPI"), opi);

        let dci = VvcDci {
            ptls: vec![(1, false, 51)],
        };
        let dci_rbsp = build_dci_rbsp(&dci, 5).expect("build DCI");
        assert_eq!(parse_dci_rbsp(&dci_rbsp).expect("parse DCI"), dci);

        let cfg = reference_config();
        let vps = parse_vps_rbsp(&build_vps_rbsp(&cfg)).expect("parse VPS");
        assert_eq!(vps.vps_id, 1);
        assert_eq!(vps.level_idc, 16);
    }

    #[test]
    fn full_sps_pps_aps_ph_and_sh_round_trip() {
        let cfg = reference_config();
        let sps_rbsp = build_sps_rbsp(&cfg);
        let full_sps = parse_full_sps_rbsp(&sps_rbsp).expect("parse full SPS");
        assert!(full_sps.transform_skip_enabled);
        assert!(full_sps.mts_enabled);
        assert!(full_sps.lfnst_enabled);
        assert_eq!(full_sps.poc_lsb_bits, 8);

        let pps = VvcPps {
            width: 128,
            height: 128,
            no_pic_partition: false,
            ctu_log2: 6,
            exp_tile_col_widths: vec![1, 1],
            exp_tile_row_heights: vec![2],
            deblocking_disabled: false,
            luma_beta_offset_div2: -1,
            luma_tc_offset_div2: 1,
            ..VvcPps::default()
        };
        let pps_rbsp = build_full_pps_rbsp(&pps);
        let parsed_pps = parse_full_pps_rbsp(&pps_rbsp).expect("parse full PPS");
        assert_eq!(parsed_pps, pps);

        let lmcs = VvcAps {
            aps_id: 1,
            chroma_present: true,
            payload: VvcApsPayload::Lmcs(VvcLmcsAps {
                min_bin_idx: 1,
                max_bin_idx: 14,
                bin_cw_delta: [0, -2, -1, 0, 1, 2, 2, 1, 0, -1, -1, 0, 1, 0, -2, 0],
                chroma_res_scaling_offset: -3,
            }),
        };
        let lmcs_rbsp = build_aps_rbsp(&lmcs).expect("build LMCS APS");
        assert_eq!(parse_aps_rbsp(&lmcs_rbsp).expect("parse LMCS APS"), lmcs);

        let alf = VvcAps {
            aps_id: 2,
            chroma_present: true,
            payload: VvcApsPayload::Alf(VvcAlfAps {
                luma_clip: true,
                luma_filters: vec![[4, -8, 12, -3, 0, 16, -20, 7, 2, -1, 32, -5]],
                luma_clip_idx: vec![[0, 1, 2, 3, 0, 1, 2, 1, 0, 0, 1, 2]],
                filter_coeff_delta_idx: [0; 25],
                chroma_clip: false,
                chroma_filters: vec![[6, -4, 10, -2, 18, -7]],
                chroma_clip_idx: vec![[0; 6]],
                cc_cb_filters: vec![[0, 4, -8, 16, -2, 0, 1]],
                cc_cr_filters: vec![],
            }),
        };
        let alf_rbsp = build_aps_rbsp(&alf).expect("build ALF APS");
        assert_eq!(parse_aps_rbsp(&alf_rbsp).expect("parse ALF APS"), alf);
    }
}
