// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Small SILK descriptor structures.

/// One normalized-LSF codebook and its entropy tables.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct NLSFCodebook {
    /// Number of stage-1 vectors.
    pub n_vectors: i16,
    /// LPC order.
    pub order: i16,
    /// Quantization step size in Q16.
    pub quant_step_size_q16: i32,
    /// Inverse quantization step size in Q6.
    pub inv_quant_step_size_q6: i16,
    /// Stage-1 codebook, Q8.
    pub cb1_nlsf_q8: &'static [u8],
    /// Stage-1 weights, Q9.
    pub cb1_wght_q9: &'static [i16],
    /// Stage-1 inverse CDF.
    pub cb1_icdf: &'static [u8],
    /// Prediction coefficients, Q8.
    pub pred_q8: &'static [u8],
    /// Stage-2 selector.
    pub ec_sel: &'static [u8],
    /// Stage-2 inverse CDF.
    pub ec_icdf: &'static [u8],
    /// Stage-2 bit rates, Q5.
    pub ec_rates_q5: &'static [u8],
    /// Minimum NLSF deltas, Q15.
    pub delta_min_q15: &'static [i16],
}
