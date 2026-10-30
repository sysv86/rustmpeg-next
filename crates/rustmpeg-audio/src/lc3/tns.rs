// SPDX-License-Identifier: Apache-2.0
//! LC3 Temporal Noise Shaping synthesis (Bluetooth LC3 v1.0.1 §3.4.6).

use super::Lc3Error;
use super::bits::BitReader;
use super::tables::{
    TNS_COEF_CUMFREQ, TNS_COEF_FREQ, TNS_FREQ_LIM, TNS_ORDER_CUMFREQ, TNS_ORDER_FREQ,
};

/// Quantized reflection coefficients, `sin(i * pi / 17)` rounded to Q15.
static RC_TABLE: [f32; 9] = [
    0.0, 0.183_746, 0.361_237, 0.526_428, 0.673_706, 0.797_997, 0.895_172, 0.961_823, 0.995_728,
];

/// TNS synthesis state for one channel.
#[derive(Debug, Clone)]
pub(crate) struct TnsSynthesis {
    dt: usize,
    nfilters: usize,
    lpc_weighting: bool,
    rc_order: [usize; 2],
    rc: [[i32; 8]; 2],
}

impl TnsSynthesis {
    pub(crate) fn new(dt: usize) -> Self {
        Self {
            dt,
            nfilters: 0,
            lpc_weighting: false,
            rc_order: [0; 2],
            rc: [[8; 8]; 2],
        }
    }

    /// Reads the TNS side information.
    pub(crate) fn load(
        &mut self,
        bits: &mut BitReader<'_>,
        bw: usize,
        nbytes: usize,
    ) -> Result<(), Lc3Error> {
        self.nfilters = 1 + usize::from(bw >= 3);
        self.lpc_weighting = (nbytes as f32) < 120.0 * (3 + self.dt) as f32 / 8.0;
        self.rc_order = [0; 2];
        self.rc = [[8; 8]; 2];
        for f in 0..self.nfilters {
            if !bits.read_bit()? {
                continue;
            }
            let weighting = usize::from(self.lpc_weighting);
            let order =
                1 + bits.ac_decode(&TNS_ORDER_CUMFREQ[weighting], &TNS_ORDER_FREQ[weighting])?;
            if order > 8 {
                return Err(Lc3Error::Bitstream("invalid TNS order"));
            }
            self.rc_order[f] = order;
            for k in 0..order {
                self.rc[f][k] = bits.ac_decode(&TNS_COEF_CUMFREQ[k], &TNS_COEF_FREQ[k])? as i32;
            }
        }
        Ok(())
    }

    /// Installs quantized coefficients for the round-trip test.
    #[cfg(test)]
    pub(crate) fn set_for_test(
        &mut self,
        nfilters: usize,
        rc_order: [usize; 2],
        rc: [[i32; 8]; 2],
    ) {
        self.nfilters = nfilters;
        self.lpc_weighting = false;
        self.rc_order = rc_order;
        self.rc = rc;
    }

    /// Applies the TNS synthesis filter to `x`.
    pub(crate) fn run(&self, x: &mut [f32], bw: usize) -> Result<(), Lc3Error> {
        let limits = TNS_FREQ_LIM[self.dt][bw];
        // The filter state is shared by the sub-band filters and only the
        // taps beyond the current order are cleared after each filter.
        let mut state = [0f32; 8];
        for f in 0..self.nfilters {
            let order = self.rc_order[f];
            if order == 0 {
                continue;
            }
            let mut rc = [0f32; 8];
            for k in 0..order {
                let quantized = self.rc[f][k] - 8;
                let magnitude = RC_TABLE[quantized.unsigned_abs() as usize];
                rc[k] = if quantized < 0 { -magnitude } else { magnitude };
            }
            let start = *limits
                .get(f)
                .ok_or(Lc3Error::Bitstream("TNS filter out of range"))?
                as usize;
            let end = *limits
                .get(f + 1)
                .ok_or(Lc3Error::Bitstream("TNS filter out of range"))?
                as usize;
            if start > end || end > x.len() {
                return Err(Lc3Error::Bitstream("TNS band out of range"));
            }
            for sample in &mut x[start..end] {
                let mut value = *sample;
                value -= state[7] * rc[7];
                for k in (0..7).rev() {
                    value -= state[k] * rc[k];
                    state[k + 1] = state[k] + rc[k] * value;
                }
                state[0] = value;
                *sample = value;
            }
            state[order..].fill(0.0);
        }
        Ok(())
    }
}
