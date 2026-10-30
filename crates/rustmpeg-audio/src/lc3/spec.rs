// SPDX-License-Identifier: Apache-2.0
//! LC3 spectrum decoding: side information, arithmetic spectrum, residual
//! bits, noise filling, and gain (Bluetooth LC3 v1.0.1 §3.4.2 and §3.4.3–3.4.5).

use super::Lc3Error;
use super::bits::BitReader;
use super::tables::{AC_SPEC_CUMFREQ, AC_SPEC_FREQ, AC_SPEC_LOOKUP, I_7M5, I_10M};

/// Band-limit table for a frame duration and sample-rate index.
pub(crate) fn band_limits(dt: usize, sr: usize) -> &'static [i32] {
    match dt {
        0 => I_7M5[sr],
        1 => I_10M[sr],
        _ => &[],
    }
}

/// State of the spectral decoder for one channel.
#[derive(Debug, Clone)]
pub(crate) struct SpectrumDecoder {
    dt: usize,
    sr: usize,
    lastnz: usize,
    lsb_mode: bool,
    g_idx: i32,
}

impl SpectrumDecoder {
    /// Creates a decoder for one channel.
    pub(crate) fn new(dt: usize, sr: usize) -> Self {
        Self {
            dt,
            sr,
            lastnz: 0,
            lsb_mode: false,
            g_idx: 0,
        }
    }

    /// Reads the spectrum side information.
    pub(crate) fn load(&mut self, bits: &mut BitReader<'_>) -> Result<(), Lc3Error> {
        let ne = *band_limits(self.dt, self.sr)
            .last()
            .ok_or(Lc3Error::InvalidConfig("missing band table"))? as usize;
        let nbits_lastnz = 32 - (ne as u32 / 2 - 1).leading_zeros();
        self.lastnz = ((bits.read_uint(nbits_lastnz)? + 1) << 1) as usize;
        self.lsb_mode = bits.read_bit()?;
        self.g_idx = bits.read_uint(8)? as i32;
        if self.lastnz > ne {
            return Err(Lc3Error::Bitstream(
                "last non-zero coefficient out of range",
            ));
        }
        Ok(())
    }

    /// Decodes the spectrum and applies gain, residual, and noise filling.
    pub(crate) fn decode(
        &mut self,
        bits: &mut BitReader<'_>,
        bw: usize,
        nbytes: usize,
    ) -> Result<Vec<f32>, Lc3Error> {
        let i = band_limits(self.dt, self.sr);
        let ne = *i
            .last()
            .ok_or(Lc3Error::InvalidConfig("missing band table"))? as usize;
        let nq = self.lastnz;
        let f_nf = bits.read_uint(3)? as i32;

        let mut x = vec![0f32; ne];
        let rate = if nbytes > 20 * (1 + self.sr) { 512 } else { 0 };
        let mut seed = 0u32;
        let mut c = 0usize;

        let mut n = 0usize;
        while n < nq {
            let mut t = c + rate;
            if n > ne / 2 {
                t += 256;
            }
            let mut u = 0u32;
            let mut v = 0u32;
            let mut lev = 0usize;
            let sym = loop {
                if lev >= 14 {
                    return Err(Lc3Error::Bitstream("spectral level out of range"));
                }
                let s = t + lev.min(3) * 1024;
                let pki = usize::from(AC_SPEC_LOOKUP[s]);
                let symbol = bits.ac_decode(&AC_SPEC_CUMFREQ[pki], &AC_SPEC_FREQ[pki])?;
                if symbol < 16 {
                    break symbol;
                }
                if !self.lsb_mode || lev > 0 {
                    u |= u32::from(bits.read_bit()?) << lev;
                    v |= u32::from(bits.read_bit()?) << lev;
                }
                lev += 1;
            };
            let a = (sym & 3) as u32;
            let b = (sym >> 2) as u32;
            u |= a << lev;
            v |= b << lev;

            let u_signed = if u != 0 && bits.read_bit()? {
                -(u as f32)
            } else {
                u as f32
            };
            let v_signed = if v != 0 && bits.read_bit()? {
                -(v as f32)
            } else {
                v as f32
            };
            x[n] = u_signed;
            x[n + 1] = v_signed;

            seed = (seed
                .wrapping_add((u & 0x7fff).wrapping_mul(n as u32))
                .wrapping_add((v & 0x7fff).wrapping_mul(n as u32 + 1)))
                & 0xffff;

            let lev3 = lev.min(3);
            let state = if lev3 <= 1 {
                1 + (a + b) * (lev3 as u32 + 1)
            } else {
                12 + lev3 as u32
            };
            c = (c & 15) * 16 + state as usize;
            n += 2;
        }

        let nbits_left = bits.bits_left().max(0);
        if self.lsb_mode {
            self.get_lsb(bits, nbits_left, &mut x, nq, &mut seed)?;
        } else {
            self.get_residual(bits, nbits_left, &mut x, nq)?;
        }

        let gain = unquantize_gain(self.g_idx - gain_offset(self.sr, nbytes));
        for value in x.iter_mut().take(nq) {
            *value *= gain;
        }
        for value in x.iter_mut().skip(nq) {
            *value = 0.0;
        }

        let zero_frame = nq <= 2 && x[0] == 0.0 && x[1] == 0.0 && self.g_idx <= 0 && f_nf >= 7;
        if !zero_frame {
            self.fill_noise(bw, f_nf, seed, gain, &mut x, nq);
        }
        Ok(x)
    }

    /// Reads the residual bits of quantized coefficients (§3.4.2.5).
    fn get_residual(
        &self,
        bits: &mut BitReader<'_>,
        mut nbits: i32,
        x: &mut [f32],
        nq: usize,
    ) -> Result<(), Lc3Error> {
        let mut index = 0;
        while index < nq && nbits > 0 {
            if x[index] == 0.0 {
                index += 1;
                continue;
            }
            if !bits.read_bit()? {
                x[index] -= if x[index] < 0.0 {
                    5.0 / 16.0
                } else {
                    3.0 / 16.0
                };
            } else {
                x[index] += if x[index] > 0.0 {
                    5.0 / 16.0
                } else {
                    3.0 / 16.0
                };
            }
            nbits -= 1;
            index += 1;
        }
        Ok(())
    }

    /// Reads the least significant bits of quantized coefficients (§3.4.2.5).
    fn get_lsb(
        &self,
        bits: &mut BitReader<'_>,
        mut nbits: i32,
        x: &mut [f32],
        nq: usize,
        seed: &mut u32,
    ) -> Result<(), Lc3Error> {
        let mut index = 0;
        while index < nq {
            if nbits <= 0 {
                break;
            }
            let a = x[index].abs();
            let b = x[index + 1].abs();
            if a.max(b) < 4.0 {
                index += 2;
                continue;
            }
            if nbits > 0 {
                nbits -= 1;
                if bits.read_bit()? {
                    if a != 0.0 {
                        x[index] += if x[index] < 0.0 { -1.0 } else { 1.0 };
                        *seed = (*seed).wrapping_add(index as u32) & 0xffff;
                    } else if nbits > 0 {
                        nbits -= 1;
                        x[index] = if bits.read_bit()? { -1.0 } else { 1.0 };
                        *seed = (*seed).wrapping_add(index as u32) & 0xffff;
                    }
                }
            }
            if nbits > 0 {
                nbits -= 1;
                if bits.read_bit()? {
                    if b != 0.0 {
                        x[index + 1] += if x[index + 1] < 0.0 { -1.0 } else { 1.0 };
                        *seed = (*seed).wrapping_add(index as u32 + 1) & 0xffff;
                    } else if nbits > 0 {
                        nbits -= 1;
                        x[index + 1] = if bits.read_bit()? { -1.0 } else { 1.0 };
                        *seed = (*seed).wrapping_add(index as u32 + 1) & 0xffff;
                    }
                }
            }
            index += 2;
        }
        Ok(())
    }

    /// Fills zeroed coefficients with noise (§3.4.5).
    fn fill_noise(&self, bw: usize, nf: i32, mut seed: u32, gain: f32, x: &mut [f32], nq: usize) {
        let bw_stop = *band_limits(self.dt, bw.min(4)).last().unwrap_or(&0) as usize;
        let w = 1 + usize::from(true) + usize::from(self.dt == 1);
        let level = gain * (8 - nf) as f32 / 16.0;

        let mut z = 0usize;
        let mut k = 6 * (3 + self.dt) - w;
        let limit = nq.min(bw_stop);
        while k < limit {
            z = if x[k] != 0.0 { 0 } else { z + 1 };
            if z > 2 * w {
                seed = (13849 + seed.wrapping_mul(31821)) & 0xffff;
                x[k - w] = if seed & 0x8000 != 0 { -level } else { level };
            }
            k += 1;
        }
        while k < bw_stop + w {
            z += 1;
            if z > 2 * w {
                seed = (13849 + seed.wrapping_mul(31821)) & 0xffff;
                x[k - w] = if seed & 0x8000 != 0 { -level } else { level };
            }
            k += 1;
        }
    }
}

/// Gain offset derived from the frame size (§3.4.3).
fn gain_offset(sr: usize, nbytes: usize) -> i32 {
    let g_off = (nbytes as i32 * 8) / (10 * (1 + sr as i32));
    105 + 5 * (1 + sr as i32) + g_off.min(115)
}

/// Unquantization gain `10^(g_int/28)` (§3.4.4).
fn unquantize_gain(mut g_int: i32) -> f32 {
    const TABLE: [f32; 28] = [
        1f32,
        1.085711f32,
        1.178769f32,
        1.279802f32,
        1.389495f32,
        1.508591f32,
        1.637894f32,
        1.778279f32,
        1.930698f32,
        2.09618f32,
        2.275846f32,
        2.470911f32,
        2.682696f32,
        2.912633f32,
        3.162278f32,
        3.43332f32,
        3.727594f32,
        4.04709f32,
        4.39397f32,
        4.770583f32,
        5.179475f32,
        5.623413f32,
        6.105402f32,
        6.628703f32,
        7.196856f32,
        7.813707f32,
        8.483429f32,
        9.210553f32,
    ];
    let mut g = 1.0f32;
    while g_int < 0 {
        g_int += 28;
        g *= 0.1;
    }
    while g_int >= 28 {
        g_int -= 28;
        g *= 10.0;
    }
    g * TABLE[g_int as usize]
}
