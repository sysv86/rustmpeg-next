// SPDX-License-Identifier: Apache-2.0
//! Exact rational time and timestamp primitives.
//!
//! Timestamps are integer tick counts plus a [`Timebase`] (seconds per tick).
//! All conversions use checked integer arithmetic on `i128`/`u128` and never
//! round through floating point unless the caller explicitly asks for seconds.

use crate::error::{CoreError, Result};

/// Positive rational number, used for timebases, frame rates, and durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational {
    numerator: u32,
    denominator: u32,
}

impl Rational {
    /// Creates a positive rational.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] when either part is zero.
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        if numerator == 0 {
            return Err(CoreError::InvalidConfig(
                "rational numerator must be non-zero".into(),
            ));
        }
        if denominator == 0 {
            return Err(CoreError::InvalidConfig(
                "rational denominator must be non-zero".into(),
            ));
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub(crate) const fn from_parts(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Numerator in lowest terms as stored by the caller.
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// Denominator in lowest terms as stored by the caller.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    /// Returns the same value reduced by the greatest common divisor.
    pub fn reduced(self) -> Self {
        let divisor = gcd(self.numerator, self.denominator);
        Self {
            numerator: self.numerator / divisor,
            denominator: self.denominator / divisor,
        }
    }

    /// Returns the reciprocal ratio.
    ///
    /// # Errors
    ///
    /// The numerator is always non-zero, so this cannot fail for a value built
    /// by [`Rational::new`]; the result is still returned as a `Result` for
    /// consistency with future unchecked constructors.
    pub fn invert(self) -> Result<Self> {
        Self::new(self.denominator, self.numerator)
    }

    /// Floating-point approximation, for display and heuristics only.
    pub fn to_f64(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

/// Seconds per tick.
pub type Timebase = Rational;

/// A signed timestamp in a specific timebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MediaTime {
    ticks: i64,
    timebase: Timebase,
}

impl MediaTime {
    /// Creates a timestamp from a tick count and a timebase.
    pub const fn new(ticks: i64, timebase: Timebase) -> Self {
        Self { ticks, timebase }
    }

    /// Timestamp zero in `timebase`.
    pub const fn zero(timebase: Timebase) -> Self {
        Self::new(0, timebase)
    }

    /// Tick count.
    pub const fn ticks(self) -> i64 {
        self.ticks
    }

    /// Timebase of this timestamp.
    pub const fn timebase(self) -> Timebase {
        self.timebase
    }

    /// Converts to another timebase, rounding to the nearest tick.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Overflow`] when the intermediate or final value does
    /// not fit.
    pub fn rescale(self, target: Timebase) -> Result<Self> {
        if target == self.timebase {
            return Ok(self);
        }
        let numerator = i128::from(self.ticks)
            .checked_mul(i128::from(target.denominator))
            .and_then(|value| value.checked_mul(i128::from(self.timebase.numerator)))
            .ok_or(CoreError::Overflow("time rescale numerator"))?;
        let denominator = i128::from(self.timebase.denominator)
            .checked_mul(i128::from(target.numerator))
            .ok_or(CoreError::Overflow("time rescale denominator"))?;
        let ticks =
            div_round(numerator, denominator).ok_or(CoreError::Overflow("time rescale result"))?;
        let ticks = i64::try_from(ticks).map_err(|_| CoreError::Overflow("time rescale result"))?;
        Ok(Self::new(ticks, target))
    }

    /// Adds two timestamps, rescaling `other` to this timestamp's timebase.
    ///
    /// # Errors
    ///
    /// Propagates rescale and signed overflow failures.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        let other = other.rescale(self.timebase)?;
        let ticks = self
            .ticks
            .checked_add(other.ticks)
            .ok_or(CoreError::Overflow("timestamp addition"))?;
        Ok(Self::new(ticks, self.timebase))
    }

    /// Subtracts `other` from this timestamp, rescaling `other` to this
    /// timestamp's timebase.
    ///
    /// # Errors
    ///
    /// Propagates rescale and signed overflow failures.
    pub fn checked_sub(self, other: Self) -> Result<Self> {
        let other = other.rescale(self.timebase)?;
        let ticks = self
            .ticks
            .checked_sub(other.ticks)
            .ok_or(CoreError::Overflow("timestamp subtraction"))?;
        Ok(Self::new(ticks, self.timebase))
    }

    /// Timestamp in seconds, as a floating-point approximation.
    pub fn to_seconds(self) -> f64 {
        self.ticks as f64 * self.timebase.to_f64()
    }

    /// Builds a timestamp from seconds, rounding to the nearest tick.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidData`] for NaN or infinite input and
    /// [`CoreError::Overflow`] when the result does not fit in `i64`.
    pub fn from_seconds(seconds: f64, timebase: Timebase) -> Result<Self> {
        if !seconds.is_finite() {
            return Err(CoreError::InvalidData(
                "timestamp seconds must be finite".into(),
            ));
        }
        let ticks_per_second = f64::from(timebase.denominator) / f64::from(timebase.numerator);
        let value = seconds * ticks_per_second;
        // `f64::round` is not available in core; this is the same
        // round-half-away-from-zero rule using only arithmetic and a cast.
        let rounded = if value >= 0.0 {
            value + 0.5
        } else {
            value - 0.5
        };
        if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
            return Err(CoreError::Overflow("timestamp from seconds"));
        }
        Ok(Self::new(rounded as i64, timebase))
    }
}

/// An unsigned duration in a specific timebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MediaDuration {
    ticks: u64,
    timebase: Timebase,
}

impl MediaDuration {
    /// Creates a duration from a tick count and a timebase.
    pub const fn new(ticks: u64, timebase: Timebase) -> Self {
        Self { ticks, timebase }
    }

    /// Zero duration in `timebase`.
    pub const fn zero(timebase: Timebase) -> Self {
        Self::new(0, timebase)
    }

    /// Tick count.
    pub const fn ticks(self) -> u64 {
        self.ticks
    }

    /// Timebase of this duration.
    pub const fn timebase(self) -> Timebase {
        self.timebase
    }

    /// Converts to another timebase, rounding to the nearest tick.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Overflow`] when the intermediate or final value does
    /// not fit.
    pub fn rescale(self, target: Timebase) -> Result<Self> {
        if target == self.timebase {
            return Ok(self);
        }
        let numerator = u128::from(self.ticks)
            .checked_mul(u128::from(target.denominator))
            .and_then(|value| value.checked_mul(u128::from(self.timebase.numerator)))
            .ok_or(CoreError::Overflow("duration rescale numerator"))?;
        let denominator = u128::from(self.timebase.denominator)
            .checked_mul(u128::from(target.numerator))
            .ok_or(CoreError::Overflow("duration rescale denominator"))?;
        let ticks = numerator
            .checked_add(denominator / 2)
            .ok_or(CoreError::Overflow("duration rescale rounding"))?
            / denominator;
        let ticks =
            u64::try_from(ticks).map_err(|_| CoreError::Overflow("duration rescale result"))?;
        Ok(Self::new(ticks, target))
    }

    /// Adds two durations, rescaling `other` to this duration's timebase.
    ///
    /// # Errors
    ///
    /// Propagates rescale and unsigned overflow failures.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        let other = other.rescale(self.timebase)?;
        let ticks = self
            .ticks
            .checked_add(other.ticks)
            .ok_or(CoreError::Overflow("duration addition"))?;
        Ok(Self::new(ticks, self.timebase))
    }

    /// Duration in seconds, as a floating-point approximation.
    pub fn to_seconds(self) -> f64 {
        self.ticks as f64 * self.timebase.to_f64()
    }
}

/// A positive frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameRate {
    numerator: u32,
    denominator: u32,
}

impl FrameRate {
    /// Creates a frame rate.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] when either part is zero.
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        if numerator == 0 || denominator == 0 {
            return Err(CoreError::InvalidConfig(
                "frame rate parts must be non-zero".into(),
            ));
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// Frames per second numerator.
    pub const fn numerator(self) -> u32 {
        self.numerator
    }

    /// Frames per second denominator.
    pub const fn denominator(self) -> u32 {
        self.denominator
    }

    /// Timebase of one frame at this rate.
    pub fn timebase(self) -> Timebase {
        Timebase::from_parts(self.denominator, self.numerator)
    }

    /// Approximate frames per second.
    pub fn to_f64(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// Divides `numerator` by a positive `denominator`, rounding to nearest with
/// halves away from zero.
fn div_round(numerator: i128, denominator: i128) -> Option<i128> {
    debug_assert!(denominator > 0);
    if numerator >= 0 {
        numerator
            .checked_add(denominator / 2)?
            .checked_div(denominator)
    } else {
        numerator
            .checked_sub(denominator / 2)?
            .checked_div(denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timebase(numerator: u32, denominator: u32) -> Timebase {
        Timebase::new(numerator, denominator).expect("valid timebase")
    }

    #[test]
    fn rational_reduces_and_inverts() {
        let value = Rational::new(30000, 1001).expect("valid");
        assert_eq!(value.reduced(), value);
        assert_eq!(value.invert().expect("invert"), timebase(1001, 30000));
        assert_eq!(
            Rational::new(10, 20).expect("valid").reduced().numerator(),
            1
        );
        assert_eq!(
            Rational::new(10, 20)
                .expect("valid")
                .reduced()
                .denominator(),
            2
        );
        assert!(Rational::new(0, 1).is_err());
        assert!(Rational::new(1, 0).is_err());
    }

    #[test]
    fn rescale_uses_exact_integer_maths() {
        let one_second = MediaTime::new(1, timebase(1, 1));
        let in_90k = one_second.rescale(timebase(1, 90_000)).expect("rescale");
        assert_eq!(in_90k.ticks(), 90_000);

        let negative = MediaTime::new(-1, timebase(1, 1));
        assert_eq!(
            negative
                .rescale(timebase(1, 90_000))
                .expect("rescale")
                .ticks(),
            -90_000
        );
    }

    #[test]
    fn rescale_rounds_to_nearest_tick() {
        let value = MediaTime::new(1, timebase(1, 1000));
        assert_eq!(value.rescale(timebase(1, 1)).expect("rescale").ticks(), 0);
        let value = MediaTime::new(3, timebase(1, 2000));
        assert_eq!(
            value.rescale(timebase(1, 1000)).expect("rescale").ticks(),
            2
        );
    }

    #[test]
    fn timestamps_add_across_timebases() {
        let a = MediaTime::new(1000, timebase(1, 1000));
        let b = MediaTime::new(1, timebase(1, 1));
        assert_eq!(a.checked_add(b).expect("add").ticks(), 2000);
        assert_eq!(a.checked_sub(b).expect("sub").ticks(), 0);
    }

    #[test]
    fn second_conversions_are_bounded() {
        let value = MediaTime::from_seconds(1.5, timebase(1, 1000)).expect("from seconds");
        assert_eq!(value.ticks(), 1500);
        assert!((value.to_seconds() - 1.5).abs() < 1e-12);
        assert!(MediaTime::from_seconds(f64::NAN, timebase(1, 1)).is_err());
        assert!(MediaTime::from_seconds(f64::INFINITY, timebase(1, 1)).is_err());
    }

    #[test]
    fn duration_rescales_without_signed_wrapping() {
        let duration = MediaDuration::new(1, timebase(1, 1));
        assert_eq!(
            duration
                .rescale(timebase(1, 90_000))
                .expect("rescale")
                .ticks(),
            90_000
        );
        assert_eq!(duration.to_seconds(), 1.0);
    }

    #[test]
    fn frame_rate_exposes_its_timebase() {
        let rate = FrameRate::new(30_000, 1001).expect("rate");
        assert_eq!(rate.timebase(), timebase(1001, 30_000));
        assert!((rate.to_f64() - 29.970_029_970_029_97).abs() < 1e-12);
        assert!(FrameRate::new(0, 1).is_err());
        assert!(FrameRate::new(1, 0).is_err());
    }
}
