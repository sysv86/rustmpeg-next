// SPDX-License-Identifier: Apache-2.0
//! Zero-copy immutable byte buffers.
//!
//! A [`Buffer`] is an `Arc<[u8]>` plus a visible range. Slicing and splitting a
//! buffer clone the `Arc` and adjust the range; the bytes are never copied.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use core::ops::{Deref, Range};

use crate::error::{CoreError, Result};

/// An immutable, reference-counted byte buffer.
#[derive(Clone, Default)]
pub struct Buffer {
    data: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl Buffer {
    /// Creates an empty buffer.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Copies a byte slice into a new buffer.
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self::from_arc(Arc::from(bytes))
    }

    /// Moves a vector into a new buffer.
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self::from_arc(Arc::from(bytes))
    }

    /// Wraps an existing shared allocation without copying it.
    pub fn from_arc(data: Arc<[u8]>) -> Self {
        let end = data.len();
        Self {
            data,
            start: 0,
            end,
        }
    }

    /// Visible length in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// `true` when no bytes are visible.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Visible bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[self.start..self.end]
    }

    /// Pointer to the first visible byte.
    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    /// Returns a zero-copy sub-range.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when `range` is reversed or extends
    /// beyond the visible bytes.
    pub fn slice(&self, range: Range<usize>) -> Result<Self> {
        if range.start > range.end || range.end > self.len() {
            return Err(CoreError::index(range.start, "buffer slice"));
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            start: self.start + range.start,
            end: self.start + range.end,
        })
    }

    /// Splits the buffer at `at`, returning the suffix and keeping the prefix
    /// in `self`. Both halves share the original allocation.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when `at` exceeds the visible length.
    pub fn split_off(&mut self, at: usize) -> Result<Self> {
        if at > self.len() {
            return Err(CoreError::index(at, "buffer split"));
        }
        let suffix = Self {
            data: Arc::clone(&self.data),
            start: self.start + at,
            end: self.end,
        };
        self.end = self.start + at;
        Ok(suffix)
    }

    /// Drops the first `count` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when `count` exceeds the length.
    pub fn advance(&mut self, count: usize) -> Result<()> {
        if count > self.len() {
            return Err(CoreError::index(count, "buffer advance"));
        }
        self.start += count;
        Ok(())
    }

    /// Shortens the buffer to `len` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when `len` exceeds the length.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len() {
            return Err(CoreError::index(len, "buffer truncate"));
        }
        self.end = self.start + len;
        Ok(())
    }

    /// `true` when the visible bytes start with `prefix`.
    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        self.as_slice().starts_with(prefix)
    }
}

impl Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Buffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl core::fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Buffer({} bytes)", self.len())
    }
}

impl From<Vec<u8>> for Buffer {
    fn from(value: Vec<u8>) -> Self {
        Self::from_vec(value)
    }
}

impl From<&[u8]> for Buffer {
    fn from(value: &[u8]) -> Self {
        Self::from_slice(value)
    }
}

impl From<Arc<[u8]>> for Buffer {
    fn from(value: Arc<[u8]>) -> Self {
        Self::from_arc(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn slicing_is_zero_copy() {
        let base = Buffer::from_vec((0..16).collect());
        let slice = base.slice(3..11).expect("slice");
        assert_eq!(slice.as_slice(), &[3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(
            slice.as_ptr(),
            base.as_ptr().wrapping_add(3),
            "a slice must share the allocation"
        );
    }

    #[test]
    fn split_off_shares_the_allocation() {
        let mut buffer = Buffer::from_vec((0..8).collect());
        let suffix = buffer.split_off(3).expect("split");
        assert_eq!(buffer.as_slice(), &[0, 1, 2]);
        assert_eq!(suffix.as_slice(), &[3, 4, 5, 6, 7]);
        assert_eq!(buffer.as_ptr().wrapping_add(3), suffix.as_ptr());
    }

    #[test]
    fn advance_and_truncate_stay_in_bounds() {
        let mut buffer = Buffer::from_slice(b"abcdef");
        buffer.advance(2).expect("advance");
        assert_eq!(buffer.as_slice(), b"cdef");
        buffer.truncate(3).expect("truncate");
        assert_eq!(buffer.as_slice(), b"cde");
        assert!(buffer.advance(4).is_err());
        assert!(buffer.truncate(4).is_err());
        assert!(buffer.slice(2..99).is_err());
    }

    #[test]
    fn equality_compares_visible_bytes() {
        let a = Buffer::from_slice(b"prefix-value");
        let b = a.slice(7..12).expect("slice");
        assert_eq!(b, Buffer::from_slice(b"value"));
        assert_eq!(format!("{b:?}"), "Buffer(5 bytes)");
    }

    #[test]
    fn empty_buffer_is_valid() {
        let buffer = Buffer::empty();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert!(buffer.starts_with(b""));
        assert!(!buffer.starts_with(b"x"));
    }
}
