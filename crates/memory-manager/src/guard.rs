// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! RAII buffer guard that returns memory to the pool on drop.
//!
//! [`BufferGuard`] is the core mechanism through which Rust's ownership
//! model enforces memory safety in the pool. When a guard is dropped,
//! it automatically returns its buffer to the free list and decrements
//! the pool's allocated-bytes counter. The borrow checker prevents
//! use-after-free at compile time.

use crate::pool::PoolInner;
use std::sync::Arc;

/// An RAII guard wrapping an allocated tensor buffer.
///
/// When a `BufferGuard` is dropped, its memory is automatically returned
/// to the [`MemoryPool`](crate::MemoryPool). The borrow checker prevents
/// use-after-free.
///
/// # Example
/// ```ignore
/// let guard = pool.allocate(1024)?;
/// guard.as_slice();          // use the buffer
/// drop(guard);               // memory returned to pool
/// // guard.as_slice();       // compile error — moved value
/// ```
pub struct BufferGuard {
    /// The raw buffer. Wrapped in `Option` so we can `take()` it in `drop()`.
    data: Option<Vec<u8>>,
    /// Handle back to the pool for deallocation tracking.
    pool: Arc<PoolInner>,
    /// Size of this allocation in bytes (for accounting).
    size_bytes: usize,
}

impl BufferGuard {
    /// Creates a new buffer guard (called internally by the pool).
    pub(crate) fn new(data: Vec<u8>, pool: Arc<PoolInner>, size_bytes: usize) -> Self {
        Self {
            data: Some(data),
            pool,
            size_bytes,
        }
    }

    /// Returns an immutable view of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        self.data.as_ref().expect("buffer already consumed")
    }

    /// Returns a mutable view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.data.as_mut().expect("buffer already consumed")
    }

    /// Returns the size of this allocation in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Interprets the buffer as a slice of `f32`.
    ///
    /// # Panics
    /// Panics if `size_bytes` is not a multiple of 4.
    pub fn as_f32_slice(&self) -> &[f32] {
        let bytes = self.as_slice();
        assert!(
            bytes.len() % 4 == 0,
            "buffer size {} is not a multiple of 4",
            bytes.len()
        );
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
    }

    /// Interprets the buffer as a mutable slice of `f32`.
    ///
    /// # Panics
    /// Panics if `size_bytes` is not a multiple of 4.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        let bytes = self.as_mut_slice();
        assert!(
            bytes.len() % 4 == 0,
            "buffer size {} is not a multiple of 4",
            bytes.len()
        );
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4) }
    }
}

impl Drop for BufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.data.take() {
            self.pool.return_buffer(buffer, self.size_bytes);
        }
    }
}

// BufferGuard is Send because Vec<u8> is Send and Arc<PoolInner> is Send.
// It is NOT Sync because &mut access to the buffer isn't synchronized.
unsafe impl Send for BufferGuard {}

impl std::fmt::Debug for BufferGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferGuard")
            .field("size_bytes", &self.size_bytes)
            .field("has_data", &self.data.is_some())
            .finish()
    }
}
