// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Arena-style memory pool with budget enforcement.
//!
//! The [`MemoryPool`] is the central allocator for tensor buffers. It:
//!
//! 1. Enforces a hard memory ceiling — allocations that would exceed the
//!    budget return `Err(OutOfMemory)`.
//! 2. Maintains a free list of returned buffers, binned by size class,
//!    to avoid repeated heap allocation in the inference hot path.
//! 3. Tracks allocation statistics for profiling.
//!
//! # Thread Safety
//! `MemoryPool` is `Send + Sync` — it can be safely shared across
//! async tasks via `Arc<MemoryPool>`.
//!
//! # Size Classes
//! Returned buffers are binned by "size class" (rounded up to the nearest
//! power of 2). When a new allocation request comes in, the pool first
//! checks if a buffer of the right size class is available in the free list.
//! This trades a small amount of memory waste for significant reduction in
//! allocation overhead.

use crate::{AllocationStats, BufferGuard, MemoryBudget, MemoryError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Minimum size class: 4 KB. Anything smaller is rounded up.
const MIN_SIZE_CLASS: usize = 4096;

/// Internal pool state, shared between the pool and buffer guards via `Arc`.
///
/// This is the "inner" type that buffer guards hold a reference to, so
/// they can return memory without needing a reference to the full `MemoryPool`.
pub struct PoolInner {
    /// The memory budget ceiling.
    budget: MemoryBudget,
    /// Currently allocated bytes (live, not yet returned).
    allocated_bytes: AtomicUsize,
    /// Free buffer cache: size_class → available buffers.
    free_buffers: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
    /// Total bytes held in the free list (for shrink accounting).
    free_list_bytes: AtomicUsize,
    /// Statistics (behind a Mutex since updates are infrequent).
    stats: Mutex<AllocationStats>,
}

impl PoolInner {
    /// Called by `BufferGuard::drop` to return a buffer to the free list.
    pub(crate) fn return_buffer(&self, buffer: Vec<u8>, size_bytes: usize) {
        // Decrement the allocated counter.
        self.allocated_bytes.fetch_sub(size_bytes, Ordering::Release);

        // Record the deallocation.
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_deallocation();
        }

        // Return buffer to the free list, keyed by size class.
        let size_class = size_class_for(size_bytes);
        self.free_list_bytes.fetch_add(buffer.len(), Ordering::Release);

        if let Ok(mut free) = self.free_buffers.lock() {
            free.entry(size_class).or_default().push(buffer);
        }
    }
}

/// The primary memory allocator for tensor buffers.
///
/// # Example
/// ```
/// use memory_manager::{MemoryPool, MemoryBudget};
///
/// let pool = MemoryPool::new(MemoryBudget::from_mb(64));
///
/// // Allocate a 1 MB buffer.
/// let guard = pool.allocate(1024 * 1024).unwrap();
/// assert_eq!(pool.allocated_bytes(), 1024 * 1024);
///
/// // Buffer is returned when guard is dropped.
/// drop(guard);
/// assert_eq!(pool.allocated_bytes(), 0);
/// ```
pub struct MemoryPool {
    inner: Arc<PoolInner>,
}

impl MemoryPool {
    /// Creates a new memory pool with the given budget.
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                budget,
                allocated_bytes: AtomicUsize::new(0),
                free_buffers: Mutex::new(HashMap::new()),
                free_list_bytes: AtomicUsize::new(0),
                stats: Mutex::new(AllocationStats::default()),
            }),
        }
    }

    /// Allocates a buffer of `size_bytes`.
    ///
    /// Returns `Err(OutOfMemory)` if the allocation would exceed the budget.
    /// If a suitably sized buffer exists in the free list, it is reused
    /// (cache hit) — otherwise a new `Vec<u8>` is allocated (cache miss).
    ///
    /// The returned [`BufferGuard`] automatically returns the buffer to
    /// the pool when dropped.
    pub fn allocate(&self, size_bytes: usize) -> Result<BufferGuard, MemoryError> {
        if size_bytes == 0 {
            return Err(MemoryError::ZeroSizedAllocation);
        }

        // Check budget.
        let current = self.inner.allocated_bytes.load(Ordering::Acquire);
        let budget = self.inner.budget.as_bytes();

        if current + size_bytes > budget {
            if let Ok(mut stats) = self.inner.stats.lock() {
                stats.record_oom();
            }
            return Err(MemoryError::OutOfMemory {
                requested_bytes: size_bytes,
                available_bytes: budget.saturating_sub(current),
                budget_bytes: budget,
            });
        }

        // Try to reuse a buffer from the free list.
        let size_class = size_class_for(size_bytes);
        let mut buffer = None;

        if let Ok(mut free) = self.inner.free_buffers.lock() {
            if let Some(class_buffers) = free.get_mut(&size_class) {
                if let Some(mut buf) = class_buffers.pop() {
                    // Found a reusable buffer. Resize if needed (the size class
                    // may be larger than requested — that's fine, we just
                    // zero the portion we'll use).
                    if buf.len() < size_bytes {
                        buf.resize(size_bytes, 0);
                    } else {
                        // Zero only the portion we hand out.
                        buf[..size_bytes].fill(0);
                    }
                    self.inner
                        .free_list_bytes
                        .fetch_sub(buf.len().min(size_class), Ordering::Release);
                    buffer = Some(buf);
                }
            }
        }

        let is_hit = buffer.is_some();
        let data = buffer.unwrap_or_else(|| vec![0u8; size_bytes]);

        // Update allocated counter.
        self.inner
            .allocated_bytes
            .fetch_add(size_bytes, Ordering::Release);

        // Update stats.
        if let Ok(mut stats) = self.inner.stats.lock() {
            if is_hit {
                stats.record_cache_hit(size_bytes);
            } else {
                stats.record_cache_miss(size_bytes);
            }
            let new_total = self.inner.allocated_bytes.load(Ordering::Acquire);
            stats.update_peak(new_total);
        }

        Ok(BufferGuard::new(data, Arc::clone(&self.inner), size_bytes))
    }

    /// Returns the number of bytes currently allocated (live, not yet returned).
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes.load(Ordering::Acquire)
    }

    /// Returns the number of bytes remaining before hitting the budget.
    pub fn available_bytes(&self) -> usize {
        let budget = self.inner.budget.as_bytes();
        let allocated = self.allocated_bytes();
        budget.saturating_sub(allocated)
    }

    /// Returns the memory budget.
    pub fn budget(&self) -> MemoryBudget {
        self.inner.budget
    }

    /// Returns a snapshot of allocation statistics.
    pub fn stats(&self) -> AllocationStats {
        self.inner
            .stats
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Evicts all cached free buffers, releasing memory back to the OS.
    ///
    /// This does not affect currently-allocated buffers — only the
    /// free list is cleared. Useful when the system is under memory
    /// pressure and we want to return unused memory.
    pub fn shrink(&self) {
        if let Ok(mut free) = self.inner.free_buffers.lock() {
            free.clear();
            self.inner.free_list_bytes.store(0, Ordering::Release);
        }
    }

    /// Returns the approximate number of bytes held in the free list.
    pub fn free_list_bytes(&self) -> usize {
        self.inner.free_list_bytes.load(Ordering::Acquire)
    }
}

// MemoryPool is Send + Sync because all interior mutability is behind
// Mutex or AtomicUsize.
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

/// Computes the size class for a given allocation size.
///
/// Returns the smallest power of 2 that is ≥ `size` and ≥ `MIN_SIZE_CLASS`.
fn size_class_for(size: usize) -> usize {
    let min = size.max(MIN_SIZE_CLASS);
    min.next_power_of_two()
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("budget", &self.inner.budget)
            .field("allocated_bytes", &self.allocated_bytes())
            .field("available_bytes", &self.available_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_drop() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        let guard = pool.allocate(1024).unwrap();
        assert_eq!(pool.allocated_bytes(), 1024);
        assert_eq!(guard.size_bytes(), 1024);

        drop(guard);
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn test_buffer_contents() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        let mut guard = pool.allocate(16).unwrap();
        // Buffer should be zeroed.
        assert!(guard.as_slice().iter().all(|&b| b == 0));

        // Write something.
        guard.as_mut_slice()[0] = 42;
        assert_eq!(guard.as_slice()[0], 42);
    }

    #[test]
    fn test_oom() {
        let pool = MemoryPool::new(MemoryBudget::from_bytes(1024));

        let _g1 = pool.allocate(512).unwrap();
        let _g2 = pool.allocate(512).unwrap();

        // This should fail — budget exhausted.
        let result = pool.allocate(1);
        assert!(matches!(result, Err(MemoryError::OutOfMemory { .. })));
    }

    #[test]
    fn test_zero_allocation() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));
        let result = pool.allocate(0);
        assert!(matches!(result, Err(MemoryError::ZeroSizedAllocation)));
    }

    #[test]
    fn test_free_list_reuse() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        // Allocate and return.
        let guard = pool.allocate(4096).unwrap();
        drop(guard);

        // Allocate again — should be a cache hit.
        let _guard2 = pool.allocate(4096).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_multiple_allocations() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(10));

        let mut guards = Vec::new();
        for _ in 0..10 {
            guards.push(pool.allocate(1024 * 100).unwrap()); // 100 KB each
        }

        assert_eq!(pool.allocated_bytes(), 10 * 100 * 1024);

        // Drop all.
        guards.clear();
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn test_available_bytes() {
        let pool = MemoryPool::new(MemoryBudget::from_bytes(10000));

        assert_eq!(pool.available_bytes(), 10000);
        let _g = pool.allocate(3000).unwrap();
        assert_eq!(pool.available_bytes(), 7000);
    }

    #[test]
    fn test_shrink() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        let g = pool.allocate(8192).unwrap();
        drop(g);
        assert!(pool.free_list_bytes() > 0);

        pool.shrink();
        assert_eq!(pool.free_list_bytes(), 0);
    }

    #[test]
    fn test_stats_peak() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        let g1 = pool.allocate(1000).unwrap();
        let g2 = pool.allocate(2000).unwrap();
        // Peak = 3000.
        drop(g1);
        // Current = 2000, but peak should still be 3000.
        drop(g2);

        let stats = pool.stats();
        assert_eq!(stats.peak_allocated_bytes, 3000);
    }

    #[test]
    fn test_stats_oom_count() {
        let pool = MemoryPool::new(MemoryBudget::from_bytes(100));
        let _ = pool.allocate(200); // OOM.
        let _ = pool.allocate(200); // OOM again.

        let stats = pool.stats();
        assert_eq!(stats.oom_count, 2);
    }

    #[test]
    fn test_f32_slice() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));
        let mut guard = pool.allocate(16).unwrap(); // 4 × f32.

        let slice = guard.as_f32_slice_mut();
        assert_eq!(slice.len(), 4);
        slice[0] = 1.0;
        slice[3] = 4.0;

        let ro = guard.as_f32_slice();
        assert_eq!(ro[0], 1.0);
        assert_eq!(ro[3], 4.0);
    }

    #[test]
    fn test_size_class() {
        assert_eq!(size_class_for(1), MIN_SIZE_CLASS); // Below minimum.
        assert_eq!(size_class_for(4096), 4096);        // Exact power of 2.
        assert_eq!(size_class_for(5000), 8192);         // Rounded up.
        assert_eq!(size_class_for(1024 * 1024), 1024 * 1024); // 1 MB.
    }

    #[test]
    fn test_debug_format() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(64));
        let debug = format!("{pool:?}");
        assert!(debug.contains("MemoryPool"));
        assert!(debug.contains("budget"));
    }

    #[test]
    fn test_returned_buffer_is_zeroed() {
        let pool = MemoryPool::new(MemoryBudget::from_mb(1));

        // Allocate, write data, then return.
        let mut g = pool.allocate(4096).unwrap();
        g.as_mut_slice().fill(0xFF);
        drop(g);

        // Re-allocate: the buffer should be zeroed.
        let g2 = pool.allocate(4096).unwrap();
        assert!(g2.as_slice()[..4096].iter().all(|&b| b == 0));
    }
}
