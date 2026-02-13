// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Allocation statistics for profiling and diagnostics.
//!
//! [`AllocationStats`] tracks cumulative metrics about how the memory pool
//! is being used: hit rates, peak usage, and OOM events. These stats are
//! essential for tuning the memory budget.

/// Cumulative statistics about memory pool usage.
///
/// Useful for understanding memory pressure and tuning the budget.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AllocationStats {
    /// Total number of allocation requests.
    pub total_allocations: u64,
    /// Number of allocations served from the free list (cache hits).
    pub cache_hits: u64,
    /// Number of allocations that required fresh memory.
    pub cache_misses: u64,
    /// Number of allocation requests that failed due to budget exhaustion.
    pub oom_count: u64,
    /// Peak memory usage in bytes.
    pub peak_allocated_bytes: usize,
    /// Total bytes ever allocated (including freed and reallocated).
    pub cumulative_allocated_bytes: u64,
    /// Total number of buffer returns (drops).
    pub total_deallocations: u64,
}

impl AllocationStats {
    /// Returns the cache hit ratio as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no allocations have been made.
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Records a successful allocation from the free list.
    pub(crate) fn record_cache_hit(&mut self, size: usize) {
        self.total_allocations += 1;
        self.cache_hits += 1;
        self.cumulative_allocated_bytes += size as u64;
    }

    /// Records a successful allocation that required new memory.
    pub(crate) fn record_cache_miss(&mut self, size: usize) {
        self.total_allocations += 1;
        self.cache_misses += 1;
        self.cumulative_allocated_bytes += size as u64;
    }

    /// Records an OOM event.
    pub(crate) fn record_oom(&mut self) {
        self.total_allocations += 1;
        self.oom_count += 1;
    }

    /// Records a deallocation (buffer returned to pool).
    pub(crate) fn record_deallocation(&mut self) {
        self.total_deallocations += 1;
    }

    /// Updates the peak allocation high-water mark if needed.
    pub(crate) fn update_peak(&mut self, current_bytes: usize) {
        if current_bytes > self.peak_allocated_bytes {
            self.peak_allocated_bytes = current_bytes;
        }
    }

    /// Returns a human-readable summary.
    pub fn summary(&self) -> String {
        let peak_mb = self.peak_allocated_bytes as f64 / (1024.0 * 1024.0);
        format!(
            "Allocations: {} total ({} hits, {} misses, {:.0}% hit rate), \
             {} OOMs, peak {:.2} MB, {} deallocations",
            self.total_allocations,
            self.cache_hits,
            self.cache_misses,
            self.cache_hit_ratio() * 100.0,
            self.oom_count,
            peak_mb,
            self.total_deallocations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let s = AllocationStats::default();
        assert_eq!(s.total_allocations, 0);
        assert_eq!(s.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut s = AllocationStats::default();
        s.record_cache_hit(100);
        s.record_cache_hit(100);
        s.record_cache_miss(200);
        assert!((s.cache_hit_ratio() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_peak_tracking() {
        let mut s = AllocationStats::default();
        s.update_peak(100);
        assert_eq!(s.peak_allocated_bytes, 100);
        s.update_peak(50);
        assert_eq!(s.peak_allocated_bytes, 100); // Doesn't decrease.
        s.update_peak(200);
        assert_eq!(s.peak_allocated_bytes, 200);
    }

    #[test]
    fn test_cumulative_bytes() {
        let mut s = AllocationStats::default();
        s.record_cache_miss(1000);
        s.record_cache_hit(500);
        assert_eq!(s.cumulative_allocated_bytes, 1500);
    }

    #[test]
    fn test_summary() {
        let mut s = AllocationStats::default();
        s.record_cache_miss(1024 * 1024);
        s.record_cache_hit(512 * 1024);
        s.update_peak(1024 * 1024);
        let summary = s.summary();
        assert!(summary.contains("2 total"));
        assert!(summary.contains("1 hits"));
        assert!(summary.contains("1 misses"));
    }
}
