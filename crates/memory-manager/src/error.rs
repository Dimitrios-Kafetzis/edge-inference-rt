// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for memory management.

/// Errors that can occur during memory allocation and management.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// The requested allocation would exceed the memory budget.
    #[error("out of memory: requested {requested_bytes} bytes, but only {available_bytes} available (budget: {budget_bytes})")]
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
        budget_bytes: usize,
    },

    /// Attempted to allocate a zero-sized buffer.
    #[error("cannot allocate zero-sized buffer")]
    ZeroSizedAllocation,

    /// An internal pool inconsistency was detected.
    #[error("pool integrity error: {0}")]
    PoolCorruption(String),
}
