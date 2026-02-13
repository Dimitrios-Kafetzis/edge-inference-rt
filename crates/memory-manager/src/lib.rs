// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # memory-manager
//!
//! A budget-enforced arena allocator for tensor buffers on memory-constrained
//! edge devices such as the Raspberry Pi 4.
//!
//! # Key Components
//!
//! - [`MemoryBudget`] — a hard memory ceiling with human-readable parsing
//!   (`"512M"`, `"1G"`, etc.).
//! - [`MemoryPool`] — the allocator: enforces the budget, maintains a free
//!   list binned by size class, and tracks statistics.
//! - [`BufferGuard`] — an RAII wrapper around allocated buffers. When a guard
//!   is dropped, the buffer is automatically returned to the pool. The borrow
//!   checker prevents use-after-free at compile time.
//! - [`AllocationStats`] — cumulative allocator metrics (peak usage, cache hit
//!   ratio, OOM count).
//!
//! # Ownership Model
//!
//! ```text
//! MemoryPool::allocate(size)
//!       │
//!       ▼
//!   BufferGuard  ◄─── owns Vec<u8>, holds Arc<PoolInner>
//!       │
//!       │  drop()
//!       ▼
//!   PoolInner::return_buffer()  ──► free list
//! ```
//!
//! The pool hands out `BufferGuard`s; each guard holds an `Arc` back to the
//! pool's inner state. On drop, the guard returns its buffer to the free list
//! and decrements the allocated-bytes counter. This is the classic RAII
//! pattern, made safe by Rust's ownership rules.
//!
//! # Example
//! ```
//! use memory_manager::{MemoryPool, MemoryBudget};
//!
//! let pool = MemoryPool::new(MemoryBudget::from_mb(64));
//!
//! // Allocate two buffers.
//! let a = pool.allocate(1024 * 1024).unwrap();  // 1 MB
//! let b = pool.allocate(512 * 1024).unwrap();   // 512 KB
//! assert_eq!(pool.allocated_bytes(), 1024 * 1024 + 512 * 1024);
//!
//! // Returning buffers is automatic.
//! drop(a);
//! assert_eq!(pool.allocated_bytes(), 512 * 1024);
//! ```

mod budget;
mod error;
mod guard;
pub mod pool;
mod stats;

pub use budget::MemoryBudget;
pub use error::MemoryError;
pub use guard::BufferGuard;
pub use pool::MemoryPool;
pub use stats::AllocationStats;
