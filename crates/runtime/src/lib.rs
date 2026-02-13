// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # runtime
//!
//! The execution engine that orchestrates partitioned transformer inference.
//!
//! The runtime takes:
//! - A validated `ModelGraph` from `model-ir`.
//! - An `ExecutionPlan` from `partition-planner`.
//! - A `MemoryPool` from `memory-manager`.
//!
//! And executes the model group by group, managing weight loading, activation
//! flow between layers, and per-layer timing/memory profiling.
//!
//! # Type-State Pipeline
//! The runtime enforces a type-safe pipeline:
//! ```text
//! InferenceEngine<Idle> → InferenceEngine<Planned> → InferenceEngine<Ready>
//! ```
//! Transitions are compile-time checked.
//!
//! # Async Execution
//! Uses `tokio` with a configurable thread pool. On RPi 4, this maps
//! well to the 4× Cortex-A72 cores.

mod config;
mod engine;
mod error;
mod metrics;
mod weight_loader;

pub use config::RuntimeConfig;
pub use engine::{EngineState, Idle, InferenceEngine, InferenceOutput, Planned, Ready};
pub use error::RuntimeError;
pub use metrics::{InferenceMetrics, LayerMetrics};
pub use weight_loader::WeightLoader;
