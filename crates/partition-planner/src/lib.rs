// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # partition-planner
//!
//! Partitions a validated `ModelGraph` into memory-budget-respecting
//! execution groups using pluggable strategies.
//!
//! # Strategies
//!
//! | Strategy | Groups | Latency | Memory usage |
//! |---|---|---|---|
//! | [`Sequential`] | N (one per layer) | Highest | Lowest |
//! | [`GreedyGrouping`] | Fewer | Medium | Medium |
//! | [`SpeculativePrefetch`] | Fewer (with headroom) | Lowest* | Medium–High |
//!
//! *When the runtime supports async prefetch.
//!
//! # Trait-Based Extensibility
//!
//! All strategies implement [`PartitionStrategy`], so new strategies can
//! be added without modifying the runtime:
//!
//! ```ignore
//! struct MyCustomStrategy;
//! impl PartitionStrategy for MyCustomStrategy {
//!     fn name(&self) -> &str { "custom" }
//!     fn plan(&self, graph: &ModelGraph<Validated>, budget: MemoryBudget)
//!         -> Result<ExecutionPlan, PlannerError> { /* ... */ }
//! }
//! ```
//!
//! # Example
//! ```no_run
//! use partition_planner::{GreedyGrouping, PartitionStrategy};
//! use memory_manager::MemoryBudget;
//! use model_ir::ModelLoader;
//! use std::path::Path;
//!
//! let graph = ModelLoader::load(Path::new("./model")).unwrap();
//! let plan = GreedyGrouping::new().plan(&graph, MemoryBudget::from_mb(512)).unwrap();
//! println!("{}", plan.summary());
//! ```

mod error;
pub(crate) mod plan;
pub mod strategy;

pub use error::PlannerError;
pub use plan::{ExecutionPlan, LayerGroup};
pub use strategy::greedy::GreedyGrouping;
pub use strategy::sequential::Sequential;
pub use strategy::speculative::SpeculativePrefetch;
pub use strategy::PartitionStrategy;

/// Selects and runs the best strategy based on the current system state.
///
/// Heuristic:
/// - If the system is resource-constrained (thermal throttling, low memory),
///   use [`Sequential`] for safety.
/// - If the budget comfortably exceeds the largest layer with 20%+ headroom,
///   use [`SpeculativePrefetch`].
/// - Otherwise, use [`GreedyGrouping`].
pub fn auto_plan(
    graph: &model_ir::ModelGraph<model_ir::graph::Validated>,
    budget: memory_manager::MemoryBudget,
    constrained: bool,
) -> Result<ExecutionPlan, PlannerError> {
    if constrained {
        tracing::info!("system constrained → using sequential strategy");
        return Sequential::new().plan(graph, budget);
    }

    let max_layer = graph.max_layer_bytes();
    let headroom_ratio = if max_layer > 0 {
        (budget.as_bytes() as f64 - max_layer as f64) / budget.as_bytes() as f64
    } else {
        1.0
    };

    if headroom_ratio > 0.25 {
        tracing::info!(
            "headroom ratio {:.0}% → using speculative-prefetch strategy",
            headroom_ratio * 100.0,
        );
        SpeculativePrefetch::default().plan(graph, budget)
    } else {
        tracing::info!(
            "headroom ratio {:.0}% → using greedy-grouping strategy",
            headroom_ratio * 100.0,
        );
        GreedyGrouping::new().plan(graph, budget)
    }
}
