// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for the partition planner.

/// Errors that can occur during partition planning.
#[derive(Debug, thiserror::Error)]
pub enum PlannerError {
    /// The memory budget is too small to fit even a single layer.
    #[error("budget too small: smallest layer requires {layer_bytes} bytes, budget is {budget_bytes}")]
    BudgetTooSmall {
        layer_bytes: usize,
        budget_bytes: usize,
    },

    /// The model graph is empty.
    #[error("cannot partition an empty model graph")]
    EmptyGraph,

    /// The chosen strategy cannot satisfy the given constraints.
    #[error("strategy '{strategy}' failed: {detail}")]
    StrategyFailed { strategy: String, detail: String },

    /// An error from the resource monitor prevented planning.
    #[error("resource monitor error: {0}")]
    ResourceError(#[from] resource_monitor::MonitorError),
}
