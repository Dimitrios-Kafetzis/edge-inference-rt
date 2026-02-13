// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! The [`PartitionStrategy`] trait and strategy implementations.

pub mod greedy;
pub mod sequential;
pub mod speculative;

use crate::{ExecutionPlan, PlannerError};
use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, ModelGraph};

/// Trait for partition strategies.
///
/// Each strategy takes a validated model graph and a memory budget,
/// and produces an [`ExecutionPlan`] that respects the budget.
///
/// Strategies are purely algorithmic — no I/O or system calls — making
/// them trivially unit-testable and amenable to property-based testing.
pub trait PartitionStrategy: Send + Sync {
    /// Human-readable name of this strategy.
    fn name(&self) -> &str;

    /// Produces an execution plan for the given model and budget.
    fn plan(
        &self,
        graph: &ModelGraph<Validated>,
        budget: MemoryBudget,
    ) -> Result<ExecutionPlan, PlannerError>;
}
