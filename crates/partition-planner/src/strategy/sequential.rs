// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Sequential partitioning strategy.
//!
//! The simplest strategy: each layer forms its own group. This maximises
//! buffer reuse (only one layer's weights are in memory at a time) at
//! the cost of higher latency from frequent weight loading.
//!
//! # When to use
//! - Memory budget is extremely tight (barely fits the largest layer).
//! - Debugging: isolates each layer for inspection.
//! - Baseline for benchmarking other strategies.

use crate::plan::PlanBuilder;
use crate::strategy::PartitionStrategy;
use crate::{ExecutionPlan, PlannerError};
use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, ModelGraph};

/// One layer per group — maximal buffer reuse, highest latency.
#[derive(Debug, Clone, Default)]
pub struct Sequential;

impl Sequential {
    pub fn new() -> Self {
        Self
    }
}

impl PartitionStrategy for Sequential {
    fn name(&self) -> &str {
        "sequential"
    }

    fn plan(
        &self,
        graph: &ModelGraph<Validated>,
        budget: MemoryBudget,
    ) -> Result<ExecutionPlan, PlannerError> {
        if graph.num_layers() == 0 {
            return Err(PlannerError::EmptyGraph);
        }

        let budget_bytes = budget.as_bytes();
        let mut builder = PlanBuilder::new(self.name(), budget_bytes);

        for layer in graph.iter_layers() {
            let mem = layer.estimated_total_bytes();

            if mem > budget_bytes {
                return Err(PlannerError::BudgetTooSmall {
                    layer_bytes: mem,
                    budget_bytes,
                });
            }

            builder.add_group(vec![layer.index], mem);
        }

        let plan = builder.build();
        plan.validate()?;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::{LayerDef, LayerType};
    use tensor_core::{DType, Shape};

    fn make_graph(n: usize, hidden: usize) -> ModelGraph<Validated> {
        let layers: Vec<LayerDef> = (0..n)
            .map(|i| LayerDef {
                name: format!("layer.{i}"),
                layer_type: LayerType::Linear,
                index: i,
                weight_names: vec![format!("w.{i}")],
                weight_shapes: vec![Shape::matrix(hidden, hidden)],
                dtype: DType::F32,
                input_shape: Shape::matrix(1, hidden),
                output_shape: Shape::matrix(1, hidden),
            })
            .collect();
        ModelGraph::new("test".into(), layers).validate().unwrap()
    }

    #[test]
    fn test_sequential_basic() {
        let graph = make_graph(4, 64);
        let plan = Sequential::new()
            .plan(&graph, MemoryBudget::from_mb(10))
            .unwrap();

        assert_eq!(plan.num_groups(), 4);
        assert_eq!(plan.total_layers(), 4);
        // Each group has exactly one layer.
        for (i, g) in plan.groups.iter().enumerate() {
            assert_eq!(g.layer_indices, vec![i]);
        }
    }

    #[test]
    fn test_sequential_single_layer() {
        let graph = make_graph(1, 64);
        let plan = Sequential::new()
            .plan(&graph, MemoryBudget::from_mb(10))
            .unwrap();
        assert_eq!(plan.num_groups(), 1);
    }

    #[test]
    fn test_sequential_budget_too_small() {
        let graph = make_graph(2, 256);
        // Each layer: 256*256*4 = 262144 bytes weight + 2*(1*256*4) = 2048 activation
        // Total per layer ≈ 264 KB.
        let result = Sequential::new().plan(&graph, MemoryBudget::from_bytes(1000));
        assert!(matches!(result, Err(PlannerError::BudgetTooSmall { .. })));
    }

    #[test]
    fn test_sequential_validates() {
        let graph = make_graph(6, 64);
        let plan = Sequential::new()
            .plan(&graph, MemoryBudget::from_mb(10))
            .unwrap();
        plan.validate().unwrap();
    }
}
