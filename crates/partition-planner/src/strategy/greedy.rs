// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Greedy grouping partitioning strategy.
//!
//! Packs consecutive layers into the same group while the combined
//! memory footprint stays within the budget. This reduces the number
//! of weight-loading rounds compared to [`crate::Sequential`], at the cost
//! of higher per-group memory usage.
//!
//! # Memory Model
//!
//! Within a group, **all** layer weights must be resident simultaneously.
//! Activations, however, are pipelined: only the current layer's input
//! and output buffers are live at once. Therefore the peak memory for
//! a group is:
//!
//! ```text
//! group_mem = sum(weight_bytes for each layer)
//!           + max(activation_bytes for any layer in group)
//! ```
//!
//! # When to use
//! - Default strategy for most workloads — a good balance between
//!   memory efficiency and execution throughput.

use crate::plan::PlanBuilder;
use crate::strategy::PartitionStrategy;
use crate::{ExecutionPlan, PlannerError};
use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, LayerDef, ModelGraph};

/// Greedy grouping: pack consecutive layers into budget-fitting groups.
#[derive(Debug, Clone, Default)]
pub struct GreedyGrouping;

impl GreedyGrouping {
    pub fn new() -> Self {
        Self
    }
}

impl PartitionStrategy for GreedyGrouping {
    fn name(&self) -> &str {
        "greedy-grouping"
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
        let layers: Vec<&LayerDef> = graph.iter_layers().collect();

        // Pre-check: every single layer must fit alone.
        for layer in &layers {
            let solo_mem = layer.estimated_total_bytes();
            if solo_mem > budget_bytes {
                return Err(PlannerError::BudgetTooSmall {
                    layer_bytes: solo_mem,
                    budget_bytes,
                });
            }
        }

        let mut builder = PlanBuilder::new(self.name(), budget_bytes);
        let mut i = 0;

        while i < layers.len() {
            // Start a new group with layer i.
            let mut group_weight_bytes = layers[i].estimated_weight_bytes();
            let mut group_max_activation = layers[i].estimated_activation_bytes();
            let mut group_indices = vec![layers[i].index];

            // Try to extend the group with subsequent layers.
            let mut j = i + 1;
            while j < layers.len() {
                let next_weight = layers[j].estimated_weight_bytes();
                let next_activation = layers[j].estimated_activation_bytes();

                let candidate_weight = group_weight_bytes + next_weight;
                let candidate_activation = group_max_activation.max(next_activation);
                let candidate_total = candidate_weight + candidate_activation;

                if candidate_total > budget_bytes {
                    break; // Adding this layer would exceed the budget.
                }

                group_weight_bytes = candidate_weight;
                group_max_activation = candidate_activation;
                group_indices.push(layers[j].index);
                j += 1;
            }

            let group_mem = group_weight_bytes + group_max_activation;
            builder.add_group(group_indices, group_mem);
            i = j;
        }

        let plan = builder.build();
        plan.validate()?;
        Ok(plan)
    }
}

/// Estimates the peak memory for a group of layers.
///
/// Exposed for use by other strategies and for testing.
#[allow(dead_code)]
pub(crate) fn estimate_group_memory(layers: &[&LayerDef]) -> usize {
    let weight_bytes: usize = layers.iter().map(|l| l.estimated_weight_bytes()).sum();
    let max_activation: usize = layers
        .iter()
        .map(|l| l.estimated_activation_bytes())
        .max()
        .unwrap_or(0);
    weight_bytes + max_activation
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::{LayerDef, LayerType};
    use tensor_core::{DType, Shape};

    /// Creates layers with configurable weight sizes.
    fn make_layers(sizes: &[(usize, usize)]) -> Vec<LayerDef> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &(hidden, _weight_dim))| {
                LayerDef {
                    name: format!("layer.{i}"),
                    layer_type: LayerType::Linear,
                    index: i,
                    weight_names: vec![format!("w.{i}")],
                    weight_shapes: vec![Shape::matrix(hidden, _weight_dim)],
                    dtype: DType::F32,
                    input_shape: Shape::matrix(1, hidden),
                    output_shape: Shape::matrix(1, hidden),
                }
            })
            .collect()
    }

    fn make_graph(layers: Vec<LayerDef>) -> ModelGraph<Validated> {
        ModelGraph::new("test".into(), layers).validate().unwrap()
    }

    fn uniform_graph(n: usize, hidden: usize) -> ModelGraph<Validated> {
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
        make_graph(layers)
    }

    #[test]
    fn test_greedy_all_fit() {
        // 4 tiny layers, huge budget → should all be in one group.
        let graph = uniform_graph(4, 16);
        let plan = GreedyGrouping::new()
            .plan(&graph, MemoryBudget::from_mb(10))
            .unwrap();

        assert_eq!(plan.num_groups(), 1);
        assert_eq!(plan.groups[0].layer_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_greedy_one_per_group() {
        // 4 layers, budget barely fits one → falls back to sequential-like.
        let graph = uniform_graph(4, 64);
        // Each layer: 64*64*4 = 16384 weights + 2*(1*64*4) = 512 activations = 16896 bytes.
        let plan = GreedyGrouping::new()
            .plan(&graph, MemoryBudget::from_bytes(17000))
            .unwrap();

        assert_eq!(plan.num_groups(), 4);
    }

    #[test]
    fn test_greedy_mixed_sizes() {
        // Layers with different weight sizes.
        // Budget = 50000 bytes.
        // L0: 32*32*4 = 4096 weights, 2*128 = 256 activation = 4352 total
        // L1: 32*64*4 = 8192 weights, 2*128 = 256 activation = 8448 total
        // L2: 32*128*4 = 16384 weights, 2*128 = 256 activation = 16640 total
        // L3: 32*32*4 = 4096 weights, 2*128 = 256 activation = 4352 total
        //
        // Group 0: L0+L1+L2 weights = 28672, max act = 256, total = 28928 (fits in 50000)
        // Adding L3: 28672+4096 = 32768 + 256 = 33024 (fits)
        // All four should fit in one group.
        let layers = make_layers(&[(32, 32), (32, 64), (32, 128), (32, 32)]);
        let graph = make_graph(layers);
        let plan = GreedyGrouping::new()
            .plan(&graph, MemoryBudget::from_bytes(50000))
            .unwrap();

        assert_eq!(plan.num_groups(), 1);
    }

    #[test]
    fn test_greedy_budget_too_small() {
        let graph = uniform_graph(2, 256);
        let result = GreedyGrouping::new().plan(&graph, MemoryBudget::from_bytes(100));
        assert!(matches!(result, Err(PlannerError::BudgetTooSmall { .. })));
    }

    #[test]
    fn test_greedy_validates() {
        let graph = uniform_graph(10, 64);
        let plan = GreedyGrouping::new()
            .plan(&graph, MemoryBudget::from_mb(1))
            .unwrap();
        plan.validate().unwrap();
    }

    #[test]
    fn test_greedy_groups_are_within_budget() {
        let graph = uniform_graph(12, 128);
        let budget = MemoryBudget::from_bytes(200_000);
        let plan = GreedyGrouping::new().plan(&graph, budget).unwrap();

        for group in &plan.groups {
            assert!(
                group.estimated_memory_bytes <= budget.as_bytes(),
                "group {} exceeds budget: {} > {}",
                group.group_index,
                group.estimated_memory_bytes,
                budget.as_bytes(),
            );
        }
    }

    #[test]
    fn test_estimate_group_memory() {
        let layers = make_layers(&[(32, 32), (32, 64)]);
        let refs: Vec<&LayerDef> = layers.iter().collect();
        let mem = estimate_group_memory(&refs);
        // Weights: 32*32*4 + 32*64*4 = 4096 + 8192 = 12288
        // Max activation: max(2*32*4, 2*32*4) = 256
        assert_eq!(mem, 12288 + 256);
    }
}
