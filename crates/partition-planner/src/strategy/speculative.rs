// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Speculative prefetch partitioning strategy.
//!
//! Extends [`crate::GreedyGrouping`] by reserving a configurable fraction of
//! the memory budget as headroom for prefetching the **next** group's
//! weights while the current group is still executing.
//!
//! # How It Works
//!
//! ```text
//! effective_budget = budget × (1 − prefetch_ratio)
//! ```
//!
//! Layers are packed greedily using the reduced `effective_budget`.
//! The remaining headroom is available to the runtime for overlapping
//! weight I/O with computation, reducing wall-clock latency.
//!
//! # Trade-offs
//! - **Fewer layers per group** than pure greedy (tighter effective budget).
//! - **Lower latency** when the runtime exploits the headroom.
//! - Requires the runtime to implement async prefetch scheduling.
//!
//! # When to use
//! - Memory budget has comfortable headroom beyond the largest layer.
//! - I/O latency (loading weights from storage/mmap) is significant.
//! - The runtime supports async weight prefetching.

use crate::plan::PlanBuilder;
use crate::strategy::PartitionStrategy;
use crate::{ExecutionPlan, PlannerError};
use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, LayerDef, ModelGraph};

/// Default prefetch headroom fraction: reserve 20% of budget.
const DEFAULT_PREFETCH_RATIO: f64 = 0.20;

/// Greedy grouping with speculative prefetch headroom.
#[derive(Debug, Clone)]
pub struct SpeculativePrefetch {
    /// Fraction of the budget to reserve for prefetching (0.0–0.5).
    prefetch_ratio: f64,
}

impl Default for SpeculativePrefetch {
    fn default() -> Self {
        Self::new(DEFAULT_PREFETCH_RATIO)
    }
}

impl SpeculativePrefetch {
    /// Creates a new speculative prefetch strategy.
    ///
    /// # Arguments
    /// * `prefetch_ratio` — fraction of the budget reserved for prefetching.
    ///   Clamped to `[0.0, 0.5]`. Typical values: 0.15–0.25.
    pub fn new(prefetch_ratio: f64) -> Self {
        Self {
            prefetch_ratio: prefetch_ratio.clamp(0.0, 0.5),
        }
    }

    /// Returns the prefetch ratio.
    pub fn prefetch_ratio(&self) -> f64 {
        self.prefetch_ratio
    }

    /// Returns the effective budget after reserving prefetch headroom.
    pub fn effective_budget(&self, budget: MemoryBudget) -> usize {
        let full = budget.as_bytes() as f64;
        (full * (1.0 - self.prefetch_ratio)) as usize
    }

    /// Returns the prefetch headroom in bytes.
    pub fn headroom_bytes(&self, budget: MemoryBudget) -> usize {
        budget.as_bytes() - self.effective_budget(budget)
    }
}

impl PartitionStrategy for SpeculativePrefetch {
    fn name(&self) -> &str {
        "speculative-prefetch"
    }

    fn plan(
        &self,
        graph: &ModelGraph<Validated>,
        budget: MemoryBudget,
    ) -> Result<ExecutionPlan, PlannerError> {
        if graph.num_layers() == 0 {
            return Err(PlannerError::EmptyGraph);
        }

        let effective_bytes = self.effective_budget(budget);
        let budget_bytes = budget.as_bytes();
        let layers: Vec<&LayerDef> = graph.iter_layers().collect();

        // Pre-check: every single layer must fit in the effective budget.
        for layer in &layers {
            let solo_mem = layer.estimated_total_bytes();
            if solo_mem > effective_bytes {
                // Try with full budget as fallback (no prefetch for this model).
                if solo_mem > budget_bytes {
                    return Err(PlannerError::BudgetTooSmall {
                        layer_bytes: solo_mem,
                        budget_bytes,
                    });
                }
                // Fall back to greedy without prefetch headroom.
                tracing::warn!(
                    "layer '{}' ({} bytes) exceeds effective budget ({} bytes); \
                     disabling prefetch headroom for this plan",
                    layer.name,
                    solo_mem,
                    effective_bytes,
                );
                return plan_greedy(&layers, budget_bytes, self.name());
            }
        }

        plan_greedy(&layers, effective_bytes, self.name())
    }
}

/// Core greedy grouping logic shared between strategies.
///
/// Uses `limit_bytes` as the per-group ceiling (which may be the full
/// budget or the effective budget after prefetch reservation).
fn plan_greedy(
    layers: &[&LayerDef],
    limit_bytes: usize,
    strategy_name: &str,
) -> Result<ExecutionPlan, PlannerError> {
    // We store the plan against the full budget for validation,
    // but pack against limit_bytes.
    let mut builder = PlanBuilder::new(strategy_name, limit_bytes);
    let mut i = 0;

    while i < layers.len() {
        let mut group_weight_bytes = layers[i].estimated_weight_bytes();
        let mut group_max_activation = layers[i].estimated_activation_bytes();
        let mut group_indices = vec![layers[i].index];

        let mut j = i + 1;
        while j < layers.len() {
            let next_weight = layers[j].estimated_weight_bytes();
            let next_activation = layers[j].estimated_activation_bytes();

            let candidate_weight = group_weight_bytes + next_weight;
            let candidate_activation = group_max_activation.max(next_activation);
            let candidate_total = candidate_weight + candidate_activation;

            if candidate_total > limit_bytes {
                break;
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

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::{LayerDef, LayerType};
    use tensor_core::{DType, Shape};

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
        ModelGraph::new("test".into(), layers).validate().unwrap()
    }

    #[test]
    fn test_prefetch_ratio_clamped() {
        let s = SpeculativePrefetch::new(0.8);
        assert!((s.prefetch_ratio() - 0.5).abs() < 1e-9);

        let s = SpeculativePrefetch::new(-0.1);
        assert!((s.prefetch_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_effective_budget() {
        let s = SpeculativePrefetch::new(0.20);
        let budget = MemoryBudget::from_bytes(10000);
        assert_eq!(s.effective_budget(budget), 8000);
        assert_eq!(s.headroom_bytes(budget), 2000);
    }

    #[test]
    fn test_fewer_layers_per_group_than_greedy() {
        // With prefetch headroom, groups should be equal or smaller than greedy.
        let graph = uniform_graph(12, 64);
        let budget = MemoryBudget::from_bytes(100_000);

        let greedy = crate::strategy::greedy::GreedyGrouping::new()
            .plan(&graph, budget)
            .unwrap();
        let spec = SpeculativePrefetch::new(0.25)
            .plan(&graph, budget)
            .unwrap();

        assert!(
            spec.num_groups() >= greedy.num_groups(),
            "speculative ({}) should have ≥ groups than greedy ({})",
            spec.num_groups(),
            greedy.num_groups(),
        );
    }

    #[test]
    fn test_speculative_all_fit() {
        // Tiny layers, huge budget → should all be in one group even with headroom.
        let graph = uniform_graph(4, 16);
        let plan = SpeculativePrefetch::new(0.20)
            .plan(&graph, MemoryBudget::from_mb(10))
            .unwrap();
        assert_eq!(plan.num_groups(), 1);
    }

    #[test]
    fn test_speculative_budget_too_small() {
        let graph = uniform_graph(2, 256);
        let result =
            SpeculativePrefetch::new(0.20).plan(&graph, MemoryBudget::from_bytes(100));
        assert!(matches!(result, Err(PlannerError::BudgetTooSmall { .. })));
    }

    #[test]
    fn test_speculative_fallback_no_headroom() {
        // Create a layer that fits in full budget but not effective budget.
        // Budget = 20000, effective = 16000, layer = 17000.
        // Should fall back to full budget (warning).
        let layers = vec![LayerDef {
            name: "big_layer".into(),
            layer_type: LayerType::Linear,
            index: 0,
            weight_names: vec!["w".into()],
            weight_shapes: vec![Shape::matrix(64, 64)], // 64*64*4 = 16384 weight
            dtype: DType::F32,
            input_shape: Shape::matrix(1, 64),  // 256
            output_shape: Shape::matrix(1, 64), // 256
            // Total: 16384 + 512 = 16896
        }];
        let graph = ModelGraph::new("test".into(), layers)
            .validate()
            .unwrap();

        // Effective budget with 0.20: 20000 * 0.8 = 16000 < 16896.
        // Full budget: 20000 > 16896.
        let plan = SpeculativePrefetch::new(0.20)
            .plan(&graph, MemoryBudget::from_bytes(20000))
            .unwrap();
        assert_eq!(plan.num_groups(), 1);
    }

    #[test]
    fn test_speculative_validates() {
        let graph = uniform_graph(10, 64);
        let plan = SpeculativePrefetch::default()
            .plan(&graph, MemoryBudget::from_mb(1))
            .unwrap();
        plan.validate().unwrap();
    }

    #[test]
    fn test_speculative_groups_within_effective_budget() {
        let graph = uniform_graph(8, 128);
        let budget = MemoryBudget::from_bytes(200_000);
        let s = SpeculativePrefetch::new(0.20);
        let effective = s.effective_budget(budget);
        let plan = s.plan(&graph, budget).unwrap();

        for group in &plan.groups {
            assert!(
                group.estimated_memory_bytes <= effective,
                "group {} exceeds effective budget: {} > {}",
                group.group_index,
                group.estimated_memory_bytes,
                effective,
            );
        }
    }
}
