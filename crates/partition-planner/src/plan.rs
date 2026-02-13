// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Execution plan: the output of the partition planner.
//!
//! A plan is a sequence of [`LayerGroup`]s. Each group is loaded into
//! memory, executed, and then its weight buffers are released before
//! the next group is loaded. The plan is the contract between the
//! planner and the runtime.

use crate::PlannerError;

/// A group of consecutive layers that will be executed together.
///
/// All weights for layers in a group are loaded into memory simultaneously,
/// and the group is executed as a unit before its memory is released.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerGroup {
    /// Index of this group in the execution order.
    pub group_index: usize,
    /// Indices of the layers in this group (into the model graph).
    pub layer_indices: Vec<usize>,
    /// Estimated peak memory for this group (weights + activations) in bytes.
    pub estimated_memory_bytes: usize,
}

impl LayerGroup {
    /// Returns the number of layers in this group.
    pub fn num_layers(&self) -> usize {
        self.layer_indices.len()
    }

    /// Returns `true` if this group is a single layer.
    pub fn is_single_layer(&self) -> bool {
        self.layer_indices.len() == 1
    }

    /// Returns the first layer index in this group.
    pub fn first_layer(&self) -> usize {
        self.layer_indices[0]
    }

    /// Returns the last layer index in this group.
    pub fn last_layer(&self) -> usize {
        *self.layer_indices.last().expect("group is non-empty")
    }
}

/// The complete execution plan produced by a [`crate::PartitionStrategy`].
///
/// Contains the ordered list of layer groups and metadata about the plan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecutionPlan {
    /// Strategy name that produced this plan.
    pub strategy_name: String,
    /// Ordered list of layer groups.
    pub groups: Vec<LayerGroup>,
    /// The memory budget used for planning.
    pub budget_bytes: usize,
    /// Peak memory across any single group.
    pub peak_memory_bytes: usize,
}

impl ExecutionPlan {
    /// Returns the total number of groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Returns the total number of layers across all groups.
    pub fn total_layers(&self) -> usize {
        self.groups.iter().map(|g| g.num_layers()).sum()
    }

    /// Validates the execution plan.
    ///
    /// Checks:
    /// - Plan is non-empty.
    /// - No group exceeds the budget.
    /// - Group indices are consecutive starting from 0.
    /// - Layer indices are strictly increasing and contiguous.
    /// - No empty groups.
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.groups.is_empty() {
            return Err(PlannerError::EmptyGraph);
        }

        let mut expected_group_idx = 0;
        let mut expected_layer_idx = 0;

        for group in &self.groups {
            // Check group index.
            if group.group_index != expected_group_idx {
                return Err(PlannerError::StrategyFailed {
                    strategy: self.strategy_name.clone(),
                    detail: format!(
                        "expected group index {expected_group_idx}, got {}",
                        group.group_index,
                    ),
                });
            }
            expected_group_idx += 1;

            // No empty groups.
            if group.layer_indices.is_empty() {
                return Err(PlannerError::StrategyFailed {
                    strategy: self.strategy_name.clone(),
                    detail: format!("group {} is empty", group.group_index),
                });
            }

            // Layer indices must be contiguous with previous group.
            for &li in &group.layer_indices {
                if li != expected_layer_idx {
                    return Err(PlannerError::StrategyFailed {
                        strategy: self.strategy_name.clone(),
                        detail: format!(
                            "expected layer index {expected_layer_idx}, got {li} in group {}",
                            group.group_index,
                        ),
                    });
                }
                expected_layer_idx += 1;
            }

            // Budget enforcement.
            if group.estimated_memory_bytes > self.budget_bytes {
                return Err(PlannerError::StrategyFailed {
                    strategy: self.strategy_name.clone(),
                    detail: format!(
                        "group {} requires {} bytes but budget is {} bytes",
                        group.group_index, group.estimated_memory_bytes, self.budget_bytes,
                    ),
                });
            }
        }

        Ok(())
    }

    /// Returns a human-readable summary of the plan.
    pub fn summary(&self) -> String {
        let peak_mb = self.peak_memory_bytes as f64 / (1024.0 * 1024.0);
        let budget_mb = self.budget_bytes as f64 / (1024.0 * 1024.0);
        let layers_per_group: Vec<usize> = self.groups.iter().map(|g| g.num_layers()).collect();
        let avg_layers = if self.groups.is_empty() {
            0.0
        } else {
            self.total_layers() as f64 / self.groups.len() as f64
        };

        format!(
            "Plan '{}': {} groups, {} layers total, \
             avg {:.1} layers/group, peak {:.2}/{:.1} MB ({:.0}% budget), \
             group sizes: {:?}",
            self.strategy_name,
            self.num_groups(),
            self.total_layers(),
            avg_layers,
            peak_mb,
            budget_mb,
            (peak_mb / budget_mb) * 100.0,
            layers_per_group,
        )
    }
}

/// Builder helper for constructing an `ExecutionPlan` incrementally.
///
/// Used internally by strategy implementations.
pub(crate) struct PlanBuilder {
    strategy_name: String,
    budget_bytes: usize,
    groups: Vec<LayerGroup>,
    peak_memory_bytes: usize,
}

impl PlanBuilder {
    /// Creates a new builder.
    pub fn new(strategy_name: &str, budget_bytes: usize) -> Self {
        Self {
            strategy_name: strategy_name.to_string(),
            budget_bytes,
            groups: Vec::new(),
            peak_memory_bytes: 0,
        }
    }

    /// Adds a group of layer indices with a known memory estimate.
    pub fn add_group(&mut self, layer_indices: Vec<usize>, estimated_memory_bytes: usize) {
        let group_index = self.groups.len();
        if estimated_memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = estimated_memory_bytes;
        }
        self.groups.push(LayerGroup {
            group_index,
            layer_indices,
            estimated_memory_bytes,
        });
    }

    /// Consumes the builder and returns the finished plan.
    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            strategy_name: self.strategy_name,
            groups: self.groups,
            budget_bytes: self.budget_bytes,
            peak_memory_bytes: self.peak_memory_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> ExecutionPlan {
        ExecutionPlan {
            strategy_name: "test".into(),
            groups: vec![
                LayerGroup {
                    group_index: 0,
                    layer_indices: vec![0, 1, 2],
                    estimated_memory_bytes: 1000,
                },
                LayerGroup {
                    group_index: 1,
                    layer_indices: vec![3, 4],
                    estimated_memory_bytes: 800,
                },
                LayerGroup {
                    group_index: 2,
                    layer_indices: vec![5],
                    estimated_memory_bytes: 500,
                },
            ],
            budget_bytes: 2000,
            peak_memory_bytes: 1000,
        }
    }

    #[test]
    fn test_validate_ok() {
        let plan = sample_plan();
        plan.validate().unwrap();
    }

    #[test]
    fn test_num_groups() {
        assert_eq!(sample_plan().num_groups(), 3);
    }

    #[test]
    fn test_total_layers() {
        assert_eq!(sample_plan().total_layers(), 6);
    }

    #[test]
    fn test_validate_empty() {
        let plan = ExecutionPlan {
            strategy_name: "empty".into(),
            groups: vec![],
            budget_bytes: 1000,
            peak_memory_bytes: 0,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_validate_exceeds_budget() {
        let plan = ExecutionPlan {
            strategy_name: "big".into(),
            groups: vec![LayerGroup {
                group_index: 0,
                layer_indices: vec![0],
                estimated_memory_bytes: 5000,
            }],
            budget_bytes: 1000,
            peak_memory_bytes: 5000,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_validate_non_contiguous_layers() {
        let plan = ExecutionPlan {
            strategy_name: "gap".into(),
            groups: vec![
                LayerGroup {
                    group_index: 0,
                    layer_indices: vec![0, 1],
                    estimated_memory_bytes: 500,
                },
                LayerGroup {
                    group_index: 1,
                    layer_indices: vec![3], // Skips layer 2!
                    estimated_memory_bytes: 500,
                },
            ],
            budget_bytes: 1000,
            peak_memory_bytes: 500,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_validate_bad_group_index() {
        let plan = ExecutionPlan {
            strategy_name: "bad_idx".into(),
            groups: vec![
                LayerGroup {
                    group_index: 0,
                    layer_indices: vec![0],
                    estimated_memory_bytes: 500,
                },
                LayerGroup {
                    group_index: 5, // Should be 1.
                    layer_indices: vec![1],
                    estimated_memory_bytes: 500,
                },
            ],
            budget_bytes: 1000,
            peak_memory_bytes: 500,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_summary() {
        let s = sample_plan().summary();
        assert!(s.contains("test"));
        assert!(s.contains("3 groups"));
        assert!(s.contains("6 layers"));
    }

    #[test]
    fn test_plan_builder() {
        let mut b = PlanBuilder::new("builder_test", 2000);
        b.add_group(vec![0, 1], 800);
        b.add_group(vec![2, 3, 4], 1200);
        let plan = b.build();

        assert_eq!(plan.num_groups(), 2);
        assert_eq!(plan.total_layers(), 5);
        assert_eq!(plan.peak_memory_bytes, 1200);
        plan.validate().unwrap();
    }

    #[test]
    fn test_layer_group_helpers() {
        let g = LayerGroup {
            group_index: 0,
            layer_indices: vec![3, 4, 5],
            estimated_memory_bytes: 1000,
        };
        assert_eq!(g.num_layers(), 3);
        assert!(!g.is_single_layer());
        assert_eq!(g.first_layer(), 3);
        assert_eq!(g.last_layer(), 5);
    }
}
