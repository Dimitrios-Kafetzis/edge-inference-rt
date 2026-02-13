// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Integration tests: end-to-end inference pipeline.
//!
//! These tests exercise the complete flow from graph construction →
//! planning → pool allocation → execution, proving that all six crates
//! compose correctly and that the type-state transitions work end-to-end.

use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, LayerDef, LayerType, ModelGraph};
use partition_planner::{
    GreedyGrouping, PartitionStrategy, Sequential, SpeculativePrefetch,
};
use runtime::{InferenceEngine, RuntimeConfig};
use tensor_core::{DType, Shape};

// ── Helpers ────────────────────────────────────────────────────

/// Builds a synthetic transformer-like graph.
fn synthetic_graph(
    name: &str,
    num_blocks: usize,
    hidden: usize,
) -> ModelGraph<Validated> {
    let mut layers = Vec::new();
    let mut idx = 0;

    // Embedding.
    layers.push(layer(&mut idx, "wte", LayerType::Embedding, hidden, hidden));
    layers.push(layer(&mut idx, "wpe", LayerType::PositionalEncoding, hidden, hidden));

    // Transformer blocks.
    for b in 0..num_blocks {
        layers.push(layer(
            &mut idx,
            &format!("h.{b}.ln_1"),
            LayerType::LayerNorm,
            hidden,
            hidden,
        ));
        layers.push(layer(
            &mut idx,
            &format!("h.{b}.attn"),
            LayerType::SelfAttention,
            hidden,
            hidden * 4,
        ));
        layers.push(layer(
            &mut idx,
            &format!("h.{b}.ln_2"),
            LayerType::LayerNorm,
            hidden,
            hidden,
        ));
        layers.push(layer(
            &mut idx,
            &format!("h.{b}.mlp"),
            LayerType::FeedForward,
            hidden,
            hidden * 4,
        ));
    }

    // Final LN + head.
    layers.push(layer(&mut idx, "ln_f", LayerType::LayerNorm, hidden, hidden));
    layers.push(layer(&mut idx, "lm_head", LayerType::Linear, hidden, hidden));

    ModelGraph::new(name.into(), layers).validate().unwrap()
}

fn layer(
    idx: &mut usize,
    name: &str,
    layer_type: LayerType,
    hidden: usize,
    weight_cols: usize,
) -> LayerDef {
    let i = *idx;
    *idx += 1;
    LayerDef {
        name: name.to_string(),
        layer_type,
        index: i,
        weight_names: vec![format!("{name}.weight")],
        weight_shapes: vec![Shape::matrix(hidden, weight_cols)],
        dtype: DType::F32,
        input_shape: Shape::matrix(1, hidden),
        output_shape: Shape::matrix(1, hidden),
    }
}

fn config() -> RuntimeConfig {
    RuntimeConfig {
        model_path: std::path::PathBuf::from("<synthetic>"),
        memory_budget: "512M".into(),
        strategy: "greedy-grouping".into(),
        num_threads: Some(2),
        prefetch_ratio: Some(0.2),
        enable_profiling: true,
    }
}

// ── Full Pipeline Tests ────────────────────────────────────────

#[tokio::test]
async fn test_end_to_end_sequential() {
    let graph = synthetic_graph("seq-test", 4, 64);
    let budget = MemoryBudget::from_mb(100);
    let plan = Sequential::new().plan(&graph, budget).unwrap();

    // Sequential: each layer is its own group.
    assert_eq!(plan.num_groups(), graph.num_layers());

    let engine = InferenceEngine::from_graph_and_plan(config(), graph, plan, budget)
        .prepare()
        .unwrap();

    let output = engine.run(&[1, 2, 3, 4, 5]).await.unwrap();

    assert_eq!(output.token_ids.len(), 5);
    assert!(output.metrics.total_duration.as_nanos() > 0);
    assert_eq!(output.metrics.layer_metrics.len(), 20); // 2 + 4*4 + 2
    assert_eq!(output.metrics.tokens_generated, 5);
}

#[tokio::test]
async fn test_end_to_end_greedy_grouping() {
    let graph = synthetic_graph("greedy-test", 4, 64);
    let budget = MemoryBudget::from_mb(100);
    let plan = GreedyGrouping::new().plan(&graph, budget).unwrap();

    // Greedy should produce fewer groups than sequential.
    assert!(plan.num_groups() <= graph.num_layers());
    assert_eq!(plan.total_layers(), graph.num_layers());

    let engine = InferenceEngine::from_graph_and_plan(config(), graph, plan, budget)
        .prepare()
        .unwrap();

    let output = engine.run(&[10, 20, 30]).await.unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

#[tokio::test]
async fn test_end_to_end_speculative_prefetch() {
    let graph = synthetic_graph("spec-test", 4, 64);
    let budget = MemoryBudget::from_mb(100);
    let plan = SpeculativePrefetch::default().plan(&graph, budget).unwrap();

    // Speculative should produce >= groups compared to pure greedy
    // (tighter effective budget).
    let greedy_plan = GreedyGrouping::new().plan(&graph, budget).unwrap();
    assert!(plan.num_groups() >= greedy_plan.num_groups());

    let engine = InferenceEngine::from_graph_and_plan(config(), graph, plan, budget)
        .prepare()
        .unwrap();

    let output = engine.run(&[42]).await.unwrap();
    assert_eq!(output.token_ids, vec![43]); // wrapping_add(1)
}

// ── Property: Budget Never Exceeded ────────────────────────────

#[test]
fn test_budget_never_exceeded() {
    // Test a range of model sizes and budgets.
    let configs: Vec<(usize, usize, usize)> = vec![
        // (num_blocks, hidden, budget_mb)
        (2, 32, 10),
        (4, 64, 50),
        (6, 128, 100),
        (8, 256, 500),
        (2, 512, 1000),
    ];

    let strategies: Vec<Box<dyn PartitionStrategy>> = vec![
        Box::new(Sequential::new()),
        Box::new(GreedyGrouping::new()),
        Box::new(SpeculativePrefetch::new(0.20)),
    ];

    for (blocks, hidden, budget_mb) in &configs {
        let graph = synthetic_graph("prop-test", *blocks, *hidden);
        let budget = MemoryBudget::from_mb(*budget_mb);

        for strategy in &strategies {
            let plan = strategy.plan(&graph, budget);
            if let Ok(plan) = plan {
                // Verify plan passes its own validation.
                plan.validate().unwrap();

                // Verify no group exceeds the budget.
                for group in &plan.groups {
                    assert!(
                        group.estimated_memory_bytes <= budget.as_bytes(),
                        "strategy '{}' produced group {} ({} bytes) exceeding budget ({} bytes) \
                         for model (blocks={}, hidden={})",
                        strategy.name(),
                        group.group_index,
                        group.estimated_memory_bytes,
                        budget.as_bytes(),
                        blocks,
                        hidden,
                    );
                }

                // Verify all layers are covered exactly once.
                let mut covered: Vec<usize> = plan
                    .groups
                    .iter()
                    .flat_map(|g| g.layer_indices.iter().copied())
                    .collect();
                covered.sort();
                let expected: Vec<usize> = (0..graph.num_layers()).collect();
                assert_eq!(
                    covered, expected,
                    "strategy '{}' did not cover all layers",
                    strategy.name(),
                );
            }
            // If plan fails (BudgetTooSmall), that's a valid outcome.
        }
    }
}

// ── Memory Lifecycle Tests ─────────────────────────────────────

#[tokio::test]
async fn test_pool_memory_fully_returned() {
    let graph = synthetic_graph("mem-test", 6, 128);
    let budget = MemoryBudget::from_mb(200);
    let plan = GreedyGrouping::new().plan(&graph, budget).unwrap();

    let engine = InferenceEngine::from_graph_and_plan(config(), graph, plan, budget)
        .prepare()
        .unwrap();

    // Run multiple times.
    for _ in 0..5 {
        let _ = engine.run(&[1, 2, 3]).await.unwrap();
    }

    // All allocations must have been returned.
    let stats = engine.memory_stats();
    assert_eq!(
        stats.total_allocations, stats.total_deallocations,
        "memory leak detected: {} allocs, {} deallocs",
        stats.total_allocations, stats.total_deallocations,
    );
}

#[tokio::test]
async fn test_free_list_reuse_across_runs() {
    let graph = synthetic_graph("reuse-test", 4, 64);
    let budget = MemoryBudget::from_mb(100);
    let plan = Sequential::new().plan(&graph, budget).unwrap();

    let engine = InferenceEngine::from_graph_and_plan(config(), graph, plan, budget)
        .prepare()
        .unwrap();

    // First run: all cache misses.
    let _ = engine.run(&[1]).await.unwrap();
    let stats1 = engine.memory_stats();
    let misses_after_first = stats1.cache_misses;

    // Second run: should get cache hits.
    let _ = engine.run(&[2]).await.unwrap();
    let stats2 = engine.memory_stats();

    assert!(
        stats2.cache_hits > 0,
        "expected cache hits on second run (misses after first: {})",
        misses_after_first,
    );
}

// ── Resource Monitor Integration ───────────────────────────────

#[test]
fn test_system_snapshot_works() {
    // Should succeed on any Linux system (including Docker).
    let snapshot = resource_monitor::snapshot();
    assert!(snapshot.is_ok(), "snapshot failed: {:?}", snapshot.err());

    let snap = snapshot.unwrap();
    // Memory should always be available.
    assert!(snap.memory.total_bytes > 0);
    assert!(snap.memory.available_bytes > 0);
}

// ── Auto-Plan Integration ──────────────────────────────────────

#[test]
fn test_auto_plan_healthy_system() {
    let graph = synthetic_graph("auto-test", 4, 64);
    let budget = MemoryBudget::from_mb(100);

    // On a healthy system (not constrained), auto_plan should
    // choose greedy or speculative.
    let plan = partition_planner::auto_plan(&graph, budget, false).unwrap();
    assert!(
        plan.strategy_name == "greedy-grouping"
            || plan.strategy_name == "speculative-prefetch",
        "expected greedy or speculative, got '{}'",
        plan.strategy_name,
    );
}

#[test]
fn test_auto_plan_constrained_system() {
    let graph = synthetic_graph("auto-test-c", 4, 64);
    let budget = MemoryBudget::from_mb(100);

    // Constrained → should always use sequential.
    let plan = partition_planner::auto_plan(&graph, budget, true).unwrap();
    assert_eq!(plan.strategy_name, "sequential");
}

// ── Config Roundtrip ───────────────────────────────────────────

#[test]
fn test_config_toml_roundtrip() {
    let config = RuntimeConfig::default();
    let toml = config.to_toml().unwrap();
    let back = RuntimeConfig::from_toml(&toml).unwrap();
    assert_eq!(back.strategy, config.strategy);
    assert_eq!(back.memory_budget, config.memory_budget);
}
