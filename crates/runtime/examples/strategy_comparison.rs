// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Example: Compare partitioning strategies on a synthetic transformer.
//!
//! Demonstrates the core value proposition of the runtime: different
//! strategies produce different groupings with different memory/latency
//! trade-offs, all verified at compile time via the type-state pipeline.
//!
//! ```bash
//! cargo run -p runtime --example strategy_comparison
//! ```

use memory_manager::MemoryBudget;
use model_ir::{graph::Validated, LayerDef, LayerType, ModelGraph};
use partition_planner::{GreedyGrouping, PartitionStrategy, Sequential, SpeculativePrefetch};
use runtime::{InferenceEngine, RuntimeConfig};
use tensor_core::{DType, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing.
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    // Build a synthetic 8-block transformer (hidden=256).
    let graph = build_graph("gpt2-like", 8, 256);
    println!("Model: {}\n", graph.summary());

    // Define strategies and budgets to compare.
    let strategies: Vec<Box<dyn PartitionStrategy>> = vec![
        Box::new(Sequential::new()),
        Box::new(GreedyGrouping::new()),
        Box::new(SpeculativePrefetch::new(0.20)),
    ];

    let budgets = [
        MemoryBudget::from_mb(10),
        MemoryBudget::from_mb(50),
        MemoryBudget::from_mb(200),
    ];

    // Compare.
    println!(
        "{:<24} {:>10} {:>8} {:>12} {:>12}",
        "Strategy", "Budget", "Groups", "Peak MB", "Layers/Group",
    );
    println!("{}", "-".repeat(70));

    for strategy in &strategies {
        for budget in &budgets {
            match strategy.plan(&graph, *budget) {
                Ok(plan) => {
                    let avg = plan.total_layers() as f64 / plan.num_groups() as f64;
                    let peak_mb = plan.peak_memory_bytes as f64 / (1024.0 * 1024.0);
                    println!(
                        "{:<24} {:>10} {:>8} {:>10.2} MB {:>10.1}",
                        strategy.name(),
                        format!("{budget}"),
                        plan.num_groups(),
                        peak_mb,
                        avg,
                    );
                }
                Err(e) => {
                    println!(
                        "{:<24} {:>10} {:>8}",
                        strategy.name(),
                        format!("{budget}"),
                        format!("FAIL: {e}"),
                    );
                }
            }
        }
    }

    // Run inference with the best strategy at 50 MB.
    println!("\n--- Running inference with greedy-grouping @ 50 MB ---\n");
    let budget = MemoryBudget::from_mb(50);
    let plan = GreedyGrouping::new().plan(&graph, budget)?;
    println!("Plan: {}\n", plan.summary());

    let config = RuntimeConfig {
        model_path: "<synthetic>".into(),
        memory_budget: "50M".into(),
        strategy: "greedy-grouping".into(),
        enable_profiling: true,
        ..Default::default()
    };

    let engine = InferenceEngine::from_graph_and_plan(config, graph, plan, budget)
        .prepare()?;

    // Run using tokio.
    let rt = tokio::runtime::Runtime::new()?;
    let output = rt.block_on(engine.run(&[1, 2, 3, 4, 5]))?;

    println!("Tokens: {:?}", output.token_ids);
    println!("Metrics: {}", output.metrics.summary());

    // Show pool stats.
    let stats = engine.memory_stats();
    println!("Pool: {}", stats.summary());

    Ok(())
}

fn build_graph(name: &str, blocks: usize, hidden: usize) -> ModelGraph<Validated> {
    let mut layers = Vec::new();
    let mut idx = 0;

    layers.push(mk(&mut idx, "wte", LayerType::Embedding, hidden, hidden));
    layers.push(mk(&mut idx, "wpe", LayerType::PositionalEncoding, hidden, hidden));

    for b in 0..blocks {
        layers.push(mk(&mut idx, &format!("h.{b}.ln_1"), LayerType::LayerNorm, hidden, hidden));
        layers.push(mk(&mut idx, &format!("h.{b}.attn"), LayerType::SelfAttention, hidden, hidden * 4));
        layers.push(mk(&mut idx, &format!("h.{b}.ln_2"), LayerType::LayerNorm, hidden, hidden));
        layers.push(mk(&mut idx, &format!("h.{b}.mlp"), LayerType::FeedForward, hidden, hidden * 4));
    }

    layers.push(mk(&mut idx, "ln_f", LayerType::LayerNorm, hidden, hidden));
    layers.push(mk(&mut idx, "lm_head", LayerType::Linear, hidden, hidden));

    ModelGraph::new(name.into(), layers).validate().unwrap()
}

fn mk(idx: &mut usize, name: &str, lt: LayerType, h: usize, w: usize) -> LayerDef {
    let i = *idx;
    *idx += 1;
    LayerDef {
        name: name.into(),
        layer_type: lt,
        index: i,
        weight_names: vec![format!("{name}.w")],
        weight_shapes: vec![Shape::matrix(h, w)],
        dtype: DType::F32,
        input_shape: Shape::matrix(1, h),
        output_shape: Shape::matrix(1, h),
    }
}
