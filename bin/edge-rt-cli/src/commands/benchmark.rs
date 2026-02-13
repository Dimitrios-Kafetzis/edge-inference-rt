// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! `edge-rt benchmark` command: sweep across memory budgets and strategies.
//!
//! Runs inference with multiple configurations and prints a comparison table
//! showing groups, peak memory, latency, and throughput for each combination.

use std::path::PathBuf;

pub async fn execute(
    model: PathBuf,
    sweep_memory: String,
    strategies_str: String,
) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║           edge-rt · Benchmark Suite                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // Parse comma-separated memory budgets.
    let budgets: Vec<memory_manager::MemoryBudget> = sweep_memory
        .split(',')
        .map(|s| {
            memory_manager::MemoryBudget::parse(s.trim())
                .map_err(|e| anyhow::anyhow!("invalid budget '{}': {e}", s.trim()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Parse comma-separated strategy names.
    let strategy_names: Vec<&str> = strategies_str.split(',').map(|s| s.trim()).collect();

    println!(
        "  Budgets:    {:?}",
        budgets.iter().map(|b| format!("{b}")).collect::<Vec<_>>(),
    );
    println!(
        "  Strategies: {:?}",
        strategy_names,
    );
    println!();

    // Try to load real model; fall back to synthetic.
    let (graph, is_synthetic) = match model_ir::ModelLoader::load(&model) {
        Ok(g) => (g, false),
        Err(_) => {
            println!("  Model not found — using synthetic GPT-2-like (16 layers).");
            (build_synthetic_graph(), true)
        }
    };

    println!("  Model: {}", graph.summary());
    println!();

    // Input tokens for each benchmark run.
    let input_tokens: Vec<u32> = (0..32).collect();

    // ── Results Table ──────────────────────────────────────────
    println!(
        "  {:<22} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "Strategy", "Budget", "Groups", "Peak MB", "Latency", "Compute", "Tok/s",
    );
    println!("  {}", "-".repeat(80));

    let mut results: Vec<BenchResult> = Vec::new();

    for &budget in &budgets {
        for &strat_name in &strategy_names {
            let result = run_single(
                &graph,
                budget,
                strat_name,
                &input_tokens,
            )
            .await;

            match result {
                Ok(r) => {
                    println!(
                        "  {:<22} {:>8} {:>8} {:>8.2} MB {:>8.2}ms {:>8.2}ms {:>8.1}",
                        r.strategy,
                        r.budget_label,
                        r.num_groups,
                        r.peak_mb,
                        r.total_ms,
                        r.compute_ms,
                        r.tokens_per_sec,
                    );
                    results.push(r);
                }
                Err(e) => {
                    println!(
                        "  {:<22} {:>8} {:>8}     FAILED: {}",
                        strat_name,
                        format!("{budget}"),
                        "-",
                        e,
                    );
                }
            }
        }
    }

    println!();

    // ── Summary ────────────────────────────────────────────────
    if results.is_empty() {
        println!("  No successful benchmark runs.");
        return Ok(());
    }

    let fastest = results
        .iter()
        .min_by(|a, b| a.total_ms.partial_cmp(&b.total_ms).unwrap())
        .unwrap();

    let most_efficient = results
        .iter()
        .min_by(|a, b| a.peak_mb.partial_cmp(&b.peak_mb).unwrap())
        .unwrap();

    println!("  Summary:");
    println!(
        "   Fastest:          {} @ {} ({:.2}ms)",
        fastest.strategy, fastest.budget_label, fastest.total_ms,
    );
    println!(
        "   Most efficient:   {} @ {} ({:.2} MB peak)",
        most_efficient.strategy, most_efficient.budget_label, most_efficient.peak_mb,
    );
    println!();

    // ── Pool Stats ─────────────────────────────────────────────
    if is_synthetic {
        println!("  Note: running with synthetic weights (zero-filled).");
        println!("  Timing reflects pool allocation + RAII lifecycle only.");
    }
    println!();

    Ok(())
}

#[derive(Debug)]
struct BenchResult {
    strategy: String,
    budget_label: String,
    num_groups: usize,
    peak_mb: f64,
    total_ms: f64,
    compute_ms: f64,
    tokens_per_sec: f64,
}

/// Runs a single benchmark configuration.
async fn run_single(
    graph: &model_ir::ModelGraph<model_ir::graph::Validated>,
    budget: memory_manager::MemoryBudget,
    strategy_name: &str,
    input_tokens: &[u32],
) -> anyhow::Result<BenchResult> {
    let config = runtime::RuntimeConfig {
        model_path: std::path::PathBuf::from("<benchmark>"),
        memory_budget: format!("{budget}"),
        strategy: strategy_name.to_string(),
        num_threads: None,
        prefetch_ratio: Some(0.2),
        enable_profiling: true,
    };

    let strat = config.create_strategy()?;
    let plan = strat.plan(graph, budget)?;
    let num_groups = plan.num_groups();

    let engine = runtime::InferenceEngine::from_graph_and_plan(
        config,
        graph.clone(),
        plan,
        budget,
    )
    .prepare()?;

    // Warm up: run once to populate free lists.
    let _ = engine.run(input_tokens).await?;

    // Timed run.
    let output = engine.run(input_tokens).await?;
    let m = &output.metrics;

    Ok(BenchResult {
        strategy: strategy_name.to_string(),
        budget_label: format!("{budget}"),
        num_groups,
        peak_mb: m.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        total_ms: m.total_duration.as_secs_f64() * 1000.0,
        compute_ms: m.total_compute_duration.as_secs_f64() * 1000.0,
        tokens_per_sec: m.tokens_per_second(),
    })
}

/// Builds a synthetic GPT-2-like model for benchmarking.
fn build_synthetic_graph() -> model_ir::ModelGraph<model_ir::graph::Validated> {
    use model_ir::{LayerType, ModelGraph};

    let hidden = 768;
    let mut layers = Vec::new();
    let mut idx = 0;

    // Embedding + positional encoding.
    layers.push(synth_layer(&mut idx, "wte", LayerType::Embedding, hidden, hidden));
    layers.push(synth_layer(&mut idx, "wpe", LayerType::PositionalEncoding, hidden, hidden));

    // 4 transformer blocks.
    for b in 0..4 {
        layers.push(synth_layer(&mut idx, &format!("h.{b}.ln_1"), LayerType::LayerNorm, hidden, hidden));
        layers.push(synth_layer(&mut idx, &format!("h.{b}.attn"), LayerType::SelfAttention, hidden, hidden * 4));
        layers.push(synth_layer(&mut idx, &format!("h.{b}.ln_2"), LayerType::LayerNorm, hidden, hidden));
        layers.push(synth_layer(&mut idx, &format!("h.{b}.mlp"), LayerType::FeedForward, hidden, hidden * 4));
    }

    layers.push(synth_layer(&mut idx, "ln_f", LayerType::LayerNorm, hidden, hidden));
    layers.push(synth_layer(&mut idx, "lm_head", LayerType::Linear, hidden, hidden));

    ModelGraph::new("gpt2-synthetic".into(), layers)
        .validate()
        .unwrap()
}

fn synth_layer(
    idx: &mut usize,
    name: &str,
    layer_type: model_ir::LayerType,
    hidden: usize,
    weight_cols: usize,
) -> model_ir::LayerDef {
    use tensor_core::{DType, Shape};

    let i = *idx;
    *idx += 1;
    model_ir::LayerDef {
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
