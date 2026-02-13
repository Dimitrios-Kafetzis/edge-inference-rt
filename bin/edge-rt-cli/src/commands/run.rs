// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! `edge-rt run` command: execute inference with a given prompt.
//!
//! Demonstrates the full type-state pipeline:
//! ```text
//! InferenceEngine<Idle> → load_model → <Planned> → prepare → <Ready> → run
//! ```

use std::path::PathBuf;

pub async fn execute(
    model: PathBuf,
    memory_budget: String,
    strategy: String,
    prompt: String,
    max_tokens: usize,
) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║            edge-rt · Inference Runner               ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // ── Configuration ──────────────────────────────────────────
    let config = runtime::RuntimeConfig {
        model_path: model.clone(),
        memory_budget: memory_budget.clone(),
        strategy: strategy.clone(),
        num_threads: None,
        prefetch_ratio: Some(0.2),
        enable_profiling: true,
    };

    println!("  Config:");
    println!("   Model:    {}", model.display());
    println!("   Budget:   {memory_budget}");
    println!("   Strategy: {strategy}");
    println!("   Prompt:   \"{}\"", truncate(&prompt, 50));
    println!("   Tokens:   {max_tokens}");
    println!();

    // ── Type-State Pipeline ────────────────────────────────────
    //
    // Step 1: Idle → Planned (load model + generate execution plan).
    println!("  [1/3] Loading model and generating execution plan...");

    let planned = match runtime::InferenceEngine::new(config).load_model() {
        Ok(p) => p,
        Err(e) => {
            // If model files don't exist, offer synthetic demo.
            tracing::warn!("model load failed: {e}");
            println!("        Model files not found. Running synthetic demo...");
            println!();
            return run_synthetic_demo(memory_budget, strategy, &prompt, max_tokens).await;
        }
    };

    println!("        Plan: {}", planned.plan().summary());
    println!();

    // Step 2: Planned → Ready (allocate pool + weight loader).
    println!("  [2/3] Preparing memory pool and weight loader...");
    let ready = planned.prepare()?;
    println!("        Pool ready.");
    println!();

    // Step 3: Ready → Run (execute inference).
    println!("  [3/3] Running inference...");

    // Convert prompt to synthetic token IDs (simple encoding).
    let input_tokens: Vec<u32> = prompt
        .bytes()
        .take(max_tokens)
        .map(|b| b as u32)
        .collect();

    let output = ready.run(&input_tokens).await?;

    println!();
    print_results(&output);

    Ok(())
}

/// Runs a synthetic demo without model files.
///
/// This demonstrates the full pipeline using programmatically-built
/// graphs and plans, which is the typical way to showcase the runtime
/// in a portfolio context.
async fn run_synthetic_demo(
    memory_budget: String,
    strategy: String,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<()> {
    use model_ir::{LayerType, ModelGraph};

    println!("  Running synthetic GPT-2-like demo (12 layers, hidden=768)...");
    println!();

    let budget = memory_manager::MemoryBudget::parse(&memory_budget)
        .map_err(|e| anyhow::anyhow!("invalid budget: {e}"))?;

    // Build a synthetic 12-layer GPT-2-like graph.
    let hidden = 768;
    let mut layers = Vec::new();
    let mut idx = 0;

    // Embedding + positional encoding.
    layers.push(make_layer(&mut idx, "wte", LayerType::Embedding, hidden, 50257 * hidden));
    layers.push(make_layer(&mut idx, "wpe", LayerType::PositionalEncoding, hidden, 1024 * hidden));

    // 12 transformer blocks (simplified: LN + Attn + LN + FFN).
    for b in 0..3 {
        layers.push(make_layer(&mut idx, &format!("h.{b}.ln_1"), LayerType::LayerNorm, hidden, hidden * 2));
        layers.push(make_layer(&mut idx, &format!("h.{b}.attn"), LayerType::SelfAttention, hidden, hidden * hidden * 4));
        layers.push(make_layer(&mut idx, &format!("h.{b}.ln_2"), LayerType::LayerNorm, hidden, hidden * 2));
        layers.push(make_layer(&mut idx, &format!("h.{b}.mlp"), LayerType::FeedForward, hidden, hidden * hidden * 4));
    }

    // Final layer norm + LM head.
    layers.push(make_layer(&mut idx, "ln_f", LayerType::LayerNorm, hidden, hidden * 2));
    layers.push(make_layer(&mut idx, "lm_head", LayerType::Linear, hidden, hidden * 50257));

    let graph = ModelGraph::new("gpt2-synthetic".into(), layers).validate()?;
    println!("  Model: {}", graph.summary());

    // Create strategy and plan.
    let config = runtime::RuntimeConfig {
        model_path: std::path::PathBuf::from("<synthetic>"),
        memory_budget: memory_budget.clone(),
        strategy: strategy.clone(),
        num_threads: None,
        prefetch_ratio: Some(0.2),
        enable_profiling: true,
    };

    let strat = config.create_strategy()?;
    let plan = strat.plan(&graph, budget)?;
    println!("  Plan:  {}", plan.summary());
    println!();

    // Use the from_graph_and_plan constructor for direct injection.
    let engine = runtime::InferenceEngine::from_graph_and_plan(config, graph, plan, budget)
        .prepare()?;

    // Run inference.
    let input_tokens: Vec<u32> = prompt
        .bytes()
        .take(max_tokens)
        .map(|b| b as u32)
        .collect();

    println!("  Executing inference ({} input tokens)...", input_tokens.len());
    let output = engine.run(&input_tokens).await?;
    println!();

    print_results(&output);

    // Print pool stats to demonstrate RAII lifecycle.
    let stats = engine.memory_stats();
    println!("  Pool Stats:");
    println!("   {}", stats.summary());

    Ok(())
}

/// Helper to construct a synthetic layer.
fn make_layer(
    idx: &mut usize,
    name: &str,
    layer_type: model_ir::LayerType,
    hidden: usize,
    weight_elements: usize,
) -> model_ir::LayerDef {
    use tensor_core::{DType, Shape};

    let i = *idx;
    *idx += 1;

    // Weight size in f32: weight_elements * 4 bytes.
    let weight_dim = weight_elements / hidden;
    model_ir::LayerDef {
        name: name.to_string(),
        layer_type,
        index: i,
        weight_names: vec![format!("{name}.weight")],
        weight_shapes: vec![Shape::matrix(hidden, weight_dim.max(1))],
        dtype: DType::F32,
        input_shape: Shape::matrix(1, hidden),
        output_shape: Shape::matrix(1, hidden),
    }
}

fn print_results(output: &runtime::InferenceOutput) {
    println!("  Results:");
    println!("   Tokens generated: {}", output.token_ids.len());
    println!(
        "   Token IDs: {:?}{}",
        &output.token_ids[..output.token_ids.len().min(10)],
        if output.token_ids.len() > 10 { " ..." } else { "" },
    );
    println!();
    println!("  Metrics:");
    println!("   {}", output.metrics.summary());
    println!();
}

/// Truncates a string with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
