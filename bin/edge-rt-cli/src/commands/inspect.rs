// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! `edge-rt inspect` command: display model structure and memory estimates.
//!
//! Loads the model manifest + SafeTensors header and prints a detailed
//! breakdown of layers, weight sizes, and activation shapes.

use std::path::PathBuf;

pub async fn execute(model: PathBuf) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║              edge-rt · Model Inspector              ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // Load and validate the model graph.
    let graph = model_ir::ModelLoader::load(&model).map_err(|e| {
        anyhow::anyhow!("failed to load model from '{}': {e}", model.display())
    })?;

    // ── Summary ────────────────────────────────────────────────
    println!("  Model: {}", graph.name);
    println!("  Layers: {}", graph.num_layers());
    println!(
        "  Total weights: {:.2} MB",
        graph.total_weight_bytes() as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  Total activations: {:.2} MB",
        graph.total_activation_bytes() as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  Largest layer: {:.2} MB",
        graph.max_layer_bytes() as f64 / (1024.0 * 1024.0),
    );
    println!();

    // ── Per-Layer Detail ───────────────────────────────────────
    println!(
        "  {:<4} {:<30} {:<18} {:>10} {:>10} {:>6}",
        "Idx", "Name", "Type", "Weights", "Activ.", "#W",
    );
    println!("  {}", "-".repeat(82));

    for layer in graph.iter_layers() {
        let w_kb = layer.estimated_weight_bytes() as f64 / 1024.0;
        let a_kb = layer.estimated_activation_bytes() as f64 / 1024.0;
        println!(
            "  {:<4} {:<30} {:<18} {:>8.1} KB {:>8.1} KB {:>4}",
            layer.index,
            truncate(&layer.name, 30),
            layer.layer_type.as_str(),
            w_kb,
            a_kb,
            layer.weight_names.len(),
        );
    }
    println!();

    // ── Budget Recommendations ─────────────────────────────────
    let min_budget_bytes = graph.max_layer_bytes();
    let total_bytes = graph.total_weight_bytes() + graph.total_activation_bytes();
    let min_mb = (min_budget_bytes as f64 / (1024.0 * 1024.0)).ceil() as usize;
    let ideal_mb = (total_bytes as f64 / (1024.0 * 1024.0)).ceil() as usize;

    println!("  Budget Recommendations:");
    println!(
        "   Minimum (sequential):     {} MB  (fits largest single layer)",
        min_mb.max(1),
    );
    println!(
        "   Ideal (single group):     {} MB  (fits all layers at once)",
        ideal_mb.max(1),
    );
    println!();

    // ── Strategy Comparison ────────────────────────────────────
    // Try each strategy at a few budget levels.
    let budgets = [
        min_mb.max(1),
        (ideal_mb / 2).max(min_mb.max(1)),
        ideal_mb.max(1),
        (ideal_mb * 2).max(1),
    ];

    println!(
        "  Strategy comparison (groups at different budgets):"
    );
    println!(
        "  {:<24} {:>8} {:>8} {:>8} {:>8}",
        "Strategy",
        format!("{}M", budgets[0]),
        format!("{}M", budgets[1]),
        format!("{}M", budgets[2]),
        format!("{}M", budgets[3]),
    );
    println!("  {}", "-".repeat(60));

    let strategies: Vec<Box<dyn partition_planner::PartitionStrategy>> = vec![
        Box::new(partition_planner::Sequential::new()),
        Box::new(partition_planner::GreedyGrouping::new()),
        Box::new(partition_planner::SpeculativePrefetch::default()),
    ];

    for strategy in &strategies {
        let mut cells = Vec::new();
        for &mb in &budgets {
            let budget = memory_manager::MemoryBudget::from_mb(mb);
            match strategy.plan(&graph, budget) {
                Ok(plan) => cells.push(format!("{}", plan.num_groups())),
                Err(_) => cells.push("OOM".to_string()),
            }
        }
        println!(
            "  {:<24} {:>8} {:>8} {:>8} {:>8}",
            strategy.name(),
            cells[0],
            cells[1],
            cells[2],
            cells[3],
        );
    }

    println!();
    Ok(())
}

/// Truncates a string to `max_len` with ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
