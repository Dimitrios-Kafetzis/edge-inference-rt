// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! `edge-rt status` command: display current system resource state.
//!
//! Reads thermal, memory, and CPU metrics from the Linux sysfs/procfs
//! interfaces. On non-RPi platforms (e.g., Docker, x86 laptops) some
//! readings may show defaults — the command still works.

pub async fn execute() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║           edge-rt · System Resource Status          ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    let snapshot = resource_monitor::snapshot()?;

    // ── Thermal ────────────────────────────────────────────────
    println!("  Thermal");
    let temp = snapshot.thermal.cpu_temp_celsius as f64;
    let bar = temp_bar(temp);
    println!("   Temperature:  {:.1} C  {bar}", temp);
    println!(
        "   Headroom:     {:.1} C to throttle threshold (80 C)",
        snapshot.thermal.headroom_celsius(),
    );
    if snapshot.thermal.is_overheating() {
        println!("   WARNING: THERMAL THROTTLING ACTIVE");
    }
    println!();

    // ── Memory ─────────────────────────────────────────────────
    println!("  Memory");
    let total = snapshot.memory.total_mb();
    let avail = snapshot.memory.available_mb();
    let used = total - avail;
    let pct = snapshot.memory.utilisation() * 100.0;
    let bar = usage_bar(snapshot.memory.utilisation());
    println!("   Total:        {} MB", total);
    println!("   Available:    {} MB", avail);
    println!("   Used:         {} MB ({:.1}%)  {bar}", used, pct);
    println!();

    // ── CPU ────────────────────────────────────────────────────
    println!("  CPU");
    println!("   Online cores: {}", snapshot.cpu.online_cores);
    println!(
        "   Frequency:    {} / {} MHz",
        snapshot.cpu.frequency_mhz, snapshot.cpu.max_frequency_mhz,
    );
    let load_pct = (snapshot.cpu.load_per_core as f64 / snapshot.cpu.online_cores as f64) * 100.0;
    println!(
        "   Load (1m):    {:.2}  ({:.0}% per core)",
        snapshot.cpu.load_per_core,
        load_pct,
    );
    if snapshot.cpu.is_throttled() {
        println!("   WARNING: CPU frequency is throttled");
    }
    println!();

    // ── Overall Assessment ─────────────────────────────────────
    let constrained = snapshot.is_resource_constrained();
    let recommended = if constrained {
        "sequential (resource-constrained)"
    } else {
        "greedy-grouping or speculative-prefetch"
    };

    println!("  Assessment");
    if constrained {
        println!("   Status:       RESOURCE CONSTRAINED");
    } else {
        println!("   Status:       System healthy");
    }
    println!("   Recommended:  {recommended}");

    let safe_budget = (avail as f64 * 0.75) as u64;
    println!("   Safe budget:  ~{safe_budget} MB (75% of available)");
    println!();
    println!("{}", snapshot.summary());

    Ok(())
}

/// Creates a visual temperature bar (0-100 C scale).
fn temp_bar(celsius: f64) -> String {
    let filled = ((celsius / 100.0) * 20.0).round() as usize;
    let filled = filled.min(20);
    let empty = 20 - filled;
    let symbol = if celsius >= 80.0 {
        "#"
    } else if celsius >= 60.0 {
        "="
    } else {
        "-"
    };
    format!("[{}{}]", symbol.repeat(filled), ".".repeat(empty))
}

/// Creates a visual usage bar (0.0-1.0 scale).
fn usage_bar(ratio: f64) -> String {
    let filled = ((ratio) * 20.0).round() as usize;
    let filled = filled.min(20);
    let empty = 20 - filled;
    let symbol = if ratio >= 0.9 {
        "#"
    } else if ratio >= 0.7 {
        "="
    } else {
        "-"
    };
    format!("[{}{}]", symbol.repeat(filled), ".".repeat(empty))
}
