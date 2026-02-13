// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # edge-rt
//!
//! Command-line interface for the edge-inference-rt runtime.
//!
//! ## Usage
//! ```bash
//! # Run inference
//! edge-rt run --model ./models/gpt2-small --memory-budget 512M --strategy greedy-grouping
//!
//! # Benchmark across memory budgets
//! edge-rt benchmark --model ./models/gpt2-small --sweep-memory 256M,512M,1G
//!
//! # Inspect model structure
//! edge-rt inspect --model ./models/gpt2-small
//! ```

mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "edge-rt",
    about = "Partitioned transformer inference runtime for edge devices",
    version,
    author
)]
struct Cli {
    /// Path to a TOML configuration file (overrides CLI arguments).
    #[arg(short, long, global = true)]
    config: Option<std::path::PathBuf>,

    /// Enable verbose logging (repeat for more: -v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model with a given prompt.
    Run {
        /// Path to the model directory.
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Memory budget (e.g., "512M", "1G").
        #[arg(short = 'b', long, default_value = "512M")]
        memory_budget: String,

        /// Partition strategy: sequential, greedy-grouping, speculative-prefetch.
        #[arg(short, long, default_value = "greedy-grouping")]
        strategy: String,

        /// Input prompt for text generation.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },

    /// Benchmark inference across multiple configurations.
    Benchmark {
        /// Path to the model directory.
        #[arg(short, long)]
        model: std::path::PathBuf,

        /// Comma-separated memory budgets to sweep (e.g., "256M,512M,1G").
        #[arg(long)]
        sweep_memory: String,

        /// Strategies to benchmark (comma-separated).
        #[arg(long, default_value = "sequential,greedy-grouping,speculative-prefetch")]
        strategies: String,
    },

    /// Inspect a model: print layer graph, memory estimates, and metadata.
    Inspect {
        /// Path to the model directory.
        #[arg(short, long)]
        model: std::path::PathBuf,
    },

    /// Display current system resource status (RPi 4 metrics).
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing/logging based on verbosity.
    commands::init_tracing(cli.verbose);

    match cli.command {
        Commands::Run {
            model,
            memory_budget,
            strategy,
            prompt,
            max_tokens,
        } => {
            commands::run::execute(model, memory_budget, strategy, prompt, max_tokens).await
        }
        Commands::Benchmark {
            model,
            sweep_memory,
            strategies,
        } => commands::benchmark::execute(model, sweep_memory, strategies).await,
        Commands::Inspect { model } => commands::inspect::execute(model).await,
        Commands::Status => commands::status::execute().await,
    }
}
