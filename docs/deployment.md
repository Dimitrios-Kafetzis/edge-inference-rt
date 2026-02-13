<!-- Copyright (c) 2025 Dimitris Kafetzis -->
<!-- Licensed under the MIT License. See LICENSE file in the project root. -->
<!-- SPDX-License-Identifier: MIT -->

# Deployment Guide: Raspberry Pi 4

This guide walks you through building, deploying, and running `edge-inference-rt` on a Raspberry Pi 4 Model B. It covers both native compilation on the Pi and cross-compilation from a development machine.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Option A: Native Build on the RPi 4](#option-a-native-build-on-the-rpi-4)
3. [Option B: Cross-Compile from a Dev Machine](#option-b-cross-compile-from-a-dev-machine)
4. [Running Inference](#running-inference)
5. [Running Benchmarks](#running-benchmarks)
6. [Inspecting Models](#inspecting-models)
7. [Monitoring System Resources](#monitoring-system-resources)
8. [Configuration Profiles](#configuration-profiles)
9. [Running the Strategy Comparison Example](#running-the-strategy-comparison-example)
10. [Interpreting Results](#interpreting-results)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **Raspberry Pi 4 Model B** (2 GB, 4 GB, or 8 GB RAM)
- microSD card (32 GB+ recommended)
- Active cooling (heatsink + fan) recommended for sustained workloads

### Software (on the RPi 4)

- **Raspberry Pi OS** (64-bit, Bookworm or later) or any `aarch64` Linux distribution
- **Rust toolchain** (1.75 or later)
- **Git**
- Standard build tools (`gcc`, `make`, `pkg-config`)

Install Rust and system dependencies:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential pkg-config git curl

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify installation
rustc --version   # should be >= 1.75
cargo --version
```

---

## Option A: Native Build on the RPi 4

This is the simplest approach. You compile directly on the Pi.

### 1. Clone the repository

```bash
git clone https://github.com/Dimitrios-Kafetzis/edge-inference-rt.git
cd edge-inference-rt
```

### 2. Build in release mode

```bash
cargo build --release
```

> **Note:** The first build on a Pi 4 may take 10–20 minutes due to dependency compilation. Subsequent builds are incremental and much faster.

### 3. Verify the build

```bash
# The CLI binary is at:
./target/release/edge-rt --version

# Run the test suite
cargo test --release
```

---

## Option B: Cross-Compile from a Dev Machine

Cross-compilation is faster since it uses your more powerful development machine.

### 1. Install the cross-compilation target

```bash
# On your dev machine (x86_64 Linux, macOS, or WSL)
rustup target add aarch64-unknown-linux-gnu

# Install the cross-compilation linker
# On Ubuntu/Debian:
sudo apt install -y gcc-aarch64-linux-gnu

# On macOS (via Homebrew):
# brew install aarch64-unknown-linux-gnu
```

### 2. Configure Cargo for cross-compilation

Create or edit `~/.cargo/config.toml`:

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
```

### 3. Build

```bash
git clone https://github.com/Dimitrios-Kafetzis/edge-inference-rt.git
cd edge-inference-rt

cargo build --release --target aarch64-unknown-linux-gnu
```

### 4. Transfer to the RPi 4

```bash
# Copy the binary to your Pi (replace <PI_IP> with the Pi's IP address)
scp target/aarch64-unknown-linux-gnu/release/edge-rt pi@<PI_IP>:~/

# SSH into the Pi
ssh pi@<PI_IP>

# Verify it runs
chmod +x ~/edge-rt
~/edge-rt --version
```

---

## Running Inference

The `edge-rt run` command executes model inference using the type-state pipeline.

### Synthetic demo (no model files required)

If you don't have model files, the runtime automatically falls back to a synthetic GPT-2-like model:

```bash
# Run with default settings (512M budget, greedy-grouping strategy)
edge-rt run --model ./models/gpt2-small \
            --prompt "The future of edge computing"

# Run with a custom memory budget and strategy
edge-rt run --model ./models/gpt2-small \
            --memory-budget 256M \
            --strategy sequential \
            --prompt "Hello world" \
            --max-tokens 32

# Run with verbose logging
edge-rt run --model ./models/gpt2-small \
            --prompt "Edge AI" \
            -vv
```

### Using a configuration file

```bash
# Use one of the provided configuration profiles
edge-rt -c configs/rpi4_default.toml run --prompt "The future of edge computing"

# Use the constrained profile for tight memory environments
edge-rt -c configs/constrained_2gb.toml run --prompt "Hello"
```

### Available partition strategies

| Strategy | Flag | Description | Best for |
|----------|------|-------------|----------|
| Sequential | `--strategy sequential` | One layer at a time, maximum buffer reuse | Minimum memory usage |
| Greedy Grouping | `--strategy greedy-grouping` | Packs layers into groups fitting within budget | Balanced memory/latency |
| Speculative Prefetch | `--strategy speculative-prefetch` | Prefetches next group while executing current | Maximum throughput |

---

## Running Benchmarks

The `edge-rt benchmark` command sweeps across multiple memory budgets and strategies, producing a comparison table.

### Basic benchmark

```bash
edge-rt benchmark --model ./models/gpt2-small \
                  --sweep-memory 256M,512M,1G
```

### Full benchmark with all strategies

```bash
edge-rt benchmark --model ./models/gpt2-small \
                  --sweep-memory 128M,256M,512M,1G \
                  --strategies sequential,greedy-grouping,speculative-prefetch
```

### Example output

```
╔══════════════════════════════════════════════════════╗
║           edge-rt · Benchmark Suite                 ║
╚══════════════════════════════════════════════════════╝

  Budgets:    ["128M", "256M", "512M", "1G"]
  Strategies: ["sequential", "greedy-grouping", "speculative-prefetch"]

  Model: gpt2-synthetic (20 layers, 16 weights)

  Strategy               Budget   Groups    Peak MB    Latency    Compute    Tok/s
  --------------------------------------------------------------------------------
  sequential               128M       20     0.50 MB     5.20ms     4.10ms    192.3
  sequential               256M       20     0.50 MB     5.18ms     4.08ms    193.1
  greedy-grouping          128M        8     1.20 MB     3.80ms     3.20ms    263.2
  greedy-grouping          256M        4     2.40 MB     2.10ms     1.80ms    476.2
  speculative-prefetch     256M        4     2.88 MB     1.90ms     1.80ms    526.3
  speculative-prefetch     512M        2     4.80 MB     1.20ms     1.00ms    833.3

  Summary:
   Fastest:          speculative-prefetch @ 512M (1.20ms)
   Most efficient:   sequential @ 128M (0.50 MB peak)
```

### Recommended benchmark workflow on RPi 4

```bash
# Step 1: Check system resources first
edge-rt status

# Step 2: Run benchmarks with RPi 4-appropriate budgets
edge-rt benchmark --model ./models/gpt2-small \
                  --sweep-memory 256M,512M,768M \
                  --strategies sequential,greedy-grouping,speculative-prefetch

# Step 3: Run with verbose logging for detailed per-layer timing
edge-rt benchmark --model ./models/gpt2-small \
                  --sweep-memory 512M \
                  -vv
```

---

## Inspecting Models

The `edge-rt inspect` command displays model structure, layer details, and budget recommendations:

```bash
edge-rt inspect --model ./models/gpt2-small
```

### Example output

```
╔══════════════════════════════════════════════════════╗
║              edge-rt · Model Inspector              ║
╚══════════════════════════════════════════════════════╝

  Model: gpt2-small
  Layers: 16
  Total weights: 124.50 MB
  Total activations: 8.20 MB
  Largest layer: 12.00 MB

  Idx  Name                           Type                  Weights  Activ.     #W
  ----------------------------------------------------------------------------------
  0    wte                            embedding           96000.0 KB    3.0 KB    1
  1    wpe                            positional_enc       3072.0 KB    3.0 KB    1
  ...

  Budget Recommendations:
   Minimum (sequential):     12 MB  (fits largest single layer)
   Ideal (single group):     133 MB (fits all layers at once)

  Strategy comparison (groups at different budgets):
  Strategy                     12M      66M     133M     266M
  ------------------------------------------------------------
  sequential                    16       16       16       16
  greedy-grouping              OOM        4        1        1
  speculative-prefetch         OOM        4        1        1
```

---

## Monitoring System Resources

The `edge-rt status` command shows real-time RPi 4 system metrics:

```bash
edge-rt status
```

### Example output

```
╔══════════════════════════════════════════════════════╗
║           edge-rt · System Resource Status          ║
╚══════════════════════════════════════════════════════╝

  Thermal
   Temperature:  52.3 C  [----------..........]
   Headroom:     27.7 C to throttle threshold (80 C)

  Memory
   Total:        3792 MB
   Available:    2841 MB
   Used:         951 MB (25.1%)  [-----...............]

  CPU
   Online cores: 4
   Frequency:    1500 / 1500 MHz
   Load (1m):    0.45  (11% per core)

  Assessment
   Status:       System healthy
   Recommended:  greedy-grouping or speculative-prefetch
   Safe budget:  ~2130 MB (75% of available)
```

Use this output to determine appropriate memory budgets before running benchmarks.

---

## Configuration Profiles

Three pre-built profiles are included in `configs/`:

### `rpi4_default.toml` — Balanced RPi 4 settings

```toml
model_path = "./models/gpt2-small"
memory_budget = "512M"
strategy = "greedy-grouping"
num_threads = 4
prefetch_ratio = 0.2
enable_profiling = true
```

```bash
edge-rt -c configs/rpi4_default.toml run --prompt "Hello"
```

### `constrained_2gb.toml` — Tight memory environments

```toml
model_path = "./models/gpt2-small"
memory_budget = "256M"
strategy = "sequential"
num_threads = 2
enable_profiling = true
```

```bash
edge-rt -c configs/constrained_2gb.toml run --prompt "Hello"
```

### `performance.toml` — Maximum throughput

```toml
model_path = "./models/gpt2-small"
memory_budget = "1G"
strategy = "speculative-prefetch"
num_threads = 4
prefetch_ratio = 0.2
enable_profiling = true
```

```bash
edge-rt -c configs/performance.toml run --prompt "Hello"
```

### Custom configuration

Create your own profile:

```toml
model_path = "./models/my-model"
memory_budget = "768M"
strategy = "greedy-grouping"
num_threads = 4
prefetch_ratio = 0.15
enable_profiling = true
```

---

## Running the Strategy Comparison Example

The `strategy_comparison` example programmatically compares all strategies:

```bash
# Native on the Pi
cargo run --release -p runtime --example strategy_comparison

# Or if you already built with cargo build --release:
./target/release/examples/strategy_comparison
```

This will output a table comparing sequential, greedy-grouping, and speculative-prefetch strategies across multiple memory budgets, then run inference with the greedy strategy to display metrics and pool statistics.

---

## Interpreting Results

### Key metrics

| Metric | Description | What to look for |
|--------|-------------|------------------|
| **Groups** | Number of execution groups (weight-loading rounds) | Fewer groups = fewer I/O rounds = lower latency |
| **Peak MB** | Maximum memory used at any point | Must stay within your budget |
| **Latency** | Total wall-clock time for one inference pass | Lower is better |
| **Compute** | Time spent on actual tensor operations | Shows overhead of memory management |
| **Tok/s** | Tokens generated per second | Higher is better for throughput |

### Strategy selection guidelines

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| RPi 4 (2 GB), other services running | `sequential` | Minimum peak memory, maximum buffer reuse |
| RPi 4 (4 GB), dedicated inference | `greedy-grouping` | Good balance of memory and latency |
| RPi 4 (8 GB) or generous budget | `speculative-prefetch` | Overlaps I/O with compute for best throughput |
| Benchmarking/profiling | Run all three | Compare trade-offs for your specific model |

### Thermal considerations

The RPi 4's Cortex-A72 throttles at 80°C. For sustained benchmarks:

- Use active cooling (fan + heatsink)
- Monitor with `edge-rt status` between runs
- Allow 30–60 seconds of cool-down between benchmark sweeps
- If `status` shows "THERMAL THROTTLING ACTIVE", wait before running benchmarks

---

## Troubleshooting

### Build fails with "linker not found"

```bash
# Install build essentials
sudo apt install -y build-essential
```

### Out of memory during compilation

The RPi 4 (2 GB model) may run out of RAM during compilation. Add swap space:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent (optional)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Alternatively, reduce parallel compilation jobs:

```bash
CARGO_BUILD_JOBS=2 cargo build --release
```

### "Model files not found" message

This is expected if you haven't placed model files in `./models/`. The runtime will automatically fall back to a synthetic demo using a programmatically-built GPT-2-like graph. This is the intended way to demonstrate and benchmark the runtime without needing actual model weights.

### Slow performance / high latency

1. Ensure you are building with `--release` (debug builds are 10–50x slower)
2. Check thermal status: `edge-rt status`
3. Close other memory-intensive applications
4. Try a smaller memory budget to reduce allocation overhead
5. Use `sequential` strategy if memory is very tight

### Permission errors reading thermal sensors

```bash
# Add your user to the relevant group
sudo usermod -aG video $USER
# Log out and back in for group changes to take effect
```
