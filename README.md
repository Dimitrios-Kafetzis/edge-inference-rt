<!-- Copyright (c) 2025 Dimitris Kafetzis -->
<!-- Licensed under the MIT License. See LICENSE file in the project root. -->
<!-- SPDX-License-Identifier: MIT -->

# edge-inference-rt

A modular, memory-aware runtime for partitioned transformer inference on resource-constrained edge devices, written in Rust.

[![CI](https://github.com/Dimitrios-Kafetzis/edge-inference-rt/actions/workflows/ci.yml/badge.svg)](https://github.com/Dimitrios-Kafetzis/edge-inference-rt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust: 1.75+](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)

## Motivation

Running large transformer models on edge devices like the Raspberry Pi 4 requires intelligent resource management: you cannot simply load the full model into 4 GB of RAM. This runtime demonstrates how to dynamically partition a transformer across layers, schedule execution under a strict memory budget, and optionally prefetch weights speculatively — all with Rust's safety guarantees ensuring no memory leaks, no data races, and deterministic resource cleanup.

The project is designed as a portfolio demonstration of idiomatic Rust for embedded/systems programming, showcasing type-state machines, RAII resource management, trait-based extensibility, zero-copy I/O, and async orchestration.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        edge-rt-cli                              │
│              (run · benchmark · inspect · status)               │
├─────────────────────────────────────────────────────────────────┤
│                          runtime                                │
│     InferenceEngine<Idle> → <Planned> → <Ready> → run()        │
├──────────────┬──────────────────┬───────────────────────────────┤
│  partition-  │  memory-manager  │  resource-monitor             │
│  planner     │  MemoryPool +    │  CPU · Memory · Thermal       │
│  Sequential  │  BufferGuard     │  (sysfs / procfs)             │
│  Greedy      │  (RAII)          │                               │
│  Speculative │                  │                               │
├──────────────┴──────────┬───────┴───────────────────────────────┤
│                      model-ir                                   │
│  ModelGraph<Loaded> → validate() → ModelGraph<Validated>        │
├─────────────────────────────────────────────────────────────────┤
│                     tensor-core                                 │
│           Tensor · Shape · DType · Ops (NEON-ready)             │
└─────────────────────────────────────────────────────────────────┘
```

## Partition Strategies

| Strategy | Groups | Latency | Memory | Use Case |
|---|---|---|---|---|
| `sequential` | N (one/layer) | Highest | Lowest | Thermally constrained, <256 MB |
| `greedy-grouping` | Fewer | Medium | Medium | Default for most workloads |
| `speculative-prefetch` | Fewer (headroom) | Lowest | Higher | Ample memory, I/O-bound |

All strategies implement the `PartitionStrategy` trait, enabling new strategies without modifying the runtime.

## Quick Start

### Prerequisites

- Rust 1.75+ (`rustup update stable`)
- For RPi 4 cross-compilation: `aarch64-unknown-linux-gnu` target

### Build & Test

```bash
# Build everything
cargo build --release

# Run the full test suite (190+ tests)
cargo test --workspace

# Generate documentation
cargo doc --workspace --no-deps --open

# Cross-compile for RPi 4
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

### CLI Commands

```bash
# Run inference (falls back to synthetic demo if model files absent)
edge-rt run --model ./models/gpt2-small \
            --memory-budget 512M \
            --strategy greedy-grouping \
            --prompt "The ship entered the port of"

# Benchmark across configurations
edge-rt benchmark --model ./models/gpt2-small \
                  --sweep-memory 256M,512M,1G \
                  --strategies sequential,greedy-grouping,speculative-prefetch

# Inspect model structure and plan comparison
edge-rt inspect --model ./models/gpt2-small

# Display system resource status (thermal, memory, CPU)
edge-rt status
```

### Example Output

```
$ edge-rt benchmark --model ./synth --sweep-memory 256M,512M,1G

╔══════════════════════════════════════════════════════╗
║           edge-rt · Benchmark Suite                 ║
╚══════════════════════════════════════════════════════╝

  Strategy                 Budget   Groups    Peak MB    Latency    Compute    Tok/s
  --------------------------------------------------------------------------------
  sequential               256 MB       20     9.01 MB   645.17ms     1.02ms     49.6
  greedy-grouping          256 MB        1    99.01 MB   685.42ms     0.69ms     46.7
  speculative-prefetch     256 MB        1    99.01 MB   642.57ms     0.72ms     49.8
  sequential               512 MB       20     9.01 MB   647.07ms     1.01ms     49.5
  greedy-grouping          512 MB        1    99.01 MB   657.27ms     0.74ms     48.7
  speculative-prefetch       1 GB        1    99.01 MB   632.99ms     0.77ms     50.6

  Summary:
   Fastest:          speculative-prefetch @ 1 GB (632.99ms)
   Most efficient:   sequential @ 256 MB (9.01 MB peak)
```

### Configuration

Use TOML presets from `configs/`:

```bash
edge-rt run -c configs/rpi4_default.toml --prompt "Hello"
edge-rt run -c configs/constrained_2gb.toml --prompt "Hello"
edge-rt run -c configs/performance.toml --prompt "Hello"
```

## Project Structure

```
edge-inference-rt/
├── crates/
│   ├── tensor-core/         # Tensor types, shapes, dtype, arithmetic ops
│   ├── resource-monitor/    # RPi 4 system metrics (CPU, memory, thermal)
│   ├── model-ir/            # Model graph IR + SafeTensors loading
│   ├── memory-manager/      # RAII memory pool with budget enforcement
│   ├── partition-planner/   # Strategy engine (sequential, greedy, speculative)
│   └── runtime/             # Async execution engine with type-state pipeline
├── bin/edge-rt-cli/         # CLI binary (run, benchmark, inspect, status)
├── configs/                 # TOML configuration presets
└── .github/workflows/       # CI: check, test, fmt, doc, cross-compile
```

**51 source files · 8,200 lines of Rust · 190+ tests · 7 workspace members**

## Key Rust Idioms Demonstrated

### Type-State Pattern
Invalid state transitions are **compile errors**, not runtime panics:

```rust
// ModelGraph: Loaded → Validated (compile-time enforced)
let graph: ModelGraph<Loaded> = ModelLoader::load(path)?;
let graph: ModelGraph<Validated> = graph.validate()?;
// graph.num_layers() only available on Validated — won't compile on Loaded

// InferenceEngine: Idle → Planned → Ready (each transition consumes self)
let engine = InferenceEngine::new(config)     // Idle
    .load_model()?                            // → Planned
    .prepare()?;                              // → Ready
let output = engine.run(&tokens).await?;      // only Ready can run
```

### RAII Buffer Guards
Memory returned to pool automatically on `Drop` — the borrow checker prevents use-after-free at compile time:

```rust
let guard: BufferGuard = pool.allocate(4096)?;  // reserves from budget
let slice: &mut [u8] = guard.as_mut_slice();    // zero-copy access
drop(guard);                                     // ← memory returned to pool
// pool.allocated_bytes() == 0 — guaranteed, no leaks possible
```

### Trait-Based Strategy
New partitioning strategies plug in without modifying the runtime:

```rust
pub trait PartitionStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn plan(&self, graph: &ModelGraph<Validated>, budget: MemoryBudget)
        -> Result<ExecutionPlan, PlannerError>;
}
// Implement the trait → auto-works with CLI, benchmarks, auto_plan()
```

### Zero-Copy I/O
SafeTensors weights loaded via `memmap2` — only the header is parsed; tensor data stays on disk until accessed:

```rust
let loader = WeightLoader::new(model_dir)?;  // mmap's the file
let guards = loader.load_layer_buffers(&layer, &pool)?;
// Data flows: disk → mmap → pool buffer (one copy, budget-tracked)
```

### Other Idioms
- **Interior mutability**: `AtomicUsize` for hot-path counters, `Mutex` only for cold-path free lists
- **Size-class binning**: returned buffers cached by power-of-2 for O(1) free-list lookups
- **Sealed traits**: `EngineState` restricts type-state markers to `Idle | Planned | Ready`
- **Builder pattern**: `PlanBuilder` keeps strategy implementations clean
- **Conditional compilation**: `#[cfg(target_arch = "aarch64")]` stubs for ARM NEON kernels

## Target Platform

**Raspberry Pi 4 Model B** (4 GB)

| Spec | Detail |
|---|---|
| SoC | BCM2711, 4× Cortex-A72 @ 1.8 GHz |
| RAM | 4 GB LPDDR4 |
| Storage | microSD / USB 3.0 SSD |
| OS | Raspberry Pi OS (64-bit) / Ubuntu Server 24.04 aarch64 |

## CI Pipeline

The GitHub Actions workflow (`ci.yml`) runs five jobs on every push/PR:

1. **Check & Clippy** — `cargo check` + `cargo clippy -- -D warnings`
2. **Test** — `cargo test --workspace` (all 190+ tests)
3. **Format** — `cargo fmt --all -- --check`
4. **Documentation** — `cargo doc --workspace --no-deps` with `-D warnings`
5. **Cross-compile** — builds for `aarch64-unknown-linux-gnu` (RPi 4 target)

## License

MIT — see [LICENSE](LICENSE) for details.
