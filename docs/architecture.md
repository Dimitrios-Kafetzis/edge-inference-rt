<!-- Copyright (c) 2025 Dimitris Kafetzis -->
<!-- Licensed under the MIT License. See LICENSE file in the project root. -->
<!-- SPDX-License-Identifier: MIT -->

# Architecture

## Overview

`edge-inference-rt` is a modular runtime for executing partitioned transformer
models on resource-constrained edge devices (targeting Raspberry Pi 4).

## Crate Dependency Graph

```text
                    ┌─────────────┐
                    │ edge-rt-cli │  (binary)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   runtime   │  (async execution engine)
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          │                │                │
   ┌──────▼───────┐ ┌─────▼──────┐ ┌───────▼─────────┐
   │  partition-   │ │  memory-   │ │ resource-       │
   │  planner      │ │  manager   │ │ monitor         │
   └──┬─────┬──────┘ └─────┬──────┘ └─────────────────┘
      │     │              │
   ┌──▼──┐  │         ┌────▼──────┐
   │model│  └────────► │ tensor-  │
   │ -ir │             │ core     │
   └──┬──┘             └──────────┘
      │
      └──────────────► tensor-core
```

## Execution Flow

1. **Load**: `model-ir` reads `model.json` + `model.safetensors` → `ModelGraph<Validated>`
2. **Plan**: `partition-planner` selects a strategy → `ExecutionPlan` with `LayerGroup`s
3. **Allocate**: `memory-manager` creates a `MemoryPool` with the configured budget
4. **Execute**: `runtime` processes groups sequentially:
   - Load group weights via `WeightLoader`
   - Execute each layer through `tensor-core` operations
   - Release group buffers (RAII via `BufferGuard`)
   - (Speculative) Prefetch next group's weights concurrently
5. **Report**: Collect `InferenceMetrics` and display results

## Key Design Decisions

- **Type-state pattern** for `ModelGraph` and `InferenceEngine` to prevent misuse.
- **RAII buffer guards** tie memory lifetimes to the borrow checker.
- **Strategy trait** makes partition algorithms pluggable and independently testable.
- **Zero-copy weight loading** via `memmap2` for SafeTensors files.
- **Conditional compilation** for `aarch64` NEON optimisations.
