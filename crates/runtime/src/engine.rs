// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! The core inference engine with type-state–enforced pipeline.
//!
//! ```text
//! InferenceEngine<Idle>
//!     │  .load_model()
//!     ▼
//! InferenceEngine<Planned>
//!     │  .prepare()
//!     ▼
//! InferenceEngine<Ready>
//!     │  .run()
//!     ▼
//!   InferenceOutput
//! ```
//!
//! Each state transition consumes the old value and returns a new one,
//! making invalid state sequences a compile error.

use crate::{InferenceMetrics, RuntimeConfig, RuntimeError, WeightLoader};
use memory_manager::{MemoryBudget, MemoryPool};
use model_ir::{graph::Validated, ModelGraph};
use partition_planner::ExecutionPlan;
use std::time::Instant;

// ── Type-state markers ─────────────────────────────────────────

/// Engine is created but no model is loaded.
#[derive(Debug)]
pub struct Idle;

/// Model is loaded and an execution plan has been generated.
#[derive(Debug)]
pub struct Planned;

/// Engine is ready to run inference.
#[derive(Debug)]
pub struct Ready;

/// Sealed trait for engine states.
pub trait EngineState: std::fmt::Debug {}
impl EngineState for Idle {}
impl EngineState for Planned {}
impl EngineState for Ready {}

// ── Inference output ───────────────────────────────────────────

/// The result of a single inference run.
#[derive(Debug)]
pub struct InferenceOutput {
    /// Generated token IDs (simulated in this implementation).
    pub token_ids: Vec<u32>,
    /// Per-layer and overall timing/memory metrics.
    pub metrics: InferenceMetrics,
}

// ── Engine ─────────────────────────────────────────────────────

/// The primary inference engine.
///
/// `S` is a type-state marker that enforces the pipeline ordering at
/// compile time. You cannot call `.run()` on an `Idle` engine or
/// `.load_model()` on a `Ready` engine — the compiler catches it.
///
/// # Example
/// ```no_run
/// use runtime::{InferenceEngine, RuntimeConfig};
///
/// # async fn example() -> Result<(), runtime::RuntimeError> {
/// let engine = InferenceEngine::new(RuntimeConfig::default())
///     .load_model()?
///     .prepare()?;
/// let output = engine.run(&[1, 2, 3]).await?;
/// println!("{}", output.metrics.summary());
/// # Ok(())
/// # }
/// ```
pub struct InferenceEngine<S: EngineState = Idle> {
    config: RuntimeConfig,
    _state: std::marker::PhantomData<S>,
    // Fields populated as the engine transitions through states:
    graph: Option<ModelGraph<Validated>>,
    plan: Option<ExecutionPlan>,
    pool: Option<MemoryPool>,
    weight_loader: Option<WeightLoader>,
    budget: Option<MemoryBudget>,
}

// ── Idle → Planned ─────────────────────────────────────────────

impl InferenceEngine<Idle> {
    /// Creates a new engine from the given configuration.
    pub fn new(config: RuntimeConfig) -> Self {
        tracing::info!("engine created with strategy '{}'", config.strategy);
        Self {
            config,
            _state: std::marker::PhantomData,
            graph: None,
            plan: None,
            pool: None,
            weight_loader: None,
            budget: None,
        }
    }

    /// Loads the model and generates an execution plan.
    /// Transitions to the `Planned` state.
    ///
    /// Steps:
    /// 1. Parse the memory budget.
    /// 2. Load and validate the model graph.
    /// 3. Select the partition strategy.
    /// 4. Generate the execution plan.
    pub fn load_model(self) -> Result<InferenceEngine<Planned>, RuntimeError> {
        let budget = self.config.parse_budget()?;
        tracing::info!("memory budget: {budget}");

        // Load model graph.
        let graph = model_ir::ModelLoader::load(&self.config.model_path)?;
        tracing::info!("{}", graph.summary());

        // Create strategy and generate plan.
        let strategy = self.config.create_strategy()?;
        tracing::info!("using strategy: {}", strategy.name());

        let plan = strategy.plan(&graph, budget)?;
        plan.validate().map_err(|e| {
            RuntimeError::InvalidPlan(format!("plan validation failed: {e}"))
        })?;
        tracing::info!("{}", plan.summary());

        Ok(InferenceEngine {
            config: self.config,
            _state: std::marker::PhantomData,
            graph: Some(graph),
            plan: Some(plan),
            pool: None,
            weight_loader: None,
            budget: Some(budget),
        })
    }

    /// Convenience: loads from a pre-built graph and plan (for testing).
    pub fn from_graph_and_plan(
        config: RuntimeConfig,
        graph: ModelGraph<Validated>,
        plan: ExecutionPlan,
        budget: MemoryBudget,
    ) -> InferenceEngine<Planned> {
        InferenceEngine {
            config,
            _state: std::marker::PhantomData,
            graph: Some(graph),
            plan: Some(plan),
            pool: None,
            weight_loader: None,
            budget: Some(budget),
        }
    }
}

// ── Planned → Ready ────────────────────────────────────────────

impl InferenceEngine<Planned> {
    /// Returns a reference to the execution plan.
    pub fn plan(&self) -> &ExecutionPlan {
        self.plan.as_ref().expect("plan must exist in Planned state")
    }

    /// Returns a reference to the model graph.
    pub fn graph(&self) -> &ModelGraph<Validated> {
        self.graph.as_ref().expect("graph must exist in Planned state")
    }

    /// Allocates the memory pool and prepares the weight loader.
    /// Transitions to the `Ready` state.
    pub fn prepare(self) -> Result<InferenceEngine<Ready>, RuntimeError> {
        let budget = self.budget.expect("budget must exist in Planned state");

        // Create memory pool.
        let pool = MemoryPool::new(budget);
        tracing::info!("memory pool created: {} available", budget);

        // Create weight loader.
        let loader = WeightLoader::new(self.config.model_path.clone())?;
        tracing::info!(
            "weight loader: {} mode",
            if loader.is_file_backed() { "file-backed" } else { "synthetic" }
        );

        Ok(InferenceEngine {
            config: self.config,
            _state: std::marker::PhantomData,
            graph: self.graph,
            plan: self.plan,
            pool: Some(pool),
            weight_loader: Some(loader),
            budget: Some(budget),
        })
    }
}

// ── Ready: run inference ───────────────────────────────────────

impl InferenceEngine<Ready> {
    /// Returns the current memory pool statistics.
    pub fn memory_stats(&self) -> memory_manager::AllocationStats {
        self.pool().stats()
    }

    /// Returns the execution plan.
    pub fn plan(&self) -> &ExecutionPlan {
        self.plan.as_ref().expect("plan exists in Ready state")
    }

    /// Returns the model graph.
    pub fn graph(&self) -> &ModelGraph<Validated> {
        self.graph.as_ref().expect("graph exists in Ready state")
    }

    /// Runs inference on the given input token IDs.
    ///
    /// Iterates through the execution plan group by group:
    /// 1. Load weights for all layers in the group.
    /// 2. Execute each layer in order.
    /// 3. Release weight buffers (via RAII drop).
    /// 4. Record per-layer metrics.
    ///
    /// Currently uses synthetic computation (tensor ops on zero-filled
    /// buffers) as a functional skeleton. Real model execution would
    /// replace the inner computation with actual attention, FFN, etc.
    pub async fn run(&self, input_tokens: &[u32]) -> Result<InferenceOutput, RuntimeError> {
        let run_start = Instant::now();
        let plan = self.plan.as_ref().unwrap();
        let graph = self.graph.as_ref().unwrap();
        let pool = self.pool();
        let loader = self.weight_loader.as_ref().unwrap();

        let mut metrics = InferenceMetrics::new(plan.num_groups());
        let profiling = self.config.enable_profiling;

        tracing::debug!(
            "starting inference: {} input tokens, {} groups",
            input_tokens.len(),
            plan.num_groups(),
        );

        // Process each group.
        for group in &plan.groups {
            tracing::debug!(
                "executing group {} ({} layers: {:?})",
                group.group_index,
                group.num_layers(),
                group.layer_indices,
            );

            // Load all weights for this group's layers.
            let group_load_start = Instant::now();
            let mut group_guards = Vec::new();

            for &layer_idx in &group.layer_indices {
                let layer = graph.layer(layer_idx).ok_or_else(|| {
                    RuntimeError::InvalidPlan(format!(
                        "layer index {} not found in graph",
                        layer_idx,
                    ))
                })?;
                let guards = loader.load_layer_buffers(layer, pool)?;
                group_guards.push((layer_idx, guards));
            }

            let group_load_duration = group_load_start.elapsed();

            // Execute each layer in the group.
            for &layer_idx in &group.layer_indices {
                let layer = graph.layer(layer_idx).unwrap();
                let layer_start = Instant::now();

                // ── Simulated computation ──────────────────────
                // In a real runtime, this would:
                // 1. Load the input activation tensor.
                // 2. Execute the layer-specific operation (matmul, attention, etc.).
                // 3. Store the output activation for the next layer.
                //
                // For this portfolio demonstration, we allocate and immediately
                // free activation buffers to exercise the memory pool, proving
                // the RAII lifecycle and budget enforcement work end-to-end.
                {
                    let act_size = layer.estimated_activation_bytes();
                    if act_size > 0 {
                        let _activation = pool.allocate(act_size).map_err(|e| {
                            RuntimeError::ExecutionError {
                                layer: layer.name.clone(),
                                source: tensor_core::TensorError::Numeric {
                                    op: "allocate_activation",
                                    detail: format!("activation allocation failed: {e}"),
                                },
                            }
                        })?;
                        // In real inference, computation happens here.
                        // The activation guard is dropped at the end of this scope.
                    }
                }

                let compute_duration = layer_start.elapsed();

                if profiling {
                    // Weight load time is amortised across the group.
                    let per_layer_load = group_load_duration / group.num_layers() as u32;
                    let peak = pool.stats().peak_allocated_bytes;
                    metrics.record_layer(
                        layer.name.clone(),
                        per_layer_load,
                        compute_duration,
                        peak,
                    );
                }
            }

            // ── Drop weight guards → memory returned to pool ──
            drop(group_guards);
            tracing::debug!(
                "group {} complete, pool: {} bytes allocated, {} bytes free-list",
                group.group_index,
                pool.allocated_bytes(),
                pool.free_list_bytes(),
            );
        }

        // Simulate token generation (in a real runtime, this would be
        // the output of the final softmax / argmax).
        let generated_tokens: Vec<u32> = input_tokens
            .iter()
            .map(|&t| t.wrapping_add(1))
            .collect();

        metrics.finalise(run_start.elapsed(), generated_tokens.len());
        tracing::info!("{}", metrics.summary());

        Ok(InferenceOutput {
            token_ids: generated_tokens,
            metrics,
        })
    }

    // ── Private helpers ────────────────────────────────────────

    fn pool(&self) -> &MemoryPool {
        self.pool.as_ref().expect("pool exists in Ready state")
    }
}

impl<S: EngineState> std::fmt::Debug for InferenceEngine<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("state", &std::any::type_name::<S>())
            .field("strategy", &self.config.strategy)
            .field(
                "has_graph",
                &self.graph.is_some(),
            )
            .field("has_plan", &self.plan.is_some())
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::{LayerDef, LayerType, ModelGraph};
    use partition_planner::{GreedyGrouping, PartitionStrategy};
    use tensor_core::{DType, Shape};

    /// Creates a small test graph and matching plan.
    fn test_graph_and_plan(
        n_layers: usize,
        hidden: usize,
        budget: MemoryBudget,
    ) -> (ModelGraph<Validated>, ExecutionPlan) {
        let layers: Vec<LayerDef> = (0..n_layers)
            .map(|i| LayerDef {
                name: format!("layer.{i}"),
                layer_type: LayerType::Linear,
                index: i,
                weight_names: vec![format!("w.{i}")],
                weight_shapes: vec![Shape::matrix(hidden, hidden)],
                dtype: DType::F32,
                input_shape: Shape::matrix(1, hidden),
                output_shape: Shape::matrix(1, hidden),
            })
            .collect();

        let graph = ModelGraph::new("test_model".into(), layers)
            .validate()
            .unwrap();
        let plan = GreedyGrouping::new().plan(&graph, budget).unwrap();
        (graph, plan)
    }

    #[test]
    fn test_idle_to_planned() {
        let budget = MemoryBudget::from_mb(10);
        let (graph, plan) = test_graph_and_plan(4, 32, budget);

        let config = RuntimeConfig {
            strategy: "greedy-grouping".into(),
            memory_budget: "10M".into(),
            ..Default::default()
        };

        let engine = InferenceEngine::from_graph_and_plan(config, graph, plan, budget);
        assert_eq!(engine.plan().num_groups(), 1);
        assert_eq!(engine.graph().num_layers(), 4);
    }

    #[test]
    fn test_planned_to_ready() {
        let budget = MemoryBudget::from_mb(10);
        let (graph, plan) = test_graph_and_plan(4, 32, budget);

        let config = RuntimeConfig::default();
        let planned = InferenceEngine::from_graph_and_plan(config, graph, plan, budget);
        let ready = planned.prepare().unwrap();

        let stats = ready.memory_stats();
        assert_eq!(stats.total_allocations, 0);
    }

    #[tokio::test]
    async fn test_full_pipeline() {
        let budget = MemoryBudget::from_mb(10);
        let (graph, plan) = test_graph_and_plan(4, 32, budget);

        let config = RuntimeConfig {
            enable_profiling: true,
            ..Default::default()
        };
        let engine = InferenceEngine::from_graph_and_plan(config, graph, plan, budget)
            .prepare()
            .unwrap();

        let output = engine.run(&[1, 2, 3]).await.unwrap();

        // Token IDs should be generated.
        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(output.token_ids, vec![2, 3, 4]);

        // Metrics should be populated.
        assert!(output.metrics.total_duration.as_nanos() > 0);
        assert_eq!(output.metrics.tokens_generated, 3);
        assert_eq!(output.metrics.layer_metrics.len(), 4);

        // Pool should be clean after run.
        assert_eq!(engine.memory_stats().total_allocations, engine.memory_stats().total_deallocations);
    }

    #[tokio::test]
    async fn test_memory_returned_after_run() {
        let budget = MemoryBudget::from_mb(10);
        let (graph, plan) = test_graph_and_plan(6, 64, budget);

        let config = RuntimeConfig::default();
        let engine = InferenceEngine::from_graph_and_plan(config, graph, plan, budget)
            .prepare()
            .unwrap();

        let _output = engine.run(&[42]).await.unwrap();

        // All memory should be returned to the pool.
        let pool_stats = engine.memory_stats();
        assert_eq!(
            pool_stats.total_allocations,
            pool_stats.total_deallocations,
            "all allocations should be deallocated after run"
        );
    }

    #[tokio::test]
    async fn test_multiple_runs() {
        let budget = MemoryBudget::from_mb(10);
        let (graph, plan) = test_graph_and_plan(3, 32, budget);

        let config = RuntimeConfig::default();
        let engine = InferenceEngine::from_graph_and_plan(config, graph, plan, budget)
            .prepare()
            .unwrap();

        // Run multiple times — pool should be reusable.
        for i in 0..5 {
            let output = engine.run(&[i as u32]).await.unwrap();
            assert_eq!(output.token_ids.len(), 1);
        }

        let stats = engine.memory_stats();
        assert!(stats.cache_hits > 0, "later runs should get cache hits");
    }

    #[test]
    fn test_debug_format() {
        let engine = InferenceEngine::new(RuntimeConfig::default());
        let debug = format!("{engine:?}");
        assert!(debug.contains("InferenceEngine"));
        assert!(debug.contains("greedy-grouping"));
    }
}
