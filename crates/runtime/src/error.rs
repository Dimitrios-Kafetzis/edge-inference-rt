// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for the inference runtime.

/// Errors that can occur during inference execution.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The execution plan is invalid or inconsistent with the model.
    #[error("invalid execution plan: {0}")]
    InvalidPlan(String),

    /// Failed to load weights from disk.
    #[error("weight loading failed for layer '{layer}': {detail}")]
    WeightLoadError { layer: String, detail: String },

    /// A tensor operation failed during layer execution.
    #[error("execution error in layer '{layer}': {source}")]
    ExecutionError {
        layer: String,
        #[source]
        source: tensor_core::TensorError,
    },

    /// Memory allocation failed during execution.
    #[error("memory error: {0}")]
    MemoryError(#[from] memory_manager::MemoryError),

    /// The partition planner returned an error.
    #[error("planner error: {0}")]
    PlannerError(#[from] partition_planner::PlannerError),

    /// Model loading failed.
    #[error("model error: {0}")]
    ModelError(#[from] model_ir::ModelError),

    /// Configuration error.
    #[error("configuration error: {0}")]
    ConfigError(String),
}
