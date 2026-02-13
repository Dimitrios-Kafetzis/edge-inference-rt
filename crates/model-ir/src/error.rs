// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for model loading and IR construction.

/// Errors that can occur when working with model representations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The model manifest file could not be read.
    #[error("failed to read manifest: {0}")]
    ManifestReadError(#[from] std::io::Error),

    /// The manifest JSON is malformed.
    #[error("failed to parse manifest: {0}")]
    ManifestParseError(#[from] serde_json::Error),

    /// A weight tensor referenced in the manifest was not found in the SafeTensors file.
    #[error("weight tensor not found: {name}")]
    WeightNotFound { name: String },

    /// The SafeTensors file could not be loaded.
    #[error("failed to load SafeTensors: {0}")]
    SafeTensorsError(String),

    /// A layer definition is invalid (e.g., incompatible shapes).
    #[error("invalid layer '{layer}': {detail}")]
    InvalidLayer { layer: String, detail: String },

    /// The model graph contains a cycle or is otherwise malformed.
    #[error("invalid model graph: {0}")]
    InvalidGraph(String),
}
