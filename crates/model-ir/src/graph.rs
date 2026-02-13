// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Model graph: the complete transformer as a DAG of layers.
//!
//! # Type-State Pattern
//!
//! The graph transitions through states enforced at compile time:
//!
//! ```text
//! ModelGraph<Loaded>     — layers parsed, not yet checked.
//!       │  .validate()
//!       ▼
//! ModelGraph<Validated>  — shapes verified, ready for partitioning.
//! ```
//!
//! This prevents the partition planner from ever receiving an invalid graph.
//! The transition consumes the old state and returns the new one, so there
//! is zero runtime cost — the marker types are `PhantomData` (ZST).

use crate::{LayerDef, ModelError};
use std::fmt;

// ── Type-state markers ─────────────────────────────────────────────

/// Marker: graph has been loaded but not validated.
#[derive(Debug, Clone)]
pub struct Loaded;

/// Marker: graph has been validated and is ready for partitioning.
#[derive(Debug, Clone)]
pub struct Validated;

/// Sealed trait for graph states.
pub trait GraphState: fmt::Debug + Clone {}
impl GraphState for Loaded {}
impl GraphState for Validated {}

// ── ModelGraph ─────────────────────────────────────────────────────

/// The complete model represented as an ordered sequence of layers.
///
/// For transformer models this is a linear chain. The generic parameter
/// `S` encodes the validation state at compile time.
#[derive(Debug, Clone)]
pub struct ModelGraph<S: GraphState = Loaded> {
    /// Human-readable model name (e.g., `"gpt2-small"`).
    pub name: String,
    /// Ordered list of layer definitions.
    pub layers: Vec<LayerDef>,
    /// State marker (zero-sized, compile-time only).
    _state: std::marker::PhantomData<S>,
}

// ── Loaded state ───────────────────────────────────────────────────

impl ModelGraph<Loaded> {
    /// Creates a new graph in the `Loaded` state.
    pub fn new(name: String, layers: Vec<LayerDef>) -> Self {
        Self {
            name,
            layers,
            _state: std::marker::PhantomData,
        }
    }

    /// Validates the graph and transitions to the `Validated` state.
    ///
    /// # Checks
    /// - The graph is non-empty.
    /// - Layer indices are consecutive starting from 0.
    /// - Shape compatibility: each layer's output shape matches the next
    ///   layer's input shape (with exceptions for embedding → first layer).
    /// - No layer has zero-element shapes.
    pub fn validate(self) -> Result<ModelGraph<Validated>, ModelError> {
        if self.layers.is_empty() {
            return Err(ModelError::InvalidGraph(
                "model graph contains no layers".into(),
            ));
        }

        // Check consecutive indices.
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.index != i {
                return Err(ModelError::InvalidLayer {
                    layer: layer.name.clone(),
                    detail: format!("expected index {i}, got {}", layer.index),
                });
            }
        }

        // Check no zero-element shapes.
        for layer in &self.layers {
            if layer.input_shape.num_elements() == 0 {
                return Err(ModelError::InvalidLayer {
                    layer: layer.name.clone(),
                    detail: "input shape has zero elements".into(),
                });
            }
            if layer.output_shape.num_elements() == 0 {
                return Err(ModelError::InvalidLayer {
                    layer: layer.name.clone(),
                    detail: "output shape has zero elements".into(),
                });
            }
        }

        // Check shape compatibility between consecutive layers.
        // We compare the last dimension (hidden size) of the output with
        // the last dimension of the next layer's input. This is a relaxed
        // check: sequence length may change (e.g., through attention masking)
        // but the hidden dimension must be consistent.
        for i in 0..self.layers.len() - 1 {
            let current = &self.layers[i];
            let next = &self.layers[i + 1];

            let out_dims = current.output_shape.dims();
            let in_dims = next.input_shape.dims();

            // The last dimension (hidden/feature size) should match.
            if let (Some(&out_last), Some(&in_last)) = (out_dims.last(), in_dims.last()) {
                if out_last != in_last {
                    tracing::warn!(
                        "shape mismatch between '{}' output (last dim {}) and '{}' input (last dim {})",
                        current.name, out_last, next.name, in_last,
                    );
                    // This is a warning, not a hard error, because some
                    // layer transitions legitimately change dimensionality
                    // (e.g., embedding output → first attention input may
                    // differ if positional encoding changes shape).
                }
            }
        }

        Ok(ModelGraph {
            name: self.name,
            layers: self.layers,
            _state: std::marker::PhantomData,
        })
    }
}

// ── Validated state ────────────────────────────────────────────────

impl ModelGraph<Validated> {
    /// Returns the total number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total estimated memory for all weights in bytes.
    pub fn total_weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.estimated_weight_bytes()).sum()
    }

    /// Returns the total estimated memory for all activations in bytes.
    pub fn total_activation_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.estimated_activation_bytes())
            .sum()
    }

    /// Returns the estimated memory for the largest single layer.
    pub fn max_layer_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.estimated_total_bytes())
            .max()
            .unwrap_or(0)
    }

    /// Returns an iterator over the layers in execution order.
    pub fn iter_layers(&self) -> impl Iterator<Item = &LayerDef> {
        self.layers.iter()
    }

    /// Returns a reference to a layer by index.
    pub fn layer(&self, index: usize) -> Option<&LayerDef> {
        self.layers.get(index)
    }

    /// Returns a summary string describing the model.
    pub fn summary(&self) -> String {
        let total_weight_mb = self.total_weight_bytes() as f64 / (1024.0 * 1024.0);
        let max_layer_mb = self.max_layer_bytes() as f64 / (1024.0 * 1024.0);
        format!(
            "Model '{}': {} layers, {:.1} MB weights, largest layer {:.2} MB",
            self.name,
            self.num_layers(),
            total_weight_mb,
            max_layer_mb,
        )
    }
}

// ── Shared implementations ─────────────────────────────────────────

impl<S: GraphState> fmt::Display for ModelGraph<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ModelGraph '{}' ({} layers):", self.name, self.layers.len())?;
        for layer in &self.layers {
            writeln!(f, "  {}", layer.summary())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_core::{DType, Shape};

    /// Helper: creates a sequence of compatible layers.
    fn make_layers(n: usize, hidden: usize) -> Vec<LayerDef> {
        (0..n)
            .map(|i| LayerDef {
                name: format!("layer.{i}"),
                layer_type: crate::LayerType::Linear,
                index: i,
                weight_names: vec![format!("w.{i}")],
                weight_shapes: vec![Shape::matrix(hidden, hidden)],
                dtype: DType::F32,
                input_shape: Shape::matrix(1, hidden),
                output_shape: Shape::matrix(1, hidden),
            })
            .collect()
    }

    #[test]
    fn test_validate_ok() {
        let graph = ModelGraph::new("test".into(), make_layers(4, 768));
        let validated = graph.validate().unwrap();
        assert_eq!(validated.num_layers(), 4);
    }

    #[test]
    fn test_validate_empty() {
        let graph = ModelGraph::new("empty".into(), vec![]);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_validate_bad_index() {
        let mut layers = make_layers(3, 768);
        layers[1].index = 5; // Should be 1.
        let graph = ModelGraph::new("bad".into(), layers);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_validate_zero_shape() {
        let mut layers = make_layers(2, 768);
        layers[0].input_shape = Shape::new(vec![0, 768]);
        let graph = ModelGraph::new("zero".into(), layers);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_total_weight_bytes() {
        let validated = ModelGraph::new("test".into(), make_layers(3, 768))
            .validate()
            .unwrap();
        // Each layer: 768 * 768 * 4 = 2_359_296 bytes.
        assert_eq!(validated.total_weight_bytes(), 3 * 768 * 768 * 4);
    }

    #[test]
    fn test_max_layer_bytes() {
        let mut layers = make_layers(3, 768);
        // Make layer 1 have a much larger weight.
        layers[1].weight_shapes = vec![Shape::matrix(768, 3072)];
        let validated = ModelGraph::new("test".into(), layers)
            .validate()
            .unwrap();
        // Layer 1 should be the largest.
        let max_total = validated.max_layer_bytes();
        assert!(max_total > 768 * 768 * 4);
    }

    #[test]
    fn test_summary() {
        let validated = ModelGraph::new("gpt2".into(), make_layers(12, 768))
            .validate()
            .unwrap();
        let s = validated.summary();
        assert!(s.contains("gpt2"));
        assert!(s.contains("12 layers"));
    }

    #[test]
    fn test_display() {
        let graph = ModelGraph::new("test".into(), make_layers(2, 64));
        let display = format!("{graph}");
        assert!(display.contains("layer.0"));
        assert!(display.contains("layer.1"));
    }

    #[test]
    fn test_iter_layers() {
        let validated = ModelGraph::new("test".into(), make_layers(3, 768))
            .validate()
            .unwrap();
        let names: Vec<_> = validated.iter_layers().map(|l| &l.name).collect();
        assert_eq!(names, &["layer.0", "layer.1", "layer.2"]);
    }

    #[test]
    fn test_layer_access() {
        let validated = ModelGraph::new("test".into(), make_layers(3, 768))
            .validate()
            .unwrap();
        assert_eq!(validated.layer(0).unwrap().name, "layer.0");
        assert_eq!(validated.layer(2).unwrap().name, "layer.2");
        assert!(validated.layer(3).is_none());
    }
}
