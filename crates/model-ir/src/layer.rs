// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Layer definitions for transformer model IR.
//!
//! Each [`LayerDef`] describes a single computation in the model graph:
//! its type, shape, weight references, and estimated memory footprint.
//! Weight data is **not** stored here — only names (keys into the SafeTensors
//! file). Weights are loaded on demand by the runtime.

use tensor_core::{DType, Shape};

/// The type of computation a layer performs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Token embedding lookup table.
    Embedding,
    /// Multi-head self-attention (QKV projection + output projection).
    SelfAttention,
    /// Feed-forward network (two linear projections with activation).
    FeedForward,
    /// Layer normalization (scale + shift).
    LayerNorm,
    /// Linear projection (e.g., final language-model head).
    Linear,
    /// Positional encoding (sinusoidal or learned).
    PositionalEncoding,
}

impl LayerType {
    /// Parses a layer type from a manifest string.
    ///
    /// Accepts both snake_case (`"self_attention"`) and common aliases
    /// (`"attn"`, `"mlp"`, `"ln"`, `"lm_head"`).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "embedding" | "embed" | "wte" => Some(Self::Embedding),
            "self_attention" | "attention" | "attn" | "mha" => Some(Self::SelfAttention),
            "feed_forward" | "feedforward" | "ffn" | "mlp" => Some(Self::FeedForward),
            "layer_norm" | "layernorm" | "ln" => Some(Self::LayerNorm),
            "linear" | "lm_head" | "head" | "proj" => Some(Self::Linear),
            "positional_encoding" | "pos_encoding" | "wpe" => Some(Self::PositionalEncoding),
            _ => None,
        }
    }

    /// Returns a human-readable label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::SelfAttention => "self_attention",
            Self::FeedForward => "feed_forward",
            Self::LayerNorm => "layer_norm",
            Self::Linear => "linear",
            Self::PositionalEncoding => "positional_encoding",
        }
    }
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata describing a single layer in the model graph.
///
/// A `LayerDef` does not own weight data — it stores references
/// (tensor names) into the SafeTensors file. Weights are loaded on demand
/// by the runtime through the memory manager.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerDef {
    /// Unique identifier for this layer (e.g., `"transformer.h.0.attn"`).
    pub name: String,
    /// The type of computation this layer performs.
    pub layer_type: LayerType,
    /// Index in the execution order (0-based).
    pub index: usize,
    /// Names of weight tensors required by this layer (keys into SafeTensors).
    pub weight_names: Vec<String>,
    /// Shapes of the weight tensors (parallel to `weight_names`).
    pub weight_shapes: Vec<Shape>,
    /// Data type for this layer's weights and computation.
    pub dtype: DType,
    /// Shape of the layer's input activation.
    pub input_shape: Shape,
    /// Shape of the layer's output activation.
    pub output_shape: Shape,
}

impl LayerDef {
    /// Estimates the memory required for this layer's weights in bytes.
    ///
    /// This is the sum of `shape.size_bytes(dtype)` for all weight tensors.
    pub fn estimated_weight_bytes(&self) -> usize {
        self.weight_shapes
            .iter()
            .map(|s| s.size_bytes(self.dtype))
            .sum()
    }

    /// Estimates the memory required for this layer's activations in bytes.
    ///
    /// Accounts for both the input and output activation buffers, since
    /// both must be live simultaneously during execution.
    pub fn estimated_activation_bytes(&self) -> usize {
        let input_bytes = self.input_shape.size_bytes(self.dtype);
        let output_bytes = self.output_shape.size_bytes(self.dtype);
        input_bytes + output_bytes
    }

    /// Total estimated memory (weights + activations) for this layer.
    pub fn estimated_total_bytes(&self) -> usize {
        self.estimated_weight_bytes() + self.estimated_activation_bytes()
    }

    /// Returns a concise summary string for display.
    pub fn summary(&self) -> String {
        let weight_kb = self.estimated_weight_bytes() as f64 / 1024.0;
        let act_kb = self.estimated_activation_bytes() as f64 / 1024.0;
        format!(
            "[{}] {} ({}) — weights: {:.1} KB, activations: {:.1} KB, {} weight tensors",
            self.index,
            self.name,
            self.layer_type,
            weight_kb,
            act_kb,
            self.weight_names.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_layer(index: usize, layer_type: LayerType) -> LayerDef {
        LayerDef {
            name: format!("layer.{index}"),
            layer_type,
            index,
            weight_names: vec!["w1".into(), "w2".into()],
            weight_shapes: vec![Shape::matrix(768, 768), Shape::vector(768)],
            dtype: DType::F32,
            input_shape: Shape::matrix(1, 768),
            output_shape: Shape::matrix(1, 768),
        }
    }

    #[test]
    fn test_weight_bytes() {
        let layer = sample_layer(0, LayerType::Linear);
        // 768*768*4 + 768*4 = 2_359_296 + 3_072 = 2_362_368
        assert_eq!(layer.estimated_weight_bytes(), 768 * 768 * 4 + 768 * 4);
    }

    #[test]
    fn test_activation_bytes() {
        let layer = sample_layer(0, LayerType::Linear);
        // input: 1*768*4 = 3072, output: 1*768*4 = 3072
        assert_eq!(layer.estimated_activation_bytes(), 3072 + 3072);
    }

    #[test]
    fn test_total_bytes() {
        let layer = sample_layer(0, LayerType::Linear);
        assert_eq!(
            layer.estimated_total_bytes(),
            layer.estimated_weight_bytes() + layer.estimated_activation_bytes()
        );
    }

    #[test]
    fn test_layer_type_from_str() {
        assert_eq!(LayerType::from_str_loose("attn"), Some(LayerType::SelfAttention));
        assert_eq!(LayerType::from_str_loose("MLP"), Some(LayerType::FeedForward));
        assert_eq!(LayerType::from_str_loose("ln"), Some(LayerType::LayerNorm));
        assert_eq!(LayerType::from_str_loose("wte"), Some(LayerType::Embedding));
        assert_eq!(LayerType::from_str_loose("lm_head"), Some(LayerType::Linear));
        assert_eq!(LayerType::from_str_loose("wpe"), Some(LayerType::PositionalEncoding));
        assert_eq!(LayerType::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_layer_type_display() {
        assert_eq!(format!("{}", LayerType::SelfAttention), "self_attention");
        assert_eq!(format!("{}", LayerType::FeedForward), "feed_forward");
    }

    #[test]
    fn test_summary() {
        let layer = sample_layer(3, LayerType::SelfAttention);
        let s = layer.summary();
        assert!(s.contains("[3]"));
        assert!(s.contains("self_attention"));
        assert!(s.contains("2 weight tensors"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let layer = sample_layer(0, LayerType::FeedForward);
        let json = serde_json::to_string(&layer).unwrap();
        let back: LayerDef = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, layer.name);
        assert_eq!(back.layer_type, layer.layer_type);
        assert_eq!(back.weight_names, layer.weight_names);
    }
}
