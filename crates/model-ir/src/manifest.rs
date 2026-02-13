// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! JSON model manifest parsing.
//!
//! The manifest (`model.json`) describes the model's architecture and maps
//! layer names to weight tensor names in the SafeTensors file.
//!
//! # Format
//! ```json
//! {
//!   "name": "gpt2-small",
//!   "architecture": "gpt2",
//!   "num_layers": 12,
//!   "hidden_size": 768,
//!   "num_attention_heads": 12,
//!   "vocab_size": 50257,
//!   "max_sequence_length": 1024,
//!   "dtype": "f32",
//!   "layers": [
//!     {
//!       "name": "transformer.wte",
//!       "layer_type": "embedding",
//!       "weights": ["wte.weight"]
//!     },
//!     ...
//!   ]
//! }
//! ```

use crate::ModelError;
use std::path::Path;

/// Top-level model manifest, deserialized from `model.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelManifest {
    /// Human-readable model name (e.g., `"gpt2-small"`).
    pub name: String,
    /// Model architecture family (e.g., `"gpt2"`, `"llama"`).
    pub architecture: String,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_sequence_length: usize,
    /// Data type for weights and computation (e.g., `"f32"`, `"f16"`).
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Layer definitions with weight mappings.
    pub layers: Vec<ManifestLayer>,
}

fn default_dtype() -> String {
    "f32".to_string()
}

/// A single layer entry in the manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestLayer {
    /// Layer name (e.g., `"transformer.h.0.attn"`).
    pub name: String,
    /// Layer type string (e.g., `"self_attention"`, `"feed_forward"`).
    pub layer_type: String,
    /// Weight tensor names in the SafeTensors file.
    pub weights: Vec<String>,
}

impl ModelManifest {
    /// Loads a manifest from a JSON file path.
    pub fn from_file(path: &Path) -> Result<Self, ModelError> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    /// Parses a manifest from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, ModelError> {
        let manifest: Self = serde_json::from_str(json)?;
        Ok(manifest)
    }

    /// Validates that the manifest is internally consistent.
    ///
    /// Checks:
    /// - At least one layer is defined.
    /// - All layer type strings are recognised.
    /// - The dtype string is valid.
    /// - No duplicate layer names.
    /// - `num_layers` is consistent with the actual transformer layer count.
    pub fn validate(&self) -> Result<(), ModelError> {
        // Must have at least one layer.
        if self.layers.is_empty() {
            return Err(ModelError::InvalidGraph(
                "manifest contains no layers".into(),
            ));
        }

        // Validate dtype string.
        parse_dtype(&self.dtype).ok_or_else(|| {
            ModelError::InvalidLayer {
                layer: self.name.clone(),
                detail: format!("unsupported dtype '{}'", self.dtype),
            }
        })?;

        // Validate layer types and check for duplicates.
        let mut seen_names = std::collections::HashSet::new();
        for layer in &self.layers {
            if !seen_names.insert(&layer.name) {
                return Err(ModelError::InvalidLayer {
                    layer: layer.name.clone(),
                    detail: "duplicate layer name".into(),
                });
            }

            if crate::LayerType::from_str_loose(&layer.layer_type).is_none() {
                return Err(ModelError::InvalidLayer {
                    layer: layer.name.clone(),
                    detail: format!("unrecognised layer type '{}'", layer.layer_type),
                });
            }
        }

        // Check that num_layers is roughly consistent.
        // Count layers of type self_attention as "transformer layers".
        let attn_count = self
            .layers
            .iter()
            .filter(|l| {
                matches!(
                    crate::LayerType::from_str_loose(&l.layer_type),
                    Some(crate::LayerType::SelfAttention)
                )
            })
            .count();

        if attn_count > 0 && attn_count != self.num_layers {
            tracing::warn!(
                "manifest declares num_layers={} but found {} attention layers",
                self.num_layers,
                attn_count,
            );
        }

        Ok(())
    }

    /// Returns the total number of unique weight tensor names across all layers.
    pub fn total_weight_count(&self) -> usize {
        let mut unique = std::collections::HashSet::new();
        for layer in &self.layers {
            for w in &layer.weights {
                unique.insert(w.as_str());
            }
        }
        unique.len()
    }
}

/// Parses a dtype string into a [`tensor_core::DType`].
pub(crate) fn parse_dtype(s: &str) -> Option<tensor_core::DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Some(tensor_core::DType::F32),
        "f16" | "float16" => Some(tensor_core::DType::F16),
        "bf16" | "bfloat16" => Some(tensor_core::DType::BF16),
        "i8" | "int8" => Some(tensor_core::DType::I8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest_json() -> &'static str {
        r#"{
            "name": "gpt2-small",
            "architecture": "gpt2",
            "num_layers": 2,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "vocab_size": 50257,
            "max_sequence_length": 1024,
            "dtype": "f32",
            "layers": [
                {
                    "name": "transformer.wte",
                    "layer_type": "embedding",
                    "weights": ["wte.weight"]
                },
                {
                    "name": "transformer.wpe",
                    "layer_type": "positional_encoding",
                    "weights": ["wpe.weight"]
                },
                {
                    "name": "transformer.h.0.ln_1",
                    "layer_type": "layer_norm",
                    "weights": ["h.0.ln_1.weight", "h.0.ln_1.bias"]
                },
                {
                    "name": "transformer.h.0.attn",
                    "layer_type": "self_attention",
                    "weights": ["h.0.attn.c_attn.weight", "h.0.attn.c_attn.bias", "h.0.attn.c_proj.weight", "h.0.attn.c_proj.bias"]
                },
                {
                    "name": "transformer.h.0.ln_2",
                    "layer_type": "layer_norm",
                    "weights": ["h.0.ln_2.weight", "h.0.ln_2.bias"]
                },
                {
                    "name": "transformer.h.0.mlp",
                    "layer_type": "feed_forward",
                    "weights": ["h.0.mlp.c_fc.weight", "h.0.mlp.c_fc.bias", "h.0.mlp.c_proj.weight", "h.0.mlp.c_proj.bias"]
                },
                {
                    "name": "transformer.h.1.ln_1",
                    "layer_type": "layer_norm",
                    "weights": ["h.1.ln_1.weight", "h.1.ln_1.bias"]
                },
                {
                    "name": "transformer.h.1.attn",
                    "layer_type": "self_attention",
                    "weights": ["h.1.attn.c_attn.weight", "h.1.attn.c_attn.bias", "h.1.attn.c_proj.weight", "h.1.attn.c_proj.bias"]
                },
                {
                    "name": "transformer.h.1.ln_2",
                    "layer_type": "layer_norm",
                    "weights": ["h.1.ln_2.weight", "h.1.ln_2.bias"]
                },
                {
                    "name": "transformer.h.1.mlp",
                    "layer_type": "feed_forward",
                    "weights": ["h.1.mlp.c_fc.weight", "h.1.mlp.c_fc.bias", "h.1.mlp.c_proj.weight", "h.1.mlp.c_proj.bias"]
                },
                {
                    "name": "transformer.ln_f",
                    "layer_type": "layer_norm",
                    "weights": ["ln_f.weight", "ln_f.bias"]
                },
                {
                    "name": "lm_head",
                    "layer_type": "linear",
                    "weights": ["lm_head.weight"]
                }
            ]
        }"#
    }

    #[test]
    fn test_parse_manifest() {
        let m = ModelManifest::from_json(sample_manifest_json()).unwrap();
        assert_eq!(m.name, "gpt2-small");
        assert_eq!(m.architecture, "gpt2");
        assert_eq!(m.num_layers, 2);
        assert_eq!(m.hidden_size, 768);
        assert_eq!(m.layers.len(), 12);
    }

    #[test]
    fn test_validate_ok() {
        let m = ModelManifest::from_json(sample_manifest_json()).unwrap();
        m.validate().unwrap();
    }

    #[test]
    fn test_validate_empty_layers() {
        let json = r#"{
            "name": "empty", "architecture": "gpt2",
            "num_layers": 0, "hidden_size": 768,
            "num_attention_heads": 12, "vocab_size": 50257,
            "max_sequence_length": 1024, "layers": []
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_validate_bad_layer_type() {
        let json = r#"{
            "name": "bad", "architecture": "gpt2",
            "num_layers": 0, "hidden_size": 768,
            "num_attention_heads": 12, "vocab_size": 50257,
            "max_sequence_length": 1024,
            "layers": [{ "name": "l0", "layer_type": "bogus", "weights": [] }]
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_validate_duplicate_names() {
        let json = r#"{
            "name": "dup", "architecture": "gpt2",
            "num_layers": 0, "hidden_size": 768,
            "num_attention_heads": 12, "vocab_size": 50257,
            "max_sequence_length": 1024,
            "layers": [
                { "name": "l0", "layer_type": "linear", "weights": ["w1"] },
                { "name": "l0", "layer_type": "linear", "weights": ["w2"] }
            ]
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_total_weight_count() {
        let m = ModelManifest::from_json(sample_manifest_json()).unwrap();
        // Count distinct weight names across all layers.
        assert!(m.total_weight_count() > 0);
    }

    #[test]
    fn test_default_dtype() {
        let json = r#"{
            "name": "no_dtype", "architecture": "gpt2",
            "num_layers": 0, "hidden_size": 768,
            "num_attention_heads": 12, "vocab_size": 50257,
            "max_sequence_length": 1024,
            "layers": [{ "name": "l0", "layer_type": "linear", "weights": [] }]
        }"#;
        let m = ModelManifest::from_json(json).unwrap();
        assert_eq!(m.dtype, "f32");
    }

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("f32"), Some(tensor_core::DType::F32));
        assert_eq!(parse_dtype("float16"), Some(tensor_core::DType::F16));
        assert_eq!(parse_dtype("BF16"), Some(tensor_core::DType::BF16));
        assert_eq!(parse_dtype("int8"), Some(tensor_core::DType::I8));
        assert_eq!(parse_dtype("garbage"), None);
    }

    #[test]
    fn test_serde_roundtrip() {
        let m = ModelManifest::from_json(sample_manifest_json()).unwrap();
        let json = serde_json::to_string_pretty(&m).unwrap();
        let back = ModelManifest::from_json(&json).unwrap();
        assert_eq!(back.name, m.name);
        assert_eq!(back.layers.len(), m.layers.len());
    }
}
