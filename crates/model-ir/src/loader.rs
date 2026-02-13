// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Model loading from manifest + SafeTensors files.
//!
//! The loader reads a model directory containing:
//! - `model.json` — the architecture manifest (see [`ModelManifest`]).
//! - `model.safetensors` — the weight file in HuggingFace SafeTensors format.
//!
//! Weight *data* is **not** loaded into memory here. The loader only reads
//! the SafeTensors header to extract tensor shapes and data types, which are
//! used to build the [`ModelGraph`]. Actual weight data is loaded on demand
//! by the runtime via memory-mapped I/O.

use crate::manifest::parse_dtype;
use crate::{graph, LayerDef, LayerType, ModelError, ModelGraph, ModelManifest};
use std::collections::HashMap;
use std::path::Path;
use tensor_core::{DType, Shape};

/// Default manifest filename.
const MANIFEST_FILE: &str = "model.json";

/// Default SafeTensors filename.
const WEIGHTS_FILE: &str = "model.safetensors";

/// Metadata for a single tensor extracted from the SafeTensors header.
#[derive(Debug, Clone)]
pub struct WeightMeta {
    /// Tensor name (key in the SafeTensors file).
    pub name: String,
    /// Shape of the tensor.
    pub shape: Shape,
    /// Data type.
    pub dtype: DType,
    /// Size in bytes.
    pub size_bytes: usize,
}

/// Loads a model from disk into a validated [`ModelGraph`].
///
/// # Example
/// ```no_run
/// use model_ir::ModelLoader;
/// use std::path::Path;
///
/// let graph = ModelLoader::load(Path::new("./models/gpt2-small")).unwrap();
/// println!("Loaded {} layers", graph.num_layers());
/// ```
pub struct ModelLoader;

impl ModelLoader {
    /// Loads and validates a model from the given directory.
    ///
    /// Steps:
    /// 1. Parse `model.json` manifest and validate it.
    /// 2. Read the SafeTensors header to extract weight tensor metadata.
    /// 3. Build [`LayerDef`]s by combining manifest info with weight shapes.
    /// 4. Construct and validate the [`ModelGraph`].
    pub fn load(model_dir: &Path) -> Result<ModelGraph<graph::Validated>, ModelError> {
        // 1. Load and validate manifest.
        let manifest = Self::load_manifest(model_dir)?;
        manifest.validate()?;

        // 2. Extract weight metadata from SafeTensors.
        let weight_meta = Self::read_weight_metadata(model_dir)?;

        // 3. Build layers.
        let layers = Self::build_layers(&manifest, &weight_meta)?;

        // 4. Construct and validate graph.
        let graph = ModelGraph::new(manifest.name.clone(), layers);
        graph.validate()
    }

    /// Loads a model from a manifest and a pre-built weight metadata map.
    ///
    /// Useful for testing without actual SafeTensors files.
    pub fn from_manifest_and_meta(
        manifest: &ModelManifest,
        weight_meta: &HashMap<String, WeightMeta>,
    ) -> Result<ModelGraph<graph::Validated>, ModelError> {
        manifest.validate()?;
        let layers = Self::build_layers(manifest, weight_meta)?;
        let graph = ModelGraph::new(manifest.name.clone(), layers);
        graph.validate()
    }

    /// Parses the manifest file from the model directory.
    fn load_manifest(model_dir: &Path) -> Result<ModelManifest, ModelError> {
        let manifest_path = model_dir.join(MANIFEST_FILE);
        ModelManifest::from_file(&manifest_path)
    }

    /// Reads the SafeTensors header to extract tensor shapes and dtypes.
    ///
    /// Uses memory-mapped I/O to avoid loading the full weight file.
    fn read_weight_metadata(
        model_dir: &Path,
    ) -> Result<HashMap<String, WeightMeta>, ModelError> {
        let weights_path = model_dir.join(WEIGHTS_FILE);
        let file = std::fs::File::open(&weights_path).map_err(|e| {
            ModelError::SafeTensorsError(format!(
                "cannot open '{}': {e}",
                weights_path.display()
            ))
        })?;

        // Memory-map the file for zero-copy header parsing.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            ModelError::SafeTensorsError(format!("mmap failed: {e}"))
        })?;

        // Deserialise the SafeTensors header (this only parses metadata,
        // not the actual tensor data).
        let tensors = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| {
            ModelError::SafeTensorsError(format!("SafeTensors parse error: {e}"))
        })?;

        let mut meta = HashMap::new();
        for (name, view) in tensors.tensors() {
            let shape = Shape::new(view.shape().to_vec());
            let dtype = convert_safetensor_dtype(view.dtype())?;
            let size_bytes = shape.size_bytes(dtype);
            meta.insert(
                name.clone(),
                WeightMeta {
                    name: name.to_string(),
                    shape,
                    dtype,
                    size_bytes,
                },
            );
        }

        Ok(meta)
    }

    /// Converts manifest entries into layer definitions, using weight
    /// metadata to determine shapes.
    fn build_layers(
        manifest: &ModelManifest,
        weight_meta: &HashMap<String, WeightMeta>,
    ) -> Result<Vec<LayerDef>, ModelError> {
        let dtype = parse_dtype(&manifest.dtype).ok_or_else(|| ModelError::InvalidLayer {
            layer: manifest.name.clone(),
            detail: format!("unsupported dtype '{}'", manifest.dtype),
        })?;

        let hidden = manifest.hidden_size;
        let seq_len = 1; // Default to batch=1, seq=1 for shape inference.

        let mut layers = Vec::with_capacity(manifest.layers.len());

        for (i, ml) in manifest.layers.iter().enumerate() {
            let layer_type = LayerType::from_str_loose(&ml.layer_type).ok_or_else(|| {
                ModelError::InvalidLayer {
                    layer: ml.name.clone(),
                    detail: format!("unrecognised layer type '{}'", ml.layer_type),
                }
            })?;

            // Collect weight shapes from SafeTensors metadata.
            let mut weight_shapes = Vec::with_capacity(ml.weights.len());
            for wname in &ml.weights {
                let meta = weight_meta.get(wname).ok_or_else(|| {
                    ModelError::WeightNotFound {
                        name: wname.clone(),
                    }
                })?;
                weight_shapes.push(meta.shape.clone());
            }

            // Infer input/output shapes from the layer type and architecture.
            let (input_shape, output_shape) =
                infer_activation_shapes(&layer_type, hidden, seq_len, manifest.vocab_size);

            layers.push(LayerDef {
                name: ml.name.clone(),
                layer_type,
                index: i,
                weight_names: ml.weights.clone(),
                weight_shapes,
                dtype,
                input_shape,
                output_shape,
            });
        }

        Ok(layers)
    }
}

/// Infers input and output activation shapes based on layer type and
/// architecture parameters.
///
/// Returns `(input_shape, output_shape)`.
fn infer_activation_shapes(
    layer_type: &LayerType,
    hidden: usize,
    seq_len: usize,
    vocab_size: usize,
) -> (Shape, Shape) {
    match layer_type {
        LayerType::Embedding => {
            // Input: token IDs [seq_len] → output: [seq_len, hidden].
            (Shape::vector(seq_len), Shape::matrix(seq_len, hidden))
        }
        LayerType::PositionalEncoding => {
            // Input & output: [seq_len, hidden].
            (
                Shape::matrix(seq_len, hidden),
                Shape::matrix(seq_len, hidden),
            )
        }
        LayerType::SelfAttention => {
            // Input & output: [seq_len, hidden].
            (
                Shape::matrix(seq_len, hidden),
                Shape::matrix(seq_len, hidden),
            )
        }
        LayerType::FeedForward => {
            // Input & output: [seq_len, hidden].
            (
                Shape::matrix(seq_len, hidden),
                Shape::matrix(seq_len, hidden),
            )
        }
        LayerType::LayerNorm => {
            // Input & output: [seq_len, hidden].
            (
                Shape::matrix(seq_len, hidden),
                Shape::matrix(seq_len, hidden),
            )
        }
        LayerType::Linear => {
            // Final projection: [seq_len, hidden] → [seq_len, vocab_size].
            (
                Shape::matrix(seq_len, hidden),
                Shape::matrix(seq_len, vocab_size),
            )
        }
    }
}

/// Converts a SafeTensors `Dtype` to our [`DType`].
fn convert_safetensor_dtype(st_dtype: safetensors::Dtype) -> Result<DType, ModelError> {
    match st_dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::I8 => Ok(DType::I8),
        other => Err(ModelError::SafeTensorsError(format!(
            "unsupported SafeTensors dtype: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a synthetic weight metadata map matching our test manifest.
    fn make_test_weight_meta(hidden: usize) -> HashMap<String, WeightMeta> {
        let mut meta = HashMap::new();

        let add = |meta: &mut HashMap<String, WeightMeta>, name: &str, shape: Shape| {
            let size_bytes = shape.size_bytes(DType::F32);
            meta.insert(
                name.to_string(),
                WeightMeta {
                    name: name.to_string(),
                    shape,
                    dtype: DType::F32,
                    size_bytes,
                },
            );
        };

        let intermediate = hidden * 4;

        // Embedding.
        add(&mut meta, "wte.weight", Shape::matrix(50257, hidden));
        // Positional encoding.
        add(&mut meta, "wpe.weight", Shape::matrix(1024, hidden));

        // Two transformer blocks.
        for b in 0..2 {
            add(
                &mut meta,
                &format!("h.{b}.ln_1.weight"),
                Shape::vector(hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.ln_1.bias"),
                Shape::vector(hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.attn.c_attn.weight"),
                Shape::matrix(hidden, 3 * hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.attn.c_attn.bias"),
                Shape::vector(3 * hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.attn.c_proj.weight"),
                Shape::matrix(hidden, hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.attn.c_proj.bias"),
                Shape::vector(hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.ln_2.weight"),
                Shape::vector(hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.ln_2.bias"),
                Shape::vector(hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.mlp.c_fc.weight"),
                Shape::matrix(hidden, intermediate),
            );
            add(
                &mut meta,
                &format!("h.{b}.mlp.c_fc.bias"),
                Shape::vector(intermediate),
            );
            add(
                &mut meta,
                &format!("h.{b}.mlp.c_proj.weight"),
                Shape::matrix(intermediate, hidden),
            );
            add(
                &mut meta,
                &format!("h.{b}.mlp.c_proj.bias"),
                Shape::vector(hidden),
            );
        }

        // Final layer norm.
        add(&mut meta, "ln_f.weight", Shape::vector(hidden));
        add(&mut meta, "ln_f.bias", Shape::vector(hidden));

        // LM head.
        add(&mut meta, "lm_head.weight", Shape::matrix(hidden, 50257));

        meta
    }

    fn sample_manifest() -> ModelManifest {
        let json = r#"{
            "name": "gpt2-test",
            "architecture": "gpt2",
            "num_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "vocab_size": 50257,
            "max_sequence_length": 1024,
            "dtype": "f32",
            "layers": [
                { "name": "wte", "layer_type": "embedding", "weights": ["wte.weight"] },
                { "name": "wpe", "layer_type": "positional_encoding", "weights": ["wpe.weight"] },
                { "name": "h.0.ln_1", "layer_type": "layer_norm", "weights": ["h.0.ln_1.weight", "h.0.ln_1.bias"] },
                { "name": "h.0.attn", "layer_type": "self_attention", "weights": ["h.0.attn.c_attn.weight", "h.0.attn.c_attn.bias", "h.0.attn.c_proj.weight", "h.0.attn.c_proj.bias"] },
                { "name": "h.0.ln_2", "layer_type": "layer_norm", "weights": ["h.0.ln_2.weight", "h.0.ln_2.bias"] },
                { "name": "h.0.mlp", "layer_type": "feed_forward", "weights": ["h.0.mlp.c_fc.weight", "h.0.mlp.c_fc.bias", "h.0.mlp.c_proj.weight", "h.0.mlp.c_proj.bias"] },
                { "name": "h.1.ln_1", "layer_type": "layer_norm", "weights": ["h.1.ln_1.weight", "h.1.ln_1.bias"] },
                { "name": "h.1.attn", "layer_type": "self_attention", "weights": ["h.1.attn.c_attn.weight", "h.1.attn.c_attn.bias", "h.1.attn.c_proj.weight", "h.1.attn.c_proj.bias"] },
                { "name": "h.1.ln_2", "layer_type": "layer_norm", "weights": ["h.1.ln_2.weight", "h.1.ln_2.bias"] },
                { "name": "h.1.mlp", "layer_type": "feed_forward", "weights": ["h.1.mlp.c_fc.weight", "h.1.mlp.c_fc.bias", "h.1.mlp.c_proj.weight", "h.1.mlp.c_proj.bias"] },
                { "name": "ln_f", "layer_type": "layer_norm", "weights": ["ln_f.weight", "ln_f.bias"] },
                { "name": "lm_head", "layer_type": "linear", "weights": ["lm_head.weight"] }
            ]
        }"#;
        ModelManifest::from_json(json).unwrap()
    }

    #[test]
    fn test_build_from_manifest_and_meta() {
        let manifest = sample_manifest();
        let meta = make_test_weight_meta(64);
        let graph = ModelLoader::from_manifest_and_meta(&manifest, &meta).unwrap();

        assert_eq!(graph.num_layers(), 12);
        assert_eq!(graph.name, "gpt2-test");
    }

    #[test]
    fn test_layer_weight_bytes() {
        let manifest = sample_manifest();
        let meta = make_test_weight_meta(64);
        let graph = ModelLoader::from_manifest_and_meta(&manifest, &meta).unwrap();

        // The attention layer has 4 weight tensors:
        // c_attn.weight [64, 192], c_attn.bias [192], c_proj.weight [64, 64], c_proj.bias [64]
        let attn = graph.layer(3).unwrap();
        assert_eq!(attn.layer_type, LayerType::SelfAttention);
        assert_eq!(attn.weight_shapes.len(), 4);
        assert!(attn.estimated_weight_bytes() > 0);
    }

    #[test]
    fn test_total_weight_bytes() {
        let manifest = sample_manifest();
        let meta = make_test_weight_meta(64);
        let graph = ModelLoader::from_manifest_and_meta(&manifest, &meta).unwrap();
        assert!(graph.total_weight_bytes() > 0);
    }

    #[test]
    fn test_missing_weight_tensor() {
        let manifest = sample_manifest();
        let meta = HashMap::new(); // Empty — all weights missing.
        let result = ModelLoader::from_manifest_and_meta(&manifest, &meta);
        assert!(matches!(result, Err(ModelError::WeightNotFound { .. })));
    }

    #[test]
    fn test_infer_shapes_embedding() {
        let (input, output) = infer_activation_shapes(&LayerType::Embedding, 768, 1, 50257);
        assert_eq!(input, Shape::vector(1));
        assert_eq!(output, Shape::matrix(1, 768));
    }

    #[test]
    fn test_infer_shapes_linear_head() {
        let (input, output) = infer_activation_shapes(&LayerType::Linear, 768, 1, 50257);
        assert_eq!(input, Shape::matrix(1, 768));
        assert_eq!(output, Shape::matrix(1, 50257));
    }

    #[test]
    fn test_infer_shapes_self_attention() {
        let (input, output) = infer_activation_shapes(&LayerType::SelfAttention, 768, 1, 50257);
        assert_eq!(input, Shape::matrix(1, 768));
        assert_eq!(output, Shape::matrix(1, 768));
    }

    #[test]
    fn test_graph_summary() {
        let manifest = sample_manifest();
        let meta = make_test_weight_meta(64);
        let graph = ModelLoader::from_manifest_and_meta(&manifest, &meta).unwrap();
        let summary = graph.summary();
        assert!(summary.contains("gpt2-test"));
        assert!(summary.contains("12 layers"));
    }
}
