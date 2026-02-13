// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Weight loading from SafeTensors files with memory-mapped I/O.
//!
//! [`WeightLoader`] provides two modes:
//!
//! 1. **File-backed** — opens `model.safetensors` via mmap and extracts
//!    tensor data on demand. This is the production path on RPi 4.
//! 2. **Synthetic** — generates random weight data for testing and
//!    benchmarking without requiring actual model files.

use memory_manager::{BufferGuard, MemoryPool};
use model_ir::LayerDef;
use std::path::{Path, PathBuf};
use tensor_core::Tensor;

/// Default SafeTensors filename.
const WEIGHTS_FILE: &str = "model.safetensors";

/// Loads weight tensors from SafeTensors files on demand.
///
/// Uses `memmap2` for zero-copy access to weight data. Weights are loaded
/// into buffers allocated from the [`MemoryPool`], ensuring budget compliance.
pub struct WeightLoader {
    /// Path to the model directory containing SafeTensors files.
    model_dir: PathBuf,
    /// Memory-mapped SafeTensors file (opened once, reused).
    mmap: Option<memmap2::Mmap>,
}

impl WeightLoader {
    /// Creates a new weight loader for the given model directory.
    ///
    /// If the SafeTensors file exists, it is memory-mapped immediately.
    /// If it does not exist, the loader operates in synthetic mode.
    pub fn new(model_dir: PathBuf) -> Result<Self, super::RuntimeError> {
        let weights_path = model_dir.join(WEIGHTS_FILE);

        let mmap = if weights_path.exists() {
            let file = std::fs::File::open(&weights_path).map_err(|e| {
                super::RuntimeError::WeightLoadError {
                    layer: "init".into(),
                    detail: format!("cannot open '{}': {e}", weights_path.display()),
                }
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
                super::RuntimeError::WeightLoadError {
                    layer: "init".into(),
                    detail: format!("mmap failed: {e}"),
                }
            })?;
            tracing::info!(
                "weight loader: mmap'd {} ({:.2} MB)",
                weights_path.display(),
                mmap.len() as f64 / (1024.0 * 1024.0),
            );
            Some(mmap)
        } else {
            tracing::warn!(
                "weight loader: '{}' not found, using synthetic mode",
                weights_path.display(),
            );
            None
        };

        Ok(Self { model_dir, mmap })
    }

    /// Creates a weight loader in synthetic mode (no file needed).
    pub fn synthetic() -> Self {
        Self {
            model_dir: PathBuf::from("<synthetic>"),
            mmap: None,
        }
    }

    /// Returns `true` if operating in file-backed mode.
    pub fn is_file_backed(&self) -> bool {
        self.mmap.is_some()
    }

    /// Returns the model directory path.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Loads all weight tensors for a given layer.
    ///
    /// In file-backed mode, reads tensor data from the mmap'd SafeTensors
    /// file. In synthetic mode, returns zero-filled tensors with the
    /// correct shapes.
    pub fn load_layer_weights(
        &self,
        layer: &LayerDef,
    ) -> Result<Vec<Tensor>, super::RuntimeError> {
        if let Some(mmap) = &self.mmap {
            self.load_from_safetensors(layer, mmap)
        } else {
            self.load_synthetic(layer)
        }
    }

    /// Loads weight tensors into pool-allocated buffer guards.
    ///
    /// This is used when the runtime wants pool-managed memory (for budget
    /// tracking) rather than owned `Tensor`s.
    pub fn load_layer_buffers(
        &self,
        layer: &LayerDef,
        pool: &MemoryPool,
    ) -> Result<Vec<BufferGuard>, super::RuntimeError> {
        let mut guards = Vec::with_capacity(layer.weight_names.len());

        for (i, shape) in layer.weight_shapes.iter().enumerate() {
            let size = shape.size_bytes(layer.dtype);
            let mut guard = pool.allocate(size).map_err(|e| {
                super::RuntimeError::WeightLoadError {
                    layer: layer.name.clone(),
                    detail: format!(
                        "pool allocation failed for weight '{}': {e}",
                        layer.weight_names[i]
                    ),
                }
            })?;

            // Fill with real data or zeros.
            if let Some(mmap) = &self.mmap {
                self.copy_tensor_data(
                    &layer.weight_names[i],
                    mmap,
                    guard.as_mut_slice(),
                )?;
            }
            // In synthetic mode, the buffer is already zeroed by the pool.

            guards.push(guard);
        }

        Ok(guards)
    }

    /// Prefetches weights for multiple layers (used by speculative strategy).
    ///
    /// Returns buffer guards that hold the memory until dropped.
    pub fn prefetch_weights(
        &self,
        layers: &[&LayerDef],
        pool: &MemoryPool,
    ) -> Result<Vec<BufferGuard>, super::RuntimeError> {
        let mut all_guards = Vec::new();
        for layer in layers {
            let guards = self.load_layer_buffers(layer, pool)?;
            all_guards.extend(guards);
        }
        Ok(all_guards)
    }

    // ── Private helpers ────────────────────────────────────────

    /// Loads weight tensors from the mmap'd SafeTensors file.
    fn load_from_safetensors(
        &self,
        layer: &LayerDef,
        mmap: &memmap2::Mmap,
    ) -> Result<Vec<Tensor>, super::RuntimeError> {
        let st = safetensors::SafeTensors::deserialize(mmap).map_err(|e| {
            super::RuntimeError::WeightLoadError {
                layer: layer.name.clone(),
                detail: format!("SafeTensors parse error: {e}"),
            }
        })?;

        let mut tensors = Vec::with_capacity(layer.weight_names.len());

        for (i, wname) in layer.weight_names.iter().enumerate() {
            let view = st.tensor(wname).map_err(|e| {
                super::RuntimeError::WeightLoadError {
                    layer: layer.name.clone(),
                    detail: format!("tensor '{}' not found: {e}", wname),
                }
            })?;

            let expected_shape = &layer.weight_shapes[i];
            let data = view.data().to_vec();
            let tensor = Tensor::from_bytes(
                expected_shape.clone(),
                layer.dtype,
                data,
            )
            .map_err(|e| super::RuntimeError::WeightLoadError {
                layer: layer.name.clone(),
                detail: format!("tensor '{}' shape mismatch: {e}", wname),
            })?;

            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Generates zero-filled synthetic tensors.
    fn load_synthetic(
        &self,
        layer: &LayerDef,
    ) -> Result<Vec<Tensor>, super::RuntimeError> {
        let tensors: Vec<Tensor> = layer
            .weight_shapes
            .iter()
            .map(|shape| Tensor::zeros(shape.clone(), layer.dtype))
            .collect();
        Ok(tensors)
    }

    /// Copies tensor data from mmap into a pre-allocated buffer.
    fn copy_tensor_data(
        &self,
        tensor_name: &str,
        mmap: &memmap2::Mmap,
        dest: &mut [u8],
    ) -> Result<(), super::RuntimeError> {
        let st = safetensors::SafeTensors::deserialize(mmap).map_err(|e| {
            super::RuntimeError::WeightLoadError {
                layer: tensor_name.to_string(),
                detail: format!("SafeTensors parse error: {e}"),
            }
        })?;

        let view = st.tensor(tensor_name).map_err(|e| {
            super::RuntimeError::WeightLoadError {
                layer: tensor_name.to_string(),
                detail: format!("tensor not found: {e}"),
            }
        })?;

        let data = view.data();
        let copy_len = data.len().min(dest.len());
        dest[..copy_len].copy_from_slice(&data[..copy_len]);

        Ok(())
    }
}

impl std::fmt::Debug for WeightLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightLoader")
            .field("model_dir", &self.model_dir)
            .field("file_backed", &self.is_file_backed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_core::{DType, Shape};

    fn sample_layer() -> LayerDef {
        LayerDef {
            name: "test_layer".into(),
            layer_type: model_ir::LayerType::Linear,
            index: 0,
            weight_names: vec!["w1".into(), "w2".into()],
            weight_shapes: vec![Shape::matrix(64, 64), Shape::vector(64)],
            dtype: DType::F32,
            input_shape: Shape::matrix(1, 64),
            output_shape: Shape::matrix(1, 64),
        }
    }

    #[test]
    fn test_synthetic_mode() {
        let loader = WeightLoader::synthetic();
        assert!(!loader.is_file_backed());

        let weights = loader.load_layer_weights(&sample_layer()).unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].shape(), &Shape::matrix(64, 64));
        assert_eq!(weights[1].shape(), &Shape::vector(64));
    }

    #[test]
    fn test_synthetic_buffer_loading() {
        let loader = WeightLoader::synthetic();
        let pool = MemoryPool::new(memory_manager::MemoryBudget::from_mb(10));
        let layer = sample_layer();

        let guards = loader.load_layer_buffers(&layer, &pool).unwrap();
        assert_eq!(guards.len(), 2);
        // w1: 64*64*4 = 16384 bytes.
        assert_eq!(guards[0].size_bytes(), 64 * 64 * 4);
        // w2: 64*4 = 256 bytes.
        assert_eq!(guards[1].size_bytes(), 64 * 4);

        // Pool should track the allocations.
        assert_eq!(pool.allocated_bytes(), 16384 + 256);

        // Drop and verify return.
        drop(guards);
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn test_prefetch_multiple_layers() {
        let loader = WeightLoader::synthetic();
        let pool = MemoryPool::new(memory_manager::MemoryBudget::from_mb(10));

        let l0 = sample_layer();
        let mut l1 = sample_layer();
        l1.name = "layer_1".into();
        l1.index = 1;

        let layers: Vec<&LayerDef> = vec![&l0, &l1];
        let guards = loader.prefetch_weights(&layers, &pool).unwrap();
        // 2 layers × 2 weights each = 4 guards.
        assert_eq!(guards.len(), 4);
    }

    #[test]
    fn test_new_missing_file() {
        let dir = std::env::temp_dir().join("edge_rt_test_no_model");
        std::fs::create_dir_all(&dir).ok();
        let loader = WeightLoader::new(dir).unwrap();
        assert!(!loader.is_file_backed());
    }
}
