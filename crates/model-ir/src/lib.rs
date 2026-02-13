// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # model-ir
//!
//! A lightweight intermediate representation (IR) for transformer models.
//!
//! Rather than depending on heavy frameworks like ONNX Runtime, this crate
//! defines a minimal IR that captures what the inference runtime needs:
//!
//! - [`LayerType`] — the kind of computation each layer performs.
//! - [`LayerDef`] — a single layer's metadata, weight references, and shape info.
//! - [`ModelGraph`] — the full model as a directed acyclic graph of layers,
//!   with a **type-state pattern** (`Loaded` → `Validated`).
//! - [`ModelLoader`] — loads models from a JSON manifest + SafeTensors weight files.
//! - [`ModelManifest`] — the JSON model descriptor.
//!
//! # Supported Model Format
//! A model is stored as:
//! - `model.json` — manifest describing architecture, layer order, and weight mapping.
//! - `model.safetensors` — weights in HuggingFace SafeTensors format.
//!
//! # Example
//! ```no_run
//! use model_ir::ModelLoader;
//! use std::path::Path;
//!
//! let graph = ModelLoader::load(Path::new("./models/gpt2-small")).unwrap();
//! println!("{}", graph.summary());
//! for layer in graph.iter_layers() {
//!     println!("  {}", layer.summary());
//! }
//! ```

mod error;
pub mod graph;
mod layer;
mod loader;
pub(crate) mod manifest;

pub use error::ModelError;
pub use graph::ModelGraph;
pub use layer::{LayerDef, LayerType};
pub use loader::{ModelLoader, WeightMeta};
pub use manifest::ModelManifest;
