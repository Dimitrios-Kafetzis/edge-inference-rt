// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # tensor-core
//!
//! Lightweight tensor types and arithmetic operations for edge inference workloads.
//!
//! This crate provides:
//! - [`Tensor`] — an n-dimensional, type-safe tensor backed by `ndarray`.
//! - [`Shape`] — compile-time and runtime shape descriptors.
//! - [`DType`] — supported element data types (f32, f16, bf16, i8).
//! - Core operations: matrix multiplication, softmax, layer normalization, GELU.
//! - Optional ARM NEON SIMD acceleration on `aarch64` targets.
//!
//! # Design Goals
//! - Zero-copy views wherever possible.
//! - No heap allocation in hot paths (operations work on pre-allocated buffers).
//! - Clean error types via `thiserror`.

mod dtype;
mod error;
mod ops;
mod shape;
mod tensor;

pub use dtype::DType;
pub use error::TensorError;
pub use ops::{gelu, layer_norm, matmul, softmax};
pub use shape::Shape;
pub use tensor::{Tensor, TensorView};
