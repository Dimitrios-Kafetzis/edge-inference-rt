// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Tensor arithmetic operations.
//!
//! Each operation works on pre-allocated output buffers to avoid heap
//! allocations in the inference hot path. ARM NEON–optimised kernels
//! are selected at compile time via `#[cfg(target_arch = "aarch64")]`.

mod gelu_op;
mod layer_norm_op;
mod matmul_op;
mod softmax_op;

pub use gelu_op::gelu;
pub use layer_norm_op::layer_norm;
pub use matmul_op::matmul;
pub use softmax_op::softmax;
