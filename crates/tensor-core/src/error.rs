// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for tensor operations.

use crate::Shape;

/// Errors that can occur during tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// The provided buffer size does not match the expected size for the given shape and dtype.
    #[error("shape mismatch: expected {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// Two tensors have incompatible shapes for the requested operation.
    #[error("incompatible shapes for {op}: {lhs:?} vs {rhs:?}")]
    ShapeMismatch {
        op: &'static str,
        lhs: Shape,
        rhs: Shape,
    },

    /// The requested data type is not supported for this operation.
    #[error("unsupported dtype {dtype:?} for operation {op}")]
    UnsupportedDType {
        op: &'static str,
        dtype: crate::DType,
    },

    /// A numeric computation failed (e.g., NaN or overflow).
    #[error("numeric error in {op}: {detail}")]
    Numeric {
        op: &'static str,
        detail: String,
    },
}
