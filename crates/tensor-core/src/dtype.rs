// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Supported tensor element data types.

/// Enumerates the numeric types a [`crate::Tensor`] can hold.
///
/// The runtime uses `DType` to decide memory layout, alignment, and which
/// compute kernels to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 floating point.
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// 8-bit signed integer (for quantised weights).
    I8,
}

impl DType {
    /// Returns the size of a single element in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
        }
    }

    /// Returns a human-readable label for this data type.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I8 => "i8",
        }
    }
}
