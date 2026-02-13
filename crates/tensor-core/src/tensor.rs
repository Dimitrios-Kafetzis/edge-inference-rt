// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Core tensor type and view abstractions.

use crate::{DType, Shape, TensorError};

/// An owned, n-dimensional tensor stored in contiguous memory.
///
/// `Tensor` is the primary data carrier in the inference pipeline.
/// It owns its data buffer and exposes immutable views via [`TensorView`].
///
/// # Memory Layout
/// Data is stored in row-major (C) order as a flat byte buffer.
/// Typed access is provided via [`as_f32_slice`](Tensor::as_f32_slice) and friends.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    data: Vec<u8>,
}

impl Tensor {
    /// Creates a new tensor filled with zeros.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::{Tensor, Shape, DType};
    /// let t = Tensor::zeros(Shape::matrix(2, 3), DType::F32);
    /// assert_eq!(t.size_bytes(), 24); // 2 * 3 * 4 bytes
    /// ```
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let size = shape.size_bytes(dtype);
        Self {
            shape,
            dtype,
            data: vec![0u8; size],
        }
    }

    /// Creates a tensor from raw bytes.
    ///
    /// Returns an error if the buffer size does not match `shape.size_bytes(dtype)`.
    pub fn from_bytes(shape: Shape, dtype: DType, data: Vec<u8>) -> Result<Self, TensorError> {
        let expected = shape.size_bytes(dtype);
        if data.len() != expected {
            return Err(TensorError::BufferSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, dtype, data })
    }

    /// Creates a tensor from a slice of `f32` values.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::{Tensor, Shape};
    /// let t = Tensor::from_f32(Shape::vector(3), &[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(t.as_f32_slice(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn from_f32(shape: Shape, values: &[f32]) -> Result<Self, TensorError> {
        let expected_elements = shape.num_elements();
        if values.len() != expected_elements {
            return Err(TensorError::BufferSizeMismatch {
                expected: expected_elements * DType::F32.size_bytes(),
                actual: values.len() * DType::F32.size_bytes(),
            });
        }
        // SAFETY: reinterpreting &[f32] as &[u8] is safe for Copy types.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: byte_slice.to_vec(),
        })
    }

    /// Returns the tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the tensor's data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns an immutable view over this tensor's data.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: &self.shape,
            dtype: self.dtype,
            data: &self.data,
        }
    }

    /// Returns the raw byte slice backing this tensor.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable reference to the raw byte buffer.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the memory footprint of this tensor in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Interprets the buffer as a slice of `f32`.
    ///
    /// # Panics
    /// Panics if `self.dtype() != DType::F32`.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice called on {:?} tensor",
            self.dtype
        );
        // SAFETY: data was constructed from f32s and is correctly aligned
        // because Vec<u8> may not be aligned for f32 — we use from_raw_parts
        // only when we know the data originated from f32.
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.shape.num_elements(),
            )
        }
    }

    /// Interprets the buffer as a mutable slice of `f32`.
    ///
    /// # Panics
    /// Panics if `self.dtype() != DType::F32`.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice_mut called on {:?} tensor",
            self.dtype
        );
        let n = self.shape.num_elements();
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, n) }
    }

    /// Fills the tensor with a constant `f32` value.
    ///
    /// # Panics
    /// Panics if `self.dtype() != DType::F32`.
    pub fn fill_f32(&mut self, value: f32) {
        let slice = self.as_f32_slice_mut();
        slice.iter_mut().for_each(|x| *x = value);
    }
}

/// A borrowed, read-only view over a [`Tensor`]'s data.
///
/// Views are zero-copy and tied to the lifetime of the source tensor,
/// enforced by the borrow checker.
#[derive(Debug)]
pub struct TensorView<'a> {
    shape: &'a Shape,
    dtype: DType,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Creates a view from raw parts (used internally by tensor ops).
    pub fn from_parts(shape: &'a Shape, dtype: DType, data: &'a [u8]) -> Self {
        Self { shape, dtype, data }
    }

    /// Returns the shape of the viewed tensor.
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    /// Returns the data type of the viewed tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the raw byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Interprets the view as a slice of `f32`.
    ///
    /// # Panics
    /// Panics if `self.dtype() != DType::F32`.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice called on {:?} view",
            self.dtype
        );
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.shape.num_elements(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(Shape::matrix(2, 3), DType::F32);
        assert_eq!(t.size_bytes(), 24);
        assert_eq!(t.shape(), &Shape::matrix(2, 3));
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.as_f32_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(Shape::matrix(2, 3), &data).unwrap();
        assert_eq!(t.as_f32_slice(), &data);
    }

    #[test]
    fn test_from_bytes_size_mismatch() {
        let result = Tensor::from_bytes(Shape::matrix(2, 3), DType::F32, vec![0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_view_lifetime() {
        let t = Tensor::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = t.view();
        assert_eq!(v.shape(), &Shape::vector(4));
        assert_eq!(v.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fill_f32() {
        let mut t = Tensor::zeros(Shape::vector(5), DType::F32);
        t.fill_f32(3.14);
        assert!(t.as_f32_slice().iter().all(|&x| (x - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_as_f32_mut() {
        let mut t = Tensor::zeros(Shape::vector(3), DType::F32);
        let slice = t.as_f32_slice_mut();
        slice[0] = 10.0;
        slice[1] = 20.0;
        slice[2] = 30.0;
        assert_eq!(t.as_f32_slice(), &[10.0, 20.0, 30.0]);
    }
}
