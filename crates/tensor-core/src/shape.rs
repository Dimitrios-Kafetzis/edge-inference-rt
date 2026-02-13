// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Tensor shape descriptors and dimension utilities.

use std::fmt;

/// Describes the dimensionality of a [`crate::Tensor`].
///
/// Shapes are immutable once created and provide convenience methods for
/// computing strides, total element counts, and broadcasting compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from the given dimensions.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Shape;
    /// let s = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(s.rank(), 3);
    /// assert_eq!(s.num_elements(), 24);
    /// ```
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Creates a scalar shape (rank 0).
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Creates a 1-D shape.
    pub fn vector(len: usize) -> Self {
        Self { dims: vec![len] }
    }

    /// Creates a 2-D shape (matrix).
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    ///
    /// For a scalar shape (rank 0), returns 1.
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Returns the dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the size of a specific dimension, or `None` if out of bounds.
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.dims.get(index).copied()
    }

    /// Computes the memory footprint in bytes for a given [`crate::DType`].
    pub fn size_bytes(&self, dtype: super::DType) -> usize {
        self.num_elements() * dtype.size_bytes()
    }

    /// Computes row-major (C-order) strides for this shape.
    ///
    /// The stride for dimension `i` is the number of elements to skip
    /// in the flat buffer to advance one step along that dimension.
    pub fn strides(&self) -> Vec<usize> {
        let rank = self.dims.len();
        if rank == 0 {
            return vec![];
        }
        let mut strides = vec![0usize; rank];
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Returns `true` if two shapes are broadcast-compatible.
    ///
    /// Shapes are compatible when, aligning dimensions from the right,
    /// each pair is either equal or one of them is 1.
    pub fn is_broadcast_compatible(&self, other: &Shape) -> bool {
        let a = &self.dims;
        let b = &other.dims;
        let mut ai = a.len();
        let mut bi = b.len();
        while ai > 0 && bi > 0 {
            ai -= 1;
            bi -= 1;
            if a[ai] != b[bi] && a[ai] != 1 && b[bi] != 1 {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the shapes are compatible for a matrix multiply:
    /// `self` is `[..., M, K]` and `other` is `[..., K, N]`.
    pub fn is_matmul_compatible(&self, other: &Shape) -> bool {
        if self.rank() < 2 || other.rank() < 2 {
            return false;
        }
        let k_lhs = self.dims[self.rank() - 1];
        let k_rhs = other.dims[other.rank() - 2];
        k_lhs == k_rhs
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Convenience: `Shape::from(vec![2, 3])`.
impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

/// Convenience: `Shape::from(&[2, 3][..])`.
impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
        assert!(s.strides().is_empty());
    }

    #[test]
    fn test_vector_shape() {
        let s = Shape::vector(5);
        assert_eq!(s.rank(), 1);
        assert_eq!(s.num_elements(), 5);
        assert_eq!(s.strides(), vec![1]);
    }

    #[test]
    fn test_matrix_shape() {
        let s = Shape::matrix(3, 4);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.num_elements(), 12);
        assert_eq!(s.strides(), vec![4, 1]);
        assert_eq!(s.size_bytes(DType::F32), 48);
    }

    #[test]
    fn test_3d_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcast_compatible() {
        let a = Shape::new(vec![1, 3]);
        let b = Shape::new(vec![4, 3]);
        assert!(a.is_broadcast_compatible(&b));

        let c = Shape::new(vec![4, 1]);
        assert!(a.is_broadcast_compatible(&c));

        let d = Shape::new(vec![4, 2]);
        assert!(!a.is_broadcast_compatible(&d));
    }

    #[test]
    fn test_matmul_compatible() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(4, 5);
        assert!(a.is_matmul_compatible(&b));

        let c = Shape::matrix(5, 5);
        assert!(!a.is_matmul_compatible(&c));
    }

    #[test]
    fn test_display() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(format!("{s}"), "[2, 3, 4]");
    }

    #[test]
    fn test_size_bytes() {
        let s = Shape::new(vec![10, 20]);
        assert_eq!(s.size_bytes(DType::F32), 800);
        assert_eq!(s.size_bytes(DType::F16), 400);
        assert_eq!(s.size_bytes(DType::I8), 200);
    }

    #[test]
    fn test_from_conversions() {
        let s1: Shape = vec![2, 3].into();
        let s2: Shape = (&[2, 3][..]).into();
        assert_eq!(s1, s2);
    }
}
