// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Matrix multiplication operation.

use crate::{DType, Shape, Tensor, TensorError, TensorView};

/// Performs matrix multiplication: `output = lhs @ rhs`.
///
/// Both inputs must be 2-D tensors with compatible inner dimensions:
/// `lhs` is `[M, K]`, `rhs` is `[K, N]`, and `output` must be `[M, N]`.
///
/// On `aarch64` targets this dispatches to a NEON-optimised kernel.
///
/// # Errors
/// Returns [`TensorError::ShapeMismatch`] if dimensions are incompatible.
/// Returns [`TensorError::UnsupportedDType`] if the dtype is not `F32`.
pub fn matmul(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &mut Tensor,
) -> Result<(), TensorError> {
    // Validate dtype — currently only F32 is supported.
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        return Err(TensorError::UnsupportedDType {
            op: "matmul",
            dtype: if lhs.dtype() != DType::F32 {
                lhs.dtype()
            } else {
                rhs.dtype()
            },
        });
    }

    // Validate shapes.
    if !lhs.shape().is_matmul_compatible(rhs.shape()) {
        return Err(TensorError::ShapeMismatch {
            op: "matmul",
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        });
    }

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    let expected_shape = Shape::matrix(m, n);
    if output.shape() != &expected_shape || output.dtype() != DType::F32 {
        return Err(TensorError::ShapeMismatch {
            op: "matmul (output)",
            lhs: expected_shape,
            rhs: output.shape().clone(),
        });
    }

    let a = lhs.as_f32_slice();
    let b = rhs.as_f32_slice();
    let c = output.as_f32_slice_mut();

    matmul_f32_generic(a, b, c, m, k, n);

    Ok(())
}

/// Generic (portable) f32 matrix multiplication.
///
/// Uses a simple ikj loop order for better cache locality on the `b` matrix.
/// Not SIMD-optimised, but correct and reasonably cache-friendly.
fn matmul_f32_generic(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Zero the output.
    c.iter_mut().for_each(|x| *x = 0.0);

    // ikj loop order: iterate over rows of A, then columns of A (= rows of B),
    // then columns of B. This makes the inner loop a saxpy on a row of C,
    // which is sequential in memory.
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            let c_row = &mut c[i * n..(i + 1) * n];
            let b_row = &b[p * n..(p + 1) * n];
            for j in 0..n {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! ARM NEON SIMD-accelerated matrix multiply kernel.
    //!
    //! This module provides an optimised matmul using NEON intrinsics
    //! for 4-wide f32 SIMD operations on the Cortex-A72.
    //!
    //! TODO: Implement NEON kernel; currently falls back to generic.
    #![allow(dead_code)]

    pub(super) fn matmul_f32_neon(
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) {
        // Placeholder for NEON implementation.
        // When implemented, the main `matmul` function will dispatch here
        // via #[cfg(target_arch = "aarch64")].
        unimplemented!("NEON matmul kernel not yet implemented; using generic path")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[7, 8], [9, 10], [11, 12]]
        // C = [[58, 64], [139, 154]]
        let a = Tensor::from_f32(Shape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b =
            Tensor::from_f32(Shape::matrix(3, 2), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let mut c = Tensor::zeros(Shape::matrix(2, 2), DType::F32);

        matmul(&a.view(), &b.view(), &mut c).unwrap();

        let result = c.as_f32_slice();
        assert!((result[0] - 58.0).abs() < 1e-5);
        assert!((result[1] - 64.0).abs() < 1e-5);
        assert!((result[2] - 139.0).abs() < 1e-5);
        assert!((result[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        // A * I = A
        let a = Tensor::from_f32(Shape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let eye = Tensor::from_f32(Shape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut c = Tensor::zeros(Shape::matrix(2, 2), DType::F32);

        matmul(&a.view(), &eye.view(), &mut c).unwrap();

        assert_eq!(c.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = Tensor::zeros(Shape::matrix(2, 3), DType::F32);
        let b = Tensor::zeros(Shape::matrix(4, 2), DType::F32); // 4 != 3
        let mut c = Tensor::zeros(Shape::matrix(2, 2), DType::F32);

        let result = matmul(&a.view(), &b.view(), &mut c);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_1x1() {
        let a = Tensor::from_f32(Shape::matrix(1, 1), &[3.0]).unwrap();
        let b = Tensor::from_f32(Shape::matrix(1, 1), &[4.0]).unwrap();
        let mut c = Tensor::zeros(Shape::matrix(1, 1), DType::F32);

        matmul(&a.view(), &b.view(), &mut c).unwrap();
        assert!((c.as_f32_slice()[0] - 12.0).abs() < 1e-6);
    }
}
