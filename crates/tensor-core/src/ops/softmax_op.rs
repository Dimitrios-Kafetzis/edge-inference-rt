// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Softmax activation operation.

use crate::{DType, Tensor, TensorError, TensorView};

/// Computes softmax along the last dimension: `output[i] = exp(x[i] - max) / sum(exp(x - max))`.
///
/// Uses the numerically stable variant that subtracts the maximum value
/// before exponentiation to prevent overflow.
///
/// Both `input` and `output` must have the same shape and be `F32`.
///
/// # Errors
/// Returns [`TensorError::ShapeMismatch`] if input and output shapes differ.
/// Returns [`TensorError::UnsupportedDType`] if the dtype is not `F32`.
pub fn softmax(input: &TensorView<'_>, output: &mut Tensor) -> Result<(), TensorError> {
    if input.dtype() != DType::F32 {
        return Err(TensorError::UnsupportedDType {
            op: "softmax",
            dtype: input.dtype(),
        });
    }

    if input.shape() != output.shape() {
        return Err(TensorError::ShapeMismatch {
            op: "softmax",
            lhs: input.shape().clone(),
            rhs: output.shape().clone(),
        });
    }

    let dims = input.shape().dims();
    if dims.is_empty() {
        // Scalar: softmax of a single value is 1.0.
        output.as_f32_slice_mut()[0] = 1.0;
        return Ok(());
    }

    let last_dim = *dims.last().unwrap();
    if last_dim == 0 {
        return Ok(()); // Empty last dimension — nothing to do.
    }

    let src = input.as_f32_slice();
    let dst = output.as_f32_slice_mut();
    let num_rows = src.len() / last_dim;

    for row in 0..num_rows {
        let offset = row * last_dim;
        let row_src = &src[offset..offset + last_dim];
        let row_dst = &mut dst[offset..offset + last_dim];

        // Find max for numerical stability.
        let max_val = row_src.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum.
        let mut sum = 0.0f32;
        for (d, &s) in row_dst.iter_mut().zip(row_src.iter()) {
            let e = (s - max_val).exp();
            *d = e;
            sum += e;
        }

        // Normalize.
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for d in row_dst.iter_mut() {
                *d *= inv_sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_softmax_uniform() {
        // Softmax of equal values → uniform distribution.
        let input = Tensor::from_f32(Shape::vector(4), &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(4), DType::F32);

        softmax(&input.view(), &mut output).unwrap();

        let result = output.as_f32_slice();
        assert!(approx_eq(result, &[0.25, 0.25, 0.25, 0.25], 1e-5));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = Tensor::from_f32(Shape::vector(5), &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(5), DType::F32);

        softmax(&input.view(), &mut output).unwrap();

        let sum: f32 = output.as_f32_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_monotonic() {
        // Larger input → larger softmax output.
        let input = Tensor::from_f32(Shape::vector(3), &[1.0, 2.0, 3.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(3), DType::F32);

        softmax(&input.view(), &mut output).unwrap();

        let r = output.as_f32_slice();
        assert!(r[0] < r[1]);
        assert!(r[1] < r[2]);
    }

    #[test]
    fn test_softmax_2d() {
        // Softmax applied row-wise on a [2, 3] tensor.
        let input = Tensor::from_f32(
            Shape::matrix(2, 3),
            &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let mut output = Tensor::zeros(Shape::matrix(2, 3), DType::F32);

        softmax(&input.view(), &mut output).unwrap();

        let r = output.as_f32_slice();
        // Row 0: sums to 1.
        let sum0: f32 = r[0..3].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        // Row 1: uniform.
        assert!(approx_eq(&r[3..6], &[1.0 / 3.0; 3], 1e-5));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without the max-subtraction trick.
        let input = Tensor::from_f32(Shape::vector(3), &[1000.0, 1001.0, 1002.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(3), DType::F32);

        softmax(&input.view(), &mut output).unwrap();

        let r = output.as_f32_slice();
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(r.iter().all(|&x| x.is_finite()));
    }
}
