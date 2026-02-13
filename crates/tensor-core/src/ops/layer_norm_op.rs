// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Layer normalization operation.

use crate::{DType, Tensor, TensorError, TensorView};

/// Applies layer normalization over the last dimension:
///
/// `output = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// # Arguments
/// * `input`  — the input tensor (any rank, normalised over last dim).
/// * `gamma`  — scale parameter, 1-D with length equal to the last dimension.
/// * `beta`   — shift parameter, 1-D with length equal to the last dimension.
/// * `eps`    — small constant for numerical stability (typically 1e-5).
/// * `output` — pre-allocated output tensor (same shape as `input`).
///
/// # Errors
/// Returns errors if shapes are incompatible or dtype is not F32.
pub fn layer_norm(
    input: &TensorView<'_>,
    gamma: &TensorView<'_>,
    beta: &TensorView<'_>,
    eps: f32,
    output: &mut Tensor,
) -> Result<(), TensorError> {
    // Validate dtype.
    if input.dtype() != DType::F32 {
        return Err(TensorError::UnsupportedDType {
            op: "layer_norm",
            dtype: input.dtype(),
        });
    }

    // Validate shapes.
    if input.shape() != output.shape() {
        return Err(TensorError::ShapeMismatch {
            op: "layer_norm (input vs output)",
            lhs: input.shape().clone(),
            rhs: output.shape().clone(),
        });
    }

    let dims = input.shape().dims();
    if dims.is_empty() {
        return Err(TensorError::ShapeMismatch {
            op: "layer_norm (scalar input)",
            lhs: input.shape().clone(),
            rhs: gamma.shape().clone(),
        });
    }

    let last_dim = *dims.last().unwrap();

    // gamma and beta must be 1-D with length == last_dim.
    if gamma.shape().rank() != 1 || gamma.shape().num_elements() != last_dim {
        return Err(TensorError::ShapeMismatch {
            op: "layer_norm (gamma)",
            lhs: gamma.shape().clone(),
            rhs: input.shape().clone(),
        });
    }
    if beta.shape().rank() != 1 || beta.shape().num_elements() != last_dim {
        return Err(TensorError::ShapeMismatch {
            op: "layer_norm (beta)",
            lhs: beta.shape().clone(),
            rhs: input.shape().clone(),
        });
    }

    let src = input.as_f32_slice();
    let dst = output.as_f32_slice_mut();
    let g = gamma.as_f32_slice();
    let b = beta.as_f32_slice();
    let num_rows = src.len() / last_dim;

    for row in 0..num_rows {
        let offset = row * last_dim;
        let row_src = &src[offset..offset + last_dim];
        let row_dst = &mut dst[offset..offset + last_dim];

        // Compute mean.
        let mean: f32 = row_src.iter().sum::<f32>() / last_dim as f32;

        // Compute variance.
        let var: f32 =
            row_src.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;

        // Normalize, scale, and shift.
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..last_dim {
            row_dst[j] = g[j] * (row_src[j] - mean) * inv_std + b[j];
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
    fn test_layer_norm_basic() {
        // Input: [1, 2, 3, 4, 5] with gamma=1, beta=0 → should be zero-mean, unit-variance.
        let input = Tensor::from_f32(Shape::vector(5), &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let gamma = Tensor::from_f32(Shape::vector(5), &[1.0; 5]).unwrap();
        let beta = Tensor::from_f32(Shape::vector(5), &[0.0; 5]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(5), DType::F32);

        layer_norm(&input.view(), &gamma.view(), &beta.view(), 1e-5, &mut output).unwrap();

        let r = output.as_f32_slice();
        // Mean of normalised output should be ~0.
        let mean: f32 = r.iter().sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5);
        // Variance should be ~1.
        let var: f32 = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 5.0;
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_gamma_beta() {
        // Constant input → output should equal beta (since normalised constant = 0).
        let input = Tensor::from_f32(Shape::vector(3), &[5.0, 5.0, 5.0]).unwrap();
        let gamma = Tensor::from_f32(Shape::vector(3), &[2.0, 2.0, 2.0]).unwrap();
        let beta = Tensor::from_f32(Shape::vector(3), &[1.0, 2.0, 3.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(3), DType::F32);

        layer_norm(&input.view(), &gamma.view(), &beta.view(), 1e-5, &mut output).unwrap();

        // (5 - 5) / sqrt(0 + eps) * 2 + beta ≈ beta
        assert!(approx_eq(output.as_f32_slice(), &[1.0, 2.0, 3.0], 1e-2));
    }

    #[test]
    fn test_layer_norm_2d() {
        // [2, 3] tensor: normalisation is per-row.
        let input = Tensor::from_f32(
            Shape::matrix(2, 3),
            &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        )
        .unwrap();
        let gamma = Tensor::from_f32(Shape::vector(3), &[1.0; 3]).unwrap();
        let beta = Tensor::from_f32(Shape::vector(3), &[0.0; 3]).unwrap();
        let mut output = Tensor::zeros(Shape::matrix(2, 3), DType::F32);

        layer_norm(&input.view(), &gamma.view(), &beta.view(), 1e-5, &mut output).unwrap();

        let r = output.as_f32_slice();
        // Each row should have mean ≈ 0.
        let mean0: f32 = r[0..3].iter().sum::<f32>() / 3.0;
        let mean1: f32 = r[3..6].iter().sum::<f32>() / 3.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_shape_mismatch() {
        let input = Tensor::from_f32(Shape::vector(3), &[1.0, 2.0, 3.0]).unwrap();
        let gamma = Tensor::from_f32(Shape::vector(4), &[1.0; 4]).unwrap(); // Wrong size.
        let beta = Tensor::from_f32(Shape::vector(3), &[0.0; 3]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(3), DType::F32);

        let result = layer_norm(&input.view(), &gamma.view(), &beta.view(), 1e-5, &mut output);
        assert!(result.is_err());
    }
}
