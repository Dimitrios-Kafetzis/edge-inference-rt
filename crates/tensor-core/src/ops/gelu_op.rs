// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Gaussian Error Linear Unit (GELU) activation.

use crate::{DType, Tensor, TensorError, TensorView};

/// Coefficient `sqrt(2/π)`.
const SQRT_2_OVER_PI: f32 = 0.7978845608;

/// Cubic coefficient in the tanh approximation.
const GELU_COEFF: f32 = 0.044715;

/// Applies the GELU activation element-wise using the fast tanh approximation:
///
/// `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// This is the approximation used by GPT-2 and many other transformers.
///
/// # Errors
/// Returns [`TensorError::ShapeMismatch`] if input and output shapes differ.
/// Returns [`TensorError::UnsupportedDType`] if the dtype is not `F32`.
pub fn gelu(input: &TensorView<'_>, output: &mut Tensor) -> Result<(), TensorError> {
    if input.dtype() != DType::F32 {
        return Err(TensorError::UnsupportedDType {
            op: "gelu",
            dtype: input.dtype(),
        });
    }

    if input.shape() != output.shape() {
        return Err(TensorError::ShapeMismatch {
            op: "gelu",
            lhs: input.shape().clone(),
            rhs: output.shape().clone(),
        });
    }

    let src = input.as_f32_slice();
    let dst = output.as_f32_slice_mut();

    for (d, &x) in dst.iter_mut().zip(src.iter()) {
        *d = gelu_scalar(x);
    }

    Ok(())
}

/// Computes GELU for a single f32 value.
#[inline(always)]
fn gelu_scalar(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gelu_zero() {
        // GELU(0) = 0.
        assert!(approx_eq(gelu_scalar(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(x) ≈ x for large positive x.
        let x = 3.0;
        let y = gelu_scalar(x);
        assert!((y - x).abs() < 0.01, "GELU(3.0) should be ≈ 3.0, got {y}");
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(x) ≈ 0 for large negative x.
        let y = gelu_scalar(-3.0);
        assert!(y.abs() < 0.01, "GELU(-3.0) should be ≈ 0, got {y}");
    }

    #[test]
    fn test_gelu_known_values() {
        // Some known approximate values.
        assert!(approx_eq(gelu_scalar(1.0), 0.8412, 0.01));
        assert!(approx_eq(gelu_scalar(-1.0), -0.1588, 0.01));
        assert!(approx_eq(gelu_scalar(0.5), 0.3457, 0.01));
    }

    #[test]
    fn test_gelu_tensor() {
        let input = Tensor::from_f32(Shape::vector(4), &[0.0, 1.0, -1.0, 2.0]).unwrap();
        let mut output = Tensor::zeros(Shape::vector(4), DType::F32);

        gelu(&input.view(), &mut output).unwrap();

        let r = output.as_f32_slice();
        assert!(approx_eq(r[0], 0.0, 1e-5));
        assert!(approx_eq(r[1], 0.8412, 0.01));
        assert!(approx_eq(r[2], -0.1588, 0.01));
        assert!(approx_eq(r[3], 1.9545, 0.01));
    }

    #[test]
    fn test_gelu_shape_mismatch() {
        let input = Tensor::zeros(Shape::vector(3), DType::F32);
        let mut output = Tensor::zeros(Shape::vector(4), DType::F32);
        assert!(gelu(&input.view(), &mut output).is_err());
    }
}
