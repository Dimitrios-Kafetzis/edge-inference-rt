// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Inference profiling metrics.
//!
//! [`InferenceMetrics`] collects per-layer and aggregate timing, memory,
//! and throughput data. These metrics are the primary tool for comparing
//! partition strategies on the RPi 4.

use std::time::Duration;

/// Metrics for a single layer's execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerMetrics {
    /// Layer name.
    pub layer_name: String,
    /// Time spent loading weights for this layer.
    pub weight_load_duration: Duration,
    /// Time spent executing the layer computation.
    pub compute_duration: Duration,
    /// Peak memory used during this layer's execution in bytes.
    pub peak_memory_bytes: usize,
}

/// Aggregate metrics for a complete inference run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceMetrics {
    /// Total wall-clock time for the inference run.
    pub total_duration: Duration,
    /// Total time spent loading weights.
    pub total_weight_load_duration: Duration,
    /// Total time spent on computation.
    pub total_compute_duration: Duration,
    /// Peak memory usage during the entire run.
    pub peak_memory_bytes: usize,
    /// Per-layer metrics.
    pub layer_metrics: Vec<LayerMetrics>,
    /// Number of groups in the execution plan.
    pub num_groups: usize,
    /// Number of tokens generated.
    pub tokens_generated: usize,
}

impl InferenceMetrics {
    /// Creates an empty metrics container.
    pub fn new(num_groups: usize) -> Self {
        Self {
            total_duration: Duration::ZERO,
            total_weight_load_duration: Duration::ZERO,
            total_compute_duration: Duration::ZERO,
            peak_memory_bytes: 0,
            layer_metrics: Vec::new(),
            num_groups,
            tokens_generated: 0,
        }
    }

    /// Records metrics for a single layer.
    pub fn record_layer(
        &mut self,
        name: String,
        weight_load: Duration,
        compute: Duration,
        peak_mem: usize,
    ) {
        self.total_weight_load_duration += weight_load;
        self.total_compute_duration += compute;
        if peak_mem > self.peak_memory_bytes {
            self.peak_memory_bytes = peak_mem;
        }
        self.layer_metrics.push(LayerMetrics {
            layer_name: name,
            weight_load_duration: weight_load,
            compute_duration: compute,
            peak_memory_bytes: peak_mem,
        });
    }

    /// Finalises metrics with the total wall-clock time and token count.
    pub fn finalise(&mut self, total: Duration, tokens: usize) {
        self.total_duration = total;
        self.tokens_generated = tokens;
    }

    /// Returns tokens per second throughput.
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs <= 0.0 || self.tokens_generated == 0 {
            return 0.0;
        }
        self.tokens_generated as f64 / secs
    }

    /// Returns a human-readable summary suitable for CLI output.
    pub fn summary(&self) -> String {
        let peak_mb = self.peak_memory_bytes as f64 / (1024.0 * 1024.0);
        let weight_pct = if self.total_duration.as_secs_f64() > 0.0 {
            (self.total_weight_load_duration.as_secs_f64()
                / self.total_duration.as_secs_f64())
                * 100.0
        } else {
            0.0
        };

        format!(
            "Inference: {:.2}ms total, {} layers in {} groups, \
             {:.2}ms weight I/O ({:.0}%), {:.2}ms compute, \
             peak {:.2} MB, {} tokens ({:.1} tok/s)",
            self.total_duration.as_secs_f64() * 1000.0,
            self.layer_metrics.len(),
            self.num_groups,
            self.total_weight_load_duration.as_secs_f64() * 1000.0,
            weight_pct,
            self.total_compute_duration.as_secs_f64() * 1000.0,
            peak_mb,
            self.tokens_generated,
            self.tokens_per_second(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_metrics() {
        let m = InferenceMetrics::new(3);
        assert_eq!(m.tokens_per_second(), 0.0);
        assert_eq!(m.num_groups, 3);
    }

    #[test]
    fn test_record_and_finalise() {
        let mut m = InferenceMetrics::new(2);
        m.record_layer("l0".into(), Duration::from_millis(5), Duration::from_millis(10), 1000);
        m.record_layer("l1".into(), Duration::from_millis(3), Duration::from_millis(8), 2000);
        m.finalise(Duration::from_millis(30), 10);

        assert_eq!(m.layer_metrics.len(), 2);
        assert_eq!(m.peak_memory_bytes, 2000);
        assert_eq!(m.tokens_generated, 10);
        assert_eq!(m.total_weight_load_duration, Duration::from_millis(8));
        assert_eq!(m.total_compute_duration, Duration::from_millis(18));
        assert!(m.tokens_per_second() > 0.0);
    }

    #[test]
    fn test_summary_format() {
        let mut m = InferenceMetrics::new(2);
        m.record_layer("l0".into(), Duration::from_millis(1), Duration::from_millis(5), 1024 * 1024);
        m.finalise(Duration::from_millis(10), 5);

        let s = m.summary();
        assert!(s.contains("Inference:"));
        assert!(s.contains("1 layers"));
        assert!(s.contains("5 tokens"));
    }

    #[test]
    fn test_tokens_per_second() {
        let mut m = InferenceMetrics::new(1);
        m.finalise(Duration::from_secs(2), 100);
        assert!((m.tokens_per_second() - 50.0).abs() < 0.01);
    }
}
