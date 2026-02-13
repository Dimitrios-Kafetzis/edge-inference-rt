// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Aggregated point-in-time system snapshot.
//!
//! A [`SystemSnapshot`] combines CPU, memory, and thermal readings into
//! a single struct. It is the primary interface consumed by the
//! `partition-planner` for resource-aware scheduling decisions.

use crate::{CpuInfo, MemoryInfo, MonitorError, ThermalInfo};
use std::time::{SystemTime, UNIX_EPOCH};

/// A complete point-in-time reading of all monitored system resources.
///
/// Used by the [`partition-planner`] to make resource-aware scheduling decisions.
/// For example, if the device is thermally throttling, the planner may switch
/// to a more conservative (lower-memory, sequential) strategy.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemSnapshot {
    /// CPU frequency and utilisation.
    pub cpu: CpuInfo,
    /// System memory state.
    pub memory: MemoryInfo,
    /// Thermal state.
    pub thermal: ThermalInfo,
    /// Unix timestamp in milliseconds when the snapshot was taken.
    pub timestamp_ms: u64,
}

impl SystemSnapshot {
    /// Captures a new snapshot by reading all system metrics.
    ///
    /// Individual subsystem failures are handled gracefully:
    /// - **Thermal**: if the thermal zone is unavailable (e.g., container),
    ///   a default temperature of 0.0 °C is used.
    /// - **CPU frequency**: if cpufreq is unavailable, defaults to 0 MHz.
    /// - **Memory**: this is the most critical reading and *must* succeed.
    pub fn capture() -> Result<Self, MonitorError> {
        let thermal = ThermalInfo::read().unwrap_or(ThermalInfo {
            cpu_temp_celsius: 0.0,
        });

        let cpu = CpuInfo::read().unwrap_or(CpuInfo {
            frequency_mhz: 0,
            max_frequency_mhz: 0,
            load_per_core: 0.0,
            online_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
        });

        let memory = MemoryInfo::read()?;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(Self {
            cpu,
            memory,
            thermal,
            timestamp_ms,
        })
    }

    /// Returns a summary string suitable for logging or CLI display.
    ///
    /// # Example output
    /// ```text
    /// System: CPU 1800 MHz (4 cores, 100%), Mem 2456/3793 MB avail (35% used), Temp 54.3°C (OK)
    /// ```
    pub fn summary(&self) -> String {
        let throttle_status = if self.cpu.is_throttled() {
            "THROTTLED".to_string()
        } else {
            format!("{}%", (self.cpu.frequency_ratio() * 100.0) as u32)
        };

        let thermal_status = if self.thermal.is_overheating() {
            "OVERHEATING"
        } else {
            "OK"
        };

        format!(
            "System: CPU {} MHz ({} cores, {throttle_status}), \
             Mem {}/{} MB avail ({:.0}% used), \
             Temp {:.1}°C ({thermal_status})",
            self.cpu.frequency_mhz,
            self.cpu.online_cores,
            self.memory.available_mb(),
            self.memory.total_mb(),
            self.memory.utilisation() * 100.0,
            self.thermal.cpu_temp_celsius,
        )
    }

    /// Returns `true` if the system is under any resource pressure that
    /// should trigger the planner to use a more conservative strategy.
    ///
    /// Conditions:
    /// - Thermal throttling is active.
    /// - Available memory is below 256 MB.
    /// - CPU load per core exceeds 0.9.
    pub fn is_resource_constrained(&self) -> bool {
        self.thermal.is_overheating()
            || self.memory.available_mb() < 256
            || self.cpu.load_per_core > 0.9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(temp: f32, avail_mb: u64, total_mb: u64, freq: u32) -> SystemSnapshot {
        SystemSnapshot {
            cpu: CpuInfo {
                frequency_mhz: freq,
                max_frequency_mhz: 1800,
                load_per_core: 0.3,
                online_cores: 4,
            },
            memory: MemoryInfo {
                total_bytes: total_mb * 1024 * 1024,
                available_bytes: avail_mb * 1024 * 1024,
                used_bytes: (total_mb - avail_mb) * 1024 * 1024,
            },
            thermal: ThermalInfo {
                cpu_temp_celsius: temp,
            },
            timestamp_ms: 1700000000000,
        }
    }

    #[test]
    fn test_capture_on_linux() {
        // Should succeed on any Linux system.
        if std::path::Path::new("/proc/meminfo").exists() {
            let snap = SystemSnapshot::capture().unwrap();
            assert!(snap.memory.total_bytes > 0);
            assert!(snap.timestamp_ms > 0);
        }
    }

    #[test]
    fn test_summary_format() {
        let snap = sample_snapshot(54.3, 2456, 3793, 1800);
        let summary = snap.summary();
        assert!(summary.contains("1800 MHz"));
        assert!(summary.contains("4 cores"));
        assert!(summary.contains("54.3°C"));
        assert!(summary.contains("OK"));
    }

    #[test]
    fn test_not_constrained() {
        let snap = sample_snapshot(55.0, 2000, 4000, 1800);
        assert!(!snap.is_resource_constrained());
    }

    #[test]
    fn test_constrained_thermal() {
        let snap = sample_snapshot(82.0, 2000, 4000, 1800);
        assert!(snap.is_resource_constrained());
    }

    #[test]
    fn test_constrained_memory() {
        let snap = sample_snapshot(55.0, 200, 4000, 1800); // Only 200 MB available
        assert!(snap.is_resource_constrained());
    }

    #[test]
    fn test_constrained_cpu() {
        let mut snap = sample_snapshot(55.0, 2000, 4000, 1800);
        snap.cpu.load_per_core = 0.95;
        assert!(snap.is_resource_constrained());
    }

    #[test]
    fn test_summary_throttled() {
        let snap = sample_snapshot(55.0, 2000, 4000, 1200); // 1200 < 1800
        let summary = snap.summary();
        assert!(summary.contains("THROTTLED"));
    }
}
