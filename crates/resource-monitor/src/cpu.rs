// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! CPU frequency and utilisation monitoring.
//!
//! Reads CPU state from:
//! - `/sys/devices/system/cpu/cpu*/cpufreq/` — current and max frequency.
//! - `/sys/devices/system/cpu/online` — online core count.
//! - `/proc/loadavg` — system load average (1-minute) as a proxy for
//!   utilisation without requiring two-sample delta computation.
//!
//! # RPi 4 specifics
//! The BCM2711 has 4× Cortex-A72 cores running at up to 1.8 GHz (default)
//! or 2.0 GHz (overclocked). When thermally throttled, the kernel reduces
//! `scaling_cur_freq` below `scaling_max_freq`.

use crate::thermal::read_sysfs_file;
use crate::MonitorError;
use std::path::Path;

/// Base sysfs path for CPU information.
const CPU_BASE: &str = "/sys/devices/system/cpu";

/// CPU state information.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuInfo {
    /// Current CPU frequency in MHz (from core 0; representative on RPi 4
    /// since all cores share the same frequency domain).
    pub frequency_mhz: u32,
    /// Maximum CPU frequency in MHz.
    pub max_frequency_mhz: u32,
    /// 1-minute load average divided by online core count, clamped to `[0.0, 1.0]`.
    ///
    /// This is a rough utilisation proxy. For precise per-core utilisation,
    /// use a polling loop with `/proc/stat` deltas.
    pub load_per_core: f32,
    /// Number of online cores.
    pub online_cores: u32,
}

impl CpuInfo {
    /// Reads current CPU information from sysfs and procfs.
    pub fn read() -> Result<Self, MonitorError> {
        let cur_freq_path = format!("{CPU_BASE}/cpu0/cpufreq/scaling_cur_freq");
        let max_freq_path = format!("{CPU_BASE}/cpu0/cpufreq/scaling_max_freq");

        let frequency_mhz = read_freq(Path::new(&cur_freq_path))?;
        let max_frequency_mhz = read_freq(Path::new(&max_freq_path))?;

        let online_cores = read_online_cores()?;
        let load_per_core = read_load_per_core(online_cores)?;

        Ok(Self {
            frequency_mhz,
            max_frequency_mhz,
            load_per_core,
            online_cores,
        })
    }

    /// Returns `true` if the CPU is being frequency-throttled.
    ///
    /// Throttling is detected when the current frequency is below the
    /// maximum, which typically happens due to thermal limits.
    pub fn is_throttled(&self) -> bool {
        self.frequency_mhz < self.max_frequency_mhz
    }

    /// Returns the frequency ratio: `current / max`, in `[0.0, 1.0]`.
    pub fn frequency_ratio(&self) -> f32 {
        if self.max_frequency_mhz == 0 {
            return 0.0;
        }
        self.frequency_mhz as f32 / self.max_frequency_mhz as f32
    }

    /// Creates a `CpuInfo` for testing or when sysfs is unavailable.
    #[cfg(test)]
    pub(crate) fn synthetic(freq: u32, max_freq: u32, cores: u32, load: f32) -> Self {
        Self {
            frequency_mhz: freq,
            max_frequency_mhz: max_freq,
            load_per_core: load,
            online_cores: cores,
        }
    }
}

/// Reads a CPU frequency value from sysfs (reported in kHz, returned as MHz).
fn read_freq(path: &Path) -> Result<u32, MonitorError> {
    let content = read_sysfs_file(path)?;
    let khz: u64 = content.parse::<u64>().map_err(|_| MonitorError::ParseError {
        path: path.display().to_string(),
        detail: format!("expected integer kHz value, got '{content}'"),
    })?;
    Ok((khz / 1000) as u32)
}

/// Determines the number of online CPU cores.
///
/// Tries `/sys/devices/system/cpu/online` first (e.g., `"0-3"` → 4 cores),
/// then falls back to counting `cpu[0-9]+` directories, and finally to
/// `std::thread::available_parallelism()`.
fn read_online_cores() -> Result<u32, MonitorError> {
    let online_path_str = format!("{CPU_BASE}/online");
    let online_path = Path::new(&online_path_str);
    if let Ok(content) = read_sysfs_file(online_path) {
        if let Some(count) = parse_cpu_range(&content) {
            return Ok(count);
        }
    }

    // Fallback: count cpu directories.
    if let Ok(entries) = std::fs::read_dir(CPU_BASE) {
        let count = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with("cpu") && name[3..].chars().all(|c| c.is_ascii_digit())
            })
            .count();
        if count > 0 {
            return Ok(count as u32);
        }
    }

    // Last resort: available_parallelism.
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .map_err(|e| MonitorError::ReadError {
            path: CPU_BASE.to_string(),
            source: e,
        })
}

/// Parses a CPU range string like `"0-3"` → 4, `"0-7"` → 8, `"0"` → 1, `"0,2-3"` → 3.
fn parse_cpu_range(s: &str) -> Option<u32> {
    let mut total = 0u32;
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start_s, end_s)) = part.split_once('-') {
            let start: u32 = start_s.trim().parse().ok()?;
            let end: u32 = end_s.trim().parse().ok()?;
            total += end - start + 1;
        } else {
            let _: u32 = part.parse().ok()?;
            total += 1;
        }
    }
    if total > 0 {
        Some(total)
    } else {
        None
    }
}

/// Reads the 1-minute load average from `/proc/loadavg` and normalises
/// it per core, clamping to `[0.0, 1.0]`.
fn read_load_per_core(online_cores: u32) -> Result<f32, MonitorError> {
    let path = Path::new("/proc/loadavg");
    if !path.exists() {
        // Not on Linux — return a default.
        return Ok(0.0);
    }

    let content = std::fs::read_to_string(path).map_err(|e| MonitorError::ReadError {
        path: path.display().to_string(),
        source: e,
    })?;

    // Format: "0.35 0.28 0.22 1/234 5678"
    // We want the first field (1-minute load average).
    let load_1m: f32 = content
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let cores = online_cores.max(1) as f32;
    let per_core = (load_1m / cores).clamp(0.0, 1.0);
    Ok(per_core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_range_simple() {
        assert_eq!(parse_cpu_range("0-3"), Some(4));
        assert_eq!(parse_cpu_range("0-7"), Some(8));
        assert_eq!(parse_cpu_range("0"), Some(1));
    }

    #[test]
    fn test_parse_cpu_range_complex() {
        assert_eq!(parse_cpu_range("0,2-3"), Some(3));
        assert_eq!(parse_cpu_range("0-1,3-5"), Some(5));
    }

    #[test]
    fn test_parse_cpu_range_invalid() {
        assert_eq!(parse_cpu_range(""), None);
        assert_eq!(parse_cpu_range("abc"), None);
    }

    #[test]
    fn test_is_throttled() {
        let normal = CpuInfo::synthetic(1800, 1800, 4, 0.5);
        assert!(!normal.is_throttled());

        let throttled = CpuInfo::synthetic(1200, 1800, 4, 0.5);
        assert!(throttled.is_throttled());
    }

    #[test]
    fn test_frequency_ratio() {
        let info = CpuInfo::synthetic(1500, 1800, 4, 0.5);
        let ratio = info.frequency_ratio();
        assert!((ratio - (1500.0 / 1800.0)).abs() < 0.001);
    }

    #[test]
    fn test_frequency_ratio_zero_max() {
        let info = CpuInfo::synthetic(0, 0, 4, 0.5);
        assert!((info.frequency_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_online_cores_fallback() {
        // This should succeed on any Linux system or via available_parallelism.
        let cores = read_online_cores().unwrap();
        assert!(cores >= 1);
    }

    #[test]
    fn test_load_per_core() {
        // On the test host /proc/loadavg should exist.
        if Path::new("/proc/loadavg").exists() {
            let load = read_load_per_core(4).unwrap();
            assert!(load >= 0.0);
            assert!(load <= 1.0);
        }
    }
}
