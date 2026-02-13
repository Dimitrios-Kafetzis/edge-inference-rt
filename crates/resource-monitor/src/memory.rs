// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! System memory monitoring via `/proc/meminfo`.
//!
//! Parses key fields from `/proc/meminfo` to determine total, available,
//! and used memory. On the RPi 4 (4 GB model), this is the primary
//! constraint for inference workloads.

use crate::MonitorError;
use std::path::Path;

/// Default path to the kernel memory info file.
const MEMINFO_PATH: &str = "/proc/meminfo";

/// System memory state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryInfo {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Available memory (as reported by the kernel) in bytes.
    ///
    /// This accounts for free memory, buffers, and reclaimable cache —
    /// it is the best estimate of how much memory a new allocation can use
    /// without causing swapping.
    pub available_bytes: u64,
    /// Memory actively used in bytes (`total - available`).
    pub used_bytes: u64,
}

impl MemoryInfo {
    /// Reads current memory information from `/proc/meminfo`.
    pub fn read() -> Result<Self, MonitorError> {
        Self::read_from(Path::new(MEMINFO_PATH))
    }

    /// Reads memory information from a specific file (for testing).
    pub(crate) fn read_from(path: &Path) -> Result<Self, MonitorError> {
        let content = std::fs::read_to_string(path).map_err(|e| MonitorError::ReadError {
            path: path.display().to_string(),
            source: e,
        })?;

        Self::parse(&content, path)
    }

    /// Parses the content of a `/proc/meminfo`-formatted string.
    pub(crate) fn parse(content: &str, source_path: &Path) -> Result<Self, MonitorError> {
        let mut total_kb: Option<u64> = None;
        let mut available_kb: Option<u64> = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            match parts[0] {
                "MemTotal:" => total_kb = parse_kb_value(parts[1], source_path)?,
                "MemAvailable:" => available_kb = parse_kb_value(parts[1], source_path)?,
                _ => {}
            }

            // Stop early once we have both values.
            if total_kb.is_some() && available_kb.is_some() {
                break;
            }
        }

        let total_kb = total_kb.ok_or_else(|| MonitorError::ParseError {
            path: source_path.display().to_string(),
            detail: "MemTotal not found".to_string(),
        })?;
        let available_kb = available_kb.ok_or_else(|| MonitorError::ParseError {
            path: source_path.display().to_string(),
            detail: "MemAvailable not found".to_string(),
        })?;

        let total_bytes = total_kb * 1024;
        let available_bytes = available_kb * 1024;
        let used_bytes = total_bytes.saturating_sub(available_bytes);

        Ok(Self {
            total_bytes,
            available_bytes,
            used_bytes,
        })
    }

    /// Returns the memory utilisation as a fraction in `[0.0, 1.0]`.
    pub fn utilisation(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.total_bytes as f64
    }

    /// Returns available memory in megabytes.
    pub fn available_mb(&self) -> u64 {
        self.available_bytes / (1024 * 1024)
    }

    /// Returns total memory in megabytes.
    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }
}

/// Parses a numeric string from `/proc/meminfo` (values are in kB).
fn parse_kb_value(s: &str, source_path: &Path) -> Result<Option<u64>, MonitorError> {
    s.parse::<u64>()
        .map(Some)
        .map_err(|_| MonitorError::ParseError {
            path: source_path.display().to_string(),
            detail: format!("expected integer kB value, got '{s}'"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MEMINFO: &str = "\
MemTotal:        3884292 kB
MemFree:          218456 kB
MemAvailable:    2456780 kB
Buffers:          123456 kB
Cached:          1987654 kB
SwapCached:            0 kB
Active:          1234567 kB
Inactive:         876543 kB
";

    #[test]
    fn test_parse_meminfo() {
        let info = MemoryInfo::parse(SAMPLE_MEMINFO, Path::new("/proc/meminfo")).unwrap();
        assert_eq!(info.total_bytes, 3884292 * 1024);
        assert_eq!(info.available_bytes, 2456780 * 1024);
        assert_eq!(info.used_bytes, (3884292 - 2456780) * 1024);
    }

    #[test]
    fn test_total_mb() {
        let info = MemoryInfo::parse(SAMPLE_MEMINFO, Path::new("/proc/meminfo")).unwrap();
        // 3884292 kB ≈ 3793 MB
        assert_eq!(info.total_mb(), 3793);
    }

    #[test]
    fn test_utilisation() {
        let info = MemoryInfo {
            total_bytes: 4_000_000_000,
            available_bytes: 1_000_000_000,
            used_bytes: 3_000_000_000,
        };
        assert!((info.utilisation() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_utilisation_zero_total() {
        let info = MemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
            used_bytes: 0,
        };
        assert!((info.utilisation() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_read_from_file() {
        let dir = std::env::temp_dir().join("edge_rt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meminfo_test");
        std::fs::write(&path, SAMPLE_MEMINFO).unwrap();
        let info = MemoryInfo::read_from(&path).unwrap();
        assert_eq!(info.total_bytes, 3884292 * 1024);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_mem_available() {
        let incomplete = "MemTotal:        3884292 kB\nMemFree:          218456 kB\n";
        let result = MemoryInfo::parse(incomplete, Path::new("/proc/meminfo"));
        assert!(matches!(result, Err(MonitorError::ParseError { .. })));
    }

    #[test]
    fn test_read_real_meminfo() {
        // This test runs on the actual host — should always succeed on Linux.
        if Path::new(MEMINFO_PATH).exists() {
            let info = MemoryInfo::read().unwrap();
            assert!(info.total_bytes > 0);
            assert!(info.available_bytes <= info.total_bytes);
        }
    }
}
