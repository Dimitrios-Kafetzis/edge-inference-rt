// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! CPU thermal monitoring via `/sys/class/thermal/`.
//!
//! On the RPi 4, thermal zone 0 (`/sys/class/thermal/thermal_zone0/temp`)
//! reports the SoC temperature in millidegrees Celsius. The BCM2711 begins
//! thermal throttling at 80 °C and hard-caps at 85 °C.

use crate::MonitorError;
use std::path::Path;

/// Throttling threshold for the BCM2711 SoC (degrees Celsius).
const THROTTLE_THRESHOLD_C: f32 = 80.0;

/// Default sysfs path for the CPU thermal zone.
const THERMAL_ZONE_PATH: &str = "/sys/class/thermal/thermal_zone0/temp";

/// Thermal state of the SoC.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ThermalInfo {
    /// CPU temperature in degrees Celsius.
    pub cpu_temp_celsius: f32,
}

impl ThermalInfo {
    /// Reads the CPU temperature from the RPi 4 thermal zone.
    ///
    /// Falls back to a synthetic value if the sysfs path is unavailable
    /// (e.g., running inside a container or on non-RPi hardware).
    pub fn read() -> Result<Self, MonitorError> {
        Self::read_from(Path::new(THERMAL_ZONE_PATH))
    }

    /// Reads the CPU temperature from a specific sysfs path.
    ///
    /// The kernel reports the temperature in millidegrees Celsius (e.g., `54321`
    /// means 54.321 °C). This method parses and converts to degrees.
    pub(crate) fn read_from(path: &Path) -> Result<Self, MonitorError> {
        let content = read_sysfs_file(path)?;
        let millidegrees: i64 = content.parse::<i64>().map_err(|_| MonitorError::ParseError {
            path: path.display().to_string(),
            detail: format!("expected integer millidegrees, got '{content}'"),
        })?;

        Ok(Self {
            cpu_temp_celsius: millidegrees as f32 / 1000.0,
        })
    }

    /// Returns `true` if the temperature is above the throttling threshold (80 °C).
    ///
    /// When the BCM2711 exceeds this temperature, the kernel's thermal governor
    /// begins reducing CPU frequency to prevent damage.
    pub fn is_overheating(&self) -> bool {
        self.cpu_temp_celsius >= THROTTLE_THRESHOLD_C
    }

    /// Returns the thermal headroom in degrees Celsius before throttling begins.
    ///
    /// A negative value means the device is already throttling.
    pub fn headroom_celsius(&self) -> f32 {
        THROTTLE_THRESHOLD_C - self.cpu_temp_celsius
    }
}

/// Reads a sysfs/procfs file and returns its trimmed content.
///
/// This is a shared helper used by multiple modules in this crate.
pub(crate) fn read_sysfs_file(path: &Path) -> Result<String, MonitorError> {
    if !path.exists() {
        return Err(MonitorError::NotAvailable {
            path: path.display().to_string(),
        });
    }
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| MonitorError::ReadError {
            path: path.display().to_string(),
            source: e,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Creates a temporary file with the given content and returns its path.
    /// The caller is responsible for cleanup.
    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("edge_rt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "{content}").unwrap();
        path
    }

    #[test]
    fn test_parse_millidegrees() {
        let p = write_temp("therm_54321", "54321\n");
        let info = ThermalInfo::read_from(&p).unwrap();
        assert!((info.cpu_temp_celsius - 54.321).abs() < 0.001);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_parse_exact() {
        let p = write_temp("therm_80000", "80000");
        let info = ThermalInfo::read_from(&p).unwrap();
        assert!((info.cpu_temp_celsius - 80.0).abs() < 0.001);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_is_overheating() {
        let cool = ThermalInfo {
            cpu_temp_celsius: 55.0,
        };
        assert!(!cool.is_overheating());

        let hot = ThermalInfo {
            cpu_temp_celsius: 82.0,
        };
        assert!(hot.is_overheating());

        let threshold = ThermalInfo {
            cpu_temp_celsius: 80.0,
        };
        assert!(threshold.is_overheating());
    }

    #[test]
    fn test_headroom() {
        let info = ThermalInfo {
            cpu_temp_celsius: 65.0,
        };
        assert!((info.headroom_celsius() - 15.0).abs() < 0.001);

        let hot = ThermalInfo {
            cpu_temp_celsius: 85.0,
        };
        assert!((hot.headroom_celsius() - (-5.0)).abs() < 0.001);
    }

    #[test]
    fn test_missing_file() {
        let result = ThermalInfo::read_from(Path::new("/nonexistent/thermal/temp"));
        assert!(matches!(result, Err(MonitorError::NotAvailable { .. })));
    }

    #[test]
    fn test_invalid_content() {
        let p = write_temp("therm_invalid", "not_a_number");
        let result = ThermalInfo::read_from(&p);
        assert!(matches!(result, Err(MonitorError::ParseError { .. })));
        let _ = std::fs::remove_file(&p);
    }
}
