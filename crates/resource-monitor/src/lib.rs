// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! # resource-monitor
//!
//! Reads RPi 4–specific system metrics from `/sys/` and `/proc/` to
//! provide real-time resource awareness to the inference runtime.
//!
//! # Monitored Metrics
//! - **CPU temperature** — thermal throttling detection.
//! - **CPU frequency** — current vs. maximum clock speed.
//! - **Available memory** — free + reclaimable memory.
//! - **CPU load** — per-core load average from `/proc/loadavg`.
//!
//! All reads are non-blocking and suitable for periodic polling.
//!
//! # Graceful Degradation
//! When running outside an RPi 4 (e.g., in a container or on x86),
//! subsystems that rely on missing sysfs paths return sensible defaults
//! rather than hard errors. Only memory info (from `/proc/meminfo`) is
//! required — everything else degrades gracefully.
//!
//! # Example
//! ```no_run
//! use resource_monitor::SystemSnapshot;
//!
//! let snap = SystemSnapshot::capture().expect("failed to read system state");
//! println!("{}", snap.summary());
//! if snap.is_resource_constrained() {
//!     println!("⚠ System under pressure — consider a conservative strategy.");
//! }
//! ```

mod cpu;
mod error;
mod memory;
mod snapshot;
pub(crate) mod thermal;

pub use cpu::CpuInfo;
pub use error::MonitorError;
pub use memory::MemoryInfo;
pub use snapshot::SystemSnapshot;
pub use thermal::ThermalInfo;

/// Captures a point-in-time snapshot of all monitored system resources.
///
/// This is a convenience wrapper around [`SystemSnapshot::capture()`].
pub fn snapshot() -> Result<SystemSnapshot, MonitorError> {
    SystemSnapshot::capture()
}
