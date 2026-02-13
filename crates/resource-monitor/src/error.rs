// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Error types for resource monitoring.

/// Errors that can occur when reading system resources.
#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    /// Failed to read a sysfs or procfs file.
    #[error("failed to read {path}: {source}")]
    ReadError {
        path: String,
        source: std::io::Error,
    },

    /// Failed to parse a numeric value from a system file.
    #[error("failed to parse value from {path}: {detail}")]
    ParseError { path: String, detail: String },

    /// The expected sysfs path does not exist (e.g., not running on RPi 4).
    #[error("sysfs path not found: {path} — is this an RPi 4?")]
    NotAvailable { path: String },
}
