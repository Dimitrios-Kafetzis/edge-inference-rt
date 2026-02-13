// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Memory budget configuration and parsing.
//!
//! A [`MemoryBudget`] represents a hard memory ceiling for the inference
//! runtime. It supports human-readable string parsing for CLI ergonomics.

use crate::MemoryError;
use std::fmt;

/// A hard memory ceiling for the inference runtime.
///
/// # Parsing
/// Supports human-readable strings with SI-style suffixes:
/// - `"512M"` or `"512MB"` → 512 × 1024² bytes
/// - `"1G"` or `"1GB"` → 1 × 1024³ bytes
/// - `"2048K"` or `"2048KB"` → 2048 × 1024 bytes
/// - `"1073741824"` → raw byte count
///
/// # Examples
/// ```
/// use memory_manager::MemoryBudget;
///
/// let b = MemoryBudget::from_mb(512);
/// assert_eq!(b.as_mb(), 512);
///
/// let b = MemoryBudget::parse("1G").unwrap();
/// assert_eq!(b.as_mb(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryBudget {
    /// Budget in bytes.
    bytes: usize,
}

impl MemoryBudget {
    /// Creates a budget from a byte count.
    pub fn from_bytes(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Creates a budget from megabytes.
    pub fn from_mb(mb: usize) -> Self {
        Self {
            bytes: mb * 1024 * 1024,
        }
    }

    /// Creates a budget from gigabytes.
    pub fn from_gb(gb: usize) -> Self {
        Self {
            bytes: gb * 1024 * 1024 * 1024,
        }
    }

    /// Returns the budget in bytes.
    pub fn as_bytes(&self) -> usize {
        self.bytes
    }

    /// Returns the budget in megabytes (truncated).
    pub fn as_mb(&self) -> usize {
        self.bytes / (1024 * 1024)
    }

    /// Parses a human-readable budget string.
    ///
    /// Accepted formats: `"512M"`, `"512MB"`, `"1G"`, `"1GB"`, `"2048K"`,
    /// `"2048KB"`, or a plain byte count like `"1073741824"`.
    /// Case-insensitive.
    pub fn parse(s: &str) -> Result<Self, MemoryError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(MemoryError::ZeroSizedAllocation);
        }

        let s_upper = s.to_uppercase();

        // Try to split into numeric part and suffix.
        let (num_str, multiplier) = if s_upper.ends_with("GB") {
            (&s[..s.len() - 2], 1024 * 1024 * 1024)
        } else if s_upper.ends_with('G') {
            (&s[..s.len() - 1], 1024 * 1024 * 1024)
        } else if s_upper.ends_with("MB") {
            (&s[..s.len() - 2], 1024 * 1024)
        } else if s_upper.ends_with('M') {
            (&s[..s.len() - 1], 1024 * 1024)
        } else if s_upper.ends_with("KB") {
            (&s[..s.len() - 2], 1024)
        } else if s_upper.ends_with('K') {
            (&s[..s.len() - 1], 1024)
        } else if s_upper.ends_with('B') {
            (&s[..s.len() - 1], 1)
        } else {
            // Plain number — treat as bytes.
            (s, 1)
        };

        let num_str = num_str.trim();
        let value: usize = num_str.parse().map_err(|_| MemoryError::PoolCorruption(
            format!("invalid budget string: '{s}' — expected a number followed by an optional suffix (M, G, K)")
        ))?;

        let bytes = value.checked_mul(multiplier).ok_or_else(|| {
            MemoryError::PoolCorruption(format!("budget overflow: '{s}'"))
        })?;

        if bytes == 0 {
            return Err(MemoryError::ZeroSizedAllocation);
        }

        Ok(Self { bytes })
    }
}

impl fmt::Display for MemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.bytes >= 1024 * 1024 * 1024 && self.bytes % (1024 * 1024 * 1024) == 0 {
            write!(f, "{} GB", self.bytes / (1024 * 1024 * 1024))
        } else if self.bytes >= 1024 * 1024 && self.bytes % (1024 * 1024) == 0 {
            write!(f, "{} MB", self.bytes / (1024 * 1024))
        } else if self.bytes >= 1024 && self.bytes % 1024 == 0 {
            write!(f, "{} KB", self.bytes / 1024)
        } else {
            write!(f, "{} B", self.bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_mb() {
        let b = MemoryBudget::from_mb(512);
        assert_eq!(b.as_bytes(), 512 * 1024 * 1024);
        assert_eq!(b.as_mb(), 512);
    }

    #[test]
    fn test_from_gb() {
        let b = MemoryBudget::from_gb(2);
        assert_eq!(b.as_mb(), 2048);
    }

    #[test]
    fn test_parse_megabytes() {
        assert_eq!(MemoryBudget::parse("512M").unwrap().as_mb(), 512);
        assert_eq!(MemoryBudget::parse("512MB").unwrap().as_mb(), 512);
        assert_eq!(MemoryBudget::parse("512m").unwrap().as_mb(), 512);
        assert_eq!(MemoryBudget::parse("512mb").unwrap().as_mb(), 512);
    }

    #[test]
    fn test_parse_gigabytes() {
        assert_eq!(MemoryBudget::parse("1G").unwrap().as_mb(), 1024);
        assert_eq!(MemoryBudget::parse("1GB").unwrap().as_mb(), 1024);
        assert_eq!(MemoryBudget::parse("2g").unwrap().as_mb(), 2048);
    }

    #[test]
    fn test_parse_kilobytes() {
        assert_eq!(MemoryBudget::parse("1024K").unwrap().as_bytes(), 1024 * 1024);
        assert_eq!(MemoryBudget::parse("1024KB").unwrap().as_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_parse_raw_bytes() {
        let b = MemoryBudget::parse("1048576").unwrap();
        assert_eq!(b.as_mb(), 1);
    }

    #[test]
    fn test_parse_with_whitespace() {
        assert_eq!(MemoryBudget::parse("  512M  ").unwrap().as_mb(), 512);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(MemoryBudget::parse("").is_err());
        assert!(MemoryBudget::parse("abc").is_err());
        assert!(MemoryBudget::parse("0M").is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", MemoryBudget::from_gb(1)), "1 GB");
        assert_eq!(format!("{}", MemoryBudget::from_mb(512)), "512 MB");
        assert_eq!(format!("{}", MemoryBudget::from_bytes(2048)), "2 KB");
        assert_eq!(format!("{}", MemoryBudget::from_bytes(100)), "100 B");
    }

    #[test]
    fn test_serde_roundtrip() {
        let b = MemoryBudget::from_mb(256);
        let json = serde_json::to_string(&b).unwrap();
        let back: MemoryBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(b, back);
    }
}
