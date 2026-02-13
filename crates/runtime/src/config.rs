// Copyright (c) 2025 Dimitris Kafetzis
//
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
//
// SPDX-License-Identifier: MIT

//! Runtime configuration loaded from TOML files or constructed programmatically.
//!
//! # TOML Format
//! ```toml
//! model_path = "./models/gpt2-small"
//! memory_budget = "512M"
//! strategy = "greedy-grouping"
//! num_threads = 4
//! prefetch_ratio = 0.2
//! enable_profiling = true
//! ```

use memory_manager::MemoryBudget;
use partition_planner::{
    GreedyGrouping, PartitionStrategy, Sequential, SpeculativePrefetch,
};
use std::path::{Path, PathBuf};

/// Configuration for the inference runtime.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeConfig {
    /// Path to the model directory.
    pub model_path: PathBuf,
    /// Memory budget for inference (human-readable, e.g., `"512M"`).
    pub memory_budget: String,
    /// Partition strategy name: `"sequential"`, `"greedy-grouping"`, `"speculative-prefetch"`.
    pub strategy: String,
    /// Number of worker threads (defaults to number of online CPU cores).
    pub num_threads: Option<usize>,
    /// Prefetch ratio for speculative strategy (ignored for other strategies).
    pub prefetch_ratio: Option<f64>,
    /// Whether to enable per-layer profiling metrics.
    #[serde(default = "default_true")]
    pub enable_profiling: bool,
}

fn default_true() -> bool {
    true
}

impl RuntimeConfig {
    /// Loads configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, super::RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            super::RuntimeError::ConfigError(format!(
                "cannot read config '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_toml(&content)
    }

    /// Parses configuration from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, super::RuntimeError> {
        toml::from_str(toml_str).map_err(|e| {
            super::RuntimeError::ConfigError(format!("TOML parse error: {e}"))
        })
    }

    /// Serialises configuration to TOML.
    pub fn to_toml(&self) -> Result<String, super::RuntimeError> {
        toml::to_string_pretty(self).map_err(|e| {
            super::RuntimeError::ConfigError(format!("TOML serialise error: {e}"))
        })
    }

    /// Parses the memory budget string into a [`MemoryBudget`].
    pub fn parse_budget(&self) -> Result<MemoryBudget, super::RuntimeError> {
        MemoryBudget::parse(&self.memory_budget)
            .map_err(|e| super::RuntimeError::ConfigError(format!("invalid budget: {e}")))
    }

    /// Resolves the number of worker threads.
    pub fn resolve_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
    }

    /// Creates the partition strategy specified by this config.
    pub fn create_strategy(&self) -> Result<Box<dyn PartitionStrategy>, super::RuntimeError> {
        match self.strategy.to_lowercase().as_str() {
            "sequential" => Ok(Box::new(Sequential::new())),
            "greedy-grouping" | "greedy" => Ok(Box::new(GreedyGrouping::new())),
            "speculative-prefetch" | "speculative" => {
                let ratio = self.prefetch_ratio.unwrap_or(0.20);
                Ok(Box::new(SpeculativePrefetch::new(ratio)))
            }
            other => Err(super::RuntimeError::ConfigError(format!(
                "unknown strategy '{other}'; expected 'sequential', 'greedy-grouping', or 'speculative-prefetch'"
            ))),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models/gpt2-small"),
            memory_budget: "512M".to_string(),
            strategy: "greedy-grouping".to_string(),
            num_threads: None,
            prefetch_ratio: Some(0.2),
            enable_profiling: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let c = RuntimeConfig::default();
        assert_eq!(c.memory_budget, "512M");
        assert_eq!(c.strategy, "greedy-grouping");
        assert!(c.enable_profiling);
    }

    #[test]
    fn test_parse_budget() {
        let c = RuntimeConfig {
            memory_budget: "256M".into(),
            ..Default::default()
        };
        let b = c.parse_budget().unwrap();
        assert_eq!(b.as_mb(), 256);
    }

    #[test]
    fn test_from_toml() {
        let toml = r#"
model_path = "/tmp/model"
memory_budget = "1G"
strategy = "sequential"
num_threads = 2
enable_profiling = false
"#;
        let c = RuntimeConfig::from_toml(toml).unwrap();
        assert_eq!(c.model_path, PathBuf::from("/tmp/model"));
        assert_eq!(c.memory_budget, "1G");
        assert_eq!(c.strategy, "sequential");
        assert_eq!(c.num_threads, Some(2));
        assert!(!c.enable_profiling);
    }

    #[test]
    fn test_to_toml_roundtrip() {
        let c = RuntimeConfig::default();
        let toml = c.to_toml().unwrap();
        let back = RuntimeConfig::from_toml(&toml).unwrap();
        assert_eq!(back.strategy, c.strategy);
        assert_eq!(back.memory_budget, c.memory_budget);
    }

    #[test]
    fn test_create_strategy_sequential() {
        let c = RuntimeConfig {
            strategy: "sequential".into(),
            ..Default::default()
        };
        let s = c.create_strategy().unwrap();
        assert_eq!(s.name(), "sequential");
    }

    #[test]
    fn test_create_strategy_greedy() {
        let c = RuntimeConfig {
            strategy: "greedy-grouping".into(),
            ..Default::default()
        };
        let s = c.create_strategy().unwrap();
        assert_eq!(s.name(), "greedy-grouping");
    }

    #[test]
    fn test_create_strategy_speculative() {
        let c = RuntimeConfig {
            strategy: "speculative-prefetch".into(),
            prefetch_ratio: Some(0.15),
            ..Default::default()
        };
        let s = c.create_strategy().unwrap();
        assert_eq!(s.name(), "speculative-prefetch");
    }

    #[test]
    fn test_create_strategy_unknown() {
        let c = RuntimeConfig {
            strategy: "bogus".into(),
            ..Default::default()
        };
        assert!(c.create_strategy().is_err());
    }

    #[test]
    fn test_resolve_threads() {
        let c = RuntimeConfig {
            num_threads: Some(8),
            ..Default::default()
        };
        assert_eq!(c.resolve_threads(), 8);

        let c2 = RuntimeConfig {
            num_threads: None,
            ..Default::default()
        };
        assert!(c2.resolve_threads() >= 1);
    }
}
