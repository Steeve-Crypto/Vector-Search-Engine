//! Multiple indexes / collections support for Phase 6.
//!
//! Allows managing multiple independent VectorEngines (e.g., per-tenant, per-domain).
//! Each collection has its own HNSW index and documents.
//! For persistence, each collection uses its own sled subdir (e.g., data/collections/{name}).
//! 
//! Simple in-memory registry for now; can be extended with sharding etc.

use crate::{EngineConfig, VectorEngine, VectorError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Manages multiple named collections, each a separate VectorEngine.
#[derive(Debug, Default)]
pub struct Collections {
    /// collection_name -> engine
    engines: HashMap<String, VectorEngine>,
    /// Base data dir for persistence (e.g. "data/collections")
    base_dir: PathBuf,
}

impl Collections {
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            engines: HashMap::new(),
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Create or get a collection. If new and persistent, opens sled for it.
    pub fn get_or_create(&mut self, name: &str, config: EngineConfig) -> Result<&mut VectorEngine, VectorError> {
        if !self.engines.contains_key(name) {
            let col_dir = self.base_dir.join(name);
            let engine = if self.base_dir.exists() {
                VectorEngine::open_persistent(&col_dir, config.clone())
                    .unwrap_or_else(|_| VectorEngine::new(config.clone()))
            } else {
                VectorEngine::new(config)
            };
            self.engines.insert(name.to_string(), engine);
        }
        Ok(self.engines.get_mut(name).unwrap())
    }

    /// List all collection names.
    pub fn list(&self) -> Vec<&str> {
        self.engines.keys().map(|s| s.as_str()).collect()
    }

    /// Delete a collection (drops from memory; for full delete, rm dir).
    pub fn delete(&mut self, name: &str) -> bool {
        self.engines.remove(name).is_some()
    }

    /// Get a collection (must exist).
    pub fn get(&self, name: &str) -> Option<&VectorEngine> {
        self.engines.get(name)
    }

    /// Get mutable (for ingest etc).
    pub fn get_mut(&mut self, name: &str) -> Option<&mut VectorEngine> {
        self.engines.get_mut(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed;

    #[test]
    fn test_collections_basic() {
        let mut cols = Collections::new("/tmp/test_collections");
        let eng = cols.get_or_create("testcol", EngineConfig::default()).unwrap();
        let emb = embed("hello collection").unwrap();
        let id = eng.ingest("hello collection".into(), emb, serde_json::json!({})).unwrap();
        assert!(eng.get(id).is_some());

        assert!(cols.list().contains(&"testcol"));
        assert!(cols.delete("testcol"));
        assert!(!cols.list().contains(&"testcol"));
    }
}

/// Distributed mode sketch (Phase 6): simple sharding by collection name hash.
/// In real distributed, this would route to remote nodes (e.g. via gRPC).
/// For now, a local sketch that could be extended with network sharding.
pub struct ShardedCollections {
    shards: Vec<Collections>,
}

impl ShardedCollections {
    pub fn new(num_shards: usize, base_dir: impl AsRef<Path>) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for i in 0..num_shards {
            let shard_dir = base_dir.as_ref().join(format!("shard-{}", i));
            shards.push(Collections::new(shard_dir));
        }
        Self { shards }
    }

    fn shard_for(&self, name: &str) -> usize {
        // Simple hash sharding
        let hash = name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash as usize) % self.shards.len()
    }

    pub fn get_or_create(&mut self, name: &str, config: EngineConfig) -> Result<&mut VectorEngine, VectorError> {
        let idx = self.shard_for(name);
        self.shards[idx].get_or_create(name, config)
    }
}
