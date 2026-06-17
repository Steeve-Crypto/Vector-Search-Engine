//! Multiple indexes / collections support for Phase 6.
//!
//! Allows managing multiple independent VectorEngines (e.g., per-tenant, per-domain).
//! Each collection has its own HNSW index and documents.
//! For persistence, each collection uses its own sled subdir (e.g., data/collections/{name}).
//! 
//! Simple in-memory registry for now; can be extended with sharding etc.

use crate::{embed, EngineConfig, Metadata, SearchResult, VectorEngine, VectorError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

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

    #[test]
    fn test_sharded_basic_routing_and_cross() {
        let mut sharded = ShardedCollections::with_replicas(3, 2, "/tmp/test_sharded");
        let emb = embed("sharded test doc").unwrap();
        let id = sharded.ingest("colA", "sharded test doc".into(), emb.clone(), serde_json::json!({})).unwrap();
        assert!(!id.is_nil());

        // route to shard
        let results = sharded.search("colA", "sharded", 5).unwrap();
        assert!(!results.is_empty());

        // cross shard demo
        let cross = sharded.search_cross_shard("test", 3, None, false).unwrap();
        // may be empty or have results
        let _ = cross;
    }
}

/// Distributed and sharded mode (Phase 9).
/// - Hash-based sharding by collection name (functional routing).
/// - Cross-shard search via fan-out + top-k merge (local demo).
/// - Basic replication sketch: writes go to primary + 1 replica.
/// - In real distributed: shards would be remote instances; route reads/writes
///   via gRPC calls to other nodes (see gRPC service for Embed/Ingest/Search).
/// - Basic replication for durability (write to N replicas).
pub struct ShardedCollections {
    shards: Vec<Collections>,
    num_replicas: usize,
}

impl ShardedCollections {
    pub fn new(num_shards: usize, base_dir: impl AsRef<Path>) -> Self {
        Self::with_replicas(num_shards, 1, base_dir)
    }

    pub fn with_replicas(num_shards: usize, num_replicas: usize, base_dir: impl AsRef<Path>) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for i in 0..num_shards {
            let shard_dir = base_dir.as_ref().join(format!("shard-{}", i));
            shards.push(Collections::new(shard_dir));
        }
        Self { shards, num_replicas: num_replicas.max(1) }
    }

    fn shard_for(&self, name: &str) -> usize {
        // Simple hash sharding (can be replaced with consistent hash ring)
        let hash = name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash as usize) % self.shards.len()
    }

    fn replica_shards(&self, primary: usize) -> Vec<usize> {
        let mut reps = vec![primary];
        for i in 1..self.num_replicas {
            reps.push( (primary + i) % self.shards.len() );
        }
        reps
    }

    pub fn get_or_create(&mut self, name: &str, config: EngineConfig) -> Result<&mut VectorEngine, VectorError> {
        let idx = self.shard_for(name);
        self.shards[idx].get_or_create(name, config)
    }

    // Phase 9: delegated ops with routing + replication sketch for writes
    pub fn ingest(&mut self, name: &str, text: String, embedding: Vec<f32>, metadata: Metadata) -> Result<Uuid, VectorError> {
        let primary = self.shard_for(name);
        let replicas = self.replica_shards(primary);
        let mut last_id = None;
        for &idx in &replicas {
            let engine = self.shards[idx].get_or_create(name, EngineConfig::default())?;
            let id = engine.ingest(text.clone(), embedding.clone(), metadata.clone())?;
            last_id = Some(id);
        }
        last_id.ok_or_else(|| VectorError::Internal("no shards".into()))
    }

    pub fn search(&mut self, name: &str, query: &str, limit: usize) -> Result<Vec<SearchResult>, VectorError> {
        let idx = self.shard_for(name);
        let engine = self.shards[idx].get_or_create(name, EngineConfig::default())?;
        let emb = embed(query).map_err(|e| VectorError::Internal(e.to_string()))?;
        engine.search(&emb, limit)
    }

    pub fn hybrid_search(&mut self, name: &str, query: &str, limit: usize) -> Result<Vec<SearchResult>, VectorError> {
        let idx = self.shard_for(name);
        let engine = self.shards[idx].get_or_create(name, EngineConfig::default())?;
        engine.hybrid_search(query, limit)
    }

    /// Cross-shard fan-out search (demo for distributed). Searches 'default' collection
    /// on every shard (or named if provided) and merges top-k by score.
    pub fn search_cross_shard(&mut self, query: &str, limit: usize, collection: Option<&str>, hybrid: bool) -> Result<Vec<SearchResult>, VectorError> {
        let mut all = vec![];
        let target_colls: Vec<String> = if let Some(c) = collection {
            vec![c.to_string()]
        } else {
            // search 'default' on each shard for global demo
            (0..self.shards.len()).map(|_| "default".to_string()).collect()
        };

        for (i, coll) in target_colls.iter().enumerate().take(self.shards.len()) {
            let idx = if collection.is_some() { self.shard_for(coll) } else { i % self.shards.len() };
            if let Some(engine) = self.shards[idx].get(coll) {
                let res = if hybrid {
                    engine.hybrid_search(query, limit * 2)?
                } else {
                    let emb = embed(query).map_err(|e| VectorError::Internal(e.to_string()))?;
                    engine.search(&emb, limit * 2)?
                };
                all.extend(res);
            }
        }
        all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(limit);
        Ok(all)
    }

    pub fn list_all(&self) -> Vec<&str> {
        let mut all = vec![];
        for s in &self.shards {
            all.extend(s.list());
        }
        all
    }
}
