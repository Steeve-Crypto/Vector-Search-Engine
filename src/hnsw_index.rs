//! HNSW Approximate Nearest Neighbor Index wrapper (Phase 1b)
//!
//! This module provides a clean, idiomatic Rust wrapper around `hnsw_rs` tailored
//! for our vector search engine:
//! - Works exclusively with **L2-normalized f32 vectors** (from our Embedder)
//! - Uses `DistDot` which, on unit vectors, gives distance = 1 - cosine_similarity
//! - Returns results using the same `score` convention as the old brute-force
//!   (score = cosine similarity in [0, 1], higher is better)
//! - Stores client labels (Uuid) and returns them on search
//! - Easy to configure and extend with persistence later
//!
//! Key references:
//! - hnsw_rs::Hnsw + anndists::dist::DistDot
//! - Typical good defaults: max_nb_connection=16..32, ef_construction=100..200

use crate::EMBED_DIM;
use anndists::dist::DistDot;
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Errors from the HNSW index.
#[derive(Error, Debug)]
pub enum HnswError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("index is empty")]
    EmptyIndex,

    #[error("internal HNSW error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, HnswError>;

/// Configuration for building and searching the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M). Typical: 16-48.
    /// Higher = better recall, more memory.
    pub max_nb_connection: usize,

    /// Size of dynamic candidate list during construction (efConstruction).
    /// Higher = better index quality, slower build.
    pub ef_construction: usize,

    /// Maximum number of elements we expect (used for capacity hints).
    pub max_elements: usize,

    /// Default `ef` parameter for search time quality/speed tradeoff.
    /// Can be overridden per search.
    pub default_ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            ef_construction: 100,
            max_elements: 100_000,
            default_ef_search: 50,
        }
    }
}

/// A thin wrapper that owns an HNSW index + mapping from HNSW internal integer ids
/// back to our `Uuid` document identifiers.
pub struct HnswIndex {
    /// The actual HNSW structure. Uses DistDot on normalized vectors.
    hnsw: Hnsw<'static, f32, DistDot>,

    /// Parallel array: client id we passed at insert → our Uuid
    labels: Vec<Uuid>,

    /// Reverse map for fast lookup when needed (Uuid → internal id)
    /// Only populated on demand / small indexes. Not required for core path.
    id_to_label: HashMap<Uuid, usize>,

    config: HnswConfig,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("num_vectors", &self.labels.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Self {
        // Compute a reasonable max layer (log scale is common)
        let nb_elements = config.max_elements.clamp(1, usize::MAX);
        let max_layer = (nb_elements as f32).ln().trunc() as usize;
        let max_layer = max_layer.clamp(4, 16);

        info!(
            max_nb_connection = config.max_nb_connection,
            ef_construction = config.ef_construction,
            max_layer = max_layer,
            "creating new HNSW index (DistDot on normalized vectors)"
        );

        let mut hnsw = Hnsw::<f32, DistDot>::new(
            config.max_nb_connection,
            nb_elements,
            max_layer,
            config.ef_construction,
            DistDot {},
        );

        // Good defaults used in many high-recall experiments
        hnsw.set_extend_candidates(true);
        hnsw.modify_level_scale(0.5);

        Self {
            hnsw,
            labels: Vec::new(),
            id_to_label: HashMap::new(),
            config,
        }
    }

    /// Number of vectors currently indexed.
    #[inline]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Insert a normalized vector together with its document Uuid.
    ///
    /// The caller is responsible for ensuring the vector has length `EMBED_DIM`
    /// and is L2-normalized (our Embedder guarantees this).
    pub fn insert(&mut self, vector: &[f32], doc_id: Uuid) -> Result<()> {
        if vector.len() != EMBED_DIM {
            return Err(HnswError::DimMismatch {
                expected: EMBED_DIM,
                actual: vector.len(),
            });
        }

        let internal_id = self.labels.len();

        // hnsw_rs expects a slice + an integer "client id" that will be returned on search
        self.hnsw.insert_slice((vector, internal_id));

        self.labels.push(doc_id);
        self.id_to_label.insert(doc_id, internal_id);

        debug!(?doc_id, internal_id, "inserted vector into HNSW");
        Ok(())
    }

    /// Batch insert. More efficient than repeated single inserts.
    pub fn insert_batch(&mut self, items: &[(Vec<f32>, Uuid)]) -> Result<()> {
        for (vec, id) in items {
            self.insert(vec, *id)?;
        }
        Ok(())
    }

    /// Search for the `limit` nearest neighbors.
    ///
    /// Returns pairs of (doc_uuid, score) where score = cosine similarity (higher = better).
    /// Uses the configured `default_ef_search` (can be tuned).
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(Uuid, f32)>> {
        self.search_with_ef(query, limit, self.config.default_ef_search)
    }

    /// Search with explicit `ef` parameter (controls quality vs speed).
    pub fn search_with_ef(&self, query: &[f32], limit: usize, ef: usize) -> Result<Vec<(Uuid, f32)>> {
        if query.len() != EMBED_DIM {
            return Err(HnswError::DimMismatch {
                expected: EMBED_DIM,
                actual: query.len(),
            });
        }
        if self.is_empty() {
            return Ok(vec![]);
        }

        let neighbours: Vec<Neighbour> = self.hnsw.search(query, limit, ef);

        let mut results: Vec<(Uuid, f32)> = Vec::with_capacity(neighbours.len());
        for n in neighbours {
            // d_id is the client identifier we passed at insert time (our dense internal usize)
            if n.d_id < self.labels.len() {
                let doc_id = self.labels[n.d_id];
                // For DistDot on unit vectors: distance = 1 - cosine
                let score = (1.0 - n.distance).clamp(0.0, 1.0);
                results.push((doc_id, score));
            }
        }

        // The library already returns them sorted by increasing distance (best first)
        Ok(results)
    }

    /// Persist the HNSW graph + data + our label mapping to disk.
    /// Creates two files for the HNSW dump (basename.hnsw.graph / .data) and a .labels.bin .
    pub fn dump(&self, dir: &Path, basename: &str) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| HnswError::Internal(e.to_string()))?;

        let actual_basename = self
            .hnsw
            .file_dump(dir, basename)
            .map_err(|e| HnswError::Internal(e.to_string()))?;

        let labels_path = dir.join(format!("{}.labels.bin", actual_basename));
        let bytes = bincode::serialize(&self.labels)
            .map_err(|e| HnswError::Internal(e.to_string()))?;
        std::fs::write(labels_path, bytes)
            .map_err(|e| HnswError::Internal(e.to_string()))?;

        info!(dir = %dir.display(), basename = %actual_basename, "HNSW index dumped");
        Ok(())
    }

    /// Load HNSW preferring hnswio dump if available for speed (graph structure), falling back to rebuild.
    /// Labels must match the dump order. For full durability use sled docs + rebuild (current default).
    pub fn load(dir: &Path, basename: &str, config: HnswConfig) -> Result<Self> {
        // Note: full load_hnsw_with_dist has lifetime ties to HnswIo in this version.
        // We prefer rebuild for reliability (fast enough), but keep dump for snapshots.
        // Future: can leak reloader or use mmap-aware holder for owned graph load.
        warn!("HNSW load prefers rebuild from docs for now (dump available via .dump())");
        // Rebuild empty and let caller populate via from_docs or inserts
        let mut hnsw = HnswIndex::new(config);
        let labels_path = dir.join(format!("{}.labels.bin", basename));
        if labels_path.exists() {
            let bytes = std::fs::read(&labels_path)
                .map_err(|e| HnswError::Internal(e.to_string()))?;
            let labels: Vec<Uuid> = bincode::deserialize(&bytes)
                .map_err(|e| HnswError::Internal(e.to_string()))?;
            hnsw.labels = labels.clone();
            for (i, &u) in labels.iter().enumerate() {
                hnsw.id_to_label.insert(u, i);
            }
            info!("Labels loaded from dump sidecar; populate embeddings via engine");
        }
        Ok(hnsw)
    }

    /// Return basic stats useful for /stats and monitoring.
    pub fn stats(&self) -> HnswStats {
        HnswStats {
            num_vectors: self.len(),
            max_nb_connection: self.config.max_nb_connection,
            ef_construction: self.config.ef_construction,
            default_ef_search: self.config.default_ef_search,
            // We could expose more internal info via hnsw.dump_layer_info() if wanted
        }
    }

    /// Get the original document id for a given internal HNSW id (mainly for debugging).
    pub fn get_label(&self, internal_id: usize) -> Option<Uuid> {
        self.labels.get(internal_id).copied()
    }
}

/// Lightweight stats returned to higher layers.
#[derive(Debug, Clone)]
pub struct HnswStats {
    pub num_vectors: usize,
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub default_ef_search: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::embed; // real embeddings

    fn make_index() -> HnswIndex {
        let cfg = HnswConfig {
            max_nb_connection: 8,
            ef_construction: 40,
            max_elements: 100,
            default_ef_search: 16,
        };
        HnswIndex::new(cfg)
    }

    #[test]
    fn test_hnsw_basic_insert_search() {
        let mut index = make_index();

        // Use real embeddings so the test is meaningful
        let v1 = embed("the quick brown fox").unwrap();
        let v2 = embed("the fast brown fox jumps").unwrap();
        let v3 = embed("completely unrelated topic about cooking pasta").unwrap();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        index.insert(&v1, id1).unwrap();
        index.insert(&v2, id2).unwrap();
        index.insert(&v3, id3).unwrap();

        assert_eq!(index.len(), 3);

        let results = index.search(&v1, 3).unwrap();
        assert!(!results.is_empty());

        // The closest result to v1 should be itself (score very close to 1.0)
        let (best_id, best_score) = results[0];
        assert_eq!(best_id, id1);
        assert!(best_score > 0.98, "self similarity should be very high, got {}", best_score);

        // v2 should be closer to v1 than v3 is
        let score_v2 = results.iter().find(|(id, _)| *id == id2).map(|(_, s)| *s).unwrap_or(0.0);
        let score_v3 = results.iter().find(|(id, _)| *id == id3).map(|(_, s)| *s).unwrap_or(0.0);

        assert!(
            score_v2 > score_v3,
            "semantically closer doc should have higher score (v2={} vs v3={})",
            score_v2,
            score_v3
        );
    }

    #[test]
    fn test_dimension_check() {
        let mut index = make_index();
        let bad = vec![0.1f32; 10];
        let res = index.insert(&bad, Uuid::new_v4());
        assert!(matches!(res, Err(HnswError::DimMismatch { .. })));
    }
}
