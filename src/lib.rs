//! Core library for the Vector Search Engine.
//!
//! This module defines the primary data structures and a basic in-memory
//! engine used during early development (Phase 0). The real implementation
//! will add:
//! - ONNX-based embeddings
//! - HNSW index (hnsw_rs)
//! - Persistence (sled)
//! - Proper concurrency (DashMap / RwLock)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

pub mod api;
pub mod embedder;
pub mod hnsw_index;

// Re-export the main embedder API for convenience at the crate root
pub use embedder::{download_model_if_needed, embed, embed_batch, Embedder, EmbedderError};

// Re-export HNSW types so higher layers (and tests) can use them directly if needed
pub use hnsw_index::{HnswConfig, HnswIndex, HnswStats};

use std::path::Path;
use tracing::{debug, info, warn};

/// Dimension of the embedding vectors (all-MiniLM-L6-v2 produces 384-dim).
pub const EMBED_DIM: usize = 384;

/// Error type for core operations.
#[derive(Error, Debug)]
pub enum VectorError {
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("empty text provided for embedding/ingest")]
    EmptyText,

    #[error("invalid query or parameters: {0}")]
    InvalidQuery(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, VectorError>;

/// Metadata is stored as arbitrary JSON for flexibility (filtering comes later).
pub type Metadata = serde_json::Value;

/// A single document in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier (UUID v4).
    pub id: Uuid,
    /// Original text content.
    pub text: String,
    /// L2-normalized embedding vector (length == EMBED_DIM).
    pub embedding: Vec<f32>,
    /// Arbitrary JSON metadata supplied at ingest time.
    pub metadata: Metadata,
}

impl Document {
    pub fn new(text: String, embedding: Vec<f32>, metadata: Metadata) -> Result<Self> {
        if text.trim().is_empty() {
            return Err(VectorError::EmptyText);
        }
        if embedding.len() != EMBED_DIM {
            return Err(VectorError::DimMismatch {
                expected: EMBED_DIM,
                actual: embedding.len(),
            });
        }
        Ok(Self {
            id: Uuid::new_v4(),
            text,
            embedding,
            metadata,
        })
    }
}

/// Search result returned to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub text: String,
    pub metadata: Metadata,
    /// Cosine similarity score (higher = more similar). For normalized vectors
    /// this is equivalent to dot product.
    pub score: f32,
    /// Optional distance (1 - score for cosine).
    pub distance: Option<f32>,
}

/// Configuration for the basic engine (Phase 0).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum number of documents to keep (simple safeguard).
    pub max_docs: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { max_docs: 1_000_000 }
    }
}

/// In-memory vector store backed by HNSW for fast ANN search.
///
/// Documents (text + metadata + embedding) are kept in a HashMap for retrieval.
/// The HNSW index provides fast approximate cosine nearest-neighbor search.
///
/// This replaced the Phase 0 brute-force implementation.
#[derive(Debug)]
pub struct VectorEngine {
    config: EngineConfig,
    /// Primary store: id -> full Document (text, embedding, metadata)
    docs: HashMap<Uuid, Document>,
    /// HNSW index for fast approximate search (cosine on normalized vectors)
    hnsw: hnsw_index::HnswIndex,
    /// Optional sled DB for persistence. When present, ingests are durably written.
    db: Option<sled::Db>,
}

impl VectorEngine {
    pub fn new(config: EngineConfig) -> Self {
        let hnsw_config = hnsw_index::HnswConfig {
            max_nb_connection: 16,
            ef_construction: 80,
            max_elements: config.max_docs,
            default_ef_search: 32,
        };

        Self {
            config,
            docs: HashMap::new(),
            hnsw: hnsw_index::HnswIndex::new(hnsw_config),
            db: None,
        }
    }

    /// Open or create a persistent engine backed by sled.
    /// Documents are stored in sled. The HNSW index is rebuilt from the stored embeddings on load.
    /// This provides restart survival (Phase 2).
    pub fn open_persistent(data_dir: &Path, config: EngineConfig) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| VectorError::Internal(e.to_string()))?;

        let db_path = data_dir.join("vecstore.sled");
        let db = sled::open(&db_path)
            .map_err(|e| VectorError::Internal(e.to_string()))?;

        let mut engine = Self::new(config);
        engine.db = Some(db.clone());

        // Rebuild docs + HNSW from sled
        let mut raw_count = 0usize;
        let mut loaded_count = 0usize;
        for res in db.iter() {
            raw_count += 1;
            let (_key, val) = res.map_err(|e| VectorError::Internal(e.to_string()))?;
            match serde_json::from_slice::<Document>(&val) {
                Ok(doc) => {
                    let id = doc.id;
                    let emb = doc.embedding.clone();
                    engine.docs.insert(id, doc);
                    let _ = engine.hnsw.insert(&emb, id);
                    loaded_count += 1;
                }
                Err(e) => {
                    warn!(error = %e, "failed to deserialize a document from sled");
                }
            }
        }
        debug!(raw_entries = raw_count, deserialized = loaded_count, "sled load scan complete");
        if loaded_count > 0 {
            info!(loaded = loaded_count, "successfully loaded documents from sled");
        }

        info!(path = %db_path.display(), count = engine.docs.len(), "loaded persistent engine from sled (HNSW rebuilt)");
        Ok(engine)
    }

    /// Ingest a document together with its (already normalized) embedding.
    /// The embedding is stored for potential re-use and also inserted into the HNSW index.
    pub fn ingest(&mut self, text: String, embedding: Vec<f32>, metadata: Metadata) -> Result<Uuid> {
        if self.docs.len() >= self.config.max_docs {
            // Simple eviction of oldest (not great, placeholder)
            if let Some((&oldest, _)) = self.docs.iter().next() {
                self.docs.remove(&oldest);
                // Note: we do not currently support deletion from HNSW (common limitation).
                // For a production system we would either rebuild or use a tombstone + filter.
            }
        }

        let doc = Document::new(text, embedding.clone(), metadata)?;
        let id = doc.id;

        // Persist to sled first (if enabled)
        // Use serde_json because bincode doesn't support serde_json::Value's deserialize_any
        if let Some(db) = &self.db {
            let key = id.as_bytes();
            let val = serde_json::to_vec(&doc)
                .map_err(|e| VectorError::Internal(e.to_string()))?;
            db.insert(key, val)
                .map_err(|e| VectorError::Internal(e.to_string()))?;
            let _ = db.flush();
            info!(db_len_after_write = db.len(), "wrote document to sled");
        }

        self.docs.insert(id, doc);

        // Insert into HNSW
        if let Err(e) = self.hnsw.insert(&embedding, id) {
            tracing::warn!(?id, error = %e, "failed to insert into HNSW index (document kept)");
        }

        Ok(id)
    }

    /// Approximate nearest neighbor search using the HNSW index.
    ///
    /// All embeddings are assumed L2-normalized.
    /// Returns top `limit` results with cosine similarity scores (higher = better).
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != EMBED_DIM {
            return Err(VectorError::DimMismatch {
                expected: EMBED_DIM,
                actual: query_embedding.len(),
            });
        }
        if limit == 0 || self.hnsw.is_empty() {
            return Ok(vec![]);
        }

        let neighbours = self
            .hnsw
            .search(query_embedding, limit)
            .map_err(|e| VectorError::Internal(e.to_string()))?;

        let results: Vec<SearchResult> = neighbours
            .into_iter()
            .filter_map(|(id, score)| {
                self.docs.get(&id).map(|doc| SearchResult {
                    id,
                    text: doc.text.clone(),
                    metadata: doc.metadata.clone(),
                    score,
                    distance: Some(1.0 - score),
                })
            })
            .collect();

        Ok(results)
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Return basic stats (used by /stats and CLI).
    pub fn stats(&self) -> EngineStats {
        let hstats = self.hnsw.stats();
        EngineStats {
            num_documents: self.docs.len(),
            embedding_dim: EMBED_DIM,
            index_type: format!(
                "HNSW (M={}, ef_c={}, ef_search={})",
                hstats.max_nb_connection, hstats.ef_construction, hstats.default_ef_search
            ),
        }
    }

    /// Get a document by ID (useful for debugging).
    pub fn get(&self, id: Uuid) -> Option<&Document> {
        self.docs.get(&id)
    }

    /// Returns a snapshot of all documents.
    /// Used by the CLI demo persistence (JSON) and will be used by sled later.
    pub fn all_docs(&self) -> Vec<Document> {
        // We iterate the docs map. Order is not guaranteed (was never a strong guarantee).
        self.docs.values().cloned().collect()
    }

    /// Create engine pre-populated from documents (used by Phase 0/1 JSON snapshot loader).
    /// Rebuilds the HNSW index from the provided embeddings.
    pub fn from_docs(docs: Vec<Document>, config: EngineConfig) -> Self {
        let mut engine = VectorEngine::new(config);
        for d in docs {
            let emb = d.embedding.clone();
            let id = d.id;
            engine.docs.insert(id, d);
            // Best effort insert into HNSW
            let _ = engine.hnsw.insert(&emb, id);
        }
        engine
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub num_documents: usize,
    pub embedding_dim: usize,
    pub index_type: String,
}

/// Compute cosine similarity between two vectors.
/// Assumes both are L2-normalized (otherwise this is dot product).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// Very simple deterministic "embedding" generator for Phase 0 demos.
/// 
/// This lets the CLI work *immediately* without downloading the ONNX model.
/// It produces a 384-dim vector that has *some* semantic signal
/// (similar texts tend to get closer embeddings).
/// 
/// When the real Embedder is ready we will replace calls to this.
pub fn simple_hash_embedding(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.to_lowercase().hash(&mut hasher);
    let seed = hasher.finish();

    let mut vec = vec![0f32; EMBED_DIM];
    let mut rng_state = seed;

    // Simple xorshift + normalize to unit length
    for val in vec.iter_mut().take(EMBED_DIM) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        // Map to roughly [-1, 1] range
        let v = ((rng_state as i64 % 10000) as f32 / 5000.0) - 1.0;
        *val = v;
    }

    // Add a tiny amount of lexical signal so "rust" and "rustacean" are closer than random
    for (i, c) in text.chars().enumerate().take(64) {
        let idx = (i * 7 + c as usize) % EMBED_DIM;
        vec[idx] += (c as u32 % 7) as f32 * 0.03;
    }

    normalize(&mut vec);
    vec
}

/// L2 normalize a vector in place. Returns the original norm (for debugging).
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0f32, 4.0];
        let norm = normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_engine_basic_flow() {
        // Uses the *real* ONNX embedder (will auto-download model if necessary)
        let mut engine = VectorEngine::new(EngineConfig::default());

        let emb1 = embed("rust programming language").expect("real embed failed");
        let emb2 = embed("python snake language").expect("real embed failed");
        let emb_query = embed("rust lang").expect("real embed failed");

        let _id1 = engine
            .ingest("rust programming language".to_string(), emb1, serde_json::json!({"lang": "rust"}))
            .unwrap();
        let _id2 = engine
            .ingest("python snake language".to_string(), emb2, serde_json::json!({"lang": "python"}))
            .unwrap();

        assert_eq!(engine.len(), 2);

        let results = engine.search(&emb_query, 5).unwrap();
        assert!(!results.is_empty());
        // With real MiniLM embeddings, "rust" query should rank the rust doc first
        assert!(
            results[0].text.to_lowercase().contains("rust"),
            "expected rust document to rank first, got: {}",
            results[0].text
        );

        let stats = engine.stats();
        assert_eq!(stats.num_documents, 2);
        assert_eq!(stats.embedding_dim, EMBED_DIM);
    }

    #[test]
    fn test_document_validation() {
        let bad = Document::new(
            "".to_string(),
            vec![0.0; EMBED_DIM],
            serde_json::Value::Null,
        );
        assert!(matches!(bad, Err(VectorError::EmptyText)));

        let bad_dim = Document::new(
            "hello".to_string(),
            vec![0.1; 10],
            serde_json::Value::Null,
        );
        assert!(matches!(bad_dim, Err(VectorError::DimMismatch { .. })));
    }
}
