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
pub mod collection;
pub mod dataset;
pub mod embedder;
pub mod grpc_stub;
pub mod hnsw_index;
pub mod quantization;

// Re-export the main embedder API for convenience at the crate root
pub use embedder::{download_model_if_needed, embed, embed_batch, Embedder, EmbedderError};

// Re-export HNSW types so higher layers (and tests) can use them directly if needed
pub use collection::{Collections, ShardedCollections};
pub use hnsw_index::{HnswConfig, HnswIndex, HnswStats};
pub use quantization::{dequantize, quantize, quantization_error, QuantizedVector, ProductQuantizer, default_product_quantizer};
// Real k-means PQ (Phase 8): ProductQuantizer::train(samples, m, k) for production-grade compression.

/// Internal struct for sled storage with quantized embedding (Phase 6 integration).
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredDocument {
    id: Uuid,
    text: String,
    embedding: Vec<u8>,  // quantized by default
    metadata: String,    // JSON as string to allow bincode
}

use std::path::Path;
use tracing::{debug, info, warn};

/// Dimension of the embedding vectors (all-MiniLM-L6-v2 produces 384-dim).
pub const EMBED_DIM: usize = 384;
// Phase 11: multi-modal prep - future image embeddings (e.g. 512 dim for CLIP) via similar ONNX.

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
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
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
    /// Concurrency: use ShardedCollections or DashMap in future for high load (Phase 9) 
    hnsw: hnsw_index::HnswIndex,
    /// Optional sled DB for persistence. When present, ingests are durably written.
    db: Option<sled::Db>,
    /// Trained PQ for storage compression (Phase 8 polish). Uses default trained instance.
    pq: Option<ProductQuantizer>,
    /// Path for HNSW dumps (Phase 9). Enables automatic + explicit persistence.
    hnsw_dump_path: Option<std::path::PathBuf>,
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
            pq: Some(default_product_quantizer()),
            hnsw_dump_path: None,
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
        engine.pq = Some(default_product_quantizer());
        engine.hnsw_dump_path = Some(data_dir.to_path_buf());

        // Phase 9: prefer HNSW dump as primary (with version check)
        let hnsw_config = hnsw_index::HnswConfig::default();
        let mut hnsw_loaded_from_dump = false;
        let ver_path = data_dir.join("hnsw.version");
        let version_ok = std::fs::read_to_string(&ver_path).map(|v| v.trim() == "1").unwrap_or(false);
        if version_ok {
            if let Ok(loaded_hnsw) = hnsw_index::HnswIndex::load(data_dir, "hnsw", hnsw_config) {
                if loaded_hnsw.len() > 0 {
                    engine.hnsw = loaded_hnsw;
                    hnsw_loaded_from_dump = true;
                    info!("HNSW graph loaded from dump (primary path, v1)");
                }
            }
        } else if ver_path.exists() {
            warn!("HNSW dump version mismatch or missing; falling back to sled rebuild");
        }

        // Load docs from sled (always). Only insert to HNSW if not loaded from dump.
        let mut raw_count = 0usize;
        let mut loaded_count = 0usize;
        for res in db.iter() {
            raw_count += 1;
            let (_key, val) = res.map_err(|e| VectorError::Internal(e.to_string()))?;
            match bincode::deserialize::<StoredDocument>(&val) {
                Ok(stored) => {
                    let id = stored.id;
                    // Phase 8: dequant using PQ if the stored bytes look like PQ codes (short), else scalar
                    let emb = if stored.embedding.len() == EMBED_DIM {
                        dequantize(&stored.embedding)
                    } else if let Some(pq) = &engine.pq {
                        pq.dequantize(&stored.embedding)
                    } else {
                        dequantize(&stored.embedding)
                    };
                    let metadata: Metadata = serde_json::from_str(&stored.metadata).unwrap_or(serde_json::Value::Null);
                    let doc = Document {
                        id,
                        text: stored.text,
                        embedding: emb.clone(),
                        metadata,
                    };
                    engine.docs.insert(id, doc);
                    if !hnsw_loaded_from_dump {
                        let _ = engine.hnsw.insert(&emb, id);
                    }
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

        let rebuild_note = if hnsw_loaded_from_dump { " (HNSW from dump)" } else { " (HNSW rebuilt)" };
        info!(path = %db_path.display(), count = engine.docs.len(), "loaded persistent engine from sled{}", rebuild_note);
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

        let metadata_str = serde_json::to_string(&metadata).unwrap_or_default();
        let doc = Document::new(text, embedding.clone(), metadata)?;
        let id = doc.id;

        // Persist to sled first (if enabled)
        // Phase 8: use trained PQ by default for much better compression (8 bytes vs 384)
        if let Some(db) = &self.db {
            let key = id.as_bytes();
            let qemb = if let Some(pq) = &self.pq {
                if pq.is_trained() {
                    pq.quantize(&embedding)
                } else {
                    quantize(&embedding)
                }
            } else {
                quantize(&embedding)
            };
            let stored = StoredDocument {
                id,
                text: doc.text.clone(),
                embedding: qemb,
                metadata: metadata_str,
            };
            let val = bincode::serialize(&stored)
                .map_err(|e| VectorError::Internal(e.to_string()))?;
            db.insert(key, val)
                .map_err(|e| VectorError::Internal(e.to_string()))?;
            let _ = db.flush();
            info!(db_len_after_write = db.len(), "wrote document to sled (using PQ by default)");

            // Phase 9: automatic periodic HNSW dump (every 100 docs) + versioning sidecar
            if let Some(dump_dir) = &self.hnsw_dump_path {
                if self.docs.len() % 100 == 0 {
                    let _ = self.hnsw.dump(dump_dir, "hnsw");
                    // simple version sidecar
                    let ver_path = dump_dir.join("hnsw.version");
                    let _ = std::fs::write(ver_path, b"1");
                }
            }
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
        self.search_with_ef(query_embedding, limit, self.hnsw.default_ef())
    }

    /// Search with explicit ef (controls quality vs speed, Phase 9).
    pub fn search_with_ef(&self, query_embedding: &[f32], limit: usize, ef: usize) -> Result<Vec<SearchResult>> {
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
            .search_with_ef(query_embedding, limit, ef)
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

    /// Simple keyword score for hybrid search (Phase 6).
    /// Uses normalized term overlap (Jaccard-like) between query words and document text.
    fn keyword_score(text: &str, query: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let query_words: Vec<&str> = query.split_whitespace().filter(|w| !w.is_empty()).collect();
        if query_words.is_empty() {
            return 0.0;
        }
        let mut matches = 0;
        for w in &query_words {
            if text_lower.contains(w) {
                matches += 1;
            }
        }
        matches as f32 / query_words.len() as f32
    }

    /// Hybrid search (Phase 6): combines vector similarity (HNSW) with keyword overlap.
    /// Weight: 0.7 vector + 0.3 keyword (tunable).
    /// Returns top-k with combined score (higher better).
    pub fn hybrid_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let query_emb = embed(query).map_err(|e| VectorError::Internal(e.to_string()))?;
        // Over-fetch from vector search for better hybrid reranking
        let vec_results = self.search(&query_emb, limit * 3)?;

        let mut combined: Vec<SearchResult> = vec_results
            .into_iter()
            .map(|mut r| {
                let kw = Self::keyword_score(&r.text, query);
                let combined_score = 0.7 * r.score + 0.3 * kw;
                r.score = combined_score.clamp(0.0, 1.0);
                r.distance = Some(1.0 - r.score);
                r
            })
            .collect();

        combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(limit);
        Ok(combined)
    }

    /// Evaluation harness (Phase 5): approximate recall@K vs brute-force on current docs.
    /// Returns average recall over the provided query embeddings.
    pub fn evaluate_recall(&self, query_embeddings: &[Vec<f32>], k: usize) -> f64 {
        if self.is_empty() || query_embeddings.is_empty() || k == 0 {
            return 1.0;
        }
        let mut total = 0.0;
        for q in query_embeddings {
            let hnsw_res = match self.search(q, k) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let hnsw_ids: std::collections::HashSet<_> = hnsw_res.into_iter().map(|r| r.id).collect();

            // brute force top-k
            let mut scored: Vec<(Uuid, f32)> = self
                .docs
                .values()
                .map(|doc| (doc.id, cosine_similarity(&doc.embedding, q)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let brute_ids: std::collections::HashSet<_> = scored.into_iter().take(k).map(|(id, _)| id).collect();

            let hits = hnsw_ids.intersection(&brute_ids).count();
            total += hits as f64 / k as f64;
        }
        total / query_embeddings.len() as f64
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
            hnsw_max_nb_connection: hstats.max_nb_connection,
            hnsw_ef_construction: hstats.ef_construction,
            hnsw_default_ef_search: hstats.default_ef_search,
        }
    }

    /// Dump the current HNSW graph for fast future restarts (Phase 8 integration).
    /// Call this after significant ingests for snapshotting. Labels sidecar is written too.
    pub fn save_hnsw(&self, dir: &Path, basename: &str) -> hnsw_index::Result<()> {
        let res = self.hnsw.dump(dir, basename);
        if res.is_ok() {
            let _ = std::fs::write(dir.join(format!("{}.version", basename)), b"1");
        }
        res
    }

    /// Train (or retrain) the internal PQ on representative sample embeddings.
    /// This can improve compression quality for your specific data distribution.
    pub fn train_pq(&mut self, samples: &[Vec<f32>]) {
        self.pq = Some(ProductQuantizer::train(samples, 8, 256));
    }

    /// Retrieval-only helper for frameworks (Phase 9 adapter).
    /// Returns top docs for a query (vector or hybrid).
    pub fn retrieve(&self, query: &str, limit: usize, hybrid: bool) -> Result<Vec<SearchResult>> {
        if hybrid {
            self.hybrid_search(query, limit)
        } else {
            let emb = embed(query).map_err(|e| VectorError::Internal(e.to_string()))?;
            self.search(&emb, limit)
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
    pub hnsw_max_nb_connection: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_default_ef_search: usize,
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

    #[test]
    fn test_evaluate_recall() {
        let mut engine = VectorEngine::new(EngineConfig::default());
        let emb1 = embed("rust is fast and safe for systems").unwrap();
        let emb1b = embed("rust programming language is fast").unwrap();  // similar
        let emb2 = embed("completely unrelated cooking recipes").unwrap();
        let _ = engine.ingest("rust is fast and safe for systems".into(), emb1.clone(), serde_json::json!({}));
        let _ = engine.ingest("completely unrelated cooking recipes".into(), emb2, serde_json::json!({}));
        let recall = engine.evaluate_recall(&[emb1b], 1);
        assert!(recall > 0.5, "recall should be reasonable for similar query, got {}", recall);
    }

    #[test]
    fn test_hybrid_search() {
        let mut engine = VectorEngine::new(EngineConfig::default());
        let _ = engine.ingest("rust is fast and safe for systems programming".into(), embed("rust is fast and safe for systems programming").unwrap(), serde_json::json!({}));
        let _ = engine.ingest("python is great for data science and ML".into(), embed("python is great for data science and ML").unwrap(), serde_json::json!({}));

        let results = engine.hybrid_search("rust performance", 2).unwrap();
        assert!(!results.is_empty());
        // Hybrid should still rank the rust doc high
        assert!(results[0].text.to_lowercase().contains("rust"));
    }
}
