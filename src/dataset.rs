//! Sample dataset loader for evaluation and demos (Phase 5).
//!
//! Provides synthetic data generation for testing recall, benchmarks,
//! and demo without external datasets (like SIFT).
//! Can be extended to load real data (e.g., from JSON or CSV).
//!
//! Example usage:
//! ```ignore
//! use vector_search_engine::dataset;
//! let data = dataset::generate_synthetic(100, 384, 42);
//! ```

use crate::{cosine_similarity, EMBED_DIM};
use serde_json::json;
use std::collections::HashSet;

/// A synthetic document: (text, embedding, metadata)
pub type SyntheticDoc = (String, Vec<f32>, serde_json::Value);

/// Generate `n` synthetic normalized vectors in `dim` dimensions using a simple LCG PRNG.
/// Text is "synthetic-doc-{i}", metadata has index.
/// This is deterministic for reproducible evals.
pub fn generate_synthetic(n: usize, dim: usize, seed: u64) -> Vec<SyntheticDoc> {
    let mut docs = Vec::with_capacity(n);
    let mut rng = seed;
    for i in 0..n {
        let mut vec = vec![0.0f32; dim];
        for j in 0..dim {
            rng = rng.wrapping_mul(6364136223846793005u64).wrapping_add(1);
            vec[j] = ((rng >> 33) as f32 / (u64::MAX >> 33) as f32) * 2.0 - 1.0;
        }
        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        let text = format!("synthetic-doc-{}", i);
        let meta = json!({"idx": i, "type": "synthetic"});
        docs.push((text, vec, meta));
    }
    docs
}

/// Load or generate data. For now, always synthetic. Extend for real files.
pub fn load_or_generate(n: usize) -> Vec<SyntheticDoc> {
    // In future: if path exists, load from JSON/CSV (e.g. SIFT format)
    // For MVP: synthetic
    generate_synthetic(n, EMBED_DIM, 42)
}

/// Compute brute-force top-k for a query against docs (for recall ground truth).
pub fn brute_top_k(docs: &[(String, Vec<f32>, serde_json::Value)], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, (_, emb, _))| (i, cosine_similarity(emb, query)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Simple recall@K using the harness in engine + brute for ground truth.
pub fn compute_recall(docs: &[(String, Vec<f32>, serde_json::Value)], engine_search_fn: impl Fn(&[f32], usize) -> Vec<usize>, queries: &[Vec<f32>], k: usize) -> f64 {
    let mut total = 0.0;
    for q in queries {
        let hnsw_top: HashSet<usize> = engine_search_fn(q, k).into_iter().collect();
        let brute_top: HashSet<usize> = brute_top_k(docs, q, k).into_iter().collect();
        let hits = hnsw_top.intersection(&brute_top).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_generation() {
        let docs = generate_synthetic(10, 384, 123);
        assert_eq!(docs.len(), 10);
        let (_, emb, _) = &docs[0];
        assert_eq!(emb.len(), 384);
        let norm: f32 = emb.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "should be unit vector");
    }

    #[test]
    fn test_brute_top_k() {
        let docs = generate_synthetic(5, 3, 1);
        let q = docs[0].1.clone();
        let top = brute_top_k(&docs, &q, 1);
        assert_eq!(top, vec![0]);
    }
}
