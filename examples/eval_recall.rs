//! Evaluation harness example for Phase 5.
//! Run: cargo run --example eval_recall
//! Uses the dataset loader for synthetic data + measures recall@K.

use vector_search_engine::{dataset, EngineConfig, VectorEngine, embed};

fn main() {
    let mut engine = VectorEngine::new(EngineConfig::default());

    // Use sample dataset loader (synthetic for demo)
    let synthetic = dataset::generate_synthetic(100, 384, 42);
    for (text, emb, meta) in &synthetic {
        let _ = engine.ingest(text.clone(), emb.clone(), meta.clone());
    }

    // Generate a few query embeddings (use real embedder or synthetic)
    let queries: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let qtext = format!("synthetic-doc-{}", i * 10);
            embed(&qtext).unwrap_or_else(|_| dataset::generate_synthetic(1, 384, 99 + i as u64)[0].1.clone())
        })
        .collect();

    let recall = engine.evaluate_recall(&queries, 5);
    println!("Recall@5 on 100 synthetic docs: {:.3}", recall);
    println!("Engine stats: num_docs={}, index={}", engine.len(), engine.stats().index_type);

    println!("(Use dataset::compute_recall in your own harness for full brute vs HNSW comparison.)");
}
