//! Evaluation harness example for Phase 5.
//! Run: cargo run --example eval_recall
//! Generates synthetic data and measures recall@K of HNSW vs brute.

use vector_search_engine::{EngineConfig, VectorEngine, embed};

fn main() {
    let mut engine = VectorEngine::new(EngineConfig::default());
    let docs = vec![
        "rust systems programming performance",
        "python data science machine learning",
        "golang concurrency web services",
        "java enterprise backend systems",
        "rust safety memory management",
    ];
    for text in &docs {
        if let Ok(emb) = embed(text) {
            let _ = engine.ingest(text.to_string(), emb, serde_json::json!({}));
        }
    }
    // Use one as query
    let query_emb = embed("rust performance").unwrap();
    let recall = engine.evaluate_recall(&[query_emb], 3);
    println!("Recall@3 on synthetic: {:.2}", recall);
    println!("Engine stats: {:?}", engine.stats());
}
