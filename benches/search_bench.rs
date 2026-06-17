//! Real benchmarks for Vector Search Engine (Phase 4)
//! Compares HNSW vs brute (via engine) and measures ingest/search latency.
//! Run with: cargo bench
//!
//! Requires the model to be present (will auto-download on first embed if needed).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use vector_search_engine::{embed, EngineConfig, VectorEngine};

fn setup_engine(n: usize) -> VectorEngine {
    let mut engine = VectorEngine::new(EngineConfig { max_docs: 10_000 });
    for i in 0..n {
        let text = format!("benchmark document number {} about rust and systems programming", i);
        if let Ok(emb) = embed(&text) {
            let _ = engine.ingest(text, emb, serde_json::json!({"idx": i}));
        }
    }
    engine
}

fn bench_ingest(c: &mut Criterion) {
    c.bench_function("ingest_100_docs", |b| {
        b.iter(|| {
            let mut eng = VectorEngine::new(EngineConfig { max_docs: 1000 });
            for i in 0..100 {
                let text = format!("doc {}", i);
                if let Ok(emb) = embed(&text) {
                    let _ = eng.ingest(black_box(text), black_box(emb), serde_json::json!({}));
                }
            }
        })
    });
}

fn bench_search_hnsw(c: &mut Criterion) {
    let engine = setup_engine(1000);
    let query = "rust systems programming";
    let qemb = embed(query).unwrap();

    c.bench_function("search_hnsw_1000_docs_k10", |b| {
        b.iter(|| {
            let results = engine.search(black_box(&qemb), 10).unwrap();
            black_box(results);
        })
    });
}

fn bench_search_latency(c: &mut Criterion) {
    let engine = setup_engine(5000);
    let queries = vec![
        "rust programming",
        "vector search engine",
        "machine learning",
        "high performance systems",
    ];

    let mut group = c.benchmark_group("search_latency");
    group.measurement_time(Duration::from_secs(10));

    for q in queries {
        let qemb = embed(q).unwrap();
        group.bench_function(format!("search_k5_{}", q.replace(' ', "_")), |b| {
            b.iter(|| {
                let _ = engine.search(black_box(&qemb), 5).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ingest,
    bench_search_hnsw,
    bench_search_latency
);
criterion_main!(benches);
