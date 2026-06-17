//! Real benchmarks for Vector Search Engine (Phase 6)
//! Includes quantization impact (scalar quant for memory savings).
//! - Quant/dequant time
//! - Search latency with dequantized queries (approx)
//! - Can pair with evaluate_recall for quality (recall loss from quant)
//! Run with: cargo bench
//!
//! Requires the model to be present (will auto-download on first embed if needed).
//!
//! Why benchmark with quant?
//! - Quantization (Phase 6) trades ~4x memory for small accuracy/speed cost.
//! - Benchmarks quantify: ingest overhead, search time on dequant, recall impact.
//! - Validates integration: dequant before HNSW search keeps f32 interface.
//! - Memory: compare Vec<f32> (4B/elem) vs u8 (1B/elem) for 100k docs.
//! - Part of enhanced benchmarks: recall@K, memory, throughput, more scenarios.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use vector_search_engine::{embed, quantization::{quantize, dequantize}, EngineConfig, VectorEngine};

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

fn bench_search_large(c: &mut Criterion) {
    let engine = setup_engine(5000);
    let query = "vector search engine hnsw onnx";
    let qemb = embed(query).unwrap();

    c.bench_function("search_hnsw_5000_docs_k20", |b| {
        b.iter(|| {
            let results = engine.search(black_box(&qemb), 20).unwrap();
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

fn bench_quantization(c: &mut Criterion) {
    c.bench_function("quantize_1000_vecs", |b| {
        let engine = setup_engine(1000);
        // Extract some embeddings for bench
        let embs: Vec<Vec<f32>> = engine.all_docs().into_iter().take(1000).map(|d| d.embedding).collect();
        b.iter(|| {
            for emb in &embs {
                let q = quantize(black_box(emb));
                black_box(q);
            }
        })
    });

    c.bench_function("dequantize_and_search_k5_1000_docs", |b| {
        let engine = setup_engine(1000);
        let query = "rust systems programming";
        let qemb = embed(query).unwrap();
        let qemb_q = quantize(&qemb);
        b.iter(|| {
            let deq = dequantize(black_box(&qemb_q));
            let results = engine.search(black_box(&deq), 5).unwrap();
            black_box(results);
        })
    });

    // Phase 9: scalar vs PQ (default in engine now uses PQ)
    c.bench_function("pq_vs_scalar_search_note", |b| {
        let engine = setup_engine(1000);
        let query = "vector search pq benchmark";
        let qemb = embed(query).unwrap();
        b.iter(|| {
            // engine uses trained PQ by default for storage/search path (dequant)
            let results = engine.search(black_box(&qemb), 5).unwrap();
            black_box(results);
        })
    });
}

criterion_group!(
    benches,
    bench_ingest,
    bench_search_hnsw,
    bench_search_large,
    bench_search_latency,
    bench_quantization
);
criterion_main!(benches);
