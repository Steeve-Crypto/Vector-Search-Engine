//! Basic search benchmark skeleton using Criterion.
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn dummy_search_bench(c: &mut Criterion) {
    c.bench_function("dummy_search_1000", |b| {
        // Placeholder - will be replaced with real engine + HNSW bench
        let docs: Vec<u32> = (0..1000).collect();
        b.iter(|| {
            black_box(docs.iter().filter(|&&x| x % 3 == 0).count())
        })
    });
}

criterion_group!(benches, dummy_search_bench);
criterion_main!(benches);
