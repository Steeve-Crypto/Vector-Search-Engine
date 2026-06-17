//! Vector Search Engine - CLI entry point (Phase 0)
//!
//! This binary provides a minimal working CLI so you can try the system
//! immediately:
//!
//!   cargo run -- ingest --text "Your document here" --meta '{"source":"cli"}'
//!   cargo run -- search --query "your query" --limit 5
//!   cargo run -- stats
//!
//! The "serve" command now runs the full Axum REST API (Phase 3).
//!
//! Real embeddings via ONNX Runtime + all-MiniLM-L6-v2 are now active.
//! The old `simple_hash_embedding` helper remains only for reference / special tests.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use vector_search_engine::{embed, EngineConfig, EMBED_DIM};

#[derive(Parser)]
#[command(
    name = "vector-search-engine",
    version,
    about = "Production-grade vector search engine (Rust + HNSW + ONNX)",
    long_about = "Ingest text with metadata. Search with semantic similarity via embeddings and HNSW."
)]
struct Cli {
    /// Enable verbose logging (RUST_LOG=debug also works)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Optional path to a sled database for future persistence (Phase 3+)
    #[arg(long, global = true, default_value = "data/vector.db")]
    db_path: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a single document (text + optional JSON metadata)
    Ingest {
        /// The text content to embed and store
        #[arg(short, long)]
        text: String,

        /// Optional JSON metadata (e.g. '{"source": "blog", "tags": ["rust"]}')
        #[arg(short, long, default_value = "{}")]
        meta: String,

        /// Print the generated document ID
        #[arg(long)]
        show_id: bool,
    },

    /// Search for nearest neighbors given a text query
    Search {
        /// Natural language query
        #[arg(short, long)]
        query: String,

        /// Maximum number of results to return
        #[arg(short, long, default_value_t = 5)]
        limit: usize,

        /// Show full metadata in output
        #[arg(long)]
        full: bool,

        /// Use hybrid search (keyword + vector, Phase 6)
        #[arg(long)]
        hybrid: bool,
    },

    /// Print engine statistics
    Stats,

    /// Start the HTTP API server (full Axum implementation)
    Serve {
        /// Host to bind to (env: HOST)
        #[arg(long, env = "HOST", default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to (env: PORT)
        #[arg(short, long, env = "PORT", default_value_t = 8080)]
        port: u16,

        /// Data directory for persistence (env: DATA_DIR)
        #[arg(long, env = "DATA_DIR", default_value = "data")]
        data_dir: String,
    },

    /// Generate a real embedding for a piece of text (debug / verification)
    Embed {
        #[arg(short, long)]
        text: String,
    },

    /// Download (or force re-download) the ONNX model + tokenizer
    DownloadModel {
        /// Force re-download even if the files already exist
        #[arg(long)]
        force: bool,
    },
}

fn init_logging(verbose: bool) {
    let filter = if verbose {
        EnvFilter::new("vector_search_engine=debug,info")
    } else {
        EnvFilter::from_default_env().add_directive("info".parse().unwrap())
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(false);

    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer);

    // Phase 7: Optional OpenTelemetry for traces (Jaeger, SigNoz, etc.)
    // Set OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 to enable
    if let Ok(otel_endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        if !otel_endpoint.is_empty() {
            use opentelemetry_otlp::WithExportConfig;
            use tracing_opentelemetry::layer;

            let tracer = opentelemetry_otlp::new_pipeline()
                .tracing()
                .with_exporter(
                    opentelemetry_otlp::new_exporter()
                        .tonic()
                        .with_endpoint(otel_endpoint),
                )
                .with_trace_config(
                    opentelemetry_sdk::trace::config()
                        .with_resource(opentelemetry_sdk::Resource::new(vec![
                            opentelemetry::KeyValue::new("service.name", "vector-search-engine"),
                        ])),
                )
                .install_batch(opentelemetry_sdk::runtime::Tokio)
                .expect("Failed to install OTEL tracer");

            let otel_layer = layer().with_tracer(tracer);

            registry.with(otel_layer).init();
            return;
        }
    }

    registry.init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    init_logging(cli.verbose);

    info!(
        "vector-search-engine starting (embed_dim={}, phase=2-persistence)",
        EMBED_DIM
    );

    // Phase 2: sled-backed persistence + HNSW (rebuilt from stored embeddings on load).
    // Documents survive restarts. HNSW graph is in-memory but rebuilt quickly from sled.
    let data_dir = std::path::PathBuf::from("data");
    let mut engine = vector_search_engine::VectorEngine::open_persistent(&data_dir, EngineConfig::default())
        .unwrap_or_else(|e| {
            warn!(error = %e, "failed to open persistent store, falling back to fresh in-memory");
            vector_search_engine::VectorEngine::new(EngineConfig::default())
        });

    match cli.command {
        Commands::Ingest { text, meta, show_id } => {
            let metadata: serde_json::Value = serde_json::from_str(&meta)
                .unwrap_or_else(|_| {
                    warn!("failed to parse metadata JSON, using empty object");
                    serde_json::json!({})
                });

            let embedding = embed(&text)?;
            let id = engine.ingest(text.clone(), embedding, metadata)?;

            if show_id {
                println!("{}", id);
            } else {
                println!(
                    "✓ Ingested document (id={})  text_len={}  meta={}",
                    id,
                    text.len(),
                    meta
                );
            }
            info!(document_id = %id, "ingested document");
            // Persistence is automatic via sled in open_persistent mode.
        }

        Commands::Search { query, limit, full, hybrid } => {
            if engine.is_empty() {
                println!("(no documents ingested yet in this session)");
                println!("Tip: run `cargo run -- ingest --text \"example text\"` first");
                return Ok(());
            }

            let results = if hybrid {
                engine.hybrid_search(&query, limit)?
            } else {
                let query_emb = embed(&query)?;
                engine.search(&query_emb, limit)?
            };

            println!("Top {} results for query: {:?} (hybrid={})", results.len(), query, hybrid);
            for (i, r) in results.iter().enumerate() {
                println!(
                    "{:2}. score={:.4}  id={}  text=\"{}\"",
                    i + 1,
                    r.score,
                    r.id,
                    truncate(&r.text, 80)
                );
                if full {
                    println!("    metadata: {}", serde_json::to_string(&r.metadata).unwrap());
                }
            }
        }

        Commands::Stats => {
            let stats = engine.stats();
            println!("=== Vector Engine Stats ===");
            println!("documents     : {}", stats.num_documents);
            println!("embedding_dim : {}", stats.embedding_dim);
            println!("index_type    : {}", stats.index_type);
            println!("(HNSW ANN mode)");
        }

        Commands::Serve { host, port, data_dir } => {
            println!("Starting Axum server on http://{}:{} ...", host, port);
            println!("Using data dir: {}", data_dir);
            println!("(Phase 8) Also starting gRPC on 0.0.0.0:50051");
            info!("loading persistent engine for server...");
            let data_path = std::path::PathBuf::from(data_dir);
            let mut collections = vector_search_engine::Collections::new(&data_path);
            let _ = collections.get_or_create("default", EngineConfig::default());

            // Share for both REST and gRPC (Phase 8)
            let collections = Arc::new(tokio::sync::Mutex::new(collections));

            // Run async servers
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                // Phase 8 gRPC (only when built with --features grpc)
                #[cfg(feature = "grpc")]
                {
                    let grpc_collections = collections.clone();
                    tokio::spawn(async move {
                        let grpc_addr = "0.0.0.0:50051";
                        if let Err(e) = vector_search_engine::grpc_stub::serve_grpc(grpc_addr, grpc_collections).await {
                            eprintln!("gRPC server error: {}", e);
                        }
                    });
                }
                #[cfg(not(feature = "grpc"))]
                {
                    println!("(gRPC disabled in this build - use --features grpc)");
                }

                // Start Axum REST
                if let Err(e) = vector_search_engine::api::run_server(&host, port, collections).await {
                    eprintln!("Axum server error: {}", e);
                }
            });
        }

        Commands::Embed { text } => {
            let emb = embed(&text)?;
            println!("dim: {}", emb.len());
            println!(
                "first 8 values: {:?}",
                &emb[..8.min(emb.len())]
            );
            // Show that normalization worked
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("L2 norm: {:.6} (should be ~1.0)", norm);
        }

        Commands::DownloadModel { force } => {
            println!("Downloading embedding model (all-MiniLM-L6-v2) ...");
            vector_search_engine::download_model_if_needed(force)?;
            println!("✓ Model ready at models/all-MiniLM-L6-v2/");
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// Legacy Phase 0 JSON persistence removed in Phase 2 (replaced by sled + HNSW rebuild).
// See VectorEngine::open_persistent for the current mechanism.

