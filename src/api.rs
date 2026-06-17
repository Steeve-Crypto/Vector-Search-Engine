//! REST API layer (Phase 3)
//!
//! Exposes the vector search engine over HTTP using Axum.
//! All endpoints are async and share state via Arc<Mutex<VectorEngine>>.
//!
//! Endpoints:
//! - POST /ingest      : single or batch document ingestion
//! - POST /search      : semantic search
//! - GET  /stats       : engine statistics
//! - GET  /health      : simple health check
//! - GET  /metrics     : Prometheus-compatible metrics (counters/gauges/histograms, no prometheus crate)
//!
//! Features:
//! - JSON request/response models
//! - Proper error handling (maps to HTTP status)
//! - CORS + request tracing middleware
//! - Rate limiting skeleton (via tower-http limit if desired)
//! - Uses the persistent engine when started via `serve`

use axum::{
    extract::{Json, Query, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Router,
};
use futures_util::stream::StreamExt;
use reqwest::Client;
use serde_json::Value;
use std::convert::Infallible;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use std::sync::LazyLock;
use tokio::sync::Mutex as TokioMutex;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, services::ServeDir, trace::TraceLayer};
use tower_governor::{governor::GovernorConfigBuilder, key_extractor::SmartIpKeyExtractor, GovernorLayer};
use sled;
use tracing::{info, instrument};

use crate::{collection::Collections, quantization::quantization_error, EngineConfig, SearchResult, embed};

// ============================================================================
// State
// ============================================================================

#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<TokioMutex<Collections>>,
}

// ============================================================================
// Request / Response Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub text: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default = "default_collection")]
    pub collection: String,
}

fn default_collection() -> String {
    "default".to_string()
}

#[derive(Debug, Deserialize)]
pub struct BatchIngestRequest {
    pub documents: Vec<IngestRequest>,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
    #[serde(default)]
    pub metadata_filter: Option<serde_json::Value>,  // Phase 6: e.g. {"source": "blog"} for equality filter on top-level metadata keys. Post-filter with over-fetch optimization.
    #[serde(default)]
    pub hybrid: bool,  // Phase 6: enable hybrid keyword + vector search
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default)]
    pub ef_search: Option<usize>,  // Phase 9: override HNSW ef for quality/speed
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Deserialize, Default)]
pub struct StatsQuery {
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub count: usize,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct StatsResponse {
    pub num_documents: usize,
    pub embedding_dim: usize,
    pub index_type: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchSearchRequest {
    pub queries: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub collection: String,
    #[serde(default)]
    pub hybrid: bool,
    #[serde(default)]
    pub ef_search: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct BatchSearchResponse {
    pub results: Vec<Vec<SearchResult>>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    pub status: String,
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
pub enum ApiError {
    Engine(crate::VectorError),
    Embed(crate::EmbedderError),
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::Engine(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Embed(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = serde_json::json!({
            "error": message
        });

        (status, Json(body)).into_response()
    }
}

impl From<crate::VectorError> for ApiError {
    fn from(e: crate::VectorError) -> Self {
        ApiError::Engine(e)
    }
}

impl From<crate::EmbedderError> for ApiError {
    fn from(e: crate::EmbedderError) -> Self {
        ApiError::Embed(e)
    }
}

// ============================================================================
// Handlers
// ============================================================================

#[instrument(skip(state), fields(collection = %payload.collection))]
pub async fn ingest_handler(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if payload.text.trim().is_empty() {
        return Err(ApiError::BadRequest("text cannot be empty".into()));
    }

    let _timer = LatencyTimer::new(&payload.collection, "false");
    let embedding = embed(&payload.text)?;
    let metadata = payload.metadata.unwrap_or(serde_json::Value::Null);

    // Phase 7: observe quantization error for this vector
    let qerr = quantization_error(&embedding);
    observe_quant_error(&payload.collection, qerr);

    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&payload.collection, EngineConfig::default())?;
    let id = engine.ingest(payload.text, embedding, metadata)?;
    // timer dropped here (records latency)

    info!(document_id = %id, collection = %payload.collection, "ingested via API");
    inc_ingest(&payload.collection);
    set_docs_gauge(&payload.collection, engine.len() as i64);
    let st = engine.stats();
    set_hnsw_gauge("num_vectors", &payload.collection, st.num_documents as i64);
    set_hnsw_gauge("max_connections", &payload.collection, st.hnsw_max_nb_connection as i64);
    set_hnsw_gauge("ef_construction", &payload.collection, st.hnsw_ef_construction as i64);
    set_hnsw_gauge("ef_search", &payload.collection, st.hnsw_default_ef_search as i64);

    Ok(Json(serde_json::json!({
        "id": id,
        "collection": payload.collection,
        "status": "ingested"
    })))
}

pub async fn batch_ingest_handler(
    State(state): State<AppState>,
    Json(payload): Json<BatchIngestRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if payload.documents.is_empty() {
        return Err(ApiError::BadRequest("documents array cannot be empty".into()));
    }

    let mut ids = Vec::with_capacity(payload.documents.len());

    for doc in payload.documents {
        if doc.text.trim().is_empty() {
            return Err(ApiError::BadRequest("text cannot be empty in batch".into()));
        }

        let embedding = embed(&doc.text)?;
        let metadata = doc.metadata.unwrap_or(serde_json::Value::Null);

        // Phase 7: per-item quant error + metrics
        let qerr = quantization_error(&embedding);
        observe_quant_error(&doc.collection, qerr);

        let mut cols = state.collections.lock().await;
        let engine = cols.get_or_create(&doc.collection, EngineConfig::default())?;
        let id = engine.ingest(doc.text, embedding, metadata)?;
        ids.push(id);
        // update per-collection gauge after each (small batches ok)
        set_docs_gauge(&doc.collection, engine.len() as i64);
        inc_ingest(&doc.collection);
    }

    info!(count = ids.len(), "batch ingested via API");

    Ok(Json(serde_json::json!({
        "ids": ids,
        "count": ids.len(),
        "status": "ingested"
    })))
}

#[utoipa::path(
    post,
    path = "/search",
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse)
    ),
    tag = "search"
)]
#[instrument(skip(state), fields(collection = %payload.collection, hybrid = %payload.hybrid))]
pub async fn search_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if payload.query.trim().is_empty() {
        return Err(ApiError::BadRequest("query cannot be empty".into()));
    }

    let limit = payload.limit.clamp(1, 1000);

    let _timer = LatencyTimer::new(&payload.collection, &payload.hybrid.to_string());
    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&payload.collection, EngineConfig::default())?;

    // Phase 6: over-fetch for post-filter optimization if metadata_filter or min_score present
    let fetch_k = if payload.metadata_filter.is_some() || payload.min_score.is_some() {
        limit * 5
    } else {
        limit
    };

    let mut results = if payload.hybrid {
        engine.hybrid_search(&payload.query, fetch_k)?
    } else {
        let query_emb = embed(&payload.query)?;
        if let Some(ef) = payload.ef_search {
            engine.search_with_ef(&query_emb, fetch_k, ef)?
        } else {
            engine.search(&query_emb, fetch_k)?
        }
    };
    // timer drops here and records latency

    inc_search(&payload.collection, &payload.hybrid.to_string());
    let n = engine.len() as i64;
    set_docs_gauge(&payload.collection, n);

    // Phase 7: keep HNSW gauges fresh on search path too
    let st = engine.stats();
    set_hnsw_gauge("num_vectors", &payload.collection, st.num_documents as i64);
    set_hnsw_gauge("max_connections", &payload.collection, st.hnsw_max_nb_connection as i64);
    set_hnsw_gauge("ef_construction", &payload.collection, st.hnsw_ef_construction as i64);
    set_hnsw_gauge("ef_search", &payload.collection, st.hnsw_default_ef_search as i64);

    if let Some(min) = payload.min_score {
        results.retain(|r| r.score >= min);
    }

    // Phase 6: Metadata filtering (post-filter optimization)
    // If filter present, we should have over-fetched from engine (see below), then apply here.
    if let Some(f) = &payload.metadata_filter {
        if let serde_json::Value::Object(map) = f {
            results.retain(|r| {
                map.iter().all(|(k, v)| {
                    r.metadata.get(k).map_or(false, |rv| rv == v)
                })
            });
        }
    }

    Ok(Json(SearchResponse {
        count: results.len(),
        results,
    }))
}

pub async fn stats_handler(
    State(state): State<AppState>,
    Query(q): Query<StatsQuery>,
) -> Json<StatsResponse> {
    let coll = q.collection.unwrap_or_else(|| "default".to_string());
    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&coll, EngineConfig::default()).unwrap();
    let stats = engine.stats();
    let n = engine.len() as i64;

    set_docs_gauge(&coll, n);

    // Phase 7: HNSW-specific gauges (per collection)
    set_hnsw_gauge("num_vectors", &coll, stats.num_documents as i64);
    set_hnsw_gauge("max_connections", &coll, stats.hnsw_max_nb_connection as i64);
    set_hnsw_gauge("ef_construction", &coll, stats.hnsw_ef_construction as i64);
    set_hnsw_gauge("ef_search", &coll, stats.hnsw_default_ef_search as i64);

    Json(StatsResponse {
        num_documents: stats.num_documents,
        embedding_dim: stats.embedding_dim,
        index_type: stats.index_type,
    })
}

pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

pub async fn batch_search_handler(
    State(state): State<AppState>,
    Json(payload): Json<BatchSearchRequest>,
) -> Result<Json<BatchSearchResponse>, ApiError> {
    if payload.queries.is_empty() {
        return Err(ApiError::BadRequest("queries cannot be empty".into()));
    }
    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&payload.collection, EngineConfig::default())?;

    let mut all = vec![];
    for q in payload.queries {
        let fetch_k = if payload.ef_search.is_some() { payload.limit * 2 } else { payload.limit }; // simple
        let res = if payload.hybrid {
            engine.hybrid_search(&q, fetch_k)?
        } else {
            let emb = embed(&q)?;
            if let Some(ef) = payload.ef_search {
                engine.search_with_ef(&emb, fetch_k, ef)?
            } else {
                engine.search(&emb, fetch_k)?
            }
        };
        all.push(res);
    }
    Ok(Json(BatchSearchResponse { results: all }))
}

// Lightweight metrics implementation (no `prometheus` crate).
// We maintain simple in-memory counters/gauges/histograms and render a
// Prometheus text exposition format on GET /metrics. This keeps full
// compatibility with Prometheus, VictoriaMetrics, Perses, Grafana, etc.
// while avoiding any extra dependency.
//
// All metrics are labeled by collection (and hybrid for searches).
// Cardinality is low (number of collections is small).

use std::collections::HashMap;
use std::time::Instant;

#[derive(Default, Clone)]
struct HistogramData {
    // buckets: (le_upper_bound, cumulative_count)
    buckets: Vec<(f64, u64)>,
    sum: f64,
    count: u64,
}

static INGEST_COUNTER: LazyLock<std::sync::Mutex<HashMap<String, u64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

static SEARCH_COUNTER: LazyLock<std::sync::Mutex<HashMap<(String, String), u64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

static DOCS_GAUGE: LazyLock<std::sync::Mutex<HashMap<String, i64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

static HNSW_NUM_VECTORS_GAUGE: LazyLock<std::sync::Mutex<HashMap<String, i64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
static HNSW_MAX_CONNECTIONS_GAUGE: LazyLock<std::sync::Mutex<HashMap<String, i64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
static HNSW_EF_CONSTRUCTION_GAUGE: LazyLock<std::sync::Mutex<HashMap<String, i64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
static HNSW_EF_SEARCH_GAUGE: LazyLock<std::sync::Mutex<HashMap<String, i64>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

static QUANT_ERROR_HIST: LazyLock<std::sync::Mutex<HashMap<String, HistogramData>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

static SEARCH_LATENCY_HIST: LazyLock<std::sync::Mutex<HashMap<(String, String), HistogramData>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

// Reasonable buckets for our data (quant error is small; latency in seconds).
fn quant_error_buckets() -> &'static [f64] {
    &[0.0005, 0.001, 0.003, 0.005, 0.01, 0.05, f64::INFINITY]
}
fn search_latency_buckets() -> &'static [f64] {
    &[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0, f64::INFINITY]
}

fn inc_ingest(collection: &str) {
    let mut m = INGEST_COUNTER.lock().unwrap();
    *m.entry(collection.to_string()).or_insert(0) += 1;
}

fn inc_search(collection: &str, hybrid: &str) {
    let mut m = SEARCH_COUNTER.lock().unwrap();
    *m.entry((collection.to_string(), hybrid.to_string())).or_insert(0) += 1;
}

fn set_docs_gauge(collection: &str, n: i64) {
    let mut m = DOCS_GAUGE.lock().unwrap();
    m.insert(collection.to_string(), n);
}

fn set_hnsw_gauge(name: &str, collection: &str, v: i64) {
    let map: &LazyLock<std::sync::Mutex<HashMap<String, i64>>> = match name {
        "num_vectors" => &HNSW_NUM_VECTORS_GAUGE,
        "max_connections" => &HNSW_MAX_CONNECTIONS_GAUGE,
        "ef_construction" => &HNSW_EF_CONSTRUCTION_GAUGE,
        "ef_search" => &HNSW_EF_SEARCH_GAUGE,
        _ => return,
    };
    let mut m = map.lock().unwrap();
    m.insert(collection.to_string(), v);
}

fn observe_quant_error(collection: &str, value: f64) {
    let mut m = QUANT_ERROR_HIST.lock().unwrap();
    let entry = m.entry(collection.to_string()).or_default();
    entry.count += 1;
    entry.sum += value;

    // Ensure we have slots for each bucket (first time)
    if entry.buckets.is_empty() {
        entry.buckets = quant_error_buckets().iter().map(|&le| (le, 0)).collect();
    }

    for (le, count) in &mut entry.buckets {
        if value <= *le {
            *count += 1;
        }
    }
}

fn record_search_latency(collection: &str, hybrid: &str, seconds: f64) {
    let mut m = SEARCH_LATENCY_HIST.lock().unwrap();
    let key = (collection.to_string(), hybrid.to_string());
    let entry = m.entry(key).or_default();
    entry.count += 1;
    entry.sum += seconds;

    if entry.buckets.is_empty() {
        entry.buckets = search_latency_buckets().iter().map(|&le| (le, 0)).collect();
    }
    for (le, count) in &mut entry.buckets {
        if seconds <= *le {
            *count += 1;
        }
    }
}

/// Simple RAII timer for latency (replaces the old prometheus HistogramTimer).
struct LatencyTimer {
    collection: String,
    hybrid: String,
    start: Instant,
}

impl LatencyTimer {
    fn new(collection: impl Into<String>, hybrid: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            hybrid: hybrid.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let secs = self.start.elapsed().as_secs_f64();
        record_search_latency(&self.collection, &self.hybrid, secs);
    }
}

pub async fn metrics_handler() -> impl IntoResponse {
    // Build Prometheus text format manually (no crate).
    let mut out = String::with_capacity(2048);

    // --- Counters ---
    {
        let m = INGEST_COUNTER.lock().unwrap();
        out.push_str("# HELP vector_ingest_total Total documents ingested\n");
        out.push_str("# TYPE vector_ingest_total counter\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!(
                "vector_ingest_total{{collection=\"{}\"}} {}\n",
                coll, val
            ));
        }
    }
    {
        let m = SEARCH_COUNTER.lock().unwrap();
        out.push_str("# HELP vector_search_total Total search requests\n");
        out.push_str("# TYPE vector_search_total counter\n");
        for ((coll, hyb), val) in m.iter() {
            out.push_str(&format!(
                "vector_search_total{{collection=\"{}\",hybrid=\"{}\"}} {}\n",
                coll, hyb, val
            ));
        }
    }

    // --- Gauges (docs + HNSW) ---
    {
        let m = DOCS_GAUGE.lock().unwrap();
        out.push_str("# HELP vector_docs_total Current number of documents in collection\n");
        out.push_str("# TYPE vector_docs_total gauge\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!("vector_docs_total{{collection=\"{}\"}} {}\n", coll, val));
        }
    }
    {
        let m = HNSW_NUM_VECTORS_GAUGE.lock().unwrap();
        out.push_str("# HELP hnsw_num_vectors Number of vectors in HNSW index per collection\n");
        out.push_str("# TYPE hnsw_num_vectors gauge\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!("hnsw_num_vectors{{collection=\"{}\"}} {}\n", coll, val));
        }
    }
    {
        let m = HNSW_MAX_CONNECTIONS_GAUGE.lock().unwrap();
        out.push_str("# HELP hnsw_max_connections Max connections (M) in HNSW per collection\n");
        out.push_str("# TYPE hnsw_max_connections gauge\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!("hnsw_max_connections{{collection=\"{}\"}} {}\n", coll, val));
        }
    }
    {
        let m = HNSW_EF_CONSTRUCTION_GAUGE.lock().unwrap();
        out.push_str("# HELP hnsw_ef_construction efConstruction parameter in HNSW per collection\n");
        out.push_str("# TYPE hnsw_ef_construction gauge\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!("hnsw_ef_construction{{collection=\"{}\"}} {}\n", coll, val));
        }
    }
    {
        let m = HNSW_EF_SEARCH_GAUGE.lock().unwrap();
        out.push_str("# HELP hnsw_ef_search Default efSearch parameter in HNSW per collection\n");
        out.push_str("# TYPE hnsw_ef_search gauge\n");
        for (coll, val) in m.iter() {
            out.push_str(&format!("hnsw_ef_search{{collection=\"{}\"}} {}\n", coll, val));
        }
    }

    // --- Histograms ---
    // quant_error
    {
        let h = QUANT_ERROR_HIST.lock().unwrap();
        out.push_str("# HELP quant_error Quantization error (RMS) histogram\n");
        out.push_str("# TYPE quant_error histogram\n");
        for (coll, data) in h.iter() {
            for (le, cnt) in &data.buckets {
                let le_str = if le.is_infinite() { "+Inf".to_string() } else { le.to_string() };
                out.push_str(&format!(
                    "quant_error_bucket{{collection=\"{}\",le=\"{}\"}} {}\n",
                    coll, le_str, cnt
                ));
            }
            out.push_str(&format!(
                "quant_error_sum{{collection=\"{}\"}} {}\n",
                coll, data.sum
            ));
            out.push_str(&format!(
                "quant_error_count{{collection=\"{}\"}} {}\n",
                coll, data.count
            ));
        }
    }

    // search latency (also covers embed+ingest time when labeled hybrid="false")
    {
        let h = SEARCH_LATENCY_HIST.lock().unwrap();
        out.push_str("# HELP vector_search_latency_seconds Search/ingest latency in seconds\n");
        out.push_str("# TYPE vector_search_latency_seconds histogram\n");
        for ((coll, hyb), data) in h.iter() {
            for (le, cnt) in &data.buckets {
                let le_str = if le.is_infinite() { "+Inf".to_string() } else { le.to_string() };
                out.push_str(&format!(
                    "vector_search_latency_seconds_bucket{{collection=\"{}\",hybrid=\"{}\",le=\"{}\"}} {}\n",
                    coll, hyb, le_str, cnt
                ));
            }
            out.push_str(&format!(
                "vector_search_latency_seconds_sum{{collection=\"{}\",hybrid=\"{}\"}} {}\n",
                coll, hyb, data.sum
            ));
            out.push_str(&format!(
                "vector_search_latency_seconds_count{{collection=\"{}\",hybrid=\"{}\"}} {}\n",
                coll, hyb, data.count
            ));
        }
    }

    (
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4; charset=utf-8")],
        out.into_bytes(),
    )
}

// Phase 6: OpenAI-compatible embeddings endpoint
// POST /v1/embeddings
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct OpenAIEmbedRequest {
    pub input: serde_json::Value, // string or array of strings
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "text-embedding-ada-002".to_string() // or our all-MiniLM
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct OpenAIEmbedResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbedData>,
    pub model: String,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct OpenAIEmbedData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct OpenAIUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// Phase 9: RAG Adapter for Private AI chat apps
// OpenAI-compatible /v1/chat/completions that performs retrieval-augmented generation.
// - Uses the local vector engine for context retrieval (embeddings + search).
// - Augments the conversation with retrieved documents.
// - Forwards to a private LLM backend (default: Ollama at http://localhost:11434/v1).
// This allows any OpenAI-compatible private chat UI (Open WebUI, etc.) to connect to this server
// for both embeddings (/v1/embeddings) and RAG chat (/v1/chat/completions) using a single base URL.

#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(default)]
    pub stream: bool,
    // Optional: specify collection for retrieval
    #[serde(default)]
    pub collection: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, utoipa::ToSchema)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct RetrieveRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default)]
    pub hybrid: bool,
}

#[utoipa::path(
    post,
    path = "/v1/embeddings",
    request_body = OpenAIEmbedRequest,
    responses(
        (status = 200, description = "Embeddings", body = OpenAIEmbedResponse)
    ),
    tag = "embed"
)]
pub async fn openai_embeddings(
    State(_state): State<AppState>,
    Json(payload): Json<OpenAIEmbedRequest>,
) -> Result<Json<OpenAIEmbedResponse>, ApiError> {
    let mut inputs: Vec<String> = vec![];
    match payload.input {
        serde_json::Value::String(s) => inputs.push(s),
        serde_json::Value::Array(arr) => {
            for v in arr {
                if let serde_json::Value::String(s) = v {
                    inputs.push(s);
                }
            }
        }
        _ => return Err(ApiError::BadRequest("input must be string or array of strings".into())),
    }

    let mut data = vec![];
    let mut total_tokens = 0;
    for (i, text) in inputs.iter().enumerate() {
        let emb = embed(text)?;
        // rough token count
        let tokens = text.split_whitespace().count();
        total_tokens += tokens;
        data.push(OpenAIEmbedData {
            object: "embedding".to_string(),
            embedding: emb,
            index: i,
        });
    }

    Ok(Json(OpenAIEmbedResponse {
        object: "list".to_string(),
        data,
        model: payload.model,
        usage: OpenAIUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

// RAG Adapter: OpenAI-compatible chat with retrieval
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = OpenAIChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = Value)
    ),
    tag = "chat"
)]
pub async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<OpenAIChatRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if payload.messages.is_empty() {
        return Err(ApiError::BadRequest("messages cannot be empty".into()));
    }

    // Extract query from last user message for retrieval
    let query = payload
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.trim().to_string())
        .unwrap_or_default();

    let collection = payload
        .collection
        .unwrap_or_else(|| "default".to_string());

    let top_k: usize = std::env::var("RAG_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    // Retrieve context using the vector engine
    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&collection, EngineConfig::default())?;

    let mut context_docs = vec![];
    let mut citations = vec![];
    if !query.is_empty() {
        // Use semantic search (can be hybrid if desired)
        let mut results = if query.split_whitespace().count() > 3 {
            // heuristic: use hybrid for longer queries
            engine.hybrid_search(&query, top_k)?
        } else {
            let emb = embed(&query)?;
            engine.search(&emb, top_k)?
        };
        results = re_rank_stub(results); // Phase 10 stub
        for (i, r) in results.iter().enumerate() {
            context_docs.push(format!("- [{}]: {}", i+1, r.text));
            citations.push(format!("Source {}: {}", i+1, r.id));
        }
    }

    // Build augmented messages with context (configurable templates - Phase 9)
    let context_template = std::env::var("RAG_CONTEXT_TEMPLATE")
        .unwrap_or_else(|_| "Context:\n{context}".to_string());
    let system_template = std::env::var("RAG_SYSTEM_TEMPLATE")
        .unwrap_or_else(|_| "You are a helpful assistant with access to the following private knowledge base. Use the context below to answer accurately. If the answer is not in the context, say so.\n\n{context}".to_string());

    let mut augmented = payload.messages.clone();
    if !context_docs.is_empty() {
        let context_text = context_docs.join("\n");
        let context_inject = context_template.replace("{context}", &context_text);
        let system_context = system_template.replace("{context}", &context_inject);
        // Prepend or update system message
        if let Some(first) = augmented.first_mut() {
            if first.role == "system" {
                first.content = format!("{}\n\n{}", system_context, first.content);
            } else {
                augmented.insert(0, OpenAIChatMessage {
                    role: "system".to_string(),
                    content: system_context,
                });
            }
        } else {
            augmented.insert(0, OpenAIChatMessage {
                role: "system".to_string(),
                content: system_context,
            });
        }
    }

    // Forward to private LLM backend (configurable, e.g. Ollama)
    let llm_base = std::env::var("LLM_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let llm_url = format!("{}/chat/completions", llm_base.trim_end_matches('/'));

    let forward_body = serde_json::json!({
        "model": payload.model,
        "messages": augmented,
        "stream": payload.stream,
    });

    let client = Client::new();
    let llm_req = client.post(&llm_url).json(&forward_body);
    let llm_resp = llm_req.send().await.map_err(|e| ApiError::Internal(format!("LLM backend error: {}", e)))?;

    if !llm_resp.status().is_success() {
        let status = llm_resp.status();
        let err_text = llm_resp.text().await.unwrap_or_default();
        return Err(ApiError::Internal(format!("LLM backend returned {}: {}", status, err_text)));
    }

    if payload.stream {
        // Streaming support: proxy SSE from LLM
        let stream = llm_resp
            .bytes_stream()
            .map(|result| {
                let data = match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes[..]);
                        // Forward OpenAI-style SSE chunks as-is
                        s.trim().to_string()
                    }
                    Err(e) => format!("error: {}", e),
                };
                Ok::<_, Infallible>(Event::default().data(data))
            });
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        return Ok(sse.into_response());
    } else {
        let body: Value = llm_resp.json().await.map_err(|e| ApiError::Internal(format!("Failed to parse LLM response: {}", e)))?;
        return Ok(Json(body).into_response());
    }
}

// Phase 10 re-ranking stub (simple score boost for now)
fn re_rank_stub(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    // Stub: boost longer docs or by score, in real would use cross-encoder
    for r in &mut results {
        r.score = (r.score + (r.text.len() as f32 / 1000.0)).min(1.0);
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// Retrieval-only helper for other frameworks (Phase 9)
#[utoipa::path(
    post,
    path = "/v1/retrieve",
    request_body = RetrieveRequest,
    responses(
        (status = 200, description = "Retrieved documents", body = Vec<SearchResult>)
    ),
    tag = "search"
)]
pub async fn retrieve_handler(
    State(state): State<AppState>,
    Json(payload): Json<RetrieveRequest>,
) -> Result<Json<Vec<SearchResult>>, ApiError> {
    let mut cols = state.collections.lock().await;
    let engine = cols.get_or_create(&payload.collection, EngineConfig::default())?;

    let results = if payload.hybrid {
        engine.hybrid_search(&payload.query, payload.limit)?
    } else {
        let emb = embed(&payload.query)?;
        engine.search(&emb, payload.limit)?
    };
    Ok(Json(results))
}

// ============================================================================
// Router + Server
// ============================================================================

/// Simple API key auth middleware (Phase 4/9)
/// Checks for X-API-Key header if API_KEY env var is set.
/// Phase 11: per-collection keys, mTLS for gRPC, audit logging.
async fn api_key_auth(
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Ok(required_key) = env::var("API_KEY") {
        if !required_key.is_empty() {
            let headers = req.headers();
            if let Some(key) = headers.get("x-api-key") {
                if key.to_str().map(|k| k == required_key).unwrap_or(false) {
                    return Ok(next.run(req).await);
                }
            }
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

// Persistent per-IP rate limiting using sled (replaces or supplements in-memory for durability)
#[allow(dead_code)]
static RATE_DB: LazyLock<std::sync::Mutex<sled::Db>> = LazyLock::new(|| {
    let _ = std::fs::create_dir_all("data");
    std::sync::Mutex::new(sled::open("data/rate.sled").expect("failed to open rate sled db"))
});

#[derive(OpenApi)]
#[openapi(
    paths(search_handler, openai_chat_completions, retrieve_handler, openai_embeddings),
    tags(
        (name = "search", description = "Vector search operations"),
        (name = "embed", description = "Embedding operations"),
        (name = "chat", description = "RAG chat completions")
    )
)]
struct ApiDoc;

#[allow(dead_code)]
async fn persistent_rate_limit_middleware(
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1")
        .split(',')
        .next()
        .unwrap_or("127.0.0.1")
        .trim()
        .to_string();

    let key = format!("rate:{}", ip).into_bytes();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    const WINDOW: u64 = 60;
    const MAX: u32 = 30;

    let (ts, c) = {
        let db = RATE_DB.lock().unwrap();
        if let Some(old) = db.get(&key).ok().flatten() {
            if let Ok((old_ts, old_c)) = bincode::deserialize::<(u64, u32)>(&old) {
                if now - old_ts > WINDOW {
                    (now, 1u32)
                } else if old_c < MAX {
                    (old_ts, old_c + 1)
                } else {
                    return Err(StatusCode::TOO_MANY_REQUESTS);
                }
            } else {
                (now, 1)
            }
        } else {
            (now, 1)
        }
    }; // guard dropped here

    {
        let db = RATE_DB.lock().unwrap();
        let _ = db.insert(key, bincode::serialize(&(ts, c)).unwrap());
        let _ = db.flush();
    }

    Ok(next.run(req).await)
}



/// Build the Axum router with all routes and middleware.
/// Also serves a simple HTMX demo UI at "/" from ./static
pub fn create_router(collections: Collections) -> Router {
    let state = AppState {
        collections: Arc::new(TokioMutex::new(collections)),
    };

    // Rate limiter config (per-IP by default)
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(5) // 5 req/sec
        .burst_size(10)
        .key_extractor(SmartIpKeyExtractor)
        .finish()
        .unwrap();

    Router::new()
        // API endpoints (protected by auth if API_KEY set)
        .route("/ingest", post(ingest_handler))
        .route("/ingest/batch", post(batch_ingest_handler))
        .route("/search", post(search_handler))
        .route("/search/batch", post(batch_search_handler))
        .route("/stats", get(stats_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/embeddings", post(openai_embeddings))  // Phase 6 OpenAI compat
        .route("/v1/chat/completions", post(openai_chat_completions))  // Phase 9 RAG Adapter for private AI chat apps
        .route("/v1/retrieve", post(retrieve_handler))  // Phase 9 retrieval-only helper for frameworks
        .route("/api-docs/openapi.json", get(|| async { Json(ApiDoc::openapi()) }))
        // Phase 4: Simple HTMX demo UI
        .nest_service("/ui", ServeDir::new("static").precompressed_gzip())
        // Redirect root to the nice UI
        .route("/", get(|| async { axum::response::Redirect::to("/ui/") }))
        .layer(middleware::from_fn(api_key_auth))
        .layer(GovernorLayer::new(governor_conf))  // re-enabled rate layer with dep (tower_governor)
        // persistent_rate_limit_middleware available but not layered due to type (uses sled for durability)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024)) // 2MB body limit
        .with_state(state)
}

/// Run the HTTP server (called from CLI serve command).
/// collections is wrapped for sharing with gRPC (Phase 8).
pub async fn run_server(
    host: &str,
    port: u16,
    collections: Arc<TokioMutex<Collections>>,
) -> anyhow::Result<()> {
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on http://{}", addr);

    let app = create_router_from_arc(collections);

    // Use with_connect_info so PeerIpKeyExtractor (and fallbacks in Smart) can get remote addr
    let app = app.into_make_service_with_connect_info::<std::net::SocketAddr>();
    axum::serve(listener, app).await?;
    Ok(())
}

/// Internal: create router from already-wrapped collections (used by gRPC sharing path).
pub fn create_router_from_arc(collections: Arc<TokioMutex<Collections>>) -> Router {
    let state = AppState { collections };
    // ... same middleware setup as create_router
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(5)
        .burst_size(10)
        .key_extractor(SmartIpKeyExtractor)
        .finish()
        .unwrap();

    Router::new()
        .route("/ingest", post(ingest_handler))
        .route("/ingest/batch", post(batch_ingest_handler))
        .route("/search", post(search_handler))
        .route("/stats", get(stats_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/embeddings", post(openai_embeddings))
        .nest_service("/ui", ServeDir::new("static").precompressed_gzip())
        .route("/", get(|| async { axum::response::Redirect::to("/ui/") }))
        .layer(middleware::from_fn(api_key_auth))
        .layer(GovernorLayer::new(governor_conf))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        .with_state(state)
}