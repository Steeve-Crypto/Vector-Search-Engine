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
//! - GET  /metrics     : Prometheus metrics (basic counters + latency histogram)
//!
//! Features:
//! - JSON request/response models
//! - Proper error handling (maps to HTTP status)
//! - CORS + request tracing middleware
//! - Rate limiting skeleton (via tower-http limit if desired)
//! - Uses the persistent engine when started via `serve`

use axum::{
    extract::{Json, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use prometheus::{Encoder, TextEncoder};
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;
use std::sync::LazyLock;
use tokio::sync::Mutex;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, services::ServeDir, trace::TraceLayer};
use tower_governor::{governor::GovernorConfigBuilder, key_extractor::SmartIpKeyExtractor, GovernorLayer};
use sled;
use tracing::info;

use crate::{SearchResult, VectorEngine, embed};

// ============================================================================
// State
// ============================================================================

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<VectorEngine>>,
}

// ============================================================================
// Request / Response Models
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub text: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct BatchIngestRequest {
    pub documents: Vec<IngestRequest>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    // Future: filters, ef_search override, etc.
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub num_documents: usize,
    pub embedding_dim: usize,
    pub index_type: String,
}

#[derive(Debug, Serialize)]
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

pub async fn ingest_handler(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if payload.text.trim().is_empty() {
        return Err(ApiError::BadRequest("text cannot be empty".into()));
    }

    let timer = SEARCH_LATENCY.with_label_values(&["ingest"]).start_timer();
    let embedding = embed(&payload.text)?;
    let metadata = payload.metadata.unwrap_or(serde_json::Value::Null);

    let mut engine = state.engine.lock().await;
    let id = engine.ingest(payload.text, embedding, metadata)?;
    drop(timer);

    info!(document_id = %id, "ingested via API");
    INGEST_COUNTER.inc();

    Ok(Json(serde_json::json!({
        "id": id,
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

        let mut engine = state.engine.lock().await;
        let id = engine.ingest(doc.text, embedding, metadata)?;
        ids.push(id);
    }

    info!(count = ids.len(), "batch ingested via API");

    Ok(Json(serde_json::json!({
        "ids": ids,
        "count": ids.len(),
        "status": "ingested"
    })))
}

pub async fn search_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if payload.query.trim().is_empty() {
        return Err(ApiError::BadRequest("query cannot be empty".into()));
    }

    let limit = payload.limit.clamp(1, 1000);

    let query_emb = embed(&payload.query)?;

    let timer = SEARCH_LATENCY.with_label_values(&["search"]).start_timer();
    let engine = state.engine.lock().await;
    let results = engine.search(&query_emb, limit)?;
    drop(timer); // ends timer
    SEARCH_COUNTER.inc();

    Ok(Json(SearchResponse {
        count: results.len(),
        results,
    }))
}

pub async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    let engine = state.engine.lock().await;
    let stats = engine.stats();

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

// Basic Prometheus metrics (incremented from handlers if desired)
static INGEST_COUNTER: std::sync::LazyLock<prometheus::IntCounter> =
    std::sync::LazyLock::new(|| {
        prometheus::register_int_counter!("vector_ingest_total", "Total documents ingested")
            .expect("failed to register ingest counter")
    });

static SEARCH_COUNTER: std::sync::LazyLock<prometheus::IntCounter> =
    std::sync::LazyLock::new(|| {
        prometheus::register_int_counter!("vector_search_total", "Total search requests")
            .expect("failed to register search counter")
    });

static SEARCH_LATENCY: std::sync::LazyLock<prometheus::HistogramVec> =
    std::sync::LazyLock::new(|| {
        prometheus::register_histogram_vec!(
            "vector_search_latency_seconds",
            "Search latency in seconds",
            &["endpoint"]
        )
        .expect("failed to register latency histogram")
    });

pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    (
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        buffer,
    )
}

// ============================================================================
// Router + Server
// ============================================================================

/// Simple API key auth middleware (Phase 4)
/// Checks for X-API-Key header if API_KEY env var is set.
/// For demo, if not set, allows all.
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
pub fn create_router(engine: VectorEngine) -> Router {
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
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
        .route("/stats", get(stats_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
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
pub async fn run_server(host: &str, port: u16, engine: VectorEngine) -> anyhow::Result<()> {
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on http://{}", addr);

    let app = create_router(engine);

    // Use with_connect_info so PeerIpKeyExtractor (and fallbacks in Smart) can get remote addr
    let app = app.into_make_service_with_connect_info::<std::net::SocketAddr>();
    axum::serve(listener, app).await?;
    Ok(())
}