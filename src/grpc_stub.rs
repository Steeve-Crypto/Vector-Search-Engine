//! Phase 8: Real gRPC server implementation (behind "grpc" feature).
//!
//! To enable:
//!   cargo build --features grpc
//!   (requires protoc / protobuf-compiler installed for build.rs)
//!
//! The implementation re-uses the same Collections as the Axum REST server.

#[cfg(feature = "grpc")]
mod grpc_impl {
    use crate::{collection::Collections, embed, EngineConfig, SearchResult};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex as TokioMutex;
    use tonic::{Request, Response, Status};

    pub mod pb {
        tonic::include_proto!("vectorsearch");
    }

    use pb::vector_service_server::{VectorService, VectorServiceServer};
    use pb::{
        EmbedRequest, EmbedResponse, Embedding, IngestRequest, IngestResponse,
        SearchRequest, SearchResponse, SearchResult as PbSearchResult, StatsRequest, StatsResponse,
    };

    /// gRPC service implementation that wraps the engine collections.
    pub struct VectorEngineGrpc {
        collections: Arc<TokioMutex<Collections>>,
    }

    impl VectorEngineGrpc {
        pub fn new(collections: Arc<TokioMutex<Collections>>) -> Self {
            Self { collections }
        }
    }

    #[tonic::async_trait]
    impl VectorService for VectorEngineGrpc {
        async fn embed(
            &self,
            request: Request<EmbedRequest>,
        ) -> Result<Response<EmbedResponse>, Status> {
            let req = request.into_inner();
            let mut embs = Vec::new();

            for text in req.texts {
                match embed(&text) {
                    Ok(v) => embs.push(Embedding { values: v }),
                    Err(e) => return Err(Status::internal(format!("embed error: {}", e))),
                }
            }

            Ok(Response::new(EmbedResponse {
                embeddings: embs,
                model: req.model.unwrap_or_else(|| "all-MiniLM-L6-v2".to_string()),
            }))
        }

        async fn search(
            &self,
            request: Request<SearchRequest>,
        ) -> Result<Response<SearchResponse>, Status> {
            let req = request.into_inner();
            let limit = req.limit.clamp(1, 1000) as usize;
            let coll = if req.collection.is_empty() { "default".to_string() } else { req.collection };

            let mut cols = self.collections.lock().await;
            let engine = cols
                .get_or_create(&coll, EngineConfig::default())
                .map_err(|e| Status::internal(e.to_string()))?;

            let fetch_k = if req.min_score.is_some() { limit * 5 } else { limit };

            let mut results: Vec<SearchResult> = if req.hybrid {
                engine
                    .hybrid_search(&req.query, fetch_k)
                    .map_err(|e| Status::internal(e.to_string()))?
            } else {
                let qemb = embed(&req.query).map_err(|e| Status::internal(e.to_string()))?;
                engine
                    .search(&qemb, fetch_k)
                    .map_err(|e| Status::internal(e.to_string()))?
            };

            if let Some(min) = req.min_score {
                results.retain(|r| r.score >= min);
            }

            let pb_results: Vec<PbSearchResult> = results
                .into_iter()
                .map(|r| PbSearchResult {
                    id: r.id.to_string(),
                    text: r.text,
                    metadata: flatten_metadata(r.metadata),
                    score: r.score,
                })
                .collect();

            Ok(Response::new(SearchResponse {
                count: pb_results.len() as u32,
                results: pb_results,
            }))
        }

        async fn ingest(
            &self,
            request: Request<IngestRequest>,
        ) -> Result<Response<IngestResponse>, Status> {
            let req = request.into_inner();
            if req.text.trim().is_empty() {
                return Err(Status::invalid_argument("text cannot be empty"));
            }

            let coll = if req.collection.is_empty() { "default".to_string() } else { req.collection };
            let metadata: serde_json::Value = req
                .metadata
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();

            let embedding = embed(&req.text).map_err(|e| Status::internal(e.to_string()))?;

            let mut cols = self.collections.lock().await;
            let engine = cols
                .get_or_create(&coll, EngineConfig::default())
                .map_err(|e| Status::internal(e.to_string()))?;

            let id = engine
                .ingest(req.text, embedding, metadata)
                .map_err(|e| Status::internal(e.to_string()))?;

            Ok(Response::new(IngestResponse {
                id: id.to_string(),
                collection: coll,
            }))
        }

        async fn stats(
            &self,
            request: Request<StatsRequest>,
        ) -> Result<Response<StatsResponse>, Status> {
            let req = request.into_inner();
            let coll = if req.collection.is_empty() { "default".to_string() } else { req.collection };

            let mut cols = self.collections.lock().await;
            let engine = cols
                .get_or_create(&coll, EngineConfig::default())
                .map_err(|e| Status::internal(e.to_string()))?;
            let st = engine.stats();

            Ok(Response::new(StatsResponse {
                num_documents: st.num_documents as u64,
                embedding_dim: st.embedding_dim as u32,
                index_type: st.index_type,
            }))
        }
    }

    fn flatten_metadata(meta: serde_json::Value) -> HashMap<String, String> {
        match meta {
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| (k, v.to_string().trim_matches('"').to_string()))
                .collect(),
            _ => HashMap::new(),
        }
    }

    /// Start the gRPC server (non-blocking, usually spawned).
    pub async fn serve_grpc(
        addr: &str,
        collections: Arc<TokioMutex<Collections>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let svc = VectorEngineGrpc::new(collections);
        let server = VectorServiceServer::new(svc);

        println!("gRPC listening on {}", addr);
        tonic::transport::Server::builder()
            .add_service(server)
            .serve(addr.parse()?)
            .await?;
        Ok(())
    }
}

#[cfg(feature = "grpc")]
pub use grpc_impl::*;

#[cfg(not(feature = "grpc"))]
pub mod pb {
    // Placeholder so the module always exists.
}

#[cfg(not(feature = "grpc"))]
/// Stub when gRPC feature disabled.
pub async fn serve_grpc(
    _addr: &str,
    _collections: std::sync::Arc<tokio::sync::Mutex<crate::collection::Collections>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("gRPC support not compiled in. Rebuild with --features grpc (and install protoc).");
    Ok(())
}
