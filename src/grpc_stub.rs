//! gRPC stub for Phase 6.
//! 
//! To implement full gRPC, add tonic = "0.10" and tonic-build to build.rs for .proto.
//! Example proto for embeddings:
//! service VectorService {
//!   rpc Embed (EmbedRequest) returns (EmbedResponse);
//! }
//! 
//! For now, this is a stub showing the interface.
//! The OpenAI /v1/embeddings in api.rs serves as compatible endpoint.

use crate::embed;

/// Stub request.
#[derive(Debug)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

/// Stub response.
#[derive(Debug)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

pub fn embed_stub(req: EmbedRequest) -> EmbedResponse {
    let embs = req.texts.into_iter().map(|t| embed(&t).unwrap_or_default()).collect();
    EmbedResponse { embeddings: embs }
}

// To use with tonic:
// #[tokio::main]
// async fn main() { ... server ... }
