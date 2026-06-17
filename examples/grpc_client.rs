//! Phase 9: basic Rust gRPC client example.
//! Requires the server built with --features grpc.
//! Uses tonic to call Embed/Search etc.
//!
//! Run server: cargo run --features grpc -- serve
//! Run: cargo run --example grpc_client

use tonic::Request;
use vector_search_engine::grpc_stub::pb::vector_service_client::VectorServiceClient;
use vector_search_engine::grpc_stub::pb::{EmbedRequest, SearchRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = VectorServiceClient::connect("http://[::1]:50051").await?;

    let embed_req = Request::new(EmbedRequest {
        texts: vec!["hello phase 9".to_string()],
        model: "all-MiniLM-L6-v2".to_string(),
    });
    let embed_resp = client.embed(embed_req).await?;
    println!("Embed dim: {}", embed_resp.into_inner().embeddings[0].values.len());

    let search_req = Request::new(SearchRequest {
        query: "hello".to_string(),
        limit: 3,
        collection: "default".to_string(),
        hybrid: false,
        min_score: None,
    });
    let search_resp = client.search(search_req).await?;
    println!("Search results: {}", search_resp.into_inner().results.len());

    println!("gRPC client example done (see tonic docs for full auth/tls).");
    Ok(())
}