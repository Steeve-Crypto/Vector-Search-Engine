//! Standalone RAG Adapter binary (Phase 9).
//! Connects a private vector search server (this project's REST) to a private LLM.
//! Exposes OpenAI compatible /v1/chat/completions with retrieval.
//!
//! Usage:
//!   cargo run --bin rag_adapter
//!   Set VECTOR_SERVER=http://localhost:8080 (default)
//!   Set LLM_BASE_URL=http://localhost:11434/v1
//!
//! Then point your Private AI chat app to http://localhost:3000 (or configured).

use axum::{extract::Json, response::IntoResponse, routing::post, Router};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::net::SocketAddr;
use tokio::net::TcpListener;

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    collection: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() {
    let vector_server = std::env::var("VECTOR_SERVER").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let llm_base = std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

    println!("RAG Adapter starting. Vector: {}, LLM: {}", vector_server, llm_base);

    let app = Router::new()
        .route("/v1/chat/completions", post(move |payload: Json<ChatRequest>| {
            let vs = vector_server.clone();
            let lb = llm_base.clone();
            async move { handle_chat(payload, vs, lb).await }
        }));

    let addr: SocketAddr = "0.0.0.0:3000".parse().unwrap();
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("Adapter listening on http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat(payload: Json<ChatRequest>, vector_server: String, llm_base: String) -> impl IntoResponse {
    let query = payload.messages.iter().rev().find(|m| m.role == "user")
        .map(|m| m.content.clone()).unwrap_or_default();
    let coll = payload.collection.clone().unwrap_or_else(|| "default".to_string());

    // Retrieve from vector server (use /v1/retrieve or /search)
    let client = Client::new();
    let retrieve_url = format!("{}/v1/retrieve", vector_server.trim_end_matches('/'));
    let ret_body = serde_json::json!({
        "query": query,
        "limit": 5,
        "collection": coll,
        "hybrid": false
    });
    let ret_resp = client.post(&retrieve_url).json(&ret_body).send().await;
    let mut context_docs = vec![];
    if let Ok(r) = ret_resp {
        if let Ok(docs) = r.json::<Vec<serde_json::Value>>().await {
            for d in docs {
                if let Some(text) = d.get("text").and_then(|t| t.as_str()) {
                    context_docs.push(text.to_string());
                }
            }
        }
    }

    // Augment
    let mut augmented = payload.messages.clone();
    if !context_docs.is_empty() {
        let ctx = context_docs.join("\n- ");
        let sys = format!("Use this context for the answer:\n- {}\n\n", ctx);
        if let Some(first) = augmented.first_mut() {
            if first.role == "system" {
                first.content = format!("{}\n\n{}", sys, first.content);
            } else {
                augmented.insert(0, ChatMessage { role: "system".to_string(), content: sys });
            }
        }
    }

    // Forward to LLM
    let llm_url = format!("{}/chat/completions", llm_base.trim_end_matches('/'));
    let fwd = serde_json::json!({
        "model": payload.model,
        "messages": augmented,
        "stream": payload.stream,
    });
    let llm_r = client.post(&llm_url).json(&fwd).send().await;
    match llm_r {
        Ok(r) => {
            if payload.stream {
                // simple non streaming proxy for bin, or implement stream
                let txt = r.text().await.unwrap_or_default();
                (axum::http::StatusCode::OK, txt).into_response()
            } else {
                let v: Value = r.json().await.unwrap_or(serde_json::json!({}));
                axum::Json(v).into_response()
            }
        }
        Err(e) => (axum::http::StatusCode::BAD_GATEWAY, format!("LLM error: {}", e)).into_response(),
    }
}