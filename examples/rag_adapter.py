#!/usr/bin/env python3
"""
Phase 9: Python RAG Adapter example for Private AI chat apps.
Uses the vector server's REST (or gRPC via grpcio) for retrieval,
augments prompt, calls LLM (Ollama or OpenAI compat).

Requires: pip install requests openai

Run:
  python examples/rag_adapter.py
  Then point chat app to http://localhost:8000 (if using fastapi) or use the function.

This is a simple script; for full server use the Rust adapter binary or integrated.
"""

import os
import json
import requests

VECTOR_SERVER = os.getenv("VECTOR_SERVER", "http://localhost:8080")
LLM_BASE = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
COLLECTION = os.getenv("RAG_COLLECTION", "default")

def retrieve(query: str, limit: int = 5, hybrid: bool = False):
    url = f"{VECTOR_SERVER}/v1/retrieve"
    payload = {
        "query": query,
        "limit": limit,
        "collection": COLLECTION,
        "hybrid": hybrid
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def chat_with_rag(messages, model="llama3.1", stream=False):
    # Extract last user query
    query = ""
    for m in reversed(messages):
        if m["role"] == "user":
            query = m["content"]
            break

    # Retrieve
    docs = retrieve(query) if query else []
    context = "\n".join([d.get("text", "") for d in docs])

    # Augment last message or add system
    augmented = list(messages)
    if context:
        sys_msg = {"role": "system", "content": f"Use this context:\n{context}\n\nAnswer the question."}
        if augmented and augmented[0]["role"] == "system":
            augmented[0]["content"] = sys_msg["content"] + "\n\n" + augmented[0]["content"]
        else:
            augmented.insert(0, sys_msg)

    # Call LLM
    url = f"{LLM_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": augmented,
        "stream": stream
    }
    r = requests.post(url, json=payload, timeout=120, stream=stream)
    r.raise_for_status()
    if stream:
        for line in r.iter_lines():
            if line:
                print(line.decode())
    else:
        return r.json()

if __name__ == "__main__":
    msgs = [{"role": "user", "content": "What is the vector engine about?"}]
    resp = chat_with_rag(msgs)
    print(json.dumps(resp, indent=2))