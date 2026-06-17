# Vector Search Engine - Use Cases

A complete, from-scratch, production-grade **vector database / semantic search engine** written in Rust.

It lets you ingest natural language documents, generate high-quality embeddings locally, and retrieve the most semantically similar items with low latency — all without sending data to any external service.

## What makes it different

- **Fully local & private** — embeddings are produced using ONNX Runtime + the `all-MiniLM-L6-v2` model. No OpenAI, no cloud.
- **Hybrid search** — combines vector similarity with keyword overlap for best-of-both-worlds results.
- **Multi-tenancy** — named collections allow isolated indexes (great for SaaS or multi-team use).
- **Memory-efficient storage** — scalar quantization + real Product Quantization (k-means trained codebooks) dramatically reduces storage footprint while keeping search quality high.
- **Dual APIs** — full-featured REST (Axum) + gRPC (tonic).
- **Production ready** — persistence (sled), observability (OpenTelemetry + Prometheus-style metrics), Docker, Kubernetes, Helm, load testing, and CI with benchmark regression.
- **Small & auditable** — written in safe Rust with minimal dependencies.

The system is useful as a standalone semantic search service or as the retrieval layer inside RAG (Retrieval-Augmented Generation) applications.

## Predictable Costs (Core Benefit)

One of the strongest advantages of this project is **predictable and low costs**:

- **Zero per-token embedding fees**: Unlike solutions that call OpenAI, Cohere, or Voyage for embeddings, this engine runs the embedding model locally using ONNX. You pay only for the compute you run (your own hardware or fixed cloud instance).
- **No usage-based vector database pricing**: Avoids expensive per-query or per-GB pricing from managed services (Pinecone, Weaviate Cloud, Qdrant Cloud, etc.).
- **Fixed infrastructure costs**: Once deployed (on your servers, Kubernetes cluster, or cheap VPS), costs are predictable and scale with your hardware, not with query volume or data size in unpredictable ways.
- **Quantization support**: Real k-means Product Quantization and scalar quantization let you store 4x–8x+ more vectors in the same memory/disk for the same budget.
- **OpenAI-compatible endpoint**: You can swap it in for OpenAI embeddings in existing apps while keeping costs under control.

This makes it especially attractive for:
- High-volume applications
- Cost-sensitive startups or enterprises
- Regulated environments where sending data to third-party APIs is prohibited
- Long-running RAG systems where embedding costs would otherwise dominate the budget

## Real-World Use Cases

| Use Case                        | Description                                                                 | Why this engine fits |
|--------------------------------|-----------------------------------------------------------------------------|----------------------|
| **Private RAG for LLMs**       | Retrieval backend for internal company chatbots / agents that must never leave the network. | Local embeddings + low latency + full control over data + predictable costs (no per-token LLM retrieval costs). |
| **Enterprise Knowledge Base**  | Semantic search across wikis, tickets, design docs, research papers, and Slack exports. | Hybrid search + metadata filtering + collections for department isolation. |
| **Semantic Product / Content Recommendations** | "People who viewed this also looked at..." or "similar articles". | High-quality 384-d embeddings + fast HNSW + ability to mix with business rules. |
| **Duplicate & Plagiarism Detection** | Find near-duplicate documents, code, or support tickets. | Quantization for large corpora + easy similarity threshold queries. |
| **Developer Experience Tools** | Semantic code search ("find code that handles user authentication"). | Works great on code + comments; can be embedded in IDEs or internal tools. |
| **Legal, Compliance & Research** | Search millions of contracts, case files, or scientific papers by meaning rather than keywords. | Strong recall with HNSW, PQ compression for huge datasets, audit-friendly Rust. |
| **Customer Support Intelligence** | Automatically suggest previous solutions for incoming tickets. | Fast hybrid search + metadata (customer tier, product, etc.). |
| **Chatbot Long-Term Memory**   | Store conversation history or user facts and retrieve relevant context for the LLM. | Simple ingest + search API; collections per user or session. |

These workloads benefit from:
- **Data sovereignty** (everything stays on your infrastructure)
- **Predictable cost** (no per-token embedding fees)
- **Low latency** (sub-millisecond search after embedding)
- **Operational simplicity** (single binary + Docker/K8s manifests included)

## Key Features

- **Embeddings**: Local ONNX + all-MiniLM-L6-v2 (no external calls)
- **Indexing**: HNSW for fast ANN cosine search
- **Hybrid Search**: Vector + keyword (0.7 / 0.3 weighting with over-fetch)
- **Multi-tenancy**: Named collections with isolated indexes
- **Quantization**: Scalar + real Product Quantization (k-means trained)
- **Predictable Costs**: Local execution eliminates usage-based embedding and vector DB fees
- **APIs**: REST + gRPC + OpenAI-compatible embeddings endpoint
- **Storage**: sled persistence (documents + quantized vectors)
- **Observability**: OpenTelemetry traces + Prometheus-compatible metrics
- **Deployment**: Docker, Kubernetes, Helm charts
- **Tooling**: CLI, web UI with Chart.js visualizations, load tests, CI benchmarks

See the main [README.md](./README.md) for quick start, architecture, and deployment instructions.