# ADR: Design Cache MCP Server Architecture

* **Status:** Accepted
* **Date:** 2026-03-22
* **Project:** mcp_server_design_cache

## Context and Problem Statement
AI agents often suffer from context loss and token limits during long, complex design conversations. There is a critical need for an "external memory" cache that can efficiently store and retrieve both high-level project goals and granular design ideas, while simultaneously minimizing token usage and ensuring rapid context retrieval.

## Decision Drivers
* **Token Efficiency:** Must offload conversation and design history from the active LLM context window.
* **Accuracy & Speed:** Requires fast and highly relevant context retrieval.
* **Data Organization:** Need to support hierarchical caching (project-level vs. idea-level discussions).
* **Security & Stability:** Must prevent abuse (rate limiting) and ensure safe database queries.

## Considered Options
1. **In-Memory Cache (e.g., Redis) + Separate Vector DB (e.g., Pinecone/Weaviate):** Fast retrieval, but introduces significant architectural complexity and operational overhead by requiring multiple distinct datastores.
2. **PostgreSQL 18 with pgvector:** Utilizing a single relational database to handle structured metadata, time-series organization, Full-Text Search (FTS), and vector embeddings seamlessly.

## Decision Outcome
Chosen Option: **PostgreSQL 18 with pgvector and Psycopg 3**

We decided to build the Design Cache MCP Server using Python Asyncio, FastMCP, and PostgreSQL 18. This approach centralizes the storage layer and provides a robust, fully-featured backend:
- **pgvector & sentence-transformers:** Enables semantic vector search by generating local embeddings (`all-MiniLM-L6-v2`) and storing them in Postgres, allowing for similarity searches directly within the database.
- **PostgreSQL FTS:** Native full-text search capabilities using GIN indexes allow for sub-millisecond keyword retrieval. The application implements Hybrid Search (Keyword + Semantic).
- **Psycopg 3:** Provides high-performance, native async database connections with an `AsyncConnectionPool`. It natively parameterizes queries, effectively mitigating SQL injection risks without degrading markdown content.
- **UUIDv7:** Adopted for primary keys to provide native, time-sortable IDs, making activity and history retrieval inherently efficient without complex secondary indexing.
- **Security:** Implemented an in-memory Thread-safe Rate Limiter (capped at 60 RPM) and applied Role-Based Access Control (RBAC) at the database layer (separating `design_readonly` and `design_readwrite` roles).

### Consequences
* **Positive:**
  - Highly efficient and accurate Hybrid Search capabilities out-of-the-box.
  - A single, unified datastore significantly simplifies the mental model and the local deployment process via Docker.
  - The system is performant, secure by design, and handles concurrent requests smoothly via asyncio.
* **Negative:**
  - Requires users to have PostgreSQL 18 and Docker installed, which slightly increases the initial setup friction compared to a pure flat-file or SQLite-based solution.
  - Generating embeddings locally with `sentence-transformers` increases the overall memory footprint and startup time of the Python server.
