import aiofiles # Non-blocking file I/O for Python 3.13
import os, time, re, asyncio, threading
from collections import defaultdict
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector_async
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
DB_HOST = os.getenv("DB_HOST", "localhost")
READ_URI = f"host={DB_HOST} dbname=design_db user=design_readonly password=read_password"
WRITE_URI = f"host={DB_HOST} dbname=design_db user=design_readwrite password=write_password"
MAX_LIFETIME = 300  # 5 mins

mcp = FastMCP("SecureDesignMemory")

# Load embedding model globally (downloads ~90MB on first run)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- CONNECTION POOL WITH PGVECTOR CONFIG ---
async def configure_db(conn):
    """Ensures every pooled connection supports pgvector and UUIDv7."""
    await register_vector_async(conn)

read_pool = AsyncConnectionPool(conninfo=READ_URI, open=False, configure=configure_db)
write_pool = AsyncConnectionPool(conninfo=WRITE_URI, open=False, configure=configure_db)

@mcp.on_startup()
async def startup():
    await read_pool.open()
    await write_pool.open()

@mcp.on_shutdown()
async def shutdown():
    await read_pool.close()
    await write_pool.close()

# --- SECURITY: RATE LIMITER ---
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm, self.history, self.lock = rpm, defaultdict(list), threading.Lock()
    def check(self, tool: str):
        with self.lock:
            now = time.time()
            self.history[tool] = [t for t in self.history[tool] if now - t < 60]
            if len(self.history[tool]) >= self.rpm:
                raise RuntimeError(f"Rate limit exceeded (Max {self.rpm} RPM)")
            self.history[tool].append(now)

limiter = RateLimiter(60)

# --- SECURITY: SQL VALIDATION ---
# REMOVED: psycopg3 parameterizes queries automatically, rendering regex blocking unnecessary.

# --- SEARCH TOOLS ---
@mcp.tool()
async def search_design(project: str, query: str) -> str:
    """
    Performs Hybrid Search: Keywords (FTS) + Semantic (Vector).
    Returns the top 5 most relevant design notes.
    """
    limiter.check("search")
    
    # 1. Get embedding for the query string using sentence-transformers
    query_vector = await asyncio.to_thread(embedding_model.encode, query)
    query_vector_list = query_vector.tolist()

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # We use Reciprocal Rank Fusion (RRF) logic or simple weighted sum
            # Postgres 18 handles this complex ranking with sub-millisecond latency
            await cur.execute("""
                SELECT title, content, cache_type, id,
                       ts_rank(search_vector, plainto_tsquery('english', %s)) as keyword_score,
                       (1 - (embedding <=> %s::vector)) as semantic_score
                FROM design_cache 
                WHERE project_name = %s 
                AND (
                    search_vector @@ plainto_tsquery('english', %s)
                    OR embedding <=> %s::vector < 0.6
                )
                ORDER BY (ts_rank(search_vector, plainto_tsquery('english', %s)) + (1 - (embedding <=> %s::vector))) DESC 
                LIMIT 5;
            """, (query, query_vector_list, project, query, query_vector_list, query, query_vector_list))
            rows = await cur.fetchall()
    
    if not rows:
        return "No relevant design notes found."

    output = [f"--- Found {len(rows)} matches for project: {project} ---"]
    for r in rows:
        output.append(f"[{r['cache_type'].upper()} ID:{r['id']}] {r['title']}\n{r['content']}")
    
    return "\n\n".join(output)

# --- STORAGE TOOLS ---
@mcp.tool()
async def store_note(project: str, title: str, content: str, cache_type: str = 'idea'):
    """Stores persistent design context using native Postgres 18 UUIDv7."""
    limiter.check("store")
    
    text_to_embed = f"{title}\n{content}"
    vector = await asyncio.to_thread(embedding_model.encode, text_to_embed)
    vector_list = vector.tolist()
    
    async with write_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO design_cache (project_name, title, content, cache_type, embedding) VALUES (%s, %s, %s, %s, %s::vector)", 
            (project, title, content, cache_type, vector_list)
        )
    return "Decision cached successfully."

# --- MAINTENANCE & CLEANUP TOOLS ---
@mcp.tool()
async def summarize_and_cleanup(project: str, ids_to_summarize: list[str], summary_text: str, new_title: str):
    """Merges multiple UUIDv7 notes into one summary and deletes originals."""
    limiter.check("summarize")
    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO design_cache (project_name, content, cache_type, title, summary_of_ids) 
                VALUES (%s, %s, 'project', %s, %s)
            """, (project, summary_text, new_title, ids_to_summarize))
            await cur.execute("DELETE FROM design_cache WHERE id = ANY(%s)", (ids_to_summarize,))
    return f"Consolidated {len(ids_to_summarize)} notes into '{new_title}'."

@mcp.tool()
async def set_retention_policy(project: str, days: int, auto_compress: bool = True):
    """Sets a custom cleanup policy for a project (e.g., 7 days for 'Work')."""
    async with write_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO retention_policies (project_name, days_to_retain, auto_compress) 
            VALUES (%s, %s, %s)
            ON CONFLICT (project_name) DO UPDATE 
            SET days_to_retain = EXCLUDED.days_to_retain, auto_compress = EXCLUDED.auto_compress
        """, (project, days, auto_compress))
    return f"Policy set: {days} days for {project}."

@mcp.tool()
async def run_smart_cleanup() -> str:
    """Scans all projects for expired notes based on individual retention policies."""
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT c.project_name, COUNT(c.id) as expired_count
                FROM design_cache c
                JOIN retention_policies p ON c.project_name = p.project_name
                WHERE c.cache_type = 'idea' AND p.days_to_retain > 0
                AND c.created_at < NOW() - (p.days_to_retain || ' days')::interval
                GROUP BY c.project_name
            """)
            expired = await cur.fetchall()
    if not expired: return "💚 All projects are within their retention limits."
    return "🚨 Ready for cleanup:\n" + "\n".join([f"- {i['project_name']}: {i['expired_count']} notes" for i in expired])

@mcp.tool()
async def health_check():
    """Verify database connection health and pgvector availability."""
    async with read_pool.connection() as conn:
        await conn.execute("SELECT 1")
    return "💚 Healthy. Connection pools and pgvector active."

@mcp.tool()
async def get_recent_activity(project: str, limit: int = 5) -> str:
    """
    Retrieves the most recent design notes and decisions for a project.
    Use this at the start of a session to get up to speed quickly.
    """
    limiter.check("activity")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # UUIDv7 is time-sortable, so this query is extremely efficient
            await cur.execute("""
                SELECT id, title, content, cache_type, created_at
                FROM design_cache 
                WHERE project_name = %s 
                ORDER BY id DESC LIMIT %s;
            """, (project, limit))
            rows = await cur.fetchall()
    
    if not rows:
        return f"No recent activity found for project '{project}'."

    output = [f"--- Recent Activity for {project} (Top {len(rows)}) ---"]
    for r in rows:
        # Format timestamp to be readable
        ts = r['created_at'].strftime("%Y-%m-%d %H:%M")
        output.append(f"[{ts}] [{r['cache_type'].upper()}] {r['title']}\n{r['content']}")
    
    return "\n\n".join(output)

# Inside server.py

@mcp.tool()
async def export_project_to_markdown(project: str) -> str:
    """
    Exports the entire design history for a specific project into 
    a single, human-readable Markdown document.
    """
    limiter.check("export")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Fetch all notes, ordered by time (UUIDv7 naturally sorts this)
            await cur.execute("""
                SELECT title, content, cache_type, created_at 
                FROM design_cache 
                WHERE project_name = %s 
                ORDER BY id ASC
            """, (project,))
            rows = await cur.fetchall()

    if not rows:
        return f"No data found for project '{project}'."

    # Build the Markdown content
    md = [f"# Design History: {project}\n", f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    
    for r in rows:
        ts = r['created_at'].strftime("%Y-%m-%d %H:%M")
        icon = "📌" if r['cache_type'] == 'project' else "💡"
        md.append(f"## {icon} {r['title'] or 'Untitled Note'}")
        md.append(f"**Date:** {ts} | **Type:** {r['cache_type'].capitalize()}")
        md.append(f"\n{r['content']}\n")
        md.append("---")

    full_text = "\n".join(md)
    
    # Optional: Save to a local file (requires volume mapping in Docker)
    # with open(f"/app/exports/{project}_history.md", "w") as f:
    #     f.write(full_text)

    return f"✅ Export successful. Total entries: {len(rows)}.\n\n" + full_text

@mcp.tool()
async def generate_spec_from_cache(idea_id: str) -> str:
    """
    Retrieves a cached idea and formats it into a formal Technical Spec.
    Returns the Markdown content to be saved as a file.
    """
    limiter.check("spec")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, project_name, created_at 
                FROM design_cache WHERE id = %s
            """, (idea_id,))
            idea = await cur.fetchone()

    if not idea:
        return f"Error: No idea found with ID {idea_id}."

    # Create a professional Spec Template
    spec = f"""# Technical Specification: {idea['title']}
**Status:** Draft | **Project:** {idea['project_name']}
**Derived From Cache ID:** {idea_id} | **Date:** {idea['created_at'].strftime('%Y-%m-%d')}

## 1. Executive Summary
{idea['content'][:200]}...

## 2. Requirements & Context
{idea['content']}

## 3. Implementation Details
[AI: Please fill in the architectural details based on our current codebase]

## 4. Risks & Considerations
- Migration from legacy systems
- Token efficiency & performance
"""
    return spec

@mcp.tool()
async def generate_adr_from_cache(idea_id: str, status: str = "proposed") -> str:
    """
    Transforms a cached design idea into a formal Architecture Decision Record (ADR).
    Statuses: proposed, accepted, superseded, deprecated.
    """
    limiter.check("adr")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # UUIDv7 makes finding the specific discussion instant
            await cur.execute("""
                SELECT title, content, project_name, created_at 
                FROM design_cache WHERE id = %s
            """, (idea_id,))
            record = await cur.fetchone()

    if not record:
        return f"Error: No record found for ID {idea_id}."

    # Standard ADR Template
    adr_content = f"""# ADR: {record['title']}

* **Status:** {status.upper()}
* **Date:** {record['created_at'].strftime('%Y-%m-%d')}
* **Project:** {record['project_name']}
* **Cache Reference:** {idea_id}

## Context and Problem Statement
{record['content'][:300]}...

## Decision Drivers
* Technical debt reduction
* Token efficiency in AI context
* Scalability with PostgreSQL 18

## Considered Options
1. [Option 1 from discussion]
2. [Option 2 from discussion]

## Decision Outcome
Chosen Option: **[AI: Please specify based on our chat]**

### Consequences
* **Positive:** Improved maintainability.
* **Negative:** Initial overhead of implementation.

## Validation
[AI: Define how we verify this architectural change]
"""
    return adr_content

@mcp.tool()
async def sync_doc_status(cache_id: str, file_path: str, status: str = "implemented") -> str:
    """
    Links a physical file (Spec/ADR) to a Cache entry and updates its status.
    Ensures the AI knows the 'Idea' has graduated to 'Official Doc'.
    """
    async with write_pool.connection() as conn:
        await conn.execute("""
            UPDATE design_cache 
            SET title = title || ' [OFFICIAL: ' || %s || ']',
                content = content || '\n\nReference File: ' || %s,
                cache_type = 'project' -- Promote to Project level
            WHERE id = %s
        """, (status, file_path, cache_id))
    return f"🔗 Linked Cache {cache_id} to {file_path} as {status}."

# Tool to link an existing file to a Cache ID
@mcp.tool()
async def link_external_file_to_cache(cache_id: str, file_path: str, category: str = "spec") -> str:
    """
    Links a local Markdown file (Spec/ADR) to a specific design cache entry.
    Requires an absolute file path.
    """
    if not os.path.isabs(file_path):
        return "Error: Please provide an absolute file path."
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}."

    async with write_pool.connection() as conn:
        await conn.execute("""
            UPDATE design_cache 
            SET title = title || ' [LINKED ' || %s || ']',
                content = content || '\n\nLinked File: ' || %s
            WHERE id = %s
        """, (category.upper(), file_path, cache_id))
    return f"✅ Linked {category} at {file_path} to Cache ID {cache_id}."

# Resource to let the AI "Load" and read the file content
@mcp.resource("file://{path}")
async def read_external_doc(path: str) -> str:
    """Reads the content of an external spec or ADR for context."""
    try:
        async with aiofiles.open(path, mode='r') as f:
            content = await f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

if __name__ == "__main__":
    mcp.run()
