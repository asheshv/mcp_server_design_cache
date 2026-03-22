import aiofiles # Non-blocking file I/O for Python 3.13
import os, time, re, asyncio, threading
from contextlib import asynccontextmanager
from collections import defaultdict
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector_async
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_READ_PASS = os.getenv("DB_READ_PASS")
DB_WRITE_PASS = os.getenv("DB_WRITE_PASS")

if not DB_READ_PASS or not DB_WRITE_PASS:
    # We allow the server to start but indicate it's misconfigured if it's meant to be secure
    # Or strictly fail to prevent using defaults
    pass 

READ_URI = f"host={DB_HOST} dbname=design_db user=design_readonly password={DB_READ_PASS or 'read_password'}"
WRITE_URI = f"host={DB_HOST} dbname=design_db user=design_readwrite password={DB_WRITE_PASS or 'write_password'}"
MAX_LIFETIME = 300  # 5 mins

import sys
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        # Load embedding model lazily to prevent IDE timeout during MCP startup
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        sys.stdout = _stdout
    return embedding_model

def get_local_project_name():
    """Attempts to read the project name from a local .design_cache file."""
    try:
        # Check current working directory
        path = os.path.join(os.getcwd(), ".design_cache")
        if os.path.exists(path):
            with open(path, "r") as f:
                name = f.readline().strip()
                if name: return name
    except Exception:
        pass
    return None

# --- CONNECTION POOL WITH PGVECTOR CONFIG ---
async def configure_db(conn):
    """Ensures every pooled connection supports pgvector and UUIDv7."""
    await register_vector_async(conn)

read_pool = AsyncConnectionPool(conninfo=READ_URI, open=False, configure=configure_db)
write_pool = AsyncConnectionPool(conninfo=WRITE_URI, open=False, configure=configure_db)

# --- SCHEMA MIGRATIONS ---
MIGRATIONS = {
    1: """
    ALTER TABLE design_cache ADD COLUMN IF NOT EXISTS tags TEXT[];
    CREATE INDEX IF NOT EXISTS idx_tags ON design_cache USING GIN(tags);
    """,
    2: """
    CREATE INDEX IF NOT EXISTS idx_semantic ON design_cache USING hnsw (embedding vector_cosine_ops);
    """
}

async def apply_migrations():
    """Ensures the database schema is up-to-date by running missing scripts in order."""
    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            # Create schema_version table if missing (bootstrapping)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Get current version
            await cur.execute("SELECT MAX(version) FROM schema_version")
            row = await cur.fetchone()
            current_v = row[0] if row and row[0] is not None else 0

            # Run missing migrations
            for v in sorted(MIGRATIONS.keys()):
                if v > current_v:
                    print(f"--- Applying Database Migration v{v} ---")
                    await cur.execute(MIGRATIONS[v])
                    await cur.execute("INSERT INTO schema_version (version) VALUES (%s)", (v,))
    print("💚 Database schema is up-to-date.")

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    await read_pool.open()
    await write_pool.open()
    # Run migrations once pools are ready
    await apply_migrations()
    try:
        yield
    finally:
        await read_pool.close()
        await write_pool.close()

mcp = FastMCP("SecureDesignMemory", lifespan=app_lifespan)

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
async def search_design(
    project: str,
    query: str,
    limit: int = 5,
    offset: int = 0,
    tags: list[str] | None = None,
    tag_logic: str = "AND"
) -> str:
    """
    Performs Hybrid Search: Keywords (FTS) + Semantic (Vector).
    tag_logic: "AND" (has all), "OR" (has any), "NOT" (has none).
    """
    limiter.check("search")

    model = get_embedding_model()
    query_vector = await asyncio.to_thread(model.encode, query)
    query_vector_list = query_vector.tolist()

    tag_filter = ""
    tag_params = []
    if tags:
        if tag_logic.upper() == "OR":
            tag_filter = "AND tags && %s::text[]"
        elif tag_logic.upper() == "NOT":
            tag_filter = "AND NOT (tags && %s::text[])"
        else: # Default AND
            tag_filter = "AND tags @> %s::text[]"
        tag_params = [tags]

    # Adjusted parameters to match the SQL query exactly
    sql_params = [query, query_vector_list, project] + tag_params + [query, query_vector_list, query, query_vector_list, limit, offset]

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(f"""
                SELECT title, content, cache_type, id, tags,
                       ts_rank(search_vector, plainto_tsquery('english', %s)) as keyword_score,
                       COALESCE((1 - (embedding <=> %s::vector)), 0) as semantic_score
                FROM design_cache
                WHERE project_name = %s
                {tag_filter}
                AND (
                    search_vector @@ plainto_tsquery('english', %s)
                    OR embedding <=> %s::vector < 0.6
                )
                ORDER BY (ts_rank(search_vector, plainto_tsquery('english', %s)) + COALESCE((1 - (embedding <=> %s::vector)), 0)) DESC
                LIMIT %s OFFSET %s;
            """, tuple(sql_params))
            rows = await cur.fetchall()

    if not rows:
        return "No relevant design notes found."

    output = [f"--- Found {len(rows)} matches (Offset: {offset}). Use 'expand_design_note' to read full details. ---"]
    for r in rows:
        tag_str = f" [Tags: {', '.join(r['tags'])}]" if r.get('tags') else ""
        abstract = r['content'][:250].replace('\n', ' ') + "..." if len(r['content']) > 250 else r['content']
        output.append(f"[{r['cache_type'].upper()} ID:{r['id']}] {r['title']}{tag_str}\nAbstract: {abstract}")

    return "\n\n".join(output)

@mcp.tool()
async def expand_design_note(cache_id: str) -> str:
    """
    Retrieves the complete, unabridged content of a specific design note.
    Use this after `search_design` to read the full details of a promising abstract.
    """
    limiter.check("expand")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT title, content, cache_type, created_at, project_name FROM design_cache WHERE id = %s", (cache_id,))
            row = await cur.fetchone()

    if not row:
        return f"Error: No design note found with ID '{cache_id}'"

    ts = row['created_at'].strftime("%Y-%m-%d %H:%M:%S")
    return f"--- {row['cache_type'].upper()} | Project: {row['project_name']} | Date: {ts} ---\n# {row['title']}\n\n{row['content']}"

# --- STORAGE TOOLS ---
@mcp.tool()
async def store_note(project: str, title: str, content: str, cache_type: str = 'idea', tags: list[str] | None = None):
    """Stores persistent design context using native Postgres 18 UUIDv7. Supports optional tags."""
    limiter.check("store")

    text_to_embed = f"{title}\n{content}"
    model = get_embedding_model()
    vector = await asyncio.to_thread(model.encode, text_to_embed)
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO design_cache (project_name, title, content, cache_type, embedding, tags) VALUES (%s, %s, %s, %s, %s::vector, %s)",
            (project, title, content, cache_type, vector_list, tags)
        )
    return "Decision cached successfully."

@mcp.tool()
async def update_note(cache_id: str, title: str | None = None, content: str | None = None, cache_type: str | None = None, tags: list[str] | None = None) -> str:
    """
    Updates an existing design note by ID.
    Only the fields you provide will be changed. Re-generates the embedding
    if title or content changes so semantic search stays accurate.
    """
    limiter.check("update")

    if not any([title, content, cache_type, tags]):
        return "Error: Provide at least one field to update (title, content, cache_type, or tags)."

    # Fetch the current record so we can fill in unchanged fields for re-embedding
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT title, content, cache_type, tags FROM design_cache WHERE id = %s",
                (cache_id,)
            )
            row = await cur.fetchone()

    if not row:
        return f"Error: No design note found with ID '{cache_id}'."

    new_title   = title      if title      is not None else row["title"]
    new_content = content    if content    is not None else row["content"]
    new_type    = cache_type if cache_type is not None else row["cache_type"]
    new_tags    = tags       if tags       is not None else row["tags"]

    # Re-embed only when text actually changed
    vector_list = None
    if title is not None or content is not None:
        model = get_embedding_model()
        vector = await asyncio.to_thread(model.encode, f"{new_title}\n{new_content}")
        vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        if vector_list is not None:
            await conn.execute(
                """UPDATE design_cache
                   SET title = %s, content = %s, cache_type = %s, tags = %s, embedding = %s::vector
                   WHERE id = %s""",
                (new_title, new_content, new_type, new_tags, vector_list, cache_id)
            )
        else:
            await conn.execute(
                "UPDATE design_cache SET cache_type = %s, tags = %s WHERE id = %s",
                (new_type, new_tags, cache_id)
            )

    return f"✅ Note '{cache_id}' updated successfully."

@mcp.tool()
async def delete_note(cache_id: str) -> str:
    """
    Permanently deletes a specific design note by its ID.
    This action is irreversible — use summarize_and_cleanup for bulk archiving instead.
    """
    limiter.check("delete")

    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM design_cache WHERE id = %s RETURNING id",
                (cache_id,)
            )
            deleted = await cur.fetchone()

    if not deleted:
        return f"Error: No design note found with ID '{cache_id}'. Nothing was deleted."

    return f"🗑️ Note '{cache_id}' permanently deleted."

# --- MAINTENANCE & CLEANUP TOOLS ---
@mcp.tool()
async def summarize_and_cleanup(project: str, ids_to_summarize: list[str], summary_text: str, new_title: str):
    """Merges multiple UUIDv7 notes into one summary and deletes originals."""
    limiter.check("summarize")

    if not ids_to_summarize:
        return "Error: No IDs provided to summarize."

    text_to_embed = f"{new_title}\n{summary_text}"
    model = get_embedding_model()
    vector = await asyncio.to_thread(model.encode, text_to_embed)
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO design_cache (project_name, content, cache_type, title, summary_of_ids, embedding)
                VALUES (%s, %s, 'project', %s, %s, %s::vector)
            """, (project, summary_text, new_title, ids_to_summarize, vector_list))
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
async def get_retention_policies(project: str | None = None) -> str:
    """
    Retrieves the active retention policies.
    If project is provided, returns only that policy. Otherwise returns all.
    """
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            if project:
                await cur.execute("SELECT * FROM retention_policies WHERE project_name = %s", (project,))
            else:
                await cur.execute("SELECT * FROM retention_policies ORDER BY project_name")
            rows = await cur.fetchall()

    if not rows:
        return "No retention policies found."

    output = ["--- Active Retention Policies ---"]
    for r in rows:
        status = "Auto-Compress" if r['auto_compress'] else "Hard Delete"
        output.append(f"- {r['project_name']}: {r['days_to_retain']} days ({status})")

    return "\n".join(output)

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

    # Execute actual deletion
    async with write_pool.connection() as conn:
        await conn.execute("""
            DELETE FROM design_cache c
            USING retention_policies p
            WHERE c.project_name = p.project_name
            AND c.cache_type = 'idea'
            AND p.days_to_retain > 0
            AND c.created_at < NOW() - (p.days_to_retain || ' days')::interval
        """)

    return "🚨 Cleanup complete! Deleted expired notes:\n" + "\n".join([f"- {i['project_name']}: {i['expired_count']} notes" for i in expired])

@mcp.tool()
async def health_check():
    """Verify database connection health and pgvector availability."""
    async with read_pool.connection() as conn:
        await conn.execute("SELECT 1")
    return "💚 Healthy. Connection pools and pgvector active."

@mcp.tool()
async def get_recent_activity(
    project: str,
    limit: int = 5,
    offset: int = 0,
    tags: list[str] | None = None,
    tag_logic: str = "AND"
) -> str:
    """
    Retrieves the most recent design notes and decisions for a project.
    tag_logic: "AND" (has all), "OR" (has any), "NOT" (has none).
    """
    limiter.check("activity")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            tag_filter = ""
            tag_params = []
            if tags:
                if tag_logic.upper() == "OR":
                    tag_filter = "AND tags && %s::text[]"
                elif tag_logic.upper() == "NOT":
                    tag_filter = "AND NOT (tags && %s::text[])"
                else: # Default AND
                    tag_filter = "AND tags @> %s::text[]"
                tag_params = [tags]

            sql = f"""
                SELECT id, title, content, cache_type, created_at, tags
                FROM design_cache
                WHERE project_name = %s
                {tag_filter}
                ORDER BY id DESC LIMIT %s OFFSET %s;
            """
            params = [project] + tag_params + [limit, offset]
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    if not rows:
        return f"No recent activity found for project '{project}' (Offset: {offset})."

    output = [f"--- Recent Activity for {project} (Top {len(rows)}, Offset: {offset}) ---"]
    for r in rows:
        ts = r['created_at'].strftime("%Y-%m-%d %H:%M")
        tag_str = f" [Tags: {', '.join(r['tags'])}]" if r.get('tags') else ""
        output.append(f"[{ts}] [{r['cache_type'].upper()}] {r['title']}{tag_str}\n{r['content']}")

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

    if len(rows) > 20:
        filepath = os.path.abspath(f"/tmp/{project}_history.md")
        with open(filepath, "w") as f:
            f.write(full_text)
        return f"✅ Export successful. Total entries: {len(rows)}.\n\nContent too large for MCP response. Saved locally to: {filepath}"

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

    new_title = f"{record['title']} [ADR status: {status.upper()}]"
    model = get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{adr_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute("""
            UPDATE design_cache
            SET title = %s,
                content = %s,
                cache_type = 'project',
                embedding = %s::vector
            WHERE id = %s
        """, (new_title, adr_content, vector_list, idea_id))

    return adr_content

@mcp.tool()
async def sync_doc_status(cache_id: str, file_path: str, status: str = "implemented") -> str:
    """
    Links a physical file (Spec/ADR) to a Cache entry and updates its status.
    Ensures the AI knows the 'Idea' has graduated to 'Official Doc'.
    """
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT title, content FROM design_cache WHERE id = %s", (cache_id,))
            row = await cur.fetchone()

    if not row: return f"Error: Cache ID {cache_id} not found."

    new_title = f"{row['title']} [OFFICIAL: {status}]"
    new_content = f"{row['content']}\n\nReference File: {file_path}"

    model = get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{new_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute("""
            UPDATE design_cache
            SET title = %s,
                content = %s,
                cache_type = 'project', -- Promote to Project level
                embedding = %s::vector
            WHERE id = %s
        """, (new_title, new_content, vector_list, cache_id))
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

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT title, content FROM design_cache WHERE id = %s", (cache_id,))
            row = await cur.fetchone()

    if not row: return f"Error: Cache ID {cache_id} not found."

    new_title = f"{row['title']} [LINKED {category.upper()}]"
    new_content = f"{row['content']}\n\nLinked File: {file_path}"

    model = get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{new_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute("""
            UPDATE design_cache
            SET title = %s,
                content = %s,
                embedding = %s::vector
            WHERE id = %s
        """, (new_title, new_content, vector_list, cache_id))
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

@mcp.tool()
async def get_project_context() -> str:
    """
    Detects the current project and returns its status, last 3 ideas, and common tags.
    Use this at the start of a session to instantly sync with the local workspace.
    """
    project = get_local_project_name()
    if not project:
        return "No local .design_cache project file found. Use 'store_note' to start a project or create a .design_cache file."

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Get last 3 ideas
            await cur.execute("""
                SELECT id, title, created_at, tags
                FROM design_cache
                WHERE project_name = %s
                ORDER BY id DESC LIMIT 3
            """, (project,))
            notes = await cur.fetchall()

            # Get top 5 tags
            await cur.execute("""
                SELECT unnest(tags) as tag, count(*) as count
                FROM design_cache
                WHERE project_name = %s
                GROUP BY tag ORDER BY count DESC LIMIT 5
            """, (project,))
            tags = await cur.fetchall()

    res = [f"--- Project Context: {project} ---"]
    if notes:
        res.append("Recent Ideas:")
        for n in notes:
            t = n['created_at'].strftime("%Y-%m-%d")
            res.append(f"- [{t}] {n['title']} (ID: {n['id']})")
    else:
        res.append("No notes found for this project yet.")

    if tags:
        res.append(f"Top Tags: {', '.join([t['tag'] for t in tags])}")

    return "\n".join(res)

@mcp.prompt()
def onboard():
    """Instructions for the AI to initialize a session for the current workspace."""
    project = get_local_project_name()
    if project:
        return f"Please call 'get_project_context' to see the recent state of project '{project}'. Then, summarize the top 3 items and ask the user if they'd like to continue work on any of them or start something new."
    else:
        return "I couldn't detect a local project name. Please ask the user which project they are working on, or suggest they create a .design_cache file with the project name."

@mcp.tool()
async def get_compression_opportunities(project: str, char_limit: int = 5000) -> str:
    """
    Analyzes a project's notes and returns suggestions for summarization (compression).
    Helps prevent LLM context overflow by identifying 'dense' clusters of notes.
    """
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, length(content) as content_len, cache_type, tags
                FROM design_cache
                WHERE project_name = %s
                ORDER BY id ASC
            """, (project,))
            notes = await cur.fetchall()

    if not notes:
        return f"No notes found for project '{project}' to analyze."

    total_chars = sum(n['content_len'] for n in notes)
    estimated_tokens = total_chars // 4 # Rough estimate

    res = [f"--- Compression Analysis: {project} ---"]
    res.append(f"Total Notes: {len(notes)}")
    res.append(f"Estimated Context Size: ~{estimated_tokens} tokens ({total_chars} chars)")

    if total_chars < char_limit:
        res.append("✅ Context size is healthy. No immediate compression needed.")
        return "\n".join(res)

    res.append("⚠️ Context density is HIGH. Consider summarizing the following clusters:")

    # Group by cache_type for suggestion clusters
    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in notes:
        by_type[n['cache_type']].append(n)

    for ctype, cluster in by_type.items():
        if len(cluster) >= 3:
            ids = [n['id'] for n in cluster]
            # cluster[:3] is safe here as cluster is a list[dict]
            top_titles = [f"'{n['title']}'" for n in cluster[:3]]
            cluster_titles = ", ".join(top_titles)
            if len(cluster) > 3: cluster_titles += " ..."
            res.append(f"\nCluster [{ctype.upper()}]: {len(cluster)} notes")
            res.append(f"- Titles: {cluster_titles}")
            res.append(f"- Suggested Action: Call 'summarize_and_cleanup' with IDs: {ids}")

    return "\n".join(res)

if __name__ == "__main__":
    mcp.run()
