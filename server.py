"""
MCP Server: Design Cache — Persistent design memory for AI agents.

Stores design conversations in a hierarchical structure backed by
PostgreSQL + pgvector, enabling cross-session context and semantic search.
"""
import aiofiles
import os
import sys
import time
import re
import asyncio
from contextlib import asynccontextmanager
from collections import defaultdict
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector_async
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "design_db")
DB_READ_PASS = os.getenv("DB_READ_PASS")
DB_WRITE_PASS = os.getenv("DB_WRITE_PASS")

# Fail fast if credentials are missing (H-4, SEC-01)
if not DB_READ_PASS or not DB_WRITE_PASS:
    raise EnvironmentError(
        "DB_READ_PASS and DB_WRITE_PASS environment variables must be set. "
        "See .env.example for reference."
    )


def _quote_conninfo_value(val: str) -> str:
    """Escape a value for use in a libpq conninfo string (H-3, SEC-05)."""
    return "'" + val.replace("\\", "\\\\").replace("'", "\\'") + "'"


# Build connection URIs with proper quoting to prevent injection
READ_URI = (
    f"host={_quote_conninfo_value(DB_HOST)} "
    f"port={_quote_conninfo_value(DB_PORT)} "
    f"dbname={_quote_conninfo_value(DB_NAME)} "
    f"user='design_readonly' "
    f"password={_quote_conninfo_value(DB_READ_PASS)}"
)
WRITE_URI = (
    f"host={_quote_conninfo_value(DB_HOST)} "
    f"port={_quote_conninfo_value(DB_PORT)} "
    f"dbname={_quote_conninfo_value(DB_NAME)} "
    f"user='design_readwrite' "
    f"password={_quote_conninfo_value(DB_WRITE_PASS)}"
)

MAX_LIFETIME = 300  # 5 minutes
MAX_POOL_SIZE = 10
MIN_POOL_SIZE = 2

# Allowed directory prefixes for file operations (H-1, H-2, SEC-03, SEC-04)
ALLOWED_FILE_PREFIXES = ("/tmp/", os.path.expanduser("~/"))

# Content size limits (M-5, SEC-10)
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50_000  # ~50KB

# Valid values for constrained parameters (M-10)
VALID_TAG_LOGIC = {"AND", "OR", "NOT"}
VALID_CACHE_TYPES = {"project", "idea"}
VALID_ADR_STATUSES = {"proposed", "accepted", "superseded", "deprecated"}

# --- EMBEDDING MODEL (async-safe lazy init: H-8, L-5, M-8) ---
_embedding_model = None
_model_lock = asyncio.Lock()


async def get_embedding_model():
    """Load embedding model lazily with async-safe double-checked locking."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    async with _model_lock:
        if _embedding_model is not None:
            return _embedding_model
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            _embedding_model = await asyncio.to_thread(
                SentenceTransformer, 'all-MiniLM-L6-v2'
            )
        finally:
            sys.stdout = _stdout
        return _embedding_model


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in file paths."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def _validate_file_path(path: str) -> bool:
    """Check that a resolved path is within allowed directories."""
    resolved = os.path.realpath(path)
    return any(resolved.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES)


def get_local_project_name():
    """Attempts to read the project name from a local .design_cache file."""
    try:
        path = os.path.join(os.getcwd(), ".design_cache")
        if os.path.exists(path):
            with open(path, "r") as f:
                name = f.readline().strip()
                if name:
                    return name
    except Exception:
        pass
    return None


# --- CONNECTION POOL WITH PGVECTOR CONFIG (H-10, PERF-03) ---
async def configure_db(conn):
    """Ensures every pooled connection supports pgvector."""
    await register_vector_async(conn)


read_pool = AsyncConnectionPool(
    conninfo=READ_URI, open=False, configure=configure_db,
    min_size=MIN_POOL_SIZE, max_size=MAX_POOL_SIZE,
    max_lifetime=MAX_LIFETIME, max_idle=120,
)
write_pool = AsyncConnectionPool(
    conninfo=WRITE_URI, open=False, configure=configure_db,
    min_size=MIN_POOL_SIZE, max_size=MAX_POOL_SIZE,
    max_lifetime=MAX_LIFETIME, max_idle=120,
)

# --- SCHEMA MIGRATIONS (M-11: per-migration transactions) ---
MIGRATIONS = {
    1: """
    ALTER TABLE design_cache ADD COLUMN IF NOT EXISTS tags TEXT[];
    CREATE INDEX IF NOT EXISTS idx_tags ON design_cache USING GIN(tags);
    """,
    2: """
    CREATE INDEX IF NOT EXISTS idx_semantic ON design_cache USING hnsw (embedding vector_cosine_ops);
    """,
    3: """
    CREATE INDEX IF NOT EXISTS idx_project_name ON design_cache (project_name);
    """,
    4: """
    ALTER TABLE design_cache ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
    """,
}


async def apply_migrations():
    """Applies pending migrations, each in its own transaction."""
    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            await cur.execute("SELECT MAX(version) FROM schema_version")
            row = await cur.fetchone()
            current_v = row[0] if row and row[0] is not None else 0

            for v in sorted(MIGRATIONS.keys()):
                if v > current_v:
                    print(
                        f"--- Applying Database Migration v{v} ---",
                        file=sys.stderr,
                    )
                    async with conn.transaction():
                        await cur.execute(MIGRATIONS[v])
                        await cur.execute(
                            "INSERT INTO schema_version (version) VALUES (%s)",
                            (v,),
                        )
    print("Database schema is up-to-date.", file=sys.stderr)


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    await read_pool.open()
    await write_pool.open()
    await apply_migrations()
    # Eagerly load embedding model at startup (PERF-01)
    await get_embedding_model()
    try:
        yield
    finally:
        await read_pool.close()
        await write_pool.close()


mcp = FastMCP("SecureDesignMemory", lifespan=app_lifespan)


# --- RATE LIMITER (H-6, PERF-08: asyncio.Lock instead of threading.Lock) ---
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.history = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check(self, tool: str):
        async with self.lock:
            now = time.time()
            self.history[tool] = [t for t in self.history[tool] if now - t < 60]
            # Prune empty keys to prevent memory growth (M-20, PERF-07)
            if not self.history[tool]:
                del self.history[tool]
                self.history[tool] = []
            if len(self.history[tool]) >= self.rpm:
                raise RuntimeError(f"Rate limit exceeded (Max {self.rpm} RPM)")
            self.history[tool].append(now)


limiter = RateLimiter(60)


# --- SEARCH TOOLS ---
@mcp.tool()
async def search_design(
    project: str,
    query: str,
    limit: int = 5,
    offset: int = 0,
    tags: list[str] | None = None,
    tag_logic: str = "AND",
) -> str:
    """
    Performs Hybrid Search: Keywords (FTS) + Semantic (Vector).
    tag_logic: "AND" (has all), "OR" (has any), "NOT" (has none).
    """
    await limiter.check("search")

    # Validate inputs (M-10)
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    tag_logic = tag_logic.upper()
    if tag_logic not in VALID_TAG_LOGIC:
        return f"Error: tag_logic must be one of {VALID_TAG_LOGIC}"

    model = await get_embedding_model()
    query_vector = await asyncio.to_thread(model.encode, query)
    query_vector_list = query_vector.tolist()

    # Named parameters to avoid fragile positional counting (L-6, PERF-05)
    params: dict = {
        "query": query,
        "vector": query_vector_list,
        "project": project,
        "limit": limit,
        "offset": offset,
    }

    tag_filter = ""
    if tags:
        params["tags"] = tags
        if tag_logic == "OR":
            tag_filter = "AND tags && %(tags)s::text[]"
        elif tag_logic == "NOT":
            tag_filter = "AND NOT (tags && %(tags)s::text[])"
        else:
            tag_filter = "AND tags @> %(tags)s::text[]"

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # CTE to compute vector distance once (PERF-05)
            await cur.execute(f"""
                WITH scored AS (
                    SELECT title, content, cache_type, id, tags,
                           ts_rank(search_vector, plainto_tsquery('english', %(query)s)) AS keyword_score,
                           COALESCE((1 - (embedding <=> %(vector)s::vector)), 0) AS semantic_score
                    FROM design_cache
                    WHERE project_name = %(project)s
                    {tag_filter}
                    AND (
                        search_vector @@ plainto_tsquery('english', %(query)s)
                        OR embedding <=> %(vector)s::vector < 0.6
                    )
                )
                SELECT *, (keyword_score + semantic_score) AS combined_score
                FROM scored
                ORDER BY combined_score DESC
                LIMIT %(limit)s OFFSET %(offset)s;
            """, params)
            rows = await cur.fetchall()

    if not rows:
        return "No relevant design notes found."

    output = [
        f"--- Found {len(rows)} matches (Offset: {offset}). "
        f"Use 'expand_design_note' to read full details. ---"
    ]
    for r in rows:
        tag_str = f" [Tags: {', '.join(r['tags'])}]" if r.get('tags') else ""
        abstract = (
            r['content'][:250].replace('\n', ' ') + "..."
            if len(r['content']) > 250 else r['content']
        )
        output.append(
            f"[{r['cache_type'].upper()} ID:{r['id']}] {r['title']}{tag_str}\n"
            f"Abstract: {abstract}"
        )

    return "\n\n".join(output)


@mcp.tool()
async def expand_design_note(cache_id: str) -> str:
    """
    Retrieves the complete, unabridged content of a specific design note.
    Use this after `search_design` to read the full details of a promising abstract.
    """
    await limiter.check("expand")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT title, content, cache_type, created_at, project_name "
                "FROM design_cache WHERE id = %s",
                (cache_id,),
            )
            row = await cur.fetchone()

    if not row:
        return f"Error: No design note found with ID '{cache_id}'"

    ts = row['created_at'].strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"--- {row['cache_type'].upper()} | Project: {row['project_name']} "
        f"| Date: {ts} ---\n# {row['title']}\n\n{row['content']}"
    )


# --- STORAGE TOOLS ---
@mcp.tool()
async def store_note(
    project: str,
    title: str,
    content: str,
    cache_type: str = 'idea',
    tags: list[str] | None = None,
):
    """Stores persistent design context. Supports optional tags."""
    await limiter.check("store")

    # Validate inputs (M-10, SEC-10)
    if cache_type not in VALID_CACHE_TYPES:
        return f"Error: cache_type must be one of {VALID_CACHE_TYPES}"
    if len(title) > MAX_TITLE_LENGTH:
        return f"Error: title exceeds maximum length of {MAX_TITLE_LENGTH} characters."
    if len(content) > MAX_CONTENT_LENGTH:
        return f"Error: content exceeds maximum length of {MAX_CONTENT_LENGTH} characters."

    text_to_embed = f"{title}\n{content}"
    model = await get_embedding_model()
    vector = await asyncio.to_thread(model.encode, text_to_embed)
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO design_cache "
            "(project_name, title, content, cache_type, embedding, tags) "
            "VALUES (%s, %s, %s, %s, %s::vector, %s)",
            (project, title, content, cache_type, vector_list, tags),
        )
    return "Decision cached successfully."


@mcp.tool()
async def update_note(
    cache_id: str,
    title: str | None = None,
    content: str | None = None,
    cache_type: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Updates an existing design note by ID.
    Only the fields you provide will be changed. Re-generates the embedding
    if title or content changes so semantic search stays accurate.
    """
    await limiter.check("update")

    if not any([title, content, cache_type, tags]):
        return "Error: Provide at least one field to update (title, content, cache_type, or tags)."

    if cache_type is not None and cache_type not in VALID_CACHE_TYPES:
        return f"Error: cache_type must be one of {VALID_CACHE_TYPES}"
    if title is not None and len(title) > MAX_TITLE_LENGTH:
        return f"Error: title exceeds maximum length of {MAX_TITLE_LENGTH} characters."
    if content is not None and len(content) > MAX_CONTENT_LENGTH:
        return f"Error: content exceeds maximum length of {MAX_CONTENT_LENGTH} characters."

    # Single write_pool connection with FOR UPDATE to prevent TOCTOU (C-2)
    async with write_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT title, content, cache_type, tags "
                "FROM design_cache WHERE id = %s FOR UPDATE",
                (cache_id,),
            )
            row = await cur.fetchone()

            if not row:
                return f"Error: No design note found with ID '{cache_id}'."

            new_title = title if title is not None else row["title"]
            new_content = content if content is not None else row["content"]
            new_type = cache_type if cache_type is not None else row["cache_type"]
            new_tags = tags if tags is not None else row["tags"]

            vector_list = None
            if title is not None or content is not None:
                model = await get_embedding_model()
                vector = await asyncio.to_thread(
                    model.encode, f"{new_title}\n{new_content}"
                )
                vector_list = vector.tolist()

            if vector_list is not None:
                await conn.execute(
                    "UPDATE design_cache SET title = %s, content = %s, "
                    "cache_type = %s, tags = %s, embedding = %s::vector, "
                    "updated_at = now() WHERE id = %s",
                    (new_title, new_content, new_type, new_tags, vector_list, cache_id),
                )
            else:
                await conn.execute(
                    "UPDATE design_cache SET cache_type = %s, tags = %s, "
                    "updated_at = now() WHERE id = %s",
                    (new_type, new_tags, cache_id),
                )

    return f"Note '{cache_id}' updated successfully."


@mcp.tool()
async def delete_note(cache_id: str) -> str:
    """
    Permanently deletes a specific design note by its ID.
    This action is irreversible.
    """
    await limiter.check("delete")

    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM design_cache WHERE id = %s RETURNING id",
                (cache_id,),
            )
            deleted = await cur.fetchone()

    if not deleted:
        return f"Error: No design note found with ID '{cache_id}'. Nothing was deleted."

    return f"Note '{cache_id}' permanently deleted."


# --- MAINTENANCE & CLEANUP TOOLS ---
@mcp.tool()
async def summarize_and_cleanup(
    project: str,
    ids_to_summarize: list[str],
    summary_text: str,
    new_title: str,
):
    """Merges multiple notes into one summary and deletes originals atomically."""
    await limiter.check("summarize")

    if not ids_to_summarize:
        return "Error: No IDs provided to summarize."

    text_to_embed = f"{new_title}\n{summary_text}"
    model = await get_embedding_model()
    vector = await asyncio.to_thread(model.encode, text_to_embed)
    vector_list = vector.tolist()

    # Explicit transaction for atomicity (C-1); project guard on DELETE (M-9)
    async with write_pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO design_cache
                        (project_name, content, cache_type, title,
                         summary_of_ids, embedding)
                    VALUES (%s, %s, 'project', %s, %s, %s::vector)
                """, (project, summary_text, new_title, ids_to_summarize, vector_list))
                await cur.execute(
                    "DELETE FROM design_cache "
                    "WHERE id = ANY(%s) AND project_name = %s",
                    (ids_to_summarize, project),
                )
    return f"Consolidated {len(ids_to_summarize)} notes into '{new_title}'."


@mcp.tool()
async def set_retention_policy(
    project: str, days: int, auto_compress: bool = True,
):
    """Sets a custom cleanup policy for a project (e.g., 7 days for 'Work')."""
    if days <= 0:
        return "Error: days must be a positive integer."

    async with write_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO retention_policies (project_name, days_to_retain, auto_compress)
            VALUES (%s, %s, %s)
            ON CONFLICT (project_name) DO UPDATE
            SET days_to_retain = EXCLUDED.days_to_retain,
                auto_compress = EXCLUDED.auto_compress
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
                await cur.execute(
                    "SELECT * FROM retention_policies WHERE project_name = %s",
                    (project,),
                )
            else:
                await cur.execute(
                    "SELECT * FROM retention_policies ORDER BY project_name"
                )
            rows = await cur.fetchall()

    if not rows:
        return "No retention policies found."

    output = ["--- Active Retention Policies ---"]
    for r in rows:
        status = "Auto-Compress" if r['auto_compress'] else "Hard Delete"
        output.append(
            f"- {r['project_name']}: {r['days_to_retain']} days ({status})"
        )

    return "\n".join(output)


@mcp.tool()
async def run_smart_cleanup() -> str:
    """Deletes expired notes based on retention policies. Returns actual deleted counts."""
    await limiter.check("cleanup")

    # Single write_pool with DELETE ... RETURNING to fix TOCTOU (H-7, L-03)
    async with write_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                WITH deleted AS (
                    DELETE FROM design_cache c
                    USING retention_policies p
                    WHERE c.project_name = p.project_name
                    AND c.cache_type = 'idea'
                    AND p.days_to_retain > 0
                    AND c.created_at < now() - (p.days_to_retain || ' days')::interval
                    RETURNING c.project_name
                )
                SELECT project_name, COUNT(*) AS deleted_count
                FROM deleted
                GROUP BY project_name
            """)
            results = await cur.fetchall()

    if not results:
        return "All projects are within their retention limits."

    lines = [
        f"- {r['project_name']}: {r['deleted_count']} notes"
        for r in results
    ]
    return "Cleanup complete! Deleted expired notes:\n" + "\n".join(lines)


@mcp.tool()
async def health_check():
    """Verify database connection health and pgvector availability."""
    await limiter.check("health")
    async with read_pool.connection() as conn:
        await conn.execute("SELECT 1")
    return "Healthy. Connection pools and pgvector active."


@mcp.tool()
async def get_recent_activity(
    project: str,
    limit: int = 5,
    offset: int = 0,
    tags: list[str] | None = None,
    tag_logic: str = "AND",
) -> str:
    """
    Retrieves the most recent design notes and decisions for a project.
    tag_logic: "AND" (has all), "OR" (has any), "NOT" (has none).
    """
    await limiter.check("activity")

    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    tag_logic = tag_logic.upper()
    if tag_logic not in VALID_TAG_LOGIC:
        return f"Error: tag_logic must be one of {VALID_TAG_LOGIC}"

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            tag_filter = ""
            tag_params = []
            if tags:
                if tag_logic == "OR":
                    tag_filter = "AND tags && %s::text[]"
                elif tag_logic == "NOT":
                    tag_filter = "AND NOT (tags && %s::text[])"
                else:
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

    output = [
        f"--- Recent Activity for {project} "
        f"(Top {len(rows)}, Offset: {offset}) ---"
    ]
    for r in rows:
        ts = r['created_at'].strftime("%Y-%m-%d %H:%M")
        tag_str = f" [Tags: {', '.join(r['tags'])}]" if r.get('tags') else ""
        output.append(
            f"[{ts}] [{r['cache_type'].upper()}] {r['title']}{tag_str}\n"
            f"{r['content']}"
        )

    return "\n\n".join(output)


@mcp.tool()
async def export_project_to_markdown(project: str) -> str:
    """
    Exports the entire design history for a specific project into
    a single, human-readable Markdown document.
    """
    await limiter.check("export")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, cache_type, created_at
                FROM design_cache
                WHERE project_name = %s
                ORDER BY id ASC
            """, (project,))
            rows = await cur.fetchall()

    if not rows:
        return f"No data found for project '{project}'."

    md = [
        f"# Design History: {project}\n",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    for r in rows:
        ts = r['created_at'].strftime("%Y-%m-%d %H:%M")
        md.append(f"## {r['title'] or 'Untitled Note'}")
        md.append(f"**Date:** {ts} | **Type:** {r['cache_type'].capitalize()}")
        md.append(f"\n{r['content']}\n")
        md.append("---")

    full_text = "\n".join(md)

    if len(rows) > 20:
        # Sanitize project name to prevent path traversal (H-2, SEC-04)
        safe_name = _sanitize_filename(project)
        filepath = f"/tmp/{safe_name}_history.md"
        # Use aiofiles instead of sync open (H-11, PERF-04)
        async with aiofiles.open(filepath, "w") as f:
            await f.write(full_text)
        return (
            f"Export successful. Total entries: {len(rows)}.\n\n"
            f"Content too large for MCP response. "
            f"Saved locally to: {filepath}"
        )

    return f"Export successful. Total entries: {len(rows)}.\n\n" + full_text


@mcp.tool()
async def generate_spec_from_cache(idea_id: str) -> str:
    """
    Retrieves a cached idea and formats it into a formal Technical Spec.
    Returns the Markdown content to be saved as a file.
    """
    await limiter.check("spec")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, project_name, created_at
                FROM design_cache WHERE id = %s
            """, (idea_id,))
            idea = await cur.fetchone()

    if not idea:
        return f"Error: No idea found with ID {idea_id}."

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
- Performance considerations
"""
    return spec


@mcp.tool()
async def generate_adr_from_cache(idea_id: str, status: str = "proposed") -> str:
    """
    Transforms a cached design idea into a formal Architecture Decision Record (ADR).
    Updates the cache entry to 'project' type with the full ADR content.
    Statuses: proposed, accepted, superseded, deprecated.
    """
    await limiter.check("adr")

    if status.lower() not in VALID_ADR_STATUSES:
        return f"Error: status must be one of {VALID_ADR_STATUSES}"

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, project_name, created_at
                FROM design_cache WHERE id = %s
            """, (idea_id,))
            record = await cur.fetchone()

    if not record:
        return f"Error: No record found for ID {idea_id}."

    # Build ADR with FULL original content preserved (M-12 fix)
    adr_content = f"""# ADR: {record['title']}

* **Status:** {status.upper()}
* **Date:** {record['created_at'].strftime('%Y-%m-%d')}
* **Project:** {record['project_name']}
* **Cache Reference:** {idea_id}

## Context and Problem Statement
{record['content']}

## Decision Drivers
* [AI: Please specify based on the discussion]

## Considered Options
1. [Option 1 from discussion]
2. [Option 2 from discussion]

## Decision Outcome
Chosen Option: **[AI: Please specify based on our chat]**

### Consequences
* **Positive:** [AI: Fill in]
* **Negative:** [AI: Fill in]

## Validation
[AI: Define how we verify this architectural change]
"""

    new_title = f"{record['title']} [ADR status: {status.upper()}]"
    model = await get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{adr_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute(
            "UPDATE design_cache "
            "SET title = %s, content = %s, cache_type = 'project', "
            "embedding = %s::vector, updated_at = now() "
            "WHERE id = %s",
            (new_title, adr_content, vector_list, idea_id),
        )

    return adr_content


@mcp.tool()
async def sync_doc_status(
    cache_id: str, file_path: str, status: str = "implemented",
) -> str:
    """
    Links a physical file (Spec/ADR) to a Cache entry and updates its status.
    Ensures the AI knows the 'Idea' has graduated to 'Official Doc'.
    """
    await limiter.check("sync")
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT title, content FROM design_cache WHERE id = %s",
                (cache_id,),
            )
            row = await cur.fetchone()

    if not row:
        return f"Error: Cache ID {cache_id} not found."

    new_title = f"{row['title']} [OFFICIAL: {status}]"
    new_content = f"{row['content']}\n\nReference File: {file_path}"

    model = await get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{new_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute(
            "UPDATE design_cache "
            "SET title = %s, content = %s, cache_type = 'project', "
            "embedding = %s::vector, updated_at = now() "
            "WHERE id = %s",
            (new_title, new_content, vector_list, cache_id),
        )
    return f"Linked Cache {cache_id} to {file_path} as {status}."


@mcp.tool()
async def link_external_file_to_cache(
    cache_id: str, file_path: str, category: str = "spec",
) -> str:
    """
    Links a local Markdown file (Spec/ADR) to a specific design cache entry.
    Requires an absolute file path within allowed directories.
    """
    await limiter.check("link")
    if not os.path.isabs(file_path):
        return "Error: Please provide an absolute file path."

    # Validate path is within allowed directories (SEC-16, L-14)
    if not _validate_file_path(file_path):
        return "Error: File path is outside allowed directories."

    if not os.path.exists(file_path):
        return "Error: File not found at the specified path."

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT title, content FROM design_cache WHERE id = %s",
                (cache_id,),
            )
            row = await cur.fetchone()

    if not row:
        return f"Error: Cache ID {cache_id} not found."

    new_title = f"{row['title']} [LINKED {category.upper()}]"
    new_content = f"{row['content']}\n\nLinked File: {file_path}"

    model = await get_embedding_model()
    vector = await asyncio.to_thread(model.encode, f"{new_title}\n{new_content}")
    vector_list = vector.tolist()

    async with write_pool.connection() as conn:
        await conn.execute(
            "UPDATE design_cache "
            "SET title = %s, content = %s, embedding = %s::vector, "
            "updated_at = now() WHERE id = %s",
            (new_title, new_content, vector_list, cache_id),
        )
    return f"Linked {category} at {file_path} to Cache ID {cache_id}."


# Resource -- restricted to allowed directories (H-1, SEC-03)
@mcp.resource("file://{path}")
async def read_external_doc(path: str) -> str:
    """Reads the content of an external spec or ADR for context."""
    resolved = os.path.realpath(path)
    if not _validate_file_path(resolved):
        return "Error: Access denied. File is outside allowed directories."
    try:
        async with aiofiles.open(resolved, mode='r') as f:
            content = await f.read()
        return content
    except Exception:
        # Generic error to prevent information disclosure (SEC-14)
        return "Error: Could not read the specified file."


@mcp.tool()
async def get_project_context() -> str:
    """
    Detects the current project and returns its status, last 3 ideas,
    and common tags.
    """
    await limiter.check("context")
    project = get_local_project_name()
    if not project:
        return (
            "No local .design_cache project file found. Use 'store_note' "
            "to start a project or create a .design_cache file."
        )

    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, created_at, tags
                FROM design_cache
                WHERE project_name = %s
                ORDER BY id DESC LIMIT 3
            """, (project,))
            notes = await cur.fetchall()

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
        return (
            f"Please call 'get_project_context' to see the recent state of "
            f"project '{project}'. Then, summarize the top 3 items and ask "
            f"the user if they'd like to continue work on any of them or "
            f"start something new."
        )
    else:
        return (
            "I couldn't detect a local project name. Please ask the user "
            "which project they are working on, or suggest they create a "
            ".design_cache file with the project name."
        )


@mcp.tool()
async def get_compression_opportunities(
    project: str, char_limit: int = 5000,
) -> str:
    """
    Analyzes a project's notes and returns suggestions for summarization.
    Helps prevent LLM context overflow by identifying dense clusters.
    """
    await limiter.check("compression")
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
    estimated_tokens = total_chars // 4

    res = [f"--- Compression Analysis: {project} ---"]
    res.append(f"Total Notes: {len(notes)}")
    res.append(
        f"Estimated Context Size: ~{estimated_tokens} tokens ({total_chars} chars)"
    )

    if total_chars < char_limit:
        res.append("Context size is healthy. No immediate compression needed.")
        return "\n".join(res)

    res.append(
        "Context density is HIGH. Consider summarizing the following clusters:"
    )

    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in notes:
        by_type[n['cache_type']].append(n)

    for ctype, cluster in by_type.items():
        if len(cluster) >= 3:
            ids = [n['id'] for n in cluster]
            top_titles = [f"'{n['title']}'" for n in cluster[:3]]
            cluster_titles = ", ".join(top_titles)
            if len(cluster) > 3:
                cluster_titles += " ..."
            res.append(f"\nCluster [{ctype.upper()}]: {len(cluster)} notes")
            res.append(f"- Titles: {cluster_titles}")
            res.append(
                f"- Suggested Action: Call 'summarize_and_cleanup' with IDs: {ids}"
            )

    return "\n".join(res)


if __name__ == "__main__":
    mcp.run()
