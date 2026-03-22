import os, time, re, asyncio, threading
from collections import defaultdict
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from mcp.server.fastmcp import FastMCP
from pgvector.psycopg import register_vector_async

# --- CONFIGURATION ---
DB_HOST = os.getenv("DB_HOST", "db") # Default to 'db' for Docker Compose
READ_URI = f"host={DB_HOST} dbname=design_db user=design_readonly password=read_password"
WRITE_URI = f"host={DB_HOST} dbname=design_db user=design_readwrite password=write_password"
MAX_LIFETIME = 300  # 30 mins

# --- INITIALIZATION ---
mcp = FastMCP("SecureDesignCache")

async def configure_connection(conn):
    await register_vector_async(conn)

# Initialize pools (do not open them yet)
read_pool = AsyncConnectionPool(
    conninfo=READ_URI, open=False, max_lifetime=MAX_LIFETIME,
    configure=configure_connection,
)
write_pool = AsyncConnectionPool(conninfo=WRITE_URI, open=False, max_lifetime=MAX_LIFETIME)

# --- LIFECYCLE HOOKS ---
@mcp.on_startup()
async def startup():
    """Opens the database connection pools when the server starts."""
    print("Connecting to PostgreSQL pools...")
    await read_pool.open()
    await write_pool.open()
    print("Database pools are active.")

@mcp.on_shutdown()
async def shutdown():
    """Gracefully closes the database connection pools on exit."""
    print("Closing PostgreSQL pools...")
    await read_pool.close()
    await write_pool.close()
    print("Database pools closed.")

# --- SECURITY: RATE LIMITER ---
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.history = defaultdict(list)
        self.lock = threading.Lock()

    def check(self, tool: str):
        with self.lock:
            now = time.time()
            self.history[tool] = [t for t in self.history[tool] if now - t < 60]
            if len(self.history[tool]) >= self.rpm:
                raise RuntimeError(f"Rate limit exceeded (Max {self.rpm} RPM)")
            self.history[tool].append(now)

limiter = RateLimiter(60)

# --- SECURITY: SQL VALIDATION ---
def validate_sql(text: str):
    patterns = [
        r"(\bor\b|\band\b)\s+\d+=\d+", 
        r"(--|#|/\*)", 
        r";\s*(\bdrop\b|\bdelete\b|\btruncate\b)",
        r"(\bunion\b\s+.*?\bselect\b)"
    ]
    for p in patterns:
        if re.search(p, text, re.I):
            raise ValueError(f"Security Alert: Restricted SQL pattern detected.")

# --- TOOLS ---
@mcp.tool()
async def search_design(project: str, query: str) -> str:
    """Search design notes using Full-Text Search. Uses read-only pool."""
    limiter.check("search")
    validate_sql(query)
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, cache_type, id FROM design_cache 
                WHERE project_name = %s AND search_vector @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC LIMIT 5;
            """, (project, query, query))
            rows = await cur.fetchall()
    
    if not rows: return "No relevant notes found."
    return "\n\n".join([f"[{r['cache_type'].upper()} ID:{r['id']}] {r['title']}\n{r['content']}" for r in rows])

@mcp.tool()
async def semantic_search(project: str, query_text: str) -> str:
    """Finds design notes by meaning, not just keywords."""
    # 1. Generate embedding for the query (Helper function required)
    query_vector = await generate_embedding(query_text) 
    
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, 1 - (embedding <=> %s) AS similarity
                FROM design_cache
                WHERE project_name = %s
                ORDER BY embedding <=> %s LIMIT 5;
            """, (query_vector, project, query_vector))
            rows = await cur.fetchall()
    return format_results(rows)

@mcp.tool()
async def store_note(project: str, title: str, content: str, cache_type: str = 'idea'):
    """Saves a new design note. Uses write-capable pool."""
    limiter.check("store")
    validate_sql(content)
    async with write_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO design_cache (project_name, title, content, cache_type) VALUES (%s, %s, %s, %s)", 
            (project, title, content, cache_type)
        )
    return "Note cached successfully."

@mcp.tool()
async def summarize_and_cleanup(project: str, ids_to_summarize: list[str], summary_text: str, new_title: str):
    """
    Merges granular notes into a summary and deletes originals.
    Note: ids_to_summarize now accepts UUID strings.
    """
    limiter.check("summarize")
    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO design_cache (project_name, content, cache_type, title, summary_of_ids) VALUES (%s, %s, 'project', %s, %s)",
                (project, summary_text, new_title, ids_to_summarize)
            )
            await cur.execute("DELETE FROM design_cache WHERE id = ANY(%s)", (ids_to_summarize,))
    return f"Consolidated {len(ids_to_summarize)} notes into '{new_title}'."

@mcp.tool()
async def compress_old_context(project: str, days_threshold: int = 7) -> str:
    """
    Identifies 'idea' notes older than the threshold and returns them 
    so the AI can generate a high-level summary to replace them.
    """
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, title, content FROM design_cache 
                WHERE project_name = %s 
                AND cache_type = 'idea' 
                AND created_at < NOW() - INTERVAL '%s days'
            """, (project, f"{days_threshold} days"))
            old_notes = await cur.fetchall()
            
    if not old_notes:
        return "No old context found to compress."
    
    # Return the raw text to the AI so it can create the summary
    formatted = "\n".join([f"ID: {n['id']} | {n['title']}: {n['content']}" for n in old_notes])
    return f"Found {len(old_notes)} old notes. Please summarize these and use 'summarize_and_cleanup' to save."

@mcp.tool()
async def health_check():
    """Verify database connection health."""
    async with read_pool.connection() as conn:
        await conn.execute("SELECT 1")
    return "💚 Healthy. Connection pools active."

@mcp.tool()
async def check_maintenance_required(size_limit_mb: int = 500, row_limit: int = 1000) -> str:
    """
    Checks if database bloat is affecting performance or token efficiency.
    Triggers if size > 100MB OR total 'idea' rows > 1000.
    """
    limit_bytes = size_limit_mb * 1024 * 1024
    
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # 1. Check Physical Disk Size
            await cur.execute("SELECT pg_database_size(current_database()) as size_bytes")
            db_size = (await cur.fetchone())['size_bytes']
            
            # 2. Check Row Density (More important for LLM context)
            await cur.execute("SELECT count(*) as count FROM design_cache WHERE cache_type = 'idea'")
            idea_count = (await cur.fetchone())['count']
            
            size_mb = db_size / (1024 * 1024)

            if db_size < limit_bytes and idea_count < row_limit:
                return f"💚 System Healthy: {size_mb:.2f}MB, {idea_count} granular ideas cached."

            # 3. If Bloated, identify the oldest project with the most 'noise'
            await cur.execute("""
                SELECT project_name, count(*) as cnt 
                FROM design_cache WHERE cache_type = 'idea'
                GROUP BY project_name ORDER BY cnt DESC LIMIT 1
            """)
            noisy_project = await cur.fetchone()
            
            return (f"🚨 MAINTENANCE SUGGESTED: System has {idea_count} ideas ({size_mb:.2f}MB). "
                    f"Project '{noisy_project['project_name']}' has the most bloat ({noisy_project['cnt']} notes). "
                    "Please search this project and use 'summarize_and_cleanup' to condense the history.")

@mcp.tool()
async def set_project_retention_policy(project: str, days: int, compress: bool = True) -> str:
    """
    Sets a custom cleanup policy for a project.
    - days: Number of days to keep granular 'idea' notes.
    - compress: If True, old notes are summarized before deletion.
    """
    async with write_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO retention_policies (project_name, days_to_retain, auto_compress)
            VALUES (%s, %s, %s)
            ON CONFLICT (project_name) DO UPDATE 
            SET days_to_retain = EXCLUDED.days_to_retain, 
                auto_compress = EXCLUDED.auto_compress,
                updated_at = CURRENT_TIMESTAMP
        """, (project, days, compress))
    return f"✅ Policy updated for '{project}': {days} days retention, auto-compress={compress}."

@mcp.tool()
async def run_smart_cleanup() -> str:
    """
    Scans all projects and identifies 'idea' notes that have 
    exceeded their specific project retention period.
    """
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Join cache with policies to find expired notes
            await cur.execute("""
                SELECT c.project_name, COUNT(c.id) as expired_count
                FROM design_cache c
                JOIN retention_policies p ON c.project_name = p.project_name
                WHERE c.cache_type = 'idea'
                AND p.days_to_retain > 0
                AND c.created_at < NOW() - (p.days_to_retain || ' days')::interval
                GROUP BY c.project_name
            """)
            expired = await cur.fetchall()

    if not expired:
        return "💚 All projects are within their retention limits."

    report = ["🚨 Retention limits exceeded for the following projects:"]
    for item in expired:
        report.append(f"- {item['project_name']}: {item['expired_count']} notes ready for compression.")
    
    return "\n".join(report) + "\n\nPlease use 'summarize_and_cleanup' to process these."

@mcp.tool()
async def hybrid_search(project: str, query: str) -> str:
    """
    Performs a 'Hybrid Search' using both Keyword (FTS) and Semantic (Vector) 
    matching to give the most accurate project memory.
    """
    # 1. Get embedding for the query
    query_vector = await get_embedding(query)
    
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # PostgreSQL 18 handles this complex join/rank very efficiently
            await cur.execute("""
                SELECT title, content, 
                       ts_rank(search_vector, plainto_tsquery('english', %s)) as keyword_score,
                       (1 - (embedding <=> %s)) as semantic_score
                FROM design_cache 
                WHERE project_name = %s 
                AND (search_vector @@ plainto_tsquery('english', %s) OR embedding <=> %s < 0.5)
                ORDER BY (keyword_score + semantic_score) DESC LIMIT 5;
            """, (query, query_vector, project, query, query_vector))
            rows = await cur.fetchall()
            
    return format_results(rows)

if __name__ == "__main__":
    # Note: FastMCP handles the event loop and triggers startup/shutdown hooks
    mcp.run()
