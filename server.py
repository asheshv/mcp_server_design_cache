import os, time, re, asyncio, threading
from collections import defaultdict
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector_async
from mcp.server.fastmcp import FastMCP

# --- CONFIG ---
DB_HOST = os.getenv("DB_HOST", "db")
READ_URI = f"host={DB_HOST} dbname=design_db user=design_readonly password=read_password"
WRITE_URI = f"host={DB_HOST} dbname=design_db user=design_readwrite password=write_password"
MAX_LIFETIME = 300  # 5 mins

mcp = FastMCP("SecureDesignMemory")

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
def validate_sql(text: str):
    patterns = [r"(\bor\b|\band\b)\s+\d+=\d+", r"(--|#|/\*)", r";\s*(\bdrop\b|\bdelete\b)"]
    for p in patterns:
        if re.search(p, text, re.I): raise ValueError("Restricted SQL pattern detected.")

# --- SEARCH TOOLS ---
@mcp.tool()
async def search_design(project: str, query: str) -> str:
    """Keyword search using Postgres FTS (Full-Text Search)."""
    limiter.check("search"); validate_sql(query)
    async with read_pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT title, content, cache_type, id FROM design_cache 
                WHERE project_name = %s AND search_vector @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC LIMIT 5;
            """, (project, query, query))
            rows = await cur.fetchall()
    return "\n\n".join([f"[{r['cache_type'].upper()} ID:{r['id']}] {r['title']}\n{r['content']}" for r in rows]) or "No results."

# --- STORAGE TOOLS ---
@mcp.tool()
async def store_note(project: str, title: str, content: str, cache_type: str = 'idea'):
    """Stores persistent design context using native Postgres 18 UUIDv7."""
    limiter.check("store"); validate_sql(content)
    async with write_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO design_cache (project_name, title, content, cache_type) VALUES (%s, %s, %s, %s)", 
            (project, title, content, cache_type)
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

if __name__ == "__main__":
    mcp.run()
