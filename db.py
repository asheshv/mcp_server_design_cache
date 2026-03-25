"""Database connection pools and schema migrations."""
import sys
import asyncio
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from config import (
    DB_HOST, DB_PORT, DB_NAME, DB_READ_PASS, DB_WRITE_PASS,
    MAX_LIFETIME, MAX_POOL_SIZE, MIN_POOL_SIZE,
)


def _quote_conninfo_value(val: str) -> str:
    """Escape a value for use in a libpq conninfo string.

    Per libpq docs, single quotes inside values are escaped by doubling
    them (not backslash-escaping). Backslashes are escaped as \\\\.
    """
    return "'" + val.replace("\\", "\\\\").replace("'", "''") + "'"


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


# --- Schema Migrations ---
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
    ALTER TABLE design_cache ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
    UPDATE design_cache SET updated_at = created_at WHERE updated_at IS NULL;
    ALTER TABLE design_cache ALTER COLUMN updated_at SET NOT NULL;
    ALTER TABLE design_cache ALTER COLUMN updated_at SET DEFAULT now();
    """,
}


async def apply_migrations():
    """Applies pending migrations with advisory lock and per-migration transactions."""
    async with write_pool.connection() as conn:
        async with conn.cursor() as cur:
            async with conn.transaction():
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT now()
                    )
                """)

            await cur.execute("SELECT pg_advisory_lock(2147483647)")
            try:
                await cur.execute("SELECT MAX(version) FROM schema_version")
                row = await cur.fetchone()
                current_v = row[0] if row and row[0] is not None else 0

                for v in sorted(MIGRATIONS.keys()):
                    if v > current_v:
                        print(f"--- Applying Migration v{v} ---", file=sys.stderr)
                        async with conn.transaction():
                            await cur.execute(MIGRATIONS[v])
                            await cur.execute(
                                "INSERT INTO schema_version (version) "
                                "VALUES (%s) ON CONFLICT (version) DO NOTHING",
                                (v,),
                            )
            finally:
                await cur.execute("SELECT pg_advisory_unlock(2147483647)")
    print("Database schema is up-to-date.", file=sys.stderr)
