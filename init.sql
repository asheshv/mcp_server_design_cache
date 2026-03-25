-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS vector;
-- NOTE: uuidv7() requires PostgreSQL 18+ (native support) or the pg_uuidv7
-- extension. If using PostgreSQL < 18, uncomment the following line:
-- CREATE EXTENSION IF NOT EXISTS pg_uuidv7;

-- 2. Main Design Cache Table
CREATE TABLE IF NOT EXISTS design_cache (
    id UUID PRIMARY KEY DEFAULT uuidv7(),
    project_name VARCHAR(255) NOT NULL,
    cache_type VARCHAR(50) CHECK (cache_type IN ('project', 'idea')),
    title VARCHAR(255),
    content TEXT NOT NULL,
    summary_of_ids UUID[],
    tags TEXT[],
    -- Use TIMESTAMPTZ and NOT NULL for correctness (DB-014, DB-015)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Full-Text Search Vector
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    -- Semantic Vector (384 is standard for all-MiniLM-L6-v2)
    embedding vector(384)
);

-- 3. High-Performance Indexes
CREATE INDEX IF NOT EXISTS idx_fts ON design_cache USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_semantic ON design_cache USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_tags ON design_cache USING GIN(tags);
-- Every query filters by project_name (DB-006)
CREATE INDEX IF NOT EXISTS idx_project_name ON design_cache (project_name);

-- 4. Retention Policies Table
CREATE TABLE IF NOT EXISTS retention_policies (
    project_name VARCHAR(255) PRIMARY KEY,
    days_to_retain INTEGER NOT NULL DEFAULT 30 CHECK (days_to_retain > 0),
    auto_compress BOOLEAN DEFAULT TRUE
);

-- 5. Secure RBAC
-- NOTE: Set passwords for these roles after creation using:
--   ALTER ROLE design_readonly PASSWORD 'your_secure_password';
--   ALTER ROLE design_readwrite PASSWORD 'your_secure_password';
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'design_readonly') THEN
        CREATE ROLE design_readonly WITH LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'design_readwrite') THEN
        CREATE ROLE design_readwrite WITH LOGIN;
    END IF;
END $$;

GRANT CONNECT ON DATABASE design_db TO design_readonly, design_readwrite;
GRANT USAGE ON SCHEMA public TO design_readonly, design_readwrite;
GRANT SELECT ON design_cache, retention_policies TO design_readonly;
GRANT ALL PRIVILEGES ON design_cache, retention_policies TO design_readwrite;

-- Future tables created in this schema inherit these grants (DB-009)
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO design_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO design_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO design_readwrite;

-- 6. Schema Versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);
-- Seed at version 4 to match init.sql schema (which already includes tags,
-- idx_semantic, idx_project_name, and updated_at). Migrations only run on
-- upgrades from older schema versions.
INSERT INTO schema_version (version) VALUES (4) ON CONFLICT DO NOTHING;
