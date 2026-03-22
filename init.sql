-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Main Design Cache Table
CREATE TABLE IF NOT EXISTS design_cache (
    id UUID PRIMARY KEY DEFAULT uuidv7(), -- Native Postgres 18 UUIDv7
    project_name VARCHAR(255) NOT NULL,
    cache_type VARCHAR(50) CHECK (cache_type IN ('project', 'idea')),
    title VARCHAR(255),
    content TEXT NOT NULL,
    summary_of_ids UUID[], -- Matches UUIDv7 ID type
    tags TEXT[], -- Array for categorization [NEW]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
CREATE INDEX IF NOT EXISTS idx_tags ON design_cache USING GIN(tags); -- Index for tags [NEW]

-- 4. Retention Policies Table
CREATE TABLE IF NOT EXISTS retention_policies (
    project_name VARCHAR(255) PRIMARY KEY,
    days_to_retain INTEGER DEFAULT 30,
    auto_compress BOOLEAN DEFAULT TRUE
);

-- 5. Secure RBAC
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'design_readonly') THEN
        CREATE ROLE design_readonly WITH LOGIN PASSWORD 'read_password';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'design_readwrite') THEN
        CREATE ROLE design_readwrite WITH LOGIN PASSWORD 'write_password';
    END IF;
END $$;

GRANT CONNECT ON DATABASE design_db TO design_readonly, design_readwrite;
GRANT USAGE ON SCHEMA public TO design_readonly, design_readwrite;
GRANT SELECT ON design_cache, retention_policies TO design_readonly;
GRANT ALL PRIVILEGES ON design_cache, retention_policies TO design_readwrite;

-- 6. Schema Versioning [NEW]
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
