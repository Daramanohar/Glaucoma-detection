
-- PostgreSQL + pgvector Schema for RAG Pipeline

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table (stores text chunks)
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id UUID PRIMARY KEY,
    parent_doc_id VARCHAR(50),
    category VARCHAR(20),  -- 'glaucoma' or 'no_glaucoma'
    text TEXT NOT NULL,
    token_count INTEGER,
    chunk_index INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metadata table (stores document metadata)
CREATE TABLE IF NOT EXISTS rag_metadata (
    chunk_id UUID PRIMARY KEY REFERENCES rag_chunks(chunk_id),
    parent_doc_id VARCHAR(50),
    category VARCHAR(20),
    title VARCHAR(500),
    section VARCHAR(100),
    condition_stage VARCHAR(50),
    audience VARCHAR(20),
    locale VARCHAR(10),
    reading_level VARCHAR(20),
    keywords TEXT[],
    safety_tags TEXT[],
    source VARCHAR(500),
    url TEXT,
    last_reviewed TIMESTAMP,
    chunk_index INTEGER,
    total_chunks INTEGER
);

-- Embeddings table with vector column
CREATE TABLE IF NOT EXISTS rag_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES rag_chunks(chunk_id),
    embedding vector(768),  -- Change to 1024 if using larger models
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_rag_chunks_category ON rag_chunks(category);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_category ON rag_metadata(category);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_condition_stage ON rag_metadata(condition_stage);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_audience ON rag_metadata(audience);
CREATE INDEX IF NOT EXISTS idx_rag_metadata_keywords ON rag_metadata USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_model ON rag_embeddings(model_name);

-- Vector similarity search index (HNSW for faster approximate search)
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector ON rag_embeddings 
USING hnsw (embedding vector_cosine_ops);

-- Composite index for filtered similarity search
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_category_model ON rag_embeddings(chunk_id, model_name) 
INCLUDE (chunk_id);

COMMENT ON TABLE rag_chunks IS 'Text chunks for RAG retrieval';
COMMENT ON TABLE rag_metadata IS 'Document metadata for filtering and ranking';
COMMENT ON TABLE rag_embeddings IS 'Vector embeddings for semantic similarity search';
