"""
Generate embeddings for RAG chunks and store in PostgreSQL
Uses sentence-transformers for embedding generation
"""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils import get_base_dir

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"[ERROR] Missing required packages: {e}")
    print("Install with: pip install sentence-transformers psycopg2-binary")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = get_base_dir()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),  # CHANGE THIS
    "database": "glaucoma_rag"
}

# Embedding model (all-MiniLM-L6-v2 = 384 dimensions, good balance)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
CHUNKS_DIR = os.path.join(BASE_DIR, "rag_data", "chunks")
METADATA_DIR = os.path.join(BASE_DIR, "rag_data", "metadata")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection():
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[OK] Connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        print("\nPlease ensure PostgreSQL is running and credentials are correct.")
        sys.exit(1)


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def clear_existing_data(conn):
    """Clear existing chunks and embeddings (optional, for re-running)."""
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE rag_embeddings CASCADE")
        cursor.execute("TRUNCATE TABLE rag_metadata CASCADE")
        cursor.execute("TRUNCATE TABLE rag_chunks CASCADE")
        conn.commit()
        print("[OK] Cleared existing data")
    except Exception as e:
        print(f"[WARNING] Warning clearing data: {e}")
        conn.rollback()
    finally:
        cursor.close()


def insert_chunks_and_metadata(conn, chunks: List[Dict], metadata: List[Dict]):
    """Insert chunks and metadata into database."""
    cursor = conn.cursor()
    
    try:
        # Insert chunks
        chunks_insert_sql = """
        INSERT INTO rag_chunks (chunk_id, parent_doc_id, category, text, token_count, chunk_index, total_chunks)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        chunks_data = [
            (
                chunk['chunk_id'],
                chunk['parent_doc_id'],
                chunk['category'],
                chunk['text'],
                chunk['token_count'],
                chunk['chunk_index'],
                chunk['total_chunks']
            )
            for chunk in chunks
        ]
        
        execute_batch(cursor, chunks_insert_sql, chunks_data, page_size=100)
        conn.commit()
        print(f"[OK] Inserted {len(chunks)} chunks")
        
        # Insert metadata
        metadata_insert_sql = """
        INSERT INTO rag_metadata (
            chunk_id, parent_doc_id, category, title, section, condition_stage,
            audience, locale, reading_level, keywords, safety_tags, source, url,
            last_reviewed, chunk_index, total_chunks
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        metadata_data = [
            (
                meta['chunk_id'],
                meta['parent_doc_id'],
                meta['category'],
                meta['title'],
                meta['section'],
                meta['condition_stage'],
                meta['audience'],
                meta['locale'],
                meta['reading_level'],
                meta['keywords'],
                meta['safety_tags'],
                meta['source'],
                meta.get('url', ''),
                meta['last_reviewed'],
                meta['chunk_index'],
                meta['total_chunks']
            )
            for meta in metadata
        ]
        
        execute_batch(cursor, metadata_insert_sql, metadata_data, page_size=100)
        conn.commit()
        print(f"[OK] Inserted {len(metadata)} metadata records")
        
    except Exception as e:
        print(f"[ERROR] Failed to insert chunks/metadata: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()


def insert_embeddings(conn, embeddings: List[Dict]):
    """Insert embeddings into database."""
    cursor = conn.cursor()
    
    try:
        insert_sql = """
        INSERT INTO rag_embeddings (chunk_id, embedding, model_name, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            updated_at = EXCLUDED.updated_at
        """
        
        embeddings_data = [
            (
                emb['chunk_id'],
                emb['embedding'],
                emb['model_name'],
                emb['created_at'],
                emb['updated_at']
            )
            for emb in embeddings
        ]
        
        execute_batch(cursor, insert_sql, embeddings_data, page_size=100)
        conn.commit()
        print(f"[OK] Inserted {len(embeddings)} embeddings")
        
    except Exception as e:
        print(f"[ERROR] Failed to insert embeddings: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(chunks: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """Generate embeddings for all chunks."""
    texts = [chunk['text'] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Prepare embedding records
    embedding_records = []
    now = datetime.now()
    
    for i, chunk in enumerate(chunks):
        embedding_records.append({
            'chunk_id': chunk['chunk_id'],
            'embedding': embeddings[i].tolist(),
            'model_name': EMBEDDING_MODEL_NAME,
            'created_at': now,
            'updated_at': now
        })
    
    return embedding_records


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("RAG Embedding Generation and Storage")
    print("="*60)
    
    # Load chunks and metadata
    print("\nLoading chunks and metadata...")
    glaucoma_chunks = load_json_file(os.path.join(CHUNKS_DIR, "glaucoma_chunks.json"))
    glaucoma_metadata = load_json_file(os.path.join(METADATA_DIR, "glaucoma_metadata.json"))
    
    no_glaucoma_chunks = load_json_file(os.path.join(CHUNKS_DIR, "no_glaucoma_chunks.json"))
    no_glaucoma_metadata = load_json_file(os.path.join(METADATA_DIR, "no_glaucoma_metadata.json"))
    
    all_chunks = glaucoma_chunks + no_glaucoma_chunks
    all_metadata = glaucoma_metadata + no_glaucoma_metadata
    
    print(f"[OK] Loaded {len(glaucoma_chunks)} glaucoma chunks")
    print(f"[OK] Loaded {len(no_glaucoma_chunks)} no-glaucoma chunks")
    print(f"[OK] Total chunks: {len(all_chunks)}")
    
    # Connect to database
    conn = get_db_connection()
    
    # Clear existing data (optional - comment out if you want to keep existing)
    # clear_existing_data(conn)
    
    # Insert chunks and metadata
    print("\nInserting chunks and metadata into database...")
    insert_chunks_and_metadata(conn, all_chunks, all_metadata)
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(This may take a while on first run as model is downloaded)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"[OK] Model loaded (dimension: {embedding_dim})")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(all_chunks, model)
    
    # Insert embeddings
    print("\nStoring embeddings in database...")
    insert_embeddings(conn, embeddings)
    
    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM rag_chunks")
    chunk_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM rag_embeddings")
    emb_count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*60)
    print("[SUCCESS] Embedding Generation Complete!")
    print("="*60)
    print(f"\nDatabase Statistics:")
    print(f"  • Chunks stored: {chunk_count}")
    print(f"  • Embeddings stored: {emb_count}")
    print(f"  • Model: {EMBEDDING_MODEL_NAME}")
    print(f"  • Embedding dimension: {embedding_dim}")
    
    print("\nNext steps:")
    print("1. Test retrieval: python scripts/test_rag_retrieval.py")
    print("2. Integrate with Streamlit app")


if __name__ == "__main__":
    main()

