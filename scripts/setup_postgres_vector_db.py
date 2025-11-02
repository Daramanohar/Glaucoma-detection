"""
PostgreSQL + pgvector Database Setup Script
Creates database, enables pgvector extension, and sets up schema for RAG pipeline
"""

import os
import sys
import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils import get_base_dir


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = get_base_dir()

# PostgreSQL connection parameters
# Modify these to match your PostgreSQL setup
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),  # CHANGE THIS to your password
    "database": "postgres"  # Connect to default postgres db first
}

# Target database for RAG
TARGET_DB_NAME = "glaucoma_rag"

# Schema file path
SCHEMA_FILE = os.path.join(BASE_DIR, "rag_data", "pgvector_schema.sql")


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def connect_to_postgres(config=None):
    """Connect to PostgreSQL server."""
    if config is None:
        config = DB_CONFIG
    
    try:
        conn = psycopg2.connect(**config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print(f"[OK] Connected to PostgreSQL server")
        return conn
    except psycopg2.Error as e:
        print(f"[ERROR] Failed to connect to PostgreSQL: {e}")
        print("\nPlease ensure PostgreSQL is running and credentials are correct.")
        sys.exit(1)


def database_exists(conn, db_name):
    """Check if database exists."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (db_name,)
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    return exists


def create_database(conn, db_name):
    """Create a new PostgreSQL database."""
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'CREATE DATABASE {db_name}')
        print(f"[OK] Created database: {db_name}")
    except psycopg2.errors.DuplicateDatabase:
        print(f"[WARNING] Database '{db_name}' already exists")
    except Exception as e:
        print(f"[ERROR] Failed to create database: {e}")
        sys.exit(1)
    finally:
        cursor.close()


def load_schema_file(schema_file):
    """Load SQL schema file."""
    if not os.path.exists(schema_file):
        print(f"[ERROR] Schema file not found: {schema_file}")
        sys.exit(1)
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        return f.read()


def execute_schema(conn, schema_sql):
    """Execute schema SQL to create tables."""
    cursor = conn.cursor()
    
    try:
        # Split by semicolon and execute each statement
        statements = schema_sql.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                except Exception as e:
                    # Ignore "already exists" errors, report others
                    if "already exists" not in str(e).lower():
                        print(f"[WARNING] Warning executing statement: {e}")
        
        conn.commit()
        print("[OK] Schema loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()


def verify_setup(conn):
    """Verify tables and pgvector extension are set up correctly."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("Verifying Database Setup")
    print("="*60)
    
    # Check pgvector extension
    cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    if cursor.fetchone():
        print("[OK] pgvector extension is enabled")
    else:
        print("[ERROR] pgvector extension is NOT enabled")
    
    # Check tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    
    expected_tables = ['rag_chunks', 'rag_embeddings', 'rag_metadata']
    print(f"\n[OK] Found {len(tables)} tables in database:")
    for table in tables:
        marker = "[OK]" if table[0] in expected_tables else "-"
        print(f"  {marker} {table[0]}")
    
    # Check indexes
    cursor.execute("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE schemaname = 'public'
        ORDER BY indexname;
    """)
    indexes = cursor.fetchall()
    print(f"\n[OK] Created {len(indexes)} indexes for optimization")
    
    cursor.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("PostgreSQL + pgvector Database Setup")
    print("="*60)
    print(f"\nTarget database: {TARGET_DB_NAME}")
    print(f"Schema file: {SCHEMA_FILE}")
    
    # Connect to PostgreSQL
    conn = connect_to_postgres()
    
    # Create database if it doesn't exist
    if not database_exists(conn, TARGET_DB_NAME):
        create_database(conn, TARGET_DB_NAME)
    else:
        print(f"[OK] Database '{TARGET_DB_NAME}' already exists")
    
    conn.close()
    
    # Connect to target database
    db_config = DB_CONFIG.copy()
    db_config["database"] = TARGET_DB_NAME
    conn = psycopg2.connect(**db_config)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    print(f"[OK] Connected to database: {TARGET_DB_NAME}")
    
    # Load and execute schema
    print("\nLoading schema file...")
    schema_sql = load_schema_file(SCHEMA_FILE)
    execute_schema(conn, schema_sql)
    
    # Verify setup
    verify_setup(conn)
    
    conn.close()
    
    print("\n" + "="*60)
    print("[SUCCESS] Database Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/generate_and_store_embeddings.py")
    print("2. Verify embeddings with: python scripts/test_rag_retrieval.py")
    print("3. Integrate with Streamlit app")


if __name__ == "__main__":
    main()

