# RAG Pipeline Setup Guide

Complete guide to set up the RAG (Retrieval-Augmented Generation) pipeline for glaucoma detection using PostgreSQL + pgvector, sentence-transformers, and Mistral-7B via Ollama.

---

## Prerequisites

1. **PostgreSQL** installed on your system
2. **pgvector** extension installed
3. **Python 3.8+** with required packages
4. **Ollama** installed and running with Mistral-7B model

---

## Step 1: Install PostgreSQL and pgvector

### Windows

**Option A: Using PostgreSQL Installer + Manual pgvector**

1. Download PostgreSQL from: https://www.postgresql.org/download/windows/
2. Install PostgreSQL (remember your password for `postgres` user)
3. Download pgvector:
   ```bash
   # From https://github.com/pgvector/pgvector/releases
   # Download the appropriate ZIP for your PostgreSQL version
   ```
4. Extract and copy files:
   - Copy `vector.dll` to `C:\Program Files\PostgreSQL\XX\lib\`
   - Copy `vector.control` to `C:\Program Files\PostgreSQL\XX\share\extension\`
   - Copy `vector--*.sql` files to `C:\Program Files\PostgreSQL\XX\share\extension\`

**Option B: Using Docker (Recommended)**

```bash
docker run --name pgvector-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg16
```

### macOS

```bash
# Install PostgreSQL
brew install postgresql

# Install pgvector
brew install pgvector

# Start PostgreSQL
brew services start postgresql
```

### Linux (Ubuntu/Debian)

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-server-dev-all

# Install pgvector
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable pgvector in database
sudo -u postgres psql -c "CREATE EXTENSION vector;"
```

---

## Step 2: Install Python Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# OR install RAG-specific packages only
pip install sentence-transformers psycopg2-binary tiktoken ollama requests
```

---

## Step 3: Configure Database Connection

Edit `scripts/setup_postgres_vector_db.py` and `scripts/generate_and_store_embeddings.py` to set your database credentials:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "YOUR_POSTGRES_PASSWORD",  # CHANGE THIS
    "database": "glaucoma_rag"
}
```

**OR** set environment variables:

```bash
# Windows (PowerShell)
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_USER="postgres"
$env:DB_PASSWORD="yourpassword"
$env:DB_NAME="glaucoma_rag"

# Linux/macOS
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_USER="postgres"
export DB_PASSWORD="yourpassword"
export DB_NAME="glaucoma_rag"
```

---

## Step 4: Create Database and Schema

Run the setup script:

```bash
python scripts/setup_postgres_vector_db.py
```

Expected output:
```
============================================================
PostgreSQL + pgvector Database Setup
============================================================

✓ Connected to PostgreSQL server
✓ Created database: glaucoma_rag
✓ Connected to database: glaucoma_rag
✓ Schema loaded successfully
✓ pgvector extension is enabled
✓ Found 3 tables in database:
  ✓ rag_chunks
  ✓ rag_embeddings
  ✓ rag_metadata
✅ Database Setup Complete!
```

---

## Step 5: Generate and Store Embeddings

Run the embedding generation script:

```bash
python scripts/generate_and_store_embeddings.py
```

This will:
1. Load chunks from `rag_data/chunks/*.json`
2. Load the sentence-transformers model (first run downloads model)
3. Generate embeddings for all chunks
4. Store embeddings in PostgreSQL

Expected output:
```
============================================================
RAG Embedding Generation and Storage
============================================================

✓ Loaded 15 glaucoma chunks
✓ Loaded 9 no-glaucoma chunks
✓ Total chunks: 24
✓ Connected to PostgreSQL database
✓ Inserted 24 chunks
✓ Inserted 24 metadata records
✓ Model loaded (dimension: 384)
✓ Generated embeddings for 24 chunks
✓ Inserted 24 embeddings

✅ Embedding Generation Complete!
```

---

## Step 6: Test RAG Retrieval

Test the retrieval system:

```bash
python scripts/rag_retrieval.py
```

Expected output:
```
============================================================
RAG Retrieval Test
============================================================

Test 1: Basic retrieval
✓ Loaded embedding model: sentence-transformers/all-MiniLM-L6-v2

1. What is Glaucoma? (similarity: 0.823)
   Section: general_info
   Text preview: Glaucoma is a group of eye diseases...

2. Understanding Elevated Eye Pressure (similarity: 0.756)
   Section: causes
   Text preview: **Causes of Elevated Eye Pressure**:...
```

---

## Step 7: Install and Set Up Ollama

### Install Ollama

**Windows/macOS/Linux:**
```bash
# Visit https://ollama.ai and download installer
# OR use curl:
curl https://ollama.ai/install.sh | sh
```

### Download Mistral-7B Model

```bash
ollama pull mistral:7b
```

### Verify Ollama is Running

```bash
ollama list
# Should show: mistral:7b
```

---

## Step 8: Update Schema for All-MiniLM-L6-v2

**Important:** The schema file uses 768 dimensions, but `all-MiniLM-L6-v2` uses **384 dimensions**.

Update `rag_data/pgvector_schema.sql`:

```sql
-- Change line 51 from:
embedding vector(768),

-- To:
embedding vector(384),
```

Then drop and recreate the embeddings table:

```sql
psql -U postgres -d glaucoma_rag
DROP TABLE IF EXISTS rag_embeddings CASCADE;
-- Re-run setup_postgres_vector_db.py
```

---

## Troubleshooting

### Issue: "Connection refused" to PostgreSQL

**Solution:**
1. Ensure PostgreSQL is running: `pg_isready` or check services
2. Check firewall settings
3. Verify `host` and `port` in DB_CONFIG

### Issue: "Extension vector does not exist"

**Solution:**
1. Ensure pgvector is installed correctly
2. Manually enable: `psql -U postgres -d glaucoma_rag -c "CREATE EXTENSION vector;"`
3. Check PostgreSQL logs for errors

### Issue: "Dimension mismatch" error

**Solution:**
- Update `pgvector_schema.sql` to use correct dimension (384 for all-MiniLM-L6-v2)
- Recreate `rag_embeddings` table
- Regenerate embeddings

### Issue: "Model download fails"

**Solution:**
1. Check internet connection
2. Try manually: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
3. Use alternative model: `all-mpnet-base-v2` (768-dim, update schema accordingly)

### Issue: Ollama not found

**Solution:**
1. Ensure Ollama is in PATH
2. Start Ollama server: `ollama serve`
3. Verify with: `ollama list`

---

## Next Steps

1. ✅ Database is set up and populated
2. ✅ Embeddings are generated and stored
3. ✅ RAG retrieval is working
4. ⏭️ Integrate with Streamlit app
5. ⏭️ Connect to Mistral-7B via Ollama
6. ⏭️ Test end-to-end pipeline

See `scripts/integrate_rag_streamlit.py` for Streamlit integration.

---

## Architecture Overview

```
User Upload → ResNet50 Model → Grad-CAM → RAG Retrieval → Mistral-7B → Detailed Report

Components:
- ResNet50: Glaucoma classification (models/resnet50_finetuned.best.h5)
- Grad-CAM: Explainability heatmaps
- PostgreSQL + pgvector: Vector database for semantic search
- sentence-transformers: Embedding generation (all-MiniLM-L6-v2)
- RAG: Retrieves relevant medical documents
- Mistral-7B (Ollama): Generates personalized patient descriptions
```

---

## Files Created

- `scripts/setup_postgres_vector_db.py` - Database setup script
- `scripts/generate_and_store_embeddings.py` - Embedding generation
- `scripts/rag_retrieval.py` - RAG retrieval module
- `RAG_SETUP_GUIDE.md` - This guide
- `rag_data/pgvector_schema.sql` - Database schema
- `requirements.txt` - Updated with RAG dependencies

---

## Verification Checklist

- [ ] PostgreSQL installed and running
- [ ] pgvector extension enabled
- [ ] Python dependencies installed
- [ ] Database created (`glaucoma_rag`)
- [ ] Schema loaded (rag_chunks, rag_embeddings, rag_metadata)
- [ ] Embeddings generated and stored (24 chunks)
- [ ] RAG retrieval test passes
- [ ] Ollama installed and Mistral-7B downloaded
- [ ] Environment variables set (if using)

---

**Ready to integrate with Streamlit!** 🚀

