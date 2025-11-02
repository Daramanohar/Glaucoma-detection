# 🚀 RAG Pipeline Quick Start Commands

Run these commands in sequence to set up your RAG pipeline.

---

## Prerequisites Check

```bash
# Check PostgreSQL is installed
psql --version

# Check Python version (should be 3.8+)
python --version

# Check if Ollama is installed
ollama --version
```

---

## Step 1: Install Dependencies

```bash
# Install Python packages
pip install sentence-transformers psycopg2-binary tiktoken ollama requests

# Or install all from requirements.txt
pip install -r requirements.txt
```

---

## Step 2: Set Up PostgreSQL (Windows PowerShell)

```powershell
# Download and run pgvector Docker container (easiest method)
docker run --name pgvector-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg16

# Verify container is running
docker ps

# Connect to database
docker exec -it pgvector-db psql -U postgres
```

**OR** if using native PostgreSQL:

```powershell
# Set environment variables
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_USER="postgres"
$env:DB_PASSWORD="postgres"  # Change this to your password
```

---

## Step 3: Set Up Database

```bash
# Create database and schema
python scripts/setup_postgres_vector_db.py
```

Expected output:
```
✓ Connected to PostgreSQL server
✓ Created database: glaucoma_rag
✓ Schema loaded successfully
✅ Database Setup Complete!
```

---

## Step 4: Generate and Store Embeddings

```bash
# Generate embeddings for all chunks
python scripts/generate_and_store_embeddings.py
```

Expected output:
```
✓ Loaded 15 glaucoma chunks
✓ Loaded 9 no-glaucoma chunks
✓ Total chunks: 24
✓ Model loaded (dimension: 384)
✓ Inserted 24 embeddings
✅ Embedding Generation Complete!
```

---

## Step 5: Test RAG Retrieval

```bash
# Test retrieval system
python scripts/rag_retrieval.py
```

Expected output:
```
Test 1: Basic retrieval
1. What is Glaucoma? (similarity: 0.823)
2. Understanding Elevated Eye Pressure (similarity: 0.756)
✅ RAG Retrieval Test Complete!
```

---

## Step 6: Set Up Ollama

```bash
# Install Ollama (if not installed)
# Download from https://ollama.ai/download

# Pull Mistral-7B model
ollama pull mistral:7b

# Verify
ollama list
```

---

## Step 7: Verify Everything Works

```bash
# Check database connection
python -c "from scripts.rag_retrieval import RAGRetriever; r=RAGRetriever(); r.connect(); r.close(); print('✓ RAG system working!')"

# Check Ollama
ollama run mistral:7b "Hello, are you working?"
```

---

## Troubleshooting Quick Fixes

### PostgreSQL not running
```bash
# Docker
docker start pgvector-db

# Windows Service
net start postgresql-x64-16
```

### pgvector extension missing
```bash
# Connect and enable
psql -U postgres -d glaucoma_rag -c "CREATE EXTENSION vector;"
```

### Dimension mismatch error
```bash
# Update schema file: rag_data/pgvector_schema.sql
# Line 42: change vector(768) to vector(384)
# Then re-run setup_postgres_vector_db.py
```

### Ollama not found
```bash
# Start Ollama server
ollama serve

# In another terminal
ollama pull mistral:7b
```

---

## Next Step: Integrate with Streamlit

Once all tests pass, run:

```bash
streamlit run streamlit_app/app.py
```

The Streamlit app will use the RAG system for generating detailed patient descriptions!

---

## Quick Test Commands

```bash
# Test 1: Database connection
python scripts/setup_postgres_vector_db.py

# Test 2: Embeddings
python scripts/generate_and_store_embeddings.py

# Test 3: Retrieval
python scripts/rag_retrieval.py

# Test 4: Ollama
ollama run mistral:7b "What is glaucoma?"

# Test 5: Full pipeline
streamlit run streamlit_app/app.py
```

---

**All tests passing? You're ready for full integration!** ✅

