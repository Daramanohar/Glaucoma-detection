# 🎯 RAG Pipeline - START HERE

Quick guide to get your RAG pipeline up and running on Windows with PowerShell.

---

## ⚡ Quick Setup (30 minutes)

### Step 1: Install PostgreSQL (5 min)

```powershell
# Download and install from:
# https://www.postgresql.org/download/windows/

# During installation:
# - Remember your password (e.g., "postgres123")
# - Default port: 5432
# - Keep all other defaults

# Verify installation:
psql --version
```

### Step 2: Install pgvector (5 min)

```powershell
# Download pre-built binaries:
# Visit: https://github.com/pgvector/pgvector/releases
# Download: windows-16-x64-vector.zip (or matching your PG version)

# Extract and copy to PostgreSQL:
# Copy vector.dll → C:\Program Files\PostgreSQL\16\lib\
# Copy vector.control → C:\Program Files\PostgreSQL\16\share\extension\
# Copy *.sql files → C:\Program Files\PostgreSQL\16\share\extension\

# Enable extension:
psql -U postgres -c "CREATE EXTENSION vector;"
```

### Step 3: Set Password and Test (2 min)

```powershell
# Set your database password
$env:DB_PASSWORD = "postgres123"  # Use YOUR password

# Test connection
psql -U postgres -c "SELECT version();"
```

### Step 4: Install Python Dependencies (3 min)

```powershell
# Install RAG packages
pip install sentence-transformers psycopg2-binary tiktoken ollama requests

# OR install all from requirements.txt
pip install -r requirements.txt
```

### Step 5: Set Up Database (2 min)

```powershell
# Run setup script
python scripts/setup_postgres_vector_db.py
```

**Expected output:**
```
✓ Connected to PostgreSQL server
✓ Created database: glaucoma_rag
✓ Schema loaded successfully
✅ Database Setup Complete!
```

### Step 6: Generate Embeddings (5 min)

```powershell
# Generate embeddings (first run downloads model)
python scripts/generate_and_store_embeddings.py
```

**Expected output:**
```
✓ Loaded 15 glaucoma chunks
✓ Loaded 9 no-glaucoma chunks
✓ Total chunks: 24
✓ Model loaded (dimension: 384)
✓ Inserted 24 embeddings
✅ Embedding Generation Complete!
```

### Step 7: Test RAG Retrieval (2 min)

```powershell
# Test retrieval
python scripts/rag_retrieval.py
```

### Step 8: Install Ollama (5 min)

```powershell
# Download from: https://ollama.ai/download

# Pull Mistral-7B model
ollama pull mistral:7b

# Verify
ollama list
```

---

## 🔍 Troubleshooting

**Run the verification script:**
```powershell
.\scripts\verify_postgres_setup.ps1
```

This checks all prerequisites and tells you what's missing.

---

## 📚 Detailed Guides

- **Windows Setup:** `WINDOWS_POSTGRES_SETUP.md`
- **RAG Pipeline:** `RAG_SETUP_GUIDE.md`
- **Quick Commands:** `QUICK_START_RAG.md`
- **Colab Reference:** `notebooks/RAG_COLAB_QUICKSTART.md`

---

## ✅ What Gets Created

```
glaucoma_rag database:
├── rag_chunks (24 text chunks)
├── rag_metadata (24 metadata records)
└── rag_embeddings (24 vector embeddings)

Files in rag_data/:
├── glaucoma/glaucoma_documents.json
├── no_glaucoma/no_glaucoma_documents.json
├── chunks/*.json
├── metadata/*.json
└── pgvector_schema.sql
```

---

## 🚀 Next Steps

Once all tests pass:

1. ✅ Database is set up
2. ✅ Embeddings are generated
3. ✅ RAG retrieval works
4. ✅ Ollama + Mistral-7B ready
5. ⏭️ **Integrate with Streamlit app**

---

## 🆘 Need Help?

**Common Issues:**

1. **"psql not found"**
   → Add PostgreSQL to PATH or use full path

2. **"Password authentication failed"**
   → See Step 3 in `WINDOWS_POSTGRES_SETUP.md`

3. **"Extension vector does not exist"**
   → Verify pgvector files copied correctly

4. **"Dimension mismatch"**
   → Schema already uses 384 dimensions ✓

**All issues covered in:** `WINDOWS_POSTGRES_SETUP.md`

---

## 📝 Checklist

- [ ] PostgreSQL installed
- [ ] Password set and remembered
- [ ] pgvector extension enabled
- [ ] Python packages installed
- [ ] Database created (glaucoma_rag)
- [ ] Embeddings generated (24 chunks)
- [ ] RAG retrieval tested
- [ ] Ollama installed with Mistral-7B

**All checked?** You're ready for Streamlit integration! 🎉

---

**Questions?** Check the detailed guides or run:
```powershell
.\scripts\verify_postgres_setup.ps1
```

