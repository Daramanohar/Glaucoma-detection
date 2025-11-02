# ✅ pgvector is Already Enabled! Next Steps

Good news! The error "extension vector already exists" means pgvector is **already installed and enabled**.

---

## What You Just Confirmed:

✅ PostgreSQL is installed  
✅ pgvector extension is enabled  
✅ You can connect to the database  

---

## Next: Set Up Your RAG Database

### Step 1: Create the RAG Database

While you're still in psql (`postgres=#`), type:

```sql
CREATE DATABASE glaucoma_rag;
```

Then:

```sql
\q
```

This exits psql.

---

### Step 2: Enable pgvector in the New Database

Back in PowerShell:

```powershell
psql -U postgres -d glaucoma_rag -c "CREATE EXTENSION vector;"
```

---

### Step 3: Set Your Password (To Avoid Prompts)

In PowerShell:

```powershell
# Replace "your_password" with your actual PostgreSQL password
$env:DB_PASSWORD = "your_password"
```

---

### Step 4: Run the Setup Scripts

```powershell
# Create schema
python scripts/setup_postgres_vector_db.py

# Generate embeddings
python scripts/generate_and_store_embeddings.py

# Test retrieval
python scripts/rag_retrieval.py
```

---

## OR: Do It All at Once

Here's a complete sequence you can copy-paste into PowerShell:

```powershell
# 1. Set your password (change this!)
$env:DB_PASSWORD = "your_postgres_password"

# 2. Create database and enable pgvector
psql -U postgres -c "CREATE DATABASE glaucoma_rag;"
psql -U postgres -d glaucoma_rag -c "CREATE EXTENSION vector;"

# 3. Set up tables and schema
python scripts/setup_postgres_vector_db.py

# 4. Generate embeddings
python scripts/generate_and_store_embeddings.py

# 5. Test everything works
python scripts/rag_retrieval.py
```

---

## Quick Verification

After running the setup, verify everything:

```powershell
# Check database exists
psql -U postgres -lqt | Select-String "glaucoma_rag"

# Check tables were created
psql -U postgres -d glaucoma_rag -c "\dt"

# Check embeddings were stored
psql -U postgres -d glaucoma_rag -c "SELECT COUNT(*) FROM rag_embeddings;"
```

Should show:
```
 count 
-------
    24
```

---

## What Happens Next

1. ✅ Database created → `glaucoma_rag`
2. ✅ Schema loaded → 3 tables (rag_chunks, rag_embeddings, rag_metadata)
3. ✅ Embeddings generated → 24 chunks embedded
4. ✅ Ready for RAG retrieval!

Then you can integrate with Streamlit app.

---

## Ready?

Exit psql (`\q`) and run the commands above! 🚀

