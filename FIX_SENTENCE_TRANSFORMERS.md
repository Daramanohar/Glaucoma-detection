# Fix sentence-transformers Installation Issue

## The Problem

sentence-transformers is failing due to torch/torchvision compatibility issues.

## Quick Fix Options

### Option 1: Reinstall sentence-transformers (Recommended)

```powershell
# Uninstall conflicting packages
pip uninstall sentence-transformers torch torchvision transformers -y

# Reinstall with compatible versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
```

### Option 2: Use CPU-only PyTorch (Simpler)

```powershell
# Install CPU-only versions
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
```

### Option 3: Use Alternative Embedding Model

If sentence-transformers still fails, we can use a different approach with TensorFlow:

```powershell
# Install Universal Sentence Encoder (Google's model)
pip install tensorflow-hub
```

Then modify the embedding generation script to use USE instead.

### Option 4: Skip sentence-transformers (Temporary)

The system will work with placeholder embeddings for testing, but **semantic search won't work properly** until real embeddings are generated.

---

## Verify Installation

```powershell
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

Should print: `OK`

If you get an error, try Option 1 or 2 above.

---

## After Fixing, Re-run Embedding Generation

```powershell
# Clear existing placeholder embeddings
python scripts/reset_rag_database.ps1

# Or manually:
psql -U postgres -d glaucoma_rag -c "TRUNCATE TABLE rag_embeddings, rag_metadata, rag_chunks CASCADE;"

# Regenerate with real embeddings
python scripts/generate_embeddings_simple.py
```

---

## For Now (Testing)

The placeholder embeddings are stored. You can test the rest of the pipeline, but **semantic search will return random results**. This is fine for testing database setup, but you'll need real embeddings for production.

---

**Recommendation:** Try **Option 2** first (CPU-only PyTorch), as it's the simplest fix.

