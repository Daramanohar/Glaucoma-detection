# Quick Start: RAG Data Setup in Google Colab

Follow these steps to set up RAG documents in Google Colab.

## Step 1: Upload Your Project to Colab

If you haven't already, upload your project files to Colab:

```python
# In a Colab cell
from google.colab import drive
drive.mount('/content/drive')

# Copy project (adjust path as needed)
!cp -r /content/drive/MyDrive/Glaucoma_detection /content/
# OR use your GitHub repo
# !git clone https://github.com/yourusername/Glaucoma_detection.git /content/Glaucoma_detection
```

## Step 2: Install Dependencies

```python
!pip install tiktoken numpy
```

## Step 3: Run the Setup Script

```python
import os
import sys

# Navigate to project directory
os.chdir('/content/Glaucoma_detection/Glaucoma_detection')

# Run setup
exec(open('notebooks/setup_rag_data.py').read())
```

**Expected Output:**
```
============================================================
RAG Data Preparation Script
============================================================

Working directory: /content/Glaucoma_detection/Glaucoma_detection
RAG data will be saved to: /content/Glaucoma_detection/Glaucoma_detection/rag_data

Processing GLAUCOMA documents...
✓ Saved 8 raw documents to rag_data/glaucoma/glaucoma_documents.json
✓ Generated 15 chunks for glaucoma
✓ Saved metadata for 15 chunks for glaucoma

Processing NO-GLAUCOMA documents...
✓ Saved 5 raw documents to rag_data/no_glaucoma/no_glaucoma_documents.json
✓ Generated 9 chunks for no_glaucoma
✓ Saved metadata for 9 chunks for no_glaucoma

Generating PostgreSQL schema...
✓ Generated PostgreSQL schema: rag_data/pgvector_schema.sql

[Summary report displayed...]

✅ RAG Data Preparation Complete!
```

## Step 4: Verify Files

```python
import os

rag_dir = '/content/Glaucoma_detection/Glaucoma_detection/rag_data'

# List all files
for root, dirs, files in os.walk(rag_dir):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024
        print(f"{path}: {size:.1f} KB")
```

## Step 5: Preview Data

```python
import json

# Preview glaucoma chunks
with open('rag_data/chunks/glaucoma_chunks.json') as f:
    chunks = json.load(f)
    print(f"Glaucoma chunks: {len(chunks)}")
    print(f"\nFirst chunk:")
    print(chunks[0]['text'][:200])

# Preview metadata
with open('rag_data/metadata/glaucoma_metadata.json') as f:
    meta = json.load(f)
    print(f"\nGlaucoma metadata: {len(meta)} records")
    print(f"Sample: {meta[0]['title']} - {meta[0]['section']}")
```

## Step 6: Download Files (Optional)

```python
import zipfile
from datetime import datetime

# Create zip
zip_name = f'/content/rag_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
with zipfile.ZipFile(zip_name, 'w') as zf:
    zf.write('rag_data', 'rag_data')
    for root, dirs, files in os.walk('rag_data'):
        for f in files:
            zf.write(os.path.join(root, f), os.path.join(root, f))

print(f"✓ Created: {zip_name}")
print(f"📥 Download from Colab files panel")
```

## Next Steps

1. **Set up database** (PostgreSQL + pgvector OR SQLite + vector extension)
2. **Load schema**: Run `pgvector_schema.sql`
3. **Generate embeddings**: Use sentence-transformers
4. **Integrate with Streamlit**: Add RAG retrieval to your app
5. **Connect Mistral-7B**: Via Ollama for text generation

## Troubleshooting

**Issue**: `NameError: name 'os' is not defined`
- **Fix**: Add `import os` at the top of your cell

**Issue**: File not found errors
- **Fix**: Check current directory with `os.getcwd()` and adjust paths

**Issue**: tiktoken not installing
- **Fix**: Run `!pip install tiktoken` in a separate cell first

**Issue**: Script runs but no files created
- **Fix**: Check write permissions, ensure `rag_data/` directory exists

## Files Generated

✅ `rag_data/glaucoma/glaucoma_documents.json` - 8 documents  
✅ `rag_data/no_glaucoma/no_glaucoma_documents.json` - 5 documents  
✅ `rag_data/chunks/glaucoma_chunks.json` - ~15 chunks  
✅ `rag_data/chunks/no_glaucoma_chunks.json` - ~9 chunks  
✅ `rag_data/metadata/glaucoma_metadata.json` - metadata  
✅ `rag_data/metadata/no_glaucoma_metadata.json` - metadata  
✅ `rag_data/pgvector_schema.sql` - database schema  
✅ `rag_data/SUMMARY.txt` - processing summary  

Total: ~24 text chunks ready for RAG pipeline integration.

