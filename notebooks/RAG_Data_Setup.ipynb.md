# RAG Data Setup for Glaucoma Detection - Google Colab

Run this notebook in Google Colab to prepare RAG documents for your glaucoma detection pipeline.

---

## Cell 1: Install Dependencies

```python
!pip install tiktoken numpy
print("✓ Dependencies installed")
```

---

## Cell 2: Run Setup Script

```python
# Copy and run the setup_rag_data.py script
exec(open('/content/Glaucoma_detection/Glaucoma_detection/notebooks/setup_rag_data.py').read())
```

**OR** import as module:

```python
import sys
sys.path.append('/content/Glaucoma_detection/Glaucoma_detection')

from notebooks.setup_rag_data import main
main()
```

---

## Cell 3: Verify Generated Files

```python
import os
from pathlib import Path

base = "/content/Glaucoma_detection/Glaucoma_detection/rag_data"

# Check structure
for root, dirs, files in os.walk(base):
    level = root.replace(base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')

# Show file sizes
print("\n📊 File Statistics:")
for dir_name in ['glaucoma', 'no_glaucoma', 'chunks', 'metadata']:
    dir_path = os.path.join(base, dir_name)
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        total_size = sum(os.path.getsize(os.path.join(dir_path, f)) 
                        for f in files if os.path.isfile(os.path.join(dir_path, f)))
        print(f"  {dir_name}: {len(files)} files, {total_size/1024:.1f} KB")
```

---

## Cell 4: Preview Generated Data

```python
import json
import os

base = "/content/Glaucoma_detection/Glaucoma_detection/rag_data"

# Preview glaucoma chunks
print("="*60)
print("Preview: Glaucoma Chunks")
print("="*60)
chunks_file = os.path.join(base, "chunks", "glaucoma_chunks.json")
with open(chunks_file, 'r') as f:
    chunks = json.load(f)
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"\nFirst chunk (tokens: {chunks[0]['token_count']}):")
        print(chunks[0]['text'][:300] + "...")

# Preview metadata
print("\n" + "="*60)
print("Preview: Metadata")
print("="*60)
metadata_file = os.path.join(base, "metadata", "glaucoma_metadata.json")
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
    print(f"\nTotal records: {len(metadata)}")
    if metadata:
        print("\nFirst metadata record:")
        print(json.dumps(metadata[0], indent=2))
```

---

## Cell 5: Download Files (Optional - for local backup)

```python
import zipfile
import os
from datetime import datetime

# Create zip file
base = "/content/Glaucoma_detection/Glaucoma_detection"
rag_dir = os.path.join(base, "rag_data")
zip_path = f"/content/rag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, files in os.walk(rag_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base)
            zipf.write(file_path, arcname)

print(f"✓ Created zip file: {zip_path}")
print(f"  File size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
print(f"\n📥 Download link:")
print(f"   {zip_path}")
```

---

## What's Next?

After running this notebook:

1. **Set up PostgreSQL + pgvector** (or use SQLite with vector extension)
2. **Load schema**: Run `pgvector_schema.sql`
3. **Generate embeddings**: Use sentence-transformers or OpenAI
4. **Implement RAG retrieval** in Streamlit app
5. **Connect Mistral-7B** via Ollama for generation

See `notebooks/rag_pipeline_setup.py` for the complete RAG pipeline integration.

