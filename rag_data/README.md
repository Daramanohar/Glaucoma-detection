# RAG Data for Glaucoma Detection Pipeline

This directory contains structured medical documents prepared for the RAG (Retrieval-Augmented Generation) pipeline.

## Directory Structure

```
rag_data/
├── glaucoma/
│   └── glaucoma_documents.json          # Raw glaucoma patient education documents
├── no_glaucoma/
│   └── no_glaucoma_documents.json       # Raw normal eye health documents
├── chunks/
│   ├── glaucoma_chunks.json             # Text chunks (glaucoma)
│   └── no_glaucoma_chunks.json          # Text chunks (normal)
├── metadata/
│   ├── glaucoma_metadata.json           # Metadata with filters (glaucoma)
│   └── no_glaucoma_metadata.json        # Metadata with filters (normal)
├── embeddings/                          # (Generated after processing)
├── pgvector_schema.sql                  # PostgreSQL schema for pgvector
└── SUMMARY.txt                          # Processing summary

```

## Document Categories

### Glaucoma Documents (8 documents)
1. **What is Glaucoma?** - General information
2. **Understanding Elevated Eye Pressure** - Causes
3. **Consequences of Untreated Glaucoma** - Progression and impact
4. **Treatment Options** - Medications, laser, surgery
5. **Lifestyle Modifications** - Self-care strategies
6. **Uncertainty in Diagnosis** - AI limitations and disclaimers
7. **Interpretation of Fundus Images** - Clinical reference
8. **Emergency Red Flags** - When to seek immediate care

### No-Glaucoma Documents (5 documents)
1. **Healthy Eye Anatomy** - Normal vision information
2. **Maintaining Eye Health** - Prevention strategies
3. **Normal vs. High Risk Features** - Risk assessment
4. **When Normal Results Need Attention** - False negatives and limitations
5. **Age-Related Changes** - Normal aging vs. disease

## Metadata Fields

Each document includes:

- `id`: Unique document identifier
- `title`: Document title
- `section`: Category (general_info, causes, consequences, improvements, uncertainty, etc.)
- `condition_stage`: suspected, early, moderate, advanced, healthy
- `audience`: patient or clinician
- `locale`: Language (en)
- `reading_level`: basic, intermediate, or advanced
- `keywords`: List of relevant terms
- `safety_tags`: Important flags (educational, urgent_awareness, non_diagnostic, etc.)
- `source`: Source organization/publication
- `url`: Reference URL (if available)

## Chunking Strategy

- **Target size**: 500-800 tokens per chunk
- **Overlap**: 100 tokens between chunks
- **Total chunks**: ~30-40 chunks across all documents
- **Tokenization**: Uses tiktoken (GPT-style) for accurate counting

## Database Schema (pgvector)

The `pgvector_schema.sql` file defines:

1. **rag_chunks**: Text chunks with indexing
2. **rag_metadata**: Document metadata with full-text search support
3. **rag_embeddings**: Vector embeddings (768-dim for all-MiniLM-L6-v2, or 1024-dim for larger models)

Includes indexes for:
- Category filtering (glaucoma vs no_glaucoma)
- Condition stage filtering
- Audience filtering
- Keyword matching (GIN index)
- Vector similarity search (HNSW index)

## Usage

### 1. Load into PostgreSQL

```bash
# Connect to PostgreSQL
psql your_database

# Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

# Load schema
\i rag_data/pgvector_schema.sql

# Load data (example)
-- This would be done programmatically
```

### 2. Generate Embeddings

```python
from sentence_transformers import SentenceTransformer
import json
import psycopg2

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to database
conn = psycopg2.connect(dbname="your_db", user="user", password="pass")

# Load chunks
with open('rag_data/chunks/glaucoma_chunks.json') as f:
    chunks = json.load(f)

# Generate and store embeddings
for chunk in chunks:
    embedding = model.encode(chunk['text']).tolist()
    # Insert into rag_embeddings table
```

### 3. RAG Retrieval Query

```sql
-- Retrieve top 5 relevant chunks for a query
SELECT 
    c.chunk_id,
    c.text,
    m.title,
    m.section,
    (1 - (e.embedding <=> query_embedding)) as similarity
FROM rag_chunks c
JOIN rag_metadata m ON c.chunk_id = m.chunk_id
JOIN rag_embeddings e ON c.chunk_id = e.chunk_id
WHERE c.category = 'glaucoma'  -- Filter by category
  AND m.condition_stage = 'early'  -- Filter by stage
  AND m.audience = 'patient'  -- Filter by audience
ORDER BY similarity DESC
LIMIT 5;
```

## Integration with Streamlit App

The RAG pipeline will:

1. Accept model prediction (glaucoma probability)
2. Classify into glaucoma vs. no-glaucoma category
3. Retrieve top 3-5 relevant chunks based on:
   - Category match
   - Condition stage (if applicable)
   - Keyword matching with Grad-CAM findings
4. Send chunks to Mistral-7B via Ollama
5. Generate personalized patient description covering:
   - **Causes**: Why this might occur
   - **Consequences**: What could happen if untreated
   - **Improvements**: Treatment and lifestyle suggestions
   - **Uncertainty**: AI limitations and next steps

## Safety and Disclaimers

All documents include appropriate safety tags:

- `educational`: Informational content
- `non_diagnostic`: AI not for clinical diagnosis
- `urgent_awareness`: When to seek immediate care
- `important_disclaimer`: Limitations and uncertainties

The system will emphasize:
- This is a research/demonstration tool
- Not a replacement for professional medical evaluation
- Regular eye exams are essential
- AI limitations (false positives/negatives)
- When to seek emergency care

## Updating Documents

To add or modify documents:

1. Edit `notebooks/setup_rag_data.py`
2. Add documents to `GLAUCOMA_DOCUMENTS` or `NO_GLAUCOMA_DOCUMENTS`
3. Re-run the setup script
4. Regenerate embeddings
5. Update database

## References

- American Academy of Ophthalmology (AAO)
- National Eye Institute (NEI)
- Prevent Blindness Foundation
- Glaucoma Research Foundation
- American Glaucoma Society

---

**Note**: This is for educational/research purposes. Not for clinical diagnosis or treatment decisions.

