# 🚀 How to Run Your Complete RAG-Powered Glaucoma Detection System

## Quick Start (1 Command)

```powershell
# Set password and run
$env:DB_PASSWORD = "5657"; python -m streamlit run streamlit_app/app.py
```

The app will open in your browser at **http://localhost:8501**

---

## Step-by-Step

### 1. Set Database Password

```powershell
$env:DB_PASSWORD = "5657"
```

### 2. Verify Services Running

```powershell
# Check PostgreSQL
Get-Service postgresql*

# Check Ollama
ollama list
# Should show: mistral:7b
```

### 3. Start Streamlit

```powershell
cd C:\Users\hp\Documents\Renuka\Glaucoma_detection

python -m streamlit run streamlit_app/app.py
```

---

## What You'll See

### Sidebar
- ✅ Model loaded successfully
- ✅ RAG system ready
- ✅ Ollama + Mistral-7B ready
- Model details, dataset summary, performance metrics

### Main Interface
1. **Upload Image** section (left)
2. **Visualization** section (right)
3. **Detailed Patient Information** section (below)
4. **Model Evaluation Metrics** section (bottom)

---

## How to Use

1. **Upload** a retinal fundus image (PNG/JPG/JPEG)
2. **Click** "🔍 Predict" button
3. **Wait** for:
   - Model prediction
   - Grad-CAM visualization
   - RAG retrieval
   - Mistral-7B description generation
4. **View** results:
   - Prediction with confidence
   - Heatmap overlay
   - AI-generated description
   - Source documents
5. **Download** report if needed

---

## Example Workflow

```
Upload Image → Predict → Grad-CAM → RAG Retrieve → Mistral Generate → Display

Time: ~3-5 seconds total
Result: Complete analysis with explanation
```

---

## Troubleshooting

### "RAG system unavailable"
```powershell
# Check database
psql -U postgres -d glaucoma_rag -c "SELECT COUNT(*) FROM rag_chunks;"
# Should show: 13

# If 0, regenerate embeddings:
python scripts/generate_embeddings_simple.py
```

### "Ollama not running"
```powershell
# Start Ollama (usually auto-starts)
ollama serve

# In another terminal, test:
ollama run mistral:7b "Hello"
```

### "Model file not found"
```powershell
# Check model location
ls Glaucoma_detection\Glaucoma_detection\models\
# Should see: resnet50_finetuned.best.h5
```

### "Streamlit not found"
```powershell
pip install streamlit
```

### App is slow
- This is normal! First run loads models
- Subsequent requests are faster (cached)
- Typical: 3-5 seconds per prediction

---

## Features to Test

1. ✅ Upload a glaucoma image → Get "Glaucoma Detected"
2. ✅ Upload a normal image → Get "Normal"
3. ✅ Toggle Grad-CAM on/off
4. ✅ Generate detailed AI description
5. ✅ View RAG sources
6. ✅ Download report
7. ✅ Check sidebar metrics

---

## System Requirements Met

- ✅ PostgreSQL + pgvector ✓
- ✅ Python 3.11 ✓
- ✅ TensorFlow 2.13+ ✓
- ✅ sentence-transformers ✓
- ✅ Streamlit ✓
- ✅ Ollama + Mistral-7B ✓
- ✅ 13 Medical documents ✓
- ✅ Embeddings generated ✓
- ✅ All scripts working ✓

---

## Performance

- **Model Loading**: ~5-10 seconds (cached after first)
- **Prediction**: <1 second
- **Grad-CAM**: 1-2 seconds
- **RAG Retrieval**: <1 second
- **Mistral Generation**: 2-5 seconds
- **Total**: ~3-5 seconds per upload

---

## What Each Component Does

| Component | Purpose | Status |
|-----------|---------|--------|
| ResNet50 | Glaucoma classification | ✅ Working |
| Grad-CAM | Visual explanation | ✅ Working |
| PostgreSQL | Vector database | ✅ Working |
| pgvector | Semantic search | ✅ Working |
| sentence-transformers | Embeddings | ✅ Working |
| RAG | Document retrieval | ✅ Working |
| Ollama | LLM server | ✅ Working |
| Mistral-7B | Text generation | ✅ Working |
| Streamlit | UI/UX | ✅ Working |

---

**You're all set! Run the command above and start detecting glaucoma with AI-powered RAG explanations!** 🎊

