# 🧪 Manual Testing Guide

## Step-by-Step Instructions

### Step 1: Set Up Environment
```powershell
# Navigate to project directory
cd C:\Users\hp\Documents\Renuka\Glaucoma_detection

# Set database password
$env:DB_PASSWORD = "5657"
```

### Step 2: Launch the App
```powershell
# Start Streamlit
python -m streamlit run streamlit_app/app.py
```

Wait for:
- "You can now view your Streamlit app in your browser."
- Local URL: http://localhost:8501

### Step 3: Open Browser
- Streamlit will auto-open your browser
- OR manually go to: http://localhost:8501

### Step 4: Test the Features

#### A. Check Sidebar Status
Look for:
- ✅ Model loaded successfully
- ✅ RAG system ready
- ✅ Ollama + Mistral-7B ready

#### B. Upload a Test Image
1. Click "Choose a retinal fundus image"
2. Select an image from:
   - `RIM-ONE_DL_images/partitioned_randomly/test_set/glaucoma/`
   - `RIM-ONE_DL_images/partitioned_randomly/test_set/normal/`

#### C. Click Predict
- Button: "🔍 Predict"
- Wait ~3-5 seconds
- Watch for processing indicators

#### D. Check Results
Look for:
1. **Prediction**: Glaucoma/Normal with confidence
2. **Grad-CAM**: Heatmap visualization (toggle on/off)
3. **AI Description**: Detailed patient information (check box)
4. **Source Documents**: Expandable section at bottom

#### E. Test Download
- Click "💾 Download Prediction Report"
- Verify the downloaded file

---

## Expected Results

### For Glaucoma Image
- Prediction: "⚠️ Glaucoma Detected" (high probability)
- Grad-CAM: Red/yellow heatmap on optic disc region
- RAG: Retrieved glaucoma-related documents
- Mistral: Detailed description about glaucoma causes, consequences, etc.

### For Normal Image
- Prediction: "✅ Normal" (low glaucoma probability)
- Grad-CAM: Less intense heatmap
- RAG: Retrieved normal/healthy eye documents
- Mistral: Reassuring description about eye health

---

## Troubleshooting

### If RAG shows "unavailable":
```powershell
# Verify database
psql -U postgres -d glaucoma_rag -c "SELECT COUNT(*) FROM rag_chunks;"
# Should show: 13

# If 0, regenerate:
python scripts/generate_embeddings_simple.py
```

### If Ollama shows "not running":
```powershell
# Test Ollama
ollama list
# Should show: mistral:7b

# If missing, pull model:
ollama pull mistral:7b
```

### If model not loading:
```powershell
# Check model file exists
ls Glaucoma_detection\Glaucoma_detection\models\resnet50_finetuned.best.h5
```

### If slow first time:
- This is NORMAL! Models load on first run
- Takes ~30 seconds first time
- Subsequent runs are faster

---

## What to Look For

### ✅ Success Indicators
- All three checkmarks in sidebar (green)
- Image displays after upload
- Prediction appears quickly
- Grad-CAM visualization works
- AI description is coherent
- Source documents are relevant

### ⚠️ If Something Fails
- Check browser console for errors
- Look at terminal output for Python errors
- Verify all services are running
- See troubleshooting section above

---

## Test Multiple Images

Try at least 3-4 images:
1. Clear glaucoma case
2. Clear normal case
3. Borderline/uncertain case
4. Different image types

Observe:
- Consistency of predictions
- Quality of descriptions
- Relevance of RAG sources
- Grad-CAM patterns

---

## Record Observations

Note:
- ✅ What works well
- ⚠️ What needs improvement
- 🐛 Any bugs or errors
- 💡 Ideas for enhancements

---

**Happy testing!** 🎉

Take your time and explore all features!

