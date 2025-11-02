# 🚀 Launch Your App with Groq

## Quick Start

### 1️⃣ Set Up API Key

```powershell
# Create secrets file
copy .streamlit\secrets.toml.template .streamlit\secrets.toml

# Edit .streamlit/secrets.toml and add your Groq API key
notepad .streamlit\secrets.toml
```

### 2️⃣ Get Free API Key

Visit: **https://console.groq.com/**

1. Sign up (free account)
2. Go to "API Keys"
3. Create a new key
4. Copy the key (starts with `gsk_...`)
5. Paste in `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

### 3️⃣ Launch App

```powershell
python -m streamlit run streamlit_app/app.py
```

Browser opens at: **http://localhost:8501**

---

## ✅ Verify Setup

### Sidebar Should Show:
- ✅ [OK] Model loaded successfully
- ✅ [OK] RAG system ready
- ✅ [OK] Groq + Llama3 ready

If you see "Groq API not configured", go back to Step 1!

---

## 🎯 Test Your App

1. **Upload Image**: Choose from test_set folder
2. **Click Predict**: Wait ~3-5 seconds
3. **See Results**: Prediction, Grad-CAM, RAG sources
4. **Generate AI Description**: Toggle "Generate detailed information"
5. **Clear Data**: Use 🗑️ button if needed

---

## 🎊 Success!

Your complete pipeline is working:
- ResNet50 → Grad-CAM → RAG → Llama3 (Groq) → UI

**Enjoy your AI-powered glaucoma detection system!** 🎉

