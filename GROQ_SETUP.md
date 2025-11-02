# 🔧 Groq API Setup Guide

## Overview

This application now uses **Groq API with Llama3-70B** instead of Ollama for AI-generated descriptions.

---

## 🎯 Quick Setup

### Step 1: Get Your Groq API Key

1. Go to https://console.groq.com/
2. Sign up or log in (free account available)
3. Navigate to **API Keys** section
4. Click **"Create API Key"**
5. Copy your API key (starts with `gsk_...`)

### Step 2: Create Secrets File

1. Copy the template file:
   ```powershell
   copy .streamlit\secrets.toml.template .streamlit\secrets.toml
   ```

2. Open `.streamlit/secrets.toml` in your editor

3. Replace `your_groq_api_key_here` with your actual API key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_api_key_here"
   ```

4. Save the file

### Step 3: Launch the App

```powershell
python -m streamlit run streamlit_app/app.py
```

---

## ✅ Verification

### Check If Setup Works

When you launch the app, check the sidebar:
- ✅ **[OK] Groq + Llama3 ready** = Working!
- ⚠️ **[WARNING] Groq API not configured** = Need to set up API key

### Test the Integration

1. Upload a test image
2. Click "Predict"
3. Toggle "Generate detailed information with AI"
4. You should see a detailed AI-generated description

---

## 🚨 Troubleshooting

### Error: "Groq API key not found"

**Solution**: 
- Check that `.streamlit/secrets.toml` exists
- Verify the file name is correct (not `.toml.template`)
- Make sure `GROQ_API_KEY = "your_key"` is properly set

### Error: "Error generating description"

**Possible Causes**:
- Invalid API key
- Network connectivity issues
- API quota exceeded

**Solution**:
- Verify API key at https://console.groq.com/
- Check your internet connection
- Verify API usage limits in Groq dashboard

### WARNING: Groq API not configured

**Solution**: 
The secrets file is missing or misconfigured. Follow Step 2 above.

---

## 🔒 Security Notes

### API Key Safety

- ✅ `.streamlit/secrets.toml` is in `.gitignore` (won't be committed)
- ✅ Never share your API key publicly
- ✅ Don't commit `secrets.toml` to version control
- ✅ Each developer needs their own API key

### File Structure

```
.streamlit/
  ├── secrets.toml              # ⚠️ DO NOT COMMIT (your API key)
  └── secrets.toml.template     # ✅ Safe to commit (template only)

.gitignore includes: .streamlit/secrets.toml
```

---

## 🎁 Groq Free Tier

Groq offers generous free tier:
- ✅ **Free API access**
- ✅ Fast inference with Llama3-70B
- ✅ No credit card required to start
- ✅ Check limits at https://console.groq.com/

---

## 📊 Available Models

The app uses `llama3-70b-8192` by default. You can change this in `scripts/groq_interface.py`:

```python
GroqInterface(model="llama3-8b-8192")  # Faster, smaller
GroqInterface(model="llama3-70b-8192") # More powerful (default)
```

---

## ✅ All Set!

Once your API key is configured, you're ready to use the complete pipeline:

**ResNet50 → Grad-CAM → RAG → Llama3 (Groq) → UI**

Enjoy your AI-powered glaucoma detection system! 🎉

