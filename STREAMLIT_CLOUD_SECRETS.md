# 🔐 Streamlit Cloud Secrets Configuration

## Quick Answer

**For basic deployment, you only need:**

```toml
GROQ_API_KEY = "your_groq_api_key"
```

That's it! Your app will work perfectly.

---

## 📋 Secrets Options

### Option 1: Minimal (Recommended for Start)

```toml
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

**What works:**
- ✅ Glaucoma detection
- ✅ Grad-CAM visualization
- ✅ AI-generated descriptions
- ✅ Clear data button

**What's missing:**
- ❌ RAG document retrieval

### Option 2: Full (With RAG)

```toml
# Required
GROQ_API_KEY = "gsk_your_actual_api_key_here"

# Optional - only if you want RAG
DB_HOST = "your_database_host"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "your_database_password"
DB_NAME = "glaucoma_rag"
```

**What works:**
- ✅ Everything above
- ✅ Plus RAG document retrieval
- ✅ Plus source citations

---

## 🚀 How to Add Secrets in Streamlit Cloud

### Step-by-Step

1. **Go to Streamlit Cloud Dashboard**: https://share.streamlit.io/
2. **Find your app**
3. **Click Settings** (⚙️ icon)
4. **Click "Secrets"** section
5. **Click "Edit secrets"**
6. **Paste** your secrets (see format above)
7. **Click "Save"**
8. **App auto-redeploys!**

---

## 💡 Recommendation

**Start with Option 1 (Groq only)**

You can add the database configuration later if you want RAG functionality.

---

## 📝 Where to Get Values

### Groq API Key
1. Visit: https://console.groq.com/
2. Sign up/login
3. Go to "API Keys"
4. Create new key
5. Copy the key

### Database Details (Optional)
If you want RAG, use one of these free providers:
- **Supabase**: https://supabase.com/
- **Neon**: https://neon.tech/
- **Railway**: https://railway.app/

See `STREAMLIT_CLOUD_DB_CONFIG.md` for detailed database setup.

---

## ✅ Verification

After adding secrets, check:
- Sidebar shows: ✅ Groq + Llama3 ready
- Upload test image works
- AI description generates

---

**Start simple with just Groq API key!** 🎉

