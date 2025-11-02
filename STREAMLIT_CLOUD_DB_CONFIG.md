# 🔧 Database Configuration for Streamlit Cloud

## Overview

For full functionality, you can add a PostgreSQL database to your Streamlit Cloud deployment.

## Option 1: Simple (No Database) - Recommended for Start

**What works without DB:**
- ✅ ResNet50 prediction
- ✅ Grad-CAM visualization
- ✅ Groq + Llama3 AI descriptions

**What doesn't work without DB:**
- ❌ RAG document retrieval
- ❌ Source citations

**For Most Users:** You can deploy without a database and everything except RAG will work!

---

## Option 2: Full Setup (With Database)

If you want RAG functionality, add database details to Streamlit Cloud secrets:

### Streamlit Cloud Secrets Configuration

Go to your app in Streamlit Cloud Dashboard → Settings → Secrets → Edit

Add:

```toml
# Groq API (REQUIRED)
GROQ_API_KEY = "gsk_your_actual_api_key"

# Database Configuration (OPTIONAL - only if you want RAG)
DB_HOST = "your_database_host"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "your_database_password"
DB_NAME = "glaucoma_rag"
```

---

## 🌐 Free Database Options for Streamlit Cloud

### Option A: Supabase (Recommended)
- **Free tier**: 500MB PostgreSQL
- **Setup**: 
  1. Go to https://supabase.com/
  2. Create free project
  3. Get connection details
  4. Add to Streamlit Cloud secrets

### Option B: Neon
- **Free tier**: 3GB PostgreSQL
- **Setup**: 
  1. Go to https://neon.tech/
  2. Create free project
  3. Get connection details
  4. Add to Streamlit Cloud secrets

### Option C: Railway
- **Free tier**: Limited PostgreSQL
- **Setup**: 
  1. Go to https://railway.app/
  2. Create database
  3. Get connection details
  4. Add to Streamlit Cloud secrets

---

## 📝 Minimal Secrets for Basic Deployment

**For deployment without RAG:**

```toml
GROQ_API_KEY = "gsk_your_actual_api_key"
```

That's it! Your app will work perfectly for predictions and AI descriptions.

---

## 🔧 Setting Up Database on Cloud

### Step 1: Create Cloud Database

Choose one of the free providers above and create a PostgreSQL database.

### Step 2: Get Connection Details

You'll receive:
- Host: `xxx.supabase.co` or similar
- Port: `5432`
- Database: `postgres` or custom name
- Username: Provided
- Password: Provided

### Step 3: Set Up Database Schema

Once you have database access:

1. **Enable pgvector extension**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Create schema**
   - Copy contents of `rag_data/pgvector_schema.sql`
   - Run in your cloud database

3. **Load data**
   - Run: `python scripts/setup_postgres_vector_db.py`
   - Run: `python scripts/generate_embeddings_simple.py`
   - (Update DB_CONFIG in scripts to use cloud credentials first)

### Step 4: Add to Streamlit Cloud Secrets

```toml
GROQ_API_KEY = "gsk_your_groq_key"
DB_HOST = "xxx.supabase.co"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "your_password"
DB_NAME = "glaucoma_rag"
```

---

## ⚡ Recommended: Start Simple

**Deploy with Groq only first**, then add database later if needed.

Your app works great without the database! RAG is a nice-to-have feature.

---

## 📊 What You Get

### Without Database
- ✅ Glaucoma detection
- ✅ Grad-CAM visualization
- ✅ AI-generated descriptions
- ✅ Clear data button
- ✅ Full user interface

### With Database (RAG)
- ✅ Everything above
- ✅ Plus: RAG document retrieval
- ✅ Plus: Source citations
- ✅ Plus: Evidence-based responses

---

## 🎯 Deployment Priority

**High Priority:**
1. Groq API key ⭐⭐⭐

**Optional:**
2. Database configuration ⭐

---

**Start with just the Groq API key! You can add the database later.** 🚀

