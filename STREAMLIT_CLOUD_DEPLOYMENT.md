# 🚀 Deploy to Streamlit Cloud

## Overview

Deploy your glaucoma detection app to Streamlit Cloud with Groq API integration!

---

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Streamlit Cloud account (free)
- ✅ Groq API key
- ✅ Git installed locally

---

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your Repository

#### 1.1 Initialize Git (if not already done)
```powershell
cd C:\Users\hp\Documents\Renuka\Glaucoma_detection
git init
git add .
git commit -m "Initial commit: Glaucoma detection with RAG and Groq"
```

#### 1.2 Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository:
   - Name: `glaucoma-detection` (or your choice)
   - Description: "AI-powered glaucoma detection with RAG and Llama3"
   - Choose: **Public** or **Private**
   - **Don't** initialize with README (we already have files)

#### 1.3 Push to GitHub

```powershell
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/glaucoma-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

### Step 2: Deploy to Streamlit Cloud

#### 2.1 Sign In to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Authorize Streamlit Cloud to access your repositories

#### 2.2 Deploy New App

1. Click **"New app"**
2. Fill in details:
   - **Repository**: Select your `glaucoma-detection` repo
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/app.py`
   - **App URL**: `glaucoma-detection` (or your choice)
   - Click **"Deploy"**

#### 2.3 Configure Secrets

**IMPORTANT**: Add your Groq API key in Streamlit Cloud secrets!

1. In Streamlit Cloud dashboard, click on your app
2. Go to **"Settings"** (⚙️ icon)
3. Click **"Secrets"**
4. Paste your Groq API key:

```toml
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

5. Click **"Save"**

#### 2.4 Advanced Settings (Optional)

If you need to set environment variables:

1. Go to **"Settings"** → **"Secrets"**
2. Add any additional configuration:

```toml
# Database (if using cloud database)
DB_HOST = "your_host"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "your_password"

# Groq API
GROQ_API_KEY = "gsk_your_key"
```

---

## ✅ Verify Deployment

### Check Your App

Once deployed, Streamlit Cloud will provide:
- **Public URL**: `https://YOUR_APP_NAME.streamlit.app`
- **Status**: Shows build progress
- **Logs**: View deployment logs

### Test Features

1. **Open your app URL**
2. **Check sidebar**: Should show:
   - ✅ Model loaded successfully
   - ✅ RAG system ready (if database configured)
   - ✅ Groq + Llama3 ready

3. **Test functionality**:
   - Upload test image
   - Get prediction
   - View Grad-CAM
   - Generate AI description

---

## 🔒 Security Notes

### What's Protected

✅ **`.streamlit/secrets.toml`** - Excluded from Git  
✅ **API keys** - Only in Streamlit Cloud secrets  
✅ **Database passwords** - Stored securely  

### What's Public

⚠️ Your code is public (if repository is public)  
⚠️ Model files (if not in .gitignore)  
⚠️ Documentation and setup guides  

---

## 🔧 Configuration Files

### Repository Structure

```
glaucoma-detection/
├── .gitignore              ✅ Secrets excluded
├── .streamlit/
│   └── secrets.toml.template  ✅ Safe to commit
├── streamlit_app/
│   └── app.py              ✅ Main app file
├── scripts/                ✅ All scripts
├── requirements.txt        ✅ Dependencies
└── README.md               ✅ Documentation
```

### What Gets Deployed

- ✅ All Python code
- ✅ Model files (if included)
- ✅ Configuration files
- ✅ Documentation

### What Doesn't Get Deployed

- ❌ `.streamlit/secrets.toml` (local only)
- ❌ Temporary files
- ❌ Local database connections
- ❌ Environment-specific paths

---

## 🌐 Public vs Private

### Public Repository

**Pros:**
- ✅ Easy sharing
- ✅ Community contribution
- ✅ Portfolio showcase

**Cons:**
- ⚠️ Code is visible
- ⚠️ Anyone can see implementation

### Private Repository

**Pros:**
- ✅ Code privacy
- ✅ Control access

**Cons:**
- 💰 Limited free tier
- ⚠️ Harder to share

**Recommendation**: Start with **Public** for free tier!

---

## 📊 Streamlit Cloud Limits

### Free Tier

- ✅ Unlimited apps
- ✅ Free hosting
- ✅ Auto-updates on push
- ⚠️ Apps sleep after 1 week of inactivity
- ⚠️ Resumes automatically when accessed

### Team Tier (Paid)

- ✅ Always-on apps
- ✅ More resources
- ✅ Priority support

---

## 🔄 Auto-Deployment

### How It Works

1. **Push to GitHub** → Streamlit Cloud detects changes
2. **Automatic rebuild** → App updates automatically
3. **Zero downtime** → Users see latest version

### Update Your App

```powershell
# Make changes locally
# Commit and push
git add .
git commit -m "Updated features"
git push origin main

# Streamlit Cloud automatically deploys!
```

---

## 🐛 Troubleshooting

### Build Fails

**Check logs**:
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. View **"Logs"** tab
4. Look for error messages

**Common issues**:
- Missing dependencies in `requirements.txt`
- Import errors
- Path issues
- Missing files

### API Key Not Working

**Solution**:
1. Verify secrets are set correctly
2. Check format: `GROQ_API_KEY = "gsk_..."`
3. Ensure no extra spaces or quotes
4. Redeploy after fixing

### Database Connection Issues

**If using PostgreSQL**:
- Use cloud database (e.g., Supabase, Neon)
- Add connection strings to Streamlit Cloud secrets
- Update `DB_HOST`, `DB_PORT`, etc.

---

## 📝 Best Practices

### 1. Keep Secrets Secret

✅ Use Streamlit Cloud secrets  
✅ Never commit API keys  
✅ Use `.gitignore` properly  

### 2. Document Your App

✅ Update README.md  
✅ Add usage instructions  
✅ Document API requirements  

### 3. Test Locally First

✅ Test before pushing  
✅ Fix errors locally  
✅ Verify secrets work  

### 4. Monitor Usage

✅ Check Groq API usage  
✅ Monitor Streamlit Cloud quotas  
✅ Set up alerts if needed  

---

## 🎉 Success!

Once deployed, your app will be:
- ✅ Publicly accessible
- ✅ Auto-updating
- ✅ Secure (API keys in secrets)
- ✅ Scalable (cloud-hosted)

**Your app URL**: `https://YOUR_APP_NAME.streamlit.app`

---

## 📚 Additional Resources

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Groq API Docs: https://console.groq.com/docs
- GitHub Guide: https://guides.github.com/

---

**Ready to deploy? Follow the steps above!** 🚀

