# 🚀 Quick GitHub + Streamlit Cloud Setup

## ⚡ Fast Track

### 1. Push to GitHub

```powershell
# Navigate to project
cd C:\Users\hp\Documents\Renuka\Glaucoma_detection

# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Glaucoma detection app with RAG and Groq"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/glaucoma-detection.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Main file: `streamlit_app/app.py`
6. Click **"Deploy"**

### 3. Add API Key

1. In Streamlit Cloud dashboard → Your app → Settings → Secrets
2. Add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_api_key_here"
   ```
3. Click **"Save"**

### 4. Done!

Your app will deploy automatically. URL: `https://YOUR_APP_NAME.streamlit.app`

---

## ✅ Checklist

- [ ] Git initialized
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed on Streamlit Cloud
- [ ] Groq API key added to Streamlit Cloud secrets
- [ ] App tested and working

---

## 🔒 Security Notes

✅ **Never commit** `.streamlit/secrets.toml` (already in .gitignore)  
✅ **Use Streamlit Cloud secrets** for API keys  
✅ **Your API key is safe** in Streamlit Cloud  

---

## 📝 Next Steps

After deployment:

1. Share your app URL
2. Test all features
3. Monitor usage
4. Update as needed (auto-deploys on push!)

---

**That's it! Your app is now live on Streamlit Cloud!** 🎉

