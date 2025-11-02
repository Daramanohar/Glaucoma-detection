# 👁️ Glaucoma Detection System

AI-powered glaucoma detection with RAG and Llama3 integration

## 🔗 Repository

**GitHub**: https://github.com/Daramanohar/Glaucoma-detection.git

## 🚀 Quick Start

### Local Development

1. **Clone Repository**
   ```bash
   git clone https://github.com/Daramanohar/Glaucoma-detection.git
   cd Glaucoma-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Database** (PostgreSQL + pgvector)
   - See `WINDOWS_POSTGRES_SETUP.md`

4. **Configure Groq API**
   - Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

5. **Run App**
   ```bash
   streamlit run streamlit_app/app.py
   ```

### Streamlit Cloud Deployment

1. **Deploy**: https://share.streamlit.io/
2. **Add Secrets**: Groq API key in Streamlit Cloud secrets
3. **Done!**: App live at `https://your-app.streamlit.app`

See `STREAMLIT_CLOUD_DEPLOYMENT.md` for detailed instructions.

## 📚 Features

- ✅ ResNet50 glaucoma detection (~90% accuracy)
- ✅ Grad-CAM visualization
- ✅ RAG document retrieval
- ✅ Llama3 AI-generated descriptions
- ✅ Clear data button
- ✅ Beautiful Streamlit UI

## 📖 Documentation

- `START_HERE.md` - Quick start guide
- `GROQ_SETUP.md` - API configuration
- `STREAMLIT_CLOUD_DEPLOYMENT.md` - Cloud deployment
- `WINDOWS_POSTGRES_SETUP.md` - Database setup

## 🔒 Security

- API keys stored in Streamlit secrets
- `.gitignore` properly configured
- No secrets in code

## 📝 License

See LICENSE file

