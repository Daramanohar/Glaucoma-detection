# Quick Start Guide

## 🚀 Getting Started in 5 Steps

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create directory structure
python scripts/setup_directories.py
```

### Step 2: Prepare Data

```bash
# Merge, preprocess, and augment images
python scripts/prepare_data.py
```

**Output**: Processed images in `processed_data/` with train/test splits

### Step 3: Train Model (Recommended in Google Colab with GPU)

```bash
# Train ResNet50 with transfer learning
python scripts/train_resnet50.py
```

**Output**: 
- Best model: `models/resnet50_finetuned.best.h5`
- Training logs: `logs/`

**Training Time**: ~30-60 minutes on Colab GPU (T4/P100)

### Step 4: Evaluate Model

```bash
# Evaluate on test set
python scripts/evaluate.py
```

**Output**: 
- Metrics: `results/metrics.json`
- Plots: `results/confusion_matrix.png`, `results/roc_auc.png`
- Predictions: `results/predictions.csv`

### Step 5: Generate Grad-CAM & Run Streamlit

```bash
# Generate explainability visualizations
python scripts/gradcam.py

# Launch Streamlit app
streamlit run streamlit_app/app.py
```

## 📋 Complete Pipeline (One Command per Step)

```bash
# 1. Setup
pip install -r requirements.txt && python scripts/setup_directories.py

# 2. Prepare data
python scripts/prepare_data.py

# 3. Train (run in Colab for GPU)
python scripts/train_resnet50.py

# 4. Evaluate
python scripts/evaluate.py

# 5. Grad-CAM
python scripts/gradcam.py

# 6. Streamlit app
streamlit run streamlit_app/app.py
```

## 🔧 Google Colab Setup

1. **Upload Project**: Zip and upload to Colab or clone from GitHub
2. **Install Dependencies**: 
   ```python
   !pip install -r requirements.txt
   ```
3. **Enable GPU**: Runtime → Change runtime type → GPU
4. **Run Scripts**: Execute the pipeline steps above

See `notebooks/setup_colab.md` for detailed Colab instructions.

## 📊 Expected Results

After training, you should see:
- **Model saved**: `models/resnet50_finetuned.best.h5`
- **Training curves**: `results/accuracy_loss.png`
- **Metrics**: Accuracy ~85-92%, AUC ~0.90-0.95
- **Grad-CAM samples**: `results/gradcam_samples/`

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: 
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **GPU Not Detected (Colab)**:
   - Runtime → Change runtime type → GPU
   - Restart runtime

3. **Out of Memory**:
   - Reduce batch size in `scripts/train_resnet50.py` (change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`)

4. **File Not Found**:
   - Ensure you're in the project root directory
   - Run `python scripts/setup_directories.py` first

## 📁 File Organization

All outputs are automatically saved to:
- `models/` - Trained models
- `results/` - Evaluation results and plots
- `logs/` - Training logs and TensorBoard
- `processed_data/` - Preprocessed images

## 🔍 Next Steps

- Check `README.md` for detailed documentation
- Review `report/project_report.md` for methodology
- Explore `config/config.yaml` for hyperparameter tuning

---

**Need Help?** Review the main `README.md` for comprehensive documentation.

