# Glaucoma Detection Using Deep Learning (ResNet50)

An end-to-end pipeline for detecting glaucoma from retinal fundus images using transfer learning with ResNet50. This project implements data preprocessing, model training, evaluation, explainability (Grad-CAM), and a Streamlit web interface.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Safety & Ethics](#safety--ethics)
- [Citations & License](#citations--license)

## 🎯 Project Overview

This project implements a binary classification system to detect glaucoma from retinal fundus images using:
- **Transfer Learning**: Pre-trained ResNet50 on ImageNet
- **Robust Training**: Two-phase fine-tuning strategy with regularization
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Explainability**: Grad-CAM visualizations
- **Web Interface**: Interactive Streamlit application

## 📊 Dataset

The project uses the **RIM-ONE DL** dataset, which contains retinal fundus images partitioned in two ways:
1. **Partitioned by Hospital**: Training and test sets divided by hospital source
2. **Partitioned Randomly**: Training and test sets randomly divided

**Data Sources**:
- `RIM-ONE_DL_images/partitioned_by_hospital/`
- `RIM-ONE_DL_images/partitioned_randomly/`

**Classes**: Glaucoma (positive) and Normal (negative)

## 📁 Directory Structure

```
project_root/
├── RIM-ONE_DL_images/          # Raw dataset
│   ├── partitioned_by_hospital/
│   └── partitioned_randomly/
├── processed_data/              # Preprocessed images
│   ├── train/Glaucoma, Normal/
│   └── test/Glaucoma, Normal/
├── models/                      # Saved models
│   ├── resnet50_finetuned.best.h5
│   └── resnet50_finetuned.final.h5
├── results/                     # Evaluation results
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_auc.png
│   ├── accuracy_loss.png
│   ├── predictions.csv
│   ├── classification_report.txt
│   └── gradcam_samples/
├── logs/                        # Training logs
│   ├── tensorboard/
│   ├── training_history.json
│   └── training_config.json
├── scripts/                     # Python scripts
│   ├── prepare_data.py
│   ├── train_resnet50.py
│   ├── evaluate.py
│   └── gradcam.py
├── streamlit_app/               # Web application
│   └── app.py
├── report/                      # Project documentation
│   └── project_report.md
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- GPU (recommended for training) - Google Colab T4/P100/A100 compatible
- CUDA (for local GPU training)

### Setup

1. **Clone the repository** (or download the project):
```bash
git clone <repository-url>
cd Glaucoma_detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **For Google Colab**:
   - Upload the project folder to Google Drive or Colab
   - Install dependencies in a Colab cell:
   ```python
   !pip install -r requirements.txt
   ```

## 📖 Usage

### Step 1: Data Preparation

Merge and preprocess images from both partitions, apply augmentations:

```bash
python scripts/prepare_data.py
```

**Output**:
- Merged training/test sets in `processed_data/`
- Augmented training images
- Data summary in `processed_data/data_summary.json`

**Key Features**:
- Resizes all images to 224×224
- Applies augmentation only to training set (rotation, zoom, flip, brightness, shear, blur)
- Saves processed images with unique filenames

### Step 2: Model Training

Train ResNet50 with transfer learning (optimized for Google Colab):

```bash
python scripts/train_resnet50.py
```

**Training Strategy**:
- **Phase A**: Train head only (base frozen) for 10 epochs
- **Phase B**: Fine-tune last 50 layers for 20 epochs
- Uses mixed precision for faster training
- Includes callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

**Hyperparameters**:
- Batch size: 32
- Initial LR: 1e-4 (Adam)
- Fine-tune LR: 1e-5
- Weight decay: 1e-4
- Dropout: 0.5, 0.3

**Output**:
- Best model: `models/resnet50_finetuned.best.h5`
- Final model: `models/resnet50_finetuned.final.h5`
- Training history: `logs/training_history.json`
- Config: `logs/training_config.json`
- TensorBoard logs: `logs/tensorboard/`

### Step 3: Evaluation

Evaluate the model on the test set:

```bash
python scripts/evaluate.py
```

**Output**:
- `results/metrics.json`: All evaluation metrics
- `results/confusion_matrix.png`: Confusion matrix visualization
- `results/roc_auc.png`: ROC curve
- `results/predictions.csv`: Predictions for all test samples
- `results/classification_report.txt`: Detailed classification report
- `results/accuracy_loss.png`: Training curves

### Step 4: Grad-CAM Explainability

Generate Grad-CAM visualizations:

```bash
python scripts/gradcam.py
```

**Output**:
- Visualizations in `results/gradcam_samples/`
- Samples for TP, FP, TN, FN categories

### Step 5: Streamlit App

Run the interactive web application:

```bash
streamlit run streamlit_app/app.py
```

**Features**:
- Image upload and prediction
- Real-time Grad-CAM visualization
- Model performance metrics display
- Confusion matrix and ROC curve visualization

## 🏗️ Model Architecture

```
Input: 224×224×3 RGB image
│
├─ ResNet50 Base (ImageNet weights, frozen initially)
│  └─ Feature extraction
│
└─ Custom Head
   ├─ GlobalAveragePooling2D()
   ├─ Dense(512) + BatchNorm + Dropout(0.5) + L2(1e-4)
   ├─ Dense(128) + BatchNorm + Dropout(0.3) + L2(1e-4)
   └─ Dense(1, sigmoid) → Binary classification
```

**Training Strategy**:
1. Freeze ResNet50, train head only
2. Unfreeze last 50 layers, fine-tune with lower LR
3. Use regularization (L2, Dropout, BatchNorm) to prevent overfitting

## 📈 Results

After training and evaluation, results are saved in `results/`:

- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrix, ROC curve, training curves
- **Predictions**: CSV file with all test predictions
- **Explainability**: Grad-CAM samples showing model attention

## 🌐 Streamlit App

The Streamlit application provides:
- **Image Upload**: Drag-and-drop interface
- **Real-time Prediction**: Instant glaucoma probability
- **Grad-CAM Visualization**: Interactive heatmap overlay
- **Model Metrics**: Performance summary
- **Results Visualization**: Confusion matrix and ROC curve

### Deployment to Streamlit Cloud

1. Push repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect repository
4. Set main file: `streamlit_app/app.py`
5. Deploy!

**Note**: Ensure model files are included in the repository or load from remote storage.

## ⚠️ Safety & Ethics

**⚠️ IMPORTANT DISCLAIMER**:

This model is intended for **research and demonstration purposes only**. 

**For Clinical Use**:
- Requires regulatory approval (FDA, CE marking, etc.)
- Needs rigorous prospective evaluation
- Must address potential biases (demographic, imaging device)
- Should be validated on diverse, representative datasets
- Requires clinical expertise integration

**Limitations**:
- Dataset size may limit generalization
- Potential demographic biases
- Imaging device/vendor biases
- Single-modality (fundus images only)

## 📚 Citations & License

### Dataset
**RIM-ONE DL**: Reference Implementation for Medical Image Open Network (DL version)

Please cite the RIM-ONE dataset appropriately if used in research.

### Code License
This codebase is provided for educational and research purposes.

## 🔮 Future Work

- **Multi-modal Integration**: Incorporate OCT scans, patient metadata, clinical features
- **Domain Adaptation**: Transfer to other retinal imaging devices
- **Additional Datasets**: Combine multiple glaucoma datasets
- **Explainability**: Additional interpretability methods (SHAP, LIME)
- **Real-time Processing**: Optimize for edge deployment

## 📞 Support

For issues, questions, or contributions, please open an issue or contact the maintainers.

## 🙏 Acknowledgments

- RIM-ONE DL dataset creators
- TensorFlow/Keras team
- Streamlit team
- Open-source community

---

**Last Updated**: 2024
**Version**: 1.0

