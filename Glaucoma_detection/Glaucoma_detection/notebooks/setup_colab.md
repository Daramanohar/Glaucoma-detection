# Google Colab Setup Guide

## Quick Start for Training in Colab

Follow these steps to set up and train the model in Google Colab:

### Step 1: Upload Project to Colab

1. Zip your project folder
2. Upload to Google Drive or directly to Colab
3. Or clone from GitHub if you've pushed it there

### Step 2: Install Dependencies

```python
# Run this in a Colab cell
!pip install tensorflow>=2.13.0
!pip install pillow opencv-python scikit-learn matplotlib seaborn
!pip install imgaug streamlit tqdm pyyaml pandas
```

### Step 3: Mount Google Drive (if using Drive)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Set Working Directory

```python
import os
os.chdir('/content/drive/MyDrive/Glaucoma_detection')  # Adjust path as needed
# Or if uploaded directly:
# os.chdir('/content/Glaucoma_detection')
```

### Step 5: Enable GPU

Go to: Runtime → Change runtime type → GPU (T4/P100/A100)

### Step 6: Check GPU

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version: ", tf.__version__)
```

### Step 7: Run Training Pipeline

```python
# Step 1: Prepare data
!python scripts/prepare_data.py

# Step 2: Train model
!python scripts/train_resnet50.py

# Step 3: Evaluate
!python scripts/evaluate.py

# Step 4: Generate Grad-CAM
!python scripts/gradcam.py
```

### Step 8: Download Results

```python
# Download models and results
from google.colab import files

# Download best model
files.download('models/resnet50_finetuned.best.h5')

# Download results (zip folder)
!zip -r results.zip results/
files.download('results.zip')
```

## Tips for Colab

1. **Memory Management**: If you run out of RAM, reduce batch size in `train_resnet50.py`
2. **Session Timeout**: Colab sessions timeout after ~90 minutes. Use checkpoints!
3. **Download Regularly**: Save models/results periodically
4. **TensorBoard**: Use Colab's built-in TensorBoard integration:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir logs/tensorboard
   ```

## Alternative: Direct Python Script Execution

You can also run everything in separate Colab cells for better debugging:

```python
# Cell 1: Setup
import sys
sys.path.append('/content/drive/MyDrive/Glaucoma_detection')

# Cell 2: Prepare data
exec(open('scripts/prepare_data.py').read())

# Cell 3: Train
exec(open('scripts/train_resnet50.py').read())

# Cell 4: Evaluate
exec(open('scripts/evaluate.py').read())
```

## Troubleshooting

- **Import Errors**: Make sure all dependencies are installed
- **GPU Not Detected**: Restart runtime and check GPU allocation
- **Out of Memory**: Reduce batch size or image size
- **File Not Found**: Check working directory paths

