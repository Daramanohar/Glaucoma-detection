# Google Colab Setup Guide

## Quick Start

### Step 1: Upload Your Project

Upload your project folder to Colab. Your base directory is:
```
/content/Glaucoma_detection/Glaucoma_detection
```

### Step 2: Run This in Colab

```python
# Cell 1: Navigate to your project
%cd /content/Glaucoma_detection/Glaucoma_detection

# Verify you're in the right place
import os
from pathlib import Path
cwd = Path(os.getcwd())
print(f"Current directory: {cwd}")
print(f"Has scripts folder: {(cwd / 'scripts').exists()}")
print(f"Has RIM-ONE_DL_images: {(cwd / 'RIM-ONE_DL_images').exists()}")
```

### Step 3: Install Requirements

```python
# Cell 2: Install dependencies
!pip install -r requirements.txt
```

### Step 4: Run Setup Scripts

```python
# Cell 3: Create directories
!python scripts/setup_directories.py

# Cell 4: Prepare data
!python scripts/prepare_data.py

# Cell 5: Train model (this takes time)
!python scripts/train_resnet50.py
```

## Important Notes

### Directory Structure

Make sure your Colab directory looks like this:
```
/content/Glaucoma_detection/Glaucoma_detection/
├── scripts/
├── RIM-ONE_DL_images/
├── requirements.txt
└── ...
```

### If Scripts Can't Find Base Directory

If you get path errors, run this before other scripts:

```python
import os
os.chdir('/content/Glaucoma_detection/Glaucoma_detection')
print(f"Now in: {os.getcwd()}")
```

### Manual Base Directory Override

If the auto-detection doesn't work, you can manually set it in the scripts. Edit `scripts/utils.py`:

```python
def get_base_dir():
    # Force specific path for Colab
    return Path("/content/Glaucoma_detection/Glaucoma_detection")
```

## Troubleshooting

**Error: `__file__` not defined**
- ✅ Fixed! All scripts now work in Colab
- Just make sure you're in the correct directory first

**Error: Module not found**
- Run: `!pip install -r requirements.txt`

**Error: Can't find data**
- Verify: `!ls RIM-ONE_DL_images/`

**Error: Out of memory**
- Reduce batch size in `scripts/train_resnet50.py` (change `BATCH_SIZE = 32` to `16`)

