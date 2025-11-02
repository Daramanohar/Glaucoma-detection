"""
Standalone Directory Setup Script for Colab
Copy and paste this entire script into a Colab cell and run it
"""

import os
from pathlib import Path

# Get base directory - works in Colab
def get_base_dir():
    """Get the base directory"""
    cwd = Path(os.getcwd())
    
    # Check if we're in the project root
    if (cwd / "scripts").exists() and (cwd / "RIM-ONE_DL_images").exists():
        return cwd
    
    # Check if we're one level up
    if (cwd.parent / "scripts").exists() and (cwd.parent / "RIM-ONE_DL_images").exists():
        return cwd.parent
    
    # Try common Colab paths
    colab_paths = [
        Path("/content/Glaucoma_detection/Glaucoma_detection"),
        Path("/content/Glaucoma_detection"),
        Path("/content/drive/MyDrive/Glaucoma_detection"),
    ]
    for path in colab_paths:
        if path.exists():
            if (path / "scripts").exists() and (path / "RIM-ONE_DL_images").exists():
                return path
    
    return cwd

# Get base directory
BASE_DIR = get_base_dir()
print(f"📁 Project root: {BASE_DIR}")

# Directories to create
DIRECTORIES = [
    "processed_data/train/Glaucoma",
    "processed_data/train/Normal",
    "processed_data/test/Glaucoma",
    "processed_data/test/Normal",
    "models",
    "results/gradcam_samples",
    "logs/tensorboard",
    "streamlit_app",
    "report",
    "notebooks",
    "config",
    "temp"
]

print("\n🔧 Creating directory structure...")
print("="*60)

# Create directories
for dir_path in DIRECTORIES:
    full_path = BASE_DIR / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ {dir_path}")

# Create __init__.py files for Python packages
print("\n📝 Creating package files...")
init_files = [
    "scripts/__init__.py",
    "streamlit_app/__init__.py"
]

for init_file in init_files:
    init_path = BASE_DIR / init_file
    if not init_path.exists():
        init_path.touch()
        print(f"✓ {init_file}")

print("\n" + "="*60)
print("✅ Directory structure created successfully!")
print(f"\n📂 Project root: {BASE_DIR}")
print("\nNext steps:")
print("  1. !python scripts/prepare_data.py")
print("  2. !python scripts/train_resnet50.py")

