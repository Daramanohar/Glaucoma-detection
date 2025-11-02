"""
Utility script to create all necessary directories for the project
Run this first before running other scripts
"""

import os
import sys
from pathlib import Path

# Import utility function for base directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_base_dir

BASE_DIR = get_base_dir()

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

def create_directories():
    """Create all necessary directories"""
    print("Creating directory structure...")
    
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {full_path}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "scripts/__init__.py",
        "streamlit_app/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = BASE_DIR / init_file
        if not init_path.exists():
            init_path.touch()
            print(f"✓ Created: {init_path}")
    
    print("\n✓ Directory structure created successfully!")
    print(f"\nProject root: {BASE_DIR}")


if __name__ == "__main__":
    create_directories()

