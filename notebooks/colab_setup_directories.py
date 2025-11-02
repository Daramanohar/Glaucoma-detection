"""
Quick setup for Google Colab
Run this first to ensure you're in the correct directory
"""

import os
from pathlib import Path

# Set your base directory here if needed
BASE_DIR = "/content/Glaucoma_detection/Glaucoma_detection"

print("Setting up directories...")
print(f"Target directory: {BASE_DIR}")

# Change to the project directory
if os.path.exists(BASE_DIR):
    os.chdir(BASE_DIR)
    print(f"✓ Changed to: {os.getcwd()}")
else:
    print(f"⚠️ Directory {BASE_DIR} doesn't exist!")
    print(f"Current directory: {os.getcwd()}")
    print("\nPlease adjust BASE_DIR variable above or:")
    print(f"1. Upload your project to {BASE_DIR}")
    print("2. Or update BASE_DIR to match your project location")

# Verify we're in the right place
cwd = Path(os.getcwd())
if (cwd / "scripts").exists() and (cwd / "RIM-ONE_DL_images").exists():
    print(f"\n✅ Success! Project root detected: {cwd}")
    print("\nNow you can run:")
    print("  !python scripts/setup_directories.py")
    print("  !python scripts/prepare_data.py")
else:
    print(f"\n⚠️ Could not find project structure in {cwd}")
    print("\nPlease ensure:")
    print("  1. You've uploaded/extracted the project")
    print("  2. You're in the directory containing 'scripts' and 'RIM-ONE_DL_images' folders")
    print(f"\nCurrent directory contents: {list(cwd.iterdir())[:10]}")

