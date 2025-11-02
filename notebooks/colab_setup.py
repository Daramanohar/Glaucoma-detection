"""
Google Colab Setup Script
Run this cell in Google Colab to set up the environment
"""

# Step 1: Install all dependencies
print("Installing dependencies...")
!pip install tensorflow>=2.13.0
!pip install keras>=2.13.0
!pip install numpy>=1.24.0
!pip install scikit-learn>=1.3.0
!pip install pillow>=10.0.0
!pip install opencv-python>=4.8.0
!pip install scikit-image>=0.21.0
!pip install imgaug>=0.4.0
!pip install matplotlib>=3.7.0
!pip install seaborn>=0.12.0
!pip install streamlit>=1.28.0
!pip install tqdm>=4.66.0
!pip install pyyaml>=6.0
!pip install pandas>=2.0.0

print("\n✓ All dependencies installed!")

# Step 2: Verify GPU
import tensorflow as tf
print("\n" + "="*50)
print("GPU Configuration")
print("="*50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if tf.config.list_physical_devices('GPU'):
    print(f"GPU Device: {tf.config.list_physical_devices('GPU')[0]}")
    print("✓ GPU is ready for training!")
else:
    print("⚠️ No GPU detected. Go to Runtime → Change runtime type → GPU")
print("="*50)

# Step 3: Check working directory
import os
print(f"\nCurrent working directory: {os.getcwd()}")

print("\n✅ Setup complete! You can now run the training scripts.")

