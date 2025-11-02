"""
Data Preparation Script for Glaucoma Detection Pipeline
Merges and preprocesses images from both partitions (partitioned_by_hospital + partitioned_randomly)
"""

import os
import shutil
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Augmentation imports
from imgaug import augmenters as iaa
import imgaug as ia

# Import utility function for base directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_base_dir

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
ia.seed(SEED)

# Directories
BASE_DIR = get_base_dir()
RAW_DATA_DIR = BASE_DIR / "RIM-ONE_DL_images"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# Source partitions
PARTITIONS = {
    "hospital": {
        "train": RAW_DATA_DIR / "partitioned_by_hospital" / "training_set",
        "test": RAW_DATA_DIR / "partitioned_by_hospital" / "test_set"
    },
    "random": {
        "train": RAW_DATA_DIR / "partitioned_randomly" / "training_set",
        "test": RAW_DATA_DIR / "partitioned_randomly" / "test_set"
    }
}

# Target directories
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
TEST_DIR = PROCESSED_DATA_DIR / "test"

CLASSES = ["Glaucoma", "Normal"]
CLASS_MAPPING = {"glaucoma": "Glaucoma", "normal": "Normal"}

IMG_SIZE = (224, 224)


def create_directories():
    """Create all necessary directories"""
    for split in [TRAIN_DIR, TEST_DIR]:
        for class_name in CLASSES:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    print("✓ Created directory structure")


def load_and_resize_image(image_path, target_size=IMG_SIZE):
    """Load and resize image to target size"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def augment_image(image, augmentations_per_image=3):
    """Apply augmentations to image and return list of augmented images"""
    augmented_images = []
    
    # Define augmentation pipeline
    seq = iaa.Sequential([
        iaa.Sometimes(0.7, iaa.Affine(
            rotate=(-20, 20),
            shear=(-10, 10),
            scale=(0.85, 1.2)
        )),
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),  # Horizontal flip
        iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.8, 1.2))),  # Brightness
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),  # Gaussian blur
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 50), sigma=5))
    ], random_order=True)
    
    for _ in range(augmentations_per_image):
        aug_img = seq(image=image)
        augmented_images.append(aug_img)
    
    return augmented_images


def merge_and_process_split(split_name="train", augment=False):
    """
    Merge images from both partitions for a given split (train or test)
    Apply augmentation only for training set
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    target_dir = TRAIN_DIR if split_name == "train" else TEST_DIR
    class_counts = {"Glaucoma": 0, "Normal": 0}
    
    for class_key, class_name in CLASS_MAPPING.items():
        print(f"\nProcessing class: {class_name} (from '{class_key}' folders)")
        
        image_files = []
        
        # Collect images from both partitions
        for partition_name, partition_paths in PARTITIONS.items():
            source_dir = partition_paths[split_name] / class_key
            if source_dir.exists():
                files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
                image_files.extend([(f, partition_name) for f in files])
        
        print(f"  Found {len(image_files)} source images")
        
        # Process each image
        processed_count = 0
        for img_path, partition in tqdm(image_files, desc=f"  Processing {class_name}"):
            # Load and resize
            img = load_and_resize_image(img_path)
            if img is None:
                continue
            
            # Generate unique filename with partition prefix to avoid clashes
            base_name = img_path.stem
            suffix = f"_{partition}_{base_name}"
            
            # Save original resized image
            original_filename = f"{suffix}_original.png"
            Image.fromarray(img).save(target_dir / class_name / original_filename)
            class_counts[class_name] += 1
            processed_count += 1
            
            # Apply augmentation only for training set
            if augment and split_name == "train":
                augmented_imgs = augment_image(img, augmentations_per_image=2)
                for idx, aug_img in enumerate(augmented_imgs):
                    aug_filename = f"{suffix}_aug{idx+1}.png"
                    Image.fromarray(aug_img).save(target_dir / class_name / aug_filename)
                    class_counts[class_name] += 1
        
        print(f"  ✓ Processed {processed_count} original images for {class_name}")
        if augment and split_name == "train":
            print(f"  ✓ Total images after augmentation: {class_counts[class_name]}")
    
    return class_counts


def generate_summary(class_counts_train, class_counts_test):
    """Generate and save summary statistics"""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "seed": SEED,
        "image_size": IMG_SIZE,
        "training_set": {
            "Glaucoma": class_counts_train["Glaucoma"],
            "Normal": class_counts_train["Normal"],
            "Total": class_counts_train["Glaucoma"] + class_counts_train["Normal"]
        },
        "test_set": {
            "Glaucoma": class_counts_test["Glaucoma"],
            "Normal": class_counts_test["Normal"],
            "Total": class_counts_test["Glaucoma"] + class_counts_test["Normal"]
        },
        "augmentation": {
            "train": True,
            "test": False,
            "augmentations_per_image": 2
        }
    }
    
    # Save summary
    summary_path = PROCESSED_DATA_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"\nTraining Set:")
    print(f"  Glaucoma: {class_counts_train['Glaucoma']}")
    print(f"  Normal:   {class_counts_train['Normal']}")
    print(f"  Total:    {summary['training_set']['Total']}")
    
    print(f"\nTest Set:")
    print(f"  Glaucoma: {class_counts_test['Glaucoma']}")
    print(f"  Normal:   {class_counts_test['Normal']}")
    print(f"  Total:    {summary['test_set']['Total']}")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    return summary


def main():
    """Main function to run data preparation pipeline"""
    print("="*60)
    print("GLaucoma Detection - Data Preparation Pipeline")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Process training set (with augmentation)
    train_counts = merge_and_process_split(split_name="train", augment=True)
    
    # Process test set (no augmentation, only resize)
    test_counts = merge_and_process_split(split_name="test", augment=False)
    
    # Generate summary
    summary = generate_summary(train_counts, test_counts)
    
    print("\n" + "="*60)
    print("✓ Data preparation completed successfully!")
    print("="*60)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")
    print(f"  - Training images: {PROCESSED_DATA_DIR / 'train'}")
    print(f"  - Test images:    {PROCESSED_DATA_DIR / 'test'}")


if __name__ == "__main__":
    main()

