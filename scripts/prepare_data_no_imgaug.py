"""
Data Preparation Script WITHOUT imgaug (using PIL/TensorFlow augmentation)
Works with NumPy 2.0 and doesn't require imgaug
"""

import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageTransform
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import random

# Set random seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Set base directory
BASE_DIR = Path("/content/Glaucoma_detection/Glaucoma_detection")
if not (BASE_DIR / "scripts").exists():
    BASE_DIR = Path(os.getcwd())

# Directories
RAW_DATA_DIR = BASE_DIR / "RIM-ONE_DL_images"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
TEST_DIR = PROCESSED_DATA_DIR / "test"

CLASSES = ["Glaucoma", "Normal"]
CLASS_MAPPING = {"glaucoma": "Glaucoma", "normal": "Normal"}
IMG_SIZE = (224, 224)

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

def create_directories():
    """Create all necessary directories"""
    for split in [TRAIN_DIR, TEST_DIR]:
        for class_name in CLASSES:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    print("✓ Created directory structure")

def load_and_resize_image(image_path, target_size=IMG_SIZE):
    """Load and resize image"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def augment_image_pil(img, augmentations_per_image=2):
    """Apply augmentations using PIL (simple and NumPy 2.0 compatible)"""
    augmented_images = []
    
    for i in range(augmentations_per_image):
        aug_img = img.copy()
        
        # Random rotation (±20 degrees)
        if random.random() < 0.7:
            angle = random.uniform(-20, 20)
            aug_img = aug_img.rotate(angle, fillcolor=(0, 0, 0))
        
        # Random horizontal flip
        if random.random() < 0.5:
            aug_img = ImageOps.mirror(aug_img)
        
        # Random brightness adjustment (±20%)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random zoom (0.85-1.2)
        if random.random() < 0.5:
            zoom_factor = random.uniform(0.85, 1.2)
            w, h = aug_img.size
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            aug_img = aug_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Crop or pad to original size
            if new_w > w or new_h > h:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                aug_img = aug_img.crop((left, top, left + w, top + h))
            else:
                # Pad
                pad_left = (w - new_w) // 2
                pad_top = (h - new_h) // 2
                new_img = Image.new('RGB', (w, h), (0, 0, 0))
                new_img.paste(aug_img, (pad_left, pad_top))
                aug_img = new_img
        
        augmented_images.append(aug_img)
    
    return augmented_images

def merge_and_process_split(split_name="train", augment=False):
    """Merge and process images"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    target_dir = TRAIN_DIR if split_name == "train" else TEST_DIR
    class_counts = {"Glaucoma": 0, "Normal": 0}
    
    for class_key, class_name in CLASS_MAPPING.items():
        print(f"\nProcessing class: {class_name} (from '{class_key}' folders)")
        image_files = []
        
        for partition_name, partition_paths in PARTITIONS.items():
            source_dir = partition_paths[split_name] / class_key
            if source_dir.exists():
                files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
                image_files.extend([(f, partition_name) for f in files])
        
        print(f"  Found {len(image_files)} source images")
        
        processed_count = 0
        for img_path, partition in tqdm(image_files, desc=f"  Processing {class_name}"):
            img = load_and_resize_image(img_path)
            if img is None:
                continue
            
            base_name = img_path.stem
            suffix = f"_{partition}_{base_name}"
            
            # Save original
            original_filename = f"{suffix}_original.png"
            img.save(target_dir / class_name / original_filename)
            class_counts[class_name] += 1
            processed_count += 1
            
            # Augment for training
            if augment and split_name == "train":
                augmented_imgs = augment_image_pil(img, augmentations_per_image=2)
                for idx, aug_img in enumerate(augmented_imgs):
                    aug_filename = f"{suffix}_aug{idx+1}.png"
                    aug_img.save(target_dir / class_name / aug_filename)
                    class_counts[class_name] += 1
        
        print(f"  ✓ Processed {processed_count} original images for {class_name}")
        if augment and split_name == "train":
            print(f"  ✓ Total images after augmentation: {class_counts[class_name]}")

def generate_summary(class_counts_train, class_counts_test):
    """Generate summary"""
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
            "augmentations_per_image": 2,
            "method": "PIL-based (NumPy 2.0 compatible)"
        }
    }
    
    summary_path = PROCESSED_DATA_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
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

# MAIN EXECUTION
print("="*60)
print("GLaucoma Detection - Data Preparation Pipeline")
print("(Using PIL-based augmentation - NumPy 2.0 compatible)")
print("="*60)

create_directories()
train_counts = merge_and_process_split(split_name="train", augment=True)
test_counts = merge_and_process_split(split_name="test", augment=False)
summary = generate_summary(train_counts, test_counts)

print("\n" + "="*60)
print("✅ Data preparation completed successfully!")
print("="*60)
print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")

