"""
Evaluation Script for Glaucoma Detection Model
Computes comprehensive metrics, confusion matrix, ROC curve, and saves predictions
"""

import os
import json
import csv
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Colab-safe base directory resolver (does not rely on __file__)
def get_base_dir():
    try:
        import os
        cwd = Path(os.getcwd())
    except Exception:
        cwd = Path(".").resolve()
    # If running from project root
    if (cwd / "scripts").exists() and (cwd / "processed_data").exists():
        return cwd
    # If running from within scripts/
    if (cwd.parent / "scripts").exists() and (cwd.parent / "processed_data").exists():
        return cwd.parent
    # Common Colab locations
    for p in [
        Path("/content/Glaucoma_detection/Glaucoma_detection"),
        Path("/content/Glaucoma_detection"),
        Path("/content/drive/MyDrive/Glaucoma_detection"),
    ]:
        if p.exists() and (p / "processed_data").exists():
            return p
    return cwd

# Set random seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directories
BASE_DIR = get_base_dir()
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create results subdirectories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "gradcam_samples").mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)  # default; will auto-detect from model if possible
BATCH_SIZE = 32
# Evaluation options
USE_TTA = True  # Test-time augmentation: original + horizontal flip
ECE_NUM_BINS = 15


def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model


def create_test_generator(target_size):
    """Create test data generator"""
    test_dir = PROCESSED_DATA_DIR / "test"
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Class indices: {test_generator.class_indices}")
    
    return test_generator


def _predict_generator(model, generator):
    generator.reset()
    return model.predict(generator, verbose=1).ravel()

def tta_predict(model, test_generator, target_size):
    """Average predictions of original and horizontally flipped test set."""
    print("\nUsing TTA: original + horizontal flip")
    p0 = _predict_generator(model, test_generator)
    # Build a second generator with horizontal flip
    tta_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    tta_gen = tta_datagen.flow_from_directory(
        test_generator.directory,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    p1 = _predict_generator(model, tta_gen)
    return (p0 + p1) / 2.0

def predict_test_set(model, test_generator, target_size):
    """Make predictions on test set (with optional TTA)"""
    print("\nGenerating predictions...")
    y_pred_proba = tta_predict(model, test_generator, target_size) if USE_TTA else _predict_generator(model, test_generator)
    y_true = test_generator.classes
    y_pred = (y_pred_proba > 0.5).astype(int)
    filenames = test_generator.filenames
    return y_true, y_pred, y_pred_proba, filenames


def compute_metrics(y_true, y_pred, y_pred_proba):
    """Compute all evaluation metrics"""
    print("\nComputing metrics...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Classification report
    class_names = ['Normal', 'Glaucoma']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "classification_report": report,
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics, fpr, tpr, thresholds

def compute_ece(y_true, y_prob, num_bins=ECE_NUM_BINS):
    """Compute Expected Calibration Error (ECE) and bin stats."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    bin_data = []
    for b in range(num_bins):
        idx = bin_ids == b
        if np.sum(idx) == 0:
            bin_data.append((bins[b], bins[b+1], 0, 0.0, 0.0))
            continue
        conf = float(np.mean(y_prob[idx]))
        acc = float(np.mean((y_true[idx] == (y_prob[idx] > 0.5)).astype(float)))
        w = float(np.sum(idx) / len(y_prob))
        ece += w * abs(acc - conf)
        bin_data.append((bins[b], bins[b+1], int(np.sum(idx)), acc, conf))
    return float(ece), bin_data

def plot_reliability_curve(bin_data, ece, save_path):
    mids = [0.5 * (b0 + b1) for b0, b1, _, _, _ in bin_data]
    accs = [acc for _, _, _, acc, _ in bin_data]
    confs = [conf for _, _, _, _, conf in bin_data]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(confs, accs, marker='o', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE={ece:.3f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics(metrics, save_path):
    """Save metrics to JSON"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    metrics_clean = convert_types(metrics)
    
    with open(save_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"✓ Metrics saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=9, color='red')
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved to: {save_path}")


def save_classification_report(metrics, save_path):
    """Save classification report to text file"""
    report = metrics['classification_report']
    
    with open(save_path, "w") as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write("="*60 + "\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*60 + "\n")
        for class_name in ['Normal', 'Glaucoma']:
            if class_name in report:
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
                f.write(f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n")
                f.write(f"  Support:   {report[class_name]['support']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Macro Average:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Weighted Average:\n")
        f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"✓ Classification report saved to: {save_path}")


def save_predictions_csv(y_true, y_pred, y_pred_proba, filenames, save_path):
    """Save predictions to CSV"""
    class_names = ['Normal', 'Glaucoma']
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_label', 'predicted_label', 'probability', 'correct'])
        
        for filename, true, pred, proba in zip(filenames, y_true, y_pred, y_pred_proba):
            true_name = class_names[true]
            pred_name = class_names[pred]
            correct = 'Yes' if true == pred else 'No'
            writer.writerow([filename, true_name, pred_name, f"{proba:.4f}", correct])
    
    print(f"✓ Predictions saved to: {save_path}")


def print_metrics_summary(metrics):
    """Print metrics summary to console"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("\n" + "-"*60)
    print("\nClassification Report:")
    report = metrics['classification_report']
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-"*65)
    for class_name in ['Normal', 'Glaucoma']:
        if class_name in report:
            print(f"{class_name:<15} {report[class_name]['precision']:<12.4f} "
                  f"{report[class_name]['recall']:<12.4f} "
                  f"{report[class_name]['f1-score']:<12.4f} "
                  f"{report[class_name]['support']:<12}")
    print("-"*65)
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} "
          f"{report['macro avg']['f1-score']:<12.4f} "
          f"{report['macro avg']['support']:<12}")
    print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} "
          f"{report['weighted avg']['f1-score']:<12.4f} "
          f"{report['weighted avg']['support']:<12}")


def main():
    """Main evaluation function"""
    print("="*60)
    print("Model Evaluation Pipeline")
    print("="*60)
    
    # Load model
    model_path = MODELS_DIR / "resnet50_finetuned.best.h5"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_resnet50.py")
        return
    
    model = load_model(model_path)
    
    # Auto-detect model input size
    target_size = IMG_SIZE
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, tuple) and len(in_shape) == 4 and in_shape[1] and in_shape[2]:
            target_size = (int(in_shape[1]), int(in_shape[2]))
            print(f"✓ Detected model input size: {target_size}")
    except Exception:
        pass

    # Create test generator
    test_generator = create_test_generator(target_size)
    
    # Make predictions
    y_true, y_pred, y_pred_proba, filenames = predict_test_set(model, test_generator, target_size)
    
    # Compute metrics
    metrics, fpr, tpr, thresholds = compute_metrics(y_true, y_pred, y_pred_proba)
    # Calibration
    ece, bin_data = compute_ece(y_true, y_pred_proba)
    metrics["ece"] = ece
    
    # Save metrics
    save_metrics(metrics, RESULTS_DIR / "metrics.json")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=['Normal', 'Glaucoma'],
        save_path=RESULTS_DIR / "confusion_matrix.png"
    )
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], RESULTS_DIR / "roc_auc.png")
    # Reliability diagram
    plot_reliability_curve(bin_data, ece, RESULTS_DIR / "reliability_curve.png")
    
    # Save classification report
    save_classification_report(metrics, RESULTS_DIR / "classification_report.txt")
    
    # Save predictions CSV
    save_predictions_csv(
        y_true, y_pred, y_pred_proba, filenames,
        RESULTS_DIR / "predictions.csv"
    )
    
    # Print summary
    print_metrics_summary(metrics)
    
    print("\n" + "="*60)
    print("✓ Evaluation completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

