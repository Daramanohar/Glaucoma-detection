"""
Evaluation Script for Glaucoma Detection Model
Computes comprehensive metrics, confusion matrix, ROC curve, and saves predictions
"""

import os
import json
import csv
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

# Set random seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create results subdirectories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "gradcam_samples").mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model


def create_test_generator():
    """Create test data generator"""
    test_dir = PROCESSED_DATA_DIR / "test"
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Class indices: {test_generator.class_indices}")
    
    return test_generator


def predict_test_set(model, test_generator):
    """Make predictions on test set"""
    print("\nGenerating predictions...")
    
    # Reset generator
    test_generator.reset()
    
    # Predict probabilities
    y_pred_proba = model.predict(test_generator, verbose=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Get filenames
    filenames = test_generator.filenames
    
    return y_true, y_pred, y_pred_proba.flatten(), filenames


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
    
    # Create test generator
    test_generator = create_test_generator()
    
    # Make predictions
    y_true, y_pred, y_pred_proba, filenames = predict_test_set(model, test_generator)
    
    # Compute metrics
    metrics, fpr, tpr, thresholds = compute_metrics(y_true, y_pred, y_pred_proba)
    
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

