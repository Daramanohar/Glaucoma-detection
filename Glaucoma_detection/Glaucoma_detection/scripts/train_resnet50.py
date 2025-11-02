"""
ResNet50 Fine-tuning Script for Glaucoma Detection
Implements transfer learning with phased training strategy
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import yaml

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "tensorboard").mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS_PHASE_A = 10  # Head training
EPOCHS_PHASE_B = 20  # Fine-tuning
WEIGHT_DECAY = 1e-4

# Enable mixed precision for faster training on GPU
try:
    set_global_policy('mixed_float16')
    print("✓ Mixed precision enabled")
except:
    print("Mixed precision not available, using float32")


def create_data_generators(use_augmentation=True):
    """Create data generators for training and validation"""
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=10,
        fill_mode='nearest',
        validation_split=0.15  # 15% for validation
    )
    
    # Validation/Test generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )
    
    train_dir = PROCESSED_DATA_DIR / "train"
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator


def build_model(base_trainable=False):
    """Build ResNet50 model with custom head"""
    
    # Base model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    base_model.trainable = base_trainable
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.L2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.L2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    return model, base_model


def get_callbacks(run_name, phase="A"):
    """Get training callbacks"""
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"resnet50_phase{phase}_epoch{{epoch:02d}}_valacc{{val_accuracy:.3f}}.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / "tensorboard" / f"{run_name}_phase{phase}"),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        callbacks.CSVLogger(
            filename=str(LOGS_DIR / f"training_history_phase{phase}.csv")
        )
    ]
    
    return callbacks_list


def train_phase_a(model, train_gen, val_gen):
    """Phase A: Train head only (base frozen)"""
    print("\n" + "="*60)
    print("PHASE A: Training Head (Base Frozen)")
    print("="*60)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    callbacks_list = get_callbacks("resnet50_finetuned", phase="A")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_A,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history


def train_phase_b(model, base_model, train_gen, val_gen):
    """Phase B: Fine-tune last layers of ResNet50"""
    print("\n" + "="*60)
    print("PHASE B: Fine-tuning ResNet50")
    print("="*60)
    
    # Unfreeze last 50 layers (approximately last 2 blocks)
    num_layers = len(base_model.layers)
    unfreeze_from = num_layers - 50
    
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    print(f"Unfroze {num_layers - unfreeze_from} layers")
    print(f"Frozen {unfreeze_from} layers")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    callbacks_list = get_callbacks("resnet50_finetuned", phase="B")
    
    # Update checkpoint to save best overall model
    best_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "resnet50_finetuned.best.h5"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks_list[0] = best_checkpoint
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_B,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history


def save_training_config(train_gen, val_gen):
    """Save training configuration"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "ResNet50",
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "initial_lr": INITIAL_LR,
        "fine_tune_lr": FINE_TUNE_LR,
        "epochs_phase_a": EPOCHS_PHASE_A,
        "epochs_phase_b": EPOCHS_PHASE_B,
        "weight_decay": WEIGHT_DECAY,
        "training_samples": train_gen.samples,
        "validation_samples": val_gen.samples,
        "class_indices": train_gen.class_indices,
        "mixed_precision": True
    }
    
    config_path = LOGS_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training config saved to: {config_path}")
    return config


def save_training_history(history_a, history_b):
    """Save complete training history"""
    # Combine histories
    combined_history = {}
    for key in history_a.history.keys():
        combined_history[key] = history_a.history[key] + history_b.history[key]
    
    history_path = LOGS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    
    # Also save as pickle for easy loading
    pickle_path = LOGS_DIR / "training_history.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(combined_history, f)
    
    print(f"✓ Training history saved to: {history_path}")
    return combined_history


def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to: {save_path}")


def main():
    """Main training function"""
    print("="*60)
    print("ResNet50 Fine-tuning for Glaucoma Detection")
    print("="*60)
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Save config
    config = save_training_config(train_gen, val_gen)
    
    # Build model
    print("\nBuilding ResNet50 model...")
    model, base_model = build_model(base_trainable=False)
    print(f"✓ Model built. Total parameters: {model.count_params():,}")
    
    # Phase A: Train head
    history_a = train_phase_a(model, train_gen, val_gen)
    
    # Phase B: Fine-tune
    history_b = train_phase_b(model, base_model, train_gen, val_gen)
    
    # Save final model
    final_model_path = MODELS_DIR / "resnet50_finetuned.final.h5"
    model.save(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")
    
    # Save training history
    combined_history = save_training_history(history_a, history_b)
    
    # Plot training curves
    plot_training_curves(combined_history, RESULTS_DIR / "accuracy_loss.png")
    
    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)
    print(f"\nBest model: {MODELS_DIR / 'resnet50_finetuned.best.h5'}")
    print(f"Training logs: {LOGS_DIR}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

