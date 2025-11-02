"""
OPTIMIZED ResNet50 Training Script
Fixed early stopping, learning rate scheduling, and class imbalance handling
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy

# Set base directory
BASE_DIR = Path("/content/Glaucoma_detection/Glaucoma_detection")
if not (BASE_DIR / "scripts").exists():
    BASE_DIR = Path(os.getcwd())

# Set random seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Directories
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "tensorboard").mkdir(parents=True, exist_ok=True)

# OPTIMIZED Hyperparameters
BATCH_SIZE = 16
IMG_SIZE = (256, 256)
INITIAL_LR = 3e-4  # Reduced for stability
FINE_TUNE_LR = 3e-6  # Lower LR for better fine-tuning stability
FINE_TUNE_LR_PHASE_C = 3e-6  # Very low LR for final full-unfreeze polish
RUN_PHASE_C = False  # Disable Phase C by default (prevents degrading a good Phase B model)
EPOCHS_PHASE_A = 20  # Head training
EPOCHS_PHASE_B = 40  # Partial unfreeze (increased)
EPOCHS_PHASE_C = 10  # Full unfreeze fine polish
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05  # Add label smoothing for stability

# Enable mixed precision
try:
    set_global_policy('mixed_float16')
    print("✓ Mixed precision enabled")
except:
    print("Mixed precision not available, using float32")

print("="*60)
print("OPTIMIZED ResNet50 Training for Glaucoma Detection")
print("="*60)

def cosine_annealing_schedule(epoch, total_epochs, max_lr, min_lr=1e-6):
    """
    Cosine annealing learning rate schedule that never goes negative
    Formula: min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * epoch / total_epochs))
    """
    cos_value = np.cos(np.pi * epoch / total_epochs)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos_value)
    return float(max(min_lr, lr))  # Ensure never below min_lr

def create_data_generators():
    """Create data generators with proper augmentation"""
    # More aggressive augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=10,
        fill_mode='nearest',
        validation_split=0.15
    )
    
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
    
    # Calculate class weights for imbalanced data
    class_counts = {}
    for class_idx, class_name in enumerate(train_generator.class_indices.keys()):
        count = (train_generator.classes == class_idx).sum()
        class_counts[class_name] = count
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_idx, class_name in enumerate(train_generator.class_indices.keys()):
        class_weights[class_idx] = total_samples / (len(class_counts) * class_counts[class_name])
    
    print(f"\n✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Class indices: {train_generator.class_indices}")
    print(f"✓ Class weights (for imbalanced data): {class_weights}")
    
    return train_generator, val_generator, class_weights

def build_model(base_trainable=False):
    """Build ResNet50 model with optimized architecture"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'  # Use global average pooling at base level
    )
    
    base_model.trainable = base_trainable
    
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    
    # Don't add extra pooling since we used pooling='avg' in base_model
    # x = layers.GlobalAveragePooling2D()(x)  # Not needed
    
    # Larger head with better regularization
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.L2(WEIGHT_DECAY),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.L2(WEIGHT_DECAY),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.L2(WEIGHT_DECAY),
        name='dense_3'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
    
    model = models.Model(inputs, outputs, name='resnet50_glaucoma')
    
    return model, base_model

def get_callbacks_phase_a():
    """Optimized callbacks for Phase A"""
    return [
        # Model checkpoint - save best model based on val_accuracy
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "resnet50_phaseA_best.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # Early stopping - monitor val_accuracy with patience
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        # Learning rate reducer (remove if using cosine scheduler, but keep as backup)
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),
        # Cosine annealing for smooth LR decay (properly bounded)
        callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing_schedule(epoch, EPOCHS_PHASE_A, INITIAL_LR, min_lr=1e-5),
            verbose=1
        ),
        # TensorBoard
        callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / "tensorboard" / "phaseA"),
            histogram_freq=1,
            write_graph=True
        ),
        # CSV logger
        callbacks.CSVLogger(
            filename=str(LOGS_DIR / "training_history_phaseA.csv")
        )
    ]

def get_callbacks_phase_b():
    """Optimized callbacks for Phase B"""
    return [
        # Best model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "resnet50_finetuned.best.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        # Learning rate reducer
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),
        # Cosine annealing (properly bounded)
        callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing_schedule(epoch, EPOCHS_PHASE_B, FINE_TUNE_LR, min_lr=1e-6),
            verbose=1
        ),
        # TensorBoard
        callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / "tensorboard" / "phaseB"),
            histogram_freq=1
        ),
        # CSV logger
        callbacks.CSVLogger(
            filename=str(LOGS_DIR / "training_history_phaseB.csv")
        )
    ]

# Create data generators
print("\n📊 Creating data generators...")
train_gen, val_gen, class_weights = create_data_generators()

# Save config
config = {
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "model": "ResNet50_Optimized",
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "initial_lr": INITIAL_LR,
    "fine_tune_lr": FINE_TUNE_LR,
    "epochs_phase_a": EPOCHS_PHASE_A,
    "epochs_phase_b": EPOCHS_PHASE_B,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "training_samples": train_gen.samples,
    "validation_samples": val_gen.samples,
    "class_indices": train_gen.class_indices,
    "class_weights": class_weights
}

config_path = LOGS_DIR / "training_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved to: {config_path}")

# Build model
print("\n🏗️ Building ResNet50 model...")
model, base_model = build_model(base_trainable=False)
print(f"✓ Model built. Total parameters: {model.count_params():,}")

# PHASE A: Train head
print("\n" + "="*60)
print("PHASE A: Training Head (Base Frozen)")
print("="*60)

# Use AdamW optimizer (weight decay optimizer)
optimizer_a = optimizers.AdamW(
    learning_rate=INITIAL_LR,
    weight_decay=WEIGHT_DECAY
)

# Use label smoothing for better generalization
loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

model.compile(
    optimizer=optimizer_a,
    loss=loss_fn,
    metrics=['accuracy', keras.metrics.AUC(name='auc'), 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print(f"Starting training with class weights: {class_weights}")
print(f"Initial LR: {INITIAL_LR}")
print(f"Label smoothing: {LABEL_SMOOTHING}")
print(f"Using cosine annealing LR schedule (bounded)")

history_a = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE_A,
    validation_data=val_gen,
    callbacks=get_callbacks_phase_a(),
    class_weight=class_weights,  # Handle class imbalance
    verbose=1
)

# Get best accuracy from Phase A
best_val_acc_a = max(history_a.history['val_accuracy'])
print(f"\n✓ Phase A completed. Best validation accuracy: {best_val_acc_a:.4f}")

# PHASE B: Fine-tune
print("\n" + "="*60)
print("PHASE B: Fine-tuning ResNet50")
print("="*60)

# Unfreeze last 100 layers for deeper adaptation
num_layers = len(base_model.layers)
unfreeze_from = max(0, num_layers - 100)

for layer in base_model.layers[:unfreeze_from]:
    layer.trainable = False
for layer in base_model.layers[unfreeze_from:]:
    layer.trainable = True

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Unfroze {trainable_count} layers (from layer {unfreeze_from})")
print(f"Frozen {num_layers - trainable_count} layers")

# Recompile with lower LR and AdamW
optimizer_b = optimizers.AdamW(
    learning_rate=FINE_TUNE_LR,
    weight_decay=WEIGHT_DECAY * 0.1  # Lower weight decay for fine-tuning
)

# Use label smoothing for better generalization
loss_fn_b = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

model.compile(
    optimizer=optimizer_b,
    loss=loss_fn_b,
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print(f"Fine-tuning LR: {FINE_TUNE_LR}")
print(f"Label smoothing: {LABEL_SMOOTHING}")
print(f"Using cosine annealing LR schedule (bounded)")

history_b = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE_B,
    validation_data=val_gen,
    callbacks=get_callbacks_phase_b(),
    class_weight=class_weights,
    verbose=1
)

# Get best accuracy from Phase B
best_val_acc_b = max(history_b.history['val_accuracy'])
print(f"\n✓ Phase B completed. Best validation accuracy: {best_val_acc_b:.4f}")

# PHASE C: Full unfreeze, very low LR polish (SGD Nesterov)
print("\n" + "="*60)
print("PHASE C: Full Unfreeze Fine-tuning")
print("="*60)

if RUN_PHASE_C:
    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True

    optimizer_c = optimizers.SGD(
        learning_rate=FINE_TUNE_LR_PHASE_C,
        momentum=0.9,
        nesterov=True
    )

    # Use label smoothing
    loss_fn_c = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=optimizer_c,
        loss=loss_fn_c,
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print(f"Full-unfreeze LR: {FINE_TUNE_LR_PHASE_C}")

    callbacks_c = [
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "resnet50_phaseC_best.h5"),  # separate file to avoid overwriting Phase B
            monitor='val_accuracy', mode='max', save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=6,
            restore_best_weights=True, verbose=1, min_delta=0.001
        ),
        callbacks.TensorBoard(log_dir=str(LOGS_DIR / "tensorboard" / "phaseC")),
        callbacks.CSVLogger(filename=str(LOGS_DIR / "training_history_phaseC.csv"))
    ]

    history_c = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_C,
        validation_data=val_gen,
        callbacks=callbacks_c,
        class_weight=class_weights,
        verbose=1
    )

# Load best model weights (choose best of Phase B vs Phase C if Phase C ran)
print("\n📥 Selecting best checkpoint...")
phase_b_path = MODELS_DIR / "resnet50_finetuned.best.h5"
phase_c_path = MODELS_DIR / "resnet50_phaseC_best.h5"

best_path = phase_b_path
if RUN_PHASE_C and phase_c_path.exists():
    # Evaluate both and pick the better on validation set
    tmp_model = keras.models.clone_model(model)
    tmp_model.build((None, *IMG_SIZE, 3))
    tmp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tmp_model.load_weights(str(phase_b_path))
    val_b = tmp_model.evaluate(val_gen, verbose=0)[1]
    tmp_model.load_weights(str(phase_c_path))
    val_c = tmp_model.evaluate(val_gen, verbose=0)[1]
    best_path = phase_c_path if val_c >= val_b else phase_b_path

print(f"Using checkpoint: {best_path.name}")
model.load_weights(str(best_path))

# Evaluate best model
print("\n📊 Evaluating best model on validation set...")
val_results = model.evaluate(val_gen, verbose=1)
print(f"✓ Best model validation accuracy: {val_results[1]:.4f}")
print(f"✓ Best model validation AUC: {val_results[2]:.4f}")

# Save final model
final_model_path = MODELS_DIR / "resnet50_finetuned.final.h5"
model.save(final_model_path)
print(f"\n✓ Final model saved to: {final_model_path}")

# Combine histories (include Phase C if run)
combined_history = {}
for key in history_a.history.keys():
    combined_history[key] = history_a.history[key] + history_b.history[key]
if 'history_c' in locals():
    for key in history_c.history.keys():
        if key in combined_history:
            combined_history[key] = combined_history[key] + history_c.history[key]
        else:
            # Align lengths by padding with last known value if needed
            pad = [history_a.history.get(key, [None])[-1]] * len(history_a.history.get(key, []))
            combined_history[key] = pad + history_b.history.get(key, []) + history_c.history[key]

# Save history
history_path = LOGS_DIR / "training_history.json"
with open(history_path, "w") as f:
    json.dump(combined_history, f, indent=2)
print(f"✓ Training history saved to: {history_path}")

# Plot training curves
print("\n📈 Generating training curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
axes[0, 0].plot(combined_history['accuracy'], label='Train Accuracy', marker='o', markersize=3)
axes[0, 0].plot(combined_history['val_accuracy'], label='Val Accuracy', marker='s', markersize=3)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].axvline(len(history_a.history['accuracy']), color='r', linestyle='--', label='Phase A → B')
axes[0, 0].legend()

# Loss
axes[0, 1].plot(combined_history['loss'], label='Train Loss', marker='o', markersize=3)
axes[0, 1].plot(combined_history['val_loss'], label='Val Loss', marker='s', markersize=3)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].axvline(len(history_a.history['loss']), color='r', linestyle='--')

# AUC
if 'auc' in combined_history:
    axes[1, 0].plot(combined_history['auc'], label='Train AUC', marker='o', markersize=3)
    axes[1, 0].plot(combined_history['val_auc'], label='Val AUC', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axvline(len(history_a.history['auc']), color='r', linestyle='--')

# Learning Rate
if 'lr' in combined_history:
    axes[1, 1].plot(combined_history['lr'], label='Learning Rate', marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axvline(len(history_a.history['lr']), color='r', linestyle='--')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_loss.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Training curves saved to: {RESULTS_DIR / 'accuracy_loss.png'}")

print("\n" + "="*60)
print("✅ OPTIMIZED Training completed successfully!")
print("="*60)
print(f"\n📊 Results Summary:")
print(f"  Phase A Best Val Accuracy: {best_val_acc_a:.4f}")
print(f"  Phase B Best Val Accuracy: {best_val_acc_b:.4f}")
if 'history_c' in locals():
    best_val_acc_c = max(history_c.history['val_accuracy']) if 'val_accuracy' in history_c.history else float('nan')
    print(f"  Phase C Best Val Accuracy: {best_val_acc_c:.4f}")
print(f"  Final Model Val Accuracy: {val_results[1]:.4f}")
print(f"  Final Model Val AUC: {val_results[2]:.4f}")
print(f"\n📁 Best model: {MODELS_DIR / 'resnet50_finetuned.best.h5'}")
print(f"📁 Training logs: {LOGS_DIR}")
print(f"📁 Results: {RESULTS_DIR}")
print("\nNext step: Run evaluation script")

