"""
Grad-CAM Implementation for Model Explainability
Generates heatmaps highlighting important regions for glaucoma detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
import argparse

# Try to import OpenCV, set to None if not available (e.g., in cloud environments)
try:
    import cv2
except ImportError:
    cv2 = None


class GradCAM:
    """Grad-CAM implementation for ResNet50"""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to use. If None, auto-detect.
        """
        self.model = model
        
        # Find the target layer (last convolutional layer in ResNet50)
        if layer_name is None:
            layer_name = self._find_target_layer()
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _find_target_layer(self):
        """Automatically find the last convolutional layer (robust)."""
        # Prefer nested ResNet base if present
        base = None
        for l in self.model.layers:
            if isinstance(l, keras.Model) and ('resnet' in l.name.lower() or 'resnet50' in l.name.lower()):
                base = l
                break

        search_layers = base.layers if base is not None else self.model.layers

        for layer in reversed(search_layers):
            try:
                shp = K.int_shape(layer.output)
                if shp is not None and len(shp) == 4:
                    return layer.name
            except Exception:
                continue

        # Known fallbacks
        for name in ['conv5_block3_out','conv5_block3_3_conv','conv5_block3_add']:
            try:
                _ = (base or self.model).get_layer(name)
                return name
            except Exception:
                pass

        return None
    
    def _build_grad_model(self):
        """Build a model that outputs both predictions and gradients"""
        # Find the base model (ResNet50)
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and ('resnet' in layer.name.lower() or 'resnet50' in layer.name.lower()):
                base_model = layer
                break
        
        if base_model is None:
            # If base model not found, use the first conv layer we can find
            for layer in self.model.layers:
                try:
                    shp = K.int_shape(layer.output)
                    if shp is not None and len(shp) == 4:
                        target_layer = layer
                        break
                except Exception:
                    continue
        else:
            # Find target layer in base model
            if self.layer_name:
                target_layer = base_model.get_layer(self.layer_name)
            else:
                # Use last conv layer of base model
                for layer in reversed(base_model.layers):
                    try:
                        shp = K.int_shape(layer.output)
                        if shp is not None and len(shp) == 4:
                            target_layer = layer
                            break
                    except Exception:
                        continue
        
        # Build gradient model
        # Use model.input which handles both named and unnamed inputs
        grad_model = keras.Model(inputs=self.model.input,
                                 outputs=[target_layer.output, self.model.output])
        
        return grad_model
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array
            pred_index: Class index to generate heatmap for (None = use predicted class)
        
        Returns:
            Heatmap as numpy array
        """
        # Handle named input layers
        if hasattr(self.model, 'input_names') and self.model.input_names:
            input_name = self.model.input_names[0]
            model_input = {input_name: img_array}
        else:
            model_input = img_array
        
        # Get predictions
        preds = self.model.predict(model_input, verbose=0)
        
        if pred_index is None:
            pred_index = np.argmax(preds[0])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(model_input, training=False)
            loss = predictions[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=None):
        """
        Overlay heatmap on original image
        
        Args:
            img: Original image (0-255 uint8)
            heatmap: Heatmap array (0-1 float)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image
        """
        if cv2 is None:
            # Fallback for environments without OpenCV
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_colored = matplotlib.pyplot.cm.jet(heatmap_normalized)[:, :, :3]
            overlay = (1 - alpha) * img / 255.0 + alpha * heatmap_colored
            return (overlay * 255).astype(np.uint8)
        
        # Original OpenCV implementation
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to uint8
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Convert image to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Overlay
        overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid


def occlusion_overlay(model, img_path, save_path, patch=32, stride=16, baseline=0.0):
    """Fallback explainability when no conv layer is available."""
    # Detect input size
    H, W = int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2])
    img = load_img(img_path, target_size=(W, H))
    arr = img_to_array(img).astype('float32')/255.0
    x0 = np.expand_dims(arr,0)
    p0 = float(model.predict(x0, verbose=0)[0][0])

    heat = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = min(y+patch, H), min(x+patch, W)
            x_occ = x0.copy()
            x_occ[:, y:y1, x:x1, :] = baseline
            p = float(model.predict(x_occ, verbose=0)[0][0])
            drop = max(0.0, p0 - p)
            heat[y:y1, x:x1] = drop

    if heat.max() > 0:
        heat /= heat.max()
    heat_255 = (heat*255).astype(np.uint8)
    
    # Use OpenCV if available, otherwise matplotlib
    if cv2 is not None:
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
        base = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        over = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, over)
    else:
        # Fallback to matplotlib
        heatmap_colored = matplotlib.pyplot.cm.jet(heat)[:, :, :3]
        overlay = (1 - 0.4) * arr + 0.4 * heatmap_colored
        over = (overlay * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil_img = Image.fromarray(over)
        pil_img.save(save_path)
    
    return heat


def preprocess_image(img_path, target_size=(256, 256)):
    """Load and preprocess image for model prediction"""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array, img


def generate_gradcam_for_sample(model_path, img_path, save_path, layer_name=None, target_size=None):
    """Generate Grad-CAM visualization for a single image"""
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    
    # Create GradCAM instance (may return None layer internally)
    grad_layer_name = layer_name
    try:
        gradcam = GradCAM(model, layer_name=layer_name)
    except Exception:
        gradcam = None
    
    # Detect model input size if not provided
    if target_size is None:
        try:
            ishape = model.inputs[0].shape
            target_size = (int(ishape[2]), int(ishape[1]))  # (W,H)
            target_size = (target_size[0], target_size[1])
        except Exception:
            target_size = (256, 256)
    # Load and preprocess image
    img_array, img = preprocess_image(img_path, target_size=target_size)
    
    # Get original image as array
    img_orig = np.array(img)
    
    if gradcam is not None and gradcam.layer_name is not None:
        try:
            heatmap = gradcam.make_gradcam_heatmap(img_array)
            overlaid = gradcam.overlay_heatmap(img_orig, heatmap)
        except Exception:
            # Fallback to occlusion
            heatmap = occlusion_overlay(model, img_path, save_path)
            return heatmap, None
    else:
        # No conv layer accessible → fallback to occlusion
        heatmap = occlusion_overlay(model, img_path, save_path)
        return heatmap, None
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_orig)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    if cv2 is not None:
        axes[2].imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    else:
        axes[2].imshow(overlaid)
    axes[2].set_title('Overlaid Heatmap', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return heatmap, overlaid


def generate_gradcam_samples(model_path, test_dir, save_dir, num_samples=20, layer_name=None, target_size=None):
    """
    Generate Grad-CAM visualizations for multiple test samples
    
    Args:
        model_path: Path to trained model
        test_dir: Directory containing test images
        save_dir: Directory to save Grad-CAM visualizations
        num_samples: Number of samples per category (TP, FP, TN, FN)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    
    # Create GradCAM instance (respect CLI layer if given)
    try:
        gradcam = GradCAM(model, layer_name=layer_name)
    except Exception:
        gradcam = None
    
    # Create test generator to get predictions
    test_datagen = ImageDataGenerator(rescale=1./255)
    # Infer target size from model if not provided
    if target_size is None:
        try:
            ishape = model.inputs[0].shape
            target_size = (int(ishape[1]), int(ishape[2]))
        except Exception:
            target_size = (256, 256)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predictions_binary = (predictions > 0.5).astype(int).flatten()
    true_labels = test_generator.classes
    filenames = test_generator.filenames
    
    # Categorize predictions
    categories = {
        'TP': [],  # True Positive (predicted glaucoma, is glaucoma)
        'FP': [],  # False Positive (predicted glaucoma, is normal)
        'TN': [],  # True Negative (predicted normal, is normal)
        'FN': []   # False Negative (predicted normal, is glaucoma)
    }
    
    for i, (true, pred, filename, proba) in enumerate(
        zip(true_labels, predictions_binary, filenames, predictions.flatten())
    ):
        if true == 1 and pred == 1:  # Glaucoma correctly predicted
            categories['TP'].append((i, filename, proba))
        elif true == 0 and pred == 1:  # Normal incorrectly predicted as glaucoma
            categories['FP'].append((i, filename, proba))
        elif true == 0 and pred == 0:  # Normal correctly predicted
            categories['TN'].append((i, filename, proba))
        elif true == 1 and pred == 0:  # Glaucoma incorrectly predicted as normal
            categories['FN'].append((i, filename, proba))
    
    # Generate Grad-CAM for samples from each category
    print("\nGenerating Grad-CAM visualizations...")
    for category, samples in categories.items():
        print(f"\n{category}: {len(samples)} samples found")
        
        # Select random samples
        num_to_select = min(num_samples, len(samples))
        selected = np.random.choice(len(samples), num_to_select, replace=False)
        
        for idx in selected:
            sample_idx, filename, proba = samples[idx]
            
            # Get full image path
            img_path = Path(test_dir) / filename
            
            if not img_path.exists():
                continue
            
            # Generate Grad-CAM
            save_path = save_dir / f"{category}_{Path(filename).stem}_proba{proba:.3f}.png"
            
            try:
                glayer = gradcam.layer_name if gradcam is not None else None
                generate_gradcam_for_sample(
                    model_path,
                    str(img_path),
                    str(save_path),
                    layer_name=glayer,
                    target_size=target_size
                )
                print(f"  ✓ Saved: {save_path.name}")
            except Exception as e:
                # Fallback: occlusion overlay
                try:
                    occlusion_overlay(model, str(img_path), str(save_path))
                    print(f"  ✓ Saved (occlusion): {save_path.name}")
                except Exception as e2:
                    print(f"  ✗ Error processing {filename}: {e2}")
    
    print(f"\n✓ Grad-CAM samples saved to: {save_dir}")


def main():
    """Generate Grad-CAM samples for evaluation (CLI-friendly)."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--num_samples', type=int, default=10, help='Samples per category (TP/FP/TN/FN)')
    parser.add_argument('--layer', type=str, default=None, help='Conv layer name to use (e.g., conv5_block3_out)')
    args, _ = parser.parse_known_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "models" / "resnet50_finetuned.best.h5"
    TEST_DIR = BASE_DIR / "processed_data" / "test"
    SAVE_DIR = BASE_DIR / "results" / "gradcam_samples"

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Detect model input size for target_size
    try:
        model_tmp = keras.models.load_model(MODEL_PATH, compile=False)
        ishape = model_tmp.inputs[0].shape
        target_size = (int(ishape[1]), int(ishape[2]))
        print(f"✓ Detected model input size: {target_size}")
    except Exception:
        target_size = (224, 224)

    generate_gradcam_samples(
        str(MODEL_PATH),
        str(TEST_DIR),
        SAVE_DIR,
        num_samples=args.num_samples,
        layer_name=args.layer,
        target_size=target_size
    )


if __name__ == "__main__":
    main()

