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
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse

# Try to import OpenCV, set to None if not available (e.g., in cloud environments)
try:
    import cv2
except ImportError:
    cv2 = None


class GradCAM:
    """Robust Grad-CAM implementation that matches model input signatures."""

    def __init__(self, model, layer_name=None, use_tf_function=False):
        """
        Args:
            model: Keras model
            layer_name: optional conv layer name
            use_tf_function: if True, gradient computation is wrapped with @tf.function (less retracing)
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = self._build_grad_model()
        self.use_tf_function = use_tf_function

        if use_tf_function:
            self._gradcam_fn = tf.function(self._compute_conv_and_logits)
        else:
            self._gradcam_fn = self._compute_conv_and_logits

        print(f"✅ Grad-CAM initialized using layer: {self.layer_name}")

    # -----------------------------------------------------------
    #  Layer finding helpers
    # -----------------------------------------------------------

    def _find_target_layer(self):
        """Auto-detects the last convolutional layer in the model."""
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and ('resnet' in layer.name.lower() or 'resnet50' in layer.name.lower()):
                base_model = layer
                break

        search_layers = base_model.layers if base_model else self.model.layers
        for layer in reversed(search_layers):
            try:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer.name
            except Exception:
                continue

        # Try by 4D output rank (robust via K.int_shape)
        from tensorflow.keras import backend as K
        for layer in reversed(self.model.layers):
            try:
                out_shape = K.int_shape(layer.output)
                if out_shape is not None and len(out_shape) == 4:
                    return layer.name
            except Exception:
                continue

        # Try known ResNet50 last conv names
        for candidate in [
            'conv5_block3_out',
            'conv5_block3_3_conv',
            'conv5_block3_add'
        ]:
            try:
                _ = self.model.get_layer(candidate)
                return candidate
            except Exception:
                pass

        raise ValueError("Could not automatically find a convolutional layer for Grad-CAM.")

    # -----------------------------------------------------------
    #  Build Gradient Model
    # -----------------------------------------------------------

    def _build_grad_model(self):
        """Builds model that outputs activations and predictions."""
        try:
            target_layer = self.model.get_layer(self.layer_name)
        except Exception:
            target_layer = None

        if target_layer is None:
            for layer in reversed(self.model.layers):
                try:
                    if isinstance(layer, tf.keras.layers.Conv2D) or (len(layer.output_shape) == 4):
                        target_layer = layer
                        break
                except Exception:
                    continue

        if target_layer is None:
            # Try known ResNet50 candidates
            for candidate in [
                'conv5_block3_out',
                'conv5_block3_3_conv',
                'conv5_block3_add'
            ]:
                try:
                    target_layer = self.model.get_layer(candidate)
                    if target_layer is not None:
                        break
                except Exception:
                    continue

        if target_layer is None:
            raise ValueError("Grad-CAM: No suitable convolutional layer found.")

        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )
        return grad_model

    # -----------------------------------------------------------
    #  Input structure handling
    # -----------------------------------------------------------

    def _pack_inputs_for_model(self, arr):
        """Ensure input format matches model input structure."""
        inputs_spec = self.grad_model.inputs
        if isinstance(inputs_spec, list):
            if len(inputs_spec) == 1:
                return arr  # model expects one input tensor
            else:
                return [arr for _ in inputs_spec]
        else:
            return arr

    # -----------------------------------------------------------
    #  Core Grad-CAM computation
    # -----------------------------------------------------------

    def _compute_conv_and_logits(self, img_tensor):
        packed = self._pack_inputs_for_model(img_tensor)
        return self.grad_model(packed, training=False)

    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap for an image array.
        img_array: np.array or tf.Tensor (1, H, W, C)
        """
        if not isinstance(img_array, (tf.Tensor, tf.Variable)):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Cast to correct dtype
        try:
            target_dtype = self.grad_model.inputs[0].dtype
            img_array = tf.cast(img_array, target_dtype)
        except Exception:
            pass

        # Predict class if not given
        preds = self.model.predict(img_array, verbose=0)
        if pred_index is None:
            pred_index = int(np.argmax(preds[0]))

        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = self._gradcam_fn(img_array)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradients are None. Check if target layer is connected to output.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(tf.matmul(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            heatmap = tf.zeros_like(heatmap)
        else:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    # -----------------------------------------------------------
    #  Heatmap overlay
    # -----------------------------------------------------------

    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=None):
        """Overlay heatmap on image."""
        if cv2 is None:
            # Fallback for environments without OpenCV
            import matplotlib.pyplot as plt
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]
            overlay = (1 - alpha) * img / 255.0 + alpha * heatmap_colored
            return (overlay * 255).astype(np.uint8)
        
        # Original OpenCV implementation
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay


def preprocess_image(img_path, target_size=(256, 256)):
    """Load and preprocess image for model prediction"""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array, img


def _infer_target_size_from_model(model):
    """Infer (H, W) from a loaded Keras model robustly."""
    # Try model.inputs[0].shape (TensorShape)
    try:
        shp = model.inputs[0].shape
        if hasattr(shp, 'as_list'):
            shp = shp.as_list()
        # shp: [None, H, W, C]
        if len(shp) >= 4 and shp[1] is not None and shp[2] is not None:
            return (int(shp[1]), int(shp[2]))
    except Exception:
        pass
    # Fallback to model.input_shape
    try:
        shp = model.input_shape
        # could be tuple or list like (None, H, W, C)
        if isinstance(shp, (list, tuple)) and len(shp) >= 4 and shp[1] and shp[2]:
            return (int(shp[1]), int(shp[2]))
    except Exception:
        pass
    return (256, 256)


def generate_gradcam_for_sample(model_path, img_path, save_path, layer_name=None, target_size=None):
    """Generate Grad-CAM visualization for a single image"""
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    
    # Create GradCAM instance (respect CLI-provided layer name)
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Resolve target size from model if not provided
    if target_size is None:
        target_size = _infer_target_size_from_model(model)

    # Load and preprocess image
    img_array, img = preprocess_image(img_path, target_size=target_size)
    
    # Get original image as array
    img_orig = np.array(img)
    
    # Generate heatmap
    heatmap = gradcam.make_gradcam_heatmap(img_array)
    
    # Overlay heatmap
    overlaid = gradcam.overlay_heatmap(img_orig, heatmap)
    
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


def generate_gradcam_samples(model_path, test_dir, save_dir, num_samples=20, target_size=(256, 256), layer_name=None):
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
    
    # Create GradCAM instance
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Create test generator to get predictions
    test_datagen = ImageDataGenerator(rescale=1./255)
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
                generate_gradcam_for_sample(
                    model_path,
                    str(img_path),
                    str(save_path),
                    layer_name=gradcam.layer_name,
                    target_size=target_size
                )
                print(f"  ✓ Saved: {save_path.name}")
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
    
    print(f"\n✓ Grad-CAM samples saved to: {save_dir}")


def get_base_dir():
    """Colab/CLI safe base dir resolver (no __file__)."""
    try:
        cwd = Path(os.getcwd())
    except Exception:
        cwd = Path(".").resolve()
    if (cwd / "scripts").exists() and (cwd / "processed_data").exists():
        return cwd
    if (cwd.parent / "scripts").exists() and (cwd.parent / "processed_data").exists():
        return cwd.parent
    for p in [
        Path("/content/Glaucoma_detection/Glaucoma_detection"),
        Path("/content/Glaucoma_detection"),
        Path("/content/drive/MyDrive/Glaucoma_detection"),
    ]:
        if p.exists() and (p / "processed_data").exists():
            return p
    return cwd

def main():
    """Generate Grad-CAM samples for evaluation (CLI-friendly)."""
    # Use parse_known_args to tolerate Jupyter/Colab's extra argv (e.g., -f ...json)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--num_samples', type=int, default=10, help='Samples per category (TP/FP/TN/FN)')
    parser.add_argument('--layer', type=str, default=None, help='Conv layer name to use for Grad-CAM (e.g., conv5_block3_out)')
    args, _ = parser.parse_known_args()

    BASE_DIR = get_base_dir()
    MODEL_PATH = BASE_DIR / "models" / "resnet50_finetuned.best.h5"
    TEST_DIR = BASE_DIR / "processed_data" / "test"
    SAVE_DIR = BASE_DIR / "results" / "gradcam_samples"

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Detect model input size
    model = keras.models.load_model(MODEL_PATH)
    target_size = (256, 256)
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, tuple) and len(in_shape) == 4 and in_shape[1] and in_shape[2]:
            target_size = (int(in_shape[1]), int(in_shape[2]))
            print(f"✓ Detected model input size: {target_size}")
    except Exception:
        pass

    # Run
    generate_gradcam_samples(
        str(MODEL_PATH),
        str(TEST_DIR),
        SAVE_DIR,
        num_samples=args.num_samples,
        target_size=target_size,
        layer_name=args.layer
    )


if __name__ == "__main__":
    main()

