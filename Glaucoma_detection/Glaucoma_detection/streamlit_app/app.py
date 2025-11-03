"""
Streamlit Web Application for Glaucoma Detection
Provides interactive interface for image upload, prediction, and Grad-CAM visualization
Updated: Auto-detect model input size
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import sys
import os

# Try to import OpenCV, set to None if not available (e.g., in cloud environments)
try:
    import cv2
except ImportError:
    cv2 = None

# Add parent directory to path to import scripts
sys.path.append(str(Path(__file__).parent.parent))

from scripts.gradcam import GradCAM, preprocess_image, generate_gradcam_for_sample

# Page configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "resnet50_finetuned.best.h5"
RESULTS_DIR = BASE_DIR / "results"

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    if MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            return model, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "Model file not found. Please train the model first."


# Fixed input size for the model
MODEL_INPUT_SIZE = (256, 256)


def get_model_input_dict(model, img_array):
    """Get the correct input format for the model (handles named inputs)"""
    if hasattr(model, 'input_names') and model.input_names:
        # Model has named inputs - use dictionary
        input_name = model.input_names[0]
        return {input_name: img_array}
    else:
        # Model accepts tensor directly
        return img_array


def predict_image(model, image):
    """Predict glaucoma probability for an image"""
    # Preprocess
    img_array = np.array(image.resize(MODEL_INPUT_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get correct input format
    model_input = get_model_input_dict(model, img_array)
    
    # Predict
    prediction = model.predict(model_input, verbose=0)
    probability = float(prediction[0][0])
    
    return probability


def display_gradcam(model, image, temp_path):
    """Generate and display Grad-CAM visualization"""
    try:
        # Validate image
        if image is None:
            st.error("Error: No image provided.")
            return None, None
        
        # Ensure image is PIL Image
        if not hasattr(image, 'resize'):
            st.error("Error: Invalid image format.")
            return None, None
        
        # Resize image to model input size
        img_resized = image.resize(MODEL_INPUT_SIZE)
        
        # Convert to numpy array and preprocess
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Ensure image data is valid (not all zeros and has correct shape)
        if np.all(img_array == 0):
            st.error("Error: Image data is invalid (all zeros). Please upload a valid image.")
            return None, None
        
        if img_array.shape != (1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3):
            st.error(f"Error: Image shape mismatch. Expected (1, {MODEL_INPUT_SIZE[0]}, {MODEL_INPUT_SIZE[1]}, 3), got {img_array.shape}")
            return None, None
        
        # Create GradCAM instance
        gradcam = GradCAM(model)
        
        # Generate visualization
        heatmap = gradcam.make_gradcam_heatmap(img_array)
        
        # Overlay
        img_orig = np.array(image)
        overlaid = gradcam.overlay_heatmap(img_orig, heatmap)
        
        return heatmap, overlaid
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def main():
    """Main Streamlit app"""
    
    # Title and header
    st.title("👁️ Glaucoma Detection System")
    st.markdown("---")
    st.markdown(
        """
        This application uses a deep learning model (ResNet50) to detect glaucoma from retinal fundus images.
        Upload an image below to get predictions and visualize which regions the model focuses on.
        """
    )
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        
        # Load model
        model, error = load_model()
        
        if model is None:
            st.error(f"⚠️ {error}")
            st.stop()
        else:
            st.success("✓ Model loaded successfully")
        
        st.markdown("---")
        
        # Model info
        st.subheader("Model Details")
        st.write("- **Architecture**: ResNet50 (Transfer Learning)")
        st.write(f"- **Input Size**: {MODEL_INPUT_SIZE[0]}×{MODEL_INPUT_SIZE[1]}×3")
        st.write("- **Output**: Binary Classification (Normal/Glaucoma)")
        
        st.markdown("---")
        
        # Dataset info (if available)
        data_summary_path = BASE_DIR / "processed_data" / "data_summary.json"
        if data_summary_path.exists():
            import json
            with open(data_summary_path) as f:
                summary = json.load(f)
            st.subheader("Dataset Summary")
            st.write(f"**Training**: {summary['training_set']['Total']} images")
            st.write(f"**Test**: {summary['test_set']['Total']} images")
        
        st.markdown("---")
        
        # Evaluation metrics (if available)
        metrics_path = RESULTS_DIR / "metrics.json"
        metrics = None
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.subheader("Model Performance")
            st.write(f"**Accuracy**: {metrics['accuracy']:.3f}")
            st.write(f"**Precision**: {metrics['precision']:.3f}")
            st.write(f"**Recall**: {metrics['recall']:.3f}")
            st.write(f"**F1-Score**: {metrics['f1_score']:.3f}")
            st.write(f"**ROC AUC**: {metrics['roc_auc']:.3f}")
        
        # Store metrics in session state for main area
        if metrics:
            st.session_state['metrics'] = metrics
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        st.markdown("[View Results](./results)")
        st.markdown("[View Report](./report)")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a retinal fundus image for glaucoma detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Prediction button
            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Predict
                    probability = predict_image(model, image)
                    
                    # Store in session state
                    st.session_state['probability'] = probability
                    st.session_state['image'] = image
                    st.session_state['filename'] = uploaded_file.name
                    
                    st.rerun()
        
        # Display prediction results
        if 'probability' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            prob = st.session_state['probability']
            
            # Prediction label
            label = "⚠️ **Glaucoma Detected**" if prob > 0.5 else "✅ **Normal**"
            color = "red" if prob > 0.5 else "green"
            
            st.markdown(f'<h3 style="color: {color};">{label}</h3>', unsafe_allow_html=True)
            
            # Probability bar
            st.progress(prob if prob > 0.5 else (1 - prob))
            st.write(f"**Confidence**: {prob:.1%}" if prob > 0.5 else f"**Confidence**: {(1-prob):.1%}")
            
            # Probability breakdown
            st.write(f"- **Glaucoma Probability**: {prob:.4f}")
            st.write(f"- **Normal Probability**: {1-prob:.4f}")
    
    with col2:
        st.subheader("🎨 Visualization")
        
        if 'probability' in st.session_state:
            # Toggle for Grad-CAM
            show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
            
            if show_gradcam:
                with st.spinner("Generating Grad-CAM visualization..."):
                    # Create temporary file
                    temp_dir = BASE_DIR / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / "uploaded_image.jpg"
                    
                    # Generate Grad-CAM
                    heatmap, overlaid = display_gradcam(
                        model,
                        st.session_state['image'],
                        temp_path
                    )
                    
                    if heatmap is not None and overlaid is not None:
                        # Display original, heatmap, and overlay
                        tab1, tab2, tab3 = st.tabs(["Original", "Heatmap", "Overlay"])
                        
                        with tab1:
                            st.image(st.session_state['image'], use_container_width=True)
                        
                        with tab2:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(heatmap, cmap='jet')
                            ax.axis('off')
                            ax.set_title('Grad-CAM Heatmap', fontsize=14)
                            st.pyplot(fig)
                        
                        with tab3:
                            if cv2 is not None:
                                st.image(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.image(overlaid, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="💾 Download Prediction Report",
                    data=f"Prediction: {label}\nProbability: {prob:.4f}\nFilename: {st.session_state['filename']}",
                    file_name="prediction_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("Enable Grad-CAM to visualize model attention")
        
        else:
            st.info("👈 Upload an image and click 'Predict' to see results")
    
    # Results section (if available)
    st.markdown("---")
    st.subheader("📈 Model Evaluation Metrics")
    
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        # Load metrics if not in session state
        if 'metrics' not in st.session_state:
            import json
            with open(metrics_path) as f:
                st.session_state['metrics'] = json.load(f)
        
        metrics = st.session_state['metrics']
        
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        with col_metrics1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col_metrics2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col_metrics3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col_metrics4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        # Display confusion matrix if available
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.markdown("---")
            st.subheader("Confusion Matrix")
            st.image(str(cm_path), use_container_width=True)
        
        # Display ROC curve if available
        roc_path = RESULTS_DIR / "roc_auc.png"
        if roc_path.exists():
            st.markdown("---")
            st.subheader("ROC Curve")
            st.image(str(roc_path), use_container_width=True)
    else:
        st.info("Run evaluation script to see metrics here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        ⚠️ <strong>Disclaimer</strong>: This model is for research and demonstration purposes only.
        Clinical deployment requires regulatory approval and rigorous prospective evaluation.
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

