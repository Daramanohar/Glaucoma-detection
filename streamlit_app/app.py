"""
Streamlit Web Application for Glaucoma Detection
Provides interactive interface for image upload, prediction, and Grad-CAM visualization
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
    st.warning("OpenCV not available - Grad-CAM overlays will be disabled")

# Add parent directory to path to import scripts
sys.path.append(str(Path(__file__).parent.parent))

# Get base directory - works in both regular Python and Colab/Streamlit
def get_base_dir():
    """Get base directory for Streamlit app"""
    try:
        if '__file__' in globals():
            base = Path(__file__).resolve().parent.parent
            # Check if nested directory structure exists
            nested_path = base / "Glaucoma_detection" / "Glaucoma_detection"
            if (nested_path / "models").exists():
                return nested_path
            return base
    except:
        pass
    
    # Fallback to current working directory or common paths
    cwd = Path(os.getcwd())
    nested_path = cwd / "Glaucoma_detection" / "Glaucoma_detection"
    if (nested_path / "models").exists():
        return nested_path
    if (cwd / "scripts").exists():
        return cwd
    elif (cwd.parent / "scripts").exists():
        return cwd.parent
    
    return cwd

from scripts.gradcam import GradCAM, preprocess_image, generate_gradcam_for_sample
from scripts.rag_retrieval import RAGRetriever, retrieve_for_prediction
from scripts.groq_interface import GroqInterface, check_groq, generate_description
import json

# Page configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
BASE_DIR = get_base_dir()
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


@st.cache_resource
def load_rag_retriever():
    """Load RAG retriever (cached)"""
    try:
        retriever = RAGRetriever()
        retriever.connect()
        return retriever, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def check_groq_status():
    """Check if Groq API is configured (cached)"""
    return check_groq()


def predict_image(model, image):
    """Predict glaucoma probability for an image"""
    # Get model input size dynamically
    input_shape = model.inputs[0].shape[1:3]
    target_size = (int(input_shape[0]), int(input_shape[1]))
    
    # Preprocess
    img_array = np.array(image.resize(target_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    probability = float(prediction[0][0])
    
    return probability


def display_gradcam(model, image, temp_path):
    """Generate and display Grad-CAM visualization"""
    try:
        # Save image temporarily
        image.save(temp_path)
        
        # Create GradCAM instance
        gradcam = GradCAM(model)
        
        # Get model input size dynamically
        input_shape = model.inputs[0].shape[1:3]
        target_size = (int(input_shape[0]), int(input_shape[1]))
        
        # Generate visualization
        img_array, _ = preprocess_image(str(temp_path), target_size=target_size)
        heatmap = gradcam.make_gradcam_heatmap(img_array)
        
        # Overlay
        img_orig = np.array(image)
        overlaid = gradcam.overlay_heatmap(img_orig, heatmap)
        
        return heatmap, overlaid
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None, None


def main():
    """Main Streamlit app"""
    
    # Load resources OUTSIDE sidebar so they're available throughout
    model, error = load_model()
    if model is None:
        st.error(f"⚠️ {error}")
        st.stop()
    
    retriever, rag_error = load_rag_retriever()
    groq_ready = check_groq_status()
    
    # Title and header
    st.title("👁️ Glaucoma Detection System")
    
    # Clear data button
    col_title, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Clear Data", help="Clear all session data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
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
        
        # Show status
        st.success("[OK] Model loaded successfully")
        
        st.markdown("---")
        
        # RAG System
        if retriever:
            st.success("[OK] RAG system ready")
        else:
            st.warning("[WARNING] RAG unavailable")
        
        # Groq Status
        if groq_ready:
            st.success("[OK] Groq + Llama3 ready")
        else:
            st.warning("[WARNING] Groq API not configured")
        
        st.markdown("---")
        
        # Model info
        st.subheader("Model Details")
        st.write("- **Architecture**: ResNet50 (Transfer Learning)")
        st.write("- **Input Size**: 256×256×3")
        st.write("- **Output**: Binary Classification (Normal/Glaucoma)")
        
        st.markdown("---")
        
        # RAG Info
        st.subheader("RAG System")
        st.write("- **Database**: PostgreSQL + pgvector")
        st.write("- **Documents**: 13 medical chunks")
        st.write("- **Embeddings**: 384-dim vectors")
        st.write("- **LLM**: Llama3-70B (Groq)")
        
        st.markdown("---")
        
        # Dataset info (if available)
        data_summary_path = BASE_DIR / "processed_data" / "data_summary.json"
        if data_summary_path.exists():
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
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Prediction button
            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Analyzing image with AI..."):
                    # Predict
                    probability = predict_image(model, image)
                    
                    # Store in session state
                    st.session_state['probability'] = probability
                    st.session_state['image'] = image
                    st.session_state['filename'] = uploaded_file.name
                    
                    # Generate Grad-CAM and retrieve RAG context
                    try:
                        temp_dir = BASE_DIR / "temp"
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / "uploaded_image.jpg"
                        image.save(temp_path)
                        
                        # Grad-CAM
                        gradcam = GradCAM(model)
                        img_array, _ = preprocess_image(str(temp_path), target_size=(256, 256))
                        heatmap = gradcam.make_gradcam_heatmap(img_array)
                        img_orig = np.array(image)
                        overlaid = gradcam.overlay_heatmap(img_orig, heatmap)
                        
                        st.session_state['heatmap'] = heatmap
                        st.session_state['overlaid'] = overlaid
                        
                        # RAG Retrieval
                        if retriever:
                            rag_results = retrieve_for_prediction(
                                prediction_prob=probability,
                                gradcam_keywords=["optic disc", "cup", "rim"],
                                top_k=3
                            )
                            st.session_state['rag_results'] = rag_results
                        
                    except Exception as e:
                        st.session_state['heatmap'] = None
                        st.session_state['overlaid'] = None
                        st.session_state['rag_results'] = []
                    
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
    
    # RAG + Mistral Description Section
    if 'probability' in st.session_state and 'rag_results' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Detailed Patient Information")
        
        # Toggle for detailed description
        show_description = st.checkbox("Generate detailed information with AI", value=True)
        
        if show_description:
            if 'description' not in st.session_state:
                with st.spinner("Generating detailed description with Llama3..."):
                    if groq_ready and st.session_state.get('rag_results'):
                        try:
                            # Generate detailed description
                            description = generate_description(
                                prediction_prob=st.session_state['probability'],
                                rag_context=st.session_state['rag_results'],
                                gradcam_keywords=["optic disc", "cup-to-disc ratio", "rim thinning"]
                            )
                            st.session_state['description'] = description
                        except Exception as e:
                            st.session_state['description'] = f"Error: {str(e)}"
                    else:
                        st.session_state['description'] = "⚠️ Groq API not configured or RAG unavailable"
            
            # Display description
            if st.session_state.get('description'):
                st.markdown("#### AI-Generated Summary:")
                st.info(st.session_state['description'])
                
                # Show RAG sources
                with st.expander("📚 View Source Information"):
                    if st.session_state.get('rag_results'):
                        for i, result in enumerate(st.session_state['rag_results'], 1):
                            st.markdown(f"**{i}. {result['title']}**")
                            st.write(f"Relevance: {result['similarity']:.2%}")
                            st.write(f"Source: {result['source']}")
                            st.caption(result['text'][:200] + "...")
                            st.markdown("---")
        
    with col2:
        st.subheader("🎨 Visualization")
        
        if 'probability' in st.session_state:
            # Toggle for Grad-CAM
            show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
            
            if show_gradcam:
                if 'overlaid' in st.session_state and st.session_state['overlaid'] is not None:
                    # Display pre-computed Grad-CAM
                    tab1, tab2, tab3 = st.tabs(["Original", "Heatmap", "Overlay"])
                    
                    with tab1:
                        st.image(st.session_state['image'], width='stretch')
                    
                    with tab2:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(st.session_state['heatmap'], cmap='jet')
                        ax.axis('off')
                        ax.set_title('Grad-CAM Heatmap', fontsize=14)
                        st.pyplot(fig)
                    
                    with tab3:
                        if cv2 is not None:
                            st.image(cv2.cvtColor(st.session_state['overlaid'], cv2.COLOR_BGR2RGB), width='stretch')
                        else:
                            st.image(st.session_state['overlaid'], width='stretch')
                    
                    # Download button
                    label = "⚠️ Glaucoma Detected" if prob > 0.5 else "✅ Normal"
                    st.download_button(
                        label="💾 Download Prediction Report",
                        data=f"Prediction: {label}\nProbability: {prob:.4f}\nFilename: {st.session_state['filename']}",
                        file_name="prediction_report.txt",
                        mime="text/plain"
                    )
                else:
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
                                st.image(st.session_state['image'], width='stretch')
                            
                            with tab2:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.imshow(heatmap, cmap='jet')
                                ax.axis('off')
                                ax.set_title('Grad-CAM Heatmap', fontsize=14)
                                st.pyplot(fig)
                            
                            with tab3:
                                if cv2 is not None:
                                    st.image(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB), width='stretch')
                                else:
                                    st.image(overlaid, width='stretch')
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
            st.image(str(cm_path), width='stretch')
        
        # Display ROC curve if available
        roc_path = RESULTS_DIR / "roc_auc.png"
        if roc_path.exists():
            st.markdown("---")
            st.subheader("ROC Curve")
            st.image(str(roc_path), width='stretch')
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

