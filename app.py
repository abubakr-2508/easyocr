import streamlit as st
import warnings
import sys
import os

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set environment variables for better compatibility
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['HF_HUB_CACHE'] = '/tmp/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only

# Page config
st.set_page_config(
    page_title="YOLO + OCR Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if this is the initial load (to prevent repeated imports during reruns)
if 'import_status' not in st.session_state:
    st.session_state.import_status = 'checking'
    
    # Import check with timeout
    import_errors = []
    
    try:
        import cv2
        cv2_available = True
    except ImportError as e:
        cv2_available = False
        import_errors.append(f"OpenCV: {e}")
    
    try:
        import numpy as np
        import easyocr
        from PIL import Image
        import torch
        import urllib.request
        from pathlib import Path
        import tempfile
        import time
        import io
        core_imports_available = True
    except ImportError as e:
        core_imports_available = False
        import_errors.append(f"Core libraries: {e}")
    
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError as e:
        yolo_available = False
        import_errors.append(f"YOLO: {e}")
    
    # Check imports
    if not (cv2_available and core_imports_available and yolo_available):
        st.error("Import Error")
        st.error("Some libraries failed to import:")
        
        for error in import_errors:
            st.error(f"• {error}")
        
        st.markdown("""
        ### Deployment Issues Detected:
        
        If you're seeing this on Render.com, try these solutions:
        
        1. **Use the fixed requirements.txt** (see below)
        2. **Reduce memory usage** by using CPU-only PyTorch
        3. **Consider upgrading to paid tier** for more resources
        
        ```
        # Lightweight requirements.txt for Render:
        streamlit>=1.28.0
        opencv-python-headless==4.8.1.78
        numpy>=1.24.0,<2.0.0
        Pillow>=10.0.0
        torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
        torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
        ultralytics==8.0.196
        easyocr==1.6.2
        python-bidi>=0.4.2
        urllib3>=1.26.0,<2.0.0
        ```
        """)
        st.stop()
    
    st.session_state.import_status = 'success'

# Monkey-patch for Pillow
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Custom CSS (simplified)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>YOLO + OCR Detection System</h1>
    <p>Optimized for Render.com deployment</p>
</div>
""", unsafe_allow_html=True)

def setup_pytorch_compatibility():
    """Setup PyTorch compatibility for deployment"""
    try:
        torch_version = torch.__version__
        st.info(f"PyTorch version: {torch_version}")
        
        # Simple compatibility check
        if hasattr(torch, 'serialization'):
            try:
                from torch.serialization import add_safe_globals
                from collections import OrderedDict
                add_safe_globals([OrderedDict])
                st.success("PyTorch compatibility configured")
            except Exception as e:
                st.warning(f"Compatibility warning: {e}")
        
        return True
        
    except Exception as e:
        st.error(f"PyTorch setup error: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_models():
    """Load models with aggressive optimization for Render"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Setup compatibility
        status_text.text("Setting up PyTorch...")
        progress_bar.progress(10)
        
        if not setup_pytorch_compatibility():
            st.warning("Continuing with compatibility issues...")
        
        # Use smaller YOLO model for deployment
        status_text.text("Loading lightweight YOLO model...")
        progress_bar.progress(30)
        
        # Try to use the smallest model first
        model_name = "yolov8n.pt"  # Nano model - smallest
        
        try:
            # Direct loading with error handling
            model = YOLO(model_name)
            model.to('cpu')  # Ensure CPU-only
            st.success("YOLO model loaded (CPU-only)")
        except Exception as e:
            st.error(f"YOLO loading failed: {e}")
            # Fallback: Create a dummy model for demo
            model = None
        
        progress_bar.progress(60)
        
        # Load OCR with reduced settings
        status_text.text("Loading OCR model...")
        progress_bar.progress(80)
        
        try:
            # Use only English, no GPU
            reader = easyocr.Reader(['en'], gpu=False, verbose=False, 
                                  download_enabled=True, detector=True, recognizer=True)
            st.success("OCR model loaded")
        except Exception as e:
            st.error(f"OCR loading failed: {e}")
            reader = None
        
        progress_bar.progress(100)
        status_text.text("Models loaded successfully!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return model, reader
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"Model loading failed: {e}")
        
        # Provide deployment-specific troubleshooting
        st.markdown("""
        ### Deployment Issues:
        
        **For Render.com users:**
        1. Free tier has limited memory (512MB)
        2. AI models are memory-intensive
        3. Consider upgrading to paid tier
        4. Or use model-as-a-service APIs instead
        
        **Quick fixes:**
        - Use smaller models
        - Reduce batch sizes
        - Enable model caching
        - Use CPU-only versions
        """)
        
        raise e

def process_image_safe(image, model, reader, ocr_settings):
    """Process image with error handling for deployment"""
    
    try:
        if model is None:
            st.warning("YOLO model not available - showing original image")
            result_image = image
        else:
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # YOLO Detection with timeout
            try:
                results = model(img_cv, verbose=False)
                annotated_frame = results[0].plot()
                result_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            except Exception as yolo_error:
                st.warning(f"YOLO processing failed: {yolo_error}")
                result_image = image
        
        # OCR Detection
        detected_text = ""
        ocr_results = []
        
        if reader is not None:
            try:
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                ocr_results = reader.readtext(img_cv, paragraph=False)
                
                for detection in ocr_results:
                    if len(detection) >= 3 and detection[2] > ocr_settings.get('confidence_threshold', 0.5):
                        detected_text += f"{detection[1]} [Conf: {detection[2]:.2f}]\n"
                        
            except Exception as ocr_error:
                st.warning(f"OCR processing failed: {ocr_error}")
        else:
            st.warning("OCR model not available")
        
        return result_image, detected_text, ocr_results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image, "", []

# Sidebar
st.sidebar.title("Settings")

# System info (simplified)
with st.sidebar.expander("System Info"):
    st.write(f"Python: {sys.version.split()[0]}")
    if 'torch' in sys.modules:
        st.write(f"PyTorch: {torch.__version__}")

# Model loading with deployment optimization
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    with st.sidebar:
        st.info("⚡ Optimized for Render.com deployment")
        if st.button("Load Models", use_container_width=True, type="primary"):
            try:
                with st.spinner("Loading models... This may take 2-3 minutes on first load"):
                    model, reader = load_models()
                    st.session_state.yolo_model = model
                    st.session_state.ocr_reader = reader
                    st.session_state.models_loaded = True
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.info("This might be due to memory limitations on free hosting. Try a paid tier or local deployment.")

if st.session_state.models_loaded:
    st.sidebar.success("Models loaded successfully!")
    
    # OCR Settings (simplified)
    st.sidebar.subheader("OCR Settings")
    ocr_settings = {
        'confidence_threshold': st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    }
    
    # Main interface
    st.subheader("Upload Image for Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for object detection and OCR"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.info(f"File: {uploaded_file.name} | Size: {uploaded_file.size} bytes")
    
    with col2:
        st.markdown("### Results")
        
        if uploaded_file and st.button("Process Image", use_container_width=True, type="primary"):
            with st.spinner("Processing image..."):
                result_image, detected_text, ocr_results = process_image_safe(
                    image, 
                    st.session_state.yolo_model, 
                    st.session_state.ocr_reader, 
                    ocr_settings
                )
                
                st.session_state.result_image = result_image
                st.session_state.detected_text = detected_text
                st.session_state.ocr_results = ocr_results
    
    # Results display
    if hasattr(st.session_state, 'result_image'):
        st.markdown("---")
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Processed Image")
            st.image(st.session_state.result_image, use_container_width=True)
        
        with col2:
            st.markdown("### Detected Text")
            
            if st.session_state.detected_text.strip():
                st.text_area("Detected Text", st.session_state.detected_text, height=200)
            else:
                st.info("No text detected in the image.")

else:
    # Welcome page optimized for deployment
    st.markdown("""
    ## Welcome to YOLO + OCR Detection System
    
    **🚀 Optimized for Render.com deployment**
    
    ### Features:
    - Upload images for analysis
    - Object detection with YOLOv8 (CPU-optimized)
    - Text recognition with EasyOCR
    - Lightweight deployment-ready version
    
    ### Deployment Notes:
    - First load takes 2-3 minutes (model download)
    - Models are cached after initial load  
    - CPU-only processing for stability
    - Memory-optimized for free hosting
    
    **Click "Load Models" in the sidebar to begin!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Optimized for Render.com | YOLO + EasyOCR
</div>
""", unsafe_allow_html=True)
