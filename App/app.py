import streamlit as st
import cv2
import numpy as np
import tempfile
import time
# integrating modules
from db_manager import AuditLogger
from model_handler import DeepfakeModel

# --- AYARLAR ---
MODEL_PATH = 'D:\\xx\\best_deepfake_model.keras' # path of the model
DB_NAME = "logs.db"

# --- INIT ---
st.set_page_config(
    page_title="Deepfake Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# caching the model (dont repeat the model loading)
@st.cache_resource
def get_model():
    return DeepfakeModel(MODEL_PATH)

@st.cache_resource
def get_logger():
    return AuditLogger(DB_NAME)

ai_engine = get_model()
logger = get_logger()

# visualization
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    h1 {
        color: #00ffbf;
        text-shadow: 0 0 20px rgba(0, 255, 191, 0.3);
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #e0e0e0;
        font-weight: 600;
    }
    
    .stButton>button {
        color: white;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(0, 255, 191, 0.1) 0%, rgba(0, 191, 255, 0.1) 100%);
        border-left: 4px solid #00ffbf;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffbf 0%, #00bfff 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(0, 255, 191, 0.3);
        border-radius: 15px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 255, 191, 0.6);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ffbf 0%, #00bfff 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# --- ARAYÜZ (SIDEBAR) ---
st.sidebar.markdown("<h1 style='text-align: center; color: #00ffbf;'>🛡️ Menu</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox(
    "Select Mode:",
    ["🏠 Home", "🖼️ Image Analysis", "🎥 Video Analysis"],
    help="Choose the analysis mode you want to use"
)

st.sidebar.markdown("---")

# System Status Card
status_color = "green" if ai_engine.model else "red"
status_text = "Model Loading Successfull" if ai_engine.model else "Model Error"
status_emoji = "✅" if ai_engine.model else "❌"

st.sidebar.markdown(f"""
<div class='info-card'>
    <h3 style='color: #00ffbf; margin-top: 0;'>📊 System Status</h3>
    <p style='color: {status_color}; font-size: 18px; font-weight: 600;'>{status_emoji} {status_text}</p>
    <p style='color: #888; font-size: 12px; margin-top: 10px;'>Model: EfficientNet-B0</p>
    <p style='color: #888; font-size: 12px;'>Database: SQLite</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 10px;'>
    <p style='color: #888; font-size: 12px; margin: 0;'>Modular Architecture v2.0</p>
    <p style='color: #888; font-size: 12px; margin: 5px 0 0 0;'>Backend Separated</p>
    <p style='color: #888; font-size: 12px; margin: 5px 0 0 0;'>Database Integrated</p>
</div>
""", unsafe_allow_html=True)

# main page
if app_mode == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🛡️ Deepfake Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 18px;'>AI-Powered Manipulation Detection Platform</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color: #00ffbf; margin: 0;'>🎯</h2>
            <h3 style='margin: 10px 0;'>High Accuracy</h3>
            <p style='color: #888;'>Advanced deep learning model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color: #00ffbf; margin: 0;'>⚡</h2>
            <h3 style='margin: 10px 0;'>Fast Analysis</h3>
            <p style='color: #888;'>Real-time detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color: #00ffbf; margin: 0;'>🔐</h2>
            <h3 style='margin: 10px 0;'>Audit Logging</h3>
            <p style='color: #888;'>Complete transaction history</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-box'>
        <h2 style='color: #00ffbf; margin-top: 0;'>🏗️ Modular Architecture v2.0</h2>
        <p style='line-height: 1.8;'>This system runs on a separated logic layer with clean code principles:</p>
        <ul style='line-height: 2;'>
            <li><strong>UI Layer:</strong> Streamlit interface (app.py) - Handles all user interactions</li>
            <li><strong>AI Layer:</strong> TensorFlow/Keras engine (model_handler.py) - Deep learning predictions</li>
            <li><strong>Data Layer:</strong> SQLite database (db_manager.py) - Audit logging and analytics</li>
        </ul>
        <p style='margin-top: 15px; color: #888;'>Each layer is independent, making the system maintainable, testable, and scalable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h2 style='color: #00ffbf; margin-top: 0;'>🚀 How to Use</h2>
        <ol style='line-height: 2;'>
            <li><strong>Select mode</strong> from the left sidebar (Image or Video)</li>
            <li><strong>Upload your file</strong> - JPG, PNG, or MP4 format</li>
            <li><strong>Click analyze</strong> and wait for AI processing</li>
            <li><strong>Review results</strong> with detailed graphs and confidence scores</li>
            <li><strong>Check logs</strong> - All transactions are automatically saved to database</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
        <h3 style='margin-top: 0;'>⚠️ Important Notes</h3>
        <ul>
            <li>Use high-resolution images for best results</li>
            <li>Video analysis may take time depending on your CPU</li>
            <li>All analysis results are logged in the database</li>
            <li>System requires face detection to work properly</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# image analysing
elif app_mode == "🖼️ Image Analysis":
    st.markdown("<h1 style='text-align: center;'>🖼️ Image Deepfake Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Detect manipulations in your images</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload an image (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(frame_rgb, use_container_width=True)
            st.markdown(f"**Size:** {frame_rgb.shape[1]} × {frame_rgb.shape[0]} pixels")
        
        with col2:
            st.markdown("### 🔍 Analysis Result")
            result_placeholder = st.empty()
            
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_btn = st.button("🚀 Analyze", use_container_width=True)
        
        if analyze_btn:
            with st.spinner('🤖 AI Engine Running...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Predicting in one line
                result_frame, probs, text = ai_engine.predict(frame_rgb)
                # --------------------------
                
                with col2:
                    result_placeholder.image(result_frame, use_container_width=True)
                    
                    if probs is not None:
                        class_names = ai_engine.class_names
                        class_idx = np.argmax(probs)
                        confidence = probs[class_idx] * 100
                        final_res = class_names[class_idx]
                        
                        if final_res == 'Original':
                            st.success(f"✅ **Result:** {text}")
                            st.balloons()
                        else:
                            st.error(f"⚠️ **Result:** {text}")
                        
                        # LOGGING
                        logger.log_transaction(uploaded_file.name, uploaded_file.getvalue(), "Image", final_res, confidence)
                        
                        
                        st.markdown("---")
                        st.markdown("### 📊 Probability Distribution")
                        
                        chart_data = dict(zip(class_names, probs * 100))
                        st.bar_chart(chart_data)
                        
                        st.markdown("### 📈 Detailed Scores")
                        for i, (name, prob) in enumerate(zip(class_names, probs)):
                            emoji = "✅" if name == "Original" else "❌"
                            st.markdown(f"{emoji} **{name}:** {prob*100:.2f}%")
                            st.progress(float(prob))
                    else:
                        st.warning("⚠️ No face detected in the image. Please try another image.")
  

# video analysing
elif app_mode == "🎥 Video Analysis":
    st.markdown("<h1 style='text-align: center;'>🎥 Real-Time Video Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Analyze all frames in your video</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
        ⚠️ <strong>Note:</strong> Video analysis processes frame by frame and may take time depending on your CPU speed. 
        Lower resolution videos will process faster.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "📁 Upload a video (MP4, AVI)",
        type=["mp4", "avi"],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_bytes)
        cap = cv2.VideoCapture(tfile.name)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        st.markdown(f"""
        <div class='info-card'>
            <strong>📹 Video Information</strong><br>
            Total Frames: {total_frames} | FPS: {fps} | Duration: {duration:.1f} seconds
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("### 🎬 Video Preview")
            st_frame = st.empty()
        
        with col2:
            st.markdown("### 📊 Live Analysis")
            st_chart = st.empty()
            st_text = st.empty()
            st_confidence = st.empty()
            
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            start_button = st.button("▶️ Start Analysis", use_container_width=True)
        with col_btn3:
            stop_button = st.button("⏹️ Stop", use_container_width=True)
        
        progress_placeholder = st.empty()
        
        if start_button:
            total_probs = np.zeros(len(ai_engine.class_names))
            detected_count = 0
            frame_count = 0
            
            while cap.isOpened():
                if stop_button:
                    st.warning("⏸️ Analysis stopped by user.")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predicting in one line
                processed_frame, probs, text = ai_engine.predict(frame_rgb)
                # --------------------------
                
                st_frame.image(processed_frame, channels="RGB", use_container_width=True)
                
                if probs is not None:
                    chart_data = dict(zip(ai_engine.class_names, probs * 100))
                    st_chart.bar_chart(chart_data)
                    
                    if "Original" in text:
                        st_text.success(f"✅ {text}")
                    else:
                        st_text.error(f"⚠️ {text}")
                    
                    confidence = np.max(probs) * 100
                    st_confidence.metric("Confidence Score", f"{confidence:.1f}%")
                    
                    total_probs += probs
                    detected_count += 1
                
                frame_count += 1
                progress = min(frame_count / total_frames, 1.0)
                progress_placeholder.progress(progress, text=f"Processed Frames: {frame_count}/{total_frames}")
            
            if detected_count > 0:
                avg_probs = total_probs / detected_count
                final_idx = np.argmax(avg_probs)
                final_res = ai_engine.class_names[final_idx]
                final_conf = avg_probs[final_idx] * 100
                
                # LOGGING
                logger.log_transaction(uploaded_video.name, video_bytes, "Video", final_res, final_conf)
                
                st.success(f"✅ Video Analysis Completed! Final Result: **{final_res}** ({final_conf:.2f}%)")
                
                
                st.markdown("### 📊 Final Average Probabilities")
                final_chart_data = dict(zip(ai_engine.class_names, avg_probs * 100))
                st.bar_chart(final_chart_data)
            else:
                st.warning("⚠️ No faces detected in the video.")
            
            cap.release()
