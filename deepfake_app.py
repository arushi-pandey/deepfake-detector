import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD MODEL
# =====================================================
IMG_SIZE = 128
model = load_model("model/deepfake_model.h5")

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    background: linear-gradient(135deg,#050816,#0a0f2c,#111827);
    color: white;
    font-family: Arial, sans-serif;
}

.main-title {
    text-align:center;
    font-size:56px;
    font-weight:800;
    color:#00ffb3;
    text-shadow:0 0 18px rgba(0,255,179,0.6);
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:25px;
}

.card {
    background: rgba(255,255,255,0.06);
    padding:25px;
    border-radius:22px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 8px 30px rgba(0,0,0,0.35);
}

.real-box {
    background: linear-gradient(135deg,#065f46,#10b981);
    padding:18px;
    border-radius:16px;
    font-size:24px;
    font-weight:700;
    text-align:center;
}

.fake-box {
    background: linear-gradient(135deg,#7f1d1d,#ef4444);
    padding:18px;
    border-radius:16px;
    font-size:24px;
    font-weight:700;
    text-align:center;
}

.warn-box {
    background: linear-gradient(135deg,#78350f,#f59e0b);
    padding:18px;
    border-radius:16px;
    font-size:24px;
    font-weight:700;
    text-align:center;
}

.metric-box {
    background:#111827;
    padding:14px;
    border-radius:14px;
    text-align:center;
    margin-top:10px;
}

.history-box {
    background:#111827;
    padding:12px;
    border-radius:12px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='main-title'>🧠 DeepFake Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI Powered Fake Image Detection System</div>", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Dashboard")

show_history = st.sidebar.button("📜 Show History")

if st.sidebar.button("🗑 Delete History"):
    st.session_state.history = []
    st.sidebar.success("History Cleared")

st.sidebar.info("""
Features:
✅ Public Access  
✅ Live Camera Detection  
✅ Upload Detection  
✅ Detection History  
✅ Premium UI
""")

# =====================================================
# HELPERS
# =====================================================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img

def predict_upload(img):
    pred = model.predict(preprocess(img), verbose=0)[0][0]

    real = float(pred)
    fake = 1 - real

    if fake > 0.5:
        return "FAKE", fake, real, fake
    else:
        return "REAL", real, real, fake

def predict_live(img):
    pred = model.predict(preprocess(img), verbose=0)[0][0]

    fake = float(pred)
    real = 1 - fake

    if fake >= 0.65:
        return "FAKE", fake, real, fake
    elif fake <= 0.35:
        return "REAL", real, real, fake
    else:
        return "UNCERTAIN", max(real, fake), real, fake

def save_history(mode, label, confidence):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "mode": mode.upper(),
        "label": label,
        "confidence": f"{confidence*100:.1f}%"
    })

def show_result(img, mode="upload"):
    col1, col2 = st.columns([1.2,1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if mode == "camera":
            label, conf, real, fake = predict_live(img)
        else:
            label, conf, real, fake = predict_upload(img)

        save_history(mode, label, conf)

        if label == "REAL":
            st.markdown("<div class='real-box'>✅ REAL IMAGE</div>", unsafe_allow_html=True)
        elif label == "FAKE":
            st.markdown("<div class='fake-box'>❌ FAKE IMAGE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>⚠️ UNCERTAIN</div>", unsafe_allow_html=True)

        st.progress(int(conf * 100))
        st.markdown(f"<div class='metric-box'>Confidence: <b>{conf*100:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>Real Score: <b>{real:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'>Fake Score: <b>{fake:.2f}</b></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# HISTORY
# =====================================================
if show_history:
    st.subheader("📜 Detection History")

    if len(st.session_state.history) == 0:
        st.info("No history available.")
    else:
        for item in st.session_state.history:
            st.markdown(
                f"<div class='history-box'>🕒 {item['time']} | {item['mode']} | {item['label']} | {item['confidence']}</div>",
                unsafe_allow_html=True
            )

# =====================================================
# LIVE CAMERA (SMALLER SIZE)
# =====================================================
st.markdown("---")
st.subheader("📷 Live Camera Detection")

left, center, right = st.columns([1,2,1])

with center:
    camera = st.camera_input("Take Live Picture")

if camera is not None:
    data = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    img = cv2.imdecode(data, 1)

    with st.spinner("Scanning Live Image..."):
        show_result(img, mode="camera")

# =====================================================
# UPLOAD
# =====================================================
st.markdown("---")
st.subheader("📤 Upload Image Detection")

files = st.file_uploader(
    "Upload Images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if files:
    for file in files:
        data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)

        with st.spinner("Analyzing Uploaded Image..."):
            show_result(img, mode="upload")

        st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;'>🚀 Final College AI Project | Designed by Arushi</div>",
    unsafe_allow_html=True
)