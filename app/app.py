#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- app/app.py
Professional Streamlit Web Application

Matches official guide (page 17) with our beautiful UI:
- Upload image → prediction + confidence + probabilities
- Grad-CAM explainability overlay
- Cached model loading for instant response
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SCALE x ODYSSEY", page_icon="🌌", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.main-header { font-size: 3rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem !important; }
.sub-header { font-size: 1.2rem !important; color: #64748b;
    text-align: center; margin-bottom: 2rem !important; }
.metric-card { background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px; padding: 1.5rem; color: white; text-align: center; }
.metric-value { font-size: 2.5rem; font-weight: 700; }
.metric-label { font-size: 0.9rem; opacity: 0.9; }
.footer { text-align: center; color: #94a3b8; font-size: 0.8rem;
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import ModelManager, InferenceResult
from gradcam import explain_image
from dataset import CLASS_NAMES_DISPLAY
from utils import get_device


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str = "checkpoints/best_model.pth"):
    device = str(get_device())  # CUDA > MPS > CPU
    return ModelManager(checkpoint_path=checkpoint_path, device=device)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown('<p class="main-header">🌌 SCALE × ODYSSEY</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Classification of Astronomical Objects</p>',
                unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.subheader("Model")
        device_type = get_device().type  # cuda | mps | cpu
        device_labels = {"cuda": "CUDA", "mps": "Apple MPS", "cpu": "CPU"}
        device = device_labels.get(device_type, device_type.upper())
        icon = "🟢" if device_type in ("cuda", "mps") else "🟡"
        st.markdown(f"{icon} **Device:** {device}")
        if device_type == "cuda":
            st.markdown(f"📟 **GPU:** {torch.cuda.get_device_name(0)}")
        st.markdown("🧠 **Backbone:** EfficientNet-B3")
        st.markdown("📊 **Classes:** 5")
        st.markdown("---")
        st.subheader("📋 Class Reference")
        for i, cls in enumerate(CLASS_NAMES_DISPLAY):
            st.markdown(f"{i+1}. {cls}")
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("**SCALE × ODYSSEY** classifies raw astronomical images into 5 "
                    "celestial categories using deep learning. Built with PyTorch + "
                    "EfficientNet-B3 + Grad-CAM.")


def render_prediction_result(result: InferenceResult):
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{result.class_name}</div>'
                    f'<div class="metric-label">Predicted Class</div></div>', unsafe_allow_html=True)
    with c2:
        color = "#22c55e" if result.confidence > 0.8 else "#f59e0b" if result.confidence > 0.5 else "#ef4444"
        st.markdown(f'<div class="metric-card" style="background: {color};">'
                    f'<div class="metric-value">{result.confidence:.1%}</div>'
                    f'<div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #11998e, #38ef7d);">'
                    f'<div class="metric-value">{result.inference_time_ms:.0f}ms</div>'
                    f'<div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)


def render_probability_chart(result: InferenceResult):
    st.markdown("---")
    st.subheader("📊 Probability Distribution")
    import pandas as pd
    sorted_items = sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame({"Class": [n for n, _ in sorted_items], "Probability": [p for _, p in sorted_items]})
    st.bar_chart(df.set_index("Class"), use_container_width=True)
    st.dataframe(df.assign(Probability=df["Probability"].apply(lambda x: f"{x:.4f}")),
                 use_container_width=True, hide_index=True)


def render_gradcam(image, manager):
    st.markdown("---")
    st.subheader("🔥 Grad-CAM Explainability")
    st.caption("Heatmap shows regions the model focused on. Red = high importance.")
    with st.spinner("Generating Grad-CAM..."):
        try:
            result = explain_image(model=manager.model, image_path=image,
                                   device=manager.device, image_size=manager.image_size)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Image**")
                st.image(image, use_column_width=True)
            with c2:
                st.markdown("**Predicted-Class CAM**")
                st.image(result["predicted_cam"], use_column_width=True)
            if result.get("true_idx") != result["pred_idx"] and "true_cam" in result:
                st.markdown("**True-Class CAM** (for comparison)")
                st.image(result["true_cam"], use_column_width=True)
            st.info(f"Model focused on highlighted regions to classify as **{result['pred_class']}**.")
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
            st.info("Ensure grad-cam is installed: pip install grad-cam")


def render_footer():
    st.markdown('<div class="footer"><p><strong>SCALE × ODYSSEY</strong> — TechOIITGN Hackathon</p>'
                '<p>Built with PyTorch · EfficientNet-B3 · Grad-CAM · Streamlit</p></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    render_header()
    render_sidebar()

    try:
        manager = load_model()
    except FileNotFoundError:
        st.error("⚠️ Model checkpoint not found at `checkpoints/best_model.pth`. "
                 "Train first: `python src/train.py`")
        render_footer()
        return

    uploaded = st.file_uploader("Drag and drop an astronomical image (JPG/PNG)",
                                type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        left, right = st.columns([1, 1])
        with left:
            st.markdown("---")
            st.subheader("🖼️ Uploaded Image")
            st.image(temp_path, use_column_width=True)

        with right:
            with st.spinner("Analyzing..."):
                result = manager.predict(temp_path)
            render_prediction_result(result)
            render_probability_chart(result)

        render_gradcam(temp_path, manager)

        try:
            os.remove(temp_path)
        except OSError:
            pass
    else:
        st.info("👆 Upload an astronomical image to begin classification.")
        st.markdown("---")
        st.subheader("📖 Example Classifications")
        examples = [
            ("Spiral Galaxy", "Spiral arm structure with central bulge"),
            ("Elliptical Galaxy", "Smooth, featureless oval shape"),
            ("Nebula", "Colorful gas and dust clouds"),
            ("Star Cluster", "Dense grouping of bright stars"),
            ("Planetary Object", "Planets, moons, or ring systems"),
        ]
        for col, (name, desc) in zip(st.columns(5), examples):
            with col:
                st.markdown(f"**{name}**")
                st.caption(desc)

    render_footer()


if __name__ == "__main__":
    main()
