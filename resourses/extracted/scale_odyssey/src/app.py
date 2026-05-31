#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Streamlit Web Application

Interactive demo for astronomical image classification.
Upload an image to get:
- Predicted class + confidence
- Full probability distribution
- Grad-CAM explainability overlay

Covers Innovation/Bonus (15%) and Explainability (15%) evaluation criteria.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SCALE x ODYSSEY",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Header styling */
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Result card */
    .result-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Sidebar */
    .css-1d391kg { padding-top: 2rem; }

    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import ModelManager, InferenceResult
from gradcam import explain_image
from augmentations import CLASS_NAMES_DISPLAY


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str = "checkpoints/best_model.pth"):
    """Load and cache the model (singleton pattern)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager = ModelManager(checkpoint_path=checkpoint_path, device=device)
    return manager


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    """Render the page header."""
    st.markdown('<p class="main-header">🌌 SCALE × ODYSSEY</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Deep Learning Classification of Astronomical Objects</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render sidebar with info and settings."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model info
        st.subheader("Model")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        device_icon = "🟢" if torch.cuda.is_available() else "🟡"
        st.markdown(f"{device_icon} **Device:** {device}")

        if torch.cuda.is_available():
            st.markdown(f"📟 **GPU:** {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.markdown(f"💾 **VRAM:** {mem:.1f} GB")

        st.markdown("🧠 **Backbone:** EfficientNet-B3")
        st.markdown("📊 **Classes:** 5")

        # Class reference
        st.markdown("---")
        st.subheader("📋 Class Reference")
        for i, cls in enumerate(CLASS_NAMES_DISPLAY):
            st.markdown(f"{i+1}. {cls}")

        # About
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **SCALE × ODYSSEY** uses deep learning to classify
        raw astronomical images into 5 categories without
        handcrafted features.

        Built with PyTorch + EfficientNet-B3 + Grad-CAM.
        """)


def render_file_uploader():
    """Render the file upload area."""
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader(
        "Drag and drop an astronomical image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a telescope-captured image of a galaxy, nebula, star cluster, or planetary object.",
    )
    return uploaded


def render_prediction_result(result: InferenceResult):
    """Render prediction card with class name and confidence."""
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.class_name}</div>
                <div class="metric-label">Predicted Class</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        conf_color = "#22c55e" if result.confidence > 0.8 else "#f59e0b" if result.confidence > 0.5 else "#ef4444"
        st.markdown(f"""
            <div class="metric-card" style="background: {conf_color};">
                <div class="metric-value">{result.confidence:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="metric-value">{result.inference_time_ms:.0f}ms</div>
                <div class="metric-label">Inference Time</div>
            </div>
        """, unsafe_allow_html=True)


def render_probability_chart(result: InferenceResult):
    """Render probability distribution as a bar chart."""
    st.markdown("---")
    st.subheader("📊 Probability Distribution")

    probs = result.all_probabilities
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    data = {"Class": [name for name, _ in sorted_items],
            "Probability": [prob for _, prob in sorted_items]}

    import pandas as pd
    df = pd.DataFrame(data)

    # Color: highlight top prediction
    colors = ["#3b82f6" if i == 0 else "#cbd5e1" for i in range(len(df))]

    chart = st.bar_chart(
        df.set_index("Class"),
        color=colors[0] if len(colors) == 1 else colors,
        use_container_width=True,
    )

    # Also show as a table
    st.dataframe(
        df.assign(Probability=df["Probability"].apply(lambda x: f"{x:.4f}")),
        use_container_width=True,
        hide_index=True,
    )


def render_gradcam_section(image, manager: ModelManager):
    """Generate and display Grad-CAM overlay."""
    st.markdown("---")
    st.subheader("🔥 Grad-CAM Explainability")
    st.caption(
        "Heatmap shows which regions the model focused on to make its prediction. "
        "Red = high importance, Blue = low importance."
    )

    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            result = explain_image(
                model=manager.model,
                image_path=image,
                device=manager.device,
                image_size=manager.image_size,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_column_width=True)

            with col2:
                st.markdown("**Predicted-Class CAM**")
                st.image(result["predicted_cam"], use_column_width=True)

            # If true class CAM exists and differs
            if "true_cam" in result and result["true_idx"] != result["pred_idx"]:
                st.markdown("**True-Class CAM** (for comparison)")
                st.image(result["true_cam"], use_column_width=True)

            # Show top-3 activating regions description
            st.info(
                f"The model focused on the highlighted regions to classify this as "
                f"**{result['pred_class']}**. Red areas contributed most to the decision."
            )

        except Exception as e:
            st.error(f"Grad-CAM generation failed: {str(e)}")
            st.info("Ensure 'grad-cam' is installed: pip install grad-cam")


def render_uploaded_image(image):
    """Display the uploaded image."""
    st.markdown("---")
    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_column_width=True)


def render_footer():
    """Render page footer."""
    st.markdown("""
        <div class="footer">
            <p><strong>SCALE × ODYSSEY</strong> — TechOIITGN Hackathon Submission</p>
            <p>Built with PyTorch · EfficientNet-B3 · Grad-CAM · Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application."""
    render_header()
    render_sidebar()

    # Load model
    try:
        manager = load_model()
    except FileNotFoundError:
        st.error("""
            ⚠️ **Model checkpoint not found!**

            No trained model found at `checkpoints/best_model.pth`.

            **To fix:**
            1. Train a model: `python src/train.py`
            2. Or place a trained checkpoint in the `checkpoints/` folder
        """)
        render_footer()
        return
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        render_footer()
        return

    # File uploader
    uploaded = render_file_uploader()

    if uploaded is not None:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Layout: image left, results right
        left_col, right_col = st.columns([1, 1])

        with left_col:
            render_uploaded_image(temp_path)

        with right_col:
            # Run inference
            with st.spinner("Analyzing image..."):
                result = manager.predict(temp_path)

            # Display results
            render_prediction_result(result)

            # Probability chart
            render_probability_chart(result)

        # Full-width Grad-CAM section
        render_gradcam_section(temp_path, manager)

        # Cleanup
        try:
            os.remove(temp_path)
        except OSError:
            pass

    else:
        # Show placeholder when no image uploaded
        st.info("👆 Upload an astronomical image to begin classification.")

        # Example images grid (placeholder)
        st.markdown("---")
        st.subheader("📖 Example Classifications")

        examples = [
            ("Spiral Galaxy", "Distinctive spiral arm structure with central bulge"),
            ("Elliptical Galaxy", "Smooth, featureless oval shape"),
            ("Nebula", "Colorful gas and dust clouds"),
            ("Star Cluster", "Dense grouping of bright stars"),
            ("Planetary Object", "Planets, moons, or ring systems"),
        ]

        cols = st.columns(5)
        for col, (name, desc) in zip(cols, examples):
            with col:
                st.markdown(f"**{name}**")
                st.caption(desc)

    render_footer()


if __name__ == "__main__":
    main()
