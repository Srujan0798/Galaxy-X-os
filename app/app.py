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
import tempfile
from contextlib import contextmanager
from pathlib import Path

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
# Lazy Imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Lazy-load heavy ML modules inside functions so importing this file for tests
# does not immediately load PyTorch + BLIP + Grad-CAM.
_manager = None


def _get_manager(checkpoint_path: str = "checkpoints/best_model.pth"):
    global _manager
    if _manager is None:
        from inference import ModelManager
        from utils import get_device
        device = str(get_device())
        _manager = ModelManager(checkpoint_path=checkpoint_path, device=device)
    return _manager


@contextmanager
def _temp_uploaded_file(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        suffix = ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(uploaded.getbuffer())
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _class_names():
    from dataset import CLASS_NAMES_DISPLAY
    return CLASS_NAMES_DISPLAY


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str = "checkpoints/best_model.pth"):
    return _get_manager(checkpoint_path)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown('<p class="main-header">🌌 SCALE × ODYSSEY</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Classification of Astronomical Objects</p>',
                unsafe_allow_html=True)
    st.markdown("---")


def _find_samples():
    samples_root = PROJECT_ROOT / "data" / "samples"
    if not samples_root.exists():
        return {}
    samples = {}
    for cls_dir in sorted(samples_root.iterdir()):
        if cls_dir.is_dir():
            images = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg"))
            if images:
                samples[cls_dir.name] = images
    return samples


def render_sidebar():
    from utils import get_device
    with st.sidebar:
        st.title("⚙️ Settings")
        st.subheader("Model")
        device_type = get_device().type  # cuda | mps | cpu
        device_labels = {"cuda": "CUDA", "mps": "Apple MPS", "cpu": "CPU"}
        device = device_labels.get(device_type, device_type.upper())
        icon = "🟢" if device_type in ("cuda", "mps") else "🟡"
        st.markdown(f"{icon} **Device:** {device}")
        if device_type == "cuda":
            import torch
            st.markdown(f"📟 **GPU:** {torch.cuda.get_device_name(0)}")
        st.markdown("🧠 **Backbone:** EfficientNet-B3")
        st.markdown("📊 **Classes:** 5")
        st.markdown("---")
        st.subheader("📋 Class Reference")
        for i, cls in enumerate(_class_names()):
            st.markdown(f"{i+1}. {cls}")
        st.markdown("---")
        st.subheader("🔬 How to read the output")
        st.markdown(
            "- **Probabilities** — the model's confidence across all 5 classes.\n"
            "- **Grad-CAM** — a heatmap of *where the model looked*; red = the "
            "regions that drove the decision, so you can sanity-check that it "
            "focused on the object and not the background.\n"
            "- **Caption** — a plain-language description generated offline from "
            "the prediction + image statistics.\n"
            "- **Anomaly / OOD** — screens for uncertain or out-of-distribution "
            "inputs using softmax entropy and max-probability."
        )
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("**SCALE × ODYSSEY** classifies raw astronomical images into 5 "
                    "celestial categories using deep learning. Built with PyTorch + "
                    "EfficientNet-B3 + Grad-CAM. All explainability and bonus features "
                    "run locally — no commercial APIs.")


def render_prediction_result(result):
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


def render_probability_chart(result):
    st.markdown("---")
    st.subheader("📊 Probability Distribution")
    import pandas as pd
    sorted_items = sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame({"Class": [n for n, _ in sorted_items], "Probability": [p for _, p in sorted_items]})
    st.bar_chart(df.set_index("Class"), use_container_width=True)
    st.dataframe(df.assign(Probability=df["Probability"].apply(lambda x: f"{x:.4f}")),
                 use_container_width=True, hide_index=True)


def render_gradcam(image, manager):
    from gradcam import explain_image
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
                st.image(image, use_container_width=True)
            with c2:
                st.markdown("**Predicted-Class CAM**")
                st.image(result["predicted_cam"], use_container_width=True)
            if result.get("true_idx") != result["pred_idx"] and "true_cam" in result:
                st.markdown("**True-Class CAM** (for comparison)")
                st.image(result["true_cam"], use_container_width=True)
            st.info(f"Model focused on highlighted regions to classify as **{result['pred_class']}**.")
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")
            st.info("Ensure grad-cam is installed: pip install grad-cam")


def render_caption(image_path: str, result):
    from bonus import generate_template_caption
    st.markdown("---")
    st.subheader("📝 Auto-Generated Caption")
    st.caption("Offline template captioner — built from the predicted class, "
               "confidence, and image brightness/contrast. No external API, no downloads.")
    cap = generate_template_caption(result.class_name, result.confidence, image=image_path)
    st.success(cap["caption"])


def render_anomaly(result):
    from bonus import detect_anomaly
    st.markdown("---")
    st.subheader("🛰️ Anomaly / Out-of-Distribution Check")
    st.caption("Softmax entropy + maximum-probability screening. Flags uncertain "
               "or novel inputs the model may not have seen in training.")
    verdict = detect_anomaly(result.all_probabilities)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Max probability", f"{verdict['max_prob']:.2f}",
                  help=f"Flagged if below {verdict['max_prob_threshold']:.2f}")
    with c2:
        st.metric("Entropy (bits)", f"{verdict['entropy']:.2f}",
                  help=f"Flagged if above {verdict['entropy_threshold']:.2f} "
                       f"(max possible {verdict['max_entropy_bits']:.2f})")
    if verdict["is_anomaly"]:
        st.warning(f"⚠️ {verdict['reason']}")
    else:
        st.info(f"✅ {verdict['reason']}")


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

    samples = _find_samples()
    display_names = {
        "spiral_galaxy": "Spiral Galaxy",
        "elliptical_galaxy": "Elliptical Galaxy",
        "nebula": "Nebula",
        "star_cluster": "Star Cluster",
        "planetary_object": "Planetary Object",
    }

    uploaded = st.file_uploader("Drag and drop an astronomical image (JPG/PNG)",
                                type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        with _temp_uploaded_file(uploaded) as temp_path:
            st.markdown("---")
            st.subheader("🖼️ Uploaded Image")
            left, right = st.columns([1, 1])
            with left:
                st.image(temp_path, use_container_width=True)
            with right:
                with st.spinner("Analyzing..."):
                    result = manager.predict(temp_path)
                render_prediction_result(result)
                render_probability_chart(result)
            render_gradcam(temp_path, manager)
            render_caption(temp_path, result)
            render_anomaly(result)
    elif samples:
        st.markdown("---")
        st.subheader("📸 Try a Sample")
        st.caption("Click a class below to load a demo image and run the full prediction + Grad-CAM pipeline.")
        cols = st.columns(len(samples))
        for col, (cls_name, paths) in zip(cols, sorted(samples.items())):
            with col:
                display = display_names.get(cls_name, cls_name)
                st.markdown(f"**{display}**")
                st.image(str(paths[0]), use_container_width=True)
                if st.button(f"Try {display}", key=f"btn_{cls_name}", use_container_width=True):
                    st.markdown("---")
                    st.subheader("🖼️ Sample Image")
                    left, right = st.columns([1, 1])
                    with left:
                        st.image(str(paths[0]), use_container_width=True)
                    with right:
                        with st.spinner("Analyzing..."):
                            result = manager.predict(str(paths[0]))
                        render_prediction_result(result)
                        render_probability_chart(result)
                    render_gradcam(str(paths[0]), manager)
                    render_caption(str(paths[0]), result)
                    render_anomaly(result)
    else:
        st.info("👆 Upload an astronomical image to begin classification.")

    render_footer()


if __name__ == "__main__":
    main()
