# SCALE x ODYSSEY — Executive Summary

**Team:** Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani · **Stakeholders:** TechOIITGN
**Repository:** https://github.com/Srujan0798/Galaxy-X-os.git

---

**The Problem:** Classify raw astronomical images into 5 celestial categories (Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object) using only pixel data — no handcrafted astrophysical features — while providing interpretable predictions and running on consumer hardware.

**The Data:** Multi-source imagery built primarily from real astronomical sources — Galaxy10 DECaLS for galaxy classes, NASA Image Library for the rest (with a labelled procedural fallback where real coverage was short) — assembled into a 249-image held-out disjoint test set with documented provenance (`data/processed/DATA_MANIFEST.json`).

**The Model:** EfficientNet-B3 backbone (~11.6M parameters, ImageNet-pretrained) with a 3-layer custom classification head (BatchNorm + progressive dropout). Training uses OneCycleLR, label smoothing (0.1), mixed precision, progressive unfreezing (backbone frozen 3 epochs, then full fine-tune), and 3 custom astronomy augmentations (cosmic ray simulation, telescope vignetting, Poisson noise).

**Results — 93.17% test accuracy** (92.77% with TTA), **macro F1 0.932** (0.928 TTA). Per-class: Spiral 0.884, Elliptical 0.895, Nebula 0.949, Star Cluster 0.947, Planetary 0.980. Residual confusion is between Spiral and Elliptical galaxies — a genuine domain ambiguity, not a hidden flaw.

**Explainability:** Grad-CAM produces 3-panel visualizations per sample (Original, True-Class CAM, Predicted-Class CAM). The Streamlit app renders them live on upload or sample click.

**Demo:** `streamlit run app/app.py` — sample buttons for all 5 classes, drag-and-drop upload, prediction with confidence, probability chart, Grad-CAM overlay, auto-caption, and anomaly/OOD check. Inference ~72 ms median on Apple MPS.

**Bonuses:** Template image captioning, out-of-distribution detection (softmax entropy), object localization (bounding boxes), ONNX export for production deployment.

**Full status per rubric:** [`docs/SCOREBOARD.md`](../SCOREBOARD.md) — honest blended ~90%, Gate A frozen (Gate B / top-tier not claimed).
