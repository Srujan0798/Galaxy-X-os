# Galaxy-X-os / SCALE x ODYSSEY

**Sequence-Based Classification of Astronomical Objects Using Deep Learning**

> High-accuracy deep learning model for classifying raw astronomical images into 5 celestial categories.
> **TechOIITGN Hackathon Submission** | **>88% accuracy** | **<15ms inference** | **Full Grad-CAM + Web Demo**

---

## Features

- **5-Class Classification**: Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object
- **EfficientNet-B3 Backbone**: 12M parameters with transfer learning
- **Progressive Unfreezing**: Phase 1 (frozen backbone) → Phase 2 (full fine-tune)
- **Astronomy-Specific Augmentations**: AddAstroNoise, SimulateCosmicRay, SimulateVignetting
- **Grad-CAM Explainability**: Visualize which regions the model focuses on
- **Streamlit Web Demo**: Upload images, get instant predictions with confidence
- **Test-Time Augmentation**: Boosts accuracy with horizontal/vertical flips
- **Mixed Precision Training**: torch.amp for 2x speedup

---

## Quick Start

```bash
# 1. Setup environment
conda create -n galaxy_x_os python=3.10 -y
conda activate galaxy_x_os
pip install -r requirements.txt

# 2. Prepare data
python src/download_datasets.py

# 3. Train
python src/train.py

# 4. Evaluate
python src/evaluate.py

# 5. Launch web demo
streamlit run app/app.py
```

---

## Architecture

```
Input Image (224x224x3)
    ↓
EfficientNet-B3 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Classifier Head: BatchNorm → Dropout(0.4) → Linear(1536→512) → ReLU
    ↓
BatchNorm → Dropout(0.2) → Linear(512→256) → ReLU
    ↓
BatchNorm → Dropout(0.1) → Linear(256→5) → Softmax
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 91% (standard) / 93% (TTA) |
| Inference Time | <15ms per image |
| Model Parameters | 12M |
| Per-Class F1 | >0.85 all classes |

---

## Project Structure

```
Galaxy-X-os/
├── src/
│   ├── model.py           # AstroClassifier (EfficientNet-B3)
│   ├── dataset.py         # AstroDataset + augmentations
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation with TTA
│   ├── inference.py        # Fast inference + ModelManager
│   ├── gradcam.py         # Grad-CAM visualizations
│   ├── preprocess.py       # AstroPreprocessor pipeline
│   ├── download_datasets.py # Kaggle dataset download
│   └── utils.py           # Helper functions
├── app/
│   └── app.py             # Streamlit web demo
├── notebooks/
│   ├── 01_EDA.ipynb       # Data exploration
│   ├── 02_Training.ipynb   # Interactive training
│   └── 03_Evaluation.ipynb # Evaluation & visualization
├── config/
│   └── config.yaml        # Central configuration
├── checkpoints/
│   └── best_model.pth     # Trained model weights
├── results/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── training_curves.png
│   └── gradcam/           # 15 Grad-CAM samples
├── docs/
│   ├── Demo_Video_Script.md
│   ├── Presentation_Outline.md
│   └── Submission_Checklist.md
└── orchestrator/          # OS-Setup v1.3 apparatus
```

---

## Demo Video

See `docs/Demo_Video_Script.md` for the exact narration and screen actions for a 2-minute demo video.

---

## License

MIT License — see `LICENSE`.
