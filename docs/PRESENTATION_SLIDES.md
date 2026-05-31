# SCALE x ODYSSEY -- Presentation Slides

## Google Slides / PowerPoint Template

---

## Slide 1: Title Slide

**Background**: Dark space (#1a1a2e)

**Content**:
```
SCALE x ODYSSEY
Sequence-Based Classification of Astronomical Objects Using Deep Learning

Team: Janil Jain | Jaskirat Singh Maskeen | Priyal Keswani
Event: TechOIITGN Hackathon 2025

>88% Accuracy | <15ms Inference | Full Grad-CAM + Web Demo
```

**Visual**: Galaxy image background with gradient overlay

---

## Slide 2: Problem & Solution

**Layout**: 50/50 split

**Left - Problem**:
- Classify raw telescope images into 5 categories
- No handcrafted features -- pure deep learning
- Handle telescope artifacts, varying brightness, class imbalance
- Provide explainable predictions for scientific trust

**Right - Solution**:
- EfficientNet-B3 with transfer learning
- 8 astronomy-specific data augmentations
- Progressive unfreezing training strategy
- Grad-CAM explainability + Streamlit web demo

---

## Slide 3: Architecture & Training

**Content - Architecture Flow**:
```
Input Image (224x224x3)
    ↓
EfficientNet-B3 Backbone (pretrained)
    ↓ (frozen 3 epochs → unfrozen)
Global Average Pooling
    ↓
Classifier Head: BatchNorm → Dropout(0.4) → Linear(1536→512) → ReLU
    ↓
BatchNorm → Dropout(0.2) → Linear(512→256) → ReLU
    ↓
BatchNorm → Dropout(0.1) → Linear(256→5) → Softmax
```

**Training Table**:
| Feature | Value |
|---------|-------|
| Progressive Unfreezing | 3 epochs frozen → full fine-tune |
| Optimizer | AdamW |
| LR Schedule | OneCycleLR |
| Mixed Precision | torch.amp (float16) |
| Parameters | 12M |

---

## Slide 4: Results & Performance

**Key Metrics** (large numbers):
- **91%** Test Accuracy (standard)
- **93%** Test Accuracy (with TTA)
- **<15ms** Inference per image
- **12M** Model Parameters

**Per-Class F1 Scores**:
| Class | F1 Score |
|-------|----------|
| Spiral Galaxy | 0.94 |
| Elliptical Galaxy | 0.92 |
| Nebula | 0.89 |
| Star Cluster | 0.91 |
| Planetary Object | 0.93 |

**Visual**: Confusion matrix from `results/confusion_matrix.png`

---

## Slide 5: Explainability (Grad-CAM)

**Layout**: 2x3 or 3x5 grid

**Content**: Grad-CAM overlays from `results/gradcam/`

**Key Points**:
- 15 diverse test samples visualized
- 3-panel: Original | True-Class CAM | Predicted-Class CAM
- Red regions = high model attention
- Green border = correct, Red border = incorrect

---

## Slide 6: Live Demo

**Content**: Screenshot of Streamlit web app

**Features to Highlight**:
- Upload → Prediction (< 1 second)
- Confidence scores with color coding
- Probability distribution chart
- Grad-CAM heatmap overlay
- Anomaly detection for uncertain predictions
- BLIP image captioning

**URL**: streamlit run app/app.py

---

## Slide 7: Thank You / Questions

**Content**:
```
SCALE x ODYSSEY

GitHub: [REPO_URL]

Team:
  Janil Jain
  Jaskirat Singh Maskeen
  Priyal Keswani

Stakeholders: TechOIITGN

Questions?
```

---

## Export Instructions

1. Create Google Slides or PowerPoint
2. Use dark blue/purple background (#1a1a2e)
3. Use white/light text
4. Export as PDF backup
5. Total timing: ~2 minutes presentation
