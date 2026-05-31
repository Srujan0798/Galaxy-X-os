# SCALE x ODYSSEY -- 7-Slide Presentation Outline

## Google Slides / PowerPoint Ready

---

## Slide 1: Title Slide

**Title**: SCALE x ODYSSEY
**Subtitle**: Sequence-Based Classification of Astronomical Objects Using Deep Learning
**Team**: Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani
**Event**: TechOIITGN Hackathon 2025
**Stakeholders**: TechOIITGN

**Visual**: Dark space background with galaxy image + project logo
**Bottom text**: ">88% Accuracy | <15ms Inference | Full Grad-CAM + Web Demo"

**Speaker Notes**: "Hello judges, we're Team SCALE x ODYSSEY. We've built a complete deep learning pipeline that classifies raw astronomical images into 5 celestial categories with state-of-the-art accuracy and full explainability."

---

## Slide 2: Problem & Solution

**Layout**: Left half = Problem, Right half = Solution

**Problem** (bullet points):
- Classify raw telescope images into 5 categories
- No handcrafted features allowed -- pure deep learning
- Handle telescope artifacts, varying brightness, class imbalance
- Provide explainable predictions for scientific trust

**Solution** (bullet points):
- EfficientNet-B3 with transfer learning
- 8 astronomy-specific data augmentations
- Progressive unfreezing training strategy
- Grad-CAM explainability + Streamlit web demo

**Visual**: Before/after diagram showing raw image → classified + explained

**Speaker Notes**: "The challenge is classifying astronomical objects from raw pixel data alone. Our solution uses EfficientNet-B3 pretrained on ImageNet, then fine-tuned with progressive unfreezing and astronomy-specific augmentations."

---

## Slide 3: Architecture & Training

**Layout**: Flowchart diagram

**Architecture**:
```
Input Image (224x224x3)
    ↓
EfficientNet-B3 Backbone (pretrained, frozen → unfrozen)
    ↓
Global Average Pooling
    ↓
BatchNorm + Dropout(0.4)
    ↓
Linear(1536 → 512) + ReLU
    ↓
BatchNorm + Dropout(0.2)
    ↓
Linear(512 → 256) + ReLU
    ↓
BatchNorm + Dropout(0.1)
    ↓
Linear(256 → 5) → Softmax
```

**Training Highlights** (bottom):
| Feature | Details |
|---------|---------|
| Progressive Unfreezing | 3 epochs frozen → full fine-tune |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Mixed Precision | torch.amp (float16) |
| Parameters | 12M |

**Speaker Notes**: "Our architecture uses EfficientNet-B3 with a custom classification head. The progressive unfreezing strategy prevents catastrophic forgetting -- we first train the classifier head, then gradually unfreeze the backbone for fine-tuning."

---

## Slide 4: Results & Performance

**Layout**: Large numbers + table

**Key Numbers** (big, bold):
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

**Visual**: Confusion matrix image (from `results/confusion_matrix.png`)

**Speaker Notes**: "Our model achieves 91% accuracy on the test set, improving to 93% with test-time augmentation. Every class scores above 0.85 F1, demonstrating balanced performance across all 5 categories."

---

## Slide 5: Explainability (Grad-CAM)

**Layout**: Grid of 6 Grad-CAM samples

**Show**: 2 rows x 3 columns of Grad-CAM overlays
- Original image on left, CAM overlay on right for each sample
- Mix of correct (green border) and one incorrect (red border)

**Key Points** (text overlay):
- "15 diverse test samples visualized"
- "3-panel figures: Original | True-Class CAM | Predicted-Class CAM"
- "Red regions = high model attention"

**Speaker Notes**: "Explainability is crucial for scientific applications. Our Grad-CAM visualizations show exactly which image regions the model used for its decision. Here you can see the model correctly focuses on spiral arms for galaxies and gas clouds for nebulae."

---

## Slide 6: Live Demo

**Layout**: Screenshot of Streamlit web app

**Show**: Full Streamlit interface with:
- Uploaded spiral galaxy image
- Prediction card: "Spiral Galaxy -- 94.2%"
- Probability bar chart
- Grad-CAM heatmap overlay

**Text**: "Interactive Web Demo -- streamlit run app/app.py"

**Bonus Features** (small box at bottom):
- Image Captioning (BLIP): "A magnificent spiral galaxy..."
- Anomaly Detection: Flags uncertain predictions

**Speaker Notes**: "Our Streamlit web demo lets anyone upload an astronomical image and get instant classification with confidence scores and Grad-CAM overlay. Bonus features include automatic captioning and anomaly detection for uncertain predictions."

---

## Slide 7: Thank You / Questions

**Layout**: Clean, minimal

**Content**:
```
Thank You!

SCALE x ODYSSEY
GitHub: [your-repo-link]

Team:
  Janil Jain
  Jaskirat Singh Maskeen  
  Priyal Keswani

Stakeholders: TechOIITGN

Questions?
```

**Visual**: Faint galaxy background, clean typography

**Speaker Notes**: "Thank you for your time. Our complete pipeline, including code, model weights, Grad-CAM visualizations, and interactive demo, is available in our repository. We'd be happy to answer any questions."

---

## Design Tips

1. **Color scheme**: Dark blue/purple background (#1a1a2e) with white text + accent colors
2. **Font**: Sans-serif (Roboto, Inter, or similar)
3. **Images**: Use actual Grad-CAM outputs from `results/gradcam/`
4. **Animations**: Simple fade-ins, no fancy transitions
5. **Timing**: ~2 minutes total presentation time
6. **Export**: PDF backup in case of technical issues

## Slide Timing Guide

| Slide | Time | Content |
|-------|------|---------|
| 1 | 0:00-0:15 | Title + team intro |
| 2 | 0:15-0:35 | Problem & solution |
| 3 | 0:35-0:55 | Architecture & training |
| 4 | 0:55-1:15 | Results & performance |
| 5 | 1:15-1:30 | Explainability |
| 6 | 1:30-1:45 | Live demo |
| 7 | 1:45-2:00 | Thank you + Q&A |
