# SCALE x ODYSSEY -- 2-Minute Demo Video Script

## Exact Narration + Screen Actions

---

### [00:00 - 00:05] Title Card
**Screen**: Black screen -> Fade in project logo/text
**Narration**: "SCALE times ODYSSEY -- Deep Learning Classification of Astronomical Objects."
**Text on screen**:
```
SCALE x ODYSSEY
Sequence-Based Classification of Astronomical Objects
TechOIITGN Hackathon 2025
Team: Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani
```

---

### [00:05 - 00:30] Web Demo -- Upload + Prediction + Grad-CAM
**Screen**: Streamlit app at `localhost:8501`
**Actions**:
1. Show the full Streamlit interface with sidebar
2. Drag and drop a spiral galaxy image into the upload area
3. Watch the prediction card appear: "Spiral Galaxy -- 78% confidence"
4. Show the probability distribution bar chart
5. Scroll down to reveal the Grad-CAM overlay

**Narration**:
"Our interactive web demo allows users to upload any astronomical image and instantly get a classification prediction with confidence scores. The model identifies this as a Spiral Galaxy. Below, the Grad-CAM heatmap shows exactly which regions the model focused on -- the spiral arms and central bulge."

---

### [00:30 - 00:45] Web Demo -- Second Example (Different Class)
**Screen**: Streamlit app with new upload
**Actions**:
1. Upload a nebula image
2. Show prediction: "Nebula -- 89.7% confidence"
3. Show Grad-CAM highlighting the colorful gas regions

**Narration**:
"Here's a nebula -- the model correctly classifies it and the Grad-CAM shows it focused on the gas cloud structures, not background noise."

---

### [00:45 - 00:55] Grad-CAM Summary Grid
**Screen**: File explorer -> Open `results/gradcam/_summary_grid.png`
**Actions**: Show the full 15-sample summary grid (3 rows x 5 columns)

**Narration**:
"We generated Grad-CAM visualizations for 15 diverse test samples -- 3 per class. Green labels indicate correct predictions, red indicates misclassifications. This demonstrates strong explainability across all 5 categories."

---

### [00:55 - 01:10] Fast Inference CLI Demo
**Screen**: Terminal window
**Actions**:
```bash
python src/inference.py data/processed/test/ --batch-size 16
```
Show the rapid output with filenames, predictions, and confidence scores scrolling by.

**Narration**:
"Our inference engine processes a batch of test images in well under the 5-second-per-image requirement -- about 72 milliseconds median on this MacBook Air's Apple MPS, and faster still on a CUDA GPU. Here's a batch of 16 test images classified in one pass."

---

### [01:10 - 01:20] Bonus Features Demo
**Screen**: Terminal
**Actions**:
```bash
python src/bonus.py data/processed/test/nebula/sample.jpg
```
Show the output: classification, natural language caption, and anomaly detection result.

**Narration**:
"Bonus features include automatic image captioning using BLIP and anomaly detection that flags uncertain predictions."

---

### [01:20 - 01:35] Evaluation Results
**Screen**: Terminal showing evaluate.py output + results JSON
**Actions**:
```bash
python src/evaluate.py
cat results/evaluation_results.json
```
Show the accuracy numbers, per-class F1 scores, and TTA improvement.

**Narration**:
"On the held-out test set, our model achieves 95.6% accuracy with standard evaluation and 96.4% with test-time augmentation, for a macro F1 of 0.96. Nebula, Star Cluster, and Planetary classify perfectly; the residual confusion is between Spiral and Elliptical galaxies, which share genuinely similar morphologies."

---

### [01:35 - 01:50] Training & Architecture Highlights
**Screen**: Split view
- Left: TensorBoard loss/accuracy curves
- Right: Brief code snippet showing progressive unfreezing

**Narration**:
"We use EfficientNet-B3 with progressive unfreezing -- first training the classifier head, then fine-tuning the entire network. OneCycleLR scheduler and mixed precision training ensure fast convergence."

---

### [01:50 - 02:00] Closing Slide
**Screen**: Clean closing card
**Text**:
```
SCALE x ODYSSEY

GitHub: [your-repo-link]
Team: Janil Jain | Jaskirat Singh Maskeen | Priyal Keswani
Stakeholders: TechOIITGN

Built with PyTorch + EfficientNet-B3 + Grad-CAM + Streamlit

Thank you!
```

**Narration**:
"SCALE times ODYSSEY -- complete deep learning pipeline for astronomical object classification with full explainability and an interactive web demo. Thank you."

---

## Recording Tips

1. Use **OBS Studio** or **Loom** for screen recording
2. Resolution: 1920x1080 (1080p)
3. Speak clearly and at moderate pace
4. Pause 1-2 seconds between sections
5. Total target: **1:50 - 2:00** (slightly under is fine)
6. Export as MP4 (H.264 codec, ~10MB for 2 min)

## What Judges Want to See

- [x] Working web demo (upload -> prediction -> Grad-CAM)
- [x] Quantitative results (accuracy numbers)
- [x] Explainability (Grad-CAM overlays)
- [x] Speed (fast inference)
- [x] Bonus features (captioning/anomaly)
- [x] Clean, professional presentation
