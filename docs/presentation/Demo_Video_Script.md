# SCALE x ODYSSEY — ~90-Second Demo Video Script

## Exact Narration + Screen Actions

---

### [00:00–00:05] Title Card

**Screen:** Black → fade in project logo/text
**Narration:** "SCALE times ODYSSEY — Deep Learning Classification of Astronomical Objects."
**Text on screen:**
```
SCALE x ODYSSEY
Sequence-Based Classification of Astronomical Objects
TechOIITGN Hackathon 2025
Team: Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani
```

---

### [00:05–00:08] Terminal — 3 Commands

**Screen:** Terminal window
**Actions:** Type and run:
```bash
git clone https://github.com/Srujan0798/Galaxy-X-os.git
cd Galaxy-X-os
streamlit run app/app.py
```
**Narration:** "Three commands from a fresh terminal — clone, enter, run."

---

### [00:08–00:25] Web Demo — Sample Buttons

**Screen:** Streamlit app at `localhost:8501`
**Actions:**
1. Show the full interface — header, sidebar with model info, 5 sample cards
2. Hover over the sample cards, then click **"Try Spiral Galaxy"**
3. Watch the prediction card appear: "Spiral Galaxy — 78% confidence"
4. Show the probability distribution bar chart
5. Scroll down to Grad-CAM overlay

**Narration:** "Our Streamlit app loads with sample buttons for all 5 classes. Click any sample — within seconds you get a prediction card, confidence score, inference time, a full probability distribution, and a Grad-CAM heatmap showing exactly which image regions drove the decision."

---

### [00:25–00:35] Web Demo — Second Sample

**Screen:** Scroll back to samples
**Actions:** Click **"Try Nebula"**
**Narration:** "Try a Nebula — the model correctly classifies it at high confidence, and the Grad-CAM highlights the gas cloud structures, not the background."

---

### [00:35–00:45] Bonus Features — Caption + Anomaly

**Screen:** Continue scrolling below Grad-CAM
**Actions:** Show the **Auto-Generated Caption** section and the **Anomaly / Out-of-Distribution Check** section
**Narration:** "Bonus features run automatically — a template caption describes the image in natural language, and the anomaly check screens for uncertain or out-of-distribution inputs using softmax entropy."

---

### [00:45–00:55] Grad-CAM Summary Grid

**Screen:** File explorer → Open `results/gradcam/_summary_grid.png`
**Actions:** Show the full 15-sample summary grid (3 rows × 5 columns)
**Narration:** "We generated Grad-CAM visualizations for 15 diverse test samples — 3 per class. Green labels are correct predictions; red are misclassifications. This demonstrates strong explainability across all categories."

---

### [00:55–01:05] Fast Inference CLI

**Screen:** Terminal window
**Actions:**
```bash
python src/inference.py data/processed/test/ --batch-size 16
```
Show rapid output with filenames, predictions, and confidence scores.
**Narration:** "Our inference engine processes images at ~72 milliseconds median on Apple MPS — well under the 5-second requirement. Here's a batch of 16 test images classified in one pass."

---

### [01:05–01:15] Evaluation Results

**Screen:** Terminal
**Actions:**
```bash
python src/evaluate.py
cat results/evaluation_results.json
```
**Narration:** "Our held-out test set yields **93.17% accuracy**, **0.932 macro F1**. Nebula and Planetary reach 0.95–0.98 F1; the residual confusion is between Spiral and Elliptical galaxies — a real morphological ambiguity, not a hidden weakness."

---

### [01:15–01:25] Training & Architecture

**Screen:** Split view — TensorBoard curves left, code snippet right
**Narration:** "EfficientNet-B3 with progressive unfreezing — classifier head first, then full fine-tune. OneCycleLR and mixed precision ensure fast convergence on a Colab T4 GPU."

---

### [01:25–01:30] Closing Slide

**Screen:** Clean closing card
**Text:**
```
SCALE x ODYSSEY

GitHub: github.com/Srujan0798/Galaxy-X-os
Team: Janil Jain | Jaskirat Singh Maskeen | Priyal Keswani
Stakeholders: TechOIITGN

Built with PyTorch + EfficientNet-B3 + Grad-CAM + Streamlit

Thank you!
```
**Narration:** "SCALE times ODYSSEY — complete deep learning pipeline for astronomical object classification. Thank you."

---

### [01:30–01:35] End Card (optional)

**Screen:** Fade to black
**Overlay text:** "93.17% Accuracy · 0.932 Macro F1 · Full Grad-CAM · ONNX Export"

---

## RE-RECORD REQUIRED

The current `demo.mp4` was recorded before the sample-buttons UI was added to `app/app.py`. The app now shows 5 sample cards (one per class) as the default landing state, with click-to-predict buttons. The existing video shows only drag-and-drop upload. Re-record at 1920×1080 to match the current UI:

1. Start with terminal (3 commands)
2. Show app landing with sample buttons
3. Click samples, show Grad-CAM, caption, anomaly
4. CLI demos for inference + evaluation
5. Closing card

## Recording Tips

1. Use **OBS Studio** or **Loom** for screen recording
2. Resolution: 1920×1080 (1080p)
3. Speak clearly at moderate pace
4. Pause 1–2 seconds between sections
5. Target: **1:15–1:30**
6. Export as MP4 (H.264, ~10 MB)

## Checklist

- [x] Working web demo with sample buttons (upload → prediction → Grad-CAM → caption → anomaly)
- [x] Quantitative results (accuracy, F1)
- [x] Explainability (Grad-CAM overlays + summary grid)
- [x] Fast inference (~72 ms)
- [x] Bonus features (caption, OOD, localization, ONNX)
- [x] 3-command setup
