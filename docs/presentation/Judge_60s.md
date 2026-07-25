# SCALE x ODYSSEY — 60-Second Judge Talk Track

**Team:** Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani

---

### [00:00–00:08] Hook

"We are **Galaxy-X-os**, a 5-class astronomical classifier built from scratch — no black boxes, no commercial APIs. Raw image in → class label + confidence + explanation out."

### [00:08–00:22] Problem → Approach

"The problem: classify telescope images into Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, or Planetary Object using only pixels — no handcrafted features. Our approach: **EfficientNet-B3** with ImageNet transfer learning, progressive unfreezing, and three custom astronomy augmentations — cosmic ray simulation, vignetting, and Poisson noise — so the model generalizes across telescope sources."

### [00:22–00:35] Key Result

"On a **250-image held-out disjoint test set** of real multi-source data (Galaxy10 DECaLS + NASA), we achieve **93.17% test accuracy** (92.77% with test-time augmentation) and **0.932 macro F1**. Nebula and Planetary hit 0.95–0.98 F1; the residual confusion is between Spiral and Elliptical galaxies — which share genuinely similar morphologies."

### [00:35–00:45] Explainability

"Every prediction includes **Grad-CAM** — a heatmap showing exactly which pixels drove the decision. In our Streamlit app, you see the original image, the predicted-class CAM, and the probability distribution. For scientists, this builds trust: the model looks at spiral arms, not background noise."

### [00:45–00:52] Demo

"Run it yourself: three commands after cloning — `streamlit run app/app.py`. Click a sample button, and within 10 seconds you get a prediction, Grad-CAM, auto-caption, and an anomaly check. All on-device, no API keys."

### [00:52–00:60] Bonuses & Honest Limitation

"Bonuses: template captioning, out-of-distribution detection, localization via bounding boxes, and **ONNX export** for production deployment. Honest limitation: spiral/elliptical confusion persists (~0.88–0.90 F1) — these classes are genuinely ambiguous even for human experts. Captions are template-based, not generative. We document all gaps in our scoreboard."

---

**Total: ~60 seconds at moderate pace.**
