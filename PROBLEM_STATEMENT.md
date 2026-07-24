# SCALE × ODYSSEY — Problem Statement

> **Sequence-Based Classification of Astronomical Objects Using Deep Learning**
> Verbatim transcription of the official problem statement / starter guide screenshots.
> Source images: `untitled folder/Screenshot 2026-05-27 ...` (12) and `resourses/Screenshot 2026-05-31 ...` (8) — both contain identical content. Safe to delete the images now; this file preserves their text.
> **Stakeholders:** Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani

### Additional Real-Data Resources

Curated public sources for real astronomical imagery (beyond what `src/prepare_data.py` already pulls):

* **Galaxy Zoo** — https://data.galaxyzoo.org — citizen-science-labelled galaxy morphology (spiral / elliptical ground truth)
* **SDSS** — https://www.sdss.org — Sloan Digital Sky Survey (deep multi-band sky imaging; basis of Galaxy10 / DECaLS)
* **ESA Hubble** — https://esahubble.org/projects/fits_liberator/datasets/ — Hubble FITS-liberator datasets
* **NASA Hubble** — https://science.nasa.gov/mission/hubble/ — mission page + data portal
* **NASA PDS** — https://pds.nasa.gov — Planetary Data System (planetary mission archives)
* **ESA PSA** — https://archives.esac.esa.int/psa/ — Planetary Science Archive

---

## Problem Statement

Develop a high-accuracy deep learning model to classify astronomical images into major celestial object categories such as Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, or Planetary Object using only raw image data, without relying on manually engineered astrophysical features or expert annotations.

The model should address key challenges in astronomical image understanding while minimizing dependence on domain-specific astronomy knowledge.

---

## Input–Output Definition

**Input:** An astronomical image: `X ∈ ℝ^(H×W×C)`
Where:
- H and W represent image dimensions
- C represents image channels (RGB or grayscale)

**Output:** A predicted class label
`y ∈ { Spiral, Elliptical, Nebula, Star Cluster, Planetary }`
representing the primary celestial object present in the image.

---

## Key Challenges

### 1. Visual Similarity Between Classes
Certain celestial objects exhibit highly similar visual patterns.

Examples:
1. Elliptical galaxies and blurred nebulae
2. Dense star clusters and galaxy cores

This creates ambiguity in classification, especially in low-resolution imagery.

Additionally:
1. Background noise,
2. telescope artifacts,
3. varying brightness levels

can significantly affect prediction quality.

### 2. Feature Extraction
Raw astronomical images lack explicit structural descriptors such as:
1. object boundaries,
2. luminosity maps,
3. morphology indicators,
4. or spectral information.

Traditional astronomy pipelines often require:
1. handcrafted feature engineering,
2. domain expertise,
3. or specialised preprocessing techniques.

The challenge is to learn meaningful representations directly from raw image data.

### 3. Model Generalisation
Astronomical images vary significantly across:
- Telescopes
- Resolutions
- Lighting conditions
- Observational environments

Models trained on one dataset may fail to generalise to:
- Noisy observations
- Rare celestial objects
- Previously unseen image distributions

---

## Technical Requirements

1. **Accuracy** — Achieve classification accuracy greater than 80% on benchmark astronomical datasets.
2. **Efficiency** — Perform inference on images up to resolution in under 5 seconds on consumer-grade hardware.
3. **Interpretability** — Provide visual explanations (e.g., Grad-CAM / attention maps) highlighting image regions responsible for predictions.

---

## Dataset Source Description

| Dataset | Source Description |
|---|---|
| Galaxy Zoo | Zooniverse Labeled galaxy morphology dataset containing spiral and elliptical galaxies. |
| DeepSky Dataset | Kaggle Collection of nebulae, galaxies, and star cluster imagery. |
| Hubble Image Archive | NASA / ESA Public astronomical imagery from Hubble telescope observations. |
| Astronomy Classification Dataset | Kaggle Preprocessed astronomical image dataset for object classification tasks. |

---

## Suggested Tasks

**Core Task (Mandatory):** Build a complete deep learning pipeline that:
- Preprocesses astronomical images
- Trains a classification model
- Evaluates model performance
- Visualises predictions and explanations

**Bonus Tasks (Optional):**

1. **Image Captioning** — Generate short natural language descriptions of astronomical images.
   Example: > "A spiral galaxy with bright central core and visible dust arms."
2. **Object Localization** — Detect and localize major celestial objects using:
   - Bounding boxes
   - Segmentation masks
3. **Anomaly Detection** — Identify unusual or rare astronomical objects that do not belong to known categories.
4. **Interactive Web Application** — Develop a web interface that allows users to:
   - Upload astronomical images
   - View predictions interactively
   - Explore attention or Grad-CAM visualizations

---

## Evaluation Criteria

| Component | Weightage |
|---|---|
| Classification Performance | 40% |
| Model Efficiency | 15% |
| Explainability & Visualization | 15% |
| Innovation / Bonus Features | 15% |
| Documentation & Presentation | 15% |

---

## Constraints and Rules

1. Teams may use pretrained open-source models with proper attribution.
2. Commercial black-box APIs should not constitute the primary solution.
3. All experiments, preprocessing steps, and architectures must be documented clearly.
4. The final submission must include reproducible code and inference instructions.

---

## Recommended Technical Stack

Participants may use: PyTorch, TensorFlow, OpenCV, HuggingFace, Streamlit / Flask / FastAPI

**Suggested architectures:** CNNs, ResNet, EfficientNet, Vision Transformers (ViT)

---

## Expected Deliverables

1. Trained model
2. Source code repository
3. Evaluation report
4. Hosted demo or local inference interface
5. Final presentation/demo video

---

## Learning Outcomes

Participants will gain experience in:
1. computer vision,
2. deep learning pipelines,
3. transfer learning,
4. model evaluation,
5. explainable AI,
6. deployment workflows

while working on a real-world astronomy-inspired machine learning problem.
