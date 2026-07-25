# HOSTILE JUDGE REPORT — Galaxy-X-os

**Date:** 2026-07-25  
**Judge:** AGENT R  
**Verdict:** **KICK OUT** — 5 P0 bugs, 4 P1 bugs, systemic credibility failure

---

## P0 — SHOWSTOPPER (Blockers)

### P0.1 `data/processed/train/` `data/processed/val/` `data/processed/test/` ALL EMPTY
Zero images. Only `.gitkeep` files. `DATA_MANIFEST.json` claims 2,484 images exist. They do not. `evaluate.py`, `train.py`, `gradcam.py` all fail immediately:
```
No images found in data/processed (train, val, test splits empty or missing).
```
**Cannot reproduce any claimed metric.** (cf. `README.md:114`, `SUBMISSION.md:72` "Data local")

### P0.2 Config/checkpoint backbone mismatch
`configs/config.yaml:12` specifies `backbone: "convnext_base"`. The committed checkpoint `best_model.pth` is **EfficientNet-B3**. Any retrain or fine-tune naive user following config will get shape-mismatch errors. The config is the canonical source of truth per `REPORT.md:77`.

### P0.3 Model misclassifies sample images ~55% of the time
On the 11 sample images (including a rogue `noise/` class), the model gets **5/11 wrong (45.5% accuracy)**:
- `spiral_galaxy` → "Planetary Object" (76%)
- `planetary_object` → "Nebula" (62%)
- `elliptical_galaxy` → "Planetary Object" (35%)
- `noise` → "Planetary Object" (44%)
The claimed 93.17% test accuracy cannot be reproduced.

### P0.4 SUBMISSION.md fabricates non-existent files and metrics
Claims as main deliverables:
- `src/detection.py` — EXISTS ONLY in `src/attic/` (abandoned)
- `src/gradcam_plus.py` — EXISTS ONLY in `src/attic/` (abandoned)
- `src/pseudo_label.py` — EXISTS ONLY in `src/attic/` (abandoned)
- `94.1% ensemble+TTA` — no ensemble checkpoints exist, `results/evaluation_results.json` shows only 93.17% single-model
- "True anchor-free detection head, NOT just attention" — code is in attic, nothing wired
- Ensemble claims "ConvNeXt-Base (88M) + Swin-B (88M) + EfficientNet-B3 (11.6M) = ~188M params" — no ensemble weights exist

### P0.5 Time to first prediction: ~20+ minutes (not insta-demo)
Quick Start says "3 steps → demo". In reality:
- `pip install -r requirements.txt` → heavy deps, transformers + torch ~2GB download
- `wget checkpoint` → 140MB download
- First inference cold-start on CPU: **4.9 seconds** for a single image (median warm CPU ~1.8s)
- No pre-built environment, no Docker image in registry, no pip-installable package
- `prepare_data.py` requires network + external APIs (NASA Image Library, astroNN) — unworkable offline

---

## P1 — MAJOR

### P1.1 Quick Start broken for offline/air-gapped evaluation
Step 2 requires downloading a 140MB checkpoint from GitHub Releases. Step 1 installs 50+ deps including PyTorch 2.4, timm, transformers, OpenCV, Albumentations. No `uv.lock` or pinned hash for all transitive deps. No `Makefile` target for dev setup that actually works.

### P1.2 `data/samples/noise/` directory is an invalid class
The samples directory contains a `noise/` folder that is NOT one of the 5 target classes. `app.py:_find_samples()` will enumerate it, the Streamlit sample picker will show it, and inference will produce a meaningless result.

### P1.3 `eval_checkpoint: "checkpoints/best_model.pth"` in config but config.backbone = "convnext_base"
Even if data existed, loading `best_model.pth` (EfficientNet-B3) into a ConvNeXt-Base model skeleton will fail with `state_dict` key mismatch. The full eval/train pipeline is unusable without manual intervention.

### P1.4 Inference latency on CPU borderline
Cold-start first prediction: **4.9s** (dangerously close to 5s limit). Median warm CPU: ~1.8s. README claims "~72ms per image on Apple MPS" but MPS was not available in this test. Claims do not specify CPU performance.

---

## P2 — MINOR

| # | Issue |
|---|-------|
| 1 | `data/samples/README.md` — `_find_samples()` in `app.py` iterates dirs but README.md is a file. Not a crash, but careless. |
| 2 | SHA256 hash in `SUBMISSION.md:79` references v1.0 release, but the committed checkpoint is undated — no local verification path. |
| 3 | `attic/` contains `detection.py`, `gradcam_plus.py`, `pseudo_label.py` — these are advertised in SUBMISSION.md as main src files. Hiding in attic suggests incomplete/unstable work. |
| 4 | `docker-compose.yml` and `Dockerfile` exist but no instructions to use them in Quick Start. Untested. |
| 5 | Streamlit app shows "Backbone: EfficientNet-B3" but config default is `convnext_base` — contradictory. |

---

## Time-to-First-Prediction Estimate

| Step | Time |
|------|------|
| `git clone` | ~5s |
| `pip install -r requirements.txt` | ~8-15 min (PyTorch + transformers ~2GB) |
| `wget checkpoint` | ~1-3 min (140MB, depends on bandwidth) |
| Cold-start inference | ~5s |
| **Total (best case)** | **~10-18 minutes** |
| **Total (cold cache, slow network)** | **~30+ minutes** |
| *Does not include data preparation (requires network + NASA API)* |

---

## Would I Score This Top 0.1%? **NO**

**Reasons:**
1. **Irreproducible metrics** — 93.17% accuracy claimed against empty data directories
2. **Systematic overclaiming** — SUBMISSION.md fabricates ensemble results (94.1%), non-existent files (detection.py, pseudo_label.py, gradcam_plus.py in src/), and features not wired into the pipeline
3. **Model underperforms** — 45.5% accuracy on included sample images (vs 93.17% claimed)
4. **Config/checkpoint mismatch** — ConvNeXt-Base configured but EfficientNet-B3 weights committed
5. **Pipeline broken** — evaluate.py, train.py, gradcam.py all fail with empty data

The submission has good scaffolding (tests pass, Streamlit launches, logging works) but the core claims are dishonestly inflated and the pipeline cannot be executed end-to-end.

---

## Required Fixes Before Gate B

1. **Commit actual data** — at minimum the test split (249 images) so metrics can be verified
2. **Fix config/checkpoint alignment** — `config.yaml` must match the committed checkpoint backbone, or commit matching ConvNeXt-Base weights
3. **Remove fabrications from SUBMISSION.md** — delete ensemble claims, remove references to non-existent `src/` files, be honest about attic status
4. **Fix model inference** — investigate why samples are misclassified at 45.5% error rate
5. **Remove `data/samples/noise/`** — it's not a valid class and breaks the sample picker
6. **Update README Quick Start** — add note about CPU cold-start latency; suggest alternative demo path using samples
7. **Wire or remove attic features** — either integrate `detection.py`, `gradcam_plus.py`, `pseudo_label.py` into main src/ or stop claiming them in submission docs
8. **Add `Makefile` target for reproducible dev setup** (e.g. `make install`, `make demo`)
9. **Pin transitive deps** — generate full `requirements.txt` with hashes or use `uv.lock` reliably
10. **Document known model limitations honestly** — the 93.17% accuracy should be caveated with "requires running prepare_data.py + train.py on GPU (12+ hours) — cannot reproduce from committed artifacts"

---

## Verdict

**KICK OUT.** The submission has a solid skeleton (tests, Streamlit, code organization) but is fatally undermined by empty data directories, config/checkpoint mismatch, fabricated claims in SUBMISSION.md, and a model that underperforms on the very samples provided. It is not Gate-B-ready.
