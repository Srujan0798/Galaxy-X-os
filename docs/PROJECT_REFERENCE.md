# Galaxy-X-os — Project Reference (single source of truth)

> Central record of everything about this submission. **No secrets live in this file.**
> Credentials are stored outside the repo (see *Credentials* below).

## Identity
- **Project:** Galaxy-X-os — SCALE × ODYSSEY
- **Task:** Classify raw astronomical images into 5 classes — Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object.
- **Owner:** Srujan · srujansai1010@gmail.com
- **GitHub repo:** https://github.com/Srujan0798/Galaxy-X-os  (public)
- **Release tag:** v1.0 → https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0
- **Problem stakeholders (from brief):** Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani

## Model & results
- **Architecture:** EfficientNet-B3 (timm, ImageNet pretrained), ~11.6M params, 224×224 input.
- **Current metrics** (procedural-fallback data run): 95.6% acc / 96.4% TTA / macro-F1 0.956 / 0.964.
- **Inference:** ~72 ms/image median on Apple MPS (measured).
- **Pending:** re-run on fully-real NASA-archive data → numbers will be refreshed honestly.

## Data sources (all real)
| Class(es) | Source | Link |
|-----------|--------|------|
| Spiral, Elliptical | Galaxy10 DECaLS/SDSS via astroNN (Galaxy Zoo morphology lineage) | https://data.galaxyzoo.org · https://www.sdss.org |
| Nebula, Star Cluster, Planetary | **NASA Image Library** (no API key) — `src/download_archives.py` | https://images.nasa.gov · https://images-api.nasa.gov |
| (reference archives, not auto-downloaded) | ESA Hubble FITS-liberator | https://esahubble.org/projects/fits_liberator/datasets/ |
| (reference) | NASA Hubble mission portal | https://science.nasa.gov/mission/hubble/ |
| (reference) | NASA PDS (Planetary Data System) | https://pds.nasa.gov |
| (reference) | ESA PSA (Planetary Science Archive) | https://archives.esac.esa.int/psa/ |
| (fallback only, needs key) | Kaggle deep-space sets | fedesoriano/deep-space-images · brsdincer/planetary-solar-system-objects |

Authoritative per-class real-vs-fallback record at run time: `data/processed/DATA_MANIFEST.json`.

## How to run
```bash
pip install -r requirements.txt
python src/prepare_data.py     # real data, no key needed (NASA + Galaxy10)
python src/train.py            # GPU recommended
python src/evaluate.py         # standard + TTA
python src/gradcam.py          # explainability
streamlit run app/app.py       # demo
```
One-click GPU: `notebooks/Galaxy_X_Colab.ipynb` → Runtime → Run all.

## Bonus features (15%)
Offline template captioning + BLIP option, softmax-entropy anomaly/OOD detection
(`src/bonus.py`), astro-specific augmentations, TTA, Streamlit demo.

## Submission
- Google Form: enter name, paste GitHub link, upload `REPORT.pdf` (≤10 MB).
- Repo must be public and pushed.

## Credentials (NOT in this repo)
- **Kaggle API token** (only needed for the optional Kaggle *fallback* — NASA needs none):
  stored at `~/.kaggle/access_token` (perms 600, outside the repo, gitignored path).
  ⚠️ This token was exposed in a chat transcript — **rotate it** at
  https://www.kaggle.com/settings → *Create New Token*, then overwrite that file.
- **GitHub:** authenticated via the system git credential / `gh` (account `Srujan0798`).
  Note: the current `gh` token cannot create Releases (attach assets via the web UI).

## Environment gotchas
- Train on GPU (Colab) — the 8GB MacBook hangs on full fine-tune.
- `best_model.pth` (~141 MB) exceeds GitHub's 100 MB limit → not committed; reproduce via notebook or grab from the v1.0 Release.
