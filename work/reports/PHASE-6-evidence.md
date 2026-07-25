# Phase 6 — Architecture Polish — Evidence

## Changes Made

### 1. Makefile — cleaned dead targets
- **Before**: `.PHONY` listed `install split train evaluate gradcam app test lint validate clean`
- **After**: Added `verify` to `.PHONY` and new `verify` target (runs tests + lint)
- No `generate_splits`, `train_head`, or other dead refs existed in Makefile (was already clean)
- Added fail-loud data checks (`@python3 -c "from utils import check_data_exists; check_data_exists()"`) to `train`, `evaluate`, `gradcam` targets

### 2. HOW_TO_RUN.md — fixed dead references
- Removed `make split` → `generate_splits.py` (dead), replaced with correct `src/prepare_data.py` description
- Removed `make train-head` → `train_head.py` (dead), replaced with `make validate` reference
- Fixed `make lint` description to match Makefile (`ruff check src/ app/ tests/`)
- Fixed project structure list: removed `augmentations` and `preprocess` (don't exist), added `tta` and `onnx_export`

### 3. Empty data fail-loud
- Added `check_data_exists()` to `src/utils.py` — checks all three splits, exits 1 with clear message if empty
- Added call at top of `main()` in `train.py`, `evaluate.py`, `gradcam.py`
- Makefile targets also run the check before each script

### 4. docs/SCOPE_GUARD.md — updated
- Localization: "as optional bonus via heatmap inspection" (Grad-CAM)
- No commercial APIs
- EfficientNet-B3 is primary backbone; other backbones for ensemble only

### 5. Dockerfile + docker-compose.yml — comments
- Dockerfile: added header comment explaining required volume mounts (checkpoints + data)
- docker-compose.yml: added version header comment explaining requirements

## Acceptance Tests

```bash
$ grep -n "generate_splits\|train_head" HOW_TO_RUN.md Makefile
→ Exit 1 (no matches found — correct)

$ python src/evaluate.py
→ "No images found in data/processed (train, val, test splits empty or missing).
   Run: python src/prepare_data.py  OR use data/samples demo via streamlit."
→ Exit 1 (correct — fail-loud with clear message)
```

## Files Modified
- `Makefile` — added verify target + data checks on train/evaluate/gradcam
- `HOW_TO_RUN.md` — fixed dead refs (generate_splits, train_head, augmentations)
- `src/utils.py` — added `check_data_exists()` helper
- `src/train.py` — added data check call
- `src/evaluate.py` — added data check call + import
- `src/gradcam.py` — added data check call + import
- `docs/SCOPE_GUARD.md` — updated scope policy
- `Dockerfile` — added mount requirement comments
- `docker-compose.yml` — added requirement comments
- `work/reports/PHASE-6-evidence.md` — this file
