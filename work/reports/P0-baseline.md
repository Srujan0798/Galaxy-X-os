# P0 Baseline Probes — 2026-07-25

## Environment
- Host: macOS, Python 3.14.3 (`/usr/local/bin/python3`)
- Device available: Apple MPS
- Git: `main...origin/main` with dirty tree:
  - `M src/model.py` (multi-backbone WIP)
  - `?? src/tta.py` (unwired advanced TTA)

## Commands + results

### 1) Data emptiness
```bash
find data/processed -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l
# → 0
ls data/processed/{train,val,test}   # only .gitkeep
```

### 2) Unit tests
```bash
python3 -m pytest tests/unit/ -v
# → 46 passed, 2 warnings in 8.12s  EXIT 0
```

### 3) E2E (weak)
```bash
python3 -m pytest tests/e2e/ -v
# → 1 passed (import modules only) EXIT 0
```

### 4) Checkpoint structure
```text
keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_val_acc, config
epoch=23  best_val_acc≈0.9516
config.model.backbone=efficientnet_b3
state keys start with backbone.conv_stem (EfficientNet)
```

### 5) Live model load + inference
```text
LOAD_OK device=mps load_s≈1.94
Model: efficientnet_b3 | Params: 11,620,397
PRED_OK synthetic: class=Star Cluster conf=0.540 ms≈1036 (first)
Subsequent: ~467–751 ms reported inference_time_ms
NPARAMS 11620397
```

### 6) Claimed metrics artifact (not re-run)
`results/evaluation_results.json`:
- standard accuracy: **0.9317**
- TTA accuracy: **0.9277**
- macro F1 standard: **0.9318**

### 7) Secrets
No `API_KEY|SECRET|PASSWORD` hits in `src/` / `app/` (spot check).

### 8) Demo video
`docs/presentation/demo.mp4` duration ≈ **75.79 s**

## Honest baseline blended score
**72%** — see `docs/SCOREBOARD.md` and `work/reports/BRUTAL_AUDIT.md`.

## Immediate red cells
- Processed dataset images
- Re-runnable evaluate
- Hosted deploy
- E2E real path
- WIP model default risk
