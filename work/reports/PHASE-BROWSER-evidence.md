# Phase BROWSER — Evidence

**Date:** 2026-07-25  
**Method:** Playwright headless Chrome against local Streamlit

## Steps
1. Start: `streamlit run app/app.py` → http://localhost:8501
2. Page loads with 5 sample buttons (Spiral, Elliptical, Nebula, Star Cluster, Planetary)
3. Click "Try Elliptical Galaxy"
4. Wait for model load + inference (~15s)
5. Screenshot taken

## Results
```
=== BROWSER GOLDEN PATH RESULTS ===
  ✅ Predicted Class — visible
  ✅ Grad-CAM — visible  
  ✅ Caption — visible
  ✅ OOD — visible

BROWSER_GOLDEN_OK
```

Screenshots: `/tmp/streamlit_initial.png`, `/tmp/streamlit_result.png`

## Full acceptance
```bash
$ python3 -m pytest tests/ -m "not network" -q
57 passed in 80.19s

$ bash scripts/verify_golden_path.sh | grep GOLDEN
GOLDEN_PATH_OK

$ python3 src/evaluate.py ; echo $?
No images found in data/processed...
Exit: 1
```

## Residual
- Streamlit tested with local checkpoint; cold clone without ckpt shows error UX (Release links)
- CI not yet run on remote
