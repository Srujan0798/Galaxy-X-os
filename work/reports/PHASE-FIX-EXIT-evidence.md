# Phase FIX-EXIT — Evidence

**Date:** 2026-07-25  
**Bug:** evaluate.py exited 0 on empty data

## Fix
Added `sys.exit(1)` to `check_data_exists()` in `src/utils.py:48`

## Acceptance
```bash
$ python3 src/evaluate.py ; echo $?
No images found in data/processed (train, val, test splits empty or missing).
Run: python src/prepare_data.py  OR use data/samples demo via streamlit.
Exit: 1

$ python3 src/train.py ; echo $?
...
No images found in data/processed (train, val, test splits empty or missing).
Exit: 1

$ python3 src/gradcam.py ; echo $?
...
Exit: 1

$ bash scripts/verify_golden_path.sh | grep GOLDEN
GOLDEN_PATH_OK
Exit: 0
```

## Files changed
- `src/utils.py` line 48: added `sys.exit(1)`
