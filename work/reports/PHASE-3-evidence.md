# PHASE 3 — TTA Depth — Evidence

**Date:** 2026-07-25
**Agent:** C
**Path chosen:** ATTIC (data/processed has 0 images — only metadata)

---

## Executed Steps

### 1. Move src/tta.py -> attic/src-archive/tta.py
```
$ mv src/tta.py attic/src-archive/tta.py
```

### 2. Remove import from src/evaluate.py
Removed line: `from tta import get_tta_transforms, get_tta_transforms_heavy  # noqa: E402`
Standard 6× TTA (inline, self-contained) remains untouched.

### 3. Verify attic/src-archive/ exists
Already existed (contains prior archive files).

---

## Acceptance Checks

```bash
$ test ! -f src/tta.py && echo "PASS"
PASS

$ test -f attic/src-archive/tta.py && echo "PASS"
PASS

$ rg -n "from tta|import tta" src/ app/ tests/
# (no output → no matches, PASS)

$ python3 -m pytest tests/unit/test_tta.py -v
# file not found → SKIP (expected; no tta test file exists)

```
**rg exit code:** 1 (no matches → clean)  
**pytest exit code:** 4 (no test file → expected)

---

## Summary

| Item | Status |
|------|--------|
| src/tta.py moved to attic | ✅ |
| Import removed from evaluate.py | ✅ |
| No tta imports in src/, app/, tests/ | ✅ |
| attic/src-archive/tta.py exists | ✅ |
| Standard 6× TTA in evaluate.py intact | ✅ (lines 178–222) |
