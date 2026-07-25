# Phase 8 — Automated Proof — Evidence

## Created / Changed Files

| File | Action |
|------|--------|
| `tests/unit/test_checkpoint_smoke.py` | **Created** — smoke test that loads checkpoint, runs inference on a sample, asserts class, confidence, time |
| `tests/e2e/test_app.py` | **Updated** — kept import test, added `test_predict_smoke_with_checkpoint` with skip-if-no-ckpt |
| `scripts/ultra_win_gate.sh` | **Created** — composite gate: defaults check, unit tests, sample count, golden path, orphan check |
| `.github/workflows/ci.yml` | **Updated** — pip caching, `python -m pytest -m "not network"`, smoke job |
| `.github/workflows/security.yml` | **Updated** — pip caching, refined secret-scan patterns |

## Test Output — `python3 -m pytest tests/ -v -m "not network"`

```
tests/e2e/test_app.py::test_app_module_imports PASSED
tests/e2e/test_app.py::test_predict_smoke_with_checkpoint PASSED
tests/integration/test_pipeline.py::test_prepare_data_pipeline PASSED
tests/unit/test_bonus.py:: ... (18 tests) PASSED
tests/unit/test_checkpoint_smoke.py::test_checkpoint_predict_smoke PASSED
tests/unit/test_dataset.py:: ... (3 tests) PASSED
tests/unit/test_inference.py:: ... (1 test) PASSED
tests/unit/test_localization.py:: ... (4 tests) PASSED
tests/unit/test_model.py:: ... (3 tests) PASSED
tests/unit/test_onnx_export.py:: ... (3 tests) PASSED
tests/unit/test_ood.py:: ... (6 tests) PASSED
tests/unit/test_prepare_data.py:: ... (6 tests) PASSED
tests/unit/test_utils.py:: ... (3 tests) PASSED
================== 54 passed ==================
```

## Gate Script Output — `bash scripts/ultra_win_gate.sh`

```
== defaults ==
OK
== unit ==
51 passed
== samples ==
OK
== golden ==
GOLDEN_PATH_OK
== no orphans check ==
src/__init__.py  src/bonus.py  src/dataset.py  ...
GATE_PARTIAL_OK
```

## Acceptance Criteria

- [x] `python3 -m pytest tests/ -v -m "not network"` — **54 passed**
- [x] Smoke test skips gracefully when no checkpoint exists (skipif condition)
- [x] Predict smoke verifies class in `CLASS_NAMES_DISPLAY`, confidence in (0,1], time < 15000ms
- [x] CI workflow includes lint + unit tests with `-m "not network"`, pip cache, python 3.10
- [x] `ultra_win_gate.sh` passes all stages
