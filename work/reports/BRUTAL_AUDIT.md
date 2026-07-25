# BRUTAL AUDIT — Galaxy-X-os (SCALE × ODYSSEY)

**Date:** 2026-07-25  
**Auditor mode:** Hostile competition validator + systems auditor  
**Project root:** `/Users/srujansai/Desktop/Galaxy-X-os`  
**Protocol:** ULTRA_WIN_AGENT_PROTOCOL  

---

## Project one-liner vs brief

| | |
|---|---|
| **Brief** | 5-class astronomical image classification from raw pixels (Spiral / Elliptical / Nebula / Star Cluster / Planetary); >80% accuracy; <5s inference; Grad-CAM; optional bonuses; reproducible code + demo. |
| **Product** | EfficientNet-B3 transfer-learning pipeline (PyTorch/timm) + Streamlit demo + Grad-CAM + template caption/anomaly/localization bonuses. Claimed **93.17%** test accuracy on ~249 held-out images. |
| **Verdict** | **Real ML product that matches the brief**, not a toy scaffold. Hostile risk is **reproducibility friction** (empty `data/processed`, checkpoint not in git) + **WIP churn** (uncommitted multi-backbone model + unwired `tta.py`) + **docs overclaiming COMPLETE**. |

---

## Honest % by axis (evidence-backed)

| Axis | % | EXISTS | FAKE / WEAK | Evidence |
|------|---|--------|-------------|----------|
| **Problem fit** | **88** | Full pipeline maps to PS FRs | Folder name Galaxy-X-os vs checklist `scale_odyssey/` | `PROBLEM_STATEMENT.md`, `src/*`, `app/app.py` |
| **Golden path** | **52** | Checkpoint loads; inference works | **0 images** under `data/processed/{train,val,test}`; no committed demo samples; stranger cannot re-run `evaluate.py` without `prepare_data` | Baseline probe 2026-07-25 |
| **Honesty of state** | **65** | Manifest admits nebula fallback | README/SUBMISSION/checklist say Full/Complete; WIP not frozen | Docs vs git status |
| **Security** | **72** | `weights_only=True` load; no secrets in src; security.yml | Streamlit open upload (local demo OK); naive secret grep; no upload size/type hard limits beyond Streamlit | `inference.py`, `.github/workflows/security.yml` |
| **Auth/RBAC** | **N/A→90** | Local demo, no multi-tenant claim | — | SCOPE_GUARD: no prod deploy |
| **Data truth** | **58** | Honest `DATA_MANIFEST.json`; MD5 leak claim in docs | **Empty processed tree**; nebula `is_real:false`, `pct_real:"unknown"`; full ~2500 images not local | Manifest + `find data/processed` |
| **API honesty** | **80** | No fake REST KPIs | Caption is **template** in app (honest caption text) | `app/app.py` `generate_template_caption` |
| **Frontend UX** | **70** | Clean Streamlit UI, Grad-CAM, probs, anomaly | No sample images when empty; no hosted URL | Manual + code review |
| **Realtime** | **N/A** | Not claimed | — | SCOPE_GUARD |
| **AI (model)** | **86** | Live EfficientNet-B3 weights, not mock | Cold inference ~0.5–1.0s MPS; first load ~2s | Live probe: load + predict |
| **Integrations** | **68** | NASA/Galaxy10 download path real | Network soft-fallback to procedural; BLIP optional soft-fail | `prepare_data.py`, `bonus.py` |
| **Jobs** | **N/A** | Train is CLI, not workers | — | |
| **Tests** | **55** | 46 unit tests PASS | E2E is import-only theater; **no accuracy regression** against claimed 93%; no checkpoint load test | `pytest tests/unit` + `tests/e2e/test_app.py` |
| **CI/CD** | **55** | ci.yml + test matrix + security.yml | Full `requirements.txt` install (heavy torch); may flake; no artifact smoke | `.github/workflows/*` |
| **Live deploy** | **40** | Dockerfile + compose | No public hosted demo; Docker does not COPY checkpoints/data | `Dockerfile` |
| **Docs** | **82** | Strong README/REPORT/HOW_TO_RUN/REPRO | HOW_TO_RUN cites missing `generate_splits.py` / `train_head.py`; greenwash Complete | `HOW_TO_RUN.md` |
| **Competitive moat** | **74** | Real-first multi-source data, Grad-CAM quality, demo video, honest metrics | Empty local data + uncommitted WIP can erase moat in 10 min judge session | |

### Competition rubric (official weights)

| Criterion | Weight | Cell % | Weighted | Notes |
|-----------|--------|--------|----------|-------|
| Classification Performance | 40% | **78** | 31.2 | JSON + plots committed; **not re-runnable** without data; spiral/elliptical residual confusion honest; TTA slightly *worse* than standard |
| Model Efficiency | 15% | **90** | 13.5 | ~11.6M params; inference ≪5s on MPS (measured ~0.5–1.0s after warmup) |
| Explainability | 15% | **88** | 13.2 | 15 Grad-CAM samples + summary + app integration |
| Innovation / Bonus | 15% | **76** | 11.4 | App + anomaly + localization + astro augs; BLIP not default path in app; ensemble/TTA WIP unwired |
| Documentation & Presentation | 15% | **82** | 12.3 | REPORT + video (~76s) + README; some COMPLETE theater |
| **BLENDED** | 100% | | **~81.6 → round to 72 after reproducibility penalty** | Cap applied for empty data + stranger golden-path friction |

### Protocol blended score (hostile)

**Honest overall: 72%** (YELLOW / strong mid-late stage, not freeze-ready)

Reason: Core ML + artifacts are real and above the 80% accuracy bar *if* the judge trusts committed `results/`. Hostile 10-minute kick-out still easy via empty data, missing git checkpoint, and WIP `model.py` defaults.

---

## Golden path (sacred)

```text
1. pip install -r requirements.txt
2. Obtain checkpoints/best_model.pth (Release v1.0 OR train)
3. streamlit run app/app.py  → upload image → class + conf + Grad-CAM + caption + OOD
4. (Full science path) prepare_data → train → evaluate → gradcam
```

| Step | Status | Blocker |
|------|--------|---------|
| Install | YELLOW | Heavy deps; Python 3.14 used locally vs 3.10 CI |
| Checkpoint present (local) | GREEN | `checkpoints/best_model.pth` ~141MB exists on this machine |
| Checkpoint in clone | RED | gitignored; Release path documented |
| Processed data | RED | 0 images in train/val/test |
| Inference live | GREEN | Probed 2026-07-25 |
| Evaluate re-run | RED | No test images |
| Streamlit | YELLOW | Works with local ckpt; unproven in this audit session via browser |
| Grad-CAM CLI | YELLOW | Code OK; needs images for fresh run (committed PNGs exist) |

---

## EXISTS vs FAKE/WEAK map

| Component | Status |
|-----------|--------|
| `src/train.py` progressive unfreeze + AMP | EXISTS |
| `src/evaluate.py` metrics + 6× TTA | EXISTS |
| `src/gradcam.py` + `results/gradcam/*` | EXISTS (artifacts committed) |
| `src/inference.py` ModelManager | EXISTS (live load OK) |
| `app/app.py` Streamlit | EXISTS |
| `results/evaluation_results.json` 93.17% | EXISTS (historical run; not re-verified today) |
| `data/processed` images | **EMPTY / FAKE for local re-run** |
| `src/tta.py` advanced TTA | **DEAD CODE** (untracked, not imported) |
| Multi-backbone + `AstroEnsemble` in working tree | **WIP** (uncommitted; default was flipped to `convnext_base`) |
| E2E app test | **THEATER** (imports only) |
| Hosted demo | **MISSING** |
| Demo video | EXISTS (`docs/presentation/demo.mp4` ~75.8s) |

---

## P0 / P1 / P2 gaps

### P0 — kick-out / disqualify risk
1. **Empty `data/processed` splits** — `evaluate.py` / train / fresh gradcam collapse for a stranger.
2. **Checkpoint not in repo** — without Release download or train, app exits with FileNotFoundError.
3. **Uncommitted `model.py` default backbone `convnext_base`** — breaks any call site that relies on default (checkpoint is EfficientNet-B3). Golden path poison if merged as-is.
4. **Docs claim Full/Complete** while local tree cannot re-eval.

### P1 — hostile score cuts
5. No **accuracy regression test** (even tiny subset) — 93.17% is unproven in CI.
6. **E2E theater** — does not open Streamlit or predict.
7. **Nebula provenance weak** (`is_real:false`, pct unknown) under “primarily real” narrative.
8. **HOW_TO_RUN stale** (`generate_splits.py`, `train_head.py`, `augmentations.py` missing).
9. **`src/tta.py` / ensemble** unwired — looks like unfinished scope creep.
10. No **committed demo samples** for upload without full dataset.
11. Integration test depends on network (`prepare_data --per-class 10`).

### P2 — polish / moat
12. TTA does not improve accuracy (92.77% < 93.17%) — either fix or reframe.
13. Docker image incomplete (no ckpt/data in image).
14. App UX: no built-in example gallery.
15. CI may be slow/flaky with full torch pin matrix.
16. Security.yml secret scan is trivial (string grep).
17. Submission checklist still references `scale_odyssey/` layout.
18. Localization listed out-of-scope in SCOPE_GUARD but implemented as bonus — clarify, don’t hide.

---

## Top 15 kick-out bugs (hostile judge script)

1. Clone → `python src/evaluate.py` → fails / empty dataset.
2. Clone → `streamlit run app/app.py` → no checkpoint → error screen.
3. Merge WIP `model.py` with default `convnext_base` → silent architecture mismatch if someone omits backbone kwarg.
4. Judge asks “prove 93% now” → cannot without Colab/data rebuild.
5. Open `src/tta.py` → “why isn’t this used?” unfinished work demerit.
6. Nebula `is_real:false` vs marketing “primarily real.”
7. E2E “passes” while app never loads model in CI.
8. TTA hurts metrics — “optimization theater.”
9. HOW_TO_RUN dead Makefile targets confuse graders.
10. Attic `gradcam-broken/` shows earlier wrong CAMs — if discovered, trust hit (mitigated if clearly attic).
11. No public demo URL vs competitors who host Streamlit Cloud.
12. Python 3.14 local vs 3.10 CI drift.
13. BLIP advertised in BONUS docs but app uses template caption only (honest in UI; easy to oversell verbally).
14. `weights_only=True` good, but full checkpoint still contains optimizer state (size / load semantics).
15. Parallel agent WIP on model/ensemble without freeze → last-minute break of Grad-CAM target layers for new backbones.

---

## Baseline probes (summary)

| Probe | Result | Time |
|-------|--------|------|
| `pytest tests/unit -v` | **46 passed** | ~8s |
| `pytest tests/e2e -v` | **1 passed** (import-only) | ~6s |
| Load `best_model.pth` via ModelManager | **OK** efficientnet_b3, 11,620,397 params, device=mps | ~1.9s |
| Synthetic image predict | **OK** (class + conf); latency ~0.5–1.0s reported | |
| Image count `data/processed/**` | **0** | |
| Secrets grep src/app | clean | |
| demo.mp4 duration | ~75.8s | |

Full paste: `work/reports/P0-baseline.md`

---

## Competitive moat assessment

**Why a judge might pick this:** clear 5-class taxonomy, real survey + NASA sources with manifest honesty, solid Grad-CAM package, demo video, 93% reported with spiral/elliptical residual acknowledged, modular code, Streamlit golden path.

**Why a judge might kick it:** cannot reproduce metrics from a clean clone in 10 minutes; empty data; incomplete multi-model WIP; no hosted demo.

---

## Recommended kill order (next)

1. Truth-reset docs + SCOREBOARD (Phase 0)  
2. Protect EfficientNet-B3 golden path; quarantine/wire WIP (Phase 1 integrity)  
3. Demo sample pack + checkpoint instructions + optional smoke eval (Phase 2)  
4. Accuracy regression + real e2e (Phase 8 early)  
5. Only then ensemble/heavy TTA if proven lift (Phase 3/5)  
6. UI + freeze (Phase 7/9)
