# ULTRA ULTIMATE 100% PLAN — Galaxy-X-os

**Status:** AGENT-READY DISPATCH DOCUMENT  
**Protocol:** ULTRA WIN AGENT PROTOCOL  
**Project root:** `/Users/srujansai/Desktop/Galaxy-X-os`  
**Repo:** https://github.com/Srujan0798/Galaxy-X-os  
**Last updated:** 2026-07-25  

| Field | Value |
|-------|--------|
| **Honest baseline** | **74%** (Phase 0 audit + Phase 1 integrity) |
| **Target** | **100% freeze** only with checklist + pasted evidence |
| **Competition** | SCALE × ODYSSEY — TechOIITGN |
| **Sacred golden path** | EfficientNet-B3 ckpt → Streamlit sample/upload → class + conf + Grad-CAM + OOD |
| **Brief** | `PROBLEM_STATEMENT.md` |
| **Audit** | `work/reports/BRUTAL_AUDIT.md` |
| **Scoreboard** | `docs/SCOREBOARD.md` |
| **Handoff** | `HANDOFF.md` (replace only, never append forever) |

> **Law:** No agent may write “done / 100% / submission-ready / production-ready” without freeze checklist green and **pasted terminal/browser evidence**.

---

# 0. Intent lock (every agent reads this first)

1. **Win the judge rubric**, not invent a new product. Weights: Performance 40% · Efficiency 15% · Explainability 15% · Bonus 15% · Docs 15%.
2. **Primary model = EfficientNet-B3** (`configs/config.yaml`, `checkpoints/best_model.pth`). Multi-backbone / ensemble is **opt-in experimental only** until separate trained weights + measured metrics exist.
3. **Sacred golden path must never break.** Every PR proves: load ckpt → predict → Grad-CAM path still works.
4. **Evidence or it didn’t happen.** Paste commands + exit codes into `work/reports/PHASE-{N}-evidence.md`.
5. **Disjoint write-sets.** Touch only your files. If you need another file, stop and request orchestrator reassignment.
6. **No fake live intelligence.** No mock labels sold as model output. Template captions must stay labeled “template.”
7. **Fail loud.** Empty data, missing ckpt, bad image → clear error with next action. No silent empty success.
8. **Confirm before** `git push`, force-push, Release overwrite, deploy to public host, or delete data/checkpoints.
9. **WIP debt is a kick-out risk.** Untracked modules without tests/docs/wiring = demerit. Wire with proof OR attic.
10. **Honesty over greenwash.** Prefer “YELLOW with evidence” over false GREEN.

### Out of scope (do not burn cycles)
- Production multi-tenant auth / RBAC  
- Real-time telescope feeds  
- Fabricating or “rounding up” metrics  
- Replacing EfficientNet as default without full retrain + new RESULTS JSON  
- Scope creep before Phase 2 green  

---

# 1. Current truth (do not argue with evidence)

### What is real
| Item | Truth |
|------|--------|
| Pipeline modules | `prepare_data`, `train`, `evaluate`, `gradcam`, `inference`, `bonus`, Streamlit `app` |
| Checkpoint (this machine) | `checkpoints/best_model.pth` ~141MB, backbone `efficientnet_b3`, epoch 23, best_val≈0.95 |
| Claimed test metrics | 93.17% acc / 0.932 macro F1 (std); TTA 92.77% — in `results/evaluation_results.json` |
| Grad-CAM package | 15 samples + summary under `results/gradcam/` |
| Demo video | `docs/presentation/demo.mp4` ~76s |
| Unit tests | 46+ pass (incl. default-backbone contract) |
| Live inference probe | MPS load OK; predict ~0.5–1s; 11.6M params |

### What is broken / weak
| Item | Truth |
|------|--------|
| `data/processed/{train,val,test}` | **0 images** |
| Checkpoint in git | **No** (GitHub 100MB limit; Release/Colab path) |
| Stranger 10-min path | **Fails** without ckpt download + samples |
| E2E tests | **Import-only theater** |
| Accuracy in CI | **Not proven** |
| Hosted demo | **None** |
| Docs | Some “Complete” theater; HOW_TO_RUN dead targets |

### WIP / untracked debt (must resolve before freeze)

| Path | Risk | Decision in this plan |
|------|------|------------------------|
| `src/model.py` (modified) | Multi-backbone + ensemble | **Keep opt-in**; default **must stay** `efficientnet_b3` (Phase 1 done) |
| `src/tta.py` | Unwired advanced TTA | Phase 3: wire **or** attic |
| `src/gradcam_plus.py` | Unwired | Phase 5/C: wire optional **or** attic |
| `src/detection.py` | Unwired localization/detection | Phase 5: integrate with bonus **or** attic |
| `src/onnx_export.py` | Unwired | Phase 5 brownie **or** attic |
| `src/pseudo_label.py` | Unwired | Phase 5 experimental **or** attic |
| `docs/MODEL_CARD.md` | May be good | Phase 4: review & link from README if accurate |

---

# 2. Score path: 74% → 100%

| Phase | Name | Est. score after | Gate |
|-------|------|------------------|------|
| 0 | Truth reset | 72% | DONE |
| 1 | Integrity fortress | **74%** | DONE (default backbone + contract test) |
| 2 | Golden path live | **84–86%** | Samples + verify script + Streamlit samples |
| 3 | Depth (TTA/WIP resolve) | **86–90%** | Wire-or-attic; no metric lies |
| 4 | Completeness & honesty docs | **90–92%** | Rubric map; kill COMPLETE theater |
| 5 | Brownies (real bonuses) | **92–94%** | App-visible bonuses with tests |
| 6 | Architecture polish | **94–95%** | Fail-loud; Makefile; empty-data UX |
| 7 | UI domination | **96–97%** | Gallery, a11y, fail states |
| 8 | Automated proof | **98–99%** | Real e2e + CI smoke |
| 9 | Hostile freeze | **100% only if checklist** | Re-audit + evidence pack |

If any phase red: report **real %** and stop claiming progress.

---

# 3. Official rubric → work items

| ID | Criterion | Wt | Now | To reach GREEN | Owner phase |
|----|-----------|----|-----|----------------|-------------|
| R1 | Classification Performance | 40% | 78 YELLOW | (a) Keep artifact honesty + SHA256 of results JSON + ckpt; (b) **mini re-eval** on samples OR documented Colab re-run path; (c) spiral/elliptical residual already honest | 2, 4, 8 |
| R2 | Model Efficiency | 15% | 90 GREEN | Benchmark table: CPU / MPS / CUDA cold+warm ms; stay ≪5s | 6, 8 |
| R3 | Explainability | 15% | 88 GREEN | Sample Grad-CAM in app from gallery; optional GradCAM++ only if tested | 2, 5, 7 |
| R4 | Innovation / Bonus | 15% | 76 YELLOW | Visible bonuses: OOD + caption + loc/ONNX optional; kill dead WIP | 3, 5 |
| R5 | Docs & Presentation | 15% | 82 YELLOW | Honest STATUS; HOW_TO_RUN fixed; SUBMISSION map; video already exists | 4, 6 |

---

# 4. Sacred golden path (never break)

```text
[Judge machine]
  pip install -r requirements.txt
  # Obtain weights (ONE of):
  #   A) GitHub Release v1.0 → checkpoints/best_model.pth
  #   B) Colab notebook notebooks/Galaxy_X_Colab.ipynb
  #   C) train after prepare_data (slow)
  streamlit run app/app.py
  → Click sample OR upload PNG
  → See: predicted class, confidence, bar chart, Grad-CAM, caption, OOD
  Optional science path:
  python src/prepare_data.py && python src/train.py && python src/evaluate.py && python src/gradcam.py
```

**Protect with:** `scripts/verify_golden_path.sh` + pytest smoke (Phase 2 + 8).

---

# 5. Dependency DAG (execution order)

```text
Phase 0 ──► Phase 1 ──► Phase 2 ──┬──► Phase 3 ──┐
                    │            ├──► Phase 4 ──┤
                    │            ├──► Phase 5 ──┼──► Phase 7 ──► Phase 8 ──► Phase 9
                    │            ├──► Phase 6 ──┤
                    └── Phase 6 early (Makefile/docs only, no app) may start after Phase 1
```

**Hard rules**
- Phase 7 **must not** start until Phase 2 merged (shared `app/app.py`).
- Phase 5 app toggles: either land in Phase 2’s sample API contract OR wait for Phase 7 merge window.
- Phase 9 only when 2+4+6+8 green and 3+5 resolved (wire or attic).

**Max parallelism after Phase 2 green:** Agents C, D, E, F, H in parallel (disjoint files).  
**Serial on app:** B then G (and E only for clearly marked checkbox blocks if coordinated).

---

# 6. Agent roster & exclusive write-sets

| Agent | Role | Phases | Exclusive write-set (ONLY these) |
|-------|------|--------|----------------------------------|
| **ORCH** | Orchestrator / freeze | 0, 9 | `work/**`, `docs/SCOREBOARD.md`, `HANDOFF.md`, this plan |
| **A** | Integrity | 1 | `src/model.py`, `tests/unit/test_model.py` — **DONE** |
| **B** | Golden path | 2 | `data/samples/**`, `scripts/verify_golden_path.sh`, `app/app.py` (samples + ckpt error only), `README.md` (Quick Start / ckpt only), `HOW_TO_RUN.md` (quick start only if F not active—prefer B owns sample section, F owns Makefile cleanup) |
| **C** | Depth / WIP TTA | 3 | `src/tta.py`, `src/evaluate.py`, `tests/unit/test_tta.py`, optionally `attic/**` moves of tta |
| **D** | Completeness docs | 4 | `SUBMISSION.md`, `BONUS_FEATURES.md`, `docs/presentation/**`, `docs/MODEL_CARD.md`, `REPORT.md` honesty lines only (no metric invention), `docs/presentation/Submission_Checklist.md` |
| **E** | Brownies | 5 | `src/bonus.py`, `src/detection.py`, `src/gradcam_plus.py`, `src/onnx_export.py`, `src/pseudo_label.py`, `tests/unit/test_bonus.py`, `tests/unit/test_localization.py`, new tests under `tests/unit/test_{detection,onnx,gradcam_plus,pseudo}.py` — **app hooks only if B/G merge protocol used** |
| **F** | Architecture | 6 | `Makefile`, `HOW_TO_RUN.md` (dead targets + structure), `docs/SCOPE_GUARD.md`, `src/evaluate.py` **only empty-dataset guard** if C not touching same lines—prefer F adds `src/utils.py` helper + evaluate calls it; `Dockerfile` honesty comments; `configs/config.yaml` comments only |
| **G** | UI domination | 7 | `app/app.py` (CSS/layout/a11y/error states) after B merged |
| **H** | Automated proof | 8 | `tests/e2e/**`, `tests/unit/test_checkpoint_smoke.py`, `tests/integration/**` (non-network prefer), `.github/workflows/**` |

### Conflict resolution
| File | Owner priority |
|------|----------------|
| `app/app.py` | B (Phase 2) → G (Phase 7) → E (optional checkboxes only with G) |
| `src/evaluate.py` | C owns TTA; F may only add empty-dir check at top of `main` if C finished or via shared helper in `utils.py` |
| `README.md` | B Quick Start; D may add “Status honesty” section once; no third writer |
| `src/model.py` | A only; freeze after Phase 1 unless ORCH reopens |

---

# 7. Phase specs (full)

---

## PHASE 0 — Truth reset — ORCH — **DONE**

**Outcome:** Baseline 72–74% locked; no false COMPLETE.  
**Artifacts:** `work/reports/BRUTAL_AUDIT.md`, `P0-baseline.md`, `docs/SCOREBOARD.md`, this plan, `HANDOFF.md`.

**Acceptance**
```bash
test -f work/reports/BRUTAL_AUDIT.md
test -f work/reports/P0-baseline.md
test -f docs/SCOREBOARD.md
test -f work/ETERNAL_FINAL_PLAN.md
```

---

## PHASE 1 — Integrity fortress — Agent A — **DONE**

**Outcome:** Default backbone = `efficientnet_b3`; ckpt still loads.  
**Done evidence:** `test_default_backbone_is_efficientnet_b3` PASS; `CKPT_LOAD_OK`.

**If regression later:** re-run Phase 1 acceptance before any model PR merge.

**Acceptance (re-verify anytime)**
```bash
python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'"
python3 -m pytest tests/unit/test_model.py -v
# if ckpt present:
python3 -c "from src.inference import ModelManager; ModelManager('checkpoints/best_model.pth'); print('CKPT_LOAD_OK')"
```

---

## PHASE 2 — Core golden path — Agent B — **CRITICAL PATH**

**Target:** +10–12% → ~84–86%  
**Mindset:** A stranger completes primary job in &lt;10 minutes.

### Tasks
1. **Create demo pack** `data/samples/{spiral_galaxy,elliptical_galaxy,nebula,star_cluster,planetary_object}/`  
   - ≥1 PNG per class (prefer 2).  
   - Sources: run `python src/prepare_data.py --per-class 10 --output-dir /tmp/gx` and copy a few, OR download small NASA thumbnails, OR export from existing Grad-CAM **inputs if available**.  
   - Add `data/samples/README.md`: “Demo only — not the full test set; do not cite accuracy from these alone.”
2. **`scripts/verify_golden_path.sh`**
   - Fail if no samples.
   - If no ckpt: print clear message + Release URL; exit 2 (not 0).
   - If ckpt: run one `ModelManager.predict` on a sample; print class + conf + ms; exit 0.
3. **Streamlit samples**
   - Sidebar or main: buttons/selectbox “Try sample: …” loading from `data/samples`.
   - Must run full predict + Grad-CAM + caption + OOD path (reuse existing renderers).
4. **Docs (minimal)**
   - README Quick Start: 3-step demo (install → get ckpt → streamlit).
   - Checkpoint: Release v1.0 link + SHA256 instructions (`docs/REPRODUCIBILITY.md` already exists—link it).
5. **Do NOT** claim samples prove 93% accuracy.

### Acceptance
```bash
test $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ') -ge 5
test -f data/samples/README.md
bash scripts/verify_golden_path.sh   # exit 0 with local ckpt; exit 2 without is OK if message clear
# Manual browser:
# streamlit run app/app.py → click sample → prediction + Grad-CAM visible
```

### Evidence file
Write `work/reports/PHASE-2-evidence.md` with script output + screenshot note or playwright log.

### Copy-paste agent prompt (B)
```text
You are Agent B (Golden Path) on Galaxy-X-os under ULTRA WIN protocol.
Read: work/ETERNAL_FINAL_PLAN.md Phase 2, PROBLEM_STATEMENT.md, HANDOFF.md.
Write-set ONLY: data/samples/**, scripts/verify_golden_path.sh, app/app.py (sample gallery + clearer missing-ckpt error), README.md Quick Start/ckpt section only.
Goal: stranger can demo in <10 min. ≥1 image per class under data/samples/. verify_golden_path.sh. Streamlit sample buttons.
Do not invent metrics. Do not touch src/model.py or src/evaluate.py.
When done: paste acceptance command output into work/reports/PHASE-2-evidence.md and update HANDOFF via orchestrator note.
Sacred path: EfficientNet-B3 ckpt → sample → class+conf+Grad-CAM+OOD.
```

---

## PHASE 3 — Depth / TTA & WIP resolve — Agent C

**Target:** +2–6% if quality improves; **0 if only cleanup** (cleanup still mandatory).  
**Depends on:** Phase 2 green recommended (not hard for attic path).

### Decision tree (pick ONE path — document choice in evidence)

**Path WIRE (only if you will measure):**
1. Integrate `src/tta.py` into `evaluate.py` via `--tta-mode {off,simple,advanced}`.
2. `simple` = existing 6× TTA (current evaluate behavior).
3. `advanced` = `tta.py` transforms.
4. Run on available data if any; if no full test set, measure on samples only and **label as smoke, not leaderboard**.
5. Unit tests for transform list length + aggregation shapes.
6. Do **not** update claimed 93% unless full test re-eval completed and JSON rewritten honestly.

**Path ATTIC (default if no GPU/data time):**
1. Move `src/tta.py` → `attic/src-archive/tta.py`.
2. Ensure nothing imports it.
3. Keep current evaluate 6× TTA as the only TTA story in docs.

### Ensemble
- `AstroEnsemble` in `model.py`: leave code, add docstring **“experimental; no submission weights”**.  
- Do not train 3 backbones unless ORCH explicitly prioritizes (days of GPU).

### Acceptance
```bash
# Path ATTIC:
test ! -f src/tta.py
test -f attic/src-archive/tta.py -o -f attic/tta.py
rg -n "from tta|import tta" src/ app/ tests/ && exit 1 || true

# Path WIRE:
python src/evaluate.py --help | grep -E 'tta-mode|tta'
python3 -m pytest tests/unit/test_tta.py -v
```

### Copy-paste prompt (C)
```text
You are Agent C (Depth/TTA) on Galaxy-X-os. Read work/ETERNAL_FINAL_PLAN.md Phase 3.
Write-set: src/tta.py, src/evaluate.py, tests/unit/test_tta.py, attic/** for moves.
Choose WIRE or ATTIC. Prefer ATTIC if no full test set. No fake accuracy claims.
Paste evidence to work/reports/PHASE-3-evidence.md.
```

---

## PHASE 4 — Product completeness & honesty — Agent D

**Target:** +3–5% → docs trust  
**Depends on:** Phase 2 for accurate “how to demo” text.

### Tasks
1. **SUBMISSION.md** — every criterion → file path → proof command. Remove false “Complete” where YELLOW.
2. **Submission_Checklist.md** — fix `scale_odyssey/` → `Galaxy-X-os` layout.
3. **BONUS_FEATURES.md** — match reality: app uses **template** caption by default; BLIP optional if available.
4. **MODEL_CARD.md** — if present: validate against metrics JSON; link from README; if wrong, fix or delete.
5. **Executive_Summary / Presentation_Outline** — replace “100%/done” with honest status + link SCOREBOARD.
6. **REPORT.md** — only fix contradictions; do not change numbers without re-eval evidence.
7. Add short **STATUS** block at top of README:
   ```text
   Demo: local Streamlit + samples | Metrics: Colab artifact 93.17% (see results/) | Scoreboard: docs/SCOREBOARD.md
   ```

### Acceptance
```bash
# No false universal Complete in checklist for missing ckpt-in-git
rg -n "Complete|100%|production-ready" SUBMISSION.md docs/presentation/Submission_Checklist.md | head -40
# Manual review: every claim has a path
test -f SUBMISSION.md && test -f docs/SCOREBOARD.md
```

### Copy-paste prompt (D)
```text
You are Agent D (Completeness/Honesty docs). Read PROBLEM_STATEMENT.md, docs/SCOREBOARD.md, work/ETERNAL_FINAL_PLAN.md Phase 4.
Write-set: SUBMISSION.md, BONUS_FEATURES.md, docs/presentation/**, docs/MODEL_CARD.md, README status blurb, REPORT.md honesty-only.
Kill greenwash. Align captions with app (template default). Map rubric → files → commands.
Evidence: work/reports/PHASE-4-evidence.md
```

---

## PHASE 5 — Brownies / innovation — Agent E

**Target:** +2–4%  
**Depends on:** Phase 2; coordinate app UI with G.

### WIP resolve matrix (mandatory for each file)

| File | Required action |
|------|-----------------|
| `src/bonus.py` | Keep; ensure CLI + functions tested; thresholds documented in docstring |
| `src/detection.py` | Integrate into bonus localization **or** attic |
| `src/gradcam_plus.py` | Optional CLI flag in gradcam **or** attic |
| `src/onnx_export.py` | Working `python src/onnx_export.py --help` + test export on tiny OR attic |
| `src/pseudo_label.py` | Document experimental OR attic |

### Product-visible brownie (pick ≥2 that work in app or CLI)
1. OOD panel (already) — keep solid  
2. Template caption (already) — keep honest  
3. Localization bbox overlay optional checkbox  
4. ONNX export documented for efficiency brownie  
5. GradCAM++ optional if quality ≥ Grad-CAM  

### Acceptance
```bash
python3 -m pytest tests/unit/test_bonus.py tests/unit/test_localization.py tests/unit/test_ood.py -v
# Every remaining src/*.py either imported by train/eval/app/bonus/gradcam OR tested OR in attic
# No untracked orphan modules without README note
```

### Copy-paste prompt (E)
```text
You are Agent E (Brownies). Read BONUS_FEATURES.md and Phase 5 of work/ETERNAL_FINAL_PLAN.md.
Write-set: src/bonus.py, src/detection.py, src/gradcam_plus.py, src/onnx_export.py, src/pseudo_label.py, related unit tests.
Wire-or-attic every WIP file. Prefer features that show in CLI or app without breaking golden path.
No mock intelligence. Evidence: work/reports/PHASE-5-evidence.md
```

---

## PHASE 6 — Architecture polish — Agent F

**Target:** +1–3%  
**Depends on:** Phase 1; can parallel Phase 2 if avoiding app.

### Tasks
1. **Makefile** — only real targets: install, split/prepare, train, evaluate, gradcam, app, test, lint, clean, verify.  
   - Remove or fix `train-head`, `generate_splits` dead refs.
2. **HOW_TO_RUN.md** — match real files under `src/` (no `augmentations.py` if missing).  
3. **Empty data fail-loud** — `evaluate.py` / `train.py` / `gradcam.py` main: if no images, exit 1 with:
   `No images in data/processed. Run: python src/prepare_data.py  OR use data/samples demo via streamlit.`
4. **SCOPE_GUARD.md** — localization = optional bonus (in scope as brownie).  
5. **Dockerfile** — comment that ckpt/data must be mounted; optional COPY samples.  
6. **Latency table** — script or doc section: one-command benchmark writing `results/latency_bench.json` if ckpt present.

### Acceptance
```bash
grep -n "generate_splits\|train_head" HOW_TO_RUN.md Makefile && exit 1 || true
make test   # or make -n test
# With empty processed (current state):
python src/evaluate.py ; test $? -ne 0
```

### Copy-paste prompt (F)
```text
You are Agent F (Architecture). Phase 6 of work/ETERNAL_FINAL_PLAN.md.
Write-set: Makefile, HOW_TO_RUN.md, docs/SCOPE_GUARD.md, Dockerfile comments, fail-loud guards (prefer utils helper), optional latency bench.
No dead Makefile targets. Evidence: work/reports/PHASE-6-evidence.md
```

---

## PHASE 7 — UI domination — Agent G

**Target:** +1–2%  
**Depends on:** Phase 2 merged.

### Tasks
1. Polish sample gallery UX (grid of 5 classes with thumbnails).  
2. Missing ckpt: big clear error + Release link + Colab link.  
3. Confidence colors accessible (not color-only).  
4. Mobile-friendly columns; reduce clutter.  
5. “How to read Grad-CAM” already in sidebar — keep.  
6. No new heavy deps.

### Acceptance
```bash
# Import still safe for tests
python3 -c "import ast; ast.parse(open('app/app.py').read())"
# Manual: streamlit run app/app.py — sample works; no-ckpt message works if renamed ckpt temporarily
```

### Copy-paste prompt (G)
```text
You are Agent G (UI). Phase 7. Write-set: app/app.py only (after Phase 2 samples exist).
Make demo feel intentional, not AI slop. Preserve golden path. Evidence: work/reports/PHASE-7-evidence.md
```

---

## PHASE 8 — Automated proof — Agent H

**Target:** +3–5%  
**Depends on:** Phase 2 samples; ckpt optional with skip.

### Tasks
1. **`tests/unit/test_checkpoint_smoke.py`**
   - Skip if no `checkpoints/best_model.pth`.
   - Else load ModelManager, predict first sample, assert class in `CLASS_NAMES_DISPLAY`, conf in (0,1], time &lt; 5000ms.
2. **Replace e2e theater** in `tests/e2e/test_app.py`
   - Keep import test.
   - Add predict smoke using samples (same skip rules).
3. **Integration** — prefer offline: transforms + empty-dir handling; keep network prepare_data test marked `@pytest.mark.network` and exclude from default CI if flaky.
4. **CI workflows**
   - `ci.yml`: lint + unit (no network).
   - Optional smoke job: if samples exist, run verify script with skip-no-ckpt.
   - Cache pip.  
   - Pin python 3.10 primary; matrix optional.
5. **Security.yml** — keep pip-audit; improve secret scan paths; don’t false-positive on “password” in comments if needed.

### Acceptance
```bash
python3 -m pytest tests/ -v -m "not network"
# With ckpt + samples:
bash scripts/verify_golden_path.sh
```

### Copy-paste prompt (H)
```text
You are Agent H (Automated Proof). Phase 8 of work/ETERNAL_FINAL_PLAN.md.
Write-set: tests/e2e/**, tests/unit/test_checkpoint_smoke.py, tests/integration/** (markers), .github/workflows/**
Real predict smoke with skip-if-no-ckpt. Kill import-only as sole e2e. Evidence: work/reports/PHASE-8-evidence.md
```

---

## PHASE 9 — Hostile freeze — ORCH only

**Target:** 100% **only if** all boxes green.

### Freeze checklist (all required)

```text
[ ] R1–R5: GREEN or honest labeled YELLOW with evidence paths in SCOREBOARD
[ ] Phase 2: verify_golden_path.sh exit 0 on machine with ckpt; samples ≥5
[ ] Phase 1 contract: default backbone efficientnet_b3 still true
[ ] No src orphan WIP: every module wired/tested or in attic
[ ] Security: weights_only on loads; no secrets in repo; pip-audit clean or documented
[ ] Golden path browser-proven (manual steps in PHASE-2 or 7 evidence)
[ ] No mock sold as live model output
[ ] pytest tests/ -m "not network" green
[ ] CI green on main (or documented local-only constraint)
[ ] README / SUBMISSION / SCOREBOARD honest; no COMPLETE theater
[ ] HANDOFF replaced with freeze summary
[ ] You would bet a hostile judge cannot kick this in 10 minutes
```

### Freeze commands pack
```bash
python3 -m pytest tests/ -v -m "not network"
bash scripts/verify_golden_path.sh
python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'"
find data/samples -type f | wc -l
# optional if data restored:
# python src/evaluate.py
ls results/evaluation_results.json results/gradcam/_summary_grid.png docs/presentation/demo.mp4
```

### If any box open
Report **REAL blended %** (update SCOREBOARD). **Never invent 100%.**

### Copy-paste prompt (ORCH freeze)
```text
You are the Freeze Orchestrator. Re-read work/reports/BRUTAL_AUDIT.md, docs/SCOREBOARD.md, all work/reports/PHASE-*-evidence.md.
Re-run freeze command pack. Hostile mindset: try to kick the project in 10 minutes.
Only if freeze checklist all green: set SCOREBOARD blended 100%, rewrite HANDOFF as FREEZE, write work/reports/FREEZE.md with pasted evidence.
Otherwise: list remaining reds and real %.
```

---

# 8. Evidence standard (every phase)

Each agent writes `work/reports/PHASE-{N}-evidence.md`:

```markdown
# Phase N evidence
Date:
Agent:
Commands run:
```
(paste)
```
Exit codes:
Files changed:
SCOREBOARD cells moved: X → Y
Residual risks:
```

ORCH updates `docs/SCOREBOARD.md` + replaces `HANDOFF.md` after each phase merge.

---

# 9. Skills / tools map (use deliberately)

| Situation | Use |
|-----------|-----|
| Unclear brief | `PROBLEM_STATEMENT.md` + SCOPE_GUARD |
| Load/train bugs | systematic-debugging; small repro |
| New behavior | TDD: test first in `tests/unit` |
| “Is it done?” | verification-before-completion + live probe |
| UI | frontend-design constraints for Streamlit |
| Pre-merge | code-reviewer on agent diff only |
| Security | weights_only audit; no secret commits |
| Data prep | `prepare_data.py --per-class N` into temp, then copy samples |

**Never:** install 30 skills and claim excellence.

---

# 10. Parallel sprint plan (suggested calendar)

### Sprint 1 (blocking) — 1 agent or serial
1. Agent B Phase 2  
2. Re-verify Phase 1 acceptance  

### Sprint 2 (parallel) — after B merges
| Agent | Work |
|-------|------|
| C | TTA wire-or-attic |
| D | Honesty docs |
| E | Brownie wire-or-attic |
| F | Makefile / fail-loud / HOW_TO_RUN |
| H | Real tests + CI |

### Sprint 3
| Agent | Work |
|-------|------|
| G | UI polish on stable app |
| ORCH | Hostile freeze |

---

# 11. Definition of 100% (non-negotiable)

You may set blended score to **100%** only when:

1. Freeze checklist in Phase 9 is **fully checked** with pasted evidence in `work/reports/FREEZE.md`.  
2. A hostile 10-minute script fails to find a kick-out:
   - empty samples  
   - missing demo instructions  
   - default backbone wrong  
   - import-only tests as sole proof  
   - dead WIP modules in `src/`  
   - false Complete claims  
3. Judge rubric: accuracy story is either re-runnable or **explicitly** “artifact from Colab run X with SHA256” and still &gt;80%.  
4. Efficiency &lt;5s proven on at least one consumer device measurement file.  
5. Grad-CAM visible in demo path.  
6. Bonus features that are advertised actually run.  
7. Docs match code.

**Anything less = real % only (update SCOREBOARD).**

---

# 12. Master acceptance script (orchestrator)

Save as `scripts/ultra_win_gate.sh` in Phase 2 or 8:

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "== defaults =="; python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'"
echo "== unit =="; python3 -m pytest tests/unit -q
echo "== samples =="; test "$(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ')" -ge 5
echo "== golden =="; bash scripts/verify_golden_path.sh || test $? -eq 2
echo "== no orphans check (manual list) =="; ls src/*.py
echo "GATE_PARTIAL_OK"
```

---

# 13. Kick-out bug burn-down (assign explicitly)

| # | Bug | Phase | Agent |
|---|-----|-------|-------|
| 1 | Empty processed / no samples | 2 | B |
| 2 | Ckpt not in git / unclear obtain | 2, 4 | B, D |
| 3 | Default backbone drift | 1 | A (done) |
| 4 | Unwired tta.py | 3 | C |
| 5 | E2E theater | 8 | H |
| 6 | No acc smoke | 8 | H |
| 7 | Nebula provenance narrative | 4 | D |
| 8 | TTA hurts metrics messaging | 3, 4 | C, D |
| 9 | No hosted demo | 2 optional; label local-only | D |
| 10 | Dead HOW_TO_RUN targets | 6 | F |
| 11 | Orphan detection/onnx/pseudo/gradcam+ | 5 | E |
| 12 | COMPLETE greenwash | 4 | D |
| 13 | App no sample gallery | 2, 7 | B, G |
| 14 | Empty evaluate silent/confusing | 6 | F |
| 15 | CI flaky/heavy | 8 | H |

---

# 14. One-page dispatcher (print this)

```text
PROJECT: Galaxy-X-os @ /Users/srujansai/Desktop/Galaxy-X-os
PLAN:    work/ETERNAL_FINAL_PLAN.md
NOW:     74%  →  assign B next (Phase 2)
ORDER:   B → (C∥D∥E∥F∥H) → G → ORCH freeze
NEVER:   claim 100% without Phase 9 FREEZE.md
GOLDEN:  efficientnet_b3 ckpt → streamlit sample → Grad-CAM
```

| Assign | Agent prompt section |
|--------|----------------------|
| Phase 2 | § Phase 2 copy-paste |
| Phase 3 | § Phase 3 copy-paste |
| Phase 4 | § Phase 4 copy-paste |
| Phase 5 | § Phase 5 copy-paste |
| Phase 6 | § Phase 6 copy-paste |
| Phase 7 | § Phase 7 copy-paste |
| Phase 8 | § Phase 8 copy-paste |
| Freeze | § Phase 9 copy-paste |

---

# 15. File ownership cheat sheet

```text
src/model.py              → A only (frozen after P1)
src/tta.py / evaluate.py  → C
src/bonus.py + orphans    → E
src/train.py              → touch only with ORCH approval (golden train path)
src/inference.py          → H may add tests only; code change needs ORCH
src/gradcam.py            → E if GradCAM++; else freeze
app/app.py                → B then G (E toggles with G)
data/samples/**           → B
scripts/*                 → B (verify), H (gate), F (make)
tests/unit/test_model.py  → A
tests/e2e/**              → H
docs/SCOREBOARD.md        → ORCH
HANDOFF.md                → ORCH
SUBMISSION.md             → D
Makefile / HOW_TO_RUN     → F (B may edit Quick Start once)
```

---

# 16. Final words for agents

You are not a hype bot.  
You are a gap-killer with evidence.  
**74% is the truth today.**  
**100% is a gate, not a vibe.**  
Execute your phase, paste proof, stop at red.

— ULTRA WIN / ETERNAL FINAL PLAN
