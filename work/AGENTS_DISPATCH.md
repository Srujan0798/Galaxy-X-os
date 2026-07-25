# AGENTS DISPATCH — Copy-paste prompts

**Plan of record:** [`work/ETERNAL_FINAL_PLAN.md`](ETERNAL_FINAL_PLAN.md)  
**Score now:** 74% · **Next critical:** Agent B Phase 2  
**Root:** `/Users/srujansai/Desktop/Galaxy-X-os`

---

## Global prefix (paste above every agent)

```text
You operate under ULTRA WIN AGENT PROTOCOL.
Project root: /Users/srujansai/Desktop/Galaxy-X-os
Read first: work/ETERNAL_FINAL_PLAN.md (your phase), HANDOFF.md, docs/SCOREBOARD.md, PROBLEM_STATEMENT.md.
Laws: evidence or it didn't happen; disjoint write-sets; no "100%/done" without freeze; fail loud; protect EfficientNet-B3 golden path; confirm before push/deploy.
When finished: write work/reports/PHASE-N-evidence.md with pasted commands+exit codes; list files changed; residual risks.
Do not invent metrics. Do not touch files outside your write-set.
```

---

## Agent A — Phase 1 Integrity — DONE (re-verify only)

**Write-set:** `src/model.py`, `tests/unit/test_model.py`

```text
[GLOBAL PREFIX]
You are Agent A. Phase 1 is DONE. Only re-verify if model.py was touched:
- default backbone == efficientnet_b3
- pytest tests/unit/test_model.py
- ModelManager load checkpoints/best_model.pth if present
Do not expand ensemble training.
```

---

## Agent B — Phase 2 Golden path — **ASSIGN FIRST**

**Write-set:** `data/samples/**`, `scripts/verify_golden_path.sh`, `app/app.py` (samples + missing-ckpt UX only), `README.md` (Quick Start / ckpt only)

```text
[GLOBAL PREFIX]
You are Agent B (Golden Path) — CRITICAL PATH.
Phase 2 of work/ETERNAL_FINAL_PLAN.md.

DO:
1. Create data/samples/{spiral_galaxy,elliptical_galaxy,nebula,star_cluster,planetary_object}/ with ≥1 PNG each.
2. data/samples/README.md: demo-only disclaimer (not full test set; not for 93% claims).
3. scripts/verify_golden_path.sh: check samples; if no ckpt exit 2 with Release/Colab instructions; if ckpt run one predict exit 0.
4. Streamlit: sample picker/buttons that run full predict + Grad-CAM + caption + OOD.
5. README Quick Start: install → get ckpt (Release v1.0) → streamlit run app/app.py → click sample.

ACCEPTANCE:
test $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ') -ge 5
bash scripts/verify_golden_path.sh
# streamlit sample path works

FORBIDDEN: src/model.py, src/evaluate.py, metric invention, claiming samples prove 93%.
Evidence: work/reports/PHASE-2-evidence.md
```

---

## Agent C — Phase 3 TTA depth

**Write-set:** `src/tta.py`, `src/evaluate.py`, `tests/unit/test_tta.py`, `attic/**`

```text
[GLOBAL PREFIX]
You are Agent C (Depth/TTA). Phase 3.
Choose PATH ATTIC (default if no full test data) or PATH WIRE (only with measured results).

ATTIC: move src/tta.py to attic/; ensure no imports; keep existing evaluate 6× TTA.
WIRE: --tta-mode {off,simple,advanced}; tests for shapes; no claim of accuracy lift without full re-eval JSON.

Ensemble stays experimental (no multi-ckpt training unless orchestrator orders).
Evidence: work/reports/PHASE-3-evidence.md
```

---

## Agent D — Phase 4 Completeness / honesty

**Write-set:** `SUBMISSION.md`, `BONUS_FEATURES.md`, `docs/presentation/**`, `docs/MODEL_CARD.md`, README status blurb, `REPORT.md` honesty-only

```text
[GLOBAL PREFIX]
You are Agent D (Docs honesty). Phase 4.
Map every PROBLEM_STATEMENT deliverable and rubric cell → file path → proof command.
Kill false Complete/100%/production-ready language.
Align caption docs with app (template default; BLIP optional).
Fix Submission_Checklist scale_odyssey/ → Galaxy-X-os.
Validate docs/MODEL_CARD.md or remove if wrong.
Evidence: work/reports/PHASE-4-evidence.md
```

---

## Agent E — Phase 5 Brownies

**Write-set:** `src/bonus.py`, `src/detection.py`, `src/gradcam_plus.py`, `src/onnx_export.py`, `src/pseudo_label.py`, related unit tests

```text
[GLOBAL PREFIX]
You are Agent E (Brownies). Phase 5.
For EACH of detection.py, gradcam_plus.py, onnx_export.py, pseudo_label.py: WIRE with test/CLI OR move to attic.
Keep bonus OOD + template caption solid.
Optional: localization overlay, ONNX export docs.
Do not break golden path. Coordinate app UI with Agent G if needed.
Evidence: work/reports/PHASE-5-evidence.md
```

---

## Agent F — Phase 6 Architecture

**Write-set:** `Makefile`, `HOW_TO_RUN.md`, `docs/SCOPE_GUARD.md`, `Dockerfile` comments, fail-loud helpers (`src/utils.py` + call sites if allowed)

```text
[GLOBAL PREFIX]
You are Agent F (Architecture). Phase 6.
Remove dead Makefile/HOW_TO_RUN targets (generate_splits, train_head, missing augmentations).
Fail loud when data/processed empty (evaluate/train/gradcam).
SCOPE_GUARD: localization is optional bonus.
Dockerfile: note mount ckpt/data.
Evidence: work/reports/PHASE-6-evidence.md
```

---

## Agent G — Phase 7 UI

**Write-set:** `app/app.py` only (after Phase 2 merged)

```text
[GLOBAL PREFIX]
You are Agent G (UI). Phase 7.
Polish sample gallery, missing-ckpt error (Release+Colab links), a11y confidence, mobile layout.
No new heavy deps. Preserve golden path behavior.
Evidence: work/reports/PHASE-7-evidence.md
```

---

## Agent H — Phase 8 Automated proof

**Write-set:** `tests/e2e/**`, `tests/unit/test_checkpoint_smoke.py`, `tests/integration/**`, `.github/workflows/**`

```text
[GLOBAL PREFIX]
You are Agent H (Proof). Phase 8.
Add checkpoint+sample predict smoke (skip if no ckpt).
Replace import-only e2e as sole test with real predict path.
Mark network tests; CI runs unit + not network.
pip cache; don't claim 93% in CI.
Evidence: work/reports/PHASE-8-evidence.md
```

---

## ORCH — Phase 9 Freeze

**Write-set:** `work/**`, `docs/SCOREBOARD.md`, `HANDOFF.md`

```text
[GLOBAL PREFIX]
You are Freeze Orchestrator. Phase 9.
Re-run freeze command pack from ETERNAL_FINAL_PLAN.md.
Hostile 10-minute kick-out attempt.
ONLY if all freeze boxes green: write work/reports/FREEZE.md with evidence, SCOREBOARD 100%, HANDOFF FREEZE.
Else: real % + remaining reds. Never invent 100%.
```

---

## Suggested assignment order

```text
1) B          → merge
2) C D E F H  → parallel (disjoint write-sets)
3) G          → after B
4) ORCH freeze
```

## Parallel safety

| Safe together | Not together |
|---------------|--------------|
| C ∥ D ∥ E ∥ F ∥ H after B | B ∥ G on app/app.py |
| A re-verify anytime | C ∥ F both rewriting evaluate.py main without helper |
| E attic moves ∥ H tests | Two agents on README freely |

---

## Definition of agent “done”

1. Acceptance commands from plan pass (pasted).  
2. `work/reports/PHASE-N-evidence.md` exists.  
3. Write-set only.  
4. Golden path not broken (verify script or manual note).  
5. No 100% claim.
