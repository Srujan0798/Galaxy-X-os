# Galaxy-X-os — Eternal Completion Design

**Date:** 2026-07-23
**Status:** Approved (design)
**Owner:** Orchestrator (Claude) + Worker agents
**Governance:** CLAUDE.md wave model — every wave ships before next begins; no delete (move to `attic/`).

## Goal

Bring Galaxy-X-os to a **provably reproducible, honest, defensible** submission state
for SCALE × ODYSSEY. Every number in the report must be reproducible from committed
artifacts. Target: >80% accuracy if compute allows; otherwise honest lower number with
hardware note. No fabricated or contradictory results.

## Problem (current real state — audited 2026-07-23)

1. `checkpoints/` empty — no `best_model.pth`. Report references it. **Not reproducible.**
2. `data/raw/`, `data/samples/` empty — pipeline produces nothing on clone.
3. Committed Grad-CAM images (`results/gradcam/`) show a broken/random model
   (wrong preds at high confidence) that **contradict** REPORT.md (which claims
   Elliptical F1=1.000, Nebula F1=1.000). Biggest scoring risk — reads as fabrication.
4. Code, docs, structure otherwise ~90% complete.

## Decisions (locked)

- **Compute:** Colab-first (free GPU); fall back to 8GB Mac (MPS) if Colab unavailable.
- **Scope:** Hybrid maximum — real trained model + honest matching results + bonus features.
- **Grad-CAMs:** Quarantine misleading images to `attic/`, regenerate from real model (W3).
- **Data:** Real-first (SDSS/Kaggle) with a documented safe fallback so the run never breaks.

## Architecture of the completion effort

GPU training runs on Colab (human-driven). Split of labor:
- **Agents** build all code + a one-click Colab notebook + doc templates.
- **User** runs one Colab notebook → produces `checkpoint + results.zip`.
- **Agents** finalize docs with real numbers, regenerate PDF, verify, commit, push.

## Waves

- **W0 Truth & Cleanup** (local): quarantine contradictory Grad-CAMs → `attic/gradcam-broken/`;
  audit `src/` runs end-to-end; fix data-pipeline bugs; make code Colab-ready.
- **W1 Data** (Colab): real fetch (SDSS/Kaggle) + deterministic fallback; disjoint stratified
  80/10/10 split; MD5 leakage check.
- **W2 Train** (Colab GPU): EfficientNet-B3 fine-tune, progressive unfreezing, target >80%;
  saves `best_model.pth`, `training_summary.json`, curves.
- **W3 Eval + Grad-CAM** (Colab): honest metrics (standard + TTA), confusion matrix,
  per-class F1, Grad-CAM regenerated from the real model (images match metrics).
- **W4 Bonus** (local, 15% weight): polished Streamlit demo; template captioning from
  prediction; anomaly/OOD detection via softmax entropy. No black-box APIs.
- **W5 Docs sync** (local): rewrite REPORT/README/SUBMISSION with real numbers; regenerate
  REPORT.pdf; verify attribution + reproducibility + inference instructions.
- **W6 Verify & ship** (local): tests + lint; verification-before-completion gate;
  commit per wave; push to GitHub.

## Deliverable to user

`notebooks/Galaxy_X_Colab.ipynb` — open in Colab, Run all, download `results.zip`,
drop into repo. Only manual step.

## Honesty guardrail

Never claim a number we cannot reproduce from committed artifacts. Report states exactly
which dataset and hardware were used. This is the "un-wipeable lane" strategy: a grader who
clones, runs, and checks finds exactly what we claim.

## Success criteria

- [ ] `python src/train.py` (or Colab notebook) runs clean and produces a checkpoint.
- [ ] `src/evaluate.py` reproduces the reported metrics from that checkpoint.
- [ ] Grad-CAM images match the reported per-class performance.
- [ ] REPORT.md / REPORT.pdf numbers == `results/evaluation_results.json`.
- [ ] Tests pass, lint clean.
- [ ] Repo pushed, public, clone-and-run reproducible.
