# Galaxy-X-os — Orchestrator Kernel (CLAUDE.md)

> Auto-loaded by Claude Code. Interchangeable with KIMI.md.

## Project Identity

- **Name:** Galaxy-X-os (SCALE x ODYSSEY)
- **Goal:** Classify raw astronomical images into 5 celestial categories using deep learning
- **Domain:** Computer Vision / ML / Astronomical Imaging
- **Tech Stack:** Python, PyTorch, EfficientNet-B3, Albumentations, Streamlit, Grad-CAM
- **Tier:** T1 (Standard)
- **MVP:** Ingest one telescope image, output one of 5 class predictions with confidence

## Quick Commands

| Command | Purpose |
|---------|---------|
| `make install` | Install dependencies |
| `make train` | Run training pipeline |
| `make evaluate` | Run evaluation + TTA |
| `make gradcam` | Generate Grad-CAM visualizations |
| `make app` | Launch Streamlit demo |
| `make test` | Run all tests |
| `make lint` | Run ruff + mypy |

## Architecture Overview

```
Raw Images → Preprocess → AstroDataset → EfficientNet-B3 → 5-Class Output
                ↓              ↓              ↓
          Augmentations   DataLoaders    Grad-CAM / Streamlit
```

## Directory Map

| Path | Purpose | Owner |
|------|---------|-------|
| `src/` | Source code | Workers |
| `app/` | Streamlit demo | Workers |
| `notebooks/` | EDA + training + evaluation | Workers |
| `config/` | YAML configs | Orchestrator |
| `data/` | Raw + processed datasets | Workers |
| `checkpoints/` | Model weights | Workers |
| `results/` | Logs, Grad-CAM, plots | Workers |
| `tests/` | All test suites | Workers |
| `orchestrator/` | Tier-1 governance | Orchestrator |
| `work/` | Task bridge | Orchestrator writes, Workers read |
| `plan/` | PRD, ARCH, EXECUTION | Orchestrator |
| `docs/` | ADRs, runbooks, conventions | Orchestrator |
| `evals/` | Eval tasks + graders | Orchestrator |

## Risk Tier

This project is **T1 — Standard**.
- No customer PII
- No compliance requirements
- No production SLA
- Internal tool / hackathon submission

## Blast Radius

| Action | Radius | Gate |
|--------|--------|------|
| Read files, run tests | r0 | Auto |
| Write to src/, modify tests | r1 | Log + proceed |
| Add deps, change CI | r2 | Await approval |
| rm -rf, force-push | r3 | Block |

## Session Recovery

If this session crashes:
1. Reopen Claude Code in this directory
2. This file auto-loads
3. Read `HANDOFF.md` for current wave state
4. Read latest `orchestrator/memory/session/*.events.jsonl`
5. Resume from last event

## Core Rules

1. Orchestrator plans and reviews. Workers execute.
2. Handoff is `work/<wave>/<task>.md` → `work/reports/<wave>/<task>.report.md`
3. Never delete — move to `attic/`, `docs/historical/`, `prompts/archive/`
4. Every wave ships before next wave begins
5. Run acceptance before approving any merge
