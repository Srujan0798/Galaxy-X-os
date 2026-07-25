# PASTE TO AGENTS — READY BLOCKS

**Project:** `/Users/srujansai/Desktop/Galaxy-X-os`  
**Full plan:** `work/TOP_0.1_PERCENT_PLAN.md`  
**Score now:** ~74% · **Next:** Agent B  

Copy each block below into a separate agent chat. Do **not** run all agents on `app/app.py` at once.

---

## 0) GLOBAL PREFIX (paste first in every agent)

```text
You operate under ULTRA WIN + TOP 0.1% DOMINATION PROTOCOL.
Project root: /Users/srujansai/Desktop/Galaxy-X-os
Master plan: work/TOP_0.1_PERCENT_PLAN.md
Also read: HANDOFF.md, docs/SCOREBOARD.md, PROBLEM_STATEMENT.md.

LAWS:
1) Evidence or it didn't happen — write work/reports/PHASE-*-evidence.md with pasted commands + exit codes
2) Touch ONLY your write-set files
3) Sacred golden path: efficientnet_b3 checkpoint → Streamlit sample/upload → class + confidence + Grad-CAM + OOD
4) Never say done/100%/top 0.1%/submission-ready without freeze evidence
5) Fail loud; no mock model outputs; no invented metrics
6) Confirm before git push/deploy/destructive ops
7) Wire-or-attic unfinished modules; no dead code in src/
8) Optimize rubric: Performance 40%, Efficiency 15%, Explainability 15%, Bonus 15%, Docs 15%

Current honest score ~74%. Raise it with proof only.
```

---

## 1) AGENT B — Phase 2 Golden Path — ASSIGN FIRST (BLOCKING)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT B — Golden Path (CRITICAL).

Write-set ONLY:
- data/samples/**
- scripts/verify_golden_path.sh
- app/app.py (sample gallery + missing-checkpoint error only)
- README.md (Quick Start + how to get checkpoint only)

DO:
1) Create data/samples/{spiral_galaxy,elliptical_galaxy,nebula,star_cluster,planetary_object}/ with ≥1 PNG each
2) data/samples/README.md: demo-only disclaimer — NOT full test set — do NOT claim 93% from samples
3) scripts/verify_golden_path.sh:
   - no samples → exit 1
   - no checkpoints/best_model.pth → print GitHub Release v1.0 + Colab notebook path → exit 2
   - with ckpt → run one ModelManager.predict on a sample → print class, conf, ms → exit 0
4) Streamlit: sample buttons/select that run full predict + probability chart + Grad-CAM + template caption + OOD
5) Missing ckpt UI: large clear error + Release link + notebooks/Galaxy_X_Colab.ipynb
6) README 3-step judge path: install → get weights → streamlit run app/app.py → click sample

ACCEPTANCE (paste into work/reports/PHASE-2-evidence.md):
test $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ') -ge 5
bash scripts/verify_golden_path.sh
# also note manual: streamlit sample path works

FORBIDDEN: src/model.py, src/evaluate.py, inventing metrics, claiming samples prove 93%.
```

---

## 2) AGENT C — Phase 3 TTA (after B, parallel OK)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT C — TTA / Depth.

Write-set: src/tta.py, src/evaluate.py, tests/unit/test_tta.py, attic/**

Choose ONE:
- ATTIC (default if data/processed empty): move src/tta.py to attic/src-archive/tta.py; ensure no imports; keep existing evaluate 6× TTA; document honestly that TTA may not beat 93.17% (artifact shows 92.77%).
- WIRE: --tta-mode {off,simple,advanced} + unit tests; no accuracy claim without full re-eval JSON.

Evidence: work/reports/PHASE-3-evidence.md
```

---

## 3) AGENT D — Phase 4 Docs honesty (after B, parallel OK)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT D — Completeness & Honesty docs.

Write-set: SUBMISSION.md, BONUS_FEATURES.md, docs/presentation/**, docs/MODEL_CARD.md,
README.md STATUS blurb (do not destroy Agent B Quick Start), REPORT.md honesty-only (no new fake numbers),
docs/REPRODUCIBILITY.md (SHA/ckpt obtain clarity).

DO:
- Map every PROBLEM_STATEMENT deliverable + rubric cell → file → proof command
- Kill false Complete/100%/production-ready language
- Fix scale_odyssey/ → Galaxy-X-os in checklists
- Caption docs match app (template default; BLIP optional)
- 60-second judge path section

Evidence: work/reports/PHASE-4-evidence.md
```

---

## 4) AGENT E — Phase 5 All bonuses + orphans (after B, parallel OK)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT E — Brownies / Innovation TOP 0.1%.

Write-set: src/bonus.py, src/detection.py, src/gradcam_plus.py, src/onnx_export.py,
src/pseudo_label.py, optional minimal src/gradcam.py flag, related tests under tests/unit/

MANDATORY for each orphan file: WIRE with CLI+test OR move to attic/. Zero dead src modules.

PS bonuses all demoable:
1) Captioning (template OK if labeled)
2) Localization bbox (from Grad-CAM heatmap OK)
3) Anomaly/OOD (keep solid)
4) Web app safe (don't break samples)

Evidence: work/reports/PHASE-5-evidence.md
Run: pytest tests/unit/test_bonus.py tests/unit/test_localization.py tests/unit/test_ood.py -v
```

---

## 5) AGENT F — Phase 6 Architecture (after B or with B if no app touch)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT F — Architecture / ops.

Write-set: Makefile, HOW_TO_RUN.md, docs/SCOPE_GUARD.md, Dockerfile, docker-compose.yml,
src/utils.py (helpers), scripts/bench_latency.py, minimal empty-data guards in train/evaluate/gradcam mains.

DO:
- Remove dead Makefile/HOW_TO_RUN targets (generate_splits, train_head, missing files)
- Empty data/processed → exit 1 with message to run prepare_data or use Streamlit samples
- bench_latency.py → results/latency_bench.json when ckpt exists (prove <<5s)
- Docker notes: mount checkpoints + data + samples

Evidence: work/reports/PHASE-6-evidence.md
```

---

## 6) AGENT G — Phase 7 UI (ONLY after B merged)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT G — UI Domination.

Write-set: app/app.py ONLY (Phase 2 samples already exist).

Polish: thumbnail sample gallery, missing-ckpt/missing-samples errors, a11y confidence,
judge explainer for Grad-CAM, optional localization toggle if E ready, mobile-friendly layout.
No heavy new deps. Preserve golden path.
Evidence: work/reports/PHASE-7-evidence.md
```

---

## 7) AGENT H — Phase 8 Automated proof (after B samples exist)

```text
[PASTE GLOBAL PREFIX ABOVE]

You are AGENT H — Automated Proof.

Write-set: tests/e2e/**, tests/unit/test_checkpoint_smoke.py, tests/integration/**,
.github/workflows/**, scripts/ultra_win_gate.sh

DO:
- Checkpoint+sample predict smoke (pytest.skip if no ckpt)
- Replace import-only e2e as sole test with real predict assertions
- Mark network tests @pytest.mark.network; CI runs -m "not network"
- CI lint+unit; pip cache; python 3.10
- ultra_win_gate.sh: backbone default + unit + samples count + golden script

Evidence: work/reports/PHASE-8-evidence.md with full pytest paste
```

---

## 8) ORCH — Phase 9 Gate A freeze

```text
[PASTE GLOBAL PREFIX ABOVE]

You are FREEZE ORCHESTRATOR — Gate A (protocol 100%).

Read all work/reports/PHASE-*-evidence.md and work/TOP_0.1_PERCENT_PLAN.md Gate A checklist.
Run master command pack from the plan. Hostile 10-minute kick-out mindset.

ONLY if all Gate A boxes green:
- write work/reports/FREEZE.md with pasted evidence
- update docs/SCOREBOARD.md
- replace HANDOFF.md

Else: report REAL % and remaining reds. NEVER invent 100%.
```

---

## 9) GATE B — TOP 0.1% (only after Gate A)

### Agent B1 — Perf provenance
```text
[PASTE GLOBAL PREFIX ABOVE]
You are AGENT B1 — Irrefutable performance story.
Write-set: docs/REPRODUCIBILITY.md, results/ARTIFACT_HASHES.md, scripts/hash_artifacts.sh
SHA256 for evaluation_results.json + key plots; ckpt hash instructions; residual spiral/elliptical narrative with Grad-CAM refs.
Do not change metric numbers without re-eval. Evidence: work/reports/PHASE-B1-evidence.md
```

### Agent B2 — Latency
```text
[PASTE GLOBAL PREFIX ABOVE]
You are AGENT B2 — Efficiency proof.
Write-set: scripts/bench_latency.py, results/latency_bench.json, REPORT.md latency lines if needed.
Cold+warm batch-1 latency on this device; must be <<5s. Evidence: work/reports/PHASE-B2-evidence.md
```

### Agent B3 — Judge presentation
```text
[PASTE GLOBAL PREFIX ABOVE]
You are AGENT B3 — Presentation domination.
Write-set: docs/presentation/Judge_60s.md, Executive_Summary.md, Demo_Video_Script.md
If demo.mp4 stale vs UI, mark RE-RECORD REQUIRED. Evidence: work/reports/PHASE-B3-evidence.md
```

### Agent B4 — Moat moments
```text
[PASTE GLOBAL PREFIX ABOVE]
You are AGENT B4 — Competitive moat (2 judge-visible moments in <2 min).
e.g. localization overlay + OOD on noise sample. Prefer app-visible.
Write-set: carefully app/app.py OR src/bonus.py + tests. Evidence: work/reports/PHASE-B4-evidence.md
```

### Agent R — Hostile judge
```text
[PASTE GLOBAL PREFIX ABOVE]
You are a HOSTILE JUDGE. Try to KICK OUT this project in 10 minutes using only README.
Write work/reports/HOSTILE_JUDGE.md: P0/P1/P2 bugs, time-to-first-prediction, top 0.1% yes/no, required fixes.
Report only unless asked to fix.
```

### ORCH Gate B
```text
[PASTE GLOBAL PREFIX ABOVE]
You are TOP 0.1% FREEZE ORCHESTRATOR Gate B.
Verify Gate A + Gate B checklists from work/TOP_0.1_PERCENT_PLAN.md with fresh commands.
Only if both green: work/reports/TOP_TIER_FREEZE.md + SCOREBOARD "TOP 0.1% READY".
Else real gaps. NEVER invent top 0.1%.
```

---

## ORDER TO PASTE

```text
1. Agent B          (wait until merged)
2. Agents C,D,E,F,H (parallel)
3. Agent G
4. ORCH Gate A
5. B1, B2, B3, B4, R (parallel-ish)
6. ORCH Gate B
```

## RULE
If an agent finishes without `work/reports/PHASE-*-evidence.md` → **not done**. Reject and resend.
