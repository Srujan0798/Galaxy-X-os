# TOP 0.1% DOMINATION PLAN — Galaxy-X-os (SCALE × ODYSSEY)

**Status:** PASTE THIS ENTIRE FILE (or per-agent sections) TO YOUR AGENTS  
**Protocol:** ULTRA WIN + TOP 0.1% EXCELLENCE LAYER  
**Project root:** `/Users/srujansai/Desktop/Galaxy-X-os`  
**Repo:** https://github.com/Srujan0798/Galaxy-X-os  
**Date:** 2026-07-25  

| Field | Value |
|-------|--------|
| **Honest score NOW** | **~74%** (strong mid; NOT top 0.1% yet) |
| **Gate A — “100% freeze”** | Hostile judge cannot kick in 10 min; all FRs proven |
| **Gate B — “TOP 0.1%”** | Gate A + domination extras: moat, craft, proof automation, presentation, zero WIP debt, reproducibility theater that works |
| **Primary brief** | `PROBLEM_STATEMENT.md` |
| **Sacred golden path** | EfficientNet-B3 `best_model.pth` → Streamlit sample/upload → class + confidence + Grad-CAM + OOD (+ loc/caption) |
| **Never claim top 0.1%** | Without Gate B checklist + pasted evidence in `work/reports/TOP_TIER_FREEZE.md` |

---

# PART I — WHAT “TOP 0.1%” MEANS HERE

## Median submission (bottom 50%)
- Colab notebook only, no app, no Grad-CAM package, 80–85% on easy/synthetic data, messy README.

## Strong submission (~top 10–20%)
- Full pipeline + Streamlit + Grad-CAM + ~90%+ + report + video.

## Top 1%
- Real multi-source data with honest manifest, residual error analysis, reproducible Colab + local, solid tests, demo video that matches repo, no broken golden path.

## TOP 0.1% (YOUR BAR)
A hostile external reviewer in **10 minutes** experiences:

1. **Zero friction demo** — clone → one command path OR samples-in-repo + ckpt link → prediction + Grad-CAM works.  
2. **Science that holds** — metrics artifact + SHA256 + how to re-run; residual spiral/elliptical explained with Grad-CAM, not hidden.  
3. **Every advertised bonus works** — caption (labeled template or BLIP), OOD, localization, optional ONNX — no dead files in `src/`.  
4. **Proof automation** — CI green; predict smoke tests; verify script.  
5. **Craft** — intentional UI, professional report/PDF, demo video, model card, submission map.  
6. **Honesty as a weapon** — DATA_MANIFEST, no COMPLETE theater, TTA not oversold if it doesn’t help.  
7. **Judge-weighted optimization** — 40% accuracy story is unassailable; efficiency measured; explainability gorgeous; bonuses visible in app; docs judge-proof.

**You are not done when “it works on my machine.”**  
**You are done when a stranger cannot find a kick-out and prefers YOU over any peer.**

---

# PART II — INTENT LOCK (ALL AGENTS)

1. Optimize for official weights: **Perf 40 · Eff 15 · XAI 15 · Bonus 15 · Docs 15**.  
2. **Default backbone forever for submission:** `efficientnet_b3` matching `checkpoints/best_model.pth`.  
3. Multi-backbone / ensemble = experimental only until new weights + new metrics JSON exist.  
4. **Evidence or it didn’t happen** → `work/reports/PHASE-XX-evidence.md`.  
5. **Disjoint write-sets** — only your files.  
6. **No mock intelligence** sold as live model.  
7. **Fail loud.**  
8. Confirm before push / force-push / public deploy / deleting data.  
9. **Wire-or-attic** every WIP module before freeze.  
10. Replace `HANDOFF.md` after each phase (orchestrator). Never append forever.  
11. Prefer **depth that judges see** over paper architecture.  
12. **TOP 0.1% extras never break golden path.**

### Out of scope (kills focus)
- Production multi-tenant SaaS auth  
- Real-time telescope feeds  
- Fabricated metrics  
- Switching default model mid-freeze without full retrain proof  
- 10 unfinished “SOTA” modules  

---

# PART III — CURRENT TRUTH (DO NOT ARGUE)

### REAL
| Item | Status |
|------|--------|
| EfficientNet-B3 pipeline train/eval/gradcam/inference/bonus/app | EXISTS |
| Local ckpt ~141MB efficientnet_b3 | EXISTS (machine) / NOT in git |
| results JSON 93.17% / TTA 92.77% | ARTIFACT exists |
| Grad-CAM 15 samples + summary | COMMITTED |
| Demo video ~76s | EXISTS |
| Unit tests ~46+ | PASS |
| Live MPS inference | PROVEN earlier session |

### BROKEN / WEAK (why you are NOT top 0.1% yet)
| Item | Status |
|------|--------|
| `data/processed` images | **0** |
| Demo samples pack | **MISSING** |
| Stranger 10-min path | **FAILS** without tribal knowledge |
| E2E | **Import theater** |
| WIP orphans | `tta.py`, `detection.py`, `gradcam_plus.py`, `onnx_export.py`, `pseudo_label.py` |
| Docs greenwash | COMPLETE language |
| Hosted demo | None (OK if local path is perfect) |
| Latency bench table | Soft / not filed |
| SHA256 freeze pack for ckpt+results | Incomplete |

---

# PART IV — SCORE LADDER

| Gate | Name | Blended | Meaning |
|------|------|---------|---------|
| Now | Baseline | **74%** | Real model, weak stranger path |
| P2 | Golden path | **84–86%** | Demo in 10 min |
| P3–6 | Depth+honesty+arch | **92–95%** | Clean repo, honest docs |
| P7–8 | UI+proof | **97–99%** | Hostile-hard |
| **Gate A** | Freeze 100% | **100% protocol** | Checklist green |
| **Gate B** | TOP 0.1% | **Beyond freeze** | Domination layer (Part VIII) all green |

---

# PART V — OFFICIAL RUBRIC → TOP 0.1% BAR

| ID | Wt | “Pass” | TOP 0.1% bar | Primary phases |
|----|-----|--------|--------------|----------------|
| R1 Perf | 40% | >80%, report metrics | Artifact + SHA256 + residual analysis + optional mini re-eval OR crystal-clear Colab re-run; confusion matrix story; no data leakage claim without MD5 proof script | 2,4,8,B1 |
| R2 Eff | 15% | <5s | `results/latency_bench.json` CPU+MPS/CUDA cold/warm; param count; optional ONNX speed note | 6,8,B2 |
| R3 XAI | 15% | Grad-CAM | App gallery Grad-CAM quality; 15 panels; summary grid; optional GradCAM++ if better; fail-loud | 2,5,7 |
| R4 Bonus | 15% | Some bonus | **All 4 PS bonuses** working: caption, localization, anomaly, web app — each demoable | 2,5,7 |
| R5 Docs | 15% | README | REPORT PDF + video + model card + SUBMISSION map + SCOREBOARD + 60s judge path | 4,7,B3 |

---

# PART VI — DEPENDENCY DAG

```text
P0(done) → P1(done) → P2(B) ──┬→ P3(C) ──┐
                               ├→ P4(D) ──┤
                               ├→ P5(E) ──┼→ P7(G) → P8(H) → P9 GateA → P10 GateB TOP0.1%
                               ├→ P6(F) ──┤
                               └→ P8 early tests ok after samples exist
```

**Parallel after P2:** C ∥ D ∥ E ∥ F ∥ H  
**Serial on app.py:** B → G (E checkboxes only with G or after G)  
**Freeze:** P9 then P10  

---

# PART VII — PHASES 0–9 (GATE A = PROTOCOL 100%)

---

## P0 — Truth reset — ORCH — DONE
Artifacts: `work/reports/BRUTAL_AUDIT.md`, `P0-baseline.md`, `docs/SCOREBOARD.md`, `HANDOFF.md`.

---

## P1 — Integrity — Agent A — DONE
- Default backbone `efficientnet_b3`
- `tests/unit/test_model.py` contract
- Ckpt still loads

**Re-verify anytime:**
```bash
python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'"
python3 -m pytest tests/unit/test_model.py -v
```

---

## P2 — Golden path — Agent B — **CRITICAL / ASSIGN FIRST**

### Goal
Stranger demos in &lt;10 minutes without asking you anything.

### Write-set ONLY
```
data/samples/**
scripts/verify_golden_path.sh
app/app.py          # samples + missing-ckpt only
README.md           # Quick Start / ckpt section only
```

### Must implement
1. **Samples:** `data/samples/{spiral_galaxy,elliptical_galaxy,nebula,star_cluster,planetary_object}/` ≥1 PNG each (prefer 2).  
2. `data/samples/README.md` — demo only; NOT the test set; do not claim 93% on these.  
3. **`scripts/verify_golden_path.sh`**
   - No samples → exit 1  
   - No ckpt → print Release URL + Colab path → exit 2  
   - Ckpt OK → one predict → print class, conf, ms → exit 0  
4. **Streamlit:** sample buttons/select → full path: predict + probs + Grad-CAM + template caption + OOD.  
5. **Missing ckpt UX:** big error + links: GitHub Release v1.0 + `notebooks/Galaxy_X_Colab.ipynb`.  
6. README 3-step Quick Start for judges.

### Acceptance
```bash
test $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ') -ge 5
test -f data/samples/README.md
bash scripts/verify_golden_path.sh   # 0 with ckpt
# Manual: streamlit run app/app.py → sample → Grad-CAM visible
```

### Evidence
`work/reports/PHASE-2-evidence.md`

### PASTE PROMPT — AGENT B
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
Read: work/TOP_0.1_PERCENT_PLAN.md (P2), work/ETERNAL_FINAL_PLAN.md, HANDOFF.md, PROBLEM_STATEMENT.md
You are AGENT B — Golden Path. CRITICAL PATH.

Write-set ONLY:
- data/samples/**
- scripts/verify_golden_path.sh
- app/app.py (sample gallery + missing-ckpt error only)
- README.md (Quick Start + checkpoint obtain only)

DO:
1) ≥1 real-looking PNG per class under data/samples/{spiral_galaxy,elliptical_galaxy,nebula,star_cluster,planetary_object}/
2) data/samples/README.md disclaimer (demo only; not full test set; not for 93% claims)
3) scripts/verify_golden_path.sh as specified in the plan
4) Streamlit sample picker runs predict + Grad-CAM + caption + OOD
5) Missing ckpt: clear error + Release v1.0 + Colab notebook links

ACCEPTANCE (paste all output into work/reports/PHASE-2-evidence.md):
test $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ') -ge 5
bash scripts/verify_golden_path.sh

FORBIDDEN: invent metrics; touch src/model.py; touch src/evaluate.py; claim samples = 93% accuracy.
Sacred path: efficientnet_b3 ckpt → sample → class+conf+Grad-CAM+OOD.
Evidence or it didn't happen. Do not claim 100% or top 0.1%.
```

---

## P3 — TTA / depth resolve — Agent C

### Write-set
```
src/tta.py
src/evaluate.py
tests/unit/test_tta.py
attic/**   # if moving
```

### Decision (pick one, document)
- **ATTIC (default if no full test set):** move `src/tta.py` → `attic/src-archive/tta.py`; keep evaluate’s existing 6× TTA; docs say TTA does not always lift (honest 92.77 vs 93.17).  
- **WIRE:** `--tta-mode {off,simple,advanced}` + tests; no leaderboard claim without full re-eval.

### Acceptance
```bash
# ATTIC:
test ! -f src/tta.py
rg -n "from tta import|import tta" src/ app/ tests/ && exit 1 || echo clean
# OR WIRE:
python src/evaluate.py --help | grep -i tta
pytest tests/unit/test_tta.py -v
```

### PASTE PROMPT — AGENT C
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
Read work/TOP_0.1_PERCENT_PLAN.md P3.
You are AGENT C — TTA/Depth.

Write-set: src/tta.py, src/evaluate.py, tests/unit/test_tta.py, attic/**

Prefer PATH ATTIC if data/processed empty. Wire only if you can measure.
Do not fake accuracy lift. Keep evaluate 6× TTA story honest (TTA may not beat standard).
Evidence: work/reports/PHASE-3-evidence.md with path chosen + commands.
No 100% claim.
```

---

## P4 — Docs honesty & judge map — Agent D

### Write-set
```
SUBMISSION.md
BONUS_FEATURES.md
docs/presentation/**
docs/MODEL_CARD.md
README.md          # STATUS blurb + links only (coordinate: B owns Quick Start; don't delete it)
REPORT.md          # honesty fixes only — NEVER invent numbers
docs/REPRODUCIBILITY.md  # SHA256 section if missing
```

### Must implement
1. Rubric table → file → proof command (no false Complete).  
2. Fix `scale_odyssey/` → `Galaxy-X-os` in checklists.  
3. Caption honesty: template default in app; BLIP optional.  
4. STATUS block on README linking SCOREBOARD.  
5. Model card accurate vs `evaluation_results.json` or delete.  
6. Judge 60-second path documented.  
7. List exact Release asset name for ckpt + expected size ~141MB.

### PASTE PROMPT — AGENT D
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
You are AGENT D — Completeness & Honesty.
Read PROBLEM_STATEMENT.md, docs/SCOREBOARD.md, work/TOP_0.1_PERCENT_PLAN.md P4.

Write-set: SUBMISSION.md, BONUS_FEATURES.md, docs/presentation/**, docs/MODEL_CARD.md,
README STATUS section, REPORT.md honesty-only, docs/REPRODUCIBILITY.md SHA section.

Kill greenwash (Complete/100%/production-ready without evidence).
Map every deliverable → path → command.
Align bonuses with real app behavior.
Evidence: work/reports/PHASE-4-evidence.md
Do not invent metrics. Do not claim top 0.1%.
```

---

## P5 — All bonuses + WIP orphans — Agent E

### Write-set
```
src/bonus.py
src/detection.py
src/gradcam_plus.py
src/onnx_export.py
src/pseudo_label.py
src/gradcam.py          # only if integrating GradCAM++ flag
tests/unit/test_bonus.py
tests/unit/test_localization.py
tests/unit/test_ood.py
tests/unit/test_detection.py      # create if wiring
tests/unit/test_onnx_export.py    # create if wiring
```

### TOP 0.1% bonus bar (PS lists 4 bonuses — all must work)

| Bonus | Requirement |
|-------|-------------|
| Captioning | Works in app (template OK if labeled); CLI path in bonus.py |
| Localization | BBox overlay from CAM heatmap; app checkbox OR CLI; tested |
| Anomaly / OOD | Already present — keep solid; document thresholds |
| Web app | Streamlit — don’t break B’s samples |

### Orphans
Each of `detection.py`, `gradcam_plus.py`, `onnx_export.py`, `pseudo_label.py`:  
**WIRE with CLI `--help` + test** OR **move to attic/**.

### PASTE PROMPT — AGENT E
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
You are AGENT E — Brownies / Innovation.

Write-set: src/bonus.py, src/detection.py, src/gradcam_plus.py, src/onnx_export.py,
src/pseudo_label.py, optional src/gradcam.py flag, related unit tests.

MANDATORY: wire-or-attic EVERY orphan. Zero dead modules in src/ at end.
Deliver all 4 PS bonuses in demoable form (caption, localization, anomaly, app-safe).
Prefer localization overlay from Grad-CAM heatmap.
ONNX export = optional efficiency brownie with working script.
No mock predictions. Don't break golden path.
Evidence: work/reports/PHASE-5-evidence.md
pytest tests/unit/test_bonus.py tests/unit/test_localization.py tests/unit/test_ood.py -v
```

---

## P6 — Architecture / ops polish — Agent F

### Write-set
```
Makefile
HOW_TO_RUN.md
docs/SCOPE_GUARD.md
Dockerfile
docker-compose.yml     # comments/volumes honesty
src/utils.py           # fail-loud helpers, latency bench helper
src/train.py           # only empty-data guard at start of main (minimal)
src/evaluate.py        # only if C finished; else guard via utils called from evaluate
src/gradcam.py         # empty-data guard only
scripts/bench_latency.py
```

### Must implement
1. Makefile: install, prepare/split, train, evaluate, gradcam, app, test, lint, verify, bench, clean — **no dead targets**.  
2. HOW_TO_RUN matches real `src/*` files.  
3. Empty `data/processed` → exit 1 with actionable message.  
4. `scripts/bench_latency.py` → `results/latency_bench.json` if ckpt exists.  
5. SCOPE_GUARD: localization optional bonus in scope.  
6. Docker: document volume mounts for ckpt/data/samples.

### PASTE PROMPT — AGENT F
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
You are AGENT F — Architecture.

Write-set: Makefile, HOW_TO_RUN.md, docs/SCOPE_GUARD.md, Dockerfile, docker-compose.yml,
src/utils.py helpers, minimal empty-data guards, scripts/bench_latency.py

Remove dead targets (generate_splits, train_head, missing modules).
Fail loud on empty dataset. Latency bench JSON when ckpt present.
Evidence: work/reports/PHASE-6-evidence.md
```

---

## P7 — UI domination — Agent G

### Write-set
```
app/app.py   # ONLY after P2 merged
```

### TOP 0.1% UI bar
1. Class sample gallery with thumbnails.  
2. Missing ckpt / missing samples: beautiful fail states.  
3. Confidence not color-only (text + %).  
4. Grad-CAM caption for judges.  
5. Optional: localization toggle, “About metrics 93.17% artifact” expander with honesty.  
6. No heavy new deps. Mobile-ok layout.

### PASTE PROMPT — AGENT G
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
You are AGENT G — UI Domination. ONLY after Phase 2 merged.

Write-set: app/app.py ONLY.

Polish sample gallery, errors, a11y, judge-facing explainers. Keep all ML paths working.
No new heavy dependencies. No metric invention.
Evidence: work/reports/PHASE-7-evidence.md with manual streamlit checklist.
```

---

## P8 — Automated proof — Agent H

### Write-set
```
tests/e2e/**
tests/unit/test_checkpoint_smoke.py
tests/integration/**
.github/workflows/**
scripts/ultra_win_gate.sh
pytest.ini or pyproject.toml markers only if needed
```

### Must implement
1. Checkpoint smoke: skip if no ckpt; else predict sample; assert valid class; time &lt; 5000ms.  
2. E2E not import-only: real predict path.  
3. `@pytest.mark.network` for prepare_data integration; CI excludes network.  
4. CI: lint + unit; cache pip; python 3.10.  
5. `scripts/ultra_win_gate.sh` runs defaults + unit + samples + golden.

### PASTE PROMPT — AGENT H
```text
ULTRA WIN + TOP 0.1%. Project: /Users/srujansai/Desktop/Galaxy-X-os
You are AGENT H — Automated Proof.

Write-set: tests/e2e/**, tests/unit/test_checkpoint_smoke.py, tests/integration/**,
.github/workflows/**, scripts/ultra_win_gate.sh

Kill e2e theater. Real predict smoke with skip-if-no-ckpt.
CI: not network by default. Gate script for orchestrator.
Evidence: work/reports/PHASE-8-evidence.md with full pytest paste.
```

---

## P9 — Gate A freeze (protocol 100%) — ORCH

### Freeze checklist Gate A
```text
[ ] R1–R5 GREEN or honest YELLOW with evidence in SCOREBOARD
[ ] P2 verify_golden_path.sh exit 0 with ckpt; ≥5 samples
[ ] Default backbone efficientnet_b3
[ ] No orphan src modules (wired or attic)
[ ] weights_only loads; no secrets
[ ] Browser golden path proven in evidence
[ ] No mock sold as model
[ ] pytest -m "not network" green
[ ] CI green or documented
[ ] Docs honest
[ ] Hostile 10-min kick-out fails
```

### PASTE PROMPT — ORCH FREEZE A
```text
You are FREEZE ORCHESTRATOR Gate A. Project: /Users/srujansai/Desktop/Galaxy-X-os
Read all work/reports/PHASE-*-evidence.md, docs/SCOREBOARD.md, work/TOP_0.1_PERCENT_PLAN.md P9.
Run freeze command pack. Hostile mindset.
If ALL Gate A boxes green: write work/reports/FREEZE.md, set SCOREBOARD protocol 100%, rewrite HANDOFF.
Else: real % + reds. NEVER invent 100%.
Then proceed to Gate B only if Gate A green.
```

---

# PART VIII — GATE B: TOP 0.1% DOMINATION LAYER (P10)

Do **after** Gate A. These separate you from “good hackathon project” into **unforgettable**.

## B1 — Irrefutable performance story (Agent D + H + optional train agent)

### Write-set
```
docs/REPRODUCIBILITY.md
results/ARTIFACT_HASHES.md          # create
scripts/hash_artifacts.sh           # create
notebooks/Galaxy_X_Colab.ipynb      # only fix cells if broken
```

### Requirements
1. SHA256 of `results/evaluation_results.json`, key PNGs, and instructions for ckpt hash after download.  
2. One paragraph: train hardware, seed 42, epoch, best_val, test N=249.  
3. Residual analysis: spiral↔elliptical with Grad-CAM references.  
4. Optional: if GPU available, re-run evaluate on rebuilt data — **only if honest**.

### PASTE PROMPT — AGENT B1 (Perf story)
```text
TOP 0.1% Gate B1. Project: /Users/srujansai/Desktop/Galaxy-X-os
Write-set: docs/REPRODUCIBILITY.md, results/ARTIFACT_HASHES.md, scripts/hash_artifacts.sh
Create irrefutable metrics provenance (hashes, train story, residual error narrative).
Do not change metric numbers without re-running evaluate.
Evidence: work/reports/PHASE-B1-evidence.md
```

---

## B2 — Efficiency domination (Agent F)

1. `results/latency_bench.json`: batch1 cold/warm on current device; document device name.  
2. Optional ONNX: export + timed if Agent E wired onnx.  
3. Param count + model size MB in REPORT one-liner.

### PASTE PROMPT — AGENT B2
```text
TOP 0.1% Gate B2 Efficiency. Write-set: scripts/bench_latency.py, results/latency_bench.json, short REPORT.md latency section update if needed.
Prove <<5s with numbers on this machine. Evidence: work/reports/PHASE-B2-evidence.md
```

---

## B3 — Presentation domination (Agent D + optional media)

1. Demo video matches current UI (re-record if UI changed heavily).  
2. Executive summary ≤1 page: problem → approach → 93.17% → Grad-CAM → demo path.  
3. Judge script: 60s talk track in `docs/presentation/Judge_60s.md`.  
4. SUBMISSION form text ready to paste.

### PASTE PROMPT — AGENT B3
```text
TOP 0.1% Gate B3 Presentation. Write-set: docs/presentation/Judge_60s.md, Executive_Summary.md, Demo_Video_Script.md
If demo.mp4 is stale vs app, note RE-RECORD REQUIRED in evidence (do not fake).
Evidence: work/reports/PHASE-B3-evidence.md
```

---

## B4 — Competitive moat features (Agent E residual)

Pick **2** that judges see in &lt;2 minutes:
1. Localization overlay on Grad-CAM (must work).  
2. OOD “reject” story with synthetic noise image sample.  
3. Side-by-side true vs pred CAM when wrong (if labels available).  
4. One-click “benchmark latency” in sidebar (optional).

No new unfinished modules.

### PASTE PROMPT — AGENT B4
```text
TOP 0.1% Gate B4 Moat. Extend only working bonus/app paths. 2 judge-visible moat moments.
Write-set: app/app.py (if G done, carefully) OR src/bonus.py + tests.
Evidence: work/reports/PHASE-B4-evidence.md
```

---

## B5 — Hostile red team (Agent R / ORCH)

### PASTE PROMPT — AGENT R (Hostile)
```text
You are a HOSTILE JUDGE trying to KICK OUT Galaxy-X-os in 10 minutes.
Project: /Users/srujansai/Desktop/Galaxy-X-os
Read PROBLEM_STATEMENT.md. Follow only README Quick Start.
Try: missing ckpt, empty data, evaluate, streamlit, wrong file upload, read SUBMISSION claims vs reality.
Write work/reports/HOSTILE_JUDGE.md with:
- Kick-out bugs found (P0/P1/P2)
- Time-to-first-prediction
- Would you score this top 0.1%? yes/no + why
- Required fixes before Gate B
Do not implement fixes unless asked; report only.
```

---

## B6 — Final TOP 0.1% freeze — ORCH

### Gate B checklist (ALL required beyond Gate A)
```text
[ ] Gate A FREEZE.md green
[ ] ARTIFACT_HASHES.md exists
[ ] latency_bench.json exists and <5000ms
[ ] All 4 PS bonuses demoable
[ ] Zero untracked orphan src modules without purpose
[ ] HOSTILE_JUDGE.md: no P0 kick-outs; time-to-predict <10 min
[ ] Judge_60s.md exists
[ ] Sample gallery looks intentional
[ ] You would bet money a top reviewer ranks this in top 0.1% of the field
```

### PASTE PROMPT — ORCH GATE B
```text
TOP 0.1% FREEZE ORCHESTRATOR. Project: /Users/srujansai/Desktop/Galaxy-X-os
Verify Gate A + Gate B checklists with fresh commands.
Only if both green: write work/reports/TOP_TIER_FREEZE.md, update SCOREBOARD "TOP 0.1% READY", HANDOFF freeze.
Else: real % and remaining reds. NEVER invent top 0.1%.
```

---

# PART IX — FILE OWNERSHIP (NO COLLISIONS)

```text
src/model.py                 → A (frozen)
src/tta.py, evaluate.py      → C (F only empty guard via utils)
src/bonus + orphans          → E
src/inference.py             → freeze; H tests only
src/gradcam.py               → E if ++ flag else freeze
src/train.py                 → F empty guard only / ORCH
src/prepare_data.py          → freeze unless data bugs
src/dataset.py, utils.py     → F utils helpers; else freeze
app/app.py                   → B then G then B4 carefully
data/samples/**              → B
scripts/verify_golden_path   → B
scripts/bench_latency        → F/B2
scripts/hash_artifacts       → B1
scripts/ultra_win_gate       → H
tests/unit/test_model        → A
tests/e2e, smoke, workflows  → H
SUBMISSION, presentation     → D/B3
Makefile, HOW_TO_RUN         → F
docs/SCOREBOARD, HANDOFF     → ORCH
work/**                      → ORCH + each agent own evidence files
```

---

# PART X — MASTER COMMAND PACK (ORCH / ANY VERIFY)

```bash
cd /Users/srujansai/Desktop/Galaxy-X-os

# Integrity
python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'; print('BACKBONE_OK')"

# Samples
echo "samples:" $(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)

# Golden
bash scripts/verify_golden_path.sh || echo "golden exit $?"

# Tests
python3 -m pytest tests/ -v -m "not network"

# Orphans check (should only be intentional modules)
ls src/*.py

# Artifacts
ls -la results/evaluation_results.json results/gradcam/_summary_grid.png docs/presentation/demo.mp4 checkpoints/best_model.pth 2>&1 | head -20

# Latency if script exists
test -f scripts/bench_latency.py && python3 scripts/bench_latency.py || true
```

---

# PART XI — KICK-OUT BURNDOWN → OWNER

| # | Kick-out | Owner | Phase |
|---|----------|-------|-------|
| 1 | Empty data / no samples | B | P2 |
| 2 | Ckpt missing unclear | B, D | P2, P4 |
| 3 | Default backbone wrong | A | P1 done |
| 4 | Dead tta.py | C | P3 |
| 5 | E2E theater | H | P8 |
| 6 | No predict smoke | H | P8 |
| 7 | Orphan modules | E | P5 |
| 8 | COMPLETE greenwash | D | P4 |
| 9 | Dead HOW_TO_RUN | F | P6 |
| 10 | Empty evaluate confusing | F | P6 |
| 11 | TTA oversold | C, D | P3, P4 |
| 12 | Nebula provenance | D | P4 |
| 13 | Weak UI | G | P7 |
| 14 | No latency proof | F | P6/B2 |
| 15 | Stale demo video | B3 | Gate B |
| 16 | Hostile 10-min fail | R | B5 |
| 17 | Missing hashes | B1 | Gate B |
| 18 | Bonus not in app | E, G | P5, P7 |

---

# PART XII — SPRINT SCHEDULE (ASSIGN THIS WAY)

### Sprint 0 (done)
ORCH P0, A P1

### Sprint 1 — BLOCKING (1 agent)
| Agent | Phase |
|-------|-------|
| **B** | **P2 Golden path** |

### Sprint 2 — PARALLEL (after B merges)
| Agent | Phase |
|-------|-------|
| C | P3 |
| D | P4 |
| E | P5 |
| F | P6 |
| H | P8 |

### Sprint 3
| Agent | Phase |
|-------|-------|
| G | P7 UI |
| ORCH | P9 Gate A |

### Sprint 4 — TOP 0.1%
| Agent | Phase |
|-------|-------|
| B1 | Perf story hashes |
| B2 | Latency |
| B3 | Judge 60s / video honesty |
| B4 | Moat moments |
| R | Hostile judge report |
| ORCH | Gate B TOP_TIER_FREEZE |

---

# PART XIII — GLOBAL PREFIX (PASTE ABOVE EVERY AGENT)

```text
You operate under ULTRA WIN + TOP 0.1% DOMINATION PROTOCOL.
Project root: /Users/srujansai/Desktop/Galaxy-X-os
Master plan: work/TOP_0.1_PERCENT_PLAN.md
Also read: HANDOFF.md, docs/SCOREBOARD.md, PROBLEM_STATEMENT.md, your phase section.

LAWS:
1) Evidence or it didn't happen — paste commands+exit codes into work/reports/PHASE-*-evidence.md
2) Disjoint write-set only — never touch other agents' files
3) Sacred golden path: efficientnet_b3 ckpt → sample/upload → class+conf+Grad-CAM+OOD
4) No "done/100%/top 0.1%/submission-ready" without Gate A/B freeze evidence
5) Fail loud; no silent empty success
6) No mock intelligence as live model
7) Confirm before git push / deploy / destructive ops
8) Prefer wire-or-attic over half-finished modules
9) Optimize judge rubric: Perf40 Eff15 XAI15 Bonus15 Docs15
10) When finished: evidence file + residual risks + files changed list

Current honest score ~74%. Your job is to raise it with proof, not hype.
```

---

# PART XIV — ONE-PAGE DISPATCHER (PRINT / PIN)

```text
PROJECT:  Galaxy-X-os
PLAN:     work/TOP_0.1_PERCENT_PLAN.md
DISPATCH: work/AGENTS_DISPATCH.md (short) + THIS FILE (full)
NOW:      74%
NEXT:     AGENT B → P2
THEN:     C∥D∥E∥F∥H → G → ORCH GateA → B1∥B2∥B3∥B4 → R → ORCH GateB
GOLDEN:   efficientnet_b3 → streamlit sample → Grad-CAM
TOP 0.1%: Gate A + Gate B checklists both green + HOSTILE_JUDGE no P0
```

| Paste section | Agent |
|---------------|-------|
| PART XIII + P2 prompt | B |
| PART XIII + P3 | C |
| PART XIII + P4 | D |
| PART XIII + P5 | E |
| PART XIII + P6 | F |
| PART XIII + P7 | G |
| PART XIII + P8 | H |
| P9 | ORCH A |
| B1–B4, R, B6 | Gate B team |

---

# PART XV — DEFINITION OF DONE (FINAL)

### Gate A “100%” (protocol)
Hostile 10-minute kick-out fails; freeze checklist green; FREEZE.md exists.

### Gate B “TOP 0.1%”
Gate A + hashes + latency file + all 4 bonuses demoable + Judge_60s + HOSTILE_JUDGE clean + intentional UI + zero src orphans + you would bet on ranking.

**If any box open: report REAL %. Never invent top 0.1%.**

---

# PART XVI — SUBMISSION DAY CHECKLIST (HUMAN)

```text
[ ] Repo public: https://github.com/Srujan0798/Galaxy-X-os
[ ] Release v1.0 has best_model.pth OR Colab path works cold
[ ] REPORT.pdf upload ready
[ ] demo.mp4 plays
[ ] bash scripts/verify_golden_path.sh on clean machine notes
[ ] SUBMISSION.md form fields filled
[ ] No secrets in git
[ ] SCOREBOARD / HANDOFF say freeze honestly
```

---

**END OF TOP 0.1% PLAN**  
Assign **Agent B** first. Everything else is theater until golden path is stranger-proof.
