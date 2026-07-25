# REASSIGN GUIDE — After hostile re-audit (84%, not 96%)

**Date:** 2026-07-25  
**Truth file:** `work/reports/HOSTILE_REAUDIT.md`  
**Do not re-run Phases 0–8 from zero.** Agents already delivered real local gains.  
**Goal:** Ship + close integrity holes → Gate A real → then Gate B.

---

## One-line truth for every agent

```text
Hostile re-audit says ~84% NOT 96%. FREEZE is incomplete. Your job is ONLY the assigned gap below.
Read work/reports/HOSTILE_REAUDIT.md first. Evidence or it didn't happen. No 100% claims.
Project: /Users/srujansai/Desktop/Galaxy-X-os
```

---

## Priority order (reassign in this sequence)

```text
P0  Agent SHIP     → coherent commit (or PR) of working tree
P0  Agent FIX-EXIT → evaluate/train/gradcam exit 1 on empty data
P1  Agent BROWSER  → Streamlit sample path proof
P1  Agent CI       → green Actions on pushed branch
P1  Agent TRUTH    → rewrite SCOREBOARD/HANDOFF/FREEZE to 84% honesty
P2  Agent DATA     → optional: tiny processed subset OR stronger artifact labels
P2  Agent GATE-A   → re-freeze only when P0–P1 green
P3  Gate B agents  → only after real Gate A
```

**Parallel safe:** FIX-EXIT ∥ TRUTH ∥ BROWSER (after SHIP if they need remote).  
**SHIP first** if graders use GitHub.

---

## AGENT SHIP (P0 — most important)

**Why:** ~45 uncommitted paths. Graders cloning GitHub **do not see** samples, verify script, e2e smoke, app gallery.

**Write-set:** git commit only (all golden-path files) — **ask user before push**

```text
[ONE-LINE TRUTH]

You are AGENT SHIP.
Read work/reports/HOSTILE_REAUDIT.md.

Goal: Create ONE coherent commit (or stacked commits) that ships the LOCAL golden path WITHOUT bringing back dead ensemble WIP.

INCLUDE:
- data/samples/** (all class images + noise + README)
- scripts/verify_golden_path.sh, ultra_win_gate.sh, bench_latency.py, hash_artifacts.sh
- app/app.py
- tests/e2e/test_app.py, tests/unit/test_checkpoint_smoke.py, tests/unit/test_onnx_export.py
- src/attic/**, attic moves for tta
- src/utils.py fail helpers, evaluate/train/gradcam empty guards (after FIX-EXIT)
- docs that are honest (SCOREBOARD/HANDOFF after TRUTH agent, or coordinate)
- results/latency_bench.json, results/ARTIFACT_HASHES.md if not huge

EXCLUDE / DO NOT:
- force-push
- claim 100%
- re-add unwired ensemble training as default
- commit checkpoints/best_model.pth (too large)

BEFORE COMMIT: run
  bash scripts/verify_golden_path.sh
  python3 -m pytest tests/ -m "not network" -q

Ask user before git push. Paste git status + commit hash into work/reports/PHASE-SHIP-evidence.md
```

---

## AGENT FIX-EXIT (P0 — integrity)

**Why:** Re-audit proved `python src/evaluate.py` with empty data **exits 0**. Hostile fail.

**Write-set:** `src/evaluate.py`, `src/train.py`, `src/gradcam.py`, `src/utils.py` (helper only)

```text
[ONE-LINE TRUTH]

You are AGENT FIX-EXIT.
Bug: empty data/processed still yields exit code 0 from evaluate.py (and possibly train/gradcam).

Requirements:
1) If no images in required splits, print clear message AND sys.exit(1)
2) Message must mention: python src/prepare_data.py OR streamlit samples demo
3) Unit/smoke: optional small test that empty dir → SystemExit/exit 1 if easy
4) Do not change metrics or model architecture

Acceptance:
  python3 src/evaluate.py ; test $? -eq 1
  # with samples still:
  bash scripts/verify_golden_path.sh  # exit 0

Evidence: work/reports/PHASE-FIX-EXIT-evidence.md with paste showing exit 1
```

---

## AGENT BROWSER (P1)

**Why:** FREEZE correctly left browser unproven. Code has samples; need proof.

**Write-set:** `work/reports/PHASE-BROWSER-evidence.md` only (or playwright script under `scripts/` if needed)

```text
[ONE-LINE TRUTH]

You are AGENT BROWSER.
Prove Streamlit golden path WITHOUT claiming 100%.

Steps:
1) streamlit run app/app.py (background)
2) Manually or via automation: load one sample (or document exact clicks)
3) Capture: predicted class visible, Grad-CAM image visible, caption, OOD panel
4) If automation impossible: write exact manual steps + require user screenshot — but try playwright/selenium if available

Do NOT change app unless broken. If broken, minimal fix in app/app.py only.

Evidence: work/reports/PHASE-BROWSER-evidence.md (steps + result). Exit criteria: "BROWSER_GOLDEN_OK" or list blockers.
```

---

## AGENT CI (P1)

**Write-set:** `.github/workflows/*` only if needed; otherwise just push+verify after SHIP

```text
[ONE-LINE TRUTH]

You are AGENT CI.
After SHIP is on a branch/PR:
1) Ensure ci.yml runs lint + pytest -m "not network" without requiring ckpt
2) Smoke tests must skip cleanly without ckpt
3) Get green check on GitHub Actions for the branch
4) Paste Actions URL + conclusion into work/reports/PHASE-CI-evidence.md

No 100% claims. Fix only CI failures.
```

---

## AGENT TRUTH (P1 — mandatory docs reset)

**Write-set:** `docs/SCOREBOARD.md`, `HANDOFF.md`, `work/reports/FREEZE.md` (mark SUPERSEDED), no other product code

```text
[ONE-LINE TRUTH]

You are AGENT TRUTH.
Hostile re-audit fixed score at ~84%. Kill 96%/100%/all-phases-GREEN theater.

Rewrite:
- docs/SCOREBOARD.md → blended 84%, Gate A NOT READY, list real reds from HOSTILE_REAUDIT.md
- HANDOFF.md → next = SHIP + FIX-EXIT + BROWSER + CI
- work/reports/FREEZE.md → header "SUPERSEDED — incomplete; see HOSTILE_REAUDIT.md"

Forbidden: inventing higher scores. Evidence: work/reports/PHASE-TRUTH-evidence.md
```

---

## AGENT DATA (P2 — optional, for R1)

**Write-set:** `data/processed/**` only if generating small set; or docs only

```text
[ONE-LINE TRUTH]

You are AGENT DATA (optional).
data/processed has 0 images → 93.17% unreproducible on clone.

Option A (preferred if network/time): python src/prepare_data.py --per-class 50 (or 100) into data/processed, verify splits non-empty, do NOT claim new 93% unless full retrain+eval.
Option B: strengthen docs that metrics are Colab artifact only + hashes in ARTIFACT_HASHES.md (coordinate with existing hashes).

Do not overwrite DATA_MANIFEST dishonestly.
Evidence: work/reports/PHASE-DATA-evidence.md
```

---

## AGENT GATE-A-REAL (P2 — only after P0+P1)

```text
[ONE-LINE TRUTH]

You are GATE-A REAL FREEZE orchestrator.
Prerequisites ALL true with paste:
[ ] SHIP committed (hash)
[ ] evaluate empty → exit 1
[ ] verify_golden_path.sh exit 0
[ ] pytest -m "not network" green
[ ] BROWSER_GOLDEN_OK or honest local-only label
[ ] CI green URL or honest "CI pending" YELLOW
[ ] SCOREBOARD says ≤ honest % (no 96% fiction)

Only then write work/reports/FREEZE_REAL.md and set Gate A checklist.
If any open: report real % — NEVER 100% unless every box green including browser preference.
```

---

## AGENT GATE-B (P3 — top 0.1%, after Gate A real)

Only assign after FREEZE_REAL:

| Sub | Focus |
|-----|--------|
| B-HOSTILE | Write HOSTILE_JUDGE.md by actually following README only on clean clone if possible |
| B-MOAT | Ensure localization + noise OOD are judge-obvious; no new orphans |
| B-PRESENT | Confirm demo.mp4 still matches UI; re-record note if not |
| B-ORCH | TOP_TIER_FREEZE only if hostile finds no P0 |

Paste from `work/TOP_0.1_PERCENT_PLAN.md` Gate B section — **after** real Gate A.

---

## Do NOT assign

| Wasteful agent | Why |
|----------------|-----|
| Full Phase 0–8 again | Already done at ~84% local |
| New backbone/ensemble training | Scope creep; no submission weights |
| “Make 100% docs” without ship | Greenwash |
| Delete samples / rewrite verify script | Working |
| Another mega plan file | Use this guide |

---

## Acceptance dashboard (your checklist as human)

```text
[ ] git log shows commit with samples + scripts + tests
[ ] remote clone (or friend machine) can run verify script after ckpt download
[ ] python src/evaluate.py ; echo $?   → 1 when empty
[ ] bash scripts/verify_golden_path.sh → 0 with ckpt
[ ] pytest tests/ -m "not network" → all pass
[ ] Streamlit sample → Grad-CAM seen (screenshot)
[ ] SCOREBOARD ≤ 90 until Gate A real; no 100%
[ ] CI green badge/URL
```

When **all** checked → honest score ~**90–93%** possible.  
**100%** only with Gate A real checklist.  
**Top 0.1%** only after Gate B + hostile clean.

---

## Suggested paste order (copy to chat threads)

1. AGENT TRUTH (5 min, parallel OK)  
2. AGENT FIX-EXIT (15 min, parallel OK)  
3. AGENT SHIP (after FIX-EXIT merge into tree)  
4. AGENT BROWSER + AGENT CI (after SHIP push)  
5. AGENT GATE-A-REAL  
6. Gate B only if you still want top 0.1%  

---

**Remember:** Agents already helped. They also lied with percentages.  
**Reassign for ship + honesty + exit codes + browser + CI — not for more status markdown.**
