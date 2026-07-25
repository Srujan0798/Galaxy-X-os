# EXTERNAL TRUTH — Cold-clone / hostile-judge score

**Date:** 2026-07-25  
**Method:** Fresh `git clone https://github.com/Srujan0798/Galaxy-X-os` + public CI + Release probe  
**Mindset:** What a race competitor or external auditor sees — **NOT** your laptop with a private ckpt and uncommitted files  

---

## APOLOGY / CORRECTION

| Earlier claim | Why it was wrong |
|---------------|------------------|
| “~84% project ready” | Scored **dirty local tree** (uncommitted samples, scripts, tests). **Judges clone GitHub.** |
| “93%” mixed into readiness | **93.17% is a training metric artifact**, not product/submission health. Conflating them is greenwash. |
| Agent “96% FREEZE” | Already rejected; still too high even as local aspiration. |

**External report “below 50%” is CREDIBLE for public submission readiness.**  
This re-score agrees: **EXTERNAL_SHIP ≈ 42%**.

---

## Three numbers (never mix them)

| Score name | Value | Meaning |
|------------|-------|---------|
| **EXTERNAL_SHIP** | **42%** | What graders get from public repo + CI today |
| **LOCAL_DIRTY** | **~72%** | Your machine with uncommitted agent work + local ckpt |
| **MODEL_ARTIFACT** | **93.17%** | Reported test accuracy JSON — **not** project score |

If anyone says “we’re at 93% ready” → **lie.**  
If anyone says “we’re at 100%” → **lie.**  
**Race ranking uses EXTERNAL_SHIP until you push and CI is green.**

---

## Cold clone facts (evidence)

```text
clone: success
data files: 4 (only .gitkeep + 2 json manifests) — ZERO sample images
scripts/: empty (.gitkeep only)
checkpoints/: empty .gitkeep
results/: HAS evaluation_results.json + gradcam (good)
tests/e2e: import-only theater
README: tells judge to "Click a sample" — SAMPLES NOT IN REPO → doc lie
requirements.txt: contains FAKE package pytorch-gradcam-plusplus
→ pip install FAILS
→ CI on main: ALL FAILURES (CI, Security, Test Matrix)
Release v1.0: best_model.pth downloadable (302 OK) — good but unreachable if install dies first
origin tip: 495b41a "v1.2 ensemble + TTA + detection + ONNX" — WIP dump, not clean product
```

CI fail excerpt:
```text
ERROR: No matching distribution found for pytorch-gradcam-plusplus>=0.1.0
```

---

## EXTERNAL_SHIP rubric (what wins the race)

| Criterion | Wt | Cell | Why so low |
|-----------|----|------|------------|
| Stranger install | 20% | **15** | Fake pip dep kills install + CI |
| Golden path demo | 25% | **25** | Need ckpt download; **no samples**; README false promise |
| Repro metrics | 20% | **40** | JSON+plots exist; cannot re-eval (no data); trust-only |
| Code quality / WIP | 15% | **35** | Dead modules on main (tta, detection, gradcam_plus, pseudo) |
| Tests/CI | 10% | **10** | CI red; e2e theater |
| Docs honesty | 10% | **55** | Strong writeup but overclaims + sample lie |
| **EXTERNAL_SHIP** | 100% | | **≈ 42%** |

Aligned with external “&lt;50%”.

---

## Why LOCAL looks better (and still not enough)

Local uncommitted has samples, verify script, better e2e, attic cleanup — **judges never see it until SHIP**.  
Even local: empty `data/processed`, evaluate exit 0 bug, browser unproven → not elite.

---

## What beats everyone in the race (dynamic target)

Top teams will have:
1. `pip install -r requirements.txt` works  
2. One-command or 3-step demo with **in-repo samples**  
3. CI green  
4. Honest metrics + Grad-CAM  
5. Clean repo (no half-merged ensemble dump)  
6. Working Streamlit  

You currently lose at **1, 2, 3, 5** on public main.

---

## Immediate kill list (order = race impact)

| # | Action | Impact on EXTERNAL_SHIP |
|---|--------|-------------------------|
| 1 | Remove fake `pytorch-gradcam-plusplus` from requirements | +15–20 |
| 2 | Commit + push samples + verify script + e2e smoke + attic cleanup | +15–20 |
| 3 | Fix README to match reality OR ship samples so “click sample” is true | +5 |
| 4 | CI green | +8 |
| 5 | evaluate/train exit 1 on empty data | +3 |
| 6 | Browser proof | +3 |
| 7 | Optional: hosted Streamlit | +5 moat |

Path: **42% → ~70%** after 1–4 shipped · **→ ~85%** with polish · **→ race-winning** only after Gate B craft.

---

## Command pack (external)

```bash
git clone https://github.com/Srujan0798/Galaxy-X-os.git /tmp/gx-x
cd /tmp/gx-x
pip install -r requirements.txt   # FAILS today
find data/samples -name '*.png' | wc -l   # 0
gh run list --limit 3               # failure
```

---

**Signed:** External-truth audit.  
**Do not use 84/93/96/100 as project readiness.**  
**Use EXTERNAL_SHIP 42% until public main is fixed.**
