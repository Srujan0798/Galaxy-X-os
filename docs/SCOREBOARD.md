# SCOREBOARD — HARD TRUTH (external-aligned)

**Updated:** 2026-07-25 — **corrected after external report**  
**Plan:** [`work/RACE_DOMINATION_PLAN.md`](../work/RACE_DOMINATION_PLAN.md) · **Evidence:** [`work/reports/EXTERNAL_TRUTH.md`](../work/reports/EXTERNAL_TRUTH.md)

## Never mix these

| Name | Value | Meaning |
|------|-------|---------|
| **EXTERNAL_SHIP** | **42%** | Public GitHub + CI + cold clone — **THIS IS THE RACE SCORE** |
| **LOCAL_DIRTY** | ~72% | Uncommitted local work + private ckpt — **NOT what graders clone** |
| **MODEL_ARTIFACT** | 93.17% | Test accuracy number in JSON — **NOT project readiness** |

Prior claims (84%, 96%, 100%) as project readiness: **INVALID**.

## Why EXTERNAL_SHIP is 42% (matches external &lt;50%)

| Killer | Status |
|--------|--------|
| `pytorch-gradcam-plusplus` fake dep | **CI + pip FAIL** on main |
| `data/samples` images | **NOT on origin** |
| `scripts/verify_*` | **NOT on origin** |
| CI on main | **ALL RED** |
| README “click a sample” | **LIE on public repo** |
| Dead WIP on main | tta, detection, gradcam_plus, pseudo |
| E2E on origin | import-only |

## Official competition rubric (EXTERNAL framing)

| Criterion | Wt | Cell % | Note |
|-----------|----|--------|------|
| Classification story | 40% | 55 | Artifact exists; unreproducible from clone without heavy work |
| Efficiency | 15% | 40 | Claimed under 5s; not proven on clean path |
| Explainability | 15% | 70 | Grad-CAM images in repo (good) |
| Bonus | 15% | 35 | Code exists; install broken; samples missing |
| Docs | 15% | 50 | Strong text; overclaim + sample lie |
| **Blended external** | | **~48%** | ≈ external &lt;50% |

With install failure weighted as hard fail on “can use product,” **ship score 42%**.

## Wave targets

| After | EXTERNAL_SHIP target |
|-------|----------------------|
| Wave 0 (deps + truth) | 48–52% |
| Wave 1 (ship samples + CI green) | **70–75%** |
| Wave 2 (polish + bonuses) | **82–88%** |
| Wave 3 (moat + hostile) | **90%+** |

## Log
| Date | Score | Note |
|------|-------|------|
| agents | claimed 96 | rejected |
| local re-audit | 84 | rejected for race — local only |
| **external truth** | **42%** | cold clone + CI red + fake pip |
