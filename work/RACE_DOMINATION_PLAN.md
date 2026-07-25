# RACE DOMINATION PLAN — Beat everyone (dynamic)

**Truth baseline:** EXTERNAL_SHIP **42%** (`work/reports/EXTERNAL_TRUTH.md`)  
**Do not score local dirty tree as “the project.”**  
**Win condition:** Public GitHub + CI + 10-minute stranger demo hard to kick; metrics honest; craft beats peers.

---

## Non-negotiable framing

```text
MODEL_ARTIFACT 93.17%  ≠  project quality
LOCAL laptop green    ≠  submission quality  
EXTERNAL_SHIP         =  what the race ranks
```

Every agent must print EXTERNAL_SHIP estimate before/after their work.

---

## Sprint map (reassign THIS)

### WAVE 0 — STOP THE BLEEDING (parallel, 1–2 hours)

| Agent | Mission | Write-set | Done when |
|-------|---------|-----------|-----------|
| **W0-DEPS** | Fix install | `requirements.txt` only | `pip install -r requirements.txt` works on clean venv; no fake packages |
| **W0-TRUTH** | Kill score lies | SCOREBOARD, HANDOFF, FREEZE superseded | EXTERNAL_SHIP 42% written; no 93/96/100 as readiness |

**W0-DEPS already started:** remove `pytorch-gradcam-plusplus` (CI killer).

---

### WAVE 1 — PUBLIC GOLDEN PATH (blocking for race)

| Agent | Mission | Write-set | Done when |
|-------|---------|-----------|-----------|
| **W1-SHIP** | One PR/commit that makes clone demoable | samples, scripts, app, tests, attic, requirements, README honesty | Fresh clone: install works; samples ≥5; streamlit can click sample after ckpt wget; verify script exists |
| **W1-EXIT** | Fail-loud | evaluate/train/gradcam/utils | empty data → **exit 1** |
| **W1-CI** | Green badge | workflows + deps | GitHub Actions green on main/PR |

**Acceptance (must run on CLEAN clone after push):**
```bash
git clone <repo> /tmp/race-check && cd /tmp/race-check
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wget -q -O checkpoints/best_model.pth \
  https://github.com/Srujan0798/Galaxy-X-os/releases/download/v1.0/best_model.pth
test $(find data/samples -name '*.png' | wc -l) -ge 5
bash scripts/verify_golden_path.sh   # exit 0
pytest tests/ -q -m "not network"
# streamlit run app/app.py → sample works
```

**Target EXTERNAL_SHIP after Wave 1: ~70–75%**

---

### WAVE 2 — OUTCLASS COMPETITORS (differentiation)

| Agent | Mission | Why judges pick YOU |
|-------|---------|---------------------|
| **W2-XAI** | Grad-CAM quality in app + 15 figures already shipped; fix any broken path | Explainability 15% weight |
| **W2-BONUS** | All 4 PS bonuses demoable in app (caption labeled, OOD, loc, web) | Bonus 15% |
| **W2-HONEST-METRICS** | Residual spiral/elliptical story + hashes + “artifact not re-run on empty clone” | Trust vs peers who fake 99% |
| **W2-PRESENT** | Judge_60s + video match UI + REPORT.pdf | Docs 15% |
| **W2-PERF** | latency_bench committed; &lt;5s proven | Efficiency 15% |

**Target EXTERNAL_SHIP: ~82–88%**

---

### WAVE 3 — TOP 0.1% MOAT

| Agent | Mission |
|-------|---------|
| **W3-HOSTED** | Streamlit Cloud / HF Space with ckpt (if allowed) — zero-friction judge |
| **W3-HOSTILE** | Clean-machine hostile report; zero P0 |
| **W3-FREEZE** | Only if EXTERNAL_SHIP ≥90 and hostile clean |

---

## PASTE PROMPTS

### Global prefix
```text
RACE MODE. Project /Users/srujansai/Desktop/Galaxy-X-os
Truth: work/reports/EXTERNAL_TRUTH.md — EXTERNAL_SHIP is ~42% until public main is fixed.
93.17% is MODEL_ARTIFACT only — never call it project readiness.
Your work only counts when a FRESH CLONE proves it (or you prove push+CI).
No 100%/top 0.1% claims. Evidence file required.
```

### W0-DEPS
```text
[GLOBAL]
You are W0-DEPS. Fix requirements.txt so pip install works.
Remove fake pytorch-gradcam-plusplus. Keep grad-cam==1.5.2.
Make onnx optional or real packages only. Prove: clean venv pip install -r requirements.txt.
Evidence: work/reports/W0-DEPS-evidence.md
```

### W1-SHIP
```text
[GLOBAL]
You are W1-SHIP — MOST IMPORTANT.
Ship to git (ask user before push) everything needed for cold-clone golden path:
data/samples/**, scripts/verify_golden_path.sh, ultra_win_gate.sh, app/app.py samples,
tests e2e predict smoke, requirements fix, attic cleanup, README that does not lie.
DO NOT leave improvements only local.
Prove acceptance commands on theory of clean clone.
Evidence: work/reports/W1-SHIP-evidence.md + commit hash
```

### W1-EXIT
```text
[GLOBAL]
You are W1-EXIT. empty data/processed → evaluate/train/gradcam sys.exit(1).
Prove: python src/evaluate.py ; echo $?  → 1
Evidence: work/reports/W1-EXIT-evidence.md
```

### W1-CI
```text
[GLOBAL]
You are W1-CI. GitHub Actions must go green. Fix workflows/deps only as needed.
No ckpt required in CI. Skip smoke if no ckpt.
Paste green run URL. Evidence: work/reports/W1-CI-evidence.md
```

### W2-* (after Wave 1 green)
Use TOP_0.1 plan phases but **only after EXTERNAL_SHIP ≥70** measured by clean clone.

---

## Human commander checklist

```text
WAVE 0
[ ] requirements installs on clean machine
[ ] SCOREBOARD shows EXTERNAL_SHIP ~42% not 93/96

WAVE 1
[ ] push contains samples + scripts
[ ] clean clone verify_golden_path exit 0
[ ] CI green URL
[ ] README sample claim is true

WAVE 2
[ ] all 4 bonuses visible
[ ] latency json in repo
[ ] hashes in repo
[ ] Judge_60s ready

WAVE 3
[ ] hostile clean
[ ] optional hosted demo
```

---

## Anti-patterns (fire agents who do these)

- Raising SCOREBOARD without clean-clone proof  
- Adding more uncommitted “SOTA” modules  
- Claiming 93% as readiness  
- Writing FREEZE while CI red  
- README features that are not in the repo  

---

## Expected score trajectory

```text
Now public:     42%  EXTERNAL_SHIP
After Wave 0–1: 70–75%
After Wave 2:   82–88%
After Wave 3:   90%+ race contender / top 0.1% only if hostile agrees
```

**Win the race by fixing what the external auditor already saw — public install, public demo, public CI — not by writing higher percentages in markdown.**
