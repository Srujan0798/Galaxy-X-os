# HOW_TO_RUN.md

## Dual-Tier Workflow (Plain Language)

### Tier 1 — Orchestrator (Claude or Kimi)

1. Reads project state, specs, reports
2. Writes task files into `work/`
3. Reviews worker reports
4. Merges output, updates state

### Tier 2 — Workers (OpenCode CLI)

1. Receive ONE self-contained task file
2. Execute, write code to repo
3. Write standardized report to `work/reports/`

### Starting a Wave

```bash
# Orchestrator runs:
/plan wave-1

# This generates:
#   .specify/specs/wave-1/{spec,plan,tasks,contracts}
#   work/wave-1/01-task.md ...

# Workers run in parallel:
cd /Users/srujansai/Desktop/Galaxy-X-os
# Open 3-5 OpenCode CLI windows
# Paste task files from work/wave-1/
```

### Daily Commands

```bash
make install      # pip install -r requirements.txt
make train        # python src/train.py
make evaluate     # python src/evaluate.py
make gradcam      # python src/gradcam.py
make app          # streamlit run app/app.py
make test         # pytest tests/ -v
make lint         # ruff check src/ && mypy src/
```
