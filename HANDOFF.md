# HANDOFF.md

> Switching sessions or orchestrators? Read this first.

## Current State

- **Active wave:** wave-1 (Foundation)
- **Status:** Ready to dispatch
- **Last action:** Project setup complete
- **Next action:** Dispatch wave-1 task files to workers

## Wave Progress

| Wave | Name | Status | Tasks | Commit | Notes |
|------|------|--------|-------|--------|-------|
| 1 | Foundation | **READY TO DISPATCH** | 0/5 | — | Setup complete, task files ready |
| 2 | Data Pipeline | pending | — | — | depends on wave-1 |
| 3 | Training + Evaluation | pending | — | — | depends on wave-2 |
| 4 | Explainability + Demo | pending | — | — | depends on wave-3 |
| 5 | Polish + Submission | pending | — | — | depends on wave-4 |

## Quick Recovery

```bash
# Verify setup
make validate

# Run tests
make test

# Start app
make app
```

## Last Session Events

See: `orchestrator/memory/session/`
