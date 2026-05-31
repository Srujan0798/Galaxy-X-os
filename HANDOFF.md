# HANDOFF.md

> Switching sessions or orchestrators? Read this first.

## Current State

- **Active wave:** wave-2 (Real Data Integration)
- **Status:** READY TO DISPATCH
- **Last action:** Wave specs and task files generated
- **Next action:** Dispatch wave-2 task files to workers

## Wave Progress

| Wave | Name | Status | Tasks | Commit | Notes |
|------|------|--------|-------|--------|-------|
| 1 | Foundation | **SHIPPED** ✅ | 5/5 | `894538e` | Full codebase + synthetic data + checkpoint |
| 2 | Real Data Integration | **READY TO DISPATCH** | 0/4 | — | 4 task files ready |
| 3 | Full Training | pending | — | — | depends on wave-2 |
| 4 | Production Demo | pending | — | — | depends on wave-3 |
| 5 | Submission Package | pending | — | — | depends on wave-4 |

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
