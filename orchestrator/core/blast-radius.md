# Blast Radius

| Radius | Examples | Containment |
|---|---|---|
| r0 — Read-only | Read files, grep, list | None needed |
| r1 — Local repo | Write to src/, run tests | Auto allowed; git protects |
| r2 — Local services | Apply DB migration to dev | Confirm; reversible |
| r3 — Remote services | Push to GitHub, send Slack | Confirm; possibly reversible |
| r4 — External humans | Email customers, file tickets | Always confirm; HARD to reverse |
| r5 — Money or data loss | Charge card, drop prod table | Block by default |

Auto mode skips r0/r1 only.
