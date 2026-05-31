# Evals — Eval-Driven Development

## Philosophy
Capability defined by tests BEFORE the agent can fulfill it.

## Framework
Harbor for containerized evals.

## Metrics
- pass@k — probability of success in at least 1 of k attempts
- pass^k — probability ALL k attempts succeed

## Workflow
1. Write eval tasks first
2. Run → pass@k = 0% (expected)
3. Implement
4. Re-run → pass@k climbs
5. Graduate to regression suite at 95%
