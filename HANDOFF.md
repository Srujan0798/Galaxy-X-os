# HANDOFF — HARD TRUTH

**EXTERNAL_SHIP: 42%** (public race score)  
**LOCAL_DIRTY: ~72%** (not shippable until push)  
**MODEL_ARTIFACT: 93.17%** (accuracy only — not readiness)  
**NOT 84%. NOT 96%. NOT 100%.**

## You were right to distrust high %
External cold-clone audit confirms **&lt;50% ship readiness**.  
I was wrong to treat local green tests as the competition score.

## Killers on public main
1. Fake pip package `pytorch-gradcam-plusplus` → **install + CI die**  
2. **No sample images** on GitHub (README still says click sample)  
3. Agent fixes **never pushed**  
4. CI **all failed** on latest push  
5. WIP modules on main  

## Read
- `work/reports/EXTERNAL_TRUTH.md`
- `work/RACE_DOMINATION_PLAN.md` ← **reassign from here**

## Assign now (Wave 0–1 only)
| Agent | Job |
|-------|-----|
| W0-DEPS | requirements installable (fake package removed locally — **must commit**) |
| W0-TRUTH | keep scores honest (done) |
| **W1-SHIP** | push samples + scripts + app + tests + attic |
| W1-EXIT | empty data exit 1 |
| W1-CI | green Actions |

## Law
No readiness number above EXTERNAL_SHIP without a **fresh clone proof**.  
No using 93% as project %.
