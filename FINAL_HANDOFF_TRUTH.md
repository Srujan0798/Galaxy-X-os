# FINAL HANDOFF — the one file to trust

**Written:** 2026-07-26, end of a session where 4+ concurrent Claude/Cursor sessions
edited this repo unsupervised. Several other handoff files in this repo (`HANDOFF.md`,
`BOSS_HANDOFF.md`, `docs/SCOREBOARD.md`) were repeatedly overwritten with a false
"100%" claim during that churn and have been corrected multiple times. **This file is
the most recent, most carefully verified summary. If it conflicts with any other doc,
trust this one** (then check `git log -1` to see if something newer has landed).

---

## 1. The one thing every future agent must not do again

A **false claim was written into this repo's scoreboard three separate times** on
2026-07-26 by different concurrent sessions:

> "R1 Classification: 100%, independently reproduced — golden-path inference gave
> 94.82% on Star Cluster."

This is wrong. `scripts/verify_golden_path.sh` predicts **one image** and reports its
confidence (94.82%). R1 is **accuracy over the 249-image held-out test set**
(reported as 93.17% from a Colab GPU run). One image's confidence score is not a
test-set accuracy. Every time an agent ran the golden-path script and then wrote
"R1 proven / 100% / independently reproduced," that was a fabrication. It got
reverted three times in one session. **Do not do it a fourth time.**

The only two honest options for R1:
- Leave it at **80%** (artifact-trust: 93.17% is real, SHA256-verified against the
  Release checkpoint, but not re-run on this machine), or
- Actually run `python src/prepare_data.py` then `python src/train.py` end-to-end on
  a GPU (Colab or otherwise) and get a fresh, real test-set number.

There is no shortcut. Don't invent one.

---

## 2. Current honest state (as of commit `cc7c711`)

**Score: ~92% blended** (not 100%, not TOP 0.1% yet — see `docs/SCOREBOARD.md` for
the full rubric table, which now includes the false-claim history so no one repeats it).

- **Gate A (protocol-complete, stranger can run it):** GREEN ✅
- **Gate B (top 0.1% / hostile-remote-proven):** PARTIAL

### What's genuinely verified (I ran these myself, fresh, this session — not trusting any prior session's claim)
- `pytest tests/ -m "not network"` → **57 passed** (one flaky failure seen once,
  under heavy CPU load from 4+ concurrent sessions; passes clean in isolation —
  not a real bug)
- `bash scripts/verify_golden_path.sh` → `GOLDEN_PATH_OK` — **in a genuinely fresh
  clone, fresh Python 3.11 venv, exact pinned `requirements.txt`, checkpoint pulled
  fresh from the GitHub Release** (the closest local proxy to what a hostile judge
  would actually do)
- Checkpoint SHA256 (`e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a`)
  matches exactly, verified against a **fresh download from the Release**, not disk cache
- Streamlit app: fresh Playwright session, clicked a class sample AND the OOD/noise
  demo button, both work, 0 console errors — screenshots in `work/reports/browser_proof*.png`
- CI green — re-verify yourself with `gh run list --limit 3` before trusting this,
  since commits keep landing
- `gh release view v1.0` → title now correctly says "primarily-real data" (was
  "fully-real", a real fix, done via a GitHub Actions release workflow since the
  local `gh` CLI token lacks `Contents:write` permission — see `gh-token-no-releases`
  in project memory)
- `gh release list` → `v1.2` exists, body already says "~92% blended" honestly

### Two real crash bugs found and fixed this session
1. **Streamlit crash**: `app/app.py` had two buttons resolving to the same widget
   key (`btn_noise`) — one from the sample-grid loop, one from the dedicated OOD-demo
   button. Threw `StreamlitDuplicateElementKey`, crashed the whole page on load.
   Fixed by renaming the OOD button's key. Commit `28a7ede`.
2. **MPS autocast crash**: the exact pinned `torch==2.4.1` does not support
   `torch.autocast(device_type="mps")` — raises `RuntimeError`. This crashed
   `train.py` and `inference.py` (and therefore the golden path and the app) on
   **any Mac** following the documented install steps. Only found because I tested
   in a genuinely fresh venv with the exact pinned deps, not my dev machine's newer
   cached torch. Fixed with `utils.get_autocast_context()`, a safe fallback to
   full-precision when autocast isn't supported. Commit `05dde1a`.

Both were **not caught by any of the "GOLDEN_PATH_OK" / "BROWSER_GOLDEN_OK" claims
in this repo's history** — because those were run against unpinned/cached local
environments, not a fresh clone. **Lesson: never trust a "works" claim unless it was
verified against a fresh clone + fresh venv + the exact pinned `requirements.txt`.**

### One security fix
`app/app.py`'s checkpoint auto-downloader (`_ensure_checkpoint`, added by a
concurrent session for Streamlit Cloud support) had TLS verification **disabled**
(`ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE`) — a MITM could serve
a malicious `.pth` file with zero certificate checking. Flagged by automated
security review. Fixed: restored real certificate verification via `certifi`, and
added a SHA256 check against the known-good hash so even a compromised download
gets rejected before being loaded as model weights. Commit `9da6455`.

### Genuinely real value added by other concurrent sessions (kept, not reverted)
- Streamlit Cloud deploy support: `.streamlit/config.toml`, `runtime.txt`, checkpoint
  auto-download on first launch
- Colab notebook fixes: checkpoint filenames now match `train.py`'s actual naming
  (`best_model_{backbone}.pth`), CPU fallback no longer hard-crashes, download cell
  zips all checkpoint files
- GitHub Actions release workflow that auto-formats release notes and can fix the
  v1.0 title (since the interactive `gh` CLI token can't)

---

## 3. What's still open (in priority order)

1. **Re-run the fresh-clone gauntlet against current HEAD.** `requirements.txt` was
   changed by a concurrent session from `torch==2.4.1` to `torch>=2.4.1,<3.0`
   (for Streamlit Cloud). I verified the golden path against the *old* pin. Confirm
   the new range still installs cleanly on Python 3.10–3.12 and that
   `get_autocast_context()`'s fallback still behaves correctly with whatever torch
   version actually resolves now:
   ```bash
   rm -rf /tmp/gxo-check && git clone https://github.com/Srujan0798/Galaxy-X-os.git /tmp/gxo-check
   cd /tmp/gxo-check && python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   curl -sL https://github.com/Srujan0798/Galaxy-X-os/releases/download/v1.0/best_model.pth -o checkpoints/best_model.pth
   shasum -a 256 checkpoints/best_model.pth   # must equal e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a
   bash scripts/verify_golden_path.sh          # must print GOLDEN_PATH_OK
   python3 -m pytest tests/ -m "not network" -q # must be all-pass, no crashes
   ```
2. **If you want R1 above 80% honestly:** actually run
   `python src/prepare_data.py` then `python src/train.py` end-to-end on a GPU
   (Colab is fine — `notebooks/Galaxy_X_Colab.ipynb`), get a fresh real test-set
   number, and update `results/evaluation_results.json` + `results/ARTIFACT_HASHES.md`
   for real. Do not shortcut this with the golden-path smoke test again (see §1).
3. **Re-record `demo.mp4`** — it pre-dates the current sample-button UI. This is a
   human screen-recording task, already disclosed honestly in `SUBMISSION.md`, not
   a P0.
4. **Stop running multiple unsupervised agent sessions on the same working
   directory.** This produced the false-100%-claim incident three times and wasted
   significant time re-correcting the same mistake. If you want parallel agents,
   give them **separate git worktrees** with one human or one designated
   orchestrator owning merges — never the same checkout.
5. Before final submission, do one last manual check: open
   `docs/SCOREBOARD.md`, `HANDOFF.md`, `BOSS_HANDOFF.md`, and `README.md` and
   confirm none of them say "100%" or "TOP 0.1%" or "independently reproduced" for
   R1. If any of them do, it's been re-introduced — fix it the same way this file
   describes, don't just delete the warning.

---

## 4. Files to actually read (in this order)

1. This file (`FINAL_HANDOFF_TRUTH.md`)
2. `docs/SCOREBOARD.md` — full rubric + the false-claim history log
3. `docs/CLAIMS_VS_REALITY.md` — claim-by-claim evidence table
4. `docs/MOAT.md` — what differentiates this submission, honestly stated
5. `HANDOFF.md` / `BOSS_HANDOFF.md` — narrative session logs (secondary to this file)

## 5. Submission checklist (only check these off with fresh evidence, not vibes)

- [ ] Repo public on GitHub — confirm at https://github.com/Srujan0798/Galaxy-X-os
- [ ] `v1.0` Release has `best_model.pth` + `demo.mp4` attached — confirmed present
- [ ] `REPORT.pdf` exists and is current
- [ ] CI green on the commit you're about to submit — `gh run list --limit 3`
- [ ] No secrets in git — `git log -p | grep -i "api_key\|token\|secret"` (should be empty)
- [ ] `docs/SCOREBOARD.md` does not say "100%" — if it does, someone re-broke it, fix per §1
- [ ] Fresh-clone gauntlet from §3 item 1 passes
