# MOAT — Galaxy-X-os

1. **Primary job vs clones.** Most hackathon EfficientNet-on-Galaxy10 entries stop at
   `train.py` + a static accuracy number. This entry ships a **runnable stranger path**
   (clone → install → sample click → prediction + Grad-CAM + caption + OOD in under
   2 minutes) that was verified in a genuinely fresh environment, not just asserted.

2. **10-min verifiable proof.** `scripts/verify_golden_path.sh` (CLI) and the Streamlit
   app (browser) both give a hostile judge a real, falsifiable pass/fail in minutes —
   not a slide claiming "it works."

3. **Security/clones skip.** N/A for this archetype (local ML research tool, no auth
   surface, no multi-tenant data) — explicitly not claimed to avoid loophole LH-15
   (claiming a security axis that doesn't exist for this project shape).

4. **Real depth.** Data provenance is honestly mixed-source and disclosed per class
   (`DATA_MANIFEST.json`, `is_real` flag) rather than a blanket "real data" claim;
   Grad-CAM, TTA, ONNX export, and anomaly/OOD detection are wired into the same
   live inference path the demo uses — not separate unwired scripts.

5. **Craft signature.** The honesty layer itself: `docs/CLAIMS_VS_REALITY.md` and
   `work/reports/HOSTILE_REAUDIT.md` are public, dated, and self-critical — most
   submissions don't publish their own audit trail, including two admitted
   scoreboard-inflation incidents (~96% claimed, corrected to ~90%) and two live
   crash bugs found and fixed via fresh-environment probing rather than reused
   session state.

6. **What we do NOT fake.**
   - Do not claim the 93.17% figure is locally re-derived — it's a Colab GPU
     artifact with a matching SHA256, and the repo says so.
   - Do not claim the demo video matches the current UI (it doesn't; disclosed).
   - Do not claim "production-ready" — this is a research/hackathon prototype,
     labeled as such.
   - Do not claim the GitHub Release text is fixed — token can't edit it; disclosed
     with the exact manual fix pasted in HANDOFF.md.
