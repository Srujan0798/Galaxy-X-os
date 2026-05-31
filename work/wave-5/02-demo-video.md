# T5.2: Demo Video Recording Prep

## Pre-Recording Checklist

Before recording, ensure:

- [ ] All evaluation outputs exist in `results/`
- [ ] Grad-CAM samples generated in `results/gradcam/`
- [ ] Streamlit app runs without errors: `streamlit run app/app.py`
- [ ] Training logs present in `results/logs/`
- [ ] Checkpoint verified: `python -c "from src.inference import ModelManager; m = ModelManager()"`

## Recording Setup

1. **Screen Resolution**: 1920x1080 (1080p)
2. **Recording Software**: OBS Studio or Loom
3. **Audio**: External mic recommended, speak clearly at moderate pace
4. **Timer**: Keep total under 2:00

## Scene Breakdown

| Time | Scene | Action |
|------|-------|--------|
| 0:00-0:05 | Title | Show project name + team |
| 0:05-0:30 | Web Demo 1 | Upload spiral galaxy, show prediction |
| 0:30-0:45 | Web Demo 2 | Upload nebula, show prediction |
| 0:45-0:55 | Grad-CAM Grid | Show `results/gradcam/_summary_grid.png` |
| 0:55-1:10 | CLI Inference | Run `python src/inference.py data/processed/test/` |
| 1:10-1:20 | Bonus | Run `python src/bonus.py` sample |
| 1:20-1:35 | Evaluation | Run `python src/evaluate.py`, show JSON |
| 1:35-1:50 | Training | Show TensorBoard curves |
| 1:50-2:00 | Closing | Thank you slide |

## Post-Recording

1. Export as MP4 (H.264, ~10MB for 2 min)
2. Upload to Google Drive / YouTube
3. Add link to README.md
