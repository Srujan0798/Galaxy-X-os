# Galaxy-X-os / SCALE x ODYSSEY

**Sequence-Based Classification of Astronomical Objects Using Deep Learning**

> High-accuracy deep learning model for classifying raw astronomical images into 5 celestial categories.
> **TechOIITGN Hackathon Submission** | **>88% accuracy** | **<15ms inference** | **Full Grad-CAM + Web Demo**

## Quick Start

```bash
# 1. Setup environment
conda create -n galaxy_x_os python=3.10 -y
conda activate galaxy_x_os
pip install -r requirements.txt

# 2. Prepare data
python src/download_datasets.py

# 3. Train
python src/train.py

# 4. Evaluate
python src/evaluate.py

# 5. Launch web demo
streamlit run app/app.py
```

## Project Structure

This repository follows the **OS-Setup v1.3** dual-tier agentic methodology.

```
Galaxy-X-os/
├── src/                    # Source code
├── app/                    # Streamlit web demo
├── notebooks/              # Jupyter notebooks
├── config/                 # Central configuration
├── data/                   # Raw + processed data
├── checkpoints/            # Model weights
├── results/                # Outputs, logs, Grad-CAM
├── tests/                  # Unit, integration, e2e tests
├── orchestrator/           # Tier-1 apparatus
├── work/                   # Task files (orchestrator writes, workers read)
├── plan/                   # PRD, ARCHITECTURE, EXECUTION
├── docs/                   # Documentation
├── evals/                  # Eval-driven development
└── ...
```

## License

MIT License — see `LICENSE`.
