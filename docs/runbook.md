# Runbook

## Daily Operations

```bash
make train    # Start training
make evaluate # Run evaluation
make app      # Launch demo
```

## Troubleshooting

### OOM during training
Reduce batch_size in config.yaml

### Grad-CAM not working
pip install grad-cam

### Model not found
Train first: python src/train.py
