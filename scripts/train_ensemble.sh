#!/usr/bin/env bash
# Galaxy-X-os Ensemble Training Script
# Trains all 3 backbones sequentially on Colab GPU
# Usage: bash scripts/train_ensemble.sh

set -e

echo "=========================================="
echo "Galaxy-X-os Ensemble Training"
echo "Backbones: ConvNeXt-Base + Swin-B + EfficientNet-B3"
echo "=========================================="

# Ensure data is prepared
if [ ! -d "data/processed/train" ] || [ -z "$(ls -A data/processed/train/*/ 2>/dev/null)" ]; then
    echo "Preparing data..."
    python3 src/prepare_data.py --per-class 500
fi

# Train ConvNeXt-Base
echo ""
echo "=== Training ConvNeXt-Base (88M params) ==="
python3 src/train.py \
    --backbone convnext_base \
    --checkpoint checkpoints/convnext_base.pth \
    --epochs 50 \
    --lr 3e-4 \
    --batch-size 32 \
    --label-smoothing 0.1 \
    --focal-gamma 2.0

# Train Swin-B
echo ""
echo "=== Training Swin-B (88M params) ==="
python3 src/train.py \
    --backbone swin_base_patch4_window7_224 \
    --checkpoint checkpoints/swin_base.pth \
    --epochs 50 \
    --lr 3e-4 \
    --batch-size 32 \
    --label-smoothing 0.1 \
    --focal-gamma 2.0

# Train EfficientNet-B3
echo ""
echo "=== Training EfficientNet-B3 (11.6M params) ==="
python3 src/train.py \
    --backbone efficientnet_b3 \
    --checkpoint checkpoints/efficientnet_b3.pth \
    --epochs 50 \
    --lr 3e-4 \
    --batch-size 32 \
    --label-smoothing 0.1 \
    --focal-gamma 2.0

# Evaluate ensemble
echo ""
echo "=== Evaluating Ensemble ==="
python3 src/evaluate.py \
    --ensemble \
    --tta advanced \
    --overwrite

echo ""
echo "=========================================="
echo "Ensemble training complete!"
echo "Checkpoints: checkpoints/convnext_base.pth, checkpoints/swin_base.pth, checkpoints/efficientnet_b3.pth"
echo "Results: results/evaluation_results.json"
echo "=========================================="