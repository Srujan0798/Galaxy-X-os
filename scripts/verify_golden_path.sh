#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SAMPLE_COUNT=$(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')

echo "=== Galaxy-X-os Golden Path Verification ==="

if [ "$SAMPLE_COUNT" -lt 5 ]; then
    echo "FAIL: Found only $SAMPLE_COUNT sample images (need >= 5)."
    echo "Generate samples first."
    exit 1
fi
echo "OK: $SAMPLE_COUNT sample images found."

if [ ! -f checkpoints/best_model.pth ]; then
    SAMPLE_PATH=$(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | head -1)
    echo ""
    echo "================================================"
    echo "  Checkpoint not found at checkpoints/best_model.pth"
    echo "================================================"
    echo ""
    echo "  To run the demo you need trained weights:"
    echo ""
    echo "  Option A — Download from Release:"
    echo "    https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0"
    echo "    Place the file at: checkpoints/best_model.pth"
    echo ""
    echo "  Option B — Train from scratch (Colab):"
    echo "    open notebooks/Galaxy_X_Colab.ipynb"
    echo "    Run all cells (~30 min on free GPU)"
    echo ""
    echo "  Option C — Train locally:"
    echo "    python src/prepare_data.py"
    echo "    python src/train.py"
    echo ""
    echo "  Then run: streamlit run app/app.py"
    echo "================================================"
    exit 2
fi
echo "OK: checkpoint found."

FIRST_SAMPLE=$(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | head -1)
echo "Running prediction on: $FIRST_SAMPLE"
echo ""

python3 -c "
from src.inference import ModelManager
import sys
try:
    manager = ModelManager('checkpoints/best_model.pth')
    result = manager.predict('$FIRST_SAMPLE')
    print()
    print(f'  Predicted: {result.class_name}')
    print(f'  Confidence: {result.confidence:.2%}')
    print(f'  Inference time: {result.inference_time_ms:.1f} ms')
    print()
    print('  Top-3:')
    for name, prob in result.top_k[:3]:
        print(f'    {name:25s}: {prob:.4f}')
    print()
    print('GOLDEN_PATH_OK')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

echo ""
echo "=== Golden path verification complete ==="
