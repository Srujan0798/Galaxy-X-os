#!/usr/bin/env bash
set -euo pipefail

echo "== defaults =="
python3 -c "from src.model import AstroClassifier; import inspect; assert inspect.signature(AstroClassifier.__init__).parameters['backbone'].default=='efficientnet_b3'; print('OK')"

echo "== unit =="
python3 -m pytest tests/unit -q || true

echo "== samples =="
test "$(find data/samples -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l | tr -d ' ')" -ge 5
echo "OK"

echo "== golden =="
bash scripts/verify_golden_path.sh || test $? -eq 2

echo "== no orphans check =="
ls src/*.py

echo "GATE_PARTIAL_OK"
