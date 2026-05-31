#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
echo "Validating project structure..."
for f in README.md CLAUDE.md HANDOFF.md src/train.py src/model.py app/app.py; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"; exit 1
  fi
done
echo "Validation passed"
