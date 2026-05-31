#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
DUPS=$(awk '/^\| [0-9]+ \|/ {print $2}' plan/EXECUTION.md | sort | uniq -d)
if [ -n "$DUPS" ]; then
  echo "DRIFT: duplicate wave numbers: $DUPS"; exit 1
fi
echo "EXECUTION.md is clean"
