#!/bin/bash
WAVE=$1
TASK=$2
FILE="orchestrator/memory/session/${WAVE}-${TASK}.events.jsonl"
if [ -f "$FILE" ]; then
  echo "Last 5 events:"
  tail -5 "$FILE"
else
  echo "No session log found"
fi
