#!/bin/bash
# Block commits with secrets
set -e
grep -r "API_KEY\|SECRET\|PASSWORD" src/ && exit 1 || true
