#!/usr/bin/env bash
# verify_live.sh — Galaxy-X-os Streamlit app live verification
# ETERNITY proof harness: probes the live Streamlit app if a URL is provided.
set -euo pipefail

STREAMLIT_URL="${1:-}"
FAILED=0

if [ -z "$STREAMLIT_URL" ]; then
    echo "Usage: $0 <streamlit-url>"
    echo "Example: $0 https://galaxy-x-os.streamlit.app"
    echo ""
    echo "If no URL, runs local golden-path verify as fallback."
    echo ""
    # Fallback: local golden path
    echo "=== LOCAL PROBE (no URL provided) ==="
    bash scripts/verify_golden_path.sh
    exit 0
fi

echo "=== PROBE 1: Health check ==="
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$STREAMLIT_URL" 2>/dev/null || echo "000")
if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "302" ]; then
    echo "PASS: HTTP $HTTP_STATUS"
else
    echo "FAIL: HTTP $HTTP_STATUS"
    FAILED=$((FAILED+1))
fi

echo ""
echo "=== PROBE 2: Page contains model info ==="
if curl -s --max-time 20 "$STREAMLIT_URL" 2>/dev/null | grep -qi "galaxy\|star\|cluster\|nebula\|prediction"; then
    echo "PASS: galaxy-related content detected"
else
    echo "FAIL: no astronomy/model content found"
    FAILED=$((FAILED+1))
fi

echo ""
echo "=== PROBE 3: No mock markers ==="
BODY=$(curl -s --max-time 20 "$STREAMLIT_URL" 2>/dev/null)
if echo "$BODY" | grep -qi "placeholder\|mock\|dummy\|not yet implemented"; then
    echo "FAIL: mock-marker text detected"
    FAILED=$((FAILED+1))
else
    echo "PASS: no mock markers found"
fi

echo ""
echo "=== SUMMARY ==="
if [ "$FAILED" -eq 0 ]; then
    echo "VERIFY_LIVE: PASS ✅"
    exit 0
else
    echo "VERIFY_LIVE: FAIL ❌ ($FAILED probe(s) failed)"
    exit 1
fi
