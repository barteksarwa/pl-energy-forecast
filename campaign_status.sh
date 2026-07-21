#!/usr/bin/env bash
# One-shot status of the deep-history campaign. Run: bash campaign_status.sh
cd "$(dirname "$0")"

PID=50796
if ps -p "$PID" > /dev/null 2>&1; then
  echo "campaign RUNNING (pid $PID, started 2026-07-22 00:15)"
else
  echo "campaign process gone — finished or died. Check the last lines below."
fi
echo
echo "=== job sequence (data/logs/campaign_deep_history.log) ==="
grep -E "START|OK|FAIL|CAMPAIGN DONE" data/logs/campaign_deep_history.log
echo
echo "=== current job tail ==="
LAST=$(ls -t data/logs/2026-07-22_*.log 2>/dev/null | head -1)
echo "[$LAST]"
tail -5 "$LAST" 2>/dev/null
echo
echo "Final report (after CAMPAIGN DONE): reports/backtests/*_deep_history_campaign.md"
