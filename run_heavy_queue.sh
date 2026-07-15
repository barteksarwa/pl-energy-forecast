#!/usr/bin/env bash
# Heavy compute queue.
# Waits for the 2-year backtest to finish, then runs:
#   1. Feature sensitivity / PCA / SHAP / ablation  (~3 h)
#   2. TFT training campaign (d32/d64/d128 ± TSO)   (~8–12 h)
#
# Run from the project root:
#   bash run_heavy_queue.sh 2>&1 | tee logs/heavy_queue.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PY="$REPO/.venv/bin/python"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -u)] Heavy queue started. Repo: $REPO"
echo "[$(date -u)] Device check:"
$PY -c "import torch; print('  MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"

# -----------------------------------------------------------
# Wait for 2-year backtest result files to appear
# (use glob — backtest stamps with Warsaw date, queue uses UTC date)
# -----------------------------------------------------------
echo "[$(date -u)] Waiting for 2-year backtest result in $REPO/reports/backtests/*_2yr_summary.csv"
WAIT_S=0
while ! ls "$REPO/reports/backtests/"*_2yr_summary.csv 2>/dev/null | grep -q .; do
    sleep 60
    WAIT_S=$((WAIT_S + 60))
    echo "[$(date -u)] ... still waiting (${WAIT_S}s elapsed)"
    # Give up after 6 hours
    if [ $WAIT_S -ge 21600 ]; then
        echo "[$(date -u)] ERROR: 2yr backtest did not finish in 6 h. Aborting."
        exit 1
    fi
done
echo "[$(date -u)] Backtest results found: $(ls "$REPO/reports/backtests/"*_2yr_summary.csv)"
echo "[$(date -u)] Starting sensitivity analysis."

# -----------------------------------------------------------
# Step 1: Sensitivity / PCA / SHAP / ablation
# -----------------------------------------------------------
echo "[$(date -u)] === SENSITIVITY ANALYSIS ==="
$PY -m src.evaluation.run_sensitivity 2>&1 | tee "$LOG_DIR/sensitivity.log"
echo "[$(date -u)] Sensitivity done."

# -----------------------------------------------------------
# Step 2: TFT campaign
# -----------------------------------------------------------
echo "[$(date -u)] === TFT CAMPAIGN ==="
$PY -m src.models.deep.run_tft_campaign 2>&1 | tee "$LOG_DIR/tft_campaign.log"
echo "[$(date -u)] TFT campaign done."

echo "[$(date -u)] All jobs complete. Check reports/backtests/ and reports/sensitivity/."
