#!/usr/bin/env bash
# Deep-history campaign: test what the 2015+ backfill buys. ~24 h.
#
# Jobs, in order (cheap CPU first, MPS training last):
#   1. weather deep backfill 2015+ (ERA5, network)
#   2. LGBM training-window sweep 365/730/1095/1460 (CPU)
#   3. crisis-regime backtest, 5-yr test incl. 2021-22 (CPU)
#   4. TFT-730 on the FULL 2-yr test, 3 seeds + ensemble (MPS)
#   5. PatchTST-730 same protocol (MPS)
#   6. collector -> reports/backtests/<date>_deep_history_campaign.md
#
# Launch:
#   nohup caffeinate -i bash run_deep_history_campaign.sh \
#     > data/logs/campaign_deep_history.log 2>&1 &
#
# Every job is resume-safe: rerun the script and finished work is
# skipped (backfill resumes, deep runner skips recorded seeds; the
# LGBM runs are cheap enough to redo).
set -uo pipefail
cd "$(dirname "$0")"
LOGD="data/logs"
mkdir -p "$LOGD"
STAMP="$(date +%F)"

run() {  # run <name> <cmd...>: log, continue on failure (collector still runs)
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name"
  if "$@" > "$LOGD/${STAMP}_${name}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOGD/${STAMP}_${name}.log)"
  fi
}

run weather_deep uv run python -u -m src.ingestion.backfill --only weather --start 2015-01-01

for DAYS in 365 730 1095 1460; do
  run "win${DAYS}" uv run python -u -m src.evaluation.run_price_backtest \
    --models price_naive_yesterday,lgbm_quantile \
    --test-start 2024-07-16 --train-days "$DAYS" \
    --tag "win${DAYS}" --by-period
done

run crisis5yr uv run python -u -m src.evaluation.run_price_backtest \
  --models price_naive_yesterday,price_naive_week,lear,lgbm_quantile \
  --test-start 2021-07-16 --train-days 365 --tag crisis5yr --by-period

run deep2yr_tft uv run python -u -m src.models.deep.run_deep2yr --model tft
run deep2yr_patchtst uv run python -u -m src.models.deep.run_deep2yr --model patchtst

run collect uv run python -u -m src.evaluation.collect_deep_history

echo "[$(date '+%F %T')] CAMPAIGN DONE"
