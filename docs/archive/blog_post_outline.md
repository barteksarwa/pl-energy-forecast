# Blog post outline — "I beat the Polish TSO's day-ahead load forecast"

Status: OUTLINE. Owner writes the final text in his own voice.
Target: personal blog / LinkedIn article / dev.to. ~1500 words, 5 charts.
Goal: recruiters and hiring managers in EU energy find it and read 3 minutes.

## Title options

1. "I beat the Polish TSO's day-ahead load forecast with ridge regression"
2. "2.13% vs 2.31%: beating PSE's load forecast with 25 features"
3. "What I learned building a production-grade energy forecast alone"

Pick 1 or 2. Numbers in titles get clicks and set the honest tone.

## Structure

### Hook (3 sentences)

Every morning at 09:00, Poland's grid operator publishes tomorrow's load
forecast. It is very good: 2.31% MAPE over two years. My model beats it —
2.08% — using ridge regression and 25 features.

### 1. The task (short)

- Day-ahead load forecasting: what, why, who pays (balancing market).
- The 09:00 D-1 cutoff. Everything after it is leakage.
- One chart: a forecast day, P10/P50/P90 band vs actual.

### 2. The honest baseline ladder (the credibility section)

- Table: naive 5.60% -> ridge 4.03% -> LSTM 3.67% -> LGBM 3.16% -> TSO 2.31%
  -> lgbm+TSO 2.14% -> ridge+TSO 2.13%. (2-yr walk-forward numbers:
  ridge_tso 2.08% vs TSO 2.23% — pick ONE table source and stay consistent:
  reports/backtests/2026-07-16_2yr_summary.csv)
- The twist: the TSO forecast is public at bid time. Using it as a FEATURE
  turns the task from "beat the expert" into "correct the expert".
- Key insight to spell out: the winning model is boring. Ridge. The value
  was in the setup: leakage-proof features, walk-forward evaluation,
  honest baselines.

### 3. Where deep learning landed (the differentiation section)

- 7 LSTM architectures + attention + TFT, all lost to ridge+TSO.
- Screening splits flattered nets by 0.6-0.9 pp — worth its own paragraph;
  this is a trap most tutorials never mention.
- Why: 25 tabular features, strong linear signal, 2 years of data.
  Nets need either more data or more structure to win here.
- This section signals seniority: knowing when NOT to use deep learning.

### 4. Production discipline (the hiring-manager section)

- Daily cron at 05:30 UTC: fetch, score yesterday, forecast tomorrow,
  write report, commit. Every report in git history = live track record.
- Shadow deployment: challenger runs N days scored-but-not-official
  before promotion. Same process real desks use.
- Gap logs, DST tests, corruption-proof cutoff tests.
- Link to repo. Charts: shadow tally, daily report screenshot.

### 5. What's next (one paragraph)

- Price forecasting on the same infrastructure (LEAR, TGE day-ahead).
- Honest close: "if your desk forecasts load or price, I'd love to talk".

## Charts to include

1. Forecast band vs actual, one good day (reports/figures/forecast_latest.png style)
2. Baseline ladder bar chart (make from 2yr summary CSV)
3. Rolling 30-day MAPE: ours vs TSO (make from backtest preds parquets)
4. SHAP summary (reports/sensitivity/shap_summary_lgbm_tso.png)
5. Screening-vs-walk-forward net comparison (the flattery trap)

## Rules for the final text

- Every number traces to a CSV in reports/backtests/. No rounding up.
- Say "beats" only where the 2-yr walk-forward says so.
- The TSO-as-feature caveat goes IN the post, prominently. Honesty is
  the brand.
- Short sentences. The owner's writing style, not marketing voice.
