# Strategic Direction — Get Hired vs. Sell It

Date: 2026-07-16. Honest advice, no padding.
Numbers below use the committed README table (2.13% vs TSO 2.31%).

## The blunt starting point

This repo is already a strong hiring asset. It is a weak business asset.
Reason: a **zone-level** PL load forecast has no paying customer.
PSE publishes one for free. ENTSO-E redistributes it. Beating it by 8% MAE
is great interview material and near-worthless as a standalone product.
What companies buy is **their portfolio's** forecast or **price** forecasts.
Keep that asymmetry in mind through everything below.

---

## PATH A — Get hired (3–6 months)

### What the repo already proves

- Beats the TSO benchmark honestly (walk-forward, cutoff-respecting, 12 months).
- Losses stay in the table. Deep nets lost to ridge. That candor is rare and hiring managers notice.
- Daily unattended ops with a git audit trail. Shadow/UAT promotion discipline.
- This is more operational maturity than most senior candidates show.

### What is missing for a real desk

1. **Price forecasting.** The best-paid jobs (Axpo, Shell, Orlen, Polenergia) are price desks.
   Load alone targets utilities only. Phase 2 (M6–M7) doubles the addressable jobs. Highest priority.
2. **A 30+ day live track record.** Backtests can be faked; committed daily reports cannot.
   Shadow tally started 2026-07-16 — this accrues for free, just do not break the cron.
3. **SQL evidence.** Every ad asks for it. One migration of parquet to DuckDB + a few
   analysis queries in a notebook is enough. Half a day.
4. **A public artifact.** A recruiter will not clone the repo. Write one blog post /
   LinkedIn article: "I beat the Polish TSO's day-ahead forecast — here's the honest table."
   An open PL benchmark is genuinely rare. This is the top-of-funnel.
5. **Interview fluency.** The owner must explain pinball loss, the 09:00 cutoff, why
   ridge beat the LSTM, and merit-order price drivers without notes. The learning notes
   thread exists — finish it, do not skip it.

### Fastest sequence

1. Now → +2 weeks: keep cron alive (track record accrues), do M6 price data + LEAR baseline.
2. +2 → +6 weeks: M7 LightGBM price model with SHAP drivers. M8 market-context docs.
3. +6 → +8 weeks: README final polish, blog post, DuckDB/SQL layer, model cards.
4. +8 weeks: start applying. Do not wait for "done". 30 daily reports exist by then.

### Cut / de-prioritize

- **Transformer challenger (M5 part 2).** LSTM already lost. A transformer will too, on
  2–3.5 years of hourly data. One paragraph explaining *why* is worth more than the model.
- Second EU zone. Nice-to-have, zero interview delta.
- Any web UI. A png in the daily report is enough.

### The 60-second pitch

"I run a day-ahead forecasting desk for the Polish power market — as a one-person
operation. Every morning it forecasts hourly load with P10/P50/P90, scores yesterday,
and explains its top drivers. Over a 12-month walk-forward backtest it beats the
Polish TSO's own forecast by 8% MAE. My deep-learning models are in the results
table too — they lost to a ridge combiner, and I can tell you exactly why. The whole
track record is timestamped in git. I built the pipeline, the models, the evaluation,
and the deployment discipline: shadow mode, promotion rules, unattended cron."

---

## PATH B — Standalone product

### Who would pay, and for what

Not for a PL zone load forecast (free from PSE). Realistic buyers:

- Small **energy retailers / balancing parties (POB)**: portfolio load forecasts.
  Needs *their* meter data. ~100+ licensed retailers in PL; maybe 20–30 small enough
  to lack in-house forecasting and big enough to pay.
- **RES operators / PPA parties**: generation + price forecasts for scheduling and settlement.
- **Industrial consumers** on spot-indexed contracts: price forecasts for load shifting.
- **Trading shops**: price forecasts — but they build in-house or buy from established
  vendors (Volue, Montel EQ, Dexter Energy, Amperon, Meteologica). Crowded field.

### Price and market size

- Forecast-feed SaaS in this space runs roughly €500–2,500/month per customer.
- Optimistic mature ceiling for a one-person PL-focused shop: ~10 customers ≈ €120k/year.
  That is year 2–3, after a sales grind. Not year 1.

### Legal / regulatory

- Selling forecasts needs **no URE license** and REMIT does not apply — REMIT governs
  trading and inside information, not analytics vendors. Barrier is not legal.
- Watch-outs: TGE price data redistribution licensing (ENTSO-E data is fine with
  attribution), liability disclaimers in contracts, and GDPR-ish handling of customer
  meter data.

### Realistic 12-month revenue

- Months 1–6: €0. Building the sellable thing (portfolio/price product) plus first outreach.
- Months 6–12: 0–2 pilot customers, likely discounted. €0–15k total. Consulting gigs
  ("build us a forecast model") are the more likely first money, €5–20k per project.
- Energy-sector sales cycles are 6–12 months. One person with no track record and no
  industry network starts at the back of that queue.

### Viable for one person?

Technically yes. Commercially, not yet. The missing ingredients — trust, references,
domain network, customer meter data — are exactly what a job provides.

---

## VERDICT

**Path A. Clearly. Do it now.**

- The repo is 70% of a hiring asset and 20% of a business asset.
- A PhD student with deep ML skills and a live, honest, operational forecasting desk
  is a strong hire *this quarter*. The same person selling forecasts cold is a weak
  vendor for at least a year.
- The job is not a detour from the business. It is the customer-discovery phase:
  you learn what desks actually pay for, and you meet the people who would buy.

**Can both run simultaneously?** Mostly no, with one exception.

- Do not build a product now. It would delay Phase 2 and the applications by months.
- The exception: take consulting-shaped side work if it lands in your lap. It is
  résumé fuel either way.
- One real conflict to check **before signing**: Polish energy-sector contracts often
  include non-compete and IP-assignment clauses. Negotiate a carve-out for this repo
  (it predates employment — keep the git history public) or Path B dies quietly.
- Revisit Path B in 12–18 months, employed, with a network and a public benchmark
  people already cite.

**This week:** keep the cron green, start M6, draft the blog post outline.
