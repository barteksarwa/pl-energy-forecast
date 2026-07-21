# Career — strategy, job market, interview prep

Merged from three notes (2026-07). Research dates: 2026-07-16.

---

## 1. Strategic direction

### Verdict: Path A — get hired (3–6 months)

The repo is 70% of a hiring asset, 20% of a business asset.
A zone-level PL load forecast has no paying customer. PSE publishes one free.
Beating it is great interview material, near-worthless as a product.
The job is not a detour. It is customer discovery: learn what desks pay for.

### What the repo already proves

- Beats the TSO benchmark honestly. Walk-forward, cutoff-respecting.
- Losses stay in the table. Deep nets lost to ridge. Hiring managers notice candor.
- Daily unattended ops with a git audit trail. Shadow/UAT promotion discipline.
- More operational maturity than most senior candidates show.

### Key decisions

- **Price forecasting was the top priority.** Now done (see interview answers below).
- **Live track record.** Shadow tally started 2026-07-16. Backtests can be faked; committed daily reports cannot. Keep the cron green.
- **SQL evidence.** One parquet-to-DuckDB migration + analysis queries. Half a day.
- **Public artifact.** One blog post: "I beat the Polish TSO's day-ahead forecast — the honest table." Recruiters will not clone the repo. This is the top-of-funnel.
- **Interview fluency.** Explain pinball loss, the 09:00 cutoff, why ridge beat the LSTM, merit-order drivers — without notes.
- **Cut:** transformer-as-flagship (nets lose at this data size; the explanation is worth more), second EU zone, any web UI.
- **Sequence:** price model → README polish, blog post, DuckDB → apply at +8 weeks. Do not wait for "done".

### Non-compete flag — check BEFORE signing

Polish energy-sector contracts often include non-compete and IP-assignment clauses.
Negotiate a carve-out for this repo. It predates employment; keep the git history public.
Otherwise Path B dies quietly.

### Path B (product) — parked, revisit in 12–18 months

- No buyer for a free-alternative zone forecast. Realistic buyers: small retailers/POBs (portfolio forecasts, need their meter data), RES/PPA operators, spot-indexed industrials.
- Trading shops build in-house or buy from Volue, Montel EQ, Dexter Energy, Amperon, Meteologica. Crowded.
- Forecast-feed SaaS runs ~€500–2,500/month per customer. One-person mature ceiling: ~10 customers ≈ €120k/year, in year 2–3.
- Realistic first 12 months: €0–15k. Consulting gigs (€5–20k/project) are the likelier first money.
- No URE license needed; REMIT does not apply to analytics vendors. Watch-outs: TGE data redistribution licensing, liability disclaimers, meter-data GDPR.
- Exception to "no Path B now": take consulting side work if it lands in your lap.

### The 60-second pitch

"I run a day-ahead forecasting desk for the Polish power market — as a one-person
operation. Every morning it forecasts hourly load and price with P10/P50/P90,
scores yesterday, and explains its top drivers. The load model beats the Polish
TSO's own forecast in a 2-year walk-forward. The price model beats the industry
LEAR baseline. My deep-learning models are in the results table too — they lost,
and I can tell you exactly why. The track record is timestamped in git. I built
the pipeline, models, evaluation, and deployment discipline: shadow mode,
promotion rules, unattended cron."

---

## 2. Job market — Poland & EU (2026)

Bottom line:
- Two lanes: **data-science forecasting** (utilities, TSOs) and **quant analysis for trading** (desks). Trading pays more, moves faster, cares about **price**, not load.
- Nobody asks for LSTMs by name. They ask for Python, time series, market understanding.
- Polish market knowledge (PSE, TGE, RB, rynek mocy) is a real differentiator locally.

### Role types

1. **Data Scientist / ML Engineer — Forecasting** (utility/TSO). ENGIE "Data Scientist Forecasting", talcom NL "Senior ML Engineer, Energy Forecasting", Vattenfall, Envelio "Energy Grid Forecast". Load, generation, renewables, grid congestion. Production ML.
2. **Quantitative Analyst — Power Trading.** InCommodities "Quantitative Power Analyst", Danske Commodities "Quantitative Analyst for Short-term Power Trading", Vattenfall Trading "Intraday Quantitative Analyst", The Mobility House "Quantitative Energy Analyst — Forecasting & Stochastic Optimization", Montel "(Senior) Quantitative Analyst for Power Market Modelling". Forecasts → signals → PnL.
3. **Power Market Analyst — fundamentals.** Shell "Power Market Analyst" (horizons up to 35 years), ENGIE "Quantitative Analyst SR". Fundamentals + economics + weather + ML.
4. **Market / Portfolio Analyst (Polish utility) — local entry lane.** "Analityk Rynku Energii/Gazu", "Analityk ds. Portfolio". Polenergia, PGE, Tauron, Orlen, Enea, ZE PAK. Often Excel/Power BI/SQL-first; Python "nice to have". Lower ML bar, easier entry. Polish required.

### Skills employers ask for

- **Python is mandatory** everywhere serious. Danske: "production-level Python code". Libraries named in ads: pandas/polars, numpy, scikit-learn, statsmodels; The Mobility House adds Nixtla, Darts, PyTorch, JAX, xarray.
- SQL expected; Git explicitly required; Excel still everywhere (esp. PL roles); R in some utility DS roles (ENGIE). Plus-points: Databricks, Spark, Docker/K8s (ML-engineer roles only).
- Ads ask for "time-series modeling, forecasting techniques, econometrics" (InCommodities) — not architectures.
- **LightGBM/XGBoost are the de-facto industry standard** (win GEFCom-style competitions). Ads rarely name them; interviewers expect them.
- **Deep learning is NOT the hiring bar.** Only dedicated ML-engineer roles name PyTorch/TensorFlow.
- Backtesting asked for by name (The Mobility House: "time-series backtesting").
- Domain: European market mechanics — Day-Ahead, Intraday/XBID, Ancillary/Balancing. Weather/NWP data (ECMWF, DWD, ERA5). Fundamentals/merit-order stack models valued at trading houses (InCommodities "fundamental stack model"; Montel "Power2Sim").
- Soft: explain results to non-technical stakeholders (named by Danske, ENGIE). English fluent; Polish for local utility roles.
- No certifications asked. Quantitative MSc/PhD wanted — the PhD is an asset, not a gap.

### Domain knowledge checklist

- Core: DAM (RDN), intraday (RDB/XBID), balancing + imbalance pricing, ancillary services, load/RES forecasting, weather-to-demand links.
- Poland-specific: PSE, TGE, URE, rynek mocy, RB reform, DSR, EU ETS/CO2 as price driver. (Definitions: interview answers below.)
- Senior/nice-to-have: stochastic optimization (MILP, MPC; Pyomo, CVXPY, Gurobi), BESS revenue stacking, flow-based coupling. Skip for now.

### Project fit

Covered and on-target: production Python + Git, ENTSO-E ingestion with TSO benchmark, population-weighted weather, LightGBM, P10/P50/P90 + pinball (ahead of most ads), honest baselines, SHAP + plain-language drivers, daily dry-run reports, DST/leakage rigor, PL zone focus, price forecasting (was gap #1, now done).

Remaining gaps, prioritized:
1. Intraday / imbalance angle — no intraday update or imbalance-price model.
2. Trading-signal / PnL framing — desks want forecast → signal → PnL, backtested economically, not just MAPE.
3. Fundamentals stack model — valued at trading houses, not required for a first role.
4. Cloud/MLOps polish — a Dockerfile would strengthen the ops story cheaply.

### Top 5 targets (ranked for a fast hire)

1. **Prop trading houses — Quantitative Power Analyst.** InCommodities, Danske Commodities (Aarhus/EU). Best pay, fastest interviews. Danske: "prior knowledge of the energy sector is not required."
2. **Utility trading desks.** Vattenfall (Katowice PL, ~500 staff — strong Poland-based option; also Hamburg/Amsterdam), Shell, ENGIE.
3. **Polish utilities — Market/Portfolio Analyst.** Polenergia, PGE, Tauron, Orlen, Enea. Reliable local fallback; the project over-qualifies on modeling.
4. **Utilities/grid-tech — DS Energy Forecasting.** ENGIE (Milan, ~39–53.5k EUR), talcom (NL, visa sponsorship), Envelio (DE/NL/ES remote).
5. **Vendors + TSO/ENTSO-E.** AleaSoft, ENTSO-E (Brussels, expanding "AI applications in electricity forecasting"), PSE. Mission fit, mid pay.

Play: lead with lanes 1–2; keep lane 3 as fallback.

### Sources (2026-07-16)

- Job ads: [InCommodities](https://incommodities.com/quantitative-power-analyst-eu) · [Danske Commodities](https://danskecommoditiesas.teamtailor.com/jobs/7960309-quantitative-analyst-for-short-term-power-trading) · [The Mobility House](https://www.remotely.de/job/the-mobility-house-quantitative-energy-analyst-forecasting-stochastic-optimizationmfd) · [Montel](https://montel.energy/careers/senior-quantitative-analyst-for-power-market-modelling) · [Vattenfall Intraday](https://careers.vattenfall.com/de/de/job/intraday-quantitative-analyst-trading-analyst-in-hamburg-jid-50031) · [Shell](https://shell.wd3.myworkdayjobs.com/en-US/ShellCareers/job/Power-Market-Analyst_R202611) · [ENGIE DS](https://jobs.engie.com/job/Data-Scientist-Forecasting/65707-en_US/) · [ENGIE Quant SR](https://jobs.engie.com/job/Quantitative-Analyst-SR-(Power-Markets-&-Forecasting)/67349-en_US) · [talcom](https://jaabz.com/jobs/227897-senior-ml-engineer-energy-forecasting-utrecht-netherlands) · [Envelio](https://www.climatetechlist.com/job/envelio-senior-data-scientist-ai-ml-engineer-energy-grid-forecast%C2%A0w-m-d-VCY0P13klPxaK4) · [Polenergia PDF](https://www.polenergia.pl/wp-content/uploads/2023/01/AnalitykRynkuGazuPolenergiaObrot.pdf) · [Polenergia Pracuj.pl](https://pracodawcy.pracuj.pl/profile/grupa-polenergia,naoxmaa,pl) · [AleaSoft](https://aleasoft.com/job-opportunities/) · [eFinancialCareers PL](https://www.efinancialcareers.com/jobs/energy-trader/in-poland)
- Context: [ENTSO-E guide](https://eujobs.co/career-guides/european-network-of-transmission-system-operators-for-electricity-career-guide) · [InCommodities guide](https://eujobs.co/career-guides/incommodities-a-s-career-guide) · [PSE market](https://www.pse.pl/web/pse-eng/areas-of-activity/electricity-market/general-information) · [TGE](https://tge.pl/about-tge) · [RB reform (BESS)](https://www.bess.energy/Balancing-Market-in-Poland.html) · [BESS certification PL](https://greenedge-solutions.com/en/balancing-market-and-bess-certification-in-poland/) · [RWE Poland / SAS](https://www.sas.com/en_us/customers/rwe-poland.html) · [ADL Poland Energy Outlook](https://www.adlittle.com/sites/default/files/reports/ADL%20Poland%20energy%20outlook%202025.pdf)
- Technical: [PL DAM ML (EPJ)](https://epj.min-pan.krakow.pl/Forecasting-electricity-prices-in-the-Polish-Day-Ahead-Market-using-machine-learning,207197,0,2.html) · [Probabilistic price forecasting review (arXiv)](https://arxiv.org/pdf/2511.05523) · [Smoothing QRA (arXiv)](https://arxiv.org/pdf/2302.00411) · [Probabilistic LightGBM load (AALTD 2023)](https://ecml-aaltd.github.io/aaltd2023/papers/Electricity%20Load%20and%20Peak%20Forecasting_%20Feature%20Engineering,%20Probabilistic%20LightGBM%20and%20Temporal%20Hierarchies.pdf)

---

## 3. Interview prep

Every answer is specific and backed by a number from this repo.
"I know this because I measured it" beats "I know this from a course."

### Project overview

**"Walk me through your project."**
A day-ahead forecasting desk for the Polish power market. Forecasts load and price, scores itself every morning, commits the report to git. Load beats PSE's forecast: 2.08% vs 2.23% MAPE, 2-year walk-forward. Price beats the LEAR standard: rMAE 0.638 vs 0.660. Runs unattended on GitHub Actions.

**"Why load AND price?"**
Load is the core skill: public data, no negatives, strong seasonality. But the highest-paying roles are trading desks, and desks care about price. Price runs on the same feature pipeline, backtest engine, and daily loop. That mirrors a real desk.

**"How long did it take?"**
A few weeks, in parallel with PhD work. Constraint: make it real work, not a tutorial — shadow deployment, walk-forward backtests, honest model cards.

### Technical — load

**"Why did ridge beat your deep learning models?"**
The TSO forecast explains 96% of skill (group ablation: removing it costs 1.97 pp MAPE). With that linear signal in, the residual is near-linear. Ridge handles it with 25 parameters; the LSTM's 100,000+ overfit. The screening split flatters nets by 0.6–0.9 pp vs honest walk-forward — I measured that trap. Same story on price: 60-trial HPO on TFT (1.27M params, 56-day context) hit 0.1157 screening pinball, then MAE 19.71 vs LEAR 18.23 in walk-forward — 8.1% worse. Root cause: monthly refits use 300–400 samples. LASSO (200 coefficients per hour) beats 1.27M params at that ratio. With 5 years of data the balance would shift.

**"What features?"**
25: load lags (24h, 48h, 168h, 7-day mean), calendar (hour, day-of-week, month, PL holidays, bridge days), weather (temperature, solar, wind — 10 cities, population-weighted), and the PSE day-ahead forecast. Composable from config in `src/features/`.

**"How did you prevent data leakage?"**
Cutoff: day D's forecast uses only data available 09:00 on D−1. A corruption test zeros all "future" values and asserts the feature matrix is unchanged. A DST test checks the 25-hour October day — "minus 24 hours" reached into the target day on the fall changeover. Real bug; the test caught it.

**"What is walk-forward cross-validation?"**
Train on 365 days, test on the next 30, shift forward. Repeat over 2 years of test data (17,450 hours). Respects time ordering, measures deployment performance. A random split would leak future patterns.

**"What metrics and why?"**
Load: MAPE (stakeholder-friendly, TSO standard), plus MAE and skill vs naive. Price: rMAE (MAE vs naive-1d) because price can be zero or negative — MAPE breaks. Plus pinball loss for quantiles and conformal coverage for the band.

### Technical — price

**"What is LEAR?"**
LASSO-Estimated AutoRegression (Ziel & Weron 2018). 24 LASSO regressions, one per delivery hour. Inputs: price lags (24h, 48h, 168h), load forecast, calendar. The standard benchmark. Ours: rMAE 0.660; our LightGBM challenger: 0.638.

**"What drives Polish day-ahead prices?"**
Merit order: the marginal plant sets the price; wind >4 GW cuts price 21% (measured). Demand: high load + low RES → gas marginal → high price. Fundamentals: TTF gas and EUA CO2 set the marginal unit's cost. Adding fuel proxies to LEAR cut winter bias from −15.9 to −4.6 EUR/MWh.

**"What is conformal calibration?"**
Raw quantile regression gave 51% coverage on a nominal 80% band. Conformal prediction: compute nonconformity scores on a calibration set, take the (1−α) quantile, add it as an offset. Rolling 90-day window, updated daily, adapts to regime shifts. After: LGBM 78.7%, LEAR 79.5%. Lives in `config/price_conformal.json`.

**"How do you handle price spikes?"**
Imperfectly, and I say so. Spike MAE (top 5% hours): 60 EUR/MWh for LGBM vs 77 naive — 22% better, but 3× pooled MAE. The ENTSO-E UMM outage feature was FLAT: aggregate unavailability is too coarse. Real spike forecasting needs unit-level outages and real-time gas. Documented gaps in the model card.

**"Why rMAE instead of MAPE for price?"**
798 hours in 3 years had negative prices (solar oversupply). MAPE divides by price — undefined or misleading near zero. rMAE is sign-agnostic with the same "better than guessing?" reading.

**"LGBM beats LEAR — why keep LEAR incumbent?"**
LGBM's 0.37 EUR/MWh edge may not survive the shadow window — backtests flatter models (measured on load). And swapping resets the daily track record, which is the product. The LGBM shadow gate opens after the TFT walk-forward clears the MPS queue.

**"Business value of 1 EUR/MWh MAE improvement?"**
A BRP with 100 MW flexible load optimises 100 MWh/hour. 1 EUR/MWh × 100 MWh × 8,760 h ≈ 876,000 EUR/year. Real value is lower — forecasts don't convert perfectly to bids — but the order of magnitude justifies the work.

**"What would CQR fix that your conformal doesn't?"**
I tested it. Hypothesis: negative prices cause lower-tail miscalibration. Implemented asymmetric CQR (separate per-tail offsets) on the same 2-year predictions. Symmetric: 79.6% (LEAR) / 78.9% (LGBM). Asymmetric: 79.1% / 78.4% — slightly worse. Surprise: q_hi > q_lo for both (LEAR 3.97 vs 2.12; LGBM 11.3 vs 7.2). The upper tail — spikes — is the real problem, not negative prices. Asymmetric gives 8–12% narrower bands (useful for position sizing) at 0.5 pp lower coverage. Symmetric stays in production; the asymmetric code is in `src/evaluation/conformal.py`.

**"Merit order as renewables grow?"**
Solar flattens midday (duck curve). Three effects: midday-evening spread widens (gas still sets the peak); negative prices multiply (87 hours in April 2026 alone); the marginal unit flips faster — a cloud over Silesia switches solar to gas in 15 minutes. Models trained on 2023 underestimate all three. The z-clip guard (±4 std) is a patch; the proper fix is online learning.

### Market structure

**"How does the Polish day-ahead market work?"**
Single Day-Ahead Coupling (SDAC) via TGE. Bids per hour for tomorrow. Gate closure 12:00 CET; results at 12:55. Our forecast lands at 09:00 D−1 — before a trader must commit positions.

**"What is the balancing market?"**
After gate closure, actuals deviate from schedule. PSE runs the rynek bilansujący in real time. Since June 2024: 15-minute settlement with scarcity pricing — balancing prices spike to a cap when the system is short. Good forecasts cut imbalance fees.

**"What is rynek mocy?"**
Poland's capacity market. Availability contracts auctioned N years ahead (currently 5). Capacity revenue is separate from energy price and can be 20–30% of a portfolio's total.

**"What are PSE, TGE, URE?"**
PSE (Polskie Sieci Elektroenergetyczne) — TSO and balancing authority. TGE (Towarowa Giełda Energii) — power exchange, spot (RDN) and intraday (RDB). URE (Urząd Regulacji Energetyki) — regulator.

### Behavioural / process

**"How do you know a model is production-ready?"**
Fourteen shadow days: the challenger forecasts every morning, gets scored, never published. After 14 valid days, compare mean daily metric vs the incumbent on a pre-agreed criterion. Win → promote and log in DECISIONS.md. Lose or tie → incumbent stays. Burden of proof is on the challenger.

**"Hardest bug?"**
Two. *Solar extrapolation:* LEAR extrapolated from training data with 40% less installed solar. 2026 solar forecasts were out of range; LEAR predicted 38,000 EUR/MWh. Fix: z-clip inputs at ±4 training std. *Offshore wind zero-variance:* Baltic offshore came online July 2026. `wind_off_fcst_mw` was all-zeros in training; z-score std clamped to 1e-6 turned 19 MW into 19,000,000 standardised. PatchTST (no sigmoid gating) gave val pinball 879 instead of ~0.14; TFT survived because its sigmoid gates saturate. Fix: zero out columns with training std < 1e-4, plus a unit test. Lesson: new RES types appear yearly; any clamp-min(eps) z-score pipeline without a zero-variance guard fails silently.

**"Why energy, coming from a different PhD field?"**
The skills transfer: time series, gradient boosting, deep learning, evaluation discipline. Energy is interesting because physics constrains the economics — it is not pure statistical arbitrage. And the impact is clear: better forecasts, lower balancing costs. I built this project to prove the transfer, not claim it.

### Questions to ask the interviewer

1. "What is your current MAPE on day-ahead load / price?"
2. "What features do you wish your production model had more of?" (Reveals what they want to improve.)
3. "How long from backtest to production? What is your shadow/UAT process?"
4. "Biggest error source — weather, market events, outages?"
5. "P10/P90 or just P50? How do you calibrate intervals?"
