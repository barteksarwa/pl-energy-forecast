# EPF resources — curated, July 2026

Curated for this repo. Every entry earns its line.
Repo context: LGBM-quantile, LEAR, CQR, CRPS ensemble (champion 17.34), zero-shot FMs (Chronos / TimesFM / Moirai), battery-arbitrage P&L, spike classifier.

## 1. Watch first

Ranked. Watch in this order.

1. **[Rafal Weron — Recent Advances in Electricity Price Forecasting](https://www.youtube.com/watch?v=fvz2JR5BDko)** (2023). The leading EPF group, spoken survey of probabilistic and distributional advances. Anchors the whole reading list and gives interview lines.
2. **[Jethro Browell — Probabilistic Energy Forecasting: Successes and Challenges](https://media.ed.ac.uk/media/Jethro+Browell,+(University+of+Glasgow)+Probabilistic+energy+forecastingA+successes+and+challenges/1_u6edef94)** (2022). Why quantiles beat points, and how to score them. Grounds the pinball/CRPS metrics this repo already uses.

## 2. Courses & tutorials

Ranked by payoff for this repo.

1. **[epftoolbox — docs](https://epftoolbox.readthedocs.io/en/latest/) + [GitHub](https://github.com/jeslago/epftoolbox)** (2021). The reference EPF toolkit: LEAR, DNN, benchmark datasets, Diebold-Mariano test. This repo already mirrors its workflow — use its DM test to prove wins.
2. **[A Gentle Introduction to Conformal Prediction — Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)** (2023). The definitive conformal tutorial. Theory backbone for the repo's CQR layer. Read the time-series section.
3. **[aangelopoulos/conformal-prediction notebooks](https://github.com/aangelopoulos/conformal-prediction)** (2024). Runnable CQR and weighted-conformal time-series notebooks. Theory to code in an afternoon.
4. **[MAPIE — time-series conformal tutorial (EnbPI + ACI)](https://mapie.readthedocs.io/en/v0.8.1/examples_regression/4-tutorials/plot_ts-tutorial.html)** (2024). Real electricity-demand data. Shows intervals that widen under changepoints. The production-grade method to add next.
5. **[MAPIE — CQR tutorial](https://mapie.readthedocs.io/en/v0.8.5/examples_regression/4-tutorials/plot_cqr_tutorial.html)** (2024). Direct code-level reference for the CQR calibration already shipped here.
6. **[Forecasting: Principles and Practice, 3rd ed — Hyndman & Athanasopoulos](https://otexts.com/fpp3/)** (2021). Free textbook. Read the seasonality, evaluation, and probabilistic chapters. Fills the foundations gap.
7. **[Forecasting for Data Scientists — Bergmeir & Hyndman](https://cbergmeir.com/talks/ffds-course/)** (2025). Free, current video course. Lecture 3.06 covers quantile regression, simulation, bootstrapping — exactly this repo's quantile stack.
8. **[Electricity Markets Crash Course — Mar Reguant](https://mreguant.github.io/em-course/)** (2024). Five days, code-first, day-ahead market economics in Python. Builds the market fluency PL/EU jobs expect.
9. **[Battery Storage Value Stacking Explained — FlexPower](https://flex-power.energy/energyblog/battery-storage-trading-strategy/)** (2024). Worked example: FCR + aFRR + day-ahead + intraday, ~90% revenue uplift, SOC and opportunity cost handled. Template for extending our single-market LP.
10. **[Energy arbitrage with linear programming — Steve Klosterman](https://www.steveklosterman.com/energy-arbitrage/)** (2020). Complete PuLP battery LP with a full-year backtest. Cleanest open template to sanity-check our arbitrage optimizer against.

## 3. Papers worth reading

One adoptable idea each. All passed verification; none excluded.

- **[Deep Learning for EPF: Day-Ahead, Intraday, Balancing — Yu, Bunn, Cremer et al.](https://arxiv.org/abs/2602.10071)** (2026). Most current review. **Idea:** adopt its backbone/head/loss taxonomy in our model cards and config — log each model as encoder + output head + loss, not as a monolith. Also flags intraday/balancing as under-studied: a PL balancing forecast would be novel work.
- **[Review of EPF Models in Day-Ahead, Intra-Day, Balancing Markets — O'Connor et al., Energies](https://www.mdpi.com/1996-1073/18/12/3097)** (2025). Peer-reviewed survey; balancing market focus. **Idea:** build forecast-error features — renewable forecast minus actual, plus net imbalance volume. Caveat: it reviews price forecasting; features transfer, benchmarks do not.
- **[Distributional neural networks for EPF — Marcjasz, Narajewski, Weron, Ziel](https://arxiv.org/abs/2207.02832)** (2023). A probability layer outputs Johnson's SU params; beats LEAR and DNN+QRA on German data, one forward pass, no QRA step. **Idea:** add a JSU parametric head and benchmark its CRPS against our ensemble and CQR.
- **[Online Multivariate Regularized Distributional Regression — Hirsch](https://arxiv.org/html/2504.02518v1)** (2025). Joint 24-dim price distribution, online LASSO via forget-factor Gramians; 80-400x faster than batch refits. Public code (ROLCH). **Idea:** wrap our LEAR with exponentially-discounted sufficient statistics and A/B it against periodic refits in the rolling backtest.
- **[Bridging Linear Models, Neural Networks and Online Learning — El Mahtout & Ziel](https://arxiv.org/abs/2601.02856)** (2026). Hybrid linear+NN with Bernstein Online Aggregation; 11-12% RMSE and 14-17% MAE cuts over six EU-market years. **Idea:** replace our ad-hoc rolling inverse-CRPS ensemble weights with BOA — online, adaptive, regret guarantees. Directly testable against champion 17.34.
- **[Conformal Prediction for Day-Ahead and Balancing EPF — O'Connor et al.](https://arxiv.org/abs/2502.04935)** (2025). EnbPI, SPCI, and a Q-Ens interval ensemble on the Irish market; scored by Winkler and battery-arbitrage profit. Tighter coverage → higher profit. **Idea:** add EnbPI/SPCI next to CQR and judge all interval methods by our battery P&L, not just pinball.
- **[Online conformalized NN ensembles — Brusaferri et al.](https://arxiv.org/abs/2404.02722)** (2024). Online conformal recalibration fixes hourly coverage failures of distributional ensembles. **Idea:** if our CQR misses coverage at specific hours, recalibrate per hour online — this is the recipe.
- **[Adaptive probabilistic forecasting of French spot prices — Dutot, Zaffran, Feron, Goude](https://arxiv.org/abs/2405.15359)** (2024). Conformal + online aggregation (OSSCP-horizon) stayed reliable through the 2021-22 price shock. **Idea:** the strongest single argument for adding online learning to our pipeline; test through our own high-volatility windows.
- **[Time Series Foundation Models for Belgian EPF](https://arxiv.org/abs/2605.17045)** (2026). Chronos-2 in ARX mode beats ML ensembles on day-ahead MAE but loses on imbalance. **Idea:** rerun our FM comparison with Chronos-2 ARX (covariates in-context). Note our Moirai covariate result was negative — this paper says the covariate mechanism matters, not covariates per se.
- **[Forecasting day-ahead electricity prices: review and open-access benchmark — Lago, Marcjasz, De Schutter, Weron](https://arxiv.org/abs/2008.08004)** (2021). The canonical EPF methodology paper behind epftoolbox. **Idea:** adopt its calibration-window discipline and DM significance testing for every "model X beats Y" claim in our results tables.
- **[Smoothing Quantile Regression Averaging — Uniejewski](https://arxiv.org/abs/2302.00411)** (2023). Smooths QRA quantiles; better tails; evaluated on reliability, sharpness, and trading profit. **Idea:** the direct alternative to spike classification — sharpen our extreme quantiles instead of classifying spikes, then compare both on P&L.
- **[Forecasting the Occurrence of Electricity Price Spikes — MDPI Forecasting](https://www.mdpi.com/2571-9394/6/1/7)** (2024). Open-access; static vs variable quantile-based spike thresholds. **Idea:** swap our spike classifier's static threshold for a rolling quantile threshold and re-score.
- **[Forecasting price spikes with decision trees — Fragkioudaki et al., IEEE EEM](https://ieeexplore.ieee.org/document/7216672/)** (2015). The foundational European spike-classification paper; every later classifier cites it. Historical anchor; full text is paywalled.
- **[Negative electricity prices: causes, impacts, responses — Sun et al.](https://www.sciencedirect.com/science/article/pii/S2096511726000071)** (2026). Comprehensive review of the negative-price regime. **Idea:** treat negative-price hours as their own regime in features and evaluation. Opens fine in a browser; blocks scrapers.
- **[Battery Storage in Continuous Intraday: Forecast vs Perfect Foresight](https://arxiv.org/abs/2501.07121)** (2025). A forecast-driven battery earns within 11% of perfect foresight. **Idea:** report our battery P&L as % of perfect foresight — the metric desks actually use to value a forecast.

## 4. PL/EU market must-knows

How the price we forecast gets made:

- The PL day-ahead price comes from pan-European coupling (SDAC). Primary source: [ENTSO-E SDAC page](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/).
- One algorithm, EUPHEMIA, clears bids across 26 countries. Plain-terms explainer by its builders: [N-SIDE](https://www.n-side.com/en/insights/en-the-single-day-ahead-coupling-sdac-and-the-pcr-euphemia-algorithm/).
- Beginner-friendly map of coupling, flow-based capacity (FBMC), and cross-border intraday (XBID): [Next Kraftwerke](https://www.next-kraftwerke.com/knowledge/market-coupling).
- The exact market we target: TGE's RDN — hourly, 15-minute, and block contracts. [TGE Day-Ahead Market](https://tge.pl/electricity-dam).

Polish specifics:

- June 2024 PSE balancing reform: 60-min → 15-min imbalance settlement, more volatility. Trader view: [Dexter Energy](https://dexterenergy.ai/news/polands-balancing-market-reform-what-short-term-power-traders-can-expect/). Regulator confirmation: [URE](https://www.ure.gov.pl/en/communication/news/382,Second-stage-of-the-Balancing-Market-reform-went-live-as-of-June-14.html).
- Rynek mocy, two views. Critique (~PLN 200bn through 2046): [Forum Energii](https://www.forum-energii.eu/en/capacity-at-any-cost). TSO's own description (auctions, DSR, BESS): [PSE](https://www.pse.pl/web/pse-eng/areas-of-activity/capacity-market/general-information).
- Solar curtailment and negative prices are reshaping PL price tails: ~590 GWh curtailed in 5 months of 2025, prices to -500 PLN/MWh. [pv magazine](https://www.pv-magazine.com/2025/07/09/solar-curtailment-on-the-rise-in-poland/). German case study of the same dynamic: [Timera Energy](https://timera-energy.com/blog/strong-growth-in-negative-prices-a-german-case-study/).

EU macro and the trading desk:

- The single best macro reference for interviews: 15-min MTU rollout, flow-based coupling, weather-driven volatility. [ACER 2025 Monitoring Report (PDF)](https://www.acer.europa.eu/sites/default/files/documents/Publications/2025_ACER_Gas_Electricity_Key_Developments.pdf).
- How desks use forecasts and value batteries: four revenue channels mapped in [Montel](https://montel.energy/resources/blog/how-energy-traders-unlock-value-from-batteries-in-power-markets); concrete BESS revenue numbers (intraday spreads ~45% above day-ahead, ~96% uplift from stacking) in [Modo Energy](https://modoenergy.com/research/intraday-day-ahead-power-trading-revenue-spread-churn-battery-energy-storage).
- Where the desk job is heading (15-min MTUs, thin margins, forecasts as core inputs): [Dexter Energy 2026 report](https://dexterenergy.ai/news/short-term-power-trading-in-the-energy-transition/). Dexter sells forecasting — read as informed but partial.

## 5. Ideas worth testing in this repo

Ranked by expected payoff per effort. Each maps to something that already exists.

1. **BOA ensemble weights.** Replace rolling inverse-CRPS weights with Bernstein Online Aggregation. Same inputs, principled combiner. Target to beat: champion CRPS 17.34. (El Mahtout & Ziel.)
2. **Score intervals by battery P&L.** We already have CQR and the arbitrage LP. Add EnbPI/SPCI via MAPIE, then rank all interval methods by trading profit, not just pinball. (O'Connor 2025.)
3. **JSU distributional head.** One cheap parametric baseline that targets skew and fat tails. Benchmark CRPS vs the ensemble. (Marcjasz et al.)
4. **Online LEAR.** Forget-factor sufficient statistics instead of periodic refits. A/B in the rolling backtest. (Hirsch.)
5. **Hourly online conformal recalibration.** Only if CQR coverage fails at specific hours — check that first. (Brusaferri et al.)
6. **Chronos-2 ARX rerun.** Our Moirai covariate run was negative. The Belgium paper says Chronos-2's in-context covariate mode is the one that works. Cheap to test with the existing FM harness. (arXiv 2605.17045.)
7. **Perfect-foresight ratio.** Report battery P&L as % of perfect foresight in the daily report. One metric change, big credibility gain. (arXiv 2501.07121.)
8. **Rolling-quantile spike threshold.** Swap the spike classifier's static threshold for a variable one; also compare against sharpened extreme quantiles (SQRA) on P&L. (MDPI 2024; Uniejewski.)
9. **Imbalance and renewable-error features.** Add wind/solar forecast error and net imbalance volume to the feature config. Config-only change by design. (Energies 2025 review.)
10. **Backbone/head/loss model cards.** Documentation convention, zero modeling risk, makes ablations legible. (Yu et al. review.)
11. **PL balancing-market forecast.** The reviews agree this is the under-studied gap. A 15-min PL imbalance price model would be genuinely novel portfolio work. Biggest effort, biggest differentiation.
12. **Multi-market LP.** Extend the single-market arbitrage LP toward stacking, following the FlexPower template. Do after 2 and 7.

## 6. Skills gap vs 2025-26 job postings

Benchmark: the live [Montel quant analyst posting](https://montel.energy/careers/senior-quantitative-analyst-for-power-market-modelling) plus the desk-facing sources above.

Covered by this repo already:

- Python/pandas, probabilistic forecasting, honest backtesting, LEAR/GBM/FM models, conformal intervals, a P&L link.

Gaps to close:

- **SQL/databases.** Postings ask for it; the repo stores flat files. Move data to DuckDB or Postgres and say so in the README.
- **Market fundamentals fluency.** Be able to explain SDAC, EUPHEMIA, FBMC, and the PSE 15-min reform in plain words. Section 4 is the syllabus.
- **Intraday and balancing markets.** All current work is day-ahead. Idea 11 closes this.
- **15-minute granularity.** MTU is going to 15 min EU-wide (ACER). Our pipeline is hourly. Check what breaks.
- **Online learning.** Desks retrain continuously; we refit on a schedule. Ideas 1 and 4 close this.
- **Statistical significance discipline.** Add DM tests to every model-comparison table. epftoolbox ships them.
- **OOP and code structure.** Postings mention it explicitly. The repo's modular config design is the story — tell it in the README.