# Energy Forecasting Job Market — Poland & EU (2026)

Research date: 2026-07-16. For a PhD student with an ML background aiming to get hired fast at a Polish or EU energy company.

Bottom line up front:
- The market splits into two lanes: **data-science forecasting** (utilities, TSOs) and **quant analysis for trading** (trading houses, trading desks).
- Trading pays more and moves faster. It wants Python plus power-market mechanics, not deep learning wizardry.
- Nobody asks for LSTMs or transformers by name. They ask for Python, time series, and market understanding.
- Poland-specific market knowledge (PSE, TGE, RB, rynek mocy) is a real differentiator for local roles.

---

## Role types found

**1. Data Scientist / ML Engineer — Forecasting (utility / TSO)**
- Titles: "Data Scientist Forecasting" (ENGIE), "Senior ML Engineer, Energy Forecasting" (talcom NL), "Data Scientist – Energy Forecasting" (Vattenfall), "Data Scientist / AI/ML Engineer – Energy Grid Forecast" (Envelio).
- Focus: load, generation, renewable, and grid-congestion forecasts. Production ML systems.
- Employers: utilities (ENGIE, Vattenfall, RWE), grid-tech firms, forecasting vendors (AleaSoft).

**2. Quantitative Analyst — Power Trading (trading house / desk)**
- Titles: "Quantitative Power Analyst" (InCommodities), "Quantitative Analyst for Short-term Power Trading" (Danske Commodities), "Intraday Quantitative Analyst" (Vattenfall Trading), "Quantitative Energy Analyst – Forecasting & Stochastic Optimization" (The Mobility House), "(Senior) Quantitative Analyst for Power Market Modelling" (Montel).
- Focus: turn forecasts into trading signals with real PnL. Day-ahead + intraday.
- Employers: proprietary trading houses (InCommodities, Danske Commodities), utility trading desks (Vattenfall, Shell, ENGIE).

**3. Power Market Analyst — Fundamentals (major / integrated)**
- Titles: "Power Market Analyst" (Shell), "Quantitative Analyst SR (Power Markets & Forecasting)" (ENGIE).
- Focus: price forecasts from day-ahead to long-term (Shell: up to 35 years). Blends fundamentals, economics, weather, ML.
- Employers: oil-and-gas majors, large utilities.

**4. Market / Portfolio Analyst (Polish utility) — the local entry lane**
- Titles: "Analityk Rynku Energii/Gazu", "Analityk ds. Portfolio", "Analityk Biznesowy" (Polenergia and peers).
- Focus: monitor and forecast electricity/gas/CO2 markets. Fundamental + technical analysis.
- Employers: Polenergia, PGE, Tauron, Orlen, Enea, ZE PAK, trading arms.
- Note: often Excel/Power BI/SQL-first, Python listed as "nice to have". Lower ML bar, easier entry point.

---

## Skill requirements by role

### Programming (near-universal)
- **Python is mandatory** across every serious forecasting/quant role. Danske Commodities: "solid coding skills with the ability to write production-level Python code."
- Common libraries named in ads: **pandas / polars, numpy, scikit-learn, statsmodels** (The Mobility House also names **Nixtla, Darts, PyTorch, JAX, xarray**).
- **R** appears in utility data-science roles (ENGIE: "Proficient programming skills in Python and R").
- **SQL** expected in most utility/analyst roles. **Git / version control** explicitly required.
- **Excel** still listed everywhere, especially Polish and fundamentals roles.
- Cloud / data platforms as a plus: Databricks, Spark (ENGIE), Docker/Kubernetes (ML-engineer roles).

### ML / modeling
- Ads ask for **"time-series modeling, forecasting techniques, and econometrics"** (InCommodities) — not specific fancy architectures.
- **LightGBM / XGBoost are the de-facto industry standard** for day-ahead price and load forecasting. Confirmed by the academic/practitioner literature (quantile LightGBM/XGBoost win competitions like GEFCom). Job ads rarely name them but interviewers expect them.
- **Deep learning is NOT a common requirement.** Only pure ML-engineer roles (talcom NL) name PyTorch/TensorFlow. Most quant/analyst ads do not. Polish academic work uses CNN+LSTM for the DAM, but that is research, not the hiring bar.
- Backtesting is asked for by name (The Mobility House: "time-series backtesting").

### Domain
- **European power market mechanics** repeatedly required: The Mobility House needs "understanding of physical and financial European power market mechanics (Day-Ahead, Intraday/XBID, and Ancillary Services/Balancing)."
- **Weather / NWP data** is a strong theme for renewables and short-term price. Nice-to-haves: **ECMWF, DWD, ERA5** NWP data (The Mobility House). InCommodities: transform "price signals, weather forecasts, fundamentals" into forecasts.
- Fundamentals modeling (supply/demand stack) valued at trading houses and majors (InCommodities "fundamental stack model"; Montel "Power2Sim").

### Soft skills (called out explicitly)
- Communicating technical results to non-technical stakeholders (Danske Commodities, ENGIE).
- English fluent everywhere. Polish required for local utility roles.

---

## Domain knowledge checklist

What employers expect you to understand. Grouped by priority for the EU quant/forecasting track.

**Core (must know):**
- Day-Ahead Market (DAM / RDN in Poland) — auction, gate closure, hourly (now 15-min) products.
- Intraday market (IDM / XBID / RDB) — continuous trading, why forecasts get updated.
- Balancing / imbalance market and imbalance pricing — Poland's RB reform brought 15-min settlement + scarcity pricing (June 2024).
- Ancillary services / reserves.
- Load, generation, and renewable (wind/solar) forecasting fundamentals.
- Weather-to-demand and weather-to-generation links (NWP data).

**Poland-specific (differentiator for local roles):**
- **PSE** — the TSO; source of actual load and the TSO day-ahead load forecast (our benchmark).
- **TGE (Towarowa Giełda Energii)** — the Polish power exchange; RDN and RDB products.
- **URE** — the regulator.
- **Rynek mocy (capacity market)** — capacity payments, auctions.
- **RB (rynek bilansujący / balancing market)** and its 2024 reform (15-min intervals, scarcity pricing).
- **DSR (demand-side response)** — growing flexibility market.
- CO2 / EU ETS prices as a price driver.

**Nice-to-have / senior:**
- Stochastic optimization: MILP, MPC, stochastic dynamic programming (Pyomo, CVXPY, Gurobi) — The Mobility House.
- Battery / BESS optimization and revenue stacking.
- Cross-border flows, flow-based market coupling.

---

## Gaps our project already covers

Strong alignment. The project hits most of the real hiring bar.

- **Python-first, production-style, tested, typed code + Git** — matches "production-level Python" ask. Strong.
- **ENTSO-E data ingestion** (actual load + TSO forecast) — real market data experience. TSO forecast as benchmark = exactly what desks do.
- **Open-Meteo weather, population-weighted over cities** — matches the NWP/weather-to-demand theme.
- **LightGBM forecasting** — the industry-standard model. Directly on target.
- **Probabilistic P10/P50/P90 + pinball loss** — ahead of most ads. Probabilistic forecasting is where the research and best desks are heading (imbalance/price forecasting literature is all probabilistic).
- **Baselines: seasonal naive + TSO forecast, honest comparison** — shows evaluation rigor employers want.
- **SHAP / interpretability + plain-language drivers** — a differentiator; explains models to non-technical managers (a named soft skill).
- **Daily dry-run + committed reports** — simulates the real morning forecaster workflow. Strong portfolio signal.
- **DST / UTC handling, no-leakage cutoff at D-1 09:00** — the exact rigor a load-forecasting interviewer probes.
- **Poland zone (PSE) focus** — targets the local market directly.

---

## Gaps our project does NOT cover yet

Where to extend if targeting specific lanes. Prioritized.

1. **Price forecasting (DAM / IDM), not just load.** Biggest gap. Trading-desk and quant roles care about **price** (RDN/RDB on TGE), imbalance price, and spreads far more than pure load. Load forecasting is a means to a price/PnL end. Adding a day-ahead price forecast for the PL zone (TGE fixing) would open the whole quant-trading lane. See note below — price is more in demand than load for the high-paying roles.
2. **Intraday / imbalance angle.** No intraday forecast update or imbalance-price model. XBID/RB mechanics are named in top ads.
3. **Trading signal / PnL framing.** Desks want "forecast -> signal -> PnL", backtested economically (not just MAPE). A simple strategy backtest (e.g., battery arbitrage or position on forecast error) would speak their language.
4. **Deep-learning benchmark.** The CLAUDE.md plan lists LSTM/transformer but they are not the hiring bar. Low priority for jobs; useful only to show breadth and because the owner already knows them. Frame honestly vs LightGBM.
5. **Fundamentals / merit-order stack model.** Trading houses (InCommodities, Montel) value a supply/demand fundamental model. Not required to land a first role.
6. **Stochastic optimization / BESS.** MILP/MPC dispatch. Senior/nice-to-have. Skip for now.
7. **Cloud/MLOps polish.** Databricks/Spark/Docker/K8s appear in ML-engineer ads. `make` + GitHub Actions cron partly covers the "ops" story; a Dockerfile would strengthen it cheaply.

---

## Specific questions answered

- **ENTSO-E experience wanted?** Not named as a keyword in most ads, but it IS the standard European load/generation data source, and TSO/utility roles assume familiarity. Having hands-on ENTSO-E ingestion is a concrete, credible plus. ENTSO-E itself hires and expects "AI applications in electricity forecasting."
- **Probabilistic or point forecasts?** Point forecasts still dominate basic ads. But probabilistic/quantile forecasting is the clear direction of the field (imbalance and price forecasting research is overwhelmingly probabilistic; quantile regression averaging, QRA/QLGBM). Shipping P10/P50/P90 puts the project ahead of the median candidate.
- **LightGBM/XGBoost standard, or deep learning expected?** **LightGBM/XGBoost are the standard.** They win the day-ahead price/load forecasting competitions and are fast and robust on tabular energy features. Deep learning is expected only in dedicated ML-engineer roles (PyTorch/TensorFlow). Most quant/forecasting-analyst ads do not require it.
- **SHAP / interpretability wanted?** Rarely a hard requirement in ads, but "explain results to non-technical stakeholders" is a repeated soft requirement. SHAP + plain-language drivers is a genuine differentiator, not table stakes.
- **Price forecasting (DAM/IDM) more in demand than load?** **Yes, for the higher-paying trading/quant lane.** Trading desks live on price, spreads, and imbalance. Load forecasting demand is concentrated in utilities/TSOs (still real, more stable, often lower paid). To go where the money and volume are, add a price forecast.
- **Certifications / domain courses mentioned?** Essentially none. Ads ask for a quantitative MSc or PhD (Math, Physics, Stats, Econometrics, Data Science, Engineering) — which the owner has. No named certificates. Domain knowledge is expected to be learned on the job or self-taught. A PhD is an asset here, not a gap.
- **Poland-specific: PSE / TGE / balancing valued?** **Yes for local roles.** Polish ads list "knowledge of Polish market design and players (PSE, URE, TGE)" as a plus. Balancing-market reform (RB, 15-min, scarcity pricing) and rynek mocy are live 2026 topics. Note: "TOMOZE" does not appear as a standard term — likely a typo; the real entities are PSE, TGE, URE, and the RB/rynek mocy mechanisms.

---

## Top 5 target employers / role types

Ranked for a fast hire given ML PhD + this portfolio project.

1. **Proprietary power trading houses — Quantitative Power Analyst.**
   InCommodities (Aarhus/EU), Danske Commodities (Aarhus). Want Python + time-series + power/weather/market data -> signals. No DL required. Domain teachable. Best pay, fastest interviews, most aligned with the project once price forecasting is added. "Prior knowledge of the energy sector is not required" (Danske) — PhD quant background is enough to get in.

2. **Utility trading desks — Intraday / Quantitative Analyst.**
   Vattenfall (Katowice PL ~500 staff, and Hamburg/Amsterdam), Shell (Power Market Analyst), ENGIE. Blend of fundamentals, ML, weather. Katowice makes Vattenfall a strong Poland-based option.

3. **Polish utilities — Market / Portfolio / Forecasting Analyst.**
   Polenergia, PGE, Tauron, Orlen, Enea. Easiest local entry lane. Lower ML bar (Excel/Power BI/SQL/Python), Polish required. Good for a first PL role; the project heavily over-qualifies on the modeling side.

4. **Utilities / grid-tech — Data Scientist, Energy Forecasting.**
   ENGIE (Milan, ~39-53.5k EUR), talcom (NL, visa sponsorship), Envelio (DE/NL/ES remote). Load/generation/grid forecasting in production. Matches the project's data-science core directly.

5. **Forecasting vendors + TSO/ENTSO-E.**
   AleaSoft (forecasting SaaS), ENTSO-E (Brussels), PSE. Pure forecasting focus. ENTSO-E is expanding in "AI applications in electricity forecasting." Good mission fit, mid pay.

**Suggested play:** lead with lane 1/2 (quant, best ROI on the project) after adding a TGE day-ahead price forecast; keep lane 3 as the reliable local fallback.

---

## Sources

Job ads and postings:
- [InCommodities — Quantitative Power Analyst, EU](https://incommodities.com/quantitative-power-analyst-eu)
- [Danske Commodities — Quantitative Analyst for Short-term Power Trading](https://danskecommoditiesas.teamtailor.com/jobs/7960309-quantitative-analyst-for-short-term-power-trading)
- [The Mobility House — Quantitative Energy Analyst / Forecasting & Stochastic Optimization](https://www.remotely.de/job/the-mobility-house-quantitative-energy-analyst-forecasting-stochastic-optimizationmfd)
- [Montel — (Senior) Quantitative Analyst for Power Market Modelling](https://montel.energy/careers/senior-quantitative-analyst-for-power-market-modelling)
- [Vattenfall — Intraday Quantitative Analyst (Hamburg)](https://careers.vattenfall.com/de/de/job/intraday-quantitative-analyst-trading-analyst-in-hamburg-jid-50031)
- [Shell — Power Market Analyst](https://shell.wd3.myworkdayjobs.com/en-US/ShellCareers/job/Power-Market-Analyst_R202611)
- [ENGIE — Data Scientist Forecasting](https://jobs.engie.com/job/Data-Scientist-Forecasting/65707-en_US/)
- [ENGIE — Quantitative Analyst SR (Power Markets & Forecasting)](https://jobs.engie.com/job/Quantitative-Analyst-SR-(Power-Markets-&-Forecasting)/67349-en_US)
- [talcom — Senior ML Engineer, Energy Forecasting (NL)](https://jaabz.com/jobs/227897-senior-ml-engineer-energy-forecasting-utrecht-netherlands)
- [Envelio — Senior Data Scientist / AI-ML Engineer, Energy Grid Forecast](https://www.climatetechlist.com/job/envelio-senior-data-scientist-ai-ml-engineer-energy-grid-forecast%C2%A0w-m-d-VCY0P13klPxaK4)
- [Polenergia — Analityk Rynku Gazu (PDF)](https://www.polenergia.pl/wp-content/uploads/2023/01/AnalitykRynkuGazuPolenergiaObrot.pdf)
- [Polenergia jobs — Pracuj.pl profile](https://pracodawcy.pracuj.pl/profile/grupa-polenergia,naoxmaa,pl)
- [AleaSoft Energy Forecasting — Job Opportunities](https://aleasoft.com/job-opportunities/)
- [Energy Trader jobs in Poland — eFinancialCareers](https://www.efinancialcareers.com/jobs/energy-trader/in-poland)

Career guides and market context:
- [ENTSO-E career guide — EUJobs.co](https://eujobs.co/career-guides/european-network-of-transmission-system-operators-for-electricity-career-guide)
- [InCommodities career guide — EUJobs.co](https://eujobs.co/career-guides/incommodities-a-s-career-guide)
- [PSE — electricity market, general information](https://www.pse.pl/web/pse-eng/areas-of-activity/electricity-market/general-information)
- [TGE — About TGE](https://tge.pl/about-tge)
- [BESS — Balancing Market in Poland (RB reform, 15-min)](https://www.bess.energy/Balancing-Market-in-Poland.html)
- [Green Edge Solutions — Balancing Market and BESS Certification in Poland](https://greenedge-solutions.com/en/balancing-market-and-bess-certification-in-poland/)
- [SAS — RWE Poland forecasting case study](https://www.sas.com/en_us/customers/rwe-poland.html)
- [Arthur D. Little — Poland Energy Outlook 2026 & Beyond (PDF)](https://www.adlittle.com/sites/default/files/reports/ADL%20Poland%20energy%20outlook%202025.pdf)

Technical / academic (models used in the field):
- [Forecasting electricity prices in the Polish Day-Ahead Market using ML (EPJ)](https://epj.min-pan.krakow.pl/Forecasting-electricity-prices-in-the-Polish-Day-Ahead-Market-using-machine-learning,207197,0,2.html)
- [Probabilistic Price Forecasting: DAM, Intra-Day, Balancing — review (arXiv)](https://arxiv.org/pdf/2511.05523)
- [Smoothing Quantile Regression Averaging for electricity prices (arXiv)](https://arxiv.org/pdf/2302.00411)
- [Feature Engineering, Probabilistic LightGBM and Temporal Hierarchies — load forecasting (AALTD 2023)](https://ecml-aaltd.github.io/aaltd2023/papers/Electricity%20Load%20and%20Peak%20Forecasting_%20Feature%20Engineering,%20Probabilistic%20LightGBM%20and%20Temporal%20Hierarchies.pdf)
