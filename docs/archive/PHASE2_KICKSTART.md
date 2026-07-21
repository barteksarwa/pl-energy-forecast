# Phase 2 kickstart — price + portfolio simulation

Prepared overnight 2026-07-15 so the morning starts with decisions, not research.

## Data: verified tonight, all keyless (PSE API v2, from 2024-06-14)

| Entity | What | Phase 2 role |
|---|---|---|
| `csdac-pln` | day-ahead SDAC price, 15-min, PLN/MWh, publishes ~15:30 D-1 | **the target** |
| `crb-rozl` | balancing settlement prices (CEN, CKOEB, CEB) | imbalance costing |
| `rce-pln` | market energy price (prosumer settlement) | PV portfolio valuation |
| `cor` | operational reserve cost | context feature |
| `his-wlk-cal` | generation by type incl. `pv`, `wi` (wind) | fundamentals features |
| `kse-load` + our forecasts | demand side | fundamentals features |

ENTSO-E token still adds: pre-2024 history, cross-border flows, outages, neighbors.

## Owner's portfolio question — the answer

Real DSO/retailer-level load data for PL is not public; that is normal
(portfolio data = commercial secret everywhere). The industry-standard POC:

**Synthetic portfolio, real physics, real prices.**
1. Portfolio load = scaled national shape + noise (or a household-profile mix).
2. Behind-meter PV = our `shortwave_radiation` × a simple performance model
   (capacity, tilt factor, temperature derating).
3. Wind = our `wind_speed_10m` → standard turbine power curve (cut-in 3,
   rated 12, cut-out 25 m/s).
4. Net load = load − PV − wind. Forecast it with the exact same pipeline
   (models are input-agnostic by design — the modular-inputs rule pays off).
5. Value forecast errors at real prices: buy day-ahead at `csdac-pln`,
   settle deviations at `crb-rozl`. Result in PLN/year — the number a
   retailer actually optimizes.

This turns the repo from "forecasts load" into "simulates a desk P&L".
Strong interview artifact; publishable angle (net-load forecasting under
prosumer growth is hot).

## Proposed milestone order (owner approves in the morning)

1. M6a: price data backfill (`csdac-pln`, `crb-rozl`, `his-wlk-cal`) + viz.
2. M6b: price baselines — naive, LEAR on fundamentals. Spike-aware metrics
   (no MAPE for prices; MAE + pinball + tail coverage).
3. M6c: portfolio POC (above) with an imbalance-cost report in PLN.
4. Then LightGBM/nets on price, same walk-forward engine.

## Open questions for the owner

- Portfolio composition for the POC: how much PV/wind vs load? (Suggest:
  500 MW peak load retailer, 150 MWp PV, 100 MW wind — typical mid-size.)
- Price target: hourly average or native 15-min? (Suggest hourly first.)
