# Hierarchical retail-demand forecast report

## Executive summary

The selected model is **HybridFourierEnsemble**. On the untouched 28-day holdout (2026-07-11 to 2026-08-07), bottom-level WAPE was **0.236**, versus **0.301** for last-value naive and **0.265** for weekly seasonal naive. This is a **21.6%** reduction in absolute error versus naive and **10.7%** versus seasonal naive.

Point forecasts are made for each Country–Region–Chain–Parent SKU series and then summed upward. They are therefore exactly coherent across Total, Country, Country–Region, Country–Region–Chain, and bottom levels.

## Project contract and assumptions

- Target: `Total_Qty_CTN` in cartons, at daily frequency.
- Prediction unit: one of 115 bottom series on one day.
- Assumed horizon and issuance cadence: 28 days, re-issued every 28 days. The business horizon and error-cost asymmetry were not supplied.
- Primary selection metric: bottom-level WAPE; MAE, RMSE, MASE, bias, and every hierarchy level are guardrails. Positive bias denotes overforecasting.
- No promotions, holidays, prices, stockouts, or planned events were supplied, so future event effects cannot be modeled explicitly.

## Data and time-series findings

- 83,950 rows cover 2024-08-08 through 2026-08-07 with no missing values or duplicate bottom-date keys.
- Hierarchy sizes: 1 total, 3 countries, 9 country–regions, 23 country–region–chains, and 115 bottom series.
- Zeros are 2.95% of observations, so demand is not strongly intermittent and Croston-style methods were not prioritized.
- Monday mean demand is 1.68 times the lowest weekday mean. Panel lag-7 correlation is 0.770.
- Weekly STL strength is 0.396 at the total and 0.267 for the median bottom series. Trend strengths are 0.277 and 0.242, respectively.
- Mean demand in the final 90 days is 14.9% above the first 90 days. The series also contains abrupt spikes, limiting models without future event flags.

## Validation and model selection

4 expanding-window validation folds were placed immediately before the untouched holdout. Each fold forecasts 28 consecutive days; every transformation and model is fit only on data available before that fold.

| Model | Bottom validation WAPE | Fold SD | Bottom validation bias |
|---|---:|---:|---:|
| HybridFourierEnsemble | 0.221 | 0.044 | -0.021 |
| RecentWeekdayMean4 | 0.244 | 0.023 | 0.038 |
| FourierTrendRidge | 0.252 | 0.047 | -0.123 |
| DirectHybridGB | 0.257 | 0.033 | 0.081 |
| SeasonalNaive7 | 0.318 | 0.118 | 0.039 |
| NaiveLast | 0.385 | 0.169 | 0.135 |

Selection used validation only. HybridFourierEnsemble had mean bottom WAPE 0.221. A fixed 50/50 average of DirectHybridGB and FourierTrendRidge, combining local nonlinear adaptation with smooth trend and seasonality.

## Final holdout evaluation

| Model | Level | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|
| NaiveLast | Total | 0.208 | 2378.14 | 3243.39 | 1.391 | 0.040 |
| NaiveLast | Country | 0.214 | 813.88 | 1134.52 | 1.332 | 0.040 |
| NaiveLast | Country_Region | 0.222 | 282.17 | 390.60 | 1.218 | 0.040 |
| NaiveLast | Country_Region_Chain | 0.239 | 119.00 | 182.15 | 1.148 | 0.040 |
| NaiveLast | Bottom | 0.301 | 29.97 | 47.65 | 1.034 | 0.040 |
| SeasonalNaive7 | Total | 0.117 | 1332.71 | 2660.30 | 0.779 | -0.076 |
| SeasonalNaive7 | Country | 0.127 | 482.40 | 937.19 | 0.781 | -0.076 |
| SeasonalNaive7 | Country_Region | 0.145 | 183.82 | 328.28 | 0.788 | -0.076 |
| SeasonalNaive7 | Country_Region_Chain | 0.174 | 86.30 | 157.16 | 0.849 | -0.076 |
| SeasonalNaive7 | Bottom | 0.265 | 26.32 | 46.10 | 0.919 | -0.076 |
| HybridFourierEnsemble | Total | 0.123 | 1405.57 | 2872.97 | 0.822 | -0.117 |
| HybridFourierEnsemble | Country | 0.128 | 487.49 | 1006.44 | 0.801 | -0.117 |
| HybridFourierEnsemble | Country_Region | 0.142 | 180.60 | 345.45 | 0.785 | -0.117 |
| HybridFourierEnsemble | Country_Region_Chain | 0.162 | 80.58 | 160.47 | 0.797 | -0.117 |
| HybridFourierEnsemble | Bottom | 0.236 | 23.49 | 42.75 | 0.809 | -0.117 |

WAPE is total absolute error divided by total actual demand. MASE scales each node's MAE by its in-training weekly-naive error. WAPE can hide node-level dispersion, so `holdout_node_metrics.csv` contains every hierarchy node.

The selected model underforecast the holdout by 11.7% at the bottom level. At the total level its WAPE was 0.123, slightly above the weekly-naive value of 0.117. This is the main hierarchy-level trade-off: the ensemble improved SKU-level allocation but did not beat weekly naive at the grand total in this holdout.

## Explanation, uncertainty, and limitations

A fixed 50/50 average of DirectHybridGB and FourierTrendRidge, combining local nonlinear adaptation with smooth trend and seasonality. The model's relationships are predictive associations, not causal effects.

The future forecast includes approximate 80% empirical intervals calibrated from absolute rolling-validation residuals by horizon and scaled by the square root of forecast demand. Bottom bounds are summed upward, so aggregate bounds are coherent but conservative. The median horizon calibration multiplier is 3.35.


For the boosting component, permutation importance on a reproducible training sample ranked the largest residual-error contributions as `seasonal_naive_7` (33.32), `mean_56` (21.36), `day_of_week` (4.71), `time_idx` (2.75), `month` (2.70). Values are increases in residual MAE when a feature is shuffled and should not be read as causal effects.

## Future 28-day forecast

For 2026-08-08 through 2026-09-04, coherent total demand is forecast at **286,401 cartons**, -10.5% versus the most recent 28 observed days. The highest forecast day is 2026-08-31 at 12,140 cartons. These values are conditional on normal calendar behavior because no future event schedule was available.

The largest limitation is missing future-known promotion, holiday, price, campaign, distribution, and stockout information. Unflagged shocks are irreducible from the provided columns, and only two annual cycles are available. The 28-day assumption must be revisited if purchasing decisions use another horizon.

## Operational recommendation

Run this as a 28-day batch forecast, refresh it whenever new daily actuals arrive, and use weekly seasonal naive as the automatic fallback. Monitor schema/key completeness, hierarchy mappings, zero rate, WAPE/MASE/bias by hierarchy level, and interval coverage after outcomes arrive. Retrain or investigate when four-week WAPE degrades materially from the backtest range or bias persists in one direction.

Before production, add event/price/stockout fields with explicit future availability, confirm the planning horizon and over-versus-underforecast cost, and validate the forecast in the inventory decision process rather than treating statistical error alone as business value.

## Interactive dashboard

[Open the Retail Demand Forecast Dashboard](retail_demand_dashboard.html)

The dashboard provides cascading Country, Region, Chain, and Parent SKU filters for exploring historical actuals, in-sample fitted estimates, and the 28-day out-of-sample forecast with approximate 80% intervals.

## Reproducibility

Run `python main.py`. The script records package versions, diagnostics, every fold, holdout metrics, feature importance when applicable, plots, and coherent future forecasts in `artifacts/`.
