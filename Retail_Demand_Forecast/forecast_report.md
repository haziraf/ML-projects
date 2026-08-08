# Hierarchical retail-demand forecast report

## Executive summary

The selected model is **HybridFourierEnsemble**. On the untouched 112-day holdout (2026-04-18 to 2026-08-07), bottom-level WAPE was **0.217**, versus **0.598** for last-value naive and **0.462** for weekly seasonal naive. This is a **63.7%** reduction in absolute error versus naive and **53.1%** versus seasonal naive.

Point forecasts are made for each Country–Region–Chain–Parent SKU series and then summed upward. They are therefore exactly coherent across Total, Country, Country–Region, Country–Region–Chain, and bottom levels.

## Project contract and assumptions

- Target: `Total_Qty_CTN` in cartons, at daily frequency.
- Prediction unit: one of 115 bottom series on one day.
- Assumed horizon and issuance cadence: 112 days, re-issued every 28 days. The business horizon and error-cost asymmetry were not supplied.
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

2 expanding-window validation folds were placed immediately before the untouched holdout. Each fold forecasts 28 consecutive days; every transformation and model is fit only on data available before that fold.

| Model | Bottom validation WAPE | Fold SD | Bottom validation bias |
|---|---:|---:|---:|
| HybridFourierEnsemble | 0.232 | 0.008 | -0.063 |
| DirectHybridGB | 0.259 | 0.014 | -0.042 |
| RecentWeekdayMean4 | 0.267 | 0.005 | -0.068 |
| FourierTrendRidge | 0.273 | 0.012 | -0.085 |
| SeasonalNaive7 | 0.278 | 0.006 | -0.104 |
| NaiveLast | 0.315 | 0.023 | -0.037 |

Selection used validation only. HybridFourierEnsemble had mean bottom WAPE 0.232. A fixed 50/50 average of DirectHybridGB and FourierTrendRidge, combining local nonlinear adaptation with smooth trend and seasonality.

## Final holdout evaluation

| Model | Level | WAPE | MAE | RMSE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|
| NaiveLast | Total | 0.558 | 6294.12 | 6646.88 | 3.698 | 0.537 |
| NaiveLast | Country | 0.559 | 2101.12 | 2402.63 | 3.316 | 0.537 |
| NaiveLast | Country_Region | 0.561 | 703.27 | 807.33 | 2.926 | 0.537 |
| NaiveLast | Country_Region_Chain | 0.565 | 277.16 | 353.55 | 2.565 | 0.537 |
| NaiveLast | Bottom | 0.598 | 58.72 | 79.29 | 2.031 | 0.537 |
| SeasonalNaive7 | Total | 0.386 | 4358.34 | 4824.12 | 2.561 | 0.324 |
| SeasonalNaive7 | Country | 0.387 | 1457.12 | 1714.59 | 2.359 | 0.324 |
| SeasonalNaive7 | Country_Region | 0.394 | 493.39 | 582.83 | 2.108 | 0.324 |
| SeasonalNaive7 | Country_Region_Chain | 0.400 | 196.44 | 260.98 | 1.890 | 0.324 |
| SeasonalNaive7 | Bottom | 0.462 | 45.36 | 65.19 | 1.562 | 0.324 |
| HybridFourierEnsemble | Total | 0.101 | 1140.48 | 2284.23 | 0.670 | -0.039 |
| HybridFourierEnsemble | Country | 0.111 | 417.06 | 816.30 | 0.701 | -0.039 |
| HybridFourierEnsemble | Country_Region | 0.127 | 159.58 | 286.23 | 0.702 | -0.039 |
| HybridFourierEnsemble | Country_Region_Chain | 0.146 | 71.62 | 134.66 | 0.747 | -0.039 |
| HybridFourierEnsemble | Bottom | 0.217 | 21.29 | 37.83 | 0.755 | -0.039 |

WAPE is total absolute error divided by total actual demand. MASE scales each node's MAE by its in-training weekly-naive error. WAPE can hide node-level dispersion, so `holdout_node_metrics.csv` contains every hierarchy node.

The selected model underforecast the holdout by 3.9% at the bottom level. At the total level its WAPE was 0.101, slightly below the weekly-naive value of 0.386. This is the main hierarchy-level trade-off: the ensemble improved SKU-level allocation but did not beat weekly naive at the grand total in this holdout.

## Explanation, uncertainty, and limitations

A fixed 50/50 average of DirectHybridGB and FourierTrendRidge, combining local nonlinear adaptation with smooth trend and seasonality. The model's relationships are predictive associations, not causal effects.

The future forecast includes approximate 80% empirical intervals calibrated from absolute rolling-validation residuals by horizon and scaled by the square root of forecast demand. Bottom bounds are summed upward, so aggregate bounds are coherent but conservative. The median horizon calibration multiplier is 2.77.


For the boosting component, permutation importance on a reproducible training sample ranked the largest residual-error contributions as `seasonal_naive_7` (33.50), `mean_56` (20.45), `day_of_week` (3.97), `month` (3.33), `cos_year` (3.05). Values are increases in residual MAE when a feature is shuffled and should not be read as causal effects.

## Future 28-day forecast

For 2026-08-08 through 2026-11-27, coherent total demand is forecast at **1,322,236 cartons**, +4.6% versus the most recent 112 observed days. The highest forecast day is 2026-11-16 at 16,121 cartons. These values are conditional on normal calendar behavior because no future event schedule was available.

The largest limitation is missing future-known promotion, holiday, price, campaign, distribution, and stockout information. Unflagged shocks are irreducible from the provided columns, and only two annual cycles are available. The 28-day assumption must be revisited if purchasing decisions use another horizon.

## Operational recommendation

Run this as a 28-day batch forecast, refresh it whenever new daily actuals arrive, and use weekly seasonal naive as the automatic fallback. Monitor schema/key completeness, hierarchy mappings, zero rate, WAPE/MASE/bias by hierarchy level, and interval coverage after outcomes arrive. Retrain or investigate when four-week WAPE degrades materially from the backtest range or bias persists in one direction.

Before production, add event/price/stockout fields with explicit future availability, confirm the planning horizon and over-versus-underforecast cost, and validate the forecast in the inventory decision process rather than treating statistical error alone as business value.

## Interactive dashboard

[Open the Retail Demand Forecast Dashboard](https://small-waterfall-e74f.hazirahrafidi.workers.dev)

The dashboard provides cascading Country, Region, Chain, and Parent SKU filters. It opens with a monthly time-series view and supports click-through drill-down to daily historical actuals, in-sample fitted estimates, and the 28-day out-of-sample forecast with approximate 80% intervals.

## Reproducibility

From the project directory, reproduce the analysis with the default settings:

```bash
cd /Users/sitinoorhazirah/ML-projects/Retail_Demand_Forecast
python main.py
```

To provide every forecasting argument explicitly:

```bash
python main.py --data data/retail_timeseries_2yr.csv --output-dir artifacts --horizon 112 --validation-folds 2
```

| Argument | Purpose | Default |
|---|---|---|
| `--data` | Input retail time-series CSV | `data/retail_timeseries_2yr.csv` |
| `--output-dir` | Forecast, metric, plot, and report directory | `artifacts` |
| `--horizon` | Number of future daily periods | `112` |
| `--validation-folds` | Number of expanding-window validation folds | `2` |

The script can also be run from another directory by using its absolute path:

```bash
python /Users/sitinoorhazirah/ML-projects/Retail_Demand_Forecast/main.py --horizon 112 --validation-folds 2
```

Display the command-line help with `python main.py --help`.

After the forecast artifacts exist, rebuild the interactive dashboard with:

```bash
npm install
python build_dashboard.py
```

Keep `--horizon 28` when rebuilding the current dashboard because its forecast and interval configuration is designed for 28 days. The forecasting script records package versions, diagnostics, every validation fold, holdout metrics, feature importance, plots, and coherent future forecasts in `artifacts/`.
