# Daily Demand Forecasting for Logistics Orders

## Technical Summary

This project forecasts the target **`Total orders`** for the next five working-day observations in the UCI Daily Demand Forecasting Orders dataset ([Link](https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders)). Four candidate methods were evaluated with six expanding-window backtests. The **weekday historical mean** model produced the lowest held-out mean absolute error (MAE) at **64.65 dataset order units**, improving on the five-day seasonal-naive baseline by **37.85%**. The selected model uses the future **day-of-week label** as its only predictor and learns one mean target value per weekday from the training history; it is not the Ridge model.

The final forecast expects the highest demand in the first forecast step, corresponding to Monday, at **390.21**, followed by lower demand from Tuesday through Friday. This result can guide short-term capacity or staffing planning, but the analysis should be treated as **shareable with caveats**, not production-ready: the dataset contains only 60 observations, has no true date column, and does not document holidays or the exact historical period.

## 1. About the Project

The project is a compact, reproducible time-series forecasting pipeline for daily logistics demand. It reads the Daily Demand Forecasting Orders data, validates its structure, evaluates several forecasting approaches without shuffling time, selects the best candidate using held-out error, trains that model on all available history, and produces a five-step forecast with uncertainty estimates.

The implementation is contained in [`main.py`](main.py). The target variable is **`Total orders`**, interpreted in the numerical units supplied by the source dataset. Because the dataset contains only one sequence and no entity identifier, the task is classified as a **single-series demand forecasting problem with future-known calendar information**.

The project intentionally uses only information that would be available before future demand occurs. Weekday, week of month, observation order, and historical target values are valid forecasting inputs. Same-day order-component columns are not used because they would not be known when a future forecast is issued.

## 2. Objective of the Project

The primary objective is to estimate demand for the next five working-day observations so that a logistics operation can anticipate likely workload and plan resources such as staffing, processing capacity, and order-handling coverage.

The technical objectives are to:

1. Forecast **`Total orders`** for a five-observation horizon.
2. Preserve chronological order throughout training and testing.
3. Compare the chosen model with simple, decision-relevant baselines.
4. Prevent target leakage by excluding same-day order totals and components.
5. Quantify out-of-sample error using MAE, RMSE, WAPE, and bias.
6. Provide an empirical uncertainty range around each future point forecast.
7. Save reproducible tables and a chart for review or downstream use.

## 3. Output of the Project

### Five-step demand forecast

| Forecast observation | Assumed weekday | Week of month | Point forecast | Lower 80% bound | Upper 80% bound |
|---:|---|---:|---:|---:|---:|
| 61 | Monday | 1 | 390.21 | 319.34 | 399.50 |
| 62 | Tuesday | 1 | 303.93 | 249.87 | 411.08 |
| 63 | Wednesday | 1 | 281.93 | 199.57 | 467.22 |
| 64 | Thursday | 1 | 273.97 | 219.35 | 370.89 |
| 65 | Friday | 1 | 263.35 | 221.29 | 387.57 |

The point estimates suggest that the operation should plan for its highest workload on the first forecast day. The shaded interval in the following chart shows how uncertainty varies by forecast step. It is based on historical backtest errors and is wider where previous forecasts were less stable.

![Actual Total Orders and Five-Step Forecast](outputs/demand_forecast.png)

The generated outputs are:

- [`outputs/demand_forecast.csv`](outputs/demand_forecast.csv): future calendar fields, point forecasts, interval bounds, and selected model.
- [`outputs/model_comparison.csv`](outputs/model_comparison.csv): held-out evaluation metrics for every candidate model.
- [`outputs/backtest_predictions.csv`](outputs/backtest_predictions.csv): fold-level actuals, predictions, errors, and horizon steps.
- [`outputs/demand_forecast.png`](outputs/demand_forecast.png): the final actual-versus-forecast chart.
- [`scripts/forecast_metrics.py`](../scripts/forecast_metrics.py): reusable and auditable metric calculations.

The pipeline can be run from the project directory with:

```bash
.env/bin/python main.py
```

The horizon can be changed with `--horizon`, although a longer horizon leaves fewer complete backtest folds and is less defensible with only 60 records.

## 4. Data, Source, and Collection

### Source and provenance

The project uses the [UCI Machine Learning Repository: Daily Demand Forecasting Orders](https://archive.ics.uci.edu/dataset/409/daily%2Bdemand%2Bforecasting%2Borders), dataset ID **409** and DOI **10.24432/C5BC8T**. UCI states that the data were collected over **60 days** from a real, large Brazilian logistics company and were used in academic research at Universidade Nove de Julho. The dataset was donated to UCI on 20 November 2017 and is distributed under the Creative Commons Attribution 4.0 license.

The public documentation describes the records as daily operational order data. It does **not** provide the exact collection dates, the company's identity, the transactional system used, timezone, holiday calendar, data-entry process, or rules for missing business days. This report therefore does not infer those details.

### How the project obtains the data

The local [`scripts/fetch_data.py`](../scripts/fetch_data.py) module calls `ucimlrepo.fetch_ucirepo(id=409)`. The feature and target DataFrames returned by UCI are combined and cached as [`data/daily_demand_forecasting_orders.csv`](data/daily_demand_forecasting_orders.csv). On subsequent runs, `main.py` reads the cached file unless `--refresh-data` is supplied.

### Dataset structure

The local data contains **60 rows and 13 columns**: 12 source features and one target. Each row represents a daily observation in the supplied sequence.

| Field group | Columns | Role in this project |
|---|---|---|
| Calendar | `Week of the month`, `Day of the week` | Future-known calendar fields used by eligible models |
| Demand timing/type | `Non-urgent order`, `Urgent order`, `Order type A`, `Order type B`, `Order type C`, `Fiscal sector orders` | Descriptive same-day values; excluded from future forecasting |
| Sector/order context | Traffic-controller-sector orders and three banking-order fields | Descriptive same-day values; excluded because future availability is not documented |
| Target | `Total orders` | Quantity to forecast |

UCI encodes Monday through Friday as day values 2 through 6. The local target has a mean of **300.87**, a median of **288.03**, and a range from **129.41** to **616.45** in the dataset's supplied units.

### Data-quality and leakage checks

- Missing values across the local dataset: **0**.
- Fully duplicated rows: **0**.
- Missing target values: **0**.
- Zero-valued targets: **0%**.
- Non-finite or negative targets: **0**.
- Actual timestamp column: **not available**.
- Series/entity identifier: **not available**.

A critical leakage check confirmed that `Order type A + Order type B + Order type C` equals `Total orders` for all 60 rows, with a maximum absolute difference of approximately **5.7 × 10⁻¹⁴**, attributable to floating-point arithmetic. Including those fields would allow a model to reconstruct the target from same-day information instead of forecasting future demand. All same-day component columns were therefore excluded.

## 5. End-to-End Model Development and Testing

### Step 1: Load and cache the source data

The pipeline first uses the cached CSV when available. If it is missing, or if `--refresh-data` is requested, the code downloads UCI dataset 409, combines its feature and target tables, and saves a new local CSV.

### Step 2: Validate the forecasting contract

Before modeling, the code checks that:

- the target, weekday, and week-of-month columns exist;
- the forecast horizon is valid relative to the available history;
- required fields are numeric and contain no missing values;
- the target is finite and non-negative;
- weekday values are within 2–6 and week-of-month values within 1–5.

An `observation_id` is then added to preserve the supplied sequence. No random split or row shuffling is used.

### Step 3: Define the forecast horizon and future calendar

The default horizon is five observations, matching one assumed Monday-to-Friday cycle. Because the source has no date column, the future frame is created by advancing the encoded weekday and rolling week 5 back to week 1. This is an explicit modeling assumption, not a recovered historical calendar.

### Step 4: Build leakage-safe candidate models

Four deliberately simple candidates were selected because the dataset is very short. The first two establish reference performance, the weekday-mean model tests whether working-day seasonality is useful, and Ridge tests whether week-of-month and a gradual time trend improve on that seasonal pattern.

#### Feature-use summary

| Candidate | Training inputs | Future inputs | Transformations | Target history used? |
|---|---|---|---|---|
| Naive last value | Most recent training value of `Total orders` | Forecast horizon length only | None | Yes: latest value only |
| Five-step seasonal naive | Final five training values of `Total orders` | Forecast horizon length only | Five-value cycle repeated with `numpy.resize` | Yes: final five values |
| **Weekday historical mean — selected** | `Day of the week` and `Total orders` from training rows | Future `Day of the week` | Group-by weekday and arithmetic mean | Yes: all targets grouped by weekday |
| Ridge calendar and trend | `Day of the week`, `Week of the month`, generated `trend`, and target | Future weekday, week of month, and continued trend | One-hot encoding for calendar fields; standardization for trend | Target is the response only; no target lags |

The `observation_id` column is retained for ordering and reporting but is **not** passed directly into any model. The Ridge candidate separately generates a zero-based `trend` value from row position.

#### Model 1: Naive last-value baseline

The naive model assumes the next observations will equal the most recently observed demand:

```text
forecast(T + h) = Total orders at T, for every horizon step h
```

- **Feature used:** the final `Total orders` value in the current training fold.
- **Features not used:** weekday, week of month, trend, and all same-day order components.
- **Training:** no parameters are optimized. The “fitted state” is one scalar: the latest target value.
- **Multi-step behavior:** the same scalar is repeated for all five forecast steps.
- **Purpose:** establishes whether a model can beat the simplest persistence assumption.

At each backtest origin, the latest value is taken from that fold's training data only. No value from the next five-row test window is visible to the baseline.

#### Model 2: Five-step seasonal-naive baseline

The seasonal-naive model assumes that the most recent five-observation demand pattern will repeat:

```text
training tail = [y(T-4), y(T-3), y(T-2), y(T-1), y(T)]
next 5 forecasts = the same five values in the same order
```

- **Features used:** the final five `Total orders` values in the training fold.
- **Season length:** 5 observations, representing the assumed five-working-day cycle.
- **Transformation:** the five-value array is repeated with `numpy.resize` when the requested horizon is longer than five.
- **Features not used:** the weekday label itself, week of month, trend, and same-day order components.
- **Training:** no numerical optimization or scaling is required.
- **Purpose:** provides the main seasonal baseline against which model lift is reported.

This baseline repeats the last five **rows**, not five explicitly date-matched weekdays. Because the source contains no actual date column, it cannot detect whether a holiday or omitted business day disrupted the row sequence.

#### Model 3: Weekday historical mean — selected model

The selected model estimates a separate expected demand level for each weekday. For a future day `d`, the forecast is:

```text
forecast(d) = mean(Total orders for all training rows where Day of the week = d)
```

- **Predictor used:** `Day of the week`.
- **Response used during fitting:** `Total orders`.
- **Learned parameters:** up to five means, one each for encoded weekdays 2, 3, 4, 5, and 6.
- **Aggregation:** ordinary arithmetic mean within each weekday group.
- **Fallback:** if a future weekday has never appeared in the training fold, the overall training-target mean is used. This fallback was not needed in the reported final fit because all five weekday categories were present.
- **Features not used:** `Week of the month`, observation trend, target lags, same-day component orders, traffic-controller orders, or banking-order fields.
- **Scaling/encoding:** none. The model performs an exact lookup from future weekday to its training-only mean.
- **Multi-step behavior:** each future row is forecast independently from its weekday category, so errors do not recursively feed into later steps.

For every backtest fold, the weekday means are recalculated using only the rows before that fold's forecast origin. After the weekday model wins model selection, the means are recalculated once using all 60 observations. The resulting full-data means—390.21 for Monday through 263.35 for Friday—form the five final point forecasts.

This model is equivalent to a categorical seasonal-mean estimator. It captures a stable average working-day pattern but cannot represent recent momentum, nonlinear interactions, holidays, or event-specific demand.

#### Model 4: Ridge calendar-and-trend regression

Ridge is the only fitted regression model among the candidates. It predicts `Total orders` from calendar categories and a linear sequence trend:

```text
prediction = intercept
           + weekday coefficients
           + week-of-month coefficients
           + trend coefficient
```

It minimizes squared prediction error plus an L2 penalty on coefficient magnitude:

```text
sum((actual - prediction)^2) + alpha * sum(coefficient^2)
```

with `alpha = 100`. The strong penalty shrinks coefficients toward zero to reduce variance and overfitting in the 60-row dataset.

The Ridge feature-processing pipeline is:

1. **`Day of the week`:** treated as categorical, not continuous. `OneHotEncoder` creates one indicator column for each observed weekday category.
2. **`Week of the month`:** also treated as categorical. A separate one-hot indicator is created for each observed week category.
3. **`trend`:** generated as `0, 1, 2, …, n−1` for a training fold and continued as `n, n+1, …` for the forecast rows.
4. **Trend scaling:** `StandardScaler` subtracts the training-fold trend mean and divides by its training-fold standard deviation. Calendar indicators are not standardized.
5. **Target:** `Total orders` remains in its original dataset units and is not scaled or transformed.
6. **Prediction constraint:** negative Ridge predictions are clipped to zero because demand cannot be negative.

When all calendar categories are present, the transformed design matrix contains approximately 11 columns: five weekday indicators, five week-of-month indicators, and one standardized trend feature. The exact encoder categories and trend-scaling statistics are fitted again inside every training fold. `handle_unknown="ignore"` ensures an unseen calendar category would not cause prediction to fail.

Ridge does **not** use lagged `Total orders`, rolling averages, or the same-day component columns. Consequently, its forecast changes only with the calendar categories and the extrapolated linear trend. Its lower absolute bias than the selected weekday model shows that the trend and regularization balanced over- and underforecasting well, but its MAE was still slightly worse.

#### Features deliberately excluded from model training

| Excluded field or group | Reason for exclusion |
|---|---|
| `Order type A`, `Order type B`, `Order type C` | Their sum reconstructs `Total orders`; using them would directly leak the target. |
| `Non-urgent order`, `Urgent order`, `Fiscal sector orders` | Same-day demand components whose future values are not documented as known at forecast issuance. |
| Traffic-controller-sector and banking-order fields | Future availability is not documented; using observed future test values would create look-ahead leakage. |
| `observation_id` | Synthetic reporting key, not a business predictor. Ridge uses a separately constructed and fold-safe trend instead. |
| True date, holiday, promotion, weather, or event features | Not present in the source dataset and therefore not invented. |

Every candidate therefore uses only training history and information assumed to be available when the forecast is issued. Feature generation, one-hot encoding, and trend scaling are refitted inside each chronological fold rather than once on the full dataset.

### Step 5: Perform chronological backtesting

Model testing uses six expanding-window origins at observations **30, 35, 40, 45, 50, and 55**. At each origin, a candidate is trained using only observations before the origin and predicts the next five observations. This produces 30 held-out predictions per model.

This design mirrors the five-step forecast task more closely than a random train/test split. It also tests models across several different historical endpoints rather than relying on a single favorable split.

### Step 6: Select and refit the model

Candidates are ranked by overall held-out MAE. The weekday historical mean has the lowest MAE and is selected. It is then fitted using all 60 observations to create the final five-step forecast.

The selected model's full-data weekday means are:

| Weekday | Historical observations | Mean `Total orders` |
|---|---:|---:|
| Monday | 11 | 390.21 |
| Tuesday | 12 | 303.93 |
| Wednesday | 13 | 281.93 |
| Thursday | 12 | 273.97 |
| Friday | 12 | 263.35 |

These values become the point forecasts because the selected model estimates demand from the historical mean for each weekday.

### Step 7: Estimate uncertainty and save artifacts

For each horizon step, the pipeline calculates the 10th and 90th percentiles of the selected model's signed backtest errors and applies those adjustments to the final point forecast. The resulting bounds are labeled an **80% empirical interval**. The tables and plot are then written to the `outputs` directory.

### Technical testing performed

- Python syntax compilation passed for `main.py`, `fetch_data.py`, and `forecast_metrics.py`.
- A complete offline run succeeded using the cached CSV.
- Metric calculations were independently recomputed from the saved backtest predictions.
- Output files were checked for successful creation and non-zero size.
- The exported PNG was visually inspected for readable labels, honest chronological ordering, interval display, and a clearly marked forecast boundary.

## 6. Model Evaluation

### Metric definitions

- **MAE:** average absolute difference between predicted and actual demand; lower is better.
- **RMSE:** square-error metric that penalizes large misses more heavily; lower is better.
- **WAPE:** total absolute error divided by total actual demand; lower is better.
- **Bias:** mean of `prediction − actual`; a negative value indicates average underforecasting.
- **Lift versus seasonal naive:** percentage reduction in MAE relative to the five-step seasonal-naive baseline; higher is better.

### Backtest results

| Model | MAE | RMSE | WAPE | Bias | Fold-MAE standard deviation | MAE lift vs. seasonal naive |
|---|---:|---:|---:|---:|---:|---:|
| **Weekday historical mean** | **64.65** | **92.60** | **20.65%** | -12.96 | 24.08 | **37.85%** |
| Ridge calendar and trend | 67.28 | 94.61 | 21.49% | -0.45 | 28.83 | 35.33% |
| Naive last value | 88.84 | 117.08 | 28.37% | -27.66 | 29.28 | 14.60% |
| Five-step seasonal naive | 104.03 | 138.17 | 33.22% | -1.55 | 44.20 | 0.00% |

### Interpretation

The weekday historical mean is selected because it produces the lowest held-out MAE and WAPE. Its MAE means the forecast missed actual demand by about **64.65 units per observation on average** across the 30 held-out predictions. Its WAPE means the total absolute error was approximately **20.65% of total held-out demand**.

The Ridge model is close, with an MAE only **2.62 units** higher and substantially less bias. This suggests that the evidence for choosing the weekday mean over Ridge is modest, even though the weekday mean ranks first. The selected model's bias of **−12.96** indicates mild average underforecasting, which may matter if under-capacity is more costly than over-capacity.

The weekday model's performance is consistent with a strong weekday pattern: average demand is highest on Monday and declines through Friday. This is predictive evidence, not proof that weekday itself causes the demand difference.

### Limitations and robustness assessment

1. **Short history:** 60 observations are insufficient for confidently estimating long-term seasonality, structural changes, or rare peaks.
2. **No real timestamps:** holidays, missing working days, exact month boundaries, elapsed time, and the original date range cannot be reconstructed.
3. **No untouched final test set:** the same rolling backtest framework is used for model comparison and performance estimation. This is reasonable for the short sample but makes the reported error somewhat optimistic relative to a fully independent future test.
4. **Small interval sample:** each horizon-specific interval is estimated from only six backtest errors. The “80%” label describes the empirical quantiles used; it is not a guarantee of future 80% coverage.
5. **Limited predictors:** the selected model captures average weekday demand but cannot respond to promotions, holidays, staffing constraints, economic events, or known future orders.
6. **Operational loss is unspecified:** model selection minimizes MAE. If underforecasting is more costly than overforecasting, a quantile forecast or asymmetric cost function would be more appropriate.

Overall validation status: **Share with caveats**. The pipeline and calculations are reproducible and leakage-safe for the stated assumptions, but the data is too short and temporally incomplete for production deployment.

## Recommended Next Steps

1. Replace observation sequence with a true daily timestamp and business calendar.
2. Collect at least 6–12 months of history so monthly seasonality, holidays, and structural changes can be evaluated.
3. Record which predictors are genuinely known at forecast issuance time, such as planned orders, promotions, and scheduled events.
4. Define the business cost of underforecasting versus overforecasting and choose metrics or quantiles accordingly.
5. Retain a final untouched test period once more history becomes available.
6. Monitor MAE, WAPE, bias, and interval coverage after each forecast cycle, using seasonal naive as the fallback model.

## Further Questions

- What real calendar dates correspond to the 60 observations?
- Were holidays or non-operating days omitted from the source data?
- Are the supplied numerical values literal order counts, scaled quantities, or aggregated workloads?
- Which order or operational signals are available before the daily forecast is issued?
- What forecast error is operationally acceptable, and is underforecasting more costly than overforecasting?
