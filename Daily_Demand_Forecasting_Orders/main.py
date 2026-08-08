"""Demand forecasting for the UCI Daily Demand Forecasting Orders data.

The source data has no calendar date, so row order is treated as the time axis.
Only calendar fields that are available before the forecast is issued are used;
same-day order components are intentionally excluded to prevent target leakage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).parent.parent))

from scripts.fetch_data import FetchData
from scripts.forecast_metrics import forecast_metrics


TARGET_COLUMN = "Total orders"
DAY_COLUMN = "Day of the week"
WEEK_COLUMN = "Week of the month"
SEASON_LENGTH = 5  # Five working days (values 2 through 6 in this dataset).
DEFAULT_HORIZON = 5
DEFAULT_DATA_PATH = Path("./data/daily_demand_forecasting_orders.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")


def load_dataset(path: Path, refresh: bool = False) -> pd.DataFrame:
    """Load the local CSV, fetching UCI dataset 409 only when necessary."""
    if path.exists() and not refresh:
        return pd.read_csv(path)

    fetcher = FetchData()
    dataset = fetcher.fetch_data(id=409)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def validate_dataset(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Validate the forecasting data contract and return an ordered copy."""
    required = {TARGET_COLUMN, DAY_COLUMN, WEEK_COLUMN}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")
    if len(frame) < max(4 * horizon, 4 * SEASON_LENGTH):
        raise ValueError(
            "Too little history for chronological backtesting: "
            f"received {len(frame)} rows for a {horizon}-step horizon."
        )

    ordered = frame.reset_index(drop=True).copy()
    for column in required:
        ordered[column] = pd.to_numeric(ordered[column], errors="raise")

    if ordered[list(required)].isna().any().any():
        raise ValueError("Target and calendar columns must not contain missing values.")
    if not np.isfinite(ordered[TARGET_COLUMN]).all():
        raise ValueError(f"{TARGET_COLUMN!r} contains non-finite values.")
    if (ordered[TARGET_COLUMN] < 0).any():
        raise ValueError(f"{TARGET_COLUMN!r} must be non-negative demand.")
    if not ordered[DAY_COLUMN].isin(range(2, 7)).all():
        raise ValueError(f"{DAY_COLUMN!r} must contain working-day values 2 through 6.")
    if not ordered[WEEK_COLUMN].isin(range(1, 6)).all():
        raise ValueError(f"{WEEK_COLUMN!r} must contain values 1 through 5.")

    ordered[DAY_COLUMN] = ordered[DAY_COLUMN].astype(int)
    ordered[WEEK_COLUMN] = ordered[WEEK_COLUMN].astype(int)
    ordered.insert(0, "observation_id", np.arange(1, len(ordered) + 1))
    return ordered


def make_future_calendar(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Continue the dataset's five-day and five-week calendar convention."""
    day = int(history.iloc[-1][DAY_COLUMN])
    week = int(history.iloc[-1][WEEK_COLUMN])
    rows: list[dict[str, int]] = []

    for step in range(1, horizon + 1):
        if day == 6:
            day = 2
            week = 1 if week == 5 else week + 1
        else:
            day += 1
        rows.append(
            {
                "observation_id": len(history) + step,
                WEEK_COLUMN: week,
                DAY_COLUMN: day,
            }
        )
    return pd.DataFrame(rows)


def predict_naive(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Repeat the most recently observed demand."""
    return np.repeat(float(train[TARGET_COLUMN].iloc[-1]), len(future))


def predict_seasonal_naive(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Repeat the last five observed working-day values."""
    season = train[TARGET_COLUMN].iloc[-SEASON_LENGTH:].to_numpy(dtype=float)
    return np.resize(season, len(future))


def predict_weekday_mean(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Forecast each working day with its training-only historical mean."""
    means = train.groupby(DAY_COLUMN)[TARGET_COLUMN].mean()
    fallback = float(train[TARGET_COLUMN].mean())
    return future[DAY_COLUMN].map(means).fillna(fallback).to_numpy(dtype=float)


def predict_ridge_calendar(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Regularized trend plus future-known weekday/week-of-month effects."""
    categorical = [DAY_COLUMN, WEEK_COLUMN]
    train_features = train[categorical].copy()
    train_features["trend"] = np.arange(len(train), dtype=float)
    future_features = future[categorical].copy()
    future_features["trend"] = np.arange(
        len(train), len(train) + len(future), dtype=float
    )

    preprocessing = ColumnTransformer(
        [
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("trend", StandardScaler(), ["trend"]),
        ]
    )
    # Strong regularization is appropriate for only 60 observations.
    model = make_pipeline(preprocessing, Ridge(alpha=100.0))
    model.fit(train_features, train[TARGET_COLUMN])
    return np.maximum(0.0, model.predict(future_features))


ForecastFunction = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
MODELS: dict[str, ForecastFunction] = {
    "naive_last_value": predict_naive,
    "seasonal_naive_5": predict_seasonal_naive,
    "weekday_mean": predict_weekday_mean,
    "ridge_calendar_trend": predict_ridge_calendar,
}


def rolling_origins(n_rows: int, horizon: int, max_folds: int = 6) -> list[int]:
    """Create expanding-window origins aligned to the production horizon."""
    first_origin = max(3 * SEASON_LENGTH, n_rows - max_folds * horizon)
    origins = list(range(first_origin, n_rows - horizon + 1, horizon))
    if len(origins) < 2:
        raise ValueError("At least two complete chronological backtest folds are required.")
    return origins[-max_folds:]


def backtest_models(
    frame: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every candidate on identical expanding-window folds."""
    records: list[dict[str, float | int | str]] = []
    for model_name, forecast_function in MODELS.items():
        for fold, origin in enumerate(rolling_origins(len(frame), horizon), start=1):
            train = frame.iloc[:origin]
            test = frame.iloc[origin : origin + horizon]
            prediction = forecast_function(train, test)
            for step, (row_index, actual, predicted) in enumerate(
                zip(test.index, test[TARGET_COLUMN], prediction), start=1
            ):
                records.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "origin": origin,
                        "horizon_step": step,
                        "observation_id": int(frame.loc[row_index, "observation_id"]),
                        "actual": float(actual),
                        "predicted": float(predicted),
                        "error": float(predicted - actual),
                        "absolute_error": float(abs(predicted - actual)),
                    }
                )

    predictions = pd.DataFrame(records)
    comparison_rows: list[dict[str, float | int | str]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        metrics = forecast_metrics(
            group["actual"].to_numpy(), group["predicted"].to_numpy()
        )
        fold_mae = group.groupby("fold")["absolute_error"].mean()
        comparison_rows.append(
            {
                "model": model_name,
                "folds": int(group["fold"].nunique()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape_pct": metrics["wape_percent"],
                "bias": metrics["bias"],
                "mean_fold_mae": float(fold_mae.mean()),
                "std_fold_mae": float(fold_mae.std(ddof=1)),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("mae").reset_index(drop=True)
    seasonal_mae = float(
        comparison.loc[comparison["model"] == "seasonal_naive_5", "mae"].iloc[0]
    )
    comparison["lift_vs_seasonal_naive_pct"] = 100 * (
        seasonal_mae - comparison["mae"]
    ) / seasonal_mae
    return comparison, predictions


def empirical_intervals(
    selected_backtests: pd.DataFrame,
    point_forecast: np.ndarray,
    coverage: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Build horizon-specific empirical error bands from held-out errors."""
    alpha = 1.0 - coverage
    lower: list[float] = []
    upper: list[float] = []
    for step, point in enumerate(point_forecast, start=1):
        errors = selected_backtests.loc[
            selected_backtests["horizon_step"] == step, "error"
        ].to_numpy(dtype=float)
        low_adjustment, high_adjustment = np.quantile(
            errors, [alpha / 2, 1 - alpha / 2]
        )
        # error = predicted - actual, so actual = predicted - error.
        lower.append(max(0.0, float(point - high_adjustment)))
        upper.append(max(0.0, float(point - low_adjustment)))
    return np.asarray(lower), np.asarray(upper)


def save_forecast_plot(
    history: pd.DataFrame, forecast: pd.DataFrame, output_path: Path
) -> None:
    """Save an actual-versus-forecast chart using observation order."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml-projects-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = history.tail(30)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        context["observation_id"],
        context[TARGET_COLUMN],
        marker="o",
        linewidth=1.8,
        label="Actual total orders",
    )
    bridge_x = np.r_[context["observation_id"].iloc[-1], forecast["observation_id"]]
    bridge_y = np.r_[context[TARGET_COLUMN].iloc[-1], forecast["forecast"]]
    axis.plot(bridge_x, bridge_y, marker="o", linewidth=2, label="Forecast")
    axis.fill_between(
        forecast["observation_id"],
        forecast["lower_80"],
        forecast["upper_80"],
        alpha=0.2,
        label="80% empirical interval",
    )
    axis.axvline(len(history) + 0.5, color="black", linestyle="--", linewidth=1)
    figure.suptitle("Demand forecast: Total orders", fontsize=16, y=0.99)
    axis.set_title(
        "Last 30 observations and next 5 forecast steps; shaded area is the 80% empirical interval",
        fontsize=10,
        color="#4b5563",
        pad=10,
    )
    axis.set(
        xlabel="Observation sequence (source data has no date column)",
        ylabel="Total orders",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_forecast(
    data_path: Path,
    output_dir: Path,
    horizon: int = DEFAULT_HORIZON,
    refresh_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run validation, select a model, fit all history, and save artifacts."""
    raw = load_dataset(data_path, refresh=refresh_data)
    frame = validate_dataset(raw, horizon=horizon)
    comparison, backtests = backtest_models(frame, horizon=horizon)
    selected_model = str(comparison.iloc[0]["model"])

    future = make_future_calendar(frame, horizon=horizon)
    point_forecast = MODELS[selected_model](frame, future)
    selected_backtests = backtests.loc[backtests["model"] == selected_model]
    lower, upper = empirical_intervals(selected_backtests, point_forecast)

    forecast = future.copy()
    forecast["forecast"] = point_forecast
    forecast["lower_80"] = lower
    forecast["upper_80"] = upper
    forecast["selected_model"] = selected_model

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "demand_forecast.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    backtests.to_csv(output_dir / "backtest_predictions.csv", index=False)
    save_forecast_plot(frame, forecast, output_dir / "demand_forecast.png")

    zero_rate = float((frame[TARGET_COLUMN] == 0).mean())
    print(f"Rows: {len(frame)} | Missing target values: 0 | Target zero rate: {zero_rate:.1%}")
    print(
        f"Backtest: {int(comparison.iloc[0]['folds'])} expanding-window folds, "
        f"{horizon} observations per fold"
    )
    print("\nModel comparison (held-out errors):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected model: {selected_model}")
    print("\nFuture demand forecast:")
    print(forecast.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")
    return forecast, comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch UCI dataset 409 again even when the local CSV exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_forecast(
        data_path=arguments.data_path,
        output_dir=arguments.output_dir,
        horizon=arguments.horizon,
        refresh_data=arguments.refresh_data,
    )
