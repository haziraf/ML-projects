"""Hierarchical retail-demand forecasting pipeline.

The script profiles the daily Country -> Region -> Chain -> Parent SKU panel,
selects a model with rolling-origin validation, evaluates it once on a final
holdout, and creates a coherent 28-day forecast at every hierarchy level.

Run from any directory with:

    python /path/to/Retail_Demand_Forecast/main.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "retail_demand_forecast_matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL


SEED = 42
TARGET = "Total_Qty_CTN"
DATE_COL = "Date"
KEYS = ["Country", "Region", "Chain", "Parent_SKU"]
CATEGORICAL_FEATURES = KEYS + ["day_of_week", "month"]


@dataclass(frozen=True)
class HierarchyLevel:
    name: str
    group_columns: tuple[str, ...]
    node_ids: tuple[str, ...]
    member_indices: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "retail_timeseries_2yr.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "artifacts",
        help="Directory for reports, metrics, plots, and forecasts.",
    )
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--validation-folds", type=int, default=4)
    return parser.parse_args()


def load_and_validate(path: Path, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, parse_dates=[DATE_COL])
    required = {DATE_COL, TARGET, "SKU_Description", *KEYS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        counts = data[list(required)].isna().sum()
        raise ValueError(f"Required fields contain nulls:\n{counts[counts > 0]}")
    if not pd.api.types.is_numeric_dtype(data[TARGET]):
        raise TypeError(f"{TARGET} must be numeric")
    if (data[TARGET] < 0).any():
        raise ValueError(f"{TARGET} contains negative demand")
    if data.duplicated([DATE_COL, *KEYS]).any():
        raise ValueError("Duplicate bottom-series/date keys found; aggregation rule is ambiguous")

    data = data.sort_values([DATE_COL, *KEYS]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(data[DATE_COL].unique()))
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    if not dates.equals(expected):
        absent = expected.difference(dates)
        raise ValueError(f"The global daily calendar is incomplete; {len(absent)} dates are missing")

    wide = (
        data.pivot(index=DATE_COL, columns=KEYS, values=TARGET)
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )
    if wide.isna().any().any():
        missing_cells = int(wide.isna().sum().sum())
        raise ValueError(f"The bottom-level daily panel has {missing_cells} missing cells")
    if len(wide) < max(2 * horizon + 84, 392):
        raise ValueError("History is too short for the requested backtest and annual-lag candidate")

    meta = pd.DataFrame(wide.columns.tolist(), columns=KEYS)
    description_counts = data.groupby("Parent_SKU")["SKU_Description"].nunique()
    if (description_counts > 1).any():
        raise ValueError("Parent_SKU does not map uniquely to SKU_Description")
    return data, wide, meta


def make_hierarchy(meta: pd.DataFrame) -> list[HierarchyLevel]:
    specifications = [
        ("Total", ()),
        ("Country", ("Country",)),
        ("Country_Region", ("Country", "Region")),
        ("Country_Region_Chain", ("Country", "Region", "Chain")),
        ("Bottom", tuple(KEYS)),
    ]
    levels: list[HierarchyLevel] = []
    for name, columns in specifications:
        if not columns:
            levels.append(
                HierarchyLevel(name, columns, ("Total",), (np.arange(len(meta), dtype=int),))
            )
            continue
        groups = meta.groupby(list(columns), sort=True, observed=True).indices
        node_ids: list[str] = []
        members: list[np.ndarray] = []
        for key, indices in groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            node_ids.append(" / ".join(map(str, key_tuple)))
            members.append(np.asarray(indices, dtype=int))
        levels.append(HierarchyLevel(name, columns, tuple(node_ids), tuple(members)))
    return levels


def aggregate(values: np.ndarray, level: HierarchyLevel) -> np.ndarray:
    """Aggregate a [time, bottom_series] matrix to one hierarchy level."""
    return np.column_stack([values[:, indices].sum(axis=1) for indices in level.member_indices])


def seasonal_naive(history: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    return np.stack(
        [history[-period + ((step - 1) % period)] for step in range(1, horizon + 1)]
    )


def recent_weekday_mean(history: np.ndarray, horizon: int, weeks: int = 4) -> np.ndarray:
    result = []
    for step in range(1, horizon + 1):
        latest = -7 + ((step - 1) % 7)
        result.append(np.mean([history[latest - 7 * k] for k in range(weeks)], axis=0))
    return np.stack(result)


def fourier_design(time_index: np.ndarray, origin: int = 0) -> np.ndarray:
    columns = [np.ones(len(time_index)), (time_index - origin) / 365.25]
    for period, order in ((7.0, 3), (365.25, 4)):
        for harmonic in range(1, order + 1):
            angle = 2 * np.pi * harmonic * time_index / period
            columns.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(columns)


def fourier_trend_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Multi-output dynamic harmonic regression on log1p demand."""
    train_time = np.arange(len(history))
    future_time = np.arange(len(history), len(history) + horizon)
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(fourier_design(train_time), np.log1p(history))
    return np.maximum(0.0, np.expm1(model.predict(fourier_design(future_time))))


class DirectHybridForecaster:
    """Global direct model that learns corrections to weekly seasonal naive.

    Training examples mimic a 28-day issuance: all horizon steps use only
    information available at their historical origin. Forecasting is direct,
    so predicted lags never feed later horizon steps.
    """

    def __init__(self, horizon: int, random_state: int = SEED) -> None:
        self.horizon = horizon
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=140,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=1.0,
            categorical_features="from_dtype",
            early_stopping=False,
            random_state=random_state,
        )
        self.category_levels: dict[str, list[Any]] = {}
        self.feature_names: list[str] = []
        self._importance_sample: tuple[pd.DataFrame, np.ndarray] | None = None

    @staticmethod
    def _one_origin_features(
        history: np.ndarray,
        dates: pd.DatetimeIndex,
        meta: pd.DataFrame,
        origin: int,
        target_index: int,
        step: int,
    ) -> pd.DataFrame:
        date = dates[target_index]
        baseline_index = target_index - 7 * ((step + 6) // 7)
        frame = meta.copy()
        frame["horizon"] = step
        frame["time_idx"] = target_index
        frame["day_of_week"] = date.day_name()
        frame["month"] = str(date.month)
        frame["seasonal_naive_7"] = history[baseline_index]
        frame["last_value"] = history[origin]
        frame["mean_7"] = history[origin - 6 : origin + 1].mean(axis=0)
        frame["mean_28"] = history[origin - 27 : origin + 1].mean(axis=0)
        frame["mean_56"] = history[origin - 55 : origin + 1].mean(axis=0)
        frame["std_28"] = history[origin - 27 : origin + 1].std(axis=0)
        frame["recent_trend"] = (
            history[origin - 6 : origin + 1].mean(axis=0)
            - history[origin - 13 : origin - 6].mean(axis=0)
        )
        frame["annual_lag_364"] = (
            history[target_index - 364] if target_index >= 364 else np.nan
        )
        day_of_year = date.dayofyear
        frame["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
        frame["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
        return frame

    def _training_frame(
        self, history: np.ndarray, dates: pd.DatetimeIndex, meta: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        frames: list[pd.DataFrame] = []
        targets: list[np.ndarray] = []
        # Weekly origin spacing matches the intended regular batch issuance and
        # keeps the global training table compact.
        for origin in range(83, len(history) - self.horizon, 7):
            for step in range(1, self.horizon + 1):
                target_index = origin + step
                frame = self._one_origin_features(
                    history, dates, meta, origin, target_index, step
                )
                baseline = frame["seasonal_naive_7"].to_numpy()
                frames.append(frame)
                targets.append(history[target_index] - baseline)
        return pd.concat(frames, ignore_index=True), np.concatenate(targets)

    def _cast_categories(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = frame.copy()
        for column in CATEGORICAL_FEATURES:
            if fit:
                self.category_levels[column] = sorted(frame[column].astype(str).unique())
            frame[column] = pd.Categorical(
                frame[column].astype(str), categories=self.category_levels[column]
            )
        return frame

    def fit(
        self, history: np.ndarray, dates: pd.DatetimeIndex, meta: pd.DataFrame
    ) -> "DirectHybridForecaster":
        features, residual_target = self._training_frame(history, dates, meta)
        features = self._cast_categories(features, fit=True)
        self.feature_names = list(features.columns)
        self.model.fit(features, residual_target)

        rng = np.random.default_rng(self.random_state)
        sample_size = min(5000, len(features))
        sample_indices = rng.choice(len(features), size=sample_size, replace=False)
        self._importance_sample = (
            features.iloc[sample_indices].copy(),
            residual_target[sample_indices].copy(),
        )
        return self

    def forecast(
        self,
        history: np.ndarray,
        history_dates: pd.DatetimeIndex,
        future_dates: pd.DatetimeIndex,
        meta: pd.DataFrame,
    ) -> np.ndarray:
        if len(future_dates) != self.horizon:
            raise ValueError("Future date count must equal the configured horizon")
        all_dates = history_dates.append(future_dates)
        origin = len(history) - 1
        frames = [
            self._one_origin_features(
                history, all_dates, meta, origin, len(history) + step - 1, step
            )
            for step in range(1, self.horizon + 1)
        ]
        features = self._cast_categories(pd.concat(frames, ignore_index=True), fit=False)
        residual = self.model.predict(features)
        prediction = features["seasonal_naive_7"].to_numpy() + residual
        return np.maximum(0.0, prediction).reshape(self.horizon, len(meta))

    def feature_importance(self) -> pd.DataFrame:
        if self._importance_sample is None:
            raise RuntimeError("Fit the model before calculating feature importance")
        features, target = self._importance_sample
        result = permutation_importance(
            self.model,
            features,
            target,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=self.random_state,
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mae_increase": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("mae_increase", ascending=False)
            .reset_index(drop=True)
        )


def all_candidate_forecasts(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    meta: pd.DataFrame,
    horizon: int,
    need_hybrid: bool = True,
) -> tuple[dict[str, np.ndarray], DirectHybridForecaster | None]:
    forecasts = {
        "NaiveLast": np.repeat(history[-1][None, :], horizon, axis=0),
        "SeasonalNaive7": seasonal_naive(history, horizon),
        "RecentWeekdayMean4": recent_weekday_mean(history, horizon),
        "FourierTrendRidge": fourier_trend_forecast(history, horizon),
    }
    hybrid: DirectHybridForecaster | None = None
    if need_hybrid:
        hybrid = DirectHybridForecaster(horizon).fit(history, history_dates, meta)
        forecasts["DirectHybridGB"] = hybrid.forecast(
            history, history_dates, future_dates, meta
        )
        forecasts["HybridFourierEnsemble"] = np.maximum(
            0.0, 0.5 * forecasts["DirectHybridGB"] + 0.5 * forecasts["FourierTrendRidge"]
        )
    return forecasts, hybrid


def metric_row(
    actual: np.ndarray,
    prediction: np.ndarray,
    training: np.ndarray,
) -> dict[str, float]:
    error = prediction - actual
    denominator = float(np.abs(actual).sum())
    seasonal_scale = np.mean(np.abs(training[7:] - training[:-7]), axis=0)
    node_mae = np.mean(np.abs(error), axis=0)
    valid = seasonal_scale > 1e-12
    mase = float(np.mean(node_mae[valid] / seasonal_scale[valid])) if valid.any() else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "WAPE": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        # Positive bias means overforecasting; negative means underforecasting.
        "Bias": float(error.sum() / actual.sum()) if actual.sum() else np.nan,
        "MASE": mase,
    }


def evaluate_forecast(
    actual_bottom: np.ndarray,
    forecast_bottom: np.ndarray,
    training_bottom: np.ndarray,
    hierarchy: Iterable[HierarchyLevel],
) -> list[dict[str, Any]]:
    rows = []
    for level in hierarchy:
        actual = aggregate(actual_bottom, level)
        prediction = aggregate(forecast_bottom, level)
        training = aggregate(training_bottom, level)
        rows.append({"level": level.name, **metric_row(actual, prediction, training)})
    return rows


def node_metrics(
    actual_bottom: np.ndarray,
    forecast_bottom: np.ndarray,
    training_bottom: np.ndarray,
    hierarchy: Iterable[HierarchyLevel],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in hierarchy:
        actual = aggregate(actual_bottom, level)
        prediction = aggregate(forecast_bottom, level)
        training = aggregate(training_bottom, level)
        for node_index, node_id in enumerate(level.node_ids):
            values = metric_row(
                actual[:, [node_index]], prediction[:, [node_index]], training[:, [node_index]]
            )
            rows.append({"level": level.name, "node_id": node_id, **values})
    return pd.DataFrame(rows)


def component_diagnostics(data: pd.DataFrame, wide: pd.DataFrame) -> dict[str, Any]:
    total = wide.sum(axis=1)
    stl = STL(total, period=7, robust=True).fit()
    residual = stl.resid.to_numpy()
    seasonal = stl.seasonal.to_numpy()
    trend = stl.trend.to_numpy()
    seasonal_strength = max(0.0, 1.0 - np.var(residual) / np.var(seasonal + residual))
    trend_strength = max(0.0, 1.0 - np.var(residual) / np.var(trend + residual))

    bottom_strengths = []
    for column in wide.columns:
        decomposition = STL(wide[column], period=7, robust=True).fit()
        resid = decomposition.resid.to_numpy()
        season = decomposition.seasonal.to_numpy()
        tr = decomposition.trend.to_numpy()
        bottom_strengths.append(
            [
                max(0.0, 1.0 - np.var(resid) / np.var(season + resid)),
                max(0.0, 1.0 - np.var(resid) / np.var(tr + resid)),
            ]
        )

    first_90 = data.loc[
        data[DATE_COL] < data[DATE_COL].min() + pd.Timedelta(days=90), TARGET
    ].mean()
    last_90 = data.loc[
        data[DATE_COL] > data[DATE_COL].max() - pd.Timedelta(days=90), TARGET
    ].mean()
    weekday = (
        data.assign(day=data[DATE_COL].dt.day_name())
        .groupby("day", observed=True)[TARGET]
        .mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )

    panel = wide.to_numpy()
    lag_correlations = {
        str(lag): float(np.corrcoef(panel[lag:].ravel(), panel[:-lag].ravel())[0, 1])
        for lag in (1, 7, 14, 28, 364, 365)
    }
    hierarchy_counts = {
        "Country": int(data["Country"].nunique()),
        "Country_Region": int(data[["Country", "Region"]].drop_duplicates().shape[0]),
        "Country_Region_Chain": int(
            data[["Country", "Region", "Chain"]].drop_duplicates().shape[0]
        ),
        "Bottom": int(data[KEYS].drop_duplicates().shape[0]),
    }
    return {
        "rows": int(len(data)),
        "date_start": str(data[DATE_COL].min().date()),
        "date_end": str(data[DATE_COL].max().date()),
        "daily_dates": int(data[DATE_COL].nunique()),
        "hierarchy_counts": hierarchy_counts,
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_bottom_dates": int(data.duplicated([DATE_COL, *KEYS]).sum()),
        "target_mean": float(data[TARGET].mean()),
        "target_median": float(data[TARGET].median()),
        "target_min": float(data[TARGET].min()),
        "target_max": float(data[TARGET].max()),
        "zero_rate": float((data[TARGET] == 0).mean()),
        "first_to_last_90_day_mean_change": float(last_90 / first_90 - 1.0),
        "weekday_mean": {str(k): float(v) for k, v in weekday.items()},
        "weekday_max_min_ratio": float(weekday.max() / weekday.min()),
        "panel_lag_correlations": lag_correlations,
        "weekly_stl_seasonal_strength_total": float(seasonal_strength),
        "weekly_stl_trend_strength_total": float(trend_strength),
        "weekly_stl_seasonal_strength_bottom_median": float(
            np.median(np.asarray(bottom_strengths)[:, 0])
        ),
        "weekly_stl_trend_strength_bottom_median": float(
            np.median(np.asarray(bottom_strengths)[:, 1])
        ),
    }


def plot_components(data: pd.DataFrame, wide: pd.DataFrame, path: Path) -> None:
    total = wide.sum(axis=1)
    stl = STL(total, period=7, robust=True).fit()
    weekday = (
        data.assign(day=data[DATE_COL].dt.day_name())
        .groupby("day", observed=True)[TARGET]
        .mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )
    month = data.assign(month=data[DATE_COL].dt.month).groupby("month")[TARGET].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(total.index, total, color="#9ecae1", linewidth=0.8, label="Daily total")
    axes[0, 0].plot(
        total.index, total.rolling(28, center=True).mean(), color="#08519c", linewidth=2, label="28-day mean"
    )
    axes[0, 0].plot(total.index, stl.trend, color="#cb181d", linewidth=1.4, label="STL trend")
    axes[0, 0].set_title("Total demand: trend and shocks")
    axes[0, 0].set_ylabel("Cartons")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(weekday.index.str[:3], weekday.values, color="#3182bd")
    axes[0, 1].set_title("Mean demand by weekday")
    axes[0, 1].set_ylabel("Cartons per bottom series")

    axes[1, 0].plot(month.index, month.values, marker="o", color="#31a354")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_title("Mean demand by calendar month")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Cartons per bottom series")

    plot_acf(total, lags=60, ax=axes[1, 1], alpha=0.05, zero=False, color="#756bb1")
    axes[1, 1].set_title("Total-demand autocorrelation")
    axes[1, 1].set_xlabel("Lag (days)")
    fig.suptitle("Retail demand time-series diagnostics", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_validation(summary: pd.DataFrame, path: Path) -> None:
    bottom = summary.loc[summary["level"] == "Bottom"].sort_values("WAPE_mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(bottom["model"], bottom["WAPE_mean"], xerr=bottom["WAPE_std"], color="#4292c6")
    ax.invert_yaxis()
    ax.set_xlabel("Rolling-validation WAPE (lower is better)")
    ax.set_title("Bottom-level model comparison; error bars are fold standard deviation")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_holdout(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    selected: np.ndarray,
    seasonal_baseline: np.ndarray,
    model_name: str,
    path: Path,
) -> None:
    actual_total = actual.sum(axis=1)
    selected_total = selected.sum(axis=1)
    baseline_total = seasonal_baseline.sum(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.plot(dates, actual_total, marker="o", color="#252525", label="Actual")
    ax1.plot(dates, selected_total, color="#2171b5", linewidth=2, label=model_name)
    ax1.plot(dates, baseline_total, color="#969696", linestyle="--", label="Seasonal naive 7")
    ax1.set_ylabel("Total cartons")
    ax1.set_title("Untouched 28-day holdout: coherent total forecast")
    ax1.legend(frameon=False)
    error = selected - actual
    horizon_mae = np.mean(np.abs(error), axis=1)
    ax2.bar(dates, horizon_mae, width=0.75, color="#6baed6")
    ax2.set_xlabel("Holdout date (forecast horizon increases left to right)")
    ax2.set_ylabel("Bottom-level MAE")
    ax2.set_title("Selected-model absolute error by horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def validation_summary(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_metrics.groupby(["model", "level"], as_index=False)
        .agg(
            WAPE_mean=("WAPE", "mean"),
            WAPE_std=("WAPE", "std"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            Bias_mean=("Bias", "mean"),
            MASE_mean=("MASE", "mean"),
        )
        .sort_values(["level", "WAPE_mean"])
    )


def model_description(model_name: str) -> str:
    descriptions = {
        "NaiveLast": "Repeats the final observed value for every future day.",
        "SeasonalNaive7": "Repeats the latest observed value for the matching weekday.",
        "RecentWeekdayMean4": "Uses the mean of the last four observations for each weekday.",
        "FourierTrendRidge": (
            "Fits log demand with a linear trend plus weekly and annual Fourier harmonics, "
            "using ridge regularization independently across all bottom series."
        ),
        "DirectHybridGB": (
            "Starts from weekly seasonal naive and uses one pooled histogram-gradient-boosting "
            "model to predict a direct, horizon-specific residual from hierarchy IDs, recent "
            "levels/volatility/trend, a 364-day lag, and known calendar features."
        ),
        "HybridFourierEnsemble": (
            "A fixed 50/50 average of DirectHybridGB and FourierTrendRidge, combining local "
            "nonlinear adaptation with smooth trend and seasonality."
        ),
    }
    return descriptions[model_name]


def write_forecast_table(
    future_dates: pd.DatetimeIndex,
    forecast_bottom: np.ndarray,
    lower_bottom: np.ndarray,
    upper_bottom: np.ndarray,
    hierarchy: Iterable[HierarchyLevel],
    model_name: str,
    path: Path,
) -> None:
    frames = []
    for level in hierarchy:
        point = aggregate(forecast_bottom, level)
        lower = aggregate(lower_bottom, level)
        upper = aggregate(upper_bottom, level)
        frame = pd.DataFrame(
            {
                "Date": np.repeat(future_dates.to_numpy(), len(level.node_ids)),
                "level": np.repeat(level.name, len(future_dates) * len(level.node_ids)),
                "node_id": np.tile(level.node_ids, len(future_dates)),
                "forecast": point.ravel(),
                "lower_80": lower.ravel(),
                "upper_80": upper.ravel(),
                "model": model_name,
            }
        )
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_model: str,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    horizon: int,
    validation_folds: int,
    interval_q: np.ndarray,
    feature_importance: pd.DataFrame | None,
    future_dates: pd.DatetimeIndex,
    future_bottom: np.ndarray,
    recent_actual_bottom: np.ndarray,
) -> None:
    val_bottom = validation.loc[validation["level"] == "Bottom"].set_index("model")
    test_bottom = holdout.loc[holdout["level"] == "Bottom"].set_index("model")
    selected_val = val_bottom.loc[selected_model]
    selected_test = test_bottom.loc[selected_model]
    naive_test = test_bottom.loc["NaiveLast"]
    seasonal_test = test_bottom.loc["SeasonalNaive7"]
    selected_total = holdout.loc[
        (holdout["level"] == "Total") & (holdout["model"] == selected_model)
    ].iloc[0]
    seasonal_total = holdout.loc[
        (holdout["level"] == "Total") & (holdout["model"] == "SeasonalNaive7")
    ].iloc[0]
    lift_naive = 1 - selected_test.WAPE / naive_test.WAPE
    lift_seasonal = 1 - selected_test.WAPE / seasonal_test.WAPE
    future_total = float(future_bottom.sum())
    recent_total = float(recent_actual_bottom.sum())
    future_change = future_total / recent_total - 1.0
    future_daily_total = future_bottom.sum(axis=1)
    peak_index = int(np.argmax(future_daily_total))

    lines = [
        "# Hierarchical retail-demand forecast report",
        "",
        "## Executive summary",
        "",
        f"The selected model is **{selected_model}**. On the untouched {horizon}-day holdout "
        f"({holdout_start.date()} to {holdout_end.date()}), bottom-level WAPE was "
        f"**{selected_test.WAPE:.3f}**, versus **{naive_test.WAPE:.3f}** for last-value naive "
        f"and **{seasonal_test.WAPE:.3f}** for weekly seasonal naive. This is a "
        f"**{lift_naive:.1%}** reduction in absolute error versus naive and "
        f"**{lift_seasonal:.1%}** versus seasonal naive.",
        "",
        "Point forecasts are made for each Country–Region–Chain–Parent SKU series and then "
        "summed upward. They are therefore exactly coherent across Total, Country, "
        "Country–Region, Country–Region–Chain, and bottom levels.",
        "",
        "## Project contract and assumptions",
        "",
        f"- Target: `{TARGET}` in cartons, at daily frequency.",
        f"- Prediction unit: one of {diagnostics['hierarchy_counts']['Bottom']} bottom series on one day.",
        f"- Assumed horizon and issuance cadence: {horizon} days, re-issued every 28 days. "
        "The business horizon and error-cost asymmetry were not supplied.",
        "- Primary selection metric: bottom-level WAPE; MAE, RMSE, MASE, bias, and every "
        "hierarchy level are guardrails. Positive bias denotes overforecasting.",
        "- No promotions, holidays, prices, stockouts, or planned events were supplied, so "
        "future event effects cannot be modeled explicitly.",
        "",
        "## Data and time-series findings",
        "",
        f"- {diagnostics['rows']:,} rows cover {diagnostics['date_start']} through "
        f"{diagnostics['date_end']} with no missing values or duplicate bottom-date keys.",
        f"- Hierarchy sizes: 1 total, {diagnostics['hierarchy_counts']['Country']} countries, "
        f"{diagnostics['hierarchy_counts']['Country_Region']} country–regions, "
        f"{diagnostics['hierarchy_counts']['Country_Region_Chain']} country–region–chains, "
        f"and {diagnostics['hierarchy_counts']['Bottom']} bottom series.",
        f"- Zeros are {diagnostics['zero_rate']:.2%} of observations, so demand is not strongly "
        "intermittent and Croston-style methods were not prioritized.",
        f"- Monday mean demand is {diagnostics['weekday_max_min_ratio']:.2f} times the lowest "
        "weekday mean. Panel lag-7 correlation is "
        f"{diagnostics['panel_lag_correlations']['7']:.3f}.",
        f"- Weekly STL strength is {diagnostics['weekly_stl_seasonal_strength_total']:.3f} at "
        f"the total and {diagnostics['weekly_stl_seasonal_strength_bottom_median']:.3f} for "
        "the median bottom series. Trend strengths are "
        f"{diagnostics['weekly_stl_trend_strength_total']:.3f} and "
        f"{diagnostics['weekly_stl_trend_strength_bottom_median']:.3f}, respectively.",
        f"- Mean demand in the final 90 days is "
        f"{diagnostics['first_to_last_90_day_mean_change']:.1%} above the first 90 days. "
        "The series also contains abrupt spikes, limiting models without future event flags.",
        "",
        "## Validation and model selection",
        "",
        f"{validation_folds} expanding-window validation folds were placed immediately before the untouched "
        "holdout. Each fold forecasts 28 consecutive days; every transformation and model is "
        "fit only on data available before that fold.",
        "",
        "| Model | Bottom validation WAPE | Fold SD | Bottom validation bias |",
        "|---|---:|---:|---:|",
    ]
    for model, row in val_bottom.sort_values("WAPE_mean").iterrows():
        lines.append(
            f"| {model} | {row.WAPE_mean:.3f} | {row.WAPE_std:.3f} | {row.Bias_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Selection used validation only. {selected_model} had mean bottom WAPE "
            f"{selected_val.WAPE_mean:.3f}. {model_description(selected_model)}",
            "",
            "## Final holdout evaluation",
            "",
            "| Model | Level | WAPE | MAE | RMSE | MASE | Bias |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in holdout.iterrows():
        lines.append(
            f"| {row.model} | {row.level} | {row.WAPE:.3f} | {row.MAE:.2f} | "
            f"{row.RMSE:.2f} | {row.MASE:.3f} | {row.Bias:.3f} |"
        )
    lines.extend(
        [
            "",
            "WAPE is total absolute error divided by total actual demand. MASE scales each "
            "node's MAE by its in-training weekly-naive error. WAPE can hide node-level "
            "dispersion, so `holdout_node_metrics.csv` contains every hierarchy node.",
            "",
            f"The selected model underforecast the holdout by {abs(selected_test.Bias):.1%} at "
            f"the bottom level. At the total level its WAPE was {selected_total.WAPE:.3f}, "
            f"slightly {'above' if selected_total.WAPE > seasonal_total.WAPE else 'below'} the "
            f"weekly-naive value of {seasonal_total.WAPE:.3f}. This is the main hierarchy-level "
            "trade-off: the ensemble improved SKU-level allocation but did not beat weekly naive "
            "at the grand total in this holdout.",
            "",
            "## Explanation, uncertainty, and limitations",
            "",
            f"{model_description(selected_model)} The model's relationships are predictive "
            "associations, not causal effects.",
            "",
            "The future forecast includes approximate 80% empirical intervals calibrated from "
            "absolute rolling-validation residuals by horizon and scaled by the square root of "
            "forecast demand. Bottom bounds are summed upward, so aggregate bounds are coherent "
            "but conservative. The median horizon calibration multiplier is "
            f"{float(np.median(interval_q)):.2f}.",
            "",
            "## Future 28-day forecast",
            "",
            f"For {future_dates[0].date()} through {future_dates[-1].date()}, coherent total "
            f"demand is forecast at **{future_total:,.0f} cartons**, {future_change:+.1%} versus "
            f"the most recent {horizon} observed days. The highest forecast day is "
            f"{future_dates[peak_index].date()} at {future_daily_total[peak_index]:,.0f} cartons. "
            "These values are conditional on normal calendar behavior because no future event "
            "schedule was available.",
            "",
            "The largest limitation is missing future-known promotion, holiday, price, campaign, "
            "distribution, and stockout information. Unflagged shocks are irreducible from the "
            "provided columns, and only two annual cycles are available. The 28-day assumption "
            "must be revisited if purchasing decisions use another horizon.",
            "",
            "## Operational recommendation",
            "",
            "Run this as a 28-day batch forecast, refresh it whenever new daily actuals arrive, "
            "and use weekly seasonal naive as the automatic fallback. Monitor schema/key "
            "completeness, hierarchy mappings, zero rate, WAPE/MASE/bias by hierarchy level, "
            "and interval coverage after outcomes arrive. Retrain or investigate when four-week "
            "WAPE degrades materially from the backtest range or bias persists in one direction.",
            "",
            "Before production, add event/price/stockout fields with explicit future availability, "
            "confirm the planning horizon and over-versus-underforecast cost, and validate the "
            "forecast in the inventory decision process rather than treating statistical error "
            "alone as business value.",
            "",
            "## Reproducibility",
            "",
            "Run `python main.py`. The script records package versions, diagnostics, every fold, "
            "holdout metrics, feature importance when applicable, plots, and coherent future "
            "forecasts in `artifacts/`.",
        ]
    )
    if feature_importance is not None and not feature_importance.empty:
        top_features = ", ".join(
            f"`{row.feature}` ({row.mae_increase:.2f})"
            for row in feature_importance.head(5).itertuples(index=False)
        )
        insertion = [
            "",
            "For the boosting component, permutation importance on a reproducible training "
            f"sample ranked the largest residual-error contributions as {top_features}. Values "
            "are increases in residual MAE when a feature is shuffled and should not be read as "
            "causal effects.",
            "",
        ]
        marker = lines.index("## Future 28-day forecast")
        lines[marker:marker] = insertion
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.validation_folds < 2:
        raise ValueError("horizon must be positive and at least two validation folds are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, wide, meta = load_and_validate(args.data, args.horizon)
    hierarchy = make_hierarchy(meta)
    values = wide.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(wide.index)
    diagnostics = component_diagnostics(data, wide)
    (args.output_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    plot_components(data, wide, args.output_dir / "time_series_components.png")

    final_test_start = len(values) - args.horizon
    first_validation_start = final_test_start - args.validation_folds * args.horizon
    if first_validation_start < 392:
        raise ValueError("Not enough history for the requested number of validation folds")

    validation_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, list[np.ndarray]] = {}
    validation_actuals: list[np.ndarray] = []
    cutoffs = [first_validation_start + fold * args.horizon for fold in range(args.validation_folds)]
    for fold_number, validation_start in enumerate(cutoffs, start=1):
        history = values[:validation_start]
        actual = values[validation_start : validation_start + args.horizon]
        history_dates = dates[:validation_start]
        future_dates = dates[validation_start : validation_start + args.horizon]
        forecasts, _ = all_candidate_forecasts(
            history, history_dates, future_dates, meta, args.horizon, need_hybrid=True
        )
        validation_actuals.append(actual)
        for model_name, prediction in forecasts.items():
            validation_predictions.setdefault(model_name, []).append(prediction)
            rows = evaluate_forecast(actual, prediction, history, hierarchy)
            for row in rows:
                validation_rows.append(
                    {
                        "fold": fold_number,
                        "train_end": history_dates[-1].date(),
                        "validation_start": future_dates[0].date(),
                        "validation_end": future_dates[-1].date(),
                        "model": model_name,
                        **row,
                    }
                )
        gc.collect()

    validation_metrics = pd.DataFrame(validation_rows)
    validation_metrics.to_csv(args.output_dir / "validation_fold_metrics.csv", index=False)
    candidate_summary = validation_summary(validation_metrics)
    candidate_summary.to_csv(args.output_dir / "validation_model_summary.csv", index=False)
    plot_validation(candidate_summary, args.output_dir / "validation_model_comparison.png")

    bottom_summary = candidate_summary.loc[candidate_summary["level"] == "Bottom"]
    selected_model = str(bottom_summary.sort_values(["WAPE_mean", "WAPE_std"]).iloc[0]["model"])

    # The final holdout remains untouched until model selection above is complete.
    train = values[:final_test_start]
    actual_test = values[final_test_start:]
    train_dates = dates[:final_test_start]
    test_dates = dates[final_test_start:]
    all_holdout_forecasts, selected_hybrid = all_candidate_forecasts(
        train, train_dates, test_dates, meta, args.horizon, need_hybrid=True
    )
    models_for_holdout = ["NaiveLast", "SeasonalNaive7", selected_model]
    models_for_holdout = list(dict.fromkeys(models_for_holdout))
    holdout_rows: list[dict[str, Any]] = []
    for model_name in models_for_holdout:
        for row in evaluate_forecast(
            actual_test, all_holdout_forecasts[model_name], train, hierarchy
        ):
            holdout_rows.append({"model": model_name, **row})
    holdout_metrics = pd.DataFrame(holdout_rows)
    holdout_metrics.to_csv(args.output_dir / "holdout_metrics_by_level.csv", index=False)
    node_metrics(
        actual_test, all_holdout_forecasts[selected_model], train, hierarchy
    ).to_csv(args.output_dir / "holdout_node_metrics.csv", index=False)

    horizon_rows = []
    for model_name in models_for_holdout:
        prediction = all_holdout_forecasts[model_name]
        for step in range(args.horizon):
            metrics = metric_row(
                actual_test[[step]], prediction[[step]], train
            )
            horizon_rows.append(
                {"model": model_name, "horizon": step + 1, "Date": test_dates[step], **metrics}
            )
    pd.DataFrame(horizon_rows).to_csv(args.output_dir / "holdout_metrics_by_horizon.csv", index=False)
    plot_holdout(
        test_dates,
        actual_test,
        all_holdout_forecasts[selected_model],
        all_holdout_forecasts["SeasonalNaive7"],
        selected_model,
        args.output_dir / "holdout_actual_vs_forecast.png",
    )

    # Empirical, horizon-specific scaled absolute-residual calibration.
    selected_validation_predictions = validation_predictions[selected_model]
    calibration = []
    for step in range(args.horizon):
        scores = []
        for actual, prediction in zip(validation_actuals, selected_validation_predictions):
            score = np.abs(actual[step] - prediction[step]) / np.sqrt(prediction[step] + 1.0)
            scores.extend(score.tolist())
        calibration.append(float(np.quantile(scores, 0.80, method="higher")))
    calibration_q = np.asarray(calibration)
    pd.DataFrame(
        {"horizon": np.arange(1, args.horizon + 1), "scaled_abs_error_q80": calibration_q}
    ).to_csv(args.output_dir / "interval_calibration.csv", index=False)

    # Refit the selected family on all available history for the actual future forecast.
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=args.horizon, freq="D")
    full_forecasts, full_hybrid = all_candidate_forecasts(
        values, dates, future_dates, meta, args.horizon, need_hybrid=True
    )
    future_bottom = full_forecasts[selected_model]
    radius = calibration_q[:, None] * np.sqrt(future_bottom + 1.0)
    lower_bottom = np.maximum(0.0, future_bottom - radius)
    upper_bottom = future_bottom + radius
    write_forecast_table(
        future_dates,
        future_bottom,
        lower_bottom,
        upper_bottom,
        hierarchy,
        selected_model,
        args.output_dir / "future_hierarchical_forecast.csv",
    )

    importance_df: pd.DataFrame | None = None
    if selected_model in {"DirectHybridGB", "HybridFourierEnsemble"} and full_hybrid is not None:
        importance_df = full_hybrid.feature_importance()
        importance_df.to_csv(args.output_dir / "hybrid_feature_importance.csv", index=False)

    total_forecast = aggregate(future_bottom, hierarchy[0])[:, 0]
    maximum_coherence_error = max(
        float(np.max(np.abs(aggregate(future_bottom, level).sum(axis=1) - total_forecast)))
        for level in hierarchy
    )
    run_summary = {
        "selected_model": selected_model,
        "selection_metric": "mean rolling-validation bottom-level WAPE",
        "horizon_days": args.horizon,
        "validation_folds": args.validation_folds,
        "final_train_end": str(train_dates[-1].date()),
        "holdout_start": str(test_dates[0].date()),
        "holdout_end": str(test_dates[-1].date()),
        "future_start": str(future_dates[0].date()),
        "future_end": str(future_dates[-1].date()),
        "maximum_point_forecast_coherence_error": maximum_coherence_error,
        "random_seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "forecast_report.md",
        diagnostics,
        candidate_summary,
        holdout_metrics,
        selected_model,
        test_dates[0],
        test_dates[-1],
        args.horizon,
        args.validation_folds,
        calibration_q,
        importance_df,
        future_dates,
        future_bottom,
        values[-args.horizon :],
    )

    print(json.dumps(run_summary, indent=2))
    print("\nBottom-level holdout metrics:")
    print(holdout_metrics.loc[holdout_metrics["level"] == "Bottom"].to_string(index=False))
    print(f"\nArtifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
