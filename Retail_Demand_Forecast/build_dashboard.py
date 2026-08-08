"""Build a self-contained interactive retail-demand forecast dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import main as forecast_pipeline


MODEL_NAME = "HybridFourierEnsemble"
ESTIMATE_START_INDEX = 84


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "retail_timeseries_2yr.csv",
    )
    parser.add_argument(
        "--forecast-artifact",
        type=Path,
        default=project_dir / "artifacts" / "future_hierarchical_forecast.csv",
    )
    parser.add_argument(
        "--interval-calibration",
        type=Path,
        default=project_dir / "artifacts" / "interval_calibration.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "artifacts" / "retail_demand_dashboard.html",
    )
    return parser.parse_args()


def safe_json(value: object) -> str:
    """Serialize for an inline script without allowing a closing script tag."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def fitted_ensemble(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    meta: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, forecast_pipeline.DirectHybridForecaster]:
    hybrid = forecast_pipeline.DirectHybridForecaster(horizon).fit(values, dates, meta)
    frames = []
    for target_index in range(ESTIMATE_START_INDEX, len(values)):
        frames.append(
            hybrid._one_origin_features(  # Reuses the production feature contract.
                values,
                dates,
                meta,
                target_index - 1,
                target_index,
                1,
            )
        )
    features = hybrid._cast_categories(pd.concat(frames, ignore_index=True), fit=False)
    hybrid_values = np.maximum(
        0.0,
        features["seasonal_naive_7"].to_numpy() + hybrid.model.predict(features),
    ).reshape(len(values) - ESTIMATE_START_INDEX, len(meta))

    time_index = np.arange(len(values))
    harmonic = Ridge(alpha=1.0, fit_intercept=False).fit(
        forecast_pipeline.fourier_design(time_index), np.log1p(values)
    )
    harmonic_values = np.maximum(
        0.0,
        np.expm1(harmonic.predict(forecast_pipeline.fourier_design(time_index))),
    )
    estimate = np.full_like(values, np.nan, dtype=float)
    estimate[ESTIMATE_START_INDEX:] = 0.5 * (
        hybrid_values + harmonic_values[ESTIMATE_START_INDEX:]
    )
    return estimate, hybrid


def future_ensemble(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    meta: pd.DataFrame,
    hybrid: forecast_pipeline.DirectHybridForecaster,
    calibration_path: Path,
    horizon: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    hybrid_forecast = hybrid.forecast(values, dates, future_dates, meta)
    harmonic_forecast = forecast_pipeline.fourier_trend_forecast(values, horizon)
    point = np.maximum(0.0, 0.5 * (hybrid_forecast + harmonic_forecast))
    calibration = pd.read_csv(calibration_path)["scaled_abs_error_q80"].to_numpy(float)
    if len(calibration) != horizon:
        raise ValueError("Interval calibration does not match the dashboard horizon")
    radius = calibration[:, None] * np.sqrt(point + 1.0)
    return future_dates, point, np.maximum(0.0, point - radius), point + radius


def verify_forecast_artifact(
    artifact_path: Path,
    future_dates: pd.DatetimeIndex,
    meta: pd.DataFrame,
    point: np.ndarray,
) -> None:
    artifact = pd.read_csv(artifact_path, parse_dates=["Date"])
    artifact = artifact.loc[artifact["level"] == "Bottom"].copy()
    expected_nodes = meta.astype(str).agg(" / ".join, axis=1).tolist()
    pivot = artifact.pivot(index="Date", columns="node_id", values="forecast")
    pivot = pivot.reindex(index=future_dates, columns=expected_nodes)
    if pivot.isna().any().any():
        raise ValueError("Forecast artifact is missing dashboard bottom-series rows")
    max_difference = float(np.max(np.abs(pivot.to_numpy() - point)))
    if max_difference > 1e-6:
        raise ValueError(f"Rebuilt forecasts differ from the validated artifact by {max_difference}")


def dashboard_payload(
    values: np.ndarray,
    estimate: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    meta: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    series = []
    for index, identifiers in meta.iterrows():
        fitted = [
            None if not np.isfinite(value) else round(float(value), 2)
            for value in estimate[:, index]
        ]
        series.append(
            {
                "country": str(identifiers["Country"]),
                "region": str(identifiers["Region"]),
                "chain": str(identifiers["Chain"]),
                "sku": str(identifiers["Parent_SKU"]),
                "actual": values[:, index].astype(int).tolist(),
                "estimate": fitted,
                "forecast": np.round(point[:, index], 2).tolist(),
                "lower": np.round(lower[:, index], 2).tolist(),
                "upper": np.round(upper[:, index], 2).tolist(),
            }
        )
    metadata = {
        "model": MODEL_NAME,
        "horizon": len(future_dates),
        "historyDates": dates.strftime("%Y-%m-%d").tolist(),
        "futureDates": future_dates.strftime("%Y-%m-%d").tolist(),
        "estimateStart": dates[ESTIMATE_START_INDEX].strftime("%Y-%m-%d"),
        "bottomSeries": len(meta),
    }
    return series, metadata


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    template_path = project_dir / "dashboard_template.html"
    chart_path = project_dir / "node_modules" / "chart.js" / "dist" / "chart.umd.min.js"
    if not chart_path.exists():
        raise FileNotFoundError("Chart.js is missing. Run `npm install` in the project directory.")

    _, wide, meta = forecast_pipeline.load_and_validate(args.data, horizon=28)
    values = wide.to_numpy(float)
    dates = pd.DatetimeIndex(wide.index)
    estimate, hybrid = fitted_ensemble(values, dates, meta, horizon=28)
    future_dates, point, lower, upper = future_ensemble(
        values, dates, meta, hybrid, args.interval_calibration, horizon=28
    )
    verify_forecast_artifact(args.forecast_artifact, future_dates, meta, point)
    series, metadata = dashboard_payload(
        values, estimate, point, lower, upper, dates, future_dates, meta
    )

    html = template_path.read_text(encoding="utf-8")
    html = html.replace("__CHART_JS__", chart_path.read_text(encoding="utf-8"))
    html = html.replace("__SERIES_JSON__", safe_json(series))
    html = html.replace("__META_JSON__", safe_json(metadata))
    if "__CHART_JS__" in html or "__SERIES_JSON__" in html or "__META_JSON__" in html:
        raise RuntimeError("Dashboard template substitution was incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    summary = {
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "history_dates": len(dates),
        "future_dates": len(future_dates),
        "bottom_series": len(meta),
        "estimate_start": metadata["estimateStart"],
        "model": MODEL_NAME,
        "embedded_chart_js": True,
        "network_requests": False,
    }
    (args.output.parent / "dashboard_build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
