#!/usr/bin/env python3
"""Compute common point-forecast metrics from numeric arrays or a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def forecast_metrics(
    actual, forecast, *, epsilon: float = 1e-12
) -> dict[str, float | int | None]:
    """Return metrics using bias = mean(forecast - actual).

    MAPE excludes observations with |actual| <= epsilon. WAPE is null when the
    total absolute actual is zero. Non-finite pairs are excluded.
    """
    actual_values = [float(value) for value in actual]
    forecast_values = [float(value) for value in forecast]
    if len(actual_values) != len(forecast_values):
        raise ValueError("actual and forecast must have identical shapes")
    pairs = [
        (actual_value, forecast_value)
        for actual_value, forecast_value in zip(actual_values, forecast_values)
        if math.isfinite(actual_value) and math.isfinite(forecast_value)
    ]
    if not pairs:
        raise ValueError("no finite actual/forecast pairs")

    errors = [forecast_value - actual_value for actual_value, forecast_value in pairs]
    absolute_errors = [abs(error) for error in errors]
    percentage_errors = [
        abs(forecast_value - actual_value) / abs(actual_value)
        for actual_value, forecast_value in pairs
        if abs(actual_value) > epsilon
    ]
    smape_terms = [
        2.0
        * abs(forecast_value - actual_value)
        / (abs(actual_value) + abs(forecast_value))
        if abs(actual_value) + abs(forecast_value) > epsilon
        else 0.0
        for actual_value, forecast_value in pairs
    ]
    denominator = sum(abs(actual_value) for actual_value, _ in pairs)
    count = len(pairs)
    return {
        "n": count,
        "mae": sum(absolute_errors) / count,
        "rmse": math.sqrt(sum(error**2 for error in errors) / count),
        "mape_percent": sum(percentage_errors) / len(percentage_errors) * 100
        if percentage_errors
        else None,
        "smape_percent": sum(smape_terms) / count * 100,
        "wape_percent": sum(absolute_errors) / denominator * 100
        if denominator > epsilon
        else None,
        "bias": sum(errors) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--actual", default="actual", help="Actual-value column")
    parser.add_argument("--forecast", default="forecast", help="Forecast-value column")
    arguments = parser.parse_args()
    with arguments.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no data rows")
    missing = {arguments.actual, arguments.forecast}.difference(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    print(
        json.dumps(
            forecast_metrics(
                [row[arguments.actual] for row in rows],
                [row[arguments.forecast] for row in rows],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
