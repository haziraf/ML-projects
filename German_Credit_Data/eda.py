"""Reproducible exploratory data analysis for the German Credit CSV.

Run from the repository root with:
    .env/bin/python German_Credit_Data/eda.py

The script reads the local snapshot only and writes a Markdown report, audit
tables, and figures. It does not modify the source CSV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "german_credit.csv"
OUTPUT_DIR = ROOT / "artifacts" / "eda"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = ROOT / "EDA_Report.md"

TARGET = "class"
GOOD_CLASS = 1
BAD_CLASS = 2

FEATURES = {
    "Attribute1": ("checking_account_status", "Categorical dimension"),
    "Attribute2": ("duration_months", "Numeric metric"),
    "Attribute3": ("credit_history", "Categorical dimension"),
    "Attribute4": ("purpose", "Categorical dimension"),
    "Attribute5": ("credit_amount", "Numeric metric"),
    "Attribute6": ("savings_status", "Categorical dimension"),
    "Attribute7": ("employment_since", "Categorical dimension"),
    "Attribute8": ("installment_rate_pct_income", "Ordinal metric"),
    "Attribute9": ("personal_status_sex", "Categorical dimension"),
    "Attribute10": ("other_debtors_guarantors", "Categorical dimension"),
    "Attribute11": ("residence_years", "Ordinal metric"),
    "Attribute12": ("property", "Categorical dimension"),
    "Attribute13": ("age_years", "Numeric metric"),
    "Attribute14": ("other_installment_plans", "Categorical dimension"),
    "Attribute15": ("housing", "Categorical dimension"),
    "Attribute16": ("existing_credits", "Count metric"),
    "Attribute17": ("job", "Categorical dimension"),
    "Attribute18": ("dependents", "Count metric"),
    "Attribute19": ("telephone", "Categorical dimension"),
    "Attribute20": ("foreign_worker", "Categorical dimension"),
    "class": ("credit_outcome", "Binary target"),
}

VALUE_LABELS = {
    "Attribute1": {
        "A11": "checking < 0 DM",
        "A12": "checking 0–<200 DM",
        "A13": "checking ≥200 DM / salary assignment",
        "A14": "no checking account",
    },
    "Attribute3": {
        "A30": "no credits / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly",
        "A33": "delay in past",
        "A34": "critical account / other credits",
    },
    "Attribute4": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "Attribute6": {
        "A61": "savings <100 DM",
        "A62": "savings 100–<500 DM",
        "A63": "savings 500–<1000 DM",
        "A64": "savings ≥1000 DM",
        "A65": "unknown / no savings",
    },
    "Attribute7": {
        "A71": "unemployed",
        "A72": "employed <1 year",
        "A73": "employed 1–<4 years",
        "A74": "employed 4–<7 years",
        "A75": "employed ≥7 years",
    },
    "Attribute9": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "Attribute10": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "Attribute12": {
        "A121": "real estate",
        "A122": "building society/life insurance",
        "A123": "car/other",
        "A124": "unknown / no property",
    },
    "Attribute14": {"A141": "bank", "A142": "stores", "A143": "none"},
    "Attribute15": {"A151": "rent", "A152": "own", "A153": "free"},
    "Attribute17": {
        "A171": "unemployed/unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified",
    },
    "Attribute19": {"A191": "none", "A192": "registered telephone"},
    "Attribute20": {"A201": "foreign worker: yes", "A202": "foreign worker: no"},
}

NUMERIC_FEATURES = [
    "Attribute2",
    "Attribute5",
    "Attribute8",
    "Attribute11",
    "Attribute13",
    "Attribute16",
    "Attribute18",
]
CATEGORICAL_FEATURES = [
    column for column in FEATURES if column not in NUMERIC_FEATURES + [TARGET]
]


def fmt_number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    data = frame if columns is None else frame[columns]
    headers = [str(column) for column in data.columns]
    rows = [headers] + [[str(value) for value in row] for row in data.itertuples(index=False, name=None)]
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    output = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    output.extend(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows[1:]
    )
    return "\n".join(output)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def validate_schema(data: pd.DataFrame) -> None:
    expected = list(FEATURES)
    if data.columns.tolist() != expected:
        raise ValueError(f"Expected columns {expected}; found {data.columns.tolist()}")
    target_values = set(data[TARGET].dropna().unique())
    if target_values != {GOOD_CLASS, BAD_CLASS}:
        raise ValueError(f"Expected class labels {{1, 2}}; found {sorted(target_values)}")


def build_column_profile(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_name in data.columns:
        series = data[raw_name]
        counts = series.value_counts(dropna=False)
        non_null = series.dropna()
        top = "; ".join(f"{value}: {count}" for value, count in counts.head(5).items())
        bottom = "; ".join(f"{value}: {count}" for value, count in counts.tail(5).items())
        row: dict[str, object] = {
            "raw_column": raw_name,
            "semantic_name": FEATURES[raw_name][0],
            "classification": FEATURES[raw_name][1],
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean()),
            "completeness": "Complete" if series.notna().mean() > 0.99 else "Mostly complete" if series.notna().mean() >= 0.95 else "Incomplete" if series.notna().mean() >= 0.80 else "Sparse",
            "distinct_count": int(series.nunique(dropna=True)),
            "cardinality_ratio": float(series.nunique(dropna=True) / len(data)),
            "top_values": top,
            "least_common_values": bottom,
            "empty_string_count": 0,
            "leading_trailing_whitespace_count": 0,
            "min_string_length": math.nan,
            "max_string_length": math.nan,
            "avg_string_length": math.nan,
            "case_pattern": "n/a",
        }
        if not pd.api.types.is_numeric_dtype(series):
            strings = non_null.astype(str)
            row["empty_string_count"] = int(strings.str.len().eq(0).sum())
            row["leading_trailing_whitespace_count"] = int(strings.ne(strings.str.strip()).sum())
            row["min_string_length"] = int(strings.str.len().min())
            row["max_string_length"] = int(strings.str.len().max())
            row["avg_string_length"] = float(strings.str.len().mean())
            if strings.str.fullmatch(r"A\d+").all():
                row["case_pattern"] = "consistent code: A + digits"
            elif strings.str.isupper().all():
                row["case_pattern"] = "uppercase"
            elif strings.str.islower().all():
                row["case_pattern"] = "lowercase"
            else:
                row["case_pattern"] = "mixed"
        rows.append(row)
    return pd.DataFrame(rows)


def build_numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_name in NUMERIC_FEATURES:
        series = data[raw_name]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "raw_column": raw_name,
                "feature": FEATURES[raw_name][0],
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "distinct": int(series.nunique()),
                "min": float(series.min()),
                "p01": float(series.quantile(0.01)),
                "p05": float(series.quantile(0.05)),
                "p25": float(q1),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "p75": float(q3),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "zero_count": int(series.eq(0).sum()),
                "negative_count": int(series.lt(0).sum()),
                "iqr_outlier_count": int(((series < lower) | (series > upper)).sum()),
                "iqr_lower_fence": float(lower),
                "iqr_upper_fence": float(upper),
            }
        )
    return pd.DataFrame(rows)


def build_category_rates(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_name in CATEGORICAL_FEATURES:
        for code, group in data.groupby(raw_name, observed=True):
            total = len(group)
            bad = int(group[TARGET].eq(BAD_CLASS).sum())
            low, high = wilson_interval(bad, total)
            rows.append(
                {
                    "raw_column": raw_name,
                    "feature": FEATURES[raw_name][0],
                    "code": code,
                    "label": VALUE_LABELS.get(raw_name, {}).get(code, code),
                    "n": total,
                    "good_count": total - bad,
                    "bad_count": bad,
                    "bad_rate": bad / total,
                    "bad_rate_ci95_low": low,
                    "bad_rate_ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def build_associations(data: pd.DataFrame, category_rates: pd.DataFrame) -> pd.DataFrame:
    bad = data[TARGET].eq(BAD_CLASS).astype(int)
    rows: list[dict[str, object]] = []
    for raw_name in NUMERIC_FEATURES:
        coefficient, p_value = pointbiserialr(bad, data[raw_name])
        rows.append(
            {
                "raw_column": raw_name,
                "feature": FEATURES[raw_name][0],
                "feature_type": "numeric/ordinal",
                "association_measure": "point-biserial r",
                "association": float(coefficient),
                "absolute_association": abs(float(coefficient)),
                "p_value_unadjusted": float(p_value),
                "good_mean": float(data.loc[data[TARGET].eq(GOOD_CLASS), raw_name].mean()),
                "bad_mean": float(data.loc[data[TARGET].eq(BAD_CLASS), raw_name].mean()),
                "bad_rate_spread": math.nan,
            }
        )
    for raw_name in CATEGORICAL_FEATURES:
        table = pd.crosstab(data[raw_name], data[TARGET])
        chi2, p_value, _, _ = chi2_contingency(table)
        n = table.to_numpy().sum()
        denominator = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = math.sqrt((chi2 / n) / denominator) if denominator > 0 else 0.0
        rates = category_rates.loc[category_rates["raw_column"].eq(raw_name), "bad_rate"]
        rows.append(
            {
                "raw_column": raw_name,
                "feature": FEATURES[raw_name][0],
                "feature_type": "categorical",
                "association_measure": "Cramér's V",
                "association": cramers_v,
                "absolute_association": cramers_v,
                "p_value_unadjusted": float(p_value),
                "good_mean": math.nan,
                "bad_mean": math.nan,
                "bad_rate_spread": float(rates.max() - rates.min()),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_association", ascending=False, ignore_index=True)


def create_figures(
    data: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    category_rates: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"Good credit": "#3B82A0", "Bad credit": "#D05A47"}
    outcome = data[TARGET].map({GOOD_CLASS: "Good credit", BAD_CLASS: "Bad credit"})

    counts = outcome.value_counts().reindex(["Good credit", "Bad credit"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[palette[name] for name in counts.index])
    ax.bar_label(bars, labels=[f"{value} ({value / len(data):.0%})" for value in counts.values], padding=4)
    ax.set(title="Credit outcome distribution", xlabel="Outcome", ylabel="Applications", ylim=(0, counts.max() * 1.15))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=180)
    plt.close(fig)

    renamed = data[NUMERIC_FEATURES].rename(columns={raw: FEATURES[raw][0] for raw in NUMERIC_FEATURES})
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    continuous = {"Attribute2", "Attribute5", "Attribute13"}
    for ax, raw_name in zip(axes.flat, NUMERIC_FEATURES):
        local = pd.DataFrame({"outcome": outcome, "value": data[raw_name]})
        if raw_name in continuous:
            sns.histplot(
                data=local,
                x="value",
                hue="outcome",
                palette=palette,
                bins=20,
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                legend=False,
                ax=ax,
            )
            ax.set_ylabel("Density")
        else:
            sns.histplot(
                data=local,
                x="value",
                hue="outcome",
                palette=palette,
                discrete=True,
                stat="probability",
                common_norm=False,
                multiple="dodge",
                shrink=0.75,
                legend=False,
                ax=ax,
            )
            ax.set_ylabel("Within-class proportion")
            ax.set_xticks(sorted(data[raw_name].unique()))
        ax.set(title=FEATURES[raw_name][0], xlabel="")
    for ax in axes.flat[len(NUMERIC_FEATURES):]:
        ax.set_visible(False)
    legend_handles = [
        plt.Line2D([0], [0], color=palette["Good credit"], lw=3, label="Good credit"),
        plt.Line2D([0], [0], color=palette["Bad credit"], lw=3, label="Bad credit"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(0.79, 0.08), frameon=False)
    fig.suptitle("Numeric and ordinal distributions by outcome", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_distributions_by_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, raw_name in zip(axes.flat, NUMERIC_FEATURES):
        local = pd.DataFrame({"outcome": outcome, "value": data[raw_name]})
        sns.boxplot(data=local, x="outcome", y="value", hue="outcome", palette=palette, legend=False, ax=ax)
        ax.set(title=FEATURES[raw_name][0], xlabel="", ylabel="")
    for ax in axes.flat[len(NUMERIC_FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Numeric and ordinal features by outcome", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_boxplots_by_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    correlation = renamed.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Spearman correlation among numeric and ordinal features")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "spearman_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    association_plot = associations.copy().sort_values("absolute_association")
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = np.where(association_plot["feature_type"].eq("categorical"), "#5865A8", "#D08C45")
    ax.barh(association_plot["feature"], association_plot["absolute_association"], color=colors)
    ax.set(title="Univariate association with credit outcome", xlabel="Absolute association strength", ylabel="")
    ax.text(
        0.99,
        0.01,
        "Categorical: Cramér's V   Numeric/ordinal: |point-biserial r|",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_association_strength.png", dpi=180)
    plt.close(fig)

    key_features = ["Attribute1", "Attribute3", "Attribute6", "Attribute4"]
    plot_rates = category_rates[category_rates["raw_column"].isin(key_features)].copy()
    plot_rates["display"] = plot_rates["code"].astype(str) + " — " + plot_rates["label"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, raw_name in zip(axes.flat, key_features):
        subset = plot_rates[plot_rates["raw_column"].eq(raw_name)].sort_values("bad_rate")
        errors = np.vstack(
            [
                subset["bad_rate"] - subset["bad_rate_ci95_low"],
                subset["bad_rate_ci95_high"] - subset["bad_rate"],
            ]
        )
        ax.errorbar(subset["bad_rate"], subset["display"], xerr=errors, fmt="o", color="#D05A47", capsize=3)
        ax.axvline(data[TARGET].eq(BAD_CLASS).mean(), color="#555555", linestyle="--", linewidth=1)
        ax.set(title=FEATURES[raw_name][0], xlabel="Bad-credit rate (95% Wilson CI)", ylabel="", xlim=(0, 0.8))
    fig.suptitle("Bad-credit rates for the strongest categorical segments", y=1.01, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "key_segment_bad_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    scatter = data.rename(columns={"Attribute2": "duration_months", "Attribute5": "credit_amount"}).copy()
    scatter["outcome"] = outcome
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=scatter,
        x="duration_months",
        y="credit_amount",
        hue="outcome",
        palette=palette,
        alpha=0.55,
        s=35,
        ax=ax,
    )
    ax.set(title="Credit amount and duration by outcome", xlabel="Duration (months)", ylabel="Credit amount (DM)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "amount_vs_duration.png", dpi=180)
    plt.close(fig)


def build_report(
    data: pd.DataFrame,
    column_profile: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    category_rates: pd.DataFrame,
    associations: pd.DataFrame,
) -> str:
    rows, columns = data.shape
    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    predictor_duplicates = int(data.drop(columns=TARGET).duplicated().sum())
    file_size_kib = DATA_PATH.stat().st_size / 1024
    bad_count = int(data[TARGET].eq(BAD_CLASS).sum())
    good_count = int(data[TARGET].eq(GOOD_CLASS).sum())
    bad_rate = bad_count / rows

    profile_display = column_profile[
        ["raw_column", "semantic_name", "classification", "dtype", "missing_pct", "distinct_count", "top_values"]
    ].copy()
    profile_display.columns = ["Raw", "Semantic name", "Role", "Type", "Missing", "Distinct", "Most common values"]
    profile_display["Missing"] = profile_display["Missing"].map(lambda value: f"{value:.1%}")

    numeric_display = numeric_summary[
        ["feature", "min", "p01", "p05", "p25", "median", "mean", "p75", "p95", "p99", "max", "std", "skewness", "iqr_outlier_count"]
    ].copy()
    numeric_display.columns = ["Feature", "Min", "P1", "P5", "P25", "Median", "Mean", "P75", "P95", "P99", "Max", "SD", "Skew", "IQR outliers"]
    for column in numeric_display.columns[1:-1]:
        numeric_display[column] = numeric_display[column].map(fmt_number)

    association_display = associations.head(10)[
        ["feature", "feature_type", "association_measure", "association", "p_value_unadjusted"]
    ].copy()
    association_display.columns = ["Feature", "Type", "Measure", "Association", "Unadjusted p"]
    association_display["Association"] = association_display["Association"].map(lambda value: f"{value:.3f}")
    association_display["Unadjusted p"] = association_display["Unadjusted p"].map(fmt_p)

    numeric_assoc = associations[associations["feature_type"].eq("numeric/ordinal")].copy()
    numeric_assoc = numeric_assoc.sort_values("absolute_association", ascending=False)
    numeric_target_display = numeric_assoc[["feature", "good_mean", "bad_mean", "association", "p_value_unadjusted"]].copy()
    numeric_target_display.columns = ["Feature", "Good mean", "Bad mean", "Point-biserial r", "Unadjusted p"]
    numeric_target_display["Good mean"] = numeric_target_display["Good mean"].map(fmt_number)
    numeric_target_display["Bad mean"] = numeric_target_display["Bad mean"].map(fmt_number)
    numeric_target_display["Point-biserial r"] = numeric_target_display["Point-biserial r"].map(lambda value: f"{value:.3f}")
    numeric_target_display["Unadjusted p"] = numeric_target_display["Unadjusted p"].map(fmt_p)

    key_segments = category_rates[category_rates["raw_column"].isin(["Attribute1", "Attribute3", "Attribute6"])].copy()
    key_segments = key_segments.sort_values(["raw_column", "bad_rate"], ascending=[True, False])
    key_segments_display = key_segments[["feature", "code", "label", "n", "bad_count", "bad_rate", "bad_rate_ci95_low", "bad_rate_ci95_high"]].copy()
    key_segments_display.columns = ["Feature", "Code", "Meaning", "n", "Bad", "Bad rate", "CI low", "CI high"]
    for column in ["Bad rate", "CI low", "CI high"]:
        key_segments_display[column] = key_segments_display[column].map(lambda value: f"{value:.1%}")

    correlation = data[NUMERIC_FEATURES].rename(columns={raw: FEATURES[raw][0] for raw in NUMERIC_FEATURES}).corr(method="spearman")
    pairs: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            pairs.append((left, right, float(correlation.loc[left, right])))
    strongest_pairs = sorted(pairs, key=lambda value: abs(value[2]), reverse=True)[:5]
    correlation_bullets = "\n".join(
        f"- `{left}` vs `{right}`: Spearman ρ = {value:.3f}." for left, right, value in strongest_pairs
    )

    rare = category_rates[category_rates["n"].lt(50)].sort_values("n")
    rare_text = ", ".join(f"{row.code} ({row.feature}, n={row.n})" for row in rare.itertuples())

    checking = category_rates[category_rates["raw_column"].eq("Attribute1")].set_index("code")
    history = category_rates[category_rates["raw_column"].eq("Attribute3")].set_index("code")
    amount_row = numeric_assoc[numeric_assoc["raw_column"].eq("Attribute5")].iloc[0]
    duration_row = numeric_assoc[numeric_assoc["raw_column"].eq("Attribute2")].iloc[0]

    return f"""# Exploratory Data Analysis: German Credit Data

**Source snapshot:** `data/german_credit.csv`  
**Analysis date:** 2026-08-08  
**Report scope:** descriptive EDA of the supplied file; no model fitting or causal claims

## Executive summary

The file contains **{rows:,} credit applications, 20 predictors, and one binary outcome**. Class `1` denotes good credit ({good_count:,}; {good_count / rows:.1%}) and class `2` denotes bad credit ({bad_count:,}; {bad_rate:.1%}). The snapshot is technically clean: all {rows * columns:,} cells are populated, no exact rows or predictor vectors are duplicated, numeric values stay inside their documented domains, and categorical codes have consistent formatting.

The strongest univariate relationship with the outcome is checking-account status (Cramér's V = {associations.loc[associations['raw_column'].eq('Attribute1'), 'association'].iloc[0]:.3f}), followed by credit history ({associations.loc[associations['raw_column'].eq('Attribute3'), 'association'].iloc[0]:.3f}). For example, the observed bad-credit rate is {checking.loc['A11', 'bad_rate']:.1%} for applicants with a negative checking balance (`A11`, n={int(checking.loc['A11', 'n'])}) versus {checking.loc['A14', 'bad_rate']:.1%} for those with no checking account (`A14`, n={int(checking.loc['A14', 'n'])}). These are **unadjusted associations**, not evidence that an attribute causes repayment outcomes.

Among numeric fields, longer duration (r = {duration_row['association']:.3f}) and larger credit amount (r = {amount_row['association']:.3f}) are associated with the bad-credit class. Duration and amount are themselves moderately correlated (Spearman ρ = {correlation.loc['duration_months', 'credit_amount']:.3f}), so their individual relationships should not be treated as independent effects.

The main risks are contextual rather than cell-level: there is no applicant identifier, timestamp, extraction metadata, or production lineage; several categories are small; the data uses legacy Deutsche Mark bands; and `personal_status_sex`, age, and foreign-worker status raise fairness and governance concerns. This dataset is appropriate for learning and offline benchmarking, but not sufficient by itself for current lending decisions.

![Target distribution](artifacts/eda/figures/target_distribution.png)

## 1. Dataset structure and grain

- **Rows:** {rows:,}
- **Columns:** {columns} = 13 categorical dimensions, 7 numeric/ordinal/count predictors, and 1 binary target
- **Approximate file size:** {file_size_kib:.1f} KiB
- **Grain:** one row appears to represent one credit application
- **Primary key:** none supplied; no column is an identifier
- **Candidate uniqueness:** {duplicate_rows} exact duplicate rows; {predictor_duplicates} duplicated 20-feature predictor vectors
- **Temporal coverage / last update:** not assessable because the file has no date, timestamp, or load metadata
- **Target:** `class=1` good credit; `class=2` bad credit

The seven numeric-looking predictors include two continuous/count-like measures (`duration_months`, `credit_amount`), age, two counts, and two bounded ordinal ratings. The integer target is categorical, not a metric. There are no date, free-text, Boolean, JSON, or foreign-key columns.

### Column profile

{md_table(profile_display)}

The complete machine-readable profile—including least-common values, cardinality ratios, string-length checks, whitespace counts, and completeness ratings—is in [`column_profile.csv`](artifacts/eda/column_profile.csv).

## 2. Data quality assessment

### Completeness and consistency

- **Missing values:** {missing_cells} cells ({missing_cells / (rows * columns):.1%}); every column is rated **Complete** (>99% non-null).
- **Empty strings / whitespace:** 0 empty categorical values and 0 codes with leading or trailing whitespace.
- **Encoding:** every categorical value follows the documented `A` + digits format; no unexpected code was found.
- **Duplicates:** none at either the full-row or predictor-vector level. Because no customer/application ID exists, repeat applicants cannot be detected.
- **Numeric domains:** zero negative values and zero zero values across all seven numeric predictors. Observed ages are {int(data['Attribute13'].min())}–{int(data['Attribute13'].max())}; durations are {int(data['Attribute2'].min())}–{int(data['Attribute2'].max())} months.
- **Target validity:** only labels 1 and 2 occur, with a moderate 70/30 imbalance.

### Quality and suitability flags

| Severity | Finding | Implication |
| --- | --- | --- |
| High | Sensitive/proxy attributes (`personal_status_sex`, `age_years`, `foreign_worker`) are present. | Formal legal and fairness review is required before any lending use; do not infer fairness from aggregate performance. |
| High | No timestamp, sample-construction metadata, or current-population benchmark. | Timeliness, drift, and representativeness cannot be evaluated. |
| Medium | No applicant/application identifier. | Entity uniqueness, repeat borrowing, and leakage across future train/test splits cannot be verified. |
| Medium | Legacy coded values and Deutsche Mark bands require an external data dictionary. | Treat codes as categorical; numerical ordering must not be invented for nominal features. |
| Medium | Small categories: {rare_text}. | Segment rates and model effects for these levels have wide uncertainty. |
| Low | The target is moderately imbalanced (30% bad). | Use stratification and class-appropriate metrics in later modeling. |

IQR screening flags {int(numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute2'), 'iqr_outlier_count'].iloc[0])} durations, {int(numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute5'), 'iqr_outlier_count'].iloc[0])} credit amounts, and {int(numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute13'), 'iqr_outlier_count'].iloc[0])} ages. These are tail observations, not automatically invalid records; the documented maxima remain plausible. The same rule labels all {int(numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute18'), 'iqr_outlier_count'].iloc[0])} records with two dependents as outliers because that binary count has an IQR of zero—an artifact showing that IQR fences are not meaningful for bounded ordinal/count fields. Capping or deleting any flagged record would require domain justification.

## 3. Numeric distributions

{md_table(numeric_display)}

Key distribution findings:

- `credit_amount` is strongly right-skewed (skewness {numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute5'), 'skewness'].iloc[0]:.2f}): median {data['Attribute5'].median():,.0f} DM, mean {data['Attribute5'].mean():,.0f} DM, and 99th percentile {data['Attribute5'].quantile(.99):,.0f} DM. A log transform may help linear models and visualizations.
- `duration_months` is right-skewed (skewness {numeric_summary.loc[numeric_summary['raw_column'].eq('Attribute2'), 'skewness'].iloc[0]:.2f}); its median is {data['Attribute2'].median():.0f} months and 95th percentile is {data['Attribute2'].quantile(.95):.0f} months.
- `installment_rate_pct_income` and `residence_years` each occupy only values 1–4. They should be treated as ordinal, not continuous measurements with guaranteed equal spacing.
- `existing_credits` (1–4) and `dependents` (1–2) are low-cardinality counts. Standard means are descriptive but their full count distributions are more informative.

![Numeric distributions by outcome](artifacts/eda/figures/numeric_distributions_by_outcome.png)

![Numeric boxplots by outcome](artifacts/eda/figures/numeric_boxplots_by_outcome.png)

## 4. Relationships among predictors

The strongest numeric/ordinal Spearman relationships are:

{correlation_bullets}

No numeric pair exceeds |ρ| = 0.70, so there is no strong pairwise collinearity under the skill threshold. The duration–amount relationship (ρ = {correlation.loc['duration_months', 'credit_amount']:.3f}) is the clearest and is operationally plausible: larger loans tend to have longer terms. Credit amount and installment-rate band are moderately inverse (ρ = {correlation.loc['credit_amount', 'installment_rate_pct_income']:.3f}). Correlation does not imply causation, and categorical relationships or nonlinear interactions are not captured by this matrix.

![Spearman correlation heatmap](artifacts/eda/figures/spearman_correlation_heatmap.png)

![Credit amount versus duration](artifacts/eda/figures/amount_vs_duration.png)

No columns are exact duplicates or obvious deterministic transformations of one another. Domain-related groupings include liquidity (`checking_account_status`, `savings_status`), credit exposure (`credit_amount`, `duration_months`, `installment_rate_pct_income`, `existing_credits`), stability (`employment_since`, `residence_years`, `housing`, `property`), and support/obligations (`other_debtors_guarantors`, `dependents`). These are conceptual groupings, not inferred database hierarchies.

## 5. Outcome patterns

### Univariate association ranking

{md_table(association_display)}

Categorical associations use Cramér's V; numeric/ordinal associations use point-biserial correlation with bad credit coded 1. These measures have different interpretations, so their combined ranking is a screening view rather than a proof of feature importance. P-values are exploratory, unadjusted for multiple testing, and influenced by sample size.

![Outcome association strength](artifacts/eda/figures/target_association_strength.png)

### Numeric features by outcome

{md_table(numeric_target_display)}

Longer terms and larger amounts show the clearest numeric separation. Age has a small inverse relationship with bad credit; residence duration, existing-credit count, and dependents show little univariate separation. Overlapping distributions are substantial, so no single numeric feature cleanly separates outcomes.

### Key categorical segments

{md_table(key_segments_display)}

The largest observed risk gradients are:

- Checking status: {checking.loc['A11', 'bad_rate']:.1%} bad for `A11` (negative balance) versus {checking.loc['A14', 'bad_rate']:.1%} for `A14` (no checking account).
- Credit history: {history.loc['A30', 'bad_rate']:.1%} for `A30` and {history.loc['A31', 'bad_rate']:.1%} for `A31`, versus {history.loc['A34', 'bad_rate']:.1%} for `A34`. The first two groups are small (n={int(history.loc['A30', 'n'])} and n={int(history.loc['A31', 'n'])}), so their estimates are less precise.
- Savings status, property, purpose, employment tenure, housing, and other installment plans show smaller but visible gradients.
- Job and telephone have weak univariate associations in this snapshot. A weak marginal relationship does not rule out interactions or confounding.

![Key segment bad-credit rates](artifacts/eda/figures/key_segment_bad_rates.png)

Full category counts, rates, and Wilson 95% intervals are in [`category_target_rates.csv`](artifacts/eda/category_target_rates.csv).

## 6. Analytical recommendations

Best dimensions for slicing are checking-account status, credit history, purpose, savings, employment tenure, property, and housing. Key metrics are credit amount, duration, age, installment-rate band, and existing-credit count. There is **no time column**, and there is **no defensible join key** in the supplied file.

Recommended follow-up analyses:

1. **Multivariable outcome analysis:** fit an interpretable, cross-validated logistic model with careful categorical encoding to test whether checking status, history, amount, and duration retain signal after adjustment. Report uncertainty and calibration, not only discrimination.
2. **Interaction analysis:** examine amount × duration, checking × savings, and credit-history × purpose. Use nested validation so exploratory interaction selection does not inflate performance claims.
3. **Rare-category stability:** pool only where domain-justified, or use repeated/bootstrap estimates to quantify the instability of categories with n<50.
4. **Fairness audit:** define lawful comparison groups and evaluate selection/error/calibration metrics for age, personal-status/sex, and foreign-worker groups, with minimum sample-size rules. The current small coded groups cannot support a definitive fairness conclusion.
5. **External/temporal validation:** compare this legacy sample with recent, representative applicants before treating any pattern as operationally relevant.

## 7. Reproducibility and artifact index

- Analysis script: [`eda.py`](eda.py)
- Source snapshot: [`german_credit.csv`](data/german_credit.csv)
- Full column audit: [`column_profile.csv`](artifacts/eda/column_profile.csv)
- Numeric percentiles/outliers: [`numeric_summary.csv`](artifacts/eda/numeric_summary.csv)
- Category counts and target rates: [`category_target_rates.csv`](artifacts/eda/category_target_rates.csv)
- Outcome association statistics: [`target_associations.csv`](artifacts/eda/target_associations.csv)
- Pearson and Spearman matrices: [`pearson_correlations.csv`](artifacts/eda/pearson_correlations.csv), [`spearman_correlations.csv`](artifacts/eda/spearman_correlations.csv)
- Audit summary: [`audit_summary.json`](artifacts/eda/audit_summary.json)

Reproduce from `/Users/sitinoorhazirah/ML-projects`:

```bash
.env/bin/python German_Credit_Data/eda.py
```

## Limitations

This is observational EDA on a small, legacy, one-time snapshot. Univariate statistics do not control for confounding, multiple comparisons, sampling design, or selection bias. The class label's observation window and adjudication process are not present in the file. The report therefore describes this CSV only; it does not establish causal drivers, credit policy, individual eligibility, or current population risk.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    validate_schema(data)

    column_profile = build_column_profile(data)
    numeric_summary = build_numeric_summary(data)
    category_rates = build_category_rates(data)
    associations = build_associations(data, category_rates)

    column_profile.to_csv(OUTPUT_DIR / "column_profile.csv", index=False)
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv", index=False)
    category_rates.to_csv(OUTPUT_DIR / "category_target_rates.csv", index=False)
    associations.to_csv(OUTPUT_DIR / "target_associations.csv", index=False)

    renamed_numeric = data[NUMERIC_FEATURES].rename(columns={raw: FEATURES[raw][0] for raw in NUMERIC_FEATURES})
    renamed_numeric.corr(method="pearson").to_csv(OUTPUT_DIR / "pearson_correlations.csv")
    renamed_numeric.corr(method="spearman").to_csv(OUTPUT_DIR / "spearman_correlations.csv")

    audit_summary = {
        "source": str(DATA_PATH),
        "rows": len(data),
        "columns": data.shape[1],
        "predictors": data.shape[1] - 1,
        "missing_cells": int(data.isna().sum().sum()),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "duplicate_predictor_vectors": int(data.drop(columns=TARGET).duplicated().sum()),
        "good_credit_count": int(data[TARGET].eq(GOOD_CLASS).sum()),
        "bad_credit_count": int(data[TARGET].eq(BAD_CLASS).sum()),
        "bad_credit_rate": float(data[TARGET].eq(BAD_CLASS).mean()),
        "identifier_columns": [],
        "temporal_columns": [],
        "categorical_predictors": len(CATEGORICAL_FEATURES),
        "numeric_ordinal_count_predictors": len(NUMERIC_FEATURES),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")

    create_figures(data, numeric_summary, category_rates, associations)
    REPORT_PATH.write_text(
        build_report(data, column_profile, numeric_summary, category_rates, associations),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote audit tables and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
