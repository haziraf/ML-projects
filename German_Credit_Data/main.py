"""End-to-end binary classification for UCI Statlog German Credit Data.

The script keeps a stratified test set untouched while all learned preprocessing,
feature selection, class weighting, and hyperparameter tuning occur on training
folds. Class 2 (bad credit) is mapped to positive class 1.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
from typing import Any
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


sys.path.append(str(Path(__file__).parent.parent))

from scripts.fetch_data import FetchData
from German_Credit_Data.feature_processing import (
    ATTRIBUTE_NAMES,
    prepare_inference_features,
)


RANDOM_STATE = 42
TARGET = "class"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"

# scikit-learn 1.8+ emits a deprecation warning for every logistic-regression
# fold while older supported releases still require the penalty parameter.
# Suppress only that compatibility noise; training failures remain errors.
warnings.filterwarnings(
    "ignore", message=".*'penalty' was deprecated.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message="Inconsistent values: penalty=.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="Could not find the number of physical cores.*", category=UserWarning
)

def load_dataset(path: Path, refresh: bool = False) -> pd.DataFrame:
    """Load the local CSV, fetching UCI dataset 144 only when necessary."""
    if path.exists() and not refresh:
        return pd.read_csv(path)

    fetcher = FetchData()
    dataset = fetcher.fetch_data(id=144)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def clean_and_engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply semantic names and deterministic, prediction-time-safe features."""
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r}; found {frame.columns.tolist()}")

    valid_targets = set(frame[TARGET].dropna().unique())
    if valid_targets != {1, 2}:
        raise ValueError(f"Expected target labels {{1, 2}}; found {sorted(valid_targets)}")

    X = prepare_inference_features(frame)
    y = frame[TARGET].map({1: 0, 2: 1}).astype("int8")
    return X, y


def dataset_profile(raw: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Return auditable data-quality and target-distribution statistics."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    counts = y.value_counts().sort_index()
    return {
        "rows": int(len(raw)),
        "raw_predictors": int(raw.shape[1] - 1),
        "modeled_predictors_after_engineering": int(X.shape[1]),
        "numeric_predictors": numeric,
        "categorical_predictors": categorical,
        "missing_cells_raw": int(raw.isna().sum().sum()),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "good_credit_count": int(counts.get(0, 0)),
        "bad_credit_count": int(counts.get(1, 0)),
        "bad_credit_prevalence": float(y.mean()),
    }


def save_eda(
    raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Save compact EDA tables and plots that answer data-quality questions."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "dtype": X.dtypes.astype(str),
            "missing_count": X.isna().sum(),
            "missing_pct": X.isna().mean(),
            "unique_count": X.nunique(dropna=False),
        }
    )
    summary.to_csv(output_dir / "feature_profile.csv", index_label="feature")
    X.select_dtypes(include=np.number).describe().T.to_csv(
        output_dir / "numeric_summary.csv", index_label="feature"
    )

    sns.set_theme(style="whitegrid")
    target_counts = y.map({0: "Good (class 1)", 1: "Bad (class 2)"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, color="#4472C4")
    ax.set(title="Target distribution", xlabel="Credit outcome", ylabel="Applications")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(figures / "target_distribution.png", dpi=160)
    plt.close(fig)

    numeric_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate_pct_income",
        "residence_years",
        "age_years",
        "existing_credits",
        "dependents",
    ]
    plot_data = X[numeric_cols].copy()
    plot_data["credit_outcome"] = y.map({0: "Good", 1: "Bad"})
    long = plot_data.melt(id_vars="credit_outcome", var_name="feature", value_name="value")
    grid = sns.FacetGrid(
        long,
        col="feature",
        col_wrap=3,
        hue="credit_outcome",
        sharex=False,
        sharey=False,
        height=2.4,
    )
    grid.map(sns.histplot, "value", stat="density", element="step", fill=False)
    grid.add_legend(title="Outcome")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Numeric feature distributions by outcome", y=1.02)
    grid.figure.savefig(figures / "numeric_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(grid.figure)

    # Save raw-label integrity separately so the 1/2 to 0/1 mapping is explicit.
    raw[TARGET].value_counts().sort_index().rename("count").to_csv(
        output_dir / "raw_target_counts.csv", index_label="raw_class"
    )


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric, categorical


def make_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    """Build a leakage-safe pipeline with model-based feature selection."""
    preprocessor, _, _ = make_preprocessor(X)
    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        threshold="median",
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("selector", selector), ("model", model)]
    )


def candidate_searches(X: pd.DataFrame, quick: bool) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Define a bounded, justified model comparison and tuning space."""
    selector_grid = ["median"] if quick else ["median", "1.25*mean"]
    logistic = LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
    )
    boosting = HistGradientBoostingClassifier(
        max_iter=200, early_stopping=False, random_state=RANDOM_STATE
    )

    if quick:
        return {
            "Logistic Regression": (
                make_pipeline(X, logistic),
                {
                    "selector__threshold": selector_grid,
                    "model__C": [0.5, 2.0],
                    "model__penalty": ["l1", "l2"],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "Random Forest": (
                make_pipeline(X, forest),
                {
                    "selector__threshold": selector_grid,
                    "model__max_depth": [None, 6],
                    "model__min_samples_leaf": [1, 5],
                    "model__max_features": ["sqrt"],
                    "model__class_weight": [None, "balanced_subsample"],
                },
            ),
            "Histogram Gradient Boosting": (
                make_pipeline(X, boosting),
                {
                    "selector__threshold": selector_grid,
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_leaf_nodes": [7, 15],
                    "model__min_samples_leaf": [20],
                    "model__class_weight": [None, "balanced"],
                },
            ),
        }

    return {
        "Logistic Regression": (
            make_pipeline(X, logistic),
            {
                "selector__threshold": selector_grid,
                "model__C": [0.1, 0.5, 2.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            make_pipeline(X, forest),
            {
                "selector__threshold": selector_grid,
                "model__max_depth": [None, 5, 8],
                "model__min_samples_leaf": [1, 3, 7],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "Histogram Gradient Boosting": (
            make_pipeline(X, boosting),
            {
                "selector__threshold": selector_grid,
                "model__learning_rate": [0.03, 0.07, 0.12],
                "model__max_leaf_nodes": [7, 15],
                "model__min_samples_leaf": [10, 25],
                "model__l2_regularization": [0.0, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


def probability_metrics(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def threshold_metrics(
    y_true: pd.Series | np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision_bad": float(precision_score(y_true, pred, zero_division=0)),
        "recall_bad": float(recall_score(y_true, pred, zero_division=0)),
        "f1_bad": float(f1_score(y_true, pred, zero_division=0)),
        "f2_bad": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def choose_f2_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Select a bad-credit recall-oriented threshold using training OOF predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.nanargmax(f2))
    return float(thresholds[best])


def bootstrap_interval(
    y_true: pd.Series,
    proba: np.ndarray,
    metric: str,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    """Compute a reproducible percentile CI, discarding one-class resamples."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_array = np.asarray(y_true)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_array), len(y_array))
        if np.unique(y_array[idx]).size < 2:
            continue
        if metric == "average_precision":
            values.append(float(average_precision_score(y_array[idx], proba[idx])))
        elif metric == "roc_auc":
            values.append(float(roc_auc_score(y_array[idx], proba[idx])))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def subgroup_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Audit descriptive errors for available sensitive/proxy groups."""
    audit = X_test[["personal_status_sex", "foreign_worker", "age_years"]].copy()
    audit["age_band"] = pd.cut(
        audit["age_years"], bins=[0, 25, 40, 60, np.inf], labels=["<=25", "26-40", "41-60", "61+"]
    )
    audit["actual"] = np.asarray(y_test)
    audit["probability"] = proba
    audit["prediction"] = (proba >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for feature in ["personal_status_sex", "foreign_worker", "age_band"]:
        for group, part in audit.groupby(feature, observed=True):
            if len(part) < 10:
                continue
            row: dict[str, Any] = {
                "feature": feature,
                "group": str(group),
                "n": int(len(part)),
                "bad_prevalence": float(part["actual"].mean()),
                "predicted_bad_rate": float(part["prediction"].mean()),
                "precision_bad": float(
                    precision_score(part["actual"], part["prediction"], zero_division=0)
                ),
                "recall_bad": float(
                    recall_score(part["actual"], part["prediction"], zero_division=0)
                ),
            }
            row["roc_auc"] = (
                float(roc_auc_score(part["actual"], part["probability"]))
                if part["actual"].nunique() == 2
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def save_feature_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save selected encoded features and holdout permutation importance."""
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    support = model.named_steps["selector"].get_support()
    selected_names = transformed_names[support]
    selected = pd.DataFrame({"transformed_feature": selected_names})
    selected.attrs["total_transformed_features"] = len(transformed_names)

    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        selected["model_importance"] = np.ravel(estimator.coef_)
        selected["importance_type"] = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        selected["model_importance"] = estimator.feature_importances_
        selected["importance_type"] = "impurity_importance"
    selected.to_csv(output_dir / "selected_transformed_features.csv", index=False)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    return selected, importance


def save_evaluation_plots(
    y_test: pd.Series,
    proba: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Good", "Bad"],
        yticklabels=["Good", "Bad"],
        ax=ax,
    )
    ax.set(title=f"Holdout confusion matrix (threshold={threshold:.3f})", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#4472C4")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].plot(recall, precision, color="#ED7D31")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Prevalence baseline")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "roc_pr_curves.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Create a Markdown table without requiring optional tabulate."""
    view = frame[columns].copy()
    for col in view.select_dtypes(include=np.number):
        view[col] = view[col].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    report_path: Path,
    profile: dict[str, Any],
    comparison: pd.DataFrame,
    best_name: str,
    best_params: dict[str, Any],
    threshold: float,
    test_metrics: dict[str, Any],
    ci: dict[str, tuple[float, float]],
    selected: pd.DataFrame,
    importance: pd.DataFrame,
    subgroup: pd.DataFrame,
    train_size: int,
    test_size: int,
    output_dir: Path,
) -> None:
    table_columns = [
        "model",
        "cv_average_precision_mean",
        "cv_average_precision_std",
        "test_average_precision",
        "test_roc_auc",
        "test_recall_bad_at_0_5",
    ]
    comparison_table = markdown_table(comparison, table_columns)
    top_features = ", ".join(importance.head(8)["feature"].tolist())
    total_transformed = int(selected.attrs.get("total_transformed_features", len(selected)))
    if len(selected) == total_transformed:
        selection_text = (
            f"GridSearchCV tested stronger pruning, but the best validation result retained all "
            f"{total_transformed} encoded features. This is an evidence-based no-deletion result, "
            "not a claim that every feature is equally important."
        )
    else:
        selection_text = (
            f"The tuned selector retained {len(selected)} of {total_transformed} encoded features."
        )
    weighted = any(
        "class_weight" in key and value is not None for key, value in best_params.items()
    )
    imbalance_text = (
        "GridSearchCV selected class weighting for the final model."
        if weighted
        else "GridSearchCV did not select class weighting for the final model; stratification, "
        "average-precision selection, and a recall-oriented threshold were retained."
    )
    subgroup_note = (
        f"A descriptive audit was produced for {len(subgroup)} test-set groups. Small groups and "
        "the dataset's coded, incomplete protected-attribute representation prevent a formal fairness conclusion."
        if not subgroup.empty
        else "Subgroup evidence was insufficient for a useful descriptive audit."
    )
    report = f"""# German Credit Binary Classification Report

## 1. Project understanding

This experiment predicts whether one credit applicant will have a bad credit outcome. UCI raw class `2` is the positive bad-credit class and class `1` is the good-credit class. The intended use is an offline benchmark or decision-support prototype, not an autonomous lending decision. Because business error costs were not supplied, model ranking uses average precision and the operating threshold uses training-only F2, which weights recall of bad credit more heavily than precision.

## 2. Data requirements

The source is [UCI Statlog (German Credit Data)]({SOURCE_URL}), fetched through `ucimlrepo` as dataset 144. Each row is one application, with {profile['raw_predictors']} application-time predictors and one outcome. The extract contains {profile['rows']} rows. The dataset does not include timestamps, repeated-customer identifiers, extraction history, or production lineage, so an IID stratified split is assumed. Licensing, retention, consent, and lawful-use requirements must be confirmed before any operational use.

## 3. Data analysis and findings

The raw extract has {profile['missing_cells_raw']} missing cells and {profile['duplicate_rows_raw']} exact duplicate rows. It contains {profile['good_credit_count']} good-credit and {profile['bad_credit_count']} bad-credit cases, so bad-credit prevalence is {profile['bad_credit_prevalence']:.1%}. This 70/30 split is moderately imbalanced, not an extreme rare-event problem. Numeric variables have materially different scales, and categorical variables are coded rather than ordinal measurements. Detailed profiles are in `feature_profile.csv` and `numeric_summary.csv`; plots are in `figures/`.

## 4. Cleaning and preprocessing

The cryptic UCI attributes were renamed to semantic fields. Numeric values are median-imputed and standardized; categoricals are most-frequent-imputed and one-hot encoded with unknown-category handling. Although this snapshot has no missing values, imputers make the saved inference pipeline robust to permitted missing input. All learned transformations are fitted inside cross-validation folds. Raw target 1 maps to 0 (good), and raw target 2 maps to 1 (bad).

## 5. Feature engineering and selection

Two prediction-time-safe ratios were added: credit amount per duration month and credit amount per existing credit. An L1-regularized logistic selector is inside the pipeline and its threshold was tuned by GridSearchCV. {selection_text} This prevents holdout leakage and avoids forcing feature deletion when validation evidence does not support it. Holdout permutation importance ranks the strongest original associations as: {top_features}. These are predictive associations, not causal effects.

## 6. Model development experiments

The benchmark is a stratified dummy classifier. Tuned candidates are logistic regression (interpretable linear baseline), random forest (bagged nonlinear interactions), and histogram gradient boosting (boosted nonlinear interactions). The bounded grids cover regularization, tree complexity, selection threshold, and class weighting. {imbalance_text} Synthetic oversampling was not used because the imbalance is moderate and SMOTE-style interpolation is questionable for one-hot-coded mixed categorical data.

## 7. Training and validation approach

The split is stratified and reproducible: {train_size} training rows and an untouched {test_size}-row holdout. GridSearchCV uses five shuffled stratified folds on training data only and refits by mean average precision. Average precision directly evaluates retrieval of the 30% bad-credit class; ROC-AUC is secondary. The final threshold of {threshold:.3f} was selected from out-of-fold training probabilities by maximum F2, never from the holdout.

## 8. Evaluation results

{comparison_table}

The selected model is **{best_name}**, chosen only by cross-validated average precision. On the untouched test set, average precision is {test_metrics['average_precision']:.3f} (bootstrap 95% CI {ci['average_precision'][0]:.3f}-{ci['average_precision'][1]:.3f}) and ROC-AUC is {test_metrics['roc_auc']:.3f} (95% CI {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}). At threshold {threshold:.3f}, bad-credit precision is {test_metrics['precision_bad']:.3f}, recall is {test_metrics['recall_bad']:.3f}, F2 is {test_metrics['f2_bad']:.3f}, with {test_metrics['false_positive']} false positives and {test_metrics['false_negative']} false negatives. The uncertainty is substantial because the holdout has only {test_size} records.

## 9. Interpretation and error analysis

Permutation importance was calculated once on the untouched holdout after model selection and is therefore descriptive, not a new selection step. The confusion matrix and ROC/precision-recall curves are in `figures/`. {subgroup_note} `personal_status_sex`, age, and foreign-worker status are protected characteristics or proxies in many lending contexts; aggregate accuracy cannot establish non-discrimination.

## 10. Recommendation, deployment considerations, limitations, and next steps

Use the saved pipeline only as a reproducible research baseline. Do not deploy it for automated credit decisions without: contemporary representative data; a documented decision policy and error costs; legal and fairness review; probability calibration on new data; subgroup sample-size requirements; human-review and adverse-action processes; and external/temporal validation. A proposed batch deployment should validate schema and ranges, version inputs and model artifacts, log outcomes, monitor missingness/category drift/prediction drift and subgroup errors, and roll back when data contracts fail. Retrain only after labels arrive and predefined degradation thresholds are breached.

The largest limitations are the small 1,000-row historical sample, absence of timestamps and geographic context, coded and incomplete protected attributes, possible dataset obsolescence, and a single random holdout. The next highest-value experiment is repeated nested cross-validation or external validation on recent applicants, followed by explicit cost-sensitive threshold selection and calibration using operational requirements.

## 11. Reproducibility and artifacts

- Data snapshot: `data/german_credit.csv`
- Complete inference pipeline: `{output_dir.name}/model.joblib`
- Metrics and parameters: `{output_dir.name}/metrics.json`, `{output_dir.name}/model_comparison.csv`
- Feature and error analysis: `{output_dir.name}/selected_transformed_features.csv`, `{output_dir.name}/permutation_importance.csv`, `{output_dir.name}/subgroup_metrics.csv`
- Figures: `{output_dir.name}/figures/`
- Random seed: {RANDOM_STATE}; split: 80/20 stratified; GridSearchCV: 5 folds
- Environment: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}
- Reproduce from the repository root: `.env/bin/python German_Credit_Data/main.py`

The model artifact has not been tested in a production serving environment.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.data_path.resolve(), refresh=args.refresh)
    X, y = clean_and_engineer(raw)
    profile = dataset_profile(raw, X, y)
    save_eda(raw, X, y, output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "recall_bad": make_scorer(recall_score, zero_division=0),
        "precision_bad": make_scorer(precision_score, zero_division=0),
        "f1_bad": make_scorer(f1_score, zero_division=0),
    }

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    params: dict[str, dict[str, Any]] = {}

    dummy = make_pipeline(X_train, DummyClassifier(strategy="prior"))
    dummy_scores = cross_validate(
        dummy, X_train, y_train, cv=cv, scoring=scoring, n_jobs=args.n_jobs
    )
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    dummy_test = probability_metrics(y_test, dummy_proba)
    dummy_at_half = threshold_metrics(y_test, dummy_proba, 0.5)
    rows.append(
        {
            "model": "Dummy prevalence baseline",
            "cv_average_precision_mean": float(dummy_scores["test_average_precision"].mean()),
            "cv_average_precision_std": float(dummy_scores["test_average_precision"].std()),
            "cv_roc_auc_mean": float(dummy_scores["test_roc_auc"].mean()),
            "test_average_precision": dummy_test["average_precision"],
            "test_roc_auc": dummy_test["roc_auc"],
            "test_recall_bad_at_0_5": dummy_at_half["recall_bad"],
            "test_precision_bad_at_0_5": dummy_at_half["precision_bad"],
            "selected_by_cv": False,
        }
    )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, (pipeline, grid) in candidate_searches(X_train, args.quick).items():
        print(f"Tuning {name}...", flush=True)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=scoring,
            refit="average_precision",
            cv=cv,
            n_jobs=args.n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best
        params[name] = search.best_params_
        proba = best.predict_proba(X_test)[:, 1]
        test = probability_metrics(y_test, proba)
        at_half = threshold_metrics(y_test, proba, 0.5)
        best_idx = search.best_index_
        rows.append(
            {
                "model": name,
                "cv_average_precision_mean": float(search.cv_results_["mean_test_average_precision"][best_idx]),
                "cv_average_precision_std": float(search.cv_results_["std_test_average_precision"][best_idx]),
                "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][best_idx]),
                "test_average_precision": test["average_precision"],
                "test_roc_auc": test["roc_auc"],
                "test_recall_bad_at_0_5": at_half["recall_bad"],
                "test_precision_bad_at_0_5": at_half["precision_bad"],
                "selected_by_cv": False,
            }
        )
        pd.DataFrame(search.cv_results_).to_csv(
            output_dir / f"gridsearch_{name.lower().replace(' ', '_')}.csv", index=False
        )

    candidate_rows = pd.DataFrame(rows).iloc[1:]
    best_name = str(
        candidate_rows.sort_values("cv_average_precision_mean", ascending=False).iloc[0]["model"]
    )
    comparison = pd.DataFrame(rows)
    comparison.loc[comparison["model"] == best_name, "selected_by_cv"] = True
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    best_model = fitted[best_name]
    oof_proba = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]
    threshold = choose_f2_threshold(y_train, oof_proba)
    oof_at_threshold = threshold_metrics(y_train, oof_proba, threshold)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_probability = probability_metrics(y_test, test_proba)
    test_at_threshold = threshold_metrics(y_test, test_proba, threshold)
    final_metrics = {**test_probability, **test_at_threshold}
    ci = {
        "average_precision": bootstrap_interval(y_test, test_proba, "average_precision"),
        "roc_auc": bootstrap_interval(y_test, test_proba, "roc_auc"),
    }

    pred = (test_proba >= threshold).astype(int)
    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=[0, 1]),
        index=["actual_good", "actual_bad"],
        columns=["predicted_good", "predicted_bad"],
    ).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(
        output_dir / "classification_report.csv", index_label="class_or_average"
    )
    pd.DataFrame(
        [
            {"dataset": "training_oof", **oof_at_threshold},
            {"dataset": "test", **test_at_threshold},
        ]
    ).to_csv(output_dir / "threshold_metrics.csv", index=False)

    selected, importance = save_feature_analysis(
        best_model, X_test, y_test, output_dir, args.n_jobs
    )
    subgroup = subgroup_metrics(X_test, y_test, test_proba, threshold)
    subgroup.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    save_evaluation_plots(y_test, test_proba, threshold, output_dir)

    # The fitted candidate consumes semantic/engineered fields. Wrap it with
    # deterministic raw UCI feature preparation so the saved artifact accepts
    # Attribute1..Attribute20 directly.
    inference_model = Pipeline(
        [
            (
                "raw_feature_engineering",
                FunctionTransformer(prepare_inference_features, validate=False),
            ),
            ("classifier", best_model),
        ]
    )
    joblib.dump(inference_model, output_dir / "model.joblib")
    reloaded_model = joblib.load(output_dir / "model.joblib")
    smoke_input = raw[list(ATTRIBUTE_NAMES)].head(3)
    smoke_probabilities = reloaded_model.predict_proba(smoke_input)[:, 1]
    expected_probabilities = best_model.predict_proba(
        prepare_inference_features(smoke_input)
    )[:, 1]
    if not np.allclose(
        smoke_probabilities, expected_probabilities, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("Reloaded raw-input inference pipeline changed predictions.")
    (output_dir / "inference_smoke_test.json").write_text(
        json.dumps(
            {
                "rows_scored": len(smoke_input),
                "probabilities_bad_credit": smoke_probabilities.tolist(),
                "reload_check": "passed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    inference_contract = {
        "raw_feature_names": [key for key in ATTRIBUTE_NAMES],
        "target_not_required_for_inference": TARGET,
        "positive_class": 1,
        "positive_class_meaning": "bad credit (raw UCI class 2)",
        "decision_threshold": threshold,
        "prediction_output": "probability of bad credit",
    }
    (output_dir / "inference_contract.json").write_text(
        json.dumps(inference_contract, indent=2), encoding="utf-8"
    )
    metrics_payload = {
        "profile": profile,
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": args.test_size,
            "random_state": RANDOM_STATE,
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        },
        "selection_metric": "average_precision",
        "selected_model": best_name,
        "best_parameters": params[best_name],
        "selected_threshold_training_oof_f2": threshold,
        "test_metrics": final_metrics,
        "bootstrap_95_percent_ci": {key: list(value) for key, value in ci.items()},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    write_report(
        args.report_path.resolve(),
        profile,
        comparison,
        best_name,
        params[best_name],
        threshold,
        final_metrics,
        ci,
        selected,
        importance,
        subgroup,
        len(X_train),
        len(X_test),
        output_dir,
    )
    print(f"Selected model: {best_name}")
    print(f"Test average precision: {final_metrics['average_precision']:.3f}")
    print(f"Test ROC-AUC: {final_metrics['roc_auc']:.3f}")
    print(f"Bad-credit recall at threshold {threshold:.3f}: {final_metrics['recall_bad']:.3f}")
    print(f"Report: {args.report_path.resolve()}")
    print(f"Artifacts: {output_dir}")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run end-to-end binary classification on UCI German Credit Data."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_dir / "data" / "german_credit.csv",
        help="Local CSV cache; downloaded from UCI if absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "artifacts",
        help="Directory for model, metrics, tables, and figures.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=project_dir / "Report.md",
        help="Generated measured-results report.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs (default 1 for portability; use -1 outside restricted sandboxes).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download the UCI data.")
    parser.add_argument(
        "--quick", action="store_true", help="Use a reduced grid for a faster smoke test."
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
