"""Deterministic raw-feature processing used by training and saved inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


ATTRIBUTE_NAMES = {
    "Attribute1": "checking_account_status",
    "Attribute2": "duration_months",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_status",
    "Attribute7": "employment_since",
    "Attribute8": "installment_rate_pct_income",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors_guarantors",
    "Attribute11": "residence_years",
    "Attribute12": "property",
    "Attribute13": "age_years",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
}


def prepare_inference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw UCI predictor columns into the model's feature contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Inference input must be a pandas DataFrame.")
    missing = [column for column in ATTRIBUTE_NAMES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required raw feature columns: {missing}")

    data = frame[list(ATTRIBUTE_NAMES)].rename(columns=ATTRIBUTE_NAMES).copy()
    duration = data["duration_months"].replace(0, np.nan)
    existing_credits = data["existing_credits"].replace(0, np.nan)
    data["credit_amount_per_month"] = data["credit_amount"] / duration
    data["credit_amount_per_existing_credit"] = (
        data["credit_amount"] / existing_credits
    )
    return data
