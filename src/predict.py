"""Validated inference interface for the persisted defect model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from src.data_preparation import (
        MACHINES_BY_SITE,
        VALID_MATERIALS,
        VALID_RANGES,
        VALID_SITES,
        VALID_TEAMS,
    )
    from src.features import add_operational_features
    from src.train_model import DECISION_THRESHOLD, MODEL_FEATURES
except ModuleNotFoundError:  # Support direct execution with ``python src/...``.
    from data_preparation import (
        MACHINES_BY_SITE,
        VALID_MATERIALS,
        VALID_RANGES,
        VALID_SITES,
        VALID_TEAMS,
    )
    from features import add_operational_features
    from train_model import DECISION_THRESHOLD, MODEL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.joblib"
REQUIRED_INPUT_COLUMNS = (
    "production_date",
    "site",
    "machine_id",
    "operator_team",
    "temperature",
    "pressure",
    "production_duration",
    "material_type",
)


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load and validate the persisted scikit-learn prediction pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run python src/train_model.py first."
        )
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Saved object is not a probability classifier")
    return pipeline


def _to_dataframe(records: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Normalize a single mapping or a DataFrame into a non-empty DataFrame."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([dict(records)])
    else:
        raise TypeError("records must be a dictionary-like mapping or a pandas DataFrame")
    if frame.empty:
        raise ValueError("records must contain at least one production record")
    return frame


def validate_prediction_input(records: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Validate inference fields and return feature-enriched records."""
    frame = _to_dataframe(records)
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    frame = frame.loc[:, list(REQUIRED_INPUT_COLUMNS)].copy()
    frame["production_date"] = pd.to_datetime(frame["production_date"], errors="coerce")
    if frame["production_date"].isna().any():
        raise ValueError("production_date must contain valid dates and times")

    for column, (lower, upper) in VALID_RANGES.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() | ~frame[column].between(lower, upper)
        if invalid.any():
            raise ValueError(f"{column} must be numeric and between {lower:g} and {upper:g}")

    category_rules = {
        "site": VALID_SITES,
        "operator_team": VALID_TEAMS,
        "material_type": VALID_MATERIALS,
    }
    for column, allowed in category_rules.items():
        invalid_values = sorted(set(frame.loc[~frame[column].isin(allowed), column].astype(str)))
        if invalid_values:
            raise ValueError(f"Invalid {column} values: {invalid_values}. Allowed values: {list(allowed)}")

    valid_machines = {machine for machines in MACHINES_BY_SITE.values() for machine in machines}
    invalid_machines = sorted(set(frame.loc[~frame["machine_id"].isin(valid_machines), "machine_id"].astype(str)))
    if invalid_machines:
        raise ValueError(f"Invalid machine_id values: {invalid_machines}")

    inconsistent = frame.apply(
        lambda row: row["machine_id"] not in MACHINES_BY_SITE[row["site"]],
        axis=1,
    )
    if inconsistent.any():
        invalid_rows = frame.index[inconsistent].tolist()
        raise ValueError(f"machine_id does not belong to the selected site for rows: {invalid_rows}")

    return add_operational_features(frame)


def _risk_category(probability: float) -> str:
    """Convert a probability to an understandable screening category."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def predict_defect(
    records: Mapping[str, Any] | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Return predicted defect class, probability, and risk category.

    Parameters
    ----------
    records:
        One mapping or a DataFrame containing one or more production records.
    model_path:
        Location of the complete persisted preprocessing and classifier pipeline.

    Returns
    -------
    pandas.DataFrame
        One result per input row with ``predicted_defect``,
        ``defect_probability``, and ``risk_category``.
    """
    featured = validate_prediction_input(records)
    pipeline = load_model(model_path)
    probabilities = pipeline.predict_proba(featured.loc[:, list(MODEL_FEATURES)])[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    return pd.DataFrame(
        {
            "predicted_defect": predictions,
            "defect_probability": probabilities,
            "risk_category": [_risk_category(value) for value in probabilities],
        },
        index=featured.index,
    )


def main() -> None:
    """Run a small example prediction from the command line."""
    example = {
        "production_date": "2025-06-15 14:30:00",
        "site": "Besancon",
        "machine_id": "M-07",
        "operator_team": "Team D",
        "temperature": 82.0,
        "pressure": 6.4,
        "production_duration": 58.0,
        "material_type": "Ceramic",
    }
    print(predict_defect(example).to_string(index=False))


if __name__ == "__main__":
    main()
