"""Evaluation utilities for binary defect-classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.joblib"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_quality_clean.csv"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"


def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    defect_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate imbalance-aware classification metrics at a chosen threshold."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    true_values = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(defect_probability, dtype=float)
    if true_values.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and defect_probability must have the same length")
    if not np.isin(true_values, (0, 1)).all():
        raise ValueError("y_true must contain only binary values 0 and 1")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("defect probabilities must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(true_values, predictions, zero_division=0)),
        "recall": float(recall_score(true_values, predictions, zero_division=0)),
        "f1_score": float(f1_score(true_values, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_values, probabilities)),
        "pr_auc": float(average_precision_score(true_values, probabilities)),
        "confusion_matrix": confusion_matrix(true_values, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            true_values,
            predictions,
            labels=[0, 1],
            target_names=["acceptable", "defective"],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_pipeline(
    pipeline: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate any fitted binary classifier exposing ``predict_proba``."""
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("pipeline must implement predict_proba")
    probabilities = pipeline.predict_proba(features)[:, 1]
    return calculate_binary_metrics(target, probabilities, threshold=threshold)


def extract_feature_effects(pipeline: BaseEstimator, top_n: int = 12) -> dict[str, Any]:
    """Return model-specific feature effects for cautious interpretation."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a fitted scikit-learn Pipeline")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_[0], dtype=float)
        effects = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        positive = effects.nlargest(top_n, "coefficient").to_dict(orient="records")
        negative = effects.nsmallest(top_n, "coefficient").to_dict(orient="records")
        return {
            "method": "logistic_regression_coefficients",
            "positive_associations": positive,
            "negative_associations": negative,
            "caution": (
                "Coefficients describe conditional associations with predicted risk; "
                "they do not establish causal effects."
            ),
        }

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        effects = pd.DataFrame({"feature": feature_names, "importance": importances})
        return {
            "method": "random_forest_impurity_importance",
            "most_important": effects.nlargest(top_n, "importance").to_dict(orient="records"),
            "caution": (
                "Impurity-based importance can favor continuous or high-cardinality features "
                "and does not establish causal effects."
            ),
        }

    return {"method": "unavailable", "caution": "This estimator has no supported effect measure."}


def print_metrics(model_name: str, metrics: dict[str, Any]) -> None:
    """Print the main business-facing metrics and confusion matrix."""
    print(f"\n{model_name}")
    print("-" * len(model_name))
    for metric in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        print(f"{metric}: {metrics[metric]:.3f}")
    matrix = metrics["confusion_matrix"]
    print(f"confusion_matrix [[TN, FP], [FN, TP]]: {matrix}")


def main() -> None:
    """Re-evaluate the persisted final model on the reproducible holdout split."""
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {DEFAULT_MODEL_PATH}. Run python src/train_model.py first."
        )

    from src.train_model import MODEL_FEATURES, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE

    data = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["production_date"])
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    del train_data
    pipeline = joblib.load(DEFAULT_MODEL_PATH)
    metrics = evaluate_pipeline(
        pipeline,
        test_data[MODEL_FEATURES],
        test_data[TARGET_COLUMN],
    )
    print_metrics("Persisted final model", metrics)

    if DEFAULT_METRICS_PATH.exists():
        saved = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
        print(f"Selected model recorded in metrics file: {saved['selected_model']}")


if __name__ == "__main__":
    main()
