"""Train, compare, interpret, and persist defect-classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.evaluate_model import evaluate_pipeline, extract_feature_effects, print_metrics
except ModuleNotFoundError:  # Support direct execution with ``python src/...``.
    from evaluate_model import evaluate_pipeline, extract_feature_effects, print_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_quality_clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "defect"
DECISION_THRESHOLD = 0.50

NUMERIC_FEATURES = (
    "temperature",
    "pressure",
    "production_duration",
    "production_month",
    "production_hour",
    "temperature_deviation",
    "pressure_deviation",
)
CATEGORICAL_FEATURES = (
    "site",
    "machine_id",
    "operator_team",
    "material_type",
    "production_weekday",
    "production_shift",
)
MODEL_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
LEAKAGE_COLUMNS = ("production_id", "production_date", "quality_score", TARGET_COLUMN)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is learned only from the training split."""
    numeric_pipeline = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        )
    )
    return ColumnTransformer(
        transformers=(
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ),
        verbose_feature_names_out=True,
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Create reproducible, class-balanced candidate model pipelines."""
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline(
            steps=(
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            )
        )
        for name, estimator in estimators.items()
    }


def validate_training_data(data: pd.DataFrame) -> None:
    """Validate model columns, target values, and leakage exclusions."""
    required = set(MODEL_FEATURES) | {TARGET_COLUMN}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    if not data[TARGET_COLUMN].isin((0, 1)).all():
        raise ValueError("defect target must contain only 0 and 1")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError("defect target must contain both classes")
    leaked = sorted(set(MODEL_FEATURES) & set(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Model feature list contains leakage columns: {leaked}")


def select_business_model(results: dict[str, dict[str, Any]]) -> str:
    """Select the candidate with highest defect recall, then PR-AUC as tie-breaker."""
    if not results:
        raise ValueError("At least one evaluated model is required")
    return max(
        results,
        key=lambda name: (results[name]["recall"], results[name]["pr_auc"]),
    )


def train_and_compare(
    data: pd.DataFrame,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit both candidates on a stratified split and choose a final pipeline."""
    validate_training_data(data)
    features = data.loc[:, list(MODEL_FEATURES)]
    target = data[TARGET_COLUMN].astype(int)

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    candidates = build_candidate_pipelines()
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(train_features, train_target)
        metrics = evaluate_pipeline(
            pipeline,
            test_features,
            test_target,
            threshold=DECISION_THRESHOLD,
        )
        candidate_metrics[name] = metrics
        print_metrics(name.replace("_", " ").title(), metrics)
        report = pd.DataFrame(metrics["classification_report"]).transpose()
        print(report.round(3).to_string())

    selected_name = select_business_model(candidate_metrics)
    selected_pipeline = candidates[selected_name]
    interpretation = extract_feature_effects(selected_pipeline, top_n=10)

    summary: dict[str, Any] = {
        "dataset": {
            "records": int(len(data)),
            "defect_rate": float(target.mean()),
            "training_records": int(len(train_features)),
            "test_records": int(len(test_features)),
            "synthetic": True,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "model_features": list(MODEL_FEATURES),
        "excluded_for_leakage": list(LEAKAGE_COLUMNS),
        "candidate_metrics": candidate_metrics,
        "selected_model": selected_name,
        "selection_rationale": (
            "Selected the candidate with the highest recall for defective products at the "
            "0.50 threshold, using PR-AUC as a tie-breaker. In quality screening, missing a "
            "defect is treated as more costly than reviewing a false alert; precision remains "
            "visible because unnecessary reviews also have an operational cost."
        ),
        "interpretation": interpretation,
        "limitations": [
            "The dataset is synthetic and cannot demonstrate real-world performance.",
            "Feature effects are predictive associations, not evidence of causation.",
            "The decision threshold must be validated against real inspection costs.",
            "Production experts must validate findings before any operational use.",
        ],
    }
    return selected_pipeline, summary


def save_training_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> tuple[Path, Path]:
    """Persist the complete fitted pipeline and JSON evaluation report."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path.resolve(), metrics_path.resolve()


def main() -> None:
    """Load processed data, compare models, and persist the selected pipeline."""
    data = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["production_date"])
    pipeline, metrics = train_and_compare(data)
    model_path, metrics_path = save_training_artifacts(pipeline, metrics)

    print(f"\nSelected model: {metrics['selected_model']}")
    print(metrics["selection_rationale"])
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
