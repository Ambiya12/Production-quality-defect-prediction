"""Tests for validated defect prediction."""

from __future__ import annotations

import pandas as pd
import pytest

from src.predict import predict_defect, validate_prediction_input


@pytest.fixture
def valid_record() -> dict[str, object]:
    return {
        "production_date": "2025-06-15 14:30:00",
        "site": "Besancon",
        "machine_id": "M-07",
        "operator_team": "Team D",
        "temperature": 78.0,
        "pressure": 5.8,
        "production_duration": 54.0,
        "material_type": "Ceramic",
    }


def test_prediction_returns_class_probability_and_risk(valid_record: dict[str, object]) -> None:
    result = predict_defect(valid_record)

    assert list(result.columns) == ["predicted_defect", "defect_probability", "risk_category"]
    assert result.loc[0, "predicted_defect"] in {0, 1}
    assert 0.0 <= result.loc[0, "defect_probability"] <= 1.0
    assert result.loc[0, "risk_category"] in {"Low", "Medium", "High"}


def test_batch_prediction_preserves_row_count(valid_record: dict[str, object]) -> None:
    records = pd.DataFrame([valid_record, {**valid_record, "temperature": 70.0}])

    result = predict_defect(records)

    assert len(result) == 2


def test_missing_required_column_has_clear_error(valid_record: dict[str, object]) -> None:
    valid_record.pop("pressure")

    with pytest.raises(ValueError, match="Missing required prediction columns.*pressure"):
        validate_prediction_input(valid_record)


def test_machine_must_belong_to_site(valid_record: dict[str, object]) -> None:
    valid_record["machine_id"] = "M-01"

    with pytest.raises(ValueError, match="does not belong"):
        validate_prediction_input(valid_record)


def test_invalid_operating_value_has_clear_error(valid_record: dict[str, object]) -> None:
    valid_record["temperature"] = 500.0

    with pytest.raises(ValueError, match="temperature must be numeric"):
        validate_prediction_input(valid_record)
