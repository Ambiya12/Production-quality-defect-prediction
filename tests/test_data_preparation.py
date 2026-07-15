"""Unit tests for cleaning and operational feature engineering."""

from __future__ import annotations

import pandas as pd

from src.data_preparation import REQUIRED_COLUMNS, VALID_RANGES, clean_production_data
from src.features import DERIVED_COLUMNS


def make_valid_data() -> pd.DataFrame:
    """Create a compact valid input frame used by multiple tests."""
    return pd.DataFrame(
        {
            "production_id": ["PRD-1", "PRD-2", "PRD-3", "PRD-4"],
            "production_date": [
                "2025-01-01 05:00:00",
                "2025-01-01 06:00:00",
                "2025-01-01 14:00:00",
                "2025-01-01 22:00:00",
            ],
            "site": ["Lyon", "Lyon", "Limoges", "Besancon"],
            "machine_id": ["M-01", "M-02", "M-04", "M-07"],
            "operator_team": ["Team A", "Team B", "Team A", "Team C"],
            "temperature": [70.0, 72.0, 68.0, 75.0],
            "pressure": [5.0, 5.2, 4.8, 5.4],
            "production_duration": [40.0, 42.0, 45.0, 48.0],
            "material_type": ["Alloy", "Ceramic", "Composite", "Alloy"],
            "quality_score": [95.0, 91.0, 93.0, 82.0],
            "defect": [0, 0, 0, 1],
        }
    )


def test_duplicate_production_ids_are_removed() -> None:
    data = pd.concat([make_valid_data(), make_valid_data().iloc[[0]]], ignore_index=True)

    cleaned = clean_production_data(data)

    assert len(cleaned) == 4
    assert cleaned["production_id"].is_unique


def test_production_date_is_converted_to_datetime() -> None:
    cleaned = clean_production_data(make_valid_data())

    assert pd.api.types.is_datetime64_any_dtype(cleaned["production_date"])


def test_invalid_numeric_and_categorical_values_are_corrected() -> None:
    data = make_valid_data()
    data.loc[0, ["temperature", "pressure", "production_duration", "quality_score"]] = [
        -999.0,
        99.0,
        -5.0,
        120.0,
    ]
    data.loc[0, ["site", "operator_team", "material_type"]] = [
        "Unknown site",
        "Team X",
        "Unknown material",
    ]

    cleaned = clean_production_data(data)
    repaired = cleaned.loc[cleaned["production_id"].eq("PRD-1")].iloc[0]

    for column, (lower, upper) in VALID_RANGES.items():
        assert lower <= repaired[column] <= upper
    assert repaired["quality_score"] == 100.0
    assert repaired["site"] == "Lyon"
    assert repaired["operator_team"] in {"Team A", "Team B", "Team C", "Team D"}
    assert repaired["material_type"] in {"Alloy", "Ceramic", "Composite", "Precious metal"}


def test_missing_numeric_values_are_imputed() -> None:
    data = make_valid_data()
    data.loc[1, ["temperature", "pressure", "production_duration"]] = pd.NA

    cleaned = clean_production_data(data)

    assert cleaned[["temperature", "pressure", "production_duration"]].isna().sum().sum() == 0


def test_shift_creation_covers_all_time_boundaries() -> None:
    cleaned = clean_production_data(make_valid_data()).set_index("production_id")

    assert cleaned.loc["PRD-1", "production_shift"] == "Night"
    assert cleaned.loc["PRD-2", "production_shift"] == "Morning"
    assert cleaned.loc["PRD-3", "production_shift"] == "Afternoon"
    assert cleaned.loc["PRD-4", "production_shift"] == "Night"


def test_clean_output_contains_required_and_derived_columns() -> None:
    cleaned = clean_production_data(make_valid_data())

    assert set(REQUIRED_COLUMNS).issubset(cleaned.columns)
    assert set(DERIVED_COLUMNS).issubset(cleaned.columns)
