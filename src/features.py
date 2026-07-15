"""Reusable operational feature engineering for training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL_TEMPERATURE = 70.0
NORMAL_PRESSURE = 5.0
REQUIRED_SOURCE_COLUMNS = ("production_date", "temperature", "pressure")
DERIVED_COLUMNS = (
    "production_month",
    "production_day",
    "production_weekday",
    "production_hour",
    "production_shift",
    "temperature_deviation",
    "pressure_deviation",
)


def production_shift_from_hour(hour: int) -> str:
    """Map an hour from 0 to 23 to its production shift."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if 6 <= hour <= 13:
        return "Morning"
    if 14 <= hour <= 21:
        return "Afternoon"
    return "Night"


def add_operational_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add time and operating-deviation features without mutating the input.

    The function deliberately uses only information available at production
    time. Quality-control outcomes such as ``quality_score`` are not used.
    """
    missing_columns = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for feature engineering: {missing_columns}")

    featured = data.copy()
    featured["production_date"] = pd.to_datetime(featured["production_date"], errors="coerce")
    if featured["production_date"].isna().any():
        raise ValueError("production_date contains missing or invalid timestamps")

    featured["production_month"] = featured["production_date"].dt.month.astype(int)
    featured["production_day"] = featured["production_date"].dt.day.astype(int)
    featured["production_weekday"] = featured["production_date"].dt.day_name()
    featured["production_hour"] = featured["production_date"].dt.hour.astype(int)
    featured["production_shift"] = np.select(
        [
            featured["production_hour"].between(6, 13),
            featured["production_hour"].between(14, 21),
        ],
        ["Morning", "Afternoon"],
        default="Night",
    )
    featured["temperature_deviation"] = (
        featured["temperature"].astype(float) - NORMAL_TEMPERATURE
    ).abs()
    featured["pressure_deviation"] = (
        featured["pressure"].astype(float) - NORMAL_PRESSURE
    ).abs()
    return featured
