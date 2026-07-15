"""Clean raw production data and create model-ready operational features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.features import DERIVED_COLUMNS, add_operational_features
except ModuleNotFoundError:  # Support direct execution with ``python src/...``.
    from features import DERIVED_COLUMNS, add_operational_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "production_quality.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "production_quality_clean.csv"

REQUIRED_COLUMNS = (
    "production_id",
    "production_date",
    "site",
    "machine_id",
    "operator_team",
    "temperature",
    "pressure",
    "production_duration",
    "material_type",
    "quality_score",
    "defect",
)
VALID_SITES = ("Lyon", "Limoges", "Besancon")
MACHINES_BY_SITE = {
    "Lyon": ("M-01", "M-02", "M-03"),
    "Limoges": ("M-04", "M-05", "M-06"),
    "Besancon": ("M-07", "M-08"),
}
VALID_MACHINES = tuple(machine for machines in MACHINES_BY_SITE.values() for machine in machines)
MACHINE_TO_SITE = {
    machine: site for site, machines in MACHINES_BY_SITE.items() for machine in machines
}
VALID_TEAMS = ("Team A", "Team B", "Team C", "Team D")
VALID_MATERIALS = ("Alloy", "Ceramic", "Composite", "Precious metal")
VALID_RANGES = {
    "temperature": (35.0, 125.0),
    "pressure": (0.5, 15.0),
    "production_duration": (1.0, 300.0),
}


def _validate_schema(data: pd.DataFrame) -> None:
    """Raise a clear error when required source columns are absent."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def _invalid_category_count(series: pd.Series, allowed: tuple[str, ...]) -> int:
    """Count non-null categorical values outside an allowed vocabulary."""
    return int((series.notna() & ~series.isin(allowed)).sum())


def data_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize completeness, duplication, and domain-rule violations."""
    report: dict[str, Any] = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_production_ids": 0,
        "invalid_values": {},
    }
    if "production_id" in data:
        report["duplicate_production_ids"] = int(data["production_id"].duplicated().sum())

    invalid_values: dict[str, int] = {}
    for column, (lower, upper) in VALID_RANGES.items():
        if column in data:
            numeric = pd.to_numeric(data[column], errors="coerce")
            invalid_values[column] = int((numeric.notna() & ~numeric.between(lower, upper)).sum())
    if "quality_score" in data:
        score = pd.to_numeric(data["quality_score"], errors="coerce")
        invalid_values["quality_score"] = int((score.notna() & ~score.between(0, 100)).sum())
    if "site" in data:
        invalid_values["site"] = _invalid_category_count(data["site"], VALID_SITES)
    if "machine_id" in data:
        invalid_values["machine_id"] = _invalid_category_count(
            data["machine_id"], VALID_MACHINES
        )
    if "operator_team" in data:
        invalid_values["operator_team"] = _invalid_category_count(
            data["operator_team"], VALID_TEAMS
        )
    if "material_type" in data:
        invalid_values["material_type"] = _invalid_category_count(
            data["material_type"], VALID_MATERIALS
        )
    report["invalid_values"] = invalid_values
    return report


def _clean_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Correct categorical values using known vocabularies and machine-site links."""
    cleaned = data.copy()

    cleaned.loc[~cleaned["machine_id"].isin(VALID_MACHINES), "machine_id"] = pd.NA
    cleaned.loc[~cleaned["site"].isin(VALID_SITES), "site"] = pd.NA
    inferred_site = cleaned["machine_id"].map(MACHINE_TO_SITE)
    cleaned["site"] = cleaned["site"].fillna(inferred_site)
    cleaned["site"] = cleaned["site"].fillna(pd.Series(cleaned["site"].mode()).iloc[0])

    for site, machines in MACHINES_BY_SITE.items():
        site_mask = cleaned["site"].eq(site)
        invalid_machine = site_mask & ~cleaned["machine_id"].isin(machines)
        fallback_machine = (
            cleaned.loc[site_mask & cleaned["machine_id"].isin(machines), "machine_id"].mode()
        )
        replacement = fallback_machine.iloc[0] if not fallback_machine.empty else machines[0]
        cleaned.loc[invalid_machine, "machine_id"] = replacement

    category_rules = {
        "operator_team": VALID_TEAMS,
        "material_type": VALID_MATERIALS,
    }
    for column, allowed in category_rules.items():
        cleaned.loc[~cleaned[column].isin(allowed), column] = pd.NA
        mode = cleaned[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Cannot impute {column}: no valid values remain")
        cleaned[column] = cleaned[column].fillna(mode.iloc[0])

    return cleaned


def _clean_numeric_values(data: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns, invalidate impossible values, and impute medians."""
    cleaned = data.copy()
    numeric_columns = (*VALID_RANGES.keys(), "quality_score", "defect")
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column, (lower, upper) in VALID_RANGES.items():
        cleaned.loc[~cleaned[column].between(lower, upper), column] = np.nan
        machine_medians = cleaned.groupby("machine_id")[column].transform("median")
        cleaned[column] = cleaned[column].fillna(machine_medians).fillna(cleaned[column].median())

    cleaned["quality_score"] = cleaned["quality_score"].clip(lower=0.0, upper=100.0)
    cleaned["quality_score"] = cleaned["quality_score"].fillna(cleaned["quality_score"].median())

    cleaned = cleaned.loc[cleaned["defect"].isin((0, 1))].copy()
    cleaned["defect"] = cleaned["defect"].astype(int)
    return cleaned


def clean_production_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated, deduplicated, imputed, and feature-enriched data."""
    _validate_schema(data)
    cleaned = data.loc[:, REQUIRED_COLUMNS].copy()
    cleaned = cleaned.drop_duplicates(subset="production_id", keep="first")
    cleaned = cleaned.dropna(subset=["production_id"])

    cleaned["production_date"] = pd.to_datetime(cleaned["production_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["production_date"])
    cleaned = _clean_categories(cleaned)
    cleaned = _clean_numeric_values(cleaned)
    cleaned = add_operational_features(cleaned)

    ordered_columns = [*REQUIRED_COLUMNS, *DERIVED_COLUMNS]
    cleaned = cleaned.loc[:, ordered_columns].sort_values("production_date").reset_index(drop=True)
    return cleaned


def save_clean_data(
    data: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the cleaned dataset to CSV and return its resolved path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return output_path.resolve()


def main() -> None:
    """Load raw data, report quality, clean it, and save the processed result."""
    raw_data = pd.read_csv(DEFAULT_INPUT_PATH)
    print("Data-quality report before cleaning:")
    print(json.dumps(data_quality_report(raw_data), indent=2))

    clean_data = clean_production_data(raw_data)
    output_path = save_clean_data(clean_data)

    print("\nData-quality report after cleaning:")
    print(json.dumps(data_quality_report(clean_data), indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
