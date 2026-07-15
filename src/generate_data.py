"""Generate a reproducible synthetic manufacturing quality dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 2_500
N_DUPLICATES = 15
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "production_quality.csv"

SITES = ("Lyon", "Limoges", "Besancon")
MACHINES_BY_SITE = {
    "Lyon": ("M-01", "M-02", "M-03"),
    "Limoges": ("M-04", "M-05", "M-06"),
    "Besancon": ("M-07", "M-08"),
}
OPERATOR_TEAMS = ("Team A", "Team B", "Team C", "Team D")
MATERIAL_TYPES = ("Alloy", "Ceramic", "Composite", "Precious metal")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert log-odds values into probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def _sample_machines(sites: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Select a machine that belongs to each sampled site."""
    return np.array([rng.choice(MACHINES_BY_SITE[site]) for site in sites])


def generate_production_data(
    n_records: int = N_RECORDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create clean synthetic records with realistic defect relationships.

    Parameters
    ----------
    n_records:
        Number of unique production records to generate before injecting errors.
    seed:
        Seed used by NumPy's random generator.

    Returns
    -------
    pandas.DataFrame
        Synthetic production records before deliberate data-quality issues.
    """
    if n_records < 1:
        raise ValueError("n_records must be a positive integer")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01 00:00:00")
    minute_offsets = np.sort(rng.integers(0, 730 * 24 * 60, size=n_records))
    production_dates = start + pd.to_timedelta(minute_offsets, unit="m")

    sites = rng.choice(SITES, size=n_records, p=(0.42, 0.33, 0.25))
    machines = _sample_machines(sites, rng)
    teams = rng.choice(OPERATOR_TEAMS, size=n_records, p=(0.28, 0.27, 0.25, 0.20))
    materials = rng.choice(MATERIAL_TYPES, size=n_records, p=(0.34, 0.22, 0.29, 0.15))

    machine_temperature_offsets = {
        "M-01": -0.5,
        "M-02": 0.4,
        "M-03": 2.0,
        "M-04": -1.2,
        "M-05": 0.2,
        "M-06": 1.0,
        "M-07": 2.7,
        "M-08": -0.8,
    }
    temperature = rng.normal(70.0, 5.2, n_records) + np.array(
        [machine_temperature_offsets[machine] for machine in machines]
    )
    pressure = rng.normal(5.0, 0.65, n_records)
    production_duration = rng.normal(43.0, 8.0, n_records)
    production_duration += np.where(materials == "Ceramic", 5.0, 0.0)
    production_duration += np.where(materials == "Precious metal", 3.0, 0.0)
    production_duration = np.clip(production_duration, 18.0, 90.0)

    temperature_deviation = np.abs(temperature - 70.0)
    pressure_deviation = np.abs(pressure - 5.0)
    night_shift = (production_dates.hour < 6) | (production_dates.hour >= 22)

    machine_risk = pd.Series(machines).map(
        {"M-03": 0.45, "M-06": 0.30, "M-07": 0.75}
    ).fillna(0.0).to_numpy()
    material_risk = pd.Series(materials).map(
        {"Ceramic": 0.55, "Composite": 0.18, "Precious metal": 0.30}
    ).fillna(0.0).to_numpy()
    team_risk = pd.Series(teams).map({"Team D": 0.30}).fillna(0.0).to_numpy()

    log_odds = (
        -3.60
        + machine_risk
        + material_risk
        + team_risk
        + 0.20 * np.maximum(temperature_deviation - 3.0, 0.0)
        + 1.20 * np.maximum(pressure_deviation - 0.25, 0.0)
        + 0.050 * np.maximum(production_duration - 45.0, 0.0)
        + 0.22 * night_shift.astype(float)
    )
    defect_probability = _sigmoid(log_odds)
    defect = rng.binomial(1, defect_probability)

    quality_score = (
        94.0
        - 14.0 * defect
        - 0.30 * temperature_deviation
        - 1.50 * pressure_deviation
        - 0.07 * np.maximum(production_duration - 45.0, 0.0)
        + rng.normal(0.0, 3.0, n_records)
    )
    quality_score = np.clip(quality_score, 0.0, 100.0)

    return pd.DataFrame(
        {
            "production_id": [f"PRD-{index:06d}" for index in range(1, n_records + 1)],
            "production_date": production_dates,
            "site": sites,
            "machine_id": machines,
            "operator_team": teams,
            "temperature": np.round(temperature, 2),
            "pressure": np.round(pressure, 2),
            "production_duration": np.round(production_duration, 2),
            "material_type": materials,
            "quality_score": np.round(quality_score, 2),
            "defect": defect.astype(int),
        }
    )


def inject_data_quality_issues(
    data: pd.DataFrame,
    seed: int = RANDOM_SEED + 1,
    n_duplicates: int = N_DUPLICATES,
) -> pd.DataFrame:
    """Add missing values, duplicates, invalid values, and plausible outliers."""
    if data.empty:
        raise ValueError("data must contain at least one record")

    rng = np.random.default_rng(seed)
    corrupted = data.copy()

    missing_counts = {
        "temperature": 22,
        "pressure": 18,
        "production_duration": 15,
        "operator_team": 12,
        "material_type": 10,
    }
    available_indices = corrupted.index.to_numpy()
    for column, count in missing_counts.items():
        selected = rng.choice(available_indices, size=min(count, len(corrupted)), replace=False)
        corrupted.loc[selected, column] = np.nan

    invalid_values = {
        "temperature": (-25.0, 260.0),
        "pressure": (-2.0, 42.0),
        "production_duration": (-15.0, 720.0),
        "quality_score": (-12.0, 128.0),
    }
    for column, values in invalid_values.items():
        selected = rng.choice(available_indices, size=len(values), replace=False)
        corrupted.loc[selected, column] = values

    corrupted.loc[rng.choice(available_indices, size=2, replace=False), "site"] = "Unknown site"
    corrupted.loc[rng.choice(available_indices, size=2, replace=False), "operator_team"] = "Team X"
    corrupted.loc[rng.choice(available_indices, size=2, replace=False), "material_type"] = "Unknown material"

    outlier_indices = rng.choice(available_indices, size=9, replace=False)
    corrupted.loc[outlier_indices[:3], "temperature"] = (42.0, 108.0, 112.0)
    corrupted.loc[outlier_indices[3:6], "pressure"] = (1.2, 10.8, 11.4)
    corrupted.loc[outlier_indices[6:], "production_duration"] = (165.0, 190.0, 220.0)

    duplicate_count = min(n_duplicates, len(corrupted))
    duplicates = corrupted.sample(n=duplicate_count, random_state=seed)
    return pd.concat([corrupted, duplicates], ignore_index=True)


def summarize_dataset(data: pd.DataFrame) -> str:
    """Return a concise, human-readable dataset summary."""
    return "\n".join(
        (
            "Synthetic production dataset generated",
            f"Rows: {len(data):,}",
            f"Columns: {data.shape[1]}",
            f"Defect rate: {data['defect'].mean():.2%}",
            f"Missing values: {int(data.isna().sum().sum()):,}",
            f"Duplicate rows: {int(data.duplicated().sum()):,}",
        )
    )


def save_dataset(data: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save production records as CSV and return the resolved output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return output_path.resolve()


def main() -> None:
    """Generate, corrupt, save, and summarize the synthetic dataset."""
    clean_data = generate_production_data()
    raw_data = inject_data_quality_issues(clean_data)
    output_path = save_dataset(raw_data)
    print(summarize_dataset(raw_data))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
