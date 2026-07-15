"""Streamlit dashboard for synthetic production quality monitoring."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_preparation import MACHINES_BY_SITE, VALID_MATERIALS, VALID_SITES, VALID_TEAMS
from src.predict import load_model, validate_prediction_input
from src.train_model import DECISION_THRESHOLD, MODEL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_quality_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "defect_model.joblib"
MIN_RELIABLE_SAMPLE = 100
COLOR_ACCEPTABLE = "#356859"
COLOR_DEFECT = "#B45145"
COLOR_ACCENT = "#C58A42"

st.set_page_config(
    page_title="Production Quality Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: #F6F4EF; border: 1px solid #E3DED2; padding: 0.8rem; border-radius: 0.45rem;}
    [data-testid="stMetricLabel"] {font-weight: 600;}
    .synthetic-banner {background: #FFF7E8; border-left: 4px solid #C58A42; padding: 0.75rem 1rem; border-radius: 0.25rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_production_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the cleaned production dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. Run python src/data_preparation.py first."
        )
    data = pd.read_csv(path, parse_dates=["production_date"])
    required = {
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
        "production_shift",
        "defect",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Clean dataset is missing required columns: {missing}")
    return data


@st.cache_resource(show_spinner=False)
def load_prediction_model(path: Path = MODEL_PATH):
    """Load the persisted prediction pipeline once per Streamlit process."""
    return load_model(path)


def quality_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Aggregate defect counts, rates, and production volumes by a dimension."""
    return (
        data.groupby(group, observed=True)["defect"]
        .agg(defect_count="sum", production_volume="size", defect_rate="mean")
        .reset_index()
        .sort_values("defect_rate", ascending=False)
    )


def render_kpis(data: pd.DataFrame) -> None:
    """Render the six requested overview KPIs."""
    columns = st.columns(6)
    metrics = (
        ("Products", f"{len(data):,}"),
        ("Defects", f"{int(data['defect'].sum()):,}"),
        ("Defect rate", f"{data['defect'].mean():.2%}"),
        ("Average quality", f"{data['quality_score'].mean():.1f}/100"),
        ("Sites", f"{data['site'].nunique():,}"),
        ("Machines", f"{data['machine_id'].nunique():,}"),
    )
    for column, (label, value) in zip(columns, metrics, strict=True):
        column.metric(label, value)


def render_overview(data: pd.DataFrame) -> None:
    """Render business KPIs and high-level production quality charts."""
    st.subheader("Production overview")
    st.caption("Portfolio demonstration using synthetic manufacturing records.")
    render_kpis(data)

    left, right = st.columns(2)
    site_quality = quality_summary(data, "site")
    with left:
        volume = px.bar(
            site_quality.sort_values("production_volume", ascending=False),
            x="site",
            y="production_volume",
            text="production_volume",
            title="Production volume by site (count)",
            labels={"site": "Site", "production_volume": "Products"},
            color_discrete_sequence=["#7895B2"],
        )
        st.plotly_chart(volume, use_container_width=True)
    with right:
        site_rate = px.bar(
            site_quality,
            x="site",
            y="defect_rate",
            text=site_quality["defect_rate"].map(lambda value: f"{value:.1%}"),
            title="Defect rate by site (rate)",
            labels={"site": "Site", "defect_rate": "Defect rate"},
            color="defect_rate",
            color_continuous_scale="Oranges",
        )
        site_rate.update_yaxes(tickformat=".0%")
        st.plotly_chart(site_rate, use_container_width=True)

    machine_quality = quality_summary(data, "machine_id")
    machine_chart = px.bar(
        machine_quality,
        x="machine_id",
        y="defect_rate",
        text=machine_quality.apply(
            lambda row: f"{row.defect_rate:.1%}<br>n={row.production_volume:,}", axis=1
        ),
        title="Defect rate by machine, with production volume",
        labels={"machine_id": "Machine", "defect_rate": "Defect rate"},
        color="defect_rate",
        color_continuous_scale="Oranges",
    )
    machine_chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(machine_chart, use_container_width=True)

    monthly = (
        data.set_index("production_date")
        .resample("MS")["defect"]
        .agg(defect_count="sum", production_volume="size", defect_rate="mean")
        .reset_index()
    )
    trend = go.Figure()
    trend.add_trace(
        go.Bar(
            x=monthly["production_date"],
            y=monthly["defect_count"],
            name="Defect count",
            marker_color=COLOR_DEFECT,
        )
    )
    trend.add_trace(
        go.Scatter(
            x=monthly["production_date"],
            y=monthly["defect_rate"],
            name="Defect rate",
            yaxis="y2",
            mode="lines+markers",
            line={"color": COLOR_ACCENT, "width": 2},
        )
    )
    trend.update_layout(
        title="Monthly defects: count and rate",
        xaxis_title="Production month",
        yaxis={"title": "Defect count"},
        yaxis2={"title": "Defect rate", "overlaying": "y", "side": "right", "tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(trend, use_container_width=True)


def apply_quality_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Build sidebar controls and return the filtered quality-analysis sample."""
    st.sidebar.header("Quality analysis filters")
    minimum_date = data["production_date"].min().date()
    maximum_date = data["production_date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Production date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    selected_sites = st.sidebar.multiselect(
        "Sites", sorted(data["site"].unique()), default=sorted(data["site"].unique())
    )
    available_machines = sorted(data.loc[data["site"].isin(selected_sites), "machine_id"].unique())
    selected_machines = st.sidebar.multiselect(
        "Machines", available_machines, default=available_machines
    )
    selected_teams = st.sidebar.multiselect(
        "Operator teams",
        sorted(data["operator_team"].unique()),
        default=sorted(data["operator_team"].unique()),
    )
    selected_materials = st.sidebar.multiselect(
        "Material types",
        sorted(data["material_type"].unique()),
        default=sorted(data["material_type"].unique()),
    )
    selected_shifts = st.sidebar.multiselect(
        "Production shifts",
        sorted(data["production_shift"].unique()),
        default=sorted(data["production_shift"].unique()),
    )

    date_values = data["production_date"].dt.date
    mask = (
        date_values.between(start_date, end_date)
        & data["site"].isin(selected_sites)
        & data["machine_id"].isin(selected_machines)
        & data["operator_team"].isin(selected_teams)
        & data["material_type"].isin(selected_materials)
        & data["production_shift"].isin(selected_shifts)
    )
    return data.loc[mask].copy()


def render_quality_analysis(data: pd.DataFrame) -> None:
    """Render filtered quality distributions and risk-group comparisons."""
    st.subheader("Interactive quality analysis")
    st.caption("Filters apply to every KPI, chart, and table on this page.")
    if data.empty:
        st.warning("No production records match the selected filters. Broaden the selection to continue.")
        return
    if len(data) < MIN_RELIABLE_SAMPLE:
        st.warning(
            f"Only {len(data)} records match these filters. Treat rates and rankings cautiously; "
            f"at least {MIN_RELIABLE_SAMPLE} records are recommended for this exploratory view."
        )

    kpi_columns = st.columns(5)
    kpis = (
        ("Filtered products", f"{len(data):,}"),
        ("Filtered defects", f"{int(data['defect'].sum()):,}"),
        ("Filtered defect rate", f"{data['defect'].mean():.2%}"),
        ("Average temperature", f"{data['temperature'].mean():.1f} °C"),
        ("Average pressure", f"{data['pressure'].mean():.2f} bar"),
    )
    for column, (label, value) in zip(kpi_columns, kpis, strict=True):
        column.metric(label, value)

    plot_data = data.assign(
        quality_status=data["defect"].map({0: "Acceptable", 1: "Defective"})
    )
    left, right = st.columns(2)
    with left:
        temperature_histogram = px.histogram(
            plot_data,
            x="temperature",
            color="quality_status",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            title="Temperature distribution (record count)",
            labels={"temperature": "Temperature (°C)", "quality_status": "Quality status"},
            color_discrete_map={"Acceptable": COLOR_ACCEPTABLE, "Defective": COLOR_DEFECT},
        )
        st.plotly_chart(temperature_histogram, use_container_width=True)
    with right:
        temperature_box = px.box(
            plot_data,
            x="quality_status",
            y="temperature",
            color="quality_status",
            points="outliers",
            title="Temperature by defect status",
            labels={"quality_status": "Quality status", "temperature": "Temperature (°C)"},
            color_discrete_map={"Acceptable": COLOR_ACCEPTABLE, "Defective": COLOR_DEFECT},
        )
        temperature_box.update_layout(showlegend=False)
        st.plotly_chart(temperature_box, use_container_width=True)

    left, right = st.columns(2)
    with left:
        pressure_box = px.box(
            plot_data,
            x="quality_status",
            y="pressure",
            color="quality_status",
            points="outliers",
            title="Pressure by defect status",
            labels={"quality_status": "Quality status", "pressure": "Pressure (bar)"},
            color_discrete_map={"Acceptable": COLOR_ACCEPTABLE, "Defective": COLOR_DEFECT},
        )
        pressure_box.update_layout(showlegend=False)
        st.plotly_chart(pressure_box, use_container_width=True)
    with right:
        duration_box = px.box(
            plot_data,
            x="quality_status",
            y="production_duration",
            color="quality_status",
            points="outliers",
            title="Production duration by defect status",
            labels={"quality_status": "Quality status", "production_duration": "Duration (minutes)"},
            color_discrete_map={"Acceptable": COLOR_ACCEPTABLE, "Defective": COLOR_DEFECT},
        )
        duration_box.update_layout(showlegend=False)
        st.plotly_chart(duration_box, use_container_width=True)

    left, right = st.columns(2)
    for container, dimension, title, label in (
        (left, "material_type", "Defect rate by material type", "Material type"),
        (right, "operator_team", "Defect rate by operator team", "Operator team"),
    ):
        summary = quality_summary(data, dimension)
        figure = px.bar(
            summary,
            x=dimension,
            y="defect_rate",
            text=summary.apply(
                lambda row: f"{row.defect_rate:.1%}<br>n={row.production_volume:,}", axis=1
            ),
            title=title,
            labels={dimension: label, "defect_rate": "Defect rate"},
            color="defect_rate",
            color_continuous_scale="Oranges",
        )
        figure.update_yaxes(tickformat=".0%")
        container.plotly_chart(figure, use_container_width=True)

    st.markdown("#### High-risk production groups")
    st.caption(
        "Observed groups ranked by defect rate. Minimum volume protects against rankings based on one or two records."
    )
    risk_groups = (
        data.groupby(["site", "machine_id", "material_type", "production_shift"], observed=True)[
            "defect"
        ]
        .agg(defect_count="sum", production_volume="size", defect_rate="mean")
        .reset_index()
    )
    minimum_group_volume = max(10, int(len(data) * 0.01))
    risk_groups = risk_groups.loc[risk_groups["production_volume"] >= minimum_group_volume]
    if risk_groups.empty:
        st.info("No groups meet the minimum volume for this filtered sample.")
    else:
        top_groups = risk_groups.nlargest(10, ["defect_rate", "production_volume"]).copy()
        top_groups["defect_rate"] = top_groups["defect_rate"].map(lambda value: f"{value:.1%}")
        st.dataframe(top_groups, use_container_width=True, hide_index=True)


def prediction_explanation(record: dict[str, Any]) -> list[str]:
    """Create cautious, rule-based context for one prediction."""
    factors: list[str] = []
    if abs(float(record["temperature"]) - 70.0) > 7.0:
        factors.append("Temperature is far from the simulated 70°C operating center.")
    if abs(float(record["pressure"]) - 5.0) > 1.0:
        factors.append("Pressure is far from the simulated 5-bar operating center.")
    if float(record["production_duration"]) > 55.0:
        factors.append("The production duration is longer than typical in the training data.")
    if record["machine_id"] in {"M-06", "M-07"}:
        factors.append("This machine had a higher observed defect rate in the synthetic sample.")
    if record["material_type"] == "Ceramic":
        factors.append("Ceramic was associated with higher predicted risk in the synthetic sample.")
    if not factors:
        factors.append("Entered conditions are close to common operating values in the synthetic sample.")
    return factors


def render_prediction_page() -> None:
    """Render the validated single-record prediction form and result."""
    st.subheader("Defect risk prediction")
    st.error(
        "This prediction is a decision-support tool and does not replace physical quality "
        "inspection or the judgement of production experts."
    )
    st.warning(
        "The model was trained only on synthetic data. It is a portfolio demonstration and is "
        "not suitable for real production deployment."
    )

    with st.form("prediction_form"):
        left, middle, right = st.columns(3)
        site = left.selectbox("Production site", VALID_SITES)
        machine = middle.selectbox("Machine", MACHINES_BY_SITE[site])
        team = right.selectbox("Operator team", VALID_TEAMS)

        left, middle, right = st.columns(3)
        material = left.selectbox("Material type", VALID_MATERIALS)
        temperature_value = middle.number_input(
            "Temperature (°C)", min_value=35.0, max_value=125.0, value=70.0, step=0.5
        )
        pressure_value = right.number_input(
            "Pressure (bar)", min_value=0.5, max_value=15.0, value=5.0, step=0.1
        )

        left, middle, right = st.columns(3)
        duration_value = left.number_input(
            "Production duration (minutes)",
            min_value=1.0,
            max_value=300.0,
            value=45.0,
            step=1.0,
        )
        production_day = middle.date_input("Production date", value=date.today())
        production_time = right.time_input("Production time", value=time(10, 0))
        submitted = st.form_submit_button("Estimate defect risk", type="primary")

    if not submitted:
        return

    record: dict[str, Any] = {
        "production_date": datetime.combine(production_day, production_time),
        "site": site,
        "machine_id": machine,
        "operator_team": team,
        "temperature": temperature_value,
        "pressure": pressure_value,
        "production_duration": duration_value,
        "material_type": material,
    }
    try:
        featured = validate_prediction_input(record)
        pipeline = load_prediction_model()
        probability = float(
            pipeline.predict_proba(featured.loc[:, list(MODEL_FEATURES)])[:, 1][0]
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        st.error(f"Prediction unavailable: {error}")
        return

    predicted_class = int(probability >= DECISION_THRESHOLD)
    risk = "Low" if probability < 0.30 else "Medium" if probability < 0.60 else "High"
    probability_column, risk_column, class_column = st.columns(3)
    probability_column.metric("Predicted defect probability", f"{probability:.1%}")
    risk_column.metric("Risk category", risk)
    class_column.metric("Final classification", "Review" if predicted_class else "Acceptable")
    st.progress(probability, text=f"Estimated defect probability: {probability:.1%}")

    if risk == "High":
        st.error("High predicted risk — prioritize physical quality inspection.")
    elif risk == "Medium":
        st.warning("Medium predicted risk — review operating conditions and inspection priority.")
    else:
        st.success("Low predicted risk — standard physical inspection still applies.")

    st.markdown("#### Factors associated with this screening result")
    for factor in prediction_explanation(record):
        st.write(f"- {factor}")
    st.caption(
        "These factors provide context from synthetic associations; they are not a causal explanation of the prediction."
    )


def main() -> None:
    """Load application resources and render the three dashboard pages."""
    st.title("Production Quality Dashboard and Defect Prediction")
    st.markdown(
        '<div class="synthetic-banner"><strong>Synthetic data notice:</strong> all records and findings in this application are simulated for an independent portfolio project.</div>',
        unsafe_allow_html=True,
    )
    try:
        production_data = load_production_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Dashboard data unavailable: {error}")
        st.stop()

    filtered_data = apply_quality_filters(production_data)
    overview_tab, quality_tab, prediction_tab = st.tabs(
        ["Production overview", "Quality analysis", "Defect prediction"]
    )
    with overview_tab:
        render_overview(production_data)
    with quality_tab:
        render_quality_analysis(filtered_data)
    with prediction_tab:
        render_prediction_page()

    st.divider()
    st.caption(
        "Demonstration only — synthetic data, predictive associations rather than causal conclusions, "
        "and no replacement for production expertise or physical inspection."
    )


if __name__ == "__main__":
    main()
