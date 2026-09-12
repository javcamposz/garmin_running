from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demo_data import build_demo_export
from garmin_etl import (
    DEFAULT_GOAL_MINUTES,
    HALF_MARATHON_KM,
    build_garmin_data,
    extract_garmin_export,
    find_export_zip,
    format_duration,
    format_pace,
    generate_half_marathon_plan,
    next_monday,
    parse_goal_time_to_minutes,
)


ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
CAMBRIDGE_2027_RACE_DATE = date(2027, 3, 14)
CAMBRIDGE_2027_TARGET_MINUTES = 98.0


st.set_page_config(
    page_title="Garmin Running Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float) -> dict:
    del mtime
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Processing Garmin export...")
def load_uploaded_export(uploaded_zip: bytes, filename: str) -> dict[str, pd.DataFrame | dict]:
    del filename
    return build_garmin_data(uploaded_zip)


@st.cache_data(show_spinner=False)
def load_demo_data() -> dict[str, pd.DataFrame | dict]:
    return build_garmin_data(build_demo_export())


def ensure_processed_data() -> None:
    required = [
        PROCESSED / "running_activities.csv",
        PROCESSED / "weekly_summary.csv",
        PROCESSED / "race_predictions.csv",
        PROCESSED / "vo2max_history.csv",
        PROCESSED / "training_history.csv",
        PROCESSED / "half_marathon_plan.csv",
        PROCESSED / "insights.json",
    ]
    if all(path.exists() for path in required):
        return
    export_zip = find_export_zip(ROOT)
    extract_garmin_export(export_zip, PROCESSED)


def read_data(uploaded_zip: bytes | None = None, uploaded_name: str | None = None) -> dict[str, pd.DataFrame | dict]:
    if uploaded_zip:
        return load_uploaded_export(uploaded_zip, uploaded_name or "garmin-export.zip")

    ensure_processed_data()
    data: dict[str, pd.DataFrame | dict] = {}
    csv_names = [
        "running_activities",
        "running_splits",
        "running_split_summaries",
        "weekly_summary",
        "race_predictions",
        "vo2max_history",
        "training_history",
        "personal_records",
        "heart_rate_zones",
        "bio_profile",
        "half_marathon_plan",
    ]
    for name in csv_names:
        path = PROCESSED / f"{name}.csv"
        data[name] = load_csv(str(path), path.stat().st_mtime) if path.exists() else pd.DataFrame()
    insights_path = PROCESSED / "insights.json"
    data["insights"] = load_json(str(insights_path), insights_path.stat().st_mtime)
    return data


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def empty_figure(message: str, height: int = 330) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def filter_by_date(df: pd.DataFrame, date_col: str, start_filter: date, end_filter: date) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.copy()
    return df[
        (df[date_col].dt.date >= start_filter)
        & (df[date_col].dt.date <= end_filter)
    ].copy()


def latest_row(df: pd.DataFrame, sort_col: str) -> pd.Series | None:
    if df.empty or sort_col not in df.columns:
        return None
    ordered = df.dropna(subset=[sort_col]).sort_values(sort_col)
    if ordered.empty:
        return None
    return ordered.iloc[-1]


def week_window(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    latest = df["start_time_local"].max()
    return df[df["start_time_local"] > latest - pd.Timedelta(weeks=weeks)].copy()


def duration_from_minutes(minutes: float | None) -> str:
    if minutes is None or pd.isna(minutes):
        return ""
    return format_duration(float(minutes) * 60)


def display_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%d %b %Y")


def metric_delta(current: float, previous: float, suffix: str = "") -> str | None:
    if pd.isna(current) or pd.isna(previous):
        return None
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}{suffix}"


def minutes_from_display(value: str | None) -> float:
    if not value:
        return np.nan
    try:
        return parse_goal_time_to_minutes(value)
    except ValueError:
        return np.nan


def time_gap_text(current_minutes: float, target_minutes: float) -> str:
    if pd.isna(current_minutes):
        return ""
    gap = current_minutes - target_minutes
    direction = "faster needed" if gap > 0 else "inside target"
    return f"{duration_from_minutes(abs(gap))} {direction}"


def optional_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def status_label(current: float, target: float, higher_is_better: bool = True, tolerance: float = 0.0) -> str:
    if pd.isna(current):
        return "No data"
    if higher_is_better:
        if current >= target:
            return "On track"
        if current >= target * (1 - tolerance):
            return "Close"
        return "Needs build"
    if current <= target:
        return "On track"
    if current <= target * (1 + tolerance):
        return "Close"
    return "Needs build"


def build_cambridge_2027_phase_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "Foundation",
                "dates": "May-Jul 2026",
                "weekly_km": "35-42",
                "focus": "Easy volume, strides, strength, clean recovery habits",
                "key_work": "1 light hill/stride session; long run 14-17 km",
            },
            {
                "phase": "Base Build",
                "dates": "Jul-Sep 2026",
                "weekly_km": "40-48",
                "focus": "Five-run rhythm or four runs plus aerobic cross-training",
                "key_work": "Threshold intro; long run 17-19 km",
            },
            {
                "phase": "Strength + Speed",
                "dates": "Sep-Nov 2026",
                "weekly_km": "45-52",
                "focus": "Running economy and 10K strength",
                "key_work": "1 km reps, hills, tune-up 5K/10K",
            },
            {
                "phase": "Threshold Base",
                "dates": "Nov-Jan 2027",
                "weekly_km": "48-56",
                "focus": "More time near threshold without turning every week hard",
                "key_work": "2 x 20 min threshold; long run 18-21 km",
            },
            {
                "phase": "Half-Specific",
                "dates": "Jan-Feb 2027",
                "weekly_km": "52-60",
                "focus": "Hold 4:39/km under controlled fatigue",
                "key_work": "3 x 3 km at HM pace; 16-18 km with HM blocks",
            },
            {
                "phase": "Taper",
                "dates": "Feb-Mar 2027",
                "weekly_km": "60% -> race",
                "focus": "Keep sharpness, reduce fatigue, protect sleep",
                "key_work": "Short HM-pace touches; race on 14 Mar",
            },
        ]
    )


def build_weekly_range_chart(phase_plan: pd.DataFrame) -> go.Figure:
    ranges = []
    for _, row in phase_plan.iterrows():
        text = str(row["weekly_km"])
        if "-" in text:
            low, high = text.split("-", maxsplit=1)
            try:
                ranges.append({"phase": row["phase"], "low": float(low), "high": float(high)})
            except ValueError:
                continue
    df = pd.DataFrame(ranges)
    fig = go.Figure()
    fig.add_bar(
        x=df["phase"],
        y=df["high"] - df["low"],
        base=df["low"],
        marker_color="#2B7A78",
        name="weekly km range",
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title=None,
        yaxis_title="weekly km",
        showlegend=False,
    )
    return fig


def add_weekly_volume_chart(weekly: pd.DataFrame) -> go.Figure:
    required_cols = {"week_start_dt", "distance_km", "acute_4w_km", "chronic_12w_km"}
    if weekly.empty or not required_cols.issubset(weekly.columns):
        return empty_figure("No weekly volume data", height=390)

    fig = go.Figure()
    fig.add_bar(
        x=weekly["week_start_dt"],
        y=weekly["distance_km"],
        name="Weekly km",
        marker_color="#2B7A78",
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start_dt"],
            y=weekly["acute_4w_km"],
            mode="lines",
            name="4-week average",
            line=dict(color="#17252A", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start_dt"],
            y=weekly["chronic_12w_km"],
            mode="lines",
            name="12-week average",
            line=dict(color="#C1666B", width=2),
        )
    )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="km",
        xaxis_title=None,
        legend_orientation="h",
        legend_y=1.08,
    )
    return fig


def add_training_load_chart(training: pd.DataFrame) -> go.Figure:
    required_cols = {"calendarDate", "loadTunnelMax", "loadTunnelMin", "weeklyTrainingLoadSum"}
    if training.empty or not required_cols.issubset(training.columns):
        return empty_figure("No Garmin training load data", height=360)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=training["calendarDate"],
            y=training["loadTunnelMax"],
            mode="lines",
            name="Load tunnel max",
            line=dict(color="#B8B8B8", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training["calendarDate"],
            y=training["loadTunnelMin"],
            mode="lines",
            name="Load tunnel min",
            fill="tonexty",
            fillcolor="rgba(184,184,184,0.22)",
            line=dict(color="#B8B8B8", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training["calendarDate"],
            y=training["weeklyTrainingLoadSum"],
            mode="lines",
            name="Garmin load",
            line=dict(color="#2B7A78", width=2),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="training load",
        xaxis_title=None,
        legend_orientation="h",
        legend_y=1.08,
    )
    return fig


def add_hr_zone_chart(df: pd.DataFrame) -> go.Figure:
    zones = [f"hr_zone_{zone}_min" for zone in range(7)]
    zone_names = ["Below Z1", "Z1", "Z2", "Z3", "Z4", "Z5", "Above Z5"]
    minutes = [df[col].sum() if col in df else 0 for col in zones]
    fig = go.Figure(
        go.Bar(
            x=zone_names,
            y=minutes,
            marker_color=["#CFD8DC", "#7BAFD4", "#6BBF59", "#E6C84F", "#E58E3E", "#C94C4C", "#7A3E65"],
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis_title="minutes",
        xaxis_title=None,
    )
    return fig


def hr_zone_definitions(heart_rate_zones: pd.DataFrame) -> pd.DataFrame:
    if heart_rate_zones.empty:
        floors = {
            "zone1Floor": 90,
            "zone2Floor": 108,
            "zone3Floor": 126,
            "zone4Floor": 144,
            "zone5Floor": 162,
            "maxHeartRateUsed": 180,
        }
    else:
        floors = heart_rate_zones.iloc[0].to_dict()

    rows = [
        ("Below Z1", 0, floors.get("zone1Floor", 90) - 1, "Very easy / warm-up"),
        ("Z1", floors.get("zone1Floor", 90), floors.get("zone2Floor", 108) - 1, "Recovery"),
        ("Z2", floors.get("zone2Floor", 108), floors.get("zone3Floor", 126) - 1, "Easy aerobic"),
        ("Z3", floors.get("zone3Floor", 126), floors.get("zone4Floor", 144) - 1, "Aerobic endurance"),
        ("Z4", floors.get("zone4Floor", 144), floors.get("zone5Floor", 162) - 1, "Steady / threshold drift"),
        ("Z5", floors.get("zone5Floor", 162), floors.get("maxHeartRateUsed", 180), "Hard intervals / race effort"),
        ("Above Z5", floors.get("maxHeartRateUsed", 180) + 1, floors.get("maxHeartRateUsed", 180) + 20, "Very hard / avoid on easy days"),
    ]
    return pd.DataFrame(rows, columns=["zone", "from_bpm", "to_bpm", "use"])


def zone_time_table(df: pd.DataFrame, heart_rate_zones: pd.DataFrame) -> pd.DataFrame:
    zone_defs = hr_zone_definitions(heart_rate_zones)
    rows = []
    total = sum(df.get(f"hr_zone_{zone}_min", pd.Series(dtype=float)).sum() for zone in range(7))
    zone_names = ["Below Z1", "Z1", "Z2", "Z3", "Z4", "Z5", "Above Z5"]
    for zone, name in enumerate(zone_names):
        minutes = df.get(f"hr_zone_{zone}_min", pd.Series(dtype=float)).sum()
        definition = zone_defs[zone_defs["zone"].eq(name)].iloc[0]
        rows.append(
            {
                "zone": name,
                "bpm": f"{int(definition['from_bpm'])}-{int(definition['to_bpm'])}",
                "use": definition["use"],
                "minutes": round(float(minutes), 1),
                "share": round(float(minutes / total * 100), 1) if total else 0,
            }
        )
    return pd.DataFrame(rows)


def zone_time_by_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    zone_names = ["Below Z1", "Z1", "Z2", "Z3", "Z4", "Z5", "Above Z5"]
    for category, group in df.groupby("activity_category"):
        for zone, name in enumerate(zone_names):
            rows.append(
                {
                    "activity_category": category,
                    "zone": name,
                    "minutes": group.get(f"hr_zone_{zone}_min", pd.Series(dtype=float)).sum(),
                }
            )
    return pd.DataFrame(rows)


def easy_pace_recommendation(valid_activities: pd.DataFrame, heart_rate_zones: pd.DataFrame) -> dict[str, str | float]:
    latest = valid_activities["start_time_local"].max()
    recent = valid_activities[valid_activities["start_time_local"] > latest - pd.Timedelta(weeks=52)].copy()
    zone_defs = hr_zone_definitions(heart_rate_zones)
    z4_floor = float(zone_defs.loc[zone_defs["zone"].eq("Z4"), "from_bpm"].iloc[0])

    candidates = recent[
        (recent["avg_hr"] < z4_floor)
        & (recent["hr_high_share"].fillna(1) <= 0.25)
        & (recent["distance_km"].between(4, 16))
    ].copy()
    source = "low-HR recent runs"
    if len(candidates) < 6:
        candidates = recent[recent["activity_category"].eq("Recovery")].copy()
        source = "recent recovery runs"
    if len(candidates) < 6:
        candidates = recent[(recent["avg_hr"] < z4_floor) & (recent["distance_km"].between(4, 16))].copy()
        source = "recent runs below the Z4 floor"

    if candidates.empty:
        easy_low = 6.0
        easy_high = 6.5
        median = 6.25
    else:
        easy_low = float(candidates["pace_min_per_km"].quantile(0.25))
        easy_high = float(candidates["pace_min_per_km"].quantile(0.75))
        median = float(candidates["pace_min_per_km"].median())

    recovery_high = easy_high + 0.25
    return {
        "source": source,
        "sample_size": int(len(candidates)),
        "easy_low": easy_low,
        "easy_high": easy_high,
        "easy_range": f"{format_pace(easy_low)} to {format_pace(easy_high)}",
        "recovery_range": f"{format_pace(easy_high)} to {format_pace(recovery_high)}",
        "median": median,
        "hr_cap": z4_floor - 1,
    }


def vo2_context_table(valid_activities: pd.DataFrame, vo2: pd.DataFrame) -> pd.DataFrame:
    has_vo2 = not vo2.empty and {"calendarDate", "vo2MaxValue"}.issubset(vo2.columns)
    current_vo2 = float(vo2.sort_values("calendarDate").iloc[-1]["vo2MaxValue"]) if has_vo2 else np.nan
    peak_vo2 = float(vo2["vo2MaxValue"].max()) if has_vo2 else np.nan
    peak_year = int(vo2.loc[vo2["vo2MaxValue"].idxmax(), "calendarDate"].year) if has_vo2 else None

    latest = valid_activities["start_time_local"].max()
    recent = valid_activities[valid_activities["start_time_local"] > latest - pd.Timedelta(weeks=52)].copy()
    peak_year_runs = (
        valid_activities[valid_activities["start_time_local"].dt.year.eq(peak_year)].copy()
        if peak_year is not None
        else pd.DataFrame()
    )

    def summarize(label: str, run_df: pd.DataFrame, vo2_value: float) -> dict[str, str | float]:
        if run_df.empty:
            return {"period": label, "vo2max": vo2_value, "weekly_km": np.nan, "runs_per_week": np.nan, "median_pace": "", "longest_run": np.nan}
        days = max((run_df["start_time_local"].max() - run_df["start_time_local"].min()).days + 1, 7)
        weeks = days / 7
        return {
            "period": label,
            "vo2max": vo2_value,
            "weekly_km": round(float(run_df["distance_km"].sum() / weeks), 1),
            "runs_per_week": round(float(len(run_df) / weeks), 1),
            "median_pace": format_pace(float(run_df["pace_min_per_km"].median())),
            "longest_run": round(float(run_df["distance_km"].max()), 1),
        }

    return pd.DataFrame(
        [
            summarize("Last 52 weeks", recent, current_vo2),
            summarize(f"Peak VO2 year ({peak_year})", peak_year_runs, peak_vo2),
        ]
    )


def classify_effort(row: pd.Series, easy_hr_cap: float) -> str:
    category = str(row.get("activity_category", ""))
    high_share = row.get("hr_high_share", np.nan)
    avg_hr = row.get("avg_hr", np.nan)
    training_effect = row.get("aerobic_training_effect", np.nan)

    if category in {"Race", "Workout"}:
        return "Hard"
    if not pd.isna(high_share) and high_share >= 0.5:
        return "Hard"
    if not pd.isna(training_effect) and training_effect >= 4.0:
        return "Hard"
    if not pd.isna(avg_hr) and avg_hr >= easy_hr_cap + 6:
        return "Hard"
    if not pd.isna(high_share) and high_share >= 0.25:
        return "Steady"
    if not pd.isna(training_effect) and training_effect >= 3.0:
        return "Steady"
    return "Easy"


def recent_run_note(row: pd.Series, easy_hr_cap: float) -> str:
    high_share_pct = row.get("high_zone_share_pct", np.nan)
    gap = row.get("recovery_gap_days", np.nan)
    avg_hr = row.get("avg_hr", np.nan)
    category = str(row.get("activity_category", ""))
    effort = str(row.get("effort", ""))

    if category in {"Recovery", "Base"} and not pd.isna(high_share_pct) and high_share_pct > 45:
        return "Easy-labeled run carried hard-zone time"
    if effort == "Hard" and not pd.isna(gap) and gap < 1.5:
        return "Hard run after a short gap"
    if category == "Recovery" and not pd.isna(avg_hr) and avg_hr > easy_hr_cap:
        return "Recovery HR was above easy cap"
    if row.get("distance_km", 0) >= 14:
        return "Long-run stimulus"
    return ""


def build_recent_runs(valid_activities: pd.DataFrame, count: int, easy_hr_cap: float) -> pd.DataFrame:
    recent = valid_activities.sort_values("start_time_local").tail(count).copy()
    if recent.empty:
        return recent

    if "hr_high_share" not in recent.columns:
        recent["hr_high_share"] = np.nan
    recent["recovery_gap_days"] = recent["start_time_local"].diff().dt.total_seconds().div(86400).round(1)
    recent["high_zone_share_pct"] = (recent["hr_high_share"] * 100).round(1)
    recent["time"] = recent["moving_min"].apply(duration_from_minutes)
    recent["date"] = recent["start_time_local"].dt.strftime("%d %b")
    recent["effort"] = recent.apply(lambda row: classify_effort(row, easy_hr_cap), axis=1)
    recent["note"] = recent.apply(lambda row: recent_run_note(row, easy_hr_cap), axis=1)
    return recent


def recent_block_summary(valid_activities: pd.DataFrame, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = valid_activities.sort_values("start_time_local")
    recent = ordered.tail(count).copy()
    previous = ordered.iloc[max(0, len(ordered) - count * 2): max(0, len(ordered) - count)].copy()
    return recent, previous


def build_recent_action_notes(
    recent: pd.DataFrame,
    previous: pd.DataFrame,
    easy_guidance: dict[str, str | float],
) -> pd.DataFrame:
    notes: list[dict[str, str]] = []
    if recent.empty:
        return pd.DataFrame(columns=["area", "signal", "next_step"])

    hard_runs = int(recent["effort"].eq("Hard").sum()) if "effort" in recent.columns else 0
    easy_labeled_hard = recent[
        recent["activity_category"].isin(["Recovery", "Base"])
        & recent["high_zone_share_pct"].fillna(0).gt(45)
    ]
    total_km = float(recent["distance_km"].sum())
    previous_km = float(previous["distance_km"].sum()) if not previous.empty else np.nan
    longest = float(recent["distance_km"].max())

    if not easy_labeled_hard.empty:
        notes.append(
            {
                "area": "Easy days",
                "signal": f"{len(easy_labeled_hard)} base/recovery runs had >45% high-zone time",
                "next_step": f"Make the next easy run {easy_guidance['easy_range']} and cap HR near {easy_guidance['hr_cap']:.0f} bpm.",
            }
        )
    if hard_runs >= max(3, len(recent) // 3):
        notes.append(
            {
                "area": "Intensity",
                "signal": f"{hard_runs} of the last {len(recent)} runs classified as hard",
                "next_step": "Keep only one clear workout before the next long run and let the other runs stay conversational.",
            }
        )
    if not pd.isna(previous_km) and previous_km > 0 and total_km > previous_km * 1.18:
        notes.append(
            {
                "area": "Load jump",
                "signal": f"Recent block rose from {previous_km:.1f} km to {total_km:.1f} km",
                "next_step": "Hold volume steady for a week before adding distance or harder work.",
            }
        )
    if longest < 14:
        notes.append(
            {
                "area": "Endurance",
                "signal": f"Longest recent run is {longest:.1f} km",
                "next_step": "Add 1-2 km to the long run every 1-2 weeks until 16-18 km feels routine.",
            }
        )
    if not notes:
        notes.append(
            {
                "area": "Recent balance",
                "signal": "No obvious load or intensity warning in the selected runs",
                "next_step": "Keep the pattern steady and use the next workout for controlled threshold or hills, not a race effort.",
            }
        )
    return pd.DataFrame(notes)


def add_recent_pace_hr_chart(recent: pd.DataFrame) -> go.Figure:
    if recent.empty:
        return empty_figure("No recent runs selected", height=340)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["start_time_local"],
            y=recent["pace_min_per_km"],
            mode="lines+markers",
            name="Pace",
            line=dict(color="#2B7A78", width=3),
            text=recent["name"],
        )
    )
    if "avg_hr" in recent.columns and recent["avg_hr"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=recent["start_time_local"],
                y=recent["avg_hr"],
                mode="lines+markers",
                name="Avg HR",
                yaxis="y2",
                line=dict(color="#C1666B", width=2),
            )
        )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title=None,
        yaxis=dict(title="min/km", autorange="reversed"),
        yaxis2=dict(title="avg HR", overlaying="y", side="right"),
        legend_orientation="h",
        legend_y=1.12,
    )
    return fig


def add_recent_intensity_chart(recent: pd.DataFrame) -> go.Figure:
    if recent.empty or "high_zone_share_pct" not in recent.columns:
        return empty_figure("No recent intensity data", height=340)
    fig = px.bar(
        recent,
        x="date",
        y="high_zone_share_pct",
        color="effort",
        hover_data=["name", "distance_km", "pace", "avg_hr"],
        color_discrete_map={"Easy": "#6BBF59", "Steady": "#E6C84F", "Hard": "#C94C4C"},
    )
    fig.add_hline(y=40, line_dash="dot", line_color="#17252A", annotation_text="easy-day ceiling")
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title=None,
        yaxis_title="% high-zone time",
        legend_orientation="h",
        legend_y=1.12,
    )
    return fig


def plan_weeks_from_dates(start: date, race: date) -> int:
    days = max((race - start).days + 1, 1)
    return min(12, max(8, int(np.ceil(days / 7))))


def main() -> None:
    with st.sidebar:
        st.header("Data source")
        source_mode = st.radio(
            "Mode",
            ["Demo data", "Upload Garmin export", "Local files"],
            help="Demo data is synthetic. Uploaded files remain in memory for this browser session.",
        )
        uploaded_export = None
        if source_mode == "Upload Garmin export":
            uploaded_export = st.file_uploader(
                "Garmin export .zip",
                type="zip",
                help="Upload the full Garmin Account Management export.",
            )
            if uploaded_export is not None:
                st.caption("Using the uploaded export for this browser session.")
        elif source_mode == "Demo data":
            st.caption("Using deterministic synthetic runs. No personal data is included.")
        else:
            st.caption("Using local processed files, or a Garmin export in data/raw/.")

    try:
        if source_mode == "Demo data":
            data = load_demo_data()
        else:
            data = read_data(
                uploaded_export.getvalue() if uploaded_export is not None else None,
                uploaded_export.name if uploaded_export is not None else None,
            )
    except FileNotFoundError:
        st.title("Garmin Running Dashboard")
        st.info(
            "No local Garmin data was found. Upload your Garmin export .zip in the sidebar, "
            "or place one export zip in data/raw/ and rerun the app."
        )
        st.stop()
    except BadZipFile:
        st.title("Garmin Running Dashboard")
        st.error("The uploaded file is not a readable zip file. Upload the Garmin export zip without unzipping it.")
        st.stop()

    activities = parse_dates(
        data["running_activities"].copy(),
        ["start_time_local", "start_time_gmt", "begin_time"],
    )
    weekly = parse_dates(data["weekly_summary"].copy(), ["week_start_dt", "week_start"])
    race_predictions = parse_dates(data["race_predictions"].copy(), ["calendarDate", "timestamp_dt"])
    vo2 = parse_dates(data["vo2max_history"].copy(), ["calendarDate", "updateTimestamp_dt"])
    training = parse_dates(data["training_history"].copy(), ["calendarDate", "timestamp_dt"])
    personal_records = data["personal_records"].copy()
    heart_rate_zones = data["heart_rate_zones"].copy()
    insights = data["insights"]

    if activities.empty or "start_time_local" not in activities.columns:
        st.title("Garmin Running Dashboard")
        st.warning("No running activities were found in this Garmin export.")
        st.stop()

    valid_mask = activities.get("is_valid_for_analysis", pd.Series(False, index=activities.index)).fillna(False)
    valid_activities = activities[valid_mask == True].copy()
    if valid_activities.empty:
        st.title("Garmin Running Dashboard")
        st.warning("Running activities were found, but none passed the analysis filters.")
        st.dataframe(activities, hide_index=True, use_container_width=True)
        st.stop()

    min_date = valid_activities["start_time_local"].min().date()
    max_date = valid_activities["start_time_local"].max().date()

    with st.sidebar:
        st.header("Controls")
        if source_mode == "Local files":
            if st.button("Rebuild Garmin data", use_container_width=True):
                extract_garmin_export(find_export_zip(ROOT), PROCESSED)
                st.cache_data.clear()
                st.rerun()
        elif source_mode == "Upload Garmin export":
            st.caption("Uploaded data is processed in memory. Rebuild only applies to local files.")

        date_range = st.date_input(
            "Activity dates",
            value=(max(min_date, max_date - timedelta(days=365)), max_date),
            min_value=min_date,
            max_value=max_date,
        )
        include_outliers = st.toggle("Include flagged runs", value=False)
        age = st.number_input("Age", min_value=18, max_value=90, value=54, step=1)
        goal_text = st.text_input("Half marathon target", value=insights["half_marathon"]["suggested_goal_time"])
        start_default = next_monday(date.today())
        start_date = st.date_input("Plan starts", value=start_default)
        race_default = start_date + timedelta(weeks=12) - timedelta(days=1)
        race_date = st.date_input("Race date", value=race_default, min_value=start_date + timedelta(weeks=8))

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_filter, end_filter = date_range
    else:
        start_filter, end_filter = min_date, max_date

    base_df = activities if include_outliers else valid_activities
    filtered = base_df[
        (base_df["start_time_local"].dt.date >= start_filter)
        & (base_df["start_time_local"].dt.date <= end_filter)
    ].copy()
    filtered_weekly = filter_by_date(weekly, "week_start_dt", start_filter, end_filter)
    filtered_training = filter_by_date(training, "calendarDate", start_filter, end_filter)

    try:
        goal_minutes = parse_goal_time_to_minutes(goal_text)
    except ValueError:
        st.sidebar.error("Use H:MM:SS or MM:SS for the target.")
        goal_minutes = DEFAULT_GOAL_MINUTES

    recent_4 = week_window(valid_activities, 4)
    recent_8 = week_window(valid_activities, 8)
    recent_12 = week_window(valid_activities, 12)
    recent_52 = week_window(valid_activities, 52)
    recent_weekly_km = recent_8["distance_km"].sum() / 8
    plan_weeks = plan_weeks_from_dates(start_date, race_date)
    plan = generate_half_marathon_plan(
        start_date=start_date,
        weeks=plan_weeks,
        base_weekly_km=recent_weekly_km,
        goal_minutes=goal_minutes,
    )

    latest_vo2 = latest_row(vo2, "calendarDate")
    latest_prediction = latest_row(race_predictions, "calendarDate")
    latest_training = latest_row(training, "calendarDate")
    latest_vo2_date = (
        display_date(latest_vo2["calendarDate"])
        if latest_vo2 is not None and "calendarDate" in latest_vo2
        else ""
    )
    latest_prediction_date = (
        display_date(latest_prediction["calendarDate"])
        if latest_prediction is not None and "calendarDate" in latest_prediction
        else ""
    )
    latest_half = insights.get("half_marathon", {}).get("latest", {})
    best_half = insights.get("half_marathon", {}).get("best", {})
    cambridge_target_pace = CAMBRIDGE_2027_TARGET_MINUTES / HALF_MARATHON_KM
    cambridge_days = max((CAMBRIDGE_2027_RACE_DATE - date.today()).days, 0)
    cambridge_weeks = cambridge_days / 7
    latest_half_minutes = minutes_from_display(latest_half.get("time"))
    best_half_minutes = minutes_from_display(best_half.get("time"))
    predicted_half_minutes = (
        float(latest_prediction["raceTimeHalf"]) / 60
        if latest_prediction is not None
        and "raceTimeHalf" in latest_prediction
        and not pd.isna(latest_prediction["raceTimeHalf"])
        else np.nan
    )
    best_gap_pct = (
        (best_half_minutes - CAMBRIDGE_2027_TARGET_MINUTES) / best_half_minutes * 100
        if not pd.isna(best_half_minutes)
        else np.nan
    )
    latest_gap_pct = (
        (latest_half_minutes - CAMBRIDGE_2027_TARGET_MINUTES) / latest_half_minutes * 100
        if not pd.isna(latest_half_minutes)
        else np.nan
    )
    current_run_frequency = len(recent_8) / 8
    high_hr_share_8 = insights.get("recent", {}).get("8_weeks", {}).get("high_hr_zone_share_pct", np.nan)
    if high_hr_share_8 is None:
        high_hr_share_8 = np.nan
    recent_longest_8 = recent_8["distance_km"].max()
    phase_plan_2027 = build_cambridge_2027_phase_plan()
    easy_guidance = easy_pace_recommendation(valid_activities, heart_rate_zones)

    best_10k = valid_activities[valid_activities["distance_km"].between(9.8, 10.3)].copy()
    best_10k_minutes = best_10k["moving_min"].min() if not best_10k.empty else np.nan
    best_5k = valid_activities[valid_activities["distance_km"].between(4.8, 5.2)].copy()
    best_5k_minutes = best_5k["moving_min"].min() if not best_5k.empty else np.nan

    st.title("Garmin Running Dashboard")
    data_insights = insights.get("data", {})
    first_activity = data_insights.get("first_activity_date")
    latest_activity = data_insights.get("activity_data_through")
    if source_mode == "Demo data":
        st.caption("Synthetic demo data. Upload your Garmin export to analyse your own history.")
    elif first_activity and latest_activity:
        st.caption(f"Running history from {first_activity} to {latest_activity}.")
    else:
        st.caption("Running history loaded from the selected Garmin export.")

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    with kpi_1:
        st.metric("Recent weekly km", f"{recent_weekly_km:.1f}", metric_delta(recent_4["distance_km"].sum() / 4, recent_weekly_km, " km"))
    with kpi_2:
        st.metric("Last 8-week longest", f"{recent_8['distance_km'].max():.1f} km")
    with kpi_3:
        st.metric(
            "Latest VO2 max",
            f"{latest_vo2['vo2MaxValue']:.0f}"
            if latest_vo2 is not None
            and "vo2MaxValue" in latest_vo2
            and not pd.isna(latest_vo2["vo2MaxValue"])
            else "",
            help="Latest VO2 max record found in the uploaded Garmin export.",
        )
        if latest_vo2_date:
            st.caption(f"Latest record: {latest_vo2_date}")
    with kpi_4:
        pred = (
            format_duration(latest_prediction["raceTimeHalf"])
            if latest_prediction is not None
            and "raceTimeHalf" in latest_prediction
            and not pd.isna(latest_prediction["raceTimeHalf"])
            else ""
        )
        st.metric("Garmin predicted HM", pred, help="Garmin race prediction, not a completed race activity.")
        if pred and latest_prediction_date:
            st.caption(f"Prediction date: {latest_prediction_date}")
    with kpi_5:
        load_label = latest_training["trainingStatus"] if latest_training is not None and "trainingStatus" in latest_training else ""
        load_value = (
            int(latest_training["weeklyTrainingLoadSum"])
            if latest_training is not None
            and "weeklyTrainingLoadSum" in latest_training
            and not pd.isna(latest_training["weeklyTrainingLoadSum"])
            else ""
        )
        st.metric("Training load", load_value, load_label)

    overview_tab, recent_runs_tab, performance_tab, load_tab, zones_vo2_tab, plan_tab, cambridge_2027_tab, data_tab = st.tabs(
        [
            "Overview",
            "Recent Runs",
            "Performance",
            "Training Load",
            "Zones & VO2",
            "Half Marathon Plan",
            "Cambridge 2027",
            "Data",
        ]
    )

    with overview_tab:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Weekly Volume")
            st.plotly_chart(add_weekly_volume_chart(filtered_weekly), use_container_width=True)
        with right:
            st.subheader("Current Read")
            st.write(f"Valid running activities: **{data_insights.get('valid_running_activities', len(valid_activities)):,}**")
            st.write(f"Valid distance: **{data_insights.get('total_valid_distance_km', valid_activities['distance_km'].sum()):,.0f} km**")
            st.write(f"Flagged outliers: **{data_insights.get('flagged_outliers', 0)}**")
            latest_half_date = display_date(latest_half.get("date"))
            best_half_date = display_date(best_half.get("date"))
            st.write(
                f"Latest completed HM effort: {latest_half.get('time', '')} "
                f"at {latest_half.get('pace', '')}"
                f"{f' ({latest_half_date})' if latest_half_date else ''}"
            )
            st.write(
                f"Best completed HM effort: {best_half.get('time', '')} "
                f"at {best_half.get('pace', '')}"
                f"{f' ({best_half_date})' if best_half_date else ''}"
            )

        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.subheader("Training Insights")
            for item in insights["recommendations"]:
                st.write(f"- {item}")
        with insight_cols[1]:
            st.subheader("Recent Windows")
            recent_table = pd.DataFrame(insights["recent"]).T.reset_index(names="window")
            st.dataframe(
                recent_table[
                    [
                        "window",
                        "runs",
                        "distance_km",
                        "avg_weekly_km",
                        "longest_run_km",
                        "avg_pace",
                        "avg_hr",
                        "high_hr_zone_share_pct",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Activity Mix")
        mix = (
            filtered.groupby("activity_category", as_index=False)
            .agg(runs=("activity_id", "count"), distance_km=("distance_km", "sum"))
            .sort_values("distance_km", ascending=False)
        )
        fig_mix = px.bar(
            mix,
            x="activity_category",
            y="distance_km",
            color="activity_category",
            text="runs",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_mix.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), showlegend=False)
        st.plotly_chart(fig_mix, use_container_width=True)

    with recent_runs_tab:
        st.subheader("Last Few Runs")
        max_recent = min(20, max(1, len(valid_activities)))
        min_recent = min(4, max_recent)
        default_recent = min(8, max_recent)
        recent_count = st.slider(
            "Runs to inspect",
            min_value=min_recent,
            max_value=max_recent,
            value=default_recent,
            step=1,
        )
        recent_base, previous_base = recent_block_summary(valid_activities, recent_count)
        recent_runs = build_recent_runs(valid_activities, recent_count, float(easy_guidance["hr_cap"]))
        previous_total_km = previous_base["distance_km"].sum() if not previous_base.empty else np.nan
        recent_total_km = recent_base["distance_km"].sum()
        recent_avg_pace = weighted_avg(recent_base["pace_min_per_km"], recent_base["distance_km"])
        recent_avg_hr = weighted_avg(recent_base["avg_hr"], recent_base["moving_min"])
        median_gap = recent_runs["recovery_gap_days"].dropna().median() if "recovery_gap_days" in recent_runs else np.nan

        summary_cols = st.columns(5)
        summary_cols[0].metric("Selected distance", f"{recent_total_km:.1f} km", metric_delta(recent_total_km, previous_total_km, " km"))
        summary_cols[1].metric("Average pace", format_pace(recent_avg_pace))
        summary_cols[2].metric("Average HR", f"{recent_avg_hr:.0f}" if not pd.isna(recent_avg_hr) else "")
        summary_cols[3].metric("Hard runs", int(recent_runs["effort"].eq("Hard").sum()) if "effort" in recent_runs else 0)
        summary_cols[4].metric("Median run gap", f"{median_gap:.1f} days" if not pd.isna(median_gap) else "")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.subheader("Pace And HR")
            st.plotly_chart(add_recent_pace_hr_chart(recent_runs), use_container_width=True)
        with chart_right:
            st.subheader("High-Zone Share")
            st.plotly_chart(add_recent_intensity_chart(recent_runs), use_container_width=True)

        st.subheader("What To Adjust Next")
        st.dataframe(
            build_recent_action_notes(recent_runs, previous_base, easy_guidance),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Recent Run Log")
        run_log_cols = [
            "date",
            "name",
            "activity_category",
            "effort",
            "distance_km",
            "time",
            "pace",
            "avg_hr",
            "high_zone_share_pct",
            "recovery_gap_days",
            "aerobic_training_effect",
            "note",
        ]
        shown_cols = [col for col in run_log_cols if col in recent_runs.columns]
        st.dataframe(
            recent_runs.sort_values("start_time_local", ascending=False)[shown_cols],
            hide_index=True,
            use_container_width=True,
        )

    with performance_tab:
        top, bottom = st.columns(2)
        with top:
            st.subheader("Pace Over Time")
            fig_pace = px.scatter(
                filtered,
                x="start_time_local",
                y="pace_min_per_km",
                size="distance_km",
                color="activity_category",
                hover_data=["name", "distance_km", "pace", "avg_hr", "aerobic_training_effect"],
                color_discrete_sequence=px.colors.qualitative.Dark24,
            )
            fig_pace.update_yaxes(autorange="reversed", title="min/km")
            fig_pace.update_layout(height=370, margin=dict(l=10, r=10, t=25, b=10), xaxis_title=None)
            st.plotly_chart(fig_pace, use_container_width=True)

        with bottom:
            st.subheader("Pace vs Heart Rate")
            hr_df = filtered[filtered["avg_hr"].notna()].copy()
            fig_hr = px.scatter(
                hr_df,
                x="avg_hr",
                y="pace_min_per_km",
                size="distance_km",
                color="activity_category",
                hover_data=["start_time_local", "name", "distance_km", "pace", "max_hr"],
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_hr.update_yaxes(autorange="reversed", title="min/km")
            fig_hr.update_layout(height=370, margin=dict(l=10, r=10, t=25, b=10), xaxis_title="avg HR")
            st.plotly_chart(fig_hr, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("VO2 Max")
            vo2_filtered = filter_by_date(vo2, "calendarDate", start_filter, end_filter)
            if vo2_filtered.empty or "vo2MaxValue" not in vo2_filtered.columns:
                st.plotly_chart(empty_figure("No VO2 max history"), use_container_width=True)
            else:
                fig_vo2 = px.line(vo2_filtered, x="calendarDate", y="vo2MaxValue", markers=True)
                fig_vo2.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), xaxis_title=None, yaxis_title="VO2 max")
                st.plotly_chart(fig_vo2, use_container_width=True)
                if latest_vo2_date:
                    st.caption(
                        f"Latest VO2 max in this export: {latest_vo2['vo2MaxValue']:.0f} "
                        f"on {latest_vo2_date}."
                    )

        with right:
            st.subheader("Garmin Race Predictions")
            st.caption("Prediction rows are Garmin estimates, not completed race activities.")
            pred_filtered = filter_by_date(race_predictions, "calendarDate", start_filter, end_filter)
            fig_pred = go.Figure()
            for col, label in [
                ("raceTime5K_min", "Predicted 5K"),
                ("raceTime10K_min", "Predicted 10K"),
                ("raceTimeHalf_min", "Predicted half"),
                ("raceTimeMarathon_min", "Predicted marathon"),
            ]:
                if col in pred_filtered:
                    fig_pred.add_trace(go.Scatter(x=pred_filtered["calendarDate"], y=pred_filtered[col], mode="lines", name=label))
            if not fig_pred.data:
                fig_pred = empty_figure("No Garmin race prediction history")
            fig_pred.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), xaxis_title=None, yaxis_title="minutes")
            st.plotly_chart(fig_pred, use_container_width=True)

        race_category = pd.Series(False, index=valid_activities.index)
        if "activity_category" in valid_activities:
            race_category = valid_activities["activity_category"].eq("Race")
        race_like = valid_activities[
            race_category | valid_activities["distance_km"].between(20.8, 21.5)
        ].copy()
        race_like["time"] = race_like["moving_min"].apply(duration_from_minutes)
        st.subheader("Completed Race and Half-Distance Efforts")
        st.dataframe(
            race_like.sort_values("start_time_local", ascending=False)[
                ["start_time_local", "name", "distance_km", "time", "pace", "avg_hr", "max_hr"]
            ].head(20),
            hide_index=True,
            use_container_width=True,
        )

    with load_tab:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Garmin Training Load")
            st.plotly_chart(add_training_load_chart(filtered_training), use_container_width=True)
        with right:
            st.subheader("Load Guardrails")
            if latest_training is not None:
                current_load = latest_training.get("weeklyTrainingLoadSum", np.nan)
                load_min = latest_training.get("loadTunnelMin", np.nan)
                load_max = latest_training.get("loadTunnelMax", np.nan)
                st.write(f"Current status: **{latest_training.get('trainingStatus', '')}**")
                st.write(f"Current load: **{current_load:.0f}**" if not pd.isna(current_load) else "Current load: **no data**")
                st.write(f"Garmin tunnel: **{load_min:.0f} to {load_max:.0f}**" if not pd.isna(load_min) and not pd.isna(load_max) else "Garmin tunnel: **no data**")
            else:
                st.write("No Garmin training load data found in this export.")
            high_hr_share_12 = insights.get("recent", {}).get("12_weeks", {}).get("high_hr_zone_share_pct", np.nan)
            st.write(f"Last 12-week high-HR share: **{optional_percent(high_hr_share_12)}**")
            if age >= 50:
                st.write("Masters guardrail: keep hard days separated by at least 48 hours when possible.")

        left, right = st.columns(2)
        with left:
            st.subheader("Heart Rate Zones")
            st.plotly_chart(add_hr_zone_chart(filtered), use_container_width=True)
        with right:
            st.subheader("Long Run Progression")
            fig_long = px.line(
                filtered_weekly,
                x="week_start_dt",
                y="long_run_km",
                markers=True,
            )
            fig_long.add_hline(y=18, line_dash="dot", line_color="#C1666B", annotation_text="18 km")
            fig_long.add_hline(y=21.1, line_dash="dot", line_color="#2B7A78", annotation_text="HM")
            fig_long.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), xaxis_title=None, yaxis_title="km")
            st.plotly_chart(fig_long, use_container_width=True)

        st.subheader("Acute / Chronic Distance")
        fig_acr = px.line(
            filtered_weekly,
            x="week_start_dt",
            y="acute_chronic_ratio",
            markers=True,
        )
        fig_acr.add_hrect(y0=0.8, y1=1.3, fillcolor="#6BBF59", opacity=0.12, line_width=0)
        fig_acr.add_hline(y=1.3, line_dash="dot", line_color="#C94C4C")
        fig_acr.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), xaxis_title=None, yaxis_title="ratio")
        st.plotly_chart(fig_acr, use_container_width=True)

    with zones_vo2_tab:
        st.subheader("Heart Rate Zone Drilldown")
        st.write(
            "Use this tab to keep easy days easy enough to support the harder work. "
            "The pace recommendation is derived from your recent low-HR and recovery runs, then capped by Garmin HR zones."
        )

        current_vo2 = (
            float(latest_vo2["vo2MaxValue"])
            if latest_vo2 is not None and "vo2MaxValue" in latest_vo2 and not pd.isna(latest_vo2["vo2MaxValue"])
            else np.nan
        )
        peak_vo2 = np.nan
        peak_vo2_date = None
        recent_peak_vo2 = np.nan
        recent_peak_vo2_date = None
        if not vo2.empty and {"vo2MaxValue", "calendarDate"}.issubset(vo2.columns):
            peak_vo2_row = vo2.sort_values(
                ["vo2MaxValue", "calendarDate"], ascending=[False, False]
            ).iloc[0]
            peak_vo2 = float(peak_vo2_row["vo2MaxValue"])
            peak_vo2_date = peak_vo2_row["calendarDate"].date()
            latest_vo2_dt = (
                pd.to_datetime(latest_vo2["calendarDate"], errors="coerce")
                if latest_vo2 is not None and "calendarDate" in latest_vo2
                else pd.NaT
            )
            recent_vo2 = vo2.copy()
            if not pd.isna(latest_vo2_dt):
                recent_vo2 = recent_vo2[recent_vo2["calendarDate"] >= latest_vo2_dt - pd.Timedelta(weeks=52)]
            if not recent_vo2.empty:
                recent_peak_row = recent_vo2.sort_values(
                    ["vo2MaxValue", "calendarDate"], ascending=[False, False]
                ).iloc[0]
                recent_peak_vo2 = float(recent_peak_row["vo2MaxValue"])
                recent_peak_vo2_date = recent_peak_row["calendarDate"].date()

        zone_cols = st.columns(5)
        zone_cols[0].metric("Easy pace now", easy_guidance["easy_range"])
        zone_cols[1].metric("Recovery pace", easy_guidance["recovery_range"])
        zone_cols[2].metric("Easy HR cap", f"{easy_guidance['hr_cap']:.0f} bpm")
        zone_cols[3].metric("Current VO2 max", f"{current_vo2:.0f}" if not pd.isna(current_vo2) else "")
        zone_cols[4].metric("Recent VO2 peak", f"{recent_peak_vo2:.0f}" if not pd.isna(recent_peak_vo2) else "")
        if latest_vo2_date:
            st.caption(f"Current VO2 max is the latest record in this export: {latest_vo2_date}.")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Zone Time")
            st.plotly_chart(add_hr_zone_chart(filtered), use_container_width=True)
        with right:
            st.subheader("Zone Definitions")
            st.dataframe(hr_zone_definitions(heart_rate_zones), hide_index=True, use_container_width=True)

        recent_zone_table = zone_time_table(recent_8, heart_rate_zones)
        st.subheader("Last 8 Weeks By Zone")
        st.dataframe(recent_zone_table, hide_index=True, use_container_width=True)

        category_rows = []
        for category, group in recent_8.groupby("activity_category"):
            zone_total = group[[f"hr_zone_{zone}_min" for zone in range(7)]].sum().sum()
            high = group[["hr_zone_4_min", "hr_zone_5_min", "hr_zone_6_min"]].sum().sum()
            category_rows.append(
                {
                    "category": category,
                    "runs": len(group),
                    "distance_km": round(float(group["distance_km"].sum()), 1),
                    "median_pace": format_pace(float(group["pace_min_per_km"].median())),
                    "avg_hr": round(weighted_avg(group["avg_hr"], group["moving_min"]), 1),
                    "high_zone_share": round(float(high / zone_total * 100), 1) if zone_total else 0,
                }
            )
        category_summary = pd.DataFrame(category_rows).sort_values("distance_km", ascending=False)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Intensity By Run Type")
            zone_category = zone_time_by_category(recent_8)
            fig_zone_cat = px.bar(
                zone_category,
                x="activity_category",
                y="minutes",
                color="zone",
                color_discrete_sequence=["#CFD8DC", "#7BAFD4", "#6BBF59", "#E6C84F", "#E58E3E", "#C94C4C", "#7A3E65"],
            )
            fig_zone_cat.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title=None,
                yaxis_title="minutes",
                legend_orientation="h",
                legend_y=1.12,
            )
            st.plotly_chart(fig_zone_cat, use_container_width=True)
        with right:
            st.subheader("Last 8 Weeks By Run Type")
            st.dataframe(category_summary, hide_index=True, use_container_width=True)

        st.subheader("Easy Run Guidance")
        st.write(
            f"Going forward, start most easy runs around **{easy_guidance['easy_range']}**. "
            f"For recovery days, use **{easy_guidance['recovery_range']}**. "
            f"The important control is effort: keep average HR below about **{easy_guidance['hr_cap']:.0f} bpm** "
            "and avoid long stretches in Z4."
        )
        st.write(
            f"This estimate uses **{easy_guidance['sample_size']}** {easy_guidance['source']} from your recent data. "
            "If heat, hills, poor sleep, or fatigue push HR upward, slow down rather than forcing the pace."
        )

        st.subheader("VO2 Max Rebuild")
        vo2_left, vo2_right = st.columns([1, 1])
        with vo2_left:
            if vo2.empty or not {"calendarDate", "vo2MaxValue"}.issubset(vo2.columns):
                fig_vo2_full = empty_figure("No VO2 max history")
            else:
                fig_vo2_full = px.line(vo2, x="calendarDate", y="vo2MaxValue", markers=False)
                if not pd.isna(recent_peak_vo2):
                    fig_vo2_full.add_hline(
                        y=recent_peak_vo2,
                        line_dash="dot",
                        line_color="#C1666B",
                        annotation_text="recent peak",
                    )
                if not pd.isna(current_vo2):
                    fig_vo2_full.add_hline(y=current_vo2, line_dash="dot", line_color="#2B7A78", annotation_text="current")
                fig_vo2_full.update_layout(
                    height=330,
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis_title=None,
                    yaxis_title="VO2 max",
                )
            st.plotly_chart(fig_vo2_full, use_container_width=True)
        with vo2_right:
            if not pd.isna(recent_peak_vo2):
                st.write(f"Recent peak VO2 max: **{recent_peak_vo2:.0f}** on **{recent_peak_vo2_date}**.")
            if not pd.isna(peak_vo2) and not pd.isna(recent_peak_vo2) and peak_vo2 > recent_peak_vo2:
                st.write(f"All-time peak in this export: **{peak_vo2:.0f}** on **{peak_vo2_date}**.")
            if pd.isna(peak_vo2):
                st.write("No VO2 max history was found in this export.")
            if not pd.isna(current_vo2) and not pd.isna(recent_peak_vo2):
                st.write(
                    f"Short-term target: rebuild from **{current_vo2:.0f}** toward "
                    f"the recent peak of **{recent_peak_vo2:.0f}**. "
                    "Treat higher VO2 goals as later-stage targets after the return-to-running block is stable."
                )
            st.write(
                "Garmin's estimate usually improves when you can run faster at the same HR, "
                "or hold the same pace at a lower HR."
            )
            st.write(
                "The priority is not more hard running. It is more sustainable aerobic volume, "
                "cleaner easy days, and one well-placed high-quality stimulus."
            )

        st.dataframe(vo2_context_table(valid_activities, vo2), hide_index=True, use_container_width=True)

        vo2_actions = pd.DataFrame(
            [
                {
                    "lever": "Aerobic base",
                    "dose": "Build toward 45-55 km/week before race-specific work",
                    "session": "Mostly easy running at the easy range above",
                    "why": "Improves pace/HR efficiency, which Garmin VO2 responds to.",
                },
                {
                    "lever": "Threshold",
                    "dose": "1 session most weeks",
                    "session": "2 x 15-20 min controlled, or 3 x 10 min",
                    "why": "Raises the speed you can hold without excessive HR drift.",
                },
                {
                    "lever": "VO2 stimulus",
                    "dose": "Every 10-14 days, not every easy week",
                    "session": "5-6 x 3 min at 5K effort, equal easy jog",
                    "why": "Direct stimulus for oxygen uptake and running economy.",
                },
                {
                    "lever": "Strides / hills",
                    "dose": "1-2 easy days per week",
                    "session": "6-8 x 15-25 sec relaxed fast",
                    "why": "Keeps mechanics sharp with low fatigue cost.",
                },
                {
                    "lever": "Long run",
                    "dose": "Most weeks, cut back every 3-4 weeks",
                    "session": "16-22 km easy; later add controlled steady finishes",
                    "why": "Supports half-marathon durability and aerobic development.",
                },
                {
                    "lever": "Recovery",
                    "dose": "Hard days separated by 48 hours where possible",
                    "session": "Easy/recovery pace, strength, sleep consistency",
                    "why": "At 54, adaptation depends on absorbing the work.",
                },
            ]
        )
        st.subheader("Training Levers To Rebuild VO2 Max")
        st.dataframe(vo2_actions, hide_index=True, use_container_width=True)

        st.info(
            "Practical rule: if an easy run cannot stay under roughly 144 bpm at 6:00-6:35/km, "
            "make it slower. Save the hard effort for threshold, VO2 intervals, and race-specific sessions."
        )

    with plan_tab:
        goal_pace = goal_minutes / HALF_MARATHON_KM
        st.subheader("Half Marathon Target")
        target_cols = st.columns(4)
        target_cols[0].metric("Target time", format_duration(goal_minutes * 60))
        target_cols[1].metric("Target pace", format_pace(goal_pace))
        target_cols[2].metric("Plan length", f"{plan_weeks} weeks")
        target_cols[3].metric("Peak week", f"{plan['target_km'].max():.0f} km")

        st.write(
            f"Default target is based on improving the latest recorded half marathon "
            f"({latest_half.get('time', '')}) toward your best recent result "
            f"({best_half.get('time', '')})."
        )

        plan_left, plan_right = st.columns([2, 1])
        with plan_left:
            fig_plan = go.Figure()
            fig_plan.add_bar(x=plan["week"], y=plan["target_km"], name="Target km", marker_color="#2B7A78")
            fig_plan.add_trace(
                go.Scatter(
                    x=plan["week"],
                    y=plan["long_run_km"],
                    mode="lines+markers",
                    name="Long run km",
                    line=dict(color="#C1666B", width=3),
                )
            )
            fig_plan.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title="week",
                yaxis_title="km",
                legend_orientation="h",
                legend_y=1.08,
            )
            st.plotly_chart(fig_plan, use_container_width=True)
        with plan_right:
            st.subheader("Pace Guide")
            st.write(f"Easy: **{plan.iloc[0]['easy_pace']}**")
            st.write(f"Steady: **{plan.iloc[0]['steady_pace']}**")
            st.write(f"Half marathon: **{plan.iloc[0]['hm_pace']}**")
            st.write("Strength: **2 short sessions most weeks**")
            st.write("Easy-day HR cap: **below Garmin Zone 4 when possible**")

        st.subheader("Weekly Plan")
        st.dataframe(
            plan[
                [
                    "week",
                    "week_start",
                    "focus",
                    "target_km",
                    "long_run_km",
                    "quality_1",
                    "quality_2",
                    "long_run",
                    "strength",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Training guidance is not medical advice. Stop or reduce load for unusual pain, illness, or persistent fatigue."
        )

    with cambridge_2027_tab:
        st.subheader("Cambridge Half Marathon 2027 Objective")
        st.write(
            "A 1:38:00 half marathon means holding "
            f"**{format_pace(cambridge_target_pace)}** for 21.1 km."
        )

        target_cols = st.columns(5)
        target_cols[0].metric("Race date used", CAMBRIDGE_2027_RACE_DATE.strftime("%d %b %Y"))
        target_cols[1].metric("Weeks available", f"{cambridge_weeks:.0f}")
        target_cols[2].metric("Target time", "1:38:00")
        target_cols[3].metric("Target pace", format_pace(cambridge_target_pace))
        target_cols[4].metric("Peak target", "55-60 km/wk")

        if not pd.isna(best_gap_pct) and best_gap_pct <= 4.5 and cambridge_weeks >= 30:
            st.success(
                "Realistic as an A-goal. The gap from your best half marathon is modest, "
                "but it needs a longer, steadier build than your current 12-week plan."
            )
        else:
            st.warning(
                "Possible, but the current data says this should be treated as a stretch goal "
                "until the volume and long-run markers improve."
            )

        gap_cols = st.columns(3)
        gap_cols[0].metric(
            "From best HM",
            best_half.get("time", ""),
            f"{best_gap_pct:.1f}% faster needed" if not pd.isna(best_gap_pct) else None,
        )
        gap_cols[1].metric(
            "From latest HM",
            latest_half.get("time", ""),
            f"{latest_gap_pct:.1f}% faster needed" if not pd.isna(latest_gap_pct) else None,
        )
        gap_cols[2].metric(
            "From Garmin prediction",
            duration_from_minutes(predicted_half_minutes),
            time_gap_text(predicted_half_minutes, CAMBRIDGE_2027_TARGET_MINUTES),
        )

        compare = pd.DataFrame(
            [
                {
                    "marker": "Latest half",
                    "minutes": latest_half_minutes,
                    "time": latest_half.get("time", ""),
                    "pace": latest_half.get("pace", ""),
                },
                {
                    "marker": "Best half",
                    "minutes": best_half_minutes,
                    "time": best_half.get("time", ""),
                    "pace": best_half.get("pace", ""),
                },
                {
                    "marker": "2027 target",
                    "minutes": CAMBRIDGE_2027_TARGET_MINUTES,
                    "time": "1:38:00",
                    "pace": format_pace(cambridge_target_pace),
                },
            ]
        ).dropna(subset=["minutes"])
        fig_compare = px.bar(
            compare,
            x="marker",
            y="minutes",
            color="marker",
            text="time",
            color_discrete_sequence=["#C1666B", "#2B7A78", "#17252A"],
        )
        fig_compare.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title=None,
            yaxis_title="minutes",
            showlegend=False,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("What Has To Change")
        readiness = pd.DataFrame(
            [
                {
                    "area": "Best half marathon",
                    "current": duration_from_minutes(best_half_minutes),
                    "target": "1:38:00",
                    "status": status_label(
                        best_half_minutes,
                        CAMBRIDGE_2027_TARGET_MINUTES,
                        higher_is_better=False,
                        tolerance=0.04,
                    ),
                    "note": "This is close enough to justify the goal.",
                },
                {
                    "area": "Latest half marathon",
                    "current": duration_from_minutes(latest_half_minutes),
                    "target": "Within 1:40 by tune-up",
                    "status": status_label(latest_half_minutes, 100, higher_is_better=False, tolerance=0.03),
                    "note": "Need to regain and then extend 2025 fitness.",
                },
                {
                    "area": "8-week average volume",
                    "current": f"{recent_weekly_km:.1f} km/wk",
                    "target": "45-55 km/wk base",
                    "status": status_label(recent_weekly_km, 45, higher_is_better=True, tolerance=0.1),
                    "note": "Build gradually; peak weeks around 55-60 km.",
                },
                {
                    "area": "Long run",
                    "current": f"{recent_longest_8:.1f} km",
                    "target": "18-22 km most weeks",
                    "status": status_label(recent_longest_8, 18, higher_is_better=True, tolerance=0.08),
                    "note": "The clearest current endurance gap.",
                },
                {
                    "area": "Run frequency",
                    "current": f"{current_run_frequency:.1f}/wk",
                    "target": "4-5/wk",
                    "status": status_label(current_run_frequency, 4.3, higher_is_better=True, tolerance=0.1),
                    "note": "Use a short easy run before adding more intensity.",
                },
                {
                    "area": "Best 10K marker",
                    "current": duration_from_minutes(best_10k_minutes),
                    "target": "44:00-44:30",
                    "status": status_label(best_10k_minutes, 44.5, higher_is_better=False, tolerance=0.03),
                    "note": "A strong predictor for 1:38 readiness.",
                },
                {
                    "area": "Best 5K marker",
                    "current": duration_from_minutes(best_5k_minutes),
                    "target": "21:30-22:00",
                    "status": status_label(best_5k_minutes, 22, higher_is_better=False, tolerance=0.03),
                    "note": "Keep speed economy alive without chasing 5K fitness.",
                },
                {
                    "area": "Easy-day discipline",
                    "current": f"{optional_percent(high_hr_share_8)} high-zone time" if optional_percent(high_hr_share_8) else "No HR-zone data",
                    "target": "<40%",
                    "status": status_label(high_hr_share_8, 40, higher_is_better=False, tolerance=0.15),
                    "note": "If Garmin zones are accurate, too many easy/base runs are too hard.",
                },
            ]
        )
        st.dataframe(readiness, hide_index=True, use_container_width=True)

        st.subheader("Long-Range Roadmap")
        road_left, road_right = st.columns([1, 1])
        with road_left:
            st.plotly_chart(build_weekly_range_chart(phase_plan_2027), use_container_width=True)
        with road_right:
            st.dataframe(phase_plan_2027, hide_index=True, use_container_width=True)

        st.subheader("Benchmark Workouts Before Race Day")
        benchmarks = pd.DataFrame(
            [
                {
                    "when": "Autumn 2026",
                    "benchmark": "10K tune-up",
                    "ready signal": "44:00-44:30 without a full taper",
                },
                {
                    "when": "Dec 2026",
                    "benchmark": "2 x 20 min threshold",
                    "ready signal": "4:35-4:45/km controlled, no fade",
                },
                {
                    "when": "Jan 2027",
                    "benchmark": "Long run",
                    "ready signal": "20-22 km easy, recovered within 48 hours",
                },
                {
                    "when": "Feb 2027",
                    "benchmark": "3 x 3 km at HM pace",
                    "ready signal": f"{format_pace(cambridge_target_pace)} with relaxed form",
                },
                {
                    "when": "Late Feb 2027",
                    "benchmark": "16-18 km specific run",
                    "ready signal": "8-10 km total at 4:39-4:45/km inside the run",
                },
            ]
        )
        st.dataframe(benchmarks, hide_index=True, use_container_width=True)

        st.info(
            "Bottom line: 1:38 is credible if you can average 45-55 km/week for several months, "
            "keep easy runs genuinely easy, and arrive in February able to run controlled HM-pace blocks. "
            "If those markers are not there by late January, 1:40-1:42 is the more defensible race target."
        )

    with data_tab:
        st.subheader("Data Freshness")
        freshness = pd.DataFrame(
            [
                {
                    "table": "Running activities",
                    "latest_record": display_date(latest_activity),
                    "latest_value": "",
                },
                {
                    "table": "VO2 max history",
                    "latest_record": latest_vo2_date,
                    "latest_value": (
                        f"{latest_vo2['vo2MaxValue']:.0f}"
                        if latest_vo2 is not None
                        and "vo2MaxValue" in latest_vo2
                        and not pd.isna(latest_vo2["vo2MaxValue"])
                        else ""
                    ),
                },
                {
                    "table": "Garmin race predictions",
                    "latest_record": latest_prediction_date,
                    "latest_value": pred,
                },
            ]
        )
        st.dataframe(freshness, hide_index=True, use_container_width=True)
        st.caption(
            "Race prediction rows are estimates from Garmin, not completed race activities. "
            "If Garmin Connect shows newer VO2 max rows than this table, upload a newer Garmin export."
        )

        st.subheader("Full Running Data")
        cols = [
            "start_time_local",
            "name",
            "activity_type",
            "activity_category",
            "distance_km",
            "moving_min",
            "pace",
            "avg_hr",
            "max_hr",
            "vo2max_value",
            "aerobic_training_effect",
            "is_valid_for_analysis",
            "outlier_reason",
            "location_name",
        ]
        existing_cols = [col for col in cols if col in activities.columns]
        shown = activities.sort_values("start_time_local", ascending=False)[existing_cols].copy()
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button(
            "Download normalized running CSV",
            data=activities.to_csv(index=False).encode("utf-8"),
            file_name="garmin_running_activities.csv",
            mime="text/csv",
            use_container_width=True,
        )
        with st.expander("Flagged runs"):
            flagged = activities[activities["is_valid_for_analysis"] == False].copy()
            st.dataframe(
                flagged[
                    [
                        "start_time_local",
                        "name",
                        "distance_km",
                        "moving_min",
                        "pace",
                        "avg_hr",
                        "outlier_reason",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
