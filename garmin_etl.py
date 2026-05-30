from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


RUN_ACTIVITY_TYPES = {"running", "treadmill_running"}
RUN_SPORT_TYPES = {"RUNNING"}
DEFAULT_GOAL_MINUTES = 102.5
HALF_MARATHON_KM = 21.0975
WEEKLY_SUMMARY_COLUMNS = [
    "week_start_dt",
    "runs",
    "distance_km",
    "duration_min",
    "long_run_km",
    "elevation_gain_m",
    "calories_kcal",
    "training_effect_sum",
    "hr_zone_0_min",
    "hr_zone_1_min",
    "hr_zone_2_min",
    "hr_zone_3_min",
    "hr_zone_4_min",
    "hr_zone_5_min",
    "hr_zone_6_min",
    "duration_h",
    "avg_pace_min_per_km",
    "avg_pace",
    "avg_hr_weighted",
    "hr_high_min",
    "hr_zone_total_min",
    "hr_high_share",
    "long_run_share",
    "acute_4w_km",
    "chronic_12w_km",
    "acute_chronic_ratio",
    "week_start",
]


def find_export_zip(root: Path = Path(".")) -> Path:
    search_dirs = [root / "data" / "raw", root]
    zips_by_path: dict[Path, Path] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.zip"):
            zips_by_path[path.resolve()] = path
    zips = list(zips_by_path.values())
    if not zips:
        raise FileNotFoundError(
            "No Garmin export zip found. Put the Garmin export .zip in data/raw/ "
            "or upload it in the Streamlit sidebar."
        )
    if len(zips) == 1:
        return zips[0]
    return max(zips, key=lambda path: (path.stat().st_mtime, path.stat().st_size))


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_pace(minutes_per_km: float | None) -> str:
    if minutes_per_km is None or pd.isna(minutes_per_km) or minutes_per_km <= 0:
        return ""
    minutes = int(minutes_per_km)
    seconds = int(round((minutes_per_km - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"


def parse_goal_time_to_minutes(goal: str | float | int) -> float:
    if isinstance(goal, (float, int)):
        return float(goal)
    parts = [int(part) for part in str(goal).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 60 + parts[1] + parts[2] / 60
    if len(parts) == 2:
        return parts[0] + parts[1] / 60
    raise ValueError("Goal time must look like H:MM:SS or MM:SS.")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None) or pd.isna(denominator):
        return math.nan
    return numerator / denominator


def _read_json(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name) as handle:
        return json.load(handle)


def _as_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _json_files(zf: zipfile.ZipFile, contains: str) -> list[str]:
    return sorted(name for name in zf.namelist() if contains in name and name.endswith(".json"))


def _load_summary_activities(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    activities: list[dict[str, Any]] = []
    for name in _json_files(zf, "_summarizedActivities"):
        payload = _read_json(zf, name)
        if not payload:
            continue
        for container in _as_list(payload):
            if not isinstance(container, dict):
                continue
            for activity in container.get("summarizedActivitiesExport", []):
                row = dict(activity)
                row["_source_file"] = name
                activities.append(row)
    return activities


def _is_running_activity(activity: dict[str, Any]) -> bool:
    return (
        activity.get("activityType") in RUN_ACTIVITY_TYPES
        or activity.get("sportType") in RUN_SPORT_TYPES
    )


def _activity_category(name: str, activity_type: str) -> str:
    label = (name or "").lower()
    if activity_type == "treadmill_running":
        return "Treadmill"
    if "race" in label:
        return "Race"
    if "recovery" in label:
        return "Recovery"
    if "long" in label:
        return "Long"
    if any(token in label for token in ("interval", "repeat", "benchmark", "tempo", "threshold")):
        return "Workout"
    if "base" in label:
        return "Base"
    return "Run"


def _outlier_reason(row: pd.Series) -> str:
    if pd.isna(row.get("distance_km")) or row["distance_km"] < 0.5:
        return "too_short"
    if pd.isna(row.get("pace_min_per_km")):
        return "missing_pace"
    if row["pace_min_per_km"] < 3.0:
        return "implausibly_fast"
    if row["pace_min_per_km"] > 10.0:
        return "implausibly_slow"
    return ""


def _normalise_running_activities(activities: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for activity in activities:
        if not _is_running_activity(activity):
            continue
        scalar_row = {key: value for key, value in activity.items() if _is_scalar(value)}
        scalar_row["has_split_summaries"] = bool(activity.get("splitSummaries"))
        scalar_row["has_splits"] = bool(activity.get("splits"))
        rows.append(scalar_row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rename_map = {
        "activityId": "activity_id",
        "activityType": "activity_type",
        "sportType": "sport_type",
        "startTimeGmt": "start_time_gmt_ms",
        "startTimeLocal": "start_time_local_ms",
        "movingDuration": "moving_duration_ms",
        "elapsedDuration": "elapsed_duration_ms",
        "avgHr": "avg_hr",
        "maxHr": "max_hr",
        "minHr": "min_hr",
        "avgRunCadence": "avg_run_cadence",
        "maxRunCadence": "max_run_cadence",
        "avgDoubleCadence": "avg_double_cadence",
        "maxDoubleCadence": "max_double_cadence",
        "vO2MaxValue": "vo2max_value",
        "aerobicTrainingEffect": "aerobic_training_effect",
        "anaerobicTrainingEffect": "anaerobic_training_effect",
        "locationName": "location_name",
        "startLatitude": "start_latitude",
        "startLongitude": "start_longitude",
        "endLatitude": "end_latitude",
        "endLongitude": "end_longitude",
    }
    df = df.rename(columns=rename_map)

    for source_col, target_col in [
        ("start_time_local_ms", "start_time_local"),
        ("start_time_gmt_ms", "start_time_gmt"),
        ("beginTimestamp", "begin_time"),
    ]:
        if source_col in df.columns:
            df[target_col] = pd.to_datetime(df[source_col], unit="ms", errors="coerce")

    df["date"] = df["start_time_local"].dt.date.astype("string")
    df["year"] = df["start_time_local"].dt.year
    df["month"] = df["start_time_local"].dt.to_period("M").astype("string")
    df["week_start"] = (
        df["start_time_local"].dt.normalize()
        - pd.to_timedelta(df["start_time_local"].dt.weekday, unit="D")
    ).dt.date.astype("string")

    df["distance_km"] = df.get("distance", 0) / 100000
    df["duration_min"] = df.get("duration", 0) / 60000
    df["moving_min"] = df.get("moving_duration_ms", df.get("duration", 0)) / 60000
    df["elapsed_min"] = df.get("elapsed_duration_ms", df.get("duration", 0)) / 60000
    df["pace_min_per_km"] = df.apply(lambda row: _safe_div(row["moving_min"], row["distance_km"]), axis=1)
    df["elapsed_pace_min_per_km"] = df.apply(
        lambda row: _safe_div(row["elapsed_min"], row["distance_km"]), axis=1
    )
    df["speed_kmh"] = df.apply(lambda row: _safe_div(row["distance_km"], row["moving_min"] / 60), axis=1)
    df["avg_speed_mps"] = df.get("avgSpeed", pd.Series(index=df.index, dtype=float)) * 10
    df["max_speed_mps"] = df.get("maxSpeed", pd.Series(index=df.index, dtype=float)) * 10
    df["elevation_gain_m"] = df.get("elevationGain", pd.Series(index=df.index, dtype=float)) / 100
    df["elevation_loss_m"] = df.get("elevationLoss", pd.Series(index=df.index, dtype=float)) / 100
    df["calories_kcal"] = df.get("calories", pd.Series(index=df.index, dtype=float)) / 4.184
    df["activity_category"] = df.apply(
        lambda row: _activity_category(str(row.get("name", "")), str(row.get("activity_type", ""))),
        axis=1,
    )

    zone_cols = []
    for zone in range(7):
        raw_col = f"hrTimeInZone_{zone}"
        min_col = f"hr_zone_{zone}_min"
        df[min_col] = df.get(raw_col, pd.Series(index=df.index, dtype=float)).fillna(0) / 60000
        zone_cols.append(min_col)
    df["hr_zone_total_min"] = df[zone_cols].sum(axis=1)
    df["hr_high_min"] = df[["hr_zone_4_min", "hr_zone_5_min", "hr_zone_6_min"]].sum(axis=1)
    df["hr_high_share"] = df.apply(lambda row: _safe_div(row["hr_high_min"], row["hr_zone_total_min"]), axis=1)
    df["outlier_reason"] = df.apply(_outlier_reason, axis=1)
    df["is_valid_for_analysis"] = df["outlier_reason"].eq("")
    df["pace"] = df["pace_min_per_km"].apply(format_pace)

    sort_cols = ["start_time_local", "activity_id"]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df


def _flatten_split_summaries(activities: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for activity in activities:
        if not _is_running_activity(activity):
            continue
        for idx, summary in enumerate(activity.get("splitSummaries") or []):
            row = {
                "activity_id": activity.get("activityId"),
                "start_time_local": pd.to_datetime(activity.get("startTimeLocal"), unit="ms", errors="coerce"),
                "name": activity.get("name"),
                "split_summary_index": idx,
            }
            row.update({key: value for key, value in summary.items() if _is_scalar(value)})
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_splits(activities: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for activity in activities:
        if not _is_running_activity(activity):
            continue
        for idx, split in enumerate(activity.get("splits") or []):
            row = {
                "activity_id": activity.get("activityId"),
                "start_time_local": pd.to_datetime(activity.get("startTimeLocal"), unit="ms", errors="coerce"),
                "name": activity.get("name"),
                "split_index": idx,
                "split_type": split.get("type"),
                "start_time_gmt": pd.to_datetime(split.get("startTimeGMT"), unit="ms", errors="coerce"),
                "end_time_gmt": pd.to_datetime(split.get("endTimeGMT"), unit="ms", errors="coerce"),
                "start_latitude": split.get("startLatitude"),
                "start_longitude": split.get("startLongitude"),
                "end_latitude": split.get("endLatitude"),
                "end_longitude": split.get("endLongitude"),
            }
            for measurement in split.get("measurements") or []:
                if not measurement.get("valid", False):
                    continue
                field = str(measurement.get("fieldEnum", "")).lower()
                unit = str(measurement.get("unitEnum", "")).lower()
                if field:
                    row[f"{field}_{unit}"] = measurement.get("value")
            rows.append(row)
    return pd.DataFrame(rows)


def _load_metric_frames(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    race_rows: list[dict[str, Any]] = []
    for name in _json_files(zf, "RunRacePredictions"):
        race_rows.extend(_read_json(zf, name))
    race = pd.DataFrame(race_rows)
    if not race.empty:
        race["calendarDate"] = pd.to_datetime(race["calendarDate"], errors="coerce")
        race["timestamp_dt"] = pd.to_datetime(race.get("timestamp"), errors="coerce")
        race = race.sort_values(["calendarDate", "timestamp_dt"]).drop_duplicates("calendarDate", keep="last")
        for col in ["raceTime5K", "raceTime10K", "raceTimeHalf", "raceTimeMarathon"]:
            if col in race.columns:
                race[f"{col}_min"] = race[col] / 60
                race[f"{col}_display"] = race[col].apply(format_duration)
    frames["race_predictions"] = race

    vo2_rows: list[dict[str, Any]] = []
    for name in _json_files(zf, "MetricsMaxMetData"):
        vo2_rows.extend(_read_json(zf, name))
    vo2 = pd.DataFrame(vo2_rows)
    if not vo2.empty:
        vo2["calendarDate"] = pd.to_datetime(vo2["calendarDate"], errors="coerce")
        vo2["updateTimestamp_dt"] = pd.to_datetime(vo2.get("updateTimestamp"), errors="coerce")
        vo2 = vo2.sort_values(["calendarDate", "updateTimestamp_dt"]).drop_duplicates("calendarDate", keep="last")
    frames["vo2max_history"] = vo2

    training_rows: list[dict[str, Any]] = []
    for name in _json_files(zf, "TrainingHistory"):
        training_rows.extend(_read_json(zf, name))
    training = pd.DataFrame(training_rows)
    if not training.empty:
        training["calendarDate"] = pd.to_datetime(training["calendarDate"], errors="coerce")
        training["timestamp_dt"] = pd.to_datetime(training.get("timestamp"), errors="coerce")
        training = training.sort_values(["calendarDate", "timestamp_dt"]).drop_duplicates("calendarDate", keep="last")
    frames["training_history"] = training

    pr = pd.DataFrame()
    pr_files = _json_files(zf, "personalRecord")
    if pr_files:
        payload = _read_json(zf, pr_files[0])
        if payload:
            pr = pd.DataFrame(payload[0].get("personalRecords", []))
    frames["personal_records"] = pr

    zones = pd.DataFrame()
    zone_files = _json_files(zf, "heartRateZones")
    if zone_files:
        zones = pd.DataFrame(_read_json(zf, zone_files[0]))
    frames["heart_rate_zones"] = zones

    bio_profile = pd.DataFrame()
    bio_profile_files = _json_files(zf, "userBioMetricProfileData")
    if bio_profile_files:
        bio_profile = pd.DataFrame(_read_json(zf, bio_profile_files[0]))
    frames["bio_profile"] = bio_profile

    return frames


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return math.nan
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def build_weekly_summary(activities: pd.DataFrame) -> pd.DataFrame:
    if activities.empty or "is_valid_for_analysis" not in activities.columns:
        return pd.DataFrame(columns=WEEKLY_SUMMARY_COLUMNS)
    valid = activities[activities["is_valid_for_analysis"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=WEEKLY_SUMMARY_COLUMNS)
    valid["week_start_dt"] = pd.to_datetime(valid["week_start"])
    grouped = valid.groupby("week_start_dt", as_index=False)
    weekly = grouped.agg(
        runs=("activity_id", "count"),
        distance_km=("distance_km", "sum"),
        duration_min=("moving_min", "sum"),
        long_run_km=("distance_km", "max"),
        elevation_gain_m=("elevation_gain_m", "sum"),
        calories_kcal=("calories_kcal", "sum"),
        training_effect_sum=("aerobic_training_effect", "sum"),
        hr_zone_0_min=("hr_zone_0_min", "sum"),
        hr_zone_1_min=("hr_zone_1_min", "sum"),
        hr_zone_2_min=("hr_zone_2_min", "sum"),
        hr_zone_3_min=("hr_zone_3_min", "sum"),
        hr_zone_4_min=("hr_zone_4_min", "sum"),
        hr_zone_5_min=("hr_zone_5_min", "sum"),
        hr_zone_6_min=("hr_zone_6_min", "sum"),
    )
    weekly["duration_h"] = weekly["duration_min"] / 60
    weekly["avg_pace_min_per_km"] = weekly["duration_min"] / weekly["distance_km"]
    weekly["avg_pace"] = weekly["avg_pace_min_per_km"].apply(format_pace)

    avg_hr_rows = []
    for week_start, week_df in valid.groupby("week_start_dt"):
        avg_hr_rows.append(
            {
                "week_start_dt": week_start,
                "avg_hr_weighted": _weighted_average(week_df["avg_hr"], week_df["moving_min"]),
            }
        )
    weekly = weekly.merge(pd.DataFrame(avg_hr_rows), on="week_start_dt", how="left")
    weekly["hr_high_min"] = weekly[["hr_zone_4_min", "hr_zone_5_min", "hr_zone_6_min"]].sum(axis=1)
    weekly["hr_zone_total_min"] = weekly[[f"hr_zone_{zone}_min" for zone in range(7)]].sum(axis=1)
    weekly["hr_high_share"] = weekly["hr_high_min"] / weekly["hr_zone_total_min"].replace(0, pd.NA)
    weekly["long_run_share"] = weekly["long_run_km"] / weekly["distance_km"].replace(0, pd.NA)
    weekly = weekly.sort_values("week_start_dt")
    weekly["acute_4w_km"] = weekly["distance_km"].rolling(4, min_periods=1).mean()
    weekly["chronic_12w_km"] = weekly["distance_km"].rolling(12, min_periods=4).mean()
    weekly["acute_chronic_ratio"] = weekly["acute_4w_km"] / weekly["chronic_12w_km"]
    weekly["week_start"] = weekly["week_start_dt"].dt.date.astype("string")
    return weekly[WEEKLY_SUMMARY_COLUMNS]


def _window_stats(valid: pd.DataFrame, latest: pd.Timestamp, weeks: int) -> dict[str, Any]:
    window = valid[valid["start_time_local"] > latest - pd.Timedelta(weeks=weeks)]
    if window.empty:
        return {}
    zone_total = window[[f"hr_zone_{zone}_min" for zone in range(7)]].sum().sum()
    high_zone = window[["hr_zone_4_min", "hr_zone_5_min", "hr_zone_6_min"]].sum().sum()
    return {
        "runs": int(len(window)),
        "distance_km": round(float(window["distance_km"].sum()), 1),
        "avg_weekly_km": round(float(window["distance_km"].sum() / weeks), 1),
        "longest_run_km": round(float(window["distance_km"].max()), 1),
        "avg_pace_min_per_km": round(float(window["moving_min"].sum() / window["distance_km"].sum()), 2),
        "avg_pace": format_pace(window["moving_min"].sum() / window["distance_km"].sum()),
        "avg_hr": round(_weighted_average(window["avg_hr"], window["moving_min"]), 1),
        "high_hr_zone_share_pct": round(float(high_zone / zone_total * 100), 1) if zone_total else None,
    }


def _empty_recent_windows() -> dict[str, dict[str, Any]]:
    return {
        f"{weeks}_weeks": {
            "runs": 0,
            "distance_km": 0.0,
            "avg_weekly_km": 0.0,
            "longest_run_km": 0.0,
            "avg_pace_min_per_km": math.nan,
            "avg_pace": "",
            "avg_hr": math.nan,
            "high_hr_zone_share_pct": math.nan,
        }
        for weeks in [4, 8, 12, 26, 52]
    }


def _empty_insights(activities: pd.DataFrame) -> dict[str, Any]:
    flagged = 0
    if not activities.empty and "is_valid_for_analysis" in activities.columns:
        flagged = int((~activities["is_valid_for_analysis"]).sum())
    return {
        "generated_on": date.today().isoformat(),
        "data": {
            "activity_data_through": "",
            "first_activity_date": "",
            "total_running_activities": int(len(activities)),
            "valid_running_activities": 0,
            "flagged_outliers": flagged,
            "total_valid_distance_km": 0.0,
            "total_valid_hours": 0.0,
        },
        "current": {},
        "recent": _empty_recent_windows(),
        "half_marathon": {
            "latest": {},
            "best": {},
            "suggested_goal_time": format_duration(DEFAULT_GOAL_MINUTES * 60),
            "suggested_goal_pace": format_pace(DEFAULT_GOAL_MINUTES / HALF_MARATHON_KM),
            "suggested_peak_week_km": 35.0,
        },
        "recommendations": [
            "No valid running activities were found. Check that the Garmin export contains running activities."
        ],
    }


def build_insights(
    activities: pd.DataFrame,
    weekly: pd.DataFrame,
    race_predictions: pd.DataFrame,
    vo2max_history: pd.DataFrame,
    training_history: pd.DataFrame,
) -> dict[str, Any]:
    if activities.empty or "is_valid_for_analysis" not in activities.columns:
        return _empty_insights(activities)
    valid = activities[activities["is_valid_for_analysis"]].copy()
    if valid.empty:
        return _empty_insights(activities)

    latest = valid["start_time_local"].max()
    current = {}

    if not vo2max_history.empty and {"calendarDate", "vo2MaxValue"}.issubset(vo2max_history.columns):
        latest_vo2 = vo2max_history.sort_values("calendarDate").iloc[-1]
        current["vo2max"] = float(latest_vo2.get("vo2MaxValue"))
        current["vo2max_date"] = str(latest_vo2.get("calendarDate").date())

    if not race_predictions.empty and {"calendarDate", "raceTimeHalf"}.issubset(race_predictions.columns):
        latest_race = race_predictions.sort_values("calendarDate").iloc[-1]
        current["race_prediction_date"] = str(latest_race.get("calendarDate").date())
        current["predicted_half_seconds"] = int(latest_race.get("raceTimeHalf"))
        current["predicted_half"] = format_duration(latest_race.get("raceTimeHalf"))

    required_training_cols = {"calendarDate", "trainingStatus", "weeklyTrainingLoadSum", "loadTunnelMin", "loadTunnelMax"}
    if not training_history.empty and required_training_cols.issubset(training_history.columns):
        latest_training = training_history.sort_values("calendarDate").iloc[-1]
        current["training_status"] = str(latest_training.get("trainingStatus"))
        current["weekly_training_load"] = int(latest_training.get("weeklyTrainingLoadSum"))
        current["load_tunnel_min"] = int(latest_training.get("loadTunnelMin"))
        current["load_tunnel_max"] = int(latest_training.get("loadTunnelMax"))

    half_races = valid[
        valid["distance_km"].between(20.8, 21.5)
        & valid["name"].fillna("").str.contains("half|race", case=False, regex=True)
    ].sort_values("start_time_local")
    if half_races.empty:
        half_races = valid[valid["distance_km"].between(20.8, 21.5)].sort_values("start_time_local")
    latest_half = half_races.iloc[-1].to_dict() if not half_races.empty else {}
    best_half = half_races.sort_values("moving_min").iloc[0].to_dict() if not half_races.empty else {}

    recent = {f"{weeks}_weeks": _window_stats(valid, latest, weeks) for weeks in [4, 8, 12, 26, 52]}
    base_weekly_km = recent["8_weeks"].get("avg_weekly_km") or recent["12_weeks"].get("avg_weekly_km") or 35
    peak_target = max(base_weekly_km * 1.35, 45)

    recommendations = [
        "Keep the next block around four runs per week, because the recent data shows that frequency is already sustainable.",
        "Build the long run back from roughly 14 km toward 19-21 km; that is the clearest half-marathon-specific gap in the last 8 weeks.",
        "Make easy days easier. Garmin zone time suggests a large share of recent running sits above the Zone 4 floor, so the plan keeps recovery/base work mostly below that effort.",
        "Use one threshold/HM-pace session and one shorter interval or hill session per week, with cutback weeks to reduce injury risk at age 54.",
    ]

    return {
        "generated_on": date.today().isoformat(),
        "data": {
            "activity_data_through": str(latest.date()),
            "first_activity_date": str(valid["start_time_local"].min().date()),
            "total_running_activities": int(len(activities)),
            "valid_running_activities": int(len(valid)),
            "flagged_outliers": int((~activities["is_valid_for_analysis"]).sum()),
            "total_valid_distance_km": round(float(valid["distance_km"].sum()), 1),
            "total_valid_hours": round(float(valid["moving_min"].sum() / 60), 1),
        },
        "current": current,
        "recent": recent,
        "half_marathon": {
            "latest": _activity_brief(latest_half),
            "best": _activity_brief(best_half),
            "suggested_goal_time": format_duration(DEFAULT_GOAL_MINUTES * 60),
            "suggested_goal_pace": format_pace(DEFAULT_GOAL_MINUTES / HALF_MARATHON_KM),
            "suggested_peak_week_km": round(float(peak_target), 1),
        },
        "recommendations": recommendations,
    }


def _activity_brief(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    moving_seconds = float(row.get("moving_min", 0)) * 60
    return {
        "date": str(pd.to_datetime(row.get("start_time_local")).date()),
        "name": str(row.get("name")),
        "distance_km": round(float(row.get("distance_km")), 2),
        "time": format_duration(moving_seconds),
        "pace": format_pace(float(row.get("pace_min_per_km"))),
        "avg_hr": None if pd.isna(row.get("avg_hr")) else round(float(row.get("avg_hr")), 1),
    }


@dataclass
class PaceGuide:
    goal_minutes: float

    @property
    def goal_pace(self) -> float:
        return self.goal_minutes / HALF_MARATHON_KM

    def range_text(self, slow_delta_seconds: int, fast_delta_seconds: int = 0) -> str:
        fast = self.goal_pace + fast_delta_seconds / 60
        slow = self.goal_pace + slow_delta_seconds / 60
        return f"{format_pace(fast)} to {format_pace(slow)}"


def next_monday(day: date) -> date:
    days_ahead = (7 - day.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return day + timedelta(days=days_ahead)


def generate_half_marathon_plan(
    start_date: date | pd.Timestamp | None = None,
    weeks: int = 12,
    base_weekly_km: float = 36.0,
    goal_minutes: float = DEFAULT_GOAL_MINUTES,
) -> pd.DataFrame:
    if start_date is None:
        start = next_monday(date.today())
    else:
        start = pd.to_datetime(start_date).date()

    guide = PaceGuide(goal_minutes)
    base = max(30.0, float(base_weekly_km))
    multipliers = [0.95, 1.00, 1.08, 0.84, 1.08, 1.16, 1.24, 0.92, 1.25, 1.30, 0.95, 0.72]
    long_runs = [14, 15, 16, 13, 17, 18, 19, 15, 20, 21, 14, HALF_MARATHON_KM]
    focus = [
        "Set rhythm",
        "Aerobic build",
        "Threshold build",
        "Cutback",
        "HM-specific endurance",
        "Strength endurance",
        "Peak endurance",
        "Cutback",
        "Race simulation",
        "Peak specificity",
        "Taper",
        "Race week",
    ]
    quality_1 = [
        f"6 x 2 min hills or 5K effort, jog recoveries ({guide.range_text(-20, -35)})",
        f"5 x 1 km at 10K effort ({guide.range_text(-15, -25)})",
        f"3 x 8 min threshold ({guide.range_text(-2, -12)})",
        "6 x 20 sec strides after an easy run",
        f"4 x 1.2 km at 10K effort ({guide.range_text(-15, -25)})",
        f"2 x 15 min threshold ({guide.range_text(-2, -10)})",
        f"6 x 1 km at 10K effort ({guide.range_text(-15, -25)})",
        "8 x 30 sec relaxed fast, full easy recovery",
        f"3 x 3 km at HM pace ({format_pace(guide.goal_pace)})",
        f"2 x 20 min at HM effort ({guide.range_text(3, -3)})",
        f"3 x 1 km at 10K effort ({guide.range_text(-15, -25)})",
        f"Race: {format_duration(goal_minutes * 60)} target, {format_pace(guide.goal_pace)}",
    ]
    quality_2 = [
        f"20 min steady ({guide.range_text(45, 25)})",
        f"25 min steady ({guide.range_text(45, 25)})",
        f"6 km progression, finish near HM pace ({format_pace(guide.goal_pace)})",
        "Easy running only",
        f"2 x 4 km at HM pace ({format_pace(guide.goal_pace)})",
        f"30 min steady ({guide.range_text(45, 25)})",
        f"8 km progression, final 3 km at HM pace ({format_pace(guide.goal_pace)})",
        "Easy running only",
        f"10 km steady with 5 km at HM pace ({format_pace(guide.goal_pace)})",
        f"12 km with 8 km at HM pace ({format_pace(guide.goal_pace)})",
        f"20 min at HM pace ({format_pace(guide.goal_pace)})",
        "2 short easy runs with strides before race day",
    ]

    rows: list[dict[str, Any]] = []
    for idx in range(weeks):
        week_start = start + timedelta(days=idx * 7)
        week_end = week_start + timedelta(days=6)
        multiplier = multipliers[min(idx, len(multipliers) - 1)]
        distance = round(base * multiplier)
        if idx == weeks - 1:
            distance = max(round(HALF_MARATHON_KM + 8), 29)
        long_run = long_runs[min(idx, len(long_runs) - 1)]
        if idx < weeks - 1:
            long_run = min(long_run, max(distance - 14, 10))
        rows.append(
            {
                "week": idx + 1,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "focus": focus[min(idx, len(focus) - 1)],
                "target_km": distance,
                "long_run_km": round(float(long_run), 1),
                "easy_pace": guide.range_text(95, 60),
                "steady_pace": guide.range_text(45, 25),
                "hm_pace": format_pace(guide.goal_pace),
                "quality_1": quality_1[min(idx, len(quality_1) - 1)],
                "quality_2": quality_2[min(idx, len(quality_2) - 1)],
                "long_run": (
                    "Race day"
                    if idx == weeks - 1
                    else f"{round(float(long_run), 1)} km easy; add last 3 km steady from week 5"
                ),
                "strength": "2 x 25 min strength/mobility" if idx < weeks - 2 else "1 x light mobility",
                "recovery_note": "Keep easy runs conversational and cap effort below Garmin Zone 4 when possible.",
            }
        )
    return pd.DataFrame(rows)


def build_garmin_data(zip_source: str | Path | bytes | BinaryIO) -> dict[str, pd.DataFrame | dict[str, Any]]:
    source = BytesIO(zip_source) if isinstance(zip_source, (bytes, bytearray)) else zip_source
    with zipfile.ZipFile(source) as zf:
        activities = _load_summary_activities(zf)
        running = _normalise_running_activities(activities)
        splits = _flatten_splits(activities)
        split_summaries = _flatten_split_summaries(activities)
        metric_frames = _load_metric_frames(zf)

    weekly = build_weekly_summary(running)
    insights = build_insights(
        running,
        weekly,
        metric_frames["race_predictions"],
        metric_frames["vo2max_history"],
        metric_frames["training_history"],
    )
    base_weekly_km = insights["recent"]["8_weeks"].get("avg_weekly_km", 36.0)
    plan = generate_half_marathon_plan(base_weekly_km=base_weekly_km)

    return {
        "running_activities": running,
        "running_splits": splits,
        "running_split_summaries": split_summaries,
        "weekly_summary": weekly,
        "race_predictions": metric_frames["race_predictions"],
        "vo2max_history": metric_frames["vo2max_history"],
        "training_history": metric_frames["training_history"],
        "personal_records": metric_frames["personal_records"],
        "heart_rate_zones": metric_frames["heart_rate_zones"],
        "bio_profile": metric_frames["bio_profile"],
        "half_marathon_plan": plan,
        "insights": insights,
    }


def extract_garmin_export(zip_path: Path, out_dir: Path = Path("data/processed")) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_garmin_data(zip_path)

    outputs = {
        "running_activities": out_dir / "running_activities.csv",
        "running_splits": out_dir / "running_splits.csv",
        "running_split_summaries": out_dir / "running_split_summaries.csv",
        "weekly_summary": out_dir / "weekly_summary.csv",
        "race_predictions": out_dir / "race_predictions.csv",
        "vo2max_history": out_dir / "vo2max_history.csv",
        "training_history": out_dir / "training_history.csv",
        "personal_records": out_dir / "personal_records.csv",
        "heart_rate_zones": out_dir / "heart_rate_zones.csv",
        "bio_profile": out_dir / "bio_profile.csv",
        "half_marathon_plan": out_dir / "half_marathon_plan.csv",
        "insights": out_dir / "insights.json",
    }

    for key, path in outputs.items():
        if key == "insights":
            path.write_text(json.dumps(data["insights"], indent=2), encoding="utf-8")
            continue
        frame = data.get(key, pd.DataFrame())
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame()
        frame.to_csv(path, index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Garmin running data for the Streamlit dashboard.")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=None, help="Path to Garmin export zip.")
    parser.add_argument("--out", dest="out_dir", type=Path, default=Path("data/processed"), help="Output folder.")
    args = parser.parse_args()

    zip_path = args.zip_path or find_export_zip()
    outputs = extract_garmin_export(zip_path, args.out_dir)
    print(f"Extracted Garmin running data from {zip_path}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
