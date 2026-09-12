from datetime import date

import pandas as pd
import pytest

from demo_data import build_demo_export
from garmin_etl import (
    build_garmin_data,
    format_duration,
    format_pace,
    generate_half_marathon_plan,
    parse_goal_time_to_minutes,
)


def test_demo_export_builds_complete_privacy_safe_dataset():
    data = build_garmin_data(build_demo_export())
    activities = data["running_activities"]

    assert len(activities) == 156
    assert activities["activity_id"].is_unique
    assert activities["is_valid_for_analysis"].all()
    assert activities["name"].str.startswith("Demo ").all()
    assert not any("latitude" in column for column in activities.columns)
    assert not data["weekly_summary"].empty
    assert not data["vo2max_history"].empty
    assert data["insights"]["data"]["valid_running_activities"] == 156
    assert data["insights"]["half_marathon"]["latest"]["time"] != ""


def test_duration_pace_and_goal_formatting():
    assert format_duration(3723) == "1:02:03"
    assert format_pace(4.65) == "4:39/km"
    assert parse_goal_time_to_minutes("1:38:00") == pytest.approx(98.0)
    assert parse_goal_time_to_minutes("48:30") == pytest.approx(48.5)


def test_invalid_goal_format_is_rejected():
    with pytest.raises(ValueError):
        parse_goal_time_to_minutes("98")


def test_half_marathon_plan_is_contiguous_and_ends_on_race_week():
    plan = generate_half_marathon_plan(
        start_date=date(2026, 9, 14), weeks=10, base_weekly_km=40, goal_minutes=98
    )

    starts = pd.to_datetime(plan["week_start"])
    assert len(plan) == 10
    assert starts.diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert plan.iloc[-1]["focus"] == "Race week"
    assert plan.iloc[-1]["long_run"] == "Race day"


def test_plan_rejects_unsupported_length():
    with pytest.raises(ValueError, match="between 1 and 12"):
        generate_half_marathon_plan(weeks=13)
