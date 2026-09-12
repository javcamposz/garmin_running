"""Deterministic synthetic Garmin export for demos and tests."""

from __future__ import annotations

import json
import random
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO


def build_demo_export() -> bytes:
    """Return a Garmin-shaped zip containing synthetic running history."""

    rng = random.Random(2026)
    start = datetime(2025, 9, 1, 7, 0, tzinfo=timezone.utc)
    activities: list[dict[str, object]] = []

    for week in range(52):
        for run_index, day_offset in enumerate((0, 2, 5)):
            when = start + timedelta(weeks=week, days=day_offset)
            if run_index == 0:
                name = "Demo Base Run"
                distance_km = 6.0 + (week % 3)
                pace = 6.05 - week * 0.006 + rng.uniform(-0.08, 0.08)
                avg_hr = 133 + rng.randint(-3, 3)
                category_zones = (0.04, 0.18, 0.48, 0.22, 0.07, 0.01, 0.0)
            elif run_index == 1:
                name = "Demo Threshold Session"
                distance_km = 7.0 + (week % 2)
                pace = 5.45 - week * 0.005 + rng.uniform(-0.07, 0.07)
                avg_hr = 151 + rng.randint(-3, 3)
                category_zones = (0.02, 0.08, 0.20, 0.28, 0.30, 0.11, 0.01)
            else:
                name = "Demo Long Run"
                distance_km = 12.0 + min(7.0, week * 0.14) + (week % 4) * 0.3
                pace = 6.18 - week * 0.004 + rng.uniform(-0.06, 0.06)
                avg_hr = 139 + rng.randint(-3, 3)
                category_zones = (0.03, 0.13, 0.42, 0.28, 0.12, 0.02, 0.0)

                if week in {43, 51}:
                    name = "Demo Half Marathon"
                    distance_km = 21.1
                    pace = 5.72 - (week - 43) * 0.012 + rng.uniform(-0.03, 0.03)
                    avg_hr = 157 + rng.randint(-2, 2)
                    category_zones = (0.01, 0.04, 0.14, 0.29, 0.39, 0.12, 0.01)

            moving_minutes = distance_km * pace
            duration_ms = int(moving_minutes * 60_000)
            zone_values = {
                f"hrTimeInZone_{zone}": int(duration_ms * share)
                for zone, share in enumerate(category_zones)
            }
            timestamp_ms = int(when.timestamp() * 1000)
            activities.append(
                {
                    "activityId": 900_000 + len(activities),
                    "activityType": "running",
                    "sportType": "RUNNING",
                    "name": name,
                    "startTimeGmt": timestamp_ms,
                    "startTimeLocal": timestamp_ms,
                    "duration": duration_ms,
                    "movingDuration": duration_ms,
                    "elapsedDuration": duration_ms + 120_000,
                    "distance": int(distance_km * 100_000),
                    "avgHr": avg_hr,
                    "maxHr": avg_hr + 18,
                    "elevationGain": int((35 + distance_km * 4) * 100),
                    "calories": int(distance_km * 62 * 4.184),
                    "aerobicTrainingEffect": round(2.5 + run_index * 0.6, 1),
                    **zone_values,
                }
            )

    metric_dates = [start + timedelta(weeks=week) for week in range(0, 52, 4)]
    race_predictions = [
        {
            "calendarDate": when.date().isoformat(),
            "timestamp": when.isoformat(),
            "raceTime5K": int((27.0 - index * 0.12) * 60),
            "raceTime10K": int((56.5 - index * 0.22) * 60),
            "raceTimeHalf": int((124.0 - index * 0.55) * 60),
            "raceTimeMarathon": int((270.0 - index * 1.0) * 60),
        }
        for index, when in enumerate(metric_dates)
    ]
    vo2_history = [
        {
            "calendarDate": when.date().isoformat(),
            "updateTimestamp": when.isoformat(),
            "vo2MaxValue": round(45.0 + index * 0.35, 1),
        }
        for index, when in enumerate(metric_dates)
    ]
    training_history = [
        {
            "calendarDate": when.date().isoformat(),
            "timestamp": when.isoformat(),
            "trainingStatus": "PRODUCTIVE",
            "weeklyTrainingLoadSum": 420 + index * 9,
            "loadTunnelMin": 360 + index * 6,
            "loadTunnelMax": 590 + index * 8,
        }
        for index, when in enumerate(metric_dates)
    ]
    heart_rate_zones = [
        {
            "zone1Floor": 92,
            "zone2Floor": 110,
            "zone3Floor": 128,
            "zone4Floor": 145,
            "zone5Floor": 163,
            "maxHeartRateUsed": 181,
        }
    ]

    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        payloads = {
            "DI_CONNECT/DI-Connect-Fitness/_summarizedActivities.json": [
                {"summarizedActivitiesExport": activities}
            ],
            "DI_CONNECT/DI-Connect-Metrics/RunRacePredictions.json": race_predictions,
            "DI_CONNECT/DI-Connect-Metrics/MetricsMaxMetData.json": vo2_history,
            "DI_CONNECT/DI-Connect-Metrics/TrainingHistory.json": training_history,
            "DI_CONNECT/DI-Connect-User/heartRateZones.json": heart_rate_zones,
        }
        for name, payload in payloads.items():
            archive.writestr(name, json.dumps(payload))
    return output.getvalue()
