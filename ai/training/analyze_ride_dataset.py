from pathlib import Path
import json

import pandas as pd
import numpy as np


# ==========================================================
# RIDEGUARDIAN - RIDE DATASET QUALITY ANALYZER
# ==========================================================

BASE_DIR = Path(r"C:\RIDER_SYSTEM\RideGuardian")

INPUT_FILE = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "processed"
    / "ride_readings_processed_with_split.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN RIDE DATASET QUALITY ANALYSIS")
print("=" * 70)

print()
print("Input:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================================
# TIMESTAMP
# ==========================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)


# ==========================================================
# NUMERIC COLUMNS
# ==========================================================

numeric_columns = [
    "speedKmh",
    "accelerationX",
    "accelerationY",
    "accelerationZ",
    "gyroX",
    "gyroY",
    "gyroZ",
    "timeSincePreviousReadingSec",
    "speedChangeKmh",
    "speedAccelerationMs2",
    "gyroMagnitudeRadS",
    "accelerationMagnitude",
    "gyroMagnitudeCalculatedRadS",
    "speedMps",
    "latitude",
    "longitude",
]


for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================================
# RIDE-LEVEL DATA
# ==========================================================

ride_columns = [
    "rideId",
    "rideStartTime",
    "rideEndTime",
    "rideDistanceKm",
    "rideDurationSec",
    "rideAverageSpeedKmh",
    "rideSensorReadingCount",
    "rideSafetyScore",
    "rideSafetyLevel",
    "rideRiskPercentage",
    "rideSpeedEvents",
    "rideHardAccelerationEvents",
    "rideHardBrakingEvents",
    "rideSuddenTurnEvents",
    "rideTotalEvents",
    "rideMaximumSpeedKmh",
]


ride_df = (
    df[ride_columns]
    .drop_duplicates(
        subset=["rideId"]
    )
    .sort_values("rideId")
    .reset_index(drop=True)
)


# ==========================================================
# BASIC DATASET SUMMARY
# ==========================================================

ride_count = df["rideId"].nunique()

reading_count = len(df)

print()
print("=" * 70)
print("BASIC DATASET")
print("=" * 70)

print()
print("Unique rides:", ride_count)
print("Sensor readings:", reading_count)

if ride_count > 0:
    print(
        "Average readings per ride:",
        round(
            reading_count / ride_count,
            2
        )
    )


# ==========================================================
# RIDE TABLE
# ==========================================================

print()
print("=" * 70)
print("RIDE SUMMARY")
print("=" * 70)

display_columns = [
    "rideId",
    "rideDistanceKm",
    "rideDurationSec",
    "rideAverageSpeedKmh",
    "rideMaximumSpeedKmh",
    "rideSafetyScore",
    "rideSafetyLevel",
    "rideRiskPercentage",
    "rideSpeedEvents",
    "rideHardAccelerationEvents",
    "rideHardBrakingEvents",
    "rideSuddenTurnEvents",
    "rideTotalEvents",
    "rideSensorReadingCount",
]

print()

print(
    ride_df[display_columns].to_string(
        index=False
    )
)


# ==========================================================
# SAFETY DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("SAFETY DISTRIBUTION")
print("=" * 70)

print()

if "rideSafetyLevel" in ride_df.columns:

    print(
        ride_df[
            "rideSafetyLevel"
        ]
        .value_counts()
        .to_string()
    )


# ==========================================================
# SAFETY SCORE DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("SAFETY SCORE")
print("=" * 70)

print()

print(
    ride_df[
        "rideSafetyScore"
    ].describe().to_string()
)


# ==========================================================
# SPEED ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("SPEED ANALYSIS")
print("=" * 70)

print()

speed = df["speedKmh"].dropna()

if len(speed):

    print(
        f"Minimum speed:      {speed.min():.2f} km/h"
    )

    print(
        f"Average speed:      {speed.mean():.2f} km/h"
    )

    print(
        f"Median speed:       {speed.median():.2f} km/h"
    )

    print(
        f"Maximum speed:      {speed.max():.2f} km/h"
    )

    print(
        f"Speed std:          {speed.std():.2f} km/h"
    )

    print()

    print(
        "Speed percentiles:"
    )

    print(
        speed.quantile(
            [
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )


# ==========================================================
# ACCELERATION ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("ACCELERATION ANALYSIS")
print("=" * 70)

print()

acc = df[
    "speedAccelerationMs2"
].dropna()

if len(acc):

    print(
        f"Minimum:       {acc.min():.3f} m/s²"
    )

    print(
        f"Average:       {acc.mean():.3f} m/s²"
    )

    print(
        f"Maximum:       {acc.max():.3f} m/s²"
    )

    print(
        f"Std:           {acc.std():.3f} m/s²"
    )

    print()

    print(
        "Strong acceleration:"
    )

    print(
        ">= +2.5 m/s²:",
        int(
            (acc >= 2.5).sum()
        )
    )

    print(
        ">= +4.0 m/s²:",
        int(
            (acc >= 4.0).sum()
        )
    )

    print()

    print(
        "Strong braking:"
    )

    print(
        "<= -2.5 m/s²:",
        int(
            (acc <= -2.5).sum()
        )
    )

    print(
        "<= -4.0 m/s²:",
        int(
            (acc <= -4.0).sum()
        )
    )


# ==========================================================
# GYROSCOPE ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("GYROSCOPE ANALYSIS")
print("=" * 70)

print()

gyro = df[
    "gyroMagnitudeRadS"
].dropna()

if len(gyro):

    print(
        f"Minimum:       {gyro.min():.3f} rad/s"
    )

    print(
        f"Average:       {gyro.mean():.3f} rad/s"
    )

    print(
        f"Maximum:       {gyro.max():.3f} rad/s"
    )

    print(
        f"Std:           {gyro.std():.3f} rad/s"
    )

    print()

    print(
        "Turn thresholds:"
    )

    print(
        ">= 1.2 rad/s:",
        int(
            (gyro >= 1.2).sum()
        )
    )

    print(
        ">= 2.0 rad/s:",
        int(
            (gyro >= 2.0).sum()
        )
    )


# ==========================================================
# GPS ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("GPS ANALYSIS")
print("=" * 70)

print()

gps_missing = (
    df[
        [
            "latitude",
            "longitude",
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)

print(
    "Rows with missing GPS:",
    int(gps_missing)
)

print(
    "GPS completeness:",
    f"{(1 - gps_missing / len(df)) * 100:.2f}%"
)


# ==========================================================
# SAMPLING INTERVAL
# ==========================================================

print()
print("=" * 70)
print("SAMPLING INTERVAL")
print("=" * 70)

interval = df[
    "timeSincePreviousReadingSec"
].dropna()

if len(interval):

    print(
        f"Minimum:       {interval.min():.3f} sec"
    )

    print(
        f"Average:       {interval.mean():.3f} sec"
    )

    print(
        f"Median:        {interval.median():.3f} sec"
    )

    print(
        f"Maximum:       {interval.max():.3f} sec"
    )

    print()

    print(
        "Interval distribution:"
    )

    print(
        interval.describe().to_string()
    )


# ==========================================================
# EVENT ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("EVENT ANALYSIS")
print("=" * 70)

event_columns = [
    "rideSpeedEvents",
    "rideHardAccelerationEvents",
    "rideHardBrakingEvents",
    "rideSuddenTurnEvents",
    "rideTotalEvents",
]

for column in event_columns:

    if column in ride_df.columns:

        print(
            f"{column}: "
            f"{ride_df[column].sum()}"
        )


# ==========================================================
# READING-LEVEL EVENT ANALYSIS
# ==========================================================

print()
print("=" * 70)
print("READING-LEVEL EVENTS")
print("=" * 70)

reading_event_columns = [
    "speedRisk",
    "hardAcceleration",
    "hardBraking",
    "suddenTurn",
]

for column in reading_event_columns:

    if column in df.columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

        print(
            f"{column}: "
            f"{int(values.sum())}"
        )


# ==========================================================
# NIGHT RIDES
# ==========================================================

print()
print("=" * 70)
print("DAY / NIGHT")
print("=" * 70)

if "isNight" in df.columns:

    night = (
        pd.to_numeric(
            df["isNight"],
            errors="coerce"
        )
        .fillna(0)
    )

    print(
        "Night readings:",
        int(
            (night == 1).sum()
        )
    )

    print(
        "Day readings:",
        int(
            (night == 0).sum()
        )
    )


# ==========================================================
# RIDE LENGTH CHECK
# ==========================================================

print()
print("=" * 70)
print("RIDE LENGTH CHECK")
print("=" * 70)

ride_reading_counts = (
    df.groupby("rideId")
    .size()
    .sort_values()
)

print()

print(
    ride_reading_counts.to_string()
)

print()

print(
    "Rides with >= 30 readings:",
    int(
        (ride_reading_counts >= 30).sum()
    )
)

print(
    "Rides with < 30 readings:",
    int(
        (ride_reading_counts < 30).sum()
    )
)


# ==========================================================
# DATA QUALITY
# ==========================================================

print()
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = (
    df.isna()
    .sum()
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
    .sort_values(
        ascending=False
    )
)

if len(missing_values):

    print(
        missing_values.to_string()
    )

else:

    print(
        "No missing values."
    )


# ==========================================================
# DUPLICATE CHECK
# ==========================================================

print()
print("=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicates = df.duplicated().sum()

print(
    "Duplicate complete rows:",
    int(duplicates)
)


# ==========================================================
# PER-RIDE ANALYSIS CSV
# ==========================================================

ride_analysis = ride_df.copy()

reading_counts = (
    df.groupby("rideId")
    .size()
    .rename(
        "actualReadingCount"
    )
)

ride_analysis = ride_analysis.merge(
    reading_counts,
    on="rideId",
    how="left"
)


ride_analysis["avgSpeedFromReadings"] = (
    df.groupby("rideId")[
        "speedKmh"
    ]
    .mean()
    .values
)


ride_analysis["maxSpeedFromReadings"] = (
    df.groupby("rideId")[
        "speedKmh"
    ]
    .max()
    .values
)


ride_analysis["avgAcceleration"] = (
    df.groupby("rideId")[
        "speedAccelerationMs2"
    ]
    .mean()
    .values
)


ride_analysis["maxAcceleration"] = (
    df.groupby("rideId")[
        "speedAccelerationMs2"
    ]
    .max()
    .values
)


ride_analysis["minAcceleration"] = (
    df.groupby("rideId")[
        "speedAccelerationMs2"
    ]
    .min()
    .values
)


ride_analysis["avgGyroMagnitude"] = (
    df.groupby("rideId")[
        "gyroMagnitudeRadS"
    ]
    .mean()
    .values
)


ride_analysis["maxGyroMagnitude"] = (
    df.groupby("rideId")[
        "gyroMagnitudeRadS"
    ]
    .max()
    .values
)


ride_analysis["hardBrakingFromSensors"] = (
    df.assign(
        hard_brake_sensor=(
            df["speedAccelerationMs2"]
            <= -2.5
        )
    )
    .groupby("rideId")[
        "hard_brake_sensor"
    ]
    .sum()
    .values
)


ride_analysis["hardAccelerationFromSensors"] = (
    df.assign(
        hard_accel_sensor=(
            df["speedAccelerationMs2"]
            >= 2.5
        )
    )
    .groupby("rideId")[
        "hard_accel_sensor"
    ]
    .sum()
    .values
)


ride_analysis["suddenTurnsFromSensors"] = (
    df.assign(
        sudden_turn_sensor=(
            df["gyroMagnitudeRadS"]
            >= 1.2
        )
    )
    .groupby("rideId")[
        "sudden_turn_sensor"
    ]
    .sum()
    .values
)


# ==========================================================
# SAVE
# ==========================================================

ride_analysis_path = (
    OUTPUT_DIR
    / "ride_quality_summary.csv"
)

ride_analysis.to_csv(
    ride_analysis_path,
    index=False
)


# ==========================================================
# SAVE SENSOR STATISTICS
# ==========================================================

sensor_stats = []

for column in numeric_columns:

    if column not in df.columns:
        continue

    series = df[column].dropna()

    if len(series) == 0:
        continue

    sensor_stats.append({

        "feature": column,

        "count": len(series),

        "missing": int(
            df[column].isna().sum()
        ),

        "mean": series.mean(),

        "std": series.std(),

        "min": series.min(),

        "median": series.median(),

        "max": series.max(),

        "p95": series.quantile(
            0.95
        ),

        "p99": series.quantile(
            0.99
        ),

    })


sensor_stats_df = pd.DataFrame(
    sensor_stats
)


sensor_stats_path = (
    OUTPUT_DIR
    / "sensor_statistics.csv"
)


sensor_stats_df.to_csv(
    sensor_stats_path,
    index=False
)


# ==========================================================
# SAVE DATASET REPORT
# ==========================================================

report = {

    "rides":
        int(ride_count),

    "sensorReadings":
        int(reading_count),

    "averageReadingsPerRide":
        float(
            reading_count / ride_count
        )
        if ride_count
        else 0,

    "ridesWithAtLeast30Readings":
        int(
            (ride_reading_counts >= 30)
            .sum()
        ),

    "ridesBelow30Readings":
        int(
            (ride_reading_counts < 30)
            .sum()
        ),

    "duplicateRows":
        int(duplicates),

    "missingValues":
        {
            str(k): int(v)
            for k, v
            in missing_values.items()
        },

    "outputDirectory":
        str(OUTPUT_DIR),

}


report_path = (
    OUTPUT_DIR
    / "dataset_quality_report.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2
    ),
    encoding="utf-8"
)


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 70)
print("DATASET ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "Ride summary:"
)

print(
    ride_analysis_path
)

print()

print(
    "Sensor statistics:"
)

print(
    sensor_stats_path
)

print()

print(
    "Quality report:"
)

print(
    report_path
)

print()

print("=" * 70)
print("NO SOURCE DATA WAS MODIFIED")
print("=" * 70)