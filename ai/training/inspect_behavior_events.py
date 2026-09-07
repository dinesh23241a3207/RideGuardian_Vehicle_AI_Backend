from pathlib import Path
import pandas as pd


# ==========================================================
# RIDEGUARDIAN - EVENT INSPECTOR
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
print("RIDEGUARDIAN BEHAVIOUR EVENT INSPECTOR")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

# Numeric conversion
numeric_columns = [
    "rideId",
    "readingIndex",
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
    "hardBraking",
    "hardAcceleration",
    "suddenTurn",
    "speedRisk",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


df = df.sort_values(
    ["rideId", "timestamp"],
    kind="stable"
).reset_index(drop=True)


# ==========================================================
# FIND HARD BRAKING
# ==========================================================

hard_braking = df[
    df["hardBraking"] == 1
].copy()


print()
print("=" * 70)
print("HARD BRAKING EVENTS")
print("=" * 70)

print()

print(
    "Total hard-braking readings:",
    len(hard_braking)
)


if len(hard_braking) == 0:

    print(
        "No hard-braking readings found."
    )

    raise SystemExit


# ==========================================================
# INSPECT EACH EVENT
# ==========================================================

EVENT_RADIUS = 15

all_context = []


for _, event in hard_braking.iterrows():

    ride_id = int(
        event["rideId"]
    )

    event_index = event.name

    print()
    print("-" * 70)

    print(
        f"Ride ID: {ride_id}"
    )

    print(
        f"Timestamp: {event['timestamp']}"
    )

    print(
        f"Reading index: {event['readingIndex']}"
    )

    print(
        f"Speed: {event['speedKmh']:.2f} km/h"
    )

    print(
        f"Speed change: "
        f"{event['speedChangeKmh']:.3f} km/h"
    )

    print(
        f"Speed acceleration: "
        f"{event['speedAccelerationMs2']:.3f} m/s²"
    )

    print(
        f"Acceleration magnitude: "
        f"{event['accelerationMagnitude']:.3f}"
    )

    print(
        f"Hard braking flag: "
        f"{event['hardBraking']}"
    )

    print(
        f"Split: {event['split']}"
    )

    print()

    print(
        "Surrounding readings:"
    )

    # Get only the same ride
    ride_df = df[
        df["rideId"] == ride_id
    ].copy()

    # Position within ride
    positions = ride_df.index.tolist()

    try:
        position = positions.index(
            event_index
        )
    except ValueError:
        continue

    start = max(
        0,
        position - EVENT_RADIUS
    )

    end = min(
        len(ride_df),
        position + EVENT_RADIUS + 1
    )

    context = ride_df.iloc[
        start:end
    ].copy()

    display_columns = [
        "rideId",
        "readingIndex",
        "timestamp",
        "speedKmh",
        "speedChangeKmh",
        "speedAccelerationMs2",
        "accelerationX",
        "accelerationY",
        "accelerationZ",
        "gyroMagnitudeRadS",
        "hardBraking",
        "hardAcceleration",
        "suddenTurn",
        "speedRisk",
        "eventType",
        "eventSeverity",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in context.columns
    ]

    print(
        context[
            display_columns
        ].to_string(index=False)
    )

    context["eventRideId"] = ride_id

    context["eventReadingIndex"] = (
        event["readingIndex"]
    )

    all_context.append(
        context
    )


# ==========================================================
# SAVE CONTEXT
# ==========================================================

if all_context:

    context_df = pd.concat(
        all_context,
        ignore_index=True
    )

    output_file = (
        OUTPUT_DIR
        / "hard_braking_context.csv"
    )

    context_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("CONTEXT SAVED")
    print("=" * 70)

    print()
    print(output_file)


# ==========================================================
# ALSO INSPECT HARD ACCELERATION
# ==========================================================

print()
print("=" * 70)
print("OTHER EVENT COUNTS")
print("=" * 70)

for column in [
    "hardAcceleration",
    "suddenTurn",
    "speedRisk",
]:

    if column in df.columns:

        count = int(
            (
                df[column] == 1
            ).sum()
        )

        print(
            f"{column}: {count}"
        )


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 70)
print("EVENT INSPECTION COMPLETE")
print("=" * 70)

print()
print("Source CSV was NOT modified.")