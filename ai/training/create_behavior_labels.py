from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================================
# RIDEGUARDIAN
# EVENT-CENTERED BEHAVIOUR DATASET CREATOR - V3
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
    / "behavior_events_v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

SEQUENCE_LENGTH = 30

# Preferred context around the target event.
BEFORE_EVENT = 15
AFTER_EVENT = 14


# ==========================================================
# FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "speedKmh",
    "speedChangeKmh",
    "speedAccelerationMs2",

    "accelerationX",
    "accelerationY",
    "accelerationZ",

    "gyroX",
    "gyroY",
    "gyroZ",

    "timeSincePreviousReadingSec",
]



# ==========================================================
# EVENT COLUMNS
# ==========================================================

EVENT_COLUMNS = [
    "speedRisk",
    "hardAcceleration",
    "hardBraking",
    "suddenTurn",
]


# ==========================================================
# LABELS
# ==========================================================

LABEL_MAP = {
    "NORMAL": 0,
    "HIGH_SPEED": 1,
    "HARD_ACCELERATION": 2,
    "HARD_BRAKING": 3,
    "SUDDEN_TURN": 4,
    "MULTI_EVENT": 5,
}


# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("EVENT-CENTERED BEHAVIOUR DATASET V3")
print("=" * 70)

print()
print("Input:")
print(INPUT_FILE)
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))
print("Columns:", len(df.columns))


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [
    "rideId",
    "timestamp",
    "split",
] + FEATURE_COLUMNS + EVENT_COLUMNS


missing = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing:

    print()
    print("Missing columns:")

    for column in missing:
        print("  -", column)

    raise ValueError(
        "Required columns are missing."
    )


# ==========================================================
# TYPE CONVERSION
# ==========================================================

df["rideId"] = pd.to_numeric(
    df["rideId"],
    errors="coerce"
)

df = df[
    df["rideId"].notna()
].copy()

df["rideId"] = df[
    "rideId"
].astype(int)


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)


for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


for column in EVENT_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ==========================================================
# SORT
# ==========================================================

df = df.sort_values(
    [
        "rideId",
        "timestamp",
    ],
    kind="stable"
).reset_index(
    drop=True
)


# ==========================================================
# EVENT HELPERS
# ==========================================================

EVENT_TO_COLUMN = {
    "HIGH_SPEED": "speedRisk",
    "HARD_ACCELERATION": "hardAcceleration",
    "HARD_BRAKING": "hardBraking",
    "SUDDEN_TURN": "suddenTurn",
}


def get_events(row):

    events = []

    for event_name, column in EVENT_TO_COLUMN.items():

        if row[column] == 1:
            events.append(event_name)

    return events


def event_priority(events):

    priority = [
        "HARD_BRAKING",
        "HARD_ACCELERATION",
        "SUDDEN_TURN",
        "HIGH_SPEED",
    ]

    for event in priority:

        if event in events:
            return event

    return None


# ==========================================================
# STORAGE
# ==========================================================

sequences = {
    "train": [],
    "val": [],
    "test": [],
}

labels = {
    "train": [],
    "val": [],
    "test": [],
}

ride_ids = {
    "train": [],
    "val": [],
    "test": [],
}

target_positions = {
    "train": [],
    "val": [],
    "test": [],
}

manifest_rows = []


# ==========================================================
# CREATE EVENT WINDOWS
# ==========================================================

print()
print("=" * 70)
print("CREATING EVENT WINDOWS")
print("=" * 70)
print()


for ride_id, ride_df in df.groupby(
    "rideId",
    sort=True
):

    ride_df = ride_df.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(
        drop=True
    )

    count = len(ride_df)

    if count < SEQUENCE_LENGTH:

        print(
            f"Ride {ride_id}: "
            f"{count} readings - too short"
        )

        continue


    split = str(
        ride_df["split"].iloc[0]
    ).lower()


    if split not in sequences:

        print(
            f"Ride {ride_id}: "
            f"invalid split '{split}'"
        )

        continue


    # ------------------------------------------------------
    # Locate actual event readings
    # ------------------------------------------------------

    event_positions = []

    for position in range(count):

        row = ride_df.iloc[position]

        events = get_events(row)

        if events:

            event_positions.append(
                (
                    position,
                    events
                )
            )


    print(
        f"Ride {ride_id}: "
        f"{count} readings, "
        f"{len(event_positions)} event readings"
    )


    # ------------------------------------------------------
    # Each real event gets its own window
    # ------------------------------------------------------

    for event_number, (
        event_position,
        center_events
    ) in enumerate(event_positions):

        # --------------------------------------------------
        # Choose a 30-reading window containing the event.
        #
        # Normal case:
        #
        # event_position - 15
        # through
        # event_position + 14
        #
        # If event is near beginning/end, shift the window.
        # --------------------------------------------------

        start = (
            event_position
            - BEFORE_EVENT
        )

        end = (
            start
            + SEQUENCE_LENGTH
        )


        # Shift forward if start is negative.

        if start < 0:

            start = 0

            end = SEQUENCE_LENGTH


        # Shift backward if end exceeds ride.

        if end > count:

            end = count

            start = (
                count
                - SEQUENCE_LENGTH
            )


        # Final safety check.

        if start < 0:
            start = 0

        if end > count:
            end = count


        window = ride_df.iloc[
            start:end
        ].copy()


        if len(window) != SEQUENCE_LENGTH:

            print(
                f"  Event {event_position}: "
                f"could not create 30-reading window"
            )

            continue


        # --------------------------------------------------
        # Find target position INSIDE window
        # --------------------------------------------------

        target_offset = (
            event_position
            - start
        )


        # --------------------------------------------------
        # Determine events inside window
        # --------------------------------------------------

        window_events = set()

        for _, row in window.iterrows():

            row_events = get_events(row)

            for event in row_events:
                window_events.add(event)


        # --------------------------------------------------
        # TARGET LABEL
        # --------------------------------------------------
        #
        # IMPORTANT:
        #
        # The event at event_position is the target.
        #
        # Other events occurring in the same window do NOT
        # automatically turn it into MULTI_EVENT.
        #
        # MULTI_EVENT is used only when the TARGET READING
        # itself contains multiple event flags.
        # --------------------------------------------------

        if len(center_events) > 1:

            label = "MULTI_EVENT"

        else:

            label = center_events[0]


        # --------------------------------------------------
        # SENSOR DATA
        # --------------------------------------------------

        sensor_df = window[
            FEATURE_COLUMNS
        ].copy()


        sensor_df = (

            sensor_df

            .interpolate(
                method="linear",
                limit_direction="both"
            )

            .fillna(0)

        )


        sequence = sensor_df.to_numpy(
            dtype=np.float32
        )


        # --------------------------------------------------
        # STORE
        # --------------------------------------------------

        sequences[split].append(
            sequence
        )

        labels[split].append(
            label
        )

        ride_ids[split].append(
            ride_id
        )

        target_positions[split].append(
            event_position
        )


        # --------------------------------------------------
        # MANIFEST
        # --------------------------------------------------

        manifest_rows.append({

            "split": split,

            "rideId": ride_id,

            "eventNumber": event_number,

            "targetReadingIndex":
                event_position,

            "windowStart":
                start,

            "windowEnd":
                end - 1,

            "targetOffset":
                target_offset,

            "sequenceLength":
                SEQUENCE_LENGTH,

            "label":
                label,

            "labelId":
                LABEL_MAP[label],

            "eventsAtTarget":
                "|".join(
                    center_events
                ),

            "eventsInWindow":
                "|".join(
                    sorted(window_events)
                ),

        })


# ==========================================================
# ADD NORMAL WINDOWS
# ==========================================================

print()
print("=" * 70)
print("ADDING NORMAL WINDOWS")
print("=" * 70)
print()


normal_counts = {
    "train": 0,
    "val": 0,
    "test": 0,
}


for ride_id, ride_df in df.groupby(
    "rideId",
    sort=True
):

    ride_df = ride_df.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(
        drop=True
    )

    count = len(ride_df)

    if count < SEQUENCE_LENGTH:
        continue


    split = str(
        ride_df["split"].iloc[0]
    ).lower()


    if split not in sequences:
        continue


    # ------------------------------------------------------
    # Create non-overlapping candidate normal windows.
    # ------------------------------------------------------

    for start in range(
        0,
        count - SEQUENCE_LENGTH + 1,
        SEQUENCE_LENGTH
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        window = ride_df.iloc[
            start:end
        ].copy()


        # Check for ANY event.

        has_event = False

        for _, row in window.iterrows():

            if get_events(row):

                has_event = True

                break


        if has_event:
            continue


        sensor_df = window[
            FEATURE_COLUMNS
        ].copy()


        sensor_df = (

            sensor_df

            .interpolate(
                method="linear",
                limit_direction="both"
            )

            .fillna(0)

        )


        sequence = sensor_df.to_numpy(
            dtype=np.float32
        )


        sequences[split].append(
            sequence
        )

        labels[split].append(
            "NORMAL"
        )

        ride_ids[split].append(
            ride_id
        )

        target_positions[split].append(
            -1
        )


        manifest_rows.append({

            "split": split,

            "rideId": ride_id,

            "eventNumber": -1,

            "targetReadingIndex": -1,

            "windowStart": start,

            "windowEnd": end - 1,

            "targetOffset": -1,

            "sequenceLength":
                SEQUENCE_LENGTH,

            "label": "NORMAL",

            "labelId":
                LABEL_MAP["NORMAL"],

            "eventsAtTarget": "",

            "eventsInWindow": "",

        })


        normal_counts[split] += 1


print(
    "Normal windows added:"
)

for split in normal_counts:

    print(
        f"  {split}: "
        f"{normal_counts[split]}"
    )


# ==========================================================
# SAVE DATASETS
# ==========================================================

print()
print("=" * 70)
print("SAVING DATASETS")
print("=" * 70)


for split in [
    "train",
    "val",
    "test",
]:

    if not sequences[split]:

        print(
            f"{split.upper()}: NO DATA"
        )

        continue


    X = np.stack(
        sequences[split]
    ).astype(
        np.float32
    )


    y = np.array(
        [
            LABEL_MAP[label]
            for label in labels[split]
        ],
        dtype=np.int64
    )


    ids = np.array(
        ride_ids[split],
        dtype=np.int64
    )


    targets = np.array(
        target_positions[split],
        dtype=np.int64
    )


    output_file = (
        OUTPUT_DIR
        / f"{split}.npz"
    )


    np.savez_compressed(

        output_file,

        X=X,

        y=y,

        ride_ids=ids,

        target_positions=targets,

    )


    print()
    print(split.upper())

    print(
        "Shape:",
        X.shape
    )

    print(
        "Sequences:",
        len(X)
    )

    print()


    for label, label_id in LABEL_MAP.items():

        amount = int(
            (y == label_id).sum()
        )

        print(
            f"  {label:20} {amount}"
        )


    print()
    print(
        "Saved:",
        output_file
    )


# ==========================================================
# SAVE MANIFEST
# ==========================================================

manifest = pd.DataFrame(
    manifest_rows
)


manifest_path = (
    OUTPUT_DIR
    / "behavior_event_manifest.csv"
)


manifest.to_csv(
    manifest_path,
    index=False
)


# ==========================================================
# SAVE CONFIG
# ==========================================================

config = {

    "sequenceLength":
        SEQUENCE_LENGTH,

    "beforeEvent":
        BEFORE_EVENT,

    "afterEvent":
        AFTER_EVENT,

    "features":
        FEATURE_COLUMNS,

    "eventColumns":
        EVENT_COLUMNS,

    "labels":
        LABEL_MAP,

    "inputFile":
        str(INPUT_FILE),

    "rawDataModified":
        False,

}


config_path = (
    OUTPUT_DIR
    / "dataset_info.json"
)


config_path.write_text(

    json.dumps(
        config,
        indent=2
    ),

    encoding="utf-8"

)


# ==========================================================
# FINAL DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("FINAL DATASET DISTRIBUTION")
print("=" * 70)
print()


for split in [
    "train",
    "val",
    "test",
]:

    print(
        f"{split.upper()}:"
    )

    for label, label_id in LABEL_MAP.items():

        amount = sum(

            1

            for item in labels[split]

            if item == label

        )

        print(
            f"  {label:20} {amount}"
        )

    print()


# ==========================================================
# EVENT COUNTS DIRECTLY FROM SOURCE
# ==========================================================

print("=" * 70)
print("SOURCE EVENT COUNTS")
print("=" * 70)
print()


for event, column in EVENT_TO_COLUMN.items():

    count = int(
        df[column].sum()
    )

    print(
        f"{event:20} {count}"
    )


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 70)
print("EVENT-CENTERED DATASET V3 COMPLETE")
print("=" * 70)

print()

print(
    "Output:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Manifest:"
)

print(
    manifest_path
)

print()

print(
    "Raw CSV was NOT modified."
)

print()
print("=" * 70)