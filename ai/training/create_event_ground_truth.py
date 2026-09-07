from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================================
# RIDEGUARDIAN
# EVENT GROUND-TRUTH DATASET
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

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
    / "event_ground_truth"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

SEQUENCE_LENGTH = 30

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

EVENT_COLUMNS = {

    "HIGH_SPEED":
        "speedRisk",

    "HARD_ACCELERATION":
        "hardAcceleration",

    "HARD_BRAKING":
        "hardBraking",

    "SUDDEN_TURN":
        "suddenTurn",

}


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
# LOAD DATA
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("EVENT GROUND-TRUTH DATASET")
print("=" * 70)

print()

print("Input:")
print(INPUT_FILE)

print()


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    "Rows loaded:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required = [

    "rideId",

    "timestamp",

    "split",

] + FEATURE_COLUMNS + list(
    EVENT_COLUMNS.values()
)


missing = [

    column

    for column in required

    if column not in df.columns

]


if missing:

    print()

    print(
        "Missing columns:"
    )

    for column in missing:

        print(
            "  -",
            column
        )

    raise ValueError(
        "Required columns are missing."
    )


# ==========================================================
# CLEAN TYPES
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


for column in EVENT_COLUMNS.values():

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
# EVENT DETECTION
# ==========================================================

def get_event_list(row):

    events = []

    for event_name, column in EVENT_COLUMNS.items():

        if row[column] == 1:

            events.append(
                event_name
            )

    return events


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


event_indices = {

    "train": [],
    "val": [],
    "test": [],

}


manifest_rows = []


# ==========================================================
# PROCESS RIDES
# ==========================================================

print()
print("=" * 70)
print("PROCESSING RIDES")
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


    count = len(
        ride_df
    )


    split = str(
        ride_df["split"].iloc[0]
    ).lower()


    if split not in sequences:

        print(
            f"Ride {ride_id}: "
            f"invalid split '{split}'"
        )

        continue


    if count < SEQUENCE_LENGTH:

        print(
            f"Ride {ride_id}: "
            f"{count} readings - too short"
        )

        continue


    # ------------------------------------------------------
    # Find every event reading
    # ------------------------------------------------------

    event_positions = []


    for position in range(count):

        row = ride_df.iloc[
            position
        ]


        events = get_event_list(
            row
        )


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
    # Create sequence around EVERY event
    # ------------------------------------------------------

    for event_number, (
        event_position,
        events
    ) in enumerate(
        event_positions
    ):

        # --------------------------------------------------
        # Window start
        # --------------------------------------------------

        start = (
            event_position
            - BEFORE_EVENT
        )


        # --------------------------------------------------
        # If event is too close to beginning,
        # shift the window right.
        # --------------------------------------------------

        if start < 0:

            start = 0


        end = (
            start
            + SEQUENCE_LENGTH
        )


        # --------------------------------------------------
        # If event is too close to end,
        # shift window left.
        # --------------------------------------------------

        if end > count:

            end = count

            start = (
                count
                - SEQUENCE_LENGTH
            )


        # --------------------------------------------------
        # Safety
        # --------------------------------------------------

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
                f"window creation failed"
            )

            continue


        # --------------------------------------------------
        # Confirm event is inside window
        # --------------------------------------------------

        target_offset = (
            event_position
            - start
        )


        if not (
            0
            <= target_offset
            < SEQUENCE_LENGTH
        ):

            print(
                f"  Event {event_position}: "
                f"target outside window"
            )

            continue


        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        if len(events) > 1:

            label = "MULTI_EVENT"

        else:

            label = events[0]


        # --------------------------------------------------
        # Sensor features
        # --------------------------------------------------

        feature_df = window[
            FEATURE_COLUMNS
        ].copy()


        feature_df = (

            feature_df

            .interpolate(
                method="linear",
                limit_direction="both"
            )

            .fillna(0)

        )


        sequence = feature_df.to_numpy(
            dtype=np.float32
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        sequences[split].append(
            sequence
        )

        labels[split].append(
            LABEL_MAP[label]
        )

        ride_ids[split].append(
            ride_id
        )

        event_indices[split].append(
            target_offset
        )


        # --------------------------------------------------
        # Timestamp
        # --------------------------------------------------

        event_timestamp = ""

        if pd.notna(
            ride_df[
                "timestamp"
            ].iloc[
                event_position
            ]
        ):

            event_timestamp = str(
                ride_df[
                    "timestamp"
                ].iloc[
                    event_position
                ]
            )


        # --------------------------------------------------
        # Manifest
        # --------------------------------------------------

        manifest_rows.append({

            "split":
                split,

            "rideId":
                ride_id,

            "eventNumber":
                event_number,

            "eventReadingIndex":
                event_position,

            "windowStart":
                start,

            "windowEnd":
                end - 1,

            "eventOffset":
                target_offset,

            "eventTimestamp":
                event_timestamp,

            "label":
                label,

            "labelId":
                LABEL_MAP[label],

            "eventsAtReading":
                "|".join(events),

        })


# ==========================================================
# ADD NORMAL WINDOWS
# ==========================================================

print()
print("=" * 70)
print("ADDING NORMAL WINDOWS")
print("=" * 70)

print()


normal_added = {

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


    count = len(
        ride_df
    )


    if count < SEQUENCE_LENGTH:

        continue


    split = str(
        ride_df["split"].iloc[0]
    ).lower()


    if split not in sequences:

        continue


    # ------------------------------------------------------
    # Candidate normal windows
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


        # --------------------------------------------------
        # Skip if ANY event exists
        # --------------------------------------------------

        contains_event = False


        for _, row in window.iterrows():

            if get_event_list(row):

                contains_event = True

                break


        if contains_event:

            continue


        # --------------------------------------------------
        # Features
        # --------------------------------------------------

        feature_df = window[
            FEATURE_COLUMNS
        ].copy()


        feature_df = (

            feature_df

            .interpolate(
                method="linear",
                limit_direction="both"
            )

            .fillna(0)

        )


        sequence = feature_df.to_numpy(
            dtype=np.float32
        )


        sequences[split].append(
            sequence
        )

        labels[split].append(
            LABEL_MAP["NORMAL"]
        )

        ride_ids[split].append(
            ride_id
        )

        event_indices[split].append(
            -1
        )


        manifest_rows.append({

            "split":
                split,

            "rideId":
                ride_id,

            "eventNumber":
                -1,

            "eventReadingIndex":
                -1,

            "windowStart":
                start,

            "windowEnd":
                end - 1,

            "eventOffset":
                -1,

            "eventTimestamp":
                "",

            "label":
                "NORMAL",

            "labelId":
                LABEL_MAP["NORMAL"],

            "eventsAtReading":
                "",

        })


        normal_added[split] += 1


print(
    "Normal windows:"
)

for split in normal_added:

    print(
        f"  {split}: "
        f"{normal_added[split]}"
    )


# ==========================================================
# SAVE
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
            f"{split.upper()}: empty"
        )

        continue


    X = np.stack(
        sequences[split]
    ).astype(
        np.float32
    )


    y = np.array(
        labels[split],
        dtype=np.int64
    )


    ids = np.array(
        ride_ids[split],
        dtype=np.int64
    )


    events = np.array(
        event_indices[split],
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

        event_indices=events,

    )


    print()

    print(
        split.upper()
    )

    print(
        "Shape:",
        X.shape
    )

    print(
        "Sequences:",
        len(X)
    )


    for label, label_id in LABEL_MAP.items():

        amount = int(
            (y == label_id).sum()
        )

        print(
            f"  {label:20} "
            f"{amount}"
        )


    print(
        "Saved:",
        output_file
    )


# ==========================================================
# MANIFEST
# ==========================================================

manifest = pd.DataFrame(
    manifest_rows
)


manifest_file = (
    OUTPUT_DIR
    / "event_ground_truth_manifest.csv"
)


manifest.to_csv(
    manifest_file,
    index=False
)


# ==========================================================
# SOURCE EVENT COUNTS
# ==========================================================

print()
print("=" * 70)
print("SOURCE EVENT COUNTS")
print("=" * 70)

print()


source_event_counts = {}


for event_name, column in EVENT_COLUMNS.items():

    count = int(
        df[column].sum()
    )


    source_event_counts[
        event_name
    ] = count


    print(
        f"{event_name:20} "
        f"{count}"
    )


# ==========================================================
# FINAL DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("FINAL DISTRIBUTION")
print("=" * 70)

print()


final_distribution = {}


for split in [
    "train",
    "val",
    "test",
]:

    print(
        f"{split.upper()}:"
    )


    final_distribution[
        split
    ] = {}


    for label, label_id in LABEL_MAP.items():

        amount = sum(

            1

            for value
            in labels[split]

            if value == label_id

        )


        final_distribution[
            split
        ][label] = amount


        print(
            f"  {label:20} "
            f"{amount}"
        )


    print()


# ==========================================================
# SAVE INFO
# ==========================================================

info = {

    "sequenceLength":
        SEQUENCE_LENGTH,

    "beforeEvent":
        BEFORE_EVENT,

    "afterEvent":
        AFTER_EVENT,

    "features":
        FEATURE_COLUMNS,

    "labels":
        LABEL_MAP,

    "sourceEventCounts":
        source_event_counts,

    "distribution":
        final_distribution,

    "rawCsvModified":
        False,

}


info_file = (
    OUTPUT_DIR
    / "dataset_info.json"
)


info_file.write_text(

    json.dumps(
        info,
        indent=2
    ),

    encoding="utf-8"

)


# ==========================================================
# COMPLETE
# ==========================================================

print("=" * 70)
print("EVENT GROUND-TRUTH DATASET COMPLETE")
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
    manifest_file
)

print()

print(
    "Raw CSV was NOT modified."
)

print()

print("=" * 70)