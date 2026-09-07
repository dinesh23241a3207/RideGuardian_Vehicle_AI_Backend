from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


# ==========================================================
# RIDEGUARDIAN
# LIVE / CSV RIDER BEHAVIOUR PREDICTION
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

# ----------------------------------------------------------
# Input processed ride CSV
# ----------------------------------------------------------

INPUT_FILE = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "processed"
    / "ride_readings_processed_with_split.csv"
)

# ----------------------------------------------------------
# Trained model
# ----------------------------------------------------------

MODEL_FILE = (
    BASE_DIR
    / "ai"
    / "models"
    / "rider_behavior"
    / "behavior_gru.keras"
)

# ----------------------------------------------------------
# Scaler
# ----------------------------------------------------------

SCALER_FILE = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4_scaled"
    / "behavior_scaler.joblib"
)

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "models"
    / "rider_behavior"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "behavior_ride_predictions.csv"
)


# ==========================================================
# SETTINGS
# ==========================================================

SEQUENCE_LENGTH = 30

STRIDE = 10


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
# LABELS
# ==========================================================

LABEL_NAMES = [

    "NORMAL",

    "HIGH_SPEED",

    "HARD_ACCELERATION",

    "HARD_BRAKING",

    "SUDDEN_TURN",

    "MULTI_EVENT",

]


# ==========================================================
# RULE-BASED EVENT COLUMNS
# ==========================================================

RULE_EVENT_COLUMNS = [

    "speedRisk",

    "hardAcceleration",

    "hardBraking",

    "suddenTurn",

]


# ==========================================================
# PRINT HEADER
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("RIDER BEHAVIOUR PREDICTION")
print("=" * 70)

print()


# ==========================================================
# CHECK FILES
# ==========================================================

for file in [

    INPUT_FILE,

    MODEL_FILE,

    SCALER_FILE,

]:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )


print(
    "Input:",
    INPUT_FILE
)

print(
    "Model:",
    MODEL_FILE
)

print(
    "Scaler:",
    SCALER_FILE
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print()
print(
    "Loading model..."
)


model = tf.keras.models.load_model(
    MODEL_FILE
)


print(
    "Model loaded."
)


# ==========================================================
# LOAD SCALER
# ==========================================================

print()
print(
    "Loading scaler..."
)


scaler = joblib.load(
    SCALER_FILE
)


print(
    "Scaler loaded."
)


# ==========================================================
# LOAD CSV
# ==========================================================

print()
print(
    "Loading ride data..."
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ==========================================================
# CHECK COLUMNS
# ==========================================================

required_columns = [

    "rideId",

] + FEATURE_COLUMNS


missing = [

    column

    for column in required_columns

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
# NUMERIC CONVERSION
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


for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================================
# SORT
# ==========================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )


    df = df.sort_values(

        [
            "rideId",
            "timestamp",
        ],

        kind="stable"

    ).reset_index(
        drop=True
    )

else:

    df = df.sort_values(
        [
            "rideId"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )


# ==========================================================
# PREDICTION STORAGE
# ==========================================================

prediction_rows = []


# ==========================================================
# PROCESS EACH RIDE
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

    ride_df = ride_df.reset_index(
        drop=True
    )


    count = len(
        ride_df
    )


    print(
        f"Ride {ride_id}: "
        f"{count} readings"
    )


    if count < SEQUENCE_LENGTH:

        print(
            "  Too short - skipped"
        )

        continue


    # ------------------------------------------------------
    # Generate rolling sequences
    # ------------------------------------------------------

    for start in range(

        0,

        count - SEQUENCE_LENGTH + 1,

        STRIDE

    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )


        window = ride_df.iloc[
            start:end
        ].copy()


        # --------------------------------------------------
        # Prepare features
        # --------------------------------------------------

        feature_data = window[
            FEATURE_COLUMNS
        ].copy()


        feature_data = (

            feature_data

            .interpolate(

                method="linear",

                limit_direction="both"

            )

            .fillna(0)

        )


        X = feature_data.to_numpy(
            dtype=np.float32
        )


        # --------------------------------------------------
        # Normalize using SAME training scaler
        # --------------------------------------------------

        X_flat = X.reshape(
            -1,
            len(FEATURE_COLUMNS)
        )


        X_scaled_flat = scaler.transform(
            X_flat
        )


        X_scaled = X_scaled_flat.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        ).astype(
            np.float32
        )


        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        probabilities = model.predict(
            X_scaled,
            verbose=0
        )[0]


        predicted_id = int(
            np.argmax(
                probabilities
            )
        )


        predicted_label = LABEL_NAMES[
            predicted_id
        ]


        confidence = float(
            probabilities[
                predicted_id
            ]
        )


        # --------------------------------------------------
        # RULE-BASED EVENTS IN WINDOW
        # --------------------------------------------------

        rule_events = set()


        if "speedRisk" in window.columns:

            if (
                pd.to_numeric(
                    window["speedRisk"],
                    errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .any()
            ):

                rule_events.add(
                    "HIGH_SPEED"
                )


        if "hardAcceleration" in window.columns:

            if (
                pd.to_numeric(
                    window["hardAcceleration"],
                    errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .any()
            ):

                rule_events.add(
                    "HARD_ACCELERATION"
                )


        if "hardBraking" in window.columns:

            if (
                pd.to_numeric(
                    window["hardBraking"],
                    errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .any()
            ):

                rule_events.add(
                    "HARD_BRAKING"
                )


        if "suddenTurn" in window.columns:

            if (
                pd.to_numeric(
                    window["suddenTurn"],
                    errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .any()
            ):

                rule_events.add(
                    "SUDDEN_TURN"
                )


        # --------------------------------------------------
        # Existing event type
        # --------------------------------------------------

        existing_event_type = ""

        if "eventType" in window.columns:

            event_values = (

                window["eventType"]

                .dropna()

                .astype(str)

                .unique()

            )


            if len(event_values) > 0:

                existing_event_type = "|".join(
                    event_values
                )


        # --------------------------------------------------
        # Timestamp
        # --------------------------------------------------

        start_timestamp = ""

        end_timestamp = ""


        if "timestamp" in window.columns:

            start_timestamp = str(
                window["timestamp"].iloc[0]
            )

            end_timestamp = str(
                window["timestamp"].iloc[-1]
            )


        # --------------------------------------------------
        # Save prediction
        # --------------------------------------------------

        row = {

            "rideId":
                int(ride_id),

            "windowStart":
                start,

            "windowEnd":
                end - 1,

            "startTimestamp":
                start_timestamp,

            "endTimestamp":
                end_timestamp,

            "predictedBehavior":
                predicted_label,

            "predictedBehaviorId":
                predicted_id,

            "confidence":
                confidence,

            "ruleEvents":
                "|".join(
                    sorted(rule_events)
                ),

            "existingEventType":
                existing_event_type,

        }


        # --------------------------------------------------
        # Add all probabilities
        # --------------------------------------------------

        for class_id, class_name in enumerate(
            LABEL_NAMES
        ):

            row[
                f"prob_{class_name}"
            ] = float(
                probabilities[class_id]
            )


        prediction_rows.append(
            row
        )


# ==========================================================
# SAVE RESULTS
# ==========================================================

print()
print("=" * 70)
print("SAVING PREDICTIONS")
print("=" * 70)


if not prediction_rows:

    raise RuntimeError(
        "No predictions were generated."
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print(
    "Prediction rows:",
    len(predictions_df)
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)


# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 70)
print("MODEL PREDICTION SUMMARY")
print("=" * 70)

print()


prediction_counts = (
    predictions_df[
        "predictedBehavior"
    ]
    .value_counts()
)


for label in LABEL_NAMES:

    count = int(
        prediction_counts.get(
            label,
            0
        )
    )

    print(
        f"{label:20} {count}"
    )


# ==========================================================
# CONFIDENCE
# ==========================================================

print()

print(
    "Average confidence:",
    round(
        predictions_df[
            "confidence"
        ].mean(),
        4
    )
)

print(
    "Minimum confidence:",
    round(
        predictions_df[
            "confidence"
        ].min(),
        4
    )
)

print(
    "Maximum confidence:",
    round(
        predictions_df[
            "confidence"
        ].max(),
        4
    )
)


# ==========================================================
# RULE VS MODEL
# ==========================================================

print()
print("=" * 70)
print("RULE-BASED EVENT SUMMARY")
print("=" * 70)

print()


rule_counts = {

    "HIGH_SPEED": 0,

    "HARD_ACCELERATION": 0,

    "HARD_BRAKING": 0,

    "SUDDEN_TURN": 0,

}


for events in predictions_df[
    "ruleEvents"
]:

    if not isinstance(
        events,
        str
    ):

        continue


    for event in rule_counts:

        if event in events.split("|"):

            rule_counts[event] += 1


for event, count in rule_counts.items():

    print(
        f"{event:20} {count}"
    )


# ==========================================================
# MODEL VS RULE MATCH
# ==========================================================

print()
print("=" * 70)
print("MODEL / RULE COMPARISON")
print("=" * 70)

print()


def normalize_rule_label(events):

    if not isinstance(
        events,
        str
    ):

        return "NONE"


    values = [

        x

        for x in events.split("|")

        if x

    ]


    if len(values) == 0:

        return "NONE"


    if len(values) > 1:

        return "MULTI_EVENT"


    return values[0]


predictions_df[
    "rulePrimaryBehavior"
] = predictions_df[
    "ruleEvents"
].apply(
    normalize_rule_label
)


comparable = predictions_df[
    predictions_df[
        "rulePrimaryBehavior"
    ] != "NONE"
].copy()


if len(comparable) > 0:

    matches = (

        comparable[
            "predictedBehavior"
        ]

        ==

        comparable[
            "rulePrimaryBehavior"
        ]

    )


    print(
        "Windows containing rule events:",
        len(comparable)
    )

    print(
        "Model/rule matches:",
        int(matches.sum())
    )

    print(
        "Model/rule match rate:",
        round(
            float(matches.mean()),
            4
        )
    )

else:

    print(
        "No rule-based event windows found."
    )


# ==========================================================
# SAVE UPDATED CSV
# ==========================================================

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# SAVE SUMMARY JSON
# ==========================================================

summary = {

    "input":
        str(INPUT_FILE),

    "model":
        str(MODEL_FILE),

    "scaler":
        str(SCALER_FILE),

    "sequenceLength":
        SEQUENCE_LENGTH,

    "stride":
        STRIDE,

    "totalPredictionWindows":
        int(len(predictions_df)),

    "predictionDistribution":
        {
            label:
                int(
                    prediction_counts.get(
                        label,
                        0
                    )
                )

            for label
            in LABEL_NAMES
        },

    "averageConfidence":
        float(
            predictions_df[
                "confidence"
            ].mean()
        ),

    "minimumConfidence":
        float(
            predictions_df[
                "confidence"
            ].min()
        ),

    "maximumConfidence":
        float(
            predictions_df[
                "confidence"
            ].max()
        ),

    "ruleEventWindows":
        int(
            len(comparable)
        ),

}


summary_file = (
    OUTPUT_DIR
    / "behavior_prediction_summary.json"
)


summary_file.write_text(

    json.dumps(
        summary,
        indent=2
    ),

    encoding="utf-8"

)


# ==========================================================
# COMPLETE
# ==========================================================

print()

print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print()

print(
    "CSV:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Summary:"
)

print(
    summary_file
)

print()

print(
    "The trained model and scaler were NOT modified."
)

print()

print("=" * 70)