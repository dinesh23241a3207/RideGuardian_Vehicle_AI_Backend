from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================================
# RIDEGUARDIAN
# BEHAVIOUR PREDICTION DIAGNOSTICS
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

PREDICTION_FILE = (
    BASE_DIR
    / "ai"
    / "models"
    / "rider_behavior"
    / "behavior_ride_predictions.csv"
)

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


DIAGNOSTIC_FILE = (
    OUTPUT_DIR
    / "behavior_prediction_diagnostics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "behavior_prediction_diagnostics.json"
)


# ==========================================================
# LABELS
# ==========================================================

LABELS = [
    "NORMAL",
    "HIGH_SPEED",
    "HARD_ACCELERATION",
    "HARD_BRAKING",
    "SUDDEN_TURN",
    "MULTI_EVENT",
]


# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("BEHAVIOUR PREDICTION DIAGNOSTICS")
print("=" * 70)

print()

if not PREDICTION_FILE.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n"
        f"{PREDICTION_FILE}"
    )


df = pd.read_csv(
    PREDICTION_FILE
)


print(
    "Prediction rows:",
    len(df)
)


# ==========================================================
# RULE LABEL
# ==========================================================

def get_rule_label(events):

    if pd.isna(events):

        return "NONE"


    events = str(events).strip()


    if not events:

        return "NONE"


    values = [

        value.strip()

        for value in events.split("|")

        if value.strip()

    ]


    if len(values) == 0:

        return "NONE"


    if len(values) > 1:

        return "MULTI_EVENT"


    return values[0]


df["ruleBehavior"] = (
    df["ruleEvents"]
    .apply(get_rule_label)
)


# ==========================================================
# MODEL LABEL
# ==========================================================

df["modelBehavior"] = (
    df["predictedBehavior"]
    .astype(str)
)


# ==========================================================
# COMPARISON
# ==========================================================

df["comparable"] = (
    df["ruleBehavior"]
    != "NONE"
)


df["correct"] = (

    df["ruleBehavior"]

    ==

    df["modelBehavior"]

)


# ==========================================================
# DIAGNOSTIC CATEGORY
# ==========================================================

def classify_result(row):

    rule = row["ruleBehavior"]

    model = row["modelBehavior"]

    confidence = float(
        row["confidence"]
    )


    if rule == "NONE":

        return "NO_RULE_EVENT"


    if rule == model:

        return "MATCH"


    if confidence < 0.50:

        return "LOW_CONFIDENCE_MISMATCH"


    if confidence < 0.70:

        return "MEDIUM_CONFIDENCE_MISMATCH"


    return "HIGH_CONFIDENCE_MISMATCH"


df["diagnostic"] = (
    df.apply(
        classify_result,
        axis=1
    )
)


# ==========================================================
# SAVE FULL DIAGNOSTIC FILE
# ==========================================================

df.to_csv(
    DIAGNOSTIC_FILE,
    index=False
)


print()

print(
    "Diagnostic CSV saved:"
)

print(
    DIAGNOSTIC_FILE
)


# ==========================================================
# EVENT COMPARISON
# ==========================================================

comparable = df[
    df["comparable"]
].copy()


print()
print("=" * 70)
print("EVENT-BY-EVENT PERFORMANCE")
print("=" * 70)

print()


event_statistics = {}


for label in LABELS:

    subset = comparable[
        comparable["ruleBehavior"]
        == label
    ]


    if len(subset) == 0:

        continue


    matches = subset[
        "correct"
    ]


    match_rate = float(
        matches.mean()
    )


    avg_confidence = float(
        subset[
            "confidence"
        ].mean()
    )


    event_statistics[label] = {

        "windows":
            int(len(subset)),

        "matches":
            int(matches.sum()),

        "mismatches":
            int((~matches).sum()),

        "matchRate":
            match_rate,

        "averageConfidence":
            avg_confidence,

    }


    print(
        f"{label:20}"
    )

    print(
        f"  Windows:       {len(subset)}"
    )

    print(
        f"  Matches:       {int(matches.sum())}"
    )

    print(
        f"  Mismatches:    {int((~matches).sum())}"
    )

    print(
        f"  Match rate:    {match_rate:.3f}"
    )

    print(
        f"  Avg confidence:{avg_confidence:.3f}"
    )

    print()


# ==========================================================
# CONFUSION TABLE
# ==========================================================

print("=" * 70)
print("RULE → MODEL CONFUSION")
print("=" * 70)

print()


confusion = pd.crosstab(

    comparable["ruleBehavior"],

    comparable["modelBehavior"],

    dropna=False

)


print(
    confusion
)


# ==========================================================
# MISMATCHES
# ==========================================================

mismatches = comparable[
    ~comparable["correct"]
].copy()


print()
print("=" * 70)
print("MISMATCH ANALYSIS")
print("=" * 70)

print()

print(
    "Total event windows:",
    len(comparable)
)

print(
    "Matches:",
    int(comparable["correct"].sum())
)

print(
    "Mismatches:",
    len(mismatches)
)


if len(mismatches) > 0:

    print()

    print(
        "Mismatch predictions:"
    )

    columns = [

        "rideId",

        "windowStart",

        "windowEnd",

        "ruleBehavior",

        "modelBehavior",

        "confidence",

    ]


    print(
        mismatches[
            columns
        ].to_string(
            index=False
        )
    )


# ==========================================================
# LOW CONFIDENCE PREDICTIONS
# ==========================================================

low_confidence = df[
    df["confidence"] < 0.50
].copy()


print()
print("=" * 70)
print("LOW CONFIDENCE")
print("=" * 70)

print()

print(
    "Predictions below 50% confidence:",
    len(low_confidence)
)


if len(low_confidence) > 0:

    print()

    print(
        low_confidence[
            [
                "rideId",
                "windowStart",
                "predictedBehavior",
                "confidence",
                "ruleEvents",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ==========================================================
# MODEL DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("MODEL DISTRIBUTION")
print("=" * 70)

print()


for label in LABELS:

    count = int(
        (
            df["modelBehavior"]
            == label
        ).sum()
    )


    print(
        f"{label:20} {count}"
    )


# ==========================================================
# RULE DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("RULE DISTRIBUTION")
print("=" * 70)

print()


for label in LABELS:

    count = int(
        (
            df["ruleBehavior"]
            == label
        ).sum()
    )


    print(
        f"{label:20} {count}"
    )


# ==========================================================
# SUMMARY
# ==========================================================

if len(comparable) > 0:

    overall_match_rate = float(
        comparable[
            "correct"
        ].mean()
    )

else:

    overall_match_rate = 0.0


summary = {

    "predictionWindows":
        int(len(df)),

    "eventWindows":
        int(len(comparable)),

    "matches":
        int(
            comparable[
                "correct"
            ].sum()
        ),

    "mismatches":
        int(len(mismatches)),

    "overallMatchRate":
        overall_match_rate,

    "averageConfidence":
        float(
            df[
                "confidence"
            ].mean()
        ),

    "lowConfidenceWindows":
        int(
            len(low_confidence)
        ),

    "eventStatistics":
        event_statistics,

}


SUMMARY_FILE.write_text(

    json.dumps(
        summary,
        indent=2
    ),

    encoding="utf-8"

)


print()
print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)

print()

print(
    "CSV:"
)

print(
    DIAGNOSTIC_FILE
)

print()

print(
    "JSON:"
)

print(
    SUMMARY_FILE
)

print()

print(
    "Model and original ride data were NOT modified."
)

print()
print("=" * 70)
