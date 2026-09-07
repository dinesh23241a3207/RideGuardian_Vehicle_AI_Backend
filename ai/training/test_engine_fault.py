import json
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# RIDEGUARDIAN
# ENGINE FAULT AI - TEST / INFERENCE
#
# Loads the trained Random Forest model and predicts
# the engine fault from 14 engine parameters.
#
# IMPORTANT:
# Fault 0/1/2/3 are kept exactly as the dataset labels.
# We do NOT invent mechanical meanings for them.
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "engine_fault"
)


MODEL_PATH = (
    MODEL_DIR
    / "engine_fault_random_forest.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "feature_names.json"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("RIDEGUARDIAN ENGINE FAULT AI")
print("ENGINE DIAGNOSIS TEST")
print("=" * 75)

print()


# ============================================================
# CHECK FILES
# ============================================================

required_files = [

    MODEL_PATH,

    FEATURES_PATH,

    CLASS_NAMES_PATH

]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading EngineFault model..."
)

model = joblib.load(
    MODEL_PATH
)

print(
    "EngineFault model loaded successfully."
)


# ============================================================
# LOAD FEATURES
# ============================================================

with open(
    FEATURES_PATH,
    "r",
    encoding="utf-8"
) as file:

    feature_names = json.load(
        file
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(
        file
    )


print()

print(
    "Engine fault classes:"
)

for index, name in enumerate(
    class_names
):

    print(
        f"  {index}: {name}"
    )


# ============================================================
# DISPLAY EXPECTED INPUTS
# ============================================================

print()
print("=" * 75)
print("ENGINE SENSOR INPUT")
print("=" * 75)

print()

print(
    "Enter the following engine values."
)

print(
    "Use the same units/format as the training dataset."
)

print()


# ============================================================
# COLLECT INPUT
# ============================================================

input_values = {}


for feature in feature_names:

    while True:

        try:

            value = float(
                input(
                    f"{feature}: "
                )
            )

            input_values[
                feature
            ] = value

            break

        except ValueError:

            print(
                "Please enter a numeric value."
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_data = pd.DataFrame(

    [
        input_values
    ],

    columns=feature_names
)


# ============================================================
# RUN PREDICTION
# ============================================================

print()
print(
    "=" * 75
)

print(
    "RUNNING ENGINE FAULT PREDICTION..."
)

print(
    "=" * 75
)


prediction = model.predict(
    input_data
)


probabilities = model.predict_proba(
    input_data
)


predicted_fault = (
    prediction[0]
)


# ============================================================
# FIND PREDICTED CLASS INDEX
# ============================================================

model_classes = list(
    model.classes_
)


predicted_index = (
    model_classes.index(
        predicted_fault
    )
)


predicted_probability = (
    probabilities[0][
        predicted_index
    ]
)


# ============================================================
# PREDICTED CLASS NAME
# ============================================================

if predicted_index < len(
    class_names
):

    predicted_class_name = (
        class_names[
            predicted_index
        ]
    )

else:

    predicted_class_name = (
        f"Fault {predicted_fault}"
    )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 75)
print("ENGINE DIAGNOSIS")
print("=" * 75)

print()

print(
    "PREDICTED FAULT:"
)

print(
    f"  {predicted_class_name}"
)

print(
    f"  Confidence: "
    f"{predicted_probability * 100:.2f}%"
)


# ============================================================
# CLASS PROBABILITIES
# ============================================================

print()
print(
    "CLASS PROBABILITIES:"
)

for i, model_class in enumerate(
    model_classes
):

    if i < len(class_names):

        name = class_names[i]

    else:

        name = (
            f"Fault {model_class}"
        )

    probability = (
        probabilities[0][i]
    )

    print(
        f"  {name}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# INPUT SUMMARY
# ============================================================

print()
print(
    "INPUT VALUES:"
)

for feature in feature_names:

    print(
        f"  {feature}: "
        f"{input_values[feature]}"
    )


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

if predicted_probability >= 0.90:

    confidence_level = (
        "VERY HIGH"
    )

elif predicted_probability >= 0.75:

    confidence_level = (
        "HIGH"
    )

elif predicted_probability >= 0.60:

    confidence_level = (
        "MODERATE"
    )

else:

    confidence_level = (
        "LOW"
    )


print()
print(
    f"CONFIDENCE LEVEL: "
    f"{confidence_level}"
)


# ============================================================
# RECOMMENDATION
#
# We intentionally DO NOT assign a mechanical fault name
# to Fault 0/1/2/3 because the dataset information currently
# available only identifies them by numeric class.
# ============================================================

print()
print(
    "RECOMMENDATION:"
)

if predicted_probability >= 0.90:

    print(
        "  The engine data strongly matches "
        f"{predicted_class_name}."
    )

    print(
        "  Further inspection is recommended "
        "to determine the underlying mechanical cause."
    )

elif predicted_probability >= 0.75:

    print(
        "  The engine data moderately/strongly "
        f"matches {predicted_class_name}."
    )

    print(
        "  Additional engine measurements and "
        "inspection are recommended."
    )

else:

    print(
        "  The prediction confidence is low."
    )

    print(
        "  Collect additional/cleaner engine data "
        "before making a maintenance decision."
    )


# ============================================================
# SAFETY NOTE
# ============================================================

print()
print(
    "NOTE:"
)

print(
    "  This AI prediction is a decision-support result."
)

print(
    "  It does not replace professional mechanical inspection."
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("ENGINE FAULT AI TEST COMPLETE")
print("=" * 75)