import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split


# ============================================================
# RIDEGUARDIAN
# ENGINE FAULT AI
#
# Dataset:
#   EngineFaultDB_Final.csv
#
# Target:
#   Fault
#
# Features:
#   AFR
#   CO
#   CO2
#   Consumption L/100KM
#   Consumption L/H
#   Force
#   HC
#   Lambda
#   MAP
#   O2
#   Power
#   RPM
#   Speed
#   TPS
#
# Model:
#   Random Forest
#
# NOTE:
#   This is NOT OBD integration.
#   The model learns from the supplied engine dataset.
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    AI_DIR
    / "datasets"
    / "EngineFaultDB_Final.csv"
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "engine_fault"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
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

RESULTS_PATH = (
    MODEL_DIR
    / "test_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.15

N_ESTIMATORS = 300

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

N_JOBS = -1


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("RIDEGUARDIAN ENGINE FAULT AI")
print("=" * 75)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n"
        f"{DATASET_PATH}"
    )


print("Dataset:")
print(DATASET_PATH)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading EngineFault dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
)


# ============================================================
# TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Fault"

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        f"was not found."
    )


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    "AFR",

    "CO",

    "CO2",

    "Consumption L/100KM",

    "Consumption L/H",

    "Force",

    "HC",

    "Lambda",

    "MAP",

    "O2",

    "Power",

    "RPM",

    "Speed",

    "TPS"
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [

    feature

    for feature in FEATURE_COLUMNS

    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        +
        "\n".join(
            missing_features
        )
    )


print()
print("Features:")

for feature in FEATURE_COLUMNS:

    print(
        f"  {feature}"
    )


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

df = df[
    FEATURE_COLUMNS
    +
    [TARGET_COLUMN]
].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

before_rows = len(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()

after_rows = len(df)

removed_rows = (
    before_rows
    -
    after_rows
)

print()
print(
    "Rows removed because of "
    "missing/invalid values: "
    f"{removed_rows}"
)


# ============================================================
# CONVERT TARGET TO INTEGER
# ============================================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(int)
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

print()

print(
    f"Usable samples: {len(df)}"
)

print(
    f"Features: {len(FEATURE_COLUMNS)}"
)


# ============================================================
# FAULT DISTRIBUTION
# ============================================================

print()
print("Fault distribution:")

fault_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

for fault, count in fault_counts.items():

    print(
        f"  Fault {int(fault)}: {int(count)}"
    )


# ============================================================
# X AND Y
# ============================================================

X = df[
    FEATURE_COLUMNS
]

y = df[
    TARGET_COLUMN
]


# ============================================================
# DATASET SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(

        X,

        y,

        test_size=(
            TEST_SIZE
            +
            VALIDATION_SIZE
        ),

        random_state=RANDOM_STATE,

        stratify=y
    )
)


relative_test_size = (
    TEST_SIZE
    /
    (
        TEST_SIZE
        +
        VALIDATION_SIZE
    )
)


X_val, X_test, y_val, y_test = (
    train_test_split(

        X_temp,

        y_temp,

        test_size=relative_test_size,

        random_state=RANDOM_STATE,

        stratify=y_temp
    )
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

print()
print("=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print()

print(
    f"Training:   {len(X_train)}"
)

print(
    f"Validation: {len(X_val)}"
)

print(
    f"Testing:    {len(X_test)}"
)

print(
    f"Total:      "
    f"{len(X_train) + len(X_val) + len(X_test)}"
)


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

print()
print("=" * 75)
print("CREATING RANDOM FOREST")
print("=" * 75)

print()

print(
    f"Trees: {N_ESTIMATORS}"
)

print(
    "Training..."
)


model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    max_features="sqrt"
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 75)
print("STARTING ENGINE FAULT TRAINING")
print("=" * 75)

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = (
    time.time()
    -
    start_time
)

print()

print(
    f"Training completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)

val_predictions = (
    model.predict(
        X_val
    )
)


val_accuracy = (
    accuracy_score(
        y_val,
        val_predictions
    )
)


val_precision, val_recall, val_f1, _ = (
    precision_recall_fscore_support(

        y_val,

        val_predictions,

        average="weighted",

        zero_division=0
    )
)


print()

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Precision: "
    f"{val_precision * 100:.2f}%"
)

print(
    f"Validation Recall: "
    f"{val_recall * 100:.2f}%"
)

print(
    f"Validation F1: "
    f"{val_f1 * 100:.2f}%"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 75)
print("FINAL ENGINE FAULT TEST")
print("=" * 75)


test_predictions = (
    model.predict(
        X_test
    )
)


test_probabilities = (
    model.predict_proba(
        X_test
    )
)


test_accuracy = (
    accuracy_score(
        y_test,
        test_predictions
    )
)


test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(

        y_test,

        test_predictions,

        average="weighted",

        zero_division=0
    )
)


print()

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Test Recall: "
    f"{test_recall * 100:.2f}%"
)

print(
    f"Test F1: "
    f"{test_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

class_labels = sorted(
    y.unique()
)

class_labels = [
    int(label)
    for label in class_labels
]

class_names = [
    f"Fault {label}"
    for label in class_labels
]


report = classification_report(

    y_test,

    test_predictions,

    labels=class_labels,

    target_names=class_names,

    zero_division=0
)


print()

print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

matrix = confusion_matrix(

    y_test,

    test_predictions,

    labels=class_labels
)


print()

print(
    matrix
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 75)
print("FEATURE IMPORTANCE")
print("=" * 75)

print()


feature_importance = (
    pd.DataFrame({

        "Feature":
            FEATURE_COLUMNS,

        "Importance":
            model.feature_importances_

    })
    .sort_values(
        "Importance",
        ascending=False
    )
)


for _, row in feature_importance.iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 75)
print("SAVING ENGINE FAULT MODEL")
print("=" * 75)

joblib.dump(

    model,

    MODEL_PATH
)


print()

print(
    "Model:"
)

print(
    MODEL_PATH
)


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(

        FEATURE_COLUMNS,

        file,

        indent=4
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(

        class_names,

        file,

        indent=4
    )


# ============================================================
# SAVE TEST RESULTS
#
# IMPORTANT:
# Convert NumPy values to normal Python values.
# This prevents:
#
# TypeError:
# Object of type int64 is not JSON serializable
# ============================================================

results = {

    "samples":
        int(len(df)),

    "training_samples":
        int(len(X_train)),

    "validation_samples":
        int(len(X_val)),

    "test_samples":
        int(len(X_test)),

    "features":
        FEATURE_COLUMNS,

    "classes":
        [int(x) for x in class_labels],

    "validation_accuracy":
        float(val_accuracy),

    "validation_precision":
        float(val_precision),

    "validation_recall":
        float(val_recall),

    "validation_f1":
        float(val_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_precision":
        float(test_precision),

    "test_recall":
        float(test_recall),

    "test_f1":
        float(test_f1),

    "confusion_matrix":
        matrix.astype(int).tolist(),

    "training_time_seconds":
        float(training_time),

    "feature_importance": {

        str(row["Feature"]):
            float(row["Importance"])

        for _, row
        in feature_importance.iterrows()

    }

}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(

        results,

        file,

        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("ENGINE FAULT AI TRAINING COMPLETE")
print("=" * 75)

print()

print("Saved files:")

print()

print(
    f"Model:\n{MODEL_PATH}"
)

print()

print(
    f"Features:\n{FEATURES_PATH}"
)

print()

print(
    f"Classes:\n{CLASS_NAMES_PATH}"
)

print()

print(
    f"Results:\n{RESULTS_PATH}"
)

print()

print(
    "Engine Fault AI is ready."
)

print("=" * 75)