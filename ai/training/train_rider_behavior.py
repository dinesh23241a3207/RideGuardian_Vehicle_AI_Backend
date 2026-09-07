from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# ==========================================================
# RIDEGUARDIAN - RIDER BEHAVIOUR BASELINE ML
# ==========================================================

BASE_DIR = Path(r"C:\RIDER_SYSTEM\RideGuardian")

DATA_FILE = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "processed"
    / "ride_features_with_split.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "ai"
    / "models"
    / "rider_behavior"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("RIDEGUARDIAN RIDER BEHAVIOUR ML")
print("=" * 70)


# ==========================================================
# LOAD DATA
# ==========================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n\n"
        "Copy ride_features_with_split.csv into:\n"
        "ai\\datasets\\rides\\processed\\"
    )


df = pd.read_csv(DATA_FILE)


print()
print("Dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "rideId",
    "split",
    "safetyScore",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ==========================================================
# DISPLAY DATASET
# ==========================================================

print()
print("Split distribution:")
print(
    df["split"].value_counts()
)


print()
print("Safety score:")
print(
    df["safetyScore"].describe()
)


# ==========================================================
# FEATURES
# ==========================================================

FEATURES = [

    "distanceKm",

    "durationSec",

    "averageSpeedKmh",

    "maximumSpeedKmh",

    "speedStdKmh",

    "meanAccelerationMagnitude",

    "maxAccelerationMagnitude",

    "meanGyroMagnitudeRadS",

    "maxGyroMagnitudeRadS",

    "meanTimeBetweenReadingsSec",

    "maxTimeBetweenReadingsSec",

    "speedEvents",

    "hardAccelerationEvents",

    "hardBrakingEvents",

    "suddenTurnEvents",

    "totalEvents",

    "sensorReadings",

    "hardBrakingRatePer100Readings",

    "hardAccelerationRatePer100Readings",

    "suddenTurnRatePer100Readings",

]


# Only use features that actually exist.
available_features = [
    feature
    for feature in FEATURES
    if feature in df.columns
]


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


print()
print("Features used:")
for feature in available_features:
    print("  ", feature)


if missing_features:

    print()
    print("WARNING - missing features:")
    for feature in missing_features:
        print("  ", feature)


if len(available_features) < 5:

    raise ValueError(
        "Too few usable ML features."
    )


# ==========================================================
# REMOVE INVALID TARGET ROWS
# ==========================================================

df = df[
    df["safetyScore"].notna()
].copy()


# ==========================================================
# TRAIN / VALIDATION / TEST
# ==========================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Train:",
    len(train_df),
    "rides"
)

print(
    "Validation:",
    len(val_df),
    "rides"
)

print(
    "Test:",
    len(test_df),
    "rides"
)


# ==========================================================
# SAFETY CHECK
# ==========================================================

if len(train_df) < 5:

    print()
    print(
        "WARNING:"
    )

    print(
        "There are very few training rides."
    )

    print(
        "This model is currently a pipeline/baseline test."
    )

    print(
        "Do NOT treat the resulting accuracy as final ML performance."
    )


# ==========================================================
# INPUT / TARGET
# ==========================================================

X_train = train_df[
    available_features
]

y_train = train_df[
    "safetyScore"
]


X_val = val_df[
    available_features
]

y_val = val_df[
    "safetyScore"
]


X_test = test_df[
    available_features
]

y_test = test_df[
    "safetyScore"
]


# ==========================================================
# MODEL
# ==========================================================

model = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "model",
            RandomForestRegressor(

                n_estimators=300,

                max_depth=8,

                min_samples_leaf=2,

                random_state=42,

                n_jobs=-1,
            ),
        ),
    ]
)


# ==========================================================
# TRAIN
# ==========================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


print()
print("Training complete.")


# ==========================================================
# VALIDATION
# ==========================================================

if len(val_df) > 0:

    val_predictions = model.predict(
        X_val
    )

    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    val_rmse = mean_squared_error(
        y_val,
        val_predictions
    ) ** 0.5

    val_r2 = r2_score(
        y_val,
        val_predictions
    )


    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"MAE : {val_mae:.3f}"
    )

    print(
        f"RMSE: {val_rmse:.3f}"
    )

    print(
        f"R²  : {val_r2:.3f}"
    )


# ==========================================================
# TEST
# ==========================================================

if len(test_df) > 0:

    test_predictions = model.predict(
        X_test
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_rmse = mean_squared_error(
        y_test,
        test_predictions
    ) ** 0.5

    test_r2 = r2_score(
        y_test,
        test_predictions
    )


    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"MAE : {test_mae:.3f}"
    )

    print(
        f"RMSE: {test_rmse:.3f}"
    )

    print(
        f"R²  : {test_r2:.3f}"
    )


    # ----------------------------------------------
    # TEST PREDICTIONS
    # ----------------------------------------------

    predictions_df = test_df[
        [
            "rideId",
            "safetyScore",
        ]
    ].copy()


    predictions_df[
        "predictedSafetyScore"
    ] = test_predictions


    predictions_df[
        "absoluteError"
    ] = abs(
        predictions_df["safetyScore"]
        -
        predictions_df["predictedSafetyScore"]
    )


    predictions_path = (
        MODEL_DIR
        / "test_predictions.csv"
    )


    predictions_df.to_csv(
        predictions_path,
        index=False
    )


    print()
    print(
        "Test predictions saved:"
    )

    print(
        predictions_path
    )


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

rf = model.named_steps[
    "model"
]


importance_df = pd.DataFrame({

    "feature":
        available_features,

    "importance":
        rf.feature_importances_,

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


importance_path = (
    MODEL_DIR
    / "feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)


print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

model_path = (
    MODEL_DIR
    / "rider_behavior_baseline.joblib"
)


joblib.dump(
    model,
    model_path
)


# ==========================================================
# SAVE FEATURE LIST
# ==========================================================

feature_list_path = (
    MODEL_DIR
    / "feature_list.txt"
)


feature_list_path.write_text(
    "\n".join(
        available_features
    ),
    encoding="utf-8",
)


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 70)
print("BASELINE ML COMPLETE")
print("=" * 70)

print()

print(
    "Model:"
)

print(
    model_path
)

print()

print(
    "Feature importance:"
)

print(
    importance_path
)

print()

print(
    "Feature list:"
)

print(
    feature_list_path
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is a BASELINE model."
)

print(
    "Only 11 rides are currently available."
)

print(
    "Collect more real rides before claiming final ML accuracy."
)

print()

print("=" * 70)