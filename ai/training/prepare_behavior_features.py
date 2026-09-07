from pathlib import Path
import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================================================
# RIDEGUARDIAN
# BEHAVIOUR FEATURE NORMALIZATION
# ==========================================================

BASE_DIR = Path(r"C:\RIDER_SYSTEM\RideGuardian")

INPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4_scaled"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

SPLITS = [
    "train",
    "val",
    "test",
]


# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("BEHAVIOUR FEATURE NORMALIZATION")
print("=" * 70)

print()

print("Input:")
print(INPUT_DIR)

print()


datasets = {}


for split in SPLITS:

    file = (
        INPUT_DIR
        / f"{split}.npz"
    )


    if not file.exists():

        raise FileNotFoundError(
            f"Missing dataset:\n{file}"
        )


    data = np.load(
        file
    )


    X = data["X"]

    y = data["y"]

    ride_ids = data["ride_ids"]


    if "target_positions" in data:

        target_positions = (
            data["target_positions"]
        )

    elif "event_indices" in data:

        target_positions = (
            data["event_indices"]
        )

    else:

        target_positions = np.full(
            len(X),
            -1,
            dtype=np.int64
        )


    datasets[split] = {

        "X": X,

        "y": y,

        "ride_ids": ride_ids,

        "target_positions":
            target_positions,

    }


    print(
        f"{split.upper()}:",
        X.shape
    )


# ==========================================================
# VALIDATE SHAPES
# ==========================================================

train_X = datasets[
    "train"
]["X"]


if train_X.ndim != 3:

    raise ValueError(
        f"Expected 3D data, got {train_X.shape}"
    )


sequence_length = train_X.shape[1]

feature_count = train_X.shape[2]


print()

print(
    "Sequence length:",
    sequence_length
)

print(
    "Feature count:",
    feature_count
)


# ==========================================================
# FIT SCALER ON TRAIN ONLY
# ==========================================================

print()

print("=" * 70)
print("FITTING SCALER")
print("=" * 70)

print()

print(
    "IMPORTANT:"
)

print(
    "Scaler is fitted ONLY on training data."
)

print()


# Convert:

# (samples, timesteps, features)

# into:

# (samples * timesteps, features)

train_flat = train_X.reshape(
    -1,
    feature_count
)


scaler = StandardScaler()


scaler.fit(
    train_flat
)


print(
    "Scaler fitted."
)


# ==========================================================
# DISPLAY STATISTICS
# ==========================================================

print()

print(
    "Training feature statistics:"
)

print()

for index in range(
    feature_count
):

    print(
        f"Feature {index}: "
        f"mean={scaler.mean_[index]:.6f}, "
        f"std={scaler.scale_[index]:.6f}"
    )


# ==========================================================
# TRANSFORM DATASETS
# ==========================================================

print()

print("=" * 70)
print("TRANSFORMING DATA")
print("=" * 70)


for split in SPLITS:

    X = datasets[
        split
    ]["X"]


    original_shape = X.shape


    flat = X.reshape(
        -1,
        feature_count
    )


    scaled_flat = scaler.transform(
        flat
    )


    scaled = scaled_flat.reshape(
        original_shape
    ).astype(
        np.float32
    )


    datasets[
        split
    ]["X_scaled"] = scaled


    print()

    print(
        f"{split.upper()}:"
    )

    print(
        "Original:",
        original_shape
    )

    print(
        "Scaled:",
        scaled.shape
    )

    print(
        "Mean:",
        round(
            float(scaled.mean()),
            6
        )
    )

    print(
        "Std:",
        round(
            float(scaled.std()),
            6
        )
    )


# ==========================================================
# SAVE
# ==========================================================

print()

print("=" * 70)
print("SAVING SCALED DATA")
print("=" * 70)


for split in SPLITS:

    data = datasets[
        split
    ]


    output_file = (
        OUTPUT_DIR
        / f"{split}.npz"
    )


    np.savez_compressed(

        output_file,

        X=data["X_scaled"],

        y=data["y"],

        ride_ids=data["ride_ids"],

        target_positions=
            data["target_positions"],

    )


    print()

    print(
        split.upper(),
        "saved:"
    )

    print(
        output_file
    )


# ==========================================================
# SAVE SCALER
# ==========================================================

scaler_file = (
    OUTPUT_DIR
    / "behavior_scaler.joblib"
)


joblib.dump(
    scaler,
    scaler_file
)


print()

print(
    "Scaler saved:"
)

print(
    scaler_file
)


# ==========================================================
# SAVE CONFIG
# ==========================================================

config = {

    "sequenceLength":
        sequence_length,

    "featureCount":
        feature_count,

    "scaler":
        "StandardScaler",

    "fitOn":
        "train_only",

    "inputDirectory":
        str(INPUT_DIR),

    "outputDirectory":
        str(OUTPUT_DIR),

}


config_file = (
    OUTPUT_DIR
    / "scaling_config.json"
)


config_file.write_text(

    json.dumps(
        config,
        indent=2
    ),

    encoding="utf-8"

)


# ==========================================================
# COMPLETE
# ==========================================================

print()

print("=" * 70)
print("NORMALIZATION COMPLETE")
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
    "Scaler:"
)

print(
    scaler_file
)

print()

print(
    "Train / Val / Test were transformed using"
)

print(
    "the TRAINING scaler only."
)

print()

print(
    "Original V4 dataset was NOT modified."
)

print()

print("=" * 70)