from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# ==========================================================
# RIDEGUARDIAN
# EVENT GROUND-TRUTH FEATURE NORMALIZATION
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

INPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "event_ground_truth_augmented"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "event_ground_truth_scaled"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("EVENT GROUND-TRUTH FEATURE NORMALIZATION")
print("=" * 70)

print()

train_file = INPUT_DIR / "train.npz"

if not train_file.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{train_file}"
    )


train = np.load(train_file)

X_train = train["X"]
y_train = train["y"]
ride_train = train["ride_ids"]
event_train = train["event_indices"]


print(
    "TRAIN:",
    X_train.shape
)


# ==========================================================
# LOAD VAL / TEST
# ==========================================================

datasets = {
    "train": (X_train, y_train, ride_train, event_train)
}


for split in ["val", "test"]:

    file = INPUT_DIR / f"{split}.npz"

    if not file.exists():
        raise FileNotFoundError(
            f"{split} file not found:\n{file}"
        )

    data = np.load(file)

    datasets[split] = (
        data["X"],
        data["y"],
        data["ride_ids"],
        data["event_indices"],
    )

    print(
        f"{split.upper()}:",
        data["X"].shape
    )


# ==========================================================
# SHAPE
# ==========================================================

sequence_length = X_train.shape[1]

feature_count = X_train.shape[2]


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
# FIT SCALER
# ==========================================================

print()
print("=" * 70)
print("FITTING TRAINING SCALER")
print("=" * 70)

print()

print(
    "IMPORTANT:"
)

print(
    "Scaler is fitted ONLY on training data."
)


scaler = StandardScaler()


train_flat = X_train.reshape(
    -1,
    feature_count
)


scaler.fit(
    train_flat
)


print()
print(
    "Scaler fitted."
)


# ==========================================================
# STATISTICS
# ==========================================================

print()
print(
    "Training feature statistics:"
)

print()


for i in range(feature_count):

    mean = scaler.mean_[i]

    std = scaler.scale_[i]

    print(
        f"Feature {i}: "
        f"mean={mean:.6f}, "
        f"std={std:.6f}"
    )


# ==========================================================
# TRANSFORM
# ==========================================================

print()
print("=" * 70)
print("TRANSFORMING DATA")
print("=" * 70)


scaled_datasets = {}


for split, (
    X,
    y,
    ride_ids,
    event_indices
) in datasets.items():

    print()

    print(
        split.upper()
    )

    print(
        "Original:",
        X.shape
    )


    X_flat = X.reshape(
        -1,
        feature_count
    )


    X_scaled_flat = scaler.transform(
        X_flat
    )


    X_scaled = X_scaled_flat.reshape(
        X.shape
    ).astype(
        np.float32
    )


    scaled_datasets[split] = (
        X_scaled,
        y,
        ride_ids,
        event_indices
    )


    print(
        "Scaled:",
        X_scaled.shape
    )

    print(
        "Mean:",
        round(
            float(X_scaled.mean()),
            6
        )
    )

    print(
        "Std:",
        round(
            float(X_scaled.std()),
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


for split, (
    X_scaled,
    y,
    ride_ids,
    event_indices
) in scaled_datasets.items():

    output_file = (
        OUTPUT_DIR
        / f"{split}.npz"
    )


    np.savez_compressed(

        output_file,

        X=X_scaled,

        y=y,

        ride_ids=ride_ids,

        event_indices=event_indices,

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
    / "event_behavior_scaler.joblib"
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
    "Training scaler was fitted ONLY on TRAIN."
)

print(
    "Original augmented dataset was NOT modified."
)

print("=" * 70)