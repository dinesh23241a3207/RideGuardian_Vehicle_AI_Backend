from pathlib import Path
import json

import numpy as np


# ==========================================================
# RIDEGUARDIAN
# TRAINING DATA AUGMENTATION
# ==========================================================

BASE_DIR = Path(r"C:\RIDER_SYSTEM\RideGuardian")

INPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4_scaled"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4_augmented"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

# Number of augmented copies for each original
# training sequence.
#
# Original sequence is ALWAYS retained.
AUGMENTATIONS_PER_SEQUENCE = 8

# Reproducible results.
RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ==========================================================
# FEATURE INDICES
# ==========================================================
#
# V4 feature order:
#
# 0 speedKmh
# 1 speedChangeKmh
# 2 speedAccelerationMs2
# 3 accelerationX
# 4 accelerationY
# 5 accelerationZ
# 6 gyroX
# 7 gyroY
# 8 gyroZ
# 9 timeSincePreviousReadingSec
#
# IMPORTANT:
#
# The data is ALREADY STANDARDIZED.
#
# Therefore the noise values below are expressed in
# standardized units, not km/h or m/s².
# ==========================================================


# Maximum Gaussian noise.
NOISE_STD = {

    # Speed
    0: 0.015,

    # Speed change
    1: 0.020,

    # Speed acceleration
    2: 0.020,

    # Accelerometer
    3: 0.015,
    4: 0.015,
    5: 0.015,

    # Gyroscope
    6: 0.015,
    7: 0.015,
    8: 0.015,

    # Sampling interval
    9: 0.005,

}


# Small temporal scaling.
#
# This slightly changes the magnitude of the sequence
# without changing its behaviour label.
MIN_SCALE = 0.98
MAX_SCALE = 1.02


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("TRAINING DATA AUGMENTATION")
print("=" * 70)

print()

train_file = (
    INPUT_DIR
    / "train.npz"
)


if not train_file.exists():

    raise FileNotFoundError(
        f"""
Training dataset not found:

{train_file}
"""
    )


data = np.load(
    train_file
)


X = data["X"]

y = data["y"]

ride_ids = data["ride_ids"]

target_positions = data[
    "target_positions"
]


print(
    "Original shape:",
    X.shape
)

print(
    "Original sequences:",
    len(X)
)


# ==========================================================
# VALIDATE
# ==========================================================

if X.ndim != 3:

    raise ValueError(
        f"Expected X to have 3 dimensions, "
        f"got {X.shape}"
    )


samples, timesteps, features = X.shape


if features != 10:

    raise ValueError(
        f"Expected 10 features, "
        f"got {features}"
    )


# ==========================================================
# AUGMENTATION FUNCTION
# ==========================================================

def augment_sequence(sequence):

    augmented = sequence.copy()


    # ------------------------------------------------------
    # 1. FEATURE-SPECIFIC GAUSSIAN NOISE
    # ------------------------------------------------------

    for feature_index, noise_std in NOISE_STD.items():

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=augmented.shape[0]
        )

        augmented[
            :,
            feature_index
        ] += noise


    # ------------------------------------------------------
    # 2. SMALL GLOBAL MAGNITUDE VARIATION
    # ------------------------------------------------------

    scale = rng.uniform(
        MIN_SCALE,
        MAX_SCALE
    )


    # Apply to dynamic physical measurements.
    #
    # Do not scale timing.

    physical_features = [
        0,  # speed
        1,  # speed change
        2,  # acceleration from speed
        3,  # accel X
        4,  # accel Y
        5,  # accel Z
        6,  # gyro X
        7,  # gyro Y
        8,  # gyro Z
    ]


    augmented[
        :,
        physical_features
    ] *= scale


    # ------------------------------------------------------
    # 3. KEEP FINITE
    # ------------------------------------------------------

    augmented = np.nan_to_num(
        augmented,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return augmented.astype(
        np.float32
    )


# ==========================================================
# CREATE AUGMENTED TRAINING SET
# ==========================================================

print()

print("=" * 70)
print("GENERATING AUGMENTED TRAINING DATA")
print("=" * 70)

print()

augmented_sequences = []

augmented_labels = []

augmented_ride_ids = []

augmented_target_positions = []

augmented_flags = []


# ----------------------------------------------------------
# Keep every original sequence
# ----------------------------------------------------------

for index in range(samples):

    augmented_sequences.append(
        X[index]
    )

    augmented_labels.append(
        y[index]
    )

    augmented_ride_ids.append(
        ride_ids[index]
    )

    augmented_target_positions.append(
        target_positions[index]
    )

    augmented_flags.append(
        0
    )


# ----------------------------------------------------------
# Generate copies
# ----------------------------------------------------------

for index in range(samples):

    for copy_number in range(
        AUGMENTATIONS_PER_SEQUENCE
    ):

        new_sequence = augment_sequence(
            X[index]
        )


        augmented_sequences.append(
            new_sequence
        )

        augmented_labels.append(
            y[index]
        )

        augmented_ride_ids.append(
            ride_ids[index]
        )

        augmented_target_positions.append(
            target_positions[index]
        )

        augmented_flags.append(
            1
        )


# ==========================================================
# STACK
# ==========================================================

X_augmented = np.stack(
    augmented_sequences
).astype(
    np.float32
)


y_augmented = np.array(
    augmented_labels,
    dtype=np.int64
)


ride_ids_augmented = np.array(
    augmented_ride_ids,
    dtype=np.int64
)


target_positions_augmented = np.array(
    augmented_target_positions,
    dtype=np.int64
)


is_augmented = np.array(
    augmented_flags,
    dtype=np.int8
)


# ==========================================================
# PRINT RESULT
# ==========================================================

print(
    "Original sequences:",
    samples
)

print(
    "Augmented copies per original:",
    AUGMENTATIONS_PER_SEQUENCE
)

print(
    "Total sequences:",
    len(X_augmented)
)

print(
    "Final shape:",
    X_augmented.shape
)


# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

LABEL_NAMES = {

    0: "NORMAL",
    1: "HIGH_SPEED",
    2: "HARD_ACCELERATION",
    3: "HARD_BRAKING",
    4: "SUDDEN_TURN",
    5: "MULTI_EVENT",

}


print()

print(
    "Class distribution:"
)

for label_id, label_name in LABEL_NAMES.items():

    count = int(
        (y_augmented == label_id).sum()
    )

    original_count = int(
        (y == label_id).sum()
    )

    print(
        f"  {label_name:20} "
        f"{count:4} "
        f"(original: {original_count})"
    )


# ==========================================================
# ORIGINAL VS AUGMENTED
# ==========================================================

print()

print(
    "Original sequences:",
    int(
        (is_augmented == 0).sum()
    )
)

print(
    "Augmented sequences:",
    int(
        (is_augmented == 1).sum()
    )
)


# ==========================================================
# SAVE
# ==========================================================

output_file = (
    OUTPUT_DIR
    / "train.npz"
)


np.savez_compressed(

    output_file,

    X=X_augmented,

    y=y_augmented,

    ride_ids=ride_ids_augmented,

    target_positions=
        target_positions_augmented,

    is_augmented=is_augmented,

)


print()

print("=" * 70)
print("AUGMENTED DATASET SAVED")
print("=" * 70)

print()

print(
    output_file
)


# ==========================================================
# COPY VALIDATION AND TEST WITHOUT MODIFICATION
# ==========================================================

for split in [
    "val",
    "test",
]:

    source = (
        INPUT_DIR
        / f"{split}.npz"
    )

    destination = (
        OUTPUT_DIR
        / f"{split}.npz"
    )


    if not source.exists():

        print(
            f"Warning: {source} not found"
        )

        continue


    source_data = np.load(
        source
    )


    save_data = {

        "X":
            source_data["X"],

        "y":
            source_data["y"],

        "ride_ids":
            source_data["ride_ids"],

    }


    if "target_positions" in source_data:

        save_data[
            "target_positions"
        ] = source_data[
            "target_positions"
        ]

    elif "event_indices" in source_data:

        save_data[
            "target_positions"
        ] = source_data[
            "event_indices"
        ]


    np.savez_compressed(
        destination,
        **save_data
    )


    print()

    print(
        f"{split.upper()} copied unchanged:"
    )

    print(
        destination
    )


# ==========================================================
# SAVE CONFIG
# ==========================================================

config = {

    "augmentationType":
        "small_gaussian_noise_and_magnitude_scaling",

    "randomSeed":
        RANDOM_SEED,

    "augmentationsPerOriginal":
        AUGMENTATIONS_PER_SEQUENCE,

    "originalTrainingSequences":
        int(samples),

    "finalTrainingSequences":
        int(len(X_augmented)),

    "sequenceLength":
        int(timesteps),

    "featureCount":
        int(features),

    "validationModified":
        False,

    "testModified":
        False,

    "sourceDirectory":
        str(INPUT_DIR),

    "outputDirectory":
        str(OUTPUT_DIR),

}


config_file = (
    OUTPUT_DIR
    / "augmentation_config.json"
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
print("AUGMENTATION COMPLETE")
print("=" * 70)

print()

print(
    "TRAIN:",
    X_augmented.shape
)

print(
    "VAL: unchanged"
)

print(
    "TEST: unchanged"
)

print()

print(
    "Original V4 scaled dataset was NOT modified."
)

print()

print("=" * 70)