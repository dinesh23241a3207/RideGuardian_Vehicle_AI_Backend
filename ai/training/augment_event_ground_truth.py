from pathlib import Path

import numpy as np


# ==========================================================
# RIDEGUARDIAN
# EVENT GROUND-TRUTH AUGMENTATION
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

INPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "event_ground_truth"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "event_ground_truth_augmented"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGET_PER_CLASS = 30

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ==========================================================
# LABELS
# ==========================================================

LABEL_NAMES = {

    0: "NORMAL",

    1: "HIGH_SPEED",

    2: "HARD_ACCELERATION",

    3: "HARD_BRAKING",

    4: "SUDDEN_TURN",

    5: "MULTI_EVENT",

}


# ==========================================================
# AUGMENTATION
# ==========================================================

def augment_sequence(sequence):

    """
    sequence shape:

        (30, 10)

    Feature order:

        0 speedKmh
        1 speedChangeKmh
        2 speedAccelerationMs2
        3 accelerationX
        4 accelerationY
        5 accelerationZ
        6 gyroX
        7 gyroY
        8 gyroZ
        9 timeSincePreviousReadingSec
    """

    augmented = sequence.copy()


    # ------------------------------------------------------
    # Speed
    # ------------------------------------------------------

    augmented[:, 0] += rng.normal(
        0,
        0.25,
        size=len(augmented)
    )


    # ------------------------------------------------------
    # Speed change
    # ------------------------------------------------------

    augmented[:, 1] += rng.normal(
        0,
        0.05,
        size=len(augmented)
    )


    # ------------------------------------------------------
    # Speed acceleration
    # ------------------------------------------------------

    augmented[:, 2] += rng.normal(
        0,
        0.03,
        size=len(augmented)
    )


    # ------------------------------------------------------
    # Accelerometer
    # ------------------------------------------------------

    augmented[:, 3:6] += rng.normal(
        0,
        0.025,
        size=(len(augmented), 3)
    )


    # ------------------------------------------------------
    # Gyroscope
    # ------------------------------------------------------

    augmented[:, 6:9] += rng.normal(
        0,
        0.015,
        size=(len(augmented), 3)
    )


    # ------------------------------------------------------
    # Sampling interval
    # ------------------------------------------------------

    augmented[:, 9] += rng.normal(
        0,
        0.003,
        size=len(augmented)
    )


    # ------------------------------------------------------
    # Physical sanity
    # ------------------------------------------------------

    # Speed cannot be negative.

    augmented[:, 0] = np.maximum(
        augmented[:, 0],
        0
    )


    # Sampling interval must remain positive.

    augmented[:, 9] = np.maximum(
        augmented[:, 9],
        0.1
    )


    return augmented.astype(
        np.float32
    )


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("EVENT GROUND-TRUTH AUGMENTATION")
print("=" * 70)

print()

train_file = (
    INPUT_DIR
    / "train.npz"
)


if not train_file.exists():

    raise FileNotFoundError(
        f"Training dataset not found:\n"
        f"{train_file}"
    )


train = np.load(
    train_file
)


X = train["X"]

y = train["y"]

ride_ids = train[
    "ride_ids"
]

event_indices = train[
    "event_indices"
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
# ORIGINAL DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("ORIGINAL DISTRIBUTION")
print("=" * 70)

print()


for label_id, label_name in LABEL_NAMES.items():

    count = int(
        (y == label_id).sum()
    )

    print(
        f"{label_name:20} {count}"
    )


# ==========================================================
# AUGMENT
# ==========================================================

X_output = []

y_output = []

ride_output = []

event_output = []


print()
print("=" * 70)
print("GENERATING BALANCED TRAINING DATA")
print("=" * 70)

print()


for label_id, label_name in LABEL_NAMES.items():

    indices = np.where(
        y == label_id
    )[0]


    original_count = len(
        indices
    )


    if original_count == 0:

        print(
            f"{label_name:20} "
            f"0 original → 0"
        )

        continue


    # ------------------------------------------------------
    # Always keep original sequences
    # ------------------------------------------------------

    for index in indices:

        X_output.append(
            X[index]
        )

        y_output.append(
            y[index]
        )

        ride_output.append(
            ride_ids[index]
        )

        event_output.append(
            event_indices[index]
        )


    # ------------------------------------------------------
    # Add augmented copies
    # ------------------------------------------------------

    required = max(
        0,
        TARGET_PER_CLASS
        - original_count
    )


    for i in range(required):

        source_index = indices[
            i % original_count
        ]


        augmented = augment_sequence(
            X[source_index]
        )


        X_output.append(
            augmented
        )

        y_output.append(
            label_id
        )

        ride_output.append(
            ride_ids[source_index]
        )

        event_output.append(
            event_indices[source_index]
        )


    print(
        f"{label_name:20} "
        f"{original_count} original → "
        f"{max(original_count, TARGET_PER_CLASS)}"
    )


# ==========================================================
# CONVERT
# ==========================================================

X_final = np.stack(
    X_output
).astype(
    np.float32
)


y_final = np.array(
    y_output,
    dtype=np.int64
)


ride_final = np.array(
    ride_output,
    dtype=np.int64
)


event_final = np.array(
    event_output,
    dtype=np.int64
)


# ==========================================================
# SHUFFLE
# ==========================================================

indices = np.arange(
    len(X_final)
)


rng.shuffle(
    indices
)


X_final = X_final[
    indices
]

y_final = y_final[
    indices
]

ride_final = ride_final[
    indices
]

event_final = event_final[
    indices
]


# ==========================================================
# SAVE TRAIN
# ==========================================================

output_train = (
    OUTPUT_DIR
    / "train.npz"
)


np.savez_compressed(

    output_train,

    X=X_final,

    y=y_final,

    ride_ids=ride_final,

    event_indices=event_final,

)


# ==========================================================
# COPY VAL / TEST UNCHANGED
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


    data = np.load(
        source
    )


    np.savez_compressed(

        destination,

        **{
            key: data[key]

            for key in data.files
        }

    )


# ==========================================================
# FINAL DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("FINAL TRAIN DISTRIBUTION")
print("=" * 70)

print()


for label_id, label_name in LABEL_NAMES.items():

    count = int(
        (y_final == label_id).sum()
    )

    print(
        f"{label_name:20} {count}"
    )


# ==========================================================
# OUTPUT
# ==========================================================

print()
print("=" * 70)
print("AUGMENTATION COMPLETE")
print("=" * 70)

print()

print(
    "Final training shape:",
    X_final.shape
)

print()

print(
    "Train:"
)

print(
    OUTPUT_DIR
    / "train.npz"
)

print()

print(
    "Validation copied unchanged:"
)

print(
    OUTPUT_DIR
    / "val.npz"
)

print()

print(
    "Test copied unchanged:"
)

print(
    OUTPUT_DIR
    / "test.npz"
)

print()

print(
    "Original ground-truth dataset was NOT modified."
)

print("=" * 70)