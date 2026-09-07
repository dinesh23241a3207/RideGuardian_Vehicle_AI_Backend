from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam


# ==========================================================
# RIDEGUARDIAN
# RIDER BEHAVIOUR GRU CLASSIFIER
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

DATA_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "behavior_events_v4_augmented"
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


# ==========================================================
# SETTINGS
# ==========================================================

SEED = 42

EPOCHS = 100

BATCH_SIZE = 16

LEARNING_RATE = 0.001

PATIENCE = 15


# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


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


NUM_CLASSES = len(
    LABEL_NAMES
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("RIDER BEHAVIOUR GRU CLASSIFIER")
print("=" * 70)

print()

print("Dataset:")
print(DATA_DIR)

print()


def load_dataset(split):

    file = (
        DATA_DIR
        / f"{split}.npz"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file}"
        )

    data = np.load(
        file
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.int64
    )

    return X, y


X_train, y_train = load_dataset(
    "train"
)

X_val, y_val = load_dataset(
    "val"
)

X_test, y_test = load_dataset(
    "test"
)


print(
    "TRAIN:",
    X_train.shape,
    y_train.shape
)

print(
    "VAL:",
    X_val.shape,
    y_val.shape
)

print(
    "TEST:",
    X_test.shape,
    y_test.shape
)


# ==========================================================
# VALIDATE SHAPES
# ==========================================================

if X_train.ndim != 3:

    raise ValueError(
        f"Expected 3D training data, "
        f"got {X_train.shape}"
    )


sequence_length = X_train.shape[1]

feature_count = X_train.shape[2]


if sequence_length != 30:

    raise ValueError(
        f"Expected sequence length 30, "
        f"got {sequence_length}"
    )


if feature_count != 10:

    raise ValueError(
        f"Expected 10 features, "
        f"got {feature_count}"
    )


print()

print(
    "Sequence length:",
    sequence_length
)

print(
    "Features:",
    feature_count
)

print(
    "Classes:",
    NUM_CLASSES
)


# ==========================================================
# DATA DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("DATA DISTRIBUTION")
print("=" * 70)

print()


def print_distribution(
    name,
    y
):

    print(
        name
    )

    for class_id, class_name in enumerate(
        LABEL_NAMES
    ):

        count = int(
            (y == class_id).sum()
        )

        print(
            f"  {class_id}: "
            f"{class_name:20} "
            f"{count}"
        )

    print()


print_distribution(
    "TRAIN",
    y_train
)

print_distribution(
    "VALIDATION",
    y_val
)

print_distribution(
    "TEST",
    y_test
)


# ==========================================================
# REMOVE COMPLETELY ABSENT TRAINING CLASSES
# ==========================================================
#
# MULTI_EVENT currently has zero training samples.
#
# We cannot train a classifier to recognize a class that
# has no training examples.
#
# Therefore the actual output classes are determined from
# the training data.
# ==========================================================

present_classes = np.unique(
    y_train
)


present_classes = sorted(
    int(x)
    for x in present_classes
)


print(
    "Classes present in training:"
)

for class_id in present_classes:

    print(
        f"  {class_id}: "
        f"{LABEL_NAMES[class_id]}"
    )


missing_training_classes = [
    class_id
    for class_id in range(NUM_CLASSES)
    if class_id not in present_classes
]


if missing_training_classes:

    print()

    print(
        "Classes without training examples:"
    )

    for class_id in missing_training_classes:

        print(
            f"  {class_id}: "
            f"{LABEL_NAMES[class_id]}"
        )

    print()

    print(
        "These classes will remain part of the label"
    )

    print(
        "definition but will not be trainable yet."
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print()


class_weights = {}


weights = compute_class_weight(

    class_weight="balanced",

    classes=np.array(
        present_classes
    ),

    y=y_train

)


for class_id, weight in zip(
    present_classes,
    weights
):

    # Limit extreme weights because the current
    # dataset is extremely small.

    weight = float(
        np.clip(
            weight,
            0.5,
            5.0
        )
    )


    class_weights[
        class_id
    ] = weight


    print(
        f"{LABEL_NAMES[class_id]:20} "
        f"{weight:.4f}"
    )


# ==========================================================
# MODEL
# ==========================================================

print()
print("=" * 70)
print("BUILDING GRU MODEL")
print("=" * 70)

print()


model = Sequential([

    Input(
        shape=(
            sequence_length,
            feature_count
        )
    ),

    GRU(
        64,
        return_sequences=True,
        dropout=0.20,
        recurrent_dropout=0.0,
    ),

    BatchNormalization(),

    GRU(
        32,
        return_sequences=False,
        dropout=0.20,
        recurrent_dropout=0.0,
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    ),

])


model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],

)


model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

model_file = (
    MODEL_DIR
    / "behavior_gru.keras"
)


callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True,

        verbose=1,

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1,

    ),

    ModelCheckpoint(

        filepath=str(
            model_file
        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1,

    ),

]


# ==========================================================
# TRAIN
# ==========================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print()


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1,

)


# ==========================================================
# SAVE FINAL BEST MODEL
# ==========================================================

model.save(
    model_file
)


print()

print(
    "Model saved:"
)

print(
    model_file
)


# ==========================================================
# TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(
    history.history
)


history_file = (
    MODEL_DIR
    / "behavior_training_history.csv"
)


history_df.to_csv(
    history_file,
    index=False
)


print()

print(
    "Training history saved:"
)

print(
    history_file
)


# ==========================================================
# PREDICTIONS
# ==========================================================

print()
print("=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

print()


test_probabilities = model.predict(
    X_test,
    verbose=0
)


test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    f"Test accuracy: "
    f"{accuracy:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print()
print(
    "Classification report:"
)

report = classification_report(

    y_test,

    test_predictions,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=LABEL_NAMES,

    zero_division=0,

)


print(
    report
)


# ==========================================================
# PRECISION / RECALL / F1
# ==========================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(

        y_test,

        test_predictions,

        labels=list(
            range(NUM_CLASSES)
        ),

        zero_division=0,

    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    test_predictions,

    labels=list(
        range(NUM_CLASSES)
    )

)


print()
print(
    "Confusion matrix:"
)

print(
    cm
)


# ==========================================================
# TEST PREDICTIONS CSV
# ==========================================================

prediction_rows = []


for index in range(
    len(y_test)
):

    actual_id = int(
        y_test[index]
    )

    predicted_id = int(
        test_predictions[index]
    )


    row = {

        "sampleIndex":
            index,

        "actualLabel":
            LABEL_NAMES[
                actual_id
            ],

        "actualLabelId":
            actual_id,

        "predictedLabel":
            LABEL_NAMES[
                predicted_id
            ],

        "predictedLabelId":
            predicted_id,

        "confidence":
            float(
                test_probabilities[
                    index,
                    predicted_id
                ]
            ),

    }


    for class_id, class_name in enumerate(
        LABEL_NAMES
    ):

        row[
            f"prob_{class_name}"
        ] = float(
            test_probabilities[
                index,
                class_id
            ]
        )


    prediction_rows.append(
        row
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_file = (
    MODEL_DIR
    / "behavior_test_predictions.csv"
)


predictions_df.to_csv(
    predictions_file,
    index=False
)


# ==========================================================
# METRICS JSON
# ==========================================================

per_class = {}


for class_id, class_name in enumerate(
    LABEL_NAMES
):

    per_class[class_name] = {

        "precision":
            float(
                precision[class_id]
            ),

        "recall":
            float(
                recall[class_id]
            ),

        "f1":
            float(
                f1[class_id]
            ),

        "support":
            int(
                support[class_id]
            ),

    }


metrics = {

    "model":
        "GRU",

    "sequenceLength":
        sequence_length,

    "featureCount":
        feature_count,

    "classes":
        LABEL_NAMES,

    "trainingSamples":
        int(len(X_train)),

    "validationSamples":
        int(len(X_val)),

    "testSamples":
        int(len(X_test)),

    "testAccuracy":
        float(accuracy),

    "perClass":
        per_class,

    "confusionMatrix":
        cm.tolist(),

    "classWeights":
        {
            str(k): float(v)
            for k, v
            in class_weights.items()
        },

    "epochsConfigured":
        EPOCHS,

    "batchSize":
        BATCH_SIZE,

    "learningRate":
        LEARNING_RATE,

}


metrics_file = (
    MODEL_DIR
    / "behavior_model_metrics.json"
)


metrics_file.write_text(

    json.dumps(
        metrics,
        indent=2
    ),

    encoding="utf-8"

)


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Model:"
)

print(
    model_file
)

print()

print(
    "History:"
)

print(
    history_file
)

print()

print(
    "Predictions:"
)

print(
    predictions_file
)

print()

print(
    "Metrics:"
)

print(
    metrics_file
)

print()

print(
    "IMPORTANT:"
)

print(
    "The current dataset is a prototype dataset."
)

print(
    "Test results should NOT be treated as production"
)

print(
    "performance because several behaviour classes have"
)

print(
    "zero real test examples."
)

print()

print("=" * 70)