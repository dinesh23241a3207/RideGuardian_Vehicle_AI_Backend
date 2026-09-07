from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


# ==========================================================
# RIDEGUARDIAN
# EVENT BEHAVIOUR GRU
# ==========================================================

BASE_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

DATA_DIR = (
    BASE_DIR
    / "ai"
    / "datasets"
    / "rides"
    / "event_ground_truth_scaled"
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


MODEL_FILE = (
    MODEL_DIR
    / "event_behavior_gru.keras"
)

RESULTS_FILE = (
    MODEL_DIR
    / "event_behavior_gru_results.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

SEED = 42

EPOCHS = 100

BATCH_SIZE = 16

LEARNING_RATE = 0.001


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

]


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN")
print("EVENT BEHAVIOUR GRU TRAINING")
print("=" * 70)

print()


def load_split(name):

    file = DATA_DIR / f"{name}.npz"

    if not file.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file}"
        )


    data = np.load(file)

    return (

        data["X"].astype(
            np.float32
        ),

        data["y"].astype(
            np.int64
        ),

    )


X_train, y_train = load_split(
    "train"
)

X_val, y_val = load_split(
    "val"
)

X_test, y_test = load_split(
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
# REMOVE UNSUPPORTED MULTI_EVENT
# ==========================================================

valid_classes = np.arange(
    len(LABEL_NAMES)
)


train_mask = np.isin(
    y_train,
    valid_classes
)

val_mask = np.isin(
    y_val,
    valid_classes
)

test_mask = np.isin(
    y_test,
    valid_classes
)


X_train = X_train[
    train_mask
]

y_train = y_train[
    train_mask
]

X_val = X_val[
    val_mask
]

y_val = y_val[
    val_mask
]

X_test = X_test[
    test_mask
]

y_test = y_test[
    test_mask
]


# ==========================================================
# DISTRIBUTION
# ==========================================================

print()
print("=" * 70)
print("DATA DISTRIBUTION")
print("=" * 70)

print()


for name, labels in [

    ("TRAIN", y_train),

    ("VAL", y_val),

    ("TEST", y_test),

]:

    print(
        name
    )

    for class_id, class_name in enumerate(
        LABEL_NAMES
    ):

        count = int(
            (labels == class_id).sum()
        )

        print(
            f"  {class_name:20} {count}"
        )

    print()


# ==========================================================
# MODEL
# ==========================================================

sequence_length = X_train.shape[1]

feature_count = X_train.shape[2]

num_classes = len(
    LABEL_NAMES
)


print("=" * 70)
print("BUILDING MODEL")
print("=" * 70)

print()


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            sequence_length,
            feature_count
        )
    ),

    tf.keras.layers.GRU(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.GRU(
        32
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    ),

])


# ==========================================================
# COMPILE
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
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

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=15,

        restore_best_weights=True,

        verbose=1,

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1,

    ),

]


# ==========================================================
# TRAIN
# ==========================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

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

    callbacks=callbacks,

    verbose=1,

)


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    MODEL_FILE
)


print()
print(
    "Model saved:"
)

print(
    MODEL_FILE
)


# ==========================================================
# TEST
# ==========================================================

print()
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

print()


test_loss, test_accuracy = (
    model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
)


probabilities = model.predict(
    X_test,
    verbose=0
)


predictions = np.argmax(
    probabilities,
    axis=1
)


accuracy = accuracy_score(
    y_test,
    predictions
)


macro_f1 = f1_score(
    y_test,
    predictions,
    labels=np.arange(
        num_classes
    ),
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    f"Loss:          {test_loss:.4f}"
)

print(
    f"Accuracy:      {accuracy:.4f}"
)

print(
    f"Macro F1:      {macro_f1:.4f}"
)

print(
    f"Weighted F1:   {weighted_f1:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print()
print(
    "Classification Report:"
)

print()


report = classification_report(

    y_test,

    predictions,

    labels=np.arange(
        num_classes
    ),

    target_names=LABEL_NAMES,

    zero_division=0,

)


print(
    report
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    predictions,

    labels=np.arange(
        num_classes
    )

)


print(
    "Confusion Matrix:"
)

print()

print(
    cm
)


# ==========================================================
# PREDICTION DISTRIBUTION
# ==========================================================

print()
print(
    "Prediction Distribution:"
)

print()


for class_id, class_name in enumerate(
    LABEL_NAMES
):

    count = int(
        (predictions == class_id).sum()
    )

    print(
        f"{class_name:20} {count}"
    )


# ==========================================================
# RESULTS
# ==========================================================

results = {

    "model":
        str(MODEL_FILE),

    "sequenceLength":
        int(sequence_length),

    "featureCount":
        int(feature_count),

    "classes":
        LABEL_NAMES,

    "trainSamples":
        int(len(X_train)),

    "validationSamples":
        int(len(X_val)),

    "testSamples":
        int(len(X_test)),

    "testLoss":
        float(test_loss),

    "accuracy":
        float(accuracy),

    "macroF1":
        float(macro_f1),

    "weightedF1":
        float(weighted_f1),

    "confusionMatrix":
        cm.tolist(),

    "classificationReport":
        classification_report(

            y_test,

            predictions,

            labels=np.arange(
                num_classes
            ),

            target_names=LABEL_NAMES,

            output_dict=True,

            zero_division=0,

        ),

}


RESULTS_FILE.write_text(

    json.dumps(
        results,
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
    MODEL_FILE
)

print()

print(
    "Results:"
)

print(
    RESULTS_FILE
)

print()

print(
    "Previous behavior_gru.keras was NOT modified."
)

print(
    "Original datasets were NOT modified."
)

print("=" * 70)