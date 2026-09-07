import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================
# RIDEGUARDIAN
# CoAtNet CNN - Tyre Condition Training
#
# Classes:
#   NEW
#   SERVICEABLE
#   UNUSABLE
#
# GPU:
#   NVIDIA RTX 4080 Laptop GPU
#
# Windows/Python 3.14 compatible:
#   num_workers = 0
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "tyre_condition"
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "coatnet"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

# IMPORTANT:
# Windows + Python 3.14 can have DataLoader worker
# subprocess problems. Keep this at 0.
NUM_WORKERS = 0

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

MODEL_NAME = "coatnet_0_rw_224.sw_in1k"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print()
print("=" * 70)
print("RIDEGUARDIAN CoAtNet TRAINING")
print("=" * 70)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "GPU Memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
    )

print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"

VAL_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"


for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDataset folder not found:\n{directory}"
        )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("Loading datasets...")


train_dataset = datasets.ImageFolder(
    str(TRAIN_DIR),
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    str(VAL_DIR),
    transform=eval_transform
)


test_dataset = datasets.ImageFolder(
    str(TEST_DIR),
    transform=eval_transform
)


class_names = train_dataset.classes

num_classes = len(class_names)


print()
print("Classes:")

for index, name in enumerate(class_names):

    print(
        f"  {index}: {name}"
    )


print()
print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# VERIFY CLASSES
# ============================================================

if val_dataset.classes != class_names:

    raise RuntimeError(
        "\nValidation classes do not match "
        "training classes.\n"
        f"Train: {class_names}\n"
        f"Validation: {val_dataset.classes}"
    )


if test_dataset.classes != class_names:

    raise RuntimeError(
        "\nTest classes do not match "
        "training classes.\n"
        f"Train: {class_names}\n"
        f"Test: {test_dataset.classes}"
    )


# ============================================================
# DATALOADERS
# ============================================================
#
# IMPORTANT:
#
# num_workers=0 is intentional.
#
# This prevents the Windows/Python 3.14:
#
# "DataLoader worker exited unexpectedly"
#
# error.
#
# The model STILL trains on the RTX 4080.
# ============================================================

print()
print(
    "Creating DataLoaders..."
)

print(
    "DataLoader workers:",
    NUM_WORKERS
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)


print()
print(
    "Training class distribution:"
)

for index, name in enumerate(class_names):

    print(
        f"  {name}: {class_counts[index]}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = class_counts.sum()

class_weights = (
    total_samples
    /
    (
        num_classes
        * class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print()
print(
    "Class weights:"
)

for index, name in enumerate(class_names):

    print(
        f"  {name}: "
        f"{class_weights[index].item():.4f}"
    )


# ============================================================
# CREATE CoAtNet
# ============================================================

print()
print(
    "Loading pretrained CoAtNet..."
)

print(
    "Model:",
    MODEL_NAME
)


try:

    model = timm.create_model(

        MODEL_NAME,

        pretrained=True,

        num_classes=num_classes
    )

except Exception as error:

    print()
    print(
        "ERROR WHILE LOADING CoAtNet:"
    )

    print(error)

    print()
    print(
        "If this is the first run, make sure "
        "the computer has internet access so "
        "the pretrained weights can be downloaded."
    )

    raise


model = model.to(DEVICE)


print(
    "CoAtNet loaded successfully."
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3
)


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = torch.cuda.is_available()


if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_targets = []


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        if USE_AMP:

            with torch.amp.autocast(
                "cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss
        /
        len(train_dataset)
    )


    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_targets = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            if USE_AMP:

                with torch.amp.autocast(
                    "cuda"
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )


            all_targets.extend(
                labels.detach()
                .cpu()
                .numpy()
            )


    loss = (
        running_loss
        /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )


    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        all_targets,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0

history = []


print()
print("=" * 70)
print("STARTING CoAtNet TRAINING")
print("=" * 70)


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    (
        train_loss,
        train_accuracy
    ) = train_one_epoch()


    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        val_loader
    )


    scheduler.step(
        val_f1
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    epoch_time = (
        time.time()
        -
        epoch_start
    )


    print()
    print(
        "-" * 70
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Precision: "
        f"{val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall: "
        f"{val_recall * 100:.2f}%"
    )

    print(
        f"Val F1: "
        f"{val_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.1f}s"
    )


    history.append({

        "epoch": epoch,

        "train_loss": float(
            train_loss
        ),

        "train_accuracy": float(
            train_accuracy
        ),

        "validation_loss": float(
            val_loss
        ),

        "validation_accuracy": float(
            val_accuracy
        ),

        "validation_precision": float(
            val_precision
        ),

        "validation_recall": float(
            val_recall
        ),

        "validation_f1": float(
            val_f1
        ),

        "learning_rate": float(
            current_lr
        )

    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1


        checkpoint_path = (
            MODEL_DIR
            / "best_coatnet.pth"
        )


        torch.save({

            "model_name":
                MODEL_NAME,

            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "num_classes":
                num_classes,

            "image_size":
                IMAGE_SIZE,

            "best_validation_f1":
                float(best_val_f1),

            "epoch":
                epoch

        }, checkpoint_path)


        print()
        print(
            "BEST MODEL SAVED"
        )

        print(
            checkpoint_path
        )


# ============================================================
# TRAINING TIME
# ============================================================

total_time = (
    time.time()
    -
    training_start
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Training time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"Best validation F1: "
    f"{best_val_f1 * 100:.2f}%"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (
    MODEL_DIR
    / "class_names.json"
)


with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    MODEL_DIR
    / "training_history.json"
)


with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR
    / "best_coatnet.pth"
)


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print()
print(
    "Best CoAtNet checkpoint loaded."
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)


(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_targets,
    test_predictions
) = evaluate(
    test_loader
)


print()
print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

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
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        test_targets,
        test_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_targets,
    test_predictions
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(cm)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results = {

    "model":
        MODEL_NAME,

    "classes":
        class_names,

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "test_precision":
        float(test_precision),

    "test_recall":
        float(test_recall),

    "test_f1":
        float(test_f1),

    "confusion_matrix":
        cm.tolist()

}


results_path = (
    MODEL_DIR
    / "test_results.json"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("MODEL FILES")
print("=" * 70)

print()
print(
    "Best model:"
)

print(
    best_model_path
)

print()
print(
    "Class names:"
)

print(
    class_names_path
)

print()
print(
    "Training history:"
)

print(
    history_path
)

print()
print(
    "Test results:"
)

print(
    results_path
)

print()
print("=" * 70)
print("CoAtNet tyre model is ready.")
print("=" * 70)