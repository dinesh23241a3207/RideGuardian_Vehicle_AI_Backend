import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import timm

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

# ============================================================
# RIDEGUARDIAN
# TYRE CONDITION CLASSIFICATION - CoAtNet
#
# Classes:
#   NEW
#   SERVICEABLE
#   UNUSABLE
#
# Dataset:
#   ai/datasets/Tyre_Condition_Dataset/
#
# IMPORTANT:
#   This is a NEW model.
#   It does NOT modify the existing vehicle_parts,
#   damage_condition, or engine_fault models.
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    AI_DIR
    / "datasets"
    / "Tyre_Condition_Dataset"
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "tyre_condition"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

RANDOM_SEED = 42

NUM_WORKERS = 0
PIN_MEMORY = True

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MODEL_NAME = "coatnet_0_rw_224.sw_in1k"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device(
        "cuda"
    )
else:
    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print()
print("=" * 70)
print(
    "RIDEGUARDIAN TYRE CONDITION CoAtNet"
)
print("=" * 70)

print()
print(
    "PyTorch:",
    torch.__version__
)

print(
    "Device:",
    DEVICE
)

print(
    "CUDA available:",
    torch.cuda.is_available()
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

    gpu_memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU Memory: {gpu_memory:.2f} GB"
    )


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nTyre dataset not found:\n"
        f"{DATASET_DIR}\n\n"
        "Expected structure:\n"
        "Tyre_Condition_Dataset/\n"
        "  NEW/\n"
        "  SERVICEABLE/\n"
        "  UNUSABLE/\n"
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=8
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
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
        ),
    ]
)


eval_transform = transforms.Compose(
    [
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
        ),
    ]
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print(
    "Loading tyre dataset..."
)

base_dataset = datasets.ImageFolder(
    DATASET_DIR
)

class_names = base_dataset.classes
targets = np.array(
    base_dataset.targets
)

num_classes = len(
    class_names
)

print()
print(
    "Classes:"
)

for index, class_name in enumerate(
    class_names
):

    print(
        f"  {index}: {class_name}"
    )

expected_classes = {
    "NEW",
    "SERVICEABLE",
    "UNUSABLE"
}

if set(class_names) != expected_classes:

    raise ValueError(
        "\nUnexpected tyre classes found.\n"
        f"Found: {class_names}\n"
        f"Expected: {sorted(expected_classes)}"
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print()
print(
    "Dataset distribution:"
)

for index, class_name in enumerate(
    class_names
):

    count = int(
        np.sum(targets == index)
    )

    print(
        f"  {class_name}: {count}"
    )

total_images = len(
    base_dataset
)

print()
print(
    "Total images:",
    total_images
)

if total_images < 30:

    raise ValueError(
        "The tyre dataset is too small for a reliable "
        "train/validation/test split."
    )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

indices = np.arange(
    total_images
)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=targets
)

temp_targets = targets[
    temp_indices
]

val_fraction_of_temp = (
    VAL_RATIO
    /
    (VAL_RATIO + TEST_RATIO)
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=(1.0 - val_fraction_of_temp),
    random_state=RANDOM_SEED,
    stratify=temp_targets
)


print()
print(
    "Split:"
)

print(
    f"  Train: {len(train_indices)}"
)

print(
    f"  Validation: {len(val_indices)}"
)

print(
    f"  Test: {len(test_indices)}"
)


# ============================================================
# DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

train_dataset_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

eval_dataset_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=eval_transform
)

train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    eval_dataset_full,
    val_indices
)

test_dataset = Subset(
    eval_dataset_full,
    test_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
    if DEVICE.type == "cuda"
    else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
    if DEVICE.type == "cuda"
    else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
    if DEVICE.type == "cuda"
    else False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_targets = targets[
    train_indices
]

train_counts = np.bincount(
    train_targets,
    minlength=num_classes
)

print()
print(
    "Training class distribution:"
)

for index, class_name in enumerate(
    class_names
):

    print(
        f"  {class_name}: "
        f"{int(train_counts[index])}"
    )


# Inverse square-root weighting keeps rare classes
# important without making weights excessively large.
class_weights = (
    1.0
    /
    np.sqrt(
        np.maximum(
            train_counts,
            1
        )
    )
)

class_weights = (
    class_weights
    /
    class_weights.mean()
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

for index, class_name in enumerate(
    class_names
):

    print(
        f"  {class_name}: "
        f"{class_weights[index].item():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print()
print(
    "=" * 70
)

print(
    "LOADING PRETRAINED CoAtNet"
)

print(
    "=" * 70
)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes
)

model = model.to(
    DEVICE
)

print()
print(
    "CoAtNet loaded successfully."
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN / VALIDATION FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

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

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

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

            outputs = model(
                images
            )

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

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss
        /
        max(total, 1)
    )

    epoch_accuracy = (
        correct
        /
        max(total, 1)
        *
        100.0
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


@torch.no_grad()
def evaluate_loader(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    epoch_loss = (
        running_loss
        /
        max(total, 1)
    )

    epoch_accuracy = (
        correct
        /
        max(total, 1)
        *
        100.0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = -1.0
best_val_loss = float("inf")

epochs_without_improvement = 0

history = []


print()
print(
    "=" * 70
)

print(
    "STARTING TYRE CONDITION TRAINING"
)

print(
    "=" * 70
)


for epoch in range(
    1,
    EPOCHS + 1
):

    start_time = time.time()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy, _, _ = (
        evaluate_loader(
            val_loader
        )
    )

    scheduler.step()

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    epoch_time = (
        time.time()
        -
        start_time
    )

    print()
    print(
        "-" * 70
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.1f}s"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time
        }
    )

    improved = (
        val_accuracy > best_val_accuracy
        or
        (
            abs(val_accuracy - best_val_accuracy)
            < 1e-8
            and
            val_loss < best_val_loss
        )
    )

    if improved:

        best_val_accuracy = (
            val_accuracy
        )

        best_val_loss = (
            val_loss
        )

        epochs_without_improvement = 0

        checkpoint_path = (
            MODEL_DIR
            /
            "best_tyre_condition_coatnet.pth"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "model_name":
                    MODEL_NAME,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy,

                "best_val_loss":
                    best_val_loss,

                "random_seed":
                    RANDOM_SEED
            },
            checkpoint_path
        )

        print()
        print(
            "BEST MODEL SAVED"
        )

        print(
            checkpoint_path
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                f"Early stopping after "
                f"{epoch} epochs."
            )

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    MODEL_DIR
    /
    "training_history.json"
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

checkpoint_path = (
    MODEL_DIR
    /
    "best_tyre_condition_coatnet.pth"
)

if not checkpoint_path.exists():

    raise FileNotFoundError(
        "Best tyre model checkpoint was not created."
    )

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    DEVICE
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print(
    "=" * 70
)

print(
    "FINAL TEST EVALUATION"
)

print(
    "=" * 70
)

test_loss, test_accuracy, y_true, y_pred = (
    evaluate_loader(
        test_loader
    )
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        zero_division=0
    )
)

print()
print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print()
print(
    "Classification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )
)

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_results = {
    "model_name": MODEL_NAME,
    "image_size": IMAGE_SIZE,
    "classes": class_names,
    "train_images": int(len(train_indices)),
    "validation_images": int(len(val_indices)),
    "test_images": int(len(test_indices)),
    "best_validation_accuracy":
        float(best_val_accuracy),
    "best_validation_loss":
        float(best_val_loss),
    "test_loss":
        float(test_loss),
    "test_accuracy":
        float(test_accuracy),
    "precision_by_class": {
        class_names[index]:
            float(precision[index])
        for index in range(num_classes)
    },
    "recall_by_class": {
        class_names[index]:
            float(recall[index])
        for index in range(num_classes)
    },
    "f1_by_class": {
        class_names[index]:
            float(f1[index])
        for index in range(num_classes)
    },
    "confusion_matrix":
        confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(num_classes)
        ).tolist()
}

test_results_path = (
    MODEL_DIR
    /
    "test_results.json"
)

with open(
    test_results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        test_results,
        file,
        indent=4
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = (
    MODEL_DIR
    /
    "class_names.json"
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
# FINAL
# ============================================================

print()
print(
    "=" * 70
)

print(
    "TYRE CONDITION TRAINING COMPLETE"
)

print(
    "=" * 70
)

print()
print(
    "Best model:"
)

print(
    checkpoint_path
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
    test_results_path
)

print()
print(
    f"Final test accuracy: "
    f"{test_accuracy:.2f}%"
)

print()
