import json
import time
from pathlib import Path

import torch
import timm

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# RIDEGUARDIAN
# VEHICLE DAMAGE CONDITION - CoAtNet
#
# Classes:
#   Broken part
#   Corrosion
#   Cracked
#   Dent
#   Flaking
#   Missing part
#   Paint chip
#   Scratch
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = (
    Path(__file__).resolve().parent.parent
)

DATASET_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "vehicle_damage_condition"
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "damage_condition"
)


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = (
    DATASET_DIR
    / "train"
)

VAL_DIR = (
    DATASET_DIR
    / "validation"
)

TEST_DIR = (
    DATASET_DIR
    / "test"
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


# ============================================================
# DATA LOADER SETTINGS
#
# IMPORTANT:
# Previous DataLoader worker crash occurred on this system.
# Keep workers at 0 for stability.
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = True


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "coatnet_0_rw_224.sw_in1k"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
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
# PRINT SYSTEM INFORMATION
# ============================================================

print()
print("=" * 70)
print(
    "RIDEGUARDIAN VEHICLE DAMAGE CoAtNet"
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
        /
        (1024 ** 3)
    )

    print(
        f"GPU Memory: {gpu_memory:.2f} GB"
    )

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not TRAIN_DIR.exists():

    raise FileNotFoundError(
        f"\nTraining dataset not found:\n"
        f"{TRAIN_DIR}"
    )


if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"\nValidation dataset not found:\n"
        f"{VAL_DIR}"
    )


if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"\nTest dataset not found:\n"
        f"{TEST_DIR}"
    )


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
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

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

print()

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = (
    train_dataset.classes
)

num_classes = len(
    class_names
)


print(
    "Classes:"
)

for index, class_name in enumerate(
    class_names
):

    print(
        f"  {index}: {class_name}"
    )

print()

print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)

print(
    f"Total images: "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

print()
print(
    "Creating DataLoaders..."
)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=False
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=False
)


print(
    f"DataLoader workers: "
    f"{NUM_WORKERS}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_counts = torch.zeros(
    num_classes,
    dtype=torch.float32
)


for _, label in train_dataset.samples:

    train_counts[label] += 1


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


# ============================================================
# CLASS WEIGHTS
#
# Inverse square-root weighting gives rare classes more
# importance without making their weight excessively large.
# ============================================================

class_weights = (
    1.0
    /
    torch.sqrt(
        train_counts
    )
)


# Normalize around mean weight 1.

class_weights = (
    class_weights
    /
    class_weights.mean()
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
        f"{class_weights[index]:.4f}"
    )


class_weights = (
    class_weights
    .to(DEVICE)
)


# ============================================================
# LOAD PRETRAINED CoAtNet
# ============================================================

print()
print("=" * 70)
print(
    "LOADING PRETRAINED CoAtNet"
)
print("=" * 70)

print()

print(
    f"Model: {MODEL_NAME}"
)

model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

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
# SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2
    )
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = (
    DEVICE.type == "cuda"
)

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

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

            with torch.amp.autocast(
                "cuda"
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

        total_loss += (
            loss.item()
            *
            images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions
            ==
            labels
        ).sum().item()

        total += (
            labels.size(0)
        )

    epoch_loss = (
        total_loss
        /
        total
    )

    epoch_accuracy = (
        correct
        /
        total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []

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

            if use_amp:

                with torch.amp.autocast(
                    "cuda"
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            total_loss += (
                loss.item()
                *
                images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions
                ==
                labels
            ).sum().item()

            total += (
                labels.size(0)
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    loss_value = (
        total_loss
        /
        total
    )

    accuracy = (
        correct
        /
        total
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            average="weighted",

            zero_division=0
        )
    )

    return {

        "loss":
            loss_value,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "labels":
            all_labels,

        "predictions":
            all_predictions
    }


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print(
    "STARTING DAMAGE CoAtNet TRAINING"
)
print("=" * 70)

print()

best_f1 = 0.0

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    start_time = time.time()

    print(
        "-" * 70
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = evaluate(
        val_loader
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        validation["f1"]
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        -
        start_time
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

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
        f"{validation['loss']:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{validation['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val Precision: "
        f"{validation['precision'] * 100:.2f}%"
    )

    print(
        f"Val Recall: "
        f"{validation['recall'] * 100:.2f}%"
    )

    print(
        f"Val F1: "
        f"{validation['f1'] * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.1f}s"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            validation["loss"],

        "val_accuracy":
            validation["accuracy"],

        "val_precision":
            validation["precision"],

        "val_recall":
            validation["recall"],

        "val_f1":
            validation["f1"],

        "learning_rate":
            current_lr,

        "epoch_time":
            epoch_time

    })

    # --------------------------------------------------------
    # Best model
    #
    # F1 is used rather than raw accuracy because the dataset
    # is highly imbalanced.
    # --------------------------------------------------------

    if validation["f1"] > best_f1:

        best_f1 = (
            validation["f1"]
        )

        epochs_without_improvement = 0

        checkpoint_path = (
            MODEL_DIR
            / "best_damage_coatnet.pth"
        )

        torch.save({

            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "model_name":
                MODEL_NAME,

            "image_size":
                IMAGE_SIZE,

            "best_val_f1":
                best_f1

        }, checkpoint_path)

        print()
        print(
            "BEST MODEL SAVED"
        )

        print(
            checkpoint_path
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "EARLY STOPPING"
        )

        print(
            "Validation F1 has not improved."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
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

print()
print("=" * 70)
print(
    "LOADING BEST DAMAGE MODEL"
)
print("=" * 70)

checkpoint_path = (
    MODEL_DIR
    / "best_damage_coatnet.pth"
)

checkpoint = torch.load(

    checkpoint_path,

    map_location=DEVICE,

    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print(
    "FINAL TEST SET EVALUATION"
)
print("=" * 70)

test_results = evaluate(
    test_loader
)


print()

print(
    f"Test Loss: "
    f"{test_results['loss']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_results['accuracy'] * 100:.2f}%"
)

print(
    f"Test Precision: "
    f"{test_results['precision'] * 100:.2f}%"
)

print(
    f"Test Recall: "
    f"{test_results['recall'] * 100:.2f}%"
)

print(
    f"Test F1: "
    f"{test_results['f1'] * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print(
    "CLASSIFICATION REPORT"
)
print("=" * 70)

report = classification_report(

    test_results["labels"],

    test_results["predictions"],

    target_names=class_names,

    zero_division=0
)

print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print(
    "CONFUSION MATRIX"
)
print("=" * 70)

matrix = confusion_matrix(

    test_results["labels"],

    test_results["predictions"]
)

print(
    matrix
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
# SAVE TEST RESULTS
# ============================================================

test_results_path = (
    MODEL_DIR
    / "test_results.json"
)

test_results_save = {

    "test_loss":
        test_results["loss"],

    "test_accuracy":
        test_results["accuracy"],

    "test_precision":
        test_results["precision"],

    "test_recall":
        test_results["recall"],

    "test_f1":
        test_results["f1"],

    "confusion_matrix":
        matrix.tolist(),

    "class_names":
        class_names
}


with open(
    test_results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        test_results_save,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    "DAMAGE CoAtNet TRAINING COMPLETE"
)
print("=" * 70)

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
    "Damage CoAtNet is ready."
)

print(
    "=" * 70
)
