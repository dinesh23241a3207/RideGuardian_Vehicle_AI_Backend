import os
import json
import random
import copy
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms

import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATASET_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian\ai\datasets\Tyre_Condition_Dataset"
)

OUTPUT_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian\ai\models\tyre_condition_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "NEW",
    "SERVICEABLE",
    "UNUSABLE",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "coatnet_0_rw_224"

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 7


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# COLLECT DATASET
# ============================================================

def collect_dataset():

    image_paths = []

    labels = []

    print()
    print("=" * 70)
    print("LOADING TYRE DATASET")
    print("=" * 70)

    for class_name in CLASSES:

        class_directory = DATASET_DIR / class_name

        if not class_directory.exists():

            raise FileNotFoundError(
                f"Class directory not found: {class_directory}"
            )

        class_images = []

        for file_path in class_directory.iterdir():

            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in VALID_EXTENSIONS
            ):

                class_images.append(file_path)

                image_paths.append(file_path)

                labels.append(
                    CLASS_TO_INDEX[class_name]
                )

        print(
            f"{class_name:<15} : "
            f"{len(class_images)} images"
        )

    print("-" * 70)

    print(
        f"TOTAL IMAGES      : "
        f"{len(image_paths)}"
    )

    return image_paths, labels


# ============================================================
# DATASET CLASS
# ============================================================

class TyreDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.transform = transform


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = self.labels[index]

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as error:

            raise RuntimeError(
                f"Failed to load image: "
                f"{image_path}"
            ) from error


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            label
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10)
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

])


evaluation_transform = transforms.Compose([

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

])


# ============================================================
# CREATE STRATIFIED SPLITS
# ============================================================

def create_splits(
    image_paths,
    labels
):

    print()
    print("=" * 70)
    print("CREATING STRATIFIED DATA SPLITS")
    print("=" * 70)


    (
        train_paths,
        temporary_paths,
        train_labels,
        temporary_labels
    ) = train_test_split(

        image_paths,

        labels,

        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),

        stratify=labels,

        random_state=SEED
    )


    (
        validation_paths,
        test_paths,
        validation_labels,
        test_labels
    ) = train_test_split(

        temporary_paths,

        temporary_labels,

        test_size=0.5,

        stratify=temporary_labels,

        random_state=SEED
    )


    print()

    print(
        f"TRAIN IMAGES      : "
        f"{len(train_paths)}"
    )

    print(
        f"VALIDATION IMAGES : "
        f"{len(validation_paths)}"
    )

    print(
        f"TEST IMAGES       : "
        f"{len(test_paths)}"
    )


    return (

        train_paths,
        validation_paths,
        test_paths,

        train_labels,
        validation_labels,
        test_labels

    )


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    name,
    labels
):

    print()
    print(
        f"{name} DISTRIBUTION"
    )

    for class_name in CLASSES:

        class_index = CLASS_TO_INDEX[
            class_name
        ]

        count = labels.count(
            class_index
        )

        print(
            f"{class_name:<15}: "
            f"{count}"
        )


# ============================================================
# CREATE BALANCED SAMPLER
# ============================================================

def create_sampler(
    labels
):

    class_counts = np.bincount(
        labels,
        minlength=len(CLASSES)
    )


    print()
    print("TRAINING CLASS COUNTS:")

    for index, count in enumerate(
        class_counts
    ):

        print(
            f"{CLASSES[index]:<15}: "
            f"{count}"
        )


    class_weights = (

        1.0
        /
        class_counts

    )


    sample_weights = [

        class_weights[label]

        for label
        in labels

    ]


    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(
            sample_weights
        ),

        replacement=True

    )


    return sampler


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING COATNET MODEL")
    print("=" * 70)

    try:

        model = timm.create_model(

            MODEL_NAME,

            pretrained=True,

            num_classes=len(
                CLASSES
            )

        )

    except Exception as error:

        print()
        print(
            "WARNING: Could not load "
            "pretrained weights."
        )

        print(
            "Trying without pretrained "
            "weights..."
        )

        print(
            f"Original error: {error}"
        )


        model = timm.create_model(

            MODEL_NAME,

            pretrained=False,

            num_classes=len(
                CLASSES
            )

        )


    model = model.to(
        DEVICE
    )


    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(

    model,

    dataloader,

    criterion,

    optimizer,

    scaler

):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in dataloader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        with torch.amp.autocast(

            device_type=DEVICE.type,

            enabled=(
                DEVICE.type == "cuda"
            )

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


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


        scaler.step(
            optimizer
        )

        scaler.update()


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

            predictions
            ==
            labels

        ).sum().item()


        total += labels.size(
            0
        )


    average_loss = (

        running_loss
        /
        total

    )


    accuracy = (

        correct
        /
        total

    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()

def evaluate(

    model,

    dataloader,

    criterion

):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    for images, labels in dataloader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        outputs = model(
            images
        )


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

            predictions.cpu().numpy()

        )


        all_labels.extend(

            labels.cpu().numpy()

        )


    average_loss = (

        running_loss
        /
        len(
            all_labels
        )

    )


    accuracy = accuracy_score(

        all_labels,

        all_predictions

    )


    macro_f1 = f1_score(

        all_labels,

        all_predictions,

        average="macro",

        zero_division=0

    )


    return (

        average_loss,

        accuracy,

        macro_f1,

        all_labels,

        all_predictions

    )


# ============================================================
# SAVE DATA SPLIT
# ============================================================

def save_split_information(

    train_paths,
    validation_paths,
    test_paths

):

    split_data = {

        "train": [

            str(path)

            for path
            in train_paths

        ],

        "validation": [

            str(path)

            for path
            in validation_paths

        ],

        "test": [

            str(path)

            for path
            in test_paths

        ]

    }


    output_path = (

        OUTPUT_DIR
        /
        "dataset_split.json"

    )


    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            split_data,

            file,

            indent=4

        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RIDEGUARDIAN TYRE CONDITION AI - VERSION 3")
    print("=" * 70)

    print()

    print(
        f"DEVICE           : "
        f"{DEVICE}"
    )

    print(
        f"MODEL            : "
        f"{MODEL_NAME}"
    )

    print(
        f"IMAGE SIZE       : "
        f"{IMAGE_SIZE}"
    )

    print(
        f"BATCH SIZE       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"MAX EPOCHS       : "
        f"{EPOCHS}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    image_paths, labels = collect_dataset()


    # ========================================================
    # CREATE SPLITS
    # ========================================================

    (

        train_paths,
        validation_paths,
        test_paths,

        train_labels,
        validation_labels,
        test_labels

    ) = create_splits(

        image_paths,

        labels

    )


    print_distribution(

        "TRAIN",

        train_labels

    )


    print_distribution(

        "VALIDATION",

        validation_labels

    )


    print_distribution(

        "TEST",

        test_labels

    )


    save_split_information(

        train_paths,

        validation_paths,

        test_paths

    )


    # ========================================================
    # CREATE DATASETS
    # ========================================================

    train_dataset = TyreDataset(

        train_paths,

        train_labels,

        transform=train_transform

    )


    validation_dataset = TyreDataset(

        validation_paths,

        validation_labels,

        transform=evaluation_transform

    )


    test_dataset = TyreDataset(

        test_paths,

        test_labels,

        transform=evaluation_transform

    )


    # ========================================================
    # CREATE SAMPLER
    # ========================================================

    train_sampler = create_sampler(

        train_labels

    )


    # ========================================================
    # CREATE DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model()


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (

        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=3

        )

    )


    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(

        DEVICE.type,

        enabled=(
            DEVICE.type == "cuda"
        )

    )


    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    best_f1 = 0.0

    best_model_state = None

    epochs_without_improvement = 0

    history = []


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(

        1,

        EPOCHS + 1

    ):

        print()
        print("=" * 70)

        print(

            f"EPOCH "
            f"{epoch}/{EPOCHS}"

        )

        print("=" * 70)


        train_loss, train_accuracy = (

            train_one_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scaler

            )

        )


        (

            validation_loss,

            validation_accuracy,

            validation_f1,

            _,

            _

        ) = evaluate(

            model,

            validation_loader,

            criterion

        )


        scheduler.step(

            validation_f1

        )


        current_lr = (

            optimizer.param_groups[0]

            ["lr"]

        )


        print()

        print(

            f"TRAIN LOSS       : "
            f"{train_loss:.4f}"

        )

        print(

            f"TRAIN ACCURACY   : "
            f"{train_accuracy * 100:.2f}%"

        )

        print()

        print(

            f"VALIDATION LOSS  : "
            f"{validation_loss:.4f}"

        )

        print(

            f"VALIDATION ACC   : "
            f"{validation_accuracy * 100:.2f}%"

        )

        print(

            f"VALIDATION F1    : "
            f"{validation_f1 * 100:.2f}%"

        )

        print(

            f"LEARNING RATE    : "
            f"{current_lr:.8f}"

        )


        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "validation_loss": validation_loss,

            "validation_accuracy": validation_accuracy,

            "validation_f1": validation_f1,

            "learning_rate": current_lr

        })


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_f1 > best_f1:

            best_f1 = validation_f1

            best_model_state = copy.deepcopy(

                model.state_dict()

            )


            best_model_path = (

                OUTPUT_DIR

                /

                "best_tyre_condition_v3.pth"

            )


            torch.save({

                "model_name": MODEL_NAME,

                "image_size": IMAGE_SIZE,

                "classes": CLASSES,

                "class_to_index": CLASS_TO_INDEX,

                "epoch": epoch,

                "validation_f1": validation_f1,

                "model_state_dict": (

                    best_model_state

                )

            },

            best_model_path)


            print()

            print(

                "NEW BEST MODEL SAVED!"

            )

            print(

                f"BEST F1: "
                f"{best_f1 * 100:.2f}%"

            )


            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (

            epochs_without_improvement

            >=

            EARLY_STOPPING_PATIENCE

        ):

            print()

            print(
                "EARLY STOPPING ACTIVATED"
            )

            break


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (

        OUTPUT_DIR

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


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL TEST EVALUATION"
    )

    print("=" * 70)


    if best_model_state is not None:

        model.load_state_dict(

            best_model_state

        )


    (

        test_loss,

        test_accuracy,

        test_f1,

        test_labels_result,

        test_predictions

    ) = evaluate(

        model,

        test_loader,

        criterion

    )


    print()

    print(

        f"TEST LOSS        : "
        f"{test_loss:.4f}"

    )

    print(

        f"TEST ACCURACY    : "
        f"{test_accuracy * 100:.2f}%"

    )

    print(

        f"TEST MACRO F1    : "
        f"{test_f1 * 100:.2f}%"

    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        test_labels_result,

        test_predictions,

        target_names=CLASSES,

        digits=4,

        zero_division=0

    )


    print()

    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    print(
        report
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(

        test_labels_result,

        test_predictions

    )


    print()

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print(
        matrix
    )


    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    results = {

        "test_loss": test_loss,

        "test_accuracy": test_accuracy,

        "test_macro_f1": test_f1,

        "confusion_matrix": matrix.tolist(),

        "classes": CLASSES

    }


    results_path = (

        OUTPUT_DIR

        /

        "final_test_results.json"

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


    print()

    print("=" * 70)

    print(
        "TYRE CONDITION V3 TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"BEST VALIDATION F1 : "
        f"{best_f1 * 100:.2f}%"
    )

    print(
        f"FINAL TEST F1      : "
        f"{test_f1 * 100:.2f}%"
    )

    print()

    print(
        f"MODEL SAVED TO:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()