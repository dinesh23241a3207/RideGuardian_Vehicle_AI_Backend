import json
import csv
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

import timm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

MODEL_DIR = (
    PROJECT_DIR
    / "ai"
    / "models"
    / "tyre_condition_v3"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_tyre_condition_v3.pth"
)

SPLIT_PATH = (
    MODEL_DIR
    / "dataset_split.json"
)

OUTPUT_DIR = (
    MODEL_DIR
    / "error_analysis"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("RIDEGUARDIAN TYRE MODEL V3 ERROR ANALYSIS")
print("=" * 70)

print()
print(f"Using device: {DEVICE}")

print()
print("Loading model checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


MODEL_NAME = checkpoint["model_name"]

CLASSES = checkpoint["classes"]

print(f"Model: {MODEL_NAME}")
print(f"Classes: {CLASSES}")


# ============================================================
# CREATE MODEL
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# DATASET
# ============================================================

class TyreTestDataset(Dataset):

    def __init__(
        self,
        image_paths,
        transform
    ):

        self.image_paths = [
            Path(path)
            for path in image_paths
        ]

        self.transform = transform


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image_tensor = self.transform(
            image
        )


        # Get actual class from folder name
        actual_class = (
            image_path.parent.name
        )

        actual_index = CLASSES.index(
            actual_class
        )


        return (
            image_tensor,
            actual_index,
            str(image_path)
        )


# ============================================================
# LOAD TEST SPLIT
# ============================================================

print()
print("Loading exact test split...")

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8"
) as file:

    split_data = json.load(
        file
    )


test_paths = split_data[
    "test"
]


print(
    f"Test images found: "
    f"{len(test_paths)}"
)


# ============================================================
# CREATE DATA LOADER
# ============================================================

test_dataset = TyreTestDataset(
    test_paths,
    transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for actual_class in CLASSES:

    for predicted_class in CLASSES:

        if actual_class != predicted_class:

            error_folder = (
                OUTPUT_DIR
                / f"{actual_class}_predicted_as_{predicted_class}"
            )

            error_folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# RUN ANALYSIS
# ============================================================

print()
print("=" * 70)
print("ANALYZING TEST PREDICTIONS")
print("=" * 70)


all_results = []

total_images = 0

correct_predictions = 0

wrong_predictions = 0


with torch.no_grad():

    for (

        images,
        actual_labels,
        image_paths

    ) in test_loader:


        images = images.to(
            DEVICE
        )


        outputs = model(
            images
        )


        probabilities = F.softmax(
            outputs,
            dim=1
        )


        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )


        for i in range(
            len(image_paths)
        ):

            total_images += 1


            actual_index = int(
                actual_labels[i]
            )


            predicted_index = int(
                predictions[i]
            )


            confidence = float(
                confidences[i].cpu()
            )


            actual_class = CLASSES[
                actual_index
            ]


            predicted_class = CLASSES[
                predicted_index
            ]


            image_path = Path(
                image_paths[i]
            )


            is_correct = (
                actual_index
                ==
                predicted_index
            )


            if is_correct:

                correct_predictions += 1

            else:

                wrong_predictions += 1


                error_folder = (

                    OUTPUT_DIR

                    /

                    f"{actual_class}_predicted_as_{predicted_class}"

                )


                # Create unique output filename
                output_filename = (

                    f"confidence_"
                    f"{confidence:.4f}_"
                    f"{image_path.name}"

                )


                destination = (

                    error_folder
                    /
                    output_filename

                )


                shutil.copy2(
                    image_path,
                    destination
                )


            # Get all class probabilities
            probability_values = probabilities[
                i
            ].cpu().tolist()


            result = {

                "image_path": str(
                    image_path
                ),

                "actual_class": actual_class,

                "predicted_class": predicted_class,

                "confidence": round(
                    confidence,
                    6
                ),

                "correct": is_correct

            }


            # Add probability for every class
            for class_index, class_name in enumerate(
                CLASSES
            ):

                result[
                    f"probability_{class_name}"
                ] = round(

                    probability_values[
                        class_index
                    ],

                    6

                )


            all_results.append(
                result
            )


# ============================================================
# SAVE CSV REPORT
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "prediction_report.csv"
)


print()
print("Saving CSV report...")


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:


    fieldnames = [

        "image_path",

        "actual_class",

        "predicted_class",

        "confidence",

        "correct"

    ]


    for class_name in CLASSES:

        fieldnames.append(

            f"probability_{class_name}"

        )


    writer = csv.DictWriter(

        csv_file,

        fieldnames=fieldnames

    )


    writer.writeheader()


    for result in all_results:

        writer.writerow(
            result
        )


# ============================================================
# CALCULATE ERROR SUMMARY
# ============================================================

error_summary = {}


for actual_class in CLASSES:

    error_summary[
        actual_class
    ] = {}


    for predicted_class in CLASSES:

        error_summary[
            actual_class
        ][
            predicted_class
        ] = 0


for result in all_results:

    actual_class = result[
        "actual_class"
    ]

    predicted_class = result[
        "predicted_class"
    ]


    error_summary[
        actual_class
    ][
        predicted_class
    ] += 1


# ============================================================
# PRINT RESULTS
# ============================================================

accuracy = (

    correct_predictions
    /
    total_images

    * 100

)


print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)


print()

print(
    f"TOTAL TEST IMAGES : "
    f"{total_images}"
)

print(
    f"CORRECT           : "
    f"{correct_predictions}"
)

print(
    f"WRONG             : "
    f"{wrong_predictions}"
)

print(
    f"ACCURACY          : "
    f"{accuracy:.2f}%"
)


# ============================================================
# PRINT CONFUSION SUMMARY
# ============================================================

print()
print("=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)


for actual_class in CLASSES:

    print()

    print(
        f"ACTUAL: "
        f"{actual_class}"
    )


    for predicted_class in CLASSES:

        count = (

            error_summary[
                actual_class
            ][
                predicted_class
            ]

        )


        print(

            f"  Predicted as "
            f"{predicted_class:<15}: "
            f"{count}"

        )


# ============================================================
# SAFETY CRITICAL ERRORS
# ============================================================

print()
print("=" * 70)
print("SAFETY CRITICAL ERRORS")
print("=" * 70)


critical_errors = [

    result

    for result
    in all_results

    if (

        result["actual_class"]
        == "UNUSABLE"

        and

        result["predicted_class"]
        != "UNUSABLE"

    )

]


print()

print(

    f"UNUSABLE tyres predicted as "
    f"safe categories: "

    f"{len(critical_errors)}"

)


for result in critical_errors:

    print()

    print(

        f"IMAGE      : "
        f"{Path(result['image_path']).name}"

    )

    print(

        f"PREDICTED  : "
        f"{result['predicted_class']}"

    )

    print(

        f"CONFIDENCE : "
        f"{result['confidence'] * 100:.2f}%"

    )


print()
print("=" * 70)

print(
    "OUTPUT LOCATION"
)

print("=" * 70)

print(
    OUTPUT_DIR
)

print()

print(
    "Open the error_analysis folder "
    "and inspect the incorrectly "
    "classified tyre images."
)

print()