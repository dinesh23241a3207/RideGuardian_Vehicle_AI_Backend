import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# RIDEGUARDIAN - DATASET PREPARATION
# Tyre Condition Dataset
#
# Original dataset is NEVER modified.
#
# Output:
# ai/datasets/processed/tyre_condition/
#     train/
#         NEW/
#         SERVICEABLE/
#         UNUSABLE/
#     validation/
#         NEW/
#         SERVICEABLE/
#         UNUSABLE/
#     test/
#         NEW/
#         SERVICEABLE/
#         UNUSABLE/
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

AI_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = AI_DIR / "datasets"

SOURCE_DIR = DATASETS_DIR / "Tyre_Condition_Dataset"

OUTPUT_DIR = DATASETS_DIR / "processed" / "tyre_condition"

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

RANDOM_STATE = 42

# 70% training
# 15% validation
# 15% testing

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

CLASSES = [
    "NEW",
    "SERVICEABLE",
    "UNUSABLE"
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_output_directories():

    print("\nCreating output directories...")

    for split in ["train", "validation", "test"]:

        for class_name in CLASSES:

            folder = OUTPUT_DIR / split / class_name

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    print("Output directories ready.")


# ============================================================
# FIND IMAGES
# ============================================================

def find_images():

    print("\nSearching for tyre images...")

    all_images = []

    for class_name in CLASSES:

        class_dir = SOURCE_DIR / class_name

        if not class_dir.exists():

            print(
                f"WARNING: Missing class folder: {class_dir}"
            )

            continue

        images = []

        for file in class_dir.rglob("*"):

            if (
                file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ):

                images.append(file)

        print(
            f"{class_name}: {len(images)} images found"
        )

        for image in images:

            all_images.append(
                (
                    image,
                    class_name
                )
            )

    return all_images


# ============================================================
# VERIFY IMAGE
# ============================================================

def is_valid_image(image_path):

    try:

        with Image.open(image_path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# VALIDATE IMAGES
# ============================================================

def validate_images(images):

    print("\nValidating images...")

    valid_images = []
    invalid_images = []

    for image_path, class_name in images:

        if is_valid_image(image_path):

            valid_images.append(
                (
                    image_path,
                    class_name
                )
            )

        else:

            invalid_images.append(
                (
                    image_path,
                    class_name
                )
            )

    print(
        f"Valid images: {len(valid_images)}"
    )

    print(
        f"Invalid images: {len(invalid_images)}"
    )

    if invalid_images:

        print("\nInvalid images:")

        for image_path, _ in invalid_images:

            print(
                f"  {image_path}"
            )

    return valid_images


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(images):

    paths = [
        str(item[0])
        for item in images
    ]

    labels = [
        item[1]
        for item in images
    ]

    # --------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_paths, temp_paths, train_labels, temp_labels = (
        train_test_split(
            paths,
            labels,
            test_size=(
                VALIDATION_SIZE + TEST_SIZE
            ),
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    # --------------------------------------------------------
    # Second split:
    #
    # temporary → 50% validation
    #             50% test
    # --------------------------------------------------------

    validation_paths, test_paths, validation_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=temp_labels
        )
    )

    train = list(
        zip(
            train_paths,
            train_labels
        )
    )

    validation = list(
        zip(
            validation_paths,
            validation_labels
        )
    )

    test = list(
        zip(
            test_paths,
            test_labels
        )
    )

    return train, validation, test


# ============================================================
# COPY DATASET
# ============================================================

def copy_split(
    split_name,
    items
):

    print(
        f"\nPreparing {split_name} dataset..."
    )

    counters = {
        class_name: 0
        for class_name in CLASSES
    }

    for source_path, class_name in items:

        source_path = Path(source_path)

        counters[class_name] += 1

        # Keep the original filename but add a unique
        # sequential prefix to prevent filename collisions.

        new_name = (
            f"{counters[class_name]:05d}_"
            f"{source_path.name}"
        )

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
            / new_name
        )

        shutil.copy2(
            source_path,
            destination
        )

    for class_name in CLASSES:

        print(
            f"  {class_name}: "
            f"{counters[class_name]}"
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    train,
    validation,
    test
):

    print("\n")
    print("=" * 60)
    print("TYRE DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Training images:   {len(train)}"
    )

    print(
        f"Validation images: {len(validation)}"
    )

    print(
        f"Test images:       {len(test)}"
    )

    print(
        f"Total images:      "
        f"{len(train) + len(validation) + len(test)}"
    )

    print("\nClasses:")

    for class_name in CLASSES:

        train_count = sum(
            1
            for _, label in train
            if label == class_name
        )

        validation_count = sum(
            1
            for _, label in validation
            if label == class_name
        )

        test_count = sum(
            1
            for _, label in test
            if label == class_name
        )

        print(
            f"\n{class_name}"
        )

        print(
            f"  Train:      {train_count}"
        )

        print(
            f"  Validation: {validation_count}"
        )

        print(
            f"  Test:       {test_count}"
        )

    print("\nOutput location:")

    print(
        OUTPUT_DIR
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("RIDEGUARDIAN DATA PREPARATION")
    print("TYRE CONDITION DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        print(
            "\nERROR:"
        )

        print(
            f"Dataset not found:\n{SOURCE_DIR}"
        )

        print(
            "\nMake sure Tyre_Condition_Dataset "
            "is inside ai/datasets."
        )

        return

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images()

    if not images:

        print(
            "\nERROR: No images found."
        )

        return

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid_images = validate_images(images)

    if len(valid_images) < 10:

        print(
            "\nERROR: Too few valid images."
        )

        return

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_dataset(
        valid_images
    )

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    copy_split(
        "train",
        train
    )

    copy_split(
        "validation",
        validation
    )

    copy_split(
        "test",
        test
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        train,
        validation,
        test
    )


if __name__ == "__main__":

    main()