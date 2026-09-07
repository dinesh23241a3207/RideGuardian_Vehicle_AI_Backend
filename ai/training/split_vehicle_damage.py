import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# RIDEGUARDIAN
# VEHICLE DAMAGE DATASET SPLITTER
#
# Input:
#   ai/datasets/processed/vehicle_photo/damage/all/
#
# Output:
#   ai/datasets/processed/vehicle_damage/
#       train/
#       validation/
#       test/
#
# IMPORTANT:
# Crops originating from the same original image are kept
# inside the SAME split to prevent data leakage.
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "vehicle_photo"
    / "damage"
    / "all"
)

OUTPUT_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "vehicle_damage"
)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# CHECK RATIOS
# ============================================================

ratio_total = (
    TRAIN_RATIO
    +
    VALIDATION_RATIO
    +
    TEST_RATIO
)

if abs(
    ratio_total - 1.0
) > 0.001:

    raise ValueError(
        "Train/validation/test ratios "
        "must add up to 1.0"
    )


# ============================================================
# CLEAN OLD OUTPUT
# ============================================================

def clean_output():

    if OUTPUT_DIR.exists():

        print()
        print(
            "Removing previous vehicle-damage split..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )


# ============================================================
# FIND CLASSES
# ============================================================

def get_classes():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"\nInput directory not found:\n"
            f"{INPUT_DIR}"
        )

    classes = []

    for directory in sorted(
        INPUT_DIR.iterdir()
    ):

        if directory.is_dir():

            classes.append(
                directory.name
            )

    if not classes:

        raise RuntimeError(
            "\nNo damage classes found."
        )

    return classes


# ============================================================
# FIND IMAGES
# ============================================================

def get_images():

    images_by_class = defaultdict(
        list
    )

    for class_name in get_classes():

        class_dir = (
            INPUT_DIR
            / class_name
        )

        for file in class_dir.iterdir():

            if (
                file.is_file()
                and
                file.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images_by_class[
                    class_name
                ].append(file)

    return images_by_class


# ============================================================
# EXTRACT ORIGINAL IMAGE ID
# ============================================================
#
# Example:
#
# Car damages 100_object_001.jpg
#
# becomes:
#
# Car damages 100
#
# Therefore all crops from the same original image
# stay in the same split.
# ============================================================

def get_original_id(
    image_path
):

    name = image_path.stem

    marker = "_object_"

    if marker in name:

        return name.split(
            marker
        )[0]

    return name


# ============================================================
# BUILD GROUPS
# ============================================================

def build_groups(
    images_by_class
):

    groups = defaultdict(
        list
    )

    for class_name, images in (
        images_by_class.items()
    ):

        for image_path in images:

            original_id = (
                get_original_id(
                    image_path
                )
            )

            groups[
                original_id
            ].append(
                (
                    class_name,
                    image_path
                )
            )

    return groups


# ============================================================
# SPLIT GROUPS
# ============================================================

def split_groups(
    groups
):

    group_ids = list(
        groups.keys()
    )

    random.shuffle(
        group_ids
    )

    total_groups = len(
        group_ids
    )

    train_count = int(
        total_groups
        *
        TRAIN_RATIO
    )

    validation_count = int(
        total_groups
        *
        VALIDATION_RATIO
    )

    train_groups = group_ids[
        :train_count
    ]

    validation_groups = group_ids[
        train_count:
        train_count
        +
        validation_count
    ]

    test_groups = group_ids[
        train_count
        +
        validation_count:
    ]

    return (
        train_groups,
        validation_groups,
        test_groups
    )


# ============================================================
# COPY GROUP
# ============================================================

def copy_group(
    group_id,
    groups,
    split_name,
    counters
):

    entries = groups[
        group_id
    ]

    for class_name, image_path in entries:

        destination_dir = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination_path = (
            destination_dir
            / image_path.name
        )

        # ----------------------------------------------------
        # Avoid filename collisions
        # ----------------------------------------------------

        if destination_path.exists():

            base = image_path.stem

            extension = image_path.suffix

            counter = 1

            while destination_path.exists():

                destination_path = (
                    destination_dir
                    /
                    f"{base}_{counter}{extension}"
                )

                counter += 1

        shutil.copy2(
            image_path,
            destination_path
        )

        counters[
            split_name
        ][class_name] += 1


# ============================================================
# COUNT OUTPUT
# ============================================================

def count_output():

    results = {}

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        results[
            split_name
        ] = {}

        split_dir = (
            OUTPUT_DIR
            / split_name
        )

        if not split_dir.exists():

            continue

        for class_dir in sorted(
            split_dir.iterdir()
        ):

            if not class_dir.is_dir():

                continue

            count = 0

            for file in class_dir.iterdir():

                if (
                    file.is_file()
                    and
                    file.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    count += 1

            results[
                split_name
            ][
                class_dir.name
            ] = count

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    print()
    print("=" * 70)
    print(
        "RIDEGUARDIAN VEHICLE DAMAGE DATASET SPLIT"
    )
    print("=" * 70)

    print()
    print(
        "Input:"
    )

    print(
        INPUT_DIR
    )

    print()
    print(
        "Split:"
    )

    print(
        "Train:      70%"
    )

    print(
        "Validation: 15%"
    )

    print(
        "Test:       15%"
    )

    print()
    print(
        "Grouping by original image to prevent data leakage."
    )

    # --------------------------------------------------------
    # Clean old output
    # --------------------------------------------------------

    clean_output()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images_by_class = get_images()

    print()
    print(
        "INPUT DATASET"
    )

    print(
        "-" * 50
    )

    total_images = 0

    for class_name in sorted(
        images_by_class
    ):

        count = len(
            images_by_class[
                class_name
            ]
        )

        total_images += count

        print(
            f"{class_name}: {count}"
        )

    print(
        f"\nTotal images: {total_images}"
    )

    # --------------------------------------------------------
    # Build original-image groups
    # --------------------------------------------------------

    groups = build_groups(
        images_by_class
    )

    print()
    print(
        f"Original image groups: "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_groups,
        validation_groups,
        test_groups
    ) = split_groups(
        groups
    )

    print()
    print(
        "GROUP SPLIT"
    )

    print(
        "-" * 50
    )

    print(
        f"Training groups: "
        f"{len(train_groups)}"
    )

    print(
        f"Validation groups: "
        f"{len(validation_groups)}"
    )

    print(
        f"Test groups: "
        f"{len(test_groups)}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    counters = {

        "train":
            defaultdict(int),

        "validation":
            defaultdict(int),

        "test":
            defaultdict(int)

    }

    # --------------------------------------------------------
    # Copy training
    # --------------------------------------------------------

    print()
    print(
        "Creating training dataset..."
    )

    for group_id in train_groups:

        copy_group(
            group_id,
            groups,
            "train",
            counters
        )

    # --------------------------------------------------------
    # Copy validation
    # --------------------------------------------------------

    print(
        "Creating validation dataset..."
    )

    for group_id in validation_groups:

        copy_group(
            group_id,
            groups,
            "validation",
            counters
        )

    # --------------------------------------------------------
    # Copy test
    # --------------------------------------------------------

    print(
        "Creating test dataset..."
    )

    for group_id in test_groups:

        copy_group(
            group_id,
            groups,
            "test",
            counters
        )

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    results = count_output()

    print()
    print("=" * 70)
    print(
        "FINAL DATASET DISTRIBUTION"
    )
    print("=" * 70)

    grand_total = 0

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print()
        print(
            split_name.upper()
        )

        print(
            "-" * 50
        )

        split_total = 0

        for class_name, count in (
            results[
                split_name
            ].items()
        ):

            print(
                f"{class_name}: {count}"
            )

            split_total += count

        print(
            f"TOTAL: {split_total}"
        )

        grand_total += split_total

    print()
    print(
        f"GRAND TOTAL: {grand_total}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print(
        "VERIFICATION"
    )

    print(
        "-" * 50
    )

    if grand_total != total_images:

        print(
            "WARNING: Image count mismatch!"
        )

        print(
            f"Input: {total_images}"
        )

        print(
            f"Output: {grand_total}"
        )

    else:

        print(
            "All images accounted for: YES"
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "input_directory":
            str(INPUT_DIR),

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "random_seed":
            RANDOM_SEED,

        "grouped_by_original_image":
            True,

        "total_input_images":
            total_images,

        "total_output_images":
            grand_total,

        "classes":
            sorted(
                images_by_class.keys()
            ),

        "distribution":
            results

    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = (
        OUTPUT_DIR
        / "split_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VEHICLE DAMAGE DATASET SPLIT COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Metadata:"
    )

    print(
        metadata_path
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()