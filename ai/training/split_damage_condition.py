import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# RIDEGUARDIAN
# DAMAGE CONDITION DATASET SPLITTER
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
# Crops from the same original image remain in the same split.
# ============================================================


AI_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "vehicle_photo"
    / "parts"
    / "all"
)

OUTPUT_DIR = (
    AI_DIR
    / "datasets"
    / "processed"
    / "vehicle_damage_condition"
)


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def get_original_id(image_path):

    name = image_path.stem

    marker = "_object_"

    if marker in name:

        return name.split(
            marker
        )[0]

    return name


def get_images():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    images_by_class = defaultdict(list)

    for class_dir in sorted(
        INPUT_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        for image in class_dir.iterdir():

            if (
                image.is_file()
                and
                image.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images_by_class[
                    class_dir.name
                ].append(image)

    return images_by_class


def build_groups(images_by_class):

    groups = defaultdict(list)

    for class_name, images in (
        images_by_class.items()
    ):

        for image in images:

            original_id = get_original_id(
                image
            )

            groups[
                original_id
            ].append(
                (
                    class_name,
                    image
                )
            )

    return groups


def split_groups(groups):

    ids = list(
        groups.keys()
    )

    random.shuffle(ids)

    total = len(ids)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    train = ids[
        :train_count
    ]

    validation = ids[
        train_count:
        train_count + validation_count
    ]

    test = ids[
        train_count + validation_count:
    ]

    return train, validation, test


def copy_group(
    group_id,
    groups,
    split_name
):

    for class_name, image in groups[
        group_id
    ]:

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        output = (
            destination
            / image.name
        )

        if output.exists():

            counter = 1

            while output.exists():

                output = (
                    destination
                    /
                    f"{image.stem}_{counter}"
                    f"{image.suffix}"
                )

                counter += 1

        shutil.copy2(
            image,
            output
        )


def count_images():

    result = {}

    for split in [
        "train",
        "validation",
        "test"
    ]:

        result[split] = {}

        split_dir = (
            OUTPUT_DIR
            / split
        )

        if not split_dir.exists():
            continue

        for class_dir in sorted(
            split_dir.iterdir()
        ):

            if not class_dir.is_dir():
                continue

            count = sum(
                1
                for file in class_dir.iterdir()
                if (
                    file.is_file()
                    and
                    file.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

            result[
                split
            ][
                class_dir.name
            ] = count

    return result


def main():

    random.seed(
        RANDOM_SEED
    )

    print()
    print("=" * 70)
    print(
        "RIDEGUARDIAN DAMAGE CONDITION DATASET SPLIT"
    )
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_DIR)

    print()
    print(
        "This is the actual DAMAGE dataset."
    )

    print()

    if OUTPUT_DIR.exists():

        print(
            "Removing previous damage-condition split..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    images_by_class = get_images()

    print()
    print("INPUT CLASSES")
    print("-" * 50)

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

    print()
    print(
        f"Total input images: {total_images}"
    )

    groups = build_groups(
        images_by_class
    )

    print()
    print(
        f"Original image groups: {len(groups)}"
    )

    train, validation, test = (
        split_groups(groups)
    )

    print()
    print("GROUP SPLIT")
    print("-" * 50)

    print(
        f"Train groups: {len(train)}"
    )

    print(
        f"Validation groups: {len(validation)}"
    )

    print(
        f"Test groups: {len(test)}"
    )

    print()
    print(
        "Creating training dataset..."
    )

    for group in train:

        copy_group(
            group,
            groups,
            "train"
        )

    print(
        "Creating validation dataset..."
    )

    for group in validation:

        copy_group(
            group,
            groups,
            "validation"
        )

    print(
        "Creating test dataset..."
    )

    for group in test:

        copy_group(
            group,
            groups,
            "test"
        )

    results = count_images()

    print()
    print("=" * 70)
    print(
        "FINAL DAMAGE DATASET DISTRIBUTION"
    )
    print("=" * 70)

    grand_total = 0

    for split in [
        "train",
        "validation",
        "test"
    ]:

        print()
        print(
            split.upper()
        )

        print("-" * 50)

        split_total = 0

        for class_name, count in (
            results[split].items()
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

    print()
    print("VERIFICATION")
    print("-" * 50)

    if grand_total == total_images:

        print(
            "All images accounted for: YES"
        )

    else:

        print(
            "WARNING: Image count mismatch!"
        )

        print(
            f"Input: {total_images}"
        )

        print(
            f"Output: {grand_total}"
        )

    metadata = {

        "input": str(INPUT_DIR),

        "output": str(OUTPUT_DIR),

        "classes": sorted(
            images_by_class.keys()
        ),

        "total_images":
            total_images,

        "grouped_by_original_image":
            True,

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "random_seed":
            RANDOM_SEED,

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

    print()
    print("=" * 70)
    print(
        "DAMAGE CONDITION SPLIT COMPLETE"
    )
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("Metadata:")
    print(metadata_path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()