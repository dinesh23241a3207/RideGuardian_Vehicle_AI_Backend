import json
import random
import shutil
from pathlib import Path


# ============================================================
# RIDEGUARDIAN BODY AI DATASET PREPARATION
# ============================================================
#
# SOURCE DATASETS:
#
# 1. Car parts dataset
#    -> ACTUAL CONTENT = DAMAGE ANNOTATIONS
#
# 2. Car damages dataset
#    -> ACTUAL CONTENT = VEHICLE PART ANNOTATIONS
#
# FORMAT:
#   JSON annotations
#   objects[].classTitle
#   objects[].geometryType
#   objects[].points.exterior
#
# OUTPUT:
#   YOLO SEGMENTATION DATASETS
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)

DATASET_ROOT = (
    PROJECT_ROOT / "ai" / "datasets"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "ai" / "body_yolo"
)


# ============================================================
# SOURCE DATASETS
# ============================================================

DAMAGE_SOURCE = (
    DATASET_ROOT
    / "Car parts dataset"
    / "File1"
)

PART_SOURCE = (
    DATASET_ROOT
    / "Car damages dataset"
    / "File1"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

DAMAGE_CLASSES = [
    "Broken part",
    "Corrosion",
    "Cracked",
    "Dent",
    "Flaking",
    "Missing part",
    "Paint chip",
    "Scratch",
]


PART_CLASSES = [
    "Back-bumper",
    "Back-door",
    "Back-wheel",
    "Back-window",
    "Back-windshield",
    "Fender",
    "Front-bumper",
    "Front-door",
    "Front-wheel",
    "Front-window",
    "Grille",
    "Headlight",
    "Hood",
    "License-plate",
    "Mirror",
    "Quarter-panel",
    "Rocker-panel",
    "Roof",
    "Tail-light",
    "Trunk",
    "Windshield",
]


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UTILITY
# ============================================================

def find_image(img_dir, json_path):
    """
    Find the image corresponding to a JSON annotation.

    Example:

        Car damages 101.png.json

    becomes:

        Car damages 101.png
    """

    original_name = json_path.name[:-5]

    exact_path = img_dir / original_name

    if exact_path.exists():
        return exact_path

    # Fallback: search by stem
    for image_path in img_dir.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.name == original_name:
            return image_path

    return None


# ============================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================

def create_output_structure(output_dir):

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            output_dir
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            output_dir
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# WRITE DATA YAML
# ============================================================

def write_data_yaml(
    output_dir,
    class_names
):

    yaml_path = (
        output_dir / "data.yaml"
    )

    lines = []

    lines.append(
        f"path: {output_dir.as_posix()}"
    )

    lines.append(
        "train: images/train"
    )

    lines.append(
        "val: images/val"
    )

    lines.append(
        "test: images/test"
    )

    lines.append(
        f"nc: {len(class_names)}"
    )

    lines.append("names:")

    for index, name in enumerate(
        class_names
    ):

        lines.append(
            f"  {index}: '{name}'"
        )

    yaml_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# CONVERT POLYGON
# ============================================================

def convert_polygon(
    polygon,
    width,
    height
):

    if not polygon:
        return None

    # YOLO segmentation requires
    # normalized x/y coordinates.

    normalized = []

    for point in polygon:

        if len(point) < 2:
            continue

        x = float(point[0])
        y = float(point[1])

        # Keep coordinates inside image bounds.
        x = max(
            0.0,
            min(x, width)
        )

        y = max(
            0.0,
            min(y, height)
        )

        x_norm = x / width
        y_norm = y / height

        normalized.append(
            x_norm
        )

        normalized.append(
            y_norm
        )

    # Polygon needs at least 3 points.

    if len(normalized) < 6:
        return None

    return normalized


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_annotation(
    json_path,
    image_path,
    output_image_path,
    output_label_path,
    class_names
):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception as error:

        print(
            f"[ERROR] JSON: {json_path.name}"
        )

        print(error)

        return False, 0


    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    size = data.get(
        "size",
        {}
    )

    width = size.get(
        "width"
    )

    height = size.get(
        "height"
    )


    if not width or not height:

        print(
            f"[SKIP] Missing image size: "
            f"{json_path.name}"
        )

        return False, 0


    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    objects = data.get(
        "objects",
        []
    )


    if not objects:

        print(
            f"[SKIP] No objects: "
            f"{json_path.name}"
        )

        return False, 0


    yolo_lines = []

    object_count = 0


    # --------------------------------------------------------
    # PROCESS EACH OBJECT
    # --------------------------------------------------------

    for obj in objects:

        class_title = obj.get(
            "classTitle"
        )

        if class_title not in class_names:

            # Ignore labels outside
            # our selected classes.

            continue


        geometry_type = obj.get(
            "geometryType"
        )

        if geometry_type != "polygon":

            continue


        points = obj.get(
            "points",
            {}
        )

        exterior = points.get(
            "exterior",
            []
        )


        polygon = convert_polygon(
            exterior,
            width,
            height
        )


        if polygon is None:

            continue


        class_id = class_names.index(
            class_title
        )


        coordinates = " ".join(
            f"{value:.6f}"
            for value in polygon
        )


        line = (
            f"{class_id} "
            f"{coordinates}"
        )


        yolo_lines.append(
            line
        )

        object_count += 1


    # --------------------------------------------------------
    # DON'T COPY IMAGES WITH NO VALID LABELS
    # --------------------------------------------------------

    if not yolo_lines:

        print(
            f"[SKIP] No valid polygons: "
            f"{json_path.name}"
        )

        return False, 0


    # --------------------------------------------------------
    # COPY IMAGE
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        output_image_path
    )


    # --------------------------------------------------------
    # WRITE LABEL
    # --------------------------------------------------------

    output_label_path.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )


    return True, object_count


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    source_dir,
    output_dir,
    class_names
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING: {dataset_name}"
    )

    print("=" * 70)

    print(
        f"Source: {source_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    print(
        f"Classes: {len(class_names)}"
    )


    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    img_dir = (
        source_dir / "img"
    )

    ann_dir = (
        source_dir / "ann"
    )


    if not img_dir.exists():

        print(
            "[ERROR] Image directory "
            "not found:"
        )

        print(img_dir)

        return


    if not ann_dir.exists():

        print(
            "[ERROR] Annotation directory "
            "not found:"
        )

        print(ann_dir)

        return


    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    create_output_structure(
        output_dir
    )


    # --------------------------------------------------------
    # WRITE YAML
    # --------------------------------------------------------

    write_data_yaml(
        output_dir,
        class_names
    )


    # --------------------------------------------------------
    # GET JSON FILES
    # --------------------------------------------------------

    json_files = sorted(
        ann_dir.glob("*.json")
    )


    print(
        f"JSON files found: "
        f"{len(json_files)}"
    )


    # --------------------------------------------------------
    # MATCH JSON WITH IMAGES
    # --------------------------------------------------------

    pairs = []

    missing_images = 0


    for json_path in json_files:

        image_path = find_image(
            img_dir,
            json_path
        )


        if image_path is None:

            missing_images += 1

            continue


        pairs.append(
            (
                json_path,
                image_path
            )
        )


    print(
        f"Matched image/annotation pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Missing images: "
        f"{missing_images}"
    )


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(
        pairs
    )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    total = len(pairs)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(total * VAL_RATIO)
    )


    splits = {

        "train":
            pairs[:train_end],

        "val":
            pairs[
                train_end:val_end
            ],

        "test":
            pairs[val_end:],
    }


    print()
    print("Split:")

    print(
        f"Train: {len(splits['train'])}"
    )

    print(
        f"Val:   {len(splits['val'])}"
    )

    print(
        f"Test:  {len(splits['test'])}"
    )


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    successful = 0
    skipped = 0
    total_objects = 0


    for split_name, split_pairs in splits.items():

        print()
        print(
            f"Preparing {split_name}..."
        )


        image_output_dir = (
            output_dir
            / "images"
            / split_name
        )

        label_output_dir = (
            output_dir
            / "labels"
            / split_name
        )


        for json_path, image_path in split_pairs:

            output_image_path = (
                image_output_dir
                / image_path.name
            )

            output_label_path = (
                label_output_dir
                / f"{image_path.stem}.txt"
            )


            success, object_count = (
                process_annotation(
                    json_path,
                    image_path,
                    output_image_path,
                    output_label_path,
                    class_names
                )
            )


            if success:

                successful += 1

                total_objects += (
                    object_count
                )

            else:

                skipped += 1


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"{dataset_name} COMPLETE"
    )

    print("-" * 70)

    print(
        f"Successful images: "
        f"{successful}"
    )

    print(
        f"Skipped images: "
        f"{skipped}"
    )

    print(
        f"Total polygon objects: "
        f"{total_objects}"
    )

    print(
        f"Output: "
        f"{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RIDEGUARDIAN BODY AI DATASET PREPARATION")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Car parts dataset -> DAMAGE"
    )

    print(
        "Car damages dataset -> VEHICLE PARTS"
    )


    # ========================================================
    # DAMAGE DATASET
    # ========================================================

    damage_output = (
        OUTPUT_ROOT
        / "damage"
    )


    process_dataset(
        dataset_name="DAMAGE AI",
        source_dir=DAMAGE_SOURCE,
        output_dir=damage_output,
        class_names=DAMAGE_CLASSES
    )


    # ========================================================
    # VEHICLE PART DATASET
    # ========================================================

    parts_output = (
        OUTPUT_ROOT
        / "parts"
    )


    process_dataset(
        dataset_name="VEHICLE PARTS AI",
        source_dir=PART_SOURCE,
        output_dir=parts_output,
        class_names=PART_CLASSES
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("BODY AI DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated datasets:")

    print(
        OUTPUT_ROOT / "damage"
    )

    print(
        OUTPUT_ROOT / "parts"
    )

    print()
    print(
        "Next step: inspect the generated "
        "dataset counts before training."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()