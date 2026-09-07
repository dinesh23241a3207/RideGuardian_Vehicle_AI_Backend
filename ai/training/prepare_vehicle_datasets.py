import json
from pathlib import Path
from collections import defaultdict

from PIL import Image


# ============================================================
# RIDEGUARDIAN
# VEHICLE PHOTO DATASET PREPARATION
#
# Correct structure:
#
# Car damages dataset/
#   File1/
#       img/
#           Car damages 100.png
#           Car damages 101.png
#           ...
#
#       ann/
#           Car damages 100.png.json
#           Car damages 101.png.json
#           ...
#
# The JSON filename contains the COMPLETE image filename.
#
# Example:
#
# Image:
#   Car damages 100.png
#
# Annotation:
#   Car damages 100.png.json
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

DATASETS_DIR = AI_DIR / "datasets"

CAR_PARTS_DIR = (
    DATASETS_DIR
    / "Car parts dataset"
)

CAR_DAMAGES_DIR = (
    DATASETS_DIR
    / "Car damages dataset"
)

OUTPUT_DIR = (
    DATASETS_DIR
    / "processed"
    / "vehicle_photo"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

MIN_CROP_WIDTH = 40

MIN_CROP_HEIGHT = 40

MIN_CROP_AREA = 1600

PADDING_RATIO = 0.10


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    folders = [

        OUTPUT_DIR
        / "damage"
        / "all",

        OUTPUT_DIR
        / "parts"
        / "all"

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(
    directory
):

    images = []

    if not directory.exists():

        return images

    for file in directory.rglob("*"):

        if not file.is_file():

            continue

        if (
            file.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(file)

    return images


# ============================================================
# FIND JSON FILES
# ============================================================

def find_json_files(
    directory
):

    json_files = []

    if not directory.exists():

        return json_files

    for file in directory.rglob("*.json"):

        if file.is_file():

            json_files.append(file)

    return json_files


# ============================================================
# BUILD EXACT IMAGE → JSON MAP
# ============================================================

def build_annotation_map(
    annotation_files
):

    annotation_map = {}

    for annotation_file in annotation_files:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # "Car damages 100.png.json"
        #
        # remove ONLY the final ".json"
        #
        # resulting name:
        #
        # "Car damages 100.png"
        # ----------------------------------------------------

        image_filename = (
            annotation_file.name[
                :-len(".json")
            ]
        )

        annotation_map[
            image_filename.lower()
        ] = annotation_file

    return annotation_map


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    annotation_file
):

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        print()
        print(
            "WARNING: Could not read:"
        )

        print(
            annotation_file
        )

        print(
            error
        )

        return None


# ============================================================
# CLEAN LABEL
# ============================================================

def clean_label(
    label
):

    if label is None:

        return None

    label = str(
        label
    ).strip()

    if not label:

        return None

    label = (
        label
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )

    return label


# ============================================================
# EXTRACT POLYGON POINTS
# ============================================================

def get_polygon_points(
    obj
):

    points = obj.get(
        "points"
    )

    if not isinstance(
        points,
        dict
    ):

        return []

    exterior = points.get(
        "exterior"
    )

    if not isinstance(
        exterior,
        list
    ):

        return []

    result = []

    for point in exterior:

        if (
            not isinstance(
                point,
                list
            )
            or
            len(point) < 2
        ):

            continue

        try:

            x = float(
                point[0]
            )

            y = float(
                point[1]
            )

            result.append(
                (x, y)
            )

        except (
            ValueError,
            TypeError
        ):

            continue

    return result


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bounding_box(
    points,
    image_width,
    image_height
):

    if not points:

        return None

    xs = [
        point[0]
        for point in points
    ]

    ys = [
        point[1]
        for point in points
    ]

    min_x = min(xs)

    max_x = max(xs)

    min_y = min(ys)

    max_y = max(ys)

    width = max_x - min_x

    height = max_y - min_y

    if width <= 0:

        return None

    if height <= 0:

        return None

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    padding_x = (
        width
        * PADDING_RATIO
    )

    padding_y = (
        height
        * PADDING_RATIO
    )

    min_x -= padding_x

    max_x += padding_x

    min_y -= padding_y

    max_y += padding_y

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    min_x = max(
        0,
        int(min_x)
    )

    min_y = max(
        0,
        int(min_y)
    )

    max_x = min(
        image_width,
        int(max_x)
    )

    max_y = min(
        image_height,
        int(max_y)
    )

    crop_width = (
        max_x
        -
        min_x
    )

    crop_height = (
        max_y
        -
        min_y
    )

    if (
        crop_width
        <
        MIN_CROP_WIDTH
    ):

        return None

    if (
        crop_height
        <
        MIN_CROP_HEIGHT
    ):

        return None

    if (
        crop_width
        *
        crop_height
        <
        MIN_CROP_AREA
    ):

        return None

    return (
        min_x,
        min_y,
        max_x,
        max_y
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    image_directory,
    annotation_directory,
    output_type
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING: {dataset_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images(
        image_directory
    )

    # --------------------------------------------------------
    # Find annotations
    # --------------------------------------------------------

    annotations = find_json_files(
        annotation_directory
    )

    print()
    print(
        f"Images found: "
        f"{len(images)}"
    )

    print(
        f"Annotations found: "
        f"{len(annotations)}"
    )

    # --------------------------------------------------------
    # Build exact map
    # --------------------------------------------------------

    annotation_map = build_annotation_map(
        annotations
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_root = (
        OUTPUT_DIR
        / output_type
        / "all"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    matched = 0

    unmatched = 0

    objects_found = 0

    crops_created = 0

    skipped = 0

    class_counts = defaultdict(
        int
    )

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(
        images,
        start=1
    ):

        # ----------------------------------------------------
        # Exact filename matching
        # ----------------------------------------------------

        image_filename = (
            image_path.name
        )

        annotation_path = (
            annotation_map.get(
                image_filename.lower()
            )
        )

        # ----------------------------------------------------
        # No annotation
        # ----------------------------------------------------

        if annotation_path is None:

            unmatched += 1

            continue

        matched += 1

        # ----------------------------------------------------
        # Load annotation
        # ----------------------------------------------------

        data = load_json(
            annotation_path
        )

        if data is None:

            skipped += 1

            continue

        # ----------------------------------------------------
        # Objects
        # ----------------------------------------------------

        objects = data.get(
            "objects",
            []
        )

        if not isinstance(
            objects,
            list
        ):

            skipped += 1

            continue

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as error:

            print()
            print(
                "WARNING: Cannot open image:"
            )

            print(
                image_path
            )

            print(
                error
            )

            skipped += 1

            continue

        image_width, image_height = (
            image.size
        )

        image_crop_count = 0

        # ====================================================
        # PROCESS OBJECTS
        # ====================================================

        for object_index, obj in enumerate(
            objects
        ):

            if not isinstance(
                obj,
                dict
            ):

                continue

            label = clean_label(
                obj.get(
                    "classTitle"
                )
            )

            if label is None:

                continue

            points = get_polygon_points(
                obj
            )

            if not points:

                continue

            bbox = get_bounding_box(
                points,
                image_width,
                image_height
            )

            if bbox is None:

                continue

            try:

                crop = image.crop(
                    bbox
                )

            except Exception:

                continue

            # ------------------------------------------------
            # Class folder
            # ------------------------------------------------

            class_directory = (
                output_root
                / label
            )

            class_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_filename = (
                f"{image_path.stem}"
                f"_object_{object_index:03d}"
                ".jpg"
            )

            output_path = (
                class_directory
                / output_filename
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            try:

                crop.save(
                    output_path,
                    "JPEG",
                    quality=95
                )

            except Exception:

                continue

            objects_found += 1

            crops_created += 1

            image_crop_count += 1

            class_counts[
                label
            ] += 1

        if image_crop_count == 0:

            skipped += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 250 == 0
            or
            index == len(images)
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(images)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        "RESULTS"
    )

    print(
        "-" * 50
    )

    print(
        f"Images matched: "
        f"{matched}"
    )

    print(
        f"Images unmatched: "
        f"{unmatched}"
    )

    print(
        f"Objects found: "
        f"{objects_found}"
    )

    print(
        f"Crops created: "
        f"{crops_created}"
    )

    print(
        f"Skipped: "
        f"{skipped}"
    )

    print()
    print(
        "Classes:"
    )

    for label, count in sorted(
        class_counts.items()
    ):

        print(
            f"  {label}: {count}"
        )

    return {

        "images":
            len(images),

        "annotations":
            len(annotations),

        "matched":
            matched,

        "unmatched":
            unmatched,

        "objects":
            objects_found,

        "crops":
            crops_created,

        "skipped":
            skipped,

        "classes":
            dict(class_counts)

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RIDEGUARDIAN VEHICLE DATASET PREPARATION")
    print("=" * 70)

    print()
    print(
        "Original datasets will NOT be modified."
    )

    print(
        "Preparing cropped vehicle-photo datasets..."
    )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    create_directories()

    # ========================================================
    # CAR DAMAGES DATASET
    # ========================================================

    damage_stats = process_dataset(

        dataset_name="Car Damages Dataset",

        image_directory=(
            CAR_DAMAGES_DIR
            / "File1"
            / "img"
        ),

        annotation_directory=(
            CAR_DAMAGES_DIR
            / "File1"
            / "ann"
        ),

        output_type="damage"

    )

    # ========================================================
    # CAR PARTS DATASET
    #
    # We use recursive searching here because its structure
    # was different from Car Damages Dataset.
    # ========================================================

    parts_stats = process_dataset(

        dataset_name="Car Parts Dataset",

        image_directory=CAR_PARTS_DIR,

        annotation_directory=CAR_PARTS_DIR,

        output_type="parts"

    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "damage_dataset":
            damage_stats,

        "parts_dataset":
            parts_stats

    }

    summary_path = (
        OUTPUT_DIR
        / "preparation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("VEHICLE DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Damage output:"
    )

    print(
        OUTPUT_DIR
        / "damage"
        / "all"
    )

    print()
    print(
        "Parts output:"
    )

    print(
        OUTPUT_DIR
        / "parts"
        / "all"
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()