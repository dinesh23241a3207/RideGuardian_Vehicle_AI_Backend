from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_ANALYSIS_DIR = Path(
    r"C:\RIDER_SYSTEM\RideGuardian\ai\models"
    r"\tyre_condition_v3\error_analysis"
)

OUTPUT_DIR = (
    ERROR_ANALYSIS_DIR
    / "contact_sheets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

THUMBNAIL_WIDTH = 300
THUMBNAIL_HEIGHT = 240

COLUMNS = 3

PADDING = 20

TEXT_HEIGHT = 80

BACKGROUND_COLOR = (
    30,
    30,
    30
)


# ============================================================
# LOAD FONT
# ============================================================

try:

    font = ImageFont.truetype(
        "arial.ttf",
        16
    )

except:

    font = ImageFont.load_default()


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    source_folder
):

    source_folder = Path(
        source_folder
    )


    images = []


    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
        "*.bmp"
    ]:

        images.extend(
            source_folder.glob(
                extension
            )
        )


    if not images:

        print()

        print(
            f"No images found in:"
        )

        print(
            source_folder
        )

        return


    print()

    print(
        "=" * 70
    )

    print(
        f"CREATING CONTACT SHEET"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Folder: {source_folder.name}"
    )

    print(
        f"Images: {len(images)}"
    )


    rows = math.ceil(

        len(images)
        /
        COLUMNS

    )


    cell_width = (

        THUMBNAIL_WIDTH
        +
        PADDING * 2

    )


    cell_height = (

        THUMBNAIL_HEIGHT
        +
        TEXT_HEIGHT
        +
        PADDING * 2

    )


    sheet_width = (

        COLUMNS
        *
        cell_width

    )


    sheet_height = (

        rows
        *
        cell_height

    )


    sheet = Image.new(

        "RGB",

        (
            sheet_width,
            sheet_height
        ),

        BACKGROUND_COLOR

    )


    draw = ImageDraw.Draw(
        sheet
    )


    for index, image_path in enumerate(
        images
    ):


        row = (

            index
            //
            COLUMNS

        )


        column = (

            index
            %
            COLUMNS

        )


        x = (

            column
            *
            cell_width

            +
            PADDING

        )


        y = (

            row
            *
            cell_height

            +
            PADDING

        )


        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )


            image.thumbnail(

                (
                    THUMBNAIL_WIDTH,
                    THUMBNAIL_HEIGHT
                )

            )


            image_x = (

                x
                +
                (
                    THUMBNAIL_WIDTH
                    -
                    image.width
                )
                //
                2

            )


            image_y = (

                y
                +
                (
                    THUMBNAIL_HEIGHT
                    -
                    image.height
                )
                //
                2

            )


            sheet.paste(

                image,

                (
                    image_x,
                    image_y
                )

            )


            filename = (
                image_path.name
            )


            # Extract confidence from filename
            if filename.startswith(
                "confidence_"
            ):

                parts = filename.split(
                    "_",
                    2
                )


                confidence = parts[1]

                original_name = (
                    parts[2]
                )


                label = (

                    f"Confidence: "
                    f"{float(confidence) * 100:.2f}%"

                )

            else:

                original_name = filename

                label = ""


            # Limit long filenames
            if len(original_name) > 35:

                original_name = (

                    original_name[:32]
                    +
                    "..."

                )


            text_y = (

                y
                +
                THUMBNAIL_HEIGHT
                +
                10

            )


            draw.text(

                (
                    x,
                    text_y
                ),

                original_name,

                fill="white",

                font=font

            )


            draw.text(

                (
                    x,
                    text_y + 25
                ),

                label,

                fill="yellow",

                font=font

            )


        except Exception as error:

            print()

            print(

                f"Could not process: "
                f"{image_path.name}"

            )

            print(
                error
            )


    output_path = (

        OUTPUT_DIR

        /

        f"{source_folder.name}"
        f"_contact_sheet.jpg"

    )


    sheet.save(

        output_path,

        quality=95

    )


    print()

    print(
        f"Saved:"
    )

    print(
        output_path
    )


# ============================================================
# SAFETY CRITICAL FOLDERS
# ============================================================

critical_folders = [

    ERROR_ANALYSIS_DIR
    /
    "UNUSABLE_predicted_as_NEW",

    ERROR_ANALYSIS_DIR
    /
    "UNUSABLE_predicted_as_SERVICEABLE"

]


# ============================================================
# RUN
# ============================================================

print()

print(
    "=" * 70
)

print(
    "RIDEGUARDIAN TYRE ERROR CONTACT SHEETS"
)

print(
    "=" * 70
)


for folder in critical_folders:

    if folder.exists():

        create_contact_sheet(
            folder
        )

    else:

        print()

        print(
            f"Folder not found:"
        )

        print(
            folder
        )


print()

print(
    "=" * 70
)

print(
    "CONTACT SHEET CREATION COMPLETE"
)

print(
    "=" * 70
)

print()

print(
    f"Output directory:"
)

print(
    OUTPUT_DIR
)