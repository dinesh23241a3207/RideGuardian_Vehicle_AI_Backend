import os
from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent


MODELS = {
    "vehicle_parts": {
        "path": (
            BASE_DIR
            / "ai"
            / "models"
            / "vehicle_parts"
            / "best_vehicle_parts_coatnet.pth"
        ),
        "url": (
            "https://media.githubusercontent.com/media/"
            "dinesh23241a3207/RideGuardian_Vehicle_AI_Backend/"
            "main/ai/models/vehicle_parts/"
            "best_vehicle_parts_coatnet.pth"
        ),
        "min_size": 50 * 1024 * 1024,
    },

    "damage_condition": {
        "path": (
            BASE_DIR
            / "ai"
            / "models"
            / "damage_condition"
            / "best_damage_coatnet.pth"
        ),
        "url": (
            "https://media.githubusercontent.com/media/"
            "dinesh23241a3207/RideGuardian_Vehicle_AI_Backend/"
            "main/ai/models/damage_condition/"
            "best_damage_coatnet.pth"
        ),
        "min_size": 50 * 1024 * 1024,
    },
}


def is_valid_model(file_path: Path, minimum_size: int) -> bool:

    if not file_path.exists():
        return False

    if file_path.stat().st_size < minimum_size:
        return False

    try:

        with open(file_path, "rb") as file:

            first_bytes = file.read(100)

        if b"git-lfs.github.com" in first_bytes:
            return False

    except Exception:
        return False

    return True


def download_model(
    model_name: str,
    model_path: Path,
    model_url: str
):

    print(f"Downloading {model_name} model...")

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = Path(
        str(model_path) + ".download"
    )

    try:

        response = requests.get(
            model_url,
            stream=True,
            timeout=600
        )

        response.raise_for_status()

        with open(
            temporary_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    file.write(chunk)

        if temporary_path.stat().st_size < 50 * 1024 * 1024:

            raise RuntimeError(
                "Downloaded model file is too small. "
                "The Git LFS binary may not have been downloaded."
            )

        os.replace(
            temporary_path,
            model_path
        )

        print(
            f"{model_name} model downloaded successfully"
        )

        print(
            f"Size: "
            f"{round(model_path.stat().st_size / 1024 / 1024, 2)} MB"
        )

    except Exception:

        if temporary_path.exists():

            temporary_path.unlink()

        raise


def ensure_models():

    print(
        "Checking Vehicle AI models..."
    )

    for model_name, config in MODELS.items():

        model_path = config["path"]

        if is_valid_model(
            model_path,
            config["min_size"]
        ):

            print(
                f"{model_name} model already available"
            )

            continue

        print(
            f"{model_name} model missing or invalid."
        )

        download_model(
            model_name,
            model_path,
            config["url"]
        )

    print(
        "All required Vehicle AI models are ready."
    )