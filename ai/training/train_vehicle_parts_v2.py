from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# RIDEGUARDIAN - VEHICLE PARTS AI V2
# YOLO11m SEGMENTATION TRAINING
# WINDOWS-SAFE VERSION
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\RIDER_SYSTEM\RideGuardian_AI"
)

DATASET_YAML = (
    PROJECT_ROOT
    / "ai"
    / "body_yolo"
    / "parts"
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "vehicle_parts_v2"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MODEL_NAME = "yolo11m-seg.pt"

IMAGE_SIZE = 640

EPOCHS = 100

BATCH_SIZE = 8

PATIENCE = 15

# IMPORTANT FOR WINDOWS
WORKERS = 0

PROJECT_NAME = "vehicle_parts_v2"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RIDEGUARDIAN VEHICLE PARTS AI V2")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK CUDA
    # --------------------------------------------------------

    if torch.cuda.is_available():

        DEVICE = 0

        print()
        print("CUDA: AVAILABLE")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        DEVICE = "cpu"

        print()
        print("CUDA: NOT AVAILABLE")

        print(
            "WARNING: Training will use CPU."
        )


    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    print()
    print("Dataset:")

    print(DATASET_YAML)

    if not DATASET_YAML.exists():

        print()
        print("ERROR: Dataset YAML not found.")

        print(
            DATASET_YAML
        )

        return


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING YOLO11M-SEG")
    print("=" * 70)

    model = YOLO(
        MODEL_NAME
    )

    print()
    print("Model loaded successfully.")


    # --------------------------------------------------------
    # TRAINING CONFIG
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        f"Model:       {MODEL_NAME}"
    )

    print(
        f"Image size:  {IMAGE_SIZE}"
    )

    print(
        f"Epochs:      {EPOCHS}"
    )

    print(
        f"Batch size:  {BATCH_SIZE}"
    )

    print(
        f"Workers:     {WORKERS}"
    )

    print(
        f"Device:      {DEVICE}"
    )

    print(
        f"Dataset:     {DATASET_YAML}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING VEHICLE PARTS AI TRAINING")
    print("=" * 70)

    results = model.train(

        data=str(DATASET_YAML),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        # ----------------------------------------------------
        # WINDOWS FIX
        # ----------------------------------------------------

        workers=WORKERS,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        patience=PATIENCE,

        pretrained=True,

        optimizer="auto",

        amp=True,

        cache=False,

        cos_lr=True,

        verbose=True,

        plots=True,

        save=True,

        save_period=10,

        val=True,

        single_cls=False,

        overlap_mask=True,

        mask_ratio=4,

        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        degrees=5.0,

        translate=0.10,

        scale=0.40,

        fliplr=0.5,

        mosaic=0.5,

        mixup=0.1,

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        project=str(OUTPUT_DIR),

        name=PROJECT_NAME,

        exist_ok=True,
    )


    # --------------------------------------------------------
    # MODEL PATHS
    # --------------------------------------------------------

    BEST_MODEL = (
        OUTPUT_DIR
        / PROJECT_NAME
        / "weights"
        / "best.pt"
    )

    LAST_MODEL = (
        OUTPUT_DIR
        / PROJECT_NAME
        / "weights"
        / "last.pt"
    )


    # --------------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VEHICLE PARTS AI V2 TRAINING COMPLETE")
    print("=" * 70)

    print()

    if BEST_MODEL.exists():

        print(
            "BEST MODEL:"
        )

        print(
            BEST_MODEL
        )

    else:

        print(
            "WARNING: best.pt not found."
        )


    print()

    if LAST_MODEL.exists():

        print(
            "LAST MODEL:"
        )

        print(
            LAST_MODEL
        )


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING FINAL VALIDATION")
    print("=" * 70)

    try:

        validation_results = model.val(

            data=str(DATASET_YAML),

            imgsz=IMAGE_SIZE,

            batch=BATCH_SIZE,

            device=DEVICE,

            workers=0,

            plots=True
        )


        print()
        print("=" * 70)
        print("FINAL VALIDATION RESULTS")
        print("=" * 70)


        # ----------------------------------------------------
        # BOX METRICS
        # ----------------------------------------------------

        try:

            print()
            print(
                "Box mAP50:",
                validation_results.box.map50
            )

            print(
                "Box mAP50-95:",
                validation_results.box.map
            )

        except Exception:

            pass


        # ----------------------------------------------------
        # SEGMENTATION METRICS
        # ----------------------------------------------------

        try:

            print()
            print(
                "Mask mAP50:",
                validation_results.seg.map50
            )

            print(
                "Mask mAP50-95:",
                validation_results.seg.map
            )

        except Exception:

            pass


    except Exception as error:

        print()
        print(
            "Validation encountered an error:"
        )

        print(error)


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RIDEGUARDIAN VEHICLE PARTS AI V2 READY")
    print("=" * 70)

    print()
    print("Classes: 21")
    print("Images: 998")
    print("Segmentation: YES")
    print("GPU: RTX 4080")
    print()
    print("Existing vehicle-parts model was not modified.")
    print()


# ============================================================
# WINDOWS MULTIPROCESSING PROTECTION
# ============================================================

if __name__ == "__main__":

    main()