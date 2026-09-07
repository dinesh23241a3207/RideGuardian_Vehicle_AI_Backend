from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================
# RIDEGUARDIAN - VEHICLE PARTS AI V2 TEST
# ============================================================


PROJECT_ROOT = Path(
    r"C:\RIDER_SYSTEM\RideGuardian"
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "vehicle_parts_v2"
    / "vehicle_parts_v2"
    / "weights"
    / "best.pt"
)


# ------------------------------------------------------------
# TEST IMAGE
# ------------------------------------------------------------
#
# CHANGE THIS PATH whenever you want to test another image.
#

IMAGE_PATH = Path(
    r"C:\RIDER_SYSTEM\RideGuardian\rear_damage.jpg"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONFIDENCE = 0.25

IMAGE_SIZE = 640


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = 0

    print()
    print(
        "Device: CUDA"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = "cpu"

    print()
    print(
        "Device: CPU"
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

print()
print("=" * 70)
print("RIDEGUARDIAN VEHICLE PARTS AI V2 TEST")
print("=" * 70)


# ------------------------------------------------------------
# CHECK MODEL
# ------------------------------------------------------------

print()
print("Model:")

print(
    MODEL_PATH
)


if not MODEL_PATH.exists():

    print()
    print("ERROR: Model not found.")

    print(
        "Check the model path above."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# CHECK IMAGE
# ------------------------------------------------------------

print()
print("Test image:")

print(
    IMAGE_PATH
)


if not IMAGE_PATH.exists():

    print()
    print("ERROR: Test image not found.")

    print()
    print(
        "Put your test image at:"
    )

    print(
        PROJECT_ROOT / "test_car.jpg"
    )

    print()
    print(
        "Or change IMAGE_PATH in this script."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print()
print(
    "Loading model..."
)

model = YOLO(
    str(MODEL_PATH)
)

print(
    "Model loaded successfully."
)


# ------------------------------------------------------------
# DISPLAY CLASSES
# ------------------------------------------------------------

print()
print(
    "Classes:"
)

for class_id, class_name in model.names.items():

    print(
        f"{class_id}: {class_name}"
    )


# ------------------------------------------------------------
# RUN PREDICTION
# ------------------------------------------------------------

print()
print("=" * 70)
print("RUNNING VEHICLE PART DETECTION")
print("=" * 70)

results = model.predict(

    source=str(IMAGE_PATH),

    imgsz=IMAGE_SIZE,

    conf=CONFIDENCE,

    device=DEVICE,

    save=True,

    save_txt=True,

    save_conf=True,

    retina_masks=True,

    verbose=True
)


# ------------------------------------------------------------
# PROCESS RESULT
# ------------------------------------------------------------

print()
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)


for result in results:

    boxes = result.boxes

    masks = result.masks


    if boxes is None or len(boxes) == 0:

        print()
        print(
            "NO VEHICLE PARTS DETECTED"
        )

        continue


    print()

    print(
        f"Detected parts: {len(boxes)}"
    )

    print()


    # --------------------------------------------------------
    # EACH DETECTION
    # --------------------------------------------------------

    for index in range(len(boxes)):

        class_id = int(
            boxes.cls[index].item()
        )

        confidence = float(
            boxes.conf[index].item()
        )

        class_name = model.names[
            class_id
        ]


        print(
            f"{index + 1}. "
            f"{class_name} "
            f"→ "
            f"{confidence * 100:.2f}%"
        )


    # --------------------------------------------------------
    # MASK INFORMATION
    # --------------------------------------------------------

    print()

    if masks is not None:

        print(
            f"Segmentation masks: "
            f"{len(masks)}"
        )

    else:

        print(
            "Segmentation masks: NONE"
        )


# ------------------------------------------------------------
# OUTPUT LOCATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print()
print(
    "Annotated result saved by YOLO "
    "inside the runs/segment directory."
)

print()
print(
    "Use the detected labels above to evaluate "
    "whether Parts V2 is identifying the correct "
    "vehicle components."
)

print()