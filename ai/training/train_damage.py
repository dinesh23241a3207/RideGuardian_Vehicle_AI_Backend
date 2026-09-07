from ultralytics import YOLO
from pathlib import Path


# ==========================================================
# RIDEGUARDIAN DAMAGE AI TRAINING
# ==========================================================

BASE_DIR = Path(r"C:\RIDER_SYSTEM\RideGuardian")

DATASET_DIR = BASE_DIR / "ai" / "body_yolo" / "damage"

DATA_YAML = DATASET_DIR / "data.yaml"

PROJECT_DIR = (
    BASE_DIR
    / "ai"
    / "models"
    / "damage"
)


# ==========================================================
# SETTINGS
# ==========================================================

MODEL = "yolo11m-seg.pt"

EPOCHS = 100

IMAGE_SIZE = 640

BATCH = 8

WORKERS = 0

DEVICE = 0


# ==========================================================
# CHECK DATASET
# ==========================================================

print("=" * 70)
print("RIDEGUARDIAN DAMAGE AI")
print("=" * 70)

print()

print("Dataset:")
print(DATASET_DIR)

print()

print("Data YAML:")
print(DATA_YAML)

print()


if not DATA_YAML.exists():

    raise FileNotFoundError(
        f"data.yaml not found:\n{DATA_YAML}"
    )


# ==========================================================
# CHECK TRAIN IMAGES
# ==========================================================

train_images = list(
    (DATASET_DIR / "images" / "train").glob("*")
)

val_images = list(
    (DATASET_DIR / "images" / "val").glob("*")
)

test_images = list(
    (DATASET_DIR / "images" / "test").glob("*")
)


print("Dataset counts")
print("-" * 70)

print("Train images:", len(train_images))
print("Val images:  ", len(val_images))
print("Test images: ", len(test_images))

print()


# ==========================================================
# LOAD PRETRAINED MODEL
# ==========================================================

print("=" * 70)
print("LOADING YOLO11m-SEG")
print("=" * 70)

model = YOLO(MODEL)


# ==========================================================
# TRAIN
# ==========================================================

print()

print("=" * 70)
print("STARTING DAMAGE AI TRAINING")
print("=" * 70)

print()

results = model.train(

    data=str(DATA_YAML),

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH,

    workers=WORKERS,

    device=DEVICE,

    project=str(PROJECT_DIR),

    name="damage_v1",

    exist_ok=True,

    pretrained=True,

    patience=20,

    save=True,

    save_period=10,

    plots=True,

    verbose=True,

)


# ==========================================================
# VALIDATION
# ==========================================================

print()

print("=" * 70)
print("FINAL DAMAGE AI VALIDATION")
print("=" * 70)

print()

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=IMAGE_SIZE,
    batch=BATCH,
    device=DEVICE,
    workers=WORKERS,
)


# ==========================================================
# RESULTS
# ==========================================================

print()

print("=" * 70)
print("DAMAGE AI TRAINING COMPLETE")
print("=" * 70)

print()

print("Dataset:")
print(DATASET_DIR)

print()

print("Classes:")
print(len(model.names))

print()

print("Class names:")
for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

print()

print("Best model should be located at:")

print(
    PROJECT_DIR
    / "damage_v1"
    / "weights"
    / "best.pt"
)

print()

print("=" * 70)
print("DAMAGE AI READY")
print("=" * 70)