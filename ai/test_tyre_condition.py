import sys
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    AI_DIR
    / "models"
    / "tyre_condition"
    / "best_tyre_condition_coatnet_v2.pth"
)

# YOUR REAL TEST IMAGE
IMAGE_PATH = Path(
    r"C:\RIDER_SYSTEM\RideGuardian\test_tyre.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

MODEL_NAME = "coatnet_0_rw_224.sw_in1k"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 65)
print("RIDEGUARDIAN TYRE CONDITION V2 TEST")
print("=" * 65)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Model:")
print(MODEL_PATH)

print()
print("Test Image:")
print(IMAGE_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print()
    print("ERROR: MODEL NOT FOUND")
    print(MODEL_PATH)

    sys.exit(1)


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():

    print()
    print("ERROR: TEST IMAGE NOT FOUND")
    print(IMAGE_PATH)

    sys.exit(1)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint.get(
    "class_names",
    [
        "NEW",
        "SERVICEABLE",
        "UNUSABLE"
    ]
)

print(
    "Classes:",
    class_names
)


# ============================================================
# CREATE MODEL
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print("Loading test image...")

try:

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

except Exception as error:

    print()
    print("ERROR: Could not open image.")
    print(error)

    sys.exit(1)


print(
    "Image size:",
    image.size
)


# ============================================================
# PREPROCESS
# ============================================================

input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Running AI prediction...")

with torch.no_grad():

    outputs = model(
        input_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("PREDICTIONS")
print("=" * 65)

results = []

for index, class_name in enumerate(
    class_names
):

    confidence = (
        probabilities[index].item()
        * 100
    )

    results.append(
        (
            class_name,
            confidence
        )
    )


results.sort(
    key=lambda x: x[1],
    reverse=True
)


for class_name, confidence in results:

    print(
        f"{class_name:15s}"
        f"{confidence:6.2f}%"
    )


# ============================================================
# FINAL PREDICTION
# ============================================================

prediction = results[0][0]

confidence = results[0][1]


print()
print("=" * 65)
print("FINAL PREDICTION")
print("=" * 65)

print(
    "CONDITION:",
    prediction
)

print(
    f"CONFIDENCE: {confidence:.2f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 65)
print("RIDEGUARDIAN INTERPRETATION")
print("=" * 65)


if prediction == "NEW":

    print(
        "STATUS: GOOD"
    )

    print(
        "The tyre appears to have "
        "little or no visible wear."
    )


elif prediction == "SERVICEABLE":

    print(
        "STATUS: MONITOR"
    )

    print(
        "The tyre appears usable "
        "but shows some wear."
    )


elif prediction == "UNUSABLE":

    print(
        "STATUS: ATTENTION REQUIRED"
    )

    print(
        "The tyre appears severely "
        "worn or damaged."
    )

    print(
        "Recommend inspection and "
        "possible replacement."
    )


print()
print("=" * 65)
print("TEST COMPLETE")
print("=" * 65)