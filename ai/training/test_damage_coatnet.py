import json
import sys
from pathlib import Path

import torch
import timm

from PIL import Image
from torchvision import transforms


# ============================================================
# RIDEGUARDIAN
# DAMAGE CoAtNet - SINGLE PHOTO INFERENCE
#
# Input:
#   A vehicle damage photograph
#
# Output:
#   Damage type
#   Confidence
#   All class probabilities
#   Recommendation
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_DIR = (
    AI_DIR
    / "models"
    / "damage_condition"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_damage_coatnet.pth"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

MODEL_NAME = (
    "coatnet_0_rw_224.sw_in1k"
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print(
    "RIDEGUARDIAN - VEHICLE DAMAGE AI TEST"
)
print("=" * 70)

print()

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

print()


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nDamage model not found:\n"
        f"{MODEL_PATH}"
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"\nClass names file not found:\n"
        f"{CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(
        file
    )


print(
    "Classes:"
)

for index, name in enumerate(
    class_names
):

    print(
        f"  {index}: {name}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading Damage CoAtNet..."
)

model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=len(
        class_names
    )
)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    DEVICE
)

model.eval()


print(
    "Damage CoAtNet loaded successfully."
)


# ============================================================
# IMAGE TRANSFORM
#
# SAME NORMALIZATION USED DURING TRAINING
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
# RECOMMENDATION
# ============================================================

def get_recommendation(
    damage,
    confidence
):

    recommendations = {

        "Broken part":
            (
                "A broken vehicle component is visible. "
                "Inspection and repair or replacement "
                "of the affected component is recommended."
            ),

        "Corrosion":
            (
                "Visible corrosion is detected. "
                "Inspect the affected area and address "
                "the corrosion before it worsens."
            ),

        "Cracked":
            (
                "A crack is detected. "
                "The affected component should be inspected "
                "promptly for structural or safety impact."
            ),

        "Dent":
            (
                "A dent is detected. "
                "Inspect the affected body panel or component "
                "for hidden damage."
            ),

        "Flaking":
            (
                "Surface flaking is detected. "
                "Inspect the affected area for paint or "
                "material deterioration."
            ),

        "Missing part":
            (
                "A missing vehicle component is detected. "
                "Inspect the vehicle and replace the missing "
                "component if required."
            ),

        "Paint chip":
            (
                "Paint damage is detected. "
                "Consider repairing the affected area to "
                "prevent further surface deterioration."
            ),

        "Scratch":
            (
                "A scratch is detected. "
                "Inspect its depth and surrounding area "
                "for additional damage."
            )
    }


    recommendation = (
        recommendations.get(
            damage,
            "Inspect the affected vehicle area."
        )
    )


    if confidence < 0.60:

        recommendation += (
            " Model confidence is low, so a clearer "
            "photograph or manual inspection is recommended."
        )

    elif confidence < 0.80:

        recommendation += (
            " Model confidence is moderate; another "
            "clear photograph may improve reliability."
        )


    return recommendation


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) >= 2:

    image_path = Path(
        sys.argv[1]
    )

else:

    print()
    print(
        "Enter the full path of the vehicle damage photo."
    )

    print()
    print(
        r'Example: C:\Users\dines\Pictures\car_damage.jpg'
    )

    print()

    image_path = Path(
        input(
            "Image path: "
        ).strip().strip('"')
    )


# ============================================================
# CHECK IMAGE
# ============================================================

if not image_path.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n"
        f"{image_path}"
    )


if not image_path.is_file():

    raise ValueError(
        f"\nPath is not a file:\n"
        f"{image_path}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print(
    "Loading image:"
)

print(
    image_path
)


try:

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

except Exception as error:

    raise RuntimeError(
        f"\nCould not open image:\n"
        f"{error}"
    )


print(
    "Image size:",
    image.size
)


# ============================================================
# PREPARE IMAGE
# ============================================================

image_tensor = transform(
    image
).unsqueeze(
    0
)

image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

print()
print(
    "Running Damage CoAtNet prediction..."
)


with torch.no_grad():

    if torch.cuda.is_available():

        with torch.amp.autocast(
            "cuda"
        ):

            outputs = model(
                image_tensor
            )

    else:

        outputs = model(
            image_tensor
        )


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = torch.softmax(
    outputs,
    dim=1
)


confidence, predicted_index = (
    torch.max(
        probabilities,
        dim=1
    )
)


predicted_index = (
    predicted_index.item()
)

confidence = (
    confidence.item()
)


damage = (
    class_names[
        predicted_index
    ]
)


# ============================================================
# ALL PROBABILITIES
# ============================================================

all_probabilities = (
    probabilities[0]
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print(
    "DAMAGE PREDICTION"
)
print("=" * 70)

print()

print(
    f"DAMAGE DETECTED: {damage}"
)

print(
    f"CONFIDENCE: {confidence * 100:.2f}%"
)


# ============================================================
# CLASS PROBABILITIES
# ============================================================

print()
print(
    "CLASS PROBABILITIES:"
)

sorted_probabilities = sorted(

    enumerate(
        all_probabilities
    ),

    key=lambda item: item[1],

    reverse=True
)


for index, probability in (
    sorted_probabilities
):

    print(
        f"  {class_names[index]}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

recommendation = get_recommendation(
    damage,
    confidence
)


print()
print(
    "RECOMMENDATION:"
)

print(
    recommendation
)


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

print()

if confidence >= 0.90:

    print(
        "CONFIDENCE LEVEL: VERY HIGH"
    )

elif confidence >= 0.80:

    print(
        "CONFIDENCE LEVEL: HIGH"
    )

elif confidence >= 0.60:

    print(
        "CONFIDENCE LEVEL: MODERATE"
    )

else:

    print(
        "CONFIDENCE LEVEL: LOW"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    "DAMAGE AI TEST COMPLETE"
)
print("=" * 70)