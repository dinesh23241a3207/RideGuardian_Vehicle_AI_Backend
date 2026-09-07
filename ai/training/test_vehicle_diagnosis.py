import json
import sys
from pathlib import Path

import torch
import timm

from PIL import Image
from torchvision import transforms


# ============================================================
# RIDEGUARDIAN
# COMBINED VEHICLE PHOTO DIAGNOSIS
#
# Models:
#   1. Vehicle Part CoAtNet
#   2. Damage Condition CoAtNet
#
# Input:
#   One vehicle photograph
#
# Output:
#   Vehicle part
#   Damage/condition
#   Confidence
#   Combined diagnosis
#   Recommendation
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = (
    Path(__file__).resolve().parent.parent
)


# ---------------- VEHICLE PART MODEL ----------------

PART_MODEL_DIR = (
    AI_DIR
    / "models"
    / "vehicle_parts"
)

PART_MODEL_PATH = (
    PART_MODEL_DIR
    / "best_vehicle_parts_coatnet.pth"
)

PART_CLASSES_PATH = (
    PART_MODEL_DIR
    / "class_names.json"
)


# ---------------- DAMAGE MODEL ----------------

DAMAGE_MODEL_DIR = (
    AI_DIR
    / "models"
    / "damage_condition"
)

DAMAGE_MODEL_PATH = (
    DAMAGE_MODEL_DIR
    / "best_damage_coatnet.pth"
)

DAMAGE_CLASSES_PATH = (
    DAMAGE_MODEL_DIR
    / "class_names.json"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = (
    "coatnet_0_rw_224.sw_in1k"
)

IMAGE_SIZE = 224


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
print("=" * 75)
print(
    "RIDEGUARDIAN - COMBINED VEHICLE PHOTO DIAGNOSIS"
)
print("=" * 75)

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
# FILE CHECK
# ============================================================

required_files = [

    PART_MODEL_PATH,

    PART_CLASSES_PATH,

    DAMAGE_MODEL_PATH,

    DAMAGE_CLASSES_PATH

]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    PART_CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:

    part_classes = json.load(
        file
    )


with open(
    DAMAGE_CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:

    damage_classes = json.load(
        file
    )


# ============================================================
# LOAD PART MODEL
# ============================================================

print(
    "Loading Vehicle Part CoAtNet..."
)

part_model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=len(
        part_classes
    )
)


part_checkpoint = torch.load(

    PART_MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


part_model.load_state_dict(
    part_checkpoint[
        "model_state_dict"
    ]
)


part_model = part_model.to(
    DEVICE
)

part_model.eval()


print(
    "Vehicle Part model loaded."
)


# ============================================================
# LOAD DAMAGE MODEL
# ============================================================

print(
    "Loading Damage CoAtNet..."
)

damage_model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=len(
        damage_classes
    )
)


damage_checkpoint = torch.load(

    DAMAGE_MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


damage_model.load_state_dict(
    damage_checkpoint[
        "model_state_dict"
    ]
)


damage_model = damage_model.to(
    DEVICE
)

damage_model.eval()


print(
    "Damage model loaded."
)


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
# RECOMMENDATION ENGINE
# ============================================================

def get_recommendation(
    part,
    damage,
    part_confidence,
    damage_confidence
):

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if (
        part_confidence < 0.60
        or
        damage_confidence < 0.60
    ):

        return (
            "AI confidence is low. "
            "Upload a clearer photograph showing "
            "the affected component."
        )


    # --------------------------------------------------------
    # CRITICAL CONDITIONS
    # --------------------------------------------------------

    if damage == "Cracked":

        return (
            f"A crack was detected on the {part}. "
            "Prompt inspection is recommended. "
            "If the component is safety-critical or "
            "structural, avoid relying on the photo alone "
            "for a replacement decision."
        )


    if damage == "Broken part":

        return (
            f"The {part} appears to have a broken component. "
            "Inspection and likely repair or replacement "
            "of the affected component is recommended."
        )


    if damage == "Missing part":

        return (
            f"A component appears to be missing from the "
            f"{part}. Inspection and replacement of the "
            "missing component is recommended."
        )


    if damage == "Corrosion":

        return (
            f"Corrosion was detected on the {part}. "
            "Inspect the affected area and address the "
            "corrosion before it progresses."
        )


    if damage == "Dent":

        return (
            f"A dent was detected on the {part}. "
            "Inspect the component for hidden damage. "
            "Repair may be appropriate depending on the "
            "depth and structural condition."
        )


    if damage == "Scratch":

        return (
            f"A scratch was detected on the {part}. "
            "Inspect its depth. Surface repair may be "
            "sufficient for minor scratches."
        )


    if damage == "Paint chip":

        return (
            f"Paint damage was detected on the {part}. "
            "Surface repair is recommended to reduce the "
            "chance of further deterioration."
        )


    if damage == "Flaking":

        return (
            f"Surface flaking was detected on the {part}. "
            "Inspect the affected area for material or "
            "paint deterioration."
        )


    return (
        f"The {part} shows a condition classified as "
        f"{damage}. Further inspection is recommended."
    )


# ============================================================
# IMAGE PATH
# ============================================================

if len(sys.argv) >= 2:

    image_path = Path(
        sys.argv[1]
    )

else:

    print()
    print(
        "Enter the full path of the vehicle photograph."
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
        f"\nThe supplied path is not a file:\n"
        f"{image_path}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print(
    "Loading photograph:"
)

print(
    image_path
)


image = Image.open(
    image_path
).convert(
    "RGB"
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
# RUN BOTH MODELS
# ============================================================

print()
print(
    "Running Vehicle Part AI..."
)

with torch.no_grad():

    if DEVICE.type == "cuda":

        with torch.amp.autocast(
            "cuda"
        ):

            part_outputs = part_model(
                image_tensor
            )

    else:

        part_outputs = part_model(
            image_tensor
        )


print(
    "Running Damage AI..."
)

with torch.no_grad():

    if DEVICE.type == "cuda":

        with torch.amp.autocast(
            "cuda"
        ):

            damage_outputs = damage_model(
                image_tensor
            )

    else:

        damage_outputs = damage_model(
            image_tensor
        )


# ============================================================
# PROBABILITIES
# ============================================================

part_probabilities = torch.softmax(
    part_outputs,
    dim=1
)


damage_probabilities = torch.softmax(
    damage_outputs,
    dim=1
)


# ============================================================
# PART PREDICTION
# ============================================================

part_confidence, part_index = (
    torch.max(
        part_probabilities,
        dim=1
    )
)


part_index = (
    part_index.item()
)

part_confidence = (
    part_confidence.item()
)

part = (
    part_classes[
        part_index
    ]
)


# ============================================================
# DAMAGE PREDICTION
# ============================================================

damage_confidence, damage_index = (
    torch.max(
        damage_probabilities,
        dim=1
    )
)


damage_index = (
    damage_index.item()
)

damage_confidence = (
    damage_confidence.item()
)

damage = (
    damage_classes[
        damage_index
    ]
)


# ============================================================
# TOP PART PREDICTIONS
# ============================================================

part_probs = (
    part_probabilities[0]
    .detach()
    .cpu()
    .tolist()
)


top_parts = sorted(

    enumerate(
        part_probs
    ),

    key=lambda x: x[1],

    reverse=True

)[:5]


# ============================================================
# TOP DAMAGE PREDICTIONS
# ============================================================

damage_probs = (
    damage_probabilities[0]
    .detach()
    .cpu()
    .tolist()
)


top_damages = sorted(

    enumerate(
        damage_probs
    ),

    key=lambda x: x[1],

    reverse=True

)[:5]


# ============================================================
# RECOMMENDATION
# ============================================================

recommendation = get_recommendation(

    part,

    damage,

    part_confidence,

    damage_confidence

)


# ============================================================
# COMBINED CONFIDENCE
# ============================================================

combined_confidence = (
    part_confidence
    *
    damage_confidence
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 75)
print(
    "VEHICLE PHOTO DIAGNOSIS"
)
print("=" * 75)

print()

print(
    "AFFECTED PART:"
)

print(
    f"  {part}"
)

print(
    f"  Confidence: "
    f"{part_confidence * 100:.2f}%"
)


print()

print(
    "DETECTED CONDITION:"
)

print(
    f"  {damage}"
)

print(
    f"  Confidence: "
    f"{damage_confidence * 100:.2f}%"
)


print()

print(
    "COMBINED DIAGNOSIS:"
)

print(
    f"  {part} — {damage}"
)

print(
    f"  Combined confidence: "
    f"{combined_confidence * 100:.2f}%"
)


# ============================================================
# TOP PART PREDICTIONS
# ============================================================

print()
print(
    "TOP PART PREDICTIONS:"
)

for index, probability in top_parts:

    print(
        f"  {part_classes[index]}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# TOP DAMAGE PREDICTIONS
# ============================================================

print()
print(
    "TOP DAMAGE PREDICTIONS:"
)

for index, probability in top_damages:

    print(
        f"  {damage_classes[index]}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

print()
print(
    "RECOMMENDATION:"
)

print(
    f"  {recommendation}"
)


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

print()

if (
    part_confidence >= 0.90
    and
    damage_confidence >= 0.90
):

    confidence_level = (
        "VERY HIGH"
    )

elif (
    part_confidence >= 0.80
    and
    damage_confidence >= 0.80
):

    confidence_level = (
        "HIGH"
    )

elif (
    part_confidence >= 0.60
    and
    damage_confidence >= 0.60
):

    confidence_level = (
        "MODERATE"
    )

else:

    confidence_level = (
        "LOW"
    )


print(
    f"OVERALL CONFIDENCE: "
    f"{confidence_level}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print(
    "COMBINED PHOTO DIAGNOSIS COMPLETE"
)
print("=" * 75)