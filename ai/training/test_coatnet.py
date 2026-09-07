import json
import sys
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


# ============================================================
# RIDEGUARDIAN
# CoAtNet - Single Tyre Image Testing
#
# Input:
#   Any tyre photograph
#
# Output:
#   NEW / SERVICEABLE / UNUSABLE
#   Confidence
#   Recommendation
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    AI_DIR
    / "models"
    / "coatnet"
)

MODEL_PATH = (
    MODEL_DIR
    / "best_coatnet.pth"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

MODEL_NAME = "coatnet_0_rw_224.sw_in1k"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print()
print("=" * 70)
print("RIDEGUARDIAN - CoAtNet TYRE TEST")
print("=" * 70)

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

print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nCoAtNet model not found:\n{MODEL_PATH}"
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

    class_names = json.load(file)


print()
print(
    "Classes:",
    class_names
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading CoAtNet..."
)


model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=len(class_names)
)


checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(DEVICE)

model.eval()


print(
    "CoAtNet loaded successfully."
)


# ============================================================
# IMAGE TRANSFORM
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
    condition,
    confidence
):

    if condition == "NEW":

        return (
            "Tyre appears to be in new/good condition. "
            "Continue normal inspection and maintenance."
        )


    if condition == "SERVICEABLE":

        return (
            "Tyre appears serviceable. "
            "Continue monitoring tread wear and inspect regularly."
        )


    if condition == "UNUSABLE":

        return (
            "Tyre appears significantly worn or unsafe. "
            "Inspection and possible replacement is recommended "
            "before continued riding."
        )


    return (
        "Condition could not be determined."
    )


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
        "Enter the full path of the tyre image."
    )

    print(
        "Example:"
    )

    print(
        r'C:\Users\dines\Pictures\tyre.jpg'
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
        f"\nImage not found:\n{image_path}"
    )


if not image_path.is_file():

    raise ValueError(
        f"\nThe specified path is not a file:\n"
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
    ).convert("RGB")

except Exception as error:

    raise RuntimeError(
        f"\nCould not open image:\n{error}"
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
).unsqueeze(0)


image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# PREDICTION
# ============================================================

print()
print(
    "Running CoAtNet prediction..."
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


condition = (
    class_names[predicted_index]
)


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

all_probabilities = (
    probabilities[0]
    .detach()
    .cpu()
    .numpy()
)


print()
print("=" * 70)
print("PREDICTION")
print("=" * 70)

print()

print(
    f"TYRE CONDITION: {condition}"
)

print(
    f"CONFIDENCE: {confidence * 100:.2f}%"
)


print()
print(
    "Class probabilities:"
)


for index, class_name in enumerate(
    class_names
):

    probability = (
        all_probabilities[index]
        * 100
    )

    print(
        f"  {class_name}: "
        f"{probability:.2f}%"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

recommendation = get_recommendation(
    condition,
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
# CONFIDENCE WARNING
# ============================================================

if confidence < 0.60:

    print()
    print(
        "WARNING:"
    )

    print(
        "The model has relatively low confidence. "
        "Use another clear tyre photograph or "
        "perform a manual inspection."
    )


elif confidence < 0.80:

    print()
    print(
        "NOTE:"
    )

    print(
        "The model has moderate confidence. "
        "A clearer photograph may improve reliability."
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TYRE AI TEST COMPLETE")
print("=" * 70)