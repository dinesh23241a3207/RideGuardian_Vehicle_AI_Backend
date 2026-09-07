import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import timm

from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

AI_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = AI_DIR / "datasets" / "Tyre_Condition_Dataset"
MODEL_DIR = AI_DIR / "models" / "tyre_condition"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 6
RANDOM_SEED = 42
NUM_WORKERS = 0
MODEL_NAME = "coatnet_0_rw_224.sw_in1k"

MODEL_PATH = MODEL_DIR / "best_tyre_condition_coatnet_v2.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names_v2.json"
HISTORY_PATH = MODEL_DIR / "training_history_v2.json"
RESULTS_PATH = MODEL_DIR / "test_results_v2.json"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print("=" * 75)
print("RIDEGUARDIAN TYRE CONDITION CoAtNet - V2")
print("=" * 75)
print("Dataset:", DATASET_DIR)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATASET_DIR}")

expected_classes = {"NEW", "SERVICEABLE", "UNUSABLE"}

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\nLoading dataset...")
base_dataset = datasets.ImageFolder(DATASET_DIR)
class_names = base_dataset.classes
targets = np.array(base_dataset.targets)
num_classes = len(class_names)

print("Classes:", class_names)

if set(class_names) != expected_classes:
    raise ValueError(f"Unexpected classes. Found: {class_names}; Expected: {sorted(expected_classes)}")

print("\nDataset distribution:")
for i, name in enumerate(class_names):
    print(f"{name:15s}: {int(np.sum(targets == i))}")
print("Total:", len(base_dataset))

all_indices = np.arange(len(base_dataset))

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=targets
)

temp_targets = targets[temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_targets
)

print("\nDataset split:")
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

train_full = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
eval_full = datasets.ImageFolder(DATASET_DIR, transform=eval_transform)

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(eval_full, val_indices)
test_dataset = Subset(eval_full, test_indices)

train_targets = targets[train_indices]
train_counts = np.bincount(train_targets, minlength=num_classes)

print("\nTraining class counts:")
for i, name in enumerate(class_names):
    print(f"{name:15s}: {int(train_counts[i])}")

# Balanced sampling prevents NEW from dominating training.
sample_class_weights = 1.0 / np.maximum(train_counts, 1)
sample_weights = np.array([sample_class_weights[label] for label in train_targets])
sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)

# Moderate loss weighting because the sampler already balances classes.
loss_weights = 1.0 / np.sqrt(np.maximum(train_counts, 1))
loss_weights = loss_weights / loss_weights.mean()
loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=DEVICE)

print("\nLoss weights:")
for i, name in enumerate(class_names):
    print(f"{name:15s}: {loss_weights[i].item():.4f}")

print("\n" + "=" * 75)
print("Loading pretrained CoAtNet")
print("=" * 75)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

print("Model loaded successfully.")

criterion = nn.CrossEntropyLoss(
    weight=loss_weights,
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

def train_one_epoch():
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (torch.argmax(outputs, 1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, 1)

        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

    return (
        total_loss / max(total_samples, 1),
        100.0 * total_correct / max(total_samples, 1),
        np.array(all_labels),
        np.array(all_predictions)
    )

best_val_accuracy = -1.0
best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

print("\n" + "=" * 75)
print("STARTING V2 TYRE TRAINING")
print("=" * 75)

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    train_loss, train_accuracy = train_one_epoch()
    val_loss, val_accuracy, _, _ = evaluate(val_loader)
    scheduler.step()

    elapsed = time.time() - start
    lr = optimizer.param_groups[0]["lr"]

    print("\n" + "-" * 75)
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.2f}%")
    print(f"Val Loss       : {val_loss:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.2f}%")
    print(f"Learning Rate  : {lr:.8f}")
    print(f"Epoch Time     : {elapsed:.1f}s")

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "learning_rate": float(lr),
        "epoch_time_seconds": float(elapsed)
    })

    improved = (
        val_accuracy > best_val_accuracy or
        (abs(val_accuracy - best_val_accuracy) < 1e-8 and val_loss < best_val_loss)
    )

    if improved:
        best_val_accuracy = val_accuracy
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "version": "v2",
            "random_seed": RANDOM_SEED
        }, MODEL_PATH)

        print("BEST MODEL SAVED")
    else:
        epochs_without_improvement += 1
        print(f"No validation improvement: {epochs_without_improvement}/{PATIENCE}")

        if epochs_without_improvement >= PATIENCE:
            print("EARLY STOPPING")
            break

with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

test_loss, test_accuracy, y_true, y_pred = evaluate(test_loader)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred,
    labels=np.arange(num_classes),
    zero_division=0
)

report = classification_report(
    y_true, y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    y_true, y_pred,
    labels=np.arange(num_classes)
)

unusable_index = class_names.index("UNUSABLE")
unusable_recall = recall[unusable_index] * 100.0

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)
print(f"\nUNUSABLE RECALL: {unusable_recall:.2f}%")

results = {
    "version": "v2",
    "model_name": MODEL_NAME,
    "classes": class_names,
    "dataset_total": int(len(base_dataset)),
    "train_images": int(len(train_indices)),
    "validation_images": int(len(val_indices)),
    "test_images": int(len(test_indices)),
    "best_validation_accuracy": float(best_val_accuracy),
    "best_validation_loss": float(best_val_loss),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "unusable_recall_percent": float(unusable_recall),
    "precision_by_class": {class_names[i]: float(precision[i]) for i in range(num_classes)},
    "recall_by_class": {class_names[i]: float(recall[i]) for i in range(num_classes)},
    "f1_by_class": {class_names[i]: float(f1[i]) for i in range(num_classes)},
    "support_by_class": {class_names[i]: int(support[i]) for i in range(num_classes)},
    "confusion_matrix": matrix.tolist()
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print("\n" + "=" * 75)
print("TYRE CONDITION V2 TRAINING COMPLETE")
print("=" * 75)
print("Best model:", MODEL_PATH)
print("Training history:", HISTORY_PATH)
print("Test results:", RESULTS_PATH)
print(f"Final test accuracy: {test_accuracy:.2f}%")
print(f"UNUSABLE recall: {unusable_recall:.2f}%")
