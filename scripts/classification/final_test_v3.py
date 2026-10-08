from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# =========================================================
# PATH
# =========================================================
PROJECT_ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "03_MODEL"
    / "models"
    / "mobilenetv3_small_v3_best.pth"
)

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "02_DATASET"
    / "external_test_baru"
)

THRESHOLD = 0.50

# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# MODEL
# =========================================================
model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()

# =========================================================
# EVALUATION
# =========================================================
total = 0
correct = 0

leopard_total = 0
leopard_correct = 0

null_total = 0
null_correct = 0

print("=" * 80)
print("FINAL EXTERNAL TEST")
print("MobileNetV3-Small V3 + Threshold 0.50")
print("=" * 80)
print()

for category_dir in sorted(EXTERNAL_DIR.iterdir()):

    if not category_dir.is_dir():
        continue

    gt = (
        "leopard_cat"
        if category_dir.name == "leopard_cat"
        else "null"
    )

    for image_path in sorted(category_dir.iterdir()):

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        ).unsqueeze(0)

        with torch.no_grad():

            output = model(image)

            probability = torch.softmax(
                output,
                dim=1
            )[0, 0].item()

        pred = (
            "leopard_cat"
            if probability >= THRESHOLD
            else "null"
        )

        correct_image = pred == gt

        total += 1

        if correct_image:
            correct += 1

        if gt == "leopard_cat":

            leopard_total += 1

            if correct_image:
                leopard_correct += 1

        else:

            null_total += 1

            if correct_image:
                null_correct += 1

        status = "✓ BENAR" if correct_image else "✗ SALAH"

        print(
            f"{image_path.name:20s} | "
            f"GT: {gt:12s} | "
            f"Leopard: {probability * 100:6.2f}% | "
            f"Pred: {pred:12s} | "
            f"{status}"
        )

# =========================================================
# METRICS
# =========================================================
accuracy = correct / total * 100
leopard_recall = leopard_correct / leopard_total * 100
null_specificity = null_correct / null_total * 100

print()
print("=" * 80)
print("HASIL FINAL")
print("=" * 80)

print(f"Accuracy           : {accuracy:.2f}%")
print(
    f"Leopard Cat Recall : "
    f"{leopard_correct}/{leopard_total} "
    f"({leopard_recall:.2f}%)"
)
print(
    f"NULL Specificity   : "
    f"{null_correct}/{null_total} "
    f"({null_specificity:.2f}%)"
)

print()
print(f"False Positive     : {null_total - null_correct}")
print(f"False Negative     : {leopard_total - leopard_correct}")

print("=" * 80)