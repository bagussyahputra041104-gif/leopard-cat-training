import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "02_DATASET" / "dataset" / "test"
MODEL_PATH = (
    BASE_DIR
    / "03_MODEL"
    / "models"
    / "resnet18_baseline_best.pth"
)

OUTPUT_DIR = BASE_DIR / "evaluation_baseline"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CEK FILE
# ============================================================

if not TEST_DIR.exists():

    print("ERROR: Folder test tidak ditemukan.")
    print(TEST_DIR)
    raise SystemExit


if not MODEL_PATH.exists():

    print("ERROR: Model tidak ditemukan.")
    print(MODEL_PATH)
    raise SystemExit


# ============================================================
# TRANSFORMASI
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
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# CLASS
# ============================================================

class_to_idx = test_dataset.class_to_idx

idx_to_class = {
    value: key
    for key, value in class_to_idx.items()
}

print()
print("=" * 75)
print("EVALUASI BASELINE")
print("=" * 75)

print()
print("Device:", DEVICE)

print()
print("Class mapping:")
print(class_to_idx)

print()
print("Jumlah test foto:", len(test_dataset))


# ============================================================
# PREDIKSI
# ============================================================

all_labels = []
all_predictions = []
all_confidences = []
all_paths = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_confidences.extend(
            confidences.cpu().tolist()
        )


# ============================================================
# AMBIL PATH FILE
# ============================================================

all_paths = [
    path
    for path, label in test_dataset.samples
]


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print()

print(
    f"{'':20}"
    f"{'Pred Leopard':>18}"
    f"{'Pred Null':>18}"
)

print(
    f"{'Actual Leopard':20}"
    f"{cm[0][0]:>18}"
    f"{cm[0][1]:>18}"
)

print(
    f"{'Actual Null':20}"
    f"{cm[1][0]:>18}"
    f"{cm[1][1]:>18}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

target_names = [
    idx_to_class[i]
    for i in range(len(idx_to_class))
]

report = classification_report(
    all_labels,
    all_predictions,
    target_names=target_names,
    digits=4
)

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print()
print(report)


# ============================================================
# SIMPAN CLASSIFICATION REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CLASSIFICATION REPORT\n\n"
    )

    file.write(report)


# ============================================================
# CEK PREDIKSI SALAH
# ============================================================

wrong_predictions = []

for path, actual, predicted, confidence in zip(
    all_paths,
    all_labels,
    all_predictions,
    all_confidences
):

    if actual != predicted:

        wrong_predictions.append({
            "file": str(path),
            "actual": idx_to_class[actual],
            "predicted": idx_to_class[predicted],
            "confidence": confidence
        })


# ============================================================
# TAMPILKAN PREDIKSI SALAH
# ============================================================

print()
print("=" * 75)
print("PREDIKSI SALAH")
print("=" * 75)

print()
print(
    f"Jumlah prediksi salah: "
    f"{len(wrong_predictions)}"
)

if wrong_predictions:

    for item in wrong_predictions:

        print()
        print(
            "File      :",
            item["file"]
        )

        print(
            "Actual    :",
            item["actual"]
        )

        print(
            "Predicted :",
            item["predicted"]
        )

        print(
            "Confidence:",
            f"{item['confidence'] * 100:.2f}%"
        )

else:

    print()
    print("Tidak ada prediksi salah pada test set.")


# ============================================================
# SIMPAN SEMUA HASIL PREDIKSI
# ============================================================

prediction_csv = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

with open(
    prediction_csv,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "file",
        "actual",
        "predicted",
        "confidence",
        "correct"
    ])

    for path, actual, predicted, confidence in zip(
        all_paths,
        all_labels,
        all_predictions,
        all_confidences
    ):

        writer.writerow([
            str(path),
            idx_to_class[actual],
            idx_to_class[predicted],
            f"{confidence:.6f}",
            actual == predicted
        ])


# ============================================================
# RINGKASAN
# ============================================================

correct_count = sum(
    actual == predicted
    for actual, predicted in zip(
        all_labels,
        all_predictions
    )
)

total_count = len(all_labels)

accuracy = (
    correct_count / total_count
    if total_count > 0
    else 0
)


print()
print("=" * 75)
print("RINGKASAN")
print("=" * 75)

print()
print(
    f"Total test foto : {total_count}"
)

print(
    f"Benar           : {correct_count}"
)

print(
    f"Salah           : "
    f"{total_count - correct_count}"
)

print(
    f"Accuracy        : "
    f"{accuracy * 100:.2f}%"
)

print()
print("File hasil:")
print(
    classification_report.__name__
)

print(
    report_path
)

print(
    prediction_csv
)

print()
print("=" * 75)
print("EVALUASI SELESAI")
print("=" * 75)
