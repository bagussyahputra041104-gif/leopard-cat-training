import os
import torch
import pandas as pd
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TEST_DIR = os.path.join(BASE_DIR, "02_DATASET", "dataset", "test")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "03_MODEL",
    "models",
    "mobilenetv3_small_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation_model2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TRANSFORM
# =========================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("EVALUASI MOBILENETV3-SMALL")
print("=" * 60)

print(f"Device: {DEVICE}")
print("Loading test dataset...")

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test images: {len(test_dataset)}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading MobileNetV3-Small...")

model = models.mobilenet_v3_small(
    weights=None
)

# Ubah classifier terakhir menjadi 2 kelas
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================================================
# EVALUATION
# =========================================================

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []
all_confidences = []

total_loss = 0.0
total_correct = 0
total_images = 0


print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        total_loss += loss.item() * images.size(0)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_images += images.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )


# =========================================================
# METRICS
# =========================================================

test_loss = total_loss / total_images
test_accuracy = total_correct / total_images


print("\n" + "=" * 60)
print("HASIL EVALUASI MOBILENETV3-SMALL")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Correct       : {total_correct}")
print(f"Total         : {total_images}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

class_names = test_dataset.classes

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)


# =========================================================
# SAVE CLASSIFICATION REPORT
# =========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("EVALUASI MOBILENETV3-SMALL\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Device: {DEVICE}\n")
    f.write(f"Test images: {total_images}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )
    f.write(
        f"Correct: {total_correct}/{total_images}\n\n"
    )

    f.write("Class Mapping:\n")
    f.write(str(test_dataset.class_to_idx))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))


# =========================================================
# SAVE PREDICTIONS
# =========================================================

prediction_rows = []

for idx, sample in enumerate(
    test_dataset.samples
):

    image_path, true_label = sample

    predicted_label = all_predictions[idx]

    confidence = all_confidences[idx]

    prediction_rows.append({
        "image_path": image_path,
        "actual": class_names[true_label],
        "predicted": class_names[predicted_label],
        "confidence": float(confidence),
        "correct": true_label == predicted_label
    })


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# =========================================================
# WRONG PREDICTIONS
# =========================================================

wrong_df = predictions_df[
    predictions_df["correct"] == False
]

wrong_path = os.path.join(
    OUTPUT_DIR,
    "wrong_predictions.csv"
)

wrong_df.to_csv(
    wrong_path,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\nFile hasil evaluasi:")

print(
    f"- Classification report:\n  {report_path}"
)

print(
    f"- Test predictions:\n  {predictions_path}"
)

print(
    f"- Wrong predictions:\n  {wrong_path}"
)

print("\nJumlah prediksi salah:", len(wrong_df))

print("\nEvaluasi selesai.")