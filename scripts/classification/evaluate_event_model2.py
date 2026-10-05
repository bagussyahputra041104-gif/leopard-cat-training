import os
import torch
import pandas as pd
from torch import nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TEST_EVENTS_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "dataset",
    "test_events.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "03_MODEL",
    "models",
    "mobilenetv3_small_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "classification",
    "evaluation_event_model2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("EVALUASI EVENT-LEVEL MOBILENETV3-SMALL")
print("=" * 60)

print(f"Device: {DEVICE}")


# =========================================================
# LOAD TEST EVENTS
# =========================================================

print("\nLoading test event dataset...")

df = pd.read_csv(
    TEST_EVENTS_CSV,
    dtype=str,
    keep_default_na=False
)

print(f"Total test events: {len(df)}")

print("Columns:")
print(df.columns.tolist())


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading MobileNetV3-Small...")

model = models.mobilenet_v3_small(
    weights=None
)

# 2 kelas:
# 0 = leopard_cat
# 1 = null
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

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
# CLASS MAPPING
# =========================================================

idx_to_class = {
    0: "leopard_cat",
    1: "null"
}


# =========================================================
# PREDICT ONE IMAGE
# =========================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = idx_to_class[
        prediction.item()
    ]

    confidence_value = confidence.item()

    return (
        predicted_class,
        confidence_value
    )


# =========================================================
# EVENT EVALUATION
# =========================================================

results = []

all_true = []
all_pred = []


print("\nRunning event-level evaluation...")


for _, row in df.iterrows():

    event_id = row["event_id"]

    true_label = row["label"]

    # -----------------------------------------------------
    # Satu event terdiri dari 3 foto
    # -----------------------------------------------------

    image_paths = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"]
    ]

    photo_predictions = []

    photo_confidences = []


    # -----------------------------------------------------
    # Prediksi masing-masing foto
    # -----------------------------------------------------

    for image_path in image_paths:

        predicted_class, confidence = predict_image(
            image_path
        )

        photo_predictions.append(
            predicted_class
        )

        photo_confidences.append(
            confidence
        )


    # -----------------------------------------------------
    # Majority Vote
    # -----------------------------------------------------

    vote_counter = Counter(
        photo_predictions
    )

    event_prediction = vote_counter.most_common(
        1
    )[0][0]


    # -----------------------------------------------------
    # Rata-rata confidence 3 foto
    # -----------------------------------------------------

    event_confidence = sum(
        photo_confidences
    ) / len(photo_confidences)


    # -----------------------------------------------------
    # Cek benar / salah
    # -----------------------------------------------------

    correct = (
        event_prediction == true_label
    )

    all_true.append(
        true_label
    )

    all_pred.append(
        event_prediction
    )


    # -----------------------------------------------------
    # Simpan hasil event
    # -----------------------------------------------------

    results.append({

        "event_id":
            event_id,

        "timestamp":
            row["timestamp"],

        "actual":
            true_label,

        "foto_1_prediction":
            photo_predictions[0],

        "foto_2_prediction":
            photo_predictions[1],

        "foto_3_prediction":
            photo_predictions[2],

        "foto_1_confidence":
            photo_confidences[0],

        "foto_2_confidence":
            photo_confidences[1],

        "foto_3_confidence":
            photo_confidences[2],

        "event_prediction":
            event_prediction,

        "event_confidence":
            event_confidence,

        "correct":
            correct
    })


# =========================================================
# DATAFRAME HASIL
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# METRICS
# =========================================================

event_accuracy = accuracy_score(
    all_true,
    all_pred
)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[
        "leopard_cat",
        "null"
    ]
)

report = classification_report(
    all_true,
    all_pred,
    labels=[
        "leopard_cat",
        "null"
    ],
    target_names=[
        "leopard_cat",
        "null"
    ],
    digits=4
)


# =========================================================
# SUMMARY
# =========================================================

total_events = len(
    results_df
)

correct_events = int(
    results_df["correct"].sum()
)

wrong_events = (
    total_events -
    correct_events
)


# =========================================================
# PRINT RESULT
# =========================================================

print("\n" + "=" * 60)
print("HASIL EVALUASI EVENT-LEVEL")
print("=" * 60)

print(
    f"Total events    : {total_events}"
)

print(
    f"Correct events  : {correct_events}"
)

print(
    f"Wrong events    : {wrong_events}"
)

print(
    f"Event Accuracy  : "
    f"{event_accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)


# =========================================================
# PRINT WRONG EVENTS
# =========================================================

if wrong_events > 0:

    print("\nEvent yang salah:")

    print(
        results_df[
            results_df["correct"] == False
        ][
            [
                "event_id",
                "actual",
                "foto_1_prediction",
                "foto_2_prediction",
                "foto_3_prediction",
                "event_prediction"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nSemua event berhasil "
        "diklasifikasikan dengan benar."
    )


# =========================================================
# SAVE EVENT PREDICTIONS
# =========================================================

event_predictions_path = os.path.join(
    OUTPUT_DIR,
    "event_predictions.csv"
)

results_df.to_csv(
    event_predictions_path,
    index=False
)


# =========================================================
# SAVE WRONG EVENTS
# =========================================================

wrong_events_df = results_df[
    results_df["correct"] == False
]

wrong_events_path = os.path.join(
    OUTPUT_DIR,
    "wrong_events.csv"
)

wrong_events_df.to_csv(
    wrong_events_path,
    index=False
)


# =========================================================
# SAVE CLASSIFICATION REPORT
# =========================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "event_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EVALUASI EVENT-LEVEL "
        "MOBILENETV3-SMALL\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Total events: {total_events}\n"
    )

    f.write(
        f"Correct events: {correct_events}\n"
    )

    f.write(
        f"Wrong events: {wrong_events}\n"
    )

    f.write(
        f"Event Accuracy: "
        f"{event_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(report)

    f.write(
        "\n\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("FILE HASIL EVALUASI")
print("=" * 60)

print(
    f"\n1. Event predictions:\n"
    f"{event_predictions_path}"
)

print(
    f"\n2. Wrong events:\n"
    f"{wrong_events_path}"
)

print(
    f"\n3. Classification report:\n"
    f"{report_path}"
)

print(
    "\nEvaluasi event-level selesai."
)