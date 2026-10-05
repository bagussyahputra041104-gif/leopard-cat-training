import os
import glob

import torch
import pandas as pd
import numpy as np

from PIL import Image

from torchvision import models, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TEST_DIR = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "dataset",
    "test"
)

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
    "resnet18_baseline_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "classification",
    "evaluation_event_baseline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 3. CLASS MAPPING
# ============================================================

class_names = [
    "leopard_cat",
    "null"
]

idx_to_class = {
    0: "leopard_cat",
    1: "null"
}


# ============================================================
# 4. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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
# 5. LOAD MODEL
# ============================================================

print("Loading model...")

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = torch.nn.Linear(
    num_features,
    len(class_names)
)


# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Ambil bobot model
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# 6. LOAD TEST EVENTS
# ============================================================

print("Loading test events...")


# keep_default_na=False penting
# karena "null" adalah label valid,
# bukan nilai kosong.

df = pd.read_csv(

    TEST_EVENTS_CSV,

    dtype=str,

    keep_default_na=False
)


# Bersihkan nama kolom
df.columns = (
    df.columns
    .str.strip()
)


# Bersihkan isi event_id
df["event_id"] = (
    df["event_id"]
    .str.strip()
)


# Bersihkan label
df["label"] = (
    df["label"]
    .str.strip()
)


print(
    "Total test events:",
    len(df)
)


# ============================================================
# 7. CHECK DATA
# ============================================================

print()
print("Checking event data...")

print(
    "Kolom CSV:",
    list(df.columns)
)

print(
    "Jumlah event leopard_cat:",
    (
        df["label"] == "leopard_cat"
    ).sum()
)

print(
    "Jumlah event null:",
    (
        df["label"] == "null"
    ).sum()
)


# ============================================================
# 8. FUNCTION PREDIKSI FOTO
# ============================================================

def predict_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    image = transform(
        image
    )


    image = image.unsqueeze(
        0
    )


    image = image.to(
        device
    )


    with torch.no_grad():

        outputs = model(
            image
        )


        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )


        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    predicted_class = (
        idx_to_class[
            prediction.item()
        ]
    )


    confidence = (
        confidence.item()
    )


    return (
        predicted_class,
        confidence
    )


# ============================================================
# 9. EVENT EVALUATION
# ============================================================

print()
print(
    "Starting event-level evaluation..."
)
print()


results = []

y_true = []

y_pred = []


# ============================================================
# LOOP SETIAP EVENT
# ============================================================

for _, row in df.iterrows():

    event_id = str(
        row["event_id"]
    ).strip()


    actual_label = str(
        row["label"]
    ).strip()


    # --------------------------------------------------------
    # Validasi event ID
    # --------------------------------------------------------

    if (
        event_id == ""
        or event_id.lower() == "nan"
    ):

        print(
            "WARNING: event_id kosong."
        )

        continue


    # --------------------------------------------------------
    # Validasi label
    # --------------------------------------------------------

    if actual_label not in class_names:

        print(
            f"WARNING: label tidak dikenali "
            f"untuk event {event_id}: "
            f"{actual_label}"
        )

        continue


    # --------------------------------------------------------
    # Ambil tiga foto event langsung dari CSV
    # --------------------------------------------------------

    image_paths = [
        str(row["foto_1"]).strip(),
        str(row["foto_2"]).strip(),
        str(row["foto_3"]).strip()
    ]


    # --------------------------------------------------------
    # Pastikan tepat 3 foto dan semuanya ada
    # --------------------------------------------------------

    image_paths = [
        path
        for path in image_paths
        if path
        and path.lower() != "nan"
        and os.path.exists(path)
    ]


    if len(image_paths) != 3:

        print(
            f"WARNING: {event_id} "
            f"memiliki {len(image_paths)} foto valid."
        )

        continue


    # --------------------------------------------------------
    # Pastikan tepat 3 foto
    # --------------------------------------------------------

    if len(image_paths) != 3:

        print(
            f"WARNING: {event_id} "
            f"memiliki {len(image_paths)} foto."
        )

        continue


    event_predictions = []

    event_confidences = []


    # ========================================================
    # PREDIKSI 3 FOTO
    # ========================================================

    for image_path in image_paths:

        predicted_class, confidence = (
            predict_image(
                image_path
            )
        )


        event_predictions.append(
            predicted_class
        )


        event_confidences.append(
            confidence
        )


    # ========================================================
    # MAJORITY VOTE
    # ========================================================

    leopard_count = (
        event_predictions.count(
            "leopard_cat"
        )
    )


    null_count = (
        event_predictions.count(
            "null"
        )
    )


    if leopard_count > null_count:

        event_prediction = (
            "leopard_cat"
        )

    else:

        event_prediction = (
            "null"
        )


    # ========================================================
    # EVENT CONFIDENCE
    # ========================================================

    event_confidence = np.mean(
        event_confidences
    )


    # ========================================================
    # CEK BENAR / SALAH
    # ========================================================

    correct = (
        event_prediction
        ==
        actual_label
    )


    # ========================================================
    # SIMPAN HASIL EVENT
    # ========================================================

    results.append({

        "event_id":
            event_id,

        "actual_label":
            actual_label,

        "foto_1_prediction":
            event_predictions[0],

        "foto_2_prediction":
            event_predictions[1],

        "foto_3_prediction":
            event_predictions[2],

        "event_prediction":
            event_prediction,

        "mean_confidence":
            event_confidence,

        "correct":
            correct
    })


    y_true.append(
        actual_label
    )


    y_pred.append(
        event_prediction
    )


# ============================================================
# 10. CHECK HASIL
# ============================================================

if len(y_true) == 0:

    print()
    print(
        "ERROR: Tidak ada event "
        "yang berhasil dievaluasi."
    )

    raise SystemExit


# ============================================================
# 11. SAVE EVENT PREDICTIONS
# ============================================================

results_df = pd.DataFrame(
    results
)


output_csv = os.path.join(

    OUTPUT_DIR,

    "event_predictions.csv"
)


results_df.to_csv(

    output_csv,

    index=False
)


# ============================================================
# 12. EVENT ACCURACY
# ============================================================

event_accuracy = np.mean(

    np.array(y_true)

    ==

    np.array(y_pred)
)


print()
print("=" * 60)

print(
    "EVENT-LEVEL EVALUATION"
)

print("=" * 60)


print(
    f"Event accuracy: "
    f"{event_accuracy * 100:.2f}%"
)


print(
    f"Total events evaluated: "
    f"{len(y_true)}"
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=class_names
)


print()
print(
    "Confusion Matrix"
)

print()


print(
    "                 Predicted"
)


print(
    "                 leopard_cat   null"
)


print(

    f"Actual leopard_cat   "
    f"{cm[0][0]:>5}       "
    f"{cm[0][1]:>5}"
)


print(

    f"Actual null          "
    f"{cm[1][0]:>5}       "
    f"{cm[1][1]:>5}"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    labels=class_names,

    target_names=class_names,

    digits=4,

    zero_division=0
)


print()

print(
    "Classification Report"
)

print()

print(
    report
)


# ============================================================
# 15. SAVE REPORT
# ============================================================

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
        "EVENT-LEVEL EVALUATION\n"
    )


    f.write(

        f"Event accuracy: "
        f"{event_accuracy * 100:.2f}%\n\n"

    )


    f.write(

        f"Total events evaluated: "
        f"{len(y_true)}\n\n"

    )


    f.write(
        "Confusion Matrix\n"
    )


    f.write(
        str(cm)
    )


    f.write(
        "\n\nClassification Report\n"
    )


    f.write(
        report
    )


# ============================================================
# 16. SAVE WRONG EVENTS
# ============================================================

wrong_events = results_df[
    results_df["correct"] == False
]


wrong_path = os.path.join(

    OUTPUT_DIR,

    "wrong_events.csv"
)


wrong_events.to_csv(

    wrong_path,

    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print()

print(
    "Event predictions saved to:"
)

print(
    output_csv
)


print()

print(
    "Report saved to:"
)

print(
    report_path
)


print()

print(
    "Wrong events saved to:"
)

print(
    wrong_path
)


print()

print("=" * 60)

print(
    "SELESAI"
)

print("=" * 60)
