import os
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TEST_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "orientation_dataset",
    "orientation_test_v2.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "03_MODEL",
    "models",
    "resnet18_orientation_v2_best.pth",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "orientation",
    "evaluation_orientation_resnet18_v2",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


PREDICTIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "orientation_test_predictions_v2.csv",
)

REPORT_TXT = os.path.join(
    OUTPUT_DIR,
    "orientation_classification_report_v2.txt",
)

CONFUSION_CSV = os.path.join(
    OUTPUT_DIR,
    "orientation_confusion_matrix_v2.csv",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("EVALUASI RESNET18 V2 - ORIENTATION")
print("=" * 70)

print(f"Device : {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

CLASS_NAMES = checkpoint["class_names"]
IMAGE_SIZE = checkpoint["image_size"]

print()
print("Model:")
print(MODEL_PATH)

print()
print("Augmentation:")
print(
    checkpoint.get(
        "augmentation",
        "unknown",
    )
)

print()
print("Classes:")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"  {index}: {class_name}"
    )


# ============================================================
# DATASET
# ============================================================

class OrientationTestDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None,
    ):

        self.df = pd.read_csv(
            csv_path,
            keep_default_na=False,
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = row["path_foto"]
        label = row["ground_truth"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File tidak ditemukan:\n{path}"
            )

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            label,
            row["event_id"],
            row["foto"],
            path,
        )


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset = OrientationTestDataset(
    TEST_CSV,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


print()
print("=" * 70)
print("TEST DATASET")
print("=" * 70)

print(
    f"Total foto : {len(test_dataset)}"
)

print(
    f"Total event: "
    f"{test_dataset.df['event_id'].nunique()}"
)


# ============================================================
# LOAD RESNET18
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES),
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

all_true = []
all_pred = []
all_confidence = []

all_event_ids = []
all_photo_names = []
all_paths = []


with torch.no_grad():

    for (
        images,
        labels,
        event_ids,
        photo_names,
        paths,
    ) in test_loader:

        images = images.to(
            DEVICE
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predictions = torch.max(
            probabilities,
            dim=1,
        )

        for i in range(
            len(labels)
        ):

            true_label = labels[i]

            predicted_label = (
                CLASS_NAMES[
                    predictions[
                        i
                    ].item()
                ]
            )

            all_true.append(
                true_label
            )

            all_pred.append(
                predicted_label
            )

            all_confidence.append(
                confidence[
                    i
                ].item()
            )

            all_event_ids.append(
                event_ids[i]
            )

            all_photo_names.append(
                photo_names[i]
            )

            all_paths.append(
                paths[i]
            )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred,
)

report = classification_report(
    all_true,
    all_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    zero_division=0,
)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=CLASS_NAMES,
)


# ============================================================
# PREDICTION DATAFRAME
# ============================================================

prediction_df = pd.DataFrame({

    "event_id":
        all_event_ids,

    "foto":
        all_photo_names,

    "path_foto":
        all_paths,

    "ground_truth":
        all_true,

    "prediction":
        all_pred,

    "confidence":
        all_confidence,
})


prediction_df["correct"] = (
    prediction_df[
        "ground_truth"
    ]
    ==
    prediction_df[
        "prediction"
    ]
)


prediction_df.to_csv(
    PREDICTIONS_CSV,
    index=False,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,

    index=CLASS_NAMES,

    columns=CLASS_NAMES,
)

cm_df.to_csv(
    CONFUSION_CSV
)


# ============================================================
# REPORT TXT
# ============================================================

correct_count = int(
    prediction_df[
        "correct"
    ].sum()
)

wrong_count = (
    len(prediction_df)
    - correct_count
)


with open(
    REPORT_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "EVALUASI RESNET18 V2 - ORIENTATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        "MODEL\n"
    )

    f.write(
        f"{MODEL_PATH}\n\n"
    )

    f.write(
        "AUGMENTATION\n"
    )

    f.write(
        "no_horizontal_flip\n\n"
    )

    f.write(
        f"Total test foto : "
        f"{len(test_dataset)}\n"
    )

    f.write(
        f"Total test event: "
        f"{test_dataset.df['event_id'].nunique()}\n"
    )

    f.write(
        f"Accuracy        : "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Correct         : "
        f"{correct_count}\n"
    )

    f.write(
        f"Wrong           : "
        f"{wrong_count}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        cm_df.to_string()
    )


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 70)
print("HASIL EVALUASI V2")
print("=" * 70)

print(
    f"Accuracy : "
    f"{accuracy:.4f}"
)

print(
    f"Correct  : "
    f"{correct_count}"
)

print(
    f"Wrong    : "
    f"{wrong_count}"
)

print()
print(
    "CLASSIFICATION REPORT"
)

print(
    "-" * 70
)

print(
    report
)

print(
    "CONFUSION MATRIX"
)

print(
    "-" * 70
)

print(
    cm_df
)

print()
print(
    "FILE OUTPUT"
)

print(
    "-" * 70
)

print(
    PREDICTIONS_CSV
)

print(
    REPORT_TXT
)

print(
    CONFUSION_CSV
)

print()
print(
    "EVALUASI V2 SELESAI."
)