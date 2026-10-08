import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "orientation_dataset",
    "orientation_train_v2.csv",
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "orientation_dataset",
    "orientation_val_v2.csv",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "03_MODEL",
    "models",
)

HISTORY_DIR = os.path.join(
    BASE_DIR,
    "03_MODEL",
    "history",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_orientation_v2_best.pth",
)

HISTORY_PATH = os.path.join(
    HISTORY_DIR,
    "resnet18_orientation_v2_history.pt",
)


# ============================================================
# PARAMETER
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "depan",
    "belakang",
    "kiri",
    "kanan",
    "tidak_yakin",
    "null",
]

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASS_NAMES)
}


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRAINING RESNET18 V2 - ORIENTATION")
print("=" * 70)

print("Perubahan V2:")
print("  - RandomHorizontalFlip DIHAPUS")
print("  - Split dataset tetap sama")
print("  - Test set tetap tidak digunakan")

print()
print(f"Device       : {DEVICE}")
print(f"Image size   : {IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print()
print("Classes:")

for index, class_name in enumerate(CLASS_NAMES):
    print(f"  {index}: {class_name}")


# ============================================================
# DATASET
# ============================================================

class OrientationDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(
            csv_path,
            keep_default_na=False,
        )

        self.transform = transform
        self.samples = []

        for _, row in self.df.iterrows():

            path = row["path_foto"]
            label = row["ground_truth"]

            if label not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Label tidak dikenal: {label}"
                )

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"File tidak ditemukan: {path}"
                )

            self.samples.append(
                (
                    path,
                    CLASS_TO_INDEX[label],
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

# PENTING:
# Tidak menggunakan RandomHorizontalFlip karena
# kiri dan kanan merupakan label yang berbeda.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = OrientationDataset(
    TRAIN_CSV,
    transform=train_transform,
)

val_dataset = OrientationDataset(
    VAL_CSV,
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(f"Train foto: {len(train_dataset)}")
print(f"Val foto  : {len(val_dataset)}")


# ============================================================
# MODEL
# ============================================================

print()
print("Memuat ResNet18 pretrained...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES),
)

model = model.to(DEVICE)


# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_val_accuracy = -1.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = (
        train_loss_total / train_total
    )

    train_accuracy = (
        train_correct / train_total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            val_loss_total += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "augmentation": "no_horizontal_flip",
            },
            MODEL_PATH,
        )

        print(
            f"  -> Best model disimpan "
            f"(Val Acc: {best_val_accuracy:.4f})"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

torch.save(
    history,
    HISTORY_PATH,
)


# ============================================================
# HASIL
# ============================================================

print()
print("=" * 70)
print("TRAINING V2 SELESAI")
print("=" * 70)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy:.4f}"
)

print()
print("Model:")
print(MODEL_PATH)

print()
print("History:")
print(HISTORY_PATH)

print()
print("Test set BELUM digunakan.")