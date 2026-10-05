import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "02_DATASET" / "dataset"
MODEL_DIR = BASE_DIR / "03_MODEL" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# CEK DATASET
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

required_dirs = [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]

for folder in required_dirs:

    if not folder.exists():

        print()
        print("ERROR: Folder dataset tidak ditemukan:")
        print(folder)
        raise SystemExit


# ============================================================
# TRANSFORMASI
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.RandomHorizontalFlip(),
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


eval_transform = transforms.Compose([
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
# DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)


# ============================================================
# DATA LOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# INFORMASI DATASET
# ============================================================

print()
print("=" * 70)
print("BASELINE IMAGE CLASSIFICATION")
print("=" * 70)

print()
print("Device:")
print(DEVICE)

print()
print("Class:")
print(train_dataset.class_to_idx)

print()
print("Jumlah foto:")
print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")


# ============================================================
# MODEL RESNET18
# ============================================================

print()
print("Mempersiapkan ResNet18 pretrained...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Ganti classifier terakhir menjadi 2 kelas
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(DEVICE)


# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FUNGSI TRAINING
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# FUNGSI VALIDASI
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    loss = running_loss / total

    accuracy = correct / total

    return loss, accuracy


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_model_state = copy.deepcopy(
    model.state_dict()
)

print()
print("=" * 70)
print("MULAI TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        evaluate(val_loader)
    )

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Simpan model terbaik berdasarkan validation accuracy
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        print(
            "-> Model terbaik diperbarui."
        )


# ============================================================
# KEMBALIKAN MODEL TERBAIK
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# SIMPAN MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "resnet18_baseline_best.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_to_idx":
            train_dataset.class_to_idx,

        "image_size":
            IMAGE_SIZE,

        "best_val_accuracy":
            best_val_accuracy
    },
    MODEL_PATH
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("EVALUASI TEST SET")
print("=" * 70)

test_loss, test_accuracy = (
    evaluate(test_loader)
)

print()
print(
    f"Test Loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SELESAI
# ============================================================

print()
print("=" * 70)
print("TRAINING SELESAI")
print("=" * 70)

print()
print("Model terbaik:")
print(MODEL_PATH)

print()
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("CATATAN:")
print("- Model ini adalah baseline awal.")
print("- Kelas: leopard_cat dan null.")
print("- Belum melakukan individual identification.")
print("- Evaluasi berikutnya perlu melihat confusion matrix.")
print("- Jangan menyimpulkan performa hanya dari accuracy.")