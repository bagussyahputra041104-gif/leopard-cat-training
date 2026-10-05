import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "val"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv3_small_best.pth"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001

NUM_WORKERS = 0


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# 4. TRANSFORM
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


val_transform = transforms.Compose([

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
# 5. LOAD DATASET
# ============================================================

print()
print("Loading dataset...")


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print(
    "Train images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Class mapping:",
    train_dataset.class_to_idx
)


# ============================================================
# 6. DATA LOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 7. LOAD MOBILENETV3-SMALL
# ============================================================

print()
print("Loading MobileNetV3-Small...")


model = models.mobilenet_v3_small(
    weights="DEFAULT"
)


# ============================================================
# 8. GANTI CLASSIFIER
# ============================================================

num_features = (
    model.classifier[-1].in_features
)


model.classifier[-1] = nn.Linear(
    num_features,
    len(train_dataset.classes)
)


model = model.to(
    device
)


print(
    "Model loaded successfully."
)


# ============================================================
# 9. LOSS & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": []

}


best_val_accuracy = 0.0


# ============================================================
# 11. TRAINING LOOP
# ============================================================

print()
print("=" * 70)

print(
    "START TRAINING MOBILENETV3-SMALL"
)

print("=" * 70)


for epoch in range(
    NUM_EPOCHS
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        100.0
        *
        correct
        /
        total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item()
                *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss
        /
        val_total
    )


    val_accuracy = (
        100.0
        *
        val_correct
        /
        val_total
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

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


    # ========================================================
    # PRINT
    # ========================================================

    print(

        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}%"

    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save({

            "model_state_dict":
                model.state_dict(),

            "class_to_idx":
                train_dataset.class_to_idx,

            "image_size":
                IMAGE_SIZE,

            "best_val_accuracy":
                best_val_accuracy

        }, MODEL_PATH)


        print(
            "  -> Best model saved."
        )


# ============================================================
# 12. FINAL
# ============================================================

print()
print("=" * 70)

print(
    "TRAINING SELESAI"
)

print("=" * 70)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)


print(
    "Model saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 13. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "mobilenetv3_small_history.pt"
)


torch.save(
    history,
    history_path
)


print()
print(
    "Training history saved to:"
)

print(
    history_path
)

print()
print(
    "SELESAI"
)