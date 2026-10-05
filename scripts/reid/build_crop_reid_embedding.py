import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

DETECTION_CSV = os.path.join(
    BASE_DIR,
    "reid_crops",
    "detection_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "05_REID" / "crop"
)

EMBEDDING_CSV = os.path.join(
    OUTPUT_DIR,
    "event_crop_embeddings.csv"
)

SIMILARITY_CSV = os.path.join(
    OUTPUT_DIR,
    "event_crop_similarity.csv"
)

PAIRS_CSV = os.path.join(
    OUTPUT_DIR,
    "most_similar_crop_pairs.csv"
)

MIN_CONFIDENCE = 0.50

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 60)
print("CROP-BASED LEOPARD CAT RE-ID BASELINE")
print("=" * 60)

print()
print(f"Device : {DEVICE}")

print(
    f"Minimum YOLO confidence : "
    f"{MIN_CONFIDENCE}"
)


# =========================================================
# LOAD DETECTIONS
# =========================================================

df = pd.read_csv(
    DETECTION_CSV,
    dtype=str,
    keep_default_na=False
)

df["confidence_num"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)

df = df[
    df["detected"].str.lower() == "true"
].copy()

before_filter = len(df)

df = df[
    df["confidence_num"] >= MIN_CONFIDENCE
].copy()

after_filter = len(df)

print()
print("=" * 60)
print("FILTER DETECTION")
print("=" * 60)

print(
    f"Detection sebelum filter : "
    f"{before_filter}"
)

print(
    f"Detection sesudah filter : "
    f"{after_filter}"
)

print(
    f"Confidence minimum       : "
    f"{MIN_CONFIDENCE}"
)


# =========================================================
# LOAD RESNET18
# =========================================================

print()
print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Hilangkan classifier.
model.fc = nn.Identity()

model = model.to(DEVICE)

model.eval()

print("Model siap.")


# =========================================================
# TRANSFORM
# =========================================================

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


# =========================================================
# FUNCTION
# =========================================================

def get_embedding(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        embedding = model(
            tensor
        )

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:

        embedding = (
            embedding / norm
        )

    return embedding


# =========================================================
# BUILD EVENT EMBEDDINGS
# =========================================================

event_embeddings = []

failed_crops = []

print()
print("=" * 60)
print("MEMBANGUN CROP EMBEDDING")
print("=" * 60)


for event_id, group in df.groupby(
    "event_id"
):

    embeddings = []

    print(
        f"{event_id} | "
        f"{len(group)} crop",
        end=" ... "
    )

    for _, row in group.iterrows():

        crop_path = row[
            "crop_path"
        ]

        if not os.path.exists(
            crop_path
        ):

            failed_crops.append(
                crop_path
            )

            continue

        try:

            embedding = get_embedding(
                crop_path
            )

            embeddings.append(
                embedding
            )

        except Exception as e:

            failed_crops.append(
                crop_path
            )

            print(
                f"\n  ERROR: {e}"
            )

    if len(embeddings) == 0:

        print(
            "TIDAK ADA EMBEDDING"
        )

        continue

    # -----------------------------------------------------
    # Average crop embeddings
    # -----------------------------------------------------

    event_embedding = np.mean(
        embeddings,
        axis=0
    )

    # Normalize again
    norm = np.linalg.norm(
        event_embedding
    )

    if norm > 0:

        event_embedding = (
            event_embedding / norm
        )

    event_embeddings.append({
        "event_id": event_id,
        "num_crops": len(
            embeddings
        ),
        "embedding": event_embedding
    })

    print("OK")


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("RINGKASAN EMBEDDING")
print("=" * 60)

print(
    f"Event dengan embedding : "
    f"{len(event_embeddings)}"
)

print(
    f"Event tanpa embedding  : "
    f"{63 - len(event_embeddings)}"
)

print(
    f"Crop gagal             : "
    f"{len(failed_crops)}"
)


# =========================================================
# SAVE EVENT EMBEDDINGS
# =========================================================

embedding_rows = []

for item in event_embeddings:

    row = {
        "event_id": item["event_id"],
        "num_crops": item["num_crops"]
    }

    for i, value in enumerate(
        item["embedding"]
    ):

        row[
            f"emb_{i + 1}"
        ] = float(value)

    embedding_rows.append(
        row
    )


embedding_df = pd.DataFrame(
    embedding_rows
)

embedding_df.to_csv(
    EMBEDDING_CSV,
    index=False
)


# =========================================================
# SIMILARITY
# =========================================================

X = np.vstack([
    item["embedding"]
    for item in event_embeddings
])

event_ids = [
    item["event_id"]
    for item in event_embeddings
]

similarity = cosine_similarity(
    X
)

similarity_df = pd.DataFrame(
    similarity,
    index=event_ids,
    columns=event_ids
)

similarity_df.to_csv(
    SIMILARITY_CSV
)


# =========================================================
# TOP PAIRS
# =========================================================

pairs = []

for i in range(
    len(event_ids)
):

    for j in range(
        i + 1,
        len(event_ids)
    ):

        pairs.append({
            "event_1": event_ids[i],
            "event_2": event_ids[j],
            "similarity": float(
                similarity[i, j]
            )
        })


pairs_df = pd.DataFrame(
    pairs
)

pairs_df = pairs_df.sort_values(
    by="similarity",
    ascending=False
)

top_pairs = pairs_df.head(
    30
)

top_pairs.to_csv(
    PAIRS_CSV,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("=" * 60)
print("TOP 30 CROP-BASED SIMILARITY")
print("=" * 60)

print()

print(
    top_pairs.to_string(
        index=False
    )
)


# =========================================================
# EVENT COVERAGE
# =========================================================

coverage = (
    embedding_df[
        [
            "event_id",
            "num_crops"
        ]
    ]
    .sort_values(
        by="num_crops",
        ascending=False
    )
)

print()
print("=" * 60)
print("JUMLAH CROP PER EVENT")
print("=" * 60)

print()

print(
    coverage.to_string(
        index=False
    )
)


# =========================================================
# SAVE FAILED
# =========================================================

if failed_crops:

    failed_path = os.path.join(
        OUTPUT_DIR,
        "failed_crops.txt"
    )

    with open(
        failed_path,
        "w",
        encoding="utf-8"
    ) as f:

        for path in failed_crops:

            f.write(
                path + "\n"
            )


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print()
print(
    "Embedding:"
)

print(
    EMBEDDING_CSV
)

print()
print(
    "Similarity:"
)

print(
    SIMILARITY_CSV
)

print()
print(
    "Top pairs:"
)

print(
    PAIRS_CSV
)