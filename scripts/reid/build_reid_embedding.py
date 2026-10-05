import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "05_REID" / "baseline"
)

EMBEDDING_CSV = os.path.join(
    OUTPUT_DIR,
    "event_embeddings.csv"
)

SIMILARITY_CSV = os.path.join(
    OUTPUT_DIR,
    "event_similarity.csv"
)

CLUSTER_CSV = os.path.join(
    OUTPUT_DIR,
    "event_clusters.csv"
)

HEATMAP_PATH = os.path.join(
    OUTPUT_DIR,
    "similarity_heatmap.png"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224


# =========================================================
# SETUP
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("LEOPARD CAT RE-ID BASELINE")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Master : {MASTER_CSV}")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

print()
print(f"Total event : {len(df)}")

required_columns = [
    "event_id",
    "timestamp",
    "foto_1",
    "foto_2",
    "foto_3"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Kolom '{col}' tidak ditemukan di leopard_cat_master.csv"
        )


# =========================================================
# LOAD PRETRAINED MODEL
# =========================================================

print()
print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Buang classifier terakhir.
# Output sekarang berupa feature vector 512 dimensi.
model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()

print("Model siap.")


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
# FUNCTION: IMAGE EMBEDDING
# =========================================================

def get_image_embedding(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            embedding = model(tensor)

        embedding = embedding.squeeze(0).cpu().numpy()

        # L2 normalization
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:

        print()
        print(f"[ERROR] Gagal membaca:")
        print(image_path)
        print(f"Alasan: {e}")

        return None


# =========================================================
# BUILD EVENT EMBEDDING
# =========================================================

event_embeddings = []

print()
print("=" * 60)
print("MEMBANGUN EVENT EMBEDDING")
print("=" * 60)

for index, row in df.iterrows():

    event_id = row["event_id"]

    print(
        f"[{index + 1:02d}/{len(df)}] "
        f"{event_id}",
        end=" ... "
    )

    photo_paths = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"]
    ]

    photo_embeddings = []

    for path in photo_paths:

        if not os.path.exists(path):
            print()
            print(f"[WARNING] File tidak ditemukan:")
            print(path)
            continue

        embedding = get_image_embedding(path)

        if embedding is not None:
            photo_embeddings.append(embedding)

    if len(photo_embeddings) == 0:

        print("GAGAL")
        continue

    # Gabungkan 3 foto menjadi 1 embedding event
    event_embedding = np.mean(
        photo_embeddings,
        axis=0
    )

    # Normalisasi lagi setelah averaging
    norm = np.linalg.norm(event_embedding)

    if norm > 0:
        event_embedding = event_embedding / norm

    event_embeddings.append({
        "event_id": event_id,
        "timestamp": row["timestamp"],
        "embedding": event_embedding
    })

    print("OK")


# =========================================================
# CHECK RESULT
# =========================================================

if len(event_embeddings) == 0:
    raise RuntimeError(
        "Tidak ada embedding yang berhasil dibuat."
    )

print()
print("=" * 60)
print("EMBEDDING SELESAI")
print("=" * 60)

print(
    f"Event berhasil : {len(event_embeddings)}"
)

print(
    f"Dimensi        : "
    f"{len(event_embeddings[0]['embedding'])}"
)


# =========================================================
# SAVE EMBEDDING CSV
# =========================================================

embedding_rows = []

for item in event_embeddings:

    row = {
        "event_id": item["event_id"],
        "timestamp": item["timestamp"]
    }

    for i, value in enumerate(item["embedding"]):

        row[f"emb_{i + 1}"] = float(value)

    embedding_rows.append(row)


embedding_df = pd.DataFrame(
    embedding_rows
)

embedding_df.to_csv(
    EMBEDDING_CSV,
    index=False
)

print()
print(f"Embedding CSV:")
print(EMBEDDING_CSV)


# =========================================================
# SIMILARITY MATRIX
# =========================================================

matrix = np.vstack([
    item["embedding"]
    for item in event_embeddings
])

event_ids = [
    item["event_id"]
    for item in event_embeddings
]

similarity_matrix = cosine_similarity(matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=event_ids,
    columns=event_ids
)

similarity_df.to_csv(
    SIMILARITY_CSV
)

print()
print(f"Similarity matrix:")
print(SIMILARITY_CSV)


# =========================================================
# HEATMAP
# =========================================================

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    similarity_matrix,
    interpolation="nearest"
)

plt.colorbar(
    label="Cosine Similarity"
)

plt.xticks(
    range(len(event_ids)),
    event_ids,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(event_ids)),
    event_ids,
    fontsize=6
)

plt.title(
    "Leopard Cat Event Similarity"
)

plt.tight_layout()

plt.savefig(
    HEATMAP_PATH,
    dpi=200
)

plt.close()

print()
print(f"Heatmap:")
print(HEATMAP_PATH)


# =========================================================
# CLUSTERING
# =========================================================

print()
print("=" * 60)
print("CLUSTERING")
print("=" * 60)

# Jarak = 1 - similarity
distance_matrix = 1 - similarity_matrix

# Pastikan nilai numerik aman
distance_matrix = np.clip(
    distance_matrix,
    0,
    2
)

# Threshold awal.
# Ini BUKAN jumlah individu sebenarnya.
# Hanya digunakan untuk membuat kandidat grup.
DISTANCE_THRESHOLD = 0.35

cluster_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=DISTANCE_THRESHOLD,
    metric="precomputed",
    linkage="average"
)

cluster_labels = cluster_model.fit_predict(
    distance_matrix
)


# =========================================================
# SAVE CLUSTERS
# =========================================================

cluster_df = pd.DataFrame({
    "event_id": event_ids,
    "cluster": cluster_labels
})

cluster_df = cluster_df.sort_values(
    by="cluster"
)

cluster_df.to_csv(
    CLUSTER_CSV,
    index=False
)

print(
    f"Jumlah candidate cluster : "
    f"{cluster_df['cluster'].nunique()}"
)

print()
print("Distribusi cluster:")

print(
    cluster_df["cluster"]
    .value_counts()
    .sort_index()
)


# =========================================================
# PRINT CLUSTER CONTENT
# =========================================================

print()
print("=" * 60)
print("HASIL CANDIDATE CLUSTER")
print("=" * 60)

for cluster_id, group in cluster_df.groupby(
    "cluster"
):

    events = group["event_id"].tolist()

    print()
    print(
        f"Cluster {cluster_id}: "
        f"{len(events)} event"
    )

    print(
        ", ".join(events)
    )


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print()
print("File hasil:")

print(
    f"1. {EMBEDDING_CSV}"
)

print(
    f"2. {SIMILARITY_CSV}"
)

print(
    f"3. {CLUSTER_CSV}"
)

print(
    f"4. {HEATMAP_PATH}"
)

print()
print(
    "CATATAN:"
)

print(
    "Cluster di atas adalah kandidat berdasarkan "
    "kemiripan embedding, bukan identitas individu "
    "yang sudah tervalidasi."
)