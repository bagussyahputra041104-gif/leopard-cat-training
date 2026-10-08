import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET", "dataset", "leopard_cat_master_fixed.csv"
)

EMBEDDING_CSV = os.path.join(
    BASE_DIR,
    os.path.join("05_REID", "crop"),
    "event_crop_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    os.path.join("05_REID", "crop"),
    "cross_time"
)

ALL_PAIRS_CSV = os.path.join(
    OUTPUT_DIR,
    "cross_time_pairs.csv"
)

TOP_PAIRS_CSV = os.path.join(
    OUTPUT_DIR,
    "cross_time_top_pairs.csv"
)

# Jangan bandingkan event yang terlalu berdekatan.
# Kita mulai dari 1 jam.
MIN_TIME_GAP_SECONDS = 60 * 60

# Ambil kandidat teratas.
TOP_N = 30


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("CROSS-TIME LEOPARD CAT RE-ID")
print("=" * 70)

print()
print(
    f"Minimum time gap : "
    f"{MIN_TIME_GAP_SECONDS / 3600:.1f} jam"
)


# =========================================================
# LOAD MASTER
# =========================================================

master = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

master["timestamp_dt"] = pd.to_datetime(
    master["timestamp"],
    errors="coerce"
)

master = master[
    [
        "event_id",
        "timestamp",
        "timestamp_dt"
    ]
].copy()


# =========================================================
# LOAD EMBEDDING
# =========================================================

embedding_df = pd.read_csv(
    EMBEDDING_CSV,
    dtype=str,
    keep_default_na=False
)

embedding_df = embedding_df.merge(
    master,
    on="event_id",
    how="left"
)

embedding_df["timestamp_dt"] = pd.to_datetime(
    embedding_df["timestamp_dt"],
    errors="coerce"
)

# Cari kolom embedding
embedding_columns = [
    column
    for column in embedding_df.columns
    if column.startswith("emb_")
]

print()
print(
    f"Event dengan embedding : "
    f"{len(embedding_df)}"
)

print(
    f"Dimensi embedding      : "
    f"{len(embedding_columns)}"
)


# =========================================================
# BUILD MATRIX
# =========================================================

X = embedding_df[
    embedding_columns
].astype(float).values

event_ids = embedding_df[
    "event_id"
].tolist()

timestamps = embedding_df[
    "timestamp_dt"
].tolist()


# =========================================================
# SIMILARITY
# =========================================================

print()
print(
    "Menghitung cosine similarity..."
)

similarity_matrix = cosine_similarity(
    X
)


# =========================================================
# BUILD CROSS-TIME PAIRS
# =========================================================

pairs = []

for i in range(
    len(event_ids)
):

    for j in range(
        i + 1,
        len(event_ids)
    ):

        time_1 = timestamps[i]
        time_2 = timestamps[j]

        if pd.isna(time_1) or pd.isna(time_2):
            continue

        time_gap_seconds = abs(
            (
                time_2 -
                time_1
            ).total_seconds()
        )

        # Skip event yang terlalu berdekatan
        if (
            time_gap_seconds <
            MIN_TIME_GAP_SECONDS
        ):
            continue

        similarity = float(
            similarity_matrix[i, j]
        )

        pairs.append({
            "event_1": event_ids[i],
            "event_2": event_ids[j],

            "timestamp_1": time_1,
            "timestamp_2": time_2,

            "time_gap_seconds":
                time_gap_seconds,

            "time_gap_hours":
                time_gap_seconds / 3600,

            "time_gap_days":
                time_gap_seconds / 86400,

            "similarity":
                similarity
        })


# =========================================================
# DATAFRAME
# =========================================================

pairs_df = pd.DataFrame(
    pairs
)

if len(pairs_df) == 0:

    print()
    print(
        "TIDAK ADA pasangan cross-time."
    )

    raise SystemExit


pairs_df = pairs_df.sort_values(
    by="similarity",
    ascending=False
).reset_index(
    drop=True
)


# =========================================================
# SAVE ALL
# =========================================================

pairs_df.to_csv(
    ALL_PAIRS_CSV,
    index=False
)


# =========================================================
# TOP PAIRS
# =========================================================

top_pairs = pairs_df.head(
    TOP_N
).copy()

top_pairs.to_csv(
    TOP_PAIRS_CSV,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("=" * 70)
print("CROSS-TIME CANDIDATE PAIRS")
print("=" * 70)

print()

display_columns = [
    "event_1",
    "event_2",
    "time_gap_hours",
    "time_gap_days",
    "similarity"
]

print(
    top_pairs[
        display_columns
    ].to_string(
        index=False
    )
)


# =========================================================
# TIME GAP DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("RENTANG WAKTU PASANGAN")
print("=" * 70)

print()

print(
    f"Total pasangan cross-time : "
    f"{len(pairs_df)}"
)

print(
    f"Minimum time gap          : "
    f"{pairs_df['time_gap_hours'].min():.2f} jam"
)

print(
    f"Median time gap           : "
    f"{pairs_df['time_gap_hours'].median():.2f} jam"
)

print(
    f"Maximum time gap          : "
    f"{pairs_df['time_gap_hours'].max():.2f} jam"
)


# =========================================================
# SIMILARITY DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("RENTANG SIMILARITY")
print("=" * 70)

print()

print(
    f"Similarity minimum : "
    f"{pairs_df['similarity'].min():.4f}"
)

print(
    f"Similarity median  : "
    f"{pairs_df['similarity'].median():.4f}"
)

print(
    f"Similarity maximum : "
    f"{pairs_df['similarity'].max():.4f}"
)


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 70)
print("SELESAI")
print("=" * 70)

print()
print("Semua pasangan:")
print(
    ALL_PAIRS_CSV
)

print()
print("Top candidate:")
print(
    TOP_PAIRS_CSV
)

