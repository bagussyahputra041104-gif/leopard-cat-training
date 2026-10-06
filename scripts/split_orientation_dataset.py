import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

GROUND_TRUTH_CSV = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "orientation",
    "orientation",
    "orientation_ground_truth.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "orientation_dataset",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

df = pd.read_csv(
    GROUND_TRUTH_CSV,
    keep_default_na=False,
)

print("=" * 60)
print("SPLIT DATASET ORIENTATION")
print("=" * 60)

print(f"Total foto       : {len(df)}")
print(f"Total event      : {df['event_id'].nunique()}")

# Pastikan semua sudah confirmed
pending = df[df["status_review"] != "CONFIRMED"]

if len(pending) > 0:
    raise ValueError(
        f"Masih ada {len(pending)} data yang belum CONFIRMED."
    )

# Pastikan tidak ada ground truth kosong
empty_gt = df[df["ground_truth"].str.strip() == ""]

if len(empty_gt) > 0:
    raise ValueError(
        f"Masih ada {len(empty_gt)} ground truth kosong."
    )

# ============================================================
# EVENT LEVEL
# ============================================================

event_df = (
    df[
        [
            "event_id",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Ambil label event dari tiga foto.
# Satu event harus masuk split yang sama.
event_labels = (
    df.groupby("event_id")["ground_truth"]
    .apply(lambda x: "|".join(sorted(set(x))))
    .reset_index(name="orientation_labels")
)

event_df = event_df.merge(
    event_labels,
    on="event_id",
    how="left",
)

# ============================================================
# SPLIT EVENT
# ============================================================

train_events, temp_events = train_test_split(
    event_df,
    test_size=0.30,
    random_state=42,
)

val_events, test_events = train_test_split(
    temp_events,
    test_size=0.50,
    random_state=42,
)

train_ids = set(train_events["event_id"])
val_ids = set(val_events["event_id"])
test_ids = set(test_events["event_id"])

# ============================================================
# VALIDASI TIDAK ADA OVERLAP
# ============================================================

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

# ============================================================
# MASUKKAN FOTO KE SPLIT
# ============================================================

def assign_split(event_id):
    if event_id in train_ids:
        return "train"

    if event_id in val_ids:
        return "val"

    if event_id in test_ids:
        return "test"

    raise ValueError(f"Event tidak dikenal: {event_id}")


df["split"] = df["event_id"].apply(assign_split)

# ============================================================
# SIMPAN CSV
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

train_csv = os.path.join(
    OUTPUT_DIR,
    "orientation_train.csv",
)

val_csv = os.path.join(
    OUTPUT_DIR,
    "orientation_val.csv",
)

test_csv = os.path.join(
    OUTPUT_DIR,
    "orientation_test.csv",
)

all_csv = os.path.join(
    OUTPUT_DIR,
    "orientation_dataset_split.csv",
)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)
df.to_csv(all_csv, index=False)

# ============================================================
# SUMMARY
# ============================================================

print()
print("EVENT SPLIT")
print("-" * 60)
print(f"Train : {len(train_ids)} event")
print(f"Val   : {len(val_ids)} event")
print(f"Test  : {len(test_ids)} event")

print()
print("PHOTO SPLIT")
print("-" * 60)
print(f"Train : {len(train_df)} foto")
print(f"Val   : {len(val_df)} foto")
print(f"Test  : {len(test_df)} foto")

print()
print("DISTRIBUSI LABEL")
print("-" * 60)

for split_name, split_df in [
    ("TRAIN", train_df),
    ("VAL", val_df),
    ("TEST", test_df),
]:
    print()
    print(split_name)
    print(split_df["ground_truth"].value_counts())

print()
print("CEK EVENT PER SPLIT")
print("-" * 60)

print(
    "Train event:",
    train_df["event_id"].nunique(),
)

print(
    "Val event  :",
    val_df["event_id"].nunique(),
)

print(
    "Test event :",
    test_df["event_id"].nunique(),
)

print()
print("FILE OUTPUT")
print("-" * 60)
print(train_csv)
print(val_csv)
print(test_csv)
print(all_csv)

print()
print("SPLIT DATASET BERHASIL.")