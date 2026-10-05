import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "02_DATASET" / "dataset"

SPLITS = ["train", "val", "test"]
LABELS = ["leopard_cat", "null"]


# ============================================================
# FUNGSI MENCARI FOLDER SUMBER
# ============================================================

def get_source_folder(photo_path):

    photo_path = Path(photo_path)

    try:
        relative_path = photo_path.relative_to(BASE_DIR)
        parts = relative_path.parts
    except ValueError:
        parts = photo_path.parts

    if len(parts) >= 2:
        return parts[-2]

    return "Tidak diketahui"


# ============================================================
# BACA CSV EVENT SETIAP SPLIT
# ============================================================

split_events = defaultdict(list)

for split in SPLITS:

    csv_file = (
        DATASET_DIR
        / f"{split}_events.csv"
    )

    if not csv_file.exists():

        print()
        print(f"ERROR: {csv_file} tidak ditemukan.")
        raise SystemExit

    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            source = get_source_folder(
                row["foto_1"]
            )

            split_events[split].append({
                "event_id": row["event_id"],
                "label": row["label"],
                "source": source
            })


# ============================================================
# HITUNG DISTRIBUSI
# ============================================================

distribution = defaultdict(
    lambda: {
        "train": 0,
        "val": 0,
        "test": 0,
        "leopard_cat": 0,
        "null": 0
    }
)

for split in SPLITS:

    for event in split_events[split]:

        source = event["source"]

        distribution[source][split] += 1
        distribution[source][event["label"]] += 1


# ============================================================
# TAMPILKAN HASIL
# ============================================================

print()
print("=" * 100)
print("AUDIT DISTRIBUSI SOURCE FOLDER PER SPLIT")
print("=" * 100)

print()

print(
    f"{'Source Folder':45}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

print("-" * 100)


for source in sorted(distribution):

    data = distribution[source]

    total = (
        data["train"]
        + data["val"]
        + data["test"]
    )

    print(
        f"{source[:45]:45}"
        f"{data['train']:>10}"
        f"{data['val']:>10}"
        f"{data['test']:>10}"
        f"{total:>10}"
    )


# ============================================================
# DETAIL LABEL
# ============================================================

print()
print("=" * 100)
print("DETAIL LABEL PER SOURCE")
print("=" * 100)

print()

print(
    f"{'Source Folder':45}"
    f"{'Leopard':>12}"
    f"{'Null':>12}"
    f"{'Total':>12}"
)

print("-" * 100)


for source in sorted(distribution):

    data = distribution[source]

    total = (
        data["leopard_cat"]
        + data["null"]
    )

    print(
        f"{source[:45]:45}"
        f"{data['leopard_cat']:>12}"
        f"{data['null']:>12}"
        f"{total:>12}"
    )


# ============================================================
# CEK SOURCE YANG MUNCUL DI LEBIH DARI SATU SPLIT
# ============================================================

print()
print("=" * 100)
print("SOURCE YANG TERSEBAR DI BEBERAPA SPLIT")
print("=" * 100)

print()

shared_sources = []

for source, data in sorted(distribution.items()):

    splits_present = []

    for split in SPLITS:

        if data[split] > 0:
            splits_present.append(split)

    if len(splits_present) > 1:

        shared_sources.append(
            (source, splits_present)
        )

        print(
            f"{source[:45]:45} -> "
            f"{', '.join(splits_present)}"
        )


if not shared_sources:

    print(
        "Tidak ada source folder yang "
        "tersebar di beberapa split."
    )


# ============================================================
# SOURCE YANG HANYA ADA DI TEST
# ============================================================

print()
print("=" * 100)
print("SOURCE YANG HANYA MUNCUL DI TEST")
print("=" * 100)

print()

test_only = []

for source, data in sorted(distribution.items()):

    if (
        data["test"] > 0
        and data["train"] == 0
        and data["val"] == 0
    ):

        test_only.append(source)

        print(
            f"{source} "
            f"({data['test']} event)"
        )


if not test_only:

    print("Tidak ada.")


# ============================================================
# SOURCE YANG HANYA ADA DI TRAIN
# ============================================================

print()
print("=" * 100)
print("SOURCE YANG HANYA MUNCUL DI TRAIN")
print("=" * 100)

print()

train_only = []

for source, data in sorted(distribution.items()):

    if (
        data["train"] > 0
        and data["val"] == 0
        and data["test"] == 0
    ):

        train_only.append(source)

        print(
            f"{source} "
            f"({data['train']} event)"
        )


if not train_only:

    print("Tidak ada.")


# ============================================================
# SELESAI
# ============================================================

print()
print("=" * 100)
print("AUDIT SELESAI")
print("=" * 100)