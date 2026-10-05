import csv
import random
import shutil
from pathlib import Path
from collections import Counter


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "event_dataset.csv"

OUTPUT_DIR = BASE_DIR / "02_DATASET" / "dataset"

# Proporsi pembagian event
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Agar hasil pembagian selalu sama setiap kali script dijalankan
RANDOM_SEED = 42


# ============================================================
# CEK KONFIGURASI
# ============================================================

total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

if abs(total_ratio - 1.0) > 0.0001:
    print("ERROR: TRAIN + VAL + TEST harus berjumlah 1.0")
    raise SystemExit


# ============================================================
# CEK FILE CSV
# ============================================================

if not CSV_FILE.exists():
    print("ERROR: event_dataset.csv tidak ditemukan.")
    print(f"Lokasi: {CSV_FILE}")
    raise SystemExit


# ============================================================
# BACA CSV
# ============================================================

events = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    required_columns = {
        "event_id",
        "label",
        "foto_1",
        "foto_2",
        "foto_3",
        "timestamp"
    }

    actual_columns = set(reader.fieldnames or [])

    missing_columns = required_columns - actual_columns

    if missing_columns:
        print("ERROR: Kolom CSV tidak lengkap.")
        print(f"Kolom ditemukan : {reader.fieldnames}")
        print(f"Kolom hilang    : {sorted(missing_columns)}")
        raise SystemExit

    for row in reader:
        events.append(row)


# ============================================================
# VALIDASI EVENT
# ============================================================

print()
print("=" * 80)
print("VALIDASI EVENT")
print("=" * 80)

print(f"Total event ditemukan : {len(events)}")

event_ids = [event["event_id"] for event in events]

duplicate_ids = [
    event_id
    for event_id, count in Counter(event_ids).items()
    if count > 1
]

if duplicate_ids:
    print()
    print("ERROR: Ditemukan event_id duplikat:")
    for event_id in duplicate_ids:
        print(f"  - {event_id}")
    raise SystemExit

print("Event ID duplikat     : 0")


# ============================================================
# VALIDASI LABEL
# ============================================================

valid_labels = {
    "leopard_cat",
    "null"
}

invalid_labels = [
    event
    for event in events
    if event["label"] not in valid_labels
]

if invalid_labels:

    print()
    print("ERROR: Ada label yang tidak dikenal:")

    for event in invalid_labels:
        print(
            f"  {event['event_id']} -> "
            f"{event['label']}"
        )

    raise SystemExit

print("Label valid            : YA")


# ============================================================
# VALIDASI 3 FOTO SETIAP EVENT
# ============================================================

missing_photos = []

for event in events:

    for column in ["foto_1", "foto_2", "foto_3"]:

        photo_path = Path(event[column])

        if not photo_path.exists():

            missing_photos.append(
                (
                    event["event_id"],
                    column,
                    str(photo_path)
                )
            )


if missing_photos:

    print()
    print("ERROR: Ada foto yang tidak ditemukan:")

    for event_id, column, path in missing_photos[:20]:

        print(
            f"  {event_id} | "
            f"{column} | "
            f"{path}"
        )

    print()
    print(
        f"Total foto bermasalah: "
        f"{len(missing_photos)}"
    )

    raise SystemExit

print("Semua foto event       : DITEMUKAN")


# ============================================================
# KELOMPOKKAN EVENT BERDASARKAN LABEL
# ============================================================

events_by_label = {
    "leopard_cat": [],
    "null": []
}

for event in events:
    events_by_label[event["label"]].append(event)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# FUNGSI PEMBAGIAN EVENT
# ============================================================

def split_events(event_list):

    shuffled = event_list.copy()

    random.shuffle(shuffled)

    total = len(shuffled)

    train_count = round(total * TRAIN_RATIO)
    val_count = round(total * VAL_RATIO)

    # Pastikan test mendapat sisa
    test_count = total - train_count - val_count

    train_events = shuffled[:train_count]

    val_events = shuffled[
        train_count:
        train_count + val_count
    ]

    test_events = shuffled[
        train_count + val_count:
    ]

    return (
        train_events,
        val_events,
        test_events
    )


# ============================================================
# BAGI MASING-MASING LABEL
# ============================================================

train_events = []
val_events = []
test_events = []

for label in ["leopard_cat", "null"]:

    train, val, test = split_events(
        events_by_label[label]
    )

    train_events.extend(train)
    val_events.extend(val)
    test_events.extend(test)


# ============================================================
# ACAK URUTAN EVENT DALAM SETIAP SPLIT
# ============================================================

random.shuffle(train_events)
random.shuffle(val_events)
random.shuffle(test_events)


# ============================================================
# RINGKASAN PEMBAGIAN
# ============================================================

def count_labels(event_list):

    counter = Counter(
        event["label"]
        for event in event_list
    )

    return (
        counter["leopard_cat"],
        counter["null"]
    )


train_leopard, train_null = count_labels(train_events)
val_leopard, val_null = count_labels(val_events)
test_leopard, test_null = count_labels(test_events)


# ============================================================
# TAMPILKAN PEMBAGIAN
# ============================================================

print()
print("=" * 80)
print("PEMBAGIAN EVENT")
print("=" * 80)

print()
print(
    f"{'Split':15}"
    f"{'Total Event':>15}"
    f"{'Leopard Cat':>15}"
    f"{'Null':>15}"
)

print("-" * 60)

print(
    f"{'Train':15}"
    f"{len(train_events):>15}"
    f"{train_leopard:>15}"
    f"{train_null:>15}"
)

print(
    f"{'Validation':15}"
    f"{len(val_events):>15}"
    f"{val_leopard:>15}"
    f"{val_null:>15}"
)

print(
    f"{'Test':15}"
    f"{len(test_events):>15}"
    f"{test_leopard:>15}"
    f"{test_null:>15}"
)

print("-" * 60)

print(
    f"{'TOTAL':15}"
    f"{len(events):>15}"
    f"{train_leopard + val_leopard + test_leopard:>15}"
    f"{train_null + val_null + test_null:>15}"
)


# ============================================================
# HAPUS DATASET LAMA JIKA ADA
# ============================================================

if OUTPUT_DIR.exists():

    print()
    print("Folder dataset lama ditemukan.")
    print("Menghapus dataset lama agar hasil bersih...")

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# BUAT STRUKTUR FOLDER
# ============================================================

for split in ["train", "val", "test"]:

    for label in ["leopard_cat", "null"]:

        folder = (
            OUTPUT_DIR
            / split
            / label
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# SALIN FOTO
# ============================================================

def copy_events(event_list, split_name):

    copied_count = 0

    for event in event_list:

        label = event["label"]
        event_id = event["event_id"]

        destination_folder = (
            OUTPUT_DIR
            / split_name
            / label
        )

        for index, column in enumerate(
            ["foto_1", "foto_2", "foto_3"],
            start=1
        ):

            source = Path(event[column])

            # Nama dibuat berdasarkan event
            # supaya tidak terjadi bentrok nama file
            destination = (
                destination_folder
                / f"{event_id}_foto_{index}"
                f"{source.suffix.lower()}"
            )

            shutil.copy2(
                source,
                destination
            )

            copied_count += 1

    return copied_count


# ============================================================
# SALIN DATASET
# ============================================================

train_copied = copy_events(
    train_events,
    "train"
)

val_copied = copy_events(
    val_events,
    "val"
)

test_copied = copy_events(
    test_events,
    "test"
)


# ============================================================
# SIMPAN DAFTAR EVENT SETIAP SPLIT
# ============================================================

def save_split_csv(event_list, filename):

    output_file = OUTPUT_DIR / filename

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "event_id",
            "label",
            "foto_1",
            "foto_2",
            "foto_3",
            "timestamp"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for event in event_list:
            writer.writerow(event)


save_split_csv(
    train_events,
    "train_events.csv"
)

save_split_csv(
    val_events,
    "val_events.csv"
)

save_split_csv(
    test_events,
    "test_events.csv"
)


# ============================================================
# HASIL AKHIR
# ============================================================

print()
print("=" * 80)
print("DATASET BERHASIL DISIAPKAN")
print("=" * 80)

print()
print(f"Lokasi dataset:")
print(OUTPUT_DIR)

print()
print("Foto yang disalin:")
print(f"Train      : {train_copied}")
print(f"Validation : {val_copied}")
print(f"Test       : {test_copied}")

print()
print("Struktur:")
print()
print("dataset/")
print("├── train/")
print("│   ├── leopard_cat/")
print("│   └── null/")
print("├── val/")
print("│   ├── leopard_cat/")
print("│   └── null/")
print("└── test/")
print("    ├── leopard_cat/")
print("    └── null/")

print()
print("CATATAN:")
print("- 1 event tetap berada dalam 1 split.")
print("- Ketiga foto dalam event tetap bersama.")
print("- Foto asli tidak dipindahkan atau dihapus.")
print("- Dataset ini masih untuk klasifikasi leopard_cat vs null.")
print("- Belum ada proses training model.")

print()
print("=" * 80)
print("SELESAI")
print("=" * 80)