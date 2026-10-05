import csv
from collections import defaultdict
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "event_dataset.csv"


# ============================================================
# CEK FILE CSV
# ============================================================

if not CSV_FILE.exists():
    print("ERROR: event_dataset.csv tidak ditemukan.")
    print(f"Lokasi yang dicari:")
    print(CSV_FILE)
    raise SystemExit


# ============================================================
# BACA EVENT DATASET
# ============================================================

events = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    # Pastikan kolom yang dibutuhkan tersedia
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
        print(f"Kolom yang ditemukan : {reader.fieldnames}")
        print(f"Kolom yang hilang    : {sorted(missing_columns)}")
        raise SystemExit

    for row in reader:
        events.append(row)


# ============================================================
# DISTRIBUSI EVENT BERDASARKAN FOLDER SUMBER
# ============================================================

distribution = defaultdict(
    lambda: {
        "leopard_cat": 0,
        "null": 0,
        "total": 0
    }
)


for event in events:

    label = event["label"]

    # --------------------------------------------------------
    # Ambil foto pertama sebagai representasi lokasi event
    # --------------------------------------------------------

    photo_1 = Path(event["foto_1"])

    # --------------------------------------------------------
    # Ubah path menjadi relatif terhadap folder proyek
    # --------------------------------------------------------

    try:
        relative_path = photo_1.relative_to(BASE_DIR)
        parts = relative_path.parts

    except ValueError:
        parts = photo_1.parts

    # --------------------------------------------------------
    # Struktur data:
    #
    # BASE_DIR
    # ├── Rerta 1-leopard cat_baseline_BC
    # │   └── nama_folder
    # │       └── foto
    #
    # atau
    #
    # └── Rerta 1-tanpa satwa_baseline_BC
    #     └── nama_folder
    #         └── foto
    #
    # Folder tepat sebelum nama file = folder sumber/event asal
    # --------------------------------------------------------

    if len(parts) >= 2:
        source_folder = parts[-2]
    else:
        source_folder = "Tidak diketahui"

    # --------------------------------------------------------
    # Masukkan ke distribusi
    # --------------------------------------------------------

    if label not in distribution[source_folder]:
        print(
            f"PERINGATAN: Label tidak dikenal: "
            f"{label}"
        )
        continue

    distribution[source_folder][label] += 1
    distribution[source_folder]["total"] += 1


# ============================================================
# HITUNG TOTAL
# ============================================================

total_events = len(events)

total_leopard = sum(
    data["leopard_cat"]
    for data in distribution.values()
)

total_null = sum(
    data["null"]
    for data in distribution.values()
)


# ============================================================
# TAMPILKAN HEADER
# ============================================================

print()
print("=" * 85)
print("AUDIT DISTRIBUSI EVENT")
print("=" * 85)

print()
print(f"File CSV       : {CSV_FILE}")
print(f"Total event    : {total_events}")
print(f"Leopard cat    : {total_leopard}")
print(f"Null           : {total_null}")


# ============================================================
# TABEL DISTRIBUSI
# ============================================================

print()
print("-" * 85)

print(
    f"{'Folder Sumber':50}"
    f"{'Leopard':>10}"
    f"{'Null':>10}"
    f"{'Total':>10}"
)

print("-" * 85)


for folder, data in sorted(distribution.items()):

    display_folder = folder

    if len(display_folder) > 50:
        display_folder = display_folder[:47] + "..."

    print(
        f"{display_folder:50}"
        f"{data['leopard_cat']:>10}"
        f"{data['null']:>10}"
        f"{data['total']:>10}"
    )


# ============================================================
# TOTAL
# ============================================================

print("-" * 85)

print(
    f"{'TOTAL':50}"
    f"{total_leopard:>10}"
    f"{total_null:>10}"
    f"{total_events:>10}"
)

print("=" * 85)


# ============================================================
# CEK KOMPOSISI SETIAP FOLDER
# ============================================================

print()
print("CEK KOMPOSISI SETIAP FOLDER")
print("-" * 85)


for folder, data in sorted(distribution.items()):

    leopard = data["leopard_cat"]
    null = data["null"]
    total = data["total"]

    if leopard == 0 and null > 0:
        status = "HANYA NULL"

    elif null == 0 and leopard > 0:
        status = "HANYA LEOPARD CAT"

    elif leopard > 0 and null > 0:
        status = "LEOPARD + NULL"

    else:
        status = "KOSONG"

    print(
        f"{folder[:50]:50} -> {status}"
        f" ({total} event)"
    )


# ============================================================
# RINGKASAN
# ============================================================

folder_count = len(distribution)

leopard_only = sum(
    1
    for data in distribution.values()
    if data["leopard_cat"] > 0 and data["null"] == 0
)

null_only = sum(
    1
    for data in distribution.values()
    if data["null"] > 0 and data["leopard_cat"] == 0
)

mixed = sum(
    1
    for data in distribution.values()
    if data["leopard_cat"] > 0 and data["null"] > 0
)


print()
print("=" * 85)
print("RINGKASAN")
print("=" * 85)

print(f"Jumlah folder sumber : {folder_count}")
print(f"Hanya leopard cat    : {leopard_only}")
print(f"Hanya null            : {null_only}")
print(f"Campuran              : {mixed}")

print()
print("AUDIT DISTRIBUSI SELESAI.")
print("=" * 85)