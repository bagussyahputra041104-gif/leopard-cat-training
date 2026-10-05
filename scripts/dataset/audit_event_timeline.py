import csv
from collections import defaultdict
from datetime import datetime
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
    print(f"Lokasi: {CSV_FILE}")
    raise SystemExit


# ============================================================
# BACA DATA EVENT
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
# KELOMPOKKAN EVENT BERDASARKAN FOLDER SUMBER
# ============================================================

folder_events = defaultdict(list)

for event in events:

    foto_1 = Path(event["foto_1"])

    try:
        relative_path = foto_1.relative_to(BASE_DIR)
        parts = relative_path.parts
    except ValueError:
        parts = foto_1.parts

    if len(parts) >= 2:
        source_folder = parts[-2]
    else:
        source_folder = "Tidak diketahui"

    folder_events[source_folder].append(event)


# ============================================================
# FUNGSI PARSING TIMESTAMP
# ============================================================

def parse_timestamp(timestamp_text):

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:

        try:
            return datetime.strptime(timestamp_text, fmt)

        except ValueError:
            continue

    return None


# ============================================================
# AUDIT TIMELINE
# ============================================================

print()
print("=" * 110)
print("AUDIT TIMELINE EVENT")
print("=" * 110)

print()
print(f"File CSV    : {CSV_FILE}")
print(f"Total event : {len(events)}")


print()
print("-" * 110)

print(
    f"{'Folder Sumber':45}"
    f"{'Event':>8}"
    f"{'Leopard':>10}"
    f"{'Null':>8}"
    f"{'Event Pertama':>22}"
    f"{'Event Terakhir':>22}"
)

print("-" * 110)


for folder in sorted(folder_events):

    folder_data = folder_events[folder]

    timestamps = []

    leopard_count = 0
    null_count = 0

    for event in folder_data:

        timestamp = parse_timestamp(event["timestamp"])

        if timestamp is not None:
            timestamps.append(timestamp)

        if event["label"] == "leopard_cat":
            leopard_count += 1

        elif event["label"] == "null":
            null_count += 1

    # --------------------------------------------------------
    # Tentukan waktu pertama dan terakhir
    # --------------------------------------------------------

    if timestamps:

        first_time = min(timestamps)
        last_time = max(timestamps)

        first_text = first_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        last_text = last_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    else:

        first_text = "Tidak tersedia"
        last_text = "Tidak tersedia"

    # --------------------------------------------------------
    # Tampilkan
    # --------------------------------------------------------

    display_folder = folder

    if len(display_folder) > 45:
        display_folder = display_folder[:42] + "..."

    print(
        f"{display_folder:45}"
        f"{len(folder_data):>8}"
        f"{leopard_count:>10}"
        f"{null_count:>8}"
        f"{first_text:>22}"
        f"{last_text:>22}"
    )


print("-" * 110)


# ============================================================
# RINGKASAN
# ============================================================

print()
print("=" * 110)
print("RINGKASAN TIMELINE")
print("=" * 110)

all_timestamps = []

for event in events:

    timestamp = parse_timestamp(event["timestamp"])

    if timestamp is not None:
        all_timestamps.append(timestamp)


if all_timestamps:

    earliest = min(all_timestamps)
    latest = max(all_timestamps)

    print()
    print(
        "Event paling awal :",
        earliest.strftime("%Y-%m-%d %H:%M:%S")
    )

    print(
        "Event paling akhir:",
        latest.strftime("%Y-%m-%d %H:%M:%S")
    )

    duration = latest - earliest

    print(
        "Rentang waktu     :",
        duration
    )

else:

    print()
    print("Timestamp tidak berhasil dibaca.")


# ============================================================
# CEK TIMESTAMP YANG GAGAL DIBACA
# ============================================================

invalid_timestamps = []

for event in events:

    timestamp_text = event["timestamp"]

    if parse_timestamp(timestamp_text) is None:

        invalid_timestamps.append(
            (
                event["event_id"],
                timestamp_text
            )
        )


print()
print("-" * 110)

print("CEK TIMESTAMP")

if invalid_timestamps:

    print(
        f"Ditemukan {len(invalid_timestamps)} timestamp "
        "yang tidak berhasil dibaca."
    )

    for event_id, timestamp_text in invalid_timestamps[:20]:

        print(
            f"  {event_id} -> {timestamp_text}"
        )

else:

    print("Semua timestamp berhasil dibaca.")


# ============================================================
# SELESAI
# ============================================================

print()
print("=" * 110)
print("AUDIT TIMELINE SELESAI.")
print("=" * 110)