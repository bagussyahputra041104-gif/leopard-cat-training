import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# KONFIGURASI PATH
# ============================================================

ROOT = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

OCCURRENCE_DIR = os.path.join(
    ROOT,
    "04_HASIL",
    "occurrence"
)

EVENT_DATASET = os.path.join(
    ROOT,
    "02_DATASET",
    "dataset",
    "event_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    OCCURRENCE_DIR,
    "visualization"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FILE INPUT
# ============================================================

HOURLY_FILE = os.path.join(
    OCCURRENCE_DIR,
    "leopard_cat_occurrence_hourly.csv"
)

PERIOD_FILE = os.path.join(
    OCCURRENCE_DIR,
    "leopard_cat_occurrence_period.csv"
)

DAILY_FILE = os.path.join(
    OCCURRENCE_DIR,
    "leopard_cat_occurrence_daily.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("VISUALISASI OCCURRENCE LEOPARD CAT")
print("=" * 60)


# ============================================================
# CEK FILE
# ============================================================

print("\nMengecek file...")

files_to_check = [
    ("Hourly", HOURLY_FILE),
    ("Period", PERIOD_FILE),
    ("Daily", DAILY_FILE),
    ("Event Dataset", EVENT_DATASET),
]

for name, path in files_to_check:

    if os.path.exists(path):
        print(f"[OK] {name}: {path}")

    else:
        print(f"[ERROR] {name}: {path}")
        raise SystemExit


# ============================================================
# MEMBACA DATA OCCURRENCE
# ============================================================

print("\nMembaca data occurrence...")

hourly = pd.read_csv(
    HOURLY_FILE,
    dtype=str,
    keep_default_na=False
)

period = pd.read_csv(
    PERIOD_FILE,
    dtype=str,
    keep_default_na=False
)

daily = pd.read_csv(
    DAILY_FILE,
    dtype=str,
    keep_default_na=False
)

print(f"Data hourly : {len(hourly)} baris")
print(f"Data period : {len(period)} baris")
print(f"Data daily  : {len(daily)} baris")


# ============================================================
# MEMBACA EVENT DATASET
# ============================================================

print("\nMembaca event dataset...")

events = pd.read_csv(
    EVENT_DATASET,
    dtype=str,
    keep_default_na=False
)

print(f"Total event : {len(events)}")


# ============================================================
# CEK KOLOM EVENT DATASET
# ============================================================

required_columns = [
    "event_id",
    "label",
    "foto_1",
    "foto_2",
    "foto_3",
    "timestamp"
]

for column in required_columns:

    if column not in events.columns:

        print("\nERROR:")
        print(f"Kolom '{column}' tidak ditemukan.")

        print("\nKolom yang tersedia:")

        for col in events.columns:
            print("-", col)

        raise SystemExit


# ============================================================
# MEMBUAT SOURCE FOLDER
# DARI PATH FOTO 1
# ============================================================

print("\nMembangun source folder dari foto_1...")


def get_source_folder(photo_path):

    if not photo_path:
        return "UNKNOWN"

    # Normalisasi slash
    photo_path = photo_path.replace("/", "\\")

    # Ambil folder tempat foto berada
    folder = os.path.basename(
        os.path.dirname(photo_path)
    )

    if not folder:
        return "UNKNOWN"

    return folder


events["source_folder"] = events["foto_1"].apply(
    get_source_folder
)


# ============================================================
# TAMPILKAN CONTOH
# ============================================================

print("\nContoh source folder:")

print(
    events[
        [
            "event_id",
            "label",
            "source_folder"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FILTER EVENT LEOPARD CAT
# ============================================================

leopard = events[
    events["label"].str.strip().str.lower()
    == "leopard_cat"
].copy()

print(
    f"\nEvent leopard cat : {len(leopard)}"
)


# ============================================================
# OCCURRENCE BERDASARKAN SOURCE FOLDER
# ============================================================

source = (
    leopard
    .groupby("source_folder")
    .size()
    .reset_index(name="event_count")
    .sort_values(
        "event_count",
        ascending=False
    )
)

print(
    "\nOccurrence berdasarkan source folder:"
)

print(
    source.to_string(index=False)
)


# ============================================================
# VISUALISASI 1
# OCCURRENCE BERDASARKAN JAM
# ============================================================

if len(hourly) > 0:

    hourly["hour"] = pd.to_numeric(
        hourly["hour"],
        errors="coerce"
    )

    hourly["event_count"] = pd.to_numeric(
        hourly["event_count"],
        errors="coerce"
    )

    hourly = hourly.dropna(
        subset=[
            "hour",
            "event_count"
        ]
    )

    hourly = hourly.sort_values("hour")

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        hourly["hour"],
        hourly["event_count"]
    )

    plt.xlabel("Jam")
    plt.ylabel("Jumlah Event")

    plt.title(
        "Occurrence Leopard Cat Berdasarkan Jam"
    )

    plt.xticks(
        range(24)
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "occurrence_hourly.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"\n[OK] Grafik hourly: {output}"
    )


# ============================================================
# VISUALISASI 2
# OCCURRENCE BERDASARKAN PERIODE
# ============================================================

if len(period) > 0:

    period["event_count"] = pd.to_numeric(
        period["event_count"],
        errors="coerce"
    )

    period = period.dropna(
        subset=["event_count"]
    )

    # Kolom asli CSV adalah:
    # time_period
    #
    # BUKAN:
    # period

    if "time_period" not in period.columns:

        print(
            "\nERROR:"
        )

        print(
            "Kolom 'time_period' tidak ditemukan "
            "pada file period."
        )

        print(
            "\nKolom yang tersedia:"
        )

        for col in period.columns:
            print("-", col)

        raise SystemExit

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        period["time_period"],
        period["event_count"]
    )

    plt.xlabel(
        "Periode Aktivitas"
    )

    plt.ylabel(
        "Jumlah Event"
    )

    plt.title(
        "Occurrence Leopard Cat Berdasarkan Periode Aktivitas"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "occurrence_period.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"[OK] Grafik period: {output}"
    )


# ============================================================
# VISUALISASI 3
# OCCURRENCE BERDASARKAN TANGGAL
# ============================================================

if len(daily) > 0:

    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="coerce"
    )

    daily["event_count"] = pd.to_numeric(
        daily["event_count"],
        errors="coerce"
    )

    daily = daily.dropna(
        subset=[
            "date",
            "event_count"
        ]
    )

    daily = daily.sort_values(
        "date"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        daily["date"],
        daily["event_count"],
        marker="o"
    )

    plt.xlabel(
        "Tanggal"
    )

    plt.ylabel(
        "Jumlah Event"
    )

    plt.title(
        "Occurrence Leopard Cat Berdasarkan Tanggal"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "occurrence_daily.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"[OK] Grafik daily: {output}"
    )


# ============================================================
# VISUALISASI 4
# OCCURRENCE BERDASARKAN SOURCE FOLDER
# ============================================================

if len(source) > 0:

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        source["source_folder"],
        source["event_count"]
    )

    plt.xlabel(
        "Source Folder"
    )

    plt.ylabel(
        "Jumlah Event Leopard Cat"
    )

    plt.title(
        "Occurrence Leopard Cat Berdasarkan Source Folder"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "occurrence_source.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"[OK] Grafik source: {output}"
    )


# ============================================================
# SIMPAN DATA SOURCE
# ============================================================

source_csv = os.path.join(
    OUTPUT_DIR,
    "occurrence_source.csv"
)

source.to_csv(
    source_csv,
    index=False
)

print(
    f"[OK] CSV source: {source_csv}"
)


# ============================================================
# SELESAI
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "VISUALISASI OCCURRENCE SELESAI"
)

print(
    "=" * 60
)

print(
    "\nSemua output disimpan di:"
)

print(
    OUTPUT_DIR
)