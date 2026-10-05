from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

EVENT_DATASET = (
    ROOT
    / "02_DATASET"
    / "dataset"
    / "event_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "04_HASIL"
    / "occurrence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. OUTPUT FILE
# ============================================================

EVENT_OUTPUT = (
    OUTPUT_DIR
    / "leopard_cat_occurrence_events.csv"
)

DAILY_OUTPUT = (
    OUTPUT_DIR
    / "leopard_cat_occurrence_daily.csv"
)

HOURLY_OUTPUT = (
    OUTPUT_DIR
    / "leopard_cat_occurrence_hourly.csv"
)

PERIOD_OUTPUT = (
    OUTPUT_DIR
    / "leopard_cat_occurrence_period.csv"
)

SOURCE_OUTPUT = (
    OUTPUT_DIR
    / "leopard_cat_occurrence_source.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "occurrence_summary.txt"
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 60)
print("ANALISIS OCCURRENCE LEOPARD CAT")
print("=" * 60)


# ============================================================
# 4. CEK FILE
# ============================================================

if not EVENT_DATASET.exists():

    print("\nERROR:")
    print("File event dataset tidak ditemukan:")
    print(EVENT_DATASET)

    raise SystemExit


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nMembaca event dataset...")

# dtype=str menjaga isi label tetap sebagai string
# dan tidak mengubah nilai "null" menjadi NaN.
df = pd.read_csv(
    EVENT_DATASET,
    dtype=str,
    keep_default_na=False
)

print(
    f"Total event dataset : {len(df)}"
)


# ============================================================
# 6. CEK KOLOM WAJIB
# ============================================================

required_columns = [
    "event_id",
    "label",
    "timestamp"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR:")
    print("Kolom yang tidak ditemukan:")

    for column in missing_columns:
        print(
            f"- {column}"
        )

    raise SystemExit


# ============================================================
# 7. NORMALISASI LABEL
# ============================================================

df["label_clean"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 8. HITUNG KELAS
# ============================================================

total_events = len(df)

leopard_mask = (
    df["label_clean"] == "leopard_cat"
)

leopard_events = int(
    leopard_mask.sum()
)

# Jumlah null dihitung dari total dikurangi leopard.
# Ini mengikuti hasil audit dataset:
# 135 total - 63 leopard_cat = 72 null.
null_events = (
    total_events
    - leopard_events
)


print(
    f"Event leopard cat   : {leopard_events}"
)

print(
    f"Event null          : {null_events}"
)


# ============================================================
# 9. KONVERSI TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamp = int(
    df["timestamp"].isna().sum()
)

print(
    f"Timestamp invalid   : {invalid_timestamp}"
)


# ============================================================
# 10. FILTER LEOPARD CAT
# ============================================================

leopard = df[
    leopard_mask
].copy()


if len(leopard) == 0:

    print("\nERROR:")
    print(
        "Tidak ditemukan event leopard_cat."
    )

    raise SystemExit


# ============================================================
# 11. FITUR TANGGAL DAN WAKTU
# ============================================================

leopard["date"] = (
    leopard["timestamp"]
    .dt.strftime("%Y-%m-%d")
)

leopard["year"] = (
    leopard["timestamp"]
    .dt.year
)

leopard["month"] = (
    leopard["timestamp"]
    .dt.month
)

leopard["month_name"] = (
    leopard["timestamp"]
    .dt.strftime("%B")
)

leopard["day"] = (
    leopard["timestamp"]
    .dt.day
)

leopard["hour"] = (
    leopard["timestamp"]
    .dt.hour
)

leopard["minute"] = (
    leopard["timestamp"]
    .dt.minute
)

leopard["time"] = (
    leopard["timestamp"]
    .dt.strftime("%H:%M:%S")
)


# ============================================================
# 12. KATEGORI PERIODE WAKTU
# ============================================================

def classify_time_period(hour):

    if 0 <= hour < 6:
        return "00:00-05:59"

    elif 6 <= hour < 12:
        return "06:00-11:59"

    elif 12 <= hour < 18:
        return "12:00-17:59"

    else:
        return "18:00-23:59"


leopard["time_period"] = (
    leopard["hour"]
    .apply(classify_time_period)
)


# ============================================================
# 13. SIMPAN DATA EVENT LEOPARD CAT
# ============================================================

event_columns = [
    "event_id",
    "label",
    "source_folder",
    "timestamp",
    "date",
    "time",
    "hour",
    "time_period"
]

event_columns = [
    column
    for column in event_columns
    if column in leopard.columns
]

leopard[
    event_columns
].sort_values(
    "timestamp"
).to_csv(
    EVENT_OUTPUT,
    index=False
)


# ============================================================
# 14. OCCURRENCE PER TANGGAL
# ============================================================

daily = (
    leopard
    .groupby("date")
    .size()
    .reset_index(
        name="event_count"
    )
)

daily = daily.sort_values(
    "date"
)

daily.to_csv(
    DAILY_OUTPUT,
    index=False
)


# ============================================================
# 15. OCCURRENCE PER JAM
# ============================================================

hourly = (
    leopard
    .groupby("hour")
    .size()
    .reset_index(
        name="event_count"
    )
)

hourly = hourly.sort_values(
    "hour"
)

hourly.to_csv(
    HOURLY_OUTPUT,
    index=False
)


# ============================================================
# 16. OCCURRENCE PER PERIODE
# ============================================================

period_order = [
    "00:00-05:59",
    "06:00-11:59",
    "12:00-17:59",
    "18:00-23:59"
]

period = (
    leopard
    .groupby("time_period")
    .size()
    .reindex(
        period_order,
        fill_value=0
    )
    .reset_index(
        name="event_count"
    )
)

period.to_csv(
    PERIOD_OUTPUT,
    index=False
)


# ============================================================
# 17. OCCURRENCE PER SOURCE FOLDER
# ============================================================

if "source_folder" in leopard.columns:

    source = (
        leopard
        .groupby("source_folder")
        .size()
        .reset_index(
            name="event_count"
        )
        .sort_values(
            "event_count",
            ascending=False
        )
    )

else:

    source = pd.DataFrame(
        columns=[
            "source_folder",
            "event_count"
        ]
    )


source.to_csv(
    SOURCE_OUTPUT,
    index=False
)


# ============================================================
# 18. RENTANG WAKTU
# ============================================================

first_detection = (
    leopard["timestamp"].min()
)

last_detection = (
    leopard["timestamp"].max()
)


# ============================================================
# 19. SUMMARY TXT
# ============================================================

summary = []

summary.append(
    "OCCURRENCE ANALYSIS - LEOPARD CAT"
)

summary.append(
    "=" * 60
)

summary.append("")

summary.append(
    "1. DATASET"
)

summary.append(
    "-" * 60
)

summary.append(
    f"Total event dataset : {total_events}"
)

summary.append(
    f"Leopard cat event   : {leopard_events}"
)

summary.append(
    f"Null event          : {null_events}"
)

summary.append(
    f"Timestamp invalid   : {invalid_timestamp}"
)

summary.append("")

summary.append(
    "2. RENTANG WAKTU"
)

summary.append(
    "-" * 60
)

summary.append(
    f"Deteksi pertama : {first_detection}"
)

summary.append(
    f"Deteksi terakhir: {last_detection}"
)

summary.append("")

summary.append(
    "3. OCCURRENCE BERDASARKAN PERIODE WAKTU"
)

summary.append(
    "-" * 60
)

for _, row in period.iterrows():

    summary.append(
        f"{row['time_period']} : "
        f"{int(row['event_count'])} event"
    )

summary.append("")

summary.append(
    "4. OCCURRENCE BERDASARKAN JAM"
)

summary.append(
    "-" * 60
)

for _, row in hourly.iterrows():

    hour_value = int(
        row["hour"]
    )

    event_count = int(
        row["event_count"]
    )

    summary.append(
        f"{hour_value:02d}:00 : "
        f"{event_count} event"
    )

summary.append("")

summary.append(
    "5. OCCURRENCE BERDASARKAN SOURCE FOLDER"
)

summary.append(
    "-" * 60
)

for _, row in source.iterrows():

    summary.append(
        f"{row['source_folder']} : "
        f"{int(row['event_count'])} event"
    )

summary.append("")

summary.append(
    "6. CATATAN"
)

summary.append(
    "-" * 60
)

summary.append(
    "Satu event merepresentasikan satu trigger "
    "kamera yang terdiri dari tiga foto."
)

summary.append(
    "Occurrence dihitung berdasarkan event "
    "dengan label leopard_cat."
)

summary.append(
    "Analisis menggunakan timestamp event "
    "pada event_dataset.csv."
)

summary.append(
    "Jumlah null merupakan total event dikurangi "
    "jumlah event leopard_cat."
)


# ============================================================
# 20. SIMPAN SUMMARY
# ============================================================

SUMMARY_OUTPUT.write_text(
    "\n".join(summary),
    encoding="utf-8"
)


# ============================================================
# 21. HASIL AKHIR
# ============================================================

print("\n" + "=" * 60)
print("OCCURRENCE ANALYSIS SELESAI")
print("=" * 60)

print(
    f"Total event dataset : {total_events}"
)

print(
    f"Leopard cat event   : {leopard_events}"
)

print(
    f"Null event          : {null_events}"
)

print(
    f"Timestamp invalid   : {invalid_timestamp}"
)

print(
    f"Deteksi pertama     : {first_detection}"
)

print(
    f"Deteksi terakhir    : {last_detection}"
)

print("\nFile yang dibuat:")

print(
    f"- {EVENT_OUTPUT}"
)

print(
    f"- {DAILY_OUTPUT}"
)

print(
    f"- {HOURLY_OUTPUT}"
)

print(
    f"- {PERIOD_OUTPUT}"
)

print(
    f"- {SOURCE_OUTPUT}"
)

print(
    f"- {SUMMARY_OUTPUT}"
)

print("\nSELESAI.")