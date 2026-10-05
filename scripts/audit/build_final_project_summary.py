import os
import pandas as pd


# ============================================================
# ROOT PROJECT
# ============================================================

ROOT = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "06_RINGKASAN",
    "final_project_summary"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILE INPUT
# ============================================================

EVENT_DATASET = os.path.join(
    ROOT,
    "02_DATASET",
    "dataset",
    "event_dataset.csv"
)

TEST_EVENTS = os.path.join(
    ROOT,
    "02_DATASET",
    "dataset",
    "test_events.csv"
)

RESNET_REPORT = os.path.join(
    ROOT,
    "04_HASIL",
    "classification",
    "evaluation_baseline",
    "classification_report.txt"
)

MOBILENET_REPORT = os.path.join(
    ROOT,
    "04_HASIL",
    "classification",
    "evaluation_model2",
    "classification_report.txt"
)

FINAL_REID = os.path.join(
    ROOT,
    "04_HASIL",
    "individual",
    "final_results",
    "final_event_results.csv"
)

OCCURRENCE_DAILY = os.path.join(
    ROOT,
    "04_HASIL",
    "occurrence",
    "leopard_cat_occurrence_daily.csv"
)

OCCURRENCE_PERIOD = os.path.join(
    ROOT,
    "04_HASIL",
    "occurrence",
    "leopard_cat_occurrence_period.csv"
)

OCCURRENCE_HOURLY = os.path.join(
    ROOT,
    "04_HASIL",
    "occurrence",
    "leopard_cat_occurrence_hourly.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)


# ============================================================
# DATASET
# ============================================================

print("\nMembaca event dataset...")

events = pd.read_csv(
    EVENT_DATASET,
    dtype=str,
    keep_default_na=False
)

events["label"] = (
    events["label"]
    .str.strip()
    .str.lower()
)

total_events = len(events)

leopard_events = len(
    events[
        events["label"] == "leopard_cat"
    ]
)

null_events = len(
    events[
        events["label"] == "null"
    ]
)

print(f"Total event    : {total_events}")
print(f"Leopard cat    : {leopard_events}")
print(f"Null           : {null_events}")


# ============================================================
# TIMESTAMP
# ============================================================

events["timestamp"] = pd.to_datetime(
    events["timestamp"],
    errors="coerce"
)

valid_timestamps = events["timestamp"].notna().sum()

if valid_timestamps > 0:

    first_detection = events["timestamp"].min()
    last_detection = events["timestamp"].max()

else:

    first_detection = None
    last_detection = None


# ============================================================
# TEST EVENT
# ============================================================

test_event_count = 0

if os.path.exists(TEST_EVENTS):

    test_df = pd.read_csv(
        TEST_EVENTS,
        dtype=str,
        keep_default_na=False
    )

    test_event_count = len(test_df)


# ============================================================
# RE-ID
# ============================================================

reid_candidates = 0
reid_unassigned = 0
candidate_groups = {}

if os.path.exists(FINAL_REID):

    reid = pd.read_csv(
        FINAL_REID,
        dtype=str,
        keep_default_na=False
    )

    if "reid_status" in reid.columns:

        reid_candidates = (
            reid["reid_status"]
            .str.upper()
            .eq("CANDIDATE")
            .sum()
        )

        reid_unassigned = (
            reid["reid_status"]
            .str.upper()
            .eq("UNASSIGNED")
            .sum()
        )

    if "candidate_individual" in reid.columns:

        grouped = (
            reid[
                reid["candidate_individual"]
                .str.upper()
                != "UNASSIGNED"
            ]
            .groupby("candidate_individual")
            .size()
        )

        candidate_groups = grouped.to_dict()


# ============================================================
# OCCURRENCE
# ============================================================

period_data = pd.DataFrame()

if os.path.exists(OCCURRENCE_PERIOD):

    period_data = pd.read_csv(
        OCCURRENCE_PERIOD,
        dtype=str,
        keep_default_na=False
    )


hourly_data = pd.DataFrame()

if os.path.exists(OCCURRENCE_HOURLY):

    hourly_data = pd.read_csv(
        OCCURRENCE_HOURLY,
        dtype=str,
        keep_default_na=False
    )


daily_data = pd.DataFrame()

if os.path.exists(OCCURRENCE_DAILY):

    daily_data = pd.read_csv(
        OCCURRENCE_DAILY,
        dtype=str,
        keep_default_na=False
    )


# ============================================================
# HASIL PERIOD
# ============================================================

period_lines = []

if len(period_data) > 0:

    for _, row in period_data.iterrows():

        period_lines.append(
            f"- {row['time_period']}: "
            f"{row['event_count']} event"
        )


# ============================================================
# HASIL GROUP RE-ID
# ============================================================

reid_group_lines = []

if candidate_groups:

    for group, count in sorted(
        candidate_groups.items()
    ):

        reid_group_lines.append(
            f"- {group}: {count} event"
        )

else:

    reid_group_lines.append(
        "- Belum ada candidate group."
    )


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_text = f"""
RINGKASAN FINAL PROYEK COMPUTER VISION
CAMERA TRAP - LEOPARD CAT
============================================================

1. DATASET DAN EVENT
------------------------------------------------------------

Total event              : {total_events}
Event leopard cat        : {leopard_events}
Event null               : {null_events}

Setiap event terdiri dari 3 foto yang berasal dari satu
trigger kamera trap.

Rentang waktu pengamatan:
Mulai                     : {first_detection}
Selesai                   : {last_detection}

Timestamp valid            : {valid_timestamps}/{total_events}


2. KLASIFIKASI
------------------------------------------------------------

Model 1: ResNet18

- Test foto             : 63
- Prediksi benar        : 63
- Akurasi test          : 100%
- Test event            : 21
- Event benar           : 21
- Akurasi event         : 100%
- Best validation       : 90%


Model 2: MobileNetV3-Small

- Test foto             : 63
- Prediksi benar        : 63
- Akurasi test          : 100%
- Test event            : 21
- Event benar           : 21
- Akurasi event         : 100%
- Best validation       : 90%


Interpretasi:

Kedua model memperoleh akurasi 100% pada test set yang
digunakan dalam eksperimen.

Hasil tersebut tidak diartikan sebagai jaminan bahwa model
akan memperoleh akurasi 100% pada data baru.

Terdapat indikasi overfitting karena akurasi training mencapai
100%, sedangkan validation accuracy berada pada sekitar 90%.


3. RE-IDENTIFICATION / INDIVIDUAL CANDIDATE
------------------------------------------------------------

Event leopard cat             : {leopard_events}
Event candidate individual    : {reid_candidates}
Event belum ter-assignment    : {reid_unassigned}

Candidate group:

{chr(10).join(reid_group_lines)}

Status candidate individual merupakan hasil pengelompokan
berdasarkan kemiripan visual dan belum dianggap sebagai
identitas individu yang tervalidasi.


4. OCCURRENCE
------------------------------------------------------------

Jumlah event leopard cat yang dianalisis: {leopard_events}

Occurrence dianalisis berdasarkan:

- Jam
- Periode aktivitas
- Tanggal
- Source folder


Occurrence berdasarkan periode:

{chr(10).join(period_lines)}


5. OUTPUT VISUALISASI
------------------------------------------------------------

Visualisasi occurrence tersedia pada:

04_HASIL\\occurrence\\visualization\\

Output:

- occurrence_hourly.png
- occurrence_period.png
- occurrence_daily.png
- occurrence_source.png
- occurrence_source.csv


6. KETERBATASAN
------------------------------------------------------------

1. Dataset relatif terbatas.

2. Kondisi foto bervariasi, termasuk foto siang/malam,
   jarak berbeda, orientasi berbeda, dan kualitas gambar
   yang tidak seragam.

3. Kedua model menunjukkan indikasi overfitting.

4. Pembagian data berbasis event masih memungkinkan adanya
   kemiripan lingkungan atau karakteristik kamera antar
   train, validation, dan test.

5. Re-identification masih berupa candidate grouping dan
   belum merupakan identifikasi individu yang tervalidasi.

6. Hasil candidate individual belum dapat digunakan sebagai
   jumlah populasi sebenarnya.

7. YOLO pretrained digunakan sebagai bantuan eksplorasi
   crop/deteksi dan belum dianggap sebagai ground truth
   detector.


7. ALUR SISTEM
------------------------------------------------------------

Camera Trap
    |
    v
3 Foto / Event
    |
    v
Event Dataset
    |
    v
Classification
    |
    +---- Leopard Cat
    |
    +---- Null
    |
    v
Re-Identification
    |
    v
Candidate Individual
    |
    v
Occurrence Analysis
    |
    v
Visualization
    |
    v
Flutter Dashboard


8. STATUS PROYEK
------------------------------------------------------------

Dataset/event preparation       : SELESAI
Classification                  : SELESAI
Event-level evaluation          : SELESAI
Re-identification baseline     : SELESAI
Occurrence analysis             : SELESAI
Occurrence visualization        : SELESAI
Dashboard Flutter               : BELUM DIMULAI


9. CATATAN PENGEMBANGAN
------------------------------------------------------------

Dashboard Flutter direncanakan sebagai tahap berikutnya.

Dashboard dapat menampilkan:

- total event
- total leopard cat
- total null
- hasil klasifikasi
- candidate individual
- occurrence berdasarkan jam
- occurrence berdasarkan periode
- occurrence berdasarkan tanggal
- source/camera
- visualisasi hasil
- detail event dan tiga foto


============================================================
KESIMPULAN STATUS
============================================================

Analisis inti proyek camera trap leopard cat telah selesai
dan hasil utama telah tersedia untuk digunakan sebagai dasar
laporan serta pengembangan dashboard.

============================================================
"""


# ============================================================
# SIMPAN TXT
# ============================================================

TXT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_project_summary.txt"
)

with open(
    TXT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary_text.strip()
    )


# ============================================================
# SIMPAN CSV RINGKASAN
# ============================================================

summary_rows = [

    ["total_event", total_events],
    ["leopard_cat_event", leopard_events],
    ["null_event", null_events],
    ["valid_timestamp", valid_timestamps],
    ["test_event", test_event_count],
    ["reid_candidate_event", reid_candidates],
    ["reid_unassigned_event", reid_unassigned],
    ["first_detection", first_detection],
    ["last_detection", last_detection],
]

summary_csv = pd.DataFrame(
    summary_rows,
    columns=[
        "metric",
        "value"
    ]
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "final_project_summary.csv"
)

summary_csv.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY SELESAI")
print("=" * 70)

print("\nOutput:")

print(
    f"- {TXT_OUTPUT}"
)

print(
    f"- {CSV_OUTPUT}"
)

print("\nStatus:")
print("Dataset/event       : SELESAI")
print("Classification      : SELESAI")
print("Re-ID               : SELESAI")
print("Occurrence          : SELESAI")
print("Visualization       : SELESAI")
print("Dashboard Flutter   : BELUM DIMULAI")