from pathlib import Path
import pandas as pd
import shutil


# ============================================================
# ROOT PROJECT
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)


# ============================================================
# FILE LABEL
# ============================================================

LABEL_FILE = (
    ROOT
    / "05_REID"
    / "evaluation"
    / "reid_manual_labels.csv"
)


# ============================================================
# BACKUP
# ============================================================

BACKUP_FILE = (
    ROOT
    / "05_REID"
    / "evaluation"
    / "reid_manual_labels_before_candidate_02.csv"
)


# ============================================================
# CEK FILE
# ============================================================

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan:\n{LABEL_FILE}"
    )


# ============================================================
# BACA LABEL
# ============================================================

df = pd.read_csv(
    LABEL_FILE,
    keep_default_na=False
)


required_columns = [
    "rank",
    "event_1",
    "event_2",
    "similarity",
    "time_gap_hours",
    "time_gap_days",
    "manual_label",
    "notes",
]

for col in required_columns:

    if col not in df.columns:
        raise KeyError(
            f"Kolom '{col}' tidak ditemukan."
        )


# ============================================================
# BACKUP
# ============================================================

shutil.copy2(
    LABEL_FILE,
    BACKUP_FILE
)

print(f"Backup dibuat:")
print(BACKUP_FILE)
print()


# ============================================================
# PASANGAN BARU
# ============================================================

new_pairs = [
    {
        "event_1": "EVT_0026",
        "event_2": "EVT_0037",
        "manual_label": "SAME_CANDIDATE",
        "notes": (
            "Phase 3 group validation: "
            "pola tubuh terlihat konsisten "
            "antara EVT_0026 dan EVT_0037."
        ),
    },
    {
        "event_1": "EVT_0037",
        "event_2": "EVT_0055",
        "manual_label": "SAME_CANDIDATE",
        "notes": (
            "Phase 3 group validation: "
            "pola tubuh dan karakteristik visual "
            "terlihat konsisten antara EVT_0037 "
            "dan EVT_0055."
        ),
    },
]


# ============================================================
# FUNGSI CEK DUPLIKAT
# ============================================================

def pair_exists(event_1, event_2):

    forward = (
        (df["event_1"] == event_1)
        &
        (df["event_2"] == event_2)
    )

    reverse = (
        (df["event_1"] == event_2)
        &
        (df["event_2"] == event_1)
    )

    return (forward | reverse).any()


# ============================================================
# TAMBAHKAN LABEL
# ============================================================

added = 0


for item in new_pairs:

    event_1 = item["event_1"]
    event_2 = item["event_2"]


    if pair_exists(
        event_1,
        event_2
    ):

        print(
            f"[SKIP] "
            f"{event_1} <-> {event_2} "
            f"sudah ada."
        )

        continue


    # --------------------------------------------------------
    # Tambahkan baris kosong dengan struktur yang sama
    # --------------------------------------------------------

    new_row = {
        "rank": "",
        "event_1": event_1,
        "event_2": event_2,
        "similarity": "",
        "time_gap_hours": "",
        "time_gap_days": "",
        "manual_label": item["manual_label"],
        "notes": item["notes"],
    }


    df = pd.concat(
        [
            df,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )


    print(
        f"[ADD] "
        f"{event_1} <-> {event_2} "
        f"= {item['manual_label']}"
    )

    added += 1


# ============================================================
# SIMPAN
# ============================================================

df.to_csv(
    LABEL_FILE,
    index=False
)


# ============================================================
# RINGKASAN
# ============================================================

print()
print("=" * 60)
print("UPDATE SELESAI")
print("=" * 60)
print()
print(f"Label baru ditambahkan : {added}")
print(f"Total label sekarang    : {len(df)}")
print()
print(f"File:")
print(LABEL_FILE)