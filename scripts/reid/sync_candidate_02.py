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
# FILE FINAL
# ============================================================

FINAL_FILE = (
    ROOT
    / "04_HASIL"
    / "individual"
    / "final_results"
    / "final_event_results.csv"
)


# ============================================================
# BACKUP
# ============================================================

BACKUP_FILE = (
    ROOT
    / "04_HASIL"
    / "individual"
    / "final_results"
    / "final_event_results_before_candidate_02.csv"
)


# ============================================================
# CEK FILE
# ============================================================

if not FINAL_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan:\n{FINAL_FILE}"
    )


# ============================================================
# BACA
# ============================================================

df = pd.read_csv(
    FINAL_FILE,
    keep_default_na=False
)


# ============================================================
# CEK KOLOM
# ============================================================

required_columns = [
    "event_id",
    "individual_candidate",
    "candidate_individual",
    "reid_status",
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
    FINAL_FILE,
    BACKUP_FILE
)

print("Backup dibuat:")
print(BACKUP_FILE)
print()


# ============================================================
# EVENT IND_CAND_02
# ============================================================

candidate_events = [
    "EVT_0026",
    "EVT_0037",
    "EVT_0055",
]


# ============================================================
# CEK STATUS SEBELUM UPDATE
# ============================================================

print("Status sebelum update:")
print()

print(
    df[
        df["event_id"].isin(candidate_events)
    ][
        [
            "event_id",
            "individual_candidate",
            "candidate_individual",
            "reid_status",
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# UPDATE
# ============================================================

mask = df["event_id"].isin(
    candidate_events
)


df.loc[
    mask,
    "individual_candidate"
] = "UNASSIGNED"


df.loc[
    mask,
    "candidate_individual"
] = "IND_CAND_02"


df.loc[
    mask,
    "reid_status"
] = "CANDIDATE"


# ============================================================
# SIMPAN
# ============================================================

df.to_csv(
    FINAL_FILE,
    index=False
)


# ============================================================
# CEK HASIL
# ============================================================

print("Status setelah update:")
print()

print(
    df[
        df["event_id"].isin(candidate_events)
    ][
        [
            "event_id",
            "individual_candidate",
            "candidate_individual",
            "reid_status",
        ]
    ].to_string(index=False)
)

print()

print("=" * 60)
print("SYNC IND_CAND_02 SELESAI")
print("=" * 60)
print()

print(
    f"Event IND_CAND_02 : {len(candidate_events)}"
)

print(
    "Event yang diubah : "
    + ", ".join(candidate_events)
)

print()

print(
    f"File final:\n{FINAL_FILE}"
)