import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

FINAL_RESULTS = BASE / "04_HASIL" / "individual" / "final_results" / "final_event_results.csv"
REID_FINAL = BASE / "individual_candidate_final.csv"

BACKUP = BASE / "04_HASIL" / "individual" / "final_results" / "final_event_results_before_reid_update.csv"


print("=" * 70)
print("SINKRONISASI FINAL RESULTS DENGAN RE-ID TERBARU")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

final_df = pd.read_csv(
    FINAL_RESULTS,
    dtype=str,
    keep_default_na=False
)

reid_df = pd.read_csv(
    REID_FINAL,
    dtype=str,
    keep_default_na=False
)


print("\nFinal Results :", len(final_df))
print("Final Re-ID   :", len(reid_df))


# =========================================================
# CEK EVENT
# =========================================================

if "event_id" not in final_df.columns:
    raise ValueError(
        "Kolom event_id tidak ditemukan di final_event_results.csv"
    )

if "event_id" not in reid_df.columns:
    raise ValueError(
        "Kolom event_id tidak ditemukan di individual_candidate_final.csv"
    )


# =========================================================
# BACKUP
# =========================================================

final_df.to_csv(
    BACKUP,
    index=False,
    encoding="utf-8-sig"
)

print("\nBackup dibuat:")
print(BACKUP)


# =========================================================
# KOLOM RE-ID
# =========================================================

required_reid = [
    "event_id",
    "candidate_individual"
]

for col in required_reid:

    if col not in reid_df.columns:

        raise ValueError(
            f"Kolom {col} tidak ditemukan "
            "di individual_candidate_final.csv"
        )


# =========================================================
# HANYA AMBIL INFORMASI RE-ID
# =========================================================

reid_update = reid_df[
    [
        "event_id",
        "candidate_individual"
    ]
].copy()


# =========================================================
# CEK DUPLIKAT EVENT DI RE-ID
# =========================================================

duplicate_reid = reid_update["event_id"].duplicated().sum()

if duplicate_reid > 0:

    raise ValueError(
        f"Terdapat {duplicate_reid} duplicate event "
        "di individual_candidate_final.csv"
    )


# =========================================================
# HAPUS KOLOM LAMA
# =========================================================

for col in [
    "candidate_individual",
    "reid_status"
]:

    if col in final_df.columns:

        final_df = final_df.drop(
            columns=[col]
        )


# =========================================================
# MERGE
# =========================================================

final_df = final_df.merge(
    reid_update,
    on="event_id",
    how="left",
    validate="one_to_one"
)


# =========================================================
# ISI STATUS RE-ID
# =========================================================

final_df["candidate_individual"] = (
    final_df["candidate_individual"]
    .replace("", "UNASSIGNED")
)

final_df["reid_status"] = (
    final_df["candidate_individual"]
    .apply(
        lambda x:
        "CANDIDATE"
        if x != "UNASSIGNED"
        else "UNASSIGNED"
    )
)


# =========================================================
# CEK EVENT YANG TIDAK TERUPDATE
# =========================================================

missing_reid = (
    final_df["candidate_individual"]
    .isna()
    .sum()
)

if missing_reid > 0:

    print(
        f"\nPERINGATAN: "
        f"{missing_reid} event tidak memiliki hasil Re-ID."
    )

    final_df["candidate_individual"] = (
        final_df["candidate_individual"]
        .fillna("UNASSIGNED")
    )

    final_df["reid_status"] = (
        final_df["candidate_individual"]
        .apply(
            lambda x:
            "CANDIDATE"
            if x != "UNASSIGNED"
            else "UNASSIGNED"
        )
    )


# =========================================================
# URUTKAN BERDASARKAN EVENT
# =========================================================

final_df = final_df.sort_values(
    "event_id"
).reset_index(drop=True)


# =========================================================
# SAVE
# =========================================================

final_df.to_csv(
    FINAL_RESULTS,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("HASIL SINKRONISASI")
print("=" * 70)

print(
    "\nTotal event :",
    len(final_df)
)

print(
    "\nStatus Re-ID:"
)

print(
    final_df["reid_status"]
    .value_counts()
)


print(
    "\nCandidate individual:"
)

print(
    final_df["candidate_individual"]
    .value_counts()
)


print("\nOutput:")
print(FINAL_RESULTS)

print("\nBackup:")
print(BACKUP)

print("\nSELESAI.")

