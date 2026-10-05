import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

EVENT_CSV = BASE / "event_dataset.csv"
MASTER_CSV = BASE / "leopard_cat_master.csv"
FINAL_CSV = (
    BASE
    / "04_HASIL" / "individual" / "final_results"
    / "final_event_results.csv"
)

INDIVIDUAL_CSV = (
    BASE
    / "individual_candidate_final.csv"
)

OUTPUT_DIR = BASE / "06_RINGKASAN" / "project_summary"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_TXT = (
    OUTPUT_DIR
    / "project_status_summary.txt"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "project_status_summary.csv"
)


# ============================================================
# HELPER
# ============================================================

def read_csv(path):

    if not path.exists():
        print(f"WARNING: file tidak ditemukan: {path}")
        return None

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def get_count(df, column, value):

    if df is None:
        return None

    if column not in df.columns:
        return None

    return int(
        (df[column] == value).sum()
    )


# ============================================================
# LOAD
# ============================================================

event_df = read_csv(EVENT_CSV)
master_df = read_csv(MASTER_CSV)
final_df = read_csv(FINAL_CSV)
individual_df = read_csv(INDIVIDUAL_CSV)


# ============================================================
# SUMMARY VALUES
# ============================================================

summary = []


def add(category, item, value, note=""):

    summary.append({
        "category": category,
        "item": item,
        "value": value,
        "note": note
    })


# ============================================================
# DATASET / EVENT
# ============================================================

if event_df is not None:

    add(
        "Dataset",
        "Total event",
        len(event_df),
        "event_dataset.csv"
    )

    if "label" in event_df.columns:

        add(
            "Dataset",
            "Leopard cat event",
            get_count(
                event_df,
                "label",
                "leopard_cat"
            )
        )

        add(
            "Dataset",
            "Null event",
            get_count(
                event_df,
                "label",
                "null"
            )
        )


if master_df is not None:

    add(
        "Leopard Cat",
        "Total leopard-cat event",
        len(master_df),
        "leopard_cat_master.csv"
    )

    if "source_folder" in master_df.columns:

        add(
            "Leopard Cat",
            "Jumlah source folder",
            master_df[
                "source_folder"
            ].nunique()
        )

    if "timestamp" in master_df.columns:

        timestamps = pd.to_datetime(
            master_df["timestamp"],
            errors="coerce"
        )

        add(
            "Leopard Cat",
            "Timestamp awal",
            str(timestamps.min())
        )

        add(
            "Leopard Cat",
            "Timestamp akhir",
            str(timestamps.max())
        )


# ============================================================
# FINAL RESULTS
# ============================================================

if final_df is not None:

    add(
        "Final Results",
        "Total event",
        len(final_df)
    )

    if "classification" in final_df.columns:

        add(
            "Classification",
            "Leopard cat",
            get_count(
                final_df,
                "classification",
                "leopard_cat"
            )
        )

        add(
            "Classification",
            "Null",
            get_count(
                final_df,
                "classification",
                "null"
            )
        )

    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    orientation_cols = [
        "orientation_foto_1",
        "orientation_foto_2",
        "orientation_foto_3"
    ]

    orientation_available = [
        col
        for col in orientation_cols
        if col in final_df.columns
    ]

    add(
        "Orientation",
        "Kolom tersedia",
        len(orientation_available),
        ", ".join(orientation_available)
    )

    if orientation_available:

        orientation_values = []

        for col in orientation_available:

            orientation_values.extend(
                final_df[col]
                .astype(str)
                .tolist()
            )

        orientation_series = pd.Series(
            orientation_values
        )

        orientation_series = (
            orientation_series[
                orientation_series
                .str.strip()
                .ne("")
            ]
        )

        for value, count in (
            orientation_series
            .value_counts()
            .items()
        ):

            add(
                "Orientation",
                str(value),
                int(count)
            )

    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    if "movement_candidate" in final_df.columns:

        movement = (
            final_df[
                "movement_candidate"
            ]
            .astype(str)
            .str.strip()
        )

        add(
            "Movement",
            "Kolom movement tersedia",
            "YES"
        )

        for value, count in (
            movement.value_counts()
            .items()
        ):

            add(
                "Movement",
                str(value),
                int(count)
            )

    # --------------------------------------------------------
    # RE-ID
    # --------------------------------------------------------

    if "reid_status" in final_df.columns:

        for value, count in (
            final_df[
                "reid_status"
            ]
            .value_counts()
            .items()
        ):

            add(
                "Re-ID",
                f"Status: {value}",
                int(count)
            )

    if "candidate_individual" in final_df.columns:

        for value, count in (
            final_df[
                "candidate_individual"
            ]
            .value_counts()
            .items()
        ):

            add(
                "Re-ID",
                f"Candidate: {value}",
                int(count)
            )


# ============================================================
# INDIVIDUAL CANDIDATE
# ============================================================

if individual_df is not None:

    add(
        "Re-ID",
        "Total candidate group",
        (
            individual_df[
                "candidate_individual"
            ]
            .replace(
                "UNASSIGNED",
                pd.NA
            )
            .dropna()
            .nunique()
            if "candidate_individual"
            in individual_df.columns
            else "N/A"
        )
    )

    if "candidate_individual" in individual_df.columns:

        grouped = (
            individual_df[
                individual_df[
                    "candidate_individual"
                ] != "UNASSIGNED"
            ]
            .groupby(
                "candidate_individual"
            )
            .size()
        )

        for group, count in grouped.items():

            add(
                "Re-ID",
                f"{group} event",
                int(count)
            )


# ============================================================
# SAVE CSV
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# BUILD TXT
# ============================================================

lines = []

lines.append("=" * 70)
lines.append("PROJECT STATUS SUMMARY")
lines.append("=" * 70)
lines.append("")

lines.append(
    "Project: Camera Trap Leopard Cat Computer Vision"
)

lines.append(
    "Status: Checkpoint setelah Re-ID Group 01"
)

lines.append("")


current_category = None

for row in summary:

    category = row["category"]

    if category != current_category:

        lines.append("")
        lines.append("-" * 70)
        lines.append(category.upper())
        lines.append("-" * 70)

        current_category = category

    line = (
        f"{row['item']}: "
        f"{row['value']}"
    )

    if row["note"]:

        line += (
            f" "
            f"({row['note']})"
        )

    lines.append(line)


# ============================================================
# INTERPRETATION / NOTES
# ============================================================

lines.extend([
    "",
    "=" * 70,
    "CATATAN METODOLOGI",
    "=" * 70,
    "",
    "1. Unit analisis utama adalah event camera trap,",
    "   bukan foto individual.",
    "",
    "2. Setiap event terdiri dari 3 foto.",
    "",
    "3. Classification membedakan leopard_cat dan null.",
    "",
    "4. Re-ID menggunakan candidate individual group,",
    "   bukan ground-truth identitas individu.",
    "",
    "5. IND_CAND_01 merupakan kelompok kandidat individu",
    "   berdasarkan similarity dan review visual manual.",
    "",
    "6. UNASSIGNED tidak berarti individu berbeda.",
    "   Artinya belum ada bukti yang cukup untuk menghubungkan",
    "   event tersebut dengan candidate group.",
    "",
    "7. Jumlah candidate group bukan estimasi populasi final.",
    "",
    "8. Hasil classification yang tinggi pada test set",
    "   tidak otomatis berarti generalisasi sempurna.",
    "",
    "9. Movement merupakan movement candidate/eksperimental",
    "   dan perlu interpretasi hati-hati.",
    "",
])


lines.append("=" * 70)
lines.append("SELESAI")
lines.append("=" * 70)


with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(lines)
    )


# ============================================================
# PRINT
# ============================================================

print("=" * 70)
print("PROJECT SUMMARY SELESAI")
print("=" * 70)

print("\nOutput TXT:")
print(OUTPUT_TXT)

print("\nOutput CSV:")
print(OUTPUT_CSV)

print("\nJumlah summary item:")
print(len(summary))

print("\n" + "=" * 70)