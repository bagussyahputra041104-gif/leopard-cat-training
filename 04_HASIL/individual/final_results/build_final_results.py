import os
import pandas as pd


# ============================================================
# PENGATURAN
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

EVENT_DATASET_CSV = os.path.join(
    BASE_DIR,
    "event_dataset.csv"
)

ORIENTATION_CSV = os.path.join(
    BASE_DIR,
    "orientation",
    "orientation_labels.csv"
)

INDIVIDUAL_CSV = os.path.join(
    BASE_DIR,
    "individual_count",
    "individual_event_assignment.csv"
)

REID_SUMMARY_CSV = os.path.join(
    BASE_DIR,
    "reid_evaluation",
    "reid_summary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_results"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "final_event_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "final_results_summary.txt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FUNGSI LOAD
# ============================================================

def load_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File tidak ditemukan:\n{path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


# ============================================================
# LOAD MASTER
# ============================================================

master = load_csv(
    MASTER_CSV
)

orientation = load_csv(
    ORIENTATION_CSV
)

individual = load_csv(
    INDIVIDUAL_CSV
)


# ============================================================
# INFORMASI AWAL
# ============================================================

print("=" * 70)
print("MEMBANGUN FINAL EVENT RESULTS")
print("=" * 70)

print()

print(
    f"Master event       : {len(master)}"
)

print(
    f"Orientation rows    : {len(orientation)}"
)

print(
    f"Individual rows     : {len(individual)}"
)

print()


# ============================================================
# SIAPKAN MASTER EVENT
# ============================================================

final = master.copy()


# ============================================================
# KLASIFIKASI
# ============================================================

# leopard_cat_master.csv hanya berisi leopard cat.
# Jadi seluruh event dalam file ini memang leopard cat.

final["classification"] = "leopard_cat"


# ============================================================
# ORIENTATION
# ============================================================

orientation = orientation[
    [
        "event_id",
        "foto",
        "arah_hadap"
    ]
].copy()


# Pivot:
#
# event_id | foto_1 | foto_2 | foto_3
#

orientation_pivot = (
    orientation
    .pivot(
        index="event_id",
        columns="foto",
        values="arah_hadap"
    )
    .reset_index()
)


# Pastikan semua kolom tersedia

for col in [
    "foto_1",
    "foto_2",
    "foto_3"
]:

    if col not in orientation_pivot.columns:

        orientation_pivot[col] = ""


orientation_pivot = orientation_pivot[
    [
        "event_id",
        "foto_1",
        "foto_2",
        "foto_3"
    ]
]


orientation_pivot = orientation_pivot.rename(
    columns={
        "foto_1": "orientation_foto_1",
        "foto_2": "orientation_foto_2",
        "foto_3": "orientation_foto_3"
    }
)


# Gabungkan

final = final.merge(
    orientation_pivot,
    on="event_id",
    how="left"
)


# ============================================================
# INDIVIDUAL CANDIDATE
# ============================================================

# Cek kolom individual

print(
    "Kolom individual_event_assignment.csv:"
)

for col in individual.columns:

    print(
        f" - {col}"
    )

print()


# Cari kolom ID individu

candidate_column = None

for candidate in [
    "individual_candidate",
    "candidate_individual",
    "individual_id",
    "candidate_group",
    "individual_group"
]:

    if candidate in individual.columns:

        candidate_column = candidate

        break


if candidate_column is None:

    raise ValueError(
        "Kolom kandidat individu tidak ditemukan."
    )


# Cari event ID

if "event_id" not in individual.columns:

    raise ValueError(
        "Kolom event_id tidak ditemukan "
        "di individual_event_assignment.csv."
    )


individual_small = individual[
    [
        "event_id",
        candidate_column
    ]
].copy()


individual_small = individual_small.rename(
    columns={
        candidate_column:
        "individual_candidate"
    }
)


# Bersihkan

individual_small[
    "individual_candidate"
] = (
    individual_small[
        "individual_candidate"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


individual_small.loc[
    individual_small[
        "individual_candidate"
    ] == "",
    "individual_candidate"
] = "UNASSIGNED"


# Gabungkan

final = final.merge(
    individual_small,
    on="event_id",
    how="left"
)


final[
    "individual_candidate"
] = (
    final[
        "individual_candidate"
    ]
    .fillna("UNASSIGNED")
)


# ============================================================
# RE-ID STATUS
# ============================================================

def determine_reid_status(value):

    if (
        value == ""
        or
        value == "UNASSIGNED"
    ):

        return "UNASSIGNED"

    return "CANDIDATE"


final[
    "reid_status"
] = final[
    "individual_candidate"
].apply(
    determine_reid_status
)


# ============================================================
# MOVEMENT
# ============================================================

# Movement otomatis kita simpan sebagai
# informasi tambahan, bukan hasil utama.

MOVEMENT_CSV = os.path.join(
    BASE_DIR,
    "movement",
    "auto_movement_candidates.csv"
)


if os.path.exists(MOVEMENT_CSV):

    movement = pd.read_csv(
        MOVEMENT_CSV,
        dtype=str,
        keep_default_na=False
    )

    movement_columns = [
        "event_id",
        "movement_candidate"
    ]

    available = [
        col
        for col in movement_columns
        if col in movement.columns
    ]

    if len(available) == 2:

        movement_small = movement[
            available
        ].copy()

        movement_small = movement_small.rename(
            columns={
                "movement_candidate":
                "movement_candidate"
            }
        )

        final = final.merge(
            movement_small,
            on="event_id",
            how="left"
        )

    else:

        final[
            "movement_candidate"
        ] = "tidak_dianalisis"

else:

    final[
        "movement_candidate"
    ] = "tidak_dianalisis"


final[
    "movement_candidate"
] = (
    final[
        "movement_candidate"
    ]
    .fillna("tidak_dianalisis")
)


# ============================================================
# SUSUN KOLOM
# ============================================================

preferred_columns = [

    "event_id",

    "label",

    "classification",

    "source_folder",

    "source_consistency",

    "timestamp",

    "foto_1",

    "foto_2",

    "foto_3",

    "orientation_foto_1",

    "orientation_foto_2",

    "orientation_foto_3",

    "movement_candidate",

    "individual_candidate",

    "reid_status"
]


existing_columns = [
    col
    for col in preferred_columns
    if col in final.columns
]


other_columns = [
    col
    for col in final.columns
    if col not in existing_columns
]


final = final[
    existing_columns
    +
    other_columns
]


# ============================================================
# SIMPAN FINAL CSV
# ============================================================

final.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# RINGKASAN
# ============================================================

total_event = len(final)

jumlah_leopard = (
    final[
        "classification"
    ]
    == "leopard_cat"
).sum()


jumlah_candidate = (
    final[
        "reid_status"
    ]
    == "CANDIDATE"
).sum()


jumlah_unassigned = (
    final[
        "reid_status"
    ]
    == "UNASSIGNED"
).sum()


jumlah_movement_tidak_yakin = (
    final[
        "movement_candidate"
    ]
    == "tidak_yakin"
).sum()


jumlah_movement_kanan = (
    final[
        "movement_candidate"
    ]
    == "kanan"
).sum()


jumlah_movement_kiri = (
    final[
        "movement_candidate"
    ]
    == "kiri"
).sum()


jumlah_movement_mendekat = (
    final[
        "movement_candidate"
    ]
    == "mendekat"
).sum()


jumlah_movement_menjauh = (
    final[
        "movement_candidate"
    ]
    == "menjauh"
).sum()


# ============================================================
# SUMMARY TXT
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL EVENT RESULTS\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )

    f.write(
        f"Total event leopard cat : "
        f"{total_event}\n"
    )

    f.write(
        f"Event classification    : "
        f"{jumlah_leopard}\n"
    )

    f.write(
        f"Event candidate Re-ID   : "
        f"{jumlah_candidate}\n"
    )

    f.write(
        f"Event unassigned        : "
        f"{jumlah_unassigned}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "MOVEMENT CANDIDATE\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        f"Tidak yakin : "
        f"{jumlah_movement_tidak_yakin}\n"
    )

    f.write(
        f"Kanan       : "
        f"{jumlah_movement_kanan}\n"
    )

    f.write(
        f"Kiri        : "
        f"{jumlah_movement_kiri}\n"
    )

    f.write(
        f"Mendekat    : "
        f"{jumlah_movement_mendekat}\n"
    )

    f.write(
        f"Menjauh     : "
        f"{jumlah_movement_menjauh}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "CATATAN:\n"
    )

    f.write(
        "Candidate Re-ID bukan identitas individu "
        "yang telah terverifikasi.\n"
    )

    f.write(
        "UNASSIGNED bukan berarti individu berbeda.\n"
    )

    f.write(
        "Movement merupakan hasil eksperimen "
        "dan bukan ground truth.\n"
    )


# ============================================================
# OUTPUT
# ============================================================

print("=" * 70)

print(
    "FINAL RESULTS BERHASIL DIBUAT"
)

print("=" * 70)

print()

print(
    f"Total event             : {total_event}"
)

print(
    f"Leopard cat             : {jumlah_leopard}"
)

print(
    f"Candidate Re-ID         : {jumlah_candidate}"
)

print(
    f"UNASSIGNED              : {jumlah_unassigned}"
)

print()

print(
    "Movement candidate:"
)

print(
    f"  tidak_yakin           : "
    f"{jumlah_movement_tidak_yakin}"
)

print(
    f"  kanan                 : "
    f"{jumlah_movement_kanan}"
)

print(
    f"  kiri                  : "
    f"{jumlah_movement_kiri}"
)

print(
    f"  mendekat              : "
    f"{jumlah_movement_mendekat}"
)

print(
    f"  menjauh               : "
    f"{jumlah_movement_menjauh}"
)

print()

print(
    "Output CSV:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "Output summary:"
)

print(
    OUTPUT_SUMMARY
)

print()

print("=" * 70)
print("SELESAI")
print("=" * 70)