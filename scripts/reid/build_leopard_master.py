import os
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

EVENT_CSV = os.path.join(
    BASE_DIR,
    "event_dataset.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("BUILD LEOPARD CAT MASTER DATASET")
print("=" * 60)


# =========================================================
# LOAD EVENT DATASET
# =========================================================

print("\nLoading event dataset...")

df = pd.read_csv(
    EVENT_CSV,
    dtype=str,
    keep_default_na=False
)

print(f"Total event: {len(df)}")


# =========================================================
# FILTER LEOPARD CAT
# =========================================================

leopard_df = df[
    df["label"] == "leopard_cat"
].copy()

print(
    f"Total leopard_cat event: "
    f"{len(leopard_df)}"
)


# =========================================================
# FUNCTION:
# FIND COMMON SOURCE FOLDER
# =========================================================

def get_source_folder(image_path):

    """
    Mengambil nama folder induk dari path foto.

    Contoh:

    ...\A-West-250m-BC-Round 8\IMG001.jpg

    menjadi:

    A-West-250m-BC-Round 8
    """

    if not image_path:
        return ""

    image_path = os.path.normpath(
        image_path
    )

    parent_folder = os.path.basename(
        os.path.dirname(image_path)
    )

    return parent_folder


# =========================================================
# DETERMINE SOURCE FOLDER
# =========================================================

source_folders = []

source_consistency = []


for _, row in leopard_df.iterrows():

    foto_paths = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"]
    ]

    folders = [
        get_source_folder(path)
        for path in foto_paths
    ]

    unique_folders = set(folders)

    if len(unique_folders) == 1:

        source_folder = folders[0]

        consistency = "consistent"

    else:

        source_folder = "MULTIPLE_SOURCE"

        consistency = "check"


    source_folders.append(
        source_folder
    )

    source_consistency.append(
        consistency
    )


leopard_df["source_folder"] = (
    source_folders
)

leopard_df["source_consistency"] = (
    source_consistency
)


# =========================================================
# REORDER COLUMNS
# =========================================================

columns = [
    "event_id",
    "label",
    "source_folder",
    "source_consistency",
    "timestamp",
    "foto_1",
    "foto_2",
    "foto_3"
]

leopard_df = leopard_df[
    columns
]


# =========================================================
# SAVE
# =========================================================

leopard_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("HASIL")
print("=" * 60)

print(
    f"Total leopard_cat event : "
    f"{len(leopard_df)}"
)

print(
    "\nDistribusi berdasarkan source folder:"
)

source_counts = (
    leopard_df[
        "source_folder"
    ]
    .value_counts()
)

for source, count in source_counts.items():

    print(
        f"- {source}: "
        f"{count} event"
    )


# =========================================================
# CHECK CONSISTENCY
# =========================================================

inconsistent = leopard_df[
    leopard_df[
        "source_consistency"
    ] == "check"
]

print(
    "\nEvent dengan source foto "
    "tidak konsisten:",
    len(inconsistent)
)


if len(inconsistent) > 0:

    print("\nEvent yang perlu dicek:")

    print(
        inconsistent[
            [
                "event_id",
                "foto_1",
                "foto_2",
                "foto_3"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "Semua event memiliki "
        "source folder yang konsisten."
    )


# =========================================================
# PREVIEW FIRST ROWS
# =========================================================

print("\n" + "=" * 60)
print("CONTOH DATA")
print("=" * 60)

print(
    leopard_df[
        [
            "event_id",
            "source_folder",
            "timestamp"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print(
    f"\nMaster dataset disimpan di:"
    f"\n{OUTPUT_CSV}"
)