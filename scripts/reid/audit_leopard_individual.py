import os
import shutil
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

EVENT_CSV = os.path.join(
    BASE_DIR,
    "event_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "preview_leopard_individual"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("AUDIT LEOPARD CAT UNTUK INDIVIDUAL IDENTIFICATION")
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
# CEK KOLOM
# =========================================================

print("\nKolom dataset:")
print(df.columns.tolist())


# =========================================================
# FILTER LEOPARD CAT
# =========================================================

leopard_df = df[
    df["label"] == "leopard_cat"
].copy()

print(
    f"\nTotal event leopard_cat: "
    f"{len(leopard_df)}"
)


# =========================================================
# INFORMASI DASAR
# =========================================================

print("\n" + "=" * 60)
print("INFORMASI LEOPARD CAT")
print("=" * 60)

print(
    f"Jumlah event leopard_cat : "
    f"{len(leopard_df)}"
)

print(
    f"Jumlah foto leopard_cat  : "
    f"{len(leopard_df) * 3}"
)


# =========================================================
# RENTANG WAKTU
# =========================================================

leopard_df["timestamp"] = pd.to_datetime(
    leopard_df["timestamp"],
    errors="coerce"
)

print(
    f"\nKemunculan pertama : "
    f"{leopard_df['timestamp'].min()}"
)

print(
    f"Kemunculan terakhir : "
    f"{leopard_df['timestamp'].max()}"
)


# =========================================================
# SORT BERDASARKAN WAKTU
# =========================================================

leopard_df = leopard_df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# =========================================================
# SELISIH ANTAR EVENT
# =========================================================

leopard_df["selisih_dengan_event_sebelumnya"] = (
    leopard_df["timestamp"].diff()
)


# =========================================================
# TAMPILKAN DAFTAR EVENT
# =========================================================

print("\n" + "=" * 60)
print("DAFTAR EVENT LEOPARD CAT")
print("=" * 60)

for idx, row in leopard_df.iterrows():

    print(
        f"{idx + 1:02d}. "
        f"{row['event_id']} | "
        f"{row['timestamp']} | "
        f"{row['selisih_dengan_event_sebelumnya']}"
    )


# =========================================================
# ANALISIS SOURCE / FOLDER
# =========================================================

print("\n" + "=" * 60)
print("SUMBER EVENT LEOPARD CAT")
print("=" * 60)


if "source_folder" in leopard_df.columns:

    source_counts = (
        leopard_df["source_folder"]
        .value_counts()
    )

    for source, count in source_counts.items():

        print(
            f"{source} : "
            f"{count} event"
        )

else:

    print(
        "Kolom source_folder tidak tersedia."
    )


# =========================================================
# BUAT PREVIEW
# =========================================================

print("\n" + "=" * 60)
print("MEMBUAT PREVIEW EVENT")
print("=" * 60)


for idx, row in leopard_df.iterrows():

    event_id = row["event_id"]

    event_folder = os.path.join(
        OUTPUT_DIR,
        f"{idx + 1:02d}_{event_id}"
    )

    os.makedirs(
        event_folder,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Ambil 3 foto event
    # -----------------------------------------------------

    image_columns = [
        "foto_1",
        "foto_2",
        "foto_3"
    ]

    for foto_idx, column in enumerate(
        image_columns,
        start=1
    ):

        source_path = row[column]

        if not os.path.exists(source_path):

            print(
                f"[WARNING] File tidak ditemukan:"
                f"\n{source_path}"
            )

            continue

        destination_path = os.path.join(
            event_folder,
            f"{event_id}_foto_{foto_idx}.jpg"
        )

        shutil.copy2(
            source_path,
            destination_path
        )


# =========================================================
# SIMPAN CSV KHUSUS LEOPARD
# =========================================================

leopard_csv = os.path.join(
    OUTPUT_DIR,
    "leopard_cat_events.csv"
)

leopard_df.to_csv(
    leopard_csv,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print(
    f"\nPreview dibuat di:"
    f"\n{OUTPUT_DIR}"
)

print(
    f"\nCSV leopard_cat:"
    f"\n{leopard_csv}"
)

print(
    "\nJumlah event yang dipreview: "
    f"{len(leopard_df)}"
)

print(
    "\nSetiap folder event berisi "
    "3 foto dari event tersebut."
)

print(
    "\nTujuan tahap ini:"
    "\n1. Melihat kemunculan leopard cat."
    "\n2. Membandingkan ciri antar-event."
    "\n3. Menentukan kandidat individu yang sama."
    "\n4. Menyiapkan label individual identification."
)

print(
    "\nAudit selesai."
)