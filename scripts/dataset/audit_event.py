from pathlib import Path
from collections import defaultdict


# ============================================================
# LOKASI DATASET
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

LEOPARD_DIR = DATASET_DIR / "01_DATA_ASLI" / "leopard_cat"
NULL_DIR = DATASET_DIR / "01_DATA_ASLI" / "tanpa_satwa"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# AUDIT KELOMPOK GAMBAR
# ============================================================

def audit_folder(name, folder):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    if not folder.exists():
        print("[ERROR] Folder tidak ditemukan:")
        print(folder)
        return

    total_images = 0

    for directory in sorted(folder.rglob("*")):

        if not directory.is_dir():
            continue

        images = sorted([
            file
            for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            continue

        total_images += len(images)

        print("\nFolder:")
        print(directory.relative_to(folder))

        print(f"Jumlah gambar: {len(images)}")

        # ----------------------------------------------------
        # CEK PEMBAGIAN 3 FOTO
        # ----------------------------------------------------

        full_events = len(images) // 3
        remainder = len(images) % 3

        print(f"Perkiraan kelompok 3 foto: {full_events}")

        if remainder == 0:
            print("Status: jumlah gambar habis dibagi 3")
        else:
            print(
                f"Status: ADA SISA {remainder} gambar "
                "yang belum membentuk kelompok 3"
            )

    print("\n" + "-" * 70)
    print(f"TOTAL GAMBAR DATASET: {total_images}")


# ============================================================
# JALANKAN AUDIT
# ============================================================

print("=" * 70)
print("AUDIT EVENT CAMERA TRAP")
print("=" * 70)

print("\nTujuan:")
print("Memeriksa apakah jumlah gambar dalam folder")
print("dapat dikelompokkan menjadi event berisi 3 foto.")

audit_folder(
    "LEOPARD CAT",
    LEOPARD_DIR
)

audit_folder(
    "TANPA SATWA / NULL",
    NULL_DIR
)

print("\n" + "=" * 70)
print("AUDIT EVENT SELESAI")
print("=" * 70)

print("\nTidak ada file dataset yang diubah.")