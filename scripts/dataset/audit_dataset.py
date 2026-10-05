from pathlib import Path

# ============================================================
# LOKASI DATASET
# ============================================================

DATASET_DIR = Path(r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang")

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
# MENCARI SEMUA GAMBAR
# ============================================================

def get_images(folder):
    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# AUDIT DATASET
# ============================================================

def audit_dataset(name, folder):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    if not folder.exists():
        print("[ERROR] Folder tidak ditemukan:")
        print(folder)
        return 0, 0

    all_images = get_images(folder)

    image_folders = []

    for directory in folder.rglob("*"):

        if not directory.is_dir():
            continue

        images_here = [
            file
            for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images_here:
            image_folders.append(
                (directory, images_here)
            )

    print("\nLokasi:")
    print(folder)

    print(f"\nTotal gambar: {len(all_images)}")
    print(f"Folder berisi gambar: {len(image_folders)}")

    print("\nDetail folder:")
    print("-" * 70)

    for nomor, (directory, images) in enumerate(
        image_folders, start=1
    ):

        relative_path = directory.relative_to(folder)

        print(f"\n{nomor}. {relative_path}")
        print(f"   Jumlah gambar: {len(images)}")

        for image in images:
            print(f"      - {image.name}")

    return len(all_images), len(image_folders)


# ============================================================
# MULAI AUDIT
# ============================================================

print("=" * 70)
print("AUDIT DATASET CAMERA TRAP")
print("=" * 70)

print("\nFolder utama dataset:")
print(DATASET_DIR)

print("\nMemeriksa dataset...")


leopard_images, leopard_folders = audit_dataset(
    "LEOPARD CAT",
    LEOPARD_DIR
)


null_images, null_folders = audit_dataset(
    "TANPA SATWA / NULL",
    NULL_DIR
)


# ============================================================
# RINGKASAN
# ============================================================

print("\n\n" + "=" * 70)
print("RINGKASAN DATASET")
print("=" * 70)

print(f"""
LEOPARD CAT
  Folder berisi gambar : {leopard_folders}
  Total gambar         : {leopard_images}

NULL / TANPA SATWA
  Folder berisi gambar : {null_folders}
  Total gambar         : {null_images}

TOTAL
  Folder berisi gambar : {leopard_folders + null_folders}
  Total gambar         : {leopard_images + null_images}
""")

print("=" * 70)
print("AUDIT SELESAI")
print("=" * 70)

print("\nTidak ada file yang diubah.")