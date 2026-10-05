from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS


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
# MEMBACA EXIF
# ============================================================

def get_exif_data(image_path):

    try:
        image = Image.open(image_path)
        exif = image.getexif()

        data = {}

        for tag_id, value in exif.items():

            tag_name = TAGS.get(tag_id, tag_id)

            data[tag_name] = value

        return data

    except Exception as e:

        return {
            "ERROR": str(e)
        }


# ============================================================
# AUDIT METADATA
# ============================================================

def audit_metadata(name, folder):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    images = []

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(file)

    images = sorted(images)

    print(f"\nTotal gambar: {len(images)}")

    print("\nMetadata 15 gambar pertama:")
    print("-" * 70)

    for image_path in images[:15]:

        exif = get_exif_data(image_path)

        print("\nFile:")
        print(image_path.relative_to(folder))

        if "ERROR" in exif:

            print("ERROR:")
            print(exif["ERROR"])
            continue

        datetime_value = exif.get("DateTimeOriginal")
        datetime_digitized = exif.get("DateTimeDigitized")
        datetime_modified = exif.get("DateTime")

        print("DateTimeOriginal :", datetime_value)
        print("DateTimeDigitized:", datetime_digitized)
        print("DateTime         :", datetime_modified)

    print("\n" + "-" * 70)


# ============================================================
# JALANKAN
# ============================================================

print("=" * 70)
print("AUDIT METADATA / TIMESTAMP CAMERA TRAP")
print("=" * 70)

audit_metadata(
    "LEOPARD CAT",
    LEOPARD_DIR
)

audit_metadata(
    "TANPA SATWA / NULL",
    NULL_DIR
)

print("\n" + "=" * 70)
print("AUDIT METADATA SELESAI")
print("=" * 70)

print("\nDataset tidak diubah.")