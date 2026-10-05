from pathlib import Path
from datetime import datetime
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
# MEMBACA TIMESTAMP
# ============================================================

def get_datetime(image_path):

    try:
        image = Image.open(image_path)
        exif = image.getexif()

        for tag_id, value in exif.items():

            tag_name = TAGS.get(tag_id, tag_id)

            if tag_name == "DateTime":
                return datetime.strptime(
                    str(value),
                    "%Y:%m:%d %H:%M:%S"
                )

    except Exception:
        pass

    return None


# ============================================================
# MEMBACA SEMUA FOTO DALAM SATU FOLDER
# ============================================================

def get_images(folder):

    images = []

    for file in folder.iterdir():

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):

            timestamp = get_datetime(file)

            if timestamp is not None:

                images.append(
                    (file, timestamp)
                )

    return sorted(
        images,
        key=lambda x: x[1]
    )


# ============================================================
# DETEKSI EVENT
# ============================================================

def detect_events(images, max_gap_seconds=5):

    events = []

    current_event = []

    for image_path, timestamp in images:

        if not current_event:

            current_event.append(
                (image_path, timestamp)
            )

            continue

        previous_timestamp = current_event[-1][1]

        gap = (
            timestamp - previous_timestamp
        ).total_seconds()

        if gap <= max_gap_seconds:

            current_event.append(
                (image_path, timestamp)
            )

        else:

            events.append(current_event)

            current_event = [
                (image_path, timestamp)
            ]

    if current_event:

        events.append(current_event)

    return events


# ============================================================
# AUDIT SATU FOLDER
# ============================================================

def audit_dataset(name, folder):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    total_images = 0
    total_events = 0

    for directory in sorted(folder.rglob("*")):

        if not directory.is_dir():
            continue

        images = get_images(directory)

        if not images:
            continue

        events = detect_events(images)

        total_images += len(images)
        total_events += len(events)

        print("\nFolder:")
        print(directory.relative_to(folder))

        print(f"Jumlah gambar : {len(images)}")
        print(f"Perkiraan event: {len(events)}")

        # Tampilkan maksimal 10 event pertama
        for nomor, event in enumerate(events[:10], start=1):

            print(
                f"\n  Event {nomor}: "
                f"{len(event)} foto"
            )

            for image_path, timestamp in event:

                print(
                    f"    {timestamp.strftime('%H:%M:%S')} "
                    f"- {image_path.name}"
                )

        if len(events) > 10:

            print(
                f"\n  ... {len(events) - 10} event lainnya"
            )

    print("\n" + "-" * 70)
    print(f"TOTAL GAMBAR : {total_images}")
    print(f"TOTAL EVENT  : {total_events}")


# ============================================================
# JALANKAN
# ============================================================

print("=" * 70)
print("AUDIT EVENT BERDASARKAN TIMESTAMP")
print("=" * 70)

print("""
Aturan sementara:
Foto yang memiliki jarak waktu <= 5 detik
dianggap masih berada dalam satu kelompok event.

Ini hanya AUDIT.
Dataset tidak diubah.
""")

audit_dataset(
    "LEOPARD CAT",
    LEOPARD_DIR
)

audit_dataset(
    "TANPA SATWA / NULL",
    NULL_DIR
)

print("\n" + "=" * 70)
print("AUDIT EVENT TIMESTAMP SELESAI")
print("=" * 70)