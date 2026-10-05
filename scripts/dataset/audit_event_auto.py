from pathlib import Path
from PIL import Image
from datetime import datetime

BASE_DIR = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

FOLDERS = [
    BASE_DIR / "01_DATA_ASLI" / "leopard_cat",
    BASE_DIR / "01_DATA_ASLI" / "tanpa_satwa",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
}

MAX_GAP_SECONDS = 3


def get_timestamp(path):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            date_time = exif.get(306)

            if date_time:
                return datetime.strptime(
                    date_time,
                    "%Y:%m:%d %H:%M:%S"
                )

    except Exception:
        pass

    return None


def collect_images(folder):
    images = []

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix in IMAGE_EXTENSIONS
        ):

            timestamp = get_timestamp(path)

            if timestamp:
                images.append({
                    "path": path,
                    "timestamp": timestamp
                })

    return images


def group_events(images):

    images.sort(
        key=lambda x: x["timestamp"]
    )

    events = []
    current = []

    for image in images:

        if not current:
            current.append(image)
            continue

        gap = (
            image["timestamp"]
            - current[-1]["timestamp"]
        ).total_seconds()

        if gap <= MAX_GAP_SECONDS:
            current.append(image)

        else:
            events.append(current)
            current = [image]

    if current:
        events.append(current)

    return events


def audit_folder(folder):

    images = collect_images(folder)
    events = group_events(images)

    counts = {}

    for event in events:

        jumlah = len(event)

        if jumlah not in counts:
            counts[jumlah] = 0

        counts[jumlah] += 1

    return images, events, counts


def print_summary(name, images, events, counts):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Total foto       : {len(images)}")
    print(f"Kandidat event   : {len(events)}")

    print("\nDistribusi event:")

    for jumlah in sorted(counts):

        print(
            f"  {jumlah:>2} foto : "
            f"{counts[jumlah]} event"
        )

    print("\nEvent yang perlu validasi:")

    nomor = 1

    for i, event in enumerate(events, start=1):

        if len(event) != 3:

            waktu_awal = event[0]["timestamp"]
            waktu_akhir = event[-1]["timestamp"]

            print(
                f"  Event {i}: "
                f"{len(event)} foto | "
                f"{waktu_awal.strftime('%H:%M:%S')} - "
                f"{waktu_akhir.strftime('%H:%M:%S')}"
            )

            nomor += 1

    if nomor == 1:
        print("  Tidak ada.")


def main():

    print("\n" + "=" * 60)
    print("AUDIT EVENT CAMERA TRAP")
    print("=" * 60)

    for folder in FOLDERS:

        if not folder.exists():

            print(
                f"\nFolder tidak ditemukan:\n{folder}"
            )

            continue

        images, events, counts = audit_folder(
            folder
        )

        if "leopard cat" in folder.name.lower():

            name = "LEOPARD CAT"

        else:

            name = "NULL / TANPA SATWA"

        print_summary(
            name,
            images,
            events,
            counts
        )

    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()