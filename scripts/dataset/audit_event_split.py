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


def print_event_group(group, label):

    awal = group[0]["timestamp"]
    akhir = group[-1]["timestamp"]

    print(
        f"    {label} | "
        f"{len(group)} foto | "
        f"{awal.strftime('%H:%M:%S')} - "
        f"{akhir.strftime('%H:%M:%S')}"
    )


def main():

    print("\n" + "=" * 65)
    print("AUDIT PEMECAHAN KANDIDAT EVENT")
    print("=" * 65)

    for folder in FOLDERS:

        if not folder.exists():
            continue

        images = collect_images(folder)
        events = group_events(images)

        if "leopard cat" in folder.name.lower():
            nama = "LEOPARD CAT"
        else:
            nama = "NULL / TANPA SATWA"

        print("\n" + "=" * 65)
        print(nama)
        print("=" * 65)

        kandidat = 0
        anomali = 0

        for nomor, event in enumerate(
            events,
            start=1
        ):

            jumlah = len(event)

            # Event normal
            if jumlah == 3:
                continue

            # Event 6, 9, 15 dst.
            if jumlah > 3 and jumlah % 3 == 0:

                kandidat += 1

                print(
                    f"\nEVENT ASLI {nomor}: "
                    f"{jumlah} foto"
                )

                print(
                    "  Kandidat pemecahan:"
                )

                for i in range(
                    0,
                    jumlah,
                    3
                ):

                    group = event[
                        i:i + 3
                    ]

                    nomor_group = (
                        i // 3
                    ) + 1

                    print_event_group(
                        group,
                        f"Kelompok {nomor_group}"
                    )

            # Event 1 atau 2 foto
            else:

                anomali += 1

                awal = event[0]["timestamp"]
                akhir = event[-1]["timestamp"]

                print(
                    f"\nANOMALI EVENT {nomor}: "
                    f"{jumlah} foto | "
                    f"{awal.strftime('%H:%M:%S')} - "
                    f"{akhir.strftime('%H:%M:%S')}"
                )

        print("\n" + "-" * 65)

        print(
            f"Kandidat event yang bisa dibagi 3 : "
            f"{kandidat}"
        )

        print(
            f"Event anomali 1-2/dll              : "
            f"{anomali}"
        )

    print("\n" + "=" * 65)
    print("SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    main()