from pathlib import Path
from PIL import Image
from datetime import datetime
import csv

BASE_DIR = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

OUTPUT_FILE = BASE_DIR / "event_dataset.csv"

FOLDERS = [
    (
        BASE_DIR / "01_DATA_ASLI" / "leopard_cat",
        "leopard_cat"
    ),
    (
        BASE_DIR / "01_DATA_ASLI" / "tanpa_satwa",
        "null"
    ),
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


def create_event_rows():

    rows = []

    event_number = 1

    for folder, label in FOLDERS:

        images = collect_images(folder)
        events = group_events(images)

        for event in events:

            jumlah = len(event)

            # Hanya event lengkap 3 foto
            if jumlah == 3:

                rows.append({
                    "event_id": f"EVT_{event_number:04d}",
                    "label": label,
                    "foto_1": str(
                        event[0]["path"]
                    ),
                    "foto_2": str(
                        event[1]["path"]
                    ),
                    "foto_3": str(
                        event[2]["path"]
                    ),
                    "timestamp": event[0][
                        "timestamp"
                    ].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                })

                event_number += 1

            # Event 6, 9, 15, dst.
            elif jumlah > 3 and jumlah % 3 == 0:

                for i in range(
                    0,
                    jumlah,
                    3
                ):

                    group = event[
                        i:i + 3
                    ]

                    rows.append({
                        "event_id": f"EVT_{event_number:04d}",
                        "label": label,
                        "foto_1": str(
                            group[0]["path"]
                        ),
                        "foto_2": str(
                            group[1]["path"]
                        ),
                        "foto_3": str(
                            group[2]["path"]
                        ),
                        "timestamp": group[0][
                            "timestamp"
                        ].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    })

                    event_number += 1

            # Event 1 atau 2 foto
            # tidak dimasukkan ke dataset final

    return rows


def save_csv(rows):

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "event_id",
            "label",
            "foto_1",
            "foto_2",
            "foto_3",
            "timestamp"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


def main():

    print("=" * 60)
    print("MEMBUAT METADATA EVENT CAMERA TRAP")
    print("=" * 60)

    rows = create_event_rows()

    save_csv(rows)

    leopard = sum(
        1
        for row in rows
        if row["label"] == "leopard_cat"
    )

    null = sum(
        1
        for row in rows
        if row["label"] == "null"
    )

    print()
    print(f"Total event       : {len(rows)}")
    print(f"Leopard cat event : {leopard}")
    print(f"Null event        : {null}")

    print()
    print("File berhasil dibuat:")
    print(OUTPUT_FILE)

    print()
    print("Foto asli TIDAK dipindahkan.")
    print("Foto asli TIDAK dihapus.")

    print("=" * 60)


if __name__ == "__main__":
    main()