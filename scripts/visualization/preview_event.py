from pathlib import Path
import csv
import shutil

BASE_DIR = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

CSV_FILE = BASE_DIR / "event_dataset.csv"
OUTPUT_DIR = BASE_DIR / "preview_event"


def main():

    print("=" * 60)
    print("MEMBUAT PREVIEW EVENT")
    print("=" * 60)

    if not CSV_FILE.exists():
        print("❌ event_dataset.csv tidak ditemukan.")
        return

    # Hapus preview lama jika ada
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir()

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    # Ambil contoh:
    # 2 leopard cat + 2 null + 2 tambahan
    selected = []

    leopard = [
        row for row in rows
        if row["label"] == "leopard_cat"
    ]

    null = [
        row for row in rows
        if row["label"] == "null"
    ]

    # Contoh leopard cat
    selected.extend(leopard[:2])

    # Contoh null
    selected.extend(null[:2])

    # Tambahkan event lain sebagai contoh
    selected.append(leopard[10])
    selected.append(null[10])

    for index, row in enumerate(
        selected,
        start=1
    ):

        event_dir = (
            OUTPUT_DIR
            / f"{index:02d}_{row['label']}_{row['event_id']}"
        )

        event_dir.mkdir()

        for foto_no in range(1, 4):

            source = Path(
                row[f"foto_{foto_no}"]
            )

            destination = (
                event_dir
                / f"foto_{foto_no}{source.suffix}"
            )

            if source.exists():

                shutil.copy2(
                    source,
                    destination
                )

        # Simpan info event
        info = event_dir / "info.txt"

        info.write_text(
            f"Event ID : {row['event_id']}\n"
            f"Label    : {row['label']}\n"
            f"Timestamp: {row['timestamp']}\n",
            encoding="utf-8"
        )

    print()
    print("✅ Preview berhasil dibuat.")
    print()
    print("Lokasi:")
    print(OUTPUT_DIR)
    print()
    print("Jumlah event preview:", len(selected))

    print("=" * 60)


if __name__ == "__main__":
    main()