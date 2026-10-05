import csv
from pathlib import Path

BASE_DIR = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

CSV_FILE = BASE_DIR / "event_dataset.csv"


def main():

    print("=" * 60)
    print("AUDIT EVENT DATASET")
    print("=" * 60)

    if not CSV_FILE.exists():

        print("❌ File event_dataset.csv tidak ditemukan.")

        return

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    total = len(rows)

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

    event_id = [
        row["event_id"]
        for row in rows
    ]

    duplicate_event = (
        len(event_id)
        - len(set(event_id))
    )

    missing_path = 0

    for row in rows:

        for column in [
            "foto_1",
            "foto_2",
            "foto_3"
        ]:

            if not Path(row[column]).exists():

                missing_path += 1

    invalid_rows = 0

    for row in rows:

        if not all(
            row[column].strip()
            for column in [
                "event_id",
                "label",
                "foto_1",
                "foto_2",
                "foto_3",
                "timestamp"
            ]
        ):

            invalid_rows += 1

    print()
    print(f"Total event          : {total}")
    print(f"Leopard cat          : {leopard}")
    print(f"Null / tanpa satwa   : {null}")

    print()
    print(f"Duplikat event ID    : {duplicate_event}")
    print(f"Path foto tidak ada  : {missing_path}")
    print(f"Baris tidak lengkap  : {invalid_rows}")

    print()

    if (
        total == 135
        and leopard == 63
        and null == 72
        and duplicate_event == 0
        and missing_path == 0
        and invalid_rows == 0
    ):

        print("✅ AUDIT CSV BERHASIL")
        print("Metadata event siap untuk tahap berikutnya.")

    else:

        print("⚠️ Ada bagian yang perlu diperiksa.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()