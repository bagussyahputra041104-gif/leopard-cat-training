import os
import pandas as pd

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "orientation",
    "orientation_labels.csv"
)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"File tidak ditemukan:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV,
    dtype=str,
    keep_default_na=False
)

print("=" * 70)
print("AUDIT LABELING ARAH HADAP LEOPARD CAT")
print("=" * 70)

print()

total = len(df)

valid_labels = [
    "depan",
    "belakang",
    "kiri",
    "kanan",
    "tidak_yakin"
]

jumlah_valid = df["arah_hadap"].isin(
    valid_labels
).sum()

jumlah_kosong = (
    df["arah_hadap"]
    .fillna("")
    .eq("")
    .sum()
)

jumlah_tidak_valid = (
    (~df["arah_hadap"].isin(valid_labels))
    &
    (df["arah_hadap"] != "")
).sum()

print(f"Total foto              : {total}")
print(f"Sudah diberi label      : {jumlah_valid}")
print(f"Belum diberi label      : {jumlah_kosong}")
print(f"Label tidak valid       : {jumlah_tidak_valid}")

print()
print("-" * 70)
print("DISTRIBUSI LABEL")
print("-" * 70)

for label in valid_labels:

    jumlah = (
        df["arah_hadap"] == label
    ).sum()

    persentase = (
        jumlah / total * 100
        if total > 0
        else 0
    )

    print(
        f"{label:15s}: "
        f"{jumlah:3d} foto "
        f"({persentase:6.2f}%)"
    )

print()
print("-" * 70)
print("DISTRIBUSI PER EVENT")
print("-" * 70)

jumlah_event = df["event_id"].nunique()

print(
    f"Total event             : {jumlah_event}"
)

print()

# Cek setiap event apakah punya 3 foto
foto_per_event = (
    df.groupby("event_id")
    .size()
)

event_tidak_3 = foto_per_event[
    foto_per_event != 3
]

print(
    f"Event dengan 3 foto     : "
    f"{(foto_per_event == 3).sum()}"
)

print(
    f"Event bukan 3 foto      : "
    f"{len(event_tidak_3)}"
)

if len(event_tidak_3) > 0:

    print()
    print("Event bermasalah:")

    for event_id, jumlah in event_tidak_3.items():

        print(
            f"  {event_id}: {jumlah} foto"
        )

print()
print("-" * 70)
print("STATUS")
print("-" * 70)

if (
    jumlah_kosong == 0
    and jumlah_tidak_valid == 0
    and len(event_tidak_3) == 0
):

    print("AUDIT BERHASIL")
    print("Semua 189 foto memiliki label yang valid.")

else:

    print("PERLU DICEK")
    print("Masih ada data yang perlu diperiksa.")

print("=" * 70)