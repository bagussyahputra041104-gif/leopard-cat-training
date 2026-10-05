import os
import pandas as pd
from PIL import Image, ImageChops, ImageStat

# ============================================================
# PENGATURAN
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

ORIENTATION_CSV = os.path.join(
    BASE_DIR,
    "orientation",
    "orientation_labels.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "movement"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "movement_candidates.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

orientation = pd.read_csv(
    ORIENTATION_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# VALIDASI
# ============================================================

required_master = [
    "event_id",
    "timestamp",
    "foto_1",
    "foto_2",
    "foto_3"
]

required_orientation = [
    "event_id",
    "foto",
    "arah_hadap"
]

for col in required_master:

    if col not in master.columns:

        raise ValueError(
            f"Kolom {col} tidak ditemukan di master CSV."
        )


for col in required_orientation:

    if col not in orientation.columns:

        raise ValueError(
            f"Kolom {col} tidak ditemukan di orientation CSV."
        )


# ============================================================
# FUNGSI PERBANDINGAN GAMBAR
# ============================================================

def image_difference(path1, path2):

    if (
        not os.path.exists(path1)
        or not os.path.exists(path2)
    ):

        return None

    try:

        img1 = Image.open(path1).convert(
            "RGB"
        )

        img2 = Image.open(path2).convert(
            "RGB"
        )

        size = (256, 256)

        img1 = img1.resize(size)
        img2 = img2.resize(size)

        diff = ImageChops.difference(
            img1,
            img2
        )

        stat = ImageStat.Stat(diff)

        mean_difference = sum(
            stat.mean
        ) / 3

        return mean_difference

    except Exception:

        return None


# ============================================================
# MENGAMBIL ORIENTASI DALAM SATU EVENT
# ============================================================

def get_orientation(
    event_id,
    foto
):

    result = orientation[
        (orientation["event_id"] == event_id)
        &
        (orientation["foto"] == foto)
    ]

    if len(result) == 0:

        return ""

    return result.iloc[0]["arah_hadap"]


# ============================================================
# ANALISIS
# ============================================================

results = []


for _, row in master.iterrows():

    event_id = row["event_id"]

    foto1 = row["foto_1"]
    foto2 = row["foto_2"]
    foto3 = row["foto_3"]

    # --------------------------------------------------------
    # ORIENTASI
    # --------------------------------------------------------

    arah1 = get_orientation(
        event_id,
        "foto_1"
    )

    arah2 = get_orientation(
        event_id,
        "foto_2"
    )

    arah3 = get_orientation(
        event_id,
        "foto_3"
    )

    # --------------------------------------------------------
    # PERUBAHAN GAMBAR
    # --------------------------------------------------------

    diff12 = image_difference(
        foto1,
        foto2
    )

    diff23 = image_difference(
        foto2,
        foto3
    )

    # --------------------------------------------------------
    # ATURAN AWAL
    # --------------------------------------------------------

    movement = "tidak_yakin"

    alasan = ""

    # Jika salah satu orientasi tidak yakin,
    # jangan memaksakan arah gerak.

    if (
        "tidak_yakin" in
        [arah1, arah2, arah3]
    ):

        movement = "tidak_yakin"

        alasan = (
            "Terdapat foto dengan "
            "orientasi tidak yakin."
        )

    else:

        # ----------------------------------------------------
        # PERUBAHAN ARAH HADAP
        # ----------------------------------------------------

        if (
            arah1 == arah2
            and
            arah2 == arah3
        ):

            movement = "tidak_yakin"

            alasan = (
                "Orientasi tetap; "
                "belum cukup bukti untuk "
                "menentukan arah gerak."
            )

        elif (
            arah1 == "kiri"
            and
            arah3 == "kanan"
        ):

            movement = "kanan"

            alasan = (
                "Orientasi berubah "
                "dari kiri ke kanan."
            )

        elif (
            arah1 == "kanan"
            and
            arah3 == "kiri"
        ):

            movement = "kiri"

            alasan = (
                "Orientasi berubah "
                "dari kanan ke kiri."
            )

        elif (
            arah1 == "depan"
            and
            arah3 == "belakang"
        ):

            movement = "menjauh"

            alasan = (
                "Orientasi berubah "
                "dari depan ke belakang."
            )

        elif (
            arah1 == "belakang"
            and
            arah3 == "depan"
        ):

            movement = "mendekat"

            alasan = (
                "Orientasi berubah "
                "dari belakang ke depan."
            )

        else:

            movement = "tidak_yakin"

            alasan = (
                "Perubahan orientasi "
                "belum cukup untuk "
                "menentukan arah gerak."
            )

    # --------------------------------------------------------
    # SIMPAN
    # --------------------------------------------------------

    results.append({

        "event_id": event_id,

        "timestamp": row["timestamp"],

        "orientasi_foto_1": arah1,

        "orientasi_foto_2": arah2,

        "orientasi_foto_3": arah3,

        "perubahan_gambar_12": diff12,

        "perubahan_gambar_23": diff23,

        "movement_candidate": movement,

        "alasan": alasan
    })


# ============================================================
# OUTPUT
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# RINGKASAN
# ============================================================

print("=" * 70)
print("MOVEMENT ANALYSIS")
print("=" * 70)

print()

print(
    f"Total event : {len(result_df)}"
)

print()

print(
    "Distribusi kandidat movement:"
)

print()

counts = (
    result_df[
        "movement_candidate"
    ]
    .value_counts()
)

for label, jumlah in counts.items():

    persen = (
        jumlah
        /
        len(result_df)
        *
        100
    )

    print(
        f"{label:15s}: "
        f"{jumlah:3d} event "
        f"({persen:6.2f}%)"
    )

print()

print("-" * 70)

print(
    f"Output : {OUTPUT_CSV}"
)

print("-" * 70)

print()

print(
    "CATATAN:"
)

print(
    "Movement yang dihasilkan merupakan "
    "candidate movement, bukan ground truth."
)

print(
    "Hasil perlu divalidasi secara visual sebelum "
    "digunakan untuk analisis akhir."
)

print()

print("=" * 70)