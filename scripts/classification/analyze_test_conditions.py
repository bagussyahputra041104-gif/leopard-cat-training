from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

PREDICTION_CSV = (
    ROOT
    / "04_HASIL"
    / "classification"
    / "evaluation_baseline"
    / "test_predictions.csv"
)

TEST_EVENTS_CSV = (
    ROOT
    / "02_DATASET"
    / "dataset"
    / "test_events.csv"
)

OUTPUT_DIR = (
    ROOT
    / "04_HASIL"
    / "classification"
    / "condition_analysis"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "condition_analysis.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "condition_summary.txt"
)


# ============================================================
# CHECK FILE
# ============================================================

if not PREDICTION_CSV.exists():
    raise FileNotFoundError(
        f"File prediksi tidak ditemukan:\n{PREDICTION_CSV}"
    )

if not TEST_EVENTS_CSV.exists():
    raise FileNotFoundError(
        f"File test events tidak ditemukan:\n{TEST_EVENTS_CSV}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

pred_df = pd.read_csv(
    PREDICTION_CSV,
    dtype=str,
    keep_default_na=False
)

event_df = pd.read_csv(
    TEST_EVENTS_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# VALIDASI KOLOM
# ============================================================

prediction_required = [
    "file",
    "actual",
    "predicted",
    "confidence",
    "correct"
]

event_required = [
    "event_id",
    "foto_1",
    "foto_2",
    "foto_3"
]

for col in prediction_required:

    if col not in pred_df.columns:
        raise ValueError(
            f"Kolom prediksi '{col}' tidak ditemukan."
        )


for col in event_required:

    if col not in event_df.columns:
        raise ValueError(
            f"Kolom event '{col}' tidak ditemukan."
        )


# ============================================================
# BUAT MAP FOTO ASLI
# ============================================================

photo_map = {}

for _, row in event_df.iterrows():

    event_id = row["event_id"]

    for photo_number in range(1, 4):

        path = row[f"foto_{photo_number}"]

        key = f"{event_id}_foto_{photo_number}"

        photo_map[key] = path


# ============================================================
# FUNGSI BRIGHTNESS
# ============================================================

def calculate_brightness(image):

    gray = image.convert("L")

    array = np.asarray(
        gray,
        dtype=np.float32
    )

    return float(
        array.mean()
    )


# ============================================================
# FUNGSI SHARPNESS
# ============================================================

def calculate_sharpness(image):

    gray = image.convert("L")

    array = np.asarray(
        gray,
        dtype=np.float32
    )

    laplacian = (
        -4 * array
        + np.roll(array, 1, axis=0)
        + np.roll(array, -1, axis=0)
        + np.roll(array, 1, axis=1)
        + np.roll(array, -1, axis=1)
    )

    return float(
        laplacian.var()
    )


# ============================================================
# ANALISIS
# ============================================================

results = []

missing = []

print()
print("=" * 65)
print("ANALISIS KONDISI FOTO TEST")
print("=" * 65)

print(
    f"Total prediksi : {len(pred_df)}"
)

print()


for _, row in pred_df.iterrows():

    filename = Path(
        row["file"]
    ).stem

    # Contoh:
    # EVT_0002_foto_1
    key = filename

    if key not in photo_map:

        missing.append(
            key
        )

        print(
            f"[WARNING] Tidak ditemukan di test_events.csv: {key}"
        )

        continue

    original_path = Path(
        photo_map[key]
    )

    # --------------------------------------------------------
    # Path lama mungkin masih menunjuk ke lokasi sebelum
    # struktur project dirapikan.
    #
    # Jika tidak ada, cari berdasarkan nama file ASLI
    # di 01_DATA_ASLI.
    # --------------------------------------------------------

    if not original_path.exists():

        raw_dir = (
            ROOT
            / "01_DATA_ASLI"
            / "leopard_cat"
        )

        candidates = list(
            raw_dir.rglob(
                original_path.name
            )
        )

        if candidates:

            original_path = candidates[0]

        else:

            missing.append(
                key
            )

            print(
                f"[WARNING] Foto asli tidak ditemukan: "
                f"{original_path.name}"
            )

            continue

    try:

        image = Image.open(
            original_path
        ).convert("RGB")

        brightness = calculate_brightness(
            image
        )

        sharpness = calculate_sharpness(
            image
        )

        results.append(
            {
                "event_id": key.rsplit(
                    "_foto_",
                    1
                )[0],

                "foto": key.rsplit(
                    "_foto_",
                    1
                )[1],

                "file_prediction": row["file"],

                "path_foto_asli": str(
                    original_path
                ),

                "actual": row["actual"],

                "predicted": row["predicted"],

                "confidence": float(
                    row["confidence"]
                ),

                "correct": row["correct"],

                "brightness": brightness,

                "sharpness": sharpness,
            }
        )

    except Exception as error:

        print(
            f"[WARNING] Gagal membaca foto: "
            f"{original_path}"
        )

        print(
            f"          {error}"
        )


# ============================================================
# CEK HASIL
# ============================================================

if not results:

    raise RuntimeError(
        "Tidak ada foto yang berhasil dianalisis."
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# KATEGORI RELATIF
# ============================================================

brightness_low = result_df[
    "brightness"
].quantile(0.33)

brightness_high = result_df[
    "brightness"
].quantile(0.66)

sharpness_low = result_df[
    "sharpness"
].quantile(0.33)

sharpness_high = result_df[
    "sharpness"
].quantile(0.66)


def brightness_category(value):

    if value <= brightness_low:
        return "RELATIF_GELAP"

    elif value >= brightness_high:
        return "RELATIF_TERANG"

    else:
        return "SEDANG"


def sharpness_category(value):

    if value <= sharpness_low:
        return "INDIKASI_BLUR_TINGGI"

    elif value >= sharpness_high:
        return "TAJAM"

    else:
        return "SEDANG"


result_df[
    "brightness_category"
] = result_df[
    "brightness"
].apply(
    brightness_category
)


result_df[
    "sharpness_category"
] = result_df[
    "sharpness"
].apply(
    sharpness_category
)


# ============================================================
# SIMPAN CSV
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# AKURASI
# ============================================================

correct_mask = (
    result_df["correct"]
    .astype(str)
    .str.lower()
    .eq("true")
)

total = len(result_df)

correct_count = int(
    correct_mask.sum()
)

wrong_count = (
    total - correct_count
)

accuracy = (
    correct_count
    / total
    * 100
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append(
    "ANALISIS KONDISI FOTO TEST - RESNET18"
)

summary.append(
    "=" * 60
)

summary.append(
    f"Total foto dianalisis : {total}"
)

summary.append(
    f"Prediksi benar        : {correct_count}"
)

summary.append(
    f"Prediksi salah        : {wrong_count}"
)

summary.append(
    f"Akurasi               : {accuracy:.2f}%"
)

summary.append("")

summary.append(
    "BRIGHTNESS"
)

summary.append(
    f"Minimum : "
    f"{result_df['brightness'].min():.2f}"
)

summary.append(
    f"Median  : "
    f"{result_df['brightness'].median():.2f}"
)

summary.append(
    f"Maximum : "
    f"{result_df['brightness'].max():.2f}"
)

summary.append("")

summary.append(
    "SHARPNESS"
)

summary.append(
    f"Minimum : "
    f"{result_df['sharpness'].min():.2f}"
)

summary.append(
    f"Median  : "
    f"{result_df['sharpness'].median():.2f}"
)

summary.append(
    f"Maximum : "
    f"{result_df['sharpness'].max():.2f}"
)

summary.append("")

summary.append(
    "KATEGORI BRIGHTNESS"
)

for name, group in result_df.groupby(
    "brightness_category"
):

    group_correct = (
        group["correct"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    group_accuracy = (
        group_correct
        / len(group)
        * 100
    )

    summary.append(
        f"{name}: "
        f"{len(group)} foto | "
        f"benar {group_correct} | "
        f"akurasi {group_accuracy:.2f}%"
    )


summary.append("")

summary.append(
    "KATEGORI SHARPNESS"
)

for name, group in result_df.groupby(
    "sharpness_category"
):

    group_correct = (
        group["correct"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    group_accuracy = (
        group_correct
        / len(group)
        * 100
    )

    summary.append(
        f"{name}: "
        f"{len(group)} foto | "
        f"benar {group_correct} | "
        f"akurasi {group_accuracy:.2f}%"
    )


summary.append("")

summary.append(
    "CATATAN"
)

summary.append(
    "Brightness dan sharpness merupakan "
    "indikator otomatis, bukan ground truth "
    "manual kondisi foto."
)

summary.append(
    "Kategori relatif ditentukan berdasarkan "
    "distribusi nilai pada data test."
)

summary.append(
    "Analisis ini digunakan sebagai analisis "
    "pendukung terhadap performa klasifikasi."
)


if missing:

    summary.append("")

    summary.append(
        f"Foto/ID yang tidak ditemukan: "
        f"{len(missing)}"
    )


with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary)
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 65)
print("SELESAI")
print("=" * 65)

print(
    f"Foto berhasil dianalisis : {total}"
)

print(
    f"Prediksi benar           : {correct_count}"
)

print(
    f"Prediksi salah           : {wrong_count}"
)

print(
    f"Akurasi                  : {accuracy:.2f}%"
)

print()

print(
    f"CSV     : {OUTPUT_CSV}"
)

print(
    f"Summary : {SUMMARY_TXT}"
)

if missing:

    print()
    print(
        f"Foto tidak ditemukan: {len(missing)}"
    )

print("=" * 65)