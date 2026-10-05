import os
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

RESULT_CSV = os.path.join(
    BASE_DIR,
    "reid_crops",
    "detection_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reid_crops",
    "contact_sheet"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "all_detections.jpg"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    RESULT_CSV,
    dtype=str,
    keep_default_na=False
)

df = df[
    df["detected"].str.lower() == "true"
].copy()

# Urutkan berdasarkan confidence tertinggi
df["confidence_num"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)

df = df.sort_values(
    by="confidence_num",
    ascending=False
)

print("=" * 60)
print("DETECTION CONTACT SHEET")
print("=" * 60)

print(
    f"Jumlah detection : {len(df)}"
)


# =========================================================
# CONTACT SHEET CONFIG
# =========================================================

THUMB_WIDTH = 260
THUMB_HEIGHT = 210

LABEL_HEIGHT = 45

CELL_WIDTH = THUMB_WIDTH
CELL_HEIGHT = THUMB_HEIGHT + LABEL_HEIGHT

COLUMNS = 5

ROWS = math.ceil(
    len(df) / COLUMNS
)

SHEET_WIDTH = (
    COLUMNS * CELL_WIDTH
)

SHEET_HEIGHT = (
    ROWS * CELL_HEIGHT
)


sheet = Image.new(
    "RGB",
    (
        SHEET_WIDTH,
        SHEET_HEIGHT
    ),
    "white"
)

draw = ImageDraw.Draw(
    sheet
)


# =========================================================
# FONT
# =========================================================

try:

    font = ImageFont.truetype(
        "arial.ttf",
        14
    )

except:

    font = ImageFont.load_default()


# =========================================================
# BUILD SHEET
# =========================================================

for index, row in enumerate(
    df.itertuples()
):

    crop_path = row.crop_path

    if not os.path.exists(
        crop_path
    ):
        continue

    try:

        image = Image.open(
            crop_path
        ).convert("RGB")

        image.thumbnail(
            (
                THUMB_WIDTH - 10,
                THUMB_HEIGHT - 10
            )
        )

        x_cell = (
            index % COLUMNS
        ) * CELL_WIDTH

        y_cell = (
            index // COLUMNS
        ) * CELL_HEIGHT

        x_image = (
            x_cell +
            (
                CELL_WIDTH -
                image.width
            ) // 2
        )

        y_image = (
            y_cell +
            5 +
            (
                THUMB_HEIGHT -
                image.height
            ) // 2
        )

        sheet.paste(
            image,
            (
                x_image,
                y_image
            )
        )

        confidence = float(
            row.confidence
        )

        label_1 = (
            f"{row.event_id} | "
            f"foto {row.photo_index}"
        )

        label_2 = (
            f"confidence = "
            f"{confidence:.3f}"
        )

        draw.text(
            (
                x_cell + 5,
                y_cell + THUMB_HEIGHT + 3
            ),
            label_1,
            fill="black",
            font=font
        )

        draw.text(
            (
                x_cell + 5,
                y_cell + THUMB_HEIGHT + 22
            ),
            label_2,
            fill="black",
            font=font
        )

    except Exception as e:

        print(
            f"Gagal membaca: "
            f"{crop_path}"
        )

        print(e)


# =========================================================
# SAVE
# =========================================================

sheet.save(
    OUTPUT_FILE,
    quality=95
)

print()
print(
    "Contact sheet berhasil dibuat:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)