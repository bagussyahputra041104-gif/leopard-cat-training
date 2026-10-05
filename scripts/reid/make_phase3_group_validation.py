from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# ROOT PROJECT
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)


# ============================================================
# INPUT
# ============================================================

EVENT_CSV = (
    ROOT
    / "02_DATASET"
    / "dataset"
    / "event_dataset.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "05_REID"
    / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "phase3_group_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EVENT YANG DIVALIDASI
# ============================================================

EVENTS = [
    "EVT_0012",
    "EVT_0026",
    "EVT_0037",
    "EVT_0055",
]


# ============================================================
# BACA DATA EVENT
# ============================================================

if not EVENT_CSV.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan:\n{EVENT_CSV}"
    )

df = pd.read_csv(EVENT_CSV)


required_columns = [
    "event_id",
    "foto_1",
    "foto_2",
    "foto_3",
]

for col in required_columns:
    if col not in df.columns:
        raise KeyError(
            f"Kolom '{col}' tidak ditemukan."
        )


# ============================================================
# MAPPING EVENT -> FOTO
# ============================================================

event_map = {}

for _, row in df.iterrows():

    event_id = row["event_id"]

    event_map[event_id] = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"],
    ]


# ============================================================
# CEK EVENT
# ============================================================

for event_id in EVENTS:

    if event_id not in event_map:

        raise ValueError(
            f"{event_id} tidak ditemukan di event_dataset.csv"
        )


# ============================================================
# FONT
# ============================================================

try:

    title_font = ImageFont.truetype(
        "arial.ttf",
        24
    )

    label_font = ImageFont.truetype(
        "arial.ttf",
        18
    )

except:

    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()


# ============================================================
# FUNGSI LOAD FOTO
# ============================================================

def load_image(path):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        return None

    try:
        return Image.open(path).convert("RGB")

    except Exception:
        return None


# ============================================================
# KONFIGURASI SHEET
# ============================================================

cell_w = 400
cell_h = 300

header_h = 80

cols = 3
rows = len(EVENTS)

sheet_w = cell_w * cols
sheet_h = header_h + cell_h * rows


# ============================================================
# BUAT CANVAS
# ============================================================

canvas = Image.new(
    "RGB",
    (sheet_w, sheet_h),
    "white"
)

draw = ImageDraw.Draw(canvas)


# ============================================================
# HEADER
# ============================================================

draw.text(
    (15, 10),
    "PHASE 3 GROUP VALIDATION",
    fill="black",
    font=title_font
)

draw.text(
    (15, 45),
    "EVT_0012 | EVT_0026 | EVT_0037 | EVT_0055",
    fill="black",
    font=label_font
)


# ============================================================
# MASUKKAN FOTO
# ============================================================

for row_index, event_id in enumerate(EVENTS):

    photos = event_map[event_id]

    for col_index, photo_path in enumerate(photos):

        x = col_index * cell_w

        y = (
            header_h
            + row_index * cell_h
        )

        img = load_image(photo_path)


        # ----------------------------------------------------
        # FOTO TIDAK DITEMUKAN
        # ----------------------------------------------------

        if img is None:

            draw.rectangle(
                [
                    x,
                    y,
                    x + cell_w - 1,
                    y + cell_h - 1
                ],
                outline="red",
                width=3
            )

            draw.text(
                (
                    x + 10,
                    y + 10
                ),
                "IMAGE NOT FOUND",
                fill="red",
                font=label_font
            )

            continue


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        img.thumbnail(
            (
                cell_w - 20,
                cell_h - 55
            )
        )


        # ----------------------------------------------------
        # POSISI TENGAH
        # ----------------------------------------------------

        px = (
            x
            + (cell_w - img.width) // 2
        )

        py = y + 8


        canvas.paste(
            img,
            (px, py)
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{event_id} - Foto "
            f"{col_index + 1}"
        )

        draw.text(
            (
                x + 10,
                y + cell_h - 35
            ),
            label,
            fill="black",
            font=label_font
        )


        # ----------------------------------------------------
        # BORDER
        # ----------------------------------------------------

        draw.rectangle(
            [
                x,
                y,
                x + cell_w - 1,
                y + cell_h - 1
            ],
            outline="gray",
            width=2
        )


# ============================================================
# SIMPAN
# ============================================================

output_path = (
    OUT_DIR
    / "phase3_group_validation_0012_0026_0037_0055.jpg"
)

canvas.save(
    output_path,
    quality=95
)


print()
print("=" * 60)
print("SELESAI")
print("=" * 60)
print()
print(f"Output:")
print(output_path)