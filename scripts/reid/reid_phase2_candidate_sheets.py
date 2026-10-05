from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang")

CSV_PATH = (
    ROOT
    / "05_REID" / "crop"
    / "cross_time"
    / "phase2_unassigned"
    / "phase2_unassigned_candidates.csv"
)

MASTER_PATH = ROOT / "leopard_cat_master.csv"

OUT_DIR = (
    ROOT
    / "05_REID" / "crop"
    / "cross_time"
    / "phase2_unassigned"
    / "candidate_sheets"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# AMBIL 5 PASANGAN TERATAS
# ============================================================

df = pd.read_csv(CSV_PATH)
master = pd.read_csv(MASTER_PATH)

TOP_N = 5
pairs = df.head(TOP_N).copy()

# Mapping event_id -> 3 foto
event_map = {}

for _, row in master.iterrows():
    event_map[row["event_id"]] = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"],
    ]


# ============================================================
# FONT
# ============================================================

try:
    font_big = ImageFont.truetype("arial.ttf", 28)
    font_small = ImageFont.truetype("arial.ttf", 20)
except:
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()


def load_image(path):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    img = Image.open(path).convert("RGB")
    return img


def make_thumbnail(img, size=(500, 350)):
    return ImageOps.contain(img, size)


# ============================================================
# BUAT SHEET
# ============================================================

for rank, row in enumerate(pairs.itertuples(index=False), start=1):

    event1 = row.event_1
    event2 = row.event_2
    similarity = float(row.similarity)
    gap_days = float(row.time_gap_days)

    photos1 = event_map[event1]
    photos2 = event_map[event2]

    sheet_width = 1600
    cell_w = 520
    cell_h = 430

    sheet_height = 100 + (cell_h * 2)

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    title = (
        f"PHASE 2 CANDIDATE {rank:02d} | "
        f"{event1} <-> {event2}"
    )

    subtitle = (
        f"Similarity: {similarity:.4f}    "
        f"Time gap: {gap_days:.2f} hari"
    )

    draw.text(
        (30, 20),
        title,
        fill="black",
        font=font_big
    )

    draw.text(
        (30, 60),
        subtitle,
        fill="black",
        font=font_small
    )

    # --------------------------------------------------------
    # EVENT 1
    # --------------------------------------------------------

    draw.text(
        (30, 110),
        event1,
        fill="black",
        font=font_big
    )

    for i, photo in enumerate(photos1):

        img = load_image(photo)
        thumb = make_thumbnail(img, (480, 340))

        x = 20 + i * 520
        y = 145

        sheet.paste(thumb, (x, y))

        draw.rectangle(
            [x, y, x + 480, y + 340],
            outline="black",
            width=2
        )

        draw.text(
            (x, y + 345),
            f"{event1} - Foto {i+1}",
            fill="black",
            font=font_small
        )

    # --------------------------------------------------------
    # EVENT 2
    # --------------------------------------------------------

    draw.text(
        (30, 545),
        event2,
        fill="black",
        font=font_big
    )

    for i, photo in enumerate(photos2):

        img = load_image(photo)
        thumb = make_thumbnail(img, (480, 340))

        x = 20 + i * 520
        y = 580

        sheet.paste(thumb, (x, y))

        draw.rectangle(
            [x, y, x + 480, y + 340],
            outline="black",
            width=2
        )

        draw.text(
            (x, y + 345),
            f"{event2} - Foto {i+1}",
            fill="black",
            font=font_small
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    out = OUT_DIR / (
        f"candidate_{rank:02d}_{event1}_{event2}.jpg"
    )

    sheet.save(out, quality=95)

    print(f"[{rank:02d}] {event1} <-> {event2}")
    print(f"     Similarity : {similarity:.4f}")
    print(f"     Gap        : {gap_days:.2f} hari")
    print(f"     Saved      : {out}")
    print()


print("=" * 70)
print("SELESAI")
print("=" * 70)
print(f"Output folder:")
print(OUT_DIR)