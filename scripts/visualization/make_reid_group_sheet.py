import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

MASTER = BASE / "leopard_cat_master.csv"

OUTPUT_DIR = (
    BASE
    / "05_REID" / "crop"
    / "cross_time"
    / "group_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "group_validation_8_events.jpg"


# ============================================================
# EVENT YANG AKAN DIBANDINGKAN
# ============================================================

EVENTS = [
    "EVT_0002",
    "EVT_0058",
    "EVT_0060",
    "EVT_0013",
    "EVT_0025",
    "EVT_0015",
    "EVT_0024",
    "EVT_0035",
]


# ============================================================
# LOAD MASTER
# ============================================================

df = pd.read_csv(
    MASTER,
    dtype=str,
    keep_default_na=False
)

df = df.set_index("event_id")


# ============================================================
# FONT
# ============================================================

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


FONT_TITLE = get_font(30)
FONT_INFO = get_font(20)
FONT_SMALL = get_font(16)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    path = Path(path)

    if not path.exists():
        return None

    try:
        return Image.open(path).convert("RGB")
    except:
        return None


# ============================================================
# BUAT PANEL EVENT
# ============================================================

def make_event_panel(event_id):

    row = df.loc[event_id]

    photos = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"],
    ]

    panel_w = 1500
    panel_h = 700

    canvas = Image.new(
        "RGB",
        (panel_w, panel_h),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    draw.text(
        (20, 15),
        event_id,
        fill="black",
        font=FONT_TITLE
    )

    draw.text(
        (20, 55),
        f"Timestamp: {row['timestamp']}",
        fill="black",
        font=FONT_INFO
    )

    draw.text(
        (20, 90),
        f"Source: {row['source_folder']}",
        fill="black",
        font=FONT_SMALL
    )

    # --------------------------------------------------------
    # FOTO
    # --------------------------------------------------------

    top = 140
    gap = 15

    photo_w = (
        panel_w - (4 * gap)
    ) // 3

    photo_h = 480

    for i, photo_path in enumerate(photos):

        x = gap + i * (
            photo_w + gap
        )

        y = top

        img = load_image(photo_path)

        if img is None:

            draw.rectangle(
                [
                    x,
                    y,
                    x + photo_w,
                    y + photo_h
                ],
                outline="red",
                width=3
            )

            draw.text(
                (x + 10, y + 10),
                "IMAGE NOT FOUND",
                fill="red",
                font=FONT_SMALL
            )

            continue

        # resize
        img.thumbnail(
            (
                photo_w - 10,
                photo_h - 10
            )
        )

        paste_x = (
            x
            + (photo_w - img.width) // 2
        )

        paste_y = (
            y
            + (photo_h - img.height) // 2
        )

        canvas.paste(
            img,
            (paste_x, paste_y)
        )

        draw.rectangle(
            [
                x,
                y,
                x + photo_w,
                y + photo_h
            ],
            outline="black",
            width=2
        )

        draw.text(
            (
                x + 5,
                y + photo_h + 10
            ),
            f"Foto {i+1}",
            fill="black",
            font=FONT_SMALL
        )

    return canvas


# ============================================================
# BUAT GROUP SHEET
# ============================================================

print("=" * 70)
print("MEMBUAT GROUP VALIDATION SHEET")
print("=" * 70)

print("\nEvent:")

for event in EVENTS:
    print(" -", event)


panels = []

for event in EVENTS:

    print(f"Membuat panel {event}")

    panel = make_event_panel(event)

    panels.append(panel)


# ============================================================
# LAYOUT 2 KOLOM
# ============================================================

columns = 2
rows = 4

panel_w = panels[0].width
panel_h = panels[0].height

gap_x = 20
gap_y = 20

sheet_w = (
    columns * panel_w
    + (columns + 1) * gap_x
)

sheet_h = (
    rows * panel_h
    + (rows + 1) * gap_y
)

sheet = Image.new(
    "RGB",
    (sheet_w, sheet_h),
    "white"
)


# ============================================================
# PASTE PANEL
# ============================================================

for i, panel in enumerate(panels):

    row = i // columns
    col = i % columns

    x = (
        gap_x
        + col * (panel_w + gap_x)
    )

    y = (
        gap_y
        + row * (panel_h + gap_y)
    )

    sheet.paste(
        panel,
        (x, y)
    )


# ============================================================
# SAVE
# ============================================================

sheet.save(
    OUTPUT,
    quality=95
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)

print("\nOutput:")
print(OUTPUT)