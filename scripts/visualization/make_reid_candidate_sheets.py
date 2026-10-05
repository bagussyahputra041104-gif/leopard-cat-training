from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_CSV = BASE_DIR / "leopard_cat_master.csv"
TOP_PAIRS_CSV = (
    BASE_DIR
    / "05_REID" / "crop"
    / "cross_time"
    / "cross_time_top_pairs.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "05_REID" / "crop"
    / "cross_time"
    / "candidate_sheets"
)

TOP_N = 10

IMAGE_WIDTH = 420
IMAGE_HEIGHT = 300

PADDING = 20
HEADER_HEIGHT = 100

BACKGROUND = "white"
TEXT_COLOR = "black"
LINE_COLOR = "black"


# ============================================================
# FONT
# ============================================================

def get_font(size, bold=False):
    """
    Coba beberapa font Windows.
    Jika tidak ditemukan, gunakan font default Pillow.
    """

    candidates = []

    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]

    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


FONT_TITLE = get_font(26, bold=True)
FONT_HEADER = get_font(22, bold=True)
FONT_NORMAL = get_font(18)
FONT_SMALL = get_font(16)


# ============================================================
# HELPER
# ============================================================

def resolve_image_path(path_string):
    """
    Mengubah path dari CSV menjadi Path yang benar.
    """

    path = Path(str(path_string))

    if path.is_absolute():
        return path

    return BASE_DIR / path


def load_image(image_path):
    """
    Membuka image dan melakukan resize dengan menjaga aspect ratio.
    """

    try:
        image = Image.open(image_path).convert("RGB")

    except Exception as e:
        print(f"[WARNING] Gagal membuka: {image_path}")
        print(f"          {e}")

        image = Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            "lightgray"
        )

        draw = ImageDraw.Draw(image)

        draw.text(
            (20, IMAGE_HEIGHT // 2),
            "IMAGE ERROR",
            fill="red",
            font=FONT_NORMAL
        )

        return image

    image.thumbnail(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        "white"
    )

    x = (IMAGE_WIDTH - image.width) // 2
    y = (IMAGE_HEIGHT - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def draw_centered_text(draw, text, y, font, width):
    """
    Menulis teks di tengah.
    """

    bbox = draw.textbbox((0, 0), text, font=font)

    text_width = bbox[2] - bbox[0]

    x = (width - text_width) // 2

    draw.text(
        (x, y),
        text,
        fill=TEXT_COLOR,
        font=font
    )


# ============================================================
# CREATE EVENT PANEL
# ============================================================

def create_event_panel(event_id, event_row, side_title):
    """
    Membuat panel yang berisi 3 foto dari satu event.
    """

    panel_width = IMAGE_WIDTH + (PADDING * 2)

    panel_height = (
        HEADER_HEIGHT
        + (IMAGE_HEIGHT * 3)
        + (PADDING * 4)
    )

    panel = Image.new(
        "RGB",
        (panel_width, panel_height),
        BACKGROUND
    )

    draw = ImageDraw.Draw(panel)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    draw_centered_text(
        draw,
        side_title,
        10,
        FONT_HEADER,
        panel_width
    )

    draw_centered_text(
        draw,
        event_id,
        42,
        FONT_NORMAL,
        panel_width
    )

    # --------------------------------------------------------
    # 3 photos
    # --------------------------------------------------------

    photo_columns = [
        "foto_1",
        "foto_2",
        "foto_3"
    ]

    y = HEADER_HEIGHT

    for idx, column in enumerate(photo_columns, start=1):

        image_path = resolve_image_path(
            event_row[column]
        )

        image = load_image(image_path)

        x = PADDING

        panel.paste(
            image,
            (x, y)
        )

        draw.rectangle(
            [
                x,
                y,
                x + IMAGE_WIDTH,
                y + IMAGE_HEIGHT
            ],
            outline=LINE_COLOR,
            width=2
        )

        draw.text(
            (x + 8, y + 8),
            f"Foto {idx}",
            fill=TEXT_COLOR,
            font=FONT_SMALL
        )

        y += IMAGE_HEIGHT + PADDING

    return panel


# ============================================================
# CREATE CANDIDATE SHEET
# ============================================================

def create_candidate_sheet(
    rank,
    event_1,
    event_2,
    row_1,
    row_2,
    similarity,
    gap_days,
    gap_hours
):

    left_panel = create_event_panel(
        event_1,
        row_1,
        "EVENT A"
    )

    right_panel = create_event_panel(
        event_2,
        row_2,
        "EVENT B"
    )

    panel_width = left_panel.width

    sheet_width = (
        panel_width * 2
        + PADDING
    )

    sheet_height = (
        HEADER_HEIGHT
        + left_panel.height
        + PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        BACKGROUND
    )

    draw = ImageDraw.Draw(sheet)

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    title = f"RE-ID CANDIDATE #{rank}"

    draw.text(
        (PADDING, 10),
        title,
        fill=TEXT_COLOR,
        font=FONT_TITLE
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    info_1 = (
        f"{event_1}  vs  {event_2}"
    )

    info_2 = (
        f"Similarity: {similarity:.4f}"
        f"   |   Time gap: {gap_days:.2f} hari"
    )

    draw.text(
        (PADDING, 45),
        info_1,
        fill=TEXT_COLOR,
        font=FONT_NORMAL
    )

    draw.text(
        (PADDING, 70),
        info_2,
        fill=TEXT_COLOR,
        font=FONT_SMALL
    )

    # --------------------------------------------------------
    # Paste panels
    # --------------------------------------------------------

    y = HEADER_HEIGHT

    sheet.paste(
        left_panel,
        (0, y)
    )

    sheet.paste(
        right_panel,
        (panel_width + PADDING, y)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"candidate_{rank:02d}_{event_1}_{event_2}.jpg"
    )

    sheet.save(
        output_path,
        quality=95
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MAKE RE-ID CANDIDATE SHEETS")
    print("=" * 70)

    print()
    print(f"Master CSV : {MASTER_CSV}")
    print(f"Pairs CSV  : {TOP_PAIRS_CSV}")
    print(f"Output     : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"Master CSV tidak ditemukan:\n{MASTER_CSV}"
        )

    if not TOP_PAIRS_CSV.exists():
        raise FileNotFoundError(
            f"Top pairs CSV tidak ditemukan:\n{TOP_PAIRS_CSV}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    master_df = pd.read_csv(
        MASTER_CSV,
        dtype=str,
        keep_default_na=False
    )

    pairs_df = pd.read_csv(
        TOP_PAIRS_CSV
    )

    print(f"Total event master : {len(master_df)}")
    print(f"Total candidate    : {len(pairs_df)}")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Index event
    # --------------------------------------------------------

    master_index = master_df.set_index(
        "event_id"
    )

    # --------------------------------------------------------
    # Take top N
    # --------------------------------------------------------

    pairs_df = pairs_df.head(TOP_N)

    print()
    print(f"Membuat Top {len(pairs_df)} candidate sheets...")
    print()

    created = []

    for rank, (_, pair) in enumerate(
        pairs_df.iterrows(),
        start=1
    ):

        event_1 = str(pair["event_1"])
        event_2 = str(pair["event_2"])

        similarity = float(
            pair["similarity"]
        )

        gap_hours = float(
            pair["time_gap_hours"]
        )

        gap_days = float(
            pair["time_gap_days"]
        )

        # ----------------------------------------------------
        # Check event
        # ----------------------------------------------------

        if event_1 not in master_index.index:
            print(
                f"[SKIP] {event_1} tidak ditemukan."
            )
            continue

        if event_2 not in master_index.index:
            print(
                f"[SKIP] {event_2} tidak ditemukan."
            )
            continue

        row_1 = master_index.loc[event_1]

        row_2 = master_index.loc[event_2]

        # ----------------------------------------------------
        # Create sheet
        # ----------------------------------------------------

        output_path = create_candidate_sheet(
            rank=rank,
            event_1=event_1,
            event_2=event_2,
            row_1=row_1,
            row_2=row_2,
            similarity=similarity,
            gap_days=gap_days,
            gap_hours=gap_hours
        )

        created.append(output_path)

        print(
            f"[{rank:02d}] "
            f"{event_1} vs {event_2} | "
            f"similarity={similarity:.4f} | "
            f"gap={gap_days:.2f} hari"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELESAI")
    print("=" * 70)

    print()
    print(f"Candidate sheet dibuat : {len(created)}")
    print(f"Folder output          :")
    print(OUTPUT_DIR)

    print()
    print("File yang dibuat:")

    for path in created:
        print(f" - {path.name}")


if __name__ == "__main__":
    main()