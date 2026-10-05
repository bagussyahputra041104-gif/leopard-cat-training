import os
import math
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "contact_sheet_leopard"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# DISPLAY CONFIG
# =========================================================

THUMB_WIDTH = 300
THUMB_HEIGHT = 220

TEXT_HEIGHT = 55

CELL_WIDTH = THUMB_WIDTH
CELL_HEIGHT = THUMB_HEIGHT + TEXT_HEIGHT

COLUMNS = 3


# =========================================================
# FONT
# =========================================================

try:

    FONT = ImageFont.truetype(
        "arial.ttf",
        16
    )

    SMALL_FONT = ImageFont.truetype(
        "arial.ttf",
        13
    )

except:

    FONT = ImageFont.load_default()

    SMALL_FONT = ImageFont.load_default()


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("MEMBUAT CONTACT SHEET LEOPARD CAT")
print("=" * 60)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

print(
    f"Total event leopard_cat: {len(df)}"
)


# =========================================================
# SOURCE FOLDERS
# =========================================================

sources = sorted(
    df["source_folder"].unique()
)

print(
    f"Total source folder: {len(sources)}"
)


# =========================================================
# HELPER
# =========================================================

def resize_image(image):

    image = image.convert("RGB")

    image.thumbnail(
        (
            THUMB_WIDTH,
            THUMB_HEIGHT
        )
    )

    canvas = Image.new(
        "RGB",
        (
            THUMB_WIDTH,
            THUMB_HEIGHT
        ),
        "white"
    )

    x = (
        THUMB_WIDTH -
        image.width
    ) // 2

    y = (
        THUMB_HEIGHT -
        image.height
    ) // 2

    canvas.paste(
        image,
        (x, y)
    )

    return canvas


# =========================================================
# CREATE CONTACT SHEET
# =========================================================

for source in sources:

    source_df = df[
        df["source_folder"] == source
    ].copy()

    source_df = source_df.sort_values(
        "timestamp"
    )

    print(
        f"\nProcessing:"
        f" {source}"
    )

    print(
        f"Event: {len(source_df)}"
    )


    # -----------------------------------------------------
    # 1 EVENT = 1 CELL
    #
    # Dalam setiap cell:
    # foto 1, foto 2, foto 3
    # -----------------------------------------------------

    event_cells = []


    for _, row in source_df.iterrows():

        event_id = row["event_id"]

        timestamp = row["timestamp"]

        foto_paths = [
            row["foto_1"],
            row["foto_2"],
            row["foto_3"]
        ]


        # -------------------------------------------------
        # Event canvas
        # -------------------------------------------------

        event_canvas = Image.new(
            "RGB",
            (
                THUMB_WIDTH,
                THUMB_HEIGHT + TEXT_HEIGHT
            ),
            "white"
        )


        # -------------------------------------------------
        # Buat strip 3 foto
        # -------------------------------------------------

        photo_width = THUMB_WIDTH // 3

        photo_height = THUMB_HEIGHT


        for i, foto_path in enumerate(
            foto_paths
        ):

            if os.path.exists(
                foto_path
            ):

                try:

                    image = Image.open(
                        foto_path
                    )

                    image = image.convert(
                        "RGB"
                    )

                    image.thumbnail(
                        (
                            photo_width,
                            photo_height
                        )
                    )

                    photo_canvas = Image.new(
                        "RGB",
                        (
                            photo_width,
                            photo_height
                        ),
                        "white"
                    )

                    x = (
                        photo_width -
                        image.width
                    ) // 2

                    y = (
                        photo_height -
                        image.height
                    ) // 2

                    photo_canvas.paste(
                        image,
                        (x, y)
                    )

                    event_canvas.paste(
                        photo_canvas,
                        (
                            i * photo_width,
                            0
                        )
                    )

                except Exception as e:

                    print(
                        f"[WARNING] "
                        f"Gagal membuka:"
                        f"\n{foto_path}"
                    )

            else:

                print(
                    f"[WARNING] "
                    f"File tidak ditemukan:"
                    f"\n{foto_path}"
                )


        # -------------------------------------------------
        # Border
        # -------------------------------------------------

        draw = ImageDraw.Draw(
            event_canvas
        )

        draw.rectangle(
            [
                0,
                0,
                THUMB_WIDTH - 1,
                THUMB_HEIGHT - 1
            ],
            outline="black",
            width=2
        )


        # -------------------------------------------------
        # Event information
        # -------------------------------------------------

        draw.text(
            (
                5,
                THUMB_HEIGHT + 4
            ),
            event_id,
            fill="black",
            font=FONT
        )

        draw.text(
            (
                5,
                THUMB_HEIGHT + 25
            ),
            timestamp,
            fill="black",
            font=SMALL_FONT
        )


        event_cells.append(
            event_canvas
        )


    # =====================================================
    # BUILD SHEET
    # =====================================================

    rows = math.ceil(
        len(event_cells) /
        COLUMNS
    )


    sheet_width = (
        COLUMNS *
        CELL_WIDTH
    )

    sheet_height = (
        rows *
        CELL_HEIGHT
    )


    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )


    # =====================================================
    # PLACE EVENTS
    # =====================================================

    for index, cell in enumerate(
        event_cells
    ):

        row_index = (
            index //
            COLUMNS
        )

        col_index = (
            index %
            COLUMNS
        )


        x = (
            col_index *
            CELL_WIDTH
        )

        y = (
            row_index *
            CELL_HEIGHT
        )


        sheet.paste(
            cell,
            (
                x,
                y
            )
        )


    # =====================================================
    # SAFE FILE NAME
    # =====================================================

    safe_source = source

    for char in [
        "\\",
        "/",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|"
    ]:

        safe_source = safe_source.replace(
            char,
            "_"
        )


    output_path = os.path.join(
        OUTPUT_DIR,
        safe_source + ".jpg"
    )


    # =====================================================
    # SAVE
    # =====================================================

    sheet.save(
        output_path,
        quality=90
    )


    print(
        f"Saved:"
        f"\n{output_path}"
    )


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print(
    f"\nContact sheet disimpan di:"
    f"\n{OUTPUT_DIR}"
)

print(
    f"\nJumlah source:"
    f" {len(sources)}"
)

print(
    "\nSetiap kotak = 1 event."
)

print(
    "Setiap kotak berisi 3 foto "
    "dari event tersebut."
)