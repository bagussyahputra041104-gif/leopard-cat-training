from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. ROOT PROJECT
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)


# ============================================================
# 2. FILE INPUT
# ============================================================

CSV_PATH = (
    ROOT
    / "05_REID"
    / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "phase3_unassigned_candidates.csv"
)

EVENT_CSV = (
    ROOT
    / "02_DATASET"
    / "dataset"
    / "event_dataset.csv"
)


# ============================================================
# 3. FOLDER OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "05_REID"
    / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "phase3_candidate_sheets"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. PASANGAN YANG AKAN DIREVIEW
# ============================================================

TARGET_PAIRS = [
    ("EVT_0012", "EVT_0055"),
    ("EVT_0037", "EVT_0055"),
    ("EVT_0026", "EVT_0055"),
    ("EVT_0005", "EVT_0012"),
    ("EVT_0012", "EVT_0061"),
]


# ============================================================
# 5. CEK FILE INPUT
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"File kandidat tidak ditemukan:\n{CSV_PATH}"
    )

if not EVENT_CSV.exists():
    raise FileNotFoundError(
        f"File event dataset tidak ditemukan:\n{EVENT_CSV}"
    )


# ============================================================
# 6. BACA CSV
# ============================================================

df_candidates = pd.read_csv(CSV_PATH)
df_events = pd.read_csv(EVENT_CSV)


print("Kolom kandidat:")
print(df_candidates.columns.tolist())

print()


# ============================================================
# 7. CEK KOLOM YANG DIBUTUHKAN
# ============================================================

required_candidate_columns = [
    "event_1",
    "event_2",
    "similarity",
    "time_gap_days",
]

for column in required_candidate_columns:
    if column not in df_candidates.columns:
        raise KeyError(
            f"Kolom '{column}' tidak ditemukan di CSV kandidat."
        )


# ============================================================
# 8. BUAT MAPPING EVENT -> 3 FOTO
# ============================================================

event_map = {}

for _, row in df_events.iterrows():

    event_id = row["event_id"]

    event_map[event_id] = [
        row["foto_1"],
        row["foto_2"],
        row["foto_3"],
    ]


# ============================================================
# 9. FONT
# ============================================================

try:
    font = ImageFont.truetype(
        "arial.ttf",
        22
    )

    small_font = ImageFont.truetype(
        "arial.ttf",
        18
    )

except:

    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


# ============================================================
# 10. FUNGSI MEMBUKA FOTO
# ============================================================

def load_image(path):

    path = Path(path)

    # Jika path bukan absolute,
    # gabungkan dengan ROOT
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        return None

    try:
        return Image.open(path).convert("RGB")

    except Exception:
        return None


# ============================================================
# 11. FUNGSI MEMBUAT CONTACT SHEET
# ============================================================

def make_sheet(
    event_a,
    event_b,
    similarity,
    gap_days,
    number
):

    photos_a = event_map.get(
        event_a,
        []
    )

    photos_b = event_map.get(
        event_b,
        []
    )


    # Ukuran setiap kotak foto
    cell_w = 420
    cell_h = 330

    # Tinggi bagian header
    header_h = 100


    # Canvas
    canvas = Image.new(
        "RGB",
        (
            cell_w * 3,
            header_h + cell_h * 2
        ),
        "white"
    )


    draw = ImageDraw.Draw(canvas)


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    title = (
        f"PHASE 3 CANDIDATE #{number}\n"
        f"{event_a}  <->  {event_b}"
    )

    draw.text(
        (15, 10),
        title,
        fill="black",
        font=font
    )


    info = (
        f"Similarity: {similarity:.4f}    "
        f"Gap: {gap_days:.2f} hari"
    )

    draw.text(
        (15, 62),
        info,
        fill="black",
        font=small_font
    )


    # --------------------------------------------------------
    # EVENT YANG AKAN DITAMPILKAN
    # --------------------------------------------------------

    all_events = [
        (
            event_a,
            photos_a,
            0
        ),
        (
            event_b,
            photos_b,
            1
        ),
    ]


    # --------------------------------------------------------
    # MASUKKAN FOTO
    # --------------------------------------------------------

    for event_id, photos, row_index in all_events:

        for col_index, photo_path in enumerate(photos):

            x = col_index * cell_w

            y = (
                header_h
                + row_index * cell_h
            )


            img = load_image(photo_path)


            # ------------------------------------------------
            # JIKA FOTO TIDAK DITEMUKAN
            # ------------------------------------------------

            if img is None:

                draw.rectangle(
                    [
                        x,
                        y,
                        x + cell_w,
                        y + cell_h
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
                    font=font
                )

                continue


            # ------------------------------------------------
            # RESIZE FOTO
            # ------------------------------------------------

            img.thumbnail(
                (
                    cell_w - 20,
                    cell_h - 60
                )
            )


            # ------------------------------------------------
            # POSISI FOTO DI TENGAH
            # ------------------------------------------------

            px = (
                x
                + (cell_w - img.width) // 2
            )

            py = y + 10


            canvas.paste(
                img,
                (
                    px,
                    py
                )
            )


            # ------------------------------------------------
            # LABEL FOTO
            # ------------------------------------------------

            label = (
                f"{event_id} - "
                f"Foto {col_index + 1}"
            )

            draw.text(
                (
                    x + 10,
                    y + cell_h - 35
                ),
                label,
                fill="black",
                font=small_font
            )


            # ------------------------------------------------
            # BORDER
            # ------------------------------------------------

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


    # ========================================================
    # 12. SIMPAN
    # ========================================================

    filename = (
        f"candidate_{number:02d}_"
        f"{event_a}_{event_b}.jpg"
    )

    output_path = (
        OUT_DIR / filename
    )


    canvas.save(
        output_path,
        quality=95
    )


    print(
        f"[OK] {output_path}"
    )


# ============================================================
# 13. PROSES 5 KANDIDAT
# ============================================================

for number, (event_a, event_b) in enumerate(
    TARGET_PAIRS,
    start=6
):

    print()
    print(
        f"Memproses Candidate #{number}: "
        f"{event_a} <-> {event_b}"
    )


    # Cari pasangan di CSV
    mask = (
        (
            (df_candidates["event_1"] == event_a)
            &
            (df_candidates["event_2"] == event_b)
        )
        |
        (
            (df_candidates["event_1"] == event_b)
            &
            (df_candidates["event_2"] == event_a)
        )
    )


    matches = df_candidates[mask]


    # --------------------------------------------------------
    # JIKA PASANGAN TIDAK DITEMUKAN
    # --------------------------------------------------------

    if matches.empty:

        print(
            f"[SKIP] "
            f"{event_a} <-> {event_b} "
            f"tidak ditemukan di CSV."
        )

        continue


    # Ambil baris pertama
    row = matches.iloc[0]


    # --------------------------------------------------------
    # BUAT SHEET
    # --------------------------------------------------------

    make_sheet(
        event_a=event_a,
        event_b=event_b,
        similarity=float(
            row["similarity"]
        ),
        gap_days=float(
            row["time_gap_days"]
        ),
        number=number
    )


# ============================================================
# 14. SELESAI
# ============================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)
print()
print(
    f"Output berada di:\n{OUT_DIR}"
)