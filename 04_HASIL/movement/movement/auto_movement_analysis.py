import os
import pandas as pd


# ============================================================
# PENGATURAN
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

DETECTION_CSV = os.path.join(
    BASE_DIR,
    "reid_crops",
    "detection_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "movement"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "auto_movement_candidates.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "auto_movement_summary.txt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CEK FILE
# ============================================================

if not os.path.exists(MASTER_CSV):

    raise FileNotFoundError(
        f"Master CSV tidak ditemukan:\n{MASTER_CSV}"
    )

if not os.path.exists(DETECTION_CSV):

    raise FileNotFoundError(
        f"Detection CSV tidak ditemukan:\n{DETECTION_CSV}"
    )


# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

detections = pd.read_csv(
    DETECTION_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# INFORMASI
# ============================================================

print("=" * 70)
print("AUTO MOVEMENT ANALYSIS")
print("=" * 70)

print()

print(
    "Kolom detection_results.csv:"
)

for col in detections.columns:

    print(
        f" - {col}"
    )

print()


# ============================================================
# KOLOM YANG KITA GUNAKAN
# ============================================================

EVENT_COL = "event_id"
PHOTO_INDEX_COL = "photo_index"

CONFIDENCE_COL = "confidence"

X1_COL = "x1"
Y1_COL = "y1"
X2_COL = "x2"
Y2_COL = "y2"

REQUIRED_COLUMNS = [
    EVENT_COL,
    PHOTO_INDEX_COL,
    X1_COL,
    Y1_COL,
    X2_COL,
    Y2_COL
]


# ============================================================
# VALIDASI
# ============================================================

missing = [
    col
    for col in REQUIRED_COLUMNS
    if col not in detections.columns
]

if missing:

    raise ValueError(
        "Kolom penting tidak ditemukan: "
        + ", ".join(missing)
    )


print(
    "Kolom yang digunakan:"
)

print(
    f"event       : {EVENT_COL}"
)

print(
    f"photo index : {PHOTO_INDEX_COL}"
)

print(
    f"confidence  : {CONFIDENCE_COL}"
)

print(
    f"x1          : {X1_COL}"
)

print(
    f"y1          : {Y1_COL}"
)

print(
    f"x2          : {X2_COL}"
)

print(
    f"y2          : {Y2_COL}"
)

print()


# ============================================================
# KONVERSI DATA NUMERIK
# ============================================================

for col in [
    PHOTO_INDEX_COL,
    X1_COL,
    Y1_COL,
    X2_COL,
    Y2_COL
]:

    detections[col] = pd.to_numeric(
        detections[col],
        errors="coerce"
    )


if CONFIDENCE_COL in detections.columns:

    detections[
        CONFIDENCE_COL
    ] = pd.to_numeric(
        detections[
            CONFIDENCE_COL
        ],
        errors="coerce"
    )

else:

    detections[
        "_confidence"
    ] = 1.0

    CONFIDENCE_COL = "_confidence"


# ============================================================
# HANYA DETEKSI VALID
# ============================================================

detections = detections.dropna(
    subset=[
        PHOTO_INDEX_COL,
        X1_COL,
        Y1_COL,
        X2_COL,
        Y2_COL
    ]
)


# ============================================================
# HANYA DETEKSI YANG BENAR-BENAR TERDETEKSI
# ============================================================

if "detected" in detections.columns:

    detected_text = (
        detections["detected"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    detections = detections[
        detected_text.isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    ]


# ============================================================
# JIKA ADA LEBIH DARI SATU DETEKSI
# DALAM FOTO, AMBIL CONFIDENCE TERTINGGI
# ============================================================

detections = (
    detections
    .sort_values(
        CONFIDENCE_COL,
        ascending=False
    )
    .drop_duplicates(
        subset=[
            EVENT_COL,
            PHOTO_INDEX_COL
        ],
        keep="first"
    )
)


# ============================================================
# FUNGSI BOX
# ============================================================

def get_box(row):

    x1 = float(
        row[X1_COL]
    )

    y1 = float(
        row[Y1_COL]
    )

    x2 = float(
        row[X2_COL]
    )

    y2 = float(
        row[Y2_COL]
    )

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    area = (
        width
        *
        height
    )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "area": area
    }


# ============================================================
# BUAT INDEX
# ============================================================

detection_lookup = {}

for _, row in detections.iterrows():

    event_id = str(
        row[EVENT_COL]
    ).strip()

    photo_index = int(
        row[PHOTO_INDEX_COL]
    )

    detection_lookup[
        (
            event_id,
            photo_index
        )
    ] = get_box(row)


# ============================================================
# ANALISIS EVENT
# ============================================================

results = []

total_event = len(master)

event_terdeteksi = 0

event_lengkap = 0

event_tidak_lengkap = 0


for _, row in master.iterrows():

    event_id = str(
        row["event_id"]
    ).strip()

    boxes = []

    for photo_index in [
        1,
        2,
        3
    ]:

        box = detection_lookup.get(
            (
                event_id,
                photo_index
            )
        )

        boxes.append(box)


    # --------------------------------------------------------
    # JUMLAH DETEKSI
    # --------------------------------------------------------

    jumlah_deteksi = sum(
        box is not None
        for box in boxes
    )

    if jumlah_deteksi > 0:

        event_terdeteksi += 1

    if jumlah_deteksi == 3:

        event_lengkap += 1

    else:

        event_tidak_lengkap += 1


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    movement = "tidak_yakin"

    alasan = (
        "Tidak semua foto memiliki deteksi."
    )

    dx_13 = ""

    dy_13 = ""

    perubahan_area = ""

    x1_norm = ""

    x2_norm = ""

    x3_norm = ""

    y1_norm = ""

    y2_norm = ""

    y3_norm = ""

    area1_norm = ""

    area2_norm = ""

    area3_norm = ""


    # --------------------------------------------------------
    # ANALISIS HANYA JIKA 3 FOTO TERDETEKSI
    # --------------------------------------------------------

    if jumlah_deteksi == 3:

        b1, b2, b3 = boxes


        # ----------------------------------------------------
        # NORMALISASI BERDASARKAN RENTANG GAMBAR
        # ----------------------------------------------------
        #
        # Karena detection_results.csv tidak menyimpan
        # ukuran gambar, kita gunakan rentang posisi
        # bounding box sebagai ukuran relatif.
        #
        # Ini hanya baseline.
        # ----------------------------------------------------

        max_x = max(
            b1["x2"],
            b2["x2"],
            b3["x2"],
            1
        )

        max_y = max(
            b1["y2"],
            b2["y2"],
            b3["y2"],
            1
        )


        # ----------------------------------------------------
        # POSISI NORMALISASI
        # ----------------------------------------------------

        x1_norm = (
            b1["center_x"]
            /
            max_x
        )

        x2_norm = (
            b2["center_x"]
            /
            max_x
        )

        x3_norm = (
            b3["center_x"]
            /
            max_x
        )


        y1_norm = (
            b1["center_y"]
            /
            max_y
        )

        y2_norm = (
            b2["center_y"]
            /
            max_y
        )

        y3_norm = (
            b3["center_y"]
            /
            max_y
        )


        # ----------------------------------------------------
        # PERUBAHAN POSISI
        # ----------------------------------------------------

        dx_13 = (
            x3_norm
            -
            x1_norm
        )

        dy_13 = (
            y3_norm
            -
            y1_norm
        )


        # ----------------------------------------------------
        # UKURAN OBJEK
        # ----------------------------------------------------

        max_area = max(
            b1["area"],
            b2["area"],
            b3["area"],
            1
        )

        area1_norm = (
            b1["area"]
            /
            max_area
        )

        area2_norm = (
            b2["area"]
            /
            max_area
        )

        area3_norm = (
            b3["area"]
            /
            max_area
        )

        perubahan_area = (
            area3_norm
            -
            area1_norm
        )


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        X_THRESHOLD = 0.05

        Y_THRESHOLD = 0.05

        AREA_THRESHOLD = 0.20


        # ----------------------------------------------------
        # HORIZONTAL
        # ----------------------------------------------------

        if dx_13 > X_THRESHOLD:

            movement = "kanan"

            alasan = (
                "Titik tengah bounding box "
                "bergeser ke kanan."
            )

        elif dx_13 < -X_THRESHOLD:

            movement = "kiri"

            alasan = (
                "Titik tengah bounding box "
                "bergeser ke kiri."
            )


        # ----------------------------------------------------
        # VERTIKAL
        # ----------------------------------------------------

        elif dy_13 < -Y_THRESHOLD:

            movement = "mendekat"

            alasan = (
                "Posisi vertikal berubah "
                "ke arah atas; kandidat "
                "mendekat berdasarkan baseline."
            )

        elif dy_13 > Y_THRESHOLD:

            movement = "menjauh"

            alasan = (
                "Posisi vertikal berubah "
                "ke arah bawah; kandidat "
                "menjauh berdasarkan baseline."
            )


        # ----------------------------------------------------
        # UKURAN
        # ----------------------------------------------------

        elif perubahan_area > AREA_THRESHOLD:

            movement = "mendekat"

            alasan = (
                "Ukuran bounding box "
                "meningkat secara relatif."
            )

        elif perubahan_area < -AREA_THRESHOLD:

            movement = "menjauh"

            alasan = (
                "Ukuran bounding box "
                "menurun secara relatif."
            )

        else:

            movement = "tidak_yakin"

            alasan = (
                "Perubahan posisi dan ukuran "
                "belum cukup besar."
            )


    # --------------------------------------------------------
    # SIMPAN HASIL
    # --------------------------------------------------------

    results.append({

        "event_id": event_id,

        "timestamp": row["timestamp"],

        "jumlah_foto_terdeteksi": jumlah_deteksi,

        "x_center_foto_1_norm": x1_norm,

        "x_center_foto_2_norm": x2_norm,

        "x_center_foto_3_norm": x3_norm,

        "y_center_foto_1_norm": y1_norm,

        "y_center_foto_2_norm": y2_norm,

        "y_center_foto_3_norm": y3_norm,

        "perubahan_x_foto_1_ke_3": dx_13,

        "perubahan_y_foto_1_ke_3": dy_13,

        "ukuran_foto_1_norm": area1_norm,

        "ukuran_foto_2_norm": area2_norm,

        "ukuran_foto_3_norm": area3_norm,

        "perubahan_ukuran": perubahan_area,

        "movement_candidate": movement,

        "alasan": alasan
    })


# ============================================================
# OUTPUT CSV
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

counts = (
    result_df[
        "movement_candidate"
    ]
    .value_counts()
)


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "AUTO MOVEMENT ANALYSIS\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )

    f.write(
        f"Total event: {total_event}\n"
    )

    f.write(
        "Event dengan minimal 1 deteksi: "
        f"{event_terdeteksi}\n"
    )

    f.write(
        "Event dengan 3/3 foto terdeteksi: "
        f"{event_lengkap}\n"
    )

    f.write(
        "Event dengan deteksi tidak lengkap: "
        f"{event_tidak_lengkap}\n\n"
    )

    f.write(
        "DISTRIBUSI MOVEMENT CANDIDATE\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    for label, jumlah in counts.items():

        persen = (
            jumlah
            /
            total_event
            *
            100
        )

        f.write(
            f"{label:15s}: "
            f"{jumlah:3d} event "
            f"({persen:6.2f}%)\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "CATATAN:\n"
    )

    f.write(
        "Hasil movement merupakan candidate movement "
        "berbasis bounding box YOLO.\n"
    )

    f.write(
        "Hasil belum dianggap sebagai ground truth.\n"
    )

    f.write(
        "Validasi visual tetap diperlukan.\n"
    )


# ============================================================
# TAMPILKAN
# ============================================================

print(
    "Event dengan minimal 1 deteksi : "
    f"{event_terdeteksi}/{total_event}"
)

print(
    "Event dengan 3/3 deteksi        : "
    f"{event_lengkap}/{total_event}"
)

print(
    "Event dengan deteksi tidak lengkap : "
    f"{event_tidak_lengkap}/{total_event}"
)

print()

print(
    "Distribusi candidate movement:"
)

print()

for label, jumlah in counts.items():

    persen = (
        jumlah
        /
        total_event
        *
        100
    )

    print(
        f"{label:15s}: "
        f"{jumlah:3d} event "
        f"({persen:6.2f}%)"
    )

print()

print(
    "Output CSV:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "Output summary:"
)

print(
    OUTPUT_SUMMARY
)

print()

print("=" * 70)
print("SELESAI")
print("=" * 70)