import os
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO


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
    "reid_crops"
)

CROP_DIR = os.path.join(
    OUTPUT_DIR,
    "crops"
)

PREVIEW_DIR = os.path.join(
    OUTPUT_DIR,
    "preview"
)

RESULT_CSV = os.path.join(
    OUTPUT_DIR,
    "detection_results.csv"
)

# YOLO COCO class "cat"
CAT_CLASS_ID = 15

# Karena kamera trap malam cukup sulit,
# kita mulai dengan confidence rendah.
CONF_THRESHOLD = 0.15

# Padding sekitar objek
PADDING_RATIO = 0.20


# =========================================================
# SETUP
# =========================================================

os.makedirs(
    CROP_DIR,
    exist_ok=True
)

os.makedirs(
    PREVIEW_DIR,
    exist_ok=True
)

print("=" * 60)
print("LEOPARD CAT OBJECT DETECTION")
print("=" * 60)

print()
print("Loading YOLO pretrained model...")

model = YOLO("yolo11n.pt")

print("Model siap.")


# =========================================================
# LOAD MASTER
# =========================================================

df = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)

print()
print(
    f"Total event : {len(df)}"
)

print(
    f"Total foto  : {len(df) * 3}"
)


# =========================================================
# HELPER
# =========================================================

def make_crop(
    image,
    box
):

    width, height = image.size

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * PADDING_RATIO
    pad_y = box_height * PADDING_RATIO

    x1 = max(
        0,
        int(x1 - pad_x)
    )

    y1 = max(
        0,
        int(y1 - pad_y)
    )

    x2 = min(
        width,
        int(x2 + pad_x)
    )

    y2 = min(
        height,
        int(y2 + pad_y)
    )

    return image.crop(
        (x1, y1, x2, y2)
    )


# =========================================================
# PROCESS
# =========================================================

results_rows = []

preview_counter = 0

for event_index, row in df.iterrows():

    event_id = row["event_id"]

    print()
    print(
        f"[EVENT {event_index + 1:02d}/{len(df)}] "
        f"{event_id}"
    )

    for photo_index in range(1, 4):

        image_path = row[
            f"foto_{photo_index}"
        ]

        if not os.path.exists(
            image_path
        ):

            print(
                f"  Foto {photo_index}: "
                f"FILE TIDAK DITEMUKAN"
            )

            results_rows.append({
                "event_id": event_id,
                "photo_index": photo_index,
                "image_path": image_path,
                "detected": False,
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "crop_path": ""
            })

            continue

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"  Foto {photo_index}: "
                f"GAGAL BUKA"
            )

            results_rows.append({
                "event_id": event_id,
                "photo_index": photo_index,
                "image_path": image_path,
                "detected": False,
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "crop_path": ""
            })

            continue


        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        prediction = model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        best_detection = None

        if prediction.boxes is not None:

            for box in prediction.boxes:

                class_id = int(
                    box.cls.item()
                )

                confidence = float(
                    box.conf.item()
                )

                if class_id != CAT_CLASS_ID:
                    continue

                if confidence < CONF_THRESHOLD:
                    continue

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                if (
                    best_detection is None
                    or confidence >
                    best_detection["confidence"]
                ):

                    best_detection = {
                        "confidence": confidence,
                        "box": coordinates
                    }


        # -------------------------------------------------
        # DETECTION FOUND
        # -------------------------------------------------

        if best_detection is not None:

            confidence = (
                best_detection["confidence"]
            )

            x1, y1, x2, y2 = (
                best_detection["box"]
            )

            crop = make_crop(
                image,
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            crop_filename = (
                f"{event_id}_"
                f"foto_{photo_index}.jpg"
            )

            crop_path = os.path.join(
                CROP_DIR,
                crop_filename
            )

            crop.save(
                crop_path,
                quality=95
            )

            print(
                f"  Foto {photo_index}: "
                f"DETECTED "
                f"conf={confidence:.3f}"
            )

            results_rows.append({
                "event_id": event_id,
                "photo_index": photo_index,
                "image_path": image_path,
                "detected": True,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": crop_path
            })


        # -------------------------------------------------
        # NO DETECTION
        # -------------------------------------------------

        else:

            print(
                f"  Foto {photo_index}: "
                f"NO CAT DETECTED"
            )

            results_rows.append({
                "event_id": event_id,
                "photo_index": photo_index,
                "image_path": image_path,
                "detected": False,
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "crop_path": ""
            })


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(
    results_rows
)

results_df.to_csv(
    RESULT_CSV,
    index=False
)


# =========================================================
# STATISTICS
# =========================================================

total_photos = len(
    results_df
)

detected_photos = int(
    results_df["detected"].sum()
)

not_detected = (
    total_photos -
    detected_photos
)

print()
print("=" * 60)
print("HASIL DETEKSI")
print("=" * 60)

print(
    f"Total foto       : {total_photos}"
)

print(
    f"Berhasil detect  : {detected_photos}"
)

print(
    f"Tidak terdeteksi : {not_detected}"
)

if total_photos > 0:

    percentage = (
        detected_photos /
        total_photos *
        100
    )

    print(
        f"Detection rate   : "
        f"{percentage:.2f}%"
    )


# =========================================================
# EVENT STATISTICS
# =========================================================

event_detection = (
    results_df
    .groupby("event_id")["detected"]
    .sum()
)

events_with_detection = int(
    (
        event_detection > 0
    ).sum()
)

events_without_detection = (
    len(event_detection) -
    events_with_detection
)

print()
print(
    f"Event dengan >=1 detection : "
    f"{events_with_detection}"
)

print(
    f"Event tanpa detection       : "
    f"{events_without_detection}"
)


# =========================================================
# PREVIEW CONTACT SHEET
# =========================================================

print()
print("=" * 60)
print("MEMBUAT PREVIEW CROP")
print("=" * 60)


detected_df = results_df[
    results_df["detected"] == True
].copy()


# Ambil maksimal 60 crop untuk preview
preview_df = detected_df.head(
    60
)


for index, item in preview_df.iterrows():

    crop_path = item["crop_path"]

    if not os.path.exists(
        crop_path
    ):
        continue

    try:

        image = Image.open(
            crop_path
        ).convert("RGB")

        image.thumbnail(
            (300, 220)
        )

        canvas = Image.new(
            "RGB",
            (320, 260),
            "white"
        )

        x = (
            320 -
            image.width
        ) // 2

        y = 25

        canvas.paste(
            image,
            (x, y)
        )

        draw = ImageDraw.Draw(
            canvas
        )

        text = (
            f"{item['event_id']} | "
            f"foto {item['photo_index']} | "
            f"conf={float(item['confidence']):.2f}"
        )

        draw.text(
            (5, 5),
            text,
            fill="black"
        )

        output_path = os.path.join(
            PREVIEW_DIR,
            f"preview_{preview_counter:03d}.jpg"
        )

        canvas.save(
            output_path,
            quality=90
        )

        preview_counter += 1

    except Exception:
        pass


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print()
print("Detection CSV:")

print(
    RESULT_CSV
)

print()
print("Crop directory:")

print(
    CROP_DIR
)

print()
print("Preview directory:")

print(
    PREVIEW_DIR
)

print()
print(
    "PENTING: detection YOLO ini masih berupa "
    "candidate object detection."
)

print(
    "Belum berarti setiap crop pasti leopard cat."
)