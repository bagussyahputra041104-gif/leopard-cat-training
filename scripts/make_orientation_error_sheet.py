import pandas as pd
from PIL import Image, ImageDraw, ImageOps

BASE = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

CSV_PATH = (
    BASE
    + r"\04_HASIL\orientation\evaluation_orientation_resnet18"
    + r"\orientation_test_predictions.csv"
)

OUTPUT_PATH = (
    BASE
    + r"\04_HASIL\orientation\evaluation_orientation_resnet18"
    + r"\error_analysis_12_wrong.jpg"
)

df = pd.read_csv(CSV_PATH, keep_default_na=False)

wrong = df[df["correct"] == False].copy().reset_index(drop=True)

print("Jumlah foto salah:", len(wrong))
print()

CELL_W = 500
CELL_H = 380

COLS = 3
ROWS = (len(wrong) + COLS - 1) // COLS

sheet = Image.new(
    "RGB",
    (CELL_W * COLS, CELL_H * ROWS),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, row in wrong.iterrows():

    path = row["path_foto"]

    image = Image.open(path).convert("RGB")

    image = ImageOps.contain(
        image,
        (CELL_W - 20, 290)
    )

    x = (i % COLS) * CELL_W
    y = (i // COLS) * CELL_H

    image_x = x + (CELL_W - image.width) // 2
    image_y = y + 10

    sheet.paste(image, (image_x, image_y))

    text_y = y + 315

    text1 = f"{row['event_id']} - {row['foto']}"

    text2 = (
        f"GT: {row['ground_truth']} | "
        f"AI: {row['prediction']}"
    )

    text3 = f"Confidence: {float(row['confidence']):.2%}"

    draw.text(
        (x + 10, text_y),
        text1,
        fill="black"
    )

    draw.text(
        (x + 10, text_y + 20),
        text2,
        fill="black"
    )

    draw.text(
        (x + 10, text_y + 40),
        text3,
        fill="black"
    )

sheet.save(
    OUTPUT_PATH,
    quality=95
)

print("Contact sheet berhasil dibuat!")
print()
print("Lokasi:")
print(OUTPUT_PATH)

sheet.show()
