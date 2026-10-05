from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang")

CSV_PATH = (
    ROOT / "05_REID" / "crop" / "reid_crop_baseline"
    / "cross_time" / "phase2b_related_candidates.csv"
)

RESULT_PATH = (
    ROOT / "04_HASIL" / "individual" / "final_results"
    / "final_event_results.csv"
)

OUT_DIR = (
    ROOT / "05_REID" / "crop" / "reid_crop_baseline"
    / "cross_time" / "phase2b_candidate_sheets"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
results = pd.read_csv(RESULT_PATH, dtype=str, keep_default_na=False)

TOP_N = min(8, len(df))
df = df.head(TOP_N)

def get_event(event_id):
    row = results[results["event_id"] == event_id]
    if row.empty:
        raise ValueError(f"Event tidak ditemukan: {event_id}")
    return row.iloc[0]

def make_sheet(rank, event1, event2, similarity, gap_days):
    r1 = get_event(event1)
    r2 = get_event(event2)

    paths1 = [r1["foto_1"], r1["foto_2"], r1["foto_3"]]
    paths2 = [r2["foto_1"], r2["foto_2"], r2["foto_3"]]

    thumbs = []
    labels = []

    for event_id, paths in [(event1, paths1), (event2, paths2)]:
        for i, p in enumerate(paths, 1):
            path = Path(p)
            if not path.exists():
                matches = list((ROOT / "01_DATA_ASLI" / "leopard_cat").rglob(path.name))
                if not matches:
                    raise FileNotFoundError(f"Foto tidak ditemukan: {path.name}")
                path = matches[0]

            img = Image.open(path).convert("RGB")
            img = ImageOps.contain(img, (500, 350))
            thumbs.append(img)
            labels.append(f"{event_id} - Foto {i}")

    W, H = 1100, 900
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"PHASE 2B CANDIDATE {rank:02d} | "
        f"{event1} <-> {event2}"
    )
    subtitle = (
        f"Similarity: {similarity:.4f} | "
        f"Gap: {gap_days:.2f} hari"
    )

    draw.text((30, 20), title, fill="black")
    draw.text((30, 50), subtitle, fill="black")

    positions = [
        (25, 100), (560, 100),
        (25, 370), (560, 370),
        (25, 640), (560, 640)
    ]

    for img, label, (x, y) in zip(thumbs, labels, positions):
        canvas.paste(img, (x, y))
        draw.text((x, y + 220), label, fill="black")

    out = OUT_DIR / f"candidate_{rank:02d}_{event1}_{event2}.jpg"
    canvas.save(out, quality=95)
    print(f"[{rank:02d}] {event1} <-> {event2} -> {out.name}")

print("=" * 70)
print("PHASE 2B — CANDIDATE SHEETS")
print("=" * 70)
print(f"Jumlah kandidat: {len(df)}")

for rank, (_, row) in enumerate(df.iterrows(), 1):
    make_sheet(
        rank,
        row["event_1"],
        row["event_2"],
        float(row["similarity"]),
        float(row["time_gap_days"])
    )

print("=" * 70)
print("SELESAI")
print(f"Output: {OUT_DIR}")



