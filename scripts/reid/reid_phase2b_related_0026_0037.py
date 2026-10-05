from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang")

INPUT = (
    ROOT
    / "05_REID" / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "cross_time_pairs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "05_REID" / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "phase2b_0026_0037"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EVENTS = {"EVT_0026", "EVT_0037"}

df = pd.read_csv(INPUT)

# Ambil semua pasangan yang berhubungan dengan EVT_0026 atau EVT_0037
related = df[
    df["event_1"].isin(TARGET_EVENTS)
    | df["event_2"].isin(TARGET_EVENTS)
].copy()

# Jangan masukkan pasangan 0026 <-> 0037 sendiri
related = related[
    ~(
        related["event_1"].isin(TARGET_EVENTS)
        & related["event_2"].isin(TARGET_EVENTS)
    )
].copy()

# Urutkan similarity tertinggi
related = related.sort_values(
    "similarity",
    ascending=False
).reset_index(drop=True)

# Hindari pasangan yang sudah kita review sebelumnya
reviewed_pairs = {
    tuple(sorted(("EVT_0026", "EVT_0037"))),
    tuple(sorted(("EVT_0012", "EVT_0026"))),
    tuple(sorted(("EVT_0012", "EVT_0037"))),
    tuple(sorted(("EVT_0005", "EVT_0026"))),
    tuple(sorted(("EVT_0005", "EVT_0037"))),
}

def pair_key(row):
    return tuple(sorted((row["event_1"], row["event_2"])))

related = related[
    ~related.apply(pair_key, axis=1).isin(reviewed_pairs)
].copy()

# Ambil maksimal 8 kandidat
top = related.head(8).copy()

output = OUTPUT_DIR / "phase2b_related_candidates.csv"
top.to_csv(output, index=False)

print("=" * 70)
print("RE-ID PHASE 2B — RELATED TO EVT_0026 & EVT_0037")
print("=" * 70)

print(f"Jumlah kandidat ditemukan : {len(related)}")
print(f"Top kandidat disimpan     : {len(top)}")
print()

for i, row in enumerate(top.itertuples(index=False), start=1):
    print(
        f"[{i:02d}] "
        f"{row.event_1} <-> {row.event_2} | "
        f"Similarity: {row.similarity:.4f} | "
        f"Gap: {row.time_gap_days:.2f} hari"
    )

print()
print("=" * 70)
print("SELESAI")
print("=" * 70)
print()
print("Output:")
print(output)