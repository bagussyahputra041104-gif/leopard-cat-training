import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

FINAL_FILE = (
    BASE
    / "04_HASIL" / "individual" / "final_results"
    / "final_event_results.csv"
)

PAIR_FILE = (
    BASE
    / "05_REID" / "crop"
    / "cross_time"
    / "cross_time_pairs.csv"
)

OUTPUT_DIR = (
    BASE
    / "05_REID" / "crop"
    / "cross_time"
    / "phase2_unassigned"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "phase2_unassigned_candidates.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("RE-ID PHASE 2 — UNASSIGNED")
print("=" * 70)

final_df = pd.read_csv(
    FINAL_FILE,
    dtype=str,
    keep_default_na=False
)

pairs_df = pd.read_csv(
    PAIR_FILE,
    dtype=str,
    keep_default_na=False
)

# angka
pairs_df["similarity"] = pd.to_numeric(
    pairs_df["similarity"],
    errors="coerce"
)

pairs_df["time_gap_hours"] = pd.to_numeric(
    pairs_df["time_gap_hours"],
    errors="coerce"
)

pairs_df["time_gap_days"] = (
    pairs_df["time_gap_hours"] / 24
)


# ============================================================
# AMBIL EVENT UNASSIGNED
# ============================================================

unassigned = set(
    final_df.loc[
        final_df["reid_status"] == "UNASSIGNED",
        "event_id"
    ]
)

print(
    f"\nTotal leopard-cat event : "
    f"{len(final_df)}"
)

print(
    f"UNASSIGNED              : "
    f"{len(unassigned)}"
)


# ============================================================
# FILTER KEDUA EVENT HARUS UNASSIGNED
# DAN MINIMAL GAP 7 HARI
# ============================================================

mask_unassigned = (
    pairs_df["event_1"].isin(unassigned)
    &
    pairs_df["event_2"].isin(unassigned)
)

filtered = pairs_df[
    mask_unassigned
].copy()

filtered = filtered[
    filtered["time_gap_days"] >= 7
].copy()


# ============================================================
# BUANG DUPLIKAT PASANGAN
# ============================================================

filtered["pair_key"] = filtered.apply(
    lambda r: "_".join(
        sorted([
            r["event_1"],
            r["event_2"]
        ])
    ),
    axis=1
)

filtered = (
    filtered
    .sort_values(
        "similarity",
        ascending=False
    )
    .drop_duplicates(
        "pair_key"
    )
    .drop(
        columns=["pair_key"]
    )
)


# ============================================================
# AMBIL TOP 20
# ============================================================

top = (
    filtered
    .sort_values(
        "similarity",
        ascending=False
    )
    .head(20)
    .reset_index(drop=True)
)

top.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print(
    f"\nPasangan UNASSIGNED "
    f"setelah filter : {len(filtered)}"
)

print(
    f"Top kandidat yang disimpan : {len(top)}"
)

print("\n" + "-" * 70)

for i, row in top.iterrows():

    print(
        f"[{i+1:02d}] "
        f"{row['event_1']} <-> "
        f"{row['event_2']} | "
        f"Similarity: "
        f"{float(row['similarity']):.4f} | "
        f"Gap: "
        f"{float(row['time_gap_days']):.2f} hari"
    )


print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)