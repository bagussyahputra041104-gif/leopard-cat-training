from pathlib import Path
import pandas as pd


ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

PAIR_CSV = (
    ROOT
    / "05_REID"
    / "crop"
    / "reid_crop_baseline"
    / "cross_time"
    / "cross_time_pairs.csv"
)

FINAL_CSV = (
    ROOT
    / "04_HASIL"
    / "individual"
    / "final_results"
    / "final_event_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "05_REID"
    / "crop"
    / "reid_crop_baseline"
    / "cross_time"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "phase3_unassigned_candidates.csv"
)


# ============================================================
# PARAMETER
# ============================================================

MIN_TIME_GAP_DAYS = 7

TOP_N = 15


# ============================================================
# LOAD
# ============================================================

pairs = pd.read_csv(
    PAIR_CSV,
    dtype=str,
    keep_default_na=False
)

final_df = pd.read_csv(
    FINAL_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# AMBIL EVENT YANG MASIH UNASSIGNED
# ============================================================

unassigned = set(
    final_df.loc[
        final_df["reid_status"] == "UNASSIGNED",
        "event_id"
    ]
)


print()
print("=" * 70)
print("RE-ID PHASE 3 — UNASSIGNED")
print("=" * 70)

print(
    f"Total leopard-cat event : {len(final_df)}"
)

print(
    f"UNASSIGNED              : {len(unassigned)}"
)

print()


# ============================================================
# FILTER PASANGAN
# ============================================================

candidates = []

for _, row in pairs.iterrows():

    event_1 = row["event_1"]
    event_2 = row["event_2"]

    # Kedua event harus masih UNASSIGNED
    if event_1 not in unassigned:
        continue

    if event_2 not in unassigned:
        continue

    # Jangan pasangan event yang sama
    if event_1 == event_2:
        continue

    similarity = float(
        row["similarity"]
    )

    gap_days = float(
        row["time_gap_days"]
    )

    # Hindari event yang terlalu berdekatan
    if gap_days < MIN_TIME_GAP_DAYS:
        continue

    candidates.append(
        {
            "event_1": event_1,
            "event_2": event_2,
            "similarity": similarity,
            "time_gap_days": gap_days,
            "time_gap_hours": float(
                row["time_gap_hours"]
            ),
        }
    )


# ============================================================
# SORT
# ============================================================

candidate_df = pd.DataFrame(
    candidates
)

if candidate_df.empty:

    print(
        "Tidak ditemukan pasangan kandidat."
    )

    raise SystemExit


candidate_df = candidate_df.sort_values(
    by="similarity",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SIMPAN SEMUA
# ============================================================

candidate_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# OUTPUT TOP
# ============================================================

print(
    f"Pasangan kandidat : {len(candidate_df)}"
)

print(
    f"Top kandidat      : {min(TOP_N, len(candidate_df))}"
)

print()

for i, row in candidate_df.head(
    TOP_N
).iterrows():

    print(
        f"[{i+1:02d}] "
        f"{row['event_1']} <-> "
        f"{row['event_2']} | "
        f"Similarity: "
        f"{row['similarity']:.4f} | "
        f"Gap: "
        f"{row['time_gap_days']:.2f} hari"
    )


print()
print(
    f"Output:"
)

print(
    OUTPUT_CSV
)

print("=" * 70)