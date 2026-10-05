from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "05_REID" / "crop"
    / "cross_time"
    / "cross_time_top_pairs.csv"
)

OUTPUT_CSV = BASE_DIR / "reid_manual_labels.csv"

TOP_N = 10


# ============================================================
# MANUAL VISUAL LABELS
# ============================================================

MANUAL_LABELS = {
    1: (
        "SAME_CANDIDATE",
        "Pola spot, bentuk tubuh, kepala, dan ekor terlihat konsisten pada beberapa pose."
    ),
    2: (
        "SAME_CANDIDATE",
        "Pola tubuh terlihat cukup konsisten, tetapi EVT_0024 lebih samping dan sebagian detail kurang jelas."
    ),
    3: (
        "SAME_CANDIDATE",
        "Beberapa ciri visual tampak sangat konsisten dan kedua event memberikan tampilan tubuh yang cukup lengkap."
    ),
    4: (
        "UNCERTAIN",
        "Similarity tinggi, tetapi EVT_0060 memiliki bagian tubuh yang kurang terlihat jelas."
    ),
    5: (
        "UNCERTAIN",
        "Pola tubuh tampak mirip, tetapi pose dan kondisi gambar berbeda cukup besar."
    ),
    6: (
        "SAME_CANDIDATE",
        "Pola spot dan proporsi tubuh terlihat cukup konsisten."
    ),
    7: (
        "SAME_CANDIDATE",
        "Pola badan dan karakteristik kepala/tubuh tampak konsisten pada beberapa foto."
    ),
    8: (
        "SAME_CANDIDATE",
        "Pola tubuh terlihat konsisten meskipun sudut pengambilan berbeda."
    ),
    9: (
        "UNCERTAIN",
        "Ada kemiripan pola, tetapi bukti visual individu belum cukup kuat."
    ),
    10: (
        "SAME_CANDIDATE",
        "Pola tubuh dan bentuk umum terlihat cukup konsisten pada beberapa foto."
    ),
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD RE-ID MANUAL LABELS")
    print("=" * 70)

    print()
    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File candidate tidak ditemukan:\n{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # Load candidate pairs
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    print(f"Total candidate tersedia : {len(df)}")
    print(f"Candidate digunakan      : Top {TOP_N}")
    print()

    df = df.head(TOP_N)

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    rows = []

    for rank, (_, pair) in enumerate(
        df.iterrows(),
        start=1
    ):

        label, notes = MANUAL_LABELS[rank]

        rows.append({
            "rank": rank,
            "event_1": pair["event_1"],
            "event_2": pair["event_2"],
            "similarity": float(pair["similarity"]),
            "time_gap_hours": float(pair["time_gap_hours"]),
            "time_gap_days": float(pair["time_gap_days"]),
            "manual_label": label,
            "notes": notes,
        })

    result = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("MANUAL LABEL SUMMARY")
    print("=" * 70)

    print()
    print(result["manual_label"].value_counts())

    print()
    print("=" * 70)
    print("DETAIL")
    print("=" * 70)
    print()

    for _, row in result.iterrows():

        print(
            f"#{int(row['rank']):02d} "
            f"{row['event_1']} <-> {row['event_2']} | "
            f"Similarity={row['similarity']:.4f} | "
            f"Gap={row['time_gap_days']:.2f} hari | "
            f"{row['manual_label']}"
        )

    print()
    print("=" * 70)
    print("SELESAI")
    print("=" * 70)

    print()
    print("File tersimpan:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()