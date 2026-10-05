from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_CSV = BASE_DIR / "leopard_cat_master.csv"
MANUAL_CSV = BASE_DIR / "reid_manual_labels.csv"

OUTPUT_CSV = BASE_DIR / "individual_candidate_final.csv"
PAIR_OUTPUT_CSV = BASE_DIR / "individual_candidate_final_pairs.csv"


# ============================================================
# HASIL REVIEW MANUAL TAMBAHAN
#
# Berdasarkan 9 cluster audit yang sudah direview.
# ============================================================

ADDITIONAL_REVIEW = [
    # event_1, event_2, similarity, gap_days, label

    (
        "EVT_0013",
        "EVT_0024",
        0.8896,
        9.76,
        "SAME_CANDIDATE"
    ),

    (
        "EVT_0013",
        "EVT_0025",
        0.8903,
        9.76,
        "SAME_CANDIDATE"
    ),

    (
        "EVT_0014",
        "EVT_0024",
        0.8412,
        9.76,
        "UNCERTAIN"
    ),

    (
        "EVT_0014",
        "EVT_0025",
        0.8444,
        9.76,
        "UNCERTAIN"
    ),

    (
        "EVT_0014",
        "EVT_0058",
        0.8761,
        42.00,
        "SAME_CANDIDATE"
    ),

    (
        "EVT_0015",
        "EVT_0024",
        0.8516,
        9.76,
        "UNCERTAIN"
    ),

    (
        "EVT_0015",
        "EVT_0025",
        0.8690,
        9.76,
        "SAME_CANDIDATE"
    ),

    (
        "EVT_0024",
        "EVT_0057",
        0.8579,
        32.24,
        "SAME_CANDIDATE"
    ),

    (
        "EVT_0025",
        "EVT_0057",
        0.8592,
        32.24,
        "SAME_CANDIDATE"
    ),
]


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item
            for item in items
        }

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINALIZE INDIVIDUAL CANDIDATES")
    print("=" * 70)

    print()
    print(f"Master CSV : {MASTER_CSV}")
    print(f"Manual CSV : {MANUAL_CSV}")
    print()

    # --------------------------------------------------------
    # LOAD MASTER
    # --------------------------------------------------------

    master_df = pd.read_csv(
        MASTER_CSV,
        dtype=str,
        keep_default_na=False
    )

    all_events = list(
        master_df["event_id"]
    )

    print(
        f"Total leopard-cat event : "
        f"{len(all_events)}"
    )

    # --------------------------------------------------------
    # LOAD MANUAL LABELS SEBELUMNYA
    # --------------------------------------------------------

    manual_df = pd.read_csv(
        MANUAL_CSV,
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Manual label sebelumnya : "
        f"{len(manual_df)} pasangan"
    )

    # --------------------------------------------------------
    # NORMALIZE OLD MANUAL PAIRS
    # --------------------------------------------------------

    pairs = []

    for _, row in manual_df.iterrows():

        event_1 = str(
            row["event_1"]
        )

        event_2 = str(
            row["event_2"]
        )

        similarity = row.get(
            "similarity",
            ""
        )

        gap_days = row.get(
            "time_gap_days",
            ""
        )

        label = str(
            row["manual_label"]
        )

        pairs.append(
            {
                "event_1": event_1,
                "event_2": event_2,
                "similarity": similarity,
                "time_gap_days": gap_days,
                "label": label,
                "source": "INITIAL_MANUAL_REVIEW"
            }
        )

    # --------------------------------------------------------
    # ADD 9 NEW VISUAL REVIEWS
    # --------------------------------------------------------

    for (
        event_1,
        event_2,
        similarity,
        gap_days,
        label
    ) in ADDITIONAL_REVIEW:

        pairs.append(
            {
                "event_1": event_1,
                "event_2": event_2,
                "similarity": similarity,
                "time_gap_days": gap_days,
                "label": label,
                "source": "CLUSTER_VISUAL_REVIEW"
            }
        )

    pair_df = pd.DataFrame(
        pairs
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATE PAIRS
    # --------------------------------------------------------

    pair_df["pair_key"] = pair_df.apply(
        lambda row: "_".join(
            sorted(
                [
                    row["event_1"],
                    row["event_2"]
                ]
            )
        ),
        axis=1
    )

    pair_df = (
        pair_df
        .drop_duplicates(
            subset=["pair_key"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SAVE FINAL PAIR TABLE
    # --------------------------------------------------------

    pair_df[
        [
            "event_1",
            "event_2",
            "similarity",
            "time_gap_days",
            "label",
            "source"
        ]
    ].to_csv(
        PAIR_OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # BUILD CANDIDATE GROUPS
    #
    # Hanya SAME_CANDIDATE yang menggabungkan event.
    # UNCERTAIN tidak menggabungkan.
    # --------------------------------------------------------

    uf = UnionFind(
        all_events
    )

    same_pairs = pair_df[
        pair_df["label"]
        == "SAME_CANDIDATE"
    ]

    for _, row in same_pairs.iterrows():

        event_1 = row["event_1"]
        event_2 = row["event_2"]

        if (
            event_1 in uf.parent
            and event_2 in uf.parent
        ):

            uf.union(
                event_1,
                event_2
            )

    # --------------------------------------------------------
    # COLLECT GROUPS
    # --------------------------------------------------------

    groups = {}

    for event_id in all_events:

        root = uf.find(
            event_id
        )

        if root not in groups:

            groups[root] = []

        groups[root].append(
            event_id
        )

    # --------------------------------------------------------
    # ONLY GROUPS WITH >= 2 EVENTS
    # --------------------------------------------------------

    candidate_groups = [
        events
        for events in groups.values()
        if len(events) >= 2
    ]

    candidate_groups.sort(
        key=lambda x: (
            -len(x),
            x[0]
        )
    )

    # --------------------------------------------------------
    # ASSIGN INDIVIDUAL CANDIDATE ID
    # --------------------------------------------------------

    event_to_candidate = {}

    for idx, events in enumerate(
        candidate_groups,
        start=1
    ):

        candidate_id = (
            f"IND_CAND_{idx:02d}"
        )

        for event_id in events:

            event_to_candidate[
                event_id
            ] = candidate_id

    # --------------------------------------------------------
    # BUILD FINAL EVENT TABLE
    # --------------------------------------------------------

    output_rows = []

    for event_id in all_events:

        if event_id in event_to_candidate:

            candidate_id = (
                event_to_candidate[event_id]
            )

            status = (
                "GROUPED_CANDIDATE"
            )

        else:

            candidate_id = (
                "UNASSIGNED"
            )

            status = (
                "NO_CONFIRMED_CANDIDATE"
            )

        output_rows.append(
            {
                "event_id": event_id,
                "candidate_individual": candidate_id,
                "status": status
            }
        )

    output_df = pd.DataFrame(
        output_rows
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL CANDIDATE GROUPS")
    print("=" * 70)

    total_grouped_events = 0

    for idx, events in enumerate(
        candidate_groups,
        start=1
    ):

        candidate_id = (
            f"IND_CAND_{idx:02d}"
        )

        total_grouped_events += (
            len(events)
        )

        print()
        print(
            f"{candidate_id} "
            f"({len(events)} event)"
        )

        for event_id in events:

            print(
                f"   - {event_id}"
            )

    # --------------------------------------------------------
    # SUMMARY COUNTS
    # --------------------------------------------------------

    unassigned = (
        len(all_events)
        - total_grouped_events
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Jumlah candidate individual : "
        f"{len(candidate_groups)}"
    )

    print(
        f"Jumlah event tergabung      : "
        f"{total_grouped_events}"
    )

    print(
        f"Jumlah event belum tergabung: "
        f"{unassigned}"
    )

    print()
    print(
        "Catatan:"
    )

    print(
        "- SAME_CANDIDATE digunakan "
        "untuk membentuk candidate group."
    )

    print(
        "- UNCERTAIN tidak digunakan "
        "untuk menggabungkan event."
    )

    print(
        "- Candidate group bukan "
        "ground-truth identitas individu."
    )

    print()
    print(
        f"Final event table :\n{OUTPUT_CSV}"
    )

    print(
        f"Final pair table  :\n{PAIR_OUTPUT_CSV}"
    )

    print()
    print("=" * 70)
    print("SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()