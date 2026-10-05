from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "reid_manual_labels.csv"

OUTPUT_CSV = BASE_DIR / "individual_candidate_labels.csv"

# Hanya hubungan yang dianggap cukup kuat secara visual
VALID_LABEL = "SAME_CANDIDATE"


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):

        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):

        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a

        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD INDIVIDUAL CANDIDATE LABELS")
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
            f"File tidak ditemukan:\n{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # Load manual labels
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_CSV)

    print(f"Total manual candidate : {len(df)}")

    # --------------------------------------------------------
    # Filter SAME_CANDIDATE
    # --------------------------------------------------------

    same_df = df[
        df["manual_label"] == VALID_LABEL
    ].copy()

    print(
        f"Candidate SAME_CANDIDATE : {len(same_df)}"
    )

    print(
        f"Candidate UNCERTAIN      : "
        f"{(df['manual_label'] == 'UNCERTAIN').sum()}"
    )

    print()

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    uf = UnionFind()

    for _, row in same_df.iterrows():

        event_1 = str(row["event_1"])
        event_2 = str(row["event_2"])

        uf.union(
            event_1,
            event_2
        )

    # --------------------------------------------------------
    # Find groups
    # --------------------------------------------------------

    groups = {}

    for event in uf.parent:

        root = uf.find(event)

        if root not in groups:
            groups[root] = []

        groups[root].append(event)

    # Sort groups by size
    groups = sorted(
        groups.values(),
        key=lambda x: (-len(x), x)
    )

    # --------------------------------------------------------
    # Assign candidate individual IDs
    # --------------------------------------------------------

    event_to_individual = {}

    for idx, events in enumerate(
        groups,
        start=1
    ):

        individual_id = (
            f"IND_CAND_{idx:02d}"
        )

        for event in events:

            event_to_individual[event] = (
                individual_id
            )

    # --------------------------------------------------------
    # Create output table
    # --------------------------------------------------------

    rows = []

    for event in sorted(
        event_to_individual.keys()
    ):

        rows.append({
            "event_id": event,
            "individual_candidate": (
                event_to_individual[event]
            ),
            "group_size": len(
                next(
                    group
                    for group in groups
                    if event in group
                )
            ),
            "label_basis": "SAME_CANDIDATE",
        })

    result = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save
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
    print("INDIVIDUAL CANDIDATE GROUPS")
    print("=" * 70)

    print()

    for idx, events in enumerate(
        groups,
        start=1
    ):

        individual_id = (
            f"IND_CAND_{idx:02d}"
        )

        print(
            f"{individual_id} "
            f"({len(events)} event)"
        )

        for event in sorted(events):

            print(
                f"   - {event}"
            )

        print()

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Jumlah individual candidate : {len(groups)}"
    )

    print(
        f"Jumlah event yang tergabung : "
        f"{len(event_to_individual)}"
    )

    print(
        f"Jumlah event belum tergabung: "
        f"{63 - len(event_to_individual)}"
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