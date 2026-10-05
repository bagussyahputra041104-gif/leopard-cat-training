from pathlib import Path
import pandas as pd
from itertools import combinations


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MANUAL_LABELS_CSV = BASE_DIR / "reid_manual_labels.csv"
INDIVIDUAL_CSV = BASE_DIR / "individual_candidate_labels.csv"
ALL_PAIRS_CSV = (
    BASE_DIR
    / "05_REID" / "crop"
    / "cross_time"
    / "cross_time_pairs.csv"
)

OUTPUT_CSV = BASE_DIR / "individual_candidate_audit.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIT INDIVIDUAL CANDIDATE")
    print("=" * 70)

    print()
    print(f"Manual labels : {MANUAL_LABELS_CSV}")
    print(f"Groups        : {INDIVIDUAL_CSV}")
    print(f"All pairs     : {ALL_PAIRS_CSV}")
    print(f"Output        : {OUTPUT_CSV}")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        MANUAL_LABELS_CSV,
        INDIVIDUAL_CSV,
        ALL_PAIRS_CSV,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan:\n{path}"
            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    manual_df = pd.read_csv(
        MANUAL_LABELS_CSV
    )

    individual_df = pd.read_csv(
        INDIVIDUAL_CSV
    )

    pairs_df = pd.read_csv(
        ALL_PAIRS_CSV
    )

    # --------------------------------------------------------
    # Group events
    # --------------------------------------------------------

    grouped = (
        individual_df
        .groupby("individual_candidate")["event_id"]
        .apply(list)
        .to_dict()
    )

    print(
        f"Jumlah candidate individual : "
        f"{len(grouped)}"
    )

    print()

    audit_rows = []

    # --------------------------------------------------------
    # Audit each group
    # --------------------------------------------------------

    for individual_id, events in grouped.items():

        print("=" * 70)
        print(individual_id)
        print("=" * 70)

        print()
        print(
            f"Jumlah event dalam group : {len(events)}"
        )

        print(
            "Event:",
            ", ".join(sorted(events))
        )

        print()

        # ----------------------------------------------------
        # Generate every possible pair
        # ----------------------------------------------------

        possible_pairs = list(
            combinations(
                sorted(events),
                2
            )
        )

        print(
            f"Total pasangan dalam group : "
            f"{len(possible_pairs)}"
        )

        print()

        for event_1, event_2 in possible_pairs:

            # ----------------------------------------------
            # Normalize pair order
            # ----------------------------------------------

            a = min(
                event_1,
                event_2
            )

            b = max(
                event_1,
                event_2
            )

            # ----------------------------------------------
            # Search in manual labels
            # ----------------------------------------------

            manual_match = manual_df[
                (
                    (
                        manual_df["event_1"] == a
                    )
                    &
                    (
                        manual_df["event_2"] == b
                    )
                )
                |
                (
                    (
                        manual_df["event_1"] == b
                    )
                    &
                    (
                        manual_df["event_2"] == a
                    )
                )
            ]

            # ----------------------------------------------
            # Search in all pair similarities
            # ----------------------------------------------

            pair_match = pairs_df[
                (
                    (
                        pairs_df["event_1"] == a
                    )
                    &
                    (
                        pairs_df["event_2"] == b
                    )
                )
                |
                (
                    (
                        pairs_df["event_1"] == b
                    )
                    &
                    (
                        pairs_df["event_2"] == a
                    )
                )
            ]

            # ----------------------------------------------
            # Determine evidence
            # ----------------------------------------------

            if not manual_match.empty:

                manual_label = str(
                    manual_match.iloc[0]["manual_label"]
                )

                similarity = float(
                    manual_match.iloc[0]["similarity"]
                )

                gap_days = float(
                    manual_match.iloc[0]["time_gap_days"]
                )

                evidence = (
                    "MANUAL_LABEL"
                )

            elif not pair_match.empty:

                manual_label = ""

                similarity = float(
                    pair_match.iloc[0]["similarity"]
                )

                gap_days = float(
                    pair_match.iloc[0]["time_gap_days"]
                )

                evidence = (
                    "SIMILARITY_ONLY"
                )

            else:

                manual_label = ""

                similarity = None

                gap_days = None

                evidence = (
                    "NO_DIRECT_EVIDENCE"
                )

            # ----------------------------------------------
            # Print
            # ----------------------------------------------

            if similarity is not None:

                similarity_text = (
                    f"{similarity:.4f}"
                )

            else:

                similarity_text = "-"

            if gap_days is not None:

                gap_text = (
                    f"{gap_days:.2f}"
                )

            else:

                gap_text = "-"

            print(
                f"{event_1} <-> {event_2} | "
                f"similarity={similarity_text} | "
                f"gap={gap_text} hari | "
                f"{evidence}"
            )

            # ----------------------------------------------
            # Save row
            # ----------------------------------------------

            audit_rows.append({

                "individual_candidate":
                    individual_id,

                "event_1":
                    event_1,

                "event_2":
                    event_2,

                "similarity":
                    similarity,

                "time_gap_days":
                    gap_days,

                "manual_label":
                    manual_label,

                "evidence":
                    evidence,
            })

        print()

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print()

    print(
        audit_df["evidence"].value_counts()
    )

    print()

    print(
        "Total pasangan yang diperiksa :",
        len(audit_df)
    )

    print(
        "Pasangan dengan label manual  :",
        (
            audit_df["evidence"]
            == "MANUAL_LABEL"
        ).sum()
    )

    print(
        "Pasangan similarity only      :",
        (
            audit_df["evidence"]
            == "SIMILARITY_ONLY"
        ).sum()
    )

    print(
        "Pasangan tanpa evidence       :",
        (
            audit_df["evidence"]
            == "NO_DIRECT_EVIDENCE"
        ).sum()
    )

    print()

    print("=" * 70)
    print("SELESAI")
    print("=" * 70)

    print()
    print("File audit:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()