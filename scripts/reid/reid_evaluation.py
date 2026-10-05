from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_CSV = BASE_DIR / "leopard_cat_master.csv"

FINAL_EVENT_CSV = (
    BASE_DIR / "individual_candidate_final.csv"
)

FINAL_PAIR_CSV = (
    BASE_DIR / "individual_candidate_final_pairs.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "05_REID" / "evaluation"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "reid_summary.csv"
)

GROUPS_CSV = (
    OUTPUT_DIR / "candidate_groups.csv"
)

RECURRENCE_CSV = (
    OUTPUT_DIR / "candidate_recurrence.csv"
)

REPORT_TXT = (
    OUTPUT_DIR / "reid_evaluation_report.txt"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RE-ID EVALUATION")
    print("=" * 70)

    print()
    print(f"Master      : {MASTER_CSV}")
    print(f"Event table : {FINAL_EVENT_CSV}")
    print(f"Pair table  : {FINAL_PAIR_CSV}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    master_df = pd.read_csv(
        MASTER_CSV,
        dtype=str,
        keep_default_na=False
    )

    event_df = pd.read_csv(
        FINAL_EVENT_CSV,
        dtype=str,
        keep_default_na=False
    )

    pair_df = pd.read_csv(
        FINAL_PAIR_CSV,
        dtype=str,
        keep_default_na=False
    )

    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    total_events = len(
        master_df
    )

    grouped_df = event_df[
        event_df[
            "status"
        ] == "GROUPED_CANDIDATE"
    ].copy()

    unassigned_df = event_df[
        event_df[
            "status"
        ] == "NO_CONFIRMED_CANDIDATE"
    ].copy()

    candidate_count = (
        grouped_df[
            "candidate_individual"
        ]
        .nunique()
    )

    grouped_event_count = (
        len(grouped_df)
    )

    unassigned_event_count = (
        len(unassigned_df)
    )

    # --------------------------------------------------------
    # PAIR COUNTS
    # --------------------------------------------------------

    same_df = pair_df[
        pair_df["label"]
        == "SAME_CANDIDATE"
    ].copy()

    uncertain_df = pair_df[
        pair_df["label"]
        == "UNCERTAIN"
    ].copy()

    different_df = pair_df[
        pair_df["label"]
        == "DIFFERENT_CANDIDATE"
    ].copy()

    same_count = len(
        same_df
    )

    uncertain_count = len(
        uncertain_df
    )

    different_count = len(
        different_df
    )

    # --------------------------------------------------------
    # CROSS-TIME SAME CANDIDATE PAIRS
    # --------------------------------------------------------

    cross_time_same = same_df[
        pd.to_numeric(
            same_df[
                "time_gap_days"
            ],
            errors="coerce"
        ) > 1
    ].copy()

    cross_time_same_count = len(
        cross_time_same
    )

    # --------------------------------------------------------
    # BUILD CANDIDATE GROUP TABLE
    # --------------------------------------------------------

    group_rows = []

    for candidate_id, group in grouped_df.groupby(
        "candidate_individual"
    ):

        events = sorted(
            group["event_id"].tolist()
        )

        group_rows.append(
            {
                "candidate_individual": candidate_id,
                "event_count": len(events),
                "events": ", ".join(events)
            }
        )

    groups_df = pd.DataFrame(
        group_rows
    )

    if len(groups_df) > 0:

        groups_df = groups_df.sort_values(
            "event_count",
            ascending=False
        )

    groups_df.to_csv(
        GROUPS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # CANDIDATE RECURRENCE
    # --------------------------------------------------------

    master_lookup = master_df[
        [
            "event_id",
            "timestamp",
            "source_folder"
        ]
    ].copy()

    master_lookup[
        "timestamp"
    ] = pd.to_datetime(
        master_lookup["timestamp"],
        errors="coerce"
    )

    event_lookup = event_df.merge(
        master_lookup,
        on="event_id",
        how="left"
    )

    recurrence_rows = []

    for candidate_id, group in event_lookup[
        event_lookup[
            "status"
        ] == "GROUPED_CANDIDATE"
    ].groupby(
        "candidate_individual"
    ):

        group = group.sort_values(
            "timestamp"
        )

        timestamps = group[
            "timestamp"
        ].dropna()

        if len(timestamps) >= 2:

            first_seen = (
                timestamps.min()
            )

            last_seen = (
                timestamps.max()
            )

            span_days = (
                last_seen
                - first_seen
            ).total_seconds() / 86400

        else:

            first_seen = (
                timestamps.iloc[0]
                if len(timestamps) == 1
                else pd.NaT
            )

            last_seen = first_seen

            span_days = 0

        recurrence_rows.append(
            {
                "candidate_individual": candidate_id,
                "event_count": len(group),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "span_days": round(
                    span_days,
                    3
                ),
                "source_count": group[
                    "source_folder"
                ].nunique()
            }
        )

    recurrence_df = pd.DataFrame(
        recurrence_rows
    )

    recurrence_df.to_csv(
        RECURRENCE_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    summary_rows = [

        {
            "metric":
                "Total leopard-cat events",
            "value":
                total_events
        },

        {
            "metric":
                "Events in candidate groups",
            "value":
                grouped_event_count
        },

        {
            "metric":
                "Events without candidate group",
            "value":
                unassigned_event_count
        },

        {
            "metric":
                "Candidate individual groups",
            "value":
                candidate_count
        },

        {
            "metric":
                "Manual SAME_CANDIDATE pairs",
            "value":
                same_count
        },

        {
            "metric":
                "Manual UNCERTAIN pairs",
            "value":
                uncertain_count
        },

        {
            "metric":
                "Manual DIFFERENT_CANDIDATE pairs",
            "value":
                different_count
        },

        {
            "metric":
                "SAME_CANDIDATE pairs >1 day",
            "value":
                cross_time_same_count
        }
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "RE-ID EVALUATION REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        f"Total leopard-cat events : "
        f"{total_events}"
    )

    report_lines.append(
        f"Candidate individual groups : "
        f"{candidate_count}"
    )

    report_lines.append(
        f"Events in candidate groups : "
        f"{grouped_event_count}"
    )

    report_lines.append(
        f"Events without candidate group : "
        f"{unassigned_event_count}"
    )

    report_lines.append("")

    report_lines.append(
        "PAIR REVIEW"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        f"SAME_CANDIDATE : "
        f"{same_count}"
    )

    report_lines.append(
        f"UNCERTAIN : "
        f"{uncertain_count}"
    )

    report_lines.append(
        f"DIFFERENT_CANDIDATE : "
        f"{different_count}"
    )

    report_lines.append("")

    report_lines.append(
        f"SAME_CANDIDATE pairs "
        f"> 1 day : "
        f"{cross_time_same_count}"
    )

    report_lines.append("")

    report_lines.append(
        "CANDIDATE GROUPS"
    )

    report_lines.append(
        "-" * 60
    )

    if len(groups_df) == 0:

        report_lines.append(
            "Tidak ada candidate group."
        )

    else:

        for _, row in groups_df.iterrows():

            report_lines.append(
                f"{row['candidate_individual']} "
                f"({row['event_count']} event): "
                f"{row['events']}"
            )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        "Candidate individual merupakan "
        "hasil pengelompokan berbasis "
        "similarity dan review visual."
    )

    report_lines.append(
        "Hasil tersebut bukan ground-truth "
        "identitas individu."
    )

    report_lines.append(
        "Event yang belum memiliki hubungan "
        "cukup kuat tetap dipertahankan "
        "sebagai event terpisah."
    )

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Total leopard-cat events       : "
        f"{total_events}"
    )

    print(
        f"Candidate individual groups    : "
        f"{candidate_count}"
    )

    print(
        f"Events dalam candidate group   : "
        f"{grouped_event_count}"
    )

    print(
        f"Events belum tergabung         : "
        f"{unassigned_event_count}"
    )

    print()

    print(
        f"SAME_CANDIDATE                 : "
        f"{same_count}"
    )

    print(
        f"UNCERTAIN                      : "
        f"{uncertain_count}"
    )

    print(
        f"DIFFERENT_CANDIDATE            : "
        f"{different_count}"
    )

    print()

    print(
        f"SAME_CANDIDATE > 1 hari        : "
        f"{cross_time_same_count}"
    )

    print()

    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print()
    print(
        f"Summary     : {SUMMARY_CSV}"
    )

    print(
        f"Groups      : {GROUPS_CSV}"
    )

    print(
        f"Recurrence  : {RECURRENCE_CSV}"
    )

    print(
        f"Report      : {REPORT_TXT}"
    )

    print()
    print("=" * 70)
    print("SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()