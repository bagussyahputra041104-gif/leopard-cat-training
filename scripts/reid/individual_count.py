from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_CSV = (
    BASE_DIR / "leopard_cat_master.csv"
)

FINAL_EVENT_CSV = (
    BASE_DIR / "individual_candidate_final.csv"
)

SUMMARY_CSV = (
    BASE_DIR
    / "reid_evaluation"
    / "reid_summary.csv"
)

RECURRENCE_CSV = (
    BASE_DIR
    / "reid_evaluation"
    / "candidate_recurrence.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "individual_count"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "individual_count_summary.csv"
)

OUTPUT_ASSIGNMENT = (
    OUTPUT_DIR / "individual_event_assignment.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR / "individual_count_report.txt"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDIVIDUAL COUNT ANALYSIS")
    print("=" * 70)

    print()
    print(f"Master      : {MASTER_CSV}")
    print(f"Event table : {FINAL_EVENT_CSV}")
    print(f"Re-ID       : {SUMMARY_CSV}")
    print(f"Recurrence  : {RECURRENCE_CSV}")
    print()

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        MASTER_CSV,
        FINAL_EVENT_CSV,
        SUMMARY_CSV,
        RECURRENCE_CSV
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan:\n{path}"
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

    summary_df = pd.read_csv(
        SUMMARY_CSV,
        dtype=str,
        keep_default_na=False
    )

    recurrence_df = pd.read_csv(
        RECURRENCE_CSV,
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
        event_df["status"]
        == "GROUPED_CANDIDATE"
    ].copy()

    unassigned_df = event_df[
        event_df["status"]
        == "NO_CONFIRMED_CANDIDATE"
    ].copy()

    candidate_groups = (
        grouped_df[
            "candidate_individual"
        ]
        .nunique()
    )

    grouped_events = len(
        grouped_df
    )

    unassigned_events = len(
        unassigned_df
    )

    # --------------------------------------------------------
    # RE-ID COVERAGE
    # --------------------------------------------------------

    if total_events > 0:

        reid_coverage = (
            grouped_events
            / total_events
            * 100
        )

    else:

        reid_coverage = 0

    # --------------------------------------------------------
    # BUILD EVENT ASSIGNMENT
    # --------------------------------------------------------

    assignment_df = event_df.copy()

    assignment_df[
        "counting_status"
    ] = assignment_df[
        "status"
    ].map({

        "GROUPED_CANDIDATE":
            "CANDIDATE_INDIVIDUAL",

        "NO_CONFIRMED_CANDIDATE":
            "UNASSIGNED"
    })

    assignment_df[
        "is_candidate_individual"
    ] = (
        assignment_df["status"]
        == "GROUPED_CANDIDATE"
    )

    assignment_df.to_csv(
        OUTPUT_ASSIGNMENT,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # CANDIDATE GROUP SUMMARY
    # --------------------------------------------------------

    group_rows = []

    for candidate_id, group in grouped_df.groupby(
        "candidate_individual"
    ):

        events = sorted(
            group["event_id"].tolist()
        )

        recurrence_match = recurrence_df[
            recurrence_df[
                "candidate_individual"
            ]
            == candidate_id
        ]

        if len(recurrence_match) > 0:

            recurrence = (
                recurrence_match.iloc[0]
            )

            first_seen = (
                recurrence["first_seen"]
            )

            last_seen = (
                recurrence["last_seen"]
            )

            span_days = float(
                recurrence["span_days"]
            )

            source_count = int(
                recurrence["source_count"]
            )

        else:

            first_seen = ""
            last_seen = ""
            span_days = 0
            source_count = 0

        group_rows.append({

            "candidate_individual":
                candidate_id,

            "event_count":
                len(events),

            "events":
                ", ".join(events),

            "first_seen":
                first_seen,

            "last_seen":
                last_seen,

            "span_days":
                span_days,

            "source_count":
                source_count,

            "interpretation":
                "Candidate individual; "
                "not ground-truth population count"
        })

    group_summary_df = pd.DataFrame(
        group_rows
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    summary_rows = [

        {
            "metric":
                "Total leopard-cat events",

            "value":
                total_events,

            "unit":
                "event"
        },

        {
            "metric":
                "Candidate individual groups",

            "value":
                candidate_groups,

            "unit":
                "candidate group"
        },

        {
            "metric":
                "Events in candidate groups",

            "value":
                grouped_events,

            "unit":
                "event"
        },

        {
            "metric":
                "Unassigned events",

            "value":
                unassigned_events,

            "unit":
                "event"
        },

        {
            "metric":
                "Re-ID candidate coverage",

            "value":
                round(
                    reid_coverage,
                    2
                ),

            "unit":
                "%"
        },

        {
            "metric":
                "Conservative candidate-group count",

            "value":
                candidate_groups,

            "unit":
                "candidate group"
        },

        {
            "metric":
                "Unassigned events are separate individuals?",

            "value":
                "NO",

            "unit":
                "not determined"
        }
    ]

    summary_output_df = pd.DataFrame(
        summary_rows
    )

    summary_output_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "INDIVIDUAL COUNT ANALYSIS"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Total leopard-cat events: "
        f"{total_events}"
    )

    report.append(
        f"Candidate individual groups: "
        f"{candidate_groups}"
    )

    report.append(
        f"Events in candidate groups: "
        f"{grouped_events}"
    )

    report.append(
        f"Unassigned events: "
        f"{unassigned_events}"
    )

    report.append(
        f"Re-ID candidate coverage: "
        f"{reid_coverage:.2f}%"
    )

    report.append("")

    report.append(
        "CANDIDATE GROUPS"
    )

    report.append(
        "-" * 70
    )

    if len(group_summary_df) == 0:

        report.append(
            "Belum terdapat candidate individual group."
        )

    else:

        for _, row in group_summary_df.iterrows():

            report.append(
                f"{row['candidate_individual']}: "
                f"{row['event_count']} event"
            )

            report.append(
                f"  Events: {row['events']}"
            )

            report.append(
                f"  First seen: {row['first_seen']}"
            )

            report.append(
                f"  Last seen: {row['last_seen']}"
            )

            report.append(
                f"  Span: {row['span_days']} hari"
            )

            report.append(
                f"  Source count: {row['source_count']}"
            )

            report.append("")

    report.append(
        "INTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Jumlah candidate individual "
        "tidak diperlakukan sebagai estimasi "
        "populasi sebenarnya."
    )

    report.append(
        "Event yang belum memiliki hubungan "
        "Re-ID yang cukup kuat tetap diberi "
        "status UNASSIGNED."
    )

    report.append(
        "UNASSIGNED tidak otomatis berarti "
        "individu yang berbeda."
    )

    report.append(
        "Candidate group merupakan hasil "
        "similarity embedding dan review visual "
        "manual, bukan ground-truth identitas."
    )

    OUTPUT_REPORT.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print("=" * 70)
    print("INDIVIDUAL COUNT SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Total leopard-cat events       : "
        f"{total_events}"
    )

    print(
        f"Candidate individual groups    : "
        f"{candidate_groups}"
    )

    print(
        f"Events dalam candidate group   : "
        f"{grouped_events}"
    )

    print(
        f"Events belum tergabung         : "
        f"{unassigned_events}"
    )

    print(
        f"Re-ID candidate coverage       : "
        f"{reid_coverage:.2f}%"
    )

    print()

    print(
        "Interpretasi:"
    )

    print(
        "Candidate group bukan "
        "jumlah populasi sebenarnya."
    )

    print(
        "Unassigned event bukan berarti "
        "individu berbeda."
    )

    print()

    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print()

    print(
        f"Summary     : {OUTPUT_SUMMARY}"
    )

    print(
        f"Assignment  : {OUTPUT_ASSIGNMENT}"
    )

    print(
        f"Report      : {OUTPUT_REPORT}"
    )

    print()

    print("=" * 70)
    print("SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()