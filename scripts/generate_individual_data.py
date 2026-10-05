from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "05_REID"
    / "evaluation"
    / "individual_candidate_final.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "leopard_cat_dashboard"
    / "lib"
    / "data"
    / "individual_data.dart"
)


def dart_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def main():
    print("=" * 60)
    print("GENERATE INDIVIDUAL DATA")
    print("=" * 60)
    print(f"Input  : {CSV_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV tidak ditemukan: {CSV_PATH}"
        )

    candidates = {}

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            event_id = row["event_id"].strip()
            candidate_id = row["candidate_individual"].strip()

            if candidate_id == "UNASSIGNED":
                continue

            if candidate_id not in candidates:
                candidates[candidate_id] = []

            candidates[candidate_id].append(event_id)

    lines = []

    lines.append(
        "import '../models/individual_candidate.dart';"
    )
    lines.append("")
    lines.append(
        "const List<IndividualCandidate> individualCandidates = ["
    )
    lines.append("")

    for candidate_id, event_ids in sorted(candidates.items()):
        title = candidate_id.replace("_", " ").title()

        lines.append("  IndividualCandidate(")
        lines.append(
            f"    candidateId: '{dart_string(candidate_id)}',"
        )
        lines.append(
            f"    title: '{dart_string(title)}',"
        )
        lines.append(
            f"    eventCount: {len(event_ids)},"
        )
        lines.append("    eventIds: [")

        for event_id in event_ids:
            lines.append(
                f"      '{dart_string(event_id)}',"
            )

        lines.append("    ],")
        lines.append("  ),")
        lines.append("")

    lines.append("];")
    lines.append("")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 60)
    print("SELESAI")
    print("=" * 60)

    for candidate_id, event_ids in sorted(candidates.items()):
        print(
            f"{candidate_id}: {len(event_ids)} event"
        )

    print()
    print(f"Total candidate : {len(candidates)}")
    print(f"File            : {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()