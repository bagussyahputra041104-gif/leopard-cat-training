from pathlib import Path
import pandas as pd


ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

FINAL_FILE = (
    ROOT
    / "04_HASIL"
    / "individual"
    / "final_results"
    / "final_event_results.csv"
)


df = pd.read_csv(
    FINAL_FILE,
    keep_default_na=False
)


print("=" * 60)
print("AUDIT FINAL RE-ID CANDIDATE GROUP")
print("=" * 60)
print()


# ============================================================
# TOTAL EVENT
# ============================================================

print(f"Total event : {len(df)}")

if len(df) == 63:
    print("[OK] Total event = 63")
else:
    print("[WARNING] Total event bukan 63")


print()


# ============================================================
# DUPLIKAT EVENT
# ============================================================

duplicates = df[
    df["event_id"].duplicated(
        keep=False
    )
]


if duplicates.empty:

    print("[OK] Tidak ada event_id duplikat")

else:

    print("[WARNING] Ada event duplikat:")
    print(
        duplicates[
            ["event_id"]
        ].to_string(index=False)
    )


print()


# ============================================================
# DISTRIBUSI CANDIDATE
# ============================================================

print("Distribusi candidate_individual:")
print()

print(
    df["candidate_individual"]
    .value_counts()
    .to_string()
)

print()


# ============================================================
# IND_CAND_01
# ============================================================

cand_01 = df[
    df["candidate_individual"]
    == "IND_CAND_01"
]

print(
    f"IND_CAND_01 : {len(cand_01)} event"
)

print(
    cand_01[
        [
            "event_id",
            "candidate_individual",
            "reid_status"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# IND_CAND_02
# ============================================================

cand_02 = df[
    df["candidate_individual"]
    == "IND_CAND_02"
]

print(
    f"IND_CAND_02 : {len(cand_02)} event"
)

print(
    cand_02[
        [
            "event_id",
            "candidate_individual",
            "reid_status"
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# UNASSIGNED
# ============================================================

unassigned = df[
    df["candidate_individual"]
    == "UNASSIGNED"
]

print(
    f"UNASSIGNED : {len(unassigned)} event"
)

print()


# ============================================================
# CEK RE-ID STATUS
# ============================================================

print("Distribusi reid_status:")
print()

print(
    df["reid_status"]
    .value_counts()
    .to_string()
)

print()


# ============================================================
# CEK OVERLAP
# ============================================================

candidate_counts = (
    df.groupby("event_id")
    ["candidate_individual"]
    .nunique()
)

overlap = candidate_counts[
    candidate_counts > 1
]


if overlap.empty:

    print(
        "[OK] Tidak ada event dengan "
        "lebih dari satu candidate group."
    )

else:

    print(
        "[WARNING] Ditemukan overlap candidate:"
    )

    print(overlap)


print()


# ============================================================
# VALIDASI IND_CAND_02
# ============================================================

expected_cand_02 = {
    "EVT_0026",
    "EVT_0037",
    "EVT_0055",
}

actual_cand_02 = set(
    cand_02["event_id"]
)


if actual_cand_02 == expected_cand_02:

    print(
        "[OK] IND_CAND_02 berisi "
        "EVT_0026, EVT_0037, EVT_0055"
    )

else:

    print(
        "[WARNING] Isi IND_CAND_02 berbeda:"
    )

    print(
        sorted(actual_cand_02)
    )


print()


# ============================================================
# SELESAI
# ============================================================

print("=" * 60)
print("AUDIT SELESAI")
print("=" * 60)