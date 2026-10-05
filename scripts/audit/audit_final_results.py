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


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("AUDIT FINAL RESULTS")
print("=" * 70)

if not FINAL_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan:\n{FINAL_FILE}"
    )

df = pd.read_csv(
    FINAL_FILE,
    dtype=str,
    keep_default_na=False
)

print(f"\nFile:")
print(FINAL_FILE)

print(f"\nTotal rows: {len(df)}")


# ============================================================
# 1. EVENT ID
# ============================================================

print("\n" + "-" * 70)
print("1. EVENT ID")
print("-" * 70)

duplicate_events = df[
    df["event_id"].duplicated(keep=False)
]

missing_event_id = (
    df["event_id"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(f"Duplicate event ID : {len(duplicate_events)}")
print(f"Missing event ID   : {missing_event_id}")


# ============================================================
# 2. FOTO
# ============================================================

print("\n" + "-" * 70)
print("2. FOTO")
print("-" * 70)

photo_columns = [
    "foto_1",
    "foto_2",
    "foto_3"
]

for col in photo_columns:

    if col not in df.columns:
        print(f"{col}: KOLOM TIDAK ADA")
        continue

    missing = (
        df[col]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{col}: "
        f"missing = {missing}"
    )


# ============================================================
# 3. ORIENTATION
# ============================================================

print("\n" + "-" * 70)
print("3. ORIENTATION")
print("-" * 70)

orientation_columns = [
    "arah_foto_1",
    "arah_foto_2",
    "arah_foto_3"
]

for col in orientation_columns:

    if col not in df.columns:
        print(f"{col}: KOLOM TIDAK ADA")
        continue

    missing = (
        df[col]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{col}: "
        f"missing = {missing}"
    )


# ============================================================
# 4. MOVEMENT
# ============================================================

print("\n" + "-" * 70)
print("4. MOVEMENT")
print("-" * 70)

if "movement" in df.columns:

    missing = (
        df["movement"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Missing movement: {missing}")

    print("\nDistribusi movement:")

    print(
        df["movement"]
        .value_counts()
    )

else:

    print("Kolom movement tidak ditemukan.")


# ============================================================
# 5. RE-ID
# ============================================================

print("\n" + "-" * 70)
print("5. RE-ID")
print("-" * 70)

if "reid_status" in df.columns:

    print("\nStatus Re-ID:")

    print(
        df["reid_status"]
        .value_counts()
    )

else:

    print("Kolom reid_status tidak ditemukan.")


if "candidate_individual" in df.columns:

    print("\nCandidate individual:")

    print(
        df["candidate_individual"]
        .value_counts()
    )

else:

    print(
        "Kolom candidate_individual "
        "tidak ditemukan."
    )


# ============================================================
# 6. TIMESTAMP
# ============================================================

print("\n" + "-" * 70)
print("6. TIMESTAMP")
print("-" * 70)

if "timestamp" in df.columns:

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid = timestamps.isna().sum()

    print(
        f"Timestamp invalid: {invalid}"
    )

    if invalid == 0:

        print(
            f"Timestamp awal : "
            f"{timestamps.min()}"
        )

        print(
            f"Timestamp akhir: "
            f"{timestamps.max()}"
        )

else:

    print("Kolom timestamp tidak ditemukan.")


# ============================================================
# 7. KOLOM
# ============================================================

print("\n" + "-" * 70)
print("7. KOLOM FINAL")
print("-" * 70)

print(
    df.columns.tolist()
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("AUDIT SELESAI")
print("=" * 70)

print(
    "\nCatatan: audit ini hanya membaca "
    "final_event_results.csv dan tidak mengubah file."
)