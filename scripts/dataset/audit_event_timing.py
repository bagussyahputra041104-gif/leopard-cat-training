import os
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "event_timing_audit"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "event_timing_details.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "event_timing_summary.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# LOAD
# =========================================================

print("=" * 70)
print("AUDIT TIMING EVENT LEOPARD CAT")
print("=" * 70)

df = pd.read_csv(
    INPUT_CSV,
    dtype=str,
    keep_default_na=False
)

df["timestamp_dt"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.sort_values(
    "timestamp_dt"
).reset_index(
    drop=True
)

print()
print(f"Total event : {len(df)}")

invalid = df["timestamp_dt"].isna().sum()

print(
    f"Timestamp invalid : {invalid}"
)


# =========================================================
# CALCULATE GAP
# =========================================================

df["previous_event"] = df["event_id"].shift(1)

df["previous_timestamp"] = (
    df["timestamp_dt"].shift(1)
)

df["gap_seconds"] = (
    df["timestamp_dt"] -
    df["previous_timestamp"]
).dt.total_seconds()


df["gap_minutes"] = (
    df["gap_seconds"] / 60
)


# =========================================================
# CLASSIFY GAP
# =========================================================

def classify_gap(seconds):

    if pd.isna(seconds):
        return "FIRST_EVENT"

    if seconds < 60:
        return "< 1 menit"

    if seconds < 5 * 60:
        return "1-5 menit"

    if seconds < 10 * 60:
        return "5-10 menit"

    if seconds < 20 * 60:
        return "10-20 menit"

    if seconds < 30 * 60:
        return "20-30 menit"

    if seconds < 60 * 60:
        return "30-60 menit"

    if seconds < 24 * 60 * 60:
        return "1-24 jam"

    return "> 24 jam"


df["gap_category"] = (
    df["gap_seconds"]
    .apply(classify_gap)
)


# =========================================================
# FLAG CLOSE EVENTS
# =========================================================

df["close_under_1min"] = (
    df["gap_seconds"] < 60
)

df["close_under_5min"] = (
    df["gap_seconds"] < 5 * 60
)

df["close_under_10min"] = (
    df["gap_seconds"] < 10 * 60
)

df["close_under_30min"] = (
    df["gap_seconds"] < 30 * 60
)


# =========================================================
# SAVE DETAIL
# =========================================================

detail_columns = [
    "event_id",
    "timestamp",
    "previous_event",
    "previous_timestamp",
    "gap_seconds",
    "gap_minutes",
    "gap_category",
    "close_under_1min",
    "close_under_5min",
    "close_under_10min",
    "close_under_30min"
]

detail_df = df[
    detail_columns
].copy()

detail_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

valid_gaps = df[
    df["gap_seconds"].notna()
]["gap_seconds"]

summary = []

summary.append({
    "kategori": "Total event",
    "jumlah": len(df)
})

summary.append({
    "kategori": "Gap < 1 menit",
    "jumlah": int(
        (valid_gaps < 60).sum()
    )
})

summary.append({
    "kategori": "Gap < 5 menit",
    "jumlah": int(
        (valid_gaps < 5 * 60).sum()
    )
})

summary.append({
    "kategori": "Gap < 10 menit",
    "jumlah": int(
        (valid_gaps < 10 * 60).sum()
    )
})

summary.append({
    "kategori": "Gap < 30 menit",
    "jumlah": int(
        (valid_gaps < 30 * 60).sum()
    )
})

summary.append({
    "kategori": "Gap 30-60 menit",
    "jumlah": int(
        (
            (valid_gaps >= 30 * 60) &
            (valid_gaps < 60 * 60)
        ).sum()
    )
})

summary.append({
    "kategori": "Gap 1-24 jam",
    "jumlah": int(
        (
            (valid_gaps >= 60 * 60) &
            (valid_gaps < 24 * 60 * 60)
        ).sum()
    )
})

summary.append({
    "kategori": "Gap > 24 jam",
    "jumlah": int(
        (valid_gaps >= 24 * 60 * 60).sum()
    )
})

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# =========================================================
# DISPLAY SUMMARY
# =========================================================

print()
print("=" * 70)
print("RINGKASAN GAP ANTAR-EVENT")
print("=" * 70)

print()

print(
    summary_df.to_string(
        index=False
    )
)


# =========================================================
# CLOSE EVENTS
# =========================================================

print()
print("=" * 70)
print("EVENT YANG BERDEKATAN (< 10 MENIT)")
print("=" * 70)

close_df = df[
    (
        df["gap_seconds"].notna()
    ) &
    (
        df["gap_seconds"] < 10 * 60
    )
].copy()

if len(close_df) == 0:

    print()
    print("Tidak ada event dengan gap < 10 menit.")

else:

    print()

    for _, row in close_df.iterrows():

        print(
            f"{row['previous_event']} "
            f"-> "
            f"{row['event_id']} | "
            f"{row['gap_seconds']:.0f} detik | "
            f"{row['timestamp']}"
        )


# =========================================================
# VERY CLOSE EVENTS
# =========================================================

print()
print("=" * 70)
print("EVENT DENGAN GAP < 1 MENIT")
print("=" * 70)

very_close = df[
    (
        df["gap_seconds"].notna()
    ) &
    (
        df["gap_seconds"] < 60
    )
].copy()

if len(very_close) == 0:

    print()
    print("Tidak ada event dengan gap < 1 menit.")

else:

    print()

    for _, row in very_close.iterrows():

        print(
            f"{row['previous_event']} "
            f"-> "
            f"{row['event_id']} | "
            f"{row['gap_seconds']:.0f} detik"
        )


# =========================================================
# MIN / MAX / MEDIAN
# =========================================================

print()
print("=" * 70)
print("STATISTIK GAP")
print("=" * 70)

if len(valid_gaps) > 0:

    print(
        f"Minimum : "
        f"{valid_gaps.min():.0f} detik"
    )

    print(
        f"Median  : "
        f"{valid_gaps.median():.0f} detik"
    )

    print(
        f"Maximum : "
        f"{valid_gaps.max():.0f} detik"
    )


# =========================================================
# FINISH
# =========================================================

print()
print("=" * 70)
print("SELESAI")
print("=" * 70)

print()
print("Detail:")
print(OUTPUT_CSV)

print()
print("Summary:")
print(SUMMARY_CSV)