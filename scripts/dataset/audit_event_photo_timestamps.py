import os
import pandas as pd
from PIL import Image


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
    "event_photo_timestamps.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# EXIF TIMESTAMP
# =========================================================

def get_photo_timestamp(path):

    try:

        image = Image.open(path)

        exif = image.getexif()

        if not exif:
            return None

        # EXIF DateTimeOriginal
        # 36867
        timestamp = exif.get(36867)

        if timestamp:
            return pd.to_datetime(
                timestamp,
                format="%Y:%m:%d %H:%M:%S",
                errors="coerce"
            )

        # EXIF DateTimeDigitized
        # 36868
        timestamp = exif.get(36868)

        if timestamp:
            return pd.to_datetime(
                timestamp,
                format="%Y:%m:%d %H:%M:%S",
                errors="coerce"
            )

        # EXIF DateTime
        # 306
        timestamp = exif.get(306)

        if timestamp:
            return pd.to_datetime(
                timestamp,
                format="%Y:%m:%d %H:%M:%S",
                errors="coerce"
            )

    except Exception:
        pass

    return None


# =========================================================
# LOAD
# =========================================================

print("=" * 70)
print("AUDIT TIMESTAMP SETIAP FOTO DALAM EVENT")
print("=" * 70)

df = pd.read_csv(
    INPUT_CSV,
    dtype=str,
    keep_default_na=False
)

print()
print(
    f"Total event : {len(df)}"
)


# =========================================================
# PROCESS
# =========================================================

rows = []

for _, event in df.iterrows():

    event_id = event["event_id"]

    photo_times = []

    for photo_number in range(1, 4):

        path = event[
            f"foto_{photo_number}"
        ]

        timestamp = get_photo_timestamp(
            path
        )

        photo_times.append(
            timestamp
        )

        rows.append({
            "event_id": event_id,
            "photo_number": photo_number,
            "image_path": path,
            "photo_timestamp": timestamp
        })


# =========================================================
# DATAFRAME
# =========================================================

result = pd.DataFrame(rows)

result["photo_timestamp"] = pd.to_datetime(
    result["photo_timestamp"],
    errors="coerce"
)

result = result.sort_values(
    [
        "event_id",
        "photo_number"
    ]
)


# =========================================================
# CALCULATE INTERNAL EVENT GAPS
# =========================================================

event_summary = []

for event_id, group in result.groupby(
    "event_id"
):

    group = group.sort_values(
        "photo_number"
    )

    timestamps = (
        group["photo_timestamp"]
        .dropna()
        .tolist()
    )

    if len(timestamps) >= 2:

        gaps = []

        for i in range(
            1,
            len(timestamps)
        ):

            gap = (
                timestamps[i]
                -
                timestamps[i - 1]
            ).total_seconds()

            gaps.append(gap)

        min_gap = min(gaps)
        max_gap = max(gaps)

    else:

        min_gap = None
        max_gap = None

    event_summary.append({
        "event_id": event_id,
        "timestamp_foto_1": (
            timestamps[0]
            if len(timestamps) >= 1
            else None
        ),
        "timestamp_foto_2": (
            timestamps[1]
            if len(timestamps) >= 2
            else None
        ),
        "timestamp_foto_3": (
            timestamps[2]
            if len(timestamps) >= 3
            else None
        ),
        "jumlah_timestamp_valid": len(
            timestamps
        ),
        "min_gap_internal_seconds": min_gap,
        "max_gap_internal_seconds": max_gap
    })


summary = pd.DataFrame(
    event_summary
)


# =========================================================
# SAVE
# =========================================================

result.to_csv(
    OUTPUT_CSV,
    index=False
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "event_internal_timing_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print()
print("=" * 70)
print("TIMESTAMP PER FOTO")
print("=" * 70)

print()

print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# STATISTICS
# =========================================================

valid_internal = summary[
    summary[
        "max_gap_internal_seconds"
    ].notna()
][
    "max_gap_internal_seconds"
]


print()
print("=" * 70)
print("STATISTIK INTERNAL EVENT")
print("=" * 70)

print()

print(
    f"Event dengan 3 timestamp valid : "
    f"{int((summary['jumlah_timestamp_valid'] == 3).sum())}"
)

print(
    f"Event timestamp tidak lengkap  : "
    f"{int((summary['jumlah_timestamp_valid'] < 3).sum())}"
)

if len(valid_internal) > 0:

    print(
        f"Minimum gap dalam event : "
        f"{valid_internal.min():.0f} detik"
    )

    print(
        f"Median gap dalam event  : "
        f"{valid_internal.median():.0f} detik"
    )

    print(
        f"Maximum gap dalam event : "
        f"{valid_internal.max():.0f} detik"
    )


# =========================================================
# CLOSE INTERNAL GAPS
# =========================================================

print()
print("=" * 70)
print("CONTOH TIMESTAMP FOTO")
print("=" * 70)

print()

print(
    summary.head(20).to_string(
        index=False
    )
)


print()
print("=" * 70)
print("SELESAI")
print("=" * 70)

print()
print(
    "Detail:"
)

print(
    OUTPUT_CSV
)

print()
print(
    "Summary:"
)

print(
    summary_path
)