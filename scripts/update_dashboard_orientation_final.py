import pandas as pd
from pathlib import Path
import shutil

BASE = Path(r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang")

DASHBOARD_CSV = (
    BASE / "04_HASIL" / "dashboard" / "dashboard_events.csv"
)

GROUND_TRUTH_CSV = (
    BASE / "04_HASIL" / "orientation" / "orientation"
    / "orientation_ground_truth.csv"
)

BACKUP_CSV = (
    BASE / "04_HASIL" / "dashboard"
    / "dashboard_events_before_orientation_final.csv"
)

print("Membaca dashboard...")
dashboard = pd.read_csv(DASHBOARD_CSV, keep_default_na=False)

print("Membaca orientation ground truth...")
gt = pd.read_csv(GROUND_TRUTH_CSV, keep_default_na=False)

print()
print("Dashboard :", len(dashboard), "event")
print("Ground truth:", len(gt), "foto")

# Pastikan GT tidak duplicate
duplicates = gt.duplicated(
    subset=["event_id", "foto"],
    keep=False
)

if duplicates.any():
    print("ERROR: duplicate event_id + foto ditemukan.")
    print(
        gt.loc[
            duplicates,
            ["event_id", "foto", "ground_truth"]
        ].to_string(index=False)
    )
    raise ValueError("Duplicate ground truth.")

# Backup
shutil.copy2(DASHBOARD_CSV, BACKUP_CSV)

print()
print("Backup dibuat:")
print(BACKUP_CSV)

# Buat mapping GT
orientation_map = {}

for _, row in gt.iterrows():
    key = (
        str(row["event_id"]),
        str(row["foto"])
    )

    orientation_map[key] = str(row["ground_truth"])

# Statistik
leopard_count = 0
null_count = 0
missing = []

# Update orientation
for index, row in dashboard.iterrows():

    event_id = str(row["event_id"])
    label = str(row["label"])

    if label == "leopard_cat":

        leopard_count += 1

        for n in [1, 2, 3]:

            foto_key = f"foto_{n}"
            arah_key = f"arah_{n}"

            key = (event_id, foto_key)

            if key not in orientation_map:
                missing.append(key)
            else:
                dashboard.at[
                    index,
                    arah_key
                ] = orientation_map[key]

    elif label == "null":

        null_count += 1

        # Tidak ada leopard cat pada event.
        # Maka orientation setiap foto = null.
        dashboard.at[index, "arah_1"] = "null"
        dashboard.at[index, "arah_2"] = "null"
        dashboard.at[index, "arah_3"] = "null"

    else:

        raise ValueError(
            f"Label event tidak dikenali: {label}"
        )

# Jangan simpan kalau GT leopard_cat tidak lengkap
if missing:

    print()
    print("ERROR: GT orientation tidak lengkap untuk event leopard_cat:")

    for item in missing:
        print(item)

    raise ValueError(
        f"Missing orientation GT: {len(missing)} foto."
    )

# Simpan
dashboard.to_csv(
    DASHBOARD_CSV,
    index=False,
    encoding="utf-8-sig"
)

print()
print("========================================")
print("UPDATE BERHASIL")
print("========================================")

print()
print("Total event dashboard :", len(dashboard))
print("Event leopard_cat     :", leopard_count)
print("Event null            :", null_count)

print()
print("Total foto orientation:")
print(len(dashboard) * 3)

print()
print("Distribusi orientation FINAL:")

all_labels = []

for col in ["arah_1", "arah_2", "arah_3"]:
    all_labels.extend(
        dashboard[col].astype(str).tolist()
    )

print(
    pd.Series(all_labels)
    .value_counts()
    .to_string()
)

print()
print("Dashboard sekarang menggunakan:")
print("- Ground Truth orientation untuk leopard_cat")
print("- null untuk event tanpa satwa")

print()
print("File:")
print(DASHBOARD_CSV)
