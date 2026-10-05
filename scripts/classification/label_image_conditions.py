from pathlib import Path
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd


# ============================================================
# PATH PROJECT
# ============================================================

ROOT = Path(
    r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"
)

FINAL_CSV = (
    ROOT
    / "04_HASIL"
    / "individual"
    / "final_results"
    / "final_event_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "04_HASIL"
    / "classification"
    / "condition_audit"
)

OUTPUT_CSV = OUTPUT_DIR / "condition_labels.csv"

RAW_DIR = ROOT / "01_DATA_ASLI" / "leopard_cat"


# ============================================================
# PILIHAN LABEL
# ============================================================

LIGHT_OPTIONS = [
    "TERANG",
    "GELAP",
]

BLUR_OPTIONS = [
    "TIDAK",
    "RINGAN",
    "BERAT",
]

VISIBILITY_OPTIONS = [
    "PENUH",
    "SEBAGIAN",
]

DISTANCE_OPTIONS = [
    "DEKAT",
    "SEDANG",
    "JAUH",
]


# ============================================================
# CEK FILE
# ============================================================

if not FINAL_CSV.exists():
    raise FileNotFoundError(
        f"\nFile tidak ditemukan:\n{FINAL_CSV}\n"
    )


# ============================================================
# BACA DATA EVENT
# ============================================================

df = pd.read_csv(
    FINAL_CSV,
    dtype=str,
    keep_default_na=False
)

required_columns = [
    "event_id",
    "foto_1",
    "foto_2",
    "foto_3",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Kolom '{column}' tidak ditemukan."
        )


# ============================================================
# BUAT DAFTAR FOTO
# ============================================================

photos = []

for _, row in df.iterrows():

    event_id = row["event_id"]

    for photo_number in range(1, 4):

        old_path = Path(
            row[f"foto_{photo_number}"]
        )

        # ----------------------------------------------------
        # Coba path langsung
        # ----------------------------------------------------

        if old_path.exists():

            photo_path = old_path

        else:

            # ------------------------------------------------
            # Kalau path lama tidak valid,
            # cari berdasarkan nama file
            # ------------------------------------------------

            candidates = list(
                RAW_DIR.rglob(old_path.name)
            )

            if not candidates:

                print(
                    f"[WARNING] Foto tidak ditemukan: "
                    f"{old_path.name}"
                )

                continue

            photo_path = candidates[0]

        photos.append(
            {
                "event_id": event_id,
                "foto": photo_number,
                "path_foto": str(photo_path),
            }
        )


if not photos:
    raise RuntimeError(
        "Tidak ada foto yang berhasil ditemukan."
    )


print()
print("=" * 60)
print("AUDIT KONDISI FOTO LEOPARD CAT")
print("=" * 60)
print(f"Total foto ditemukan : {len(photos)}")
print("=" * 60)
print()


# ============================================================
# FOLDER OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD LABEL LAMA
# ============================================================

labels = {}

if OUTPUT_CSV.exists():

    old_df = pd.read_csv(
        OUTPUT_CSV,
        dtype=str,
        keep_default_na=False
    )

    for _, row in old_df.iterrows():

        key = (
            row["event_id"],
            int(row["foto"])
        )

        labels[key] = {
            "cahaya": row["cahaya"],
            "blur": row["blur"],
            "visibilitas": row["visibilitas"],
            "jarak": row["jarak"],
        }


# ============================================================
# TKINTER
# ============================================================

root = tk.Tk()

root.title(
    "Audit Kondisi Foto Leopard Cat"
)

# Ukuran awal
root.geometry(
    "1250x900"
)

# Ukuran minimum
root.minsize(
    1050,
    800
)

root.configure(
    bg="#1e1e1e"
)


# ============================================================
# VARIABLE
# ============================================================

current_index = 0

light_var = tk.StringVar()
blur_var = tk.StringVar()
visibility_var = tk.StringVar()
distance_var = tk.StringVar()


# ============================================================
# FUNGSI SIMPAN CSV
# ============================================================

def write_csv():

    rows = []

    for item in photos:

        key = (
            item["event_id"],
            item["foto"]
        )

        if key not in labels:
            continue

        data = labels[key]

        rows.append(
            {
                "event_id": item["event_id"],
                "foto": item["foto"],
                "path_foto": item["path_foto"],
                "cahaya": data["cahaya"],
                "blur": data["blur"],
                "visibilitas": data["visibilitas"],
                "jarak": data["jarak"],
            }
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "event_id",
                "foto",
                "path_foto",
                "cahaya",
                "blur",
                "visibilitas",
                "jarak",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CARI FOTO PERTAMA YANG BELUM DILABEL
# ============================================================

def find_first_unlabeled():

    for index, item in enumerate(photos):

        key = (
            item["event_id"],
            item["foto"]
        )

        if key not in labels:
            return index

    return 0


# ============================================================
# SIMPAN LABEL FOTO SEKARANG
# ============================================================

def save_current():

    item = photos[current_index]

    cahaya = light_var.get()
    blur = blur_var.get()
    visibilitas = visibility_var.get()
    jarak = distance_var.get()

    if not cahaya:
        messagebox.showwarning(
            "Belum lengkap",
            "Pilih kondisi CAHAYA terlebih dahulu."
        )
        return False

    if not blur:
        messagebox.showwarning(
            "Belum lengkap",
            "Pilih kondisi BLUR terlebih dahulu."
        )
        return False

    if not visibilitas:
        messagebox.showwarning(
            "Belum lengkap",
            "Pilih VISIBILITAS terlebih dahulu."
        )
        return False

    if not jarak:
        messagebox.showwarning(
            "Belum lengkap",
            "Pilih JARAK terlebih dahulu."
        )
        return False

    key = (
        item["event_id"],
        item["foto"]
    )

    labels[key] = {
        "cahaya": cahaya,
        "blur": blur,
        "visibilitas": visibilitas,
        "jarak": jarak,
    }

    write_csv()

    return True


# ============================================================
# LOAD LABEL YANG SUDAH ADA
# ============================================================

def load_current_labels():

    item = photos[current_index]

    key = (
        item["event_id"],
        item["foto"]
    )

    if key in labels:

        data = labels[key]

        light_var.set(
            data["cahaya"]
        )

        blur_var.set(
            data["blur"]
        )

        visibility_var.set(
            data["visibilitas"]
        )

        distance_var.set(
            data["jarak"]
        )

    else:

        light_var.set("")
        blur_var.set("")
        visibility_var.set("")
        distance_var.set("")


# ============================================================
# UPDATE PROGRESS
# ============================================================

def update_progress():

    completed = len(labels)
    total = len(photos)

    progress_label.config(
        text=(
            f"Foto {current_index + 1} / {total}"
            f"     |     "
            f"Sudah dilabel: {completed} / {total}"
        )
    )


# ============================================================
# TAMPILKAN FOTO
# ============================================================

def display_image():

    item = photos[current_index]

    image_path = Path(
        item["path_foto"]
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        # Ukuran area gambar
        max_width = 900
        max_height = 430

        ratio = min(
            max_width / image.width,
            max_height / image.height
        )

        # Jangan memperbesar gambar kecil
        ratio = min(
            ratio,
            1.0
        )

        new_width = int(
            image.width * ratio
        )

        new_height = int(
            image.height * ratio
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo

        info_label.config(
            text=(
                f"{item['event_id']}   |   "
                f"Foto {item['foto']}\n"
                f"{image_path.name}"
            )
        )

    except Exception as error:

        image_label.config(
            image="",
            text="GAGAL MEMBUKA FOTO",
            fg="red"
        )

        image_label.image = None

        info_label.config(
            text=str(error)
        )


# ============================================================
# REFRESH
# ============================================================

def refresh():

    display_image()

    load_current_labels()

    update_progress()


# ============================================================
# FOTO BERIKUTNYA
# ============================================================

def next_photo():

    global current_index

    if not save_current():
        return

    if current_index < len(photos) - 1:

        current_index += 1

        refresh()

    else:

        write_csv()

        messagebox.showinfo(
            "Selesai",
            "Semua foto sudah selesai dilabel."
        )


# ============================================================
# FOTO SEBELUMNYA
# ============================================================

def previous_photo():

    global current_index

    if current_index > 0:

        current_index -= 1

        refresh()


# ============================================================
# SELESAI
# ============================================================

def finish():

    write_csv()

    messagebox.showinfo(
        "Data tersimpan",
        (
            f"Data telah disimpan.\n\n"
            f"{OUTPUT_CSV}\n\n"
            f"Jumlah foto dilabel: {len(labels)}"
        )
    )

    root.destroy()


# ============================================================
# JUDUL
# ============================================================

title_label = tk.Label(
    root,
    text="AUDIT KONDISI FOTO LEOPARD CAT",
    font=(
        "Arial",
        22,
        "bold"
    ),
    fg="white",
    bg="#1e1e1e"
)

title_label.pack(
    pady=(12, 3)
)


# ============================================================
# PROGRESS
# ============================================================

progress_label = tk.Label(
    root,
    text="",
    font=(
        "Arial",
        12
    ),
    fg="#dddddd",
    bg="#1e1e1e"
)

progress_label.pack(
    pady=2
)


# ============================================================
# INFO FOTO
# ============================================================

info_label = tk.Label(
    root,
    text="",
    font=(
        "Arial",
        11
    ),
    fg="#cccccc",
    bg="#1e1e1e"
)

info_label.pack(
    pady=4
)


# ============================================================
# AREA FOTO
# ============================================================

image_frame = tk.Frame(
    root,
    bg="#111111",
    height=450
)

image_frame.pack(
    fill="x",
    padx=15,
    pady=5
)

image_frame.pack_propagate(
    False
)


image_label = tk.Label(
    image_frame,
    bg="#111111"
)

image_label.pack(
    expand=True
)


# ============================================================
# PANEL LABEL
# ============================================================

control_frame = tk.Frame(
    root,
    bg="#303030"
)

control_frame.pack(
    fill="x",
    padx=15,
    pady=5
)


# ============================================================
# FUNGSI MEMBUAT PILIHAN
# ============================================================

def create_option_group(
    parent,
    title,
    variable,
    options
):

    frame = tk.Frame(
        parent,
        bg="#303030"
    )

    frame.pack(
        side="left",
        expand=True,
        padx=10,
        pady=8
    )

    tk.Label(
        frame,
        text=title,
        font=(
            "Arial",
            10,
            "bold"
        ),
        fg="white",
        bg="#303030"
    ).pack(
        pady=(0, 3)
    )

    for option in options:

        tk.Radiobutton(
            frame,
            text=option,
            variable=variable,
            value=option,
            font=(
                "Arial",
                10
            ),
            fg="white",
            bg="#303030",
            selectcolor="#505050",
            activebackground="#303030",
            activeforeground="white"
        ).pack(
            anchor="w"
        )


# ============================================================
# BUAT PILIHAN
# ============================================================

create_option_group(
    control_frame,
    "CAHAYA",
    light_var,
    LIGHT_OPTIONS
)

create_option_group(
    control_frame,
    "BLUR",
    blur_var,
    BLUR_OPTIONS
)

create_option_group(
    control_frame,
    "VISIBILITAS",
    visibility_var,
    VISIBILITY_OPTIONS
)

create_option_group(
    control_frame,
    "JARAK",
    distance_var,
    DISTANCE_OPTIONS
)


# ============================================================
# TOMBOL NAVIGASI
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#1e1e1e"
)

button_frame.pack(
    pady=8
)


tk.Button(
    button_frame,
    text="← SEBELUMNYA",
    font=(
        "Arial",
        11,
        "bold"
    ),
    width=16,
    height=2,
    command=previous_photo
).pack(
    side="left",
    padx=8
)


tk.Button(
    button_frame,
    text="SIMPAN & LANJUT →",
    font=(
        "Arial",
        11,
        "bold"
    ),
    width=20,
    height=2,
    command=next_photo
).pack(
    side="left",
    padx=8
)


tk.Button(
    button_frame,
    text="SELESAI",
    font=(
        "Arial",
        11,
        "bold"
    ),
    width=12,
    height=2,
    command=finish
).pack(
    side="left",
    padx=8
)


# ============================================================
# START
# ============================================================

current_index = find_first_unlabeled()

refresh()

root.mainloop()