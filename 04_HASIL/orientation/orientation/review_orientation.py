import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# ============================================================
# PATH PROJECT
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "02_DATASET",
    "dataset",
    "leopard_cat_master_fixed.csv"
)

LABEL_LAMA_CSV = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "orientation",
    "orientation",
    "orientation_labels.csv"
)

GROUND_TRUTH_CSV = os.path.join(
    BASE_DIR,
    "04_HASIL",
    "orientation",
    "orientation",
    "orientation_ground_truth.csv"
)


# ============================================================
# LABEL GROUND TRUTH
# ============================================================

LABELS = [
    "depan",
    "belakang",
    "kiri",
    "kanan",
    "tidak_yakin",
    "null"
]

STATUS_CONFIRMED = "CONFIRMED"
STATUS_PENDING = "REVIEW_PENDING"


# ============================================================
# CEK FILE
# ============================================================

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(
        f"Master tidak ditemukan:\n{MASTER_CSV}"
    )

if not os.path.exists(LABEL_LAMA_CSV):
    raise FileNotFoundError(
        f"Label lama tidak ditemukan:\n{LABEL_LAMA_CSV}"
    )


# ============================================================
# BACA MASTER
# ============================================================

master = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# BACA LABEL LAMA
# ============================================================

label_lama = pd.read_csv(
    LABEL_LAMA_CSV,
    dtype=str,
    keep_default_na=False
)

label_lama["arah_hadap"] = (
    label_lama["arah_hadap"]
    .fillna("")
    .str.strip()
    .str.lower()
)


# ============================================================
# BUAT DATA GROUND TRUTH
# ============================================================

records = []

for _, row in master.iterrows():

    for nomor in ["foto_1", "foto_2", "foto_3"]:

        records.append({
            "event_id": row["event_id"],
            "foto": nomor,
            "path_foto": row[nomor],
            "label_lama": "",
            "ground_truth": "",
            "status_review": STATUS_PENDING
        })


data_gt = pd.DataFrame(records)


# ============================================================
# MASUKKAN LABEL LAMA
# ============================================================

for _, row in label_lama.iterrows():

    kondisi = (
        (data_gt["event_id"] == row["event_id"])
        &
        (data_gt["foto"] == row["foto"])
    )

    data_gt.loc[
        kondisi,
        "label_lama"
    ] = row["arah_hadap"]


# ============================================================
# BACA GROUND TRUTH YANG SUDAH ADA
# ============================================================

if os.path.exists(GROUND_TRUTH_CSV):

    gt_lama = pd.read_csv(
        GROUND_TRUTH_CSV,
        dtype=str,
        keep_default_na=False
    )

    kolom_wajib = [
        "event_id",
        "foto",
        "ground_truth",
        "status_review"
    ]

    if all(
        kolom in gt_lama.columns
        for kolom in kolom_wajib
    ):

        for _, row in gt_lama.iterrows():

            kondisi = (
                (data_gt["event_id"] == row["event_id"])
                &
                (data_gt["foto"] == row["foto"])
            )

            data_gt.loc[
                kondisi,
                "ground_truth"
            ] = row["ground_truth"]

            data_gt.loc[
                kondisi,
                "status_review"
            ] = row["status_review"]


# ============================================================
# VARIABEL
# ============================================================

indeks_sekarang = 0
gambar_sekarang = None


# ============================================================
# SIMPAN
# ============================================================

def simpan_data():

    data_gt.to_csv(
        GROUND_TRUTH_CSV,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# PROGRESS
# ============================================================

def hitung_selesai():

    return (
        data_gt["status_review"]
        == STATUS_CONFIRMED
    ).sum()


def hitung_pending():

    return (
        data_gt["status_review"]
        == STATUS_PENDING
    ).sum()


def perbarui_progress():

    selesai = hitung_selesai()
    total = len(data_gt)
    pending = hitung_pending()

    label_progress.config(
        text=(
            f"Ground Truth: {selesai} / {total}   |   "
            f"Pending: {pending}"
        )
    )


# ============================================================
# CARI DATA BELUM CONFIRMED
# ============================================================

def cari_belum_confirmed():

    for i in range(len(data_gt)):

        if (
            data_gt.loc[
                i,
                "status_review"
            ]
            != STATUS_CONFIRMED
        ):

            return i

    return None


# ============================================================
# TAMPILKAN FOTO
# ============================================================

def tampilkan_foto():

    global gambar_sekarang

    row = data_gt.iloc[indeks_sekarang]

    label_event.config(
        text=(
            f"{row['event_id']}   |   "
            f"{row['foto'].upper()}"
        )
    )

    label_lama.config(
        text=(
            f"Label lama : "
            f"{row['label_lama'] or '-'}"
        )
    )

    label_gt.config(
        text=(
            f"Ground Truth : "
            f"{row['ground_truth'] or '-'}"
        )
    )

    label_status.config(
        text=(
            f"Status : "
            f"{row['status_review']}"
        )
    )

    path = row["path_foto"]

    if not os.path.exists(path):

        label_gambar.config(
            image="",
            text="FOTO TIDAK DITEMUKAN"
        )

        perbarui_progress()
        return

    try:

        gambar = Image.open(path)

        gambar.thumbnail(
            (1120, 390),
            Image.Resampling.LANCZOS
        )

        gambar_sekarang = ImageTk.PhotoImage(
            gambar
        )

        label_gambar.config(
            image=gambar_sekarang,
            text=""
        )

        label_gambar.image = gambar_sekarang

    except Exception as error:

        label_gambar.config(
            image="",
            text=f"Terjadi kesalahan:\n{error}"
        )

    perbarui_progress()


# ============================================================
# FOTO BERIKUTNYA YANG BELUM CONFIRMED
# ============================================================

def tampilkan_berikutnya():

    global indeks_sekarang

    indeks_baru = cari_belum_confirmed()

    if indeks_baru is None:

        simpan_data()
        perbarui_progress()

        messagebox.showinfo(
            "Review Selesai",
            "Semua 189 foto sudah memiliki Ground Truth."
        )

        return

    indeks_sekarang = indeks_baru

    tampilkan_foto()


# ============================================================
# TETAPKAN GROUND TRUTH
# ============================================================

def tetapkan_ground_truth(label):

    global indeks_sekarang

    data_gt.loc[
        indeks_sekarang,
        "ground_truth"
    ] = label

    data_gt.loc[
        indeks_sekarang,
        "status_review"
    ] = STATUS_CONFIRMED

    simpan_data()

    tampilkan_berikutnya()


# ============================================================
# PAKAI LABEL LAMA
# ============================================================

def pakai_label_lama():

    global indeks_sekarang

    label = data_gt.loc[
        indeks_sekarang,
        "label_lama"
    ]

    if label not in LABELS:

        messagebox.showwarning(
            "Label Tidak Valid",
            "Label lama tidak dapat digunakan."
        )

        return

    data_gt.loc[
        indeks_sekarang,
        "ground_truth"
    ] = label

    data_gt.loc[
        indeks_sekarang,
        "status_review"
    ] = STATUS_CONFIRMED

    simpan_data()

    tampilkan_berikutnya()


# ============================================================
# REVIEW NANTI
# ============================================================

def review_nanti():

    global indeks_sekarang

    data_gt.loc[
        indeks_sekarang,
        "ground_truth"
    ] = ""

    data_gt.loc[
        indeks_sekarang,
        "status_review"
    ] = STATUS_PENDING

    simpan_data()

    if indeks_sekarang < len(data_gt) - 1:
        indeks_sekarang += 1
    else:
        indeks_sekarang = 0

    tampilkan_foto()


# ============================================================
# FOTO SEBELUMNYA
# ============================================================

def foto_sebelumnya():

    global indeks_sekarang

    if indeks_sekarang > 0:

        indeks_sekarang -= 1

        tampilkan_foto()


# ============================================================
# FOTO BERIKUTNYA MANUAL
# ============================================================

def foto_berikutnya():

    global indeks_sekarang

    if indeks_sekarang < len(data_gt) - 1:

        indeks_sekarang += 1

        tampilkan_foto()


# ============================================================
# KEMBALI KE BELUM CONFIRMED
# ============================================================

def kembali_ke_belum_confirmed():

    global indeks_sekarang

    indeks_baru = cari_belum_confirmed()

    if indeks_baru is None:

        messagebox.showinfo(
            "Selesai",
            "Semua foto sudah memiliki Ground Truth."
        )

        return

    indeks_sekarang = indeks_baru

    tampilkan_foto()


# ============================================================
# TUTUP PROGRAM
# ============================================================

def tutup_program():

    simpan_data()

    root.destroy()


# ============================================================
# WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Ground Truth Review - Leopard Cat"
)

root.geometry(
    "1350x950"
)

root.minsize(
    1150,
    800
)

root.configure(
    bg="#f5f5f5"
)

root.protocol(
    "WM_DELETE_WINDOW",
    tutup_program
)


# ============================================================
# JUDUL
# ============================================================

judul = tk.Label(
    root,
    text="GROUND TRUTH REVIEW - LEOPARD CAT",
    font=("Arial", 22, "bold"),
    bg="#f5f5f5"
)

judul.pack(
    pady=(10, 3)
)


# ============================================================
# EVENT
# ============================================================

label_event = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#f5f5f5"
)

label_event.pack(
    pady=2
)


# ============================================================
# LABEL LAMA
# ============================================================

label_lama = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg="#f5f5f5"
)

label_lama.pack(
    pady=1
)


# ============================================================
# GROUND TRUTH
# ============================================================

label_gt = tk.Label(
    root,
    text="",
    font=("Arial", 13, "bold"),
    bg="#f5f5f5"
)

label_gt.pack(
    pady=1
)


# ============================================================
# STATUS
# ============================================================

label_status = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    bg="#f5f5f5"
)

label_status.pack(
    pady=1
)


# ============================================================
# PROGRESS
# ============================================================

label_progress = tk.Label(
    root,
    text="",
    font=("Arial", 12, "bold"),
    bg="#f5f5f5"
)

label_progress.pack(
    pady=3
)


# ============================================================
# AREA FOTO
# ============================================================

frame_gambar = tk.Frame(
    root,
    bg="black",
    width=1120,
    height=390
)

frame_gambar.pack(
    padx=20,
    pady=5
)

frame_gambar.pack_propagate(
    False
)


label_gambar = tk.Label(
    frame_gambar,
    bg="black",
    fg="white",
    font=("Arial", 14)
)

label_gambar.pack(
    expand=True
)


# ============================================================
# FRAME LABEL
# ============================================================

frame_label = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame_label.pack(
    pady=5
)


def buat_tombol(
    teks,
    nilai,
    kolom
):

    tombol = tk.Button(
        frame_label,
        text=teks,
        width=13,
        height=2,
        font=("Arial", 10, "bold"),
        command=lambda: tetapkan_ground_truth(nilai)
    )

    tombol.grid(
        row=0,
        column=kolom,
        padx=3
    )


buat_tombol("DEPAN", "depan", 0)
buat_tombol("BELAKANG", "belakang", 1)
buat_tombol("KIRI", "kiri", 2)
buat_tombol("KANAN", "kanan", 3)
buat_tombol("TIDAK YAKIN", "tidak_yakin", 4)
buat_tombol("NULL", "null", 5)


# ============================================================
# KONFIRMASI LABEL LAMA
# ============================================================

frame_konfirmasi = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame_konfirmasi.pack(
    pady=4
)


tk.Button(
    frame_konfirmasi,
    text="✓ PAKAI LABEL LAMA",
    width=24,
    height=2,
    font=("Arial", 10, "bold"),
    command=pakai_label_lama
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    frame_konfirmasi,
    text="REVIEW NANTI",
    width=20,
    height=2,
    font=("Arial", 10),
    command=review_nanti
).grid(
    row=0,
    column=1,
    padx=5
)


# ============================================================
# NAVIGASI
# ============================================================

frame_navigasi = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame_navigasi.pack(
    pady=(2, 6)
)


tk.Button(
    frame_navigasi,
    text="← SEBELUMNYA",
    width=18,
    height=2,
    font=("Arial", 10),
    command=foto_sebelumnya
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    frame_navigasi,
    text="BERIKUTNYA →",
    width=18,
    height=2,
    font=("Arial", 10),
    command=foto_berikutnya
).grid(
    row=0,
    column=1,
    padx=5
)


tk.Button(
    frame_navigasi,
    text="KE BELUM CONFIRMED",
    width=22,
    height=2,
    font=("Arial", 10),
    command=kembali_ke_belum_confirmed
).grid(
    row=0,
    column=2,
    padx=5
)


# ============================================================
# MULAI
# ============================================================

tampilkan_berikutnya()

root.mainloop()