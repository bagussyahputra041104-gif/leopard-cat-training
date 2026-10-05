import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# ============================================================
# PENGATURAN
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

MASTER_CSV = os.path.join(
    BASE_DIR,
    "leopard_cat_master.csv"
)

FOLDER_OUTPUT = os.path.join(
    BASE_DIR,
    "orientation"
)

FILE_OUTPUT = os.path.join(
    FOLDER_OUTPUT,
    "orientation_labels.csv"
)

os.makedirs(
    FOLDER_OUTPUT,
    exist_ok=True
)


# ============================================================
# LABEL YANG DIGUNAKAN
# ============================================================

LABELS = [
    "depan",
    "belakang",
    "kiri",
    "kanan",
    "tidak_yakin"
]


# ============================================================
# MEMBACA MASTER DATA
# ============================================================

if not os.path.exists(MASTER_CSV):

    raise FileNotFoundError(
        f"File master tidak ditemukan:\n{MASTER_CSV}"
    )


df = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# MEMBUAT DAFTAR 189 FOTO
# ============================================================

records = []

for _, row in df.iterrows():

    for foto in [
        "foto_1",
        "foto_2",
        "foto_3"
    ]:

        records.append({

            "event_id": row["event_id"],

            "foto": foto,

            "path_foto": row[foto],

            "arah_hadap": ""
        })


data_label = pd.DataFrame(
    records
)


# ============================================================
# MEMBACA LABEL YANG SUDAH ADA
# ============================================================

if os.path.exists(FILE_OUTPUT):

    data_lama = pd.read_csv(
        FILE_OUTPUT,
        dtype=str,
        keep_default_na=False
    )

    if (
        "event_id" in data_lama.columns
        and
        "foto" in data_lama.columns
        and
        "arah_hadap" in data_lama.columns
    ):

        data_lama["arah_hadap"] = (
            data_lama["arah_hadap"]
            .fillna("")
            .str.strip()
            .str.lower()
        )

        # Hanya menerima label yang valid
        data_lama.loc[
            ~data_lama["arah_hadap"].isin(LABELS),
            "arah_hadap"
        ] = ""

        # Memasukkan label lama
        for _, baris in data_lama.iterrows():

            kondisi = (
                (data_label["event_id"] == baris["event_id"])
                &
                (data_label["foto"] == baris["foto"])
            )

            data_label.loc[
                kondisi,
                "arah_hadap"
            ] = baris["arah_hadap"]


# ============================================================
# VARIABEL
# ============================================================

indeks_sekarang = 0

gambar_sekarang = None


# ============================================================
# MENYIMPAN DATA
# ============================================================

def simpan_data():

    data_label.to_csv(
        FILE_OUTPUT,
        index=False
    )


# ============================================================
# MENGHITUNG PROGRESS
# ============================================================

def perbarui_progress():

    jumlah_sudah_dilabel = (
        data_label["arah_hadap"]
        .isin(LABELS)
        .sum()
    )

    jumlah_total = len(data_label)

    label_progress.config(
        text=(
            f"Progress Pelabelan: "
            f"{jumlah_sudah_dilabel} / "
            f"{jumlah_total}"
        )
    )


# ============================================================
# MENCARI FOTO BERIKUTNYA YANG BELUM DILABEL
# ============================================================

def cari_foto_belum_dilabel():

    for i in range(
        len(data_label)
    ):

        if data_label.loc[
            i,
            "arah_hadap"
        ] not in LABELS:

            return i

    return None


# ============================================================
# MENAMPILKAN FOTO
# ============================================================

def tampilkan_foto():

    global gambar_sekarang

    baris = data_label.iloc[
        indeks_sekarang
    ]

    label_event.config(
        text=(
            f"Event: {baris['event_id']}   |   "
            f"{baris['foto']}"
        )
    )

    path = baris["path_foto"]

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
            (
                950,
                470
            ),
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
# MENAMPILKAN FOTO BERIKUTNYA
# ============================================================

def tampilkan_berikutnya():

    global indeks_sekarang

    indeks_baru = cari_foto_belum_dilabel()

    if indeks_baru is None:

        simpan_data()

        perbarui_progress()

        messagebox.showinfo(
            "Selesai",
            "Semua foto sudah selesai diberi label."
        )

        return

    indeks_sekarang = indeks_baru

    tampilkan_foto()


# ============================================================
# MEMBERI LABEL
# ============================================================

def beri_label(label):

    global indeks_sekarang

    data_label.loc[
        indeks_sekarang,
        "arah_hadap"
    ] = label

    simpan_data()

    tampilkan_berikutnya()


# ============================================================
# LEWATI FOTO
# ============================================================

def lewati_foto():

    global indeks_sekarang

    jumlah_foto = len(data_label)

    indeks_awal = indeks_sekarang

    for langkah in range(
        1,
        jumlah_foto + 1
    ):

        indeks_baru = (
            indeks_awal + langkah
        ) % jumlah_foto

        if data_label.loc[
            indeks_baru,
            "arah_hadap"
        ] not in LABELS:

            indeks_sekarang = indeks_baru

            tampilkan_foto()

            return


# ============================================================
# JENDELA UTAMA
# ============================================================

root = tk.Tk()

root.title(
    "Pelabelan Arah Hadap Leopard Cat"
)

root.geometry(
    "1200x750"
)

root.minsize(
    1000,
    650
)

root.configure(
    bg="#f5f5f5"
)


# ============================================================
# JUDUL
# ============================================================

judul = tk.Label(
    root,
    text="PELABELAN ARAH HADAP LEOPARD CAT",
    font=("Arial", 20, "bold"),
    bg="#f5f5f5"
)

judul.pack(
    pady=(15, 5)
)


# ============================================================
# INFORMASI EVENT
# ============================================================

label_event = tk.Label(
    root,
    text="",
    font=("Arial", 14),
    bg="#f5f5f5"
)

label_event.pack(
    pady=3
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
    width=1000,
    height=480
)

frame_gambar.pack(
    padx=20,
    pady=8
)

frame_gambar.pack_propagate(
    False
)


label_gambar = tk.Label(
    frame_gambar,
    bg="black",
    fg="white"
)

label_gambar.pack(
    expand=True
)


# ============================================================
# TOMBOL LABEL
# ============================================================

frame_tombol = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame_tombol.pack(
    pady=8
)


def buat_tombol(
    teks,
    nilai,
    kolom
):

    tombol = tk.Button(
        frame_tombol,
        text=teks,
        width=14,
        height=2,
        font=("Arial", 11, "bold"),
        command=lambda: beri_label(nilai)
    )

    tombol.grid(
        row=0,
        column=kolom,
        padx=5
    )


buat_tombol(
    "DEPAN",
    "depan",
    0
)

buat_tombol(
    "BELAKANG",
    "belakang",
    1
)

buat_tombol(
    "KIRI",
    "kiri",
    2
)

buat_tombol(
    "KANAN",
    "kanan",
    3
)

buat_tombol(
    "TIDAK YAKIN",
    "tidak_yakin",
    4
)


# ============================================================
# TOMBOL LEWATI
# ============================================================

tombol_lewati = tk.Button(
    root,
    text="LEWATI / BELUM YAKIN",
    width=25,
    height=2,
    font=("Arial", 10),
    command=lewati_foto
)

tombol_lewati.pack(
    pady=(2, 10)
)


# ============================================================
# MULAI
# ============================================================

tampilkan_berikutnya()

root.mainloop()