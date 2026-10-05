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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "movement",
    "bounding_box"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "bounding_box_annotations.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MASTER
# ============================================================

df = pd.read_csv(
    MASTER_CSV,
    dtype=str,
    keep_default_na=False
)


# ============================================================
# BUAT DAFTAR FOTO
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
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "x_center": "",
            "y_center": "",
            "status": ""
        })


data = pd.DataFrame(records)


# ============================================================
# LOAD ANOTASI SEBELUMNYA
# ============================================================

if os.path.exists(OUTPUT_CSV):

    lama = pd.read_csv(
        OUTPUT_CSV,
        dtype=str,
        keep_default_na=False
    )

    kolom = [
        "event_id",
        "foto",
        "x1",
        "y1",
        "x2",
        "y2",
        "x_center",
        "y_center",
        "status"
    ]

    for _, row in lama.iterrows():

        kondisi = (
            (data["event_id"] == row["event_id"])
            &
            (data["foto"] == row["foto"])
        )

        for c in kolom[2:]:

            data.loc[
                kondisi,
                c
            ] = row[c]


# ============================================================
# VARIABEL
# ============================================================

indeks = 0

gambar_tampil = None

gambar_asli = None

skala_x = 1

skala_y = 1

x_mulai = None

y_mulai = None

kotak_sekarang = None


# ============================================================
# SIMPAN
# ============================================================

def simpan():

    data.to_csv(
        OUTPUT_CSV,
        index=False
    )


# ============================================================
# CARI FOTO BELUM DIANNOTASI
# ============================================================

def cari_belum():

    for i in range(
        len(data)
    ):

        if data.loc[
            i,
            "status"
        ] != "annotated":

            return i

    return None


# ============================================================
# PROGRESS
# ============================================================

def progress():

    selesai = (
        data["status"]
        == "annotated"
    ).sum()

    label_progress.config(
        text=(
            f"Anotasi: "
            f"{selesai} / {len(data)}"
        )
    )


# ============================================================
# TAMPILKAN FOTO
# ============================================================

def tampilkan():

    global gambar_tampil
    global gambar_asli
    global skala_x
    global skala_y
    global kotak_sekarang

    row = data.iloc[indeks]

    label_event.config(
        text=(
            f"Event: {row['event_id']}   |   "
            f"{row['foto']}"
        )
    )

    path = row["path_foto"]

    if not os.path.exists(path):

        canvas.delete("all")

        canvas.create_text(
            500,
            250,
            text="FOTO TIDAK DITEMUKAN",
            fill="white",
            font=("Arial", 18)
        )

        progress()

        return

    gambar_asli = Image.open(
        path
    ).convert(
        "RGB"
    )

    lebar_asli, tinggi_asli = (
        gambar_asli.size
    )

    max_width = 1000
    max_height = 500

    rasio = min(
        max_width / lebar_asli,
        max_height / tinggi_asli,
        1
    )

    lebar_baru = int(
        lebar_asli * rasio
    )

    tinggi_baru = int(
        tinggi_asli * rasio
    )

    gambar = gambar_asli.resize(
        (
            lebar_baru,
            tinggi_baru
        ),
        Image.Resampling.LANCZOS
    )

    skala_x = (
        lebar_asli
        /
        lebar_baru
    )

    skala_y = (
        tinggi_asli
        /
        tinggi_baru
    )

    gambar_tampil = ImageTk.PhotoImage(
        gambar
    )

    canvas.delete("all")

    canvas.config(
        width=lebar_baru,
        height=tinggi_baru
    )

    canvas.create_image(
        0,
        0,
        anchor="nw",
        image=gambar_tampil
    )

    kotak_sekarang = None

    # Kalau sebelumnya sudah ada anotasi,
    # tampilkan kembali kotaknya.

    if row["status"] == "annotated":

        try:

            x1 = float(row["x1"]) / skala_x
            y1 = float(row["y1"]) / skala_y
            x2 = float(row["x2"]) / skala_x
            y2 = float(row["y2"]) / skala_y

            kotak_sekarang = canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline="red",
                width=3
            )

        except Exception:
            pass

    progress()


# ============================================================
# MULAI DRAG
# ============================================================

def mulai_drag(event):

    global x_mulai
    global y_mulai
    global kotak_sekarang

    x_mulai = event.x
    y_mulai = event.y

    if kotak_sekarang is not None:

        canvas.delete(
            kotak_sekarang
        )

    kotak_sekarang = canvas.create_rectangle(
        x_mulai,
        y_mulai,
        x_mulai,
        y_mulai,
        outline="red",
        width=3
    )


# ============================================================
# SAAT DRAG
# ============================================================

def saat_drag(event):

    if kotak_sekarang is not None:

        canvas.coords(
            kotak_sekarang,
            x_mulai,
            y_mulai,
            event.x,
            event.y
        )


# ============================================================
# SELESAI DRAG
# ============================================================

def selesai_drag(event):

    global kotak_sekarang

    if kotak_sekarang is None:
        return

    x1 = min(
        x_mulai,
        event.x
    )

    y1 = min(
        y_mulai,
        event.y
    )

    x2 = max(
        x_mulai,
        event.x
    )

    y2 = max(
        y_mulai,
        event.y
    )

    # Pastikan kotak punya ukuran
    if (
        abs(x2 - x1) < 5
        or
        abs(y2 - y1) < 5
    ):

        canvas.delete(
            kotak_sekarang
        )

        kotak_sekarang = None

        return

    # Konversi koordinat tampilan
    # ke koordinat gambar asli.

    asli_x1 = round(
        x1 * skala_x,
        2
    )

    asli_y1 = round(
        y1 * skala_y,
        2
    )

    asli_x2 = round(
        x2 * skala_x,
        2
    )

    asli_y2 = round(
        y2 * skala_y,
        2
    )

    center_x = round(
        (
            asli_x1
            +
            asli_x2
        ) / 2,
        2
    )

    center_y = round(
        (
            asli_y1
            +
            asli_y2
        ) / 2,
        2
    )

    data.loc[
        indeks,
        "x1"
    ] = asli_x1

    data.loc[
        indeks,
        "y1"
    ] = asli_y1

    data.loc[
        indeks,
        "x2"
    ] = asli_x2

    data.loc[
        indeks,
        "y2"
    ] = asli_y2

    data.loc[
        indeks,
        "x_center"
    ] = center_x

    data.loc[
        indeks,
        "y_center"
    ] = center_y

    data.loc[
        indeks,
        "status"
    ] = "annotated"

    simpan()

    progress()


# ============================================================
# ULANG / RESET
# ============================================================

def reset_box():

    global kotak_sekarang

    canvas.delete(
        "all"
    )

    kotak_sekarang = None

    tampilkan()


# ============================================================
# SKIP
# ============================================================

def skip():

    global indeks

    jumlah = len(data)

    for langkah in range(
        1,
        jumlah + 1
    ):

        berikut = (
            indeks
            +
            langkah
        ) % jumlah

        if data.loc[
            berikut,
            "status"
        ] != "annotated":

            indeks = berikut

            tampilkan()

            return


# ============================================================
# SIMPAN & LANJUT
# ============================================================

def simpan_lanjut():

    global indeks

    if data.loc[
        indeks,
        "status"
    ] != "annotated":

        messagebox.showwarning(
            "Belum ada kotak",
            "Silakan gambar bounding box leopard cat terlebih dahulu."
        )

        return

    for langkah in range(
        1,
        len(data) + 1
    ):

        berikut = (
            indeks
            +
            langkah
        ) % len(data)

        if data.loc[
            berikut,
            "status"
        ] != "annotated":

            indeks = berikut

            tampilkan()

            return

    messagebox.showinfo(
        "Selesai",
        "Semua foto sudah dianotasi."
    )


# ============================================================
# WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Anotasi Bounding Box Leopard Cat"
)

root.geometry(
    "1200x750"
)

root.configure(
    bg="#f5f5f5"
)


# ============================================================
# JUDUL
# ============================================================

judul = tk.Label(
    root,
    text="ANNOTASI POSISI LEOPARD CAT",
    font=("Arial", 20, "bold"),
    bg="#f5f5f5"
)

judul.pack(
    pady=10
)


label_event = tk.Label(
    root,
    text="",
    font=("Arial", 14),
    bg="#f5f5f5"
)

label_event.pack()


label_progress = tk.Label(
    root,
    text="",
    font=("Arial", 12, "bold"),
    bg="#f5f5f5"
)

label_progress.pack(
    pady=5
)


# ============================================================
# CANVAS
# ============================================================

canvas_frame = tk.Frame(
    root,
    bg="black"
)

canvas_frame.pack(
    pady=10
)


canvas = tk.Canvas(
    canvas_frame,
    width=1000,
    height=500,
    bg="black",
    highlightthickness=0
)

canvas.pack()


canvas.bind(
    "<ButtonPress-1>",
    mulai_drag
)

canvas.bind(
    "<B1-Motion>",
    saat_drag
)

canvas.bind(
    "<ButtonRelease-1>",
    selesai_drag
)


# ============================================================
# TOMBOL
# ============================================================

frame_tombol = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame_tombol.pack(
    pady=10
)


tk.Button(
    frame_tombol,
    text="SIMPAN & LANJUT",
    width=20,
    height=2,
    command=simpan_lanjut
).grid(
    row=0,
    column=0,
    padx=5
)


tk.Button(
    frame_tombol,
    text="ULANGI KOTAK",
    width=20,
    height=2,
    command=reset_box
).grid(
    row=0,
    column=1,
    padx=5
)


tk.Button(
    frame_tombol,
    text="LEWATI",
    width=15,
    height=2,
    command=skip
).grid(
    row=0,
    column=2,
    padx=5
)


# ============================================================
# PETUNJUK
# ============================================================

petunjuk = tk.Label(
    root,
    text=(
        "Cara: klik dan tahan mouse di sudut kiri atas leopard cat, "
        "tarik sampai sudut kanan bawah tubuh leopard cat."
    ),
    font=("Arial", 10),
    bg="#f5f5f5"
)

petunjuk.pack(
    pady=5
)


# ============================================================
# MULAI
# ============================================================

indeks_awal = cari_belum()

if indeks_awal is None:

    messagebox.showinfo(
        "Selesai",
        "Semua foto sudah dianotasi."
    )

else:

    indeks = indeks_awal

    tampilkan()


root.mainloop()