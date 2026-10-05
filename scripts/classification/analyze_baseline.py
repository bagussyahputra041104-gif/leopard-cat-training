import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = r"C:\Users\bagus\OneDrive\Bagus\OneDrive\Dokumen\Magang"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "analysis_baseline"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. TRAINING LOG
# ============================================================

epochs = list(range(1, 11))

train_loss = [
    0.1145,
    0.0111,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000
]

train_accuracy = [
    96.45,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00
]

val_loss = [
    0.4268,
    0.6702,
    0.7023,
    0.7088,
    0.7643,
    0.7086,
    0.7725,
    0.8278,
    0.8173,
    0.8410
]

val_accuracy = [
    90.00,
    90.00,
    90.00,
    90.00,
    90.00,
    90.00,
    90.00,
    90.00,
    90.00,
    90.00
]


# ============================================================
# 3. BUAT DATAFRAME
# ============================================================

df = pd.DataFrame({

    "epoch": epochs,

    "train_loss": train_loss,

    "train_accuracy": train_accuracy,

    "val_loss": val_loss,

    "val_accuracy": val_accuracy

})


# ============================================================
# 4. CARI BEST EPOCH
# ============================================================

best_epoch = (
    df.loc[
        df["val_accuracy"].idxmax(),
        "epoch"
    ]
)

best_val_accuracy = (
    df["val_accuracy"].max()
)

best_val_loss = (
    df.loc[
        df["val_loss"].idxmin(),
        "val_loss"
    ]
)


# ============================================================
# 5. SIMPAN DATA TRAINING
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 6. GRAFIK ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    df["epoch"],
    df["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Train vs Validation Accuracy"
)

plt.xticks(
    epochs
)

plt.legend()

plt.grid(
    True
)

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. GRAFIK LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Train vs Validation Loss"
)

plt.xticks(
    epochs
)

plt.legend()

plt.grid(
    True
)

loss_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. ANALISIS SEDERHANA
# ============================================================

final_train_accuracy = (
    df.iloc[-1]["train_accuracy"]
)

final_val_accuracy = (
    df.iloc[-1]["val_accuracy"]
)

initial_val_loss = (
    df.iloc[0]["val_loss"]
)

final_val_loss = (
    df.iloc[-1]["val_loss"]
)


analysis_lines = []

analysis_lines.append(
    "ANALISIS BASELINE RESNET18"
)

analysis_lines.append(
    "=" * 60
)

analysis_lines.append("")

analysis_lines.append(
    f"Jumlah epoch: {len(df)}"
)

analysis_lines.append(
    f"Best epoch berdasarkan validation accuracy: "
    f"{int(best_epoch)}"
)

analysis_lines.append(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

analysis_lines.append(
    f"Validation loss terendah: "
    f"{best_val_loss:.4f}"
)

analysis_lines.append("")

analysis_lines.append(
    f"Train accuracy akhir: "
    f"{final_train_accuracy:.2f}%"
)

analysis_lines.append(
    f"Validation accuracy akhir: "
    f"{final_val_accuracy:.2f}%"
)

analysis_lines.append(
    f"Validation loss awal: "
    f"{initial_val_loss:.4f}"
)

analysis_lines.append(
    f"Validation loss akhir: "
    f"{final_val_loss:.4f}"
)

analysis_lines.append("")

analysis_lines.append(
    "Interpretasi:"
)

analysis_lines.append(
    "- Train accuracy meningkat hingga 100%."
)

analysis_lines.append(
    "- Validation accuracy tetap pada 90%."
)

analysis_lines.append(
    "- Validation loss meningkat setelah epoch pertama."
)

analysis_lines.append(
    "- Pola tersebut menunjukkan adanya indikasi "
    "model semakin menyesuaikan diri terhadap data training."
)

analysis_lines.append(
    "- Hasil test 100% harus dibaca secara khusus "
    "untuk test set yang digunakan."
)

analysis_lines.append(
    "- Hasil ini belum menjadi bukti bahwa model "
    "akan memperoleh performa yang sama pada data baru."
)

analysis_text = "\n".join(
    analysis_lines
)


# ============================================================
# 9. SIMPAN HASIL ANALISIS
# ============================================================

analysis_path = os.path.join(
    OUTPUT_DIR,
    "baseline_analysis.txt"
)

with open(
    analysis_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        analysis_text
    )


# ============================================================
# 10. TAMPILKAN HASIL
# ============================================================

print()
print("=" * 60)

print(
    "ANALISIS BASELINE RESNET18"
)

print("=" * 60)

print()

print(
    f"Best epoch: {int(best_epoch)}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Validation loss terendah: "
    f"{best_val_loss:.4f}"
)

print()

print(
    f"Train accuracy akhir: "
    f"{final_train_accuracy:.2f}%"
)

print(
    f"Validation accuracy akhir: "
    f"{final_val_accuracy:.2f}%"
)

print(
    f"Validation loss awal: "
    f"{initial_val_loss:.4f}"
)

print(
    f"Validation loss akhir: "
    f"{final_val_loss:.4f}"
)

print()

print(
    "File hasil disimpan di:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Accuracy curve:"
)

print(
    accuracy_path
)

print()

print(
    "Loss curve:"
)

print(
    loss_path
)

print()

print(
    "Training history:"
)

print(
    csv_path
)

print()

print(
    "Analysis:"
)

print(
    analysis_path
)

print()

print("=" * 60)

print(
    "SELESAI"
)

print("=" * 60)