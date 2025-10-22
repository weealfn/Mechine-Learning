# Pertemuan 14 - Data Preparation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Baca dataset mentah
# -----------------------------
df = pd.read_csv("D:\SEMESTER 5\Machine Learning\projekku\kelulusan_mahasiswa.csv")
print("Dataset awal:", df.shape)
print(df.head())

# -----------------------------
# 2. Cleaning
# -----------------------------
# Cek missing value
print("\nMissing value:")
print(df.isnull().sum())

# Hapus duplikat (jika ada)
df = df.drop_duplicates()

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
print("\nStatistik deskriptif:")
print(df.describe())

# Boxplot IPK
sns.boxplot(x=df["IPK"])
plt.title("Boxplot IPK")
plt.savefig("boxplot_ipk.png", dpi=120)
plt.close()

# Histogram IPK
sns.histplot(df["IPK"], bins=10, kde=True)
plt.title("Histogram IPK")
plt.savefig("hist_ipk.png", dpi=120)
plt.close()

# Scatterplot
sns.scatterplot(x="IPK", y="Waktu_Belajar_Jam", data=df, hue="Lulus")
plt.title("Scatterplot IPK vs Waktu Belajar")
plt.savefig("scatter_ipk_vs_study.png", dpi=120)
plt.close()

# Heatmap korelasi
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.savefig("heatmap_corr.png", dpi=120)
plt.close()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
df["Rasio_Absensi"] = df["Jumlah_Absensi"] / 14
df["IPK_x_Study"] = df["IPK"] * df["Waktu_Belajar_Jam"]

# -----------------------------
# 5. Simpan hasil bersih
# -----------------------------
df.to_csv("processed_kelulusan.csv", index=False)
print("\nFile processed_kelulusan.csv berhasil dibuat!")
