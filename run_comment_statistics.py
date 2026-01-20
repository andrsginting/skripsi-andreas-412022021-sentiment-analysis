import os
import pandas as pd

# =============================
# KONFIGURASI FOLDER
# =============================
SCRAPING_FOLDER = "scrapping/dataset"
OUTPUT_FOLDER = "statistics/comment_scraping_stats"

# =============================
# MEMBUAT FOLDER OUTPUT
# =============================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =============================
# FUNGSI PEMROSESAN STATISTIK
# =============================
def process_file_stats(filepath):

    # Baca CSV
    df = pd.read_csv(filepath)

    # Validasi kolom yang wajib
    required_cols = ["thread_id", "comment", "likes_count", "is_reply"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan pada file: {filepath}")

    # Statistik dasar
    total_comments = len(df)
    total_top_level = len(df[df["is_reply"] == False])
    total_replies = len(df[df["is_reply"] == True])
    total_likes = df["likes_count"].sum()
    avg_likes = df["likes_count"].mean()

    # Komentar dengan likes tertinggi
    max_likes = df["likes_count"].max()
    df_max = df[df["likes_count"] == max_likes].iloc[0]
    comment_with_max_likes = df_max["comment"]

    # Nama file detail
    detail_csv_name = os.path.basename(filepath).replace(".csv", "_details.csv")
    detail_output_path = os.path.join(OUTPUT_FOLDER, detail_csv_name)

    # Simpan versi detail per file
    df.to_csv(detail_output_path, index=False)

    return {
        "file": os.path.basename(filepath),
        "total_comments": total_comments,
        "total_top_level": total_top_level,
        "total_replies": total_replies,
        "total_likes": total_likes,
        "avg_likes_per_comment": round(avg_likes, 3),
        "max_likes": max_likes,
        "comment_with_max_likes": comment_with_max_likes,
        "detail_csv": detail_csv_name
    }

# =============================
# MAIN EXECUTION
# =============================
def main():
    print("\n=== Menghitung Statistik Komentar YouTube ===\n")

    files = [f for f in os.listdir(SCRAPING_FOLDER) if f.endswith(".csv")]
    if len(files) == 0:
        print("Tidak ada file CSV pada folder scrapping/dataset.")
        return

    all_stats = []

    for fname in files:
        filepath = os.path.join(SCRAPING_FOLDER, fname)
        print(f"Memproses file: {fname} ...")

        stats = process_file_stats(filepath)
        all_stats.append(stats)

    # Simpan summary keseluruhan
    summary_df = pd.DataFrame(all_stats)
    summary_output = os.path.join(OUTPUT_FOLDER, "comment_scraping_summary.csv")
    summary_df.to_csv(summary_output, index=False)

    print("\n=== SELESAI ===")
    print(f"Hasil ringkasan disimpan ke: {summary_output}")
    print(f"Folder detail: {OUTPUT_FOLDER}\n")


if __name__ == "__main__":
    main()
