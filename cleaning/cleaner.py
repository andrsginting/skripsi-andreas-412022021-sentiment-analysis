# cleaning/cleaner.py
import os
import pandas as pd
from tqdm import tqdm
from .text_utils import clean_comment_pipeline, detect_empty_reason

SCRAP_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "scrapping", "dataset")
CLEAN_DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(CLEAN_DATASET_DIR, exist_ok=True)

def list_available_datasets():
    """List semua file CSV di folder scrapping/dataset."""
    files = [f for f in os.listdir(SCRAP_DATASET_DIR) if f.endswith(".csv")]
    files.sort()
    return files


def clean_dataset(file_name: str):
    """Membersihkan satu file dataset dan menyimpannya ke folder cleaning/dataset."""
    input_path = os.path.join(SCRAP_DATASET_DIR, file_name)
    output_name = file_name.replace(".csv", "_cleaned.csv")
    output_path = os.path.join(CLEAN_DATASET_DIR, output_name)

    print(f"\n[PROCESS] Membersihkan file: {file_name}")

    # === Baca dataset mentah ===
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Gagal membaca file {file_name}: {e}")
        return

    # === Validasi kolom wajib ===
    if "comment" not in df.columns or "likes_count" not in df.columns:
        print(f"[ERROR] Kolom 'comment' atau 'likes_count' tidak ditemukan pada {file_name}.")
        return

    # === Siapkan kolom opsional agar tidak error ===
    if "thread_id" not in df.columns:
        df["thread_id"] = None
    if "is_reply" not in df.columns:
        df["is_reply"] = False

    success_count, fail_count = 0, 0
    cleaned_comments = []

    # === Lakukan cleaning komentar ===
    for i, text in tqdm(enumerate(df["comment"].astype(str)), total=len(df), desc=f"Cleaning {file_name}", ncols=90):
        try:
            cleaned = clean_comment_pipeline(text)
            cleaned_comments.append(cleaned)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Gagal membersihkan baris {i}: {e}")
            cleaned_comments.append(text)
            fail_count += 1

    df["cleaned_comment"] = cleaned_comments

    # === Deteksi komentar kosong ===
    df["empty_reason"] = df.apply(
        lambda x: detect_empty_reason(x["comment"]) if not str(x["cleaned_comment"]).strip() else None,
        axis=1
    )

    # === Hitung statistik kosong ===
    empty_rows = df[df["cleaned_comment"].astype(str).str.strip() == ""]
    count_emoji = (empty_rows["empty_reason"] == "emoji_saja").sum()
    count_punct = (empty_rows["empty_reason"] == "tanda_baca_saja").sum()
    count_other = (empty_rows["empty_reason"] == "lainnya").sum()
    total_deleted = len(empty_rows)

    # === Hapus baris kosong ===
    df = df[df["cleaned_comment"].astype(str).str.strip() != ""]

    # === Simpan hasil akhir dengan kolom penting ===
    cleaned_df = pd.DataFrame({
        "thread_id": df["thread_id"],
        "cleaned_comment": df["cleaned_comment"],
        "likes_count": df["likes_count"],
        "is_reply": df["is_reply"]
    })

    try:
        cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=1)
        print(f"[DONE] File selesai dibersihkan: {output_path}")
        print(f"[SUMMARY] Total baris: {len(df)} | Berhasil diproses: {success_count} | Gagal: {fail_count}")

        print(f"[FILTER] Baris kosong dihapus: {total_deleted}")
        if total_deleted > 0:
            print(f" ├─ Hanya emoji: {count_emoji}")
            print(f" ├─ Hanya tanda baca: {count_punct}")
            print(f" └─ Lainnya: {count_other}")
        else:
            print("[FILTER] Tidak ada baris kosong yang dihapus.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file {output_name}: {e}")
