import pandas as pd
import os
from .word_counter import count_words

def process_word_counts(files, output_folder="dataset_count_word"):
    """
    Menghasilkan:
    - file jumlah kata per komentar
    - rata-rata kata per file
    - global average antar semua file
    """
    os.makedirs(output_folder, exist_ok=True)

    summary = []

    total_words_all = 0
    total_comments_all = 0

    for fname, df in files:
        df["word_count"] = df["cleaned_comment"].apply(count_words)

        # simpan hasil untuk tiap file
        df.to_csv(f"{output_folder}/count_words_{fname}.csv", index=False)

        # hitung rata-rata
        avg_words = df["word_count"].mean()

        summary.append({
            "file": fname,
            "total_comments": len(df),
            "total_words": df["word_count"].sum(),
            "avg_words_per_comment": avg_words
        })

        total_comments_all += len(df)
        total_words_all += df["word_count"].sum()

    # simpan summary per file
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_folder}/summary_avg_words_per_file.csv", index=False)

    # global average
    global_avg = total_words_all / total_comments_all if total_comments_all > 0 else 0
    pd.DataFrame([{"global_avg_words_per_comment": global_avg}]).to_csv(
        f"{output_folder}/summary_global_avg_words.csv",
        index=False
    )
