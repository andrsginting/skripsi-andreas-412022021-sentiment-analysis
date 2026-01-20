from count_words_pipeline.file_loader import load_clean_files
from count_words_pipeline.summarizer import process_word_counts

def main():
    print("\n=== Menghitung jumlah kata per komentar ===")

    files = load_clean_files()

    print(f"Memuat {len(files)} file dari folder cleaning/dataset")

    process_word_counts(files)

    print("\n✔ Proses selesai!")
    print("Folder hasil: dataset_count_word/")
    print("Hasil disimpan dalam:")
    print("- count_words_<file>.csv")
    print("- summary_avg_words_per_file.csv")
    print("- summary_global_avg_words.csv\n")

if __name__ == "__main__":
    main()
