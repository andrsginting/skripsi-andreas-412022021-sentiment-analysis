# run_cleaning.py
import os
from cleaning.cleaner import list_available_datasets, clean_dataset

def run():
    print("📁 Daftar dataset tersedia di folder 'scrapping/dataset':\n")
    datasets = list_available_datasets()

    if not datasets:
        print("[INFO] Tidak ada file CSV yang ditemukan di folder scrapping/dataset.")
        return

    for idx, f in enumerate(datasets, start=1):
        print(f"{idx}. {f}")

    choices = input("\nMasukkan nomor file yang ingin dibersihkan (pisahkan dengan koma, contoh: 1,3,4): ").strip()
    if not choices:
        print("Tidak ada input diberikan. Program dihentikan.")
        return

    selected_indices = []
    for c in choices.split(","):
        c = c.strip()
        if c.isdigit():
            selected_indices.append(int(c))

    print("\nMemulai proses cleaning...\n")
    for idx in selected_indices:
        if 1 <= idx <= len(datasets):
            file_name = datasets[idx - 1]
            try:
                clean_dataset(file_name)
            except Exception as e:
                print(f"[CRITICAL] Cleaning gagal total pada {file_name}: {e}")
        else:
            print(f"[WARN] Nomor {idx} tidak valid, dilewati.")

    print("\n✅ Semua proses cleaning selesai. Hasil tersimpan di folder 'cleaning/dataset'.")


if __name__ == "__main__":
    run()
