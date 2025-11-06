# run_sentiment.py
import os
from sentiment.sentiment_inference import analyze_and_save, infer_single_sentence

def run():
    print("=== MODE ANALISIS SENTIMEN ===")
    print("1️⃣ Analisis dataset CSV dari folder cleaning/dataset")
    print("2️⃣ Uji coba langsung di terminal (input manual)")
    mode = input("\nPilih mode [1/2]: ").strip()

    if mode == "2":
        # ---------------------------
        # Interactive Inference Mode
        # ---------------------------
        print("\n🧠 Mode uji coba interaktif aktif.")
        print("Ketik kalimat untuk dianalisis. Ketik 'exit' untuk keluar.\n")

        while True:
            text = input("🗨️  Masukkan kalimat: ").strip()
            if text.lower() in ["exit", "quit", "keluar"]:
                print("👋 Keluar dari mode interaktif.")
                break
            if not text:
                continue
            infer_single_sentence(text, model_name="mdhugol/indonesia-bert-sentiment-classification")
        return

    # ---------------------------
    # Batch CSV Mode
    # ---------------------------
    CLEAN_DIR = os.path.join("cleaning", "dataset")
    SENTIMENT_DIR = os.path.join("sentiment", "dataset")
    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
    if not files:
        print("[INFO] Tidak ada dataset bersih ditemukan di cleaning/dataset.")
        return

    print("\n📂 Daftar file hasil cleaning:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {f}")

    selected = input("\nPilih file untuk analisis (pisahkan dengan koma, contoh: 1,3,5): ").strip()
    selected_indices = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]

    for idx in selected_indices:
        if 1 <= idx <= len(files):
            inp = os.path.join(CLEAN_DIR, files[idx-1])
            outp = os.path.join(SENTIMENT_DIR, files[idx-1].replace(".csv", "_sentiment.csv"))
            analyze_and_save(inp, outp, model_name="mdhugol/indonesia-bert-sentiment-classification")
        else:
            print(f"[WARN] Nomor {idx} tidak valid.")

    print("\n✅ Semua analisis sentimen selesai. Hasil tersimpan di 'sentiment/dataset'.")

if __name__ == "__main__":
    run()
