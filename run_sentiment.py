# run_sentiment.py
import os
from sentiment.sentiment_inference import analyze_and_save
from sentiment.contextual_inference import adjust_sentiment_contextually
from sentiment.aggregation import aggregate_thread_sentiments
from sentiment.sentiment_inference import infer_single_sentence

def run():
    print("=== MODE ANALISIS SENTIMEN (UPDATED) ===")
    print("1️⃣ Analisis dataset CSV dari folder cleaning/dataset")
    print("2️⃣ Uji coba langsung di terminal (input manual)")
    mode = input("\nPilih mode [1/2]: ").strip()

    # =====================================================
    # MODE 2 — INTERAKTIF (uji coba satu kalimat)
    # =====================================================
    if mode == "2":
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

    # =====================================================
    # MODE 1 — PROSES BATCH CSV
    # =====================================================
    CLEAN_DIR = os.path.join("cleaning", "dataset")
    BASE_SENTIMENT_DIR = os.path.join("sentiment", "dataset")

    SENTIMENT_DIR = os.path.join(BASE_SENTIMENT_DIR, "sentiment")
    CONTEXTUAL_DIR = os.path.join(BASE_SENTIMENT_DIR, "contextual")
    SUMMARY_DIR = os.path.join(BASE_SENTIMENT_DIR, "summary")

    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    os.makedirs(CONTEXTUAL_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # --- Ambil file CSV dari cleaning/dataset ---
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
        if not (1 <= idx <= len(files)):
            print(f"[WARN] Nomor {idx} tidak valid.")
            continue

        base_name = files[idx-1].replace(".csv", "")
        inp = os.path.join(CLEAN_DIR, files[idx-1])

        print("\n====================================================")
        print(f"▶ MEMPROSES FILE: {files[idx-1]}")
        print("====================================================")

        # =====================================================
        # Tahap 1 — SENTIMENT INFERENCE + predicted_label
        # =====================================================
        out_sentiment = os.path.join(SENTIMENT_DIR, f"{base_name}_sentiment.csv")
        analyze_and_save(
            csv_path=inp,
            output_path=out_sentiment,
            batch_size=64,
            model_name="mdhugol/indonesia-bert-sentiment-classification"
        )
        print(f"[DONE] Sentiment inference tersimpan di: {out_sentiment}")

        # =====================================================
        # Tahap 2 — CONTEXTUAL ADJUSTMENT
        # =====================================================
        print(f"\n[STEP] Menjalankan contextual inference untuk: {base_name}")
        df_context = adjust_sentiment_contextually(out_sentiment)
        out_context = os.path.join(CONTEXTUAL_DIR, f"{base_name}_contextual.csv")
        df_context.to_csv(out_context, index=False, encoding="utf-8-sig")
        print(f"[DONE] Contextual file disimpan di: {out_context}")

        # =====================================================
        # Tahap 3 — THREAD-LEVEL AGGREGATION
        # =====================================================
        print(f"\n[STEP] Menjalankan agregasi thread-level untuk: {base_name}")
        df_summary = aggregate_thread_sentiments(out_context)
        out_summary = os.path.join(SUMMARY_DIR, f"{base_name}_summary.csv")
        df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        print(f"[DONE] Summary file disimpan di: {out_summary}")

        print("====================================================\n")

    print("\n🎉 Semua analisis selesai.")
    print(f"  📁 Sentiment  : {SENTIMENT_DIR}")
    print(f"  📁 Contextual : {CONTEXTUAL_DIR}")
    print(f"  📁 Summary    : {SUMMARY_DIR}")
    print("====================================================")

if __name__ == "__main__":
    run()
