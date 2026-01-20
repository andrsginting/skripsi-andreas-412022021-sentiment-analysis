# sentiment/runners/run_sentiment_inference.py
import os
from sentiment.sentiment_inference import analyze_and_save

CLEAN_DIR = os.path.join("cleaning", "dataset")
OUTPUT_DIR = os.path.join("sentiment", "dataset", "sentiment")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
    if not files:
        print("[INFO] Tidak ada file CSV di cleaning/dataset")
        return

    print("\n📂 Dataset tersedia:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")

    choice = input("\nPilih file (contoh: 1,3,5): ").strip()
    idxs = [int(x)-1 for x in choice.split(",") if x.strip().isdigit()]

    for i in idxs:
        if i < 0 or i >= len(files):
            continue

        fname = files[i]
        base = fname.replace(".csv", "")
        inp = os.path.join(CLEAN_DIR, fname)
        out = os.path.join(OUTPUT_DIR, f"{base}_sentiment.csv")

        print(f"\n▶ Sentiment inference: {fname}")
        analyze_and_save(
            csv_path=inp,
            output_path=out,
            batch_size=64,
            model_name="mdhugol/indonesia-bert-sentiment-classification"
        )

        print(f"✓ Saved: {out}")

    print("\n🎉 Sentiment inference selesai.")

if __name__ == "__main__":
    main()
