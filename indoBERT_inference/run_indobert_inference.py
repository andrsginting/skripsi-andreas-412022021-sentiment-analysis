# indoBERT_inference/run_indobert_inference.py
import os
from pathlib import Path
import pandas as pd
from sentiment.sentiment_inference import compute_sentiment_scores

CLEAN_DIR = Path("cleaning/dataset")
OUT_DIR = Path("indoBERT_inference/indoBERT_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted([p for p in CLEAN_DIR.iterdir() if p.suffix.lower() == ".csv"])
    if not files:
        print("[INFO] Tidak ada file CSV di cleaning/dataset")
        return

    for p in files:
        print(f"[PROCESS] Running IndoBERT inference for {p.name}")
        df = pd.read_csv(p)
        if "cleaned_comment" not in df.columns:
            raise ValueError(f"File {p} harus memiliki kolom 'cleaned_comment'")

        df_out = compute_sentiment_scores(df, batch_size=64, model_name="mdhugol/indonesia-bert-sentiment-classification")
        out_path = OUT_DIR / f"{p.stem}_scores.csv"
        # Simpan kolom penting
        save_cols = [c for c in ["thread_id", "cleaned_comment", "likes_count", "is_reply", "sentiment_score", "predicted_label"] if c in df_out.columns]
        df_out[save_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
